"""§12.2 可用性与降级，端到端（AssistService → 适配器 → 引擎）：AI 出任何问题，客服都能照常工作，
只是暂时没有推荐；服务本身永远不抛错。

| 故障 | 降级方式 |
|---|---|
| 大模型超时或报错 | 退回 FAQ 和模板推荐；再不行就不推荐 |
| 判断层不可用 | 用关键词规则路由；高风险类型一律不推荐 |
| 平台接口限流或故障 | 用缓存数据并注明时间；查不到就推荐"我帮您查一下" |
| 知识库故障 | 不推荐 |
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from e2e_support import (
    Chat,
    demo_service,
    enable_auto,
    failing_llm,
    route,
    scripted_llm,
    scripted_qa,
    scripted_router,
)

from rodit.core.enums import DraftKind, Intent, Mode, RouteAction
from rodit.core.errors import DecisionError, KnowledgeUnavailable, LLMError
from rodit.engine.engine import REASON_KNOWLEDGE
from rodit.qa.gate import QA_UNAVAILABLE
from rodit.testing.fakes import FakeDecisionProvider
from rodit.testing.fixtures import SHOP_A

LOGISTICS_TOOLS = {"get_order", "get_logistics", "list_recent_orders", "list_aftersales"}


# ================================================================ 大模型超时或报错


def test_llm_down_everywhere_falls_back_to_faq_and_templates():
    """整个大模型不可用（生产组装：路由、质检、生成共用一个大模型）：
    路由退回关键词规则，FAQ / 模板照常推荐，需要大模型写的就不推荐，高风险一律不推荐。"""
    llm = failing_llm(LLMError("大模型超时（模拟）"))
    service = demo_service(llm=llm)  # 不注入判断层：按默认配置 incloud_llm 组装
    chat = Chat(service, "c_llm_down")

    faq = chat.say("什么时候发货")
    assert faq.route.degraded and faq.route.provider == "rules"
    assert "decision" in faq.degraded
    [card] = faq.suggestions
    assert card.kind == DraftKind.FAQ and "48 小时" in card.text
    assert any("判断层暂不可用" in h for h in card.risk_hints)

    logistics = chat.say("我的快递到哪了", order_id="o_1002")
    [card] = logistics.suggestions
    assert card.kind == DraftKind.TEMPLATE and "截至" in card.text

    size = chat.say("165 110斤穿什么码", item_id="i_555")
    assert size.route.intent == Intent.SIZE_ADVICE
    assert size.suggestions == [] and size.no_suggestion_reason

    risky = chat.say("显示签收了但我根本没收到", order_id="o_1003")
    assert risky.route.intent == Intent.NOT_RECEIVED
    assert risky.route.action == RouteAction.NO_SUGGESTION and risky.route.rule == 4
    assert risky.suggestions == []

    assert llm.calls_for("generate")  # 确实试过调大模型生成
    assert all(
        c.kind != DraftKind.LLM for r in (faq, logistics, size, risky) for c in r.suggestions
    )


def test_generation_down_with_judge_up_falls_back_to_logistics_template():
    """判断层正常、只有生成的大模型报错：签收未收到退回物流模板（带取件指引），尺码问题不推荐。"""
    llm = scripted_llm(
        {"没收到": LLMError("生成超时（模拟）"), "穿什么码": LLMError("生成超时（模拟）")}
    )
    service = demo_service(
        llm=llm,
        router=scripted_router(
            [
                ("没收到", route(Intent.NOT_RECEIVED, 0.93)),
                ("穿什么码", route(Intent.SIZE_ADVICE, 0.92)),
            ]
        ),
        qa=scripted_qa(),
    )
    chat = Chat(service, "c_gen_down")
    risky = chat.say("显示签收了但我根本没收到", order_id="o_1003")
    assert risky.suggestions and all(c.kind == DraftKind.TEMPLATE for c in risky.suggestions)
    by_evidence = {e.id: c for c in risky.suggestions for e in c.evidence}
    card = by_evidence["tool:logistics:o_1003"]  # 驿站信息和取件指引（§7.3.1）
    assert "菜鸟驿站" in card.text and "截至" in card.text
    assert "请核查物流签收细分状态" in card.risk_hints
    assert "申请售后" in by_evidence["platform:taobao:apply_aftersale"].text  # 没找到就引导申请售后

    size = chat.say("165 110斤穿什么码", item_id="i_555")
    assert size.suggestions == [] and size.no_suggestion_reason


# ================================================================ 判断层不可用


def test_judge_down_uses_keyword_routing_and_never_recommends_high_risk():
    down = DecisionError("判断层不可用（模拟）")
    service = demo_service(
        router=FakeDecisionProvider({}, fail=down),
        qa=FakeDecisionProvider({}, fail=down),
    )
    chat = Chat(service, "c_judge_down")

    ship = chat.say("什么时候发货")
    assert ship.route.degraded and ship.route.provider == "rules"
    assert ship.route.intent == Intent.SHIPPING_POLICY
    [card] = ship.suggestions  # FAQ 只跑硬规则，照常推荐
    assert card.kind == DraftKind.FAQ
    assert any("判断层暂不可用" in h for h in card.risk_hints)

    promo = chat.say("能便宜点吗", item_id="i_555")
    assert promo.route.intent == Intent.PRICE_PROMO
    assert promo.route.action == RouteAction.NO_SUGGESTION and promo.route.rule == 4
    assert promo.suggestions == []

    # 大模型写的回答没有判断层质检，不推荐（§12.2 质检不可用时不推荐未经质检的生成内容）
    size = chat.say("165 110斤穿什么码", item_id="i_555")
    assert size.route.intent == Intent.SIZE_ADVICE and size.route.action == RouteAction.ANSWER
    assert size.suggestions == []
    assert QA_UNAVAILABLE in (size.no_suggestion_reason or "")
    assert "decision:qa" in size.degraded


# ================================================================ 全自动模式下任何故障


def test_auto_mode_failure_switches_back_to_assist():
    """§12.2 全自动模式下任何故障 → 立即切回智能辅助，由客服接管：
    不自动发送、转人工，推荐照常给客服。"""
    service = demo_service(router=FakeDecisionProvider({}, fail=DecisionError("判断层不可用")))
    adapter = enable_auto(service)
    result = Chat(service, "c_auto_down", access="A", robot=True).say("在吗")
    assert result.mode == Mode.AUTO and result.route.degraded
    assert result.auto_reply is None and result.transfer_to_human
    assert result.suggestions  # 问候模板照常推荐给接手的客服
    assert adapter.sent_messages(SHOP_A) == []
    assert len(adapter.transfers(SHOP_A)) == 1


# ================================================================ 平台接口限流或故障


def test_platform_tool_down_uses_cached_data_with_as_of_time():
    service = demo_service(router=scripted_router([("快递", route(Intent.LOGISTICS_TRACK, 0.95))]))
    unit = service.unit
    chat = Chat(service, "c_tool_cache")
    fresh = chat.say("我的快递到哪了", order_id="o_1002")
    [card] = fresh.suggestions
    assert "截至" in card.text and "10:30" in card.text

    # 平台限流，缓存也已过期（TTL 300 秒）：仍用缓存，注明原来的时间
    unit.tools.fail_on |= LOGISTICS_TOOLS
    unit.clock.set(unit.clock.now() + timedelta(minutes=10))
    stale = chat.say("快递到哪了呀", order_id="o_1002")
    [card] = stale.suggestions
    assert card.kind == DraftKind.TEMPLATE
    assert "截至" in card.text and "10:30" in card.text
    stats = unit.gateway.stats(SHOP_A)[SHOP_A]
    assert stats["get_logistics"].stale_served >= 1


def test_platform_tool_down_without_cache_recommends_checking():
    service = demo_service(router=scripted_router([("快递", route(Intent.LOGISTICS_TRACK, 0.95))]))
    service.unit.tools.fail_on |= LOGISTICS_TOOLS
    result = Chat(service, "c_tool_down").say("我的快递到哪了", order_id="o_1002")
    [card] = result.suggestions
    assert card.kind == DraftKind.TEMPLATE and "我帮您查一下" in card.text
    assert any(code.startswith("tool:") for code in result.degraded)
    assert any("实时数据暂时查不到" in h for h in card.risk_hints)


# ================================================================ 知识库故障


KNOWLEDGE_READS = (
    "get_policies",
    "get_item_card",
    "list_item_cards",
    "get_knowledge",
    "list_knowledge",
    "search_knowledge",
)


def _down(*args, **kwargs):
    # 与 PostgreSQL 仓储一致：连不上、语句超时抛 KnowledgeUnavailable（rodit.storage）
    raise KnowledgeUnavailable("知识库暂时不可用（模拟）")


def _broken(*args, **kwargs):
    raise RuntimeError("仓储内部错误（模拟）")


@pytest.mark.parametrize("error", [_down, _broken], ids=["unavailable", "unexpected"])
@pytest.mark.parametrize(
    ("text", "intent", "item_id"),
    [
        ("什么时候发货", Intent.SHIPPING_POLICY, None),
        ("这个是纯棉的吗", Intent.PRODUCT_SPEC, "i_555"),
        ("165 110斤穿什么码", Intent.SIZE_ADVICE, "i_555"),
    ],
)
def test_knowledge_repo_failure_gives_no_suggestion(monkeypatch, error, text, intent, item_id):
    service = demo_service(router=scripted_router([(text, route(intent, 0.92))]))
    for method in KNOWLEDGE_READS:
        monkeypatch.setattr(service.unit.repos.knowledge, method, error)
    result = Chat(service, "c_kb").say(text, item_id=item_id)
    assert result.suggestions == []
    assert result.no_suggestion_reason
    assert {"knowledge", "understanding"} & set(result.degraded)


def test_knowledge_repo_down_before_routing_gives_no_suggestion(monkeypatch):
    """连店铺都读不到（知识库整个挂了）：不推荐，也不抛错。"""
    service = demo_service()
    monkeypatch.setattr(service.unit.repos.knowledge, "get_shop", _down)
    result = Chat(service, "c_kb_shop").say("在吗")
    assert result.suggestions == [] and result.route is None
    assert result.degraded == ["knowledge"]
    assert result.no_suggestion_reason == REASON_KNOWLEDGE
