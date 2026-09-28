"""端到端回归：实时工具证据与售后相关的缺陷（集成审查 B1，每条对应一处最小修复）。

都走 build_demo_unit + AssistService（接入层 → 引擎 → 检索 + 工具证据 → 生成 → 质检），
平台数据全部是合成的演示世界。
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from e2e_support import (
    Chat,
    demo_service,
    llm_answer,
    prompt_text,
    route,
    scripted_llm,
    scripted_qa,
    scripted_router,
    stored_texts,
)

from rodit.core.audit import AuditAction
from rodit.core.enums import (
    AfterSaleStatus,
    DraftKind,
    Intent,
    KnowledgeSource,
    KnowledgeStatus,
)
from rodit.core.models import KnowledgeItem, TraceEvent
from rodit.testing.fixtures import SHOP_A, build_demo_world


def _retrieves(service) -> list[dict[str, Any]]:
    events = service.unit.repos.audit.list_events(SHOP_A, action=AuditAction.RETRIEVE)
    return [e.detail for e in events]


# ================================================================ 退换资格 vs FAQ


RETURN_FAQ = "亲，支持七天无理由退货哦，收到不满意可以直接申请~"


def _world_with_return_faq():
    world = build_demo_world()
    world.faqs.append(
        KnowledgeItem(
            shop_id=SHOP_A,
            question="能退货吗",
            answer=RETURN_FAQ,
            source=KnowledgeSource.MERCHANT,
            status=KnowledgeStatus.APPROVED,
            intent=Intent.RETURN_POLICY,
            approved_by="merchant",
            created_at=world.now - timedelta(days=7),
        )
    )
    return world


def test_realtime_return_eligibility_overrides_contradicting_faq():
    """订单实时退换资格（优先级 1）说不能无理由退，FAQ 说"支持七天无理由退货"：
    FAQ 要被覆盖，不能当标准回答直接推荐（§5.2.1、§7.3.1）。

    修复前 tool:return 证据没有 topic，冲突处理不拿它和 FAQ 比，FAQ 相似度 1.0 直接推荐。"""
    service = demo_service(
        world=_world_with_return_faq(),
        router=scripted_router([("退货", route(Intent.RETURN_POLICY, 0.95))]),
    )
    # o_1007 定制款不支持七天无理由；o_1005 签收 23 天，超过期限
    for buyer, order_id in (("BUYER_R8", "o_1007"), ("BUYER_R5", "o_1005")):
        result = Chat(service, f"c_ret_{order_id}", buyer=buyer).say("能退货吗", order_id=order_id)
        assert all(c.kind != DraftKind.FAQ for c in result.suggestions), order_id
        assert all("支持七天无理由退货" not in c.text for c in result.suggestions), order_id
        detail = _retrieves(service)[-1]
        assert f"tool:return:{order_id}" in detail["evidence_ids"]
        assert any(i.startswith("faq:") for i in detail["overridden_ids"]), detail

    # 可以退的订单（o_1004 签收 5 天）：FAQ 和实时资格一致，照常直接用标准回答（§5.4）
    ok = Chat(service, "c_ret_o_1004", buyer="BUYER_R3").say("能退货吗", order_id="o_1004")
    assert ok.suggestions and ok.suggestions[0].kind == DraftKind.FAQ
    assert ok.suggestions[0].text == RETURN_FAQ
    assert not any(i.startswith("faq:") for i in _retrieves(service)[-1]["overridden_ids"])


# ================================================================ 实时活动 vs 店铺满减政策


def test_realtime_activity_config_overrides_promotions_policy():
    """i_888 定制款不参加店铺满减（实时活动配置：当前没有活动）：店铺政策"满 199 减 20"
    要被实时数据覆盖，不能推荐给买家（§5.2.1 实时接口优先级 1）。

    修复前价格证据的 topic 是 price:{item}，和政策主题 promotions 永远对不上。"""
    service = demo_service(router=scripted_router([("优惠", route(Intent.PRICE_PROMO, 0.95))]))
    result = Chat(service, "c_promo_888").say("这件有优惠吗，能用满减吗", item_id="i_888")
    assert all("199" not in c.text for c in result.suggestions)
    detail = _retrieves(service)[-1]
    assert "tool:price:i_888" in detail["evidence_ids"]
    assert "policy:promotions" in detail["overridden_ids"]
    assert "policy:promotions" not in detail["evidence_ids"]

    # i_555 参加店铺满减：实时活动和政策一致，两条都保留
    Chat(service, "c_promo_555").say("这件有优惠吗，能用满减吗", item_id="i_555")
    detail = _retrieves(service)[-1]
    assert {"tool:price:i_555", "policy:promotions"} <= set(detail["evidence_ids"])
    assert "policy:promotions" not in detail["overridden_ids"]


# ================================================================ 平台故障时用缓存：注明真实时间


def test_stale_aftersale_cache_carries_real_fetch_time_and_warns_agent():
    """平台接口故障时用缓存数据要注明数据的获取时间（§12.2 用缓存数据并注明时间），
    并提醒客服核实——不能把半小时前的"退款中"说成"截至现在"。

    修复前订单、售后单没有获取时间，回答写"截至 <现在>"，EngineResult.degraded 也是空的。"""
    service = demo_service(router=scripted_router([("退款", route(Intent.REFUND_PROGRESS, 0.93))]))
    unit = service.unit
    chat = Chat(service, "c_refund_stale", buyer="BUYER_REFUND")
    fresh = chat.say("退款什么时候到账")
    [card] = fresh.suggestions
    fetched = unit.clock.now()
    assert "退款中" in card.text and f"截至 {fetched:%H:%M}" in card.text
    assert fresh.degraded == []

    # 平台上已退款成功；半小时后平台接口故障，缓存已过 TTL（300 秒）
    unit.tools.set_aftersale_status(
        SHOP_A, "as_2008", AfterSaleStatus.REFUND_SUCCESS, at=fetched + timedelta(minutes=5)
    )
    unit.clock.set(fetched + timedelta(minutes=30))
    unit.tools.fail_on |= {"list_aftersales", "list_recent_orders", "get_order"}
    stale = chat.say("退款到账了吗")
    [card] = stale.suggestions
    now = unit.clock.now()
    assert f"截至 {fetched:%H:%M}" in card.text, card.text
    assert f"{now:%H:%M}" not in card.text
    assert "tool:aftersale" in stale.degraded
    assert any("实时数据暂时查不到" in h for h in card.risk_hints)
    stats = unit.gateway.stats(SHOP_A)[SHOP_A]
    assert stats["list_aftersales"].stale_served >= 1


def test_order_card_cache_carries_real_fetch_time_when_platform_is_down():
    """带订单卡片问退款进度（o_1001 / as_2001）：订单和售后单都来自过期缓存时，
    回答写缓存的时间，订单、售后单都标降级（卡片提示客服核实）。"""
    service = demo_service(router=scripted_router([("退款", route(Intent.REFUND_PROGRESS, 0.93))]))
    unit = service.unit
    chat = Chat(service, "c_refund_card")
    chat.say("退款什么时候到账", order_id="o_1001")
    fetched = unit.clock.now()
    unit.clock.set(fetched + timedelta(minutes=40))
    unit.tools.fail_on |= {"list_aftersales", "list_recent_orders", "get_order"}
    stale = chat.say("退款什么时候到账呀", order_id="o_1001")
    [card] = stale.suggestions
    assert f"截至 {fetched:%H:%M}" in card.text, card.text
    assert {"tool:order", "tool:aftersale"} <= set(stale.degraded)


def test_cache_older_than_max_stale_is_not_served():
    """过期太久的缓存不再兜底（build_unit 设了 max_stale_s）：不拿 3 小时前的"退款中"回答，
    按查不到处理（标降级，卡片提示客服核实）。"""
    service = demo_service(router=scripted_router([("退款", route(Intent.REFUND_PROGRESS, 0.93))]))
    unit = service.unit
    chat = Chat(service, "c_refund_old", buyer="BUYER_REFUND")
    chat.say("退款什么时候到账")
    unit.clock.set(unit.clock.now() + timedelta(hours=3))
    unit.tools.fail_on |= {"list_aftersales", "list_recent_orders", "get_order"}
    result = chat.say("退款到账了吗")
    assert all("退款中" not in c.text for c in result.suggestions)
    assert "tool:order" in result.degraded
    assert all(any("实时数据暂时查不到" in h for h in c.risk_hints) for c in result.suggestions)
    stats = unit.gateway.stats(SHOP_A)[SHOP_A]
    assert stats["list_recent_orders"].stale_served == 0


# ================================================================ 库存：一个尺码对应多个 SKU


def test_stock_for_size_matching_several_skus_is_not_item_aggregate():
    """买家问"M 码还有货吗"：M 码对应黑色 M、白色 M 两个 SKU，都卖完了，只有黑色 L 有货。
    依据不能是整款合计的"有货，库存 3 件"（会被当成 M 码有货），要写清哪些规格缺货。"""
    world = build_demo_world()
    for i, s in enumerate(world.stocks):
        if s.item_id == "i_555":
            qty = {"k_1": 0, "k_2": 3, "k_3": 0, "k_4": 0}[s.sku_id or ""]
            world.stocks[i] = s.model_copy(update={"quantity": qty, "in_stock": qty > 0})

    seen: list[str] = []

    def answer(request):
        prompt = prompt_text(request)
        seen.append(prompt)
        line = next(ln for ln in prompt.splitlines() if "[tool:stock:i_555" in ln)
        eid = re.search(r"\[(tool:stock:[^\]]+)\]", line).group(1)
        return llm_answer("亲，M 码目前暂时缺货哦，黑色 L 码有货~", eid)

    llm = scripted_llm({"M": answer})
    service = demo_service(
        world=world,
        llm=llm,
        router=scripted_router([("有货", route(Intent.STOCK, 0.95))]),
        qa=scripted_qa(),
    )
    result = Chat(service, "c_stock_m").say("M 码还有货吗", item_id="i_555")
    assert seen, "没有调用生成"
    stock_lines = [ln for ln in seen[0].splitlines() if "tool:stock:i_555" in ln]
    assert stock_lines
    text = "\n".join(stock_lines)
    assert "有货，库存 3 件" not in text  # 不能只给整款合计
    assert "缺货" in text
    for spec in ("黑色", "白色"):
        assert re.search(rf"{spec}[^；]*M", text), text
    assert result.suggestions


# ================================================================ 退货问题找订单要看当前商品


def test_return_question_about_viewed_item_does_not_borrow_other_order():
    """买家在看 i_888（定制衬衫，没买过）问能不能七天无理由退：不能拿他另一单牛仔裤
    （o_1010）的退换资格当依据（§7.3.1 能不能退 = 店铺政策 + 这件商品的订单状态）。"""
    service = demo_service(router=scripted_router([("退货", route(Intent.RETURN_POLICY, 0.95))]))
    Chat(service, "c_ret_item").say("这件定制的衬衫支持七天无理由退货吗", item_id="i_888")
    detail = _retrieves(service)[-1]
    assert not any(i.startswith(("tool:return:", "tool:order:")) for i in detail["evidence_ids"])
    assert "policy:non_returnable" in detail["evidence_ids"]


# ================================================================ 取件码的常见写法


def test_pickup_code_formats_never_reach_answer_prompt_or_records():
    """快递柜 / 驿站轨迹里取件码的各种写法都不能出现在回答、提示词和记录里（§7.2）。"""
    codes = {
        "您的快递已放入丰巢柜，取件码【60125477】": "60125477",
        "请凭 8-2-3021 到幸福里驿站取件": "8-2-3021",
        "已签收，取件码 A12 3456": "3456",
    }
    for desc, code in codes.items():
        world = build_demo_world()
        trace = next(t for t in world.traces if t.order_id == "o_1010")
        last = trace.events[-1]
        events = [*trace.events[:-1], TraceEvent(**{**last.model_dump(), "description": desc})]
        world.traces[world.traces.index(trace)] = trace.model_copy(update={"events": events})
        service = demo_service(
            world=world, router=scripted_router([("快递", route(Intent.LOGISTICS_TRACK, 0.95))])
        )
        result = Chat(service, "c_pickup").say("我的快递到哪了", order_id="o_1010")
        assert result.suggestions
        for card in result.suggestions:
            assert code not in card.text, (desc, card.text)
        for source, text in stored_texts(service.unit, SHOP_A).items():
            assert code not in text, (desc, source)
