"""端到端验证中发现的缺陷的回归测试（每条对应一处最小修复，见各测试的说明）。"""

from __future__ import annotations

from e2e_support import (
    Chat,
    current_message,
    demo_service,
    enable_auto,
    generate_calls,
    llm_answer,
    prompt_text,
    route,
    scripted_llm,
    scripted_qa,
    scripted_router,
)

from rodit.core.audit import AuditAction
from rodit.core.enums import DraftKind, Intent, RouteAction
from rodit.testing.fakes import router_answers
from rodit.testing.fixtures import BUYER_REFUND, SHOP_A


def test_refund_progress_reports_latest_aftersale_of_the_order():
    """同一订单有两张售后单（as_2006 早先已撤销关闭、as_2008 退款中）：买家问"退款什么时候到"，
    应该回答正在处理的那张（最近更新的），而不是早就关闭的那张（§7.3.1 查售后单状态说明当前进度）。

    修复前 rodit.generation.evidence_view.refund_progress 在当前订单有多张售后单时取第一张
    （按创建时间最早），回答"已关闭，如仍需售后可以重新申请"，与事实相反。"""
    service = demo_service(router=scripted_router([("退款", route(Intent.REFUND_PROGRESS, 0.93))]))
    world_aftersales = [a for a in service.unit.tools.world.aftersales if a.order_id == "o_1012"]
    assert {a.aftersale_id for a in world_aftersales} == {"as_2006", "as_2008"}
    assert all(a.buyer_ref == BUYER_REFUND for a in world_aftersales)

    result = Chat(service, "c_refund", buyer="BUYER_REFUND").say(
        "退款什么时候到", order_id="o_1012"
    )
    [card] = result.suggestions
    assert card.kind == DraftKind.TEMPLATE
    assert "tool:aftersale:as_2008" in {e.id for e in card.evidence}
    assert "退款中" in card.text and "已关闭" not in card.text
    assert result.shop_id == SHOP_A


def test_auto_reply_carries_ai_prefix_exactly_once_even_if_model_copies_it():
    """全自动发出去的消息开头恰好一个 AI 生成内容标识（§12.3）。

    会话历史里上一条自动回复带着"【智能客服】"，提示词的【对话】里模型能看到它；模型照着写、
    回答也以"【智能客服】"开头时，修复前引擎再加一次，发给买家的是"【智能客服】【智能客服】…"。
    修复：rodit.engine.engine 加前缀前先去掉回答开头已有的前缀。"""
    prefix = "【智能客服】"
    copied = llm_answer(prefix + "亲，这款面料是棉 95%、氨纶 5% 哦~", "item:i_555:attr:面料")
    service = demo_service(
        llm=scripted_llm({"纯棉": copied}),
        router=scripted_router(
            [("在吗", route(Intent.GREETING, 0.98)), ("纯棉", route(Intent.PRODUCT_SPEC, 0.98))]
        ),
        qa=scripted_qa(),
    )
    adapter = enable_auto(service)
    chat = Chat(service, "c_prefix", access="A", robot=True)
    chat.say("在吗")
    result = chat.say("这个是纯棉的吗", item_id="i_555")
    assert result.auto_reply is not None
    sent = [m.text for m in adapter.sent_messages(SHOP_A)]
    assert len(sent) == 2
    for text in sent:
        assert text.startswith(prefix) and text.count(prefix) == 1, text
    assert sent[1] == prefix + "亲，这款面料是棉 95%、氨纶 5% 哦~"


STALE = "24 小时内发货"  # 商品详情页的旧说法；店铺政策是 48 小时（§5.2.1 以店铺政策为准）


def test_stale_detail_sentence_is_overridden_even_when_policy_is_not_needed():
    """商品参数类问题不需要店铺政策，但详情页逐句都会进依据：详情页里和店铺政策冲突的旧说法
    （"现货付款后 24 小时内发货"）也要被政策覆盖，不能给大模型看（§5.2.1）。

    多问题消息里两条分别检索："商品参数"那条的依据里有 24 小时、"发货"那条是 48 小时，
    同一条消息的两张推荐可能互相矛盾（质检的数字一致性也拦不住：24 在依据里）。"""

    def answer(request):
        # 只看这一条的【要求】：多问题时每条都写明"本条只回答关于…的部分"
        if "关于发货与快递的部分" in prompt_text(request):
            return llm_answer("亲，现货付款后 48 小时内发货哦~", "policy:ship_time")
        return llm_answer("亲，这款面料是棉 95%、氨纶 5% 哦~", "item:i_555:attr:面料")

    llm = scripted_llm({"面料": answer})
    multi = router_answers(
        Intent.PRODUCT_SPEC,
        0.55,
        probabilities={"product_spec": 0.55, "shipping_policy": 0.4, "other": 0.05},
        multi_intent=0.9,
    )
    service = demo_service(llm=llm, router=scripted_router([("面料", multi)]), qa=scripted_qa())
    result = Chat(service, "c_stale").say("这条裙子什么面料？多久能发货？", item_id="i_555")

    assert result.route.action == RouteAction.SPLIT_MULTI
    assert {c.intent for c in result.suggestions} == {Intent.PRODUCT_SPEC, Intent.SHIPPING_POLICY}
    calls = generate_calls(llm)
    assert calls and all("面料" in current_message(c) for c in calls)
    for call in calls:
        assert STALE not in prompt_text(call)
    retrieves = service.unit.repos.audit.list_events(SHOP_A, action=AuditAction.RETRIEVE)
    spec = next(e for e in retrieves if e.detail["intent"] == Intent.PRODUCT_SPEC)
    assert "item:i_555:detail:1" in spec.detail["overridden_ids"]
    assert "policy:ship_time" not in spec.detail["evidence_ids"]  # 政策只当裁判，不进依据
    assert "item:i_555:detail:0" in spec.detail["evidence_ids"]  # 不冲突的句子照常保留
