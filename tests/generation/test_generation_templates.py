"""确定性模板（方案 §5.4、§5.7、§7.2、§7.3.1、§11.2–§11.4、§12.2）。"""

from __future__ import annotations

from datetime import timedelta

from rodit.core.enums import DraftKind, Intent, PolicyKey, RouteAction
from rodit.core.evidence import PlatformRuleKey
from rodit.core.models import EvidenceBundle, LogisticsAnalysis
from rodit.generation import templates as T
from rodit.testing.fakes import make_context
from rodit.testing.fixtures import SHOP_B

SCRIPT_11_4 = (
    "亲，平台规定不能通过站外联系和交易哦，您直接在店铺下单就好，我帮您看看有没有可以用的优惠券。"
)


def test_off_platform_script_is_exact_standard_script(world):
    ctx = make_context("加个微信吧，转账给你便宜点", world)
    draft = T.off_platform_script(ctx, EvidenceBundle())
    assert draft.text == SCRIPT_11_4
    assert draft.kind == DraftKind.TEMPLATE
    assert draft.template_id == T.OFF_PLATFORM and draft.variant == "script"
    assert draft.evidence_ids == []


def test_off_platform_script_cites_platform_rule_and_adapts_salutation(world, ev):
    rule = ev.platform_rule(PlatformRuleKey.OFF_PLATFORM, "禁止站外联系和交易")
    ctx = make_context("加个微信", world)
    assert T.off_platform_script(ctx, EvidenceBundle(evidences=[rule])).evidence_ids == [rule.id]
    ctx_b = make_context("加个微信", world, shop_id=SHOP_B)  # 称呼"您"、语气专业
    assert T.off_platform_script(ctx_b).text == (
        "您好，平台规定不能通过站外联系和交易，"
        "您直接在店铺下单就好，我帮您看看有没有可以用的优惠券。"
    )


def test_greeting_cites_service_hours(world, ev):
    ctx = make_context("在吗？", world)
    hours = ev.policy(PolicyKey.SERVICE_HOURS)
    draft = T.greeting(ctx, EvidenceBundle(evidences=[hours]))
    assert draft.text == "亲，在的哦，请问有什么可以帮您？客服在线时间 9:00–23:00。"
    assert draft.evidence_ids == ["policy:service_hours"]
    plain = T.greeting(ctx, EvidenceBundle())
    assert plain.text == "亲，在的哦，请问有什么可以帮您？" and plain.evidence_ids == []


def test_greeting_service_hours_without_label(world, ev):
    ctx = make_context("在吗？", world, shop_id=SHOP_B)
    hours = ev.policy(PolicyKey.SERVICE_HOURS).model_copy(
        update={"content": "9:00–18:00", "data": {"key": "service_hours", "value": "9:00–18:00"}}
    )
    draft = T.greeting(ctx, EvidenceBundle(evidences=[hours]))
    assert draft.text == "您好，在的，请问有什么可以帮您？客服在线时间：9:00–18:00。"


def test_clarify_templates(world, route):
    ctx = make_context("这个多少钱", world, candidate_item_ids=["i_555", "i_777"])
    item = T.clarify_item(ctx)
    assert item.text == "亲，请问您问的是哪一款呢？"
    assert item.variant == "clarify" and item.need_clarify and item.template_id == T.CLARIFY_ITEM
    assert T.clarify_order(ctx).text == "亲，请问您问的是哪个订单呢？"
    q = T.clarify_question(ctx, "请问您问的是订单 o_1001 还是 o_1002 呢？")
    assert q.text == "亲，请问您问的是订单 o_1001 还是 o_1002 呢？"
    assert q.template_id == T.CLARIFY_ORDER
    assert (
        T.clarify_question(ctx, "亲，请问您问的是哪一款呢？").text == "亲，请问您问的是哪一款呢？"
    )
    assert T.clarify_question(ctx, "请问是哪一款？").template_id == T.CLARIFY_ITEM


def test_clarify_intent_from_top_intents(world, route):
    ctx = make_context("这个怎么弄", world)
    r = route(
        Intent.USAGE,
        RouteAction.CLARIFY,
        rule=7,
        top=[(Intent.USAGE, 0.45), (Intent.SIZE_ADVICE, 0.4)],
    )
    draft = T.clarify_intent(ctx, r)
    assert draft.text == "亲，请问您是想了解使用方法还是尺码、选型呢？"
    assert draft.template_id == T.CLARIFY_INTENT and draft.variant == "clarify"
    # "其他"不作为选项
    one = route(
        Intent.STOCK, RouteAction.CLARIFY, rule=7, top=[(Intent.OTHER, 0.5), (Intent.STOCK, 0.4)]
    )
    assert T.clarify_intent(ctx, one).text == "亲，请问您是想了解库存、补货吗？"
    none = route(Intent.OTHER, RouteAction.CLARIFY, rule=7, top=[(Intent.OTHER, 0.5)])
    assert T.clarify_intent(ctx, none).text == "亲，请问您具体想咨询什么问题呢？"


def test_soothe_and_tool_unavailable(world):
    ctx = make_context("怎么回事！！", world)
    soothe = T.soothe(ctx)
    assert soothe.text.startswith("亲，非常抱歉") and soothe.variant == "soothe"
    assert T.soothe(make_context("？", world, shop_id=SHOP_B)).text.startswith("您好，非常抱歉")
    unavailable = T.tool_unavailable(ctx)
    assert unavailable.text == "亲，我帮您查一下，请稍等哦。"
    assert unavailable.template_id == T.TOOL_UNAVAILABLE


def test_logistics_in_transit_says_as_of_10_30(world, ev):
    """§11.2：模板生成，写明"截至 10:30"，不承诺到货时间。"""
    ctx = make_context("我的快递到哪了", world, order_id="o_1002")
    evidence, analysis = ev.logistics("o_1002", summary="运输中，已到达杭州转运中心")
    draft = T.logistics(ctx, evidence, analysis)
    assert draft.text == "亲，截至 10:30，包裹最新状态：运输中，已到达杭州转运中心。"
    assert "截至 10:30" in draft.text
    assert draft.evidence_ids == ["tool:logistics:o_1002"]
    assert "预计" not in draft.text and "明天" not in draft.text


def test_logistics_station_pickup_guidance(world, ev):
    """§11.3 / §7.2：驿站签收给取件指引；不是当天的时间带日期。"""
    ctx = make_context("显示签收了但我没收到", world, order_id="o_1003")
    evidence = ev.o_1003()
    analysis = ev.logistics(
        "o_1003",
        summary="已签收，菜鸟驿站代收",
        pickup="包裹放在菜鸟驿站（幸福里店），请带上取件码到驿站领取",
    )[1]
    draft = T.logistics(ctx, evidence, analysis)
    assert "截至 09-27 15:23" in draft.text
    assert "菜鸟驿站（幸福里店），请带上取件码到驿站领取。" in draft.text


def test_logistics_stalled_buyer_hint_and_eta(world, ev):
    ctx = make_context("怎么还不到", world, order_id="o_1006")
    evidence = ev.o_1006_stalled()
    analysis = LogisticsAnalysis.model_validate(evidence.data["analysis"])
    draft = T.logistics(ctx, evidence, analysis)
    assert "已经帮您联系快递催查，有新进展会及时告诉您。" in draft.text
    assert "预计" not in draft.text
    eta = world.now + timedelta(days=1, hours=8)
    with_eta, a2 = ev.logistics("o_1002", summary="运输中", eta=eta)
    assert "平台显示预计 09-29 18:35 前送达" in T.logistics(ctx, with_eta, a2).text


def test_logistics_summary_with_as_of_is_not_duplicated(world, ev):
    ctx = make_context("到哪了", world, order_id="o_1002")
    evidence, analysis = ev.logistics(
        "o_1002", summary="运输中，截至 09-28 10:30 已到达杭州转运中心"
    )
    text = T.logistics(ctx, evidence, analysis).text
    assert text.count("截至") == 1
    assert text == "亲，运输中，截至 09-28 10:30 已到达杭州转运中心。"


def test_refund_progress_with_platform_timing(world, ev):
    ctx = make_context("退款什么时候到", world)
    evidence, progress = ev.refund("as_2008", "退款中，商家已同意退款，等待平台处理")
    timing = ev.platform_rule(
        PlatformRuleKey.REFUND_TIMING, "退款成功后原路退回，到账时间以平台为准"
    )
    draft = T.refund_progress(ctx, evidence, progress, timing=timing)
    assert draft.text == (
        "亲，截至 03:23，您的售后进度：退款中，商家已同意退款，等待平台处理。"
        "退款成功后原路退回，到账时间以平台为准。"
    )
    assert draft.evidence_ids == ["tool:aftersale:as_2008", timing.id]
    assert T.refund_progress(ctx, evidence, progress).evidence_ids == ["tool:aftersale:as_2008"]


def test_apply_aftersale_guidance(world, ev):
    ctx = make_context("怎么申请退款", world)
    rule = ev.platform_rule(
        PlatformRuleKey.APPLY_AFTERSALE,
        "在「我的淘宝-已买到的宝贝」找到订单，点「退款/售后」发起申请",
    )
    draft = T.apply_aftersale(ctx, rule)
    assert draft.text == (
        "亲，申请售后的方法：在「我的淘宝-已买到的宝贝」找到订单，点「退款/售后」发起申请。"
    )
    assert draft.evidence_ids == [rule.id]


def test_asks_how_to_apply():
    for text in ("怎么申请退款", "如何退货？", "在哪里申请售后", "退款怎么申请", "怎么退"):
        assert T.asks_how_to_apply(text), text
    for text in ("怎么还没退款", "退款什么时候到", "能退吗"):
        assert not T.asks_how_to_apply(text), text
