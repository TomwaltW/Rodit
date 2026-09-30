"""答案生成器（方案 §5.4 "能不用大模型就不用"、§5.7、§6.6、§11.2、§11.4、§12.2）。"""

from __future__ import annotations

import json

import pytest

from rodit.core.config import GenerationConfig
from rodit.core.enums import DraftKind, Intent, Mode, PolicyKey, Role, RouteAction
from rodit.core.errors import LLMError, LLMTimeout
from rodit.core.evidence import MISSING_ITEM, MISSING_ORDER, PlatformRuleKey, unavailable
from rodit.core.models import EvidenceBundle, RoutePart
from rodit.core.ports import AnswerGenerator
from rodit.generation import DefaultAnswerGenerator
from rodit.generation import templates as T
from rodit.testing.fakes import FakeLLMClient, make_context

CFG = GenerationConfig()


def llm_reply(answer: str, *ids: str, clarify: bool = False) -> dict:
    return {"answer": answer, "evidence_ids": list(ids), "need_clarify": clarify}


def by_variant(request) -> dict:
    """直接回答和附带追问返回不同文字。"""
    prompt = request.messages[1].content
    if "追问一个问题" in prompt:
        return llm_reply(
            "亲，这款面料是棉 95%、氨纶 5%，您平时穿什么码呢？", "item:i_555:attr:面料"
        )
    return llm_reply("亲，这款面料是棉 95%、氨纶 5% 哦", "item:i_555:attr:面料")


def gen(llm: FakeLLMClient, **kw) -> DefaultAnswerGenerator:
    return DefaultAnswerGenerator(llm, CFG, **kw)


def test_implements_protocol():
    assert isinstance(gen(FakeLLMClient(responses=[])), AnswerGenerator)


def test_off_platform_script_without_llm(world, route):
    llm = FakeLLMClient(responses=[])
    ctx = make_context("加个微信吧，转账给你便宜点", world)
    drafts = gen(llm).generate(
        ctx, route(Intent.OFF_PLATFORM, RouteAction.OFF_PLATFORM_SCRIPT, rule=1), EvidenceBundle()
    )
    assert [d.template_id for d in drafts] == [T.OFF_PLATFORM]
    assert drafts[0].text.startswith("亲，平台规定不能通过站外联系和交易哦")
    assert llm.calls == []


def test_clarify_route_uses_top_intents(world, route):
    llm = FakeLLMClient(responses=[])
    r = route(
        Intent.USAGE,
        RouteAction.CLARIFY,
        rule=7,
        top=[(Intent.USAGE, 0.5), (Intent.MALFUNCTION, 0.4)],
    )
    drafts = gen(llm).generate(make_context("这个不行啊", world), r, EvidenceBundle())
    assert [d.text for d in drafts] == ["亲，请问您是想了解使用方法还是使用故障呢？"]
    assert llm.calls == []


def test_evidence_clarify_question(world, route):
    llm = FakeLLMClient(responses=[])
    bundle = EvidenceBundle(clarify_question="请问您问的是哪一款呢？")
    drafts = gen(llm).generate(
        make_context("多少钱", world, candidate_item_ids=["i_555", "i_777"]),
        route(Intent.PRICE_PROMO),
        bundle,
    )
    assert [d.text for d in drafts] == ["亲，请问您问的是哪一款呢？"]
    assert drafts[0].need_clarify and drafts[0].variant == "clarify"
    assert llm.calls == []


def test_greeting_template_cites_service_hours(world, ev, route):
    llm = FakeLLMClient(responses=[])
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.SERVICE_HOURS)])
    drafts = gen(llm).generate(make_context("在吗？", world), route(Intent.GREETING), bundle)
    assert len(drafts) == 1
    assert drafts[0].template_id == T.GREETING
    assert drafts[0].evidence_ids == ["policy:service_hours"]
    assert "9:00–23:00" in drafts[0].text
    assert llm.calls == []


def test_logistics_uses_template_not_llm(world, ev, route):
    """§11.2：物流问题模板就能覆盖，不调大模型。"""
    llm = FakeLLMClient(responses=[])
    ctx = make_context("我的快递到哪了", world, order_id="o_1002")
    drafts = gen(llm).generate(
        ctx, route(Intent.LOGISTICS_TRACK), EvidenceBundle(evidences=[ev.o_1002()])
    )
    assert len(drafts) == 1 and drafts[0].template_id == T.LOGISTICS
    assert "截至 10:30" in drafts[0].text
    assert llm.calls == []


def test_logistics_picks_current_order(world, ev, route):
    llm = FakeLLMClient(responses=[])
    ctx = make_context("到哪了", world, order_id="o_1003")
    bundle = EvidenceBundle(evidences=[ev.o_1002(), ev.o_1003()])
    drafts = gen(llm).generate(ctx, route(Intent.LOGISTICS_TRACK), bundle)
    assert drafts[0].evidence_ids == ["tool:logistics:o_1003"]
    assert "菜鸟驿站（幸福里店）" in drafts[0].text


def test_refund_progress_template(world, ev, route):
    llm = FakeLLMClient(responses=[])
    evidence, _ = ev.refund("as_2008", "退款中，等待平台处理")
    timing = ev.platform_rule(PlatformRuleKey.REFUND_TIMING, "退款成功后原路退回")
    drafts = gen(llm).generate(
        make_context("退款什么时候到", world),
        route(Intent.REFUND_PROGRESS),
        EvidenceBundle(evidences=[evidence, timing]),
    )
    assert [d.template_id for d in drafts] == [T.REFUND_PROGRESS]
    assert drafts[0].evidence_ids == ["tool:aftersale:as_2008", timing.id]
    assert llm.calls == []


def test_tool_unavailable_without_other_evidence(world, route):
    llm = FakeLLMClient(responses=[])
    bundle = EvidenceBundle(unavailable=[unavailable("order"), unavailable("logistics")])
    drafts = gen(llm).generate(
        make_context("到哪了", world, order_id="o_1002"), route(Intent.LOGISTICS_TRACK), bundle
    )
    assert [d.text for d in drafts] == ["亲，我帮您查一下，请稍等哦。"]
    assert llm.calls == []


@pytest.mark.parametrize(
    ("missing", "expected"),
    [(MISSING_ORDER, "亲，请问您问的是哪个订单呢？"), (MISSING_ITEM, "亲，请问您问的是哪一款呢？")],
)
def test_missing_order_or_item_asks(world, route, missing, expected):
    llm = FakeLLMClient(responses=[])
    drafts = gen(llm).generate(
        make_context("能退吗", world),
        route(Intent.RETURN_POLICY),
        EvidenceBundle(missing=[missing]),
    )
    assert [d.text for d in drafts] == [expected]
    assert llm.calls == []


def test_no_evidence_no_suggestion(world, route):
    """§5.7：依据不足、也无法澄清 → 不推荐，也不调大模型。"""
    llm = FakeLLMClient(responses=[])
    bundle = EvidenceBundle(missing=["policy:invoice"])
    assert (
        gen(llm).generate(make_context("能开发票吗", world), route(Intent.INVOICE_GIFT), bundle)
        == []
    )
    assert llm.calls == []


def test_faq_direct_at_threshold(world, ev, route):
    llm = FakeLLMClient(responses=[])
    faq = ev.faq("发什么快递", score=CFG.faq_direct_min_score, faq_id=7)
    drafts = gen(llm).generate(
        make_context("发什么快递呀", world),
        route(Intent.SHIPPING_POLICY),
        EvidenceBundle(evidences=[faq, ev.policy(PolicyKey.COURIER)]),
    )
    assert len(drafts) == 1
    d = drafts[0]
    assert d.kind == DraftKind.FAQ and d.faq_id == 7 and d.evidence_ids == ["faq:7"]
    assert d.text == "亲，默认发中通快递哦，需要顺丰的话补 10 元运费差价就可以啦~"
    assert llm.calls == []


def test_faq_below_threshold_goes_to_llm(world, ev, route):
    llm = FakeLLMClient(lambda r: llm_reply("亲，默认发中通快递哦", "policy:courier"))
    faq = ev.faq("发什么快递", score=CFG.faq_direct_min_score - 0.01)
    drafts = gen(llm).generate(
        make_context("你们用啥快递", world, config_update={"suggestion_count": 1}),
        route(Intent.SHIPPING_POLICY),
        EvidenceBundle(evidences=[faq, ev.policy(PolicyKey.COURIER)]),
    )
    assert [d.kind for d in drafts] == [DraftKind.LLM]
    assert len(llm.calls_for("generate")) == 1


def test_faq_overridden_is_not_used_directly(world, ev, route):
    llm = FakeLLMClient(lambda r: llm_reply("亲，现货付款后 48 小时内发货", "policy:ship_time"))
    faq = ev.faq("什么时候发货", score=0.99, faq_id=2)
    bundle = EvidenceBundle(
        evidences=[faq, ev.policy(PolicyKey.SHIP_TIME)], overridden_ids=["faq:2"]
    )
    drafts = gen(llm).generate(
        make_context("啥时候发货", world), route(Intent.SHIPPING_POLICY), bundle
    )
    assert all(d.kind == DraftKind.LLM for d in drafts)


def test_faq_then_llm_when_enabled(world, ev, route):
    llm = FakeLLMClient(by_variant)
    faq = ev.faq("这款会透吗", score=0.93, faq_id=5)
    drafts = gen(llm, llm_alongside_faq=True).generate(
        make_context("会透吗", world, item_id="i_555"),
        route(Intent.PRODUCT_SPEC),
        EvidenceBundle(evidences=[faq, ev.item_attr("面料")]),
    )
    assert [d.kind for d in drafts] == [DraftKind.FAQ, DraftKind.LLM]
    assert drafts[1].variant == "direct"
    assert len(llm.calls) == 1


def test_llm_direct_and_followup(world, ev, route):
    llm = FakeLLMClient(by_variant)
    ctx = make_context("这个是纯棉的吗？", world, item_id="i_555")
    drafts = gen(llm).generate(
        ctx, route(Intent.PRODUCT_SPEC), EvidenceBundle(evidences=[ev.item_attr("面料")])
    )
    assert [d.variant for d in drafts] == ["direct", "with_followup"]
    assert all(d.kind == DraftKind.LLM for d in drafts)
    assert drafts[0].evidence_ids == ["item:i_555:attr:面料"]
    calls = llm.calls_for("generate")
    assert len(calls) == 2
    assert all(c.json_mode and c.shop_id == ctx.shop_id for c in calls)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_suggestion_count_respected(world, ev, route, count):
    llm = FakeLLMClient(by_variant)
    ctx = make_context("纯棉吗", world, item_id="i_555", config_update={"suggestion_count": count})
    drafts = gen(llm).generate(
        ctx, route(Intent.PRODUCT_SPEC), EvidenceBundle(evidences=[ev.item_attr("面料")])
    )
    assert 1 <= len(drafts) <= count
    assert len(llm.calls) == min(count, 2)


def test_soothe_count_and_llm(world, ev, route):
    r = route(Intent.QUALITY_ISSUE, RouteAction.SOOTHE, rule=3)
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.QUALITY_ISSUE)])
    reply = llm_reply(
        "亲，非常抱歉给您添麻烦了，收到质量问题商品请在申请售后时上传清晰照片，核实后退款或补发",
        "policy:quality_issue",
    )
    llm = FakeLLMClient(lambda r: reply)
    drafts = gen(llm).generate(make_context("东西是坏的！！", world), r, bundle)
    assert [d.template_id for d in drafts] == [T.SOOTHE, None]
    assert drafts[1].kind == DraftKind.LLM
    assert "道歉或安抚" in llm.calls[0].messages[1].content
    # 只有一个名额：只给安抚话术，不调大模型
    llm1 = FakeLLMClient(responses=[])
    one = gen(llm1).generate(
        make_context("坏了！！", world, config_update={"suggestion_count": 1}), r, bundle
    )
    assert [d.template_id for d in one] == [T.SOOTHE] and llm1.calls == []
    # 没有依据：只有安抚话术
    llm2 = FakeLLMClient(responses=[])
    empty = gen(llm2).generate(make_context("坏了！！", world), r, EvidenceBundle())
    assert [d.template_id for d in empty] == [T.SOOTHE] and llm2.calls == []


def test_need_clarify_becomes_clarify_variant(world, ev, route):
    def reply(request):
        if "追问一个问题" in request.messages[1].content:
            return llm_reply("亲，这款是棉的，您平时穿什么码呢？")
        return llm_reply("亲，请问您平时穿什么码呢？", clarify=True)

    llm = FakeLLMClient(reply)
    drafts = gen(llm).generate(
        make_context("我该买哪个码", world, item_id="i_777"),
        route(Intent.SIZE_ADVICE),
        EvidenceBundle(evidences=[ev.item_attr("面料", item_id="i_777")]),
    )
    # 追问之后不再用第二条（两条并行生成，结果和逐条生成一样）
    assert len(drafts) == 1 and drafts[0].variant == "clarify" and drafts[0].need_clarify
    assert drafts[0].text == "亲，请问您平时穿什么码呢？"
    assert len(llm.calls) == 2  # 并行时每条各调一次，不重试


def test_model_evidence_ids_kept_as_is(world, ev, route):
    llm = FakeLLMClient(lambda r: llm_reply("亲，是纯棉的", "item:i_555:attr:面料", "faq:404"))
    drafts = gen(llm).generate(
        make_context("纯棉吗", world, item_id="i_555", config_update={"suggestion_count": 1}),
        route(Intent.PRODUCT_SPEC),
        EvidenceBundle(evidences=[ev.item_attr("面料")]),
    )
    assert drafts[0].evidence_ids == ["item:i_555:attr:面料", "faq:404"]


def test_llm_failure_falls_back_to_templates(world, ev, route):
    """§12.2：大模型超时 / 报错 → 退回模板；没有模板就不推荐。"""
    rule = ev.platform_rule(PlatformRuleKey.APPLY_AFTERSALE, "在订单页点「申请售后」发起")
    bundle = EvidenceBundle(
        evidences=[ev.policy(PolicyKey.SEVEN_DAY_RETURN), rule],
        unavailable=[unavailable("order")],
    )
    llm = FakeLLMClient(lambda r: LLMTimeout("超时"))
    drafts = gen(llm).generate(make_context("能退吗", world), route(Intent.RETURN_POLICY), bundle)
    assert [d.template_id for d in drafts] == [T.APPLY_AFTERSALE, T.TOOL_UNAVAILABLE]
    assert len(llm.calls) == 2  # 两条并行各调一次，超时后不再重复调用
    # 第一条（直接回答）超时：和逐条调用一样，后面那条的结果也不用，退回模板
    first_times_out = FakeLLMClient(
        lambda r: (
            llm_reply("亲，支持七天无理由，您穿什么码呢？", "policy:seven_day_return")
            if "追问一个问题" in r.messages[1].content
            else LLMTimeout("超时")
        )
    )
    again = gen(first_times_out).generate(
        make_context("能退吗", world), route(Intent.RETURN_POLICY), bundle
    )
    assert [d.template_id for d in again] == [T.APPLY_AFTERSALE, T.TOOL_UNAVAILABLE]
    llm2 = FakeLLMClient(lambda r: LLMError("限流"))
    no_template = EvidenceBundle(evidences=[ev.item_attr("面料")])
    assert (
        gen(llm2).generate(
            make_context("纯棉吗", world, item_id="i_555"), route(Intent.PRODUCT_SPEC), no_template
        )
        == []
    )


def test_unparseable_output_tries_next_variant(world, ev, route):
    # 两条并行生成（调用先后不固定）：按变体给回复，直接回答那条格式不对
    llm = FakeLLMClient(
        lambda r: (
            json.dumps(llm_reply("亲，棉 95%，您穿什么码呢？"), ensure_ascii=False)
            if "追问一个问题" in r.messages[1].content
            else "亲，是纯棉的哦"
        )
    )
    drafts = gen(llm).generate(
        make_context("纯棉吗", world, item_id="i_555"),
        route(Intent.PRODUCT_SPEC),
        EvidenceBundle(evidences=[ev.item_attr("面料")]),
    )
    assert [d.variant for d in drafts] == ["with_followup"]
    assert len(llm.calls) == 2


def test_how_to_apply_uses_guidance_first(world, ev, route):
    rule = ev.platform_rule(PlatformRuleKey.APPLY_AFTERSALE, "在订单页点「申请售后」发起")
    llm = FakeLLMClient(lambda r: llm_reply("亲，在订单页申请售后就可以哦", rule.id))
    drafts = gen(llm).generate(
        make_context("怎么申请退款", world),
        route(Intent.RETURN_POLICY),
        EvidenceBundle(evidences=[rule, ev.policy(PolicyKey.SEVEN_DAY_RETURN)]),
    )
    assert drafts[0].template_id == T.APPLY_AFTERSALE
    assert drafts[1].kind == DraftKind.LLM and len(drafts) == 2
    assert len(llm.calls) == 1  # 模板占了一个名额，大模型只补一条


def test_split_multi_part_prompt(world, ev, route):
    llm = FakeLLMClient(lambda r: llm_reply("亲，现货付款后 48 小时内发货", "policy:ship_time"))
    original = route(
        Intent.PRODUCT_SPEC,
        RouteAction.SPLIT_MULTI,
        rule=5,
        parts=[
            RoutePart(intent=Intent.PRODUCT_SPEC, confidence=0.6),
            RoutePart(intent=Intent.SHIPPING_POLICY, confidence=0.4),
        ],
    )
    part = original.model_copy(
        update={"intent": Intent.SHIPPING_POLICY, "parts": [], "action": RouteAction.ANSWER}
    )
    gen(llm).generate(
        make_context("纯棉吗？什么时候发货", world, item_id="i_555"),
        part,
        EvidenceBundle(evidences=[ev.policy(PolicyKey.SHIP_TIME)]),
    )
    assert "本条只回答关于发货与快递的部分" in llm.calls[0].messages[1].content


def test_retry_feedback_reaches_prompt(world, ev, route):
    llm = FakeLLMClient(
        lambda r: llm_reply("亲，截至 10:30 已到达杭州转运中心", "tool:logistics:o_1002")
    )
    gen(llm).generate(
        make_context("明天能到吗", world, order_id="o_1002"),
        route(Intent.NOT_RECEIVED, high_risk=True),
        EvidenceBundle(evidences=[ev.o_1002()]),
        retry_feedback=["包含依据中没有的时效承诺：明天"],
    )
    assert all(
        "上次回答未通过质检：包含依据中没有的时效承诺：明天" in c.messages[1].content
        for c in llm.calls
    )


@pytest.mark.parametrize(
    "action", [RouteAction.NO_SUGGESTION, RouteAction.HUMAN_PRIORITY, RouteAction.TRANSFER_HUMAN]
)
def test_no_generation_actions(world, ev, route, action):
    llm = FakeLLMClient(responses=[])
    drafts = gen(llm).generate(
        make_context("我要投诉", world), route(Intent.COMPLAINT, action, rule=2), EvidenceBundle()
    )
    assert drafts == [] and llm.calls == []


def test_never_adds_auto_reply_prefix(world, ev, route):
    llm = FakeLLMClient(by_variant)
    ctx = make_context("纯棉吗", world, item_id="i_555", mode=Mode.AUTO)
    drafts = gen(llm).generate(
        ctx, route(Intent.PRODUCT_SPEC), EvidenceBundle(evidences=[ev.item_attr("面料")])
    )
    assert drafts and not any(d.text.startswith(CFG.auto_reply_prefix) for d in drafts)
    greeting = gen(llm).generate(ctx, route(Intent.GREETING), EvidenceBundle())
    assert not greeting[0].text.startswith(CFG.auto_reply_prefix)


def test_prompts_have_no_pii(world, ev, route):
    llm = FakeLLMClient(by_variant)
    ctx = make_context(
        "我电话 13912345678，寄到杭州市西湖区文三路100号，这个纯棉吗",
        world,
        item_id="i_555",
        history=[(Role.BUYER, "我住幸福小区 3 栋")],
    )
    gen(llm).generate(
        ctx, route(Intent.PRODUCT_SPEC), EvidenceBundle(evidences=[ev.item_attr("面料")])
    )
    prompt = llm.all_prompt_text()
    for secret in ("13912345678", "文三路", "幸福小区", ctx.buyer_ref):
        assert secret not in prompt
