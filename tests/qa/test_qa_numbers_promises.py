"""数字一致性与时效承诺（§5.5 第一道、§7.2、§11.5）：
日期时间的分量不放行别的数字、"30 号送达""周五前到"算时效承诺、买家的数字只放行身体尺寸、
否定 / 不确定的说法不算承诺、"没法保证""最大码"不算绝对化用语。"""

from __future__ import annotations

import pytest

from rodit.core.config import QAGateConfig
from rodit.core.decision import DecisionRequest
from rodit.core.enums import DraftKind, PolicyKey
from rodit.core.lexicon import find_absolute_terms
from rodit.core.models import Draft, EvidenceBundle
from rodit.core.questions import Q_GROUNDED, QA_QUESTIONS
from rodit.core.states import build_qa_state
from rodit.qa import NumberSet, blocking_hits, check_hard_rules, unsupported_numbers
from rodit.testing.fakes import make_context

CFG = QAGateConfig()


def run(text, ctx, evidence=None):
    bundle = evidence or EvidenceBundle()
    draft = Draft(text=text, kind=DraftKind.LLM, evidence_ids=bundle.ids())
    return check_hard_rules(text, ctx=ctx, draft=draft, evidence=bundle, config=CFG)


def blocking(text, ctx, evidence=None) -> list[str]:
    return [h.rule for h in blocking_hits(run(text, ctx, evidence)[1])]


def _o_1002_realtime(ev):
    """和真实工具证据一样，content 里带 fmt_time 写法的时间（"截至 09-28 10:30"）。"""
    return ev.logistics(
        "o_1002",
        summary="运输中，已到达杭州转运中心",
        content="运输中，截至 09-28 10:30 已到达杭州转运中心；付款时间 09-25 10:35",
    )[0]


# ---------------------------------------------------------------- 日期时间的分量


def test_date_time_components_do_not_support_other_numbers():
    support = NumberSet()
    support.add_text("截至 09-28 10:30")
    texts = [t.text for t in unsupported_numbers("满 28 减 10，30 元", support)]
    assert texts == ["28", "10", "30 元"]
    # 日期、时间整体照样有依据
    assert unsupported_numbers("截至 09-28 10:30", support) == []
    assert unsupported_numbers("9 月 28 日", support) == []


def test_clock_hours_are_compared_as_times():
    """证据写 16:00，回答写"16 点""下午 4 点"照样有依据；分量不再当普通数字。"""
    support = NumberSet()
    support.add_text("当天 16:00 前付款当天发货；客服 9:00–23:30")
    assert unsupported_numbers("16 点前付款当天发货", support) == []
    assert unsupported_numbers("下午 4 点前付款", support) == []
    assert unsupported_numbers("晚上 11 点半下班", support) == []
    assert [t.text for t in unsupported_numbers("17 点前付款", support)] == ["17 点"]
    assert [t.text for t in unsupported_numbers("满 16 减 9", support)] == ["16", "9"]


def test_realtime_bundle_times_do_not_whitelist_compensation(world, ev):
    ctx = make_context("到哪了", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[_o_1002_realtime(ev)])
    assert "number_mismatch" in blocking("亲，给您补偿 10 元红包", ctx, bundle)
    assert "number_mismatch" in blocking("亲，给您补偿 35 元红包", ctx, bundle)
    assert blocking("亲，截至 09-28 10:30 已到达杭州转运中心", ctx, bundle) == []


@pytest.mark.parametrize(
    "text", ["亲，包裹运输中，预计 30 号送达", "亲，周五前能到哦", "亲，周末就能到"]
)
def test_day_of_month_and_weekday_promises(world, ev, text):
    ctx = make_context("什么时候到", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[_o_1002_realtime(ev)])
    _, hits = run(text, ctx, bundle)
    assert "time_promise" in [h.rule for h in hits], text


def test_day_of_month_promise_allowed_with_platform_eta(world, ev):
    """接口给了预计送达时间（"平台预计送达时间 09-30 18:00"）时，"30 号送达"有依据。"""
    ctx = make_context("什么时候到", world, order_id="o_1002")
    eta = ev.logistics(
        "o_1002",
        summary="运输中",
        content="运输中，截至 09-28 10:30 已到达杭州转运中心；平台预计送达时间 09-30 18:00",
    )[0]
    bundle = EvidenceBundle(evidences=[eta])
    assert blocking("亲，平台预计 30 号送达哦", ctx, bundle) == []
    assert "time_promise" in blocking("亲，预计 29 号送达哦", ctx, bundle)


# ---------------------------------------------------------------- 买家说的数字


def _price_bundle(ev):
    price = ev.policy(PolicyKey.PROMOTIONS)
    return EvidenceBundle(
        evidences=[price.model_copy(update={"content": "售价 259.00 元；店铺满 199 减 20"})]
    )


def test_buyer_price_is_not_evidence(world, ev):
    """审查复现：买家问"150 包邮卖不卖"，回答"可以 150 包邮"不能因为买家说过 150 就放行。"""
    ctx = make_context("150 包邮卖不卖？", world, item_id="i_555")
    assert "number_mismatch" in blocking("亲，可以 150 包邮卖给您哦", ctx, _price_bundle(ev))
    ctx2 = make_context("打 8 折行吗", world, item_id="i_555")
    assert "number_mismatch" in blocking("亲，可以给您打 8 折", ctx2, _price_bundle(ev))


def test_buyer_duration_is_not_evidence(world, ev):
    ctx = make_context("3 天能到吗", world)
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.SHIP_TIME)])
    for text in ["亲，3 天左右可以到的", "亲，3 天后就能到货", "亲，48 小时左右就能到"]:
        rules = blocking(text, ctx, bundle)
        assert "time_promise" in rules, text
    assert "number_mismatch" in blocking("亲，3 天左右可以到的", ctx, bundle)


def test_buyer_body_measurements_still_supported(world, ev):
    ctx = make_context("身高 165 体重 110 斤穿什么码", world, item_id="i_555")
    bundle = EvidenceBundle(evidences=[ev.size_advice()])
    assert blocking("亲，身高 165、体重 110 斤建议选 M 码哦", ctx, bundle) == []
    assert blocking("亲，165 cm 的话建议 M 码", ctx, bundle) == []


def test_offline_judge_does_not_take_buyer_price_as_evidence(world, ev):
    from rodit.app.offline import HeuristicQAJudge

    ctx = make_context("150 包邮卖不卖？", world, item_id="i_555")
    bundle = _price_bundle(ev)

    def grounded(answer: str) -> float:
        draft = Draft(text=answer, kind=DraftKind.LLM, evidence_ids=bundle.ids())
        request = DecisionRequest(
            model="offline",
            state=build_qa_state(ctx, draft, bundle),
            questions={Q_GROUNDED: QA_QUESTIONS[Q_GROUNDED]},
        )
        return HeuristicQAJudge().decide(request).answers[Q_GROUNDED].noul

    assert grounded("亲，可以 150 包邮卖给您哦") < 0.5
    assert grounded("亲，现在 259 元，满 199 减 20 哦") > 0.5


# ---------------------------------------------------------------- 否定 / 不确定不算承诺


@pytest.mark.parametrize(
    "text",
    [
        "亲，暂时没法确定明天能不能送到，有更新第一时间告诉您",
        "亲，这个不一定明天能到哦",
        "亲，明天能不能到要看快递的派送情况",
    ],
)
def test_negated_or_uncertain_time_is_not_a_promise(world, ev, text):
    ctx = make_context("明天能到吗？", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    assert blocking(text, ctx, bundle) == [], text


def test_real_promise_still_blocked_next_to_negation(world, ev):
    ctx = make_context("明天能到吗？", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    assert "time_promise" in blocking("亲，不用担心，明天就能到", ctx, bundle)
    assert "time_promise" in blocking("亲，没问题，明天能到", ctx, bundle)


@pytest.mark.parametrize(
    "text",
    [
        "到货时间我们没法保证哦",
        "这个无法保证",
        "不能保证一定到",
        "这款最大码是 XL",
        "最小码是 S",
        "最大号是 XL",
        "最小号是 S",
    ],
)
def test_honest_limits_and_size_range_are_not_absolute_terms(text):
    assert "保证" not in find_absolute_terms(text)
    assert "最" not in find_absolute_terms(text)


def test_size_answer_with_largest_size_passes(world, ev):
    ctx = make_context("170 穿什么码", world, item_id="i_555")
    bundle = EvidenceBundle(evidences=[ev.size_advice()])
    rules = blocking("亲，这款最大码是 XL，身高 170 cm 建议 L 码", ctx, bundle)
    assert "absolute_term" not in rules
    # 真正的绝对化用语照样拦截
    assert "absolute_term" in blocking("亲，这款是全网最好的", ctx, bundle)
    assert "absolute_term" in blocking("亲，保证明天到", ctx, bundle)


@pytest.mark.parametrize(
    ("question", "answer", "ok"),
    [
        # §11.1 的写法：买家没写单位的身高，回答里复述不算编造
        ("165 110斤穿什么码", "亲，您 165、110 斤建议 M 码", True),
        ("165 110斤穿什么码", "亲，身高 165 cm、体重 110 斤建议 M 码", True),
        ("我165，穿什么码", "亲，身高 165 建议 M 码", True),
        # 同样没写单位，但分句在谈价格 / 时间：不算依据
        ("150 包邮卖不卖？", "亲，可以 150 包邮卖给您哦", False),
        ("165 110斤穿什么码，3 天能到吗", "亲，3 天能到哦", False),
        ("我165，150 包邮卖吗，穿什么码", "亲，可以 150 包邮哦", False),
        ("165 110斤穿什么码，3 天能到吗", "亲，您 165、110 斤建议 M 码", True),
        # 买家说的身高不能给回答里的价格当依据
        ("165 110斤穿什么码", "亲，这款 165 元", False),
    ],
)
def test_buyer_bare_size_numbers_in_sizing_clause(question, answer, ok):
    ns = NumberSet()
    ns.add_text("尺码建议：M 码适合 100–115 斤")
    ns.add_buyer_text(question)
    assert (unsupported_numbers(answer, ns) == []) is ok
