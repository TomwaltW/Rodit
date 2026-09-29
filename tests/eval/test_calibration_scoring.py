"""§13.2 评分、§13.3 阈值校准、§13.1 评测集的自检。"""

from __future__ import annotations

import pytest

from rodit.core.enums import Confidence, DraftKind, Intent, Mode, RouteAction
from rodit.core.evidence import INTENT_NEEDS
from rodit.core.models import EngineResult, EvidenceRefView, RouteDecision, SuggestionCard
from rodit.eval.calibration import Sample, calibrate, pick_threshold, reliability
from rodit.eval.dataset import CASES, EvalCase, cases
from rodit.eval.scoring import score_case, summarize
from rodit.testing import fixtures
from rodit.testing.fixtures import build_demo_world


def test_pick_threshold_matches_doc_algorithm():
    # 预测为 size_advice：0.95/0.9/0.85 都对，0.7 错，0.6 对
    samples = [
        ("size_advice", 0.95, "size_advice"),
        ("size_advice", 0.90, "size_advice"),
        ("size_advice", 0.85, "size_advice"),
        ("size_advice", 0.70, "product_spec"),
        ("size_advice", 0.60, "size_advice"),
        ("product_spec", 0.80, "size_advice"),
    ]
    pick = pick_threshold(samples, "size_advice", 0.90)
    # t ≤ 0.60 时 5 条里 4 条对（0.8 < 0.9）；t ∈ (0.70, 0.85] 时 3 条全对，覆盖率 3/5
    assert pick is not None
    assert pick.threshold == pytest.approx(0.71)
    assert pick.precision == 1.0 and pick.coverage == pytest.approx(0.6)
    assert pick.recall == pytest.approx(0.6) and pick.support == 5
    loose = pick_threshold(samples, "size_advice", 0.75)
    assert loose.threshold == pytest.approx(0.50) and loose.coverage == pytest.approx(1.0)
    assert pick_threshold(samples, "size_advice", 1.01) is None
    assert pick_threshold(samples, "greeting", 0.9) is None  # 没有该类型的样本


def test_calibrate_and_reliability():
    samples = [Sample("a", 0.95, "a"), Sample("a", 0.9, "b"), Sample("b", 0.55, "b")]
    picks = calibrate(samples, ["a", "b"], target_precision=0.9)
    assert picks["a"].threshold == pytest.approx(0.91)
    assert picks["b"].threshold == pytest.approx(0.5)
    bins, ece = reliability(samples, bins=5)
    assert sum(b.count for b in bins) == 3
    assert 0.0 <= ece <= 1.0


def test_dataset_is_consistent_with_demo_world():
    world = build_demo_world()
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))
    assert {c.expected_intent for c in CASES} == set(Intent)  # 20 个问题类型都有用例
    test = cases("test")
    assert 0.1 <= len(test) / len(CASES) <= 0.35 and cases("dev")
    for c in CASES:
        shop_id = getattr(fixtures, c.shop)
        buyer = getattr(fixtures, c.buyer)
        if c.item_id:
            world.item(shop_id, c.item_id)
        if c.order_id:
            assert world.order(shop_id, c.order_id).buyer_ref == buyer, c.id
        assert INTENT_NEEDS[c.expected_intent] is not None


def _result(case: EvalCase, text: str, *, intent=None, action=RouteAction.ANSWER, cites=()):
    route = RouteDecision(
        action=action,
        rule=6,
        intent=intent or case.expected_intent,
        intent_confidence=0.93,
        provider="fake",
    )
    cards = (
        [
            SuggestionCard(
                text=text,
                intent=route.intent,
                kind=DraftKind.LLM,
                confidence=Confidence.HIGH,
                evidence=[EvidenceRefView(id=i, label=i) for i in cites],
            )
        ]
        if text
        else []
    )
    return EngineResult(
        shop_id="s_1001",
        conversation_id="c",
        msg_ids=["m"],
        mode=Mode.ASSIST,
        route=route,
        suggestions=cards,
    )


def test_score_case_dimensions():
    case = next(c for c in CASES if c.id == "ship-01")
    good = score_case(
        case, _result(case, "亲，现货付款后 48 小时内发货哦~", cites=["policy:ship_time"])
    )
    assert (good.factual, good.on_topic, good.compliance, good.expression) == (2, 2, 2, 2)
    wrong = score_case(case, _result(case, "亲，24 小时内发货哦", cites=["policy:ship_time"]))
    assert wrong.factual == 0
    uncited = score_case(case, _result(case, "亲，48 小时内发货"))
    assert uncited.factual == 1
    absolute = score_case(
        case, _result(case, "亲，48 小时发货，全网最快", cites=["policy:ship_time"])
    )
    assert absolute.compliance == 0
    spec = next(c for c in CASES if c.id == "spec-07")
    silk = _result(spec, "亲，面料：桑蚕丝 100%", cites=["item:i_888:attr:面料"])
    assert score_case(spec, silk).compliance == 0  # 没有商品卡片可核对
    assert score_case(spec, silk, evidence_text="面料：桑蚕丝 100%").compliance == 2
    off = score_case(case, _result(case, "加我微信聊", cites=["policy:ship_time"]))
    assert off.compliance == 0
    wrong_intent = score_case(
        case, _result(case, "亲，48 小时", intent=Intent.STOCK, cites=["policy:ship_time"])
    )
    assert wrong_intent.on_topic == 1 and not wrong_intent.intent_ok
    none = score_case(case, _result(case, ""))
    assert none.factual == 0 and none.expression == 0
    long = score_case(case, _result(case, "亲，48 小时" + "哦" * 130, cites=["policy:ship_time"]))
    assert long.expression == 1


def test_no_suggestion_cases_and_gate():
    cmp = next(c for c in CASES if c.id == "cmp-01")
    s1 = score_case(cmp, _result(cmp, "", action=RouteAction.HUMAN_PRIORITY))
    assert (s1.factual, s1.on_topic, s1.compliance, s1.expression) == (2, 2, 2, 2)
    ship = next(c for c in CASES if c.id == "ship-01")
    s2 = score_case(ship, _result(ship, "亲，48 小时内发货", cites=["policy:ship_time"]))
    report = summarize([s1, s2], [cmp, ship])
    assert report.passes_gate and report.full_marks["factual"] == 1.0
    s3 = score_case(ship, _result(ship, "亲，24 小时内发货", cites=["policy:ship_time"]))
    report = summarize([s1, s3], [cmp, ship])
    assert not report.passes_gate and len(report.failures) == 1
    assert report.confusion["complaint"]["complaint"] == 1


@pytest.mark.parametrize(
    ("text", "phrase", "said"),
    [
        ("亲，现货付款后 48 小时内发货，不能保证明天发出哦", "保证明天", False),
        ("亲，这款不是纯棉的哦，面料是棉 95%", "纯棉的哦", False),
        ("亲，这款没有抗皱功能哦", "抗皱", False),
        ("亲，客服这边没法额外给您打折哦", "给您打折", False),
        ("亲，目前没有赠品活动", "有赠品", False),
        ("亲，这款是纯棉的哦", "纯棉的哦", True),
        ("亲，保证明天到", "保证明天", True),
        ("亲，不一定今天到，但保证明天到", "保证明天", True),  # 否定不跨小句
        ("亲，不过保证明天到", "保证明天", True),  # "不过"是转折，不是否定
    ],
)
def test_must_not_include_ignores_negated_or_uncertain_mentions(text, phrase, said):
    """和质检硬规则一致：否定、不确定说法里提到的错误说法不算说了这句错话。"""
    from rodit.eval.scoring import _says

    assert _says(text, phrase) is said


def test_must_include_ignores_spacing():
    case = next(c for c in CASES if c.id == "ship-01")
    case = case.model_copy(update={"must_include": ["10 月 8 日"], "must_cite": []})
    result = _result(case, "亲，预计10月8日补货哦")
    assert score_case(case, result).factual == 2
