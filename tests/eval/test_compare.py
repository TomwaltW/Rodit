"""判断层离线对比评测（rodit.eval.compare；方案 §6.3、§6.4、§13.3，§15 Phase 0 / Phase 3）。

不联网：境内大模型用 FakeLLMClient，Jev 用注入的假传输层（dict → dict），或真正的
JevHttpTransport 配假的 urlopen。检查：
- 指标的算法（ECE、Brier、macro-F1、一致率、§13.3 阈值）用手算的小样本核对；
- 请求：和线上路由同一个入口构造（逐条和完整引擎里路由器收到的请求比对），各判断层收到同一份，
  发给 Jev 的只多 synthetic=True，评测集每条都通过个人信息检查；不是合成评测集的用例整轮拒绝；
- 判断层的运行：关键词规则、完美的 / 校准差的 / 准确率低的假判断层、失败、超时、拒绝发送、
  用量、耗时；
- §6.3 结论两个方向都能判出来，容差、没有对照、没有测试集、全部失败都有交代；
- 文本报告和 JSON（没有买家原文、没有密钥、接口只留主机）；命令行参数和退出码。
"""

from __future__ import annotations

import io
import itertools
import json
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

import pytest

from rodit.core.decision import DecisionRequest, DecisionResponse
from rodit.core.enums import Intent, RouteAction
from rodit.core.errors import ComplianceError, DecisionError, LLMError, LLMTimeout
from rodit.core.pii import contains_pii, mask_pii
from rodit.core.ports import LLMRequest
from rodit.core.questions import (
    Q_EMOTION,
    Q_INTENT,
    Q_MULTI_INTENT,
    Q_NEED_HUMAN,
    ROUTER_MODEL,
    ROUTER_QUESTIONS,
)
from rodit.decision import ClassifierProvider, JevConditions, RuleProvider, write_artifact
from rodit.decision._common import iter_strings
from rodit.eval import compare as cmp
from rodit.eval.calibration import Sample
from rodit.eval.cases_base import EvalCase
from rodit.eval.compare import (
    CONDITION_LABELS,
    NOT_AUTHORIZATION,
    SYNTHETIC_NOTICE,
    CalibrationReport,
    CompareCase,
    CompareReport,
    FailureSummary,
    LatencySummary,
    ProviderInfo,
    ProviderReport,
    ProviderSpec,
    SplitScore,
    ThresholdReport,
    agreement,
    brier_score,
    build_compare_cases,
    calibration_of,
    compare,
    emotion_label,
    intent_scores,
    jev_request,
    jev_spec,
    jev_verdict,
    llm_spec,
    main,
    need_human_label,
    render_text,
    rules_spec,
    run_provider,
    samples_of,
    threshold_rows,
    top_confusions,
)
from rodit.eval.dataset import CASES, cases
from rodit.eval.harness import demo_harness
from rodit.eval.jev_transport import API_KEY_ENV, BASE_URL_ENV, JevHttpTransport, JevTimeout
from rodit.eval.runner import run_case
from rodit.testing.fakes import FakeLLMClient, router_answers
from rodit.testing.fixtures import DEFAULT_NOW

BY_ID = {c.id: c for c in CASES}
KEY = "jev-synthetic-compare-key-DO-NOT-LEAK"  # 合成的假密钥，只用来检查不外泄
JEV_BASE = "https://jev.example.test"


@pytest.fixture(scope="module")
def all_cases() -> list[CompareCase]:
    return build_compare_cases(cases("all"))


@pytest.fixture(scope="module")
def small(all_cases) -> list[CompareCase]:
    """dev 和 test 各取一些，覆盖多个类型（快）。"""
    dev = [c for c in all_cases if c.split == "dev"][::6]
    test = [c for c in all_cases if c.split == "test"][::3]
    return dev + test


def _key(state: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in state.items() if k != "synthetic"}, sort_keys=True)


def _wrong_intent(case: CompareCase) -> Intent:
    return next(i for i in Intent if i not in case.ok_intents)


def _answers(case: CompareCase | None, *, wrong: bool, confidence: float) -> dict[str, Any]:
    if case is None:  # 连通性检查的问候
        return router_answers(Intent.GREETING, confidence)
    intent = _wrong_intent(case) if wrong else case.expected
    return router_answers(
        intent,
        confidence,
        need_human=0.95 if case.need_human_label else 0.02,
        emotion=3.5 if case.emotion_label else 0.5,
    )


class Scripted:
    """假的境内判断层：按用例标注作答。wrong 里的用例答错；fail 里的抛 DecisionError。"""

    accepts_buyer_data = True

    def __init__(
        self,
        compare_cases: Iterable[CompareCase],
        *,
        name: str = "rules",
        wrong: Callable[[CompareCase], bool] = lambda c: False,
        confidence: Callable[[CompareCase, bool], float] = lambda c, wrong: 1.0,
        fail: frozenset[str] = frozenset(),
    ) -> None:
        self.name = name
        self.by_key = {_key(c.request.state): c for c in compare_cases}
        self.wrong = wrong
        self.confidence = confidence
        self.fail = fail
        self.requests: list[DecisionRequest] = []

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        case = self.by_key[_key(request.state)]
        if case.id in self.fail:
            raise DecisionError("假判断层不可用")
        wrong = self.wrong(case)
        answers = _answers(case, wrong=wrong, confidence=self.confidence(case, wrong))
        return DecisionResponse(model=request.model, answers=answers)


class JevScript:
    """假的 Jev 传输层（dict → dict，Jev 格式）。按用例作答；可以设定答错、出错、坏响应、用量。"""

    def __init__(
        self,
        compare_cases: Iterable[CompareCase],
        *,
        wrong: Callable[[CompareCase], bool] = lambda c: False,
        confidence: Callable[[CompareCase, bool], float] = lambda c, wrong: 1.0,
        errors: dict[str, Exception] | None = None,
        bad: dict[str, Any] | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.by_key = {_key(c.request.state): c for c in compare_cases}
        self.wrong = wrong
        self.confidence = confidence
        self.errors = errors or {}
        self.bad = bad or {}
        self.usage = usage
        self.payloads: list[dict[str, Any]] = []
        self.totals = [0, 0, 0]

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        case = self.by_key.get(_key(payload["state"]))
        if case is not None and case.id in self.errors:
            raise self.errors[case.id]
        if case is not None and case.id in self.bad:
            return self.bad[case.id]
        wrong = case is not None and self.wrong(case)
        conf = self.confidence(case, wrong) if case is not None else 0.9
        answers = _answers(case, wrong=wrong, confidence=conf)
        if self.usage:
            self.totals[0] += self.usage["input"]
            self.totals[1] += self.usage["output"]
            self.totals[2] += 1
        return {
            "model": payload["model"],
            "answers": {k: a.model_dump(mode="json") for k, a in answers.items()},
        }


class JevScriptWithUsage(JevScript):
    def usage_totals(self) -> tuple[int, int, int]:
        return self.totals[0], self.totals[1], self.totals[2]


def _spec(provider: Any, name: str = "rules") -> ProviderSpec:
    return ProviderSpec(ProviderInfo(name=name, label=name, kind="domestic"), provider)


# ================================================================ 指标算法（手算核对）


def test_brier_score_by_hand():
    samples = [Sample("a", 1.0, "a"), Sample("a", 0.5, "b"), Sample("b", 0.8, "b")]
    assert brier_score(samples) == round((0 + 0.25 + 0.04) / 3, 4)
    assert brier_score([]) is None
    assert brier_score([Sample("a", 1.0, "b")]) == 1.0


def test_ece_uses_ten_equal_width_bins_by_hand():
    samples = [
        Sample("a", 0.95, "a"),
        Sample("a", 0.95, "b"),  # [0.9, 1.0]：平均 0.95，准确率 0.5
        Sample("a", 0.35, "a"),
        Sample("a", 0.35, "a"),  # [0.3, 0.4)：平均 0.35，准确率 1.0
    ]
    report = calibration_of(samples)
    assert report.n == 4
    assert report.ece == round(2 / 4 * 0.45 + 2 / 4 * 0.65, 4) == 0.55
    assert [(b.lower, b.upper, b.count) for b in report.bins] == [(0.3, 0.4, 2), (0.9, 1.0, 2)]
    assert report.mean_confidence == 0.65 and report.accuracy == 0.75
    # 1.0 进最后一箱、0.0 进第一箱；完美校准时 ECE = 0
    edge = calibration_of([Sample("a", 1.0, "a"), Sample("a", 0.0, "b")])
    assert [(b.lower, b.upper) for b in edge.bins] == [(0.0, 0.1), (0.9, 1.0)]
    assert edge.ece == 0.0 and edge.brier == 0.0
    empty = calibration_of([])
    assert empty.n == 0 and empty.ece is None and empty.bins == []


def test_intent_scores_and_macro_f1_by_hand():
    A, B, C = Intent.PRODUCT_SPEC, Intent.SIZE_ADVICE, Intent.STOCK  # noqa: N806
    rows = [
        (A, frozenset({A}), A),
        (A, frozenset({A}), B),  # 误判
        (B, frozenset({B}), B),
        (B, frozenset({B, A}), A),  # 可接受的备选：算对
        (C, frozenset({C}), None),  # 调用失败：按错算
    ]
    table, macro = intent_scores(rows)
    assert list(table) == ["product_spec", "size_advice", "stock"]  # 枚举顺序
    a, b, c = table["product_spec"], table["size_advice"], table["stock"]
    # 准确率按用例的真实类型：第 4 条（真实 B、答了可接受的 A）算 B 答对
    assert (a.n, a.correct, a.accuracy, a.n_pred, a.precision) == (2, 1, 0.5, 2, 1.0)
    assert (b.n, b.correct, b.accuracy, b.n_pred, b.precision) == (2, 2, 1.0, 2, 0.5)
    # F1 按算对口径（同 samples_of）：第 4 条的真实类型记成 A，只算 A 的一个 TP
    # A：TP 2（第 1、4 条）、真实 3 条、预测 2 条 → R 2/3、P 1；B：TP 1、真实 1 条、预测 2 条
    assert a.f1 == round(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3), 4) == 0.8
    assert b.f1 == round(2 * 0.5 * 1.0 / 1.5, 4)
    assert (c.n, c.correct, c.accuracy, c.n_pred, c.precision, c.f1) == (1, 0, 0.0, 0, None, 0.0)
    assert macro == round((0.8 + 2 / 3 + 0) / 3, 4)
    assert intent_scores([]) == ({}, None)


def test_macro_f1_does_not_penalise_acceptable_alternatives():
    """答了可接受的备选类型是答对：不能让两个类型各缺一半（真实那边没有预测、预测那边没有真实）。"""
    A, B = Intent.PRODUCT_SPEC, Intent.SIZE_ADVICE  # noqa: N806
    table, macro = intent_scores([(B, frozenset({B, A}), A)])
    assert macro == 1.0
    assert (table["product_spec"].precision, table["product_spec"].f1) == (1.0, 1.0)
    b = table["size_advice"]
    assert (b.n, b.correct, b.accuracy, b.n_pred, b.f1) == (1, 1, 1.0, 0, None)  # 不计 F1


def test_macro_f1_is_perfect_when_every_answer_is_acceptable(all_cases):
    """有备选类型的用例全都答备选类型（都算对）：准确率和 macro-F1 都是 1。"""
    with_alt = [c for c in all_cases if len(c.ok_intents) > 1]
    assert len(with_alt) >= 5

    def alternative(c: CompareCase) -> Intent:
        return next(i for i in Intent if i in c.ok_intents and i != c.expected)

    outcomes = [
        cmp.CaseOutcome(case_id=c.id, intent=alternative(c), confidence=0.9) for c in with_alt
    ]
    score = cmp.split_score(with_alt, outcomes)
    assert score.accuracy == 1.0
    assert score.macro_f1 == 1.0


def test_samples_record_acceptable_predictions_as_correct(all_cases):
    case = next(c for c in all_cases if len(c.ok_intents) > 1)
    alt = next(i for i in case.ok_intents if i != case.expected)
    wrong = _wrong_intent(case)
    outcomes = [
        cmp.CaseOutcome(case_id=case.id, intent=alt, confidence=0.8),
        cmp.CaseOutcome(case_id=case.id, intent=wrong, confidence=0.7),
        cmp.CaseOutcome(case_id=case.id, error_kind="error", error_type="DecisionError"),
    ]
    samples = samples_of([case, case, case], outcomes)
    assert samples == [
        Sample(alt.value, 0.8, alt.value),
        Sample(wrong.value, 0.7, case.expected.value),
    ]


def test_agreement_by_hand():
    pairs = [(True, 0.9), (True, 0.5), (False, 0.1), (False, 0.7), (False, 0.6999999999)]
    a = agreement(pairs, 0.7)
    assert (a.n, a.positives, a.hits, a.negatives, a.false_alarms) == (5, 2, 1, 3, 2)
    assert a.agree == 2 and a.rate == 0.4  # 阈值相等（含浮点误差）算触发，同 routing.policy
    assert agreement([], 0.7) is None


def test_top_confusions_ignore_failures_and_acceptable_answers(all_cases):
    c1 = next(c for c in all_cases if c.expected == Intent.URGE_SHIP and len(c.ok_intents) == 1)
    c2 = next(c for c in all_cases if c.expected == Intent.STOCK and len(c.ok_intents) == 1)
    outcomes = [
        cmp.CaseOutcome(case_id=c1.id, intent=Intent.SHIPPING_POLICY, confidence=0.9),
        cmp.CaseOutcome(case_id=c1.id, intent=Intent.SHIPPING_POLICY, confidence=0.9),
        cmp.CaseOutcome(case_id=c2.id, intent=Intent.OTHER, confidence=0.4),
        cmp.CaseOutcome(case_id=c2.id, error_kind="timeout", error_type="JevTimeout"),
        cmp.CaseOutcome(case_id=c2.id, intent=Intent.STOCK, confidence=0.9),
    ]
    rows = top_confusions([c1, c1, c2, c2, c2], outcomes)
    assert [(r.true, r.predicted, r.count) for r in rows] == [
        ("urge_ship", "shipping_policy", 2),
        ("stock", "other", 1),
    ]
    assert len(top_confusions([c1, c1, c2, c2, c2], outcomes, limit=1)) == 1


def test_threshold_rows_pick_on_calibration_and_validate_on_holdout():
    spec = Intent.PRODUCT_SPEC.value
    other = Intent.OTHER.value
    calib = [Sample(spec, 0.95, spec)] * 9 + [Sample(spec, 0.6, other)] + [Sample(other, 0.9, spec)]
    holdout = (
        [Sample(spec, 0.97, spec)] * 3 + [Sample(spec, 0.96, other)] + [Sample(other, 0.5, spec)]
    )
    rows = threshold_rows(
        calib,
        holdout,
        [Intent.PRODUCT_SPEC, Intent.GREETING],
        0.95,
        configured={Intent.PRODUCT_SPEC: 0.97},
    )
    row, empty = rows
    assert row.intent == spec and row.configured == 0.97
    assert row.support == 10  # 校准集里真实为 product_spec 的：9 + 被判成 other 的 1
    assert row.pick is not None
    # 0.50–0.60：10 条里 9 条对（90% < 95%）；0.61 起排除了 0.60 那条误判，达标且覆盖率最高
    assert row.pick.threshold == 0.61
    assert (row.pick.precision, row.pick.coverage) == (1.0, 0.9)
    loose = threshold_rows(calib, None, [Intent.PRODUCT_SPEC], 0.9)[0].pick
    assert loose is not None and (loose.threshold, loose.coverage) == (0.5, 1.0)
    assert row.holdout_support == 4 and row.holdout_picked == 4
    assert row.holdout_precision == 0.75 and row.holdout_coverage == 1.0
    assert empty.support == 0 and empty.pick is None and empty.configured is None
    assert empty.holdout_support == 0 and empty.holdout_precision is None
    unreachable = threshold_rows([Sample(spec, 0.99, other)] * 2, None, [Intent.PRODUCT_SPEC], 0.9)
    assert unreachable[0].pick is None and unreachable[0].holdout_support is None


# ================================================================ 代理标注


@pytest.mark.parametrize(
    ("case_id", "need_human", "emotion"),
    [
        ("cmp-01", True, None),  # 投诉：要人工；情绪说不清
        ("cmp-02", True, None),
        ("human-01", True, None),  # 只允许"提醒人工"
        ("other-07", True, None),  # 标了 human
        ("emo-01", None, True),  # 允许人工但不是必须；情绪到了安抚线
        ("urge-09", None, True),
        ("emo-04", False, None),  # 可以安抚也可以正常回答：情绪说不清
        ("other-10", False, None),  # 标了 emotion，但不是安抚 / 人工（正面情绪）
        ("spec-01", False, False),
        ("off-01", False, False),
        ("multi-01", False, False),
    ],
)
def test_proxy_labels_from_route_annotations(case_id, need_human, emotion):
    case = BY_ID[case_id]
    assert need_human_label(case) is need_human
    assert emotion_label(case) is emotion


def test_proxy_labels_cover_most_cases():
    nh = [need_human_label(c) for c in CASES]
    emo = [emotion_label(c) for c in CASES]
    assert sum(1 for x in nh if x) >= 10 and sum(1 for x in nh if x is False) >= 250
    assert sum(1 for x in emo if x) >= 4 and sum(1 for x in emo if x is False) >= 250


# ================================================================ 请求


def test_requests_are_router_requests_built_by_core_states(all_cases):
    assert len(all_cases) == len(CASES) == 324
    assert [c.id for c in all_cases] == [c.id for c in CASES]
    for c in all_cases:
        assert c.synthetic is True
        assert c.request.model == ROUTER_MODEL
        assert c.request.questions == ROUTER_QUESTIONS
        assert set(c.request.state) == {"platform", "messages", "item_card", "order_card"}
        assert c.request.state["messages"][-1]["role"] == "buyer"
        assert "synthetic" not in c.request.state
    sku = next(c for c in all_cases if c.id == "stock-01")
    assert sku.request.state["item_card"]["item_id"] == "i_555"
    order = next(c for c in all_cases if c.id == "ctx-03")
    assert order.request.state["order_card"]["order_id"] == "o_1002"
    ref = next(c for c in all_cases if c.id == "ctx-01")
    assert [m["role"] for m in ref.request.state["messages"]] == ["buyer", "agent", "buyer"]


@pytest.mark.parametrize(
    "case_id", ["spec-01", "stock-01", "ctx-01", "ctx-02", "ctx-03", "multi-01", "cmp-01"]
)
def test_requests_equal_what_the_live_router_receives(all_cases, case_id):
    """和完整引擎里路由器实际收到的请求逐字段相同（同一个 build_request、同样的会话理解）。"""

    class Recorder:
        name = "rules"
        accepts_buyer_data = True

        def __init__(self) -> None:
            self.inner = RuleProvider()
            self.requests: list[DecisionRequest] = []

        def decide(self, request: DecisionRequest) -> DecisionResponse:
            self.requests.append(request)
            return self.inner.decide(request)

    recorder = Recorder()
    run_case(demo_harness(None, recorder), BY_ID[case_id])
    mine = next(c for c in all_cases if c.id == case_id)
    assert recorder.requests, "引擎没有调路由判断层"
    assert recorder.requests[0].model_dump(mode="json") == mine.request.model_dump(mode="json")


def test_every_dataset_state_sent_to_jev_is_synthetic_and_pii_free(all_cases):
    for c in all_cases:
        request = jev_request(c)
        assert request.state["synthetic"] is True
        assert {k: v for k, v in request.state.items() if k != "synthetic"} == c.request.state
        assert request.questions == c.request.questions and request.model == c.request.model
        assert not [s for s in iter_strings(request.state) if contains_pii(s)], c.id
        assert "synthetic" not in c.request.state  # 原请求没被改
    pii_cases = [c for c in CASES if "pii" in c.tags]
    assert pii_cases
    by_id = {c.id: c for c in all_cases}
    for case in pii_cases:
        sent = json.dumps(jev_request(by_id[case.id]).state, ensure_ascii=False)
        assert mask_pii(case.text) in sent


def test_only_dataset_cases_count_as_synthetic():
    original = BY_ID["spec-01"]
    homemade = EvalCase(id="x-1", text="这个是纯棉的吗？", expected_intent=Intent.PRODUCT_SPEC)
    edited = original.model_copy(update={"text": "这个是纯棉的吗？我手机13812345678"})
    built = build_compare_cases([original, homemade, edited])
    assert [c.synthetic for c in built] == [True, False, False]
    with pytest.raises(ComplianceError, match="合成评测集"):
        jev_request(built[1])
    transport = JevScript(built)
    with pytest.raises(ComplianceError):
        run_provider(jev_spec(transport), built)  # 整轮拒绝：一条也不发
    with pytest.raises(ComplianceError):
        compare(built, [rules_spec(), jev_spec(transport)])
    assert transport.payloads == []
    # 境内判断层不受影响
    assert len(run_provider(rules_spec(), built)) == 3


def test_every_provider_gets_the_same_request(small):
    first = Scripted(small, name="rules")
    second = Scripted(small, name="classifier")
    jev = JevScript(small)
    report = compare(small, [_spec(second, "classifier"), _spec(first), jev_spec(jev)])
    assert [p.info.name for p in report.providers] == ["rules", "classifier", "jev"]  # 固定顺序
    assert len(first.requests) == len(second.requests) == len(jev.payloads) == len(small)
    for a, b, payload, case in zip(
        first.requests, second.requests, jev.payloads, small, strict=True
    ):
        assert a.model_dump(mode="json") == b.model_dump(mode="json")
        assert a.model_dump(mode="json") == case.request.model_dump(mode="json")
        assert a is not case.request  # 深拷贝：判断层改了也不影响别的
        assert payload["model"] == "jev-1.13.0"
        assert payload["questions"] == a.model_dump(mode="json")["questions"]
        assert payload["state"] == {**a.state, "synthetic": True}


# ================================================================ 运行判断层


def test_rules_provider_over_the_whole_dataset(all_cases):
    report = compare(all_cases, [rules_spec()])
    (r,) = report.providers
    assert r.info.name == "rules" and r.info.kind == "domestic"
    assert r.overall.n == r.overall.answered == 324
    assert set(r.by_split) == {"dev", "test"}
    assert r.by_split["dev"].n + r.by_split["test"].n == 324
    assert 0.5 < r.overall.accuracy < 1.0
    ok = sum(1 for row in report.cases if row.results["rules"].ok)
    assert r.overall.correct == ok and r.overall.accuracy == round(ok / 324, 4)
    assert sum(row.n for row in r.by_intent.values()) == 324
    assert r.calibration.n == 324 and r.calibration.ece is not None
    assert sum(b.count for b in r.calibration.bins) == 324
    assert r.need_human is not None and r.emotion is not None
    assert r.need_human.threshold == 0.7 and r.emotion.threshold == 3.0
    assert r.failures.failures == r.failures.timeouts == r.failures.refused == 0
    assert r.tokens is None
    t = r.thresholds
    assert (t.calibrated_on, t.validated_on) == ("dev", "test")
    assert {row.intent for row in t.auto} == {
        "greeting",
        "shipping_policy",
        "logistics_track",
        "invoice_gift",
        "return_policy",
        "product_spec",
    }
    assert report.verdict.status == "not_run"
    assert [c.status for c in report.verdict.conditions] == ["manual"] * 3 + ["not_run"]


def test_rules_predictions_match_the_live_router(all_cases):
    """关键词规则在对比评测里的判断，和完整引擎里路由器记下的类型一致（同一份请求）。"""
    report = compare(all_cases, [rules_spec()])
    by_id = {row.id: row for row in report.cases}
    for case_id in ("spec-01", "size-01", "ship-01", "off-01", "ctx-02"):
        harness = demo_harness()
        result = run_case(harness, BY_ID[case_id])
        assert result is not None and result.route is not None
        assert by_id[case_id].results["rules"].intent == result.route.intent.value
        assert by_id[case_id].results["rules"].confidence == pytest.approx(
            result.route.intent_confidence
        )


def test_perfect_and_miscalibrated_fake_providers(all_cases):
    perfect = Scripted(all_cases, name="rules")
    overconfident = Scripted(
        all_cases,
        name="classifier",
        wrong=lambda c: c.id < "c",  # 确定的一部分答错（id 以 a、b 开头的）
        confidence=lambda c, wrong: 0.99,  # 对错都说 0.99
    )
    report = compare(all_cases, [_spec(perfect), _spec(overconfident, "classifier")])
    p, m = report.providers
    assert p.overall.accuracy == 1.0 and p.overall.macro_f1 == 1.0
    assert p.calibration.ece == 0.0 and p.calibration.brier == 0.0
    assert p.top_confusions == []
    assert p.need_human.rate == 1.0 and p.emotion.rate == 1.0
    assert all(row.pick is not None and row.pick.threshold == 0.5 for row in p.thresholds.assist)
    wrong_n = sum(1 for c in all_cases if c.id < "c")
    assert 0 < wrong_n < 324
    assert m.overall.correct == 324 - wrong_n
    acc = (324 - wrong_n) / 324
    assert m.calibration.ece == pytest.approx(0.99 - acc, abs=1e-4)
    assert m.calibration.brier == pytest.approx(
        (wrong_n * 0.99**2 + (324 - wrong_n) * 0.01**2) / 324, abs=1e-4
    )
    assert m.top_confusions and m.top_confusions[0].count >= 1


def test_failures_count_as_wrong_but_calibration_only_uses_answers(small):
    failing = {small[0].id, small[1].id}
    spec = _spec(Scripted(small, fail=frozenset(failing)))
    r = compare(small, [spec]).providers[0]
    n = len(small)
    assert r.overall.n == n and r.overall.answered == n - 2
    assert r.overall.accuracy == round((n - 2) / n, 4)
    assert r.overall.answered_accuracy == 1.0
    assert r.calibration.n == n - 2 and r.calibration.ece == 0.0
    assert r.failures.failures == 2 and r.failures.timeouts == 0
    assert r.failures.by_type == {"DecisionError": 2}
    assert {c.id for c in r.failures.cases} == failing
    assert r.by_intent[small[0].expected.value].correct < r.by_intent[small[0].expected.value].n


def test_latency_uses_the_injected_timer(small):
    ticks = itertools.count()

    def timer() -> float:  # 每次读数 +10 毫秒：一次判断读两次，耗时 10 毫秒
        return next(ticks) * 0.010

    report = compare(small, [rules_spec()], timer=timer)
    r = report.providers[0]
    assert r.latency.n == len(small)
    assert r.latency.p50_ms == r.latency.p95_ms == r.latency.max_ms == pytest.approx(10.0)
    assert all(row.results["rules"].latency_ms == pytest.approx(10.0) for row in report.cases)


def test_progress_is_called_per_case(small):
    seen: list[tuple[str, int, int, str]] = []
    compare(
        small,
        [rules_spec()],
        progress=lambda name, i, n, case, o: seen.append((name, i, n, case.id)),
    )
    assert [s[1] for s in seen] == list(range(1, len(small) + 1))
    assert {s[0] for s in seen} == {"rules"} and {s[2] for s in seen} == {len(small)}


def test_compare_rejects_bad_input(small):
    with pytest.raises(ValueError, match="没有用例"):
        compare([], [rules_spec()])
    with pytest.raises(ValueError, match="没有判断层"):
        compare(small, [])
    with pytest.raises(ValueError, match="重名"):
        compare(small, [rules_spec(), rules_spec()])


# ================================================================ 境内大模型（FakeLLMClient）


def _prompt_state(request: LLMRequest) -> dict[str, Any]:
    user = request.messages[-1].content
    head = "## 状态（state）\n"
    return json.loads(user[user.index(head) + len(head) : user.index("\n\n## 问题")])


def _llm(
    compare_cases: Iterable[CompareCase], *, wrong: frozenset[str] = frozenset()
) -> FakeLLMClient:
    by_key = {_key(c.request.state): c for c in compare_cases}

    def handler(request: LLMRequest) -> Any:
        if request.purpose == "other":  # 连通性检查
            return "OK"
        assert request.purpose == "route"
        case = by_key[_key(_prompt_state(request))]
        intent = _wrong_intent(case) if case.id in wrong else case.expected
        return {
            Q_INTENT: {"choice": intent.value},
            Q_NEED_HUMAN: {"answer": "是" if case.need_human_label else "否", "p": 0.9},
            Q_EMOTION: {"score": 4 if case.emotion_label else 0},
            Q_MULTI_INTENT: {"answer": "否", "p": 0.1},
        }

    return FakeLLMClient(handler, model="fake-judge")


def test_incloud_llm_votes_and_tokens(small):
    wrong = frozenset({small[2].id})
    spec = llm_spec(_llm(small, wrong=wrong))
    assert spec.info.vote_samples == 3 and spec.info.model == "fake-judge"
    assert spec.info.timeout_s == 1.0  # DecisionConfig 默认：路由单次 ≤ 1 秒
    r = compare(small, [spec]).providers[0]
    assert r.overall.answered == len(small)
    assert r.overall.correct == len(small) - 1
    assert r.calibration.mean_confidence == 1.0  # 三票一致 → 占比 1.0
    # 三票投票：是否题阈值按票数粒度换算（0.7 → 2/3），同线上 routing.policy
    assert r.need_human.threshold == pytest.approx(2 / 3)
    assert r.need_human.rate == 1.0
    assert r.tokens is not None and r.tokens.reported_cases == len(small)
    assert r.tokens.input_tokens > 0 and r.tokens.output_tokens > 0
    calls = len(small) * 3
    assert r.tokens.input_tokens + r.tokens.output_tokens > calls  # 每次调用都计了
    report = compare(small, [rules_spec(), spec])
    assert report.judge_prompt_ver is not None  # 跑了大模型时记下判断层提示词版本


def test_incloud_llm_failures_and_timeouts(small):
    target = small[0]
    by_key = {_key(c.request.state): c for c in small}
    base = _llm(small)

    def handler(request: LLMRequest) -> Any:
        case = by_key[_key(_prompt_state(request))]
        if case.id == target.id:
            return LLMTimeout("大模型调用超时（1 秒）")
        if case.id == small[1].id:
            return LLMError("大模型接口限流（HTTP 429）")
        return base._handler(request)

    r = compare(small, [llm_spec(FakeLLMClient(handler))]).providers[0]
    assert r.failures.failures == 2 and r.failures.timeouts == 1
    kinds = {c.id: c.kind for c in r.failures.cases}
    assert kinds == {target.id: "timeout", small[1].id: "error"}
    assert set(r.failures.by_type) == {"LLMTimeout", "LLMError"}


# ================================================================ Jev（注入的假传输层）


def test_jev_path_with_jev_format_responses(small):
    transport = JevScriptWithUsage(small, usage={"input": 100, "output": 0})
    spec = jev_spec(transport, model="jev-laya-multilingual")
    assert spec.info.kind == "jev" and spec.info.model == "jev-laya-multilingual"
    assert spec.provider.purpose == "offline_eval"
    assert spec.provider.accepts_buyer_data is False
    assert spec.provider.conditions == JevConditions()  # 条件默认都不满足
    report = compare(small, [rules_spec(), spec])
    jev = report.providers[1]
    assert jev.overall.accuracy == 1.0
    assert jev.tokens is not None
    assert (jev.tokens.input_tokens, jev.tokens.output_tokens) == (100 * len(small), 0)
    assert {p["model"] for p in transport.payloads} == {"jev-laya-multilingual"}
    assert all(p["state"]["synthetic"] is True for p in transport.payloads)


def test_jev_malformed_responses_timeouts_and_errors(small):
    a, b, c, d, e = (x.id for x in small[:5])
    transport = JevScript(
        small,
        errors={a: JevTimeout("Jev 调用超时（10 秒）"), b: ConnectionError("boom")},
        bad={
            c: {"model": "jev-1.13.0"},  # 没有 answers
            d: {
                "model": "jev-1.13.0",
                "answers": {"intent": {"type": "choice", "choice": "nope", "confidence": 0.9}},
            },
            e: ["not", "a", "dict"],
        },
    )
    r = compare(small, [rules_spec(), jev_spec(transport)]).providers[1]
    assert r.overall.answered == len(small) - 5
    assert r.failures.failures == 5 and r.failures.timeouts == 1 and r.failures.refused == 0
    kinds = {f.id: (f.kind, f.error_type) for f in r.failures.cases}
    assert kinds[a] == ("timeout", "JevTimeout")
    assert kinds[b] == ("error", "DecisionError")
    assert {kinds[x][1] for x in (c, d, e)} == {"DecisionError"}
    messages = {f.id: f.message for f in r.failures.cases}
    assert "超时" in messages[a]
    assert r.overall.accuracy == round((len(small) - 5) / len(small), 4)


def test_jev_refuses_states_with_pii_without_sending(small):
    """state 里有个人信息时 JevProvider 拒绝发送，记为拒绝发送，传输层收不到。"""
    target = small[0]
    state = json.loads(json.dumps(target.request.state))
    state["item_card"] = {"item_id": "i_555", "category": "女装", "title": "联系 13812345678"}
    tainted = replace(target, request=target.request.model_copy(update={"state": state}))
    transport = JevScript(small)
    r = compare([tainted, *small[1:]], [rules_spec(), jev_spec(transport)]).providers[1]
    assert r.failures.refused == 1 and r.failures.failures == 0
    (failed,) = r.failures.cases
    assert failed.id == target.id and failed.kind == "refused"
    assert failed.error_type == "ComplianceError"
    assert "13812345678" not in failed.message
    assert len(transport.payloads) == len(small) - 1
    assert all("13812345678" not in json.dumps(p, ensure_ascii=False) for p in transport.payloads)
    assert r.latency.n == len(small) - 1  # 没发出的不计耗时


# ================================================================ §6.3 结论


def _score(acc: float | None, ece: float | None, *, n: int = 50, answered: int | None = None):
    answered = n if answered is None else answered
    correct = round((acc or 0) * n)
    return SplitScore(
        n=n,
        answered=answered,
        correct=correct,
        accuracy=acc,
        answered_accuracy=acc,
        macro_f1=None,
        ece=ece,
        brier=None,
    )


def _report(
    name: str,
    kind: str,
    acc: float | None,
    ece: float | None,
    *,
    splits: tuple[str, ...] = ("dev", "test"),
    answered: int | None = None,
    failures: int = 0,
) -> ProviderReport:
    score = _score(acc, ece, answered=answered)
    return ProviderReport(
        info=ProviderInfo(name=name, label=name, kind=kind),
        overall=score,
        by_split={s: score for s in splits},
        by_intent={},
        top_confusions=[],
        need_human=None,
        emotion=None,
        calibration=CalibrationReport(
            n=0, ece=None, brier=None, mean_confidence=None, accuracy=None
        ),
        thresholds=ThresholdReport(
            calibrated_on="dev",
            validated_on="test",
            n_calibration=0,
            n_validation=0,
            assist_precision=0.9,
            auto_precision=0.98,
            assist=[],
            auto=[],
        ),
        latency=LatencySummary(n=0, p50_ms=None, p95_ms=None, max_ms=None),
        failures=FailureSummary(failures=failures, timeouts=0, refused=0, by_type={}, cases=[]),
    )


def test_verdict_met_when_jev_reaches_the_best_domestic_level():
    reports = [
        _report("rules", "domestic", 0.80, 0.05),
        _report("incloud_llm", "domestic", 0.92, 0.04),
        _report("jev", "jev", 0.915, 0.055),
    ]
    v = jev_verdict(reports)
    assert v.status == "met" and v.split == "test"
    assert v.domestic == "incloud_llm"  # 准确率最高的境内方案
    assert v.accuracy_ok and v.ece_ok
    assert [(c.name, c.status) for c in v.conditions] == [
        ("in_cloud_deployment", "manual"),
        ("operator_and_dpa", "manual"),
        ("platform_approved", "manual"),
        ("chinese_eval_passed", "met"),
    ]
    assert v.synthetic_only == SYNTHETIC_NOTICE and v.not_authorization == NOT_AUTHORIZATION


def test_verdict_unmet_on_accuracy_or_calibration_with_numbers():
    domestic = _report("incloud_llm", "domestic", 0.92, 0.04)
    low = jev_verdict([domestic, _report("jev", "jev", 0.85, 0.03)])
    assert low.status == "unmet" and low.accuracy_ok is False and low.ece_ok is True
    assert "85.0%" in low.reasons[0] and "92.0%" in low.reasons[0]
    miscal = jev_verdict([domestic, _report("jev", "jev", 0.95, 0.20)])
    assert miscal.status == "unmet" and miscal.accuracy_ok is True and miscal.ece_ok is False
    assert "0.2000" in miscal.reasons[1] and "0.0400" in miscal.reasons[1]
    assert miscal.conditions[-1].status == "unmet"


@pytest.mark.parametrize(
    ("jev_acc", "jev_ece", "acc_tol", "ece_tol", "status"),
    [
        (0.91, 0.06, 0.01, 0.02, "met"),  # 正好在容差上
        (0.9099, 0.06, 0.01, 0.02, "unmet"),
        (0.91, 0.0601, 0.01, 0.02, "unmet"),
        (0.88, 0.06, 0.05, 0.02, "met"),  # 放宽容差
        (0.92, 0.045, 0.0, 0.0, "unmet"),  # 不给容差
        (0.92, 0.04, 0.0, 0.0, "met"),
    ],
)
def test_verdict_tolerances(jev_acc, jev_ece, acc_tol, ece_tol, status):
    reports = [
        _report("classifier", "domestic", 0.92, 0.04),
        _report("jev", "jev", jev_acc, jev_ece),
    ]
    v = jev_verdict(reports, accuracy_tolerance=acc_tol, ece_tolerance=ece_tol)
    assert v.status == status
    assert (v.accuracy_tolerance, v.ece_tolerance) == (acc_tol, ece_tol)


def test_verdict_best_domestic_ties_go_to_lower_ece():
    reports = [
        _report("classifier", "domestic", 0.90, 0.10),
        _report("incloud_llm", "domestic", 0.90, 0.03),
        _report("jev", "jev", 0.90, 0.06),
    ]
    v = jev_verdict(reports)
    assert v.domestic == "incloud_llm" and v.status == "unmet" and v.ece_ok is False


def test_verdict_edge_cases():
    domestic = _report("classifier", "domestic", 0.8, 0.05)
    assert jev_verdict([domestic]).status == "not_run"
    alone = jev_verdict([_report("jev", "jev", 0.9, 0.01)])
    assert alone.status == "undetermined" and "境内方案" in alone.reasons[0]
    assert "--llm env" in alone.reasons[0] and "--classifier" in alone.reasons[0]
    dev_only = jev_verdict(
        [
            _report("classifier", "domestic", 0.8, 0.05, splits=("dev",)),
            _report("jev", "jev", 0.9, 0.01, splits=("dev",)),
        ]
    )
    assert dev_only.status == "undetermined" and dev_only.split == "dev"
    assert "测试集" in dev_only.reasons[0]
    dead = jev_verdict([domestic, _report("jev", "jev", 0.0, None, answered=0, failures=50)])
    assert dead.status == "unmet" and "没有答出" in dead.reasons[0]
    assert any("50 条调用失败" in r for r in dead.reasons)


def test_verdict_from_a_limit_sample_is_undetermined_but_keeps_the_numbers():
    """--limit 抽样（可能只抽到一两条测试集用例）不下结论，满足、未满足都不判；数字照列作参考。"""
    domestic = _report("incloud_llm", "domestic", 0.92, 0.04)
    good = jev_verdict([domestic, _report("jev", "jev", 0.93, 0.03)], partial=True)
    assert good.status == "undetermined" and good.partial
    assert good.conditions[-1].status == "undetermined"
    assert good.accuracy_ok is True and good.ece_ok is True
    assert (good.domestic, good.jev_accuracy, good.domestic_accuracy) == ("incloud_llm", 0.93, 0.92)
    assert "部分评测" in good.reasons[0]
    assert any("93.0%" in r and "92.0%" in r for r in good.reasons)
    bad = jev_verdict([domestic, _report("jev", "jev", 0.5, 0.3)], partial=True)
    assert bad.status == "undetermined" and (bad.accuracy_ok, bad.ece_ok) == (False, False)
    dead = jev_verdict(
        [domestic, _report("jev", "jev", 0.0, None, answered=0, failures=50)], partial=True
    )
    assert dead.status == "undetermined" and dead.conditions[-1].status == "undetermined"


def test_verdict_needs_the_in_cloud_llm_or_classifier_as_baseline():
    """§6.4：境内路由方案是平台云内大模型和路由分类器，关键词规则只是兜底（置信度没有校准过），
    只对照规则时条件 4 无法判定（数字照列作参考）。"""
    rules = _report("rules", "domestic", 0.80, 0.05)
    v = jev_verdict([rules, _report("jev", "jev", 0.95, 0.01)])
    assert v.status == "undetermined" and v.conditions[-1].status == "undetermined"
    assert "境内大模型或分类器" in v.reasons[0] and "--llm env" in v.reasons[0]
    assert v.domestic == "rules" and v.accuracy_ok is True and v.ece_ok is True
    # 规则和分类器都跑了：只拿分类器对照（规则准确率再高、校准再差也不作对照）
    v = jev_verdict(
        [
            _report("rules", "domestic", 0.95, 0.30),
            _report("classifier", "domestic", 0.90, 0.02),
            _report("jev", "jev", 0.95, 0.30),
        ]
    )
    assert v.domestic == "classifier" and v.status == "unmet"
    assert v.accuracy_ok is True and v.ece_ok is False


def test_condition_labels_match_jev_conditions():
    assert list(CONDITION_LABELS) == list(JevConditions.model_fields)


def _classifier_baseline(compare_cases: Iterable[CompareCase]) -> ProviderSpec:
    """假的路由分类器：id 以 1 结尾的答错，置信度对的 0.95、错的 0.6。"""
    return _spec(
        Scripted(
            compare_cases,
            name="classifier",
            wrong=lambda c: c.id.endswith("1"),
            confidence=lambda c, wrong: 0.6 if wrong else 0.95,
        ),
        "classifier",
    )


def test_verdict_both_ways_end_to_end(all_cases):
    """完整跑一遍：Jev 好的时候判满足，校准差 / 准确率低的时候判未满足（对照路由分类器）；
    只有关键词规则作对照时无法判定。"""
    rules_only = compare(all_cases, [rules_spec(), jev_spec(JevScript(all_cases))])
    assert rules_only.verdict.status == "undetermined"
    assert rules_only.verdict.domestic == "rules" and rules_only.verdict.n == 66
    rules_test = rules_only.providers[0].by_split["test"]
    assert rules_only.verdict.domestic_accuracy == rules_test.accuracy

    good = compare(
        all_cases,
        [rules_spec(), _classifier_baseline(all_cases), jev_spec(JevScript(all_cases))],
    )
    assert good.verdict.status == "met"
    assert good.verdict.domestic == "classifier" and good.verdict.n == 66
    assert good.verdict.domestic_accuracy == good.providers[1].by_split["test"].accuracy
    assert good.verdict.jev_accuracy == 1.0 and good.verdict.jev_ece == 0.0

    overconfident = JevScript(all_cases, confidence=lambda c, wrong: 0.3)  # 全对却只给 0.3
    bad_cal = compare(all_cases, [_classifier_baseline(all_cases), jev_spec(overconfident)])
    assert bad_cal.verdict.status == "unmet"
    assert bad_cal.verdict.accuracy_ok is True and bad_cal.verdict.ece_ok is False

    sloppy = JevScript(
        all_cases,
        wrong=lambda c: c.id[0] in "abcdefghijklmnopqrs",  # 大部分答错
        confidence=lambda c, wrong: 0.0 if wrong else 1.0,  # 校准完美
    )
    low_acc = compare(all_cases, [_classifier_baseline(all_cases), jev_spec(sloppy)])
    assert low_acc.verdict.status == "unmet"
    assert low_acc.verdict.accuracy_ok is False and low_acc.verdict.ece_ok is True


def test_verdict_with_limit_is_undetermined_end_to_end(all_cases):
    """compare(limit=...)：抽样跑（哪怕 Jev 全对）也不判满足。"""
    sample = [c for c in all_cases if c.split == "test"][:1]
    report = compare(
        sample,
        [_classifier_baseline(all_cases), jev_spec(JevScript(all_cases))],
        split="test",
        limit=1,
    )
    v = report.verdict
    assert v.partial and v.n == 1 and v.jev_accuracy == 1.0
    assert v.status == "undetermined" and v.conditions[-1].status == "undetermined"
    assert "：无法判定" in render_text(report) and "：满足" not in render_text(report)


# ================================================================ 报告


def test_text_report_sections_and_notices(small):
    report = compare(small, [rules_spec(), _classifier_baseline(small), jev_spec(JevScript(small))])
    text = render_text(report)
    for needle in (
        "Rodit 判断层离线对比",
        "== 问题类型",
        "== 按问题类型的准确率",
        "== 主要误判",
        "== §6.6 规则 2",
        "== 概率校准",
        "== §13.3 阈值（用 dev 校准，test 上验证）",
        "智能辅助，target_precision = 0.90",
        "target_precision = 0.98",
        "== 耗时",
        "== §6.3 Jev 重新引入生产链路的条件",
        SYNTHETIC_NOTICE,
        NOT_AUTHORIZATION,
        "4. chinese_eval_passed",
        "：满足",
    ):
        assert needle in text, needle
    assert text.count("需人工确认") == 3
    assert "jev：Jev（境外处理" in text
    assert "代理标注" in text
    assert "没有报告 token 用量" in text


def test_json_round_trips_and_has_no_buyer_text(small, all_cases):
    report = compare(all_cases, [rules_spec(), jev_spec(JevScript(all_cases))])
    raw = json.dumps(report.model_dump(mode="json"), ensure_ascii=False)
    again = CompareReport.model_validate(json.loads(raw))
    assert again == report
    long_texts = [c.text for c in CASES if len(c.text) >= 6]
    assert long_texts
    assert not [t for t in long_texts if t in raw]
    assert "13812345678" not in raw
    row = report.cases[0]
    assert set(row.results) == {"rules", "jev"}
    assert row.id == all_cases[0].id


# ================================================================ 命令行


def _run(argv: list[str], **kw: Any) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out, err=err, **kw)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def no_jev_env(monkeypatch):
    for name in (BASE_URL_ENV, API_KEY_ENV, "RODIT_JEV_TIMEOUT_S", "RODIT_JEV_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_cli_default_runs_rules_only(no_jev_env):
    code, out, err = _run(["--limit", "20", "--seed", "3"])
    assert code == 0, err
    assert "部分评测：随机抽 20 条（seed 3）" in out
    assert "  - rules：" in out and "  - jev" not in out
    assert "本次没有跑 Jev" in out


def test_cli_json_to_stdout_and_file(tmp_path):
    code, out, _ = _run(["--split", "test", "--json"])
    assert code == 0
    data = json.loads(out)
    assert data["split"] == "test" and data["n"] == 66
    assert data["providers"][0]["info"]["name"] == "rules"
    assert data["providers"][0]["thresholds"]["calibrated_on"] == "test"
    target = tmp_path / "sub" / "compare.json"
    code, out, _ = _run(["--limit", "10", "--json", str(target)])
    assert code == 0 and "完整结果（JSON）已写入" in out and "Rodit 判断层离线对比" in out
    assert json.loads(target.read_text(encoding="utf-8"))["n"] == 10


def test_cli_targets_and_tolerances_are_passed_through():
    code, out, _ = _run(
        ["--limit", "40", "--json", "--target-assist", "0.8", "--target-auto", "0.95"]
    )
    assert code == 0
    t = json.loads(out)["providers"][0]["thresholds"]
    assert (t["assist_precision"], t["auto_precision"]) == (0.8, 0.95)
    code, out, _ = _run(
        ["--limit", "5", "--json", "--accuracy-tolerance", "0.03", "--ece-tolerance", "0"]
    )
    v = json.loads(out)["verdict"]
    assert (v["accuracy_tolerance"], v["ece_tolerance"]) == (0.03, 0.0)


@pytest.mark.parametrize(
    ("argv", "words"),
    [
        (["--providers", "jev"], "--jev env"),
        (["--providers", "rules,classifier"], "--classifier"),
        (["--providers", "incloud_llm"], "--llm env"),
        (["--classifier", "/nonexistent/model.json"], "模型文件"),
        (["--config", "/nonexistent/rodit.yaml"], "没有运行"),
    ],
)
def test_cli_config_errors_exit_2(argv, words, no_jev_env):
    code, out, err = _run(argv)
    assert code == 2 and out == ""
    assert "对比评测没有运行" in err and words in err


@pytest.mark.parametrize(
    "argv",
    [
        ["--providers", "foo"],
        ["--providers", ""],
        ["--jev", "file"],
        ["--llm", "offline"],
        ["--limit", "0"],
        ["--target-assist", "1.5"],
        ["--ece-tolerance", "-0.1"],
    ],
)
def test_cli_bad_arguments_exit_2(argv):
    with pytest.raises(SystemExit) as ei:
        _run(argv)
    assert ei.value.code == 2


def test_cli_jev_env_without_variables_names_them(no_jev_env):
    code, _, err = _run(["--jev", "env"])
    assert code == 2
    assert BASE_URL_ENV in err and API_KEY_ENV in err


def test_cli_llm_env_without_variables_exits_2(monkeypatch):
    for name in ("RODIT_LLM_BASE_URL", "RODIT_LLM_MODEL", "RODIT_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    code, _, err = _run(["--llm", "env", "--limit", "3"])
    assert code == 2 and "RODIT_LLM_BASE_URL" in err


def test_cli_llm_env_with_injected_client(all_cases):
    code, out, err = _run(
        ["--llm", "env", "--limit", "12", "--json"], client_factory=lambda: _llm(all_cases)
    )
    assert code == 0, err
    names = [p["info"]["name"] for p in json.loads(out)["providers"]]
    assert names == ["rules", "incloud_llm"]
    assert "[incloud_llm 12/12]" in err  # 联网的判断层逐条显示进度
    code, _, err = _run(
        ["--llm", "env", "--limit", "3"],
        client_factory=lambda: FakeLLMClient(lambda r: LLMError("大模型接口拒绝请求（HTTP 401）")),
    )
    assert code == 2 and "大模型连通性检查失败" in err


def test_cli_jev_env_with_injected_transport(all_cases, no_jev_env):
    transport = JevScript(all_cases)
    code, out, err = _run(["--jev", "env"], jev_factory=lambda: (transport, None))
    assert code == 0, err
    # 默认只有关键词规则作境内对照：条件 4 无法判定，提示加 --llm env / --classifier
    assert "  - jev：" in out and "chinese_eval_passed" in out
    assert "：无法判定" in out and "：满足" not in out
    assert "境内大模型或分类器（--llm env / --classifier）" in out
    first = transport.payloads[0]
    assert first["state"]["messages"] == [{"role": "buyer", "text": "在吗"}]  # 连通性检查
    assert first["state"]["synthetic"] is True
    assert len(transport.payloads) == 1 + 324
    # 再加上境内大模型（--llm env）：对照它判定
    code, out, err = _run(
        ["--llm", "env", "--jev", "env", "--json"],
        client_factory=lambda: _llm(all_cases),
        jev_factory=lambda: (JevScript(all_cases), None),
    )
    assert code == 0, err
    data = json.loads(out)
    assert [p["info"]["name"] for p in data["providers"]] == ["rules", "incloud_llm", "jev"]
    assert data["verdict"]["domestic"] == "incloud_llm" and data["verdict"]["status"] == "met"
    # --limit 抽样：不下结论
    code, out, err = _run(
        ["--llm", "env", "--jev", "env", "--limit", "40", "--json"],
        client_factory=lambda: _llm(all_cases),
        jev_factory=lambda: (JevScript(all_cases), None),
    )
    assert code == 0, err
    verdict = json.loads(out)["verdict"]
    assert verdict["partial"] is True and verdict["status"] == "undetermined"
    sent: list[dict[str, Any]] = []

    def broken(payload: dict[str, Any]) -> dict[str, Any]:
        sent.append(payload)
        raise JevTimeout("Jev 调用超时（10 秒）")

    code, _, err = _run(["--jev", "env"], jev_factory=lambda: (broken, None))
    assert code == 2 and "Jev 连通性检查失败" in err and "超时" in err
    assert len(sent) == 1  # 连通性检查失败就不跑整轮


def test_cli_jev_env_with_real_transport_never_leaks_the_key(all_cases, monkeypatch, tmp_path):
    """真正的 JevHttpTransport + 假 urlopen：报告里接口只留主机，任何输出里都没有密钥。"""
    monkeypatch.setenv(BASE_URL_ENV, f"{JEV_BASE}/prefix")
    monkeypatch.setenv(API_KEY_ENV, KEY)
    script = JevScript(all_cases)
    seen: list[urllib.request.Request] = []

    class Resp:
        status = 200

        def __init__(self, body: bytes) -> None:
            self.body = body

        def read(self) -> bytes:
            return self.body

        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def opener(request: urllib.request.Request, timeout: float) -> Resp:
        seen.append(request)
        body = script(json.loads(request.data))
        body["usage"] = {"input_tokens": 50, "output_tokens": 1}
        return Resp(json.dumps(body).encode())

    def factory() -> tuple[JevHttpTransport, str | None]:  # 同 transport_from_env，换掉 urlopen
        return JevHttpTransport.from_env(opener=opener), "jev-laya-multilingual"

    target = tmp_path / "out.json"
    code, out, err = _run(
        ["--jev", "env", "--json", str(target), "--limit", "30"], jev_factory=factory
    )
    assert code == 0, err
    data = json.loads(target.read_text(encoding="utf-8"))
    jev = data["providers"][1]
    assert jev["info"]["endpoint"] == JEV_BASE  # 只有协议和主机，没有路径
    assert jev["info"]["model"] == "jev-laya-multilingual"
    assert jev["info"]["timeout_s"] == 10.0
    assert jev["tokens"]["input_tokens"] == 50 * 30
    for text in (out, err, target.read_text(encoding="utf-8")):
        assert KEY not in text and "/prefix" not in text
    assert all(r.full_url == f"{JEV_BASE}/prefix/api/v1/systemone" for r in seen)
    assert all(dict(r.header_items())["Authorization"] == f"Bearer {KEY}" for r in seen)


def test_cli_classifier_artifact(tmp_path, all_cases):
    train = [(c.text, c.expected_intent) for c in CASES if c.split == "dev"]
    clf = ClassifierProvider(RuleProvider()).fit(train, seed_examples=True)
    path = tmp_path / "clf.json"
    write_artifact(path, clf.to_artifact(trained_at=DEFAULT_NOW))
    code, out, err = _run(["--classifier", str(path), "--json"])
    assert code == 0, err
    data = json.loads(out)
    names = [p["info"]["name"] for p in data["providers"]]
    assert names == ["rules", "classifier"]
    info = data["providers"][1]["info"]
    assert info["classifier_sha256"] and len(info["classifier_sha256"]) == 64
    assert data["providers"][1]["overall"]["answered"] == 324
    # 只跑分类器（显式 --providers）
    code, out, _ = _run(["--providers", "classifier", "--classifier", str(path), "--json"])
    assert [p["info"]["name"] for p in json.loads(out)["providers"]] == ["classifier"]
    # 坏的模型文件
    path.write_text("{}", encoding="utf-8")
    code, _, err = _run(["--classifier", str(path)])
    assert code == 2 and "模型文件不能用" in err


def test_cli_dev_split_with_jev_is_undetermined(all_cases, no_jev_env):
    code, out, err = _run(
        ["--split", "dev", "--limit", "40", "--json"],
        jev_factory=lambda: (JevScript(all_cases), None),
    )
    assert code == 0, err
    # 没给 --jev env：默认只有 rules（factory 不会被调用）
    assert [p["info"]["name"] for p in json.loads(out)["providers"]] == ["rules"]
    code, out, err = _run(
        ["--split", "dev", "--limit", "40", "--json", "--jev", "env"],
        jev_factory=lambda: (JevScript(all_cases), None),
    )
    assert code == 0, err
    verdict = json.loads(out)["verdict"]
    assert verdict["status"] == "undetermined" and verdict["split"] == "dev"


def test_python_dash_m_entry_point():
    proc = subprocess.run(
        [sys.executable, "-m", "rodit.eval.compare", "--limit", "5", "--json"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["n"] == 5
    proc = subprocess.run(
        [sys.executable, "-m", "rodit.eval.compare", "--providers", "jev"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 2 and "--jev env" in proc.stderr


def test_route_annotation_constants_exist():
    """代理标注依赖的路由结果名（防止枚举改名后标注悄悄全变成"说不清"）。"""
    assert RouteAction.HUMAN_PRIORITY.value == "human_priority"
    assert RouteAction.SOOTHE.value == "soothe"
