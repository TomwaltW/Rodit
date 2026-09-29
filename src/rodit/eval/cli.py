"""评测命令行（方案 §13）：python -m rodit.eval [--split dev|test|all] [--json]

在合成评测集（rodit.eval.dataset）上跑本地演示单元（rodit.app.demo.build_demo_unit：
离线大模型只复述依据、关键词规则路由、启发式质检判断，全部合成数据、不联网），输出：

- 问题类型准确率：总体 + 按问题类型；路由结果准确率、出推荐的比例；
- §13.2 四个维度（事实正确、切题、合规、表达）的满分率；
- §13.2 上线门槛：只看测试集（§13.1 留出的 20%）——事实正确、合规满分率都 ≥ 95%；
- §13.3 智能辅助阈值（target_precision = 0.90）：只用开发集的路由结果校准（测试集不参与调参），
  另给出可靠性分箱和期望校准误差（ECE）；
- 失败清单：每条用例的问题（缺要点、缺依据、类型不符、违规……）和实际推荐。

离线替身的回答质量远不如真实大模型，这里的数字是回归基线（§13.5：换模型、改提示词、升级判断层
之前先跑，指标不下降才上线），不是上线判断。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict

from rodit.app.demo import build_demo_unit
from rodit.core.enums import INTENT_LABELS, Intent, Platform
from rodit.core.models import EngineResult
from rodit.core.ports import ConversationRepository
from rodit.eval.calibration import (
    ReliabilityBin,
    Sample,
    ThresholdPick,
    calibrate,
    reliability,
)
from rodit.eval.dataset import EvalCase, cases
from rodit.eval.runner import EngineLike, run_eval
from rodit.eval.scoring import DIMENSIONS, CaseScore, summarize
from rodit.testing import fixtures
from rodit.testing.fixtures import DemoWorld, build_demo_world

Split = Literal["dev", "test", "all"]

DEMO_PLATFORM = Platform.TAOBAO  # 评测集用店铺 A（淘宝）
ASSIST_PRECISION = 0.90  # §13.3 智能辅助阈值
GATE = 0.95  # §13.2 上线门槛
DIMENSION_LABELS = {
    "factual": "事实正确",
    "on_topic": "切题",
    "compliance": "合规",
    "expression": "表达",
}


# ================================================================ 演示单元


@dataclass
class DemoHarness:
    """runner.EvalHarness 的实现：一个全新的本地演示单元（每条用例一套，互不影响）。"""

    engine: EngineLike
    conversations: ConversationRepository
    world: DemoWorld


def demo_harness() -> DemoHarness:
    world = build_demo_world()
    unit = build_demo_unit(DEMO_PLATFORM, world=world)  # 淘宝单元：店铺 A、B
    return DemoHarness(engine=unit.engine, conversations=unit.repos.conversations, world=world)


def _check_cases(selected: Sequence[EvalCase]) -> None:
    """评测集的店铺都要在演示单元的平台上（§9.1 每个平台一个单元）。"""
    world = build_demo_world()
    for case in selected:
        shop = world.shop(getattr(fixtures, case.shop))
        if shop.platform != DEMO_PLATFORM:
            raise ValueError(
                f"用例 {case.id} 的店铺属于 {shop.platform}，演示单元是 {DEMO_PLATFORM}"
            )


# ================================================================ 报告


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SplitSummary(_Model):
    n: int
    intent_accuracy: float
    action_accuracy: float
    suggestion_rate: float
    full_marks: dict[str, float]
    mean_scores: dict[str, float]


class IntentSummary(_Model):
    label: str
    n: int
    intent_accuracy: float
    action_accuracy: float
    suggestion_rate: float
    full_marks: dict[str, float]


class GateVerdict(_Model):
    split: Literal["test"] = "test"
    n: int
    threshold: float
    factual: float
    compliance: float
    passes: bool


class Calibration(_Model):
    target_precision: float
    samples_split: str  # 校准用的是哪部分数据
    n_samples: int
    thresholds: dict[str, ThresholdPick | None]


class Reliability(_Model):
    n_samples: int
    bins: list[ReliabilityBin]
    ece: float


class Failure(_Model):
    id: str
    split: str
    expected_intent: Intent
    predicted_intent: Intent | None
    action: str | None
    scores: dict[str, int]
    problems: list[str]
    answer: str | None
    no_suggestion_reason: str | None


class EvalCliReport(_Model):
    split: Split
    n: int
    split_sizes: dict[str, int]
    overall: SplitSummary
    by_split: dict[str, SplitSummary]
    by_intent: dict[str, IntentSummary]
    gate: GateVerdict | None
    calibration: Calibration
    reliability: Reliability
    confusion: dict[str, dict[str, int]]
    failures: list[Failure]


def _rate(values: Sequence[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _split_summary(scores: Sequence[CaseScore], selected: Sequence[EvalCase]) -> SplitSummary:
    report = summarize(scores, selected, gate=GATE)
    return SplitSummary(
        n=report.n,
        intent_accuracy=round(report.intent_accuracy, 4),
        action_accuracy=round(report.action_accuracy, 4),
        suggestion_rate=round(report.suggestion_rate, 4),
        full_marks=report.full_marks,
        mean_scores=report.mean_scores,
    )


def _by_intent(scores: Sequence[CaseScore], by_id: dict[str, EvalCase]) -> dict[str, IntentSummary]:
    groups: dict[Intent, list[CaseScore]] = {}
    for s in scores:
        groups.setdefault(by_id[s.case_id].expected_intent, []).append(s)
    out: dict[str, IntentSummary] = {}
    for intent in Intent:  # 按枚举顺序，和方案 §5.1 的表一致
        group = groups.get(intent)
        if not group:
            continue
        out[intent.value] = IntentSummary(
            label=INTENT_LABELS[intent],
            n=len(group),
            intent_accuracy=_rate([s.intent_ok for s in group]),
            action_accuracy=_rate([s.action_ok for s in group]),
            suggestion_rate=_rate([s.has_suggestion for s in group]),
            full_marks={d: _rate([getattr(s, d) == 2 for s in group]) for d in DIMENSIONS},
        )
    return out


def _samples(scores: Sequence[CaseScore], by_id: dict[str, EvalCase]) -> list[Sample]:
    """(预测类型, 置信度, 真实类型)：没有路由结果的用例不算（§13.3 只校准路由）。"""
    return [
        Sample(
            predicted=str(s.predicted_intent),
            confidence=s.intent_confidence,
            true=str(by_id[s.case_id].expected_intent),
        )
        for s in scores
        if s.predicted_intent is not None and s.intent_confidence is not None
    ]


def _failure(score: CaseScore, case: EvalCase, result: EngineResult | None) -> Failure:
    cards = result.suggestions if result is not None else []
    return Failure(
        id=score.case_id,
        split=score.split,
        expected_intent=case.expected_intent,
        predicted_intent=score.predicted_intent,
        action=str(score.action) if score.action is not None else None,
        scores={d: getattr(score, d) for d in DIMENSIONS},
        problems=list(score.problems),
        answer=cards[0].text if cards else None,
        no_suggestion_reason=result.no_suggestion_reason if result is not None else None,
    )


def evaluate(split: Split = "all", *, target_precision: float = ASSIST_PRECISION) -> EvalCliReport:
    """在演示单元上跑评测集的一部分，汇总成报告。"""
    selected = cases(split)
    if not selected:
        raise ValueError(f"评测集 {split} 为空")
    _check_cases(selected)
    run = run_eval(selected, demo_harness, gate=GATE, assist_precision=target_precision)
    by_id = {c.id: c for c in selected}
    scores = run.scores

    by_split: dict[str, SplitSummary] = {}
    for name in ("dev", "test"):
        part = [s for s in scores if s.split == name]
        if part:
            by_split[name] = _split_summary(part, selected)

    test_scores = [s for s in scores if s.split == "test"]
    gate: GateVerdict | None = None
    if test_scores:
        test_report = summarize(test_scores, selected, gate=GATE)
        gate = GateVerdict(
            n=test_report.n,
            threshold=GATE,
            factual=test_report.full_marks["factual"],
            compliance=test_report.full_marks["compliance"],
            passes=test_report.passes_gate,
        )

    # §13.1：测试集不参与调参——有开发集时只用开发集校准
    dev_scores = [s for s in scores if s.split == "dev"]
    calib_scores, calib_split = (dev_scores, "dev") if dev_scores else (scores, split)
    calib_samples = _samples(calib_scores, by_id)
    intents = sorted({str(by_id[s.case_id].expected_intent) for s in calib_scores})
    thresholds = calibrate(calib_samples, intents, target_precision=target_precision)

    all_samples = _samples(scores, by_id)
    bins, ece = reliability(all_samples) if all_samples else ([], 0.0)
    report = summarize(scores, selected, gate=GATE)
    return EvalCliReport(
        split=split,
        n=len(scores),
        split_sizes={name: sum(1 for s in scores if s.split == name) for name in ("dev", "test")},
        overall=_split_summary(scores, selected),
        by_split=by_split,
        by_intent=_by_intent(scores, by_id),
        gate=gate,
        calibration=Calibration(
            target_precision=target_precision,
            samples_split=calib_split,
            n_samples=len(calib_samples),
            thresholds=thresholds,
        ),
        reliability=Reliability(n_samples=len(all_samples), bins=bins, ece=ece),
        confusion=report.confusion,
        failures=[
            _failure(s, by_id[s.case_id], run.results.get(s.case_id)) for s in report.failures
        ],
    )


# ================================================================ 输出


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _summary_line(name: str, s: SplitSummary) -> str:
    marks = "  ".join(f"{DIMENSION_LABELS[d]} {_pct(s.full_marks[d])}" for d in DIMENSIONS)
    return (
        f"{name:<6} n={s.n:<3} 类型准确率 {_pct(s.intent_accuracy)}  路由 {_pct(s.action_accuracy)}"
        f"  出推荐 {_pct(s.suggestion_rate)}  |  满分率：{marks}"
    )


def render_text(report: EvalCliReport) -> str:
    lines: list[str] = []
    add = lines.append
    sizes = "、".join(f"{k} {v} 条" for k, v in report.split_sizes.items() if v)
    add(f"Rodit 评测（§13）  评测集：{report.split}（{report.n} 条：{sizes}）")
    add("单元：本地演示（离线大模型只复述依据、关键词规则路由、启发式质检；合成数据，不联网）")
    add("")
    add("== 总体（§13.2 四个维度的满分率）")
    add(_summary_line("全部", report.overall))
    if len(report.by_split) > 1:
        for name, summary in report.by_split.items():
            add(_summary_line(name, summary))
    add("")
    add("== §13.2 上线门槛（只看测试集：事实正确、合规满分率都 ≥ 95%）")
    gate = report.gate
    if gate is None:
        add("本次没有测试集用例（--split test 或 all 才做判定）")
    else:
        verdict = "通过" if gate.passes else "未通过"
        add(
            f"测试集 n={gate.n}：事实正确 {_pct(gate.factual)}，合规 {_pct(gate.compliance)}"
            f" → {verdict}"
        )
    add("")
    add("== 按问题类型")
    add(f"{'类型':<18}{'n':>3}  {'类型准确率':>8}  {'出推荐':>6}  {'事实':>6}  {'切题':>6}")
    for intent, s in report.by_intent.items():
        add(
            f"{intent:<18}{s.n:>3}  {_pct(s.intent_accuracy):>8}  {_pct(s.suggestion_rate):>6}"
            f"  {_pct(s.full_marks['factual']):>6}  {_pct(s.full_marks['on_topic']):>6}"
        )
    wrong = [
        f"{true} → {pred} ×{n}"
        for true, preds in sorted(report.confusion.items())
        for pred, n in sorted(preds.items())
        if pred != true
    ]
    if wrong:
        add("误判（真实 → 预测，未计入可接受的备选类型）：" + "；".join(wrong))
    add("")
    cal = report.calibration
    add(
        f"== §13.3 智能辅助阈值（target_precision = {cal.target_precision:.2f}，"
        f"用 {cal.samples_split} 的 {cal.n_samples} 条路由结果校准）"
    )
    if cal.samples_split == "test":
        add("注意：只跑了测试集，阈值是用测试集算的，仅供参考；正式阈值用开发集校准（§13.1）")
    for intent, pick in cal.thresholds.items():
        if pick is None:
            add(f"{intent:<18}达不到目标准确率")
        else:
            add(
                f"{intent:<18}阈值 {pick.threshold:.2f}  准确率 {_pct(pick.precision)}"
                f"  覆盖率 {_pct(pick.coverage)}  召回 {_pct(pick.recall)}  (n={pick.support})"
            )
    rel = report.reliability
    add(f"可靠性（{rel.n_samples} 条）：ECE = {rel.ece:.4f}")
    for b in rel.bins:
        add(
            f"  [{b.lower:.1f}, {b.upper:.1f}{']' if b.upper == 1.0 else ')'}  n={b.count:<3}"
            f" 平均置信度 {b.mean_confidence:.2f}  实际准确率 {_pct(b.accuracy)}"
        )
    add("")
    add(f"== 失败清单（{len(report.failures)} 条：任一维度不是满分）")
    for f in report.failures:
        dims = " ".join(f"{DIMENSION_LABELS[d]}{f.scores[d]}" for d in DIMENSIONS)
        predicted = f.predicted_intent.value if f.predicted_intent else "—"
        add(
            f"- {f.id}（{f.split}）期望 {f.expected_intent.value}，实际 {predicted} / {f.action}"
            f"  [{dims}]"
        )
        for p in f.problems:
            add(f"    问题：{p}")
        if f.answer is not None:
            add(f"    推荐：{f.answer}")
        elif f.no_suggestion_reason:
            add(f"    没有推荐：{f.no_suggestion_reason}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rodit.eval",
        description="在合成评测集上跑本地演示单元，输出 §13.2 评分、上线门槛和 §13.3 阈值校准",
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--json", action="store_true", help="输出 JSON（便于回归比较）")
    args = parser.parse_args(argv)
    out = out or sys.stdout
    report = evaluate(args.split)
    if args.json:
        payload: dict[str, Any] = report.model_dump(mode="json")
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=out)
    else:
        print(render_text(report), file=out)
    return 0


__all__ = ["DemoHarness", "EvalCliReport", "demo_harness", "evaluate", "main", "render_text"]
