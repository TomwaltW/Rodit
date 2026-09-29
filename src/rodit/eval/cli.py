"""评测命令行（方案 §13）：python -m rodit.eval [选项]

在合成评测集（rodit.eval.dataset）上跑演示单元（rodit.eval.harness），输出：

- 问题类型准确率：总体 + 按问题类型；路由结果准确率、出推荐的比例；
- §13.2 四个维度（事实正确、切题、合规、表达）的满分率；
- §13.2 上线门槛：只看测试集（§13.1 留出的 20%）——事实正确、合规满分率都 ≥ 95%；
- §13.3 智能辅助阈值（target_precision = 0.90）：只用开发集的路由结果校准（测试集不参与调参），
  另给出可靠性分箱和期望校准误差（ECE）；
- §12.1 耗时：逐条和汇总（各环节、端到端的 P50 / P95，对照 §12.1 目标），最慢的几条；
- §14 大模型用量：逐条和汇总的 token、调用次数、失败次数，按用途（路由 / 生成 / 质检）的占比；
  给了单价时估算成本；
- 降级统计（§12.2）；失败清单：每条用例的问题、实际推荐；接大模型时还有模型草稿和质检原因；
- §13.5 回归比较（--baseline）：指标比基线下降超过容差时退出码为 1。

两种模式（--llm）：
- offline（默认）：离线替身——生成只复述依据、关键词规则路由、启发式质检，全部合成数据、不联网。
  离线替身的回答质量远不如真实大模型，这里的数字是回归基线（§13.5），不是上线判断；
- env：真实的 OpenAI 兼容大模型（rodit.eval.llm_env：RODIT_LLM_BASE_URL / RODIT_LLM_MODEL /
  RODIT_LLM_API_KEY / RODIT_LLM_TIMEOUT_S），路由和质检按 EngineConfig.decision 组装
  （默认 incloud_llm）。只在本地开发机上这样做：密钥放本地 .env，不进仓库；评测集全是合成数据。

部分评测：--intent 只跑某个问题类型，--limit N 用 --seed 随机抽 N 条（可复现）。
完整结果：--json 输出 JSON 到标准输出；--json out.json 写文件（文本报告照常输出）。
退出码：0 正常；1 回归比较有指标下降；2 配置错误（环境变量、基线文件、筛选后没有用例等）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Literal, TextIO

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rodit.app.offline import EvidenceEchoLLM
from rodit.core.config import EngineConfig, load_config
from rodit.core.enums import INTENT_LABELS, Intent, ProviderName
from rodit.core.errors import LLMError
from rodit.core.models import EngineResult
from rodit.core.ports import LLMClient
from rodit.eval.baseline import METRIC_LABELS, BaselineCheck, compare_to_baseline, load_baseline
from rodit.eval.calibration import (
    ReliabilityBin,
    Sample,
    ThresholdPick,
    calibrate,
    reliability,
)
from rodit.eval.dataset import EvalCase, cases
from rodit.eval.harness import DEMO_PLATFORM, DemoHarness, demo_harness, llm_harness
from rodit.eval.llm_env import (
    PRICE_INPUT_ENV,
    PRICE_OUTPUT_ENV,
    client_from_env,
    describe_endpoint,
    describe_model,
    preflight,
    resolve_pricing,
)
from rodit.eval.perf import (
    DAYS_PER_MONTH,
    MESSAGES_PER_SHOP_DAY,
    PURPOSE_LABELS,
    CaseUsage,
    LatencyReport,
    Pricing,
    UsageReport,
    case_usage,
    latency_report,
    usage_report,
)
from rodit.eval.runner import CaseTrace, Progress, run_eval, select_cases
from rodit.eval.scoring import DIMENSIONS, CaseScore, summarize
from rodit.testing import fixtures
from rodit.testing.fixtures import build_demo_world

Split = Literal["dev", "test", "all"]
LLMMode = Literal["offline", "env"]
Add = Callable[[str], None]  # 文本报告逐行追加

ASSIST_PRECISION = 0.90  # §13.3 智能辅助阈值
GATE = 0.95  # §13.2 上线门槛
SLOWEST_SHOWN = 5  # 文本报告里列出最慢的几条
STAGE_SHORT = {"route": "路由", "retrieve": "检索", "generate": "生成", "qa": "质检"}
OFFLINE_UNIT_LINE = (
    "单元：本地演示（离线大模型只复述依据、关键词规则路由、启发式质检；合成数据，不联网）"
)
DIMENSION_LABELS = {
    "factual": "事实正确",
    "on_topic": "切题",
    "compliance": "合规",
    "expression": "表达",
}


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


class LLMInfo(_Model):
    """这次评测用的大模型和判断层（不含密钥；接口地址只留主机）。"""

    mode: LLMMode
    model: str | None = None
    endpoint: str | None = None
    decision_provider: str  # 路由的判断层（离线模式固定为关键词规则）
    route_votes: int | None = None  # 路由自一致性投票次数（§6.4）
    qa_votes: int | None = None
    route_timeout_s: float | None = None


class CaseRecord(_Model):
    """逐条结果（--json 的完整结果、--baseline 的比较对象）。不含买家原文和回答文字。"""

    id: str
    split: str
    expected_intent: Intent
    predicted_intent: Intent | None
    intent_confidence: float | None
    intent_ok: bool
    action: str | None
    action_ok: bool
    has_suggestion: bool
    scores: dict[str, int]
    latency_ms: dict[str, int] = Field(default_factory=dict)
    degraded: list[str] = Field(default_factory=list)
    usage: CaseUsage | None = None  # 离线模式不计量


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
    # 模型草稿（含质检不通过后重写的；已脱敏、去重；只有接大模型时有）
    drafts: list[str] = Field(default_factory=list)
    qa_reasons: list[str] = Field(default_factory=list)  # 质检不通过的原因（已脱敏）
    degraded: list[str] = Field(default_factory=list)
    latency_ms: dict[str, int] = Field(default_factory=dict)


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
    intent: str | None = None  # --intent 筛选
    limit: int | None = None  # --limit 抽样
    seed: int | None = None  # 抽样用的种子（只在 limit 生效时有）
    llm: LLMInfo
    cases: list[CaseRecord] = Field(default_factory=list)
    latency: LatencyReport
    usage: UsageReport | None = None
    degraded_counts: dict[str, int] = Field(default_factory=dict)
    baseline: BaselineCheck | None = None


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


def _failure(
    score: CaseScore, case: EvalCase, result: EngineResult | None, trace: CaseTrace
) -> Failure:
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
        drafts=list(dict.fromkeys(trace.drafts)),  # 一次生成的几条候选常常相同，只列一次
        qa_reasons=list(trace.qa_reasons),
        degraded=list(trace.degraded),
        latency_ms=dict(trace.latency_ms),
    )


def _case_record(
    score: CaseScore, case: EvalCase, trace: CaseTrace, pricing: Pricing | None
) -> CaseRecord:
    return CaseRecord(
        id=score.case_id,
        split=score.split,
        expected_intent=case.expected_intent,
        predicted_intent=score.predicted_intent,
        intent_confidence=score.intent_confidence,
        intent_ok=score.intent_ok,
        action=str(score.action) if score.action is not None else None,
        action_ok=score.action_ok,
        has_suggestion=score.has_suggestion,
        scores={d: getattr(score, d) for d in DIMENSIONS},
        latency_ms=dict(trace.latency_ms),
        degraded=list(trace.degraded),
        usage=case_usage(trace.llm_calls, pricing) if trace.metered else None,
    )


def _llm_info(llm: LLMClient | None, config: EngineConfig | None) -> LLMInfo:
    if llm is None:
        return LLMInfo(
            mode="offline", model=EvidenceEchoLLM.model, decision_provider=ProviderName.RULES.value
        )
    decision = (config or EngineConfig()).decision
    return LLMInfo(
        mode="env",
        model=describe_model(llm),
        endpoint=describe_endpoint(llm),
        decision_provider=decision.provider.value,
        route_votes=decision.vote_samples,
        qa_votes=decision.qa_vote_samples,
        route_timeout_s=decision.timeout_s,
    )


def evaluate(
    split: Split = "all",
    *,
    target_precision: float = ASSIST_PRECISION,
    intent: Intent | str | None = None,
    limit: int | None = None,
    seed: int = 0,
    llm: LLMClient | None = None,
    config: EngineConfig | None = None,
    pricing: Pricing | None = None,
    progress: Progress | None = None,
) -> EvalCliReport:
    """在演示单元上跑评测集的一部分，汇总成报告。

    llm 为空用离线替身；给了就用它（真实模型或测试替身），路由和质检按 config.decision 组装。
    intent / limit / seed 见 runner.select_cases；pricing 为空时只报 token。
    """
    wanted = Intent(intent) if intent is not None else None
    selected = select_cases(cases(split), intent=wanted, limit=limit, seed=seed)
    if not selected:
        where = f"（问题类型 {wanted.value}）" if wanted is not None else ""
        raise ValueError(f"评测集 {split}{where} 为空")
    _check_cases(selected)
    if llm is None:
        make_harness = partial(demo_harness, config)
    else:
        make_harness = partial(llm_harness, llm, config)
    run = run_eval(
        selected, make_harness, gate=GATE, assist_precision=target_precision, progress=progress
    )
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
    traces = [run.traces[s.case_id] for s in scores]
    metered = any(t.metered for t in traces)
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
            _failure(s, by_id[s.case_id], run.results.get(s.case_id), run.traces[s.case_id])
            for s in report.failures
        ],
        intent=wanted.value if wanted is not None else None,
        limit=limit,
        seed=seed if limit is not None else None,
        llm=_llm_info(llm, config),
        cases=[_case_record(s, by_id[s.case_id], run.traces[s.case_id], pricing) for s in scores],
        latency=latency_report([t.latency_ms for t in traces]),
        usage=usage_report([t.llm_calls for t in traces], pricing) if metered else None,
        degraded_counts=dict(Counter(d for t in traces for d in t.degraded).most_common()),
    )


# ================================================================ 输出


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def _num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def _summary_line(name: str, s: SplitSummary) -> str:
    marks = "  ".join(f"{DIMENSION_LABELS[d]} {_pct(s.full_marks[d])}" for d in DIMENSIONS)
    return (
        f"{name:<6} n={s.n:<3} 类型准确率 {_pct(s.intent_accuracy)}  路由 {_pct(s.action_accuracy)}"
        f"  出推荐 {_pct(s.suggestion_rate)}  |  满分率：{marks}"
    )


def _unit_line(info: LLMInfo) -> str:
    if info.mode == "offline":
        return OFFLINE_UNIT_LINE
    where = f"（{info.endpoint}）" if info.endpoint else ""
    return (
        f"单元：本地演示 + 大模型 {info.model or '未知'}{where}；"
        f"路由判断层 {info.decision_provider}"
        f"（投票 {info.route_votes} 次，单次超时 {info.route_timeout_s:g} 秒），"
        f"质检投票 {info.qa_votes} 次；合成数据"
    )


def _stage_breakdown(latency: dict[str, int]) -> str:
    parts = [f"{label} {latency[k]}" for k, label in STAGE_SHORT.items() if latency.get(k)]
    return f"（{' / '.join(parts)}）" if parts else ""


def _render_latency(report: EvalCliReport, add: Add) -> None:
    lat = report.latency
    add(f"== §12.1 耗时（毫秒，{lat.n} 条；没跑的环节记 0；目标对照 P95）")
    if report.llm.mode == "offline":
        add("离线替身的耗时只说明代码路径，不代表真实大模型")
    add(f"{'环节':<8}{'n':>4}  {'P50':>7}  {'P95':>7}  {'最大':>7}  目标")
    for stage in lat.stages.values():
        if stage.target_ms is None:
            verdict = "—"
        else:
            status = (
                "" if stage.within_target is None else (" 达标" if stage.within_target else " 超标")
            )
            verdict = f"≤ {stage.target_ms}{status}"
        add(
            f"{stage.label:<8}{stage.count:>4}  {_ms(stage.p50):>7}  {_ms(stage.p95):>7}"
            f"  {_ms(stage.max):>7}  {verdict}"
        )
    slowest = sorted(
        (c for c in report.cases if c.latency_ms.get("total") is not None),
        key=lambda c: c.latency_ms["total"],
        reverse=True,
    )[:SLOWEST_SHOWN]
    if slowest:
        add(
            "最慢的几条："
            + "；".join(
                f"{c.id} {c.latency_ms['total']}{_stage_breakdown(c.latency_ms)}" for c in slowest
            )
        )


def _render_usage(report: EvalCliReport, add: Add) -> None:
    add("== §14 大模型用量（token）")
    usage = report.usage
    if usage is None:
        add("离线替身不调用大模型，不统计用量（--llm env 才有）")
        return
    errors = ""
    if usage.errors:
        kinds = "、".join(f"{k} ×{v}" for k, v in usage.error_types.items())
        errors = f"（失败 {usage.errors}：{kinds}）"
    add(
        f"调用 {usage.calls} 次{errors}；"
        f"输入 {usage.input_tokens:,}、输出 {usage.output_tokens:,}，"
        f"合计 {usage.total_tokens:,}；平均每条 {_num(usage.mean_tokens_per_case, 0)}"
    )
    for purpose, p in usage.by_purpose.items():
        share = _pct(p.token_share) if p.token_share is not None else "—"
        lat = p.call_latency_ms
        add(
            f"  {PURPOSE_LABELS.get(purpose, purpose)}：调用 {p.calls}（失败 {p.errors}），"
            f"输入 {p.input_tokens:,}、输出 {p.output_tokens:,}，占 {share}；"
            f"单次耗时 P50 {_ms(lat.p50)} / P95 {_ms(lat.p95)} 毫秒"
        )
    if usage.pricing is None:
        add(
            "成本：没有单价，只报 token（--price-input / --price-output，"
            f"或环境变量 {PRICE_INPUT_ENV} / {PRICE_OUTPUT_ENV}，每 1000 token 的价格）"
        )
    else:
        add(
            f"成本：单价 输入 {usage.pricing.input_per_1k:g} / 输出 {usage.pricing.output_per_1k:g}"
            f"（每 1000 token）→ 合计 {_num(usage.cost)}，平均每条 {_num(usage.mean_cost_per_case)}"
        )
    monthly = f"每店每月 ≈ {_num(usage.projected_monthly_tokens_per_shop, 0)} token"
    if usage.projected_monthly_cost_per_shop is not None:
        monthly += f"，成本 ≈ {_num(usage.projected_monthly_cost_per_shop, 2)}"
    add(
        f"§14 示例口径（每店每天 {MESSAGES_PER_SHOP_DAY} 条消息 × {DAYS_PER_MONTH} 天）：{monthly}"
        "（评测集的问题类型分布和真实咨询不同，只作量级参考）"
    )


def _render_failure(f: Failure, add: Add) -> None:
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
    for i, draft in enumerate(f.drafts, 1):
        add(f"    模型草稿 {i}：{draft}")
    shown = f.no_suggestion_reason if f.answer is None else None
    for reason in f.qa_reasons:
        if not shown or reason not in shown:  # "没有推荐"里已经写了的原因不重复
            add(f"    质检原因：{reason}")
    if f.degraded:
        add(f"    降级：{'、'.join(f.degraded)}")


def _render_baseline(check: BaselineCheck, add: Add) -> None:
    add(f"== §13.5 回归比较（基线 {check.source or '—'}，容差 {check.tolerance * 100:g} 个百分点）")
    add(
        f"共同用例 {check.common_cases} 条（只在本次 {check.only_current} 条、"
        f"只在基线 {check.only_baseline} 条，不比较）"
    )
    for scope in ("all", "test"):
        rows = [d for d in check.deltas if d.scope == scope]
        if not rows:
            continue
        name = "全部" if scope == "all" else "测试集"
        cells = "  ".join(
            f"{METRIC_LABELS[d.metric]} {_pct(d.baseline)} → {_pct(d.current)}"
            f"（{d.delta * 100:+.1f}）{' ↓回归' if d.regressed else ''}"
            for d in rows
        )
        add(f"{name:<6} n={rows[0].n:<3} {cells}")
    add("→ 指标没有下降" if check.passed else "→ 有指标下降超过容差：不能上线（退出码 1）")


def render_text(report: EvalCliReport) -> str:
    lines: list[str] = []
    add = lines.append
    sizes = "、".join(f"{k} {v} 条" for k, v in report.split_sizes.items() if v)
    add(f"Rodit 评测（§13）  评测集：{report.split}（{report.n} 条：{sizes}）")
    add(_unit_line(report.llm))
    picks = []
    if report.intent:
        picks.append(f"只跑问题类型 {report.intent}")
    if report.limit is not None:
        picks.append(f"随机抽 {report.limit} 条（seed {report.seed}）")
    if picks:
        add("部分评测：" + "；".join(picks))
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
    _render_latency(report, add)
    add("")
    _render_usage(report, add)
    if report.degraded_counts:
        add("")
        add(
            "== 降级（§12.2）：" + "、".join(f"{k} ×{v}" for k, v in report.degraded_counts.items())
        )
    add("")
    add(f"== 失败清单（{len(report.failures)} 条：任一维度不是满分）")
    for f in report.failures:
        _render_failure(f, add)
    if report.baseline is not None:
        add("")
        _render_baseline(report.baseline, add)
    return "\n".join(lines)


# ================================================================ 命令行


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def _tolerance(text: str) -> float:
    value = float(text)
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError("必须在 [0, 1) 之间（0.02 = 2 个百分点）")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rodit.eval",
        description=(
            "在合成评测集上跑演示单元（离线替身或真实大模型），输出 §13.2 评分、上线门槛、"
            "§13.3 阈值校准、§12.1 耗时、§14 用量，可与基线做 §13.5 回归比较"
        ),
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="完整结果（JSON）：不带 PATH 输出到标准输出；带 PATH 写文件，文本报告照常输出",
    )
    parser.add_argument(
        "--llm",
        choices=["offline", "env"],
        default="offline",
        help="offline：离线替身（默认，不联网）；env：环境变量 RODIT_LLM_* 配置的真实大模型",
    )
    parser.add_argument("--intent", choices=[i.value for i in Intent], help="只跑某个问题类型")
    parser.add_argument("--limit", type=_positive_int, help="随机抽 N 条（按 --seed，可复现）")
    parser.add_argument("--seed", type=int, default=0, help="--limit 抽样的随机种子（默认 0）")
    parser.add_argument("--config", metavar="YAML", help="引擎配置（如 config/rodit.yaml）")
    parser.add_argument(
        "--price-input", type=float, metavar="PRICE", help="输入单价：每 1000 token 的价格"
    )
    parser.add_argument(
        "--price-output", type=float, metavar="PRICE", help="输出单价：每 1000 token 的价格"
    )
    parser.add_argument("--baseline", metavar="PATH", help="基线（之前一次 --json 的输出）")
    parser.add_argument(
        "--tolerance",
        type=_tolerance,
        default=0.0,
        help="回归容差（比例，默认 0：§13.5 指标不下降）",
    )
    return parser


def _progress_printer(err: TextIO) -> Progress:
    def show(index: int, total: int, case: EvalCase, trace: CaseTrace) -> None:
        ms = trace.latency_ms.get("total")
        failed = sum(1 for c in trace.llm_calls if c.get("error"))
        calls = f"，大模型调用 {len(trace.llm_calls)} 次" + (f"（失败 {failed}）" if failed else "")
        print(f"[{index}/{total}] {case.id} {_ms(ms)} 毫秒{calls}", file=err, flush=True)

    return show


def main(
    argv: Sequence[str] | None = None,
    out: TextIO | None = None,
    *,
    err: TextIO | None = None,
    client_factory: Callable[[], LLMClient] | None = None,
) -> int:
    """命令行入口，返回退出码（见模块说明）。

    client_factory：--llm env 时构造大模型客户端（默认 llm_env.client_from_env；测试注入替身）。
    """
    args = build_parser().parse_args(argv)
    out = out or sys.stdout
    err = err or sys.stderr
    try:
        pricing = resolve_pricing(args.price_input, args.price_output)
        config = load_config(args.config) if args.config else None
        baseline = load_baseline(args.baseline) if args.baseline else None
        llm: LLMClient | None = None
        if args.llm == "env":
            llm = (client_factory or client_from_env)()
            try:
                preflight(llm)
            except LLMError as e:  # 客户端的错误信息里已经抹掉了密钥
                raise RuntimeError(f"大模型连通性检查失败：{e}") from None
        report = evaluate(
            args.split,
            intent=args.intent,
            limit=args.limit,
            seed=args.seed,
            llm=llm,
            config=config,
            pricing=pricing,
            progress=_progress_printer(err) if llm is not None else None,
        )
        if baseline is not None:
            check = compare_to_baseline(
                [c.model_dump(mode="json") for c in report.cases],
                baseline,
                tolerance=args.tolerance,
                source=args.baseline,
            )
            report = report.model_copy(update={"baseline": check})
    except (ValueError, RuntimeError, OSError, yaml.YAMLError) as e:
        print(f"评测没有运行：{e}", file=err)
        return 2
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.json == "-":
        print(payload, file=out)
    else:
        print(render_text(report), file=out)
        if args.json:
            try:
                target = Path(args.json).expanduser()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(payload + "\n", encoding="utf-8")
            except OSError as e:
                print(f"完整结果写不进 {args.json}：{type(e).__name__}", file=err)
                return 2
            print(f"\n完整结果（JSON）已写入 {args.json}", file=out)
    if report.baseline is not None and not report.baseline.passed:
        return 1
    return 0


__all__ = [
    "CaseRecord",
    "DemoHarness",
    "EvalCliReport",
    "LLMInfo",
    "build_parser",
    "demo_harness",
    "evaluate",
    "main",
    "render_text",
]
