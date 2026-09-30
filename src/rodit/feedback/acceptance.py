"""Phase 1 验收报告（方案 §15 Phase 1 验收标准："满足 13.2 的门槛；采纳率 ≥ 60%；推荐 P95 ≤ 5 秒；
平台违规为 0"，§1.3 目标，§13.4、§13.5）。纯函数，不读写仓储：数据由 python -m rodit.ops
acceptance-report 收集（rodit.ops.jobs.rollout）。

每家店一份、所有店合在一起（pooled）一份；是否通过看合在一起的那份（§15 是整个阶段的验收），
每家店的结果供参考（量小的店单独看容易数据不足）。条件：

- test_set：§13.2 测试集门槛。整个单元一份：和线上配置指纹一致、用真实大模型跑的评测结论
  （rodit.eval.fingerprint.test_set_evidence），只放在合在一起的那份里；
- adoption：截至 until 的最近 2 个完整 ISO 周，每周按推荐组算的采纳率都 ≥ 60%（§13.4 连续 2 周），
  每周至少 ASSIST_MIN_GROUPS 个推荐组；全自动发出去的不算（feedback.metrics）。值是较低的那一周；
- latency_p95：窗口 [since, until) 里出了推荐（或超过时间预算）的买家消息，推荐出现的耗时
  （engine.result 的 latency_ms.since_received——含排队和会话锁等待；没有时用 total）的
  P95 ≤ 5000 毫秒（§1.3、§12.1）；
- platform_violations：复核人员按周登记的平台违规（shop.rollout_incident），窗口里每个完整周都要
  登记，合计为 0；
- compliance_scan：窗口里实际发出去的文字的合规扫描（feedback.compliance，代理指标），疑似违规为 0；
- weekly_review：窗口里每周人工复核（§13.5）已打分的样本合在一起，事实正确、合规满分率都 ≥ 95%
  （§13.2），至少 DEFAULT_MIN_SCORED 条。

每个条件是 passed / failed / unchecked（缺数据，detail 写明缺什么）；unchecked 也不算通过。
报告里只有店铺 ID、计数、比例和周期，没有任何文字。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.core.audit import LATENCY_SINCE_RECEIVED, RESULT_BUDGET_EXCEEDED, AuditAction, ResultKey
from rodit.core.clock import CST
from rodit.core.models import AuditEvent, QualityReview, SuggestionRecord
from rodit.feedback.compliance import COMPLIANCE_RULES, ComplianceSummary
from rodit.feedback.metrics import E2E_P95_TARGET_MS, percentile
from rodit.feedback.review import (
    DEFAULT_MIN_SCORED,
    LAUNCH_MIN_FULL_RATE,
    GateStatus,
    launch_gate,
)
from rodit.feedback.rollout import (
    ASSIST_MIN_ADOPTION,
    ASSIST_MIN_GROUPS,
    ASSIST_PERIODS,
    IncidentTotal,
    TestSetEvidence,
    week_windows,
    weekly_adoption,
)

LATENCY_TOTAL = "total"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CriterionStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNCHECKED = "unchecked"  # 缺数据，没法判断（不算通过）


CRITERION_LABELS: dict[str, str] = {
    "test_set": "§13.2 测试集门槛（事实正确、合规满分率 ≥ 95%）",
    "adoption": "采纳率 ≥ 60%（最近 2 个完整周，按推荐组）",
    "latency_p95": "推荐出现 P95 ≤ 5 秒",
    "platform_violations": "平台违规为 0（复核人员按周登记）",
    "compliance_scan": "发出去的文字没有疑似违规（合规扫描，代理指标）",
    "weekly_review": "每周复核事实正确、合规满分率 ≥ 95%（§13.5）",
}
"""条件名 → 中文说明，也是报告里条件的顺序。"""

STATUS_LABELS: dict[CriterionStatus, str] = {
    CriterionStatus.PASSED: "通过",
    CriterionStatus.FAILED: "没通过",
    CriterionStatus.UNCHECKED: "缺数据",
}


class Criterion(_Model):
    name: str
    label: str
    status: CriterionStatus
    value: float | None = None
    threshold: float | None = None
    detail: str = ""


def _criterion(
    name: str,
    status: CriterionStatus,
    *,
    value: float | None = None,
    threshold: float | None = None,
    detail: str = "",
) -> Criterion:
    return Criterion(
        name=name,
        label=CRITERION_LABELS[name],
        status=status,
        value=value,
        threshold=threshold,
        detail=detail,
    )


class ShopAcceptanceInput(_Model):
    """一家店的验收数据（rodit.ops.jobs.rollout 从仓储里收集）。"""

    shop_id: str
    stage: str | None = None  # 当前上线阶段（feedback.rollout.RolloutStage 的值），只作说明
    records: list[SuggestionRecord] = Field(default_factory=list)  # 覆盖采纳率的 2 周
    latencies_ms: list[float] = Field(default_factory=list)  # message_latencies
    violations: IncidentTotal
    compliance: ComplianceSummary
    reviews: list[QualityReview] = Field(default_factory=list)


class ShopAcceptance(_Model):
    shop_id: str
    stage: str | None = None
    passed: bool
    criteria: list[Criterion]


class AcceptanceReport(_Model):
    as_of: AwareDatetime
    since: AwareDatetime
    until: AwareDatetime
    adoption_weeks: list[str]
    running_fingerprint: str | None = None  # 线上配置的指纹（测试集依据要和它一致）
    test_set: TestSetEvidence
    pooled: list[Criterion]
    shops: list[ShopAcceptance] = Field(default_factory=list)
    passed: bool
    failing: list[str] = Field(default_factory=list)  # 合在一起没通过的条件
    unchecked: list[str] = Field(default_factory=list)  # 合在一起缺数据的条件


# ================================================================ 数据整理


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def message_latencies(
    events: Iterable[AuditEvent], *, since: datetime, until: datetime
) -> list[float]:
    """engine.result 事件 → 每条买家消息的推荐出现耗时（毫秒）。只算出了推荐或超过时间预算的消息；
    优先 latency_ms.since_received（从平台收到消息算起），没有时用 total。"""
    out: list[float] = []
    for e in events:
        if e.action != AuditAction.RESULT.value or not since <= e.created_at < until:
            continue
        d = e.detail
        suggestions = _number(d.get(ResultKey.SUGGESTIONS.value)) or 0
        if suggestions <= 0 and d.get(RESULT_BUDGET_EXCEEDED) is not True:
            continue
        latency = d.get(ResultKey.LATENCY.value)
        if isinstance(latency, Mapping):
            value = _number(latency.get(LATENCY_SINCE_RECEIVED))
            if value is None:
                value = _number(latency.get(LATENCY_TOTAL))
        else:
            value = _number(latency)
        if value is not None:
            out.append(value)
    return out


# ================================================================ 各条件


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def adoption_criterion(records: Sequence[SuggestionRecord], until: datetime) -> Criterion:
    weekly = weekly_adoption(records, until, weeks=ASSIST_PERIODS)
    parts: list[str] = []
    low: list[float] = []
    thin: list[str] = []
    rates: list[float] = []
    for (period, _, _), report in zip(week_windows(until, ASSIST_PERIODS), weekly, strict=True):
        o = report.overall
        parts.append(f"{period} {_rate(o.group_adoption_rate)}（推荐组 {o.groups}）")
        if o.groups < ASSIST_MIN_GROUPS or o.group_adoption_rate is None:
            thin.append(period)
            continue
        rates.append(o.group_adoption_rate)
        if o.group_adoption_rate < ASSIST_MIN_ADOPTION:
            low.append(o.group_adoption_rate)
    value = round(min(rates), 4) if rates else None
    detail = "；".join(parts)
    if low:
        status = CriterionStatus.FAILED
    elif thin:
        status = CriterionStatus.UNCHECKED
        detail += f"；{'、'.join(thin)} 推荐组不到 {ASSIST_MIN_GROUPS} 个，数据不足"
    else:
        status = CriterionStatus.PASSED
    return _criterion("adoption", status, value=value, threshold=ASSIST_MIN_ADOPTION, detail=detail)


def latency_criterion(latencies: Sequence[float]) -> Criterion:
    p95 = percentile(latencies, 0.95)
    if p95 is None:
        return _criterion(
            "latency_p95",
            CriterionStatus.UNCHECKED,
            threshold=E2E_P95_TARGET_MS,
            detail="窗口里没有出推荐的消息",
        )
    status = CriterionStatus.PASSED if p95 <= E2E_P95_TARGET_MS else CriterionStatus.FAILED
    return _criterion(
        "latency_p95",
        status,
        value=p95,
        threshold=E2E_P95_TARGET_MS,
        detail=f"{len(latencies)} 条消息，P95 {p95:.0f} 毫秒",
    )


def violations_criterion(total: IncidentTotal) -> Criterion:
    by_week = "、".join(f"{w} {n}" for w, n in total.by_week.items()) or "没有登记"
    if total.count is None:
        return _criterion(
            "platform_violations",
            CriterionStatus.UNCHECKED,
            threshold=0,
            detail=total.reason or "缺登记",
        )
    status = CriterionStatus.PASSED if total.count == 0 else CriterionStatus.FAILED
    return _criterion(
        "platform_violations",
        status,
        value=total.count,
        threshold=0,
        detail=f"登记的平台违规：{by_week}",
    )


def scan_criterion(summary: ComplianceSummary) -> Criterion:
    rules = "、".join(f"{r} {n}" for r, n in summary.rule_counts.items()) or "没有命中"
    status = CriterionStatus.PASSED if summary.flagged == 0 else CriterionStatus.FAILED
    return _criterion(
        "compliance_scan",
        status,
        value=summary.flagged,
        threshold=0,
        detail=(
            f"扫描编辑后发送的推荐 {summary.scanned_suggestions} 条、发出的消息 "
            f"{summary.scanned_messages} 条，疑似违规 {summary.flagged} 条（{rules}）"
        ),
    )


def review_criterion(reviews: Sequence[QualityReview]) -> Criterion:
    gate = launch_gate(reviews, min_scored=DEFAULT_MIN_SCORED, min_full_rate=LAUNCH_MIN_FULL_RATE)
    rates = [r for r in (gate.factual_full_rate, gate.compliance_full_rate) if r is not None]
    status = {
        GateStatus.PASSED: CriterionStatus.PASSED,
        GateStatus.FAILED: CriterionStatus.FAILED,
        GateStatus.INSUFFICIENT: CriterionStatus.UNCHECKED,
    }[gate.status]
    return _criterion(
        "weekly_review",
        status,
        value=round(min(rates), 4) if rates else None,
        threshold=LAUNCH_MIN_FULL_RATE,
        detail=(
            f"已打分 {gate.scored} 条：事实正确 {_rate(gate.factual_full_rate)}、"
            f"合规 {_rate(gate.compliance_full_rate)}；{gate.detail}"
        ),
    )


def test_set_criterion(evidence: TestSetEvidence) -> Criterion:
    if evidence.passed is None:
        return _criterion(
            "test_set",
            CriterionStatus.UNCHECKED,
            threshold=LAUNCH_MIN_FULL_RATE,
            detail=evidence.reason or "没有评测结论",
        )
    return _criterion(
        "test_set",
        CriterionStatus.PASSED if evidence.passed else CriterionStatus.FAILED,
        threshold=LAUNCH_MIN_FULL_RATE,
        detail=evidence.detail,
    )


test_set_criterion.__test__ = False  # type: ignore[attr-defined]  # 名字以 test_ 开头，不是用例


# ================================================================ 合在一起


def pool_incidents(totals: Sequence[tuple[str, IncidentTotal]]) -> IncidentTotal | None:
    """各店的登记合计：有大于 0 的 → 合计；都登记全了且为 0 → 0；否则 None（reason 列出缺的店）。"""
    if not totals:
        return None
    first = totals[0][1]
    known = sum(t.known for _, t in totals)
    missing = [(shop, t) for shop, t in totals if t.count is None]
    by_week: dict[str, int] = {}
    for _, t in totals:
        for week, n in t.by_week.items():
            by_week[week] = by_week.get(week, 0) + n
    count: int | None
    reason: str | None = None
    if known > 0:
        count = known
    elif not missing:
        count = 0
    else:
        count = None
        reason = "；".join(f"{shop}：{t.reason}" for shop, t in missing)
    return IncidentTotal(
        kind=first.kind,
        weeks=list(first.weeks),
        by_week=by_week,
        missing=sorted({w for _, t in missing for w in t.missing}),
        known=known,
        count=count,
        reason=reason,
    )


def pool_scans(summaries: Sequence[ComplianceSummary]) -> ComplianceSummary | None:
    if not summaries:
        return None
    counts: dict[str, int] = {}
    by_period: dict[str, int] = {}
    for s in summaries:
        for rule, n in s.rule_counts.items():
            counts[rule] = counts.get(rule, 0) + n
        for period, n in s.by_period.items():
            by_period[period] = by_period.get(period, 0) + n
    return ComplianceSummary(
        since=min(s.since for s in summaries),
        until=max(s.until for s in summaries),
        scanned_suggestions=sum(s.scanned_suggestions for s in summaries),
        scanned_messages=sum(s.scanned_messages for s in summaries),
        flagged=sum(s.flagged for s in summaries),
        rule_counts={r: counts[r] for r in COMPLIANCE_RULES if counts.get(r)},
        by_period=dict(sorted(by_period.items())),
    )


def _shop_criteria(item: ShopAcceptanceInput, until: datetime) -> list[Criterion]:
    return [
        adoption_criterion(item.records, until),
        latency_criterion(item.latencies_ms),
        violations_criterion(item.violations),
        scan_criterion(item.compliance),
        review_criterion(item.reviews),
    ]


def _no_shops(name: str) -> Criterion:
    return _criterion(name, CriterionStatus.UNCHECKED, detail="没有店铺")


def acceptance_report(
    inputs: Sequence[ShopAcceptanceInput],
    *,
    test_set: TestSetEvidence,
    since: datetime,
    until: datetime,
    now: datetime,
    running_fingerprint: str | None = None,
) -> AcceptanceReport:
    """每家店的条件 + 合在一起的条件（见模块说明）。passed 看合在一起的那份。"""
    if since >= until:
        raise ValueError("since 要早于 until")
    shops = []
    for item in inputs:
        criteria = _shop_criteria(item, until)
        shops.append(
            ShopAcceptance(
                shop_id=item.shop_id,
                stage=item.stage,
                passed=all(c.status == CriterionStatus.PASSED for c in criteria),
                criteria=criteria,
            )
        )
    violations = pool_incidents([(i.shop_id, i.violations) for i in inputs])
    scans = pool_scans([i.compliance for i in inputs])
    pooled = [
        test_set_criterion(test_set),
        adoption_criterion([r for i in inputs for r in i.records], until),
        latency_criterion([v for i in inputs for v in i.latencies_ms]),
        violations_criterion(violations) if violations else _no_shops("platform_violations"),
        scan_criterion(scans) if scans else _no_shops("compliance_scan"),
        review_criterion([q for i in inputs for q in i.reviews]),
    ]
    if not inputs:
        pooled = [
            c if c.name == "test_set" else _no_shops(c.name) for c in pooled
        ]  # 没有店铺：除测试集外都没法判断
    failing = [c.name for c in pooled if c.status == CriterionStatus.FAILED]
    unchecked = [c.name for c in pooled if c.status == CriterionStatus.UNCHECKED]
    return AcceptanceReport(
        as_of=now,
        since=since,
        until=until,
        adoption_weeks=[p for p, _, _ in week_windows(until, ASSIST_PERIODS)],
        running_fingerprint=running_fingerprint,
        test_set=test_set,
        pooled=pooled,
        shops=shops,
        passed=not failing and not unchecked,
        failing=failing,
        unchecked=unchecked,
    )


# ================================================================ 文本报告


def _value(c: Criterion) -> str:
    if c.value is None:
        return "—"
    if c.name in ("adoption", "weekly_review"):
        return f"{c.value:.2f}"
    return f"{c.value:.0f}"


def _line(c: Criterion) -> str:
    threshold = "" if c.threshold is None else f"（门槛 {c.threshold:g}）"
    value = "" if c.value is None else f" {c.name}={_value(c)}{threshold}"
    return f"  [{STATUS_LABELS[c.status]}] {c.label}：{value.strip() or '—'}；{c.detail}"


def _when(at: datetime) -> str:
    return f"{at.astimezone(CST):%Y-%m-%d %H:%M}"


def render_text(report: AcceptanceReport) -> str:
    lines = [
        "Rodit Phase 1 验收报告（§15）",
        f"窗口 {_when(report.since)} — {_when(report.until)}（东八区）；"
        f"采纳率看 {'、'.join(report.adoption_weeks)}；生成于 {_when(report.as_of)}",
    ]
    if report.running_fingerprint:
        lines.append(f"线上配置指纹 {report.running_fingerprint[:12]}")
    lines.append("")
    verdict = "通过" if report.passed else "没通过"
    lines.append(f"== 合在一起（{len(report.shops)} 家店铺）：{verdict}")
    lines.extend(_line(c) for c in report.pooled)
    for shop in report.shops:
        lines.append("")
        stage = f"，{shop.stage}" if shop.stage else ""
        lines.append(
            f"== 店铺 {shop.shop_id}{stage}：{'通过' if shop.passed else '没通过'}（供参考）"
        )
        lines.extend(_line(c) for c in shop.criteria)
    lines.append("")
    if report.passed:
        lines.append("结论：通过（所有条件都达到）")
    else:
        by_name = {c.name: c for c in report.pooled}
        parts = []
        if report.failing:
            parts.append(
                "没达到：" + "、".join(f"{n}={_value(by_name[n])}" for n in report.failing)
            )
        if report.unchecked:
            parts.append("缺数据：" + "、".join(report.unchecked))
        lines.append("结论：没通过。" + "；".join(parts))
    return "\n".join(lines)


__all__ = [
    "CRITERION_LABELS",
    "AcceptanceReport",
    "Criterion",
    "CriterionStatus",
    "ShopAcceptance",
    "ShopAcceptanceInput",
    "acceptance_report",
    "adoption_criterion",
    "latency_criterion",
    "message_latencies",
    "pool_incidents",
    "pool_scans",
    "render_text",
    "review_criterion",
    "scan_criterion",
    "test_set_criterion",
    "violations_criterion",
]
