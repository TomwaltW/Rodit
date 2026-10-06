"""平台客服考核指标的接入前后对比（方案 §1.3 目标"平台客服指标较接入前改善"、§2.1 客服考核、
§12.4 "平台客服指标接入前后对比"、§15 Phase 2 "商家报表（平台指标接入前后对比）"）。

纯函数，只依赖 rodit.core；数据来自 core.ports.PlatformMetricsRepository（平台同步、商家登记），
接入日期由 rodit.app.platform_metrics 按 resolve_go_live 的顺序确定。

口径（报表里的每个数都能按这里复算）：

- 每个（指标、日期）只取一条：平台 > 商家登记 > 系统估算（SOURCE_PRECEDENCE）。
- 接入日当天算接入后。接入前 = [接入日 − window_days, 接入日 − 1]；接入后 = [接入日,
  min(接入日 + window_days − 1, 今天)]（默认各 28 天，window_days 在 7–90 之间）。
- 窗口均值：窗口里每天一个数；这些天都有样本量（sample_size，会话 / 售后单数）且合计大于 0 时
  按样本量加权，否则算术平均。窗口的样本量合计只在每天都有样本量时给出。
- 变化 delta = 接入后 − 接入前；相对变化 = delta / 接入前（接入前为 0 时为空）。
- 方向按指标的 higher_is_better（core.platform_metrics）：|相对变化| < 1% 算持平（接入前为 0 时
  只有 delta 为 0 才算持平）；前后任一窗口没有数据 = 数据不足。
- 可靠性提示（不改结论，只提示）：任一窗口有数据的天数 < 7、样本量合计已知且 < 100、前后用的
  数据来源不同（如接入前是商家登记、接入后是平台数据）、用了系统估算的数。
- 周序列（给图表用）：以接入日为界按 7 天一组，接入前的组不跨过接入日；每组一个均值（同上加权）。
- 总体结论（§1.3 "较接入前改善"）只看前后都有数据的指标：至少一项变好、没有变差 = improved；
  至少一项变差、没有变好 = worse；两样都有 = mixed；全部持平 = unchanged；没有前后都有数据的
  指标 = insufficient；接入日期未知 = no_go_live。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import MetricSource, Platform, PlatformMetric
from rodit.core.models import AuditEvent, PlatformMetricSample
from rodit.core.platform_metrics import PLATFORM_METRIC_SPECS, MetricUnit, metrics_for
from rodit.feedback.rollout import RolloutStage

# ================================================================ 常量

DEFAULT_WINDOW_DAYS = 28
"""接入前、接入后各看多少天（§12.4；四周正好盖住按周起伏）。"""
MIN_WINDOW_DAYS = 7
MAX_WINDOW_DAYS = 90
MIN_RELIABLE_DAYS = 7
"""窗口里有数据的天数少于这个数时提示不可靠。"""
MIN_RELIABLE_SAMPLES = 100
"""窗口的样本量（会话 / 售后单）合计已知且少于这个数时提示不可靠。"""
UNCHANGED_RELATIVE = 0.01
"""|相对变化| 小于这个比例算持平。"""
MAX_HISTORY_DAYS = 400
"""商家登记、平台同步、查询最多回溯多少天（够一年的同比，再多就是陈年数据）。"""
MAX_SAMPLE_SIZE = 10_000_000
"""一天一项指标的样本量上限（再大就是录错了）。"""

VALUE_LIMITS: dict[MetricUnit, float] = {
    "ratio": 1.0,  # 比例 0–1
    "seconds": 86400.0,  # 平均响应时长不会超过一天
    "hours": 720.0,  # 售后平均审核时长不会超过 30 天
}
"""各单位的取值上限（下限都是 0）。"""

UNIT_LABELS: dict[MetricUnit, str] = {"ratio": "比例", "seconds": "秒", "hours": "小时"}

SOURCE_PRECEDENCE: tuple[MetricSource, ...] = (
    MetricSource.PLATFORM,
    MetricSource.MERCHANT,
    MetricSource.COMPUTED,
)
"""同一天同一指标有多条时取哪条：越靠前越优先。"""

SOURCE_LABELS: dict[MetricSource, str] = {
    MetricSource.PLATFORM: "平台数据",
    MetricSource.MERCHANT: "商家登记",
    MetricSource.COMPUTED: "系统估算",
}

Direction = Literal["improved", "worse", "unchanged", "insufficient"]
DIRECTION_LABELS: dict[str, str] = {
    "improved": "变好",
    "worse": "变差",
    "unchanged": "持平",
    "insufficient": "数据不足",
}

Verdict = Literal["improved", "mixed", "worse", "unchanged", "insufficient", "no_go_live"]
VERDICT_LABELS: dict[str, str] = {
    "improved": "较接入前改善",
    "mixed": "有升有降",
    "worse": "较接入前变差",
    "unchanged": "和接入前持平",
    "insufficient": "数据不足，还不能判断",
    "no_go_live": "接入日期未知，请填写接入日期",
}

ReliabilityFlag = Literal[
    "no_before",
    "no_after",
    "few_days_before",
    "few_days_after",
    "small_sample_before",
    "small_sample_after",
    "sources_differ",
    "computed",
]
FLAG_LABELS: dict[str, str] = {
    "no_before": "接入前没有数据",
    "no_after": "接入后没有数据",
    "few_days_before": f"接入前有数据的天数少于 {MIN_RELIABLE_DAYS} 天",
    "few_days_after": f"接入后有数据的天数少于 {MIN_RELIABLE_DAYS} 天",
    "small_sample_before": f"接入前样本量合计少于 {MIN_RELIABLE_SAMPLES}",
    "small_sample_after": f"接入后样本量合计少于 {MIN_RELIABLE_SAMPLES}",
    "sources_differ": "接入前后的数据来源不同，口径可能不一致",
    "computed": "用了系统估算的数，口径近似，只作参考",
}

GoLiveSource = Literal["query", "rollout", "authorization", "unknown"]
GO_LIVE_LABELS: dict[str, str] = {
    "query": "按填写的接入日期",
    "rollout": "按第一次切到智能辅助的日期（上线记录，§13.4）",
    "authorization": "按店铺开通授权的日期",
    "unknown": "接入日期未知：没有上线记录，授权也没有开通时间，请填写接入日期",
}

ROLLOUT_STAGE_ACTION = AuditAction.ROLLOUT_STAGE.value
"""rodit.app.rollout 切换上线阶段时写的审计事件；detail 的 to 是目标阶段。"""
ASSIST_STAGE = RolloutStage.ASSIST.value


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ================================================================ 校验


def check_value(metric: PlatformMetric, value: float) -> float:
    """一项指标的取值：有限数，在 [0, 单位上限] 之内（比例 0–1，时长有合理上限）。"""
    spec = PLATFORM_METRIC_SPECS[PlatformMetric(metric)]
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{spec.label}的数值不合法") from None
    if not math.isfinite(number):
        raise ValueError(f"{spec.label}的数值必须是有限数")
    limit = VALUE_LIMITS[spec.unit]
    if not 0 <= number <= limit:
        if spec.unit == "ratio":
            raise ValueError(f"{spec.label}是 0–1 之间的比例（如 0.91 表示 91%）")
        raise ValueError(f"{spec.label}要在 0–{limit:g} {UNIT_LABELS[spec.unit]}之间")
    return number


def check_sample_size(sample_size: int | None) -> int | None:
    if sample_size is None:
        return None
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise ValueError("样本量必须是整数")
    if not 0 <= sample_size <= MAX_SAMPLE_SIZE:
        raise ValueError(f"样本量要在 0–{MAX_SAMPLE_SIZE} 之间")
    return sample_size


def check_metric(platform: Platform, metric: PlatformMetric | str) -> PlatformMetric:
    """指标必须是本平台考核的（§9.1 每个平台单元只看本平台的指标）。"""
    try:
        m = PlatformMetric(metric)
    except ValueError:
        raise ValueError("未知的平台指标") from None
    if m not in metrics_for(platform):
        raise ValueError(f"{PLATFORM_METRIC_SPECS[m].label}不是本平台考核的指标")
    return m


def check_window_days(window_days: int) -> int:
    if isinstance(window_days, bool) or not isinstance(window_days, int):
        raise ValueError("对比天数必须是整数")
    if not MIN_WINDOW_DAYS <= window_days <= MAX_WINDOW_DAYS:
        raise ValueError(f"对比天数要在 {MIN_WINDOW_DAYS}–{MAX_WINDOW_DAYS} 天之间")
    return window_days


def check_day(day: date, today: date, what: str = "日期") -> date:
    """登记 / 查询的日期：不能晚于今天，最多回溯 MAX_HISTORY_DAYS 天。"""
    if day > today:
        raise ValueError(f"{what}不能晚于今天")
    if day < today - timedelta(days=MAX_HISTORY_DAYS):
        raise ValueError(f"{what}最多回溯 {MAX_HISTORY_DAYS} 天")
    return day


# ================================================================ 接入日期


class GoLive(_Model):
    """报表用的接入日期和它的来源。day 为空表示不知道（报表请商家填写）。"""

    day: date | None
    source: GoLiveSource
    label: str


def first_assist_day(events: Iterable[AuditEvent]) -> date | None:
    """上线记录（shop.rollout_stage）里第一次切到智能辅助的日期（东八区）。"""
    times = [
        e.created_at
        for e in events
        if e.action == ROLLOUT_STAGE_ACTION and e.detail.get("to") == ASSIST_STAGE
    ]
    return min(times).astimezone(CST).date() if times else None


def resolve_go_live(
    explicit: date | None,
    *,
    stage_events: Iterable[AuditEvent] = (),
    granted_at: datetime | None = None,
) -> GoLive:
    """接入日期：填写的 > 第一次切到智能辅助（上线记录）> 店铺授权的开通时间 > 未知。"""
    if explicit is not None:
        return GoLive(day=explicit, source="query", label=GO_LIVE_LABELS["query"])
    rollout = first_assist_day(stage_events)
    if rollout is not None:
        return GoLive(day=rollout, source="rollout", label=GO_LIVE_LABELS["rollout"])
    if granted_at is not None:
        day = granted_at.astimezone(CST).date()
        return GoLive(day=day, source="authorization", label=GO_LIVE_LABELS["authorization"])
    return GoLive(day=None, source="unknown", label=GO_LIVE_LABELS["unknown"])


# ================================================================ 取数


def pick_daily(
    samples: Iterable[PlatformMetricSample],
) -> dict[tuple[PlatformMetric, date], PlatformMetricSample]:
    """每个（指标、日期）按来源优先级取一条（平台 > 商家登记 > 系统估算）；数值不是有限数的跳过。"""
    rank = {s: i for i, s in enumerate(SOURCE_PRECEDENCE)}
    out: dict[tuple[PlatformMetric, date], PlatformMetricSample] = {}
    for s in samples:
        if not math.isfinite(s.value):
            continue
        key = (s.metric, s.day)
        current = out.get(key)
        if current is None or rank[s.source] < rank[current.source]:
            out[key] = s
    return out


def _mean(rows: Sequence[PlatformMetricSample]) -> tuple[float | None, bool, int | None]:
    """(均值, 是否按样本量加权, 样本量合计)。见模块说明的口径。"""
    if not rows:
        return None, False, None
    sizes = [r.sample_size for r in rows]
    total = sum(s for s in sizes if s is not None) if all(s is not None for s in sizes) else None
    if total:
        value = sum(r.value * (r.sample_size or 0) for r in rows) / total
        return round(value, 6), True, total
    return round(sum(r.value for r in rows) / len(rows), 6), False, total


class WindowStats(_Model):
    """一个窗口（接入前 / 接入后）里一项指标的汇总。"""

    start: date
    end: date
    span_days: int  # 窗口的天数（接入后截到今天）
    days: int = 0  # 有数据的天数
    mean: float | None = None
    weighted: bool = False  # 按样本量加权
    sample_size: int | None = None  # 样本量合计（每天都有样本量时才有）
    sources: dict[MetricSource, int] = Field(default_factory=dict)  # 来源 → 天数


def window_stats(
    daily: Mapping[tuple[PlatformMetric, date], PlatformMetricSample],
    metric: PlatformMetric,
    start: date,
    end: date,
) -> WindowStats:
    span = max(0, (end - start).days + 1)
    days = (start + timedelta(days=i) for i in range(span))
    rows = [daily[(metric, d)] for d in days if (metric, d) in daily]
    mean, weighted, total = _mean(rows)
    sources: dict[MetricSource, int] = {}
    for r in rows:
        sources[r.source] = sources.get(r.source, 0) + 1
    return WindowStats(
        start=start,
        end=end,
        span_days=span,
        days=len(rows),
        mean=mean,
        weighted=weighted,
        sample_size=total,
        sources={s: sources[s] for s in SOURCE_PRECEDENCE if s in sources},
    )


class WeekPoint(_Model):
    """周序列的一点。index 是相对接入日的周序号：-1 是接入前最后一周，0 是接入后第一周。"""

    index: int
    phase: Literal["before", "after"]
    start: date
    end: date
    days: int
    mean: float | None


def weekly_series(
    daily: Mapping[tuple[PlatformMetric, date], PlatformMetricSample],
    metric: PlatformMetric,
    go_live: date,
    window_days: int,
    today: date,
) -> list[WeekPoint]:
    """以接入日为界按 7 天一组（接入前的组不跨过接入日），没开始的周不列。"""
    weeks = math.ceil(window_days / 7)
    before_start = go_live - timedelta(days=window_days)
    after_end = min(go_live + timedelta(days=window_days - 1), today)
    points: list[WeekPoint] = []
    for k in range(-weeks, weeks):
        start = go_live + timedelta(days=7 * k)
        end = start + timedelta(days=6)
        if k < 0:
            start, end = max(start, before_start), min(end, go_live - timedelta(days=1))
        else:
            end = min(end, after_end)
        if start > end:
            continue
        stats = window_stats(daily, metric, start, end)
        points.append(
            WeekPoint(
                index=k,
                phase="before" if k < 0 else "after",
                start=start,
                end=end,
                days=stats.days,
                mean=stats.mean,
            )
        )
    return points


# ================================================================ 对比


class MetricComparison(_Model):
    """一项指标的接入前后对比。"""

    metric: PlatformMetric
    label: str
    unit: MetricUnit
    higher_is_better: bool
    note: str  # 平台口径说明
    before: WindowStats
    after: WindowStats
    delta: float | None = None  # 接入后 − 接入前
    relative_change: float | None = None  # delta / 接入前（接入前为 0 时为空）
    direction: Direction
    direction_label: str
    flags: list[ReliabilityFlag] = Field(default_factory=list)
    flag_notes: list[str] = Field(default_factory=list)
    reliable: bool
    weekly: list[WeekPoint] = Field(default_factory=list)


def _direction(
    before: float | None, after: float | None, higher_is_better: bool
) -> tuple[Direction, float | None, float | None]:
    if before is None or after is None:
        return "insufficient", None, None
    delta = round(after - before, 6)
    if before == 0:
        relative = None
        unchanged = delta == 0
    else:
        relative = round(delta / before, 6)
        unchanged = abs(relative) < UNCHANGED_RELATIVE
    if unchanged:
        return "unchanged", delta, relative
    improved = (delta > 0) == higher_is_better
    return ("improved" if improved else "worse"), delta, relative


def _flags(before: WindowStats, after: WindowStats) -> list[ReliabilityFlag]:
    flags: list[ReliabilityFlag] = []
    for name, stats in (("before", before), ("after", after)):
        if not stats.days:
            flags.append("no_before" if name == "before" else "no_after")
            continue
        if stats.days < MIN_RELIABLE_DAYS:
            flags.append("few_days_before" if name == "before" else "few_days_after")
        if stats.sample_size is not None and stats.sample_size < MIN_RELIABLE_SAMPLES:
            flags.append("small_sample_before" if name == "before" else "small_sample_after")
    if before.days and after.days and set(before.sources) != set(after.sources):
        flags.append("sources_differ")
    if MetricSource.COMPUTED in before.sources or MetricSource.COMPUTED in after.sources:
        flags.append("computed")
    return flags


def window_bounds(go_live: date, window_days: int, today: date) -> tuple[date, date, date, date]:
    """(接入前起, 接入前止, 接入后起, 接入后止)，都含当天；接入后截到今天。"""
    return (
        go_live - timedelta(days=window_days),
        go_live - timedelta(days=1),
        go_live,
        min(go_live + timedelta(days=window_days - 1), today),
    )


def compare_metric(
    daily: Mapping[tuple[PlatformMetric, date], PlatformMetricSample],
    metric: PlatformMetric,
    *,
    go_live: date,
    window_days: int,
    today: date,
) -> MetricComparison:
    spec = PLATFORM_METRIC_SPECS[metric]
    b_start, b_end, a_start, a_end = window_bounds(go_live, window_days, today)
    before = window_stats(daily, metric, b_start, b_end)
    after = window_stats(daily, metric, a_start, a_end)
    direction, delta, relative = _direction(before.mean, after.mean, spec.higher_is_better)
    flags = _flags(before, after)
    return MetricComparison(
        metric=metric,
        label=spec.label,
        unit=spec.unit,
        higher_is_better=spec.higher_is_better,
        note=spec.note,
        before=before,
        after=after,
        delta=delta,
        relative_change=relative,
        direction=direction,
        direction_label=DIRECTION_LABELS[direction],
        flags=flags,
        flag_notes=[FLAG_LABELS[f] for f in flags],
        reliable=not flags,
        weekly=weekly_series(daily, metric, go_live, window_days, today),
    )


def overall_verdict(comparisons: Sequence[MetricComparison]) -> Verdict:
    """§1.3 "较接入前改善"：只看前后都有数据的指标（见模块说明）。"""
    covered = [c for c in comparisons if c.direction != "insufficient"]
    if not covered:
        return "insufficient"
    improved = any(c.direction == "improved" for c in covered)
    worse = any(c.direction == "worse" for c in covered)
    if improved and worse:
        return "mixed"
    if improved:
        return "improved"
    if worse:
        return "worse"
    return "unchanged"


class PlatformMetricsReport(_Model):
    """平台客服指标接入前后对比报表（§12.4、§15 Phase 2 商家报表）。"""

    shop_id: str
    platform: Platform
    today: date
    window_days: int
    go_live: date | None
    go_live_source: GoLiveSource
    go_live_label: str
    before_start: date | None = None
    before_end: date | None = None
    after_start: date | None = None
    after_end: date | None = None
    metrics: list[MetricComparison] = Field(default_factory=list)
    verdict: Verdict
    verdict_label: str
    counts: dict[str, int] = Field(default_factory=dict)  # 方向 → 指标数
    reliable: bool = False  # 有结论、而且参与结论的指标都没有可靠性提示
    notes: list[str] = Field(default_factory=list)


def build_report(
    *,
    shop_id: str,
    platform: Platform,
    samples: Iterable[PlatformMetricSample],
    go_live: GoLive,
    window_days: int = DEFAULT_WINDOW_DAYS,
    today: date,
) -> PlatformMetricsReport:
    """接入前后对比。samples 只取这家店铺、本平台考核的指标（别的忽略）。"""
    window_days = check_window_days(window_days)
    allowed = set(metrics_for(platform))
    rows = [s for s in samples if s.shop_id == shop_id and s.metric in allowed]
    if go_live.day is None:
        return PlatformMetricsReport(
            shop_id=shop_id,
            platform=platform,
            today=today,
            window_days=window_days,
            go_live=None,
            go_live_source=go_live.source,
            go_live_label=go_live.label,
            verdict="no_go_live",
            verdict_label=VERDICT_LABELS["no_go_live"],
            notes=["接入日期未知：请填写接入日期（智能辅助开始使用的那一天）后再对比"],
        )
    if go_live.day > today:
        raise ValueError("接入日期不能晚于今天")
    daily = pick_daily(rows)
    comparisons = [
        compare_metric(daily, m, go_live=go_live.day, window_days=window_days, today=today)
        for m in metrics_for(platform)
    ]
    verdict = overall_verdict(comparisons)
    counts = {d: sum(1 for c in comparisons if c.direction == d) for d in DIRECTION_LABELS}
    covered = [c for c in comparisons if c.direction != "insufficient"]
    notes: list[str] = []
    missing = [c.label for c in comparisons if c.direction == "insufficient"]
    if missing:
        notes.append(
            f"{'、'.join(missing)}：接入前或接入后没有数据，不参与结论（同步平台数据或手工登记）"
        )
    shaky = [c.label for c in covered if not c.reliable]
    if shaky:
        notes.append(f"{'、'.join(shaky)}：数据有可靠性提示，结论仅供参考")
    b_start, b_end, a_start, a_end = window_bounds(go_live.day, window_days, today)
    if (a_end - a_start).days + 1 < window_days:
        notes.append(f"接入后只有 {(a_end - a_start).days + 1} 天（接入不满 {window_days} 天）")
    return PlatformMetricsReport(
        shop_id=shop_id,
        platform=platform,
        today=today,
        window_days=window_days,
        go_live=go_live.day,
        go_live_source=go_live.source,
        go_live_label=go_live.label,
        before_start=b_start,
        before_end=b_end,
        after_start=a_start,
        after_end=a_end,
        metrics=comparisons,
        verdict=verdict,
        verdict_label=VERDICT_LABELS[verdict],
        counts=counts,
        reliable=bool(covered) and all(c.reliable for c in covered),
        notes=notes,
    )


__all__ = [
    "DEFAULT_WINDOW_DAYS",
    "DIRECTION_LABELS",
    "FLAG_LABELS",
    "GO_LIVE_LABELS",
    "MAX_HISTORY_DAYS",
    "MAX_SAMPLE_SIZE",
    "MAX_WINDOW_DAYS",
    "MIN_RELIABLE_DAYS",
    "MIN_RELIABLE_SAMPLES",
    "MIN_WINDOW_DAYS",
    "SOURCE_LABELS",
    "SOURCE_PRECEDENCE",
    "UNCHANGED_RELATIVE",
    "UNIT_LABELS",
    "VALUE_LIMITS",
    "VERDICT_LABELS",
    "GoLive",
    "MetricComparison",
    "PlatformMetricsReport",
    "WeekPoint",
    "WindowStats",
    "build_report",
    "check_day",
    "check_metric",
    "check_sample_size",
    "check_value",
    "check_window_days",
    "compare_metric",
    "first_assist_day",
    "overall_verdict",
    "pick_daily",
    "resolve_go_live",
    "weekly_series",
    "window_bounds",
    "window_stats",
]
