"""每周人工复核（方案 §13.5 "每周从推荐中抽 50 条人工复核"，按 §13.2 四个维度打分）。

纯函数，不读写仓储（组装在 rodit.app.rollout）。口径：

- 周期：ISO 周（"2026-W40"，core.models.QUALITY_PERIOD_PATTERN），按东八区（core.clock.CST）划分：
  周一 00:00 含、下周一 00:00 不含。默认周期是 now 所在周的上一个完整周。
- 抽样池：周期内生成（created_at）的全部推荐——展示过的、原样 / 编辑后发送的、弃用的、被刷新掉的、
  影子模式的（SAMPLE_STATUSES），都是引擎给出的推荐，都要能复核。
- 分层抽样：按问题类型分层，按比例分配（最大余数法，精确分数计算）；样本数不少于出现的类型数时，
  每个出现过的类型至少一条（从"多分到最多"的类型里挪）。层内按 sha256(店铺, 周期, 推荐 id) 排序，
  同一店铺、同一周期结果固定，不用全局随机数。
- 重抽只补不换：已抽中的样本一条不删；不够 size 条时（后来又有推荐进来）按分配缺口补齐，
  每次补缺口最大的类型（同样大按 Intent 定义顺序）。仓储 add_many 按（店铺, 周期, 推荐）幂等。
- 报表：各维度满分率（2 分的比例）、平均分、0 / 1 / 2 分分布，按问题类型拆分；§13.2 上线门槛
  "事实正确和合规两项的满分率都 ≥ 95%"（含 95%，用分数精确比较）。已打分的样本少于 min_scored
  条时为"数据不足"：95% 的门槛下 20 条才分得出一条错（19/20 = 95%）。跳过的样本不计入满分率。
  报表只有计数和分数，没有推荐文字、买家问题、复核备注。
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from enum import StrEnum
from fractions import Fraction

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.core.clock import CST
from rodit.core.enums import (
    INTENT_LABELS,
    DraftKind,
    Intent,
    QualityReviewStatus,
    RejectReason,
    SuggestionStatus,
)
from rodit.core.models import (
    QUALITY_PERIOD_PATTERN,
    QualityReview,
    QualityScores,
    SuggestionRecord,
)
from rodit.feedback.service import minimize_text

DEFAULT_SAMPLE_SIZE = 50  # §13.5 每周抽 50 条
MAX_SAMPLE_SIZE = 200
LAUNCH_MIN_FULL_RATE = 0.95  # §13.2 上线门槛：事实正确、合规满分率 ≥ 95%
DEFAULT_MIN_SCORED = 20  # 已打分样本少于这个数：数据不足（95% 门槛下 20 条才分得出一条错）
FULL_SCORE = 2

SAMPLE_STATUSES: frozenset[SuggestionStatus] = frozenset(
    {
        SuggestionStatus.SHOWN,
        SuggestionStatus.SENT,
        SuggestionStatus.EDITED_SENT,
        SuggestionStatus.REJECTED,
        SuggestionStatus.EXPIRED,
        SuggestionStatus.SHADOW,
    }
)
"""抽样池里的推荐状态：引擎生成的每一条推荐（含影子模式的，§13.4 影子对比也要人工评分）。"""

DIMENSIONS: dict[str, str] = {
    "factual": "事实正确",
    "relevance": "切题",
    "compliance": "合规",
    "expression": "表达",
}
"""§13.2 四个评分维度（键同 core.models.QualityScores 的字段）。"""

GATE_DIMENSIONS: tuple[str, ...] = ("factual", "compliance")
"""§13.2 上线门槛看的两个维度。"""

_PERIOD_RE = re.compile(QUALITY_PERIOD_PATTERN)
_INTENT_ORDER: dict[Intent, int] = {intent: i for i, intent in enumerate(Intent)}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ================================================================ 周期（ISO 周，东八区）


def _require_aware(dt: datetime) -> datetime:
    if not isinstance(dt, datetime) or dt.tzinfo is None:
        raise ValueError("需要带时区的时间")
    return dt


def iso_period(dt: datetime) -> str:
    """时间所在的 ISO 周（按东八区），如 "2026-W40"。"""
    year, week, _ = _require_aware(dt).astimezone(CST).isocalendar()
    return f"{year:04d}-W{week:02d}"


def week_start(dt: datetime) -> datetime:
    """时间所在 ISO 周的周一 00:00（东八区）。"""
    local = _require_aware(dt).astimezone(CST)
    monday = local.date() - timedelta(days=local.weekday())
    return datetime(monday.year, monday.month, monday.day, tzinfo=CST)


def check_period(period: str) -> str:
    """校验周期写法和取值（该年确实有这一周），返回原值；不合法抛 ValueError。"""
    if not isinstance(period, str) or not _PERIOD_RE.fullmatch(period):
        raise ValueError("周期要写成 ISO 周，如 2026-W40")
    year, week = int(period[:4]), int(period[6:])
    try:
        date.fromisocalendar(year, week, 1) + timedelta(days=7)  # 周的结束也要表示得出来
    except ValueError:
        raise ValueError(f"{year} 年没有第 {week} 周") from None
    except OverflowError:
        raise ValueError("周期超出支持的范围") from None
    return period


def period_range(period: str) -> tuple[datetime, datetime]:
    """周期 → [周一 00:00, 下周一 00:00)（东八区）。"""
    check_period(period)
    monday = date.fromisocalendar(int(period[:4]), int(period[6:]), 1)
    start = datetime(monday.year, monday.month, monday.day, tzinfo=CST)
    return start, start + timedelta(days=7)


def previous_period(now: datetime, *, weeks: int = 1) -> str:
    """now 所在周往前数第 weeks 个完整周（默认上一周，§13.5 每周复核上一周的推荐）。"""
    if weeks < 1:
        raise ValueError("weeks 至少为 1")
    return iso_period(week_start(now) - timedelta(days=7 * weeks))


def recent_periods(now: datetime, count: int) -> list[str]:
    """截至上一个完整周的最近 count 个 ISO 周，时间升序、首尾相接。"""
    return [previous_period(now, weeks=k) for k in range(count, 0, -1)]


def periods_between(since: datetime, until: datetime) -> list[str]:
    """从 since 所在周到 until 所在周（都含）的 ISO 周，时间升序。"""
    out: list[str] = []
    cursor = week_start(since)
    while cursor <= until:
        out.append(iso_period(cursor))
        cursor += timedelta(days=7)
    return out


# ================================================================ 抽样（§13.5）


def sample_key(shop_id: str, period: str, suggestion_id: int) -> str:
    """层内排序键：同一店铺、同一周期结果固定，不同周期互不相关。"""
    raw = f"{shop_id}\x1f{period}\x1f{suggestion_id}".encode()
    return hashlib.sha256(raw).hexdigest()


def eligible_records(
    records: Iterable[SuggestionRecord], *, shop_id: str, period: str
) -> list[SuggestionRecord]:
    """周期内生成的、本店铺的、可抽样状态的推荐（有 id）。"""
    start, end = period_range(period)
    return [
        r
        for r in records
        if r.shop_id == shop_id
        and r.id is not None
        and r.status in SAMPLE_STATUSES
        and start <= r.created_at < end
    ]


def allocate(counts: Mapping[Intent, int], size: int) -> dict[Intent, int]:
    """把 size 条样本按各问题类型的推荐数按比例分配（最大余数法）。

    总数不超过 size 时全取；size 不少于出现的类型数时每个类型至少一条（从分配超出比例最多、
    且不止一条的类型里挪一条）。同样大时按 Intent 定义顺序，结果固定。
    """
    if size < 0:
        raise ValueError("size 不能为负")
    strata = [i for i in Intent if counts.get(i, 0) > 0]
    total = sum(counts[i] for i in strata)
    if total <= size:
        return {i: counts[i] for i in strata}
    quotas = {i: Fraction(size * counts[i], total) for i in strata}
    alloc = {i: int(quotas[i]) for i in strata}  # 向下取整（非负）
    left = size - sum(alloc.values())
    by_remainder = sorted(strata, key=lambda i: (-(quotas[i] - alloc[i]), _INTENT_ORDER[i]))
    for i in by_remainder[:left]:
        alloc[i] += 1
    if size >= len(strata):
        for i in strata:
            if alloc[i]:
                continue
            donor = max(
                (j for j in strata if alloc[j] > 1),
                key=lambda j: (alloc[j] - quotas[j], alloc[j], -_INTENT_ORDER[j]),
            )
            alloc[donor] -= 1
            alloc[i] = 1
    return alloc


def plan_sample(
    records: Iterable[SuggestionRecord],
    existing: Iterable[QualityReview],
    *,
    shop_id: str,
    period: str,
    now: datetime,
    size: int = DEFAULT_SAMPLE_SIZE,
) -> list[QualityReview]:
    """这次要新写入的样本（pending，按抽中顺序）。

    existing 是这个店铺、这个周期已经抽中的样本：一条不删，只把总数补到 size（池子不够时取完为止）。
    第一次抽样时新样本正好是 allocate 的分配；补抽时按各类型"分配 − 已抽中"的缺口，每次补缺口
    最大的类型，层内按 sample_key 顺序取还没抽中的。
    """
    if not 1 <= size <= MAX_SAMPLE_SIZE:
        raise ValueError(f"每次抽样条数要在 1–{MAX_SAMPLE_SIZE} 之间")
    check_period(period)
    sampled = {q.suggestion_id for q in existing if q.shop_id == shop_id and q.period == period}
    need = size - len(sampled)
    if need <= 0:
        return []
    pool = eligible_records(records, shop_id=shop_id, period=period)
    ordered = sorted(pool, key=lambda r: sample_key(shop_id, period, r.id or 0))
    by_intent: dict[Intent, list[SuggestionRecord]] = {}
    for r in ordered:
        by_intent.setdefault(r.intent, []).append(r)
    target = allocate({i: len(v) for i, v in by_intent.items()}, size)
    have = Counter(r.intent for r in ordered if r.id in sampled)
    remaining = {i: [r for r in v if r.id not in sampled] for i, v in by_intent.items()}
    deficit = {i: max(0, target.get(i, 0) - have[i]) for i in by_intent}
    picked: list[SuggestionRecord] = []
    while len(picked) < need:
        candidates = [i for i in by_intent if deficit[i] > 0 and remaining[i]]
        if not candidates:
            break
        best = max(candidates, key=lambda i: (deficit[i], -_INTENT_ORDER[i]))
        picked.append(remaining[best].pop(0))
        deficit[best] -= 1
    return [
        QualityReview(
            shop_id=shop_id,
            period=period,
            suggestion_id=r.id or 0,
            intent=r.intent,
            sampled_at=now,
        )
        for r in picked
    ]


class SampleResult(_Model):
    """一次抽样的结果（只有计数）。"""

    shop_id: str
    period: str
    since: AwareDatetime
    until: AwareDatetime
    size: int
    eligible: int  # 周期内可抽的推荐数
    existing: int  # 这次之前已抽中的
    added: int  # 这次新抽中的
    total: int  # 抽中的总数
    by_intent: dict[Intent, int] = Field(default_factory=dict)  # 抽中的总数按问题类型


# ================================================================ 复核时看的内容（§9.2 留访问记录）


class ReviewedSuggestion(_Model):
    """被复核的推荐：推荐文字、买家问题、编辑后发出的文字（都再过一遍个人信息最小化）和依据 ID。"""

    text: str
    question_text: str | None = None
    edited_text: str | None = None
    kind: DraftKind
    status: SuggestionStatus
    intent: Intent
    intent_confidence: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    reject_reason: RejectReason | None = None
    created_at: AwareDatetime


class ReviewItem(_Model):
    """复核列表的一条：样本 + 推荐内容。推荐已不存在时 suggestion 为 None。"""

    id: int
    period: str
    suggestion_id: int
    intent: Intent
    intent_label: str
    status: QualityReviewStatus
    scores: QualityScores | None = None
    reviewer: str | None = None
    note: str | None = None
    sampled_at: AwareDatetime
    reviewed_at: AwareDatetime | None = None
    suggestion: ReviewedSuggestion | None = None


def _minimized(text: str | None) -> str | None:
    return minimize_text(text) if text else text


def review_item(review: QualityReview, record: SuggestionRecord | None) -> ReviewItem:
    """样本 + 推荐 → 复核列表的一条。文字在写入时已经最小化，这里再兜底一次（mask_pii）；
    推荐不属于同一店铺、同一 id 时不带内容（纵深防御，§9.2）。"""
    if review.id is None:
        raise ValueError("样本还没有 id")
    if record is not None and (record.shop_id, record.id) != (review.shop_id, review.suggestion_id):
        record = None
    suggestion = None
    if record is not None:
        suggestion = ReviewedSuggestion(
            text=minimize_text(record.text),
            question_text=_minimized(record.question_text),
            edited_text=_minimized(record.edited_text),
            kind=record.kind,
            status=record.status,
            intent=record.intent,
            intent_confidence=record.intent_confidence,
            evidence_ids=list(record.evidence_ids),
            reject_reason=record.reject_reason,
            created_at=record.created_at,
        )
    return ReviewItem(
        id=review.id,
        period=review.period,
        suggestion_id=review.suggestion_id,
        intent=review.intent,
        intent_label=INTENT_LABELS[review.intent],
        status=review.status,
        scores=review.scores,
        reviewer=review.reviewer,
        note=_minimized(review.note),
        sampled_at=review.sampled_at,
        reviewed_at=review.reviewed_at,
        suggestion=suggestion,
    )


# ================================================================ 报表与 §13.2 上线门槛


class GateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT = "insufficient_data"  # 已打分的样本不够


class DimensionStats(_Model):
    """一个评分维度。rate / mean 在没有已打分样本时为 None。"""

    label: str
    scored: int = 0
    full: int = 0  # 2 分的条数
    full_rate: float | None = None
    mean: float | None = None
    distribution: dict[int, int] = Field(default_factory=dict)  # 分数 → 条数（0 / 1 / 2）


class LaunchGate(_Model):
    """§13.2 上线门槛："事实正确"和"合规"两项的满分率都 ≥ 95%（含 95%）。

    status：已打分的样本少于 min_scored 条为 insufficient_data（factual_ok / compliance_ok 仍按
    已有数据算）；否则两项都达标为 passed，有一项不达标为 failed。
    """

    status: GateStatus
    min_full_rate: float = LAUNCH_MIN_FULL_RATE
    min_scored: int
    sampled: int = 0
    scored: int = 0
    skipped: int = 0
    pending: int = 0
    factual_full_rate: float | None = None
    compliance_full_rate: float | None = None
    factual_ok: bool = False
    compliance_ok: bool = False
    detail: str = ""


class QualityIntentStats(_Model):
    intent: Intent
    label: str
    sampled: int = 0
    scored: int = 0
    skipped: int = 0
    pending: int = 0
    dimensions: dict[str, DimensionStats] = Field(default_factory=dict)


class QualityReport(_Model):
    """一个周期的复核报表（§13.5）。by_intent 按 Intent 定义顺序，只含有样本的类型。

    没有任何文字（推荐、买家问题、备注都不在报表里）。
    """

    shop_id: str
    period: str | None = None
    since: AwareDatetime | None = None
    until: AwareDatetime | None = None
    sampled: int = 0
    scored: int = 0
    skipped: int = 0
    pending: int = 0
    dimensions: dict[str, DimensionStats] = Field(default_factory=dict)
    gate: LaunchGate
    by_intent: dict[Intent, QualityIntentStats] = Field(default_factory=dict)


def meets_rate(full: int, total: int, min_rate: float) -> bool:
    """full / total ≥ min_rate（含等于），用分数精确比较；total 为 0 时不达标。"""
    if total <= 0:
        return False
    return Fraction(full, total) >= Fraction(str(min_rate))


def _scored(reviews: Sequence[QualityReview]) -> list[QualityScores]:
    return [
        r.scores for r in reviews if r.status == QualityReviewStatus.SCORED and r.scores is not None
    ]


def dimension_stats(reviews: Sequence[QualityReview]) -> dict[str, DimensionStats]:
    """各维度的满分率、平均分、分数分布（只算已打分的样本）。"""
    scores = _scored(reviews)
    out: dict[str, DimensionStats] = {}
    for key, label in DIMENSIONS.items():
        values = [int(getattr(s, key)) for s in scores]
        full = sum(1 for v in values if v == FULL_SCORE)
        dist = Counter(values)
        out[key] = DimensionStats(
            label=label,
            scored=len(values),
            full=full,
            full_rate=full / len(values) if values else None,
            mean=sum(values) / len(values) if values else None,
            distribution={score: dist.get(score, 0) for score in range(FULL_SCORE + 1)},
        )
    return out


def _counts(reviews: Sequence[QualityReview]) -> Counter[QualityReviewStatus]:
    return Counter(r.status for r in reviews)


def launch_gate(
    reviews: Sequence[QualityReview],
    *,
    min_scored: int = DEFAULT_MIN_SCORED,
    min_full_rate: float = LAUNCH_MIN_FULL_RATE,
) -> LaunchGate:
    """§13.2 上线门槛（事实正确、合规满分率 ≥ 95%），跳过和待复核的样本不算。"""
    dims = dimension_stats(reviews)
    counts = _counts(reviews)
    scored = counts[QualityReviewStatus.SCORED]
    factual, compliance = dims["factual"], dims["compliance"]
    factual_ok = meets_rate(factual.full, factual.scored, min_full_rate)
    compliance_ok = meets_rate(compliance.full, compliance.scored, min_full_rate)
    need = max(1, min_scored)
    if scored < need:
        status, detail = GateStatus.INSUFFICIENT, f"已打分 {scored} 条，至少要 {need} 条"
    elif factual_ok and compliance_ok:
        status, detail = GateStatus.PASSED, "事实正确、合规满分率都达到门槛"
    else:
        checks = (("factual", factual_ok), ("compliance", compliance_ok))
        failed = [DIMENSIONS[key] for key, ok in checks if not ok]
        status, detail = GateStatus.FAILED, f"{'、'.join(failed)}满分率低于 {min_full_rate:.0%}"
    return LaunchGate(
        status=status,
        min_full_rate=min_full_rate,
        min_scored=need,
        sampled=len(reviews),
        scored=scored,
        skipped=counts[QualityReviewStatus.SKIPPED],
        pending=counts[QualityReviewStatus.PENDING],
        factual_full_rate=factual.full_rate,
        compliance_full_rate=compliance.full_rate,
        factual_ok=factual_ok,
        compliance_ok=compliance_ok,
        detail=detail,
    )


def quality_report(
    reviews: Iterable[QualityReview],
    *,
    shop_id: str,
    period: str | None = None,
    min_scored: int = DEFAULT_MIN_SCORED,
) -> QualityReport:
    """一个周期（period 为空时是传入的全部样本）的复核报表。只统计本店铺的样本。"""
    picked = [r for r in reviews if r.shop_id == shop_id and (period is None or r.period == period)]
    since = until = None
    if period is not None:
        since, until = period_range(period)
    counts = _counts(picked)
    by_intent: dict[Intent, QualityIntentStats] = {}
    for intent in Intent:
        subset = [r for r in picked if r.intent == intent]
        if not subset:
            continue
        c = _counts(subset)
        by_intent[intent] = QualityIntentStats(
            intent=intent,
            label=INTENT_LABELS[intent],
            sampled=len(subset),
            scored=c[QualityReviewStatus.SCORED],
            skipped=c[QualityReviewStatus.SKIPPED],
            pending=c[QualityReviewStatus.PENDING],
            dimensions=dimension_stats(subset),
        )
    return QualityReport(
        shop_id=shop_id,
        period=period,
        since=since,
        until=until,
        sampled=len(picked),
        scored=counts[QualityReviewStatus.SCORED],
        skipped=counts[QualityReviewStatus.SKIPPED],
        pending=counts[QualityReviewStatus.PENDING],
        dimensions=dimension_stats(picked),
        gate=launch_gate(picked, min_scored=min_scored),
        by_intent=by_intent,
    )


__all__ = [
    "DEFAULT_MIN_SCORED",
    "DEFAULT_SAMPLE_SIZE",
    "DIMENSIONS",
    "FULL_SCORE",
    "GATE_DIMENSIONS",
    "LAUNCH_MIN_FULL_RATE",
    "MAX_SAMPLE_SIZE",
    "SAMPLE_STATUSES",
    "DimensionStats",
    "GateStatus",
    "LaunchGate",
    "QualityIntentStats",
    "QualityReport",
    "ReviewItem",
    "ReviewedSuggestion",
    "SampleResult",
    "allocate",
    "check_period",
    "dimension_stats",
    "eligible_records",
    "iso_period",
    "launch_gate",
    "meets_rate",
    "period_range",
    "periods_between",
    "plan_sample",
    "previous_period",
    "quality_report",
    "recent_periods",
    "review_item",
    "sample_key",
    "week_start",
]
