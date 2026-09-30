"""上线节奏：当前阶段、影子对比、各阶段门槛和建议（方案 §13.4；门槛数值见 §13.2、§13.4）。

纯函数，不读写仓储（组装在 rodit.app.rollout）。口径：

- 阶段：生效配置 shadow_mode → 影子模式；否则 auto_reply_enabled → 全自动（已开启，还要平台准入，
  只对开放的类型、高置信度生效）；否则智能辅助。这里只能在影子和智能辅助之间切换。
- 影子 → 智能辅助（§13.4 "测试集和影子对比都达到 13.2 的门槛"）：
  影子对比部分 = 窗口内影子推荐的人工复核样本（每周抽样里状态为 SHADOW 的推荐）按 §13.2 打分，
  事实正确、合规满分率都 ≥ 95%，且已打分的样本不少于 min_scored 条；
  测试集部分来自离线评测（rodit.eval）的结论：python -m rodit.ops record-eval 把评测报告记成
  eval.verdict 系统事件，只有配置指纹和线上配置一致、用真实大模型跑的结论才算
  （rodit.eval.fingerprint 的 test_set_evidence）；没有这样的结论时列为 unchecked，并写明原因；
  字符重合度（shadow_report 的 match_rate）只是参考，帮复核人员挑差异大的样本，不作为门槛。
  窗口默认是最近两个完整 ISO 周加本周（§13.4 影子模式 1–2 周）。
- 智能辅助 → 下一阶段（§13.4 "连续 2 周采纳率 ≥ 60%，平台违规为 0"）：截至上一个完整周的
  最近 2 周，phase_gate(min_periods=2)，每周至少 ASSIST_MIN_GROUPS 个推荐组（全自动发出去的推荐
  不算，见 feedback.metrics）。平台违规分两条报告：
  platform_violations = 复核人员按周登记的平台违规通知（shop.rollout_incident，见 incident_total：
  这 2 周都登记过、或登记的条数已经大于 0 才算检查过，否则 unchecked 并写明缺哪几周）；
  compliance_scan = 实际发出去的文字的合规扫描（feedback.compliance，代理指标，同一窗口到现在）：
  扫描命中只是候选，复核人员逐条确认——确认是违规的条数要为 0；还有没确认的（且没有确认的违规）时
  这一条是 unchecked，写明几条、哪些 ID；误报标记后不计。
- 全自动（按类型，§13.4 "该类型连续 4 周原样采纳率 ≥ 90%，且没有因此产生的投诉"）：
  config.routing.auto_threshold 列出的类型各跑 intent_gate(min_periods=4)，每周至少
  AUTO_MIN_GROUPS 个推荐组；投诉 = 复核人员按周登记的投诉（最近 4 周，口径同平台违规），
  没登记全时列为 unchecked。开不开由平台准入和人工决定，这里只报告。
- 复核人员的登记（shop.rollout_incident）：每条是某一周的总数，同一类型、同一周再登记以最后一次为准
  （改错用）；没有也要登记 0，门槛才算检查过。0 要在那一周结束之后登记（或结束后再登记一次确认）
  才算：周中登记的 0 只说明到登记时还没有，平台通知常常滞后（provisional_zero）。本周的登记
  （不是完整周）有大于 0 的也计入。
- GateResult.passed 只看已检查的条件；unchecked 的条件由复核人员确认（切换时写进审计）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from rodit.core.audit import AuditAction
from rodit.core.config import RoutingConfig
from rodit.core.enums import Intent, QualityReviewStatus, SuggestionStatus
from rodit.core.models import (
    QUALITY_PERIOD_PATTERN,
    AuditEvent,
    QualityReview,
    ShopConfig,
    StoredMessage,
    SuggestionRecord,
)
from rodit.feedback.compliance import ComplianceSummary, pending_reason
from rodit.feedback.metrics import (
    AdoptionReport,
    GateCheck,
    GateResult,
    ShadowIntentStats,
    ShadowReport,
    adoption_report,
    agent_replies_by_msg,
    intent_gate,
    phase_gate,
    shadow_report,
)
from rodit.feedback.review import (
    DEFAULT_MIN_SCORED,
    LAUNCH_MIN_FULL_RATE,
    LaunchGate,
    QualityReport,
    iso_period,
    launch_gate,
    period_range,
    week_start,
)

SHADOW_WEEKS = 2  # 影子对比窗口：最近两个完整周加本周（§13.4 影子模式 1–2 周）
ASSIST_PERIODS = 2  # §13.4 连续 2 周采纳率 ≥ 60%
AUTO_PERIODS = 4  # §13.4 该类型连续 4 周原样采纳率 ≥ 90%
ASSIST_MIN_ADOPTION = 0.6
AUTO_MIN_AS_IS = 0.9
ASSIST_MIN_GROUPS = 20  # 每周至少这么多推荐组，采纳率才算数
AUTO_MIN_GROUPS = 10  # 每个类型每周至少这么多推荐组
SHADOW_MATCH_THRESHOLD = 0.5
SHADOW_CONTEXT_MESSAGES = 500  # 影子对比时每个会话最多读的消息数（最近的）


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RolloutStage(StrEnum):
    SHADOW = "shadow"  # 影子模式：生成不展示，事后对比
    ASSIST = "assist"  # 智能辅助：展示推荐，客服决定
    AUTO = "auto"  # 已开全自动（只对开放的类型、高置信度生效）


STAGE_LABELS: dict[RolloutStage, str] = {
    RolloutStage.SHADOW: "影子模式",
    RolloutStage.ASSIST: "智能辅助",
    RolloutStage.AUTO: "全自动（已开启）",
}

SWITCHABLE_STAGES: frozenset[RolloutStage] = frozenset({RolloutStage.SHADOW, RolloutStage.ASSIST})
"""接口能切换到的阶段；全自动的开关（auto_reply_enabled）要平台准入，不在这里改。"""


class RolloutAdvice(StrEnum):
    STAY_SHADOW = "stay_shadow"
    READY_FOR_ASSIST = "ready_for_assist"
    STAY_ASSIST = "stay_assist"
    READY_FOR_NEXT_STAGE = "ready_for_next_stage"
    AUTO_ENABLED = "auto_enabled"


ADVICE_TEXT: dict[RolloutAdvice, str] = {
    RolloutAdvice.STAY_SHADOW: "影子对比还没达到 §13.2 门槛，继续影子模式",
    RolloutAdvice.READY_FOR_ASSIST: (
        "影子对比已达到门槛，可以切到智能辅助——未检查的条件需要复核人员确认"
    ),
    RolloutAdvice.STAY_ASSIST: "还没达到 §13.4 智能辅助的门槛，继续智能辅助",
    RolloutAdvice.READY_FOR_NEXT_STAGE: (
        "智能辅助已达到门槛：可以评估半自动 / 全自动（按类型开放，需平台准入和人工确认）"
    ),
    RolloutAdvice.AUTO_ENABLED: "已开全自动：按类型监控门槛，不达标的类型应关闭自动发送",
}

UNCHECKED_LABELS: dict[str, str] = {
    "test_set": (
        "测试集评分（§13.1、§13.2：离线评测 rodit.eval 的结论，"
        "python -m rodit.ops record-eval 记录）"
    ),
    "platform_violations": "平台违规为 0（§13.4：复核人员按周登记平台下发的违规通知）",
    "complaints": "没有因自动回复产生的投诉（§13.4：复核人员按周登记）",
    "compliance_scan": (
        "发出去的文字没有违规（合规扫描命中的疑似违规由复核人员逐条确认，误报标记后不计）"
    ),
    "consecutive": "统计周期首尾相接",
}
"""没有数据、需要人工确认的条件（GateResult.unchecked 的名字 → 说明）。"""

COMPLIANCE_SCAN_CHECK = "compliance_scan"
"""智能辅助门槛里合规扫描（代理指标）的条件名。"""


def stage_of(config: ShopConfig) -> RolloutStage:
    """生效配置 → 当前阶段（和 engine.understanding.effective_mode 的优先级一致）。"""
    if config.shadow_mode:
        return RolloutStage.SHADOW
    if config.auto_reply_enabled:
        return RolloutStage.AUTO
    return RolloutStage.ASSIST


# ================================================================ 复核人员登记的违规、投诉（§13.4）


class IncidentKind(StrEnum):
    VIOLATION = "violation"  # 平台下发的违规通知（话术违规、站外引流等）
    COMPLAINT = "complaint"  # 买家投诉（全自动门槛：因自动回复产生的投诉）


INCIDENT_KIND_LABELS: dict[IncidentKind, str] = {
    IncidentKind.VIOLATION: "平台违规",
    IncidentKind.COMPLAINT: "投诉",
}

INCIDENT_GATE_CHECKS: dict[IncidentKind, str] = {
    IncidentKind.VIOLATION: "platform_violations",
    IncidentKind.COMPLAINT: "complaints",
}
"""登记的类型 → 门槛里的条件名。"""

MAX_INCIDENT_COUNT = 10_000


class IncidentEntry(_Model):
    """复核人员的一条登记（shop.rollout_incident 审计事件）：某一周某类事件的总数。没有原文。"""

    shop_id: str
    kind: IncidentKind
    week: str = Field(pattern=QUALITY_PERIOD_PATTERN)
    count: int = Field(ge=0, le=MAX_INCIDENT_COUNT)
    note: str | None = None  # 复核人员的备注（已做个人信息最小化）
    reviewer: str
    recorded_at: AwareDatetime


def incident_detail(entry: IncidentEntry) -> dict[str, Any]:
    """写进 shop.rollout_incident 的 detail（复核人取 actor，时间取 created_at）。"""
    return {"kind": entry.kind.value, "week": entry.week, "count": entry.count, "note": entry.note}


def parse_incidents(events: Iterable[AuditEvent], *, shop_id: str) -> list[IncidentEntry]:
    """审计事件 → 登记（按时间升序）。别的店铺、别的动作、字段不对的事件跳过。"""
    out: list[IncidentEntry] = []
    for e in events:
        if e.shop_id != shop_id or e.action != AuditAction.ROLLOUT_INCIDENT.value:
            continue
        d = e.detail
        note = d.get("note")
        try:
            out.append(
                IncidentEntry(
                    shop_id=shop_id,
                    kind=d.get("kind"),
                    week=d.get("week"),
                    count=d.get("count"),
                    note=note if isinstance(note, str) else None,
                    reviewer=e.actor,
                    recorded_at=e.created_at,
                )
            )
        except ValidationError:
            continue
    return sorted(out, key=lambda x: x.recorded_at)


def current_entries(
    entries: Iterable[IncidentEntry],
) -> dict[tuple[IncidentKind, str], IncidentEntry]:
    """每个（类型、周）以最后一次登记为准。"""
    latest: dict[tuple[IncidentKind, str], IncidentEntry] = {}
    for e in sorted(entries, key=lambda x: x.recorded_at):
        latest[(e.kind, e.week)] = e
    return latest


class IncidentTotal(_Model):
    """一段窗口里某类登记的合计（门槛用）。count 为 None 表示没法判断（缺登记，见 reason）。"""

    kind: IncidentKind
    weeks: list[str]  # 必须登记的周（完整周）
    by_week: dict[str, int] = Field(default_factory=dict)  # 已登记的周（含本周）→ 条数
    missing: list[str] = Field(default_factory=list)  # 还要登记（或结束后再确认）的周
    provisional: list[str] = Field(default_factory=list)  # 其中登记了 0、但是在那一周结束前登记的
    known: int = 0  # 已登记的条数合计
    count: int | None = None
    reason: str | None = None


def provisional_zero(entry: IncidentEntry, *, as_of: datetime | None = None) -> bool:
    """这条登记是不是"暂定的 0"：条数为 0、登记时那一周还没结束——它只说明到登记时为止没有，
    不能代表整周（通知常常滞后）。as_of 给了、那一周到 as_of 还没结束时不算暂定（那一周本来就只能
    看到现在为止，如不到一周的验收窗口）。"""
    if entry.count > 0:
        return False
    end = period_range(entry.week)[1]
    if as_of is not None and end > as_of:
        return False
    return entry.recorded_at < end


def incident_total(
    entries: Iterable[IncidentEntry],
    kind: IncidentKind,
    weeks: Sequence[str],
    *,
    extra_weeks: Sequence[str] = (),
    as_of: datetime | None = None,
) -> IncidentTotal:
    """weeks 是必须登记的周，extra_weeks（如本周）有登记就计入。as_of：按哪个时间点判断
    （上线状态用现在，验收报告用窗口终点）。

    必须登记的周里，登记的 0 要在那一周结束之后登记（或结束后再登记一次确认）才算检查过：
    周一早上登记的 0 不能代表整周（provisional_zero），这一周仍算缺登记，reason 写明要在结束后确认。
    大于 0 的登记什么时候登记都算。

    登记的条数大于 0 → count = 合计（门槛不通过，不管缺不缺登记）；都登记了且合计为 0 → count = 0；
    否则 count = None，reason 写明缺哪几周。"""
    latest = current_entries(e for e in entries if e.kind == kind)
    wanted = list(dict.fromkeys([*weeks, *extra_weeks]))
    required = set(weeks)
    provisional = [
        w for w in weeks if (kind, w) in latest and provisional_zero(latest[(kind, w)], as_of=as_of)
    ]
    by_week = {
        w: latest[(kind, w)].count
        for w in wanted
        if (kind, w) in latest and not (w in required and w in provisional)
    }
    missing = [w for w in weeks if w not in by_week]
    known = sum(by_week.values())
    label = INCIDENT_KIND_LABELS[kind]
    count: int | None
    reason: str | None = None
    if known > 0:
        count = known
    elif not missing:
        count = 0
    else:
        count = None
        parts = []
        unregistered = [w for w in missing if w not in provisional]
        if unregistered:
            parts.append(
                f"还没登记 {'、'.join(unregistered)} 的{label}"
                "（复核人员按周登记平台下发的通知，没有也要登记 0）"
            )
        if provisional:
            parts.append(
                f"{'、'.join(provisional)} 的{label}登记 0 是在那一周结束前登记的，"
                "只能说明登记时还没有：那一周结束后再登记一次确认"
            )
        reason = "；".join(parts)
    return IncidentTotal(
        kind=kind,
        weeks=list(weeks),
        by_week=by_week,
        missing=missing,
        provisional=provisional,
        known=known,
        count=count,
        reason=reason,
    )


class TestSetEvidence(_Model):
    """测试集门槛（§13.1、§13.2）的依据：离线评测的结论（eval.verdict）。

    passed 为 None 表示没有能用的结论（reason 写明原因）；否则 detail 描述用的是哪一次评测。
    由 rodit.eval.fingerprint.test_set_evidence 按配置指纹挑选。"""

    __test__ = False  # 不是 pytest 的测试类

    passed: bool | None = None
    reason: str | None = None
    detail: str = ""
    verdict_at: AwareDatetime | None = None
    config_fingerprint: str | None = None  # 那次评测的配置指纹
    running_fingerprint: str | None = None  # 线上配置的指纹


class RolloutEvidence(_Model):
    """门槛里原来没有数据来源的条件现在的依据（§13.4）。"""

    test_set: TestSetEvidence
    platform_violations: IncidentTotal  # 智能辅助门槛的 2 周（+ 本周）
    complaints: IncidentTotal  # 全自动门槛的 4 周（+ 本周）
    compliance: ComplianceSummary  # 智能辅助门槛的 2 周到现在


# ================================================================ 采纳率周报（§13.4）


def week_windows(now: datetime, weeks: int) -> list[tuple[str, datetime, datetime]]:
    """截至上一个完整周的最近 weeks 个 ISO 周：(周期, 起, 止)，时间升序、首尾相接。"""
    current = week_start(now)
    out = []
    for k in range(weeks, 0, -1):
        start = current - timedelta(days=7 * k)
        out.append((iso_period(start), start, start + timedelta(days=7)))
    return out


def weekly_adoption(
    records: Iterable[SuggestionRecord], now: datetime, *, weeks: int = AUTO_PERIODS
) -> list[AdoptionReport]:
    """每周一份采纳率报表（带 since / until，phase_gate 据此检查首尾相接）。"""
    records = list(records)
    return [
        adoption_report(records, since=start, until=end)
        for _, start, end in week_windows(now, weeks)
    ]


class WeeklyAdoption(_Model):
    """一周的推荐组采纳情况（§13.4 门槛按推荐组算）。"""

    period: str
    since: AwareDatetime
    until: AwareDatetime
    groups: int = 0
    adopted_groups: int = 0
    as_is_groups: int = 0
    group_adoption_rate: float | None = None
    group_as_is_rate: float | None = None
    shadow_excluded: int = 0


def summarize_week(report: AdoptionReport) -> WeeklyAdoption:
    if report.since is None or report.until is None:
        raise ValueError("周报要有 since / until")
    o = report.overall
    return WeeklyAdoption(
        period=iso_period(report.since),
        since=report.since,
        until=report.until,
        groups=o.groups,
        adopted_groups=o.adopted_groups,
        as_is_groups=o.as_is_groups,
        group_adoption_rate=o.group_adoption_rate,
        group_as_is_rate=o.group_as_is_rate,
        shadow_excluded=report.shadow_excluded,
    )


def assist_gate(
    weekly: Sequence[AdoptionReport],
    *,
    platform_violations: int | None = None,
    potential_violations: int | None = None,
    unreviewed_violations: int = 0,
    min_groups: int = ASSIST_MIN_GROUPS,
) -> GateResult:
    """智能辅助 → 下一阶段：最近 2 周（weekly 的最后两份）推荐组采纳率都 ≥ 60%，平台违规为 0。

    platform_violations：复核人员登记的平台违规（None → unchecked）；
    potential_violations：合规扫描命中、复核人员确认是违规的条数（代理指标，单独一条
    compliance_scan；None 时不加这一条）；
    unreviewed_violations：扫描命中、复核人员还没确认的条数——只是候选，不算不通过：没有确认的
    违规、又有没确认的时，compliance_scan 列为 unchecked（由复核人员确认或标记误报）。"""
    gate = phase_gate(
        list(weekly)[-ASSIST_PERIODS:],
        min_adoption=ASSIST_MIN_ADOPTION,
        min_periods=ASSIST_PERIODS,
        min_samples=min_groups,
        platform_violations=platform_violations,
    )
    if potential_violations is None:
        return gate
    if potential_violations == 0 and unreviewed_violations > 0:
        return gate.model_copy(update={"unchecked": [*gate.unchecked, COMPLIANCE_SCAN_CHECK]})
    checks = [
        *gate.checks,
        GateCheck(
            name=COMPLIANCE_SCAN_CHECK,
            passed=potential_violations == 0,
            value=potential_violations,
            threshold=0,
            detail=("实际发出去的文字里复核人员确认的违规条数（合规扫描命中后逐条确认，代理指标）"),
        ),
    ]
    return gate.model_copy(
        update={"checks": checks, "passed": bool(weekly) and all(c.passed for c in checks)}
    )


def auto_intent_gates(
    weekly: Sequence[AdoptionReport],
    intents: Iterable[Intent],
    *,
    complaints: int | None = None,
    min_groups: int = AUTO_MIN_GROUPS,
) -> dict[Intent, GateResult]:
    """开放全自动的各类型：最近 4 周（weekly 的最后四份）原样采纳率都 ≥ 90%，没有投诉。"""
    periods = list(weekly)[-AUTO_PERIODS:]
    wanted = set(intents)
    return {
        intent: intent_gate(
            periods,
            intent,
            min_as_is=AUTO_MIN_AS_IS,
            min_periods=AUTO_PERIODS,
            min_samples=min_groups,
            complaints=complaints,
        )
        for intent in Intent
        if intent in wanted
    }


# ================================================================ 影子对比（§13.4 第一阶段）


def shadow_window(now: datetime, *, weeks: int = SHADOW_WEEKS) -> datetime:
    """默认影子对比窗口的开始：最近 weeks 个完整 ISO 周的周一 00:00（到现在为止，不设上限）。"""
    return week_start(now) - timedelta(days=7 * weeks)


def _in_window(r: SuggestionRecord, since: datetime | None, until: datetime | None) -> bool:
    return (since is None or r.created_at >= since) and (until is None or r.created_at < until)


def shadow_comparison(
    records: Iterable[SuggestionRecord],
    conversations: Iterable[Sequence[StoredMessage]],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    match_threshold: float = SHADOW_MATCH_THRESHOLD,
) -> ShadowReport:
    """窗口内的影子推荐和客服实际回复对比。conversations 是涉及的各会话的消息（时间升序，
    ConversationRepository.recent_messages），客服回复用 agent_replies_by_msg 对到买家消息上。
    records 只取 SHADOW；调用方只传一个店铺的数据（msg_id 在店铺内对应）。"""
    shadow = [
        r for r in records if r.status == SuggestionStatus.SHADOW and _in_window(r, since, until)
    ]
    replies: dict[str, list[str]] = {}
    for messages in conversations:
        for msg_id, texts in agent_replies_by_msg(messages).items():
            replies.setdefault(msg_id, []).extend(texts)
    return shadow_report(shadow, replies, match_threshold=match_threshold)


class ShadowSummary(_Model):
    """影子对比汇总（不含逐条结果）。"""

    since: AwareDatetime | None = None
    until: AwareDatetime | None = None
    groups: int = 0
    compared: int = 0
    no_reply: int = 0
    match_threshold: float = SHADOW_MATCH_THRESHOLD
    mean_similarity: float | None = None
    median_similarity: float | None = None
    match_rate: float | None = None
    by_intent: dict[Intent, ShadowIntentStats] = Field(default_factory=dict)


def summarize_shadow(
    report: ShadowReport, *, since: datetime | None, until: datetime | None
) -> ShadowSummary:
    return ShadowSummary(
        since=since,
        until=until,
        groups=report.groups,
        compared=report.compared,
        no_reply=report.no_reply,
        match_threshold=report.match_threshold,
        mean_similarity=report.mean_similarity,
        median_similarity=report.median_similarity,
        match_rate=report.match_rate,
        by_intent=report.by_intent,
    )


def shadow_reviews(
    reviews: Iterable[QualityReview],
    records_by_id: Mapping[int, SuggestionRecord],
    *,
    shop_id: str,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[QualityReview]:
    """复核样本里属于窗口内影子推荐的那些（按推荐 id 去重，同一推荐在不同周期被抽中只算一次，
    取最近复核过的）。"""
    picked: dict[int, QualityReview] = {}
    for q in reviews:
        r = records_by_id.get(q.suggestion_id)
        if (
            q.shop_id != shop_id
            or r is None
            or r.shop_id != shop_id
            or r.status != SuggestionStatus.SHADOW
            or not _in_window(r, since, until)
        ):
            continue
        cur = picked.get(q.suggestion_id)
        if cur is None or _review_rank(q) > _review_rank(cur):
            picked[q.suggestion_id] = q
    return [picked[k] for k in sorted(picked)]


def _review_rank(q: QualityReview) -> tuple[int, datetime]:
    done = q.status != QualityReviewStatus.PENDING
    return (1 if done else 0, q.reviewed_at or q.sampled_at)


def shadow_gate(
    quality: LaunchGate, *, test_set_passed: bool | None = None, test_set_detail: str | None = None
) -> GateResult:
    """影子 → 智能辅助：影子推荐的人工复核达到 §13.2 门槛；测试集结果没传时列为 unchecked。
    test_set_detail：测试集依据的说明（哪一次评测、各项满分率）。"""
    checks = [
        GateCheck(
            name="shadow_review.scored",
            passed=quality.scored >= quality.min_scored,
            value=quality.scored,
            threshold=quality.min_scored,
            detail="已打分的影子推荐样本数",
        ),
        GateCheck(
            name="shadow_review.factual_full_rate",
            passed=quality.factual_ok,
            value=quality.factual_full_rate,
            threshold=quality.min_full_rate,
            detail="事实正确满分率（§13.2）",
        ),
        GateCheck(
            name="shadow_review.compliance_full_rate",
            passed=quality.compliance_ok,
            value=quality.compliance_full_rate,
            threshold=quality.min_full_rate,
            detail="合规满分率（§13.2）",
        ),
    ]
    unchecked: list[str] = []
    if test_set_passed is None:
        unchecked.append("test_set")
    else:
        checks.append(
            GateCheck(
                name="test_set",
                passed=bool(test_set_passed),
                detail=test_set_detail or "离线测试集（§13.1）",
            )
        )
    return GateResult(
        stage=RolloutStage.SHADOW.value,
        passed=all(c.passed for c in checks),
        checks=checks,
        unchecked=unchecked,
    )


def gate_summary(gate: GateResult) -> dict[str, object]:
    """写审计用的门槛摘要：没通过的条件、未检查的条件、各条件的值（没有文字内容）。"""
    return {
        "stage": gate.stage,
        "passed": gate.passed,
        "failed": [c.name for c in gate.checks if not c.passed],
        "unchecked": list(gate.unchecked),
        "values": {c.name: c.value for c in gate.checks if c.value is not None},
    }


# ================================================================ 汇总与建议


class UncheckedCondition(_Model):
    name: str
    description: str
    reason: str | None = None  # 为什么没法检查（缺哪几周的登记、评测结论的指纹对不上……）


class RolloutStatus(_Model):
    """一个店铺的上线状态（§13.4）：当前阶段、建议、各阶段门槛和依据的数据。

    blocking / unchecked 针对当前阶段的下一道门槛：影子模式看 shadow_gate，智能辅助看
    assist_gate，全自动看各类型的 intent_gates。
    """

    shop_id: str
    as_of: AwareDatetime
    stage: RolloutStage
    stage_label: str
    advice: RolloutAdvice
    advice_text: str
    blocking: list[str] = Field(default_factory=list)  # 没通过的条件名
    unchecked: list[UncheckedCondition] = Field(default_factory=list)  # 需要人工确认的条件
    shadow: ShadowSummary
    shadow_quality: LaunchGate  # 窗口内影子推荐的复核结果
    shadow_gate: GateResult
    weekly: list[WeeklyAdoption] = Field(default_factory=list)
    assist_gate: GateResult
    intent_gates: dict[Intent, GateResult] = Field(default_factory=dict)
    latest_quality: QualityReport | None = None  # 最近一个有样本的周期的复核报表
    evidence: RolloutEvidence | None = None  # 测试集、平台违规、投诉、合规扫描的依据（没有文字）


def _unchecked(
    names: Iterable[str], reasons: Mapping[str, str | None] | None = None
) -> list[UncheckedCondition]:
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    reasons = reasons or {}
    return [
        UncheckedCondition(name=n, description=UNCHECKED_LABELS.get(n, n), reason=reasons.get(n))
        for n in seen
    ]


def rollout_status(
    *,
    shop_id: str,
    config: ShopConfig,
    routing: RoutingConfig,
    now: datetime,
    records: Sequence[SuggestionRecord],
    shadow: ShadowReport,
    shadow_since: datetime,
    reviews: Iterable[QualityReview],
    latest_quality: QualityReport | None = None,
    test_set_passed: bool | None = None,
    platform_violations: int | None = None,
    complaints: int | None = None,
    evidence: RolloutEvidence | None = None,
    min_scored: int = DEFAULT_MIN_SCORED,
    assist_min_groups: int = ASSIST_MIN_GROUPS,
    auto_min_groups: int = AUTO_MIN_GROUPS,
) -> RolloutStatus:
    """汇总上线状态。records 要覆盖最近 4 个完整周和影子窗口（shadow_since 起、不设上限）；
    shadow 是影子窗口内的对比（shadow_comparison）；reviews 是影子窗口涉及周期的复核样本。

    evidence 给了时，测试集、平台违规、投诉取自它（覆盖 test_set_passed / platform_violations /
    complaints），并加上合规扫描一条（复核人员确认的违规；只有没确认的疑似违规时是 unchecked）；
    没法检查的条件带上原因。"""
    stage = stage_of(config)
    reasons: dict[str, str | None] = {}
    test_set_detail: str | None = None
    potential: int | None = None
    unreviewed = 0
    if evidence is not None:
        test_set_passed = evidence.test_set.passed
        test_set_detail = evidence.test_set.detail or None
        platform_violations = evidence.platform_violations.count
        complaints = evidence.complaints.count
        potential = evidence.compliance.confirmed
        unreviewed = evidence.compliance.pending
        reasons = {
            "test_set": evidence.test_set.reason,
            "platform_violations": evidence.platform_violations.reason,
            "complaints": evidence.complaints.reason,
            "compliance_scan": pending_reason(evidence.compliance),
        }
    by_id = {r.id: r for r in records if r.id is not None and r.shop_id == shop_id}
    in_shadow = shadow_reviews(reviews, by_id, shop_id=shop_id, since=shadow_since)
    quality = launch_gate(in_shadow, min_scored=min_scored, min_full_rate=LAUNCH_MIN_FULL_RATE)
    s_gate = shadow_gate(quality, test_set_passed=test_set_passed, test_set_detail=test_set_detail)
    own = [r for r in records if r.shop_id == shop_id]
    weekly = weekly_adoption(own, now, weeks=AUTO_PERIODS)
    a_gate = assist_gate(
        weekly,
        platform_violations=platform_violations,
        potential_violations=potential,
        unreviewed_violations=unreviewed,
        min_groups=assist_min_groups,
    )
    i_gates = auto_intent_gates(
        weekly, routing.auto_threshold, complaints=complaints, min_groups=auto_min_groups
    )
    if stage == RolloutStage.SHADOW:
        gates = [s_gate]
        advice = RolloutAdvice.READY_FOR_ASSIST if s_gate.passed else RolloutAdvice.STAY_SHADOW
    elif stage == RolloutStage.ASSIST:
        gates = [a_gate]
        advice = RolloutAdvice.READY_FOR_NEXT_STAGE if a_gate.passed else RolloutAdvice.STAY_ASSIST
    else:
        gates = list(i_gates.values())
        advice = RolloutAdvice.AUTO_ENABLED
    blocking: list[str] = []
    for gate in gates:
        prefix = f"{gate.stage}:" if stage == RolloutStage.AUTO else ""
        blocking.extend(f"{prefix}{c.name}" for c in gate.checks if not c.passed)
    return RolloutStatus(
        shop_id=shop_id,
        as_of=now,
        stage=stage,
        stage_label=STAGE_LABELS[stage],
        advice=advice,
        advice_text=ADVICE_TEXT[advice],
        blocking=blocking,
        unchecked=_unchecked((n for gate in gates for n in gate.unchecked), reasons),
        shadow=summarize_shadow(shadow, since=shadow_since, until=now),
        shadow_quality=quality,
        shadow_gate=s_gate,
        weekly=[summarize_week(w) for w in weekly],
        assist_gate=a_gate,
        intent_gates=i_gates,
        latest_quality=latest_quality,
        evidence=evidence,
    )


__all__ = [
    "ADVICE_TEXT",
    "ASSIST_MIN_GROUPS",
    "AUTO_MIN_GROUPS",
    "COMPLIANCE_SCAN_CHECK",
    "INCIDENT_GATE_CHECKS",
    "INCIDENT_KIND_LABELS",
    "MAX_INCIDENT_COUNT",
    "SHADOW_CONTEXT_MESSAGES",
    "SHADOW_MATCH_THRESHOLD",
    "SHADOW_WEEKS",
    "STAGE_LABELS",
    "SWITCHABLE_STAGES",
    "UNCHECKED_LABELS",
    "IncidentEntry",
    "IncidentKind",
    "IncidentTotal",
    "RolloutAdvice",
    "RolloutEvidence",
    "RolloutStage",
    "RolloutStatus",
    "ShadowSummary",
    "TestSetEvidence",
    "UncheckedCondition",
    "WeeklyAdoption",
    "assist_gate",
    "auto_intent_gates",
    "current_entries",
    "gate_summary",
    "incident_detail",
    "incident_total",
    "parse_incidents",
    "provisional_zero",
    "rollout_status",
    "shadow_comparison",
    "shadow_gate",
    "shadow_reviews",
    "shadow_window",
    "stage_of",
    "summarize_shadow",
    "summarize_week",
    "week_windows",
    "weekly_adoption",
]
