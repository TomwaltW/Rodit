"""上线节奏与持续评估（方案 §13.4 影子模式对比、各阶段门槛；§13.5 每周从推荐中抽 50 条人工复核）。

所有写操作都在 processing(shop_id) 里做（和取消订购互斥，§8.4），操作人取自调用方身份，写审计
（core.audit.AuditAction 的 shop.rollout_stage / quality.*）。

- 查看：上线状态（当前阶段、建议、影子对比、各阶段门槛）、影子对比明细、复核报表——只有计数、分数、
  推荐 ID，没有文字，不写审计；
- 每周抽样（quality.sampled，只记条数）、复核打分（quality.scored，只记分数，不记备注原文）；
- 复核列表返回推荐文字和买家问题（写入时已做个人信息最小化，返回前再兜底一次），每次返回都写
  quality.viewed（谁、哪个周期、几条、哪些样本；没有文字）——§9.2 查看原始对话要留访问记录；
- 阶段切换只在影子和智能辅助之间：切到影子随时可以（安全方向：不再展示）；影子 → 智能辅助要过
  §13.4 的门槛（feedback.rollout.shadow_gate），没过时只能强制切换并写明原因；门槛摘要、是否强制、
  原因写进 shop.rollout_stage。已经在目标阶段时什么都不写。全自动的开关（auto_reply_enabled）
  要平台准入，不在这里改。
写操作开始前先确认店铺还在（取消订购删掉店铺后不会把样本、审计、配置写回来）。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.app.unit import Unit
from rodit.core.audit import AuditAction
from rodit.core.config import effective_shop_config
from rodit.core.enums import Intent, QualityReviewStatus, ShopStatus, SuggestionStatus
from rodit.core.errors import ActionNotAllowed, ShopInactive
from rodit.core.models import (
    AuditEvent,
    QualityReview,
    QualityScores,
    Shop,
    SuggestionRecord,
    strip_control_chars,
)
from rodit.feedback.metrics import GateResult, ShadowReport
from rodit.feedback.review import (
    DEFAULT_SAMPLE_SIZE,
    MAX_SAMPLE_SIZE,
    QualityReport,
    ReviewItem,
    SampleResult,
    check_period,
    eligible_records,
    iso_period,
    launch_gate,
    period_range,
    periods_between,
    plan_sample,
    previous_period,
    quality_report,
    review_item,
)
from rodit.feedback.rollout import (
    AUTO_PERIODS,
    SHADOW_CONTEXT_MESSAGES,
    STAGE_LABELS,
    SWITCHABLE_STAGES,
    RolloutStage,
    RolloutStatus,
    gate_summary,
    rollout_status,
    shadow_comparison,
    shadow_gate,
    shadow_reviews,
    shadow_window,
    stage_of,
    week_windows,
)
from rodit.feedback.service import minimize_text

ShopGate = Callable[[str], AbstractContextManager[None]]
"""AssistService.processing：按店铺的读方闸门（和取消订购互斥）。"""

MAX_NOTE_CHARS = 500
LATEST_QUALITY_LOOKBACK = 4  # 最近一个有样本的周期：本周和之前 4 周里找

MSG_GATE_NOT_PASSED = "影子对比还没达到进入智能辅助的门槛（§13.4）；确需切换请强制切换并写明原因"
MSG_AUTO_ENABLED = (
    "店铺开着全自动（auto_reply_enabled），关掉影子模式会直接进入全自动；请先关闭全自动"
)
MSG_FORCE_NOTE = "强制切换要写明原因（note）"
MSG_STAGE = "只能切换到影子模式（shadow）或智能辅助（assist）；全自动的开关不在这里"


class RolloutGateNotPassed(ActionNotAllowed):
    """影子 → 智能辅助的门槛没过、又没有强制切换（接口层返回 409 和门槛结果）。"""

    def __init__(self, gate: GateResult) -> None:
        super().__init__(MSG_GATE_NOT_PASSED)
        self.gate = gate


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StageChange(_Model):
    """一次阶段切换的结果。changed=False 表示已经在目标阶段（没有写任何东西）。"""

    shop_id: str
    previous: RolloutStage
    stage: RolloutStage
    stage_label: str
    changed: bool
    forced: bool = False  # 门槛没过、强制切换
    gate: GateResult | None = None  # 影子 → 智能辅助时的门槛结果


class ReviewList(_Model):
    """复核列表（含推荐文字和买家问题，只给复核人员，每次返回都留访问记录）。"""

    shop_id: str
    period: str
    status: QualityReviewStatus | None = None
    count: int
    items: list[ReviewItem] = Field(default_factory=list)


class ShadowComparison(_Model):
    """影子对比明细的窗口 + 结果（ShadowReport 含逐组的推荐 ID 和相似度，不含文字）。"""

    shop_id: str
    since: AwareDatetime
    until: AwareDatetime
    report: ShadowReport


def _clean_note(note: str | None) -> str | None:
    """复核备注 / 强制切换的原因：去控制字符、去首尾空白、个人信息最小化；空的当作没有。"""
    if note is None:
        return None
    text = strip_control_chars(str(note)).strip()
    if not text:
        return None
    if len(text) > MAX_NOTE_CHARS:
        raise ValueError(f"备注最多 {MAX_NOTE_CHARS} 个字")
    return minimize_text(text)


def _require_person(user_id: str, what: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError(f"{what}不能为空")
    return user_id.strip()


class RolloutService:
    def __init__(self, unit: Unit, processing: ShopGate) -> None:
        self.unit = unit
        self._processing = processing
        # 同一进程内同一时间只有一个抽样在算"已抽中 → 补多少"（两个复核人同时点抽样不会多抽）
        self._sample_lock = threading.Lock()

    # ================================================================ 公共

    def _shop(self, shop_id: str) -> Shop:
        """店铺不存在（含取消订购后已删除）或已停用：不读也不写（§8.4）。"""
        shop = self.unit.repos.knowledge.get_shop(shop_id)
        if shop is None:
            raise ShopInactive(shop_id)
        if shop.status == ShopStatus.SUSPENDED:
            raise ShopInactive(shop_id, shop.status)
        return shop

    def _audit(self, shop_id: str, actor: str, action: AuditAction, detail: dict[str, Any]) -> None:
        self.unit.repos.audit.record(
            AuditEvent(
                shop_id=shop_id,
                actor=actor,
                action=action.value,
                detail=detail,
                created_at=self.unit.clock.now(),
            )
        )

    def _period(self, period: str | None, now: datetime) -> str:
        return check_period(period) if period else previous_period(now)

    def _records(
        self,
        shop_id: str,
        *,
        since: datetime,
        until: datetime | None = None,
        status: SuggestionStatus | None = None,
    ) -> list[SuggestionRecord]:
        return [
            r
            for r in self.unit.repos.suggestions.list(shop_id, status=status, since=since)
            if r.shop_id == shop_id and (until is None or r.created_at < until)
        ]

    def _reviews(self, shop_id: str, periods: list[str]) -> list[QualityReview]:
        repo = self.unit.repos.quality_reviews
        return [q for p in periods for q in repo.list(shop_id, period=p) if q.shop_id == shop_id]

    def _shadow_report(
        self,
        shop_id: str,
        records: list[SuggestionRecord],
        since: datetime,
        until: datetime | None = None,
    ) -> ShadowReport:
        """窗口内的影子推荐和客服实际回复对比：按涉及的会话读消息（每个会话最近的若干条）。
        until 为空时不设上限。"""
        convs = sorted(
            {
                r.conversation_id
                for r in records
                if r.status == SuggestionStatus.SHADOW
                and since <= r.created_at
                and (until is None or r.created_at < until)
            }
        )
        repo = self.unit.repos.conversations
        messages = [
            repo.recent_messages(shop_id, conv, limit=SHADOW_CONTEXT_MESSAGES) for conv in convs
        ]
        return shadow_comparison(records, messages, since=since, until=until)

    def _latest_quality(self, shop_id: str, now: datetime) -> QualityReport | None:
        periods = [iso_period(now)] + [
            previous_period(now, weeks=k) for k in range(1, LATEST_QUALITY_LOOKBACK + 1)
        ]
        for period in periods:
            reviews = self.unit.repos.quality_reviews.list(shop_id, period=period)
            if reviews:
                return quality_report(reviews, shop_id=shop_id, period=period)
        return None

    # ================================================================ 上线状态、影子对比（§13.4）

    def status(
        self,
        shop_id: str,
        *,
        test_set_passed: bool | None = None,
        platform_violations: int | None = None,
        complaints: int | None = None,
    ) -> RolloutStatus:
        """当前阶段、建议和各阶段门槛。只读。没有数据来源的条件（测试集、平台违规、投诉）
        不传时列为 unchecked。"""
        now = self.unit.clock.now()
        shop = self._shop(shop_id)
        config = effective_shop_config(self.unit.config, shop)
        shadow_since = shadow_window(now)
        since = min(shadow_since, week_windows(now, AUTO_PERIODS)[0][1])
        records = self._records(shop_id, since=since)
        return rollout_status(
            shop_id=shop_id,
            config=config,
            routing=self.unit.config.routing,
            now=now,
            records=records,
            shadow=self._shadow_report(shop_id, records, shadow_since),
            shadow_since=shadow_since,
            reviews=self._reviews(shop_id, periods_between(shadow_since, now)),
            latest_quality=self._latest_quality(shop_id, now),
            test_set_passed=test_set_passed,
            platform_violations=platform_violations,
            complaints=complaints,
        )

    def shadow(
        self, shop_id: str, *, since: datetime | None = None, until: datetime | None = None
    ) -> ShadowComparison:
        """影子对比明细（§13.4）：since 默认最近两个完整周的开始；until 不含，默认不设上限
        （显示为现在）。只读。"""
        now = self.unit.clock.now()
        self._shop(shop_id)
        since = since or shadow_window(now)
        if until is not None and since >= until:
            raise ValueError("since 要早于 until")
        records = self._records(shop_id, since=since, until=until, status=SuggestionStatus.SHADOW)
        report = self._shadow_report(shop_id, records, since, until)
        return ShadowComparison(shop_id=shop_id, since=since, until=until or now, report=report)

    def _shadow_gate(self, shop_id: str, now: datetime) -> GateResult:
        """和 status().shadow_gate 同一口径（影子窗口内的复核样本；测试集列为 unchecked）。"""
        since = shadow_window(now)
        records = self._records(shop_id, since=since, status=SuggestionStatus.SHADOW)
        by_id = {r.id: r for r in records if r.id is not None}
        reviews = self._reviews(shop_id, periods_between(since, now))
        picked = shadow_reviews(reviews, by_id, shop_id=shop_id, since=since)
        return shadow_gate(launch_gate(picked))

    # ================================================================ 阶段切换（§13.4）

    def set_stage(
        self,
        shop_id: str,
        stage: RolloutStage | str,
        *,
        operator: str,
        force: bool = False,
        note: str | None = None,
        role: str | None = None,
    ) -> StageChange:
        """切换影子 / 智能辅助。谁能切到哪个阶段由接口层按角色判断（商家只能切回影子）。

        影子 → 智能辅助门槛没过：force=False 抛 RolloutGateNotPassed（409，配置不变）；
        force=True 必须写原因（note，否则 ValueError，不看门槛结果先检查），照常切换并在审计里记
        forced。切到影子不看 force（安全方向，随时可以）。
        """
        target = RolloutStage(stage)
        if target not in SWITCHABLE_STAGES:
            raise ValueError(MSG_STAGE)
        operator = _require_person(operator, "操作人")
        clean_note = _clean_note(note)
        force = bool(force) and target == RolloutStage.ASSIST
        if force and clean_note is None:
            raise ValueError(MSG_FORCE_NOTE)
        repo = self.unit.repos.knowledge
        with self._processing(shop_id):  # 读改写不能跨过取消订购（否则会把删掉的店铺写回来）
            shop = self._shop(shop_id)
            config = effective_shop_config(self.unit.config, shop)
            current = stage_of(config)
            if current == target:
                return StageChange(
                    shop_id=shop_id,
                    previous=current,
                    stage=current,
                    stage_label=STAGE_LABELS[current],
                    changed=False,
                )
            gate: GateResult | None = None
            forced = False
            if target == RolloutStage.ASSIST:
                if config.auto_reply_enabled:
                    raise ActionNotAllowed(MSG_AUTO_ENABLED)
                gate = self._shadow_gate(shop_id, self.unit.clock.now())
                if not gate.passed:
                    if not force:
                        raise RolloutGateNotPassed(gate)
                    forced = True
            overrides = {**shop.config_overrides, "shadow_mode": target == RolloutStage.SHADOW}
            repo.upsert_shop(shop.model_copy(update={"config_overrides": overrides}))
            self._audit(
                shop_id,
                operator,
                AuditAction.ROLLOUT_STAGE,
                {
                    "from": current.value,
                    "to": target.value,
                    "role": role,
                    "forced": forced,
                    "force_requested": force,
                    "gate": gate_summary(gate) if gate is not None else None,
                    "note": clean_note,  # 强制切换的原因（已做个人信息最小化）
                },
            )
            return StageChange(
                shop_id=shop_id,
                previous=current,
                stage=target,
                stage_label=STAGE_LABELS[target],
                changed=True,
                forced=forced,
                gate=gate,
            )

    # ================================================================ 每周人工复核（§13.5）

    def sample(
        self,
        shop_id: str,
        *,
        reviewer: str,
        period: str | None = None,
        size: int = DEFAULT_SAMPLE_SIZE,
    ) -> SampleResult:
        """抽一个周期（默认上一个完整周）的复核样本。重抽只补不换（见 feedback.review）。"""
        now = self.unit.clock.now()
        period = self._period(period, now)
        since, until = period_range(period)
        if since > now:
            raise ValueError("这个周期还没开始，不能抽样")
        if not 1 <= size <= MAX_SAMPLE_SIZE:
            raise ValueError(f"每次抽样条数要在 1–{MAX_SAMPLE_SIZE} 之间")
        reviewer = _require_person(reviewer, "复核人")
        repo = self.unit.repos.quality_reviews
        with self._processing(shop_id):
            self._shop(shop_id)
            with self._sample_lock:
                records = self._records(shop_id, since=since, until=until)
                existing = repo.list(shop_id, period=period)
                new = plan_sample(
                    records, existing, shop_id=shop_id, period=period, now=now, size=size
                )
                if new:
                    repo.add_many(new)
                total = repo.list(shop_id, period=period)
            eligible = len(eligible_records(records, shop_id=shop_id, period=period))
            by_intent = {i: n for i in Intent if (n := sum(1 for q in total if q.intent == i))}
            result = SampleResult(
                shop_id=shop_id,
                period=period,
                since=since,
                until=until,
                size=size,
                eligible=eligible,
                existing=len(existing),
                added=len(total) - len(existing),
                total=len(total),
                by_intent=by_intent,
            )
            if result.added:
                self._audit(
                    shop_id,
                    reviewer,
                    AuditAction.QUALITY_SAMPLED,
                    {
                        "period": period,
                        "size": size,
                        "eligible": eligible,
                        "existing": result.existing,
                        "added": result.added,
                        "total": result.total,
                        "by_intent": {i.value: n for i, n in by_intent.items()},
                    },
                )
            return result

    def list_reviews(
        self,
        shop_id: str,
        *,
        viewer: str,
        period: str | None = None,
        status: QualityReviewStatus | str | None = None,
    ) -> ReviewList:
        """复核列表（默认上一个完整周）：样本 + 推荐文字、买家问题、编辑后发出的文字、依据 ID。

        返回了内容就写 quality.viewed（§9.2 查看原始对话要留访问记录）：谁、哪个周期、几条、
        哪些样本，不含文字。和取消订购互斥（删掉的店铺不会留下访问记录）。
        """
        now = self.unit.clock.now()
        period = self._period(period, now)
        wanted = QualityReviewStatus(status) if status is not None else None
        viewer = _require_person(viewer, "查看人")
        suggestions = self.unit.repos.suggestions
        with self._processing(shop_id):
            self._shop(shop_id)
            reviews = [
                q
                for q in self.unit.repos.quality_reviews.list(shop_id, period=period, status=wanted)
                if q.shop_id == shop_id
            ]
            items = [review_item(q, suggestions.get(shop_id, q.suggestion_id)) for q in reviews]
            if items:
                self._audit(
                    shop_id,
                    viewer,
                    AuditAction.QUALITY_VIEWED,
                    {
                        "period": period,
                        "status": wanted.value if wanted else None,
                        "count": len(items),
                        "review_ids": [i.id for i in items],
                    },
                )
        return ReviewList(
            shop_id=shop_id, period=period, status=wanted, count=len(items), items=items
        )

    def score(
        self,
        shop_id: str,
        review_id: int,
        *,
        status: QualityReviewStatus | str,
        reviewer: str,
        scores: QualityScores | Mapping[str, int] | None = None,
        note: str | None = None,
    ) -> QualityReview:
        """记下复核结果（§13.2 四个维度 0–2 分，或跳过）。可以改判，历史在审计里。

        备注做个人信息最小化后存进样本；审计只记分数和"有没有备注"，不记备注原文。
        样本不存在或属于别的店铺：KeyError（404）。
        """
        result = QualityReviewStatus(status)
        if scores is not None and not isinstance(scores, QualityScores):
            scores = QualityScores.model_validate(scores)
        if result == QualityReviewStatus.PENDING:
            raise ValueError("复核结果只能是打分（scored）或跳过（skipped）")
        if result == QualityReviewStatus.SCORED and scores is None:
            raise ValueError("打分（scored）要给出四个维度的分数")
        if result == QualityReviewStatus.SKIPPED and scores is not None:
            raise ValueError("跳过（skipped）不能带分数")
        reviewer = _require_person(reviewer, "复核人")
        clean_note = _clean_note(note)
        repo = self.unit.repos.quality_reviews
        with self._processing(shop_id):
            self._shop(shop_id)
            current = repo.get(shop_id, review_id)
            if current is None or current.shop_id != shop_id:
                raise KeyError(review_id)
            updated = repo.record(
                shop_id,
                review_id,
                status=result,
                scores=scores,
                reviewer=reviewer,
                note=clean_note,
                reviewed_at=self.unit.clock.now(),
            )
            self._audit(
                shop_id,
                reviewer,
                AuditAction.QUALITY_SCORED,
                {
                    "review_id": review_id,
                    "suggestion_id": current.suggestion_id,
                    "period": current.period,
                    "intent": current.intent.value,
                    "status": result.value,
                    "previous_status": current.status.value,
                    "scores": scores.model_dump() if scores is not None else None,
                    "has_note": clean_note is not None,
                },
            )
            return updated

    def report(self, shop_id: str, *, period: str | None = None) -> QualityReport:
        """一个周期（默认上一个完整周）的复核报表：计数、各维度满分率和平均分、§13.2 门槛，
        按问题类型拆分。只读，没有文字。"""
        now = self.unit.clock.now()
        period = self._period(period, now)
        self._shop(shop_id)
        reviews = self.unit.repos.quality_reviews.list(shop_id, period=period)
        return quality_report(reviews, shop_id=shop_id, period=period)


__all__ = [
    "MAX_NOTE_CHARS",
    "ReviewList",
    "RolloutGateNotPassed",
    "RolloutService",
    "ShadowComparison",
    "StageChange",
]
