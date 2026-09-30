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
- 门槛里原来没有数据来源的条件（rollout_evidence）：测试集 = eval.verdict 系统事件里和线上一致
  （配置、问题定义、生成环节和判断层的提示词、判断层和分类器、大模型）、用真实大模型跑的最近一次
  结论（rodit.eval.fingerprint；分类器按这个进程实际加载的模型文件核对，
  见 unit_classifier_sha256）；平台违规、投诉 = 复核人员按周登记的 shop.rollout_incident
  （record_incident：周、条数、备注，不记原文）；另加实际发出去的文字的合规扫描
  （feedback.compliance，代理指标，单独一条）：扫描命中是候选，复核人员逐条确认或标记误报
  （review_compliance，写 compliance.reviewed：推荐 / 消息 ID、结论、规则名、备注，不记文字），
  确认的违规才算不通过、没确认的列为 unchecked。查看上线状态只读这些，不写任何东西。
写操作开始前先确认店铺还在（取消订购删掉店铺后不会把样本、审计、配置写回来）。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.app.unit import Repos, Unit
from rodit.core.audit import AuditAction
from rodit.core.config import EngineConfig, effective_shop_config
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
from rodit.core.ports import AuditLog
from rodit.decision.classifier import ClassifierProvider
from rodit.eval.fingerprint import current_fingerprint, test_set_evidence
from rodit.feedback.acceptance import ShopAcceptanceInput, message_latencies
from rodit.feedback.compliance import (
    COMPLIANCE_REVIEWED,
    ComplianceDecision,
    ComplianceReviewEntry,
    ComplianceRules,
    ComplianceWeek,
    check_flag_ref,
    find_flag,
    pairing_window,
    parse_reviews,
    review_detail,
    scan_sent,
    summarize_scan,
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
    week_start,
)
from rodit.feedback.rollout import (
    ASSIST_PERIODS,
    AUTO_PERIODS,
    MAX_INCIDENT_COUNT,
    SHADOW_CONTEXT_MESSAGES,
    STAGE_LABELS,
    SWITCHABLE_STAGES,
    IncidentEntry,
    IncidentKind,
    RolloutEvidence,
    RolloutStage,
    RolloutStatus,
    TestSetEvidence,
    current_entries,
    gate_summary,
    incident_detail,
    incident_total,
    parse_incidents,
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
INCIDENT_LIST_WEEKS = 8  # 登记列表默认看本周和之前 8 周
COMPLIANCE_REVIEW_WEEKS = 13  # 复核人员能确认 / 标记误报的疑似违规：本周和之前 13 周扫描出来的

MSG_GATE_NOT_PASSED = "影子对比还没达到进入智能辅助的门槛（§13.4）；确需切换请强制切换并写明原因"
MSG_AUTO_ENABLED = (
    "店铺开着全自动（auto_reply_enabled），关掉影子模式会直接进入全自动；请先关闭全自动"
)
MSG_FORCE_NOTE = "强制切换要写明原因（note）"
MSG_STAGE = "只能切换到影子模式（shadow）或智能辅助（assist）；全自动的开关不在这里"

logger = logging.getLogger(__name__)


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


class IncidentRecord(_Model):
    """一条平台违规 / 投诉登记（§13.4）。current：这一类、这一周以它为准（最后一次登记）。"""

    kind: IncidentKind
    week: str
    count: int
    note: str | None = None
    reviewer: str
    recorded_at: AwareDatetime
    current: bool = True

    @classmethod
    def of(cls, entry: IncidentEntry, *, current: bool) -> IncidentRecord:
        return cls(
            kind=entry.kind,
            week=entry.week,
            count=entry.count,
            note=entry.note,
            reviewer=entry.reviewer,
            recorded_at=entry.recorded_at,
            current=current,
        )


class ComplianceReviewRecord(_Model):
    """复核人员对一条疑似违规的结论（compliance.reviewed）。没有文字：推荐 / 消息 ID、命中的规则名、
    结论、备注（已做个人信息最小化）。"""

    suggestion_id: int | None = None
    msg_id: str | None = None
    decision: ComplianceDecision
    rules: list[str] = Field(default_factory=list)
    note: str | None = None
    reviewer: str
    recorded_at: AwareDatetime

    @classmethod
    def of(cls, entry: ComplianceReviewEntry) -> ComplianceReviewRecord:
        return cls(
            suggestion_id=entry.suggestion_id,
            msg_id=entry.msg_id,
            decision=entry.decision,
            rules=list(entry.rules),
            note=entry.note,
            reviewer=entry.reviewer,
            recorded_at=entry.recorded_at,
        )


class IncidentList(_Model):
    """一段时间里的登记（新的周在前，同一周新的登记在前）。只有计数和复核人员的备注，没有原文。"""

    shop_id: str
    since: AwareDatetime
    items: list[IncidentRecord] = Field(default_factory=list)


# ================================================================ 门槛的依据（§13.4）


def unit_classifier_sha256(unit: Unit) -> str | None:
    """这个进程的路由实际加载的分类器模型文件的 SHA-256（ClassifierProvider.file_sha256）。
    路由不是分类器、分类器不是从模型文件加载的（测试里现训的）返回 None：这时按
    RODIT_CLASSIFIER_PATH 读文件（rodit.eval.fingerprint.running_classifier_sha256）。"""
    provider = unit.router.provider
    return provider.file_sha256 if isinstance(provider, ClassifierProvider) else None


def eval_evidence(
    audit: AuditLog,
    config: EngineConfig,
    *,
    environ: Mapping[str, str] | None = None,
    loaded_classifier_sha256: str | None = None,
) -> TestSetEvidence:
    """测试集依据：eval.verdict 系统事件里和线上一致（配置、问题定义、生成环节和判断层的提示词、
    判断层和分类器、大模型都一样）、用真实大模型跑的最近一次结论。线上的大模型按部署规则从配置和
    环境变量解析（rodit.eval.fingerprint.current_fingerprint；environ 默认 os.environ，
    和线上单元读的是同一份）；分类器：同一进程里的调用方传 serve 实际加载的
    loaded_classifier_sha256（模型文件启动后被换掉也照旧），别的进程（运维任务）不传，
    读 RODIT_CLASSIFIER_PATH 指向的文件。"""
    events = audit.list_system_events(action=AuditAction.EVAL_VERDICT.value)
    running = current_fingerprint(
        config, environ=environ, loaded_classifier_sha256=loaded_classifier_sha256
    )
    return test_set_evidence(events, running)


def incident_entries(audit: AuditLog, shop_id: str, *, since: datetime) -> list[IncidentEntry]:
    """since 以后记下的登记（登记的周不会晚于登记时间：将来的周不让登记）。"""
    events = audit.list_events(shop_id, action=AuditAction.ROLLOUT_INCIDENT.value, since=since)
    return parse_incidents(events, shop_id=shop_id)


def compliance_reviews(
    audit: AuditLog, shop_id: str, *, since: datetime
) -> list[ComplianceReviewEntry]:
    """since 以后记下的复核人员结论（确认 / 误报）。疑似违规发生之后才能确认，since 取扫描
    窗口的起点。"""
    events = audit.list_events(shop_id, action=COMPLIANCE_REVIEWED, since=since)
    return parse_reviews(events, shop_id=shop_id)


def compliance_weeks(
    repos: Repos, config: EngineConfig, shop: Shop, *, since: datetime, until: datetime
) -> list[ComplianceWeek]:
    """[since, until) 里这家店实际发出去的文字的合规扫描，按 ISO 周（feedback.compliance）。
    禁用词用店铺的生效配置，绝对化用语用 config.qa_gate。推荐和消息按 pairing_window 多取一段：
    窗口开始前展示、窗口里发出的推荐也要配上它的消息，同一次发送在哪个窗口里都记在推荐上
    （上线状态、验收报告列出的 ID，review_compliance 的扫描里都有）。"""
    rules = ComplianceRules.from_config(config.qa_gate, effective_shop_config(config, shop))
    earliest, latest = pairing_window(since, until)
    edited = [
        r
        for r in repos.suggestions.list(
            shop.id, status=SuggestionStatus.EDITED_SENT, since=earliest
        )
        if r.shop_id == shop.id and r.created_at < latest
    ]
    messages = repos.conversations.list_outgoing(shop.id, since=earliest, until=until)
    return scan_sent(edited, messages, shop_id=shop.id, rules=rules, since=since, until=until)


def rollout_evidence(
    repos: Repos,
    config: EngineConfig,
    shop: Shop,
    now: datetime,
    *,
    loaded_classifier_sha256: str | None = None,
) -> RolloutEvidence:
    """上线状态用的依据：测试集结论（loaded_classifier_sha256 见 eval_evidence）；智能辅助门槛的
    2 周（+ 本周）的平台违规登记和合规扫描；全自动门槛的 4 周（+ 本周）的投诉登记。只读。"""
    assist = week_windows(now, ASSIST_PERIODS)
    auto = week_windows(now, AUTO_PERIODS)
    current = [iso_period(now)]
    entries = incident_entries(repos.audit, shop.id, since=auto[0][1])
    weeks = compliance_weeks(repos, config, shop, since=assist[0][1], until=now)
    reviews = compliance_reviews(repos.audit, shop.id, since=assist[0][1])
    return RolloutEvidence(
        test_set=eval_evidence(
            repos.audit, config, loaded_classifier_sha256=loaded_classifier_sha256
        ),
        platform_violations=incident_total(
            entries,
            IncidentKind.VIOLATION,
            [p for p, _, _ in assist],
            extra_weeks=current,
            as_of=now,
        ),
        complaints=incident_total(
            entries, IncidentKind.COMPLAINT, [p for p, _, _ in auto], extra_weeks=current, as_of=now
        ),
        compliance=summarize_scan(weeks, reviews=reviews),
    )


def acceptance_input(
    repos: Repos, config: EngineConfig, shop: Shop, *, since: datetime, until: datetime
) -> ShopAcceptanceInput:
    """一家店的 Phase 1 验收数据（feedback.acceptance）：采纳率的 2 周推荐记录、窗口里的推荐
    出现耗时、平台违规登记（窗口里每个完整周都要登记，登记的 0 要在那一周结束后登记才算；窗口
    不到一周时要登记所在的周）、合规扫描（带复核人员的确认 / 误报标记）、每周复核样本。只读。"""
    adoption_start = week_windows(until, ASSIST_PERIODS)[0][1]
    records = [
        r
        for r in repos.suggestions.list(shop.id, since=adoption_start)
        if r.shop_id == shop.id and r.created_at < until
    ]
    results = repos.audit.list_events(shop.id, action=AuditAction.RESULT.value, since=since)
    periods = periods_between(since, until - timedelta(microseconds=1))
    complete = [p for p in periods if period_range(p)[1] <= until]
    required = complete or periods
    entries = incident_entries(repos.audit, shop.id, since=week_start(since))
    reviews = [
        q
        for p in periods
        for q in repos.quality_reviews.list(shop.id, period=p)
        if q.shop_id == shop.id
    ]
    return ShopAcceptanceInput(
        shop_id=shop.id,
        stage=stage_of(effective_shop_config(config, shop)).value,
        records=records,
        latencies_ms=message_latencies(results, since=since, until=until),
        violations=incident_total(
            entries,
            IncidentKind.VIOLATION,
            required,
            extra_weeks=[p for p in periods if p not in required],
            as_of=until,
        ),
        compliance=summarize_scan(
            compliance_weeks(repos, config, shop, since=since, until=until),
            reviews=compliance_reviews(repos.audit, shop.id, since=since),
        ),
        reviews=reviews,
    )


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

    def _audit(
        self, shop_id: str, actor: str, action: AuditAction | str, detail: dict[str, Any]
    ) -> None:
        self.unit.repos.audit.record(
            AuditEvent(
                shop_id=shop_id,
                actor=actor,
                action=str(action),
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

    def status(self, shop_id: str) -> RolloutStatus:
        """当前阶段、建议和各阶段门槛。只读。测试集、平台违规、投诉取自 rollout_evidence
        （评测结论、复核人员的登记），另加合规扫描；没有数据的条件列为 unchecked 并写明原因。"""
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
            evidence=rollout_evidence(
                self.unit.repos,
                self.unit.config,
                shop,
                now,
                loaded_classifier_sha256=unit_classifier_sha256(self.unit),
            ),
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
        """和 status().shadow_gate 同一口径（影子窗口内的复核样本；测试集取自评测结论）。"""
        since = shadow_window(now)
        records = self._records(shop_id, since=since, status=SuggestionStatus.SHADOW)
        by_id = {r.id: r for r in records if r.id is not None}
        reviews = self._reviews(shop_id, periods_between(since, now))
        picked = shadow_reviews(reviews, by_id, shop_id=shop_id, since=since)
        test_set = eval_evidence(
            self.unit.repos.audit,
            self.unit.config,
            loaded_classifier_sha256=unit_classifier_sha256(self.unit),
        )
        return shadow_gate(
            launch_gate(picked),
            test_set_passed=test_set.passed,
            test_set_detail=test_set.detail or None,
        )

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
            # 只写 shadow_mode 一个键（仓储里原子地读改写）：商家同时改别的配置也不会互相覆盖
            previous_override = shop.config_overrides.get("shadow_mode")  # None：用的是默认值
            if (
                repo.update_shop_config(shop_id, {"shadow_mode": target == RolloutStage.SHADOW})
                is None
            ):
                raise ShopInactive(shop_id)
            try:
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
            except Exception:
                # 审计写不进去就撤回切换（覆盖项恢复原样，None 删掉键回到默认）：切换必须留痕；
                # 重试时阶段还没变，会重新检查门槛、重新写审计，而不是"已经切过了"的空操作
                try:
                    repo.update_shop_config(shop_id, {"shadow_mode": previous_override})
                except Exception as undo_error:  # noqa: BLE001 — 撤回失败只记日志，原异常照抛
                    logger.error(
                        "阶段切换的审计没写成，撤回切换也失败（店铺 %s，%s）",
                        shop_id,
                        type(undo_error).__name__,
                    )
                raise
            return StageChange(
                shop_id=shop_id,
                previous=current,
                stage=target,
                stage_label=STAGE_LABELS[target],
                changed=True,
                forced=forced,
                gate=gate,
            )

    # ================================================================ 平台违规、投诉登记（§13.4）

    def record_incident(
        self,
        shop_id: str,
        *,
        kind: IncidentKind | str,
        week: str,
        count: int,
        reviewer: str,
        note: str | None = None,
    ) -> IncidentRecord:
        """复核人员登记某一周平台下发的违规通知、买家投诉的总数（没有也登记 0）。写
        shop.rollout_incident（周、条数、备注——个人信息最小化后；不记通知或投诉的原文）。
        同一类、同一周再登记以最后一次为准。还没开始的周不能登记。
        那一周还没结束时登记的 0 是暂定的：门槛只认那一周结束后登记（或再确认）的 0
        （feedback.rollout.incident_total），大于 0 的随时登记都算。"""
        kind = IncidentKind(kind)
        week = check_period(week)
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("条数要是整数")
        if not 0 <= count <= MAX_INCIDENT_COUNT:
            raise ValueError(f"条数要在 0–{MAX_INCIDENT_COUNT} 之间")
        reviewer = _require_person(reviewer, "复核人")
        clean_note = _clean_note(note)
        now = self.unit.clock.now()
        if period_range(week)[0] > now:
            raise ValueError("这一周还没开始，不能登记")
        with self._processing(shop_id):
            self._shop(shop_id)
            entry = IncidentEntry(
                shop_id=shop_id,
                kind=kind,
                week=week,
                count=count,
                note=clean_note,
                reviewer=reviewer,
                recorded_at=now,
            )
            self._audit(shop_id, reviewer, AuditAction.ROLLOUT_INCIDENT, incident_detail(entry))
        return IncidentRecord.of(entry, current=True)

    def incidents(self, shop_id: str, *, weeks: int = INCIDENT_LIST_WEEKS) -> IncidentList:
        """本周和之前 weeks 周里的登记（含被后来的登记覆盖掉的，current=False）。只读。"""
        if not 0 <= weeks <= 520:
            raise ValueError("weeks 要在 0–520 之间")
        now = self.unit.clock.now()
        self._shop(shop_id)
        since = week_start(now) - timedelta(days=7 * weeks)
        entries = incident_entries(self.unit.repos.audit, shop_id, since=since)
        latest = {id(e) for e in current_entries(entries).values()}
        # 新的周在前；同一周新的登记在前（同一时间按记录顺序，后记的在前）
        ordered = sorted(enumerate(entries), key=lambda x: (x[1].week, x[1].recorded_at, x[0]))
        items = [IncidentRecord.of(e, current=id(e) in latest) for _, e in reversed(ordered)]
        return IncidentList(shop_id=shop_id, since=since, items=items)

    # ================================================================ 合规扫描的确认（§13.4）

    def review_compliance(
        self,
        shop_id: str,
        *,
        decision: ComplianceDecision | str,
        reviewer: str,
        suggestion_id: int | None = None,
        msg_id: str | None = None,
        note: str | None = None,
    ) -> ComplianceReviewRecord:
        """复核人员确认合规扫描的一条疑似违规（confirmed：确实违规，门槛不通过）或标记误报
        （dismissed：不计入）。suggestion_id（编辑后发送的推荐）和 msg_id（客服 / 机器人发出的消息）
        给且只给一个，要是本周和之前 COMPLIANCE_REVIEW_WEEKS 周扫描出来的疑似违规（否则 KeyError，
        接口层 404）。同一条再标记以最后一次为准。写 compliance.reviewed：ID、结论、命中的规则名、
        备注（个人信息最小化），不记文字。"""
        result = ComplianceDecision(decision)
        suggestion_id, msg_id = check_flag_ref(suggestion_id, msg_id)
        reviewer = _require_person(reviewer, "复核人")
        clean_note = _clean_note(note)
        now = self.unit.clock.now()
        since = week_start(now) - timedelta(days=7 * COMPLIANCE_REVIEW_WEEKS)
        with self._processing(shop_id):
            shop = self._shop(shop_id)
            weeks = compliance_weeks(
                self.unit.repos, self.unit.config, shop, since=since, until=now
            )
            flag = find_flag(weeks, suggestion_id=suggestion_id, msg_id=msg_id)
            if flag is None:
                raise KeyError(suggestion_id if suggestion_id is not None else msg_id)
            entry = ComplianceReviewEntry(
                shop_id=shop_id,
                suggestion_id=suggestion_id,
                msg_id=msg_id,
                decision=result,
                rules=list(flag.rules),
                note=clean_note,
                reviewer=reviewer,
                recorded_at=now,
            )
            self._audit(shop_id, reviewer, COMPLIANCE_REVIEWED, review_detail(entry))
        return ComplianceReviewRecord.of(entry)

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
    "COMPLIANCE_REVIEW_WEEKS",
    "INCIDENT_LIST_WEEKS",
    "MAX_NOTE_CHARS",
    "ComplianceReviewRecord",
    "IncidentList",
    "IncidentRecord",
    "ReviewList",
    "RolloutGateNotPassed",
    "RolloutService",
    "ShadowComparison",
    "StageChange",
    "acceptance_input",
    "compliance_reviews",
    "compliance_weeks",
    "eval_evidence",
    "unit_classifier_sha256",
    "incident_entries",
    "rollout_evidence",
]
