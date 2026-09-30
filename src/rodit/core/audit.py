"""审计：事件名约定 + 内存实现（开发和测试用；生产用 storage 里的 PostgreSQL 实现）。"""

from __future__ import annotations

import threading
from datetime import datetime
from enum import StrEnum

from rodit.core.enums import DecisionStage
from rodit.core.models import AuditEvent, DecisionLogEntry


class AuditAction(StrEnum):
    """audit_log.action 的取值（§4.1 原则 6、§12.4）。detail 不放买家原文。"""

    # 问答引擎（每条消息）
    UNDERSTAND = "engine.understand"
    ROUTE = "engine.route"
    RETRIEVE = "engine.retrieve"
    GENERATE = "engine.generate"
    QA = "engine.qa"
    SUGGEST = "engine.suggest"
    AUTO_REPLY = "engine.auto_reply"
    RESULT = "engine.result"  # 每条消息一条汇总：intent、action、硬规则命中、降级、各环节耗时
    # 反馈闭环
    FEEDBACK_SENT = "feedback.sent"
    FEEDBACK_EDITED = "feedback.edited_sent"
    FEEDBACK_REJECTED = "feedback.rejected"
    FEEDBACK_BAD_WORDING = "feedback.bad_wording"
    KNOWLEDGE_CANDIDATE = "knowledge.candidate"
    KNOWLEDGE_APPROVED = "knowledge.approved"
    KNOWLEDGE_RETIRED = "knowledge.retired"
    # 商家控制台（§5.2 知识中心、§9.1 知识编辑 / 配置、§9.3 店铺配置）
    KNOWLEDGE_CREATED = "knowledge.created"  # 商家录入 FAQ
    KNOWLEDGE_EDITED = "knowledge.edited"  # 商家改 FAQ（旧条目下线、新条目上线）
    POLICY_UPDATED = "knowledge.policy_updated"
    ITEM_CARD_UPDATED = "knowledge.item_card_updated"  # 商家改商品卡片上自己填的字段
    ITEM_SYNC = "knowledge.item_sync"  # 平台商品同步（§5.2.3）：新增 / 更新 / 已下架的条数
    CONFIG_UPDATED = "shop.config_updated"
    # 上线节奏与持续评估（§13.4、§13.5）
    ROLLOUT_STAGE = "shop.rollout_stage"  # 影子 ↔ 智能辅助切换：谁、门槛结果、是否强制
    QUALITY_SAMPLED = "quality.sampled"  # 每周抽样
    QUALITY_SCORED = "quality.scored"  # 复核打分 / 跳过
    QUALITY_VIEWED = "quality.viewed"  # 复核人查看推荐和买家问题（§9.2 查看原始对话要留访问记录）
    # 店铺授权与生命周期（§8.4 开通 / 续期 / 取消订购 / 授权过期，§12.4 授权过期告警）
    AUTH_EXPIRED = "auth.expired"
    AUTH_REVOKED = "auth.revoked"
    AUTH_GRANTED = "auth.granted"  # 开通 / 重新授权：权限、到期时间（不记令牌）
    AUTH_REFRESHED = "auth.refreshed"  # 续期成功：新的到期时间
    AUTH_REFRESH_FAILED = "auth.refresh_failed"  # 续期被拒（需要重新授权）：原因类型
    SHOP_STATUS_CHANGED = "shop.status_changed"  # trial / active / suspended 切换：谁、从哪到哪
    # 系统事件（shop_id 为空）：取消订购 / 过期清理之后留下的记录，detail 只有店铺 ID、原因和
    # 删除条数；店铺的数据删掉了，这条记录还在（AuditLog.list_system_events 读）
    SHOP_PURGED = "shop.purged"
    # 串店（§9.2、§12.4 串店检查）：只记在哪一步拦下的，不记任何数据
    TENANT_VIOLATION = "tenant.violation"
    # 上线节奏（§13.4）：复核人员登记的平台违规通知、买家投诉（周、条数，不记原文）
    ROLLOUT_INCIDENT = "shop.rollout_incident"
    # 合规扫描（§13.4）：实际发出去的文字里疑似违规（规则 ID、推荐 / 消息 ID，不记原文）
    COMPLIANCE_FLAG = "compliance.flag"
    # 复核人员确认 / 标记误报一条疑似违规（规则名、推荐 / 消息 ID、备注，不记原文）
    COMPLIANCE_REVIEWED = "compliance.reviewed"
    # 系统事件（shop_id 为空）：离线评测的结论（§13.5 上线前回归），带配置指纹
    EVAL_VERDICT = "eval.verdict"
    # 售后审核
    REVIEW_SUGGEST = "aftersale.suggest"
    REVIEW_TRANSITION = "aftersale.transition"
    REVIEW_CONFIRM = "aftersale.confirm"
    REVIEW_ACTION = "aftersale.action"
    # 大模型用量（§12.4 每个店铺的模型成本）
    LLM_USAGE = "llm.usage"


class ResultKey(StrEnum):
    """engine.result 事件 detail 的键：引擎（写）和报表（读）共用这一份。detail 里不放买家原文。"""

    INTENT = "intent"  # str | None：路由的问题类型
    ACTION = "action"  # str | None：RouteAction
    RULE = "rule"  # int | None：命中的 §6.6 规则序号
    SUGGESTIONS = "suggestions"  # int：推荐卡片数（影子模式也算）
    NO_SUGGESTION = "no_suggestion_reason"  # str | None：EngineResult.no_suggestion_reason
    QA_CHECKED = "qa_checked"  # int：QualityGate.check 次数（含重新生成后的复检）
    QA_FAILED = "qa_failed"  # int：其中不通过的次数
    QA_REASONS = "qa_reasons"  # list[str]：不通过的原因（已最小化）
    HARD_HITS = "hard_hits"  # list[str]：硬规则命中（每次一项，含 format / length 自动修正）
    DEGRADED = "degraded"  # list[str]：同 EngineResult.degraded
    TRANSFER = "transfer_to_human"  # bool
    LATENCY = "latency_ms"  # dict[str, int]：各环节耗时，端到端为 "total"
    MISSING = "missing"  # list[str]：同 EngineResult.missing_knowledge


RESULT_DETAIL_KEYS: tuple[str, ...] = tuple(k.value for k in ResultKey)
"""engine.result 的 detail 每次都有的键（按这个顺序）。"""

RESULT_BUDGET_EXCEEDED = "budget_exceeded"
"""engine.result detail 的可选键（bool，只写 True）：这条消息超过了时间预算
（LatencyConfig.message_budget_s）。只在超时的时候写，正常的消息 detail 仍然正好是
RESULT_DETAIL_KEYS；degraded 里同时有 DEGRADED_BUDGET。报表按这个常量读，没有这个键就是没超时。"""

OPTIONAL_RESULT_DETAIL_KEYS: tuple[str, ...] = (RESULT_BUDGET_EXCEEDED,)
"""engine.result detail 里可能出现、但不是每次都有的键。"""

SYSTEM_EVENT_ACTIONS: frozenset[str] = frozenset(
    {AuditAction.SHOP_PURGED.value, AuditAction.EVAL_VERDICT.value}
)
"""只以系统事件（shop_id 为空）记录的动作：不属于任何店铺，清理店铺数据时不删。"""

# EngineResult.degraded 的取值（除 "llm"、"decision"、"tool:*" 这些已有的之外）
DEGRADED_TENANT_ISOLATION = "tenant_isolation"
"""处理中发现串店（TenantIsolationError）：不推荐，另记一条 tenant.violation。"""
DEGRADED_BUDGET = "budget"
"""超过每条消息的时间预算（LatencyConfig.message_budget_s）：没做完的步骤丢掉。"""

# EngineResult.latency_ms 的键（除各环节和 "total" 之外）
LATENCY_SINCE_RECEIVED = "since_received"
"""从平台收到消息（StandardMessage.received_at）到出结果的耗时：含排队、会话锁等待（§12.1）。"""


class InMemoryAuditSink:
    """实现 core.ports.AuditLog。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[AuditEvent] = []
        self.decisions: list[DecisionLogEntry] = []

    def record(self, event: AuditEvent) -> None:
        with self._lock:
            self.events.append(event)

    def log_decision(self, entry: DecisionLogEntry) -> None:
        with self._lock:
            self.decisions.append(entry)

    def list_events(
        self, shop_id: str, *, action: str | None = None, since: datetime | None = None
    ) -> list[AuditEvent]:
        with self._lock:
            return [
                e
                for e in self.events
                if e.shop_id == shop_id
                and (action is None or e.action == action)
                and (since is None or e.created_at >= since)
            ]

    def list_decisions(
        self,
        shop_id: str,
        *,
        stage: DecisionStage | None = None,
        since: datetime | None = None,
    ) -> list[DecisionLogEntry]:
        with self._lock:
            return [
                d
                for d in self.decisions
                if d.shop_id == shop_id
                and (stage is None or d.stage == stage)
                and (since is None or d.created_at >= since)
            ]

    def list_system_events(
        self, *, action: str | None = None, since: datetime | None = None
    ) -> list[AuditEvent]:
        with self._lock:
            found = [
                e
                for e in self.events
                if e.shop_id is None
                and (action is None or e.action == action)
                and (since is None or e.created_at >= since)
            ]
        # 按时间升序，同一时间按写入顺序（sorted 是稳定的），和 PostgreSQL 实现一致
        return sorted(found, key=lambda e: e.created_at)

    def purge_shop(self, shop_id: str) -> int:
        with self._lock:
            before = len(self.events) + len(self.decisions)
            self.events = [e for e in self.events if e.shop_id != shop_id]
            self.decisions = [d for d in self.decisions if d.shop_id != shop_id]
            return before - len(self.events) - len(self.decisions)

    def actions(self, shop_id: str | None = None) -> list[str]:
        """按时间顺序列出 action，便于测试断言。"""
        return [e.action for e in self.events if shop_id is None or e.shop_id == shop_id]
