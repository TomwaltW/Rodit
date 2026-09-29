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
    # 店铺授权（§8.4 取消订购 / 授权过期，§12.4 授权过期告警）
    AUTH_EXPIRED = "auth.expired"
    AUTH_REVOKED = "auth.revoked"
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

    def purge_shop(self, shop_id: str) -> int:
        with self._lock:
            before = len(self.events) + len(self.decisions)
            self.events = [e for e in self.events if e.shop_id != shop_id]
            self.decisions = [d for d in self.decisions if d.shop_id != shop_id]
            return before - len(self.events) - len(self.decisions)

    def actions(self, shop_id: str | None = None) -> list[str]:
        """按时间顺序列出 action，便于测试断言。"""
        return [e.action for e in self.events if shop_id is None or e.shop_id == shop_id]
