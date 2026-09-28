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
    # 售后审核
    REVIEW_SUGGEST = "aftersale.suggest"
    REVIEW_TRANSITION = "aftersale.transition"
    REVIEW_CONFIRM = "aftersale.confirm"
    REVIEW_ACTION = "aftersale.action"
    # 大模型用量（§12.4 每个店铺的模型成本）
    LLM_USAGE = "llm.usage"


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

    def actions(self, shop_id: str | None = None) -> list[str]:
        """按时间顺序列出 action，便于测试断言。"""
        return [e.action for e in self.events if shop_id is None or e.shop_id == shop_id]
