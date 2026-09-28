"""审计的内存实现（开发和测试用；生产用 storage 里的 PostgreSQL 实现）。"""

from __future__ import annotations

import threading

from rodit.core.models import AuditEvent, DecisionLogEntry


class InMemoryAuditSink:
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

    def actions(self, shop_id: str | None = None) -> list[str]:
        """按时间顺序列出 action，便于测试断言。"""
        return [e.action for e in self.events if shop_id is None or e.shop_id == shop_id]
