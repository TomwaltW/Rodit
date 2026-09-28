"""多实例部署：两个商家子账号在不同实例上同时审核同一条候选知识（§5.8 只有商家审核过的才入库）。

每个实例的 FeedbackService 只有进程内的锁；共享的知识库必须做比较后更新，
否则"读到候选 → 另一实例驳回 → 这边审核通过"会把被驳回的候选变成已审核、拿去回答买家。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.enums import KnowledgeSource, KnowledgeStatus
from rodit.core.errors import ActionNotAllowed
from rodit.core.models import KnowledgeItem
from rodit.feedback import KNOWLEDGE_RETIRED, FeedbackService
from rodit.inmemory import (
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemorySuggestionRepository,
)
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A


class _ReadThenInterleave:
    """包一层知识库：get_knowledge 读完之后先让"另一个实例"做一次操作，再把读到的结果返回。"""

    def __init__(self, inner: InMemoryKnowledgeRepository) -> None:
        self._inner = inner
        self.after_read: Callable[[], Any] | None = None

    def get_knowledge(self, shop_id: str, knowledge_id: int) -> KnowledgeItem | None:
        item = self._inner.get_knowledge(shop_id, knowledge_id)
        hook, self.after_read = self.after_read, None
        if hook is not None:
            hook()
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class Cluster:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.knowledge = InMemoryKnowledgeRepository(self.clock)
        self.audit = InMemoryAuditSink()
        self.approved: list[KnowledgeItem] = []
        self.racing = _ReadThenInterleave(self.knowledge)
        self.instance_1 = self._service(self.racing)
        self.instance_2 = self._service(self.knowledge)
        self.candidate = self.knowledge.add_knowledge(
            KnowledgeItem(
                shop_id=SHOP_A,
                question="能改地址吗",
                answer="亲，发货前可以改哦~",
                source=KnowledgeSource.ADOPTED,
                status=KnowledgeStatus.CANDIDATE,
            )
        )

    def _service(self, knowledge: Any) -> FeedbackService:
        return FeedbackService(
            InMemorySuggestionRepository(),
            knowledge,
            InMemoryFeedbackRepository(),
            self.audit,
            self.clock,
            on_approved=self.approved.append,
        )

    def actions(self) -> list[str]:
        return [e.action for e in self.audit.events]


def test_retire_on_other_instance_wins_over_stale_approve():
    c = Cluster()
    kid = c.candidate.id
    c.racing.after_read = lambda: c.instance_2.retire(SHOP_A, kid, merchant="sub_b")
    with pytest.raises(ActionNotAllowed):
        c.instance_1.approve_candidate(SHOP_A, kid, merchant="sub_a")
    assert c.knowledge.get_knowledge(SHOP_A, kid).status == KnowledgeStatus.RETIRED
    assert c.actions() == [KNOWLEDGE_RETIRED]
    assert c.approved == []


def test_approve_on_other_instance_wins_over_stale_retire():
    c = Cluster()
    kid = c.candidate.id
    c.racing.after_read = lambda: c.instance_2.approve_candidate(SHOP_A, kid, merchant="sub_b")
    with pytest.raises(ActionNotAllowed):
        c.instance_1.retire(SHOP_A, kid, merchant="sub_a")
    got = c.knowledge.get_knowledge(SHOP_A, kid)
    assert got.status == KnowledgeStatus.APPROVED and got.approved_by == "sub_b"
    assert c.actions() == [AuditAction.KNOWLEDGE_APPROVED.value]
    # 刷新后看到已审核，再下线是正常操作
    assert c.instance_1.retire(SHOP_A, kid, merchant="sub_a").status == KnowledgeStatus.RETIRED
