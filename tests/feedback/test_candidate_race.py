"""候选知识去重跨实例（§5.8）：两个实例几乎同时把同样的编辑回复变成候选知识。

FeedbackService 先查"同样的候选是否已在等审核"，没有才新建；进程内的锁管不到别的实例，
两边的检查都可能读到"没有"。存储保证同一条候选只建一次（add_knowledge 抛 ConcurrencyConflict），
FeedbackService 接住冲突、重新读到先建的那条并归到它上面——和今天读到已有候选时一样：
返回它、审计里记它的 id，不报错、推荐照常记为编辑后发送。
这里用"第一次查询读到过时结果"的知识仓储模拟另一个实例刚提交的候选还没被看到。
"""

from __future__ import annotations

from typing import Any

import pytest

from rodit.core import evidence as ev
from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.enums import DraftKind, Intent, KnowledgeStatus, SuggestionStatus
from rodit.core.errors import ConcurrencyConflict
from rodit.core.models import KnowledgeItem, SuggestionRecord
from rodit.feedback import FeedbackService
from rodit.inmemory import (
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemorySuggestionRepository,
)
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import AGENT, CONV, SHOP_A

EDITED = "亲，这款面料是95%棉加5%氨纶哦"


class _StaleCandidates:
    """包一层知识仓储：前 stale 次查候选时读到"没有"（另一个实例的写入还没被看到）。"""

    def __init__(self, inner: InMemoryKnowledgeRepository, *, stale: int = 1) -> None:
        self._inner = inner
        self.stale = stale
        self.adds = 0

    def list_knowledge(self, shop_id: str, **kw: Any) -> list[KnowledgeItem]:
        if kw.get("status") == KnowledgeStatus.CANDIDATE and self.stale > 0:
            self.stale -= 1
            return []
        return self._inner.list_knowledge(shop_id, **kw)

    def add_knowledge(self, item: KnowledgeItem) -> KnowledgeItem:
        self.adds += 1
        return self._inner.add_knowledge(item)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _record(clock: FakeClock, msg_id: str) -> SuggestionRecord:
    return SuggestionRecord(
        shop_id=SHOP_A,
        conversation_id=CONV,
        msg_id=msg_id,
        question_text="这款是什么面料",
        item_id="i_555",
        intent=Intent.PRODUCT_SPEC,
        text="亲，这款是棉的哦",
        kind=DraftKind.LLM,
        evidence_ids=[ev.item_attr("i_555", "面料")],
        status=SuggestionStatus.SHOWN,
        created_at=clock.now(),
    )


def _setup(stale: int) -> tuple[FeedbackService, FeedbackService, Any, Any, InMemoryAuditSink]:
    clock = FakeClock()
    suggestions = InMemorySuggestionRepository()
    knowledge = InMemoryKnowledgeRepository(clock)
    audit = InMemoryAuditSink()
    feedback = InMemoryFeedbackRepository()
    first = FeedbackService(suggestions, knowledge, feedback, audit, clock)
    view = _StaleCandidates(knowledge, stale=stale)
    second = FeedbackService(suggestions, view, feedback, audit, clock)  # 另一个实例（另一把锁）
    return first, second, suggestions, view, audit


def test_conflicting_candidate_from_another_instance_is_reused():
    first, second, suggestions, view, audit = _setup(stale=1)
    a = suggestions.add(_record(FakeClock(), "m_a"))
    b = suggestions.add(_record(FakeClock(), "m_b"))
    made = first.record_edited(SHOP_A, a.id, agent_id=AGENT, edited_text=EDITED)
    assert made.candidate is not None

    out = second.record_edited(SHOP_A, b.id, agent_id=AGENT, edited_text=EDITED)

    assert view.adds == 1  # 检查读到过时结果，确实去新建了，被存储挡下
    assert out.candidate is not None and out.candidate.id == made.candidate.id
    assert out.candidate_skipped is None
    assert out.record.status == SuggestionStatus.EDITED_SENT
    pending = view.list_knowledge(SHOP_A, status=KnowledgeStatus.CANDIDATE)
    assert [k.id for k in pending] == [made.candidate.id]
    linked = audit.list_events(SHOP_A, action=AuditAction.KNOWLEDGE_CANDIDATE.value)
    assert [e.detail["knowledge_id"] for e in linked] == [made.candidate.id] * 2
    edited = audit.list_events(SHOP_A, action=AuditAction.FEEDBACK_EDITED.value)
    assert [e.detail["candidate_id"] for e in edited] == [made.candidate.id] * 2


def test_conflict_that_never_resolves_rolls_feedback_back():
    """一直冲突、又一直读不到那条候选（极端情况）：不无限重试，按写失败处理——推荐退回展示中，
    客服可以重试。"""
    first, second, suggestions, view, _ = _setup(stale=99)
    a = suggestions.add(_record(FakeClock(), "m_a"))
    b = suggestions.add(_record(FakeClock(), "m_b"))
    first.record_edited(SHOP_A, a.id, agent_id=AGENT, edited_text=EDITED)

    with pytest.raises(ConcurrencyConflict):
        second.record_edited(SHOP_A, b.id, agent_id=AGENT, edited_text=EDITED)

    assert 1 < view.adds <= 3
    assert suggestions.get(SHOP_A, b.id).status == SuggestionStatus.SHOWN
