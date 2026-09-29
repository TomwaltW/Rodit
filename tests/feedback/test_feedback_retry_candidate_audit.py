"""编辑后发送写到一半失败、客服重试（§5.8、§4.1 原则 6 一切留痕）：候选知识不能没有审计。

第一次候选知识已经写进去，审计 feedback.edited_sent 失败，推荐退回展示中；重试时候选知识按内容
去重找到上次那条——这次反馈就归到它上面（返回它、审计里记上它的 id、补上 knowledge.candidate），
不能说成"重复、没生成"，让商家审核队列里多一条查不到来历的候选。
"""

from __future__ import annotations

from typing import Any

import pytest

from rodit.core import evidence as ev
from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.enums import DraftKind, Intent, KnowledgeStatus, SuggestionStatus
from rodit.core.errors import StorageUnavailable
from rodit.core.models import AuditEvent, SuggestionRecord
from rodit.feedback import FeedbackService
from rodit.inmemory import (
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemorySuggestionRepository,
)
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import AGENT, CONV, SHOP_A


class _AuditFailsOnce(InMemoryAuditSink):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next = True

    def record(self, event: AuditEvent) -> None:
        if self.fail_next:
            self.fail_next = False
            raise StorageUnavailable("数据库连接失败")
        super().record(event)


def _record(clock: FakeClock, msg_id: str = "m_1") -> SuggestionRecord:
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


def test_retry_after_audit_failure_links_the_candidate_left_by_the_first_attempt():
    clock = FakeClock()
    suggestions = InMemorySuggestionRepository()
    knowledge = InMemoryKnowledgeRepository(clock)
    audit = _AuditFailsOnce()
    service = FeedbackService(suggestions, knowledge, InMemoryFeedbackRepository(), audit, clock)
    rec = suggestions.add(_record(clock))
    edited = "亲，这款面料是95%棉加5%氨纶哦"

    def edit() -> Any:
        return service.record_edited(SHOP_A, rec.id, agent_id=AGENT, edited_text=edited)

    with pytest.raises(StorageUnavailable):
        edit()
    assert suggestions.get(SHOP_A, rec.id).status == SuggestionStatus.SHOWN
    out = edit()  # 客服重试

    [candidate] = knowledge.list_knowledge(SHOP_A, status=KnowledgeStatus.CANDIDATE)
    assert out.candidate is not None and out.candidate.id == candidate.id
    assert out.candidate_skipped is None
    [edited_event] = audit.list_events(SHOP_A, action=AuditAction.FEEDBACK_EDITED.value)
    assert edited_event.detail["candidate_id"] == candidate.id
    assert edited_event.detail["candidate_skipped"] is None
    [linked] = audit.list_events(SHOP_A, action=AuditAction.KNOWLEDGE_CANDIDATE.value)
    assert linked.detail["knowledge_id"] == candidate.id
    assert linked.detail["suggestion_id"] == rec.id
