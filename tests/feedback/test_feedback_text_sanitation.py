"""客服编辑后的文字里有 NUL 等控制字符：推荐记录（model_copy 不做校验）和候选知识都要清洗掉，
否则内存部署照常、数据库部署写不进去（PostgreSQL 的 TEXT 不接受 NUL）。"""

from __future__ import annotations

import pytest

from rodit.core.audit import InMemoryAuditSink
from rodit.core.enums import DraftKind, Intent, SuggestionStatus
from rodit.core.models import SuggestionRecord
from rodit.feedback import FeedbackService
from rodit.inmemory import (
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemorySuggestionRepository,
)
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import AGENT, CONV, SHOP_A


def _setup() -> tuple[FeedbackService, InMemorySuggestionRepository, SuggestionRecord]:
    clock = FakeClock()
    suggestions = InMemorySuggestionRepository()
    service = FeedbackService(
        suggestions,
        InMemoryKnowledgeRepository(clock),
        InMemoryFeedbackRepository(),
        InMemoryAuditSink(),
        clock,
    )
    rec = suggestions.add(
        SuggestionRecord(
            shop_id=SHOP_A,
            conversation_id=CONV,
            msg_id="m_1",
            question_text="这款会起球吗",
            intent=Intent.PRODUCT_SPEC,
            text="亲，不容易起球哦",
            kind=DraftKind.LLM,
            status=SuggestionStatus.SHOWN,
            created_at=clock.now(),
        )
    )
    return service, suggestions, rec


def test_edited_text_control_chars_are_stripped():
    service, suggestions, rec = _setup()
    out = service.record_edited(
        SHOP_A, rec.id, agent_id=AGENT, edited_text="亲\x00，正常穿着\x07不容易起球哦\n"
    )
    assert out.record.edited_text == "亲，正常穿着不容易起球哦"
    assert suggestions.get(SHOP_A, rec.id).edited_text == "亲，正常穿着不容易起球哦"
    assert out.candidate is not None and out.candidate.answer == "亲，正常穿着不容易起球哦"


def test_edited_text_of_only_control_chars_is_empty():
    service, suggestions, rec = _setup()
    with pytest.raises(ValueError):
        service.record_edited(SHOP_A, rec.id, agent_id=AGENT, edited_text="\x00\x01 ")
    assert suggestions.get(SHOP_A, rec.id).status == SuggestionStatus.SHOWN
