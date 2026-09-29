"""NUL 等 C0 控制字符在模型边界去掉（换行、制表符保留）：内存实现和 PostgreSQL 实现写入结果一致。

PostgreSQL 的 TEXT / JSONB 不接受 NUL（\\x00），内存实现照单全收；不在模型边界统一处理，
同一条买家消息在数据库部署里写不进去（消息去重、多轮上下文、推荐记录都丢），内存部署却正常。
"""

from __future__ import annotations

from rodit.core.audit import AuditAction
from rodit.core.enums import DecisionStage, Platform
from rodit.core.models import (
    AuditEvent,
    DecisionLogEntry,
    Shop,
    StoredMessage,
    strip_control_chars,
)
from rodit.inmemory import (
    InMemoryAfterSaleReviewRepository,
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemoryQualityReviewRepository,
    InMemoryShopAuthRepository,
    InMemorySuggestionRepository,
)
from rodit.testing import contracts
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A, build_demo_world

NOW = build_demo_world().now


def test_inmemory_text_sanitation_contract():
    contracts.check_text_sanitation(
        lambda: contracts.RepoSet(
            knowledge=InMemoryKnowledgeRepository(FakeClock()),
            conversations=InMemoryConversationRepository(),
            suggestions=InMemorySuggestionRepository(),
            feedback=InMemoryFeedbackRepository(),
            reviews=InMemoryAfterSaleReviewRepository(),
            auth=InMemoryShopAuthRepository(),
            quality_reviews=InMemoryQualityReviewRepository(),
        )
    )


def test_strip_control_chars_keeps_newline_and_tab():
    assert strip_control_chars(contracts.DIRTY_TEXT) == contracts.CLEAN_TEXT
    assert strip_control_chars("".join(chr(i) for i in range(32))) == "\t\n"
    assert strip_control_chars("正常文字 abc，123！") == "正常文字 abc，123！"
    assert strip_control_chars("") == ""
    once = strip_control_chars(contracts.DIRTY_TEXT)
    assert strip_control_chars(once) == once


def test_json_fields_are_sanitized_recursively():
    event = AuditEvent(
        shop_id=SHOP_A,
        actor="system",
        action=AuditAction.FEEDBACK_SENT.value,
        detail={"q": "a\x00b", "list": ["c\x01d", {"k\x00": "e\x1ff"}], "n": 3, "none": None},
        created_at=NOW,
    )
    assert event.detail == {"q": "ab", "list": ["cd", {"k": "ef"}], "n": 3, "none": None}
    entry = DecisionLogEntry(
        shop_id=SHOP_A,
        stage=DecisionStage.ROUTE,
        provider="rules",
        questions_ver="v1",
        answers={"intent": {"choice": "size\x00_advice"}},
        created_at=NOW,
    )
    assert entry.answers == {"intent": {"choice": "size_advice"}}
    shop = Shop(id=SHOP_A, platform=Platform.TAOBAO, config_overrides={"display_name": "小\x00雅"})
    assert shop.config_overrides == {"display_name": "小雅"}


def test_stored_message_text_is_sanitized():
    m = StoredMessage(
        msg_id="m_1",
        shop_id=SHOP_A,
        conversation_id="c_1",
        role="agent",
        text="亲\x00，在的",
        created_at=NOW,
    )
    assert m.text == "亲，在的"
