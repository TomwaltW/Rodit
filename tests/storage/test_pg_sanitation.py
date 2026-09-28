"""PostgreSQL 仓储跑文本清洗契约（NUL 等控制字符），结果和内存实现一致。"""

from __future__ import annotations

import pytest

from rodit.core.models import AuditEvent
from rodit.storage import (
    PgAfterSaleReviewRepository,
    PgAuditLog,
    PgConversationRepository,
    PgFeedbackRepository,
    PgKnowledgeRepository,
    PgShopAuthRepository,
    PgSuggestionRepository,
)
from rodit.testing import contracts
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A, build_demo_world

pytestmark = pytest.mark.pg


def test_pg_text_sanitation_contract(db, truncate, token_key):
    def make() -> contracts.RepoSet:
        truncate()
        return contracts.RepoSet(
            knowledge=PgKnowledgeRepository(db, clock=FakeClock()),
            conversations=PgConversationRepository(db),
            suggestions=PgSuggestionRepository(db),
            feedback=PgFeedbackRepository(db),
            reviews=PgAfterSaleReviewRepository(db),
            auth=PgShopAuthRepository(db, encryption_key=token_key),
        )

    contracts.check_text_sanitation(make)


def test_pg_audit_detail_with_nul_is_stored(db, truncate):
    audit = PgAuditLog(db)
    now = build_demo_world().now
    audit.record(
        AuditEvent(
            shop_id=SHOP_A,
            actor="kf_01",
            action="feedback.sent",
            detail={"note": "a\x00b", "nested": ["c\x00"]},
            created_at=now,
        )
    )
    assert [e.detail for e in audit.list_events(SHOP_A)] == [{"note": "ab", "nested": ["c"]}]
