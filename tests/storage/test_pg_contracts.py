"""PostgreSQL 仓储跑共用的契约测试（rodit.testing.contracts），每个检查前清空所有业务表。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest

from rodit.storage import (
    Database,
    PgAfterSaleReviewRepository,
    PgAuditLog,
    PgConversationRepository,
    PgFeedbackRepository,
    PgKnowledgeRepository,
    PgQualityReviewRepository,
    PgShopAuthRepository,
    PgSuggestionRepository,
)
from rodit.testing import contracts
from rodit.testing.fakes import FakeClock

pytestmark = pytest.mark.pg

T = TypeVar("T")


def fresh(truncate: Callable[[], None], build: Callable[[], T]) -> Callable[[], T]:
    def make() -> T:
        truncate()
        return build()

    return make


def test_knowledge_repository_contract(db, truncate):
    contracts.check_knowledge_repository(
        fresh(truncate, lambda: PgKnowledgeRepository(db, clock=FakeClock()))
    )


def test_conversation_repository_contract(db, truncate):
    contracts.check_conversation_repository(fresh(truncate, lambda: PgConversationRepository(db)))


def test_suggestion_repository_contract(db, truncate):
    contracts.check_suggestion_repository(fresh(truncate, lambda: PgSuggestionRepository(db)))


def test_feedback_repository_contract(db, truncate):
    contracts.check_feedback_repository(fresh(truncate, lambda: PgFeedbackRepository(db)))


def test_quality_review_repository_contract(db, truncate):
    contracts.check_quality_review_repository(
        fresh(truncate, lambda: PgQualityReviewRepository(db))
    )


def test_aftersale_review_repository_contract(db, truncate):
    contracts.check_aftersale_review_repository(
        fresh(truncate, lambda: PgAfterSaleReviewRepository(db))
    )


def test_shop_auth_repository_contract(db, truncate, token_key):
    contracts.check_shop_auth_repository(
        fresh(truncate, lambda: PgShopAuthRepository(db, encryption_key=token_key))
    )


def test_shop_auth_refresh_fields_contract(db, truncate, token_key):
    contracts.check_shop_auth_refresh_fields(
        fresh(truncate, lambda: PgShopAuthRepository(db, encryption_key=token_key))
    )


def test_mark_needs_reauth_contract(db, truncate, token_key):
    contracts.check_mark_needs_reauth(
        fresh(truncate, lambda: PgShopAuthRepository(db, encryption_key=token_key))
    )


def test_expiring_shop_ids_contract(db, truncate, token_key):
    contracts.check_expiring_shop_ids(
        fresh(truncate, lambda: PgShopAuthRepository(db, encryption_key=token_key))
    )


def test_list_outgoing_contract(db, truncate):
    contracts.check_list_outgoing(fresh(truncate, lambda: PgConversationRepository(db)))


def test_system_events_contract(db, truncate):
    contracts.check_system_events(fresh(truncate, lambda: PgAuditLog(db)))


def test_cross_shop_isolation(db, truncate, token_key):
    contracts.check_cross_shop_isolation(
        fresh(
            truncate,
            lambda: contracts.RepoSet(
                knowledge=PgKnowledgeRepository(db, clock=FakeClock()),
                conversations=PgConversationRepository(db),
                suggestions=PgSuggestionRepository(db),
                feedback=PgFeedbackRepository(db),
                reviews=PgAfterSaleReviewRepository(db),
                auth=PgShopAuthRepository(db, encryption_key=token_key),
                quality_reviews=PgQualityReviewRepository(db),
                audit=PgAuditLog(db),
            ),
        )
    )


def test_recall_before_message_contract(db, truncate):
    contracts.check_recall_before_message(
        lambda clock: fresh(truncate, lambda: PgConversationRepository(db, clock=clock))()
    )


def test_candidate_uniqueness_contract(db, truncate):
    contracts.check_candidate_uniqueness(
        fresh(truncate, lambda: PgKnowledgeRepository(db, clock=FakeClock()))
    )


def test_purge_is_idempotent_contract(pg_dsn, truncate, token_key):
    """清理按 2 行一批：每张表都要分好几个事务删完，结果和内存实现一样。"""
    small = Database(pg_dsn, purge_batch_rows=2)
    contracts.check_purge_is_idempotent(
        fresh(
            truncate,
            lambda: contracts.RepoSet(
                knowledge=PgKnowledgeRepository(small, clock=FakeClock()),
                conversations=PgConversationRepository(small),
                suggestions=PgSuggestionRepository(small),
                feedback=PgFeedbackRepository(small),
                reviews=PgAfterSaleReviewRepository(small),
                auth=PgShopAuthRepository(small, encryption_key=token_key),
                quality_reviews=PgQualityReviewRepository(small),
            ),
        )
    )


def test_atomic_shop_and_card_updates_contract(db, truncate):
    contracts.check_atomic_shop_and_card_updates(
        fresh(truncate, lambda: PgKnowledgeRepository(db, clock=FakeClock()))
    )


def test_policy_delete_contract(db, truncate):
    contracts.check_policy_delete(
        fresh(truncate, lambda: PgKnowledgeRepository(db, clock=FakeClock()))
    )
