"""PostgreSQL 仓储跑共用的契约测试（rodit.testing.contracts），每个检查前清空所有业务表。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest

from rodit.storage import (
    PgAfterSaleReviewRepository,
    PgConversationRepository,
    PgFeedbackRepository,
    PgKnowledgeRepository,
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


def test_aftersale_review_repository_contract(db, truncate):
    contracts.check_aftersale_review_repository(
        fresh(truncate, lambda: PgAfterSaleReviewRepository(db))
    )


def test_shop_auth_repository_contract(db, truncate, token_key):
    contracts.check_shop_auth_repository(
        fresh(truncate, lambda: PgShopAuthRepository(db, encryption_key=token_key))
    )


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
            ),
        )
    )
