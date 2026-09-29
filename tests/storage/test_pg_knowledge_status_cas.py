"""PostgreSQL 知识库跑 set_knowledge_status 比较后更新的契约；两个事务真正并发时只有一个生效。"""

from __future__ import annotations

import threading

import pytest

from rodit.core.enums import KnowledgeSource, KnowledgeStatus
from rodit.core.errors import ConcurrencyConflict
from rodit.core.models import KnowledgeItem
from rodit.storage import PgKnowledgeRepository
from rodit.testing import contracts
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A

pytestmark = pytest.mark.pg


def test_pg_knowledge_status_compare_and_set(db, truncate):
    def make() -> PgKnowledgeRepository:
        truncate()
        return PgKnowledgeRepository(db, clock=FakeClock())

    contracts.check_knowledge_status_compare_and_set(make)


def test_concurrent_approve_and_retire_have_one_winner(db, truncate):
    repo = PgKnowledgeRepository(db, clock=FakeClock())
    k = repo.add_knowledge(
        KnowledgeItem(
            shop_id=SHOP_A,
            question="能改地址吗",
            answer="亲，发货前可以改哦~",
            source=KnowledgeSource.ADOPTED,
            status=KnowledgeStatus.CANDIDATE,
        )
    )
    barrier = threading.Barrier(2)
    results: list[str] = []

    def act(status: KnowledgeStatus) -> None:
        barrier.wait()
        try:
            repo.set_knowledge_status(
                SHOP_A, k.id, status, expected_status=KnowledgeStatus.CANDIDATE
            )
            results.append(status.value)
        except ConcurrencyConflict:
            results.append("conflict")

    threads = [
        threading.Thread(target=act, args=(s,))
        for s in (KnowledgeStatus.APPROVED, KnowledgeStatus.RETIRED)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results).count("conflict") == 1
    winner = next(r for r in results if r != "conflict")
    assert repo.get_knowledge(SHOP_A, k.id).status.value == winner
