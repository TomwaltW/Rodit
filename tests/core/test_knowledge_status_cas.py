"""内存知识库跑 set_knowledge_status 比较后更新的契约（PostgreSQL 版见 tests/storage）。"""

from __future__ import annotations

from rodit.inmemory import InMemoryKnowledgeRepository
from rodit.testing import contracts
from rodit.testing.fakes import FakeClock


def test_inmemory_knowledge_status_compare_and_set():
    contracts.check_knowledge_status_compare_and_set(
        lambda: InMemoryKnowledgeRepository(FakeClock())
    )
