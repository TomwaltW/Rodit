"""Database 的语句超时 / 等锁超时在每个事务里生效，知识库被锁住时按故障降级（§12.2）。"""

from __future__ import annotations

import threading
import time

import pytest

from rodit.core.enums import KnowledgeSource, KnowledgeStatus
from rodit.core.errors import KnowledgeUnavailable
from rodit.core.models import KnowledgeItem
from rodit.storage import Database, PgKnowledgeRepository
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A

pytestmark = pytest.mark.pg


def test_session_applies_statement_and_lock_timeouts(pg_dsn):
    db = Database(pg_dsn, statement_timeout_ms=1500, lock_timeout_ms=250)
    with db.session(SHOP_A) as conn:
        assert conn.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "1500ms"
        assert conn.execute("SHOW lock_timeout").fetchone()["lock_timeout"] == "250ms"
    with Database(pg_dsn).session(SHOP_A) as conn:  # 不设就是库的默认值（不限时）
        assert conn.execute("SHOW lock_timeout").fetchone()["lock_timeout"] == "0"


def test_lock_wait_becomes_knowledge_unavailable(pg_dsn, admin_db, truncate):
    repo = PgKnowledgeRepository(
        Database(pg_dsn, statement_timeout_ms=5000, lock_timeout_ms=200), clock=FakeClock()
    )
    k = repo.add_knowledge(
        KnowledgeItem(
            shop_id=SHOP_A,
            question="能改地址吗",
            answer="亲，发货前可以改哦~",
            source=KnowledgeSource.ADOPTED,
            status=KnowledgeStatus.CANDIDATE,
        )
    )
    locked, release = threading.Event(), threading.Event()

    def hold_lock() -> None:  # 模拟一个长事务（如清理店铺数据）锁住这一行
        with admin_db.session(None) as conn:
            conn.execute("SELECT id FROM knowledge_item WHERE id = %s FOR UPDATE", (k.id,))
            locked.set()
            release.wait(10)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert locked.wait(5)
        started = time.monotonic()
        with pytest.raises(KnowledgeUnavailable):
            repo.set_knowledge_status(SHOP_A, k.id, KnowledgeStatus.APPROVED)
        assert time.monotonic() - started < 3  # 等锁 0.2 秒就放弃，不会一直挂着
    finally:
        release.set()
        holder.join()
    assert repo.get_knowledge(SHOP_A, k.id).status == KnowledgeStatus.CANDIDATE
