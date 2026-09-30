"""撤回先于消息到达（§8.3 平台推送乱序、§5.3 撤回的消息不再进上下文）在 PostgreSQL 上：

- 两个实例同时处理撤回和被撤回的消息：不管哪边的事务先写、后提交，消息都不能留在会话里
  （按 (店铺, msg_id) 的事务级 advisory lock 串行，后做的一方看得到先提交的结果）；
- 丢弃的消息只占住 msg_id，库里没有它的文字和卡片（原文不留）；
- 墓碑有界：过期的在记新墓碑时顺带删掉。
全部是合成数据。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from rodit.core.ports import RECALL_TOMBSTONE_TTL
from rodit.storage import Database, PgConversationRepository
from rodit.testing.fakes import FakeClock, make_message
from rodit.testing.fixtures import SHOP_A, build_demo_world

pytestmark = pytest.mark.pg

CONV = "conv_recall_race"


class _PausingConn:
    """执行完第一条含 pattern 的语句后停下（事务还没提交），等 release。"""

    def __init__(self, conn: Any, owner: _PausingDatabase) -> None:
        self._conn = conn
        self._owner = owner

    def execute(self, query: Any, params: Any = None) -> Any:
        cur = self._conn.execute(query, params)
        owner = self._owner
        if owner.pattern in str(query) and not owner.paused.is_set():
            owner.paused.set()
            assert owner.release.wait(10)
        return cur

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _PausingDatabase(Database):
    """模拟另一个实例：它的事务写到一半还没提交。"""

    def __init__(self, dsn: str, *, pattern: str) -> None:
        super().__init__(dsn)
        self.pattern = pattern
        self.paused = threading.Event()
        self.release = threading.Event()

    @contextmanager
    def session(self, shop_id: str | None, **kwargs: Any) -> Iterator[Any]:
        with super().session(shop_id, **kwargs) as conn:
            yield _PausingConn(conn, self)


def _run(fn: Any, out: dict[str, Any], key: str) -> threading.Thread:
    def target() -> None:
        out[key] = fn()

    thread = threading.Thread(target=target)
    thread.start()
    return thread


def _tombstones(admin_db: Database, msg_id: str) -> int:
    with admin_db.session(None) as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM message_recall WHERE msg_id = %s", (msg_id,)
        ).fetchone()
    return row["n"]


def test_recall_racing_an_uncommitted_message_still_recalls_it(pg_dsn, admin_db, truncate):
    """实例 1 正在写这条消息（已插入、未提交），实例 2 收到撤回：撤回要等它提交后再撤。"""
    clock = FakeClock()
    msg = make_message("有M码吗", world=build_demo_world(), msg_id="m_race1", conversation_id=CONV)
    slow = _PausingDatabase(pg_dsn, pattern="INSERT INTO message (")
    writer = PgConversationRepository(slow, clock=clock)
    other = PgConversationRepository(Database(pg_dsn), clock=clock)
    out: dict[str, Any] = {}

    t1 = _run(lambda: writer.add_message(msg), out, "added")
    assert slow.paused.wait(5)
    t2 = _run(lambda: other.recall_message(SHOP_A, CONV, "m_race1"), out, "recalled")
    t2.join(0.3)
    assert t2.is_alive()  # 等实例 1 的事务结束，而不是当成"没见过"记墓碑
    slow.release.set()
    t1.join(5)
    t2.join(5)

    assert out == {"added": True, "recalled": True}
    assert other.recent_messages(SHOP_A, CONV) == []
    assert _tombstones(admin_db, "m_race1") == 0


def test_message_racing_an_uncommitted_recall_is_dropped(pg_dsn, admin_db, truncate):
    """实例 1 正在记墓碑（已插入、未提交），实例 2 收到这条消息：要等墓碑提交后丢弃消息。"""
    clock = FakeClock()
    msg = make_message(
        "我的手机号13812345678，有M码吗",
        world=build_demo_world(),
        msg_id="m_race2",
        conversation_id=CONV,
    )
    slow = _PausingDatabase(pg_dsn, pattern="INSERT INTO message_recall")
    recaller = PgConversationRepository(slow, clock=clock)
    other = PgConversationRepository(Database(pg_dsn), clock=clock)
    out: dict[str, Any] = {}

    t1 = _run(lambda: recaller.recall_message(SHOP_A, CONV, "m_race2"), out, "recalled")
    assert slow.paused.wait(5)
    t2 = _run(lambda: other.add_message(msg), out, "added")
    t2.join(0.3)
    assert t2.is_alive()
    slow.release.set()
    t1.join(5)
    t2.join(5)

    assert out == {"recalled": False, "added": False}  # 丢弃：不保存、不回答
    assert other.recent_messages(SHOP_A, CONV) == []
    assert _tombstones(admin_db, "m_race2") == 0  # 墓碑已用掉
    with admin_db.session(None) as conn:  # 只占住 msg_id：没有文字和卡片
        [row] = conn.execute(
            "SELECT text, item_id, order_ref FROM message WHERE msg_id = 'm_race2'"
        ).fetchall()
    assert row == {"text": None, "item_id": None, "order_ref": None}


def test_expired_tombstones_are_cleaned_up_when_new_ones_are_recorded(pg_dsn, admin_db, truncate):
    clock = FakeClock()
    repo = PgConversationRepository(Database(pg_dsn), clock=clock)
    for i in range(5):
        assert repo.recall_message(SHOP_A, CONV, f"m_old{i}") is False
    assert _count_all(admin_db) == 5
    clock.advance(seconds=RECALL_TOMBSTONE_TTL.total_seconds() + 1)
    assert repo.recall_message(SHOP_A, CONV, "m_new") is False
    assert _count_all(admin_db) == 1  # 过期的 5 条已删
    # 过期的墓碑再收到同一条撤回：重新计时（不能拿着过期的墓碑当已经记过）
    repo.recall_message(SHOP_A, CONV, "m_again")
    clock.advance(seconds=RECALL_TOMBSTONE_TTL.total_seconds() + 1)
    repo.recall_message(SHOP_A, CONV, "m_again")
    clock.advance(hours=1)
    late = make_message("在吗", world=build_demo_world(), msg_id="m_again", conversation_id=CONV)
    assert repo.add_message(late) is False


def _count_all(admin_db: Database) -> int:
    with admin_db.session(None) as conn:
        return conn.execute("SELECT count(*) AS n FROM message_recall").fetchone()["n"]
