"""§8.4 清理店铺数据不受请求级语句 / 等锁超时约束，请求路径仍然有上限（§12.1、§12.2）。

生产组装给请求用的 Database 设了 statement_timeout / lock_timeout（默认 2 秒 / 1 秒）。
取消订购时 AuthGuard.revoke 先删令牌，再逐个调用 purge_shop：大店铺一次 DELETE 几百万行，
在请求级上限下每次都被取消回滚，令牌已删、数据永远删不掉。这里用很小的超时 + 合成数据复现：
行数超过语句超时能删的量、清理时表被别的事务锁住一会儿，清理都要做完；普通请求照样超时。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import timedelta

import psycopg
import pytest
from psycopg import errors
from pydantic import SecretStr

from rodit.adapters.auth import AuthGuard
from rodit.app.production import postgres_repos, production_database
from rodit.core.models import ShopAuth
from rodit.storage import (
    Database,
    PgAfterSaleReviewRepository,
    PgAuditLog,
    PgConversationRepository,
    PgFeedbackRepository,
    PgKnowledgeRepository,
    PgSuggestionRepository,
)
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A, SHOP_B

pytestmark = pytest.mark.pg

BULK_ROWS = 100_000
"""本机删 10 万行约 40 毫秒，是下面 10 毫秒语句超时的 4 倍以上。"""


def _count(admin_db: Database, table: str, shop_id: str) -> int:
    with admin_db.session(None) as conn:
        row = conn.execute(
            f"SELECT count(*) AS n FROM {table} WHERE shop_id = %s", (shop_id,)
        ).fetchone()
    return row["n"]


def _bulk_load(admin_db: Database, shop_id: str, n: int) -> None:
    """管理员灌入合成的会话消息和审计记录（大店铺一两年的量级缩小版）。"""
    with admin_db.session(None) as conn:
        conn.execute(
            "INSERT INTO conversation (shop_id, id, buyer_ref, started_at)"
            " VALUES (%s, 'conv_bulk', 'buyer_ref_synthetic', now())",
            (shop_id,),
        )
        conn.execute(
            "INSERT INTO message (shop_id, msg_id, conversation_id, role, text, created_at)"
            " SELECT %s, 'm_' || g, 'conv_bulk', 'buyer', '这件有现货吗', now()"
            " FROM generate_series(1, %s) g",
            (shop_id, n),
        )
        conn.execute(
            "INSERT INTO audit_log (shop_id, actor, action, detail, created_at)"
            " SELECT %s, 'system', 'route', '{\"intent\": \"size\"}', now()"
            " FROM generate_series(1, %s) g",
            (shop_id, n),
        )


def test_unbounded_session_skips_request_timeouts_but_keeps_tenant_scope(pg_dsn):
    db = Database(pg_dsn, statement_timeout_ms=1500, lock_timeout_ms=250)
    with db.session(SHOP_A, unbounded=True) as conn:
        assert conn.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "0"
        assert conn.execute("SHOW lock_timeout").fetchone()["lock_timeout"] == "0"
        # 角色和店铺范围照常设置：清理仍然在行级安全下，只能删本店的数据
        assert conn.execute("SELECT current_user AS u").fetchone()["u"] == "rodit_app"
        scope = conn.execute("SELECT current_setting('rodit.shop_id') AS s").fetchone()["s"]
        assert scope == SHOP_A
    with db.session(SHOP_A) as conn:  # 默认（请求路径）仍然有上限
        assert conn.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "1500ms"
        assert conn.execute("SHOW lock_timeout").fetchone()["lock_timeout"] == "250ms"


def test_production_revoke_purges_more_rows_than_request_timeout_allows(
    pg_dsn, admin_db, truncate, token_key, monkeypatch
):
    """复现 purge-under-request-statement-timeout / purge-hits-request-statement-timeout：
    与 AssistService.revoke_shop 同一条调用链
    （生产组装的库 + AuthGuard.revoke + repos.purgers()）。"""
    monkeypatch.setenv("RODIT_DATABASE_DSN", pg_dsn)
    monkeypatch.setenv("RODIT_TOKEN_KEY", token_key)
    monkeypatch.setenv("RODIT_DB_STATEMENT_TIMEOUT_MS", "10")
    monkeypatch.setenv("RODIT_DB_LOCK_TIMEOUT_MS", "10")
    _bulk_load(admin_db, SHOP_A, BULK_ROWS)
    _bulk_load(admin_db, SHOP_B, 3)
    clock = FakeClock()
    db = production_database()
    assert (db.statement_timeout_ms, db.lock_timeout_ms) == (10, 10)
    repos = postgres_repos(db, clock=clock)
    repos.auth.save(
        ShopAuth(
            shop_id=SHOP_A,
            access_token=SecretStr("tok-synthetic-only"),
            scopes=["order.read"],
            expires_at=clock.now() + timedelta(days=30),
        )
    )

    counts = AuthGuard(repos.auth, clock).revoke(SHOP_A, repos.purgers())

    assert counts["auth"] == 1
    assert counts["PgConversationRepository"] == BULK_ROWS + 1  # 消息 + 会话
    assert counts["PgAuditLog"] == BULK_ROWS
    for table in ("message", "conversation", "audit_log", "shop_auth"):
        assert _count(admin_db, table, SHOP_A) == 0
    assert _count(admin_db, "message", SHOP_B) == 3  # 别的店铺不受影响
    assert _count(admin_db, "audit_log", SHOP_B) == 3
    # 请求路径仍然有上限：同一个库上的普通查询照样超时
    with pytest.raises(errors.QueryCanceled), db.session(SHOP_A) as conn:
        conn.execute("SELECT pg_sleep(0.2)")


PURGERS: list[tuple[Callable[[Database], object], str]] = [
    (lambda db: PgKnowledgeRepository(db, clock=FakeClock()), "knowledge_item"),
    (PgConversationRepository, "message"),
    (PgSuggestionRepository, "suggestion"),
    (PgFeedbackRepository, "labeled_sample"),
    (PgAfterSaleReviewRepository, "aftersale_review"),
    (PgAuditLog, "decision_log"),
]


@pytest.mark.parametrize(("make_repo", "table"), PURGERS, ids=[table for _, table in PURGERS])
def test_purge_waits_out_a_lock_longer_than_request_timeouts(
    pg_dsn, admin_db, truncate, make_repo, table
):
    """清理时表被别的事务（导出、迁移）锁住 0.4 秒：请求级等锁 50 毫秒 / 语句 150 毫秒就放弃，
    清理要等到锁释放后做完，不能在令牌已删之后半途失败。"""
    db = Database(pg_dsn, statement_timeout_ms=150, lock_timeout_ms=50)
    repo = make_repo(db)
    locked, release = threading.Event(), threading.Event()

    def hold_lock() -> None:  # SHARE 锁与 DELETE 要的 ROW EXCLUSIVE 冲突
        with admin_db.session(None) as conn:
            conn.execute(f"LOCK TABLE {table} IN SHARE MODE")
            locked.set()
            release.wait(10)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert locked.wait(5)
        # 请求路径：同样的 DELETE 等锁 50 毫秒就报错
        with pytest.raises(psycopg.OperationalError), db.session(SHOP_A) as conn:
            conn.execute(f"DELETE FROM {table} WHERE shop_id = %s", (SHOP_A,))
        timer = threading.Timer(0.4, release.set)
        timer.start()
        started = time.monotonic()
        removed = repo.purge_shop(SHOP_A)
        waited = time.monotonic() - started
    finally:
        release.set()
        holder.join()
    assert removed == 0
    assert waited >= 0.2  # 确实等过了请求级的两个上限
