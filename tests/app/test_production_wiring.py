"""生产组装给数据库设语句超时和锁等待超时（§12.1 检索 ≤ 0.5 秒、§12.2 知识库故障 → 不推荐）。

PostgreSQL 默认 statement_timeout = 0（不限时）：查询被锁住或存储卡住时请求线程一直挂着，
还占着会话锁，"超时 → KnowledgeUnavailable → 不推荐"的降级永远触发不了。
这里不连数据库：Database 构造时不建连接，只核对组装传进去的参数。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from rodit.app import production
from rodit.core.config import EngineConfig, LLMConfig
from rodit.core.enums import Platform
from rodit.storage import Database
from rodit.testing.fakes import InMemoryToolGateway
from rodit.testing.fixtures import SHOP_A, build_demo_world


@pytest.fixture
def prod_env(monkeypatch) -> dict[str, Any]:
    monkeypatch.setenv("RODIT_DATABASE_DSN", "dbname=rodit_never_connected")
    monkeypatch.setenv("RODIT_TOKEN_KEY", "rodit-test-token-key-synthetic-only")
    monkeypatch.setenv("RODIT_BUYER_SALT", "salt-synthetic-only")
    for name in ("RODIT_DB_STATEMENT_TIMEOUT_MS", "RODIT_DB_LOCK_TIMEOUT_MS"):
        monkeypatch.delenv(name, raising=False)
    created: list[Database] = []
    real_init = Database.__init__

    def recording_init(self: Database, *args: Any, **kw: Any) -> None:
        real_init(self, *args, **kw)
        created.append(self)

    monkeypatch.setattr(Database, "__init__", recording_init)
    return {"databases": created}


def _build() -> Any:
    world = build_demo_world()
    return production.build_production_unit(
        Platform.TAOBAO,
        config=EngineConfig(llm=LLMConfig(base_url="https://llm.example.invalid/v1", model="m")),
        tools=InMemoryToolGateway(world, Platform.TAOBAO),
        adapters=[],
    )


def test_production_unit_sets_statement_and_lock_timeouts(prod_env):
    _build()
    (db,) = prod_env["databases"]
    assert db.statement_timeout_ms == production.DEFAULT_STATEMENT_TIMEOUT_MS
    assert db.lock_timeout_ms == production.DEFAULT_LOCK_TIMEOUT_MS
    assert 0 < db.lock_timeout_ms <= db.statement_timeout_ms <= 5000  # 端到端 P95 ≤ 5 秒（§12.1）


def test_timeouts_come_from_env(prod_env, monkeypatch):
    monkeypatch.setenv("RODIT_DB_STATEMENT_TIMEOUT_MS", "800")
    monkeypatch.setenv("RODIT_DB_LOCK_TIMEOUT_MS", "300")
    _build()
    (db,) = prod_env["databases"]
    assert (db.statement_timeout_ms, db.lock_timeout_ms) == (800, 300)


@pytest.mark.parametrize("value", ["0", "-5", "abc", "1.5"])
def test_invalid_timeout_env_fails_at_startup(prod_env, monkeypatch, value):
    monkeypatch.setenv("RODIT_DB_STATEMENT_TIMEOUT_MS", value)
    with pytest.raises((ValueError, RuntimeError), match="RODIT_DB_STATEMENT_TIMEOUT_MS"):
        _build()


def test_migrate_runs_without_request_timeouts(prod_env, monkeypatch):
    """迁移（建表、加列）可能比请求慢得多：迁移用的连接不设请求级超时。"""
    monkeypatch.setattr(Database, "migrate", lambda self: [])
    assert production.migrate() == []
    (db,) = prod_env["databases"]
    assert db.statement_timeout_ms is None and db.lock_timeout_ms is None


class _FakeResult:
    rowcount = 0

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> list[Any]:
        return []


class _FakeConn:
    def execute(self, *args: Any, **kw: Any) -> _FakeResult:
        return _FakeResult()


def test_production_purge_runs_without_request_timeouts(prod_env, monkeypatch):
    """§8.4 取消订购：repos.purgers() 的每个 purge_shop 都开不限时的事务（大店铺一次删
    几百万行，请求级的 2 秒上限下永远删不完）；普通请求仍然用有上限的事务。
    不连数据库，只记录 session 参数。"""
    sessions: list[bool] = []

    @contextmanager
    def recording_session(self: Database, shop_id: str | None, **kw: Any) -> Iterator[Any]:
        sessions.append(kw.get("unbounded", False))
        yield _FakeConn()

    monkeypatch.setattr(Database, "session", recording_session)
    unit = _build()
    for purger in unit.repos.purgers():
        purger.purge_shop(SHOP_A)
    assert sessions and all(sessions)
    sessions.clear()
    unit.repos.audit.list_events(SHOP_A)
    unit.repos.knowledge.get_shop(SHOP_A)
    assert sessions == [False, False]


def test_audit_wrapper_wraps_every_audit_writer(prod_env):
    """部署入口用 audit_wrapper 给审计套一层（采集运行指标）：引擎、路由留痕、质检留痕、
    大模型计量都写进套好的那一层；取消订购的清理也经过它。不传时仍是 PgAuditLog 本身。"""
    from rodit.storage import PgAuditLog

    class Wrapped:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

    world = build_demo_world()
    unit = production.build_production_unit(
        Platform.TAOBAO,
        config=EngineConfig(llm=LLMConfig(base_url="https://llm.example.invalid/v1", model="m")),
        tools=InMemoryToolGateway(world, Platform.TAOBAO),
        adapters=[],
        audit_wrapper=Wrapped,
    )
    audit = unit.repos.audit
    assert isinstance(audit, Wrapped) and isinstance(audit.inner, PgAuditLog)
    assert unit.llm._audit is audit  # type: ignore[attr-defined]  大模型计量
    assert unit.engine._audit_sink is audit  # type: ignore[attr-defined]
    assert unit.router.audit is audit  # 路由留痕（decision_log）
    assert unit.gate._audit is audit  # type: ignore[attr-defined]  质检留痕
    assert audit in unit.repos.purgers()
    assert isinstance(_build().repos.audit, PgAuditLog)
