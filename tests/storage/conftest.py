"""PostgreSQL 测试夹具。

每次测试会话在本机 PostgreSQL 里建一个名字唯一的临时库、执行迁移，结束时删除；
连不上（或没有建库权限）时整组 @pytest.mark.pg 测试自动跳过，`uv run pytest` 照常通过。
管理连接串来自环境变量 RODIT_TEST_PG_DSN，默认 "dbname=postgres"（本机 unix socket，
见 scripts/dev-pg.sh）。所有数据都是合成的（rodit.testing.fixtures）。
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from rodit.storage import BUSINESS_TABLES, MIGRATIONS, Database

ADMIN_DSN = os.environ.get("RODIT_TEST_PG_DSN", "dbname=postgres")
TOKEN_KEY = "rodit-test-token-key-synthetic-only"  # 只用于测试的合成密钥


@pytest.fixture(scope="session")
def pg_admin_dsn() -> str:
    try:
        with psycopg.connect(ADMIN_DSN, autocommit=True, connect_timeout=3):
            pass
    except psycopg.Error as e:
        pytest.skip(f"PostgreSQL 不可用（{type(e).__name__}），跳过数据库测试")
    return ADMIN_DSN


@pytest.fixture(scope="session")
def make_database(pg_admin_dsn: str) -> Iterator[Callable[[], str]]:
    """建一个临时空库，返回它的连接串；会话结束时全部删除。"""
    created: list[str] = []

    def make() -> str:
        name = f"rodit_test_{uuid.uuid4().hex[:12]}"
        try:
            with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
                admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        except psycopg.Error as e:
            pytest.skip(f"没有建库权限（{type(e).__name__}），跳过数据库测试")
        created.append(name)
        return make_conninfo(pg_admin_dsn, dbname=name)

    yield make
    with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
        for name in created:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.fixture(scope="session")
def pg_dsn(make_database: Callable[[], str]) -> str:
    dsn = make_database()
    assert Database(dsn).migrate() == [m.version for m in MIGRATIONS]
    return dsn


@pytest.fixture(scope="session")
def db(pg_dsn: str) -> Database:
    """应用视角：每个事务 SET LOCAL ROLE rodit_app，受行级安全约束。"""
    return Database(pg_dsn)


@pytest.fixture(scope="session")
def admin_db(pg_dsn: str) -> Database:
    """管理员视角（测试用超级用户、不切换角色）：绕过行级安全，用来造数据和核对原始数据。"""
    return Database(pg_dsn, app_role=None)


@pytest.fixture
def token_key() -> str:
    return TOKEN_KEY


@pytest.fixture
def truncate(admin_db: Database) -> Callable[[], None]:
    """清空所有业务表：请求这个夹具时先清空一次；契约测试的工厂函数每次新建仓储前也调用它。"""

    def run() -> None:
        tables = sql.SQL(", ").join(sql.Identifier(t) for t in BUSINESS_TABLES)
        with admin_db.session(None) as conn:
            conn.execute(sql.SQL("TRUNCATE {} RESTART IDENTITY").format(tables))

    run()
    return run
