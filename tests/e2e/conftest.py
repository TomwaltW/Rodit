"""端到端测试夹具。共用的组装代码在 e2e_support.py（importlib 模式下测试模块不能 import
conftest，所以放进单独的模块，并把本目录加进 sys.path）。

PostgreSQL 夹具（@pytest.mark.pg）：每次测试会话建一个名字唯一的临时库、执行迁移，结束时删除；
连不上或没有建库权限时整组跳过（与 tests/storage 的做法一致，管理连接串来自 RODIT_TEST_PG_DSN）。
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ADMIN_DSN = os.environ.get("RODIT_TEST_PG_DSN", "dbname=postgres")


@pytest.fixture(scope="session")
def e2e_pg_dsn() -> Iterator[str]:
    """迁移好的临时空库的连接串；PostgreSQL 不可用时跳过。"""
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from rodit.storage import MIGRATIONS, Database

    name = f"rodit_e2e_{uuid.uuid4().hex[:12]}"
    try:
        with psycopg.connect(ADMIN_DSN, autocommit=True, connect_timeout=3) as admin:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    except psycopg.Error as e:
        pytest.skip(f"PostgreSQL 不可用或没有建库权限（{type(e).__name__}），跳过")
    dsn = make_conninfo(ADMIN_DSN, dbname=name)
    try:
        assert Database(dsn).migrate() == [m.version for m in MIGRATIONS]
        yield dsn
    finally:
        with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )
