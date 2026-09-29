"""rodit.ops 测试夹具。公共代码在 ops_support.py（importlib 模式下测试模块不能 import conftest，
所以放进单独的模块，并把本目录加进 sys.path）。

- ops_env：清掉所有 RODIT_* 部署变量，换成合成值（生产模式）；
- token_file：合成令牌文件（店铺 A 的客服、店长、平台对接，店铺 B 的客服，一个抓取令牌）；
- 每个测试结束后恢复根日志器（configure_logging 会改它）；
- ops_pg_dsn：PostgreSQL 临时空库（没有迁移）；连不上或没有建库权限时跳过（@pytest.mark.pg）。
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ops_support import OPS_ENV_NAMES, PROD_ENV  # noqa: E402

from rodit.api import Principal  # noqa: E402
from rodit.ops.tokens import (  # noqa: E402
    TokenFile,
    issue_metrics_token,
    issue_principal_token,
    write_token_file,
)
from rodit.testing.fixtures import AGENT, MANAGER, SHOP_A, SHOP_B  # noqa: E402

ADMIN_DSN = os.environ.get("RODIT_TEST_PG_DSN", "dbname=postgres")


@pytest.fixture
def ops_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    for name in OPS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in PROD_ENV.items():
        monkeypatch.setenv(name, value)
    return dict(PROD_ENV)


@dataclass
class Tokens:
    path: Path
    tokens: dict[str, str]  # 名字 → 令牌原文
    principals: dict[str, Principal]

    def auth(self, name: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.tokens[name]}"}


CALLERS: dict[str, tuple[str, str, str]] = {
    "agent_a": (SHOP_A, AGENT, "agent"),
    "manager_a": (SHOP_A, MANAGER, "manager"),
    "integration_a": (SHOP_A, "tb_push_a", "integration"),
    "agent_b": (SHOP_B, "kf_b01", "agent"),
}


@pytest.fixture
def token_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tokens:
    issued = {name: issue_principal_token(*spec) for name, spec in CALLERS.items()}  # type: ignore[arg-type]
    metrics = issue_metrics_token("prometheus")
    path = tmp_path / "secrets" / "rodit_tokens.json"
    write_token_file(
        path,
        TokenFile(principals=[i.entry for i in issued.values()], metrics=[metrics.entry]),
    )
    monkeypatch.setenv("RODIT_TOKENS_FILE", str(path))
    tokens = {name: i.token for name, i in issued.items()}
    tokens["metrics"] = metrics.token
    principals = {name: i.entry.principal() for name, i in issued.items()}  # type: ignore[union-attr]
    return Tokens(path=path, tokens=tokens, principals=principals)


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    watched = {n: logging.getLogger(n) for n in ("uvicorn", "uvicorn.error", "uvicorn.access")}
    saved = {n: (list(lg.handlers), lg.propagate) for n, lg in watched.items()}
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    for name, lg in watched.items():
        lg.handlers[:], lg.propagate = saved[name]


@pytest.fixture(scope="session")
def ops_pg_admin() -> str:
    import psycopg

    try:
        with psycopg.connect(ADMIN_DSN, autocommit=True, connect_timeout=3):
            pass
    except psycopg.Error as e:
        pytest.skip(f"PostgreSQL 不可用（{type(e).__name__}），跳过数据库测试")
    return ADMIN_DSN


@pytest.fixture
def ops_pg_dsn(ops_pg_admin: str) -> Iterator[str]:
    """一个没有迁移过的临时空库；测试结束删除。"""
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    name = f"rodit_ops_{uuid.uuid4().hex[:12]}"
    try:
        with psycopg.connect(ops_pg_admin, autocommit=True) as admin:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    except psycopg.Error as e:
        pytest.skip(f"没有建库权限（{type(e).__name__}），跳过数据库测试")
    try:
        yield make_conninfo(ops_pg_admin, dbname=name)
    finally:
        with psycopg.connect(ops_pg_admin, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )
