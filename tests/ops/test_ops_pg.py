"""部署入口跑在真实 PostgreSQL 上（方案 §9.1、§9.2、§10）：本地试用初始化、就绪、指标。

@pytest.mark.pg：连不上 PostgreSQL 时整组跳过。大模型用离线替身（本地试用、没配大模型），
生产模式只查就绪、不发业务请求（不调用大模型、不联网）。
"""

from __future__ import annotations

import io
import re

import pytest
from ops_support import TestClient, parse_exposition, sample

from rodit.ops.serve import build_server, local_init
from rodit.ops.settings import ServerSettings
from rodit.storage import Database
from rodit.testing.fixtures import SHOP_A, SHOP_B

pytestmark = pytest.mark.pg

_TOKEN_LINE = re.compile(r"^\s+(\S+)\s+(?:店铺 (\S+)\s+账号 (\S+)|(\S+))\s+Bearer (rdt_\S+)$")


def _tokens(output: str) -> dict[tuple[str, str], str]:
    """local-init 打印的令牌 → {(角色, 店铺或名字): 令牌}。"""
    found: dict[tuple[str, str], str] = {}
    for line in output.splitlines():
        m = _TOKEN_LINE.match(line)
        if m:
            role, shop, _user, name, token = m.groups()
            found[(role, shop or name)] = token
    return found


@pytest.fixture
def local_env(ops_env, ops_pg_dsn, monkeypatch, tmp_path):
    for name in ("RODIT_LLM_BASE_URL", "RODIT_LLM_MODEL", "RODIT_LLM_API_KEY"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("RODIT_ENV", "local")
    monkeypatch.setenv("RODIT_DATABASE_DSN", ops_pg_dsn)  # 应用角色（SET LOCAL ROLE rodit_app）
    monkeypatch.setenv("RODIT_MIGRATE_DSN", ops_pg_dsn)
    monkeypatch.setenv("RODIT_TOKENS_FILE", str(tmp_path / "tokens.json"))
    return ops_pg_dsn


def test_local_init_then_serve_on_postgres(local_env):
    settings = ServerSettings.from_env()
    out = io.StringIO()
    assert local_init(settings, out) == 0
    text = out.getvalue()
    assert "已写入合成演示数据" in text
    tokens = _tokens(text)
    assert {("agent", SHOP_A), ("integration", SHOP_B), ("metrics", "local")} <= set(tokens)

    # 重复执行：不重复写数据、不重新生成令牌
    again = io.StringIO()
    assert local_init(settings, again) == 0
    assert "跳过写入" in again.getvalue() and "不重新生成" in again.getvalue()
    assert "rdt_" not in again.getvalue()

    server = build_server(settings, database=Database(local_env))
    client = TestClient(server.app)
    ready = client.get("/readyz")
    assert ready.status_code == 200, ready.json()
    assert ready.json()["checks"]["migrations"]["ok"] is True

    ask = client.post(
        f"/v1/shops/{SHOP_A}/standalone/ask",
        json={"session_id": "web_pg", "question": "能开发票吗"},
        headers={"Authorization": f"Bearer {tokens[('agent', SHOP_A)]}"},
    )
    assert ask.status_code == 200, ask.text
    assert ask.json()["suggestions"]

    metrics = client.get(
        "/metrics", headers={"Authorization": f"Bearer {tokens[('metrics', 'local')]}"}
    )
    s = parse_exposition(metrics.text)
    assert sample(s, "rodit_engine_messages_total", shop_id=SHOP_A, outcome="suggested") == 1
    assert sample(s, "rodit_auth_expiring_shops") == 0  # 合成授权 30 天后才到期
    assert sample(s, "rodit_auth_checked_shops") == 2
    # 审计照常落库（装饰器只是旁路采集）
    events = server.unit.repos.audit.list_events(SHOP_A, action="engine.result")
    assert len(events) == 1


def test_production_mode_c_readiness_on_postgres(ops_env, ops_pg_dsn, token_file, monkeypatch):
    monkeypatch.setenv("RODIT_DATABASE_DSN", ops_pg_dsn)
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    server = build_server(ServerSettings.from_env())  # 就绪检查用生产组装同一套数据库设置
    client = TestClient(server.app)
    not_ready = client.get("/readyz").json()
    assert not_ready["status"] == "not_ready"

    from rodit.ops.serve import migrate_database

    monkeypatch.setenv("RODIT_MIGRATE_DSN", ops_pg_dsn)
    migrate_database()
    server.readiness.check(fresh=True)
    ready = client.get("/readyz")
    assert ready.status_code == 200, ready.json()
