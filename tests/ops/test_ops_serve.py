"""部署入口的组装（方案 §9.1、§8.1 模式 C、§12.3、§12.4）：环境变量 → 设置 → 平台单元 + HTTP 应用。

不连数据库（Database 构造时不建连接，就绪检查用假数据库）、不调用大模型（生产模式下不发业务请求）。
"""

from __future__ import annotations

import io
import json
import re
import warnings
from pathlib import Path
from typing import Any

import pytest
from ops_support import (
    SYNTHETIC_LLM_KEY,
    SYNTHETIC_TOKEN_KEY,
    FakeReadinessDb,
    TestClient,
    label_values,
    parse_exposition,
    sample,
)

from rodit.adapters import NullToolGateway, StandaloneAssistAdapter
from rodit.app.demo import seed_unit, taobao_event
from rodit.app.unit import in_memory_repos
from rodit.core.enums import Platform
from rodit.ops import serve
from rodit.ops.bindings import BindingError, PlatformBindings, is_mock
from rodit.ops.instrument import MeteredToolGateway, ObservedAuditLog
from rodit.ops.serve import StartupError, build_server, healthcheck, main
from rodit.ops.settings import DEFAULT_TOKENS_FILE, ServerSettings, SettingsError
from rodit.ops.tokens import TokenFileAuthenticator, TokenFileError
from rodit.storage import PgAuditLog
from rodit.storage.db import MIGRATIONS
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A, SHOP_B, build_demo_world

ALL_VERSIONS = tuple(m.version for m in MIGRATIONS)


def _settings(**env: str) -> ServerSettings:
    import os

    for name, value in env.items():
        os.environ[name] = value
    return ServerSettings.from_env()


# ================================================================ 设置


def test_settings_defaults(ops_env):
    s = ServerSettings.from_env()
    assert s.env == "production" and s.platform == Platform.TAOBAO
    assert s.tokens_file == DEFAULT_TOKENS_FILE
    assert (s.host, s.port, s.log_format) == ("127.0.0.1", 8000, "json")
    assert not s.mode_c_only and not s.metrics_public and not s.migrate_on_start
    assert s.deploy_mode == "platform"
    assert SYNTHETIC_TOKEN_KEY not in repr(s)  # 设置里没有任何密钥


def test_settings_from_env(ops_env, monkeypatch):
    for name, value in {
        "RODIT_ENV": "LOCAL",
        "RODIT_PLATFORM": "Douyin",
        "RODIT_MODE_C_ONLY": "yes",
        "RODIT_PORT": "9000",
        "RODIT_LOG_LEVEL": "debug",
        "RODIT_METRICS_PUBLIC": "1",
        "RODIT_AUTH_EXPIRY_WARN_DAYS": "3",
    }.items():
        monkeypatch.setenv(name, value)
    s = ServerSettings.from_env()
    assert (s.env, s.platform, s.mode_c_only, s.port, s.log_level) == (
        "local",
        Platform.DOUYIN,
        True,
        9000,
        "DEBUG",
    )
    assert s.metrics_public and s.auth_expiry_warn_days == 3 and s.deploy_mode == "local"


@pytest.mark.parametrize(
    "name, value, match",
    [
        ("RODIT_PLATFORM", "", "RODIT_PLATFORM"),
        ("RODIT_PLATFORM", "jd", "platform"),
        ("RODIT_ENV", "staging", "env"),
        ("RODIT_MODE_C_ONLY", "maybe", "RODIT_MODE_C_ONLY"),
        ("RODIT_PORT", "http", "RODIT_PORT"),
        ("RODIT_PORT", "70000", "port"),
        ("RODIT_READINESS_CACHE_S", "-1", "RODIT_READINESS_CACHE_S"),
        ("RODIT_LOG_FORMAT", "xml", "log_format"),
    ],
)
def test_bad_settings(ops_env, monkeypatch, name, value, match):
    monkeypatch.setenv(name, value)
    with pytest.raises(SettingsError, match=match):
        ServerSettings.from_env()


# ================================================================ 生产模式：平台绑定


def _prod_server(monkeypatch, *, db: FakeReadinessDb | None = None, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return build_server(ServerSettings.from_env(), database=db or FakeReadinessDb(ALL_VERSIONS))


def test_production_refuses_to_start_without_platform_bindings(ops_env, token_file):
    with pytest.raises(StartupError, match="RODIT_MODE_C_ONLY=1"):
        build_server(ServerSettings.from_env(), database=FakeReadinessDb())


def test_production_mode_c_only(ops_env, token_file, monkeypatch):
    server = _prod_server(monkeypatch, RODIT_MODE_C_ONLY="1")
    unit = server.unit
    assert unit.adapters.keys() == [(Platform.TAOBAO, "C")]
    assert isinstance(unit.adapters.get(Platform.TAOBAO, "C"), StandaloneAssistAdapter)
    assert isinstance(unit.tools, MeteredToolGateway)
    assert isinstance(unit.tools.inner, NullToolGateway)
    assert unit.action_port is None
    # 生产组装：PostgreSQL 仓储，审计套上指标采集，大模型计量也经过它
    assert isinstance(unit.repos.audit, ObservedAuditLog)
    assert isinstance(unit.repos.audit.inner, PgAuditLog)
    assert unit.llm._audit is unit.repos.audit  # type: ignore[attr-defined]
    assert server.auth_monitor is None  # 模式 C 没有平台授权，不告警
    assert server.bindings.mode_c_only

    client = TestClient(server.app)
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"
    assert ready.headers["cache-control"] == "no-store"
    s = parse_exposition(client.get("/metrics", headers=token_file.auth("metrics")).text)
    assert sample(s, "rodit_build_info", platform="taobao", mode="mode_c_only", version="0.0.1")
    # 平台事件接口没有注册：404，不会碰到任何适配器
    event = client.post("/v1/platform/B/events", json={}, headers=token_file.auth("integration_a"))
    assert event.status_code == 404


def test_production_with_platform_bindings_from_env(ops_env, token_file, monkeypatch):
    server = _prod_server(monkeypatch, RODIT_PLATFORM_BINDINGS="ops_support:cloud_bindings")
    assert server.unit.adapters.keys() == [(Platform.TAOBAO, "B"), (Platform.TAOBAO, "C")]
    assert server.auth_monitor is not None
    assert server.bindings.shop_ids is not None
    assert not is_mock(server.bindings.tools)


def test_production_rejects_mock_bindings(ops_env, token_file, monkeypatch):
    monkeypatch.setenv("RODIT_PLATFORM_BINDINGS", "ops_support:mock_bindings")
    with pytest.raises(BindingError, match="模拟实现") as exc:
        build_server(ServerSettings.from_env(), database=FakeReadinessDb())
    for name in ("InMemoryToolGateway", "TaobaoMockAdapter", "FakeActionPort"):
        assert name in str(exc.value)


def test_bindings_and_mode_c_are_exclusive(ops_env, token_file, monkeypatch):
    monkeypatch.setenv("RODIT_PLATFORM_BINDINGS", "ops_support:cloud_bindings")
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    with pytest.raises(StartupError, match="不能同时设置"):
        build_server(ServerSettings.from_env(), database=FakeReadinessDb())


@pytest.mark.parametrize(
    "spec, match",
    [
        ("no_colon", "格式"),
        ("no_such_module_xyz:build", "找不到"),
        ("ops_support:missing", "没有"),
        ("ops_support:NOT_CALLABLE", "不是可调用"),
        ("ops_support:not_bindings", "PlatformBindings"),
    ],
)
def test_bad_binding_specs(ops_env, token_file, monkeypatch, spec, match):
    monkeypatch.setenv("RODIT_PLATFORM_BINDINGS", spec)
    with pytest.raises(BindingError, match=match):
        build_server(ServerSettings.from_env(), database=FakeReadinessDb())


def test_bindings_for_another_platform_rejected(ops_env, token_file, monkeypatch):
    def wrong_platform(ctx):
        return PlatformBindings(
            tools=NullToolGateway(Platform.PDD),
            adapters=[StandaloneAssistAdapter(Platform.PDD, ctx.clock, ctx.hasher)],
        )

    with pytest.raises(BindingError, match="不属于本单元"):
        build_server(
            ServerSettings.from_env(), bindings_factory=wrong_platform, database=FakeReadinessDb()
        )


def test_null_tools_only_for_mode_c(ops_env, token_file):
    def sneaky(ctx):
        return PlatformBindings(
            tools=NullToolGateway(ctx.platform),
            adapters=[StandaloneAssistAdapter(ctx.platform, ctx.clock, ctx.hasher)],
        )

    with pytest.raises(BindingError, match="NullToolGateway"):
        build_server(ServerSettings.from_env(), bindings_factory=sneaky, database=FakeReadinessDb())


# ================================================================ 生产模式：启动前检查


@pytest.mark.parametrize(
    "missing",
    ["RODIT_DATABASE_DSN", "RODIT_TOKEN_KEY", "RODIT_BUYER_SALT", "RODIT_LLM_BASE_URL"],
)
def test_missing_required_env_fails_fast(ops_env, token_file, monkeypatch, missing):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.delenv(missing)
    with pytest.raises(StartupError, match=missing) as exc:
        build_server(ServerSettings.from_env(), database=FakeReadinessDb())
    for value in (SYNTHETIC_TOKEN_KEY, SYNTHETIC_LLM_KEY, "rodit_never_connected"):
        assert value not in str(exc.value)


def test_missing_llm_key_means_not_ready(ops_env, token_file, monkeypatch):
    monkeypatch.delenv("RODIT_LLM_API_KEY")
    server = _prod_server(monkeypatch, RODIT_MODE_C_ONLY="1")
    ready = TestClient(server.app).get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["checks"]["llm"] == {"ok": False, "reason": "missing_api_key"}


def test_database_not_migrated_means_not_ready(ops_env, token_file, monkeypatch):
    server = _prod_server(monkeypatch, db=FakeReadinessDb(versions=[1]), RODIT_MODE_C_ONLY="1")
    body = TestClient(server.app).get("/readyz").json()
    assert body["status"] == "not_ready"
    assert body["checks"]["migrations"]["reason"] == "behind"


def test_missing_token_file_fails_fast(ops_env, monkeypatch, tmp_path):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_TOKENS_FILE", str(tmp_path / "absent.json"))
    with pytest.raises(TokenFileError, match="不存在"):
        build_server(ServerSettings.from_env(), database=FakeReadinessDb())


# ================================================================ /metrics 的访问控制


def test_metrics_requires_token_in_production(ops_env, token_file, monkeypatch):
    client = TestClient(_prod_server(monkeypatch, RODIT_MODE_C_ONLY="1").app)
    assert client.get("/metrics").status_code == 401
    assert (
        client.get("/metrics", headers={"Authorization": "Bearer nope-nope-nope-nope"}).status_code
        == 401
    )
    # 业务令牌不能抓指标
    assert client.get("/metrics", headers=token_file.auth("agent_a")).status_code == 401
    ok = client.get("/metrics", headers=token_file.auth("metrics"))
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("text/plain; version=0.0.4")


def test_production_without_metrics_tokens_refuses_unless_public(ops_env, monkeypatch, tmp_path):
    from rodit.ops.tokens import append_entries, issue_principal_token

    path = tmp_path / "tokens.json"
    append_entries(path, [issue_principal_token(SHOP_A, "kf_02", "agent").entry])
    monkeypatch.setenv("RODIT_TOKENS_FILE", str(path))
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    with pytest.raises(StartupError, match="RODIT_METRICS_PUBLIC"):
        build_server(ServerSettings.from_env(), database=FakeReadinessDb())
    monkeypatch.setenv("RODIT_METRICS_PUBLIC", "1")
    server = build_server(ServerSettings.from_env(), database=FakeReadinessDb())
    assert TestClient(server.app).get("/metrics").status_code == 200


def test_ops_routes_in_openapi_without_duplicate_operation_ids(ops_env, token_file, monkeypatch):
    server = _prod_server(monkeypatch, RODIT_MODE_C_ONLY="1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # FastAPI 对重复的 operation_id 发 UserWarning
        schema = server.app.openapi()
    assert {"/healthz", "/readyz", "/metrics"} <= set(schema["paths"])
    assert TestClient(server.app).get("/healthz").json() == {"status": "ok"}


# ================================================================ 本地试用：完整链路（内存仓储）


@pytest.fixture
def local_server(ops_env, token_file, monkeypatch):
    """本地试用模式：合成平台绑定 + 离线替身 + 内存仓储（不连数据库）。"""
    for name in ("RODIT_LLM_BASE_URL", "RODIT_LLM_MODEL", "RODIT_LLM_API_KEY"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("RODIT_ENV", "local")
    monkeypatch.setenv("RODIT_AUTH_EXPIRY_CHECK_S", "0")
    world = build_demo_world()
    clock = FakeClock(world.now)
    server = build_server(
        ServerSettings.from_env(),
        clock=clock,
        repos=in_memory_repos(clock),
        database=FakeReadinessDb(ALL_VERSIONS),
    )
    seed_unit(server.unit, world)
    return server, TestClient(server.app), token_file, clock


def test_local_flow_counts_suggestions_and_latency(local_server):
    server, client, tokens, _ = local_server
    assert is_mock(server.bindings.tools)  # 本地试用允许合成数据
    ask = client.post(
        f"/v1/shops/{SHOP_A}/standalone/ask",
        json={"session_id": "web_1", "question": "能开发票吗"},
        headers=tokens.auth("agent_a"),
    )
    assert ask.status_code == 200, ask.text
    assert ask.json()["suggestions"]
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready" and ready["checks"]["llm"] == {"ok": True, "offline": True}

    s = parse_exposition(client.get("/metrics", headers=tokens.auth("metrics")).text)
    assert sample(s, "rodit_engine_messages_total", shop_id=SHOP_A, outcome="suggested") == 1
    assert sample(s, "rodit_suggestions_total", shop_id=SHOP_A) >= 1
    assert sample(s, "rodit_engine_latency_seconds_count", stage="total") == 1
    for stage in ("understand", "route", "retrieve", "generate", "qa"):
        assert sample(s, "rodit_engine_latency_seconds_count", stage=stage) == 1
    assert sample(s, "rodit_build_info", platform="taobao", mode="local", version="0.0.1") == 1
    route = "/v1/shops/{shop_id}/standalone/ask"
    assert sample(s, "rodit_http_requests_total", route=route, method="POST", status="200") == 1
    assert sample(s, "rodit_audit_events_total", action="engine.result") == 1


def test_local_flow_platform_events_and_tool_calls(local_server):
    server, client, tokens, clock = local_server
    raw = taobao_event("我的快递到哪了", msg_id="ops_1", order_id="o_1002", session_id="c_ops")
    raw["send_time_ms"] = int(clock.now().timestamp() * 1000)
    resp = client.post("/v1/platform/B/events", json=raw, headers=tokens.auth("integration_a"))
    assert resp.status_code in (200, 204), resp.text
    s = parse_exposition(client.get("/metrics", headers=tokens.auth("metrics")).text)
    calls = {
        dict(labels)["tool"]: v
        for (name, labels), v in s.items()
        if name == "rodit_platform_tool_calls_total"
    }
    assert calls, "平台接口调用应当计数"


def test_tenant_isolation_is_counted(local_server):
    _, client, tokens, _ = local_server
    resp = client.post(
        f"/v1/shops/{SHOP_B}/standalone/ask",
        json={"session_id": "web_1", "question": "能开发票吗"},
        headers=tokens.auth("agent_a"),
    )
    assert resp.status_code == 403
    s = parse_exposition(client.get("/metrics", headers=tokens.auth("metrics")).text)
    route = "/v1/shops/{shop_id}/standalone/ask"
    assert sample(s, "rodit_http_errors_total", route=route, code="tenant_isolation") == 1


def test_metrics_labels_contain_no_buyer_data(local_server):
    """买家文本、手机号、地址、昵称、buyer_ref、会话 ID 都不能出现在指标里（§5.3、§12.3）。"""
    server, client, tokens, clock = local_server
    pii_question = "我手机13812345678，寄到北京市朝阳区建国路88号3号楼2单元501，这件是纯棉的吗"
    client.post(
        f"/v1/shops/{SHOP_A}/standalone/ask",
        json={"session_id": "web_pii_7f3a", "question": pii_question, "request_id": "r1"},
        headers=tokens.auth("agent_a"),
    )
    raw = taobao_event(pii_question, msg_id="ops_pii", session_id="c_pii_9")
    raw["send_time_ms"] = int(clock.now().timestamp() * 1000)
    client.post("/v1/platform/B/events", json=raw, headers=tokens.auth("integration_a"))
    text = client.get("/metrics", headers=tokens.auth("metrics")).text
    s = parse_exposition(text)
    values = label_values(s)
    conv = server.unit.repos.conversations.get_conversation(SHOP_A, "c_pii_9")
    assert conv is not None and conv.buyer_ref
    for leaked in (
        "13812345678",
        "建国路",
        "纯棉",
        "合成昵称",
        conv.buyer_ref,
        "web_pii_7f3a",
        "c_pii_9",
        "ops_pii",
    ):
        assert leaked not in text, leaked
    allowed = re.compile(r"^[A-Za-z0-9_.:/{}+\-]+$")
    assert all(allowed.match(v) for v in values), values
    label_names = {k for (_, labels) in s for k, _ in labels}
    assert label_names <= {
        "shop_id",
        "outcome",
        "intent",
        "action",
        "stage",
        "le",
        "rule",
        "blocking",
        "kind",
        "purpose",
        "direction",
        "tool",
        "route",
        "method",
        "status",
        "code",
        "platform",
        "mode",
        "version",
    }


def test_auth_expiring_gauge(local_server):
    _, client, tokens, clock = local_server
    s = parse_exposition(client.get("/metrics", headers=tokens.auth("metrics")).text)
    assert sample(s, "rodit_auth_expiring_shops") == 0
    assert sample(s, "rodit_auth_checked_shops") == 2  # 令牌文件和合成订购列表里的店铺 A、B
    clock.advance(days=25)  # 演示授权 30 天后到期，进入 7 天告警窗口
    s = parse_exposition(client.get("/metrics", headers=tokens.auth("metrics")).text)
    assert sample(s, "rodit_auth_expiring_shops") == 2
    assert sample(s, "rodit_shop_auth_expiring", shop_id=SHOP_A) == 1


def test_revoke_removes_shop_metrics(local_server):
    server, client, tokens, _ = local_server
    client.post(
        f"/v1/shops/{SHOP_A}/standalone/ask",
        json={"session_id": "web_1", "question": "能开发票吗"},
        headers=tokens.auth("agent_a"),
    )
    server.service.revoke_shop(SHOP_A)
    metrics_text = server.metrics.render()
    s = parse_exposition(metrics_text)
    assert sample(s, "rodit_engine_messages_total", shop_id=SHOP_A, outcome="suggested") is None


# ================================================================ 命令行


def test_main_serve_runs_uvicorn_with_our_logging(ops_env, token_file, monkeypatch):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_PORT", "9123")
    calls: list[tuple[Any, dict[str, Any]]] = []
    assert main(["serve"], run=lambda app, **kw: calls.append((app, kw))) == 0
    ((app, kw),) = calls
    assert kw["host"] == "127.0.0.1" and kw["port"] == 9123
    assert kw["log_config"] is None and kw["server_header"] is False
    assert isinstance(app.state.rodit_authenticator, TokenFileAuthenticator)


def test_main_defaults_to_serve(ops_env, token_file, monkeypatch):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    calls: list[Any] = []
    assert main([], run=lambda app, **kw: calls.append(app)) == 0
    assert len(calls) == 1


def test_main_startup_failure_exit_code_and_log(ops_env, token_file, capsys):
    ran: list[Any] = []
    assert main(["serve"], run=lambda *a, **k: ran.append(a)) == serve.EXIT_STARTUP
    assert not ran
    err = capsys.readouterr().err
    record = json.loads(err.strip().splitlines()[-1])
    assert record["level"] == "ERROR" and "启动失败" in record["msg"]
    assert SYNTHETIC_TOKEN_KEY not in err and SYNTHETIC_LLM_KEY not in err


def test_main_migrate_on_start(ops_env, token_file, monkeypatch):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_MIGRATE_ON_START", "1")
    done: list[str] = []
    monkeypatch.setattr(serve, "migrate", lambda env: done.append(env) or [])
    monkeypatch.setattr(serve, "grant_readiness_access", lambda db: done.append("grant"))
    assert main(["serve"], run=lambda *a, **k: None) == 0
    assert done == ["RODIT_DATABASE_DSN", "grant"]


def test_main_migrate_prefers_owner_dsn(ops_env, monkeypatch):
    monkeypatch.delenv("RODIT_PLATFORM")  # 迁移不需要平台设置
    monkeypatch.setenv("RODIT_MIGRATE_DSN", "dbname=owner_never_connected")
    done: list[Any] = []
    monkeypatch.setattr(serve, "migrate", lambda env: done.append(env) or [1, 2])
    monkeypatch.setattr(serve, "grant_readiness_access", lambda db: done.append(db.app_role))
    assert main(["migrate"]) == 0
    assert done == ["RODIT_MIGRATE_DSN", None]  # 授权用属主身份（不切换到应用角色）


def test_main_migrate_without_dsn(ops_env, monkeypatch):
    monkeypatch.delenv("RODIT_DATABASE_DSN")
    assert main(["migrate"]) == serve.EXIT_STARTUP


def test_main_migrate_database_down(ops_env, monkeypatch):
    from rodit.core.errors import StorageUnavailable

    def down(env: str) -> list[int]:
        raise StorageUnavailable("数据库连接失败")

    monkeypatch.setattr(serve, "migrate", down)
    assert main(["migrate"]) == serve.EXIT_FAILURE


def test_issue_token_command(tmp_path, capsys):
    out = io.StringIO()
    path = tmp_path / "tokens.json"
    args = ["issue-token", "--shop-id", SHOP_A, "--user-id", "kf_09", "--role", "agent"]
    assert main([*args, "--append", str(path)], out=out) == 0
    token = re.search(r"Bearer (rdt_\S+)", out.getvalue()).group(1)
    auth = TokenFileAuthenticator(path)
    assert auth.authenticate(token).user_id == "kf_09"
    assert token not in path.read_text(encoding="utf-8")

    out = io.StringIO()
    assert main(["issue-token", "--metrics", "--name", "prom"], out=out) == 0
    text = out.getvalue()
    entry = json.loads(text.split("metrics 段：", 1)[1])
    assert entry["name"] == "prom" and len(entry["sha256"]) == 64
    assert main(["issue-token", "--shop-id", SHOP_A], out=io.StringIO()) == serve.EXIT_STARTUP


def test_issue_token_with_expiry(tmp_path):
    out = io.StringIO()
    path = tmp_path / "tokens.json"
    args = ["issue-token", "--shop-id", SHOP_A, "--user-id", "kf_09", "--role", "agent"]
    assert main([*args, "--expires-days", "30", "--append", str(path)], out=out) == 0
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["principals"][0]["expires_at"]


class _Resp:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_healthcheck_command():
    seen: list[str] = []

    def ok(url: str, timeout: float) -> _Resp:
        seen.append(url)
        return _Resp(200)

    def down(url: str, timeout: float) -> _Resp:
        raise ConnectionRefusedError

    assert healthcheck({"RODIT_PORT": "8123"}, opener=ok) == 0
    assert seen == ["http://127.0.0.1:8123/healthz"]
    assert healthcheck({}, opener=lambda u, timeout: _Resp(503)) == 1
    assert healthcheck({}, opener=down) == 1
    assert healthcheck({"RODIT_PORT": "80; rm -rf /"}, opener=ok) == 1


def test_healthcheck_via_main(monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _Resp(200))
    assert main(["healthcheck"]) == 0


def test_local_init_refused_in_production(ops_env, token_file):
    with pytest.raises(StartupError, match="本地试用"):
        serve.local_init(ServerSettings.from_env(), io.StringIO())
