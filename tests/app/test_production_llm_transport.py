"""大模型出口可注入（附录 A.3 "平台云内能否调用外部的境内大模型 API、走什么通道"、§9.1、§12.3）：
平台要求经它可报备的通道调用大模型时，平台云代码在 PlatformBindings.llm_transport 里给出口，
生产单元的大模型客户端经它发请求。只换出口：请求内容、密钥（每次调用时读环境变量）、计量
（llm.usage）、缺密钥时的就绪语义都不变；不给时仍是默认的 urllib 出口。

不连数据库（Database 构造时不建连接）、不联网（出口是假的，地址是 .invalid），数据全部合成。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
from starlette.exceptions import StarletteDeprecationWarning

from rodit.adapters import StandaloneAssistAdapter
from rodit.app import production
from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.config import EngineConfig, LLMConfig
from rodit.core.enums import Platform
from rodit.core.errors import LLMError
from rodit.core.ports import LLMMessage, LLMRequest
from rodit.llm.openai_compat import urllib_transport
from rodit.ops.bindings import BindingContext, PlatformBindings
from rodit.ops.serve import build_server
from rodit.ops.settings import ServerSettings
from rodit.ops.tokens import TokenFile, issue_metrics_token, write_token_file
from rodit.testing.fakes import InMemoryToolGateway
from rodit.testing.fixtures import SHOP_A, build_demo_world

with warnings.catch_warnings():
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

BASE_URL = "https://llm.example.invalid/v1"
SYNTHETIC_KEY = "sk-synthetic-only-transport-0000"


class RecordingTransport:
    """平台云代码的出口（假的）：记下每次请求，回一个 OpenAI 兼容的响应。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float
    ) -> tuple[int, dict[str, Any]]:
        self.calls.append((url, dict(headers), body, timeout_s))
        return 200, {
            "model": body.get("model", ""),
            "choices": [{"message": {"content": "好的，这就为您查询"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 21, "completion_tokens": 7},
        }


@pytest.fixture
def prod_env(monkeypatch) -> None:
    monkeypatch.setenv("RODIT_DATABASE_DSN", "dbname=rodit_never_connected")
    monkeypatch.setenv("RODIT_TOKEN_KEY", "rodit-test-token-key-synthetic-only")
    monkeypatch.setenv("RODIT_BUYER_SALT", "salt-synthetic-only")
    monkeypatch.setenv("RODIT_LLM_API_KEY", SYNTHETIC_KEY)
    for name in ("RODIT_DB_STATEMENT_TIMEOUT_MS", "RODIT_DB_LOCK_TIMEOUT_MS"):
        monkeypatch.delenv(name, raising=False)


def _request() -> LLMRequest:
    return LLMRequest(
        purpose="generate",
        messages=[LLMMessage(role="user", content="这件什么时候发货")],
        shop_id=SHOP_A,
    )


def _unit(config: EngineConfig, **kw: Any):
    sink = InMemoryAuditSink()
    unit = production.build_production_unit(
        Platform.TAOBAO,
        config=config,
        tools=InMemoryToolGateway(build_demo_world(), Platform.TAOBAO),
        adapters=[],
        audit_wrapper=lambda inner: sink,  # 审计记到内存里（不连数据库）
        **kw,
    )
    return unit, sink


CONFIGURED = EngineConfig(llm=LLMConfig(base_url=BASE_URL, model="synthetic-model"))


def test_injected_transport_carries_every_llm_call(prod_env):
    transport = RecordingTransport()
    unit, sink = _unit(CONFIGURED, llm_transport=transport)
    # 单元里用大模型的地方（生成、判断层）拿到的都是这一个客户端
    assert unit.generator._llm is unit.llm  # type: ignore[attr-defined]
    assert unit.router.provider.llm is unit.llm  # type: ignore[attr-defined]

    response = unit.llm.complete(_request())
    assert response.text == "好的，这就为您查询"
    assert len(transport.calls) == 1
    url, headers, body, timeout = transport.calls[0]
    assert url == f"{BASE_URL}/chat/completions"
    assert headers["Authorization"] == f"Bearer {SYNTHETIC_KEY}"  # 密钥照常每次从环境变量读
    assert body["model"] == "synthetic-model"
    assert body["messages"] == [{"role": "user", "content": "这件什么时候发货"}]
    assert "shop_id" not in body and "purpose" not in body  # 只用于本地计量，不发给模型服务
    assert timeout == CONFIGURED.llm.timeout_s

    # 计量不变：每次调用一条 llm.usage（店铺、用途、token 数），不含密钥
    (usage,) = [e for e in sink.events if e.action == AuditAction.LLM_USAGE.value]
    assert usage.shop_id == SHOP_A
    assert usage.detail["purpose"] == "generate"
    assert (usage.detail["input_tokens"], usage.detail["output_tokens"]) == (21, 7)
    assert SYNTHETIC_KEY not in str(usage.detail)


def test_transport_also_used_when_llm_comes_from_env(prod_env, monkeypatch):
    monkeypatch.setenv("RODIT_LLM_BASE_URL", BASE_URL)
    monkeypatch.setenv("RODIT_LLM_MODEL", "env-model")
    transport = RecordingTransport()
    unit, _ = _unit(EngineConfig(), llm_transport=transport)
    unit.llm.complete(_request())
    assert [c[2]["model"] for c in transport.calls] == ["env-model"]


def test_default_path_is_unchanged(prod_env):
    unit, _ = _unit(CONFIGURED)
    assert unit.llm._inner._transport is urllib_transport  # type: ignore[attr-defined]


def test_missing_key_still_fails_before_the_transport(prod_env, monkeypatch):
    monkeypatch.delenv("RODIT_LLM_API_KEY")
    transport = RecordingTransport()
    unit, sink = _unit(CONFIGURED, llm_transport=transport)
    with pytest.raises(LLMError, match="RODIT_LLM_API_KEY"):
        unit.llm.complete(_request())
    assert transport.calls == []
    (usage,) = [e for e in sink.events if e.action == AuditAction.LLM_USAGE.value]
    assert usage.detail["error"] == "LLMError"


# ================================================================ 部署入口：平台绑定 → 生产单元


class CloudTools:
    """ "平台云代码"的只读工具（内部是合成演示世界；不是 rodit.testing 的类，生产模式能用）。"""

    def __init__(self, inner: InMemoryToolGateway) -> None:
        self._inner = inner
        self.platform = inner.platform

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


TRANSPORT = RecordingTransport()


def cloud_bindings_with_transport(ctx: BindingContext) -> PlatformBindings:
    tools = InMemoryToolGateway(build_demo_world(ctx.clock.now()), ctx.platform)
    return PlatformBindings(
        tools=CloudTools(tools),
        adapters=[StandaloneAssistAdapter(ctx.platform, ctx.clock, ctx.hasher)],
        llm_transport=TRANSPORT,
    )


@pytest.fixture
def served(prod_env, monkeypatch, tmp_path: Path):
    for name in ("RODIT_ENV", "RODIT_MODE_C_ONLY", "RODIT_PLATFORM_BINDINGS", "RODIT_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RODIT_PLATFORM", "taobao")
    monkeypatch.setenv("RODIT_LLM_BASE_URL", BASE_URL)
    monkeypatch.setenv("RODIT_LLM_MODEL", "synthetic-model")
    metrics_token = issue_metrics_token("prometheus")
    path = tmp_path / "tokens.json"
    write_token_file(path, TokenFile(principals=[], metrics=[metrics_token.entry]))
    monkeypatch.setenv("RODIT_TOKENS_FILE", str(path))
    TRANSPORT.calls.clear()
    return {"Authorization": f"Bearer {metrics_token.token}"}


class _Readyz:
    """就绪检查用的假数据库（全部迁移）。"""

    def __init__(self) -> None:
        from rodit.storage.db import MIGRATIONS

        self.versions = [m.version for m in MIGRATIONS]

    def session(self, shop_id: str | None, **kw: Any) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def cm():
            yield self

        return cm()

    def execute(self, statement: str, *args: Any) -> Any:
        rows = [{"version": v} for v in self.versions]

        class _R:
            def fetchall(self_inner) -> list[dict[str, int]]:
                return rows

        return _R()


def test_serve_wires_bindings_transport_into_the_production_unit(served):
    server = build_server(
        ServerSettings.from_env(),
        bindings_factory=cloud_bindings_with_transport,
        database=_Readyz(),
    )
    assert server.bindings.llm_transport is TRANSPORT
    server.unit.llm.complete(_request())
    assert len(TRANSPORT.calls) == 1
    client = TestClient(server.app)
    assert client.get("/readyz").json()["checks"]["llm"] == {"ok": True}
    text = client.get("/metrics", headers=served).text
    # 计量照常进指标（审计写不进假库也照样采集）
    assert 'rodit_llm_calls_total{purpose="generate",outcome="ok"} 1' in text
    assert (
        'rodit_llm_tokens_total{shop_id="s_1001",purpose="generate",direction="input"} 21' in text
    )


def test_serve_with_transport_but_no_key_is_not_ready(served, monkeypatch):
    monkeypatch.delenv("RODIT_LLM_API_KEY")
    server = build_server(
        ServerSettings.from_env(),
        bindings_factory=cloud_bindings_with_transport,
        database=_Readyz(),
    )
    ready = TestClient(server.app).get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["checks"]["llm"] == {"ok": False, "reason": "missing_api_key"}
