"""大模型请求参数（RODIT_LLM_EXTRA_BODY / JSON_MODE / LOGPROBS 或配置的 llm 段）在部署里的一条规则：
启动前检查拦下不合法的值、就绪检查报原因、生产组装的客户端发的就是这些参数——和评测指纹
（eval.fingerprint.running_llm_options）同一个解析（core.config.llm_request_options）。

合成环境变量和合成密钥；数据库不真的连（FakeReadinessDb），大模型不调用。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ops_support import SYNTHETIC_LLM_KEY, FakeReadinessDb

from rodit.core.config import EngineConfig, LLMConfig, LLMRequestOptions, load_config
from rodit.eval.fingerprint import running_llm_options
from rodit.ops.health import llm_probe
from rodit.ops.serve import StartupError, build_server
from rodit.ops.settings import ServerSettings
from rodit.storage.db import MIGRATIONS

ALL_VERSIONS = tuple(m.version for m in MIGRATIONS)

ENV = {
    "RODIT_LLM_BASE_URL": "https://llm.example.invalid/v1",
    "RODIT_LLM_MODEL": "m",
    "RODIT_LLM_API_KEY": "sk-synthetic-ops-options",
}


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("RODIT_LLM_EXTRA_BODY", "{not json"),
        ("RODIT_LLM_EXTRA_BODY", '{"api_key": "sk-synthetic-ops-options"}'),
        ("RODIT_LLM_JSON_MODE", "sometimes"),
        ("RODIT_LLM_LOGPROBS", "2"),
    ],
)
def test_readiness_reports_invalid_request_options_without_values(name, raw):
    result = llm_probe(LLMConfig(), environ={**ENV, name: raw})()["llm"].as_dict()
    assert result == {"ok": False, "reason": "invalid_request_options"}


def test_readiness_accepts_valid_request_options():
    env = {
        **ENV,
        "RODIT_LLM_EXTRA_BODY": '{"enable_thinking": false}',
        "RODIT_LLM_JSON_MODE": "off",
        "RODIT_LLM_LOGPROBS": "OFF",
    }
    assert llm_probe(LLMConfig(), environ=env)()["llm"].ok


def _server(monkeypatch, **env: str):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return build_server(ServerSettings.from_env(), database=FakeReadinessDb(ALL_VERSIONS))


def test_startup_refuses_invalid_request_options(ops_env, token_file, monkeypatch):
    secret = f"Bearer {SYNTHETIC_LLM_KEY}"
    with pytest.raises(StartupError, match="RODIT_LLM_EXTRA_BODY") as exc:
        _server(monkeypatch, RODIT_LLM_EXTRA_BODY='{"metadata": {"note": "' + secret + '"}}')
    assert SYNTHETIC_LLM_KEY not in str(exc.value) and "看起来像密钥" in str(exc.value)
    with pytest.raises(StartupError, match="RODIT_LLM_LOGPROBS"):
        _server(monkeypatch, RODIT_LLM_EXTRA_BODY="", RODIT_LLM_LOGPROBS="maybe")


def test_production_unit_sends_the_environment_options(ops_env, token_file, monkeypatch):
    server = _server(
        monkeypatch,
        RODIT_LLM_EXTRA_BODY=' {"thinking": {"type": "disabled"}}\n',
        RODIT_LLM_LOGPROBS="off",
    )
    options = server.unit.llm.options  # type: ignore[attr-defined]  # MeteredLLMClient 转发
    assert options == LLMRequestOptions({"thinking": {"type": "disabled"}}, True, False)
    # 线上指纹、就绪检查用的是同一个解析
    assert running_llm_options(EngineConfig()) == options.describe()
    assert server.readiness.check().as_dict()["checks"]["llm"]["ok"] is True


def test_production_unit_uses_the_pinned_config_options(
    ops_env, token_file, monkeypatch, tmp_path: Path
):
    path = tmp_path / "rodit.yaml"
    path.write_text(
        "llm:\n"
        "  base_url: https://cfg-llm.example.invalid/v1\n"
        "  model: cfg-model\n"
        "  extra_body: {enable_thinking: false}\n"
        "  json_mode: false\n",
        encoding="utf-8",
    )
    server = _server(
        monkeypatch,
        RODIT_CONFIG=str(path),
        RODIT_LLM_EXTRA_BODY='{"thinking": {"type": "disabled"}}',  # 配置钉了接口：不看
        RODIT_LLM_LOGPROBS="off",
    )
    unit_options = server.unit.llm.options  # type: ignore[attr-defined]
    assert unit_options == LLMRequestOptions({"enable_thinking": False}, False, True)
    assert server.unit.llm.model == "cfg-model"
    assert running_llm_options(load_config(path)) == unit_options.describe()
