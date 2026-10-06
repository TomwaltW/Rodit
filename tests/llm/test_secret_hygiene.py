"""密钥和像密钥的值不出现在任何输出里（报错、能力检查、横幅、自检报告、评测指纹）。

- 配置校验的报错（pydantic）不回显输入值：llm 段里误写的 api_key、extra_body 的值；
- 密钥中间有换行 / 空白（从两行的密钥文件读进来）：发请求之前就拒绝，报错只有变量名
  （否则 http.client 的 "Invalid header value b'Bearer …'" 会把整个密钥带出来）；
- base_url 带用户名密码：urllib 用不了，构造时就拒绝，不回显；
- RODIT_LLM_TIMEOUT_S 不是正数：报错写变量名；
- extra_body 的值不进给人看的输出和评测指纹（只有键名；指纹里是值的 SHA-256），
  像凭据的键名（app_key、sk、jwt、sign……）、像密钥的值（JWT、AKIA……）在校验时就拒绝。
只用假传输层和合成值，不联网。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pytest
from pydantic import ValidationError

from rodit.core.config import EngineConfig, LLMRequestOptions, check_extra_body
from rodit.core.errors import LLMError
from rodit.core.ports import LLMMessage, LLMRequest
from rodit.eval.llm_env import probe_capabilities, render_capabilities
from rodit.llm import OpenAICompatibleClient
from rodit.llm.openai_compat import _scrub

BASE = "https://llm.example.invalid/v1"


def _ok(url, headers, body, timeout_s) -> tuple[int, dict[str, Any]]:
    return 200, {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {}}


def _req() -> LLMRequest:
    return LLMRequest(purpose="other", messages=[LLMMessage(role="user", content="x")])


# ================================================================ 配置报错不回显输入值


@pytest.mark.parametrize(
    ("data", "secret"),
    [
        ({"llm": {"api_key": "sk-0123456789abcdefSYNTHETICKEY"}}, "0123456789abcdef"),
        (  # 只写了请求参数、没写接口：整个 llm 段的报错里不能有 extra_body 的值
            {"llm": {"extra_body": {"x": "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"}}},
            "a5b4c3d2e1f0",
        ),
        ({"decision": {"timeout_s": "secret-ish-value-777"}}, "secret-ish-value-777"),
    ],
)
def test_config_validation_errors_do_not_echo_input_values(data, secret):
    with pytest.raises(ValidationError) as exc:
        EngineConfig.model_validate(data)
    assert secret not in str(exc.value)


# ================================================================ 密钥里有换行、空白


@pytest.mark.parametrize(
    "key",
    [
        "sk-firsthalf0123456789\nsecondhalfABCDEFGHIJ",
        "sk-firsthalf0123 secondhalfABCD",
        "sk-firsthalf\x07secondhalf",
    ],
)
def test_key_with_whitespace_or_control_chars_is_refused_by_name(monkeypatch, key):
    monkeypatch.setenv("RODIT_LLM_API_KEY", key)
    sent: list[dict] = []

    def transport(url, headers, body, timeout_s):
        sent.append(headers)
        return _ok(url, headers, body, timeout_s)

    client = OpenAICompatibleClient(base_url=BASE, model="m", transport=transport)
    with pytest.raises(LLMError) as exc:
        client.complete(_req())
    text = str(exc.value)
    assert "RODIT_LLM_API_KEY" in text and "firsthalf" not in text and "secondhalf" not in text
    assert sent == []  # 没发出去
    report = probe_capabilities(client)
    lines = "\n".join(render_capabilities(report))
    assert "firsthalf" not in lines and "secondhalf" not in lines and "RODIT_LLM_API_KEY" in lines


def test_scrub_also_removes_the_repr_escaped_key():
    key = "sk-first\nsecond"
    assert "first" not in _scrub("Invalid header value b'Bearer sk-first\\nsecond'", key)


# ================================================================ base_url 带用户名密码


@pytest.mark.parametrize(
    "url",
    [
        "https://gwuser:GatewayPassw0rd@llm.example.invalid/v1",
        "https://gwuser@llm.example.invalid/v1",
        "http://:GatewayPassw0rd@127.0.0.1:9/v1",
    ],
)
def test_base_url_with_credentials_is_refused_without_echoing_them(url):
    with pytest.raises(ValueError) as exc:
        OpenAICompatibleClient(base_url=url, model="m")
    assert "GatewayPassw0rd" not in str(exc.value) and "gwuser" not in str(exc.value)
    assert "用户名或密码" in str(exc.value)


# ================================================================ 超时


@pytest.mark.parametrize("value", ["10s", "nan", "inf", "-1", "0"])
def test_timeout_env_must_be_a_positive_number(value):
    env = {"RODIT_LLM_BASE_URL": BASE, "RODIT_LLM_MODEL": "m", "RODIT_LLM_TIMEOUT_S": value}
    with pytest.raises(ValueError, match="环境变量 RODIT_LLM_TIMEOUT_S 必须是正数"):
        OpenAICompatibleClient.from_env(environ=env)


@pytest.mark.parametrize("value", [math.nan, math.inf, 0.0, -1.0])
def test_client_timeout_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="timeout_s"):
        OpenAICompatibleClient(base_url=BASE, model="m", timeout_s=value)


# ================================================================ extra_body 的值


GATEWAY = {"route_tag": "gw-0f1e2d3c4b5a69788796", "thinking": {"type": "disabled"}}


def test_human_facing_options_show_only_extra_body_keys(monkeypatch):
    monkeypatch.setenv("RODIT_LLM_API_KEY", "synthetic-key-12345678")
    client = OpenAICompatibleClient(base_url=BASE, model="m", transport=_ok, extra_body=GATEWAY)
    report = probe_capabilities(client, check_logprobs=False)
    shown = "\n".join(render_capabilities(report)) + report.model_dump_json()
    assert "gw-0f1e2d3c4b5a69788796" not in shown and "disabled" not in shown
    assert "route_tag" in shown and "thinking" in shown
    assert json.loads(client.options.summary()) == {
        "extra_body_keys": ["route_tag", "thinking"],
        "json_mode": True,
        "logprobs": True,
    }


def test_fingerprint_options_hash_the_extra_body_values():
    options = LLMRequestOptions(GATEWAY)
    described = json.loads(options.describe())
    canonical = json.dumps(GATEWAY, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert described == {
        "extra_body_keys": ["route_tag", "thinking"],
        "extra_body_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "json_mode": True,
        "logprobs": True,
    }
    assert "gw-0f1e2d3c4b5a69788796" not in options.describe()
    # 值一改指纹就变；没有 extra_body 时没有哈希
    other = LLMRequestOptions({**GATEWAY, "route_tag": "gw-other"})
    assert other.describe() != options.describe()
    assert "extra_body_sha256" not in json.loads(LLMRequestOptions().describe())


@pytest.mark.parametrize(
    "extra",
    [
        {"app_key": "x"},
        {"appKey": "x"},
        {"ak": "x"},
        {"sk": "x"},
        {"jwt": "x"},
        {"sign": "x"},
        {"passphrase": "x"},
        {"gateway": {"x_key": "x"}},
        {"key": "x"},
    ],
)
def test_more_credential_like_keys_are_rejected(extra):
    with pytest.raises(ValueError, match="看起来像凭据"):
        check_extra_body(extra)


@pytest.mark.parametrize(
    "value", ["eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.c2lnbmF0dXJl", "AKIAIOSFODNN7EXAMPLE"]
)
def test_more_secret_looking_values_are_rejected(value):
    with pytest.raises(ValueError, match="看起来像密钥") as exc:
        check_extra_body({"note": value})
    assert value not in str(exc.value)


def test_known_benign_keys_still_pass():
    extra = {"prompt_cache_key": "shop-faq", "top_k": 20, "keyword_boost": True}
    assert check_extra_body(extra) == extra
