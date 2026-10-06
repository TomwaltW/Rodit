"""模型服务差异的请求参数：extra_body、JSON 模式 / logprobs 开关、<think> 前缀、HTTP 错误的状态码。

方案 D5（境内已备案模型，OpenAI 兼容接口）：各家对 response_format、logprobs、思考模式的支持
不一样，差异用数据表达（环境变量 RODIT_LLM_EXTRA_BODY / JSON_MODE / LOGPROBS 或配置的 llm 段），
不按厂商名写分支。规则和 base_url / model 同一条（core.config.llm_endpoint）。
只用假传输层和合成值，不联网；合成的"密钥"只用来检查不外泄。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from rodit.core.config import (
    LLM_CONTROLLED_FIELDS,
    LLM_EXTRA_BODY_MAX_CHARS,
    EngineConfig,
    LLMConfig,
    LLMRequestOptions,
    check_extra_body,
    config_fingerprint,
    env_request_options,
    llm_request_options,
    load_config,
)
from rodit.core.errors import LLMError
from rodit.core.ports import LLMMessage, LLMRequest
from rodit.llm import LLMHTTPError, MeteredLLMClient, OpenAICompatibleClient
from rodit.testing.fakes import FakeClock

KEY = "sk-synthetic-provider-options-0001"  # 合成密钥，只用于测试
BASE = "https://llm.example.invalid/v1"
THINKING_OFF = {"thinking": {"type": "disabled"}}


class Transport:
    """记录请求体，按顺序回放预设的 (状态码, 响应体)；用完重复最后一个。"""

    def __init__(self, *replies: tuple[int, dict[str, Any]]) -> None:
        self.replies = list(replies) or [(200, ok("OK"))]
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, url, headers, body, timeout_s):
        self.bodies.append(body)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def ok(text: str, *, logprobs: Any = None) -> dict[str, Any]:
    choice: dict[str, Any] = {"message": {"content": text}, "finish_reason": "stop"}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return {"model": "served", "choices": [choice], "usage": {}}


def option_logprobs(*pairs: tuple[str, float]) -> dict[str, Any]:
    first = pairs[0]
    return {
        "content": [
            {
                "token": first[0],
                "logprob": first[1],
                "top_logprobs": [{"token": t, "logprob": lp} for t, lp in pairs],
            }
        ]
    }


def req(**kw: Any) -> LLMRequest:
    return LLMRequest(
        purpose=kw.pop("purpose", "route"),
        messages=[LLMMessage(role="user", content="只输出 JSON")],
        **kw,
    )


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)
    for name in ("EXTRA_BODY", "JSON_MODE", "LOGPROBS", "TIMEOUT_S"):
        monkeypatch.delenv(f"RODIT_LLM_{name}", raising=False)
    return KEY


# ================================================================ 开关：不发服务不支持的参数


def test_toggles_off_drop_response_format_and_logprobs(api_key):
    t = Transport((200, ok('{"a": 1}', logprobs=option_logprobs(("a", -0.1)))))
    c = OpenAICompatibleClient(
        base_url=BASE, model="m", transport=t, json_mode=False, logprobs=False
    )
    resp = c.complete(req(json_mode=True, want_logprobs=True, temperature=0.7))
    body = t.bodies[0]
    assert "response_format" not in body
    assert "logprobs" not in body and "top_logprobs" not in body
    assert body["temperature"] == 0.7  # 别的字段照旧
    # 服务自己多返回了 logprobs 也不读：关着就是关着
    assert resp.logprobs is None
    assert c.options == LLMRequestOptions(json_mode=False, logprobs=False)


def test_toggles_on_by_default(api_key):
    t = Transport((200, ok("a", logprobs=option_logprobs(("a", -0.1), ("b", -2.4)))))
    c = OpenAICompatibleClient(base_url=BASE, model="m", transport=t)
    resp = c.complete(req(json_mode=True, want_logprobs=True))
    assert t.bodies[0]["response_format"] == {"type": "json_object"}
    assert t.bodies[0]["logprobs"] is True and t.bodies[0]["top_logprobs"] == 5
    assert resp.logprobs is not None and set(resp.logprobs) == {"a", "b"}
    assert c.options == LLMRequestOptions()


# ================================================================ extra_body 的校验


@pytest.mark.parametrize(
    "extra",
    [
        THINKING_OFF,
        {"enable_thinking": False},
        {"reasoning_effort": "none"},
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},  # 计数不是凭据
        {"thinking_budget": 128, "top_k": 20, "prompt_cache_key": "shop-faq"},
        {"reasoning_split": True, "thinking": {"type": "disabled"}},
    ],
)
def test_ordinary_provider_parameters_are_accepted(extra):
    assert check_extra_body(extra) == extra


@pytest.mark.parametrize("field", sorted(LLM_CONTROLLED_FIELDS))
def test_controlled_fields_are_rejected(field):
    with pytest.raises(ValueError, match=f"不能设置 {field}"):
        check_extra_body({field: 1})


def test_controlled_field_spellings_are_normalised():
    with pytest.raises(ValueError, match="maxTokens"):
        check_extra_body({"maxTokens": 10})


@pytest.mark.parametrize(
    "extra",
    [
        {"api_key": "x"},
        {"apiKey": "x"},
        {"X-Api-Key": "x"},
        {"Authorization": "x"},
        {"access_token": "x"},
        {"accessToken": "x"},
        {"client_secret": "x"},
        {"password": "x"},
        {"metadata": {"auth": {"user": "x"}}},  # 任何一层
        {"headers": [{"cookie": "x"}]},
    ],
)
def test_credential_like_keys_are_rejected_without_echoing_values(extra):
    with pytest.raises(ValueError, match="看起来像凭据") as exc:
        check_extra_body(extra)
    assert "'x'" not in str(exc.value) and ": x" not in str(exc.value)


@pytest.mark.parametrize("value", [KEY, f"Bearer {KEY}", "bce-v3/ALTAK-synthetic/xyz"])
def test_secret_looking_values_are_rejected_without_echoing_them(value):
    with pytest.raises(ValueError, match="看起来像密钥") as exc:
        check_extra_body({"metadata": {"note": value}})
    assert value not in str(exc.value) and "metadata.note" in str(exc.value)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ([1, 2], "JSON 对象"),
        ("thinking", "JSON 对象"),
        ({"x": float("nan")}, "NaN"),
        ({"x": {1, 2}}, "JSON 值"),
        ({1: "x"}, "字符串"),
        ({"x": "y" * LLM_EXTRA_BODY_MAX_CHARS}, "太大"),
        ({"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}, "嵌套太深"),
    ],
)
def test_malformed_extra_body_is_rejected(extra, match):
    with pytest.raises(ValueError, match=match):
        check_extra_body(extra)


def test_client_validates_extra_body_and_refuses_the_api_key_inside_it(api_key):
    with pytest.raises(ValueError, match="看起来像凭据"):
        OpenAICompatibleClient(base_url=BASE, model="m", extra_body={"api_key": "x"})
    # 密钥换了个不像凭据的键名、值也不以 sk- 开头：发请求之前拦下，不发出去
    sneaky = KEY.removeprefix("sk-")
    t = Transport()
    c = OpenAICompatibleClient(base_url=BASE, model="m", transport=t, extra_body={"x": "ok"})
    c.complete(req())
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("RODIT_LLM_API_KEY", sneaky)
        c2 = OpenAICompatibleClient(
            base_url=BASE, model="m", transport=t, extra_body={"note": sneaky}
        )
        with pytest.raises(LLMError, match="不能放大模型密钥") as exc:
            c2.complete(req())
        assert sneaky not in str(exc.value)
    assert len(t.bodies) == 1


# ================================================================ 环境变量（from_env）


def test_from_env_reads_extra_body_and_switches_with_whitespace(api_key, monkeypatch):
    monkeypatch.setenv("RODIT_LLM_BASE_URL", BASE)
    monkeypatch.setenv("RODIT_LLM_MODEL", "m")
    monkeypatch.setenv("RODIT_LLM_EXTRA_BODY", '  {"enable_thinking": false}\n')
    monkeypatch.setenv("RODIT_LLM_JSON_MODE", " On\n")
    monkeypatch.setenv("RODIT_LLM_LOGPROBS", "off ")
    t = Transport()
    c = OpenAICompatibleClient.from_env(transport=t)
    assert c.options == LLMRequestOptions({"enable_thinking": False}, True, False)
    c.complete(req(json_mode=True, want_logprobs=True))
    body = t.bodies[0]
    assert body["enable_thinking"] is False and body["model"] == "m"
    assert body["response_format"] == {"type": "json_object"} and "logprobs" not in body
    # 和就绪检查、指纹用的同一个解析
    assert llm_request_options(LLMConfig()) == c.options == env_request_options()


@pytest.mark.parametrize("raw", ["", "  ", None])
def test_unset_or_blank_env_means_defaults(api_key, monkeypatch, raw):
    env = {"RODIT_LLM_EXTRA_BODY": raw, "RODIT_LLM_JSON_MODE": raw, "RODIT_LLM_LOGPROBS": raw}
    clean = {k: v for k, v in env.items() if v is not None}
    assert env_request_options(clean) == LLMRequestOptions()


@pytest.mark.parametrize(
    ("name", "raw", "match"),
    [
        ("EXTRA_BODY", '{"thinking": ', "不是合法的 JSON"),
        ("EXTRA_BODY", '["thinking"]', "JSON 对象"),
        ("EXTRA_BODY", '{"temperature": 0.6}', "不能设置 temperature"),
        ("EXTRA_BODY", '{"api_key": "' + KEY + '"}', "看起来像凭据"),
        ("EXTRA_BODY", "[" * 5000 + "]" * 5000, "太长"),
        ("JSON_MODE", "maybe", "只能是 on 或 off"),
        ("LOGPROBS", "2", "只能是 on 或 off"),
    ],
)
def test_invalid_env_values_fail_with_the_variable_name_but_not_the_value(
    api_key, monkeypatch, name, raw, match
):
    monkeypatch.setenv("RODIT_LLM_BASE_URL", BASE)
    monkeypatch.setenv("RODIT_LLM_MODEL", "m")
    monkeypatch.setenv(f"RODIT_LLM_{name}", raw)
    with pytest.raises(ValueError, match=match) as exc:
        OpenAICompatibleClient.from_env()
    assert f"RODIT_LLM_{name}" in str(exc.value) and KEY not in str(exc.value)


def test_env_extra_body_holding_the_api_key_value_is_rejected(api_key, monkeypatch):
    monkeypatch.setenv("RODIT_LLM_EXTRA_BODY", '{"note": "' + KEY.removeprefix("sk-") + '"}')
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY.removeprefix("sk-"))
    with pytest.raises(ValueError, match="不能放大模型密钥") as exc:
        env_request_options()
    assert KEY.removeprefix("sk-") not in str(exc.value)


# ================================================================ 配置（llm 段）和同一条规则


def test_config_options_apply_only_with_a_configured_endpoint(api_key, monkeypatch):
    monkeypatch.setenv("RODIT_LLM_BASE_URL", "https://env.example.invalid/v1")
    monkeypatch.setenv("RODIT_LLM_MODEL", "env-model")
    monkeypatch.setenv("RODIT_LLM_EXTRA_BODY", '{"enable_thinking": false}')
    monkeypatch.setenv("RODIT_LLM_LOGPROBS", "off")
    pinned = LLMConfig(
        base_url="https://cfg.example.invalid/v1",
        model="cfg-model",
        timeout_s=4.0,
        extra_body=THINKING_OFF,
        json_mode=False,
    )
    t = Transport()
    c = OpenAICompatibleClient.from_config(pinned, transport=t)
    # 配置钉了接口：整组用配置（环境变量的 extra_body、logprobs 开关都不掺进来）
    assert (c.model, c.timeout_s) == ("cfg-model", 4.0)
    assert c.options == LLMRequestOptions(THINKING_OFF, json_mode=False, logprobs=True)
    assert llm_request_options(pinned) == c.options
    c.complete(req(json_mode=True, want_logprobs=True))
    assert t.bodies[0]["thinking"] == {"type": "disabled"} and "enable_thinking" not in t.bodies[0]
    assert "response_format" not in t.bodies[0] and t.bodies[0]["logprobs"] is True
    # 没钉接口：整组取环境变量
    env_client = OpenAICompatibleClient.from_config(LLMConfig(), transport=t)
    assert env_client.model == "env-model"
    assert env_client.options == LLMRequestOptions({"enable_thinking": False}, True, False)
    assert llm_request_options(LLMConfig()) == env_client.options


def test_request_options_without_a_configured_endpoint_are_a_config_error():
    """只写了请求参数、没写接口的配置在环境变量那一路不会生效：读配置时就报错，不悄悄忽略。"""
    for llm in (
        {"extra_body": THINKING_OFF},
        {"model": "m", "json_mode": False},
        {"base_url": BASE, "logprobs": False},
    ):
        with pytest.raises(ValidationError, match="只在配置里同时写了 llm.base_url 和 llm.model"):
            EngineConfig.model_validate({"llm": llm})
    # 写的是默认值（model_dump 再读回来）照样通过
    dumped = EngineConfig().model_dump(mode="json")
    assert dumped["llm"]["json_mode"] is True and dumped["llm"]["extra_body"] is None
    assert EngineConfig.model_validate(dumped) == EngineConfig()
    assert EngineConfig.model_validate({"llm": {"extra_body": {}}}).llm.extra_body == {}


def test_config_errors_never_echo_a_secret_from_extra_body():
    with pytest.raises(ValidationError) as exc:
        EngineConfig.model_validate(
            {"llm": {"base_url": BASE, "model": "m", "extra_body": {"api_key": KEY}}}
        )
    assert "看起来像凭据" in str(exc.value) and KEY not in str(exc.value)
    with pytest.raises(ValidationError) as exc:
        LLMConfig(base_url=BASE, model="m", extra_body={"note": f"Bearer {KEY}"})
    assert KEY not in str(exc.value)


def test_config_endpoint_is_stripped_like_the_environment():
    cfg = LLMConfig(base_url=f" {BASE}\n", model=" m ")
    assert (cfg.base_url, cfg.model) == (BASE, "m")
    assert LLMConfig(base_url="  ", model="m").base_url is None  # 只有空白等于没写


def test_config_extra_body_changes_the_config_fingerprint(tmp_path):
    path = tmp_path / "rodit.yaml"
    path.write_text(
        "llm:\n"
        f"  base_url: {BASE}\n"
        "  model: synthetic-model\n"
        "  extra_body: {thinking: {type: disabled}}\n"
        "  logprobs: false\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.llm.extra_body == THINKING_OFF and cfg.llm.logprobs is False
    plain = cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"extra_body": None})})
    assert config_fingerprint(cfg) != config_fingerprint(plain)


# ================================================================ 思考过程、HTTP 错误、计量包装


@pytest.mark.parametrize(
    ("content", "text"),
    [
        (
            '<think>用户问发货，输出 {"intent": "<选项键>"}</think>\n{"intent": "a"}',
            '{"intent": "a"}',
        ),
        ('\n<THINK>想一想</THINK>{"answer": "亲"}', '{"answer": "亲"}'),
        ('{"answer": "<think> 在中间不动"}', '{"answer": "<think> 在中间不动"}'),
    ],
)
def test_leading_reasoning_block_is_removed(api_key, content, text):
    c = OpenAICompatibleClient(base_url=BASE, model="m", transport=Transport((200, ok(content))))
    assert c.complete(req()).text == text


def test_reasoning_without_an_answer_is_an_error_naming_the_fix(api_key):
    c = OpenAICompatibleClient(
        base_url=BASE, model="m", transport=Transport((200, ok("<think>想了很久…")))
    )
    with pytest.raises(LLMError, match="RODIT_LLM_EXTRA_BODY"):
        c.complete(req())


def test_http_errors_carry_status_and_code_but_no_message_text(api_key):
    body = {"error": {"type": "invalid_request_error", "code": "x", "message": f"bad {KEY}"}}
    c = OpenAICompatibleClient(base_url=BASE, model="m", transport=Transport((400, body)))
    with pytest.raises(LLMHTTPError) as exc:
        c.complete(req())
    assert isinstance(exc.value, LLMError)
    assert (exc.value.status, exc.value.code) == (400, "invalid_request_error / x")
    assert (
        "HTTP 400" in str(exc.value) and KEY not in str(exc.value) and "bad" not in str(exc.value)
    )


def test_metered_client_exposes_the_inner_options(api_key):
    inner = OpenAICompatibleClient(base_url=BASE, model="m", logprobs=False)

    class Sink:
        def record(self, event):  # noqa: ANN001
            pass

    metered = MeteredLLMClient(inner, Sink(), FakeClock())
    assert metered.options is inner.options
