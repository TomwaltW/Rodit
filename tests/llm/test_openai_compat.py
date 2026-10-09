"""OpenAICompatibleClient：请求构造、响应解析、错误映射、密钥不外泄（只用假传输层，不访问网络）。"""

from __future__ import annotations

import base64
import http.client
import io
import json
import math
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import pytest

from rodit.core.errors import LLMError, LLMTimeout
from rodit.core.ports import LLMClient, LLMMessage, LLMRequest
from rodit.llm import OpenAICompatibleClient, openai_compat, urllib_transport

KEY = "sk-synthetic-test-key-not-real-0001"  # 合成密钥，只用于测试
BASE = "https://llm.example.invalid/compatible-mode/v1"
# 合成密钥的两半：中间夹一个字符，看两半会不会分别漏出去（转义之后整串对不上，按原文抹不掉）
HEAD, TAIL = "sk-synthetic-HEADPART", "TAILPART-0004"


class FakeTransport:
    """记录每次调用；按预设返回 (状态码, 响应体) 或抛出异常。"""

    def __init__(self, status: int = 200, body: Any = None, exc: Exception | None = None):
        self.status = status
        self.body = body if body is not None else ok_body()
        self.exc = exc
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(self, url, headers, body, timeout_s):
        self.calls.append((url, headers, body, timeout_s))
        if self.exc is not None:
            raise self.exc
        return self.status, self.body


def ok_body(text: str = "亲，默认发中通快递哦~", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-synthetic",
        "model": "qwen-synthetic-served",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 321, "completion_tokens": 17, "total_tokens": 338},
    }
    body.update(extra)
    return body


def req(**kw: Any) -> LLMRequest:
    return LLMRequest(
        purpose=kw.pop("purpose", "generate"),
        messages=kw.pop(
            "messages",
            [
                LLMMessage(role="system", content="你是小雅女装旗舰店的客服"),
                LLMMessage(role="user", content="发什么快递"),
            ],
        ),
        **kw,
    )


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)
    return KEY


def client(transport: FakeTransport, **kw: Any) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(base_url=BASE, model="qwen-synthetic", transport=transport, **kw)


def cause(error: LLMError) -> tuple[Any, Any, Any]:
    """失败原因（core.errors.LLMError）：状态码、哪一类、模型服务给的代码。"""
    return error.status, error.kind, error.code


# ---------------------------------------------------------------- 请求


def test_implements_llm_client_protocol():
    assert isinstance(client(FakeTransport()), LLMClient)


def test_request_payload_headers_and_default_timeout(api_key):
    t = FakeTransport()
    resp = client(t).complete(req(temperature=0.3, max_tokens=400))
    assert resp.text == "亲，默认发中通快递哦~"
    [(url, headers, body, timeout)] = t.calls
    assert url == BASE + "/chat/completions"
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert headers["Content-Type"] == "application/json"
    assert body == {
        "model": "qwen-synthetic",
        "messages": [
            {"role": "system", "content": "你是小雅女装旗舰店的客服"},
            {"role": "user", "content": "发什么快递"},
        ],
        "temperature": 0.3,
        "max_tokens": 400,
        "stream": False,
    }
    assert timeout == 10.0
    # purpose / shop_id 只用于本地计量，不发给模型服务
    assert "purpose" not in body and "shop_id" not in body


def test_json_mode_and_logprobs_flags(api_key):
    t = FakeTransport()
    client(t, top_logprobs=7).complete(req(json_mode=True, want_logprobs=True))
    body = t.calls[0][2]
    assert body["response_format"] == {"type": "json_object"}
    assert body["logprobs"] is True and body["top_logprobs"] == 7


def test_request_timeout_overrides_default(api_key):
    t = FakeTransport()
    c = client(t, timeout_s=8.0)
    c.complete(req(timeout_s=0.8))
    c.complete(req())
    assert [call[3] for call in t.calls] == [0.8, 8.0]


def test_base_url_trailing_slash_and_extra_body(api_key):
    t = FakeTransport()
    c = OpenAICompatibleClient(
        base_url=BASE + "/",
        model="qwen-synthetic",
        transport=t,
        extra_body={"enable_thinking": False, "model": "不能覆盖标准字段"},
    )
    c.complete(req())
    url, _, body, _ = t.calls[0]
    assert url == BASE + "/chat/completions"
    assert body["enable_thinking"] is False and body["model"] == "qwen-synthetic"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": "llm.example.invalid/v1"},
        {"model": ""},
        {"timeout_s": 0},
        {"top_logprobs": 0},
        {"top_logprobs": 21},
        {"api_key_env": ""},
    ],
)
def test_constructor_validation(kwargs):
    args = {"base_url": BASE, "model": "m", **kwargs}
    with pytest.raises(ValueError):
        OpenAICompatibleClient(**args)


def test_from_env(monkeypatch):
    monkeypatch.setenv("RODIT_LLM_BASE_URL", BASE)
    monkeypatch.setenv("RODIT_LLM_MODEL", "qwen-synthetic")
    monkeypatch.setenv("RODIT_LLM_TIMEOUT_S", "3.5")
    c = OpenAICompatibleClient.from_env(transport=FakeTransport())
    assert (c.base_url, c.model, c.timeout_s, c.api_key_env) == (
        BASE,
        "qwen-synthetic",
        3.5,
        "RODIT_LLM_API_KEY",
    )
    monkeypatch.delenv("RODIT_LLM_MODEL")
    with pytest.raises(RuntimeError, match="RODIT_LLM_MODEL"):
        OpenAICompatibleClient.from_env()


def test_from_env_strips_whitespace_like_llm_endpoint(monkeypatch, api_key):
    """从密钥文件注入的值常带换行、空格：和 core.config.llm_endpoint 一样去掉首尾空白。
    否则就绪检查说配好了、启动时却因为地址前面有空格崩掉；模型名多一个换行，请求里的
    model 和评测指纹记下的都跟线上算出来的对不上。"""
    from rodit.core.config import LLMConfig, llm_endpoint

    monkeypatch.setenv("RODIT_LLM_BASE_URL", f" {BASE}\n")
    monkeypatch.setenv("RODIT_LLM_MODEL", "qwen-synthetic \n")
    monkeypatch.setenv("RODIT_LLM_TIMEOUT_S", " 3.5\n")
    transport = FakeTransport()
    c = OpenAICompatibleClient.from_env(transport=transport)
    assert (c.base_url, c.model, c.timeout_s) == (BASE, "qwen-synthetic", 3.5)
    assert llm_endpoint(LLMConfig())[:2] == (c.base_url, c.model)
    c.complete(req())
    url, _, body, _ = transport.calls[0]
    assert url == f"{BASE}/chat/completions" and body["model"] == "qwen-synthetic"

    monkeypatch.setenv("RODIT_LLM_MODEL", " \n")  # 只有空白：和没设一样
    with pytest.raises(RuntimeError, match="RODIT_LLM_MODEL"):
        OpenAICompatibleClient.from_env()


# ---------------------------------------------------------------- 响应


def test_usage_and_model_are_mapped(api_key):
    resp = client(FakeTransport()).complete(req())
    assert (resp.model, resp.input_tokens, resp.output_tokens) == ("qwen-synthetic-served", 321, 17)
    assert resp.logprobs is None


def test_missing_usage_and_model_fall_back(api_key):
    body = ok_body()
    del body["usage"], body["model"]
    resp = client(FakeTransport(body=body)).complete(req())
    assert (resp.model, resp.input_tokens, resp.output_tokens) == ("qwen-synthetic", 0, 0)


def test_segmented_content_is_joined(api_key):
    body = ok_body()
    body["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "亲，"},
        {"type": "text", "text": "在的~"},
    ]
    assert client(FakeTransport(body=body)).complete(req()).text == "亲，在的~"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_calls"),
        (None, None),
        ("", None),  # 空字符串等于没给
        (3, None),  # 不是字符串：当没给
        (["length"], None),
    ],
)
def test_finish_reason_is_passed_through(api_key, given, expected):
    body = ok_body()
    body["choices"][0]["finish_reason"] = given
    assert client(FakeTransport(body=body)).complete(req()).finish_reason == expected


def test_finish_reason_absent_is_none(api_key):
    body = ok_body()
    del body["choices"][0]["finish_reason"]
    assert client(FakeTransport(body=body)).complete(req()).finish_reason is None


@pytest.mark.parametrize("text", ["", "   ", '{"answer": "亲，这款'])
def test_truncated_or_empty_output_is_returned_with_its_tokens(api_key, text):
    """被 max_tokens 截断（finish_reason length）、文本是空的：照常返回，不抛异常。token 已经花了，
    抛异常的话这次调用按 0 token 计量；这一票算不算数由调用方定（判断层、生成各有各的处理）。"""
    body = ok_body(text=text)
    body["choices"][0]["finish_reason"] = "length"
    body["usage"] = {"prompt_tokens": 100, "completion_tokens": 304}
    resp = client(FakeTransport(body=body)).complete(req(max_tokens=304))
    assert (resp.text, resp.finish_reason) == (text, "length")
    assert (resp.input_tokens, resp.output_tokens) == (100, 304)


def _lp(token: str, p: float, top: list[tuple[str, float]] | None = None) -> dict[str, Any]:
    pos: dict[str, Any] = {"token": token, "logprob": math.log(p), "bytes": None}
    if top is not None:
        pos["top_logprobs"] = [{"token": t, "logprob": math.log(q)} for t, q in top]
    return pos


def test_logprobs_become_option_probabilities(api_key):
    body = ok_body(text='"A"')
    body["choices"][0]["logprobs"] = {
        "content": [
            _lp('"', 0.99, [('"', 0.99)]),  # 只有引号：跳过
            _lp("A", 0.7, [("A", 0.7), (" B", 0.2), ("B", 0.05), ("  ", 0.01)]),
            _lp('"', 0.99),
        ]
    }
    resp = client(FakeTransport(body=body)).complete(req(want_logprobs=True))
    assert resp.logprobs is not None
    assert resp.logprobs.keys() == {"A", "B"}
    assert resp.logprobs["A"] == pytest.approx(0.7)
    assert resp.logprobs["B"] == pytest.approx(0.25)  # " B" 和 "B" 合并


def test_logprobs_without_top_candidates_use_chosen_token(api_key):
    body = ok_body(text="size_advice")
    body["choices"][0]["logprobs"] = {"content": [_lp("size", 0.6)]}
    resp = client(FakeTransport(body=body)).complete(req(want_logprobs=True))
    assert resp.logprobs == {"size": pytest.approx(0.6)}


def test_nan_logprobs_are_ignored(api_key):
    body = ok_body(text="A")
    body["choices"][0]["logprobs"] = {
        "content": [
            {
                "token": "A",
                "logprob": float("nan"),
                "top_logprobs": [
                    {"token": "A", "logprob": float("nan")},
                    {"token": "B", "logprob": math.log(0.1)},
                ],
            }
        ]
    }
    resp = client(FakeTransport(body=body)).complete(req(want_logprobs=True))
    assert resp.logprobs == {"B": pytest.approx(0.1)}


def test_logprobs_absent_or_not_requested(api_key):
    body = ok_body()
    assert client(FakeTransport(body=body)).complete(req(want_logprobs=True)).logprobs is None
    body["choices"][0]["logprobs"] = {"content": [_lp("A", 0.9, [("A", 0.9)])]}
    assert client(FakeTransport(body=body)).complete(req()).logprobs is None  # 没要就不给
    body["choices"][0]["logprobs"] = {"content": "garbage"}
    assert client(FakeTransport(body=body)).complete(req(want_logprobs=True)).logprobs is None


# ---------------------------------------------------------------- 错误


@pytest.mark.parametrize(
    ("status", "fragment"),
    [(429, "限流"), (500, "HTTP 500"), (502, "HTTP 502"), (503, "HTTP 503"), (400, "HTTP 400")],
)
def test_http_errors_become_llm_error(api_key, status, fragment):
    body = {"error": {"message": "买家原话：我的手机号 13800000000", "type": "server_error"}}
    with pytest.raises(LLMError) as ei:
        client(FakeTransport(status=status, body=body)).complete(req())
    assert not isinstance(ei.value, LLMTimeout)
    assert fragment in str(ei.value) and "server_error" in str(ei.value)
    assert "13800000000" not in str(ei.value)  # 错误原文不进异常消息


@pytest.mark.parametrize("exc", [TimeoutError("read timed out"), TimeoutError()])
def test_transport_timeout_becomes_llm_timeout(api_key, exc):
    with pytest.raises(LLMTimeout, match="超时"):
        client(FakeTransport(exc=exc)).complete(req(timeout_s=0.5))


@pytest.mark.parametrize(
    "exc",
    [
        urllib.error.URLError("Name or service not known"),
        ConnectionResetError("reset by peer"),
        http.client.IncompleteRead(b"partial"),
        ValueError("响应体不是 JSON 对象"),
    ],
)
def test_network_and_decode_errors_become_llm_error(api_key, exc):
    with pytest.raises(LLMError) as ei:
        client(FakeTransport(exc=exc)).complete(req())
    assert not isinstance(ei.value, LLMTimeout)
    assert ei.value.__cause__ is None  # 不链原始异常，避免带出传输层细节


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": "oops"},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": "text"}]},
        ["not", "a", "dict"],
    ],
)
def test_malformed_body_becomes_llm_error(api_key, body):
    with pytest.raises(LLMError, match="格式不对"):
        client(FakeTransport(body=body)).complete(req())


def test_content_filter_becomes_llm_error(api_key):
    body = ok_body(text="")
    body["choices"][0]["finish_reason"] = "content_filter"
    with pytest.raises(LLMError, match="内容安全"):
        client(FakeTransport(body=body)).complete(req())


# ---------------------------------------------------------------- 失败原因（status / kind / code）


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (429, "rate_limit"),
        (401, "auth"),
        (403, "auth"),
        (402, "billing"),
        (400, "bad_request"),
        (404, "bad_request"),
        (422, "bad_request"),
        (500, "server"),
        (502, "server"),
        (503, "server"),
        (504, "server"),
        (301, "redirect"),
        (302, "redirect"),
        (307, "redirect"),
    ],
)
def test_http_errors_carry_status_kind_and_vendor_code(api_key, status, kind):
    body = {
        "error": {
            "message": "买家原话：我的手机号 13800000000",
            "type": "synthetic_error_type",
            "code": "synthetic_code",
        }
    }
    with pytest.raises(LLMError) as ei:
        client(FakeTransport(status=status, body=body)).complete(req())
    assert cause(ei.value) == (status, kind, "synthetic_code")
    assert "13800000000" not in repr(vars(ei.value))  # 错误原文不进失败原因


@pytest.mark.parametrize(
    ("body", "code"),
    [
        # error.code 优先，没有（或者是空的）用 error.type
        (
            {"error": {"type": "invalid_request_error", "code": "invalid_api_key"}},
            "invalid_api_key",
        ),
        ({"error": {"type": "rate_limit_reached_error"}}, "rate_limit_reached_error"),
        ({"error": {"type": "quota_error", "code": None}}, "quota_error"),
        ({"error": {"type": "quota_error", "code": "  "}}, "quota_error"),
        ({"error": {"code": 1113, "message": "余额不足"}}, "1113"),  # 数字代码
        ({"error": {"code": " padded_code \n"}}, "padded_code"),
        ({"error": {"code": "c" * 100}}, "c" * 64),  # 最多 64 个字符
        # 读不出代码：不猜，更不拿错误消息原文顶上
        ({"error": {"message": "只有消息原文，买家手机号 13800000000"}}, None),
        ({"error": {"code": True, "type": False}}, None),
        ({"error": {"code": {"nested": 1}, "type": ["list"]}}, None),
        ({"error": "错误是一句话，不是对象"}, None),
        ({"code": "top_level_code_is_not_read"}, None),
        ({}, None),
    ],
)
def test_vendor_error_code_is_short_and_never_the_message(api_key, body, code):
    with pytest.raises(LLMError) as ei:
        client(FakeTransport(status=400, body=body)).complete(req())
    assert cause(ei.value) == (400, "bad_request", code)


@pytest.mark.parametrize("exc", [TimeoutError("read timed out"), TimeoutError()])
def test_timeout_has_the_timeout_kind(api_key, exc):
    with pytest.raises(LLMTimeout) as ei:
        client(FakeTransport(exc=exc)).complete(req(timeout_s=0.5))
    assert cause(ei.value) == (None, "timeout", None)


@pytest.mark.parametrize(
    ("exc", "kind", "types"),
    [
        (urllib.error.URLError("Name or service not known"), "network", "URLError"),
        (
            urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
            "network",
            "URLError（ConnectionRefusedError）",
        ),
        (
            urllib.error.URLError(socket.gaierror(8, "nodename nor servname provided")),
            "network",
            "URLError（gaierror）",
        ),
        (ConnectionResetError("reset by peer"), "network", "ConnectionResetError"),
        (http.client.IncompleteRead(b"partial"), "network", "IncompleteRead"),
        (http.client.RemoteDisconnected("closed"), "network", "RemoteDisconnected"),
        (RuntimeError("平台通道故障"), "network", "RuntimeError"),
        # 按传输层的约定，ValueError 是 2xx 但响应体不是 JSON 对象
        (ValueError("响应体不是 JSON 对象"), "malformed", "ValueError"),
        (json.JSONDecodeError("Expecting value", "<html>", 0), "malformed", "JSONDecodeError"),
    ],
)
def test_transport_failures_have_a_kind_and_report_types_only(api_key, exc, kind, types):
    """没收到能用的响应：没有状态码；消息里只有异常的类型名（URLError 连同里面那一层），
    不带异常的文字。"""
    with pytest.raises(LLMError) as ei:
        client(FakeTransport(exc=exc)).complete(req())
    assert not isinstance(ei.value, LLMTimeout)
    assert cause(ei.value) == (None, kind, None)
    assert str(ei.value) == f"大模型接口调用失败：{types}"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (200, {}, (200, "malformed", None)),
        (200, {"choices": []}, (200, "malformed", None)),
        (200, {"choices": [{"message": {"content": None}}]}, (200, "malformed", None)),
        (201, {"choices": [{"message": "text"}]}, (201, "malformed", None)),
        (200, ["not", "a", "dict"], (200, "malformed", None)),
        (502, ["not", "a", "dict"], (502, "malformed", None)),  # 状态码照带
    ],
)
def test_unreadable_body_is_malformed_and_keeps_the_status(api_key, status, body, expected):
    with pytest.raises(LLMError, match="格式不对") as ei:
        client(FakeTransport(status=status, body=body)).complete(req())
    assert cause(ei.value) == expected


def test_error_object_in_a_200_body_gives_the_vendor_code(api_key):
    """有的模型服务出错时也回 HTTP 200，错误放在响应体的 error 里：读不出回答（malformed），
    代码照样带上，不然只看得到"没有 choices"。"""
    body = {"error": {"code": "InvalidApiKey", "message": "Invalid API-key provided."}}
    with pytest.raises(LLMError, match="没有 choices") as ei:
        client(FakeTransport(body=body)).complete(req())
    assert cause(ei.value) == (200, "malformed", "InvalidApiKey")


@pytest.mark.parametrize("reason", ["content_filter", "sensitive"])
def test_content_filter_has_its_own_kind(api_key, reason):
    body = ok_body(text="")
    body["choices"][0]["finish_reason"] = reason
    with pytest.raises(LLMError, match="内容安全") as ei:
        client(FakeTransport(body=body)).complete(req())
    # 不带状态码：评测按 kind 归类（rodit.eval.health.error_label 先看状态码，带上 200 的话
    # 归成 http_200，看不出是内容安全）
    assert cause(ei.value) == (None, "content_filter", None)


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (
            LLMError("通道限流", status=429, kind="rate_limit", code="channel_busy"),
            (429, "rate_limit", "channel_busy"),
        ),
        (LLMError("通道故障"), (None, None, None)),  # 没带原因：不替它猜
        (LLMTimeout("通道超时"), (None, "timeout", None)),
        (
            LLMTimeout("网关超时", status=504, code="gateway_timeout"),
            (504, "timeout", "gateway_timeout"),
        ),
        (LLMTimeout("通道不通", kind="network"), (None, "network", None)),
        (LLMError("余额不足", status=402, kind="billing", code=1113), (402, "billing", "1113")),
        # code 里带出了密钥：和消息一样抹掉（这一项会原样进审计）
        (
            LLMError(f"通道回显了 {KEY}", status=401, kind="auth", code=f"bad_key:{KEY}"),
            (401, "auth", "bad_key:***"),
        ),
        (LLMError("x", kind="server", code="c" * 100), (None, "server", "c" * 64)),
    ],
)
def test_llm_error_from_the_transport_keeps_its_cause(api_key, raised, expected):
    """传输层（比如平台云的通道）自己抛的 LLMError：抹掉密钥后按原类型重抛，失败原因照带。"""
    with pytest.raises(LLMError) as ei:
        client(FakeTransport(exc=raised)).complete(req())
    assert type(ei.value) is type(raised) and ei.value is not raised
    assert cause(ei.value) == expected
    assert str(ei.value) == str(raised).replace(KEY, "***")
    assert ei.value.__cause__ is None


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_api_key(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("RODIT_LLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("RODIT_LLM_API_KEY", value)
    t = FakeTransport()
    with pytest.raises(LLMError, match="RODIT_LLM_API_KEY") as ei:
        client(t).complete(req())
    assert t.calls == []
    assert cause(ei.value) == (None, "config", None)  # 本地配置不对，请求没发出去


def test_api_key_is_read_at_call_time(monkeypatch):
    t = FakeTransport()
    c = client(t, api_key_env="RODIT_TEST_LLM_KEY")
    monkeypatch.setenv("RODIT_TEST_LLM_KEY", "sk-synthetic-first")
    c.complete(req())
    monkeypatch.setenv("RODIT_TEST_LLM_KEY", "sk-synthetic-rotated")
    c.complete(req())
    assert [call[1]["Authorization"] for call in t.calls] == [
        "Bearer sk-synthetic-first",
        "Bearer sk-synthetic-rotated",
    ]


@pytest.mark.parametrize(
    "transport",
    [
        FakeTransport(exc=OSError(f"proxy said: bad header Authorization: Bearer {KEY}")),
        FakeTransport(exc=LLMError(f"custom transport leaked {KEY}")),
        FakeTransport(exc=LLMTimeout(f"custom timeout leaked {KEY}")),
        FakeTransport(status=401, body={"error": {"type": f"invalid_key {KEY}", "code": KEY}}),
    ],
)
def test_api_key_never_leaks_into_exceptions(api_key, transport):
    c = client(transport)
    with pytest.raises(LLMError) as ei:
        c.complete(req())
    # vars(异常)：失败原因（status / kind / code）会原样进审计和评测报告
    for text in (str(ei.value), repr(ei.value), repr(vars(ei.value)), repr(c), repr(vars(c))):
        assert KEY not in text


@pytest.mark.parametrize(
    "bad_key",
    [
        f"{HEAD}\n{TAIL}",  # .env 里带引号的多行取值
        f"{HEAD}\r{TAIL}",
        f"{HEAD}\r\n{TAIL}",
        f"“{HEAD}{TAIL}”",  # 粘贴时带上了中文引号
        f"{HEAD}é{TAIL}",  # latin-1 编得出来，http.client 会放行：照样不收
        f"{HEAD}\N{LINE SEPARATOR}{TAIL}",  # Unicode 的行分隔符
    ],
    ids=["lf", "cr", "crlf", "curly-quotes", "latin-1", "line-separator"],
)
def test_unusable_api_key_is_rejected_without_echoing_it(monkeypatch, bad_key):
    """密钥中间有换行、有非 ASCII 字符：读出来就拦下，消息里只有变量名。交给 http.client 的话，
    它报"请求头不合法"时会把整个头值（转义过的密钥）打出来。"""
    monkeypatch.setenv("RODIT_LLM_API_KEY", bad_key)
    t = FakeTransport()
    c = client(t)
    with pytest.raises(LLMError, match="RODIT_LLM_API_KEY") as ei:
        c.complete(req())
    assert t.calls == []  # 请求没发出去
    assert not isinstance(ei.value, LLMTimeout)
    assert cause(ei.value) == (None, "config", None)
    for text in (str(ei.value), repr(ei.value), repr(vars(ei.value)), repr(c), repr(vars(c))):
        assert HEAD not in text and TAIL not in text


@pytest.mark.parametrize("tail", ["\n", "\r\n", " \t\r\n"])
def test_line_break_after_the_api_key_is_still_stripped(monkeypatch, tail):
    """从密钥文件、CRLF 的 .env 注入的值末尾常带换行：去掉就能用，不算密钥不合法。"""
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY + tail)
    t = FakeTransport()
    client(t).complete(req())
    assert t.calls[0][1]["Authorization"] == f"Bearer {KEY}"


ODD_KEY = f"{HEAD}\\{TAIL}"  # 合法（ASCII、没有换行），但带反斜杠：转义之后和原文对不上
_ODD_HEADER = f"Bearer {ODD_KEY}"


@pytest.mark.parametrize(
    "exc",
    [
        # http.client 报请求头不合法的写法：整个头值的 bytes repr
        ValueError(f"Invalid header value {_ODD_HEADER.encode()!r}"),
        OSError(f"proxy said: bad header Authorization: {_ODD_HEADER!r}"),
        OSError(f"proxy said: {json.dumps({'Authorization': _ODD_HEADER})}"),
        OSError(f"cannot send {urllib.parse.quote(_ODD_HEADER, safe='')}"),
        OSError(f"rejected {base64.b64encode(_ODD_HEADER.encode()).decode()}"),
        OSError(5, f"strerror with {_ODD_HEADER!r}"),
        urllib.error.URLError(OSError(f"tunnel refused {_ODD_HEADER!r}")),
        urllib.error.URLError(f"reason is text: {_ODD_HEADER!r}"),
        RuntimeError("first arg is clean", _ODD_HEADER),
    ],
    ids=[
        "header-bytes-repr",
        "str-repr",
        "json",
        "percent-encoded",
        "base64",
        "strerror",
        "urlerror-reason",
        "urlerror-text",
        "second-arg",
    ],
)
def test_api_key_cannot_leak_in_an_escaped_form(monkeypatch, exc):
    """传输层的异常文字里带着转义过的密钥（repr、JSON、百分号编码、base64……）：按原文替换
    抹不干净，所以这类异常只报类型名，文字一概不带。"""
    monkeypatch.setenv("RODIT_LLM_API_KEY", ODD_KEY)
    c = client(FakeTransport(exc=exc))
    with pytest.raises(LLMError) as ei:
        c.complete(req())
    escaped = (
        HEAD,
        TAIL,
        urllib.parse.quote(_ODD_HEADER, safe=""),
        base64.b64encode(_ODD_HEADER.encode()).decode(),
        base64.b64encode(ODD_KEY.encode()).decode(),
    )
    for text in (str(ei.value), repr(ei.value), repr(vars(ei.value)), repr(c), repr(vars(c))):
        for fragment in escaped:
            assert fragment not in text
    assert str(ei.value).startswith("大模型接口调用失败：")


URL_SECRETS = ("svcuser", "URLPASSWORD", "PATHSECRET", "QUERYSECRET")
SECRET_URL = (
    "https://svcuser:URLPASSWORD@llm.example.invalid:8443/v1/tenant-PATHSECRET?sig=QUERYSECRET"
)


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda url: OSError(f"cannot reach {url}"),
        lambda url: http.client.InvalidURL(
            f"URL can't contain control characters. {urllib.parse.urlsplit(url).path!r}"
        ),
        lambda url: http.client.InvalidURL(
            f"nonnumeric port: {urllib.parse.urlsplit(url).netloc!r}"
        ),
        lambda url: ConnectionError(
            "HTTPSConnectionPool(host='llm.example.invalid', port=8443): Max retries exceeded "
            f"with url: {url.split('8443', 1)[1]}"
        ),
        lambda url: urllib.error.URLError(OSError(f"connect to {url} failed")),
        lambda url: ValueError(f"not JSON, from {urllib.parse.quote(url, safe='')}"),
    ],
    ids=["whole-url", "path-repr", "netloc", "path-and-query", "urlerror-reason", "quoted"],
)
def test_transport_errors_never_echo_the_base_url(api_key, make_exc):
    """传输层的异常文字常带着请求地址（requests / httpx 的连接错误、http.client 的 InvalidURL）：
    地址里的路径、查询参数、用户名密码都不能进消息。"""

    def transport(url, headers, body, timeout_s):
        raise make_exc(url)

    c = OpenAICompatibleClient(base_url=SECRET_URL, model="m", transport=transport)
    with pytest.raises(LLMError) as ei:
        c.complete(req())
    for text in (str(ei.value), repr(ei.value), repr(vars(ei.value))):
        for secret in (*URL_SECRETS, KEY):
            assert secret not in text


@pytest.fixture
def no_network(monkeypatch):
    """走真正的默认出口（urllib），套接字全部封死：下面的用例都是还没连接就出错的，
    真去连的话这里先报错。"""
    for name in (
        "http_proxy",
        "HTTP_PROXY",
        "https_proxy",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "*")  # 也不看系统的代理设置

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("测试不该联网")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.mark.parametrize(
    "base_url",
    [
        # 地址里多了一个空格：http.client 把请求路径（https）或整个地址（http）引在报错里
        "https://llm.example.invalid/v1/tenant-PATHSECRET x?sig=QUERYSECRET",
        "http://llm.example.invalid/v1/tenant-PATHSECRET x?sig=QUERYSECRET",
        "https://llm.example.invalid/v1/tenant-PATHSECRET\nx?sig=QUERYSECRET",
        # 带用户名密码、没写端口：密码被当成端口，报"端口不是数字"时原样打出来
        "https://svcuser:URLPASSWORD@llm.example.invalid/v1",
        "https://llm.example.invalid:PORTSECRET/v1",
    ],
    ids=["https-space", "http-space", "newline", "userinfo-without-port", "bad-port"],
)
def test_default_transport_never_echoes_the_base_url(no_network, api_key, base_url):
    c = OpenAICompatibleClient(base_url=base_url, model="m")
    with pytest.raises(LLMError) as ei:
        c.complete(req())
    # 确实是 http.client 在连接之前拒的（不是上面封套接字的那个 AssertionError）
    assert str(ei.value) == "大模型接口调用失败：InvalidURL"
    assert cause(ei.value) == (None, "network", None)
    for text in (str(ei.value), repr(ei.value), repr(vars(ei.value))):
        for secret in (*URL_SECRETS, "PORTSECRET", KEY):
            assert secret not in text


def test_default_transport_never_sees_a_key_with_a_line_break(no_network, monkeypatch):
    """transport-11 的原样复现：默认出口 + 中间有换行的密钥。以前 http.client 报
    Invalid header value b'Bearer …'，密钥的两半都在消息里。"""
    monkeypatch.setenv("RODIT_LLM_API_KEY", f"{HEAD}\n{TAIL}")
    c = OpenAICompatibleClient(base_url=BASE, model="m")
    with pytest.raises(LLMError, match="RODIT_LLM_API_KEY") as ei:
        c.complete(req())
    assert ei.value.kind == "config"
    assert HEAD not in str(ei.value) and TAIL not in str(ei.value)


# ---------------------------------------------------------------- 默认传输层（urllib，打桩不联网）


class _FakeHTTPResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_urllib_transport_success(monkeypatch):
    seen: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float):
        seen.update(
            url=request.full_url,
            method=request.get_method(),
            auth=request.get_header("Authorization"),
            body=json.loads(request.data),
            timeout=timeout,
        )
        return _FakeHTTPResponse(json.dumps(ok_body()).encode())

    monkeypatch.setattr(openai_compat, "_urlopen", fake_urlopen)
    status, body = urllib_transport(
        BASE + "/chat/completions", {"Authorization": "Bearer x"}, {"model": "m", "q": "中文"}, 2.0
    )
    assert status == 200 and body["choices"][0]["message"]["content"] == "亲，默认发中通快递哦~"
    assert seen == {
        "url": BASE + "/chat/completions",
        "method": "POST",
        "auth": "Bearer x",
        "body": {"model": "m", "q": "中文"},
        "timeout": 2.0,
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(b'{"error": {"type": "rate_limit"}}', {"error": {"type": "rate_limit"}}), (b"<html>", {})],
)
def test_urllib_transport_http_error_returns_status_and_body(monkeypatch, raw, expected):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many", {}, io.BytesIO(raw))

    monkeypatch.setattr(openai_compat, "_urlopen", fake_urlopen)
    assert urllib_transport(BASE, {}, {}, 1.0) == (429, expected)


def test_urllib_transport_timeout_and_non_json(monkeypatch, api_key):
    def timeout_urlopen(request, timeout):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(openai_compat, "_urlopen", timeout_urlopen)
    with pytest.raises(TimeoutError):
        urllib_transport(BASE, {}, {}, 1.0)
    with pytest.raises(LLMTimeout):
        OpenAICompatibleClient(base_url=BASE, model="m").complete(req())

    monkeypatch.setattr(
        openai_compat, "_urlopen", lambda request, timeout: _FakeHTTPResponse(b"not json")
    )
    with pytest.raises(ValueError):
        urllib_transport(BASE, {}, {}, 1.0)
    with pytest.raises(LLMError):
        OpenAICompatibleClient(base_url=BASE, model="m").complete(req())
