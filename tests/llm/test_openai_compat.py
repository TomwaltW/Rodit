"""OpenAICompatibleClient：请求构造、响应解析、错误映射、密钥不外泄（只用假传输层，不访问网络）。"""

from __future__ import annotations

import http.client
import io
import json
import math
import urllib.error
import urllib.request
from typing import Any

import pytest

from rodit.core.errors import LLMError, LLMTimeout
from rodit.core.ports import LLMClient, LLMMessage, LLMRequest
from rodit.llm import OpenAICompatibleClient, urllib_transport

KEY = "sk-synthetic-test-key-not-real-0001"  # 合成密钥，只用于测试
BASE = "https://llm.example.invalid/compatible-mode/v1"


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


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_api_key(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("RODIT_LLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("RODIT_LLM_API_KEY", value)
    t = FakeTransport()
    with pytest.raises(LLMError, match="RODIT_LLM_API_KEY"):
        client(t).complete(req())
    assert t.calls == []


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
    for text in (str(ei.value), repr(ei.value), repr(c), repr(vars(c))):
        assert KEY not in text


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

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
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

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert urllib_transport(BASE, {}, {}, 1.0) == (429, expected)


def test_urllib_transport_timeout_and_non_json(monkeypatch, api_key):
    def timeout_urlopen(request, timeout):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", timeout_urlopen)
    with pytest.raises(TimeoutError):
        urllib_transport(BASE, {}, {}, 1.0)
    with pytest.raises(LLMTimeout):
        OpenAICompatibleClient(base_url=BASE, model="m").complete(req())

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout: _FakeHTTPResponse(b"not json")
    )
    with pytest.raises(ValueError):
        urllib_transport(BASE, {}, {}, 1.0)
    with pytest.raises(LLMError):
        OpenAICompatibleClient(base_url=BASE, model="m").complete(req())
