"""Jev 的 HTTP 传输层（rodit.eval.jev_transport；方案 §6.3 离线评测、§6.5 Jev 格式）。

不联网：urlopen 用替身（注入 opener），重定向用本机 127.0.0.1 上的两个假服务。
检查：地址、密钥、超时的环境变量；请求的地址、方法、请求头、请求体；HTTP 非 2xx、超时、网络错误、
响应不是 JSON 都变成 DecisionError 的子类；密钥不出现在 repr、异常消息、日志里；不跟随重定向；
响应里的用量取出来累计，不交给 JevProvider。
"""

from __future__ import annotations

import http.client
import http.server
import io
import json
import logging
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from rodit.core.errors import ComplianceError, DecisionError
from rodit.core.questions import Q_INTENT, ROUTER_MODEL, ROUTER_QUESTIONS
from rodit.decision import JevConditions, JevProvider
from rodit.eval import jev_transport
from rodit.eval.jev_transport import (
    API_KEY_ENV,
    BASE_URL_ENV,
    DEFAULT_TIMEOUT_S,
    JEV_PATH,
    MODEL_ENV,
    TIMEOUT_ENV,
    JevConfigError,
    JevHttpTransport,
    JevTimeout,
    JevTransportError,
    transport_from_env,
)
from rodit.testing.fakes import router_answers

KEY = "jev-synthetic-test-key-DO-NOT-LEAK-0001"  # 合成的假密钥，只用来检查不外泄
BASE = "https://jev.example.test"
PAYLOAD = {
    "model": "jev-1.13.0",
    "state": {
        "synthetic": True,
        "platform": "taobao",
        "messages": [{"role": "buyer", "text": "165 110斤穿什么码"}],
    },
    "questions": {"intent": {"type": "choice", "instructions": "问什么", "criteria": {"a": None}}},
}


def _jev_body(model: str = "jev-1.13.0", **extra: Any) -> dict[str, Any]:
    answers = router_answers("size_advice", 0.9)
    return {
        "model": model,
        "answers": {k: a.model_dump(mode="json") for k, a in answers.items()},
        **extra,
    }


class _Resp:
    def __init__(self, body: bytes, status: int = 200, read_error: Exception | None = None):
        self.status = status
        self._body = body
        self._read_error = read_error

    def read(self) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _Opener:
    """urlopen 替身：记下请求和超时，按设定返回响应或抛异常。"""

    def __init__(self, result: Any = None) -> None:
        self.calls: list[tuple[urllib.request.Request, float]] = []
        self.result = result if result is not None else _Resp(json.dumps(_jev_body()).encode())

    def __call__(self, request: urllib.request.Request, timeout: float) -> Any:
        self.calls.append((request, timeout))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


@pytest.fixture
def key(monkeypatch) -> str:
    monkeypatch.setenv(API_KEY_ENV, KEY)
    return KEY


def _transport(opener: _Opener, **kw: Any) -> JevHttpTransport:
    return JevHttpTransport(base_url=BASE, opener=opener, **kw)


def _http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(f"{BASE}{JEV_PATH}", code, "err", {}, io.BytesIO(body))  # type: ignore[arg-type]


# ================================================================ 环境变量


def test_from_env_reads_base_url_timeout_and_model():
    env = {
        BASE_URL_ENV: f" {BASE}/ ",
        API_KEY_ENV: KEY,
        TIMEOUT_ENV: "3.5",
        MODEL_ENV: " jev-laya-multilingual ",
    }
    transport, model = transport_from_env(env)
    assert transport.base_url == BASE
    assert transport.endpoint == f"{BASE}{JEV_PATH}"
    assert transport.timeout_s == 3.5
    assert transport.api_key_env == API_KEY_ENV
    assert model == "jev-laya-multilingual"
    default, no_model = transport_from_env({BASE_URL_ENV: BASE, API_KEY_ENV: KEY})
    assert default.timeout_s == DEFAULT_TIMEOUT_S == 10.0
    assert no_model is None


@pytest.mark.parametrize(
    ("env", "names"),
    [
        ({}, [BASE_URL_ENV, API_KEY_ENV]),
        ({BASE_URL_ENV: BASE}, [API_KEY_ENV]),
        ({API_KEY_ENV: KEY, BASE_URL_ENV: "  "}, [BASE_URL_ENV]),
        ({BASE_URL_ENV: BASE, API_KEY_ENV: "  \n"}, [API_KEY_ENV]),
    ],
)
def test_from_env_names_missing_variables_without_values(env, names):
    with pytest.raises(JevConfigError) as ei:
        JevHttpTransport.from_env(env)
    message = str(ei.value)
    for name in names:
        assert name in message
    assert KEY not in message and BASE not in message


@pytest.mark.parametrize(
    "url",
    [
        "http://jev.example.test",  # 密钥不走明文
        "ftp://jev.example.test",
        "https://",
        "https://user:pass@jev.example.test",
        "https://jev.example.test?x=1",
        "https://jev.example.test#frag",
        f"https://jev.example.test{JEV_PATH}",
        "https://jev.example.test:notaport",
    ],
)
def test_base_url_must_be_plain_https(url):
    with pytest.raises(JevConfigError) as ei:
        JevHttpTransport(base_url=url)
    assert "pass" not in str(ei.value)


@pytest.mark.parametrize("timeout", ["0", "-1", "abc", "nan", "inf"])
def test_timeout_must_be_a_positive_number(timeout):
    with pytest.raises(JevConfigError, match=TIMEOUT_ENV):
        JevHttpTransport.from_env({BASE_URL_ENV: BASE, API_KEY_ENV: KEY, TIMEOUT_ENV: timeout})


def test_base_url_may_have_a_path_prefix():
    t = JevHttpTransport(base_url="https://jev.example.test:8443/proxy/")
    assert t.endpoint == f"https://jev.example.test:8443/proxy{JEV_PATH}"


# ================================================================ 请求


def test_posts_jev_json_with_bearer_key_read_at_call_time(monkeypatch, key):
    opener = _Opener()
    transport = _transport(opener, timeout_s=4)
    body = transport(PAYLOAD)
    assert body == _jev_body()
    (request, timeout) = opener.calls[0]
    assert request.full_url == f"{BASE}{JEV_PATH}"
    assert request.get_method() == "POST"
    assert timeout == 4
    headers = {k.lower(): v for k, v in request.header_items()}
    assert headers["authorization"] == f"Bearer {KEY}"
    assert headers["content-type"] == "application/json"
    sent = request.data.decode("utf-8")
    assert json.loads(sent) == PAYLOAD
    assert "穿什么码" in sent  # 中文不转义（ensure_ascii=False）
    # 密钥每次调用时读：换了就用新的
    monkeypatch.setenv(API_KEY_ENV, "jev-synthetic-rotated-key")
    transport(PAYLOAD)
    assert (
        dict(opener.calls[1][0].header_items())["Authorization"]
        == "Bearer jev-synthetic-rotated-key"
    )


def test_missing_key_at_call_time_fails_without_sending(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    opener = _Opener()
    with pytest.raises(JevTransportError, match=API_KEY_ENV):
        _transport(opener)(PAYLOAD)
    assert opener.calls == []


def test_repr_has_no_key(key):
    transport = _transport(_Opener())
    assert KEY not in repr(transport)
    assert API_KEY_ENV in repr(transport)


# ================================================================ 失败


@pytest.mark.parametrize(
    ("code", "words"),
    [
        (401, API_KEY_ENV),
        (403, API_KEY_ENV),
        (404, "拒绝请求"),
        (422, "拒绝请求"),
        (429, "限流"),
        (500, "服务错误"),
        (503, "服务错误"),
        (302, "重定向"),
    ],
)
def test_http_errors_become_transport_errors_without_body(key, code, words):
    echo = f'{{"error": "bad key {KEY}", "echo": "165 110斤穿什么码"}}'.encode()
    opener = _Opener(_http_error(code, echo))
    with pytest.raises(JevTransportError) as ei:
        _transport(opener)(PAYLOAD)
    err = ei.value
    assert isinstance(err, DecisionError)
    assert err.status == code
    assert str(code) in str(err) and words in str(err)
    assert KEY not in str(err) and "穿什么码" not in str(err)
    assert err.__cause__ is None and err.__suppress_context__


def test_non_2xx_status_without_exception_is_an_error(key):
    with pytest.raises(JevTransportError) as ei:
        _transport(_Opener(_Resp(b"{}", status=500)))(PAYLOAD)
    assert ei.value.status == 500


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timed out"),  # socket.timeout 是它的别名
        urllib.error.URLError(TimeoutError("timed out")),  # 连接阶段超时
    ],
)
def test_timeouts_become_jev_timeout(key, error):
    with pytest.raises(JevTimeout) as ei:
        _transport(_Opener(error), timeout_s=2.5)(PAYLOAD)
    assert isinstance(ei.value, JevTransportError)
    assert "2.5" in str(ei.value)


def test_timeout_while_reading_the_body(key):
    opener = _Opener(_Resp(b"", read_error=TimeoutError("read timed out")))
    with pytest.raises(JevTimeout):
        _transport(opener)(PAYLOAD)


@pytest.mark.parametrize(
    ("error", "name"),
    [
        (urllib.error.URLError(ConnectionRefusedError(f"refused {KEY}")), "ConnectionRefusedError"),
        (urllib.error.URLError("unknown host"), "str"),
        (ConnectionResetError(f"reset {KEY}"), "ConnectionResetError"),
        (http.client.RemoteDisconnected("closed"), "RemoteDisconnected"),
        (http.client.IncompleteRead(b"par"), "IncompleteRead"),
    ],
)
def test_network_errors_become_transport_errors(key, error, name):
    with pytest.raises(JevTransportError) as ei:
        _transport(_Opener(error))(PAYLOAD)
    assert not isinstance(ei.value, JevTimeout)
    assert name in str(ei.value)
    assert KEY not in str(ei.value)


@pytest.mark.parametrize("raw", [b"not json", b"[1, 2]", b'"text"', b"\xff\xfe\x00", b""])
def test_bad_json_becomes_transport_error(key, raw):
    with pytest.raises(JevTransportError, match="JSON"):
        _transport(_Opener(_Resp(raw)))(PAYLOAD)


def test_key_never_reaches_logs_or_exceptions(key, caplog):
    caplog.set_level(logging.DEBUG)
    errors = [
        _http_error(401, KEY.encode()),
        urllib.error.URLError(OSError(KEY)),
        TimeoutError(KEY),
        OSError(KEY),
    ]
    for error in errors:
        with pytest.raises(JevTransportError) as ei:
            _transport(_Opener(error))(PAYLOAD)
        assert KEY not in str(ei.value) and KEY not in repr(ei.value)
    _transport(_Opener())(PAYLOAD)
    assert all(KEY not in r.getMessage() for r in caplog.records)
    assert KEY not in caplog.text


# ================================================================ 用量


def test_usage_is_taken_out_of_the_response_and_accumulated(key):
    transport = _transport(_Opener())
    assert transport.usage_totals() == (0, 0, 0)
    assert transport(PAYLOAD) == _jev_body()  # 没有用量：原样返回，不计
    assert transport.usage_totals() == (0, 0, 0)
    transport._opener = _Opener(
        _Resp(json.dumps(_jev_body(usage={"input_tokens": 120, "output_tokens": 0})).encode())
    )
    body = transport(PAYLOAD)
    assert "usage" not in body and body == _jev_body()
    transport._opener = _Opener(
        _Resp(json.dumps(_jev_body(usage={"prompt_tokens": 30, "completion_tokens": 5})).encode())
    )
    transport(PAYLOAD)
    assert transport.usage_totals() == (150, 5, 2)
    for junk in ("lots", {"input_tokens": "many"}, {"input_tokens": True}, None):
        transport._opener = _Opener(_Resp(json.dumps(_jev_body(usage=junk)).encode()))
        assert "usage" not in transport(PAYLOAD)
    assert transport.usage_totals() == (150, 5, 2)


@pytest.mark.parametrize(
    ("usage", "totals"),
    [
        ({"input_tokens": float("inf"), "output_tokens": 3}, (0, 3, 1)),
        ({"input_tokens": float("nan")}, (0, 0, 0)),
        ({"prompt_tokens": float("-inf"), "completion_tokens": float("nan")}, (0, 0, 0)),
    ],
)
def test_non_finite_usage_is_ignored_and_the_answer_is_kept(key, usage, totals):
    """json.loads 收 Infinity / NaN：用量不是有限数时当作没报告，不能让一个好的回答变成异常。"""
    raw = json.dumps(_jev_body(usage=usage)).encode()
    assert b"Infinity" in raw or b"NaN" in raw
    transport = _transport(_Opener(_Resp(raw)))
    assert transport(PAYLOAD) == _jev_body()
    assert transport.usage_totals() == totals
    provider = JevProvider(transport, JevConditions(), purpose="offline_eval")
    from rodit.core.decision import DecisionRequest

    state = {"synthetic": True, "messages": [{"role": "buyer", "text": "在吗"}]}
    response = provider.decide(
        DecisionRequest(model=ROUTER_MODEL, state=state, questions=ROUTER_QUESTIONS)
    )
    assert response.choice(Q_INTENT).choice == "size_advice"


def test_other_unknown_fields_are_left_for_jev_provider_to_reject(key):
    """格式变了（多了不认识的字段）照样交给 JevProvider，由它按"判断层不可用"报错。"""
    opener = _Opener(_Resp(json.dumps(_jev_body(debug={"trace": 1})).encode()))
    provider = JevProvider(_transport(opener), JevConditions(), purpose="offline_eval")
    from rodit.core.decision import DecisionRequest

    request = DecisionRequest(
        model=ROUTER_MODEL,
        state={"synthetic": True, "messages": [{"role": "buyer", "text": "在吗"}]},
        questions=ROUTER_QUESTIONS,
    )
    with pytest.raises(DecisionError, match="格式不对"):
        provider.decide(request)


def test_works_as_jev_provider_transport_and_keeps_the_guard(key):
    from rodit.core.decision import DecisionRequest

    opener = _Opener()
    provider = JevProvider(_transport(opener), JevConditions(), purpose="offline_eval")
    state = {"synthetic": True, "messages": [{"role": "buyer", "text": "在吗"}]}
    response = provider.decide(
        DecisionRequest(model=ROUTER_MODEL, state=state, questions=ROUTER_QUESTIONS)
    )
    assert response.choice(Q_INTENT).choice == "size_advice"
    sent = json.loads(opener.calls[0][0].data)
    assert sent["model"] == "jev-1.13.0" and sent["state"] == state
    # 护栏在传输层之前：个人信息、没有合成标记都不会发出去
    for bad in (
        {"synthetic": True, "messages": [{"role": "buyer", "text": "我手机13812345678"}]},
        {"messages": [{"role": "buyer", "text": "在吗"}]},
    ):
        with pytest.raises(ComplianceError):
            provider.decide(
                DecisionRequest(model=ROUTER_MODEL, state=bad, questions=ROUTER_QUESTIONS)
            )
    assert len(opener.calls) == 1
    # 传输层的超时由 JevProvider 包成 DecisionError，原异常留在 __cause__ 里
    timeout = JevProvider(
        _transport(_Opener(TimeoutError())), JevConditions(), purpose="offline_eval"
    )
    with pytest.raises(DecisionError) as ei:
        timeout.decide(DecisionRequest(model=ROUTER_MODEL, state=state, questions=ROUTER_QUESTIONS))
    assert isinstance(ei.value.__cause__, JevTimeout)


# ================================================================ 不跟随重定向（本机假服务）


class _Server:
    def __init__(self, handler: type[http.server.BaseHTTPRequestHandler]) -> None:
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def redirect_servers(monkeypatch) -> Iterator[tuple[_Server, list[dict[str, Any]]]]:
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    stolen: list[dict[str, Any]] = []

    class Thief(http.server.BaseHTTPRequestHandler):
        def _record(self) -> None:
            stolen.append({"path": self.path, "headers": dict(self.headers)})
            body = json.dumps(_jev_body()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _record  # noqa: N815

        def log_message(self, *args: Any) -> None:
            pass

    thief = _Server(Thief)

    class Redirector(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            codes = [seg for seg in self.path.split("/") if seg.isdigit() and len(seg) == 3]
            self.send_response(int(codes[0]) if codes else 302)
            self.send_header("Location", f"{thief.base}/steal")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    api = _Server(Redirector)
    try:
        yield api, stolen
    finally:
        api.close()
        thief.close()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_is_refused_and_key_is_not_forwarded(redirect_servers, key, status):
    api, stolen = redirect_servers

    def to_local(request: urllib.request.Request, timeout: float) -> Any:
        # 地址必须是 https（假服务是 http）：把同一个请求原样转给本机假服务，走真正的 _urlopen
        local = urllib.request.Request(
            f"{api.base}/{status}{JEV_PATH}",
            data=request.data,
            headers=dict(request.header_items()),
            method="POST",
        )
        return jev_transport._urlopen(local, timeout)

    transport = JevHttpTransport(base_url=BASE, opener=to_local, timeout_s=5)
    with pytest.raises(JevTransportError) as ei:
        transport(PAYLOAD)
    assert stolen == [], "Authorization 被带到了重定向目标"
    assert ei.value.status == status and "重定向" in str(ei.value)
    assert KEY not in str(ei.value)
