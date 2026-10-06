"""--llm env 的启动检查（showcase.preflight）：连通性 + JSON 模式探测，每类故障一句中文诊断，
只显示接口主机，不显示密钥。故障用注入的传输层或本机的模型替身（llm_stub.py）造，不联网；
"网络策略拦截"用本机一个对 CONNECT 回 403 的假代理造（和沙箱出口代理拒绝时一样的错误）。"""

from __future__ import annotations

import io
import socket
import ssl
import threading
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from llm_stub import StubLLM

from rodit.core.errors import LLMError, LLMTimeout
from rodit.llm import OpenAICompatibleClient
from rodit.showcase.__main__ import EXIT_USAGE, main
from rodit.showcase.preflight import (
    JSON_REQUIRED,
    PreflightFailed,
    classify,
    kind_of,
    run_preflight,
    short_label,
)

KEY = "sk-synthetic-preflight-key-0000"


def _client(base_url: str = "https://llm.example.test/v1", **kwargs) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(base_url=base_url, model="demo-model", **kwargs)


def _raising(exc: BaseException):
    def transport(url, headers, body, timeout_s):
        raise exc

    return transport


def _status(status: int, body: dict | None = None):
    def transport(url, headers, body_, timeout_s):
        return status, body or {}

    return transport


def _fail(client, *, judge_llm: bool = False) -> PreflightFailed:
    with pytest.raises(PreflightFailed) as info:
        run_preflight(client, judge_llm=judge_llm, vote_samples=3)
    return info.value


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)


# ================================================================ 分类（注入的传输层）


@pytest.mark.parametrize(
    ("exc", "kind", "phrases"),
    [
        (
            urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden")),
            "egress",
            ["网络策略没放行 https://llm.example.test", "允许的域名", "新开一个会话"],
        ),
        (
            urllib.error.URLError(
                ssl.SSLCertVerificationError(
                    1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
                )
            ),
            "tls",
            ["TLS 证书校验失败", "不要关闭证书校验"],
        ),
        (
            urllib.error.URLError(socket.gaierror(-2, "Name or service not known")),
            "dns",
            ["解析不了 https://llm.example.test"],
        ),
        (
            urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
            "refused",
            ["拒绝连接"],
        ),
        (TimeoutError("timed out"), "timeout", ["10 秒内没有响应", "RODIT_LLM_TIMEOUT_S"]),
        (ValueError("Expecting value: line 1 column 1"), "format", ["不是 OpenAI 兼容格式"]),
    ],
)
def test_network_failures_are_classified(exc, kind, phrases):
    failed = _fail(_client(transport=_raising(exc)))
    assert failed.kind == kind and failed.stage == "连通性检查"
    for phrase in phrases:
        assert phrase in str(failed)
    assert KEY not in str(failed)


@pytest.mark.parametrize(
    ("status", "kind", "phrase"),
    [
        (401, "auth", "拒绝了密钥（HTTP 401）"),
        (403, "forbidden", "拒绝访问（HTTP 403）"),
        (404, "not_found", "找不到接口或模型（HTTP 404）"),
        (429, "rate_limit", "并发 3 个请求"),
        (500, "server", "服务端出错（HTTP 500）"),
        (503, "server", "HTTP 503"),
        (302, "redirect", "重定向（HTTP 302）"),
        (400, "bad_param", "JSON 模式"),
    ],
)
def test_http_failures_are_classified(status, kind, phrase):
    body = {"error": {"type": "x", "code": "y", "message": f"bad key {KEY}"}}
    failed = _fail(_client(transport=_status(status, body)))
    assert failed.kind == kind and phrase in str(failed)
    assert KEY not in str(failed) and "bad key" not in str(failed)


def test_non_openai_response_is_a_format_problem():
    failed = _fail(_client(transport=_status(200, {"data": "hello"})))
    assert failed.kind == "format" and "OpenAI 兼容" in str(failed)


def test_missing_key_is_reported_by_variable_name(monkeypatch):
    monkeypatch.delenv("RODIT_LLM_API_KEY")
    failed = _fail(_client(transport=_status(200, {})))
    assert failed.kind == "no_key" and "RODIT_LLM_API_KEY" in str(failed)


def test_short_labels():
    assert short_label(LLMTimeout("x"), timeout_s=1.0) == "超时（限时 1 秒）"
    assert short_label(LLMError("大模型服务错误（HTTP 502）")) == "服务端错误（HTTP 502）"
    assert short_label(LLMError("大模型接口限流（HTTP 429）")) == "限流或额度用完（HTTP 429）"
    assert short_label(LLMError("大模型接口调用失败：URLError: Tunnel connection failed: 403"))
    assert kind_of(LLMError("大模型输出被内容安全策略拦截")) == "filtered"
    assert classify(LLMError("大模型接口拒绝请求（HTTP 418）"), "h").startswith("调用 h 失败")


# ================================================================ 真实 HTTP（本机的模型替身）


def _stub_client(stub: StubLLM, **kwargs) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(base_url=stub.base_url, model=stub.model, **kwargs)


@pytest.fixture
def stub_key(stub: StubLLM, monkeypatch: pytest.MonkeyPatch) -> StubLLM:
    monkeypatch.setenv("RODIT_LLM_API_KEY", stub.api_key)
    return stub


def test_passing_preflight_reports_timings(stub_key: StubLLM):
    result = run_preflight(_stub_client(stub_key), judge_llm=True)
    assert set(result.steps_ms) == {"connect", "json"}
    assert result.host == stub_key.base_url.removesuffix("/v1") and result.model == "stub-chat"
    assert result.logprobs is False  # 替身默认不返回 logprobs
    [probe] = stub_key.bodies("probe")
    assert probe["response_format"] == {"type": "json_object"} and probe["logprobs"] is True
    offline = run_preflight(_stub_client(stub_key), judge_llm=False)
    assert offline.logprobs is None
    assert "logprobs" not in stub_key.bodies("probe")[-1]


def test_logprobs_are_detected(monkeypatch):
    with StubLLM(logprobs="tokens") as stub:
        monkeypatch.setenv("RODIT_LLM_API_KEY", stub.api_key)
        assert run_preflight(_stub_client(stub), judge_llm=True).logprobs is True


def test_wrong_key_401_never_echoes_the_key(stub: StubLLM, monkeypatch):
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)
    failed = _fail(_stub_client(stub))
    assert failed.kind == "auth" and "HTTP 401" in str(failed)
    assert KEY not in str(failed) and KEY[-8:] not in str(failed)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("403", "forbidden"), ("404", "not_found"), ("429", "rate_limit"), ("500", "server")],
)
def test_stub_http_failures(stub_key: StubLLM, kind: str, expected: str):
    stub_key.fail("preflight", kind)
    assert _fail(_stub_client(stub_key)).kind == expected


def test_wrong_base_path_is_a_404(stub_key: StubLLM):
    client = OpenAICompatibleClient(
        base_url=stub_key.base_url.removesuffix("/v1"), model=stub_key.model
    )
    failed = _fail(client)
    assert failed.kind == "not_found" and "RODIT_LLM_BASE_URL" in str(failed)


def test_json_mode_rejected(stub_key: StubLLM):
    stub_key.fail("probe", "400_response_format")
    failed = _fail(_stub_client(stub_key), judge_llm=True)
    assert failed.stage == "JSON 模式探测" and failed.kind == "bad_param"
    assert "JSON 模式" in str(failed) and "logprobs" not in str(failed)


def test_logprobs_rejected(stub_key: StubLLM):
    stub_key.fail("probe", "400_logprobs")
    failed = _fail(_stub_client(stub_key), judge_llm=True)
    assert "logprobs" in str(failed) and "--judge offline" in str(failed)
    # 去掉 logprobs 又探了一次，分清了是哪个参数
    assert [bool(b.get("logprobs")) for b in stub_key.bodies("probe")] == [True, False]
    # 判断层不用大模型时不请求 logprobs，探测照常通过
    assert run_preflight(_stub_client(stub_key), judge_llm=False).logprobs is None


def test_non_json_output_fails_the_probe(stub_key: StubLLM):
    stub_key.fail("probe", "not_json")
    failed = _fail(_stub_client(stub_key))
    assert failed.kind == "json" and JSON_REQUIRED in str(failed)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("content_filter", "filtered"), ("malformed", "format"), ("no_choices", "format")],
)
def test_bad_responses(stub_key: StubLLM, kind: str, expected: str):
    stub_key.fail("preflight", kind)
    assert _fail(_stub_client(stub_key)).kind == expected


def test_timeout(stub_key: StubLLM):
    stub_key.delay("preflight", 1.0)
    failed = _fail(_stub_client(stub_key, timeout_s=0.2))
    assert failed.kind == "timeout" and "0.2 秒" in str(failed)


def test_connection_refused():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    failed = _fail(_client(f"http://127.0.0.1:{port}/v1"))
    assert failed.kind == "refused"


# ================================================================ 网络策略拦截（假的出口代理）


class _DenyConnect(BaseHTTPRequestHandler):
    def do_CONNECT(self) -> None:  # noqa: N802
        self.send_response(403, "Forbidden")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format, *args) -> None:  # noqa: A002
        pass


@pytest.fixture
def denying_proxy(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DenyConnect)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    proxy = f"http://127.0.0.1:{server.server_address[1]}"
    for name in ("https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, proxy)
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    try:
        yield proxy
    finally:
        server.shutdown()
        server.server_close()


def test_egress_denied_by_the_network_policy(denying_proxy: str):
    """沙箱的出口代理对没放行的域名回 403：诊断说清楚要在环境的网络设置里放行这个主机。"""
    failed = _fail(_client("https://llm.blocked.example.test/v1"))
    assert failed.kind == "egress"
    assert "网络策略没放行 https://llm.blocked.example.test" in str(failed)
    assert "/v1" not in str(failed) and KEY not in str(failed)


# ================================================================ 命令行


def test_cli_exits_2_with_the_diagnosis(stub_env: StubLLM):
    stub_env.fail("probe", "400_logprobs")
    out = io.StringIO()
    runs: list = []
    code = main(["--llm", "env"], out, run=lambda app, **kw: runs.append(app))
    text = out.getvalue()
    assert code == EXIT_USAGE and runs == []
    assert "拒绝启动：大模型JSON 模式探测没通过" in text and "logprobs" in text
    assert stub_env.api_key not in text


def test_cli_with_judge_offline_does_not_ask_for_logprobs(stub_env: StubLLM):
    stub_env.fail("probe", "400_logprobs")
    out = io.StringIO()
    runs: list = []
    code = main(["--llm", "env", "--judge", "offline"], out, run=lambda app, **kw: runs.append(app))
    assert code == 0 and runs
    assert "判断层：离线规则" in out.getvalue()
