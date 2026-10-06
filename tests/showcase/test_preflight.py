"""--llm env 的启动检查（showcase.preflight）：就是评测的能力检查
（rodit.eval.llm_env.probe_capabilities：普通调用、JSON 模式，路由投票时再加 logprobs），
每类故障一句中文诊断，参数被拒绝时写明要改哪个开关；只显示接口主机，不显示密钥。
故障用注入的传输层或本机的模型替身（llm_stub.py）造，不联网；"网络策略拦截"用本机一个对
CONNECT 回 403 的假代理造（和沙箱出口代理拒绝时一样的错误）。"""

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

JSON_TOGGLE = "RODIT_LLM_JSON_MODE=off"
LOGPROBS_TOGGLE = "RODIT_LLM_LOGPROBS=off"
EXTRA_BODY = "RODIT_LLM_EXTRA_BODY"


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
    assert failed.kind == kind and failed.stage == "大模型连通性检查"
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
        (400, "bad_param", EXTRA_BODY),  # 普通调用不带 response_format：不是 JSON 模式的事
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


def _choice(message: dict) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", **message}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8},
    }


@pytest.mark.parametrize(
    "message",
    [
        {"content": "<think>\n好的，用户让我"},  # 思考放在正文里，被 max_tokens 截在 <think> 里
        {"content": None, "reasoning_content": "好的，用户让我只回复"},  # 思考放在单独的字段里
        {"content": "", "reasoning_content": "好的，用户让我只回复"},
    ],
)
def test_thinking_model_cut_off_is_diagnosed_as_thinking(message):
    """默认开思考模式的模型：普通调用（max_tokens 8）只来得及输出思考。诊断要指向关闭思考模式，
    不是"调用失败 / 网络不通"或"不是 OpenAI 兼容格式"。"""
    failed = _fail(_client(transport=_status(200, _choice(message))), judge_llm=True)
    assert failed.kind == "thinking" and failed.stage == "大模型连通性检查"
    text = str(failed)
    assert "思考模式" in text and EXTRA_BODY in text
    assert "调用失败" not in text and "OpenAI 兼容格式" not in text
    hints = failed.report.hints
    assert hints and all("稍后再试" not in h for h in hints)
    assert any(EXTRA_BODY in h for h in hints)


@pytest.mark.parametrize("style", ["truncated", "reasoning"])
def test_thinking_stub_cut_off_by_the_token_limit(stub: StubLLM, monkeypatch, style: str):
    """本机替身：思考放在正文里、请求的 token 上限小时 </think> 来不及输出（truncated）；
    思考放在 reasoning_content 里、正文为空（reasoning）。"""
    monkeypatch.setenv("RODIT_LLM_API_KEY", stub.api_key)
    stub.thinking, stub.thinking_style = True, style
    failed = _fail(_stub_client(stub), judge_llm=True)
    assert failed.kind == "thinking" and "思考模式" in str(failed)
    assert short_label(LLMError(failed.report.check("plain").error or "")) == "思考模式没关"
    # 关掉思考模式就过了
    ok = run_preflight(
        _stub_client(stub, extra_body={"thinking": {"type": "disabled"}}), judge_llm=True
    )
    assert ok.report.reachable


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
    """路由投票：普通调用、JSON 模式、logprobs 三个请求，各一次，不另发探测。"""
    result = run_preflight(_stub_client(stub_key), judge_llm=True)
    assert set(result.steps_ms) == {"plain", "json_mode", "logprobs"}
    assert result.host == stub_key.base_url.removesuffix("/v1") and result.model == "stub-chat"
    assert result.logprobs is False  # 替身默认不返回 logprobs
    assert [r.purpose for r in stub_key.requests] == ["preflight", "probe_json", "probe_logprobs"]
    [probe] = stub_key.bodies("probe_json")
    assert probe["response_format"] == {"type": "json_object"} and "logprobs" not in probe
    [lp] = stub_key.bodies("probe_logprobs")
    assert lp["logprobs"] is True and "response_format" not in lp
    assert "logprobs：通过" in "\n".join(result.lines())
    assert "没返回选项概率" in "\n".join(result.lines())
    view = result.view()
    assert [c["status"] for c in view["checks"]] == ["ok", "ok", "ok"]
    assert view["checks"][2]["output_ok"] is False and LOGPROBS_TOGGLE in "".join(view["hints"])


def test_judge_offline_does_not_probe_logprobs(stub_key: StubLLM):
    """路由不用大模型投票时请求不带 logprobs：不检查（少一个请求），记为没检查。"""
    offline = run_preflight(_stub_client(stub_key), judge_llm=False)
    assert offline.logprobs is None and set(offline.steps_ms) == {"plain", "json_mode"}
    assert stub_key.count("probe_logprobs") == 0 and stub_key.count() == 2
    assert offline.report.check("logprobs").status == "skipped"


def test_logprobs_are_detected(monkeypatch):
    with StubLLM(logprobs="tokens") as stub:
        monkeypatch.setenv("RODIT_LLM_API_KEY", stub.api_key)
        assert run_preflight(_stub_client(stub), judge_llm=True).logprobs is True


def test_toggled_off_capabilities_are_not_requested(stub_key: StubLLM):
    """JSON 模式、logprobs 都关了：JSON 模式检查不带 response_format，logprobs 不检查。"""
    result = run_preflight(_stub_client(stub_key, json_mode=False, logprobs=False), judge_llm=True)
    assert stub_key.count() == 2 and result.logprobs is None
    assert all("response_format" not in b and "logprobs" not in b for b in stub_key.bodies())
    text = "\n".join(result.lines())
    assert '"json_mode":false' in text and '"logprobs":false' in text
    assert "RODIT_LLM_JSON_MODE=off：不发 response_format" in text and "logprobs：已关闭" in text


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
    stub_key.fail("probe_json", "400_response_format")
    failed = _fail(_stub_client(stub_key), judge_llm=True)
    assert failed.stage == "大模型 JSON 模式检查" and failed.kind == "bad_param"
    text = str(failed)
    assert "不接受 JSON 模式参数 response_format（HTTP 400）" in text
    assert JSON_TOGGLE in text and EXTRA_BODY in text and LOGPROBS_TOGGLE not in text
    assert failed.report is not None and failed.report.check("json_mode").status == "rejected"


def test_logprobs_rejected(stub_key: StubLLM):
    stub_key.fail("probe_logprobs", "400_logprobs")
    failed = _fail(_stub_client(stub_key), judge_llm=True)
    assert failed.stage == "大模型 logprobs 检查" and failed.kind == "bad_param"
    assert LOGPROBS_TOGGLE in str(failed) and "--judge offline" in str(failed)
    # 两种参数分开检查，不用去掉 logprobs 再探一次
    assert stub_key.count("probe_json") == 1 and stub_key.count("probe_logprobs") == 1
    # 判断层不用大模型时不请求 logprobs，检查照常通过
    assert run_preflight(_stub_client(stub_key), judge_llm=False).logprobs is None


def test_every_blocking_problem_is_reported_at_once(stub_key: StubLLM):
    """JSON 模式和 logprobs 都被拒绝：一次说全，不用改一个、重启、再看下一个。"""
    stub_key.fail("probe_json", "400_response_format")
    stub_key.fail("probe_logprobs", "400_logprobs")
    failed = _fail(_stub_client(stub_key), judge_llm=True)
    assert failed.stage == "大模型 JSON 模式检查"
    assert JSON_TOGGLE in str(failed) and "另外大模型 logprobs 检查也没通过" in str(failed)
    assert LOGPROBS_TOGGLE in str(failed)


def test_logprobs_trouble_that_may_pass_only_warns(stub_key: StubLLM):
    """logprobs 检查限流 / 超时：不拦启动（换个时间可能就过了），提示留在报告里。"""
    stub_key.fail("probe_logprobs", "429")
    result = run_preflight(_stub_client(stub_key), judge_llm=True)
    check = result.report.check("logprobs")
    assert check.status == "failed" and check.http_status == 429 and result.logprobs is None
    assert any("限流" in hint for hint in result.report.hints)


def test_json_check_that_times_out_blocks(stub_key: StubLLM):
    """JSON 模式检查没有结果（超时）：每条消息都要用 JSON，拒绝启动，按超时诊断。"""
    stub_key.timeout_sleep_s = 1.0
    stub_key.fail("probe_json", "timeout")
    failed = _fail(_stub_client(stub_key, timeout_s=0.3))
    assert failed.stage == "大模型 JSON 模式检查" and failed.kind == "timeout"
    assert "0.3 秒内没有响应" in str(failed)


def test_non_json_output_fails_the_probe(stub_key: StubLLM):
    stub_key.fail("probe_json", "not_json")
    failed = _fail(_stub_client(stub_key))
    assert failed.kind == "json" and JSON_REQUIRED in str(failed)
    assert EXTRA_BODY in str(failed) and JSON_TOGGLE not in str(failed)
    # JSON 模式关着时，提示把它打开
    stub_key.reset()
    stub_key.fail("probe_json", "not_json")
    failed = _fail(_stub_client(stub_key, json_mode=False))
    assert "去掉 RODIT_LLM_JSON_MODE=off" in str(failed)


def test_rejected_plain_call_names_the_extra_body_keys(stub_key: StubLLM):
    stub_key.fail("preflight", "400")
    failed = _fail(_stub_client(stub_key, extra_body={"thinking": {"type": "disabled"}}))
    assert failed.stage == "大模型连通性检查" and failed.kind == "bad_param"
    assert "RODIT_LLM_EXTRA_BODY（或 llm.extra_body）里的参数（thinking）" in str(failed)


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
    # 要放行的是域名（不带协议、端口）；能力检查的提示也不能叫人"稍后再试"（重试没有用）
    assert "把域名 llm.blocked.example.test 加进允许的域名" in str(failed)
    hints = failed.report.hints
    assert hints and all("稍后再试" not in h for h in hints)
    assert any("llm.blocked.example.test" in h and "重试没有用" in h for h in hints)


@pytest.mark.parametrize(
    ("exc", "phrase"),
    [
        (urllib.error.URLError(socket.gaierror(-2, "Name or service not known")), "解析不了域名"),
        (urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")), "连接被拒绝"),
        (
            urllib.error.URLError(
                ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] verify failed")
            ),
            "TLS 证书校验失败",
        ),
    ],
)
def test_network_hints_match_the_failure(exc, phrase):
    """能力检查的提示按故障类型给：域名解析、连接被拒、证书，不是笼统的"稍后再试"。"""
    failed = _fail(_client(transport=_raising(exc)))
    hint = failed.report.check("plain").hint or ""
    assert phrase in hint and "稍后再试" not in hint


# ================================================================ 接口写在配置文件里：诊断说 llm.*


def _config_main(tmp_path, stub: StubLLM, llm_yaml: str) -> tuple[int, str]:
    config = tmp_path / "rodit.yaml"
    config.write_text(
        f"llm:\n  base_url: {stub.base_url}\n  model: {stub.model}\n{llm_yaml}", encoding="utf-8"
    )
    out = io.StringIO()
    code = main(["--llm", "env", "--config", str(config)], out, run=lambda app, **kw: None)
    return code, out.getvalue()


def test_config_endpoint_timeout_names_llm_timeout_s(stub_env: StubLLM, tmp_path, monkeypatch):
    """接口写在配置里（llm.base_url / llm.model）时 RODIT_LLM_TIMEOUT_S 不起作用：诊断和提示都要说
    llm.timeout_s。"""
    monkeypatch.setenv("RODIT_LLM_TIMEOUT_S", "30")
    stub_env.delay("preflight", 1.0)
    code, text = _config_main(tmp_path, stub_env, "  timeout_s: 0.3\n")
    assert code == EXIT_USAGE and "0.3 秒内没有响应" in text
    assert "llm.timeout_s" in text and "RODIT_LLM_TIMEOUT_S" not in text


def test_config_endpoint_404_names_llm_base_url(stub_env: StubLLM, tmp_path):
    stub_env.fail("preflight", "404")
    code, text = _config_main(tmp_path, stub_env, "")
    assert code == EXIT_USAGE and "HTTP 404" in text
    assert "llm.base_url" in text and "llm.model" in text
    assert "RODIT_LLM_BASE_URL" not in text and "RODIT_LLM_MODEL" not in text


def test_config_key_variable_is_named_in_the_401_diagnosis(
    stub_env: StubLLM, tmp_path, monkeypatch
):
    """llm.api_key_env 换了变量名：401 的诊断说的是这个变量，不是 RODIT_LLM_API_KEY。"""
    monkeypatch.setenv("SHOP_DEMO_LLM_KEY", "synthetic-wrong-key-123")
    code, text = _config_main(tmp_path, stub_env, "  api_key_env: SHOP_DEMO_LLM_KEY\n")
    assert code == EXIT_USAGE and "HTTP 401" in text
    assert "SHOP_DEMO_LLM_KEY" in text and "RODIT_LLM_API_KEY" not in text
    assert "synthetic-wrong-key-123" not in text


# ================================================================ 命令行


def test_cli_exits_2_with_the_diagnosis(stub_env: StubLLM):
    stub_env.fail("probe_logprobs", "400_logprobs")
    out = io.StringIO()
    runs: list = []
    code = main(["--llm", "env"], out, run=lambda app, **kw: runs.append(app))
    text = out.getvalue()
    assert code == EXIT_USAGE and runs == []
    # 先打能力检查的每一步（和评测打的一样），再说拒绝启动的原因和要改的开关
    assert "大模型能力检查（stub-chat @ http://127.0.0.1:" in text
    assert "  普通调用：通过" in text and "  logprobs：被拒绝" in text
    assert "拒绝启动：大模型 logprobs 检查没通过" in text and LOGPROBS_TOGGLE in text
    assert text.index("大模型能力检查") < text.index("拒绝启动")
    assert stub_env.api_key not in text


def test_cli_with_judge_offline_does_not_ask_for_logprobs(stub_env: StubLLM):
    stub_env.fail("probe_logprobs", "400_logprobs")
    out = io.StringIO()
    runs: list = []
    code = main(["--llm", "env", "--judge", "offline"], out, run=lambda app, **kw: runs.append(app))
    assert code == 0 and runs
    text = out.getvalue()
    assert "判断层：离线规则" in text and "logprobs：没检查" in text
    assert stub_env.count("probe_logprobs") == 0
