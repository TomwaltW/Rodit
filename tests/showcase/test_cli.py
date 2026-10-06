"""python -m rodit.showcase：横幅（合成数据、离线替身、仅演示）、监听地址、--llm env 的限制。"""

from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llm_stub import StubLLM

from rodit.showcase.__main__ import DEFAULT_HOST, DEFAULT_PORT, EXIT_USAGE, is_loopback, main
from rodit.testing.fakes import FakeLLMClient


class _Run:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict]] = []

    def __call__(self, app, **kwargs):
        self.calls.append((app, kwargs))


def _no_llm():
    raise AssertionError("没有 --llm env 时不能读 RODIT_LLM_* 环境变量")


def _run(argv, *, llm_factory=_no_llm):
    out = io.StringIO()
    run = _Run()
    code = main(argv, out, run=run, llm_factory=llm_factory)
    return code, out.getvalue(), run


def test_default_banner_and_loopback_host(monkeypatch):
    # 环境里即使配了大模型，默认也不读（离线替身）
    monkeypatch.setenv("RODIT_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("RODIT_LLM_MODEL", "should-not-be-used")
    code, text, run = _run([])
    assert code == 0
    assert "【仅演示】" in text and "合成数据" in text and "离线替身" in text
    assert "不会动任何真实订单" in text
    assert "没有任何真实图片" in text
    assert f"http://127.0.0.1:{DEFAULT_PORT}/" in text
    assert "注意" not in text
    [(app, kwargs)] = run.calls
    assert isinstance(app, FastAPI)
    assert kwargs == {"host": DEFAULT_HOST, "port": DEFAULT_PORT}
    assert app.state.hub.offline is True
    assert app.state.hub.built() == []  # 平台单元用到时才组装


def test_platform_and_port_options():
    code, text, run = _run(["--platform", "douyin", "--port", "8200"])
    assert code == 0 and "http://127.0.0.1:8200/" in text
    app, kwargs = run.calls[0]
    assert kwargs["port"] == 8200
    assert app.state.hub.default_platform.value == "douyin"


def test_lan_host_prints_a_warning():
    code, text, run = _run(["--host", "0.0.0.0"])
    assert code == 0
    assert "注意" in text and "0.0.0.0" in text and "没有登录" in text
    assert run.calls[0][1]["host"] == "0.0.0.0"


def test_llm_env_refuses_a_non_loopback_host():
    called: list[bool] = []

    def factory():
        called.append(True)
        return FakeLLMClient(handler=lambda r: "{}")

    for host in ("0.0.0.0", "192.168.1.20", "::", "showcase.example"):
        code, text, run = _run(["--llm", "env", "--host", host], llm_factory=factory)
        assert code == EXIT_USAGE and "拒绝启动" in text
        assert run.calls == []
    assert called == []  # 拒绝之前不读环境变量


def test_llm_env_on_loopback_uses_the_real_client():
    client = FakeLLMClient(handler=lambda r: "{}", model="domestic-model")
    code, text, run = _run(["--llm", "env"], llm_factory=lambda: client)
    assert code == 0 and "真实大模型" in text and "domestic-model" in text
    app = run.calls[0][0]
    assert app.state.hub.offline is False and app.state.hub.llm is client


def test_llm_env_misconfigured_exits_2():
    def factory():
        raise RuntimeError("环境变量 RODIT_LLM_BASE_URL 和 RODIT_LLM_MODEL 必须设置")

    code, text, run = _run(["--llm", "env", "--host", "localhost"], llm_factory=factory)
    assert code == EXIT_USAGE and "RODIT_LLM_BASE_URL" in text and run.calls == []


def test_env_factory_reads_the_environment(monkeypatch):
    """真的 --llm env：用 rodit.eval.llm_env 的检查（没配密钥时报错，不打印密钥）；配好了先对
    本机的模型替身（tests/showcase/llm_stub.py）做启动检查，过了才起服务。"""
    with StubLLM(model="demo-model") as stub:
        monkeypatch.setenv("RODIT_LLM_BASE_URL", stub.base_url)
        monkeypatch.setenv("RODIT_LLM_MODEL", "demo-model")
        monkeypatch.delenv("RODIT_LLM_API_KEY", raising=False)
        out = io.StringIO()
        run = _Run()
        code = main(["--llm", "env"], out, run=run)
        assert code == EXIT_USAGE and "RODIT_LLM_API_KEY" in out.getvalue()
        assert stub.count() == 0
        monkeypatch.setenv("RODIT_LLM_API_KEY", stub.api_key)
        out = io.StringIO()
        code = main(["--llm", "env"], out, run=run)
        assert code == 0 and "demo-model" in out.getvalue()
        assert stub.api_key not in out.getvalue()
        # 启动检查就是评测的能力检查：普通调用、JSON 模式、logprobs（路由投票）各一次
        assert [r.purpose for r in stub.requests] == ["preflight", "probe_json", "probe_logprobs"]
        assert "启动检查通过——大模型能力检查（demo-model @ http://127.0.0.1:" in out.getvalue()
        assert run.calls  # 检查过了才起服务


@pytest.mark.parametrize("argv", [["--port", "70000"], ["--port", "-1"]])
def test_bad_port(argv):
    code, text, run = _run(argv)
    assert code == EXIT_USAGE and run.calls == []


@pytest.mark.parametrize("argv", [["--platform", "jd"], ["--llm", "cloud"], ["--port", "x"]])
def test_bad_arguments_exit_2(argv, capsys):
    code, _, run = _run(argv)
    assert code == 2 and run.calls == []
    assert "usage" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("localhost", True),
        ("LOCALHOST", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", False),
        ("::", False),
        ("10.0.0.5", False),
        ("example.com", False),
    ],
)
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected


@pytest.mark.parametrize(
    ("host", "url"),
    [
        ("127.0.0.1", "http://127.0.0.1:8100/"),
        ("localhost", "http://localhost:8100/"),
        ("127.0.0.2", "http://127.0.0.2:8100/"),
        ("::1", "http://[::1]:8100/"),
        ("0.0.0.0", "http://127.0.0.1:8100/"),
        ("::", "http://[::1]:8100/"),  # 监听 :: 只收 IPv6（asyncio 设了 IPV6_V6ONLY）
        ("192.168.1.20", "http://192.168.1.20:8100/"),
        ("fd00::20", "http://[fd00::20]:8100/"),
    ],
)
def test_printed_url_matches_the_host(host: str, url: str):
    code, text, run = _run(["--host", host, "--port", "8100"])
    assert code == 0 and f"打开：{url}" in text
    assert run.calls[0][1]["host"] == host


def test_app_only_accepts_hosts_for_its_bind_address():
    """监听地址交给应用：只监听本机时，Host 是别的域名（DNS 重绑定）的请求被拒；局域网演示不限。"""
    _, _, run = _run([])
    app = run.calls[0][0]
    assert TestClient(app, base_url="http://rebind.example:8100").get("/").status_code == 400
    assert TestClient(app, base_url="http://localhost:8100").get("/").status_code == 200
    _, _, run = _run(["--host", "0.0.0.0"])
    lan = run.calls[0][0]
    assert TestClient(lan, base_url="http://192.168.1.20:8100").get("/").status_code == 200


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RODIT_LLM_TIMEOUT_S", "abc"),
        ("RODIT_LLM_TIMEOUT_S", "-5"),
        ("RODIT_LLM_BASE_URL", "llm.example/v1"),  # 没有 http:// 或 https://
    ],
)
def test_llm_env_with_malformed_settings_exits_2(monkeypatch, name: str, value: str):
    """RODIT_LLM_* 格式不对（超时不是正数、地址没有协议）：拒绝启动、退出码 2，不抛异常、
    不打印密钥。"""
    monkeypatch.setenv("RODIT_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("RODIT_LLM_MODEL", "demo-model")
    monkeypatch.setenv("RODIT_LLM_API_KEY", "synthetic-test-key")
    monkeypatch.setenv(name, value)
    out = io.StringIO()
    run = _Run()
    code = main(["--llm", "env"], out, run=run)
    assert code == EXIT_USAGE and run.calls == []
    text = out.getvalue()
    assert "拒绝启动" in text and "RODIT_LLM_" in text
    assert "synthetic-test-key" not in text
