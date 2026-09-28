"""默认传输层不跟随重定向：Authorization（大模型 API key）不能被带到别的主机或明文 HTTP。

用本机 127.0.0.1 上的两个假服务：模型接口返回 302 指向"另一个主机"（另一个端口），
另一个服务记录收到的请求头。不访问外部网络。
"""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from rodit.core.errors import LLMError
from rodit.core.ports import LLMMessage, LLMRequest
from rodit.llm import OpenAICompatibleClient, urllib_transport

KEY = "sk-synthetic-redirect-test-0001"  # 合成密钥，只用于测试


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
def servers(monkeypatch) -> Iterator[tuple[_Server, _Server, list[dict[str, Any]]]]:
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    stolen: list[dict[str, Any]] = []

    class Thief(http.server.BaseHTTPRequestHandler):
        def _record(self) -> None:
            stolen.append({"path": self.path, "headers": dict(self.headers)})
            body = json.dumps(
                {"choices": [{"message": {"content": "被转走了"}, "finish_reason": "stop"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _record  # noqa: N815

        def log_message(self, *args: Any) -> None:  # 不往测试输出里打日志
            pass

    thief = _Server(Thief)

    class Redirector(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            # 路径里的 3 位数段就是要返回的状态码（/v1/307/chat/completions），默认 302
            codes = [seg for seg in self.path.split("/") if seg.isdigit() and len(seg) == 3]
            self.send_response(int(codes[0]) if codes else 302)
            self.send_header("Location", f"{thief.base}/steal")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    api = _Server(Redirector)
    try:
        yield api, thief, stolen
    finally:
        api.close()
        thief.close()


def _req() -> LLMRequest:
    return LLMRequest(purpose="generate", messages=[LLMMessage(role="user", content="发什么快递")])


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_is_refused_and_key_is_not_forwarded(servers, monkeypatch, status):
    api, _thief, stolen = servers
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)
    client = OpenAICompatibleClient(base_url=f"{api.base}/v1/{status}", model="m", timeout_s=5)
    with pytest.raises(LLMError) as ei:
        client.complete(_req())
    assert stolen == [], "Authorization 被带到了重定向目标"
    assert KEY not in str(ei.value)
    assert str(status) in str(ei.value)


def test_transport_returns_redirect_status_without_following(servers):
    api, _thief, stolen = servers
    status, body = urllib_transport(
        f"{api.base}/v1/chat/completions", {"Authorization": f"Bearer {KEY}"}, {"q": 1}, 5.0
    )
    assert status == 302 and body == {}
    assert stolen == []
