"""演示站只认监听地址对应的 Host（防 DNS 重绑定），写操作拒绝别的网站发来的跨站请求。

DNS 重绑定：恶意网页把自己的域名解析到 127.0.0.1，浏览器就把它当同源，能读写只监听本机的演示站
（没有登录；--llm env 时还会花本机配置的大模型额度）。Host 头里是恶意域名，按监听地址放行就挡住了。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from showcase_support import NOW

from rodit.showcase.app import allowed_hosts, create_showcase_app
from rodit.showcase.state import ShowcaseHub
from rodit.testing.fakes import FakeClock


def _client(host: str | None = None, base_url: str = "http://127.0.0.1") -> TestClient:
    hub = ShowcaseHub(clock=FakeClock(NOW), history_seed=None)
    app = create_showcase_app(hub) if host is None else create_showcase_app(hub, host=host)
    return TestClient(app, base_url=base_url)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", ["127.0.0.1", "localhost", "[::1]"]),
        ("localhost", ["127.0.0.1", "localhost", "[::1]"]),
        ("::1", ["127.0.0.1", "localhost", "[::1]"]),
        ("[::1]", ["127.0.0.1", "localhost", "[::1]"]),
        ("127.0.0.2", ["127.0.0.1", "localhost", "[::1]", "127.0.0.2"]),
        ("0.0.0.0", ["*"]),
        ("::", ["*"]),
        ("192.168.1.20", ["192.168.1.20"]),
        ("fd00::20", ["[fd00::20]"]),
        ("Showcase.Example", ["showcase.example"]),
    ],
)
def test_allowed_hosts_follow_the_bind_host(host: str, expected: list[str]):
    assert allowed_hosts(host) == expected


@pytest.mark.parametrize(
    "base_url",
    ["http://127.0.0.1:8100", "http://localhost:8100", "http://[::1]:8100", "http://127.0.0.1"],
)
def test_loopback_bind_accepts_loopback_hosts(base_url: str):
    client = _client(base_url=base_url)
    assert client.get("/").status_code == 200
    r = client.get("/showcase/api/scenarios", params={"platform": "pdd"})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "base_url", ["http://rebind.example:8100", "http://192.168.1.20:8100", "http://testserver"]
)
def test_loopback_bind_rejects_other_hosts(base_url: str):
    """默认（只监听本机）：Host 是别的域名（DNS 重绑定）一律 400，接口和页面都不处理。"""
    client = _client(base_url=base_url)
    assert client.get("/").status_code == 400
    r = client.post("/showcase/api/reset", json={"platform": "taobao"})
    assert r.status_code == 400
    assert client.app.state.hub.built() == []  # 没有组装、没有重置


def test_lan_bind_accepts_any_host():
    """--host 0.0.0.0（局域网演示）：访问的地址各不相同，不限制 Host（--llm env 照旧拒绝启动）。"""
    for base_url in ("http://192.168.1.20:8100", "http://demo-laptop.local:8100"):
        assert _client("0.0.0.0", base_url).get("/").status_code == 200
    assert _client("::", "http://[fd00::20]:8100").get("/").status_code == 200


def test_explicit_host_accepts_only_that_host():
    assert _client("192.168.1.20", "http://192.168.1.20:8100").get("/").status_code == 200
    assert _client("192.168.1.20", "http://rebind.example:8100").get("/").status_code == 400
    assert _client("::1", "http://[::1]:8100").get("/").status_code == 200


def test_cross_site_writes_are_refused():
    """别的网站的页面往演示站发 POST（浏览器带 Origin）：Origin 和 Host 对不上就 403，不处理。"""
    client = _client(base_url="http://127.0.0.1:8100")
    for origin in ("http://evil.example", "http://127.0.0.1:9999", "null", "http://localhost:8100"):
        r = client.post(
            "/showcase/api/reset", json={"platform": "taobao"}, headers={"Origin": origin}
        )
        assert r.status_code == 403, origin
        assert r.json()["error"] == "forbidden"
    assert client.app.state.hub.built() == []
    # 演示站自己的页面（同源）、不带 Origin 的命令行工具照常
    r = client.post(
        "/showcase/api/mask", json={"text": "你好"}, headers={"Origin": "http://127.0.0.1:8100"}
    )
    assert r.status_code == 200
    assert client.post("/showcase/api/mask", json={"text": "你好"}).status_code == 200
    # 只查写操作：GET 带别的 Origin 照常（浏览器的跨站 GET 读不到响应）
    r = client.get("/", headers={"Origin": "http://evil.example"})
    assert r.status_code == 200


def test_same_origin_on_the_default_port():
    client = _client("0.0.0.0", "http://demo-laptop.local")
    r = client.post(
        "/showcase/api/mask", json={"text": "你好"}, headers={"Origin": "http://demo-laptop.local"}
    )
    assert r.status_code == 200
    ipv6 = _client("::1", "http://[::1]:8100")
    r = ipv6.post(
        "/showcase/api/mask", json={"text": "你好"}, headers={"Origin": "http://[::1]:8100"}
    )
    assert r.status_code == 200
