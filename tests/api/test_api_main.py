"""python -m rodit.api：本地演示（合成数据、随机开发令牌、只监听本机）。"""

from __future__ import annotations

import io

from api_support import TestClient

from rodit.api.__main__ import (
    DEMO_HOST,
    DEMO_PORT,
    DEMO_PRINCIPALS,
    build_demo_app,
    demo_page_url,
    main,
)
from rodit.testing.fixtures import SHOP_A, SHOP_B


def test_demo_app_serves_requests_with_generated_tokens():
    app, tokens = build_demo_app()
    assert sorted(tokens.values(), key=repr) == sorted(DEMO_PRINCIPALS, key=repr)
    by_role = {(p.shop_id, p.role): t for t, p in tokens.items()}
    client = TestClient(app)
    agent = {"Authorization": f"Bearer {by_role[(SHOP_A, 'agent')]}"}
    r = client.post(
        f"/v1/shops/{SHOP_A}/standalone/ask",
        headers=agent,
        json={"session_id": "web_1", "question": "能开发票吗"},
    )
    assert r.status_code == 200 and r.json()["suggestions"]
    other = {"Authorization": f"Bearer {by_role[(SHOP_B, 'agent')]}"}
    r = client.post(
        f"/v1/shops/{SHOP_A}/standalone/ask",
        headers=other,
        json={"session_id": "web_1", "question": "能开发票吗"},
    )
    assert r.status_code == 403


def test_tokens_are_random_per_start():
    _, first = build_demo_app()
    _, second = build_demo_app()
    assert not set(first) & set(second)
    assert all(len(t) >= 32 for t in first)


def test_main_prints_demo_banner_and_binds_localhost():
    calls = []

    def fake_run(app, **kwargs):
        calls.append((app, kwargs))

    out = io.StringIO()
    assert main(out, run=fake_run) == 0
    [(app, kwargs)] = calls
    assert kwargs == {"host": "127.0.0.1", "port": 8000}
    assert (DEMO_HOST, DEMO_PORT) == ("127.0.0.1", 8000)
    text = out.getvalue()
    assert "仅本地演示" in text
    assert text.count("Bearer demo-") == len(DEMO_PRINCIPALS)
    token = next(line.split("Bearer ")[1] for line in text.splitlines() if " agent " in line)
    r = TestClient(app).get(
        f"/v1/shops/{SHOP_A}/reports/qa", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 403 and r.json()["error"] == "forbidden_role"


def test_main_prints_the_web_page_url_and_the_page_is_served():
    """模式 C 网页工具（§8.1）：启动信息里有页面地址，页面在演示应用里能打开。"""
    calls = []
    out = io.StringIO()
    main(out, run=lambda app, **kwargs: calls.append(app))
    assert demo_page_url() == "http://127.0.0.1:8000/assist"
    assert f"网页工具（模式 C）：{demo_page_url()}" in out.getvalue()
    [app] = calls
    r = TestClient(app).get("/assist")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
