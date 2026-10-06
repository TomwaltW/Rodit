"""python -m rodit.api：本地演示（合成数据、随机开发令牌、只监听本机）。"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

from api_support import TestClient

from rodit.api.__main__ import (
    DEMO_HOST,
    DEMO_PORT,
    DEMO_PRINCIPALS,
    build_demo_app,
    demo_page_url,
    demo_shop_ids,
    main,
)
from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import MetricSource
from rodit.testing.fixtures import DEMO_GO_LIVE_DAYS_AGO, SHOP_A, SHOP_B


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


def test_main_prints_the_web_page_urls_and_the_pages_are_served():
    """网页工具：话术助手（模式 C，§8.1）、商家控制台、售后审核工作台（§7.3.2）的地址都打印出来，
    页面在演示应用里都能打开。"""
    calls = []
    out = io.StringIO()
    main(out, run=lambda app, **kwargs: calls.append(app))
    assert demo_page_url() == "http://127.0.0.1:8000/assist"
    assert demo_page_url("aftersale") == "http://127.0.0.1:8000/aftersale"
    lines = out.getvalue().splitlines()
    [app] = calls
    client = TestClient(app)
    for page, name in (
        ("assist", "话术助手"),
        ("console", "商家控制台"),
        ("aftersale", "售后审核工作台"),
    ):
        [line] = [x for x in lines if f"{demo_page_url(page)} " in x]
        assert name in line
        r = client.get(f"/{page}")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]


def test_demo_app_has_platform_metrics_synced_so_the_report_has_data():
    """启动前每家演示店铺都同步了合成的平台客服指标（走 PlatformMetricsService，审计留痕），
    控制台「报表」页一打开就有接入前后对比（§12.4）：接入日期未知时页面按 28 天前算，
    前后两段都有数。"""
    app, tokens = build_demo_app()
    assert demo_shop_ids() == [SHOP_A, SHOP_B]
    by_role = {(p.shop_id, p.role): t for t, p in tokens.items()}
    client = TestClient(app)
    merchant = {"Authorization": f"Bearer {by_role[(SHOP_A, 'merchant')]}"}
    r = client.get(f"/v1/shops/{SHOP_A}/platform-metrics", headers=merchant)
    assert r.status_code == 200, r.text
    samples = r.json()["samples"]
    assert samples and {s["source"] for s in samples} == {MetricSource.PLATFORM.value}
    # 页面在接入日期未知时按东八区今天往前 28 天算（同 console.js），这里照着传
    now = datetime.fromisoformat(client.get("/v1/me", headers=merchant).json()["server_time"])
    go_live = (now.astimezone(CST).date() - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)).isoformat()
    r = client.get(
        f"/v1/shops/{SHOP_A}/reports/platform-metrics",
        headers=merchant,
        params={"go_live": go_live},
    )
    assert r.status_code == 200, r.text
    report = r.json()
    assert report["go_live"] == go_live
    [row] = report["metrics"]
    assert row["before"]["days"] == row["after"]["days"] == DEMO_GO_LIVE_DAYS_AGO
    assert row["after"]["mean"] > row["before"]["mean"]  # 合成数据：接入后 3 分钟响应率更高
    # 每家店铺各同步一次，操作人 system，没有买家数据
    service = app.state.rodit_service
    for shop_id in demo_shop_ids():
        events = service.unit.repos.audit.list_events(
            shop_id, action=AuditAction.PLATFORM_METRICS_SYNCED.value
        )
        assert [e.actor for e in events] == ["system"]
        assert events[0].detail["written"] == 56  # 默认同步 56 天：接入前、接入后各 28 天
        assert len(service.unit.repos.platform_metrics.list(shop_id)) == 56
