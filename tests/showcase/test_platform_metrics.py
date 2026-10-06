"""演示站数据看板里的平台客服指标接入前后对比（方案 §1.3、§12.4、§15 Phase 2 商家报表）：
用商家报表同一个服务（rodit.app.platform_metrics.PlatformMetricsService）——演示单元组装时
系统从平台指标接口（合成数据）同步进单元仓储，看板按演示设定的接入日出报表，每个数都能按
rodit.feedback.platform_metrics 的口径复算。"""

from __future__ import annotations

from datetime import timedelta

import pytest
from showcase_support import NOW, Showcase

from rodit.app.platform_metrics import PLATFORM_METRICS_SYNCED
from rodit.core.clock import CST
from rodit.core.enums import Platform
from rodit.core.platform_metrics import metrics_for
from rodit.feedback.platform_metrics import build_report, resolve_go_live
from rodit.showcase.app import API_PREFIX
from rodit.showcase.dashboard import platform_report
from rodit.showcase.state import minute_floor
from rodit.testing.fixtures import DEMO_GO_LIVE_DAYS_AGO

API = API_PREFIX
TODAY = minute_floor(NOW).astimezone(CST).date()
GO_LIVE = TODAY - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)


@pytest.mark.parametrize("platform", list(Platform))
def test_demo_build_syncs_the_platform_metrics_as_system(showcase: Showcase, platform):
    demo = showcase.hub.get(platform)
    sync = demo.metrics_sync
    assert sync is not None
    assert sync.since == TODAY - timedelta(days=56) and sync.until == TODAY - timedelta(days=1)
    assert sync.days == 56 and sync.written == 56 * len(metrics_for(platform))
    assert sync.metrics == metrics_for(platform)
    rows = demo.unit.repos.platform_metrics.list(demo.shop_id)
    assert len(rows) == sync.written and all(r.source.value == "platform" for r in rows)
    [event] = demo.unit.repos.audit.list_events(demo.shop_id, action=PLATFORM_METRICS_SYNCED)
    assert event.actor == "system" and event.detail["written"] == sync.written


@pytest.mark.parametrize("platform", list(Platform))
def test_dashboard_uses_the_merchant_report(showcase: Showcase, platform):
    pm = showcase.dashboard(platform)["platform_metrics"]
    assert pm["go_live"] == GO_LIVE.isoformat() and pm["window_days"] == 28
    assert pm["before"] == [
        (GO_LIVE - timedelta(days=28)).isoformat(),
        (GO_LIVE - timedelta(days=1)).isoformat(),
    ]
    assert pm["after"] == [GO_LIVE.isoformat(), (TODAY - timedelta(days=1)).isoformat()]
    assert pm["verdict"] == "improved" and pm["verdict_label"] == "较接入前改善"
    assert pm["reliable"] is True and pm["notes"] == []
    assert [m["metric"] for m in pm["metrics"]] == [m.value for m in metrics_for(platform)]
    for m in pm["metrics"]:
        assert m["improved"] is True and m["direction_label"] == "变好" and m["reliable"]
        assert m["days_before"] == 28 and m["days_after"] == 28  # 今天的数平台还没出
        assert round(m["after"] - m["before"], 4) == pytest.approx(m["delta"], abs=1e-4)
        assert (m["delta"] > 0) == m["higher_is_better"]
        weekly = m["weekly"]
        assert [w["phase"] for w in weekly] == ["before"] * 4 + ["after"] * 4
        assert [w["index"] for w in weekly] == [-4, -3, -2, -1, 0, 1, 2, 3]
        assert all(w["days"] == 7 and w["mean"] is not None for w in weekly)
        assert weekly[4]["start"] == GO_LIVE.isoformat()
    assert pm["synced"]["written"] == 56 * len(metrics_for(platform))


def test_numbers_match_the_feedback_report_function(showcase: Showcase):
    """看板上的数就是 build_report 对同步进来的数据算出来的（没有另一套算法）。"""
    demo = showcase.hub.get(Platform.DOUYIN)
    pm = showcase.dashboard("douyin")["platform_metrics"]
    expected = build_report(
        shop_id=demo.shop_id,
        platform=Platform.DOUYIN,
        samples=demo.unit.repos.platform_metrics.list(demo.shop_id),
        go_live=resolve_go_live(GO_LIVE),
        today=TODAY,
    )
    assert expected == platform_report(demo)
    assert [round(m.before.mean, 4) for m in expected.metrics] == [
        m["before"] for m in pm["metrics"]
    ]
    assert [round(m.after.mean, 4) for m in expected.metrics] == [m["after"] for m in pm["metrics"]]
    response = pm["metrics"][0]
    assert response["label"] == "飞鸽平均响应时长" and response["unit"] == "seconds"
    assert response["before"] > 45 and response["after"] < 28


def test_overview_target_uses_the_report_verdict(showcase: Showcase):
    data = showcase.get(f"{API}/overview", platform="taobao").json()
    target = next(t for t in data["targets"] if t["key"] == "platform_metric")
    assert target["ok"] is True
    assert target["value"].startswith("旺旺 3 分钟人工响应率：78") and "→ 91" in target["value"]
    assert target["note"] == f"较接入前改善 · 接入日 {GO_LIVE.isoformat()}，前后各 28 天"


def test_reset_resyncs(showcase: Showcase):
    showcase.dashboard("pdd")
    showcase.post(f"{API}/reset", {"platform": "pdd"})
    demo = showcase.hub.get(Platform.PDD)
    events = demo.unit.repos.audit.list_events(demo.shop_id, action=PLATFORM_METRICS_SYNCED)
    assert len(events) == 1  # 新单元重新同步一次
    assert showcase.dashboard("pdd")["platform_metrics"]["verdict"] == "improved"
