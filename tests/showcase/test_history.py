"""演示站回放的合成历史：数据看板一打开就有内容（§12.4），而且数字说得通——
采纳率在 65–75% 左右且逐周上升、第 1 周是影子模式（§13.4）、有质检拦截（§5.5）、
发出去的没有违规（§1.3）、平台客服指标接入后改善；同样的输入结果完全一样，组装在 5 秒内。"""

from __future__ import annotations

from datetime import timedelta

import pytest
from showcase_support import NOW, Showcase

from rodit.core.clock import CST
from rodit.core.enums import Intent, Platform, SuggestionStatus
from rodit.feedback import adoption_report
from rodit.showcase.seed import (
    ARRIVAL_QUESTIONS,
    HISTORY_DAYS,
    WEEKLY_VOLUME,
    AdoptionQuota,
    go_live_at,
    history_plan,
    history_start,
)
from rodit.showcase.state import ShowcaseHub, minute_floor
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import DEMO_GO_LIVE_DAYS_AGO

WORLD_NOW = minute_floor(NOW)


@pytest.mark.parametrize("platform", list(Platform))
def test_plan_is_deterministic_and_inside_the_window(platform):
    plan = history_plan(platform, WORLD_NOW, seed=7)
    assert plan == history_plan(platform, WORLD_NOW, seed=7)
    assert plan != history_plan(platform, WORLD_NOW, seed=8)
    assert [m.at for m in plan] == sorted(m.at for m in plan)
    start = history_start(WORLD_NOW)
    assert start.tzinfo is not None and start.hour == 0
    assert start.date() == (WORLD_NOW.astimezone(CST) - timedelta(days=HISTORY_DAYS)).date()
    assert all(start <= m.at < WORLD_NOW for m in plan)
    assert all(9 <= m.at.astimezone(CST).hour <= 22 for m in plan)  # 客服在线时间
    assert len(plan) <= sum(WEEKLY_VOLUME)
    arrival = [m for m in plan if m.text == ARRIVAL_QUESTIONS[platform][0]]
    assert len(arrival) >= 8  # 每周两条"明天能到吗"，让质检拦截有数据
    if platform == Platform.PDD:  # 模式 C 查不到订单：历史里不带订单卡片
        assert all(m.order_id is None for m in plan)
    else:
        assert any(m.order_id for m in plan)


def test_go_live_matches_the_platform_metric_split():
    go_live = go_live_at(WORLD_NOW)
    assert go_live.hour == 0
    assert go_live.date() == WORLD_NOW.date() - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)
    assert go_live - history_start(WORLD_NOW) == timedelta(days=7)  # 第 1 周是影子模式


def test_adoption_quota_tracks_the_probability():
    quota = AdoptionQuota()
    taken = [quota.take(0.7) for _ in range(100)]
    assert sum(taken) == 70
    assert sum(taken[:10]) == 7  # 小样本里也不大起大落


@pytest.mark.parametrize("platform", list(Platform))
def test_history_gives_a_believable_dashboard(seeded: Showcase, platform):
    d = seeded.dashboard(platform)
    k = d["kpis"]
    assert 0.62 <= k["adoption_rate"] <= 0.78
    assert k["as_is_rate"] < k["adoption_rate"]
    assert k["groups"] > 120 and k["messages"] > 200
    assert k["violations"] == 0
    assert k["e2e_p95_ms"] is not None and k["e2e_p95_ms"] <= k["e2e_target_ms"]
    assert k["pending_candidates"] > 0 and k["knowledge_gaps"] > 0
    weekly = d["weekly"]
    assert weekly[0]["shadow"] is True and weekly[0]["adoption_rate"] is None
    assert weekly[0]["shadow_records"] > 0
    rates = [w["adoption_rate"] for w in weekly[1:]]
    assert all(r is not None for r in rates)
    assert rates[-1] > rates[0]  # 越用越准（§5.8）
    assert all(abs(b - a) < 0.15 for a, b in zip(rates, rates[1:], strict=False))
    assert d["qa"]["messages_intercepted"] >= 4
    assert {h["rule"] for h in d["qa"]["hard_hits"]} >= {"absolute_term", "time_promise"}
    assert sum(r["count"] for r in d["reject_reasons"]) > 10
    assert len({i["intent"] for i in d["intents"]}) >= 12  # 售前售后各类问题都有
    assert d["loop"]["labels"] > 0 and d["loop"]["top_gaps"]
    metrics = d["platform_metrics"]
    assert metrics["metrics"] and all(m["improved"] for m in metrics["metrics"])
    assert metrics["verdict"] == "improved"
    assert metrics["go_live"] == go_live_at(WORLD_NOW).date().isoformat()


def test_history_records_match_the_summary(seeded: Showcase):
    demo = seeded.hub.get(Platform.TAOBAO)
    summary = demo.history
    records = demo.unit.repos.suggestions.list(demo.shop_id)
    statuses = [r.status for r in records]
    assert statuses.count(SuggestionStatus.SENT) == summary.actions["sent"]
    assert statuses.count(SuggestionStatus.EDITED_SENT) == summary.actions["edited"]
    assert statuses.count(SuggestionStatus.REJECTED) == summary.actions["rejected"]
    assert statuses.count(SuggestionStatus.SHADOW) > 0
    assert summary.shadow_messages > 0 and summary.messages > 200
    assert all(r.created_at < WORLD_NOW + timedelta(minutes=5) for r in records)
    report = adoption_report(records)
    assert report.shadow_excluded == statuses.count(SuggestionStatus.SHADOW)
    # 影子记录都在接入之前，接入之后没有影子记录
    go_live = go_live_at(WORLD_NOW)
    assert all(r.created_at < go_live for r in records if r.status == SuggestionStatus.SHADOW)
    assert all(r.status != SuggestionStatus.SHADOW for r in records if r.created_at >= go_live)
    # 演示店铺最后停在智能辅助（抖店的合成数据本来是影子模式）
    for platform in Platform:
        d = seeded.hub.get(platform)
        assert d.unit.shop_config(d.shop_id).shadow_mode is False
        assert d.unit.shop_config(d.shop_id).manager_ids == ["kf_manager"]


def test_history_does_not_degrade_or_touch_other_shops(seeded: Showcase):
    demo = seeded.hub.get(Platform.TAOBAO)
    results = demo.unit.repos.audit.list_events(demo.shop_id, action="engine.result")
    assert results and all(not e.detail["degraded"] for e in results)
    assert demo.unit.repos.suggestions.list("s_2002") == []  # 同单元的店铺 B 没有被写入


def test_history_is_reproducible():
    clock = FakeClock(NOW)
    a = ShowcaseHub(clock=clock).get(Platform.PDD)
    b = ShowcaseHub(clock=clock).get(Platform.PDD)
    assert a.history == b.history
    ra = [(r.intent, r.status, r.text) for r in a.unit.repos.suggestions.list(a.shop_id)]
    rb = [(r.intent, r.status, r.text) for r in b.unit.repos.suggestions.list(b.shop_id)]
    assert ra == rb


@pytest.mark.parametrize("platform", list(Platform))
def test_each_platform_builds_quickly(seeded: Showcase, platform):
    """第一次打开某个平台要回放历史：目标 5 秒内（实测约 1.5 秒）。这里留出并发跑测试时的余量，
    只挡数量级的退化。"""
    demo = seeded.hub.get(platform)
    assert demo.build_ms < 8000
    assert demo.history.messages > 200


def test_live_clock_moves_forward_only(showcase: Showcase):
    demo = showcase.hub.get(Platform.TAOBAO)
    assert demo.now == WORLD_NOW and demo.clock.now() == WORLD_NOW
    demo.tick()
    assert demo.clock.now() == NOW  # 拨到"真实时间"
    showcase.clock.set(NOW - timedelta(hours=1))  # 真实时间往回走（改了系统时间）也不倒退
    demo.tick()
    assert demo.clock.now() == NOW
    showcase.clock.set(NOW + timedelta(minutes=2))
    view = showcase.chat("taobao", "logistics")
    assert view["message"]["at"].startswith((NOW + timedelta(minutes=2)).isoformat()[:16])


def test_history_intents_cover_presale_and_aftersale(seeded: Showcase):
    demo = seeded.hub.get(Platform.DOUYIN)
    intents = {r.intent for r in demo.unit.repos.suggestions.list(demo.shop_id)}
    assert {Intent.SIZE_ADVICE, Intent.LOGISTICS_TRACK, Intent.RETURN_POLICY} <= intents
