"""演示站的半自动和上线节奏（方案 §5.6 半自动、§13.4 影子 → 智能辅助 → 半自动 → 全自动、
§15 Phase 3 全自动先在非工作时间试行）。

- 半自动是复核人员经 rodit.app.rollout.set_semi_auto 开的（门槛没过时强制开启、写明原因，留审计），
  开了的类型里高置信度的简单问题，引擎给第一张卡片打 prefill；
- 模式 B 插件把它填进客服输入框（适配器的 input_prefill），模式 C 只标"可直接发送"；
- "上线节奏"的阶段、半自动门槛、客服工作时间都取生产代码的口径。
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from showcase_support import NOW, Showcase

from rodit.core.enums import Intent, Platform
from rodit.feedback.rollout import SEMI_AUTO_CHANGED
from rodit.showcase.app import API_PREFIX
from rodit.showcase.data import REVIEWER, SEMI_AUTO_INTENTS, SEMI_AUTO_NOTE
from rodit.showcase.rollout import AUTO_NOTES

API = API_PREFIX
B_PLATFORMS = [Platform.TAOBAO, Platform.DOUYIN]


def _cards(view: dict) -> list[dict]:
    return view["result"]["suggestions"]


def _rollout(showcase: Showcase, platform) -> dict:
    r = showcase.get(f"{API}/scenarios", platform=str(platform))
    assert r.status_code == 200, r.text
    return r.json()["rollout"]


# ================================================================ 开启（复核人员，留审计）


@pytest.mark.parametrize("platform", list(Platform))
def test_semi_auto_is_enabled_by_a_reviewer_through_the_rollout_service(
    showcase: Showcase, platform
):
    demo = showcase.hub.get(platform)
    assert demo.unit.shop_config(demo.shop_id).semi_auto_intents == list(SEMI_AUTO_INTENTS)
    events = demo.unit.repos.audit.list_events(demo.shop_id, action=SEMI_AUTO_CHANGED)
    assert [e.detail["intent"] for e in events] == [i.value for i in SEMI_AUTO_INTENTS]
    for event in events:
        assert event.actor == REVIEWER and event.detail["role"] == "reviewer"
        assert event.detail["enabled"] is True
        # 没有回放历史：没有推荐数据，门槛不可能过——强制开启，原因写进审计
        assert event.detail["forced"] is True and event.detail["note"] == SEMI_AUTO_NOTE
        assert event.detail["gate"]["passed"] is False
    assert all(change.forced for change in demo.semi_auto.values())


def test_with_history_the_gate_still_fails_and_is_forced(seeded: Showcase):
    """合成历史每类每周只有几组推荐（§13.4 要每周至少 10 组、原样采纳率 ≥ 90%）：如实强制开启。"""
    for platform in Platform:
        demo = seeded.hub.get(platform)
        assert all(change.forced for change in demo.semi_auto.values())
        rollout = _rollout(seeded, platform)
        rows = rollout["semi_auto"]["intents"]
        assert [r["intent"] for r in rows] == [i.value for i in SEMI_AUTO_INTENTS]
        assert all(r["forced"] and not r["gate_passed"] for r in rows)
        assert any(w["groups"] > 0 for r in rows for w in r["weeks"])
        assert all(w["groups"] < 10 for r in rows for w in r["weeks"])


# ================================================================ 卡片上的 prefill


@pytest.mark.parametrize("platform", B_PLATFORMS)
@pytest.mark.parametrize("scenario", ["ship_quick", "invoice"])
def test_mode_b_prefills_the_agent_input_box(showcase: Showcase, platform, scenario):
    """高置信度（规则路由 0.9 ≥ semi_auto_threshold）的发货、发票问题：第一张卡片 prefill，
    插件把它填进客服输入框；客服按发送仍是人工回复。"""
    view = showcase.chat(platform, scenario)
    u = view["trace"]["understand"]
    assert u["rule"] == 6 and u["confidence"] >= 0.9
    [card] = _cards(view)
    assert card["kind"] == "faq" and card["confidence"] == "high" and card["prefill"] is True
    result = view["result"]
    assert result["prefill_mode"] == "input"
    assert result["input_prefill"] == card["suggestion_id"]
    out = showcase.feedback(platform, card["suggestion_id"], "sent").json()
    assert out["status"] == "sent" and out["bubble"]["text"] == card["text"]


@pytest.mark.parametrize("scenario", ["ship_quick", "invoice"])
def test_mode_c_marks_the_card_instead_of_filling_an_input(showcase: Showcase, scenario):
    """拼多多模式 C 没有平台输入框：卡片标"可直接发送"，客服复制回多多客服。"""
    view = showcase.chat("pdd", scenario)
    [card] = _cards(view)
    assert card["prefill"] is True
    assert view["result"]["prefill_mode"] == "copy" and view["result"]["input_prefill"] is None


@pytest.mark.parametrize(
    ("platform", "scenario"),
    [
        (p, s)
        for p in Platform
        for s in ("size", "ship_time", "off_platform", "qa_block", "emotion", "multi", "price")
        if (p, s) != (Platform.PDD, "qa_block")
    ],
)
def test_other_scenarios_are_not_prefilled(showcase: Showcase, platform, scenario):
    """没开半自动的类型（尺码、催发货）、高风险（规则 4）、分条（规则 5）、站外话术（规则 1）、
    安抚（规则 3）：都不填输入框。"""
    view = showcase.chat(platform, scenario)
    assert not any(c["prefill"] for c in _cards(view))
    assert view["result"]["input_prefill"] is None


def test_pdd_rewrite_after_qa_block_is_marked_because_shipping_is_semi_auto(showcase: Showcase):
    """拼多多的质检拦截问法（"明天能到吗？"）路由到发货政策（规则 6）：第一稿被拦下，重写稿通过
    质检，引擎按最终那份草稿判断半自动（engine.semi_auto），所以标"可直接发送"；页面的说明写明了。"""
    view = showcase.chat("pdd", "qa_block")
    assert view["trace"]["understand"]["intent"] == "shipping_policy"
    assert view["trace"]["qa_summary"]["blocked"] == 1
    [card] = _cards(view)
    assert card["prefill"] is True and "24 小时" in card["text"]
    scenarios = showcase.get(f"{API}/scenarios", platform="pdd").json()["scenarios"]
    expect = next(s for s in scenarios if s["id"] == "qa_block")["expect"]
    assert "可直接发送" in expect and "客服核对" in expect


def test_alerts_block_prefill(showcase: Showcase):
    """有标红提醒（这里是图片摘要）时不填：提醒要先看（§5.6）。"""
    view = showcase.chat("taobao", "photo_size", "什么时候发货呀？", ["img_buyer_product_1"])
    assert view["result"]["alerts"]
    assert not any(c["prefill"] for c in _cards(view))
    assert view["result"]["input_prefill"] is None


def test_prefill_survives_in_the_saved_conversation(showcase: Showcase):
    showcase.chat("taobao", "ship_quick")
    scenarios = showcase.get(f"{API}/scenarios", platform="taobao").json()["scenarios"]
    latest = next(s for s in scenarios if s["id"] == "ship_quick")["latest"]
    assert latest["result"]["suggestions"][0]["prefill"] is True
    assert latest["result"]["prefill_mode"] == "input"


# ================================================================ 上线节奏


@pytest.mark.parametrize("platform", list(Platform))
def test_rollout_explainer_steps(showcase: Showcase, platform):
    rollout = _rollout(showcase, platform)
    assert rollout["stage"] == "assist" and rollout["stage_label"] == "智能辅助"
    steps = rollout["steps"]
    assert [s["key"] for s in steps] == ["shadow", "assist", "semi_auto", "auto"]
    assert [s["label"] for s in steps] == ["影子模式", "智能辅助", "半自动", "全自动"]
    assert [s["state"] for s in steps[:3]] == ["done", "current", "partial"]
    assert steps[3]["text"] == AUTO_NOTES[platform]
    semi = rollout["semi_auto"]
    assert semi["threshold"] == 0.9 and semi["min_as_is"] == 0.9
    assert semi["periods"] == 2 and semi["min_groups"] == 10
    assert [r["label"] for r in semi["intents"]] == ["发货与快递", "发票 / 赠品 / 保障"]
    auto = rollout["auto"]
    assert auto["service_hours"] == ["09:00-23:00"] and auto["timezone"].startswith("北京时间")
    assert auto["enabled"] is False and auto["active_now"] is False
    assert auto["schedule"] == "off_hours"


def test_only_taobao_can_ever_go_full_auto(showcase: Showcase):
    """§2.1：淘宝准入后可做模式 A；抖店飞鸽没有公开的消息收发接口；拼多多模式 C 没有发送接口。"""
    states = {p: _rollout(showcase, p)["steps"][3]["state"] for p in Platform}
    assert states == {
        Platform.TAOBAO: "next",
        Platform.DOUYIN: "unavailable",
        Platform.PDD: "unavailable",
    }
    assert _rollout(showcase, "taobao")["auto"]["available"] is True
    assert _rollout(showcase, "pdd")["auto"]["available"] is False


def test_service_hours_follow_the_real_clock(showcase: Showcase):
    """全自动先在非工作时间试行：现在（10:38）是客服工作时间，23:00 起是非工作时间。"""
    auto = _rollout(showcase, "taobao")["auto"]
    assert auto["in_service_hours"] is True and auto["next_switch_at"] == "09-28 23:00"
    showcase.clock.set(NOW.replace(hour=23, minute=30))
    auto = _rollout(showcase, "taobao")["auto"]
    assert auto["in_service_hours"] is False and auto["next_switch_at"] == "09-29 09:00"
    assert auto["active_now"] is False  # 全自动没开：非工作时间也不自动


def test_rollout_dates_come_from_the_replayed_history(seeded: Showcase):
    steps = _rollout(seeded, "douyin")["steps"]
    demo = seeded.hub.get(Platform.DOUYIN)
    first_day = (demo.now - timedelta(days=35)).strftime("%m-%d")
    go_live = (demo.now - timedelta(days=28)).strftime("%m-%d")
    assert f"本店 {first_day} 起的第 1 周" in steps[0]["text"]
    assert f"本店 {go_live} 起" in steps[1]["text"]


def test_semi_auto_intents_are_low_risk():
    """开半自动的类型不能是高风险类型（§6.6 规则 4），也要走规则 6 的推荐。"""
    from rodit.core.config import EngineConfig
    from rodit.feedback.rollout import semi_auto_blocked

    routing = EngineConfig().routing
    assert all(semi_auto_blocked(i, routing) is None for i in SEMI_AUTO_INTENTS)
    assert Intent.PRICE_PROMO not in SEMI_AUTO_INTENTS
