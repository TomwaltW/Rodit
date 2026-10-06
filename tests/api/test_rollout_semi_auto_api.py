"""半自动开关与全自动生效时段的 HTTP 接口（方案 §5.6、§13.4、§15 Phase 3、§9.2 串店、§4.1 原则 3
操作人取自身份）。401 / 串店 / 角色的逐路由测试在 test_api_auth.py（路由清单
api_routes_rollout.py）。数据全部合成。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from api_support import Api, shop_snapshot

from rodit.core.clock import CST
from rodit.core.enums import Intent, SuggestionStatus
from rodit.core.models import SuggestionRecord
from rodit.feedback.rollout import SEMI_AUTO_CHANGED
from rodit.testing.fixtures import SHOP_A, SHOP_B

BASE = f"/v1/shops/{SHOP_A}"
SEMI = f"{BASE}/rollout/semi-auto"
W38_START = datetime(2026, 9, 14, tzinfo=CST)
W39_START = datetime(2026, 9, 21, tzinfo=CST)
PHONE = "13812345678"


def ready(api: Api, intent: Intent = Intent.SHIPPING_POLICY, shop_id: str = SHOP_A) -> None:
    """两个完整周、每周 20 组、95% 原样采纳：半自动门槛能过。"""
    repo = api.unit.repos.suggestions
    for start in (W38_START, W39_START):
        for i in range(20):
            repo.add(
                SuggestionRecord(
                    shop_id=shop_id,
                    conversation_id=f"conv_{i % 4}",
                    msg_id=f"sa_{shop_id}_{intent.value}_{start:%d}_{i}",
                    question_text="什么时候发货",
                    intent=intent,
                    text="亲，现货付款后 48 小时内发货哦~",
                    status=SuggestionStatus.SENT if i < 19 else SuggestionStatus.EDITED_SENT,
                    created_at=start + timedelta(hours=3, minutes=i),
                )
            )


def semi_auto_intents(api: Api, shop_id: str = SHOP_A) -> list[str] | None:
    shop = api.unit.repos.knowledge.get_shop(shop_id)
    assert shop is not None
    return shop.config_overrides.get("semi_auto_intents")


def events(api: Api, shop_id: str = SHOP_A):
    return api.unit.repos.audit.list_events(shop_id, action=SEMI_AUTO_CHANGED)


# ================================================================ 开启：只有复核人员，按门槛


def test_reviewer_enables_when_the_gate_passes(api: Api):
    ready(api)
    r = api.post(SEMI, "reviewer_a", json={"intent": "shipping_policy", "enabled": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] is True and body["forced"] is False
    assert body["gate"]["passed"] is True and body["semi_auto_intents"] == ["shipping_policy"]
    assert semi_auto_intents(api) == ["shipping_policy"]
    [event] = events(api)
    assert event.actor == "qa_reviewer_01" and event.detail["role"] == "reviewer"
    # 推荐卡片带上 prefill（模式 C 网页工具上显示"可直接发送"）
    r = api.post(
        f"{BASE}/standalone/ask",
        "agent_a",
        json={"session_id": "w1", "question": "什么时候发货", "request_id": "q1"},
    )
    assert r.status_code == 200 and [c["prefill"] for c in r.json()["suggestions"]] == [True]


@pytest.mark.parametrize("caller", ["merchant_a", "manager_a", "agent_a", "integration_a"])
def test_only_reviewers_can_enable(api: Api, caller: str):
    ready(api)
    before = shop_snapshot(api, SHOP_A)
    body = {"intent": "shipping_policy", "enabled": True, "force": True, "note": "我要开"}
    r = api.post(SEMI, caller, json=body)
    assert r.status_code == 403 and r.json()["error"] == "forbidden_role"
    assert shop_snapshot(api, SHOP_A) == before


def test_gate_failure_is_409_with_the_gate_and_force_needs_a_note(api: Api):
    before = shop_snapshot(api, SHOP_A)
    r = api.post(SEMI, "reviewer_a", json={"intent": "shipping_policy", "enabled": True})
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "gate_not_passed" and "强制" in body["message"]
    assert body["gate"]["stage"] == "semi_auto:shipping_policy"
    assert body["gate"]["passed"] is False and body["gate"]["unchecked"] == []
    assert shop_snapshot(api, SHOP_A) == before
    r = api.post(
        SEMI, "reviewer_a", json={"intent": "shipping_policy", "enabled": True, "force": True}
    )
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert shop_snapshot(api, SHOP_A) == before
    r = api.post(
        SEMI,
        "reviewer_a",
        json={
            "intent": "shipping_policy",
            "enabled": True,
            "force": True,
            "note": f"种子商家同意先试，联系 {PHONE}",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["forced"] is True and r.json()["gate"]["passed"] is False
    [event] = events(api)
    assert event.detail["forced"] is True and PHONE not in event.detail["note"]


@pytest.mark.parametrize("intent", ["price_promo", "complaint", "not_received", "off_platform"])
def test_high_risk_and_non_answer_intents_are_refused_even_with_force(api: Api, intent: str):
    before = shop_snapshot(api, SHOP_A)
    body = {"intent": intent, "enabled": True, "force": True, "note": "强制也不行"}
    r = api.post(SEMI, "reviewer_a", json=body)
    assert r.status_code == 409 and r.json()["error"] == "action_not_allowed"
    assert shop_snapshot(api, SHOP_A) == before


# ================================================================ 关闭：商家、复核人员


@pytest.mark.parametrize("caller", ["merchant_a", "reviewer_a"])
def test_merchant_and_reviewer_can_disable(api: Api, caller: str):
    api.unit.repos.knowledge.update_shop_config(
        SHOP_A, {"semi_auto_intents": ["shipping_policy", "invoice_gift"]}
    )
    r = api.post(SEMI, caller, json={"intent": "shipping_policy", "enabled": False})
    assert r.status_code == 200, r.text
    assert r.json()["changed"] is True and r.json()["semi_auto_intents"] == ["invoice_gift"]
    assert semi_auto_intents(api) == ["invoice_gift"]
    [event] = events(api)
    assert event.detail["enabled"] is False and event.detail["role"] == caller.split("_")[0]
    # 再关一次：本来就关着，什么都不写
    r = api.post(SEMI, caller, json={"intent": "shipping_policy", "enabled": False})
    assert r.status_code == 200 and r.json()["changed"] is False and len(events(api)) == 1


@pytest.mark.parametrize("caller", ["manager_a", "agent_a", "integration_a"])
def test_others_cannot_disable(api: Api, caller: str):
    r = api.post(SEMI, caller, json={"intent": "shipping_policy", "enabled": False})
    assert r.status_code == 403 and r.json()["error"] == "forbidden_role"


# ================================================================ 请求校验、串店、取消订购


@pytest.mark.parametrize(
    "body",
    [
        {"intent": "shipping_policy", "enabled": True, "operator": "qa_other"},
        {"intent": "shipping_policy", "enabled": True, "shop_id": SHOP_B},
        {"intent": "shipping_policy", "enabled": "true"},  # 开关必须是布尔值
        {"intent": "shipping_policy", "enabled": 1},
        {"intent": "nope", "enabled": True},
        {"enabled": True},
        {"intent": "shipping_policy", "enabled": True, "force": True, "note": "x" * 501},
        {"intent": "shipping_policy", "auto_reply_schedule": "always", "enabled": True},
    ],
)
def test_bad_bodies_are_422_and_change_nothing(api: Api, body: dict):
    ready(api)
    before = shop_snapshot(api, SHOP_A)
    r = api.post(SEMI, "reviewer_a", json=body)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert shop_snapshot(api, SHOP_A) == before


def test_cross_shop_is_rejected(api: Api):
    ready(api)
    before = shop_snapshot(api, SHOP_A)
    for caller in ("reviewer_b", "merchant_b"):
        r = api.post(SEMI, caller, json={"intent": "shipping_policy", "enabled": False})
        assert r.status_code == 403 and r.json()["error"] == "tenant_isolation"
    r = api.post(SEMI, "reviewer_b", json={"intent": "shipping_policy", "enabled": True})
    assert r.status_code == 403
    assert shop_snapshot(api, SHOP_A) == before
    # B 店的复核人员开 B 店的：B 店自己的推荐记录不够，门槛不过；A 店达标不算 B 店的
    r = api.post(
        f"/v1/shops/{SHOP_B}/rollout/semi-auto",
        "reviewer_b",
        json={"intent": "shipping_policy", "enabled": True},
    )
    assert r.status_code == 409 and r.json()["error"] == "gate_not_passed"


def test_semi_auto_after_revoke_is_refused(api: Api):
    assert api.post(f"{BASE}/revoke", "merchant_a").status_code == 200
    r = api.post(SEMI, "reviewer_a", json={"intent": "shipping_policy", "enabled": False})
    assert r.status_code == 403 and r.json()["error"] == "shop_inactive"
    assert api.unit.repos.knowledge.get_shop(SHOP_A) is None
    assert api.unit.repos.audit.list_events(SHOP_A) == []


# ================================================================ 上线状态：半自动、全自动生效时段


@pytest.mark.parametrize("caller", ["manager_a", "merchant_a", "reviewer_a"])
def test_status_reports_semi_auto_and_schedule_without_texts(api: Api, caller: str):
    ready(api)
    api.unit.repos.knowledge.update_shop_config(
        SHOP_A, {"service_hours": ["09:00-23:00"], "semi_auto_intents": ["invoice_gift"]}
    )
    r = api.get(f"{BASE}/rollout/status", caller)
    assert r.status_code == 200, r.text
    body = r.json()
    semi = body["semi_auto"]
    assert semi["enabled"] == ["invoice_gift"] and semi["threshold"] == 0.9
    rows = {row["intent"]: row for row in semi["intents"]}
    assert rows["shipping_policy"]["gate"]["passed"] is True
    assert rows["price_promo"]["allowed"] is False and rows["price_promo"]["gate"] is None
    schedule = body["auto_schedule"]
    assert schedule["service_hours"] == ["09:00-23:00"]
    assert (
        schedule["auto_reply_enabled"] is False and schedule["auto_reply_schedule"] == "off_hours"
    )
    assert schedule["in_service_hours"] is True and schedule["auto_active_now"] is False
    assert schedule["next_switch_at"].startswith("2026-09-28T23:00:00")
    assert "48 小时" not in r.text and "什么时候发货" not in r.text


def test_merchant_cannot_change_the_auto_schedule_or_semi_auto_in_settings(api: Api):
    before = shop_snapshot(api, SHOP_A)
    for changes in (
        {"auto_reply_schedule": "always"},
        {"auto_reply_enabled": True},
        {"semi_auto_intents": ["shipping_policy"]},
    ):
        r = api.post(f"{BASE}/console/config", "merchant_a", json={"changes": changes})
        assert r.status_code == 409 and r.json()["error"] == "action_not_allowed"
    assert shop_snapshot(api, SHOP_A) == before
