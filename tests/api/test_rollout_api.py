"""上线节奏与每周复核的 HTTP 接口（方案 §13.4 影子对比、阶段切换；§13.5 每周抽样复核；§9.2 查看原文
留访问记录、串店；§4.1 原则 3 操作人取自身份）。401 / 串店 / 角色的逐路由测试在 test_api_auth.py
（路由清单 api_routes_rollout.py）。数据全部合成。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from api_support import Api, shop_snapshot

from rodit.app.demo import taobao_event
from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import Intent, SuggestionStatus
from rodit.core.models import SuggestionRecord
from rodit.testing.fixtures import SHOP_A, SHOP_B

EVENTS = "/v1/platform/B/events"
BASE = f"/v1/shops/{SHOP_A}"
W39 = "2026-W39"
W39_START = datetime(2026, 9, 21, tzinfo=CST)
PHONE = "13812345678"
FULL = {"factual": 2, "relevance": 2, "compliance": 2, "expression": 2}


def add_records(api: Api, n: int, *, shop_id: str = SHOP_A, question: str = "什么时候发货"):
    repo = api.unit.repos.suggestions
    return [
        repo.add(
            SuggestionRecord(
                shop_id=shop_id,
                conversation_id=f"conv_h{i}",
                msg_id=f"h{shop_id}_{i}",
                question_text=question,
                intent=Intent.SHIPPING_POLICY if i % 3 else Intent.INVOICE_GIFT,
                text="亲，现货付款后 48 小时内发货哦~",
                status=SuggestionStatus.SHOWN,
                created_at=W39_START + timedelta(hours=i),
            )
        )
        for i in range(n)
    ]


def shadow_mode(api: Api) -> bool:
    shop = api.unit.repos.knowledge.get_shop(SHOP_A)
    assert shop is not None
    return bool(shop.config_overrides.get("shadow_mode"))


def audit(api: Api, action: AuditAction, shop_id: str = SHOP_A):
    return api.unit.repos.audit.list_events(shop_id, action=action.value)


# ================================================================ 上线状态、影子对比


@pytest.mark.parametrize("caller", ["manager_a", "merchant_a", "reviewer_a"])
def test_status_and_report_are_readable_without_texts(api: Api, caller: str):
    add_records(api, 3, question=f"电话 {PHONE} 什么时候发货")
    r = api.get(f"{BASE}/rollout/status", caller)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stage"] == "assist" and body["advice"] == "stay_assist"
    assert [u["name"] for u in body["unchecked"]] == ["platform_violations"]
    assert len(body["weekly"]) == 4 and body["weekly"][-1]["period"] == W39
    assert body["weekly"][-1]["groups"] == 3
    r = api.get(f"{BASE}/quality/report", caller, params={"period": W39})
    assert r.status_code == 200
    assert r.json()["period"] == W39 and r.json()["gate"]["status"] == "insufficient_data"
    for path in ("/rollout/status", "/rollout/shadow", "/quality/report"):
        text = api.get(f"{BASE}{path}", caller).text
        assert PHONE not in text and "什么时候发货" not in text and "48 小时" not in text
    assert audit(api, AuditAction.QUALITY_VIEWED) == []  # 没有文字的查看不记访问记录


def test_invalid_periods_are_rejected(api: Api):
    for period in ("2026-W99", "2026-40", "2025-W53", "９９９９-W52", "9999-W52"):
        r = api.get(f"{BASE}/quality/report", "reviewer_a", params={"period": period})
        assert r.status_code == 422, (period, r.text)
        r = api.post(f"{BASE}/quality/sample", "reviewer_a", json={"period": period})
        assert r.status_code == 422
    r = api.post(f"{BASE}/quality/sample", "reviewer_a", json={"period": "2026-W41"})
    assert r.status_code == 422 and "还没开始" in r.json()["message"]
    for size in (0, 201):
        assert (
            api.post(f"{BASE}/quality/sample", "reviewer_a", json={"size": size}).status_code == 422
        )
    assert api.unit.repos.quality_reviews.list(SHOP_A) == []


def test_shadow_mode_comparison_over_http(api: Api):
    """复核人员切到影子模式 → 买家消息只生成不展示 → 客服自己回复（回声）→ 影子对比。"""
    r = api.post(f"{BASE}/rollout/stage", "reviewer_a", json={"stage": "shadow"})
    assert r.status_code == 200, r.text
    assert r.json()["changed"] is True and r.json()["previous"] == "assist"
    assert shadow_mode(api)
    t0 = int(api.unit.clock.now().timestamp() * 1000)
    r = api.post(
        EVENTS, "integration_a", json=taobao_event("什么时候发货", msg_id="hs1", send_time_ms=t0)
    )
    assert r.status_code == 200 and r.json()["shadow"] is True and r.json()["suggestions"] == []
    echo = {
        "seller_id": SHOP_A,
        "session_id": "conv_demo",
        "msg_id": "hs_echo",
        "msg_type": "text",
        "direction": "out",
        "content": "亲，现货付款后 48 小时内发货哦~",
        "send_time_ms": t0 + 3_000,
    }
    assert api.post(EVENTS, "integration_a", json=echo).status_code == 204
    r = api.get(f"{BASE}/rollout/shadow", "manager_a")
    assert r.status_code == 200
    body = r.json()
    assert body["groups"] == 1 and body["compared"] == 1 and body["match_rate"] == 1.0
    [item] = body["items"]
    assert item["similarity"] == 1.0 and item["intent"] == "shipping_policy"
    assert set(item) == {"suggestion_id", "intent", "similarity"}  # 不返回平台消息 ID
    assert "48 小时" not in r.text and "hs1" not in r.text
    since = (api.unit.clock.now() + timedelta(minutes=1)).isoformat()
    later = api.get(f"{BASE}/rollout/shadow", "reviewer_a", params={"since": since})
    assert later.status_code == 200 and later.json()["groups"] == 0
    status = api.get(f"{BASE}/rollout/status", "merchant_a").json()
    assert status["stage"] == "shadow" and status["advice"] == "stay_shadow"
    assert status["shadow"]["compared"] == 1
    assert [u["name"] for u in status["unchecked"]] == ["test_set"]


# ================================================================ 阶段切换（§13.4）


def test_merchant_can_only_switch_back_to_shadow(api: Api):
    before = shop_snapshot(api, SHOP_A)
    for body in ({"stage": "assist"}, {"stage": "assist", "force": True, "note": "我要上线"}):
        r = api.post(f"{BASE}/rollout/stage", "merchant_a", json=body)
        assert r.status_code == 403 and r.json()["error"] == "forbidden_role"
    assert shop_snapshot(api, SHOP_A) == before
    r = api.post(f"{BASE}/rollout/stage", "merchant_a", json={"stage": "shadow"})
    assert r.status_code == 200 and r.json()["stage"] == "shadow" and shadow_mode(api)
    [event] = audit(api, AuditAction.ROLLOUT_STAGE)
    assert event.actor == "owner_a" and event.detail["role"] == "merchant"
    # 再切一次：已经是影子模式，什么都不写
    r = api.post(f"{BASE}/rollout/stage", "merchant_a", json={"stage": "shadow"})
    assert r.status_code == 200 and r.json()["changed"] is False
    assert len(audit(api, AuditAction.ROLLOUT_STAGE)) == 1
    # 商家不能切回智能辅助
    r = api.post(f"{BASE}/rollout/stage", "merchant_a", json={"stage": "assist"})
    assert r.status_code == 403 and shadow_mode(api)


def test_assist_gate_failure_is_409_with_the_gate_and_changes_nothing(api: Api):
    api.post(f"{BASE}/rollout/stage", "reviewer_a", json={"stage": "shadow"})
    before = shop_snapshot(api, SHOP_A)
    r = api.post(f"{BASE}/rollout/stage", "reviewer_a", json={"stage": "assist"})
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "gate_not_passed" and "强制" in body["message"]
    gate = body["gate"]
    assert gate["stage"] == "shadow" and gate["passed"] is False
    assert gate["unchecked"] == ["test_set"]
    assert {c["name"] for c in gate["checks"] if not c["passed"]} >= {"shadow_review.scored"}
    assert shop_snapshot(api, SHOP_A) == before and shadow_mode(api)
    # 强制切换必须写明原因
    r = api.post(f"{BASE}/rollout/stage", "reviewer_a", json={"stage": "assist", "force": True})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert shop_snapshot(api, SHOP_A) == before
    r = api.post(
        f"{BASE}/rollout/stage",
        "reviewer_a",
        json={"stage": "assist", "force": True, "note": "离线测试集已通过，店长同意先上"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["forced"] is True and r.json()["stage"] == "assist"
    assert r.json()["gate"]["passed"] is False
    assert not shadow_mode(api)
    event = audit(api, AuditAction.ROLLOUT_STAGE)[-1]
    assert event.actor == "qa_reviewer_01" and event.detail["forced"] is True
    assert event.detail["note"] == "离线测试集已通过，店长同意先上"
    for body in ({"stage": "auto"}, {"stage": "assist", "note": "x" * 501}):
        assert api.post(f"{BASE}/rollout/stage", "reviewer_a", json=body).status_code == 422


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/rollout/stage", {"stage": "shadow", "operator": "qa_other"}),
        ("/rollout/stage", {"stage": "shadow", "shop_id": SHOP_B}),
        ("/quality/sample", {"reviewer": "qa_other"}),
        ("/quality/sample", {"shop_id": SHOP_B}),
        ("/quality/reviews/1", {"status": "skipped", "reviewer": "qa_other"}),
    ],
)
def test_identity_fields_in_rollout_bodies_are_rejected(api: Api, path: str, body: dict):
    before = shop_snapshot(api, SHOP_A)
    r = api.post(f"{BASE}{path}", "reviewer_a", json=body)
    assert r.status_code == 422 and "Extra inputs are not permitted" in r.json()["message"]
    assert shop_snapshot(api, SHOP_A) == before


# ================================================================ 每周复核（§13.5）


def test_weekly_review_flow_over_http(api: Api):
    add_records(api, 6, question=f"我电话 {PHONE}，什么时候发货")
    r = api.post(f"{BASE}/quality/sample", "reviewer_a", json={})
    assert r.status_code == 200, r.text
    sample = r.json()
    assert sample["period"] == W39 and sample["added"] == 6 and sample["total"] == 6
    assert sample["by_intent"] == {"shipping_policy": 4, "invoice_gift": 2}
    [sampled] = audit(api, AuditAction.QUALITY_SAMPLED)
    assert sampled.actor == "qa_reviewer_01"

    r = api.get(f"{BASE}/quality/reviews", "reviewer_a")
    assert r.status_code == 200
    listed = r.json()
    assert listed["period"] == W39 and listed["count"] == 6
    first = listed["items"][0]
    assert first["suggestion"]["question_text"] == "我电话 [手机号]，什么时候发货"
    assert first["suggestion"]["text"] == "亲，现货付款后 48 小时内发货哦~"
    assert PHONE not in r.text
    [viewed] = audit(api, AuditAction.QUALITY_VIEWED)
    assert viewed.actor == "qa_reviewer_01" and viewed.detail["count"] == 6
    assert "什么时候发货" not in str(viewed.detail)

    r = api.post(
        f"{BASE}/quality/reviews/{first['id']}",
        "reviewer_a",
        json={"status": "scored", "scores": FULL, "note": f"没问题，但买家电话 {PHONE} 别外传"},
    )
    assert r.status_code == 200, r.text
    scored = r.json()
    assert scored["status"] == "scored" and scored["reviewer"] == "qa_reviewer_01"
    assert scored["note"] == "没问题，但买家电话 [手机号] 别外传" and scored["scores"] == FULL
    [event] = audit(api, AuditAction.QUALITY_SCORED)
    assert event.detail["scores"] == FULL and "note" not in event.detail
    r = api.post(
        f"{BASE}/quality/reviews/{listed['items'][1]['id']}",
        "reviewer_a",
        json={"status": "skipped"},
    )
    assert r.status_code == 200 and r.json()["scores"] is None

    r = api.get(f"{BASE}/quality/reviews", "reviewer_a", params={"status": "pending"})
    assert r.json()["count"] == 4
    assert len(audit(api, AuditAction.QUALITY_VIEWED)) == 2
    report = api.get(f"{BASE}/quality/report", "manager_a")
    assert report.status_code == 200
    rep = report.json()
    assert (rep["sampled"], rep["scored"], rep["skipped"], rep["pending"]) == (6, 1, 1, 4)
    assert rep["dimensions"]["factual"]["full_rate"] == 1.0
    assert "别外传" not in report.text and PHONE not in report.text
    # 店长、商家看不到复核原文
    for caller in ("manager_a", "merchant_a"):
        assert api.get(f"{BASE}/quality/reviews", caller).status_code == 403


def test_score_request_validation_and_unknown_reviews(api: Api):
    add_records(api, 1)
    api.post(f"{BASE}/quality/sample", "reviewer_a", json={"period": W39})
    [review] = api.unit.repos.quality_reviews.list(SHOP_A)
    url = f"{BASE}/quality/reviews/{review.id}"
    for body in (
        {"status": "scored"},
        {"status": "skipped", "scores": FULL},
        {"status": "pending"},
        {"status": "scored", "scores": {**FULL, "factual": 3}},
        {"status": "scored", "scores": {**FULL, "extra": 1}},
        {"status": "scored", "scores": FULL, "note": "x" * 501},
    ):
        r = api.post(url, "reviewer_a", json=body)
        assert r.status_code == 422, body
    assert (
        api.post(f"{BASE}/quality/reviews/0", "reviewer_a", json={"status": "skipped"}).status_code
        == 422
    )
    r = api.post(f"{BASE}/quality/reviews/999", "reviewer_a", json={"status": "skipped"})
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    assert api.unit.repos.quality_reviews.get(SHOP_A, review.id).status == "pending"


def test_review_id_of_another_shop_is_not_found(api: Api):
    """§9.2：店铺 A 的复核人员用店铺 B 的样本 ID 打分 → 404，店铺 B 的样本不变、不透露内容。"""
    add_records(api, 1, shop_id=SHOP_B, question=f"B 店买家 {PHONE}")
    r = api.post(f"/v1/shops/{SHOP_B}/quality/sample", "reviewer_b", json={"period": W39})
    assert r.status_code == 200 and r.json()["added"] == 1
    [b_review] = api.unit.repos.quality_reviews.list(SHOP_B)
    before = shop_snapshot(api, SHOP_B)
    r = api.post(f"{BASE}/quality/reviews/{b_review.id}", "reviewer_a", json={"status": "skipped"})
    assert r.status_code == 404 and "B 店" not in r.text
    r = api.get(f"{BASE}/quality/reviews", "reviewer_a", params={"period": W39})
    assert r.status_code == 200 and r.json()["count"] == 0
    assert shop_snapshot(api, SHOP_B) == before
    # 店铺 B 的复核人员打不了店铺 A 路径下的样本（串店 403）
    r = api.post(f"{BASE}/quality/reviews/{b_review.id}", "reviewer_b", json={"status": "skipped"})
    assert r.status_code == 403 and r.json()["error"] == "tenant_isolation"
    assert shop_snapshot(api, SHOP_B) == before


def test_rollout_writes_after_revoke_are_refused(api: Api):
    add_records(api, 2)
    api.post(f"{BASE}/quality/sample", "reviewer_a", json={"period": W39})
    [review_id, _] = [q.id for q in api.unit.repos.quality_reviews.list(SHOP_A)]
    assert api.post(f"{BASE}/revoke", "merchant_a").status_code == 200
    calls = (
        ("POST", "/quality/sample", {"period": W39}),
        ("POST", f"/quality/reviews/{review_id}", {"status": "skipped"}),
        ("GET", "/quality/reviews", None),
        ("POST", "/rollout/stage", {"stage": "shadow"}),
        ("GET", "/rollout/status", None),
    )
    for method, path, body in calls:
        r = api.call(method, f"{BASE}{path}", "reviewer_a", json=body)
        assert r.status_code == 403 and r.json()["error"] == "shop_inactive", path
    assert api.unit.repos.quality_reviews.list(SHOP_A) == []
    assert api.unit.repos.knowledge.get_shop(SHOP_A) is None
    assert api.unit.repos.audit.list_events(SHOP_A) == []
