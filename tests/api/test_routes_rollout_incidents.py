"""复核人员登记平台违规通知和投诉（方案 §13.4 "平台违规为 0""没有因此产生的投诉"），上线状态据此
检查门槛；测试集只认配置指纹和线上一致的评测结论（§13.2、§13.5）。401 / 串店 / 角色的逐路由测试在
test_api_auth.py（路由清单 api_routes_rollout.py）。数据全部合成（演示时钟在 2026-09-28 周一，
W40）。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from api_support import Api

from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import Role
from rodit.eval.fingerprint import current_fingerprint, verdict_event, verdict_from_report
from rodit.testing.fakes import make_message
from rodit.testing.fixtures import SHOP_A

BASE = f"/v1/shops/{SHOP_A}"
W38, W39 = "2026-W38", "2026-W39"
W39_START = datetime(2026, 9, 21, tzinfo=CST)
PHONE = "13812345678"


def incidents(api: Api):
    return api.unit.repos.audit.list_events(SHOP_A, action=AuditAction.ROLLOUT_INCIDENT.value)


def record(api: Api, week: str, count: int, *, kind: str = "violation", who: str = "reviewer_a"):
    return api.post(
        f"{BASE}/rollout/incidents", who, json={"kind": kind, "week": week, "count": count}
    )


def status(api: Api, who: str = "manager_a") -> dict:
    r = api.get(f"{BASE}/rollout/status", who)
    assert r.status_code == 200, r.text
    return r.json()


def checks(gate: dict) -> dict[str, dict]:
    return {c["name"]: c for c in gate["checks"]}


# ================================================================ 登记（§13.4）


def test_reviewer_records_an_incident_and_it_is_audited(api: Api):
    r = api.post(
        f"{BASE}/rollout/incidents",
        "reviewer_a",
        json={"kind": "violation", "week": W39, "count": 2, "note": f"平台通知，联系 {PHONE}"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body == {
        "kind": "violation",
        "week": W39,
        "count": 2,
        "note": "平台通知，联系 [手机号]",
        "reviewer": "qa_reviewer_01",
        "recorded_at": body["recorded_at"],
        "current": True,
    }
    [event] = incidents(api)
    assert event.actor == "qa_reviewer_01" and event.shop_id == SHOP_A
    assert event.detail == {
        "kind": "violation",
        "week": W39,
        "count": 2,
        "note": "平台通知，联系 [手机号]",
    }
    assert PHONE not in str(event.detail)
    # 店长、商家能看登记列表（只有周、条数、备注）
    listed = api.get(f"{BASE}/rollout/incidents", "merchant_a")
    assert listed.status_code == 200
    [item] = listed.json()["items"]
    assert item["count"] == 2 and item["current"] is True


@pytest.mark.parametrize("caller", ["agent_a", "merchant_a", "manager_a", "integration_a"])
def test_only_reviewers_can_record_incidents(api: Api, caller: str):
    r = record(api, W39, 0, who=caller)
    assert r.status_code == 403 and r.json()["error"] == "forbidden_role", r.text
    assert incidents(api) == []


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "violation", "week": "2026-W41", "count": 0},  # 还没开始的周
        {"kind": "violation", "week": "2026-W99", "count": 0},
        {"kind": "violation", "week": W39, "count": -1},
        {"kind": "violation", "week": W39, "count": "3"},
        {"kind": "violation", "week": W39, "count": True},
        {"kind": "violation", "week": W39, "count": 10_001},
        {"kind": "refund", "week": W39, "count": 0},
        {"kind": "violation", "week": W39, "count": 0, "reviewer": "someone_else"},
        {"kind": "violation", "week": W39, "count": 0, "note": "长" * 501},
    ],
)
def test_invalid_incidents_are_rejected_without_writing(api: Api, body: dict):
    r = api.post(f"{BASE}/rollout/incidents", "reviewer_a", json=body)
    assert r.status_code == 422, r.text
    assert incidents(api) == []


# ================================================================ 上线状态（§13.4）


def test_platform_violations_are_checked_once_both_weeks_are_recorded(api: Api):
    s = status(api)
    assert s["stage"] == "assist"
    [unchecked] = s["unchecked"]
    assert unchecked["name"] == "platform_violations"
    assert W38 in unchecked["reason"] and W39 in unchecked["reason"]
    assert record(api, W38, 0).status_code == 201
    [unchecked] = status(api)["unchecked"]
    assert W38 not in unchecked["reason"] and W39 in unchecked["reason"]

    assert record(api, W39, 0).status_code == 201
    s = status(api)
    assert s["unchecked"] == []
    gate = checks(s["assist_gate"])
    assert gate["platform_violations"]["passed"] and gate["platform_violations"]["value"] == 0
    assert gate["compliance_scan"]["passed"] and gate["compliance_scan"]["value"] == 0
    assert s["evidence"]["platform_violations"]["by_week"] == {W38: 0, W39: 0}

    # 登记了 1 条：门槛不通过；改正（同一周再登记）以最后一次为准
    assert record(api, W39, 1).status_code == 201
    s = status(api)
    assert "platform_violations" in s["blocking"] and not s["assist_gate"]["passed"]
    assert checks(s["assist_gate"])["platform_violations"]["value"] == 1
    assert record(api, W39, 0).status_code == 201
    assert "platform_violations" not in status(api)["blocking"]
    items = api.get(f"{BASE}/rollout/incidents", "reviewer_a").json()["items"]
    assert [(i["week"], i["count"], i["current"]) for i in items] == [
        (W39, 0, True),
        (W39, 1, False),
        (W39, 0, False),
        (W38, 0, True),
    ]


def test_a_violation_recorded_this_week_also_fails_the_gate(api: Api):
    assert record(api, "2026-W40", 1).status_code == 201  # 本周（没登记全也算：已知有违规）
    s = status(api)
    assert "platform_violations" in s["blocking"] and s["unchecked"] == []


def test_complaints_are_checked_for_auto_stage(api: Api):
    repo = api.unit.repos.knowledge
    shop = repo.get_shop(SHOP_A)
    repo.upsert_shop(
        shop.model_copy(
            update={"config_overrides": {**shop.config_overrides, "auto_reply_enabled": True}}
        )
    )
    s = status(api)
    assert s["stage"] == "auto"
    [unchecked] = s["unchecked"]
    assert unchecked["name"] == "complaints" and "2026-W36" in unchecked["reason"]
    for week in ("2026-W36", "2026-W37", W38, W39):
        assert record(api, week, 0, kind="complaint").status_code == 201
    s = status(api)
    assert s["unchecked"] == []
    for gate in s["intent_gates"].values():
        assert checks(gate)["complaints"]["passed"]
    assert record(api, W39, 2, kind="complaint").status_code == 201
    s = status(api)
    assert all(not checks(g)["complaints"]["passed"] for g in s["intent_gates"].values())


def test_compliance_scan_flags_sent_text_without_returning_it(api: Api):
    conv = api.unit.repos.conversations
    at = W39_START + timedelta(days=2)
    conv.add_message(make_message("有优惠吗", at=at, conversation_id="conv_scan", msg_id="b_scan"))
    conv.add_outgoing(
        shop_id=SHOP_A,
        conversation_id="conv_scan",
        msg_id="out_scan_1",
        role=Role.AGENT,
        text="这款是全网最低价，加我微信 abc123",
        at=at + timedelta(minutes=1),
    )
    for week in (W38, W39):
        record(api, week, 0)
    r = api.get(f"{BASE}/rollout/status", "manager_a")
    s = r.json()
    gate = checks(s["assist_gate"])
    # 扫描命中只是候选：等复核人员确认，列为 unchecked（写明 ID），不算没通过
    assert "compliance_scan" not in gate and "compliance_scan" in s["assist_gate"]["unchecked"]
    [unchecked] = [u for u in s["unchecked"] if u["name"] == "compliance_scan"]
    assert "out_scan_1" in unchecked["reason"]
    assert gate["platform_violations"]["passed"]  # 两条分开报告
    scan = s["evidence"]["compliance"]
    assert scan["message_ids"] == ["out_scan_1"] and scan["by_period"][W39] == 1
    assert scan["pending"] == 1 and scan["pending_message_ids"] == ["out_scan_1"]
    # 店铺 A 的禁用词里有"全网最低"（合成配置）
    assert scan["rule_counts"] == {"off_platform": 1, "absolute_term": 1, "banned_word": 1}
    for word in ("全网最低价", "微信", "abc123", "有优惠吗"):
        assert word not in r.text
    # 复核人员确认是违规：门槛没通过
    body = {"msg_id": "out_scan_1", "decision": "confirmed"}
    confirmed = api.post(f"{BASE}/rollout/compliance-reviews", "reviewer_a", json=body)
    assert confirmed.status_code == 201, confirmed.text
    s = status(api)
    gate = checks(s["assist_gate"])
    assert not gate["compliance_scan"]["passed"] and gate["compliance_scan"]["value"] == 1
    assert "compliance_scan" in s["blocking"]
    assert s["evidence"]["compliance"]["confirmed"] == 1


def test_compliance_review_endpoint_validates_the_flag(api: Api):
    """确认 / 标记误报：只能是最近扫描出来的疑似违规（否则 404），ID 给且只给一个（否则 422），
    复核人取自调用方身份（请求体带 reviewer 422）。"""
    conv = api.unit.repos.conversations
    at = W39_START + timedelta(days=2)
    conv.add_message(make_message("有优惠吗", at=at, conversation_id="conv_fp", msg_id="b_fp"))
    conv.add_outgoing(
        shop_id=SHOP_A,
        conversation_id="conv_fp",
        msg_id="out_fp_1",
        role=Role.AGENT,
        text="亲，面料：桑蚕丝 100%",
        at=at + timedelta(minutes=1),
    )
    path = f"{BASE}/rollout/compliance-reviews"
    for body in (
        {"decision": "dismissed"},
        {"msg_id": "out_fp_1", "suggestion_id": 1, "decision": "dismissed"},
        {"msg_id": "out_fp_1", "decision": "ignore"},
        {"suggestion_id": "1", "decision": "dismissed"},
        {"msg_id": "out_fp_1", "decision": "dismissed", "reviewer": "someone_else"},
    ):
        r = api.post(path, "reviewer_a", json=body)
        assert r.status_code == 422, (body, r.text)
    r = api.post(path, "reviewer_a", json={"msg_id": "out_nope", "decision": "dismissed"})
    assert r.status_code == 404, r.text
    r = api.post(
        path,
        "reviewer_a",
        json={"msg_id": "out_fp_1", "decision": "dismissed", "note": f"误报，联系 {PHONE}"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["decision"] == "dismissed" and body["reviewer"] == "qa_reviewer_01"
    assert body["rules"] == ["absolute_term"] and PHONE not in r.text
    scan = status(api)["evidence"]["compliance"]
    assert (scan["flagged"], scan["dismissed"], scan["pending"]) == (1, 1, 0)


def test_test_set_is_checked_only_with_a_matching_verdict(api: Api, monkeypatch):
    # 线上的大模型（合成的地址和模型名）：测试集依据要知道线上用的是哪个模型
    monkeypatch.setenv("RODIT_LLM_BASE_URL", "https://llm.example.invalid/v1")
    monkeypatch.setenv("RODIT_LLM_MODEL", "synthetic-model")
    assert api.post(f"{BASE}/rollout/stage", "reviewer_a", json={"stage": "shadow"}).status_code
    s = status(api)
    [unchecked] = s["unchecked"]
    assert unchecked["name"] == "test_set" and "record-eval" in unchecked["reason"]

    audit = api.unit.repos.audit
    running = current_fingerprint(api.unit.config)
    report = {
        "split": "all",
        "n": 324,
        "gate": {"n": 66, "threshold": 0.95, "factual": 0.97, "compliance": 1.0, "passes": True},
        "llm": {"mode": "env", "model": "synthetic-model", "decision_provider": "incloud_llm"},
        # 运行健康度（rodit.eval.health）：一轮干净的评测。真实大模型跑的报告没有这一块不收
        "health": {
            "ceiling": 0.02,
            "overall": {"cases": 324},
            "test": {"cases": 66},
            "valid": True,
        },
        "fingerprint": running.model_dump(mode="json"),
    }
    stale = {**report, "fingerprint": {**report["fingerprint"], "config_fingerprint": "0" * 64}}
    now = api.unit.clock.now()
    audit.record(verdict_event(verdict_from_report(stale), at=now - timedelta(hours=1)))
    [unchecked] = status(api)["unchecked"]
    assert unchecked["name"] == "test_set" and "不一致" in unchecked["reason"]

    audit.record(verdict_event(verdict_from_report(report), at=now - timedelta(minutes=30)))
    s = status(api)
    assert s["unchecked"] == []
    test_set = checks(s["shadow_gate"])["test_set"]
    assert test_set["passed"] and "synthetic-model" in test_set["detail"]
    assert s["evidence"]["test_set"]["running_fingerprint"] == running.config_fingerprint
