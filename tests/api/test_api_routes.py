"""各路由的正常流程（§8.1 模式 C、§8.2 平台事件、§5.8 反馈、§5.2.3 问卷、§12.4 报表、
§7.3.2 售后审核、§8.4 取消订购、§13.4 影子模式）。"""

from __future__ import annotations

import json

import pytest
from api_support import EVENTS_PATH, MERCHANT_A, Api

from rodit.app.demo import taobao_event
from rodit.core.enums import AfterSaleStatus, KnowledgeStatus, SuggestionStatus
from rodit.core.errors import ToolUnavailable
from rodit.core.pii import BuyerRefHasher
from rodit.testing.fixtures import RAW_BUYERS, SHOP_A, SHOP_B, SHOP_D, TEST_SALT

A = f"/v1/shops/{SHOP_A}"

# 响应里不该出现的内部字段（§5.3 最小化个人信息、§12.3）
FORBIDDEN_KEYS = {
    "buyer_ref",
    "msg_ids",
    "msg_id",
    "route",
    "answers",
    "qa_scores",
    "latency_ms",
    "question_text",
    "note_raw",
    "access_token",
}


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def _event(api: Api, text: str, msg_id: str, **kw):
    return api.post(EVENTS_PATH, "integration_a", json=taobao_event(text, msg_id=msg_id, **kw))


# ================================================================ 模式 C（§8.1）


def test_standalone_ask_returns_cards_with_evidence_labels(api: Api):
    r = api.post(
        f"{A}/standalone/ask", "agent_a", json={"session_id": "web_1", "question": "能开发票吗"}
    )
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["shop_id"] == SHOP_A and view["mode"] == "assist" and view["shadow"] is False
    assert view["conversation_id"] == "standalone:web_1"
    [card] = view["suggestions"]
    assert "发票" in card["text"] and card["suggestion_id"] >= 1
    assert card["confidence"] in {"high", "medium"}
    assert card["evidence"] and all(set(e) == {"id", "label"} for e in card["evidence"])
    assert {"alerts", "no_suggestion_reason", "missing_knowledge"} <= set(view)
    assert not (_keys(view) & FORBIDDEN_KEYS)
    # 客服账号取自身份：写进了引擎处理的消息
    [msg] = api.unit.repos.conversations.recent_messages(SHOP_A, "standalone:web_1")
    assert msg.text == "能开发票吗"


def test_standalone_ask_with_item_and_request_id_dedup(api: Api):
    body = {
        "session_id": "web_2",
        "question": "165 110斤穿什么码",
        "item_id": "i_555",
        "request_id": "q1",
    }
    r = api.post(f"{A}/standalone/ask", "agent_a", json=body)
    assert r.status_code == 200
    assert "M" in r.json()["suggestions"][0]["text"]
    # 同一 request_id 重试：重复消息，不再处理
    again = api.post(f"{A}/standalone/ask", "agent_a", json=body)
    assert again.status_code == 204 and again.content == b""


@pytest.mark.parametrize(
    "body",
    [
        {"session_id": "w"},
        {"question": "在吗"},
        {"session_id": "", "question": "在吗"},
        {"session_id": "w", "question": ""},
        {"session_id": "w", "question": "在吗" * 1500},
        {"session_id": "w" * 200, "question": "在吗"},
    ],
)
def test_standalone_ask_validates_body(api: Api, body):
    r = api.post(f"{A}/standalone/ask", "agent_a", json=body)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_shadow_shop_hides_suggestions_in_mode_c(douyin_api: Api):
    """§13.4 影子模式：生成、记录，但不展示。"""
    r = douyin_api.post(
        f"/v1/shops/{SHOP_D}/standalone/ask",
        "agent_d",
        json={"session_id": "w", "question": "什么时候发货"},
    )
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["shadow"] is True and view["mode"] == "shadow"
    assert view["suggestions"] == [] and view["auto_reply"] is None
    assert view["alerts"] == [] and view["missing_knowledge"] == []
    records = douyin_api.unit.repos.suggestions.list(SHOP_D)
    assert records and all(r.status == SuggestionStatus.SHADOW for r in records)
    assert not any(r.text in json.dumps(view, ensure_ascii=False) for r in records)


def test_shadow_shop_hides_suggestions_for_platform_events(douyin_api: Api):
    raw = {
        "shop_id": SHOP_D,
        "conversation_short_id": "dy_conv_1",
        "server_message_id": "dy_1",
        "sender": {"role": "customer", "uid": "dy_demo_buyer_001", "nickname": "合成"},
        "content": {"text": "什么时候发货"},
        "create_time": int(douyin_api.unit.clock.now().timestamp()),
    }
    r = douyin_api.post(EVENTS_PATH, "integration_d", json=raw)
    assert r.status_code == 200, r.text
    assert r.json()["shadow"] is True and r.json()["suggestions"] == []
    adapter = douyin_api.unit.adapters.get(douyin_api.unit.platform, "B")
    assert adapter.sidebar(SHOP_D, "dy_conv_1") == []


# ================================================================ 平台事件（§8.1 模式 A / B）


def test_platform_event_flows_to_sidebar_and_hides_buyer_identity(api: Api):
    r = _event(api, "165 110斤穿什么码", "e1", item_id="i_555")
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["conversation_id"] == "conv_demo" and view["suggestions"]
    adapter = api.unit.adapters.get(api.unit.platform, "B")
    assert [c.suggestion_id for c in adapter.sidebar(SHOP_A, "conv_demo")] == [
        s["suggestion_id"] for s in view["suggestions"]
    ]
    # 响应里没有买家身份：buyer_ref、平台原始买家 ID、昵称、消息 ID 都不出现
    _, raw_buyer = RAW_BUYERS["BUYER"]
    buyer_ref = BuyerRefHasher(TEST_SALT)("taobao", raw_buyer)
    text = r.text
    for secret in (buyer_ref, raw_buyer, "合成昵称", "kf_02", '"e1"'):
        assert secret not in text
    assert not (_keys(view) & FORBIDDEN_KEYS)


def test_platform_event_duplicates_and_non_buyer_events_return_204(api: Api):
    assert _event(api, "什么时候发货", "e2").status_code == 200
    assert _event(api, "什么时候发货", "e2").status_code == 204  # 按 msg_id 去重（§8.3）
    image = {**taobao_event("x", msg_id="e3"), "msg_type": "image"}
    assert api.post(EVENTS_PATH, "integration_a", json=image).status_code == 204
    echo = {**taobao_event("好的亲", msg_id="e4"), "direction": "out"}
    assert api.post(EVENTS_PATH, "integration_a", json=echo).status_code == 204
    garbage = {"hello": "world"}
    assert api.post(EVENTS_PATH, "integration_a", json=garbage).status_code == 204
    # 客服回声记进了会话上下文（适配器处理了），但不产生推荐
    adapter = api.unit.adapters.get(api.unit.platform, "B")
    assert [m.msg_id for m in adapter.get_context(SHOP_A, "conv_demo").recent_messages] == [
        "e2",
        "e4",
    ]


def test_platform_event_access_mode_checks(api: Api):
    raw = taobao_event("什么时候发货", msg_id="e5")
    r = api.post("/v1/platform/C/events", "integration_a", json=raw)
    assert r.status_code == 422  # 模式 C 走 standalone/ask
    r = api.post("/v1/platform/A/events", "integration_a", json=raw)
    assert r.status_code == 404 and r.json()["error"] == "not_found"  # 演示单元没有注册模式 A
    r = api.post(EVENTS_PATH, "integration_a", json=[raw])
    assert r.status_code == 422


def test_platform_event_rejected_when_shop_auth_expired(api: Api):
    """§8.4：授权过期或取消订购后停止处理；适配器也不记录这条消息。"""
    api.unit.repos.auth.delete(SHOP_A)
    r = _event(api, "在吗", "e6", session_id="conv_expired")
    assert r.status_code == 403 and r.json()["error"] == "shop_not_authorized"
    adapter = api.unit.adapters.get(api.unit.platform, "B")
    assert adapter.get_context(SHOP_A, "conv_expired").recent_messages == []


def test_suspended_shop_is_rejected_everywhere(api: Api):
    repo = api.unit.repos.knowledge
    repo.upsert_shop(repo.get_shop(SHOP_A).model_copy(update={"status": "suspended"}))
    assert _event(api, "在吗", "e7").json()["error"] == "shop_inactive"
    r = api.post(f"{A}/standalone/ask", "agent_a", json={"session_id": "w", "question": "在吗"})
    assert r.status_code == 403 and r.json()["error"] == "shop_inactive"
    assert api.get(f"{A}/reports/qa", "manager_a").status_code == 403


# ================================================================ 反馈（§5.8）


def test_feedback_sent_then_second_click_conflicts(api: Api):
    sid = _event(api, "什么时候发货", "f1").json()["suggestions"][0]["suggestion_id"]
    r = api.post(f"{A}/suggestions/{sid}/sent", "agent_a")
    assert r.status_code == 200
    assert r.json() == {
        "suggestion_id": sid,
        "status": "sent",
        "candidate_knowledge_id": None,
        "candidate_skipped": None,
        "knowledge_gap": None,
        "labeled": True,
    }
    r = api.post(f"{A}/suggestions/{sid}/sent", "agent_a", json={})
    assert r.status_code == 409 and r.json()["error"] == "action_not_allowed"
    r = api.post(f"{A}/suggestions/999999/sent", "agent_a")
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    r = api.post(f"{A}/suggestions/0/sent", "agent_a")
    assert r.status_code == 422


def test_feedback_on_another_shops_suggestion_id_is_not_found(api: Api):
    """推荐 ID 属于店铺 B：店铺 A 的客服用自己店铺的路径访问，和"不存在"一样（§9.2）。"""
    raw = taobao_event("什么时候发货", msg_id="fb", seller_id=SHOP_B, session_id="conv_b")
    sid = api.post(EVENTS_PATH, "integration_b", json=raw).json()["suggestions"][0]["suggestion_id"]
    r = api.post(f"{A}/suggestions/{sid}/sent", "agent_a")
    assert r.status_code == 404
    assert api.unit.repos.suggestions.get(SHOP_B, sid).status == SuggestionStatus.SHOWN


def test_edited_feedback_creates_candidate_and_merchant_reviews_it(api: Api):
    sid = _event(api, "新疆能到吗", "k1").json()["suggestions"][0]["suggestion_id"]
    r = api.post(
        f"{A}/suggestions/{sid}/edited",
        "agent_a",
        json={"edited_text": "亲，新疆可以发，补 15 元运费哦~"},
    )
    assert r.status_code == 200, r.text
    kid = r.json()["candidate_knowledge_id"]
    assert r.json()["status"] == "edited_sent" and kid is not None

    r = api.get(f"{A}/knowledge/candidates", "merchant_a")
    assert r.status_code == 200
    [candidate] = [k for k in r.json() if k["id"] == kid]
    assert candidate["status"] == "candidate" and "15 元" in candidate["answer"]
    assert "embedding" not in candidate

    r = api.post(f"{A}/knowledge/candidates/{kid}/approve", "merchant_a")
    assert r.status_code == 200
    assert r.json()["status"] == "approved" and r.json()["approved_by"] == MERCHANT_A
    r = api.post(f"{A}/knowledge/candidates/{kid}/approve", "merchant_a")
    assert r.status_code == 409  # 已经不是候选
    r = api.post(f"{A}/knowledge/{kid}/retire", "merchant_a", json={})
    assert r.status_code == 200 and r.json()["status"] == "retired"
    assert api.unit.repos.knowledge.get_knowledge(SHOP_A, kid).status == KnowledgeStatus.RETIRED
    r = api.post(f"{A}/knowledge/{kid}/retire", "merchant_a")
    assert r.status_code == 409
    r = api.post(f"{A}/knowledge/candidates/999999/approve", "merchant_a")
    assert r.status_code == 404


def test_rejected_feedback_feeds_gaps_and_labels(api: Api):
    sid = _event(api, "这个是纯棉的吗", "g1", item_id="i_555").json()["suggestions"][0][
        "suggestion_id"
    ]
    r = api.post(f"{A}/suggestions/{sid}/rejected", "agent_a", json={"reason": "missing_knowledge"})
    assert r.status_code == 200, r.text
    marker = r.json()["knowledge_gap"]
    assert r.json()["status"] == "rejected" and marker
    r = api.get(f"{A}/knowledge/gaps", "manager_a")
    assert r.status_code == 200
    gaps = r.json()
    assert any(g["marker"] == marker for g in gaps)
    assert all(set(g) == {"marker", "description"} for g in gaps)

    card = _event(api, "什么时候发货", "g2", session_id="conv_g2").json()["suggestions"][0]
    r = api.post(
        f"{A}/suggestions/{card['suggestion_id']}/rejected",
        "agent_a",
        json={"reason": "wrong_intent", "true_intent": "logistics_track"},
    )
    assert r.status_code == 200 and r.json()["labeled"] is True
    [label] = api.unit.repos.feedback.list_labels(SHOP_A)
    assert label.true_intent == "logistics_track" and label.predicted_intent == card["intent"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"reason": "nope"},
        {"reason": "wrong_intent", "true_intent": "nope"},
        {"edited_text": "x"},
    ],
)
def test_rejected_feedback_validates_body(api: Api, body):
    r = api.post(f"{A}/suggestions/1/rejected", "agent_a", json=body)
    assert r.status_code == 422


def test_edited_feedback_requires_text(api: Api):
    for body in ({}, {"edited_text": ""}, {"edited_text": "x" * 2001}):
        assert api.post(f"{A}/suggestions/1/edited", "agent_a", json=body).status_code == 422


# ================================================================ 问卷（附录 C）


def test_questionnaire_writes_policies_and_config(api: Api):
    r = api.post(
        f"{A}/questionnaire",
        "merchant_a",
        json={
            "answers": {
                "7": "可以开电子专用发票，确认收货后 2 个工作日内开出",
                "15": "专业一点，称呼用您",
            }
        },
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"policies": ["invoice"], "config": ["salutation", "tone"]}
    policies = api.unit.repos.knowledge.get_policies(SHOP_A)
    assert "专用发票" in policies["invoice"].value
    shop = api.unit.repos.knowledge.get_shop(SHOP_A)
    assert shop.config_overrides["salutation"] == "您"
    assert api.unit.shop_config(SHOP_A).tone == "专业"


@pytest.mark.parametrize(
    "body",
    [
        {"answers": {}},
        {"answers": {"99": "x"}},
        {"answers": {"abc": "x"}},
        {"answers": {"7": 1}},
        {},
    ],
)
def test_questionnaire_rejects_bad_answers(api: Api, body):
    r = api.post(f"{A}/questionnaire", "merchant_a", json=body)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert "invoice" in api.unit.repos.knowledge.get_policies(SHOP_A)  # 原政策没动


# ================================================================ 报表（§12.4）


def test_reports_adoption_and_qa(api: Api):
    sid = _event(api, "什么时候发货", "rp1").json()["suggestions"][0]["suggestion_id"]
    api.post(f"{A}/suggestions/{sid}/sent", "agent_a")
    r = api.get(f"{A}/reports/adoption", "manager_a")
    assert r.status_code == 200, r.text
    report = r.json()
    assert report["overall"]["sent"] == 1 and report["shop_ids"] == [SHOP_A]
    assert "shipping_policy" in report["by_intent"]

    future = api.unit.clock.now().replace(year=2030).isoformat()
    r = api.get(f"{A}/reports/adoption", "merchant_a", params={"since": future})
    assert r.status_code == 200 and r.json()["overall"]["total"] == 0

    r = api.get(f"{A}/reports/qa", "merchant_a")
    assert r.status_code == 200
    assert r.json()["messages"] == 1 and r.json()["shop_id"] == SHOP_A

    r = api.get(f"{A}/reports/qa", "manager_a", params={"since": "2026-09-28T00:00:00"})
    assert r.status_code == 422  # since 必须带时区
    r = api.get(f"{A}/reports/qa", "manager_a", params={"since": "昨天"})
    assert r.status_code == 422


# ================================================================ 售后审核（§7.3.2）


def test_aftersale_review_negotiate_and_confirm(api: Api):
    url = f"{A}/aftersales/as_2002"
    r = api.post(f"{url}/review", "agent_a")
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["state"] == "confirm" and view["matched_rules"] == ["R3"]
    assert view["suggestion"] == "agree_return" and view["requires_manager"] is False
    assert view["platform"]["status"] == "pending_review"
    assert view["platform"]["amount"] == "199.00"
    assert [(h["from_state"], h["to_state"]) for h in view["history"]] == [
        ("new", "suggest"),
        ("suggest", "confirm"),
    ]

    r = api.post(f"{url}/negotiate", "agent_a", json={"note": "买家电话 13812345678，先协商换码"})
    assert r.status_code == 200 and r.json()["state"] == "negotiate"
    assert r.json()["platform"] is None  # 只有待确认时才附平台状态
    last = r.json()["history"][-1]
    assert last["by"] == "kf_02" and "13812345678" not in (last["note"] or "")

    r = api.post(f"{url}/negotiation-done", "agent_a", json={"outcome": "买家同意退货"})
    assert r.status_code == 200 and r.json()["state"] == "confirm"
    version = r.json()["platform"]["status_version"]

    stale = api.post(
        f"{url}/confirm", "agent_a", json={"action": "agree_return", "expected_status_version": "0"}
    )
    assert stale.status_code == 409 and stale.json()["error"] == "action_not_allowed"

    r = api.post(
        f"{url}/confirm",
        "agent_a",
        json={"action": "agree_return", "expected_status_version": version},
    )
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "wait_return" and r.json()["last_result"]["ok"] is True
    assert [c["aftersale_id"] for c in api.unit.action_port.calls] == ["as_2002"]


def test_aftersale_changed_during_negotiation_needs_refresh(api: Api):
    """协商期间平台上的售后单变了：视图提示先刷新；按新版本直接确认也会被拒（建议可能过期）。"""
    url = f"{A}/aftersales/as_2002"
    view = api.post(f"{url}/review", "agent_a").json()
    assert view["platform_status_version"] == view["platform"]["status_version"]
    assert view["needs_refresh"] is False
    api.post(f"{url}/negotiate", "agent_a", json={"note": "先协商换码"})
    api.unit.clock.advance(minutes=5)
    api.unit.tools.set_aftersale_status(
        SHOP_A, "as_2002", AfterSaleStatus.PENDING_REVIEW, at=api.unit.clock.now()
    )
    view = api.post(f"{url}/negotiation-done", "agent_a", json={"outcome": "买家坚持退货"}).json()
    assert view["state"] == "confirm" and view["needs_refresh"] is True
    latest = view["platform"]["status_version"]
    body = {"action": "agree_return", "expected_status_version": latest}
    assert api.post(f"{url}/confirm", "agent_a", json=body).status_code == 409
    assert api.unit.action_port.calls == []
    view = api.post(f"{url}/review", "agent_a").json()  # 刷新：按新版本重新给建议
    assert view["needs_refresh"] is False and view["platform_status_version"] == latest
    r = api.post(f"{url}/confirm", "agent_a", json=body)
    assert r.status_code == 200 and r.json()["state"] == "wait_return"


def test_aftersale_review_hides_buyer_note_and_ref(api: Api):
    r = api.post(f"{A}/aftersales/as_2003/review", "agent_a")  # R4 签收核查
    assert r.status_code == 200
    assert r.json()["state"] == "verify" and r.json()["platform"] is None
    info = api.unit.tools.get_aftersale(SHOP_A, "as_2003")
    assert info.note and info.note not in r.text and info.buyer_ref not in r.text
    assert "buyer_ref" not in _keys(r.json())


def test_aftersale_errors_map_to_409(api: Api):
    r = api.post(
        f"{A}/aftersales/as_2007/confirm",
        "agent_a",
        json={"action": "agree_refund", "expected_status_version": "1"},
    )
    assert r.status_code == 409  # 还没有审核记录
    r = api.post(f"{A}/aftersales/as_2003/negotiate", "agent_a")
    assert r.status_code == 409
    r = api.post(
        f"{A}/aftersales/as_2002/confirm",
        "agent_a",
        json={"action": "delete", "expected_status_version": "1"},
    )
    assert r.status_code == 422


def test_unknown_or_other_shop_aftersale_is_404(api: Api):
    for aftersale_id in ("nope", "as_9001"):  # as_9001 是 B 店的售后单
        r = api.post(f"{A}/aftersales/{aftersale_id}/review", "agent_a")
        assert r.status_code == 404 and aftersale_id not in r.text


class _DownTools:
    """接口层读平台售后单时平台故障（审核助手持有自己的网关，不受影响）。"""

    def __init__(self, platform):
        self.platform = platform

    def get_aftersale(self, shop_id, aftersale_id):
        raise ToolUnavailable("售后接口限流（模拟）")


class _OtherShopTools:
    """平台接口返回了别的店铺的售后单（不应该发生）：接口层不展示。"""

    def __init__(self, inner):
        self.inner = inner
        self.platform = inner.platform

    def get_aftersale(self, shop_id, aftersale_id):
        info = self.inner.get_aftersale(shop_id, aftersale_id)
        return info.model_copy(update={"shop_id": SHOP_B})


def test_platform_snapshot_is_optional_when_platform_is_down(api: Api, monkeypatch):
    monkeypatch.setattr(api.unit, "tools", _DownTools(api.unit.platform))
    r = api.post(f"{A}/aftersales/as_2001/review", "agent_a")
    assert r.status_code == 200 and r.json()["state"] == "confirm" and r.json()["platform"] is None
    monkeypatch.setattr(api.unit, "tools", _OtherShopTools(api.service.unit.gateway.uncached()))
    r = api.post(f"{A}/aftersales/as_2001/review", "agent_a")
    assert r.status_code == 200 and r.json()["platform"] is None


# ================================================================ 取消订购（§8.4）


def test_revoke_purges_shop_and_blocks_later_requests(api: Api):
    assert _event(api, "什么时候发货", "v1").status_code == 200
    api.post(f"{A}/aftersales/as_2001/review", "agent_a")
    r = api.post(f"{A}/revoke", "merchant_a")
    assert r.status_code == 200, r.text
    counts = r.json()["counts"]
    assert counts["auth"] == 1 and r.json()["total"] == sum(counts.values()) > 5
    assert api.unit.repos.knowledge.get_shop(SHOP_A) is None

    # 之后店铺 A 的一切请求都是 403（§8.4 停止处理）
    r = _event(api, "在吗", "v2")
    assert r.status_code == 403 and r.json()["error"] in {"shop_inactive", "shop_not_authorized"}
    r = api.post(f"{A}/standalone/ask", "agent_a", json={"session_id": "w", "question": "在吗"})
    assert r.status_code == 403 and r.json()["error"] == "shop_inactive"
    assert api.get(f"{A}/reports/adoption", "merchant_a").status_code == 403
    assert api.post(f"{A}/aftersales/as_2001/review", "agent_a").status_code == 403

    # 取消订购可以重试（幂等）
    r = api.post(f"{A}/revoke", "merchant_a")
    assert r.status_code == 200 and r.json()["counts"]["auth"] == 0

    # 别的店铺不受影响
    raw = taobao_event("什么时候发货", msg_id="v3", seller_id=SHOP_B, session_id="conv_b")
    assert api.post(EVENTS_PATH, "integration_b", json=raw).status_code == 200
