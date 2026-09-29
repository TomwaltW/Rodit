"""商家控制台接口（§9.1、§5.2 知识中心、§5.2.3 商品同步、§9.3 店铺配置）。

鉴权、串店、角色的通用检查由 test_api_auth 按 api_routes_console.ROUTES 覆盖；这里测业务行为、
请求体校验、错误码、路径冲突和串店的业务 ID（别的店铺的商品、FAQ）。全部合成数据，不联网。
"""

from __future__ import annotations

import pytest
from api_support import Api, shop_snapshot

from rodit.core.audit import AuditAction
from rodit.core.enums import KnowledgeSource, KnowledgeStatus, PolicyKey
from rodit.core.models import ItemCard, KnowledgeItem, Platform
from rodit.testing.fixtures import SHOP_A, SHOP_B

C = f"/v1/shops/{SHOP_A}/console"
MERCHANT = "owner_a"  # api_support.CALLERS["merchant_a"]
QUESTION = "发票抬头可以写公司吗"


def _faq_id(api: Api, question: str, shop_id: str = SHOP_A) -> int:
    return next(
        k.id
        for k in api.unit.repos.knowledge.list_knowledge(shop_id, status=None)
        if k.question == question
    )


def _add_b_only_item(api: Api) -> None:
    api.unit.repos.knowledge.upsert_item_card(
        ItemCard(
            shop_id=SHOP_B,
            platform=Platform.TAOBAO,
            item_id="i_b_only",
            title="男士衬衫（只在店铺 B）",
            updated_at=api.unit.clock.now(),
        )
    )


# ================================================================ FAQ


def test_list_faqs_with_filters(api: Api):
    r = api.get(f"{C}/faqs", "manager_a")
    assert r.status_code == 200
    assert {f["status"] for f in r.json()} == {"approved", "candidate", "retired"}
    r = api.get(f"{C}/faqs", "merchant_a", params={"status": "candidate"})
    assert [f["question"] for f in r.json()] == ["可以寄到新疆吗"]
    r = api.get(f"{C}/faqs", "merchant_a", params={"item_id": "i_555"})
    assert [f["question"] for f in r.json()] == ["这款会透吗"]
    assert api.get(f"{C}/faqs", "merchant_a", params={"status": "nope"}).status_code == 422


def test_create_faq_then_ask_uses_it(api: Api):
    r = api.post(
        f"{C}/faqs",
        "merchant_a",
        json={"question": QUESTION, "answer": "亲，可以开公司抬头的电子普通发票哦~"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    new_id = body["knowledge"]["id"]
    assert body["knowledge"]["status"] == "approved" and body["knowledge"]["source"] == "merchant"
    assert body["knowledge"]["approved_by"] == MERCHANT
    assert body["retired"] is None and body["warnings"] == []
    r = api.post(
        f"/v1/shops/{SHOP_A}/standalone/ask",
        "agent_a",
        json={"session_id": "w1", "question": QUESTION},
    )
    assert r.status_code == 200
    ids = [e["id"] for card in r.json()["suggestions"] for e in card["evidence"]]
    assert ids == [f"faq:{new_id}"]


def test_create_faq_returns_quality_warnings(api: Api):
    r = api.post(
        f"{C}/faqs",
        "merchant_a",
        json={"question": "能便宜点吗", "answer": "亲，加我微信给你全网最低价~"},
    )
    assert r.status_code == 200
    rules = {w["rule"]: w["blocks_answer"] for w in r.json()["warnings"]}
    assert rules["off_platform"] is True and rules["banned_word"] is True


def test_edit_faq_retires_the_old_id(api: Api):
    old_id = _faq_id(api, "有运费险吗")
    r = api.post(f"{C}/faqs/{old_id}/edit", "merchant_a", json={"answer": "亲，每单都送运费险哦~"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["retired"]["id"] == old_id and body["retired"]["status"] == "retired"
    assert body["retired"]["answer"] == "亲，每个订单都送运费险的哦~"  # 旧内容原样
    new = body["knowledge"]
    assert new["id"] != old_id and new["answer"] == "亲，每单都送运费险哦~"
    assert new["intent"] == "invoice_gift"  # 没提交的字段保持原样
    # 再改旧 id：已经下线
    r = api.post(f"{C}/faqs/{old_id}/edit", "merchant_a", json={"answer": "x"})
    assert r.status_code == 409 and r.json()["error"] == "action_not_allowed"
    [event] = api.unit.repos.audit.list_events(SHOP_A, action=AuditAction.KNOWLEDGE_EDITED.value)
    assert event.actor == MERCHANT and event.detail["old_knowledge_id"] == old_id


def test_edit_faq_validation(api: Api):
    kid = _faq_id(api, "有运费险吗")
    for body in ({}, {"answer": ""}, {"answer": None}, {"question": "  "}, {"intent": "nope"}):
        r = api.post(f"{C}/faqs/{kid}/edit", "merchant_a", json=body)
        assert r.status_code == 422, body
    same = api.unit.repos.knowledge.get_knowledge(SHOP_A, kid)
    r = api.post(f"{C}/faqs/{kid}/edit", "merchant_a", json={"answer": same.answer})
    assert r.status_code == 422 and "没有变化" in r.json()["message"]
    assert api.post(f"{C}/faqs/99999/edit", "merchant_a", json={"answer": "x"}).status_code == 404
    assert api.unit.repos.knowledge.get_knowledge(SHOP_A, kid).status == KnowledgeStatus.APPROVED


def test_approve_edited(api: Api):
    kid = _faq_id(api, "可以寄到新疆吗")
    r = api.post(
        f"{C}/candidates/{kid}/approve-edited",
        "merchant_a",
        json={"question": "新疆能发货吗", "answer": "亲，新疆可以发，需补运费 15 元哦~"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["retired"]["id"] == kid
    assert body["knowledge"]["status"] == "approved" and body["knowledge"]["source"] == "adopted"
    assert body["knowledge"]["approved_by"] == MERCHANT
    # 候选已经处理过：不能再审
    r = api.post(f"/v1/shops/{SHOP_A}/knowledge/candidates/{kid}/approve", "merchant_a")
    assert r.status_code == 409


def test_approve_edited_without_changes_is_plain_approval(api: Api):
    kid = _faq_id(api, "可以寄到新疆吗")
    candidate = api.unit.repos.knowledge.get_knowledge(SHOP_A, kid)
    r = api.post(
        f"{C}/candidates/{kid}/approve-edited", "merchant_a", json={"answer": candidate.answer}
    )
    assert r.status_code == 200
    assert r.json()["knowledge"]["id"] == kid and r.json()["retired"] is None
    approved = _faq_id(api, "发什么快递")
    r = api.post(f"{C}/candidates/{approved}/approve-edited", "merchant_a", json={"answer": "x"})
    assert r.status_code == 409


# ================================================================ 请求体里的身份字段、串店的业务 ID


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/faqs", {"question": "在吗", "answer": "亲，在的", "approved_by": "someone"}),
        ("/faqs", {"question": "在吗", "answer": "亲，在的", "shop_id": SHOP_B}),
        ("/faqs", {"question": "在吗", "answer": "亲，在的", "source": "merchant"}),
        ("/faqs/1/edit", {"answer": "亲，在的", "merchant": "someone"}),
        ("/candidates/6/approve-edited", {"answer": "亲，在的", "approved_by": "x"}),
        ("/policies/invoice", {"value": "可以开", "operator": "someone"}),
        ("/items/i_555", {"care": ["手洗"], "merchant": "someone"}),
        ("/items/i_555", {"title": "改标题"}),  # 平台字段不能改
        ("/items/i_555", {"updated_at": "2026-01-01T00:00:00+08:00"}),
        ("/items/sync", {"item_id": "i_555", "operator": "someone"}),
        ("/config", {"changes": {"tone": "专业"}, "merchant": "someone"}),
    ],
)
def test_identity_and_platform_fields_in_body_are_rejected(api: Api, path: str, body: dict):
    before = shop_snapshot(api, SHOP_A)
    r = api.post(f"{C}{path}", "merchant_a", json=body)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert "Extra inputs are not permitted" in r.json()["message"]
    assert shop_snapshot(api, SHOP_A) == before


def test_items_and_faqs_of_another_shop_cannot_be_used(api: Api):
    _add_b_only_item(api)
    b_faq = _faq_id(api, "发什么快递", SHOP_B)
    b_candidate = api.unit.repos.knowledge.add_knowledge(
        KnowledgeItem(
            shop_id=SHOP_B,
            question="能便宜吗",
            answer="您好，不能",
            source=KnowledgeSource.ADOPTED,
            status=KnowledgeStatus.CANDIDATE,
        )
    )
    before_a = shop_snapshot(api, SHOP_A)
    before_b = shop_snapshot(api, SHOP_B)
    r = api.post(
        f"{C}/faqs",
        "merchant_a",
        json={"question": "会缩水吗", "answer": "亲，不会", "item_id": "i_b_only"},
    )
    assert r.status_code == 422 and "本店没有这个商品" in r.json()["message"]
    assert SHOP_B not in r.json()["message"]
    cases = [
        ("POST", f"{C}/faqs/{b_faq}/edit", {"answer": "亲，改了"}),
        ("POST", f"{C}/candidates/{b_candidate.id}/approve-edited", {"answer": "亲，可以"}),
        ("GET", f"{C}/items/i_b_only", None),
        ("POST", f"{C}/items/i_b_only", {"care": ["手洗"]}),
        ("POST", f"{C}/items/sync", {"item_id": "i_b_only"}),
    ]
    for method, url, body in cases:
        r = api.call(method, url, "merchant_a", json=body)
        assert r.status_code == 404 and r.json()["error"] == "not_found", (url, r.text)
    assert shop_snapshot(api, SHOP_B) == before_b
    assert shop_snapshot(api, SHOP_A) == before_a


# ================================================================ 店铺政策


def test_policies_list_and_update(api: Api):
    r = api.get(f"{C}/policies", "manager_a")
    assert r.status_code == 200
    rows = r.json()
    assert [row["key"] for row in rows] == [k.value for k in PolicyKey]
    assert next(row for row in rows if row["key"] == "warranty")["value"] is None
    r = api.post(f"{C}/policies/warranty", "merchant_a", json={"value": "整机保修一年"})
    assert r.status_code == 200
    assert r.json()["value"] == "整机保修一年" and r.json()["label"] == "保修"
    assert api.unit.repos.knowledge.get_policies(SHOP_A)["warranty"].value == "整机保修一年"
    assert api.post(f"{C}/policies/nope", "merchant_a", json={"value": "x"}).status_code == 422
    assert api.post(f"{C}/policies/warranty", "merchant_a", json={"value": ""}).status_code == 422
    assert api.post(f"{C}/policies/warranty", "merchant_a", json={"value": " "}).status_code == 422


def test_policy_change_is_effective_for_the_next_answer(api: Api):
    r = api.post(f"{C}/policies/ship_time", "merchant_a", json={"value": "付款后 72 小时内发货"})
    assert r.status_code == 200
    r = api.post(
        f"/v1/shops/{SHOP_A}/standalone/ask",
        "agent_a",
        json={"session_id": "w1", "question": "什么时候发货"},
    )
    assert "72" in r.json()["suggestions"][0]["text"]


# ================================================================ 商品卡片与同步


def test_item_list_detail_and_update(api: Api):
    r = api.get(f"{C}/items", "manager_a")
    assert r.status_code == 200
    first = r.json()[0]
    assert first["item_id"] == "i_555"
    assert first["filled"] == ["size_advice", "care", "forbidden_claims"]
    r = api.get(f"{C}/items/i_777", "merchant_a")
    assert r.status_code == 200
    card = r.json()
    assert card["size_advice"] is None and card["merchant_fields"] == [
        "size_advice",
        "care",
        "usage",
        "forbidden_claims",
    ]
    r = api.post(
        f"{C}/items/i_777",
        "merchant_a",
        json={"size_advice": "偏小一码", "usage": ["机洗前翻面"], "forbidden_claims": None},
    )
    assert r.status_code == 200, r.text
    card = r.json()
    assert card["size_advice"] == "偏小一码" and card["usage"] == ["机洗前翻面"]
    assert card["changed"] == ["size_advice", "usage"]
    assert card["title"] == "高腰阔腿牛仔裤"
    assert api.post(f"{C}/items/i_777", "merchant_a", json={}).status_code == 422
    assert api.get(f"{C}/items/i_nope", "merchant_a").status_code == 404
    [event] = api.unit.repos.audit.list_events(SHOP_A, action=AuditAction.ITEM_CARD_UPDATED.value)
    assert event.actor == MERCHANT and event.detail == {
        "item_id": "i_777",
        "fields": ["size_advice", "usage"],
    }


def test_sync_and_item_update_paths_do_not_collide(api: Api):
    """POST /items/sync 是同步，POST /items/{item_id} 是改卡片；两个都能用。"""
    r = api.post(f"{C}/items/sync", "merchant_a")
    assert r.status_code == 200, r.text
    report = r.json()
    assert report["scope"] == "shop" and report["created"] == ["i_999"]
    assert report["updated"] == ["i_555"] and report["off_sale"] == ["i_888"]
    assert report["ocr_failures"] == 1 and report["counts"]["created"] == 1
    r = api.post(f"{C}/items/i_999", "merchant_a", json={"size_advice": "均码"})
    assert r.status_code == 200 and r.json()["item_id"] == "i_999"
    r = api.post(f"{C}/items/sync", "merchant_a", json={"item_id": "i_999"})
    assert r.status_code == 200
    assert r.json()["scope"] == "item" and r.json()["unchanged"] == ["i_999"]
    # 同步保留商家刚填的尺码建议
    assert api.unit.repos.knowledge.get_item_card(SHOP_A, "i_999").size_advice == "均码"
    # GET /items/sync 不是同步：当作商品 ID，没有这个商品
    assert api.get(f"{C}/items/sync", "merchant_a").status_code == 404


def test_resync_is_idempotent_over_http(api: Api):
    assert api.post(f"{C}/items/sync", "merchant_a").status_code == 200
    items_before = shop_snapshot(api, SHOP_A)["items"]
    r = api.post(f"{C}/items/sync", "merchant_a", json={})
    assert r.status_code == 200
    assert r.json()["created"] == [] and r.json()["updated"] == []
    assert shop_snapshot(api, SHOP_A)["items"] == items_before


def test_sync_unavailable_is_503(api: Api):
    api.unit.catalog = None
    r = api.post(f"{C}/items/sync", "merchant_a")
    assert r.status_code == 503 and r.json()["error"] == "sync_unavailable"


def test_sync_catalog_failure_is_503(api: Api):
    before = shop_snapshot(api, SHOP_A)
    api.unit.catalog.fail_on.add("list_items")
    r = api.post(f"{C}/items/sync", "merchant_a")
    assert r.status_code == 503 and r.json()["error"] == "tool_unavailable"
    assert shop_snapshot(api, SHOP_A) == before


# ================================================================ 店铺配置（§9.3）


def test_config_get_and_update(api: Api):
    r = api.get(f"{C}/config", "manager_a")
    assert r.status_code == 200
    body = r.json()
    assert body["effective"]["tone"] == "亲切" and body["locked"] == [
        "shadow_mode",
        "auto_reply_enabled",
    ]
    assert "banned_words" in body["overrides"]
    r = api.post(
        f"{C}/config",
        "merchant_a",
        json={"changes": {"tone": "专业", "banned_words": None, "aftersale_amount_limit": "800"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] == ["aftersale_amount_limit", "tone"]
    assert body["removed"] == ["banned_words"]
    assert body["effective"]["banned_words"] == []  # 回到默认
    assert "banned_words" not in body["overrides"]
    assert body["effective"]["aftersale_amount_limit"] == "800"
    shop = api.unit.repos.knowledge.get_shop(SHOP_A)
    assert shop.config_overrides["tone"] == "专业"


@pytest.mark.parametrize("key", ["shadow_mode", "auto_reply_enabled"])
def test_rollout_switches_are_409(api: Api, key: str):
    before = shop_snapshot(api, SHOP_A)
    r = api.post(f"{C}/config", "merchant_a", json={"changes": {key: False}})
    assert r.status_code == 409 and r.json()["error"] == "action_not_allowed"
    assert "上线节奏" in r.json()["message"]
    assert shop_snapshot(api, SHOP_A) == before


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"nope": 1},
        {"max_answer_chars": 5},
        {"tone": "热情"},
        {"suggestion_count": 0},
        {"manager_ids": ["bot"]},
        {"aftersale_amount_limit": "-5"},
    ],
)
def test_invalid_config_is_422(api: Api, changes: dict):
    before = shop_snapshot(api, SHOP_A)
    r = api.post(f"{C}/config", "merchant_a", json={"changes": changes})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert shop_snapshot(api, SHOP_A) == before


def test_manager_can_read_but_not_write(api: Api):
    assert api.get(f"{C}/config", "manager_a").status_code == 200
    r = api.post(f"{C}/config", "manager_a", json={"changes": {"tone": "专业"}})
    assert r.status_code == 403 and r.json()["error"] == "forbidden_role"


def test_console_after_revoke_is_rejected(api: Api):
    assert api.post(f"/v1/shops/{SHOP_A}/revoke", "merchant_a").status_code == 200
    r = api.post(f"{C}/policies/invoice", "merchant_a", json={"value": "可以开"})
    assert r.status_code == 403 and r.json()["error"] == "shop_inactive"
    assert api.unit.repos.knowledge.get_shop(SHOP_A) is None
    assert api.unit.repos.knowledge.get_policies(SHOP_A) == {}
