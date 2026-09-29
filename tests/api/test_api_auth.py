"""鉴权与租户隔离（§9.2 串店测试在 HTTP 层、§8.4、§7.3.2 店长确认、§4.1 原则 3 人在回路）。"""

from __future__ import annotations

import re

import pytest
from api_support import (
    CALLERS,
    EVENTS_PATH,
    SHOP_A_CALLERS,
    SHOP_B_CALLERS,
    SHOP_ROUTES,
    TOKENS,
    Api,
    Route,
    TestClient,
    auth,
    shop_snapshot,
)
from pydantic import ValidationError

from rodit.api import Principal, StaticTokenAuthenticator, bearer_token, create_app
from rodit.app.demo import taobao_event
from rodit.core.audit import AuditAction
from rodit.core.enums import SuggestionStatus
from rodit.testing.fixtures import AGENT, MANAGER, SHOP_A, SHOP_B

ROUTE_IDS = [r.name for r in SHOP_ROUTES]


def test_every_shop_route_is_listed(api: Api):
    """应用里每个店铺范围的路由都在 SHOP_ROUTES 里（下面的串店、角色、401 测试才覆盖得到）。

    路由表从 OpenAPI 文档读（不依赖 FastAPI 内部的路由包装），路径模板转成正则去匹配清单里的 URL。
    """
    listed = {(r.method, r.url(SHOP_A)) for r in SHOP_ROUTES}
    paths = api.client.app.openapi()["paths"]
    missing = []
    for template, operations in paths.items():
        if not template.startswith("/v1/shops/{shop_id}"):
            continue
        pattern = re.compile("^" + re.sub(r"\{[^/]+\}", "[^/]+", template) + "$")
        for method in operations:
            m = method.upper()
            if not any(lm == m and pattern.match(url) for lm, url in listed):
                missing.append(f"{m} {template}")
    assert len(paths) > 10
    assert missing == [], f"这些路由没有列进 tests/api 的路由清单：{missing}"


# ================================================================ 身份与令牌


def test_principal_requires_a_real_person_for_human_roles():
    with pytest.raises(ValidationError):
        Principal(shop_id=SHOP_A, user_id="system", role="agent")
    with pytest.raises(ValidationError):
        Principal(shop_id=SHOP_A, user_id=" Bot ", role="manager")
    with pytest.raises(ValidationError):
        Principal(shop_id=SHOP_A, user_id="kf", role="admin")  # 不认识的角色
    with pytest.raises(ValidationError):
        Principal(shop_id="", user_id="kf", role="agent")
    # 平台对接不是人，可以用系统账号名
    assert Principal(shop_id=SHOP_A, user_id="system", role="integration").role == "integration"


def test_static_authenticator_matches_exact_tokens_and_never_shows_them():
    token = "dev-token-0123456789abcdef"
    p = CALLERS["agent_a"]
    authn = StaticTokenAuthenticator({token: p})
    assert authn.authenticate(token) == p
    assert authn.authenticate(token + "x") is None
    assert authn.authenticate(token[:-1]) is None
    assert authn.authenticate("") is None
    assert authn.authenticate("令牌") is None
    assert token not in repr(authn)
    with pytest.raises(ValueError):
        StaticTokenAuthenticator({"short": p})  # 太短的令牌容易猜
    with pytest.raises(TypeError):
        StaticTokenAuthenticator({token: {"shop_id": SHOP_A}})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("", None),
        ("Bearer", None),
        ("Bearer   ", None),
        ("Basic abcdef", None),
        ("bearer abc", "abc"),
        ("Bearer  abc ", "abc"),
    ],
)
def test_bearer_token_parsing(header, expected):
    assert bearer_token(header) == expected


def test_healthz_needs_no_auth(api: Api):
    r = api.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


# ================================================================ 401：每个路由


@pytest.mark.parametrize("route", SHOP_ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer not-a-known-token-0000000000"},
        {"Authorization": f"Basic {TOKENS['agent_a']}"},
        {"Authorization": TOKENS["agent_a"]},  # 缺 Bearer
    ],
    ids=["missing", "unknown", "basic", "no-scheme"],
)
def test_every_shop_route_rejects_missing_or_unknown_token(api: Api, route: Route, headers):
    before = shop_snapshot(api, SHOP_A)
    r = api.call(route.method, route.url(SHOP_A), headers=headers, json=route.body)
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"
    assert r.headers["www-authenticate"] == "Bearer"
    assert TOKENS["agent_a"] not in r.text  # 不回显令牌
    assert shop_snapshot(api, SHOP_A) == before


def test_events_endpoint_rejects_missing_or_unknown_token(api: Api):
    raw = taobao_event("什么时候发货", msg_id="t401")
    for headers in ({}, {"Authorization": "Bearer nope-nope-nope-nope-nope"}):
        r = api.post(EVENTS_PATH, headers=headers, json=raw)
        assert r.status_code == 401 and r.json()["error"] == "unauthorized"
    assert api.unit.repos.suggestions.list(SHOP_A) == []


# ================================================================ 403：串店（§9.2）
# 每个路由 × 店铺 B 的每种角色


def _prepare_shop_a(api: Api) -> None:
    """让店铺 A 有推荐、候选知识、待确认的售后审核、会话上下文，越权请求有真实目标可打。"""
    r = api.post(EVENTS_PATH, "integration_a", json=taobao_event("新疆能到吗", msg_id="p1"))
    assert r.status_code == 200
    card = r.json()["suggestions"][0]
    r = api.post(
        f"/v1/shops/{SHOP_A}/suggestions/{card['suggestion_id']}/edited",
        "agent_a",
        json={"edited_text": "亲，新疆可以发，补 15 元运费哦~"},
    )
    assert r.status_code == 200 and r.json()["candidate_knowledge_id"] is not None
    r = api.post(EVENTS_PATH, "integration_a", json=taobao_event("能开发票吗", msg_id="p2"))
    assert r.status_code == 200 and r.json()["suggestions"]
    for aftersale_id in ("as_2001", "as_2005"):
        r = api.post(f"/v1/shops/{SHOP_A}/aftersales/{aftersale_id}/review", "agent_a")
        assert r.status_code == 200 and r.json()["state"] == "confirm"


@pytest.mark.parametrize("route", SHOP_ROUTES, ids=ROUTE_IDS)
def test_every_shop_route_rejects_every_role_of_another_shop(api: Api, route: Route):
    """§9.2 串店测试：用店铺 B 的身份（四种角色）访问店铺 A 的每个接口，一律 403，
    且店铺 A 什么都没变。"""
    _prepare_shop_a(api)
    before = shop_snapshot(api, SHOP_A, ("conv_demo",))
    for caller in SHOP_B_CALLERS:
        r = api.call(route.method, route.url(SHOP_A), caller, json=route.body)
        assert r.status_code == 403, (caller, r.text)
        assert r.json() == {"error": "tenant_isolation", "message": r.json()["message"]}
        assert SHOP_A not in r.json()["message"]  # 不透露目标店铺的任何信息
    assert shop_snapshot(api, SHOP_A, ("conv_demo",)) == before


@pytest.mark.parametrize("route", SHOP_ROUTES, ids=ROUTE_IDS)
def test_cross_shop_is_rejected_even_for_unknown_shops(api: Api, route: Route):
    """路径里的店铺不存在时也是 403（不是 404），不能用来探测别的店铺是否存在。"""
    r = api.call(route.method, route.url("s_9999"), "agent_a", json=route.body)
    assert r.status_code == 403 and r.json()["error"] == "tenant_isolation"


def test_cross_shop_platform_event_is_rejected_before_the_adapter_sees_it(api: Api):
    """平台对接身份属于店铺 B，推送店铺 A 的事件：403，且适配器里不留痕迹（去重、上下文、撤回）。"""
    raw = taobao_event("什么时候发货", msg_id="x1", session_id="conv_x")
    r = api.post(EVENTS_PATH, "integration_b", json=raw)
    assert r.status_code == 403 and r.json()["error"] == "tenant_isolation"
    adapter = api.unit.adapters.get(api.unit.platform, "B")
    assert adapter.get_context(SHOP_A, "conv_x").recent_messages == []
    assert api.unit.repos.suggestions.list(SHOP_A) == []
    # 同一个 msg_id 没有被店铺 B 的请求"占用"：店铺 A 自己推送照常处理
    r = api.post(EVENTS_PATH, "integration_a", json=raw)
    assert r.status_code == 200 and r.json()["suggestions"]
    assert [m.msg_id for m in adapter.get_context(SHOP_A, "conv_x").recent_messages] == ["x1"]

    # 店铺 B 也不能替店铺 A 撤回消息、伪造客服回声
    recall = {**raw, "msg_id": "x2", "msg_type": "recall", "recall_msg_id": "x1"}
    echo = {**raw, "msg_id": "x3", "direction": "out", "content": "伪造的客服回复"}
    for evil in (recall, echo):
        assert api.post(EVENTS_PATH, "integration_b", json=evil).status_code == 403
    assert [m.msg_id for m in adapter.get_context(SHOP_A, "conv_x").recent_messages] == ["x1"]


def test_integration_principal_can_push_its_own_shop(api: Api):
    raw = taobao_event("什么时候发货", msg_id="b1", seller_id=SHOP_B, session_id="conv_b")
    r = api.post(EVENTS_PATH, "integration_b", json=raw)
    assert r.status_code == 200
    assert r.json()["shop_id"] == SHOP_B
    assert api.unit.repos.suggestions.list(SHOP_A) == []


# ================================================================ 403：角色


@pytest.mark.parametrize("route", SHOP_ROUTES, ids=ROUTE_IDS)
def test_roles_outside_the_route_are_rejected(api: Api, route: Route):
    before = shop_snapshot(api, SHOP_A)
    for caller in SHOP_A_CALLERS:
        if CALLERS[caller].role in route.roles:
            continue
        r = api.call(route.method, route.url(SHOP_A), caller, json=route.body)
        assert r.status_code == 403, (caller, r.text)
        assert r.json()["error"] == "forbidden_role"
    assert shop_snapshot(api, SHOP_A) == before


def test_events_endpoint_is_for_platform_integration_only(api: Api):
    raw = taobao_event("什么时候发货", msg_id="r1")
    for caller in ("agent_a", "manager_a", "merchant_a"):
        r = api.post(EVENTS_PATH, caller, json=raw)
        assert r.status_code == 403 and r.json()["error"] == "forbidden_role"
    assert api.unit.repos.suggestions.list(SHOP_A) == []


# ================================================================ 操作人、by_manager
# 取自身份，不取请求体


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/aftersales/as_2005/confirm",
            {"action": "agree_return", "expected_status_version": "v", "by_manager": True},
        ),
        (
            "/aftersales/as_2005/confirm",
            {"action": "agree_return", "expected_status_version": "v", "operator": MANAGER},
        ),
        ("/aftersales/as_2005/negotiate", {"note": "x", "operator": MANAGER}),
        ("/aftersales/as_2005/negotiation-done", {"operator": MANAGER}),
        ("/aftersales/as_2005/review", {"operator": MANAGER}),
        ("/suggestions/1/sent", {"agent_id": "kf_other"}),
        ("/suggestions/1/edited", {"edited_text": "改", "agent_id": "kf_other"}),
        ("/suggestions/1/rejected", {"reason": "other", "agent_id": "kf_other"}),
        ("/standalone/ask", {"session_id": "w", "question": "在吗", "agent_id": "kf_other"}),
        ("/standalone/ask", {"session_id": "w", "question": "在吗", "shop_id": SHOP_B}),
    ],
)
def test_identity_fields_in_body_are_rejected(api: Api, path: str, body: dict):
    before = shop_snapshot(api, SHOP_A)
    r = api.post(f"/v1/shops/{SHOP_A}{path}", "agent_a", json=body)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert "Extra inputs are not permitted" in r.json()["message"]
    assert shop_snapshot(api, SHOP_A) == before


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/knowledge/candidates/1/approve", {"merchant": "someone"}),
        ("/knowledge/1/retire", {"merchant": "someone"}),
        ("/revoke", {"shop_id": SHOP_B}),
    ],
)
def test_merchant_identity_fields_in_body_are_rejected(api: Api, path: str, body: dict):
    r = api.post(f"/v1/shops/{SHOP_A}{path}", "merchant_a", json=body)
    assert r.status_code == 422
    assert api.unit.repos.auth.get(SHOP_A) is not None  # 取消订购没有执行


def _status_version(api: Api, aftersale_id: str) -> str:
    r = api.post(f"/v1/shops/{SHOP_A}/aftersales/{aftersale_id}/review", "agent_a")
    assert r.status_code == 200
    return r.json()["platform"]["status_version"]


def test_r6_confirm_requires_manager_role(api: Api):
    """§7.3.2 R6：金额超过上限必须店长确认。by_manager 只由店长角色推出。"""
    version = _status_version(api, "as_2005")
    body = {"action": "agree_return", "expected_status_version": version}
    url = f"/v1/shops/{SHOP_A}/aftersales/as_2005/confirm"
    for caller in ("agent_a", "merchant_a"):
        r = api.post(url, caller, json=body)
        assert r.status_code == 403 and r.json()["error"] == "manager_required"
    assert api.unit.action_port.calls == []

    r = api.post(url, "manager_a", json=body)
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["state"] == "wait_return" and view["confirmed_by"] == MANAGER
    assert view["last_result"]["ok"] is True
    [call] = api.unit.action_port.calls
    assert call["operator"] == MANAGER
    [confirm] = api.unit.repos.audit.list_events(SHOP_A, action=AuditAction.REVIEW_CONFIRM)
    assert confirm.actor == MANAGER and confirm.detail["by_manager"] is True


def test_manager_role_alone_is_not_enough_for_r6(api: Api):
    """店长角色但不在店铺的 manager_ids 里：审核助手照样拒绝（409），不调用平台。"""
    version = _status_version(api, "as_2005")
    r = api.post(
        f"/v1/shops/{SHOP_A}/aftersales/as_2005/confirm",
        "fake_manager_a",
        json={"action": "agree_return", "expected_status_version": version},
    )
    assert r.status_code == 409 and r.json()["error"] == "action_not_allowed"
    assert "店长" in r.json()["message"]
    assert api.unit.action_port.calls == []


def test_agent_confirm_records_the_agent_as_operator_without_manager_flag(api: Api):
    version = _status_version(api, "as_2001")  # R2：未发货仅退款，不需要店长
    r = api.post(
        f"/v1/shops/{SHOP_A}/aftersales/as_2001/confirm",
        "agent_a",
        json={"action": "agree_refund", "expected_status_version": version},
    )
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "agreed" and r.json()["confirmed_by"] == AGENT
    [confirm] = api.unit.repos.audit.list_events(SHOP_A, action=AuditAction.REVIEW_CONFIRM)
    assert confirm.actor == AGENT and confirm.detail["by_manager"] is False


def test_feedback_agent_comes_from_principal(api: Api):
    r = api.post(EVENTS_PATH, "integration_a", json=taobao_event("什么时候发货", msg_id="f1"))
    sid = r.json()["suggestions"][0]["suggestion_id"]
    r = api.post(f"/v1/shops/{SHOP_A}/suggestions/{sid}/sent", "manager_a")
    assert r.status_code == 200
    record = api.unit.repos.suggestions.get(SHOP_A, sid)
    assert record.status == SuggestionStatus.SENT and record.agent_id == MANAGER
    [event] = api.unit.repos.audit.list_events(SHOP_A, action=AuditAction.FEEDBACK_SENT)
    assert event.actor == MANAGER


def test_tokens_of_one_app_do_not_work_on_another(api: Api):
    """每个应用有自己的 Authenticator（没有全局状态）。"""
    other_token = "another-app-token-000000"
    other = TestClient(
        create_app(
            api.service, authenticator=StaticTokenAuthenticator({other_token: CALLERS["manager_a"]})
        )
    )
    r = other.get(f"/v1/shops/{SHOP_A}/reports/qa", headers=auth("manager_a"))
    assert r.status_code == 401
    r = other.get(
        f"/v1/shops/{SHOP_A}/reports/qa", headers={"Authorization": f"Bearer {other_token}"}
    )
    assert r.status_code == 200
    assert (
        api.get(
            f"/v1/shops/{SHOP_A}/reports/qa", headers={"Authorization": f"Bearer {other_token}"}
        ).status_code
        == 401
    )


def test_create_app_checks_its_arguments(api: Api):
    with pytest.raises(TypeError):
        create_app(object(), authenticator=StaticTokenAuthenticator({}))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        create_app(api.service, authenticator=object())  # type: ignore[arg-type]
