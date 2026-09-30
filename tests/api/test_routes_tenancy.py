"""平台订购推送接口 POST /v1/platform/{access}/subscription（方案 §8.4、§9.2 串店）。

只给平台对接（integration）用、只能是自己的店铺；解析不出来 204；开通 / 续订写店铺和授权；
取消订购删除该店铺在本单元的全部数据并留下系统事件 shop.purged。全部是合成数据和合成令牌。
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from api_support import EVENTS_PATH, TestClient

from rodit.api import Principal, StaticTokenAuthenticator, create_app
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.core.audit import AuditAction
from rodit.core.enums import Platform
from rodit.core.scopes import MVP_SCOPES, SCOPE_REFUND_WRITE
from rodit.testing.fakes import FakeSubscriptionParser, parse_subscription, subscription_push
from rodit.testing.fixtures import SHOP_A, SHOP_B

NEW = "s_new"
PATH = "/v1/platform/B/subscription"
GRANT_TOKEN = "tok-http-grant-synthetic-3e8b"

CALLERS: dict[str, Principal] = {
    "push_a": Principal(shop_id=SHOP_A, user_id="tb_push_a", role="integration"),
    "push_b": Principal(shop_id=SHOP_B, user_id="tb_push_b", role="integration"),
    "push_new": Principal(shop_id=NEW, user_id="tb_push_new", role="integration"),
    "agent_a": Principal(shop_id=SHOP_A, user_id="kf_02", role="agent"),
    "merchant_a": Principal(shop_id=SHOP_A, user_id="owner_a", role="merchant"),
}
TOKENS = {name: f"test-token-tenancy-{name}-0123456789" for name in CALLERS}


class Api:
    def __init__(self, service: AssistService) -> None:
        authenticator = StaticTokenAuthenticator({TOKENS[n]: p for n, p in CALLERS.items()})
        self.service = service
        self.unit = service.unit
        self.client = TestClient(create_app(service, authenticator=authenticator))

    def post(self, path: str, who: str | None, **kw):
        headers = {"Authorization": f"Bearer {TOKENS[who]}"} if who else {}
        return self.client.post(path, headers=headers, **kw)


@pytest.fixture
def api() -> Api:
    unit = build_demo_unit(Platform.TAOBAO)
    unit.parse_subscription = FakeSubscriptionParser(Platform.TAOBAO)  # 平台云代码注入的解析
    return Api(AssistService(unit))


def _push(api: Api, kind: str, shop_id: str, **kw) -> dict:
    kw.setdefault("at", api.unit.clock.now())
    if kind in ("subscribed", "renewed"):
        kw.setdefault("access_token", GRANT_TOKEN)
    return subscription_push(kind, shop_id, **kw)


def _purged(api: Api) -> list:
    return api.unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)


def test_subscribed_push_onboards_the_callers_shop(api: Api):
    raw = _push(api, "subscribed", NEW, category="女装", buyer_nick="合成昵称")
    r = api.post(PATH, "push_new", json=raw)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "shop_id": NEW,
        "kind": "subscribed",
        "action": "granted",
        "status": "trial",
        "counts": None,
        "total": None,
    }
    assert GRANT_TOKEN not in r.text
    shop = api.unit.repos.knowledge.get_shop(NEW)
    assert shop is not None and shop.category == "女装"
    auth = api.unit.repos.auth.get(NEW)
    assert auth.access_token.get_secret_value() == GRANT_TOKEN
    assert set(auth.scopes) == MVP_SCOPES
    [granted] = api.unit.repos.audit.list_events(NEW, action=AuditAction.AUTH_GRANTED)
    assert granted.actor == "platform"


def test_renewed_push_replaces_the_token(api: Api):
    later = api.unit.clock.now() + timedelta(days=90)
    raw = _push(api, "renewed", SHOP_A, access_token="tok-renewed-http", expires_at=later)
    r = api.post(PATH, "push_a", json=raw)
    assert r.status_code == 200 and r.json()["action"] == "granted"
    auth = api.unit.repos.auth.get(SHOP_A)
    assert auth.access_token.get_secret_value() == "tok-renewed-http" and auth.expires_at == later


def test_cancelled_push_revokes_and_platform_events_stop(api: Api):
    first = api.post(EVENTS_PATH, "push_a", json=taobao_event("能开发票吗", msg_id="h1"))
    assert first.status_code == 200, first.text
    cancel = _push(api, "cancelled", SHOP_A)
    r = api.post(PATH, "push_a", json=cancel)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "revoked" and body["kind"] == "cancelled" and body["status"] is None
    assert body["counts"]["auth"] == 1 and body["total"] == sum(body["counts"].values()) > 1
    assert api.unit.repos.auth.get(SHOP_A) is None
    assert api.unit.repos.knowledge.get_shop(SHOP_A) is None
    [record] = _purged(api)
    assert record.detail == {
        "shop_id": SHOP_A,
        "reason": "cancelled",
        "counts": body["counts"],
        "event_at": cancel["at"],
    }
    # 之后这家店的平台事件一律 403
    after = api.post(EVENTS_PATH, "push_a", json=taobao_event("在吗", msg_id="h2"))
    assert after.status_code == 403
    # 重推同一条取消：已经处理过的旧事件，什么都不做（200，平台不再重推）
    again = api.post(PATH, "push_a", json=cancel)
    assert again.status_code == 200 and again.json()["action"] == "ignored"
    assert again.json()["total"] is None and len(_purged(api)) == 1


def test_cross_shop_push_is_403_and_changes_nothing(api: Api):
    before_auth = api.unit.repos.auth.get(SHOP_B)
    before_shop = api.unit.repos.knowledge.get_shop(SHOP_B)
    for kind in ("cancelled", "auth_revoked", "subscribed"):
        r = api.post(PATH, "push_a", json=_push(api, kind, SHOP_B))
        assert r.status_code == 403 and r.json()["error"] == "tenant_isolation"
        assert SHOP_B not in r.text
    assert api.unit.repos.auth.get(SHOP_B) == before_auth
    assert api.unit.repos.knowledge.get_shop(SHOP_B) == before_shop
    assert _purged(api) == []
    assert api.unit.repos.audit.list_events(SHOP_B, action=AuditAction.AUTH_GRANTED) == []
    # 还没开通的店铺也一样：别人的推送不能替它开通
    r = api.post(PATH, "push_a", json=_push(api, "subscribed", NEW))
    assert r.status_code == 403 and api.unit.repos.knowledge.get_shop(NEW) is None


@pytest.mark.parametrize("who", ["agent_a", "merchant_a"])
def test_only_the_integration_role_can_push(api: Api, who: str):
    r = api.post(PATH, who, json=_push(api, "cancelled", SHOP_A))
    assert r.status_code == 403 and r.json()["error"] == "forbidden_role"
    assert api.unit.repos.auth.get(SHOP_A) is not None


def test_missing_token_is_401(api: Api):
    r = api.post(PATH, None, json=_push(api, "cancelled", SHOP_A))
    assert r.status_code == 401
    assert api.unit.repos.auth.get(SHOP_A) is not None


def test_unrecognised_push_is_204(api: Api):
    r = api.post(PATH, "push_a", json={"type": "trade_created", "shop_id": SHOP_A})
    assert r.status_code == 204
    # 别的平台的推送：本单元的解析不认（§9.1 数据不跨云）
    r = api.post(PATH, "push_a", json=_push(api, "cancelled", SHOP_A, platform=Platform.DOUYIN))
    assert r.status_code == 204
    assert api.unit.repos.auth.get(SHOP_A) is not None


def test_excess_scopes_are_422_and_named(api: Api):
    scopes = sorted(MVP_SCOPES | {SCOPE_REFUND_WRITE})
    r = api.post(PATH, "push_new", json=_push(api, "subscribed", NEW, scopes=scopes))
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert SCOPE_REFUND_WRITE in r.json()["message"]
    assert GRANT_TOKEN not in r.text
    assert api.unit.repos.knowledge.get_shop(NEW) is None and api.unit.repos.auth.get(NEW) is None


def test_other_platform_event_from_a_lenient_parser_is_422(api: Api):
    api.unit.parse_subscription = parse_subscription  # 不按平台过滤的解析
    raw = _push(api, "subscribed", NEW, platform=Platform.DOUYIN)
    r = api.post(PATH, "push_new", json=raw)
    assert r.status_code == 422 and "douyin" in r.json()["message"]
    assert api.unit.repos.knowledge.get_shop(NEW) is None


def test_unit_without_subscription_parser_is_404(api: Api):
    api.unit.parse_subscription = None
    r = api.post(PATH, "push_a", json=_push(api, "cancelled", SHOP_A))
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    assert api.unit.repos.auth.get(SHOP_A) is not None


def test_access_mode_must_be_a_registered_platform_channel(api: Api):
    raw = _push(api, "cancelled", SHOP_A)
    assert api.post("/v1/platform/C/subscription", "push_a", json=raw).status_code == 422
    assert api.post("/v1/platform/A/subscription", "push_a", json=raw).status_code == 404
    assert api.unit.repos.auth.get(SHOP_A) is not None


def test_route_is_in_the_api_schema(api: Api):
    paths = api.client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/v1/platform/{access}/subscription"]
