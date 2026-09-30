"""店铺授权在售后审核、商品同步路径上的检查（方案 §8.4 授权过期时停止处理、最小权限：
写售后到二期再申请；§4.1 原则 3）。

授权无效（过期、需要重新授权、没有授权）时，售后审核的每一步、商品同步都在调平台接口之前停下
（AuthExpired → 接口层 403 shop_not_authorized），平台只读工具、商品接口一次都不调；
客服确认执行写操作还要有 refund.write，没有就不碰写接口。模式 C 不需要平台授权，照常可用。
"""

from __future__ import annotations

import warnings
from datetime import timedelta

import pytest
from pydantic import SecretStr
from starlette.exceptions import StarletteDeprecationWarning

from rodit.api import Principal, StaticTokenAuthenticator, create_app
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit
from rodit.app.service import aftersale_scopes
from rodit.core.audit import AuditAction
from rodit.core.enums import AfterSaleAction, Platform, ReviewState
from rodit.core.errors import ActionNotAllowed, AuthExpired
from rodit.core.models import ShopAuth
from rodit.core.scopes import MVP_SCOPES, PHASE2_SCOPES, SCOPE_REFUND_WRITE
from rodit.testing.fixtures import AGENT, SHOP_A

with warnings.catch_warnings():
    # 依赖冻结：开发依赖里是 httpx（不是 httpx2），starlette 导入测试客户端时会发弃用警告
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

AGENT_TOKEN = "test-token-agent-a-auth-0123456789"
MERCHANT_TOKEN = "test-token-merchant-a-auth-0123456789"


@pytest.fixture
def service() -> AssistService:
    """演示单元带售后写接口（替身）时默认授权含写售后权限；这里的店铺 A 只授权了只读（MVP），
    才能测"没有 refund.write 不碰写接口"。"""
    svc = AssistService(build_demo_unit(Platform.TAOBAO))
    auth = svc.unit.repos.auth.get(SHOP_A)
    assert auth is not None
    svc.unit.repos.auth.save(auth.model_copy(update={"scopes": sorted(MVP_SCOPES)}))
    return svc


def _auth(service: AssistService, *, days: float = 30, scopes=MVP_SCOPES, **kw) -> None:
    unit = service.unit
    unit.repos.auth.save(
        ShopAuth(
            shop_id=SHOP_A,
            access_token=SecretStr("tok-enforcement-synthetic"),
            scopes=sorted(scopes),
            expires_at=unit.clock.now() + timedelta(days=days),
            **kw,
        )
    )


def _tool_calls(service: AssistService) -> int:
    return len(service.unit.tools.calls)


def _to_confirm(service: AssistService, aftersale_id: str = "as_2001") -> str:
    """跑到待客服确认，返回平台状态版本（授权有效时）。"""
    review = service.review_aftersale(SHOP_A, aftersale_id)
    assert review.state == ReviewState.CONFIRM
    return service.unit.tools.get_aftersale(SHOP_A, aftersale_id).status_version


# ================================================================ 授权过期：一次平台接口都不调


def test_expired_auth_stops_review_before_any_platform_read(service: AssistService):
    _auth(service, days=-1)  # 昨天过期，但还没取消订购
    before = _tool_calls(service)
    with pytest.raises(AuthExpired, match="过期"):
        service.review_aftersale(SHOP_A, "as_2001")
    assert _tool_calls(service) == before == 0
    assert service.unit.repos.reviews.get(SHOP_A, "as_2001") is None


def test_every_aftersale_step_checks_auth_first(service: AssistService):
    version = _to_confirm(service, "as_2002")
    service.negotiate_aftersale(SHOP_A, "as_2002", operator=AGENT)
    _auth(service, days=-1)
    before = _tool_calls(service)
    calls = service.unit.action_port.calls
    with pytest.raises(AuthExpired):
        service.review_aftersale(SHOP_A, "as_2002")  # 刷新
    with pytest.raises(AuthExpired):
        service.negotiate_aftersale(SHOP_A, "as_2002", operator=AGENT)
    with pytest.raises(AuthExpired):
        service.negotiation_done(SHOP_A, "as_2002", operator=AGENT, outcome="买家同意换码")
    with pytest.raises(AuthExpired):
        service.confirm_aftersale(
            SHOP_A,
            "as_2002",
            AfterSaleAction.AGREE_RETURN,
            operator=AGENT,
            expected_status_version=version,
        )
    assert _tool_calls(service) == before and calls == []
    assert service.unit.repos.reviews.get(SHOP_A, "as_2002").state == ReviewState.NEGOTIATE


@pytest.mark.parametrize(
    "state",
    [
        {"needs_reauth": True},  # 续期被拒 / 平台拒绝了令牌
        {"scopes": ["order.read", "logistics.read"]},  # 缺只读权限
    ],
)
def test_needs_reauth_or_missing_read_scopes_stop_review(service: AssistService, state):
    scopes = state.get("scopes", MVP_SCOPES)
    _auth(service, scopes=scopes, needs_reauth=state.get("needs_reauth", False))
    with pytest.raises(AuthExpired):
        service.review_aftersale(SHOP_A, "as_2001")
    assert _tool_calls(service) == 0


def test_revoked_shop_review_is_refused(service: AssistService):
    service.revoke_shop(SHOP_A)
    with pytest.raises(AuthExpired, match="没有授权"):
        service.review_aftersale(SHOP_A, "as_2001")
    assert _tool_calls(service) == 0


# ================================================================ 写售后要 refund.write


@pytest.mark.parametrize(
    ("aftersale_id", "action"),
    [("as_2001", AfterSaleAction.AGREE_REFUND), ("as_2002", AfterSaleAction.AGREE_RETURN)],
)
def test_confirm_under_mvp_scopes_never_touches_the_write_port(
    service: AssistService, aftersale_id: str, action: AfterSaleAction
):
    version = _to_confirm(service, aftersale_id)
    assert SCOPE_REFUND_WRITE not in service.unit.repos.auth.get(SHOP_A).scopes
    with pytest.raises(AuthExpired, match=SCOPE_REFUND_WRITE):
        service.confirm_aftersale(
            SHOP_A, aftersale_id, action, operator=AGENT, expected_status_version=version
        )
    assert service.unit.action_port.calls == []
    assert service.unit.repos.reviews.get(SHOP_A, aftersale_id).state == ReviewState.CONFIRM
    audit = service.unit.repos.audit
    assert audit.list_events(SHOP_A, action=AuditAction.REVIEW_CONFIRM) == []
    assert audit.list_events(SHOP_A, action=AuditAction.REVIEW_ACTION) == []


def test_refuse_also_needs_refund_write(service: AssistService):
    version = _to_confirm(service, "as_2002")
    with pytest.raises(AuthExpired, match=SCOPE_REFUND_WRITE):
        service.confirm_aftersale(
            SHOP_A,
            "as_2002",
            AfterSaleAction.REFUSE,
            operator=AGENT,
            expected_status_version=version,
        )
    assert service.unit.action_port.calls == []


def test_confirm_with_refund_write_calls_the_port_once(service: AssistService):
    version = _to_confirm(service)
    _auth(service, scopes=PHASE2_SCOPES)
    done = service.confirm_aftersale(
        SHOP_A,
        "as_2001",
        AfterSaleAction.AGREE_REFUND,
        operator=AGENT,
        expected_status_version=version,
    )
    assert done.state == ReviewState.AGREED
    [call] = service.unit.action_port.calls
    assert call["operator"] == AGENT


def test_without_a_write_port_confirm_is_refused_as_before_without_scope_check():
    """MVP 单元没接写接口：审核助手本来就拒绝执行（409），不因缺写权限改成 403。"""
    service = AssistService(build_demo_unit(Platform.TAOBAO, with_action_port=False))
    version = _to_confirm(service)
    with pytest.raises(ActionNotAllowed):
        service.confirm_aftersale(
            SHOP_A,
            "as_2001",
            AfterSaleAction.AGREE_REFUND,
            operator=AGENT,
            expected_status_version=version,
        )
    # 授权过期时仍然先停下
    _auth(service, days=-1)
    with pytest.raises(AuthExpired):
        service.confirm_aftersale(
            SHOP_A,
            "as_2001",
            AfterSaleAction.AGREE_REFUND,
            operator=AGENT,
            expected_status_version=version,
        )


def test_aftersale_scopes_cover_every_write_action():
    for action in AfterSaleAction:
        assert aftersale_scopes(action) == MVP_SCOPES | {SCOPE_REFUND_WRITE}


# ================================================================ 商品同步


def test_item_sync_checks_auth_before_calling_the_catalog(service: AssistService):
    catalog = service.unit.catalog
    _auth(service, days=-1)
    with pytest.raises(AuthExpired):
        service.console.sync_items(SHOP_A, merchant="owner_a")
    assert catalog.calls == []
    _auth(service, needs_reauth=True)
    with pytest.raises(AuthExpired):
        service.console.sync_items(SHOP_A, item_id="i_555", merchant="owner_a")
    assert catalog.calls == []
    _auth(service)
    report = service.console.sync_items(SHOP_A, merchant="owner_a")
    assert catalog.calls and report is not None


# ================================================================ 模式 C 不需要平台授权


def test_mode_c_still_works_with_expired_auth(service: AssistService):
    _auth(service, days=-1)
    result = service.ask_standalone(shop_id=SHOP_A, session_id="web_1", question="能开发票吗")
    assert result is not None and result.suggestions


# ================================================================ HTTP：403 shop_not_authorized


def _client(service: AssistService) -> TestClient:
    authenticator = StaticTokenAuthenticator(
        {
            AGENT_TOKEN: Principal(shop_id=SHOP_A, user_id=AGENT, role="agent"),
            MERCHANT_TOKEN: Principal(shop_id=SHOP_A, user_id="owner_a", role="merchant"),
        }
    )
    return TestClient(create_app(service, authenticator=authenticator))


def test_http_confirm_without_write_scope_is_403_and_port_untouched(service: AssistService):
    client = _client(service)
    agent = {"Authorization": f"Bearer {AGENT_TOKEN}"}
    base = f"/v1/shops/{SHOP_A}/aftersales/as_2001"
    view = client.post(f"{base}/review", headers=agent).json()
    r = client.post(
        f"{base}/confirm",
        headers=agent,
        json={"action": "agree_refund", "expected_status_version": view["platform_status_version"]},
    )
    assert r.status_code == 403 and r.json()["error"] == "shop_not_authorized"
    assert SCOPE_REFUND_WRITE in r.json()["message"]
    assert service.unit.action_port.calls == []


def test_http_review_and_sync_with_expired_auth_are_403(service: AssistService):
    client = _client(service)
    _auth(service, days=-1)
    r = client.post(
        f"/v1/shops/{SHOP_A}/aftersales/as_2001/review",
        headers={"Authorization": f"Bearer {AGENT_TOKEN}"},
    )
    assert r.status_code == 403 and r.json()["error"] == "shop_not_authorized"
    r = client.post(
        f"/v1/shops/{SHOP_A}/console/items/sync",
        headers={"Authorization": f"Bearer {MERCHANT_TOKEN}"},
    )
    assert r.status_code == 403 and r.json()["error"] == "shop_not_authorized"
    assert _tool_calls(service) == 0 and service.unit.catalog.calls == []
