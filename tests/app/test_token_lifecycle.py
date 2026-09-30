"""店铺令牌续期（方案 §8.4 "按期刷新"）：FakeClock + FakeTokenRefresher（平台续期接口一律 mock）。

续期成功换新令牌；平台拒绝 → 标记需要重新授权，之后平台事件 403 shop_not_authorized；
平台暂时不可用 → 什么都不改；令牌不进日志和审计。全部是合成令牌。
"""

from __future__ import annotations

import json
import logging
import warnings
from contextlib import contextmanager
from datetime import timedelta

import pytest
from pydantic import SecretStr
from starlette.exceptions import StarletteDeprecationWarning

from rodit.api import Principal, StaticTokenAuthenticator, create_app
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.app.token_lifecycle import (
    RefreshOutcome,
    RefreshUnavailable,
    refresh_expiring_tokens,
)
from rodit.core.audit import AuditAction
from rodit.core.config import EngineConfig, TenancyConfig
from rodit.core.enums import Platform
from rodit.core.errors import AuthExpired, TokenDecryptionError, ToolUnavailable
from rodit.core.models import RefreshedToken, ShopAuth
from rodit.core.scopes import MVP_SCOPES
from rodit.testing.fakes import FakeTokenRefresher
from rodit.testing.fixtures import SHOP_A, SHOP_B

with warnings.catch_warnings():
    # 依赖冻结：开发依赖里是 httpx（不是 httpx2），starlette 导入测试客户端时会发弃用警告
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

OLD_ACCESS = "tok-old-access-synthetic-51"
OLD_REFRESH = "tok-old-refresh-synthetic-52"
PUSH_TOKEN = "test-token-integration-a-0123456789"


def _service(config: EngineConfig | None = None) -> tuple[AssistService, FakeTokenRefresher]:
    unit = build_demo_unit(Platform.TAOBAO, config=config)
    refresher = FakeTokenRefresher(Platform.TAOBAO, clock=unit.clock)
    unit.token_refresher = refresher
    return AssistService(unit), refresher


@pytest.fixture
def svc() -> tuple[AssistService, FakeTokenRefresher]:
    return _service()


def _grant(
    service: AssistService,
    shop_id: str,
    *,
    hours: float,
    refresh: str | None = OLD_REFRESH,
    refresh_days: float | None = 60,
    needs_reauth: bool = False,
) -> ShopAuth:
    now = service.unit.clock.now()
    auth = ShopAuth(
        shop_id=shop_id,
        access_token=SecretStr(f"{OLD_ACCESS}-{shop_id}"),
        scopes=sorted(MVP_SCOPES),
        expires_at=now + timedelta(hours=hours),
        refresh_token=SecretStr(refresh) if refresh else None,
        refresh_expires_at=(
            now + timedelta(days=refresh_days) if refresh and refresh_days is not None else None
        ),
        needs_reauth=needs_reauth,
    )
    service.unit.repos.auth.save(auth)
    return auth


def _audit_text(service: AssistService) -> str:
    return json.dumps(
        [e.model_dump(mode="json") for e in service.unit.repos.audit.events], ensure_ascii=False
    )


def test_expiring_token_is_refreshed_and_audited(svc):
    service, refresher = svc
    unit = service.unit
    now = unit.clock.now()
    _grant(service, SHOP_A, hours=12)  # 24 小时内到期：续
    before_b = unit.repos.auth.get(SHOP_B)  # 30 天后才到期：不动

    report = service.tenancy.refresh_tokens()
    assert report.of(RefreshOutcome.REFRESHED) == [SHOP_A] and report.checked == 1
    assert refresher.calls_for(SHOP_A) == 1 and refresher.calls_for(SHOP_B) == 0
    [(_, sent)] = refresher.calls
    assert sent.get_secret_value() == OLD_REFRESH  # 用的是店铺自己的续期令牌

    auth = unit.repos.auth.get(SHOP_A)
    assert auth.access_token.get_secret_value().startswith("tok-refreshed-s_1001-")
    assert auth.expires_at == now + timedelta(days=30)
    assert auth.refresh_token.get_secret_value().startswith("tok-refresh-s_1001-")
    assert auth.refresh_expires_at == now + timedelta(days=90)
    assert auth.refreshed_at == now and auth.needs_reauth is False
    assert sorted(auth.scopes) == sorted(MVP_SCOPES)
    assert unit.repos.auth.get(SHOP_B) == before_b

    [event] = unit.repos.audit.list_events(SHOP_A, action=AuditAction.AUTH_REFRESHED)
    assert event.actor == "system"
    assert event.detail == {
        "previous_expires_at": (now + timedelta(hours=12)).isoformat(),
        "expires_at": (now + timedelta(days=30)).isoformat(),
        "refresh_rotated": True,
    }
    # 已经续过了：下一轮不再续
    assert service.tenancy.refresh_tokens().checked == 0


def test_already_expired_access_token_is_recovered_with_a_valid_refresh_token(svc):
    service, _ = svc
    _grant(service, SHOP_A, hours=-5)
    with pytest.raises(AuthExpired):
        service.unit.auth.require(SHOP_A)
    assert service.tenancy.refresh_tokens().of(RefreshOutcome.REFRESHED) == [SHOP_A]
    service.unit.auth.require(SHOP_A)


def test_refresh_window_follows_config():
    service, refresher = _service(EngineConfig(tenancy=TenancyConfig(refresh_ahead_hours=2)))
    _grant(service, SHOP_A, hours=3)
    assert service.tenancy.refresh_tokens().checked == 0
    service.unit.clock.advance(hours=1.5)
    assert service.tenancy.refresh_tokens().of(RefreshOutcome.REFRESHED) == [SHOP_A]


def test_platform_keeping_the_refresh_token_keeps_the_old_one():
    unit = build_demo_unit(Platform.TAOBAO)
    unit.token_refresher = FakeTokenRefresher(
        Platform.TAOBAO, clock=unit.clock, rotate_refresh=False
    )
    service = AssistService(unit)
    old = _grant(service, SHOP_A, hours=1)
    service.tenancy.refresh_tokens()
    auth = unit.repos.auth.get(SHOP_A)
    assert auth.refresh_token.get_secret_value() == OLD_REFRESH
    assert auth.refresh_expires_at == old.refresh_expires_at
    [event] = unit.repos.audit.list_events(SHOP_A, action=AuditAction.AUTH_REFRESHED)
    assert event.detail["refresh_rotated"] is False


def test_rejected_refresh_marks_needs_reauth_and_platform_events_get_403(svc):
    service, refresher = svc
    unit = service.unit
    _grant(service, SHOP_A, hours=6)  # 还没过期，但续期被拒（商家撤销了授权）
    refresher.script(SHOP_A, AuthExpired("续期令牌已失效（模拟）"))

    report = service.tenancy.refresh_tokens()
    assert report.of(RefreshOutcome.NEEDS_REAUTH) == [SHOP_A]
    auth = unit.repos.auth.get(SHOP_A)
    assert auth.needs_reauth is True
    assert auth.access_token.get_secret_value() == f"{OLD_ACCESS}-{SHOP_A}"  # 令牌没动
    [failed] = unit.repos.audit.list_events(SHOP_A, action=AuditAction.AUTH_REFRESH_FAILED)
    assert failed.detail["reason"] == "rejected"

    # 应用服务：停止处理
    with pytest.raises(AuthExpired):
        service.handle_event("B", taobao_event("在吗", msg_id="nr_1"))
    # HTTP：平台事件接口 403 shop_not_authorized
    push = Principal(shop_id=SHOP_A, user_id="tb_push_a", role="integration")
    app = create_app(service, authenticator=StaticTokenAuthenticator({PUSH_TOKEN: push}))
    r = TestClient(app).post(
        "/v1/platform/B/events",
        json=taobao_event("在吗", msg_id="nr_2"),
        headers={"Authorization": f"Bearer {PUSH_TOKEN}"},
    )
    assert r.status_code == 403 and r.json()["error"] == "shop_not_authorized"

    # 已标记：下一轮不再调平台，等商家重新授权
    again = service.tenancy.refresh_tokens()
    assert again.of(RefreshOutcome.ALREADY_REAUTH) == [SHOP_A]
    assert refresher.calls_for(SHOP_A) == 1


def test_tool_unavailable_leaves_everything_unchanged_until_next_run(svc):
    service, refresher = svc
    unit = service.unit
    before = _grant(service, SHOP_A, hours=3)
    refresher.script(SHOP_A, ToolUnavailable("限流（模拟）", reason="throttled"))

    report = service.tenancy.refresh_tokens()
    assert report.of(RefreshOutcome.RETRY) == [SHOP_A]
    assert unit.repos.auth.get(SHOP_A) == before
    assert unit.repos.audit.list_events(SHOP_A, action=AuditAction.AUTH_REFRESHED) == []
    assert unit.repos.audit.list_events(SHOP_A, action=AuditAction.AUTH_REFRESH_FAILED) == []
    unit.auth.require(SHOP_A)  # 照常处理

    assert service.tenancy.refresh_tokens().of(RefreshOutcome.REFRESHED) == [SHOP_A]


def test_tokens_never_reach_logs_or_audit(svc, caplog):
    service, refresher = svc
    _grant(service, SHOP_A, hours=1)
    _grant(service, SHOP_B, hours=2)
    _grant(service, "s_extra", hours=3)
    # 平台云代码的异常说明里带了令牌：日志只记类型 / 原因，不记说明
    refresher.script(SHOP_A, ToolUnavailable(f"平台返回 {OLD_REFRESH} 限流", reason="throttled"))
    refresher.script(SHOP_B, AuthExpired(f"refresh_token {OLD_REFRESH} invalid"))
    refresher.script("s_extra", RuntimeError(f"bug: {OLD_ACCESS}"))
    with caplog.at_level(logging.DEBUG):
        first = service.tenancy.refresh_tokens()
        second = service.tenancy.refresh_tokens()  # 店铺 A、s_extra 这次续成功
    assert first.of(RefreshOutcome.RETRY) == [SHOP_A]
    assert first.of(RefreshOutcome.NEEDS_REAUTH) == [SHOP_B]
    assert first.of(RefreshOutcome.ERROR) == ["s_extra"]
    assert set(second.of(RefreshOutcome.REFRESHED)) == {SHOP_A, "s_extra"}
    new_tokens = [
        service.unit.repos.auth.get(s).access_token.get_secret_value() for s in (SHOP_A, "s_extra")
    ]
    secrets = [OLD_ACCESS, OLD_REFRESH, "tok-refreshed-", "tok-refresh-", *new_tokens]
    assert caplog.records  # 确实记了日志
    for secret in secrets:
        assert secret not in caplog.text
        assert secret not in _audit_text(service)
    assert "throttled" in caplog.text and "RuntimeError" in caplog.text


def test_shops_that_cannot_be_refreshed_are_skipped_without_calling_the_platform(svc):
    service, refresher = svc
    unit = service.unit
    _grant(service, SHOP_A, hours=2, refresh=None)
    _grant(service, SHOP_B, hours=2, refresh_days=-1)  # 续期令牌自己过期了
    _grant(service, "s_reauth", hours=2, needs_reauth=True)
    report = service.tenancy.refresh_tokens()
    assert report.of(RefreshOutcome.NO_REFRESH_TOKEN) == [SHOP_A]
    assert report.of(RefreshOutcome.REFRESH_EXPIRED) == [SHOP_B]
    assert report.of(RefreshOutcome.ALREADY_REAUTH) == ["s_reauth"]
    assert refresher.calls == []
    # 续期令牌过期不提前停止处理：access_token 到期之前照常
    assert unit.repos.auth.get(SHOP_B).needs_reauth is False
    unit.auth.require(SHOP_B)
    counts = report.counts()
    assert counts["no_refresh_token"] == 1 and counts["refreshed"] == 0
    assert set(counts) == {o.value for o in RefreshOutcome}


def test_unreadable_token_is_marked_for_reauth(svc):
    service, refresher = svc
    unit = service.unit
    _grant(service, SHOP_A, hours=1)
    real_get = unit.repos.auth.get

    def get(shop_id: str):
        if shop_id == SHOP_A:
            raise TokenDecryptionError("模拟：换过密钥")
        return real_get(shop_id)

    unit.repos.auth.get = get  # type: ignore[method-assign]
    report = service.tenancy.refresh_tokens()
    unit.repos.auth.get = real_get  # type: ignore[method-assign]
    assert report.of(RefreshOutcome.NEEDS_REAUTH) == [SHOP_A]
    assert refresher.calls == []
    assert unit.repos.auth.get(SHOP_A).needs_reauth is True
    [failed] = unit.repos.audit.list_events(SHOP_A, action=AuditAction.AUTH_REFRESH_FAILED)
    assert failed.detail == {"reason": "token_unreadable"}


def test_grant_replaced_during_refresh_is_not_overwritten(svc):
    """续期期间商家重新订购了（平台推了新令牌）：续期结果不覆盖新授权。"""
    service, _ = svc
    unit = service.unit
    _grant(service, SHOP_A, hours=1)
    renewed = ShopAuth(
        shop_id=SHOP_A,
        access_token=SecretStr("tok-renewed-meanwhile"),
        scopes=sorted(MVP_SCOPES),
        expires_at=unit.clock.now() + timedelta(days=365),
    )

    class RacingRefresher:
        platform = Platform.TAOBAO

        def refresh(self, shop_id: str, refresh_token: SecretStr) -> RefreshedToken:
            unit.repos.auth.save(renewed)  # 同时到达的续订
            return RefreshedToken(
                access_token=SecretStr("tok-from-refresh"),
                expires_at=unit.clock.now() + timedelta(days=30),
            )

    unit.token_refresher = RacingRefresher()
    report = service.tenancy.refresh_tokens()
    assert report.of(RefreshOutcome.SKIPPED) == [SHOP_A]
    assert unit.repos.auth.get(SHOP_A).access_token.get_secret_value() == "tok-renewed-meanwhile"


def test_shop_revoked_after_listing_is_skipped(svc):
    service, refresher = svc
    unit = service.unit
    real = unit.repos.auth.expiring_shop_ids

    def listing(before):  # 列出的店铺在续期前被取消订购了
        return [("s_gone", unit.clock.now()), *real(before)]

    unit.repos.auth.expiring_shop_ids = listing  # type: ignore[method-assign]
    report = service.tenancy.refresh_tokens()
    assert report.of(RefreshOutcome.SKIPPED) == ["s_gone"]
    assert refresher.calls == []


def test_each_refresh_runs_inside_the_shop_gate(svc):
    """续期在 processing(shop_id) 里做：和取消订购互斥（取消订购等它做完再删令牌）。"""
    service, refresher = svc
    _grant(service, SHOP_A, hours=1)
    _grant(service, SHOP_B, hours=2)
    held: list[str] = []
    inside: list[tuple[str, list[str]]] = []

    @contextmanager
    def gate(shop_id: str):
        held.append(shop_id)
        try:
            yield
        finally:
            held.remove(shop_id)

    real = refresher.refresh

    def refresh(shop_id: str, token: SecretStr) -> RefreshedToken:
        inside.append((shop_id, list(held)))
        return real(shop_id, token)

    refresher.refresh = refresh  # type: ignore[method-assign]
    report = refresh_expiring_tokens(service.unit, processing=gate)
    assert report.of(RefreshOutcome.REFRESHED) == [SHOP_A, SHOP_B]
    assert inside == [(SHOP_A, [SHOP_A]), (SHOP_B, [SHOP_B])]
    assert held == []


def test_refresh_without_a_refresher_is_refused():
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    assert service.unit.token_refresher is None
    with pytest.raises(RefreshUnavailable, match="token_refresher"):
        service.tenancy.refresh_tokens()
