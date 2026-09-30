"""店铺生命周期（方案 §8.4 开通 / 续订 / 取消订购 / 授权过期，§9.1 租户开通，§10 shop）。

AssistService.tenancy 跑在演示单元（内存仓储、合成数据）上：开通后能出推荐、权限和平台检查、
状态切换、订购事件、取消订购后的系统事件、过期清理。令牌全部是合成的。
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from pydantic import SecretStr

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.app.tenancy import (
    PLATFORM_ACTOR,
    PURGE_REASON_EXPIRED,
    STATUS_TRANSITIONS,
    PurgeIncomplete,
)
from rodit.core.audit import AuditAction
from rodit.core.config import EngineConfig, TenancyConfig
from rodit.core.enums import Platform, ShopStatus
from rodit.core.errors import ActionNotAllowed, AuthExpired, ShopInactive, TokenDecryptionError
from rodit.core.models import ShopAuth, SubscriptionEvent
from rodit.core.scopes import MVP_SCOPES, PHASE2_SCOPES, SCOPE_REFUND_WRITE
from rodit.engine.engine import REASON_SHOP_INACTIVE
from rodit.testing.fakes import parse_subscription, subscription_push
from rodit.testing.fixtures import SHOP_A, SHOP_B

NEW = "s_new"
TOKEN = "tok-onboard-synthetic-7f3a"
REFRESH = "tok-refresh-synthetic-9c1d"
MERCHANT = "owner_new"
FAQ_Q = "能开发票吗"
FAQ_A = "亲，支持开具电子发票，确认收货后在订单详情里申请就可以啦~"


def _service(config: EngineConfig | None = None) -> AssistService:
    return AssistService(build_demo_unit(Platform.TAOBAO, config=config))


@pytest.fixture
def service() -> AssistService:
    return _service()


def _event(service: AssistService, kind: str = "subscribed", shop_id: str = NEW, **kw):
    kw.setdefault("at", service.unit.clock.now())
    if kind in ("subscribed", "renewed"):
        kw.setdefault("access_token", TOKEN)
    event = parse_subscription(subscription_push(kind, shop_id, **kw))
    assert event is not None
    return event


def _ask(service: AssistService, text: str, *, msg_id: str, shop_id: str = NEW):
    return service.handle_event("B", taobao_event(text, msg_id=msg_id, seller_id=shop_id))


def _add_faq(service: AssistService, shop_id: str = NEW) -> None:
    service.console.create_faq(shop_id, question=FAQ_Q, answer=FAQ_A, merchant=MERCHANT)


def _no_token_text(service: AssistService) -> None:
    """审计里没有令牌原文（§8.4 令牌只在加密的授权里）。"""
    dumped = json.dumps(
        [e.model_dump(mode="json") for e in service.unit.repos.audit.events], ensure_ascii=False
    )
    assert TOKEN not in dumped and REFRESH not in dumped


# ================================================================ 开通


def test_onboard_then_handle_event_produces_suggestions(service: AssistService):
    unit = service.unit
    # 开通前：没有授权，平台事件直接拒绝
    with pytest.raises(AuthExpired):
        _ask(service, FAQ_Q, msg_id="m0")

    event = _event(service, refresh_token=REFRESH, category="女装")
    shop = service.tenancy.onboard(event)
    assert shop.id == NEW and shop.platform == Platform.TAOBAO
    assert shop.status == ShopStatus.TRIAL and shop.category == "女装"
    assert unit.repos.knowledge.get_shop(NEW) == shop
    auth = unit.repos.auth.get(NEW)
    assert auth.access_token.get_secret_value() == TOKEN
    assert auth.refresh_token.get_secret_value() == REFRESH
    assert sorted(auth.scopes) == sorted(MVP_SCOPES) and auth.needs_reauth is False
    assert auth.expires_at == event.expires_at
    assert unit.auth.is_authorized(NEW)

    _add_faq(service)
    result = _ask(service, FAQ_Q, msg_id="m1")
    assert result is not None and result.suggestions
    assert result.suggestions[0].text == FAQ_A

    [granted] = unit.repos.audit.list_events(NEW, action=AuditAction.AUTH_GRANTED)
    assert granted.actor == "system"
    assert granted.detail["created"] is True and granted.detail["kind"] == "subscribed"
    assert granted.detail["scopes"] == sorted(MVP_SCOPES)
    assert granted.detail["refresh_token"] is True
    _no_token_text(service)


def test_onboard_accepts_explicit_active_status(service: AssistService):
    shop = service.tenancy.onboard(_event(service), status=ShopStatus.ACTIVE, actor="ops")
    assert shop.status == ShopStatus.ACTIVE
    with pytest.raises(ValueError, match="trial 或 active"):
        service.tenancy.onboard(_event(service, shop_id="s_new2"), status=ShopStatus.SUSPENDED)
    assert service.unit.repos.knowledge.get_shop("s_new2") is None


def test_phase1_onboarding_with_refund_write_is_rejected_and_names_it(service: AssistService):
    scopes = sorted(MVP_SCOPES | {SCOPE_REFUND_WRITE})
    with pytest.raises(ValueError) as err:
        service.tenancy.onboard(_event(service, scopes=scopes))
    assert SCOPE_REFUND_WRITE in str(err.value) and "第 1 阶段" in str(err.value)
    for scope in MVP_SCOPES:  # 只报多出来的
        assert scope not in str(err.value)
    # 什么都没写
    assert service.unit.repos.knowledge.get_shop(NEW) is None
    assert service.unit.repos.auth.get(NEW) is None
    assert service.unit.repos.audit.list_events(NEW) == []


def test_unknown_extra_scope_is_rejected_too(service: AssistService):
    with pytest.raises(ValueError, match="trade.write"):
        service.tenancy.onboard(_event(service, scopes=sorted(MVP_SCOPES | {"trade.write"})))


def test_phase2_onboarding_allows_refund_write():
    config = EngineConfig(tenancy=TenancyConfig(onboarding_phase=2))
    service = _service(config)
    service.tenancy.onboard(_event(service, scopes=sorted(PHASE2_SCOPES)))
    assert set(service.unit.repos.auth.get(NEW).scopes) == PHASE2_SCOPES


def test_missing_mvp_scopes_are_rejected(service: AssistService):
    with pytest.raises(ValueError, match="refund.read"):
        service.tenancy.onboard(_event(service, scopes=["order.read", "logistics.read"]))
    assert service.unit.repos.knowledge.get_shop(NEW) is None


def test_wrong_platform_is_rejected(service: AssistService):
    event = _event(service, platform=Platform.DOUYIN)
    with pytest.raises(ValueError, match="douyin"):
        service.tenancy.onboard(event)
    with pytest.raises(ValueError, match="数据不跨云"):
        service.tenancy.handle_subscription(event)
    assert service.unit.repos.knowledge.get_shop(NEW) is None
    assert service.unit.repos.auth.get(NEW) is None


def test_onboard_rejects_non_grant_events(service: AssistService):
    with pytest.raises(ValueError, match="不是开通"):
        service.tenancy.onboard(_event(service, "cancelled"))


def test_renewal_keeps_shop_state_and_replaces_the_token(service: AssistService):
    unit = service.unit
    service.tenancy.onboard(_event(service, category="女装"))
    service.tenancy.set_status(NEW, ShopStatus.SUSPENDED, actor="ops")
    unit.repos.knowledge.update_shop_config(NEW, {"salutation": "您"})
    unit.auth.mark_needs_reauth(NEW)
    later = unit.clock.now() + timedelta(days=20)
    renewed = _event(
        service, "renewed", access_token="tok-renewed-synthetic", expires_at=later + timedelta(30)
    )
    shop = service.tenancy.onboard(renewed)
    # 店铺状态（停用是人定的）和店铺配置不因续订改变；令牌换新、needs_reauth 清掉
    assert shop.status == ShopStatus.SUSPENDED
    assert shop.config_overrides == {"salutation": "您"} and shop.category == "女装"
    auth = unit.repos.auth.get(NEW)
    assert auth.access_token.get_secret_value() == "tok-renewed-synthetic"
    assert auth.needs_reauth is False and auth.refresh_token is None
    grants = unit.repos.audit.list_events(NEW, action=AuditAction.AUTH_GRANTED)
    assert [g.detail["created"] for g in grants] == [True, False]
    assert grants[-1].detail["kind"] == "renewed"


def test_renewal_without_scopes_reuses_and_rechecks_current_scopes(service: AssistService):
    unit = service.unit
    service.tenancy.onboard(_event(service))
    renewed = SubscriptionEvent(
        kind="renewed",
        shop_id=NEW,
        platform=Platform.TAOBAO,
        at=unit.clock.now(),
        access_token=SecretStr("tok-renewed-noscope"),
        expires_at=unit.clock.now() + timedelta(days=60),
    )
    service.tenancy.onboard(renewed)
    assert set(unit.repos.auth.get(NEW).scopes) == MVP_SCOPES
    # 现有授权里有本阶段不允许的权限（二期改回一期）：续订照样拒绝
    auth = unit.repos.auth.get(NEW)
    unit.repos.auth.save(auth.model_copy(update={"scopes": sorted(PHASE2_SCOPES)}))
    with pytest.raises(ValueError, match=SCOPE_REFUND_WRITE):
        service.tenancy.onboard(renewed)


def test_onboard_status_change_on_existing_shop_follows_transitions(service: AssistService):
    service.tenancy.onboard(_event(service), status=ShopStatus.ACTIVE)
    with pytest.raises(ActionNotAllowed):
        service.tenancy.onboard(_event(service), status=ShopStatus.TRIAL)
    assert service.unit.repos.knowledge.get_shop(NEW).status == ShopStatus.ACTIVE


# ================================================================ 店铺状态


def test_status_transitions_are_audited_and_suspended_blocks_processing(service: AssistService):
    unit = service.unit
    service.tenancy.onboard(_event(service))
    _add_faq(service)
    assert _ask(service, FAQ_Q, msg_id="m1").suggestions

    service.tenancy.set_status(NEW, ShopStatus.ACTIVE, actor="ops")
    service.tenancy.set_status(NEW, ShopStatus.SUSPENDED, actor="ops")
    # 停用：问答、模式 C、控制台一律停止处理
    blocked = _ask(service, FAQ_Q, msg_id="m2")
    assert blocked is not None and blocked.suggestions == []
    assert blocked.no_suggestion_reason == REASON_SHOP_INACTIVE
    with pytest.raises(ShopInactive):
        service.ask_standalone(shop_id=NEW, session_id="web", question=FAQ_Q)
    with pytest.raises(ShopInactive):
        service.console.create_faq(NEW, question="包邮吗", answer="亲，全场包邮哦~", merchant="o")

    shop = service.tenancy.set_status(NEW, ShopStatus.ACTIVE, actor="ops")
    assert shop.status == ShopStatus.ACTIVE
    assert _ask(service, FAQ_Q, msg_id="m3").suggestions

    changes = unit.repos.audit.list_events(NEW, action=AuditAction.SHOP_STATUS_CHANGED)
    assert [(c.detail["from"], c.detail["to"]) for c in changes] == [
        ("trial", "active"),
        ("active", "suspended"),
        ("suspended", "active"),
    ]
    assert {c.actor for c in changes} == {"ops"}


def test_set_status_rejects_unknown_shop_bad_transition_and_blank_actor(service: AssistService):
    with pytest.raises(KeyError):
        service.tenancy.set_status("s_nope", ShopStatus.ACTIVE, actor="ops")
    service.tenancy.onboard(_event(service), status=ShopStatus.ACTIVE)
    with pytest.raises(ActionNotAllowed, match="active 切换到 trial"):
        service.tenancy.set_status(NEW, ShopStatus.TRIAL, actor="ops")
    with pytest.raises(ValueError):
        service.tenancy.set_status(NEW, ShopStatus.SUSPENDED, actor="  ")
    # 状态没变：什么都不写
    before = len(service.unit.repos.audit.list_events(NEW))
    assert service.tenancy.set_status(NEW, ShopStatus.ACTIVE, actor="ops").status == "active"
    assert len(service.unit.repos.audit.list_events(NEW)) == before
    assert ShopStatus.TRIAL not in STATUS_TRANSITIONS[ShopStatus.ACTIVE]


# ================================================================ 订购事件 / 取消订购


def test_cancelled_subscription_revokes_and_leaves_a_system_record(service: AssistService):
    unit = service.unit
    assert unit.repos.knowledge.get_shop(SHOP_A) is not None
    outcome = service.tenancy.handle_subscription(_event(service, "cancelled", SHOP_A))
    assert outcome.action == "revoked" and outcome.counts["auth"] == 1
    assert sum(outcome.counts.values()) > 1
    # 令牌、店铺、店铺审计都删了；别的店不受影响
    assert unit.repos.auth.get(SHOP_A) is None
    assert unit.repos.knowledge.get_shop(SHOP_A) is None
    assert unit.repos.audit.list_events(SHOP_A) == []
    assert unit.repos.knowledge.get_shop(SHOP_B) is not None
    with pytest.raises(AuthExpired):
        _ask(service, "在吗", msg_id="after", shop_id=SHOP_A)

    [record] = unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.shop_id is None and record.actor == PLATFORM_ACTOR
    assert set(record.detail) == {"shop_id", "reason", "counts"}
    assert record.detail == {"shop_id": SHOP_A, "reason": "cancelled", "counts": outcome.counts}


def test_auth_revoked_is_handled_like_cancel(service: AssistService):
    outcome = service.tenancy.handle_subscription(_event(service, "auth_revoked", SHOP_B))
    assert outcome.action == "revoked"
    [record] = service.unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.detail["reason"] == "auth_revoked" and record.detail["shop_id"] == SHOP_B


def test_subscribed_and_renewed_events_onboard(service: AssistService):
    outcome = service.tenancy.handle_subscription(_event(service))
    assert outcome.action == "granted" and outcome.shop.status == ShopStatus.TRIAL
    renewed = service.tenancy.handle_subscription(_event(service, "renewed"))
    assert renewed.action == "granted"
    grants = service.unit.repos.audit.list_events(NEW, action=AuditAction.AUTH_GRANTED)
    assert [g.actor for g in grants] == [PLATFORM_ACTOR, PLATFORM_ACTOR]


def test_same_shop_can_onboard_again_after_revoke(service: AssistService):
    unit = service.unit
    service.tenancy.onboard(_event(service))
    _add_faq(service)
    assert _ask(service, FAQ_Q, msg_id="m1").suggestions
    service.tenancy.handle_subscription(_event(service, "cancelled"))
    assert unit.repos.knowledge.get_shop(NEW) is None

    unit.clock.advance(hours=1)
    again = service.tenancy.handle_subscription(_event(service, access_token="tok-second-grant"))
    assert again.action == "granted" and again.shop.status == ShopStatus.TRIAL
    assert unit.repos.auth.get(NEW).access_token.get_secret_value() == "tok-second-grant"
    # 数据都删过了：重新开通是一家新店（FAQ 要重新录入）；同一条消息 ID 也不会被当成重复
    empty = _ask(service, FAQ_Q, msg_id="m1")
    assert empty is not None and empty.suggestions == []
    _add_faq(service)
    assert _ask(service, FAQ_Q, msg_id="m2").suggestions
    [granted] = unit.repos.audit.list_events(NEW, action=AuditAction.AUTH_GRANTED)
    assert granted.detail["created"] is True


def test_stale_grant_arriving_after_the_purge_is_ignored(service: AssistService):
    unit = service.unit
    sent_at = unit.clock.now()
    unit.clock.advance(minutes=5)
    service.tenancy.handle_subscription(_event(service, "cancelled", SHOP_A))
    # 取消之前发出的开通推送（平台重推）晚到：不重新开通
    stale = service.tenancy.handle_subscription(_event(service, "subscribed", SHOP_A, at=sent_at))
    assert stale.action == "ignored"
    assert unit.repos.auth.get(SHOP_A) is None and unit.repos.knowledge.get_shop(SHOP_A) is None
    # 直接调 onboard（运维开通）不受影响
    service.tenancy.onboard(_event(service, "subscribed", SHOP_A, at=sent_at))
    assert unit.repos.auth.get(SHOP_A) is not None


def test_revoke_requires_reason_and_propagates_purge_failures(service: AssistService):
    with pytest.raises(ValueError):
        service.tenancy.revoke(SHOP_A, reason=" ")
    assert service.unit.repos.auth.get(SHOP_A) is not None

    class Broken:
        def purge_shop(self, shop_id: str) -> int:
            raise RuntimeError("磁盘满了（模拟）")

    service.unit.extra_purgers.append(Broken())
    with pytest.raises(RuntimeError):
        service.tenancy.revoke(SHOP_A, reason="ops", actor="ops")
    # 没删完：不写 shop.purged（要重试）；令牌已删，处理已停止
    assert service.unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED) == []
    assert service.unit.repos.auth.get(SHOP_A) is None


# ================================================================ 过期清理


def _expire(service: AssistService, shop_id: str, days_ago: float) -> None:
    unit = service.unit
    auth = unit.repos.auth.get(shop_id)
    expires = unit.clock.now() - timedelta(days=days_ago)
    unit.repos.auth.save(auth.model_copy(update={"expires_at": expires}))


def test_purge_expired_only_removes_shops_past_the_grace_period(service: AssistService):
    unit = service.unit
    service.tenancy.onboard(_event(service))
    _expire(service, SHOP_A, 31)  # 过期 31 天：超过 30 天宽限期
    _expire(service, SHOP_B, 29)  # 过期 29 天：还在宽限期内
    _expire(service, NEW, 30)  # 刚好 30 天：不算"超过"

    preview = service.tenancy.purge_expired(dry_run=True)
    assert preview == {SHOP_A: {}}
    assert unit.repos.auth.get(SHOP_A) is not None  # dry_run 不删

    purged = service.tenancy.purge_expired()
    assert list(purged) == [SHOP_A] and purged[SHOP_A]["auth"] == 1
    assert unit.repos.knowledge.get_shop(SHOP_A) is None
    assert unit.repos.knowledge.get_shop(SHOP_B) is not None
    assert unit.repos.knowledge.get_shop(NEW) is not None

    [record] = unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.detail == {
        "shop_id": SHOP_A,
        "reason": PURGE_REASON_EXPIRED,
        "counts": purged[SHOP_A],
    }
    assert record.actor == "system"
    # 再跑一次：没有要删的
    assert service.tenancy.purge_expired() == {}


def test_purge_expired_uses_the_configured_grace_period():
    service = _service(EngineConfig(tenancy=TenancyConfig(purge_after_expiry_days=0)))
    _expire(service, SHOP_A, 0.01)
    assert service.tenancy.purge_expired(dry_run=True) == {SHOP_A: {}}
    later = service.unit.clock.now() + timedelta(days=60)
    # now 参数：按那个时间算（两家店都过期了）
    assert set(service.tenancy.purge_expired(now=later, dry_run=True)) == {SHOP_A, SHOP_B}


def test_purge_expired_skips_shops_reauthorized_after_listing(service: AssistService):
    unit = service.unit
    _expire(service, SHOP_A, 40)
    real_get = unit.repos.auth.get
    calls = {"n": 0}

    def get(shop_id: str):
        calls["n"] += 1
        auth = real_get(shop_id)
        if auth is not None:  # 列出之后商家重新授权了
            return auth.model_copy(update={"expires_at": unit.clock.now() + timedelta(days=30)})
        return auth

    unit.repos.auth.get = get  # type: ignore[method-assign]
    assert service.tenancy.purge_expired() == {}
    assert calls["n"] == 1
    unit.repos.auth.get = real_get  # type: ignore[method-assign]
    assert unit.repos.knowledge.get_shop(SHOP_A) is not None


def test_purge_expired_continues_after_a_failure_and_reports_it(service: AssistService):
    unit = service.unit
    _expire(service, SHOP_A, 40)
    _expire(service, SHOP_B, 50)

    class FailsForB:
        def purge_shop(self, shop_id: str) -> int:
            if shop_id == SHOP_B:
                raise RuntimeError("模拟清理失败")
            return 0

    unit.extra_purgers.append(FailsForB())
    with pytest.raises(PurgeIncomplete) as err:
        service.tenancy.purge_expired()
    assert err.value.failed == [SHOP_B] and list(err.value.purged) == [SHOP_A]
    [record] = unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.detail["shop_id"] == SHOP_A


def test_expired_shop_with_unreadable_token_is_still_purged(service: AssistService):
    """令牌读不出来（换过密钥）也照样清理：列出、删除都不解密。"""
    unit = service.unit
    _expire(service, SHOP_A, 40)
    real_get = unit.repos.auth.get

    def get(shop_id: str):
        if shop_id == SHOP_A:
            raise TokenDecryptionError("模拟：密钥不对")
        return real_get(shop_id)

    unit.repos.auth.get = get  # type: ignore[method-assign]
    purged = service.tenancy.purge_expired()
    assert list(purged) == [SHOP_A]


def test_audit_never_contains_tokens(service: AssistService):
    service.tenancy.onboard(_event(service, refresh_token=REFRESH))
    service.tenancy.set_status(NEW, ShopStatus.ACTIVE, actor="ops")
    service.tenancy.handle_subscription(_event(service, "renewed", refresh_token=REFRESH))
    service.tenancy.handle_subscription(_event(service, "cancelled"))
    _no_token_text(service)
    auth = ShopAuth(
        shop_id=NEW,
        access_token=SecretStr(TOKEN),
        scopes=sorted(MVP_SCOPES),
        expires_at=service.unit.clock.now(),
    )
    assert TOKEN not in repr(auth)
