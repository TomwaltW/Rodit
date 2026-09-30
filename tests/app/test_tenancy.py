"""店铺生命周期（方案 §8.4 开通 / 续订 / 取消订购 / 授权过期，§9.1 租户开通，§10 shop）。

AssistService.tenancy 跑在演示单元（内存仓储、合成数据）上：开通后能出推荐、权限和平台检查、
状态切换、订购事件、取消订购后的系统事件、过期清理。令牌全部是合成的。
"""

from __future__ import annotations

import json
import threading
import time
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
from rodit.app.token_lifecycle import RefreshOutcome
from rodit.core.audit import AuditAction
from rodit.core.config import EngineConfig, TenancyConfig
from rodit.core.enums import Platform, ShopStatus
from rodit.core.errors import ActionNotAllowed, AuthExpired, ShopInactive, TokenDecryptionError
from rodit.core.models import ShopAuth, SubscriptionEvent
from rodit.core.scopes import MVP_SCOPES, PHASE2_SCOPES, SCOPE_REFUND_WRITE
from rodit.engine.engine import REASON_SHOP_INACTIVE
from rodit.testing.fakes import FakeTokenRefresher, parse_subscription, subscription_push
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
    cancel = _event(service, "cancelled", SHOP_A)
    outcome = service.tenancy.handle_subscription(cancel)
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
    assert set(record.detail) == {"shop_id", "reason", "counts", "event_at"}
    assert record.detail == {
        "shop_id": SHOP_A,
        "reason": "cancelled",
        "counts": outcome.counts,
        "event_at": cancel.at.isoformat(),  # 触发清理的平台事件时间
    }


def test_auth_revoked_is_handled_like_cancel(service: AssistService):
    outcome = service.tenancy.handle_subscription(_event(service, "auth_revoked", SHOP_B))
    assert outcome.action == "revoked"
    [record] = service.unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.detail["reason"] == "auth_revoked" and record.detail["shop_id"] == SHOP_B


def test_subscribed_and_renewed_events_onboard(service: AssistService):
    outcome = service.tenancy.handle_subscription(_event(service))
    assert outcome.action == "granted" and outcome.shop.status == ShopStatus.TRIAL
    service.unit.clock.advance(days=1)  # 续订晚于开通
    renewed = service.tenancy.handle_subscription(_event(service, "renewed"))
    assert renewed.action == "granted"
    grants = service.unit.repos.audit.list_events(NEW, action=AuditAction.AUTH_GRANTED)
    assert [g.actor for g in grants] == [PLATFORM_ACTOR, PLATFORM_ACTOR]


def test_same_shop_can_onboard_again_after_revoke(service: AssistService):
    unit = service.unit
    service.tenancy.onboard(_event(service))
    _add_faq(service)
    assert _ask(service, FAQ_Q, msg_id="m1").suggestions
    unit.clock.advance(minutes=30)  # 取消晚于开通
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


# ---------------------------------------------------------------- 订购事件的顺序（平台重推、晚到）


def _handle(service: AssistService, event: SubscriptionEvent):
    return service.tenancy.handle_subscription(event)


def _purge_records(service: AssistService) -> list:
    return service.unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)


def test_replayed_cancel_after_resubscribe_is_ignored(service: AssistService):
    """商家误取消（T1）后重新订购（T2）；平台重推 T1 的取消：不删重新订购的店铺
    （令牌、知识、推荐记录），也不再写清理记录。"""
    unit = service.unit
    assert _handle(service, _event(service)).action == "granted"  # T0
    unit.clock.advance(hours=1)
    cancel = _event(service, "cancelled")  # T1
    assert _handle(service, cancel).action == "revoked"
    unit.clock.advance(hours=1)
    again = _event(service, access_token="tok-after-resubscribe")  # T2
    assert _handle(service, again).action == "granted"
    _add_faq(service)
    assert _ask(service, FAQ_Q, msg_id="m1").suggestions
    unit.clock.advance(hours=1)

    replay = _handle(service, cancel)
    assert replay.action == "ignored" and replay.counts == {}
    auth = unit.repos.auth.get(NEW)
    assert auth is not None and auth.access_token.get_secret_value() == "tok-after-resubscribe"
    assert unit.auth.is_authorized(NEW)
    shown_before = len(unit.repos.suggestions.list(NEW))
    assert shown_before > 0  # 重新订购之后的推荐记录还在
    assert _ask(service, FAQ_Q, msg_id="m2").suggestions[0].text == FAQ_A  # 知识还在
    assert len(_purge_records(service)) == 1
    # 重推 T2 的开通（已经处理过）：也是旧事件
    assert _handle(service, again).action == "ignored"


def test_cancel_older_than_the_current_grant_is_ignored(service: AssistService):
    """取消订购（T1）的推送晚于之后的重新授权（T2）才到：按平台事件时间是旧事件，不清理。"""
    unit = service.unit
    _handle(service, _event(service))
    unit.clock.advance(hours=1)
    late_cancel = _event(service, "cancelled")  # 发出之后没送到，平台稍后重推
    unit.clock.advance(hours=1)
    renewed = _event(service, "renewed", access_token="tok-renewed-later")
    assert _handle(service, renewed).action == "granted"
    _add_faq(service)

    assert _handle(service, late_cancel).action == "ignored"
    assert unit.auth.is_authorized(NEW)
    assert unit.repos.auth.get(NEW).access_token.get_secret_value() == "tok-renewed-later"
    assert _ask(service, FAQ_Q, msg_id="m1").suggestions
    assert _purge_records(service) == []


def test_cancel_retried_after_a_failed_purge_does_not_wipe_a_resubscribed_shop(
    service: AssistService,
):
    """取消订购第一次清理失败（平台会重推）；商家随后重新订购；平台重推那条旧的取消：不清理。"""
    unit = service.unit
    _handle(service, _event(service))
    unit.clock.advance(hours=1)
    cancel = _event(service, "cancelled")

    class FailsOnce:
        failed = False

        def purge_shop(self, shop_id: str) -> int:
            if not self.failed:
                self.failed = True
                raise RuntimeError("磁盘满了（模拟）")
            return 0

    unit.extra_purgers.append(FailsOnce())
    with pytest.raises(RuntimeError):
        _handle(service, cancel)
    assert _purge_records(service) == []  # 没删完：要重试
    unit.clock.advance(hours=2)
    assert _handle(service, _event(service, access_token="tok-resubscribed")).action == "granted"
    _add_faq(service)

    assert _handle(service, cancel).action == "ignored"
    assert unit.auth.is_authorized(NEW)
    assert _ask(service, FAQ_Q, msg_id="m1").suggestions


def test_older_grant_arriving_after_a_newer_one_keeps_the_newer_token(service: AssistService):
    unit = service.unit
    old = _event(service, access_token="tok-old", refresh_token="tok-refresh-old")  # T0
    unit.clock.advance(hours=1)
    newer_expiry = unit.clock.now() + timedelta(days=60)
    new = _event(
        service,
        "renewed",
        access_token="tok-new",
        refresh_token="tok-refresh-new",
        expires_at=newer_expiry,
    )
    assert _handle(service, new).action == "granted"

    assert _handle(service, old).action == "ignored"  # T0 的开通推送晚到
    auth = unit.repos.auth.get(NEW)
    assert auth.access_token.get_secret_value() == "tok-new"
    assert auth.refresh_token.get_secret_value() == "tok-refresh-new"
    assert auth.expires_at == newer_expiry and auth.granted_at == new.at


def test_replayed_grant_after_a_refresh_keeps_the_rotated_tokens(service: AssistService):
    """续期换了令牌（平台同时换了续期令牌、旧的作废）之后，平台重推当初那条开通：不拿旧令牌覆盖。"""
    unit = service.unit
    unit.token_refresher = FakeTokenRefresher(Platform.TAOBAO, clock=unit.clock)
    grant = _event(service, refresh_token=REFRESH, expires_at=unit.clock.now() + timedelta(hours=2))
    assert _handle(service, grant).action == "granted"
    assert service.tenancy.refresh_tokens().of(RefreshOutcome.REFRESHED) == [NEW]
    rotated = unit.repos.auth.get(NEW)
    assert rotated.access_token.get_secret_value() != TOKEN

    assert _handle(service, grant).action == "ignored"
    assert unit.repos.auth.get(NEW) == rotated  # 令牌、续期令牌、版本都没变


def test_resubscribe_sent_after_the_cancel_is_granted_even_when_the_purge_ran_long(
    service: AssistService,
):
    """大店铺清理要十分钟；商家取消 2 分钟后重新订购，推送在清理完之后才到：按事件时间比，
    重新订购晚于取消，照常开通（不拿清理做完的时间比）。"""
    unit = service.unit
    cancel_at = unit.clock.now()

    class BigShop:
        def purge_shop(self, shop_id: str) -> int:
            unit.clock.advance(minutes=10)
            return 0

    unit.extra_purgers.append(BigShop())
    assert _handle(service, _event(service, "cancelled", SHOP_A, at=cancel_at)).action == "revoked"
    resub = _event(
        service,
        "subscribed",
        SHOP_A,
        at=cancel_at + timedelta(minutes=2),
        access_token="tok-resubscribed-a",
    )
    assert _handle(service, resub).action == "granted"
    assert unit.repos.auth.get(SHOP_A).access_token.get_secret_value() == "tok-resubscribed-a"


def test_resubscribe_newer_than_a_late_processed_cancel_is_granted(service: AssistService):
    """取消（T0）晚了 30 秒才处理，重新订购（T0 + 10 秒）在那之后才到：照常开通。"""
    unit = service.unit
    cancel_at = unit.clock.now()
    unit.clock.advance(seconds=10)
    resub_at = unit.clock.now()
    unit.clock.advance(seconds=20)
    assert _handle(service, _event(service, "cancelled", SHOP_A, at=cancel_at)).action == "revoked"
    resub = _event(service, "subscribed", SHOP_A, at=resub_at, access_token="tok-resub-late")
    assert _handle(service, resub).action == "granted"
    assert unit.auth.is_authorized(SHOP_A)


def test_stale_grant_arriving_while_the_cancel_is_purging_is_ignored(service: AssistService):
    """取消订购正在清理（大店铺）时，取消之前发出的开通推送到了：在写方闸门里比较，
    清理记录写在放开闸门之前，所以它看得到清理记录，不会在清理之后把旧令牌写回来。"""
    unit = service.unit
    sent_at = unit.clock.now()
    unit.clock.advance(minutes=5)
    started, release = threading.Event(), threading.Event()

    class Slow:
        def purge_shop(self, shop_id: str) -> int:
            started.set()
            release.wait(5)
            return 0

    unit.extra_purgers.append(Slow())
    out: dict[str, object] = {}
    cancel = _event(service, "cancelled", SHOP_A)
    stale = _event(service, "subscribed", SHOP_A, at=sent_at, access_token="tok-stale")
    first = threading.Thread(target=lambda: out.setdefault("cancel", _handle(service, cancel)))
    first.start()
    assert started.wait(5)
    second = threading.Thread(target=lambda: out.setdefault("stale", _handle(service, stale)))
    second.start()
    time.sleep(0.1)
    assert "stale" not in out  # 等在写方闸门外
    release.set()
    first.join(5)
    second.join(5)
    assert out["cancel"].action == "revoked" and out["stale"].action == "ignored"
    assert unit.repos.auth.get(SHOP_A) is None and unit.repos.knowledge.get_shop(SHOP_A) is None


def test_stale_cancel_arriving_while_a_newer_grant_is_written_is_ignored(service: AssistService):
    """重新订购正在写（持有写方闸门）时，更早的取消推送到了：在清理的闸门里判断，不删新授权。"""
    unit = service.unit
    cancel = _event(service, "cancelled")  # T0
    unit.clock.advance(minutes=5)
    grant = _event(service, access_token="tok-regrant")  # T0 + 5 分钟
    entered, release = threading.Event(), threading.Event()
    knowledge = unit.repos.knowledge
    real_upsert = knowledge.upsert_shop

    def slow_upsert(shop):
        entered.set()
        release.wait(5)
        return real_upsert(shop)

    knowledge.upsert_shop = slow_upsert  # type: ignore[method-assign]
    out: dict[str, object] = {}
    first = threading.Thread(target=lambda: out.setdefault("grant", _handle(service, grant)))
    first.start()
    assert entered.wait(5)
    second = threading.Thread(target=lambda: out.setdefault("cancel", _handle(service, cancel)))
    second.start()
    time.sleep(0.1)
    release.set()
    first.join(5)
    second.join(5)
    knowledge.upsert_shop = real_upsert  # type: ignore[method-assign]
    assert out["grant"].action == "granted" and out["cancel"].action == "ignored"
    assert unit.auth.is_authorized(NEW) and knowledge.get_shop(NEW) is not None


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
        # 不是平台事件触发的：删掉的授权没有开通事件时间（直接写入的），按开始清理的时间
        "event_at": unit.clock.now().isoformat(),
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


# ================================================================ 开始清理的标记、没做完的清理


class FailsOnce:
    """第一次清理时抛错（模拟清理到一半失败：令牌已经删了，数据没删完），之后正常。"""

    def __init__(self) -> None:
        self.failed = False

    def purge_shop(self, shop_id: str) -> int:
        if not self.failed:
            self.failed = True
            raise RuntimeError("磁盘满了（模拟）")
        return 0


def _markers(service: AssistService) -> list:
    return service.unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGE_STARTED)


def _grant_a(service: AssistService, **kw) -> SubscriptionEvent:
    """店铺 A 经平台推送开通（记下平台开通时间 granted_at）。"""
    grant = _event(service, "subscribed", SHOP_A, **kw)
    assert _handle(service, grant).action == "granted"
    return grant


def test_purge_writes_a_start_marker_before_deleting_the_token(service: AssistService):
    """开始清理的标记（系统事件）在写方闸门里、删令牌之前写；做完之后清理记录不早于它。"""
    unit = service.unit
    seen: dict[str, object] = {}
    real_delete = unit.repos.auth.delete

    def delete(shop_id: str) -> bool:
        seen["markers"] = [m.detail for m in _markers(service)]
        return real_delete(shop_id)

    unit.repos.auth.delete = delete  # type: ignore[method-assign]
    cancel = _event(service, "cancelled", SHOP_A)
    assert _handle(service, cancel).action == "revoked"
    assert seen["markers"] == [
        {"shop_id": SHOP_A, "reason": "cancelled", "event_at": cancel.at.isoformat()}
    ]
    [marker] = _markers(service)
    assert marker.shop_id is None and marker.actor == PLATFORM_ACTOR
    [record] = _purge_records(service)
    assert record.created_at >= marker.created_at
    assert service.tenancy.unfinished_purges() == []  # 做完了
    assert unit.repos.audit.list_events(SHOP_A) == []  # 标记不属于店铺，清理不删它


def test_failed_expiry_purge_is_finished_by_the_next_round(service: AssistService):
    """过期清理删了令牌之后出错（没写 shop.purged，purge_expired 再也列不出它）：unfinished_purges
    列出来，下一轮 purge_expired 接着删完，原因和 event_at（那份授权的开通时间）沿用。"""
    unit = service.unit
    grant = _grant_a(service)  # T0
    _expire(service, SHOP_A, 40)
    unit.extra_purgers.append(FailsOnce())
    with pytest.raises(PurgeIncomplete):
        service.tenancy.purge_expired()
    assert unit.repos.auth.stamp(SHOP_A) is None and _purge_records(service) == []
    assert service.tenancy.expired_shops() == []
    [pending] = service.tenancy.unfinished_purges()
    assert (pending.shop_id, pending.reason, pending.event_at) == (
        SHOP_A,
        PURGE_REASON_EXPIRED,
        grant.at,
    )
    assert service.tenancy.purge_expired(dry_run=True) == {}  # dry_run 只列过期的，不删

    unit.clock.advance(hours=6)  # R：下一轮
    purged = service.tenancy.purge_expired()
    assert list(purged) == [SHOP_A]
    [record] = _purge_records(service)
    assert record.detail["reason"] == PURGE_REASON_EXPIRED and record.actor == "system"
    assert record.detail["event_at"] == grant.at.isoformat()  # 不是重试时的本机时间 R
    assert service.tenancy.unfinished_purges() == []
    assert service.tenancy.purge_expired() == {}  # 做完了，不再重试


def test_retry_after_a_partial_purge_keeps_the_grant_time_as_event_at(service: AssistService):
    """复现（修复前）：运维取消订购删了令牌之后出错；重试（revoke 或下一轮）时授权已经不在，
    清理记录的 event_at 退回重试时的本机时间 R。商家在 T0 和 R 之间真的重新订购了、推送在重试
    之后才到，就被当成旧事件忽略。现在沿用开始清理的标记里的 T0，照常开通。"""
    unit = service.unit
    grant = _grant_a(service)  # T0
    unit.clock.advance(hours=1)
    unit.extra_purgers.append(FailsOnce())
    with pytest.raises(RuntimeError):
        service.tenancy.revoke(SHOP_A, reason="ops", actor="ops")
    unit.clock.advance(hours=2)
    resub = _event(service, "subscribed", SHOP_A, access_token="tok-resub-between")  # P
    unit.clock.advance(hours=3)
    service.tenancy.revoke(SHOP_A, reason="ops", actor="ops")  # R：重试
    [record] = _purge_records(service)
    assert record.detail["event_at"] == grant.at.isoformat()

    assert _handle(service, resub).action == "granted"  # T0 < P < R，推送在 R 之后才到
    assert unit.auth.is_authorized(SHOP_A)


def test_replayed_grant_between_a_failed_purge_and_its_retry_is_ignored(service: AssistService):
    """清理删了令牌之后出错、还没接着删完：被删的那份授权（T0）的开通推送晚到，不写回来
    （开始清理的标记的 event_at 挡住）。"""
    unit = service.unit
    grant = _grant_a(service, access_token="tok-before-purge")
    unit.clock.advance(hours=1)
    unit.extra_purgers.append(FailsOnce())
    with pytest.raises(RuntimeError):
        service.tenancy.revoke(SHOP_A, reason="ops", actor="ops")
    assert _handle(service, grant).action == "ignored"
    assert unit.repos.auth.stamp(SHOP_A) is None
    [pending] = service.tenancy.unfinished_purges()
    assert (pending.shop_id, pending.reason) == (SHOP_A, "ops")


def test_replayed_cancel_after_a_failed_purge_finishes_it(service: AssistService):
    """取消订购不拿开始清理的标记比：清理没做完时平台重推同一条取消，照常接着删完。"""
    unit = service.unit
    cancel = _event(service, "cancelled", SHOP_A)
    unit.extra_purgers.append(FailsOnce())
    with pytest.raises(RuntimeError):
        _handle(service, cancel)
    assert _handle(service, cancel).action == "revoked"
    [record] = _purge_records(service)
    assert record.detail["event_at"] == cancel.at.isoformat()
    assert service.tenancy.unfinished_purges() == []
    assert unit.repos.knowledge.get_shop(SHOP_A) is None


def test_unfinished_purge_is_not_resumed_once_the_shop_subscribed_again(service: AssistService):
    """清理没做完、商家随后重新订购：新授权、新数据都不删（列出时不算没做完；列出之后才重新
    开通的，在写方闸门里确认还没有授权时也不删）。"""
    unit = service.unit
    _grant_a(service)
    _expire(service, SHOP_A, 40)
    unit.extra_purgers.append(FailsOnce())
    with pytest.raises(PurgeIncomplete):
        service.tenancy.purge_expired()
    listed = service.tenancy.unfinished_purges()
    assert [p.shop_id for p in listed] == [SHOP_A]

    unit.clock.advance(hours=1)
    resub = _event(service, "subscribed", SHOP_A, access_token="tok-resubscribed-a")
    assert _handle(service, resub).action == "granted"  # 晚于那次清理的开通：照常开通
    _add_faq(service, SHOP_A)
    assert service.tenancy.unfinished_purges() == []

    unit.clock.advance(hours=6)
    tenancy = service.tenancy
    tenancy.unfinished_purges = lambda: listed  # type: ignore[method-assign]  # 列出之后才重新开通
    assert tenancy.purge_expired() == {}
    assert unit.auth.is_authorized(SHOP_A)
    assert unit.repos.auth.get(SHOP_A).access_token.get_secret_value() == "tok-resubscribed-a"
    assert _purge_records(service) == []
    assert _ask(service, FAQ_Q, msg_id="m1", shop_id=SHOP_A).suggestions[0].text == FAQ_A


def test_marker_at_the_same_instant_as_the_previous_purge_still_counts(service: AssistService):
    """上一次清理刚做完、同一时刻又清理这家店且失败：新的标记写在上一条清理记录之后，
    照样算没做完（按写入时间判断，不会被上一条清理记录当成做完）。"""
    unit = service.unit
    service.tenancy.revoke(SHOP_A, reason="ops", actor="ops")
    unit.extra_purgers.append(FailsOnce())
    with pytest.raises(RuntimeError):
        service.tenancy.revoke(SHOP_A, reason="again", actor="ops")  # 时钟没走
    first, second = _markers(service)
    [record] = _purge_records(service)
    assert second.created_at > record.created_at >= first.created_at
    [pending] = service.tenancy.unfinished_purges()
    assert (pending.shop_id, pending.reason) == (SHOP_A, "again")
    assert list(service.tenancy.purge_expired()) == [SHOP_A]
    assert service.tenancy.unfinished_purges() == []


def test_marker_from_a_process_with_a_faster_clock_is_closed_by_the_next_round(
    service: AssistService,
):
    """没做完的那次清理是时钟快 10 秒的进程写的标记；接着删的进程时钟慢：它的标记、清理记录
    照样排在那条标记之后，一轮就做完（不会因为写入时间倒挂一直重试）。"""
    unit = service.unit
    slow = unit.clock.now()
    unit.clock.set(slow + timedelta(seconds=10))  # 时钟快的进程
    unit.extra_purgers.append(FailsOnce())
    with pytest.raises(RuntimeError):
        service.tenancy.revoke(SHOP_A, reason="ops", actor="ops")
    unit.clock.set(slow)  # 时钟慢的进程接着删
    assert list(service.tenancy.purge_expired()) == [SHOP_A]
    first, second = _markers(service)
    [record] = _purge_records(service)
    assert record.created_at >= second.created_at > first.created_at
    assert service.tenancy.unfinished_purges() == []
    assert service.tenancy.purge_expired() == {}


# ================================================================ 运维开通的平台开通时间


def test_ops_onboarding_does_not_stamp_local_time_as_the_grant_time(service: AssistService):
    """复现（修复前）：直接调 onboard（运维开通）把本机时间 N 记成平台开通时间；商家在 N 之前
    取消订购、推送在 N 之后才到，被当成旧事件忽略，数据留到过期清理（约 60 天）。现在不给
    granted_at 时不记（不参与比较），这条取消照常删除；给了就按平台开通时间比。"""
    unit = service.unit
    service.tenancy.onboard(_event(service), actor="ops")  # N
    assert unit.repos.auth.stamp(NEW).granted_at is None
    early_cancel = _event(service, "cancelled", at=unit.clock.now() - timedelta(minutes=10))
    assert _handle(service, early_cancel).action == "revoked"
    assert unit.repos.auth.get(NEW) is None and unit.repos.knowledge.get_shop(NEW) is None

    unit.clock.advance(hours=1)
    p0 = unit.clock.now() - timedelta(minutes=30)  # 服务市场订单上的开通时间
    service.tenancy.onboard(_event(service), actor="ops", granted_at=p0)
    assert unit.repos.auth.stamp(NEW).granted_at == p0
    older = _event(service, "cancelled", at=p0 - timedelta(seconds=1))
    assert _handle(service, older).action == "ignored"
    assert unit.auth.is_authorized(NEW)
    # 平台推送的开通照旧记事件时间
    unit.clock.advance(hours=1)
    renewed = _event(service, "renewed")
    assert _handle(service, renewed).action == "granted"
    assert unit.repos.auth.stamp(NEW).granted_at == renewed.at


def test_audit_never_contains_tokens(service: AssistService):
    service.tenancy.onboard(_event(service, refresh_token=REFRESH))
    service.tenancy.set_status(NEW, ShopStatus.ACTIVE, actor="ops")
    service.unit.clock.advance(days=1)
    renewed = _event(service, "renewed", refresh_token=REFRESH)
    assert service.tenancy.handle_subscription(renewed).action == "granted"
    service.unit.clock.advance(days=1)
    assert service.tenancy.handle_subscription(_event(service, "cancelled")).action == "revoked"
    _no_token_text(service)
    auth = ShopAuth(
        shop_id=NEW,
        access_token=SecretStr(TOKEN),
        scopes=sorted(MVP_SCOPES),
        expires_at=service.unit.clock.now(),
    )
    assert TOKEN not in repr(auth)
