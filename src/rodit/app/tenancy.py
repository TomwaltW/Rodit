"""店铺生命周期（方案 §8.4 开通 / 续订 / 取消订购 / 授权过期，§9.1 租户开通，
§10 shop / shop_auth）。

- onboard：订购开通 / 续订（SubscriptionEvent，kind 为 subscribed / renewed）——核对平台是本单元的
  平台（§9.1 数据不跨云）、权限不超出 TenancyConfig.onboarding_phase 允许的范围（§8.4 最小权限，
  超出时报出多出来的权限）、也不缺 MVP 必需的只读权限；新店铺写 Shop（默认 TRIAL），已有店铺
  保留状态和配置；保存加密的授权（ShopAuth，含续期令牌，needs_reauth 清掉）；审计 auth.granted
  （权限、到期时间，不记令牌）。同一家店取消订购（数据已删）之后可以再开通；
- set_status：trial → active → suspended → active（STATUS_TRANSITIONS），审计 shop.status_changed；
  停用（SUSPENDED）的店铺问答、控制台一律停止处理（引擎、接口层按 ShopInactive 处理）；
- handle_subscription：平台推送的订购事件（BindingContext.on_subscription、
  POST /v1/platform/{access}/subscription）：subscribed / renewed → onboard；
  cancelled / auth_revoked → 取消订购（revoke）+ 系统事件 shop.purged；
  按平台的事件时间（SubscriptionEvent.at）排序：这家店已经处理过不早于它的订购事件（现在这份授权的
  ShopAuth.granted_at、清理记录 shop.purged 的 event_at），它就是晚到或重推的旧事件，什么都不做
  （ignored）——旧的取消订购不会删掉之后重新订购的店铺，旧的开通不会盖掉新令牌，也不会把取消之前的
  授权写回来。比较和写入在同一次写方闸门里做（开通在 exclusive 里；取消订购在 revoke_shop 的闸门里
  判断、清理、写 shop.purged），期间插不进这家店的别的订购事件；
- revoke：取消订购（AssistService.revoke_shop：删令牌、停止处理、删除该店铺在本单元的全部数据）
  + 系统事件 shop.purged（shop_id 为空，detail 只有店铺 ID、原因、各项删除条数和 event_at；清理之后
  还在，AuditLog.list_system_events 读）。event_at 是触发清理的平台事件时间；不是平台事件触发的
  （运维取消订购、过期清理）用删掉的那份授权的开通事件时间（只比平台时间和平台时间：挡住那次开通
  和更早的重推，之后平台真有新的开通照样开通），不知道时用开始清理的时间；
- purge_expired：授权过期超过 TenancyConfig.purge_after_expiry_days 的店铺 → revoke
  （原因 expired；是否还过期在写方闸门里再确认）；dry_run 只列出、不删；
- refresh_tokens：令牌续期（rodit.app.token_lifecycle）。

改店铺记录（开通、切换状态）在 exclusive(shop_id)（写方闸门，和取消订购同一把）里做：等在途处理
做完，读改写之间不会插进同一进程里的其他修改。revoke 自己取写方闸门，不能在任何闸门里调。
直接调 onboard（运维开通）不比较事件时间（照样记下 granted_at）。
令牌不进日志、审计和异常消息（SecretStr）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Protocol

from rodit.adapters.auth import excess_scopes
from rodit.app.token_lifecycle import RefreshReport, refresh_expiring_tokens
from rodit.app.unit import Unit
from rodit.core.audit import AuditAction
from rodit.core.enums import ShopStatus
from rodit.core.errors import ActionNotAllowed, RoditError, TokenDecryptionError
from rodit.core.models import AuditEvent, Shop, ShopAuth, SubscriptionEvent, SubscriptionKind
from rodit.core.scopes import MVP_SCOPES

logger = logging.getLogger(__name__)

ShopGate = Callable[[str], AbstractContextManager[None]]
"""AssistService.processing / exclusive：按店铺的读方 / 写方闸门（和取消订购互斥）。"""


class Revoke(Protocol):
    """AssistService.revoke_shop：删令牌、停止处理、按约定删除该店铺在本单元的全部数据。

    guard、on_purged 都在这家店的写方闸门里调：guard 返回 False 时什么都不删、返回 None；
    on_purged(各项删除条数) 在删完之后、放开闸门之前调（写清理记录）。"""

    def __call__(
        self,
        shop_id: str,
        *,
        guard: Callable[[], bool] | None = None,
        on_purged: Callable[[dict[str, int]], None] | None = None,
    ) -> dict[str, int] | None: ...


SYSTEM_ACTOR = "system"
PLATFORM_ACTOR = "platform"  # 平台推送的订购事件

GRANT_KINDS: frozenset[str] = frozenset({"subscribed", "renewed"})
REVOKE_KINDS: frozenset[str] = frozenset({"cancelled", "auth_revoked"})

PURGE_REASON_EXPIRED = "expired"
"""shop.purged 的原因：授权过期超过宽限期（purge_expired）。取消订购用事件的 kind
（cancelled / auth_revoked），运维命令默认 ops，商家在控制台取消用 PURGE_REASON_MERCHANT。"""

PURGE_REASON_MERCHANT = "merchant_revoke"
"""shop.purged 的原因：商家经 POST /v1/shops/{shop_id}/revoke 取消（操作人是商家账号）。"""

_EVENT_CLOCK_SKEW = timedelta(days=1)
"""找清理记录时按写入时间（本单元的时钟）预先筛选的余量：清理记录的 event_at 是平台的事件时间，
写入时间在它之后（清理做完才写），只有平台时钟比本单元快时才会早一点。"""

STATUS_TRANSITIONS: dict[ShopStatus, frozenset[ShopStatus]] = {
    ShopStatus.TRIAL: frozenset({ShopStatus.ACTIVE, ShopStatus.SUSPENDED}),
    ShopStatus.ACTIVE: frozenset({ShopStatus.SUSPENDED}),
    ShopStatus.SUSPENDED: frozenset({ShopStatus.ACTIVE, ShopStatus.TRIAL}),
}
"""允许的店铺状态切换（§10 shop.status）：试用 → 正式 → 停用 → 恢复（停用前是试用的可以回到试用）。
正式不能退回试用。"""

ONBOARD_STATUSES: frozenset[ShopStatus] = frozenset({ShopStatus.TRIAL, ShopStatus.ACTIVE})

SubscriptionAction = Literal["granted", "revoked", "ignored"]


@dataclass(frozen=True)
class SubscriptionOutcome:
    """handle_subscription 做了什么：granted（开通 / 续订，shop 是店铺现状）、
    revoked（取消订购，counts 是各项删除条数）、ignored（晚到的旧事件）。没有令牌。"""

    kind: SubscriptionKind
    shop_id: str
    action: SubscriptionAction
    shop: Shop | None = None
    counts: dict[str, int] = field(default_factory=dict)


class PurgeIncomplete(RoditError):
    """过期清理有店铺没删完（令牌可能已删，数据还在）：purged 是删完的店铺 → 条数，
    failed 是要用 revoke --shop-id 重试的店铺（purge_shop 是幂等的）。消息里只有店铺 ID。"""

    def __init__(self, purged: dict[str, dict[str, int]], failed: list[str]) -> None:
        self.purged = purged
        self.failed = failed
        super().__init__(f"过期清理没有完成的店铺：{', '.join(failed)}（请用 revoke 重试）")


class TenancyService:
    """店铺生命周期（§8.4）。AssistService.tenancy。"""

    def __init__(
        self,
        unit: Unit,
        processing: ShopGate,
        revoke: Revoke,
        *,
        exclusive: ShopGate | None = None,
    ) -> None:
        self._unit = unit
        self._processing = processing
        self._revoke = revoke
        # 没给写方闸门时退回读方闸门（仍和取消订购互斥，只是不和在途处理互斥）
        self._exclusive = exclusive or processing

    # ================================================================ 开通 / 续订

    def onboard(
        self,
        event: SubscriptionEvent,
        *,
        actor: str = SYSTEM_ACTOR,
        status: ShopStatus | None = None,
    ) -> Shop:
        """订购开通 / 续订：写店铺和授权，返回店铺现状。

        status：新店铺的状态（默认 TRIAL）；已有店铺默认保留原状态，给了就按 STATUS_TRANSITIONS
        切换（不允许的切换抛 ActionNotAllowed）。只能是 TRIAL / ACTIVE。
        平台不对、权限超出本阶段或缺 MVP 必需的权限、不是开通 / 续订事件都抛 ValueError，
        什么都不写。
        """
        actor = _require_actor(actor)
        self._check_grant(event, status)
        with self._exclusive(event.shop_id):
            shop, created = self._onboard_locked(event, actor=actor, status=status)
        _log_granted(event, created)
        return shop

    def _check_grant(self, event: SubscriptionEvent, status: ShopStatus | None) -> None:
        """开通 / 续订事件本身的检查（不读库）：事件类型、平台、开通时的状态。"""
        if event.kind not in GRANT_KINDS:
            raise ValueError(f"{event.kind} 不是开通 / 续订事件")
        self._check_platform(event)
        if status is not None and ShopStatus(status) not in ONBOARD_STATUSES:
            raise ValueError("开通时店铺状态只能是 trial 或 active")

    def _onboard_locked(
        self, event: SubscriptionEvent, *, actor: str, status: ShopStatus | None
    ) -> tuple[Shop, bool]:
        """写店铺和授权：调用方持有这家店的写方闸门，事件已经过 _check_grant。
        返回 (店铺现状, 是否新建)。"""
        # 模型校验保证开通 / 续订事件带令牌和到期时间
        assert event.access_token is not None and event.expires_at is not None
        shop_id = event.shop_id
        now = self._unit.clock.now()
        scopes = self._granted_scopes(event)
        knowledge = self._unit.repos.knowledge
        existing = knowledge.get_shop(shop_id)
        created = existing is None
        if existing is None:
            shop = Shop(
                id=shop_id,
                platform=self._unit.platform,
                category=event.category,
                status=ShopStatus(status or ShopStatus.TRIAL),
            )
        else:
            if existing.platform != self._unit.platform:  # 纵深防御：库里不该有别的平台
                raise ValueError(f"店铺 {shop_id} 属于 {existing.platform}（§9.1 数据不跨云）")
            changes: dict[str, Any] = {}
            if event.category and event.category != existing.category:
                changes["category"] = event.category
            if status is not None and ShopStatus(status) != existing.status:
                _check_transition(shop_id, existing.status, ShopStatus(status))
                changes["status"] = ShopStatus(status)
            shop = existing.model_copy(update=changes)
        if created or shop != existing:
            knowledge.upsert_shop(shop)
        has_refresh = event.refresh_token is not None
        self._unit.repos.auth.save(
            ShopAuth(
                shop_id=shop_id,
                access_token=event.access_token,
                scopes=sorted(scopes),
                expires_at=event.expires_at,
                refresh_token=event.refresh_token,
                refresh_expires_at=event.refresh_expires_at if has_refresh else None,
                needs_reauth=False,
                granted_at=event.at,  # 订购事件顺序的高水位
            )
        )
        if existing is not None and existing.status != shop.status:
            self._record_status(shop_id, actor, existing.status, shop.status, now)
        self._record(
            shop_id,
            AuditAction.AUTH_GRANTED,
            actor,
            now,
            {
                "kind": event.kind,
                "created": created,
                "status": shop.status.value,
                "scopes": sorted(scopes),
                "expires_at": event.expires_at.isoformat(),
                "refresh_token": has_refresh,
                "refresh_expires_at": (
                    event.refresh_expires_at.isoformat()
                    if has_refresh and event.refresh_expires_at
                    else None
                ),
                "event_at": event.at.isoformat(),
            },
        )
        return shop, created

    def _granted_scopes(self, event: SubscriptionEvent) -> frozenset[str]:
        """这次授权的权限（§8.4 最小权限）：不能超出本阶段允许的范围，也不能缺 MVP 必需的只读权限。
        续订事件没带权限时沿用现有授权的权限（照样按本阶段检查）。"""
        scopes = frozenset(s.strip() for s in event.scopes if s.strip())
        if not scopes and event.kind == "renewed":
            try:
                current = self._unit.repos.auth.get(event.shop_id)
            except TokenDecryptionError:
                current = None
            if current is not None:
                scopes = frozenset(current.scopes)
        phase = self._unit.config.tenancy.onboarding_phase
        excess = excess_scopes(scopes, phase=phase)
        if excess:
            raise ValueError(
                f"店铺 {event.shop_id} 申请的权限超出第 {phase} 阶段允许的范围："
                f"{', '.join(sorted(excess))}（§8.4 最小权限）"
            )
        missing = MVP_SCOPES - scopes
        if missing:
            raise ValueError(
                f"店铺 {event.shop_id} 的授权缺少必需的权限：{', '.join(sorted(missing))}"
                "（§8.4 MVP 需要读订单、读物流、读售后、读商品）"
            )
        return scopes

    def _check_platform(self, event: SubscriptionEvent) -> None:
        if event.platform != self._unit.platform:
            raise ValueError(
                f"订购事件属于 {event.platform.value}，本单元是 {self._unit.platform.value}"
                "（§9.1 数据不跨云）"
            )

    # ================================================================ 店铺状态

    def set_status(self, shop_id: str, status: ShopStatus, *, actor: str) -> Shop:
        """切换店铺状态（STATUS_TRANSITIONS），返回更新后的店铺。店铺不存在抛 KeyError；
        不允许的切换抛 ActionNotAllowed；状态没变时什么都不写。"""
        actor = _require_actor(actor)
        target = ShopStatus(status)
        with self._exclusive(shop_id):
            knowledge = self._unit.repos.knowledge
            shop = knowledge.get_shop(shop_id)
            if shop is None:
                raise KeyError(shop_id)
            if shop.status == target:
                return shop
            _check_transition(shop_id, shop.status, target)
            updated = shop.model_copy(update={"status": target})
            knowledge.upsert_shop(updated)
            self._record_status(shop_id, actor, shop.status, target, self._unit.clock.now())
        logger.info("店铺 %s 状态 %s → %s", shop_id, shop.status.value, target.value)
        return updated

    # ================================================================ 订购事件

    def handle_subscription(self, event: SubscriptionEvent) -> SubscriptionOutcome:
        """平台推送的订购事件（开通、续订、取消订购、撤销授权），见模块说明。
        平台不对抛 ValueError；开通 / 续订的检查同 onboard。开通、取消订购都要取这家店的写方闸门：
        不能在处理这家店的事件时（持有 processing）调用。"""
        self._check_platform(event)
        shop_id = event.shop_id
        if event.kind in GRANT_KINDS:
            self._check_grant(event, None)
            with self._exclusive(shop_id):
                if self._superseded(event):
                    _log_stale(event)
                    return SubscriptionOutcome(kind=event.kind, shop_id=shop_id, action="ignored")
                shop, created = self._onboard_locked(event, actor=PLATFORM_ACTOR, status=None)
            _log_granted(event, created)
            return SubscriptionOutcome(
                kind=event.kind, shop_id=shop_id, action="granted", shop=shop
            )
        if event.kind in REVOKE_KINDS:
            counts = self._purge(
                shop_id,
                reason=event.kind,
                actor=PLATFORM_ACTOR,
                event_at=event.at,
                guard=lambda: not self._superseded(event),
            )
            if counts is None:
                _log_stale(event)
                return SubscriptionOutcome(kind=event.kind, shop_id=shop_id, action="ignored")
            return SubscriptionOutcome(
                kind=event.kind, shop_id=shop_id, action="revoked", counts=counts
            )
        raise ValueError(f"不认识的订购事件：{event.kind}")

    def _superseded(self, event: SubscriptionEvent) -> bool:
        """这家店已经处理过不早于 event 的订购事件（event 是晚到或重推的旧事件）：现在这份授权的
        开通事件时间（ShopAuth.granted_at），或者清理记录 shop.purged 的 event_at（以前的记录没有，
        按写入时间）不早于 event.at。只读时间，不解密令牌。调用方持有这家店的写方闸门。"""
        shop_id, at = event.shop_id, event.at
        stamp = self._unit.repos.auth.stamp(shop_id)
        if stamp is not None and stamp.granted_at is not None and stamp.granted_at >= at:
            return True
        purged = self._unit.repos.audit.list_system_events(
            action=AuditAction.SHOP_PURGED, since=at - _EVENT_CLOCK_SKEW
        )
        return any(e.detail.get("shop_id") == shop_id and _purge_event_at(e) >= at for e in purged)

    # ================================================================ 取消订购 / 过期清理

    def revoke(self, shop_id: str, *, reason: str, actor: str = SYSTEM_ACTOR) -> dict[str, int]:
        """取消订购：删令牌、停止处理、删除该店铺在本单元的全部数据（AssistService.revoke_shop），
        再写系统事件 shop.purged（清理之后还在；event_at 见模块说明）。返回各项删除条数。

        清理失败时异常向外抛，不写 shop.purged（令牌可能已删；purge_shop 是幂等的，重试即可）。
        不能在任何闸门里调（revoke_shop 自己取写方闸门）。"""
        counts = self._purge(shop_id, reason=reason, actor=actor)
        assert counts is not None  # 没有 guard：一定会删
        return counts

    def _purge(
        self,
        shop_id: str,
        *,
        reason: str,
        actor: str,
        event_at: datetime | None = None,
        guard: Callable[[], bool] | None = None,
    ) -> dict[str, int] | None:
        """revoke 的实现。guard 在写方闸门里判断，返回 False 时什么都不删、返回 None；
        shop.purged 在删完之后、放开闸门之前写：等在闸门外的订购事件一定看得到它。
        event_at：触发清理的平台事件时间；没有时（运维、过期清理）在闸门里、删之前读删掉的那份
        授权的开通事件时间，不知道时用开始清理的时间（见模块说明）。"""
        actor = _require_actor(actor)
        reason = (reason or "").strip()
        if not reason:
            raise ValueError("取消订购要写明原因")
        started = self._unit.clock.now()
        marks: list[datetime] = []  # 这次清理记下的 event_at（在闸门里定）

        def checked() -> bool:
            if guard is not None and not guard():
                return False
            if event_at is not None:
                marks.append(event_at)
            else:
                stamp = self._unit.repos.auth.stamp(shop_id)
                granted = stamp.granted_at if stamp is not None else None
                marks.append(granted or started)
            return True

        def record(counts: dict[str, int]) -> None:
            effective_at = marks[-1] if marks else (event_at or started)
            self._unit.repos.audit.record(
                AuditEvent(
                    shop_id=None,  # 系统事件：店铺的审计已经删了，这条要留下来
                    actor=actor,
                    action=AuditAction.SHOP_PURGED,
                    detail={
                        "shop_id": shop_id,
                        "reason": reason,
                        "counts": dict(counts),
                        "event_at": effective_at.isoformat(),
                    },
                    created_at=self._unit.clock.now(),
                )
            )

        try:
            result = self._revoke(shop_id, guard=checked, on_purged=record)
        except Exception as exc:
            logger.error(
                "取消订购没有完成（%s）：店铺 %s，请重试（清理是幂等的）",
                type(exc).__name__,
                shop_id,
            )
            raise
        if result is None:
            return None
        counts = dict(result)
        logger.info(
            "店铺 %s 已取消订购并删除数据（%s）：共 %d 条", shop_id, reason, sum(counts.values())
        )
        return counts

    def expired_shops(self, *, now: datetime | None = None) -> list[str]:
        """授权过期超过 purge_after_expiry_days 的店铺（按到期时间）。只读店铺 ID 和到期时间。"""
        cutoff = self._purge_cutoff(now)
        return [
            shop_id
            for shop_id, expires_at in self._unit.repos.auth.expiring_shop_ids(cutoff)
            if expires_at < cutoff
        ]

    def purge_expired(
        self, *, now: datetime | None = None, dry_run: bool = False
    ) -> dict[str, dict[str, int]]:
        """授权过期超过宽限期的店铺：取消订购并删除数据（原因 expired）。返回 店铺 → 各项删除条数
        （dry_run 时只列出店铺，条数为空）。

        删之前逐店在写方闸门里再确认一次还过期（列出之后续期成功、重新授权了的不删；确认和删除
        之间插不进同一进程里的开通）。有店铺没删完时其余照删，最后抛 PurgeIncomplete
        （带上删完的和没删完的）。"""
        now = now or self._unit.clock.now()
        due = self.expired_shops(now=now)
        if dry_run:
            return {shop_id: {} for shop_id in due}
        cutoff = self._purge_cutoff(now)
        purged: dict[str, dict[str, int]] = {}
        failed: list[str] = []
        for shop_id in due:
            try:
                counts = self._purge(
                    shop_id,
                    reason=PURGE_REASON_EXPIRED,
                    actor=SYSTEM_ACTOR,
                    guard=lambda s=shop_id: self._still_expired(s, cutoff),
                )
            except Exception:  # noqa: BLE001 — revoke 已记日志；其余店铺照删
                failed.append(shop_id)
                continue
            if counts is None:
                logger.info("过期清理：店铺 %s 列出之后授权变了，不删", shop_id)
                continue
            purged[shop_id] = counts
        if failed:
            raise PurgeIncomplete(purged, failed)
        return purged

    def _purge_cutoff(self, now: datetime | None) -> datetime:
        now = now or self._unit.clock.now()
        return now - timedelta(days=self._unit.config.tenancy.purge_after_expiry_days)

    def _still_expired(self, shop_id: str, cutoff: datetime) -> bool:
        try:
            auth = self._unit.repos.auth.get(shop_id)
        except TokenDecryptionError:
            return True  # 令牌读不出来：按列出时的到期时间处理
        return auth is not None and auth.expires_at < cutoff

    # ================================================================ 续期

    def refresh_tokens(self, *, now: datetime | None = None) -> RefreshReport:
        """令牌续期（rodit.app.token_lifecycle.refresh_expiring_tokens）。"""
        return refresh_expiring_tokens(self._unit, processing=self._processing, now=now)

    # ================================================================ 审计

    def _record_status(
        self, shop_id: str, actor: str, before: ShopStatus, after: ShopStatus, now: datetime
    ) -> None:
        self._record(
            shop_id,
            AuditAction.SHOP_STATUS_CHANGED,
            actor,
            now,
            {"from": before.value, "to": after.value},
        )

    def _record(
        self, shop_id: str, action: AuditAction, actor: str, now: datetime, detail: dict[str, Any]
    ) -> None:
        self._unit.repos.audit.record(
            AuditEvent(shop_id=shop_id, actor=actor, action=action, detail=detail, created_at=now)
        )


def _purge_event_at(record: AuditEvent) -> datetime:
    """清理记录对应的事件时间：detail 里的 event_at；以前的记录没有，按写入时间。"""
    raw = record.detail.get("event_at")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return record.created_at


def _log_granted(event: SubscriptionEvent, created: bool) -> None:
    logger.info("店铺 %s 已%s（%s）", event.shop_id, "开通" if created else "更新授权", event.kind)


def _log_stale(event: SubscriptionEvent) -> None:
    logger.warning(
        "订购事件（%s）不晚于这家店已处理的订购事件（晚到或重推）：店铺 %s 不处理",
        event.kind,
        event.shop_id,
    )


def _check_transition(shop_id: str, before: ShopStatus, after: ShopStatus) -> None:
    if after not in STATUS_TRANSITIONS[before]:
        raise ActionNotAllowed(f"店铺 {shop_id} 不能从 {before.value} 切换到 {after.value}")


def _require_actor(actor: str) -> str:
    actor = (actor or "").strip()
    if not actor:
        raise ValueError("操作人不能为空（§4.1 原则 6 一切留痕）")
    return actor


__all__ = [
    "GRANT_KINDS",
    "ONBOARD_STATUSES",
    "PLATFORM_ACTOR",
    "PURGE_REASON_EXPIRED",
    "PURGE_REASON_MERCHANT",
    "REVOKE_KINDS",
    "STATUS_TRANSITIONS",
    "SYSTEM_ACTOR",
    "PurgeIncomplete",
    "Revoke",
    "ShopGate",
    "SubscriptionAction",
    "SubscriptionOutcome",
    "TenancyService",
]
