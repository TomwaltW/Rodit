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
  事件时间早于这家店最近一次清理的开通 / 续订是晚到的旧事件，不再开通（ignored）；
- revoke：取消订购（AssistService.revoke_shop：删令牌、停止处理、删除该店铺在本单元的全部数据）
  + 系统事件 shop.purged（shop_id 为空，detail 只有店铺 ID、原因、各项删除条数；清理之后还在，
  AuditLog.list_system_events 读）；
- purge_expired：授权过期超过 TenancyConfig.purge_after_expiry_days 的店铺 → revoke
  （原因 expired）；dry_run 只列出、不删；
- refresh_tokens：令牌续期（rodit.app.token_lifecycle）。

改店铺记录（开通、切换状态）在 exclusive(shop_id)（写方闸门，和取消订购同一把）里做：等在途处理
做完，读改写之间不会插进同一进程里的其他修改。revoke 自己取写方闸门，不能在任何闸门里调。
令牌不进日志、审计和异常消息（SecretStr）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

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

Revoke = Callable[[str], dict[str, int]]
"""AssistService.revoke_shop：删令牌、停止处理、按约定删除该店铺在本单元的全部数据。"""

SYSTEM_ACTOR = "system"
PLATFORM_ACTOR = "platform"  # 平台推送的订购事件

GRANT_KINDS: frozenset[str] = frozenset({"subscribed", "renewed"})
REVOKE_KINDS: frozenset[str] = frozenset({"cancelled", "auth_revoked"})

PURGE_REASON_EXPIRED = "expired"
"""shop.purged 的原因：授权过期超过宽限期（purge_expired）。取消订购用事件的 kind
（cancelled / auth_revoked），运维命令默认 ops。"""

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
        if event.kind not in GRANT_KINDS:
            raise ValueError(f"{event.kind} 不是开通 / 续订事件")
        self._check_platform(event)
        if status is not None and ShopStatus(status) not in ONBOARD_STATUSES:
            raise ValueError("开通时店铺状态只能是 trial 或 active")
        # 模型校验保证开通 / 续订事件带令牌和到期时间
        assert event.access_token is not None and event.expires_at is not None
        shop_id = event.shop_id
        now = self._unit.clock.now()
        with self._exclusive(shop_id):
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
                },
            )
        logger.info("店铺 %s 已%s（%s）", shop_id, "开通" if created else "更新授权", event.kind)
        return shop

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
        if event.kind in GRANT_KINDS:
            if self._purged_after(event.shop_id, event.at):
                logger.warning(
                    "订购事件晚于清理到达（事件时间早于最近一次清理）：店铺 %s 不重新开通",
                    event.shop_id,
                )
                return SubscriptionOutcome(kind=event.kind, shop_id=event.shop_id, action="ignored")
            shop = self.onboard(event, actor=PLATFORM_ACTOR)
            return SubscriptionOutcome(
                kind=event.kind, shop_id=event.shop_id, action="granted", shop=shop
            )
        if event.kind in REVOKE_KINDS:
            counts = self.revoke(event.shop_id, reason=event.kind, actor=PLATFORM_ACTOR)
            return SubscriptionOutcome(
                kind=event.kind, shop_id=event.shop_id, action="revoked", counts=counts
            )
        raise ValueError(f"不认识的订购事件：{event.kind}")

    def _purged_after(self, shop_id: str, at: datetime) -> bool:
        """这家店在 at 之后清理过（开通 / 续订事件是清理之前发出的旧事件）。"""
        purged = self._unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED, since=at)
        return any(e.detail.get("shop_id") == shop_id and e.created_at > at for e in purged)

    # ================================================================ 取消订购 / 过期清理

    def revoke(self, shop_id: str, *, reason: str, actor: str = SYSTEM_ACTOR) -> dict[str, int]:
        """取消订购：删令牌、停止处理、删除该店铺在本单元的全部数据（AssistService.revoke_shop），
        再写系统事件 shop.purged（清理之后还在）。返回各项删除条数。

        清理失败时异常向外抛，不写 shop.purged（令牌可能已删；purge_shop 是幂等的，重试即可）。
        不能在任何闸门里调（revoke_shop 自己取写方闸门）。"""
        actor = _require_actor(actor)
        reason = (reason or "").strip()
        if not reason:
            raise ValueError("取消订购要写明原因")
        try:
            counts = dict(self._revoke(shop_id))
        except Exception as exc:
            logger.error(
                "取消订购没有完成（%s）：店铺 %s，请重试（清理是幂等的）",
                type(exc).__name__,
                shop_id,
            )
            raise
        self._unit.repos.audit.record(
            AuditEvent(
                shop_id=None,  # 系统事件：店铺的审计已经删了，这条要留下来
                actor=actor,
                action=AuditAction.SHOP_PURGED,
                detail={"shop_id": shop_id, "reason": reason, "counts": counts},
                created_at=self._unit.clock.now(),
            )
        )
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

        删之前逐店再确认一次还过期（列出之后续期成功、重新授权了的不删）。有店铺没删完时
        其余照删，最后抛 PurgeIncomplete（带上删完的和没删完的）。"""
        now = now or self._unit.clock.now()
        due = self.expired_shops(now=now)
        if dry_run:
            return {shop_id: {} for shop_id in due}
        cutoff = self._purge_cutoff(now)
        purged: dict[str, dict[str, int]] = {}
        failed: list[str] = []
        for shop_id in due:
            if not self._still_expired(shop_id, cutoff):
                logger.info("过期清理：店铺 %s 列出之后授权变了，不删", shop_id)
                continue
            try:
                purged[shop_id] = self.revoke(
                    shop_id, reason=PURGE_REASON_EXPIRED, actor=SYSTEM_ACTOR
                )
            except Exception:  # noqa: BLE001 — revoke 已记日志；其余店铺照删
                failed.append(shop_id)
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
