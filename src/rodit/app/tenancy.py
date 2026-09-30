"""店铺生命周期（方案 §8.4 开通 / 续订 / 取消订购 / 授权过期，§9.1、§10 shop / shop_auth）。

T0 只放契约骨架（签名 + 文档），实现由 Track A 完成：

- onboard：订购开通（SubscriptionEvent，kind 为 subscribed / renewed）——核对平台是本单元的平台、
  权限不超出 TenancyConfig.onboarding_phase 允许的范围（adapters.auth.excess_scopes），
  写店铺（Shop，TRIAL / ACTIVE）和加密的授权（ShopAuth，含续期令牌），审计 auth.granted；
  同一家店取消订购之后可以再开通；
- set_status：trial → active → suspended → active，审计 shop.status_changed；
- handle_subscription：平台推送的订购事件（BindingContext.on_subscription、平台对接接口）：
  subscribed / renewed → onboard；cancelled / auth_revoked → 取消订购（revoke）+ 系统事件
  shop.purged（shop_id 为空，detail 只有店铺 ID、原因、删除条数，清理之后还在）；
- purge_expired：授权过期超过 TenancyConfig.purge_after_expiry_days 的店铺 → 取消订购 +
  shop.purged；dry_run 只列出、不删。

写操作在 processing(shop_id) 里做（和取消订购互斥）；revoke 是 AssistService.revoke_shop
（它自己取写方闸门，不能在 processing 里调）。令牌不进日志、审计和异常消息。
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime

from rodit.app.unit import Unit
from rodit.core.enums import ShopStatus
from rodit.core.models import Shop, SubscriptionEvent

ShopGate = Callable[[str], AbstractContextManager[None]]
"""AssistService.processing：按店铺的读方闸门（和取消订购互斥）。"""

Revoke = Callable[[str], dict[str, int]]
"""AssistService.revoke_shop：删令牌、停止处理、按约定删除该店铺在本单元的全部数据。"""


class TenancyService:
    """店铺生命周期（§8.4）。AssistService.tenancy。"""

    def __init__(self, unit: Unit, processing: ShopGate, revoke: Revoke) -> None:
        self._unit = unit
        self._processing = processing
        self._revoke = revoke

    def onboard(
        self,
        event: SubscriptionEvent,
        *,
        actor: str = "system",
        status: ShopStatus = ShopStatus.TRIAL,
    ) -> Shop:
        """订购开通 / 续订：写店铺和授权，返回店铺。平台不对、权限超出范围抛 ValueError。"""
        raise NotImplementedError("TenancyService.onboard 由 Track A 实现")

    def set_status(self, shop_id: str, status: ShopStatus, *, actor: str) -> Shop:
        """切换店铺状态（trial / active / suspended），返回更新后的店铺；店铺不存在抛 KeyError。"""
        raise NotImplementedError("TenancyService.set_status 由 Track A 实现")

    def handle_subscription(self, event: SubscriptionEvent) -> None:
        """平台推送的订购事件（开通、续订、取消订购、撤销授权）。"""
        raise NotImplementedError("TenancyService.handle_subscription 由 Track A 实现")

    def purge_expired(
        self, *, now: datetime | None = None, dry_run: bool = False
    ) -> dict[str, dict[str, int]]:
        """授权过期超过宽限期的店铺：取消订购并删除数据。返回 店铺 → 各项删除条数
        （dry_run 时只列出店铺，条数为空）。"""
        raise NotImplementedError("TenancyService.purge_expired 由 Track A 实现")


__all__ = ["Revoke", "ShopGate", "TenancyService"]
