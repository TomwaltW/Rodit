"""店铺授权与权限（方案 §8.4、§12.3、§12.4）。

- 商家在服务市场订购后授权，令牌按店铺保存（ShopAuthRepository，只在对应平台云内、加密落库）；
- 最小权限：MVP 只申请读订单、读物流、读售后、读商品（MVP_SCOPES）；写售后到二期再申请
  （PHASE2_SCOPES）；
- 处理任何店铺的数据前先 AuthGuard.require：没有授权、已过期、权限不够、被标记需要重新授权
  （needs_reauth：续期被拒、平台拒绝了当前令牌）都抛 AuthExpired，调用方停止处理
  （§8.4 授权过期时停止处理）；
- 平台云代码（只读工具、适配器）取令牌用 AuthGuardTokenProvider（core.ports.TokenProvider）：
  授权无效时抛 AuthExpired，平台拒绝令牌时 report_rejected 标记需要重新授权；
- 商家取消订购或授权过期：AuthGuard.revoke 删除令牌，并按约定删除该店铺的数据
  （调用各仓储的 purge_shop）；
- 授权即将过期告警（§12.4）：AuthGuard.expiring。

权限名是我们内部的叫法，申请授权时由各平台的实现映射成平台的权限包名【待确认】；
定义在 core.scopes（端口的默认参数要用），这里原样导出。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Literal, Protocol, runtime_checkable

from pydantic import SecretStr

from rodit.core.clock import Clock
from rodit.core.errors import AuthExpired, TokenDecryptionError
from rodit.core.models import ShopAuth
from rodit.core.ports import ShopAuthRepository
from rodit.core.scopes import (
    MVP_SCOPES,
    PHASE2_SCOPES,
    SCOPE_ITEM_READ,
    SCOPE_LOGISTICS_READ,
    SCOPE_ORDER_READ,
    SCOPE_REFUND_READ,
    SCOPE_REFUND_WRITE,
)

__all__ = [
    "MVP_SCOPES",
    "PHASE2_SCOPES",
    "SCOPE_ITEM_READ",
    "SCOPE_LOGISTICS_READ",
    "SCOPE_ORDER_READ",
    "SCOPE_REFUND_READ",
    "SCOPE_REFUND_WRITE",
    "AuthGuard",
    "AuthGuardTokenProvider",
    "Purger",
    "SupportsPurge",
    "excess_scopes",
    "scopes_for_phase",
]


def scopes_for_phase(phase: Literal[1, 2]) -> frozenset[str]:
    """该阶段应当申请的全部权限：1 = MVP（只读），2 = 加上写售后。"""
    if phase == 1:
        return MVP_SCOPES
    if phase == 2:
        return PHASE2_SCOPES
    raise ValueError(f"未知阶段：{phase!r}")


def excess_scopes(scopes: Iterable[str], *, phase: Literal[1, 2] = 1) -> frozenset[str]:
    """超出该阶段所需的权限（§8.4 最小权限）：申请授权前检查，非空说明申请多了。"""
    return frozenset(scopes) - scopes_for_phase(phase)


@runtime_checkable
class SupportsPurge(Protocol):
    """有 purge_shop 的仓储（core.ports 第 2 组都有）或适配器（BaseMockAdapter.purge_shop）。"""

    def purge_shop(self, shop_id: str) -> int: ...


Purger = SupportsPurge | Callable[[str], int]


class AuthGuard:
    """店铺授权检查（§8.4）。"""

    def __init__(self, repo: ShopAuthRepository, clock: Clock) -> None:
        self._repo = repo
        self._clock = clock

    def require(self, shop_id: str, scopes: Iterable[str] = MVP_SCOPES) -> ShopAuth:
        """返回有效的授权；没有授权、已过期（expires_at ≤ 现在）或缺权限时抛 AuthExpired。

        令牌读不出来（旧版加密、换过密钥、数据损坏）也按需要重新授权处理，抛 AuthExpired。
        异常信息不含令牌（ShopAuth.access_token 是 SecretStr）。
        """
        try:
            auth = self._repo.get(shop_id)
        except TokenDecryptionError as exc:
            raise AuthExpired(f"店铺 {shop_id} 的令牌读不出来，请重新授权（方案 §8.4）") from exc
        if auth is None:
            raise AuthExpired(f"店铺 {shop_id} 没有授权或已取消订购，停止处理（方案 §8.4）")
        if auth.expires_at <= self._clock.now():
            raise AuthExpired(f"店铺 {shop_id} 的授权已于 {auth.expires_at.isoformat()} 过期")
        if auth.needs_reauth:
            raise AuthExpired(f"店铺 {shop_id} 的授权需要商家重新授权（续期被拒或令牌失效，§8.4）")
        missing = frozenset(scopes) - frozenset(auth.scopes)
        if missing:
            raise AuthExpired(f"店铺 {shop_id} 的授权缺少权限：{sorted(missing)}")
        return auth

    def is_authorized(self, shop_id: str, scopes: Iterable[str] = MVP_SCOPES) -> bool:
        try:
            self.require(shop_id, scopes)
        except AuthExpired:
            return False
        return True

    def expiring(self, shop_ids: Iterable[str], *, within: timedelta) -> list[str]:
        """授权在 within 之内到期（含已过期、没有授权、令牌读不出来、需要重新授权）的店铺，
        按传入顺序（§12.4 授权过期告警）。"""
        deadline = self._clock.now() + within
        out: list[str] = []
        for shop_id in shop_ids:
            try:
                auth = self._repo.get(shop_id)
            except TokenDecryptionError:
                out.append(shop_id)  # 需要重新授权
                continue
            if auth is None or auth.expires_at <= deadline or auth.needs_reauth:
                out.append(shop_id)
        return out

    def mark_needs_reauth(self, shop_id: str) -> bool:
        """标记这家店需要重新授权（之后 require 不通过）；有授权返回 True。不读令牌。"""
        return bool(self._repo.mark_needs_reauth(shop_id))

    def revoke(
        self, shop_id: str, purge: Purger | Iterable[Purger] | None = None
    ) -> dict[str, int]:
        """取消订购 / 授权过期：先删除令牌（立即停止处理），再按约定删除该店铺的数据（§8.4）。

        purge 可以是一个 purge_shop 函数，或一组有 purge_shop 方法的仓储 / 适配器（也可以混着
        传函数）。返回每一项删除的条数，键是仓储类名或函数名（重名时加 #2、#3），
        另有 "auth"（删除的令牌数，0 或 1）。某一项清理失败时异常向外抛；令牌已经删除，
        purge_shop 是幂等的，可以重试。
        不读令牌（不解密）：旧版加密、换过密钥、数据损坏的令牌读不出来，取消订购照样删除。
        """
        had_token = bool(self._repo.delete(shop_id))
        counts: dict[str, int] = {"auth": int(had_token)}
        for purger in _purgers(purge):
            name = _unique_name(_label(purger), counts)
            if isinstance(purger, SupportsPurge):
                counts[name] = purger.purge_shop(shop_id)
            else:
                counts[name] = purger(shop_id)
        return counts


class AuthGuardTokenProvider:
    """core.ports.TokenProvider 的默认实现（基于 AuthGuard）：给平台云代码取店铺的有效令牌。

    access_token 先过 AuthGuard.require（没有授权、过期、缺权限、需要重新授权都抛 AuthExpired），
    返回 SecretStr；report_rejected 标记需要重新授权。令牌不进日志和异常消息。
    """

    def __init__(self, guard: AuthGuard) -> None:
        self._guard = guard

    def __repr__(self) -> str:
        return "AuthGuardTokenProvider()"

    def access_token(self, shop_id: str, scopes: Iterable[str] = MVP_SCOPES) -> SecretStr:
        return self._guard.require(shop_id, scopes).access_token

    def report_rejected(self, shop_id: str) -> None:
        self._guard.mark_needs_reauth(shop_id)


def _purgers(purge: Purger | Iterable[Purger] | None) -> list[Purger]:
    if purge is None:
        return []
    if isinstance(purge, SupportsPurge) or callable(purge):
        return [purge]
    return list(purge)


def _label(purger: Purger) -> str:
    if isinstance(purger, SupportsPurge):
        return type(purger).__name__
    return getattr(purger, "__name__", type(purger).__name__)


def _unique_name(name: str, taken: dict[str, int]) -> str:
    if name not in taken:
        return name
    n = 2
    while f"{name}#{n}" in taken:
        n += 1
    return f"{name}#{n}"
