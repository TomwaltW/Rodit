"""带短时缓存的只读工具网关（方案 §5.2.1 实时接口带短时缓存、§12.1、§12.2、§12.4、§9.2）。

CachingToolGateway 包在任意 ToolGateway 外面，本身也是 ToolGateway（只读，没有写方法）：

- 缓存按 shop_id 分命名空间（core.ports.Cache），TTL 内直接用缓存，省平台接口额度（§12.1）；
- 平台限流或故障（ToolUnavailable）时，有缓存就用缓存（不管是否过期，可用 max_stale_s 限制），
  数据里的 fetched_at 保持原来的获取时间，回答据此写"截至"（§12.2 "用缓存数据并注明时间"）；
  没有缓存就把 ToolUnavailable 抛给调用方（推荐"我帮您查一下"）；
- 防御性串店检查：内层返回的任何对象 shop_id 与请求的不一致 → TenantIsolationError，不缓存（§9.2）；
- 按店铺、方法统计调用量、缓存命中、平台调用、失败次数，供额度监控（§12.4），见 stats()。

注意：
- Cache 的写入时间由缓存实现自己的时钟给出，缓存和网关要注入同一个 Clock，TTL 才准。
- 售后审核重读（§7.3.2 防止误操作）不能读缓存：审核助手要用 uncached() 拿到内层网关。
- OrderInfo / AfterSaleInfo 没有 fetched_at 字段，用过期缓存时无法把获取时间带给调用方
  （物流、库存、价格可以）；见回执 contract_issues。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

from rodit.core.clock import Clock
from rodit.core.enums import AfterSaleStatus, Platform
from rodit.core.errors import TenantIsolationError, ToolUnavailable
from rodit.core.models import (
    AfterSaleInfo,
    LogisticsTrace,
    OrderInfo,
    PriceInfo,
    StockInfo,
)
from rodit.core.ports import Cache, ToolGateway

_V = TypeVar("_V")


class ToolCallStats(BaseModel):
    """一个店铺、一个工具方法的调用统计（§12.4 平台接口调用量、失败率）。"""

    model_config = ConfigDict(extra="forbid")

    requests: int = 0  # 调用方请求次数
    cache_hits: int = 0  # TTL 内命中缓存
    upstream_calls: int = 0  # 实际调用平台接口的次数（消耗额度）
    failures: int = 0  # 平台接口抛 ToolUnavailable 的次数
    stale_served: int = 0  # 平台失败时用过期缓存兜底的次数
    tenant_violations: int = 0  # 内层返回了其他店铺的数据


def _copy(value: Any) -> Any:
    """缓存里存副本、取出也给副本，避免调用方改动缓存内容。"""
    if isinstance(value, BaseModel):
        return value.model_copy(deep=True)
    if isinstance(value, list):
        return [_copy(v) for v in value]
    return value


class CachingToolGateway:
    """实现 core.ports.ToolGateway（只读）。"""

    def __init__(
        self,
        inner: ToolGateway,
        cache: Cache,
        clock: Clock,
        *,
        ttl_s: int,
        stale_ok: bool = True,
        max_stale_s: int | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._clock = clock
        self._ttl = timedelta(seconds=ttl_s)
        self._stale_ok = stale_ok
        self._max_stale = timedelta(seconds=max_stale_s) if max_stale_s is not None else None
        self.platform: Platform = inner.platform
        self._lock = threading.Lock()
        self._stats: dict[str, dict[str, ToolCallStats]] = {}

    # ---------------------------------------------------------------- 辅助

    @property
    def inner(self) -> ToolGateway:
        return self._inner

    def uncached(self) -> ToolGateway:
        """不带缓存的内层网关：售后审核重读状态用（§7.3.2）。"""
        return self._inner

    def stats(self, shop_id: str | None = None) -> dict[str, dict[str, ToolCallStats]]:
        """{shop_id: {方法名: ToolCallStats}} 的快照；给了 shop_id 只返回该店。"""
        with self._lock:
            return {
                s: {m: st.model_copy() for m, st in methods.items()}
                for s, methods in self._stats.items()
                if shop_id is None or s == shop_id
            }

    def _bump(self, shop_id: str, method: str, field: str) -> None:
        with self._lock:
            st = self._stats.setdefault(shop_id, {}).setdefault(method, ToolCallStats())
            setattr(st, field, getattr(st, field) + 1)

    def _fresh(self, written_at: datetime, now: datetime) -> bool:
        return now - written_at < self._ttl

    def _check_tenant(self, method: str, shop_id: str, value: Any) -> None:
        items = value if isinstance(value, list) else [value]
        for obj in items:
            if obj is None:
                continue
            other = getattr(obj, "shop_id", shop_id)
            if other != shop_id:
                self._bump(shop_id, method, "tenant_violations")
                raise TenantIsolationError(
                    f"工具 {method} 为店铺 {shop_id} 返回了店铺 {other} 的数据（§9.2 串店）"
                )

    def _call(self, method: str, shop_id: str, key: str, fetch: Callable[[], _V]) -> _V:
        self._bump(shop_id, method, "requests")
        now = self._clock.now()
        hit = self._cache.get(shop_id, key)
        if hit is not None and self._fresh(hit[1], now):
            self._bump(shop_id, method, "cache_hits")
            return _copy(hit[0])
        self._bump(shop_id, method, "upstream_calls")
        try:
            value = fetch()
        except ToolUnavailable:
            self._bump(shop_id, method, "failures")
            if (
                self._stale_ok
                and hit is not None
                and (self._max_stale is None or now - hit[1] <= self._max_stale)
            ):
                self._bump(shop_id, method, "stale_served")
                return _copy(hit[0])
            raise
        self._check_tenant(method, shop_id, value)
        self._cache.set(shop_id, key, _copy(value))
        return value

    # ---------------------------------------------------------------- ToolGateway

    def get_order(self, shop_id: str, order_id: str) -> OrderInfo | None:
        return self._call(
            "get_order",
            shop_id,
            f"order|{order_id}",
            lambda: self._inner.get_order(shop_id, order_id),
        )

    def list_recent_orders(
        self, shop_id: str, buyer_ref: str, *, limit: int = 5
    ) -> list[OrderInfo]:
        return self._call(
            "list_recent_orders",
            shop_id,
            f"recent_orders|{buyer_ref}|{limit}",
            lambda: self._inner.list_recent_orders(shop_id, buyer_ref, limit=limit),
        )

    def get_logistics(self, shop_id: str, order_id: str) -> LogisticsTrace | None:
        return self._call(
            "get_logistics",
            shop_id,
            f"logistics|{order_id}",
            lambda: self._inner.get_logistics(shop_id, order_id),
        )

    def get_aftersale(self, shop_id: str, aftersale_id: str) -> AfterSaleInfo | None:
        return self._call(
            "get_aftersale",
            shop_id,
            f"aftersale|{aftersale_id}",
            lambda: self._inner.get_aftersale(shop_id, aftersale_id),
        )

    def list_aftersales(
        self,
        shop_id: str,
        *,
        order_id: str | None = None,
        buyer_ref: str | None = None,
        status: AfterSaleStatus | None = None,
        since: datetime | None = None,
    ) -> list[AfterSaleInfo]:
        key = "aftersales|{}|{}|{}|{}".format(
            order_id or "", buyer_ref or "", status or "", since.isoformat() if since else ""
        )
        return self._call(
            "list_aftersales",
            shop_id,
            key,
            lambda: self._inner.list_aftersales(
                shop_id, order_id=order_id, buyer_ref=buyer_ref, status=status, since=since
            ),
        )

    def get_stock(self, shop_id: str, item_id: str, sku_id: str | None = None) -> StockInfo | None:
        return self._call(
            "get_stock",
            shop_id,
            f"stock|{item_id}|{sku_id or ''}",
            lambda: self._inner.get_stock(shop_id, item_id, sku_id),
        )

    def get_price(self, shop_id: str, item_id: str) -> PriceInfo | None:
        return self._call(
            "get_price",
            shop_id,
            f"price|{item_id}",
            lambda: self._inner.get_price(shop_id, item_id),
        )
