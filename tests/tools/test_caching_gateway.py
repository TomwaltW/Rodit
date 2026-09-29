"""T3 带缓存的只读网关：TTL、过期缓存兜底、串店检查、调用统计（方案 §12.1、§12.2、§12.4、§9.2）。"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from rodit.core.errors import TenantIsolationError, ToolError, ToolUnavailable
from rodit.core.ports import ToolGateway
from rodit.inmemory import InMemoryCache
from rodit.testing.fakes import FakeClock, InMemoryToolGateway
from rodit.testing.fixtures import BUYER, SHOP_A, SHOP_B, build_demo_world
from rodit.tools.gateway import CachingToolGateway, ToolCallStats

TTL = 300


@pytest.fixture
def env():
    w = build_demo_world()
    clock = FakeClock(w.now)
    inner = InMemoryToolGateway(w)
    gw = CachingToolGateway(inner, InMemoryCache(clock), clock, ttl_s=TTL)
    return w, clock, inner, gw


def _inner_calls(inner: InMemoryToolGateway, name: str) -> int:
    return sum(1 for c in inner.calls if c[0] == name)


def test_is_read_only_tool_gateway(env):
    _, _, inner, gw = env
    assert isinstance(gw, ToolGateway)
    assert gw.platform == inner.platform
    assert gw.uncached() is inner and gw.inner is inner
    for write in ("agree_refund", "agree_return", "refuse"):
        assert not hasattr(gw, write)


def test_fresh_within_ttl_then_expires(env):
    _, clock, inner, gw = env
    first = gw.get_logistics(SHOP_A, "o_1002")
    assert gw.get_logistics(SHOP_A, "o_1002") == first
    assert _inner_calls(inner, "get_logistics") == 1
    clock.advance(seconds=TTL - 1)
    gw.get_logistics(SHOP_A, "o_1002")
    assert _inner_calls(inner, "get_logistics") == 1
    clock.advance(seconds=1)  # 正好到 TTL：过期
    gw.get_logistics(SHOP_A, "o_1002")
    assert _inner_calls(inner, "get_logistics") == 2


def test_returns_copies_not_cached_object(env):
    _, _, _, gw = env
    order = gw.get_order(SHOP_A, "o_1001")
    order.lines.clear()
    assert gw.get_order(SHOP_A, "o_1001").lines


def test_stale_cache_served_on_tool_unavailable_with_original_fetched_at(env):
    w, clock, inner, gw = env
    fresh = gw.get_logistics(SHOP_A, "o_1002")
    clock.advance(hours=3)
    inner.fail_on = {"get_logistics"}
    stale = gw.get_logistics(SHOP_A, "o_1002")
    assert stale == fresh
    assert stale.fetched_at == w.now  # 原来的获取时间，回答据此写"截至"
    st = gw.stats(SHOP_A)[SHOP_A]["get_logistics"]
    assert (st.requests, st.upstream_calls, st.failures, st.stale_served) == (2, 2, 1, 1)


def test_stale_orders_and_lists_also_served(env):
    _, clock, inner, gw = env
    order = gw.get_order(SHOP_A, "o_1001")
    recent = gw.list_recent_orders(SHOP_A, BUYER)
    clock.advance(seconds=TTL + 1)
    inner.fail_on = {"get_order", "list_recent_orders"}
    assert gw.get_order(SHOP_A, "o_1001") == order
    assert gw.list_recent_orders(SHOP_A, BUYER) == recent


def test_no_cache_reraises(env):
    _, _, inner, gw = env
    inner.fail_on = {"get_price"}
    with pytest.raises(ToolUnavailable):
        gw.get_price(SHOP_A, "i_555")
    assert gw.stats()[SHOP_A]["get_price"].failures == 1


def test_stale_ok_false_reraises():
    w = build_demo_world()
    clock = FakeClock(w.now)
    inner = InMemoryToolGateway(w)
    gw = CachingToolGateway(inner, InMemoryCache(clock), clock, ttl_s=TTL, stale_ok=False)
    gw.get_stock(SHOP_A, "i_555", "k_1")
    clock.advance(seconds=TTL)
    inner.fail_on = {"get_stock"}
    with pytest.raises(ToolUnavailable):
        gw.get_stock(SHOP_A, "i_555", "k_1")


def test_max_stale_limits_fallback():
    w = build_demo_world()
    clock = FakeClock(w.now)
    inner = InMemoryToolGateway(w)
    gw = CachingToolGateway(inner, InMemoryCache(clock), clock, ttl_s=TTL, max_stale_s=3600)
    gw.get_price(SHOP_A, "i_555")
    inner.fail_on = {"get_price"}
    clock.advance(seconds=3600)
    assert gw.get_price(SHOP_A, "i_555") is not None  # 正好 1 小时：还能兜底
    clock.advance(seconds=1)
    with pytest.raises(ToolUnavailable):
        gw.get_price(SHOP_A, "i_555")


def test_other_tool_errors_are_not_masked(env):
    _, clock, inner, gw = env

    def boom(*_: Any, **__: Any) -> None:
        raise ToolError("鉴权失败")

    gw.get_order(SHOP_A, "o_1001")
    inner.get_order = boom  # type: ignore[method-assign]
    clock.advance(seconds=TTL)
    with pytest.raises(ToolError):
        gw.get_order(SHOP_A, "o_1001")


def test_cache_is_namespaced_by_shop(env):
    _, _, inner, gw = env
    assert gw.get_order(SHOP_A, "o_1001") is not None
    assert gw.get_order(SHOP_B, "o_1001") is None  # B 店拿不到 A 店的缓存
    assert [o.order_id for o in gw.list_recent_orders(SHOP_B, BUYER)] == ["o_9001"]
    assert gw.get_order(SHOP_B, "o_1001") is None  # None 也按店缓存
    assert _inner_calls(inner, "get_order") == 2


def test_list_aftersales_key_includes_filters(env):
    w, _, inner, gw = env
    by_order = gw.list_aftersales(SHOP_A, order_id="o_1012")
    by_buyer = gw.list_aftersales(SHOP_A, buyer_ref=BUYER)
    since = gw.list_aftersales(SHOP_A, buyer_ref=BUYER, since=w.now - timedelta(hours=1))
    assert [a.aftersale_id for a in by_order] == ["as_2006", "as_2008"]
    assert [a.aftersale_id for a in by_buyer] == ["as_2001", "as_2003"]
    assert since == []
    assert _inner_calls(inner, "list_aftersales") == 3
    gw.list_aftersales(SHOP_A, order_id="o_1012")
    assert _inner_calls(inner, "list_aftersales") == 3


class _WrongShop:
    """内层网关把数据标成别的店（模拟实现缺陷），用来测防御性串店检查。"""

    def __init__(self, inner: InMemoryToolGateway, leak_to: str) -> None:
        self.platform = inner.platform
        self._inner = inner
        self._leak = leak_to

    def _mark(self, v: Any) -> Any:
        if isinstance(v, list):
            return [self._mark(x) for x in v]
        return v.model_copy(update={"shop_id": self._leak}) if v is not None else None

    def __getattr__(self, name: str) -> Any:
        fn = getattr(self._inner, name)
        return lambda *a, **k: self._mark(fn(*a, **k))


def test_tenant_violation_raises_and_is_not_cached():
    w = build_demo_world()
    clock = FakeClock(w.now)
    cache = InMemoryCache(clock)
    gw = CachingToolGateway(_WrongShop(InMemoryToolGateway(w), SHOP_B), cache, clock, ttl_s=TTL)
    with pytest.raises(TenantIsolationError):
        gw.get_order(SHOP_A, "o_1001")
    with pytest.raises(TenantIsolationError):
        gw.list_aftersales(SHOP_A, buyer_ref=BUYER)
    assert cache.get(SHOP_A, "order|o_1001") is None
    assert gw.stats()[SHOP_A]["get_order"].tenant_violations == 1


def test_stats_per_shop_and_method(env):
    _, _, inner, gw = env
    gw.get_order(SHOP_A, "o_1001")
    gw.get_order(SHOP_A, "o_1001")
    gw.get_price(SHOP_A, "i_555")
    gw.get_order(SHOP_B, "o_9001")
    stats = gw.stats()
    assert stats[SHOP_A]["get_order"] == ToolCallStats(requests=2, cache_hits=1, upstream_calls=1)
    assert stats[SHOP_A]["get_price"].upstream_calls == 1
    assert stats[SHOP_B]["get_order"].requests == 1
    assert set(gw.stats(SHOP_B)) == {SHOP_B}
    # 快照：改返回值不影响内部计数
    stats[SHOP_A]["get_order"].requests = 99
    assert gw.stats()[SHOP_A]["get_order"].requests == 2


# ---------------------------------------------------------------- 订单 / 售后单的获取时间（§12.2）


def test_order_and_aftersale_are_stamped_with_fetch_time(env):
    """订单、售后单平台不带获取时间：网关在取到时记下，过期缓存兜底时仍是原来的时间，
    回答才能写"截至 <真实时间>"（§12.2 用缓存数据并注明时间）。"""
    w, clock, inner, gw = env
    fetched = clock.now()
    assert w.order(SHOP_A, "o_1001").fetched_at is None  # 平台数据本身没有
    assert gw.get_order(SHOP_A, "o_1001").fetched_at == fetched
    assert all(o.fetched_at == fetched for o in gw.list_recent_orders(SHOP_A, BUYER))
    assert gw.get_aftersale(SHOP_A, "as_2001").fetched_at == fetched
    listed = gw.list_aftersales(SHOP_A, buyer_ref=BUYER)
    assert listed and all(a.fetched_at == fetched for a in listed)

    clock.advance(hours=1)
    inner.fail_on = {"get_order", "list_recent_orders", "get_aftersale", "list_aftersales"}
    assert gw.get_order(SHOP_A, "o_1001").fetched_at == fetched
    assert all(o.fetched_at == fetched for o in gw.list_recent_orders(SHOP_A, BUYER))
    assert gw.get_aftersale(SHOP_A, "as_2001").fetched_at == fetched
    assert all(a.fetched_at == fetched for a in gw.list_aftersales(SHOP_A, buyer_ref=BUYER))

    # 平台恢复：重新取到的数据是新的获取时间
    inner.fail_on = set()
    assert gw.get_order(SHOP_A, "o_1001").fetched_at == clock.now()
