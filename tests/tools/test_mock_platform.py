"""T3 模拟平台工具：各平台原始报文 → 解析、归一、最小化、脱敏（方案 §7.1、§7.2、§5.3）。"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from rodit.core.enums import AfterSaleStatus, LogisticsStatus, Platform, SignedDetail
from rodit.core.errors import ToolUnavailable
from rodit.core.models import OrderInfo, TraceEvent
from rodit.core.pii import contains_pii
from rodit.core.ports import ToolGateway
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import (
    BUYER,
    BUYER_RISKY,
    SHOP_A,
    SHOP_B,
    SHOP_C,
    build_demo_world,
)
from rodit.tools.mock_platform import MockPlatformTools

PLATFORMS = list(Platform)


@pytest.fixture
def world():
    return build_demo_world()


@pytest.mark.parametrize("platform", PLATFORMS)
def test_is_tool_gateway(world, platform):
    tools = MockPlatformTools(world, platform)
    assert isinstance(tools, ToolGateway)
    assert tools.platform == platform


@pytest.mark.parametrize("platform", PLATFORMS)
def test_orders_round_trip_through_raw_payload(world, platform):
    tools = MockPlatformTools(world, platform)
    for o in world.orders:
        assert tools.get_order(o.shop_id, o.order_id) == o, o.order_id


@pytest.mark.parametrize("platform", PLATFORMS)
def test_raw_order_payload_is_platform_shaped_and_receiver_fields_dropped(world, platform):
    tools = MockPlatformTools(world, platform)
    order = tools.get_order(SHOP_A, "o_1002")
    raw = tools.last_raw
    receiver_keys = {
        Platform.TAOBAO: {"receiver_name", "receiver_mobile", "receiver_address"},
        Platform.DOUYIN: {"encrypt_post_receiver", "encrypt_post_tel", "encrypt_post_addr"},
        Platform.PDD: {"receiver_name", "receiver_phone", "address"},
    }[platform]
    assert receiver_keys <= set(raw)
    dumped = order.model_dump(mode="json")
    for key in receiver_keys:
        assert str(raw[key]) not in str(dumped)
    assert set(OrderInfo.model_fields) >= set(dumped)
    # 金额单位：抖店 / 拼多多是分
    if platform == Platform.TAOBAO:
        assert raw["payment"] == "199.00"
    else:
        assert raw["pay_amount"] == 19900
    assert order.total_amount == Decimal("199.00")


@pytest.mark.parametrize("platform", PLATFORMS)
def test_signed_orders_decoded_from_sign_time(world, platform):
    tools = MockPlatformTools(world, platform)
    signed = tools.get_order(SHOP_A, "o_1003")
    assert signed.status == world.order(SHOP_A, "o_1003").status
    raw = tools.last_raw
    if platform == Platform.TAOBAO:
        assert raw["status"] == "WAIT_BUYER_CONFIRM_GOODS" and raw["sign_time"]
    if platform == Platform.DOUYIN:
        assert raw["order_status"] == 3 and raw["receipt_time"]


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("order_id", ["o_1002", "o_1003", "o_1004", "o_1006", "o_1008", "o_1009"])
def test_traces_normalize_to_same_status(world, platform, order_id):
    tools = MockPlatformTools(world, platform)
    expected = world.trace(SHOP_A, order_id)
    got = tools.get_logistics(SHOP_A, order_id)
    assert got.status == expected.status
    assert got.signed_detail == expected.signed_detail
    assert got.pickup_info == expected.pickup_info
    assert [e.time for e in got.events] == [e.time for e in expected.events]
    assert [e.description for e in got.events] == [e.description for e in expected.events]
    assert got.fetched_at == expected.fetched_at
    assert tools.unknown_codes == set()


def test_trace_raw_codes_are_platform_specific(world):
    expected_last = {
        Platform.TAOBAO: "SIGN_LOCKER",
        Platform.DOUYIN: "42",
        Platform.PDD: "CAB_SIGN",
    }
    for platform, code in expected_last.items():
        tools = MockPlatformTools(world, platform)
        trace = tools.get_logistics(SHOP_A, "o_1010")
        assert trace.events[-1].raw_status == code
        assert trace.status == LogisticsStatus.SIGNED
        assert trace.signed_detail == SignedDetail.LOCKER
        assert trace.pickup_info == "丰巢柜（幸福里 3 号楼）"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_unknown_logistics_code_keeps_last_known_status_and_is_flagged(world, platform):
    trace = world.trace(SHOP_A, "o_1002")
    weird = TraceEvent(
        time=trace.events[-1].time + timedelta(minutes=1),
        description="网点新状态",
        raw_status="BRAND_NEW_CODE",
    )
    world.traces[world.traces.index(trace)] = trace.model_copy(
        update={"events": [*trace.events, weird]}
    )
    tools = MockPlatformTools(world, platform)
    got = tools.get_logistics(SHOP_A, "o_1002")
    assert got.status == LogisticsStatus.IN_TRANSIT  # 保持上一个已知状态
    assert got.events[-1].raw_status == "BRAND_NEW_CODE"
    assert tools.unknown_codes == {"BRAND_NEW_CODE"}


@pytest.mark.parametrize("platform", PLATFORMS)
def test_aftersales_round_trip_with_status_version(world, platform):
    tools = MockPlatformTools(world, platform)
    for a in world.aftersales:
        got = tools.get_aftersale(a.shop_id, a.aftersale_id)
        assert got == a, a.aftersale_id
        assert got.status_version == a.status_version


@pytest.mark.parametrize("platform", PLATFORMS)
def test_platform_free_text_is_masked(world, platform):
    a = world.aftersale(SHOP_A, "as_2003")
    world.aftersales[world.aftersales.index(a)] = a.model_copy(
        update={"note": "没收到！电话13812345678，送到杭州市西湖区文三路90号"}
    )
    t = world.trace(SHOP_A, "o_1003")
    last = t.events[-1].model_copy(update={"description": "已签收，菜鸟驿站代收，取件码 6-2-3301"})
    world.traces[world.traces.index(t)] = t.model_copy(update={"events": [*t.events[:-1], last]})

    tools = MockPlatformTools(world, platform)
    note = tools.get_aftersale(SHOP_A, "as_2003").note
    assert "13812345678" not in note and "文三路90号" not in note
    assert not contains_pii(note)
    desc = tools.get_logistics(SHOP_A, "o_1003").events[-1].description
    assert "3301" not in desc and "取件码" in desc


@pytest.mark.parametrize("platform", PLATFORMS)
def test_list_recent_orders_order_and_isolation(world, platform):
    tools = MockPlatformTools(world, platform)
    got = tools.list_recent_orders(SHOP_A, BUYER, limit=10)
    keys = [(o.created_at, o.order_id) for o in got]
    assert keys == sorted(keys, reverse=True)
    assert {o.shop_id for o in got} == {SHOP_A}
    assert len(tools.list_recent_orders(SHOP_A, BUYER)) == 5
    assert [o.order_id for o in tools.list_recent_orders(SHOP_B, BUYER)] == ["o_9001"]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_list_aftersales_filters_and_order(world, platform):
    tools = MockPlatformTools(world, platform)
    risky = tools.list_aftersales(SHOP_A, buyer_ref=BUYER_RISKY)
    assert [a.aftersale_id for a in risky] == ["as_3001", "as_3002", "as_3003", "as_3004"]
    since = world.now - timedelta(days=10)
    assert [
        a.aftersale_id for a in tools.list_aftersales(SHOP_A, buyer_ref=BUYER_RISKY, since=since)
    ] == [
        "as_3003",
        "as_3004",
    ]
    assert [a.aftersale_id for a in tools.list_aftersales(SHOP_A, order_id="o_1012")] == [
        "as_2006",
        "as_2008",
    ]
    closed = tools.list_aftersales(SHOP_A, status=AfterSaleStatus.CLOSED)
    assert {a.status for a in closed} == {AfterSaleStatus.CLOSED}
    assert [a.aftersale_id for a in tools.list_aftersales(SHOP_B)] == ["as_9001"]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_stock_and_price(world, platform):
    tools = MockPlatformTools(world, platform)
    sku = tools.get_stock(SHOP_A, "i_555", "k_3")
    assert (sku.in_stock, sku.quantity, sku.restock_note) == (False, 0, "预计 10 月 8 日补货")
    summary = tools.get_stock(SHOP_A, "i_555")
    assert summary.sku_id is None and summary.in_stock and summary.quantity == 35
    unknown_qty = tools.get_stock(SHOP_C, "i_c1")
    assert unknown_qty.quantity is None and unknown_qty.in_stock
    assert tools.get_stock(SHOP_A, "i_nope") is None
    assert tools.get_stock(SHOP_B, "i_555", "k_1") is None  # 不跨店
    price = tools.get_price(SHOP_A, "i_555")
    assert price.price == Decimal("259.00") and price.promotions == ["店铺满 199 减 20"]
    assert tools.get_price(SHOP_A, "i_888").promotions == []
    assert tools.get_price(SHOP_B, "i_555") is None


def test_clock_sets_fetched_at(world):
    clock = FakeClock(world.now + timedelta(minutes=7))
    tools = MockPlatformTools(world, Platform.DOUYIN, clock=clock)
    assert tools.get_logistics(SHOP_A, "o_1002").fetched_at == clock.now()
    assert tools.get_price(SHOP_A, "i_555").fetched_at == clock.now()


def test_not_found_and_cross_shop(world):
    tools = MockPlatformTools(world, Platform.TAOBAO)
    assert tools.get_order(SHOP_B, "o_1001") is None
    assert tools.get_logistics(SHOP_A, "o_1001") is None  # 未发货没有轨迹
    assert tools.get_aftersale(SHOP_B, "as_2001") is None


def test_fail_on_raises_tool_unavailable(world):
    tools = MockPlatformTools(world, Platform.PDD)
    tools.fail_on = {"get_logistics"}
    with pytest.raises(ToolUnavailable):
        tools.get_logistics(SHOP_A, "o_1002")
    assert tools.calls[-1] == ("get_logistics", SHOP_A)
