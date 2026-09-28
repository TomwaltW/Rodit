"""T3 平台状态归一与平台自由文本脱敏（方案 §7.2、§5.3）。"""

from __future__ import annotations

import pytest

from rodit.core.enums import (
    AfterSaleReason,
    AfterSaleStatus,
    AfterSaleType,
    LogisticsStatus,
    OrderStatus,
    Platform,
    SignedDetail,
)
from rodit.core.errors import ToolUnavailable
from rodit.core.pii import ADDRESS_MASK, PHONE_MASK, contains_pii
from rodit.testing.fixtures import SHOP_A, build_demo_world
from rodit.tools.normalize import (
    AFTERSALE_REASON_CODES,
    AFTERSALE_STATUS_CODES,
    AFTERSALE_TYPE_CODES,
    FALLBACK_LOGISTICS_STATUS,
    LOGISTICS_CODES,
    ORDER_STATUS_CODES,
    PICKUP_CODE_MASK,
    UnknownPlatformCode,
    mask_free_text,
    mask_platform_text,
    normalize_aftersale_reason,
    normalize_aftersale_status,
    normalize_aftersale_type,
    normalize_logistics_code,
    normalize_order_status,
    normalize_trace_status,
    scrub_pickup_code,
)

L, D = LogisticsStatus, SignedDetail


# ---------------------------------------------------------------- 物流状态码


@pytest.mark.parametrize(
    ("platform", "code", "expected"),
    [
        (Platform.TAOBAO, "ACCEPT", (L.PICKED_UP, None)),
        (Platform.TAOBAO, "TRANSPORT", (L.IN_TRANSIT, None)),
        (Platform.TAOBAO, "DELIVERING", (L.OUT_FOR_DELIVERY, None)),
        (Platform.TAOBAO, "SIGN", (L.SIGNED, D.SELF)),
        (Platform.TAOBAO, "SIGN_STATION", (L.SIGNED, D.STATION)),
        (Platform.TAOBAO, "SIGN_LOCKER", (L.SIGNED, D.LOCKER)),
        (Platform.TAOBAO, "FAILED", (L.EXCEPTION, None)),
        (Platform.TAOBAO, "REJECT", (L.RETURNING, None)),
        (Platform.DOUYIN, "41", (L.SIGNED, D.STATION)),
        (Platform.DOUYIN, "42", (L.SIGNED, D.LOCKER)),
        (Platform.DOUYIN, "7", (L.RETURNING, None)),
        (Platform.PDD, "STA_SIGN", (L.SIGNED, D.STATION)),
        (Platform.PDD, "CAB_SIGN", (L.SIGNED, D.LOCKER)),
        (Platform.PDD, "FAIL", (L.EXCEPTION, None)),
    ],
)
def test_logistics_code_mapping(platform, code, expected):
    assert normalize_logistics_code(platform, code) == expected


def test_logistics_code_unknown_or_empty_is_none():
    assert normalize_logistics_code(Platform.TAOBAO, "WEIRD") is None
    assert normalize_logistics_code(Platform.TAOBAO, None) is None
    # 码表按平台区分：淘宝的码在抖店不认
    assert normalize_logistics_code(Platform.DOUYIN, "SIGN_STATION") is None


def test_every_platform_covers_every_logistics_status_and_signed_detail():
    for platform, table in LOGISTICS_CODES.items():
        values = set(table.values())
        for status in LogisticsStatus:
            if status == LogisticsStatus.SIGNED:
                for detail in SignedDetail:
                    assert (status, detail) in values, (platform, detail)
            else:
                assert (status, None) in values, (platform, status)


def test_trace_status_takes_last_known_code():
    norm = normalize_trace_status(Platform.TAOBAO, ["ACCEPT", "TRANSPORT", "SIGN_STATION"])
    assert (norm.status, norm.signed_detail, norm.unknown_codes) == (L.SIGNED, D.STATION, [])


def test_trace_status_unknown_code_keeps_last_known_and_flags():
    norm = normalize_trace_status(
        Platform.TAOBAO, ["ACCEPT", "TRANSPORT", "NEW_CODE", None, "NEW_CODE", "X2"]
    )
    assert norm.status == L.IN_TRANSIT
    assert norm.signed_detail is None
    assert norm.unknown_codes == ["NEW_CODE", "X2"]


def test_trace_status_nothing_known_falls_back_to_in_transit():
    assert normalize_trace_status(Platform.PDD, ["??"]).status == FALLBACK_LOGISTICS_STATUS
    assert normalize_trace_status(Platform.PDD, ["??"]).unknown_codes == ["??"]
    empty = normalize_trace_status(Platform.PDD, [])
    assert empty.status == LogisticsStatus.IN_TRANSIT and empty.unknown_codes == []


# ---------------------------------------------------------------- 订单 / 售后状态码


def test_order_status_signed_via_sign_time():
    assert (
        normalize_order_status(Platform.TAOBAO, "WAIT_BUYER_CONFIRM_GOODS") == OrderStatus.SHIPPED
    )
    assert (
        normalize_order_status(Platform.TAOBAO, "WAIT_BUYER_CONFIRM_GOODS", signed=True)
        == OrderStatus.SIGNED
    )
    # 签收时间只影响"已发货"：已关闭的订单即使签收过也是关闭
    assert (
        normalize_order_status(Platform.TAOBAO, "TRADE_CLOSED", signed=True) == OrderStatus.CLOSED
    )
    assert normalize_order_status(Platform.DOUYIN, 3, signed=True) == OrderStatus.SIGNED
    assert normalize_order_status(Platform.PDD, 3) == OrderStatus.SIGNED


@pytest.mark.parametrize("platform", list(Platform))
def test_unknown_order_and_aftersale_codes_degrade_as_tool_unavailable(platform):
    for fn in (normalize_order_status, normalize_aftersale_status, normalize_aftersale_type):
        with pytest.raises(UnknownPlatformCode) as exc:
            fn(platform, "no-such-code")
        assert isinstance(exc.value, ToolUnavailable)


@pytest.mark.parametrize("platform", list(Platform))
def test_unknown_aftersale_reason_is_other(platform):
    assert normalize_aftersale_reason(platform, "买家写的奇怪原因") == AfterSaleReason.OTHER


@pytest.mark.parametrize("platform", list(Platform))
def test_every_enum_value_has_a_platform_code(platform):
    """演示编码要求每个内部取值都能编码成平台码（新增枚举值时提醒补映射表）。"""
    assert set(AFTERSALE_STATUS_CODES[platform].values()) == set(AfterSaleStatus)
    assert set(AFTERSALE_TYPE_CODES[platform].values()) == set(AfterSaleType)
    assert set(AFTERSALE_REASON_CODES[platform].values()) == set(AfterSaleReason)
    order_values = set(ORDER_STATUS_CODES[platform].values())
    assert order_values | {OrderStatus.SIGNED} == set(OrderStatus)


# ---------------------------------------------------------------- 平台自由文本脱敏


def test_mask_free_text_masks_pii_and_pickup_code():
    text = "快递员电话13812345678，已放到杭州市西湖区文三路90号3单元，取件码 3-2-1234"
    out = mask_free_text(text)
    assert PHONE_MASK in out and ADDRESS_MASK in out
    assert "13812345678" not in out and "文三路90号" not in out
    assert "1234" not in out and f"取件码 {PICKUP_CODE_MASK}" in out
    assert not contains_pii(out)


def test_mask_free_text_keeps_clean_text_verbatim():
    # 没有个人信息时不做 NFKC 归一，中文全角标点保持原样
    assert mask_free_text("已签收，菜鸟驿站代收") == "已签收，菜鸟驿站代收"
    assert mask_free_text(None) is None
    assert mask_free_text("") == ""


@pytest.mark.parametrize(
    "text",
    ["取件码：A12-3456", "凭取货码8866取件", "提货码为 5566", "开柜码 778899"],
)
def test_scrub_pickup_code_variants(text):
    out = scrub_pickup_code(text)
    assert PICKUP_CODE_MASK in out
    assert not any(ch.isdigit() for ch in out)


def test_mask_platform_text_trace_event_and_trace():
    w = build_demo_world()
    trace = w.trace(SHOP_A, "o_1010")
    dirty_event = trace.events[-1].model_copy(
        update={"description": "已签收，请凭取件码 9-9-8888 取件，有问题打13912345678"}
    )
    dirty = trace.model_copy(
        update={
            "events": [*trace.events[:-1], dirty_event],
            "pickup_info": "丰巢柜（幸福里 3 号楼）取件码 1122",
        }
    )
    masked = mask_platform_text(dirty)
    assert masked is not dirty
    desc = masked.events[-1].description
    assert "8888" not in desc and "13912345678" not in desc
    assert masked.pickup_info == f"丰巢柜（幸福里 3 号楼）取件码 {PICKUP_CODE_MASK}"
    # 干净的轨迹原样保留，原对象不被修改
    assert masked.events[0] == trace.events[0]
    assert dirty.events[-1].description.endswith("13912345678")


def test_mask_platform_text_aftersale_note():
    w = build_demo_world()
    info = w.aftersale(SHOP_A, "as_2003").model_copy(
        update={"note": "没收到，联系我 13812345678，地址文三路90号"}
    )
    masked = mask_platform_text(info)
    assert "13812345678" not in masked.note and "文三路90号" not in masked.note
    assert masked.aftersale_id == info.aftersale_id and masked.status == info.status


def test_mask_platform_text_rejects_other_types():
    with pytest.raises(TypeError):
        mask_platform_text(build_demo_world().order(SHOP_A, "o_1001"))  # type: ignore[type-var]
