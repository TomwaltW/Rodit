"""T3 售后审核规则 R1–R8 与叠加规则 R6 / R7（方案 §7.3.2）。"""

from __future__ import annotations

import itertools
from datetime import timedelta
from decimal import Decimal

import pytest

from rodit.aftersale.rules import REASON_WAIT_SELLER_CONFIRM, evaluate, repeat_count
from rodit.core.config import AfterSaleConfig, EngineConfig, effective_shop_config
from rodit.core.enums import (
    AfterSaleReason,
    AfterSaleStatus,
    AfterSaleType,
    OrderStatus,
    ReviewAdvice,
    ReviewState,
)
from rodit.core.models import AfterSaleInfo
from rodit.testing.fixtures import BUYER_RISKY, SHOP_A, build_demo_world

A, S = ReviewAdvice, ReviewState


@pytest.fixture
def world():
    return build_demo_world()


def _eval(
    world, info: AfterSaleInfo, *, order=..., trace=..., history=None, config=None, shop=None
):
    if order is ...:
        order = next(
            (o for o in world.orders if o.shop_id == info.shop_id and o.order_id == info.order_id),
            None,
        )
    if trace is ...:
        trace = next(
            (t for t in world.traces if t.shop_id == info.shop_id and t.order_id == info.order_id),
            None,
        )
    if history is None:
        history = [
            a
            for a in world.aftersales
            if a.shop_id == info.shop_id and a.buyer_ref == info.buyer_ref
        ]
    return evaluate(
        info,
        order=order,
        trace=trace,
        buyer_history=history,
        config=config or AfterSaleConfig(),
        shop_config=shop or effective_shop_config(EngineConfig(), world.shop(info.shop_id)),
        now=world.now,
    )


@pytest.mark.parametrize(
    ("aftersale_id", "rules", "advice", "next_state", "manager", "risky"),
    [
        ("as_2001", ["R2"], A.AGREE, S.CONFIRM, False, False),
        ("as_2002", ["R3"], A.AGREE_RETURN, S.CONFIRM, False, False),
        ("as_2003", ["R4"], A.NEGOTIATE, S.VERIFY, False, False),
        ("as_2004", ["R5"], A.MANUAL, S.EVIDENCE, False, False),
        ("as_2005", ["R3", "R6"], A.AGREE_RETURN, S.CONFIRM, True, False),
        ("as_2006", ["R1"], A.NONE, S.PLATFORM, False, False),
        ("as_2007", ["R8"], A.MANUAL, S.CONFIRM, False, False),
        ("as_2008", ["R1"], A.NONE, S.PLATFORM, False, False),
        ("as_2009", ["R5"], A.MANUAL, S.CONFIRM, False, False),
        ("as_3004", ["R3", "R7"], A.MANUAL, S.CONFIRM, False, True),
    ],
)
def test_demo_world_rules(world, aftersale_id, rules, advice, next_state, manager, risky):
    s = _eval(world, world.aftersale(SHOP_A, aftersale_id))
    assert s.matched_rules == rules
    assert s.advice == advice
    assert s.next_state == next_state
    assert s.requires_manager is manager
    assert bool(s.risk_flags) is risky
    assert s.reason


def test_reasons_explain_the_rule(world):
    assert "以平台为准" in _eval(world, world.aftersale(SHOP_A, "as_2001")).reason
    r3 = _eval(world, world.aftersale(SHOP_A, "as_2002")).reason
    assert "签收 5 天" in r3 and "仓库确认收到后再同意退款" in r3
    r4 = _eval(world, world.aftersale(SHOP_A, "as_2003")).reason
    assert "菜鸟驿站（幸福里店）" in r4 and "24 小时后仍未找到，建议同意" in r4
    assert "上传照片" in _eval(world, world.aftersale(SHOP_A, "as_2004")).reason
    assert "客服查看凭证后决定" in _eval(world, world.aftersale(SHOP_A, "as_2009")).reason
    assert "不支持七天无理由" in _eval(world, world.aftersale(SHOP_A, "as_2007")).reason
    r6 = _eval(world, world.aftersale(SHOP_A, "as_2005")).reason
    assert "1554.00 元超过店铺上限 500 元" in r6 and "店长" in r6
    r7 = _eval(world, world.aftersale(SHOP_A, "as_3004"))
    assert r7.risk_flags == ["30 天内售后 4 次（≥ 3）"]
    assert "人工决定（R7）" in r7.reason


def test_r1_is_terminal_even_with_overlays(world):
    info = world.aftersale(SHOP_A, "as_3001").model_copy(update={"amount": Decimal("9999")})
    s = _eval(world, info)
    assert s.matched_rules == ["R1"] and s.advice == A.NONE and not s.requires_manager
    assert s.risk_flags == []


@pytest.mark.parametrize(
    "status",
    [
        AfterSaleStatus.WAIT_BUYER_RETURN,
        AfterSaleStatus.REFUNDING,
        AfterSaleStatus.REFUND_SUCCESS,
        AfterSaleStatus.REFUSED,
        AfterSaleStatus.CLOSED,
    ],
)
def test_r1_for_every_non_actionable_status(world, status):
    info = world.aftersale(SHOP_A, "as_2001").model_copy(update={"status": status})
    s = _eval(world, info)
    assert (s.matched_rules, s.next_state) == (["R1"], S.PLATFORM)


def test_wait_seller_confirm_return_refund_suggests_refund(world):
    info = world.aftersale(SHOP_A, "as_2002").model_copy(
        update={"status": AfterSaleStatus.WAIT_SELLER_CONFIRM}
    )
    s = _eval(world, info)
    assert (s.matched_rules, s.advice, s.next_state) == (["R3"], A.AGREE, S.CONFIRM)
    assert REASON_WAIT_SELLER_CONFIRM == "仓库确认收货后同意退款"
    assert REASON_WAIT_SELLER_CONFIRM in s.reason
    exchange = info.model_copy(update={"type": AfterSaleType.EXCHANGE})
    assert _eval(world, exchange).advice == A.MANUAL


def test_r2_requires_refund_only_and_unshipped(world):
    as_2001 = world.aftersale(SHOP_A, "as_2001")
    return_refund = as_2001.model_copy(update={"type": AfterSaleType.RETURN_REFUND})
    assert _eval(world, return_refund).matched_rules == ["R8"]
    shipped = world.order(SHOP_A, "o_1001").model_copy(update={"status": OrderStatus.SHIPPED})
    assert _eval(world, as_2001, order=shipped, trace=None).matched_rules == ["R8"]


def test_r3_seven_day_boundary(world):
    info = world.aftersale(SHOP_A, "as_2002")
    order = world.order(SHOP_A, "o_1004")
    day7 = world.now.replace(hour=0, minute=1) - timedelta(days=7)
    assert _eval(world, info, order=order.model_copy(update={"signed_at": day7})).matched_rules == [
        "R3"
    ]
    day8 = day7 - timedelta(minutes=2)
    s = _eval(world, info, order=order.model_copy(update={"signed_at": day8}), trace=None)
    assert s.matched_rules == ["R8"] and "超过 7 天" in s.reason


def test_r3_needs_no_reason_and_return_refund(world):
    info = world.aftersale(SHOP_A, "as_2002")
    assert _eval(
        world, info.model_copy(update={"reason": AfterSaleReason.OTHER})
    ).matched_rules == ["R8"]
    refund_only = info.model_copy(update={"type": AfterSaleType.REFUND_ONLY})
    assert _eval(world, refund_only).matched_rules == ["R8"]


def test_r4_falls_back_to_order_sign_time_without_trace(world):
    info = world.aftersale(SHOP_A, "as_2003")
    s = _eval(world, info, trace=None)
    assert (s.matched_rules, s.next_state) == (["R4"], S.VERIFY)
    unsigned = world.order(SHOP_A, "o_1003").model_copy(
        update={"signed_at": None, "status": OrderStatus.SHIPPED}
    )
    s = _eval(world, info, order=unsigned, trace=None)
    assert s.matched_rules == ["R8"] and "物流未显示签收" in s.reason
    # 轨迹显示还在运输中：不算签收
    in_transit = world.trace(SHOP_A, "o_1002").model_copy(update={"order_id": "o_1003"})
    assert _eval(world, info, trace=in_transit).matched_rules == ["R8"]


@pytest.mark.parametrize(
    "reason",
    [
        AfterSaleReason.QUALITY,
        AfterSaleReason.DAMAGED,
        AfterSaleReason.WRONG_ITEM,
        AfterSaleReason.MISSING_ITEM,
    ],
)
def test_r5_reasons(world, reason):
    info = world.aftersale(SHOP_A, "as_2004").model_copy(update={"reason": reason})
    assert _eval(world, info).next_state == S.EVIDENCE
    with_photos = info.model_copy(update={"evidence_count": 1})
    s = _eval(world, with_photos)
    assert (s.matched_rules, s.advice, s.next_state) == (["R5"], A.MANUAL, S.CONFIRM)


def test_missing_order_degrades_to_manual(world):
    s = _eval(world, world.aftersale(SHOP_A, "as_2001"), order=None, trace=None)
    assert s.matched_rules == ["R8"] and s.advice == A.MANUAL and "订单信息查不到" in s.reason


def test_r6_boundary(world):
    info = world.aftersale(SHOP_A, "as_2001")
    at_limit = _eval(world, info.model_copy(update={"amount": Decimal("500.00")}))
    assert "R6" not in at_limit.matched_rules and not at_limit.requires_manager
    over = _eval(world, info.model_copy(update={"amount": Decimal("500.01")}))
    assert over.matched_rules == ["R2", "R6"] and over.requires_manager
    assert over.advice == A.AGREE  # R6 只要求店长确认，不改建议


def test_r6_uses_shop_config_limit(world):
    shop = effective_shop_config(EngineConfig(), world.shop(SHOP_A)).model_copy(
        update={"aftersale_amount_limit": Decimal("100")}
    )
    assert _eval(world, world.aftersale(SHOP_A, "as_2001"), shop=shop).requires_manager


def _risky_history(world, days: list[float]) -> list[AfterSaleInfo]:
    base = world.aftersale(SHOP_A, "as_3001")
    return [
        base.model_copy(
            update={"aftersale_id": f"as_h{i}", "created_at": world.now - timedelta(days=d)}
        )
        for i, d in enumerate(days)
    ]


def test_r7_threshold_and_window_boundary(world):
    current = world.aftersale(SHOP_A, "as_3004")
    two_before = _risky_history(world, [5, 30])  # 正好 30 天前的也算
    s = _eval(world, current, history=two_before)
    assert "R7" in s.matched_rules and s.risk_flags == ["30 天内售后 3 次（≥ 3）"]
    outside = _risky_history(world, [5]) + [
        _risky_history(world, [30])[0].model_copy(
            update={
                "aftersale_id": "as_old",
                "created_at": world.now - timedelta(days=30, seconds=1),
            }
        )
    ]
    s = _eval(world, current, history=outside)
    assert "R7" not in s.matched_rules and s.advice == A.AGREE_RETURN


def test_r7_counts_current_once_and_ignores_other_buyers(world):
    current = world.aftersale(SHOP_A, "as_3004")
    assert repeat_count(current, [], window_days=30, now=world.now) == 1
    assert repeat_count(current, [current, current], window_days=30, now=world.now) == 1
    others = [
        a.model_copy(update={"buyer_ref": "b_other"}) for a in _risky_history(world, [1, 2, 3])
    ]
    other_shop = [a.model_copy(update={"shop_id": "s_2002"}) for a in _risky_history(world, [1, 2])]
    assert repeat_count(current, others + other_shop, window_days=30, now=world.now) == 1
    assert repeat_count(current, _risky_history(world, [1, 2]), window_days=30, now=world.now) == 3


def test_r7_forces_manual_over_agree_and_negotiate(world):
    history = _risky_history(world, [1, 2])
    r2 = world.aftersale(SHOP_A, "as_2001").model_copy(update={"buyer_ref": BUYER_RISKY})
    s = _eval(world, r2, history=history)
    assert s.matched_rules == ["R2", "R7"] and s.advice == A.MANUAL
    r4 = world.aftersale(SHOP_A, "as_2003").model_copy(update={"buyer_ref": BUYER_RISKY})
    s = _eval(world, r4, history=history)
    assert s.matched_rules == ["R4", "R7"] and s.advice == A.MANUAL and s.next_state == S.VERIFY


def test_r6_and_r7_together(world):
    info = world.aftersale(SHOP_A, "as_3004").model_copy(update={"amount": Decimal("800")})
    s = _eval(world, info)
    assert s.matched_rules == ["R3", "R6", "R7"]
    assert s.requires_manager and s.advice == A.MANUAL


def test_config_thresholds_are_used(world):
    cfg = AfterSaleConfig(repeat_threshold=5, verify_hours=12)
    assert "R7" not in _eval(world, world.aftersale(SHOP_A, "as_3004"), config=cfg).matched_rules
    assert "12 小时后" in _eval(world, world.aftersale(SHOP_A, "as_2003"), config=cfg).reason


def test_never_suggests_refuse(world):
    base = world.aftersale(SHOP_A, "as_2001")
    orders = [world.order(SHOP_A, oid) for oid in ("o_1001", "o_1003", "o_1005", "o_1007")]
    for status, kind, reason, order, count, amount in itertools.product(
        AfterSaleStatus,
        AfterSaleType,
        AfterSaleReason,
        [*orders, None],
        (0, 2),
        (Decimal("10"), Decimal("5000")),
    ):
        info = base.model_copy(
            update={
                "status": status,
                "type": kind,
                "reason": reason,
                "evidence_count": count,
                "amount": amount,
                "order_id": order.order_id if order else "o_none",
            }
        )
        s = _eval(world, info, order=order, trace=..., history=_risky_history(world, [1, 2]))
        assert s.advice != ReviewAdvice.REFUSE
