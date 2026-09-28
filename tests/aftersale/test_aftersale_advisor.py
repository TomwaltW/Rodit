"""T3 售后买家侧问答依据：能不能退、退款进度（方案 §7.3.1）。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rodit.aftersale.advisor import DefaultAfterSaleAdvisor, days_since_signed
from rodit.core.clock import CST
from rodit.core.config import AfterSaleConfig
from rodit.core.enums import AfterSaleStatus, OrderStatus
from rodit.core.ports import AfterSaleAdvisor
from rodit.testing.fixtures import SHOP_A, build_demo_world


@pytest.fixture
def world():
    return build_demo_world()


@pytest.fixture
def advisor():
    return DefaultAfterSaleAdvisor(AfterSaleConfig())


def test_is_aftersale_advisor(advisor):
    assert isinstance(advisor, AfterSaleAdvisor)


def test_days_since_signed_counts_calendar_days():
    now = datetime(2026, 9, 28, 10, 35, tzinfo=CST)
    assert days_since_signed(datetime(2026, 9, 28, 0, 1, tzinfo=CST), now) == 0
    assert days_since_signed(datetime(2026, 9, 27, 23, 59, tzinfo=CST), now) == 1
    assert days_since_signed(datetime(2026, 9, 21, 23, 59, tzinfo=CST), now) == 7
    # 按东八区自然日算：UTC 09-27 16:30 是东八区 09-28 00:30，算签收当天
    assert days_since_signed(datetime(2026, 9, 27, 16, 30, tzinfo=UTC), now) == 0
    assert days_since_signed(datetime(2026, 9, 27, 15, 30, tzinfo=UTC), now) == 1


def _elig(advisor, world, order_id, **update):
    order = world.order(SHOP_A, order_id).model_copy(update=update)
    return advisor.return_eligibility(order, world.policy_map(SHOP_A), world.now)


def test_unshipped_refund_before_shipping(advisor, world):
    e = _elig(advisor, world, "o_1001")
    assert e.eligible is True and e.days_since_signed is None
    assert "还没发货" in e.reason and "仅退款" in e.reason


def test_in_transit_supports_return_after_signing(advisor, world):
    e = _elig(advisor, world, "o_1002")
    assert e.eligible is True and e.days_since_signed is None
    assert "签收后 7 天内" in e.reason


def test_signed_within_seven_days_cites_policies(advisor, world):
    e = _elig(advisor, world, "o_1004")
    assert e.eligible is True and e.days_since_signed == 5
    assert "店铺政策" in e.reason
    assert "支持七天无理由退换货" in e.reason
    assert "七天无理由退货的运费由买家承担" in e.reason
    assert "所有订单都赠送运费险" in e.reason


def test_over_seven_days_not_for_no_reason_but_quality_ok(advisor, world):
    e = _elig(advisor, world, "o_1005")
    assert e.eligible is False and e.days_since_signed == 23
    assert "超过 7 天" in e.reason and "质量问题" in e.reason
    assert "请在申请售后时上传清晰照片" in e.reason  # 引用质量问题政策


def test_custom_item_not_returnable(advisor, world):
    e = _elig(advisor, world, "o_1007")
    assert e.eligible is False and e.days_since_signed == 2
    assert "定制" in e.reason and "质量问题" in e.reason
    assert "定制款、贴身内衣不支持七天无理由退换" in e.reason


def test_seven_day_boundary(advisor, world):
    day7 = world.now.replace(hour=0, minute=1) - timedelta(days=7)
    day8 = day7 - timedelta(minutes=2)  # 前一天 23:59
    assert _elig(advisor, world, "o_1004", signed_at=day7).eligible is True
    e8 = _elig(advisor, world, "o_1004", signed_at=day8)
    assert e8.eligible is False and e8.days_since_signed == 8


def test_config_changes_window(world):
    advisor = DefaultAfterSaleAdvisor(AfterSaleConfig(seven_day_days=3))
    e = advisor.return_eligibility(world.order(SHOP_A, "o_1004"), {}, world.now)
    assert e.eligible is False and "超过 3 天" in e.reason


def test_closed_unpaid_and_missing_sign_time(advisor, world):
    assert _elig(advisor, world, "o_1001", status=OrderStatus.CLOSED).eligible is False
    unpaid = _elig(advisor, world, "o_1001", status=OrderStatus.UNPAID)
    assert unpaid.eligible is None and "取消" in unpaid.reason
    no_sign = _elig(advisor, world, "o_1004", signed_at=None, status=OrderStatus.COMPLETED)
    assert no_sign.eligible is None and "人工" in no_sign.reason


def test_without_policies_reason_has_no_policy_citation(advisor, world):
    e = advisor.return_eligibility(world.order(SHOP_A, "o_1004"), {}, world.now)
    assert e.eligible is True and "店铺政策" not in e.reason


@pytest.mark.parametrize("status", list(AfterSaleStatus))
def test_refund_progress_every_status(advisor, world, status):
    info = world.aftersale(SHOP_A, "as_2008").model_copy(update={"status": status})
    p = advisor.refund_progress(info, world.now)
    assert p.aftersale_id == "as_2008" and p.status == status and p.as_of == world.now
    assert "售后单更新于 09-28 03:23" in p.summary
    # 不做硬承诺：不写具体到账天数
    assert "天内到账" not in p.summary and "保证" not in p.summary


def test_refund_progress_refunding_mentions_platform_rules(advisor, world):
    p = advisor.refund_progress(world.aftersale(SHOP_A, "as_2008"), world.now)
    assert p.summary.startswith("退款中：")
    assert "原路退回" in p.summary and "以平台规则为准" in p.summary
