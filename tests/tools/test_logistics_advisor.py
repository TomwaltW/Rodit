"""T3 物流问答规则与异常识别（方案 §7.2、§11.2、§11.3）。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from rodit.core.config import LogisticsConfig
from rodit.core.enums import AnomalyKind, LogisticsStatus, SignedDetail
from rodit.core.models import LogisticsTrace, TraceEvent
from rodit.core.ports import LogisticsAdvisor
from rodit.testing.fixtures import SHOP_A, build_demo_world
from rodit.tools.logistics import DefaultLogisticsAdvisor

STALL_ALERT = "物流超过 48 小时未更新，建议向快递催查"


@pytest.fixture
def world():
    return build_demo_world()


@pytest.fixture
def advisor():
    return DefaultLogisticsAdvisor(LogisticsConfig())


def _kinds(analysis) -> list[AnomalyKind]:
    return [a.kind for a in analysis.anomalies]


def test_is_logistics_advisor(advisor):
    assert isinstance(advisor, LogisticsAdvisor)


def test_in_transit_summary_as_of_last_event(world, advisor):
    trace = world.trace(SHOP_A, "o_1002")
    a = advisor.analyze(trace, world.now)
    assert a.status == LogisticsStatus.IN_TRANSIT
    assert a.as_of == trace.events[-1].time
    assert a.summary == "运输中，截至 09-28 10:30 已到达杭州转运中心"
    assert a.anomalies == []
    assert a.pickup_guidance is None
    assert a.eta is None  # 接口没给预计送达时间就不承诺


def test_station_signed_gives_pickup_guidance(world, advisor):
    a = advisor.analyze(world.trace(SHOP_A, "o_1003"), world.now)
    assert a.status == LogisticsStatus.SIGNED
    assert a.pickup_guidance == "包裹已放在菜鸟驿站（幸福里店），请凭取件码取件"
    assert _kinds(a) == [AnomalyKind.STATION_OR_LOCKER]
    anomaly = a.anomalies[0]
    assert anomaly.buyer_hint == a.pickup_guidance
    assert anomaly.agent_alert is None
    assert a.summary.startswith("已签收（驿站代收），截至 09-27 15:23")


def test_locker_signed_gives_pickup_guidance(world, advisor):
    a = advisor.analyze(world.trace(SHOP_A, "o_1010"), world.now)
    assert a.pickup_guidance == "包裹已放在丰巢柜（幸福里 3 号楼），请凭取件码取件"
    assert _kinds(a) == [AnomalyKind.STATION_OR_LOCKER]


def test_self_signed_has_no_anomaly(world, advisor):
    a = advisor.analyze(world.trace(SHOP_A, "o_1004"), world.now)
    assert a.status == LogisticsStatus.SIGNED and a.anomalies == [] and a.pickup_guidance is None
    assert "本人签收" in a.summary


def test_stalled_over_48_hours(world, advisor):
    a = advisor.analyze(world.trace(SHOP_A, "o_1006"), world.now)
    assert _kinds(a) == [AnomalyKind.STALLED]
    stalled = a.anomalies[0]
    assert stalled.agent_alert == STALL_ALERT
    assert stalled.buyer_hint == "已帮您催查"
    assert "74 小时" in stalled.detail


def test_exception_alerts_agent_and_is_not_stalled(world, advisor):
    a = advisor.analyze(world.trace(SHOP_A, "o_1008"), world.now)
    assert a.status == LogisticsStatus.EXCEPTION
    assert _kinds(a) == [AnomalyKind.EXCEPTION]
    assert a.anomalies[0].agent_alert and "跟进" in a.anomalies[0].agent_alert


def test_returning_alerts_agent_and_links_aftersale(world, advisor):
    a = advisor.analyze(world.trace(SHOP_A, "o_1009"), world.now)
    assert a.status == LogisticsStatus.RETURNING
    assert _kinds(a) == [AnomalyKind.RETURNING]  # 退回中不判停滞
    assert "售后" in a.anomalies[0].agent_alert


def _trace(world, *, last_ago: timedelta, status=LogisticsStatus.IN_TRANSIT, **kw):
    return LogisticsTrace(
        shop_id=SHOP_A,
        order_id="o_x",
        waybill_no="ZT0000",
        courier="中通快递",
        status=status,
        events=[
            TraceEvent(time=world.now - timedelta(days=5), description="快件已揽收"),
            TraceEvent(time=world.now - last_ago, description="到达转运中心"),
        ],
        fetched_at=world.now,
        **kw,
    )


@pytest.mark.parametrize(
    ("ago", "stalled"),
    [
        (timedelta(hours=48), False),  # 正好 48 小时：不算"超过"
        (timedelta(hours=48, minutes=1), True),
        (timedelta(hours=47, minutes=59), False),
    ],
)
def test_stall_boundary(world, advisor, ago, stalled):
    a = advisor.analyze(_trace(world, last_ago=ago), world.now)
    assert (AnomalyKind.STALLED in _kinds(a)) is stalled


@pytest.mark.parametrize(
    "status",
    [LogisticsStatus.PICKED_UP, LogisticsStatus.OUT_FOR_DELIVERY, LogisticsStatus.EXCEPTION],
)
def test_stall_applies_to_unsigned_statuses(world, advisor, status):
    a = advisor.analyze(_trace(world, last_ago=timedelta(days=4), status=status), world.now)
    assert AnomalyKind.STALLED in _kinds(a)


def test_signed_never_stalls(world, advisor):
    trace = _trace(
        world,
        last_ago=timedelta(days=10),
        status=LogisticsStatus.SIGNED,
        signed_detail=SignedDetail.SELF,
    )
    assert advisor.analyze(trace, world.now).anomalies == []


def test_stall_hours_from_config(world):
    advisor = DefaultLogisticsAdvisor(LogisticsConfig(stall_hours=24))
    a = advisor.analyze(_trace(world, last_ago=timedelta(hours=25)), world.now)
    assert a.anomalies[0].agent_alert == "物流超过 24 小时未更新，建议向快递催查"


def test_no_events_uses_fetched_at_and_never_stalls(world, advisor):
    trace = LogisticsTrace(
        shop_id=SHOP_A,
        order_id="o_x",
        waybill_no="ZT0000",
        courier="中通快递",
        status=LogisticsStatus.IN_TRANSIT,
        fetched_at=world.now - timedelta(days=9),
    )
    a = advisor.analyze(trace, world.now)
    assert a.as_of == trace.fetched_at
    assert a.summary.endswith("暂无物流轨迹")
    assert a.anomalies == []


def test_eta_only_when_platform_gives_it(world, advisor):
    eta = world.now + timedelta(days=1)
    a = advisor.analyze(
        _trace(world, last_ago=timedelta(hours=1), estimated_delivery=eta), world.now
    )
    assert a.eta == eta


def test_pickup_guidance_never_contains_pickup_code(world, advisor):
    trace = _trace(
        world,
        last_ago=timedelta(hours=2),
        status=LogisticsStatus.SIGNED,
        signed_detail=SignedDetail.STATION,
        pickup_info="兔喜驿站 取件码 5-3-2201",
    )
    a = advisor.analyze(trace, world.now)
    assert "2201" not in a.pickup_guidance
    assert "2201" not in a.anomalies[0].detail


def test_pickup_guidance_without_station_name(world, advisor):
    trace = _trace(
        world,
        last_ago=timedelta(hours=2),
        status=LogisticsStatus.SIGNED,
        signed_detail=SignedDetail.LOCKER,
    )
    assert advisor.analyze(trace, world.now).pickup_guidance == "包裹已放在快递柜，请凭取件码取件"


def test_summary_masks_pii_in_description(world, advisor):
    trace = _trace(world, last_ago=timedelta(hours=1))
    trace.events[-1] = trace.events[-1].model_copy(
        update={"description": "派送中，快递员电话13812345678"}
    )
    assert "13812345678" not in advisor.analyze(trace, world.now).summary
