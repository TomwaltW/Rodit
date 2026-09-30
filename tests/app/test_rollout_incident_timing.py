"""复核人员登记的 0 只在那一周结束后登记才算检查过整周（方案 §13.4 "平台违规为 0""没有因此
产生的投诉"、§15 验收）：周一早上登记的 0 只说明到登记时还没有，平台通知常常滞后几天。
演示单元 + 合成数据（演示时钟在 2026-09-28 周一 10:35，W40）。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit
from rodit.app.rollout import acceptance_input
from rodit.core.clock import CST
from rodit.core.enums import Platform
from rodit.feedback.rollout import (
    IncidentEntry,
    IncidentKind,
    incident_total,
    provisional_zero,
)
from rodit.testing.fixtures import SHOP_A

REVIEWER = "qa_reviewer_01"
W40, W41 = "2026-W40", "2026-W41"
W40_START = datetime(2026, 9, 28, tzinfo=CST)
W41_END = W40_START + timedelta(days=14)


@pytest.fixture
def service() -> AssistService:
    return AssistService(build_demo_unit(Platform.TAOBAO))


def entry(week: str, count: int, at: datetime, kind=IncidentKind.VIOLATION) -> IncidentEntry:
    return IncidentEntry(
        shop_id=SHOP_A, kind=kind, week=week, count=count, reviewer=REVIEWER, recorded_at=at
    )


def test_a_zero_registered_before_the_week_ended_does_not_check_the_week(service: AssistService):
    ro = service.rollout
    assert service.unit.clock.now() < W40_START + timedelta(days=1)  # W40 周一
    ro.record_incident(SHOP_A, kind="violation", week=W40, count=0, reviewer=REVIEWER)
    service.unit.clock.advance(days=14)  # W42 周一：W40、W41 都结束了
    ro.record_incident(SHOP_A, kind="violation", week=W41, count=0, reviewer=REVIEWER)
    status = ro.status(SHOP_A)
    pv = status.evidence.platform_violations
    assert pv.weeks == [W40, W41]
    assert pv.missing == [W40] and pv.provisional == [W40] and pv.count is None
    assert W40 in (pv.reason or "") and "结束后" in (pv.reason or "")
    assert W40 not in pv.by_week and pv.by_week[W41] == 0
    [unchecked] = [u for u in status.unchecked if u.name == "platform_violations"]
    assert "结束后" in (unchecked.reason or "")
    # 那一周结束后再登记一次确认：算检查过
    ro.record_incident(SHOP_A, kind="violation", week=W40, count=0, reviewer=REVIEWER)
    pv = ro.status(SHOP_A).evidence.platform_violations
    assert pv.missing == [] and pv.provisional == [] and pv.count == 0 and pv.reason is None


def test_a_count_above_zero_counts_whenever_it_was_registered(service: AssistService):
    ro = service.rollout
    ro.record_incident(SHOP_A, kind="violation", week=W40, count=2, reviewer=REVIEWER)
    service.unit.clock.advance(days=14)
    pv = ro.status(SHOP_A).evidence.platform_violations
    assert pv.count == 2 and pv.by_week[W40] == 2 and pv.provisional == []


def test_a_correction_to_zero_mid_week_is_still_provisional(service: AssistService):
    ro = service.rollout
    ro.record_incident(SHOP_A, kind="violation", week=W40, count=1, reviewer=REVIEWER)
    service.unit.clock.advance(days=2)
    ro.record_incident(SHOP_A, kind="violation", week=W40, count=0, reviewer=REVIEWER)
    service.unit.clock.advance(days=12)
    ro.record_incident(SHOP_A, kind="violation", week=W41, count=0, reviewer=REVIEWER)
    pv = ro.status(SHOP_A).evidence.platform_violations
    assert pv.provisional == [W40] and pv.count is None


def test_complaints_follow_the_same_rule(service: AssistService):
    ro = service.rollout
    ro.record_incident(SHOP_A, kind="complaint", week=W40, count=0, reviewer=REVIEWER)
    service.unit.clock.advance(days=7 * 5)  # W45 周一：W41–W44 是全自动门槛的 4 周
    for week in ("2026-W41", "2026-W42", "2026-W43", "2026-W44"):
        ro.record_incident(SHOP_A, kind="complaint", week=week, count=0, reviewer=REVIEWER)
    complaints = ro.status(SHOP_A).evidence.complaints
    assert complaints.count == 0 and complaints.missing == []
    early = incident_total(
        [entry("2026-W44", 0, datetime(2026, 10, 26, 9, tzinfo=CST), IncidentKind.COMPLAINT)],
        IncidentKind.COMPLAINT,
        ["2026-W44"],
        as_of=datetime(2026, 11, 2, 12, tzinfo=CST),
    )
    assert early.count is None and early.provisional == ["2026-W44"]


def test_acceptance_uses_the_window_end(service: AssistService):
    ro = service.rollout
    ro.record_incident(SHOP_A, kind="violation", week=W40, count=0, reviewer=REVIEWER)
    service.unit.clock.advance(days=14)
    ro.record_incident(SHOP_A, kind="violation", week=W41, count=0, reviewer=REVIEWER)
    shop = service.unit.repos.knowledge.get_shop(SHOP_A)
    data = acceptance_input(
        service.unit.repos, service.unit.config, shop, since=W40_START, until=W41_END
    )
    assert data.violations.weeks == [W40, W41]
    assert data.violations.missing == [W40] and data.violations.count is None


def test_a_week_that_has_not_ended_at_the_evaluation_time_keeps_the_old_rule():
    """不到一周的验收窗口：要登记的是还没结束的那一周，只能看到窗口终点为止——登记了 0 就算。"""
    at = W40_START + timedelta(days=2)
    zero = entry(W40, 0, at)
    assert not provisional_zero(zero, as_of=W40_START + timedelta(days=3))
    assert provisional_zero(zero, as_of=W40_START + timedelta(days=7))
    assert provisional_zero(zero)  # 不给时间点：按登记时那一周有没有结束
    assert not provisional_zero(entry(W40, 3, at))
    assert not provisional_zero(entry(W40, 0, W40_START + timedelta(days=7)))  # 结束那一刻登记
    short = incident_total(
        [zero], IncidentKind.VIOLATION, [W40], as_of=W40_START + timedelta(days=3)
    )
    assert short.count == 0 and short.missing == []
