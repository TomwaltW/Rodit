"""售后审核时长（§2.1 抖店"售后平均审核时长"、§7.4）：rodit.aftersale.durations 的纯函数。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rodit.aftersale.durations import (
    CREATED_AT_KEY,
    DECISION_STATES,
    DurationSummary,
    application_times,
    decision_times,
    first_decision,
    merge_decisions,
    review_duration,
    summarize,
    waiting_seconds,
)
from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import ReviewState
from rodit.core.models import AuditEvent

T0 = datetime(2026, 9, 28, 10, 0, tzinfo=CST)


def _step(frm: str, to: str, at: datetime | str, by: str | None = None) -> dict:
    entry: dict = {"from": frm, "to": to, "at": at.isoformat() if isinstance(at, datetime) else at}
    if by is not None:
        entry["by"] = by
    return entry


def _action(
    aftersale_id: str,
    at: datetime,
    *,
    ok: bool = True,
    duplicate: bool = False,
    actor: str = "kf_02",
    shop_id: str = "s_1001",
    action: str = AuditAction.REVIEW_ACTION.value,
) -> AuditEvent:
    return AuditEvent(
        shop_id=shop_id,
        actor=actor,
        action=action,
        detail={"aftersale_id": aftersale_id, "ok": ok, "duplicate": duplicate},
        created_at=at,
    )


# ================================================================ 第一次由人确认执行


def test_decision_states_are_what_confirmation_leads_to():
    assert DECISION_STATES == {ReviewState.AGREED, ReviewState.REFUSED, ReviewState.WAIT_RETURN}


def test_first_decision_is_the_first_human_confirmation():
    history = [
        _step("new", "suggest", T0),
        _step("suggest", "confirm", T0),
        _step("confirm", "negotiate", T0 + timedelta(hours=1), by="kf_02"),
        _step("negotiate", "confirm", T0 + timedelta(hours=2), by="kf_02"),
        _step("confirm", "wait_return", T0 + timedelta(hours=3), by="kf_02"),
        _step("wait_return", "confirm", T0 + timedelta(hours=30)),
        _step("confirm", "agreed", T0 + timedelta(hours=31), by="kf_03"),
    ]
    # R3：同意退货是第一次审核；之后同意退款是同一张单的第二次操作，不另算
    assert first_decision(history) == T0 + timedelta(hours=3)


def test_first_decision_ignores_system_moves_and_broken_entries():
    history = [
        # 系统发现商家在平台后台同意了退货：没有操作人，不是工作台里的确认
        _step("confirm", "wait_return", T0),
        _step("confirm", "agreed", T0 + timedelta(hours=1), by="system"),
        _step("confirm", "agreed", T0 + timedelta(hours=1), by="  "),
        _step("confirm", "refused", "不是时间", by="kf_02"),
        _step("confirm", "refused", "2026-09-28T12:00:00", by="kf_02"),  # 没有时区
        {"from": "confirm", "to": "agreed"},  # 缺 at
        "不是字典",
        _step("verify", "confirm", T0 + timedelta(hours=2), by="kf_02"),  # 不是确认执行
        _step("confirm", "platform", T0 + timedelta(hours=2), by="kf_02"),  # 平台已处理不算
    ]
    assert first_decision(history) is None
    assert first_decision([]) is None
    later = _step("confirm", "refused", T0 + timedelta(hours=5), by="kf_02")
    earlier = _step("confirm", "agreed", T0 + timedelta(hours=4), by="kf_manager")
    assert first_decision([*history, later, earlier]) == T0 + timedelta(hours=4)


def test_first_decision_accepts_other_timezones():
    at = datetime(2026, 9, 28, 2, 0, tzinfo=UTC)
    assert first_decision([_step("confirm", "agreed", at, by="kf_02")]) == at


# ================================================================ 审计事件


def test_decision_times_take_the_first_successful_platform_call_per_aftersale():
    events = [
        _action("as_1", T0 + timedelta(hours=2)),
        _action("as_1", T0 + timedelta(hours=1)),
        _action("as_2", T0, ok=False),  # 平台返回失败
        _action("as_2", T0 + timedelta(hours=3), duplicate=True),  # 幂等键重复，没有再调平台
        _action("as_3", T0, actor="system"),
        _action("as_4", T0, action=AuditAction.REVIEW_CONFIRM.value),  # 确认事件不是平台结果
        _action("as_5", T0 + timedelta(minutes=5)),
    ]
    assert decision_times(events) == {
        "as_1": T0 + timedelta(hours=1),
        "as_5": T0 + timedelta(minutes=5),
    }


def test_decision_times_skip_events_without_an_aftersale_id():
    bad = AuditEvent(
        shop_id="s_1001",
        actor="kf_02",
        action=AuditAction.REVIEW_ACTION.value,
        detail={"ok": True},
        created_at=T0,
    )
    assert decision_times([bad]) == {}


def _with_created(event: AuditEvent, created: object) -> AuditEvent:
    return event.model_copy(update={"detail": {**event.detail, CREATED_AT_KEY: created}})


def test_application_times_come_from_the_confirmation_audit():
    """申请时间取确认 / 平台结果审计里记下的（统计时不读平台）；没记、格式不对的跳过。"""
    applied = T0 - timedelta(hours=5)
    events = [
        _with_created(_action("as_1", T0, ok=False), applied.isoformat()),  # 失败的也记了申请时间
        _with_created(_action("as_2", T0, action=AuditAction.REVIEW_CONFIRM.value), "2026-09-01"),
        _with_created(_action("as_3", T0), "not-a-time"),
        _action("as_4", T0),  # 加这个字段之前确认的
        _with_created(
            _action("as_5", T0, action=AuditAction.REVIEW_SUGGEST.value), applied.isoformat()
        ),
        _with_created(_action("as_6", T0, action=AuditAction.REVIEW_CONFIRM.value), applied),
    ]
    assert application_times(events) == {"as_1": applied, "as_6": applied}


def test_merge_decisions_keeps_the_earliest():
    a = {"as_1": T0 + timedelta(hours=2), "as_2": T0}
    b = {"as_1": T0 + timedelta(hours=1), "as_3": T0 + timedelta(hours=5)}
    assert merge_decisions(a, b) == {
        "as_1": T0 + timedelta(hours=1),
        "as_2": T0,
        "as_3": T0 + timedelta(hours=5),
    }
    assert merge_decisions() == {}


# ================================================================ 时长与汇总


def test_review_duration_never_goes_negative():
    assert review_duration(T0, T0 + timedelta(hours=3)) == timedelta(hours=3)
    assert review_duration(T0, T0 - timedelta(minutes=1)) == timedelta(0)


def test_waiting_seconds():
    assert waiting_seconds(T0, T0 + timedelta(hours=1, seconds=30.9)) == 3630
    assert waiting_seconds(T0, T0 - timedelta(seconds=5)) == 0


def test_summarize_empty_has_no_numbers():
    assert summarize([]) == DurationSummary()
    assert summarize([]).count == 0 and summarize([]).average_s is None


@pytest.mark.parametrize(
    ("hours", "median", "p90"),
    [
        ([1], 1, 1),
        ([1, 3], 2, 3),
        ([4, 1, 2], 2, 4),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5.5, 9),
    ],
)
def test_summarize_reports_average_median_p90_and_max(hours, median, p90):
    s = summarize(timedelta(hours=h) for h in hours)
    assert s.count == len(hours)
    assert s.average_s == pytest.approx(sum(hours) / len(hours) * 3600)
    assert s.median_s == pytest.approx(median * 3600)
    assert s.p90_s == pytest.approx(p90 * 3600)  # 最近秩法
    assert s.max_s == pytest.approx(max(hours) * 3600)
