"""客服工作时间（core.schedule，附录 C 第 13 题、§15 Phase 3 全自动先在非工作时间试行）。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from rodit.core.models import ShopConfig
from rodit.core.schedule import (
    MAX_SERVICE_WINDOWS,
    normalize_service_hours,
    parse_window,
    within_service_hours,
)

CST = timezone(timedelta(hours=8))


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, 6, hour, minute, tzinfo=CST)


def test_normalize_unifies_spelling_and_drops_duplicates():
    assert normalize_service_hours(["9:00 - 23:00", "09:00-23:00", "22:00~02:00"]) == [
        "09:00-23:00",
        "22:00-02:00",
    ]
    assert normalize_service_hours(["00:00-24:00"]) == ["00:00-24:00"]


@pytest.mark.parametrize(
    "text", ["", "9-23", "24:00-02:00", "09:60-10:00", "10:00-10:00", "08:00-24:30", "abc"]
)
def test_bad_windows_are_rejected(text: str):
    with pytest.raises(ValueError):
        parse_window(text)


def test_too_many_windows():
    lines = [f"{h:02d}:00-{h:02d}:30" for h in range(MAX_SERVICE_WINDOWS + 1)]
    with pytest.raises(ValueError):
        normalize_service_hours(lines)


def test_within_service_hours_including_midnight_wrap():
    day = ["09:00-23:00"]
    assert within_service_hours(day, at(9)) is True
    assert within_service_hours(day, at(22, 59)) is True
    assert within_service_hours(day, at(23)) is False  # 结束时刻不算在内
    assert within_service_hours(day, at(3)) is False
    night = ["22:00-02:00"]
    assert within_service_hours(night, at(23, 30)) is True
    assert within_service_hours(night, at(1, 59)) is True
    assert within_service_hours(night, at(2)) is False
    assert within_service_hours(["00:00-24:00"], at(4)) is True


def test_within_uses_beijing_time_and_needs_timezone():
    # UTC 14:30 = 北京时间 22:30
    assert within_service_hours(["09:00-23:00"], datetime(2026, 10, 6, 14, 30, tzinfo=UTC))
    assert not within_service_hours(["09:00-23:00"], datetime(2026, 10, 6, 15, 30, tzinfo=UTC))
    with pytest.raises(ValueError):
        within_service_hours(["09:00-23:00"], datetime(2026, 10, 6, 10, 0))


def test_no_windows_means_unknown():
    assert within_service_hours([], at(3)) is None


def test_shop_config_validates_and_normalizes_service_hours():
    assert ShopConfig(service_hours=["9:00-23:00"]).service_hours == ["09:00-23:00"]
    assert ShopConfig().auto_reply_schedule == "off_hours"
    assert ShopConfig().semi_auto_intents == []
    with pytest.raises(ValueError):
        ShopConfig(service_hours=["25:00-26:00"])
