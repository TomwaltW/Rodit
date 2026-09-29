"""时间。业务代码一律通过 Clock 取当前时间，测试里换成 testing.fakes.FakeClock。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

CST = timezone(timedelta(hours=8), name="CST")  # 东八区；不依赖系统 tzdata


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(CST)


def fmt_time(dt: datetime, *, with_date: bool = True) -> str:
    """回答里的时间写法，如 "09-28 10:30"；with_date=False 时只写 "10:30"。"""
    local = dt.astimezone(CST)
    return local.strftime("%m-%d %H:%M" if with_date else "%H:%M")
