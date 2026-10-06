"""客服工作时间（附录 C 第 13 题、§15 Phase 3 "全自动先在非工作时间试行"）。

ShopConfig.service_hours 是一组"HH:MM-HH:MM"时段（北京时间），例如 ["09:00-23:00"]；
结束早于开始表示跨午夜（"22:00-02:00"），结束可以写 24:00。

- parse_window / parse_service_hours：校验并解析，格式不对抛 ValueError（ShopConfig 用它校验）；
- within_service_hours：某个时刻是否在工作时间内；没配置时段时返回 None（不知道），调用方自己定
  怎么处理（全自动按"不知道就不自动"处理，§4.1 原则 3）。

平台的考核时段（如拼多多 3 分钟人工回复率只考核 8:00–23:00）不在这里，见 core.platform_metrics。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

SERVICE_TZ = timezone(timedelta(hours=8), "CST")
"""工作时间按北京时间算（三个平台的考核时段都是北京时间）。"""

MAX_SERVICE_WINDOWS = 6
"""一家店最多配几个时段（如上午、下午、晚上分开）。"""

_WINDOW = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*[-–~至]\s*(\d{1,2}):(\d{2})\s*$")


@dataclass(frozen=True)
class ServiceWindow:
    """一个工作时段：从一天开始算的分钟数，end 可以小于 start（跨午夜）。"""

    start: int  # 0..1439
    end: int  # 1..1440

    @property
    def crosses_midnight(self) -> bool:
        return self.end < self.start

    def contains(self, minute: int) -> bool:
        if self.crosses_midnight:
            return minute >= self.start or minute < self.end
        return self.start <= minute < self.end

    def text(self) -> str:
        return f"{_hhmm(self.start)}-{_hhmm(self.end)}"


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def parse_window(text: str) -> ServiceWindow:
    """一个时段的文字（如 "09:00-23:00"）→ ServiceWindow。

    开始不能是 24:00，开始和结束不能相同（全天就写 00:00-24:00）。"""
    m = _WINDOW.match(text or "")
    if not m:
        raise ValueError(f"工作时段格式应为 HH:MM-HH:MM：{text!r}")
    sh, sm, eh, em = (int(g) for g in m.groups())
    if sm > 59 or em > 59 or sh > 23 or eh > 24 or (eh == 24 and em != 0):
        raise ValueError(f"工作时段里的时间不对：{text!r}")
    start, end = sh * 60 + sm, eh * 60 + em
    if start == end:
        raise ValueError(f"工作时段的开始和结束不能相同：{text!r}")
    return ServiceWindow(start=start, end=end)


def parse_service_hours(lines: Iterable[str]) -> list[ServiceWindow]:
    windows = [parse_window(line) for line in lines]
    if len(windows) > MAX_SERVICE_WINDOWS:
        raise ValueError(f"工作时段最多 {MAX_SERVICE_WINDOWS} 个")
    return windows


def normalize_service_hours(lines: Iterable[str]) -> list[str]:
    """校验并统一写法（"9:00 - 23:00" → "09:00-23:00"），保持顺序、去掉重复。"""
    seen: list[str] = []
    for window in parse_service_hours(lines):
        text = window.text()
        if text not in seen:
            seen.append(text)
    return seen


def within_service_hours(lines: Iterable[str], at: datetime) -> bool | None:
    """at（带时区）是否在工作时间内；没配置时段返回 None。"""
    if at.tzinfo is None:
        raise ValueError("时间必须带时区")
    windows = parse_service_hours(lines)
    if not windows:
        return None
    local = at.astimezone(SERVICE_TZ)
    minute = local.hour * 60 + local.minute
    return any(w.contains(minute) for w in windows)


__all__ = [
    "MAX_SERVICE_WINDOWS",
    "SERVICE_TZ",
    "ServiceWindow",
    "normalize_service_hours",
    "parse_service_hours",
    "parse_window",
    "within_service_hours",
]
