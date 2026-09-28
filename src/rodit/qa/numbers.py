"""数字一致性（方案 §5.5 第一道"数字一致性"）：回答里的价格、时效、尺码、百分比、身高体重等数字，
必须能在证据（或买家自己说的话）里找到。

- 先做 NFKC 归一（全角数字、全角冒号变半角），去掉千分位逗号；"48小时"和"48 小时"一样处理。
- 数值比较：259 和 259.00 相同，09 和 9 相同。
- 时间（10:30）、日期（09-28、10 月 8 日）整体比较：证据里出现同一时间 / 日期才算有依据，
  不拆成单个数字分别放行。
- 证据 data 里的 ISO 时间戳（2026-09-28T10:30:00+08:00）只提供日期和时间（同时按东八区折算，
  回答里的时间由 core.clock.fmt_time 按东八区写），不把年月日时分秒当普通数字，
  避免"35 元""9 日"被时间戳里的分钟、月份放行。
- 紧跟在字母或下划线后面的数字是编号（o_1002、ZT7300000001、k_1），两边都不参与比较。
- 中文数字只在时长、折扣、重量、金额单位前识别（"三天""两个工作日""七天无理由""五折"），
  "第一"不算；"这一件""一块洗"这类量词用法不算。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation

from rodit.core.clock import CST

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_ISO_DATETIME = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)
_TIME = re.compile(r"(?<![\d.])([01]?\d|2[0-4]):([0-5]\d)(?!\d)")
# "10 月 8 日"；"09-28"（fmt_time 的写法：月份两位）。"3-5 天""10-15 个工作日"是范围，不是日期
_DATE_CN = re.compile(r"(?<![\d.])(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)")
_DATE_MD = re.compile(
    r"(?<![\d.:-])(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?![\d.:-])"
    r"(?!\s*(?:个|天|日|小时|元|块|%|号|件|码|cm|厘米|斤|公斤|kg))"
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CN_DIGITS = "零一二两三四五六七八九十"
_CN_NUMBER = re.compile(
    rf"(?<!第)([{_CN_DIGITS}]{{1,3}})(?=\s*(?:个)?(?:小时|工作日|天|日|周|星期|月|年|折|斤|元))"
)
_UNIT_AFTER = re.compile(
    r"\s*(个工作日|工作日|个小时|小时|分钟|天|日|周|个月|月|年|元|块|%|cm|厘米|mm|毫米|kg|公斤|斤"
    r"|克|ml|毫升|件|个|码|折|号|岁)",
    re.IGNORECASE,
)
_IDENTIFIER_CHAR = re.compile(r"[A-Za-z_]")

_CN_VALUES = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def normalize(text: str) -> str:
    """NFKC 归一并去掉千分位逗号。"""
    return _THOUSANDS.sub("", unicodedata.normalize("NFKC", text or ""))


def cn_to_int(s: str) -> int | None:
    """ "七"→7、"十五"→15、"二十"→20、"两"→2；不支持的写法返回 None。"""
    if not s:
        return None
    if "十" not in s:
        if len(s) != 1:
            return None
        return _CN_VALUES.get(s)
    tens_part, _, ones_part = s.partition("十")
    if "十" in ones_part or len(tens_part) > 1 or len(ones_part) > 1:
        return None
    tens = 1 if not tens_part else _CN_VALUES.get(tens_part)
    ones = 0 if not ones_part else _CN_VALUES.get(ones_part)
    if tens is None or ones is None:
        return None
    return tens * 10 + ones


def _is_identifier(text: str, start: int) -> bool:
    return start > 0 and bool(_IDENTIFIER_CHAR.match(text[start - 1]))


def _to_cst(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt.astimezone(CST) if dt.tzinfo is not None else None


def _to_decimal(s: str) -> Decimal | None:
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _valid_date(month: int, day: int) -> bool:
    return 1 <= month <= 12 and 1 <= day <= 31


@dataclass(frozen=True)
class NumberToken:
    """回答里的一个数字、时间或日期。text 是带单位的原文片段，用于拦截原因。"""

    text: str
    value: Decimal | None = None
    time: tuple[int, int] | None = None
    date: tuple[int, int] | None = None  # (月, 日)


@dataclass
class NumberSet:
    """证据和买家消息里出现过的数字、时间、日期。"""

    values: set[Decimal] = field(default_factory=set)
    times: set[tuple[int, int]] = field(default_factory=set)
    dates: set[tuple[int, int]] = field(default_factory=set)

    def add_text(self, text: str) -> None:
        norm = normalize(text)
        for m in _ISO_DATETIME.finditer(norm):
            month, day = int(m.group(2)), int(m.group(3))
            self.dates.add((month, day))
            if m.group(4) is not None:
                self.times.add((int(m.group(4)), int(m.group(5))))
                local = _to_cst(m.group(0))
                if local is not None:
                    self.dates.add((local.month, local.day))
                    self.times.add((local.hour, local.minute))
        norm = _ISO_DATETIME.sub(" ", norm)
        for pattern in (_DATE_CN, _DATE_MD):
            for m in pattern.finditer(norm):
                month, day = int(m.group(1)), int(m.group(2))
                if _valid_date(month, day):
                    self.dates.add((month, day))
        for m in _TIME.finditer(norm):
            if not _is_identifier(norm, m.start()):
                self.times.add((int(m.group(1)), int(m.group(2))))
        for m in _NUMBER.finditer(norm):
            if _is_identifier(norm, m.start()):
                continue
            value = _to_decimal(m.group(0))
            if value is not None:
                self.values.add(value)
        for m in _CN_NUMBER.finditer(norm):
            n = cn_to_int(m.group(1))
            if n is not None:
                self.values.add(Decimal(n))

    def supports(self, token: NumberToken) -> bool:
        if token.date is not None:
            return token.date in self.dates
        if token.time is not None:
            return token.time in self.times
        return token.value is not None and token.value in self.values


def extract_numbers(text: str) -> list[NumberToken]:
    """回答里的数字、时间、日期，按出现顺序。编号（字母、下划线后面的数字）不算。"""
    norm = normalize(text)
    found: list[tuple[int, NumberToken]] = []
    taken: list[tuple[int, int]] = []

    def free(m: re.Match[str]) -> bool:
        return not _is_identifier(norm, m.start()) and not any(
            m.start() < e and s < m.end() for s, e in taken
        )

    for pattern in (_DATE_CN, _DATE_MD):
        for m in pattern.finditer(norm):
            month, day = int(m.group(1)), int(m.group(2))
            if free(m) and _valid_date(month, day):
                taken.append(m.span())
                found.append((m.start(), NumberToken(text=m.group(0), date=(month, day))))
    for m in _TIME.finditer(norm):
        if free(m):
            taken.append(m.span())
            time = (int(m.group(1)), int(m.group(2)))
            found.append((m.start(), NumberToken(text=m.group(0), time=time)))
    for m in _NUMBER.finditer(norm):
        value = _to_decimal(m.group(0))
        if not free(m) or value is None:
            continue
        unit = _UNIT_AFTER.match(norm, m.end())
        label = m.group(0)
        if unit:
            label += unit.group(1) if unit.group(1) == "%" else f" {unit.group(1)}"
        found.append((m.start(), NumberToken(text=label, value=value)))
    for m in _CN_NUMBER.finditer(norm):
        n = cn_to_int(m.group(1))
        if n is None:
            continue
        unit = _UNIT_AFTER.match(norm, m.end())
        label = m.group(1) + (unit.group(1) if unit else "")
        found.append((m.start(), NumberToken(text=label, value=Decimal(n))))
    found.sort(key=lambda x: x[0])
    return [token for _, token in found]


def unsupported_numbers(text: str, support: NumberSet) -> list[NumberToken]:
    """回答里在 support 中找不到的数字（按原文片段去重）。"""
    out: list[NumberToken] = []
    seen: set[str] = set()
    for token in extract_numbers(text):
        if support.supports(token) or token.text in seen:
            continue
        seen.add(token.text)
        out.append(token)
    return out
