"""数字一致性（方案 §5.5 第一道"数字一致性"）：回答里的价格、时效、尺码、百分比、身高体重等数字，
必须能在证据里找到；买家自己说的数字只放行身高体重、尺码这类身体尺寸（价格、折扣、时长不行）。

- 先做 NFKC 归一（全角数字、全角冒号变半角），去掉千分位逗号；"48小时"和"48 小时"一样处理。
- 数值比较：259 和 259.00 相同，09 和 9 相同。
- 时间（10:30）、日期（09-28、10 月 8 日）整体比较：证据里出现同一时间 / 日期才算有依据，
  不拆成单个数字分别放行。证据里没跟时分的"10-12"可能是范围（"每包 10-12 片"），
  "1:20 兑水"是比例：日期 / 时间照样记下，里面的数字也照样算有依据；
  "10-15 分钟""10-20㎡""10-12 岁"带了单位，是范围，不是日期。
- 证据 data 里的 ISO 时间戳（2026-09-28T10:30:00+08:00）只提供日期和时间（同时按东八区折算，
  回答里的时间由 core.clock.fmt_time 按东八区写），不把年月日时分秒当普通数字，
  避免"35 元""9 日"被时间戳里的分钟、月份放行。fmt_time 写法（"截至 09-28 10:30"）也一样：
  日期、时间整体记下，月日时分不再当普通数字（"满 28 减 10，30 元"不能被它放行）。
- "30 号"（没写月份）是某月的几号：证据里有这一天的日期（任何月份）或同样的数字才算有依据。
- 买家没写单位的身高（"163 108斤想穿宽松一点选什么码"）：问尺码的消息里、所在分句没有价格
  时间用语（按词或"数字 + 单位"认，"一点""日常""夏天""到""发"不算）时算身体尺寸；
  身体尺寸的范围（"身高161-163"）两头都算。
- 回答里的"16 点""4 点半"按时间比较（"下午 4 点"也认 16:00），证据里有同一时间就算有依据。
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
from rodit.core.numerals import CN_NUMBER_BEFORE_UNIT, cn_to_int

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_ISO_DATETIME = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)
_TIME = re.compile(r"(?<![\d.])([01]?\d|2[0-4]):([0-5]\d)(?!\d)")
# "16 点""4 点半""9 点 30 分"：和证据里的 16:00 / 16:30 按时间比较（"下午 4 点"也认 16:00）
_TIME_CN = re.compile(r"(?<![\d.])([01]?\d|2[0-4])\s*点(?:\s*(半)|\s*([0-5]?\d)\s*分)?")
# "10 月 8 日"；"09-28"（fmt_time 的写法：月份两位）。"3-5 天""10-15 个工作日"是范围，不是日期
_DATE_CN = re.compile(r"(?<![\d.])(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)")
_DATE_MD = re.compile(
    r"(?<![\d.:-])(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(?![\d.:-])"
    r"(?!\s*(?:个|天|日|小时|分钟|分(?!批)|秒|周(?![一二三四五六日天末])|月|年|岁|元|块|%|号|件|码|cm|厘米|mm|毫米|m|米|平|度|期(?!间)"
    r"|斤|公斤|kg|g|克|ml|毫升|片|粒|次|倍|人))"
)
# fmt_time 写法（"截至 09-28 10:30"）：日期后面紧跟时分，整体是一个时间点
_DATE_TIME_MD = re.compile(
    r"(?<![\d.:-])(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\s+([01]?\d|2[0-4]):([0-5]\d)(?!\d)"
)
# 比例不是钟点（"按 1:20 兑水稀释""比例 1:1"）
_RATIO_BEFORE = re.compile(r"(?:比例|配比)\s*(?:为|是)?\s*$")
_RATIO_AFTER = re.compile(r"\s*(?:的?比例|配比|兑(?!换)|稀释|调配|混合|勾兑)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_CN_NUMBER = CN_NUMBER_BEFORE_UNIT  # 中文数字（core.numerals）；cn_to_int 也从这里导出
_UNIT_AFTER = re.compile(
    r"\s*(个工作日|工作日|个小时|小时|分钟|天|日|周|个月|月|年|元|块|%|cm|厘米|mm|毫米|kg|公斤|斤"
    r"|克|ml|毫升|件|个|码|折|号|岁)",
    re.IGNORECASE,
)
_IDENTIFIER_CHAR = re.compile(r"[A-Za-z_]")
# 没写月份的"30 号"（写了月份的由 _DATE_CN 整体取走）
_DAY_OF_MONTH = re.compile(r"(?<![\d.])(?<!月)(?<!月\s)(\d{1,2})\s*号")

# 身体尺寸（§5.1 尺码建议）：买家说过的这类数字，回答里复述不算编造
_BODY_UNITS = frozenset({"cm", "厘米", "mm", "毫米", "kg", "公斤", "斤", "码", "岁"})
_BODY_BEFORE = re.compile(
    r"(?:身高|体重|腰围|胸围|臀围|肩宽|脚长|脚宽|腿长|臂长|袖长|衣长|裤长|年龄)"
    r"\s*[:：]?\s*(?:是|有|约|大概|差不多)?\s*$"
)
# 买家没写单位的数字（"165 110斤穿什么码"里的 165）：消息在问尺码、
# 所在分句又不涉及价格和时间时才算身体尺寸。价格、时间用语按词或"数字 + 单位"认，
# 不用单字："想穿宽松一点""日常穿""夏天穿""161到163之间""给我发什么码"都是在问尺码
_CLAUSE_SPLIT = re.compile(r"[，,。.！!？?；;\n]")
_BODY_CONTEXT = re.compile(
    r"斤|kg|公斤|cm|厘米|身高|体重|胸围|腰围|臀围|肩宽|脚长|穿|尺码|码", re.IGNORECASE
)
_RISK_CONTEXT = re.compile(
    r"元|块钱|\d\s*块|钱|价|包邮|便宜|优惠|折扣|打折|\d\s*折|满\s*\d|减\s*\d|卖|运费|补偿|红包|券"
    r"|\d\s*(?:个)?\s*(?:小时|天|日|周|星期|月|号|点)|几号|几点|几天|多久|哪天"
    r"|发货|发出|到货|送到|送达|签收|到账|今天|明天|后天|周末|到手|少点|砍价"
)
# 身体尺寸的范围（"身高161-163""160到165cm"）：一头是身体尺寸，另一头也是
_RANGE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:到|至|[-~–—])\s*(\d+(?:\.\d+)?)(?![\d.])")


def normalize(text: str) -> str:
    """NFKC 归一并去掉千分位逗号。"""
    return _THOUSANDS.sub("", unicodedata.normalize("NFKC", text or ""))


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
    """回答里的一个数字、时间或日期。text 是带单位的原文片段，用于拦截原因。

    - day：没写月份的"30 号"里的几号；
    - clock：回答里"16 点""4 点半"这种写法的 (时, 分)，value 是钟点数（证据里写的是"16 点"时用）；
    - body：身体尺寸（单位是 cm / 斤 / 码……，或前面是"身高""体重"……），买家说过就算有依据。
    """

    text: str
    value: Decimal | None = None
    time: tuple[int, int] | None = None
    date: tuple[int, int] | None = None  # (月, 日)
    day: int | None = None
    clock: tuple[int, int] | None = None
    body: bool = False
    unit: str | None = None  # 数字后面的单位（没有单位为 None）

    @property
    def bare(self) -> bool:
        """没有单位的普通数字（不是日期、时间、几号、钟点）。"""
        return (
            self.unit is None
            and self.date is None
            and self.time is None
            and self.day is None
            and self.clock is None
        )


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    """把这些片段换成等长空格（位置不变），后面的普通数字匹配就不会再取到它们的分量。"""
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


@dataclass
class NumberSet:
    """证据里出现过的数字、时间、日期；buyer_values 是买家消息里的数字（只放行身体尺寸）。"""

    values: set[Decimal] = field(default_factory=set)
    times: set[tuple[int, int]] = field(default_factory=set)
    dates: set[tuple[int, int]] = field(default_factory=set)
    buyer_values: set[Decimal] = field(default_factory=set)

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
        # fmt_time 写法、写了月份的日期、钟点时间整体记下，再把它们遮掉：月日时分不当普通数字
        spans: list[tuple[int, int]] = []
        for m in _DATE_TIME_MD.finditer(norm):
            self.dates.add((int(m.group(1)), int(m.group(2))))
            self.times.add((int(m.group(3)), int(m.group(4))))
            spans.append(m.span())
        for m in _DATE_CN.finditer(norm):
            month, day = int(m.group(1)), int(m.group(2))
            if _valid_date(month, day):
                self.dates.add((month, day))
                spans.append(m.span())
        # 没跟时分的"10-12"可能是范围（"每包 10-12 片"）：日期记下，数字不遮
        for m in _DATE_MD.finditer(norm):
            month, day = int(m.group(1)), int(m.group(2))
            if _valid_date(month, day):
                self.dates.add((month, day))
        for m in _TIME.finditer(norm):
            if not _is_identifier(norm, m.start()):
                self.times.add((int(m.group(1)), int(m.group(2))))
                if not _is_ratio(norm, m):
                    spans.append(m.span())
        norm = _blank(norm, spans)
        self.values |= _plain_values(norm)

    def add_buyer_text(self, text: str) -> None:
        """买家自己说的身体尺寸：带身体单位（cm / 斤 / 码……）或前面是"身高""体重"……的数字，
        以及问尺码的消息里没写单位的数字（"165 110斤穿什么码"的 165；分句里不能有价格、时间用语）。
        买家说的时间、日期、价格、时长都不算依据（§5.5：价格和时效必须在证据里找到）；
        回答里也只有身体尺寸或没写单位的数字能用它们当依据（见 supports）。"""
        norm = normalize(text)
        asks_size = bool(_BODY_CONTEXT.search(norm))
        for clause in _CLAUSE_SPLIT.split(norm):
            sizing = asks_size and not _RISK_CONTEXT.search(clause)
            for token in extract_numbers(clause):
                if token.value is not None and (token.body or (sizing and token.bare)):
                    self.buyer_values.add(token.value)
            for m in _RANGE.finditer(clause):
                if _is_identifier(clause, m.start()):
                    continue
                unit = _UNIT_AFTER.match(clause, m.end())
                if _is_body(clause, m.start(), unit.group(1) if unit else None):
                    self.buyer_values |= {Decimal(m.group(1)), Decimal(m.group(2))}

    def supports(self, token: NumberToken) -> bool:
        if token.date is not None:
            return token.date in self.dates
        if token.time is not None:
            return token.time in self.times
        if token.value is None:
            return False
        if token.day is not None and any(d == token.day for _, d in self.dates):
            return True
        if token.clock is not None:
            hour, minute = token.clock
            if (hour, minute) in self.times or (hour < 12 and (hour + 12, minute) in self.times):
                return True
        if token.value in self.values:
            return True
        return (token.body or token.bare) and token.value in self.buyer_values


def _is_ratio(norm: str, m: re.Match[str]) -> bool:
    """ "按 1:20 兑水""比例 1:1"：冒号两边是比例，不是钟点。"""
    before = norm[max(0, m.start() - 6) : m.start()]
    return bool(_RATIO_BEFORE.search(before) or _RATIO_AFTER.match(norm, m.end()))


def _plain_values(norm: str) -> set[Decimal]:
    """普通数字和单位前的中文数字（编号里的数字不算）。"""
    values: set[Decimal] = set()
    for m in _NUMBER.finditer(norm):
        if _is_identifier(norm, m.start()):
            continue
        value = _to_decimal(m.group(0))
        if value is not None:
            values.add(value)
    for m in _CN_NUMBER.finditer(norm):
        n = cn_to_int(m.group(1))
        if n is not None:
            values.add(Decimal(n))
    return values


def _is_body(norm: str, start: int, unit: str | None) -> bool:
    if unit is not None and unit.lower() in _BODY_UNITS:
        return True
    return bool(_BODY_BEFORE.search(norm[max(0, start - 8) : start]))


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
    for m in _TIME_CN.finditer(norm):
        if free(m):
            taken.append(m.span())
            hour = int(m.group(1))
            minute = 30 if m.group(2) else int(m.group(3) or 0)
            token = NumberToken(text=m.group(0), value=Decimal(hour), clock=(hour, minute))
            found.append((m.start(), token))
    for m in _DAY_OF_MONTH.finditer(norm):
        day = int(m.group(1))
        if free(m) and 1 <= day <= 31:
            taken.append(m.span())
            token = NumberToken(text=f"{m.group(1)} 号", value=Decimal(day), day=day)
            found.append((m.start(), token))
    for m in _NUMBER.finditer(norm):
        value = _to_decimal(m.group(0))
        if not free(m) or value is None:
            continue
        unit = _UNIT_AFTER.match(norm, m.end())
        unit_text = unit.group(1) if unit else None
        label = m.group(0)
        if unit_text:
            label += unit_text if unit_text == "%" else f" {unit_text}"
        body = _is_body(norm, m.start(), unit_text)
        token = NumberToken(text=label, value=value, body=body, unit=unit_text)
        found.append((m.start(), token))
    for m in _CN_NUMBER.finditer(norm):
        n = cn_to_int(m.group(1))
        if n is None:
            continue
        unit = _UNIT_AFTER.match(norm, m.end())
        unit_text = unit.group(1) if unit else None
        label = m.group(1) + (unit_text or "")
        body = _is_body(norm, m.start(), unit_text)
        token = NumberToken(text=label, value=Decimal(n), body=body, unit=unit_text)
        found.append((m.start(), token))
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
