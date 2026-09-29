"""中文数字（方案 §5.2.1 冲突判断、§5.5 数字一致性共用）。

"两天""七十二小时""七天无理由""五折"里的中文数字转成整数。core.evidence（证据冲突）和
qa.numbers（回答数字核对）都用这一份；qa.numbers 再导出 cn_to_int，保持原来的用法。

只认紧挨在单位前面的中文数字（时长、折扣、重量、金额），"第一""这一件""一块洗"不算；
星期几（"周六日""周六周日""星期二天天发"）不算；更大数字的一部分也不单独转换
（"一千二百元"原样保留，不能变成"一千200元"）。支持到百位（"一百二十元"→120）。
"""

from __future__ import annotations

import re

CN_DIGITS = "零一二两三四五六七八九十"

CN_VALUES = {
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

_WEEKDAY_DIGITS = "一二三四五六"
CN_NUMBER_BEFORE_UNIT = re.compile(
    rf"(?<!第)(?<![{CN_DIGITS}百千万])"
    # "周六日""星期二天天"：周 / 星期 / 礼拜后面的一到六是星期几，不是数字
    rf"(?!(?<=周)[{_WEEKDAY_DIGITS}])(?!(?<=星期)[{_WEEKDAY_DIGITS}])(?!(?<=礼拜)[{_WEEKDAY_DIGITS}])"
    rf"([{CN_DIGITS}百]{{1,6}})(?![{CN_DIGITS}百千万])"
    r"(?=\s*(?:个)?(?:小时|工作日|天|日|周|星期|月|年|折|斤|元))"
)
"""单位前面的中文数字；group(1) 是数字本身。"""


def cn_to_int(s: str) -> int | None:
    """ "七"→7、"十五"→15、"二十"→20、"两"→2、"一百二十"→120、"一百零五"→105、
    "一百二"→120（口语）；不支持的写法返回 None。"""
    if not s:
        return None
    if "百" in s:
        head, _, rest = s.partition("百")
        hundreds = CN_VALUES.get(head) if len(head) == 1 else None
        if not hundreds:
            return None
        if not rest:
            return hundreds * 100
        if len(rest) == 1 and rest in CN_VALUES and rest != "零":
            return hundreds * 100 + CN_VALUES[rest] * 10  # "一百二" = 120
        if rest[0] == "零":
            ones = CN_VALUES.get(rest[1:]) if len(rest) == 2 else None
            return None if not ones else hundreds * 100 + ones
        tail = cn_to_int(rest) if "十" in rest else None
        return None if tail is None or tail < 10 else hundreds * 100 + tail
    if "十" not in s:
        if len(s) != 1:
            return None
        return CN_VALUES.get(s)
    tens_part, _, ones_part = s.partition("十")
    if "十" in ones_part or len(tens_part) > 1 or len(ones_part) > 1:
        return None
    tens = 1 if not tens_part else CN_VALUES.get(tens_part)
    ones = 0 if not ones_part else CN_VALUES.get(ones_part)
    if tens is None or ones is None:
        return None
    return tens * 10 + ones


def arabize(text: str) -> str:
    """把单位前面的中文数字换成阿拉伯数字："两天内发货"→"2天内发货"、"七十二小时"→"72小时"。

    认不出的写法（"两三天"）原样保留。
    """

    def repl(m: re.Match[str]) -> str:
        n = cn_to_int(m.group(1))
        return m.group(0) if n is None else str(n)

    return CN_NUMBER_BEFORE_UNIT.sub(repl, text or "")
