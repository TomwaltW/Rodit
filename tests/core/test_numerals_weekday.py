"""中文数字转换不能把星期几、更大数字的一部分当成时长 / 金额（§5.2.1 只在真冲突时覆盖）。

复审发现："周六日"变成"周6日"（6 天）、"周六周日"变成"周6周日"（6 周）、"一百二十元"变成
"一百20元"，和店铺政策一致的 FAQ 因为这些凭空多出的数字被判冲突、从证据里删掉。
"""

from __future__ import annotations

import pytest

from rodit.core import evidence as ev
from rodit.core.numerals import arabize, cn_to_int

SHIP_48 = "现货付款后 48 小时内发货；预售商品以商品页标注的发货时间为准"


@pytest.mark.parametrize(
    "text",
    ["周六日也正常发货", "周六周日照常发货", "星期六日不休息", "礼拜六日照常发", "第一天"],
)
def test_weekday_names_are_not_numbers(text):
    assert arabize(text) == text


@pytest.mark.parametrize(
    ("text", "out"),
    [
        ("两天内发货", "2天内发货"),
        ("七十二小时", "72小时"),
        ("七天无理由", "7天无理由"),
        ("一周两天休息", "1周2天休息"),
        ("每周七天在线", "每周7天在线"),
        ("一百二十元", "120元"),
        ("两百五十元", "250元"),
        ("一百零五元", "105元"),
    ],
)
def test_numbers_before_units_still_convert(text, out):
    assert arabize(text) == out


def test_hundreds():
    assert [cn_to_int(s) for s in ("一百", "一百二十", "一百二", "一百零五", "百", "零百")] == [
        100,
        120,
        120,
        105,
        None,
        None,
    ]


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("付款后 48 小时内发货，周六日也正常发货哦", SHIP_48),
        ("48 小时内发货，周六周日照常发货", SHIP_48),
        ("亲，全店满一百二十元就包邮哦", "全店满 120 元包邮，偏远地区除外"),
    ],
)
def test_consistent_text_with_weekdays_or_large_numerals_is_not_a_conflict(lower, higher):
    assert not ev.contradicts(lower, reference=[higher], same_topic=[higher])


def test_different_large_numeral_still_conflicts():
    ref = "全店满 120 元包邮，偏远地区除外"
    assert ev.contradicts("亲，全店满一百五十元就包邮哦", reference=[ref], same_topic=[ref])


def test_weekday_is_not_tagged_as_ship_time():
    assert "ship_time" not in ev.detect_topics("周六日也正常发货")
    assert "ship_time" in ev.detect_topics("现货两天内发货")
