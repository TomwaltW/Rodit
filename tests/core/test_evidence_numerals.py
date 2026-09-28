"""冲突判断认中文数字和时长（§5.2.1：详情页 / FAQ 写"两天"、政策改成"72 小时"时以政策为准）。"""

from __future__ import annotations

import pytest

from rodit.core import evidence as ev

SHIP_72 = "发货时效：付款后 72 小时内发货"


def _conflict(lower: str, higher: str) -> bool:
    return ev.contradicts(lower, reference=[higher], same_topic=[higher])


def test_chinese_numeral_duration_conflicts_with_policy():
    """审查发现：FAQ "两天内发货"没有阿拉伯数字，和 72 小时政策比不出冲突。"""
    assert _conflict("问：什么时候发货\n答：您好，付款后两天内发货。", SHIP_72)
    assert _conflict("付款后二十四小时内发货", SHIP_72)
    assert _conflict("付款后 2 天内发货", SHIP_72)


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("付款后三天内发货", SHIP_72),  # 三天 = 72 小时
        ("付款后七十二小时内发货", SHIP_72),
        ("付款后 3 天内发货", SHIP_72),
        ("付款后两天内发货", "付款后 48 小时内发货"),
        ("支持七天无理由退货", "支持 7 天无理由退货"),
        ("48h 内发货", "付款后两天内发货"),
    ],
)
def test_same_duration_in_other_units_agrees(lower, higher):
    assert not _conflict(lower, higher)


def test_duration_value_must_match_a_duration_not_any_number():
    """ "2 天"不能被政策里别的数字（满 2 件）放行；非时长的数字照旧逐个比较。"""
    assert _conflict("付款后两天内发货", "付款后 72 小时内发货，满 2 件包邮")
    assert not _conflict("满 2 件包邮", "付款后 72 小时内发货，满 2 件包邮")
    assert _conflict("满 3 件包邮", "付款后 72 小时内发货，满 2 件包邮")


def test_detect_topics_understands_chinese_numerals():
    assert "ship_time" in ev.detect_topics("现货两天内发货")
    assert "ship_time" in ev.detect_topics("现货 48 小时内发货")


def test_core_numeral_helper():
    from rodit.core.numerals import cn_to_int

    assert [cn_to_int(s) for s in ("七", "两", "十五", "二十", "七十二", "")] == [
        7,
        2,
        15,
        20,
        72,
        None,
    ]
    # qa.numbers 仍然导出同一个函数（向后兼容）
    from rodit.qa import numbers

    assert numbers.cn_to_int is cn_to_int


def test_numbers_compare_by_value_not_spelling():
    """ "259 元"和实时价格"259.00 元"是同一个数，不算冲突；不同的数照常冲突。"""
    ref = ["实时价格：售价 259.00 元，活动：店铺满 199 减 20"]
    assert not ev.contradicts(
        "问：多少钱\n答：这款 259 元，满 199 减 20", reference=ref, same_topic=ref
    )
    assert ev.contradicts("问：多少钱\n答：这款 239 元", reference=ref, same_topic=ref)
