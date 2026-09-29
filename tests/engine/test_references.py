"""指代识别的纯文本规则（方案 §5.3、§5.7）。"""

from __future__ import annotations

import pytest

from rodit.engine.references import (
    expresses_confusion,
    find_known_order_id,
    has_demonstrative,
    hinted_order_ids,
    match_skus,
    names_item,
    refers_by_sku,
    refers_to_recent_order,
    value_as_reference,
    value_mentioned,
)
from rodit.testing.fixtures import SHOP_A, build_demo_world

WORLD = build_demo_world()
DRESS = WORLD.item(SHOP_A, "i_555")  # 黑 M / 黑 L / 白 S / 白 M
JEANS = WORLD.item(SHOP_A, "i_777")  # 浅蓝 26 / 27 / 28


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("这个有M码吗", True),
        ("那件还有吗", True),
        ("这款是纯棉的吗", True),
        ("那一款呢", True),
        ("它能机洗吗", True),
        ("其它颜色有吗", False),
        ("有M码吗", False),
    ],
)
def test_demonstratives(text: str, expected: bool) -> None:
    assert has_demonstrative(text) is expected


def test_item_named_by_category_leaf() -> None:
    assert names_item(JEANS, "牛仔裤是高腰的吗")
    assert not names_item(JEANS, "连衣裙是高腰的吗")
    assert names_item(DRESS, "那条连衣裙")


@pytest.mark.parametrize(
    ("value", "text", "expected"),
    [
        ("L", "L码的有吗", True),
        ("L", "l码呢", True),  # 忽略大小写
        ("L", "ＸＬ码呢", False),  # 全角归一后是 XL，不能匹配到 L
        ("M", "165cm穿多大", False),  # cm 里的 m 不算
        ("M", "M码", True),
        ("黑色", "那件黑色的", True),
        ("黑色", "那件黑的还有吗", True),  # 口语"黑的"
        ("黑色", "黑白配", False),
        ("27", "27码的有吗", True),
        ("27", "腰围27", True),
        ("27", "27号能发货吗", False),  # 日期，不是尺码
        ("27", "穿27的", True),
        ("27", "127斤穿什么码", False),
        ("27", "我身高 165，体重 27 公斤", False),
        ("浅蓝", "浅蓝色的", True),
    ],
)
def test_value_mentioned(value: str, text: str, expected: bool) -> None:
    assert value_mentioned(value, text) is expected


def test_match_skus_by_colour_and_size() -> None:
    assert [s.sku_id for s in match_skus(DRESS, "那件黑色的L码有吗") or []] == ["k_2"]
    assert [s.sku_id for s in match_skus(DRESS, "黑色的") or []] == ["k_1", "k_2"]
    assert [s.sku_id for s in match_skus(DRESS, "白色S码") or []] == ["k_3"]
    assert [s.sku_id for s in match_skus(DRESS, "黑色和白色都要M码") or []] == ["k_1", "k_4"]
    assert match_skus(DRESS, "XL码有吗") is None  # 这件商品没有 XL：没有提到它的规格值
    assert match_skus(DRESS, "黑色S码") == []  # 提到了规格值，但没有这个组合
    assert match_skus(DRESS, "能机洗吗") is None
    assert [s.sku_id for s in match_skus(JEANS, "27码的还有吗") or []] == ["k_72"]


@pytest.mark.parametrize(
    ("value", "text", "expected"),
    [
        ("黑色", "那件黑色的还有吗", True),
        ("黑色", "黑的那件呢", True),
        ("黑色", "那件黑色", True),
        ("黑色", "黑色那件还有吗", True),
        ("L", "L码的那件", True),
        ("27", "27码的呢", True),
        ("M", "这个有M码吗", False),  # 在问尺码，不是指代
        ("黑色", "有黑色吗", False),
    ],
)
def test_value_as_reference(value: str, text: str, expected: bool) -> None:
    assert value_as_reference(value, text) is expected


def test_refers_by_sku() -> None:
    assert refers_by_sku(DRESS, "那件黑色的还有吗")
    assert not refers_by_sku(JEANS, "那件黑色的还有吗")  # 牛仔裤没有黑色
    assert not refers_by_sku(DRESS, "这个有M码吗")


@pytest.mark.parametrize(
    "text",
    ["刚买的那个到哪了", "上次买的裙子", "最近一单发了吗", "刚刚下单的那件", "昨天拍的发货没"],
)
def test_recent_order_phrases(text: str) -> None:
    assert refers_to_recent_order(text)


def test_recent_order_phrase_negative() -> None:
    assert not refers_to_recent_order("我想买一件")


def test_order_ids_from_raw_text() -> None:
    assert find_known_order_id("o_1002 到哪了", ["o_1001", "o_1002"]) == "o_1002"
    assert find_known_order_id("xo_1002 到哪了", ["o_1002"]) is None  # 必须整词
    assert find_known_order_id("到哪了", []) is None
    assert hinted_order_ids("订单号：o_1009，还有订单号 o_1009") == ["o_1009"]
    long_id = "3721958464123456789"
    assert hinted_order_ids(f"订单号是{long_id}怎么还没发") == [long_id]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("没听懂", True),
        ("你这答非所问啊", True),
        ("我不是问这个", True),
        ("你在说什么", True),
        ("看不懂", True),
        ("好的谢谢", False),
        ("什么时候发货", False),
    ],
)
def test_confusion(text: str, expected: bool) -> None:
    assert expresses_confusion(text) is expected
