"""个人信息最小化的漏网格式（§5.3、§12.3）：楼宇名、没有常见关键词的小区、路名后的"北段"、
任意分组的手机号、中文数字手机号、"微 信 / 薇 / v："。

集成审查给出的原始例子用了真实地名；这里按同样的结构换成虚构的"示例市演示区样例路"等，
门牌、楼栋、房号、号码的写法保持一致（合成数据，不对应真实的人或地点）。
"""

from __future__ import annotations

import pytest

from rodit.core.pii import contains_pii, mask_pii


@pytest.mark.parametrize(
    "text,expected",
    [
        # 门牌号后面还有楼宇名 + 座 / 层 / 楼 / 室 / 区
        ("改地址到示例市演示区样例路478号演示时代广场A座1201室", "改地址到[地址]"),
        ("示例市演示区样例村东路18号演示国际大厦C座15层1508", "[地址]"),
        ("示例市演示区样例北路88号演示大厦6楼", "[地址]"),
        ("示例市样例大道北段1700号演示中心E1区", "[地址]"),  # 大道和门牌号之间有"北段"
        # 小区名里没有"小区 / 花园 / 苑……"关键词：一区、3 期、村 3 组
        ("我住在样例苑一区5幢2单元302", "我住在[地址]"),
        ("演示城3期12栋1单元1801", "[地址]"),
        ("XX村3组15号", "[地址]"),
        ("寄到样例村3组15号", "寄到[地址]"),
        # 手机号：任意分组、中文数字
        ("电话1381 234 5678", "电话[手机号]"),
        ("电话1381-234-5678", "电话[手机号]"),
        ("电话13 81 23 45 678", "电话[手机号]"),
        ("电话一三八一二三四五六七八", "电话[手机号]"),
        ("电话幺三八 幺二三四 五六七八", "电话[手机号]"),
        ("电话138一二三四五六七八", "电话[手机号]"),
        # 站外联系方式
        ("我微 信 abcd1234", "我微 信 [联系方式]"),
        ("薇：zhangsan888", "薇：[联系方式]"),
        ("v：demo_user_88", "v：[联系方式]"),
        ("加V: demo_user_88 聊", "加V: [联系方式] 聊"),
    ],
)
def test_mask_pii_masks_review_gap_formats(text, expected):
    assert mask_pii(text) == expected
    assert contains_pii(text)


def test_mask_pii_end_to_end_message_from_review():
    """审查里端到端复现用的整条消息：地址、小区、手机号都要遮掉，其余原样保留。"""
    text = (
        "改地址到示例市演示区样例路478号演示时代广场A座1201室，"
        "或者我家样例苑一区5幢2单元302，电话1381 234 5678"
    )
    masked = mask_pii(text)
    for leftover in ("1201", "A座", "时代广场", "5幢", "302", "1381", "5678"):
        assert leftover not in masked, masked
    assert masked.startswith("改地址到[地址]，或者我")
    assert masked.endswith("，电话[手机号]")


@pytest.mark.parametrize(
    "text",
    [
        "什么时候上市",
        "我知道了，谢谢",
        "这件是纯棉的吗",
        "三天内发货吗",
        "买三送一还有吗",
        "我家里3号有人收",
        "这个月里5号前能到吗",
        "预售第二期3号发货吗",
        "活动第一期5号开始吗",
        "农村5号能到吗",
        "165 110斤穿什么码",
        "身高170 体重130 穿什么码",
        "S M L XL 分别是 155 160 165 170",
        "满 199 减 20，139 元能买吗",
        "快递单号 ZT7300000001 到哪了",
        "我的订单号 3719283746519283746 到哪了",
        "订单 1381 2345 6789 能改地址吗",
        "一二三四五六是什么意思",  # 中文数字不足 7 个
        "发到城里要几天",
        "这个是薇薇安款吗",
        "dev 版本还有货吗",
    ],
)
def test_mask_pii_keeps_ordinary_product_talk(text):
    assert mask_pii(text) == text
    assert not contains_pii(text)


def test_street_number_without_building_keeps_following_question():
    """门牌号后面紧跟问题（没有楼宇名）时，只遮地址，问题原样保留。"""
    assert mask_pii("能寄到样例路100号吗我想换成广场那边") == "能寄到[地址]吗我想换成广场那边"
    assert mask_pii("寄到样例路100号可以吗") == "寄到[地址]可以吗"


def test_new_formats_are_idempotent():
    for text in (
        "改地址到示例市演示区样例路478号演示时代广场A座1201室",
        "电话一三八一二三四五六七八",
        "我微 信 abcd1234",
    ):
        once = mask_pii(text)
        assert mask_pii(once) == once
