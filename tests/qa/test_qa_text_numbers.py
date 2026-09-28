"""文本清洗与数字抽取（方案 §5.5 长度与格式、数字一致性）。"""

from __future__ import annotations

from decimal import Decimal

import pytest

from rodit.qa import NumberSet, extract_numbers, strip_markdown, truncate, unsupported_numbers

# ---------------------------------------------------------------- Markdown


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("**亲**，这款是__纯棉__的", "亲，这款是纯棉的"),
        ("## 面料说明\n棉 95%", "面料说明\n棉 95%"),
        ("面料是 `棉 95%`", "面料是 棉 95%"),
        ("详情见[商品页](https://item.example.cn/1)哦", "详情见商品页哦"),
        ("- 面料：棉\n- 厚薄：常规", "面料：棉\n厚薄：常规"),
        ("1. 先洗\n2. 再晾", "先洗\n再晾"),
        ("> 引用\n正文", "引用\n正文"),
        ("这款*很*显瘦", "这款很显瘦"),
        ("~~原价~~现价", "原价现价"),
        ("第一行<br>第二行", "第一行第二行"),
        ("```\n亲，在的\n```", "亲，在的"),
    ],
)
def test_strip_markdown(raw, clean):
    out, changed = strip_markdown(raw)
    assert out == clean
    assert changed


def test_plain_text_is_untouched():
    text = "亲，这款面料是棉 95%，价格 259.00 元~ 5*3 件"
    out, changed = strip_markdown(f"  {text}\n\n")
    assert out == text
    assert not changed


# ---------------------------------------------------------------- 截断


def test_truncate_at_sentence_boundary():
    text = "亲，这款是常规厚度。白色款在强光下会略透，建议搭配浅色内搭。黑色款不透哦！"
    assert truncate(text, 100) == text
    out = truncate(text, 30)
    assert out == "亲，这款是常规厚度。白色款在强光下会略透，建议搭配浅色内搭。"
    assert len(out) <= 30


def test_truncate_falls_back_to_clause_then_hard_cut():
    clause = (
        "亲，这款连衣裙采用棉和氨纶混纺面料，版型修身显瘦，适合通勤和日常穿着，搭配小白鞋很好看"
    )
    out = truncate(clause, 25)
    assert out.endswith("…") and len(out) <= 25
    assert out == "亲，这款连衣裙采用棉和氨纶混纺面料…"
    no_punct = "这款连衣裙采用棉和氨纶混纺面料版型修身显瘦适合通勤和日常穿着"
    hard = truncate(no_punct, 10)
    assert hard == no_punct[:9] + "…" and len(hard) == 10


def test_truncate_ignores_too_short_leading_sentence():
    text = "亲！" + "这款连衣裙采用棉和氨纶混纺面料，版型修身显瘦，适合通勤和日常穿着" * 2
    out = truncate(text, 30)
    assert len(out) <= 30 and out != "亲！"


# ---------------------------------------------------------------- 数字


def _values(text: str) -> list[object]:
    return [t.time if t.time else t.value for t in extract_numbers(text)]


def test_extract_numbers_normalizes():
    assert _values("165cm、110 斤") == [Decimal(165), Decimal(110)]
    assert _values("４８小时") == [Decimal(48)]  # 全角数字
    assert _values("价格 ¥1,299.00") == [Decimal("1299.00")]
    assert _values("截至 １０：３０") == [(10, 30)]  # 全角冒号
    assert _values("七天无理由、两个工作日、第一件") == [Decimal(7), Decimal(2)]
    assert _values("订单 o_1002、运单 ZT7300000001、k_1") == []  # 编号不算
    assert _values("这一件很显瘦，和裤子一块洗就行") == []  # 量词用法不算
    assert [t.text for t in extract_numbers("72小时、95%、3 天")] == ["72 小时", "95%", "3 天"]


def test_number_set_supports():
    support = NumberSet()
    support.add_text("现货付款后 48 小时内发货；价格 259.00；客服 9:00–23:00")
    support.add_text('{"as_of": "2026-09-28T02:30:00+00:00", "sku": "k_7"}')
    assert unsupported_numbers("48 小时内发货", support) == []
    assert unsupported_numbers("只要 259 元", support) == []
    assert unsupported_numbers("09:00 到 23:00", support) == []
    # ISO 时间戳按东八区折算，时分秒不当普通数字
    assert unsupported_numbers("截至 10:30", support) == []
    assert unsupported_numbers("截至 02:30", support) == []
    assert [t.text for t in unsupported_numbers("30 元", support)] == ["30 元"]
    assert [t.text for t in unsupported_numbers("7 件", support)] == ["7 件"]  # k_7 是编号
    assert [t.text for t in unsupported_numbers("10:45", support)] == ["10:45"]
    # 同一片段只报一次
    assert len(unsupported_numbers("72 小时，72 小时", support)) == 1


def test_dates_are_compared_as_dates():
    support = NumberSet()
    support.add_text('{"as_of": "2026-09-28T10:30:00+08:00"}；预计 10 月 8 日补货')
    assert unsupported_numbers("截至 09-28 10:30", support) == []
    assert unsupported_numbers("9 月 28 日、10月8号", support) == []
    assert [t.text for t in unsupported_numbers("预计 9 月 29 日", support)] == ["9 月 29 日"]
    # 时间戳里的月份、分钟不当普通数字
    assert [t.text for t in unsupported_numbers("9 天", support)] == ["9 天"]
    assert [t.text for t in unsupported_numbers("30 元", support)] == ["30 元"]
    # 范围不是日期
    tokens = extract_numbers("3-5 天、10-15 个工作日")
    assert [t.date for t in tokens] == [None] * 4
    assert [t.value for t in tokens] == [3, 5, 10, 15]
