"""文字处理：个人信息最小化（含 core.pii 地址漏洞的本地补丁）、字面相关度。"""

from __future__ import annotations

import pytest

from rodit.knowledge.text import mask_text, minimized, relevance


@pytest.mark.parametrize(
    ("text", "leak"),
    [
        ("住幸福路 18 号 3 栋 201", ("幸福路", "18", "201")),
        ("寄到西湖区文三路 100 号", ("文三路", "100")),
        ("我住在阳光小区 5 栋 302", ("阳光小区", "302")),
        ("手机 138 1234 5678", ("1234",)),
    ],
)
def test_mask_text_closes_address_gaps(text, leak):
    masked = mask_text(text)
    assert all(part not in masked for part in leak), masked


def test_mask_text_keeps_non_pii_numbers():
    assert mask_text("165 110斤穿什么码") == "165 110斤穿什么码"
    assert "1234567890123456" in mask_text("订单号 1234567890123456 到哪了")


def test_minimized_leaves_clean_text_untouched():
    text = "面料：棉 95%、氨纶 5%（亲肤），可以机洗！"
    assert minimized(text) == text  # 全角标点、空格都不变
    masked = minimized("急事打 13900001111，客服 9:00–23:00")
    assert "13900001111" not in masked and "[手机号]" in masked


def test_relevance():
    assert relevance("面料亲肤吗", "面料亲肤透气。") > relevance("面料亲肤吗", "收腰设计显瘦。")
    assert relevance("", "任何文字") == 0.0
    assert 0.0 <= relevance("纯棉的吗", "纯棉") <= 1.0
