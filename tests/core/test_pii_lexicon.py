"""个人信息最小化（§5.3）与共享词表（§5.5、§6.6 规则 1）。

测试用的证件号地区码是 99（不存在），地址是虚构的"测试省示例市"等，不对应真实的人或地点。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rodit.core.lexicon import (
    find_absolute_terms,
    find_off_platform_in_answer,
    find_off_platform_request,
)
from rodit.core.pii import BuyerRefHasher, contains_pii, hash_buyer_id, mask_pii


def _synthetic_id(prefix17: str) -> str:
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    check = "10X98765432"[sum(int(c) * w for c, w in zip(prefix17, weights, strict=True)) % 11]
    return prefix17 + check


SYNTHETIC_ID = _synthetic_id("99010119900101123")  # 地区码 99 不存在


@pytest.mark.parametrize(
    "text,expected",
    [
        ("我手机 13812345678 你打给我", "我手机 [手机号] 你打给我"),
        ("电话 +86 138-1234-5678", "电话 [手机号]"),
        ("电话１３８１２３４５６７８", "电话[手机号]"),  # 全角数字
        ("138.1234.5678", "[手机号]"),
        ("座机 0571-88886666", "座机 [手机号]"),
        (f"身份证 {SYNTHETIC_ID}", "身份证 [证件号]"),
        ("邮箱 abc.def@example.com", "邮箱 [邮箱]"),
        ("发票发到邮箱abc@example.com可以吗", "发票发到邮箱[邮箱]可以吗"),
        ("卡号 4111111111111111", "卡号 [卡号]"),
        ("加我微信 demo_user_2026", "加我微信 [联系方式]"),
        ("QQ 2345678901", "QQ [联系方式]"),
        ("vx: abc12345 加一下", "vx: [联系方式] 加一下"),
        ("寄到测试省示例市演示区样例路 1 号 3 栋", "寄到[地址]"),
        ("改地址到演示区样例路100号3栋2单元501", "改地址到[地址]"),
        ("送到示例市演示区样例路 100 号", "送到[地址]"),
        ("我住演示花园小区5栋1单元302室", "我住[地址]"),
        ("地址改成示例市演示区样例路88号", "地址改成[地址]"),
        ("我想改地址测试省示例市演示区样例路100号", "我想改地址[地址]"),
        ("地址：示例市演示区科技园样例路88号", "地址:[地址]"),
        ("改成演示区样例路88号院2号楼", "改成[地址]"),
        ("能寄到测试省示例市演示区样例路100号吗 顺丰要多久", "能寄到[地址]吗 顺丰要多久"),
    ],
)
def test_mask_pii_masks(text, expected):
    assert mask_pii(text) == expected
    assert contains_pii(text)


@pytest.mark.parametrize(
    "text",
    [
        "165 110斤穿什么码",
        "快递单号 ZT7300000001 到哪了",
        "满 199 减 20 吗",
        "发到北京市几号能到",
        "寄到上海市大概几号到啊",
        "深圳市的仓库发货吗几号能到",
        "杭州市今天能发货吗我想知道",
        "你们是哪个城市发货 几号能到",
        "什么时候上市 几号有货",
        "怎么能省点 活动到几号",
        "这个颜色在城市里穿会不会太亮 M号还有吗",
        "这款什么时候上市我想知道几号发货",
        "我在超市看到同款比你们便宜你知道吗",
        "我的订单号 3719283746519283746 到哪了",  # 平台订单号保留
        "订单 230928123456789012 能改地址吗",
        "订单 2837465019283746501 到哪了",
        "我知道3号能到",
        "新疆能到吗",
        "腰围 70cm 胸围 88 穿 M 吗",
        "能加个腰带吗",
    ],
)
def test_mask_pii_keeps_normal_text(text):
    assert mask_pii(text) == text
    assert not contains_pii(text)


def test_mask_pii_is_idempotent():
    for text in ("我手机 13812345678", "改地址到演示区样例路100号3栋"):
        once = mask_pii(text)
        assert mask_pii(once) == once


def test_no_realistic_id_numbers_in_repo_fixtures():
    """合成数据守卫：测试和夹具里不出现地区码有效、校验位正确的 18 位证件号。"""
    root = Path(__file__).resolve().parents[2]
    pattern = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    for path in [*(root / "tests").rglob("*.py"), *(root / "src/rodit/testing").rglob("*.py")]:
        for m in pattern.finditer(path.read_text(encoding="utf-8")):
            s = m.group(0)
            ok = "10X98765432"[sum(int(c) * w for c, w in zip(s[:17], weights, strict=True)) % 11]
            if s[17].upper() == ok:
                assert s.startswith("99"), f"{path}: 疑似真实证件号 {s}"


def test_hash_buyer_id_requires_salt_and_is_scoped():
    a = hash_buyer_id("taobao", "tb_buyer_1", salt="s1")
    assert a == hash_buyer_id("taobao", "tb_buyer_1", salt="s1")
    assert a != hash_buyer_id("douyin", "tb_buyer_1", salt="s1")
    assert a != hash_buyer_id("taobao", "tb_buyer_1", salt="s2")
    assert a.startswith("b_") and len(a) == 26 and "tb_buyer_1" not in a
    with pytest.raises(ValueError):
        hash_buyer_id("taobao", "x", salt="")
    hasher = BuyerRefHasher("s1")
    assert hasher("taobao", "tb_buyer_1") == a
    assert "s1" not in repr(hasher)


def test_buyer_ref_hasher_from_env(monkeypatch):
    monkeypatch.delenv("RODIT_BUYER_SALT", raising=False)
    with pytest.raises(RuntimeError):
        BuyerRefHasher.from_env()
    monkeypatch.setenv("RODIT_BUYER_SALT", "from-env")
    assert BuyerRefHasher.from_env()("pdd", "x") == hash_buyer_id("pdd", "x", salt="from-env")


# ---------------------------------------------------------------- 词表


def test_doc_approved_answers_have_no_absolute_terms():
    """§11.2、§11.5 里方案认可的回答不能被绝对化用语规则拦截。"""
    approved = [
        "亲，包裹目前在运输中，截至 10:30 已到达杭州转运中心，"
        "我们会持续关注，有更新第一时间告诉您。",
        "您的包裹正在派送中，截至 10:30 的最新状态是派送中。",
        "包裹已放在最近的菜鸟驿站，请凭取件码取件。",
        "这款不一定适合所有人，建议参考尺码表。",
    ]
    for text in approved:
        assert find_absolute_terms(text) == [], text


def test_absolute_terms_are_caught():
    assert find_absolute_terms("亲，明天一定能到哦！全网最低价") == ["一定", "最"]
    assert find_absolute_terms("绝对是第一名，100%纯棉") == ["绝对", "第一", "100%"]
    assert find_absolute_terms("质量最好", exceptions=[]) == ["最"]


def test_off_platform_request_vs_answer():
    assert find_off_platform_request("加个微信吧，转账给你便宜点")
    assert find_off_platform_request("加个v吧")
    assert find_off_platform_request("能支付宝转账吗")
    assert not find_off_platform_request("这个是纯棉的吗")
    script = (
        "亲，平台规定不能通过站外联系和交易哦，您直接在店铺下单就好，"
        "我帮您看看有没有可以用的优惠券。"
    )
    assert find_off_platform_in_answer(script) == []  # §11.4 标准话术
    assert find_off_platform_in_answer("加我微信 demo123 转账便宜")
    assert find_off_platform_in_answer("详情见 https://example.com")
    assert find_off_platform_in_answer("可以打我电话 13812345678")
    assert find_off_platform_in_answer("可以支付宝转账给我")
    assert not find_off_platform_in_answer("可以再加个腰带哦")
