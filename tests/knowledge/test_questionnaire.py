"""店铺政策问卷（§5.2.3、附录 C）：题目数据、回答拆分到政策键、禁用词、语气与称呼。"""

from __future__ import annotations

from datetime import datetime

import pytest

from rodit.core.config import EngineConfig, effective_shop_config
from rodit.core.enums import Platform, PolicyKey
from rodit.core.models import SHOP_CONFIG_FIELDS, Shop
from rodit.knowledge import (
    QUESTIONNAIRE,
    QUESTIONNAIRE_BY_ID,
    apply_questionnaire,
    find_gaps,
    parse_banned_words,
    parse_tone,
    question_for_policy,
    split_policy_answer,
)
from rodit.testing.fixtures import DEFAULT_NOW

APPENDIX_C = [
    "付款后多久发货？预售商品怎么说明？",
    "默认发哪家快递？能否指定快递，要不要加钱？",
    "哪些地区不发货或需要补运费？",
    "是否支持七天无理由退货？哪些商品不支持？",
    "退货运费由谁承担？是否送运费险？",
    "质量问题怎么处理？需要买家提供哪些凭证？",
    "能开发票吗？什么类型、多久开出？",
    "有没有赠品？什么条件下送？",
    "下单后能否改地址、改规格？在什么时间之前可以改？",
    "目前有哪些优惠？客服可以承诺什么、不能承诺什么？",
    "保修期多久？怎么报修？",
    "尺码偏大还是偏小？（服饰类）",
    "客服的工作时间？",
    "哪些话绝对不能说？（商家禁用词、不能宣传的功效）",
    '回复语气偏好（亲切 / 专业 / 简洁），称呼用"亲"还是"您"？',
]

FULL_ANSWERS = {
    1: "现货付款后 48 小时内发货；预售商品以商品页标注的发货时间为准",
    2: "默认发中通快递，可以指定顺丰，需补运费差价 10 元",
    3: "新疆、西藏需补运费 15 元；港澳台及海外暂不发货",
    4: "支持七天无理由退换货，要求吊牌完好；定制款、贴身内衣不支持",
    5: "七天无理由退货运费买家承担，质量问题店铺承担；所有订单都送运费险",
    6: "请买家申请售后时上传清晰照片，核实后退款或补发",
    7: "可以开电子普通发票，确认收货后 3 个工作日内开出",
    8: "目前没有赠品活动",
    9: "发货前可以改地址和规格，发货后不能改",
    10: "店铺满 199 减 20，可以叠加平台券；客服不能承诺额外折扣、赠品或到货时间",
    11: "服饰不涉及保修",
    12: "版型偏小，建议拍大一码",
    13: "9:00–23:00",
    14: "不能说假一赔十、全网最低、“最便宜”",
    15: "亲切一点，称呼用亲",
}


def test_fifteen_questions_match_appendix_c():
    assert [q.id for q in QUESTIONNAIRE] == list(range(1, 16))
    assert [q.text for q in QUESTIONNAIRE] == APPENDIX_C
    assert QUESTIONNAIRE_BY_ID[7].text == APPENDIX_C[6]


def test_every_policy_key_is_asked_exactly_once():
    keys = [k for q in QUESTIONNAIRE for k in q.policy_keys]
    assert sorted(keys) == sorted(PolicyKey)
    assert all(q.policy_keys for q in QUESTIONNAIRE if q.id <= 13)
    assert QUESTIONNAIRE_BY_ID[14].config_fields == ("banned_words",)
    assert QUESTIONNAIRE_BY_ID[15].config_fields == ("tone", "salutation")
    assert all(set(q.config_fields) <= SHOP_CONFIG_FIELDS for q in QUESTIONNAIRE)
    assert question_for_policy(PolicyKey.INVOICE).id == 7
    assert question_for_policy("presale").id == 1
    assert question_for_policy("nope") is None


def test_apply_full_questionnaire():
    entries, patch = apply_questionnaire("s_new", FULL_ANSWERS, DEFAULT_NOW)
    values = {e.key: e.value for e in entries}
    assert set(values) == {k.value for k in PolicyKey}
    assert all(e.shop_id == "s_new" and e.updated_at == DEFAULT_NOW for e in entries)
    assert values["ship_time"] == "现货付款后 48 小时内发货"
    assert values["presale"] == "预售商品以商品页标注的发货时间为准"
    assert values["courier"] == "默认发中通快递"
    assert values["courier_choice"] == "可以指定顺丰，需补运费差价 10 元"
    assert values["seven_day_return"] == "支持七天无理由退换货，要求吊牌完好"
    assert values["non_returnable"] == "定制款、贴身内衣不支持"
    assert values["return_shipping"] == "七天无理由退货运费买家承担，质量问题店铺承担"
    assert values["shipping_insurance"] == "所有订单都送运费险"
    assert values["promotions"] == "店铺满 199 减 20，可以叠加平台券"
    assert values["promise_boundary"] == "客服不能承诺额外折扣、赠品或到货时间"
    assert values["remote_areas"] == FULL_ANSWERS[3]  # 单键题整段写入
    assert values["service_hours"] == "9:00–23:00"
    assert patch == {
        "banned_words": ["假一赔十", "全网最低", "最便宜"],
        "tone": "亲切",
        "salutation": "亲",
        "service_hours": ["09:00-23:00"],  # 第 13 题看得懂：同时写进店铺配置（北京时间）
    }
    # 补丁能直接作为店铺覆盖项，生效配置里能看到
    shop = Shop(id="s_new", platform=Platform.TAOBAO, category="女装", config_overrides=patch)
    config = effective_shop_config(EngineConfig(), shop)
    assert config.banned_words == ["假一赔十", "全网最低", "最便宜"]
    assert config.tone == "亲切" and config.salutation == "亲"


@pytest.mark.parametrize(
    ("qid", "answer", "expected"),
    [
        (1, "付款后 24 小时内发货", {PolicyKey.SHIP_TIME: "付款后 24 小时内发货"}),
        (
            1,
            "预售 15 天内发货，现货 48 小时发",
            {PolicyKey.PRESALE: "预售 15 天内发货", PolicyKey.SHIP_TIME: "现货 48 小时发"},
        ),
        (4, "不支持七天无理由退货", {PolicyKey.SEVEN_DAY_RETURN: "不支持七天无理由退货"}),
        (2, "不能指定快递", {PolicyKey.COURIER_CHOICE: "不能指定快递"}),
        (5, "送运费险", {PolicyKey.SHIPPING_INSURANCE: "送运费险"}),
        (3, "  ", {}),
    ],
)
def test_split_policy_answer(qid, answer, expected):
    assert split_policy_answer(QUESTIONNAIRE_BY_ID[qid], answer) == expected


def test_blank_and_partial_answers():
    entries, patch = apply_questionnaire("s_new", {7: "  ", 13: "9 点到 22 点"}, DEFAULT_NOW)
    assert [(e.key, e.value) for e in entries] == [("service_hours", "9 点到 22 点")]
    assert patch == {"service_hours": ["09:00-22:00"]}


def test_invalid_input():
    with pytest.raises(ValueError, match="题号"):
        apply_questionnaire("s_new", {16: "x"}, DEFAULT_NOW)
    with pytest.raises(ValueError, match="时区"):
        apply_questionnaire("s_new", {1: "48 小时"}, datetime(2026, 9, 28, 10, 0))
    with pytest.raises(TypeError):
        apply_questionnaire("s_new", {1: 48}, DEFAULT_NOW)  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("answer", "words"),
    [
        ("假一赔十、全网最低", ["假一赔十", "全网最低"]),
        ("禁用词：“最便宜”“第一品牌”，不要说防晒", ["最便宜", "第一品牌", "防晒"]),
        ("假一赔十，假一赔十", ["假一赔十"]),
        ("没有", []),
        ("无", []),
    ],
)
def test_parse_banned_words(answer, words):
    assert parse_banned_words(answer) == words


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("亲切，称呼用亲", {"tone": "亲切", "salutation": "亲"}),
        ("专业一点，称呼您", {"tone": "专业", "salutation": "您"}),
        ("简洁，不要用亲，用您", {"tone": "简洁", "salutation": "您"}),
        ("亲切", {"tone": "亲切"}),  # "亲切"里的"亲"不是称呼
        ("不要太正式，热情些", {"tone": "亲切"}),
        ("禁止使用亲，叫您", {"salutation": "您"}),
        ("随便", {}),
    ],
)
def test_parse_tone(answer, expected):
    assert parse_tone(answer) == expected


def test_questionnaire_fills_all_policy_gaps(repo):
    shop = Shop(id="s_new", platform=Platform.TAOBAO, category="女装")
    repo.upsert_shop(shop)
    before = find_gaps(repo, "s_new")
    assert len(before) == len(PolicyKey) - 1  # 服饰不要求保修
    entries, _ = apply_questionnaire("s_new", FULL_ANSWERS, DEFAULT_NOW)
    for e in entries:
        repo.upsert_policy(e)
    assert find_gaps(repo, "s_new") == []
