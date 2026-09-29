"""类目问题清单与知识缺失检查（§5.2.3 第 3 步、§5.8 提醒商家补知识）。"""

from __future__ import annotations

import pytest

from rodit.core import evidence as ev
from rodit.core.enums import Intent, Platform, PolicyKey
from rodit.core.errors import KnowledgeUnavailable
from rodit.core.models import ItemCard, PolicyEntry, Shop, Sku
from rodit.knowledge import (
    APPAREL_CHECKLIST,
    GENERIC_CHECKLIST,
    checklist_for,
    describe_gap,
    find_gaps,
)
from rodit.testing.fakes import make_context
from rodit.testing.fixtures import SHOP_A, SHOP_B, SHOP_C


@pytest.mark.parametrize(
    ("category", "checklist"),
    [
        ("女装", APPAREL_CHECKLIST),
        ("女装/连衣裙", APPAREL_CHECKLIST),
        ("男装", APPAREL_CHECKLIST),
        ("家电", GENERIC_CHECKLIST),
        ("食品", GENERIC_CHECKLIST),
        (None, GENERIC_CHECKLIST),
    ],
)
def test_checklist_for(category, checklist):
    assert checklist_for(category) is checklist


def test_checklists_cover_policy_keys():
    apparel = {k for e in APPAREL_CHECKLIST for k in e.policy_keys}
    generic = {k for e in GENERIC_CHECKLIST for k in e.policy_keys}
    assert apparel == set(PolicyKey) - {PolicyKey.WARRANTY}
    assert generic == set(PolicyKey) - {PolicyKey.SIZE_TENDENCY}
    assert all(e.question.strip() for e in (*APPAREL_CHECKLIST, *GENERIC_CHECKLIST))


def test_demo_shop_a_only_misses_jeans_size_advice(repo):
    # i_888 是均码定制衬衫：不要求尺码表和尺码建议；保修对服饰不要求
    assert find_gaps(repo, SHOP_A) == ["item:i_777:size_advice"]


def test_demo_shop_b_and_c(repo):
    b = find_gaps(repo, SHOP_B)
    assert "policy:ship_time" not in b and "policy:courier" not in b
    assert "policy:invoice" in b and "policy:warranty" not in b
    assert b[-2:] == ["item:i_555:size_chart", "item:i_555:care"]  # B 的商品有尺码建议
    assert "item:i_555:size_advice" not in b
    c = find_gaps(repo, SHOP_C)
    assert c[-3:] == ["item:i_c1:size_chart", "item:i_c1:size_advice", "item:i_c1:care"]
    # 只报本店的：A 店没有的缺失不会出现在 B 店里，反之亦然
    assert all(g.startswith("policy:") or g.startswith("item:i_555:") for g in b)


def test_generic_category_shop(repo, world):
    repo.upsert_shop(Shop(id="s_home", platform=Platform.PDD, category="家电"))
    repo.upsert_item_card(
        ItemCard(
            shop_id="s_home",
            platform=Platform.PDD,
            item_id="i_fan",
            title="台式小风扇",
            category="家电/风扇",
            attributes={"功率": "15W"},
            updated_at=world.now,
        )
    )
    repo.upsert_item_card(
        ItemCard(
            shop_id="s_home",
            platform=Platform.PDD,
            item_id="i_lamp",
            title="护眼台灯",
            usage=["插电后长按开关 2 秒开机"],
            updated_at=world.now,
        )
    )
    gaps = find_gaps(repo, "s_home")
    assert "policy:warranty" in gaps and "policy:size_tendency" not in gaps
    assert gaps[-2:] == ["item:i_fan:usage", "item:i_lamp:attributes"]
    assert not any("size" in g or "care" in g for g in gaps if g.startswith("item:"))


def test_attribute_alias_and_one_size(repo, world):
    repo.upsert_item_card(
        ItemCard(
            shop_id=SHOP_A,
            platform=Platform.TAOBAO,
            item_id="i_999",
            title="针织开衫",
            category="女装/针织衫",
            attributes={"材质成分": "羊毛 30%"},  # 属性名不叫"面料"也算
            skus=[Sku(sku_id="k_991", 颜色="灰色", 尺码="F")],  # 均码
            care=["平铺晾干"],
            updated_at=world.now,
        )
    )
    assert not any(g.startswith("item:i_999") for g in find_gaps(repo, SHOP_A))
    repo.upsert_item_card(
        ItemCard(
            shop_id=SHOP_A,
            platform=Platform.TAOBAO,
            item_id="i_998",
            title="开衫",
            attributes={"颜色": "灰色"},
            skus=[Sku(sku_id="k_981", 尺码="M")],
            updated_at=world.now,
        )
    )
    gaps = [g for g in find_gaps(repo, SHOP_A) if g.startswith("item:i_998")]
    # 商品没有类目时用店铺类目（女装）的清单
    assert gaps == [
        "item:i_998:attr:面料",
        "item:i_998:size_chart",
        "item:i_998:size_advice",
        "item:i_998:care",
    ]


def test_item_category_without_checklist_uses_generic(repo, world):
    """女装店里上架的非服饰商品（类目没有专门清单）按通用清单检查，不要求尺码。"""
    repo.upsert_item_card(
        ItemCard(
            shop_id=SHOP_A,
            platform=Platform.TAOBAO,
            item_id="i_997",
            title="衣物除毛器",
            category="家居/清洁工具",
            attributes={"电源": "USB 充电"},
            updated_at=world.now,
        )
    )
    gaps = [g for g in find_gaps(repo, SHOP_A) if g.startswith("item:i_997")]
    assert gaps == ["item:i_997:usage"]


def test_blank_policy_value_is_a_gap(repo, world):
    repo.upsert_policy(PolicyEntry(shop_id=SHOP_A, key="invoice", value="  ", updated_at=world.now))
    assert "policy:invoice" in find_gaps(repo, SHOP_A)


def test_gap_markers_match_retrieval_markers(repo, world, make_retriever, route):
    """检索时的缺失标记和缺失检查一致，引擎写 KnowledgeGap 时可以直接对上。"""
    ctx = make_context("穿什么码", world, item_id="i_777")
    missing = make_retriever().retrieve(ctx, route(Intent.SIZE_ADVICE)).missing
    assert missing == ["item:i_777:size_advice"]
    assert set(missing) <= set(find_gaps(repo, SHOP_A))


def test_unknown_shop_and_repository_failure(repo, flaky):
    with pytest.raises(KeyError):
        find_gaps(repo, "s_nope")
    flaky.fail_on = {"list_item_cards"}
    with pytest.raises(KnowledgeUnavailable):
        find_gaps(flaky, SHOP_A)


@pytest.mark.parametrize(
    ("marker", "text"),
    [
        ("policy:invoice", "店铺政策 · 发票（问卷第 7 题）"),
        ("policy:size_tendency", "店铺政策 · 尺码偏向（问卷第 12 题）"),
        ("item:i_777:size_advice", "商品 i_777 · 尺码建议"),
        ("item:i_1:attr:面料", "商品 i_1 · 属性 · 面料"),
        ("item:?", "会话里没有识别到当前商品"),
        ("tool:order", "tool:order"),
    ],
)
def test_describe_gap(marker, text):
    assert describe_gap(marker) == text


def test_describe_gap_for_field_of_unknown_current_item():
    """缺当前商品时的字段标记（item:?:size_advice）不显示成"商品 ? · …"。"""
    assert describe_gap(ev.missing_item_field(None, "size_advice")) == "当前商品 · 尺码建议"
