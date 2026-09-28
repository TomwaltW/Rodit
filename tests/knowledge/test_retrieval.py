"""找依据（§5.2.4）：各问题类型在演示世界上的证据、冲突处理（§5.2.1）、澄清、工具证据合并、
长度预算、知识库故障（§12.2）、店铺隔离（§9.2）、个人信息最小化（§5.3）。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from rodit.core import evidence as ev
from rodit.core.config import GenerationConfig
from rodit.core.enums import EvidenceSource, Intent, KnowledgeSource, KnowledgeStatus, PolicyKey
from rodit.core.errors import KnowledgeUnavailable, TenantIsolationError, ToolUnavailable
from rodit.core.evidence import Need
from rodit.core.models import Evidence, EvidenceBundle, KnowledgeItem, PolicyEntry, Sku
from rodit.core.ports import EvidenceRetriever
from rodit.inmemory import HashingEmbedder
from rodit.knowledge import CLARIFY_ITEM_QUESTION, FaqSearcher, route_intents
from rodit.knowledge.items import CONSERVATIVE_NOTE
from rodit.testing.fakes import make_context
from rodit.testing.fixtures import SHOP_A, SHOP_B, SHOP_C, SHOP_D


def _ids(bundle: EvidenceBundle) -> list[str]:
    return bundle.ids()


def _assert_well_formed(bundle: EvidenceBundle) -> None:
    """所有证据按优先级排列、ID 不重复、被覆盖的不在证据里。"""
    prios = [e.priority for e in bundle.evidences]
    assert prios == sorted(prios)
    assert len(set(_ids(bundle))) == len(bundle.evidences)
    assert not set(bundle.overridden_ids) & set(_ids(bundle))


class FakeToolProvider:
    """最小的 ToolEvidenceProvider 替身：按预设返回，记录调用。"""

    def __init__(self, bundle: EvidenceBundle | None = None, *, fail: Exception | None = None):
        self.bundle = bundle or EvidenceBundle()
        self.fail = fail
        self.calls: list[tuple[str, Intent, frozenset[Need]]] = []

    def collect(self, ctx, intent, needs):
        self.calls.append((ctx.shop_id, intent, needs))
        if self.fail is not None:
            raise self.fail
        return self.bundle.model_copy(deep=True)


def _order_evidence(world, order_id="o_1001") -> Evidence:
    order = world.order(SHOP_A, order_id)
    return Evidence(
        id=ev.tool_order(order_id),
        source=EvidenceSource.REALTIME,
        label=ev.tool_label("order"),
        content=f"订单 {order_id}：已付款，未发货",
        data={"order": order.model_dump(mode="json")},
        fetched_at=world.now,
    )


def test_implements_evidence_retriever_protocol(make_retriever):
    assert isinstance(make_retriever(), EvidenceRetriever)


def test_route_intents_union_and_dedupe(route):
    r = route(Intent.PRODUCT_SPEC, Intent.PRODUCT_SPEC, Intent.RETURN_POLICY)
    assert route_intents(r) == [Intent.PRODUCT_SPEC, Intent.RETURN_POLICY]


# ================================================================ 商品参数


def test_product_spec_takes_only_the_asked_attribute(world, make_retriever, route):
    ctx = make_context("这个是纯棉的吗", world, item_id="i_555", sku_id="k_1")
    b = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))
    _assert_well_formed(b)
    ids = _ids(b)
    assert ids[:3] == ["item:i_555:title", "item:i_555:attr:面料", "item:i_555:sku"]
    assert "item:i_555:attr:厚薄" not in ids and "item:i_555:attr:版型" not in ids
    fabric = b.get("item:i_555:attr:面料")
    assert fabric.source == EvidenceSource.ITEM
    assert fabric.label == "商品属性 · 面料"
    assert fabric.data == {"name": "面料", "value": "棉 95%、氨纶 5%"}
    assert "棉 95%" in fabric.content
    title = b.get("item:i_555:title")
    assert title.data == {"title": "法式碎花连衣裙"}
    sku = b.get("item:i_555:sku")
    assert [s["sku_id"] for s in sku.data["skus"]] == ["k_1", "k_2", "k_3", "k_4"]
    assert set(sku.data) == {"skus"}
    assert "买家当前选择的规格：黑色 / M" in sku.content and "实时" in sku.content
    assert b.missing == [] and b.clarify_question is None and b.unavailable == []


def test_product_spec_detail_split_by_sentence_with_topics(world, make_retriever, route):
    ctx = make_context("这个是纯棉的吗", world, item_id="i_555")
    b = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))
    details = [e for e in b.evidences if ":detail:" in e.id]
    assert [(e.id, e.data["seg"]) for e in details] == [
        ("item:i_555:detail:0", 0),
        ("item:i_555:detail:1", 1),
        ("item:i_555:detail:2", 2),
    ]
    assert details[1].data["text"] == "现货付款后 24 小时内发货。"
    assert details[1].topic == PolicyKey.SHIP_TIME
    assert details[0].topic is None and details[2].topic is None
    assert all(e.label == "商品属性 · 详情页" and set(e.data) == {"text", "seg"} for e in details)


def test_product_spec_without_specific_attribute_gives_all(world, make_retriever, route):
    ctx = make_context("这款怎么样", world, item_id="i_555")
    ids = _ids(make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC)))
    for name in ("面料", "厚薄", "袖长", "版型"):
        assert f"item:i_555:attr:{name}" in ids


def test_asked_attribute_missing_on_card(world, make_retriever, route):
    ctx = make_context("这条裤子会透吗", world, item_id="i_777")
    b = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))
    assert b.missing == ["item:i_777:attr:厚薄"]
    # 问到的属性卡片上没有：给全部属性作为上下文
    assert {"item:i_777:attr:面料", "item:i_777:attr:弹力"} <= set(_ids(b))


def test_item_specific_faq_is_retrieved_for_its_item(world, make_retriever, route):
    ctx = make_context("这款会透吗", world, item_id="i_555")
    b = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))
    faqs = [e for e in b.evidences if e.source == EvidenceSource.FAQ]
    assert len(faqs) == 1
    faq = faqs[0]
    assert faq.score == 1.0 and faq.label == "FAQ · 这款会透吗"
    assert set(faq.data) == {"question", "answer", "faq_id"}
    assert faq.id == ev.faq(faq.data["faq_id"])
    assert "item:i_555:attr:厚薄" in _ids(b)
    # 其他商品看不到这条商品专属 FAQ
    other = make_context("这款会透吗", world, item_id="i_777")
    assert not any(
        e.source == EvidenceSource.FAQ
        for e in make_retriever().retrieve(other, route(Intent.PRODUCT_SPEC)).evidences
    )


# ================================================================ 尺码


def test_size_advice_with_merchant_advice(world, make_retriever, route):
    ctx = make_context("165 110斤穿什么码", world, item_id="i_555")
    b = make_retriever().retrieve(ctx, route(Intent.SIZE_ADVICE))
    _assert_well_formed(b)
    ids = _ids(b)
    assert ids[0] == "policy:size_tendency"
    assert {"item:i_555:size_chart", "item:i_555:size_advice"} <= set(ids)
    chart = b.get("item:i_555:size_chart")
    assert chart.data["conservative"] is False and len(chart.data["rows"]) == 4
    assert set(chart.data) == {"rows", "conservative"}
    assert CONSERVATIVE_NOTE not in chart.content
    advice = b.get("item:i_555:size_advice")
    assert "160–165 cm、体重 100–115 斤建议 M" in advice.data["text"]
    assert advice.label == "商品属性 · 尺码建议"
    assert b.missing == []


def test_size_advice_only_chart_is_conservative(world, make_retriever, route):
    ctx = make_context("165 110斤穿什么码", world, item_id="i_777")
    b = make_retriever().retrieve(ctx, route(Intent.SIZE_ADVICE))
    assert b.missing == ["item:i_777:size_advice"]
    assert "item:i_777:size_advice" not in _ids(b)
    chart = b.get("item:i_777:size_chart")
    assert chart.data["conservative"] is True
    assert chart.content.endswith(CONSERVATIVE_NOTE)


def test_size_advice_without_chart_or_advice(world, make_retriever, route):
    ctx = make_context("穿什么码", world, shop_id=SHOP_C, item_id="i_c1")
    b = make_retriever().retrieve(ctx, route(Intent.SIZE_ADVICE))
    assert "item:i_c1:size_chart" in b.missing and "item:i_c1:size_advice" in b.missing
    assert "policy:size_tendency" in b.missing  # 店铺 C 没填尺码偏向


# ================================================================ 政策、平台规则、冲突


def test_shipping_policy_conflict_48h_beats_24h(world, make_retriever, route):
    """§5.2.1：详情页"24 小时发货"这一句被店铺政策"48 小时"覆盖，其他句子保留。"""
    ctx = make_context("纯棉的吗？什么时候发货", world, item_id="i_555")
    b = make_retriever().retrieve(
        ctx, route(Intent.PRODUCT_SPEC, Intent.PRODUCT_SPEC, Intent.SHIPPING_POLICY)
    )
    _assert_well_formed(b)
    ids = _ids(b)
    assert "48 小时" in b.get("policy:ship_time").content
    assert "item:i_555:detail:1" not in ids
    assert "item:i_555:detail:1" in b.overridden_ids
    assert "item:i_555:detail:0" in ids
    assert b.get("item:i_555:detail:2").data["text"] == "面料亲肤透气。"
    assert not any("24 小时" in e.content for e in b.evidences)


def test_shipping_policy_policies_and_outdated_faq(world, repo, make_retriever, route):
    ctx = make_context("什么时候发货", world)
    b = make_retriever().retrieve(ctx, route(Intent.SHIPPING_POLICY))
    _assert_well_formed(b)
    assert _ids(b) == [
        "policy:ship_time",
        "policy:courier",
        "policy:courier_choice",
        "policy:remote_areas",
        "policy:presale",
    ]
    ship = b.get("policy:ship_time")
    assert ship.label == "店铺政策 · 发货时效" and ship.topic == "ship_time"
    assert ship.data == {"key": "ship_time", "value": world.policy_map(SHOP_A)["ship_time"].value}
    # FAQ 回答涉及发货时效，同主题以店铺政策为准（被覆盖）
    assert len(b.overridden_ids) == 1 and b.overridden_ids[0].startswith("faq:")

    # 方案 §5.2.1 的例子：政策改成 72 小时，FAQ 还写着 48 小时 → 以政策为准
    repo.upsert_policy(
        PolicyEntry(
            shop_id=SHOP_A, key="ship_time", value="付款后 72 小时内发货", updated_at=world.now
        )
    )
    b2 = make_retriever().retrieve(ctx, route(Intent.SHIPPING_POLICY))
    assert "72 小时" in b2.get("policy:ship_time").content
    assert not any("48 小时" in e.content for e in b2.evidences)


def test_return_policy_policies_platform_rule_and_tools_unavailable(world, make_retriever, route):
    ctx = make_context("能退吗？运费谁出", world, order_id="o_1004")
    b = make_retriever().retrieve(ctx, route(Intent.RETURN_POLICY))
    _assert_well_formed(b)
    assert _ids(b) == [
        "platform:taobao:apply_aftersale",
        "policy:seven_day_return",
        "policy:non_returnable",
        "policy:return_shipping",
        "policy:shipping_insurance",
    ]
    assert b.get("platform:taobao:apply_aftersale").source == EvidenceSource.PLATFORM
    assert b.unavailable == ["tool:order", "tool:return"]  # 没有注入工具证据
    assert b.missing == []


def test_invoice_gift(world, make_retriever, route):
    ctx = make_context("能开发票吗", world)
    b = make_retriever().retrieve(ctx, route(Intent.INVOICE_GIFT))
    assert _ids(b) == ["policy:invoice", "policy:gift", "policy:shipping_insurance"]
    assert b.missing == ["policy:warranty"]  # 店铺 A 没填保修
    assert "3 个工作日" in b.get("policy:invoice").content


def test_modify_order_platform_rule(world, make_retriever, route):
    ctx = make_context("能改个地址吗", world, order_id="o_1001")
    b = make_retriever().retrieve(ctx, route(Intent.MODIFY_ORDER))
    assert _ids(b) == ["platform:taobao:modify_order", "policy:modify_order"]
    assert b.unavailable == ["tool:order"]
    rule = b.get("platform:taobao:modify_order")
    assert rule.data["key"] == "modify_order" and "发货后" in rule.content


@pytest.mark.parametrize(("shop", "platform"), [(SHOP_C, "pdd"), (SHOP_D, "douyin")])
def test_platform_rule_follows_shop_platform(world, make_retriever, route, shop, platform):
    ctx = make_context("能退吗", world, shop_id=shop)
    b = make_retriever().retrieve(ctx, route(Intent.RETURN_POLICY))
    assert _ids(b)[0] == f"platform:{platform}:apply_aftersale"
    assert "policy:seven_day_return" in b.missing


def test_refund_progress_platform_rule(world, make_retriever, route):
    ctx = make_context("退款什么时候到", world)
    b = make_retriever().retrieve(ctx, route(Intent.REFUND_PROGRESS))
    assert _ids(b) == ["platform:taobao:refund_timing"]
    assert b.unavailable == ["tool:order", "tool:aftersale"]


def test_greeting_uses_service_hours(world, make_retriever, route):
    b = make_retriever().retrieve(make_context("在吗？", world), route(Intent.GREETING))
    assert _ids(b) == ["policy:service_hours"]


def test_other_intent_faq_only_keeps_direct_answer(world, make_retriever, route):
    """没有检索店铺政策时，FAQ 不会被覆盖，字面相同的问题分数为 1.0（可直答）。"""
    b = make_retriever().retrieve(make_context("能开发票吗？", world), route(Intent.OTHER))
    assert [e.source for e in b.evidences] == [EvidenceSource.FAQ]
    assert b.evidences[0].score == 1.0 and b.evidences[0].topic == "invoice"


def test_complaint_retrieves_nothing(flaky, world, make_retriever, route):
    flaky.fail_on = {"*"}
    b = make_retriever(repo_=flaky).retrieve(
        make_context("我要投诉", world), route(Intent.COMPLAINT)
    )
    assert b == EvidenceBundle()
    assert flaky.calls == []


# ================================================================ 洗涤与类目通用知识


def test_usage_item_care_overrides_category_tip(world, make_retriever, route):
    ctx = make_context("能机洗吗", world, item_id="i_777")
    b = make_retriever().retrieve(ctx, route(Intent.USAGE))
    care = b.get("item:i_777:care")
    assert care.data == {"care": ["反面机洗", "首次洗涤会轻微掉色，建议单独洗"]}
    assert care.label == "商品属性 · 洗涤保养"
    assert "category:apparel_wash" in b.overridden_ids
    assert not any(e.source == EvidenceSource.CATEGORY for e in b.evidences)
    assert b.missing == []


def test_usage_falls_back_to_category_knowledge(world, make_retriever, route):
    ctx = make_context("能机洗吗？会掉色吗", world, shop_id=SHOP_C, item_id="i_c1")
    b = make_retriever().retrieve(ctx, route(Intent.USAGE))
    _assert_well_formed(b)
    cat = b.get("category:apparel_wash")
    assert cat.source == EvidenceSource.CATEGORY and cat.priority == 6
    assert cat.label == "类目通用知识 · 服饰洗涤" and set(cat.data) == {"text"}
    assert b.evidences[-1].id == "category:apparel_wash"
    assert b.missing == ["item:i_c1:care"]


def test_category_knowledge_only_when_relevant(world, make_retriever, route):
    ctx = make_context("这个怎么搭配", world, shop_id=SHOP_C, item_id="i_c1")
    b = make_retriever().retrieve(ctx, route(Intent.USAGE))
    assert not any(e.source == EvidenceSource.CATEGORY for e in b.evidences)
    assert b.missing == ["item:i_c1:care", "item:i_c1:usage"]


# ================================================================ 多问题、澄清


def test_split_multi_unions_needs(world, make_retriever, route):
    ctx = make_context("纯棉吗？能退吗", world, item_id="i_555")
    b = make_retriever().retrieve(
        ctx, route(Intent.PRODUCT_SPEC, Intent.PRODUCT_SPEC, Intent.RETURN_POLICY)
    )
    _assert_well_formed(b)
    ids = set(_ids(b))
    assert {"item:i_555:attr:面料", "platform:taobao:apply_aftersale"} <= ids
    assert {"policy:seven_day_return", "policy:return_shipping"} <= ids
    assert b.unavailable == ["tool:order", "tool:return"]


def test_ambiguous_item_asks_which_one(world, make_retriever, route):
    ctx = make_context("这个是纯棉的吗", world, candidate_item_ids=["i_555", "i_777"])
    b = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))
    assert b.clarify_question == CLARIFY_ITEM_QUESTION == "请问您问的是哪一款呢？"
    assert not any(e.id.startswith("item:") for e in b.evidences)
    assert ev.MISSING_ITEM not in b.missing


@pytest.mark.parametrize("candidates", [[], ["i_555"], ["i_555", "i_555"]])
def test_no_item_without_ambiguity_is_missing(world, make_retriever, route, candidates):
    ctx = make_context("这个是纯棉的吗", world, candidate_item_ids=candidates)
    ctx = ctx.model_copy(update={"item": None, "candidate_item_ids": candidates})
    b = make_retriever().retrieve(ctx, route(Intent.SIZE_ADVICE))
    assert b.clarify_question is None
    assert b.missing == [ev.MISSING_ITEM]
    assert _ids(b) == ["policy:size_tendency"]  # 政策照常给


# ================================================================ 工具证据


def test_tool_provider_bundle_is_merged(world, make_retriever, route):
    tool_bundle = EvidenceBundle(
        evidences=[_order_evidence(world, "o_1004")],
        missing=["order:?"],
        unavailable=["tool:logistics"],
        overridden_ids=["tool:x"],
        clarify_question="请问是哪一笔订单呢？",
        alerts=["物流停滞超过 48 小时，建议向快递催查"],
    )
    provider = FakeToolProvider(tool_bundle)
    ctx = make_context("能退吗", world, order_id="o_1004")
    b = make_retriever(tool_provider=provider).retrieve(ctx, route(Intent.RETURN_POLICY))
    _assert_well_formed(b)
    assert provider.calls == [
        (SHOP_A, Intent.RETURN_POLICY, frozenset({Need.ORDER, Need.RETURN_ELIGIBILITY}))
    ]
    assert _ids(b)[0] == "tool:order:o_1004"  # 实时数据优先级最高，排最前
    assert "platform:taobao:apply_aftersale" in _ids(b)
    assert b.missing == ["order:?"]
    assert b.unavailable == ["tool:logistics"]
    assert b.overridden_ids == ["tool:x"]
    assert b.clarify_question == "请问是哪一笔订单呢？"
    assert b.alerts == ["物流停滞超过 48 小时，建议向快递催查"]


def test_realtime_evidence_overrides_policy_on_same_topic(world, make_retriever, route):
    price = Evidence(
        id=ev.tool_price("i_888"),
        source=EvidenceSource.REALTIME,
        label=ev.tool_label("price"),
        content="定制款不参加店铺满减",
        topic=PolicyKey.PROMOTIONS,
        fetched_at=world.now,
    )
    provider = FakeToolProvider(EvidenceBundle(evidences=[price]))
    ctx = make_context("有优惠吗", world, item_id="i_888")
    b = make_retriever(tool_provider=provider).retrieve(ctx, route(Intent.PRICE_PROMO))
    assert _ids(b) == ["tool:price:i_888", "policy:promise_boundary"]
    assert b.overridden_ids == ["policy:promotions"]
    assert provider.calls[0][2] == frozenset({Need.PRICE})


def test_item_clarify_takes_precedence_over_tool_clarify(world, make_retriever, route):
    provider = FakeToolProvider(EvidenceBundle(clarify_question="请问是哪个规格呢？"))
    ctx = make_context("还有货吗", world, candidate_item_ids=["i_555", "i_777"])
    b = make_retriever(tool_provider=provider).retrieve(ctx, route(Intent.STOCK))
    assert b.clarify_question == CLARIFY_ITEM_QUESTION
    assert provider.calls[0][1:] == (Intent.STOCK, frozenset({Need.STOCK}))


def test_tool_provider_failure_marks_unavailable(world, make_retriever, route):
    provider = FakeToolProvider(fail=ToolUnavailable("限流（模拟）"))
    ctx = make_context("快递到哪了", world, order_id="o_1002")
    b = make_retriever(tool_provider=provider).retrieve(ctx, route(Intent.NOT_RECEIVED))
    assert b.unavailable == ["tool:order", "tool:logistics", "tool:aftersale"]
    assert _ids(b) == ["platform:taobao:apply_aftersale"]


def test_split_multi_calls_provider_per_intent_and_dedupes(world, make_retriever, route):
    provider = FakeToolProvider(EvidenceBundle(evidences=[_order_evidence(world)]))
    ctx = make_context("到哪了？能改地址吗", world, order_id="o_1001")
    b = make_retriever(tool_provider=provider).retrieve(
        ctx, route(Intent.LOGISTICS_TRACK, Intent.LOGISTICS_TRACK, Intent.MODIFY_ORDER)
    )
    assert [c[1] for c in provider.calls] == [Intent.LOGISTICS_TRACK, Intent.MODIFY_ORDER]
    assert provider.calls[1][2] == frozenset({Need.ORDER})
    assert _ids(b).count("tool:order:o_1001") == 1


def test_tool_only_intent_does_not_touch_knowledge(flaky, world, make_retriever, route):
    flaky.fail_on = {"*"}  # 知识库挂了也不影响只需要工具的问题
    ctx = make_context("我的快递到哪了", world, order_id="o_1002")
    b = make_retriever(repo_=flaky).retrieve(ctx, route(Intent.LOGISTICS_TRACK))
    assert b.unavailable == ["tool:order", "tool:logistics"]
    assert flaky.calls == []


# ================================================================ 长度预算


def test_budget_trims_lowest_priority_first(world, make_retriever, route):
    ctx = make_context("能机洗吗？会掉色吗", world, shop_id=SHOP_C, item_id="i_c1")
    full = make_retriever().retrieve(ctx, route(Intent.USAGE))
    total = sum(len(e.content) for e in full.evidences)
    cat = full.get("category:apparel_wash")
    budget = total - len(cat.content)
    b = make_retriever(config=GenerationConfig(max_evidence_chars=budget)).retrieve(
        ctx, route(Intent.USAGE)
    )
    assert _ids(b) == [i for i in _ids(full) if i != "category:apparel_wash"]
    assert sum(len(e.content) for e in b.evidences) <= budget


def test_budget_keeps_policy_and_platform_last(world, make_retriever, route):
    ctx = make_context("能退吗", world)
    full = make_retriever().retrieve(ctx, route(Intent.RETURN_POLICY))
    rule = full.get("platform:taobao:apply_aftersale")
    b = make_retriever(config=GenerationConfig(max_evidence_chars=len(rule.content) + 30)).retrieve(
        ctx, route(Intent.RETURN_POLICY)
    )
    assert _ids(b)[0] == "platform:taobao:apply_aftersale"
    assert sum(len(e.content) for e in b.evidences) <= len(rule.content) + 30
    assert all(e.source in {EvidenceSource.PLATFORM, EvidenceSource.POLICY} for e in b.evidences)


def test_budget_drops_irrelevant_detail_sentences_before_structured_fields(
    world, make_retriever, route
):
    ctx = make_context("面料亲肤吗", world, item_id="i_555")
    full = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))
    structured = [e for e in full.evidences if ":detail:" not in e.id]
    relevant = full.get("item:i_555:detail:2")  # "面料亲肤透气。"
    budget = sum(len(e.content) for e in structured) + len(relevant.content)
    b = make_retriever(config=GenerationConfig(max_evidence_chars=budget)).retrieve(
        ctx, route(Intent.PRODUCT_SPEC)
    )
    ids = _ids(b)
    assert all(e.id in ids for e in structured)
    assert "item:i_555:detail:2" in ids
    assert "item:i_555:detail:0" not in ids and "item:i_555:detail:1" not in ids


# ================================================================ 故障与隔离


@pytest.mark.parametrize(
    ("method", "intent", "text"),
    [
        ("get_policies", Intent.SHIPPING_POLICY, "什么时候发货"),
        ("list_knowledge", Intent.OTHER, "能开发票吗"),
        ("search_knowledge", Intent.OTHER, "你们开发票吗"),
        ("set_knowledge_embedding", Intent.OTHER, "你们开发票吗"),
    ],
)
def test_repository_failure_raises_knowledge_unavailable(
    flaky, world, make_retriever, route, method, intent, text
):
    flaky.fail_on = {method}
    with pytest.raises(KnowledgeUnavailable):
        make_retriever(repo_=flaky).retrieve(make_context(text, world), route(intent))


def test_embedder_failure_degrades_to_lexical_faq(spy, world, make_retriever, route):
    spy.fail = True
    b = make_retriever().retrieve(make_context("能开发票吗", world), route(Intent.OTHER))
    assert [e.score for e in b.evidences] == [1.0]
    b2 = make_retriever().retrieve(make_context("你们开发票吗", world), route(Intent.OTHER))
    assert b2.evidences == []


def test_shop_b_gets_only_shop_b_knowledge(world, make_retriever, route):
    """店铺 B 与 A 同平台、相同 item_id、相同 FAQ 问题（§9.2 串店测试）。"""
    ctx = make_context("什么面料的？发什么快递", world, shop_id=SHOP_B, item_id="i_555")
    b = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))
    assert b.get("item:i_555:attr:面料").data["value"] == "涤纶 100%"
    assert b.get("item:i_555:title").data["title"] == "男士休闲裤"
    text = "\n".join(e.content for e in b.evidences)
    assert "棉 95%" not in text and "中通" not in text and "碎花" not in text
    faq = make_retriever().retrieve(
        make_context("发什么快递", world, shop_id=SHOP_B), route(Intent.OTHER)
    )
    assert [e.data["answer"] for e in faq.evidences] == ["您好，默认发圆通快递。"]
    ship = make_retriever().retrieve(
        make_context("什么时候发货", world, shop_id=SHOP_B), route(Intent.SHIPPING_POLICY)
    )
    assert ship.get("policy:ship_time").data["value"] == "付款后 72 小时内发货"
    assert "policy:remote_areas" in ship.missing  # B 没填，不会拿 A 的补


def test_item_from_another_shop_is_refused(world, make_retriever, route):
    ctx = make_context("纯棉吗", world, shop_id=SHOP_B)
    ctx = ctx.model_copy(update={"item": world.item(SHOP_A, "i_555")})
    with pytest.raises(TenantIsolationError):
        make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))


# ================================================================ 个人信息最小化


def test_buyer_pii_never_reaches_embedder_or_evidence(spy, world, make_retriever, route):
    text = "我手机13812345678，住幸福路 18 号 3 栋 201，这款会透吗"
    ctx = make_context(text, world, item_id="i_555")
    b = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))
    sent = "\n".join(spy.texts)
    assert "13812345678" not in sent and "幸福路" not in sent
    contents = "\n".join(e.content for e in b.evidences)
    assert "13812345678" not in contents and "幸福路" not in contents
    assert "item:i_555:attr:厚薄" in _ids(b)


def test_merchant_text_with_contact_is_masked_in_content(world, repo, make_retriever, route):
    repo.upsert_policy(
        PolicyEntry(
            shop_id=SHOP_A,
            key="service_hours",
            value="客服在线 9:00–23:00，急事打 13900001111",
            updated_at=world.now,
        )
    )
    b = make_retriever().retrieve(make_context("在吗", world), route(Intent.GREETING))
    e = b.get("policy:service_hours")
    assert "13900001111" not in e.content and "[手机号]" in e.content
    assert "9:00" in e.content


def test_faq_answer_topics_and_scores(world, repo, make_retriever, route):
    """FAQ 回答涉及多个主题：任一主题被政策占据就整条移除。"""
    repo.add_knowledge(
        KnowledgeItem(
            shop_id=SHOP_A,
            question="新疆能发吗",
            answer="亲，新疆可以发，偏远地区需补运费 15 元。付款后 48 小时内发货哦~",
            source=KnowledgeSource.MERCHANT,
            status=KnowledgeStatus.APPROVED,
            created_at=world.now - timedelta(days=1),
        )
    )
    b = make_retriever().retrieve(make_context("新疆能发吗", world), route(Intent.OTHER))
    faq = b.evidences[0]
    assert faq.topic == "ship_time"  # 按 PolicyKey 顺序取第一个主题
    assert faq.score == 1.0
    b2 = make_retriever().retrieve(make_context("新疆能发吗", world), route(Intent.SHIPPING_POLICY))
    assert faq.id in b2.overridden_ids


class LeakyRepo:
    """有 bug 的仓储实现：不管查哪家店，都返回店铺 B 的数据（§9.2 防御性检查）。"""

    def __init__(self, inner, leak: set[str]) -> None:
        self._inner = inner
        self._leak = leak

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def list_knowledge(self, shop_id, **kw):
        return self._inner.list_knowledge(SHOP_B if "list" in self._leak else shop_id, **kw)

    def search_knowledge(self, shop_id, embedding, **kw):
        target = SHOP_B if "search" in self._leak else shop_id
        return self._inner.search_knowledge(target, embedding, **kw)

    def get_policies(self, shop_id):
        return self._inner.get_policies(SHOP_B if "policy" in self._leak else shop_id)


@pytest.mark.parametrize(
    ("leak", "intent"),
    [
        ({"list"}, Intent.OTHER),
        ({"search"}, Intent.OTHER),
        ({"policy"}, Intent.SHIPPING_POLICY),
    ],
)
def test_cross_shop_rows_from_a_buggy_repository_are_refused(
    repo, world, make_retriever, route, leak, intent
):
    FaqSearcher(repo, HashingEmbedder()).index(SHOP_B)  # B 的 FAQ 有向量，才会被错误地搜出来
    leaky = LeakyRepo(repo, leak)
    with pytest.raises(TenantIsolationError):
        make_retriever(repo_=leaky, embedder=HashingEmbedder()).retrieve(
            make_context("你们发什么快递", world), route(intent)
        )


def test_sku_without_props_does_not_expose_internal_ids(world, make_retriever, route):
    card = world.item(SHOP_A, "i_555").model_copy(update={"skus": [Sku(sku_id="k_internal_9")]})
    ctx = make_context("这款怎么样", world, item_id="i_555", sku_id="k_internal_9")
    ctx = ctx.model_copy(update={"item": card})
    sku = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC)).get("item:i_555:sku")
    assert "k_internal_9" not in sku.content and "默认规格" in sku.content
    assert sku.data["skus"][0]["sku_id"] == "k_internal_9"
