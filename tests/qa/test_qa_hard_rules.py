"""质检第一道：硬规则（方案 §5.5、§2.1 话术禁区、§7.2、§11.4、§11.5）。"""

from __future__ import annotations

import pytest

from rodit.core import evidence as E
from rodit.core.config import QAGateConfig
from rodit.core.enums import DraftKind, EvidenceSource, PolicyKey, Role
from rodit.core.models import Draft, Evidence, EvidenceBundle
from rodit.qa import blocking_hits, check_hard_rules
from rodit.testing.fakes import make_context
from rodit.testing.fixtures import SHOP_B

CFG = QAGateConfig()
SCRIPT_11_4 = (
    "亲，平台规定不能通过站外联系和交易哦，您直接在店铺下单就好，我帮您看看有没有可以用的优惠券。"
)
REGENERATED_11_5 = (
    "亲，包裹目前在运输中，截至 10:30 已到达杭州转运中心，我们会持续关注，有更新第一时间告诉您。"
)


def run(text, ctx, evidence=None, ids=None, kind=DraftKind.LLM, config=CFG):
    bundle = evidence or EvidenceBundle()
    draft = Draft(text=text, kind=kind, evidence_ids=ids if ids is not None else bundle.ids())
    return check_hard_rules(text, ctx=ctx, draft=draft, evidence=bundle, config=config)


def rules(hits) -> list[str]:
    return [h.rule for h in hits]


def test_off_platform_standard_script_passes(world):
    """§11.4：标准话术里"站外"是否定用法，不拦截。"""
    ctx = make_context("加个微信吧，转账给你便宜点", world)
    text, hits = run(SCRIPT_11_4, ctx, kind=DraftKind.TEMPLATE)
    assert text == SCRIPT_11_4 and hits == []


def test_11_5_promise_is_blocked(world, ev):
    """§11.5：物流运输中、没有预计送达时间，"明天一定能到"拦截。"""
    ctx = make_context("明天能到吗？", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    _, hits = run("亲，明天一定能到哦！", ctx, bundle)
    assert set(rules(hits)) == {"time_promise", "absolute_term"}
    details = [h.detail for h in hits]
    assert "包含依据中没有的时效承诺：明天" in details
    assert "包含绝对化用语：一定" in details
    assert blocking_hits(hits) == hits


def test_11_5_regenerated_answer_passes(world, ev):
    ctx = make_context("明天能到吗？", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    text, hits = run(REGENERATED_11_5, ctx, bundle)
    assert hits == [] and text == REGENERATED_11_5


def test_time_promise_allowed_when_evidence_says_so(world, ev):
    ctx = make_context("今天能发吗", world)
    policy = ev.policy(PolicyKey.SHIP_TIME).model_copy(
        update={"content": "当天 16:00 前付款当天发货，一般 3 天内送达"}
    )
    bundle = EvidenceBundle(evidences=[policy])
    assert run("亲，16:00 前付款当天发货，一般 3 天内送达哦", ctx, bundle)[1] == []
    _, hits = run("亲，今天就发，2 天就能到", ctx, bundle)
    assert [h.span for h in hits if h.rule == "time_promise"] == ["今天", "2 天就能到"]


@pytest.mark.parametrize(
    "text",
    ["马上就到啦", "后天送达", "今晚能到", "次日达哦", "两天内送到", "48 小时内送达"],
)
def test_time_promise_variants(world, ev, text):
    ctx = make_context("什么时候到", world)
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.SHIP_TIME)])
    _, hits = run(f"亲，{text}", ctx, bundle)
    assert "time_promise" in rules(hits), text


def test_restock_date_is_not_a_duration_promise(world):
    """ "10 月 8 日到货"是日期不是时长；日期数字在证据里就放行。"""
    ctx = make_context("白色 S 码什么时候有货", world, item_id="i_555", sku_id="k_3")
    stock = world.stocks[2]
    stock_ev = Evidence(
        id=E.tool_stock("i_555", "k_3"),
        source=EvidenceSource.REALTIME,
        label=E.tool_label("stock"),
        content="白色 S 码暂时缺货，预计 10 月 8 日补货",
        data={"stock": stock.model_dump(mode="json")},
        fetched_at=stock.fetched_at,
    )
    bundle = EvidenceBundle(evidences=[stock_ev])
    assert run("亲，白色 S 码暂时缺货，预计 10 月 8 日到货哦", ctx, bundle)[1] == []
    _, hits = run("亲，预计 10 月 9 日到货哦", ctx, bundle)
    assert [h.rule for h in hits] == ["number_mismatch"] and hits[0].span == "10 月 9 日"


def test_ship_time_is_not_a_delivery_promise(world, ev):
    ctx = make_context("什么时候发货", world)
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.SHIP_TIME)])
    assert run("亲，现货付款后 48 小时内发货哦", ctx, bundle)[1] == []


def test_number_mismatch_hours(world, ev):
    ctx = make_context("多久发货", world)
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.SHIP_TIME)])
    _, hits = run("亲，现货付款后 72 小时内发货哦", ctx, bundle)
    assert rules(hits) == ["number_mismatch"]
    assert hits[0].detail == "包含依据中找不到的数字：72 小时"
    # 全角数字、有没有空格都一样
    assert run("亲，现货付款后４８小时内发货", ctx, bundle)[1] == []


def test_number_mismatch_price(world, ev):
    ctx = make_context("多少钱", world, item_id="i_555")
    price = world.prices[0]
    price_ev = Evidence(
        id=E.tool_price("i_555"),
        source=EvidenceSource.REALTIME,
        label=E.tool_label("price"),
        content="售价 259.00 元；店铺满 199 减 20",
        data={"price": price.model_dump(mode="json")},
        fetched_at=price.fetched_at,
    )
    bundle = EvidenceBundle(evidences=[price_ev])
    assert run("亲，现在 259 元，满 199 减 20 哦", ctx, bundle)[1] == []
    _, hits = run("亲，现在只要 239 元哦", ctx, bundle)
    assert [h.span for h in hits] == ["239 元"]


def test_buyer_numbers_are_not_fabrication(world, ev):
    """买家自己说的身高体重，回答里复述不算编造。"""
    ctx = make_context(
        "165 110斤穿什么码", world, item_id="i_555", history=[(Role.BUYER, "我平时穿 M")]
    )
    bundle = EvidenceBundle(evidences=[ev.size_advice()])
    assert run("亲，165 cm、110 斤建议选 M 码哦", ctx, bundle)[1] == []
    _, hits = run("亲，180 cm 的话建议 XL 码", ctx, bundle)
    assert [h.span for h in hits] == ["180 cm"]


def test_times_must_appear_in_evidence(world, ev):
    ctx = make_context("到哪了", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    _, hits = run("亲，截至 10:45 已到达杭州转运中心", ctx, bundle)
    assert [h.span for h in hits] == ["10:45"]


def test_identifiers_are_not_numbers(world, ev):
    ctx = make_context("到哪了", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    text = "亲，订单 o_1002 的运单号 ZT7300000001 截至 10:30 已到达杭州转运中心"
    assert run(text, ctx, bundle)[1] == []


def test_forbidden_claim(world, ev):
    ctx = make_context("防晒吗", world, item_id="i_555")
    bundle = EvidenceBundle(evidences=[ev.item_attr("面料")])
    _, hits = run("亲，这款面料透气又防晒哦", ctx, bundle)
    assert rules(hits) == ["forbidden_claim"]
    assert hits[0].detail == "包含商家禁止的卖点：防晒"
    _, hits = run("亲，这款不仅透气还抗菌", ctx, bundle)
    assert [h.span for h in hits] == ["抗菌"]
    # 否定说法不是卖点
    assert run("亲，这款没有防晒功能哦", ctx, bundle)[1] == []
    assert run("亲，这款不具备抗菌效果", ctx, bundle)[1] == []
    # 没有当前商品时只看证据里带的卖点禁区
    ctx2 = make_context("防晒吗", world)
    claims = ev.item_attr("面料").model_copy(
        update={"data": {"name": "面料", "value": "棉", "forbidden_claims": ["抗皱"]}}
    )
    _, hits = run("亲，这款很抗皱", ctx2, EvidenceBundle(evidences=[claims]))
    assert rules(hits) == ["forbidden_claim"]


def test_banned_word_and_tenant_config(world):
    ctx = make_context("是正品吗", world)
    _, hits = run("亲，我们家假一赔十哦", ctx)
    assert rules(hits) == ["banned_word"]
    assert hits[0].detail == "包含商家禁用词：假一赔十"
    # 店铺 B 没有配置这个禁用词（配置按店铺隔离）
    ctx_b = make_context("是正品吗", world, shop_id=SHOP_B)
    assert run("您好，我们家假一赔十", ctx_b)[1] == []


@pytest.mark.parametrize(
    ("text", "span"),
    [
        ("亲，加我vx号abc123聊", "vx"),
        ("亲，加个v吧", "加个v"),
        ("亲，留个手机号给我", "手机号"),
        ("亲，点这个链接 https://shop.example.cn/p", "https://"),
        ("亲，可以扫码付款", "扫码"),
        ("亲，打我电话 13812345678", "[手机号]"),
        ("亲，支付宝转账更便宜", "支付宝"),
    ],
)
def test_off_platform_variants(world, text, span):
    ctx = make_context("加微信", world)
    _, hits = run(text, ctx)
    off = [h for h in hits if h.rule == "off_platform"]
    assert off, text
    assert span in [h.span for h in off]
    # 质检原因里不带出手机号原文
    assert "13812345678" not in " ".join(h.detail for h in hits)


def test_off_platform_link_hidden_in_markdown(world):
    ctx = make_context("有链接吗", world)
    text, hits = run("亲，[点这里领券](https://coupon.example.cn/x)", ctx)
    assert text == "亲，点这里领券"
    assert "off_platform" in rules(hits) and "format" in rules(hits)


def test_markdown_and_length_are_non_blocking(world, ev):
    ctx = make_context("纯棉吗", world, item_id="i_555", config_update={"max_answer_chars": 30})
    bundle = EvidenceBundle(evidences=[ev.item_attr("面料")])
    raw = (
        "**亲**，这款面料是：\n- 棉 95%\n- 氨纶 5%\n"
        "穿着亲肤舒适。洗涤建议手洗，不要暴晒，晾干后再收起来哦。"
    )
    text, hits = run(raw, ctx, bundle)
    assert rules(hits) == ["format", "length"]
    assert blocking_hits(hits) == []
    assert "**" not in text and "- " not in text
    assert len(text) <= 30
    assert text.endswith("。")


def test_unknown_evidence(world, ev):
    ctx = make_context("纯棉吗", world, item_id="i_555")
    bundle = EvidenceBundle(
        evidences=[ev.item_attr("面料")], overridden_ids=["item:i_555:detail:1"]
    )
    _, hits = run(
        "亲，是棉 95% 的",
        ctx,
        bundle,
        ids=["item:i_555:attr:面料", "faq:404", "item:i_555:detail:1", "faq:404"],
    )
    assert [h.detail for h in hits] == [
        "引用了依据中不存在的来源：faq:404",
        "引用了已被更高优先级依据覆盖的来源：item:i_555:detail:1",
    ]


def test_absolute_terms_and_exceptions(world, ev):
    ctx = make_context("到哪了", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    assert run("亲，最新状态是已到达杭州转运中心，有更新第一时间告诉您", ctx, bundle)[1] == []
    _, hits = run("亲，这是全网最好的", ctx, bundle)
    assert [h.span for h in hits] == ["最"]
    custom = QAGateConfig(absolute_terms=["顶配"], absolute_term_exceptions=[])
    _, hits = run("亲，这是顶配款，最好看", ctx, bundle, config=custom)
    assert [h.span for h in hits] == ["顶配"]
