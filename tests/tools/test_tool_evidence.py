"""T3 工具证据：id / label / data 契约、找订单与澄清、降级、串店、个人信息（方案 §7、§12.2）。"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from rodit.aftersale.advisor import DefaultAfterSaleAdvisor
from rodit.core import evidence as ev
from rodit.core.enums import EvidenceSource, Intent, LogisticsStatus, PolicyKey
from rodit.core.errors import TenantIsolationError, ToolUnavailable
from rodit.core.evidence import Need
from rodit.core.models import (
    AfterSaleInfo,
    Evidence,
    EvidenceBundle,
    LogisticsAnalysis,
    LogisticsTrace,
    OrderInfo,
    PriceInfo,
    RefundProgress,
    ReturnEligibility,
    StockInfo,
)
from rodit.core.pii import contains_pii
from rodit.core.ports import ToolEvidenceProvider
from rodit.inmemory import InMemoryCache
from rodit.knowledge.conflict import Candidate, resolve_conflicts
from rodit.testing.fakes import FakeClock, InMemoryToolGateway, make_context
from rodit.testing.fixtures import (
    BUYER,
    BUYER_R3,
    BUYER_R5,
    BUYER_R8,
    BUYER_REFUND,
    SHOP_A,
    SHOP_B,
    build_demo_world,
)
from rodit.tools.evidence import (
    CLARIFY_ORDER_QUESTION,
    DefaultToolEvidenceProvider,
    resolve_order,
)
from rodit.tools.gateway import CachingToolGateway
from rodit.tools.logistics import DefaultLogisticsAdvisor

ALL_TOOL_NEEDS = ev.TOOL_NEEDS


class Env:
    def __init__(self, *, cached: bool = False) -> None:
        self.w = build_demo_world()
        self.clock = FakeClock(self.w.now)
        self.inner = InMemoryToolGateway(self.w)
        self.gw = (
            CachingToolGateway(self.inner, InMemoryCache(self.clock), self.clock, ttl_s=300)
            if cached
            else self.inner
        )
        self.provider = DefaultToolEvidenceProvider(
            self.gw,
            self.w.policy_map,
            DefaultLogisticsAdvisor(),
            DefaultAfterSaleAdvisor(),
            self.clock,
        )

    def collect(self, intent: Intent, needs=None, **ctx_kw) -> EvidenceBundle:
        ctx = make_context("问题", self.w, **ctx_kw)
        return self.provider.collect(ctx, intent, frozenset(needs or ev.needs_for(intent)))


@pytest.fixture
def env():
    return Env()


def test_is_tool_evidence_provider(env):
    assert isinstance(env.provider, ToolEvidenceProvider)


def test_only_tool_needs_are_handled(env):
    bundle = env.collect(Intent.SHIPPING_POLICY, order_id="o_1002")
    assert bundle == EvidenceBundle()
    assert env.inner.calls == []


# ---------------------------------------------------------------- id / label / data 契约


def test_logistics_track_evidence_contract(env):
    bundle = env.collect(Intent.LOGISTICS_TRACK, order_id="o_1002")
    assert bundle.ids() == [ev.tool_order("o_1002"), ev.tool_logistics("o_1002")]
    order_ev, log_ev = bundle.evidences
    assert all(e.source == EvidenceSource.REALTIME for e in bundle.evidences)
    assert order_ev.label == "实时数据 · 订单"
    assert log_ev.label == "实时数据 · 物流轨迹"
    assert OrderInfo.model_validate(order_ev.data["order"]) == env.w.order(SHOP_A, "o_1002")
    trace = LogisticsTrace.model_validate(log_ev.data["trace"])
    analysis = LogisticsAnalysis.model_validate(log_ev.data["analysis"])
    assert trace == env.w.trace(SHOP_A, "o_1002")
    assert analysis.status == LogisticsStatus.IN_TRANSIT
    assert set(order_ev.data) == {"order"} and set(log_ev.data) == {"trace", "analysis"}
    assert order_ev.fetched_at == env.w.now
    assert log_ev.fetched_at == trace.fetched_at
    assert "截至 09-28 10:30 已到达杭州转运中心" in log_ev.content
    assert "不要承诺到货时间" in log_ev.content
    assert "o_1002" in order_ev.content and "已发货" in order_ev.content
    assert bundle.alerts == [] and bundle.missing == [] and bundle.unavailable == []
    # data 是纯 JSON
    json.dumps([e.data for e in bundle.evidences])


def test_stalled_logistics_alert_goes_to_bundle(env):
    bundle = env.collect(Intent.LOGISTICS_TRACK, order_id="o_1006")
    assert bundle.alerts == ["物流超过 48 小时未更新，建议向快递催查"]
    assert "可以告诉买家：已帮您催查" in bundle.get(ev.tool_logistics("o_1006")).content


def test_exception_and_returning_alerts(env):
    assert env.collect(Intent.LOGISTICS_TRACK, order_id="o_1008").alerts
    returning = env.collect(Intent.LOGISTICS_TRACK, order_id="o_1009").alerts
    assert returning and "售后" in returning[0]


def test_not_received_with_station_and_aftersale(env):
    bundle = env.collect(Intent.NOT_RECEIVED, order_id="o_1003")
    assert bundle.ids() == [
        ev.tool_order("o_1003"),
        ev.tool_logistics("o_1003"),
        ev.tool_aftersale("as_2003"),
    ]
    log_ev = bundle.get(ev.tool_logistics("o_1003"))
    assert "取件指引：包裹已放在菜鸟驿站（幸福里店），请凭取件码取件" in log_ev.content
    as_ev = bundle.get(ev.tool_aftersale("as_2003"))
    assert as_ev.label == "实时数据 · 售后单"
    info = AfterSaleInfo.model_validate(as_ev.data["aftersale"])
    progress = RefundProgress.model_validate(as_ev.data["progress"])
    assert info == env.w.aftersale(SHOP_A, "as_2003")
    assert progress.aftersale_id == "as_2003" and progress.as_of == env.w.now
    # 买家的售后说明不进证据文字（§5.3）
    assert "根本没收到" not in as_ev.content
    assert bundle.alerts == []  # 驿站签收只给取件指引，不提醒客服


def test_refund_progress_by_order(env):
    bundle = env.collect(Intent.REFUND_PROGRESS, order_id="o_1012", buyer_ref=BUYER_REFUND)
    assert bundle.ids() == [
        ev.tool_order("o_1012"),
        ev.tool_aftersale("as_2006"),
        ev.tool_aftersale("as_2008"),
    ]
    refunding = bundle.get(ev.tool_aftersale("as_2008"))
    assert "退款中" in refunding.content and "以平台规则为准" in refunding.content


def test_return_eligibility_evidence(env):
    bundle = env.collect(Intent.RETURN_POLICY, order_id="o_1004", buyer_ref=BUYER_R3)
    assert bundle.ids() == [ev.tool_order("o_1004"), ev.tool_return("o_1004")]
    ret = bundle.get(ev.tool_return("o_1004"))
    assert ret.label == "实时数据 · 退换资格"
    elig = ReturnEligibility.model_validate(ret.data["eligibility"])
    assert elig.eligible is True and elig.days_since_signed == 5
    assert "可以无理由退货" in ret.content
    assert "支持七天无理由退换货" in ret.content  # 引用店铺政策
    custom = env.collect(Intent.RETURN_POLICY, order_id="o_1007").get(ev.tool_return("o_1007"))
    assert ReturnEligibility.model_validate(custom.data["eligibility"]).eligible is False
    assert "定制款" in custom.content


def test_stock_evidence_with_sku(env):
    bundle = env.collect(Intent.STOCK, item_id="i_555", sku_id="k_3")
    assert bundle.ids() == [ev.tool_stock("i_555", "k_3")]
    stock_ev = bundle.evidences[0]
    assert stock_ev.label == "实时数据 · 实时库存"
    assert stock_ev.topic == ev.topic_for_stock("i_555", "k_3")
    assert StockInfo.model_validate(stock_ev.data["stock"]).in_stock is False
    assert "颜色：白色" in stock_ev.content and "尺码：S" in stock_ev.content
    assert "暂时缺货，预计 10 月 8 日补货" in stock_ev.content
    assert stock_ev.fetched_at == env.w.now


def test_stock_item_level_summary(env):
    stock_ev = env.collect(Intent.STOCK, item_id="i_555").evidences[0]
    assert stock_ev.id == ev.tool_stock("i_555")
    assert stock_ev.topic == ev.topic_for_stock("i_555")
    # 白色 S 缺货：不给整款合计"有货，库存 35 件"，逐个规格写清
    assert "有货，库存 35 件" not in stock_ev.content
    assert "部分规格有货" in stock_ev.content
    assert "暂时缺货：白色 / S（预计 10 月 8 日补货）" in stock_ev.content
    assert "黑色 / M 库存 12 件" in stock_ev.content


def test_price_evidence(env):
    bundle = env.collect(Intent.PRICE_PROMO, item_id="i_555")
    assert bundle.ids() == [ev.tool_price("i_555")]
    price_ev = bundle.evidences[0]
    assert price_ev.label == "实时数据 · 价格与活动"
    assert price_ev.topic == PolicyKey.PROMOTIONS.value  # 和店铺满减政策比较（§5.2.1）
    assert PriceInfo.model_validate(price_ev.data["price"]).promotions == ["店铺满 199 减 20"]
    assert "售价 259.00 元" in price_ev.content
    assert "当前没有活动" in env.collect(Intent.PRICE_PROMO, item_id="i_888").evidences[0].content


def test_stock_or_price_without_item_is_missing_item(env):
    bundle = env.collect(Intent.STOCK, needs={Need.STOCK, Need.PRICE})
    assert bundle.missing == [ev.MISSING_ITEM]
    assert bundle.evidences == []


# ---------------------------------------------------------------- 找订单


def _orders(w, buyer):
    return sorted(
        (o for o in w.orders if o.shop_id == SHOP_A and o.buyer_ref == buyer),
        key=lambda o: (o.created_at, o.order_id),
        reverse=True,
    )


def test_resolve_order_rules():
    w = build_demo_world()
    orders = _orders(w, BUYER)
    # 已发货 / 已签收的有好几单，最近两单相隔不到 24 小时 → 澄清
    res = resolve_order(Intent.LOGISTICS_TRACK, orders)
    assert res.needs_clarify and res.candidates == ["o_1010", "o_1002"]
    # 不设时间窗口 → 取最近一单
    assert resolve_order(Intent.LOGISTICS_TRACK, orders, window=timedelta(0)).order_id == "o_1010"
    assert resolve_order(Intent.URGE_SHIP, orders).order_id == "o_1001"
    assert resolve_order(Intent.RETURN_POLICY, orders).order_id == "o_1010"
    r5 = _orders(w, BUYER_R5)
    assert resolve_order(Intent.QUALITY_ISSUE, r5).order_id == "o_1013"
    # 退款进度：按售后单挑；as_2001 和 as_2003 同时申请 → 澄清
    buyer_as = [a for a in w.aftersales if a.buyer_ref == BUYER and a.shop_id == SHOP_A]
    res = resolve_order(Intent.REFUND_PROGRESS, orders, aftersales=buyer_as)
    assert res.needs_clarify and set(res.candidates) == {"o_1001", "o_1003"}
    assert resolve_order(Intent.REFUND_PROGRESS, orders, aftersales=[]).order_id is None
    assert resolve_order(Intent.URGE_SHIP, []).order_id is None


def test_provider_picks_order_from_recent_orders(env):
    bundle = env.collect(Intent.URGE_SHIP, with_recent_orders=True)
    assert bundle.get(ev.tool_order("o_1001")) is not None
    assert "承诺最晚发货时间" in bundle.get(ev.tool_order("o_1001")).content
    assert bundle.clarify_question is None


def test_provider_clarifies_when_several_orders_plausible(env):
    bundle = env.collect(Intent.LOGISTICS_TRACK, with_recent_orders=True)
    assert bundle.clarify_question == CLARIFY_ORDER_QUESTION == "请问您问的是哪个订单呢？"
    assert bundle.evidences == [] and bundle.missing == []


def test_provider_looks_up_recent_orders_when_context_has_none(env):
    bundle = env.collect(Intent.URGE_SHIP)
    assert ("list_recent_orders", SHOP_A, (BUYER,)) in env.inner.calls
    # o_1001 未发货：只有订单证据，没有物流轨迹
    assert bundle.ids() == [ev.tool_order("o_1001")]


def test_provider_missing_order(env):
    bundle = env.collect(Intent.LOGISTICS_TRACK, buyer_ref="b_nobody")
    assert bundle.missing == [ev.MISSING_ORDER]
    assert bundle.evidences == [] and bundle.clarify_question is None
    # 有订单但没有符合问题类型的
    bundle = env.collect(Intent.URGE_SHIP, buyer_ref=BUYER_R3, with_recent_orders=True)
    assert bundle.missing == [ev.MISSING_ORDER]


def test_provider_refund_progress_without_order_card(env):
    bundle = env.collect(Intent.REFUND_PROGRESS, buyer_ref=BUYER_REFUND, with_recent_orders=True)
    assert bundle.ids() == [
        ev.tool_order("o_1012"),
        ev.tool_aftersale("as_2006"),
        ev.tool_aftersale("as_2008"),
    ]
    # 买家的售后单只查一次（找订单时查的记录复用）
    assert sum(1 for c in env.inner.calls if c[0] == "list_aftersales") == 1
    clarify = env.collect(Intent.REFUND_PROGRESS, with_recent_orders=True)
    assert clarify.clarify_question == CLARIFY_ORDER_QUESTION


def test_provider_ignores_recent_orders_of_other_buyers(env):
    ctx = make_context("问题", env.w, with_recent_orders=True)
    foreign = env.w.order(SHOP_A, "o_1004")  # BUYER_R3 的订单
    ctx = ctx.model_copy(update={"recent_orders": [foreign]})
    bundle = env.provider.collect(ctx, Intent.RETURN_POLICY, ev.needs_for(Intent.RETURN_POLICY))
    assert bundle.missing == [ev.MISSING_ORDER] and bundle.evidences == []


# ---------------------------------------------------------------- 降级（§12.2）


@pytest.mark.parametrize(
    ("method", "intent", "kw", "marker", "remaining"),
    [
        ("get_logistics", Intent.LOGISTICS_TRACK, {"order_id": "o_1002"}, "tool:logistics", 1),
        ("list_aftersales", Intent.NOT_RECEIVED, {"order_id": "o_1003"}, "tool:aftersale", 2),
        ("list_recent_orders", Intent.URGE_SHIP, {}, "tool:order", 0),
        ("get_stock", Intent.STOCK, {"item_id": "i_555"}, "tool:stock", 0),
        ("get_price", Intent.PRICE_PROMO, {"item_id": "i_555"}, "tool:price", 0),
    ],
)
def test_tool_unavailable_is_marked_not_raised(env, method, intent, kw, marker, remaining):
    env.inner.fail_on = {method}
    bundle = env.collect(intent, **kw)
    assert bundle.unavailable == [marker]
    assert marker == ev.unavailable(marker.split(":")[1])
    assert len(bundle.evidences) == remaining


def test_refund_progress_aftersale_lookup_unavailable(env):
    env.inner.fail_on = {"list_aftersales"}
    bundle = env.collect(Intent.REFUND_PROGRESS, buyer_ref=BUYER_REFUND, with_recent_orders=True)
    assert bundle.unavailable == ["tool:aftersale"] and bundle.evidences == []


def test_stale_cache_keeps_original_fetched_at():
    env = Env(cached=True)
    env.collect(Intent.LOGISTICS_TRACK, order_id="o_1002")
    env.clock.advance(hours=2)
    env.inner.fail_on = {"get_logistics"}
    bundle = env.collect(Intent.LOGISTICS_TRACK, order_id="o_1002")
    assert bundle.unavailable == []
    log_ev = bundle.get(ev.tool_logistics("o_1002"))
    assert log_ev.fetched_at == env.w.now  # 两小时前的数据，回答里写原来的时间


# ---------------------------------------------------------------- 串店与个人信息


def test_order_from_other_shop_is_rejected(env):
    ctx = make_context("问题", env.w, order_id="o_1002")
    ctx = ctx.model_copy(update={"order": env.w.order(SHOP_B, "o_9001")})
    with pytest.raises(TenantIsolationError):
        env.provider.collect(ctx, Intent.LOGISTICS_TRACK, ev.needs_for(Intent.LOGISTICS_TRACK))


def test_other_shop_sees_nothing(env):
    bundle = env.collect(Intent.URGE_SHIP, shop_id=SHOP_B, with_recent_orders=True)
    assert bundle.ids() == [ev.tool_order("o_9001")]


def test_evidence_contents_contain_no_pii(env):
    w = env.w
    phone, addr, code = "13812345678", "文三路90号", "6-2-3301"
    t = w.trace(SHOP_A, "o_1003")
    dirty = t.events[-1].model_copy(
        update={"description": f"已签收，取件码 {code}，快递员{phone}，放在{addr}门口"}
    )
    w.traces[w.traces.index(t)] = t.model_copy(update={"events": [*t.events[:-1], dirty]})
    a = w.aftersale(SHOP_A, "as_2003")
    w.aftersales[w.aftersales.index(a)] = a.model_copy(
        update={"note": f"我电话{phone}，地址杭州市西湖区{addr}，一直没收到"}
    )
    needs = ALL_TOOL_NEEDS
    bundles = [
        env.collect(Intent.NOT_RECEIVED, needs=needs, order_id="o_1003", item_id="i_555"),
        env.collect(Intent.RETURN_POLICY, needs=needs, order_id="o_1003"),
    ]
    for bundle in bundles:
        assert bundle.evidences
        for e in bundle.evidences:
            assert not contains_pii(e.content), e.id
            blob = e.content + json.dumps(e.data, ensure_ascii=False)
            for secret in (phone, addr, "3301"):
                assert secret not in blob, (e.id, secret)


# ---------------------------------------------------------------- 实时数据覆盖同主题说法（§5.2.1）


def _lower(source: EvidenceSource, eid: str, content: str, topic: str) -> Candidate:
    return Candidate(Evidence(id=eid, source=source, label=eid, content=content, topic=topic))


def test_price_evidence_overrides_contradicting_promotions_policy(env):
    """价格与活动证据的主题是 promotions：和店铺满减政策冲突时（i_888 当前没有活动），
    政策被实时数据覆盖；一致时（i_555 参加满减）两者都保留。走真实的 provider 和冲突处理。"""
    policy = env.w.policy_map(SHOP_A)[PolicyKey.PROMOTIONS.value].value
    promo = _lower(EvidenceSource.POLICY, "policy:promotions", policy, PolicyKey.PROMOTIONS.value)
    for item_id, expected_lost in (("i_888", ["policy:promotions"]), ("i_555", [])):
        bundle = env.collect(Intent.PRICE_PROMO, item_id=item_id)
        price_ev = bundle.get(ev.tool_price(item_id))
        assert price_ev is not None and price_ev.topic == PolicyKey.PROMOTIONS.value
        _, lost = resolve_conflicts([Candidate.of(price_ev), promo])
        assert lost == expected_lost, item_id


def test_not_returnable_eligibility_overrides_contradicting_faq(env):
    """订单不能无理由退时，退换资格证据（主题 seven_day_return、否定说法）覆盖
    "支持七天无理由退货"的 FAQ；可以退时不覆盖（政策原文里的否定词不能误删一致的 FAQ）。"""
    faq = _lower(
        EvidenceSource.FAQ,
        "faq:99",
        "问：能退货吗\n答：亲，支持七天无理由退货哦",
        PolicyKey.SEVEN_DAY_RETURN.value,
    )
    for buyer, order_id, expected_lost in (
        (BUYER_R8, "o_1007", ["faq:99"]),  # 定制款
        (BUYER_R5, "o_1005", ["faq:99"]),  # 签收 23 天
        (BUYER_R3, "o_1004", []),  # 签收 5 天，可以退
    ):
        bundle = env.collect(Intent.RETURN_POLICY, order_id=order_id, buyer_ref=buyer)
        ret = bundle.get(ev.tool_return(order_id))
        assert ret is not None
        _, lost = resolve_conflicts([Candidate.of(ret), faq])
        assert lost == expected_lost, order_id


# ---------------------------------------------------------------- 过期缓存：获取时间与降级（§12.2）


def test_stale_order_and_aftersale_keep_fetch_time_and_are_flagged():
    env = Env(cached=True)
    provider = DefaultToolEvidenceProvider(
        env.gw,
        env.w.policy_map,
        DefaultLogisticsAdvisor(),
        DefaultAfterSaleAdvisor(),
        env.clock,
        stale_after=timedelta(seconds=300),
    )
    needs = ev.needs_for(Intent.REFUND_PROGRESS)
    ctx = make_context("退款什么时候到", env.w, buyer_ref=BUYER_REFUND)
    fresh = provider.collect(ctx, Intent.REFUND_PROGRESS, needs)
    fetched = env.clock.now()
    assert fresh.unavailable == []
    assert fresh.get(ev.tool_aftersale("as_2008")).fetched_at == fetched

    # 4 分钟后：TTL 内命中缓存，不算降级，时间是当初的获取时间
    env.clock.advance(minutes=4)
    cached = provider.collect(ctx, Intent.REFUND_PROGRESS, needs)
    assert cached.unavailable == []
    assert cached.get(ev.tool_order("o_1012")).fetched_at == fetched

    # 半小时后平台故障：过期缓存兜底，写原来的获取时间，并标降级（卡片提示客服核实）
    env.clock.advance(minutes=26)
    env.inner.fail_on = {"list_aftersales", "list_recent_orders", "get_order"}
    stale = provider.collect(ctx, Intent.REFUND_PROGRESS, needs)
    as_ev = stale.get(ev.tool_aftersale("as_2008"))
    assert as_ev.fetched_at == fetched
    assert RefundProgress.model_validate(as_ev.data["progress"]).as_of == fetched
    assert stale.get(ev.tool_order("o_1012")).fetched_at == fetched
    assert stale.unavailable == ["tool:order", "tool:aftersale"]


def test_stale_order_card_is_flagged():
    """会话理解给的订单卡片（经带缓存的网关取到）过期了：订单证据写原来的获取时间并标降级。"""
    env = Env(cached=True)
    provider = DefaultToolEvidenceProvider(
        env.gw,
        env.w.policy_map,
        DefaultLogisticsAdvisor(),
        DefaultAfterSaleAdvisor(),
        env.clock,
        stale_after=timedelta(seconds=300),
    )
    order = env.gw.get_order(SHOP_A, "o_1007")
    fetched = env.clock.now()
    env.clock.advance(hours=1)
    ctx = make_context("能退吗", env.w, buyer_ref=BUYER_R8).model_copy(update={"order": order})
    bundle = provider.collect(ctx, Intent.RETURN_POLICY, ev.needs_for(Intent.RETURN_POLICY))
    assert bundle.get(ev.tool_order("o_1007")).fetched_at == fetched
    assert bundle.get(ev.tool_return("o_1007")).fetched_at == fetched
    assert bundle.unavailable == ["tool:order"]


# ---------------------------------------------------------------- 库存：没定到 SKU 时逐个规格写


def test_stock_without_sku_lists_out_of_stock_specs(env):
    """买家问"M 码还有货吗"：对应黑色 M、白色 M 两个 SKU（会话理解定不到一个 SKU）：
    整款合计"有货"会被当成 M 码有货，证据要写清哪些规格缺货。"""
    for i, s in enumerate(env.w.stocks):
        if s.item_id == "i_555":
            qty = {"k_1": 0, "k_2": 3, "k_3": 0, "k_4": 0}[s.sku_id or ""]
            env.w.stocks[i] = s.model_copy(update={"quantity": qty, "in_stock": qty > 0})
    bundle = env.collect(Intent.STOCK, item_id="i_555")
    [stock_ev] = bundle.evidences
    assert stock_ev.id == ev.tool_stock("i_555")
    assert stock_ev.topic == ev.topic_for_stock("i_555")
    assert "有货，库存 3 件" not in stock_ev.content
    assert "部分规格有货" in stock_ev.content
    missing = stock_ev.content.split("暂时缺货：", 1)[1].split("；", 1)[0]
    for spec in ("黑色 / M", "白色 / M", "白色 / S"):
        assert spec in missing, stock_ev.content
    assert "黑色 / L" not in missing
    skus = [StockInfo.model_validate(s) for s in stock_ev.data["skus"]]
    assert {s.sku_id: s.in_stock for s in skus} == {
        "k_1": False,
        "k_2": True,
        "k_3": False,
        "k_4": False,
    }
    assert StockInfo.model_validate(stock_ev.data["stock"]).in_stock is True


def test_stock_without_sku_all_in_stock_keeps_summary(env):
    for i, s in enumerate(env.w.stocks):
        if s.item_id == "i_555":
            env.w.stocks[i] = s.model_copy(update={"quantity": 5, "in_stock": True})
    [stock_ev] = env.collect(Intent.STOCK, item_id="i_555").evidences
    assert "有货，库存 20 件" in stock_ev.content and "缺货" not in stock_ev.content


def test_stock_without_sku_per_sku_lookup_unavailable_is_labelled(env):
    """按 SKU 查失败时只剩整款合计：写明是整款合计、没有区分规格，不能当成某个规格有货。"""
    original = env.inner.get_stock

    def flaky(shop_id, item_id, sku_id=None):
        if sku_id is not None:
            raise ToolUnavailable("限流（模拟）")
        return original(shop_id, item_id, sku_id)

    env.inner.get_stock = flaky  # type: ignore[method-assign]
    [stock_ev] = env.collect(Intent.STOCK, item_id="i_555").evidences
    assert "整款合计" in stock_ev.content and "未区分规格" in stock_ev.content


# ---------------------------------------------------------------- 退货问题找订单要看当前商品


def test_return_policy_order_must_contain_current_item(env):
    """买家在看 i_888（没买过）问能不能退：不拿他别的订单（o_1010 牛仔裤）的退换资格。"""
    bundle = env.collect(Intent.RETURN_POLICY, item_id="i_888", with_recent_orders=True)
    assert bundle.evidences == []
    assert bundle.missing == [] and bundle.clarify_question is None
    # 买过当前商品：用含这件商品的那一单
    bundle = env.collect(Intent.RETURN_POLICY, item_id="i_777", with_recent_orders=True)
    assert bundle.ids() == [ev.tool_order("o_1010"), ev.tool_return("o_1010")]
    bundle = env.collect(Intent.QUALITY_ISSUE, item_id="i_555", with_recent_orders=True)
    assert bundle.get(ev.tool_order("o_1003")) is not None
    # 没有当前商品：照旧按问题类型挑最近一单
    bundle = env.collect(Intent.RETURN_POLICY, with_recent_orders=True)
    assert bundle.get(ev.tool_return("o_1010")) is not None
