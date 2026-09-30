"""§13.1 评测集自检：规模与覆盖、按问题类型分层的测试集、初版用例的分组不变、
标注和合成演示世界一致（依据 ID 真实存在且属于这位买家 / 这件商品，要点来自演示世界的事实或方案）、
标注之间不自相矛盾（§6.6 路由规则）、文字里的个人信息都是合成的且回答不能复述。

这些检查不跑引擎：标注的对错只看方案和演示世界，不看离线流水线的输出。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pytest

from rodit.core import evidence as ev
from rodit.core.clock import fmt_time
from rodit.core.config import EngineConfig
from rodit.core.enums import Intent, KnowledgeStatus, Platform, Role, RouteAction
from rodit.eval.cases_base import EvalCase, hash_split
from rodit.eval.dataset import CASES, CASES_V1, cases
from rodit.inmemory.repositories import InMemoryKnowledgeRepository
from rodit.knowledge.category import CATEGORY_TIPS, category_group
from rodit.knowledge.platform_rules import PLATFORM_RULES
from rodit.testing import fixtures
from rodit.testing.fixtures import DemoWorld, build_demo_world, seed_knowledge

# 初版 81 条按 id 哈希分到测试集的用例：扩充后分组不能变
LEGACY_TEST_IDS = frozenset(
    {
        "spec-01",
        "spec-07",
        "size-01",
        "usage-03",
        "promo-04",
        "ship-02",
        "logi-05",
        "urge-02",
        "mod-02",
        "ret-05",
        "cmp-02",
        "emo-01",
        "off-02",
        "off-04",
        "ctx-03",
    }
)

# 要点（must_include）不在演示世界的数据里、而是方案规定的回答写法
DOC_PHRASES = {
    "截至": "§7.2 物流回答里始终写明“截至 {时间}”",
    "催查": "§7.2 物流停滞：买家来问时推荐“已催查”话术",
    "哪": "§5.2.4 商品 / 订单不确定时推荐一句澄清问题（请问您问的是哪一款 / 哪个订单）",
}

# 用例文字里出现的手机号都是手造的
SYNTHETIC_PHONES = frozenset(
    {"13812345678", "13800005678", "13900001111", "13600003333", "15800002222"}
)
_PHONE = re.compile(r"(?<!\d)1[3-9]\d(?:[\s-]?\d{4}){2}(?!\d)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")

HIGH_RISK = frozenset(EngineConfig().routing.high_risk)  # 默认值即方案 §6.6
_NO_CARD_ACTIONS = {RouteAction.HUMAN_PRIORITY, RouteAction.NO_SUGGESTION}


# ================================================================ 演示世界里存在的依据


@dataclass
class _Evidence:
    """一个店铺在演示世界里能产生的依据 ID，及其归属（商品 / 订单）。"""

    shop_wide: set[str] = field(default_factory=set)  # 政策、平台规则、FAQ、类目通用知识
    by_item: dict[str, str] = field(default_factory=dict)  # 依据 ID → 商品
    by_order: dict[str, str] = field(default_factory=dict)  # 依据 ID → 订单

    def all(self) -> set[str]:
        return self.shop_wide | set(self.by_item) | set(self.by_order)


def _faq_ids(world: DemoWorld) -> dict[tuple[str, str], int]:
    return seed_knowledge(InMemoryKnowledgeRepository(), world)


def _evidence(world: DemoWorld, shop_id: str) -> _Evidence:
    shop = world.shop(shop_id)
    out = _Evidence()
    out.shop_wide |= {ev.policy(p.key) for p in world.policies if p.shop_id == shop_id}
    out.shop_wide |= {
        ev.platform_rule(platform, key)
        for platform, key in PLATFORM_RULES
        if platform == shop.platform
    }
    out.shop_wide |= {
        ev.category(tip.key) for tip in CATEGORY_TIPS.get(category_group(shop.category), ())
    }
    faq_ids = _faq_ids(world)
    for k in world.faqs:
        if k.shop_id != shop_id or k.status != KnowledgeStatus.APPROVED:
            continue
        fid = ev.faq(faq_ids[(k.shop_id, k.question)])
        if k.item_id:
            out.by_item[fid] = k.item_id
        else:
            out.shop_wide.add(fid)
    for card in world.items:
        if card.shop_id != shop_id:
            continue
        iid = card.item_id
        ids = [ev.item_title(iid), *(ev.item_attr(iid, name) for name in card.attributes)]
        if card.skus:
            ids.append(ev.item_sku(iid))
        if card.size_chart:
            ids.append(ev.item_size_chart(iid))
        if card.size_advice:
            ids.append(ev.item_size_advice(iid))
        if card.care:
            ids.append(ev.item_care(iid))
        if card.usage:
            ids.append(ev.item_usage(iid))
        ids += [
            ev.item_detail(iid, seg)
            for seg, _ in enumerate(ev.split_sentences(card.detail_text or ""))
        ]
        out.by_item.update(dict.fromkeys(ids, iid))
    for s in world.stocks:
        if s.shop_id == shop_id:
            out.by_item[ev.tool_stock(s.item_id)] = s.item_id
            if s.sku_id:
                out.by_item[ev.tool_stock(s.item_id, s.sku_id)] = s.item_id
    for p in world.prices:
        if p.shop_id == shop_id:
            out.by_item[ev.tool_price(p.item_id)] = p.item_id
    for o in world.orders:
        if o.shop_id == shop_id:
            out.by_order[ev.tool_order(o.order_id)] = o.order_id
            out.by_order[ev.tool_return(o.order_id)] = o.order_id
    for t in world.traces:
        if t.shop_id == shop_id:
            out.by_order[ev.tool_logistics(t.order_id)] = t.order_id
    for a in world.aftersales:
        if a.shop_id == shop_id:
            out.by_order[ev.tool_aftersale(a.aftersale_id)] = a.order_id
    return out


def _facts(world: DemoWorld, case: EvalCase) -> str:
    """这条用例能用到的演示世界事实（店铺政策、平台规则、FAQ、类目知识、当前商品、这位买家的订单）。"""
    shop_id = getattr(fixtures, case.shop)
    buyer = getattr(fixtures, case.buyer)
    shop = world.shop(shop_id)
    parts: list[str] = [p.value for p in world.policies if p.shop_id == shop_id]
    parts += [r.text for (platform, _), r in PLATFORM_RULES.items() if platform == shop.platform]
    parts += [tip.text for tip in CATEGORY_TIPS.get(category_group(shop.category), ())]
    for k in world.faqs:
        if k.shop_id == shop_id and k.status == KnowledgeStatus.APPROVED:
            if k.item_id is None or k.item_id == case.item_id:
                parts += [k.question, k.answer]
    if case.item_id:
        card = world.item(shop_id, case.item_id)
        parts += [card.title, card.size_advice or "", card.detail_text or ""]
        parts += [f"{k}：{v}" for k, v in card.attributes.items()]
        parts += [v for sku in card.skus for v in sku.props().values()]
        parts += [str(v) for row in card.size_chart for v in row.values()]
        parts += [*card.care, *card.usage]
        for s in world.stocks:
            if s.shop_id == shop_id and s.item_id == case.item_id:
                parts += [str(s.quantity), s.restock_note or ""]
        for p in world.prices:
            if p.shop_id == shop_id and p.item_id == case.item_id:
                parts += [str(p.price), *p.promotions]
    orders = {o.order_id for o in world.orders if o.shop_id == shop_id and o.buyer_ref == buyer}
    for o in world.orders:
        if o.order_id in orders and o.shop_id == shop_id:
            parts += [o.order_id, str(o.total_amount), o.courier or "", o.waybill_no or ""]
            parts += [ln.title for ln in o.lines]
    for t in world.traces:
        if t.shop_id == shop_id and t.order_id in orders:
            parts += [t.courier, t.waybill_no, t.pickup_info or ""]
            for e in t.events:
                parts += [e.description, e.location or "", fmt_time(e.time)]
    for a in world.aftersales:
        if a.shop_id == shop_id and a.order_id in orders:
            parts += [a.aftersale_id, str(a.amount)]
    return "\n".join(parts)


@pytest.fixture(scope="module")
def world() -> DemoWorld:
    return build_demo_world()


# ================================================================ 规模、覆盖、分组


def test_size_and_coverage():
    assert 300 <= len(CASES) <= 500  # §13.1：每个首发类目 300–500 条
    by_intent = Counter(c.expected_intent for c in CASES)
    assert set(by_intent) == set(Intent)
    assert min(by_intent.values()) >= 8, by_intent
    tags = Counter(t for c in CASES for t in c.tags)
    minimum = {  # 略低于当前条数：删用例时要有意识地改这里
        "multi": 12,
        "reference": 14,
        "pii": 8,
        "emotion": 10,
        "typo": 5,
        "colloquial": 14,
        "knowledge_gap": 10,
        "no_item_card": 3,
        "no_order_card": 5,
        "ambiguous_order": 2,
        "human": 10,
        "forbidden_claim": 4,
        "conflict": 3,
    }
    for tag, n in minimum.items():
        assert tags[tag] >= n, (tag, tags[tag])
    for flow in ("§11.1", "§11.2", "§11.3", "§11.4", "§11.5"):  # §11 的五个核心流程
        assert tags[flow] >= 3, (flow, tags[flow])
    assert sum(1 for c in CASES if c.item_id) >= 110  # 带商品卡片
    assert sum(1 for c in CASES if c.order_id) >= 85  # 带订单卡片
    assert sum(1 for c in CASES if c.history) >= 14  # 多轮
    assert sum(1 for c in CASES if c.no_suggestion_ok) >= 35  # 不给推荐也对


def test_ids_unique_and_well_formed():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))
    for c in CASES:
        assert re.fullmatch(r"[a-z]+-\d{2}", c.id), c.id
        assert c.text.strip(), c.id
    keys = [(c.text, c.item_id, c.sku_id, c.order_id, c.buyer, tuple(c.history)) for c in CASES]
    dup = [k for k, n in Counter(keys).items() if n > 1]
    assert not dup, dup


def test_legacy_cases_keep_their_split():
    assert len(CASES_V1) == 81 and CASES[:81] == CASES_V1
    for c in CASES_V1:
        assert c.holdout is None and c.split == hash_split(c.id), c.id
    assert {c.id for c in CASES_V1 if c.split == "test"} == LEGACY_TEST_IDS
    # 扩充的用例都显式标注分组，以后再加用例不会改变已有用例的分组
    assert all(c.holdout is not None for c in CASES[81:])


def test_holdout_is_stratified_by_intent():
    test = cases("test")
    assert 0.18 <= len(test) / len(CASES) <= 0.22
    assert {c.expected_intent for c in test} == set(Intent)  # 每个问题类型都在测试集里
    total = Counter(c.expected_intent for c in CASES)
    held = Counter(c.expected_intent for c in test)
    for intent, n in total.items():
        assert 0.1 <= held[intent] / n <= 0.4, (intent, held[intent], n)
    assert len(cases("dev")) + len(test) == len(CASES)


# ================================================================ 和演示世界一致


def test_cases_refer_to_demo_world(world: DemoWorld):
    for c in CASES:
        shop_id = getattr(fixtures, c.shop)
        buyer = getattr(fixtures, c.buyer)
        assert world.shop(shop_id).platform == Platform.TAOBAO, c.id  # 评测跑淘宝演示单元
        assert fixtures.RAW_BUYERS[c.buyer][0] == "taobao", c.id
        if c.sku_id:
            assert c.item_id, c.id
        if c.item_id:
            card = world.item(shop_id, c.item_id)
            if c.sku_id:
                assert card.find_sku(c.sku_id) is not None, c.id
        if c.order_id:
            order = world.order(shop_id, c.order_id)
            assert order.buyer_ref == buyer, c.id
            if c.item_id:
                assert any(ln.item_id == c.item_id for ln in order.lines), c.id
        for role, text in c.history:
            assert role in (Role.BUYER, Role.AGENT) and text.strip(), c.id
        if c.history:
            assert c.history[-1][0] == Role.AGENT, c.id  # 当前消息接在客服回复之后


def test_must_cite_ids_exist_and_belong_to_the_case(world: DemoWorld):
    """依据 ID 必须是演示世界里真能查到的；商品依据属于当前商品，
    订单依据属于这位买家（带了订单卡片时就是这一单）。"""
    evidence = {sid: _evidence(world, sid) for sid in world.shops}
    for c in CASES:
        shop_id = getattr(fixtures, c.shop)
        buyer = getattr(fixtures, c.buyer)
        known = evidence[shop_id]
        for group in c.must_cite:
            assert group and len(group) == len(set(group)), c.id
            for eid in group:
                assert eid in known.all(), (c.id, eid)
                if eid in known.by_item:
                    assert known.by_item[eid] == c.item_id, (c.id, eid)
                if eid in known.by_order:
                    order = world.order(shop_id, known.by_order[eid])
                    assert order.buyer_ref == buyer, (c.id, eid)
                    if c.order_id:
                        assert order.order_id == c.order_id, (c.id, eid)


def test_must_include_points_come_from_world_or_doc(world: DemoWorld):
    """要点来自演示世界的事实（政策、商品、这位买家的订单 / 物流 / 售后……）或方案规定的写法。"""
    for c in CASES:
        facts = _facts(world, c)
        for point in c.must_include:
            assert point.strip(), c.id
            assert point in facts or point in DOC_PHRASES, (c.id, point)


# ================================================================ 标注自洽（§5.1、§5.7、§6.6）


def test_labels_are_consistent_with_routing_rules():
    for c in CASES:
        ok_intents = {c.expected_intent, *c.acceptable_intents}
        actions = set(c.expected_actions)
        assert c.expected_intent not in c.acceptable_intents, c.id
        assert len(c.acceptable_intents) == len(set(c.acceptable_intents)), c.id
        assert len(c.expected_actions) == len(actions), c.id
        assert RouteAction.TRANSFER_HUMAN not in actions, c.id  # 评测跑智能辅助模式
        if c.expected_intent == Intent.OFF_PLATFORM:  # 规则 1
            assert c.expected_actions == [RouteAction.OFF_PLATFORM_SCRIPT], c.id
        if c.expected_intent == Intent.COMPLAINT:  # 投诉不推荐，提醒人工（§5.1）
            assert RouteAction.HUMAN_PRIORITY in actions, c.id
        if c.expected_intent in HIGH_RISK:  # 规则 4：高风险类型要带风险提示
            assert actions, c.id
            if ok_intents <= HIGH_RISK:
                assert RouteAction.ANSWER not in actions, c.id
        if actions & _NO_CARD_ACTIONS:  # 提醒人工 / 不推荐时本来就没有推荐
            assert c.no_suggestion_ok, c.id


def test_include_and_exclude_do_not_contradict():
    for c in CASES:
        assert all(s.strip() for s in c.must_not_include), c.id
        for good in c.must_include:
            for bad in c.must_not_include:
                assert good not in bad and bad not in good, (c.id, good, bad)


def test_pii_is_synthetic_and_never_echoed():
    for c in CASES:
        texts = [c.text, *(t for _, t in c.history)]
        for text in texts:
            for m in _PHONE.finditer(text):
                assert re.sub(r"\D", "", m.group(0)) in SYNTHETIC_PHONES, (c.id, m.group(0))
            for m in _EMAIL.finditer(text):
                assert m.group(0).endswith("@example.com"), (c.id, m.group(0))
        if "pii" in c.tags:
            # 回答不能复述买家留的手机号、地址、邮箱、微信号（§5.3 最小化个人信息）
            assert any(s in c.text for s in c.must_not_include), c.id
