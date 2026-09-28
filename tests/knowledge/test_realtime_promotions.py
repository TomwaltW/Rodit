"""实时价格与活动 vs 店铺满减政策 / FAQ（§5.2.1 实时接口优先级 1，只在真冲突时覆盖）。

复审发现：价格证据改用 promotions 主题后，通用的 contradicts 拿整段政策和很短的价格内容比，
政策多写一句"定制款不参加"或活动日期就算冲突，和实时活动一致的政策被删掉，
买家问"能叠加券吗"时提示词里没了"可以叠加平台券"。
这里走真实的 KnowledgeRetriever + DefaultToolEvidenceProvider（i_555 参加店铺满 199 减 20，
i_888 定制款当前没有活动）。
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from rodit.aftersale.advisor import DefaultAfterSaleAdvisor
from rodit.core.config import GenerationConfig
from rodit.core.enums import (
    EvidenceSource,
    Intent,
    KnowledgeSource,
    KnowledgeStatus,
    PolicyKey,
    RouteAction,
)
from rodit.core.models import (
    Evidence,
    EvidenceBundle,
    KnowledgeItem,
    PriceInfo,
    RouteDecision,
    RoutePart,
)
from rodit.inmemory import HashingEmbedder, InMemoryKnowledgeRepository
from rodit.knowledge import FaqSearcher, KnowledgeRetriever
from rodit.knowledge.conflict import Candidate, resolve_conflicts
from rodit.testing.fakes import FakeClock, InMemoryToolGateway, make_context
from rodit.testing.fixtures import SHOP_A, build_demo_world, seed_knowledge
from rodit.tools.evidence import DefaultToolEvidenceProvider
from rodit.tools.logistics import DefaultLogisticsAdvisor

PROMO = "policy:promotions"


def _retrieve(
    item_id: str,
    *,
    policy: str | None = None,
    faq: tuple[str, str] | None = None,
    question: str = "现在有什么优惠？能叠加券吗",
    also: Intent | None = None,
) -> EvidenceBundle:
    w = build_demo_world()
    if policy is not None:
        for i, p in enumerate(w.policies):
            if p.shop_id == SHOP_A and p.key == PolicyKey.PROMOTIONS.value:
                w.policies[i] = p.model_copy(update={"value": policy})
    if faq is not None:
        w.faqs.append(
            KnowledgeItem(
                shop_id=SHOP_A,
                question=faq[0],
                answer=faq[1],
                source=KnowledgeSource.MERCHANT,
                status=KnowledgeStatus.APPROVED,
                approved_by="merchant",
                intent=Intent.PRICE_PROMO,
                created_at=w.now - timedelta(days=1),
            )
        )
    clock = FakeClock(w.now)
    repo = InMemoryKnowledgeRepository(clock)
    seed_knowledge(repo, w)
    tools = DefaultToolEvidenceProvider(
        InMemoryToolGateway(w),
        w.policy_map,
        DefaultLogisticsAdvisor(),
        DefaultAfterSaleAdvisor(),
        clock,
    )
    retriever = KnowledgeRetriever(
        repo, FaqSearcher(repo, HashingEmbedder(256)), GenerationConfig(), tool_provider=tools
    )
    route = RouteDecision(
        action=RouteAction.SPLIT_MULTI if also else RouteAction.ANSWER,
        rule=5 if also else 6,
        intent=Intent.PRICE_PROMO,
        intent_confidence=0.9,
        parts=[RoutePart(intent=also, confidence=0.45)] if also else [],
        provider="test",
    )
    return retriever.retrieve(make_context(question, w, item_id=item_id), route)


@pytest.mark.parametrize(
    "policy",
    [
        "店铺满 199 减 20，可以叠加平台券",
        "国庆期间（10 月 1 日至 10 月 7 日）店铺满 199 减 20，可以叠加平台券",
        "店铺满 199 减 20，可以叠加平台券；定制款不参加",
        "9 月 20 日至 10 月 7 日店铺满 199 减 20，可以叠加平台券",
        "店铺满 199 减 20，可以叠加平台券，特价商品不参加满减",
    ],
)
def test_consistent_promotions_policy_is_kept(policy):
    """实时活动就是"店铺满 199 减 20"：政策多说了日期、叠加平台券、哪些商品不参加，都不算冲突。"""
    b = _retrieve("i_555", policy=policy)
    assert PROMO not in b.overridden_ids, policy
    kept = b.get(PROMO)
    assert kept is not None and "可以叠加平台券" in kept.content
    assert b.get("tool:price:i_555") is not None


@pytest.mark.parametrize(
    ("item_id", "policy"),
    [
        ("i_555", "店铺满 299 减 30，可以叠加平台券"),  # 满减档位和实时活动对不上
        ("i_888", "店铺满 199 减 20，可以叠加平台券"),  # 实时：当前没有活动
        ("i_888", "全场 9 折，可以叠加平台券"),
    ],
)
def test_contradicting_promotions_policy_is_overridden(item_id, policy):
    b = _retrieve(item_id, policy=policy)
    assert PROMO in b.overridden_ids, policy
    assert b.get(PROMO) is None
    assert b.get(f"tool:price:{item_id}") is not None


def test_policy_saying_no_activity_agrees_with_realtime_no_activity():
    b = _retrieve("i_888", policy="目前店铺没有满减活动，可以用平台券")
    assert PROMO not in b.overridden_ids
    assert b.get(PROMO) is not None


def test_faq_claiming_activity_is_overridden_when_realtime_has_none():
    """ "有满减吗？什么面料"（价格 + 商品参数，后者要检索 FAQ）：FAQ 说全店满 199 减 20，
    i_888 实时没有活动 → FAQ 被覆盖；i_555 参加满减 → FAQ 保留。"""
    faq = ("有满减吗", "亲，全店满 199 减 20 哦")
    for item_id, overridden in (("i_888", True), ("i_555", False)):
        b = _retrieve(item_id, faq=faq, question="有满减吗", also=Intent.PRODUCT_SPEC)
        faq_ids = [e.id for e in b.evidences if e.id.startswith("faq:")]
        lost = [i for i in b.overridden_ids if i.startswith("faq:")]
        assert bool(lost) is overridden, (item_id, faq_ids, b.overridden_ids)
        assert bool(faq_ids) is not overridden, item_id


def _price(promotions: list[str]) -> Candidate:
    w = build_demo_world()
    price = PriceInfo(
        shop_id=SHOP_A,
        item_id="i_1",
        price=259,
        promotions=promotions,
        fetched_at=w.now,
    )
    return Candidate.of(
        Evidence(
            id="tool:price:i_1",
            source=EvidenceSource.REALTIME,
            label="实时数据 · 价格与活动",
            content="价格",
            data={"price": price.model_dump(mode="json")},
            topic="price:i_1",
            fetched_at=w.now,
        )
    )


def _policy(text: str) -> Candidate:
    return Candidate(
        Evidence(
            id=PROMO,
            source=EvidenceSource.POLICY,
            label="店铺政策 · 优惠活动",
            content=text,
            topic=PolicyKey.PROMOTIONS.value,
        )
    )


@pytest.mark.parametrize(
    ("promotions", "policy", "lost"),
    [
        (["店铺满 199 减 20"], "店铺满 199 减 20，定制款不参加", []),
        (["店铺满 199 减 20", "满 299 减 40"], "满 199 减 20、满 299 减 40", []),
        (["店铺满 199 减 20"], "满 199 减 30", [PROMO]),
        (["店铺满 199 减 20"], "满 199 减 20，第二件 8 折", [PROMO]),
        ([], "满 199 减 20", [PROMO]),
        ([], "当前没有活动", []),
    ],
)
def test_promotions_compared_by_activity_terms(promotions, policy, lost):
    _, got = resolve_conflicts([_price(promotions), _policy(policy)])
    assert got == lost
