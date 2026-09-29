"""和店铺发货政策一致、只是多说了"周六日也正常发货"的 FAQ 不能被当成冲突删掉（§5.2.1、§5.4）。

复审发现：中文数字转换把"周六日"变成"周6日"（6 天），政策里没有 6 天，FAQ 被判冲突覆盖，
买家问"周末发货吗"时证据里没有周末发货的说法。
"""

from __future__ import annotations

from datetime import timedelta

from rodit.core.enums import Intent, KnowledgeSource, KnowledgeStatus
from rodit.core.models import KnowledgeItem
from rodit.testing.fakes import make_context
from rodit.testing.fixtures import SHOP_A


def _faq(world, question: str, answer: str) -> KnowledgeItem:
    return KnowledgeItem(
        shop_id=SHOP_A,
        question=question,
        answer=answer,
        source=KnowledgeSource.MERCHANT,
        status=KnowledgeStatus.APPROVED,
        approved_by="merchant",
        intent=Intent.SHIPPING_POLICY,
        created_at=world.now - timedelta(days=1),
    )


def test_weekend_shipping_faq_consistent_with_policy_is_kept(world, repo, make_retriever, route):
    faq_id = repo.add_knowledge(
        _faq(world, "周末发货吗", "亲，周六日也正常发货哦，现货付款后 48 小时内发货")
    ).id
    ctx = make_context("周末发货吗", world)
    b = make_retriever().retrieve(ctx, route(Intent.SHIPPING_POLICY))
    assert f"faq:{faq_id}" not in b.overridden_ids
    assert b.get(f"faq:{faq_id}") is not None
    assert b.get("policy:ship_time") is not None


def test_weekend_faq_with_different_duration_still_overridden(world, repo, make_retriever, route):
    faq_id = repo.add_knowledge(
        _faq(world, "周末发货吗", "亲，周六日也正常发货哦，现货付款后 24 小时内发货")
    ).id
    ctx = make_context("周末发货吗", world)
    b = make_retriever().retrieve(ctx, route(Intent.SHIPPING_POLICY))
    assert f"faq:{faq_id}" in b.overridden_ids
