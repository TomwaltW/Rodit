"""检索层：中文数字写的旧 FAQ 被政策覆盖（§5.2.1）；长度预算按提示词里的整行计（§14）。"""

from __future__ import annotations

from datetime import timedelta

from rodit.core.config import GenerationConfig
from rodit.core.enums import EvidenceSource, Intent, KnowledgeSource, KnowledgeStatus
from rodit.core.evidence_line import evidence_line_chars
from rodit.core.models import Evidence, KnowledgeItem
from rodit.knowledge.conflict import Candidate, trim_to_budget
from rodit.testing.fakes import make_context
from rodit.testing.fixtures import SHOP_B


def _faq(world, shop_id: str, question: str, answer: str) -> KnowledgeItem:
    return KnowledgeItem(
        shop_id=shop_id,
        question=question,
        answer=answer,
        source=KnowledgeSource.MERCHANT,
        status=KnowledgeStatus.APPROVED,
        created_at=world.now - timedelta(days=1),
    )


def test_stale_faq_with_chinese_numerals_is_overridden(world, repo, make_retriever, route):
    """审查复现：店铺 B 政策 72 小时，FAQ 写"两天内发货" → FAQ 被覆盖，不能当直接回答。"""
    repo.add_knowledge(_faq(world, SHOP_B, "什么时候发货", "您好，付款后两天内发货。"))
    ctx = make_context("什么时候发货", world, shop_id=SHOP_B)
    b = make_retriever().retrieve(ctx, route(Intent.SHIPPING_POLICY))
    faq_ids = [i for i in b.overridden_ids if i.startswith("faq:")]
    assert faq_ids, b.overridden_ids
    assert not any("两天" in e.content for e in b.evidences)
    assert "72 小时" in b.get("policy:ship_time").content


def test_faq_with_same_duration_in_chinese_is_kept(world, repo, make_retriever, route):
    """ "三天内发货"和 72 小时一致：两者都保留（§5.4 FAQ 仍可直接用）。"""
    repo.add_knowledge(_faq(world, SHOP_B, "什么时候发货", "您好，付款后三天内发货。"))
    ctx = make_context("什么时候发货", world, shop_id=SHOP_B)
    b = make_retriever().retrieve(ctx, route(Intent.SHIPPING_POLICY))
    assert not [i for i in b.overridden_ids if i.startswith("faq:")]
    assert any("三天" in e.content for e in b.evidences)


def _detail(n: int, text: str) -> Evidence:
    return Evidence(
        id=f"item:i_long:detail:{n}",
        source=EvidenceSource.ITEM,
        label="商品详情",
        content=text,
    )


def test_trim_budgets_the_rendered_prompt_line():
    """预算按提示词里的整行（[id] 标签：内容 + 换行）计，不只数 content。"""
    cands = [Candidate(_detail(n, "字" * 20), rank=0.1) for n in range(5)]
    cands.append(Candidate(_detail(9, "相关" * 10), rank=0.9))
    per_line = evidence_line_chars(cands[0].evidence)
    assert per_line > 20
    budget = per_line * 3  # 按 content 算放得下 6 条，按整行只放得下 2 条
    kept = trim_to_budget(cands, budget)
    ids = [c.evidence.id for c in kept]
    assert "item:i_long:detail:9" in ids  # rank 最高的留下
    assert sum(evidence_line_chars(c.evidence) for c in kept) + len(kept) - 1 <= budget
    assert len(kept) == 2


def test_default_budget_unchanged_for_small_bundles():
    cands = [Candidate(_detail(n, "字" * 20)) for n in range(3)]
    assert trim_to_budget(cands, GenerationConfig().max_evidence_chars) == cands
