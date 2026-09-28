"""FAQ 语义检索（§5.2.4）：懒索引、分数语义、店铺隔离（§9.2）、状态与商品范围、降级。"""

from __future__ import annotations

import pytest

from rodit.core.config import GenerationConfig
from rodit.core.enums import KnowledgeStatus
from rodit.core.errors import TenantIsolationError
from rodit.knowledge import EXACT_MATCH_SCORE, SEMANTIC_SCORE_CAP, FaqSearcher, normalize_question
from rodit.testing.fixtures import SHOP_A, SHOP_B

DIRECT = GenerationConfig().faq_direct_min_score


def _questions(hits) -> list[str]:
    return [k.question for k, _ in hits]


# ---------------------------------------------------------------- 索引


def test_index_embeds_only_approved_questions_and_is_idempotent(repo, faq_ids, spy):
    searcher = FaqSearcher(repo, spy)
    assert searcher.index(SHOP_A) == 5  # 店铺 A 已审核的 5 条（含商品专属）
    assert sorted(spy.texts) == sorted(
        ["发什么快递", "什么时候发货", "能开发票吗", "有运费险吗", "这款会透吗"]
    )  # 对"问题"算向量，不是回答
    for (shop, question), kid in faq_ids.items():
        item = repo.get_knowledge(shop, kid)
        if shop == SHOP_A and item.status == KnowledgeStatus.APPROVED:
            assert item.embedding is not None and len(item.embedding) == spy.dim
        else:  # 候选、已下线、别的店都不动
            assert item.embedding is None, question
    assert searcher.index(SHOP_A) == 0
    assert searcher.index(SHOP_B) == 1


def test_index_recomputes_embeddings_with_wrong_dimension(repo, faq_ids, spy):
    kid = faq_ids[(SHOP_A, "发什么快递")]
    repo.set_knowledge_embedding(SHOP_A, kid, [1.0, 0.0, 0.0])  # 旧向量模型留下的
    searcher = FaqSearcher(repo, spy)
    assert searcher.index(SHOP_A) == 5
    assert len(repo.get_knowledge(SHOP_A, kid).embedding) == spy.dim
    # 维度不一致的旧向量不会让检索出错
    assert searcher.search(SHOP_A, "发什么快递")[0][0].id == kid


def test_search_indexes_lazily_and_persists_embeddings(repo, faq_ids, spy):
    searcher = FaqSearcher(repo, spy)
    kid = faq_ids[(SHOP_A, "什么时候发货")]
    assert repo.get_knowledge(SHOP_A, kid).embedding is None
    searcher.search(SHOP_A, "什么时候能发货呢")
    assert repo.get_knowledge(SHOP_A, kid).embedding is not None
    # 之后新审核通过的 FAQ，下一次检索就能找到
    cand = faq_ids[(SHOP_A, "可以寄到新疆吗")]
    assert "可以寄到新疆吗" not in _questions(searcher.search(SHOP_A, "可以寄到新疆吗"))
    repo.set_knowledge_status(SHOP_A, cand, KnowledgeStatus.APPROVED, approved_by="merchant")
    hits = searcher.search(SHOP_A, "可以寄到新疆吗")
    assert hits[0][0].id == cand and hits[0][1] == EXACT_MATCH_SCORE


# ---------------------------------------------------------------- 分数语义


@pytest.mark.parametrize(
    "query",
    [
        "什么时候发货",
        "什么时候发货？",
        "亲，什么时候发货呀？",
        "请问什么时候发货呢",
        "在吗？什么时候发货",
    ],
)
def test_normalized_equal_question_scores_exactly_one(repo, spy, query):
    hits = FaqSearcher(repo, spy).search(SHOP_A, query)
    assert hits[0][0].question == "什么时候发货"
    assert hits[0][1] == EXACT_MATCH_SCORE == 1.0


@pytest.mark.parametrize(
    "query",
    [
        "什么时候发货到新疆",  # 包含 FAQ 问题，但问的是另一件事：不加分
        "什么时候能发货呢",
        "你们发什么快递",
        "啥时候能发出来",
    ],
)
def test_paraphrase_is_plain_cosine_and_stays_below_direct_threshold(repo, spy, query):
    hits = FaqSearcher(repo, spy).search(SHOP_A, query)
    assert all(score <= SEMANTIC_SCORE_CAP < EXACT_MATCH_SCORE for _, score in hits)
    assert all(score < DIRECT for _, score in hits)


def test_ranking_is_by_score_desc(repo, spy):
    hits = FaqSearcher(repo, spy).search(SHOP_A, "你们发什么快递", top_k=5)
    assert hits[0][0].question == "发什么快递"
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_min_score_and_top_k(repo, spy):
    searcher = FaqSearcher(repo, spy)
    assert len(searcher.search(SHOP_A, "你们发什么快递", top_k=5)) > 1
    assert _questions(searcher.search(SHOP_A, "你们发什么快递", min_score=0.5)) == ["发什么快递"]
    assert len(searcher.search(SHOP_A, "你们发什么快递", top_k=1)) == 1
    assert searcher.search(SHOP_A, "你们发什么快递", top_k=0) == []
    assert searcher.search(SHOP_A, "   ") == []
    assert searcher.search(SHOP_A, "？？") == []


# ---------------------------------------------------------------- 隔离、状态、商品范围


@pytest.mark.parametrize("query", ["发什么快递", "默认发什么快递呀", "快递", "什么时候发货"])
def test_shop_b_faq_never_returned_for_shop_a(repo, spy, query):
    """店铺 B 与 A 同平台、FAQ 问题相同、答案不同（§9.2 串店测试）。"""
    searcher = FaqSearcher(repo, spy)
    for item_id in (None, "i_555"):
        hits = searcher.search(SHOP_A, query, item_id=item_id, top_k=10)
        assert hits and all(k.shop_id == SHOP_A for k, _ in hits)
        assert not any("圆通" in k.answer for k, _ in hits)
    b_hits = searcher.search(SHOP_B, query, top_k=10)
    assert all(k.shop_id == SHOP_B for k, _ in b_hits)
    assert not any("中通" in k.answer for k, _ in b_hits)


def test_shop_b_exact_match_returns_b_answer(repo, spy):
    hits = FaqSearcher(repo, spy).search(SHOP_B, "发什么快递")
    assert [(k.answer, s) for k, s in hits] == [("您好，默认发圆通快递。", 1.0)]


@pytest.mark.parametrize("query", ["可以寄到新疆吗", "双十一有活动吗"])
def test_candidate_and_retired_faqs_excluded(repo, spy, query):
    hits = FaqSearcher(repo, spy).search(SHOP_A, query, top_k=10)
    assert query not in _questions(hits)
    assert all(k.status == KnowledgeStatus.APPROVED for k, _ in hits)


def test_item_specific_faq_only_for_its_item(repo, spy):
    searcher = FaqSearcher(repo, spy)
    assert "这款会透吗" not in _questions(searcher.search(SHOP_A, "这款会透吗", top_k=10))
    assert "这款会透吗" not in _questions(
        searcher.search(SHOP_A, "这款会透吗", item_id="i_777", top_k=10)
    )
    hits = searcher.search(SHOP_A, "这款会透吗", item_id="i_555")
    assert hits[0][0].question == "这款会透吗" and hits[0][1] == 1.0
    # 同 item_id 的店铺 B 商品看不到 A 的商品专属 FAQ
    assert all(
        k.shop_id == SHOP_B for k, _ in searcher.search(SHOP_B, "这款会透吗", item_id="i_555")
    )


# ---------------------------------------------------------------- 个人信息与降级


def test_query_is_masked_before_it_reaches_the_embedder(repo, spy):
    FaqSearcher(repo, spy).search(
        SHOP_A, "我手机13812345678，住幸福路 18 号 3 栋 201，什么时候发货"
    )
    joined = "\n".join(spy.texts)
    assert "13812345678" not in joined
    assert "幸福路" not in joined and "18" not in joined and "201" not in joined
    assert "[手机号]" in joined and "[地址]" in joined


def test_embedder_failure_falls_back_to_lexical_match(repo, spy):
    spy.fail = True
    searcher = FaqSearcher(repo, spy)
    hits = searcher.search(SHOP_A, "亲，什么时候发货呀")
    assert _questions(hits) == ["什么时候发货"] and hits[0][1] == 1.0
    assert searcher.search(SHOP_A, "你们发哪家快递") == []
    with pytest.raises(RuntimeError):
        searcher.index(SHOP_A)  # 显式索引时不吞掉错误


def test_repository_failure_propagates(flaky, spy):
    flaky.fail_on = {"list_knowledge"}
    with pytest.raises(ConnectionError):
        FaqSearcher(flaky, spy).search(SHOP_A, "什么时候发货")


def test_foreign_rows_from_repository_are_refused(repo, spy):
    """仓储实现有 bug、返回了别家店的 FAQ：抛 TenantIsolationError，不拿来回答（§9.2）。"""

    class Leaky:
        def __getattr__(self, name):
            return getattr(repo, name)

        def list_knowledge(self, shop_id, **kw):
            return repo.list_knowledge(SHOP_B, **kw)

    with pytest.raises(TenantIsolationError):
        FaqSearcher(Leaky(), spy).search(SHOP_A, "发什么快递")
    with pytest.raises(TenantIsolationError):
        FaqSearcher(Leaky(), spy).index(SHOP_A)


def test_normalize_question():
    assert normalize_question("亲，什么时候发货呀？") == "什么时候发货"
    assert normalize_question("请问一下能开发票吗") == "能开发票"
    assert normalize_question("在吗？在吗？发什么快递") == "发什么快递"
    assert normalize_question("亲子装有吗") == "亲子装有"  # "亲"后面没有标点，不是称呼
    assert normalize_question("ＶＩＰ 有优惠吗") == "vip有优惠"
    assert normalize_question("") == ""
