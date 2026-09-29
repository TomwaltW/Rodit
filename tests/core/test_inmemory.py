"""内存实现跑仓储契约测试（rodit.testing.contracts），另测缓存和离线向量。"""

from __future__ import annotations

from rodit.inmemory import (
    HashingEmbedder,
    InMemoryAfterSaleReviewRepository,
    InMemoryCache,
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemoryShopAuthRepository,
    InMemorySuggestionRepository,
    cosine,
)
from rodit.testing import contracts
from rodit.testing.fakes import FakeClock


def test_knowledge_repository_contract():
    contracts.check_knowledge_repository(lambda: InMemoryKnowledgeRepository(FakeClock()))


def test_conversation_repository_contract():
    contracts.check_conversation_repository(InMemoryConversationRepository)


def test_suggestion_repository_contract():
    contracts.check_suggestion_repository(InMemorySuggestionRepository)


def test_feedback_repository_contract():
    contracts.check_feedback_repository(InMemoryFeedbackRepository)


def test_aftersale_review_repository_contract():
    contracts.check_aftersale_review_repository(InMemoryAfterSaleReviewRepository)


def test_shop_auth_repository_contract():
    contracts.check_shop_auth_repository(InMemoryShopAuthRepository)


def test_cross_shop_isolation():
    contracts.check_cross_shop_isolation(
        lambda: contracts.RepoSet(
            knowledge=InMemoryKnowledgeRepository(FakeClock()),
            conversations=InMemoryConversationRepository(),
            suggestions=InMemorySuggestionRepository(),
            feedback=InMemoryFeedbackRepository(),
            reviews=InMemoryAfterSaleReviewRepository(),
            auth=InMemoryShopAuthRepository(),
        )
    )


def test_cache_is_namespaced_by_shop_and_records_write_time():
    clock = FakeClock()
    cache = InMemoryCache(clock, max_entries=3)
    cache.set("s_a", "order:o_1", {"v": 1})
    assert cache.get("s_a", "order:o_1") == ({"v": 1}, clock.now())
    assert cache.get("s_b", "order:o_1") is None
    clock.advance(minutes=1)
    cache.set("s_a", "k2", 2)
    cache.set("s_a", "k3", 3)
    cache.set("s_a", "k4", 4)  # 超出上限，淘汰最早写入的
    assert cache.get("s_a", "order:o_1") is None
    assert cache.get("s_a", "k4")[1] == clock.now()
    cache.clear_shop("s_a")
    assert cache.get("s_a", "k4") is None


def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder(dim=128)
    a, b, c = emb.embed(["发什么快递", "发什么快递？", "这款会透吗"])
    assert a == HashingEmbedder(dim=128).embed_one("发什么快递")
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
    assert abs(cosine(a, b) - 1.0) < 1e-9  # 标点不影响
    assert cosine(a, emb.embed_one("发啥快递")) > cosine(a, c)
    assert emb.embed_one("") == [0.0] * 128


# ---------------------------------------------------------------- 撤回先于消息、候选去重、清理


def test_recall_before_message_contract():
    contracts.check_recall_before_message(lambda clock: InMemoryConversationRepository(clock=clock))


def test_candidate_uniqueness_contract():
    contracts.check_candidate_uniqueness(lambda: InMemoryKnowledgeRepository(FakeClock()))


def test_purge_is_idempotent_contract():
    contracts.check_purge_is_idempotent(
        lambda: contracts.RepoSet(
            knowledge=InMemoryKnowledgeRepository(FakeClock()),
            conversations=InMemoryConversationRepository(),
            suggestions=InMemorySuggestionRepository(),
            feedback=InMemoryFeedbackRepository(),
            reviews=InMemoryAfterSaleReviewRepository(),
            auth=InMemoryShopAuthRepository(),
        )
    )
