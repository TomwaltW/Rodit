"""core.ports 的内存实现：仓储、缓存、离线向量。

用于单元测试、本地演示、模式 C 单进程部署；和 PostgreSQL 实现（rodit.storage）语义一致，
共用 rodit.testing.contracts 的契约测试。负责：T0（契约层的一部分，并行开发期间冻结）。
"""

from rodit.inmemory.cache import InMemoryCache
from rodit.inmemory.embedding import HashingEmbedder, cosine
from rodit.inmemory.repositories import (
    InMemoryAfterSaleReviewRepository,
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemoryShopAuthRepository,
    InMemorySuggestionRepository,
)

__all__ = [
    "HashingEmbedder",
    "InMemoryAfterSaleReviewRepository",
    "InMemoryCache",
    "InMemoryConversationRepository",
    "InMemoryFeedbackRepository",
    "InMemoryKnowledgeRepository",
    "InMemoryShopAuthRepository",
    "InMemorySuggestionRepository",
    "cosine",
]
