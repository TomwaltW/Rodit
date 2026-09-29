"""PostgreSQL 存储（方案 §9.2、§10）：建表脚本、行级安全、各仓储协议的实现。

负责：T7。只依赖 rodit.core。
"""

from rodit.storage.db import (
    APP_ROLE,
    BUSINESS_TABLES,
    MIGRATIONS,
    TRAINER_ROLE,
    Database,
    Migration,
    StorageUnavailable,
)
from rodit.storage.repositories import (
    PgAfterSaleReviewRepository,
    PgAuditLog,
    PgConversationRepository,
    PgFeedbackRepository,
    PgKnowledgeRepository,
    PgShopAuthRepository,
    PgSuggestionRepository,
    TokenDecryptionError,
)

__all__ = [
    "APP_ROLE",
    "BUSINESS_TABLES",
    "MIGRATIONS",
    "TRAINER_ROLE",
    "Database",
    "Migration",
    "PgAfterSaleReviewRepository",
    "PgAuditLog",
    "PgConversationRepository",
    "PgFeedbackRepository",
    "PgKnowledgeRepository",
    "PgShopAuthRepository",
    "PgSuggestionRepository",
    "StorageUnavailable",
    "TokenDecryptionError",
]
