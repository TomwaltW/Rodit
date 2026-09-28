"""生产部署的组装（方案 §9.1 每个平台云一个单元、§4.4 技术选型、§12.3 安全与合规）。

部署在对应平台云（聚石塔 / 抖店云 / 多多云）内，数据不出云、不出境：
- 仓储：PostgreSQL（平台云的 RDS），行级安全按店铺隔离（§9.2）；
- 大模型：境内已备案模型（D5），OpenAI 兼容接口，按店铺计量用量（§12.4）；
- buyer_ref 的盐、令牌加密密钥、数据库连接串、大模型密钥都只从环境变量读，仓库里没有真实值；
- 平台只读工具（订单 / 物流 / 售后接口）、接入适配器、售后写接口必须在平台云内实现并注入：
  沙箱里调不通开放平台接口（CLAUDE.md：平台接口一律 mock），本仓库只有模拟实现。

环境变量：RODIT_DATABASE_DSN、RODIT_TOKEN_KEY、RODIT_BUYER_SALT、
RODIT_LLM_BASE_URL、RODIT_LLM_MODEL、RODIT_LLM_API_KEY、RODIT_LLM_TIMEOUT_S（可选）。
"""

from __future__ import annotations

from collections.abc import Iterable

from rodit.app.unit import Repos, Unit, build_unit
from rodit.core.clock import Clock, SystemClock
from rodit.core.config import EngineConfig
from rodit.core.enums import Platform
from rodit.core.pii import BuyerRefHasher
from rodit.core.ports import AfterSaleActionPort, PlatformAdapter, ToolGateway
from rodit.llm import MeteredLLMClient, OpenAICompatibleClient
from rodit.storage import (
    Database,
    PgAfterSaleReviewRepository,
    PgAuditLog,
    PgConversationRepository,
    PgFeedbackRepository,
    PgKnowledgeRepository,
    PgShopAuthRepository,
    PgSuggestionRepository,
)


def postgres_repos(db: Database, *, clock: Clock, token_env: str = "RODIT_TOKEN_KEY") -> Repos:
    """一个单元的 PostgreSQL 仓储。令牌加密密钥从环境变量读（§8.4 令牌加密存储）。"""
    return Repos(
        knowledge=PgKnowledgeRepository(db, clock=clock),
        conversations=PgConversationRepository(db),
        suggestions=PgSuggestionRepository(db),
        feedback=PgFeedbackRepository(db),
        reviews=PgAfterSaleReviewRepository(db),
        auth=PgShopAuthRepository.from_env(db, env_name=token_env),
        audit=PgAuditLog(db),
    )


def migrate(dsn_env: str = "RODIT_DATABASE_DSN") -> list[int]:
    """部署步骤：以数据库属主身份建表、建角色、开行级安全（幂等）。返回本次应用的版本号。"""
    return Database.from_env(dsn_env).migrate()


def build_production_unit(
    platform: Platform,
    *,
    config: EngineConfig,
    tools: ToolGateway,
    adapters: Iterable[PlatformAdapter],
    action_port: AfterSaleActionPort | None = None,
    clock: Clock | None = None,
) -> Unit:
    """生产单元：tools / adapters / action_port 由平台云里的实现注入（见模块说明）。

    MVP 只申请读权限（§8.4 最小权限），action_port 通常为空：售后审核只给建议。
    """
    clock = clock or SystemClock()
    db = Database.from_env()
    repos = postgres_repos(db, clock=clock)
    if config.llm.base_url and config.llm.model:
        client = OpenAICompatibleClient(
            base_url=config.llm.base_url,
            model=config.llm.model,
            api_key_env=config.llm.api_key_env,
            timeout_s=config.llm.timeout_s,
        )
    else:
        client = OpenAICompatibleClient.from_env(api_key_env=config.llm.api_key_env)
    llm = MeteredLLMClient(client, repos.audit, clock)
    return build_unit(
        platform=platform,
        config=config,
        clock=clock,
        hasher=BuyerRefHasher.from_env(config.privacy.buyer_salt_env),
        repos=repos,
        llm=llm,
        tools=tools,
        adapters=adapters,
        action_port=action_port,
    )
