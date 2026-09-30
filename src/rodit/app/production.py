"""生产部署的组装（方案 §9.1 每个平台云一个单元、§4.4 技术选型、§12.3 安全与合规）。

部署在对应平台云（聚石塔 / 抖店云 / 多多云）内，数据不出云、不出境：
- 仓储：PostgreSQL（平台云的 RDS），行级安全按店铺隔离（§9.2）；
- 大模型：境内已备案模型（D5），OpenAI 兼容接口，按店铺计量用量（§12.4）；
- buyer_ref 的盐、令牌加密密钥、数据库连接串、大模型密钥都只从环境变量读，仓库里没有真实值；
- 平台只读工具（订单 / 物流 / 售后接口）、接入适配器、售后写接口必须在平台云内实现并注入：
  沙箱里调不通开放平台接口（CLAUDE.md：平台接口一律 mock），本仓库只有模拟实现。

环境变量：RODIT_DATABASE_DSN、RODIT_TOKEN_KEY、RODIT_BUYER_SALT、
RODIT_LLM_BASE_URL、RODIT_LLM_MODEL、RODIT_LLM_API_KEY、RODIT_LLM_TIMEOUT_S（可选）、
RODIT_DB_STATEMENT_TIMEOUT_MS / RODIT_DB_LOCK_TIMEOUT_MS（可选，数据库语句 / 等锁超时，毫秒）。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import replace

from rodit.app.unit import Repos, Unit, build_unit
from rodit.core.clock import Clock, SystemClock
from rodit.core.config import EngineConfig
from rodit.core.decision import DecisionProvider
from rodit.core.enums import Platform
from rodit.core.pii import BuyerRefHasher
from rodit.core.ports import (
    AfterSaleActionPort,
    AuditLog,
    ImageTextExtractor,
    ItemCatalog,
    PlatformAdapter,
    ToolGateway,
)
from rodit.llm import MeteredLLMClient, OpenAICompatibleClient
from rodit.storage import (
    Database,
    PgAfterSaleReviewRepository,
    PgAuditLog,
    PgConversationRepository,
    PgFeedbackRepository,
    PgKnowledgeRepository,
    PgQualityReviewRepository,
    PgShopAuthRepository,
    PgSuggestionRepository,
)

DEFAULT_STATEMENT_TIMEOUT_MS = 2000
"""请求用连接的语句超时。检索的目标是 ≤ 0.5 秒（§12.1），这里是兜底上限：远低于端到端 5 秒，
超时就按知识库 / 存储故障降级（§12.2），不让请求线程一直挂着。"""
DEFAULT_LOCK_TIMEOUT_MS = 1000
"""请求用连接的等锁上限：被清理店铺数据、迁移等长事务挡住时尽快报错降级。"""


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(f"环境变量 {name} 必须是正整数（毫秒），现在是 {raw!r}")
    return value


def production_database(dsn_env: str = "RODIT_DATABASE_DSN") -> Database:
    """请求用的数据库：连接串、语句超时、等锁超时都从环境变量读（后两者有默认值）。

    取消订购时的清理（§8.4，AuthGuard.revoke 调各仓储的 purge_shop）也用这个库，但开不限时的
    事务（Database.session 的 unbounded）：大店铺一次删几百万行，2 秒的请求上限下永远删不完。"""
    return Database.from_env(
        dsn_env,
        statement_timeout_ms=_positive_int_env(
            "RODIT_DB_STATEMENT_TIMEOUT_MS", DEFAULT_STATEMENT_TIMEOUT_MS
        ),
        lock_timeout_ms=_positive_int_env("RODIT_DB_LOCK_TIMEOUT_MS", DEFAULT_LOCK_TIMEOUT_MS),
    )


def postgres_repos(db: Database, *, clock: Clock, token_env: str = "RODIT_TOKEN_KEY") -> Repos:
    """一个单元的 PostgreSQL 仓储。令牌加密密钥从环境变量读（§8.4 令牌加密存储）。"""
    return Repos(
        knowledge=PgKnowledgeRepository(db, clock=clock),
        conversations=PgConversationRepository(db, clock=clock),
        suggestions=PgSuggestionRepository(db),
        feedback=PgFeedbackRepository(db),
        reviews=PgAfterSaleReviewRepository(db),
        auth=PgShopAuthRepository.from_env(db, env_name=token_env),
        audit=PgAuditLog(db),
        quality_reviews=PgQualityReviewRepository(db),
    )


def migrate(dsn_env: str = "RODIT_DATABASE_DSN") -> list[int]:
    """部署步骤：以数据库属主身份建表、建角色、开行级安全（幂等）。返回本次应用的版本号。
    迁移可能比请求慢得多，不设请求级的语句 / 等锁超时。"""
    return Database.from_env(dsn_env).migrate()


def build_production_unit(
    platform: Platform,
    *,
    config: EngineConfig,
    tools: ToolGateway,
    adapters: Iterable[PlatformAdapter],
    action_port: AfterSaleActionPort | None = None,
    clock: Clock | None = None,
    audit_wrapper: Callable[[AuditLog], AuditLog] | None = None,
    catalog: ItemCatalog | None = None,
    ocr: ImageTextExtractor | None = None,
    route_provider: DecisionProvider | None = None,
) -> Unit:
    """生产单元：tools / adapters / action_port 由平台云里的实现注入（见模块说明）。

    MVP 只申请读权限（§8.4 最小权限），action_port 通常为空：售后审核只给建议。
    audit_wrapper：给审计套一层（部署入口用它采集运行指标，rodit.ops.instrument），
    单元里所有写审计的地方（引擎、判断层留痕、大模型计量）都经过套好的那一层。
    route_provider：路由用的判断层，为空时按 config.decision 组装；decision.provider 是
    classifier 时由部署入口从模型文件加载好传进来（§6.4 Phase 2）。质检仍按 config.decision
    组装（classifier / rules 时质检用大模型，§6.4 "质检用硬规则 + 大模型"）。
    """
    clock = clock or SystemClock()
    db = production_database()
    repos = postgres_repos(db, clock=clock)
    if audit_wrapper is not None:
        repos = replace(repos, audit=audit_wrapper(repos.audit))
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
        catalog=catalog,
        ocr=ocr,
        route_provider=route_provider,
    )
