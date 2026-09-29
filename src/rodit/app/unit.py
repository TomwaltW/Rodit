"""组装一个平台单元（方案 §9.1 单元化部署：一套代码，每个平台云各部署一份）。

单元里有：仓储（内存或 PostgreSQL）、平台工具（带缓存）、判断层与路由、知识检索、生成、质检、
问答引擎、反馈闭环、售后审核助手、接入适配器、店铺授权。各模块之间只通过 core.ports 的协议相连，
这里是唯一把具体实现接在一起的地方（docs/架构与分工.md §1）。

build_unit 只接收"已经造好"的外部依赖（仓储、大模型、平台工具、适配器、写接口），本身不读环境变量；
本地演示见 rodit.app.demo，生产部署见 rodit.app.production。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta

from rodit.adapters import AdapterRegistry, AuthGuard
from rodit.aftersale import DefaultAfterSaleAdvisor, ReviewAssistant
from rodit.core.audit import InMemoryAuditSink
from rodit.core.clock import Clock
from rodit.core.config import EngineConfig, effective_shop_config
from rodit.core.decision import DecisionProvider
from rodit.core.enums import Platform
from rodit.core.errors import ShopInactive
from rodit.core.models import KnowledgeItem, ShopConfig
from rodit.core.pii import BuyerRefHasher
from rodit.core.ports import (
    AfterSaleActionPort,
    AfterSaleReviewRepository,
    AuditLog,
    Cache,
    ConversationRepository,
    Embedder,
    FeedbackRepository,
    KnowledgeRepository,
    LLMClient,
    PlatformAdapter,
    ShopAuthRepository,
    SuggestionRepository,
    ToolGateway,
)
from rodit.decision import RuleProvider, build_decision_provider
from rodit.engine import AssistEngine
from rodit.feedback import FeedbackService
from rodit.generation import DefaultAnswerGenerator
from rodit.inmemory import (
    HashingEmbedder,
    InMemoryAfterSaleReviewRepository,
    InMemoryCache,
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemoryShopAuthRepository,
    InMemorySuggestionRepository,
)
from rodit.knowledge import FaqSearcher, KnowledgeRetriever
from rodit.qa import DefaultQualityGate
from rodit.routing import DecisionRouter, RoutingPolicy
from rodit.tools import CachingToolGateway, DefaultLogisticsAdvisor, DefaultToolEvidenceProvider

TOOL_CACHE_MAX_STALE_S = 3600
"""平台接口故障时最多用 1 小时内的缓存兜底（§12.2 用缓存数据并注明时间）；更旧的订单、售后、
物流数据宁可推荐"我帮您查一下"，也不拿来回答。"""


@dataclass
class Repos:
    """一个单元的全部仓储（同一个平台云内的一套库，§9.1）。"""

    knowledge: KnowledgeRepository
    conversations: ConversationRepository
    suggestions: SuggestionRepository
    feedback: FeedbackRepository
    reviews: AfterSaleReviewRepository
    auth: ShopAuthRepository
    audit: AuditLog

    def purgers(self) -> list[object]:
        """§8.4 取消订购后要按约定删除数据的仓储（都有 purge_shop）。令牌由 AuthGuard 删除。"""
        return [
            self.knowledge,
            self.conversations,
            self.suggestions,
            self.feedback,
            self.reviews,
            self.audit,
        ]


def in_memory_repos(clock: Clock) -> Repos:
    return Repos(
        knowledge=InMemoryKnowledgeRepository(clock),
        conversations=InMemoryConversationRepository(clock=clock),
        suggestions=InMemorySuggestionRepository(),
        feedback=InMemoryFeedbackRepository(),
        reviews=InMemoryAfterSaleReviewRepository(),
        auth=InMemoryShopAuthRepository(),
        audit=InMemoryAuditSink(),
    )


@dataclass
class Unit:
    """组装好的平台单元。字段都是可以直接调用的组件，测试和接口层按需取用。"""

    platform: Platform
    config: EngineConfig
    clock: Clock
    hasher: BuyerRefHasher
    repos: Repos
    llm: LLMClient
    embedder: Embedder
    cache: Cache
    tools: ToolGateway  # 平台原始只读工具
    gateway: CachingToolGateway  # 带缓存和降级的工具（问答链路用）
    searcher: FaqSearcher
    retriever: KnowledgeRetriever
    router: DecisionRouter
    generator: DefaultAnswerGenerator
    gate: DefaultQualityGate
    engine: AssistEngine
    feedback: FeedbackService
    reviews: ReviewAssistant
    adapters: AdapterRegistry
    auth: AuthGuard
    action_port: AfterSaleActionPort | None = None
    extra_purgers: list[object] = field(default_factory=list)

    def shop_config(self, shop_id: str) -> ShopConfig:
        """店铺的生效配置（平台默认 → 类目默认 → 店铺覆盖，§9.3）。"""
        shop = self.repos.knowledge.get_shop(shop_id)
        if shop is None:
            raise ShopInactive(shop_id)
        return effective_shop_config(self.config, shop)


def build_unit(
    *,
    platform: Platform,
    config: EngineConfig,
    clock: Clock,
    hasher: BuyerRefHasher,
    repos: Repos,
    llm: LLMClient,
    tools: ToolGateway,
    adapters: Iterable[PlatformAdapter],
    action_port: AfterSaleActionPort | None = None,
    embedder: Embedder | None = None,
    cache: Cache | None = None,
    route_provider: DecisionProvider | None = None,
    qa_provider: DecisionProvider | None = None,
) -> Unit:
    """把各模块接成一个平台单元。

    - route_provider / qa_provider 为空时按 config.decision 组装
      （rodit.decision.build_decision_provider）；
      路由的兜底永远是关键词规则（§12.2 判断层不可用 → 关键词路由）。
    - action_port 为空表示没有申请写售后权限（MVP，§8.4 最小权限）：审核助手只给建议。
    """
    if tools.platform != platform:
        raise ValueError(f"平台工具属于 {tools.platform}，单元是 {platform}（§9.1 数据不跨云）")
    cache = cache or InMemoryCache(clock)
    embedder = embedder or HashingEmbedder()
    ttl_s = config.logistics.cache_ttl_s
    gateway = CachingToolGateway(
        tools, cache, clock, ttl_s=ttl_s, max_stale_s=max(ttl_s, TOOL_CACHE_MAX_STALE_S)
    )
    searcher = FaqSearcher(repos.knowledge, embedder)
    tool_evidence = DefaultToolEvidenceProvider(
        gateway,
        repos.knowledge.get_policies,
        DefaultLogisticsAdvisor(config.logistics),
        DefaultAfterSaleAdvisor(config.aftersale),
        clock,
        stale_after=timedelta(seconds=ttl_s),  # 比 TTL 还旧的订单 / 售后单只能来自过期缓存
    )
    retriever = KnowledgeRetriever(
        repos.knowledge, searcher, config.generation, tool_provider=tool_evidence
    )
    router = DecisionRouter(
        route_provider or build_decision_provider(config.decision, purpose="route", llm=llm),
        RuleProvider(),
        RoutingPolicy(config.routing),
        repos.audit,
        clock,
        history_rounds=config.generation.history_rounds,
    )
    gate = DefaultQualityGate(
        qa_provider or build_decision_provider(config.decision, purpose="qa", llm=llm),
        config.qa_gate,
        repos.audit,
        clock,
    )
    generator = DefaultAnswerGenerator(llm, config.generation)
    engine = AssistEngine(
        config=config,
        knowledge=repos.knowledge,
        conversations=repos.conversations,
        suggestions=repos.suggestions,
        tools=gateway,
        router=router,
        retriever=retriever,
        generator=generator,
        gate=gate,
        audit=repos.audit,
        clock=clock,
        gaps=repos.feedback,
    )

    def index_approved(item: KnowledgeItem) -> None:
        searcher.index(item.shop_id)  # 商家审核通过的新 FAQ 立刻可检索（§5.8）

    feedback = FeedbackService(
        repos.suggestions,
        repos.knowledge,
        repos.feedback,
        repos.audit,
        clock,
        on_approved=index_approved,
    )

    def shop_config(shop_id: str) -> ShopConfig:
        shop = repos.knowledge.get_shop(shop_id)
        if shop is None:
            raise ShopInactive(shop_id)
        return effective_shop_config(config, shop)

    reviews = ReviewAssistant(
        gateway,
        repos.reviews,
        repos.audit,
        clock,
        config.aftersale,
        shop_config=shop_config,
        action_port=action_port,
    )
    registry = AdapterRegistry()
    for adapter in adapters:
        if adapter.platform != platform:
            raise ValueError(f"适配器属于 {adapter.platform}，单元是 {platform}")
        registry.register(adapter)
    return Unit(
        platform=platform,
        config=config,
        clock=clock,
        hasher=hasher,
        repos=repos,
        llm=llm,
        embedder=embedder,
        cache=cache,
        tools=tools,
        gateway=gateway,
        searcher=searcher,
        retriever=retriever,
        router=router,
        generator=generator,
        gate=gate,
        engine=engine,
        feedback=feedback,
        reviews=reviews,
        adapters=registry,
        auth=AuthGuard(repos.auth, clock),
        action_port=action_port,
    )
