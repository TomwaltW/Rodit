"""模块间接口（端口）。各模块只依赖这里的协议，实现按依赖注入组装（见 docs/架构与分工.md）。

分三组：
1. 外部依赖：大模型、向量、平台工具、平台接入、审计、缓存；
2. 仓储：知识、会话、推荐、反馈、售后审核、授权——内存实现和 PostgreSQL 实现都要满足；
3. 问答引擎的步骤："懂、找、写、判"分开（§4.1 原则 2），可以分别评估、分别替换。

所有仓储方法都带 shop_id，实现必须按 shop_id 隔离（§9.2），不得跨店返回数据。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from rodit.core.enums import (
    AfterSaleAction,
    AfterSaleStatus,
    Intent,
    KnowledgeStatus,
    Platform,
    ReviewState,
    Role,
    SuggestionStatus,
)
from rodit.core.evidence import Need
from rodit.core.models import (
    ActionResult,
    AfterSaleInfo,
    AfterSaleReview,
    AuditEvent,
    Conversation,
    ConversationContext,
    DecisionLogEntry,
    Draft,
    EvidenceBundle,
    ItemCard,
    KnowledgeGap,
    KnowledgeItem,
    LabeledSample,
    LogisticsAnalysis,
    LogisticsTrace,
    OrderInfo,
    PlatformContext,
    PolicyEntry,
    PriceInfo,
    QAResult,
    RefundProgress,
    ReturnEligibility,
    RouteDecision,
    SendResult,
    Shop,
    ShopAuth,
    StandardMessage,
    StockInfo,
    StoredMessage,
    SuggestionCard,
    SuggestionRecord,
)

# ================================================================ 1. 外部依赖

# ---------------------------------------------------------------- 大模型（藏在接口后面）


class LLMMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: Literal["route", "generate", "qa", "other"]  # 用于成本统计和测试替身分流
    messages: list[LLMMessage]
    temperature: float = 0.0
    max_tokens: int = 512
    json_mode: bool = False  # 要求输出 JSON
    want_logprobs: bool = False  # 接口支持时读取选项概率（§6.4）


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    logprobs: dict[str, float] | None = None  # 选项 → 概率，接口不支持时为 None


@runtime_checkable
class LLMClient(Protocol):
    """境内已备案大模型（D5），经平台可报备的通道调用。

    失败时抛 core.errors.LLMError / LLMTimeout。测试用 testing.fakes.FakeLLMClient，
    单元测试不依赖网络。提示词里不放买家昵称、手机号、收件地址（调用方负责 mask_pii）。
    """

    def complete(self, request: LLMRequest) -> LLMResponse: ...


@runtime_checkable
class Embedder(Protocol):
    """向量模型。维度按所选模型调整（§10 VECTOR(1024)）。"""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------- 平台工具（§7，问答只读）


@runtime_checkable
class ToolGateway(Protocol):
    """订单、物流、售后、库存、价格的只读接口（§7.1），统一鉴权和限流。

    - 查不到返回 None / 空列表；限流或故障抛 ToolUnavailable（§12.2）。
    - 只能返回 shop_id 这家店的数据。
    - 返回的数据已做最小化（没有收件人信息），物流状态已归一。
    """

    platform: Platform

    def get_order(self, shop_id: str, order_id: str) -> OrderInfo | None: ...

    def list_recent_orders(
        self, shop_id: str, buyer_ref: str, *, limit: int = 5
    ) -> list[OrderInfo]: ...

    def get_logistics(self, shop_id: str, order_id: str) -> LogisticsTrace | None: ...

    def get_aftersale(self, shop_id: str, aftersale_id: str) -> AfterSaleInfo | None: ...

    def list_aftersales(
        self,
        shop_id: str,
        *,
        order_id: str | None = None,
        buyer_ref: str | None = None,
        status: AfterSaleStatus | None = None,
        since: datetime | None = None,
    ) -> list[AfterSaleInfo]: ...

    def get_stock(
        self, shop_id: str, item_id: str, sku_id: str | None = None
    ) -> StockInfo | None: ...

    def get_price(self, shop_id: str, item_id: str) -> PriceInfo | None: ...


@runtime_checkable
class AfterSaleActionPort(Protocol):
    """售后写接口（§7.1 同意 / 拒绝）。只有售后审核助手能用，且必须经客服确认（§7.3）。

    - supported 列出该平台支持的操作（拼多多没有查到拒绝接口）。
    - 同一 idempotency_key 重复调用不得重复操作平台，返回 duplicate=True。
    - refuse 的 reason_code：抖店必须传（§7.3.2）。
    """

    platform: Platform
    supported: frozenset[AfterSaleAction]

    def agree_refund(
        self, shop_id: str, aftersale_id: str, *, operator: str, idempotency_key: str
    ) -> ActionResult: ...

    def agree_return(
        self, shop_id: str, aftersale_id: str, *, operator: str, idempotency_key: str
    ) -> ActionResult: ...

    def refuse(
        self,
        shop_id: str,
        aftersale_id: str,
        *,
        operator: str,
        idempotency_key: str,
        reason_code: str | None = None,
        reason_text: str | None = None,
    ) -> ActionResult: ...


# ---------------------------------------------------------------- 平台接入层（§8.2）


@runtime_checkable
class PlatformAdapter(Protocol):
    """平台接入适配器。平台差异只在这里，引擎只认 StandardMessage（§4.1 原则 5）。

    - on_message：模式 A 平台推送 / 模式 B 插件上报 → 标准消息；不是买家消息或无法识别时返回 None。
      平台买家 ID 在这里哈希成 buyer_ref，昵称等个人信息不向后传。
    - send：只有模式 A 的全自动能用；智能辅助模式下调用应抛 ActionNotAllowed。
    """

    platform: Platform

    def on_message(self, raw: dict) -> StandardMessage | None: ...

    def get_context(self, shop_id: str, conversation_id: str) -> PlatformContext: ...

    def show_suggestions(
        self, shop_id: str, conversation_id: str, cards: list[SuggestionCard]
    ) -> None: ...

    def send(self, shop_id: str, conversation_id: str, text: str) -> SendResult: ...

    def tools(self) -> ToolGateway: ...


# ---------------------------------------------------------------- 审计与缓存


@runtime_checkable
class AuditSink(Protocol):
    """一切留痕（§4.1 原则 6）。detail 里不放买家原文，只放最小化后的内容。"""

    def record(self, event: AuditEvent) -> None: ...

    def log_decision(self, entry: DecisionLogEntry) -> None: ...


@runtime_checkable
class Cache(Protocol):
    """短时缓存（平台接口有额度）。键必须包含 shop_id（§9.2 缓存按店铺隔离）。"""

    def get(self, key: str) -> tuple[object, datetime] | None:
        """返回 (值, 写入时间)；不存在返回 None。过期判断由调用方按写入时间做。"""
        ...

    def set(self, key: str, value: object) -> None: ...


# ================================================================ 2. 仓储


@runtime_checkable
class KnowledgeRepository(Protocol):
    """知识中心的存储（§5.2、§10 shop / shop_policy / item_card / knowledge_item）。"""

    def get_shop(self, shop_id: str) -> Shop | None: ...

    def upsert_shop(self, shop: Shop) -> None: ...

    def get_policies(self, shop_id: str) -> dict[str, PolicyEntry]: ...

    def upsert_policy(self, entry: PolicyEntry) -> None: ...

    def get_item_card(self, shop_id: str, item_id: str) -> ItemCard | None: ...

    def list_item_cards(self, shop_id: str) -> list[ItemCard]: ...

    def upsert_item_card(self, card: ItemCard) -> None: ...

    def add_knowledge(self, item: KnowledgeItem) -> KnowledgeItem:
        """新增 FAQ，返回带 id 的记录。"""
        ...

    def get_knowledge(self, shop_id: str, knowledge_id: int) -> KnowledgeItem | None: ...

    def list_knowledge(
        self,
        shop_id: str,
        *,
        status: KnowledgeStatus | None = KnowledgeStatus.APPROVED,
        item_id: str | None = None,
    ) -> list[KnowledgeItem]:
        """status=None 表示不过滤；item_id 非空时返回通用 FAQ + 该商品的 FAQ。"""
        ...

    def set_knowledge_status(
        self,
        shop_id: str,
        knowledge_id: int,
        status: KnowledgeStatus,
        *,
        approved_by: str | None = None,
    ) -> KnowledgeItem: ...

    def set_knowledge_embedding(
        self, shop_id: str, knowledge_id: int, embedding: list[float]
    ) -> None: ...


@runtime_checkable
class ConversationRepository(Protocol):
    """会话与消息（§10 conversation / message）。按 (shop_id, msg_id) 去重。"""

    def add_message(self, msg: StandardMessage) -> bool:
        """保存买家消息（没有会话时自动建会话）；msg_id 重复时返回 False、不重复保存。"""
        ...

    def add_outgoing(
        self,
        *,
        shop_id: str,
        conversation_id: str,
        msg_id: str,
        role: Role,
        text: str,
        at: datetime,
    ) -> bool:
        """保存客服 / 系统发出的消息，用于多轮上下文。"""
        ...

    def get_conversation(self, shop_id: str, conversation_id: str) -> Conversation | None: ...

    def recent_messages(
        self, shop_id: str, conversation_id: str, *, limit: int = 20
    ) -> list[StoredMessage]:
        """最近 limit 条消息，时间升序。"""
        ...


@runtime_checkable
class SuggestionRepository(Protocol):
    """推荐记录（§10 suggestion）。"""

    def add(self, record: SuggestionRecord) -> SuggestionRecord:
        """保存并返回带 id 的记录。"""
        ...

    def get(self, shop_id: str, suggestion_id: int) -> SuggestionRecord | None: ...

    def update(self, record: SuggestionRecord) -> SuggestionRecord: ...

    def list(
        self,
        shop_id: str,
        *,
        conversation_id: str | None = None,
        status: SuggestionStatus | None = None,
        since: datetime | None = None,
    ) -> list[SuggestionRecord]: ...


@runtime_checkable
class FeedbackRepository(Protocol):
    """反馈闭环的派生数据（§5.8）：知识缺失提醒、判断层标注集。"""

    def add_gap(self, gap: KnowledgeGap) -> None: ...

    def list_gaps(self, shop_id: str) -> list[KnowledgeGap]: ...

    def add_label(self, sample: LabeledSample) -> None: ...

    def list_labels(self, shop_id: str | None = None) -> list[LabeledSample]:
        """shop_id=None 时返回所有店铺（仅用于同一平台单元内训练分类器）。"""
        ...


@runtime_checkable
class AfterSaleReviewRepository(Protocol):
    """售后审核记录（§10 aftersale_review），version 乐观锁。"""

    def get(self, shop_id: str, aftersale_id: str) -> AfterSaleReview | None: ...

    def save(self, review: AfterSaleReview, *, expected_version: int | None) -> AfterSaleReview:
        """expected_version=None 表示新建（已存在则抛 ConcurrencyConflict）；
        否则只有当前 version == expected_version 才写入，写入后 version + 1，
        不一致抛 ConcurrencyConflict。返回写入后的记录。"""
        ...

    def list_by_state(
        self, shop_id: str, states: Sequence[ReviewState] | None = None
    ) -> list[AfterSaleReview]: ...


@runtime_checkable
class ShopAuthRepository(Protocol):
    """店铺授权令牌（§8.4）。令牌加密存储，只在平台云内。"""

    def get(self, shop_id: str) -> ShopAuth | None: ...

    def save(self, auth: ShopAuth) -> None: ...

    def delete(self, shop_id: str) -> None: ...


# ================================================================ 3. 问答引擎的步骤


@runtime_checkable
class Router(Protocol):
    """路由：判断层 + §6.6 规则 → RouteDecision。判断层不可用时自己降级到关键词规则。"""

    def route(self, ctx: ConversationContext) -> RouteDecision: ...


@runtime_checkable
class ToolEvidenceProvider(Protocol):
    """把实时工具数据变成证据（订单、物流、售后、退换资格、库存、价格）。

    只处理 core.evidence.TOOL_NEEDS 里的 need；工具不可用时在 bundle.unavailable 标记，
    需要订单但找不到时在 bundle.missing 标记 MISSING_ORDER 并给出 clarify_question。
    """

    def collect(
        self, ctx: ConversationContext, intent: Intent, needs: frozenset[Need]
    ) -> EvidenceBundle: ...


@runtime_checkable
class EvidenceRetriever(Protocol):
    """找依据：知识中心（结构化优先 + 语义检索补充）+ 工具证据，按优先级解决冲突（§5.2）。"""

    def retrieve(self, ctx: ConversationContext, route: RouteDecision) -> EvidenceBundle: ...


@runtime_checkable
class AnswerGenerator(Protocol):
    """写回答（§5.4）：能用 FAQ / 模板就不用大模型；大模型只根据证据写。

    返回按推荐顺序排列的候选（最多 ctx.shop.config.suggestion_count 条），可以为空。
    retry_feedback 是上一次质检不通过的原因，重新生成时带上。
    大模型失败时退回 FAQ / 模板（§12.2），都没有就返回空列表。
    """

    def generate(
        self,
        ctx: ConversationContext,
        route: RouteDecision,
        evidence: EvidenceBundle,
        *,
        retry_feedback: list[str] | None = None,
    ) -> list[Draft]: ...


@runtime_checkable
class QualityGate(Protocol):
    """把关（§5.5）：硬规则 + 判断层质检。FAQ / 模板只跑硬规则（§11.2、§11.4）。"""

    def check(
        self,
        ctx: ConversationContext,
        route: RouteDecision,
        draft: Draft,
        evidence: EvidenceBundle,
    ) -> QAResult: ...


@runtime_checkable
class LogisticsAdvisor(Protocol):
    """物流问答规则与异常识别（§7.2）。"""

    def analyze(self, trace: LogisticsTrace, now: datetime) -> LogisticsAnalysis: ...


@runtime_checkable
class AfterSaleAdvisor(Protocol):
    """售后买家侧问答的依据（§7.3.1）。"""

    def return_eligibility(
        self, order: OrderInfo, policies: dict[str, PolicyEntry], now: datetime
    ) -> ReturnEligibility: ...

    def refund_progress(self, aftersale: AfterSaleInfo, now: datetime) -> RefundProgress: ...
