"""模块间接口（端口）。各模块只依赖这里的协议，实现按依赖注入组装（见 docs/架构与分工.md）。

分三组：
1. 外部依赖：大模型、向量、平台工具、平台接入、审计、缓存；
2. 仓储：知识、会话、推荐、反馈、售后审核、授权——内存实现（rodit.inmemory）和
   PostgreSQL 实现（rodit.storage）都要满足，共用 rodit.testing.contracts 里的契约测试；
3. 问答引擎的步骤："懂、找、写、判"分开（§4.1 原则 2），可以分别评估、分别替换。

所有仓储方法都带 shop_id，实现必须按 shop_id 隔离（§9.2），不得跨店返回数据。

留痕分工（§4.1 原则 6、§10）：
- decision_log：stage=route 由 Router 实现写，stage=qa 由 QualityGate 实现写，各写一条聚合记录；
- audit_log：引擎每条消息写各环节事件（core.audit.AuditAction），售后审核助手写审核与操作事件，
  反馈闭环写客服操作事件；
- suggestion：引擎为每张推荐卡片写 SHOWN / SHADOW 记录，
  并在同一会话来新消息时把旧的 SHOWN 改成 EXPIRED。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from rodit.core.enums import (
    AfterSaleAction,
    AfterSaleStatus,
    DecisionStage,
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
    HumanConfirmation,
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
    timeout_s: float | None = None  # 单次调用超时；路由 ≤ 1 秒（§12.1）
    # 成本归属（§12.4 每个店铺的模型成本）。调用方知道就填；不填时计量实现读
    # core.context.current_shop_id()（引擎处理每条消息时设置）。
    shop_id: str | None = None


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    # 接口支持 logprobs 时：输出里第一个有意义的 token（去掉空白、引号、括号、冒号、逗号）的
    # 候选 → 概率（已从对数概率换算成 0–1）；只适合"只输出一个选项标签"的请求。不支持时为 None。
    logprobs: dict[str, float] | None = None


@runtime_checkable
class LLMClient(Protocol):
    """境内已备案大模型（D5），经平台可报备的通道调用。

    失败时抛 core.errors.LLMError / LLMTimeout。测试用 testing.fakes.FakeLLMClient，
    单元测试不依赖网络。提示词里不放买家昵称、手机号、收件地址（调用方负责 mask_pii）。
    """

    def complete(self, request: LLMRequest) -> LLMResponse: ...


@runtime_checkable
class Embedder(Protocol):
    """向量模型。输出 L2 归一化向量，维度 dim（§10 VECTOR(1024) 按所选模型调整）。"""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------- 平台工具（§7，问答只读）


@runtime_checkable
class ToolGateway(Protocol):
    """订单、物流、售后、库存、价格的只读接口（§7.1），统一鉴权和限流。

    - 查不到返回 None / 空列表；限流或故障抛 ToolUnavailable（§12.2）。
    - 只能返回 shop_id 这家店的数据。
    - 返回的数据已做最小化：没有收件人信息；AfterSaleInfo.note、TraceEvent.description
      这类平台自由文本由实现先过 mask_pii；物流状态已归一（§7.2）。
    - 按 buyer_ref 查询时，实现方在平台云内把 buyer_ref 映射回平台买家 ID（或对拉到的记录用
      同一个 BuyerRefHasher 计算后过滤）；映射不出单元。
    - 排序：list_recent_orders 按 created_at 降序（同一时间按 order_id 降序）；
      list_aftersales 按 created_at 升序（同一时间按 aftersale_id 升序）。
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

    def get_stock(self, shop_id: str, item_id: str, sku_id: str | None = None) -> StockInfo | None:
        """sku_id 为空时返回商品级汇总：任一 SKU 有货即有货；任一 SKU 数量未知则数量为 None；
        全部无货时带上补货说明。"""
        ...

    def get_price(self, shop_id: str, item_id: str) -> PriceInfo | None: ...


@runtime_checkable
class AfterSaleActionPort(Protocol):
    """售后写接口（§7.1 同意 / 拒绝）。只有售后审核助手能用，且必须带客服确认（§7.3）。

    - supported 列出该平台支持的操作（拼多多没有查到拒绝接口）；不支持的操作抛 ActionNotAllowed。
    - 实现调用平台前重读售后单：状态不在 core.enums.ACTIONABLE_STATUSES[操作] 里，或
      status_version ≠ confirmation.expected_status_version，就不调用平台，返回
      ok=False、platform_code="STATUS_CHANGED"。
    - 同一 idempotency_key（core.models.aftersale_idempotency_key）重复调用不得重复操作平台，
      返回上次结果并标记 duplicate=True。
    - refuse 的 reason_code：抖店必须传（§7.3.2），缺了抛 ActionNotAllowed。
    - 售后单不存在或不属于该店铺：抛 ActionNotAllowed。
    """

    platform: Platform
    supported: frozenset[AfterSaleAction]

    def agree_refund(
        self,
        shop_id: str,
        aftersale_id: str,
        *,
        confirmation: HumanConfirmation,
        idempotency_key: str,
    ) -> ActionResult: ...

    def agree_return(
        self,
        shop_id: str,
        aftersale_id: str,
        *,
        confirmation: HumanConfirmation,
        idempotency_key: str,
    ) -> ActionResult: ...

    def refuse(
        self,
        shop_id: str,
        aftersale_id: str,
        *,
        confirmation: HumanConfirmation,
        idempotency_key: str,
        reason_code: str | None = None,
        reason_text: str | None = None,
    ) -> ActionResult: ...


# ---------------------------------------------------------------- 平台接入层（§8.2）


@runtime_checkable
class PlatformAdapter(Protocol):
    """平台接入适配器。平台差异只在这里，引擎只认 StandardMessage（§4.1 原则 5）。

    - on_message：模式 A 平台推送 / 模式 B 插件上报 → 标准消息；不是买家消息、重复消息或无法识别时
      返回 None。平台买家 ID 用注入的 BuyerRefHasher 哈希成 buyer_ref，昵称等个人信息不向后传。
    - send：只有模式 A 的全自动能用（text 已由引擎加好 AI 标识前缀）；
      其他接入模式抛 ActionNotAllowed。
    - transfer_to_human：全自动模式下转人工（§5.7、GB/T 47746）；不支持的平台抛 ActionNotAllowed。
    """

    platform: Platform

    def on_message(self, raw: dict) -> StandardMessage | None: ...

    def get_context(self, shop_id: str, conversation_id: str) -> PlatformContext: ...

    def show_suggestions(
        self, shop_id: str, conversation_id: str, cards: list[SuggestionCard]
    ) -> None: ...

    def send(self, shop_id: str, conversation_id: str, text: str) -> SendResult: ...

    def transfer_to_human(self, shop_id: str, conversation_id: str, reason: str) -> SendResult: ...

    def tools(self) -> ToolGateway: ...


# ---------------------------------------------------------------- 审计与缓存


@runtime_checkable
class AuditSink(Protocol):
    """一切留痕（§4.1 原则 6）。detail 里不放买家原文，只放最小化后的内容。"""

    def record(self, event: AuditEvent) -> None: ...

    def log_decision(self, entry: DecisionLogEntry) -> None: ...


@runtime_checkable
class AuditLog(AuditSink, Protocol):
    """可查询的审计（§12.4 质检拦截率、各环节耗时等报表用）。按店铺查询，结果按时间升序。"""

    def list_events(
        self, shop_id: str, *, action: str | None = None, since: datetime | None = None
    ) -> list[AuditEvent]: ...

    def list_decisions(
        self,
        shop_id: str,
        *,
        stage: DecisionStage | None = None,
        since: datetime | None = None,
    ) -> list[DecisionLogEntry]: ...

    def purge_shop(self, shop_id: str) -> int:
        """删除该店铺的审计记录（§8.4 按约定删除数据），返回删除条数。"""
        ...


@runtime_checkable
class Cache(Protocol):
    """短时缓存（平台接口有额度）。按店铺分命名空间（§9.2 缓存按店铺隔离）。"""

    def get(self, shop_id: str, key: str) -> tuple[object, datetime] | None:
        """返回 (值, 写入时间)；不存在返回 None。过期判断由调用方按写入时间做。"""
        ...

    def set(self, shop_id: str, key: str, value: object) -> None: ...

    def purge_shop(self, shop_id: str) -> int:
        """删除该店铺的全部缓存（§8.4），返回删除条数。"""
        ...


# ================================================================ 2. 仓储


@runtime_checkable
class KnowledgeRepository(Protocol):
    """知识中心的存储（§5.2、§10 shop / shop_policy / item_card / knowledge_item）。

    get_shop 返回店铺原样（config_overrides 是店铺自己的覆盖项）；生效配置用
    core.config.effective_shop_config 计算。
    """

    def get_shop(self, shop_id: str) -> Shop | None: ...

    def upsert_shop(self, shop: Shop) -> None: ...

    def get_policies(self, shop_id: str) -> dict[str, PolicyEntry]: ...

    def upsert_policy(self, entry: PolicyEntry) -> None: ...

    def get_item_card(self, shop_id: str, item_id: str) -> ItemCard | None: ...

    def list_item_cards(self, shop_id: str) -> list[ItemCard]:
        """按 item_id 升序。"""
        ...

    def upsert_item_card(self, card: ItemCard) -> None: ...

    def add_knowledge(self, item: KnowledgeItem) -> KnowledgeItem:
        """新增 FAQ（忽略传入的 id），返回带新 id 的记录；created_at 为空时由实现补上。"""
        ...

    def get_knowledge(self, shop_id: str, knowledge_id: int) -> KnowledgeItem | None: ...

    def list_knowledge(
        self,
        shop_id: str,
        *,
        status: KnowledgeStatus | None = KnowledgeStatus.APPROVED,
        item_id: str | None = None,
    ) -> list[KnowledgeItem]:
        """按 id 升序。status=None 表示不过滤状态。
        item_id=None：返回本店全部 FAQ（含商品专属的，商家后台列表用）；
        item_id=X：返回通用 FAQ（item_id 为空）+ 商品 X 的 FAQ（回答时用）。"""
        ...

    def search_knowledge(
        self,
        shop_id: str,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        item_id: str | None = None,
        status: KnowledgeStatus = KnowledgeStatus.APPROVED,
    ) -> list[tuple[KnowledgeItem, float]]:
        """向量检索（§5.2.4 语义检索补充）。只在已有 embedding 的记录里找；
        item_id=None 只找通用 FAQ，item_id=X 找通用 + 商品 X 的。
        分数 = 余弦相似度截断到 [0, 1]，按分数降序（同分按 id 升序）。
        查询向量和已存向量维度不一致时抛 ValueError（换了向量模型要先重建索引）。"""
        ...

    def set_knowledge_status(
        self,
        shop_id: str,
        knowledge_id: int,
        status: KnowledgeStatus,
        *,
        approved_by: str | None = None,
    ) -> KnowledgeItem:
        """不存在（或不属于该店）抛 KeyError。"""
        ...

    def set_knowledge_embedding(
        self, shop_id: str, knowledge_id: int, embedding: list[float]
    ) -> None:
        """不存在（或不属于该店）抛 KeyError。"""
        ...

    def purge_shop(self, shop_id: str) -> int:
        """删除该店铺的全部知识（取消订购 / 授权过期后按约定删除，§8.4），返回删除条数。"""
        ...


@runtime_checkable
class ConversationRepository(Protocol):
    """会话与消息（§10 conversation / message）。按 (shop_id, msg_id) 去重。"""

    def add_message(self, msg: StandardMessage) -> bool:
        """保存买家消息（没有会话时自动建会话）；msg_id 重复时返回 False、不重复保存。
        StoredMessage.created_at = msg.received_at；
        商品卡片、订单卡片存进 item_id / sku_id / order_ref。"""
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
        """保存客服 / 系统发出的消息，用于多轮上下文。msg_id 重复或会话不存在返回 False。"""
        ...

    def get_conversation(self, shop_id: str, conversation_id: str) -> Conversation | None: ...

    def recent_messages(
        self, shop_id: str, conversation_id: str, *, limit: int = 20
    ) -> list[StoredMessage]:
        """最近 limit 条消息，按 created_at 升序（同一时间按写入顺序）。"""
        ...

    def purge_shop(self, shop_id: str) -> int: ...


@runtime_checkable
class SuggestionRepository(Protocol):
    """推荐记录（§10 suggestion）。"""

    def add(self, record: SuggestionRecord) -> SuggestionRecord:
        """保存并返回带新 id 的记录（忽略传入的 id）。"""
        ...

    def get(self, shop_id: str, suggestion_id: int) -> SuggestionRecord | None: ...

    def update(
        self, record: SuggestionRecord, *, expected_status: SuggestionStatus | None = None
    ) -> SuggestionRecord:
        """按 (record.shop_id, record.id) 整条更新；不存在抛 KeyError。
        expected_status 不为空时是比较后更新：当前状态不等于它就抛 ConcurrencyConflict、不写入
        （多实例部署时防止同一条推荐被重复反馈，或客服刚发送的推荐被过期覆盖）。"""
        ...

    def list(
        self,
        shop_id: str,
        *,
        conversation_id: str | None = None,
        status: SuggestionStatus | None = None,
        since: datetime | None = None,
    ) -> list[SuggestionRecord]:
        """按 created_at 升序（同一时间按 id 升序）。"""
        ...

    def purge_shop(self, shop_id: str) -> int: ...


@runtime_checkable
class FeedbackRepository(Protocol):
    """反馈闭环的派生数据（§5.8）：知识缺失提醒、判断层标注集。"""

    def add_gap(self, gap: KnowledgeGap) -> None: ...

    def list_gaps(self, shop_id: str) -> list[KnowledgeGap]:
        """按 created_at 升序。"""
        ...

    def add_label(self, sample: LabeledSample) -> None: ...

    def list_labels(self, shop_id: str) -> list[LabeledSample]:
        """按 created_at 升序。"""
        ...

    def export_training_labels(self) -> list[LabeledSample]:
        """导出本平台单元内所有店铺的标注数据，只用于训练路由分类器（§6.4）。
        不对商家接口开放；数据不出单元。"""
        ...

    def purge_shop(self, shop_id: str) -> int: ...


@runtime_checkable
class AfterSaleReviewRepository(Protocol):
    """售后审核记录（§10 aftersale_review），version 乐观锁。"""

    def get(self, shop_id: str, aftersale_id: str) -> AfterSaleReview | None: ...

    def save(self, review: AfterSaleReview, *, expected_version: int | None) -> AfterSaleReview:
        """expected_version=None 表示新建：已存在则抛 ConcurrencyConflict，写入后 version = 1。
        否则只有当前 version == expected_version 才写入，写入后 version = expected_version + 1；
        不一致或不存在抛 ConcurrencyConflict。传入 review.version 被忽略。返回写入后的记录。"""
        ...

    def list_by_state(
        self, shop_id: str, states: Sequence[ReviewState] | None = None
    ) -> list[AfterSaleReview]:
        """按 aftersale_id 升序。"""
        ...

    def purge_shop(self, shop_id: str) -> int: ...


@runtime_checkable
class ShopAuthRepository(Protocol):
    """店铺授权令牌（§8.4）。令牌加密存储，只在平台云内。"""

    def get(self, shop_id: str) -> ShopAuth | None: ...

    def save(self, auth: ShopAuth) -> None: ...

    def delete(self, shop_id: str) -> None: ...


# ================================================================ 3. 问答引擎的步骤

# 每种路由结果要跑哪些步骤见 core.pipeline.ACTION_PLANS。


@runtime_checkable
class Router(Protocol):
    """路由：判断层 + §6.6 规则 → RouteDecision。

    - 判断层不可用（DecisionError / LLMError）时自己退回关键词规则，RouteDecision.degraded=True；
      规则也失败时返回 NO_SUGGESTION，不向外抛。
    - intent=complaint 按 §5.1 / §5.7 给 HUMAN_PRIORITY（不推荐、提醒人工）；
      degraded 且高风险类型 → NO_SUGGESTION（§12.2）。
    - 写 decision_log（stage=route），一次路由一条。
    """

    def route(self, ctx: ConversationContext) -> RouteDecision: ...


@runtime_checkable
class ToolEvidenceProvider(Protocol):
    """把实时工具数据变成证据（订单、物流、售后、退换资格、库存、价格）。

    只处理 core.evidence.TOOL_NEEDS 里的 need；工具不可用时在 bundle.unavailable 标记，
    需要订单但找不到时在 bundle.missing 标记 MISSING_ORDER，多个订单都可能时给出 clarify_question；
    物流异常等要提醒客服的事放进 bundle.alerts。
    """

    def collect(
        self, ctx: ConversationContext, intent: Intent, needs: frozenset[Need]
    ) -> EvidenceBundle: ...


@runtime_checkable
class EvidenceRetriever(Protocol):
    """找依据：知识中心（结构化优先 + 语义检索补充）+ 工具证据，按优先级解决冲突（§5.2）。

    按 core.evidence.needs_for(route.intent) 决定找什么；知识库故障抛 KnowledgeUnavailable。
    """

    def retrieve(self, ctx: ConversationContext, route: RouteDecision) -> EvidenceBundle: ...


@runtime_checkable
class AnswerGenerator(Protocol):
    """写回答（§5.4）。所有推荐文字都由它产出，包括模板：

    - OFF_PLATFORM_SCRIPT → 平台规则标准话术（§11.4）；SOOTHE → 安抚话术（有依据时加一条回答）；
      CLARIFY → 按 route.top_intents 的澄清问题；evidence.clarify_question → 商品 / 订单澄清；
      工具查不到、又没有其他依据 → "我帮您查一下"（§12.2）；
    - 其余按"FAQ 直答 → 模板 → 大模型"的顺序（§5.4 能不用大模型就不用），大模型只根据证据写。

    返回按推荐顺序排列的候选（最多 ctx.config.suggestion_count 条），可以为空。
    retry_feedback 是上一次质检不通过的原因，重新生成时带上。
    大模型失败时退回 FAQ / 模板（§12.2），都没有就返回空列表。不加全自动的 AI 标识前缀（引擎加）。
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
    """把关（§5.5）：硬规则 + 判断层质检。FAQ / 模板只跑硬规则（§11.2、§11.4）。

    写 decision_log（stage=qa）：跑了判断层质检时一次一条。
    """

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
