"""模块间接口（端口）。各模块只依赖这里的协议，实现按依赖注入组装（见 docs/架构与分工.md）。

分三组：
1. 外部依赖：大模型、向量、平台工具、平台接入、审计、缓存；
2. 仓储：知识、会话、推荐、反馈、售后审核、授权——内存实现（rodit.inmemory）和
   PostgreSQL 实现（rodit.storage）都要满足，共用 rodit.testing.contracts 里的契约测试；
3. 问答引擎的步骤："懂、找、写、判"分开（§4.1 原则 2），可以分别评估、分别替换。

所有仓储方法都带 shop_id，实现必须按 shop_id 隔离（§9.2），不得跨店返回数据。
例外只有三个，都在文档里写明了能返回什么：FeedbackRepository.export_training_labels（训练导出）、
ShopAuthRepository.expiring_shop_ids（只有店铺 ID 和到期时间，没有令牌）、
AuditLog.list_system_events（只有没有店铺归属的系统事件）。

留痕分工（§4.1 原则 6、§10）：
- decision_log：stage=route 由 Router 实现写，stage=qa 由 QualityGate 实现写，各写一条聚合记录；
- audit_log：引擎每条消息写各环节事件（core.audit.AuditAction），售后审核助手写审核与操作事件，
  反馈闭环写客服操作事件；
- suggestion：引擎为每张推荐卡片写 SHOWN / SHADOW 记录，
  并在同一会话来新消息时把旧的 SHOWN 改成 EXPIRED。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, SecretStr

from rodit.core.enums import (
    AfterSaleAction,
    AfterSaleStatus,
    DecisionStage,
    Intent,
    KnowledgeStatus,
    Platform,
    PlatformMetric,
    QualityReviewStatus,
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
    ImageFinding,
    ImageRef,
    ItemCard,
    KnowledgeGap,
    KnowledgeItem,
    LabeledSample,
    LogisticsAnalysis,
    LogisticsTrace,
    OrderInfo,
    PlatformContext,
    PlatformItem,
    PlatformMetricSample,
    PolicyEntry,
    PriceInfo,
    QAResult,
    QualityReview,
    QualityScores,
    RefreshedToken,
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
from rodit.core.scopes import MVP_SCOPES

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
    # 接口返回的 finish_reason（stop / length / ……）；接口没给时为 None。
    # "length" 表示输出被 max_tokens 截断了。
    finish_reason: str | None = None


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


# ---------------------------------------------------------------- 店铺授权（§8.4）


@runtime_checkable
class TokenRefreshPort(Protocol):
    """平台的令牌续期接口（§8.4）。开放平台接口在平台云内调用：生产实现由平台云代码注入
    （PlatformBindings.token_refresher），测试用 rodit.testing.fakes.FakeTokenRefresher。

    - 续期令牌无效、过期或被商家撤销：抛 core.errors.AuthExpired（需要商家重新授权）；
    - 限流、故障、超时：抛 core.errors.ToolUnavailable（下次再试）。
    令牌进出都是 SecretStr；实现不得把令牌写进日志或异常消息。
    """

    platform: Platform

    def refresh(self, shop_id: str, refresh_token: SecretStr) -> RefreshedToken: ...


@runtime_checkable
class TokenProvider(Protocol):
    """给平台云代码（平台只读工具、适配器）取店铺的有效令牌（§8.4）。

    - access_token：授权有效（没过期、权限够、没被标记需要重新授权）时返回令牌，否则抛
      core.errors.AuthExpired，调用方停止调用平台接口；
    - report_rejected：平台拒绝了这个令牌（如返回令牌失效）：标记需要重新授权，之后
      access_token 抛 AuthExpired，授权告警里能看到。token 传被拒的那个令牌（access_token 给出的
      SecretStr）：只在它还是这家店现在的令牌时标记——续期、重新授权换了新令牌之后，拿着旧令牌的
      在途调用被拒不影响新授权。不传时标记现在的授权（分不出被拒的是不是旧令牌，旧的调用方式）。
    默认实现是 rodit.adapters.auth.AuthGuardTokenProvider（基于 AuthGuard）。
    """

    def access_token(self, shop_id: str, scopes: Iterable[str] = MVP_SCOPES) -> SecretStr: ...

    def report_rejected(self, shop_id: str, token: SecretStr | None = None) -> None: ...


# ---------------------------------------------------------------- 平台接入层（§8.2）


@runtime_checkable
class ItemCatalog(Protocol):
    """平台商品接口（只读，§5.2.3 商品自动同步）：拉商品标题、属性、SKU、详情文字和详情图引用。

    开放平台接口要在平台云内调用：生产实现由平台云注入，沙箱和测试用
    rodit.testing.fakes.InMemoryItemCatalog。接口故障抛 core.errors.ToolUnavailable。
    只返回 shop_id 这家店铺的商品（§9.2）。
    """

    platform: Platform

    def list_items(self, shop_id: str) -> list[PlatformItem]:
        """该店铺的全部商品（含已下架，on_sale=False），按 item_id 升序。"""
        ...

    def get_item(self, shop_id: str, item_id: str) -> PlatformItem | None: ...


@runtime_checkable
class ImageTextExtractor(Protocol):
    """详情图 OCR（§5.2.1 "详情图 OCR 文字"）。在平台云内运行，图片不出云。

    image_ref 是 PlatformItem.detail_images 里的引用。失败抛 core.errors.ToolUnavailable
    （同步照常进行，只是这张图的文字缺失）。
    """

    def extract_text(self, shop_id: str, image_ref: str) -> str: ...


@runtime_checkable
class ImageUnderstanding(Protocol):
    """买家图片 / 售后凭证的理解（§15 Phase 3 "图片理解：破损照片、尺码截图"）。

    平台云内的境内视觉模型（和 D5 同样的备案与报备要求），图片不出云。生产实现由平台云注入，
    沙箱和测试用 rodit.testing.fakes.FakeImageUnderstanding。
    - 返回的 summary / extracted_text 必须已过 core.pii.mask_pii（面单上的手机号、地址）；
    - 识别失败、超时、限流抛 core.errors.ToolUnavailable（调用方按 §12.2 降级：当作没有图片理解，
      照常给推荐，提醒客服自己看图）；
    - 只看 shop_id 这家店铺的图片（§9.2）。
    """

    def analyze(self, shop_id: str, image: ImageRef) -> ImageFinding: ...


@runtime_checkable
class PlatformMetricsSource(Protocol):
    """平台客服考核数据的只读接口（§12.4 平台客服指标；三个平台的数据接口名【待确认】）。

    平台云内实现；沙箱和测试用 rodit.testing.fakes.InMemoryPlatformMetricsSource。
    只返回本平台、shop_id 这家店铺、[since, until] 日期范围内（含两端）的数据，
    source 一律是 MetricSource.PLATFORM。故障抛 core.errors.ToolUnavailable。
    """

    platform: Platform

    def fetch(self, shop_id: str, since: date, until: date) -> list[PlatformMetricSample]: ...


class MessageRecall(BaseModel):
    """PlatformAdapter.on_event 的撤回事件：会话里的一条消息被撤回（§5.3 撤回的消息不再算作
    最近几轮对话，原文也不再保留）。"""

    model_config = ConfigDict(extra="forbid")

    shop_id: str
    conversation_id: str
    msg_id: str  # 撤回事件本身的平台消息 ID（撤回是幂等的，平台重推同一条撤回照常交给应用层）
    recalled_msg_id: str  # 被撤回的那条消息


@runtime_checkable
class PlatformAdapter(Protocol):
    """平台接入适配器。平台差异只在这里，引擎只认 StandardMessage（§4.1 原则 5）。

    - on_message：模式 A 平台推送 / 模式 B 插件上报 → 标准消息；不是买家消息、重复消息或无法识别时
      返回 None。平台买家 ID 用注入的 BuyerRefHasher 哈希成 buyer_ref，昵称等个人信息不向后传。
    - on_event：on_message 的完整版，应用层用它：买家消息同 on_message；客服 / 机器人发出的消息
      回声 → StoredMessage（role 为 AGENT / BOT，created_at 取事件时间），撤回 → MessageRecall；
      应用层据此把客服手打的回复写进会话、把撤回的消息从会话里去掉（§5.3 最近几轮对话）。
      其他事件、重复的买家消息（按 msg_id 去重）、无法识别时返回 None；回声和撤回不去重
      （幂等：应用层写库失败时平台重推能再应用一次，仓储按 msg_id 去重）。
    - send：只有模式 A 的全自动能用（text 已由引擎加好 AI 标识前缀）；
      其他接入模式抛 ActionNotAllowed。
    - transfer_to_human：全自动模式下转人工（§5.7、GB/T 47746）；不支持的平台抛 ActionNotAllowed。
    - show_suggestions：把一批推荐推到侧边栏 / 工作台，替换上一批（空列表表示清空）。
      alerts 是标红提醒客服的话（EngineResult.alerts，如物流停滞、买家要求人工）；
      notice 是没有推荐时的原因（EngineResult.no_suggestion_reason），客服知道为什么是空的。
      两者都不含买家原文；新的一批连同提醒一起替换旧的（不传即清空旧的提醒）。
    """

    platform: Platform

    def peek_shop_id(self, raw: dict) -> str | None:
        """不改变任何状态地解析出事件所属店铺（不去重、不记上下文）；无法识别返回 None。

        应用层在 on_message 之前用它核对店铺和授权，
        避免一个店铺的事件改动另一个店铺的状态（§9.2）。
        """
        ...

    def on_message(self, raw: dict) -> StandardMessage | None: ...

    def on_event(self, raw: dict) -> StandardMessage | StoredMessage | MessageRecall | None: ...

    def get_context(self, shop_id: str, conversation_id: str) -> PlatformContext: ...

    def show_suggestions(
        self,
        shop_id: str,
        conversation_id: str,
        cards: list[SuggestionCard],
        *,
        alerts: Sequence[str] = (),
        notice: str | None = None,
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

    def list_system_events(
        self, *, action: str | None = None, since: datetime | None = None
    ) -> list[AuditEvent]:
        """没有店铺归属的系统事件（shop_id 为空，如 shop.purged、eval.verdict），按时间升序。
        只返回 shop_id 为空的行，不会返回任何店铺的事件（PostgreSQL 实现在不设店铺范围的事务里读，
        行级安全只放行 shop_id 为空的行）。系统事件的 detail 只放 ID、计数、原因，不放买家数据。"""
        ...

    def purge_shop(self, shop_id: str) -> int:
        """删除该店铺的审计记录（§8.4 按约定删除数据），返回删除条数。
        系统事件（shop_id 为空）不删：取消订购的清理记录（shop.purged）要在清理之后留下来。"""
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

    def update_shop_config(self, shop_id: str, changes: Mapping[str, Any]) -> Shop | None:
        """原子地改店铺配置覆盖项（§9.3）：changes 里值为 None 的键删掉（回到类目 / 平台默认），
        其余写入；合并后按 Shop 校验，未知的键、非法的取值抛 ValueError，什么都不写。
        店铺不存在返回 None（不新建）。读改写在一把锁 / 一个事务里：并发改不同的键不会互相覆盖
        （商家改配置、复核人员切换影子模式、填问卷可能同时发生）。返回更新后的店铺。"""
        ...

    def get_policies(self, shop_id: str) -> dict[str, PolicyEntry]: ...

    def upsert_policy(self, entry: PolicyEntry) -> None: ...

    def delete_policy(self, shop_id: str, key: str) -> bool:
        """删掉一项店铺政策（商家清空已经不适用的政策，如赠品活动结束）。删了返回 True，
        本来就没有返回 False。"""
        ...

    def get_item_card(self, shop_id: str, item_id: str) -> ItemCard | None: ...

    def list_item_cards(self, shop_id: str) -> list[ItemCard]:
        """按 item_id 升序。"""
        ...

    def upsert_item_card(self, card: ItemCard) -> None: ...

    def update_item_card(
        self, shop_id: str, item_id: str, changes: Mapping[str, Any]
    ) -> ItemCard | None:
        """原子地改商品卡片的部分字段（商家改自己填的字段、商品同步改平台维护的字段，§5.2.3）：
        changes 的键必须是 ItemCard 的字段，shop_id / item_id / platform 不能改；值按字段校验
        （None 清空可选字段）。不合法抛 ValueError，什么都不写。卡片不存在返回 None（不新建）。
        读改写在一把锁 / 一个事务里：商家改尺码建议和同步改标题同时发生，两边的修改都留下。"""
        ...

    def add_knowledge(self, item: KnowledgeItem) -> KnowledgeItem:
        """新增 FAQ（忽略传入的 id），返回带新 id 的记录；created_at 为空时由实现补上。

        等审核的候选知识不重复（§5.8）：新增的是 CANDIDATE，而本店已有问题、回答、商品、问题类型
        都相同的 CANDIDATE 时抛 ConcurrencyConflict、不写入（多个实例同时把同一条编辑回复变成候选，
        只能建出一条；由存储保证，调用方读到的"没有"可能已经过时）。"""
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
        expected_status: KnowledgeStatus | None = None,
    ) -> KnowledgeItem:
        """不存在（或不属于该店）抛 KeyError。
        expected_status 不为空时是比较后更新：当前状态不等于它就抛 ConcurrencyConflict、不写入
        （多实例部署时防止商家审核通过和驳回同时发生，后写的覆盖先写的）。
        改成 CANDIDATE 时和已有的候选重复（见 add_knowledge）也抛 ConcurrencyConflict、不写入。"""
        ...

    def set_knowledge_embedding(
        self, shop_id: str, knowledge_id: int, embedding: list[float]
    ) -> None:
        """不存在（或不属于该店）抛 KeyError。"""
        ...

    def purge_shop(self, shop_id: str) -> int:
        """删除该店铺的全部知识（取消订购 / 授权过期后按约定删除，§8.4），返回删除条数。"""
        ...


RECALL_TOMBSTONE_TTL = timedelta(days=3)
"""撤回先于消息到达时墓碑的默认有效期（ConversationRepository.recall_message）：平台推送乱序、
重推都在这个时间之内【待确认，按平台消息服务的积压 / 重推上限调整】；过期的墓碑不再生效。"""


@runtime_checkable
class ConversationRepository(Protocol):
    """会话与消息（§10 conversation / message）。按 (shop_id, msg_id) 去重。"""

    def add_message(self, msg: StandardMessage) -> bool:
        """保存买家消息（没有会话时自动建会话）；msg_id 重复时返回 False、不重复保存。
        StoredMessage.created_at = msg.received_at；
        商品卡片、订单卡片存进 item_id / sku_id / order_ref。
        这条消息先被撤回过（见 recall_message 的墓碑）：不保存它的文字和卡片、返回 False
        （引擎当作重复消息，不回答），会话照常建立，之后按撤回过的消息去重。"""
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
        """保存客服 / 系统发出的消息，用于多轮上下文。msg_id 重复或会话不存在返回 False。
        这条消息先被撤回过（会话存在时）：同 add_message，不保存、返回 False。"""
        ...

    def get_conversation(self, shop_id: str, conversation_id: str) -> Conversation | None: ...

    def recent_messages(
        self, shop_id: str, conversation_id: str, *, limit: int = 20
    ) -> list[StoredMessage]:
        """最近 limit 条消息，按 created_at 升序（同一时间按写入顺序）；不含已撤回的消息。"""
        ...

    def recall_message(self, shop_id: str, conversation_id: str, msg_id: str) -> bool:
        """消息被撤回（§5.3）：删掉它的文字和卡片，之后 recent_messages 不再返回它；msg_id 仍算
        见过（平台重推这条消息时 add_message / add_outgoing 返回 False）。撤回了返回 True。

        找不到这条消息、或已经撤回过时返回 False。本店从没见过这个 msg_id（平台推送乱序，
        撤回先到，§8.3）：按 (shop_id, conversation_id, msg_id) 记一条墓碑，有效期
        （实现的构造参数，默认 RECALL_TOMBSTONE_TTL，按注入的时钟判断）内这条消息到达时丢弃
        （见 add_message / add_outgoing）；墓碑用掉或过期即删，数量有界。多个实例同时处理
        撤回和这条消息时，结果和先后处理一样（消息不会留下来）。"""
        ...

    def list_outgoing(
        self,
        shop_id: str,
        *,
        since: datetime,
        until: datetime,
        roles: Sequence[Role] = (Role.AGENT, Role.BOT),
    ) -> list[StoredMessage]:
        """本店在 since ≤ created_at < until 之间发出的消息（默认客服和机器人，跨会话），
        按 created_at 升序（同一时间按写入顺序）；不含已撤回的消息。上线门槛的合规扫描用
        （§13.4 实际发出去的文字里有没有违规用语）。只返回 shop_id 这家店的消息。"""
        ...

    def purge_shop(self, shop_id: str) -> int:
        """删除该店铺的会话、消息和撤回墓碑（§8.4），返回删除条数；再调一次返回 0。"""
        ...


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

    def export_training_labels(
        self, *, since: datetime | None = None, until: datetime | None = None
    ) -> list[LabeledSample]:
        """导出本平台单元内所有店铺的标注数据，只用于训练路由分类器（§6.4）和每月重算阈值
        （§13.5）；since ≤ created_at < until，按 created_at 升序。
        不对商家接口开放；数据不出单元。"""
        ...

    def purge_shop(self, shop_id: str) -> int: ...


@runtime_checkable
class QualityReviewRepository(Protocol):
    """每周人工复核样本（§13.5）。同一店铺、同一周期（ISO 周）、同一推荐只有一条。"""

    def add_many(self, reviews: Sequence[QualityReview]) -> list[QualityReview]:
        """写入抽中的样本（status 必须是 pending，id 为 None）。幂等：同一店铺、周期、推荐已有的，
        不重复写、原样返回已有的那条（不管它现在是什么状态）。返回和输入一一对应、都带 id。
        reviews 可以跨店铺（每条按自己的 shop_id 写入）。"""
        ...

    def get(self, shop_id: str, review_id: int) -> QualityReview | None: ...

    def list(
        self,
        shop_id: str,
        *,
        period: str | None = None,
        status: QualityReviewStatus | None = None,
    ) -> list[QualityReview]:
        """按 id 升序。"""
        ...

    def record(
        self,
        shop_id: str,
        review_id: int,
        *,
        status: QualityReviewStatus,
        scores: QualityScores | None,
        reviewer: str,
        note: str | None,
        reviewed_at: datetime,
    ) -> QualityReview:
        """记下复核结果（scored 带 scores；skipped 不带）。可以改判：后一次覆盖前一次
        （历史在审计里）。status 不能是 pending、结果不合法抛 ValueError；样本不存在
        （含别的店铺的样本）抛 KeyError。返回更新后的样本。"""
        ...

    def purge_shop(self, shop_id: str) -> int: ...


@runtime_checkable
class PlatformMetricsRepository(Protocol):
    """平台客服考核指标（§12.4、§15 Phase 2 商家报表：平台指标接入前后对比）。

    按店铺隔离（§9.2）。"""

    def upsert(self, sample: PlatformMetricSample) -> PlatformMetricSample:
        """按（店铺、指标、日期、来源）写入：已有就覆盖
        value、sample_size、recorded_by、recorded_at。"""
        ...

    def list(
        self,
        shop_id: str,
        *,
        metric: PlatformMetric | None = None,
        since: date | None = None,
        until: date | None = None,
    ) -> list[PlatformMetricSample]:
        """该店铺的数据，since / until 是日期（含两端），按（日期、指标、来源）升序。"""
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


class GrantStamp(NamedTuple):
    """一家店现在这份授权的版本和开通事件时间（ShopAuth.version、granted_at）：不含令牌，
    读的时候不解密（ShopAuthRepository.stamp）。"""

    version: int
    granted_at: datetime | None


@runtime_checkable
class ShopAuthRepository(Protocol):
    """店铺授权令牌（§8.4）。令牌（access_token、refresh_token）加密存储，只在平台云内。

    每次写入（save / replace）分配新的版本（ShopAuth.version，全单元不重复），标记需要重新授权
    不换版本。续期任务（另一个进程）和平台推送（服务进程）会同时改同一家店的授权，续期结果、
    被拒的标记都按读到的版本比较并交换：期间授权被换掉（重新授权、续订）或删掉（取消订购）时
    什么都不改。"""

    def get(self, shop_id: str) -> ShopAuth | None: ...

    def save(self, auth: ShopAuth) -> None:
        """整条写入（新建或覆盖），包括续期令牌、有效期、needs_reauth 和 granted_at；
        分配新的版本（忽略 auth.version）。"""
        ...

    def replace(self, auth: ShopAuth, *, expected_version: int) -> bool:
        """只改现有的这一份授权：存的版本还是 expected_version 时整条覆盖（同 save，分配新的版本），
        返回 True；授权已被换掉（版本不同）或已删除返回 False，什么都不写——不新建，取消订购删掉的
        授权不会被写回来。续期任务保存新令牌用。"""
        ...

    def stamp(self, shop_id: str) -> GrantStamp | None:
        """这家店授权的版本和开通事件时间；没有授权返回 None。不解密（令牌读不出来的也能读）。"""
        ...

    def delete(self, shop_id: str) -> bool:
        """删除令牌：删掉了返回 True，本来就没有返回 False。
        不解密：令牌读不出来（旧版加密、换过密钥）时也能删（取消订购，§8.4）。"""
        ...

    def expiring_shop_ids(self, before: datetime) -> list[tuple[str, datetime]]:
        """access_token 在 before 之前（expires_at ≤ before，含已过期的）到期的店铺：
        [(shop_id, expires_at)]，按 expires_at 升序（同一时间按 shop_id）。续期任务、过期清理、
        按店铺汇总的报表用（给一个很远的时间就是本单元全部有授权的店铺）。

        这是唯一的跨店读取授权的路径：只返回店铺 ID 和到期时间，从不返回令牌，也不解密
        （令牌读不出来的店铺照样列出来）。PostgreSQL 实现经 SECURITY DEFINER 函数
        rodit_auth_expiring 读（行级安全不允许跨店读 shop_auth）。"""
        ...

    def mark_needs_reauth(self, shop_id: str, *, expected_version: int | None = None) -> bool:
        """标记这家店需要重新授权（续期被拒、平台拒绝了当前令牌）：之后 AuthGuard.require
        不通过。标记了返回 True（已经标记过也是 True），没有这家店的授权返回 False。不解密，
        不换版本（续期成功照样能按原来的版本写回新令牌、清掉标记）。

        expected_version：被拒的是哪一份授权（读到的 ShopAuth.version）。给了时只在存的还是这一份
        时标记（比较并交换）；已被换掉（续期、重新授权）或已删除返回 False，什么都不改。"""
        ...


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
