"""标准数据模型（方案 §5.2.2、§5.3、§5.5、§5.6、§7、§8.3、§10）。

约定：
- 所有时间都是带时区的 datetime（AwareDatetime，默认东八区，见 core.clock.CST）。
- 金额用 Decimal。
- 模型里没有买家昵称、手机号、收件地址字段（§5.3 最小化个人信息）：
  订单只保留状态、商品、金额、时间等回答需要的字段；买家只用 buyer_ref（平台买家 ID 的哈希）。
- Evidence.data 里放结构化原始数据（model_dump(mode="json")），content 放给大模型看的文字。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.core.decision import DecisionResponse
from rodit.core.enums import (
    AfterSaleAction,
    AfterSaleReason,
    AfterSaleStatus,
    AfterSaleType,
    AnomalyKind,
    Confidence,
    DecisionStage,
    DraftKind,
    EvidenceSource,
    Intent,
    KnowledgeSource,
    KnowledgeStatus,
    LogisticsStatus,
    Mode,
    OrderStatus,
    Platform,
    RejectReason,
    ReviewAdvice,
    ReviewState,
    Role,
    RouteAction,
    ShopStatus,
    SignedDetail,
    SuggestionStatus,
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- 标准会话（§8.3）


class ItemRef(_Model):
    """会话里的商品卡片：买家从哪个商品进来，或发来的商品链接。"""

    item_id: str
    sku_id: str | None = None


class OrderRef(_Model):
    """会话里的订单卡片：买家发来的订单。"""

    order_id: str


class StandardMessage(_Model):
    """标准会话消息（§8.3）。引擎只认这个格式，平台差异在接入层。

    - buyer_ref 是平台买家 ID 的哈希（core.pii.hash_buyer_id），不存昵称。
    - agent_id 是当前接待的客服子账号。
    - 同一平台内按 msg_id 去重，按 conversation_id 保证顺序。
    """

    platform: Platform
    shop_id: str
    conversation_id: str
    msg_id: str
    buyer_ref: str
    text: str
    item_card: ItemRef | None = None
    order_card: OrderRef | None = None
    agent_id: str | None = None
    mode: Mode = Mode.ASSIST
    received_at: AwareDatetime
    role: Role = Role.BUYER
    extension: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- 店铺与配置（§9.3、§10）


class ShopConfig(_Model):
    """店铺可配置项（§9.3）。阈值由我们统一校准，不在这里。"""

    display_name: str = ""  # 店铺名，用于提示词"你是{店铺名}的客服"
    tone: Literal["亲切", "专业", "简洁"] = "亲切"
    salutation: Literal["亲", "您"] = "亲"
    max_answer_chars: int = Field(default=120, ge=20, le=500)
    banned_words: list[str] = Field(default_factory=list)  # 附录 C 第 14 题
    disabled_intents: list[Intent] = Field(default_factory=list)  # 关闭的问题类型
    suggestion_count: int = Field(default=2, ge=1, le=3)  # 推荐条数 1–3（§5.4）
    aftersale_amount_limit: Decimal = Decimal("500")  # R6：超过必须店长确认
    auto_reply_enabled: bool = False  # 全自动总开关（后期，且需平台准入）


class Shop(_Model):
    id: str
    platform: Platform
    category: str | None = None  # 主营类目，如 "女装"
    status: ShopStatus = ShopStatus.ACTIVE
    config: ShopConfig = Field(default_factory=ShopConfig)


class ShopAuth(_Model):
    """店铺授权（§8.4）。access_token 只存放在对应平台云内，落库时加密。"""

    shop_id: str
    access_token: str
    scopes: list[str]
    expires_at: AwareDatetime


# ---------------------------------------------------------------- 知识（§5.2）


class PolicyEntry(_Model):
    shop_id: str
    key: str  # 通常是 PolicyKey 的取值
    value: str
    updated_at: AwareDatetime


class Sku(BaseModel):
    """SKU。规格属性（颜色、尺码……）直接作为额外字段，和 §5.2.2 的 JSON 一致。"""

    model_config = ConfigDict(extra="allow")

    sku_id: str
    stock_source: str = "realtime"

    def props(self) -> dict[str, str]:
        return {k: str(v) for k, v in (self.model_extra or {}).items()}


class ItemCard(_Model):
    """商品知识卡片（§5.2.2）。价格和库存不写死在卡片里，回答时实时查。"""

    shop_id: str
    platform: Platform
    item_id: str
    title: str
    category: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    skus: list[Sku] = Field(default_factory=list)
    size_chart: list[dict[str, str | int | float]] = Field(default_factory=list)
    size_advice: str | None = None  # 商家填写，尺码推荐最重要的依据
    care: list[str] = Field(default_factory=list)
    usage: list[str] = Field(default_factory=list)  # 使用 / 安装说明
    forbidden_claims: list[str] = Field(default_factory=list)  # 不能说的卖点，质检用
    detail_text: str | None = None  # 详情页文字 + 详情图 OCR 文字
    updated_at: AwareDatetime

    def find_sku(self, sku_id: str) -> Sku | None:
        return next((s for s in self.skus if s.sku_id == sku_id), None)


class KnowledgeItem(_Model):
    """FAQ 与优质回答（§10 knowledge_item）。"""

    id: int | None = None
    shop_id: str
    question: str
    answer: str
    source: KnowledgeSource
    status: KnowledgeStatus
    item_id: str | None = None  # 只适用于某个商品的 FAQ
    intent: Intent | None = None
    approved_by: str | None = None
    created_at: AwareDatetime | None = None
    embedding: list[float] | None = None


class KnowledgeGap(_Model):
    """知识缺失提醒（§5.8 "知识缺失 → 提醒商家补知识"，§5.7 "提示缺哪类知识"）。"""

    shop_id: str
    intent: Intent | None = None
    missing: str  # 如 "policy:invoice"、"item:i_555:size_advice"
    question_text: str | None = None  # 已做个人信息最小化
    source: Literal["engine", "feedback"] = "engine"
    created_at: AwareDatetime


class LabeledSample(_Model):
    """判断层标注数据（§5.8 "理解错误 → 判断层标注集"，§6.4 训练小模型分类器）。"""

    shop_id: str
    text: str  # 已做个人信息最小化
    predicted_intent: Intent | None = None
    true_intent: Intent | None = None
    source: Literal["feedback_reject", "feedback_adopt", "manual"] = "manual"
    created_at: AwareDatetime


# ---------------------------------------------------------------- 平台工具数据（§7，只读）


class OrderLine(_Model):
    item_id: str
    sku_id: str | None = None
    title: str
    quantity: int = 1
    price: Decimal


class OrderInfo(_Model):
    """订单（最小化字段，§5.3）：没有收件人姓名、电话、地址。"""

    shop_id: str
    order_id: str
    buyer_ref: str
    status: OrderStatus
    lines: list[OrderLine]
    total_amount: Decimal
    created_at: AwareDatetime
    paid_at: AwareDatetime | None = None
    shipped_at: AwareDatetime | None = None
    signed_at: AwareDatetime | None = None
    waybill_no: str | None = None
    courier: str | None = None
    seven_day_return: bool = True  # 商品是否支持七天无理由（定制商品等例外）
    promised_ship_by: AwareDatetime | None = None  # 平台/店铺承诺的最晚发货时间


class StockInfo(_Model):
    shop_id: str
    item_id: str
    sku_id: str | None = None
    quantity: int | None = None  # 平台不给具体数量时为 None
    in_stock: bool
    restock_note: str | None = None
    fetched_at: AwareDatetime


class PriceInfo(_Model):
    """价格与活动配置（price_promo 的依据）。"""

    shop_id: str
    item_id: str
    price: Decimal
    promotions: list[str] = Field(default_factory=list)  # 如 "店铺满 199 减 20"
    fetched_at: AwareDatetime


class TraceEvent(_Model):
    time: AwareDatetime
    description: str
    location: str | None = None
    raw_status: str | None = None  # 平台原始状态码，便于排查映射


class LogisticsTrace(_Model):
    """物流轨迹（已归一，§7.2）。events 按时间升序。"""

    shop_id: str
    order_id: str
    waybill_no: str
    courier: str
    status: LogisticsStatus
    signed_detail: SignedDetail | None = None
    pickup_info: str | None = None  # 驿站 / 快递柜名称，不含取件码
    events: list[TraceEvent] = Field(default_factory=list)
    estimated_delivery: AwareDatetime | None = None  # 只有平台接口直接给出时才有
    fetched_at: AwareDatetime

    @property
    def last_event(self) -> TraceEvent | None:
        return self.events[-1] if self.events else None


class LogisticsAnomaly(_Model):
    kind: AnomalyKind
    detail: str
    agent_alert: str | None = None  # 提醒客服的话，如"建议向快递催查"
    buyer_hint: str | None = None  # 推荐给买家的话的要点，如取件指引


class LogisticsAnalysis(_Model):
    """物流问答用的分析结果（§7.2 回答规则 + 异常识别）。"""

    status: LogisticsStatus
    as_of: AwareDatetime  # 回答里写"截至 {as_of}"
    summary: str  # 如"运输中，截至 09-28 10:30 已到达杭州转运中心"
    anomalies: list[LogisticsAnomaly] = Field(default_factory=list)
    pickup_guidance: str | None = None  # 驿站 / 快递柜取件指引
    eta: AwareDatetime | None = None  # 只有接口给出预计送达时间时才有，否则不承诺到货时间


class AfterSaleInfo(_Model):
    """售后单（平台数据，已归一）。"""

    shop_id: str
    aftersale_id: str
    order_id: str
    buyer_ref: str
    type: AfterSaleType
    reason: AfterSaleReason
    status: AfterSaleStatus
    amount: Decimal
    created_at: AwareDatetime
    updated_at: AwareDatetime
    evidence_count: int = 0  # 买家上传的凭证（照片）数量
    status_version: str = "0"  # 平台状态版本（如修改时间戳），用于幂等键和重读校验
    note: str | None = None  # 买家填写的说明（已做个人信息最小化）


class ReturnEligibility(_Model):
    """能不能退（§7.3.1）：店铺政策 + 订单状态（是否签收、签收几天、是否支持七天无理由）。"""

    order_id: str
    eligible: bool | None  # None 表示依据不足，需要人工判断
    days_since_signed: int | None = None
    reason: str  # 给大模型和客服看的解释


class RefundProgress(_Model):
    """退款进度（§7.3.1）：当前进度 + 平台的一般到账规则。"""

    aftersale_id: str
    status: AfterSaleStatus
    summary: str
    as_of: AwareDatetime


class ActionResult(_Model):
    """售后写操作的平台返回。"""

    ok: bool
    action: AfterSaleAction
    aftersale_id: str
    idempotency_key: str
    platform_code: str | None = None
    message: str | None = None
    duplicate: bool = False  # 幂等键重复，没有重复调用平台


# ---------------------------------------------------------------- 会话理解（§5.3）


class Turn(_Model):
    """一轮对话：同一角色的连发消息合并后的一条。"""

    role: Role
    text: str
    at: AwareDatetime
    msg_ids: list[str] = Field(default_factory=list)


class ConversationContext(_Model):
    """问答引擎各步骤共享的会话上下文（由 engine 的会话理解构建）。

    - text 是原文；任何要进提示词的文本都必须先过 core.pii.mask_pii。
    - item / order 是指代理解之后的"当前商品 / 当前订单"。
    - candidate_item_ids 有多个且 item 为空时，说明商品不确定，需要澄清（§5.2.4）。
    """

    platform: Platform
    shop: Shop
    conversation_id: str
    buyer_ref: str
    mode: Mode = Mode.ASSIST
    agent_id: str | None = None
    current: Turn  # 当前（合并后的）买家消息
    history: list[Turn] = Field(default_factory=list)  # 之前的对话，时间升序，最多 5 轮
    item: ItemCard | None = None
    sku_id: str | None = None
    candidate_item_ids: list[str] = Field(default_factory=list)
    order: OrderInfo | None = None
    recent_orders: list[OrderInfo] = Field(default_factory=list)  # 该买家在本店的最近订单
    unresolved_turns: int = 0  # 买家连续表示没听懂 / 答非所问的轮数（§5.7）
    now: AwareDatetime

    @property
    def shop_id(self) -> str:
        return self.shop.id


# ---------------------------------------------------------------- 证据（§5.2.4、§5.4）


class Evidence(_Model):
    """一条依据。id 按 core.evidence 里的规则生成，回答的 evidence_ids 引用它。"""

    id: str
    source: EvidenceSource
    label: str  # 展示给客服的来源，如"商品属性 · 面料""店铺政策 · 发货时效"
    content: str  # 给大模型看的文字（已最小化）
    data: dict[str, Any] = Field(default_factory=dict)
    topic: str | None = None  # 同一事实主题（如 "ship_time"），冲突时高优先级覆盖低优先级
    score: float | None = None  # 语义检索相似度
    fetched_at: AwareDatetime | None = None  # 实时数据的获取时间，回答里写"截至"

    @property
    def priority(self) -> int:
        return self.source.priority


class EvidenceBundle(_Model):
    """证据包：检索 + 工具调用的结果。"""

    evidences: list[Evidence] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)  # 缺的知识，如 "policy:invoice"
    unavailable: list[str] = Field(default_factory=list)  # 暂时查不到的工具，如 "tool:logistics"
    overridden_ids: list[str] = Field(default_factory=list)  # 被高优先级覆盖掉的证据
    clarify_question: str | None = None  # 商品 / 订单不确定时的澄清问题

    def ids(self) -> list[str]:
        return [e.id for e in self.evidences]

    def get(self, evidence_id: str) -> Evidence | None:
        return next((e for e in self.evidences if e.id == evidence_id), None)

    @property
    def is_empty(self) -> bool:
        return not self.evidences


# ---------------------------------------------------------------- 路由（§6.6）


class RouteDecision(_Model):
    action: RouteAction
    rule: int = Field(ge=1, le=8)  # 命中的 §6.6 规则序号
    intent: Intent
    intent_confidence: float
    top_intents: list[tuple[Intent, float]] = Field(default_factory=list)  # 前两名，澄清用
    need_human: float = 0.0
    emotion: float = 0.0
    multi_intent: float = 0.0
    high_risk: bool = False
    auto_eligible: bool = False  # 规则 6 且达到该类型的自动阈值、该类型已开放自动
    risk_hints: list[str] = Field(default_factory=list)  # 如"涉及价格承诺，请确认"
    alerts: list[str] = Field(default_factory=list)  # 给客服的提醒，如"买家要求人工，请优先处理"
    degraded: bool = False  # 判断层不可用，用关键词规则路由（§12.2）
    provider: str
    answers: DecisionResponse | None = None


# ---------------------------------------------------------------- 生成与质检（§5.4、§5.5）


class Draft(_Model):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    kind: DraftKind
    need_clarify: bool = False
    variant: Literal["direct", "with_followup", "clarify", "soothe", "script"] = "direct"
    template_id: str | None = None
    faq_id: int | None = None


class HardRuleHit(_Model):
    rule: Literal[
        "off_platform",
        "absolute_term",
        "banned_word",
        "forbidden_claim",
        "number_mismatch",
        "time_promise",
        "length",
        "format",
        "unknown_evidence",
    ]
    detail: str
    span: str | None = None


class QAResult(_Model):
    passed: bool
    text: str  # 清洗后的文本（去 Markdown、超长截断）
    hard_hits: list[HardRuleHit] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)  # grounded / promise_risk / tone ...
    reasons: list[str] = Field(default_factory=list)  # 不通过的原因，展示给客服 / 带回重新生成
    model_checked: bool = False  # 是否跑了判断层质检（FAQ / 模板只跑硬规则）
    degraded: bool = False  # 判断层质检不可用


# ---------------------------------------------------------------- 推荐与反馈（§5.6、§5.8、§10）


class EvidenceRefView(_Model):
    id: str
    label: str


class SuggestionCard(_Model):
    """智能辅助的推荐卡片（§5.6）。"""

    suggestion_id: int | None = None
    text: str
    intent: Intent
    kind: DraftKind
    evidence: list[EvidenceRefView] = Field(default_factory=list)
    confidence: Confidence
    risk_hints: list[str] = Field(default_factory=list)
    qa_scores: dict[str, float] = Field(default_factory=dict)


class EngineResult(_Model):
    """问答引擎处理一条买家消息的结果。"""

    shop_id: str
    conversation_id: str
    msg_ids: list[str]
    mode: Mode
    route: RouteDecision | None = None
    suggestions: list[SuggestionCard] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)  # 标红提醒客服
    no_suggestion_reason: str | None = None
    missing_knowledge: list[str] = Field(default_factory=list)
    auto_reply: SuggestionCard | None = None  # 全自动模式下直接发送的回复
    transfer_to_human: bool = False
    degraded: list[str] = Field(default_factory=list)  # 如 ["llm", "decision", "tool:logistics"]
    latency_ms: dict[str, int] = Field(default_factory=dict)  # 各环节耗时


class SuggestionRecord(_Model):
    """推荐记录（§10 suggestion）。"""

    id: int | None = None
    shop_id: str
    conversation_id: str
    msg_id: str | None = None  # 触发推荐的买家消息
    question_text: str | None = None  # 买家问题（已做个人信息最小化），反馈闭环生成候选知识用
    intent: Intent
    text: str
    kind: DraftKind = DraftKind.LLM
    evidence_ids: list[str] = Field(default_factory=list)
    qa_scores: dict[str, float] | None = None
    status: SuggestionStatus
    edited_text: str | None = None
    reject_reason: RejectReason | None = None
    agent_id: str | None = None
    latency_ms: int | None = None
    created_at: AwareDatetime


# ---------------------------------------------------------------- 会话存储（§10）


class Conversation(_Model):
    id: str
    shop_id: str
    buyer_ref: str
    started_at: AwareDatetime
    closed_at: AwareDatetime | None = None


class StoredMessage(_Model):
    msg_id: str
    shop_id: str
    conversation_id: str
    role: Role
    text: str | None = None
    item_id: str | None = None
    sku_id: str | None = None
    order_ref: str | None = None
    created_at: AwareDatetime


# ---------------------------------------------------------------- 售后审核（§7.3.2、§10）


class AfterSaleReview(_Model):
    shop_id: str
    aftersale_id: str
    platform_status: AfterSaleStatus
    state: ReviewState
    matched_rules: list[str] = Field(default_factory=list)  # 如 ["R2", "R6"]
    suggestion: ReviewAdvice | None = None
    reason: str | None = None  # 建议理由，展示给客服
    requires_manager: bool = False  # R6：必须店长确认
    risk_flags: list[str] = Field(default_factory=list)  # 如 R7 "30 天内售后 ≥ 3 次"
    verify_started_at: AwareDatetime | None = None  # R4 签收核查开始时间
    confirmed_by: str | None = None
    action_result: dict[str, Any] | None = None
    version: int = 0  # 乐观锁，也用于幂等键
    updated_at: AwareDatetime | None = None

    @property
    def matched_rule(self) -> str | None:
        """与 §10 的 matched_rule TEXT 列对应。"""
        return ",".join(self.matched_rules) or None


# ---------------------------------------------------------------- 审计（§4.1 原则 6、§10）


class AuditEvent(_Model):
    shop_id: str | None
    actor: str  # system / agent_id / merchant
    action: str  # 如 "route"、"generate"、"qa"、"suggest"、"feedback"、"aftersale.confirm"
    detail: dict[str, Any] = Field(default_factory=dict)  # 不放买家原文，只放最小化后的内容
    created_at: AwareDatetime


class DecisionLogEntry(_Model):
    shop_id: str
    conversation_id: str | None = None
    stage: DecisionStage
    provider: str
    questions_ver: str
    answers: dict[str, Any]
    route_result: str | None = None
    latency_ms: int | None = None
    created_at: AwareDatetime


# ---------------------------------------------------------------- 平台接入层（§8.2）


class PlatformContext(_Model):
    """PlatformAdapter.get_context 的返回：商品卡片、订单卡片、最近对话。"""

    shop_id: str
    conversation_id: str
    item_card: ItemRef | None = None
    order_card: OrderRef | None = None
    recent_messages: list[StoredMessage] = Field(default_factory=list)


class SendResult(_Model):
    ok: bool
    platform_msg_id: str | None = None
    error: str | None = None
