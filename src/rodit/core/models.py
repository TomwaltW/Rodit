"""标准数据模型（方案 §5.2.2、§5.3、§5.5、§5.6、§7、§8.3、§10）。

约定：
- 所有时间都是带时区的 datetime（AwareDatetime，默认东八区，见 core.clock.CST）。
- 金额用 Decimal。
- 模型里没有买家昵称、手机号、收件地址字段（§5.3 最小化个人信息）：
  订单只保留状态、商品、金额、时间等回答需要的字段；买家只用 buyer_ref（平台买家 ID 的哈希）。
- Evidence.data 里放结构化原始数据（model_dump(mode="json")），content 放给大模型看的文字。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

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
    QualityReviewStatus,
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


# ---------------------------------------------------------------- 文本清洗

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f]")


def strip_control_chars(value: str) -> str:
    """去掉 NUL 和其他 C0 控制字符（保留换行、制表符）。

    PostgreSQL 的 TEXT / JSONB 不接受 NUL，内存实现照单全收；会写进存储的自由文本在模型边界
    统一清洗，两种实现对同一段文字的结果一致
    （契约见 rodit.testing.contracts.check_text_sanitation）。
    只在构造 / model_validate 时生效；model_copy(update=...) 不做校验，调用方自己先清洗。
    """
    return _CONTROL_CHARS.sub("", value) if value else value


def _strip_json(value: Any) -> Any:
    """JSON 值里的字符串（含键）逐个清洗，其余原样。"""
    if isinstance(value, str):
        return strip_control_chars(value)
    if isinstance(value, dict):
        return {
            (strip_control_chars(k) if isinstance(k, str) else k): _strip_json(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_strip_json(v) for v in value]
    return value


CleanText = Annotated[str, AfterValidator(strip_control_chars)]
"""会写进存储的自由文本：去掉 NUL 等控制字符。"""
CleanJson = Annotated[dict[str, Any], AfterValidator(_strip_json)]
"""会写进 JSONB 的 dict：字符串逐个清洗。"""
SizeRow = Annotated[dict[str, str | int | float], AfterValidator(_strip_json)]
"""尺码表的一行（写进 JSONB）：字符串逐个清洗。"""


# 约定：模型里的 dict[str, Any] 字段（Shop.config_overrides、Evidence.data、
# AfterSaleReview.action_result、AuditEvent.detail、DecisionLogEntry.answers、
# StandardMessage.extension）只放 JSON 原生值
# （str / int / float / bool / None / list / dict）；Decimal、时间写成字符串。这样内存实现和
# PostgreSQL（JSONB）读回来的值一致。


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
    text: CleanText
    item_card: ItemRef | None = None
    order_card: OrderRef | None = None
    agent_id: str | None = None
    mode: Mode = Mode.ASSIST
    received_at: AwareDatetime
    role: Role = Role.BUYER
    extension: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- 店铺与配置（§9.3、§10）


class ShopConfig(_Model):
    """店铺的"生效配置"（§9.3）：平台默认 → 类目默认 → 店铺覆盖，逐层合并后的结果。

    由 core.config.effective_shop_config 算出；阈值由我们统一校准，不在这里。
    """

    display_name: str = ""  # 店铺名，用于提示词"你是{店铺名}的客服"
    tone: Literal["亲切", "专业", "简洁"] = "亲切"
    salutation: Literal["亲", "您"] = "亲"
    max_answer_chars: int = Field(default=120, ge=20, le=500)
    banned_words: list[str] = Field(default_factory=list)  # 附录 C 第 14 题
    disabled_intents: list[Intent] = Field(default_factory=list)  # 关闭的问题类型
    suggestion_count: int = Field(default=2, ge=1, le=3)  # 推荐条数 1–3（§5.4）
    aftersale_amount_limit: Decimal = Decimal("500")  # R6：超过必须店长确认
    manager_ids: list[str] = Field(default_factory=list)  # 店长子账号（R6 必须店长确认）
    shadow_mode: bool = False  # 影子模式：生成但不展示（§13.4 上线第一阶段）
    auto_reply_enabled: bool = False  # 全自动总开关（后期，且需平台准入）


SHOP_CONFIG_FIELDS = frozenset(ShopConfig.model_fields)


class Shop(_Model):
    """店铺（§10 shop）。config_overrides 只存店铺自己改过的项（§10 shop.config JSONB），
    生效配置要用 core.config.effective_shop_config 合并平台默认和类目默认后得到。"""

    id: str
    platform: Platform
    category: str | None = None  # 主营类目，如 "女装"
    status: ShopStatus = ShopStatus.ACTIVE
    config_overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("config_overrides")
    @classmethod
    def _check_overrides(cls, v: dict[str, Any]) -> dict[str, Any]:
        unknown = set(v) - SHOP_CONFIG_FIELDS
        if unknown:
            raise ValueError(f"未知的店铺配置项：{sorted(unknown)}")
        v = _strip_json(v)  # 存 JSONB：去掉 NUL 等控制字符
        ShopConfig.model_validate(v)  # 取值必须合法
        return v


def merge_overrides(shop: Shop, changes: Mapping[str, Any]) -> dict[str, Any]:
    """店铺配置覆盖项的合并规则（KnowledgeRepository.update_shop_config 共用）：
    值为 None 的键删掉（回到默认），其余覆盖。不校验，校验交给 Shop。"""
    merged = dict(shop.config_overrides)
    for key, value in changes.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


class ShopAuth(_Model):
    """店铺授权（§8.4）。access_token 只存放在对应平台云内，落库时加密；
    用 SecretStr 防止出现在 repr / 日志里。

    - refresh_token：平台给的续期令牌（有的平台没有），和 access_token 一样加密落库；
      refresh_expires_at 是它的有效期（过了只能让商家重新授权）。
    - refreshed_at：最近一次续期成功的时间（开通时为空）。
    - needs_reauth：续期被平台拒绝、或平台拒绝了当前令牌（TokenProvider.report_rejected）：
      停止处理（AuthGuard.require 不通过），等商家重新授权；重新保存授权时按传入值覆盖。
    """

    shop_id: str
    access_token: SecretStr
    scopes: list[str]
    expires_at: AwareDatetime
    refresh_token: SecretStr | None = None
    refresh_expires_at: AwareDatetime | None = None
    refreshed_at: AwareDatetime | None = None
    needs_reauth: bool = False


SubscriptionKind = Literal["subscribed", "renewed", "cancelled", "auth_revoked"]
"""服务市场的订购事件（§8.4）：订购、续订、取消订购、商家撤销授权。"""


class SubscriptionEvent(_Model):
    """平台推送的订购 / 授权事件（§8.4），由平台云代码解析成这个格式
    （PlatformBindings.parse_subscription；测试用 rodit.testing.fakes.parse_subscription）。

    - subscribed / renewed 带上新的令牌（access_token，可能还有 refresh_token）和权限；
    - cancelled / auth_revoked 只需要 shop_id：停止处理并按约定删除数据。
    令牌是 SecretStr，不出现在 repr / 日志里；没有买家信息。
    """

    kind: SubscriptionKind
    shop_id: str = Field(min_length=1)
    platform: Platform
    at: AwareDatetime
    scopes: list[str] = Field(default_factory=list)
    access_token: SecretStr | None = None
    refresh_token: SecretStr | None = None
    expires_at: AwareDatetime | None = None
    refresh_expires_at: AwareDatetime | None = None
    category: str | None = None  # 主营类目（开通时写进 Shop.category）

    @model_validator(mode="after")
    def _grant_has_token(self) -> SubscriptionEvent:
        if self.kind in ("subscribed", "renewed") and (
            self.access_token is None or self.expires_at is None
        ):
            raise ValueError("订购 / 续订事件必须带 access_token 和 expires_at")
        return self


class RefreshedToken(_Model):
    """TokenRefreshPort.refresh 的结果：新的令牌和有效期。平台不换续期令牌时 refresh_token 为空
    （调用方沿用旧的）。"""

    access_token: SecretStr
    expires_at: AwareDatetime
    refresh_token: SecretStr | None = None
    refresh_expires_at: AwareDatetime | None = None


# ---------------------------------------------------------------- 知识（§5.2）


class PolicyEntry(_Model):
    shop_id: str
    key: str  # 通常是 PolicyKey 的取值
    value: CleanText
    updated_at: AwareDatetime


class Sku(BaseModel):
    """SKU。规格属性（颜色、尺码……）直接作为额外字段，和 §5.2.2 的 JSON 一致。"""

    model_config = ConfigDict(extra="allow")

    sku_id: CleanText
    stock_source: str = "realtime"

    @model_validator(mode="after")
    def _clean_props(self) -> Sku:
        # 规格属性来自平台商品接口（商品同步），会写进 JSONB：键和值都去掉 NUL 等控制字符
        if self.__pydantic_extra__:
            self.__pydantic_extra__ = _strip_json(self.__pydantic_extra__)
        return self

    def props(self) -> dict[str, str]:
        return {k: str(v) for k, v in (self.model_extra or {}).items()}


class ItemCard(_Model):
    """商品知识卡片（§5.2.2）。价格和库存不写死在卡片里，回答时实时查。"""

    shop_id: str
    platform: Platform
    item_id: str
    title: CleanText
    category: CleanText | None = None
    attributes: dict[str, CleanText] = Field(default_factory=dict)
    skus: list[Sku] = Field(default_factory=list)
    size_chart: list[SizeRow] = Field(default_factory=list)
    size_advice: CleanText | None = None  # 商家填写，尺码推荐最重要的依据
    care: list[CleanText] = Field(default_factory=list)
    usage: list[CleanText] = Field(default_factory=list)  # 使用 / 安装说明
    forbidden_claims: list[CleanText] = Field(default_factory=list)  # 不能说的卖点，质检用
    detail_text: CleanText | None = None  # 详情页文字 + 详情图 OCR 文字
    # 上次商品同步时平台给的详情文字（不含 OCR；平台没有文字时为空串，从没同步过为 None）。
    # 只给商品同步判断"平台文字变没变"用（有详情图识别失败时决定能不能保留上次的合并结果），回答不用
    platform_detail_text: CleanText | None = None
    updated_at: AwareDatetime

    def find_sku(self, sku_id: str) -> Sku | None:
        return next((s for s in self.skus if s.sku_id == sku_id), None)

    def patched(self, changes: Mapping[str, Any]) -> ItemCard:
        """改部分字段后的新卡片（KnowledgeRepository.update_item_card 共用）：按模型重新校验，
        不认识的字段、改归属（shop_id / item_id / platform）抛 ValueError。"""
        unknown = set(changes) - set(ItemCard.model_fields)
        if unknown:
            raise ValueError(f"商品卡片没有这些字段：{sorted(unknown)}")
        fixed = {"shop_id", "item_id", "platform"} & set(changes)
        if fixed:
            raise ValueError(f"商品卡片的归属不能改：{sorted(fixed)}")
        return ItemCard.model_validate({**self.model_dump(), **changes})


class PlatformItem(_Model):
    """平台商品接口返回的商品（§5.2.3 商品自动同步）：只有平台维护的字段。

    同步时覆盖 ItemCard 的标题、类目、属性、SKU、尺码表、详情文字；商家填写的字段
    （size_advice、care、usage、forbidden_claims）不在这里，同步不会动。
    detail_images 是详情图在平台上的引用（图片 ID 或平台 CDN 地址），交给 ImageTextExtractor
    在平台云内做 OCR，文字并入 detail_text。价格、库存不在这里（回答时实时查）。
    """

    shop_id: str
    platform: Platform
    item_id: str
    title: CleanText
    category: CleanText | None = None
    attributes: dict[str, CleanText] = Field(default_factory=dict)
    skus: list[Sku] = Field(default_factory=list)
    size_chart: list[SizeRow] = Field(default_factory=list)
    detail_text: CleanText | None = None
    detail_images: list[str] = Field(default_factory=list)
    on_sale: bool = True  # 已下架的商品同步时不删卡片，只报告出来
    updated_at: AwareDatetime


class KnowledgeItem(_Model):
    """FAQ 与优质回答（§10 knowledge_item）。"""

    id: int | None = None
    shop_id: str
    question: CleanText
    answer: CleanText
    source: KnowledgeSource
    status: KnowledgeStatus
    item_id: str | None = None  # 只适用于某个商品的 FAQ
    intent: Intent | None = None
    approved_by: str | None = None
    created_at: AwareDatetime | None = None
    embedding: list[float] | None = None


class KnowledgeGap(_Model):
    """知识缺失提醒（§5.8 "知识缺失 → 提醒商家补知识"，§5.7 "提示缺哪类知识"）。

    missing 的写法：引擎记 EvidenceBundle.missing 里的标记（core.evidence 的 missing_* 构造），
    或 core.evidence.missing_any(intent)（没有依据、也说不清缺哪条）；反馈闭环记
    missing_item_field(item_id, intent) 或 missing_any(intent)。
    """

    shop_id: str
    intent: Intent | None = None
    missing: str  # 如 "policy:invoice"、"item:i_555:size_advice"
    question_text: CleanText | None = None  # 已做个人信息最小化
    source: Literal["engine", "feedback"] = "engine"
    created_at: AwareDatetime


class LabeledSample(_Model):
    """判断层标注数据（§5.8 "理解错误 → 判断层标注集"，§6.4 训练小模型分类器）。"""

    shop_id: str
    text: CleanText  # 已做个人信息最小化
    predicted_intent: Intent | None = None
    # 路由给 predicted_intent 的置信度（SuggestionRecord.intent_confidence），§13.5 每月按采纳反馈
    # 重算阈值用（§13.3 的 (预测类型, 置信度, 真实类型) 三元组）；手工标注、旧数据为 None
    predicted_confidence: float | None = Field(default=None, ge=0, le=1)
    # 这条推荐走的 §6.6 路由规则（SuggestionRecord.route_rule）：重算阈值只用规则 6
    # （按置信度出推荐）的样本，安抚话术（规则 3）、分条（规则 5）不看辅助阈值
    route_rule: int | None = Field(default=None, ge=1, le=8)
    true_intent: Intent | None = None
    # feedback_adopt 原样发送、feedback_edit 编辑后发送（类型判断对了、只是改了说法）、
    # feedback_reject 点了"理解错误"
    source: Literal["feedback_reject", "feedback_adopt", "feedback_edit", "manual"] = "manual"
    created_at: AwareDatetime


class QualityScores(_Model):
    """§13.2 回答质量评分：每个维度 0 / 1 / 2 分。"""

    factual: int = Field(ge=0, le=2)  # 事实正确
    relevance: int = Field(ge=0, le=2)  # 切题
    compliance: int = Field(ge=0, le=2)  # 合规
    expression: int = Field(ge=0, le=2)  # 表达


QUALITY_PERIOD_PATTERN = r"^[0-9]{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$"
"""复核周期：ISO 周，如 "2026-W40"（datetime.isocalendar() 的年和周）。只认 ASCII 数字，
和库里 quality_review 的 CHECK 一致（正则的数字类会放过全角数字）。"""


class QualityReview(_Model):
    """每周人工复核的一条样本（§13.5 每周从推荐中抽 50 条人工复核，按 §13.2 打分）。

    只存推荐 ID 和打分，不复制推荐文字和买家问题：复核时从 SuggestionRecord 读（已做个人信息
    最小化），取消订购时随店铺数据一起删除。同一店铺、同一周期、同一推荐只有一条。
    scored 必须有 scores、reviewer、reviewed_at；pending 不能有 scores；skipped 要有 reviewer。
    """

    id: int | None = None
    shop_id: str
    period: str = Field(pattern=QUALITY_PERIOD_PATTERN)
    suggestion_id: int = Field(ge=1)
    intent: Intent
    status: QualityReviewStatus = QualityReviewStatus.PENDING
    scores: QualityScores | None = None
    reviewer: str | None = None
    note: CleanText | None = None  # 复核人备注（已做个人信息最小化）
    sampled_at: AwareDatetime
    reviewed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _consistent(self) -> QualityReview:
        if self.status == QualityReviewStatus.SCORED:
            if self.scores is None or not self.reviewer or self.reviewed_at is None:
                raise ValueError("已复核（scored）的样本必须有 scores、reviewer、reviewed_at")
        elif self.status == QualityReviewStatus.PENDING:
            if self.scores is not None or self.reviewer or self.reviewed_at is not None:
                raise ValueError("待复核（pending）的样本不能有 scores、reviewer、reviewed_at")
        elif not self.reviewer or self.reviewed_at is None:
            raise ValueError("跳过（skipped）的样本必须有 reviewer、reviewed_at")
        return self


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
    fetched_at: AwareDatetime | None = None  # 从平台读取的时间（接口故障用缓存时，回答写"截至"）


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
    note: str | None = None  # 买家填写的说明；ToolGateway 实现负责先过 mask_pii
    fetched_at: AwareDatetime | None = None  # 从平台读取的时间


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


class HumanConfirmation(_Model):
    """客服对一次售后写操作的确认（§4.1 原则 3、§7.3）。写接口只接受带确认的调用。

    - operator 是确认人的客服子账号，不能是 system；
    - expected_status_version 是客服确认时看到的平台状态版本，写接口实现重读后不一致就不操作；
    - by_manager：R6 金额超过上限时必须店长确认（ShopConfig.manager_ids）。
    """

    operator: str = Field(min_length=1)
    confirmed_at: AwareDatetime
    expected_status_version: str
    by_manager: bool = False

    @field_validator("operator")
    @classmethod
    def _human(cls, v: str) -> str:
        if not v.strip() or v.strip().lower() in {"system", "bot", "auto"}:
            raise ValueError("售后写操作必须由人确认")
        return v.strip()


def aftersale_idempotency_key(aftersale_id: str, action: str, status_version: str) -> str:
    """幂等键 = 售后单 ID + 操作类型 + 平台状态版本（§7.3.2 防止误操作）。"""
    return f"{aftersale_id}:{action}:{status_version}"


class ActionResult(_Model):
    """售后写操作的平台返回。ok=False 时 platform_code 说明原因（如 STATUS_CHANGED）。"""

    ok: bool
    action: AfterSaleAction
    aftersale_id: str
    idempotency_key: str
    platform_code: str | None = None
    message: str | None = None
    duplicate: bool = False  # 幂等键重复，没有重复调用平台


# ---------------------------------------------------------------- 会话理解（§5.3）


class Turn(_Model):
    """一轮对话：同一角色的连发消息合并后的一条。text 是原文（指代、订单号识别要用原文）。"""

    role: Role
    text: str
    at: AwareDatetime
    msg_ids: list[str] = Field(default_factory=list)


class ConversationContext(_Model):
    """问答引擎各步骤共享的会话上下文（由 engine 的会话理解构建）。

    - Turn.text 是原文；进判断层用 core.states 的构造函数（内部已脱敏），进生成提示词前
      必须先过 core.pii.mask_pii。
    - config 是店铺的生效配置（平台默认 → 类目默认 → 店铺覆盖），各步骤只读它，
      不读 shop 上的覆盖项。
    - mode 是本条消息的生效模式：店铺开了影子模式就是 SHADOW；
      消息是 AUTO 且店铺开了全自动才是 AUTO；
      其余为 ASSIST（由 engine 决定，不直接信任接入层）。
    - item / order 是指代理解之后的"当前商品 / 当前订单"。
    - candidate_item_ids 有多个且 item 为空时，说明商品不确定，需要澄清（§5.2.4）。
    """

    platform: Platform
    shop: Shop
    config: ShopConfig
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
    """一条依据。id 按 core.evidence 里的规则生成，回答的 evidence_ids 引用它。

    - topic 取 core.evidence.Topic 的值（或 None）；同一 topic 只保留优先级最高的来源（§5.2.1）。
    - score 只用于 FAQ：查询与 FAQ 问题向量的余弦相似度（L2 归一化），截断到 [0, 1]；
      和 GenerationConfig.faq_direct_min_score 比较，决定能否直接用标准回答。
    """

    id: str
    source: EvidenceSource
    label: str  # 展示给客服的来源，如"商品属性 · 面料""店铺政策 · 发货时效"
    content: str  # 给大模型看的文字（已最小化）
    data: dict[str, Any] = Field(default_factory=dict)
    topic: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    fetched_at: AwareDatetime | None = None  # 实时数据的获取时间，回答里写"截至"

    @property
    def priority(self) -> int:
        return self.source.priority


class EvidenceBundle(_Model):
    """证据包：检索 + 工具调用的结果。

    - 被高优先级覆盖的证据从 evidences 里移除，只把 id 记在 overridden_ids（§5.2.1），
      生成和质检只看 evidences。
    - alerts 是查依据时发现的、要提醒客服的事，如物流停滞、疑难件（§7.2），
      引擎并入推荐卡片的风险提示。
    """

    evidences: list[Evidence] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)  # 缺的知识，如 "policy:invoice"
    unavailable: list[str] = Field(default_factory=list)  # 暂时查不到的工具，如 "tool:logistics"
    overridden_ids: list[str] = Field(default_factory=list)  # 被高优先级覆盖掉的证据
    clarify_question: str | None = None  # 商品 / 订单不确定时的澄清问题
    alerts: list[str] = Field(default_factory=list)

    def ids(self) -> list[str]:
        return [e.id for e in self.evidences]

    def get(self, evidence_id: str) -> Evidence | None:
        return next((e for e in self.evidences if e.id == evidence_id), None)

    @property
    def is_empty(self) -> bool:
        return not self.evidences


# ---------------------------------------------------------------- 路由（§6.6）


class RoutePart(_Model):
    """多问题消息里的一个问题（§6.6 规则 5 "按问题分条推荐"）。"""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)


class RouteDecision(_Model):
    """路由结果。写 decision_log（stage=route）的是 Router 实现，不是引擎。

    - top_intents：按概率降序的前两名，规则 7 澄清用。
    - parts：action == SPLIT_MULTI 时由 Router 填写：
      概率 ≥ routing.multi_intent_min_prob 的问题类型，
      按概率降序，最多 routing.max_parts 个。引擎对每个 part 用
      route.model_copy(update={"intent": part.intent, "parts": [],
                               "action": ANSWER 或 HIGH_RISK_ANSWER})
      分别检索、生成、质检，每个 part 出一条推荐。
    - auto_eligible：规则 6 命中、达到该类型的自动阈值、该类型已开放自动、店铺开了全自动、没有降级。
    """

    action: RouteAction
    rule: int = Field(ge=1, le=8)  # 命中的 §6.6 规则序号
    intent: Intent
    intent_confidence: float = Field(ge=0.0, le=1.0)
    top_intents: list[tuple[Intent, float]] = Field(default_factory=list)
    parts: list[RoutePart] = Field(default_factory=list)
    need_human: float = 0.0
    emotion: float = 0.0
    multi_intent: float = 0.0
    high_risk: bool = False
    auto_eligible: bool = False
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
    """质检结果。写 decision_log（stage=qa）的是 QualityGate 实现，不是引擎。

    hard_hits 里 format / length 是自动修正（改写 text），不算拦截；其余命中都拦截。
    """

    passed: bool
    text: str  # 清洗后的文本（去 Markdown、超长截断）
    hard_hits: list[HardRuleHit] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)  # grounded / promise_risk / tone ...
    reasons: list[str] = Field(default_factory=list)  # 不通过的原因，展示给客服 / 带回重新生成
    model_checked: bool = False  # 是否跑了判断层质检（FAQ / 模板只跑硬规则）
    degraded: bool = False  # 判断层质检不可用
    provider: str | None = None  # 实际使用的判断层实现
    answers: DecisionResponse | None = None  # 判断层原始答案（模型质检时）


NON_BLOCKING_HARD_RULES = frozenset({"format", "length"})


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
    """问答引擎处理一条买家消息的结果。

    - mode == SHADOW 时 suggestions 只用于事后对比，app 层不得展示给客服（§13.4）。
    - auto_reply 只在全自动条件全部满足时出现，text 已带 AI 生成内容标识前缀
      （GenerationConfig.auto_reply_prefix，§12.3），且带前缀后仍满足长度限制；前缀只由引擎添加。
    """

    shop_id: str
    conversation_id: str
    msg_ids: list[str]
    mode: Mode
    route: RouteDecision | None = None
    suggestions: list[SuggestionCard] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)  # 标红提醒客服
    no_suggestion_reason: str | None = None
    missing_knowledge: list[str] = Field(default_factory=list)
    auto_reply: SuggestionCard | None = None
    transfer_to_human: bool = False
    degraded: list[str] = Field(default_factory=list)  # 如 ["llm", "decision", "tool:logistics"]
    latency_ms: dict[str, int] = Field(default_factory=dict)  # 各环节耗时


class SuggestionRecord(_Model):
    """推荐记录（§10 suggestion）。

    由引擎在返回前为每张推荐卡片写入（SHOWN，影子模式为 SHADOW），
    并回填 SuggestionCard.suggestion_id；
    同一会话来了新的买家消息时，引擎把之前 SHOWN 的记录改成 EXPIRED。客服操作由反馈闭环更新。
    """

    id: int | None = None
    shop_id: str
    conversation_id: str
    msg_id: str | None = None  # 触发推荐的买家消息
    question_text: CleanText | None = None  # 买家问题（已做个人信息最小化），反馈闭环生成候选知识用
    item_id: str | None = None  # 当前商品：采纳后生成的候选 FAQ 按商品限定范围
    intent: Intent
    # 路由给 intent 的置信度（多问题分条时是这一条的概率，§6.6 规则 5）；§13.5 重算阈值用
    intent_confidence: float | None = Field(default=None, ge=0, le=1)
    route_rule: int | None = Field(default=None, ge=1, le=8)  # 命中的 §6.6 路由规则（整条消息的）
    text: CleanText
    kind: DraftKind = DraftKind.LLM
    evidence_ids: list[str] = Field(default_factory=list)
    qa_scores: dict[str, float] | None = None
    status: SuggestionStatus
    edited_text: CleanText | None = None
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
    text: CleanText | None = None
    item_id: str | None = None
    sku_id: str | None = None
    order_ref: str | None = None
    created_at: AwareDatetime


# ---------------------------------------------------------------- 售后审核（§7.3.2、§10）


class AfterSaleReview(_Model):
    shop_id: str
    aftersale_id: str
    platform_status: AfterSaleStatus
    # 最近一次和平台同步时的状态版本：当前建议就是按这个版本的售后单给的。
    # 确认时平台版本已经变了 → 建议可能过期，要求刷新（§7.3.2 调用前重新读取）
    platform_status_version: str | None = None
    state: ReviewState
    matched_rules: list[str] = Field(default_factory=list)  # 如 ["R2", "R6"]
    suggestion: ReviewAdvice | None = None
    reason: CleanText | None = None  # 建议理由，展示给客服
    requires_manager: bool = False  # R6：必须店长确认
    risk_flags: list[str] = Field(default_factory=list)  # 如 R7 "30 天内售后 ≥ 3 次"
    verify_started_at: AwareDatetime | None = None  # R4 签收核查开始时间
    confirmed_by: str | None = None
    action_result: CleanJson | None = None
    version: int = 0  # 乐观锁（幂等键用平台的 status_version，见 aftersale_idempotency_key）
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
    detail: CleanJson = Field(default_factory=dict)  # 不放买家原文，只放最小化后的内容
    created_at: AwareDatetime


class DecisionLogEntry(_Model):
    shop_id: str
    conversation_id: str | None = None
    stage: DecisionStage
    provider: str
    questions_ver: str
    answers: CleanJson
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
