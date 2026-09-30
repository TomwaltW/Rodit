"""接口的请求和响应模型（方案 §5.6 推荐卡片、§5.8 反馈、§7.3.2 售后审核、§8.1 模式 C、§12.4 报表）。

请求模型一律 extra="forbid"：操作人（agent_id / operator / merchant）、by_manager、shop_id 都取自
调用方身份和路径，请求体里带这些字段直接 422，不会被悄悄采用（§4.1 原则 3）。

响应模型只放客服 / 商家需要看的内容（§5.3 最小化个人信息、§12.3）：没有 buyer_ref、消息 ID、
平台原始买家 ID、判断层原始答案、质检分数、各环节耗时；售后单只给状态、类型、原因、金额、
凭证数和状态版本（确认时要带上），没有买家说明。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.api.auth import PrincipalRole
from rodit.core import evidence as ev
from rodit.core.enums import (
    INTENT_LABELS,
    AfterSaleAction,
    AfterSaleReason,
    AfterSaleStatus,
    AfterSaleType,
    Confidence,
    DraftKind,
    Intent,
    KnowledgeSource,
    KnowledgeStatus,
    Mode,
    Platform,
    RejectReason,
    ReviewAdvice,
    ReviewState,
    SuggestionStatus,
)
from rodit.core.models import (
    AfterSaleInfo,
    AfterSaleReview,
    EngineResult,
    KnowledgeItem,
    SuggestionCard,
)
from rodit.knowledge import describe_gap

MAX_TEXT_CHARS = 2000
MAX_NOTE_CHARS = 500
MAX_ID_CHARS = 128


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid")


Id = Annotated[str, Field(min_length=1, max_length=MAX_ID_CHARS)]


# ================================================================ 请求


class EmptyRequest(_Request):
    """没有参数的操作（发送、审核通过、下线、生成审核建议、取消订购）。

    请求体可以省略或为 {}；带任何字段（如 agent_id、merchant）都是 422。
    """


class StandaloneAskRequest(_Request):
    """模式 C 独立话术助手（§8.1）：客服粘贴的买家问题。客服账号取自调用方身份。"""

    session_id: Id
    question: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    item_id: Id | None = None
    sku_id: Id | None = None
    request_id: Id | None = None  # 同一会话内重试时去重


class EditedRequest(_Request):
    """编辑后发送（§5.8）。"""

    edited_text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class RejectedRequest(_Request):
    """不合适（§5.8）：原因；理解错误时可以给出真实的问题类型。"""

    reason: RejectReason
    true_intent: Intent | None = None


class QuestionnaireRequest(_Request):
    """店铺政策问卷（附录 C）：题号 → 回答。"""

    answers: dict[int, Annotated[str, Field(max_length=MAX_TEXT_CHARS)]] = Field(min_length=1)


class NegotiateRequest(_Request):
    """客服选择协商（§7.3.2）。备注由审核助手脱敏后留痕。"""

    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class NegotiationDoneRequest(_Request):
    """协商有结果（§7.3.2）。"""

    outcome: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class ConfirmRequest(_Request):
    """客服确认售后写操作（§7.3.2）。

    expected_status_version 是客服确认时看到的平台状态版本（审核视图的 platform.status_version），
    助手重读后不一致就不操作。确认人、是否店长取自调用方身份，不在这里。
    """

    action: AfterSaleAction
    expected_status_version: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    reason_code: Id | None = None  # 抖店拒绝必须带原因码
    reason_text: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


# ================================================================ 缺失知识（§5.7、§5.8）


class GapView(_View):
    """要提醒商家补的知识（§5.2.3、§5.7、§5.8）。"""

    marker: str  # 如 "policy:invoice"
    description: str  # 如 "店铺政策 · 发票（问卷第 7 题）"


def describe_missing(marker: str) -> str:
    """缺失标记 → 给客服、商家看的说明。

    知识类缺失沿用知识中心的说明（rodit.knowledge.describe_gap，
    如"店铺政策 · 发票（问卷第 7 题）"）；
    这里补上它不认识的几类：会话里没有订单、平台规则、说不清具体缺哪条（"knowledge:{intent}"）。
    认不出的标记原样返回。
    """
    if marker == ev.MISSING_ORDER:
        return "会话里没有订单信息"
    kind, _, rest = marker.partition(":")
    try:
        if kind == "knowledge":
            return f"「{INTENT_LABELS[Intent(rest)]}」相关的知识"
        if kind == "platform":
            platform, _, key = rest.partition(":")
            return ev.platform_rule_label(platform, key)
    except ValueError:
        return marker
    return describe_gap(marker)


def gap_view(marker: str) -> GapView:
    return GapView(marker=marker, description=describe_missing(marker))


# ================================================================ 推荐（§5.6）


class EvidenceLabelView(_View):
    """依据来源，如"商品属性 · 面料""店铺政策 · 发货时效"。"""

    id: str
    label: str


class SuggestionView(_View):
    """推荐卡片（§5.6）：推荐回答、依据、置信度、风险提示。suggestion_id 用于反馈。"""

    suggestion_id: int | None = None
    text: str
    intent: Intent
    kind: DraftKind
    confidence: Confidence
    evidence: list[EvidenceLabelView] = Field(default_factory=list)
    risk_hints: list[str] = Field(default_factory=list)

    @classmethod
    def from_card(cls, card: SuggestionCard) -> SuggestionView:
        return cls(
            suggestion_id=card.suggestion_id,
            text=card.text,
            intent=card.intent,
            kind=card.kind,
            confidence=card.confidence,
            evidence=[EvidenceLabelView(id=e.id, label=e.label) for e in card.evidence],
            risk_hints=list(card.risk_hints),
        )


class EngineResultView(_View):
    """一条买家消息的处理结果（给客服工作台 / 模式 C 网页工具看）。

    影子模式（§13.4）：shadow=true，推荐、自动回复、提醒、缺失知识一律不返回（生成了也只记录，
    事后对比）；此时 suggestions 为空不代表没有生成。
    auto_reply 只在全自动（mode=auto）时出现，表示引擎给出了自动回复（由适配器发送，§8.2）。
    """

    shop_id: str
    conversation_id: str
    mode: Mode
    shadow: bool = False
    suggestions: list[SuggestionView] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    no_suggestion_reason: str | None = None
    missing_knowledge: list[str] = Field(default_factory=list)
    missing_knowledge_details: list[GapView] = Field(default_factory=list)  # 同上，带中文说明
    auto_reply: SuggestionView | None = None
    transfer_to_human: bool = False
    degraded: list[str] = Field(default_factory=list)

    @classmethod
    def from_result(cls, result: EngineResult) -> EngineResultView:
        base: dict[str, Any] = {
            "shop_id": result.shop_id,
            "conversation_id": result.conversation_id,
            "mode": result.mode,
            "transfer_to_human": result.transfer_to_human,
            "degraded": list(result.degraded),
        }
        if result.mode == Mode.SHADOW:
            return cls(**base, shadow=True)
        auto = result.auto_reply if result.mode == Mode.AUTO else None
        return cls(
            **base,
            suggestions=[SuggestionView.from_card(c) for c in result.suggestions],
            alerts=list(result.alerts),
            no_suggestion_reason=result.no_suggestion_reason,
            missing_knowledge=list(result.missing_knowledge),
            missing_knowledge_details=[gap_view(m) for m in result.missing_knowledge],
            auto_reply=SuggestionView.from_card(auto) if auto is not None else None,
        )


# ================================================================ 反馈（§5.8）


class FeedbackView(_View):
    """一次客服操作的结果。只给状态和派生数据的 ID，不回传推荐原文和买家问题。"""

    suggestion_id: int
    status: SuggestionStatus
    candidate_knowledge_id: int | None = None  # 编辑后发送生成的候选知识（等商家审核）
    candidate_skipped: str | None = None  # 没有生成候选知识的原因
    knowledge_gap: str | None = None  # 知识缺失时记下的缺失标记
    labeled: bool = False  # 理解错误时是否写进了判断层标注集


class KnowledgeView(_View):
    """知识条目（商家审核用）。问题是已做个人信息最小化的买家问题（§5.3）。"""

    id: int
    question: str
    answer: str
    status: KnowledgeStatus
    source: KnowledgeSource
    item_id: str | None = None
    intent: Intent | None = None
    approved_by: str | None = None
    created_at: AwareDatetime | None = None

    @classmethod
    def from_item(cls, item: KnowledgeItem) -> KnowledgeView:
        assert item.id is not None, "仓储返回的知识必须有 id"
        return cls(
            id=item.id,
            question=item.question,
            answer=item.answer,
            status=item.status,
            source=item.source,
            item_id=item.item_id,
            intent=item.intent,
            approved_by=item.approved_by,
            created_at=item.created_at,
        )


class QuestionnaireView(_View):
    """问卷写入结果：写进了哪些政策键、改了哪些店铺配置项。"""

    policies: list[str]
    config: list[str]


class RevokeView(_View):
    """取消订购后删除的数据条数（§8.4）。counts 的键是各存储的名字，另有 auth（令牌数）。"""

    counts: dict[str, int]
    total: int


# ================================================================ 售后审核（§7.3.2）


class AfterSaleSnapshotView(_View):
    """平台售后单的当前状态（最小化：没有买家 ID、买家说明）。确认时带上 status_version。"""

    status: AfterSaleStatus
    status_version: str
    type: AfterSaleType
    reason: AfterSaleReason
    amount: Decimal
    evidence_count: int = 0

    @classmethod
    def from_info(cls, info: AfterSaleInfo) -> AfterSaleSnapshotView:
        return cls(
            status=info.status,
            status_version=info.status_version,
            type=info.type,
            reason=info.reason,
            amount=info.amount,
            evidence_count=info.evidence_count,
        )


class ReviewStepView(_View):
    """审核状态机的一次迁移。note 已由审核助手脱敏。"""

    from_state: ReviewState
    to_state: ReviewState
    at: AwareDatetime
    by: str | None = None
    note: str | None = None


class ActionResultView(_View):
    """最近一次平台写操作的结果。"""

    ok: bool
    action: AfterSaleAction
    platform_code: str | None = None
    message: str | None = None
    duplicate: bool = False


class AfterSaleReviewView(_View):
    """售后审核记录：规则建议、理由、状态、是否须店长确认、迁移历史。

    platform 只在待客服确认（state=confirm）时给出，是接口层重读的平台当前状态；
    读不到平台数据时为空（审核本身不受影响）。
    platform_status_version 是当前建议所依据的平台状态版本；needs_refresh 表示平台上的售后单
    在那之后又变了（如协商期间买家改了申请）：建议可能过期，先调 review 刷新再确认。
    """

    aftersale_id: str
    state: ReviewState
    platform_status: AfterSaleStatus
    platform_status_version: str | None = None
    matched_rules: list[str] = Field(default_factory=list)
    suggestion: ReviewAdvice | None = None
    reason: str | None = None
    requires_manager: bool = False
    risk_flags: list[str] = Field(default_factory=list)
    verify_started_at: AwareDatetime | None = None
    confirmed_by: str | None = None
    version: int = 0
    updated_at: AwareDatetime | None = None
    history: list[ReviewStepView] = Field(default_factory=list)
    last_result: ActionResultView | None = None
    platform: AfterSaleSnapshotView | None = None
    needs_refresh: bool = False

    @classmethod
    def from_review(
        cls, review: AfterSaleReview, *, platform: AfterSaleInfo | None = None
    ) -> AfterSaleReviewView:
        # 元数据的键见 rodit.aftersale.state_machine；缺键的条目跳过（不能让 KeyError 变成 404）
        meta = review.action_result or {}
        history = [
            ReviewStepView(
                from_state=step["from"],
                to_state=step["to"],
                at=step["at"],
                by=step.get("by"),
                note=step.get("note"),
            )
            for step in meta.get("history", [])
            if isinstance(step, dict) and {"from", "to", "at"} <= step.keys()
        ]
        raw_result = meta.get("result")
        last_result = (
            ActionResultView(
                ok=raw_result["ok"],
                action=raw_result["action"],
                platform_code=raw_result.get("platform_code"),
                message=raw_result.get("message"),
                duplicate=bool(raw_result.get("duplicate", False)),
            )
            if isinstance(raw_result, dict) and {"ok", "action"} <= raw_result.keys()
            else None
        )
        return cls(
            aftersale_id=review.aftersale_id,
            state=review.state,
            platform_status=review.platform_status,
            platform_status_version=review.platform_status_version,
            matched_rules=list(review.matched_rules),
            suggestion=review.suggestion,
            reason=review.reason,
            requires_manager=review.requires_manager,
            risk_flags=list(review.risk_flags),
            verify_started_at=review.verify_started_at,
            confirmed_by=review.confirmed_by,
            version=review.version,
            updated_at=review.updated_at,
            history=history,
            last_result=last_result,
            platform=AfterSaleSnapshotView.from_info(platform) if platform is not None else None,
            needs_refresh=platform is not None
            and platform.status_version != review.platform_status_version,
        )


class HealthView(_View):
    status: str = "ok"


# ================================================================ 模式 C 网页工具（§8.1）


class LLMDisclosureView(_View):
    """对外公示的大模型信息（§12.3 生成式人工智能服务：公示所用模型的名称和备案号 / 上线编号）。
    来自部署配置的 llm.public_name / llm.filing_no；没填的项不返回。没有接口地址、模型 ID、密钥。"""

    public_name: str | None = None
    filing_no: str | None = None


class MeView(_View):
    """当前令牌对应的身份（网页工具登录用）。只有调用方自己的信息，没有令牌、没有店铺数据。

    llm：配置了公示信息时才有（网页工具页脚显示）；没配置时响应里没有这个字段。"""

    shop_id: str
    user_id: str
    role: PrincipalRole
    platform: Platform  # 本平台单元（§9.1）
    llm: LLMDisclosureView | None = None


class ItemSummaryView(_View):
    """商品列表里的一项（模式 C 选当前商品，§5.3）：只有 ID 和标题。"""

    item_id: str
    title: str


__all__ = [
    "ActionResultView",
    "AfterSaleReviewView",
    "AfterSaleSnapshotView",
    "ConfirmRequest",
    "EditedRequest",
    "EmptyRequest",
    "EngineResultView",
    "EvidenceLabelView",
    "FeedbackView",
    "GapView",
    "HealthView",
    "ItemSummaryView",
    "KnowledgeView",
    "LLMDisclosureView",
    "MeView",
    "NegotiateRequest",
    "NegotiationDoneRequest",
    "QuestionnaireRequest",
    "QuestionnaireView",
    "RejectedRequest",
    "ReviewStepView",
    "RevokeView",
    "StandaloneAskRequest",
    "SuggestionView",
    "describe_missing",
    "gap_view",
]
