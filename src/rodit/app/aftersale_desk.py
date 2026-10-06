"""售后审核工作台（方案 §7.3.2 商家侧审核助手、§15 Phase 2 "售后审核助手（申请写售后权限）"、
§2.1 抖店体验分"售后平均审核时长"、§7.4 审核助手直接改善这个指标）。

单张售后单的规则建议、协商、确认执行在审核助手（rodit.aftersale.ReviewAssistant）和
AssistService（review_aftersale / negotiate_aftersale / negotiation_done / confirm_aftersale）里；
这里补上客服的工作台：待审核列表、批量"拉取并跑规则"、单张详情（能做哪些操作、为什么不能做）、
统计（各状态的单数、各规则的待处理单数、平均审核时长）。

- 读（列表、详情、统计）不写任何东西：不建审核记录、不推进状态机、不写审计。先查店铺授权（§8.4
  授权过期时停止处理），授权无效抛 AuthExpired，一次平台接口都不调。
- 列表 = 平台的待审核售后单（带缓存的只读网关，§12.1 省额度）+ 本地还没结束的审核记录
  （签收核查中、等待凭证、待客服确认、协商中、退货中……平台待审核列表里可能已经没有）。
  平台数据来自短时缓存：审核记录比缓存新时以审核记录同步到的平台状态为准；缓存比审核记录新、
  状态或状态版本不一样时标"平台已变化"，提示刷新（确认执行前审核助手还会重读，§7.3.2）。
  平台待审核列表读不到（ToolUnavailable）时只列本地审核记录，并标明（§12.2）。
- 批量跑规则走 AssistService.review_aftersale（和单张一样：店铺闸门、授权检查、审核助手、审计、
  状态机都不变），一张失败（平台接口故障、并发冲突……）只记在这一行，不影响其他行；授权失效、
  店铺停用时停下，剩下的不再跑。R5 凭证的图片识别这一批共用预算、超时一次就熔断
  （aftersale.evidence_vision.evidence_batch）：视觉模型挂住时一次请求不会拖到几分钟。
- 详情重读平台（不走缓存：确认要带最新的状态版本）；读不到时退回缓存数据并标明，此时不能确认。
  可执行的操作按平台差异（§7.1、§7.3.2：拼多多没有拒绝接口、淘宝拒绝要先协商、抖店拒绝要原因码）、
  写接口和写售后权限（MVP 只给建议，§8.4）、审核状态、平台状态、R6 店长确认逐条给出不能做的原因。
  这些只是界面提示，真正的检查在确认接口和审核助手里。
- 买家说明（AfterSaleInfo.note）网关已经打码（ToolGateway 协议），这里再过一遍 mask_pii 并截短；
  不返回 buyer_ref（§5.3 最小化个人信息）。
- 店铺隔离（§9.2）：所有读取都按 shop_id；网关、仓储返回别的店铺的数据时丢掉或报错，不往外给。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.aftersale import (
    AFTERSALE_STATUS_LABELS,
    TERMINAL_STATES,
    history,
    last_result,
    was_negotiated,
)
from rodit.aftersale.assistant import (
    ACTION_LABELS,
    MSG_MANAGER,
    MSG_NO_PORT,
    MSG_PDD_REFUSE,
    MSG_STALE,
    MSG_TAOBAO_REFUSE,
)
from rodit.aftersale.durations import (
    DECISION_STATES,
    DurationSummary,
    decision_times,
    first_decision,
    merge_decisions,
    review_duration,
    summarize,
    waiting_seconds,
)
from rodit.aftersale.evidence_vision import EVIDENCE_BATCH_BUDGET_S, evidence_batch
from rodit.app.service import aftersale_scopes
from rodit.app.unit import Unit
from rodit.core.audit import AuditAction
from rodit.core.enums import (
    ACTIONABLE_STATUSES,
    AfterSaleAction,
    AfterSaleReason,
    AfterSaleStatus,
    AfterSaleType,
    Platform,
    ReviewAdvice,
    ReviewState,
    is_actionable,
)
from rodit.core.errors import (
    AuthExpired,
    RoditError,
    ShopInactive,
    TenantIsolationError,
    ToolUnavailable,
)
from rodit.core.evidence import PLATFORM_NAMES
from rodit.core.models import AfterSaleInfo, AfterSaleReview, strip_control_chars
from rodit.core.pii import mask_pii

# ================================================================ 常量与中文名

DEFAULT_QUEUE_LIMIT = 50
MAX_QUEUE_LIMIT = 200
DEFAULT_BATCH = 20
MAX_BATCH = 50
MAX_NOTE_CHARS = 200
"""列表、详情里买家说明最多显示多少字（打码后截短）。"""

_PENDING = AfterSaleStatus.PENDING_REVIEW
_WRITE_SCOPES = aftersale_scopes(AfterSaleAction.AGREE_REFUND)
"""确认执行要的权限：只读权限 + 写售后（§8.4，二期才申请）。"""

UNREVIEWED = "unreviewed"
"""列表筛选：平台上待审核、本地还没跑规则的售后单。"""

QUEUE_FILTERS: tuple[str, ...] = (UNREVIEWED, *(s.value for s in ReviewState))
"""列表 state 参数的取值：unreviewed 或审核状态（§7.3.2 状态机）。不给时列出还没结束的。"""

OPEN_STATES: tuple[ReviewState, ...] = tuple(s for s in ReviewState if s not in TERMINAL_STATES)
"""还没结束的审核状态（客服的工作量）。"""

REVIEW_STATE_LABELS: dict[ReviewState, str] = {
    ReviewState.NEW: "待审核",
    ReviewState.SUGGEST: "规则建议",
    ReviewState.VERIFY: "签收核查中",
    ReviewState.EVIDENCE: "等待凭证",
    ReviewState.CONFIRM: "待客服确认",
    ReviewState.NEGOTIATE: "协商中",
    ReviewState.WAIT_RETURN: "退货中",
    ReviewState.AGREED: "已同意",
    ReviewState.REFUSED: "已拒绝",
    ReviewState.PLATFORM: "平台已处理",
}
"""§7.3.2 审核状态机各状态的中文名（退货中 = 已同意退货，等买家寄回 / 仓库确认收货）。"""

STATE_FLOW: tuple[ReviewState, ...] = (
    ReviewState.NEW,
    ReviewState.SUGGEST,
    ReviewState.VERIFY,
    ReviewState.EVIDENCE,
    ReviewState.CONFIRM,
    ReviewState.NEGOTIATE,
    ReviewState.WAIT_RETURN,
    ReviewState.AGREED,
    ReviewState.REFUSED,
    ReviewState.PLATFORM,
)
"""网页状态条的顺序（§7.3.2 状态机图从左到右）。"""

ADVICE_LABELS: dict[ReviewAdvice, str] = {
    ReviewAdvice.AGREE: "建议同意",
    ReviewAdvice.AGREE_RETURN: "建议同意退货",
    ReviewAdvice.REFUSE: "建议拒绝",
    ReviewAdvice.NEGOTIATE: "建议先协商",
    ReviewAdvice.MANUAL: "人工决定",
    ReviewAdvice.NONE: "不操作，只同步状态",
}


class RuleInfo(BaseModel):
    """§7.3.2 审核规则表的一行。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: str
    advice: str


RULES: dict[str, RuleInfo] = {
    "R1": RuleInfo(condition="售后单已不是待审核状态", advice="不操作，同步状态"),
    "R2": RuleInfo(
        condition="订单未发货，申请仅退款",
        advice="建议同意（平台自身可能也有自动处理机制，以平台为准）",
    ),
    "R3": RuleInfo(
        condition="签收 ≤ 7 天，商品支持七天无理由，申请退货退款",
        advice="建议同意退货；仓库确认收到后同意退款",
    ),
    "R4": RuleInfo(
        condition="物流显示签收，买家称没收到",
        advice="建议先协商：提供驿站或快递柜信息，发起快递核查；24 小时后仍未找到，建议同意",
    ),
    "R5": RuleInfo(condition="质量问题、破损、错漏发", advice="需要凭证；客服查看后决定"),
    "R6": RuleInfo(condition="金额超过店铺设定的上限", advice="必须店长确认"),
    "R7": RuleInfo(condition="同一买家 30 天内售后 ≥ 3 次", advice="标记风险，人工决定"),
    "R8": RuleInfo(condition="以上都不命中", advice="人工决定"),
}
"""§7.3.2 审核规则 R1–R8（条件和建议照抄方案；具体天数、小时数以店铺生效配置为准）。"""

PLATFORM_ACTIONS: dict[Platform, frozenset[AfterSaleAction]] = {
    Platform.TAOBAO: frozenset(AfterSaleAction),
    Platform.DOUYIN: frozenset(AfterSaleAction),
    Platform.PDD: frozenset({AfterSaleAction.AGREE_REFUND, AfterSaleAction.AGREE_RETURN}),
}
"""各平台开放的售后写接口（§7.1：拼多多没有查到拒绝接口）。
单元接了写接口时以它的 supported 为准。"""

PLATFORM_NOTES: dict[Platform, str] = {
    Platform.TAOBAO: (
        "淘宝：同意退款要子账号有权限，默认还要短信验证；2025-10-15 起部分场景要先协商才能拒绝，"
        "拒绝率明显偏高会被阻断拒绝接口。拒绝要非常保守，协商优先。"
    ),
    Platform.DOUYIN: (
        "抖店：体验分考核「售后平均审核时长」，及时处理能直接改善这个指标；拒绝时必须传拒绝原因码。"
    ),
    Platform.PDD: "拼多多：没有查到拒绝售后接口，拒绝只能在商家后台手动操作【待确认】。",
}
"""§7.3.2 "平台差异要写进规则"，工作台详情里原样提示。"""

REFUSE_HINTS: dict[Platform, str] = {
    Platform.TAOBAO: "协商优先：先选「协商」，协商没有结果再考虑拒绝（§7.3.2）。",
    Platform.DOUYIN: "抖店拒绝必须填写拒绝原因码（以抖店开放平台的原因码为准）。",
    Platform.PDD: MSG_PDD_REFUSE,
}

MSG_NO_WRITE_SCOPE = "店铺授权没有写售后权限（refund.write，二期才申请），请在平台后台操作"
MSG_NOT_REVIEWED = "还没跑规则：先「拉取并跑规则」"
MSG_PLATFORM_UNAVAILABLE = "平台接口暂时不可用，显示的是缓存数据；刷新后再确认"
MSG_PLATFORM_GONE = "售后单在平台已查不到，请刷新同步状态"
MSG_RETURN_FOR_REFUND_ONLY = "仅退款的售后单不需要同意退货，请选择同意退款"


class NextStep(StrEnum):
    """列表里每张售后单下一步该做什么。"""

    RUN_RULES = "run_rules"  # 还没跑规则
    CONFIRM = "confirm"  # 待客服确认
    REFRESH = "refresh"  # 平台变了 / 签收核查满时 / 凭证到了 / 买家寄回了：刷新推进状态机
    NEGOTIATING = "negotiating"  # 协商中：有结果后点"协商有结果"
    WAIT_VERIFY = "wait_verify"  # 签收核查中，还没满时
    WAIT_EVIDENCE = "wait_evidence"  # 等买家补凭证
    WAIT_RETURN = "wait_return"  # 等买家寄回
    DONE = "done"  # 已结束


NEXT_STEP_LABELS: dict[NextStep, str] = {
    NextStep.RUN_RULES: "未跑规则",
    NextStep.CONFIRM: "待客服确认",
    NextStep.REFRESH: "需要刷新",
    NextStep.NEGOTIATING: "协商中，有结果后回来确认",
    NextStep.WAIT_VERIFY: "签收核查中，等核查结果",
    NextStep.WAIT_EVIDENCE: "等买家补凭证",
    NextStep.WAIT_RETURN: "等买家寄回",
    NextStep.DONE: "已结束",
}

NEEDS_ACTION: frozenset[NextStep] = frozenset(
    {NextStep.RUN_RULES, NextStep.CONFIRM, NextStep.REFRESH}
)
"""客服现在就能推进的（列表排在前面）；其余在等买家、等核查、等协商结果。"""


# ================================================================ 结果模型（接口直接返回）


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueueRow(_Model):
    """待审核列表的一行。没跑规则时 reviewed=false，审核相关字段为空。

    platform_status 是这一行用来判断的平台状态：审核记录比平台缓存新时取审核记录同步到的，
    否则取平台数据。平台数据读不到时 platform_unavailable=true，只有审核记录里的字段。
    """

    aftersale_id: str
    order_id: str | None = None
    type: AfterSaleType | None = None
    reason: AfterSaleReason | None = None
    platform_status: AfterSaleStatus | None = None
    amount: Decimal | None = None
    created_at: AwareDatetime | None = None
    waiting_s: int | None = None  # 还没结束的：申请到现在多少秒
    evidence_count: int | None = None
    note: str | None = None  # 买家说明（已打码、截短）
    reviewed: bool = False
    state: ReviewState | None = None
    suggestion: ReviewAdvice | None = None
    reason_text: str | None = None  # 规则给出的理由
    matched_rules: list[str] = Field(default_factory=list)
    requires_manager: bool = False
    risk_flags: list[str] = Field(default_factory=list)
    updated_at: AwareDatetime | None = None
    next_step: NextStep
    needs_action: bool = False
    platform_changed: bool = False
    platform_unavailable: bool = False


class AfterSaleQueue(_Model):
    """待审核列表。total / needs_action 是筛选后、截断（limit）前的条数。"""

    as_of: AwareDatetime
    rows: list[QueueRow] = Field(default_factory=list)
    total: int = 0
    needs_action: int = 0
    unreviewed: int = 0
    platform_unavailable: bool = False  # 平台待审核列表读不到：只列出了本地审核记录


class DeskStats(_Model):
    """工作台统计（§2.1 售后平均审核时长、§12.4）。

    - by_state：还没结束的状态（含 unreviewed 未跑规则）是当前的工作量，不受 since 影响；
      已结束的状态（已同意、已拒绝、平台已处理）只数 since 之后结束的（审核记录的更新时间）。
    - pending_by_rule：还没结束的审核记录各命中了哪些规则（一张单可以命中多条，各算一次）。
    - review_duration：since 之后由人确认执行的售后单，申请 → 第一次确认执行的时长
      （rodit.aftersale.durations 的口径）；decided 是条数，duration_unknown 是平台申请时间读不到、
      没算进时长的。
    - waiting：还没结束的售后单已经等了多久。
    """

    as_of: AwareDatetime
    since: AwareDatetime | None = None
    by_state: dict[str, int] = Field(default_factory=dict)
    open_total: int = 0
    needs_action: int = 0
    pending_by_rule: dict[str, int] = Field(default_factory=dict)
    review_duration: DurationSummary = Field(default_factory=DurationSummary)
    decided: int = 0
    duration_unknown: int = 0
    waiting: DurationSummary = Field(default_factory=DurationSummary)
    platform_unavailable: bool = False


class ActionOption(_Model):
    """详情里一种写操作：能不能做，不能做的原因（只是界面提示，真正的检查在确认接口）。"""

    action: AfterSaleAction
    label: str
    available: bool
    recommended: bool = False  # 和规则建议一致
    blocked_reason: str | None = None
    needs_reason_code: bool = False  # 抖店拒绝要原因码


@dataclass
class DeskDetail:
    """一张售后单的详情（接口层转成 schemas_aftersale.DeskDetailView）。"""

    aftersale_id: str
    as_of: datetime
    review: AfterSaleReview | None
    info: AfterSaleInfo | None
    platform_unavailable: bool  # 平台重读失败（info 可能是缓存数据或为空）
    note: str | None
    waiting_s: int | None
    next_step: NextStep
    actions: list[ActionOption]
    recommended_action: AfterSaleAction | None
    can_run_rules: bool
    can_refresh: bool
    can_negotiate: bool
    can_finish_negotiation: bool
    manager_required: bool
    caller_is_manager: bool  # 调用方能做 R6 店长确认（店长角色且在店铺配置的店长名单里）
    write_enabled: bool  # 单元接了售后写接口
    write_scope: bool  # 店铺授权有写售后权限
    negotiated: bool  # 协商过（淘宝拒绝的前提）
    supported_actions: list[AfterSaleAction]
    platform_note: str
    refuse_hint: str | None


@dataclass
class BatchOutcome:
    """批量跑规则的一行：成功时 review 是审核记录，失败时 error 是异常（接口层转成错误码）。"""

    aftersale_id: str
    review: AfterSaleReview | None = None
    error: Exception | None = None


@dataclass
class BatchResult:
    as_of: datetime
    outcomes: list[BatchOutcome] = field(default_factory=list)
    remaining: int = 0  # 还没跑规则、这次没轮到（超过 limit 或中途停下）的
    stopped: Exception | None = None  # 中途停下的原因（授权失效、店铺停用）


# ================================================================ 纯函数


def masked_note(note: str | None) -> str | None:
    """买家说明 → 可以给客服看的文字：去控制字符、打码（网关已打过，再过一遍兜底）、截短。"""
    if not note:
        return None
    text = mask_pii(strip_control_chars(note)).strip()
    if not text:
        return None
    if len(text) > MAX_NOTE_CHARS:
        text = text[: MAX_NOTE_CHARS - 1] + "…"
    return text


AFTER_ACTION_STATUS: dict[AfterSaleAction, AfterSaleStatus] = {
    AfterSaleAction.AGREE_REFUND: AfterSaleStatus.REFUNDING,
    AfterSaleAction.AGREE_RETURN: AfterSaleStatus.WAIT_BUYER_RETURN,
    AfterSaleAction.REFUSE: AfterSaleStatus.REFUSED,
}
"""平台写操作成功后售后单的平台状态（同意退款 → 退款中、同意退货 → 等买家退货、拒绝 → 已拒绝）。"""


def believed_status(review: AfterSaleReview) -> tuple[AfterSaleStatus, str | None]:
    """审核记录认为的平台状态和状态版本。

    客服确认执行成功后，审核记录里同步到的是执行前重读的状态（审核助手执行后不再重读平台）：
    这时按执行的操作推出执行后的状态，状态版本未知（None，不拿来比较）。"""
    result = last_result(review)
    if (
        review.state in DECISION_STATES
        and review.confirmed_by
        and result is not None
        and result.ok
        and result.action in AFTER_ACTION_STATUS
    ):
        return AFTER_ACTION_STATUS[result.action], None
    return review.platform_status, review.platform_status_version


def review_is_newer(review: AfterSaleReview, info: AfterSaleInfo) -> bool:
    """审核记录比这份平台数据新（或同时）：取审核记录认为的平台状态（believed_status）。

    平台数据没有获取时间（没经过缓存网关）时当作刚读到的，比审核记录新。"""
    if review.updated_at is None:
        return False
    return info.fetched_at is not None and review.updated_at >= info.fetched_at


def platform_changed(review: AfterSaleReview, info: AfterSaleInfo) -> bool:
    """平台数据比审核记录新，且状态（或已知的状态版本）和审核记录认为的不一样。"""
    if review_is_newer(review, info):
        return False
    status, version = believed_status(review)
    return info.status != status or (version is not None and info.status_version != version)


def effective_status(
    review: AfterSaleReview | None, info: AfterSaleInfo | None
) -> AfterSaleStatus | None:
    """判断用的平台状态：审核记录和平台数据哪个新取哪个。"""
    if review is None:
        return info.status if info is not None else None
    if info is None or review_is_newer(review, info):
        return believed_status(review)[0]
    return info.status


def verify_due(review: AfterSaleReview, now: datetime, verify_hours: int) -> bool:
    """R4 签收核查已满 verify_hours（刷新后审核助手会给出建议）。"""
    started = review.verify_started_at
    return started is not None and now - started >= timedelta(hours=verify_hours)


def next_step(
    review: AfterSaleReview | None,
    info: AfterSaleInfo | None,
    *,
    now: datetime,
    verify_hours: int,
    platform_missing: bool = False,
) -> NextStep:
    """这张售后单下一步该做什么（列表排序和提示用）。

    platform_missing：平台读成功了，但售后单已经查不到（刷新后审核助手会同步成"平台已处理"）。"""
    if review is None:
        return NextStep.RUN_RULES
    state = review.state
    if state in TERMINAL_STATES:
        return NextStep.DONE
    if state in (ReviewState.NEW, ReviewState.SUGGEST) or platform_missing:
        return NextStep.REFRESH
    changed = info is not None and platform_changed(review, info)
    status = effective_status(review, info)
    waiting_return = (
        state == ReviewState.WAIT_RETURN and status == AfterSaleStatus.WAIT_BUYER_RETURN
    )
    if changed and status is not None and not is_actionable(status) and not waiting_return:
        return NextStep.REFRESH  # 平台已处理（买家撤销、平台自动处理）：刷新同步状态
    if state == ReviewState.CONFIRM:
        return NextStep.REFRESH if changed else NextStep.CONFIRM
    if state == ReviewState.VERIFY:
        return NextStep.REFRESH if verify_due(review, now, verify_hours) else NextStep.WAIT_VERIFY
    if state == ReviewState.EVIDENCE:
        has_evidence = changed and info is not None and info.evidence_count > 0
        return NextStep.REFRESH if has_evidence else NextStep.WAIT_EVIDENCE
    if state == ReviewState.WAIT_RETURN:
        return NextStep.WAIT_RETURN if waiting_return else NextStep.REFRESH
    return NextStep.NEGOTIATING


def recommended_action(review: AfterSaleReview | None) -> AfterSaleAction | None:
    """规则建议对应的写操作：建议同意退货 → 同意退货；建议同意 → 同意退款；其余没有。"""
    if review is None or review.state != ReviewState.CONFIRM:
        return None
    if review.suggestion == ReviewAdvice.AGREE_RETURN:
        return AfterSaleAction.AGREE_RETURN
    if review.suggestion == ReviewAdvice.AGREE:
        return AfterSaleAction.AGREE_REFUND
    return None


def _sort_key(row: QueueRow) -> tuple[int, int, float, str]:
    """要处理的在前，再按申请时间从早到晚（等得最久的先处理，§2.1 审核时长）；没有时间的排后。"""
    created = row.created_at
    return (
        0 if row.needs_action else 1,
        0 if created is not None else 1,
        created.timestamp() if created is not None else 0.0,
        row.aftersale_id,
    )


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


# ================================================================ 服务


class AfterSaleDesk:
    """售后审核工作台的应用服务。review：AssistService.review_aftersale（批量跑规则逐张调用）。

    读操作不写任何东西；批量跑规则的每一张都走 review（店铺闸门、授权、审计都在里面）。
    """

    def __init__(
        self,
        unit: Unit,
        review: Callable[[str, str], AfterSaleReview],
        *,
        vision_budget_s: float = EVIDENCE_BATCH_BUDGET_S,
    ) -> None:
        self.unit = unit
        self._review = review
        self._vision_budget_s = vision_budget_s  # 批量跑规则时凭证识别共用的预算（秒）

    # ---------------------------------------------------------------- 列表

    def queue(
        self,
        shop_id: str,
        *,
        state: str | None = None,
        status: AfterSaleStatus | None = None,
        rule: str | None = None,
        limit: int = DEFAULT_QUEUE_LIMIT,
    ) -> AfterSaleQueue:
        """待审核列表（只读）。state 不给时列出还没结束的（含未跑规则）；
        给 unreviewed 只列未跑规则的；给已结束的状态时按最近更新在前列出那些审核记录。"""
        if state is not None and state not in QUEUE_FILTERS:
            raise ValueError(f"不认识的状态筛选：{state}")
        if not 1 <= limit <= MAX_QUEUE_LIMIT:
            raise ValueError(f"limit 要在 1–{MAX_QUEUE_LIMIT} 之间")
        self.unit.auth.require(shop_id)
        now = self.unit.clock.now()
        closed = state not in (None, UNREVIEWED) and ReviewState(state) in TERMINAL_STATES
        if closed:
            return self._closed_queue(shop_id, ReviewState(state), status, rule, limit, now)
        rows, list_failed = self._open_rows(shop_id, now)
        if state == UNREVIEWED:
            rows = [r for r in rows if not r.reviewed]
        elif state is not None:
            rows = [r for r in rows if r.state == ReviewState(state)]
        rows = [r for r in rows if _matches(r, status, rule)]
        rows.sort(key=_sort_key)
        return AfterSaleQueue(
            as_of=now,
            rows=rows[:limit],
            total=len(rows),
            needs_action=sum(1 for r in rows if r.needs_action),
            unreviewed=sum(1 for r in rows if not r.reviewed),
            platform_unavailable=list_failed,
        )

    def _closed_queue(
        self,
        shop_id: str,
        state: ReviewState,
        status: AfterSaleStatus | None,
        rule: str | None,
        limit: int,
        now: datetime,
    ) -> AfterSaleQueue:
        """已结束的审核记录：按审核记录筛选、最近更新在前，只给显示出来的那几行读平台数据。"""
        reviews = [
            r
            for r in self.unit.repos.reviews.list_by_state(shop_id, [state])
            if r.shop_id == shop_id
            and r.state == state
            and (status is None or believed_status(r)[0] == status)
            and (rule is None or rule in r.matched_rules)
        ]
        reviews.sort(key=lambda r: (r.updated_at or _EPOCH, r.aftersale_id), reverse=True)
        rows = []
        for rec in reviews[:limit]:
            info, unavailable = self._cached_info(shop_id, rec.aftersale_id)
            rows.append(
                self._row(rec.aftersale_id, rec, info, now, unavailable=unavailable, missing=False)
            )
        return AfterSaleQueue(as_of=now, rows=rows, total=len(reviews))

    def _open_rows(self, shop_id: str, now: datetime) -> tuple[list[QueueRow], bool]:
        """平台待审核 + 本地还没结束的审核记录（已结束的不列）。返回（行，平台列表是否读不到）。"""
        repo = self.unit.repos.reviews
        pending: dict[str, AfterSaleInfo] = {}
        list_failed = False
        try:
            listed = self.unit.gateway.list_aftersales(shop_id, status=_PENDING)
        except ToolUnavailable:
            listed, list_failed = [], True
        for info in listed:
            if info.shop_id == shop_id:  # 纵深防御：网关已按店铺隔离（§9.2）
                pending.setdefault(info.aftersale_id, info)
        open_reviews = {
            r.aftersale_id: r
            for r in repo.list_by_state(shop_id, OPEN_STATES)
            if r.shop_id == shop_id and r.state in OPEN_STATES
        }
        rows: list[QueueRow] = []
        for aftersale_id, info in pending.items():
            rec = open_reviews.pop(aftersale_id, None)
            if rec is None:
                rec = repo.get(shop_id, aftersale_id)
                if rec is not None and (rec.shop_id != shop_id or rec.state in TERMINAL_STATES):
                    continue  # 已经处理完（缓存里的平台列表还没更新）
            rows.append(self._row(aftersale_id, rec, info, now, unavailable=False, missing=False))
        for aftersale_id, rec in open_reviews.items():  # 平台待审核列表里没有的（退货中等）
            info, unavailable = self._cached_info(shop_id, aftersale_id)
            missing = not unavailable and info is None
            rows.append(
                self._row(aftersale_id, rec, info, now, unavailable=unavailable, missing=missing)
            )
        return rows, list_failed

    def _cached_info(self, shop_id: str, aftersale_id: str) -> tuple[AfterSaleInfo | None, bool]:
        """带缓存地读一张售后单。返回（售后单，是否读不到）；别的店铺的数据丢掉。"""
        try:
            info = self.unit.gateway.get_aftersale(shop_id, aftersale_id)
        except ToolUnavailable:
            return None, True
        if info is not None and info.shop_id != shop_id:
            return None, True
        return info, False

    def _row(
        self,
        aftersale_id: str,
        rec: AfterSaleReview | None,
        info: AfterSaleInfo | None,
        now: datetime,
        *,
        unavailable: bool,
        missing: bool,
    ) -> QueueRow:
        step = next_step(
            rec,
            info,
            now=now,
            verify_hours=self.unit.config.aftersale.verify_hours,
            platform_missing=missing,
        )
        changed = missing or (rec is not None and info is not None and platform_changed(rec, info))
        open_ = step != NextStep.DONE
        row = QueueRow(
            aftersale_id=aftersale_id,
            platform_status=effective_status(rec, info),
            next_step=step,
            needs_action=step in NEEDS_ACTION,
            platform_changed=changed and rec is not None and rec.state not in TERMINAL_STATES,
            platform_unavailable=unavailable,
        )
        if info is not None:
            row = row.model_copy(
                update={
                    "order_id": info.order_id,
                    "type": info.type,
                    "reason": info.reason,
                    "amount": info.amount,
                    "created_at": info.created_at,
                    "waiting_s": waiting_seconds(info.created_at, now) if open_ else None,
                    "evidence_count": info.evidence_count,
                    "note": masked_note(info.note),
                }
            )
        if rec is not None:
            row = row.model_copy(
                update={
                    "reviewed": True,
                    "state": rec.state,
                    "suggestion": rec.suggestion,
                    "reason_text": rec.reason,
                    "matched_rules": list(rec.matched_rules),
                    "requires_manager": rec.requires_manager,
                    "risk_flags": list(rec.risk_flags),
                    "updated_at": rec.updated_at,
                }
            )
        return row

    # ---------------------------------------------------------------- 批量跑规则

    def review_pending(self, shop_id: str, *, limit: int = DEFAULT_BATCH) -> BatchResult:
        """平台上待审核、还没有建议的售后单，按申请时间从早到晚最多跑 limit 张（§7.3.2 拉取售后单
        并跑规则）。平台待审核列表读不到时抛 ToolUnavailable（不知道该跑哪些）。"""
        if not 1 <= limit <= MAX_BATCH:
            raise ValueError(f"limit 要在 1–{MAX_BATCH} 之间")
        self.unit.auth.require(shop_id)
        now = self.unit.clock.now()
        listed = self.unit.gateway.list_aftersales(shop_id, status=_PENDING)
        repo = self.unit.repos.reviews
        targets: list[str] = []
        for info in sorted(listed, key=lambda a: (a.created_at, a.aftersale_id)):
            if info.shop_id != shop_id or info.aftersale_id in targets:
                continue
            rec = repo.get(shop_id, info.aftersale_id)
            if rec is None or rec.suggestion is None:
                targets.append(info.aftersale_id)
        result = BatchResult(as_of=now)
        chosen = targets[:limit]
        result.remaining = len(targets) - len(chosen)
        with evidence_batch(self._vision_budget_s):  # 凭证识别这一批共用预算、超时一次就熔断
            for i, aftersale_id in enumerate(chosen):
                try:
                    review = self._review(shop_id, aftersale_id)
                    result.outcomes.append(BatchOutcome(aftersale_id, review=review))
                except (AuthExpired, ShopInactive) as exc:
                    result.outcomes.append(BatchOutcome(aftersale_id, error=exc))
                    result.stopped = exc
                    result.remaining += len(chosen) - i - 1
                    break
                except (RoditError, KeyError) as exc:
                    result.outcomes.append(BatchOutcome(aftersale_id, error=exc))
        return result

    # ---------------------------------------------------------------- 详情

    def detail(
        self, shop_id: str, aftersale_id: str, *, operator: str, is_manager: bool
    ) -> DeskDetail:
        """一张售后单的详情（只读）。售后单在平台和本地都没有时抛 KeyError（404）；
        平台读不到、本地也没有审核记录时抛 ToolUnavailable（503，不知道有没有这张单）。"""
        unit = self.unit
        unit.auth.require(shop_id)
        now = unit.clock.now()
        rec = unit.repos.reviews.get(shop_id, aftersale_id)
        if rec is not None and rec.shop_id != shop_id:
            raise TenantIsolationError(f"审核记录不属于店铺 {shop_id}（§9.2）")
        info: AfterSaleInfo | None = None
        unavailable = False
        try:
            info = unit.tools.get_aftersale(shop_id, aftersale_id)  # 不走缓存（§7.3.2 重新读取）
        except ToolUnavailable:
            unavailable = True
            try:
                info = unit.gateway.get_aftersale(shop_id, aftersale_id)  # 只用来显示
            except ToolUnavailable:
                info = None
        if info is not None and info.shop_id != shop_id:
            raise TenantIsolationError(f"售后单 {aftersale_id} 不属于店铺 {shop_id}（§9.2）")
        if rec is None and info is None:
            if unavailable:
                raise ToolUnavailable("平台售后单暂时读不到")
            raise KeyError(f"售后单 {aftersale_id} 不存在或不属于店铺 {shop_id}")

        shop_config = unit.shop_config(shop_id)
        caller_is_manager = is_manager and operator in shop_config.manager_ids
        port = unit.action_port
        supported = port.supported if port is not None else PLATFORM_ACTIONS[unit.platform]
        write_scope = unit.auth.is_authorized(shop_id, _WRITE_SCOPES)
        manager_required = bool(
            (rec is not None and rec.requires_manager)
            or (info is not None and info.amount > shop_config.aftersale_amount_limit)
        )
        missing = not unavailable and info is None
        step = next_step(
            rec,
            None if unavailable else info,
            now=now,
            verify_hours=unit.config.aftersale.verify_hours,
            platform_missing=missing,
        )
        negotiated = rec is not None and was_negotiated(rec)
        context = _ActionContext(
            platform=unit.platform,
            review=rec,
            info=info,
            unavailable=unavailable,
            supported=supported,
            write_enabled=port is not None,
            write_scope=write_scope,
            manager_required=manager_required,
            caller_is_manager=caller_is_manager,
            negotiated=negotiated,
        )
        recommended = recommended_action(rec)
        actions = [
            _action_option(context, action, recommended=action == recommended)
            for action in AfterSaleAction
        ]
        open_ = rec is None or rec.state not in TERMINAL_STATES
        return DeskDetail(
            aftersale_id=aftersale_id,
            as_of=now,
            review=rec,
            info=info,
            platform_unavailable=unavailable,
            note=masked_note(info.note) if info is not None else None,
            waiting_s=waiting_seconds(info.created_at, now) if info is not None and open_ else None,
            next_step=step,
            actions=actions,
            recommended_action=recommended,
            can_run_rules=rec is None and info is not None and not unavailable,
            can_refresh=rec is not None and rec.state not in TERMINAL_STATES,
            can_negotiate=rec is not None and rec.state == ReviewState.CONFIRM,
            can_finish_negotiation=rec is not None and rec.state == ReviewState.NEGOTIATE,
            manager_required=manager_required,
            caller_is_manager=caller_is_manager,
            write_enabled=port is not None,
            write_scope=write_scope,
            negotiated=negotiated,
            supported_actions=[a for a in AfterSaleAction if a in supported],
            platform_note=PLATFORM_NOTES[unit.platform],
            refuse_hint=REFUSE_HINTS.get(unit.platform),
        )

    # ---------------------------------------------------------------- 统计

    def stats(self, shop_id: str, *, since: datetime | None = None) -> DeskStats:
        """工作台统计（只读，口径见 DeskStats）。"""
        unit = self.unit
        unit.auth.require(shop_id)
        now = unit.clock.now()
        rows, list_failed = self._open_rows(shop_id, now)
        by_state: dict[str, int] = {UNREVIEWED: 0, **{s.value: 0 for s in ReviewState}}
        pending_by_rule: dict[str, int] = {rule: 0 for rule in RULES}
        for row in rows:
            key = row.state.value if row.state is not None else UNREVIEWED
            by_state[key] += 1
            for rule in row.matched_rules:
                pending_by_rule[rule] = pending_by_rule.get(rule, 0) + 1
        reviews = [r for r in unit.repos.reviews.list_by_state(shop_id) if r.shop_id == shop_id]
        recent = [r for r in reviews if since is None or (r.updated_at and r.updated_at >= since)]
        for rec in recent:
            if rec.state in TERMINAL_STATES:
                by_state[rec.state.value] += 1

        events = unit.repos.audit.list_events(
            shop_id, action=AuditAction.REVIEW_ACTION.value, since=since
        )
        from_history: dict[str, datetime] = {}
        for rec in recent:
            at = first_decision(history(rec))
            if at is not None:
                from_history[rec.aftersale_id] = at
        decided = {
            aftersale_id: at
            for aftersale_id, at in merge_decisions(
                decision_times(e for e in events if e.shop_id == shop_id), from_history
            ).items()
            if since is None or at >= since
        }
        durations: list[timedelta] = []
        unknown = 0
        for aftersale_id, at in sorted(decided.items()):
            info, unavailable = self._cached_info(shop_id, aftersale_id)
            if info is None or unavailable:
                unknown += 1
                continue
            durations.append(review_duration(info.created_at, at))
        waits = [timedelta(seconds=r.waiting_s) for r in rows if r.waiting_s is not None]
        return DeskStats(
            as_of=now,
            since=since,
            by_state=by_state,
            open_total=len(rows),
            needs_action=sum(1 for r in rows if r.needs_action),
            pending_by_rule=pending_by_rule,
            review_duration=summarize(durations),
            decided=len(decided),
            duration_unknown=unknown,
            waiting=summarize(waits),
            platform_unavailable=list_failed,
        )


# ================================================================ 内部


def _matches(row: QueueRow, status: AfterSaleStatus | None, rule: str | None) -> bool:
    return (status is None or row.platform_status == status) and (
        rule is None or rule in row.matched_rules
    )


@dataclass(frozen=True)
class _ActionContext:
    platform: Platform
    review: AfterSaleReview | None
    info: AfterSaleInfo | None
    unavailable: bool
    supported: frozenset[AfterSaleAction]
    write_enabled: bool
    write_scope: bool
    manager_required: bool
    caller_is_manager: bool
    negotiated: bool


def _blocked_reason(ctx: _ActionContext, action: AfterSaleAction) -> str | None:
    """这种写操作现在不能做的原因（第一条命中的）；能做时为 None。顺序和确认时的检查一致：
    平台开放的接口 → 写接口 / 写权限 → 审核状态 → 平台状态 → R6 店长 → 平台规则（淘宝先协商）。"""
    label = ACTION_LABELS[action]
    if action not in ctx.supported:
        if action == AfterSaleAction.REFUSE and ctx.platform == Platform.PDD:
            return MSG_PDD_REFUSE
        name = PLATFORM_NAMES.get(ctx.platform, str(ctx.platform))
        return f"{name}不支持{label}，请在商家后台手动操作"
    if not ctx.write_enabled:
        return MSG_NO_PORT
    if not ctx.write_scope:
        return MSG_NO_WRITE_SCOPE
    rec, info = ctx.review, ctx.info
    if rec is None:
        return MSG_NOT_REVIEWED
    if rec.state != ReviewState.CONFIRM:
        return f"当前是「{REVIEW_STATE_LABELS[rec.state]}」，只有待客服确认时才能执行"
    if ctx.unavailable:
        return MSG_PLATFORM_UNAVAILABLE
    if info is None:
        return MSG_PLATFORM_GONE
    if info.status not in ACTIONABLE_STATUSES[action]:
        return f"售后单当前是「{AFTERSALE_STATUS_LABELS[info.status]}」，不能{label}"
    if action == AfterSaleAction.AGREE_RETURN and info.type == AfterSaleType.REFUND_ONLY:
        return MSG_RETURN_FOR_REFUND_ONLY
    if info.status != rec.platform_status or info.status_version != rec.platform_status_version:
        return MSG_STALE
    if ctx.manager_required and not ctx.caller_is_manager:
        return MSG_MANAGER
    if action == AfterSaleAction.REFUSE and ctx.platform == Platform.TAOBAO and not ctx.negotiated:
        return MSG_TAOBAO_REFUSE
    return None


def _action_option(
    ctx: _ActionContext, action: AfterSaleAction, *, recommended: bool
) -> ActionOption:
    reason = _blocked_reason(ctx, action)
    return ActionOption(
        action=action,
        label=ACTION_LABELS[action],
        available=reason is None,
        recommended=recommended,
        blocked_reason=reason,
        needs_reason_code=action == AfterSaleAction.REFUSE and ctx.platform == Platform.DOUYIN,
    )


__all__ = [
    "ADVICE_LABELS",
    "DEFAULT_BATCH",
    "DEFAULT_QUEUE_LIMIT",
    "MAX_BATCH",
    "MAX_QUEUE_LIMIT",
    "NEEDS_ACTION",
    "NEXT_STEP_LABELS",
    "OPEN_STATES",
    "PLATFORM_ACTIONS",
    "PLATFORM_NOTES",
    "QUEUE_FILTERS",
    "REFUSE_HINTS",
    "REVIEW_STATE_LABELS",
    "RULES",
    "STATE_FLOW",
    "UNREVIEWED",
    "ActionOption",
    "AfterSaleDesk",
    "AfterSaleQueue",
    "BatchOutcome",
    "BatchResult",
    "DeskDetail",
    "DeskStats",
    "NextStep",
    "QueueRow",
    "RuleInfo",
    "effective_status",
    "masked_note",
    "next_step",
    "platform_changed",
    "recommended_action",
    "review_is_newer",
    "verify_due",
]
