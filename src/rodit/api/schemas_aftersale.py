"""售后审核工作台接口的请求和响应模型（方案 §7.3.2 商家侧审核助手、§15 Phase 2、
§2.1 售后平均审核时长）。

列表（AfterSaleQueue）和统计（DeskStats）直接用 rodit.app.aftersale_desk 的结果模型；
这里是批量跑规则的请求和结果、单张详情。详情沿用 rodit.api.schemas 的审核视图
（AfterSaleReviewView）和平台售后单快照（AfterSaleSnapshotView，确认时带上 status_version）。

最小化（§5.3、§12.3）：没有 buyer_ref、平台原始买家 ID；买家说明已打码、截短。
错误码和统一错误格式（rodit.api.errors）一致：批量跑规则里单张失败时，那一行带错误码和说明。
"""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.api.errors import MSG_INTERNAL, MSG_NOT_FOUND, MSG_TENANT, safe_message
from rodit.api.schemas import AfterSaleReviewView, AfterSaleSnapshotView
from rodit.app.aftersale_desk import (
    DEFAULT_BATCH,
    MAX_BATCH,
    ActionOption,
    BatchResult,
    DeskDetail,
    NextStep,
)
from rodit.core.enums import AfterSaleAction, ReviewAdvice, ReviewState
from rodit.core.errors import (
    ActionNotAllowed,
    AuthExpired,
    ConcurrencyConflict,
    ShopInactive,
    StorageUnavailable,
    TenantIsolationError,
    ToolUnavailable,
)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ================================================================ 批量跑规则


class ReviewPendingRequest(_Request):
    """批量"拉取并跑规则"（§7.3.2）：平台上待审核、还没有建议的售后单，按申请时间从早到晚最多跑
    limit 张。请求体可以省略（默认 20 张）；带操作人等字段是 422。"""

    limit: int = Field(default=DEFAULT_BATCH, ge=1, le=MAX_BATCH)


class BatchRowView(_View):
    """批量跑规则的一行：成功时是规则结果；失败时 error 是错误码（同统一错误格式），
    message 是说明。"""

    aftersale_id: str
    ok: bool
    state: ReviewState | None = None
    suggestion: ReviewAdvice | None = None
    matched_rules: list[str] = Field(default_factory=list)
    requires_manager: bool = False
    error: str | None = None
    message: str | None = None


class BatchReviewView(_View):
    """批量跑规则的结果。remaining：还没跑规则、这次没轮到的张数；stopped：中途停下的原因
    （授权失效、店铺停用），之后的没有跑。"""

    as_of: AwareDatetime
    rows: list[BatchRowView] = Field(default_factory=list)
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    remaining: int = 0
    stopped: str | None = None

    @classmethod
    def from_result(cls, result: BatchResult) -> BatchReviewView:
        rows: list[BatchRowView] = []
        for outcome in result.outcomes:
            review = outcome.review
            if review is not None:
                rows.append(
                    BatchRowView(
                        aftersale_id=outcome.aftersale_id,
                        ok=True,
                        state=review.state,
                        suggestion=review.suggestion,
                        matched_rules=list(review.matched_rules),
                        requires_manager=review.requires_manager,
                    )
                )
                continue
            code, message = error_of(outcome.error)
            rows.append(
                BatchRowView(
                    aftersale_id=outcome.aftersale_id, ok=False, error=code, message=message
                )
            )
        succeeded = sum(1 for r in rows if r.ok)
        return cls(
            as_of=result.as_of,
            rows=rows,
            attempted=len(rows),
            succeeded=succeeded,
            failed=len(rows) - succeeded,
            remaining=result.remaining,
            stopped=error_of(result.stopped)[1] if result.stopped is not None else None,
        )


# (异常类型, 错误码, 是否用异常自带的说明, 固定说明)：和 rodit.api.errors 的映射一致
_ERRORS: tuple[tuple[type[BaseException], str, bool, str], ...] = (
    (ToolUnavailable, "tool_unavailable", False, "平台接口暂时不可用，请稍后重试"),
    (ConcurrencyConflict, "conflict", True, "数据已被修改，请刷新后重试"),
    (ActionNotAllowed, "action_not_allowed", True, "操作不被允许"),
    (AuthExpired, "shop_not_authorized", True, "店铺没有授权或授权已过期（§8.4）"),
    (ShopInactive, "shop_inactive", True, "店铺不存在或已停用（§8.4）"),
    (TenantIsolationError, "tenant_isolation", False, MSG_TENANT),
    (StorageUnavailable, "storage_unavailable", False, "存储暂时不可用，请稍后重试"),
    (KeyError, "not_found", False, MSG_NOT_FOUND),
)


def error_of(exc: BaseException | None) -> tuple[str, str]:
    """异常 → (错误码, 可以给调用方看的说明)。说明打码、截短；存储、内部错误用固定说明。"""
    for exc_type, code, use_message, fallback in _ERRORS:
        if isinstance(exc, exc_type):
            return code, safe_message(str(exc), fallback) if use_message else fallback
    return "internal_error", MSG_INTERNAL


# ================================================================ 详情


class TicketView(_View):
    """平台售后单的申请信息（快照里没有的部分）。note 是买家说明（网关已打码，这里再打码、截短）；
    fetched_at 是平台数据的获取时间（平台读不到、显示缓存时据此写"截至"）。"""

    order_id: str
    created_at: AwareDatetime
    updated_at: AwareDatetime
    waiting_s: int | None = None
    note: str | None = None
    fetched_at: AwareDatetime | None = None


class DeskDetailView(_View):
    """一张售后单的详情：审核记录（没跑规则时为空）、平台当前状态（重读，确认时带 status_version）、
    申请信息、能做哪些操作和不能做的原因、平台差异提示。

    - actions：同意退款 / 同意退货 / 拒绝，各自能不能做（blocked_reason 是原因），recommended 表示
      和规则建议一致；needs_reason_code：抖店拒绝要原因码。只是界面提示，真正的检查在确认接口。
    - manager_required：R6 金额超过上限，必须店长确认；caller_is_manager：调用方能做店长确认
      （店长角色且在店铺配置的店长名单里）。
    - write_enabled：单元接了售后写接口；write_scope：店铺授权有写售后权限（refund.write）。
      MVP 两项都没有，只给建议（§8.4、D6）。
    - platform_unavailable：平台重读失败，platform / ticket 可能是缓存数据或为空，此时不能确认。
    """

    aftersale_id: str
    as_of: AwareDatetime
    review: AfterSaleReviewView | None = None
    platform: AfterSaleSnapshotView | None = None
    ticket: TicketView | None = None
    platform_unavailable: bool = False
    next_step: NextStep
    actions: list[ActionOption] = Field(default_factory=list)
    recommended_action: AfterSaleAction | None = None
    can_run_rules: bool = False
    can_refresh: bool = False
    can_negotiate: bool = False
    can_finish_negotiation: bool = False
    manager_required: bool = False
    caller_is_manager: bool = False
    write_enabled: bool = False
    write_scope: bool = False
    negotiated: bool = False
    supported_actions: list[AfterSaleAction] = Field(default_factory=list)
    platform_note: str
    refuse_hint: str | None = None

    @classmethod
    def from_detail(cls, detail: DeskDetail) -> DeskDetailView:
        info = detail.info
        review = detail.review
        # 审核视图里的 platform（待客服确认时附上）用这次重读到的；
        # 读不到时不附（不拿缓存冒充当前状态）
        fresh = info if info is not None and not detail.platform_unavailable else None
        return cls(
            aftersale_id=detail.aftersale_id,
            as_of=detail.as_of,
            review=(
                AfterSaleReviewView.from_review(
                    review, platform=fresh if review.state == ReviewState.CONFIRM else None
                )
                if review is not None
                else None
            ),
            platform=AfterSaleSnapshotView.from_info(info) if info is not None else None,
            ticket=(
                TicketView(
                    order_id=info.order_id,
                    created_at=info.created_at,
                    updated_at=info.updated_at,
                    waiting_s=detail.waiting_s,
                    note=detail.note,
                    fetched_at=info.fetched_at,
                )
                if info is not None
                else None
            ),
            platform_unavailable=detail.platform_unavailable,
            next_step=detail.next_step,
            actions=list(detail.actions),
            recommended_action=detail.recommended_action,
            can_run_rules=detail.can_run_rules,
            can_refresh=detail.can_refresh,
            can_negotiate=detail.can_negotiate,
            can_finish_negotiation=detail.can_finish_negotiation,
            manager_required=detail.manager_required,
            caller_is_manager=detail.caller_is_manager,
            write_enabled=detail.write_enabled,
            write_scope=detail.write_scope,
            negotiated=detail.negotiated,
            supported_actions=list(detail.supported_actions),
            platform_note=detail.platform_note,
            refuse_hint=detail.refuse_hint,
        )


__all__ = [
    "BatchReviewView",
    "BatchRowView",
    "DeskDetailView",
    "ReviewPendingRequest",
    "TicketView",
    "error_of",
]
