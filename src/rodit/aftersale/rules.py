"""售后审核规则 R1–R8（方案 §7.3.2）。纯函数：输入平台数据，输出建议和下一个状态，不做任何操作。

执行顺序：
1. R1 售后单已不是待审核状态（不在任何可操作状态里）→ 不操作，同步状态（NONE → PLATFORM），
   命中即停，不叠加其他规则；
2. 退货退款的第二步（R3 "仓库确认收到后同意退款"）：平台状态是 WAIT_SELLER_CONFIRM
   （买家已寄回）且是退货退款 → 建议同意退款（AGREE），标 R3；
3. 基础建议取第一条命中的：
   R2 订单未发货且申请仅退款 → 同意（平台自身可能也会自动处理，以平台为准）；
   R3 签收 ≤ 7 天、商品支持七天无理由、七天无理由退货退款 → 同意退货，仓库确认收到后再同意退款；
   R4 物流显示签收、买家称没收到 → 先协商，进入签收核查（VERIFY）；
   R5 质量问题 / 破损 / 错漏发 → 没有凭证进入等待凭证（EVIDENCE），有凭证由客服查看后决定；
   R8 以上都不命中 → 人工决定；
4. 叠加：R6 金额超过店铺上限（严格大于）→ 必须店长确认；
   R7 同一买家 R7 窗口内售后次数（含当前这单）≥ 阈值 → 标记风险，建议改为人工决定。

从不自动建议拒绝（REFUSE）：淘宝 2025-10-15 起部分场景要先协商才能拒绝，拒绝率偏高会被阻断
拒绝接口——"拒绝要非常保守，优先建议协商"。拒绝只能由客服在确认时自己选择。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rodit.aftersale.advisor import AFTERSALE_STATUS_LABELS, days_since_signed
from rodit.core.config import AfterSaleConfig
from rodit.core.enums import (
    AfterSaleReason,
    AfterSaleStatus,
    AfterSaleType,
    LogisticsStatus,
    OrderStatus,
    ReviewAdvice,
    ReviewState,
    SignedDetail,
    is_actionable,
)
from rodit.core.models import AfterSaleInfo, LogisticsTrace, OrderInfo, ShopConfig

EVIDENCE_REASONS = frozenset(
    {
        AfterSaleReason.QUALITY,
        AfterSaleReason.DAMAGED,
        AfterSaleReason.WRONG_ITEM,
        AfterSaleReason.MISSING_ITEM,
    }
)
"""R5：需要凭证的售后原因。"""

_NOT_SHIPPED = frozenset({OrderStatus.UNPAID, OrderStatus.PAID})

_SIGNED_PLACE = {
    SignedDetail.SELF: "本人签收",
    SignedDetail.STATION: "驿站代收",
    SignedDetail.LOCKER: "快递柜代收",
    SignedDetail.OTHER: "他人代收",
}

REASON_WAIT_SELLER_CONFIRM = "仓库确认收货后同意退款"


class ReviewSuggestion(BaseModel):
    """规则引擎的输出（对应 aftersale_review 的 matched_rule / suggestion / reason 等字段）。"""

    model_config = ConfigDict(extra="forbid")

    matched_rules: list[str]
    advice: ReviewAdvice
    next_state: ReviewState
    reason: str
    requires_manager: bool = False
    risk_flags: list[str] = Field(default_factory=list)
    # R5 凭证的图片理解（aftersale.evidence_vision）：只有张数和类型，写进 aftersale.suggest 审计；
    # 规则本身不填（规则不看图片），由审核助手在识别过凭证时补上
    evidence_vision: dict[str, Any] | None = None


def _signed_at(order: OrderInfo | None, trace: LogisticsTrace | None) -> datetime | None:
    if order is not None and order.signed_at is not None:
        return order.signed_at
    if trace is not None and trace.status == LogisticsStatus.SIGNED and trace.last_event:
        return trace.last_event.time
    return None


def _logistics_signed(order: OrderInfo | None, trace: LogisticsTrace | None) -> bool:
    """R4 "物流显示签收"：优先看轨迹；查不到轨迹时退回订单的签收时间。"""
    if trace is not None:
        return trace.status == LogisticsStatus.SIGNED
    return order is not None and order.signed_at is not None


def repeat_count(
    info: AfterSaleInfo, buyer_history: Sequence[AfterSaleInfo], *, window_days: int, now: datetime
) -> int:
    """R7：同一买家在本店 window_days 天内的售后次数（含当前这单，按 aftersale_id 去重）。"""
    since = now - timedelta(days=window_days)
    ids = {
        a.aftersale_id
        for a in buyer_history
        if a.shop_id == info.shop_id and a.buyer_ref == info.buyer_ref and a.created_at >= since
    }
    ids.add(info.aftersale_id)
    return len(ids)


def _base(
    info: AfterSaleInfo,
    order: OrderInfo | None,
    trace: LogisticsTrace | None,
    config: AfterSaleConfig,
    now: datetime,
) -> tuple[str, ReviewAdvice, ReviewState, str]:
    """基础建议：(规则, 建议, 下一个状态, 理由)。"""
    # 退货退款第二步：买家已寄回
    if info.status == AfterSaleStatus.WAIT_SELLER_CONFIRM:
        if info.type == AfterSaleType.RETURN_REFUND:
            return (
                "R3",
                ReviewAdvice.AGREE,
                ReviewState.CONFIRM,
                f"买家已寄回退货：{REASON_WAIT_SELLER_CONFIRM}（R3 第二步）",
            )
        return (
            "R8",
            ReviewAdvice.MANUAL,
            ReviewState.CONFIRM,
            "买家已寄回商品，请仓库确认收货后人工决定",
        )

    # R2：订单未发货，申请仅退款
    if (
        order is not None
        and order.status in _NOT_SHIPPED
        and info.type == AfterSaleType.REFUND_ONLY
    ):
        return (
            "R2",
            ReviewAdvice.AGREE,
            ReviewState.CONFIRM,
            "订单未发货，买家申请仅退款：建议同意（平台自身可能也会自动处理，以平台为准）",
        )

    signed_at = _signed_at(order, trace)
    days = days_since_signed(signed_at, now) if signed_at is not None else None
    n = config.seven_day_days

    # R3：签收 ≤ 7 天、支持七天无理由、七天无理由退货退款
    if (
        order is not None
        and days is not None
        and days <= n
        and order.seven_day_return
        and info.type == AfterSaleType.RETURN_REFUND
        and info.reason == AfterSaleReason.NO_REASON
    ):
        return (
            "R3",
            ReviewAdvice.AGREE_RETURN,
            ReviewState.CONFIRM,
            f"签收 {days} 天（≤ {n} 天），商品支持七天无理由，申请退货退款："
            "建议同意退货；仓库确认收到后再同意退款",
        )

    # R4：物流显示签收，买家称没收到
    if info.reason == AfterSaleReason.NOT_RECEIVED and _logistics_signed(order, trace):
        where = ""
        if trace is not None and trace.signed_detail is not None:
            where = _SIGNED_PLACE[trace.signed_detail]
            if trace.pickup_info:
                where += f"（{trace.pickup_info}）"
            where = f"：{where}"
        return (
            "R4",
            ReviewAdvice.NEGOTIATE,
            ReviewState.VERIFY,
            f"物流显示已签收{where}，买家称没收到：建议先协商，提供驿站或快递柜信息并发起快递核查；"
            f"{config.verify_hours} 小时后仍未找到，建议同意",
        )

    # R5：质量问题、破损、错漏发
    if info.reason in EVIDENCE_REASONS:
        if info.evidence_count == 0:
            return (
                "R5",
                ReviewAdvice.MANUAL,
                ReviewState.EVIDENCE,
                "质量 / 破损 / 错漏发需要凭证：请买家在售后单里上传照片，凭证补充后由客服查看决定",
            )
        return (
            "R5",
            ReviewAdvice.MANUAL,
            ReviewState.CONFIRM,
            f"质量 / 破损 / 错漏发，买家已上传 {info.evidence_count} 张凭证：客服查看凭证后决定",
        )

    # R8：以上都不命中
    details: list[str] = []
    if order is None:
        details.append("订单信息查不到")
    elif info.reason == AfterSaleReason.NO_REASON and not order.seven_day_return:
        details.append("商品不支持七天无理由（如定制商品）")
    elif info.reason == AfterSaleReason.NO_REASON and days is not None and days > n:
        details.append(f"已签收 {days} 天，超过 {n} 天无理由期限")
    elif info.reason == AfterSaleReason.NOT_RECEIVED:
        details.append("物流未显示签收")
    detail = f"（{'；'.join(details)}）" if details else ""
    return ("R8", ReviewAdvice.MANUAL, ReviewState.CONFIRM, f"以上规则都不命中{detail}：人工决定")


def evaluate(
    info: AfterSaleInfo,
    *,
    order: OrderInfo | None,
    trace: LogisticsTrace | None,
    buyer_history: Sequence[AfterSaleInfo],
    config: AfterSaleConfig,
    shop_config: ShopConfig,
    now: datetime,
) -> ReviewSuggestion:
    """对一张售后单跑 R1–R8，返回建议。order / trace 查不到时传 None（规则会保守地落到人工）。"""
    # R1：已不是待审核状态，命中即停
    if not is_actionable(info.status):
        return ReviewSuggestion(
            matched_rules=["R1"],
            advice=ReviewAdvice.NONE,
            next_state=ReviewState.PLATFORM,
            reason=f"售后单已是「{AFTERSALE_STATUS_LABELS[info.status]}」，不操作，只同步状态",
        )

    rule, advice, next_state, reason = _base(info, order, trace, config, now)
    matched = [rule]
    notes: list[str] = []
    risk_flags: list[str] = []
    requires_manager = False

    # R6：金额超过店铺上限 → 必须店长确认
    limit = shop_config.aftersale_amount_limit
    if info.amount > limit:
        matched.append("R6")
        requires_manager = True
        notes.append(f"金额 {info.amount} 元超过店铺上限 {limit} 元，必须店长确认（R6）")

    # R7：同一买家窗口内售后次数 ≥ 阈值 → 标记风险，人工决定
    count = repeat_count(info, buyer_history, window_days=config.repeat_window_days, now=now)
    if count >= config.repeat_threshold:
        matched.append("R7")
        flag = f"{config.repeat_window_days} 天内售后 {count} 次（≥ {config.repeat_threshold}）"
        risk_flags.append(flag)
        if advice != ReviewAdvice.MANUAL:
            advice = ReviewAdvice.MANUAL
        notes.append(f"该买家 {flag}，标记风险，人工决定（R7）")

    if notes:
        reason = reason + "；" + "；".join(notes)
    return ReviewSuggestion(
        matched_rules=matched,
        advice=advice,
        next_state=next_state,
        reason=reason,
        requires_manager=requires_manager,
        risk_flags=risk_flags,
    )
