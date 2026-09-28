"""售后买家侧问答的依据（方案 §7.3.1）。

- 能不能退、运费谁出：店铺政策 + 订单状态（是否签收、签收几天、商品是否支持七天无理由）；
- 退款什么时候到：售后单当前进度 + 平台的一般到账规则。不做硬承诺（不写具体到账天数）。

七天无理由的天数按东八区自然日计算：签收当天为第 0 天，"自签收次日起算 7 天"，
即签收后第 7 个自然日结束前都在期限内（《消费者权益保护法》第二十五条及配套办法；
具体写法请法务确认，本实现不构成法律意见）。审核规则 R3 用同一个口径（days_since_signed）。
"""

from __future__ import annotations

from datetime import datetime

from rodit.core.clock import CST, fmt_time
from rodit.core.config import AfterSaleConfig
from rodit.core.enums import AfterSaleStatus, OrderStatus, PolicyKey
from rodit.core.models import (
    AfterSaleInfo,
    OrderInfo,
    PolicyEntry,
    RefundProgress,
    ReturnEligibility,
)

AFTERSALE_STATUS_LABELS: dict[AfterSaleStatus, str] = {
    AfterSaleStatus.PENDING_REVIEW: "待商家处理",
    AfterSaleStatus.WAIT_BUYER_RETURN: "等待买家退货",
    AfterSaleStatus.WAIT_SELLER_CONFIRM: "买家已寄回，等待商家确认收货",
    AfterSaleStatus.REFUNDING: "退款中",
    AfterSaleStatus.REFUND_SUCCESS: "退款成功",
    AfterSaleStatus.REFUSED: "商家已拒绝",
    AfterSaleStatus.CLOSED: "已关闭",
}

_PROGRESS_TEXT: dict[AfterSaleStatus, str] = {
    AfterSaleStatus.PENDING_REVIEW: "售后申请已提交，正在等待商家处理，处理结果以平台通知为准",
    AfterSaleStatus.WAIT_BUYER_RETURN: (
        "商家已同意退货，请按售后页面提供的退货地址寄回商品并填写运单号，仓库确认收货后处理退款"
    ),
    AfterSaleStatus.WAIT_SELLER_CONFIRM: "退货已寄回，等待仓库确认收货，确认后会处理退款",
    AfterSaleStatus.REFUNDING: "退款处理中，款项会原路退回，到账时间以平台规则为准",
    AfterSaleStatus.REFUND_SUCCESS: "退款已完成，款项原路退回，到账时间以平台和支付渠道的规则为准",
    AfterSaleStatus.REFUSED: "这次售后申请已被拒绝，如有疑问可以联系客服协商",
    AfterSaleStatus.CLOSED: "这次售后申请已关闭（已撤销或超时），如仍需售后可以在订单页重新申请",
}


def days_since_signed(signed_at: datetime, now: datetime) -> int:
    """签收后的自然日天数（东八区）：签收当天为 0，次日为 1。"""
    return (now.astimezone(CST).date() - signed_at.astimezone(CST).date()).days


def _policy(policies: dict[str, PolicyEntry], key: PolicyKey) -> str | None:
    entry = policies.get(key.value)
    return entry.value.strip() if entry is not None and entry.value.strip() else None


def _with_policies(text: str, policies: dict[str, PolicyEntry], keys: list[PolicyKey]) -> str:
    """在理由后面引用店铺政策原文（有才引用）。"""
    cited = [v for v in (_policy(policies, k) for k in keys) if v]
    if not cited:
        return text
    return text + "。店铺政策：" + "；".join(cited)


class DefaultAfterSaleAdvisor:
    """实现 core.ports.AfterSaleAdvisor。"""

    def __init__(self, config: AfterSaleConfig | None = None) -> None:
        self._config = config or AfterSaleConfig()

    def return_eligibility(
        self, order: OrderInfo, policies: dict[str, PolicyEntry], now: datetime
    ) -> ReturnEligibility:
        n = self._config.seven_day_days
        quality = [PolicyKey.QUALITY_ISSUE]
        no_reason = [
            PolicyKey.SEVEN_DAY_RETURN,
            PolicyKey.RETURN_SHIPPING,
            PolicyKey.SHIPPING_INSURANCE,
        ]

        def result(
            eligible: bool | None, reason: str, days: int | None = None
        ) -> ReturnEligibility:
            return ReturnEligibility(
                order_id=order.order_id, eligible=eligible, days_since_signed=days, reason=reason
            )

        if order.status == OrderStatus.UNPAID:
            return result(None, "订单还未付款，不涉及退款；不想要可以直接取消订单")
        if order.status == OrderStatus.CLOSED:
            return result(False, "订单已关闭（已取消或已退款），不能再申请退货")
        if order.status == OrderStatus.PAID:
            return result(True, "订单还没发货，可以在订单页申请仅退款（未发货退款）")

        signed_at = order.signed_at
        if not order.seven_day_return:
            days = days_since_signed(signed_at, now) if signed_at is not None else None
            reason = _with_policies(
                "该商品不支持七天无理由退换（如定制商品）；如有质量问题，仍可以申请售后",
                policies,
                [PolicyKey.NON_RETURNABLE, *quality],
            )
            return result(False, reason, days)
        if signed_at is None:
            if order.status == OrderStatus.SHIPPED:
                reason = _with_policies(
                    f"包裹还在运输中，签收后 {n} 天内（自签收次日起算）支持七天无理由退货",
                    policies,
                    no_reason,
                )
                return result(True, reason)
            return result(None, "缺少签收时间，能否无理由退货需要人工确认")

        days = days_since_signed(signed_at, now)
        if days > n:
            reason = _with_policies(
                f"已签收 {days} 天，超过 {n} 天无理由退货期限；如有质量问题，仍可以申请售后",
                policies,
                quality,
            )
            return result(False, reason, days)
        reason = _with_policies(
            f"已签收 {days} 天，在 {n} 天无理由退货期内（自签收次日起算）",
            policies,
            no_reason,
        )
        return result(True, reason, days)

    def refund_progress(self, aftersale: AfterSaleInfo, now: datetime) -> RefundProgress:
        label = AFTERSALE_STATUS_LABELS[aftersale.status]
        text = _PROGRESS_TEXT[aftersale.status]
        summary = f"{label}：{text}（售后单更新于 {fmt_time(aftersale.updated_at)}）"
        return RefundProgress(
            aftersale_id=aftersale.aftersale_id, status=aftersale.status, summary=summary, as_of=now
        )
