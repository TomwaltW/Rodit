"""店铺生命周期接口的响应模型（方案 §8.4 开通 / 取消订购）。

请求体是平台的原始订购推送（原样的 dict，由平台云代码的 parse_subscription 解析），这里只有响应。
响应里没有令牌、没有权限以外的授权信息。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rodit.app.tenancy import SubscriptionAction, SubscriptionOutcome
from rodit.core.enums import ShopStatus
from rodit.core.models import SubscriptionKind


class SubscriptionView(BaseModel):
    """订购事件处理结果：granted（开通 / 续订，带店铺状态）、revoked（取消订购，带各项删除条数和
    合计）、ignored（过时的事件，没有处理：事件时间不晚于这家店已经处理过的最新一个订购事件——
    重放的、乱序迟到的开通或取消都算，见 rodit.app.tenancy）。"""

    model_config = ConfigDict(extra="forbid")

    shop_id: str
    kind: SubscriptionKind
    action: SubscriptionAction
    status: ShopStatus | None = None
    counts: dict[str, int] | None = None
    total: int | None = None

    @classmethod
    def from_outcome(cls, outcome: SubscriptionOutcome) -> SubscriptionView:
        revoked = outcome.action == "revoked"
        return cls(
            shop_id=outcome.shop_id,
            kind=outcome.kind,
            action=outcome.action,
            status=outcome.shop.status if outcome.shop is not None else None,
            counts=dict(outcome.counts) if revoked else None,
            total=sum(outcome.counts.values()) if revoked else None,
        )


__all__ = ["SubscriptionView"]
