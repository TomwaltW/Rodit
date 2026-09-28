"""模式 C 的平台工具：没有平台数据（方案 §8.1）。

独立话术助手不接平台接口，所以订单、物流、售后、库存、价格一律"查不到"：
问答引擎只能回答售前和政策类问题（依据来自知识中心），需要订单的问题按 §12.2 走缺失 / 兜底。
"""

from __future__ import annotations

from datetime import datetime

from rodit.core.enums import AfterSaleStatus, Platform
from rodit.core.models import (
    AfterSaleInfo,
    LogisticsTrace,
    OrderInfo,
    PriceInfo,
    StockInfo,
)


class NullToolGateway:
    """实现 core.ports.ToolGateway：所有查询返回 None / 空列表，不抛异常、不访问任何平台。"""

    def __init__(self, platform: Platform) -> None:
        self.platform = platform

    def __repr__(self) -> str:
        return f"NullToolGateway(platform={self.platform.value!r})"

    def get_order(self, shop_id: str, order_id: str) -> OrderInfo | None:
        return None

    def list_recent_orders(
        self, shop_id: str, buyer_ref: str, *, limit: int = 5
    ) -> list[OrderInfo]:
        return []

    def get_logistics(self, shop_id: str, order_id: str) -> LogisticsTrace | None:
        return None

    def get_aftersale(self, shop_id: str, aftersale_id: str) -> AfterSaleInfo | None:
        return None

    def list_aftersales(
        self,
        shop_id: str,
        *,
        order_id: str | None = None,
        buyer_ref: str | None = None,
        status: AfterSaleStatus | None = None,
        since: datetime | None = None,
    ) -> list[AfterSaleInfo]:
        return []

    def get_stock(self, shop_id: str, item_id: str, sku_id: str | None = None) -> StockInfo | None:
        return None

    def get_price(self, shop_id: str, item_id: str) -> PriceInfo | None:
        return None
