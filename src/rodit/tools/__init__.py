"""平台工具（方案 §7.1、§7.2）：只读网关（缓存、限流降级）、物流状态归一与异常识别、工具证据。

负责：T3。只依赖 rodit.core。
"""

from rodit.tools.evidence import (
    CLARIFY_ORDER_QUESTION,
    DefaultToolEvidenceProvider,
    OrderResolution,
    resolve_order,
)
from rodit.tools.gateway import CachingToolGateway, ToolCallStats
from rodit.tools.logistics import DefaultLogisticsAdvisor, pickup_guidance
from rodit.tools.mock_platform import MockPlatformTools, WorldData
from rodit.tools.normalize import (
    LOGISTICS_CODES,
    NormalizedStatus,
    UnknownPlatformCode,
    mask_free_text,
    mask_platform_text,
    normalize_aftersale_reason,
    normalize_aftersale_status,
    normalize_aftersale_type,
    normalize_logistics_code,
    normalize_order_status,
    normalize_trace_status,
    scrub_pickup_code,
)

__all__ = [
    "CLARIFY_ORDER_QUESTION",
    "LOGISTICS_CODES",
    "CachingToolGateway",
    "DefaultLogisticsAdvisor",
    "DefaultToolEvidenceProvider",
    "MockPlatformTools",
    "NormalizedStatus",
    "OrderResolution",
    "ToolCallStats",
    "UnknownPlatformCode",
    "WorldData",
    "mask_free_text",
    "mask_platform_text",
    "normalize_aftersale_reason",
    "normalize_aftersale_status",
    "normalize_aftersale_type",
    "normalize_logistics_code",
    "normalize_order_status",
    "normalize_trace_status",
    "pickup_guidance",
    "resolve_order",
    "scrub_pickup_code",
]
