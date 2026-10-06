"""平台接入层（方案 §8）：淘宝 / 抖店 / 拼多多的模拟适配器、模式 C 独立话术助手、店铺授权。

负责：T5。只依赖 rodit.core。
"""

from rodit.adapters.auth import (
    MVP_SCOPES,
    PHASE2_SCOPES,
    SCOPE_ITEM_READ,
    SCOPE_LOGISTICS_READ,
    SCOPE_ORDER_READ,
    SCOPE_REFUND_READ,
    SCOPE_REFUND_WRITE,
    AuthGuard,
    AuthGuardTokenProvider,
    SupportsPurge,
    excess_scopes,
    scopes_for_phase,
)
from rodit.adapters.base import (
    ACCESS_MODES,
    PII_FIELD_NAMES,
    AccessMode,
    BaseMockAdapter,
    InputPrefill,
    OutgoingMessage,
    RawEvent,
    SidebarPayload,
    TransferRecord,
    cst_from_epoch_ms,
    cst_from_epoch_s,
)
from rodit.adapters.douyin import DouyinMockAdapter
from rodit.adapters.null_tools import NullToolGateway
from rodit.adapters.pdd import PddMockAdapter
from rodit.adapters.registry import AdapterRegistry
from rodit.adapters.standalone import StandaloneAssistAdapter
from rodit.adapters.taobao import TaobaoMockAdapter

__all__ = [
    "ACCESS_MODES",
    "MVP_SCOPES",
    "PHASE2_SCOPES",
    "PII_FIELD_NAMES",
    "SCOPE_ITEM_READ",
    "SCOPE_LOGISTICS_READ",
    "SCOPE_ORDER_READ",
    "SCOPE_REFUND_READ",
    "SCOPE_REFUND_WRITE",
    "AccessMode",
    "AdapterRegistry",
    "AuthGuard",
    "AuthGuardTokenProvider",
    "BaseMockAdapter",
    "DouyinMockAdapter",
    "InputPrefill",
    "NullToolGateway",
    "OutgoingMessage",
    "PddMockAdapter",
    "RawEvent",
    "SidebarPayload",
    "StandaloneAssistAdapter",
    "SupportsPurge",
    "TaobaoMockAdapter",
    "TransferRecord",
    "cst_from_epoch_ms",
    "cst_from_epoch_s",
    "excess_scopes",
    "scopes_for_phase",
]
