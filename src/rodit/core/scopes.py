"""店铺授权的权限名（§8.4 最小权限）。

权限名是我们内部的叫法，申请授权时由各平台的实现映射成平台的权限包名【待确认】。
放在 core 里是因为端口（core.ports.TokenProvider）的默认参数要用；
rodit.adapters.auth 原样导出，旧的导入写法照常可用。
"""

from __future__ import annotations

SCOPE_ORDER_READ = "order.read"
SCOPE_LOGISTICS_READ = "logistics.read"
SCOPE_REFUND_READ = "refund.read"
SCOPE_ITEM_READ = "item.read"
SCOPE_REFUND_WRITE = "refund.write"

MVP_SCOPES: frozenset[str] = frozenset(
    {SCOPE_ORDER_READ, SCOPE_LOGISTICS_READ, SCOPE_REFUND_READ, SCOPE_ITEM_READ}
)
"""MVP 只申请只读权限（§8.4 最小权限）。问答链路处理消息前要求具备这些权限。"""

PHASE2_SCOPES: frozenset[str] = MVP_SCOPES | {SCOPE_REFUND_WRITE}
"""二期售后审核助手（§15 Phase 2）才申请写售后权限。"""
