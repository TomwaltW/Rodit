"""商家控制台（方案 §9.1 "知识编辑、配置、报表都跟着单元走"、§5.2 知识中心、§5.2.3 商品同步、
§9.3 店铺配置）。

所有写操作都在 processing(shop_id) 里做（和取消订购互斥，§8.4：删完的店铺不会被写回来），
操作人取自调用方身份，写审计（core.audit.AuditAction 的 knowledge.* / shop.config_updated）。
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

from rodit.app.unit import Unit

ShopGate = Callable[[str], AbstractContextManager[None]]
"""AssistService.processing：按店铺的读方闸门（和取消订购互斥）。"""


class ConsoleService:
    def __init__(self, unit: Unit, processing: ShopGate) -> None:
        self.unit = unit
        self._processing = processing


__all__ = ["ConsoleService"]
