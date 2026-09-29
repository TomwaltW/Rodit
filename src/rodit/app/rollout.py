"""上线节奏与持续评估（方案 §13.4 影子模式对比、各阶段门槛；§13.5 每周从推荐中抽 50 条人工复核）。

所有写操作都在 processing(shop_id) 里做（和取消订购互斥，§8.4），操作人取自调用方身份，写审计
（core.audit.AuditAction 的 shop.rollout_stage / quality.*）。
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

from rodit.app.unit import Unit

ShopGate = Callable[[str], AbstractContextManager[None]]
"""AssistService.processing：按店铺的读方闸门（和取消订购互斥）。"""


class RolloutService:
    def __init__(self, unit: Unit, processing: ShopGate) -> None:
        self.unit = unit
        self._processing = processing


__all__ = ["RolloutService"]
