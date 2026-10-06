"""售后审核工作台接口（方案 §7.3.2 商家侧审核助手、§15 Phase 2）。

待实现（Track A）：待审核售后单列表、单张售后单详情等。单张售后单的规则建议、协商、确认执行
在 rodit.api.app（/v1/shops/{shop_id}/aftersales/{aftersale_id}/…）。
鉴权用 rodit.api.deps 的依赖；业务逻辑在 rodit.app。
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
