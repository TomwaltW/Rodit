"""售后审核工作台（§7.3.2 商家侧审核助手、§15 Phase 2）的店铺范围路由清单（§9.2 串店测试覆盖）。

角色和已有的售后审核接口一样：客服、店长、商家（HUMAN）。
每加一个店铺范围的路由就在 ROUTES 里加一条：body 用能通过校验的合成请求体，roles 是允许的角色。
"""

from __future__ import annotations

from api_routes import HUMAN, Route

_A = "/v1/shops/{shop}/aftersales"

ROUTES: tuple[Route, ...] = (
    Route("GET", _A, roles=HUMAN),
    Route("POST", f"{_A}/review-pending", {"limit": 5}, HUMAN),
    Route("GET", f"{_A}/stats", roles=HUMAN),
    Route("GET", f"{_A}/as_2001", roles=HUMAN),
)
