"""商家控制台（知识编辑、店铺配置、商品同步，§9.1、§5.2）的店铺范围路由清单（§9.2 串店测试覆盖）。

每加一个店铺范围的路由就在 ROUTES 里加一条：body 用能通过校验的合成请求体，roles 是允许的角色。
"""

from __future__ import annotations

from api_routes import CONSOLE, HUMAN, MERCHANT, REVIEWER, Route  # noqa: F401

ROUTES: tuple[Route, ...] = ()
