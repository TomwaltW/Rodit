"""商家控制台（知识编辑、店铺配置、商品同步，§9.1、§5.2）的店铺范围路由清单（§9.2 串店测试覆盖）。

每加一个店铺范围的路由就在 ROUTES 里加一条：body 用能通过校验的合成请求体，roles 是允许的角色。
"""

from __future__ import annotations

from api_routes import CONSOLE, HUMAN, MERCHANT, REVIEWER, Route  # noqa: F401

_C = "/v1/shops/{shop}/console"

ROUTES: tuple[Route, ...] = (
    Route("GET", f"{_C}/faqs", roles=CONSOLE),
    Route(
        "POST",
        f"{_C}/faqs",
        {"question": "可以用花呗吗", "answer": "亲，支持花呗分期哦~"},
        MERCHANT,
    ),
    Route("POST", f"{_C}/faqs/1/edit", {"answer": "亲，默认发中通快递哦~"}, MERCHANT),
    Route(
        "POST",
        f"{_C}/candidates/6/approve-edited",
        {"answer": "亲，新疆可以发，补 15 元运费哦~"},
        MERCHANT,
    ),
    Route("GET", f"{_C}/policies", roles=CONSOLE),
    Route("POST", f"{_C}/policies/invoice", {"value": "可以开电子普通发票"}, MERCHANT),
    Route("GET", f"{_C}/items", roles=CONSOLE),
    Route("POST", f"{_C}/items/sync", roles=MERCHANT),
    Route("GET", f"{_C}/items/i_555", roles=CONSOLE),
    Route("POST", f"{_C}/items/i_555", {"size_advice": "版型偏小，建议拍大一码"}, MERCHANT),
    Route("GET", f"{_C}/config", roles=CONSOLE),
    Route("POST", f"{_C}/config", {"changes": {"tone": "专业"}}, MERCHANT),
)
