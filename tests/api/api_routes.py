"""店铺范围路由的清单（§9.2 串店测试要覆盖每一个）。

按功能分文件，并行开发时各改各的：核心路由在这里，商家控制台在 api_routes_console.py，
上线节奏与每周复核在 api_routes_rollout.py；api_support.SHOP_ROUTES 把它们拼在一起。
test_api_auth.test_every_shop_route_is_listed 检查应用里的每个店铺范围路由都在清单里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Route:
    """一个店铺范围的路由。path 里的 {shop} 换成要访问的店铺。"""

    method: str
    path: str
    body: Any = None
    roles: frozenset[str] = frozenset({"agent", "manager", "merchant"})

    def url(self, shop: str) -> str:
        return self.path.format(shop=shop)

    @property
    def name(self) -> str:
        return f"{self.method} {self.path}"


HUMAN = frozenset({"agent", "manager", "merchant"})
CONSOLE = frozenset({"manager", "merchant"})
MERCHANT = frozenset({"merchant"})
REVIEWER = frozenset({"reviewer"})

CORE_ROUTES: tuple[Route, ...] = (
    Route("POST", "/v1/shops/{shop}/standalone/ask", {"session_id": "w", "question": "能开发票吗"}),
    Route("GET", "/v1/shops/{shop}/items"),
    Route("POST", "/v1/shops/{shop}/suggestions/1/sent"),
    Route("POST", "/v1/shops/{shop}/suggestions/1/edited", {"edited_text": "亲，可以开的哦"}),
    Route("POST", "/v1/shops/{shop}/suggestions/1/rejected", {"reason": "other"}),
    Route("GET", "/v1/shops/{shop}/knowledge/candidates", roles=MERCHANT),
    Route("POST", "/v1/shops/{shop}/knowledge/candidates/1/approve", roles=MERCHANT),
    Route("POST", "/v1/shops/{shop}/knowledge/1/retire", roles=MERCHANT),
    Route("GET", "/v1/shops/{shop}/knowledge/gaps", roles=CONSOLE),
    Route("POST", "/v1/shops/{shop}/questionnaire", {"answers": {"7": "不开发票"}}, MERCHANT),
    Route("GET", "/v1/shops/{shop}/reports/adoption", roles=CONSOLE),
    Route("GET", "/v1/shops/{shop}/reports/qa", roles=CONSOLE),
    Route("POST", "/v1/shops/{shop}/aftersales/as_2001/review"),
    Route("POST", "/v1/shops/{shop}/aftersales/as_2001/negotiate", {"note": "先协商"}),
    Route("POST", "/v1/shops/{shop}/aftersales/as_2001/negotiation-done", {"outcome": "谈妥"}),
    Route(
        "POST",
        "/v1/shops/{shop}/aftersales/as_2001/confirm",
        {"action": "agree_refund", "expected_status_version": "1"},
    ),
    Route("POST", "/v1/shops/{shop}/revoke", roles=MERCHANT),
)
