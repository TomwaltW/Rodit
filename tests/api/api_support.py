"""HTTP 接口测试的公共部分：演示单元、开发令牌、测试客户端、全部店铺范围路由的清单。

importlib 模式下测试模块不能 import conftest，所以放在单独的模块里，
由 conftest 把本目录加进 sys.path。
数据全部来自合成演示世界（rodit.testing.fixtures），不联网。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from starlette.exceptions import StarletteDeprecationWarning

from rodit.api import Principal, StaticTokenAuthenticator, create_app
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit
from rodit.core.enums import Platform
from rodit.testing.fixtures import AGENT, MANAGER, SHOP_A, SHOP_B, SHOP_C, SHOP_D

with warnings.catch_warnings():
    # 依赖冻结：开发依赖里是 httpx（不是 httpx2），starlette 导入测试客户端时会发弃用警告
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

MERCHANT_A = "owner_a"

CALLERS: dict[str, Principal] = {
    "agent_a": Principal(shop_id=SHOP_A, user_id=AGENT, role="agent"),
    "manager_a": Principal(shop_id=SHOP_A, user_id=MANAGER, role="manager"),
    "merchant_a": Principal(shop_id=SHOP_A, user_id=MERCHANT_A, role="merchant"),
    "integration_a": Principal(shop_id=SHOP_A, user_id="tb_push_a", role="integration"),
    # 店铺 A 的"店长"角色，但不在店铺配置的 manager_ids 里（R6 仍然不能确认）
    "fake_manager_a": Principal(shop_id=SHOP_A, user_id="kf_not_manager", role="manager"),
    "agent_b": Principal(shop_id=SHOP_B, user_id="kf_b01", role="agent"),
    "manager_b": Principal(shop_id=SHOP_B, user_id="kf_b_mgr", role="manager"),
    "merchant_b": Principal(shop_id=SHOP_B, user_id="owner_b", role="merchant"),
    "integration_b": Principal(shop_id=SHOP_B, user_id="tb_push_b", role="integration"),
    "agent_c": Principal(shop_id=SHOP_C, user_id="kf_c01", role="agent"),
    "agent_d": Principal(shop_id=SHOP_D, user_id="kf_d01", role="agent"),
    "integration_d": Principal(shop_id=SHOP_D, user_id="dy_push_d", role="integration"),
}

TOKENS: dict[str, str] = {name: f"test-token-{name}-0123456789abcdef" for name in CALLERS}
"""合成的开发令牌（只在测试进程里有效）。"""

SHOP_B_CALLERS = ("agent_b", "manager_b", "merchant_b", "integration_b")


def auth(name: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKENS[name]}"}


@dataclass
class Api:
    client: TestClient
    service: AssistService

    @property
    def unit(self):
        return self.service.unit

    def call(self, method: str, path: str, who: str | None = None, **kwargs: Any):
        headers = dict(kwargs.pop("headers", {}) or {})
        if who is not None:
            headers.update(auth(who))
        return self.client.request(method, path, headers=headers, **kwargs)

    def post(self, path: str, who: str | None = None, **kwargs: Any):
        return self.call("POST", path, who, **kwargs)

    def get(self, path: str, who: str | None = None, **kwargs: Any):
        return self.call("GET", path, who, **kwargs)


def build_api(platform: Platform = Platform.TAOBAO, *, raise_server_exceptions: bool = True) -> Api:
    service = AssistService(build_demo_unit(platform))
    authenticator = StaticTokenAuthenticator({TOKENS[n]: p for n, p in CALLERS.items()})
    app = create_app(service, authenticator=authenticator)
    return Api(TestClient(app, raise_server_exceptions=raise_server_exceptions), service)


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


_HUMAN = frozenset({"agent", "manager", "merchant"})
_CONSOLE = frozenset({"manager", "merchant"})
_MERCHANT = frozenset({"merchant"})

SHOP_ROUTES: tuple[Route, ...] = (
    Route("POST", "/v1/shops/{shop}/standalone/ask", {"session_id": "w", "question": "能开发票吗"}),
    Route("POST", "/v1/shops/{shop}/suggestions/1/sent"),
    Route("POST", "/v1/shops/{shop}/suggestions/1/edited", {"edited_text": "亲，可以开的哦"}),
    Route("POST", "/v1/shops/{shop}/suggestions/1/rejected", {"reason": "other"}),
    Route("GET", "/v1/shops/{shop}/knowledge/candidates", roles=_MERCHANT),
    Route("POST", "/v1/shops/{shop}/knowledge/candidates/1/approve", roles=_MERCHANT),
    Route("POST", "/v1/shops/{shop}/knowledge/1/retire", roles=_MERCHANT),
    Route("GET", "/v1/shops/{shop}/knowledge/gaps", roles=_CONSOLE),
    Route("POST", "/v1/shops/{shop}/questionnaire", {"answers": {"7": "不开发票"}}, _MERCHANT),
    Route("GET", "/v1/shops/{shop}/reports/adoption", roles=_CONSOLE),
    Route("GET", "/v1/shops/{shop}/reports/qa", roles=_CONSOLE),
    Route("POST", "/v1/shops/{shop}/aftersales/as_2001/review"),
    Route("POST", "/v1/shops/{shop}/aftersales/as_2001/negotiate", {"note": "先协商"}),
    Route("POST", "/v1/shops/{shop}/aftersales/as_2001/negotiation-done", {"outcome": "谈妥"}),
    Route(
        "POST",
        "/v1/shops/{shop}/aftersales/as_2001/confirm",
        {"action": "agree_refund", "expected_status_version": "1"},
    ),
    Route("POST", "/v1/shops/{shop}/revoke", roles=_MERCHANT),
)
"""全部店铺范围的路由（§9.2 串店测试要覆盖每一个）。"""

EVENTS_PATH = "/v1/platform/B/events"


def shop_snapshot(api: Api, shop_id: str, conversations: tuple[str, ...] = ()) -> dict[str, Any]:
    """一家店在单元里的全部可观察状态，用来断言"越权请求什么都没改"。"""
    repos = api.unit.repos
    adapter = api.unit.adapters.get(api.unit.platform, "B")
    shop = repos.knowledge.get_shop(shop_id)
    return {
        "shop": shop.model_dump() if shop else None,
        "auth": repos.auth.get(shop_id) is not None,
        "policies": {k: v.value for k, v in repos.knowledge.get_policies(shop_id).items()},
        "knowledge": [
            (k.id, k.status, k.approved_by)
            for k in repos.knowledge.list_knowledge(shop_id, status=None)
        ],
        "suggestions": [(s.id, s.status, s.agent_id) for s in repos.suggestions.list(shop_id)],
        "reviews": [
            (r.aftersale_id, r.state, r.version) for r in repos.reviews.list_by_state(shop_id)
        ],
        "audit": len(repos.audit.list_events(shop_id)),
        "gaps": len(repos.feedback.list_gaps(shop_id)),
        "actions": [c for c in (api.unit.action_port.calls if api.unit.action_port else [])],
        "context": {
            c: [m.msg_id for m in adapter.get_context(shop_id, c).recent_messages]
            for c in conversations
        },
    }
