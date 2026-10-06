"""HTTP 接口测试的公共部分：演示单元、开发令牌、测试客户端、全部店铺范围路由的清单。

importlib 模式下测试模块不能 import conftest，所以放在单独的模块里，
由 conftest 把本目录加进 sys.path。
数据全部来自合成演示世界（rodit.testing.fixtures），不联网。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from api_routes import CORE_ROUTES, Route
from api_routes_aftersale import ROUTES as AFTERSALE_ROUTES
from api_routes_console import ROUTES as CONSOLE_ROUTES
from api_routes_rollout import ROUTES as ROLLOUT_ROUTES
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
    "reviewer_a": Principal(shop_id=SHOP_A, user_id="qa_reviewer_01", role="reviewer"),
    "reviewer_b": Principal(shop_id=SHOP_B, user_id="qa_reviewer_02", role="reviewer"),
    "agent_c": Principal(shop_id=SHOP_C, user_id="kf_c01", role="agent"),
    "agent_d": Principal(shop_id=SHOP_D, user_id="kf_d01", role="agent"),
    "integration_d": Principal(shop_id=SHOP_D, user_id="dy_push_d", role="integration"),
}

TOKENS: dict[str, str] = {name: f"test-token-{name}-0123456789abcdef" for name in CALLERS}
"""合成的开发令牌（只在测试进程里有效）。"""

SHOP_B_CALLERS = ("agent_b", "manager_b", "merchant_b", "integration_b", "reviewer_b")
SHOP_A_CALLERS = ("agent_a", "manager_a", "merchant_a", "integration_a", "reviewer_a")


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


SHOP_ROUTES: tuple[Route, ...] = CORE_ROUTES + CONSOLE_ROUTES + ROLLOUT_ROUTES + AFTERSALE_ROUTES
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
        "items": [(c.item_id, c.updated_at) for c in repos.knowledge.list_item_cards(shop_id)],
        "quality": [(q.id, q.status, q.reviewer) for q in repos.quality_reviews.list(shop_id)],
        "audit": len(repos.audit.list_events(shop_id)),
        "gaps": len(repos.feedback.list_gaps(shop_id)),
        "actions": [c for c in (api.unit.action_port.calls if api.unit.action_port else [])],
        "context": {
            c: [m.msg_id for m in adapter.get_context(shop_id, c).recent_messages]
            for c in conversations
        },
    }
