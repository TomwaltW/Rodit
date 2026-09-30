"""店铺生命周期接口（方案 §8.4 开通 / 续订 / 取消订购、§9.2 串店检查）。

| 接口 | 角色 |
|---|---|
| POST /v1/platform/{access}/subscription | 平台对接（integration），只能是自己的店铺 |

平台服务市场的订购推送（开通、续订、取消订购、撤销授权）原样转过来：先用本单元的订购推送解析
（Unit.parse_subscription，平台云代码注入）无副作用地解析，解析不出来返回 204；事件的店铺必须是
调用方的店铺（403 tenant_isolation，什么都不改）；然后交给
AssistService.tenancy.handle_subscription：
开通 / 续订写店铺和授权（权限超出本阶段、平台不对 → 422），取消订购删除该店铺在本单元的全部数据，
留下系统事件 shop.purged。店铺还没开通时也能推（不查店铺状态），停用的店铺也能取消订购。
鉴权用 rodit.api.deps 的依赖；业务逻辑在 rodit.app.tenancy。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Path, Response

from rodit.api.deps import IntegrationDep, ServiceDep
from rodit.api.errors import ApiError, tenant_violation
from rodit.api.schemas_tenancy import SubscriptionView

MSG_NO_SUBSCRIPTION_PARSER = "本单元没有接入平台订购推送（parse_subscription）"

router = APIRouter()


@router.post(
    "/v1/platform/{access}/subscription",
    response_model=SubscriptionView,
    responses={204: {"description": "不是订购推送或无法识别，没有处理"}},
    tags=["店铺"],
)
def platform_subscription(
    access: Annotated[str, Path(pattern="^[AB]$", description="A 平台推送 / B 插件上报")],
    raw: Annotated[dict[str, Any], Body()],
    principal: IntegrationDep,
    service: ServiceDep,
) -> SubscriptionView | Response:
    """平台订购推送（§8.4）：开通、续订、取消订购、撤销授权。只给平台对接用，只能是自己的店铺。"""
    unit = service.unit
    unit.adapters.get(unit.platform, access)  # 没有注册该接入模式 → KeyError → 404
    parse = unit.parse_subscription
    if parse is None:
        raise ApiError(404, "not_found", MSG_NO_SUBSCRIPTION_PARSER)
    event = parse(dict(raw))
    if event is None:
        return Response(status_code=204)
    if event.shop_id != principal.shop_id:
        raise tenant_violation()
    outcome = service.tenancy.handle_subscription(event)
    return SubscriptionView.from_outcome(outcome)


__all__ = ["MSG_NO_SUBSCRIPTION_PARSER", "router"]
