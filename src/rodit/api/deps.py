"""接口层的依赖（FastAPI Depends）：服务、调用方身份、按店铺和角色的鉴权（方案 §8.4、§9.2）。

app.py 和各个功能路由（routes_console.py、routes_rollout.py …）共用这一份，鉴权顺序一致：
    令牌 → Principal（401）→ 路径 shop_id 与 principal.shop_id 一致（403 tenant_isolation）
    → 角色（403 forbidden_role）→ 店铺存在且未停用（403 shop_inactive；取消订购接口除外）。
店铺范围的路由都要用这里的 *Dep（§9.2 串店测试覆盖全部路由）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Body, Depends, Path, Query, Request
from pydantic import AwareDatetime

from rodit.api.auth import (
    CONSOLE_ROLES,
    HUMAN_ROLES,
    INTEGRATION_ROLES,
    MERCHANT_ROLES,
    REVIEWER_ROLES,
    Authenticator,
    Principal,
    bearer_token,
)
from rodit.api.errors import ApiError, tenant_violation, unauthorized
from rodit.api.schemas import EmptyRequest
from rodit.app.service import AssistService
from rodit.core.enums import ShopStatus
from rodit.core.errors import ShopInactive

MSG_FORBIDDEN_ROLE = "当前角色不能执行这个操作"


def _service(request: Request) -> AssistService:
    return request.app.state.rodit_service


def _principal(request: Request) -> Principal:
    """令牌 → 调用方身份。缺少、格式不对、未知都是 401；令牌不进日志、不进响应。"""
    token = bearer_token(request.headers.get("authorization"))
    if token is None:
        raise unauthorized()
    authenticator: Authenticator = request.app.state.rodit_authenticator
    principal = authenticator.authenticate(token)
    if not isinstance(principal, Principal):
        raise unauthorized()
    return principal


ServiceDep = Annotated[AssistService, Depends(_service)]
PrincipalDep = Annotated[Principal, Depends(_principal)]


def _integration(principal: PrincipalDep) -> Principal:
    """平台事件接口只给平台对接（integration 角色）用，客服、商家的令牌不行。"""
    if principal.role not in INTEGRATION_ROLES:
        raise ApiError(403, "forbidden_role", MSG_FORBIDDEN_ROLE)
    return principal


IntegrationDep = Annotated[Principal, Depends(_integration)]


def _human(principal: PrincipalDep) -> Principal:
    """人的角色（客服、店长、商家）；平台对接的令牌不能登录网页工具。"""
    if principal.role not in HUMAN_ROLES:
        raise ApiError(403, "forbidden_role", MSG_FORBIDDEN_ROLE)
    return principal


HumanDep = Annotated[Principal, Depends(_human)]


def require_active_shop(service: AssistService, shop_id: str) -> None:
    """店铺不存在（含取消订购后已删除）或已停用：停止处理（§8.4）。"""
    shop = service.unit.repos.knowledge.get_shop(shop_id)
    if shop is None:
        raise ShopInactive(shop_id)
    if shop.status == ShopStatus.SUSPENDED:
        raise ShopInactive(shop_id, shop.status)


def shop_guard(roles: frozenset[str], *, require_active: bool = True) -> Callable[..., Principal]:
    """店铺范围路由的鉴权：路径 shop_id 必须是调用方自己的店铺（§9.2），角色必须在 roles 里。"""

    def guard(shop_id: str, principal: PrincipalDep, service: ServiceDep) -> Principal:
        if principal.shop_id != shop_id:
            raise tenant_violation()
        if principal.role not in roles:
            raise ApiError(403, "forbidden_role", MSG_FORBIDDEN_ROLE)
        if require_active:
            require_active_shop(service, shop_id)
        return principal

    return guard


AgentDep = Annotated[Principal, Depends(shop_guard(HUMAN_ROLES))]
ConsoleDep = Annotated[Principal, Depends(shop_guard(CONSOLE_ROLES))]
MerchantDep = Annotated[Principal, Depends(shop_guard(MERCHANT_ROLES))]
RevokeDep = Annotated[Principal, Depends(shop_guard(MERCHANT_ROLES, require_active=False))]
ReviewerDep = Annotated[Principal, Depends(shop_guard(REVIEWER_ROLES))]
"""§13.5 每周人工复核、§13.4 上线阶段切换：我们的复核人员（按店铺授权，查看要留访问记录）。"""

SuggestionId = Annotated[int, Path(ge=1)]
KnowledgeId = Annotated[int, Path(ge=1)]
AfterSaleId = Annotated[str, Path(min_length=1, max_length=128)]
Since = Annotated[AwareDatetime | None, Query(description="只统计这个时间之后（带时区）")]


EmptyBody = Annotated[EmptyRequest | None, Body()]


__all__ = [
    "AfterSaleId",
    "AgentDep",
    "ConsoleDep",
    "EmptyBody",
    "HumanDep",
    "IntegrationDep",
    "KnowledgeId",
    "MSG_FORBIDDEN_ROLE",
    "MerchantDep",
    "PrincipalDep",
    "ReviewerDep",
    "RevokeDep",
    "ServiceDep",
    "Since",
    "SuggestionId",
    "require_active_shop",
    "shop_guard",
]
