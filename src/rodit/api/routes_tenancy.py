"""店铺生命周期接口（方案 §8.4 开通 / 取消订购、§9.2 串店检查）。

T0 只放一个空的路由（已挂进 create_app），接口由 Track A 实现：
平台对接角色推送订购事件（POST /v1/platform/{access}/subscription，只能是自己的店铺）等。
鉴权一律用 rodit.api.deps 的依赖；业务逻辑在 rodit.app.tenancy（AssistService.tenancy）。
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

__all__ = ["router"]
