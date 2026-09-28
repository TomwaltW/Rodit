"""HTTP 接口（模式 C 独立话术助手、平台事件、反馈、知识审核、报表、售后审核、取消订购）。

- app.create_app(service, authenticator=...)：应用工厂，一个平台单元一份（§9.1），没有全局状态；
- auth：调用方身份 Principal、Authenticator 协议、开发用 StaticTokenAuthenticator（§9.2 租户隔离）；
- errors：统一的错误 JSON 和异常映射（§12.2、§12.3）；
- schemas：请求 / 响应模型（只放客服、商家需要看的内容，§5.3）；
- python -m rodit.api：本地演示（合成数据，仅本机 127.0.0.1:8000）。

负责：集成阶段（TΩ）。可以依赖所有子包，但只调用 rodit.app.AssistService
（外加 Unit 字段的只读查询）。
"""

from rodit.api.app import create_app
from rodit.api.auth import (
    CONSOLE_ROLES,
    HUMAN_ROLES,
    INTEGRATION_ROLES,
    MERCHANT_ROLES,
    Authenticator,
    Principal,
    PrincipalRole,
    StaticTokenAuthenticator,
    bearer_token,
)
from rodit.api.errors import ApiError

__all__ = [
    "CONSOLE_ROLES",
    "HUMAN_ROLES",
    "INTEGRATION_ROLES",
    "MERCHANT_ROLES",
    "ApiError",
    "Authenticator",
    "Principal",
    "PrincipalRole",
    "StaticTokenAuthenticator",
    "bearer_token",
    "create_app",
]
