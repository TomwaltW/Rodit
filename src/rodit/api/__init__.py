"""HTTP 接口（模式 C 独立话术助手、平台事件、反馈、知识审核、报表、售后审核、取消订购）。

- app.create_app(service, authenticator=...)：应用工厂，一个平台单元一份（§9.1），没有全局状态；
- deps：共用的依赖和鉴权（令牌 → 身份 → 串店 → 角色 → 店铺状态，§9.2）；
- routes_console：商家控制台（知识编辑、店铺配置、商品同步，§9.1、§5.2）；
- routes_rollout：上线节奏与每周人工复核（§13.4、§13.5）；
- auth：调用方身份 Principal、Authenticator 协议、开发用 StaticTokenAuthenticator（§9.2 租户隔离）；
- errors：统一的错误 JSON 和异常映射（§12.2、§12.3）；
- schemas：请求 / 响应模型（只放客服、商家需要看的内容，§5.3）；
- web：模式 C 独立话术助手网页（GET / 和 /assist，自带的静态页面，§8.1、§5.6、§5.8、§12.4）；
- python -m rodit.api：本地演示（合成数据，仅本机 127.0.0.1:8000，网页工具在 /assist）。

负责：集成阶段（TΩ）。可以依赖所有子包，但只调用 rodit.app.AssistService
（外加 Unit 字段的只读查询）。
"""

from rodit.api.app import create_app
from rodit.api.auth import (
    CONSOLE_ROLES,
    HUMAN_ROLES,
    INTEGRATION_ROLES,
    MERCHANT_ROLES,
    PERSON_ROLES,
    REVIEWER_ROLES,
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
    "PERSON_ROLES",
    "REVIEWER_ROLES",
    "ApiError",
    "Authenticator",
    "Principal",
    "PrincipalRole",
    "StaticTokenAuthenticator",
    "bearer_token",
    "create_app",
]
