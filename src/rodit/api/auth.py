"""接口层的调用方身份与鉴权（方案 §8.4 店铺授权与权限、§9.2 租户隔离、§12.3 安全与合规）。

- 每个请求带 ``Authorization: Bearer <token>``；Authenticator 把令牌换成 Principal（调用方身份）。
  缺少令牌、格式不对、令牌未知一律 401，不区分原因，也不回显令牌。
- Principal 绑定一个店铺：店铺范围的路由路径里都有 {shop_id}，和 principal.shop_id 不一致就 403
  （§9.2 串店）。
- 角色：
  agent 客服（模式 C 提问、反馈、售后审核）；
  manager 店长（客服能做的都能做，外加 R6 店长确认，§7.3.2；看缺失知识和报表）；
  merchant 商家（审核候选知识、填问卷、取消订购；看缺失知识和报表）；
  integration 平台对接（模式 A 平台推送 / 模式 B 插件上报，只能调平台事件接口，§8.1）。
- 操作人（反馈的 agent_id、售后确认的 operator、审核知识的 merchant）和 by_manager 一律取自
  Principal，请求体里不接受这些字段（§4.1 原则 3、§7.3.2 谁确认的要可信）。
- StaticTokenAuthenticator 只用于本地开发和测试；生产环境由平台云内的登录体系（如工作台子账号
  登录态）实现 Authenticator。令牌只在内存里做常数时间比较，不写日志、不出现在 repr 和错误信息里。
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

PrincipalRole = Literal["agent", "manager", "merchant", "integration"]

HUMAN_ROLES: frozenset[str] = frozenset({"agent", "manager", "merchant"})
"""人的角色：可以提问、反馈、做售后审核操作（写操作必须有人确认，§4.1 原则 3）。"""

CONSOLE_ROLES: frozenset[str] = frozenset({"manager", "merchant"})
"""能看商家控制台数据（缺失知识、报表，§9.1、§12.4）的角色。"""

MERCHANT_ROLES: frozenset[str] = frozenset({"merchant"})
"""商家本人：审核候选知识（§5.8）、填政策问卷（附录 C）、取消订购（§8.4）。"""

INTEGRATION_ROLES: frozenset[str] = frozenset({"integration"})
"""平台对接：只能调平台事件接口（模式 A / B，§8.1）。"""

MIN_TOKEN_CHARS = 16

_NOT_HUMAN = frozenset({"system", "bot", "auto"})


class Principal(BaseModel):
    """调用方身份：属于哪个店铺、是谁、什么角色。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shop_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=64)
    role: PrincipalRole

    @model_validator(mode="after")
    def _human_user(self) -> Principal:
        if not self.user_id.strip():
            raise ValueError("user_id 不能为空")
        if self.role in HUMAN_ROLES and self.user_id.strip().lower() in _NOT_HUMAN:
            raise ValueError("客服、店长、商家的 user_id 必须是具体的人（§4.1 原则 3）")
        return self

    @property
    def is_manager(self) -> bool:
        """R6 店长确认只认店长角色（§7.3.2）；by_manager 由此推出，不取请求体。"""
        return self.role == "manager"


@runtime_checkable
class Authenticator(Protocol):
    """令牌 → 调用方身份。认不出返回 None（接口层返回 401）。实现不得记录令牌。"""

    def authenticate(self, token: str) -> Principal | None: ...


class StaticTokenAuthenticator:
    """固定令牌表（仅本地开发和测试用，生产不用）。令牌至少 16 个字符。"""

    def __init__(self, tokens: Mapping[str, Principal]) -> None:
        entries: list[tuple[bytes, Principal]] = []
        for token, principal in tokens.items():
            if not isinstance(token, str) or len(token.strip()) < MIN_TOKEN_CHARS:
                raise ValueError(f"令牌至少 {MIN_TOKEN_CHARS} 个字符")
            if not isinstance(principal, Principal):
                raise TypeError("令牌必须对应一个 Principal")
            entries.append((token.encode("utf-8"), principal))
        self._entries = tuple(entries)

    def authenticate(self, token: str) -> Principal | None:
        if not isinstance(token, str) or not token:
            return None
        candidate = token.encode("utf-8", "replace")
        match: Principal | None = None
        for secret, principal in self._entries:  # 全部比较一遍，耗时不随命中位置变化
            if hmac.compare_digest(secret, candidate):
                match = principal
        return match

    def __repr__(self) -> str:
        return f"StaticTokenAuthenticator(<{len(self._entries)} 个令牌>)"


def bearer_token(authorization: str | None) -> str | None:
    """``Authorization`` 头 → 令牌；不是 ``Bearer <token>`` 格式时返回 None。"""
    if not authorization:
        return None
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


__all__ = [
    "CONSOLE_ROLES",
    "HUMAN_ROLES",
    "INTEGRATION_ROLES",
    "MERCHANT_ROLES",
    "MIN_TOKEN_CHARS",
    "Authenticator",
    "Principal",
    "PrincipalRole",
    "StaticTokenAuthenticator",
    "bearer_token",
]
