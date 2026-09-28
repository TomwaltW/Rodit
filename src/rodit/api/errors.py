"""接口层的错误映射（方案 §12.2 可用性与降级、§12.3 日志脱敏）。

所有错误都返回同一种 JSON：``{"error": 错误码, "message": 说明}``。

| 异常 | 状态码 | 错误码 |
|---|---|---|
| 令牌缺失或无效（ApiError） | 401 | unauthorized |
| 跨店（ApiError） | 403 | tenant_isolation |
| 角色不符（ApiError） | 403 | forbidden_role |
| R6 不是店长（ApiError） | 403 | manager_required |
| KeyError | 404 | not_found |
| ActionNotAllowed | 409 | action_not_allowed |
| ConcurrencyConflict | 409 | conflict |
| ValueError、请求校验失败 | 422 | validation_error |
| AuthExpired | 403 | shop_not_authorized |
| ShopInactive | 403 | shop_inactive |
| TenantIsolationError | 403 | tenant_isolation |
| ComplianceError | 403 | compliance |
| StorageUnavailable | 503 | storage_unavailable |
| ToolUnavailable / KnowledgeUnavailable | 503 | tool_unavailable / knowledge_unavailable |
| 其他 RoditError、未预期的异常 | 500 | internal_error |

说明文字的约束：
- 不含令牌、连接串、买家个人信息：存储、内部错误一律用固定说明；其余用业务异常自带的说明
  （各模块已做最小化），再过一遍 mask_pii 兜底，并截断长度；
- 不回显请求体：请求校验失败只报字段位置和原因，不带输入值（FastAPI 默认会带 input）；
- 跨店（TenantIsolationError）用固定说明，不透露别的店铺有没有这条数据（§9.2）。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from rodit.core.errors import (
    ActionNotAllowed,
    AuthExpired,
    ComplianceError,
    ConcurrencyConflict,
    KnowledgeUnavailable,
    RoditError,
    ShopInactive,
    StorageUnavailable,
    TenantIsolationError,
    ToolUnavailable,
)
from rodit.core.pii import contains_pii, mask_pii

logger = logging.getLogger("rodit.api")

MAX_MESSAGE_CHARS = 300

MSG_UNAUTHORIZED = "缺少或无效的访问令牌"
MSG_TENANT = "不能访问其他店铺的数据（§9.2）"
MSG_NOT_FOUND = "资源不存在或不属于该店铺"
MSG_INTERNAL = "服务内部错误，请稍后重试"
MSG_VALIDATION = "请求参数不合法"


class ApiError(Exception):
    """接口层自己判定的错误（鉴权、角色、跨店）。message 是固定文字，不含请求内容。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = dict(headers or {})


def unauthorized() -> ApiError:
    return ApiError(401, "unauthorized", MSG_UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})


def tenant_violation() -> ApiError:
    return ApiError(403, "tenant_isolation", MSG_TENANT)


def safe_message(text: object, fallback: str) -> str:
    """业务异常的说明 → 可以返回给调用方的文字：个人信息打码、截断；空时用 fallback。"""
    message = text if isinstance(text, str) else ""
    message = message.strip()
    if not message:
        return fallback
    if contains_pii(message):
        message = mask_pii(message)
    if len(message) > MAX_MESSAGE_CHARS:
        message = message[: MAX_MESSAGE_CHARS - 1] + "…"
    return message


def error_response(
    status_code: int, code: str, message: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        {"error": code, "message": message}, status_code=status_code, headers=dict(headers or {})
    )


def _describe_errors(errors: Iterable[Mapping[str, Any]]) -> str:
    """校验错误 → "位置：原因" 列表。只用 loc 和 msg，不带 input / ctx（可能是请求原文）。"""
    parts: list[str] = []
    for err in errors:
        loc = err.get("loc") or ()
        where = ".".join(str(p) for p in loc) if isinstance(loc, Sequence) else ""
        msg = str(err.get("msg") or "不合法")
        parts.append(f"{where}：{msg}" if where else msg)
    return safe_message(f"{MSG_VALIDATION}：" + "；".join(parts), MSG_VALIDATION)


# (异常类型, 状态码, 错误码, 是否用异常自带的说明, 固定说明)
_RODIT_ERRORS: tuple[tuple[type[RoditError], int, str, bool, str], ...] = (
    (ActionNotAllowed, 409, "action_not_allowed", True, "操作不被允许"),
    (ConcurrencyConflict, 409, "conflict", True, "数据已被修改，请刷新后重试"),
    (AuthExpired, 403, "shop_not_authorized", True, "店铺没有授权或授权已过期（§8.4）"),
    (ShopInactive, 403, "shop_inactive", True, "店铺不存在或已停用（§8.4）"),
    (TenantIsolationError, 403, "tenant_isolation", False, MSG_TENANT),
    (ComplianceError, 403, "compliance", False, "违反合规约束，已拒绝处理（§12.3）"),
    (StorageUnavailable, 503, "storage_unavailable", False, "存储暂时不可用，请稍后重试"),
    (ToolUnavailable, 503, "tool_unavailable", False, "平台接口暂时不可用，请稍后重试"),
    (KnowledgeUnavailable, 503, "knowledge_unavailable", False, "知识库暂时不可用，请稍后重试"),
    (RoditError, 500, "internal_error", False, MSG_INTERNAL),
)

_HTTP_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
}


def _rodit_handler(status_code: int, code: str, use_message: bool, fallback: str):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        if status_code >= 500:
            # 只记异常类型和路由，不记说明（可能含连接串等）和请求内容
            logger.error("接口错误 %s：%s %s", type(exc).__name__, request.method, request.url.path)
        message = safe_message(str(exc), fallback) if use_message else fallback
        return error_response(status_code, code, message)

    return handler


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(exc.status_code, exc.code, exc.message, exc.headers)


async def _key_error(request: Request, exc: Exception) -> JSONResponse:
    return error_response(404, "not_found", MSG_NOT_FOUND)


async def _value_error(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, ValidationError):
        message = _describe_errors(exc.errors(include_input=False, include_context=False))
    else:
        message = safe_message(str(exc), MSG_VALIDATION)
    return error_response(422, "validation_error", message)


async def _request_validation(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return error_response(422, "validation_error", _describe_errors(exc.errors()))


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _HTTP_CODES.get(exc.status_code, "http_error")
    message = MSG_NOT_FOUND if exc.status_code == 404 else safe_message(exc.detail, code)
    return error_response(exc.status_code, code, message, getattr(exc, "headers", None))


async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.error("未预期的异常 %s：%s %s", type(exc).__name__, request.method, request.url.path)
    return error_response(500, "internal_error", MSG_INTERNAL)


def install_error_handlers(app: FastAPI) -> None:
    """注册错误映射。Starlette 按异常类的 MRO 找处理函数，子类优先。"""
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(RequestValidationError, _request_validation)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(KeyError, _key_error)
    app.add_exception_handler(ValueError, _value_error)
    for exc_type, status_code, code, use_message, fallback in _RODIT_ERRORS:
        app.add_exception_handler(
            exc_type, _rodit_handler(status_code, code, use_message, fallback)
        )
    app.add_exception_handler(Exception, _unexpected)


__all__ = [
    "MAX_MESSAGE_CHARS",
    "ApiError",
    "error_response",
    "install_error_handlers",
    "safe_message",
    "tenant_violation",
    "unauthorized",
]
