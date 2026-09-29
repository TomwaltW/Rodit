"""结构化 JSON 日志 + 个人信息 / 密钥过滤（方案 §12.3 日志脱敏、§5.3 最小化个人信息）。

configure_logging 给根日志器装一个输出到 stderr 的处理器（uvicorn 的日志也经过它）：

- PiiGuardFilter（装在处理器上，子日志器冒泡上来的记录也逃不掉）：把消息（已代入参数）、
  异常堆栈、stack_info 都先做下面的清理，再交给格式化：
  1. 部署环境里的密钥原文（RODIT_TOKEN_KEY、RODIT_BUYER_SALT、RODIT_LLM_API_KEY、
     数据库连接串等，启动时从环境变量取值，只存在内存里）→ ``***``；
  2. ``Bearer <令牌>``、rdt_ 开头的访问令牌、连接串里的密码（``postgresql://用户:密码@``、
     ``password=…``）→ ``***``；
  3. core.pii.mask_pii：手机号、证件号、邮箱、银行卡、微信 / QQ 号、收件地址打码。
- JsonFormatter：一行一个 JSON 对象：ts（UTC）、level、logger、msg，有异常时 exc / exc_type，
  另外只输出白名单里的 extra 字段（shop_id、stage、event、route、status、duration_ms、platform），
  其余 extra 一律不输出（避免把整条消息、请求体之类的对象带进日志）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal, TextIO

from rodit.core.pii import mask_pii

SECRET_ENV_NAMES: tuple[str, ...] = (
    "RODIT_DATABASE_DSN",
    "RODIT_MIGRATE_DSN",
    "RODIT_TOKEN_KEY",
    "RODIT_BUYER_SALT",
    "RODIT_LLM_API_KEY",
    "RODIT_DB_OWNER_PASSWORD",
    "RODIT_DB_APP_PASSWORD",
    "POSTGRES_PASSWORD",
)
"""这些环境变量的值当作密钥：出现在日志里的原文一律换成 ***。"""

EXTRA_FIELDS: tuple[str, ...] = (
    "shop_id",
    "stage",
    "event",
    "route",
    "status",
    "duration_ms",
    "platform",
)
"""JSON 日志里允许输出的 extra 字段。"""

REDACTED = "***"
MIN_SECRET_CHARS = 6  # 太短的值（如 "1"）替换了只会把正常文字弄乱

_HANDLER_MARK = "_rodit_ops_handler"
_BEARER = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}")
_ACCESS_TOKEN = re.compile(r"\brdt_[A-Za-z0-9_-]{8,}")
_URL_PASSWORD = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s@/]+@")
_KV_PASSWORD = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|api_key|apikey|token)(\s*[=:]\s*)[^\s,;&]+"
)


def secrets_from_env(
    names: Iterable[str] = SECRET_ENV_NAMES, environ: Mapping[str, str] | None = None
) -> list[str]:
    """取出这些环境变量的值（只在内存里用来做替换，不输出）。太短的值不算。"""
    env = os.environ if environ is None else environ
    values = {env.get(n, "").strip() for n in names}
    return sorted((v for v in values if len(v) >= MIN_SECRET_CHARS), key=len, reverse=True)


def scrub(text: str, secrets: Iterable[str] = ()) -> str:
    """日志文字的清理：密钥原文 → 令牌、密码 → 个人信息打码（顺序见模块说明）。"""
    if not text:
        return text
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, REDACTED)
    text = _BEARER.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
    text = _ACCESS_TOKEN.sub(REDACTED, text)
    text = _URL_PASSWORD.sub(lambda m: f"{m.group(1)}{REDACTED}@", text)
    text = _KV_PASSWORD.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return mask_pii(text)


class PiiGuardFilter(logging.Filter):
    """日志记录的清理（见模块说明）。改写记录本身：之后的处理器、格式化拿到的都是清理过的。"""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(s for s in secrets if s)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001  参数和格式对不上：只留格式串
            message = str(record.msg)
        record.msg = scrub(message, self._secrets)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = scrub(record.exc_text, self._secrets)
        if record.stack_info:
            record.stack_info = scrub(record.stack_info, self._secrets)
        for name in EXTRA_FIELDS:
            value = record.__dict__.get(name)
            if isinstance(value, str):
                record.__dict__[name] = scrub(value, self._secrets)
        return True


class JsonFormatter(logging.Formatter):
    """一行一个 JSON 对象（见模块说明）。中文原样输出（ensure_ascii=False）。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for name in EXTRA_FIELDS:
            value = record.__dict__.get(name)
            if isinstance(value, str | int | float | bool):
                payload[name] = value
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            payload["exc"] = record.exc_text
        if record.stack_info:
            payload["stack"] = record.stack_info
        return json.dumps(payload, ensure_ascii=False, default=str)


LogFormat = Literal["json", "text"]


def configure_logging(
    level: str | int = "INFO",
    fmt: LogFormat = "json",
    *,
    secrets: Iterable[str] | None = None,
    stream: TextIO | None = None,
) -> logging.Handler:
    """给根日志器装（或换掉之前装的）处理器。secrets 为空时从 SECRET_ENV_NAMES 取值。

    uvicorn 用 log_config=None 启动时不改日志配置，它的 uvicorn.error / uvicorn.access
    记录冒泡到根日志器，同样经过过滤。可以重复调用（测试里换输出流）。"""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    setattr(handler, _HANDLER_MARK, True)
    handler.addFilter(PiiGuardFilter(secrets_from_env() if secrets is None else secrets))
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    elif fmt == "text":
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    else:
        raise ValueError(f"日志格式只能是 json 或 text：{fmt!r}")
    root.addHandler(handler)
    root.setLevel(level if isinstance(level, int) else level.upper())
    # uvicorn 按自己的 log_level / access_log 设过级别、关过冒泡的话都还原：由根日志器统一过滤和输出
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)
        logger.disabled = False
    return handler


__all__ = [
    "EXTRA_FIELDS",
    "REDACTED",
    "SECRET_ENV_NAMES",
    "JsonFormatter",
    "PiiGuardFilter",
    "configure_logging",
    "scrub",
    "secrets_from_env",
]
