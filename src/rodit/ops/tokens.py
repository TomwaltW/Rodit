"""生产用的调用方鉴权：从挂载的密钥文件读访问令牌的哈希（方案 §8.4、§9.2、§12.3）。

文件格式（JSON，放在平台云的密钥管理里，以文件方式挂进容器，默认 /run/secrets/rodit_tokens.json）::

    {
      "version": 1,
      "principals": [
        {"sha256": "<令牌的 SHA-256，64 位小写十六进制>",
         "shop_id": "s_1001", "user_id": "kf_02", "role": "agent",
         "expires_at": "2026-12-31T23:59:59+08:00"}          可选：到期后拒绝
      ],
      "metrics": [
        {"sha256": "<抓取 /metrics 用的令牌的 SHA-256>", "name": "prometheus"}
      ]
    }

- 文件里只有哈希，没有令牌原文：文件泄露也拿不到能用的令牌。令牌是 32 字节随机数
  （issue_token 生成，前缀 rdt_），直接 SHA-256 就足够，不需要加盐慢哈希；按哈希查表，
  攻击者控制不了哈希值，查表耗时不泄露令牌内容。
- 令牌原文只在 issue_token 生成时出现一次（打印给运维分发），之后不落盘、不进日志、不进 repr。
- 按间隔（reload_interval_s）检查文件是否变化（大小、修改时间、inode），变了就重新读——
  平台云的密钥轮换（K8s Secret 换软链接）不用重启。新文件读不出来时继续用上一版并记警告
  （只记异常类型）。
- 身份对象复用接口层的 rodit.api.Principal（角色、店铺、"客服必须是具体的人" 这些校验都一样）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from rodit.api.auth import MIN_TOKEN_CHARS, Principal, PrincipalRole
from rodit.core.clock import Clock, SystemClock

logger = logging.getLogger(__name__)

TOKEN_PREFIX = "rdt_"
TOKEN_FILE_VERSION = 1
MAX_TOKEN_CHARS = 512
_HEX64 = r"^[0-9a-f]{64}$"


def hash_token(token: str) -> str:
    """令牌 → 文件里存的哈希（SHA-256，小写十六进制）。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    """生成一个新令牌：rdt_ + 32 字节随机数（URL 安全的 base64）。"""
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


class PrincipalEntry(BaseModel):
    """令牌文件里的一个调用方。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(pattern=_HEX64)
    shop_id: str
    user_id: str
    role: PrincipalRole
    expires_at: AwareDatetime | None = None

    def principal(self) -> Principal:
        return Principal(shop_id=self.shop_id, user_id=self.user_id, role=self.role)

    @model_validator(mode="after")
    def _valid_principal(self) -> PrincipalEntry:
        self.principal()  # 店铺、账号、角色的校验和接口层一致
        return self


class MetricsEntry(BaseModel):
    """抓取 /metrics 的令牌（Prometheus 等），不属于任何店铺。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(pattern=_HEX64)
    name: str = Field(min_length=1, max_length=64)
    expires_at: AwareDatetime | None = None


class TokenFile(BaseModel):
    """令牌文件（见模块说明）。同一个哈希不能出现两次。"""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = TOKEN_FILE_VERSION
    principals: list[PrincipalEntry] = Field(default_factory=list)
    metrics: list[MetricsEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> TokenFile:
        seen: set[str] = set()
        for entry in [*self.principals, *self.metrics]:
            if entry.sha256 in seen:
                raise ValueError("令牌文件里有重复的令牌哈希")
            seen.add(entry.sha256)
        return self

    @classmethod
    def load(cls, path: str | Path) -> TokenFile:
        text = Path(path).read_text(encoding="utf-8")
        return cls.model_validate(json.loads(text))

    def dump(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"


class TokenFileError(RuntimeError):
    """令牌文件不存在、读不出来或格式不对。消息里只有路径和原因类别，没有文件内容。"""


def _load(path: Path) -> TokenFile:
    try:
        return TokenFile.load(path)
    except FileNotFoundError as e:
        raise TokenFileError(f"令牌文件不存在：{path}") from e
    except (OSError, UnicodeDecodeError) as e:
        raise TokenFileError(f"令牌文件读不出来：{path}（{type(e).__name__}）") from e
    except (ValueError, ValidationError) as e:  # json.JSONDecodeError 是 ValueError
        raise TokenFileError(f"令牌文件格式不对：{path}（{type(e).__name__}）") from None


@dataclass(frozen=True)
class _Table:
    principals: dict[str, PrincipalEntry]
    metrics: dict[str, MetricsEntry]
    stamp: tuple[int, int, int]


def _stamp(path: Path) -> tuple[int, int, int]:
    st = path.stat()  # 跟随软链接：K8s Secret 轮换时软链接指向新文件
    return (st.st_mtime_ns, st.st_size, st.st_ino)


class TokenFileAuthenticator:
    """实现 rodit.api.Authenticator：令牌 → Principal（按哈希查令牌文件）。

    构造时读一次文件（读不出来抛 TokenFileError，启动失败）；之后每 reload_interval_s 秒最多
    检查一次文件是否变化。reload_interval_s=0 表示每次都检查（测试用）。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        reload_interval_s: float = 30.0,
        clock: Clock | None = None,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        if reload_interval_s < 0:
            raise ValueError("reload_interval_s 不能为负")
        self._path = Path(path)
        self._interval = reload_interval_s
        self._clock = clock or SystemClock()
        self._timer = timer
        self._lock = threading.Lock()
        self._table = self._read()
        self._checked_at = self._timer()

    def __repr__(self) -> str:
        table = self._table
        return (
            f"TokenFileAuthenticator(path={str(self._path)!r}, "
            f"<{len(table.principals)} 个调用方令牌>, <{len(table.metrics)} 个指标令牌>)"
        )

    @property
    def path(self) -> Path:
        return self._path

    def _read(self) -> _Table:
        try:
            stamp = _stamp(self._path)
        except FileNotFoundError as e:
            raise TokenFileError(f"令牌文件不存在：{self._path}") from e
        except OSError as e:
            raise TokenFileError(f"令牌文件读不出来：{self._path}（{type(e).__name__}）") from e
        data = _load(self._path)
        return _Table(
            principals={p.sha256: p for p in data.principals},
            metrics={m.sha256: m for m in data.metrics},
            stamp=stamp,
        )

    def reload(self) -> bool:
        """立即检查文件；变了就重新读。读不出来时保留上一版，返回 False。"""
        with self._lock:
            self._checked_at = self._timer()
            try:
                if _stamp(self._path) == self._table.stamp:
                    return True
                self._table = self._read()
            except (OSError, TokenFileError) as exc:
                logger.warning("令牌文件重新加载失败，继续使用上一版：%s", type(exc).__name__)
                return False
            logger.info(
                "令牌文件已重新加载：%d 个调用方令牌，%d 个指标令牌",
                len(self._table.principals),
                len(self._table.metrics),
            )
            return True

    def _current(self) -> _Table:
        if self._timer() - self._checked_at >= self._interval:
            self.reload()
        return self._table

    def _valid(self, entry: PrincipalEntry | MetricsEntry | None) -> bool:
        return entry is not None and (
            entry.expires_at is None or entry.expires_at > self._clock.now()
        )

    @staticmethod
    def _digest(token: object) -> str | None:
        if not isinstance(token, str) or not MIN_TOKEN_CHARS <= len(token) <= MAX_TOKEN_CHARS:
            return None
        return hash_token(token)

    def authenticate(self, token: str) -> Principal | None:
        digest = self._digest(token)
        if digest is None:
            return None
        entry = self._current().principals.get(digest)
        return entry.principal() if self._valid(entry) and entry is not None else None

    def authenticate_metrics(self, token: str | None) -> bool:
        """/metrics 的抓取令牌是否有效。"""
        digest = self._digest(token)
        return digest is not None and self._valid(self._current().metrics.get(digest))

    @property
    def has_metrics_tokens(self) -> bool:
        return bool(self._current().metrics)

    def shop_ids(self) -> list[str]:
        """令牌文件里出现的店铺（本单元服务的店铺，授权过期告警用）。"""
        return sorted({p.shop_id for p in self._current().principals.values()})


# ================================================================ 签发（运维命令）


@dataclass(frozen=True)
class IssuedToken:
    token: str  # 原文：只打印一次
    entry: PrincipalEntry | MetricsEntry

    def __repr__(self) -> str:
        return f"IssuedToken(entry={self.entry!r}, token=***)"


def issue_principal_token(
    shop_id: str, user_id: str, role: PrincipalRole, *, expires_at: datetime | None = None
) -> IssuedToken:
    token = new_token()
    entry = PrincipalEntry(
        sha256=hash_token(token),
        shop_id=shop_id,
        user_id=user_id,
        role=role,
        expires_at=expires_at,
    )
    return IssuedToken(token, entry)


def issue_metrics_token(name: str, *, expires_at: datetime | None = None) -> IssuedToken:
    token = new_token()
    return IssuedToken(
        token, MetricsEntry(sha256=hash_token(token), name=name, expires_at=expires_at)
    )


def append_entries(path: str | Path, entries: Iterable[PrincipalEntry | MetricsEntry]) -> TokenFile:
    """把新条目加进令牌文件（没有就新建）；原子替换，权限 0600。返回写入后的内容。"""
    target = Path(path)
    data = _load(target) if target.exists() else TokenFile()  # 格式不对 → TokenFileError
    principals = list(data.principals)
    metrics = list(data.metrics)
    for entry in entries:
        (principals if isinstance(entry, PrincipalEntry) else metrics).append(entry)  # type: ignore[arg-type]
    updated = TokenFile(principals=principals, metrics=metrics)
    write_token_file(target, updated)
    return updated


def write_token_file(path: str | Path, data: TokenFile | Mapping[str, Any]) -> None:
    """原子写入令牌文件（先写临时文件再改名），权限 0600。"""
    target = Path(path)
    content = data if isinstance(data, TokenFile) else TokenFile.model_validate(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content.dump())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


__all__ = [
    "MAX_TOKEN_CHARS",
    "TOKEN_PREFIX",
    "IssuedToken",
    "MetricsEntry",
    "PrincipalEntry",
    "TokenFile",
    "TokenFileAuthenticator",
    "TokenFileError",
    "append_entries",
    "hash_token",
    "issue_metrics_token",
    "issue_principal_token",
    "new_token",
    "write_token_file",
]
