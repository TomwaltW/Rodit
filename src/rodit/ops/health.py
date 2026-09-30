"""就绪检查（/readyz，方案 §12.2 可用性、§9.1 单元化部署）。

就绪 = 这个进程现在可以接流量：
- database：能用应用角色连上数据库（Database.session，受语句超时约束）；
- migrations：schema_migrations 里已经有本版本代码知道的全部迁移（MIGRATIONS）；
  库里多出本版本不认识的更新迁移（新版本先迁移、旧实例还在跑）仍算就绪，detail 里列出来；
- llm：大模型地址、模型名已配置，密钥环境变量有值（只看有没有，不读出来、不输出）。
  不真的调用模型：探针很频繁，调用要花钱，模型故障本来就按 §12.2 降级。

存活（/healthz）只说明进程活着，不查任何依赖——依赖故障时重启进程没有用，应该摘流量。

返回内容只有检查名、是否通过和原因代码（unreachable / timeout / not_migrated / behind /
no_privilege / missing_api_key …），没有连接串、异常说明和店铺数据，不需要鉴权。
检查结果缓存 cache_s 秒（默认 2 秒），探针再频繁也不会把数据库连接打满。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import psycopg

from rodit.core.config import LLMConfig
from rodit.core.errors import StorageUnavailable
from rodit.storage import MIGRATIONS, Migration

logger = logging.getLogger(__name__)

DEFAULT_CACHE_S = 2.0


@dataclass(frozen=True)
class CheckResult:
    """一项检查的结果。detail 只放原因代码和数字，不放异常说明。"""

    ok: bool
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, **dict(self.detail)}


@dataclass(frozen=True)
class Readiness:
    ready: bool
    checks: Mapping[str, CheckResult]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.ready else "not_ready",
            "checks": {name: r.as_dict() for name, r in self.checks.items()},
        }


Probe = Callable[[], Mapping[str, CheckResult]]
"""一个探针可以一次给出几项检查（数据库连通和迁移版本共用一条连接）。"""


class SessionDatabase(Protocol):
    """rodit.storage.Database 里就绪检查用到的部分。"""

    def session(self, shop_id: str | None, **kw: Any) -> Any: ...


class ReadinessChecker:
    """跑全部探针并缓存结果。探针自己抛异常时该探针记为失败（原因 error），不影响其他探针。"""

    def __init__(
        self,
        probes: Mapping[str, Probe],
        *,
        cache_s: float = DEFAULT_CACHE_S,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        if cache_s < 0:
            raise ValueError("cache_s 不能为负")
        self._probes = dict(probes)
        self._cache_s = cache_s
        self._timer = timer
        self._lock = threading.Lock()
        self._cached: tuple[float, Readiness] | None = None

    def check(self, *, fresh: bool = False) -> Readiness:
        with self._lock:  # 同时来的探针只查一次
            now = self._timer()
            if not fresh and self._cached is not None and now - self._cached[0] < self._cache_s:
                return self._cached[1]
            results: dict[str, CheckResult] = {}
            for name, probe in self._probes.items():
                try:
                    results.update(probe())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("就绪检查 %s 出错：%s", name, type(exc).__name__)
                    results[name] = CheckResult(False, {"reason": "error"})
            readiness = Readiness(
                ready=bool(results) and all(r.ok for r in results.values()), checks=results
            )
            if not readiness.ready:
                failed = sorted(n for n, r in results.items() if not r.ok)
                logger.warning("未就绪：%s", ",".join(failed))
            self._cached = (self._timer(), readiness)
            return readiness


# ================================================================ 数据库 + 迁移


def database_probe(db: SessionDatabase, migrations: Iterable[Migration] = MIGRATIONS) -> Probe:
    """用应用角色开一个事务，读 schema_migrations。给出 database、migrations 两项。

    应用角色（rodit_app）读 schema_migrations 需要 SELECT 权限：rodit.ops.migrate 在迁移后授权；
    没有权限时 migrations 记为 no_privilege（数据库本身算连通）。
    """
    known = sorted({m.version for m in migrations})
    latest = known[-1] if known else 0

    def probe() -> dict[str, CheckResult]:
        started = time.perf_counter()
        connected = False  # 进了事务（连上、切到应用角色、设好店铺范围）之后才算数据库可用
        try:
            with db.session(None) as conn:
                connected = True
                rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        except StorageUnavailable:
            return _db_down("unreachable")
        except psycopg.Error as exc:
            reason = _reason(exc)
            if not connected:
                if reason not in {"timeout", "unreachable"}:
                    # 如登录角色不是 rodit_app 的成员（SET ROLE 被拒）：应用本身就跑不起来
                    logger.warning("就绪检查：数据库错误 %s", type(exc).__name__)
                return _db_down(reason)
            if reason in {"not_migrated", "no_privilege"}:
                return _db_up(started, CheckResult(False, {"reason": reason, "latest": latest}))
            logger.warning("就绪检查：读迁移版本出错 %s", type(exc).__name__)
            return _db_down(reason)
        applied = {int(_row_version(r)) for r in rows}
        pending = [v for v in known if v not in applied]
        unknown = sorted(v for v in applied if v not in set(known))
        detail: dict[str, Any] = {"current": max(applied) if applied else None, "latest": latest}
        if pending:
            detail["reason"] = "behind" if applied else "not_migrated"
            detail["pending"] = pending
        if unknown:
            detail["unknown"] = unknown  # 更新版本的迁移：旧实例照常就绪
        return _db_up(started, CheckResult(not pending, detail))

    return probe


def _reason(exc: BaseException) -> str:
    """psycopg 异常 → 原因代码（不含异常说明）。"""
    if isinstance(exc, psycopg.errors.UndefinedTable):
        return "not_migrated"
    if isinstance(exc, psycopg.errors.InsufficientPrivilege):
        return "no_privilege"
    if isinstance(exc, psycopg.errors.QueryCanceled | psycopg.errors.LockNotAvailable):
        return "timeout"
    if isinstance(exc, psycopg.OperationalError):
        return "unreachable"
    return "error"


def _row_version(row: Any) -> Any:
    return row["version"] if isinstance(row, Mapping) else row[0]


def _db_down(reason: str) -> dict[str, CheckResult]:
    return {
        "database": CheckResult(False, {"reason": reason}),
        "migrations": CheckResult(False, {"reason": "unknown"}),
    }


def _db_up(started: float, migrations: CheckResult) -> dict[str, CheckResult]:
    latency_ms = round((time.perf_counter() - started) * 1000)
    return {
        "database": CheckResult(True, {"latency_ms": latency_ms}),
        "migrations": migrations,
    }


# ================================================================ 大模型


def llm_probe(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
    env_prefix: str = "RODIT_LLM_",
) -> Probe:
    """大模型是否配好：地址、模型名（配置文件优先，否则环境变量），密钥环境变量有值。
    和 rodit.app.production.build_production_unit 取值的规则一致。只看有没有，不输出值。"""

    def probe() -> dict[str, CheckResult]:
        env = os.environ if environ is None else environ
        base_url = config.base_url or env.get(f"{env_prefix}BASE_URL", "")
        model = config.model or env.get(f"{env_prefix}MODEL", "")
        if not base_url.strip() or not model.strip():
            return {"llm": CheckResult(False, {"reason": "not_configured"})}
        if not env.get(config.api_key_env, "").strip():
            return {"llm": CheckResult(False, {"reason": "missing_api_key"})}
        return {"llm": CheckResult(True, {})}

    return probe


def static_probe(name: str, result: CheckResult) -> Probe:
    """固定结果的探针（如本地试用的离线大模型替身：{"offline": true}）。"""
    return lambda: {name: result}


__all__ = [
    "DEFAULT_CACHE_S",
    "CheckResult",
    "Probe",
    "Readiness",
    "ReadinessChecker",
    "database_probe",
    "llm_probe",
    "static_probe",
]
