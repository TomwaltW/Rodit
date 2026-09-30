"""单副本租约：一个平台单元同一时间只跑一个 serve 进程（方案 §9.1 单元化部署、§16 运维负担）。

为什么只能跑一个：会话串行锁、适配器的去重记录、短时缓存都在进程内（CLAUDE.md 选型记录：多实例
部署时再换 Redis）。两个进程同时接同一个平台单元的流量，同一会话的两条消息会被并发处理，去重记录
各记各的，同一条消息可能推荐两次。这个约束原来只写在文档里，这里在启动时强制执行：

- 租约 = PostgreSQL 会话级 advisory lock：pg_try_advisory_lock(LEASE_NAMESPACE, 平台键)，
  键按平台单元算（lease_key），同一个库里三个平台单元互不影响。用两个 int4 的形式：和迁移、
  撤回串行用的单个 bigint 键不在同一个键空间（pg_locks.objsubid 不同），不会互相挡住；
- 锁挂在一条专用连接上，进程活多久拿多久（不参与请求，不设语句超时）。进程退出（包括被杀）时
  连接断开，数据库自动释放；这条连接设了 TCP keepalive / tcp_user_timeout，主机宕机时数据库
  一分钟左右发现并释放，本进程这边的检查也不会一直挂着；
- serve 启动时拿不到（另一个进程占着）就拒绝启动：LeaseHeld，退出码 2，日志写明占用者
  （application_name 里有平台、主机名和进程号，不含连接串）；连不上数据库退出码 1（编排系统会
  重启重试）；
- 就绪检查多一项 lease（probe）：确认这条连接还活着、锁还在本连接上（查 pg_locks）。连接断了
  （数据库重启、网络故障）锁就没了，这时重连再抢：抢到了照常就绪（reacquired），被别的进程抢走了
  就一直不就绪（held_elsewhere，摘流量、日志报错），不会出现两个进程同时就绪；
- 连接串用 RODIT_DATABASE_DSN（应用的登录账号；advisory lock 不需要任何表权限，库没迁移也能拿）。
  必须直连 PostgreSQL 或经会话模式的连接池：事务模式的连接池（如 PgBouncer transaction 模式）下
  会话级的锁没有意义，就绪检查会发现锁不在本连接上（not_held）。

滚动发布：先停旧进程（它退出时释放租约），新进程再起来；新进程先起的话会以退出码 2 退出，由编排系统
重启重试，直到旧进程退出。
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import zlib
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol

import psycopg

from rodit.core.clock import CST
from rodit.core.enums import Platform
from rodit.core.errors import StorageUnavailable
from rodit.ops.health import CheckResult

logger = logging.getLogger("rodit.ops")

LEASE_NAMESPACE = 0x526F6469
"""两个 int4 形式的 advisory lock 的第一个键（"Rodi"）：本仓库所有单副本租约都在这个命名空间里。"""

DSN_ENV = "RODIT_DATABASE_DSN"
DEFAULT_CONNECT_TIMEOUT_S = 5
APPLICATION_NAME_PREFIX = "rodit-serve"

CLIENT_KEEPALIVE: Mapping[str, int] = {
    "keepalives": 1,
    "keepalives_idle": 10,
    "keepalives_interval": 5,
    "keepalives_count": 3,
    "tcp_user_timeout": 5000,  # 毫秒：发出去的数据 5 秒没有确认就断开（检查不会一直挂着）
}
"""租约连接的 libpq 参数（本进程这边发现连接已断）。只作用于 TCP 连接，unix socket 上忽略。"""

SERVER_KEEPALIVE: tuple[tuple[str, str], ...] = (
    ("tcp_keepalives_idle", "30"),
    ("tcp_keepalives_interval", "10"),
    ("tcp_keepalives_count", "3"),
)
"""租约连接的服务端设置（数据库发现本进程所在主机没了，一分钟左右释放锁）。普通用户可以设。"""

_HOLDS_LOCK = (
    "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND granted"
    " AND pid = pg_backend_pid() AND objsubid = 2"
    " AND classid = %s::bigint::oid AND objid = %s::bigint::oid)"
)
_HOLDER = (
    "SELECT a.application_name, a.backend_start FROM pg_locks l"
    " JOIN pg_stat_activity a ON a.pid = l.pid"
    " WHERE l.locktype = 'advisory' AND l.granted AND l.objsubid = 2"
    " AND l.classid = %s::bigint::oid AND l.objid = %s::bigint::oid"
    " AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database())"
    " LIMIT 1"
)


def lease_key(platform: Platform | str) -> tuple[int, int]:
    """平台单元 → (LEASE_NAMESPACE, 平台键)。平台键是 crc32（跨进程、跨版本稳定），折成 int4。"""
    value = Platform(platform).value
    raw = zlib.crc32(f"rodit.serve:{value}".encode())
    return LEASE_NAMESPACE, raw - 2**32 if raw >= 2**31 else raw


def _unsigned(key: int) -> int:
    """int4 键在 pg_locks 里是 oid（无符号）。"""
    return key & 0xFFFFFFFF


class LeaseError(RuntimeError):
    """拿不到单副本租约（配置不对或已被占用）。消息里没有连接串。"""


class LeaseHeld(LeaseError):
    """另一个进程占着这个平台单元的租约。"""

    def __init__(self, platform: Platform | str, *, holder: str | None) -> None:
        self.platform = Platform(platform)
        self.holder = holder
        who = f"（占用者：{holder}）" if holder else ""
        super().__init__(
            f"平台单元 {self.platform.value} 已有一个 serve 进程在运行{who}：一个平台单元只能跑"
            "一个进程（会话串行锁、去重记录、短时缓存都在进程内，§9.1）。确认旧进程已经退出后再启动；"
            "旧进程退出时数据库自动释放租约"
        )


class Lease(Protocol):
    """serve 用到的租约接口（ServeLease；测试注入假的）。"""

    def acquire(self) -> object: ...

    def release(self) -> None: ...

    def probe(self) -> Mapping[str, CheckResult]: ...


def default_application_name(platform: Platform | str) -> str:
    """rodit-serve/平台/主机名/进程号（PostgreSQL 只保留前 63 个字节）。不含任何密钥。"""
    host = "".join(c for c in socket.gethostname() if c.isascii() and c.isprintable())[:32]
    return f"{APPLICATION_NAME_PREFIX}/{Platform(platform).value}/{host or '?'}/{os.getpid()}"[:63]


class ServeLease:
    """一个平台单元的单副本租约（见模块说明）。线程安全；acquire / release 都可以重复调用。

    dsn 不给时在 acquire 时从环境变量 dsn_env 读（构造时不读、不连库）。
    """

    def __init__(
        self,
        platform: Platform | str,
        *,
        dsn: str | None = None,
        dsn_env: str = DSN_ENV,
        connect_timeout_s: int = DEFAULT_CONNECT_TIMEOUT_S,
        application_name: str | None = None,
    ) -> None:
        self.platform = Platform(platform)
        self.key = lease_key(self.platform)
        self._dsn = dsn
        self._dsn_env = dsn_env
        self._connect_timeout_s = connect_timeout_s
        self.application_name = application_name or default_application_name(self.platform)
        self._lock = threading.Lock()
        self._conn: psycopg.Connection[Any] | None = None
        self._state = "idle"  # idle → held ⇄ lost → released

    def __repr__(self) -> str:  # 连接串可能含密码，不出现在日志里
        return f"ServeLease(platform={self.platform.value!r}, held={self.held}, dsn=***)"

    def __enter__(self) -> ServeLease:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()

    @property
    def held(self) -> bool:
        return self._conn is not None

    # ---------------------------------------------------------------- 拿、放

    def acquire(self) -> ServeLease:
        """拿租约。已被占用抛 LeaseHeld；连不上数据库抛 StorageUnavailable。"""
        with self._lock:
            if self._conn is not None:
                return self
            self._conn = self._try_lock()
            self._state = "held"
        logger.info("已拿到单副本租约：platform=%s", self.platform.value)
        return self

    def release(self) -> None:
        """放租约（进程退出前调用；没拿到时什么也不做）。"""
        with self._lock:
            conn, self._conn = self._conn, None
            if self._state != "idle":
                self._state = "released"
        if conn is None:
            return
        try:
            conn.execute("SELECT pg_advisory_unlock(%s, %s)", self.key)
        except psycopg.Error:
            pass  # 连接已经断了：锁随连接一起释放了
        finally:
            conn.close()
        logger.info("已释放单副本租约：platform=%s", self.platform.value)

    # ---------------------------------------------------------------- 就绪检查

    def probe(self) -> dict[str, CheckResult]:
        """就绪检查的 lease 项（原因代码：not_acquired / released / held_elsewhere / lost /
        not_held）。连接断了时重连再抢，抢到了是 ok + reacquired。"""
        with self._lock:
            if self._state in ("idle", "released"):
                reason = "not_acquired" if self._state == "idle" else "released"
                return {"lease": CheckResult(False, {"reason": reason})}
            conn = self._conn
            if conn is not None:
                try:
                    row = conn.execute(_HOLDS_LOCK, self._oid_key()).fetchone()
                except psycopg.Error as exc:
                    logger.error("单副本租约的连接断了（%s），重新获取", type(exc).__name__)
                else:
                    if row is not None and row[0]:
                        return {"lease": CheckResult(True, {})}
                    # 连接还在、锁却不在本连接上：事务模式的连接池，或者被人手工释放了
                    logger.error("单副本租约不在本进程的连接上（连接池是事务模式？），重新获取")
                self._close(conn)
                self._conn = None
                self._state = "lost"
            try:
                self._conn = self._try_lock()
            except LeaseHeld as exc:
                logger.error("单副本租约被另一个进程拿走了，本进程不就绪：%s", exc.holder or "?")
                return {"lease": CheckResult(False, {"reason": "held_elsewhere"})}
            except (StorageUnavailable, psycopg.Error, LeaseError) as exc:
                logger.error("单副本租约重新获取失败：%s", type(exc).__name__)
                return {"lease": CheckResult(False, {"reason": "lost"})}
            self._state = "held"
            logger.warning("单副本租约已重新获取：platform=%s", self.platform.value)
            return {"lease": CheckResult(True, {"reacquired": True})}

    # ---------------------------------------------------------------- 内部

    def _oid_key(self) -> tuple[int, int]:
        return _unsigned(self.key[0]), _unsigned(self.key[1])

    def _resolve_dsn(self) -> str:
        dsn = self._dsn if self._dsn is not None else os.environ.get(self._dsn_env, "")
        if not dsn.strip():
            raise LeaseError(f"缺少必需的环境变量：{self._dsn_env}（单副本租约要连数据库）")
        return dsn

    def _try_lock(self) -> psycopg.Connection[Any]:
        """新开一条连接抢锁：抢到返回连接；被占用时关掉连接、抛 LeaseHeld。调用方持有 _lock。"""
        dsn = self._resolve_dsn()
        try:
            conn = psycopg.connect(
                dsn,
                autocommit=True,
                connect_timeout=self._connect_timeout_s,
                application_name=self.application_name,
                **CLIENT_KEEPALIVE,
            )
        except psycopg.OperationalError as exc:
            raise StorageUnavailable("数据库连接失败（单副本租约）") from exc
        try:
            for name, value in SERVER_KEEPALIVE:
                conn.execute("SELECT set_config(%s, %s, false)", (name, value))
            row = conn.execute("SELECT pg_try_advisory_lock(%s, %s)", self.key).fetchone()
            if row is not None and row[0]:
                return conn
            holder = self._holder(conn)
        except BaseException:
            self._close(conn)
            raise
        self._close(conn)
        raise LeaseHeld(self.platform, holder=holder)

    def _holder(self, conn: psycopg.Connection[Any]) -> str | None:
        """占用者的 application_name 和连接时间（看不到时返回 None，不影响报错）。"""
        try:
            row = conn.execute(_HOLDER, self._oid_key()).fetchone()
        except psycopg.Error:
            return None
        if row is None or not row[0]:
            return None
        name = "".join(c for c in str(row[0]) if c.isprintable())[:63]
        started = row[1]
        if isinstance(started, datetime):
            return f"{name}，自 {started.astimezone(CST):%Y-%m-%d %H:%M:%S} 起"
        return name

    @staticmethod
    def _close(conn: psycopg.Connection[Any]) -> None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001  关不掉也没关系：连接已经不用了
            pass


def serve_lease(platform: Platform | str) -> ServeLease:
    """serve 默认用的租约：连接串从 RODIT_DATABASE_DSN 读（acquire 时）。"""
    return ServeLease(platform)


__all__ = [
    "APPLICATION_NAME_PREFIX",
    "LEASE_NAMESPACE",
    "Lease",
    "LeaseError",
    "LeaseHeld",
    "ServeLease",
    "default_application_name",
    "lease_key",
    "serve_lease",
]
