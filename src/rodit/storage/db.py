"""数据库连接、迁移与带店铺范围的事务（方案 §9.2 租户隔离、§10 数据模型）。

- migrate()：按版本执行 MIGRATIONS 里的脚本，已执行的记在 schema_migrations，重复调用不做事；
  用事务级 advisory lock 串行化，多个实例同时启动也只执行一次。迁移用连接串里的登录角色
  （表的所有者）执行，不切换到应用角色。
- session(shop_id)：一个事务。配置了 app_role 时先 SET LOCAL ROLE（应用角色没有 BYPASSRLS，
  行级安全生效），再用 set_config('rodit.shop_id', …, true) 设置本事务的店铺范围。
  仓储的每个方法都开一个 session，SQL 里仍然显式写 WHERE shop_id，行级安全是第二道防线。
- 每个 session 新建一条连接、用完关闭：简单、店铺范围不会残留到别的请求。
  MVP 的吞吐够用（unix socket / 同机房连接毫秒级）；需要时换连接池（psycopg_pool），
  店铺范围和角色都是 SET LOCAL / set_config(…, true)，事务结束自动恢复，换池不用改仓储代码。

生产部署（每个平台单元一份库，§9.1）：
- 迁移账号：库的所有者，执行 migrate()
  （需要能建 rodit_app / rodit_trainer 角色，或由运维预先建好）。
- 应用账号：一个登录角色，GRANT rodit_app TO 它（不能是超级用户、不能带 BYPASSRLS）；
  连接串放在环境变量里（默认 RODIT_DATABASE_DSN），不进仓库。
- 训练导出任务：单独的登录角色，只 GRANT rodit_trainer（见 schema.sql 开头的说明）。
- 授权令牌在应用里加解密（PgShopAuthRepository），密钥和明文令牌不作为查询参数发给数据库，
  数据库的语句日志 / 慢查询日志里不会出现它们。
- 请求用的 Database 要设 statement_timeout_ms / lock_timeout_ms（生产组装从环境变量读，
  见 rodit.app.production）：库的默认值是不限时，语句被锁住或存储卡住时请求线程会一直挂着，
  "超时 → KnowledgeUnavailable → 不推荐"（§12.2）也就触发不了。迁移用的连接不设；
  §8.4 清理店铺数据（各仓储的 purge_shop）用 session(…, unbounded=True)，也不受这两个上限约束。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from rodit.core.errors import RoditError, StorageUnavailable  # noqa: F401  (re-export)

APP_ROLE = "rodit_app"
TRAINER_ROLE = "rodit_trainer"
SHOP_SETTING = "rodit.shop_id"

BUSINESS_TABLES: tuple[str, ...] = (
    "shop",
    "shop_auth",
    "shop_policy",
    "item_card",
    "knowledge_item",
    "conversation",
    "message",
    "decision_log",
    "suggestion",
    "aftersale_review",
    "audit_log",
    "knowledge_gap",
    "labeled_sample",
)
"""所有带店铺归属的表，全部启用并强制行级安全（shop 表的店铺列是 id，其余是 shop_id）。"""

_MIGRATION_LOCK = 0x526F646974  # advisory lock 的键（"Rodit"），只用于串行化迁移

Conn = psycopg.Connection[dict[str, Any]]


@dataclass(frozen=True)
class Migration:
    version: int
    name: str  # rodit.storage 包内的 SQL 文件名

    def sql(self) -> str:
        return resources.files("rodit.storage").joinpath(self.name).read_text(encoding="utf-8")


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "schema.sql"),
    Migration(2, "migration_0002_aftersale_status_version.sql"),
)
"""按版本升序；已发布的脚本不再修改，结构变化加新版本（tests/storage/test_migrations.py
按内容哈希锁定已发布的脚本）。"""


class Database:
    """PostgreSQL 连接工厂。

    - dsn：libpq 连接串，如 "dbname=rodit" 或 "postgresql://user@host/rodit"。
    - app_role：session 里 SET LOCAL ROLE 的角色，默认 rodit_app；None 表示不切换
      （登录角色本身就是受行级安全约束的应用角色，或测试里用管理员身份直接读写）。
    - trainer_role：export 训练数据时切换的角色（session(None, trainer=True)）。
    - statement_timeout_ms：每个事务的语句超时（§12.1 检索 ≤ 0.5 秒），None 用库的默认值。
    - lock_timeout_ms：每个事务等锁的上限（被清理、迁移等长事务挡住时尽快报错降级），
      None 用库的默认值。超时都抛 psycopg.OperationalError 的子类，知识库仓储转成
      KnowledgeUnavailable。
    """

    def __init__(
        self,
        dsn: str,
        *,
        app_role: str | None = APP_ROLE,
        trainer_role: str | None = TRAINER_ROLE,
        connect_timeout_s: int = 5,
        statement_timeout_ms: int | None = None,
        lock_timeout_ms: int | None = None,
    ) -> None:
        if not dsn:
            raise ValueError("数据库连接串不能为空")
        if statement_timeout_ms is not None and statement_timeout_ms <= 0:
            raise ValueError("statement_timeout_ms 必须为正数")
        if lock_timeout_ms is not None and lock_timeout_ms <= 0:
            raise ValueError("lock_timeout_ms 必须为正数")
        self._dsn = dsn
        self.app_role = app_role
        self.trainer_role = trainer_role
        self._connect_timeout_s = connect_timeout_s
        self._statement_timeout_ms = statement_timeout_ms
        self._lock_timeout_ms = lock_timeout_ms

    @property
    def statement_timeout_ms(self) -> int | None:
        return self._statement_timeout_ms

    @property
    def lock_timeout_ms(self) -> int | None:
        return self._lock_timeout_ms

    def __repr__(self) -> str:  # 连接串可能含密码，不出现在日志里
        return f"Database(app_role={self.app_role!r}, dsn=***)"

    @classmethod
    def from_env(cls, env_name: str = "RODIT_DATABASE_DSN", **kwargs: Any) -> Database:
        dsn = os.environ.get(env_name, "")
        if not dsn:
            raise RuntimeError(f"环境变量 {env_name} 未设置：需要 PostgreSQL 连接串")
        return cls(dsn, **kwargs)

    # ---------------------------------------------------------------- 连接

    def _connect(self) -> Conn:
        try:
            return psycopg.connect(
                self._dsn, row_factory=dict_row, connect_timeout=self._connect_timeout_s
            )
        except psycopg.OperationalError as e:
            raise StorageUnavailable("数据库连接失败") from e

    @contextmanager
    def session(
        self, shop_id: str | None, *, trainer: bool = False, unbounded: bool = False
    ) -> Iterator[Conn]:
        """一个事务：正常退出提交，异常回滚，连接随即关闭。

        shop_id=None 表示不设店铺范围：在行级安全下读不到任何业务数据，只能写没有店铺归属的
        审计事件。trainer=True 切换到训练导出角色（只能读 labeled_sample）。

        unbounded=True 不设本库配置的语句 / 等锁超时（用库的默认值，与 migrate() 相同），
        只给 §8.4 清理店铺数据（purge_shop）用：大店铺一次删几百万行，远超请求级的上限，
        每次都被取消回滚的话令牌已删、数据却永远删不掉。角色和店铺范围照常设置，行级安全
        仍然生效。请求路径一律用默认值（有上限）。
        """
        if shop_id is not None and not shop_id.strip():
            raise ValueError("shop_id 不能是空字符串")
        role = self.trainer_role if trainer else self.app_role
        if trainer and role is None:
            raise ValueError("没有配置训练导出角色（trainer_role）")
        conn = self._connect()
        with conn:  # 退出时提交或回滚，然后关闭
            conn.execute("SET LOCAL TIME ZONE 'UTC'")  # 读出的时间统一再转成东八区
            if role is not None:
                conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))
            conn.execute("SELECT set_config(%s, %s, true)", (SHOP_SETTING, shop_id or ""))
            for setting, value in (
                ("statement_timeout", self._statement_timeout_ms),
                ("lock_timeout", self._lock_timeout_ms),
            ):
                if value is not None and not unbounded:
                    conn.execute("SELECT set_config(%s, %s, true)", (setting, str(value)))
            yield conn

    # ---------------------------------------------------------------- 迁移

    def migrate(self) -> list[int]:
        """执行未执行过的迁移，返回本次执行的版本号（已是最新时为空列表）。"""
        conn = self._connect()
        with conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version INT PRIMARY KEY,"
                " name TEXT NOT NULL,"
                " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            done = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations")}
            applied: list[int] = []
            for m in sorted(MIGRATIONS, key=lambda m: m.version):
                if m.version in done:
                    continue
                conn.execute(m.sql())  # 无参数：按简单查询协议执行多条语句
                conn.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                    (m.version, m.name),
                )
                applied.append(m.version)
            return applied

    def applied_versions(self) -> list[int]:
        conn = self._connect()
        with conn:
            exists = conn.execute("SELECT to_regclass('schema_migrations') AS t").fetchone()
            if exists is None or exists["t"] is None:
                return []
            rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version")
            return [r["version"] for r in rows]
