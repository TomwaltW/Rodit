"""平台客服考核指标的仓储（core.ports.PlatformMetricsRepository；方案 §12.4、§9.2、§8.4）：
内存实现和 PostgreSQL 实现跑同一套契约（含串店）；PostgreSQL 另测行级安全（训练导出角色读不到）、
数值约束、控制字符清洗、迁移版本 6 的升级路径、取消订购时一并分批删除。
@pytest.mark.pg 的用例连不上 PostgreSQL 时跳过。全部是合成数据。"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, timedelta

import psycopg
import pytest
from pydantic import ValidationError

from rodit.adapters.auth import AuthGuard
from rodit.app.production import postgres_repos
from rodit.core.enums import MetricSource, PlatformMetric
from rodit.core.models import PlatformMetricSample
from rodit.core.ports import PlatformMetricsRepository
from rodit.inmemory import InMemoryPlatformMetricsRepository
from rodit.storage import MIGRATIONS, Database, PgPlatformMetricsRepository
from rodit.storage import db as db_module
from rodit.testing import contracts
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A, SHOP_B, build_demo_world

NOW = build_demo_world().now
DAY = date(2026, 9, 20)
TB = PlatformMetric.TAOBAO_3MIN_RESPONSE_RATE


def _row(shop_id: str, day: date = DAY, value: float = 0.9, **kw: object) -> PlatformMetricSample:
    data: dict[str, object] = {
        "shop_id": shop_id,
        "metric": TB,
        "day": day,
        "value": value,
        "source": MetricSource.PLATFORM,
        "sample_size": 100,
        "recorded_at": NOW,
        **kw,
    }
    return PlatformMetricSample.model_validate(data)


# ================================================================ 内存实现


def test_in_memory_repository_contract():
    contracts.check_platform_metrics_repository(InMemoryPlatformMetricsRepository)


def test_in_memory_cross_shop_isolation():
    contracts.check_platform_metrics_cross_shop_isolation(InMemoryPlatformMetricsRepository)


def test_recorded_by_is_cleaned_by_the_model_and_stored_cleaned_in_memory():
    """登记人账号是 CleanText：模型校验时就去掉 NUL 等控制字符，内存实现存的也是干净的
    （和 PostgreSQL 实现一致，换实现时行为不变）。"""
    row = _row(SHOP_A, source=MetricSource.MERCHANT, recorded_by="owner\x00_a\x07")
    assert row.recorded_by == "owner_a"
    repo = InMemoryPlatformMetricsRepository()
    assert repo.upsert(row).recorded_by == "owner_a"
    assert [r.recorded_by for r in repo.list(SHOP_A)] == ["owner_a"]
    assert _row(SHOP_A).recorded_by is None  # 平台同步的没有登记人


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan, -0.1])
def test_model_rejects_non_finite_or_negative_values(bad):
    """value 只能是 0 及以上的有限数（allow_inf_nan=False，和库里的 CHECK 一致）：inf / NaN 进不了
    模型，也就进不了任何一种仓储。"""
    with pytest.raises(ValidationError):
        _row(SHOP_A, value=bad)
    assert _row(SHOP_A, value=0.0).value == 0.0


# ================================================================ PostgreSQL


def _fresh(truncate: Callable[[], None], db: Database) -> Callable[[], PlatformMetricsRepository]:
    def make() -> PlatformMetricsRepository:
        truncate()
        return PgPlatformMetricsRepository(db)

    return make


@pytest.mark.pg
def test_pg_repository_contract(db, truncate):
    repo = PgPlatformMetricsRepository(db)
    assert isinstance(repo, PlatformMetricsRepository)
    contracts.check_platform_metrics_repository(_fresh(truncate, db))


@pytest.mark.pg
def test_pg_cross_shop_isolation(db, truncate):
    contracts.check_platform_metrics_cross_shop_isolation(_fresh(truncate, db))


@pytest.mark.pg
def test_pg_rows_of_another_shop_are_invisible_even_without_where(db, admin_db, truncate):
    """行级安全（§9.2）：应用角色直接跑不带 WHERE 的 SQL 也只看得到本店的；不设店铺范围
    什么都看不到；不能往别的店铺写、也不能把行挪到别的店铺。"""
    repo = PgPlatformMetricsRepository(db)
    repo.upsert(_row(SHOP_A))
    repo.upsert(_row(SHOP_B, value=0.5))
    with db.session(SHOP_A) as conn:
        rows = conn.execute("SELECT shop_id, value FROM platform_metric").fetchall()
    assert [(r["shop_id"], r["value"]) for r in rows] == [(SHOP_A, 0.9)]
    with db.session(None) as conn:
        assert conn.execute("SELECT count(*) AS n FROM platform_metric").fetchone()["n"] == 0
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
        with db.session(SHOP_A) as conn:
            conn.execute(
                "INSERT INTO platform_metric (shop_id, metric, day, value, source, recorded_at)"
                " VALUES (%s, 'taobao_3min_response_rate', %s, 0.1, 'merchant', now())",
                (SHOP_B, DAY),
            )
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
        with db.session(SHOP_A) as conn:
            conn.execute("UPDATE platform_metric SET shop_id = %s", (SHOP_B,))
    with db.session(SHOP_A) as conn:
        assert conn.execute("DELETE FROM platform_metric").rowcount == 1
    with admin_db.session(None) as conn:
        left = conn.execute("SELECT shop_id FROM platform_metric").fetchall()
    assert [r["shop_id"] for r in left] == [SHOP_B]


@pytest.mark.pg
def test_pg_trainer_role_cannot_read_platform_metrics(db, truncate):
    PgPlatformMetricsRepository(db).upsert(_row(SHOP_A))
    for shop in (None, SHOP_A):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with db.session(shop, trainer=True) as conn:
                conn.execute("SELECT * FROM platform_metric")


@pytest.mark.pg
@pytest.mark.parametrize("bad", [math.inf, math.nan, -1.0])
def test_pg_rejects_non_finite_or_negative_values(db, truncate, bad):
    """数值约束（迁移 6）：只能是 0 及以上的有限数。模型挡住了负数、NaN 和无穷大，model_copy
    不校验时数据库兜底。"""
    repo = PgPlatformMetricsRepository(db)
    with pytest.raises(psycopg.errors.CheckViolation):
        repo.upsert(_row(SHOP_A).model_copy(update={"value": bad}))
    with pytest.raises(psycopg.errors.CheckViolation):
        repo.upsert(_row(SHOP_A).model_copy(update={"sample_size": -1}))
    assert repo.list(SHOP_A) == []


@pytest.mark.pg
def test_pg_recorded_by_control_characters_are_removed(db, truncate):
    """TEXT 不接受 NUL：登记人账号写入前去掉控制字符（应用层登记时也已清洗）。"""
    repo = PgPlatformMetricsRepository(db)
    saved = repo.upsert(_row(SHOP_A, source=MetricSource.MERCHANT, recorded_by="owner\x00_a\x07"))
    assert saved.recorded_by == "owner_a"
    assert repo.list(SHOP_A)[0].recorded_by == "owner_a"


@pytest.mark.pg
def test_pg_times_are_read_back_in_beijing_time(db, truncate):
    repo = PgPlatformMetricsRepository(db)
    repo.upsert(_row(SHOP_A))
    [row] = repo.list(SHOP_A)
    assert row.recorded_at == NOW and row.recorded_at.utcoffset() == timedelta(hours=8)


@pytest.mark.pg
def test_pg_revoke_purges_platform_metrics(db, truncate, token_key, monkeypatch):
    """取消订购（§8.4）：postgres_repos 的清理里有平台指标，只删这家店，返回条数。"""
    monkeypatch.setenv("RODIT_TOKEN_KEY", token_key)
    clock = FakeClock()
    repos = postgres_repos(db, clock=clock)
    assert isinstance(repos.platform_metrics, PgPlatformMetricsRepository)
    assert repos.platform_metrics in repos.purgers()
    assert repos.purgers()[-1] is repos.audit  # 审计最后删
    for shop in (SHOP_A, SHOP_B):
        for i in range(3):
            repos.platform_metrics.upsert(_row(shop, DAY - timedelta(days=i)))
    counts = AuthGuard(repos.auth, clock).revoke(SHOP_A, repos.purgers())
    assert counts["PgPlatformMetricsRepository"] == 3
    assert repos.platform_metrics.list(SHOP_A) == []
    assert len(repos.platform_metrics.list(SHOP_B)) == 3


# ================================================================ 迁移版本 6


def _tables(dsn: str) -> set[str]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        return {r[0] for r in rows}


@pytest.mark.pg
def test_v6_upgrades_a_v5_database_and_is_rerunnable(make_database, monkeypatch):
    """已部署到版本 5 的库：迁移只执行版本 6，新表带行级安全；脚本再执行一次不出错、数据不变。"""
    dsn = make_database()
    monkeypatch.setattr(db_module, "MIGRATIONS", tuple(m for m in MIGRATIONS if m.version <= 5))
    assert Database(dsn).migrate() == [1, 2, 3, 4, 5]
    assert "platform_metric" not in _tables(dsn)
    monkeypatch.undo()

    assert Database(dsn).migrate() == [6]
    assert Database(dsn).migrate() == []
    assert Database(dsn).applied_versions() == [1, 2, 3, 4, 5, 6]
    assert "platform_metric" in _tables(dsn)
    with psycopg.connect(dsn) as conn:
        forced = conn.execute(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class"
            " WHERE relname = 'platform_metric'"
        ).fetchone()
        policies = conn.execute(
            "SELECT policyname FROM pg_policies WHERE tablename = 'platform_metric'"
        ).fetchall()
    assert forced == (True, True)
    assert [p[0] for p in policies] == ["shop_isolation"]

    repo = PgPlatformMetricsRepository(Database(dsn))
    saved = repo.upsert(_row(SHOP_A))
    assert repo.list(SHOP_B) == []  # 行级安全照样生效
    v6 = next(m for m in MIGRATIONS if m.version == 6)
    with Database(dsn, app_role=None).session(None) as conn:
        conn.execute(v6.sql())  # 再执行一次：不报错、数据不变
    assert repo.list(SHOP_A) == [saved]


@pytest.mark.pg
def test_fresh_database_gets_every_version_including_6(make_database):
    dsn = make_database()
    assert Database(dsn).migrate() == [m.version for m in MIGRATIONS]
    assert MIGRATIONS[-1].version == 6 and "platform_metric" in _tables(dsn)
