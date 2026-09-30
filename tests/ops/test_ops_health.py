"""就绪检查（/readyz）：数据库可达、迁移到最新、大模型已配置（方案 §12.2）。

不连数据库的部分用假数据库；真实 PostgreSQL 的部分（应用角色读 schema_migrations 的权限）
标 @pytest.mark.pg，连不上时跳过。
"""

from __future__ import annotations

import psycopg
import pytest
from ops_support import FakeReadinessDb

from rodit.core.config import LLMConfig
from rodit.core.errors import StorageUnavailable
from rodit.ops.health import (
    CheckResult,
    ReadinessChecker,
    database_probe,
    llm_probe,
    static_probe,
)
from rodit.storage import MIGRATIONS, Database

LATEST = max(m.version for m in MIGRATIONS)
ALL = [m.version for m in MIGRATIONS]


def _check(db, **llm_env) -> dict:
    env = {
        "RODIT_LLM_BASE_URL": "https://llm.example.invalid/v1",
        "RODIT_LLM_MODEL": "m",
        "RODIT_LLM_API_KEY": "sk-synthetic",
        **llm_env,
    }
    checker = ReadinessChecker(
        {"database": database_probe(db), "llm": llm_probe(LLMConfig(), environ=env)}, cache_s=0
    )
    return checker.check().as_dict()


def test_ready_when_everything_ok():
    result = _check(FakeReadinessDb(versions=ALL))
    assert result["status"] == "ready"
    assert result["checks"]["database"]["ok"] is True
    assert result["checks"]["migrations"] == {"ok": True, "current": LATEST, "latest": LATEST}
    assert result["checks"]["llm"] == {"ok": True}


def test_newer_unknown_migrations_still_ready():
    """新版本先迁移、旧实例还在跑：旧实例照常就绪。"""
    result = _check(FakeReadinessDb(versions=[*ALL, LATEST + 1]))
    assert result["status"] == "ready"
    assert result["checks"]["migrations"]["unknown"] == [LATEST + 1]


@pytest.mark.parametrize(
    "versions, reason, pending",
    [([], "not_migrated", ALL), (ALL[:-1], "behind", ALL[-1:])],
)
def test_pending_migrations_not_ready(versions, reason, pending):
    result = _check(FakeReadinessDb(versions=versions))
    assert result["status"] == "not_ready"
    assert result["checks"]["database"]["ok"] is True
    migrations = result["checks"]["migrations"]
    assert migrations["ok"] is False and migrations["reason"] == reason
    assert migrations["pending"] == pending


@pytest.mark.parametrize(
    "db, database_reason, migrations_reason",
    [
        (
            FakeReadinessDb(enter_error=StorageUnavailable("数据库连接失败")),
            "unreachable",
            "unknown",
        ),
        (
            FakeReadinessDb(enter_error=psycopg.errors.InsufficientPrivilege("SET ROLE")),
            "no_privilege",
            "unknown",
        ),
        (
            FakeReadinessDb(query_error=psycopg.errors.QueryCanceled("statement timeout")),
            "timeout",
            "unknown",
        ),
        (FakeReadinessDb(query_error=psycopg.errors.UndefinedTable("x")), None, "not_migrated"),
        (
            FakeReadinessDb(query_error=psycopg.errors.InsufficientPrivilege("x")),
            None,
            "no_privilege",
        ),
        (FakeReadinessDb(query_error=psycopg.errors.DataError("x")), "error", "unknown"),
    ],
)
def test_database_failures(db, database_reason, migrations_reason):
    result = _check(db)
    assert result["status"] == "not_ready"
    database = result["checks"]["database"]
    if database_reason is None:
        assert database["ok"] is True
    else:
        assert database == {"ok": False, "reason": database_reason}
    assert result["checks"]["migrations"]["reason"] == migrations_reason


def test_failure_details_never_contain_error_text():
    db = FakeReadinessDb(
        enter_error=psycopg.OperationalError('connection to "db" failed: password=hunter2')
    )
    text = str(_check(db))
    assert "hunter2" not in text and "password" not in text


@pytest.mark.parametrize(
    "env, reason",
    [
        ({"RODIT_LLM_BASE_URL": ""}, "not_configured"),
        ({"RODIT_LLM_MODEL": " "}, "not_configured"),
        ({"RODIT_LLM_API_KEY": ""}, "missing_api_key"),
    ],
)
def test_llm_not_configured(env, reason):
    result = _check(FakeReadinessDb(versions=ALL), **env)
    assert result["status"] == "not_ready"
    assert result["checks"]["llm"] == {"ok": False, "reason": reason}


def test_llm_config_file_takes_precedence_and_key_value_is_never_reported():
    probe = llm_probe(
        LLMConfig(base_url="https://llm.example.invalid/v1", model="m", api_key_env="MY_KEY"),
        environ={"MY_KEY": "sk-very-secret-value"},
    )
    result = probe()["llm"]
    assert result.ok and "sk-very-secret-value" not in str(result.as_dict())


def test_half_configured_file_falls_back_to_the_environment_for_both_fields():
    """配置里只写了地址：和 build_production_unit 一样两项都取环境变量，不把配置的地址和环境变量
    的模型名拼在一起（core.config.llm_endpoint）。"""
    config = LLMConfig(base_url="https://file.example.invalid/v1", api_key_env="MY_KEY")
    only_model = {"RODIT_LLM_MODEL": "m", "MY_KEY": "k"}
    assert llm_probe(config, environ=only_model)()["llm"].as_dict() == {
        "ok": False,
        "reason": "not_configured",
    }
    both = {**only_model, "RODIT_LLM_BASE_URL": "https://env.example.invalid/v1"}
    assert llm_probe(config, environ=both)()["llm"].ok


def test_results_are_cached():
    db = FakeReadinessDb(versions=ALL)
    now = [0.0]
    checker = ReadinessChecker({"database": database_probe(db)}, cache_s=2, timer=lambda: now[0])
    checker.check()
    checker.check()
    assert db.sessions == 1
    now[0] = 2.5
    checker.check()
    assert db.sessions == 2
    checker.check(fresh=True)
    assert db.sessions == 3


def test_probe_that_raises_is_a_failed_check():
    def broken():
        raise RuntimeError("意外")

    checker = ReadinessChecker(
        {"broken": broken, "llm": static_probe("llm", CheckResult(True, {"offline": True}))},
        cache_s=0,
    )
    result = checker.check().as_dict()
    assert result["status"] == "not_ready"
    assert result["checks"]["broken"] == {"ok": False, "reason": "error"}
    assert result["checks"]["llm"] == {"ok": True, "offline": True}


def test_no_probes_is_not_ready():
    assert ReadinessChecker({}, cache_s=0).check().ready is False


# ================================================================ PostgreSQL


@pytest.mark.pg
def test_readiness_on_postgres(ops_pg_dsn, monkeypatch):
    """应用角色读 schema_migrations 要授权：只跑 Database.migrate 时是 no_privilege，
    跑 rodit.ops 的迁移命令（迁移 + 授权）后就绪。"""
    from rodit.ops.serve import migrate_database

    admin = Database(ops_pg_dsn, app_role=None)
    fresh = database_probe(admin)()
    assert fresh["database"].ok and fresh["migrations"].detail["reason"] == "not_migrated"

    assert admin.migrate() == ALL
    app_db = Database(ops_pg_dsn, statement_timeout_ms=2000, lock_timeout_ms=1000)
    no_grant = database_probe(app_db)()
    assert no_grant["database"].ok
    assert no_grant["migrations"].detail["reason"] == "no_privilege"

    monkeypatch.setenv("RODIT_MIGRATE_DSN", ops_pg_dsn)
    assert migrate_database() == []  # 已是最新，只补授权；重复执行不出错
    assert migrate_database() == []
    ready = ReadinessChecker({"database": database_probe(app_db)}, cache_s=0).check()
    assert ready.ready, ready.as_dict()
    assert ready.checks["migrations"].detail == {"current": LATEST, "latest": LATEST}


@pytest.mark.pg
def test_readiness_unreachable_database(ops_pg_admin):
    db = Database("host=127.0.0.1 port=1 dbname=nowhere connect_timeout=1")
    result = database_probe(db)()
    assert result["database"].detail == {"reason": "unreachable"}
