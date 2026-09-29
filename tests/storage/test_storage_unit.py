"""不需要 PostgreSQL 的存储测试：连不上时的异常类型、连接串不外泄、建表脚本的静态检查。"""

from __future__ import annotations

import re

import pytest

from rodit.core.errors import KnowledgeUnavailable
from rodit.storage import (
    BUSINESS_TABLES,
    MIGRATIONS,
    Database,
    PgKnowledgeRepository,
    PgSuggestionRepository,
    StorageUnavailable,
)

# 不存在的 unix socket 目录：立即连接失败，不访问网络
UNREACHABLE = "host=/nonexistent-rodit-socket-dir dbname=x user=u password=synthetic-pw-0001"


def test_database_repr_hides_dsn():
    db = Database(UNREACHABLE)
    assert "synthetic-pw" not in repr(db) and "nonexistent" not in repr(db)


@pytest.mark.parametrize("kwargs", [{"dsn": ""}, {"dsn": "dbname=x", "statement_timeout_ms": 0}])
def test_database_validation(kwargs):
    with pytest.raises(ValueError):
        Database(**kwargs)


def test_database_from_env(monkeypatch):
    monkeypatch.delenv("RODIT_DATABASE_DSN", raising=False)
    with pytest.raises(RuntimeError, match="RODIT_DATABASE_DSN"):
        Database.from_env()
    monkeypatch.setenv("RODIT_DATABASE_DSN", "dbname=rodit")
    assert Database.from_env(app_role=None).app_role is None


def test_unreachable_database_raises_storage_unavailable_without_dsn():
    with pytest.raises(StorageUnavailable) as ei:
        with Database(UNREACHABLE, connect_timeout_s=1).session("s_1001"):
            pass
    assert "synthetic-pw" not in str(ei.value)


def test_knowledge_outage_maps_to_knowledge_unavailable():
    db = Database(UNREACHABLE, connect_timeout_s=1)
    with pytest.raises(KnowledgeUnavailable):  # §12.2 知识库故障 → 不推荐
        PgKnowledgeRepository(db).get_shop("s_1001")
    with pytest.raises(StorageUnavailable):  # 其他仓储原样抛出
        PgSuggestionRepository(db).list("s_1001")


def test_schema_enables_forced_rls_for_every_business_table():
    script = MIGRATIONS[0].sql()
    assert [m.version for m in MIGRATIONS] == sorted({m.version for m in MIGRATIONS})
    loop_tables = set(re.findall(r"'(\w+)'", script.split("FOREACH t IN ARRAY ARRAY[")[1][:400]))
    assert loop_tables | {"shop"} == set(BUSINESS_TABLES)
    assert "ALTER TABLE shop FORCE ROW LEVEL SECURITY" in script
    for table in BUSINESS_TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {table} (" in script
    # 业务表的店铺列都是 NOT NULL（audit_log 例外：无店铺的系统事件）
    for table in set(BUSINESS_TABLES) - {"shop", "audit_log"}:
        body = script.split(f"CREATE TABLE IF NOT EXISTS {table} (")[1].split(");")[0]
        assert re.search(r"shop_id\s+TEXT (PRIMARY KEY|NOT NULL)", body), table
