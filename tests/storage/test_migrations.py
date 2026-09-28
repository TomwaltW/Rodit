"""迁移脚本的发布纪律（storage/db.py：已发布的脚本不再修改，结构变化加新版本）。

- 已发布的迁移脚本按内容哈希锁定：改了已发布的脚本，已经迁移过的库不会再执行它
  （schema_migrations 里已有这个版本），新代码读写新列就会失败。
- 按"只执行过 v1 的库"模拟已部署的单元，升级到最新版本后仓储照常工作。
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

import psycopg
import pytest

from rodit.core.enums import AfterSaleStatus, ReviewState
from rodit.core.models import AfterSaleReview
from rodit.storage import MIGRATIONS, Database, PgAfterSaleReviewRepository
from rodit.storage import db as db_module
from rodit.testing.fixtures import SHOP_A, build_demo_world

RELEASED_SHA256 = {
    1: "5d8ef0f0a189c13cad0aeb782ec13c89e2424f7b00dfe7e0b62575e7069496d6",
    2: "89d1e349d11d445e6f010e84c5e3686428c877bafe3dcea09800d013f3685d1b",
}
"""已发布迁移脚本的 SHA-256。新增版本时在这里补一行；已有的行不能改。"""


def test_released_migration_scripts_are_unchanged():
    by_version = {m.version: m for m in MIGRATIONS}
    for version, digest in RELEASED_SHA256.items():
        script = by_version[version].sql().encode("utf-8")
        assert hashlib.sha256(script).hexdigest() == digest, (
            f"迁移 {version} 已发布，不能修改；结构变化请加新版本"
        )


def test_migration_versions_are_unique_and_ascending():
    versions = [m.version for m in MIGRATIONS]
    assert versions == sorted(set(versions)) and versions[0] == 1


def _columns(dsn: str, table: str) -> set[str]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (table,),
        ).fetchall()
    return {r[0] for r in rows}


@pytest.mark.pg
def test_database_on_released_v1_upgrades_to_latest(make_database, monkeypatch):
    dsn = make_database()
    # 模拟 v2 发布之前就已部署的库：只执行 v1
    monkeypatch.setattr(db_module, "MIGRATIONS", MIGRATIONS[:1])
    assert Database(dsn).migrate() == [1]
    assert "platform_status_version" not in _columns(dsn, "aftersale_review")
    monkeypatch.undo()

    later = [m.version for m in MIGRATIONS if m.version > 1]
    assert 2 in later
    assert Database(dsn).migrate() == later
    assert Database(dsn).migrate() == []
    assert Database(dsn).applied_versions() == [m.version for m in MIGRATIONS]
    assert "platform_status_version" in _columns(dsn, "aftersale_review")

    repo = PgAfterSaleReviewRepository(Database(dsn))
    now = build_demo_world().now
    saved = repo.save(
        AfterSaleReview(
            shop_id=SHOP_A,
            aftersale_id="as_2002",
            platform_status=AfterSaleStatus.PENDING_REVIEW,
            platform_status_version="v3",
            state=ReviewState.NEW,
            updated_at=now,
        ),
        expected_version=None,
    )
    assert saved.version == 1
    got = repo.get(SHOP_A, "as_2002")
    assert got is not None and got.platform_status_version == "v3"
    repo.save(
        got.model_copy(update={"platform_status_version": "v4", "updated_at": now + timedelta(1)}),
        expected_version=1,
    )
    assert repo.get(SHOP_A, "as_2002").platform_status_version == "v4"
