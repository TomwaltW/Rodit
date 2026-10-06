"""迁移脚本的发布纪律（storage/db.py：已发布的脚本不再修改，结构变化加新版本）。

- 已发布的迁移脚本按内容哈希锁定：改了已发布的脚本，已经迁移过的库不会再执行它
  （schema_migrations 里已有这个版本），新代码读写新列就会失败。
- 按"只执行过 v1 的库"模拟已部署的单元，升级到最新版本后仓储照常工作。
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import timedelta

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from rodit.core.enums import AfterSaleStatus, ReviewState
from rodit.core.models import AfterSaleReview
from rodit.storage import MIGRATIONS, Database, PgAfterSaleReviewRepository
from rodit.storage import db as db_module
from rodit.testing.fixtures import SHOP_A, SHOP_B, build_demo_world

RELEASED_SHA256 = {
    1: "5d8ef0f0a189c13cad0aeb782ec13c89e2424f7b00dfe7e0b62575e7069496d6",
    2: "89d1e349d11d445e6f010e84c5e3686428c877bafe3dcea09800d013f3685d1b",
    3: "254848388c403503c430f4dd79bd5df1b22feb57858060aa16c2d2ea7bdadf39",
    4: "9b8387a668d5ffe74618b2dd0f165861c2581de087385b2406935e14cd93df8a",
    5: "dda73ec62288557e43c76065fe2df968944fdd4909629198a7dfae42b4dacf09",
    6: "89a4769ee4744a242bcd3bc3c97dfe0d7f8648bf182a3da80a193675620910eb",
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


@pytest.mark.pg
def test_v3_retires_duplicate_pending_candidates_as_non_superuser_owner(pg_admin_dsn):
    """已部署的库里可能已有重复的候选知识（多实例并发写入留下的），唯一索引建不起来：
    版本 3 每组留 id 最小的一条、其余改成已下线，再建索引。生产上迁移账号是表的所有者、
    不是超级用户，受 FORCE 行级安全约束——去重要跨所有店铺做，做完 FORCE 必须恢复。"""
    suffix = uuid.uuid4().hex[:10]
    owner, dbname = f"rodit_test_owner_{suffix}", f"rodit_test_v3_{suffix}"
    with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(owner)))
        admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(dbname), sql.Identifier(owner)
            )
        )
    dsn = make_conninfo(pg_admin_dsn, dbname=dbname)
    insert = (
        "INSERT INTO knowledge_item (shop_id, question, answer, source, status, item_id, intent,"
        " created_at) VALUES (%s, %s, '亲，发货前可以改哦', 'adopted', %s, %s, %s, now())"
        " RETURNING id"
    )
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(owner)))
            for migration in MIGRATIONS[:2]:
                conn.execute(migration.sql())
            ids: dict[str, list[int]] = {}
            rows = [
                ("a_same", SHOP_A, "能改地址吗", "candidate", "i_555", "modify_order"),
                ("a_same", SHOP_A, "能改地址吗", "candidate", "i_555", "modify_order"),
                ("a_same", SHOP_A, "能改地址吗", "candidate", "i_555", "modify_order"),
                ("a_nulls", SHOP_A, "能改地址吗", "candidate", None, None),
                ("a_nulls", SHOP_A, "能改地址吗", "candidate", None, None),
                ("a_approved", SHOP_A, "能改地址吗", "approved", "i_555", "modify_order"),
                ("a_other_item", SHOP_A, "能改地址吗", "candidate", "i_777", "modify_order"),
                ("b_same", SHOP_B, "能改地址吗", "candidate", "i_555", "modify_order"),
                ("b_same", SHOP_B, "能改地址吗", "candidate", "i_555", "modify_order"),
            ]
            for group, shop, question, status, item, intent in rows:
                conn.execute("SELECT set_config('rodit.shop_id', %s, false)", (shop,))
                new_id = conn.execute(insert, (shop, question, status, item, intent)).fetchone()
                ids.setdefault(group, []).append(new_id[0])
            conn.execute("SELECT set_config('rodit.shop_id', '', false)")
            conn.execute(MIGRATIONS[2].sql())  # 版本 3，所有者身份、没有店铺范围

            forced = conn.execute(
                "SELECT relforcerowsecurity FROM pg_class WHERE relname = 'knowledge_item'"
            ).fetchone()[0]
            assert forced is True  # FORCE 已恢复：所有者自己仍受行级安全约束
            assert conn.execute("SELECT count(*) FROM knowledge_item").fetchone()[0] == 0
            conn.execute("SELECT set_config('rodit.shop_id', %s, false)", (SHOP_A,))
            with pytest.raises(psycopg.errors.UniqueViolation):
                conn.execute(insert, (SHOP_A, "能改地址吗", "candidate", "i_555", "modify_order"))
        with psycopg.connect(dsn) as admin_conn:  # 超级用户：绕过行级安全核对全部店铺
            status = dict(admin_conn.execute("SELECT id, status FROM knowledge_item").fetchall())
    finally:
        with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(dbname))
            )
            admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(owner)))

    for group in ("a_same", "a_nulls", "b_same"):
        first, *rest = ids[group]
        assert status[first] == "candidate", group
        assert [status[i] for i in rest] == ["retired"] * len(rest), group
    assert status[ids["a_approved"][0]] == "approved"
    assert status[ids["a_other_item"][0]] == "candidate"


def test_v4_upgrades_a_v3_database_with_data_and_is_rerunnable(make_database, monkeypatch):
    """迁移到版本 3 后有了数据，再迁移版本 4：旧推荐、旧标注的新列为空，复核表可用；
    版本 4 的脚本再执行一次不出错（IF NOT EXISTS / DROP POLICY IF EXISTS）。"""
    from rodit.core.enums import DraftKind, Intent, SuggestionStatus
    from rodit.core.models import LabeledSample, QualityReview, SuggestionRecord
    from rodit.storage import (
        PgFeedbackRepository,
        PgQualityReviewRepository,
        PgSuggestionRepository,
    )

    dsn = make_database()
    monkeypatch.setattr(db_module, "MIGRATIONS", tuple(m for m in MIGRATIONS if m.version <= 3))
    assert Database(dsn).migrate() == [1, 2, 3]
    monkeypatch.undo()
    now = build_demo_world().now
    with Database(dsn, app_role=None).session(None) as conn:  # 版本 3 的表结构：没有新列
        conn.execute(
            "INSERT INTO suggestion (shop_id, conversation_id, intent, text, kind, status,"
            " created_at) VALUES (%s, 'c1', 'greeting', '亲，在的', 'template', 'shown', %s)",
            (SHOP_A, now),
        )
        conn.execute(
            "INSERT INTO labeled_sample (shop_id, text, source, created_at)"
            " VALUES (%s, '在吗', 'manual', %s)",
            (SHOP_A, now),
        )
    # 升级到最新（版本 4 及之后的版本）
    assert Database(dsn).migrate() == [m.version for m in MIGRATIONS if m.version >= 4]
    db = Database(dsn)
    [old] = PgSuggestionRepository(db).list(SHOP_A)
    assert old.intent_confidence is None and old.route_rule is None
    [label] = PgFeedbackRepository(db).list_labels(SHOP_A)
    assert label.predicted_confidence is None and label.route_rule is None
    new = PgSuggestionRepository(db).add(
        SuggestionRecord(
            shop_id=SHOP_A,
            conversation_id="c1",
            intent=Intent.GREETING,
            intent_confidence=0.9,
            route_rule=6,
            text="亲，在的",
            kind=DraftKind.TEMPLATE,
            status=SuggestionStatus.SHOWN,
            created_at=now,
        )
    )
    assert (new.intent_confidence, new.route_rule) == (0.9, 6)
    PgFeedbackRepository(db).add_label(
        LabeledSample(shop_id=SHOP_A, text="在吗", route_rule=6, created_at=now)
    )
    [review] = PgQualityReviewRepository(db).add_many(
        [
            QualityReview(
                shop_id=SHOP_A,
                period="2026-W40",
                suggestion_id=new.id or 1,
                intent=Intent.GREETING,
                sampled_at=now,
            )
        ]
    )
    assert PgQualityReviewRepository(db).list(SHOP_B) == []  # 行级安全照样生效
    v4 = next(m for m in MIGRATIONS if m.version == 4)
    with Database(dsn, app_role=None).session(None) as conn:
        conn.execute(v4.sql())  # 再执行一次：不报错、数据不变
    assert PgQualityReviewRepository(db).get(SHOP_A, review.id) == review


def test_v5_upgrades_a_v4_database_with_auth_and_is_rerunnable(make_database, monkeypatch):
    """迁移到版本 4 后已经有授权和系统事件，再迁移版本 5：旧授权没有续期令牌、
    needs_reauth 为 false，照样读得出来、列得进快到期的店铺；系统事件变得可读；
    版本 5 的脚本再执行一次不出错。"""
    from pydantic import SecretStr

    from rodit.core.models import ShopAuth
    from rodit.storage import PgAuditLog, PgShopAuthRepository

    key = "rodit-test-token-key-synthetic-only"
    dsn = make_database()
    monkeypatch.setattr(db_module, "MIGRATIONS", tuple(m for m in MIGRATIONS if m.version <= 4))
    assert Database(dsn).migrate() == [1, 2, 3, 4]
    monkeypatch.undo()
    now = build_demo_world().now
    sealer = PgShopAuthRepository(Database(dsn), encryption_key=key)
    with Database(dsn, app_role=None).session(None) as conn:  # 版本 4 的表结构：没有新列
        for shop, days in ((SHOP_A, 1), (SHOP_B, 10)):
            conn.execute(
                "INSERT INTO shop_auth (shop_id, access_token, scopes, expires_at)"
                " VALUES (%s, %s, %s, %s)",
                (
                    shop,
                    sealer._encrypt(shop, f"tok-{shop}-v4"),
                    ["order.read"],
                    now + timedelta(days=days),
                ),
            )
        conn.execute(
            "INSERT INTO audit_log (shop_id, actor, action, detail, created_at)"
            " VALUES (NULL, 'system', 'llm.usage', '{}', %s)",
            (now,),
        )
    assert Database(dsn).migrate() == [m.version for m in MIGRATIONS if m.version >= 5]
    assert Database(dsn).migrate() == []
    db = Database(dsn)
    repo = PgShopAuthRepository(db, encryption_key=key)
    old = repo.get(SHOP_A)
    assert old is not None and old.access_token.get_secret_value() == f"tok-{SHOP_A}-v4"
    assert old.refresh_token is None and old.refreshed_at is None and old.needs_reauth is False
    assert repo.expiring_shop_ids(now + timedelta(days=2)) == [(SHOP_A, now + timedelta(days=1))]
    assert [e.action for e in PgAuditLog(db).list_system_events()] == ["llm.usage"]
    repo.save(
        ShopAuth(
            shop_id=SHOP_A,
            access_token=SecretStr("tok-a-v5"),
            scopes=["order.read"],
            expires_at=now + timedelta(days=3),
            refresh_token=SecretStr("tok-a-refresh-v5"),
        )
    )
    v5 = next(m for m in MIGRATIONS if m.version == 5)
    with Database(dsn, app_role=None).session(None) as conn:
        conn.execute(v5.sql())  # 再执行一次：不报错、数据不变
    after = repo.get(SHOP_A)
    assert after is not None and after.refresh_token is not None
    assert after.refresh_token.get_secret_value() == "tok-a-refresh-v5"
    assert repo.mark_needs_reauth(SHOP_B) is True
    assert [s for s, _ in repo.expiring_shop_ids(now + timedelta(days=30))] == [SHOP_A, SHOP_B]


@pytest.mark.pg
def test_v5_rerun_by_another_role_keeps_the_policy_on_the_function_owner(pg_admin_dsn):
    """库由非超级用户的属主迁移；DBA 用超级用户重新执行版本 5 的脚本：函数属主不变
    （CREATE OR REPLACE），读表的策略也要留在属主身上，函数照样列得出快到期的店铺。"""
    suffix = uuid.uuid4().hex[:10]
    owner, dbname = f"rodit_test_owner_{suffix}", f"rodit_test_rerun_{suffix}"
    with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(owner)))
        admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(dbname), sql.Identifier(owner)
            )
        )
    dsn = make_conninfo(pg_admin_dsn, dbname=dbname)
    now = build_demo_world().now
    v5 = next(m for m in MIGRATIONS if m.version == 5)

    def policy_roles_and_owner(conn: psycopg.Connection) -> tuple[list[str], str]:
        roles = conn.execute(
            "SELECT roles::text[] FROM pg_policies"
            " WHERE tablename = 'shop_auth' AND policyname = 'auth_expiring_definer'"
        ).fetchone()[0]
        fn_owner = conn.execute(
            "SELECT pg_get_userbyid(proowner) FROM pg_proc"
            " WHERE oid = 'rodit_auth_expiring(timestamptz)'::regprocedure"
        ).fetchone()[0]
        return list(roles), fn_owner

    try:
        with psycopg.connect(dsn) as conn:
            conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(owner)))
            for migration in MIGRATIONS:
                conn.execute(migration.sql())
            for shop, days in ((SHOP_A, 1), (SHOP_B, 5)):
                conn.execute("SELECT set_config('rodit.shop_id', %s, false)", (shop,))
                conn.execute(
                    "INSERT INTO shop_auth (shop_id, access_token, scopes, expires_at)"
                    " VALUES (%s, 'x'::bytea, '{order.read}', %s)",
                    (shop, now + timedelta(days=days)),
                )
            conn.execute("SELECT set_config('rodit.shop_id', '', false)")
            conn.execute("RESET ROLE")
            assert policy_roles_and_owner(conn) == ([owner], owner)
            conn.commit()
        with psycopg.connect(dsn) as conn:  # 超级用户重新执行版本 5
            conn.execute(v5.sql())
            assert policy_roles_and_owner(conn) == ([owner], owner)
            conn.commit()
        db = Database(dsn)  # 应用视角：SET LOCAL ROLE rodit_app
        with db.session(None) as conn:
            rows = conn.execute(
                "SELECT shop_id FROM rodit_auth_expiring(%s)", (now + timedelta(days=10),)
            ).fetchall()
        assert [r["shop_id"] for r in rows] == [SHOP_A, SHOP_B]
    finally:
        with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(dbname))
            )
            admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(owner)))
            admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(owner)))
