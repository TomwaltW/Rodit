"""行级安全（方案 §9.2）：即使 SQL 漏写 WHERE shop_id，也读不到、写不进别的店铺的数据。

先用各仓储给 A、B 两家店在每张业务表里都写数据，再用应用角色（rodit_app）直接跑
不带 WHERE 的原始 SQL，逐表核对。
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr

from rodit.core.enums import (
    AfterSaleStatus,
    DecisionStage,
    DraftKind,
    Intent,
    ReviewState,
    SuggestionStatus,
)
from rodit.core.models import (
    AfterSaleReview,
    AuditEvent,
    DecisionLogEntry,
    KnowledgeGap,
    LabeledSample,
    QualityReview,
    ShopAuth,
    SuggestionRecord,
)
from rodit.storage import (
    BUSINESS_TABLES,
    MIGRATIONS,
    Database,
    PgAfterSaleReviewRepository,
    PgAuditLog,
    PgConversationRepository,
    PgFeedbackRepository,
    PgKnowledgeRepository,
    PgQualityReviewRepository,
    PgShopAuthRepository,
    PgSuggestionRepository,
)
from rodit.testing.fakes import FakeClock, make_message
from rodit.testing.fixtures import SHOP_A, SHOP_B, build_demo_world, seed_knowledge

pytestmark = pytest.mark.pg

NOW = build_demo_world().now


def shop_column(table: str) -> str:
    return "id" if table == "shop" else "shop_id"


@pytest.fixture
def seeded(db: Database, truncate, token_key: str) -> None:
    """A、B 两家店在每张业务表里都有数据（全部通过仓储、以应用角色写入）。"""
    world = build_demo_world()
    seed_knowledge(PgKnowledgeRepository(db, clock=FakeClock()), world)
    conversations = PgConversationRepository(db)
    suggestions = PgSuggestionRepository(db)
    feedback = PgFeedbackRepository(db)
    reviews = PgAfterSaleReviewRepository(db)
    auth = PgShopAuthRepository(db, encryption_key=token_key)
    audit = PgAuditLog(db)
    quality = PgQualityReviewRepository(db)
    for shop in (SHOP_A, SHOP_B):
        conversations.add_message(
            make_message("在吗", world=world, shop_id=shop, msg_id=f"m_{shop}")
        )
        # 撤回先于消息到达：message_recall 里的墓碑（迁移版本 3）
        assert conversations.recall_message(shop, "conv_889", f"m_{shop}_later") is False
        sug = suggestions.add(
            SuggestionRecord(
                shop_id=shop,
                conversation_id="conv_889",
                intent=Intent.GREETING,
                text=f"{shop} 的推荐",
                kind=DraftKind.TEMPLATE,
                status=SuggestionStatus.SHOWN,
                created_at=NOW,
            )
        )
        # 每周人工复核样本（迁移版本 4）
        quality.add_many(
            [
                QualityReview(
                    shop_id=shop,
                    period="2026-W40",
                    suggestion_id=sug.id or 1,
                    intent=Intent.GREETING,
                    sampled_at=NOW,
                )
            ]
        )
        feedback.add_gap(KnowledgeGap(shop_id=shop, missing="policy:invoice", created_at=NOW))
        feedback.add_label(LabeledSample(shop_id=shop, text="在吗", created_at=NOW))
        reviews.save(
            AfterSaleReview(
                shop_id=shop,
                aftersale_id="as_1",
                platform_status=AfterSaleStatus.PENDING_REVIEW,
                state=ReviewState.NEW,
            ),
            expected_version=None,
        )
        auth.save(
            ShopAuth(
                shop_id=shop,
                access_token=SecretStr(f"tok-{shop}-synthetic"),
                scopes=["order.read"],
                expires_at=NOW + timedelta(days=30),
            )
        )
        audit.record(
            AuditEvent(shop_id=shop, actor="system", action="engine.result", created_at=NOW)
        )
        audit.log_decision(
            DecisionLogEntry(
                shop_id=shop,
                conversation_id="conv_889",
                stage=DecisionStage.ROUTE,
                provider="fake",
                questions_ver="q1",
                answers={},
                created_at=NOW,
            )
        )


def test_every_table_has_forced_rls_and_a_shop_policy(admin_db: Database):
    with admin_db.session(None) as conn:
        rows = conn.execute(
            "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity FROM pg_class c"
            " JOIN pg_namespace n ON n.oid = c.relnamespace"
            " WHERE n.nspname = 'public' AND c.relkind = 'r' AND c.relname <> 'schema_migrations'"
        ).fetchall()
        policies = {
            r["tablename"]
            for r in conn.execute(
                "SELECT tablename FROM pg_policies WHERE policyname = 'shop_isolation'"
            )
        }
    assert {r["relname"] for r in rows} == set(BUSINESS_TABLES)
    assert all(r["relrowsecurity"] and r["relforcerowsecurity"] for r in rows)
    assert policies == set(BUSINESS_TABLES)


def test_app_roles_cannot_bypass_rls(admin_db: Database):
    with admin_db.session(None) as conn:
        rows = conn.execute(
            "SELECT rolname, rolsuper, rolbypassrls, rolcanlogin FROM pg_roles"
            " WHERE rolname IN ('rodit_app', 'rodit_trainer') ORDER BY rolname"
        ).fetchall()
    assert [r["rolname"] for r in rows] == ["rodit_app", "rodit_trainer"]
    assert not any(r["rolsuper"] or r["rolbypassrls"] or r["rolcanlogin"] for r in rows)


def test_select_without_where_returns_only_own_shop(db, admin_db, seeded):
    for table in BUSINESS_TABLES:
        col = shop_column(table)
        with admin_db.session(None) as conn:  # 数据确实两家店都有
            everyone = {r["s"] for r in conn.execute(f"SELECT {col} AS s FROM {table}")}
        assert {SHOP_A, SHOP_B} <= everyone, table
        with db.session(SHOP_A) as conn:
            mine = [r[col] for r in conn.execute(f"SELECT * FROM {table}")]  # 故意不写 WHERE
        assert mine and set(mine) == {SHOP_A}, table


def test_no_shop_scope_sees_nothing(db, seeded):
    for table in BUSINESS_TABLES:
        with db.session(None) as conn:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        assert rows == [], table


def test_update_and_delete_without_where_only_touch_own_shop(db, admin_db, seeded):
    with db.session(SHOP_A) as conn:
        assert conn.execute("UPDATE suggestion SET text = 'A 改过'").rowcount == 1
    with admin_db.session(None) as conn:
        texts = {r["shop_id"]: r["text"] for r in conn.execute("SELECT * FROM suggestion")}
    assert texts == {SHOP_A: "A 改过", SHOP_B: f"{SHOP_B} 的推荐"}
    for table in BUSINESS_TABLES:
        col = shop_column(table)
        with db.session(SHOP_A) as conn:
            conn.execute(f"DELETE FROM {table}")
        with admin_db.session(None) as conn:
            left = {r["s"] for r in conn.execute(f"SELECT {col} AS s FROM {table}")}
        assert SHOP_A not in left and SHOP_B in left, table


def test_cannot_insert_or_move_rows_into_another_shop(db, admin_db, seeded):
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
        with db.session(SHOP_A) as conn:
            conn.execute(
                "INSERT INTO suggestion (shop_id, conversation_id, intent, text, kind, status,"
                " created_at) VALUES (%s, 'c', 'greeting', 'x', 'template', 'shown', now())",
                (SHOP_B,),
            )
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
        with db.session(SHOP_A) as conn:
            conn.execute("UPDATE knowledge_gap SET shop_id = %s", (SHOP_B,))
    with admin_db.session(None) as conn:
        counts = conn.execute(
            "SELECT shop_id, count(*) AS n FROM knowledge_gap GROUP BY shop_id ORDER BY shop_id"
        ).fetchall()
    assert [(r["shop_id"], r["n"]) for r in counts] == [(SHOP_A, 1), (SHOP_B, 1)]


def test_repositories_cannot_read_other_shop_even_by_id(db, seeded):
    # 仓储层显式按店铺查询 + 行级安全：A 用 B 的推荐 id 读不到
    b_id = PgSuggestionRepository(db).list(SHOP_B)[0].id
    assert PgSuggestionRepository(db).get(SHOP_A, b_id) is None


def test_system_audit_events_are_readable_only_without_shop_scope(db, admin_db, truncate):
    """迁移版本 5 的 system_event_select：不设店铺范围时应用能读 shop_id 为空的系统事件
    （取消订购的清理记录、评测结论），读不到任何店铺的事件；设了店铺范围时读不到系统事件。"""
    audit = PgAuditLog(db)
    audit.record(AuditEvent(shop_id=None, actor="system", action="llm.usage", created_at=NOW))
    audit.record(AuditEvent(shop_id=SHOP_A, actor="system", action="llm.usage", created_at=NOW))
    audit.record(AuditEvent(shop_id=SHOP_B, actor="system", action="llm.usage", created_at=NOW))
    with admin_db.session(None) as conn:
        shops = sorted((r["shop_id"] or "") for r in conn.execute("SELECT shop_id FROM audit_log"))
    assert shops == ["", SHOP_A, SHOP_B]
    assert [e.shop_id for e in audit.list_events(SHOP_A)] == [SHOP_A]
    assert [e.shop_id for e in audit.list_system_events()] == [None]
    with db.session(None) as conn:  # 不设店铺：只有系统事件
        rows = conn.execute("SELECT shop_id FROM audit_log").fetchall()
        assert [r["shop_id"] for r in rows] == [None]
    with db.session(SHOP_A) as conn:  # 设了店铺：只有本店的，没有系统事件
        rows = conn.execute("SELECT shop_id FROM audit_log").fetchall()
        assert [r["shop_id"] for r in rows] == [SHOP_A]
    # 读得到也改不了、删不掉（只有 SELECT 策略放行系统事件）
    with db.session(None) as conn:
        assert conn.execute("UPDATE audit_log SET actor = 'x'").rowcount == 0
        assert conn.execute("DELETE FROM audit_log").rowcount == 0
    assert len(audit.list_system_events()) == 1
    # 设置了店铺范围时不能写无店铺事件（防止借无店铺事件绕过隔离）
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with db.session(SHOP_A) as conn:
            conn.execute(
                "INSERT INTO audit_log (shop_id, actor, action, created_at)"
                " VALUES (NULL, 'system', 'x', now())"
            )


def test_trainer_role_reads_all_labels_and_nothing_else(db, seeded):
    labels = PgFeedbackRepository(db).export_training_labels()
    assert {s.shop_id for s in labels} == {SHOP_A, SHOP_B}
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with db.session(None, trainer=True) as conn:
            conn.execute("SELECT * FROM message")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with db.session(SHOP_A, trainer=True) as conn:
            conn.execute(
                "INSERT INTO labeled_sample (shop_id, text, source, created_at)"
                " VALUES (%s, 'x', 'manual', now())",
                (SHOP_A,),
            )
    # 应用角色不设店铺时读不到标注（导出只能走 trainer 角色）
    with db.session(None) as conn:
        assert conn.execute("SELECT count(*) AS n FROM labeled_sample").fetchone()["n"] == 0


def test_app_role_cannot_touch_migrations_or_truncate(db, seeded):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with db.session(SHOP_A) as conn:
            conn.execute("SELECT * FROM schema_migrations")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with db.session(SHOP_A) as conn:
            conn.execute("TRUNCATE suggestion")


def test_session_rejects_blank_shop_and_missing_trainer_role(pg_dsn):
    db = Database(pg_dsn)
    with pytest.raises(ValueError):
        with db.session(""):
            pass
    with pytest.raises(ValueError):
        with db.session("   "):
            pass
    with pytest.raises(ValueError):
        with Database(pg_dsn, trainer_role=None).session(None, trainer=True):
            pass


def test_non_superuser_owner_can_migrate_and_is_still_bound_by_rls(pg_admin_dsn):
    """生产上迁移账号是库的所有者（不是超级用户）：能执行迁移；
    FORCE 之后所有者自己也受行级安全约束。"""
    suffix = uuid.uuid4().hex[:10]
    owner, dbname = f"rodit_test_owner_{suffix}", f"rodit_test_owned_{suffix}"
    with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(owner)))
        admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(dbname), sql.Identifier(owner)
            )
        )
    try:
        with psycopg.connect(make_conninfo(pg_admin_dsn, dbname=dbname)) as conn:
            conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(owner)))
            for migration in MIGRATIONS:  # 全部版本（版本 3 新建了 message_recall）
                conn.execute(migration.sql())
            owned = conn.execute(
                "SELECT count(*) FROM pg_tables WHERE schemaname = 'public'"
                " AND tableowner = current_user"
            ).fetchone()[0]
            assert owned == len(BUSINESS_TABLES)
            conn.execute("SELECT set_config('rodit.shop_id', %s, true)", (SHOP_A,))
            conn.execute(
                "INSERT INTO knowledge_gap (shop_id, missing, source, created_at)"
                " VALUES (%s, 'policy:invoice', 'engine', now())",
                (SHOP_A,),
            )
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                with conn.transaction():
                    conn.execute(
                        "INSERT INTO knowledge_gap (shop_id, missing, source, created_at)"
                        " VALUES (%s, 'policy:invoice', 'engine', now())",
                        (SHOP_B,),
                    )
            conn.execute("SELECT set_config('rodit.shop_id', '', true)")
            assert conn.execute("SELECT count(*) FROM knowledge_gap").fetchone()[0] == 0
    finally:
        with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(dbname))
            )
            admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(owner)))


# ================================================================ 迁移版本 5：跨店列出快到期的店铺


def test_auth_expiring_function_is_locked_down(db, admin_db, truncate, token_key):
    """rodit_auth_expiring：SECURITY DEFINER、固定 search_path、属于表的属主、
    只有 rodit_app 能执行、只返回店铺 ID 和到期时间；应用角色自己仍然不能跨店读 shop_auth。"""
    with admin_db.session(None) as conn:
        fn = conn.execute(
            "SELECT p.prosecdef, p.proconfig, pg_get_userbyid(p.proowner) AS owner,"
            " (SELECT tableowner FROM pg_tables WHERE tablename = 'shop_auth') AS table_owner,"
            " has_function_privilege('rodit_app', p.oid, 'EXECUTE') AS app,"
            " has_function_privilege('rodit_trainer', p.oid, 'EXECUTE') AS trainer,"
            " p.proacl IS NOT NULL AS has_acl,"
            " EXISTS (SELECT 1 FROM aclexplode(p.proacl) a WHERE a.grantee = 0) AS public,"
            " pg_get_function_result(p.oid) AS result"
            " FROM pg_proc p WHERE p.proname = 'rodit_auth_expiring'"
        ).fetchall()
    assert len(fn) == 1
    (row,) = fn
    assert row["prosecdef"] is True
    assert "search_path=pg_catalog, pg_temp" in row["proconfig"]
    assert row["owner"] == row["table_owner"]
    assert row["app"] is True and row["trainer"] is False
    # PUBLIC 没有执行权限（proacl 为空表示默认权限，也就是授给了 PUBLIC）
    assert row["has_acl"] is True and row["public"] is False
    assert row["result"] == "TABLE(shop_id text, expires_at timestamp with time zone)"

    auth = PgShopAuthRepository(db, encryption_key=token_key)
    for shop, days in ((SHOP_A, 1), (SHOP_B, 2)):
        auth.save(
            ShopAuth(
                shop_id=shop,
                access_token=SecretStr(f"tok-{shop}-synthetic"),
                scopes=["order.read"],
                expires_at=NOW + timedelta(days=days),
                refresh_token=SecretStr(f"tok-{shop}-refresh"),
            )
        )
    with db.session(None) as conn:
        rows = conn.execute("SELECT * FROM rodit_auth_expiring(%s)", (NOW + timedelta(3),))
        got = [dict(r) for r in rows]
        assert [set(r) for r in got] == [{"shop_id", "expires_at"}] * 2
        assert [r["shop_id"] for r in got] == [SHOP_A, SHOP_B]
        # 应用角色自己打开函数的开关也读不到别的店铺（那条策略只给属主）
        conn.execute("SELECT set_config('rodit.auth_expiring_scan', 'on', true)")
        assert conn.execute("SELECT count(*) AS n FROM shop_auth").fetchone()["n"] == 0
    with db.session(SHOP_A) as conn:
        assert [r["shop_id"] for r in conn.execute("SELECT shop_id FROM shop_auth")] == [SHOP_A]
    # 训练导出角色不能调用
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with db.session(None, trainer=True) as conn:
            conn.execute("SELECT * FROM rodit_auth_expiring(now())")


def test_auth_expiring_works_for_non_superuser_owner(pg_admin_dsn):
    """生产上迁移账号（函数属主）不是超级用户、受 FORCE 行级安全约束：函数照样能跨店列出，
    属主在函数外仍然读不到别的店铺的授权。"""
    suffix = uuid.uuid4().hex[:10]
    owner, dbname = f"rodit_test_owner_{suffix}", f"rodit_test_auth_{suffix}"
    with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(owner)))
        admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(dbname), sql.Identifier(owner)
            )
        )
    dsn = make_conninfo(pg_admin_dsn, dbname=dbname)
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
                    (shop, NOW + timedelta(days=days)),
                )
            conn.execute("SELECT set_config('rodit.shop_id', '', false)")
            # 属主在函数外：FORCE 行级安全照旧
            assert conn.execute("SELECT count(*) FROM shop_auth").fetchone()[0] == 0
            conn.commit()
        db = Database(dsn)  # 应用视角：SET LOCAL ROLE rodit_app
        with db.session(None) as conn:
            rows = conn.execute(
                "SELECT shop_id FROM rodit_auth_expiring(%s)", (NOW + timedelta(days=2),)
            ).fetchall()
        assert [r["shop_id"] for r in rows] == [SHOP_A]
        with db.session(None) as conn:
            assert conn.execute("SELECT count(*) AS n FROM shop_auth").fetchone()["n"] == 0
    finally:
        with psycopg.connect(pg_admin_dsn, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(dbname))
            )
            admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(owner)))
            admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(owner)))
