"""§8.4 取消订购后删除店铺数据：分批、每批一个短事务，删到没有为止。

一次 DELETE 删掉一家大店铺的全部数据，会在一个事务里锁住几百万行、撑大一个事务（WAL、回滚段），
同一张表上别的店铺的写入也要等它。改成每批最多 Database.purge_batch_rows 行、每批单独提交：
- 用语句级触发器（转换表）记下每个事务删了几行：每个事务都不超过批大小，一张表分多个事务删完；
- 清理到一半断了（令牌已经删了）：已经删掉的批次留在库里，重试接着删，两次加起来正好删完；
- 别的店铺的数据一行不动。
每个测试用单独的临时库（触发器和记录表不能留在共用的库里：共用库要求每张表都是业务表）。
全部是合成数据。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import psycopg
import pytest
from psycopg import sql

from rodit.adapters.auth import AuthGuard
from rodit.app.production import postgres_repos
from rodit.core.errors import StorageUnavailable
from rodit.storage import BUSINESS_TABLES, Database, PgConversationRepository
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A, SHOP_B

pytestmark = pytest.mark.pg

BATCH = 7
ROWS_A = 40  # 大店铺（缩小版）：每张表 40 行，按 7 行一批要删 6 批
ROWS_B = 3

# 每张业务表的合成数据：%(s)s 店铺、%(n)s 行数（shop / shop_auth 每店一行）
LOADERS: dict[str, str] = {
    "shop": "INSERT INTO shop (id, platform, status) VALUES (%(s)s, 'taobao', 'active')",
    "shop_auth": (
        "INSERT INTO shop_auth (shop_id, access_token, scopes, expires_at)"
        " VALUES (%(s)s, '\\x00'::bytea, '{}', now())"
    ),
    "shop_policy": (
        "INSERT INTO shop_policy (shop_id, key, value, updated_at)"
        " SELECT %(s)s, 'k' || g, 'v', now() FROM generate_series(1, %(n)s) g"
    ),
    "item_card": (
        "INSERT INTO item_card (shop_id, item_id, card, updated_at)"
        " SELECT %(s)s, 'i_' || g, '{}', now() FROM generate_series(1, %(n)s) g"
    ),
    "knowledge_item": (
        "INSERT INTO knowledge_item (shop_id, question, answer, source, status, created_at)"
        " SELECT %(s)s, 'q' || g, '亲，有的', 'merchant', 'approved', now()"
        " FROM generate_series(1, %(n)s) g"
    ),
    "conversation": (
        "INSERT INTO conversation (shop_id, id, buyer_ref, started_at)"
        " SELECT %(s)s, 'c' || g, 'b_synthetic', now() FROM generate_series(1, %(n)s) g"
    ),
    "message": (
        "INSERT INTO message (shop_id, msg_id, conversation_id, role, text, created_at)"
        " SELECT %(s)s, 'm' || g, 'c' || (1 + g %% %(n)s), 'buyer', '在吗', now()"
        " FROM generate_series(1, %(n)s) g"
    ),
    "decision_log": (
        "INSERT INTO decision_log (shop_id, stage, provider, questions_ver, answers, created_at)"
        " SELECT %(s)s, 'route', 'fake', 'q1', '{}', now() FROM generate_series(1, %(n)s) g"
    ),
    "suggestion": (
        "INSERT INTO suggestion (shop_id, conversation_id, intent, text, kind, status, created_at)"
        " SELECT %(s)s, 'c1', 'greeting', '亲，在的', 'template', 'shown', now()"
        " FROM generate_series(1, %(n)s) g"
    ),
    "aftersale_review": (
        "INSERT INTO aftersale_review (shop_id, aftersale_id, platform_status, state)"
        " SELECT %(s)s, 'as_' || g, 'pending_review', 'new' FROM generate_series(1, %(n)s) g"
    ),
    "audit_log": (
        "INSERT INTO audit_log (shop_id, actor, action, detail, created_at)"
        " SELECT %(s)s, 'system', 'engine.result', '{}', now() FROM generate_series(1, %(n)s) g"
    ),
    "knowledge_gap": (
        "INSERT INTO knowledge_gap (shop_id, missing, source, created_at)"
        " SELECT %(s)s, 'policy:invoice', 'engine', now() FROM generate_series(1, %(n)s) g"
    ),
    "labeled_sample": (
        "INSERT INTO labeled_sample (shop_id, text, source, created_at)"
        " SELECT %(s)s, '在吗', 'manual', now() FROM generate_series(1, %(n)s) g"
    ),
    "message_recall": (
        "INSERT INTO message_recall (shop_id, conversation_id, msg_id, recalled_at, expires_at)"
        " SELECT %(s)s, 'c1', 'r' || g, now(), now() + interval '1 day'"
        " FROM generate_series(1, %(n)s) g"
    ),
    "quality_review": (
        "INSERT INTO quality_review (shop_id, period, suggestion_id, intent, status, sampled_at)"
        " SELECT %(s)s, '2026-W40', g, 'greeting', 'pending', now()"
        " FROM generate_series(1, %(n)s) g"
    ),
    "platform_metric": (
        "INSERT INTO platform_metric (shop_id, metric, day, value, source, recorded_at)"
        " SELECT %(s)s, 'taobao_3min_response_rate', DATE '2026-01-01' + g, 0.9, 'platform', now()"
        " FROM generate_series(1, %(n)s) g"
    ),
}
SINGLE_ROW = frozenset({"shop", "shop_auth"})

PROBE = """
CREATE TABLE purge_probe (tbl TEXT NOT NULL, txid BIGINT NOT NULL, n INT NOT NULL);
CREATE FUNCTION purge_probe_log() RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  INSERT INTO purge_probe SELECT TG_TABLE_NAME, txid_current(), count(*) FROM old_rows;
  RETURN NULL;
END
$$;
"""


def _col(table: str) -> str:
    return "id" if table == "shop" else "shop_id"


def _expected(table: str, n: int) -> int:
    return 1 if table in SINGLE_ROW else n


@pytest.fixture
def purge_dsn(make_database: Callable[[], str]) -> str:
    """迁移好的临时库，每张业务表都装上"每条 DELETE 语句删了几行、在哪个事务里"的记录触发器。"""
    dsn = make_database()
    Database(dsn).migrate()
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(PROBE)
        for table in BUSINESS_TABLES:
            conn.execute(
                sql.SQL(
                    "CREATE TRIGGER purge_probe AFTER DELETE ON {}"
                    " REFERENCING OLD TABLE AS old_rows"
                    " FOR EACH STATEMENT EXECUTE FUNCTION purge_probe_log()"
                ).format(sql.Identifier(table))
            )
    return dsn


def _load(dsn: str) -> None:
    assert set(BUSINESS_TABLES) <= set(LOADERS), "新的业务表要补合成数据"
    with psycopg.connect(dsn) as conn:
        for shop_id, n in ((SHOP_A, ROWS_A), (SHOP_B, ROWS_B)):
            for table in BUSINESS_TABLES:
                conn.execute(LOADERS[table], {"s": shop_id, "n": n})


def _counts(dsn: str, shop_id: str) -> dict[str, int]:
    with psycopg.connect(dsn) as conn:
        return {
            table: conn.execute(
                sql.SQL("SELECT count(*) FROM {} WHERE {} = %s").format(
                    sql.Identifier(table), sql.Identifier(_col(table))
                ),
                (shop_id,),
            ).fetchone()[0]
            for table in BUSINESS_TABLES
        }


def _batches(dsn: str) -> dict[str, list[int]]:
    """每张表：各个事务删掉的行数（只算删到了的事务）。"""
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT tbl, txid, sum(n) AS n FROM purge_probe GROUP BY tbl, txid HAVING sum(n) > 0"
        ).fetchall()
    out: dict[str, list[int]] = {}
    for table, _, n in rows:
        out.setdefault(table, []).append(int(n))
    return out


def test_revoke_purges_every_table_in_bounded_batches(purge_dsn, token_key, monkeypatch):
    monkeypatch.setenv("RODIT_TOKEN_KEY", token_key)
    _load(purge_dsn)
    clock = FakeClock()
    repos = postgres_repos(Database(purge_dsn, purge_batch_rows=BATCH), clock=clock)

    counts = AuthGuard(repos.auth, clock).revoke(SHOP_A, repos.purgers())

    assert counts["auth"] == 1
    assert counts["PgConversationRepository"] == 3 * ROWS_A  # 消息 + 会话 + 撤回墓碑
    assert counts["PgKnowledgeRepository"] == 1 + 3 * ROWS_A  # 店铺 + 政策 + 商品 + FAQ
    assert counts["PgAuditLog"] == 2 * ROWS_A
    assert _counts(purge_dsn, SHOP_A) == dict.fromkeys(BUSINESS_TABLES, 0)
    assert _counts(purge_dsn, SHOP_B) == {t: _expected(t, ROWS_B) for t in BUSINESS_TABLES}
    batches = _batches(purge_dsn)
    for table in set(BUSINESS_TABLES) - SINGLE_ROW:
        per_tx = batches[table]
        assert sum(per_tx) == ROWS_A, table
        assert max(per_tx) <= BATCH, (table, per_tx)  # 每个事务最多删一批
        assert len(per_tx) == math.ceil(ROWS_A / BATCH), (table, per_tx)


class _FailsOnSession(Database):
    """第 fail_on 个不限时事务（清理用的）开不出来：模拟清理到一半数据库连接断了。"""

    def __init__(self, dsn: str, *, fail_on: int, **kwargs: object) -> None:
        super().__init__(dsn, **kwargs)  # type: ignore[arg-type]
        self.fail_on = fail_on
        self.unbounded = 0

    @contextmanager
    def session(
        self, shop_id: str | None, *, trainer: bool = False, unbounded: bool = False
    ) -> Iterator[psycopg.Connection]:
        if unbounded:
            self.unbounded += 1
            if self.unbounded == self.fail_on:
                raise StorageUnavailable("数据库连接失败")
        with super().session(shop_id, trainer=trainer, unbounded=unbounded) as conn:
            yield conn


def test_purge_interrupted_midway_keeps_finished_batches_and_retry_finishes(purge_dsn):
    _load(purge_dsn)
    flaky = _FailsOnSession(purge_dsn, fail_on=3, purge_batch_rows=BATCH)
    repo = PgConversationRepository(flaky)

    with pytest.raises(StorageUnavailable):
        repo.purge_shop(SHOP_A)
    left = _counts(purge_dsn, SHOP_A)
    assert left["message"] == ROWS_A - 2 * BATCH  # 前两批已经提交，不随失败回滚

    removed = repo.purge_shop(SHOP_A)  # 重试（AuthGuard.revoke 失败后可以再调一次）

    assert removed == left["message"] + left["conversation"] + left["message_recall"]
    after = _counts(purge_dsn, SHOP_A)
    assert (after["message"], after["conversation"], after["message_recall"]) == (0, 0, 0)
    assert repo.purge_shop(SHOP_A) == 0  # 幂等
    assert _counts(purge_dsn, SHOP_B)["message"] == ROWS_B
