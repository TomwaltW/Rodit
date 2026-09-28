"""PostgreSQL 仓储在契约测试之外的行为：迁移、并发与乐观锁、令牌加密、类型往返、
与内存实现的一致性、知识库故障降级、审计与大模型用量。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import SecretStr

from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.config import EngineConfig, effective_shop_config
from rodit.core.context import shop_scope
from rodit.core.enums import (
    AfterSaleStatus,
    DecisionStage,
    DraftKind,
    Intent,
    KnowledgeSource,
    KnowledgeStatus,
    Platform,
    ReviewState,
    Role,
    SuggestionStatus,
)
from rodit.core.errors import ConcurrencyConflict, KnowledgeUnavailable
from rodit.core.models import (
    AfterSaleReview,
    AuditEvent,
    DecisionLogEntry,
    KnowledgeItem,
    Shop,
    ShopAuth,
    SuggestionRecord,
)
from rodit.core.ports import LLMMessage, LLMRequest
from rodit.inmemory import (
    HashingEmbedder,
    InMemoryConversationRepository,
    InMemoryKnowledgeRepository,
)
from rodit.llm import MeteredLLMClient
from rodit.storage import (
    Database,
    PgAfterSaleReviewRepository,
    PgAuditLog,
    PgConversationRepository,
    PgKnowledgeRepository,
    PgShopAuthRepository,
    PgSuggestionRepository,
    TokenDecryptionError,
)
from rodit.storage.db import MIGRATIONS
from rodit.testing.fakes import FakeClock, FakeLLMClient, make_message
from rodit.testing.fixtures import SHOP_A, SHOP_B, build_demo_world, seed_knowledge

pytestmark = pytest.mark.pg

NOW = build_demo_world().now


def run_concurrently(n: int, fn: Callable[[], Any]) -> list[Any]:
    """n 个线程同时（barrier 对齐）执行 fn，返回每个线程的结果或异常。"""
    barrier = threading.Barrier(n)
    results: list[Any] = [None] * n

    def worker(i: int) -> None:
        barrier.wait()
        try:
            results[i] = fn()
        except Exception as e:  # 收集起来由测试断言
            results[i] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return results


def wait_for_lock_waiter(admin_db: Database, timeout_s: float = 10.0) -> None:
    """等到有一个会话在等行锁（用 pg_stat_activity 判断，不靠固定 sleep）。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with admin_db.session(None) as conn:
            n = conn.execute(
                "SELECT count(*) AS n FROM pg_stat_activity"
                " WHERE datname = current_database() AND wait_event_type = 'Lock'"
            ).fetchone()["n"]
        if n:
            return
        time.sleep(0.02)
    raise AssertionError("没有会话在等锁")


def review(aftersale_id: str = "as_2002", **kw: Any) -> AfterSaleReview:
    return AfterSaleReview(
        shop_id=kw.pop("shop_id", SHOP_A),
        aftersale_id=aftersale_id,
        platform_status=AfterSaleStatus.PENDING_REVIEW,
        state=kw.pop("state", ReviewState.NEW),
        **kw,
    )


# ================================================================ 迁移


def test_migrate_is_idempotent_and_keeps_data(db, admin_db, truncate):
    PgKnowledgeRepository(db).upsert_shop(Shop(id=SHOP_A, platform=Platform.TAOBAO))
    assert db.migrate() == []
    assert db.applied_versions() == [m.version for m in MIGRATIONS]
    with admin_db.session(None) as conn:  # 脚本本身也能重复执行
        for m in MIGRATIONS:
            conn.execute(m.sql())
        rows = conn.execute("SELECT version, name FROM schema_migrations").fetchall()
    assert [(r["version"], r["name"]) for r in rows] == [(m.version, m.name) for m in MIGRATIONS]
    assert PgKnowledgeRepository(db).get_shop(SHOP_A) is not None


def test_concurrent_migrate_runs_once(make_database):
    dsn = make_database()
    assert Database(dsn).applied_versions() == []
    results = run_concurrently(4, lambda: Database(dsn).migrate())
    every = [m.version for m in MIGRATIONS]
    assert sorted(results, key=len) == [[], [], [], every]
    assert Database(dsn).applied_versions() == every


# ================================================================ 乐观锁（§10 version）


def test_stale_writer_conflicts(db, truncate):
    repo = PgAfterSaleReviewRepository(db)
    repo.save(review(), expected_version=None)
    seen_by_agent_1 = repo.get(SHOP_A, "as_2002")
    seen_by_agent_2 = repo.get(SHOP_A, "as_2002")
    repo.save(seen_by_agent_1.model_copy(update={"state": ReviewState.SUGGEST}), expected_version=1)
    with pytest.raises(ConcurrencyConflict, match="实际 2"):
        repo.save(
            seen_by_agent_2.model_copy(update={"state": ReviewState.PLATFORM}),
            expected_version=seen_by_agent_2.version,
        )
    assert repo.get(SHOP_A, "as_2002").state == ReviewState.SUGGEST


def test_update_waits_for_open_transaction_then_conflicts(db, admin_db, truncate):
    repo = PgAfterSaleReviewRepository(db)
    first = repo.save(review(), expected_version=None)
    outcome: dict[str, Any] = {}

    def second_writer() -> None:
        try:
            outcome["result"] = repo.save(first, expected_version=1)
        except ConcurrencyConflict as e:
            outcome["result"] = e

    with db.session(SHOP_A) as holder:  # 会话 1：改了还没提交，持有行锁
        holder.execute(
            "UPDATE aftersale_review SET state = 'suggest', version = version + 1"
            " WHERE shop_id = %s AND aftersale_id = %s AND version = 1",
            (SHOP_A, "as_2002"),
        )
        t = threading.Thread(target=second_writer)
        t.start()
        wait_for_lock_waiter(admin_db)  # 会话 2 在等锁
    t.join(10)  # 会话 1 提交后，会话 2 重新检查条件：version 已是 2
    assert isinstance(outcome["result"], ConcurrencyConflict)
    got = repo.get(SHOP_A, "as_2002")
    assert (got.version, got.state) == (2, ReviewState.SUGGEST)


def test_concurrent_updates_have_exactly_one_winner(db, truncate):
    repo = PgAfterSaleReviewRepository(db)
    base = repo.save(review(), expected_version=None)
    results = run_concurrently(
        6,
        lambda: repo.save(
            base.model_copy(update={"state": ReviewState.SUGGEST}), expected_version=1
        ),
    )
    winners = [r for r in results if isinstance(r, AfterSaleReview)]
    assert len(winners) == 1 and winners[0].version == 2
    assert all(isinstance(r, ConcurrencyConflict) for r in results if r is not winners[0])
    assert repo.get(SHOP_A, "as_2002").version == 2


def test_concurrent_creates_have_exactly_one_winner(db, truncate):
    repo = PgAfterSaleReviewRepository(db)
    results = run_concurrently(6, lambda: repo.save(review("as_new"), expected_version=None))
    assert sum(isinstance(r, AfterSaleReview) for r in results) == 1
    assert sum(isinstance(r, ConcurrencyConflict) for r in results) == 5


def test_review_list_order_is_code_point_and_rules_are_validated(db, truncate):
    repo = PgAfterSaleReviewRepository(db)
    ids = ["as_10", "as-2", "AS_3", "as_9", "as_1"]
    for i in ids:
        repo.save(review(i), expected_version=None)
    assert [r.aftersale_id for r in repo.list_by_state(SHOP_A)] == sorted(ids)
    assert repo.list_by_state(SHOP_A, []) == []
    with pytest.raises(ValueError, match="规则名"):
        repo.save(review("as_x", matched_rules=["R1,R2"]), expected_version=None)
    assert repo.get(SHOP_A, "as_x") is None


def test_review_json_and_datetimes_round_trip(db, truncate):
    repo = PgAfterSaleReviewRepository(db)
    utc_time = datetime(2026, 9, 28, 2, 35, tzinfo=UTC)  # 等于东八区 10:35
    saved = repo.save(
        review(
            matched_rules=["R4"],
            verify_started_at=utc_time,
            action_result={"ok": True, "amount": Decimal("259.00"), "at": NOW, "codes": [1, 2]},
            updated_at=NOW,
        ),
        expected_version=None,
    )
    got = repo.get(SHOP_A, "as_2002")
    assert got == saved
    assert got.verify_started_at == utc_time and got.verify_started_at.utcoffset() == timedelta(
        hours=8
    )
    # JSONB 里的 Decimal / datetime 存成字符串（pydantic JSON 模式）
    assert got.action_result == {
        "ok": True,
        "amount": "259.00",
        "at": "2026-09-28T10:35:00+08:00",
        "codes": [1, 2],
    }
    assert got.matched_rules == ["R4"] and got.matched_rule == "R4"


# ================================================================ 令牌加密（§8.4）


def auth(token: str = "tok-synthetic-secret-123", shop_id: str = SHOP_A) -> ShopAuth:
    return ShopAuth(
        shop_id=shop_id,
        access_token=SecretStr(token),
        scopes=["order.read"],
        expires_at=NOW + timedelta(days=30),
    )


def test_token_is_encrypted_at_rest(db, admin_db, truncate, token_key):
    repo = PgShopAuthRepository(db, encryption_key=token_key)
    repo.save(auth())
    with admin_db.session(None) as conn:
        first = bytes(conn.execute("SELECT access_token FROM shop_auth").fetchone()["access_token"])
    repo.save(auth())  # 同一个令牌再存一次：随机 nonce，密文不同
    with admin_db.session(None) as conn:
        second = bytes(
            conn.execute("SELECT access_token FROM shop_auth").fetchone()["access_token"]
        )
    assert b"tok-synthetic-secret-123" not in first and b"tok-synthetic" not in second
    assert first != second
    # 应用里加密（AES-256-GCM），库里只有密文；密钥从不发给数据库（见 test_pg_token_crypto.py）
    assert first.startswith(b"RDT1") and second.startswith(b"RDT1")
    assert repo.get(SHOP_A).access_token.get_secret_value() == "tok-synthetic-secret-123"
    assert token_key not in repr(repo)


def test_wrong_key_fails_without_leaking(db, truncate, token_key):
    PgShopAuthRepository(db, encryption_key=token_key).save(auth())
    other_key = "another-synthetic-key-0000"
    with pytest.raises(TokenDecryptionError) as ei:
        PgShopAuthRepository(db, encryption_key=other_key).get(SHOP_A)
    text = str(ei.value) + repr(ei.value) + str(ei.value.__cause__)
    assert other_key not in text and token_key not in text and "tok-synthetic" not in text


def test_key_validation_and_from_env(db, monkeypatch, token_key):
    with pytest.raises(ValueError):
        PgShopAuthRepository(db, encryption_key="short")
    monkeypatch.delenv("RODIT_TOKEN_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RODIT_TOKEN_KEY"):
        PgShopAuthRepository.from_env(db)
    monkeypatch.setenv("RODIT_TOKEN_KEY", token_key)
    assert isinstance(PgShopAuthRepository.from_env(db), PgShopAuthRepository)


def test_auth_purge_counts(db, truncate, token_key):
    repo = PgShopAuthRepository(db, encryption_key=token_key)
    repo.save(auth(shop_id=SHOP_B))
    assert repo.purge_shop(SHOP_A) == 0
    assert repo.purge_shop(SHOP_B) == 1 and repo.get(SHOP_B) is None


# ================================================================ 知识


def test_shop_config_overrides_round_trip(db, truncate):
    repo = PgKnowledgeRepository(db)
    shop = Shop(
        id=SHOP_A,
        platform=Platform.TAOBAO,
        category="女装",
        config_overrides={
            "aftersale_amount_limit": Decimal("88.80"),
            "disabled_intents": [Intent.COMPLAINT],
            "banned_words": ["全网最低"],
            "max_answer_chars": 90,
        },
    )
    repo.upsert_shop(shop)
    got = repo.get_shop(SHOP_A)
    assert got.config_overrides == {
        "aftersale_amount_limit": "88.80",
        "disabled_intents": ["complaint"],
        "banned_words": ["全网最低"],
        "max_answer_chars": 90,
    }
    cfg = effective_shop_config(EngineConfig(), got)
    assert cfg.aftersale_amount_limit == Decimal("88.80")
    assert cfg.disabled_intents == [Intent.COMPLAINT] and cfg.max_answer_chars == 90
    repo.upsert_shop(got.model_copy(update={"config_overrides": {}}))
    assert repo.get_shop(SHOP_A).config_overrides == {}


def test_knowledge_matches_inmemory_implementation(db, truncate):
    """同一份演示数据写进两种实现：列表、检索、purge 的结果一致（id 都从 1 开始）。"""
    world = build_demo_world()
    pg = PgKnowledgeRepository(db, clock=FakeClock())
    mem = InMemoryKnowledgeRepository(FakeClock())
    seed_knowledge(pg, world)
    seed_knowledge(mem, world)
    embedder = HashingEmbedder(dim=64)
    for item in mem.list_knowledge(SHOP_A, status=None):
        vec = embedder.embed_one(item.question)
        pg.set_knowledge_embedding(SHOP_A, item.id, vec)
        mem.set_knowledge_embedding(SHOP_A, item.id, vec)
    for kwargs in (
        {},
        {"status": None},
        {"item_id": "i_555"},
        {"item_id": "i_777", "status": KnowledgeStatus.CANDIDATE},
    ):
        strip = [
            k.model_copy(update={"embedding": None}) for k in pg.list_knowledge(SHOP_A, **kwargs)
        ]
        want = [
            k.model_copy(update={"embedding": None}) for k in mem.list_knowledge(SHOP_A, **kwargs)
        ]
        assert strip == want, kwargs
    query = embedder.embed_one("发啥快递呀")
    for item_id in (None, "i_555"):
        got = pg.search_knowledge(SHOP_A, query, top_k=3, item_id=item_id)
        want = mem.search_knowledge(SHOP_A, query, top_k=3, item_id=item_id)
        assert [k.id for k, _ in got] == [k.id for k, _ in want]
        assert [s for _, s in got] == pytest.approx([s for _, s in want], abs=1e-6)
    assert pg.list_item_cards(SHOP_A) == mem.list_item_cards(SHOP_A)
    assert pg.get_policies(SHOP_A) == mem.get_policies(SHOP_A)
    assert pg.purge_shop(SHOP_B) == mem.purge_shop(SHOP_B) == 6  # 店铺 1 + 政策 3 + 商品 1 + FAQ 1


def test_search_ties_top_k_and_dimension_mismatch(db, truncate):
    repo = PgKnowledgeRepository(db, clock=FakeClock())

    def add(q: str, item_id: str | None = None) -> int:
        saved = repo.add_knowledge(
            KnowledgeItem(
                shop_id=SHOP_A,
                question=q,
                answer="答",
                source=KnowledgeSource.MERCHANT,
                status=KnowledgeStatus.APPROVED,
                item_id=item_id,
                embedding=[0.6, 0.8, 0.0],
            )
        )
        return saved.id

    a, b, c = add("甲"), add("乙"), add("丙", item_id="i_x")
    hits = repo.search_knowledge(SHOP_A, [0.6, 0.8, 0.0])
    assert [k.id for k, _ in hits] == [a, b]  # 同分按 id；商品专属的 c 不在通用范围
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)
    assert [k.id for k, _ in repo.search_knowledge(SHOP_A, [0.6, 0.8, 0.0], item_id="i_x")] == [
        a,
        b,
        c,
    ]
    assert repo.search_knowledge(SHOP_A, [0.6, 0.8, 0.0], top_k=0) == []
    assert repo.get_knowledge(SHOP_A, a).embedding == pytest.approx([0.6, 0.8, 0.0])
    with pytest.raises(ValueError, match="维度"):  # 与内存实现一致：维度不一致直接报错
        repo.search_knowledge(SHOP_A, [1.0, 0.0])


def test_created_at_filled_by_clock_and_returned_in_cst(db, truncate):
    clock = FakeClock(datetime(2026, 9, 28, 1, 0, tzinfo=UTC))
    repo = PgKnowledgeRepository(db, clock=clock)
    saved = repo.add_knowledge(
        KnowledgeItem(
            shop_id=SHOP_A,
            question="q",
            answer="a",
            source=KnowledgeSource.MERCHANT,
            status=KnowledgeStatus.CANDIDATE,
        )
    )
    assert saved.created_at == clock.now() and saved.created_at.tzinfo == CST


def test_statement_timeout_becomes_knowledge_unavailable(pg_dsn, admin_db, truncate):
    repo = PgKnowledgeRepository(Database(pg_dsn, statement_timeout_ms=200))
    repo.upsert_shop(Shop(id=SHOP_A, platform=Platform.TAOBAO))
    with admin_db.session(None) as holder:  # 另一个事务锁住这一行
        holder.execute("SELECT id FROM shop WHERE id = %s FOR UPDATE", (SHOP_A,))
        with pytest.raises(KnowledgeUnavailable):
            repo.upsert_shop(Shop(id=SHOP_A, platform=Platform.TAOBAO, category="女装"))
    assert repo.get_shop(SHOP_A).category is None  # 超时的写入已回滚


# ================================================================ 会话


def test_duplicate_message_does_not_create_other_conversation(db, truncate):
    repo = PgConversationRepository(db)
    world = build_demo_world()
    assert repo.add_message(make_message("在吗", world=world, msg_id="m1", conversation_id="c1"))
    assert not repo.add_message(
        make_message("在吗", world=world, msg_id="m1", conversation_id="c2")
    )
    assert repo.get_conversation(SHOP_A, "c2") is None
    assert not repo.add_outgoing(
        shop_id=SHOP_A, conversation_id="c_nope", msg_id="m2", role=Role.AGENT, text="x", at=NOW
    )
    assert repo.recent_messages(SHOP_A, "c_nope") == []


def test_same_timestamp_keeps_write_order_and_times_are_cst(db, truncate):
    repo = PgConversationRepository(db)
    world = build_demo_world()
    at_utc = NOW.astimezone(UTC)
    for i in range(4):
        repo.add_message(make_message(f"第{i}条", world=world, msg_id=f"m{i}", at=at_utc))
    msgs = repo.recent_messages(SHOP_A, "conv_889", limit=3)
    assert [m.msg_id for m in msgs] == ["m1", "m2", "m3"]
    assert all(m.created_at == NOW and m.created_at.tzinfo == CST for m in msgs)
    assert repo.recent_messages(SHOP_A, "conv_889", limit=0) == []


def test_concurrent_duplicate_pushes_store_once(db, truncate):
    repo = PgConversationRepository(db)
    msg = make_message("在吗", world=build_demo_world(), msg_id="m_dup")
    results = run_concurrently(6, lambda: repo.add_message(msg))
    assert sorted(results) == [False] * 5 + [True]
    assert [m.msg_id for m in repo.recent_messages(SHOP_A, msg.conversation_id)] == ["m_dup"]


def test_conversation_purge_matches_inmemory(db, truncate):
    world = build_demo_world()
    pg, mem = PgConversationRepository(db), InMemoryConversationRepository()
    for repo in (pg, mem):
        for i in range(3):
            repo.add_message(make_message("x", world=world, shop_id=SHOP_B, msg_id=f"m{i}"))
        repo.add_message(make_message("x", world=world, shop_id=SHOP_B, conversation_id="c2"))
    assert pg.purge_shop(SHOP_B) == mem.purge_shop(SHOP_B) == 6  # 2 个会话 + 4 条消息


# ================================================================ 推荐


def suggestion(**kw: Any) -> SuggestionRecord:
    return SuggestionRecord(
        shop_id=SHOP_A,
        conversation_id="c1",
        intent=Intent.SHIPPING_POLICY,
        text="亲，默认发中通快递哦~",
        kind=DraftKind.FAQ,
        status=SuggestionStatus.SHOWN,
        created_at=NOW,
        **kw,
    )


def test_suggestion_update_errors_and_optional_fields(db, truncate):
    repo = PgSuggestionRepository(db)
    saved = repo.add(suggestion())
    assert saved.qa_scores is None and saved.evidence_ids == [] and saved.msg_id is None
    with pytest.raises(KeyError):
        repo.update(suggestion())  # 没有 id
    with pytest.raises(KeyError):
        repo.update(saved.model_copy(update={"id": saved.id + 100}))
    expired = repo.update(saved.model_copy(update={"status": SuggestionStatus.EXPIRED}))
    assert expired == repo.get(SHOP_A, saved.id) and expired.status == SuggestionStatus.EXPIRED
    assert [r.id for r in repo.list(SHOP_A, since=NOW)] == [saved.id]  # since 含边界
    assert repo.list(SHOP_A, since=NOW + timedelta(microseconds=1)) == []


# ================================================================ 审计


def test_audit_log_filters_order_and_purge(db, truncate):
    audit = PgAuditLog(db)
    for i, action in enumerate(["engine.route", "engine.result", "engine.route"]):
        audit.record(
            AuditEvent(
                shop_id=SHOP_A,
                actor="system",
                action=action,
                detail={"i": i, "amount": Decimal("1.50"), "at": NOW},
                created_at=NOW - timedelta(minutes=3 - i),
            )
        )
    audit.record(AuditEvent(shop_id=SHOP_B, actor="kf_02", action="engine.route", created_at=NOW))
    audit.log_decision(
        DecisionLogEntry(
            shop_id=SHOP_A,
            conversation_id="c1",
            stage=DecisionStage.ROUTE,
            provider="incloud_llm",
            questions_ver="route-v1",
            answers={"intent": {"type": "choice", "choice": "shipping_policy", "confidence": 0.9}},
            route_result="answer",
            latency_ms=320,
            created_at=NOW,
        )
    )
    audit.log_decision(
        DecisionLogEntry(
            shop_id=SHOP_A,
            stage=DecisionStage.QA,
            provider="incloud_llm",
            questions_ver="qa-v1",
            answers={},
            created_at=NOW - timedelta(minutes=1),
        )
    )
    events = audit.list_events(SHOP_A)
    assert [e.detail["i"] for e in events] == [0, 1, 2]
    assert events[0].detail["amount"] == "1.50" and events[0].detail["at"] == NOW.isoformat()
    assert [e.detail["i"] for e in audit.list_events(SHOP_A, action="engine.route")] == [0, 2]
    assert [e.detail["i"] for e in audit.list_events(SHOP_A, since=NOW - timedelta(minutes=2))] == [
        1,
        2,
    ]
    assert [d.stage for d in audit.list_decisions(SHOP_A)] == [
        DecisionStage.QA,
        DecisionStage.ROUTE,
    ]
    [route] = audit.list_decisions(SHOP_A, stage=DecisionStage.ROUTE)
    assert route.latency_ms == 320 and route.answers["intent"]["choice"] == "shipping_policy"
    assert audit.list_decisions(SHOP_A, since=NOW + timedelta(seconds=1)) == []
    assert [e.actor for e in audit.list_events(SHOP_B)] == ["kf_02"]
    assert audit.purge_shop(SHOP_A) == 5
    assert audit.list_events(SHOP_A) == [] and len(audit.list_events(SHOP_B)) == 1


def test_llm_usage_is_metered_per_shop(db, truncate):
    audit = PgAuditLog(db)
    clock = FakeClock()
    metered = MeteredLLMClient(FakeLLMClient(lambda r: "亲，在的~"), audit, clock)
    request = LLMRequest(purpose="generate", messages=[LLMMessage(role="user", content="在吗")])
    with shop_scope(SHOP_A):
        metered.complete(request)
        metered.complete(request)
    metered.complete(request.model_copy(update={"shop_id": SHOP_B}))
    usage_a = audit.list_events(SHOP_A, action=AuditAction.LLM_USAGE)
    assert len(usage_a) == 2 and all(e.created_at == clock.now() for e in usage_a)
    assert sum(e.detail["input_tokens"] for e in usage_a) == 4  # FakeLLMClient 按字符计
    assert len(audit.list_events(SHOP_B, action=AuditAction.LLM_USAGE)) == 1
