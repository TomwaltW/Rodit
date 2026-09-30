"""生产组装跑在 PostgreSQL 上：撤回、影子模式、取消订购（§5.3、§8.3、§13.4、§8.4）。

和内存实现同样的行为要在 PostgreSQL 上成立：
- 撤回刚收到的买家消息：推荐过期、侧边栏清空；
- 撤回先于那条消息到达：消息到了丢弃，库里只有占住 msg_id 的一行（没有文字）；
- 影子模式下迟到的旧消息不让同一轮多一套影子记录；
- 取消订购：按小批分多个事务删，删完这家店铺在所有业务表里的数据。
大模型用离线替身，平台接口是模拟实现，全部是合成数据。
"""

from __future__ import annotations

import pytest
from psycopg import sql

from rodit.adapters import StandaloneAssistAdapter, TaobaoMockAdapter
from rodit.app import AssistService, build_unit
from rodit.app.demo import SUPPORTED_ACTIONS, seed_unit, taobao_event
from rodit.app.offline import EvidenceEchoLLM, HeuristicQAJudge
from rodit.app.production import postgres_repos
from rodit.core.config import EngineConfig
from rodit.core.enums import Platform, SuggestionStatus
from rodit.core.pii import BuyerRefHasher
from rodit.decision import RuleProvider
from rodit.storage import BUSINESS_TABLES, Database
from rodit.testing.fakes import FakeActionPort, FakeClock, InMemoryToolGateway
from rodit.testing.fixtures import SHOP_A, TEST_SALT, build_demo_world

pytestmark = pytest.mark.pg

TOKEN_KEY = "rodit-e2e-token-key-synthetic-only"  # 只用于测试的合成密钥
CONV = "c_pg_recall"


def _admin(dsn: str) -> Database:
    return Database(dsn, app_role=None)


@pytest.fixture
def pg_service(e2e_pg_dsn: str, monkeypatch: pytest.MonkeyPatch) -> AssistService:
    with _admin(e2e_pg_dsn).session(None) as conn:
        tables = sql.SQL(", ").join(sql.Identifier(t) for t in BUSINESS_TABLES)
        conn.execute(sql.SQL("TRUNCATE {} RESTART IDENTITY").format(tables))
    monkeypatch.setenv("RODIT_TOKEN_KEY", TOKEN_KEY)
    world = build_demo_world()
    clock = FakeClock(world.now)
    hasher = BuyerRefHasher(TEST_SALT)
    tools = InMemoryToolGateway(world, Platform.TAOBAO)
    unit = build_unit(
        platform=Platform.TAOBAO,
        config=EngineConfig(),
        clock=clock,
        hasher=hasher,
        # 清理按 3 行一批：取消订购要分好几个事务删完
        repos=postgres_repos(Database(e2e_pg_dsn, purge_batch_rows=3), clock=clock),
        llm=EvidenceEchoLLM(),
        tools=tools,
        adapters=[
            TaobaoMockAdapter(tools, clock, hasher, access="B"),
            StandaloneAssistAdapter(Platform.TAOBAO, clock, hasher),
        ],
        action_port=FakeActionPort(
            Platform.TAOBAO,
            supported=SUPPORTED_ACTIONS[Platform.TAOBAO],
            gateway=tools,
            clock=clock,
        ),
        route_provider=RuleProvider(),
        qa_provider=HeuristicQAJudge(),
    )
    seed_unit(unit, world)
    return AssistService(unit)


def _ms(service: AssistService, offset_s: float = 0) -> int:
    return int((service.unit.clock.now().timestamp() + offset_s) * 1000)


def _event(service: AssistService, text: str, msg_id: str, offset_s: float = 0) -> dict:
    return taobao_event(text, msg_id=msg_id, session_id=CONV, send_time_ms=_ms(service, offset_s))


def _recall(msg_id: str, recalled: str) -> dict:
    return {
        "seller_id": SHOP_A,
        "session_id": CONV,
        "msg_id": msg_id,
        "msg_type": "recall",
        "recall_msg_id": recalled,
    }


def test_recall_expires_suggestions_and_clears_sidebar_on_postgres(pg_service):
    service = pg_service
    adapter = service.unit.adapters.get(Platform.TAOBAO, "B")
    result = service.handle_event("B", _event(service, "什么时候发货", "q1"))
    assert result is not None and result.suggestions
    assert adapter.sidebar(SHOP_A, CONV) == result.suggestions

    assert service.handle_event("B", _recall("q1_recall", "q1")) is None

    assert adapter.sidebar(SHOP_A, CONV) == []
    records = service.unit.repos.suggestions.list(SHOP_A, conversation_id=CONV)
    assert records and {r.status for r in records} == {SuggestionStatus.EXPIRED}


def test_message_recalled_before_it_arrives_is_dropped_on_postgres(pg_service, e2e_pg_dsn):
    service = pg_service
    assert service.handle_event("B", _recall("e1_recall", "e1")) is None
    early = _event(service, "我的手机号13812345678，这件有M码吗", "e1", 2)
    assert service.handle_event("B", early) is None
    assert service.handle_event("B", early) is None  # 平台重推

    assert service.unit.repos.conversations.recent_messages(SHOP_A, CONV) == []
    assert service.unit.repos.suggestions.list(SHOP_A, conversation_id=CONV) == []
    with _admin(e2e_pg_dsn).session(None) as conn:
        rows = conn.execute("SELECT msg_id, text FROM message WHERE shop_id = %s", (SHOP_A,))
        assert [(r["msg_id"], r["text"]) for r in rows] == [("e1", None)]
        left = conn.execute("SELECT count(*) AS n FROM message_recall").fetchone()["n"]
    assert left == 0  # 墓碑已用掉


def test_shadow_late_message_keeps_one_record_set_per_turn_on_postgres(pg_service):
    service = pg_service
    repo = service.unit.repos.knowledge
    shop = repo.get_shop(SHOP_A)
    assert shop is not None
    overrides = {**shop.config_overrides, "shadow_mode": True}
    repo.upsert_shop(shop.model_copy(update={"config_overrides": overrides}))
    first = service.handle_event("B", _event(service, "什么时候发货", "w2", 60))
    assert first is not None and first.suggestions
    before = service.unit.repos.suggestions.list(SHOP_A, conversation_id=CONV)
    assert before and {r.status for r in before} == {SuggestionStatus.SHADOW}

    late = service.handle_event("B", _event(service, "在吗", "w1"))

    assert late is not None and late.suggestions
    assert service.unit.repos.suggestions.list(SHOP_A, conversation_id=CONV) == before


def test_revoke_purges_in_small_batches_on_postgres(pg_service, e2e_pg_dsn):
    service = pg_service
    for i in range(4):
        service.handle_event("B", _event(service, "什么时候发货", f"p{i}", 60 * i))
    service.handle_event("B", _recall("p9_recall", "p9"))  # 一条撤回墓碑

    counts = service.revoke_shop(SHOP_A)

    assert counts["PgConversationRepository"] >= 4 + 1 + 1
    with _admin(e2e_pg_dsn).session(None) as conn:
        for table in BUSINESS_TABLES:
            column = "id" if table == "shop" else "shop_id"
            n = conn.execute(
                sql.SQL("SELECT count(*) AS n FROM {} WHERE {} = %s").format(
                    sql.Identifier(table), sql.Identifier(column)
                ),
                (SHOP_A,),
            ).fetchone()["n"]
            assert n == 0, table
