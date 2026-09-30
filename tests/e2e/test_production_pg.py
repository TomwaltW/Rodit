"""生产组装跑在 PostgreSQL 上（§9.1 单元、§9.2 租户隔离、§10 数据模型）。

build_unit + rodit.app.production.postgres_repos（行级安全、令牌加密），大模型用离线替身（不联网），
平台工具 / 适配器 / 写接口用演示实现（平台接口一律 mock）。PostgreSQL 连不上时整组跳过。
"""

from __future__ import annotations

import pytest
from e2e_support import Chat
from psycopg import sql

from rodit.adapters import StandaloneAssistAdapter, TaobaoMockAdapter
from rodit.app import AssistService, build_unit
from rodit.app.demo import SUPPORTED_ACTIONS, seed_unit, taobao_event
from rodit.app.offline import EvidenceEchoLLM, HeuristicQAJudge
from rodit.app.production import postgres_repos
from rodit.core.audit import AuditAction
from rodit.core.config import EngineConfig
from rodit.core.enums import (
    AfterSaleAction,
    DraftKind,
    Intent,
    Platform,
    ReviewState,
    RouteAction,
    SuggestionStatus,
)
from rodit.core.errors import AuthExpired
from rodit.core.pii import BuyerRefHasher
from rodit.core.scopes import PHASE2_SCOPES
from rodit.decision import RuleProvider
from rodit.storage import BUSINESS_TABLES, Database
from rodit.testing.fakes import FakeActionPort, FakeClock, InMemoryToolGateway
from rodit.testing.fixtures import AGENT, SHOP_A, SHOP_B, TEST_SALT, build_demo_world

pytestmark = pytest.mark.pg

TOKEN_KEY = "rodit-e2e-token-key-synthetic-only"  # 只用于测试的合成密钥


@pytest.fixture
def pg_service(e2e_pg_dsn: str, monkeypatch: pytest.MonkeyPatch) -> AssistService:
    admin = Database(e2e_pg_dsn, app_role=None)
    with admin.session(None) as conn:
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
        repos=postgres_repos(Database(e2e_pg_dsn), clock=clock),  # 应用角色，受行级安全约束
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
    seed_unit(unit, world)  # 演示世界里的淘宝店铺（A、B）：店铺、授权（加密落库）、政策、商品、FAQ
    return AssistService(unit)


def test_flows_on_postgres(pg_service: AssistService):
    service = pg_service
    unit = service.unit
    chat = Chat(service, "c_pg")

    # §11.1 尺码咨询 → 推荐落库 → 客服原样发送
    size = chat.say("165 110斤穿什么码", item_id="i_555")
    assert size.degraded == []
    [card] = size.suggestions
    assert "M" in card.text and card.suggestion_id is not None
    assert [e.id for e in card.evidence] == ["item:i_555:size_advice"]
    service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    assert unit.repos.suggestions.get(SHOP_A, card.suggestion_id).status == SuggestionStatus.SENT
    assert service.adoption(SHOP_A).overall.sent == 1

    # §11.2 物流 → 模板；§11.4 站外 → 标准话术
    logistics = chat.say("我的快递到哪了", order_id="o_1002")
    [card] = logistics.suggestions
    assert card.kind == DraftKind.TEMPLATE and "截至" in card.text and "10:30" in card.text
    off = chat.say("加个微信吧，转账给你便宜点")
    assert off.route.action == RouteAction.OFF_PLATFORM_SCRIPT and "站外" in off.suggestions[0].text

    # 重复推送不重复处理（message 表按 msg_id 去重）
    dup = taobao_event("165 110斤穿什么码", msg_id="c_pg_1", session_id="c_pg", item_id="i_555")
    assert service.handle_event("B", dup) is None

    # 新消息到达后，上一批推荐过期（§5.3）
    records = unit.repos.suggestions.list(SHOP_A, conversation_id="c_pg")
    assert {r.status for r in records} == {
        SuggestionStatus.SENT,
        SuggestionStatus.EXPIRED,
        SuggestionStatus.SHOWN,
    }
    msgs = unit.repos.conversations.recent_messages(SHOP_A, "c_pg")
    assert [m.role for m in msgs].count("buyer") == 3

    # §7.3.2 售后审核（R2）：建议 → 客服确认 → 写接口 → 状态和留痕都在库里
    review = service.review_aftersale(SHOP_A, "as_2001")
    assert review.matched_rules == ["R2"] and review.state == ReviewState.CONFIRM
    version = unit.tools.get_aftersale(SHOP_A, "as_2001").status_version
    # 这家店申请了写售后权限（二期，§8.4）：客服确认后执行写操作要有 refund.write
    auth = unit.repos.auth.get(SHOP_A)
    unit.repos.auth.save(auth.model_copy(update={"scopes": sorted(PHASE2_SCOPES)}))
    done = service.confirm_aftersale(
        SHOP_A,
        "as_2001",
        AfterSaleAction.AGREE_REFUND,
        operator=AGENT,
        expected_status_version=version,
    )
    assert done.state == ReviewState.AGREED
    assert unit.repos.reviews.get(SHOP_A, "as_2001").state == ReviewState.AGREED
    assert len(unit.action_port.calls) == 1
    audit = unit.repos.audit
    assert audit.list_events(SHOP_A, action=AuditAction.REVIEW_ACTION)
    assert len(audit.list_events(SHOP_A, action=AuditAction.RESULT)) == 3
    assert audit.list_decisions(SHOP_A)


def test_cross_shop_isolation_on_postgres(pg_service: AssistService, e2e_pg_dsn: str):
    """§9.2 串店检查：同一个单元、同一个 item_id、同一个买家，A、B 两家店的数据互不可见；
    即使 SQL 不带 WHERE，应用角色在 B 的店铺范围内也读不到 A 的行（行级安全）。"""
    service = pg_service
    unit = service.unit
    repos = unit.repos
    a = Chat(service, "c_shared", seller_id=SHOP_A)
    b = Chat(service, "c_shared", seller_id=SHOP_B)

    a_ship = a.say("什么时候发货")
    a_size = a.say("165 110斤穿什么码", item_id="i_555")
    service.review_aftersale(SHOP_A, "as_2003")
    b_ship = b.say("什么时候发货")
    b_courier = b.say("发什么快递")
    b.say("165 110斤穿什么码", item_id="i_555")

    assert "48" in a_ship.suggestions[0].text and "160–165" in a_size.suggestions[0].text
    assert "72" in b_ship.suggestions[0].text and "48" not in b_ship.suggestions[0].text
    assert "圆通" in b_courier.suggestions[0].text
    b_prompts = "\n".join(
        m.content for c in unit.llm.calls if c.shop_id == SHOP_B for m in c.messages
    )
    for a_only in ("法式碎花连衣裙", "48 小时", "中通", "160–165"):
        assert a_only not in b_prompts

    a_ids = {r.id for r in repos.suggestions.list(SHOP_A)}
    b_ids = {r.id for r in repos.suggestions.list(SHOP_B)}
    assert a_ids and b_ids and not a_ids & b_ids
    assert all(repos.suggestions.get(SHOP_B, i) is None for i in a_ids)
    assert {m.shop_id for m in repos.conversations.recent_messages(SHOP_B, "c_shared")} == {SHOP_B}
    assert repos.reviews.get(SHOP_B, "as_2003") is None
    assert repos.reviews.get(SHOP_A, "as_2003") is not None
    assert all(e.shop_id == SHOP_B for e in repos.audit.list_events(SHOP_B))
    assert all(d.shop_id == SHOP_B for d in repos.audit.list_decisions(SHOP_B))

    # 行级安全：不带 WHERE 的原始 SQL
    db = Database(e2e_pg_dsn)
    for table in ("suggestion", "message", "audit_log", "decision_log", "aftersale_review"):
        with db.session(SHOP_B) as conn:
            rows = conn.execute(sql.SQL("SELECT shop_id FROM {}").format(sql.Identifier(table)))
            assert {r["shop_id"] for r in rows} <= {SHOP_B}, table
        with db.session(None) as conn:
            count = conn.execute(
                sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(table))
            ).fetchone()
            assert count["n"] == 0, table

    # 取消订购（§8.4）：A 的数据全部删除，B 不受影响
    counts = service.revoke_shop(SHOP_A)
    assert counts["auth"] == 1
    assert repos.suggestions.list(SHOP_A) == [] and repos.reviews.get(SHOP_A, "as_2003") is None
    assert repos.knowledge.get_shop(SHOP_A) is None
    assert repos.suggestions.list(SHOP_B) and repos.knowledge.get_shop(SHOP_B) is not None
    with pytest.raises(AuthExpired):
        a.say("在吗")
    assert b.say("在吗").route.intent == Intent.GREETING
