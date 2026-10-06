"""生产组装跑在 PostgreSQL 上时的买家图片（§15 Phase 3、§10 message 表、§8.3 去重、§5.3 撤回）。

只发图的消息在 message 表里存空文字（没有新列，图片标识只记在单元缓存里）；推荐记录的买家问题是
图片描述；随后的文字消息按合并连发看到前一条的图片；撤回、重推照常。PostgreSQL 连不上时整组跳过。
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from psycopg import sql

from rodit.adapters import TaobaoMockAdapter
from rodit.app import AssistService, build_unit
from rodit.app.demo import SUPPORTED_ACTIONS, seed_unit, taobao_event
from rodit.app.offline import EvidenceEchoLLM, HeuristicQAJudge
from rodit.app.production import postgres_repos
from rodit.core.config import EngineConfig
from rodit.core.enums import Intent, Platform, ReviewState, Role, SuggestionStatus
from rodit.core.pii import BuyerRefHasher
from rodit.decision import RuleProvider
from rodit.storage import BUSINESS_TABLES, Database
from rodit.testing.fakes import (
    FakeActionPort,
    FakeClock,
    FakeImageUnderstanding,
    InMemoryToolGateway,
)
from rodit.testing.fixtures import DEMO_IMAGE_FINDINGS, SHOP_A, TEST_SALT, build_demo_world

pytestmark = pytest.mark.pg

TOKEN_KEY = "rodit-e2e-token-key-synthetic-only"  # 只用于测试的合成密钥
DAMAGE = "【买家发来图片：破损（外包装一角压扁，裤腿有一处约 3 厘米的撕裂）】"


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
        adapters=[TaobaoMockAdapter(tools, clock, hasher, access="B")],
        action_port=FakeActionPort(
            Platform.TAOBAO,
            supported=SUPPORTED_ACTIONS[Platform.TAOBAO],
            gateway=tools,
            clock=clock,
        ),
        route_provider=RuleProvider(),
        qa_provider=HeuristicQAJudge(),
        vision=FakeImageUnderstanding(DEMO_IMAGE_FINDINGS),
    )
    seed_unit(unit, world)
    return AssistService(unit)


def _image(service: AssistService, msg_id: str, *, after_s: float, **kw) -> dict:
    clock = service.unit.clock
    clock.set(clock.now() + timedelta(seconds=after_s))
    raw = taobao_event(
        "[图片]",
        msg_id=msg_id,
        session_id="c_pg_img",
        send_time_ms=int(clock.now().timestamp() * 1000),
        **kw,
    )
    raw.update(msg_type="image", pic_key="img_buyer_damage_1")
    return raw


def _text(service: AssistService, text: str, msg_id: str, *, after_s: float) -> dict:
    clock = service.unit.clock
    clock.set(clock.now() + timedelta(seconds=after_s))
    return taobao_event(
        text,
        msg_id=msg_id,
        session_id="c_pg_img",
        send_time_ms=int(clock.now().timestamp() * 1000),
    )


def test_image_only_messages_on_postgres(pg_service: AssistService):
    service = pg_service
    unit = service.unit

    # 只发图：存空文字，推荐照常落库
    first = service.handle_event("B", _image(service, "pg_p1", after_s=60, order_id="o_1013"))
    assert first is not None and first.route.intent == Intent.QUALITY_ISSUE
    assert first.degraded == []
    [stored] = unit.repos.conversations.recent_messages(SHOP_A, "c_pg_img")
    assert stored.role == Role.BUYER and (stored.text or "") == ""
    assert stored.order_ref == "o_1013"
    records = unit.repos.suggestions.list(SHOP_A, conversation_id="c_pg_img")
    assert records and {r.question_text for r in records} == {DAMAGE}
    assert all(r.status == SuggestionStatus.SHOWN for r in records)

    # 平台重推同一条：按 msg_id 去重，不再识别
    assert service.handle_event("B", _image(service, "pg_p1", after_s=0)) is None
    assert len(unit.vision.calls) == 1

    # 1 秒后补一句文字：和图片合并成一轮，识别结果复用；上一批推荐过期（§5.3）
    second = service.handle_event("B", _text(service, "收到就是这样的", "pg_t1", after_s=1))
    assert second is not None and second.msg_ids == ["pg_p1", "pg_t1"]
    assert second.route.intent == Intent.QUALITY_ISSUE
    assert len(unit.vision.calls) == 1
    latest = [
        r
        for r in unit.repos.suggestions.list(SHOP_A, conversation_id="c_pg_img")
        if r.status == SuggestionStatus.SHOWN
    ]
    assert latest and {r.question_text for r in latest} == {f"收到就是这样的 {DAMAGE}"}

    # 撤回图片消息：会话里去掉，推荐过期
    recall = taobao_event("", msg_id="pg_r1", session_id="c_pg_img")
    recall.update(msg_type="recall", recall_msg_id="pg_p1")
    assert service.handle_event("B", recall) is None
    ids = [m.msg_id for m in unit.repos.conversations.recent_messages(SHOP_A, "c_pg_img")]
    assert "pg_p1" not in ids and "pg_t1" in ids


def test_r5_evidence_summary_on_postgres(pg_service: AssistService):
    review = pg_service.review_aftersale(SHOP_A, "as_2009")
    assert review.matched_rules == ["R5"] and review.state == ReviewState.CONFIRM
    assert review.reason.endswith(
        "图片理解（仅供参考）：3 张凭证中 2 张显示破损（裤腿撕裂，线头外露；外包装压痕明显）"
    )
    stored = pg_service.unit.repos.reviews.get(SHOP_A, "as_2009")
    assert stored is not None and stored.reason == review.reason
