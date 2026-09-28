"""生产组装跑在 PostgreSQL 上：发出的回复写回会话、撤回、自由文本里的控制字符（§5.3、§10）。

和内存实现同样的行为要在 PostgreSQL 上成立：回声接替写回的记录后能被撤回；带个人信息的编辑回复
只存一份；协商备注、问卷回答里的 NUL 不让 JSONB 写入失败。大模型用离线替身，平台接口是模拟实现。
"""

from __future__ import annotations

import pytest
from psycopg import sql

from rodit.adapters import StandaloneAssistAdapter, TaobaoMockAdapter
from rodit.aftersale.state_machine import history
from rodit.app import AssistService, build_unit
from rodit.app.demo import SUPPORTED_ACTIONS, seed_unit, taobao_event
from rodit.app.offline import EvidenceEchoLLM, HeuristicQAJudge
from rodit.app.production import postgres_repos
from rodit.core.config import EngineConfig
from rodit.core.enums import Platform, ReviewState, Role
from rodit.core.pii import BuyerRefHasher
from rodit.decision import RuleProvider
from rodit.storage import BUSINESS_TABLES, Database
from rodit.testing.fakes import FakeActionPort, FakeClock, InMemoryToolGateway
from rodit.testing.fixtures import AGENT, SHOP_A, TEST_SALT, build_demo_world

pytestmark = pytest.mark.pg

TOKEN_KEY = "rodit-e2e-token-key-synthetic-only"  # 只用于测试的合成密钥
CONV = "c_pg_out"


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
        repos=postgres_repos(Database(e2e_pg_dsn), clock=clock),
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


def _out(text: str, msg_id: str, send_time_ms: int) -> dict:
    return {
        "seller_id": SHOP_A,
        "session_id": CONV,
        "msg_id": msg_id,
        "msg_type": "text",
        "direction": "out",
        "content": text,
        "send_time_ms": send_time_ms,
    }


def _stored(service: AssistService) -> list[tuple[str, Role]]:
    msgs = service.unit.repos.conversations.recent_messages(SHOP_A, CONV)
    return [(m.msg_id, m.role) for m in msgs]


def test_echo_takes_over_mirrored_reply_and_can_be_recalled_on_postgres(pg_service):
    service = pg_service
    result = service.handle_event(
        "B", taobao_event("什么时候发货", msg_id="q1", session_id=CONV, send_time_ms=_ms(service))
    )
    assert result is not None
    card = result.suggestions[0]
    service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    service.handle_event("B", _out(card.text, "plat_out_1", _ms(service, 1)))
    assert _stored(service) == [("q1", Role.BUYER), ("plat_out_1", Role.AGENT)]
    recall = {
        "seller_id": SHOP_A,
        "session_id": CONV,
        "msg_id": "recall_1",
        "msg_type": "recall",
        "recall_msg_id": "plat_out_1",
    }
    assert service.handle_event("B", recall) is None
    assert _stored(service) == [("q1", Role.BUYER)]


def test_edited_reply_with_pii_is_stored_once_on_postgres(pg_service):
    service = pg_service
    result = service.handle_event(
        "B", taobao_event("退货寄到哪里", msg_id="a1", session_id=CONV, send_time_ms=_ms(service))
    )
    assert result is not None
    sent = "亲，退货请寄到浙江省杭州市余杭区文一西路1001号3幢，收件人小王，电话13900001111"
    service.feedback_edited(
        SHOP_A, result.suggestions[0].suggestion_id, agent_id=AGENT, edited_text=sent
    )
    service.handle_event("B", _out(sent, "a2", _ms(service, 3)))
    assert _stored(service) == [("a1", Role.BUYER), ("a2", Role.AGENT)]


def test_nul_in_negotiate_note_and_questionnaire_answer_on_postgres(pg_service):
    service = pg_service
    service.review_aftersale(SHOP_A, "as_2007")
    saved = service.negotiate_aftersale(
        SHOP_A, "as_2007", operator=AGENT, note="买家同意换货\x00，明天寄回"
    )
    assert saved.state == ReviewState.NEGOTIATE
    assert history(saved)[-1]["note"] == "买家同意换货，明天寄回"

    out = service.apply_questionnaire(SHOP_A, {14: "最好\x00、第一"})
    assert out["config"] == ["banned_words"]
    shop = service.unit.repos.knowledge.get_shop(SHOP_A)
    assert shop is not None
    assert "最好" in shop.config_overrides["banned_words"]
