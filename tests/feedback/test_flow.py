"""接入层 + 反馈闭环 + 报表串起来（方案 §4.2、§5.8、§13.4）：引擎部分用手写的推荐记录代替。"""

from __future__ import annotations

from rodit.adapters import AuthGuard, TaobaoMockAdapter
from rodit.core import evidence as ev
from rodit.core.audit import InMemoryAuditSink
from rodit.core.enums import (
    Confidence,
    DraftKind,
    Intent,
    KnowledgeStatus,
    SuggestionStatus,
)
from rodit.core.models import SuggestionCard, SuggestionRecord
from rodit.core.pii import BuyerRefHasher, mask_pii
from rodit.feedback import (
    FeedbackService,
    adoption_report,
    agent_replies_by_msg,
    phase_gate,
    shadow_report,
)
from rodit.inmemory import (
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemorySuggestionRepository,
)
from rodit.testing.fakes import FakeClock, InMemoryToolGateway
from rodit.testing.fixtures import (
    BUYER,
    CONV,
    DEFAULT_NOW,
    RAW_BUYERS,
    SHOP_A,
    TEST_SALT,
    build_demo_world,
    seed_knowledge,
)


def _raw(msg_id: str, content: str, *, direction: str = "in") -> dict:
    return {
        "seller_id": SHOP_A,
        "session_id": CONV,
        "msg_id": msg_id,
        "msg_type": "text",
        "direction": direction,
        "buyer_open_uid": RAW_BUYERS["BUYER"][1],
        "buyer_nick": "小美爱穿搭",
        "content": content,
        "item_id": "i_555",
        "sub_account": "小雅女装旗舰店:kf_02",
        "send_time_ms": int(DEFAULT_NOW.timestamp() * 1000),
    }


def test_assist_flow_from_platform_message_to_adoption_report():
    world = build_demo_world()
    clock = FakeClock()
    adapter = TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, BuyerRefHasher(TEST_SALT), access="B"
    )
    suggestions = InMemorySuggestionRepository()
    knowledge = InMemoryKnowledgeRepository(clock)
    seed_knowledge(knowledge, world)
    service = FeedbackService(
        suggestions, knowledge, InMemoryFeedbackRepository(), InMemoryAuditSink(), clock
    )

    msg = adapter.on_message(_raw("m1", "165 110斤穿什么码"))
    assert msg.buyer_ref == BUYER
    # 引擎（这里手写）为每张卡片写 SHOWN 记录并回填 suggestion_id
    cards = []
    for text in ("亲，建议选 M 码哦~", "亲，M 码更合适"):
        record = suggestions.add(
            SuggestionRecord(
                shop_id=msg.shop_id,
                conversation_id=msg.conversation_id,
                msg_id=msg.msg_id,
                question_text=mask_pii(msg.text),
                item_id=msg.item_card.item_id,
                intent=Intent.SIZE_ADVICE,
                text=text,
                kind=DraftKind.LLM,
                evidence_ids=[ev.item_size_advice("i_555")],
                status=SuggestionStatus.SHOWN,
                agent_id=msg.agent_id,
                latency_ms=1800,
                created_at=clock.now(),
            )
        )
        cards.append(
            SuggestionCard(
                suggestion_id=record.id,
                text=text,
                intent=Intent.SIZE_ADVICE,
                kind=DraftKind.LLM,
                confidence=Confidence.HIGH,
            )
        )
    adapter.show_suggestions(msg.shop_id, msg.conversation_id, cards)
    shown = adapter.sidebar(SHOP_A, CONV)

    # 客服点第一张卡片"编辑后发送"
    out = service.record_edited(
        SHOP_A, shown[0].suggestion_id, agent_id="kf_02", edited_text="亲，建议 M 码，版型偏小哦~"
    )
    assert out.candidate is not None
    service.approve_candidate(SHOP_A, out.candidate.id, merchant="merchant_owner")
    faq_ids = {k.id for k in knowledge.list_knowledge(SHOP_A, status=KnowledgeStatus.APPROVED)}
    assert out.candidate.id in faq_ids

    # 引擎在买家新消息到达时把另一张 SHOWN 改成 EXPIRED（这里手写）
    other = suggestions.get(SHOP_A, shown[1].suggestion_id)
    suggestions.update(other.model_copy(update={"status": SuggestionStatus.EXPIRED}))

    report = adoption_report(suggestions.list(SHOP_A))
    assert report.overall.adoption_rate == 0.5  # 按条
    assert report.overall.group_adoption_rate == 1.0  # 按推荐组
    assert phase_gate(report, platform_violations=0).passed


def test_shadow_flow_uses_agent_replies_seen_by_adapter():
    world = build_demo_world()
    clock = FakeClock()
    adapter = TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, BuyerRefHasher(TEST_SALT), access="B"
    )
    adapter.on_message(_raw("m1", "发什么快递"))
    adapter.on_message(_raw("a1", "亲，默认发中通快递哦", direction="out"))
    shadow = SuggestionRecord(
        id=1,
        shop_id=SHOP_A,
        conversation_id=CONV,
        msg_id="m1",
        intent=Intent.SHIPPING_POLICY,
        text="亲，默认发中通快递哦~",
        status=SuggestionStatus.SHADOW,
        created_at=clock.now(),
    )
    replies = agent_replies_by_msg(adapter.get_context(SHOP_A, CONV).recent_messages)
    report = shadow_report([shadow], replies)
    assert report.compared == 1 and report.items[0].similarity == 1.0


def test_revoked_shop_stops_processing():
    """取消订购后（§8.4）app 层先查授权再处理；这里只验证组件能配合。"""
    from rodit.core.errors import AuthExpired
    from rodit.inmemory import InMemoryShopAuthRepository

    world = build_demo_world()
    clock = FakeClock()
    adapter = TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, BuyerRefHasher(TEST_SALT), access="B"
    )
    suggestions = InMemorySuggestionRepository()
    guard = AuthGuard(InMemoryShopAuthRepository(), clock)
    adapter.on_message(_raw("m1", "在吗"))
    counts = guard.revoke(SHOP_A, [adapter, suggestions])
    assert counts == {"auth": 0, "TaobaoMockAdapter": 1, "InMemorySuggestionRepository": 0}
    try:
        guard.require(SHOP_A)
    except AuthExpired:
        pass
    else:
        raise AssertionError("撤销授权后应停止处理")
    assert adapter.get_context(SHOP_A, CONV).recent_messages == []
