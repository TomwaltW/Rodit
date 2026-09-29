"""组装与应用服务（§9.1 单元化、§4.2 消息流程、§5.8 反馈、§7.3.2 售后审核、§8.4 取消订购）。"""

from __future__ import annotations

import pytest

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, main, taobao_event
from rodit.core.audit import RESULT_DETAIL_KEYS, AuditAction
from rodit.core.enums import (
    AfterSaleAction,
    Intent,
    Mode,
    Platform,
    RejectReason,
    ReviewState,
    Role,
    RouteAction,
    ShopStatus,
    SuggestionStatus,
)
from rodit.core.errors import (
    ActionNotAllowed,
    AuthExpired,
    ShopInactive,
    TenantIsolationError,
)
from rodit.testing.fixtures import AGENT, MANAGER, SHOP_A, SHOP_B, SHOP_C, SHOP_D


@pytest.fixture
def service() -> AssistService:
    return AssistService(build_demo_unit(Platform.TAOBAO))


def _event(text: str, n: int, **kw):
    return taobao_event(text, msg_id=f"t_{n}", **kw)


def test_units_hold_only_their_platform_shops():
    """§9.1：每个平台单元只有本平台的店铺。"""
    for platform, shops in (
        (Platform.TAOBAO, {SHOP_A, SHOP_B}),
        (Platform.PDD, {SHOP_C}),
        (Platform.DOUYIN, {SHOP_D}),
    ):
        unit = build_demo_unit(platform)
        present = {s for s in (SHOP_A, SHOP_B, SHOP_C, SHOP_D) if unit.repos.knowledge.get_shop(s)}
        assert present == shops


def test_buyer_message_flows_to_sidebar_with_evidence(service: AssistService):
    result = service.handle_event("B", _event("165 110斤穿什么码", 1, item_id="i_555"))
    assert result is not None and result.route.intent == Intent.SIZE_ADVICE
    [card] = result.suggestions
    assert "M" in card.text and card.suggestion_id is not None
    assert [e.id for e in card.evidence] == ["item:i_555:size_advice"]
    adapter = service.unit.adapters.get(Platform.TAOBAO, "B")
    assert adapter.sidebar(SHOP_A, "conv_demo") == result.suggestions
    # 重复推送不重复处理（§8.3 按 msg_id 去重）
    assert service.handle_event("B", _event("165 110斤穿什么码", 1, item_id="i_555")) is None
    # 每条消息恰好一条 engine.result，字段齐全
    [event] = service.unit.repos.audit.list_events(SHOP_A, action=AuditAction.RESULT)
    assert set(event.detail) == set(RESULT_DETAIL_KEYS)


def test_expired_auth_stops_processing(service: AssistService):
    """§8.4：授权过期或取消订购后停止处理。"""
    service.unit.repos.auth.delete(SHOP_A)
    with pytest.raises(AuthExpired):
        service.handle_event("B", _event("在吗", 2))


def test_feedback_sent_mirrors_agent_reply_and_closes_loop(service: AssistService):
    result = service.handle_event("B", _event("什么时候发货", 3))
    card = result.suggestions[0]
    outcome = service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    assert outcome.record.status == SuggestionStatus.SENT
    msgs = service.unit.repos.conversations.recent_messages(SHOP_A, "conv_demo")
    assert msgs[-1].role == Role.AGENT and msgs[-1].text == card.text
    with pytest.raises(ActionNotAllowed):  # 已反馈过的推荐不能再反馈
        service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    report = service.adoption(SHOP_A)
    assert report.overall.sent == 1


def test_new_message_expires_previous_suggestions(service: AssistService):
    first = service.handle_event("B", _event("什么时候发货", 4))
    service.unit.clock.set(service.unit.clock.now().replace(minute=40))
    service.handle_event("B", _event("能开发票吗", 5, send_time_ms=None))
    old = service.unit.repos.suggestions.get(SHOP_A, first.suggestions[0].suggestion_id)
    assert old.status == SuggestionStatus.EXPIRED


def test_edited_feedback_creates_candidate_knowledge_for_review(service: AssistService):
    result = service.handle_event("B", _event("新疆能到吗", 6))
    card = result.suggestions[0]
    service.feedback_edited(
        SHOP_A, card.suggestion_id, agent_id=AGENT, edited_text="亲，新疆可以发，补 15 元运费哦~"
    )
    pending = service.pending_candidates(SHOP_A)
    assert any("15 元" in k.answer for k in pending)
    approved = service.approve_candidate(SHOP_A, pending[-1].id, merchant="店长")
    assert approved.status == "approved"


def test_rejected_missing_knowledge_shows_up_in_gaps(service: AssistService):
    result = service.handle_event("B", _event("这个是纯棉的吗", 7, item_id="i_555"))
    assert result.suggestions
    service.feedback_rejected(
        SHOP_A,
        result.suggestions[0].suggestion_id,
        agent_id=AGENT,
        reason=RejectReason.MISSING_KNOWLEDGE,
    )
    gaps = service.knowledge_gaps(SHOP_A)
    assert gaps and all({"marker", "description"} <= set(g) for g in gaps)


def test_standalone_mode_c_answers_policy_questions_without_platform_data():
    service = AssistService(build_demo_unit(Platform.PDD))
    result = service.ask_standalone(shop_id=SHOP_C, session_id="web_1", question="什么时候发货")
    assert result is not None and result.suggestions
    assert "24" in result.suggestions[0].text
    # 模式 C 查不到订单：物流问题给"我帮您查一下"或澄清
    logistics = service.ask_standalone(shop_id=SHOP_C, session_id="web_1", question="快递到哪了")
    assert logistics is not None


def test_shadow_shop_generates_but_shows_nothing():
    """§13.4 影子模式：店铺 D 开了影子模式，推荐只记录不展示。"""
    unit = build_demo_unit(Platform.DOUYIN)
    service = AssistService(unit)
    raw = {
        "shop_id": SHOP_D,
        "conversation_short_id": "dy_conv_1",
        "server_message_id": "dy_1",
        "sender": {"role": "customer", "uid": "dy_demo_buyer_001", "nickname": "合成"},
        "content": {"text": "什么时候发货"},
        "create_time": int(unit.clock.now().timestamp()),
    }
    result = service.handle_event("B", raw)
    assert result is not None
    assert result.mode == Mode.SHADOW
    adapter = unit.adapters.get(Platform.DOUYIN, "B")
    assert adapter.sidebar(SHOP_D, result.conversation_id) == []
    records = unit.repos.suggestions.list(SHOP_D)
    assert records and all(r.status == SuggestionStatus.SHADOW for r in records)


def test_aftersale_review_and_confirm_requires_manager_for_r6(service: AssistService):
    review = service.review_aftersale(SHOP_A, "as_2005")
    assert review.requires_manager and review.state == ReviewState.CONFIRM
    info = service.unit.tools.get_aftersale(SHOP_A, "as_2005")
    with pytest.raises(ActionNotAllowed):
        service.confirm_aftersale(
            SHOP_A,
            "as_2005",
            AfterSaleAction.AGREE_RETURN,
            operator=AGENT,
            expected_status_version=info.status_version,
        )
    done = service.confirm_aftersale(
        SHOP_A,
        "as_2005",
        AfterSaleAction.AGREE_RETURN,
        operator=MANAGER,
        expected_status_version=info.status_version,
        by_manager=True,
    )
    assert done.state == ReviewState.WAIT_RETURN
    assert service.unit.action_port.calls[-1]["action"] == AfterSaleAction.AGREE_RETURN


def test_revoke_shop_purges_every_store(service: AssistService):
    service.handle_event("B", _event("什么时候发货", 8))
    service.review_aftersale(SHOP_A, "as_2001")
    counts = service.revoke_shop(SHOP_A)
    assert counts["auth"] == 1 and sum(counts.values()) > 5
    repos = service.unit.repos
    assert repos.knowledge.get_shop(SHOP_A) is None
    assert repos.suggestions.list(SHOP_A) == []
    assert repos.reviews.get(SHOP_A, "as_2001") is None
    assert repos.audit.list_events(SHOP_A) == []
    assert repos.knowledge.get_shop(SHOP_B) is not None  # 别的店铺不受影响
    with pytest.raises(AuthExpired):
        service.handle_event("B", _event("在吗", 9))


def test_off_platform_request_gets_standard_script(service: AssistService):
    result = service.handle_event("B", _event("加个微信吧，转账给你便宜点", 10))
    assert result.route.action == RouteAction.OFF_PLATFORM_SCRIPT
    assert "站外" in result.suggestions[0].text


def test_demo_script_runs(capsys):
    assert main() == 0
    out = capsys.readouterr().out
    assert "§11.1" in out and "身高 160–165 cm" in out and "R6" in out


def test_expired_auth_leaves_no_dedup_record(service: AssistService):
    """§8.4 / §9.2：授权检查在适配器改状态之前；授权恢复后同一条消息仍能正常处理。"""
    auth = service.unit.repos.auth.get(SHOP_A)
    service.unit.repos.auth.delete(SHOP_A)
    with pytest.raises(AuthExpired):
        service.handle_event("B", _event("什么时候发货", 11))
    adapter = service.unit.adapters.get(Platform.TAOBAO, "B")
    assert adapter.get_context(SHOP_A, "conv_demo").recent_messages == []
    service.unit.repos.auth.save(auth)
    assert service.handle_event("B", _event("什么时候发货", 11)) is not None


def test_event_for_other_shop_is_rejected_before_any_state_change(service: AssistService):
    """§9.2：调用方身份和事件店铺不符时直接拒绝，不在另一个店铺留下去重或会话记录。"""
    raw = _event("什么时候发货", 12, seller_id=SHOP_B)
    with pytest.raises(TenantIsolationError):
        service.handle_event("B", raw, expected_shop_id=SHOP_A)
    adapter = service.unit.adapters.get(Platform.TAOBAO, "B")
    assert adapter.get_context(SHOP_B, "conv_demo").recent_messages == []
    assert service.unit.repos.audit.list_events(SHOP_B) == []
    # 身份相符时正常处理（说明上面没被去重吃掉）
    assert service.handle_event("B", raw, expected_shop_id=SHOP_B) is not None


def test_unrecognizable_event_is_ignored(service: AssistService):
    assert service.handle_event("B", {"foo": "bar"}) is None
    assert service.handle_event("B", _event("在吗", 13, seller_id="")) is None


@pytest.mark.parametrize("status", [None, ShopStatus.SUSPENDED])
def test_standalone_rejects_missing_or_suspended_shop(status):
    """模式 C 不需要平台授权，但店铺必须存在且没停用（§8.4）。"""
    unit = build_demo_unit(Platform.TAOBAO)
    service = AssistService(unit)
    shop_id = "s_missing"
    if status is not None:
        shop_id = SHOP_A
        shop = unit.repos.knowledge.get_shop(SHOP_A)
        unit.repos.knowledge.upsert_shop(shop.model_copy(update={"status": status}))
    with pytest.raises(ShopInactive):
        service.ask_standalone(shop_id=shop_id, session_id="web_x", question="能开发票吗")
