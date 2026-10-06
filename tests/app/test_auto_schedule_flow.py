"""应用流程：全自动先在非工作时间试行、半自动（方案 §15 Phase 3、§5.6、§12.2、§13.4）。

淘宝模式 A（机器人接待）：客服工作时间内退回智能辅助、侧边栏展示推荐、会话转给在岗客服；
工作时间以外照常全自动（带 AI 标识）。工作时间由商家在设置页或问卷第 13 题填（北京时间）。
模式 B 插件：开了半自动的类型，侧边栏把推荐填进输入框（input_prefill），客服按发送后照常记
"原样发送"。全部合成数据，不联网。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rodit.adapters import TaobaoMockAdapter
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import Intent, Mode, Platform, SuggestionStatus
from rodit.engine.modes import RESULT_AUTO_GATE
from rodit.testing.fakes import FakeDecisionProvider, router_answers
from rodit.testing.fixtures import AGENT, SHOP_A, SHOP_B

PREFIX = "【智能客服】"
MERCHANT = "owner_a"


def auto_service(**overrides: Any) -> tuple[AssistService, TaobaoMockAdapter]:
    unit = build_demo_unit(
        Platform.TAOBAO,
        route_provider=FakeDecisionProvider(router_answers(Intent.GREETING, 0.98)),
    )
    adapter = TaobaoMockAdapter(
        unit.tools, unit.clock, unit.hasher, access="A", auto_send_enabled=True
    )
    unit.adapters.register(adapter)
    unit.repos.knowledge.update_shop_config(SHOP_A, {"auto_reply_enabled": True, **overrides})
    return AssistService(unit), adapter


def robot(service: AssistService, text: str, msg_id: str, when: datetime, **kw: Any) -> Any:
    service.unit.clock.set(when)
    raw = taobao_event(text, msg_id=msg_id, send_time_ms=int(when.timestamp() * 1000), **kw)
    raw["reception"] = "robot"
    return service.handle_event("A", raw)


def cst(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, 28, hour, minute, tzinfo=CST)


def test_service_hours_hand_robot_conversations_to_agents():
    service, adapter = auto_service()
    service.console.update_config(SHOP_A, {"service_hours": ["9:00-23:00"]}, merchant=MERCHANT)
    result = robot(service, "在吗", "r1", cst(10, 35))
    assert result is not None and result.mode == Mode.ASSIST
    assert result.auto_reply is None and result.transfer_to_human is True
    assert adapter.sent_messages(SHOP_A) == []
    [transfer] = adapter.transfers(SHOP_A)
    assert transfer.conversation_id == "conv_demo"
    # 侧边栏照常展示推荐，接手的客服直接能用
    assert adapter.sidebar(SHOP_A, "conv_demo") == result.suggestions and result.suggestions
    [event] = service.unit.repos.audit.list_events(SHOP_A, action=AuditAction.RESULT.value)
    assert event.detail[RESULT_AUTO_GATE] == "service_hours"
    # 转过人工的会话，后面的消息由客服接待（接入层不再标全自动）
    follow = robot(service, "发什么快递", "r2", cst(23, 30))
    assert follow is not None and follow.mode == Mode.ASSIST and follow.auto_reply is None


def test_off_hours_robot_conversation_is_answered_automatically():
    service, adapter = auto_service(service_hours=["09:00-23:00"])
    result = robot(service, "在吗", "r1", cst(23, 30), session_id="night")
    assert result is not None and result.mode == Mode.AUTO and result.auto_reply is not None
    [sent] = adapter.sent_messages(SHOP_A)
    assert sent.text.startswith(PREFIX) and sent.conversation_id == "night"
    assert adapter.transfers(SHOP_A) == []
    record = service.unit.repos.suggestions.get(SHOP_A, result.auto_reply.suggestion_id)
    assert record is not None and record.status == SuggestionStatus.SENT


def test_unknown_service_hours_never_go_auto():
    service, adapter = auto_service()
    result = robot(service, "在吗", "r1", cst(3))
    assert result is not None and result.mode == Mode.ASSIST and result.transfer_to_human
    assert adapter.sent_messages(SHOP_A) == []


def test_questionnaire_answer_sets_service_hours_for_the_schedule():
    service, adapter = auto_service()
    out = service.apply_questionnaire(SHOP_A, {13: "早9点到晚11点"})
    assert out["config"] == ["service_hours"]
    shop = service.unit.repos.knowledge.get_shop(SHOP_A)
    assert shop is not None and shop.config_overrides["service_hours"] == ["09:00-23:00"]
    assert robot(service, "在吗", "r1", cst(22, 59), session_id="s1").mode == Mode.ASSIST
    assert robot(service, "在吗", "r2", cst(23), session_id="s2").mode == Mode.AUTO


def test_always_schedule_is_set_by_us_not_the_merchant():
    service, adapter = auto_service(service_hours=["09:00-23:00"], auto_reply_schedule="always")
    result = robot(service, "在吗", "r1", cst(10, 35))
    assert result is not None and result.mode == Mode.AUTO and result.auto_reply is not None


def test_schedule_is_per_shop():
    """店铺 A 填了工作时间、开了全自动，店铺 B 没开：B 店照常按智能辅助处理，不受 A 影响。"""
    service, adapter = auto_service(service_hours=["09:00-23:00"])
    b = robot(service, "在吗", "rb", cst(23, 30), seller_id=SHOP_B, buyer="BUYER_REFUND")
    assert b is not None and b.mode == Mode.ASSIST and b.transfer_to_human is False
    a = robot(service, "在吗", "ra", cst(23, 30))
    assert a is not None and a.mode == Mode.AUTO
    assert [m.conversation_id for m in adapter.sent_messages(SHOP_B)] == []


# ================================================================ 半自动（模式 B 插件）


def plugin_service() -> AssistService:
    unit = build_demo_unit(
        Platform.TAOBAO,
        route_provider=FakeDecisionProvider(router_answers(Intent.SHIPPING_POLICY, 0.95)),
    )
    return AssistService(unit)


def test_semi_auto_prefills_the_plugin_input_box_and_sending_is_still_the_agent():
    service = plugin_service()
    service.unit.repos.knowledge.update_shop_config(
        SHOP_A, {"semi_auto_intents": ["shipping_policy"]}
    )
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="m1"))
    assert result is not None and result.suggestions[0].prefill is True
    adapter = service.unit.adapters.get(Platform.TAOBAO, "B")
    payload = adapter.sidebar_payload(SHOP_A, "conv_demo")
    assert payload is not None and payload.input_prefill is not None
    sid = payload.input_prefill.suggestion_id
    assert sid == result.suggestions[0].suggestion_id
    assert payload.input_prefill.text == result.suggestions[0].text
    # 没有任何东西自动发出：客服按发送才发，反馈照常记"原样发送"
    record = service.unit.repos.suggestions.get(SHOP_A, sid)
    assert record is not None and record.status == SuggestionStatus.SHOWN
    service.feedback_sent(SHOP_A, sid, agent_id=AGENT)
    record = service.unit.repos.suggestions.get(SHOP_A, sid)
    assert record is not None and record.status == SuggestionStatus.SENT
    report = service.adoption(SHOP_A)
    assert report.overall.as_is_groups == 1 and report.overall.groups == 1


def test_without_semi_auto_the_plugin_input_box_stays_empty():
    service = plugin_service()
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="m1"))
    assert result is not None and result.suggestions and not result.suggestions[0].prefill
    adapter = service.unit.adapters.get(Platform.TAOBAO, "B")
    payload = adapter.sidebar_payload(SHOP_A, "conv_demo")
    assert payload is not None and payload.input_prefill is None


def test_standalone_page_gets_the_flag_on_the_card():
    """模式 C 网页工具：卡片上带 prefill（网页显示"半自动：可直接发送"），没有平台输入框可填。"""
    service = plugin_service()
    service.unit.repos.knowledge.update_shop_config(
        SHOP_A, {"semi_auto_intents": ["shipping_policy"]}
    )
    result = service.ask_standalone(
        shop_id=SHOP_A, session_id="web1", question="什么时候发货", request_id="q1"
    )
    assert result is not None and [c.prefill for c in result.suggestions] == [True]
