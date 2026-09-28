"""应用服务的消息流程（§8.3 按会话保证顺序、§5.3 最近几轮、§5.7 / §12.2 全自动故障转人工、
§8.4 取消订购后删除数据）。全部合成数据，不联网。"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest

from rodit.adapters import TaobaoMockAdapter
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.core.enums import Intent, Mode, Platform, Role, SuggestionStatus
from rodit.core.models import SendResult
from rodit.testing.fakes import FakeDecisionProvider, router_answers
from rodit.testing.fixtures import AGENT, SHOP_A, SHOP_C

CONV = "conv_demo"


def _ms(service: AssistService, offset_s: float = 0) -> int:
    return int((service.unit.clock.now().timestamp() + offset_s) * 1000)


def _echo(text: str, msg_id: str, send_time_ms: int, *, session_id: str = CONV) -> dict:
    """客服 / 机器人发出的消息回声（淘宝 direction=out）。"""
    return {
        "seller_id": SHOP_A,
        "session_id": session_id,
        "msg_id": msg_id,
        "msg_type": "text",
        "direction": "out",
        "content": text,
        "send_time_ms": send_time_ms,
    }


def _recall(msg_id: str, recalled: str, *, session_id: str = CONV) -> dict:
    return {
        "seller_id": SHOP_A,
        "session_id": session_id,
        "msg_id": msg_id,
        "msg_type": "recall",
        "recall_msg_id": recalled,
    }


def _auto_service(intent: Intent = Intent.GREETING) -> tuple[AssistService, TaobaoMockAdapter]:
    """模式 A 全自动：店铺开了全自动总开关，路由判断层给出高置信度的可自动回复问题。"""
    unit = build_demo_unit(
        Platform.TAOBAO, route_provider=FakeDecisionProvider(router_answers(intent, 0.98))
    )
    adapter = TaobaoMockAdapter(
        unit.tools, unit.clock, unit.hasher, access="A", auto_send_enabled=True
    )
    unit.adapters.register(adapter)
    repo = unit.repos.knowledge
    shop = repo.get_shop(SHOP_A)
    assert shop is not None
    overrides = {**shop.config_overrides, "auto_reply_enabled": True}
    repo.upsert_shop(shop.model_copy(update={"config_overrides": overrides}))
    return AssistService(unit), adapter


def _robot(text: str, msg_id: str, **kw: Any) -> dict:
    raw = taobao_event(text, msg_id=msg_id, **kw)
    raw["reception"] = "robot"
    return raw


# ================================================================ 乱序（§8.3、§5.3）


def test_late_older_message_does_not_put_stale_answer_on_sidebar():
    """B（t0+1s）先到先处理，A（t0）后到：不能只回答旧问题 A、把 B 的推荐顶掉。

    迟到的 A 存进会话后，重新处理会话里最新的买家 Turn（A 与 B 相隔 1 秒，合并成一个 Turn），
    侧边栏展示的是这一 Turn 的推荐。
    """
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    t0 = _ms(service)
    b = service.handle_event(
        "B", taobao_event("有M码吗", msg_id="b", item_id="i_555", send_time_ms=t0 + 1000)
    )
    assert b is not None and b.msg_ids == ["b"]
    a = service.handle_event("B", taobao_event("什么时候发货", msg_id="a", send_time_ms=t0))
    assert a is not None
    assert a.msg_ids == ["a", "b"]  # 处理的是最新的 Turn（A 合并进去），不是单独的 A
    records = service.unit.repos.suggestions.list(SHOP_A, conversation_id=CONV)
    shown = [r for r in records if r.status == SuggestionStatus.SHOWN]
    assert shown and all(r.msg_id == "b" for r in shown)
    adapter = service.unit.adapters.get(Platform.TAOBAO, "B")
    assert adapter.sidebar(SHOP_A, CONV) == a.suggestions


# ======================================================== 全自动发送失败 → 转人工（§12.2、§5.7）


def test_auto_reply_send_not_ok_transfers_to_human(monkeypatch):
    service, adapter = _auto_service()
    monkeypatch.setattr(
        adapter, "send", lambda *a, **kw: SendResult(ok=False, error="platform 500")
    )
    result = service.handle_event("A", _robot("在吗", "r1"))
    assert result is not None and result.mode == Mode.AUTO
    assert result.transfer_to_human is True
    [transfer] = adapter.transfers(SHOP_A)
    assert transfer.conversation_id == CONV
    # 没发出去：推荐仍是 SHOWN，留给接手的客服
    record = service.unit.repos.suggestions.get(SHOP_A, result.suggestions[0].suggestion_id)
    assert record is not None and record.status == SuggestionStatus.SHOWN


def test_auto_reply_send_exception_transfers_and_does_not_propagate(monkeypatch):
    service, adapter = _auto_service()

    def boom(*a: Any, **kw: Any) -> SendResult:
        raise RuntimeError("平台接口超时")

    monkeypatch.setattr(adapter, "send", boom)
    result = service.handle_event("A", _robot("在吗", "r2"))
    assert result is not None and result.transfer_to_human is True
    assert len(adapter.transfers(SHOP_A)) == 1


def test_failed_transfer_is_logged_as_error(monkeypatch, caplog):
    service, adapter = _auto_service()
    monkeypatch.setattr(adapter, "send", lambda *a, **kw: SendResult(ok=False, error="x"))
    monkeypatch.setattr(
        adapter, "transfer_to_human", lambda *a, **kw: SendResult(ok=False, error="y")
    )
    with caplog.at_level(logging.ERROR, logger="rodit.app.service"):
        result = service.handle_event("A", _robot("在吗", "r3"))
    assert result is not None
    assert any(r.levelno >= logging.ERROR and "转人工" in r.getMessage() for r in caplog.records)

    def raises(*a: Any, **kw: Any) -> SendResult:
        raise RuntimeError("转人工接口故障")

    monkeypatch.setattr(adapter, "transfer_to_human", raises)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="rodit.app.service"):
        result = service.handle_event("A", _robot("在吗", "r4", session_id="conv_2"))
    assert result is not None  # 转人工失败不抛给平台（否则重推会被去重丢掉）
    assert any(r.levelno >= logging.ERROR and "转人工" in r.getMessage() for r in caplog.records)


def test_auto_message_with_knowledge_outage_transfers_to_human(monkeypatch):
    """全自动消息、店铺读不出来（知识库故障）：不能既不回复也不转人工（§12.2）。"""
    from rodit.core.errors import KnowledgeUnavailable

    service, adapter = _auto_service()

    def down(shop_id: str) -> Any:
        raise KnowledgeUnavailable("knowledge down")

    monkeypatch.setattr(service.unit.repos.knowledge, "get_shop", down)
    result = service.handle_event("A", _robot("在吗", "r5"))
    assert result is not None and result.transfer_to_human is True
    assert len(adapter.transfers(SHOP_A)) == 1


# ======================================================== 客服回声、撤回写进会话（§5.3、§8.2）


def test_agent_echo_reaches_conversation_history():
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    t0 = _ms(service)
    service.handle_event(
        "B", taobao_event("有M码吗", msg_id="e1", item_id="i_555", send_time_ms=t0)
    )
    assert service.handle_event("B", _echo("有的亲，M码有货", "e2", t0 + 1000)) is None
    result = service.handle_event(
        "B", taobao_event("那发什么快递", msg_id="e3", send_time_ms=t0 + 1800)
    )
    assert result is not None
    assert result.msg_ids == ["e3"]  # 客服回复隔开了两条买家消息，不合并
    msgs = service.unit.repos.conversations.recent_messages(SHOP_A, CONV)
    assert [(m.msg_id, m.role) for m in msgs] == [
        ("e1", Role.BUYER),
        ("e2", Role.AGENT),
        ("e3", Role.BUYER),
    ]
    # 平台重推同一条回声：不重复存
    service.handle_event("B", _echo("有的亲，M码有货", "e2", t0 + 1000))
    assert len(service.unit.repos.conversations.recent_messages(SHOP_A, CONV)) == 3


def test_echo_of_reply_already_mirrored_by_feedback_is_not_stored_twice():
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="f1"))
    card = result.suggestions[0]
    service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    service.handle_event("B", _echo(card.text, "f2", _ms(service, 5)))
    msgs = service.unit.repos.conversations.recent_messages(SHOP_A, CONV)
    assert [m.role for m in msgs] == [Role.BUYER, Role.AGENT]


def test_feedback_after_echo_does_not_store_reply_twice():
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="g1"))
    card = result.suggestions[0]
    service.handle_event("B", _echo(card.text, "g2", _ms(service, 5)))
    service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    msgs = service.unit.repos.conversations.recent_messages(SHOP_A, CONV)
    assert [(m.msg_id, m.role) for m in msgs] == [("g1", Role.BUYER), ("g2", Role.AGENT)]


def test_echo_of_auto_reply_is_not_stored_twice():
    service, adapter = _auto_service()
    result = service.handle_event("A", _robot("在吗", "h1"))
    assert result is not None and result.auto_reply is not None
    [sent] = adapter.sent_messages(SHOP_A)
    service.handle_event("A", _echo(sent.text, sent.platform_msg_id, _ms(service, 1)))
    msgs = service.unit.repos.conversations.recent_messages(SHOP_A, CONV)
    assert [m.role for m in msgs] == [Role.BUYER, Role.BOT]


def test_recalled_buyer_message_leaves_history_and_its_text_is_not_kept():
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    t0 = _ms(service)
    service.handle_event(
        "B", taobao_event("我的手机号13812345678，这件有M码吗", msg_id="k1", send_time_ms=t0)
    )
    assert service.handle_event("B", _recall("k1_recall", "k1")) is None
    repo = service.unit.repos.conversations
    assert repo.recent_messages(SHOP_A, CONV) == []
    result = service.handle_event(
        "B", taobao_event("那黑色的吧，什么时候发货", msg_id="k3", send_time_ms=t0 + 60_000)
    )
    assert result is not None and result.msg_ids == ["k3"]
    texts = [m.text for m in repo.recent_messages(SHOP_A, CONV)]
    assert texts == ["那黑色的吧，什么时候发货"]
    # 平台重推被撤回的那条：仍按 msg_id 去重，不会再存回来
    again = taobao_event("我的手机号13812345678，这件有M码吗", msg_id="k1", send_time_ms=t0)
    assert service.handle_event("B", again) is None
    assert "13812345678" not in "".join(m.text or "" for m in repo.recent_messages(SHOP_A, CONV))


# ================================================================ 会话锁有界（资源）


def test_conversation_locks_do_not_accumulate():
    service = AssistService(build_demo_unit(Platform.PDD))
    for i in range(50):
        service.ask_standalone(shop_id=SHOP_C, session_id=f"web_{i}", question="什么时候发货")
    assert len(service._locks) == 0


# ================================================================ 取消订购与在途处理（§8.4）


def test_revoke_waits_for_in_flight_message_and_leaves_no_rows():
    """处理到一半时商家取消订购：在途处理写下的记录也要删掉（§8.4 按约定删除数据）。"""
    unit = build_demo_unit(Platform.TAOBAO)
    service = AssistService(unit)
    inflight = threading.Event()
    revoke_started = threading.Event()
    revoke_done = threading.Event()
    original = unit.engine._retriever.retrieve

    def slow_retrieve(ctx: Any, route: Any) -> Any:
        inflight.set()
        assert revoke_started.wait(5)
        revoke_done.wait(0.3)  # 修复后取消订购要等在途处理做完：这里等不到，超时后继续
        return original(ctx, route)

    unit.engine._retriever.retrieve = slow_retrieve
    errors: list[BaseException] = []

    def process() -> None:
        try:
            service.handle_event("B", taobao_event("这件是纯棉的吗", msg_id="p1", item_id="i_555"))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def revoke() -> None:
        assert inflight.wait(5)
        revoke_started.set()
        service.revoke_shop(SHOP_A)
        revoke_done.set()

    threads = [threading.Thread(target=process), threading.Thread(target=revoke)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors
    assert revoke_done.is_set()
    repos = unit.repos
    assert not unit.auth.is_authorized(SHOP_A)
    assert repos.suggestions.list(SHOP_A) == []
    assert repos.audit.list_events(SHOP_A) == []
    assert repos.audit.list_decisions(SHOP_A) == []
    assert repos.feedback.list_gaps(SHOP_A) == []
    assert repos.conversations.recent_messages(SHOP_A, CONV) == []
    adapter = unit.adapters.get(Platform.TAOBAO, "B")
    assert adapter.get_context(SHOP_A, CONV).recent_messages == []


def test_revoked_shop_is_not_processed_after_revoke():
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    service.revoke_shop(SHOP_A)
    from rodit.core.errors import AuthExpired

    with pytest.raises(AuthExpired):
        service.handle_event("B", taobao_event("在吗", msg_id="q1"))
