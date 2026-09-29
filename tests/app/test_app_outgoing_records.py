"""客服 / 机器人发出的回复写回会话（§5.3 最近几轮对话）：同一次发送的两份记录、撤回、并发、失败。

同一次发送最多有两份记录：反馈 / 全自动写回的（msg_id 为 agent:{推荐 id} / bot:{推荐 id}）和平台
回声（平台消息 ID）。只有这两份之间去重：两条平台消息永远是两条，中间隔着买家消息的也不是同一次
发送。回声比写回的晚到时，会话里那条换成回声（平台消息 ID），之后撤回它能找到。
撤回写不进去时抛给平台重推，重推照常生效（§12.3 最小化：撤回的原文不留）。
全部合成数据，不联网。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest

from rodit.adapters import TaobaoMockAdapter
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.core.audit import AuditAction
from rodit.core.enums import Intent, Mode, Platform, Role, SuggestionStatus
from rodit.core.errors import StorageUnavailable
from rodit.core.models import SendResult
from rodit.core.pii import mask_pii
from rodit.feedback import agent_replies_by_msg
from rodit.testing.fakes import FakeDecisionProvider, router_answers
from rodit.testing.fixtures import AGENT, SHOP_A

CONV = "conv_demo"


def _service() -> AssistService:
    return AssistService(build_demo_unit(Platform.TAOBAO))


def _ms(service: AssistService, offset_s: float = 0) -> int:
    return int((service.unit.clock.now().timestamp() + offset_s) * 1000)


def _echo(text: str, msg_id: str, send_time_ms: int) -> dict:
    """客服 / 机器人发出的消息回声（淘宝 direction=out）。"""
    return {
        "seller_id": SHOP_A,
        "session_id": CONV,
        "msg_id": msg_id,
        "msg_type": "text",
        "direction": "out",
        "content": text,
        "send_time_ms": send_time_ms,
    }


def _recall(msg_id: str, recalled: str) -> dict:
    return {
        "seller_id": SHOP_A,
        "session_id": CONV,
        "msg_id": msg_id,
        "msg_type": "recall",
        "recall_msg_id": recalled,
    }


def _stored(service: AssistService) -> list[tuple[str, Role]]:
    msgs = service.unit.repos.conversations.recent_messages(SHOP_A, CONV)
    return [(m.msg_id, m.role) for m in msgs]


def _auto_service() -> tuple[AssistService, TaobaoMockAdapter]:
    """模式 A 全自动：店铺开了全自动总开关，路由判断层给出高置信度的可自动回复问题。"""
    unit = build_demo_unit(
        Platform.TAOBAO, route_provider=FakeDecisionProvider(router_answers(Intent.GREETING, 0.98))
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


# ================================================================ 撤回可以重试


def test_recall_that_failed_to_store_is_applied_when_the_platform_re_pushes_it(monkeypatch):
    """撤回写会话失败一次：不能吞掉（平台当成已送达，适配器也记成见过），要让平台重推并生效。"""
    service = _service()
    repo = service.unit.repos.conversations
    t0 = _ms(service)
    service.handle_event(
        "B", taobao_event("我的手机号13812345678，这件有M码吗", msg_id="k1", send_time_ms=t0)
    )
    real = repo.recall_message
    left = {"failures": 1}

    def flaky(*args: Any, **kw: Any) -> bool:
        if left["failures"]:
            left["failures"] -= 1
            raise StorageUnavailable("会话库暂时不可用")
        return real(*args, **kw)

    monkeypatch.setattr(repo, "recall_message", flaky)
    with pytest.raises(StorageUnavailable):
        service.handle_event("B", _recall("k1_recall", "k1"))
    assert service.handle_event("B", _recall("k1_recall", "k1")) is None  # 平台重推同一个撤回
    assert repo.recent_messages(SHOP_A, CONV) == []
    result = service.handle_event(
        "B", taobao_event("什么时候发货", msg_id="k3", send_time_ms=t0 + 60_000)
    )
    assert result is not None and result.msg_ids == ["k3"]
    assert "13812345678" not in "".join(m.text or "" for m in repo.recent_messages(SHOP_A, CONV))


# ================================================================ 回声晚到：换成平台消息 ID


def test_recalling_an_adopted_reply_after_its_echo_removes_it():
    """客服点发送（先写回 agent:{id}），回声带平台消息 ID 晚到，客服再撤回：
    回复不能留在上下文里。"""
    service = _service()
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="q1"))
    assert result is not None
    card = result.suggestions[0]
    service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    service.handle_event("B", _echo(card.text, "plat_out_1", _ms(service, 1)))
    assert _stored(service) == [("q1", Role.BUYER), ("plat_out_1", Role.AGENT)]
    assert service.handle_event("B", _recall("recall_1", "plat_out_1")) is None
    assert _stored(service) == [("q1", Role.BUYER)]


def test_bot_reply_stored_without_platform_id_is_taken_over_by_its_echo(monkeypatch):
    """全自动发送没返回平台消息 ID（存成 bot:{id}）：回声到了换成平台消息 ID，仍记为机器人；
    撤回这条回声能从会话里去掉。"""
    service, adapter = _auto_service()
    monkeypatch.setattr(adapter, "send", lambda *a, **kw: SendResult(ok=True))
    result = service.handle_event("A", _robot("在吗", "h1"))
    assert result is not None and result.auto_reply is not None
    [(buyer_id, _), (bot_id, bot_role)] = _stored(service)
    assert buyer_id == "h1" and bot_id.startswith("bot:") and bot_role == Role.BOT
    service.handle_event("A", _echo(result.auto_reply.text, "p_bot_1", _ms(service, 1)))
    assert _stored(service) == [("h1", Role.BUYER), ("p_bot_1", Role.BOT)]
    service.handle_event("A", _recall("rc_1", "p_bot_1"))
    assert _stored(service) == [("h1", Role.BUYER)]


# ================================================================ 不是同一次发送的不去重


def test_same_words_replied_to_two_buyer_messages_are_both_kept():
    """客服两次回同一句话，中间隔着买家的新问题：第二条是新回复，不能当成重复丢掉
    （否则下一条买家消息会和已经回答过的那条合并成一个 Turn，§5.3）。"""
    service = _service()
    t0 = _ms(service)
    service.handle_event(
        "B", taobao_event("有M码吗", msg_id="m1", item_id="i_555", send_time_ms=t0)
    )
    service.handle_event("B", _echo("好的亲，稍等", "e1", t0 + 1000))
    service.handle_event(
        "B", taobao_event("L码呢", msg_id="m2", item_id="i_555", send_time_ms=t0 + 2000)
    )
    service.handle_event("B", _echo("好的亲，稍等", "e2", t0 + 3000))
    result = service.handle_event(
        "B", taobao_event("发什么快递", msg_id="m3", send_time_ms=t0 + 4000)
    )
    assert _stored(service) == [
        ("m1", Role.BUYER),
        ("e1", Role.AGENT),
        ("m2", Role.BUYER),
        ("e2", Role.AGENT),
        ("m3", Role.BUYER),
    ]
    assert result is not None and result.msg_ids == ["m3"]


def test_repeated_short_reply_counts_as_the_answer_to_each_question():
    """ "有的亲"回答了两个问题：影子对比（§13.4）按买家消息找客服回复，两个问题都有回复。"""
    service = _service()
    t0 = _ms(service)
    service.handle_event(
        "B", taobao_event("M码有货吗", msg_id="b1", item_id="i_555", send_time_ms=t0)
    )
    service.handle_event("B", _echo("有的亲", "e1", t0 + 10_000))
    service.handle_event("B", taobao_event("L码呢", msg_id="b2", send_time_ms=t0 + 30_000))
    service.handle_event("B", _echo("有的亲", "e2", t0 + 40_000))
    msgs = service.unit.repos.conversations.recent_messages(SHOP_A, CONV)
    replies = agent_replies_by_msg(msgs)
    assert replies["b1"] == ["有的亲"] and replies["b2"] == ["有的亲"]


def test_two_platform_messages_with_the_same_text_are_two_messages():
    """两条平台消息（不同的平台消息 ID）永远是两条，哪怕文字相同、挨得很近。"""
    service = _service()
    t0 = _ms(service)
    service.handle_event("B", taobao_event("在吗", msg_id="n1", send_time_ms=t0))
    service.handle_event("B", _echo("在的亲", "n2", t0 + 1000))
    service.handle_event("B", _echo("在的亲", "n3", t0 + 1500))
    assert _stored(service) == [("n1", Role.BUYER), ("n2", Role.AGENT), ("n3", Role.AGENT)]


def test_same_adopted_reply_for_two_questions_is_mirrored_twice():
    """两次采纳同一句推荐（两个买家问题）：两次写回都存（两份都是写回的记录，不是同一次发送）。"""
    service = _service()
    first = service.handle_event("B", taobao_event("什么时候发货", msg_id="s1"))
    assert first is not None
    service.feedback_sent(SHOP_A, first.suggestions[0].suggestion_id, agent_id=AGENT)
    service.unit.clock.advance(seconds=30)
    second = service.handle_event(
        "B", taobao_event("什么时候发货呀", msg_id="s2", send_time_ms=_ms(service))
    )
    assert second is not None
    assert second.suggestions[0].text == first.suggestions[0].text
    service.unit.clock.advance(seconds=5)
    service.feedback_sent(SHOP_A, second.suggestions[0].suggestion_id, agent_id=AGENT)
    roles = [role for _, role in _stored(service)]
    assert roles == [Role.BUYER, Role.AGENT, Role.BUYER, Role.AGENT]


def test_echo_does_not_take_over_a_record_across_a_buyer_message():
    """写回的记录和回声之间隔着买家消息：不是同一次发送，两条都留。"""
    service = _service()
    first = service.handle_event("B", taobao_event("什么时候发货", msg_id="x1"))
    assert first is not None
    card = first.suggestions[0]
    service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    service.handle_event("B", taobao_event("好的", msg_id="x2", send_time_ms=_ms(service, 20)))
    service.handle_event("B", _echo(card.text, "x3", _ms(service, 30)))
    assert _stored(service) == [
        ("x1", Role.BUYER),
        (f"agent:{card.suggestion_id}", Role.AGENT),
        ("x2", Role.BUYER),
        ("x3", Role.AGENT),
    ]


# ================================================================ 编辑后的回复带个人信息


SENT_WITH_PII = "亲，退货请寄到浙江省杭州市余杭区文一西路1001号3幢，收件人小王，电话13900001111"


@pytest.mark.parametrize("order", ["feedback_then_echo", "echo_then_feedback"])
def test_edited_reply_with_address_and_phone_is_stored_once(order):
    """反馈记录里的编辑文字脱敏过，回声是原文：仍是同一次发送，只存一份（用平台消息 ID）。"""
    service = _service()
    result = service.handle_event("B", taobao_event("退货寄到哪里", msg_id="a1"))
    assert result is not None
    card = result.suggestions[0]

    def feedback() -> None:
        service.feedback_edited(
            SHOP_A, card.suggestion_id, agent_id=AGENT, edited_text=SENT_WITH_PII
        )

    def echo() -> None:
        service.handle_event("B", _echo(SENT_WITH_PII, "a2", _ms(service, 3)))

    for step in (feedback, echo) if order == "feedback_then_echo" else (echo, feedback):
        step()
    assert _stored(service) == [("a1", Role.BUYER), ("a2", Role.AGENT)]


def test_echo_is_matched_with_a_mirrored_record_that_was_masked():
    """写回的记录做过个人信息最小化（旧数据），回声是原文：按脱敏后的形式比较，仍是同一次发送。"""
    service = _service()
    service.handle_event("B", taobao_event("仓库电话多少", msg_id="z1"))
    sent = "亲，仓库电话13900001111，工作日都有人接哦"
    masked = mask_pii(sent)
    assert masked != sent
    service.unit.repos.conversations.add_outgoing(
        shop_id=SHOP_A,
        conversation_id=CONV,
        msg_id="agent:99",
        role=Role.AGENT,
        text=masked,
        at=service.unit.clock.now(),
    )
    service.handle_event("B", _echo(sent, "z2", _ms(service, 2)))
    assert _stored(service) == [("z1", Role.BUYER), ("z2", Role.AGENT)]


# ================================================================ 反馈写回和回声同时到


def _race(
    service: AssistService,
    first: Callable[[], Any],
    second: Callable[[], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> list[BaseException]:
    """first 读完会话、还没写入时，second 也来读（修复前两边都读到"还没有"，各存一份）。

    修复后 second 在等会话锁：first 等不到它（最多 0.5 秒），照常写入后放锁，second 再读。
    """
    repo = service.unit.repos.conversations
    real = repo.recent_messages
    first_read = threading.Event()
    second_read = threading.Event()
    reads = {"n": 0}
    guard = threading.Lock()

    def recent(*args: Any, **kw: Any) -> Any:
        out = real(*args, **kw)
        with guard:
            n = reads["n"]
            reads["n"] += 1
        if n == 0:
            first_read.set()
            second_read.wait(0.5)
        else:
            second_read.set()
        return out

    monkeypatch.setattr(repo, "recent_messages", recent)
    errors: list[BaseException] = []

    def run(op: Callable[[], Any]) -> None:
        try:
            op()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t1 = threading.Thread(target=run, args=(first,))
    t1.start()
    assert first_read.wait(5)
    t2 = threading.Thread(target=run, args=(second,))
    t2.start()
    t1.join(10)
    t2.join(10)
    monkeypatch.setattr(repo, "recent_messages", real)
    return errors


@pytest.mark.parametrize("feedback_first", [True, False])
def test_feedback_and_echo_arriving_together_store_the_reply_once(monkeypatch, feedback_first):
    """一次点击同时触发插件的 feedback_sent 和平台回声：检查和写入要按会话串行。"""
    service = _service()
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="f1"))
    assert result is not None
    card = result.suggestions[0]

    def feedback() -> None:
        service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)

    def echo() -> None:
        service.handle_event("B", _echo(card.text, "f2", _ms(service, 1)))

    ops = (feedback, echo) if feedback_first else (echo, feedback)
    assert _race(service, *ops, monkeypatch=monkeypatch) == []
    assert _stored(service) == [("f1", Role.BUYER), ("f2", Role.AGENT)]


# ================================================================ 写回失败不影响已完成的反馈


@pytest.mark.parametrize("edited", [False, True])
def test_feedback_is_reported_done_when_only_the_mirror_fails(monkeypatch, caplog, edited):
    """反馈已经记下（状态、标注、审计），只是写回会话失败：不能报失败——重试会被拒（409）。"""
    service = _service()
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="w1"))
    assert result is not None
    card = result.suggestions[0]
    repo = service.unit.repos.conversations

    def down(*a: Any, **kw: Any) -> Any:
        raise StorageUnavailable("会话库暂时不可用 13812345678")

    monkeypatch.setattr(repo, "recent_messages", down)
    monkeypatch.setattr(repo, "add_outgoing", down)
    with caplog.at_level(logging.WARNING, logger="rodit.app.service"):
        if edited:
            outcome = service.feedback_edited(
                SHOP_A, card.suggestion_id, agent_id=AGENT, edited_text="亲，今天就发哦"
            )
        else:
            outcome = service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    expected = SuggestionStatus.EDITED_SENT if edited else SuggestionStatus.SENT
    assert outcome.record.status == expected
    record = service.unit.repos.suggestions.get(SHOP_A, card.suggestion_id)
    assert record is not None and record.status == expected
    action = AuditAction.FEEDBACK_EDITED if edited else AuditAction.FEEDBACK_SENT
    assert service.unit.repos.audit.list_events(SHOP_A, action=action)
    assert "StorageUnavailable" in caplog.text and "13812345678" not in caplog.text


# ================================================================ 全自动：展示推荐失败


def test_auto_mode_show_suggestions_failure_still_answers_the_buyer(monkeypatch, caplog):
    """全自动模式下推送工作台失败：照常发送自动回复（发送失败再转人工），不能抛给平台——
    重推会被去重，买家既没有回复也没人接（§12.2）。"""
    service, adapter = _auto_service()

    def boom(*a: Any, **kw: Any) -> None:
        raise RuntimeError("workstation push failed")

    monkeypatch.setattr(adapter, "show_suggestions", boom)
    with caplog.at_level(logging.WARNING, logger="rodit.app.service"):
        result = service.handle_event("A", _robot("在吗", "sf1"))
    assert result is not None and result.mode == Mode.AUTO
    assert len(adapter.sent_messages(SHOP_A)) == 1
    assert "RuntimeError" in caplog.text


def test_auto_mode_show_and_send_both_failing_transfers_to_human(monkeypatch):
    service, adapter = _auto_service()

    def boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("platform down")

    monkeypatch.setattr(adapter, "show_suggestions", boom)
    monkeypatch.setattr(adapter, "send", boom)
    result = service.handle_event("A", _robot("在吗", "sf2"))
    assert result is not None and result.transfer_to_human is True
    assert len(adapter.transfers(SHOP_A)) == 1


# ================================================================ 问卷回答里的控制字符


def test_questionnaire_answers_are_cleaned_before_shop_config_is_written():
    """店铺配置用 model_copy 更新（不走校验）：回答里的 NUL 要先去掉，
    否则 PostgreSQL 的 JSONB 写不进去，而内存实现照存（两种实现结果不一致）。"""
    service = _service()
    out = service.apply_questionnaire(SHOP_A, {14: "最好\x00、第一"})
    assert out["config"] == ["banned_words"]
    shop = service.unit.repos.knowledge.get_shop(SHOP_A)
    assert shop is not None
    words = shop.config_overrides["banned_words"]
    assert words and all("\x00" not in w for w in words)
    assert "最好" in words


# ---------------------------------------------------------------- 复审第三轮


def test_buyer_message_sent_after_the_agent_does_not_split_one_send():
    """写回的时间是服务端收到反馈的时间（晚于实际发送）：买家在客服发送之后、反馈到达之前发的
    消息不能把同一次发送拆成两份；之后撤回回声，这条回复就从会话里去掉。"""
    service = _service()
    clock = service.unit.clock
    base = clock.now()
    t0 = _ms(service)
    card = service.handle_event("B", taobao_event("什么时候发货", msg_id="b1", send_time_ms=t0))
    card = card.suggestions[0]
    clock.set(base + timedelta(seconds=5.4))  # 反馈到达（客服 t0+5.0s 点的发送）
    service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    clock.set(base + timedelta(seconds=5.6))  # 买家 t0+5.2s 发的消息到达
    service.handle_event("B", taobao_event("有人吗", msg_id="b2", send_time_ms=t0 + 5200))
    clock.set(base + timedelta(seconds=5.8))  # 回声到达（平台发送时间 t0+5.0s）
    service.handle_event("B", _echo(card.text, "e1", t0 + 5000))
    msgs = service.unit.repos.conversations.recent_messages(SHOP_A, CONV)
    assert sum(1 for m in msgs if m.text == card.text) == 1
    assert [m.msg_id for m in msgs] == ["b1", "e1", "b2"]
    service.handle_event("B", _recall("r1", "e1"))
    assert all(
        m.text != card.text for m in service.unit.repos.conversations.recent_messages(SHOP_A, CONV)
    )


def test_echo_take_over_failure_is_retried_so_a_later_recall_works():
    """回声接替写回记录时写库失败：抛给平台重推，重推照常接替；之后按平台 msg_id 撤回能找到。"""
    service = _service()
    t0 = _ms(service)
    card = service.handle_event("B", taobao_event("什么时候发货", msg_id="q1", send_time_ms=t0))
    card = card.suggestions[0]
    service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    repo = service.unit.repos.conversations
    original = repo.recall_message
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any):
        calls["n"] += 1
        if calls["n"] == 1:
            raise StorageUnavailable("会话库暂时不可用")
        return original(*args, **kwargs)

    repo.recall_message = flaky  # type: ignore[method-assign]
    echo = _echo(card.text, "plat_out_1", t0 + 1000)
    with pytest.raises(StorageUnavailable):
        service.handle_event("B", echo)
    service.handle_event("B", echo)  # 平台重推
    assert [m for m, _ in _stored(service)] == ["q1", "plat_out_1"]
    service.handle_event("B", _recall("r1", "plat_out_1"))
    assert [m for m, _ in _stored(service)] == ["q1"]
