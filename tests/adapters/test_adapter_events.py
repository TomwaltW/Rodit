"""on_event：客服 / 机器人回声和撤回也交给应用层（方案 §5.3 最近几轮对话、§8.2 getContext）。

on_message 只返回买家消息（行为不变）；on_event 额外返回回声（StoredMessage，role=agent）和
撤回（MessageRecall），应用层据此把客服手打的回复写进会话、把撤回的消息从会话里去掉。
"""

from __future__ import annotations

import pytest

from rodit.adapters import DouyinMockAdapter, NullToolGateway, PddMockAdapter, TaobaoMockAdapter
from rodit.core.enums import Platform, Role
from rodit.core.models import StandardMessage, StoredMessage
from rodit.core.ports import MessageRecall, PlatformAdapter
from rodit.testing.fakes import InMemoryToolGateway
from rodit.testing.fixtures import CONV, DEFAULT_NOW, SHOP_A, SHOP_C, SHOP_D


@pytest.fixture
def taobao(world, clock, hasher) -> TaobaoMockAdapter:
    return TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="B")


def test_on_event_is_part_of_the_adapter_protocol(taobao):
    assert isinstance(taobao, PlatformAdapter)
    assert callable(taobao.on_event)


def test_taobao_echo_and_recall_are_surfaced(taobao, tb):
    buyer = taobao.on_event(tb(msg_id="b1", content="有M码吗"))
    assert isinstance(buyer, StandardMessage) and buyer.msg_id == "b1"
    echo = taobao.on_event(
        tb(msg_id="a1", direction="out", content=" 有的亲 ", buyer_open_uid=None)
    )
    assert isinstance(echo, StoredMessage)
    assert (echo.shop_id, echo.conversation_id, echo.msg_id) == (SHOP_A, CONV, "a1")
    assert echo.role == Role.AGENT and echo.text == "有的亲"
    assert echo.created_at == DEFAULT_NOW  # 用事件时间
    # 重复的回声：None（按 msg_id 去重）
    assert taobao.on_event(tb(msg_id="a1", direction="out", content="有的亲")) is None
    recall = taobao.on_event(tb(msg_id="r1", msg_type="recall", recall_msg_id="b1"))
    assert recall == MessageRecall(
        shop_id=SHOP_A, conversation_id=CONV, msg_id="r1", recalled_msg_id="b1"
    )
    # 平台重推同一个撤回：照样交给应用层（撤回是幂等的；应用层上次没写进会话时靠重推补上）
    assert taobao.on_event(tb(msg_id="r1", msg_type="recall", recall_msg_id="b1")) == recall
    assert [m.msg_id for m in taobao.get_context(SHOP_A, CONV).recent_messages] == ["a1"]


def test_on_message_still_returns_only_buyer_messages(taobao, tb):
    assert taobao.on_message(tb(msg_id="b1")) is not None
    assert taobao.on_message(tb(msg_id="a1", direction="out", content="亲")) is None
    assert taobao.on_message(tb(msg_id="r1", msg_type="recall", recall_msg_id="b1")) is None


def test_echo_with_empty_text_or_recall_without_target_is_dropped(taobao, tb):
    taobao.on_event(tb(msg_id="b1"))
    assert taobao.on_event(tb(msg_id="a1", direction="out", content="   ")) is None
    assert taobao.on_event(tb(msg_id="r0", msg_type="recall", recall_msg_id=None)) is None


def test_echo_for_conversation_unknown_locally_is_surfaced_but_not_recorded(taobao, tb):
    """适配器本地没有这个会话（重启、被淘汰）：本地不记，交给应用层——会话仓储里有这个会话
    才会存（ConversationRepository.add_outgoing 对不存在的会话返回 False）。"""
    echo = taobao.on_event(tb(msg_id="a0", direction="out", content="亲~", session_id="c_x"))
    assert isinstance(echo, StoredMessage) and echo.conversation_id == "c_x"
    assert taobao.get_context(SHOP_A, "c_x").recent_messages == []


def test_recall_of_unknown_conversation_is_still_surfaced(taobao, tb):
    """适配器本地没有这个会话（重启、被淘汰）时，会话仓储里可能还有：撤回照样交给应用层。"""
    recall = taobao.on_event(tb(msg_id="r9", msg_type="recall", recall_msg_id="b_old"))
    assert isinstance(recall, MessageRecall) and recall.recalled_msg_id == "b_old"


def test_pdd_and_douyin_echoes(clock, hasher, world, pdd, dy):
    pdd_adapter = PddMockAdapter(NullToolGateway(Platform.PDD), clock, hasher, access="C")
    pdd_adapter.on_event(pdd(msg_id="g1"))
    cs = pdd_adapter.on_event(pdd(msg_id="cs", **{"from": {"role": "mall_cs", "uid": "kf_c01"}}))
    assert isinstance(cs, StoredMessage) and cs.role == Role.AGENT and cs.shop_id == SHOP_C
    rc = pdd_adapter.on_event(pdd(msg_id="rc", type=1002, recall_msg_id="g1"))
    assert isinstance(rc, MessageRecall) and rc.recalled_msg_id == "g1"

    douyin = DouyinMockAdapter(
        InMemoryToolGateway(world, Platform.DOUYIN), clock, hasher, access="B"
    )
    douyin.on_event(dy(server_message_id="d1"))
    echo = douyin.on_event(
        dy(server_message_id="d2", sender={"role": "customer_service", "uid": "kf_d01"})
    )
    assert isinstance(echo, StoredMessage) and echo.role == Role.AGENT and echo.shop_id == SHOP_D
