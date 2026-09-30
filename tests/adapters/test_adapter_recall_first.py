"""撤回先于被撤回的消息到达（§8.3 平台推送乱序、§5.3 撤回的消息不再进上下文）：接入层的本地上下文
（get_context）也不留那条消息。是否丢弃由会话仓储的撤回墓碑决定，接入层照常把消息交给应用层。"""

from __future__ import annotations

import pytest

from rodit.adapters import TaobaoMockAdapter
from rodit.core.models import StandardMessage, StoredMessage
from rodit.core.ports import MessageRecall
from rodit.testing.fakes import InMemoryToolGateway
from rodit.testing.fixtures import CONV, SHOP_A


@pytest.fixture
def taobao(world, clock, hasher) -> TaobaoMockAdapter:
    return TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, hasher, access="B", dedup_capacity=3
    )


def _ids(adapter: TaobaoMockAdapter, conversation_id: str = CONV) -> list[str]:
    return [m.msg_id for m in adapter.get_context(SHOP_A, conversation_id).recent_messages]


def test_buyer_message_recalled_before_it_arrives_stays_out_of_local_context(taobao, tb):
    taobao.on_event(tb(msg_id="b0", content="在吗", item_id=None))
    recall = taobao.on_event(tb(msg_id="r1", msg_type="recall", recall_msg_id="b1"))
    assert isinstance(recall, MessageRecall) and recall.recalled_msg_id == "b1"

    late = taobao.on_event(tb(msg_id="b1", content="我的手机号13812345678", item_id="i_777"))

    assert isinstance(late, StandardMessage) and late.msg_id == "b1"  # 照常交给应用层
    assert _ids(taobao) == ["b0"]
    assert taobao.get_context(SHOP_A, CONV).item_card is None  # 它带的商品卡片也不算
    assert taobao.on_event(tb(msg_id="b1", content="我的手机号13812345678")) is None  # 重推去重


def test_agent_echo_recalled_before_it_arrives_stays_out_of_local_context(taobao, tb):
    taobao.on_event(tb(msg_id="b0", content="在吗"))
    taobao.on_event(tb(msg_id="r1", msg_type="recall", recall_msg_id="a1"))
    echo = taobao.on_event(tb(msg_id="a1", direction="out", content="有的亲"))
    assert isinstance(echo, StoredMessage)
    assert _ids(taobao) == ["b0"]


def test_recall_marks_are_per_conversation_bounded_and_purged(taobao, tb):
    taobao.on_event(tb(msg_id="r1", msg_type="recall", recall_msg_id="x1"))
    taobao.on_event(tb(msg_id="x1", content="在吗", session_id="c_other"))  # 别的会话：照常
    assert _ids(taobao, "c_other") == ["x1"]
    for i in range(5):  # 每个店铺最多记 dedup_capacity（3）条，最早的淘汰
        taobao.on_event(tb(msg_id=f"r_{i}", msg_type="recall", recall_msg_id=f"y{i}"))
    taobao.on_event(tb(msg_id="y0", content="在吗"))
    taobao.on_event(tb(msg_id="y4", content="在吗"))
    assert _ids(taobao) == ["y0"]
    taobao.on_event(tb(msg_id="r_z", msg_type="recall", recall_msg_id="z1"))
    taobao.purge_shop(SHOP_A)
    taobao.on_event(tb(msg_id="z1", content="在吗"))
    assert _ids(taobao) == ["z1"]
