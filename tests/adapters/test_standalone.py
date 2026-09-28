"""模式 C 独立话术助手（方案 §8.1）：客服粘贴问题，没有平台数据，合成的会话级 buyer_ref。"""

from __future__ import annotations

from datetime import datetime

import pytest

from rodit.adapters import NullToolGateway, StandaloneAssistAdapter
from rodit.core.enums import AfterSaleStatus, Confidence, DraftKind, Intent, Mode, Platform
from rodit.core.errors import ActionNotAllowed
from rodit.core.models import ItemRef, SuggestionCard
from rodit.core.pii import PHONE_MASK
from rodit.testing.fixtures import BUYER, BUYER_PDD, SHOP_A, SHOP_B, SHOP_C


@pytest.fixture
def sa(clock, hasher) -> StandaloneAssistAdapter:
    return StandaloneAssistAdapter(Platform.PDD, clock, hasher)


def _raw(**over):
    raw = {"shop_id": SHOP_C, "session_id": "web_1", "question": "打底衫是纯棉的吗"}
    raw.update(over)
    return raw


def test_pasted_question_to_standard_message(sa, hasher, clock):
    msg = sa.on_message(_raw(item_id="i_c1", agent_id="kf_c01", request_id="req_1"))
    assert msg is not None
    assert msg.platform == Platform.PDD and msg.shop_id == SHOP_C
    assert msg.conversation_id == "standalone:web_1"
    assert msg.msg_id == "standalone:web_1:req:req_1"
    assert msg.text == "打底衫是纯棉的吗"
    assert msg.item_card == ItemRef(item_id="i_c1")
    assert msg.order_card is None
    assert msg.agent_id == "kf_c01"
    assert msg.mode == Mode.ASSIST and msg.received_at == clock.now()
    assert msg.extension == {"source": "standalone"}
    # 没有平台买家 ID：buyer_ref 由网页会话合成，不会撞上任何真实买家
    assert msg.buyer_ref == hasher("pdd", "standalone:web_1")
    assert msg.buyer_ref not in {BUYER, BUYER_PDD}


def test_buyer_ref_is_stable_within_session_and_differs_across_sessions(sa):
    a1 = sa.on_message(_raw(question="在吗"))
    a2 = sa.on_message(_raw(question="有 XL 吗"))
    b1 = sa.on_message(_raw(session_id="web_2", question="在吗"))
    assert a1.buyer_ref == a2.buyer_ref != b1.buyer_ref
    assert a1.msg_id != a2.msg_id  # 没有 request_id 时每次提交都是新消息
    assert [m.msg_id for m in sa.get_context(SHOP_C, "standalone:web_1").recent_messages] == [
        a1.msg_id,
        a2.msg_id,
    ]


def test_pasted_personal_info_is_masked_at_entry(sa):
    msg = sa.on_message(_raw(question="收件电话 13812345678 能改成下午送吗"))
    assert "13812345678" not in msg.model_dump_json()
    assert PHONE_MASK in msg.text
    stored = sa.get_context(SHOP_C, "standalone:web_1").recent_messages[0]
    assert "13812345678" not in (stored.text or "")


def test_text_without_personal_info_keeps_original_punctuation(sa):
    msg = sa.on_message(_raw(question="这个是纯棉的吗？会起球吗！"))
    assert msg.text == "这个是纯棉的吗？会起球吗！"


def test_no_order_data_in_mode_c(sa):
    """模式 C 不接收订单号（查不了），工具也全部查不到（§8.1 没有订单数据）。"""
    msg = sa.on_message(_raw(order_id="o_1001", question="我的快递到哪了"))
    assert msg.order_card is None
    tools = sa.tools()
    assert isinstance(tools, NullToolGateway) and tools.platform == Platform.PDD
    assert tools.get_order(SHOP_C, "o_1001") is None
    assert tools.list_recent_orders(SHOP_C, msg.buyer_ref) == []
    assert tools.get_logistics(SHOP_C, "o_1001") is None
    assert tools.get_aftersale(SHOP_C, "as_1") is None
    assert tools.list_aftersales(SHOP_C, status=AfterSaleStatus.PENDING_REVIEW) == []
    assert tools.get_stock(SHOP_C, "i_c1") is None
    assert tools.get_price(SHOP_C, "i_c1") is None


def test_request_id_dedups_retries(sa):
    assert sa.on_message(_raw(request_id="r1")) is not None
    assert sa.on_message(_raw(request_id="r1")) is None
    assert sa.on_message(_raw(request_id="r1", shop_id=SHOP_B)) is not None  # 另一个店铺
    assert sa.on_message(_raw(request_id="r1", session_id="web_2")) is not None  # 另一个会话


def test_generated_msg_ids_survive_restart(clock, hasher):
    """没有 request_id 时，进程重启（序号从 1 重来）后的消息 ID 也不会和之前的撞上。"""
    before = StandaloneAssistAdapter(Platform.PDD, clock, hasher).on_message(_raw())
    clock.advance(seconds=1)
    after = StandaloneAssistAdapter(Platform.PDD, clock, hasher).on_message(_raw())
    assert before.msg_id != after.msg_id
    assert before.msg_id.startswith("standalone:web_1:auto:")


@pytest.mark.parametrize(
    "over",
    [
        {"session_id": None},
        {"session_id": ""},
        {"question": None},
        {"question": 42},
        {"question": "   "},
        {"shop_id": None},
    ],
)
def test_invalid_submissions(sa, over):
    assert sa.on_message(_raw(**over)) is None


def test_suggestions_show_in_web_tool_but_never_send(sa):
    sa.on_message(_raw())
    card = SuggestionCard(
        text="亲，这款是纯棉的哦~",
        intent=Intent.PRODUCT_SPEC,
        kind=DraftKind.FAQ,
        confidence=Confidence.HIGH,
    )
    sa.show_suggestions(SHOP_C, "standalone:web_1", [card])
    assert [c.text for c in sa.sidebar(SHOP_C, "standalone:web_1")] == ["亲，这款是纯棉的哦~"]
    with pytest.raises(ActionNotAllowed):
        sa.send(SHOP_C, "standalone:web_1", "亲~")
    with pytest.raises(ActionNotAllowed):
        sa.transfer_to_human(SHOP_C, "standalone:web_1", "x")


@pytest.mark.parametrize("platform", list(Platform))
def test_works_for_every_platform(clock, hasher, platform):
    adapter = StandaloneAssistAdapter(platform, clock, hasher)
    assert adapter.access == "C"
    msg = adapter.on_message(_raw(shop_id=SHOP_A))
    assert msg.platform == platform
    assert isinstance(msg.received_at, datetime) and msg.received_at.tzinfo is not None
