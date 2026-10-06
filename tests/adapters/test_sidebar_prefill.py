"""半自动的侧边栏内容（方案 §5.6 "推荐直接填进输入框，客服按发送即可（视平台插件能力）"、§8.2）。

- 卡片上的 prefill 原样带给插件；一批里最多留一条（第一条带推荐编号的）；有标红提醒时一条不留；
- sidebar_payload.input_prefill：模式 A / B 的插件把这条填进客服输入框；模式 C（网页工具、拼多多）
  没有平台输入框，只在卡片上标记；
- 按店铺、会话分开（§9.2）；新的一批替换旧的。
"""

from __future__ import annotations

import pytest

from rodit.adapters import (
    DouyinMockAdapter,
    InputPrefill,
    NullToolGateway,
    PddMockAdapter,
    SidebarPayload,
    StandaloneAssistAdapter,
    TaobaoMockAdapter,
)
from rodit.core.enums import Confidence, DraftKind, Intent, Platform
from rodit.core.models import SuggestionCard
from rodit.testing.fakes import InMemoryToolGateway
from rodit.testing.fixtures import CONV, SHOP_A, SHOP_B, SHOP_C, SHOP_D


def card(text: str, sid: int | None = 1, *, prefill: bool = False) -> SuggestionCard:
    return SuggestionCard(
        suggestion_id=sid,
        text=text,
        intent=Intent.SHIPPING_POLICY,
        kind=DraftKind.FAQ,
        confidence=Confidence.HIGH,
        prefill=prefill,
    )


@pytest.fixture
def plugin(world, clock, hasher) -> TaobaoMockAdapter:
    return TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="B")


def test_plugin_payload_fills_the_input_box_with_the_prefill_card(plugin, tb):
    plugin.on_message(tb())
    plugin.show_suggestions(
        SHOP_A, CONV, [card("亲，48 小时内发货", 7, prefill=True), card("亲，现货当天发", 8)]
    )
    payload = plugin.sidebar_payload(SHOP_A, CONV)
    assert isinstance(payload, SidebarPayload)
    assert payload.input_prefill == InputPrefill(suggestion_id=7, text="亲，48 小时内发货")
    assert [c.prefill for c in payload.cards] == [True, False]
    assert payload.alerts == [] and payload.notice is None
    assert [c.prefill for c in plugin.sidebar(SHOP_A, CONV)] == [True, False]


def test_assist_reception_in_mode_a_also_prefills(world, clock, hasher, tb):
    adapter = TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, hasher, access="A", auto_send_enabled=True
    )
    adapter.on_message(tb())
    adapter.show_suggestions(SHOP_A, CONV, [card("亲，48 小时内发货", 3, prefill=True)])
    payload = adapter.sidebar_payload(SHOP_A, CONV)
    assert payload is not None and payload.input_prefill is not None
    assert payload.input_prefill.suggestion_id == 3
    assert adapter.sent_messages(SHOP_A) == []  # 只填进输入框，不是自动发送


def test_douyin_plugin_payload(world, clock, hasher, dy):
    adapter = DouyinMockAdapter(
        InMemoryToolGateway(world, Platform.DOUYIN), clock, hasher, access="B"
    )
    msg = adapter.on_message(dy())
    assert msg is not None
    adapter.show_suggestions(SHOP_D, msg.conversation_id, [card("亲，48 小时内发货", prefill=True)])
    payload = adapter.sidebar_payload(SHOP_D, msg.conversation_id)
    assert payload is not None and payload.input_prefill is not None
    assert payload.input_prefill.text == "亲，48 小时内发货"


def test_mode_c_marks_the_card_but_has_no_platform_input_box(clock, hasher, pdd):
    standalone = StandaloneAssistAdapter(Platform.TAOBAO, clock, hasher)
    msg = standalone.on_message({"shop_id": SHOP_A, "session_id": "web1", "question": "发货吗"})
    assert msg is not None
    standalone.show_suggestions(SHOP_A, msg.conversation_id, [card("亲，48 小时", prefill=True)])
    payload = standalone.sidebar_payload(SHOP_A, msg.conversation_id)
    assert payload is not None and payload.input_prefill is None
    assert payload.cards[0].prefill is True  # 网页工具上显示"可直接发送"
    pdd_adapter = PddMockAdapter(NullToolGateway(Platform.PDD), clock, hasher, access="C")
    msg = pdd_adapter.on_message(pdd())
    assert msg is not None
    pdd_adapter.show_suggestions(SHOP_C, msg.conversation_id, [card("亲", prefill=True)])
    payload = pdd_adapter.sidebar_payload(SHOP_C, msg.conversation_id)
    assert payload is not None and payload.input_prefill is None


def test_at_most_one_prefill_card_with_a_suggestion_id(plugin, tb):
    plugin.on_message(tb())
    plugin.show_suggestions(
        SHOP_A,
        CONV,
        [
            card("没有编号", None, prefill=True),  # 发出去记不了反馈：不填
            card("第一条带编号的", 5, prefill=True),
            card("第二条带编号的", 6, prefill=True),
        ],
    )
    assert [c.prefill for c in plugin.sidebar(SHOP_A, CONV)] == [False, True, False]
    payload = plugin.sidebar_payload(SHOP_A, CONV)
    assert payload is not None and payload.input_prefill is not None
    assert payload.input_prefill.suggestion_id == 5


def test_alerts_suppress_the_prefill(plugin, tb):
    """有标红提醒（要人工、情绪激动……）时客服要先看提醒，不往输入框里填。"""
    plugin.on_message(tb())
    plugin.show_suggestions(
        SHOP_A,
        CONV,
        [card("亲，48 小时内发货", 9, prefill=True)],
        alerts=["买家情绪激动，建议先安抚"],
    )
    payload = plugin.sidebar_payload(SHOP_A, CONV)
    assert payload is not None
    assert payload.input_prefill is None and [c.prefill for c in payload.cards] == [False]
    assert payload.alerts == ["买家情绪激动，建议先安抚"]


def test_new_batch_replaces_the_prefill(plugin, tb):
    plugin.on_message(tb())
    plugin.show_suggestions(SHOP_A, CONV, [card("旧的", 1, prefill=True)])
    plugin.show_suggestions(SHOP_A, CONV, [card("新的", 2)], notice=None)
    payload = plugin.sidebar_payload(SHOP_A, CONV)
    assert payload is not None and payload.input_prefill is None
    plugin.show_suggestions(SHOP_A, CONV, [], notice="没有把握判断问题类型")
    payload = plugin.sidebar_payload(SHOP_A, CONV)
    assert payload is not None and payload.cards == [] and payload.input_prefill is None
    assert payload.notice == "没有把握判断问题类型"


def test_payload_is_per_shop_and_conversation(plugin, tb):
    plugin.on_message(tb())
    plugin.on_message(tb(seller_id=SHOP_B, msg_id="tb_b_1"))
    plugin.show_suggestions(SHOP_A, CONV, [card("A 店推荐", 4, prefill=True)])
    payload_b = plugin.sidebar_payload(SHOP_B, CONV)  # 同一个会话 ID，B 店看不到 A 店的
    assert payload_b is not None and payload_b.cards == [] and payload_b.input_prefill is None
    assert plugin.sidebar_payload(SHOP_A, "conv_never_seen") is None
    assert plugin.sidebar_payload("s_unknown", CONV) is None


def test_payload_is_a_copy(plugin, tb):
    plugin.on_message(tb())
    cards = [card("A 店推荐", 4, prefill=True)]
    plugin.show_suggestions(SHOP_A, CONV, cards)
    cards[0].prefill = False  # 调用方改自己的对象，不影响侧边栏
    payload = plugin.sidebar_payload(SHOP_A, CONV)
    assert payload is not None and payload.cards[0].prefill is True
    payload.cards[0].prefill = False
    again = plugin.sidebar_payload(SHOP_A, CONV)
    assert again is not None and again.input_prefill is not None


def test_purge_shop_drops_the_payload(plugin, tb):
    plugin.on_message(tb())
    plugin.show_suggestions(SHOP_A, CONV, [card("A 店推荐", 4, prefill=True)])
    plugin.purge_shop(SHOP_A)
    assert plugin.sidebar_payload(SHOP_A, CONV) is None
