"""BaseMockAdapter 的公共行为（方案 §8.2）：发送 / 转人工的接入模式限制、侧边栏、会话上下文、
构造校验、extension 白名单、店铺隔离、按店铺清理、并发去重。"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

import pytest

from rodit.adapters import (
    BaseMockAdapter,
    DouyinMockAdapter,
    NullToolGateway,
    PddMockAdapter,
    RawEvent,
    StandaloneAssistAdapter,
    TaobaoMockAdapter,
)
from rodit.core.enums import Confidence, DraftKind, Intent, Mode, Platform, Role
from rodit.core.errors import ActionNotAllowed
from rodit.core.models import SuggestionCard
from rodit.core.ports import PlatformAdapter
from rodit.testing.fakes import InMemoryToolGateway
from rodit.testing.fixtures import CONV, SHOP_A, SHOP_B, SHOP_C, SHOP_D


def _card(text: str, intent: Intent = Intent.SIZE_ADVICE) -> SuggestionCard:
    return SuggestionCard(
        suggestion_id=1,
        text=text,
        intent=intent,
        kind=DraftKind.LLM,
        confidence=Confidence.HIGH,
    )


@pytest.fixture
def auto(world, clock, hasher) -> TaobaoMockAdapter:
    return TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, hasher, access="A", auto_send_enabled=True
    )


@pytest.fixture
def plugin(world, clock, hasher) -> TaobaoMockAdapter:
    return TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="B")


def test_all_adapters_satisfy_platform_adapter_protocol(world, clock, hasher):
    adapters = [
        TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="A"),
        DouyinMockAdapter(InMemoryToolGateway(world, Platform.DOUYIN), clock, hasher, access="B"),
        PddMockAdapter(NullToolGateway(Platform.PDD), clock, hasher, access="C"),
        StandaloneAssistAdapter(Platform.TAOBAO, clock, hasher),
    ]
    for adapter in adapters:
        assert isinstance(adapter, PlatformAdapter)
        assert adapter.tools().platform == adapter.platform


# ================================================================ 发送（§8.2 仅模式 A 的全自动）


def test_send_in_mode_a_with_auto_enabled(auto, tb, clock):
    auto.on_message(tb(reception="robot"))
    result = auto.send(SHOP_A, CONV, "【智能客服】亲，建议选 M 码哦~")
    assert result.ok and result.platform_msg_id == "taobao_out_000001"
    sent = auto.sent_messages(SHOP_A)
    assert [(m.conversation_id, m.text, m.at) for m in sent] == [
        (CONV, "【智能客服】亲，建议选 M 码哦~", clock.now())
    ]
    ctx = auto.get_context(SHOP_A, CONV)
    assert [m.role for m in ctx.recent_messages] == [Role.BUYER, Role.BOT]
    assert auto.sent_messages(SHOP_B) == []


def test_send_rejects_empty_text_and_unknown_conversation(auto, tb):
    auto.on_message(tb())
    assert auto.send(SHOP_A, CONV, "   ").ok is False
    assert auto.send(SHOP_A, "conv_unknown", "亲~").error == "会话不存在"
    # 用别的店铺的身份往这个会话发：会话按店铺隔离，不存在
    assert auto.send(SHOP_B, CONV, "亲~").ok is False
    assert auto.sent_messages(SHOP_A) == [] and auto.sent_messages(SHOP_B) == []


def test_send_not_allowed_outside_mode_a_auto(world, clock, hasher, tb, dy, pdd):
    assist_a = TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="A")
    plugin = TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="B")
    douyin = DouyinMockAdapter(
        InMemoryToolGateway(world, Platform.DOUYIN), clock, hasher, access="B"
    )
    pdd_adapter = PddMockAdapter(NullToolGateway(Platform.PDD), clock, hasher, access="C")
    standalone = StandaloneAssistAdapter(Platform.TAOBAO, clock, hasher)
    assist_a.on_message(tb())
    plugin.on_message(tb())
    douyin.on_message(dy())
    pdd_adapter.on_message(pdd())
    cases = [
        (assist_a, SHOP_A, CONV),
        (plugin, SHOP_A, CONV),
        (douyin, SHOP_D, "dyc_001"),
        (pdd_adapter, SHOP_C, "pddc_001"),
        (standalone, SHOP_A, "standalone:web_1"),
    ]
    for adapter, shop, conv in cases:
        with pytest.raises(ActionNotAllowed):
            adapter.send(shop, conv, "亲~")
        assert adapter.sent_messages(shop) == []


def test_auto_send_requires_mode_a(world, clock, hasher):
    with pytest.raises(ValueError, match="只有模式 A"):
        TaobaoMockAdapter(
            InMemoryToolGateway(world), clock, hasher, access="B", auto_send_enabled=True
        )


# ================================================================ 转人工（§5.7）


def test_transfer_to_human_in_mode_a(auto, tb, pii):
    auto.on_message(tb(msg_id="m1", reception="robot"))
    result = auto.transfer_to_human(SHOP_A, CONV, f"买家要求人工，留了手机 {pii['phone']}")
    assert result.ok
    [record] = auto.transfers(SHOP_A)
    assert record.conversation_id == CONV
    assert pii["phone"] not in record.reason and "[手机号]" in record.reason
    # 转过人工的会话，之后机器人接待的消息也不再标记全自动，也不能再自动发送
    assert auto.on_message(tb(msg_id="m2", reception="robot")).mode == Mode.ASSIST
    blocked = auto.send(SHOP_A, CONV, "【智能客服】亲~")
    assert blocked.ok is False and "转人工" in blocked.error
    assert auto.sent_messages(SHOP_A) == []
    assert auto.transfer_to_human(SHOP_A, "conv_unknown", "x").ok is False
    assert auto.transfers(SHOP_B) == []


def test_transfer_to_human_in_mode_a_without_auto_is_allowed(world, clock, hasher, tb):
    assist_a = TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="A")
    assist_a.on_message(tb())
    assert assist_a.transfer_to_human(SHOP_A, CONV, "投诉").ok


def test_transfer_not_allowed_outside_mode_a(plugin, tb, clock, hasher):
    plugin.on_message(tb())
    with pytest.raises(ActionNotAllowed):
        plugin.transfer_to_human(SHOP_A, CONV, "买家要求人工")
    standalone = StandaloneAssistAdapter(Platform.PDD, clock, hasher)
    with pytest.raises(ActionNotAllowed):
        standalone.transfer_to_human(SHOP_C, "standalone:web", "x")


# ================================================================ 侧边栏（§5.6、模式 B）


def test_show_suggestions_replaces_previous_cards(plugin, tb):
    plugin.on_message(tb())
    plugin.show_suggestions(SHOP_A, CONV, [_card("建议 M 码"), _card("建议 L 码")])
    assert [c.text for c in plugin.sidebar(SHOP_A, CONV)] == ["建议 M 码", "建议 L 码"]
    plugin.show_suggestions(SHOP_A, CONV, [_card("亲，48 小时内发货", Intent.SHIPPING_POLICY)])
    assert [c.text for c in plugin.sidebar(SHOP_A, CONV)] == ["亲，48 小时内发货"]
    plugin.show_suggestions(SHOP_A, CONV, [])
    assert plugin.sidebar(SHOP_A, CONV) == []


def test_sidebar_is_isolated_and_copied(plugin, tb):
    plugin.on_message(tb())
    plugin.on_message(tb(seller_id=SHOP_B))
    cards = [_card("A 店推荐")]
    plugin.show_suggestions(SHOP_A, CONV, cards)
    cards[0].text = "调用方改了自己的对象"
    assert plugin.sidebar(SHOP_A, CONV)[0].text == "A 店推荐"
    assert plugin.sidebar(SHOP_B, CONV) == []  # 同一个会话 ID，B 店看不到 A 店的推荐
    shown = plugin.sidebar(SHOP_A, CONV)
    shown.clear()
    assert len(plugin.sidebar(SHOP_A, CONV)) == 1


def test_show_suggestions_for_unknown_conversation_is_dropped(plugin):
    plugin.show_suggestions(SHOP_A, "conv_never_seen", [_card("x")])
    assert plugin.sidebar(SHOP_A, "conv_never_seen") == []


# ================================================================ 会话上下文（§8.2 getContext）


def test_get_context_tracks_cards_and_recent_messages(world, clock, hasher, tb):
    adapter = TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, hasher, access="B", context_limit=3
    )
    adapter.on_message(tb(msg_id="m1", item_id="i_555", sku_id="k_1"))
    adapter.on_message(tb(msg_id="m2", msg_type="order", content="", item_id=None, tid="o_1001"))
    adapter.on_message(tb(msg_id="m3", content="这单什么时候发", item_id=None))
    adapter.on_message(tb(msg_id="m4", content="急用", item_id=None))
    ctx = adapter.get_context(SHOP_A, CONV)
    assert ctx.item_card.item_id == "i_555" and ctx.item_card.sku_id == "k_1"
    assert ctx.order_card.order_id == "o_1001"
    assert [m.msg_id for m in ctx.recent_messages] == ["m2", "m3", "m4"]  # 只保留最近 3 条
    assert ctx.recent_messages[0].order_ref == "o_1001"
    # 新商品卡片覆盖旧的
    adapter.on_message(tb(msg_id="m5", item_id="i_777"))
    assert adapter.get_context(SHOP_A, CONV).item_card.item_id == "i_777"


def test_get_context_sorts_out_of_order_pushes(plugin, tb):
    from rodit.testing.fixtures import DEFAULT_NOW

    ms = int(DEFAULT_NOW.timestamp() * 1000)
    plugin.on_message(tb(msg_id="late", send_time_ms=ms + 5000))
    plugin.on_message(tb(msg_id="early", send_time_ms=ms))
    ctx = plugin.get_context(SHOP_A, CONV)
    assert [m.msg_id for m in ctx.recent_messages] == ["early", "late"]


def test_get_context_unknown_or_other_shop_is_empty(plugin, tb):
    plugin.on_message(tb())
    for shop, conv in ((SHOP_A, "conv_x"), (SHOP_B, CONV)):
        ctx = plugin.get_context(shop, conv)
        assert (ctx.shop_id, ctx.conversation_id) == (shop, conv)
        assert ctx.item_card is None and ctx.order_card is None and ctx.recent_messages == []


def test_max_conversations_evicts_least_recent(world, clock, hasher, tb):
    adapter = TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, hasher, access="B", max_conversations=2
    )
    for i, conv in enumerate(("c1", "c2", "c1", "c3")):
        adapter.on_message(tb(msg_id=f"m{i}", session_id=conv))
    assert adapter.get_context(SHOP_A, "c2").recent_messages == []
    assert len(adapter.get_context(SHOP_A, "c1").recent_messages) == 2


def test_agent_echo_for_unknown_conversation_is_not_recorded(plugin, tb):
    echo = tb(msg_id="a1", direction="out", content="亲~", session_id="conv_other")
    assert plugin.on_message(echo) is None
    assert plugin.get_context(SHOP_A, "conv_other").recent_messages == []


# ================================================================ 构造校验、工具


def test_tools_returns_injected_gateway(world, clock, hasher):
    gateway = InMemoryToolGateway(world)
    adapter = TaobaoMockAdapter(gateway, clock, hasher, access="B")
    assert adapter.tools() is gateway


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"access": "D"}, "未知的接入模式"),
        ({"access": "C"}, "只支持接入模式"),
        ({"access": "B", "dedup_capacity": 0}, "必须 ≥ 1"),
        ({"access": "B", "context_limit": 0}, "必须 ≥ 1"),
    ],
)
def test_constructor_validation(world, clock, hasher, kwargs, match):
    with pytest.raises(ValueError, match=match):
        TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, **kwargs)


def test_tool_gateway_must_belong_to_same_platform(world, clock, hasher):
    """每个平台一个部署单元（§2.2、§9.1）：不能把别的平台的工具接到这个适配器上。"""
    with pytest.raises(ValueError, match="不跨云"):
        TaobaoMockAdapter(InMemoryToolGateway(world, Platform.PDD), clock, hasher, access="B")


# ================================================================ extension 白名单（§5.3）


def test_extension_whitelist_cannot_contain_pii_fields():
    with pytest.raises(TypeError, match="个人信息"):

        class _Leaky(BaseMockAdapter):  # noqa: F841
            platform = Platform.TAOBAO
            extension_fields = frozenset({"msg_type", "buyer_nick"})


class _ExtensionProbe(BaseMockAdapter):
    """测试用：把原始消息里的 ext 原样交给公共逻辑，检查过滤规则。"""

    platform = Platform.TAOBAO
    extension_fields = frozenset({"note", "count", "flag", "nested"})

    def _parse(self, raw: Mapping[str, Any]) -> RawEvent | None:
        return RawEvent(
            kind="buyer",
            shop_id=SHOP_A,
            conversation_id=CONV,
            msg_id=raw["id"],
            raw_buyer_id="tb_demo_buyer_001",
            text="在吗",
            extension=raw["ext"],
        )


def test_extension_drops_unlisted_nested_and_pii_values(world, clock, hasher, pii):
    probe = _ExtensionProbe(InMemoryToolGateway(world), clock, hasher, access="B")
    msg = probe.on_message(
        {
            "id": "e1",
            "ext": {
                "note": f"买家留言 {pii['phone']}",  # 字符串含个人信息：丢弃
                "count": 3,
                "flag": True,
                "nested": {"a": 1},  # 不是标量：丢弃
                "unlisted": "x",  # 不在白名单：丢弃
                "buyer_nick": pii["nick"],
            },
        }
    )
    assert msg.extension == {"count": 3, "flag": True}
    ok = probe.on_message({"id": "e2", "ext": {"note": "来自直播间"}})
    assert ok.extension == {"note": "来自直播间"}


def test_naive_timestamp_from_parser_is_rejected(world, clock, hasher):
    from datetime import datetime

    class _Naive(_ExtensionProbe):
        def _parse(self, raw):
            event = super()._parse(raw)
            return RawEvent(**{**event.__dict__, "at": datetime(2026, 9, 28, 10, 35)})

    adapter = _Naive(InMemoryToolGateway(world), clock, hasher, access="B")
    assert adapter.on_message({"id": "n1", "ext": {}}) is None


# ================================================================ 清理与并发


def test_purge_shop_clears_only_that_shop(auto, tb):
    auto.on_message(tb(msg_id="m1", reception="robot"))
    auto.on_message(tb(msg_id="m1", seller_id=SHOP_B))
    auto.show_suggestions(SHOP_A, CONV, [_card("x")])
    auto.send(SHOP_A, CONV, "【智能客服】亲~")
    auto.transfer_to_human(SHOP_A, CONV, "买家要求人工")
    assert auto.purge_shop(SHOP_A) == 3  # 1 个会话 + 1 条发出的消息 + 1 条转人工
    assert auto.get_context(SHOP_A, CONV).recent_messages == []
    assert auto.sidebar(SHOP_A, CONV) == []
    assert auto.sent_messages(SHOP_A) == [] and auto.transfers(SHOP_A) == []
    assert len(auto.get_context(SHOP_B, CONV).recent_messages) == 1
    assert auto.on_message(tb(msg_id="m1")) is not None  # 去重记录也清掉了
    assert auto.purge_shop("s_none") == 0


def test_concurrent_duplicates_are_accepted_once(plugin, tb):
    results: list[object] = []
    barrier = threading.Barrier(8)

    def push() -> None:
        barrier.wait()
        results.append(plugin.on_message(tb(msg_id="dup")))

    threads = [threading.Thread(target=push) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r is not None for r in results) == 1
    assert len(plugin.get_context(SHOP_A, CONV).recent_messages) == 1


def test_peek_shop_id_does_not_change_state(plugin, tb):
    """peek_shop_id 只解析：不去重、不记上下文（应用层先用它核对店铺和授权，§9.2）。"""
    raw = tb(msg_id="peek_1")
    assert plugin.peek_shop_id(raw) == SHOP_A
    assert plugin.peek_shop_id(raw) == SHOP_A
    assert plugin.get_context(SHOP_A, CONV).recent_messages == []
    assert plugin.on_message(raw) is not None  # 没被 peek 当成重复
    assert plugin.on_message(raw) is None


@pytest.mark.parametrize("raw", [None, [], "x", {}, {"seller_id": None}, {"seller_id": "  "}])
def test_peek_shop_id_returns_none_for_unrecognizable_events(plugin, raw):
    assert plugin.peek_shop_id(raw) is None
