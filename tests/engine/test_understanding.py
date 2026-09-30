"""会话理解（方案 §5.3、§5.7、§9.3）：合并连发、最近几轮、商品 / SKU / 订单指代、生效配置与模式。"""

from __future__ import annotations

from typing import Any

import pytest
from engine_support import Kit, build_kit

from rodit.core.config import EngineConfig
from rodit.core.enums import Mode, Platform, Role, ShopStatus
from rodit.core.errors import KnowledgeUnavailable, TenantIsolationError, ToolUnavailable
from rodit.core.models import ConversationContext, StandardMessage
from rodit.engine import (
    DEGRADED_TOOL_ORDER,
    ConversationBuilder,
    ShopInactive,
    Understanding,
    count_unresolved,
    effective_mode,
    keep_rounds,
    merge_turns,
)
from rodit.testing.fixtures import (
    BUYER,
    BUYER_DOUYIN,
    BUYER_R3,
    CONV,
    SHOP_A,
    SHOP_B,
    SHOP_C,
    SHOP_D,
)


def builder(kit: Kit, config: EngineConfig | None = None) -> ConversationBuilder:
    return ConversationBuilder(
        kit.knowledge, kit.conversations, kit.tools, config or kit.config, kit.clock
    )


def say(kit: Kit, text: str, *, after_s: float = 60, **kw: Any) -> StandardMessage:
    msg = kit.msg(text, after_s=after_s, **kw)
    assert kit.conversations.add_message(msg)
    return msg


def understand(kit: Kit, msg: StandardMessage, **kw: Any) -> Understanding:
    return builder(kit, **kw).build_with_notes(msg)


def ctx_of(kit: Kit, msg: StandardMessage, **kw: Any) -> ConversationContext:
    und = understand(kit, msg, **kw)
    return und.context


# ================================================================ 合并连发、最近几轮（§5.3）


def test_messages_within_merge_window_are_merged(kit: Kit) -> None:
    m1 = say(kit, "165")
    m2 = say(kit, "110斤穿什么码", after_s=1.5)
    ctx = ctx_of(kit, m2)
    assert ctx.current.text == "165 110斤穿什么码"
    assert ctx.current.msg_ids == [m1.msg_id, m2.msg_id]
    assert ctx.current.at == m2.received_at
    assert ctx.history == []


def test_merge_window_boundary(kit: Kit) -> None:
    m1 = say(kit, "在吗")
    m2 = say(kit, "问一下", after_s=2.0)  # 正好 2 秒：合并
    m3 = say(kit, "什么时候发货", after_s=2.01)  # 超过 2 秒：新的一轮
    ctx = ctx_of(kit, m3)
    assert ctx.current.msg_ids == [m3.msg_id]
    assert [t.msg_ids for t in ctx.history] == [[m1.msg_id, m2.msg_id]]


def test_merge_is_chained_and_only_same_role() -> None:
    kit_ = build_kit()
    a = say(kit_, "第一句")
    kit_.agent_says("客服回复", after_s=1)  # 不同角色：不合并
    b = say(kit_, "第二句", after_s=1)
    c = say(kit_, "第三句", after_s=1.9)
    turns = merge_turns(kit_.conversations.recent_messages(SHOP_A, CONV), 2.0)
    assert [t.role for t in turns] == [Role.BUYER, Role.AGENT, Role.BUYER]
    assert turns[0].msg_ids == [a.msg_id]
    assert turns[2].msg_ids == [b.msg_id, c.msg_id]


def test_history_keeps_last_five_rounds_with_replies(kit: Kit) -> None:
    for i in range(7):
        say(kit, f"问题{i}")
        kit.agent_says(f"回答{i}")
    last = say(kit, "最后一个问题")
    ctx = ctx_of(kit, last)
    texts = [t.text for t in ctx.history]
    assert texts == [
        "问题2",
        "回答2",
        "问题3",
        "回答3",
        "问题4",
        "回答4",
        "问题5",
        "回答5",
        "问题6",
        "回答6",
    ]
    assert ctx.current.text == "最后一个问题"


def test_keep_rounds_keeps_everything_when_short_and_leading_agent_turns(kit: Kit) -> None:
    say(kit, "在吗")
    kit.agent_says("在的亲")
    turns = merge_turns(kit.conversations.recent_messages(SHOP_A, CONV), 2.0)
    assert keep_rounds(turns, 5) == turns
    assert keep_rounds(turns, 0) == []
    assert [t.text for t in keep_rounds(turns, 1)] == ["在吗", "在的亲"]


def test_history_rounds_follow_config(kit: Kit) -> None:
    for i in range(4):
        say(kit, f"问题{i}")
        kit.agent_says(f"回答{i}")
    last = say(kit, "现在")
    config = EngineConfig.model_validate({"generation": {"history_rounds": 2}})
    ctx = ctx_of(kit, last, config=config)
    assert [t.text for t in ctx.history] == ["问题2", "回答2", "问题3", "回答3"]


def test_current_turn_is_cut_at_this_message(kit: Kit) -> None:
    """处理较早的消息时，之后已写入的消息不算进当前 Turn。"""
    m1 = say(kit, "什么时候发货")
    say(kit, "发什么快递")
    ctx = ctx_of(kit, m1)
    assert ctx.current.msg_ids == [m1.msg_id]
    assert ctx.history == []


def test_message_not_stored_is_still_the_current_turn(kit: Kit) -> None:
    earlier = say(kit, "在吗")
    msg = kit.msg("什么时候发货")  # 没先保存（例如会话存储故障）
    ctx = ctx_of(kit, msg)
    assert ctx.current.msg_ids == [msg.msg_id]
    assert [t.msg_ids for t in ctx.history] == [[earlier.msg_id]]


def test_conversation_storage_failure_degrades_to_this_message_only(kit: Kit) -> None:
    class Broken:
        def recent_messages(self, *a: Any, **kw: Any) -> list[Any]:
            raise RuntimeError("db down")

    msg = kit.msg("什么时候发货", item_id="i_555")
    b = ConversationBuilder(kit.knowledge, Broken(), kit.tools, kit.config, kit.clock)  # type: ignore[arg-type]
    und = b.build_with_notes(msg)
    assert und.context.current.text == "什么时候发货"
    assert und.context.item is not None and und.context.item.item_id == "i_555"
    assert "storage" in und.degraded


# ================================================================ 商品与 SKU 指代（§5.3、§5.2.4）


def test_item_card_on_message_sets_item_and_sku(kit: Kit) -> None:
    msg = say(kit, "这个是纯棉的吗", item_id="i_555", sku_id="k_1")
    ctx = ctx_of(kit, msg)
    assert ctx.item is not None and ctx.item.item_id == "i_555"
    assert ctx.sku_id == "k_1"
    assert ctx.candidate_item_ids == ["i_555"]


def test_item_carried_over_from_earlier_message(kit: Kit) -> None:
    say(kit, "在吗", item_id="i_555", sku_id="k_4")
    msg = say(kit, "能机洗吗")
    ctx = ctx_of(kit, msg)
    assert ctx.item is not None and ctx.item.item_id == "i_555"
    assert ctx.sku_id == "k_4"  # 沿用卡片上的 SKU


@pytest.mark.parametrize(
    ("text", "card_sku", "expected"),
    [
        ("那件黑色的L码还有吗", None, "k_2"),
        ("L码的呢", "k_1", "k_2"),
        ("白色S码有吗", None, "k_3"),
        ("黑的还有吗", "k_1", "k_1"),  # 两个黑色 SKU，卡片上的在其中：保留
        ("黑的还有吗", "k_4", None),  # 卡片上是白色：不确定
        ("黑的还有吗", None, None),
        ("黑色S码有吗", "k_1", None),  # 没有这个组合
        ("还有货吗", "k_4", "k_4"),  # 没提规格：沿用卡片
    ],
)
def test_sku_reference(kit: Kit, text: str, card_sku: str | None, expected: str | None) -> None:
    msg = say(kit, text, item_id="i_555", sku_id=card_sku)
    assert ctx_of(kit, msg).sku_id == expected


def test_sku_reference_on_item_from_history(kit: Kit) -> None:
    say(kit, "看看这条", item_id="i_777")
    msg = say(kit, "27码的还有吗")
    ctx = ctx_of(kit, msg)
    assert ctx.item is not None and ctx.item.item_id == "i_777"
    assert ctx.sku_id == "k_72"


def test_demonstrative_with_two_items_is_ambiguous(kit: Kit) -> None:
    say(kit, "看看", item_id="i_555")
    say(kit, "这个呢", item_id="i_777")
    msg = say(kit, "这个是纯棉的吗")
    ctx = ctx_of(kit, msg)
    assert ctx.item is None
    assert ctx.sku_id is None
    assert ctx.candidate_item_ids == ["i_777", "i_555"]  # 最近的在前


def test_size_question_with_two_items_stays_ambiguous(kit: Kit) -> None:
    """买家说"这个有 M 码吗"：只有连衣裙有 M 码，但这是在问尺码，不能据此认定是连衣裙。"""
    say(kit, "看看", item_id="i_555")
    say(kit, "这个呢", item_id="i_777")
    msg = say(kit, "这个有M码吗")
    assert ctx_of(kit, msg).item is None


def test_ambiguity_resolved_by_sku_value(kit: Kit) -> None:
    say(kit, "看看", item_id="i_555")
    say(kit, "这个呢", item_id="i_777")
    msg = say(kit, "那件黑色的还有吗")  # 只有连衣裙有黑色
    ctx = ctx_of(kit, msg)
    assert ctx.item is not None and ctx.item.item_id == "i_555"
    assert ctx.sku_id is None  # 两个黑色 SKU、历史卡片没带 SKU


def test_item_named_explicitly_wins(kit: Kit) -> None:
    say(kit, "看看", item_id="i_777")
    say(kit, "这个呢", item_id="i_555")
    msg = say(kit, "那条牛仔裤是高腰吗")
    ctx = ctx_of(kit, msg)
    assert ctx.item is not None and ctx.item.item_id == "i_777"


def test_no_demonstrative_defaults_to_most_recent_item(kit: Kit) -> None:
    say(kit, "看看", item_id="i_555")
    say(kit, "这个呢", item_id="i_777")
    msg = say(kit, "有M码吗")
    ctx = ctx_of(kit, msg)
    assert ctx.item is not None and ctx.item.item_id == "i_777"


def test_explicit_card_beats_ambiguity(kit: Kit) -> None:
    say(kit, "看看", item_id="i_555")
    say(kit, "这个呢", item_id="i_777")
    msg = say(kit, "这个有M码吗", item_id="i_555")
    ctx = ctx_of(kit, msg)
    assert ctx.item is not None and ctx.item.item_id == "i_555"
    assert ctx.candidate_item_ids == ["i_555", "i_777"]


def test_unknown_item_card_keeps_candidate_but_no_item(kit: Kit) -> None:
    msg = say(kit, "这个呢", item_id="i_not_synced")
    ctx = ctx_of(kit, msg)
    assert ctx.item is None
    assert ctx.candidate_item_ids == ["i_not_synced"]


def test_item_card_lookup_failure_is_degraded(kit: Kit) -> None:
    class FlakyKnowledge:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def get_item_card(self, shop_id: str, item_id: str) -> None:
            raise KnowledgeUnavailable("down")

    msg = say(kit, "这个是纯棉的吗", item_id="i_555")
    b = ConversationBuilder(
        FlakyKnowledge(kit.knowledge),  # type: ignore[arg-type]
        kit.conversations,
        kit.tools,
        kit.config,
        kit.clock,
    )
    und = b.build_with_notes(msg)
    assert und.context.item is None
    assert "knowledge" in und.degraded


# ================================================================ 订单指代（§5.3）


def test_recent_orders_are_loaded_newest_first(kit: Kit) -> None:
    msg = say(kit, "在吗")
    ctx = ctx_of(kit, msg)
    assert [o.order_id for o in ctx.recent_orders] == [
        "o_1001",
        "o_1010",
        "o_1002",
        "o_1008",
        "o_1003",
    ]
    assert ctx.order is None
    assert ("list_recent_orders", SHOP_A, (BUYER,)) in kit.tools.calls


def test_just_bought_refers_to_most_recent_order(kit: Kit) -> None:
    msg = say(kit, "刚买的那个什么时候发")
    ctx = ctx_of(kit, msg)
    assert ctx.order is not None and ctx.order.order_id == "o_1001"


def test_order_card_is_fetched(kit: Kit) -> None:
    msg = say(kit, "这个到哪了", order_id="o_1006")  # 不在最近 5 单里：要查一次
    ctx = ctx_of(kit, msg)
    assert ctx.order is not None and ctx.order.order_id == "o_1006"
    assert ("get_order", SHOP_A, ("o_1006",)) in kit.tools.calls


def test_order_card_in_recent_orders_needs_no_extra_call(kit: Kit) -> None:
    msg = say(kit, "这个到哪了", order_id="o_1002")
    ctx = ctx_of(kit, msg)
    assert ctx.order is not None and ctx.order.order_id == "o_1002"
    assert not [c for c in kit.tools.calls if c[0] == "get_order"]


def test_order_card_from_earlier_message(kit: Kit) -> None:
    say(kit, "帮我看下这单", order_id="o_1003")
    msg = say(kit, "显示签收了但我没收到")
    ctx = ctx_of(kit, msg)
    assert ctx.order is not None and ctx.order.order_id == "o_1003"


def test_card_on_this_message_beats_just_bought(kit: Kit) -> None:
    msg = say(kit, "刚买的这个到哪了", order_id="o_1002")
    assert ctx_of(kit, msg).order.order_id == "o_1002"  # type: ignore[union-attr]


def test_just_bought_beats_older_order_card(kit: Kit) -> None:
    say(kit, "这单", order_id="o_1003")
    msg = say(kit, "刚买的那个呢")
    assert ctx_of(kit, msg).order.order_id == "o_1001"  # type: ignore[union-attr]


def test_order_number_typed_in_raw_text(kit: Kit) -> None:
    msg = say(kit, "o_1002 到哪了")
    assert ctx_of(kit, msg).order.order_id == "o_1002"  # type: ignore[union-attr]
    msg = say(kit, "订单号：o_1009 怎么还没到")  # 不在最近 5 单：按订单号查
    assert ctx_of(kit, msg).order.order_id == "o_1009"  # type: ignore[union-attr]


def test_typed_order_number_of_another_buyer_is_ignored(kit: Kit) -> None:
    msg = say(kit, "订单号 o_1004 到哪了")  # BUYER_R3 的订单
    assert kit.world.order(SHOP_A, "o_1004").buyer_ref == BUYER_R3
    assert ctx_of(kit, msg).order is None


def test_order_tool_failure_degrades(kit: Kit) -> None:
    kit.tools.fail_on = {"list_recent_orders", "get_order"}
    msg = say(kit, "这个到哪了", order_id="o_1006")
    und = understand(kit, msg)
    assert und.context.recent_orders == []
    assert und.context.order is None
    assert und.degraded == [DEGRADED_TOOL_ORDER]


def test_recent_orders_failure_only(kit: Kit) -> None:
    kit.tools.fail_on = {"list_recent_orders"}
    msg = say(kit, "这个到哪了", order_id="o_1002")
    und = understand(kit, msg)
    assert und.context.order is not None and und.context.order.order_id == "o_1002"
    assert und.degraded == [DEGRADED_TOOL_ORDER]


@pytest.mark.parametrize("method", ["list_recent_orders", "get_order"])
def test_cross_shop_order_is_not_reported_as_a_tool_outage(
    kit: Kit, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """平台工具返回了别的店铺的订单（网关抛 TenantIsolationError）：不是"工具不可用"，要往外抛，
    由引擎记串店、不推荐（§9.2、§12.4 串店检查）。"""

    def cross_shop(*a: Any, **kw: Any) -> Any:
        raise TenantIsolationError("订单属于别的店铺")

    monkeypatch.setattr(kit.tools, method, cross_shop)
    msg = say(kit, "订单 o_1099 发货了吗")  # 不在最近订单里：会单独查订单
    with pytest.raises(TenantIsolationError):
        understand(kit, msg)


def test_tool_unavailable_type_is_what_gateway_raises(kit: Kit) -> None:
    kit.tools.fail_on = {"list_recent_orders"}
    with pytest.raises(ToolUnavailable):
        kit.tools.list_recent_orders(SHOP_A, BUYER)


# ================================================================ 没听懂（§5.7）


def test_unresolved_turns_count_consecutive_confusion(kit: Kit) -> None:
    say(kit, "什么时候发货")
    kit.agent_says("亲，默认发中通")
    say(kit, "没听懂")
    kit.agent_says("亲，付款后 48 小时内发货")
    msg = say(kit, "你这答非所问啊")
    assert ctx_of(kit, msg).unresolved_turns == 2
    kit.agent_says("抱歉")
    msg = say(kit, "好的谢谢")
    assert ctx_of(kit, msg).unresolved_turns == 0


def test_unresolved_turns_reset_by_normal_buyer_turn(kit: Kit) -> None:
    say(kit, "没听懂")
    say(kit, "什么时候发货")
    msg = say(kit, "我不是问这个")
    assert ctx_of(kit, msg).unresolved_turns == 1


def test_count_unresolved_on_turns(kit: Kit) -> None:
    say(kit, "看不懂")
    say(kit, "你在说什么")
    turns = merge_turns(kit.conversations.recent_messages(SHOP_A, CONV), 2.0)
    assert count_unresolved(turns) == 2
    assert count_unresolved([]) == 0


# ================================================================ 店铺、生效配置、模式（§9.3）


def test_effective_config_layers(kit: Kit) -> None:
    config = EngineConfig.model_validate(
        {
            "shop_defaults_by_platform": {"pdd": {"max_answer_chars": 100}},
            "shop_defaults_by_category": {"女装": {"tone": "亲切"}},
        }
    )
    msg = say(kit, "在吗", shop_id=SHOP_C)
    ctx = ctx_of(kit, msg, config=config)
    assert ctx.platform == Platform.PDD
    assert ctx.config.max_answer_chars == 100  # 平台默认
    assert ctx.config.tone == "简洁"  # 店铺覆盖 > 类目默认
    assert ctx.config.display_name == "拼拼小铺"
    assert ctx.shop.status == ShopStatus.TRIAL  # 试用店铺照常处理


@pytest.mark.parametrize(
    ("shop_overrides", "msg_mode", "expected"),
    [
        ({}, Mode.ASSIST, Mode.ASSIST),
        ({}, Mode.AUTO, Mode.ASSIST),  # 店铺没开全自动：不信任接入层
        ({"auto_reply_enabled": True}, Mode.AUTO, Mode.AUTO),
        ({"auto_reply_enabled": True}, Mode.ASSIST, Mode.ASSIST),
        ({"auto_reply_enabled": True, "shadow_mode": True}, Mode.AUTO, Mode.SHADOW),
        ({}, Mode.SHADOW, Mode.ASSIST),
    ],
)
def test_effective_mode(
    kit: Kit, shop_overrides: dict[str, Any], msg_mode: Mode, expected: Mode
) -> None:
    if shop_overrides:
        kit.set_shop_config(SHOP_A, **shop_overrides)
    msg = say(kit, "在吗", mode=msg_mode)
    assert ctx_of(kit, msg).mode == expected


def test_shadow_shop_is_shadow(kit: Kit) -> None:
    msg = say(kit, "在吗", shop_id=SHOP_D, buyer_ref=BUYER_DOUYIN, mode=Mode.AUTO)
    ctx = ctx_of(kit, msg)
    assert ctx.mode == Mode.SHADOW
    assert effective_mode(Mode.ASSIST, ctx.config) == Mode.SHADOW


def test_missing_or_suspended_shop_raises(kit: Kit) -> None:
    msg = kit.msg("在吗", shop_id="s_unknown", platform=Platform.TAOBAO)
    with pytest.raises(ShopInactive):
        builder(kit).build(msg)
    shop = kit.knowledge.get_shop(SHOP_B)
    assert shop is not None
    kit.knowledge.upsert_shop(shop.model_copy(update={"status": ShopStatus.SUSPENDED}))
    with pytest.raises(ShopInactive) as exc:
        builder(kit).build(kit.msg("在吗", shop_id=SHOP_B))
    assert exc.value.status == ShopStatus.SUSPENDED


def test_non_buyer_message_is_rejected(kit: Kit) -> None:
    msg = kit.msg("在的").model_copy(update={"role": Role.AGENT})
    with pytest.raises(ValueError):
        builder(kit).build(msg)


def test_context_fields(kit: Kit) -> None:
    msg = say(kit, "在吗", agent_id="kf_09")
    ctx = ctx_of(kit, msg)
    assert ctx.now == kit.clock.now()
    assert ctx.agent_id == "kf_09"
    assert ctx.buyer_ref == BUYER
    assert ctx.conversation_id == CONV
    assert ctx.shop_id == SHOP_A


# ================================================================ 店铺隔离（§9.2）


def test_same_conversation_id_in_two_shops_is_isolated(kit: Kit) -> None:
    say(kit, "A 店的问题", item_id="i_777")
    msg_b = say(kit, "B 店：这个是什么面料", shop_id=SHOP_B, item_id="i_555")
    ctx = ctx_of(kit, msg_b)
    assert ctx.history == []
    assert ctx.candidate_item_ids == ["i_555"]
    assert ctx.item is not None and ctx.item.title == "男士休闲裤"  # B 店自己的商品卡片
    assert [o.order_id for o in ctx.recent_orders] == ["o_9001"]
    msg_b2 = say(kit, "刚买的那个呢", shop_id=SHOP_B)
    assert ctx_of(kit, msg_b2).order.order_id == "o_9001"  # type: ignore[union-attr]


def test_order_card_from_other_shop_is_not_resolved(kit: Kit) -> None:
    msg = say(kit, "这个到哪了", shop_id=SHOP_B, order_id="o_1002")  # A 店的订单
    assert ctx_of(kit, msg).order is None
