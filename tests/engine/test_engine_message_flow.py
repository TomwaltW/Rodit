"""引擎的消息顺序与全自动兜底（§8.3 按会话保证顺序、§5.3 合并连发与指代、§12.2 全自动任何故障
切回人工）。"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from engine_support import CLARIFY_ITEM, Kit, make_route

from rodit.core.enums import Intent, Mode, Platform, SuggestionStatus
from rodit.core.errors import KnowledgeUnavailable
from rodit.testing.fakes import make_message
from rodit.testing.fixtures import CONV, SHOP_A

# ================================================================ 乱序（§8.3）


def test_late_older_message_reprocesses_latest_turn(kit: Kit) -> None:
    engine = kit.engine()
    newer = kit.msg("有M码吗", item_id="i_555")
    first = engine.handle(newer)
    assert first is not None and first.suggestions
    older = make_message(
        "什么时候发货", world=kit.world, msg_id="late", at=newer.received_at - timedelta(seconds=1)
    )
    result = engine.handle(older)
    assert result is not None
    # 处理的是会话里最新的买家 Turn（迟到的那条在 1 秒内，合并进去），不是只回答旧问题
    assert result.msg_ids == ["late", newer.msg_id]
    assert kit.router.calls[-1].current.text == "什么时候发货 有M码吗"
    shown = kit.suggestions.list(SHOP_A, conversation_id=CONV, status=SuggestionStatus.SHOWN)
    assert shown and all(r.msg_id == newer.msg_id for r in shown)
    # 迟到的消息照样存进会话（之后几轮的上下文），重复推送照样去重
    assert engine.handle(older) is None


def test_late_message_outside_merge_window_goes_to_history(kit: Kit) -> None:
    engine = kit.engine()
    newer = kit.msg("有M码吗", item_id="i_555")
    engine.handle(newer)
    older = make_message(
        "什么时候发货", world=kit.world, msg_id="late", at=newer.received_at - timedelta(seconds=30)
    )
    result = engine.handle(older)
    assert result is not None and result.msg_ids == [newer.msg_id]
    ctx = kit.router.calls[-1]
    assert ctx.current.text == "有M码吗"
    assert [t.text for t in ctx.history] == ["什么时候发货"]


def test_late_message_in_auto_mode_is_not_auto_sent_again(kit: Kit) -> None:
    """全自动：最新那条已经自动回复过，迟到的旧消息不再自动发一遍，转人工由客服接管。"""
    kit.set_shop_config(SHOP_A, auto_reply_enabled=True, auto_reply_schedule="always")
    kit.router.decide = make_route(Intent.SHIPPING_POLICY, confidence=0.97, auto_eligible=True)
    engine = kit.engine()
    newer = kit.msg("什么时候发货", mode=Mode.AUTO)
    first = engine.handle(newer)
    assert first is not None and first.auto_reply is not None
    older = make_message(
        "在吗",
        world=kit.world,
        msg_id="late_auto",
        at=newer.received_at - timedelta(seconds=30),
        mode=Mode.AUTO,
    )
    result = engine.handle(older)
    assert result is not None and result.mode == Mode.AUTO
    assert result.auto_reply is None and result.transfer_to_human is True


def test_equal_timestamp_is_not_treated_as_late(kit: Kit) -> None:
    engine = kit.engine()
    first = kit.msg("有M码吗", item_id="i_555")
    engine.handle(first)
    same = make_message("什么时候发货", world=kit.world, msg_id="same", at=first.received_at)
    result = engine.handle(same)
    assert result is not None and result.msg_ids == [first.msg_id, "same"]


# ======================================================== 全自动：店铺读不出来也要转人工（§12.2）


def test_auto_message_shop_load_failure_transfers(kit: Kit, monkeypatch: Any) -> None:
    kit.set_shop_config(SHOP_A, auto_reply_enabled=True)

    def down(shop_id: str) -> Any:
        raise KnowledgeUnavailable("down")

    monkeypatch.setattr(kit.knowledge, "get_shop", down)
    result = kit.engine().handle(kit.msg("在吗", mode=Mode.AUTO))
    assert result is not None
    assert result.degraded == ["knowledge"]
    assert result.transfer_to_human is True


def test_assist_message_shop_load_failure_does_not_transfer(kit: Kit, monkeypatch: Any) -> None:
    def down(shop_id: str) -> Any:
        raise KnowledgeUnavailable("down")

    monkeypatch.setattr(kit.knowledge, "get_shop", down)
    result = kit.engine().handle(kit.msg("在吗"))
    assert result is not None and result.transfer_to_human is False


def test_auto_message_config_failure_transfers(kit: Kit, monkeypatch: Any) -> None:
    kit.set_shop_config(SHOP_A, auto_reply_enabled=True)
    engine = kit.engine()

    def broken(shop: Any) -> Any:
        raise ValueError("bad config")

    monkeypatch.setattr(engine.builder, "shop_config", broken)
    result = engine.handle(kit.msg("在吗", mode=Mode.AUTO))
    assert result is not None and result.transfer_to_human is True


def test_inactive_shop_auto_message_is_not_transferred(kit: Kit) -> None:
    """店铺不存在 / 已停用：停止处理（§8.4），不去动平台会话。"""
    result = kit.engine().handle(
        kit.msg("在吗", shop_id="s_unknown", platform=Platform.TAOBAO, mode=Mode.AUTO)
    )
    assert result is not None and result.transfer_to_human is False


# ======================================================== 合并的 Turn 里先发的商品卡片（§5.3）


def test_item_card_sent_earlier_in_merged_turn_is_used(kit: Kit) -> None:
    engine = kit.engine()
    engine.handle(kit.msg("这条牛仔裤什么面料", item_id="i_777"))
    engine.handle(kit.msg("", item_id="i_555"))  # 商品卡片
    result = engine.handle(kit.msg("这个有M码吗", after_s=1))
    assert result is not None and len(result.msg_ids) == 2
    ctx = kit.router.calls[-1]
    assert ctx.item is not None and ctx.item.item_id == "i_555"
    assert [c.text for c in result.suggestions] != [CLARIFY_ITEM]


def test_follow_up_merged_with_carded_message_keeps_item(kit: Kit) -> None:
    engine = kit.engine()
    engine.handle(kit.msg("这条牛仔裤什么面料", item_id="i_777"))
    engine.handle(kit.msg("这个有M码吗", item_id="i_555"))
    engine.handle(kit.msg("还有这个是什么面料", after_s=1))
    ctx = kit.router.calls[-1]
    assert ctx.current.text == "这个有M码吗 还有这个是什么面料"
    assert ctx.item is not None and ctx.item.item_id == "i_555"
