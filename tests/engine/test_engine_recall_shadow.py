"""撤回的消息不再进上下文、按它给的推荐也不能再发（§5.3 新消息到达时刷新推荐）；
影子模式下迟到的旧消息让引擎重算最新一轮时，同一轮不记两套影子记录（§13.4 事后对比不重复计数）。"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from engine_support import Kit, make_route

from rodit.core.audit import AuditAction
from rodit.core.enums import Confidence, Intent, Mode, RouteAction, SuggestionStatus
from rodit.core.errors import ConcurrencyConflict, StorageUnavailable
from rodit.core.models import SuggestionRecord
from rodit.engine.cards import RECALL_REFRESH_HINT
from rodit.feedback import shadow_report
from rodit.testing.fakes import make_message
from rodit.testing.fixtures import BUYER_DOUYIN, CONV, SHOP_A, SHOP_D

# ================================================================ 撤回（§5.3）


def _shown(kit: Kit, shop_id: str = SHOP_A) -> list[SuggestionRecord]:
    return kit.suggestions.list(shop_id, conversation_id=CONV, status=SuggestionStatus.SHOWN)


def test_recall_expires_the_suggestions_shown_for_that_message(kit: Kit) -> None:
    engine = kit.engine()
    msg = kit.msg("什么时候发货")
    result = engine.handle(msg)
    assert result is not None and result.suggestions
    ids = {c.suggestion_id for c in result.suggestions}

    remaining = engine.on_recall(SHOP_A, CONV, msg.msg_id)

    assert remaining == []  # 侧边栏清空
    assert _shown(kit) == []
    for sid in ids:
        assert kit.suggestions.get(SHOP_A, sid).status == SuggestionStatus.EXPIRED  # type: ignore[arg-type,union-attr]
    assert engine.on_recall(SHOP_A, CONV, msg.msg_id) is None  # 平台重推撤回：没有变化，不用刷新


def test_recall_of_an_older_message_keeps_current_suggestions(kit: Kit) -> None:
    engine = kit.engine()
    older = kit.msg("在吗")
    engine.handle(older)
    newer = kit.msg("什么时候发货")
    current = engine.handle(newer)
    assert current is not None and current.suggestions

    assert engine.on_recall(SHOP_A, CONV, older.msg_id) is None
    assert [r.id for r in _shown(kit)] == [c.suggestion_id for c in current.suggestions]


def test_recall_returns_the_suggestions_still_shown(kit: Kit) -> None:
    """会话里还有别的消息的推荐在展示（例如上次刷新推荐时存储降级、没能过期）：
    侧边栏按剩下的刷新。记录里没有置信度和风险提示：按"中"、提醒客服核实。"""
    engine = kit.engine()
    msg = kit.msg("什么时候发货")
    engine.handle(msg)
    [kept] = _shown(kit)
    other = kit.suggestions.add(kept.model_copy(update={"msg_id": "m_other", "text": "亲，在的"}))

    remaining = engine.on_recall(SHOP_A, CONV, msg.msg_id)

    assert remaining is not None
    [card] = remaining
    assert card.suggestion_id == other.id and card.text == "亲，在的"
    assert card.intent == other.intent and card.kind == other.kind
    assert card.confidence == Confidence.MEDIUM
    assert card.risk_hints == [RECALL_REFRESH_HINT]
    assert [e.id for e in card.evidence] == other.evidence_ids


def test_recall_leaves_shadow_records_alone(kit: Kit) -> None:
    engine = kit.engine()
    msg = kit.msg("什么时候发货", shop_id=SHOP_D, buyer_ref=BUYER_DOUYIN)
    result = engine.handle(msg)
    assert result is not None and result.mode == Mode.SHADOW and result.suggestions
    assert engine.on_recall(SHOP_D, CONV, msg.msg_id) is None
    records = kit.suggestions.list(SHOP_D)
    assert {r.status for r in records} == {SuggestionStatus.SHADOW}
    # 状态不动，但撤回的消息原文不留（§5.3）：每周复核也看不到
    assert all(r.question_text is None for r in records)


def test_recall_forgets_the_question_on_records_the_agent_already_used(kit: Kit) -> None:
    engine = kit.engine()
    msg = kit.msg("什么时候发货")
    engine.handle(msg)
    [first, *_] = kit.suggestions.list(SHOP_A)
    kit.suggestions.update(
        first.model_copy(update={"status": SuggestionStatus.SENT}),
        expected_status=SuggestionStatus.SHOWN,
    )
    engine.on_recall(SHOP_A, CONV, msg.msg_id)
    records = {r.id: r for r in kit.suggestions.list(SHOP_A)}
    assert records[first.id].status == SuggestionStatus.SENT  # 客服的操作为准
    assert all(r.question_text is None for r in records.values())


def test_recall_does_not_override_what_the_agent_just_did(kit: Kit, monkeypatch: Any) -> None:
    engine = kit.engine()
    msg = kit.msg("什么时候发货")
    engine.handle(msg)

    def sent_meanwhile(record: SuggestionRecord, **kw: Any) -> SuggestionRecord:
        raise ConcurrencyConflict("客服刚发送了这条")

    monkeypatch.setattr(kit.suggestions, "update", sent_meanwhile)
    assert engine.on_recall(SHOP_A, CONV, msg.msg_id) is None


def test_recall_storage_failure_propagates(kit: Kit, monkeypatch: Any) -> None:
    """推荐记录写不进去：抛给应用层（平台会重推撤回，过期是幂等的）。"""
    engine = kit.engine()
    msg = kit.msg("什么时候发货")
    engine.handle(msg)

    def down(*a: Any, **kw: Any) -> Any:
        raise StorageUnavailable("数据库连接失败")

    monkeypatch.setattr(kit.suggestions, "list", down)
    with pytest.raises(StorageUnavailable):
        engine.on_recall(SHOP_A, CONV, msg.msg_id)


# ================================================================ 影子模式 + 乱序（§13.4、§8.3）


def _late(kit: Kit, newer: Any, seconds: float, shop_id: str = SHOP_D) -> Any:
    return make_message(
        "在吗",
        world=kit.world,
        shop_id=shop_id,
        buyer_ref=BUYER_DOUYIN,
        msg_id=f"late_{seconds}",
        at=newer.received_at - timedelta(seconds=seconds),
    )


def test_late_message_in_shadow_mode_does_not_record_the_turn_twice(kit: Kit) -> None:
    engine = kit.engine()
    newer = kit.msg("什么时候发货", shop_id=SHOP_D, buyer_ref=BUYER_DOUYIN)
    first = engine.handle(newer)
    assert first is not None and first.mode == Mode.SHADOW and first.suggestions
    before = kit.suggestions.list(SHOP_D)

    for seconds in (30, 40):  # 两条迟到的旧消息：每条都让引擎重算最新一轮
        again = engine.handle(_late(kit, newer, seconds))
        assert again is not None and again.mode == Mode.SHADOW
        assert again.msg_ids == [newer.msg_id] and again.suggestions

    assert kit.suggestions.list(SHOP_D) == before  # 这一轮只有第一次的那套影子记录
    report = shadow_report(kit.suggestions.list(SHOP_D), {newer.msg_id: "亲，48 小时内发货"})
    assert report.groups == len({(r.msg_id, r.intent) for r in before})
    suggest = kit.events(AuditAction.SUGGEST, SHOP_D)
    assert [e.detail.get("shadow_duplicate", False) for e in suggest] == [False, True, True]


def test_late_message_in_shadow_mode_records_a_turn_that_had_no_records(kit: Kit) -> None:
    """第一次处理这一轮没出推荐（没有影子记录）：重算出了推荐就照常记下（每轮一套）。"""
    kit.router.decide = make_route(Intent.COMPLAINT, RouteAction.HUMAN_PRIORITY, rule=2)
    engine = kit.engine()
    newer = kit.msg("什么时候发货", shop_id=SHOP_D, buyer_ref=BUYER_DOUYIN)
    engine.handle(newer)
    assert kit.suggestions.list(SHOP_D) == []

    kit.router.decide = make_route(Intent.SHIPPING_POLICY)
    again = engine.handle(_late(kit, newer, 30))

    assert again is not None and again.suggestions
    records = kit.suggestions.list(SHOP_D)
    assert records and {(r.msg_id, r.status) for r in records} == {
        (newer.msg_id, SuggestionStatus.SHADOW)
    }


def test_late_message_in_assist_mode_still_refreshes_suggestions(kit: Kit) -> None:
    """智能辅助照旧：旧的展示中推荐过期，重算的这一轮写新的（侧边栏刷新）。"""
    engine = kit.engine()
    newer = kit.msg("什么时候发货")
    first = engine.handle(newer)
    assert first is not None and first.suggestions
    again = engine.handle(
        make_message(
            "在吗", world=kit.world, msg_id="late_a", at=newer.received_at - timedelta(seconds=30)
        )
    )
    assert again is not None and again.suggestions
    statuses = [r.status for r in kit.suggestions.list(SHOP_A, conversation_id=CONV)]
    assert statuses.count(SuggestionStatus.EXPIRED) == len(first.suggestions)
    assert statuses.count(SuggestionStatus.SHOWN) == len(again.suggestions)
