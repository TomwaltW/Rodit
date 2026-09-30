"""撤回事件走完整的应用流程（§5.3 撤回的消息不再进上下文、新消息到达时刷新推荐；
§8.3 平台推送乱序）。

- 撤回刚收到的买家消息：按它给的推荐过期，侧边栏清空，客服再点发送会被拒；
- 撤回先于被撤回的消息到达：消息到了也不保存、不回答、不上侧边栏，平台重推也一样；
- 推荐记录写不进去：撤回事件报错让平台重推，重推时补上；
- 影子模式：迟到的旧消息不会让同一轮多一套影子记录（§13.4）。
全部合成数据，不联网。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.core.enums import Platform, Role, SuggestionStatus
from rodit.core.errors import ActionNotAllowed, StorageUnavailable
from rodit.core.ports import RECALL_TOMBSTONE_TTL
from rodit.engine.cards import RECALL_REFRESH_HINT
from rodit.feedback import shadow_report
from rodit.testing.fixtures import AGENT, SHOP_A

CONV = "conv_demo"


def _ms(service: AssistService, offset_s: float = 0) -> int:
    return int((service.unit.clock.now().timestamp() + offset_s) * 1000)


def _recall(msg_id: str, recalled: str) -> dict:
    return {
        "seller_id": SHOP_A,
        "session_id": CONV,
        "msg_id": msg_id,
        "msg_type": "recall",
        "recall_msg_id": recalled,
    }


def _service() -> tuple[AssistService, Any]:
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    return service, service.unit.adapters.get(Platform.TAOBAO, "B")


def _records(service: AssistService) -> list[Any]:
    return service.unit.repos.suggestions.list(SHOP_A, conversation_id=CONV)


def test_recall_expires_its_suggestions_and_clears_the_sidebar():
    service, adapter = _service()
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="r1"))
    assert result is not None and result.suggestions
    assert adapter.sidebar(SHOP_A, CONV) == result.suggestions

    assert service.handle_event("B", _recall("r1_recall", "r1")) is None

    assert adapter.sidebar(SHOP_A, CONV) == []
    assert {r.status for r in _records(service)} == {SuggestionStatus.EXPIRED}
    with pytest.raises(ActionNotAllowed):  # 侧边栏没来得及刷新时客服点了发送：拒绝
        service.feedback_sent(SHOP_A, result.suggestions[0].suggestion_id, agent_id=AGENT)
    assert service.unit.repos.conversations.recent_messages(SHOP_A, CONV) == []


def test_recall_of_an_older_message_leaves_the_sidebar_alone():
    service, adapter = _service()
    t0 = _ms(service)
    service.handle_event("B", taobao_event("在吗", msg_id="o1", send_time_ms=t0))
    current = service.handle_event(
        "B", taobao_event("什么时候发货", msg_id="o2", send_time_ms=t0 + 60_000)
    )
    assert current is not None and current.suggestions

    service.handle_event("B", _recall("o1_recall", "o1"))

    assert adapter.sidebar(SHOP_A, CONV) == current.suggestions
    shown = [r for r in _records(service) if r.status == SuggestionStatus.SHOWN]
    assert [r.id for r in shown] == [c.suggestion_id for c in current.suggestions]


def test_sidebar_is_refreshed_with_suggestions_still_shown():
    service, adapter = _service()
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="s1"))
    assert result is not None
    kept = _records(service)[0]
    # 另一条消息的推荐还在展示（上次刷新推荐时存储降级、没能过期）
    other = service.unit.repos.suggestions.add(
        kept.model_copy(update={"msg_id": "s0", "text": "亲，在的"})
    )

    service.handle_event("B", _recall("s1_recall", "s1"))

    [card] = adapter.sidebar(SHOP_A, CONV)
    assert card.suggestion_id == other.id and card.risk_hints == [RECALL_REFRESH_HINT]


def test_message_recalled_before_it_arrives_is_dropped():
    """撤回先到（平台推送乱序）：被撤回的消息到达时不保存、不回答、不上侧边栏。"""
    service, adapter = _service()
    t0 = _ms(service)
    first = service.handle_event("B", taobao_event("在吗", msg_id="e0", send_time_ms=t0))
    assert first is not None
    before = adapter.sidebar(SHOP_A, CONV)
    assert service.handle_event("B", _recall("e1_recall", "e1")) is None

    early = taobao_event("我的手机号13812345678，这件有M码吗", msg_id="e1", send_time_ms=t0 + 5000)
    assert service.handle_event("B", early) is None
    assert service.handle_event("B", early) is None  # 平台重推

    repo = service.unit.repos.conversations
    assert [(m.msg_id, m.role) for m in repo.recent_messages(SHOP_A, CONV)] == [("e0", Role.BUYER)]
    assert all(r.msg_id != "e1" for r in _records(service))
    assert adapter.sidebar(SHOP_A, CONV) == before
    context = adapter.get_context(SHOP_A, CONV)  # 接入层的本地上下文也不留它
    assert [m.msg_id for m in context.recent_messages] == ["e0"]
    assert "13812345678" not in "".join(m.text or "" for m in context.recent_messages)


def test_recall_expiry_failure_is_redelivered(monkeypatch: Any):
    """推荐记录写不进去：撤回事件报错（平台会重推），重推时过期并刷新侧边栏。"""
    service, adapter = _service()
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="f1"))
    assert result is not None and result.suggestions
    repo = service.unit.repos.suggestions
    real_update = repo.update
    calls = {"n": 0}

    def flaky(record: Any, **kw: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise StorageUnavailable("数据库连接失败")
        return real_update(record, **kw)

    monkeypatch.setattr(repo, "update", flaky)
    with pytest.raises(StorageUnavailable):
        service.handle_event("B", _recall("f1_recall", "f1"))
    assert service.unit.repos.conversations.recent_messages(SHOP_A, CONV) == []

    assert service.handle_event("B", _recall("f1_recall", "f1")) is None  # 平台重推

    assert {r.status for r in _records(service)} == {SuggestionStatus.EXPIRED}
    assert adapter.sidebar(SHOP_A, CONV) == []


def test_recall_clears_sidebar_even_if_a_later_read_would_fail(monkeypatch: Any):
    """过期提交之后再读推荐列表失败的话，重推时已经没有可过期的记录、侧边栏就刷新不了；
    剩下的卡片从过期之前读到的列表里算，只读一次。"""
    service, adapter = _service()
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="g1"))
    assert result is not None and result.suggestions
    repo = service.unit.repos.suggestions
    real_list = repo.list
    calls = {"n": 0}

    def flaky(*args: Any, **kw: Any) -> Any:
        if kw.get("status") == SuggestionStatus.SHOWN:
            calls["n"] += 1
            if calls["n"] == 2:  # 过期之后的第二次读
                raise StorageUnavailable("数据库连接失败")
        return real_list(*args, **kw)

    monkeypatch.setattr(repo, "list", flaky)
    assert service.handle_event("B", _recall("g1_recall", "g1")) is None
    assert {r.status for r in _records(service)} == {SuggestionStatus.EXPIRED}
    assert adapter.sidebar(SHOP_A, CONV) == []


def test_shadow_shop_late_message_does_not_double_count_shadow_comparison():
    service, adapter = _service()
    repo = service.unit.repos.knowledge
    shop = repo.get_shop(SHOP_A)
    assert shop is not None
    overrides = {**shop.config_overrides, "shadow_mode": True}
    repo.upsert_shop(shop.model_copy(update={"config_overrides": overrides}))
    t0 = _ms(service)
    first = service.handle_event(
        "B", taobao_event("什么时候发货", msg_id="w2", send_time_ms=t0 + 60_000)
    )
    assert first is not None and first.suggestions
    records = _records(service)

    late = service.handle_event("B", taobao_event("在吗", msg_id="w1", send_time_ms=t0))

    assert late is not None and late.suggestions
    assert _records(service) == records
    assert {r.status for r in records} == {SuggestionStatus.SHADOW}
    assert adapter.sidebar(SHOP_A, CONV) == []  # 影子模式什么都不展示
    report = shadow_report(_records(service), {"w2": "亲，48 小时内发货"})
    assert report.groups == len({(r.msg_id, r.intent) for r in records})


def test_recall_tombstone_expiry_follows_the_unit_clock():
    """撤回先到留下的标记按单元的时钟过期（组装时把时钟传给会话仓储）：
    过了保留期才到的同 ID 消息照常保存、照常出推荐。"""
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    clock = service.unit.clock
    service.handle_event("B", _recall("r_early", "b_late"))
    clock.set(clock.now() + RECALL_TOMBSTONE_TTL + timedelta(hours=1))
    result = service.handle_event(
        "B", taobao_event("什么时候发货", msg_id="b_late", send_time_ms=_ms(service))
    )
    assert result is not None and result.suggestions
    stored = service.unit.repos.conversations.recent_messages(SHOP_A, CONV)
    assert [m.msg_id for m in stored] == ["b_late"]
