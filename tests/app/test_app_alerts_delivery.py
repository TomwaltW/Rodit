"""提醒送达客服（§5.6 推荐卡片 + 标红提醒、§5.7 转人工提示）：引擎给出的 alerts 和"为什么没有推荐"
要和推荐卡片一起推到侧边栏，而不是只留在 EngineResult 里。

- 适配器（BaseMockAdapter）：show_suggestions 带 alerts / notice，和卡片一起替换上一批；
- 应用服务（AssistService）：把 result.alerts 和 result.no_suggestion_reason 交给适配器；影子模式
  什么都不展示；撤回刷新侧边栏时清掉提醒。全部合成数据。
"""

from __future__ import annotations

import pytest

from rodit.adapters import StandaloneAssistAdapter, TaobaoMockAdapter
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.core.enums import DraftKind, Intent, Mode, Platform
from rodit.core.models import EngineResult, SuggestionCard
from rodit.core.pii import BuyerRefHasher
from rodit.testing.fakes import FakeClock, InMemoryToolGateway
from rodit.testing.fixtures import SHOP_A, TEST_SALT, build_demo_world

CONV = "conv_demo"


@pytest.fixture
def service() -> AssistService:
    return AssistService(build_demo_unit(Platform.TAOBAO))


def _plugin(service: AssistService) -> TaobaoMockAdapter:
    adapter = service.unit.adapters.get(Platform.TAOBAO, "B")
    assert isinstance(adapter, TaobaoMockAdapter)
    return adapter


def _card(text: str) -> SuggestionCard:
    return SuggestionCard(
        suggestion_id=1,
        text=text,
        intent=Intent.SHIPPING_POLICY,
        kind=DraftKind.FAQ,
        confidence="high",
    )


# ================================================================ 适配器


def test_adapter_stores_alerts_and_notice_with_the_cards():
    world = build_demo_world()
    clock = FakeClock(world.now)
    plugin = TaobaoMockAdapter(
        InMemoryToolGateway(world), clock, BuyerRefHasher(TEST_SALT), access="B"
    )
    plugin.on_event(taobao_event("在吗", msg_id="a1"))
    plugin.show_suggestions(
        SHOP_A,
        CONV,
        [_card("亲，48 小时内发货")],
        alerts=["物流停滞 3 天，建议催查", "  "],
        notice=None,
    )
    assert [c.text for c in plugin.sidebar(SHOP_A, CONV)] == ["亲，48 小时内发货"]
    assert plugin.sidebar_alerts(SHOP_A, CONV) == ["物流停滞 3 天，建议催查"]  # 空白的丢掉
    assert plugin.sidebar_notice(SHOP_A, CONV) is None
    # 新的一批连同提醒一起替换
    plugin.show_suggestions(SHOP_A, CONV, [], notice="买家要求人工，不推荐")
    assert plugin.sidebar(SHOP_A, CONV) == [] and plugin.sidebar_alerts(SHOP_A, CONV) == []
    assert plugin.sidebar_notice(SHOP_A, CONV) == "买家要求人工，不推荐"
    # 旧的三参数调用：清掉提醒
    plugin.show_suggestions(SHOP_A, CONV, [_card("x")])
    assert plugin.sidebar_alerts(SHOP_A, CONV) == [] and plugin.sidebar_notice(SHOP_A, CONV) is None
    # 提醒里万一带了个人信息，展示前做最小化
    plugin.show_suggestions(SHOP_A, CONV, [], alerts=["买家留了手机号 13812345678"])
    [alert] = plugin.sidebar_alerts(SHOP_A, CONV)
    assert "13812345678" not in alert
    # 没见过的会话不展示
    plugin.show_suggestions(SHOP_A, "conv_never_seen", [], alerts=["x"], notice="y")
    assert plugin.sidebar_alerts(SHOP_A, "conv_never_seen") == []
    assert plugin.sidebar_notice(SHOP_A, "conv_never_seen") is None


def test_standalone_adapter_shows_alerts_too():
    clock = FakeClock()
    sa = StandaloneAssistAdapter(Platform.PDD, clock, BuyerRefHasher(TEST_SALT))
    msg = sa.on_message({"shop_id": "s_3003", "session_id": "web_1", "question": "在吗"})
    assert msg is not None
    sa.show_suggestions("s_3003", msg.conversation_id, [], alerts=["请优先处理"], notice="原因")
    assert sa.sidebar_alerts("s_3003", msg.conversation_id) == ["请优先处理"]
    assert sa.sidebar_notice("s_3003", msg.conversation_id) == "原因"


# ================================================================ 应用服务


def test_complaint_alert_and_reason_reach_the_sidebar(service: AssistService):
    """买家要人工 / 投诉：没有推荐卡片，但客服要看到标红提醒和原因（§5.7）。"""
    now_ms = int(service.unit.clock.now().timestamp() * 1000)
    first = taobao_event("我要投诉你们，转人工", msg_id="al1", send_time_ms=now_ms - 60_000)
    result = service.handle_event("B", first)
    assert result is not None and result.suggestions == []
    assert result.alerts and result.no_suggestion_reason
    plugin = _plugin(service)
    assert plugin.sidebar(SHOP_A, CONV) == []
    assert plugin.sidebar_alerts(SHOP_A, CONV) == result.alerts
    assert plugin.sidebar_notice(SHOP_A, CONV) == result.no_suggestion_reason
    # 下一条正常的消息：提醒和原因随新的一批替换掉
    nxt = service.handle_event(
        "B", taobao_event("什么时候发货", msg_id="al2", send_time_ms=now_ms)
    )  # 隔一分钟：不和上一条合并（§5.3 合并连发）
    assert nxt is not None and nxt.suggestions and not nxt.alerts
    assert plugin.sidebar(SHOP_A, CONV) == nxt.suggestions
    assert plugin.sidebar_alerts(SHOP_A, CONV) == []
    assert plugin.sidebar_notice(SHOP_A, CONV) is None


def test_alerts_travel_with_cards(service: AssistService, monkeypatch):
    """有推荐也有提醒（如物流异常）：两者一起展示。"""
    engine = service.unit.engine
    real = engine.handle

    def with_alert(msg):
        result = real(msg)
        return result.model_copy(update={"alerts": ["物流停滞超过 48 小时，建议向快递催查"]})

    monkeypatch.setattr(engine, "handle", with_alert)
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="al3"))
    assert result is not None and result.suggestions
    plugin = _plugin(service)
    assert plugin.sidebar(SHOP_A, CONV) == result.suggestions
    assert plugin.sidebar_alerts(SHOP_A, CONV) == ["物流停滞超过 48 小时，建议向快递催查"]


def test_shadow_mode_shows_nothing(service: AssistService, monkeypatch):
    engine = service.unit.engine
    real = engine.handle

    def shadow(msg):
        result: EngineResult = real(msg)
        return result.model_copy(
            update={"mode": Mode.SHADOW, "alerts": ["不该出现"], "no_suggestion_reason": "不该出现"}
        )

    monkeypatch.setattr(engine, "handle", shadow)
    service.handle_event("B", taobao_event("什么时候发货", msg_id="al4"))
    plugin = _plugin(service)
    assert plugin.sidebar(SHOP_A, CONV) == []
    assert plugin.sidebar_alerts(SHOP_A, CONV) == [] and plugin.sidebar_notice(SHOP_A, CONV) is None


def test_recall_refresh_clears_alerts(service: AssistService):
    """撤回：那条消息的推荐过期、侧边栏刷新，提醒也一起清掉。"""
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="al5"))
    assert result is not None and result.suggestions
    plugin = _plugin(service)
    plugin.show_suggestions(SHOP_A, CONV, result.suggestions, alerts=["旧提醒"], notice=None)
    recall = {
        "seller_id": SHOP_A,
        "session_id": CONV,
        "msg_id": "al5_recall",
        "msg_type": "recall",
        "recall_msg_id": "al5",
    }
    assert service.handle_event("B", recall) is None
    assert plugin.sidebar(SHOP_A, CONV) == []
    assert plugin.sidebar_alerts(SHOP_A, CONV) == []


def test_service_passes_alerts_as_keywords(service: AssistService, monkeypatch):
    """平台云内的适配器实现也按端口签名收到 alerts / notice（关键字参数）。"""
    calls: list[dict] = []
    plugin = _plugin(service)

    def record(shop_id, conversation_id, cards, *, alerts=(), notice=None):
        calls.append({"cards": cards, "alerts": list(alerts), "notice": notice})

    monkeypatch.setattr(plugin, "show_suggestions", record)
    result = service.handle_event("B", taobao_event("我要投诉你们，转人工", msg_id="al6"))
    assert result is not None
    assert calls == [{"cards": [], "alerts": result.alerts, "notice": result.no_suggestion_reason}]
