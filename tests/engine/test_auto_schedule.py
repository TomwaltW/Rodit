"""全自动先在非工作时间试行（方案 §15 Phase 3、§5.6、§4.1 原则 3、§12.2）：
生效模式按客服工作时间判断。

- 消息是 AUTO、店铺开了全自动：auto_reply_schedule=always → 全自动；off_hours（默认）→ 只在客服
  工作时间（北京时间）以外全自动，工作时间内 / 没填工作时间 → 退回智能辅助并转人工，engine.result
  另加 auto_gate（没有买家原文）；
- 边界：开始那一分钟是工作时间，结束那一分钟不是；跨午夜的时段；时钟是 UTC 也按北京时间算；
- 路由的 auto_eligible 和生效模式一致（真实的 DecisionRouter + RoutingPolicy）。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from engine_support import Kit, make_route

from rodit.core.audit import RESULT_DETAIL_KEYS, AuditAction, InMemoryAuditSink
from rodit.core.clock import CST
from rodit.core.config import RoutingConfig
from rodit.core.enums import Intent, Mode, RouteAction
from rodit.core.models import ShopConfig
from rodit.decision import RuleProvider
from rodit.engine import ConversationBuilder, effective_mode
from rodit.engine.modes import (
    AUTO_GATE_NO_SERVICE_HOURS,
    AUTO_GATE_SERVICE_HOURS,
    AUTO_GATE_UNKNOWN_TIME,
    AUTO_GATES,
    RESULT_AUTO_GATE,
    auto_gate,
    resolve_mode,
)
from rodit.routing import DecisionRouter, RoutingPolicy
from rodit.testing.fakes import FakeDecisionProvider, router_answers
from rodit.testing.fixtures import BUYER_DOUYIN, SHOP_A, SHOP_B, SHOP_D

PREFIX = "【智能客服】"
DAY = ["09:00-23:00"]
NIGHT = ["22:00-02:00"]


def cst(hour: int, minute: int = 0, second: int = 0, *, day: int = 28) -> datetime:
    return datetime(2026, 9, day, hour, minute, second, tzinfo=CST)


def config(**kw: Any) -> ShopConfig:
    kw.setdefault("auto_reply_enabled", True)
    return ShopConfig(**kw)


# ================================================================ 规则（engine.modes）


@pytest.mark.parametrize(
    ("kw", "msg_mode", "at", "expected"),
    [
        # 默认 off_hours：工作时间内拦下，工作时间外全自动
        ({"service_hours": DAY}, Mode.AUTO, cst(10, 35), (Mode.ASSIST, AUTO_GATE_SERVICE_HOURS)),
        ({"service_hours": DAY}, Mode.AUTO, cst(23, 30), (Mode.AUTO, None)),
        ({"service_hours": DAY}, Mode.AUTO, cst(3), (Mode.AUTO, None)),
        # 边界：开始那一分钟算工作时间，结束那一分钟不算
        ({"service_hours": DAY}, Mode.AUTO, cst(9), (Mode.ASSIST, AUTO_GATE_SERVICE_HOURS)),
        ({"service_hours": DAY}, Mode.AUTO, cst(8, 59, 59), (Mode.AUTO, None)),
        ({"service_hours": DAY}, Mode.AUTO, cst(22, 59, 59), (Mode.ASSIST, "service_hours")),
        ({"service_hours": DAY}, Mode.AUTO, cst(23), (Mode.AUTO, None)),
        # 跨午夜的时段（夜班客服）
        ({"service_hours": NIGHT}, Mode.AUTO, cst(23, 30), (Mode.ASSIST, "service_hours")),
        ({"service_hours": NIGHT}, Mode.AUTO, cst(1, 59), (Mode.ASSIST, "service_hours")),
        ({"service_hours": NIGHT}, Mode.AUTO, cst(2), (Mode.AUTO, None)),
        ({"service_hours": NIGHT}, Mode.AUTO, cst(12), (Mode.AUTO, None)),
        # 多个时段：中间休息时全自动
        (
            {"service_hours": ["08:00-12:00", "13:30-22:00"]},
            Mode.AUTO,
            cst(12, 30),
            (Mode.AUTO, None),
        ),
        (
            {"service_hours": ["08:00-12:00", "13:30-22:00"]},
            Mode.AUTO,
            cst(13, 30),
            (Mode.ASSIST, "service_hours"),
        ),
        # 全天都是工作时间：永远不自动
        ({"service_hours": ["00:00-24:00"]}, Mode.AUTO, cst(3), (Mode.ASSIST, "service_hours")),
        # 没填工作时间：不知道客服在不在，不自动
        ({}, Mode.AUTO, cst(3), (Mode.ASSIST, AUTO_GATE_NO_SERVICE_HOURS)),
        ({}, Mode.AUTO, None, (Mode.ASSIST, AUTO_GATE_NO_SERVICE_HOURS)),
        # 不知道处理时刻
        ({"service_hours": DAY}, Mode.AUTO, None, (Mode.ASSIST, AUTO_GATE_UNKNOWN_TIME)),
        # always：全天全自动，不看工作时间（也不需要填）
        ({"auto_reply_schedule": "always"}, Mode.AUTO, cst(10), (Mode.AUTO, None)),
        (
            {"auto_reply_schedule": "always", "service_hours": DAY},
            Mode.AUTO,
            cst(10),
            (Mode.AUTO, None),
        ),
        ({"auto_reply_schedule": "always"}, Mode.AUTO, None, (Mode.AUTO, None)),
        # 总开关没开、消息不是 AUTO：智能辅助，没有拦下原因（本来就不是全自动）
        (
            {"auto_reply_enabled": False, "service_hours": DAY},
            Mode.AUTO,
            cst(3),
            (Mode.ASSIST, None),
        ),
        ({"service_hours": DAY}, Mode.ASSIST, cst(3), (Mode.ASSIST, None)),
        ({"service_hours": DAY}, Mode.SHADOW, cst(3), (Mode.ASSIST, None)),
        # 影子模式优先于一切
        ({"service_hours": DAY, "shadow_mode": True}, Mode.AUTO, cst(3), (Mode.SHADOW, None)),
        ({"shadow_mode": True}, Mode.AUTO, cst(10), (Mode.SHADOW, None)),
    ],
)
def test_resolve_mode_matrix(kw, msg_mode, at, expected) -> None:
    decided = resolve_mode(msg_mode, config(**kw), at)
    assert (decided.mode, decided.auto_gate) == expected
    assert effective_mode(msg_mode, config(**kw), at) == expected[0]
    assert decided.auto_gate is None or decided.auto_gate in AUTO_GATES


@pytest.mark.parametrize(
    ("utc", "inside"),
    [
        (datetime(2026, 9, 28, 0, 59, tzinfo=UTC), False),  # 北京 08:59
        (datetime(2026, 9, 28, 1, 0, tzinfo=UTC), True),  # 北京 09:00
        (datetime(2026, 9, 28, 14, 59, tzinfo=UTC), True),  # 北京 22:59
        (datetime(2026, 9, 28, 15, 0, tzinfo=UTC), False),  # 北京 23:00
        (datetime(2026, 9, 28, 20, 0, tzinfo=UTC), False),  # 北京次日 04:00（UTC 还是当天）
    ],
)
def test_utc_clock_is_judged_in_beijing_time(utc: datetime, inside: bool) -> None:
    gate = auto_gate(config(service_hours=DAY), utc)
    assert gate == (AUTO_GATE_SERVICE_HOURS if inside else None)


def test_effective_mode_without_time_never_goes_auto_off_hours() -> None:
    assert effective_mode(Mode.AUTO, config(service_hours=DAY)) == Mode.ASSIST
    assert effective_mode(Mode.AUTO, config(auto_reply_schedule="always")) == Mode.AUTO


# ================================================================ 会话理解


def test_builder_judges_mode_and_context_at_the_same_instant(kit: Kit) -> None:
    kit.set_shop_config(SHOP_A, auto_reply_enabled=True, service_hours=DAY)
    builder = ConversationBuilder(
        kit.knowledge, kit.conversations, kit.tools, kit.config, kit.clock
    )
    kit.clock.set(cst(22, 59))
    msg = kit.msg("在吗", mode=Mode.AUTO)  # 拨到 23:00:00 —— 工作时间外
    kit.conversations.add_message(msg)
    und = builder.build_with_notes(msg)
    assert und.context.now == cst(23) and und.context.mode == Mode.AUTO
    assert und.auto_gate is None
    kit.clock.set(cst(9))
    msg = kit.msg("在吗", mode=Mode.AUTO, after_s=0)
    kit.conversations.add_message(msg)
    und = builder.build_with_notes(msg)
    assert und.context.mode == Mode.ASSIST and und.auto_gate == AUTO_GATE_SERVICE_HOURS


# ================================================================ 引擎


def enable(kit: Kit, **overrides: Any) -> None:
    overrides.setdefault("service_hours", DAY)
    kit.set_shop_config(SHOP_A, auto_reply_enabled=True, **overrides)
    kit.router.decide = make_route(Intent.SHIPPING_POLICY, confidence=0.97, auto_eligible=True)


def at(kit: Kit, when: datetime, text: str = "什么时候发货", **kw: Any):
    kit.clock.set(when)
    kw.setdefault("mode", Mode.AUTO)
    result = kit.engine().handle(kit.msg(text, after_s=0, **kw))
    assert result is not None
    return result


def result_detail(kit: Kit, shop_id: str = SHOP_A) -> dict[str, Any]:
    return kit.result_events(shop_id)[-1].detail


def test_in_service_hours_falls_back_to_assist_and_transfers(kit: Kit) -> None:
    enable(kit)
    result = at(kit, cst(10, 35))
    assert result.mode == Mode.ASSIST
    assert result.auto_reply is None
    assert [c.text for c in result.suggestions] == ["亲，发货与快递：现货付款后 48 小时内发货"]
    assert result.transfer_to_human is True  # 机器人接待的会话交给在岗客服
    assert kit.events(AuditAction.AUTO_REPLY) == []
    detail = result_detail(kit)
    assert detail[RESULT_AUTO_GATE] == AUTO_GATE_SERVICE_HOURS
    assert set(detail) == {*RESULT_DETAIL_KEYS, RESULT_AUTO_GATE}
    assert detail["transfer_to_human"] is True
    [understand] = kit.events(AuditAction.UNDERSTAND)
    assert understand.detail["mode"] == "assist"
    record = kit.suggestions.list(SHOP_A)[-1]
    assert record.status.value == "shown"  # 推荐照常记下、展示给客服


def test_off_hours_auto_replies(kit: Kit) -> None:
    enable(kit)
    result = at(kit, cst(23, 30))
    assert result.mode == Mode.AUTO
    assert result.auto_reply is not None and result.auto_reply.text.startswith(PREFIX)
    assert result.transfer_to_human is False
    detail = result_detail(kit)
    assert RESULT_AUTO_GATE not in detail and tuple(detail) == RESULT_DETAIL_KEYS


@pytest.mark.parametrize(
    ("when", "auto"),
    [
        (cst(8, 59, 59), True),
        (cst(9), False),
        (cst(22, 59, 59), False),
        (cst(23), True),
    ],
)
def test_boundary_minutes(kit: Kit, when: datetime, auto: bool) -> None:
    enable(kit)
    result = at(kit, when)
    assert (result.auto_reply is not None) is auto
    assert result.mode == (Mode.AUTO if auto else Mode.ASSIST)
    assert result.transfer_to_human is not auto


def test_midnight_crossing_window(kit: Kit) -> None:
    enable(kit, service_hours=NIGHT)
    assert at(kit, cst(1, 30)).mode == Mode.ASSIST
    assert at(kit, cst(2, 0), text="发什么快递").mode == Mode.AUTO
    assert at(kit, cst(15, 0), text="还有货吗").mode == Mode.AUTO
    assert at(kit, cst(22, 0), text="在吗").mode == Mode.ASSIST


def test_utc_engine_clock_uses_beijing_service_hours(kit: Kit) -> None:
    """服务器时钟是 UTC：UTC 14:30 是北京 22:30（工作时间内），UTC 15:30 是北京 23:30。"""
    enable(kit)
    inside = at(kit, datetime(2026, 9, 28, 14, 30, tzinfo=UTC))
    assert inside.mode == Mode.ASSIST and inside.transfer_to_human
    outside = at(kit, datetime(2026, 9, 28, 15, 30, tzinfo=UTC), text="发什么快递")
    assert outside.mode == Mode.AUTO and outside.auto_reply is not None


def test_no_service_hours_never_auto(kit: Kit) -> None:
    enable(kit, service_hours=[])
    for when in (cst(3), cst(10), cst(23, 30)):
        result = at(kit, when)
        assert result.mode == Mode.ASSIST and result.auto_reply is None
        assert result.transfer_to_human is True
        assert result_detail(kit)[RESULT_AUTO_GATE] == AUTO_GATE_NO_SERVICE_HOURS


def test_always_schedule_ignores_service_hours(kit: Kit) -> None:
    enable(kit, auto_reply_schedule="always")
    result = at(kit, cst(10, 35))
    assert result.mode == Mode.AUTO and result.auto_reply is not None
    assert RESULT_AUTO_GATE not in result_detail(kit)


def test_assist_messages_are_untouched_by_the_schedule(kit: Kit) -> None:
    """客服接待的消息（不是机器人接待）：本来就是智能辅助，不转人工、不记 auto_gate。"""
    enable(kit)
    result = at(kit, cst(10, 35), mode=Mode.ASSIST)
    assert result.mode == Mode.ASSIST and result.transfer_to_human is False
    assert RESULT_AUTO_GATE not in result_detail(kit)


def test_shop_switch_off_is_plain_assist(kit: Kit) -> None:
    """总开关没开：和以前一样只按智能辅助处理，不是"被时段拦下"。"""
    kit.set_shop_config(SHOP_A, service_hours=DAY)
    result = at(kit, cst(23, 30))
    assert result.mode == Mode.ASSIST and result.transfer_to_human is False
    assert RESULT_AUTO_GATE not in result_detail(kit)


def test_shadow_shop_is_shadow_regardless_of_schedule(kit: Kit) -> None:
    kit.set_shop_config(SHOP_D, auto_reply_enabled=True, service_hours=DAY)
    kit.router.decide = make_route(Intent.SHIPPING_POLICY, confidence=0.97, auto_eligible=True)
    kit.clock.set(cst(10, 35))
    msg = kit.msg("什么时候发货", shop_id=SHOP_D, buyer_ref=BUYER_DOUYIN, mode=Mode.AUTO, after_s=0)
    result = kit.engine().handle(msg)
    assert result is not None and result.mode == Mode.SHADOW
    assert result.transfer_to_human is False and result.auto_reply is None
    assert RESULT_AUTO_GATE not in result_detail(kit, SHOP_D)


def test_gate_is_per_shop(kit: Kit) -> None:
    """店铺 A 在工作时间内拦下，不影响店铺 B 的配置（B 全天全自动），各记各的（§9.2）。"""
    enable(kit)
    kit.set_shop_config(SHOP_B, auto_reply_enabled=True, auto_reply_schedule="always")
    a = at(kit, cst(10, 35))
    b = at(kit, cst(10, 36), shop_id=SHOP_B, conversation_id="c_b", buyer_ref="BUYER_B")
    assert a.mode == Mode.ASSIST and b.mode == Mode.AUTO
    assert result_detail(kit)[RESULT_AUTO_GATE] == AUTO_GATE_SERVICE_HOURS
    assert RESULT_AUTO_GATE not in result_detail(kit, SHOP_B)


def test_gated_message_that_fails_later_still_transfers(kit: Kit) -> None:
    """拦下之后会话理解又失败：没有上下文，照样交给在岗客服（不能既不回也不转）。"""
    enable(kit)
    engine = kit.engine()

    def boom(*args: Any, **kw: Any) -> Any:
        raise RuntimeError("boom")

    engine.builder.build_with_notes = boom  # type: ignore[method-assign]
    kit.clock.set(cst(10, 35))
    result = engine.handle(kit.msg("什么时候发货", mode=Mode.AUTO, after_s=0))
    assert result is not None
    assert result.suggestions == [] and result.transfer_to_human is True
    assert result_detail(kit)[RESULT_AUTO_GATE] == AUTO_GATE_SERVICE_HOURS


def test_gated_high_risk_question_follows_assist_rules(kit: Kit) -> None:
    """拦下后路由按智能辅助走：高风险类型给推荐 + 风险提示（自动模式下才是转人工路由）。"""
    enable(kit)
    kit.router.decide = lambda ctx: make_route(
        Intent.PRICE_PROMO,
        RouteAction.TRANSFER_HUMAN if ctx.mode == Mode.AUTO else RouteAction.HIGH_RISK_ANSWER,
        rule=4,
        high_risk=True,
        risk_hints=["涉及价格承诺，请确认"],
    )
    result = at(kit, cst(10, 35), text="能便宜点吗")
    assert result.route is not None and result.route.action == RouteAction.HIGH_RISK_ANSWER
    assert result.suggestions and result.suggestions[0].risk_hints
    assert result.transfer_to_human is True


# ================================================================ 路由的 auto_eligible


def real_router(kit: Kit) -> DecisionRouter:
    provider = FakeDecisionProvider(router_answers("shipping_policy", 0.97))
    return DecisionRouter(
        provider, RuleProvider(), RoutingPolicy(RoutingConfig()), InMemoryAuditSink(), kit.clock
    )


@pytest.mark.parametrize(
    ("when", "mode", "eligible"),
    [(cst(10, 35), Mode.ASSIST, False), (cst(23, 30), Mode.AUTO, True)],
)
def test_route_auto_eligible_agrees_with_effective_mode(
    kit: Kit, when: datetime, mode: Mode, eligible: bool
) -> None:
    kit.set_shop_config(SHOP_A, auto_reply_enabled=True, service_hours=DAY)
    kit.clock.set(when)
    engine = kit.engine(router=real_router(kit))
    result = engine.handle(kit.msg("什么时候发货", mode=Mode.AUTO, after_s=0))
    assert result is not None and result.route is not None
    assert result.mode == mode
    assert result.route.auto_eligible is eligible
    assert (result.auto_reply is not None) is eligible
    [route_event] = kit.events(AuditAction.ROUTE)
    assert route_event.detail["auto_eligible"] is eligible


def test_late_auto_message_keeps_the_gate_of_its_own_turn(kit: Kit) -> None:
    """乱序：迟到的旧消息让引擎重算最新一轮；按处理时刻判断（工作时间内照样拦下、转人工）。"""
    enable(kit)
    kit.clock.set(cst(10, 35))
    engine = kit.engine()
    newer = kit.msg("什么时候发货", mode=Mode.AUTO, after_s=0)
    engine.handle(newer)
    older = newer.model_copy(
        update={"msg_id": "late_1", "received_at": newer.received_at - timedelta(seconds=30)}
    )
    again = engine.handle(older)
    assert again is not None and again.msg_ids == [newer.msg_id]
    assert again.mode == Mode.ASSIST and again.transfer_to_human is True
    assert result_detail(kit)[RESULT_AUTO_GATE] == AUTO_GATE_SERVICE_HOURS
