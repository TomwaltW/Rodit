"""上线节奏里的半自动门槛和全自动生效时段（方案 §13.4 半自动 "该类型原样采纳率 ≥ 90%"、§5.6、
§15 Phase 3 先在非工作时间试行）：纯逻辑。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import count

import pytest

from rodit.core.clock import CST
from rodit.core.config import RoutingConfig
from rodit.core.enums import Intent, SuggestionStatus
from rodit.core.models import ShopConfig, SuggestionRecord
from rodit.feedback.rollout import (
    SEMI_AUTO_HIGH_RISK,
    SEMI_AUTO_MIN_GROUPS,
    SEMI_AUTO_NOT_ANSWER,
    SEMI_AUTO_PERIODS,
    AutoSchedule,
    next_service_switch,
    rollout_status,
    semi_auto_blocked,
    semi_auto_gate,
    semi_auto_status,
    shadow_comparison,
    shadow_window,
    weekly_adoption,
)
from rodit.testing.fixtures import CONV, DEFAULT_NOW, SHOP_A, SHOP_B

S = SuggestionStatus
SHIP, SIZE, INVOICE = Intent.SHIPPING_POLICY, Intent.SIZE_ADVICE, Intent.INVOICE_GIFT
NOW = DEFAULT_NOW  # 2026-09-28（周一）10:35，W40
W40 = datetime(2026, 9, 28, tzinfo=CST)
_ids = count(1)


def week(k: int) -> datetime:
    return W40 - timedelta(days=7 * k)


def groups(
    start: datetime,
    n: int,
    as_is: int,
    *,
    intent: Intent = SHIP,
    other: SuggestionStatus = S.EDITED_SENT,
    shop_id: str = SHOP_A,
) -> list[SuggestionRecord]:
    """n 个推荐组（每组一张卡片），前 as_is 个原样发送，其余按 other。"""
    out = []
    for i in range(n):
        sid = next(_ids)
        out.append(
            SuggestionRecord(
                id=sid,
                shop_id=shop_id,
                conversation_id=CONV,
                msg_id=f"m{sid}",
                intent=intent,
                text="亲，48 小时内发货哦",
                status=S.SENT if i < as_is else other,
                created_at=start + timedelta(hours=1, minutes=i),
            )
        )
    return out


def weekly(records: list[SuggestionRecord]):
    return weekly_adoption(records, NOW, weeks=SEMI_AUTO_PERIODS)


# ================================================================ 门槛


def test_gate_passes_with_two_weeks_of_90_percent_as_is():
    records = [*groups(week(2), 10, 9), *groups(week(1), 20, 19)]
    gate = semi_auto_gate(weekly(records), SHIP)
    assert gate.passed and gate.stage == "semi_auto:shipping_policy"
    assert gate.unchecked == []  # 不看投诉（半自动仍由客服按发送）
    rates = [c.value for c in gate.checks if c.name.startswith("period_")]
    assert rates == [0.9, 0.95]


@pytest.mark.parametrize(
    ("records", "failed"),
    [
        # 第 1 周只有 89%
        (
            lambda: [*groups(week(2), 100, 89), *groups(week(1), 20, 20)],
            "period_1.group_as_is_rate",
        ),
        # 推荐组不够（每周至少 10 个）
        (lambda: [*groups(week(2), 9, 9), *groups(week(1), 20, 20)], "period_1.group_as_is_rate"),
        # 第 2 周没有这类推荐
        (lambda: groups(week(2), 20, 20), "period_2.group_as_is_rate"),
    ],
)
def test_gate_fails(records, failed):
    gate = semi_auto_gate(weekly(records()), SHIP)
    assert not gate.passed
    assert failed in [c.name for c in gate.checks if not c.passed]


def test_gate_counts_only_this_intent_and_this_window():
    records = [
        *groups(week(2), 20, 20, intent=SIZE),
        *groups(week(1), 20, 20, intent=SIZE),
        *groups(week(3), 20, 20),  # 窗口之前
        *groups(W40, 20, 20),  # 本周（还没过完）
    ]
    assert semi_auto_gate(weekly(records), SIZE).passed
    assert not semi_auto_gate(weekly(records), SHIP).passed


def test_rejected_and_expired_are_not_as_is():
    rejected = [*groups(week(2), 20, 17, other=S.REJECTED), *groups(week(1), 20, 20)]
    assert not semi_auto_gate(weekly(rejected), SHIP).passed


def test_min_groups_is_configurable():
    records = [*groups(week(2), 5, 5), *groups(week(1), 5, 5)]
    assert not semi_auto_gate(weekly(records), SHIP).passed
    assert semi_auto_gate(weekly(records), SHIP, min_groups=5).passed


def test_more_weeks_of_reports_use_the_last_two():
    records = [*groups(week(4), 20, 0), *groups(week(2), 20, 19), *groups(week(1), 20, 19)]
    four = weekly_adoption(records, NOW, weeks=4)
    assert semi_auto_gate(four, SHIP).passed


# ================================================================ 能不能开


@pytest.mark.parametrize("intent", RoutingConfig().high_risk)
def test_high_risk_intents_are_blocked(intent):
    assert semi_auto_blocked(intent, RoutingConfig()) == SEMI_AUTO_HIGH_RISK


def test_complaint_is_blocked_even_if_not_listed_as_high_risk():
    assert semi_auto_blocked(Intent.COMPLAINT, RoutingConfig(high_risk=[])) == SEMI_AUTO_HIGH_RISK


@pytest.mark.parametrize("intent", [Intent.OFF_PLATFORM, Intent.OTHER])
def test_non_answer_intents_are_blocked(intent):
    assert semi_auto_blocked(intent, RoutingConfig()) == SEMI_AUTO_NOT_ANSWER


@pytest.mark.parametrize("intent", [SHIP, INVOICE, Intent.GREETING, Intent.LOGISTICS_TRACK])
def test_simple_intents_are_allowed(intent):
    assert semi_auto_blocked(intent, RoutingConfig()) is None


def test_status_lists_every_intent_with_gates_only_for_allowed_ones():
    records = [*groups(week(2), 20, 19), *groups(week(1), 20, 19)]
    config = ShopConfig(semi_auto_intents=[INVOICE, SHIP, Intent.PRICE_PROMO])
    status = semi_auto_status(weekly(records), config, RoutingConfig())
    assert status.enabled == [Intent.PRICE_PROMO, SHIP, INVOICE]  # 按问题类型的顺序
    assert [r.intent for r in status.intents] == list(Intent)
    rows = {r.intent: r for r in status.intents}
    assert rows[SHIP].enabled and rows[SHIP].allowed and rows[SHIP].gate.passed
    assert rows[SIZE].gate is not None and not rows[SIZE].gate.passed and not rows[SIZE].enabled
    # 高风险类型被直接写进了配置：照样报告为开着、不允许（控制台提示关闭）
    assert rows[Intent.PRICE_PROMO].enabled and not rows[Intent.PRICE_PROMO].allowed
    assert rows[Intent.PRICE_PROMO].gate is None
    assert rows[Intent.PRICE_PROMO].blocked_reason == SEMI_AUTO_HIGH_RISK
    assert status.threshold == RoutingConfig().semi_auto_threshold
    assert status.min_groups == SEMI_AUTO_MIN_GROUPS and status.periods == SEMI_AUTO_PERIODS


def test_rollout_status_carries_semi_auto_and_schedule():
    records = [*groups(week(2), 20, 19), *groups(week(1), 20, 19)]
    schedule = AutoSchedule(auto_reply_enabled=False, auto_reply_schedule="off_hours")
    status = rollout_status(
        shop_id=SHOP_A,
        config=ShopConfig(semi_auto_intents=[SHIP]),
        routing=RoutingConfig(),
        now=NOW,
        records=[*records, *groups(week(1), 20, 0, shop_id=SHOP_B)],
        shadow=shadow_comparison(records, [], since=shadow_window(NOW)),
        shadow_since=shadow_window(NOW),
        reviews=[],
        auto_schedule=schedule,
    )
    assert status.auto_schedule == schedule
    assert status.semi_auto is not None and status.semi_auto.enabled == [SHIP]
    ship = next(r for r in status.semi_auto.intents if r.intent == SHIP)
    assert ship.gate is not None and ship.gate.passed  # 别的店铺的推荐不算（§9.2）
    assert "48 小时" not in status.model_dump_json()  # 没有推荐文字


# ================================================================ 下一次进出工作时间


def cst(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=CST)


@pytest.mark.parametrize(
    ("lines", "now", "expected"),
    [
        (["09:00-23:00"], cst(28, 10, 35), cst(28, 23)),
        (["09:00-23:00"], cst(28, 23, 30), cst(29, 9)),
        (["09:00-23:00"], cst(28, 3), cst(28, 9)),
        (["09:00-23:00"], cst(28, 9), cst(28, 23)),  # 正好在开始那一分钟：下一次是离开
        (["09:00-23:00"], cst(28, 22, 59), cst(28, 23)),
        (["22:00-02:00"], cst(28, 23), cst(29, 2)),
        (["22:00-02:00"], cst(28, 1), cst(28, 2)),
        (["22:00-02:00"], cst(28, 12), cst(28, 22)),
        (["08:00-12:00", "13:30-22:00"], cst(28, 12, 15), cst(28, 13, 30)),
        (["08:00-12:00", "12:00-22:00"], cst(28, 11), cst(28, 22)),  # 首尾相接不算切换
        (["00:00-24:00"], cst(28, 10), None),
        ([], cst(28, 10), None),
    ],
)
def test_next_service_switch(lines, now, expected):
    assert next_service_switch(lines, now) == expected


def test_next_service_switch_with_a_utc_clock():
    now = datetime(2026, 9, 28, 14, 30, tzinfo=UTC)  # 北京 22:30
    assert next_service_switch(["09:00-23:00"], now) == cst(28, 23)
