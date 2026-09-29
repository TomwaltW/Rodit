"""上线节奏的纯逻辑（方案 §13.4）：当前阶段、采纳率周报与门槛、影子对比、影子 → 智能辅助的门槛、
汇总建议与需要人工确认的条件。"""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import count

import pytest

from rodit.core.clock import CST
from rodit.core.config import RoutingConfig
from rodit.core.enums import Intent, QualityReviewStatus, Role, SuggestionStatus
from rodit.core.models import (
    QualityReview,
    QualityScores,
    ShopConfig,
    StoredMessage,
    SuggestionRecord,
)
from rodit.feedback import (
    RolloutAdvice,
    RolloutStage,
    launch_gate,
    rollout_status,
    shadow_comparison,
    shadow_gate,
    stage_of,
)
from rodit.feedback.rollout import (
    ASSIST_MIN_GROUPS,
    assist_gate,
    auto_intent_gates,
    gate_summary,
    shadow_reviews,
    shadow_window,
    week_windows,
    weekly_adoption,
)
from rodit.testing.fixtures import CONV, DEFAULT_NOW, SHOP_A, SHOP_B

S = SuggestionStatus
SHIP, SIZE = Intent.SHIPPING_POLICY, Intent.SIZE_ADVICE
NOW = DEFAULT_NOW  # 2026-09-28（周一）10:35，W40
W40 = datetime(2026, 9, 28, tzinfo=CST)
_ids = count(1)


def week(k: int) -> datetime:
    """当前周往前第 k 周的周一 00:00。"""
    return W40 - timedelta(days=7 * k)


def groups(
    start: datetime,
    n: int,
    adopted: int,
    *,
    intent: Intent = SHIP,
    adopted_status: SuggestionStatus = S.SENT,
    other: SuggestionStatus = S.REJECTED,
    shop_id: str = SHOP_A,
) -> list[SuggestionRecord]:
    """n 个推荐组（每组一张卡片），前 adopted 个被采纳。"""
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
                status=adopted_status if i < adopted else other,
                created_at=start + timedelta(hours=1, minutes=i),
            )
        )
    return out


def shadow_rec(sid: int, at: datetime, *, msg_id: str, intent: Intent = SHIP, text: str = ""):
    return SuggestionRecord(
        id=sid,
        shop_id=SHOP_A,
        conversation_id=CONV,
        msg_id=msg_id,
        intent=intent,
        text=text or "亲，现货付款后 48 小时内发货哦~",
        status=S.SHADOW,
        created_at=at,
    )


def review(
    rid: int,
    suggestion_id: int,
    *,
    factual: int = 2,
    compliance: int = 2,
    period: str = "2026-W39",
    status: QualityReviewStatus = QualityReviewStatus.SCORED,
    reviewed_at: datetime | None = None,
) -> QualityReview:
    done = status != QualityReviewStatus.PENDING
    return QualityReview(
        id=rid,
        shop_id=SHOP_A,
        period=period,
        suggestion_id=suggestion_id,
        intent=SHIP,
        status=status,
        scores=QualityScores(factual=factual, relevance=2, compliance=compliance, expression=2)
        if status == QualityReviewStatus.SCORED
        else None,
        reviewer="qa_reviewer_01" if done else None,
        sampled_at=NOW,
        reviewed_at=(reviewed_at or NOW) if done else None,
    )


def msg(msg_id: str, role: Role, text: str, minutes: int) -> StoredMessage:
    return StoredMessage(
        msg_id=msg_id,
        shop_id=SHOP_A,
        conversation_id=CONV,
        role=role,
        text=text,
        created_at=week(1) + timedelta(minutes=minutes),
    )


# ================================================================ 阶段


def test_stage_follows_effective_mode_priority():
    assert stage_of(ShopConfig()) == RolloutStage.ASSIST
    assert stage_of(ShopConfig(auto_reply_enabled=True)) == RolloutStage.AUTO
    # 影子模式优先（和 engine.understanding.effective_mode 一致）
    assert stage_of(ShopConfig(shadow_mode=True, auto_reply_enabled=True)) == RolloutStage.SHADOW


def test_week_windows_are_consecutive_full_weeks():
    windows = week_windows(NOW, 4)
    assert [p for p, _, _ in windows] == ["2026-W36", "2026-W37", "2026-W38", "2026-W39"]
    assert windows[-1][2] == W40  # 截至上一个完整周
    assert all(windows[i][2] == windows[i + 1][1] for i in range(3))
    assert shadow_window(NOW) == week(2)


# ================================================================ 智能辅助 → 下一阶段（§13.4）


def test_assist_gate_needs_two_consecutive_weeks_at_sixty_percent():
    records = [
        *groups(week(3), 30, 0),  # W37 很差，但只看最近 2 周
        *groups(week(2), 25, 15),  # W38 60%（含）
        *groups(week(1), 30, 20),  # W39 67%
        *groups(W40, 5, 0),  # 本周不算（不是完整周）
    ]
    weekly = weekly_adoption(records, NOW)
    assert [w.since for w in weekly] == [week(4), week(3), week(2), week(1)]
    gate = assist_gate(weekly)
    assert gate.stage == "assist" and gate.passed
    assert gate.unchecked == ["platform_violations"]  # 没有数据来源，要人工确认
    assert {c.name: c.passed for c in gate.checks} == {
        "periods": True,
        "consecutive": True,
        "period_1.group_adoption_rate": True,
        "period_2.group_adoption_rate": True,
    }
    assert assist_gate(weekly, platform_violations=0).unchecked == []
    assert not assist_gate(weekly, platform_violations=1).passed


def test_assist_gate_fails_below_threshold_or_with_too_few_groups():
    low = weekly_adoption([*groups(week(2), 25, 14), *groups(week(1), 30, 20)], NOW)  # 56%
    assert not assist_gate(low).passed
    few = weekly_adoption([*groups(week(2), 10, 10), *groups(week(1), 30, 30)], NOW)
    gate = assist_gate(few)
    assert not gate.passed and ASSIST_MIN_GROUPS == 20
    assert "推荐组 10" in gate.checks[2].detail
    assert assist_gate(few, min_groups=10).passed
    # 影子模式的推荐不算采纳率（那一周推荐组为 0）
    shadow_week = weekly_adoption(
        [*groups(week(2), 25, 25, adopted_status=S.SHADOW), *groups(week(1), 30, 30)], NOW
    )
    assert shadow_week[-2].overall.groups == 0 and not assist_gate(shadow_week).passed


def test_auto_intent_gates_need_four_weeks_of_as_is_adoption():
    records = []
    for k in (4, 3, 2, 1):
        records += groups(week(k), 10, 9, intent=SHIP)  # 90% 原样采纳
        records += groups(week(k), 10, 10, intent=SIZE, adopted_status=S.EDITED_SENT)
    weekly = weekly_adoption(records, NOW)
    intents = RoutingConfig().auto_threshold
    gates = auto_intent_gates(weekly, intents)
    assert list(gates) == [i for i in Intent if i in intents]  # Intent 定义顺序
    assert gates[SHIP].passed and gates[SHIP].unchecked == ["complaints"]
    assert gates[SHIP].stage == "intent:shipping_policy"
    assert not gates[Intent.GREETING].passed  # 没有数据
    assert SIZE not in gates  # 尺码推荐不开放全自动
    # 编辑后发送不算原样采纳
    assert not auto_intent_gates(weekly, [SIZE])[SIZE].passed
    assert not auto_intent_gates(weekly, [SHIP], complaints=1)[SHIP].passed
    three_weeks = weekly_adoption(records, NOW, weeks=3)
    assert not auto_intent_gates(three_weeks, [SHIP])[SHIP].passed


# ================================================================ 影子对比（§13.4 第一阶段）


def test_shadow_comparison_matches_agent_replies_across_conversations():
    at = week(1) + timedelta(minutes=1)
    records = [
        shadow_rec(1, at, msg_id="b1"),
        shadow_rec(2, at, msg_id="b2", intent=SIZE, text="亲，165 建议 M 码"),
        shadow_rec(3, week(3), msg_id="old"),  # 窗口外
        SuggestionRecord(  # 不是影子推荐
            id=4,
            shop_id=SHOP_A,
            conversation_id=CONV,
            msg_id="b3",
            intent=SHIP,
            text="亲，48 小时内发货哦",
            status=S.SHOWN,
            created_at=at,
        ),
    ]
    conv1 = [
        msg("b1", Role.BUYER, "什么时候发货", 0),
        msg("a1", Role.AGENT, "亲，现货付款后 48 小时内发货哦~", 1),
    ]
    conv2 = [msg("b2", Role.BUYER, "165 穿什么码", 0), msg("x1", Role.BOT, "自动回复", 1)]
    report = shadow_comparison(records, [conv1, conv2], since=week(2))
    assert report.groups == 2 and report.compared == 1 and report.no_reply == 1
    assert report.match_rate == 1.0
    by_id = {i.suggestion_id: i.similarity for i in report.items}
    assert by_id == {1: 1.0, 2: None}  # 机器人自动回复不算客服回复
    assert shadow_comparison(records, [], since=week(2), until=at).groups == 0  # until 不含


def test_shadow_reviews_keep_only_shadow_suggestions_in_the_window():
    at = week(1) + timedelta(hours=2)
    records = {
        1: shadow_rec(1, at, msg_id="b1"),
        2: shadow_rec(2, week(3), msg_id="b2"),  # 窗口外
        3: groups(at, 1, 1)[0].model_copy(update={"id": 3}),  # 展示过的推荐（智能辅助阶段）
        4: shadow_rec(4, at, msg_id="b4").model_copy(update={"shop_id": SHOP_B}),
    }
    reviews = [
        review(1, 1, period="2026-W39", status=QualityReviewStatus.PENDING),
        review(2, 1, period="2026-W40", reviewed_at=NOW),  # 同一推荐在两个周期被抽中：取复核过的
        review(3, 2),
        review(4, 3),
        review(5, 4),
        review(6, 99),  # 推荐已不存在
    ]
    picked = shadow_reviews(reviews, records, shop_id=SHOP_A, since=week(2))
    assert [(q.id, q.suggestion_id) for q in picked] == [(2, 1)]


def test_shadow_gate_uses_scored_shadow_reviews_and_lists_test_set_as_unchecked():
    passing = launch_gate([review(i, i) for i in range(1, 21)])
    gate = shadow_gate(passing)
    assert gate.stage == "shadow" and gate.passed and gate.unchecked == ["test_set"]
    assert [c.name for c in gate.checks] == [
        "shadow_review.scored",
        "shadow_review.factual_full_rate",
        "shadow_review.compliance_full_rate",
    ]
    assert shadow_gate(passing, test_set_passed=True).unchecked == []
    assert not shadow_gate(passing, test_set_passed=False).passed
    failing = launch_gate([review(i, i, compliance=1 if i <= 2 else 2) for i in range(1, 21)])
    gate = shadow_gate(failing)
    assert not gate.passed
    assert [c.name for c in gate.checks if not c.passed] == ["shadow_review.compliance_full_rate"]
    few = shadow_gate(launch_gate([review(i, i) for i in range(1, 6)]))
    assert [c.name for c in few.checks if not c.passed] == ["shadow_review.scored"]
    summary = gate_summary(gate)
    assert summary == {
        "stage": "shadow",
        "passed": False,
        "failed": ["shadow_review.compliance_full_rate"],
        "unchecked": ["test_set"],
        "values": {
            "shadow_review.scored": 20,
            "shadow_review.factual_full_rate": 1.0,
            "shadow_review.compliance_full_rate": 0.9,
        },
    }


# ================================================================ 汇总与建议


def _status(config: ShopConfig, records, reviews, **kw):
    return rollout_status(
        shop_id=SHOP_A,
        config=config,
        routing=RoutingConfig(),
        now=NOW,
        records=records,
        shadow=shadow_comparison(records, [], since=shadow_window(NOW)),
        shadow_since=shadow_window(NOW),
        reviews=reviews,
        **kw,
    )


def test_status_in_shadow_mode_recommends_assist_only_when_the_gate_passes():
    at = week(1) + timedelta(hours=1)
    records = [shadow_rec(i, at, msg_id=f"b{i}") for i in range(1, 26)]
    good = [review(i, i) for i in range(1, 26)]
    status = _status(ShopConfig(shadow_mode=True), records, good)
    assert status.stage == RolloutStage.SHADOW and status.stage_label == "影子模式"
    assert status.advice == RolloutAdvice.READY_FOR_ASSIST
    assert status.blocking == [] and [u.name for u in status.unchecked] == ["test_set"]
    assert "rodit.eval" in status.unchecked[0].description
    assert status.shadow_quality.scored == 25 and status.shadow.groups == 25
    assert status.shadow.no_reply == 25  # 没有传会话消息
    assert len(status.weekly) == 4 and status.weekly[-1].period == "2026-W39"
    assert status.weekly[-1].shadow_excluded == 25

    bad = [review(i, i, factual=1 if i <= 3 else 2) for i in range(1, 26)]
    status = _status(ShopConfig(shadow_mode=True), records, bad)
    assert status.advice == RolloutAdvice.STAY_SHADOW
    assert status.blocking == ["shadow_review.factual_full_rate"]
    # 其他店铺的推荐、样本不算
    other = [r.model_copy(update={"shop_id": SHOP_B}) for r in records]
    assert _status(ShopConfig(shadow_mode=True), other, good).shadow_quality.scored == 0


def test_status_in_assist_and_auto_stages():
    records = [*groups(week(2), 25, 15), *groups(week(1), 30, 20)]
    status = _status(ShopConfig(), records, [])
    assert status.stage == RolloutStage.ASSIST
    assert status.advice == RolloutAdvice.READY_FOR_NEXT_STAGE
    assert [u.name for u in status.unchecked] == ["platform_violations"]
    status = _status(ShopConfig(), records[:30], [])
    assert status.advice == RolloutAdvice.STAY_ASSIST
    assert "period_2.group_adoption_rate" in status.blocking

    auto = _status(ShopConfig(auto_reply_enabled=True), records, [], complaints=None)
    assert auto.stage == RolloutStage.AUTO and auto.advice == RolloutAdvice.AUTO_ENABLED
    assert [u.name for u in auto.unchecked] == ["complaints"]
    assert "intent:greeting:periods" not in auto.blocking  # 4 个周期都有报表
    assert any(b.startswith("intent:shipping_policy:period_") for b in auto.blocking)
    assert set(auto.intent_gates) == set(RoutingConfig().auto_threshold)
    dumped = auto.model_dump_json()
    assert "48 小时" not in dumped  # 状态里没有推荐文字


@pytest.mark.parametrize("platform_violations", [0, None])
def test_status_passes_platform_violations_through(platform_violations):
    records = [*groups(week(2), 25, 15), *groups(week(1), 30, 20)]
    status = _status(ShopConfig(), records, [], platform_violations=platform_violations)
    names = [u.name for u in status.unchecked]
    assert names == ([] if platform_violations == 0 else ["platform_violations"])
