"""全自动发送的回复不算进采纳率（方案 §1.3 采纳率是客服原样 / 小改后发送的比例、§13.4 全自动
按类型监控）：agent_id="bot" 的推荐单独列在 auto 一行，不改变采纳率、原样采纳率和上线门槛。
数据全部合成。"""

from __future__ import annotations

from datetime import timedelta

from rodit.core.enums import DraftKind, Intent, SuggestionStatus
from rodit.core.models import SuggestionRecord
from rodit.feedback.metrics import AUTO_AGENT_ID, adoption_report, is_auto_sent
from rodit.feedback.review import week_start
from rodit.feedback.rollout import assist_gate, weekly_adoption
from rodit.testing.fixtures import DEFAULT_NOW, SHOP_A

S = SuggestionStatus
SHIP, SIZE = Intent.SHIPPING_POLICY, Intent.SIZE_ADVICE
_ids = iter(range(1, 10_000))


def rec(
    msg_id: str,
    intent: Intent,
    status: SuggestionStatus,
    *,
    agent_id: str | None = None,
    latency: int | None = None,
    minutes: float = 0,
    base=DEFAULT_NOW,
) -> SuggestionRecord:
    return SuggestionRecord(
        id=next(_ids),
        shop_id=SHOP_A,
        conversation_id=f"conv_{msg_id}",
        msg_id=msg_id,
        intent=intent,
        text="亲，现货付款后 48 小时内发货哦~",
        kind=DraftKind.LLM,
        status=status,
        agent_id=agent_id,
        latency_ms=latency,
        created_at=base + timedelta(minutes=minutes),
    )


def human_records() -> list[SuggestionRecord]:
    return [
        rec("h1", SHIP, S.SENT, agent_id="kf_a01", latency=900),
        rec("h2", SHIP, S.EDITED_SENT, agent_id="kf_a01", latency=1100),
        rec("h3", SHIP, S.REJECTED, agent_id="kf_a01", latency=1300),
        rec("h4", SIZE, S.SHOWN, latency=1500),
    ]


def bot_records(n: int = 10, *, intent: Intent = SHIP) -> list[SuggestionRecord]:
    return [
        rec(f"b{i}", intent, S.SENT, agent_id=AUTO_AGENT_ID, latency=20_000, minutes=i)
        for i in range(n)
    ]


def test_ten_bot_sent_records_leave_the_rates_unchanged_and_show_up_under_auto():
    human = human_records()
    before = adoption_report(human)
    after = adoption_report([*human, *bot_records(10)])
    for field in (
        "total",
        "sent",
        "groups",
        "adoption_rate",
        "as_is_rate",
        "group_adoption_rate",
        "group_as_is_rate",
        "latency_ms",
    ):
        assert getattr(after.overall, field) == getattr(before.overall, field), field
    assert after.overall.as_is_rate == 0.25 and after.overall.group_as_is_rate == 0.25
    assert after.by_intent == before.by_intent
    # 全自动单独一行：10 条、10 组，都是发货政策；耗时也只在这一行
    assert before.auto.sent == 0 and before.auto.groups == 0
    assert after.auto.sent == 10 and after.auto.groups == 10
    assert after.auto.by_intent == {SHIP: 10}
    assert after.auto.latency_ms.count == 10 and after.auto.latency_ms.p95 == 20_000
    assert after.overall.latency_ms.p95 == 1500  # 自动发送的耗时不混进客服推荐


def test_auto_sent_means_sent_by_the_bot_only():
    assert is_auto_sent(rec("x", SHIP, S.SENT, agent_id="bot"))
    # 客服发的、全自动没发出去（留给客服，仍是 SHOWN）的都算客服推荐
    assert not is_auto_sent(rec("x", SHIP, S.SENT, agent_id="kf_a01"))
    assert not is_auto_sent(rec("x", SHIP, S.SHOWN, agent_id=None))
    assert not is_auto_sent(rec("x", SHIP, S.SHADOW, agent_id="bot"))


def test_a_group_the_bot_answered_moves_to_the_auto_line_as_a_whole():
    """同一条消息、同一类型的几张卡片里有一张被全自动发出去：整组算全自动，
    其余卡片（没展示给客服）不压低客服的采纳率。"""
    human = human_records()
    bot_group = [
        rec("g1", SHIP, S.SENT, agent_id=AUTO_AGENT_ID),
        rec("g1", SHIP, S.EXPIRED),  # 同一会话、同一消息、同一类型
    ]
    report = adoption_report([*human, *bot_group])
    assert report.overall.groups == 4 and report.overall.total == 4
    assert report.auto.sent == 1 and report.auto.groups == 1


def test_assist_gate_ignores_bot_sent_weeks():
    """§13.4 智能辅助门槛按客服的采纳率算：全自动发出去的 100 条不会把 50% 的采纳率抬过 60%。"""
    monday = week_start(DEFAULT_NOW)
    now = monday + timedelta(days=14, hours=1)
    records = []
    for k, start in enumerate((monday, monday + timedelta(days=7))):
        for i in range(30):
            status = S.SENT if i % 2 == 0 else S.REJECTED
            records.append(rec(f"w{k}_{i}", SHIP, status, agent_id="kf_a01", base=start, minutes=i))
        records += [
            rec(f"wb{k}_{i}", SHIP, S.SENT, agent_id=AUTO_AGENT_ID, base=start, minutes=i)
            for i in range(100)
        ]
    weekly = weekly_adoption(records, now, weeks=2)
    assert [w.overall.group_adoption_rate for w in weekly] == [0.5, 0.5]
    assert [w.auto.sent for w in weekly] == [100, 100]
    assert not assist_gate(weekly).passed
