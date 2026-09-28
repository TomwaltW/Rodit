"""报表口径（方案 §1.3、§12.4、§13.4）：采纳率（按条 / 按推荐组）、弃用原因、类型占比、耗时分位数、
质检拦截与降级、影子对比、上线阶段门槛。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.enums import DraftKind, Intent, RejectReason, Role, SuggestionStatus
from rodit.core.models import AuditEvent, StoredMessage, SuggestionRecord
from rodit.feedback import (
    RESULT_ACTION,
    RESULT_DEGRADED,
    RESULT_HARD_HITS,
    RESULT_INTENT,
    RESULT_LATENCY,
    RESULT_NO_SUGGESTION,
    RESULT_QA_CHECKED,
    RESULT_QA_FAILED,
    RESULT_QA_REASONS,
    RESULT_SUGGESTIONS,
    RESULT_TRANSFER,
    AdoptionReport,
    adoption_report,
    agent_replies_by_msg,
    char_similarity,
    intent_gate,
    percentile,
    phase_gate,
    qa_report,
    shadow_report,
)
from rodit.testing.fixtures import CONV, DEFAULT_NOW, SHOP_A, SHOP_B

S = SuggestionStatus
SHIP, SIZE = Intent.SHIPPING_POLICY, Intent.SIZE_ADVICE
_ids = iter(range(1, 10_000))


def rec(
    msg_id: str | None,
    intent: Intent,
    status: SuggestionStatus,
    *,
    latency: int | None = None,
    reason: RejectReason | None = None,
    minutes: float = 0,
    text: str = "亲，默认发中通快递哦",
    shop_id: str = SHOP_A,
    conversation_id: str = CONV,
) -> SuggestionRecord:
    return SuggestionRecord(
        id=next(_ids),
        shop_id=shop_id,
        conversation_id=conversation_id,
        msg_id=msg_id,
        intent=intent,
        text=text,
        kind=DraftKind.LLM,
        status=status,
        reject_reason=reason,
        latency_ms=latency,
        created_at=DEFAULT_NOW + timedelta(minutes=minutes),
    )


def dataset() -> list[SuggestionRecord]:
    return [
        # 发货政策：4 组 6 条
        rec("m1", SHIP, S.SENT, latency=1000),
        rec("m1", SHIP, S.EXPIRED, latency=1000),
        rec("m2", SHIP, S.EDITED_SENT, latency=1200),
        rec("m2", SHIP, S.EXPIRED, latency=1200),
        rec("m3", SHIP, S.REJECTED, latency=800, reason=RejectReason.MISSING_KNOWLEDGE),
        rec("m4", SHIP, S.SHOWN, latency=3000),
        # 尺码：2 组 2 条
        rec("m5", SIZE, S.SENT, latency=900),
        rec("m6", SIZE, S.REJECTED, latency=6000, reason=RejectReason.WRONG_INTENT),
        # 影子：不计入
        rec("m7", SHIP, S.SHADOW, latency=500),
        rec("m7", SHIP, S.SHADOW, latency=500),
        rec("m8", SIZE, S.SHADOW),
    ]


# ================================================================ 通用


def test_percentile_nearest_rank():
    values = list(range(1, 21))
    assert percentile(values, 0.5) == 10
    assert percentile(values, 0.95) == 19
    assert percentile(values, 1.0) == 20
    assert percentile([7], 0.95) == 7
    assert percentile([], 0.5) is None
    assert percentile([3, 1, 2], 0.5) == 2
    for q in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            percentile(values, q)


def test_char_similarity():
    assert char_similarity("亲，默认发中通快递哦~", "亲 默认发中通快递哦") == 1.0  # 忽略标点空白
    assert char_similarity("abcd", "abce") == pytest.approx(2 * 2 / 6)  # 二元组 Dice
    assert char_similarity("中通快递", "顺丰速运") == 0.0
    assert char_similarity("", "  ") == 1.0
    assert char_similarity("好", "") == 0.0
    assert char_similarity("好", "好的") == pytest.approx(2 * 1 / 3)  # 一个字时退回单字
    assert char_similarity("１２３", "123") == 1.0  # NFKC


# ================================================================ 采纳率


def test_adoption_report_math():
    report = adoption_report(dataset())
    assert report.shop_ids == [SHOP_A]
    assert report.shadow_excluded == 3
    o = report.overall
    assert (o.total, o.shown, o.sent, o.edited_sent, o.rejected, o.expired) == (8, 1, 2, 1, 2, 2)
    assert o.adoption_rate == pytest.approx(3 / 8)
    assert o.as_is_rate == pytest.approx(2 / 8)
    assert o.edited_rate == pytest.approx(1 / 8)
    assert o.reject_rate == pytest.approx(2 / 8)
    assert o.expired_rate == pytest.approx(2 / 8)
    assert (o.groups, o.adopted_groups, o.as_is_groups) == (6, 3, 2)
    assert o.group_adoption_rate == pytest.approx(0.5)
    assert o.group_as_is_rate == pytest.approx(2 / 6)
    assert o.share == 1.0 and o.label == "全部" and o.intent is None
    assert o.reject_reasons == {
        RejectReason.MISSING_KNOWLEDGE: 1,
        RejectReason.WRONG_INTENT: 1,
        RejectReason.BAD_WORDING: 0,
        RejectReason.WRONG_FACT: 0,
        RejectReason.OTHER: 0,
    }
    # 整体耗时按消息去重：[800, 900, 1000, 1200, 3000, 6000]
    assert (o.latency_ms.count, o.latency_ms.p50, o.latency_ms.p95) == (6, 1000, 6000)

    assert list(report.by_intent) == [i for i in Intent if i in (SHIP, SIZE)]  # 按定义顺序
    ship = report.by_intent[SHIP]
    assert (ship.total, ship.sent, ship.edited_sent, ship.rejected, ship.expired, ship.shown) == (
        6,
        1,
        1,
        1,
        2,
        1,
    )
    assert ship.adoption_rate == pytest.approx(2 / 6)
    assert (ship.groups, ship.adopted_groups, ship.as_is_groups) == (4, 2, 1)
    assert ship.group_adoption_rate == pytest.approx(0.5)
    assert ship.group_as_is_rate == pytest.approx(0.25)
    assert ship.share == pytest.approx(4 / 6)
    assert ship.label == "发货与快递"
    assert ship.reject_reasons[RejectReason.MISSING_KNOWLEDGE] == 1
    assert (ship.latency_ms.p50, ship.latency_ms.p95) == (1000, 3000)
    size = report.by_intent[SIZE]
    assert (size.total, size.groups, size.adopted_groups) == (2, 2, 1)
    assert size.group_adoption_rate == pytest.approx(0.5)
    assert size.share == pytest.approx(2 / 6)
    assert size.reject_reasons[RejectReason.WRONG_INTENT] == 1
    # 报表可以序列化（给商家后台 / HTTP 接口）
    assert AdoptionReport.model_validate_json(report.model_dump_json()) == report


def test_adoption_report_card_vs_group_rate():
    """一次给两条推荐、客服每次都采纳其中一条：按条最多 50%，按推荐组是 100%。"""
    records = []
    for i in range(10):
        records += [rec(f"m{i}", SHIP, S.SENT), rec(f"m{i}", SHIP, S.EXPIRED)]
    o = adoption_report(records).overall
    assert o.adoption_rate == pytest.approx(0.5)
    assert o.group_adoption_rate == 1.0


def test_split_message_counts_latency_once_but_groups_per_intent():
    records = [
        rec("m1", SHIP, S.SENT, latency=2000),
        rec("m1", Intent.INVOICE_GIFT, S.SENT, latency=2000),
        rec("m2", SHIP, S.REJECTED, latency=4000),
    ]
    report = adoption_report(records)
    assert report.overall.groups == 3
    assert report.overall.latency_ms.count == 2
    assert report.by_intent[Intent.INVOICE_GIFT].share == pytest.approx(1 / 3)


def test_adoption_report_time_window_and_edge_cases():
    records = [
        rec("a", SHIP, S.SENT, minutes=0),
        rec("b", SHIP, S.SENT, minutes=10),
        rec("c", SHIP, S.REJECTED, minutes=20),
    ]
    window = adoption_report(
        records,
        since=DEFAULT_NOW + timedelta(minutes=10),
        until=DEFAULT_NOW + timedelta(minutes=20),
    )
    assert window.overall.total == 1 and window.overall.sent == 1  # since 含，until 不含
    empty = adoption_report([])
    assert empty.overall.total == 0 and empty.overall.adoption_rate is None
    assert empty.overall.group_adoption_rate is None and empty.by_intent == {}
    assert empty.overall.latency_ms.p50 is None
    only_shadow = adoption_report([rec("s", SHIP, S.SHADOW)])
    assert only_shadow.shadow_excluded == 1 and only_shadow.overall.total == 0
    # 没有原因的弃用归入"其他"；没有 msg_id 的记录各自成组
    odd = adoption_report([rec(None, SHIP, S.REJECTED), rec(None, SHIP, S.SENT)])
    assert odd.overall.reject_reasons[RejectReason.OTHER] == 1
    assert odd.overall.groups == 2


def test_adoption_report_lists_shops():
    report = adoption_report([rec("a", SHIP, S.SENT), rec("a", SHIP, S.SENT, shop_id=SHOP_B)])
    assert report.shop_ids == [SHOP_A, SHOP_B]
    assert report.overall.groups == 2  # 不同店铺的同一 msg_id 不合并


# ================================================================ 质检报表


def _result(detail: dict, *, shop_id: str = SHOP_A, minutes: float = 0) -> AuditEvent:
    return AuditEvent(
        shop_id=shop_id,
        actor="system",
        action=AuditAction.RESULT.value,
        detail=detail,
        created_at=DEFAULT_NOW + timedelta(minutes=minutes),
    )


def test_qa_report():
    audit = InMemoryAuditSink()
    audit.record(
        _result(
            {
                RESULT_INTENT: "logistics_track",
                RESULT_ACTION: "answer",
                RESULT_SUGGESTIONS: 1,
                RESULT_QA_CHECKED: 2,
                RESULT_QA_FAILED: 1,
                RESULT_QA_REASONS: ["promise_risk 0.86 > 0.3"],
                RESULT_HARD_HITS: ["time_promise", "absolute_term", "format"],
                RESULT_DEGRADED: [],
                RESULT_LATENCY: {"route": 600, "generate": 2100, "total": 3900},
            }
        )
    )
    audit.record(
        _result(
            {
                RESULT_INTENT: "size_advice",
                RESULT_ACTION: "answer",
                RESULT_SUGGESTIONS: 2,
                RESULT_QA_CHECKED: 2,
                RESULT_QA_FAILED: 0,
                RESULT_HARD_HITS: [{"rule": "length", "detail": "截断"}],  # 自动修正，不算拦截
                RESULT_LATENCY: {"route": 400, "total": 2500},
            },
            minutes=1,
        )
    )
    audit.record(
        _result(
            {
                RESULT_INTENT: "complaint",
                RESULT_ACTION: "human_priority",
                RESULT_SUGGESTIONS: 0,
                RESULT_NO_SUGGESTION: "投诉，提醒人工处理",
                RESULT_DEGRADED: ["decision"],
                RESULT_TRANSFER: True,
                RESULT_LATENCY: 700,
            },
            minutes=2,
        )
    )
    audit.record(_result({RESULT_SUGGESTIONS: 0, RESULT_DEGRADED: True}, minutes=3))
    audit.record(_result({RESULT_QA_CHECKED: 5, RESULT_QA_FAILED: 5}, shop_id=SHOP_B))
    audit.record(
        AuditEvent(
            shop_id=SHOP_A,
            actor="kf_02",
            action=AuditAction.FEEDBACK_SENT.value,
            detail={RESULT_QA_CHECKED: 9},
            created_at=DEFAULT_NOW,
        )
    )

    r = qa_report(audit, SHOP_A)
    assert r.messages == 4
    assert (r.qa_checked, r.qa_failed) == (4, 1)
    assert r.interception_rate == pytest.approx(0.25)
    assert (r.messages_checked, r.messages_intercepted) == (2, 1)
    assert r.message_interception_rate == pytest.approx(0.5)
    assert r.hard_hit_counts == {"time_promise": 1, "absolute_term": 1, "format": 1, "length": 1}
    assert r.blocking_hit_counts == {"time_promise": 1, "absolute_term": 1}
    assert r.qa_reason_counts == {"promise_risk 0.86 > 0.3": 1}
    assert r.degraded_messages == 2 and r.degraded_rate == pytest.approx(0.5)
    assert r.degraded_counts == {"decision": 1, "unknown": 1}
    assert r.action_counts == {"answer": 2, "human_priority": 1}
    assert r.intent_counts == {"logistics_track": 1, "size_advice": 1, "complaint": 1}
    assert r.no_suggestion == 2 and r.no_suggestion_reasons == {"投诉，提醒人工处理": 1}
    assert r.transfers == 1
    assert r.latency_ms["total"].count == 3
    assert (r.latency_ms["total"].p50, r.latency_ms["total"].p95) == (2500, 3900)
    assert r.latency_ms["route"].p95 == 600
    assert r.p95_within_target is True

    later = qa_report(audit, SHOP_A, DEFAULT_NOW + timedelta(minutes=1))
    assert later.messages == 3
    window = qa_report(audit, SHOP_A, until=DEFAULT_NOW + timedelta(minutes=1))
    assert window.messages == 1
    b = qa_report(audit, SHOP_B)
    assert b.messages == 1 and b.interception_rate == 1.0 and b.p95_within_target is None


def test_qa_report_flags_slow_p95_and_tolerates_bad_detail():
    audit = InMemoryAuditSink()
    for i, total in enumerate([1000, 2000, 7000]):
        audit.record(_result({RESULT_LATENCY: {"total": total}}, minutes=i))
    audit.record(
        _result(
            {
                RESULT_QA_CHECKED: "3",  # 类型不对：当作 0
                RESULT_QA_FAILED: True,
                RESULT_HARD_HITS: "time_promise",  # 单个字符串也接受
                RESULT_DEGRADED: "llm",
                RESULT_QA_REASONS: {"not": "a list"},
                RESULT_LATENCY: {"total": "slow", "route": True},
                RESULT_ACTION: 5,
            },
            minutes=9,
        )
    )
    r = qa_report(audit, SHOP_A)
    assert r.messages == 4 and r.p95_within_target is False  # P95 = 7000 ms > 5 秒
    assert (r.qa_checked, r.qa_failed) == (0, 0)
    assert r.hard_hit_counts == {"time_promise": 1}
    assert r.messages_checked == 1 and r.messages_intercepted == 1  # 拦截类硬规则命中
    assert r.degraded_counts == {"llm": 1} and r.qa_reason_counts == {}
    assert r.action_counts == {} and r.latency_ms["total"].count == 3
    assert "route" not in r.latency_ms
    empty = qa_report(InMemoryAuditSink(), SHOP_A)
    assert empty.messages == 0 and empty.interception_rate is None


# ================================================================ 影子对比


def _msg(msg_id: str, role: Role, text: str, minutes: float) -> StoredMessage:
    return StoredMessage(
        msg_id=msg_id,
        shop_id=SHOP_A,
        conversation_id=CONV,
        role=role,
        text=text,
        created_at=DEFAULT_NOW + timedelta(minutes=minutes),
    )


def test_agent_replies_by_msg():
    messages = [
        _msg("a0", Role.AGENT, "欢迎光临", 0),  # 买家还没说话：不对应任何消息
        _msg("b1", Role.BUYER, "在吗", 1),
        _msg("b2", Role.BUYER, "发什么快递", 2),
        _msg("a1", Role.AGENT, "亲，在的", 3),
        _msg("a2", Role.AGENT, "默认发中通快递哦", 4),
        _msg("b3", Role.BUYER, "能到新疆吗", 5),
        _msg("x1", Role.BOT, "【智能客服】自动回复", 6),
        _msg("a3", Role.AGENT, "新疆需补 15 元运费", 7),
        _msg("b4", Role.BUYER, "好的", 8),
    ]
    assert agent_replies_by_msg(messages) == {
        "b1": ["亲，在的", "默认发中通快递哦"],
        "b2": ["亲，在的", "默认发中通快递哦"],
        "b3": ["新疆需补 15 元运费"],
        "b4": [],
    }


def test_shadow_report():
    records = [
        rec("b2", SHIP, S.SHADOW, text="亲，默认发中通快递哦~"),
        rec("b2", SHIP, S.SHADOW, text="亲，我们发顺丰"),
        rec("b3", SHIP, S.SHADOW, text="亲，新疆需要补 15 元运费哦"),
        rec("b4", Intent.GREETING, S.SHADOW, text="亲，还有什么可以帮您"),
        rec("b5", SIZE, S.SHADOW, text="建议 M 码"),
        rec("b2", SHIP, S.SENT, text="展示过的推荐不参与影子对比"),
    ]
    sent = {
        "b2": ["亲，在的", "默认发中通快递哦"],
        "b3": "新疆需补 15 元运费",
        "b4": [],
        "b5": "  ",
    }
    report = shadow_report(records, sent, match_threshold=0.5)
    assert (report.groups, report.compared, report.no_reply) == (4, 2, 2)
    by_msg = {i.msg_id: i for i in report.items}
    assert by_msg["b2"].suggestion_id == records[0].id  # 组内取最像的一条
    assert by_msg["b2"].similarity == pytest.approx(
        char_similarity("亲，默认发中通快递哦~", "默认发中通快递哦"), abs=1e-4
    )
    assert by_msg["b4"].similarity is None and by_msg["b5"].similarity is None
    scores = sorted(i.similarity for i in report.items if i.similarity is not None)
    assert report.mean_similarity == pytest.approx(sum(scores) / 2)
    assert report.median_similarity == scores[0]
    assert report.match_rate == pytest.approx(sum(s >= 0.5 for s in scores) / 2)
    assert report.by_intent[SHIP].compared == 2 and report.by_intent[SIZE].compared == 0
    assert report.by_intent[SIZE].mean_similarity is None
    # 报表里没有任何文字，只有 ID 和分数
    dumped = report.model_dump_json()
    assert "中通" not in dumped and "新疆" not in dumped
    empty = shadow_report([], {})
    assert empty.groups == 0 and empty.mean_similarity is None and empty.match_rate is None


# ================================================================ 阶段门槛（§13.4）


def _week(start_days: int, adopted: int, total: int, intent: Intent = SHIP) -> AdoptionReport:
    since = DEFAULT_NOW + timedelta(days=start_days)
    records = [
        rec(f"w{start_days}_{i}", intent, S.SENT if i < adopted else S.REJECTED, minutes=1)
        for i in range(total)
    ]
    records = [r.model_copy(update={"created_at": since + timedelta(hours=1)}) for r in records]
    return adoption_report(records, since=since, until=since + timedelta(days=7))


def test_phase_gate_single_report():
    report = adoption_report(dataset())  # 按推荐组 50%
    assert not phase_gate(report).passed
    result = phase_gate(report, min_adoption=0.5)  # Phase 0 验收：模式 C 采纳率 ≥ 50%
    assert result.passed and result.unchecked == ["platform_violations"]
    rate = next(c for c in result.checks if c.name == "period_1.group_adoption_rate")
    assert rate.value == pytest.approx(0.5) and rate.threshold == 0.5
    assert not phase_gate(report, min_adoption=0.5, min_samples=7).passed  # 样本不够
    assert not phase_gate(adoption_report([])).passed


def test_phase_gate_two_consecutive_weeks_and_violations():
    w1, w2 = _week(0, 13, 20), _week(7, 12, 20)  # 65%、60%
    ok = phase_gate([w1, w2], min_periods=2, platform_violations=0)
    assert ok.passed and ok.unchecked == []
    assert {c.name for c in ok.checks} >= {"periods", "consecutive", "platform_violations"}
    assert not phase_gate([w1, w2], min_periods=2, platform_violations=1).passed
    assert not phase_gate([w1], min_periods=2, platform_violations=0).passed  # 只有 1 周
    low = _week(7, 11, 20)  # 55%
    assert not phase_gate([w1, low], min_periods=2, platform_violations=0).passed
    gap = _week(14, 13, 20)  # 和第一周之间隔了一周
    not_consecutive = phase_gate([w1, gap], min_periods=2, platform_violations=0)
    assert not not_consecutive.passed
    assert next(c for c in not_consecutive.checks if c.name == "consecutive").passed is False
    no_window = adoption_report(dataset())
    assert "consecutive" in phase_gate([no_window, no_window], min_adoption=0.5).unchecked


def test_intent_gate_for_semi_and_full_auto():
    weeks = [_week(7 * i, 19, 20) for i in range(4)]  # 原样采纳 95%
    semi = intent_gate(weeks[0], SHIP)
    assert semi.passed and semi.stage == "intent:shipping_policy"
    assert semi.unchecked == ["complaints"]
    full = intent_gate(weeks, SHIP, min_periods=4, complaints=0)
    assert full.passed
    assert not intent_gate(weeks[:3], SHIP, min_periods=4, complaints=0).passed
    assert not intent_gate(weeks, SHIP, min_periods=4, complaints=1).passed
    assert not intent_gate(weeks[0], SIZE).passed  # 这一周没有该类型的数据
    edited = adoption_report(
        [rec(f"e{i}", SHIP, S.EDITED_SENT) for i in range(10)]
    )  # 全是编辑后发送：采纳率 100%，原样采纳率 0
    assert phase_gate(edited).passed
    assert not intent_gate(edited, SHIP).passed
