"""Phase 1 验收（方案 §15：满足 §13.2 的门槛；采纳率 ≥ 60%；推荐 P95 ≤ 5 秒；平台违规为 0；
§13.5 每周复核按 §13.2 打分）：每家店一份、所有店合在一起一份，缺数据的条件不算通过。纯函数，
数据全部合成。"""

from __future__ import annotations

from datetime import datetime, timedelta

from rodit.core.audit import LATENCY_SINCE_RECEIVED, RESULT_BUDGET_EXCEEDED, AuditAction, ResultKey
from rodit.core.clock import CST
from rodit.core.enums import DraftKind, Intent, QualityReviewStatus, SuggestionStatus
from rodit.core.models import AuditEvent, QualityReview, QualityScores, SuggestionRecord
from rodit.feedback.acceptance import (
    CriterionStatus,
    ShopAcceptanceInput,
    acceptance_report,
    message_latencies,
    render_text,
)
from rodit.feedback.compliance import ComplianceSummary
from rodit.feedback.rollout import IncidentEntry, IncidentKind, TestSetEvidence, incident_total
from rodit.testing.fixtures import SHOP_A, SHOP_B

P, F, U = CriterionStatus.PASSED, CriterionStatus.FAILED, CriterionStatus.UNCHECKED
W38 = datetime(2026, 9, 14, tzinfo=CST)
W39 = W38 + timedelta(days=7)
NOW = W39 + timedelta(days=9, hours=12)  # 2026-09-30 周三（W40）
SINCE = datetime(2026, 9, 1, tzinfo=CST)  # W36 周二
WEEKS = ["2026-W36", "2026-W37", "2026-W38", "2026-W39"]
PASSING_TEST_SET = TestSetEvidence(passed=True, detail="合成的评测结论：达标")
_ids = iter(range(1, 100_000))


def records(shop_id: str, start: datetime, groups: int, adopted: int) -> list[SuggestionRecord]:
    return [
        SuggestionRecord(
            id=next(_ids),
            shop_id=shop_id,
            conversation_id=f"c{i % 5}",
            msg_id=f"{shop_id}_{start:%m%d}_{i}",
            intent=Intent.SHIPPING_POLICY,
            text="亲，现货付款后 48 小时内发货哦~",
            kind=DraftKind.LLM,
            status=SuggestionStatus.SENT if i < adopted else SuggestionStatus.REJECTED,
            agent_id="kf_a01",
            created_at=start + timedelta(hours=1, minutes=i),
        )
        for i in range(groups)
    ]


def reviews(shop_id: str, n: int, *, bad: int = 0) -> list[QualityReview]:
    out = []
    for i in range(n):
        scores = QualityScores(factual=1 if i < bad else 2, relevance=2, compliance=2, expression=2)
        out.append(
            QualityReview(
                id=next(_ids),
                shop_id=shop_id,
                period="2026-W39",
                suggestion_id=i + 1,
                intent=Intent.SHIPPING_POLICY,
                status=QualityReviewStatus.SCORED,
                scores=scores,
                reviewer="qa_reviewer_01",
                sampled_at=W39 + timedelta(days=7),
                reviewed_at=W39 + timedelta(days=7, hours=1),
            )
        )
    return out


def violations(shop_id: str, counts: dict[str, int]):
    entries = [
        IncidentEntry(
            shop_id=shop_id,
            kind=IncidentKind.VIOLATION,
            week=week,
            count=count,
            reviewer="qa_reviewer_01",
            recorded_at=NOW - timedelta(days=1),
        )
        for week, count in counts.items()
    ]
    return incident_total(entries, IncidentKind.VIOLATION, WEEKS, extra_weeks=["2026-W40"])


def scan(flagged: int = 0) -> ComplianceSummary:
    return ComplianceSummary(since=SINCE, until=NOW, flagged=flagged)


def shop_input(shop_id: str, *, adoption=(25, 25), latency=1200.0, **over) -> ShopAcceptanceInput:
    fields = {
        "shop_id": shop_id,
        "stage": "assist",
        "records": [
            *records(shop_id, W38, 30, adoption[0]),
            *records(shop_id, W39, 30, adoption[1]),
        ],
        "latencies_ms": [latency] * 40,
        "violations": violations(shop_id, dict.fromkeys(WEEKS, 0)),
        "compliance": scan(),
        "reviews": reviews(shop_id, 25),
    }
    fields.update(over)
    return ShopAcceptanceInput(**fields)


def report(inputs, test_set=PASSING_TEST_SET):
    return acceptance_report(
        inputs, test_set=test_set, since=SINCE, until=NOW, now=NOW, running_fingerprint="f" * 64
    )


def statuses(criteria) -> dict[str, CriterionStatus]:
    return {c.name: c.status for c in criteria}


def test_all_criteria_pass():
    r = report([shop_input(SHOP_A), shop_input(SHOP_B)])
    assert r.passed and r.failing == [] and r.unchecked == []
    assert statuses(r.pooled) == {
        "test_set": P,
        "adoption": P,
        "latency_p95": P,
        "platform_violations": P,
        "compliance_scan": P,
        "weekly_review": P,
    }
    assert [s.shop_id for s in r.shops] == [SHOP_A, SHOP_B] and all(s.passed for s in r.shops)
    assert "test_set" not in statuses(r.shops[0].criteria)  # 测试集是整个单元的
    assert r.adoption_weeks == ["2026-W38", "2026-W39"]
    assert "通过" in render_text(r)


def test_adoption_below_sixty_percent_fails_and_is_named():
    r = report([shop_input(SHOP_A, adoption=(17, 16))])  # 56.7% / 53.3%
    assert not r.passed and r.failing == ["adoption"]
    adoption = next(c for c in r.pooled if c.name == "adoption")
    assert adoption.status == F and adoption.value == 0.5333 and adoption.threshold == 0.6
    text = render_text(r)
    assert "adoption=0.53" in text and "没通过" in text


def test_pooled_adoption_weighs_shops_by_groups():
    # A 70% / 70%，B 50% / 50%（各 30 组）→ 合在一起 60%，通过；B 自己不通过
    r = report([shop_input(SHOP_A, adoption=(21, 21)), shop_input(SHOP_B, adoption=(15, 15))])
    assert r.passed
    by_shop = {s.shop_id: s for s in r.shops}
    assert by_shop[SHOP_A].passed and not by_shop[SHOP_B].passed
    assert statuses(by_shop[SHOP_B].criteria)["adoption"] == F


def test_too_few_groups_or_no_latency_data_are_unchecked_not_passed():
    thin = shop_input(SHOP_A, records=records(SHOP_A, W38, 5, 5), latencies_ms=[])
    r = report([thin])
    assert not r.passed and set(r.unchecked) >= {"adoption", "latency_p95"}
    crit = {c.name: c for c in r.pooled}
    assert "推荐组" in crit["adoption"].detail


def test_latency_p95_over_five_seconds_fails():
    slow = shop_input(SHOP_A, latencies_ms=[1000.0] * 90 + [6000.0] * 10)
    r = report([slow])
    assert r.failing == ["latency_p95"]
    crit = {c.name: c for c in r.pooled}["latency_p95"]
    assert crit.value == 6000.0 and crit.threshold == 5000


def test_violations_and_the_scan_are_reported_separately():
    one = violations(SHOP_A, {**dict.fromkeys(WEEKS, 0), "2026-W38": 1})
    r = report([shop_input(SHOP_A, violations=one)])
    assert r.failing == ["platform_violations"]
    r = report([shop_input(SHOP_A, compliance=scan(2))])
    assert r.failing == ["compliance_scan"]
    missing = violations(SHOP_A, {"2026-W36": 0, "2026-W37": 0})
    r = report([shop_input(SHOP_A, violations=missing), shop_input(SHOP_B)])
    assert r.unchecked == ["platform_violations"] and not r.passed
    crit = {c.name: c for c in r.pooled}["platform_violations"]
    assert SHOP_A in crit.detail and "2026-W38" in crit.detail


def test_weekly_review_and_test_set():
    r = report([shop_input(SHOP_A, reviews=reviews(SHOP_A, 25, bad=2))])  # 92%
    assert r.failing == ["weekly_review"]
    r = report([shop_input(SHOP_A, reviews=reviews(SHOP_A, 5))])
    assert r.unchecked == ["weekly_review"]
    r = report([shop_input(SHOP_A)], test_set=TestSetEvidence(reason="还没有评测结论"))
    assert r.unchecked == ["test_set"]
    assert "还没有评测结论" in {c.name: c for c in r.pooled}["test_set"].detail
    r = report([shop_input(SHOP_A)], test_set=TestSetEvidence(passed=False, detail="没达标"))
    assert r.failing == ["test_set"]


def test_no_shops_is_not_a_pass():
    r = report([])
    assert not r.passed and "adoption" in r.unchecked


def _result(at: datetime, latency: dict, suggestions: int = 1, **extra) -> AuditEvent:
    return AuditEvent(
        shop_id=SHOP_A,
        actor="engine",
        action=AuditAction.RESULT.value,
        detail={
            ResultKey.LATENCY.value: latency,
            ResultKey.SUGGESTIONS.value: suggestions,
            **extra,
        },
        created_at=at,
    )


def test_message_latencies_prefer_time_since_received():
    at = W39 + timedelta(hours=2)
    events = [
        _result(at, {"total": 900, LATENCY_SINCE_RECEIVED: 1500}),
        _result(at, {"total": 800}),
        _result(at, {"total": 300}, suggestions=0),  # 没出推荐：不是"推荐出现"
        _result(at, {"total": 4600}, suggestions=0, **{RESULT_BUDGET_EXCEEDED: True}),  # 超时算
        _result(at - timedelta(days=30), {"total": 100}),  # 窗口外
        _result(at, {"total": "x"}),
    ]
    assert message_latencies(events, since=SINCE, until=NOW) == [1500.0, 800.0, 4600.0]
