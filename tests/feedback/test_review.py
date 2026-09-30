"""每周人工复核的纯逻辑（方案 §13.5 每周抽 50 条、§13.2 四个维度打分和上线门槛）：
ISO 周（东八区）、分层抽样、只补不换、复核报表与门槛、复核列表的个人信息兜底。"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

import pytest

from rodit.core.clock import CST
from rodit.core.enums import (
    INTENT_LABELS,
    DraftKind,
    Intent,
    QualityReviewStatus,
    SuggestionStatus,
)
from rodit.core.models import QualityReview, QualityScores, SuggestionRecord
from rodit.feedback import (
    DEFAULT_SAMPLE_SIZE,
    GateStatus,
    iso_period,
    launch_gate,
    period_range,
    plan_sample,
    previous_period,
    quality_report,
    review_item,
)
from rodit.feedback.review import (
    SAMPLE_STATUSES,
    allocate,
    check_period,
    eligible_records,
    meets_rate,
    periods_between,
    recent_periods,
    sample_key,
    week_start,
)
from rodit.testing.fixtures import CONV, DEFAULT_NOW, SHOP_A, SHOP_B

S = SuggestionStatus
Q = QualityReviewStatus
SHIP, SIZE, STOCK, INVOICE = (
    Intent.SHIPPING_POLICY,
    Intent.SIZE_ADVICE,
    Intent.STOCK,
    Intent.INVOICE_GIFT,
)
W39 = "2026-W39"  # 2026-09-21（周一）至 2026-09-27（周日），DEFAULT_NOW 在 W40 的周一
W39_START = datetime(2026, 9, 21, tzinfo=CST)


def rec(
    sid: int,
    intent: Intent = SHIP,
    *,
    at: datetime | None = None,
    status: SuggestionStatus = S.SHOWN,
    shop_id: str = SHOP_A,
    text: str = "亲，48 小时内发货哦",
    question: str | None = "什么时候发货",
) -> SuggestionRecord:
    return SuggestionRecord(
        id=sid,
        shop_id=shop_id,
        conversation_id=CONV,
        msg_id=f"m{sid}",
        question_text=question,
        intent=intent,
        text=text,
        status=status,
        created_at=at or W39_START + timedelta(hours=sid % 150),
    )


def pool(spec: dict[Intent, int], *, start: int = 1) -> list[SuggestionRecord]:
    out, sid = [], start
    for intent, n in spec.items():
        for _ in range(n):
            out.append(rec(sid, intent))
            sid += 1
    return out


def scored(
    factual: int = 2,
    compliance: int = 2,
    *,
    intent: Intent = SHIP,
    rid: int = 1,
    relevance: int = 2,
    expression: int = 2,
) -> QualityReview:
    return QualityReview(
        id=rid,
        shop_id=SHOP_A,
        period=W39,
        suggestion_id=rid,
        intent=intent,
        status=Q.SCORED,
        scores=QualityScores(
            factual=factual, relevance=relevance, compliance=compliance, expression=expression
        ),
        reviewer="qa_reviewer_01",
        sampled_at=DEFAULT_NOW,
        reviewed_at=DEFAULT_NOW,
    )


# ================================================================ 周期（ISO 周，东八区）


def test_iso_period_boundaries_are_in_cst():
    """周日 23:59（东八区）还是上一周，周一 00:00 是新的一周；UTC 时间先换到东八区再算。"""
    sunday_night = datetime(2026, 9, 27, 23, 59, 59, tzinfo=CST)
    monday = datetime(2026, 9, 28, 0, 0, tzinfo=CST)
    assert iso_period(sunday_night) == W39
    assert iso_period(monday) == "2026-W40"
    # UTC 周日 16:00 = 东八区周一 00:00
    assert iso_period(datetime(2026, 9, 27, 16, 0, tzinfo=UTC)) == "2026-W40"
    assert iso_period(datetime(2026, 9, 27, 15, 59, tzinfo=UTC)) == W39
    start, end = period_range(W39)
    assert start == W39_START and end == monday
    assert start <= sunday_night < end and not monday < end
    with pytest.raises(ValueError):
        iso_period(datetime(2026, 9, 28))  # 不带时区


def test_period_range_and_year_edges():
    assert period_range("2026-W01")[0] == datetime(2025, 12, 29, tzinfo=CST)  # ISO 年从周一开始
    assert period_range("2026-W53")[1] == datetime(2027, 1, 4, tzinfo=CST)  # 2026 有 53 周
    assert iso_period(datetime(2027, 1, 3, 12, tzinfo=CST)) == "2026-W53"
    for bad in (
        "2025-W53",
        "2026-W54",
        "2026-W00",
        "2026W40",
        "2026-W4",
        "",
        "2026-W40\n",
        "２０２６-W39",  # 全角数字：库里的 CHECK 只认 ASCII，这里也不认
        "9999-W52",  # 周的结束超出日期范围：按不合法处理，不是 500
    ):
        with pytest.raises(ValueError):
            period_range(bad)
    assert check_period(W39) == W39


def test_default_period_is_the_previous_full_week():
    assert previous_period(DEFAULT_NOW) == W39
    assert previous_period(datetime(2026, 9, 27, 23, 59, tzinfo=CST)) == "2026-W38"
    assert previous_period(datetime(2026, 1, 1, 9, tzinfo=CST)) == "2025-W52"
    assert recent_periods(DEFAULT_NOW, 3) == ["2026-W37", "2026-W38", W39]
    assert week_start(DEFAULT_NOW) == datetime(2026, 9, 28, tzinfo=CST)
    assert periods_between(W39_START, DEFAULT_NOW) == [W39, "2026-W40"]
    assert periods_between(DEFAULT_NOW, DEFAULT_NOW) == ["2026-W40"]


# ================================================================ 抽样池与分配


def test_eligible_records_are_every_generated_suggestion_in_the_period():
    inside = [rec(i + 1, status=s) for i, s in enumerate(S)]
    assert {r.status for r in inside} == SAMPLE_STATUSES == set(S)
    before = rec(20, at=W39_START - timedelta(seconds=1))
    at_end = rec(21, at=W39_START + timedelta(days=7))  # 下周一 00:00 不含
    last = rec(22, at=W39_START + timedelta(days=7) - timedelta(seconds=1))
    other_shop = rec(23, shop_id=SHOP_B)
    got = eligible_records([*inside, before, at_end, last, other_shop], shop_id=SHOP_A, period=W39)
    assert [r.id for r in got] == [1, 2, 3, 4, 5, 6, 22]


def test_allocate_is_proportional_with_at_least_one_per_intent():
    assert allocate({SHIP: 60, SIZE: 30, STOCK: 10}, 10) == {SHIP: 6, SIZE: 3, STOCK: 1}
    # 最大余数：6.6 / 3.3 / 0.1 → 6 / 3 / 0 → 余 1 给余数最大的 SHIP；STOCK 至少一条，从超出比例
    # 最多的 SHIP 挪
    assert allocate({SHIP: 66, SIZE: 33, STOCK: 1}, 10) == {SHIP: 6, SIZE: 3, STOCK: 1}
    alloc = allocate({SHIP: 91, SIZE: 9}, 10)
    assert alloc == {SHIP: 9, SIZE: 1}
    # 总数不超过 size：全取
    assert allocate({SHIP: 3, SIZE: 2}, 50) == {SHIP: 3, SIZE: 2}
    # size 比类型数少：没法每类一条，按比例
    small = allocate({SHIP: 10, SIZE: 5, STOCK: 5, INVOICE: 5}, 2)
    assert sum(small.values()) == 2 and small[SHIP] == 1
    # 余数相同按 Intent 定义顺序（PRODUCT_SPEC 在 SIZE_ADVICE 之前）
    assert allocate({Intent.PRODUCT_SPEC: 1, SIZE: 1, STOCK: 1}, 2) == {
        Intent.PRODUCT_SPEC: 1,
        SIZE: 1,
        STOCK: 0,
    }
    for spec, size in (({SHIP: 7, SIZE: 13, STOCK: 29, INVOICE: 1}, 17), ({SHIP: 5}, 3)):
        alloc = allocate(spec, size)
        assert sum(alloc.values()) == size and all(alloc[i] >= 1 for i in spec)
        assert all(alloc[i] <= spec[i] for i in spec)


# ================================================================ 分层抽样（§13.5）


def test_sample_is_stratified_deterministic_and_default_fifty():
    records = pool({SHIP: 60, SIZE: 30, STOCK: 9, INVOICE: 1})
    plan = plan_sample(records, [], shop_id=SHOP_A, period=W39, now=DEFAULT_NOW)
    assert len(plan) == DEFAULT_SAMPLE_SIZE == 50
    assert Counter(q.intent for q in plan) == {SHIP: 30, SIZE: 15, STOCK: 4, INVOICE: 1}
    assert all(
        q.status == Q.PENDING and q.period == W39 and q.shop_id == SHOP_A and q.id is None
        for q in plan
    )
    by_id = {r.id: r for r in records}
    assert all(by_id[q.suggestion_id].intent == q.intent for q in plan)
    # 同一店铺、同一周期：顺序打乱也抽中同一批（不用全局随机数）
    again = plan_sample(list(reversed(records)), [], shop_id=SHOP_A, period=W39, now=DEFAULT_NOW)
    assert {q.suggestion_id for q in again} == {q.suggestion_id for q in plan}
    # 层内按 sample_key 排序：抽中的是每个类型 key 最小的那几条
    ship_ids = sorted(
        (r.id for r in records if r.intent == SHIP), key=lambda i: sample_key(SHOP_A, W39, i)
    )
    assert {q.suggestion_id for q in plan if q.intent == SHIP} == set(ship_ids[:30])
    # 换个周期（同样的推荐 id）结果不同：排序键和周期有关
    assert sample_key(SHOP_A, W39, 1) != sample_key(SHOP_A, "2026-W40", 1)
    assert sample_key(SHOP_A, W39, 1) != sample_key(SHOP_B, W39, 1)


def test_sample_takes_everything_when_fewer_than_size():
    records = pool({SHIP: 3, SIZE: 2})
    plan = plan_sample(records, [], shop_id=SHOP_A, period=W39, now=DEFAULT_NOW)
    assert sorted(q.suggestion_id for q in plan) == [1, 2, 3, 4, 5]
    other = [rec(100, shop_id=SHOP_B)]
    assert plan_sample(other, [], shop_id=SHOP_A, period=W39, now=DEFAULT_NOW) == []


def _stored(plan: list[QualityReview], start: int = 1) -> list[QualityReview]:
    return [q.model_copy(update={"id": start + i}) for i, q in enumerate(plan)]


def test_resample_is_idempotent_and_tops_up_without_dropping():
    """重抽只补不换：已抽中的不删；够 size 时什么都不加；新推荐进来后补到 size。"""
    first_pool = pool({SHIP: 3, SIZE: 1})
    first = _stored(plan_sample(first_pool, [], shop_id=SHOP_A, period=W39, now=DEFAULT_NOW))
    assert len(first) == 4
    # 同样的数据再抽：没有新样本
    assert plan_sample(first_pool, first, shop_id=SHOP_A, period=W39, now=DEFAULT_NOW) == []
    # 又来了 40 条推荐：补到 10 条，原来的 4 条都还在（不在新计划里重复出现）
    more = first_pool + pool({SHIP: 20, STOCK: 20}, start=100)
    top_up = plan_sample(more, first, shop_id=SHOP_A, period=W39, now=DEFAULT_NOW, size=10)
    assert len(top_up) == 6
    assert not {q.suggestion_id for q in top_up} & {q.suggestion_id for q in first}
    # 补的是缺口最大的类型：目标分配 SHIP 5 / SIZE 1 / STOCK 4，已有 SHIP 3、SIZE 1
    total = Counter(q.intent for q in [*first, *top_up])
    assert total == allocate({SHIP: 23, SIZE: 1, STOCK: 20}, 10) == {SHIP: 5, SIZE: 1, STOCK: 4}
    # 已抽中的多于 size：什么都不加，也不删
    assert plan_sample(more, first, shop_id=SHOP_A, period=W39, now=DEFAULT_NOW, size=2) == []


def test_top_up_when_existing_samples_are_over_represented():
    """先抽时只有 SHIP，后来 SIZE 大量进来：补抽照样补到 size，缺口大的 SIZE 先补。"""
    early = pool({SHIP: 5})
    first = _stored(plan_sample(early, [], shop_id=SHOP_A, period=W39, now=DEFAULT_NOW, size=5))
    later = early + pool({SIZE: 45}, start=200)
    top_up = plan_sample(later, first, shop_id=SHOP_A, period=W39, now=DEFAULT_NOW, size=10)
    assert len(top_up) == 5 and {q.intent for q in top_up} == {SIZE}


def test_sample_size_is_bounded():
    for size in (0, 201):
        with pytest.raises(ValueError):
            plan_sample([], [], shop_id=SHOP_A, period=W39, now=DEFAULT_NOW, size=size)
    with pytest.raises(ValueError):
        plan_sample([], [], shop_id=SHOP_A, period="2026-W99", now=DEFAULT_NOW)


# ================================================================ 报表与 §13.2 门槛


def test_launch_gate_threshold_is_inclusive():
    """19/20 = 95% 刚好达标；18/20 不达标。"""
    reviews = [scored(rid=i + 1) for i in range(19)] + [scored(1, 2, rid=20)]
    gate = launch_gate(reviews)
    assert gate.status == GateStatus.PASSED
    assert gate.factual_full_rate == 0.95 and gate.factual_ok and gate.compliance_ok
    reviews[0] = scored(2, 1, rid=1)
    gate = launch_gate(reviews)
    assert gate.status == GateStatus.PASSED  # 事实、合规各错一条，都是 95%
    reviews[1] = scored(0, 2, rid=2)
    gate = launch_gate(reviews)
    assert gate.status == GateStatus.FAILED and not gate.factual_ok and gate.compliance_ok
    assert "事实正确" in gate.detail and "合规" not in gate.detail
    # 57/60 = 95%，用分数精确比较
    assert meets_rate(57, 60, 0.95) and not meets_rate(56, 60, 0.95) and not meets_rate(0, 0, 0.95)


def test_launch_gate_needs_enough_scored_samples():
    """已打分少于 20 条：数据不足（跳过、待复核的不算）。"""
    few = [scored(rid=i + 1) for i in range(19)]
    skipped = QualityReview(
        id=99,
        shop_id=SHOP_A,
        period=W39,
        suggestion_id=99,
        intent=SHIP,
        status=Q.SKIPPED,
        reviewer="qa_reviewer_01",
        sampled_at=DEFAULT_NOW,
        reviewed_at=DEFAULT_NOW,
    )
    pending = QualityReview(
        id=98, shop_id=SHOP_A, period=W39, suggestion_id=98, intent=SHIP, sampled_at=DEFAULT_NOW
    )
    gate = launch_gate([*few, skipped, pending])
    assert gate.status == GateStatus.INSUFFICIENT
    assert (gate.sampled, gate.scored, gate.skipped, gate.pending) == (21, 19, 1, 1)
    assert gate.factual_ok  # 已有数据照样算
    assert launch_gate([*few, skipped], min_scored=19).status == GateStatus.PASSED
    empty = launch_gate([])
    assert empty.status == GateStatus.INSUFFICIENT and empty.factual_full_rate is None
    assert launch_gate([scored(rid=1)], min_scored=0).status == GateStatus.PASSED  # 至少 1 条


def test_quality_report_counts_dimensions_and_intents_without_texts():
    reviews = [
        scored(2, 2, rid=1, expression=1),
        scored(1, 2, rid=2, intent=SIZE, relevance=0),
        scored(2, 0, rid=3, intent=SIZE),
        QualityReview(
            id=4, shop_id=SHOP_A, period=W39, suggestion_id=4, intent=SHIP, sampled_at=DEFAULT_NOW
        ),
        QualityReview(
            id=5,
            shop_id=SHOP_A,
            period=W39,
            suggestion_id=5,
            intent=STOCK,
            status=Q.SKIPPED,
            reviewer="qa_reviewer_01",
            note="推荐已无意义 13812345678",
            sampled_at=DEFAULT_NOW,
            reviewed_at=DEFAULT_NOW,
        ),
        scored(rid=6).model_copy(update={"period": "2026-W38"}),  # 别的周期
        scored(rid=7).model_copy(update={"shop_id": SHOP_B}),  # 别的店铺
    ]
    report = quality_report(reviews, shop_id=SHOP_A, period=W39)
    assert (report.sampled, report.scored, report.skipped, report.pending) == (5, 3, 1, 1)
    assert (report.since, report.until) == period_range(W39)
    factual = report.dimensions["factual"]
    assert (factual.scored, factual.full, factual.label) == (3, 2, "事实正确")
    assert factual.full_rate == pytest.approx(2 / 3) and factual.mean == pytest.approx(5 / 3)
    assert factual.distribution == {0: 0, 1: 1, 2: 2}
    assert report.dimensions["relevance"].mean == pytest.approx(4 / 3)
    assert report.dimensions["expression"].full == 2
    assert report.gate.status == GateStatus.INSUFFICIENT
    assert list(report.by_intent) == [SIZE, STOCK, SHIP]  # Intent 定义顺序
    size = report.by_intent[SIZE]
    assert (size.sampled, size.scored, size.label) == (2, 2, INTENT_LABELS[SIZE])
    assert size.dimensions["compliance"].full_rate == 0.5
    assert report.by_intent[STOCK].skipped == 1
    dumped = report.model_dump_json()
    assert "13812345678" not in dumped and "推荐已无意义" not in dumped
    # 不指定周期：本店铺全部样本
    assert quality_report(reviews, shop_id=SHOP_A).sampled == 6


# ================================================================ 复核列表（§9.2 兜底脱敏）


def test_review_item_masks_pii_defensively():
    """写入时本来已做最小化；直接用构造函数造一条带手机号的记录，列表照样打码。"""
    record = rec(
        7,
        text="亲，已经帮您备注了，有问题打 13812345678",
        question="我电话 13912345678，地址杭州市西湖区文三路 100 号，什么时候发货",
        status=S.EDITED_SENT,
    ).model_copy(update={"edited_text": "好的，联系 13812345678", "evidence_ids": ["faq:2"]})
    review = QualityReview(
        id=3, shop_id=SHOP_A, period=W39, suggestion_id=7, intent=SHIP, sampled_at=DEFAULT_NOW
    )
    item = review_item(review, record)
    dumped = item.model_dump_json()
    for raw in ("13812345678", "13912345678", "文三路"):
        assert raw not in dumped
    s = item.suggestion
    assert s is not None and "[手机号]" in s.text and "[手机号]" in (s.question_text or "")
    assert s.edited_text == "好的，联系 [手机号]"
    assert s.evidence_ids == ["faq:2"] and s.kind == DraftKind.LLM and s.status == S.EDITED_SENT
    assert item.intent_label == INTENT_LABELS[SHIP] and item.status == Q.PENDING
    # 推荐不属于这个店铺 / 不是这条：不带内容
    assert review_item(review, record.model_copy(update={"shop_id": SHOP_B})).suggestion is None
    assert review_item(review, record.model_copy(update={"id": 8})).suggestion is None
    assert review_item(review, None).suggestion is None
    with pytest.raises(ValueError):
        review_item(review.model_copy(update={"id": None}), record)
