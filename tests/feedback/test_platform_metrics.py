"""平台客服指标接入前后对比的口径（rodit.feedback.platform_metrics；方案 §1.3、§12.4、
§15 Phase 2）：来源优先级、按样本量加权、前后窗口、好坏方向、可靠性提示、周序列、总体结论、
接入日期的来源。全部是合成数据。"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pytest

from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import MetricSource, Platform, PlatformMetric
from rodit.core.models import AuditEvent, PlatformMetricSample
from rodit.core.platform_metrics import metrics_for
from rodit.feedback.platform_metrics import (
    DEFAULT_WINDOW_DAYS,
    FLAG_LABELS,
    MAX_HISTORY_DAYS,
    MIN_RELIABLE_SAMPLES,
    VERDICT_LABELS,
    GoLive,
    build_report,
    check_day,
    check_metric,
    check_sample_size,
    check_value,
    check_window_days,
    compare_metric,
    first_assist_day,
    overall_verdict,
    pick_daily,
    resolve_go_live,
    weekly_series,
    window_bounds,
    window_stats,
)
from rodit.testing.fixtures import (
    DEMO_GO_LIVE_DAYS_AGO,
    SHOP_A,
    SHOP_B,
    SHOP_D,
    build_demo_world,
    demo_platform_metrics,
)

TB = PlatformMetric.TAOBAO_3MIN_RESPONSE_RATE
DY_RESP = PlatformMetric.DOUYIN_AVG_RESPONSE_S
DY_DISSAT = PlatformMetric.DOUYIN_DISSATISFACTION_RATE
DY_HELP = PlatformMetric.DOUYIN_HELP_RATE
DY_AFTERSALE = PlatformMetric.DOUYIN_AFTERSALE_REVIEW_H
TODAY = date(2026, 9, 28)
GO_LIVE = TODAY - timedelta(days=28)
AT = datetime(2026, 9, 28, 10, 0, tzinfo=CST)

P, M, C = MetricSource.PLATFORM, MetricSource.MERCHANT, MetricSource.COMPUTED


def s(
    day: date,
    value: float,
    *,
    metric: PlatformMetric = TB,
    source: MetricSource = P,
    size: int | None = None,
    shop: str = SHOP_A,
) -> PlatformMetricSample:
    return PlatformMetricSample(
        shop_id=shop,
        metric=metric,
        day=day,
        value=value,
        source=source,
        sample_size=size,
        recorded_at=AT,
    )


def series(
    start: date, days: int, value: float, *, size: int | None = 200, **kw
) -> list[PlatformMetricSample]:
    return [s(start + timedelta(days=i), value, size=size, **kw) for i in range(days)]


def known(day: date | None = GO_LIVE) -> GoLive:
    return GoLive(day=day, source="query", label="按填写的接入日期")


# ================================================================ 取值校验


@pytest.mark.parametrize(
    ("metric", "value", "ok"),
    [
        (TB, 0.0, True),
        (TB, 1.0, True),
        (TB, 0.915, True),
        (TB, 1.01, False),  # 比例超过 1（填成了百分数）
        (TB, 91.5, False),
        (TB, -0.01, False),
        (DY_RESP, 0.0, True),
        (DY_RESP, 86400.0, True),
        (DY_RESP, 86400.5, False),
        (DY_AFTERSALE, 720.0, True),
        (DY_AFTERSALE, 721.0, False),
        (TB, math.inf, False),
        (TB, math.nan, False),
    ],
)
def test_check_value_bounds_by_unit(metric, value, ok):
    if ok:
        assert check_value(metric, value) == value
    else:
        with pytest.raises(ValueError):
            check_value(metric, value)


def test_check_value_messages_explain_the_unit():
    with pytest.raises(ValueError, match="0–1 之间的比例"):
        check_value(TB, 91.0)
    with pytest.raises(ValueError, match="秒"):
        check_value(DY_RESP, 1e9)
    with pytest.raises(ValueError, match="不合法"):
        check_value(TB, "很多")  # type: ignore[arg-type]


def test_check_metric_only_accepts_this_platform():
    assert check_metric(Platform.TAOBAO, "taobao_3min_response_rate") == TB
    with pytest.raises(ValueError, match="不是本平台考核的指标"):
        check_metric(Platform.TAOBAO, DY_HELP)
    with pytest.raises(ValueError, match="未知的平台指标"):
        check_metric(Platform.PDD, "pdd_nope")
    for platform in Platform:
        for m in metrics_for(platform):
            assert check_metric(platform, m.value) == m


@pytest.mark.parametrize("bad", [-1, 10_000_001, True, 1.5, "100"])
def test_check_sample_size_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        check_sample_size(bad)  # type: ignore[arg-type]


def test_check_sample_size_accepts_none_and_range():
    assert check_sample_size(None) is None
    assert check_sample_size(0) == 0 and check_sample_size(10_000_000) == 10_000_000


@pytest.mark.parametrize("bad", [6, 91, 0, True, 28.0])
def test_check_window_days_range(bad):
    with pytest.raises(ValueError):
        check_window_days(bad)  # type: ignore[arg-type]
    assert check_window_days(7) == 7 and check_window_days(90) == 90


def test_check_day_not_future_and_within_history():
    assert check_day(TODAY, TODAY) == TODAY
    oldest = TODAY - timedelta(days=MAX_HISTORY_DAYS)
    assert check_day(oldest, TODAY) == oldest
    with pytest.raises(ValueError, match="不能晚于今天"):
        check_day(TODAY + timedelta(days=1), TODAY)
    with pytest.raises(ValueError, match=f"{MAX_HISTORY_DAYS} 天"):
        check_day(oldest - timedelta(days=1), TODAY, "登记的日期")


# ================================================================ 来源优先级、窗口、加权


def test_pick_daily_prefers_platform_then_merchant_then_computed():
    d1, d2, d3 = GO_LIVE, GO_LIVE + timedelta(days=1), GO_LIVE + timedelta(days=2)
    rows = [
        s(d1, 0.5, source=C),
        s(d1, 0.6, source=M),
        s(d1, 0.7, source=P),  # 三个来源都有：平台
        s(d2, 0.8, source=M),
        s(d2, 0.4, source=C),  # 商家 > 估算（先后无关）
        s(d3, 0.3, source=C),  # 只有估算
        s(d3, 0.9, metric=DY_HELP, source=M),  # 另一项指标各自取
    ]
    for order in (rows, list(reversed(rows))):
        daily = pick_daily(order)
        assert daily[(TB, d1)].value == 0.7
        assert daily[(TB, d2)].value == 0.8
        assert daily[(TB, d3)].value == 0.3 and daily[(TB, d3)].source == C
        assert daily[(DY_HELP, d3)].value == 0.9
        assert len(daily) == 4


def test_pick_daily_skips_non_finite_values():
    daily = pick_daily([s(GO_LIVE, math.inf, source=P), s(GO_LIVE, 0.6, source=M)])
    assert daily[(TB, GO_LIVE)].value == 0.6  # 平台的数不是有限数：退到商家登记的


def test_window_bounds_go_live_day_counts_as_after_and_after_is_capped_at_today():
    assert window_bounds(GO_LIVE, 28, TODAY) == (
        GO_LIVE - timedelta(days=28),
        GO_LIVE - timedelta(days=1),
        GO_LIVE,
        GO_LIVE + timedelta(days=27),  # = TODAY - 1
    )
    recent = TODAY - timedelta(days=5)
    assert window_bounds(recent, 28, TODAY)[3] == TODAY
    assert window_bounds(TODAY, 7, TODAY)[2:] == (TODAY, TODAY)


def test_window_mean_is_weighted_by_sample_size_when_every_day_has_one():
    rows = [s(GO_LIVE, 0.5, size=100), s(GO_LIVE + timedelta(days=1), 1.0, size=300)]
    stats = window_stats(pick_daily(rows), TB, GO_LIVE, GO_LIVE + timedelta(days=6))
    assert stats.weighted is True and stats.sample_size == 400
    assert stats.mean == pytest.approx((0.5 * 100 + 1.0 * 300) / 400)
    assert stats.days == 2 and stats.span_days == 7 and stats.sources == {P: 2}


def test_window_mean_falls_back_to_plain_average_when_a_size_is_missing_or_all_zero():
    rows = [s(GO_LIVE, 0.5, size=100), s(GO_LIVE + timedelta(days=1), 1.0, size=None)]
    stats = window_stats(pick_daily(rows), TB, GO_LIVE, GO_LIVE + timedelta(days=1))
    assert stats.weighted is False and stats.sample_size is None
    assert stats.mean == pytest.approx(0.75)
    zero = [s(GO_LIVE, 0.2, size=0), s(GO_LIVE + timedelta(days=1), 0.4, size=0)]
    stats = window_stats(pick_daily(zero), TB, GO_LIVE, GO_LIVE + timedelta(days=1))
    assert stats.weighted is False and stats.sample_size == 0
    assert stats.mean == pytest.approx(0.3)


def test_window_stats_counts_only_days_inside_the_window_and_sources_in_order():
    rows = [
        s(GO_LIVE - timedelta(days=1), 0.1),  # 窗口外
        s(GO_LIVE, 0.6, source=M),
        s(GO_LIVE + timedelta(days=1), 0.8, source=C),
        s(GO_LIVE + timedelta(days=2), 0.7, source=P),
    ]
    stats = window_stats(pick_daily(rows), TB, GO_LIVE, GO_LIVE + timedelta(days=2))
    assert stats.days == 3 and stats.mean == pytest.approx(0.7)
    assert list(stats.sources.items()) == [(P, 1), (M, 1), (C, 1)]
    empty = window_stats({}, TB, GO_LIVE, GO_LIVE + timedelta(days=6))
    assert empty.days == 0 and empty.mean is None and empty.sources == {}


# ================================================================ 方向、可靠性


def _compare(rows, metric=TB, *, go_live=GO_LIVE, window=28, today=TODAY):
    return compare_metric(
        pick_daily(rows), metric, go_live=go_live, window_days=window, today=today
    )


def _before_after(metric, before, after, *, size=200, **kw):
    start = GO_LIVE - timedelta(days=28)
    return series(start, 28, before, metric=metric, size=size, **kw) + series(
        GO_LIVE, 28, after, metric=metric, size=size, **kw
    )


def test_higher_is_better_metric_going_up_is_improved():
    c = _compare(_before_after(TB, 0.78, 0.91))
    assert c.direction == "improved" and c.direction_label == "变好"
    assert c.before.mean == pytest.approx(0.78) and c.after.mean == pytest.approx(0.91)
    # 报表里的数保留 6 位小数
    assert c.delta == pytest.approx(0.13) and c.relative_change == round(0.13 / 0.78, 6)
    assert c.reliable and c.flags == [] and c.higher_is_better is True
    assert c.label == "旺旺 3 分钟人工响应率" and c.unit == "ratio" and c.note


def test_lower_is_better_metric_going_down_is_improved_and_up_is_worse():
    faster = _compare(_before_after(DY_RESP, 48.0, 26.0), DY_RESP)
    assert faster.direction == "improved" and faster.delta == pytest.approx(-22.0)
    slower = _compare(_before_after(DY_RESP, 26.0, 48.0), DY_RESP)
    assert slower.direction == "worse" and slower.direction_label == "变差"
    lower_rate = _compare(_before_after(TB, 0.9, 0.8))
    assert lower_rate.direction == "worse"


def test_changes_under_one_percent_are_unchanged():
    c = _compare(_before_after(TB, 0.900, 0.905))  # 相对变化 0.56%
    assert c.direction == "unchanged" and c.delta == pytest.approx(0.005)
    c = _compare(_before_after(TB, 0.900, 0.910))  # 相对变化 1.1%
    assert c.direction == "improved"


def test_zero_before_value_has_no_relative_change():
    same = _compare(_before_after(DY_HELP, 0.0, 0.0), DY_HELP)
    assert same.direction == "unchanged" and same.relative_change is None and same.delta == 0
    up = _compare(_before_after(DY_HELP, 0.0, 0.01), DY_HELP)
    assert up.direction == "worse" and up.relative_change is None  # 求助率越低越好


def test_missing_window_is_insufficient_with_flags():
    only_after = _compare(series(GO_LIVE, 28, 0.9))
    assert only_after.direction == "insufficient" and only_after.delta is None
    assert only_after.flags == ["no_before"] and not only_after.reliable
    assert only_after.flag_notes == [FLAG_LABELS["no_before"]]
    nothing = _compare([])
    assert nothing.flags == ["no_before", "no_after"]
    assert nothing.before.mean is None and nothing.after.mean is None


def test_few_days_and_small_samples_are_flagged_but_still_compared():
    rows = series(GO_LIVE - timedelta(days=3), 3, 0.8, size=10) + series(GO_LIVE, 28, 0.9)
    c = _compare(rows)
    assert c.direction == "improved"  # 结论照算，只提示
    assert c.flags == ["few_days_before", "small_sample_before"]
    assert not c.reliable
    assert c.before.sample_size == 30 < MIN_RELIABLE_SAMPLES


def test_unknown_sample_sizes_are_not_flagged_as_small():
    c = _compare(_before_after(TB, 0.8, 0.9, size=None))
    assert c.before.sample_size is None and c.before.weighted is False
    assert c.flags == [] and c.reliable


def test_sources_differing_between_windows_and_computed_values_are_flagged():
    rows = series(GO_LIVE - timedelta(days=28), 28, 0.8, source=M) + series(GO_LIVE, 28, 0.9)
    c = _compare(rows)
    assert c.flags == ["sources_differ"] and c.before.sources == {M: 28}
    rows = series(GO_LIVE - timedelta(days=28), 28, 0.8, source=C) + series(
        GO_LIVE, 28, 0.9, source=C
    )
    c = _compare(rows)
    assert c.flags == ["computed"]  # 前后都是估算：来源相同，但口径近似


# ================================================================ 周序列


def test_weekly_series_splits_at_go_live_without_crossing_it():
    rows = _before_after(TB, 0.8, 0.9)
    points = weekly_series(pick_daily(rows), TB, GO_LIVE, 28, TODAY)
    assert [p.index for p in points] == [-4, -3, -2, -1, 0, 1, 2, 3]
    assert [p.phase for p in points] == ["before"] * 4 + ["after"] * 4
    assert points[3].end == GO_LIVE - timedelta(days=1) and points[4].start == GO_LIVE
    assert points[0].start == GO_LIVE - timedelta(days=28)
    assert all(p.days == 7 for p in points)
    assert [round(p.mean, 6) for p in points] == [0.8] * 4 + [0.9] * 4


def test_weekly_series_clips_partial_weeks_to_the_window():
    go_live = TODAY - timedelta(days=20)
    rows = series(go_live - timedelta(days=10), 21, 0.8, size=None)
    points = weekly_series(pick_daily(rows), TB, go_live, 10, TODAY)
    # 前后各 10 天：第 -2 周只剩 3 天；接入后第 1 周截到接入后第 10 天
    assert [(p.index, p.start, p.end) for p in points] == [
        (-2, go_live - timedelta(days=10), go_live - timedelta(days=8)),
        (-1, go_live - timedelta(days=7), go_live - timedelta(days=1)),
        (0, go_live, go_live + timedelta(days=6)),
        (1, go_live + timedelta(days=7), go_live + timedelta(days=9)),
    ]
    assert [p.days for p in points] == [3, 7, 7, 3]
    empty = weekly_series({}, TB, go_live, 10, TODAY)
    assert len(empty) == 4 and all(p.mean is None and p.days == 0 for p in empty)


def test_weekly_series_skips_weeks_that_have_not_started():
    go_live = TODAY - timedelta(days=10)
    rows = series(go_live, 11, 0.9)
    points = weekly_series(pick_daily(rows), TB, go_live, 28, TODAY)
    after = [p for p in points if p.phase == "after"]
    # 接入第 11 天：第 0 周 7 天、第 1 周到今天 4 天，第 2、3 周还没开始
    assert [(p.index, p.start, p.end, p.days) for p in after] == [
        (0, go_live, go_live + timedelta(days=6), 7),
        (1, go_live + timedelta(days=7), TODAY, 4),
    ]
    assert [p.index for p in points if p.phase == "before"] == [-4, -3, -2, -1]


# ================================================================ 总体结论


def _with(direction):
    c = _compare(_before_after(TB, 0.8, 0.9))
    return c.model_copy(update={"direction": direction})


@pytest.mark.parametrize(
    ("directions", "verdict"),
    [
        (["improved", "improved"], "improved"),
        (["improved", "unchanged"], "improved"),
        (["improved", "insufficient"], "improved"),
        (["worse", "unchanged"], "worse"),
        (["improved", "worse"], "mixed"),
        (["unchanged", "unchanged"], "unchanged"),
        (["insufficient", "insufficient"], "insufficient"),
        ([], "insufficient"),
    ],
)
def test_overall_verdict(directions, verdict):
    assert overall_verdict([_with(d) for d in directions]) == verdict


# ================================================================ 报表


def test_report_on_demo_douyin_data_shows_every_metric_improved():
    world = build_demo_world()
    today = world.now.astimezone(CST).date()
    go_live = today - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)
    report = build_report(
        shop_id=SHOP_D,
        platform=Platform.DOUYIN,
        samples=demo_platform_metrics(world),
        go_live=known(go_live),
        today=today,
    )
    assert report.verdict == "improved" and report.verdict_label == VERDICT_LABELS["improved"]
    assert [c.metric for c in report.metrics] == metrics_for(Platform.DOUYIN)
    assert all(c.direction == "improved" for c in report.metrics)
    assert report.reliable and report.notes == []
    assert report.counts == {"improved": 4, "worse": 0, "unchanged": 0, "insufficient": 0}
    by = {c.metric: c for c in report.metrics}
    assert by[DY_RESP].before.mean == pytest.approx(48.0, rel=0.02)
    assert by[DY_RESP].after.mean == pytest.approx(26.0, rel=0.02)
    assert by[DY_AFTERSALE].before.sample_size == 25 * 28  # 每天 25 单售后
    assert all(c.before.days == c.after.days == 28 for c in report.metrics)
    assert all(len(c.weekly) == 8 for c in report.metrics)
    assert (report.before_start, report.after_end) == (
        go_live - timedelta(days=28),
        today - timedelta(days=1),
    )


def test_report_only_uses_this_shop_and_this_platforms_metrics():
    rows = _before_after(TB, 0.8, 0.9)
    noise = _before_after(TB, 0.1, 0.1, shop=SHOP_B) + _before_after(DY_HELP, 0.9, 0.1)
    report = build_report(
        shop_id=SHOP_A,
        platform=Platform.TAOBAO,
        samples=rows + noise,
        go_live=known(),
        today=TODAY,
    )
    [c] = report.metrics
    assert c.metric == TB and c.after.mean == pytest.approx(0.9)
    assert report.verdict == "improved"


def test_report_without_data_is_insufficient_and_says_what_is_missing():
    report = build_report(
        shop_id=SHOP_D, platform=Platform.DOUYIN, samples=[], go_live=known(), today=TODAY
    )
    assert report.verdict == "insufficient" and not report.reliable
    assert report.counts["insufficient"] == 4
    assert "飞鸽平均响应时长" in report.notes[0] and "手工登记" in report.notes[0]


def test_report_with_unknown_go_live_asks_for_one():
    report = build_report(
        shop_id=SHOP_A,
        platform=Platform.TAOBAO,
        samples=_before_after(TB, 0.8, 0.9),
        go_live=GoLive(day=None, source="unknown", label="接入日期未知"),
        today=TODAY,
    )
    assert report.verdict == "no_go_live" and report.metrics == []
    assert report.go_live is None and report.before_start is None
    assert "请填写接入日期" in report.notes[0]


def test_report_notes_unreliable_metrics_and_a_short_after_window():
    go_live = TODAY - timedelta(days=3)
    rows = series(go_live - timedelta(days=28), 28, 0.8) + series(go_live, 4, 0.9)
    report = build_report(
        shop_id=SHOP_A,
        platform=Platform.TAOBAO,
        samples=rows,
        go_live=known(go_live),
        today=TODAY,
    )
    assert report.verdict == "improved" and not report.reliable
    assert report.metrics[0].flags == ["few_days_after"]
    assert any("仅供参考" in n for n in report.notes)
    assert any("接入后只有 4 天" in n for n in report.notes)


def test_report_rejects_future_go_live_and_bad_window():
    with pytest.raises(ValueError, match="不能晚于今天"):
        build_report(
            shop_id=SHOP_A,
            platform=Platform.TAOBAO,
            samples=[],
            go_live=known(TODAY + timedelta(days=1)),
            today=TODAY,
        )
    with pytest.raises(ValueError):
        build_report(
            shop_id=SHOP_A,
            platform=Platform.TAOBAO,
            samples=[],
            go_live=known(),
            window_days=200,
            today=TODAY,
        )


def test_report_default_window_is_28_days():
    report = build_report(
        shop_id=SHOP_A, platform=Platform.TAOBAO, samples=[], go_live=known(), today=TODAY
    )
    assert report.window_days == DEFAULT_WINDOW_DAYS == 28


# ================================================================ 接入日期


def _stage(at: datetime, to: str, shop: str = SHOP_A) -> AuditEvent:
    return AuditEvent(
        shop_id=shop,
        actor="qa_reviewer_01",
        action=AuditAction.ROLLOUT_STAGE.value,
        detail={"from": "shadow" if to == "assist" else "assist", "to": to},
        created_at=at,
    )


def test_go_live_prefers_explicit_then_first_switch_to_assist_then_authorization():
    first = datetime(2026, 8, 31, 9, 0, tzinfo=CST)
    events = [
        _stage(first + timedelta(days=10), "assist"),
        _stage(first - timedelta(days=3), "shadow"),  # 切到影子不算
        _stage(first, "assist"),  # 第一次切到智能辅助（不按列表顺序）
    ]
    granted = datetime(2026, 7, 1, 12, 0, tzinfo=CST)
    explicit = resolve_go_live(date(2026, 8, 1), stage_events=events, granted_at=granted)
    assert (explicit.day, explicit.source) == (date(2026, 8, 1), "query")
    rollout = resolve_go_live(None, stage_events=events, granted_at=granted)
    assert (rollout.day, rollout.source) == (date(2026, 8, 31), "rollout")
    assert "智能辅助" in rollout.label
    auth = resolve_go_live(None, stage_events=[events[1]], granted_at=granted)
    assert (auth.day, auth.source) == (date(2026, 7, 1), "authorization")
    unknown = resolve_go_live(None)
    assert (unknown.day, unknown.source) == (None, "unknown")
    assert "请填写接入日期" in unknown.label


def test_go_live_dates_are_beijing_dates():
    # UTC 8 月 30 日 17:00 = 北京时间 8 月 31 日 1:00
    utc_evening = datetime.fromisoformat("2026-08-30T17:00:00+00:00")
    assert first_assist_day([_stage(utc_evening, "assist")]) == date(2026, 8, 31)
    assert resolve_go_live(None, granted_at=utc_evening).day == date(2026, 8, 31)


def test_first_assist_day_ignores_other_actions():
    other = AuditEvent(
        shop_id=SHOP_A,
        actor="owner_a",
        action=AuditAction.CONFIG_UPDATED.value,
        detail={"to": "assist"},
        created_at=AT,
    )
    assert first_assist_day([other]) is None
