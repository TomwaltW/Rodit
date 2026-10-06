"""平台客服指标的应用服务（rodit.app.platform_metrics；方案 §12.4、§15 Phase 2、§8.4、§9.2）：
平台同步（授权检查、整段核对后才写、故障时什么都不写、串店拒绝、逐条跳过）、商家登记（校验、覆盖、
没变不写）、列表、接入日期的来源、报表；写操作在 processing 闸门里、记审计；取消订购一并删除。
内存演示单元 + 合成数据，不联网。"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta

import pytest
from pydantic import SecretStr

from rodit.app.demo import build_demo_unit
from rodit.app.platform_metrics import (
    DEFAULT_SYNC_DAYS,
    TENANT_STAGE_SYNC,
    MetricsSourceMissing,
    PlatformMetricsService,
    PlatformMetricsUnavailable,
)
from rodit.app.service import AssistService
from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import MetricSource, Platform, PlatformMetric, ShopStatus
from rodit.core.errors import AuthExpired, ShopInactive, TenantIsolationError, ToolUnavailable
from rodit.core.models import PlatformMetricSample, ShopAuth
from rodit.core.scopes import MVP_SCOPES
from rodit.testing.fakes import parse_subscription, subscription_push
from rodit.testing.fixtures import (
    DEMO_GO_LIVE_DAYS_AGO,
    SHOP_A,
    SHOP_B,
    SHOP_C,
    SHOP_D,
)

TB = PlatformMetric.TAOBAO_3MIN_RESPONSE_RATE
DY_HELP = PlatformMetric.DOUYIN_HELP_RATE
MERCHANT = "owner_a"


def make(platform: Platform = Platform.TAOBAO) -> tuple[AssistService, PlatformMetricsService]:
    service = AssistService(build_demo_unit(platform))
    return service, PlatformMetricsService(service.unit, service.processing)


def today_of(service: AssistService) -> date:
    return service.unit.clock.now().astimezone(CST).date()


def events(service: AssistService, shop_id: str, action: str) -> list:
    return service.unit.repos.audit.list_events(shop_id, action=action)


class ListSource:
    """平台指标接口的替身：原样返回给定的数据（核对逐条跳过、串店）。"""

    def __init__(self, platform: Platform, rows: list[PlatformMetricSample]) -> None:
        self.platform = platform
        self.rows = rows
        self.calls: list[tuple[str, date, date]] = []

    def fetch(self, shop_id: str, since: date, until: date) -> list[PlatformMetricSample]:
        self.calls.append((shop_id, since, until))
        return [r.model_copy() for r in self.rows]


def sample(
    shop_id: str,
    day: date,
    value: float,
    *,
    metric: PlatformMetric = TB,
    source: MetricSource = MetricSource.PLATFORM,
    size: int | None = 100,
    at: datetime,
) -> PlatformMetricSample:
    return PlatformMetricSample(
        shop_id=shop_id,
        metric=metric,
        day=day,
        value=value,
        source=source,
        sample_size=size,
        recorded_at=at,
    )


# ================================================================ 平台同步


def test_sync_writes_platform_rows_for_the_default_range_and_audits():
    service, metrics = make()
    today = today_of(service)
    result = metrics.sync(SHOP_A, actor="kf_mgr")
    assert (result.since, result.until) == (today - timedelta(days=56), today - timedelta(days=1))
    assert result.fetched == result.written == DEFAULT_SYNC_DAYS == 56
    assert result.skipped == {} and result.metrics == [TB] and result.days == 56
    assert service.unit.metrics_source.calls == [(SHOP_A, result.since, result.until)]
    rows = service.unit.repos.platform_metrics.list(SHOP_A)
    assert len(rows) == 56 and {r.source for r in rows} == {MetricSource.PLATFORM}
    assert all(r.recorded_by is None and r.recorded_at == service.unit.clock.now() for r in rows)
    assert all(r.sample_size == 260 for r in rows)
    [event] = events(service, SHOP_A, AuditAction.PLATFORM_METRICS_SYNCED.value)
    assert event.actor == "kf_mgr"
    assert event.detail == {
        "since": result.since.isoformat(),
        "until": result.until.isoformat(),
        "fetched": 56,
        "written": 56,
        "skipped": {},
        "metrics": ["taobao_3min_response_rate"],
    }
    assert service.unit.repos.platform_metrics.list(SHOP_B) == []  # 只写这家店


def test_sync_is_idempotent_and_leaves_merchant_entries_alone():
    service, metrics = make()
    day = today_of(service) - timedelta(days=3)
    metrics.record(SHOP_A, metric=TB, day=day, value=0.5, merchant=MERCHANT)
    metrics.sync(SHOP_A, actor="system", days=7)
    first = service.unit.repos.platform_metrics.list(SHOP_A)
    metrics.sync(SHOP_A, actor="ops", days=7)
    second = service.unit.repos.platform_metrics.list(SHOP_A)
    assert [(r.day, r.source, r.value) for r in first] == [
        (r.day, r.source, r.value) for r in second
    ]
    merchant = [r for r in second if r.source == MetricSource.MERCHANT]
    assert [(r.day, r.value, r.recorded_by) for r in merchant] == [(day, 0.5, MERCHANT)]
    assert len(second) == 8  # 7 天平台数据 + 1 条商家登记
    assert [
        e.actor for e in events(service, SHOP_A, AuditAction.PLATFORM_METRICS_SYNCED.value)
    ] == ["system", "ops"]


def test_sync_on_douyin_writes_all_four_metrics():
    service, metrics = make(Platform.DOUYIN)
    result = metrics.sync(SHOP_D, actor="system")
    assert result.written == 4 * 56
    assert [m.value for m in result.metrics] == [
        "douyin_avg_response_s",
        "douyin_dissatisfaction_rate",
        "douyin_help_rate",
        "douyin_aftersale_review_h",
    ]


def test_sync_failure_writes_nothing():
    service, metrics = make()
    service.unit.metrics_source.fail = True
    with pytest.raises(ToolUnavailable):
        metrics.sync(SHOP_A, actor="system")
    assert service.unit.repos.platform_metrics.list(SHOP_A) == []
    assert events(service, SHOP_A, AuditAction.PLATFORM_METRICS_SYNCED.value) == []


def test_sync_without_source_or_repository_is_a_clear_error():
    service, metrics = make()
    service.unit.metrics_source = None
    with pytest.raises(MetricsSourceMissing, match="手工登记"):
        metrics.sync(SHOP_A, actor="system")
    service.unit.repos.platform_metrics = None
    for call in (
        lambda: metrics.sync(SHOP_A, actor="system"),
        lambda: metrics.report(SHOP_A),
        lambda: metrics.list_samples(SHOP_A),
        lambda: metrics.record(SHOP_A, metric=TB, day=today_of(service), value=0.9, merchant="o"),
    ):
        with pytest.raises(PlatformMetricsUnavailable):
            call()


def test_sync_checks_authorization_before_calling_the_platform():
    service, metrics = make()
    source = service.unit.metrics_source
    service.unit.repos.auth.delete(SHOP_A)
    with pytest.raises(AuthExpired):
        metrics.sync(SHOP_A, actor="system")
    # 缺权限也不行
    service.unit.repos.auth.save(
        ShopAuth(
            shop_id=SHOP_A,
            access_token=SecretStr("tok-synthetic"),
            scopes=sorted(MVP_SCOPES - {"item.read"}),
            expires_at=service.unit.clock.now() + timedelta(days=1),
        )
    )
    with pytest.raises(AuthExpired):
        metrics.sync(SHOP_A, actor="system")
    assert source.calls == [] and service.unit.repos.platform_metrics.list(SHOP_A) == []


def test_sync_refuses_suspended_or_deleted_shops():
    service, metrics = make()
    shop = service.unit.repos.knowledge.get_shop(SHOP_A)
    service.unit.repos.knowledge.upsert_shop(
        shop.model_copy(update={"status": ShopStatus.SUSPENDED})
    )
    with pytest.raises(ShopInactive):
        metrics.sync(SHOP_A, actor="system")
    with pytest.raises(ShopInactive):
        metrics.sync("s_nope", actor="system")
    assert service.unit.metrics_source.calls == []


def test_sync_rejects_rows_of_another_shop_and_writes_nothing():
    service, metrics = make()
    today = today_of(service)
    now = service.unit.clock.now()
    source = ListSource(
        Platform.TAOBAO,
        [
            sample(SHOP_A, today - timedelta(days=2), 0.9, at=now),
            sample(SHOP_B, today - timedelta(days=2), 0.1, at=now),
        ],
    )
    service.unit.metrics_source = source
    with pytest.raises(TenantIsolationError):
        metrics.sync(SHOP_A, actor="system", days=7)
    repo = service.unit.repos.platform_metrics
    assert repo.list(SHOP_A) == [] and repo.list(SHOP_B) == []
    [violation] = events(service, SHOP_A, AuditAction.TENANT_VIOLATION.value)
    assert violation.detail == {"stage": TENANT_STAGE_SYNC}  # 只记在哪一步拦下的
    assert events(service, SHOP_A, AuditAction.PLATFORM_METRICS_SYNCED.value) == []


def test_sync_skips_bad_rows_one_by_one_and_counts_them():
    service, metrics = make()
    today = today_of(service)
    now = service.unit.clock.now()
    good = today - timedelta(days=2)
    rows = [
        sample(SHOP_A, good, 0.9, at=now),
        sample(SHOP_A, good, 0.8, at=now),  # 同一天重复：只写第一条
        sample(SHOP_A, good, 0.1, metric=DY_HELP, at=now),  # 别的平台的指标
        sample(SHOP_A, today, 0.9, at=now),  # 今天：不在范围内（到昨天为止）
        sample(SHOP_A, today - timedelta(days=30), 0.9, at=now),  # 早于范围
        sample(SHOP_A, today - timedelta(days=3), 0.9, source=MetricSource.MERCHANT, at=now),
        sample(SHOP_A, today - timedelta(days=4), 1.5, at=now),  # 比例超过 1
        sample(SHOP_A, today - timedelta(days=5), 0.9, size=20_000_000, at=now),  # 样本量不对
        sample(SHOP_A, today - timedelta(days=6), 0.95, at=now),
    ]
    service.unit.metrics_source = ListSource(Platform.TAOBAO, rows)
    result = metrics.sync(SHOP_A, actor="system", days=7)
    assert result.fetched == 9 and result.written == 2
    assert result.skipped == {
        "other_platform": 1,
        "out_of_range": 2,
        "not_platform_source": 1,
        "invalid_value": 2,
        "duplicate": 1,
    }
    stored = service.unit.repos.platform_metrics.list(SHOP_A)
    assert [(r.day, r.value, r.source) for r in stored] == [
        (today - timedelta(days=6), 0.95, MetricSource.PLATFORM),
        (good, 0.9, MetricSource.PLATFORM),
    ]
    [event] = events(service, SHOP_A, AuditAction.PLATFORM_METRICS_SYNCED.value)
    assert event.detail["skipped"] == result.skipped


@pytest.mark.parametrize("days", [0, -1, 401, True, 7.0])
def test_sync_days_must_be_1_to_400(days):
    service, metrics = make()
    with pytest.raises(ValueError):
        metrics.sync(SHOP_A, actor="system", days=days)
    assert service.unit.metrics_source.calls == []


@pytest.mark.parametrize("actor", ["", "  ", "bot", "auto", None])
def test_sync_needs_a_person_or_the_system(actor):
    _, metrics = make()
    with pytest.raises(ValueError):
        metrics.sync(SHOP_A, actor=actor)  # type: ignore[arg-type]


def test_sync_actor_control_characters_are_removed():
    service, metrics = make()
    metrics.sync(SHOP_A, actor="kf\x00_mgr", days=1)
    assert events(service, SHOP_A, AuditAction.PLATFORM_METRICS_SYNCED.value)[0].actor == "kf_mgr"


def test_writes_happen_inside_the_processing_gate():
    """同步、登记的写入都在 processing(shop_id) 里（和取消订购互斥，§8.4）。"""
    service, _ = make()
    inside: list[str] = []

    @contextmanager
    def gate(shop_id: str):
        inside.append(shop_id)
        try:
            yield
        finally:
            inside.remove(shop_id)

    repo = service.unit.repos.platform_metrics
    seen: list[bool] = []
    original = repo.upsert

    def upsert(s):
        seen.append(s.shop_id in inside)
        return original(s)

    repo.upsert = upsert  # type: ignore[method-assign]
    metrics = PlatformMetricsService(service.unit, gate)
    metrics.sync(SHOP_A, actor="system", days=3)
    metrics.record(SHOP_A, metric=TB, day=today_of(service), value=0.9, merchant=MERCHANT)
    assert seen == [True] * 4


def test_revoke_deletes_synced_and_recorded_metrics_and_later_writes_are_refused():
    service, metrics = make()
    metrics.sync(SHOP_A, actor="system", days=7)
    metrics.record(SHOP_A, metric=TB, day=today_of(service), value=0.9, merchant=MERCHANT)
    metrics.sync(SHOP_B, actor="system", days=7)
    counts = service.revoke_shop(SHOP_A)
    assert counts is not None and counts["InMemoryPlatformMetricsRepository"] == 8
    repo = service.unit.repos.platform_metrics
    assert repo.list(SHOP_A) == [] and len(repo.list(SHOP_B)) == 7
    with pytest.raises(ShopInactive):  # 删完的店铺不会被写回来
        metrics.sync(SHOP_A, actor="system", days=7)
    with pytest.raises(ShopInactive):
        metrics.record(SHOP_A, metric=TB, day=today_of(service), value=0.9, merchant=MERCHANT)
    assert repo.list(SHOP_A) == []


# ================================================================ 商家登记


def test_record_writes_a_merchant_row_with_the_merchant_and_audits():
    service, metrics = make()
    day = today_of(service) - timedelta(days=40)
    result = metrics.record(
        SHOP_A,
        metric="taobao_3min_response_rate",
        day=day,
        value=0.81,
        sample_size=230,
        merchant=MERCHANT,
    )
    assert result.changed and not result.replaced
    assert result.sample.source == MetricSource.MERCHANT and result.sample.recorded_by == MERCHANT
    assert service.unit.repos.platform_metrics.list(SHOP_A) == [result.sample]
    [event] = events(service, SHOP_A, AuditAction.PLATFORM_METRICS_RECORDED.value)
    assert event.actor == MERCHANT
    assert event.detail == {
        "metric": "taobao_3min_response_rate",
        "day": day.isoformat(),
        "value": 0.81,
        "sample_size": 230,
        "replaced": False,
        "previous_value": None,
    }


def test_record_again_replaces_and_unchanged_is_a_no_op():
    service, metrics = make()
    day = today_of(service)
    metrics.record(SHOP_A, metric=TB, day=day, value=0.8, merchant=MERCHANT)
    again = metrics.record(SHOP_A, metric=TB, day=day, value=0.82, merchant="owner_a2")
    assert again.changed and again.replaced and again.sample.recorded_by == "owner_a2"
    same = metrics.record(SHOP_A, metric=TB, day=day, value=0.82, merchant=MERCHANT)
    assert not same.changed and not same.replaced and same.sample.recorded_by == "owner_a2"
    recorded = events(service, SHOP_A, AuditAction.PLATFORM_METRICS_RECORDED.value)
    assert [e.detail["previous_value"] for e in recorded] == [None, 0.8]  # 没变的不记
    # 平台同步的那一条另存，登记不覆盖它
    metrics.sync(SHOP_A, actor="system", days=1)
    rows = service.unit.repos.platform_metrics.list(SHOP_A, since=day - timedelta(days=1))
    assert {r.source for r in rows} == {MetricSource.PLATFORM, MetricSource.MERCHANT}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metric": DY_HELP}, "不是本平台考核的指标"),
        ({"metric": "nope"}, "未知的平台指标"),
        ({"value": 91.0}, "比例"),
        ({"value": -0.1}, "0–1"),
        ({"value": float("inf")}, "有限数"),
        ({"day_offset": -1}, "不能晚于今天"),
        ({"day_offset": 401}, "最多回溯 400 天"),
        ({"sample_size": -1}, "样本量"),
        ({"sample_size": 10_000_001}, "样本量"),
        ({"merchant": "system"}, "具体的人"),
        ({"merchant": " "}, "不能为空"),
    ],
)
def test_record_validation_writes_nothing(kwargs, message):
    service, metrics = make()
    offset = kwargs.pop("day_offset", 1)
    args = {
        "metric": TB,
        "day": today_of(service) - timedelta(days=offset),
        "value": 0.9,
        "merchant": MERCHANT,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        metrics.record(SHOP_A, **args)
    assert service.unit.repos.platform_metrics.list(SHOP_A) == []
    assert events(service, SHOP_A, AuditAction.PLATFORM_METRICS_RECORDED.value) == []


def test_record_oldest_allowed_day_is_400_days_back():
    service, metrics = make()
    day = today_of(service) - timedelta(days=400)
    assert metrics.record(SHOP_A, metric=TB, day=day, value=0.7, merchant=MERCHANT).changed


def test_record_on_a_suspended_shop_is_refused():
    service, metrics = make()
    shop = service.unit.repos.knowledge.get_shop(SHOP_A)
    service.unit.repos.knowledge.upsert_shop(
        shop.model_copy(update={"status": ShopStatus.SUSPENDED})
    )
    with pytest.raises(ShopInactive):
        metrics.record(SHOP_A, metric=TB, day=today_of(service), value=0.9, merchant=MERCHANT)
    assert service.unit.repos.platform_metrics.list(SHOP_A) == []


# ================================================================ 列表


def test_list_samples_default_range_filters_and_validation():
    service, metrics = make()
    today = today_of(service)
    metrics.sync(SHOP_A, actor="system", days=80)
    listed = metrics.list_samples(SHOP_A)
    assert (listed.since, listed.until) == (today - timedelta(days=55), today)
    assert len(listed.samples) == 55  # 平台数据到昨天
    narrow = metrics.list_samples(
        SHOP_A, since=today - timedelta(days=3), metric="taobao_3min_response_rate"
    )
    assert [s.day for s in narrow.samples] == [today - timedelta(days=d) for d in (3, 2, 1)]
    with pytest.raises(ValueError, match="不能晚于"):
        metrics.list_samples(SHOP_A, since=today, until=today - timedelta(days=1))
    with pytest.raises(ValueError, match="400"):
        metrics.list_samples(SHOP_A, since=today - timedelta(days=400), until=today)
    with pytest.raises(ValueError, match="不是本平台"):
        metrics.list_samples(SHOP_A, metric=DY_HELP)
    assert metrics.list_samples(SHOP_B).samples == []


# ================================================================ 接入日期与报表


def test_report_without_go_live_asks_for_one():
    _, metrics = make()
    report = metrics.report(SHOP_A)
    assert report.verdict == "no_go_live" and report.go_live_source == "unknown"
    assert report.metrics == []


def test_report_after_sync_with_explicit_go_live_shows_improvement():
    service, metrics = make()
    metrics.sync(SHOP_A, actor="system")
    go_live = today_of(service) - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)
    report = metrics.report(SHOP_A, go_live=go_live)
    assert report.go_live == go_live and report.go_live_source == "query"
    assert report.verdict == "improved" and report.reliable
    [c] = report.metrics
    assert c.before.mean == pytest.approx(0.78, rel=0.02)
    assert c.after.mean == pytest.approx(0.91, rel=0.02)
    # 窗口改成 14 天
    short = metrics.report(SHOP_A, go_live=go_live, window_days=14)
    assert short.window_days == 14 and short.metrics[0].before.days == 14


def test_report_go_live_from_first_switch_to_assist():
    service, metrics = make()
    metrics.sync(SHOP_A, actor="system")
    clock = service.unit.clock
    now = clock.now()
    clock.set(now - timedelta(days=40))
    service.rollout.set_stage(SHOP_A, "shadow", operator=MERCHANT)
    clock.set(now - timedelta(days=DEMO_GO_LIVE_DAYS_AGO))
    service.rollout.set_stage(
        SHOP_A, "assist", operator="qa_reviewer_01", force=True, note="合成测试：强制切换"
    )
    clock.set(now - timedelta(days=10))
    service.rollout.set_stage(SHOP_A, "shadow", operator=MERCHANT)
    clock.set(now - timedelta(days=5))
    service.rollout.set_stage(
        SHOP_A, "assist", operator="qa_reviewer_01", force=True, note="合成测试：再切回来"
    )
    clock.set(now)
    report = metrics.report(SHOP_A)
    assert report.go_live_source == "rollout"
    assert report.go_live == (now - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)).astimezone(CST).date()
    assert report.verdict == "improved"


def test_report_go_live_from_authorization_granted_at():
    service, metrics = make()
    granted = service.unit.clock.now() - timedelta(days=20)
    auth = service.unit.repos.auth.get(SHOP_A)
    service.unit.repos.auth.save(auth.model_copy(update={"granted_at": granted}))
    report = metrics.report(SHOP_A)
    assert report.go_live_source == "authorization"
    assert report.go_live == granted.astimezone(CST).date()


def _subscribe(service: AssistService, kind: str, at: datetime):
    """平台推送的开通 / 续订（合成令牌）；at 是平台的事件时间。"""
    event = parse_subscription(subscription_push(kind, SHOP_A, at=at))
    assert event is not None
    return service.tenancy.handle_subscription(event)


def test_report_go_live_from_authorization_ignores_later_renewals():
    """续订会改写 ShopAuth.granted_at；接入日期按这次开通的第一条 auth.granted（平台事件时间）。"""
    service, metrics = make()
    clock = service.unit.clock
    now = clock.now()
    subscribed_at = now - timedelta(days=31)  # 平台的开通事件时间
    clock.set(now - timedelta(days=30))  # 晚一天才送到
    assert _subscribe(service, "subscribed", subscribed_at).action == "granted"
    clock.set(now - timedelta(days=2))
    assert _subscribe(service, "renewed", clock.now()).action == "granted"
    clock.set(now)
    assert service.unit.repos.auth.stamp(SHOP_A).granted_at == now - timedelta(days=2)
    report = metrics.report(SHOP_A)
    assert report.go_live_source == "authorization"
    assert report.go_live == subscribed_at.astimezone(CST).date()
    assert "续订" in report.go_live_label


def test_report_go_live_after_reonboarding_without_granted_at():
    """取消订购删掉了上一次开通的审计；运维重新开通（不知道平台开通时间，granted_at 为空）后
    按这次开通的 auth.granted，之后续订也不变。"""
    service, metrics = make()
    clock = service.unit.clock
    now = clock.now()
    clock.set(now - timedelta(days=60))
    _subscribe(service, "subscribed", clock.now())
    clock.set(now - timedelta(days=40))
    service.tenancy.revoke(SHOP_A, reason="ops", actor="ops")
    clock.set(now - timedelta(days=20))
    reonboard = parse_subscription(subscription_push("subscribed", SHOP_A, at=clock.now()))
    service.tenancy.onboard(reonboard, actor="ops")  # 运维开通：不给 granted_at
    assert service.unit.repos.auth.stamp(SHOP_A).granted_at is None
    clock.set(now - timedelta(days=5))
    _subscribe(service, "renewed", clock.now())
    clock.set(now)
    report = metrics.report(SHOP_A)
    assert report.go_live_source == "authorization"
    assert report.go_live == (now - timedelta(days=20)).astimezone(CST).date()


def test_report_explicit_go_live_is_validated():
    service, metrics = make()
    today = today_of(service)
    with pytest.raises(ValueError, match="不能晚于今天"):
        metrics.report(SHOP_A, go_live=today + timedelta(days=1))
    with pytest.raises(ValueError, match="400"):
        metrics.report(SHOP_A, go_live=today - timedelta(days=401))
    with pytest.raises(ValueError):
        metrics.report(SHOP_A, go_live=today, window_days=3)
    assert metrics.report(SHOP_A, go_live=today).after_end == today


def test_reports_are_per_shop():
    service, metrics = make()
    metrics.sync(SHOP_B, actor="system")
    go_live = today_of(service) - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)
    assert metrics.report(SHOP_A, go_live=go_live).verdict == "insufficient"
    report_b = metrics.report(SHOP_B, go_live=go_live)
    assert report_b.shop_id == SHOP_B and report_b.verdict == "improved"  # 0.82 → 0.84


def test_pdd_unit_reports_its_own_metric():
    service, metrics = make(Platform.PDD)
    metrics.sync(SHOP_C, actor="system")
    go_live = today_of(service) - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)
    report = metrics.report(SHOP_C, go_live=go_live)
    assert [c.metric for c in report.metrics] == [PlatformMetric.PDD_3MIN_REPLY_RATE]
    assert report.verdict == "improved"


def test_audit_actions_live_in_core_and_old_names_are_aliases():
    """两个事件名已并入 core.audit.AuditAction，取值不变（存量审计照读）；
    模块里的旧名字只是别名。"""
    from rodit.app import platform_metrics as module

    assert module.PLATFORM_METRICS_SYNCED == AuditAction.PLATFORM_METRICS_SYNCED.value
    assert module.PLATFORM_METRICS_RECORDED == AuditAction.PLATFORM_METRICS_RECORDED.value
    assert (module.PLATFORM_METRICS_SYNCED, module.PLATFORM_METRICS_RECORDED) == (
        "platform_metrics.synced",
        "platform_metrics.recorded",
    )
