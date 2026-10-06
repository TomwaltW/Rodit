"""Phase 2 / 3 新契约（T0）：平台客服指标、图片理解的端口、内存实现、替身和平台绑定。"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest
from ops_support import cloud_bindings

from rodit.app.demo import build_demo_unit
from rodit.app.service import AssistService
from rodit.core.clock import CST
from rodit.core.config import EngineConfig
from rodit.core.enums import ImageKind, MetricSource, Platform, PlatformMetric
from rodit.core.errors import ToolUnavailable
from rodit.core.models import ImageRef, PlatformMetricSample
from rodit.core.pii import BuyerRefHasher
from rodit.core.platform_metrics import PLATFORM_METRIC_SPECS, metric_labels, metrics_for
from rodit.core.ports import (
    ImageUnderstanding,
    PlatformMetricsRepository,
    PlatformMetricsSource,
)
from rodit.inmemory import InMemoryPlatformMetricsRepository
from rodit.ops.bindings import BindingContext, BindingError, validate_bindings
from rodit.testing.fakes import FakeClock, FakeImageUnderstanding, InMemoryPlatformMetricsSource
from rodit.testing.fixtures import (
    DEMO_GO_LIVE_DAYS_AGO,
    DEMO_IMAGE_FINDINGS,
    SHOP_A,
    SHOP_B,
    SHOP_D,
    TEST_SALT,
    build_demo_world,
    demo_platform_metrics,
)


def _sample(shop_id: str, metric: PlatformMetric, days_ago: int, value: float, source=None):
    world = build_demo_world()
    return PlatformMetricSample(
        shop_id=shop_id,
        metric=metric,
        day=world.now.astimezone(CST).date() - timedelta(days=days_ago),
        value=value,
        source=source or MetricSource.MERCHANT,
        recorded_by="owner_demo",
        recorded_at=world.now,
    )


def test_every_platform_has_metrics_and_labels():
    for platform in Platform:
        assert metrics_for(platform)
    labels = metric_labels()
    assert set(labels) == {m.value for m in PlatformMetric}
    assert labels["douyin_avg_response_s"]["higher_is_better"] is False
    assert PLATFORM_METRIC_SPECS[PlatformMetric.PDD_3MIN_REPLY_RATE].unit == "ratio"


def test_in_memory_metrics_repository_upserts_filters_and_isolates_shops():
    repo = InMemoryPlatformMetricsRepository()
    assert isinstance(repo, PlatformMetricsRepository)
    m = PlatformMetric.TAOBAO_3MIN_RESPONSE_RATE
    repo.upsert(_sample(SHOP_A, m, 3, 0.8))
    repo.upsert(_sample(SHOP_A, m, 3, 0.85))  # 同一天同一来源：覆盖
    repo.upsert(_sample(SHOP_A, m, 3, 0.9, MetricSource.PLATFORM))  # 不同来源并存
    repo.upsert(_sample(SHOP_A, m, 10, 0.7))
    repo.upsert(_sample(SHOP_B, m, 3, 0.5))
    rows = repo.list(SHOP_A)
    assert [(r.value, r.source) for r in rows] == [
        (0.7, MetricSource.MERCHANT),
        (0.85, MetricSource.MERCHANT),
        (0.9, MetricSource.PLATFORM),
    ]
    since = rows[1].day
    assert len(repo.list(SHOP_A, since=since)) == 2
    assert len(repo.list(SHOP_A, until=since - timedelta(days=1))) == 1
    assert all(r.shop_id == SHOP_A for r in repo.list(SHOP_A))
    assert repo.purge_shop(SHOP_A) == 3
    assert repo.list(SHOP_A) == [] and len(repo.list(SHOP_B)) == 1


def test_demo_metrics_source_returns_only_its_platform_and_shop():
    world = build_demo_world()
    source = InMemoryPlatformMetricsSource(demo_platform_metrics(world), Platform.DOUYIN)
    assert isinstance(source, PlatformMetricsSource)
    today = world.now.astimezone(CST).date()
    rows = source.fetch(SHOP_D, today - timedelta(days=7), today)
    assert rows and {r.shop_id for r in rows} == {SHOP_D}
    assert {r.metric for r in rows} <= set(metrics_for(Platform.DOUYIN))
    assert source.fetch(SHOP_A, today - timedelta(days=60), today) == []  # 淘宝店不在抖店单元
    before = [
        r
        for r in source.fetch(SHOP_D, today - timedelta(days=56), today)
        if r.metric == PlatformMetric.DOUYIN_AVG_RESPONSE_S
    ]
    old = [r.value for r in before if (today - r.day).days > DEMO_GO_LIVE_DAYS_AGO]
    new = [r.value for r in before if (today - r.day).days <= DEMO_GO_LIVE_DAYS_AGO]
    assert sum(new) / len(new) < sum(old) / len(old)  # 接入后响应变快
    source.fail = True
    with pytest.raises(ToolUnavailable):
        source.fetch(SHOP_D, today, today)


def test_fake_image_understanding_masks_text_and_fails_unknown_images():
    vision = FakeImageUnderstanding(DEMO_IMAGE_FINDINGS)
    assert isinstance(vision, ImageUnderstanding)
    shot = vision.analyze(SHOP_A, ImageRef(ref="img_buyer_logistics_1"))
    assert shot.kind == ImageKind.LOGISTICS_SCREENSHOT
    assert "13812345678" not in (shot.extracted_text or "")
    size = vision.analyze(SHOP_A, ImageRef(ref="img_buyer_size_1"))
    assert size.kind == ImageKind.SIZE_REFERENCE and "165" in (size.extracted_text or "")
    with pytest.raises(ToolUnavailable):
        vision.analyze(SHOP_A, ImageRef(ref="img_unreadable"))
    only_b = FakeImageUnderstanding(DEMO_IMAGE_FINDINGS, shop_ids=[SHOP_B])
    with pytest.raises(ToolUnavailable):
        only_b.analyze(SHOP_A, ImageRef(ref="img_buyer_damage_1"))


def test_demo_unit_has_vision_and_metrics_and_revoke_purges_metrics():
    unit = build_demo_unit(Platform.TAOBAO)
    assert unit.vision is not None and unit.metrics_source is not None
    assert unit.metrics_source.platform == Platform.TAOBAO
    repo = unit.repos.platform_metrics
    assert repo is not None
    repo.upsert(_sample(SHOP_A, PlatformMetric.TAOBAO_3MIN_RESPONSE_RATE, 1, 0.9))
    counts = AssistService(unit).revoke_shop(SHOP_A)
    assert counts is not None and repo.list(SHOP_A) == []


def test_aftersale_evidence_images_round_trip_in_world():
    a = build_demo_world().aftersale(SHOP_A, "as_2009")
    assert a.evidence_count == 3 and len(a.evidence_images) == 3


def test_bindings_check_metrics_source_platform_and_reject_mocks_in_production():
    world = build_demo_world()
    ctx = BindingContext(
        platform=Platform.TAOBAO,
        config=EngineConfig(),
        clock=FakeClock(),
        hasher=BuyerRefHasher(TEST_SALT),
    )
    wrong = replace(
        cloud_bindings(ctx),
        metrics_source=InMemoryPlatformMetricsSource(demo_platform_metrics(world), Platform.PDD),
    )
    with pytest.raises(BindingError, match="平台指标接口不属于本单元"):
        validate_bindings(wrong, Platform.TAOBAO, allow_mock=True)
    for field, value in (
        ("vision", FakeImageUnderstanding(DEMO_IMAGE_FINDINGS)),
        (
            "metrics_source",
            InMemoryPlatformMetricsSource(demo_platform_metrics(world), Platform.TAOBAO),
        ),
    ):
        bindings = replace(cloud_bindings(ctx), **{field: value})
        assert validate_bindings(bindings, Platform.TAOBAO, allow_mock=True) is bindings
        with pytest.raises(BindingError, match="模拟实现"):
            validate_bindings(bindings, Platform.TAOBAO, allow_mock=False)
    with pytest.raises(BindingError, match="vision"):
        validate_bindings(
            replace(cloud_bindings(ctx), vision=object()), Platform.TAOBAO, allow_mock=True
        )
