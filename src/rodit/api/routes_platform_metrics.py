"""平台客服指标接口（方案 §12.4 "平台客服指标接入前后对比"、§15 Phase 2 商家报表、§1.3 目标
"平台客服指标较接入前改善"）。

鉴权一律用 rodit.api.deps 的 *Dep 或 shop_guard（§9.2 串店检查，停用的店铺 403 shop_inactive）；
业务逻辑在 rodit.app.platform_metrics（口径在 rodit.feedback.platform_metrics）。

| 接口 | 角色 |
|---|---|
| GET reports/platform-metrics（接入前后对比报表，?go_live=&window_days=） | 店长、商家、复核 |
| GET platform-metrics（原始数据，?since=&until=&metric=） | 店长、商家、复核 |
| POST platform-metrics（商家登记一天的一项指标） | 商家 |
| POST platform-metrics/sync（从平台接口同步最近 days 天，?days=） | 店长、商家 |

登记人、操作人一律取自调用方身份，不取请求体。错误码：本单元没有指标存储 503
platform_metrics_unavailable；没接平台指标接口 503 metrics_source_not_configured（可以手工
登记）；平台指标接口故障 503 metrics_source_unavailable（这次什么都没有写）；授权无效 403
shop_not_authorized；接口返回了别的店铺的数据 403 tenant_isolation（什么都没有写）。
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from rodit.api.auth import CONSOLE_ROLES, REVIEWER_ROLES, Principal
from rodit.api.deps import ConsoleDep, EmptyBody, MerchantDep, ServiceDep, shop_guard
from rodit.api.errors import ApiError
from rodit.api.schemas_platform_metrics import MetricRecordRequest
from rodit.app.platform_metrics import (
    DEFAULT_SYNC_DAYS,
    MetricsSourceMissing,
    PlatformMetricsService,
    PlatformMetricsUnavailable,
    RecordResult,
    SampleList,
    SyncResult,
)
from rodit.app.service import AssistService
from rodit.core.enums import PlatformMetric
from rodit.core.errors import ToolUnavailable
from rodit.feedback.platform_metrics import (
    DEFAULT_WINDOW_DAYS,
    MAX_HISTORY_DAYS,
    MAX_WINDOW_DAYS,
    MIN_WINDOW_DAYS,
    PlatformMetricsReport,
)

ERROR_UNAVAILABLE = "platform_metrics_unavailable"
ERROR_SOURCE_MISSING = "metrics_source_not_configured"
ERROR_SOURCE_DOWN = "metrics_source_unavailable"
MSG_SOURCE_DOWN = "平台指标接口暂时不可用，请稍后重试；这次什么都没有写"

TAG = "平台客服指标"

MetricsReadDep = Annotated[Principal, Depends(shop_guard(CONSOLE_ROLES | REVIEWER_ROLES))]
"""看报表和原始数据（只有指标、日期、数值）：店长、商家、复核人员。"""

_503 = {
    503: {
        "description": (
            "没有指标存储（platform_metrics_unavailable）、没接平台指标接口"
            "（metrics_source_not_configured）或平台指标接口故障（metrics_source_unavailable）"
        )
    }
}

router = APIRouter()


def _metrics(service: AssistService) -> PlatformMetricsService:
    return PlatformMetricsService(service.unit, service.processing)


def _unavailable(exc: Exception) -> ApiError:
    if isinstance(exc, MetricsSourceMissing):
        return ApiError(503, ERROR_SOURCE_MISSING, str(exc))
    if isinstance(exc, ToolUnavailable):
        return ApiError(503, ERROR_SOURCE_DOWN, MSG_SOURCE_DOWN)
    return ApiError(503, ERROR_UNAVAILABLE, str(exc))


@router.get(
    "/v1/shops/{shop_id}/reports/platform-metrics",
    response_model=PlatformMetricsReport,
    responses=_503,
    tags=[TAG],
)
def platform_metrics_report(
    principal: MetricsReadDep,
    service: ServiceDep,
    go_live: Annotated[
        date | None,
        Query(description="接入日期（YYYY-MM-DD，算接入后）；不填时用第一次切到智能辅助的日期"),
    ] = None,
    window_days: Annotated[
        int, Query(ge=MIN_WINDOW_DAYS, le=MAX_WINDOW_DAYS, description="接入前、接入后各看几天")
    ] = DEFAULT_WINDOW_DAYS,
) -> PlatformMetricsReport:
    """平台客服指标接入前后对比（§12.4）：每项指标接入前 / 接入后的均值、变化、变好还是变差、
    可靠性提示、按周的序列，以及总体结论（§1.3 "较接入前改善"）。接入日期：填写的 > 第一次切到
    智能辅助 > 店铺开通授权的日期；都没有时结论是 no_go_live，请填写。只读。"""
    try:
        return _metrics(service).report(principal.shop_id, go_live=go_live, window_days=window_days)
    except PlatformMetricsUnavailable as exc:
        raise _unavailable(exc) from None


@router.get(
    "/v1/shops/{shop_id}/platform-metrics",
    response_model=SampleList,
    responses=_503,
    tags=[TAG],
)
def platform_metrics_list(
    principal: MetricsReadDep,
    service: ServiceDep,
    since: Annotated[date | None, Query(description="起始日期（含）；默认结束日期前 55 天")] = None,
    until: Annotated[date | None, Query(description="结束日期（含）；默认今天")] = None,
    metric: Annotated[PlatformMetric | None, Query(description="只看这项指标")] = None,
) -> SampleList:
    """原始数据：平台同步、商家登记、系统估算各一条，不做取舍（报表按平台 > 商家 > 估算取）。
    一次最多 400 天。只读。"""
    try:
        return _metrics(service).list_samples(
            principal.shop_id, since=since, until=until, metric=metric
        )
    except PlatformMetricsUnavailable as exc:
        raise _unavailable(exc) from None


@router.post(
    "/v1/shops/{shop_id}/platform-metrics",
    response_model=RecordResult,
    responses=_503,
    tags=[TAG],
)
def platform_metrics_record(
    body: MetricRecordRequest, principal: MerchantDep, service: ServiceDep
) -> RecordResult:
    """商家登记一天的一项平台指标（从平台后台抄录，source=merchant）：指标必须是本平台考核的，
    比例 0–1，日期不晚于今天、最多回溯 400 天。和已登记的一样时不写（changed=false）。
    登记人取自调用方身份，记审计 platform_metrics.recorded。"""
    try:
        return _metrics(service).record(
            principal.shop_id,
            metric=body.metric,
            day=body.day,
            value=body.value,
            sample_size=body.sample_size,
            merchant=principal.user_id,
        )
    except PlatformMetricsUnavailable as exc:
        raise _unavailable(exc) from None


@router.post(
    "/v1/shops/{shop_id}/platform-metrics/sync",
    response_model=SyncResult,
    responses=_503,
    tags=[TAG],
)
def platform_metrics_sync(
    principal: ConsoleDep,
    service: ServiceDep,
    days: Annotated[
        int, Query(ge=1, le=MAX_HISTORY_DAYS, description="同步最近几天（到昨天为止）")
    ] = DEFAULT_SYNC_DAYS,
    body: EmptyBody = None,
) -> SyncResult:
    """从平台接口同步最近 days 天的客服考核数据（source=platform；平台云内的只读接口）。
    先查店铺授权；整段拉完、逐条核对后才写，接口故障时什么都不写（503 metrics_source_unavailable）。
    重复同步结果一样。操作人取自调用方身份，记审计 platform_metrics.synced。"""
    try:
        return _metrics(service).sync(principal.shop_id, actor=principal.user_id, days=days)
    except (PlatformMetricsUnavailable, MetricsSourceMissing, ToolUnavailable) as exc:
        raise _unavailable(exc) from None


__all__ = [
    "ERROR_SOURCE_DOWN",
    "ERROR_SOURCE_MISSING",
    "ERROR_UNAVAILABLE",
    "MetricsReadDep",
    "router",
]
