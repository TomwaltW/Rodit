"""运维接口：/healthz（存活）、/readyz（就绪）、/metrics（Prometheus），方案 §12.2、§12.4。

install_ops 把它们挂到 rodit.api.create_app 造好的应用上（include_router），并装上 HTTP 指标
中间件；接口层代码不用改。接口层自带一个 GET /healthz：先注册的先匹配，两个返回一样的内容；
这里的 /healthz 不进 OpenAPI 文档、用单独的 operation_id，接口层哪天去掉自己的也照样有存活检查。

- /healthz：不查依赖，进程活着就是 200；
- /readyz：见 rodit.ops.health，200 就绪 / 503 未就绪，只有检查名、是否通过和原因代码；
- /metrics：Prometheus 文本格式。指标里有店铺 ID，所以默认要令牌（令牌文件的 metrics 段，
  ``Authorization: Bearer <令牌>``）；只有部署设置明确允许（RODIT_METRICS_PUBLIC=1）时才开放。
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from rodit.api.auth import bearer_token
from rodit.api.errors import unauthorized
from rodit.ops.health import ReadinessChecker
from rodit.ops.instrument import HttpMetricsMiddleware, RoditMetrics
from rodit.ops.metrics import CONTENT_TYPE

MetricsGuard = Callable[[str | None], bool]
"""抓取令牌（可能为 None）→ 是否放行。"""

_NO_STORE = {"Cache-Control": "no-store"}


def build_ops_router(
    *,
    readiness: ReadinessChecker,
    metrics: RoditMetrics,
    metrics_guard: MetricsGuard | None,
) -> APIRouter:
    """metrics_guard 为 None 表示 /metrics 不要令牌（只在部署设置明确允许时这样用）。"""
    router = APIRouter(tags=["运维"])

    @router.get("/healthz", include_in_schema=False, operation_id="ops_healthz")
    def healthz() -> dict[str, str]:
        """存活检查（不查依赖，不需要鉴权）。"""
        return {"status": "ok"}

    @router.get("/readyz", responses={503: {"description": "未就绪"}}, operation_id="ops_readyz")
    def readyz() -> JSONResponse:
        """就绪检查：数据库、迁移版本、大模型配置（不需要鉴权，不返回店铺数据）。"""
        result = readiness.check()
        return JSONResponse(
            result.as_dict(), status_code=200 if result.ready else 503, headers=_NO_STORE
        )

    @router.get("/metrics", response_class=Response, operation_id="ops_metrics")
    def metrics_endpoint(request: Request) -> Response:
        """Prometheus 指标（§12.4 质量与技术指标）。"""
        if metrics_guard is not None:
            token = bearer_token(request.headers.get("authorization"))
            if not metrics_guard(token):
                raise unauthorized()
        return Response(metrics.render(), media_type=CONTENT_TYPE, headers=_NO_STORE)

    return router


def install_ops(
    app: FastAPI,
    *,
    readiness: ReadinessChecker,
    metrics: RoditMetrics,
    metrics_guard: MetricsGuard | None,
) -> FastAPI:
    """把运维接口和 HTTP 指标中间件装到应用上（应用启动前调用）。"""
    router = build_ops_router(readiness=readiness, metrics=metrics, metrics_guard=metrics_guard)
    app.include_router(router)
    app.add_middleware(HttpMetricsMiddleware, metrics=metrics)
    return app


__all__ = ["MetricsGuard", "build_ops_router", "install_ops"]
