"""平台客服指标（§12.4 接入前后对比、§15 Phase 2 商家报表）的店铺范围路由清单（§9.2 串店测试覆盖）。

每加一个店铺范围的路由就在 ROUTES 里加一条：body 用能通过校验的合成请求体，roles 是允许的角色。
"""

from __future__ import annotations

from api_routes import CONSOLE, MERCHANT, REVIEWER, Route

METRICS_READ = CONSOLE | REVIEWER  # 报表、原始数据（只有指标、日期、数值）：店长、商家、复核人员

RECORD_BODY = {"metric": "taobao_3min_response_rate", "day": "2026-09-20", "value": 0.9}

ROUTES: tuple[Route, ...] = (
    Route("GET", "/v1/shops/{shop}/reports/platform-metrics", roles=METRICS_READ),
    Route("GET", "/v1/shops/{shop}/platform-metrics", roles=METRICS_READ),
    Route("POST", "/v1/shops/{shop}/platform-metrics", RECORD_BODY, MERCHANT),
    Route("POST", "/v1/shops/{shop}/platform-metrics/sync", roles=CONSOLE),
)
