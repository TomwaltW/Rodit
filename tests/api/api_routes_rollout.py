"""上线节奏与每周人工复核（§13.4、§13.5）的店铺范围路由清单（§9.2 串店测试覆盖）。

每加一个店铺范围的路由就在 ROUTES 里加一条：body 用能通过校验的合成请求体，roles 是允许的角色。
"""

from __future__ import annotations

from api_routes import CONSOLE, HUMAN, MERCHANT, REVIEWER, Route  # noqa: F401

ROLLOUT_READ = CONSOLE | REVIEWER  # 看上线状态、影子对比、复核报表（没有文字）
STAGE = MERCHANT | REVIEWER  # 切回影子：商家、复核人员；切到智能辅助只有复核人员（路由内再判）

ROUTES: tuple[Route, ...] = (
    Route("GET", "/v1/shops/{shop}/rollout/status", roles=ROLLOUT_READ),
    Route("GET", "/v1/shops/{shop}/rollout/shadow", roles=ROLLOUT_READ),
    Route("POST", "/v1/shops/{shop}/rollout/stage", {"stage": "shadow"}, STAGE),
    Route(
        "POST",
        "/v1/shops/{shop}/rollout/incidents",
        {"kind": "violation", "week": "2026-W39", "count": 0},
        REVIEWER,
    ),
    Route("GET", "/v1/shops/{shop}/rollout/incidents", roles=ROLLOUT_READ),
    Route("POST", "/v1/shops/{shop}/quality/sample", {"size": 5}, REVIEWER),
    Route("GET", "/v1/shops/{shop}/quality/reviews", roles=REVIEWER),
    Route("POST", "/v1/shops/{shop}/quality/reviews/1", {"status": "skipped"}, REVIEWER),
    Route("GET", "/v1/shops/{shop}/quality/report", roles=ROLLOUT_READ),
)
