"""平台客服指标接口（方案 §12.4 "平台客服指标接入前后对比"、§15 Phase 2 商家报表）。

待实现（Track B）：商家登记平台指标、拉取平台数据、接入前后对比报表。
鉴权用 rodit.api.deps 的依赖；业务逻辑在 rodit.app / rodit.feedback。
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
