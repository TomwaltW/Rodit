"""商家控制台接口（方案 §9.1 "知识编辑、配置、报表"、§5.2 知识中心、§5.2.3 商品同步、
§9.3 店铺配置）。

鉴权一律用 rodit.api.deps 的 *Dep（§9.2 串店检查）；业务逻辑在 rodit.app.console。
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
