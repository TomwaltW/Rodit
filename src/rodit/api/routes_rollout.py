"""上线节奏与持续评估接口（方案 §13.4 影子对比与阶段门槛、§13.5 每周人工复核）。

鉴权一律用 rodit.api.deps 的 *Dep（§9.2 串店检查）；业务逻辑在 rodit.app.rollout。
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
