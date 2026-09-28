"""问答引擎测试夹具。替身和组装代码在 engine_support.py（importlib 模式下测试模块不能 import
conftest，所以放进单独的模块，并把本目录加进 sys.path）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from engine_support import Kit, build_kit  # noqa: E402


@pytest.fixture
def kit() -> Kit:
    return build_kit()
