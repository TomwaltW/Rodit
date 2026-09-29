"""HTTP 接口测试夹具。公共代码在 api_support.py（importlib 模式下测试模块不能 import conftest，
所以放进单独的模块，并把本目录加进 sys.path）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from api_support import Api, build_api  # noqa: E402

from rodit.core.enums import Platform  # noqa: E402


@pytest.fixture
def api() -> Api:
    """淘宝单元（店铺 A、B），模式 B 插件 + 模式 C 网页工具。"""
    return build_api(Platform.TAOBAO)


@pytest.fixture
def douyin_api() -> Api:
    """抖店单元（店铺 D 开了影子模式，§13.4）。"""
    return build_api(Platform.DOUYIN)
