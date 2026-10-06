"""客户演示站的测试夹具。公共代码在 showcase_support.py（importlib 模式下测试模块不能 import
conftest，所以放进单独的模块，并把本目录加进 sys.path）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from showcase_support import Showcase, build_showcase  # noqa: E402


@pytest.fixture
def showcase() -> Showcase:
    """每个测试一份新的演示站，不回放历史（快）。"""
    return build_showcase()


@pytest.fixture(scope="module")
def seeded() -> Showcase:
    """回放了近 5 周历史的演示站（同一个模块共用；只用于不改数据的检查）。"""
    return build_showcase(history=True)


@pytest.fixture
def fresh_seeded() -> Showcase:
    """回放了历史、每个测试一份（会改数据的检查用）。"""
    return build_showcase(history=True)
