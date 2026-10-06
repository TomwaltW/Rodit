"""客户演示站的测试夹具。公共代码在 showcase_support.py（importlib 模式下测试模块不能 import
conftest，所以放进单独的模块，并把本目录加进 sys.path）。"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from llm_stub import StubLLM  # noqa: E402
from showcase_support import Showcase, build_showcase  # noqa: E402


@pytest.fixture
def stub() -> Iterator[StubLLM]:
    """本机随机端口上的 OpenAI 兼容模型替身（llm_stub.py），测试结束关掉。"""
    with StubLLM() as server:
        yield server


@pytest.fixture
def stub_env(stub: StubLLM, monkeypatch: pytest.MonkeyPatch) -> StubLLM:
    """RODIT_LLM_* 指向模型替身（真实的 OpenAICompatibleClient 走真实的 HTTP 调它）。"""
    monkeypatch.setenv("RODIT_LLM_BASE_URL", stub.base_url)
    monkeypatch.setenv("RODIT_LLM_MODEL", stub.model)
    monkeypatch.setenv("RODIT_LLM_API_KEY", stub.api_key)
    monkeypatch.delenv("RODIT_LLM_TIMEOUT_S", raising=False)
    return stub


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
