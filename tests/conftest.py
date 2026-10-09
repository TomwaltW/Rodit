"""全部测试共用的环境隔离。

测试不读开发机 shell 里的部署变量，也不碰真实的大模型接口（CLAUDE.md：单元测试不依赖网络）。
准备真实大模型评测的 shell 正好导出了这些变量（README：source .env、RODIT_CONFIG）：不清掉的话，
--llm env 的命令行测试会按那份配置跑（配置指纹对不上、条数对不上），演示脚本的测试会把合成会话
真的发到模型服务。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

DEPLOYMENT_ENV = (
    "RODIT_CONFIG",
    "RODIT_LLM_BASE_URL",
    "RODIT_LLM_MODEL",
    "RODIT_LLM_API_KEY",
    "RODIT_LLM_TIMEOUT_S",
    "RODIT_LLM_PRICE_INPUT_PER_1K",
    "RODIT_LLM_PRICE_OUTPUT_PER_1K",
    "RODIT_JEV_BASE_URL",
    "RODIT_JEV_API_KEY",
    "RODIT_CLASSIFIER_PATH",
)
"""整个测试会话里清掉的环境变量。要用的测试自己 monkeypatch.setenv。
RODIT_TEST_PG_DSN（PostgreSQL 测试的开关）和代理变量不动。"""


@pytest.fixture(scope="session", autouse=True)
def _no_deployment_env() -> Iterator[None]:
    """会话级：模块级的 fixture（如 test_fingerprint_setup 的 classifier_eval）先于任何函数级的
    autouse fixture 执行，逐个文件加函数级的 delenv 盖不住它们。会话结束时恢复原值。"""
    with pytest.MonkeyPatch.context() as patch:
        for name in DEPLOYMENT_ENV:
            patch.delenv(name, raising=False)
        yield
