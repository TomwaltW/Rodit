"""用真实大模型跑评测时的环境配置（`python -m rodit.eval --llm env`）。

大模型：rodit.llm.OpenAICompatibleClient.from_env——RODIT_LLM_BASE_URL、RODIT_LLM_MODEL（必填）、
RODIT_LLM_TIMEOUT_S（可选）；密钥 RODIT_LLM_API_KEY 由客户端在每次调用时自己读。
这里只检查密钥"有没有设置"，不读出、不打印、不写进报告；报告里的接口地址只留主机名
（URL 里的用户名密码、路径、查询参数都不要）。密钥只放本地 .env（已 gitignore）或
云端环境的密钥设置，不进仓库（CLAUDE.md）。

单价（§14，不是密钥）：命令行 --price-input / --price-output 优先，其次环境变量
RODIT_LLM_PRICE_INPUT_PER_1K / RODIT_LLM_PRICE_OUTPUT_PER_1K（每 1000 token 的价格）；
两个都没有时只报 token，只给了一个视为配置错误。

跑之前先做一次连通性检查（preflight）：一个几个 token 的请求，地址、模型名、密钥不对时立刻报错
（客户端的错误信息里已经抹掉了密钥），不白跑一整轮、也不让整轮结果全是降级。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from rodit.core.ports import LLMClient, LLMMessage, LLMRequest
from rodit.eval.perf import Pricing
from rodit.llm import OpenAICompatibleClient

ENV_PREFIX = "RODIT_LLM_"
PRICE_INPUT_ENV = "RODIT_LLM_PRICE_INPUT_PER_1K"
PRICE_OUTPUT_ENV = "RODIT_LLM_PRICE_OUTPUT_PER_1K"
KEY_HINT = "密钥只放本地 .env 或环境的密钥设置，不要提交进仓库"


def client_from_env(*, prefix: str = ENV_PREFIX, **kwargs: Any) -> OpenAICompatibleClient:
    """按环境变量构造客户端（kwargs 原样交给 OpenAICompatibleClient.from_env，测试用来注入
    transport）。地址或模型没设、密钥没设时抛 RuntimeError（信息里只有变量名）。"""
    client = OpenAICompatibleClient.from_env(prefix=prefix, **kwargs)
    if not os.environ.get(client.api_key_env, "").strip():
        raise RuntimeError(
            f"环境变量 {client.api_key_env} 未设置：没有大模型 API key（{KEY_HINT}）"
        )
    return client


def describe_endpoint(client: object) -> str | None:
    """报告里显示的接口地址：只留协议和主机（端口），没有 base_url 的客户端返回 None。"""
    base_url = getattr(client, "base_url", None)
    if not isinstance(base_url, str) or not base_url:
        return None
    parts = urlsplit(base_url)
    if not parts.hostname:
        return None
    try:
        port = f":{parts.port}" if parts.port else ""
    except ValueError:  # 端口不是数字：不显示
        port = ""
    return f"{parts.scheme}://{parts.hostname}{port}"


def describe_model(client: object) -> str | None:
    model = getattr(client, "model", None)
    return model if isinstance(model, str) and model else None


def preflight(client: LLMClient) -> None:
    """连通性检查：失败时照常抛 LLMError / LLMTimeout（调用方报错退出）。"""
    client.complete(
        LLMRequest(
            purpose="other",
            messages=[LLMMessage(role="user", content="连通性检查，只回复：OK")],
            temperature=0.0,
            max_tokens=8,
        )
    )


def _price(flag: float | None, env_name: str, env: Mapping[str, str]) -> float | None:
    if flag is not None:
        value = flag
    else:
        raw = env.get(env_name, "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"环境变量 {env_name} 必须是数字（每 1000 token 的价格）") from None
    if not 0 <= value < float("inf"):  # 也挡住 NaN、无穷大
        raise ValueError(f"单价必须是非负的有限数字（{env_name} 或对应的命令行参数）")
    return value


def resolve_pricing(
    price_input: float | None,
    price_output: float | None,
    env: Mapping[str, str] | None = None,
) -> Pricing | None:
    """命令行优先，其次环境变量；都没有返回 None（只报 token）。"""
    env = os.environ if env is None else env
    pi = _price(price_input, PRICE_INPUT_ENV, env)
    po = _price(price_output, PRICE_OUTPUT_ENV, env)
    if pi is None and po is None:
        return None
    if pi is None or po is None:
        raise ValueError(
            "输入、输出单价要一起给（--price-input / --price-output，或环境变量 "
            f"{PRICE_INPUT_ENV} / {PRICE_OUTPUT_ENV}）"
        )
    return Pricing(input_per_1k=pi, output_per_1k=po)


__all__ = [
    "PRICE_INPUT_ENV",
    "PRICE_OUTPUT_ENV",
    "client_from_env",
    "describe_endpoint",
    "describe_model",
    "preflight",
    "resolve_pricing",
]
