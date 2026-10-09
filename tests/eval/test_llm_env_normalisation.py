"""环境变量给的大模型地址、模型名带首尾空白（从密钥文件、挂载文件注入时常带换行、空格）时，
评测实际跑的客户端（--llm env）记下的指纹和线上按部署规则算出来的指纹（core.config.llm_endpoint）
要一致：两边都去掉首尾空白。否则 differences 报一条两边看起来一模一样的"大模型"差异，
record-eval、上线节奏、验收报告都不认这次评测（方案 §13.5）。

接口地址的端口也一样：写不写协议的默认端口（https 的 443、http 的 80）是同一个接口，评测记下的地址
（llm_env.describe_endpoint）不带默认端口。

数据全部合成，不联网（不发请求，只构造客户端）。
"""

from __future__ import annotations

import pytest

from rodit.core.config import EngineConfig
from rodit.eval.cli import _run_fingerprint, env_client
from rodit.eval.fingerprint import (
    CLASSIFIER_PATH_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    current_fingerprint,
    differences,
    running_llm,
)
from rodit.eval.llm_env import describe_endpoint

KEY = "sk-synthetic-only-0000000000"  # 合成密钥，只用于测试


class _Client:
    """只有 base_url 的客户端（describe_endpoint 只读这一项）。"""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url


@pytest.mark.parametrize(
    ("base_url", "model"),
    [
        ("https://llm.example.invalid/v1", "m "),
        ("https://llm.example.invalid/v1", "synthetic-model\n"),
        (" https://llm.example.invalid/v1\n", " synthetic-model"),
    ],
)
def test_env_whitespace_does_not_split_eval_and_running_fingerprints(monkeypatch, base_url, model):
    monkeypatch.setenv(LLM_BASE_URL_ENV, base_url)
    monkeypatch.setenv(LLM_MODEL_ENV, model)
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)
    monkeypatch.delenv(CLASSIFIER_PATH_ENV, raising=False)
    config = EngineConfig()

    client = env_client(config)
    assert client.model == model.strip()
    assert running_llm(config) == (model.strip(), "https://llm.example.invalid")
    evaluated = _run_fingerprint(client, config, None, None)
    running = current_fingerprint(config)
    assert evaluated.llm_model == running.llm_model == model.strip()
    assert differences(evaluated, running) == []


@pytest.mark.parametrize(
    ("base_url", "shown"),
    [
        ("https://llm.example.invalid/v1", "https://llm.example.invalid"),
        ("https://llm.example.invalid:443/v1", "https://llm.example.invalid"),
        ("http://llm.example.invalid:80/v1", "http://llm.example.invalid"),
        ("HTTPS://user:pw@LLM.example.invalid:443/v1?x=1", "https://llm.example.invalid"),
        # 不是这个协议的默认端口：照写
        ("https://llm.example.invalid:8443/v1", "https://llm.example.invalid:8443"),
        ("https://llm.example.invalid:80/v1", "https://llm.example.invalid:80"),
        ("http://llm.example.invalid:443/v1", "http://llm.example.invalid:443"),
        ("http://10.0.0.8:8080/v1", "http://10.0.0.8:8080"),
        ("https://llm.example.invalid:abc/v1", "https://llm.example.invalid"),  # 端口不是数字
    ],
)
def test_describe_endpoint_drops_the_default_port(base_url: str, shown: str):
    assert describe_endpoint(_Client(base_url)) == shown


def test_the_default_port_does_not_change_what_the_eval_records(monkeypatch):
    """https://host 和 https://host:443 是同一个接口：评测指纹里记下的接口地址一样。"""
    monkeypatch.setenv(LLM_MODEL_ENV, "synthetic-model")
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)
    config = EngineConfig()
    recorded = set()
    for base_url in ("https://llm.example.invalid/v1", "https://llm.example.invalid:443/v1"):
        monkeypatch.setenv(LLM_BASE_URL_ENV, base_url)
        recorded.add(_run_fingerprint(env_client(config), config, None, None).llm_endpoint)
    assert recorded == {"https://llm.example.invalid"}
