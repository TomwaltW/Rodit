"""T0 契约（docs/架构与分工.md §4.4 真实大模型评测前的加固）：大模型调用的失败原因、finish_reason、
llm.extra_body、连通性检查和评测报告 health 字段的接口。数据全部合成，不联网（只构造客户端，
不发请求）。

运行健康度的规则（rodit.eval.health）在 tests/eval/test_health.py；--skip-preflight 的行为在
tests/eval/test_cli_llm.py。这里只定并行期间不会变的名字、签名和默认值。
"""

from __future__ import annotations

import datetime
import inspect
from pathlib import Path
from typing import Any, get_args, get_type_hints

import pytest
import yaml
from pydantic import ValidationError

from rodit.core.config import (
    LLM_EXTRA_BODY_RESERVED,
    EngineConfig,
    LLMConfig,
    config_fingerprint,
    load_config,
)
from rodit.core.errors import LLM_ERROR_KINDS, LLMError, LLMErrorKind, LLMTimeout, RoditError
from rodit.core.ports import LLMClient, LLMMessage, LLMRequest, LLMResponse
from rodit.eval.cli import EvalCliReport, build_parser
from rodit.eval.health import RunHealth
from rodit.eval.llm_env import PreflightFailed, preflight, via_proxy
from rodit.llm import OpenAICompatibleClient
from rodit.testing.fakes import FakeLLMClient

YAML = Path(__file__).resolve().parents[2] / "config" / "rodit.yaml"
THINKING_OFF = {"thinking": {"type": "disabled"}}
PING = LLMRequest(purpose="other", messages=[LLMMessage(role="user", content="合成的请求")])


def _engine(extra_body: dict[str, Any]) -> EngineConfig:
    return EngineConfig(llm=LLMConfig(extra_body=extra_body))


# ================================================================ 异常：失败原因


def test_llm_error_carries_an_optional_cause():
    plain = LLMError("大模型接口限流（HTTP 429）")
    assert isinstance(plain, RoditError) and str(plain) == "大模型接口限流（HTTP 429）"
    assert (plain.status, plain.kind, plain.code) == (None, None, None)
    empty = LLMError()
    assert str(empty) == "" and (empty.status, empty.kind, empty.code) == (None, None, None)
    full = LLMError("限流", status=429, kind="rate_limit", code="rate_limit_exceeded")
    assert str(full) == "限流" and full.args == ("限流",)
    assert (full.status, full.kind, full.code) == (429, "rate_limit", "rate_limit_exceeded")
    assert LLMError("密钥的环境变量没设", kind="config").status is None  # 请求没发出去：没有状态码
    with pytest.raises(ValueError, match="quota"):
        LLMError("x", kind="quota")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        LLMError("x", 429)  # type: ignore[misc]  # 原因只能按关键字传


def test_llm_error_kinds_are_enumerated():
    assert (
        LLM_ERROR_KINDS
        == get_args(LLMErrorKind)
        == (
            "timeout",
            "rate_limit",
            "auth",
            "billing",
            "bad_request",
            "content_filter",
            "server",
            "redirect",
            "network",
            "malformed",
            "config",
        )
    )
    for kind in LLM_ERROR_KINDS:
        assert LLMError("x", kind=kind).kind == kind


def test_llm_timeout_defaults_to_the_timeout_kind():
    plain = LLMTimeout("大模型调用超时（1 秒）")
    assert isinstance(plain, LLMError) and str(plain) == "大模型调用超时（1 秒）"
    assert (plain.status, plain.kind, plain.code) == (None, "timeout", None)
    assert LLMTimeout().kind == "timeout" and str(LLMTimeout()) == ""
    gateway = LLMTimeout("网关超时", status=504, code="gateway_timeout")
    assert (gateway.status, gateway.kind, gateway.code) == (504, "timeout", "gateway_timeout")
    assert LLMTimeout("x", kind="network").kind == "network"
    with pytest.raises(ValueError):
        LLMTimeout("x", kind="slow")  # type: ignore[arg-type]
    # 和 LLMError 同一套参数，只有 kind 的默认值不同
    params = inspect.signature(LLMTimeout).parameters
    assert list(params) == list(inspect.signature(LLMError).parameters)
    assert list(params) == ["msg", "status", "kind", "code"]
    assert params["kind"].default == "timeout"
    assert inspect.signature(LLMError).parameters["kind"].default is None


@pytest.mark.parametrize("error_type", [LLMError, LLMTimeout])
def test_existing_ways_of_raising_keep_working(error_type: type[LLMError]):
    """没带原因的老写法照旧能用：一个位置参数、不带参数、直接抛类、按原类型重建。"""
    original = error_type("合成的故障")
    rebuilt = type(original)(str(original))  # 客户端抹掉密钥后按原类型重抛
    assert type(rebuilt) is error_type and str(rebuilt) == "合成的故障"
    with pytest.raises(LLMError):
        raise error_type
    fake = FakeLLMClient(handler=lambda request: error_type("合成的故障"))
    with pytest.raises(error_type, match="合成的故障"):
        fake.complete(PING)


# ================================================================ 端口：finish_reason


def test_llm_response_finish_reason_is_optional():
    assert LLMResponse(text="好的").finish_reason is None
    cut = LLMResponse(text='{"answer": "亲，这款', finish_reason="length")
    assert cut.finish_reason == "length"
    assert LLMResponse.model_validate(cut.model_dump(mode="json")) == cut
    assert LLMResponse(text="好的", finish_reason="stop").finish_reason == "stop"
    # 不知道 finish_reason 的实现（测试替身）不用填
    fake = FakeLLMClient(handler=lambda request: "OK")
    assert isinstance(fake, LLMClient) and fake.complete(PING).finish_reason is None


# ================================================================ 配置：llm.extra_body


def test_llm_extra_body_defaults_to_empty_and_accepts_vendor_parameters():
    assert LLMConfig().extra_body == {} and EngineConfig().llm.extra_body == {}
    assert LLMConfig().extra_body is not LLMConfig().extra_body
    for body in (
        THINKING_OFF,
        {"enable_thinking": False},
        {"top_p": 0.8, "stop": ["\n\n"], "seed": 7, "user": None, "metadata": {"标签": "合成"}},
    ):
        assert LLMConfig(extra_body=body).extra_body == body
    loaded = EngineConfig.model_validate({"llm": {"extra_body": THINKING_OFF}})
    assert loaded.llm.extra_body == THINKING_OFF
    with pytest.raises(ValidationError):
        LLMConfig(extra_body=["enable_thinking"])  # type: ignore[arg-type]  # 必须是对象


def test_reserved_keys_are_the_ones_the_client_fills_per_request():
    assert isinstance(LLM_EXTRA_BODY_RESERVED, frozenset)
    assert LLM_EXTRA_BODY_RESERVED == {
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "stream",
        "response_format",
        "logprobs",
        "top_logprobs",
    }
    # 和客户端实际填的字段对得上（只构造请求体，不发请求）
    client = OpenAICompatibleClient(base_url="https://llm.example.invalid/v1", model="synthetic")
    filled: set[str] = set()
    for flags in ({"json_mode": True}, {"want_logprobs": True}):
        filled |= set(client.build_payload(PING.model_copy(update=flags)))
    assert filled == LLM_EXTRA_BODY_RESERVED


@pytest.mark.parametrize("key", sorted(LLM_EXTRA_BODY_RESERVED))
def test_llm_extra_body_rejects_fields_the_client_fills_itself(key: str):
    with pytest.raises(ValidationError) as caught:
        LLMConfig(extra_body={"enable_thinking": False, key: 1})
    (error,) = caught.value.errors()
    assert error["loc"] == ("extra_body",)
    assert key in error["msg"] and "enable_thinking" not in error["msg"]  # 点名是哪个键
    with pytest.raises(ValidationError):
        EngineConfig.model_validate({"llm": {"extra_body": {key: 1}}})


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        {"deep": [float("-inf")]},
        {1, 2},
        b"bytes",
        object(),
        datetime.date(2026, 1, 1),  # YAML 里没加引号的日期读出来是 date
    ],
    ids=["nan", "inf", "nested-inf", "set", "bytes", "object", "date"],
)
def test_llm_extra_body_rejects_values_that_are_not_json(value: Any):
    with pytest.raises(ValidationError) as caught:
        LLMConfig(extra_body={"enable_thinking": False, "vendor_option": value})
    (error,) = caught.value.errors()
    assert "vendor_option" in error["msg"] and "enable_thinking" not in error["msg"]


def test_extra_body_is_part_of_the_config_fingerprint():
    base = config_fingerprint(EngineConfig())
    assert config_fingerprint(_engine({})) == base
    off = config_fingerprint(_engine(THINKING_OFF))
    alibaba = config_fingerprint(_engine({"enable_thinking": False}))
    assert len({base, off, alibaba}) == 3
    # 嵌套里面一个取值不同也算不同
    assert config_fingerprint(_engine({"thinking": {"type": "enabled"}})) not in {base, off}
    # 同样的取值，键的顺序不同：指纹一样
    one = _engine({"a": 1, "b": {"x": 1, "y": 2}})
    other = _engine({"b": {"y": 2, "x": 1}, "a": 1})
    assert config_fingerprint(one) == config_fingerprint(other)


def test_yaml_keeps_extra_body_as_a_commented_example():
    cfg = load_config(YAML)
    assert cfg.llm.extra_body == {} and cfg.llm == EngineConfig().llm
    # 注释里的两个例子去掉注释就是能用的配置
    lines = [line.strip() for line in YAML.read_text(encoding="utf-8").splitlines()]
    examples = [
        yaml.safe_load(line.removeprefix("# "))["extra_body"]
        for line in lines
        if line.startswith("# extra_body:")
    ]
    assert examples == [THINKING_OFF, {"enable_thinking": False}]
    for body in examples:
        assert LLMConfig(extra_body=body).extra_body == body


# ================================================================ 评测：连通性检查和报告的接口


def test_preflight_names_and_signatures():
    assert issubclass(PreflightFailed, RuntimeError)
    assert not issubclass(PreflightFailed, LLMError)  # 命令行按 RuntimeError 报出，不再包一层
    params = inspect.signature(preflight).parameters
    assert list(params) == ["client", "config"] and params["config"].default is None
    hints = get_type_hints(preflight)
    assert hints["config"] == EngineConfig | None and hints["return"] == list[str]
    assert list(inspect.signature(via_proxy).parameters) == ["client"]
    assert get_type_hints(via_proxy)["return"] == bool | None


def test_via_proxy_is_unknown_without_an_http_client_or_with_an_injected_transport():
    def transport(url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float):
        raise AssertionError("不该发请求")

    injected = OpenAICompatibleClient(
        base_url="https://llm.example.invalid/v1", model="synthetic", transport=transport
    )
    assert via_proxy(injected) is None
    assert via_proxy(FakeLLMClient(handler=lambda request: "OK")) is None
    assert via_proxy(object()) is None


def test_eval_report_has_an_optional_health_block():
    field = EvalCliReport.model_fields["health"]
    assert field.default is None and not field.is_required()
    assert set(get_args(field.annotation)) == {RunHealth, type(None)}


def test_skip_preflight_flag_is_off_by_default():
    parser = build_parser()
    assert parser.parse_args([]).skip_preflight is False
    assert parser.parse_args(["--llm", "env", "--skip-preflight"]).skip_preflight is True
