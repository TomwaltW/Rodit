"""MeteredLLMClient：按店铺记录大模型用量（§12.4），失败也记，计量故障不影响调用。"""

from __future__ import annotations

import json
import logging
import urllib.error

import pytest

from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.context import shop_scope
from rodit.core.errors import LLMError, LLMTimeout
from rodit.core.ports import LLMClient, LLMMessage, LLMRequest, LLMResponse
from rodit.eval.health import error_label
from rodit.llm import MeteredLLMClient, OpenAICompatibleClient
from rodit.testing.fakes import FakeClock, FakeLLMClient
from rodit.testing.fixtures import SHOP_A, SHOP_B


def req(text: str = "发什么快递", **kw) -> LLMRequest:
    return LLMRequest(
        purpose=kw.pop("purpose", "generate"),
        messages=[LLMMessage(role="user", content=text)],
        **kw,
    )


class StepTimer:
    """每次调用前进固定秒数的假单调时钟。"""

    def __init__(self, step: float) -> None:
        self.t = 100.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


class BrokenSink:
    def record(self, event):
        raise RuntimeError("audit db down")

    def log_decision(self, entry):
        raise RuntimeError("audit db down")


def test_implements_llm_client_protocol():
    metered = MeteredLLMClient(FakeLLMClient(responses=[]), InMemoryAuditSink(), FakeClock())
    assert isinstance(metered, LLMClient)
    assert metered.model == "fake-llm"


def test_success_records_usage_for_request_shop():
    clock, sink = FakeClock(), InMemoryAuditSink()
    inner = FakeLLMClient(
        responses=[LLMResponse(text="ok", model="m-served", input_tokens=120, output_tokens=8)]
    )
    metered = MeteredLLMClient(inner, sink, clock, timer=StepTimer(0.25))
    resp = metered.complete(req(shop_id=SHOP_A, purpose="route"))
    assert resp.text == "ok"
    [event] = sink.events
    assert event.shop_id == SHOP_A
    assert event.action == AuditAction.LLM_USAGE == "llm.usage"
    assert event.actor == "system"
    assert event.created_at == clock.now()
    assert event.detail == {
        "purpose": "route",
        "model": "m-served",
        "input_tokens": 120,
        "output_tokens": 8,
        "latency_ms": 250,
    }


def test_shop_falls_back_to_context_and_request_wins():
    sink = InMemoryAuditSink()
    metered = MeteredLLMClient(FakeLLMClient(lambda r: "ok"), sink, FakeClock())
    with shop_scope(SHOP_A):
        metered.complete(req())
        metered.complete(req(shop_id=SHOP_B))
    metered.complete(req())  # 没有店铺归属
    assert [e.shop_id for e in sink.events] == [SHOP_A, SHOP_B, None]


def test_failure_is_recorded_and_reraised():
    sink = InMemoryAuditSink()
    inner = FakeLLMClient(responses=[LLMTimeout("slow")])
    metered = MeteredLLMClient(inner, sink, FakeClock(), timer=StepTimer(1.0))
    with pytest.raises(LLMTimeout):
        metered.complete(req(shop_id=SHOP_A, purpose="qa"))
    [event] = sink.events
    assert event.detail == {
        "purpose": "qa",
        "model": "fake-llm",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 1000,
        "error": "LLMTimeout",
        "kind": "timeout",  # LLMTimeout 的失败原因默认就是 timeout
    }


def test_failure_without_a_cause_keeps_the_old_detail():
    """没带失败原因的 LLMError（老写法、别的 LLMClient 实现）：detail 和加可选键之前一模一样。"""
    sink = InMemoryAuditSink()
    inner = FakeLLMClient(responses=[LLMError("boom")])
    metered = MeteredLLMClient(inner, sink, FakeClock(), timer=StepTimer(1.0))
    with pytest.raises(LLMError, match="boom"):
        metered.complete(req(shop_id=SHOP_A, purpose="qa"))
    [event] = sink.events
    assert event.detail == {
        "purpose": "qa",
        "model": "fake-llm",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 1000,
        "error": "LLMError",
    }


@pytest.mark.parametrize(
    ("error", "cause"),
    [
        (
            LLMError("限流", status=429, kind="rate_limit", code="rate_limit_exceeded"),
            {"kind": "rate_limit", "status": 429, "code": "rate_limit_exceeded"},
        ),
        # 哪项知道写哪项
        (LLMError("连不上", kind="network"), {"kind": "network"}),
        (LLMError("服务出错", status=503, kind="server"), {"kind": "server", "status": 503}),
        (LLMError("没说哪一类", status=418), {"status": 418}),
        (
            LLMTimeout("网关超时", status=504, code="gateway_timeout"),
            {"kind": "timeout", "status": 504, "code": "gateway_timeout"},
        ),
        (LLMTimeout("特意不给原因", kind=None), {}),
        # 类型不对的不写：状态码要是整数（评测的 error_label 只认整数），代码要是非空的字符串
        (LLMError("x", status="429", kind="rate_limit"), {"kind": "rate_limit"}),  # type: ignore[arg-type]
        (LLMError("x", status=True, code="  "), {}),
        (LLMError("x", status=402, code=1113), {"status": 402, "code": "1113"}),  # type: ignore[arg-type]
        (LLMError("x", code={"nested": "对象"}), {}),  # type: ignore[arg-type]
    ],
)
def test_failure_records_the_cause_it_knows(error: LLMError, cause: dict):
    sink = InMemoryAuditSink()
    metered = MeteredLLMClient(
        FakeLLMClient(responses=[error]), sink, FakeClock(), timer=StepTimer(0.5)
    )
    with pytest.raises(LLMError):
        metered.complete(req(shop_id=SHOP_A, purpose="route"))
    [event] = sink.events
    assert event.detail == {
        "purpose": "route",
        "model": "fake-llm",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 500,
        "error": type(error).__name__,
        **cause,
    }


def test_cause_is_only_read_from_llm_errors():
    """别的异常碰巧也有 status / code 这样的属性（urllib 的 HTTPError）：不是失败原因，不写。"""

    class Odd(RuntimeError):
        status = 500
        kind = "server"
        code = "odd"

    sink = InMemoryAuditSink()
    metered = MeteredLLMClient(FakeLLMClient(responses=[Odd("bug")]), sink, FakeClock())
    with pytest.raises(Odd):
        metered.complete(req(shop_id=SHOP_A))
    assert sink.events[0].detail.keys() == {
        "purpose",
        "model",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "error",
    }
    assert sink.events[0].detail["error"] == "Odd"


def test_failure_detail_never_contains_the_error_message():
    """失败原因只有三个短字段；异常消息（可能带模型服务回显的买家文字）不进 detail。"""
    sink = InMemoryAuditSink()
    error = LLMError("模型服务回显：买家手机号 13800000000", status=400, kind="bad_request")
    metered = MeteredLLMClient(FakeLLMClient(responses=[error]), sink, FakeClock())
    with pytest.raises(LLMError):
        metered.complete(req(shop_id=SHOP_A))
    dumped = json.dumps(sink.events[0].detail, ensure_ascii=False)
    assert "13800000000" not in dumped and "回显" not in dumped


# ---------------------------------------------------------------- 成功但不正常：截断、空文本


def _usage_detail(response: LLMResponse) -> dict:
    sink = InMemoryAuditSink()
    metered = MeteredLLMClient(
        FakeLLMClient(responses=[response]), sink, FakeClock(), timer=StepTimer(0.25)
    )
    assert metered.complete(req(shop_id=SHOP_A, purpose="route")) == response  # 照常返回
    [event] = sink.events
    return event.detail


@pytest.mark.parametrize("finish_reason", [None, "stop", ""])
def test_ordinary_success_detail_is_unchanged(finish_reason):
    """正常结束（stop）或者接口没给 finish_reason：detail 和加可选键之前一模一样。"""
    response = LLMResponse(
        text="ok", model="m-served", input_tokens=120, output_tokens=8, finish_reason=finish_reason
    )
    assert _usage_detail(response) == {
        "purpose": "route",
        "model": "m-served",
        "input_tokens": 120,
        "output_tokens": 8,
        "latency_ms": 250,
    }


@pytest.mark.parametrize(
    ("text", "finish_reason", "extra"),
    [
        # 被 max_tokens 截断：token 照常计量，多记一项 finish_reason
        ('{"intent": "logis', "length", {"finish_reason": "length"}),
        ("", "length", {"finish_reason": "length", "empty": True}),
        ("", "stop", {"empty": True}),
        (" \n\t", None, {"empty": True}),  # 只有空白也算空
        ("ok", "tool_calls", {"finish_reason": "tool_calls"}),
    ],
)
def test_truncated_or_empty_success_is_flagged(text, finish_reason, extra):
    response = LLMResponse(
        text=text,
        model="m-served",
        input_tokens=120,
        output_tokens=304,
        finish_reason=finish_reason,
    )
    assert _usage_detail(response) == {
        "purpose": "route",
        "model": "m-served",
        "input_tokens": 120,
        "output_tokens": 304,
        "latency_ms": 250,
        **extra,
    }


def test_audit_failure_does_not_break_the_call(caplog):
    metered = MeteredLLMClient(FakeLLMClient(lambda r: "ok"), BrokenSink(), FakeClock())
    with caplog.at_level(logging.WARNING, logger="rodit.llm.metering"):
        assert metered.complete(req(shop_id=SHOP_A)).text == "ok"
    assert "RuntimeError" in caplog.text
    failing = MeteredLLMClient(
        FakeLLMClient(responses=[LLMError("boom")]), BrokenSink(), FakeClock()
    )
    with pytest.raises(LLMError, match="boom"):  # 抛出的是原调用的异常，不是审计的
        failing.complete(req(shop_id=SHOP_A))


def test_usage_detail_never_contains_prompt_or_output_text(caplog):
    sink = InMemoryAuditSink()
    secret_text = "我的手机号 13800000000，地址幸福路 88 号"
    metered = MeteredLLMClient(FakeLLMClient(lambda r: "回复原文"), sink, FakeClock())
    metered.complete(req(secret_text, shop_id=SHOP_A))
    dumped = json.dumps([e.detail for e in sink.events], ensure_ascii=False)
    assert "13800000000" not in dumped and "幸福路" not in dumped and "回复原文" not in dumped


def test_model_falls_back_to_inner_model_when_response_has_none():
    sink = InMemoryAuditSink()
    inner = FakeLLMClient(responses=[LLMResponse(text="x")], model="configured-model")
    MeteredLLMClient(inner, sink, FakeClock()).complete(req(shop_id=SHOP_A))
    assert sink.events[0].detail["model"] == "configured-model"


def test_wraps_openai_compatible_client(monkeypatch):
    monkeypatch.setenv("RODIT_LLM_API_KEY", "sk-synthetic-meter")
    body = {
        "model": "served",
        "choices": [{"message": {"content": "亲~"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 3},
    }
    inner = OpenAICompatibleClient(
        base_url="https://llm.example.invalid/v1",
        model="configured",
        transport=lambda url, headers, payload, timeout: (200, body),
    )
    sink = InMemoryAuditSink()
    with shop_scope(SHOP_B):
        MeteredLLMClient(inner, sink, FakeClock()).complete(req())
    [event] = sink.list_events(SHOP_B, action=AuditAction.LLM_USAGE)
    assert event.detail["input_tokens"] == 50 and event.detail["output_tokens"] == 3
    assert event.detail["model"] == "served"
    assert "sk-synthetic-meter" not in json.dumps(event.detail)


def _metered_openai_call(monkeypatch, outcome, *, key: str | None = "sk-synthetic-meter") -> dict:
    """真实的 OpenAICompatibleClient（假 transport）包一层计量，返回这次调用的 llm.usage detail。
    outcome 是 (状态码, 响应体)，或者 transport 要抛的异常。"""
    if key is None:
        monkeypatch.delenv("RODIT_LLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("RODIT_LLM_API_KEY", key)

    def transport(url, headers, payload, timeout):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    inner = OpenAICompatibleClient(
        base_url="https://llm.example.invalid/v1", model="configured", transport=transport
    )
    sink = InMemoryAuditSink()
    metered = MeteredLLMClient(inner, sink, FakeClock(), timer=StepTimer(0.25))
    try:
        metered.complete(req(shop_id=SHOP_A, purpose="route"))
    except LLMError:
        pass
    [event] = sink.events
    assert event.detail.pop("latency_ms") == 250
    assert event.detail.pop("purpose") == "route"
    return event.detail


def _choice(content, finish_reason: str) -> dict:
    return {
        "model": "served",
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 304},
    }


_FAILED = {"model": "configured", "input_tokens": 0, "output_tokens": 0}


@pytest.mark.parametrize(
    ("outcome", "key", "cause", "label"),
    [
        (
            (
                429,
                {
                    "error": {
                        "type": "limit_error",
                        "code": "rate_limit_exceeded",
                        "message": "原文",
                    }
                },
            ),
            "sk-synthetic-meter",
            {
                "error": "LLMError",
                "kind": "rate_limit",
                "status": 429,
                "code": "rate_limit_exceeded",
            },
            "http_429:rate_limit_exceeded",
        ),
        (
            (402, {"error": {"message": "余额不足"}}),
            "sk-synthetic-meter",
            {"error": "LLMError", "kind": "billing", "status": 402},
            "http_402",
        ),
        (
            TimeoutError("timed out"),
            "sk-synthetic-meter",
            {"error": "LLMTimeout", "kind": "timeout"},
            "timeout",
        ),
        (
            urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
            "sk-synthetic-meter",
            {"error": "LLMError", "kind": "network"},
            "network",
        ),
        ((200, _choice("好的", "stop")), None, {"error": "LLMError", "kind": "config"}, "config"),
        (
            (200, _choice("好的", "stop")),
            "sk-synthetic\nmeter",
            {"error": "LLMError", "kind": "config"},
            "config",
        ),
        (
            (200, _choice("", "content_filter")),
            "sk-synthetic-meter",
            {"error": "LLMError", "kind": "content_filter"},
            "content_filter",
        ),
        # 思考模式没关时常见的形态：content 是 null。读不出回答，状态码照带；报告里的标签按原因
        # （malformed），不写成 http_200
        (
            (200, _choice(None, "stop")),
            "sk-synthetic-meter",
            {"error": "LLMError", "kind": "malformed", "status": 200},
            "malformed",
        ),
    ],
    ids=[
        "429",
        "402",
        "timeout",
        "refused",
        "no-key",
        "bad-key",
        "content-filter",
        "content-null",
    ],
)
def test_openai_client_failures_are_metered_with_their_cause(
    monkeypatch, outcome, key, cause, label
):
    """客户端给的失败原因原样进 llm.usage，评测报告按 rodit.eval.health.error_label 归类
    （docs/架构与分工.md §4.4）。"""
    detail = _metered_openai_call(monkeypatch, outcome, key=key)
    assert detail == {**_FAILED, **cause}
    assert error_label(detail) == label
    assert "原文" not in json.dumps(detail, ensure_ascii=False)


@pytest.mark.parametrize(
    ("content", "finish_reason", "extra"),
    [
        ("好的", "stop", {}),
        ('{"intent": "logis', "length", {"finish_reason": "length"}),
        ("", "length", {"finish_reason": "length", "empty": True}),
        # 截断时 content 是 null（思考模型把额度花在思考上）：和空字符串一样，token 不丢
        (None, "length", {"finish_reason": "length", "empty": True}),
        # 输出在思考块里被截断：思考内容不算回答，是空的
        ('<think>先拟一版 {"intent": "logis', "length", {"finish_reason": "length", "empty": True}),
        ("<think>想完了，没有回答</think>", "stop", {"empty": True}),
    ],
)
def test_openai_client_truncation_is_metered_with_its_tokens(
    monkeypatch, content, finish_reason, extra
):
    """被截断、空文本的回答按成功计量：token 照记（抛异常的话就成了 0 token，成本估算偏低），
    另外标上 finish_reason / empty。"""
    detail = _metered_openai_call(monkeypatch, (200, _choice(content, finish_reason)))
    assert detail == {"model": "served", "input_tokens": 100, "output_tokens": 304, **extra}
