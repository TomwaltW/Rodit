"""MeteredLLMClient：按店铺记录大模型用量（§12.4），失败也记，计量故障不影响调用。"""

from __future__ import annotations

import json
import logging

import pytest

from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.context import shop_scope
from rodit.core.errors import LLMError, LLMTimeout
from rodit.core.ports import LLMClient, LLMMessage, LLMRequest, LLMResponse
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
