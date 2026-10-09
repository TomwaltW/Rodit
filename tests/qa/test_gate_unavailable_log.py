"""质检：判断层不可用时把原因写进日志（§12.2 降级、§12.3 日志脱敏；真实大模型评测前的加固）。"""

from __future__ import annotations

import logging

from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import DecisionConfig, QAGateConfig
from rodit.core.enums import DraftKind, Intent
from rodit.core.errors import DecisionError, LLMError, LLMTimeout
from rodit.core.models import Draft, EvidenceBundle, QAResult
from rodit.core.ports import LLMResponse
from rodit.decision import build_llm_provider
from rodit.qa import QA_UNAVAILABLE, DefaultQualityGate
from rodit.testing.fakes import (
    FakeClock,
    FakeDecisionProvider,
    FakeLLMClient,
    make_context,
    qa_answers,
)

GATE_LOGGER = "rodit.qa.gate"
UNAVAILABLE_HEAD = "判断层质检不可用，这份大模型草稿不放行："
BUYER_TEXT = "这个是纯棉的吗？我手机 13812345678"
ANSWER = "亲，这款面料是棉 95%、氨纶 5% 哦"


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == GATE_LOGGER and r.levelno == logging.WARNING
    ]


def _check(provider, world, ev, route, caplog, *, kind: DraftKind = DraftKind.LLM) -> QAResult:
    gate = DefaultQualityGate(provider, QAGateConfig(), InMemoryAuditSink(), FakeClock())
    ctx = make_context(BUYER_TEXT, world, item_id="i_555")
    bundle = EvidenceBundle(evidences=[ev.item_attr("面料")])
    draft = Draft(text=ANSWER, kind=kind, evidence_ids=["item:i_555:attr:面料"])
    with caplog.at_level(logging.WARNING):
        return gate.check(ctx, route(Intent.PRODUCT_SPEC), draft, bundle)


def _failing(error: Exception, cause: BaseException | None = None) -> FakeDecisionProvider:
    error.__cause__ = cause
    return FakeDecisionProvider(qa_answers(), fail=error)


def test_judge_failure_logs_the_reason_and_its_root_cause_once(world, ev, route, caplog):
    llm = FakeLLMClient(lambda r: LLMTimeout("大模型调用超时（1 秒）"))
    provider = build_llm_provider(DecisionConfig(), purpose="qa", llm=llm)
    result = _check(provider, world, ev, route, caplog)
    assert result.degraded and not result.passed and result.reasons == [QA_UNAVAILABLE]
    [line] = _warnings(caplog)
    assert line == (
        UNAVAILABLE_HEAD
        + "DecisionError：大模型判断全部失败（1 次：超时 1）；根因 LLMTimeout（timeout）"
    )
    # 没有买家原文、草稿、依据、state、提示词
    for leaked in ("13812345678", "纯棉", "棉 95%", "面料", "## 状态", "你是电商平台"):
        assert leaked not in caplog.text


def test_truncated_judge_answer_is_logged_as_truncated(world, ev, route, caplog):
    """质检只采一次：回答被 max_tokens 截断就是质检不可用，日志里读得出原因。"""
    cut = LLMResponse(text='{"grounded": {"answer": "是", "p": 0.9', finish_reason="length")
    provider = build_llm_provider(DecisionConfig(), purpose="qa", llm=FakeLLMClient(lambda r: cut))
    result = _check(provider, world, ev, route, caplog)
    assert result.degraded and result.reasons == [QA_UNAVAILABLE]
    assert _warnings(caplog) == [
        UNAVAILABLE_HEAD + "DecisionError：大模型判断全部失败（1 次：截断 1）"
    ]


def test_logged_reason_is_masked_and_capped(world, ev, route, caplog):
    message = "判断层返回了手机 13812345678\n" + "很长的一段话" * 100
    result = _check(_failing(DecisionError(message)), world, ev, route, caplog)
    assert result.degraded
    [line] = _warnings(caplog)
    assert line.startswith(
        UNAVAILABLE_HEAD + "DecisionError：判断层返回了手机 [手机号] 很长的一段话"
    )
    assert "13812345678" not in caplog.text
    assert len(line) == len(UNAVAILABLE_HEAD + "DecisionError：") + 200  # 消息截到 200 字


def test_root_cause_text_is_never_logged(world, ev, route, caplog):
    """异常链最里层的原因只记类型（LLMError 另记 kind / status / code），它的消息不进日志。"""
    leak = "买家原话：帮我把地址改到文三路 188 号"
    cases = [
        (ValueError(f"input_value='{leak}'"), "；根因 ValueError"),
        (
            LLMError(leak, status=429, kind="rate_limit", code="rate_limit_exceeded"),
            "；根因 LLMError（rate_limit，HTTP 429，rate_limit_exceeded）",
        ),
    ]
    for cause, tail in cases:
        caplog.clear()
        provider = _failing(DecisionError("Jev 响应格式不对：1 处错误"), cause)
        assert _check(provider, world, ev, route, caplog).degraded
        assert _warnings(caplog) == [
            UNAVAILABLE_HEAD + "DecisionError：Jev 响应格式不对：1 处错误" + tail
        ]
        assert "买家原话" not in caplog.text and "文三路" not in caplog.text


def test_errors_that_are_not_business_errors_are_logged_by_type_only(world, ev, route, caplog):
    """判断层实现自己出错（KeyError / TypeError）：消息里是什么说不准，只记类型。"""
    for error in (TypeError("买家原话：13812345678"), KeyError("买家原话")):
        caplog.clear()
        assert _check(_failing(error), world, ev, route, caplog).degraded
        assert _warnings(caplog) == [UNAVAILABLE_HEAD + type(error).__name__]
        assert "买家原话" not in caplog.text


def test_llm_error_reaching_the_gate_is_logged_with_its_attributes(world, ev, route, caplog):
    error = LLMError("大模型服务错误（HTTP 503）", status=503, kind="server")
    assert _check(_failing(error), world, ev, route, caplog).degraded
    assert _warnings(caplog) == [
        UNAVAILABLE_HEAD + "LLMError（server，HTTP 503）：大模型服务错误（HTTP 503）"
    ]


def test_no_warning_when_nothing_failed(world, ev, route, caplog):
    """判断层答出来了（通过或拦截）、FAQ 只跑硬规则、没有配置判断层：都不记这条警告。"""
    passed = _check(FakeDecisionProvider(qa_answers()), world, ev, route, caplog)
    blocked = _check(FakeDecisionProvider(qa_answers(grounded=0.2)), world, ev, route, caplog)
    faq = _check(_failing(DecisionError("不该调用")), world, ev, route, caplog, kind=DraftKind.FAQ)
    unconfigured = _check(None, world, ev, route, caplog)
    assert passed.passed and not blocked.passed and not blocked.degraded
    assert faq.passed and not faq.model_checked
    assert unconfigured.degraded and unconfigured.reasons == [QA_UNAVAILABLE]
    assert _warnings(caplog) == []
