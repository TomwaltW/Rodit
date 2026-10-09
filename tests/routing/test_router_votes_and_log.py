"""DecisionRouter：投票粒度、全自动的关键词"要求人工"兜底、decision_log 写失败不影响路由、
判断层不可用时把原因写进日志。

（集成审查修复；真实大模型评测前的加固）
"""

from __future__ import annotations

import logging

from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import DecisionConfig, RoutingConfig
from rodit.core.decision import ChoiceAnswer, DecisionResponse
from rodit.core.enums import Intent, Mode, RouteAction
from rodit.core.errors import DecisionError, LLMError, LLMTimeout, StorageUnavailable
from rodit.core.ports import LLMResponse
from rodit.core.questions import Q_INTENT
from rodit.decision import RuleProvider, build_llm_provider
from rodit.routing import ALERT_MULTI_UNSPLIT, DecisionRouter, RoutingPolicy
from rodit.testing.fakes import (
    FakeClock,
    FakeDecisionProvider,
    FakeLLMClient,
    make_context,
    router_answers,
)
from rodit.testing.fixtures import build_demo_world

W = build_demo_world()
A = RouteAction
AUTO_CFG = {"auto_reply_enabled": True}


def _router(provider, *, audit=None):
    audit = audit if audit is not None else InMemoryAuditSink()
    router = DecisionRouter(
        provider, RuleProvider(), RoutingPolicy(RoutingConfig()), audit, FakeClock(W.now)
    )
    return router, audit


def _vote(intent: str, *, need_human: bool = False, multi: bool = False) -> dict:
    yes_no = {True: {"answer": "是", "p": 0.9}, False: {"answer": "否", "p": 0.1}}
    return {
        "intent": {"choice": intent},
        "need_human": yes_no[need_human],
        "emotion": {"score": 0},
        "multi_intent": yes_no[multi],
    }


def _default_route_provider(votes: list[dict] | None = None, *, every: dict | None = None):
    """默认配置（vote_samples=3、过半数有效票）的路由判断层：按顺序给出采样，或每次都给 every。"""
    llm = FakeLLMClient(lambda r: every) if every is not None else FakeLLMClient(responses=votes)
    return build_llm_provider(DecisionConfig(), purpose="route", llm=llm)


def test_two_of_three_votes_for_a_human_transfer_in_auto_mode():
    """审查复现：need_human 2/3 票，默认阈值 0.7 → 以前落到规则 6 自动回复。"""
    provider = _default_route_provider(
        [
            _vote("greeting", need_human=True),
            _vote("greeting", need_human=True),
            _vote("greeting"),
        ]
    )
    router, _ = _router(provider)
    ctx = make_context("你好，在吗", W, mode=Mode.AUTO, config_update=AUTO_CFG)
    d = router.route(ctx)
    assert (d.action, d.rule, d.auto_eligible) == (A.TRANSFER_HUMAN, 2, False)


def test_auto_mode_keyword_need_human_overrides_a_missed_judgement():
    provider = _default_route_provider(every=_vote("greeting"))
    router, _ = _router(provider)
    ctx = make_context("你好，我要找人工客服", W, mode=Mode.AUTO, config_update=AUTO_CFG)
    d = router.route(ctx)
    assert (d.action, d.rule, d.degraded, d.provider) == (A.TRANSFER_HUMAN, 2, False, "incloud_llm")
    # 智能辅助：以判断层为准（有人把关），不额外看关键词
    assisted = router.route(make_context("你好，我要找人工客服", W))
    assert (assisted.action, assisted.rule, assisted.need_human) == (A.ANSWER, 6, 0.0)
    # 全自动、没提人工：照常自动回复
    calm = router.route(make_context("你好，在吗", W, mode=Mode.AUTO, config_update=AUTO_CFG))
    assert (calm.rule, calm.auto_eligible) == (6, True)


def test_multi_intent_with_unanimous_main_intent_alerts_the_agent():
    provider = _default_route_provider(every=_vote("product_spec", multi=True))
    router, _ = _router(provider)
    d = router.route(make_context("这个是纯棉的吗？另外什么时候发货？", W))
    assert (d.action, d.rule, d.multi_intent) == (A.ANSWER, 6, 1.0)
    assert ALERT_MULTI_UNSPLIT in d.alerts


class _BrokenAudit(InMemoryAuditSink):
    def log_decision(self, entry) -> None:  # noqa: ANN001
        raise StorageUnavailable("decision_log 写入失败：连接数已满 13812345678")


def test_decision_log_failure_does_not_block_routing(caplog):
    provider = FakeDecisionProvider(router_answers("shipping_policy", 0.92), name="incloud_llm")
    router, _ = _router(provider, audit=_BrokenAudit())
    with caplog.at_level(logging.WARNING):
        d = router.route(make_context("什么时候发货", W))
    assert (d.action, d.intent, d.degraded) == (A.ANSWER, Intent.SHIPPING_POLICY, False)
    assert "StorageUnavailable" in caplog.text
    assert "13812345678" not in caplog.text  # 只记异常类型，不记消息


# ------------------------------------------------- 判断层不可用：原因写进日志（§12.2、§12.3）

ROUTER_LOGGER = "rodit.routing.router"
FALLBACK_HEAD = "路由判断层不可用，退回关键词规则："


def _fallback_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == ROUTER_LOGGER and r.levelno == logging.WARNING
    ]


def _failing(error: Exception, cause: BaseException | None = None) -> FakeDecisionProvider:
    error.__cause__ = cause
    return FakeDecisionProvider(router_answers("stock"), name="incloud_llm", fail=error)


def test_judge_fallback_logs_the_reason_and_its_root_cause_once(caplog):
    llm = FakeLLMClient(lambda r: LLMTimeout("大模型调用超时（1 秒）"))
    router, _ = _router(build_llm_provider(DecisionConfig(), purpose="route", llm=llm))
    with caplog.at_level(logging.WARNING):
        d = router.route(make_context("什么时候发货，我手机 13812345678", W))
    assert (d.degraded, d.provider) == (True, "rules")
    [line] = _fallback_warnings(caplog)
    assert line == (
        FALLBACK_HEAD
        + "DecisionError：大模型判断全部失败（3 次：超时 3）；根因 LLMTimeout（timeout）"
    )
    # 没有买家原文、state、提示词
    for leaked in ("13812345678", "什么时候发货", "## 状态", '"messages"', "你是电商平台"):
        assert leaked not in caplog.text


def test_truncated_votes_are_logged_as_truncated(caplog):
    """三票都被 max_tokens 截断：调用本身没有失败，日志里照样读得出为什么退回了关键词规则。"""
    cut = LLMResponse(text='{"intent": {"choice": "sto', finish_reason="length")
    llm = FakeLLMClient(lambda r: cut)
    router, _ = _router(build_llm_provider(DecisionConfig(), purpose="route", llm=llm))
    with caplog.at_level(logging.WARNING):
        d = router.route(make_context("还有货吗", W))
    assert (d.degraded, d.provider) == (True, "rules")
    assert _fallback_warnings(caplog) == [
        FALLBACK_HEAD + "DecisionError：大模型判断全部失败（3 次：截断 3）"
    ]


def test_logged_reason_is_masked_and_capped(caplog):
    """validate_response 的消息会引用判断层给的选项文字（模型给什么就是什么）：遮个人信息、截断。"""
    bogus = "手机 13812345678\n" + "很长的一段话" * 100

    class Sloppy:
        name = "sloppy"
        accepts_buyer_data = True

        def decide(self, request):
            answers = {
                **router_answers("stock"),
                Q_INTENT: ChoiceAnswer(choice=bogus, confidence=1.0),
            }
            return DecisionResponse(model=request.model, answers=answers)

    router, _ = _router(Sloppy())
    with caplog.at_level(logging.WARNING):
        d = router.route(make_context("还有货吗", W))
    assert d.degraded and d.provider == "rules"
    [line] = _fallback_warnings(caplog)
    assert line.startswith(FALLBACK_HEAD + "DecisionError：问题 'intent' 的选项 '手机 [手机号]")
    assert "13812345678" not in caplog.text
    assert len(line) == len(FALLBACK_HEAD + "DecisionError：") + 200  # 消息截到 200 字
    # 消息里的换行压成空格：一条降级一行日志
    caplog.clear()
    router, _ = _router(_failing(DecisionError("判断层不可用\n  第二行")))
    with caplog.at_level(logging.WARNING):
        router.route(make_context("还有货吗", W))
    assert _fallback_warnings(caplog) == [FALLBACK_HEAD + "DecisionError：判断层不可用 第二行"]


def test_root_cause_text_is_never_logged(caplog):
    """异常链最里层的原因只记类型（LLMError 另记 kind / status / code）：它的消息可能带着模型输出
    （比如校验错误里的输入值），不进日志。"""
    leak = "买家原话：帮我把地址改到文三路 188 号"
    cases = [
        (ValueError(f"input_value='{leak}'"), "；根因 ValueError"),
        (
            LLMError(leak, status=429, kind="rate_limit", code="rate_limit_exceeded"),
            "；根因 LLMError（rate_limit，HTTP 429，rate_limit_exceeded）",
        ),
        (LLMError(leak), "；根因 LLMError"),
    ]
    for cause, tail in cases:
        caplog.clear()
        router, _ = _router(_failing(DecisionError("Jev 响应格式不对：1 处错误"), cause))
        with caplog.at_level(logging.WARNING):
            assert router.route(make_context("还有货吗", W)).degraded
        assert _fallback_warnings(caplog) == [
            FALLBACK_HEAD + "DecisionError：Jev 响应格式不对：1 处错误" + tail
        ]
        assert "买家原话" not in caplog.text and "文三路" not in caplog.text


def test_llm_error_reaching_the_router_is_logged_with_its_attributes(caplog):
    error = LLMError("大模型接口限流（HTTP 429）", status=429, kind="rate_limit")
    router, _ = _router(_failing(error))
    with caplog.at_level(logging.WARNING):
        assert router.route(make_context("还有货吗", W)).degraded
    assert _fallback_warnings(caplog) == [
        FALLBACK_HEAD + "LLMError（rate_limit，HTTP 429）：大模型接口限流（HTTP 429）"
    ]


def test_no_fallback_warning_when_the_judge_answers(caplog):
    provider = FakeDecisionProvider(router_answers("shipping_policy", 0.92), name="incloud_llm")
    router, _ = _router(provider)
    with caplog.at_level(logging.WARNING):
        d = router.route(make_context("什么时候发货", W))
    assert not d.degraded and _fallback_warnings(caplog) == []
