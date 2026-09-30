"""规则 1 因追问 second_intent 命中时，主要问题的提醒不丢（方案 §6.6 规则 1–4、
§5.7 投诉一律提醒人工）。

"东西质量太差我要投诉！你们找个人工来！要不加个微信私下退款？"：主要问题是投诉、要人工、情绪激动，
站外交易只出现在追问里。规则 1 照样给平台规则标准话术，但规则 2–4 会给的提醒（要人工、情绪激动、
投诉）和高风险标记要带上。合成数据，不联网。
"""

from __future__ import annotations

import itertools
import threading

import pytest

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.app.offline import EvidenceEchoLLM
from rodit.core.config import RoutingConfig
from rodit.core.decision import ChoiceAnswer, DecisionResponse
from rodit.core.enums import Intent, Mode, RouteAction
from rodit.core.questions import NO_SECOND_INTENT, Q_SECOND_INTENT, ROUTER_MODEL
from rodit.routing import ALERT_COMPLAINT, RoutingPolicy
from rodit.routing.policy import ALERT_EMOTION, ALERT_NEED_HUMAN
from rodit.testing.fakes import make_context, router_answers, scripted_judge_llm
from rodit.testing.fixtures import build_demo_world

W = build_demo_world()
A = RouteAction
OFF_2_OF_3 = {"off_platform": 2 / 3, NO_SECOND_INTENT: 1 / 3}


def _resp(
    intent: str,
    *,
    need_human: float = 0.0,
    emotion: float = 0.0,
    second: dict[str, float] | None = None,
) -> DecisionResponse:
    answers = router_answers(intent, 1.0, need_human=need_human, emotion=emotion, multi_intent=1.0)
    if second is not None:
        top = max(second, key=lambda k: second[k])
        answers[Q_SECOND_INTENT] = ChoiceAnswer(
            choice=top, confidence=second[top], probabilities=second
        )
    return DecisionResponse(model=ROUTER_MODEL, answers=answers)


def _apply(resp: DecisionResponse, mode: Mode = Mode.ASSIST):
    ctx = make_context("x", W)
    return RoutingPolicy(RoutingConfig()).apply(
        resp,
        mode=mode,
        shop_config=ctx.config,
        degraded=False,
        provider="incloud_llm",
        vote_samples=3,
    )


@pytest.mark.parametrize("mode", [Mode.ASSIST, Mode.SHADOW, Mode.AUTO])
@pytest.mark.parametrize(
    ("intent", "signals", "alerts", "high_risk"),
    [
        ("complaint", {}, [ALERT_COMPLAINT], True),
        ("complaint", {"need_human": 1.0, "emotion": 4.0}, [ALERT_NEED_HUMAN, ALERT_EMOTION], True),
        ("logistics_track", {"need_human": 1.0}, [ALERT_NEED_HUMAN], False),
        ("quality_issue", {"emotion": 3.3}, [ALERT_EMOTION], True),  # 默认配置里是高风险类型
    ],
)
def test_an_off_platform_aside_keeps_the_escalation_alerts(
    mode, intent, signals, alerts, high_risk
):
    resp = _resp(intent, second=OFF_2_OF_3, **signals)
    d = _apply(resp, mode)
    assert (d.action, d.rule, d.intent) == (A.OFF_PLATFORM_SCRIPT, 1, Intent.OFF_PLATFORM)
    assert d.alerts == alerts
    assert d.high_risk is high_risk
    # 追问没提到站外交易时，同样的信号照常走规则 2–4 并给同样的提醒
    plain = _apply(_resp(intent, second={NO_SECOND_INTENT: 1.0}, **signals), mode)
    assert plain.rule in (2, 3, 4) and set(plain.alerts) <= set(alerts) | {ALERT_COMPLAINT}
    assert plain.alerts[0] == alerts[0]


def test_off_platform_as_the_primary_question_is_unchanged():
    """主要问题就是站外交易：规则 1 照旧，不带别的提醒、不标高风险。"""
    d = _apply(_resp("off_platform", need_human=1.0))
    assert (d.action, d.rule, d.alerts, d.high_risk) == (A.OFF_PLATFORM_SCRIPT, 1, [], False)
    # 追问提到、主要问题没有升级信号：没有额外提醒（高风险标记按主要问题：价格是高风险类型）
    d = _apply(_resp("price_promo", second=OFF_2_OF_3))
    assert (d.action, d.rule, d.alerts, d.high_risk) == (A.OFF_PLATFORM_SCRIPT, 1, [], True)
    d = _apply(_resp("size_advice", second=OFF_2_OF_3))
    assert (d.action, d.rule, d.alerts, d.high_risk) == (A.OFF_PLATFORM_SCRIPT, 1, [], False)


def _run(votes: list[dict]):
    """演示单元 + 投票制大模型判断层（按顺序轮流给出 votes 里的答案）。"""
    cycle = itertools.cycle(votes)
    lock = threading.Lock()

    def script(_request):
        with lock:
            return next(cycle)

    judge = scripted_judge_llm(route=script)
    echo = EvidenceEchoLLM()

    class _LLM:
        model = "synthetic-judge"

        def complete(self, request):
            return (
                echo.complete(request) if request.purpose == "generate" else judge.complete(request)
            )

    service = AssistService(build_demo_unit(llm=_LLM()))
    text = "东西质量太差我要投诉！你们找个人工来！要不加个微信私下退款？"
    return service.handle_event("B", taobao_event(text, msg_id="m_rule1"))


def test_end_to_end_the_agent_still_sees_the_need_human_alert():
    yes = {"answer": "是", "p": 0.9}
    base = {
        "intent": {"choice": "complaint"},
        "need_human": yes,
        "emotion": {"score": 4},
        "multi_intent": yes,
    }
    off = {**base, "second_intent": {"choice": "off_platform"}}
    none = {**base, "second_intent": {"choice": NO_SECOND_INTENT}}
    result = _run([off, off, none])
    assert result is not None and result.route is not None
    assert result.route.action == A.OFF_PLATFORM_SCRIPT and result.route.rule == 1
    assert ALERT_NEED_HUMAN in result.alerts and ALERT_EMOTION in result.alerts
    assert result.route.high_risk is True
