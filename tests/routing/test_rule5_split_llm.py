"""规则 5 按问题分条：大模型投票给出第二个问题类型后能分条（方案 §6.4、§6.6 规则 5、§13.5）。

以前一票只能选一个 intent，各票都选同一个"主要"类型，规则 5 分不出第二条，只能往下走到规则 6
并提醒客服可能漏答（ALERT_MULTI_UNSPLIT）。现在同一个请求里追问 second_intent，
multi_intent 达到阈值时把主要问题和追问的票数占比合起来分条。
"""

from __future__ import annotations

import pytest

from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import DecisionConfig, RoutingConfig
from rodit.core.decision import ChoiceAnswer, DecisionResponse
from rodit.core.enums import Intent, Mode, RouteAction
from rodit.core.errors import LLMError
from rodit.core.questions import (
    NO_SECOND_INTENT,
    Q_SECOND_INTENT,
    QA_QUESTIONS_VER,
    ROUTER_MODEL,
    ROUTER_QUESTIONS,
    ROUTER_QUESTIONS_VER,
    ROUTER_VOTE_QUESTIONS,
)
from rodit.decision import RuleProvider, build_llm_provider
from rodit.routing import (
    ALERT_COMPLAINT,
    ALERT_MULTI_UNSPLIT,
    DecisionRouter,
    RoutingPolicy,
)
from rodit.testing.fakes import (
    FakeClock,
    FakeDecisionProvider,
    FakeLLMClient,
    judge_questions,
    make_context,
    router_answers,
    scripted_judge_llm,
)
from rodit.testing.fixtures import SHOP_A, build_demo_world

W = build_demo_world()
A = RouteAction
TWO_QUESTIONS = "165 110斤穿什么码？什么时候发货？"
YES = {"answer": "是", "p": 0.9}
NO = {"answer": "否", "p": 0.1}


class _SpyProvider:
    """记录收到的请求再转给真正的实现（看路由器问了哪些问题）。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = inner.name
        self.accepts_buyer_data = True
        self.requests = []

    def decide(self, request):
        self.requests.append(request)
        return self.inner.decide(request)


def _router(provider, *, fallback=None, config: RoutingConfig | None = None):
    audit = InMemoryAuditSink()
    router = DecisionRouter(
        provider,
        fallback or RuleProvider(),
        RoutingPolicy(config or RoutingConfig()),
        audit,
        FakeClock(W.now),
    )
    return router, audit


def _llm_router(script, *, samples: int = 3, config: RoutingConfig | None = None):
    llm = scripted_judge_llm(route=script)
    provider = build_llm_provider(DecisionConfig(vote_samples=samples), purpose="route", llm=llm)
    router, audit = _router(provider, config=config)
    return router, audit, llm


def _plain_llm_router(output: dict):
    """每票都原样输出 output 的大模型（模拟不理会追问、只答四个问题的老输出）。"""
    llm = FakeLLMClient(lambda r: {"need_human": NO, "emotion": {"score": 0}, **output})
    provider = build_llm_provider(DecisionConfig(), purpose="route", llm=llm)
    router, _ = _router(provider)
    return router


# ---------------------------------------------------------------- 验收 (a)–(c)


def test_llm_votes_with_a_second_intent_split_into_two_parts():
    """验收 (a)：3/3 票 intent=size_advice、second_intent=shipping_policy、multi_intent=是。"""
    script = {"intent": {"choice": "size_advice"}, "second_intent": {"choice": "shipping_policy"}}
    router, audit, llm = _llm_router({**script, "multi_intent": YES})
    d = router.route(make_context(TWO_QUESTIONS, W, item_id="i_555"))
    assert (d.action, d.rule) == (A.SPLIT_MULTI, 5)
    assert [p.intent for p in d.parts] == [Intent.SIZE_ADVICE, Intent.SHIPPING_POLICY]
    assert [p.confidence for p in d.parts] == [1.0, 1.0]
    assert ALERT_MULTI_UNSPLIT not in d.alerts and d.alerts == []
    assert (d.intent, d.intent_confidence, d.multi_intent) == (Intent.SIZE_ADVICE, 1.0, 1.0)
    assert d.degraded is False and d.provider == "incloud_llm"
    # 验收 (c)：一票一个请求，每个请求都带着追问
    calls = llm.calls_for("route")
    assert len(calls) == DecisionConfig().vote_samples == 3
    expected = {k: q.type for k, q in ROUTER_VOTE_QUESTIONS.items()}
    assert all(judge_questions(c) == expected for c in calls)
    # decision_log：新的问题版本号、追问的答案都留痕
    (entry,) = audit.decisions
    assert entry.questions_ver == ROUTER_QUESTIONS_VER
    assert entry.answers["answers"][Q_SECOND_INTENT]["choice"] == "shipping_policy"
    assert entry.route_result == "split_multi"


def test_before_the_followup_the_same_votes_could_not_be_split():
    """502379f 的行为（大模型不答追问）：规则 6 + 漏答提醒。追问答了才分条。"""
    router = _plain_llm_router({"intent": {"choice": "size_advice"}, "multi_intent": YES})
    d = router.route(make_context(TWO_QUESTIONS, W))
    assert (d.action, d.rule, d.parts) == (A.ANSWER, 6, [])
    assert ALERT_MULTI_UNSPLIT in d.alerts


def test_single_question_is_never_split():
    """验收 (b)：multi_intent=否、second_intent=无 → 不分条，和不追问时的结果一样。"""
    single = {"intent": {"choice": "size_advice"}, "multi_intent": NO}
    router, _, _ = _llm_router({**single, "second_intent": {"choice": NO_SECOND_INTENT}})
    d = router.route(make_context("165 110斤穿什么码", W))
    old = _plain_llm_router(single).route(make_context("165 110斤穿什么码", W))
    assert (d.action, d.rule, d.parts, d.alerts) == (A.ANSWER, 6, [], [])
    assert (d.action, d.rule, d.intent, d.intent_confidence, d.top_intents) == (
        old.action,
        old.rule,
        old.intent,
        old.intent_confidence,
        old.top_intents,
    )


@pytest.mark.parametrize("samples", [1, 3, 5])
def test_call_count_equals_vote_samples(samples):
    script = {
        "intent": {"choice": "size_advice"},
        "second_intent": {"choice": "shipping_policy"},
        "multi_intent": YES,
    }
    router, _, llm = _llm_router(script, samples=samples)
    d = router.route(make_context(TWO_QUESTIONS, W))
    assert len(llm.calls) == samples  # 没有为追问多跑一轮
    assert d.action == A.SPLIT_MULTI


def test_versions_and_question_counts():
    assert len(ROUTER_QUESTIONS) <= 64 and len(ROUTER_VOTE_QUESTIONS) <= 64
    assert ROUTER_QUESTIONS_VER != "f0d8e020f300"  # 502379f 的版本号
    assert QA_QUESTIONS_VER == "3af232eb45d2"  # 质检问题没动


# ---------------------------------------------------------------- 只有投票制判断层才追问


def test_other_judges_are_asked_only_the_plain_router_questions():
    """追问只问投票制的大模型判断层；别的实现（这里是假判断层）照旧只问四个必答问题。"""
    fake = FakeDecisionProvider(router_answers("size_advice"), name="incloud_llm")
    rules = _SpyProvider(RuleProvider())
    router, audit = _router(fake, fallback=rules)
    d = router.route(make_context(TWO_QUESTIONS, W))
    assert fake.requests[0].questions == ROUTER_QUESTIONS
    assert (d.action, d.provider, d.degraded) == (A.ANSWER, "incloud_llm", False)
    assert audit.decisions[0].questions_ver == ROUTER_QUESTIONS_VER
    assert rules.requests == []  # 判断层正常：不问兜底


def test_degraded_fallback_is_asked_the_plain_router_questions():
    llm = FakeLLMClient(lambda r: LLMError("限流"))
    provider = build_llm_provider(DecisionConfig(), purpose="route", llm=llm)
    rules = _SpyProvider(RuleProvider())
    router, audit = _router(provider, fallback=rules)
    d = router.route(make_context(TWO_QUESTIONS, W))
    assert (d.degraded, d.provider) == (True, "rules")
    (request,) = rules.requests
    assert request.questions == ROUTER_QUESTIONS  # 规则判断层不答追问
    assert Q_SECOND_INTENT in {q for c in llm.calls for q in judge_questions(c)}
    assert Q_SECOND_INTENT not in audit.decisions[0].answers["answers"]
    # 规则判断层自己的 intent 概率照旧能分条（§12.2 降级路由）
    assert (d.action, [p.intent for p in d.parts]) == (
        A.SPLIT_MULTI,
        [Intent.SIZE_ADVICE, Intent.SHIPPING_POLICY],
    )


def test_auto_mode_multi_question_still_transfers_to_a_human():
    script = {
        "intent": {"choice": "size_advice"},
        "second_intent": {"choice": "shipping_policy"},
        "multi_intent": YES,
    }
    router, _, _ = _llm_router(script)
    ctx = make_context(TWO_QUESTIONS, W, mode=Mode.AUTO, config_update={"auto_reply_enabled": True})
    d = router.route(ctx)
    assert (d.action, d.rule, d.parts) == (A.TRANSFER_HUMAN, 5, [])


# ---------------------------------------------------------------- 策略：合并只在规则 5 里做


def _resp(
    intent: dict[str, float], second: dict[str, float] | None, *, multi: float
) -> DecisionResponse:
    top = max(intent, key=lambda k: intent[k])
    answers = router_answers(top, intent[top], probabilities=intent, multi_intent=multi)
    if second is not None:
        s = max(second, key=lambda k: second[k])
        answers[Q_SECOND_INTENT] = ChoiceAnswer(
            choice=s, confidence=second[s], probabilities=second
        )
    return DecisionResponse(model=ROUTER_MODEL, answers=answers)


def _apply(resp: DecisionResponse, *, samples: int | None = 3, shop_update=None, degraded=False):
    ctx = make_context("x", W, config_update=shop_update)
    return RoutingPolicy(RoutingConfig()).apply(
        resp,
        mode=Mode.ASSIST,
        shop_config=ctx.config,
        degraded=degraded,
        provider="incloud_llm",
        vote_samples=samples,
    )


def test_below_the_multi_intent_threshold_the_followup_is_ignored():
    """只有 1/3 票说有多个问题：不分条，也不提醒漏答；主要问题的概率不被追问改动。"""
    below = {"shipping_policy": 1 / 3, NO_SECOND_INTENT: 2 / 3}
    resp = _resp({"size_advice": 1.0}, below, multi=1 / 3)
    d = _apply(resp)
    assert (d.action, d.rule, d.parts, d.alerts) == (A.ANSWER, 6, [], [])
    assert d.top_intents == [(Intent.SIZE_ADVICE, 1.0)]
    # 2/3 票（默认 0.7 换算到 3 票的粒度）：分条
    above = {"shipping_policy": 2 / 3, NO_SECOND_INTENT: 1 / 3}
    two_thirds = _resp({"size_advice": 1.0}, above, multi=2 / 3)
    split = _apply(two_thirds)
    assert split.action == A.SPLIT_MULTI
    assert [(p.intent, p.confidence) for p in split.parts] == [
        (Intent.SIZE_ADVICE, 1.0),
        (Intent.SHIPPING_POLICY, pytest.approx(2 / 3)),
    ]


def test_mention_shares_rank_the_parts():
    resp = _resp(
        {"shipping_policy": 2 / 3, "size_advice": 1 / 3},
        {"size_advice": 2 / 3, "stock": 1 / 3},
        multi=1.0,
    )
    d = _apply(resp)
    assert d.intent == Intent.SHIPPING_POLICY  # 主要问题不变
    assert [(p.intent, round(p.confidence, 6)) for p in d.parts] == [
        (Intent.SIZE_ADVICE, 1.0),
        (Intent.SHIPPING_POLICY, round(2 / 3, 6)),
        (Intent.STOCK, round(1 / 3, 6)),
    ]


def test_followup_that_cannot_stand_alone_still_alerts():
    # 追问是"其他"：分不出第二条 → 往下走，提醒漏答
    other = _apply(_resp({"size_advice": 1.0}, {"other": 1.0}, multi=1.0))
    assert (other.action, other.rule) == (A.ANSWER, 6)
    assert ALERT_MULTI_UNSPLIT in other.alerts
    # 追问是投诉：投诉不单独成条，提醒人工
    complaint = _apply(_resp({"size_advice": 1.0}, {"complaint": 1.0}, multi=1.0))
    assert complaint.parts == [] and ALERT_COMPLAINT in complaint.alerts
    assert ALERT_MULTI_UNSPLIT in complaint.alerts
    # 店铺关闭的类型不单独成条
    closed = _apply(
        _resp({"size_advice": 1.0}, {"shipping_policy": 1.0}, multi=1.0),
        shop_update={"disabled_intents": ["shipping_policy"]},
    )
    assert closed.parts == [] and ALERT_MULTI_UNSPLIT in closed.alerts
    # 降级时的高风险类型不单独成条（§12.2）
    risky = _apply(_resp({"size_advice": 1.0}, {"price_promo": 1.0}, multi=1.0), degraded=True)
    assert risky.parts == []


def test_followup_high_risk_part_gets_its_risk_hint():
    d = _apply(_resp({"size_advice": 1.0}, {"price_promo": 1.0}, multi=1.0))
    assert d.action == A.SPLIT_MULTI
    assert [p.intent for p in d.parts] == [Intent.SIZE_ADVICE, Intent.PRICE_PROMO]
    assert len(d.risk_hints) == 1


def test_shop_scope_of_the_logged_decision():
    script = {
        "intent": {"choice": "size_advice"},
        "multi_intent": NO,
        "second_intent": {"choice": NO_SECOND_INTENT},
    }
    router, audit, _ = _llm_router(script)
    router.route(make_context("165 110斤穿什么码", W))
    assert [e.shop_id for e in audit.decisions] == [SHOP_A]


def test_off_platform_as_the_second_question_still_triggers_rule_1():
    """ "这件多少钱？加个微信聊"：主要问题是价格，站外交易只出现在追问里——规则 1 优先，
    给平台规则标准话术，不把站外交易当成分条里的一条回答（§6.6 规则 1 防止违规）。"""
    resp = _resp({"price_promo": 1.0}, {"off_platform": 1.0}, multi=1.0)
    d = _apply(resp)
    assert (d.action, d.rule, d.intent) == (A.OFF_PLATFORM_SCRIPT, 1, Intent.OFF_PLATFORM)
    # 只有 1/3 票提到：不到 off_platform 阈值（0.6），照常分条或回答
    weak = _resp({"price_promo": 1.0}, {"off_platform": 1 / 3, NO_SECOND_INTENT: 2 / 3}, multi=1.0)
    assert _apply(weak).rule != 1
