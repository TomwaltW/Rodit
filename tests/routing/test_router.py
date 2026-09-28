"""DecisionRouter：§6.5 请求、降级到规则、decision_log、个人信息与店铺隔离（§6.6、§12.2、§9.2）。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import RoutingConfig
from rodit.core.context import current_shop_id
from rodit.core.decision import (
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
)
from rodit.core.enums import DecisionStage, Intent, Mode, Role, RouteAction
from rodit.core.errors import ComplianceError, DecisionError, LLMError, LLMTimeout
from rodit.core.ports import LLMRequest, Router
from rodit.core.questions import (
    Q_INTENT,
    Q_NEED_HUMAN,
    ROUTER_MODEL,
    ROUTER_QUESTIONS,
    ROUTER_QUESTIONS_VER,
)
from rodit.decision import InCloudLLMProvider, JevConditions, JevProvider, RuleProvider
from rodit.routing import NO_PROVIDER, DecisionRouter, RoutingPolicy
from rodit.testing.fakes import (
    FakeClock,
    FakeDecisionProvider,
    FakeLLMClient,
    make_context,
    router_answers,
)
from rodit.testing.fixtures import BUYER, CONV, SHOP_A, SHOP_B, build_demo_world

W = build_demo_world()
A = RouteAction


def _router(provider, fallback=None, *, clock=None, audit=None, **kw):
    audit = audit if audit is not None else InMemoryAuditSink()
    router = DecisionRouter(
        provider,
        fallback or RuleProvider(),
        RoutingPolicy(RoutingConfig()),
        audit,
        clock or FakeClock(W.now),
        **kw,
    )
    return router, audit


def _llm_vote(intent: str = "size_advice") -> dict:
    return {
        "intent": {"choice": intent},
        "need_human": {"answer": "否", "p": 0.05},
        "emotion": {"score": 0},
        "multi_intent": {"answer": "否", "p": 0.05},
    }


# ---------------------------------------------------------------- 正常路由与 §6.5 请求


def test_route_builds_the_doc_request_and_logs_one_decision():
    provider = FakeDecisionProvider(router_answers("size_advice", 0.92), name="incloud_llm")
    router, audit = _router(provider)
    ctx = make_context(
        "165 110斤穿什么码",
        W,
        item_id="i_555",
        history=[(Role.BUYER, "在吗"), (Role.AGENT, "亲，在的")],
    )
    d = router.route(ctx)
    assert isinstance(router, Router)
    assert (d.action, d.rule, d.intent, d.provider, d.degraded) == (
        A.ANSWER,
        6,
        Intent.SIZE_ADVICE,
        "incloud_llm",
        False,
    )

    (request,) = provider.requests
    assert request.model == ROUTER_MODEL == "router-v1"
    assert request.questions == ROUTER_QUESTIONS
    assert set(request.state) == {"platform", "messages", "item_card", "order_card"}
    assert request.state["platform"] == "taobao"
    assert request.state["messages"] == [
        {"role": "buyer", "text": "在吗"},
        {"role": "agent", "text": "亲,在的"},  # 过 mask_pii 时做了 NFKC 归一
        {"role": "buyer", "text": "165 110斤穿什么码"},
    ]
    assert request.state["item_card"] == {
        "item_id": "i_555",
        "category": "女装/连衣裙",
        "title": "法式碎花连衣裙",
    }
    dumped = request.model_dump(mode="json")
    assert set(dumped["questions"]["intent"]["criteria"]) == {i.value for i in Intent}

    (entry,) = audit.decisions
    assert entry.shop_id == SHOP_A and entry.conversation_id == CONV
    assert entry.stage == DecisionStage.ROUTE
    assert entry.provider == "incloud_llm"
    assert entry.questions_ver == ROUTER_QUESTIONS_VER
    assert entry.answers == d.answers.model_dump(mode="json")  # type: ignore[union-attr]
    assert entry.route_result == "answer"
    assert entry.latency_ms == 0 and entry.created_at == W.now
    assert audit.events == []  # audit_log 由引擎写


def test_history_rounds_is_passed_to_the_state_builder():
    provider = FakeDecisionProvider(router_answers("stock"))
    router, _ = _router(provider, history_rounds=1)
    history = [(Role.BUYER, "第一句"), (Role.BUYER, "第二句"), (Role.AGENT, "回复")]
    router.route(make_context("有货吗", W, history=history))
    texts = [m["text"] for m in provider.requests[0].state["messages"]]
    assert texts == ["第二句", "回复", "有货吗"]


def test_latency_is_measured_with_the_injected_clock():
    clock = FakeClock(W.now)

    def answers(request: DecisionRequest):
        clock.advance(milliseconds=250)
        return router_answers("stock")

    router, audit = _router(FakeDecisionProvider(answers), clock=clock)
    router.route(make_context("有货吗", W))
    assert audit.decisions[0].latency_ms == 250
    assert audit.decisions[0].created_at == W.now + timedelta(milliseconds=250)


def test_mode_and_shop_config_come_from_the_context():
    provider = FakeDecisionProvider(router_answers("greeting", 0.97))
    router, _ = _router(provider)
    auto = make_context("在吗", W, mode=Mode.AUTO, config_update={"auto_reply_enabled": True})
    assert router.route(auto).auto_eligible is True
    assert router.route(make_context("在吗", W, mode=Mode.AUTO)).auto_eligible is False
    closed = make_context("在吗", W, config_update={"disabled_intents": ["greeting"]})
    assert router.route(closed).action == A.NO_SUGGESTION


# ---------------------------------------------------------------- 降级（§12.2）


@pytest.mark.parametrize(
    "error", [DecisionError("判断层不可用"), LLMError("大模型报错"), LLMTimeout("超时")]
)
def test_judge_failure_falls_back_to_rules(error):
    provider = FakeDecisionProvider(router_answers("stock"), name="incloud_llm", fail=error)
    router, audit = _router(provider)
    d = router.route(make_context("还有货吗", W))
    assert (d.action, d.intent, d.degraded, d.provider) == (
        A.ANSWER,
        Intent.STOCK,
        True,
        "rules",
    )
    assert d.auto_eligible is False
    (entry,) = audit.decisions
    assert entry.provider == "rules" and entry.route_result == "answer"


def test_incomplete_answers_count_as_judge_failure():
    partial = {Q_NEED_HUMAN: NoulAnswer(noul=0.1)}

    class Sloppy:
        name = "sloppy"
        accepts_buyer_data = True

        def decide(self, request: DecisionRequest) -> DecisionResponse:
            return DecisionResponse(model=request.model, answers=partial)

    router, _ = _router(Sloppy())
    d = router.route(make_context("还有货吗", W))
    assert d.degraded and d.provider == "rules"


def test_degraded_high_risk_gets_no_suggestion():
    failing = FakeDecisionProvider(router_answers("other"), fail=DecisionError("x"))
    router, _ = _router(failing)
    d = router.route(make_context("能便宜点吗", W))
    assert (d.action, d.rule, d.intent, d.degraded) == (
        A.NO_SUGGESTION,
        4,
        Intent.PRICE_PROMO,
        True,
    )


def test_degraded_rules_still_catch_off_platform_and_complaints():
    failing = FakeDecisionProvider(router_answers("other"), fail=LLMError("x"))
    router, _ = _router(failing)
    assert router.route(make_context("加个微信吧，转账便宜点", W)).action == A.OFF_PLATFORM_SCRIPT
    assert router.route(make_context("我要投诉你们", W)).action == A.HUMAN_PRIORITY
    assert router.route(make_context("嗯", W)).action == A.NO_SUGGESTION


def test_both_failing_returns_no_suggestion_without_raising():
    failing = FakeDecisionProvider(router_answers("other"), fail=DecisionError("x"))
    broken_rules = FakeDecisionProvider(router_answers("other"), name="rules", fail=LLMError("y"))
    router, audit = _router(failing, broken_rules)
    d = router.route(make_context("还有货吗", W))
    assert (d.action, d.rule, d.intent, d.degraded, d.provider) == (
        A.NO_SUGGESTION,
        8,
        Intent.OTHER,
        True,
        NO_PROVIDER,
    )
    (entry,) = audit.decisions
    assert (
        entry.provider == "none" and entry.answers == {} and entry.route_result == "no_suggestion"
    )


# ---------------------------------------------------------------- 合规（§6.2、§6.3）


def test_router_refuses_providers_that_cannot_take_buyer_data():
    no_buyer = FakeDecisionProvider(router_answers("other"), accepts_buyer_data=False)
    offline_jev = JevProvider(lambda p: {}, JevConditions(), purpose="offline_eval")
    for bad in (no_buyer, offline_jev):
        with pytest.raises(ComplianceError):
            _router(bad)
        with pytest.raises(ComplianceError):
            _router(RuleProvider(), bad)


def test_prompts_never_contain_buyer_pii_end_to_end():
    llm = FakeLLMClient(lambda r: _llm_vote("modify_order"))
    provider = InCloudLLMProvider(llm, samples=3, temperature=0.7, purpose="route")
    router, audit = _router(provider)
    text = "我的手机 13812345678，收货地址改成杭州市西湖区文三路 188 号 3 单元 501 室"
    ctx = make_context(
        text, W, order_id="o_1001", history=[(Role.BUYER, "我微信号 abc_12345 加一下")]
    )
    d = router.route(ctx)
    assert d.action == A.HIGH_RISK_ANSWER and d.provider == "incloud_llm"
    prompts = llm.all_prompt_text()
    assert len(llm.calls) == 3
    for secret in ("13812345678", "文三路", "188 号", "abc_12345", BUYER):
        assert secret not in prompts
    assert "o_1001" in prompts  # 订单号保留，回答要用
    assert all(c.purpose == "route" and c.json_mode for c in llm.calls)
    log_text = str(audit.decisions[0].model_dump(mode="json"))
    assert "13812345678" not in log_text and "文三路" not in log_text


# ---------------------------------------------------------------- 店铺隔离与成本归属（§9.2、§12.4）


def test_decision_log_and_llm_cost_are_attributed_to_the_right_shop():
    seen: list[str | None] = []

    def handler(r: LLMRequest) -> dict:
        seen.append(current_shop_id())
        return _llm_vote("stock")

    llm = FakeLLMClient(handler)
    provider = InCloudLLMProvider(llm, samples=3, temperature=0.7, purpose="route")
    router, audit = _router(provider)
    router.route(make_context("还有货吗", W, shop_id=SHOP_B))
    assert seen == [SHOP_B] * 3
    assert {c.shop_id for c in llm.calls} == {SHOP_B}
    assert current_shop_id() is None  # 路由结束后不残留
    assert [e.shop_id for e in audit.list_decisions(SHOP_B)] == [SHOP_B]
    assert audit.list_decisions(SHOP_A) == []


def test_real_llm_votes_flow_into_the_policy():
    llm = FakeLLMClient(
        responses=[_llm_vote("size_advice"), _llm_vote("product_spec"), _llm_vote("size_advice")]
    )
    provider = InCloudLLMProvider(llm, samples=3, temperature=0.7, purpose="route")
    router, audit = _router(provider)
    d = router.route(make_context("165 110斤穿什么码", W))
    assert d.action == A.ANSWER and d.intent == Intent.SIZE_ADVICE
    assert d.intent_confidence == pytest.approx(2 / 3, abs=1e-3)
    logged = audit.decisions[0].answers
    assert logged["model"] == ROUTER_MODEL
    assert logged["answers"][Q_INTENT]["choice"] == "size_advice"
