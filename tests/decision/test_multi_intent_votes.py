"""多问题分条：大模型投票在同一个请求里给出第二个问题类型（§6.4 自一致性投票、§6.5、§6.6 规则 5）。

second_intent 是跟在 multi_intent 后面的追问：只问投票制的大模型判断层（一票只能选一个 intent，
认不出第二个类型）；规则、分类器、Jev 的 intent 概率本来就分摊到各个问题类型上，不问。
"""

from __future__ import annotations

import json

import pytest

import rodit.core.questions as questions_module
from rodit.core.decision import ChoiceAnswer, ChoiceQuestion, DecisionRequest, NoulQuestion
from rodit.core.enums import Intent
from rodit.core.errors import DecisionError
from rodit.core.ports import LLMRequest, LLMResponse
from rodit.core.questions import (
    NO_SECOND_INTENT,
    Q_EMOTION,
    Q_INTENT,
    Q_MULTI_INTENT,
    Q_NEED_HUMAN,
    Q_SECOND_INTENT,
    QA_QUESTIONS,
    QA_QUESTIONS_VER,
    ROUTER_MODEL,
    ROUTER_QUESTIONS,
    ROUTER_QUESTIONS_VER,
    ROUTER_VOTE_QUESTIONS,
    followup_questions,
    questions_version,
    with_followups,
)
from rodit.core.states import build_router_state
from rodit.decision import ClassifierProvider, InCloudLLMProvider, RuleProvider
from rodit.decision.voting import mention_probabilities
from rodit.testing.fakes import FakeLLMClient, make_context
from rodit.testing.fixtures import build_demo_world

W = build_demo_world()
TWO_QUESTIONS = "165 110斤穿什么码？什么时候发货？"

# 合并 #2（502379f）时的版本号：路由问题加了追问要变，质检问题不能变
BASE_ROUTER_QUESTIONS_VER = "f0d8e020f300"
BASE_QA_QUESTIONS_VER = "3af232eb45d2"


def _request(text: str = TWO_QUESTIONS, questions=None) -> DecisionRequest:
    ctx = make_context(text, W)
    return DecisionRequest(
        model=ROUTER_MODEL,
        state=build_router_state(ctx),
        questions=questions if questions is not None else ROUTER_VOTE_QUESTIONS,
    )


def _vote(intent: str, second: str | None = None, *, multi: bool = False) -> dict:
    out = {
        Q_INTENT: {"choice": intent},
        Q_NEED_HUMAN: {"answer": "否", "p": 0.05},
        Q_EMOTION: {"score": 0},
        Q_MULTI_INTENT: {"answer": "是", "p": 0.9} if multi else {"answer": "否", "p": 0.1},
    }
    if second is not None:
        out[Q_SECOND_INTENT] = {"choice": second}
    return out


def _provider(llm, samples: int = 3, **kw) -> InCloudLLMProvider:
    kw.setdefault("temperature", 0.7 if samples > 1 else 0.0)
    return InCloudLLMProvider(llm, samples=samples, purpose="route", **kw)


# ---------------------------------------------------------------- 问题定义与版本（§6.5、§13.5）


def test_second_intent_is_a_followup_of_intent_and_multi_intent():
    followups = followup_questions(ROUTER_QUESTIONS)
    assert list(followups) == [Q_SECOND_INTENT]
    second = followups[Q_SECOND_INTENT]
    assert isinstance(second, ChoiceQuestion)
    # 选项 = intent 的全部选项 + 无（只有一个问题）
    assert second.options == [i.value for i in Intent] + [NO_SECOND_INTENT]
    assert "multi_intent" in second.instructions and "intent" in second.instructions
    # 投票制判断层的路由请求：四个必答问题在前，追问排最后（紧跟 multi_intent）
    assert list(ROUTER_VOTE_QUESTIONS) == [*ROUTER_QUESTIONS, Q_SECOND_INTENT]
    assert len(ROUTER_VOTE_QUESTIONS) <= 64  # Jev 单次请求的问题上限
    assert len(ROUTER_QUESTIONS) <= 64
    # 没有 intent 或 multi_intent 就没有追问；质检问题不受影响
    assert followup_questions({Q_INTENT: ROUTER_QUESTIONS[Q_INTENT]}) == {}
    assert followup_questions({Q_MULTI_INTENT: ROUTER_QUESTIONS[Q_MULTI_INTENT]}) == {}
    assert followup_questions(QA_QUESTIONS) == {}
    assert with_followups(QA_QUESTIONS) == QA_QUESTIONS


def test_followup_options_follow_the_asked_intent_options():
    narrow = ChoiceQuestion(
        instructions="主要问题", criteria={"stock": None, "size_advice": "尺码"}
    )
    qs = {Q_INTENT: narrow, Q_MULTI_INTENT: NoulQuestion(instructions="多个问题？")}
    second = followup_questions(qs)[Q_SECOND_INTENT]
    assert isinstance(second, ChoiceQuestion)
    assert second.options == ["stock", "size_advice", NO_SECOND_INTENT]
    # 调用方自己给了 second_intent：照用，不覆盖
    custom = ChoiceQuestion(instructions="自定义", criteria={"stock": None, NO_SECOND_INTENT: None})
    assert with_followups({**qs, Q_SECOND_INTENT: custom})[Q_SECOND_INTENT] == custom
    # intent 的选项里已经有"无"：没法区分，不追问
    clash = ChoiceQuestion(
        instructions="主要问题", criteria={"stock": None, NO_SECOND_INTENT: None}
    )
    assert followup_questions({**qs, Q_INTENT: clash}) == {}


def test_questions_version_covers_the_followup(monkeypatch):
    assert ROUTER_QUESTIONS_VER == questions_version(ROUTER_QUESTIONS)
    assert ROUTER_QUESTIONS_VER == questions_version(ROUTER_VOTE_QUESTIONS)
    assert ROUTER_QUESTIONS_VER != BASE_ROUTER_QUESTIONS_VER  # 加了追问：版本号变了
    assert QA_QUESTIONS_VER == BASE_QA_QUESTIONS_VER == questions_version(QA_QUESTIONS)
    # 追问的文字改了，路由问题的版本号跟着变（decision_log.questions_ver，§13.5 先跑回归）
    monkeypatch.setattr(questions_module, "SECOND_INTENT_INSTRUCTIONS", "改过的说明")
    assert questions_version(ROUTER_QUESTIONS) != ROUTER_QUESTIONS_VER
    assert questions_version(QA_QUESTIONS) == QA_QUESTIONS_VER


# ---------------------------------------------------------------- 一票一个请求（§6.4）


def test_one_request_per_vote_carries_the_followup():
    llm = FakeLLMClient(lambda r: _vote("size_advice", "shipping_policy", multi=True))
    resp = _provider(llm).decide(_request())
    assert len(llm.calls) == 3  # 投票次数，没有额外往返
    for call in llm.calls:
        user = call.messages[1].content
        assert user.count("### second_intent（单选题）") == 1
        assert f"- {NO_SECOND_INTENT}：" in user
        assert '"second_intent": {"choice": "<选项键>"}' in user
    assert set(resp.answers) == set(ROUTER_VOTE_QUESTIONS)
    second = resp.choice(Q_SECOND_INTENT)
    assert (second.choice, second.confidence) == ("shipping_policy", 1.0)
    assert second.probabilities == {"shipping_policy": 1.0}
    # 主要问题的概率不受追问影响（规则 1、6、7 和校准照旧看它）
    intent = resp.choice(Q_INTENT)
    assert (intent.choice, intent.probabilities) == ("size_advice", {"size_advice": 1.0})
    assert resp.noul(Q_MULTI_INTENT) == 1.0


def test_single_question_answers_none():
    llm = FakeLLMClient(lambda r: _vote("size_advice", NO_SECOND_INTENT))
    resp = _provider(llm).decide(_request("165 110斤穿什么码"))
    second = resp.choice(Q_SECOND_INTENT)
    assert (second.choice, second.probabilities) == (NO_SECOND_INTENT, {NO_SECOND_INTENT: 1.0})
    assert mention_probabilities(resp.choice(Q_INTENT), second) == {"size_advice": 1.0}


def test_missing_or_invalid_second_intent_counts_as_none_not_as_failure():
    """老格式的输出（没有 second_intent）、选了不存在的选项：都按"无"计，不让整次路由降级。"""
    llm = FakeLLMClient(
        responses=[
            _vote("size_advice", multi=True),  # 没答追问
            _vote("size_advice", "not_an_intent", multi=True),
            {**_vote("size_advice", multi=True), Q_SECOND_INTENT: {"choice": None}},
        ]
    )
    resp = _provider(llm, min_votes=2).decide(_request())
    assert resp.choice(Q_SECOND_INTENT).probabilities == {NO_SECOND_INTENT: 1.0}
    assert resp.choice(Q_INTENT).choice == "size_advice"


def test_votes_are_paired_within_each_sample():
    """同一票里：追问和主要问题相同、或这一票说只有一个问题（multi_intent 否）→ 这一票按"无"计。"""
    llm = FakeLLMClient(
        responses=[
            _vote("size_advice", "size_advice", multi=True),  # 和主要问题相同：不算第二个问题
            _vote("size_advice", "shipping_policy", multi=False),  # 自相矛盾：以 multi_intent 为准
            _vote("size_advice", "shipping_policy", multi=True),
        ]
    )
    resp = _provider(llm).decide(_request())
    second = resp.choice(Q_SECOND_INTENT)
    assert second.probabilities == pytest.approx(
        {NO_SECOND_INTENT: 2 / 3, "shipping_policy": 1 / 3}
    )
    assert second.choice == NO_SECOND_INTENT


def test_mixed_votes_give_mention_shares():
    llm = FakeLLMClient(
        responses=[
            _vote("size_advice", "shipping_policy", multi=True),
            _vote("shipping_policy", "size_advice", multi=True),
            _vote("size_advice", NO_SECOND_INTENT, multi=False),
        ]
    )
    resp = _provider(llm).decide(_request())
    intent, second = resp.choice(Q_INTENT), resp.choice(Q_SECOND_INTENT)
    assert intent.probabilities == pytest.approx({"size_advice": 2 / 3, "shipping_policy": 1 / 3})
    assert second.probabilities == pytest.approx(
        {"size_advice": 1 / 3, "shipping_policy": 1 / 3, NO_SECOND_INTENT: 1 / 3}
    )
    # 每个类型"被问到"的票数占比：同一票里主要问题和追问不会相同，直接相加
    assert mention_probabilities(intent, second) == pytest.approx(
        {"size_advice": 1.0, "shipping_policy": 2 / 3}
    )


def test_samples_without_a_valid_main_intent_do_not_vote_on_the_followup():
    broken = _vote("size_advice", "shipping_policy", multi=True)
    broken[Q_INTENT] = {"choice": "not_an_intent"}
    llm = FakeLLMClient(
        responses=[
            broken,
            _vote("size_advice", "shipping_policy", multi=True),
            _vote("size_advice", NO_SECOND_INTENT, multi=True),
        ]
    )
    resp = _provider(llm).decide(_request())
    assert resp.choice(Q_SECOND_INTENT).probabilities == pytest.approx(
        {"shipping_policy": 0.5, NO_SECOND_INTENT: 0.5}
    )


def test_all_samples_failing_still_raises():
    from rodit.core.errors import LLMError

    llm = FakeLLMClient(lambda r: LLMError("限流"))
    with pytest.raises(DecisionError):
        _provider(llm).decide(_request())


def test_single_sample_uses_its_own_followup():
    llm = FakeLLMClient(lambda r: _vote("size_advice", "shipping_policy", multi=True))
    resp = _provider(llm, samples=1).decide(_request())
    assert resp.choice(Q_SECOND_INTENT).probabilities == {"shipping_policy": 1.0}
    assert resp.noul(Q_MULTI_INTENT) == pytest.approx(0.9)  # 只采一次：用自报的 p


# ---------------------------------------------------------------- 选项概率（logprobs）归属


def test_logprobs_are_requested_for_the_first_choice_question_only():
    def handler(r: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=json.dumps(_vote("size_advice", "shipping_policy", multi=True)),
            logprobs={"size_advice": 0.7, "product_spec": 0.3},
        )

    llm = FakeLLMClient(handler)
    resp = _provider(llm, samples=1).decide(_request())
    assert llm.calls[0].want_logprobs is True  # 输出的第一个问题是 intent（单选）
    assert resp.choice(Q_INTENT).probabilities == pytest.approx(
        {"size_advice": 0.7, "product_spec": 0.3}
    )
    # 追问和 intent 同一套选项：第一个 token 的概率不能记到它头上，它只看票
    assert resp.choice(Q_SECOND_INTENT).probabilities == {"shipping_policy": 1.0}


def test_logprobs_are_not_requested_when_they_cannot_be_attributed():
    qs = {
        "q_yes": NoulQuestion(instructions="是否？"),
        "q_a": ChoiceQuestion(instructions="选一个", criteria={"a": None, "b": None}),
        "q_b": ChoiceQuestion(instructions="再选一个", criteria={"a": None, "b": None}),
    }
    llm = FakeLLMClient(
        lambda r: LLMResponse(
            text=json.dumps({"q_yes": "是", "q_a": "a", "q_b": "b"}), logprobs={"b": 0.9, "a": 0.1}
        )
    )
    resp = _provider(llm, samples=1).decide(
        DecisionRequest(model="m", state={"synthetic": True}, questions=qs)
    )
    assert llm.calls[0].want_logprobs is False  # 两个单选题、第一个问题又不是单选：归不了
    assert resp.choice("q_a").probabilities == {"a": 1.0}  # 只看票
    assert resp.choice("q_b").probabilities == {"b": 1.0}
    # 只有一个单选题：照旧读（选项名对得上才用）
    single = FakeLLMClient(lambda r: {"q_yes": "是", "q_a": "a"})
    _provider(single, samples=1).decide(
        DecisionRequest(
            model="m", state={"synthetic": True}, questions={k: qs[k] for k in ("q_yes", "q_a")}
        )
    )
    assert single.calls[0].want_logprobs is True


def test_plain_router_request_is_unchanged():
    """直接问四个必答问题（不带追问）：和以前一样，不多答、不多问。"""
    llm = FakeLLMClient(lambda r: _vote("size_advice", "shipping_policy", multi=True))
    resp = _provider(llm).decide(_request(questions=ROUTER_QUESTIONS))
    assert set(resp.answers) == set(ROUTER_QUESTIONS)
    assert "second_intent" not in llm.calls[0].messages[1].content
    assert llm.calls[0].want_logprobs is True


# ---------------------------------------------------------------- 合并：被问到的概率


def _choice(probs: dict[str, float]) -> ChoiceAnswer:
    top = max(probs, key=lambda k: probs[k])
    return ChoiceAnswer(choice=top, confidence=probs[top], probabilities=probs)


def test_mention_probabilities_add_and_cap():
    intent = _choice({"size_advice": 0.6, "stock": 0.4})
    second = _choice({"stock": 0.8, "shipping_policy": 0.1, NO_SECOND_INTENT: 0.1})
    assert mention_probabilities(intent, second) == pytest.approx(
        {"size_advice": 0.6, "stock": 1.0, "shipping_policy": 0.1}
    )
    assert mention_probabilities(intent, None) == {"size_advice": 0.6, "stock": 0.4}
    assert mention_probabilities(None, second) == pytest.approx(
        {"stock": 0.8, "shipping_policy": 0.1}
    )
    assert mention_probabilities(None, None) == {}
    # probabilities 为空时按 choice / confidence
    bare = ChoiceAnswer(choice="stock", confidence=0.9)
    assert mention_probabilities(bare, ChoiceAnswer(choice=NO_SECOND_INTENT, confidence=1.0)) == {
        "stock": 0.9
    }


# ---------------------------------------------------------------- 规则、分类器保持兼容


def test_rule_and_classifier_providers_keep_their_multi_intent_probabilities():
    """规则、分类器不回答追问：intent 概率本来就分摊到各个类型上（规则 5 直接用），和以前一样。"""
    rules = RuleProvider().decide(_request(questions=ROUTER_QUESTIONS))
    assert set(rules.answers) == set(ROUTER_QUESTIONS)
    probs = rules.choice(Q_INTENT).probabilities
    assert probs["size_advice"] >= 0.2 and probs["shipping_policy"] >= 0.2
    assert rules.noul(Q_MULTI_INTENT) >= 0.7
    with pytest.raises(DecisionError, match="second_intent"):
        RuleProvider().decide(_request())  # 追问只问投票制的大模型判断层

    clf = ClassifierProvider(fallback=RuleProvider()).fit(
        [("什么时候发货", Intent.SHIPPING_POLICY), ("穿什么码", Intent.SIZE_ADVICE)],
        seed_examples=True,
    )
    resp = clf.decide(_request(questions=ROUTER_QUESTIONS))
    assert set(resp.answers) == set(ROUTER_QUESTIONS)
    assert resp.noul(Q_MULTI_INTENT) == rules.noul(Q_MULTI_INTENT)  # 兜底照旧由规则回答
