"""能力开关关掉、模型输出不规范时，判断层和生成照样能用（方案 §6.4 投票、§5.4 输出解析、§12.2）。

- logprobs 关掉（或服务不返回）：路由的置信度按投票占比算，不因为缺 logprobs 降级；
  JSON 模式下第一个有内容的 token 是键名，开着 logprobs 也用不上——同样按票数占比；
- JSON 模式关掉：请求里没有 response_format，模型按提示词输出 JSON，前后带话、包在代码块里
  都能解析（判断层 decision.voting、生成 generation.parse 原有的宽容解析）；
- 思考过程放在正文开头（<think>…</think>，里面还可能有 JSON 草稿）：客户端去掉它，
  判断层和生成拿到的是后面的回答，不会把草稿当答案。
真实的 OpenAICompatibleClient + 假传输层，合成数据，不联网。
"""

from __future__ import annotations

import itertools
import json
import threading
from typing import Any

import pytest

from rodit.core.decision import DecisionRequest
from rodit.core.ports import LLMMessage, LLMRequest
from rodit.core.questions import (
    Q_ANSWERS_QUESTION,
    Q_EMOTION,
    Q_GROUNDED,
    Q_INTENT,
    Q_MULTI_INTENT,
    Q_NEED_HUMAN,
    Q_PLATFORM_VIOLATION,
    Q_PROMISE_RISK,
    Q_TONE,
    QA_MODEL,
    QA_QUESTIONS,
    ROUTER_MODEL,
    ROUTER_QUESTIONS,
)
from rodit.core.states import build_router_state
from rodit.decision.llm import InCloudLLMProvider
from rodit.decision.voting import parse_sample
from rodit.generation.parse import parse_generation_output
from rodit.llm import OpenAICompatibleClient
from rodit.testing.fakes import make_context
from rodit.testing.fixtures import build_demo_world

KEY = "sk-synthetic-compat-paths-0001"  # 合成密钥，只用于测试
BASE = "https://llm.example.invalid/v1"
W = build_demo_world()


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)


class Cycle:
    """线程安全地轮流回放内容（投票并发调用）；记下每个请求体。"""

    def __init__(self, contents: list[str], *, logprobs: Any = None) -> None:
        self._it = itertools.cycle(contents)
        self._lock = threading.Lock()
        self.logprobs = logprobs
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, url, headers, body, timeout_s):
        with self._lock:
            self.bodies.append(body)
            content = next(self._it)
        choice: dict[str, Any] = {"message": {"content": content}, "finish_reason": "stop"}
        if self.logprobs is not None:
            choice["logprobs"] = self.logprobs
        return 200, {"model": "served", "choices": [choice], "usage": {}}


def route_vote(intent: str) -> str:
    return json.dumps(
        {
            Q_INTENT: {"choice": intent},
            Q_NEED_HUMAN: {"answer": "否", "p": 0.1},
            Q_EMOTION: {"score": 0},
            Q_MULTI_INTENT: {"answer": "否", "p": 0.1},
        },
        ensure_ascii=False,
    )


def router_request() -> DecisionRequest:
    ctx = make_context("165 110斤穿什么码", W)
    return DecisionRequest(
        model=ROUTER_MODEL, state=build_router_state(ctx), questions=ROUTER_QUESTIONS
    )


def route(client: OpenAICompatibleClient) -> dict[str, float]:
    provider = InCloudLLMProvider(client, samples=3, temperature=0.7, purpose="route")
    answer = provider.decide(router_request()).answers[Q_INTENT]
    return dict(answer.probabilities)  # type: ignore[union-attr]


VOTES = [route_vote("size_advice"), route_vote("size_advice"), route_vote("product_spec")]
RATIOS = {"size_advice": pytest.approx(2 / 3, abs=1e-6), "product_spec": pytest.approx(1 / 3)}
# JSON 模式下真实服务返回的 logprobs：第一个有内容的 token 是键名 "intent"，对不上任何选项
KEY_NAME_LOGPROBS = {
    "content": [
        {"token": '{"', "logprob": 0.0, "top_logprobs": [{"token": '{"', "logprob": 0.0}]},
        {
            "token": "intent",
            "logprob": -0.01,
            "top_logprobs": [
                {"token": "intent", "logprob": -0.01},
                {"token": "need", "logprob": -5},
            ],
        },
    ]
}


# ================================================================ 路由：没有 logprobs 时按票数占比


def test_route_votes_without_logprobs_when_the_switch_is_off():
    t = Cycle(VOTES, logprobs=KEY_NAME_LOGPROBS)
    client = OpenAICompatibleClient(base_url=BASE, model="m", transport=t, logprobs=False)
    assert route(client) == RATIOS
    assert len(t.bodies) == 3
    assert all("logprobs" not in b and "top_logprobs" not in b for b in t.bodies)
    assert all(b["response_format"] == {"type": "json_object"} for b in t.bodies)


@pytest.mark.parametrize("returned", [None, KEY_NAME_LOGPROBS], ids=["none", "key-name-token"])
def test_route_votes_when_logprobs_are_requested_but_unusable(returned):
    """开着 logprobs：服务不返回，或 JSON 模式下第一个 token 是键名——都退回票数占比。"""
    t = Cycle(VOTES, logprobs=returned)
    client = OpenAICompatibleClient(base_url=BASE, model="m", transport=t)
    assert route(client) == RATIOS
    assert all(b["logprobs"] is True for b in t.bodies)


# ================================================================ JSON 模式关掉：靠提示词，宽容解析


QA_VOTE = {
    Q_GROUNDED: {"answer": "是", "p": 0.95},
    Q_ANSWERS_QUESTION: {"answer": "是", "p": 0.9},
    Q_PROMISE_RISK: {"answer": "否", "p": 0.05},
    Q_PLATFORM_VIOLATION: {"answer": "否", "p": 0.02},
    Q_TONE: {"score": 3},
}


@pytest.mark.parametrize(
    "wrap",
    [
        lambda j: f"好的，以下是判断结果：\n```json\n{j}\n```\n以上。",
        lambda j: f"判断如下：{j}（仅供参考）",
        lambda j: f'<think>先看依据，草稿 {{"grounded": "否"}}</think>\n{j}',
    ],
    ids=["fenced", "leading-text", "think-with-draft"],
)
def test_qa_judge_parses_prompt_only_json(wrap):
    t = Cycle([wrap(json.dumps(QA_VOTE, ensure_ascii=False))])
    client = OpenAICompatibleClient(base_url=BASE, model="m", transport=t, json_mode=False)
    provider = InCloudLLMProvider(client, samples=1, temperature=0.0, purpose="qa")
    request = DecisionRequest(
        model=QA_MODEL,
        state={"answer": "亲，默认发中通快递哦", "evidence": [], "messages": []},
        questions=QA_QUESTIONS,
    )
    answers = provider.decide(request).answers
    assert answers[Q_GROUNDED].noul == pytest.approx(0.95)  # type: ignore[union-attr]
    assert answers[Q_TONE].score == 3  # type: ignore[union-attr]
    assert "response_format" not in t.bodies[0]


def test_reasoning_with_a_template_would_have_cost_the_vote():
    """为什么要在客户端去掉 <think>：思考里抄了一遍回答格式，判断层取第一个 {...} 就拿到模板。"""
    raw = '<think>格式是 {"intent": {"choice": "<选项键>"}}</think>' + route_vote("size_advice")
    assert Q_INTENT not in parse_sample(ROUTER_QUESTIONS, raw)  # 不去掉：这一票作废
    client = OpenAICompatibleClient(base_url=BASE, model="m", transport=Cycle([raw]))
    text = client.complete(
        LLMRequest(purpose="route", messages=[LLMMessage(role="user", content="JSON")])
    ).text
    assert parse_sample(ROUTER_QUESTIONS, text)[Q_INTENT] == "size_advice"


# ================================================================ 生成：思考里的草稿不会被当成回答


DRAFT = '<think>先写个草稿 {"answer": "草稿：发顺丰"}，再核对依据</think>\n'
FINAL = '{"answer": "亲，默认发中通快递哦", "evidence_ids": ["faq_shipping"]}'


def test_reasoning_with_a_draft_would_have_become_the_answer():
    """不去掉 <think>：生成的解析从第一个 { 解码，拿到的是思考里的草稿。"""
    assert parse_generation_output(DRAFT + "好的：" + FINAL).answer == "草稿：发顺丰"


@pytest.mark.parametrize("json_mode", [True, False])
@pytest.mark.parametrize(
    "after",
    [f"好的，以下是回答：\n```json\n{FINAL}\n```", f"好的：{FINAL}"],
    ids=["fenced", "inline"],
)
def test_generation_output_after_reasoning_and_leading_text(json_mode, after):
    t = Cycle([DRAFT + after])
    client = OpenAICompatibleClient(base_url=BASE, model="m", transport=t, json_mode=json_mode)
    resp = client.complete(
        LLMRequest(
            purpose="generate",
            messages=[LLMMessage(role="user", content="按 JSON 输出")],
            json_mode=True,
        )
    )
    out = parse_generation_output(resp.text)
    assert out.answer == "亲，默认发中通快递哦" and out.evidence_ids == ["faq_shipping"]
    assert ("response_format" in t.bodies[0]) is json_mode
