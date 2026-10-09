"""平台云内大模型判断层：提示词、自一致性投票、宽容解析、并行采样与上下文传递（§6.4、§6.5）。"""

from __future__ import annotations

import json
import math
import threading

import pytest

from rodit.core.context import current_shop_id, shop_scope
from rodit.core.decision import ChoiceQuestion, DecisionRequest, NoulQuestion, ScoreQuestion
from rodit.core.errors import LLM_ERROR_KINDS, DecisionError, LLMError, LLMTimeout
from rodit.core.ports import LLMRequest, LLMResponse
from rodit.core.questions import (
    Q_EMOTION,
    Q_GROUNDED,
    Q_INTENT,
    Q_MULTI_INTENT,
    Q_NEED_HUMAN,
    Q_TONE,
    QA_MODEL,
    QA_QUESTIONS,
    ROUTER_MODEL,
    ROUTER_QUESTIONS,
    ROUTER_VOTE_QUESTIONS,
)
from rodit.core.states import build_router_state
from rodit.decision.llm import (
    ERROR_KIND_LABELS,
    SYSTEM_PROMPT,
    InCloudLLMProvider,
    build_judge_messages,
)
from rodit.decision.voting import (
    choice_distribution,
    extract_json_object,
    parse_noul_vote,
    parse_sample,
)
from rodit.llm import OpenAICompatibleClient
from rodit.testing.fakes import FakeLLMClient, make_context
from rodit.testing.fixtures import SHOP_B, build_demo_world

W = build_demo_world()


def _request(text: str = "165 110斤穿什么码", **kw) -> DecisionRequest:
    ctx = make_context(text, W, **kw)
    return DecisionRequest(
        model=ROUTER_MODEL, state=build_router_state(ctx), questions=ROUTER_QUESTIONS
    )


def _vote(intent="size_advice", need="否", p=0.1, emotion=0, multi="否") -> dict:
    return {
        Q_INTENT: {"choice": intent},
        Q_NEED_HUMAN: {"answer": need, "p": p},
        Q_EMOTION: {"score": emotion},
        Q_MULTI_INTENT: {"answer": multi, "p": 0.1 if multi == "否" else 0.9},
    }


def _provider(llm, samples=3, **kw) -> InCloudLLMProvider:
    kw.setdefault("temperature", 0.7)
    kw.setdefault("purpose", "route")
    return InCloudLLMProvider(llm, samples=samples, **kw)


# ---------------------------------------------------------------- 提示词与请求（§6.5）


def test_prompt_contains_state_and_every_question_definition():
    request = _request(item_id="i_555")
    messages = build_judge_messages(request)
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == SYSTEM_PROMPT
    user = messages[1].content
    assert json.dumps(request.state, ensure_ascii=False, indent=2) in user
    for key in ROUTER_QUESTIONS:
        assert f"### {key}" in user
    intent_q = ROUTER_QUESTIONS[Q_INTENT]
    assert isinstance(intent_q, ChoiceQuestion)
    for option, definition in intent_q.criteria.items():
        assert f"- {option}：{definition}" in user
    assert "0 = 平静" in user and "4 = 极度愤怒" in user
    assert '"choice"' in user and '"answer"' in user and '"score"' in user


def test_llm_request_fields():
    llm = FakeLLMClient(lambda r: _vote())
    provider = _provider(llm, samples=1, temperature=0.0, timeout_s=0.8)
    with shop_scope(SHOP_B):
        provider.decide(_request())
    (call,) = llm.calls
    assert call.purpose == "route"
    assert call.json_mode is True
    assert call.temperature == 0.0
    assert call.timeout_s == 0.8
    assert call.want_logprobs is False  # JSON 模式下读不出选项概率：不请求
    assert call.shop_id == SHOP_B  # 成本归属（§12.4）


# ---------------------------------------------------------------- 投票（§6.4）


def test_majority_vote_two_of_three():
    llm = FakeLLMClient(
        responses=[
            _vote("size_advice", need="否", emotion=0),
            _vote("product_spec", need="是", p=0.9, emotion=1),
            _vote("size_advice", need="否", emotion=2, multi="是"),
        ]
    )
    resp = _provider(llm).decide(_request())
    intent = resp.choice(Q_INTENT)
    assert intent.choice == "size_advice"
    assert intent.confidence == pytest.approx(2 / 3, abs=1e-3)
    assert intent.probabilities == pytest.approx({"size_advice": 2 / 3, "product_spec": 1 / 3})
    assert resp.noul(Q_NEED_HUMAN) == pytest.approx(1 / 3, abs=1e-3)  # "是"的票数占比
    assert resp.score(Q_EMOTION) == pytest.approx(1.0)  # 平均等级
    assert resp.noul(Q_MULTI_INTENT) == pytest.approx(1 / 3, abs=1e-3)
    assert len(llm.calls) == 3


def test_tie_is_broken_by_criteria_order():
    llm = FakeLLMClient(responses=[_vote("stock"), _vote("product_spec"), _vote("size_advice")])
    intent = _provider(llm).decide(_request()).choice(Q_INTENT)
    assert intent.choice == "product_spec"  # criteria 里排第一
    assert intent.confidence == pytest.approx(1 / 3, abs=1e-3)


def test_invalid_votes_are_dropped_per_question():
    broken = _vote("size_advice")
    broken[Q_INTENT] = {"choice": "not_an_intent"}
    broken[Q_EMOTION] = {"score": "很生气"}
    llm = FakeLLMClient(
        responses=[_vote("size_advice", emotion=1), broken, "抱歉，我无法回答", _vote("stock")]
    )
    resp = _provider(llm, samples=4).decide(_request())
    intent = resp.choice(Q_INTENT)
    assert intent.probabilities == pytest.approx({"size_advice": 0.5, "stock": 0.5})
    assert intent.choice == "size_advice"
    assert resp.score(Q_EMOTION) == pytest.approx(0.5)  # 只数 1 和 0 两张有效票
    assert resp.noul(Q_NEED_HUMAN) == 0.0  # 三张有效票都是"否"


def test_all_votes_invalid_for_a_question_raises():
    votes = [_vote(), _vote(), _vote()]
    for v in votes:
        del v[Q_EMOTION]
    llm = FakeLLMClient(responses=votes)
    with pytest.raises(DecisionError, match="emotion"):
        _provider(llm).decide(_request())


def test_all_samples_failing_raises_decision_error():
    llm = FakeLLMClient(lambda r: LLMTimeout("超时"))
    with pytest.raises(DecisionError) as e:
        _provider(llm).decide(_request())
    assert isinstance(e.value.__cause__, LLMError)


def test_failed_samples_are_dropped():
    llm = FakeLLMClient(responses=[LLMError("限流"), _vote("stock"), _vote("stock")])
    intent = _provider(llm).decide(_request()).choice(Q_INTENT)
    assert (intent.choice, intent.confidence) == ("stock", 1.0)


def test_quorum_rejects_too_few_valid_votes():
    llm = FakeLLMClient(responses=[LLMError("限流"), LLMError("限流"), _vote("stock")])
    with pytest.raises(DecisionError, match="有效票"):
        _provider(llm, min_votes=2).decide(_request())
    with pytest.raises(ValueError):
        _provider(llm, min_votes=4)
    with pytest.raises(ValueError):
        _provider(llm, samples=0)


def test_single_sample_uses_reported_probability():
    llm = FakeLLMClient(
        responses=[
            {**_vote(), Q_NEED_HUMAN: {"answer": "是", "p": 0.83}},
            {**_vote(), Q_NEED_HUMAN: {"answer": "是", "p": 1.7}},  # 截到 [0, 1]
            {**_vote(), Q_NEED_HUMAN: {"answer": "否"}},  # 没有 p：按答案取 0
            {**_vote(), Q_NEED_HUMAN: {"answer": "否", "p": 0.9}},  # 报的是把握：换算成 0.1
        ]
    )
    provider = _provider(llm, samples=1, temperature=0.0)
    got = [provider.decide(_request()).noul(Q_NEED_HUMAN) for _ in range(4)]
    assert got == pytest.approx([0.83, 1.0, 0.0, 0.1])


def test_score_votes_are_clipped_to_levels():
    llm = FakeLLMClient(responses=[_vote(emotion=9), _vote(emotion=-2), _vote(emotion="愤怒")])
    assert _provider(llm).decide(_request()).score(Q_EMOTION) == pytest.approx((4 + 0 + 3) / 3)


def test_logprobs_override_vote_shares():
    def handler(r: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=json.dumps(_vote("size_advice")),
            logprobs={"size_advice": 0.7, "product_spec": 0.2, "other": 0.1, "junk": 0.5},
        )

    intent = _provider(FakeLLMClient(handler), samples=1).decide(_request()).choice(Q_INTENT)
    assert intent.choice == "size_advice"
    assert intent.probabilities == pytest.approx(
        {"size_advice": 0.7, "product_spec": 0.2, "other": 0.1}
    )

    def log_handler(r: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=json.dumps(_vote("stock")),
            logprobs={"stock": math.log(0.6), "price_promo": math.log(0.3)},
        )

    intent = _provider(FakeLLMClient(log_handler), samples=3).decide(_request()).choice(Q_INTENT)
    assert intent.probabilities == pytest.approx({"stock": 2 / 3, "price_promo": 1 / 3})


# ---------------------------------------------------------------- 不请求选项概率（logprobs）


@pytest.mark.parametrize(
    "questions",
    [
        ROUTER_QUESTIONS,
        ROUTER_VOTE_QUESTIONS,
        QA_QUESTIONS,
        {"q_pick": ChoiceQuestion(instructions="选一个", criteria={"a": None, "b": None})},
    ],
    ids=["route", "route_votes", "qa", "single_choice"],
)
def test_judge_requests_never_ask_for_logprobs(questions):
    """判断层的请求都是 JSON 模式：输出里第一个有内容的 token 是键名，读不出选项概率，一律不请求
    （不支持 logprobs 的接口可能把整个请求拒掉）。"""
    request = DecisionRequest(model="m", state={"synthetic": True}, questions=dict(questions))
    llm = FakeLLMClient(lambda r: "{}")  # 替身不作答：这里只看发出去的请求
    with pytest.raises(DecisionError):
        _provider(llm, samples=1, temperature=0.0).decide(request)
    (call,) = llm.calls
    assert call.json_mode is True
    assert call.want_logprobs is False


def _completion(text: str, *, logprobs: bool) -> dict:
    """OpenAI 格式的响应体（合成）。logprobs=True 时带上选项概率：JSON 模式下第一个有内容的位置是
    键名 intent，它的候选里混进了一个和选项同名的 token（other，概率十万分之一）。"""
    choice: dict = {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
    if logprobs:
        choice["logprobs"] = {
            "content": [
                {"token": '{"', "logprob": -0.01, "top_logprobs": []},
                {
                    "token": "intent",
                    "logprob": math.log(0.999),
                    "top_logprobs": [
                        {"token": "intent", "logprob": math.log(0.999)},
                        {"token": "other", "logprob": math.log(1e-5)},
                    ],
                },
            ]
        }
    return {"model": "synthetic-served", "choices": [choice], "usage": {}}


@pytest.mark.parametrize("vendor", ["rejects_logprobs", "honours_logprobs"])
def test_real_client_sends_no_logprobs_so_the_votes_decide(monkeypatch, vendor):
    """经真实客户端（假 transport，不联网）：请求体里没有 logprobs / top_logprobs。
    不支持的接口不再把每一次路由调用都拒掉；照给的接口也没有机会让键名位置上一个和选项同名的
    候选（归一化之后成了 1.0）盖过三张一致的票。"""
    monkeypatch.setenv("RODIT_LLM_API_KEY", "sk-synthetic-test-key-not-real-0002")
    bodies: list[dict] = []
    answer = json.dumps(_vote("size_advice"), ensure_ascii=False)

    def transport(url, headers, body, timeout_s):
        bodies.append(body)
        asked = "logprobs" in body or "top_logprobs" in body
        if asked and vendor == "rejects_logprobs":
            return 400, {"error": {"type": "invalid_request_error", "code": "invalid_parameter"}}
        return 200, _completion(answer, logprobs=asked)

    client = OpenAICompatibleClient(
        base_url="https://llm.example.invalid/v1", model="synthetic", transport=transport
    )
    intent = _provider(client).decide(_request()).choice(Q_INTENT)
    assert (intent.choice, intent.confidence) == ("size_advice", 1.0)
    assert intent.probabilities == {"size_advice": 1.0}
    assert len(bodies) == 3
    for body in bodies:
        assert "logprobs" not in body and "top_logprobs" not in body
        assert body["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------- 没算上的票（截断、空回答、失败）


def _cut(vote: dict) -> LLMResponse:
    """被 max_tokens 截断的回答（finish_reason 是 length）。文本故意给成完整的 JSON：
    即使读得出来也不能解析、不能计票。"""
    return LLMResponse(text=json.dumps(vote, ensure_ascii=False), finish_reason="length")


def _decision_error(llm, **kw) -> DecisionError:
    with pytest.raises(DecisionError) as e:
        _provider(llm, **kw).decide(_request())
    return e.value


def test_truncated_response_is_a_lost_vote_and_is_not_parsed():
    cut = _cut(_vote("stock", need="是", p=0.9, emotion=4, multi="是"))
    llm = FakeLLMClient(responses=[cut, _vote("size_advice"), _vote("size_advice")])
    resp = _provider(llm).decide(_request())
    assert resp.choice(Q_INTENT).probabilities == {"size_advice": 1.0}  # 只数没被截断的两票
    assert resp.noul(Q_NEED_HUMAN) == 0.0
    assert resp.score(Q_EMOTION) == 0.0
    assert resp.noul(Q_MULTI_INTENT) == 0.0


def test_truncated_response_does_not_vote_on_the_followup_either():
    """追问按票和主要问题配对：被截断的那一票不出"无"，也不出第二个问题。"""
    request = DecisionRequest(
        model=ROUTER_MODEL, state=_request().state, questions=ROUTER_VOTE_QUESTIONS
    )
    second = {"second_intent": {"choice": "shipping_policy"}}
    cut = _cut({**_vote("size_advice", multi="是"), **second})
    whole = {**_vote("size_advice", multi="是"), "second_intent": {"choice": "stock"}}
    llm = FakeLLMClient(responses=[cut, whole, whole])
    resp = _provider(llm).decide(request)
    assert resp.choice("second_intent").probabilities == {"stock": 1.0}


def test_blank_response_is_a_lost_vote():
    llm = FakeLLMClient(responses=[LLMResponse(text=" \n"), LLMResponse(text=""), _vote("stock")])
    error = _decision_error(llm, min_votes=2)
    assert str(error) == "问题 'intent' 只有 1 张有效票，少于 2（3 次：空回答 2）"
    # 够票时照常出结果，空回答只是少一票
    llm = FakeLLMClient(responses=[LLMResponse(text=""), _vote("stock"), _vote("stock")])
    intent = _provider(llm, min_votes=2).decide(_request()).choice(Q_INTENT)
    assert intent.probabilities == {"stock": 1.0}


def test_an_answer_with_two_json_objects_is_a_lost_vote():
    """先写一份草稿、后面才是答案（两个顶层 JSON 对象）：分不清哪个算数，这一票不计——
    以前取第一个，判的是草稿。"""
    good = _vote("stock")
    draft = json.dumps(_vote("other"), ensure_ascii=False)
    two = LLMResponse(text=draft + "\n最终答案：" + json.dumps(good, ensure_ascii=False))
    error = _decision_error(FakeLLMClient(lambda r: two), samples=1, temperature=0.0)
    assert str(error) == "大模型判断全部失败（1 次：读不出 1）"
    # 三票里一票是这样的：另外两票照常计
    llm = FakeLLMClient(responses=[two, good, good])
    intent = _provider(llm, min_votes=2).decide(_request()).choice(Q_INTENT)
    assert intent.probabilities == {"stock": 1.0}


def test_all_votes_lost_names_each_cause_with_its_count():
    timeout = LLMTimeout("大模型调用超时（1 秒）")
    llm = FakeLLMClient(responses=[timeout, timeout, _cut(_vote("stock"))])
    error = _decision_error(llm)
    assert str(error) == "大模型判断全部失败（3 次：超时 2、截断 1）"
    assert error.__cause__ is timeout  # 调用失败的那一票照旧挂在异常链上


def test_all_votes_truncated_or_blank_have_no_call_error_to_chain():
    llm = FakeLLMClient(responses=[_cut(_vote()), _cut(_vote()), LLMResponse(text="")])
    error = _decision_error(llm)
    assert str(error) == "大模型判断全部失败（3 次：截断 2、空回答 1）"
    assert error.__cause__ is None
    # 质检只采一次：被截断就是整次判断不可用
    error = _decision_error(FakeLLMClient(lambda r: _cut(_vote())), samples=1, temperature=0.0)
    assert str(error) == "大模型判断全部失败（1 次：截断 1）"
    # 输出额度全花在思考上：文本是空的、finish_reason 是 length，算截断（提示的是 max_tokens）
    spent = LLMResponse(text="", finish_reason="length")
    error = _decision_error(FakeLLMClient(lambda r: spent), samples=1, temperature=0.0)
    assert str(error) == "大模型判断全部失败（1 次：截断 1）"


def test_quorum_message_tells_unreadable_votes_from_lost_ones():
    """回答完整、但这道题读不出有效票的算"读不出"，和调用失败、截断分开数。"""
    unreadable = _vote()
    unreadable[Q_EMOTION] = {"score": "很生气"}
    llm = FakeLLMClient(responses=[unreadable, unreadable, _vote()])
    error = _decision_error(llm, min_votes=2)
    assert str(error) == "问题 'emotion' 只有 1 张有效票，少于 2（3 次：读不出 2）"
    llm = FakeLLMClient(responses=[unreadable, unreadable, unreadable])
    error = _decision_error(llm)
    assert str(error) == "问题 'emotion' 的所有投票都无效（3 次：读不出 3）"
    # 三票三种去向；票数一样时按原因名排，写法固定
    llm = FakeLLMClient(responses=[LLMTimeout("超时"), _cut(_vote()), unreadable])
    error = _decision_error(llm)
    assert str(error) == "问题 'emotion' 的所有投票都无效（3 次：截断 1、读不出 1、超时 1）"
    assert error.__cause__ is None  # 有效票不够的报错不挂某一次调用的异常：原因都在消息里


class _BareLLMError(LLMError):
    """没有 status / kind / code 的替身（不调 LLMError.__init__）。"""

    def __init__(self, msg: str) -> None:
        Exception.__init__(self, msg)


class _BareLLMTimeout(LLMTimeout):
    def __init__(self, msg: str) -> None:
        Exception.__init__(self, msg)


@pytest.mark.parametrize(
    ("error", "cause"),
    [
        (LLMTimeout("x"), "超时"),
        (LLMError("x", kind="network"), "连不上"),
        (LLMError("x", kind="malformed"), "响应格式不对"),
        (LLMError("x", kind="config"), "本地配置不对"),
        (LLMError("x", status=503, kind="server"), "模型服务出错（HTTP 503）"),
        (
            LLMError("x", status=401, kind="auth", code="invalid_api_key"),
            "密钥无效或没有权限（HTTP 401，invalid_api_key）",
        ),
        (
            LLMError("x", status=429, kind="rate_limit", code="rate_limit_exceeded"),
            "限流（HTTP 429，rate_limit_exceeded）",
        ),
        (  # 429 既用来限流也用来报额度用完：靠 kind 分开
            LLMError("x", status=429, kind="billing", code="insufficient_quota"),
            "余额或额度用完（HTTP 429，insufficient_quota）",
        ),
        (LLMError("x", status=402, kind="billing"), "余额或额度用完（HTTP 402）"),
        (
            LLMError("x", status=400, kind="bad_request", code=1210),  # type: ignore[arg-type]
            "请求被拒（HTTP 400，1210）",
        ),
        (LLMError("x", status=418), "HTTP 418"),  # 没有 kind：只有状态码
        (LLMError("x", status=400, code="  bad\nparam "), "HTTP 400（bad param）"),
        (LLMError("x", code="quota"), "LLMError（quota）"),
        (LLMError("x", code="c" * 200), "LLMError（" + "c" * 64 + "）"),  # 错误代码截短
        (LLMError("x", status=True, code=""), "LLMError"),  # type: ignore[arg-type]
        (LLMError("x"), "LLMError"),  # 什么都没带：只有异常类型
        (_BareLLMError("x"), "_BareLLMError"),
        (_BareLLMTimeout("x"), "超时"),
    ],
)
def test_call_failure_is_named_from_the_error_attributes(error, cause):
    llm = FakeLLMClient(lambda r: error)
    raised = _decision_error(llm, samples=1, temperature=0.0)
    assert str(raised) == f"大模型判断全部失败（1 次：{cause} 1）"


def test_every_llm_error_kind_has_a_label():
    assert set(ERROR_KIND_LABELS) == set(LLM_ERROR_KINDS)
    assert len(set(ERROR_KIND_LABELS.values())) == len(LLM_ERROR_KINDS)  # 各说各的，不重名


def test_failure_messages_never_carry_model_or_vendor_text():
    """消息里只有票数、原因的类别和 LLMError 上的状态码 / 错误代码：没有异常消息、没有模型输出。"""
    secret = "买家留的手机 13812345678"
    errors = [
        LLMError(secret, status=429, kind="rate_limit", code="rate_limit_exceeded"),
        LLMError(secret, status=429, kind="rate_limit", code="rate_limit_exceeded"),
        LLMResponse(text=secret + '{"intent"', finish_reason="length"),
    ]
    error = _decision_error(FakeLLMClient(responses=errors))
    limited = "限流（HTTP 429，rate_limit_exceeded）"
    assert str(error) == f"大模型判断全部失败（3 次：{limited} 2、截断 1）"
    unreadable = FakeLLMClient(lambda r: secret)
    assert str(_decision_error(unreadable)) == "问题 'intent' 的所有投票都无效（3 次：读不出 3）"


def test_responses_without_a_finish_reason_attribute_still_count():
    """替身给的回答没有 finish_reason 这一项时照常解析、计票。"""

    def handler(r: LLMRequest) -> LLMResponse:
        response = LLMResponse(text=json.dumps(_vote("stock")))
        del response.__dict__["finish_reason"]
        assert not hasattr(response, "finish_reason")
        return response

    intent = _provider(FakeLLMClient(handler)).decide(_request()).choice(Q_INTENT)
    assert intent.probabilities == {"stock": 1.0}


# ---------------------------------------------------------------- 宽容解析


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"a": 1}\n```',
        '好的，结果如下：{"a": 1} 以上。',
        '{"a": 1}',
        '```\n{"a": 1}\n```\n补充说明 {"b": 2}',
    ],
)
def test_extract_json_object(text):
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_object_handles_braces_in_strings_and_garbage():
    assert extract_json_object('前缀 {"a": "}{"} 后缀') == {"a": "}{"}
    assert extract_json_object("没有 JSON") is None
    assert extract_json_object("") is None
    assert extract_json_object("[1, 2]") is None


@pytest.mark.parametrize(
    ("raw", "yes", "p"),
    [
        ("是", True, None),
        ("否", False, None),
        ("Yes", True, None),
        ("no", False, None),
        (True, True, None),
        ("false", False, None),
        ({"answer": "是的"}, True, None),
        ({"answer": "是", "p": "0.8"}, True, 0.8),
        ({"noul": 0.3}, False, 0.3),
        ({"p": 0.7}, True, 0.7),
        ("0.95", True, 0.95),
    ],
)
def test_noul_vote_parsing(raw, yes, p):
    vote = parse_noul_vote(raw)
    assert vote is not None and vote.yes is yes
    assert vote.p == (pytest.approx(p) if p is not None else None)


def test_parse_sample_tolerates_wrappers_case_and_labels():
    text = (
        "```json\n"
        + json.dumps(
            {
                "answers": {
                    "INTENT": {"choice": " Size_Advice（尺码 / 选型）"},
                    "need_human": "否",
                    "emotion": {"score": "不满"},
                    "multi_intent": {"answer": "maybe"},
                }
            },
            ensure_ascii=False,
        )
        + "\n```"
    )
    votes = parse_sample(ROUTER_QUESTIONS, text)
    assert votes[Q_INTENT] == "size_advice"
    assert votes[Q_EMOTION] == 2.0
    assert votes[Q_NEED_HUMAN].yes is False  # type: ignore[union-attr]
    assert Q_MULTI_INTENT not in votes  # 解析不了的票丢弃


# ---------------------------------------------------------------- 并行与上下文（§12.1、§12.4）


class _BarrierLLM:
    """只有 3 个调用同时在进行时才会返回：证明采样是并行的。"""

    def __init__(self, n: int) -> None:
        self.barrier = threading.Barrier(n, timeout=5)
        self.threads: set[str] = set()
        self.shops: list[str | None] = []
        self._lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.barrier.wait()
        with self._lock:
            self.threads.add(threading.current_thread().name)
            self.shops.append(current_shop_id())
        return LLMResponse(text=json.dumps(_vote("stock")))


def test_samples_run_in_parallel_and_see_the_shop_context():
    llm = _BarrierLLM(3)
    with shop_scope(SHOP_B):
        resp = _provider(llm).decide(_request())
    assert resp.choice(Q_INTENT).choice == "stock"
    assert len(llm.threads) == 3
    assert llm.shops == [SHOP_B] * 3


def test_context_propagates_into_fake_llm_handler():
    seen: list[str | None] = []

    def handler(r: LLMRequest) -> dict:
        seen.append(current_shop_id())
        return _vote()

    with shop_scope("s_ctx"):
        _provider(FakeLLMClient(handler), samples=3).decide(_request())
    assert seen == ["s_ctx"] * 3


# ---------------------------------------------------------------- 个人信息与质检题型


def test_prompt_never_contains_buyer_pii():
    text = "我手机13812345678，地址改到杭州市西湖区文三路 188 号 3 单元 501 室，能改吗"
    llm = FakeLLMClient(lambda r: _vote("modify_order"))
    request = _request(text, order_id="o_1001")
    _provider(llm).decide(request)
    prompts = llm.all_prompt_text()
    assert "13812345678" not in prompts
    assert "文三路" not in prompts and "501" not in prompts
    assert "[手机号]" in prompts and "[地址]" in prompts
    assert request.state["order_card"]["order_id"] in prompts  # 订单号不是个人信息，保留


def test_qa_questions_use_the_same_provider():
    request = DecisionRequest(
        model=QA_MODEL, state={"answer": "亲，48 小时内发货"}, questions=QA_QUESTIONS
    )
    answer = {
        "grounded": {"answer": "是", "p": 0.95},
        "answers_question": {"answer": "是", "p": 0.9},
        "promise_risk": {"answer": "否", "p": 0.05},
        "platform_violation": {"answer": "否", "p": 0.02},
        "tone": {"score": 3},
    }
    llm = FakeLLMClient(lambda r: answer)
    resp = _provider(llm, samples=1, temperature=0.0, purpose="qa").decide(request)
    assert resp.noul(Q_GROUNDED) == pytest.approx(0.95)
    assert resp.score(Q_TONE) == 3
    assert llm.calls[0].purpose == "qa"
    assert llm.calls[0].want_logprobs is False  # 判断层一律不请求选项概率


def test_generic_questions_work():
    questions = {
        "q_yes": NoulQuestion(instructions="是否？"),
        "q_pick": ChoiceQuestion(instructions="选一个", criteria={"a": None, "b": "选项 B"}),
        "q_score": ScoreQuestion(instructions="打分", criteria=["低", "高"]),
    }
    request = DecisionRequest(model="m", state={"synthetic": True}, questions=questions)
    llm = FakeLLMClient(lambda r: {"q_yes": "yes", "q_pick": "b", "q_score": 1})
    resp = _provider(llm, samples=2).decide(request)
    assert resp.noul("q_yes") == 1.0
    assert resp.choice("q_pick").choice == "b"
    assert resp.score("q_score") == 1.0
    assert "- a\n" in llm.calls[0].messages[1].content


def test_provider_identity():
    p = _provider(FakeLLMClient(lambda r: _vote()))
    assert p.name == "incloud_llm" and p.accepts_buyer_data is True


def test_choice_distribution_edge_cases():
    q = ROUTER_QUESTIONS[Q_INTENT]
    assert isinstance(q, ChoiceQuestion)
    assert choice_distribution(q, None) is None
    assert choice_distribution(q, {}) is None
    assert choice_distribution(q, {"junk": 0.9}) is None  # 对不上任何选项
    assert choice_distribution(q, {"stock": 2.0}) is None  # 既不是概率也不是对数概率
    assert choice_distribution(q, {"stock": -0.1, "other": 0.5}) is None  # 正负混杂
    assert choice_distribution(q, {"stock": 0.0, "other": 0.0}) is None
    assert choice_distribution(q, {"stock": float("nan")}) is None
    assert choice_distribution(q, {"STOCK": 0.3, "other": 0.1}) == pytest.approx(
        {"stock": 0.75, "other": 0.25}
    )
