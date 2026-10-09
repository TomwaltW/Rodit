"""开跑前的连通性检查（rodit.eval.llm_env：preflight / run_preflight / via_proxy）：发的就是评测
要发的请求，整轮每次调用都会失败的问题用几次调用就暴露出来（方案 §13、§12.1、§12.2；
docs/架构与分工.md §4.4）。

不联网：大模型用 FakeLLMClient（失败原因直接带 status / kind / code），或真实的
OpenAICompatibleClient 配脚本化的 transport（密钥是 monkeypatch 设的合成值，只用来检查不外泄）；
耗时用注入的假时钟（每个线程各走各的），不真的等；代理只改环境变量，不碰真实的代理。数据全部合成。
"""

from __future__ import annotations

import io
import itertools
import json
import threading
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from rodit.app.offline import EvidenceEchoLLM
from rodit.core.config import EngineConfig
from rodit.core.enums import Intent
from rodit.core.errors import LLM_ERROR_KINDS, LLMError, LLMTimeout
from rodit.core.pii import contains_pii, mask_pii
from rodit.core.ports import LLMMessage, LLMRequest, LLMResponse
from rodit.decision.llm import SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT
from rodit.eval import compare
from rodit.eval.cli import main as eval_main
from rodit.eval.dataset import CASES, cases
from rodit.eval.harness import llm_harness
from rodit.eval.llm_env import (
    PREFLIGHT_QUESTION,
    PreflightFailed,
    preflight,
    run_preflight,
    via_proxy,
)
from rodit.eval.runner import run_case
from rodit.llm import OpenAICompatibleClient
from rodit.testing.fakes import FakeLLMClient, judge_buyer_text

KEY = "sk-synthetic-preflight-key-DO-NOT-LEAK"  # 合成的假密钥，只用来检查不外泄
BASE_URL = "https://llm.example.invalid/v1"
PROXY = "http://proxy-user:proxy-secret@proxy.example.invalid:7897"  # 合成的代理地址
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY")
EXPECTED = {mask_pii(c.text): c.expected_intent for c in CASES}

ROUTE_OK: dict[str, Any] = {
    "intent": {"choice": "size_advice"},
    "need_human": {"answer": "否", "p": 0.05},
    "emotion": {"score": 0},
    "multi_intent": {"answer": "否", "p": 0.05},
    "second_intent": {"choice": "无"},
}
QA_OK: dict[str, Any] = {
    "grounded": {"answer": "是", "p": 0.95},
    "answers_question": {"answer": "是", "p": 0.95},
    "promise_risk": {"answer": "否", "p": 0.05},
    "platform_violation": {"answer": "否", "p": 0.02},
    "tone": {"score": 3},
}
GENERATION_OK: dict[str, Any] = {
    "answer": "亲，建议拍 L 码哦",
    "evidence_ids": ["item:i_555:size_advice"],
    "need_clarify": False,
}
ANSWERS: dict[str, Any] = {"route": ROUTE_OK, "qa": QA_OK, "generate": GENERATION_OK}
Handler = Callable[[LLMRequest], Any]


class ThreadClock:
    """假的单调时钟（秒）：每个线程各走各的，并发的几票互不影响；只用来量单次调用的耗时。"""

    def __init__(self) -> None:
        self._local = threading.local()

    def __call__(self) -> float:
        return getattr(self._local, "now", 0.0)

    def spend(self, seconds: float) -> None:
        self._local.now = self() + seconds


def scripted(
    clock: ThreadClock | None = None,
    *,
    latency: Mapping[str, float | list[float]] | None = None,
    answers: Mapping[str, Any] | None = None,
    fail: Callable[[LLMRequest], Exception | None] | None = None,
) -> Handler:
    """按用途作答的假模型（FakeLLMClient 的 handler）。

    latency：用途 → 这种请求要花的秒数（列表就每次调用取下一个），在假时钟上走；比请求自带的超时
    长就像真实客户端那样等到超时、抛 LLMTimeout。fail(request) 返回异常就抛它。"""
    queues = {k: list(v) for k, v in (latency or {}).items() if isinstance(v, list)}

    def handler(request: LLMRequest) -> Any:
        if clock is not None:
            pending = queues.get(request.purpose)
            seconds = pending.pop(0) if pending else (latency or {}).get(request.purpose, 0.0)
            assert isinstance(seconds, int | float)
            if request.timeout_s is not None and seconds > request.timeout_s:
                clock.spend(request.timeout_s)
                return LLMTimeout(f"大模型调用超时（{request.timeout_s:g} 秒）")
            clock.spend(seconds)
        failure = fail(request) if fail is not None else None
        return failure if failure is not None else (answers or ANSWERS)[request.purpose]

    return handler


def _fake(handler: Handler | None = None) -> FakeLLMClient:
    return FakeLLMClient(handler=handler or scripted(), model="synthetic-llm")


def _config(**sections: dict[str, Any]) -> EngineConfig:
    return EngineConfig.model_validate(sections)


def _failure(client: Any, config: EngineConfig | None = None, **kw: Any) -> str:
    with pytest.raises(PreflightFailed) as caught:
        run_preflight(client, config, **kw)
    return str(caught.value)


def _run_model(request: LLMRequest) -> Any:
    """评测整轮都答得上的假模型：路由按用例标注作答（检查用的合成会话答 other），质检通过，
    生成复述依据。"""
    if request.purpose == "route":
        intent = EXPECTED.get(judge_buyer_text(request), Intent.OTHER)
        return {**ROUTE_OK, "intent": {"choice": intent.value}}
    if request.purpose == "qa":
        return QA_OK
    return EvidenceEchoLLM().complete(request)


Reply = tuple[int, dict[str, Any]]


class Vendor:
    """脚本化的模型服务（OpenAICompatibleClient 的 transport）：记下每次请求的用途、请求体和超时，
    按 _run_model 作答。

    override(用途, 请求体) 返回 (状态码, 响应体) 时原样回它；seconds 是每次请求要花的秒数，
    在假时钟 clock 上走，比这次请求的超时长就等到超时、抛 TimeoutError（同 urllib 的传输层）。"""

    def __init__(
        self,
        override: Callable[[str, dict[str, Any]], Reply | None] | None = None,
        *,
        clock: ThreadClock | None = None,
        seconds: float = 0.0,
    ) -> None:
        self.override = override
        self.clock = clock
        self.seconds = seconds
        self.seen: list[tuple[str, dict[str, Any], float]] = []
        self._lock = threading.Lock()

    def __call__(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float
    ) -> Reply:
        messages = [LLMMessage(**m) for m in body["messages"]]
        if messages[0].content == JUDGE_SYSTEM_PROMPT:
            purpose = "route" if "### intent" in messages[-1].content else "qa"
        else:
            purpose = "generate"
        with self._lock:
            self.seen.append((purpose, body, timeout_s))
        if self.clock is not None:
            self.clock.spend(min(self.seconds, timeout_s))
            if self.seconds > timeout_s:
                raise TimeoutError("合成的超时")
        reply = self.override(purpose, body) if self.override is not None else None
        if reply is not None:
            return reply
        answer = _run_model(LLMRequest(purpose=purpose, messages=messages))  # type: ignore[arg-type]
        text = answer.text if isinstance(answer, LLMResponse) else json.dumps(answer)
        return 200, {
            "model": "synthetic-model",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    def client(self, base_url: str = BASE_URL) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(base_url=base_url, model="synthetic-model", transport=self)

    def shapes(self) -> dict[str, set[tuple[Any, ...]]]:
        """用途 → 发出去的请求形态：请求体的字段、温度、max_tokens、response_format、超时。"""
        out: dict[str, set[tuple[Any, ...]]] = {}
        for purpose, body, timeout_s in self.seen:
            shape = (
                tuple(sorted(body)),
                body["temperature"],
                body["max_tokens"],
                json.dumps(body.get("response_format"), sort_keys=True),
                timeout_s,
            )
            out.setdefault(purpose, set()).add(shape)
        return out


# ================================================================ 发的就是评测要发的请求


def test_preflight_sends_the_requests_the_run_sends(monkeypatch):
    """检查的 5 次请求和评测里同一个环节发的形态一样（请求体的字段、温度、max_tokens、
    response_format、超时）：拿同一个模型服务真跑几条用例来比。"""
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)
    vendor = Vendor()
    client = vendor.client()
    summary = preflight(client)
    assert Counter(p for p, _, _ in vendor.seen) == {"route": 3, "qa": 1, "generate": 1}
    defaults = {"route": (0.7, 1.0), "qa": (0.0, 1.0), "generate": (0.3, 2.5)}  # 温度、超时
    for purpose, body, timeout_s in vendor.seen:
        assert body["response_format"] == {"type": "json_object"}, purpose
        assert PREFLIGHT_QUESTION in body["messages"][-1]["content"], purpose  # 合成会话
        assert (body["temperature"], timeout_s) == defaults[purpose], purpose
    assert summary[0].startswith("大模型连通性检查通过") and "5 次调用" in summary[0]
    assert KEY not in "\n".join(summary)
    checked = vendor.shapes()

    vendor.seen.clear()
    for case in [c for c in cases("all") if c.expected_intent == Intent.SIZE_ADVICE][:20]:
        run_case(llm_harness(client), case)
        if {p for p, _, _ in vendor.seen} == {"route", "qa", "generate"}:
            break
    ran = vendor.shapes()
    assert set(ran) == {"route", "qa", "generate"}  # 这几条用例把三个环节都跑到了
    assert checked["route"] == ran["route"] and checked["qa"] == ran["qa"]
    assert checked["generate"] <= ran["generate"]


def test_every_shape_follows_the_config():
    """温度、投票次数、max_tokens、超时都按这次评测的配置来，没有写死的。"""
    config = _config(
        decision={
            "vote_samples": 5,
            "qa_vote_samples": 3,
            "vote_temperature": 0.5,
            "timeout_s": 3.0,
        },
        generation={"temperature": 0.9, "max_tokens": 256, "timeout_s": 6.0},
    )
    fake = _fake()
    text = "\n".join(run_preflight(fake, config))
    route, qa, generate = (fake.calls_for(p) for p in ("route", "qa", "generate"))
    assert len(route) == 5 and {(r.temperature, r.timeout_s) for r in route} == {(0.5, 3.0)}
    assert len(qa) == 3 and {(r.temperature, r.timeout_s) for r in qa} == {(0.5, 3.0)}
    assert [(r.temperature, r.max_tokens, r.timeout_s) for r in generate] == [(0.9, 256, 6.0)]
    assert all(r.json_mode for r in fake.calls) and len(fake.calls) == 9
    assert "9 次调用" in text
    assert "路由（5 票并发）" in text and "质检（3 票并发）" in text and "生成（1 次）" in text
    assert text.count("单次超时 3 秒（decision.timeout_s）") == 2
    assert "单次超时 6 秒（generation.timeout_s）" in text

    single = _fake()  # 只投一票：判断层的温度是 0（decision.factory），不并发
    text = "\n".join(run_preflight(single, _config(decision={"vote_samples": 1})))
    assert [r.temperature for r in single.calls_for("route")] == [0.0]
    assert "路由（1 次）" in text and len(single.calls) == 3


def test_check_requests_are_synthetic_and_as_large_as_real_ones():
    fake = _fake()
    run_preflight(fake)
    sizes = {r.purpose: sum(len(m.content) for m in r.messages) for r in fake.calls}
    # 评测集 324 条实测的提示词长度：路由 2242–2563 字、质检 1470–2636 字、生成 592–1341 字
    assert sizes["route"] >= 2242 and sizes["qa"] >= 1470 and sizes["generate"] >= 592
    for request in fake.calls:
        for message in request.messages:
            assert not contains_pii(message.content), request.purpose
        assert PREFLIGHT_QUESTION in request.messages[-1].content
    # 评测集里没有这句话：检查请求和评测请求分得开
    assert not contains_pii(PREFLIGHT_QUESTION)
    assert all(PREFLIGHT_QUESTION not in c.text and c.text not in PREFLIGHT_QUESTION for c in CASES)


# ================================================================ 通过时的摘要


def test_summary_lists_calls_latencies_timeouts_and_the_network_path():
    clock = ThreadClock()
    model = scripted(clock, latency={"route": [0.42, 0.51, 0.45], "qa": 0.38, "generate": 1.1})
    fake = _fake(model)
    assert run_preflight(fake, timer=clock) == [
        "大模型连通性检查通过：按这次评测要发的请求试了 5 次调用（合成数据，不计入评测用量）",
        "  路由（3 票并发）：耗时 0.42 / 0.45 / 0.51 秒，单次超时 1 秒（decision.timeout_s）",
        "  质检（1 次）：耗时 0.38 秒，单次超时 1 秒（decision.timeout_s）",
        "  生成（1 次）：耗时 1.10 秒，单次超时 2.5 秒（generation.timeout_s）",
        "  网络路径：不确定经没经代理（不是直接走 HTTP 的客户端，或者注入了 transport）",
    ]
    assert len(fake.calls) == 5  # 不重试、不遍历评测集


def test_a_latency_above_70_percent_of_the_timeout_is_a_warning_not_a_failure():
    clock = ThreadClock()
    model = scripted(clock, latency={"route": [0.2, 0.8, 0.2], "qa": 0.7, "generate": 1.9})
    summary = run_preflight(_fake(model), timer=clock)
    warnings = [line.strip() for line in summary if "注意：" in line]
    # 路由 0.80 > 1 × 70%；质检 0.70 正好不算；生成 1.90 > 2.5 × 70%
    assert len(warnings) == 2
    assert warnings[0].startswith("注意：最慢的一票用了 0.80 秒，超过单次超时 1 秒的 70%")
    assert "间歇性超时" in warnings[0] and "调大 decision.timeout_s" in warnings[0]
    assert "不能当上线依据" in warnings[0]  # 放宽超时的配置指纹不同
    assert warnings[1].startswith("注意：最慢的一次用了 1.90 秒，超过单次超时 2.5 秒的 70%")
    assert "调大 generation.timeout_s" in warnings[1]
    assert summary.index("    " + warnings[0]) == 2  # 紧跟在路由那一行后面


@pytest.mark.parametrize("provider", ["classifier", "rules"])
def test_routing_without_the_llm_judge_skips_the_routing_shape(provider: str):
    fake = _fake()
    config = _config(decision={"provider": provider})
    summary = run_preflight(fake, config, timer=ThreadClock())
    assert fake.calls_for("route") == [] and len(fake.calls) == 2  # 质检、生成照查
    assert summary[0].startswith("大模型连通性检查通过") and "2 次调用" in summary[0]
    assert (
        summary[1]
        == f"  路由：没有检查——这份配置的路由不走大模型判断层（decision.provider 是 {provider}）"
    )
    assert summary[2].startswith("  质检（1 次）") and summary[3].startswith("  生成（1 次）")


def test_a_jev_judge_leaves_only_the_generation_shape():
    config = _config(decision={"provider": "jev", "jev_production_approved": True})
    fake = _fake()
    summary = run_preflight(fake, config, timer=ThreadClock())
    assert [r.purpose for r in fake.calls] == ["generate"]
    assert "路由：没有检查" in summary[1] and "质检：没有检查" in summary[2]


# ================================================================ 没过：每种原因


def test_rejected_response_format_fails_every_shape_and_names_the_cause():
    def no_json_mode(request: LLMRequest) -> Exception | None:
        if not request.json_mode:
            return None
        return LLMError(
            "大模型接口拒绝请求（HTTP 400）",
            status=400,
            kind="bad_request",
            code="invalid_parameter",
        )

    fake = _fake(scripted(fail=no_json_mode))
    message = _failure(fake, timer=ThreadClock())
    first, *rest = message.splitlines()
    assert first == (
        "大模型连通性检查失败：按这次评测要发的请求试了 3 种、5 次调用（合成数据），"
        "路由、质检、生成没过，评测没有开跑。"
    )
    cause = "请求被拒（HTTP 400，错误代码 invalid_parameter）——大模型接口拒绝请求（HTTP 400）"
    assert f"    - 3 票：{cause}" in rest and rest.count(f"    - 1 次：{cause}") == 2
    assert (
        "  路由（3 票并发）：没过，耗时 0.00 / 0.00 / 0.00 秒，单次超时 1 秒（decision.timeout_s）"
        in rest
    )
    steps = rest[rest.index("下一步：") + 1 :]
    assert len(steps) == 1  # 同一种原因只说一次
    assert "response_format json_object" in steps[0] and "模型名" in steps[0]
    assert len(fake.calls) == 5  # 没有重试


def test_a_rejected_parameter_fails_only_the_shape_that_sends_it():
    def low_temperature_only(request: LLMRequest) -> Exception | None:
        if request.temperature <= 0.5:
            return None
        return LLMError("合成的故障：温度超出范围", status=400, code="invalid_temperature")

    message = _failure(_fake(scripted(fail=low_temperature_only)))
    assert "路由没过，评测没有开跑" in message.splitlines()[0]
    assert "  路由（3 票并发）：没过" in message
    assert "- 3 票：请求被拒（HTTP 400，错误代码 invalid_temperature）" in message  # 只有状态码也认
    assert "  质检（1 次）：通过" in message and "  生成（1 次）：通过" in message
    assert "decision.vote_temperature" in message


def test_a_vendor_without_temperature_zero_is_a_matter_of_config():
    """只投一票的质检温度是 0：拒绝 0 的模型服务在这里报出来，把 decision.qa_vote_samples 调成多票
    （用 vote_temperature）就过了——检查自己不再写死任何温度。"""

    def no_zero(request: LLMRequest) -> Exception | None:
        if request.temperature > 0:
            return None
        return LLMError("合成的故障：温度必须大于 0", status=400, kind="bad_request")

    message = _failure(_fake(scripted(fail=no_zero)))
    assert "质检没过，评测没有开跑" in message.splitlines()[0]
    assert "decision.qa_vote_samples（只投一票时温度是 0）" in message
    summary = run_preflight(_fake(scripted(fail=no_zero)), _config(decision={"qa_vote_samples": 3}))
    assert summary[0].startswith("大模型连通性检查通过")


def test_slow_judge_calls_fail_with_the_latency_next_to_the_timeout():
    """从这台机器调一次判断层要 1.2 秒：按线上的 1 秒超时全部超时，生成（2.5 秒）来得及。"""
    clock = ThreadClock()
    slow = {"route": 1.2, "qa": 1.2, "generate": 1.2}
    fake = _fake(scripted(clock, latency=slow))
    message = _failure(fake, timer=clock)
    lines = message.splitlines()
    assert "路由、质检没过，评测没有开跑" in lines[0]
    assert (
        "  路由（3 票并发）：没过，耗时 1.00 / 1.00 / 1.00 秒，单次超时 1 秒（decision.timeout_s）"
        in lines
    )
    assert "    - 3 票：超时——大模型调用超时（1 秒）" in lines
    assert "  质检（1 次）：没过，耗时 1.00 秒，单次超时 1 秒（decision.timeout_s）" in lines
    assert "  生成（1 次）：通过，耗时 1.20 秒，单次超时 2.5 秒（generation.timeout_s）" in lines
    (step,) = lines[lines.index("下一步：") + 1 :]
    for part in (
        "decision.timeout_s / generation.timeout_s",
        "--config",
        "不能当上线依据",
        "网络路径",
        "代理",
        "RODIT_LLM_TIMEOUT_S 管不到",
    ):
        assert part in step, part
    assert len(fake.calls) == 5  # 超时了也不重试

    relaxed = _config(decision={"timeout_s": 3.0})  # 探索性的评测：换一份放宽超时的配置
    summary = run_preflight(_fake(scripted(clock, latency=slow)), relaxed, timer=clock)
    assert (
        "  路由（3 票并发）：耗时 1.20 / 1.20 / 1.20 秒，单次超时 3 秒（decision.timeout_s）"
        in summary
    )
    assert not any("注意：" in line for line in summary)


def test_a_single_lost_vote_fails_the_check():
    """三票里丢一票，判断层自己照样答得出来；检查不放过——整轮评测里每条都会丢票。"""
    clock = ThreadClock()
    votes = itertools.count()

    def one_slow_vote(request: LLMRequest) -> Any:
        if request.purpose == "route" and next(votes) == 1:
            clock.spend(1.0)
            return LLMTimeout("大模型调用超时（1 秒）")
        clock.spend(0.2)
        return ANSWERS[request.purpose]

    lines = _failure(_fake(one_slow_vote), timer=clock).splitlines()
    assert "路由没过，评测没有开跑" in lines[0]
    assert (
        "  路由（3 票并发）：没过，耗时 0.20 / 0.20 / 1.00 秒，单次超时 1 秒（decision.timeout_s）"
        in lines
    )
    assert "    - 1 票：超时——大模型调用超时（1 秒）" in lines
    assert "  质检（1 次）：通过，耗时 0.20 秒，单次超时 1 秒（decision.timeout_s）" in lines
    # 另外两票是通过的：除了这种原因的下一步，还提醒可能是偶发的
    assert lines[-1].startswith("  - 同一种请求里有的调用过了、有的没过：也可能是偶发的")
    assert "可以再试一次" in lines[-1]
    assert lines[-2].startswith("  - 超时：") and lines[-3] == "下一步："


@pytest.mark.parametrize(
    ("response", "problem", "step"),
    [
        (
            LLMResponse(text="", model="synthetic-llm", finish_reason="length"),
            "输出被 max_tokens 截断（finish_reason 是 length），正文是空的",
            "输出被 max_tokens 截断：模型很可能开着思考模式",
        ),
        (
            LLMResponse(text='{"intent": {"cho', model="synthetic-llm", finish_reason="length"),
            "输出被 max_tokens 截断（finish_reason 是 length）",
            "generation.max_tokens",
        ),
        (
            LLMResponse(text="  \n", model="synthetic-llm", finish_reason="stop"),
            "调用成功，但回的正文是空的",
            "回了空的正文：模型很可能开着思考模式",
        ),
    ],
    ids=["blank-and-cut", "cut", "blank"],
)
def test_blank_or_truncated_output_points_at_thinking_mode(
    response: LLMResponse, problem: str, step: str
):
    fake = _fake(lambda request: response)
    lines = _failure(fake).splitlines()
    assert "路由、质检、生成没过" in lines[0]
    assert f"    - 3 票：{problem}" in lines and lines.count(f"    - 1 次：{problem}") == 2
    (hint,) = lines[lines.index("下一步：") + 1 :]
    assert step in hint
    assert "llm.extra_body" in hint and "--config" in hint
    assert "thinking: {type: disabled}" in hint and "enable_thinking: false" in hint
    assert len(fake.calls) == 5


@pytest.mark.parametrize(
    ("purpose", "answer", "problem"),
    [
        (
            "route",
            {**ROUTE_OK, "intent": {"choice": "尺码 / 选型"}},  # 答了中文名，不是选项键
            "- 3 票：没有答出 intent（没答、读不出，或者不在给的选项里）",
        ),
        (
            "qa",
            {**QA_OK, "tone": {"score": "3（礼貌）"}},
            "- 1 次：没有答出 tone（没答、读不出，或者不在给的选项里）",
        ),
        ("route", "好的，这位买家在问尺码。", "- 3 票：回答里读不出要的 JSON 答案"),
        ("qa", {"reasoning": "回答有依据"}, "- 1 次：回答里读不出要的 JSON 答案"),
        (
            "generate",
            "好的，建议拍 L 码哦",
            "- 1 次：生成的输出解析不了——大模型输出里找不到 JSON 对象",
        ),
        (
            "generate",
            {"answer": "", "evidence_ids": []},
            "- 1 次：生成的输出解析不了——大模型输出格式不对：answer",
        ),
    ],
    ids=[
        "intent-label",
        "tone-text",
        "judge-prose",
        "judge-other-json",
        "draft-prose",
        "draft-empty",
    ],
)
def test_output_the_stage_cannot_parse_fails_that_shape(purpose: str, answer: Any, problem: str):
    fake = _fake(scripted(answers={**ANSWERS, purpose: answer}))
    lines = _failure(fake).splitlines()
    label = {"route": "路由", "qa": "质检", "generate": "生成"}[purpose]
    assert f"{label}没过，评测没有开跑" in lines[0]
    assert f"    {problem}" in lines
    (hint,) = lines[lines.index("下一步：") + 1 :]
    assert hint.startswith("  - 回答读不出来：") and "JSON 模式" in hint
    assert sum("：通过，" in line for line in lines) == 2  # 另外两种请求照常查完、通过


def test_one_unreadable_vote_out_of_three_fails_the_check():
    votes = itertools.count()

    def one_bad_vote(request: LLMRequest) -> Any:
        if request.purpose == "route" and next(votes) == 2:
            return "抱歉，我无法判断。"
        return ANSWERS[request.purpose]

    lines = _failure(_fake(one_bad_vote)).splitlines()
    assert "    - 1 票：回答里读不出要的 JSON 答案" in lines
    assert lines[-2].startswith("  - 回答读不出来：")
    assert "也可能是偶发的，可以再试一次；反复出现的话" in lines[-1]


def test_the_follow_up_question_may_stay_unanswered():
    """路由的追问 second_intent 答不出来按"无"算（decision.llm）：检查也不要求它。"""
    four = {k: v for k, v in ROUTE_OK.items() if k != "second_intent"}
    fake = _fake(scripted(answers={**ANSWERS, "route": four}))
    assert run_preflight(fake)[0].startswith("大模型连通性检查通过")
    assert "### second_intent" in fake.calls_for("route")[0].messages[-1].content  # 问了


class LimitedVendor:
    """并发上限是 limit 的假模型服务（实现 LLMClient，自己不加锁）：路由的几票要同时在途才放行
    （Barrier，等不齐就是没有并发发出），超过上限的那几票回 429。"""

    model = "synthetic-llm"

    def __init__(self, limit: int, votes: int = 3) -> None:
        self.limit = limit
        self.calls: list[LLMRequest] = []
        self._barrier = threading.Barrier(votes)
        self._lock = threading.Lock()
        self._in_flight = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            self.calls.append(request)
        if request.purpose != "route":
            return LLMResponse(text=json.dumps(ANSWERS[request.purpose]), model=self.model)
        with self._lock:
            self._in_flight += 1
            position = self._in_flight
        self._barrier.wait(timeout=10)
        if position > self.limit:
            raise LLMError(
                "大模型接口限流（HTTP 429）",
                status=429,
                kind="rate_limit",
                code="rate_limit_exceeded",
            )
        return LLMResponse(text=json.dumps(ROUTE_OK), model=self.model)


@pytest.mark.parametrize(("limit", "lost"), [(1, 2), (2, 1)])
def test_http_429_on_concurrent_votes(limit: int, lost: int):
    """账号的并发上限低于一条消息要的三票：limit 2 时只丢一票、判断层照样答得出，也要报出来。"""
    vendor = LimitedVendor(limit)
    lines = _failure(vendor, timer=ThreadClock()).splitlines()
    assert "路由没过，评测没有开跑" in lines[0]
    cause = "被限流（HTTP 429，错误代码 rate_limit_exceeded）——大模型接口限流（HTTP 429）"
    assert f"    - {lost} 票：{cause}" in lines
    assert "  质检（1 次）：通过，耗时 0.00 秒，单次超时 1 秒（decision.timeout_s）" in lines
    hint, occasional = lines[lines.index("下一步：") + 1 :]
    assert "并发或速率上限低于一条消息要的量（路由 3 票同时发）" in hint
    assert "decision.vote_samples" in hint and "余额" in hint
    assert "有的调用过了、有的没过" in occasional  # 另外几票是通过的
    assert len(vendor.calls) == 5


def test_401_names_the_key_variable_and_never_the_key(monkeypatch):
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)
    fake = _fake(lambda r: LLMError("大模型接口拒绝请求（HTTP 401）", status=401, kind="auth"))
    fake.api_key_env = "RODIT_LLM_API_KEY"  # type: ignore[attr-defined]
    message = _failure(fake)
    assert "路由、质检、生成没过" in message
    assert "- 3 票：密钥无效或没有权限（HTTP 401）——大模型接口拒绝请求（HTTP 401）" in message
    assert "检查环境变量 RODIT_LLM_API_KEY 里的密钥是不是这个接口、这个模型的" in message
    assert KEY not in message and len(fake.calls) == 5


def test_real_client_errors_are_reported_without_key_or_credentials(monkeypatch):
    """真实客户端 + 脚本化的 transport：接口拒绝 response_format 时三种请求都没过；说明里没有密钥，
    接口地址只留主机。（客户端把 HTTP 状态码放进 LLMError.status 之前，这里只有它给的说明文字。）"""
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)
    refusal = {"error": {"type": "invalid_request_error", "code": "response_format_unsupported"}}
    vendor = Vendor(lambda purpose, body: (400, refusal) if "response_format" in body else None)
    client = vendor.client("https://user:pw@llm.example.invalid:8443/v1")
    message = _failure(client)
    assert "路由、质检、生成没过，评测没有开跑" in message.splitlines()[0]
    assert "- 3 票：" in message and message.count("- 1 次：") == 2
    assert KEY not in message and "user:pw" not in message
    assert len(vendor.seen) == 5


def test_real_client_timeouts_show_the_configured_limit(monkeypatch):
    """真实客户端：每次请求要 1.2 秒（假时钟），传输层按请求自带的超时掐断——判断层的两种请求
    超时，生成来得及；客户端默认的超时（这里 30 秒，即 RODIT_LLM_TIMEOUT_S）没有用上。"""
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)
    clock = ThreadClock()
    vendor = Vendor(clock=clock, seconds=1.2)
    client = OpenAICompatibleClient(
        base_url=BASE_URL, model="synthetic-model", transport=vendor, timeout_s=30.0
    )
    lines = _failure(client, timer=clock).splitlines()
    assert "路由、质检没过，评测没有开跑" in lines[0]
    assert (
        "  路由（3 票并发）：没过，耗时 1.00 / 1.00 / 1.00 秒，单次超时 1 秒（decision.timeout_s）"
        in lines
    )
    assert "  质检（1 次）：没过，耗时 1.00 秒，单次超时 1 秒（decision.timeout_s）" in lines
    assert "  生成（1 次）：通过，耗时 1.20 秒，单次超时 2.5 秒（generation.timeout_s）" in lines
    assert sum(line.startswith("    - 3 票：超时") for line in lines) == 1
    assert "RODIT_LLM_TIMEOUT_S 管不到这些请求" in lines[-1]
    assert sorted(t for _, _, t in vendor.seen) == [1.0, 1.0, 1.0, 1.0, 2.5]


@pytest.mark.parametrize("content", ["", None], ids=["empty-string", "null"])
def test_a_reasoning_model_that_returns_no_text_is_stopped_here(monkeypatch, content):
    """思考内容把 max_tokens 用完的模型：正文是空串时调用算成功（用量里看不出失败），是 null 时
    客户端报"没有文本内容"——两种都在这里拦下，不等整轮跑完才发现全是降级。"""
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)
    payload = {
        "model": "synthetic-model",
        "choices": [
            {
                "message": {"content": content, "reasoning_content": "合成的思考过程……"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 304},
    }
    vendor = Vendor(lambda purpose, body: (200, payload))
    message = _failure(vendor.client())
    assert "路由、质检、生成没过，评测没有开跑" in message.splitlines()[0]
    assert len(vendor.seen) == 5
    if content == "":
        assert "正文是空的" in message and "llm.extra_body" in message


@pytest.mark.parametrize(
    ("error", "label", "step"),
    [
        (
            LLMError("合成的故障", status=402),
            "余额或额度用完（HTTP 402）",
            "充值或换一个有额度的账号",
        ),
        (LLMError("合成的故障", status=403), "密钥无效或没有权限（HTTP 403）", "密钥的环境变量"),
        (LLMError("合成的故障", status=404), "请求被拒（HTTP 404）", "模型名不对"),
        (LLMError("合成的故障", status=429), "被限流（HTTP 429）", "decision.vote_samples"),
        (LLMError("合成的故障", status=503), "模型服务出错（HTTP 503）", "稍后再试"),
        (LLMError("合成的故障", status=302), "返回了重定向（HTTP 302）", "RODIT_LLM_BASE_URL"),
        # 抛出的地方给了 kind 就用它：有的服务额度用完也回 429
        (
            LLMError("合成的故障", status=429, kind="billing", code="insufficient_quota"),
            "余额或额度用完（HTTP 429，错误代码 insufficient_quota）",
            "充值或换一个有额度的账号",
        ),
        (LLMError("合成的故障", kind="network"), "连不上——合成的故障", "NO_PROXY"),
        (LLMError("合成的故障", kind="malformed"), "响应读不出来——合成的故障", "llm.extra_body"),
        (LLMError("合成的故障", kind="content_filter"), "被内容安全拦截——", "内容安全"),
        (LLMError("合成的故障", kind="config"), "本地配置不对——", "请求没有发出去"),
        (LLMTimeout("合成的故障", status=504), "超时（HTTP 504）", "RODIT_LLM_TIMEOUT_S 管不到"),
        (LLMError("合成的故障"), "调用失败——合成的故障", "客户端没有给出 HTTP 状态码或错误代码"),
        (LLMError("合成的故障", status=True), "调用失败——合成的故障", "客户端没有给出"),  # type: ignore[arg-type]
    ],
    ids=lambda value: None if isinstance(value, str) else f"{value.status}-{value.kind}",
)
def test_the_cause_comes_from_kind_then_status_and_never_from_the_message(
    error: LLMError, label: str, step: str
):
    lines = _failure(_fake(lambda request: error)).splitlines()
    assert any(line.startswith(f"    - 3 票：{label}") for line in lines), lines
    (hint,) = lines[lines.index("下一步：") + 1 :]
    assert step in hint


@pytest.mark.parametrize("kind", LLM_ERROR_KINDS)
def test_every_failure_kind_has_a_label_and_one_next_step(kind: str):
    """core.errors.LLMErrorKind 里的每一类都报得出来：有中文说明，有且只有一条下一步。"""
    error = LLMError("合成的故障", kind=kind)  # type: ignore[arg-type]
    lines = _failure(_fake(lambda request: error)).splitlines()
    (problem,) = [line for line in lines if line.startswith("    - 3 票：")]
    assert problem.endswith("——合成的故障") and "调用失败" not in problem
    (hint,) = lines[lines.index("下一步：") + 1 :]
    assert hint.startswith("  - ") and hint.endswith("。")


def test_errors_that_are_not_llm_errors_are_not_swallowed():
    """LLMError 以外的异常是程序错误（或者替身没写这种请求），照常抛出来，不当成接口的问题报。"""
    with pytest.raises(ZeroDivisionError):
        run_preflight(_fake(lambda request: ZeroDivisionError("合成的程序错误")))


# ================================================================ 只查路由（判断层对比评测）


def _first_route_request() -> Any:
    return compare.build_compare_cases(cases("all")[:1])[0].request


def test_route_only_check_sends_the_given_request_through_the_llm_judge():
    """python -m rodit.eval.compare --llm env 只调路由判断层（不管 decision.provider 写的是什么），
    请求里是四个必答问题、没有追问：检查原样发这一份，不查质检和生成。"""
    request = _first_route_request()
    config = _config(decision={"provider": "classifier", "vote_samples": 5, "timeout_s": 2.0})
    fake = _fake(_run_model)
    summary = run_preflight(fake, config, route_request=request, timer=ThreadClock())
    assert [r.purpose for r in fake.calls] == ["route"] * 5
    prompt = fake.calls[0].messages[-1].content
    assert "### intent" in prompt and "### second_intent" not in prompt
    assert json.dumps(request.state["messages"][-1]["text"], ensure_ascii=False) in prompt
    assert {r.timeout_s for r in fake.calls} == {2.0}
    assert summary[0].startswith("大模型连通性检查通过") and "5 次调用" in summary[0]
    assert summary[1].startswith("  路由（5 票并发）：") and len(summary) == 3  # 路由、网络路径

    four_questions = {k: v for k, v in ROUTE_OK.items() if k != "second_intent"}
    partial = {k: v for k, v in four_questions.items() if k != "emotion"}
    message = _failure(_fake(scripted(answers={"route": partial})), route_request=request)
    assert "按这次评测要发的请求试了 1 种、3 次调用（合成数据），路由没过" in message
    assert "- 3 票：没有答出 emotion" in message


def test_compare_cli_checks_routing_with_the_first_request_of_the_run():
    fake = _fake(_run_model)
    out, err = io.StringIO(), io.StringIO()
    code = compare.main(
        ["--llm", "env", "--limit", "3", "--json"], out, err=err, client_factory=lambda: fake
    )
    assert code == 0, err.getvalue()
    log = err.getvalue()
    assert "大模型连通性检查通过：按这次评测要发的请求试了 3 次调用" in log
    assert (
        log.index("  路由（3 票并发）：")
        < log.index("  网络路径：")
        < log.index("[incloud_llm 1/3]")
    )
    assert "质检" not in log and "生成" not in log
    assert [r.purpose for r in fake.calls] == ["route"] * (3 + 3 * 3)
    assert fake.calls[0].messages == fake.calls[3].messages  # 检查发的就是第一条用例的请求
    assert json.loads(out.getvalue())["providers"][1]["info"]["name"] == "incloud_llm"

    broken = _fake(lambda r: LLMError("大模型接口拒绝请求（HTTP 401）", status=401, kind="auth"))
    out, err = io.StringIO(), io.StringIO()
    code = compare.main(
        ["--llm", "env", "--limit", "3"], out, err=err, client_factory=lambda: broken
    )
    assert code == 2 and out.getvalue() == ""
    assert "对比评测没有运行：大模型连通性检查失败" in err.getvalue()
    assert "路由没过" in err.getvalue() and "Traceback" not in err.getvalue()
    assert len(broken.calls) == 3  # 只有检查的三票，没有开跑


# ================================================================ 评测命令行


def test_eval_cli_prints_the_summary_before_the_first_case_and_keeps_it_out_of_usage():
    fake = _fake(_run_model)
    out, err = io.StringIO(), io.StringIO()
    code = eval_main(
        ["--llm", "env", "--limit", "1", "--json"], out=out, err=err, client_factory=lambda: fake
    )
    assert code == 0, err.getvalue()
    log = err.getvalue()
    assert log.index("大模型连通性检查通过") < log.index("  网络路径：") < log.index("[1/1]")
    assert json.loads(out.getvalue())["usage"]["calls"] == len(fake.calls) - 5


def test_eval_cli_reports_the_failed_check_and_does_not_run():
    def cut(request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="", model="synthetic-llm", finish_reason="length")

    fake = _fake(cut)
    out, err = io.StringIO(), io.StringIO()
    code = eval_main(
        ["--llm", "env", "--limit", "1"], out=out, err=err, client_factory=lambda: fake
    )
    assert code == 2 and out.getvalue() == ""
    log = err.getvalue()
    assert log.startswith("评测没有运行：大模型连通性检查失败：")
    assert "llm.extra_body" in log and "Traceback" not in log
    assert len(fake.calls) == 5


# ================================================================ 是否经代理


@pytest.fixture
def proxy_env(monkeypatch):
    """先清掉这台机器上的代理变量（大小写都清），再按用例设。"""
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.delenv("REQUEST_METHOD", raising=False)

    def setenv(values: Mapping[str, str]) -> None:
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    return setenv


def _http_client(base_url: str = BASE_URL) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(base_url=base_url, model="synthetic-model")  # 只构造，不发请求


def _urllib_sets_a_proxy(url: str) -> bool:
    """urllib 默认的 ProxyHandler（客户端的默认传输层用的就是它）会不会给这个地址的请求设代理。
    只走它的判断，不发请求。"""
    request = urllib.request.Request(url)
    host = request.host
    proxy_open = getattr(urllib.request.ProxyHandler(), f"{request.type}_open", None)
    if proxy_open is not None:
        proxy_open(request)
    return request.host != host


@pytest.mark.parametrize(
    ("env", "base_url", "expected"),
    [
        ({"HTTPS_PROXY": PROXY}, BASE_URL, True),
        ({"https_proxy": PROXY}, BASE_URL, True),
        ({"HTTPS_PROXY": PROXY, "NO_PROXY": "localhost,127.0.0.1"}, BASE_URL, True),
        ({"HTTPS_PROXY": PROXY, "NO_PROXY": "llm.example.invalid"}, BASE_URL, False),
        ({"HTTPS_PROXY": PROXY, "NO_PROXY": ".example.invalid"}, BASE_URL, False),
        ({"HTTPS_PROXY": PROXY, "NO_PROXY": "*"}, BASE_URL, False),
        # 小写的 no_proxy 优先：只在大写的里写 * 不管用
        ({"HTTPS_PROXY": PROXY, "NO_PROXY": "*", "no_proxy": "localhost"}, BASE_URL, True),
        (
            {"HTTPS_PROXY": PROXY, "NO_PROXY": "llm.example.invalid"},
            "https://llm.example.invalid:8443/v1",
            False,
        ),
        # 地址里带用户名密码时 urllib 拿 "user:pw@主机" 去对免代理名单：整名对不上，后缀对得上
        (
            {"HTTPS_PROXY": PROXY, "NO_PROXY": "llm.example.invalid"},
            "https://user:pw@llm.example.invalid/v1",
            True,
        ),
        (
            {"HTTPS_PROXY": PROXY, "NO_PROXY": ".example.invalid"},
            "https://user:pw@llm.example.invalid/v1",
            False,
        ),
        # 只给 http 配了代理：https 的接口直连
        ({"HTTP_PROXY": PROXY}, BASE_URL, False),
        ({"HTTP_PROXY": PROXY}, "http://10.0.0.8:8080/v1", True),
        ({"HTTP_PROXY": PROXY, "NO_PROXY": "10.0.0.8"}, "http://10.0.0.8:8080/v1", False),
    ],
)
def test_via_proxy_follows_the_environment_like_urllib(
    proxy_env, env: dict[str, str], base_url: str, expected: bool
):
    proxy_env(env)
    client = _http_client(base_url)
    assert _urllib_sets_a_proxy(client.endpoint) is expected  # 表里写的就是 urllib 实际的行为
    assert via_proxy(client) is expected


def test_via_proxy_is_false_without_any_proxy(proxy_env, monkeypatch):
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})  # 环境变量、系统设置里都没有
    assert via_proxy(_http_client()) is False


def test_via_proxy_is_unknown_when_it_cannot_tell(proxy_env):
    proxy_env({"HTTPS_PROXY": PROXY})
    assert via_proxy(Vendor().client()) is None  # 注入了 transport
    fake = _fake()
    fake.base_url = BASE_URL  # type: ignore[attr-defined]  # 有地址，但不是 HTTP 客户端
    assert via_proxy(fake) is None
    assert via_proxy(object()) is None


class _Wired(OpenAICompatibleClient):
    """默认传输层的真实客户端，只把 complete 换成假模型（不发请求）。"""

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text=json.dumps(ANSWERS[request.purpose]), model=self.model)


@pytest.mark.parametrize(
    ("env", "line"),
    [
        (
            {"HTTPS_PROXY": PROXY},
            "  网络路径：经环境里配置的代理访问 https://llm.example.invalid，上面的耗时含代理这一段；"
            "要直连就把这个主机加进 NO_PROXY（同时有小写的 no_proxy 时以小写的为准）",
        ),
        (
            {"HTTPS_PROXY": PROXY, "NO_PROXY": "llm.example.invalid"},
            "  网络路径：直连 https://llm.example.invalid（没有经代理）",
        ),
    ],
    ids=["proxied", "direct"],
)
def test_the_network_path_line_never_shows_the_proxy_address(proxy_env, env, line):
    proxy_env(env)
    client = _Wired(base_url="https://llm.example.invalid:443/v1", model="synthetic-model")
    summary = run_preflight(client)
    assert summary[-1] == line
    failed = _failure(_Unreachable(base_url=client.base_url, model="synthetic-model"))
    assert line in failed.splitlines()  # 没过的说明里也有这一行
    for text in ("\n".join(summary), failed):
        assert "proxy.example.invalid" not in text and "proxy-secret" not in text


class _Unreachable(_Wired):
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMError("合成的故障：连接被拒", kind="network")
