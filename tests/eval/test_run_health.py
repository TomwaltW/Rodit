"""python -m rodit.eval --llm env：运行健康度、退出码、如实标注和收尾
（docs/架构与分工.md §4.4，任务 C2）。

对应方案 §13.2、§13.3、§12.1、§12.2、§14。调用失败、被截断、读不出时链路照常降级、评测照常跑完，
但那组数字量的是兜底逻辑，不是模型。这里用脚本化的假模型（按用途、按第几条用例、第几票作答）
造出各种"模型没被评到"的一轮，核对报告里的 health、文本报告的提示、退出码和 JSON；另有校准只用
判断层路由的用例、耗时和用量如实标注、输出路径先验证、中断和连续失败的处理、RODIT_CONFIG。

不联网，数据全部合成。固定抽 12 条（--limit 12 --seed 3）：dev、test 各 6 条，其中 5 条要调大模型
生成、送判断层质检；12 条里容许出问题的是 1 条（health.allowed(12, 0.02)）。
这里的故障都算开跑之后才出现的（一开始就有的问题归连通性检查管，rodit.eval.llm_env.preflight），
所以一律带 --skip-preflight：不依赖连通性检查发什么请求；只有"一次调用都不发"的那几条不带。
失败原因的标签（health.error_label）要等 llm.usage 的 detail 带上 kind / status / code 才细分
（rodit.llm.metering，任务 A）；在那之前是异常类型名。这里两种都认，只核对它确实来自 detail。
"""

from __future__ import annotations

import io
import json
import os
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rodit.app.offline import EvidenceEchoLLM
from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.clock import CST
from rodit.core.config import EngineConfig, LatencyConfig, LLMConfig
from rodit.core.enums import Intent
from rodit.core.errors import LLMError, LLMTimeout
from rodit.core.models import AuditEvent
from rodit.core.pii import mask_pii
from rodit.core.ports import LLMMessage, LLMRequest, LLMResponse
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
)
from rodit.engine import AssistEngine
from rodit.eval import runner
from rodit.eval.cli import (
    DEFAULTS_WARNING,
    STOP_AFTER_FAILED_CASES,
    build_parser,
    env_client,
    evaluate,
    main,
    render_text,
)
from rodit.eval.dataset import CASES, cases
from rodit.eval.harness import CallRecorder, llm_harness
from rodit.eval.health import DEFAULT_CEILING, RunHealth, allowed, assess, error_label
from rodit.eval.runner import (
    BACKGROUND_WAIT_S,
    CallStats,
    CaseTrace,
    EvalAborted,
    select_cases,
    trace_case,
)
from rodit.testing import fixtures
from rodit.testing.fakes import FakeLLMClient

MODEL = "synthetic-llm"
SAMPLE = ["--limit", "12", "--seed", "3"]
ARGV = ["--llm", "env", "--skip-preflight", *SAMPLE]
WITH_PREFLIGHT = ["--llm", "env", *SAMPLE]
PICKED = select_cases(cases("all"), limit=12, seed=3)
ORDER = {mask_pii(c.text): i for i, c in enumerate(PICKED)}
EXPECTED = {mask_pii(c.text): c.expected_intent for c in CASES}
DEV = [i for i, c in enumerate(PICKED) if c.split == "dev"]
TEST = [i for i, c in enumerate(PICKED) if c.split == "test"]
INVALID = "!! 这一轮没有有效评到模型"
QA_PASS = {
    Q_GROUNDED: {"answer": "是", "p": 0.95},
    Q_ANSWERS_QUESTION: {"answer": "是", "p": 0.95},
    Q_PROMISE_RISK: {"answer": "否", "p": 0.05},
    Q_PLATFORM_VIOLATION: {"answer": "否", "p": 0.02},
    Q_TONE: {"score": 3},
}

Fault = Callable[[LLMRequest, int, int], Any]
"""fault(请求, 第几条用例（从 0 数）, 同一条请求第几次出现) → 代替正常回答的结果；None 表示照常
回答。路由的三票是同一条请求（第 1、2、3 次），所以"每条用例丢一票"这样的安排和线程调度无关。"""


@pytest.fixture(autouse=True)
def _no_deployment_config(monkeypatch):
    """这里的评测都按内置默认值跑：不受运行环境里 RODIT_CONFIG 的影响（--llm env 会读它）。"""
    monkeypatch.delenv("RODIT_CONFIG", raising=False)


def test_the_sample_is_what_the_scenarios_assume():
    assert len(ORDER) == len(PICKED) == 12  # 按买家消息认用例：不能有重复
    assert len(DEV) == len(TEST) == 6
    assert allowed(12, DEFAULT_CEILING) == allowed(6, DEFAULT_CEILING) == 1


def _buyer_text(request: LLMRequest) -> str:
    """路由请求里的当前买家消息（已脱敏）。"""
    user = request.messages[-1].content
    head = "## 状态（state）\n"
    state = json.loads(user[user.index(head) + len(head) : user.index("\n\n## 问题")])
    return state["messages"][-1]["text"]


class Vendor:
    """脚本化的模型服务（FakeLLMClient 的 handler）。默认每次都答得上：路由答用例标注的类型、
    质检全部通过、生成复述依据（离线替身 EvidenceEchoLLM 的文字）。fault 见上。
    FakeLLMClient 在锁里调 handler，这里不用再加锁。"""

    def __init__(self, fault: Fault | None = None) -> None:
        self.fault = fault
        self.echo = EvidenceEchoLLM()
        self.case = -1  # 正在跑第几条用例：从路由请求里的买家消息认出来
        self.seen: Counter[str] = Counter()

    def __call__(self, request: LLMRequest) -> Any:
        if request.purpose == "other":  # 连通性检查
            return "OK"
        if request.purpose == "route":
            self.case = ORDER.get(_buyer_text(request), -1)
        key = request.purpose + "".join(m.content for m in request.messages)
        self.seen[key] += 1
        if self.fault is not None:
            result = self.fault(request, self.case, self.seen[key])
            if result is not None:
                return result
        return self.answer(request)

    def answer(self, request: LLMRequest) -> Any:
        if request.purpose == "route":
            intent = EXPECTED.get(_buyer_text(request), Intent.OTHER)
            return {
                Q_INTENT: {"choice": intent.value},
                Q_NEED_HUMAN: {"answer": "否", "p": 0.05},
                Q_EMOTION: {"score": 0},
                Q_MULTI_INTENT: {"answer": "否", "p": 0.05},
            }
        if request.purpose == "qa":
            return dict(QA_PASS)
        return self.echo.complete(request).text


@dataclass
class Run:
    code: int
    out: str
    err: str
    data: dict[str, Any] | None  # --json 写出的完整结果；没写出来为 None
    client: FakeLLMClient

    @property
    def health(self) -> RunHealth:
        assert self.data is not None
        return RunHealth.model_validate(self.data["health"])

    def cases_where(self, **fields: Any) -> list[dict[str, Any]]:
        assert self.data is not None
        return [c for c in self.data["cases"] if all(c[k] == v for k, v in fields.items())]


def _run(tmp_path: Path, fault: Fault | None = None, *extra: str) -> Run:
    client = FakeLLMClient(handler=Vendor(fault), model=MODEL)
    target = tmp_path / "report.json"
    out, err = io.StringIO(), io.StringIO()
    argv = [*ARGV, *extra, "--json", str(target)]
    code = main(argv, out=out, err=err, client_factory=lambda: client)
    data = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
    return Run(code, out.getvalue(), err.getvalue(), data, client)


def _rejected(status: int, kind: str, code: str | None = None) -> LLMError:
    """模型服务用这个 HTTP 状态码拒绝了请求（失败原因按契约带在异常上）。"""
    return LLMError(f"合成的故障：HTTP {status}", status=status, kind=kind, code=code)  # type: ignore[arg-type]


def _check_rule(run: Run) -> RunHealth:
    """报告里的 valid / reasons 就是 health.assess 按这次的上限算出来的，和文本、退出码一致。"""
    health = run.health
    assert health.reasons == assess(health.overall, health.test, ceiling=health.ceiling)
    assert health.valid is (not health.reasons)
    lines = run.out.splitlines()
    for reason in health.reasons:
        assert lines.count(f"!!   {reason}") == 2  # 开头一次，上线门槛旁边一次
    return health


# ================================================================ 算数的一轮


def test_a_healthy_run_is_valid_and_exits_zero(tmp_path):
    run = _run(tmp_path)
    assert run.code == 0, run.err
    health = _check_rule(run)
    assert health.valid and health.reasons == [] and health.ceiling == DEFAULT_CEILING
    overall = health.overall
    assert (overall.cases, overall.route_fallback, overall.route_lost_votes) == (12, 0, 0)
    assert (overall.qa_judged, overall.qa_unavailable) == (5, 0)
    assert (overall.generation_attempted, overall.generation_unusable) == (5, 0)
    assert (overall.engine_errors, overall.budget_cut) == (0, 0)
    assert health.test is not None and health.test.cases == 6
    assert health.routed_by == {"incloud_llm": 12} and health.late_calls == 0
    route = health.by_purpose["route"]
    assert (route.calls, route.failed, route.timeouts) == (36, 0, 0)
    assert (route.truncated, route.empty, route.unusable, route.errors) == (0, 0, 0, {})
    assert list(health.by_purpose) == ["route", "generate", "qa"]
    assert sum(p.calls for p in health.by_purpose.values()) == run.data["usage"]["calls"]
    # 逐条：实际做出路由的判断层、发出的票和算得上的票
    assert len(run.cases_where(route_provider="incloud_llm", route_votes=3)) == 12
    assert all(c["route_votes_usable"] == 3 for c in run.data["cases"])
    # 文本：开头一行关键计数，没有"不算数"的提示；健康度一节每次都有
    lines = run.out.splitlines()
    assert lines[3].startswith("运行健康度：有效（上限 2%）。全部 12 条：路由退回关键词规则 0、")
    assert INVALID not in run.out and "!!" not in run.out
    assert "== 运行健康度（这一轮算不算评到了模型：上限 2%，路由丢票上限 10%" in run.out
    assert "  路由：调用 36，失败 0；有响应的里面被截断 0、为空 0、用不上 0" in run.out
    assert "实际做出路由的判断层：incloud_llm 12 条" in run.out
    assert "类型准确率分开看：判断层路由 12 条 100.0%；退回关键词规则 0 条；" in run.out
    assert "路由丢票的用例" not in run.out and "没有参与校准" not in run.out
    assert "退出码 3" not in run.err


def test_offline_runs_have_no_health_block():
    """离线替身不调大模型：没有 health，文本报告里也没有这一节，退出码照旧。"""
    report = evaluate("test", limit=6)
    assert report.health is None
    text = render_text(report)
    assert "运行健康度" not in text and "!!" not in text
    assert all(c.route_provider == "rules" and c.route_votes is None for c in report.cases)


# ================================================================ 不算数的一轮


def test_every_routing_call_rejected(tmp_path):
    """路由请求全部被拒（如接口不认某个参数，HTTP 400）：每条都退回关键词规则。"""

    def fault(request: LLMRequest, case: int, nth: int):
        if request.purpose == "route":
            return _rejected(400, "bad_request", "unsupported_parameter")
        return None

    run = _run(tmp_path, fault)
    assert run.code == 3
    health = _check_rule(run)
    assert not health.valid
    assert health.reasons == [
        "路由：12 条里有 12 条（100.0%）没用上判断层（退回关键词规则），上限 1 条。",
        "测试集：路由：6 条里有 6 条（100.0%）没用上判断层（退回关键词规则），上限 1 条。",
    ]
    assert health.overall.route_fallback == 12 and health.overall.route_lost_votes == 0
    assert health.routed_by == {"rules": 12}
    route = health.by_purpose["route"]
    assert (route.calls, route.failed, route.unusable) == (36, 36, 0)
    # 失败原因来自 llm.usage 的 detail：带了状态码就是 http_400:<代码>，否则是异常类型名
    assert sum(route.errors.values()) == 36
    assert set(route.errors) <= {"http_400:unsupported_parameter", "LLMError"}
    assert run.data["usage"]["error_types"] == route.errors
    assert health.by_purpose["generate"].failed == health.by_purpose["qa"].failed == 0
    # 报告头还写着配置的判断层，逐条和健康度写的是实际的
    assert run.data["llm"]["decision_provider"] == "incloud_llm"
    assert len(run.cases_where(route_provider="rules")) == 12
    # 文本：开头和上线门槛旁边各提示一次，原因逐条列出
    assert run.out.count(INVALID) == 2
    lines = run.out.splitlines()
    assert lines[3].startswith(INVALID) and lines[4] == f"!!   {health.reasons[0]}"
    gate = next(i for i, line in enumerate(lines) if line.startswith("== §13.2 上线门槛"))
    assert lines[gate + 1].startswith("测试集 n=6：") and lines[gate + 2].startswith(INVALID)
    assert "兜底逻辑" in lines[3] and "不能当成模型的评测结果" in lines[3]
    assert "实际做出路由的判断层：rules 12 条" in run.out
    assert "运行健康度：有效" not in run.out
    # 标准错误：进度里看得到降级，结尾说明退出码
    assert "降级：decision" in run.err and "这一轮没有有效评到模型（退出码 3）" in run.err


def _forty_percent(request: LLMRequest, case: int, nth: int):
    """四成的路由票被限流（HTTP 429），按用例和票排定：每 5 条里两条丢两票以上（退回关键词
    规则）、两条丢一票（判断层照常给出结果）、一条不丢。"""
    lost = [(1, 2), (3,), (), (1,), (2, 3)][case % 5]
    if request.purpose == "route" and nth in lost:
        return _rejected(429, "rate_limit")
    return None


def test_forty_percent_of_routing_votes_rate_limited(tmp_path):
    run = _run(tmp_path, _forty_percent)
    assert run.code == 3
    health = _check_rule(run)
    fallback = [i for i in range(12) if i % 5 in (0, 4)]  # 0、4、5、9、10
    one_lost = [i for i in range(12) if i % 5 in (1, 3)]  # 1、3、6、8、11
    assert health.overall.route_fallback == len(fallback) == 5
    assert health.overall.route_lost_votes == len(one_lost) == 5
    assert health.test is not None
    assert health.test.route_fallback == len(set(fallback) & set(TEST)) == 3
    assert health.test.route_lost_votes == len(set(one_lost) & set(TEST)) == 2
    assert health.routed_by == {"incloud_llm": 7, "rules": 5}
    route = health.by_purpose["route"]
    assert (route.calls, route.failed) == (36, 15)  # 36 票里丢 15 票：约四成
    assert sum(route.errors.values()) == 15 and set(route.errors) <= {"http_429", "LLMError"}
    assert [r.split("（")[0] for r in health.reasons] == [
        "路由：12 条里有 5 条",  # 退回关键词规则
        "路由：12 条里有 5 条",  # 丢票
        "测试集：路由：6 条里有 3 条",
        "测试集：路由：6 条里有 2 条",
    ]
    # 逐条：退回的那几条是关键词规则路由的；丢一票的还是判断层，两票算数
    ids = [c.id for c in PICKED]
    assert [c["id"] for c in run.cases_where(route_provider="rules")] == [ids[i] for i in fallback]
    lost = run.cases_where(route_provider="incloud_llm", route_votes_usable=2)
    assert [c["id"] for c in lost] == [ids[i] for i in one_lost]
    assert run.data["degraded_counts"] == {"decision": 5}
    # 文本：两种路由的类型准确率分开，丢票的用例列出来
    assert "类型准确率分开看：判断层路由 7 条 " in run.out and "；退回关键词规则 5 条 " in run.out
    assert "路由丢票的用例 5 条（算得上的票 / 发出的票；" in run.out
    assert all(f"{ids[i]} 2/3" in run.out for i in one_lost)
    assert "失败的 15 次调用按 0 token 计" in run.out


def test_calibration_uses_only_judge_routed_cases(tmp_path):
    """§13.3：退回关键词规则的用例不参与校准和可靠性分箱（置信度不是一个刻度），报告写明排除了几条。"""
    run = _run(tmp_path, _forty_percent)
    fallback = {i for i in range(12) if i % 5 in (0, 4)}
    cal, rel = run.data["calibration"], run.data["reliability"]
    assert cal["samples_split"] == "dev"
    assert cal["left_out_fallback"] == len(fallback & set(DEV)) == 2
    assert cal["n_samples"] == len(DEV) - 2 and cal["left_out_no_route"] == 0
    assert (rel["n_samples"], rel["left_out_fallback"], rel["left_out_no_route"]) == (7, 5, 0)
    assert sum(b["count"] for b in rel["bins"]) == 7
    # 判断层的置信度是投票占比（三票全中 1.0，两票 1.0）；关键词规则的 0.4–0.9 没有混进来
    judge = [c for c in run.data["cases"] if c["route_provider"] == "incloud_llm"]
    rules = [c for c in run.data["cases"] if c["route_provider"] == "rules"]
    assert {c["intent_confidence"] for c in judge} == {1.0}
    assert all(c["intent_confidence"] < 1.0 for c in rules)
    assert [(b["lower"], b["count"]) for b in rel["bins"]] == [(0.8, 7)]
    assert "用 dev 的 4 条路由结果校准" in run.out
    assert (
        "只用配置的判断层做出的路由：dev 里另有 2 条退回了关键词规则"
        "（它的置信度和判断层的不是一个刻度，不能混在一起），没有参与校准"
    ) in run.out
    assert "可靠性（7 条，不含没参与的 5 条）：ECE = " in run.out
    assert "注意：有 5 条用例路由丢了票（见运行健康度）" in run.out
    # 没有退回的一轮：一条都不排除，也不多一行说明
    healthy = _run(tmp_path)
    assert healthy.data["calibration"]["n_samples"] == 6
    assert healthy.data["calibration"]["left_out_fallback"] == 0
    assert "没有参与校准" not in healthy.out and "可靠性（12 条）：" in healthy.out


def test_blank_truncated_routing_answers_without_a_failed_call(tmp_path):
    """路由每次都是 HTTP 200，但文本为空、finish_reason 是 length（思考把 max_tokens 用完了）：
    没有一次调用失败，每条照样退回关键词规则。"""

    def fault(request: LLMRequest, case: int, nth: int):
        if request.purpose == "route":
            return LLMResponse(
                text="", model=MODEL, input_tokens=900, output_tokens=304, finish_reason="length"
            )
        return None

    run = _run(tmp_path, fault)
    assert run.code == 3
    health = _check_rule(run)
    route = health.by_purpose["route"]
    assert (route.calls, route.failed, route.errors) == (36, 0, {})
    assert (route.truncated, route.empty, route.unusable) == (36, 36, 36)
    assert health.overall.route_fallback == 12 and health.routed_by == {"rules": 12}
    assert run.data["usage"]["errors"] == 0  # 用量那一节看不出任何问题
    assert "  路由：调用 36，失败 0；有响应的里面被截断 36、为空 36、用不上 36" in run.out
    assert run.out.count(INVALID) == 2
    assert "（失败" not in run.err and run.err.count("降级：decision") == 12  # 进度里只有降级标记


def test_one_unreadable_vote_per_case_leaves_no_degraded_code(tmp_path):
    """每条用例有一票读不出来（模型答了一句话，不是 JSON）：没有失败的调用、没有任何降级标记，
    路由仍然出自判断层，只是三票变两票。"""

    def fault(request: LLMRequest, case: int, nth: int):
        if request.purpose == "route" and nth == 1:
            return "好的，我看了一下，这位买家是在问物流。"
        return None

    run = _run(tmp_path, fault)
    assert run.code == 3
    health = _check_rule(run)
    assert run.data["degraded_counts"] == {} and "== 降级" not in run.out
    assert run.data["usage"]["errors"] == 0
    assert health.overall.route_fallback == 0 and health.routed_by == {"incloud_llm": 12}
    assert health.overall.route_lost_votes == 12
    route = health.by_purpose["route"]
    assert (route.calls, route.failed, route.unusable) == (36, 0, 12)
    assert (route.truncated, route.empty) == (0, 0)
    lost = "至少有一票没算上（调用失败、被截断、读不出），上限 1 条。"
    assert health.reasons == [
        f"路由：12 条里有 12 条（100.0%）{lost}",
        f"测试集：路由：6 条里有 6 条（100.0%）{lost}",
    ]
    assert len(run.cases_where(route_votes=3, route_votes_usable=2)) == 12
    assert "路由丢票的用例 12 条" in run.out and f"{PICKED[0].id} 2/3" in run.out
    assert "降级" not in run.err  # 进度行里也没有：只有健康度看得出来
    # 丢票的上限比退回关键词规则的宽（10%）：12 条里丢 1 条不算问题
    one = _run(
        tmp_path, lambda r, case, nth: "嗯" if (r.purpose, case, nth) == ("route", 0, 1) else None
    )
    assert one.code == 0 and one.health.overall.route_lost_votes == 1 and one.health.valid


def test_every_generation_output_unparseable(tmp_path):
    """生成每次都是 HTTP 200，但不是 §5.4 的 JSON：调用没有失败，引擎也不记降级，
    只是没有一条可用草稿。"""

    def fault(request: LLMRequest, case: int, nth: int):
        return "抱歉，我没办法按这个格式回答。" if request.purpose == "generate" else None

    run = _run(tmp_path, fault)
    assert run.code == 3
    health = _check_rule(run)
    overall = health.overall
    assert overall.generation_attempted == overall.generation_unusable == 5
    assert health.test is not None and health.test.generation_unusable == 4
    generate = health.by_purpose["generate"]
    assert generate.calls == generate.unusable > 0 and generate.failed == 0
    assert (overall.route_fallback, overall.route_lost_votes, overall.qa_unavailable) == (0, 0, 0)
    assert "decision" not in run.data["degraded_counts"] and run.data["usage"]["errors"] == 0
    assert health.reasons[0] == (
        "生成：调过大模型生成的 5 条里有 5 条（100.0%）"
        "没有一条可用草稿（调用全失败，或输出全都解析不了），上限 1 条。"
    )
    assert "生成没有可用草稿 5 / 5（调过大模型生成的）" in run.out
    # 生成调用全部失败（限流）也一样算没有可用草稿
    failing = _run(
        tmp_path,
        lambda r, case, nth: _rejected(429, "rate_limit") if r.purpose == "generate" else None,
    )
    assert failing.code == 3
    assert failing.health.overall.generation_unusable == 5
    assert failing.health.by_purpose["generate"].unusable == 0  # 没有响应，谈不上用不上


def test_qa_judge_unavailable(tmp_path):
    """质检的判断层调用全部超时：草稿只跑了硬规则（decision:qa）。"""

    def fault(request: LLMRequest, case: int, nth: int):
        return LLMTimeout("合成的故障：超时") if request.purpose == "qa" else None

    run = _run(tmp_path, fault)
    assert run.code == 3
    health = _check_rule(run)
    assert health.overall.qa_judged == health.overall.qa_unavailable == 5
    assert health.test is not None and health.test.qa_unavailable == 4
    qa = health.by_purpose["qa"]
    assert qa.calls == qa.failed == qa.timeouts == 5
    assert sum(qa.errors.values()) == 5 and set(qa.errors) <= {"timeout", "LLMTimeout"}
    assert health.reasons[0] == (
        "质检：有草稿送检的 5 条里有 5 条（100.0%）质检判断层不可用（只跑了硬规则），上限 1 条。"
    )
    assert run.data["degraded_counts"] == {"decision:qa": 5}
    assert "质检判断层不可用 5 / 5（有草稿送检的）" in run.out
    assert health.overall.route_fallback == 0 and health.routed_by == {"incloud_llm": 12}


# ================================================================ 退出码（--allow-degraded 等）


def _three_dev_cases_fall_back(request: LLMRequest, case: int, nth: int):
    if request.purpose == "route" and case in DEV[:3]:
        return _rejected(503, "server")
    return None


def test_allow_degraded_keeps_the_verdict_but_exits_zero(tmp_path):
    strict = _run(tmp_path, _three_dev_cases_fall_back)
    assert strict.code == 3 and strict.health.overall.route_fallback == 3
    allowed_run = _run(tmp_path, _three_dev_cases_fall_back, "--allow-degraded")
    assert allowed_run.code == 0
    # 报告和 JSON 一个字都不让步：valid 还是 false（record-eval 照样不认）
    assert allowed_run.health == strict.health and not allowed_run.health.valid
    assert allowed_run.out.count(INVALID) == 2
    assert "加了 --allow-degraded，退出码不因此改变" in allowed_run.err
    assert "退出码 3" not in allowed_run.err


def test_max_degraded_changes_the_ceiling_for_this_run_only(tmp_path):
    run = _run(tmp_path, _three_dev_cases_fall_back, "--max-degraded", "0.5")
    assert run.code == 0, run.err
    health = _check_rule(run)
    assert health.ceiling == 0.5 and health.valid and health.overall.route_fallback == 3
    assert "运行健康度：有效（上限 50%，--max-degraded 给的；默认 2%" in run.out
    # 读报告的一方（record-eval）按默认上限重新算，这一轮照样不算数
    assert assess(health.overall, health.test) == [
        "路由：12 条里有 3 条（25.0%）没用上判断层（退回关键词规则），上限 1 条。"
    ]
    # 上限调严到 0 也至少容许 1 条（health.allowed）
    assert _run(tmp_path, None, "--max-degraded", "0").health.valid
    parser = build_parser()
    assert parser.parse_args([]).max_degraded == DEFAULT_CEILING
    assert parser.parse_args([]).allow_degraded is False
    for bad in ("2", "-0.1", "nan", "50%"):  # 比例，不是百分数
        with pytest.raises(SystemExit) as exc:
            main(["--max-degraded", bad], out=io.StringIO(), err=io.StringIO())
        assert exc.value.code == 2
    client = FakeLLMClient(handler=Vendor(), model=MODEL)
    with pytest.raises(ValueError, match="上限必须在"):  # 直接调 evaluate：开跑前就拦下
        evaluate("all", limit=12, seed=3, llm=client, ceiling=2.0)
    assert client.calls == []


def test_an_invalid_run_outranks_a_baseline_regression(tmp_path):
    """同时有回归和"没评到模型"：退出码 3（兜底逻辑的数字比不出回归）；
    --allow-degraded 时才是 1。"""
    base = tmp_path / "base.json"
    client = FakeLLMClient(handler=Vendor(), model=MODEL)
    argv = [*ARGV, "--json", str(base)]
    assert main(argv, out=io.StringIO(), err=io.StringIO(), client_factory=lambda: client) == 0

    def fault(request: LLMRequest, case: int, nth: int):  # 路由全部被拒：类型准确率下降
        return _rejected(400, "bad_request") if request.purpose == "route" else None

    worse = _run(tmp_path, fault, "--baseline", str(base))
    assert worse.code == 3 and worse.data["baseline"]["passed"] is False
    assert "↓回归" in worse.out and worse.out.count(INVALID) == 2
    assert _run(tmp_path, fault, "--baseline", str(base), "--allow-degraded").code == 1


def test_help_and_module_docstring_list_the_exit_codes():
    import rodit.eval.cli as cli

    text = build_parser().format_help()
    for word in ("--allow-degraded", "--max-degraded", "--min-interval", "退出码：0 正常；1 回归"):
        assert word in text, word
    assert "3 这一轮没有有效评到模型" in text and "130 被中断" in text
    doc = cli.__doc__ or ""
    assert "退出码：0 正常；1 回归比较有指标下降；2 " in doc
    assert "3 这一轮没有有效评到模型" in doc and "130 被中断" in doc


# ================================================================ 连续失败提前停下、中断


def _outage(request: LLMRequest, case: int, nth: int):
    return _rejected(429, "rate_limit")


def test_stops_after_eight_consecutive_cases_whose_calls_all_failed(tmp_path):
    run = _run(tmp_path, _outage)
    assert run.code == 2 and run.out == "" and run.data is None
    assert STOP_AFTER_FAILED_CASES == 8
    assert "[8/12]" in run.err and "[9/12]" not in run.err  # 第 8 条跑完就停，后面的没有调用
    last = run.err.strip().splitlines()[-1]
    assert last.startswith("评测中止：连续 8 条用例的大模型调用全部失败（")
    assert "跑到第 8 / 12 条停下：没有生成报告，也没有保存结果" in last
    assert "--allow-degraded" in last and "--min-interval" in last
    failed = len(run.client.calls)  # 8 条用例发出的调用，全部失败
    assert f"（http_429 ×{failed}）" in last or f"（LLMError ×{failed}）" in last
    assert list(tmp_path.iterdir()) == []  # 没有报告，也没有留下临时文件
    # --allow-degraded：不停，跑完出报告
    full = _run(tmp_path, _outage, "--allow-degraded")
    assert full.code == 0 and full.data["n"] == 12 and not full.health.valid
    assert full.health.overall.route_fallback == 12


def test_a_case_that_gets_through_resets_the_count(tmp_path):
    def fault(request: LLMRequest, case: int, nth: int):
        return None if case == 7 else _rejected(429, "rate_limit")  # 第 8 条照常，其余全失败

    run = _run(tmp_path, fault)
    assert run.code == 3 and "评测中止" not in run.err  # 7 条 + 4 条，都不到 8 条
    assert run.data["n"] == 12 and run.health.overall.route_fallback == 11


def test_run_eval_stop_is_opt_in_and_counts_only_llm_failures():
    """runner.run_eval 默认不停（evaluate 直接调用时照旧出报告）；只有调用抛 LLMError / LLMTimeout
    才算失败，没调过大模型的用例不打断连续计数。"""
    client = FakeLLMClient(handler=Vendor(_outage), model=MODEL)
    report = evaluate("all", limit=12, seed=3, llm=client)
    assert report.n == 12 and report.health is not None and not report.health.valid
    with pytest.raises(EvalAborted) as exc:
        evaluate("all", limit=12, seed=3, llm=client, stop_after_failed=3)
    assert (exc.value.finished, exc.value.total, exc.value.streak) == (3, 12, 3)
    assert sum(exc.value.labels.values()) >= 9 and "连续 3 条" in str(exc.value)
    failed, crashed, idle = CallStats(calls=3, failed=3), CallStats(calls=3), CallStats()
    assert CaseTrace(calls={"route": failed}).all_calls_failed
    assert CaseTrace(calls={"route": failed, "qa": idle}).all_calls_failed
    assert not CaseTrace(
        calls={"route": failed, "generate": CallStats(calls=1, usable=1)}
    ).all_calls_failed
    assert not CaseTrace(calls={"route": crashed}).all_calls_failed  # 别的异常不是模型服务的失败
    assert not CaseTrace().all_calls_failed  # 没调过大模型


def test_keyboard_interrupt_reports_progress_and_saves_nothing(tmp_path):
    def fault(request: LLMRequest, case: int, nth: int):
        if case == 5:  # 第 6 条用例跑到一半按了 Ctrl-C
            raise KeyboardInterrupt
        return None

    run = _run(tmp_path, fault)
    assert run.code == 130 and run.out == "" and run.data is None
    last = run.err.strip().splitlines()[-1]
    assert (
        last
        == "评测被中断：已经跑完 5 / 12 条，结果没有保存（没有报告，也没有 JSON）；重跑会从头开始"
    )
    assert "Traceback" not in run.err and list(tmp_path.iterdir()) == []

    def at_once(request: LLMRequest, case: int, nth: int):
        raise KeyboardInterrupt

    early = _run(tmp_path, at_once)
    assert early.code == 130 and "评测被中断：还没有跑完一条，结果没有保存" in early.err


# ================================================================ §12.1 耗时如实标注


def _row(out: str, label: str) -> str:
    return next(line for line in out.splitlines() if line.startswith(label))


def test_end_to_end_latency_is_not_called_within_target_once_a_call_timed_out(tmp_path):
    healthy = _run(tmp_path)
    assert _row(healthy.out, "端到端").endswith("≤ 5000 达标")
    assert _row(healthy.out, "路由  ").endswith("≤ 1000 达标  ｜0 / 36（0.0%）")
    lat = healthy.data["latency"]
    assert lat["e2e_p95_within_target"] is True and lat["limited_by_budget"] is False
    assert (lat["budget_cut"], lat["llm_timeouts"]) == (0, 0)
    assert "受时间预算限制" not in healthy.out

    def fault(request: LLMRequest, case: int, nth: int):  # 一条用例的一票到超时才结束
        if (request.purpose, case, nth) == ("route", 0, 1):
            return LLMTimeout("合成的故障：超时")
        return None

    slow = _run(tmp_path, fault)
    assert slow.code == 0 and slow.health.valid  # 丢一票在上限之内：这一轮算数
    row = _row(slow.out, "端到端")
    assert row.endswith("≤ 5000 受时间预算限制") and "达标" not in row
    lat = slow.data["latency"]
    assert lat["e2e_p95_within_target"] is None
    assert lat["stages"]["total"]["within_target"] is None and lat["stages"]["total"]["p95"] < 5000
    assert lat["limited_by_budget"] is True and (lat["budget_cut"], lat["llm_timeouts"]) == (0, 1)
    route = lat["stages"]["route"]
    assert (route["llm_calls"], route["llm_timeouts"], route["within_target"]) == (36, 1, True)
    assert _row(slow.out, "路由  ").endswith("｜1 / 36（2.8%）")
    assert "端到端受时间预算限制：有 0 条用例被时间预算截断、1 次调用到超时才结束。" in slow.out
    assert "不是模型服务的真实耗时，不能算达标" in slow.out


def test_latency_section_states_the_network_path(tmp_path, monkeypatch):
    run = _run(tmp_path)
    assert run.data["llm"]["via_proxy"] is None  # 注入的客户端：不知道走没走代理
    assert (
        "网络路径：未知。耗时是从这台机器量的（含到模型服务的网络），不是平台云内的耗时" in run.out
    )
    for answer, label in ((True, "经本机代理"), (False, "直连")):
        monkeypatch.setattr("rodit.eval.cli.via_proxy", lambda client, answer=answer: answer)
        run = _run(tmp_path)
        assert run.data["llm"]["via_proxy"] is answer and f"网络路径：{label}。" in run.out
    offline = evaluate("test", limit=3)
    assert offline.llm.via_proxy is None and "网络路径" not in render_text(offline)


def test_calls_that_outlive_the_message_are_waited_for_and_counted_as_late(monkeypatch):
    """生成调用到时间预算还没返回：引擎不等它，handle() 先返回（budget）。评测等后台调用做完再读
    计量，它们才进得了用量和调用次数；迟到的条数单独记。"""
    release = threading.Event()
    original = AssistEngine.wait_background

    def releasing(self: AssistEngine, shop_id: str, timeout: float | None = None) -> bool:
        release.set()  # 评测在 handle() 返回之后才来等：这时才放生成调用返回
        return original(self, shop_id, timeout)

    monkeypatch.setattr(AssistEngine, "wait_background", releasing)
    fake = FakeLLMClient(handler=Vendor(), model=MODEL)

    class SlowGeneration:
        model = MODEL

        def complete(self, request: LLMRequest) -> LLMResponse:
            if request.purpose == "generate":
                release.wait(timeout=30)
            return fake.complete(request)

    config = EngineConfig(latency=LatencyConfig(message_budget_s=1.0, min_llm_remaining_s=0.1))
    try:
        report = evaluate("all", intent="size_advice", limit=1, llm=SlowGeneration(), config=config)
    finally:
        release.set()
    health = report.health
    assert health is not None
    generate = health.by_purpose["generate"]
    assert health.late_calls == generate.calls == 2 and generate.failed == 0
    assert report.usage is not None and report.usage.by_purpose["generate"].calls == 2
    assert report.usage.by_purpose["generate"].input_tokens > 0  # 迟到的调用的 token 也在
    assert health.overall.budget_cut == 1 and report.degraded_counts == {"budget": 1}
    assert health.valid  # 时间预算截断照实展示，不算"模型没被评到"
    assert report.latency.budget_cut == 1 and report.latency.limited_by_budget
    assert report.latency.e2e_p95_within_target is None
    text = render_text(report)
    assert "handle() 返回之后才结束的调用 2 次" in text
    assert "端到端受时间预算限制：有 1 条用例被时间预算截断、0 次调用到超时才结束。" in text


def test_trace_case_waits_for_background_work_before_reading_usage():
    case = PICKED[0]
    shop_id = getattr(fixtures, case.shop)
    usage = InMemoryAuditSink()

    def record(purpose: str) -> None:
        usage.record(
            AuditEvent(
                shop_id=shop_id,
                actor="system",
                action=AuditAction.LLM_USAGE.value,
                detail={"purpose": purpose, "model": MODEL, "input_tokens": 1, "output_tokens": 1},
                created_at=datetime(2026, 10, 9, 10, 0, tzinfo=CST),
            )
        )

    class Engine:
        def __init__(self) -> None:
            self.waited: list[tuple[str, float | None]] = []

        def handle(self, msg: Any) -> None:
            return None

        def wait_background(self, shop_id: str, timeout: float | None = None) -> bool:
            self.waited.append((shop_id, timeout))
            record("qa")  # 到截止时间被丢下的调用，这时才做完
            return True

    engine = Engine()
    record("route")
    trace = trace_case(SimpleNamespace(engine=engine, usage=usage), case, None)  # type: ignore[arg-type]
    assert engine.waited == [(shop_id, BACKGROUND_WAIT_S)]
    assert trace.late_calls == 1 and [d["purpose"] for d in trace.llm_calls] == ["route", "qa"]
    assert trace.metered and trace.route_provider is None and trace.calls == {}
    # 引擎没有 wait_background（别的 EngineLike 实现）：不等，照常读
    bare = SimpleNamespace(engine=SimpleNamespace(handle=lambda msg: None), usage=usage)
    assert trace_case(bare, case, None).late_calls == 0  # type: ignore[arg-type]
    # 不计量的 harness：不等，也没有计量
    unmetered = trace_case(SimpleNamespace(engine=engine), case, None)  # type: ignore[arg-type]
    assert unmetered.metered is False and len(engine.waited) == 1


def test_min_interval_spaces_the_starts_of_consecutive_cases(monkeypatch):
    naps: list[float] = []
    monkeypatch.setattr(runner.time, "sleep", naps.append)
    out = io.StringIO()
    argv = ["--limit", "4", "--min-interval", "30", "--json"]
    assert main(argv, out=out, err=io.StringIO()) == 0
    assert len(naps) == 3 and all(0 < nap <= 30 for nap in naps)  # 第一条之前不等
    total = json.loads(out.getvalue())["latency"]["stages"]["total"]
    assert total["max"] < 1000  # 等待在 handle() 之外：不计入耗时
    naps.clear()
    assert main(["--limit", "4", "--json"], out=io.StringIO(), err=io.StringIO()) == 0
    assert naps == []  # 默认不等
    assert build_parser().parse_args([]).min_interval == 0.0
    for bad in ("-1", "inf", "nan"):
        with pytest.raises(SystemExit) as exc:
            main(["--min-interval", bad], out=io.StringIO(), err=io.StringIO())
        assert exc.value.code == 2


# ================================================================ §14 用量如实标注


def _as_text(answer: Any) -> str:
    return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)


def test_failed_calls_are_listed_per_purpose_with_an_estimate(tmp_path):
    run = _run(tmp_path, _forty_percent)
    usage = run.data["usage"]
    route = usage["by_purpose"]["route"]
    assert (route["calls"], route["errors"]) == (36, 15)
    metered = route["input_tokens"] + route["output_tokens"]  # 成功的 21 次
    assert route["failed_tokens_estimate"] == pytest.approx(15 * metered / 21)
    assert usage["failed_tokens_estimate"] == pytest.approx(route["failed_tokens_estimate"])
    assert usage["by_purpose"]["generate"]["failed_tokens_estimate"] == 0  # 没有失败的调用
    assert "失败的 15 次调用按 0 token 计，上面的合计、成本和每店每月都不含它们" in run.out
    assert "按同用途成功调用的平均值估算（只是估算，不是计量）" in run.out
    assert f"  路由：失败 15 次，估算 ≈ {route['failed_tokens_estimate']:,.0f} token" in run.out
    assert "  生成：失败" not in run.out
    assert "按 0 token 计" not in _run(tmp_path).out  # 没有失败的调用：不多这几行

    def all_rejected(request: LLMRequest, case: int, nth: int):
        return _rejected(400, "bad_request") if request.purpose == "route" else None

    run = _run(tmp_path, all_rejected)  # 路由没有一次成功：没有可参照的，估不出
    assert run.data["usage"]["by_purpose"]["route"]["failed_tokens_estimate"] is None
    assert run.data["usage"]["failed_tokens_estimate"] is None
    assert "  路由：失败 36 次，估不出（这个用途没有带用量的成功调用可参照）" in run.out


def test_zero_usage_is_reported_as_missing_not_as_zero(tmp_path):
    vendor = Vendor()

    def no_usage(request: LLMRequest, case: int, nth: int):  # 接口没有返回 usage
        return LLMResponse(text=_as_text(vendor.answer(request)), model=MODEL)

    run = _run(tmp_path, no_usage)
    usage = run.data["usage"]
    assert run.code == 0 and usage["total_tokens"] == 0
    assert usage["zero_usage_calls"] == usage["calls"] > 0
    assert usage["by_purpose"]["route"]["zero_usage_calls"] == 36
    assert (
        f"注意：有 {usage['calls']} 次成功的调用 token 数是 0——接口没有返回用量（usage），"
        "上面的 token 和成本少算了，不是真的为 0"
    ) in run.out
    healthy = _run(tmp_path)
    assert healthy.data["usage"]["zero_usage_calls"] == 0 and "token 数是 0" not in healthy.out


def test_served_models_are_shown_and_a_different_name_is_flagged(tmp_path):
    run = _run(tmp_path)
    calls = run.data["usage"]["calls"]
    assert run.data["usage"]["served_models"] == {MODEL: calls}
    assert f"模型服务回显的模型名（成功的调用）：{MODEL} ×{calls}" in run.out
    assert "回显的模型名和配置的" not in run.out
    vendor = Vendor()
    snapshot = f"{MODEL}-2026-09-01"  # 配置的名字是别名，接口回显的是具体版本

    def echoes_snapshot(request: LLMRequest, case: int, nth: int):
        if (request.purpose, case) == ("route", 0):
            return LLMError("合成的故障")  # 失败的调用没有回显，不算
        return LLMResponse(
            text=_as_text(vendor.answer(request)),
            model=snapshot,
            input_tokens=100,
            output_tokens=20,
        )

    run = _run(tmp_path, echoes_snapshot)
    usage = run.data["usage"]
    assert usage["served_models"] == {snapshot: usage["calls"] - usage["errors"]}
    assert run.data["llm"]["model"] == run.data["fingerprint"]["llm_model"] == MODEL
    assert f"模型服务回显的模型名（成功的调用）：{snapshot} ×" in run.out
    assert f"注意：回显的模型名和配置的 {MODEL} 不一样（配置的可能只是别名）" in run.out


# ================================================================ 完整结果：先验证、原子写、写得出


def _main(argv: list[str], client: FakeLLMClient) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err, client_factory=lambda: client)
    return code, out.getvalue(), err.getvalue()


def test_an_unwritable_json_target_is_found_before_the_first_call(tmp_path):
    client = FakeLLMClient(handler=Vendor(), model=MODEL)
    blocker = tmp_path / "a_file"
    blocker.write_text("合成", encoding="utf-8")
    for target in (tmp_path, blocker / "sub" / "out.json"):  # 目标是个目录；上级是个文件
        code, out, err = _main([*WITH_PREFLIGHT, "--json", str(target)], client)
        assert code == 2 and out == ""
        assert f"评测没有运行：完整结果写不进 {target}（" in err and "Traceback" not in err
        assert client.calls == []  # 连连通性检查都没有发
    assert [p.name for p in tmp_path.iterdir()] == ["a_file"]
    assert blocker.read_text(encoding="utf-8") == "合成"


def test_json_is_written_through_a_temp_file_before_the_text_report(tmp_path, monkeypatch):
    import rodit.eval.cli as cli

    target = tmp_path / "runs" / "llm.json"  # 上级目录开跑前就建好
    seen: list[bool] = []
    real_render = cli.render_text

    def render(report: Any) -> str:
        seen.append(target.exists())  # 渲染文本报告时 JSON 已经落盘
        return real_render(report)

    monkeypatch.setattr(cli, "render_text", render)
    client = FakeLLMClient(handler=Vendor(), model=MODEL)
    code, out, _ = _main([*ARGV, "--json", str(target)], client)
    assert code == 0 and seen == [True] and f"完整结果（JSON）已写入 {target}" in out
    assert [p.name for p in target.parent.iterdir()] == ["llm.json"]  # 没有留下临时文件
    assert json.loads(target.read_text(encoding="utf-8"))["n"] == 12

    def broken(report: Any) -> str:
        raise RuntimeError("合成的故障：渲染文本报告出错")

    monkeypatch.setattr(cli, "render_text", broken)
    second = tmp_path / "runs" / "second.json"
    with pytest.raises(RuntimeError):
        _main([*ARGV, "--json", str(second)], client)
    assert json.loads(second.read_text(encoding="utf-8"))["n"] == 12  # 结果没有跟着丢


def test_baseline_and_json_may_name_the_same_file(tmp_path):
    target = tmp_path / "llm.json"
    assert (
        _main([*ARGV, "--json", str(target)], FakeLLMClient(handler=Vendor(), model=MODEL))[0] == 0
    )
    before = target.read_text(encoding="utf-8")
    untouched: list[bool] = []

    def fault(request: LLMRequest, case: int, nth: int):
        untouched.append(target.read_text(encoding="utf-8") == before)  # 跑的过程中基线原样
        return None

    client = FakeLLMClient(handler=Vendor(fault), model=MODEL)
    code, out, _ = _main([*ARGV, "--baseline", str(target), "--json", str(target)], client)
    assert code == 0 and untouched and all(untouched)
    data = json.loads(target.read_text(encoding="utf-8"))  # 跑完才换成新的
    assert data["baseline"]["passed"] is True and data["baseline"]["common_cases"] == 12
    assert [p.name for p in tmp_path.iterdir()] == ["llm.json"]


def test_json_target_behind_a_symlink_or_on_a_device(tmp_path):
    """换名换的是符号链接指的那个文件，链接留着；设备这类目标没法换名，直接写（离线替身就够测）。"""
    real = tmp_path / "data" / "llm.json"
    real.parent.mkdir()
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "latest.json"
    link.symlink_to(real)
    assert main(["--limit", "3", "--json", str(link)], out=io.StringIO(), err=io.StringIO()) == 0
    assert link.is_symlink() and json.loads(real.read_text(encoding="utf-8"))["n"] == 3
    assert [p.name for p in real.parent.iterdir()] == ["llm.json"]
    err = io.StringIO()
    assert main(["--limit", "3", "--json", os.devnull], out=io.StringIO(), err=err) == 0
    assert err.getvalue() == ""


def test_a_failed_final_write_still_prints_the_report(tmp_path, monkeypatch):
    import rodit.eval.cli as cli

    def refuse(src: Any, dst: Any) -> None:
        raise PermissionError("合成的故障：换名失败")

    monkeypatch.setattr(cli.os, "replace", refuse)
    target = tmp_path / "llm.json"
    code, out, err = _main(
        [*ARGV, "--json", str(target)], FakeLLMClient(handler=Vendor(), model=MODEL)
    )
    assert code == 2 and f"完整结果写不进 {target}：PermissionError" in err
    assert "== 总体" in out and "已写入" not in out  # 文本报告照样给出来
    assert list(tmp_path.iterdir()) == []  # 临时文件删掉了


def test_a_lone_surrogate_in_model_text_does_not_lose_the_run(tmp_path):
    lone = "\ud83d"  # 落单的代理项：UTF-8 写不出

    def fault(request: LLMRequest, case: int, nth: int):
        if request.purpose == "generate":
            return {"answer": f"亲，这款是真丝的{lone}", "evidence_ids": [], "need_clarify": False}
        if request.purpose == "qa":  # 质检不通过：草稿进失败清单
            return {**QA_PASS, Q_GROUNDED: {"answer": "否", "p": 0.1}}
        return None

    def run(*extra: str) -> tuple[int, str]:
        raw = io.BytesIO()
        out = io.TextIOWrapper(raw, encoding="utf-8")  # 和真实的标准输出一样按 UTF-8 严格编码
        client = FakeLLMClient(handler=Vendor(fault), model=MODEL)
        code = main([*ARGV, *extra], out=out, err=io.StringIO(), client_factory=lambda: client)
        out.flush()
        return code, raw.getvalue().decode("utf-8")

    target = tmp_path / "llm.json"
    code, text = run("--json", str(target))
    assert code == 0
    assert "模型草稿 1：亲，这款是真丝的\\ud83d" in text  # 写成转义，其余照常
    data = json.loads(target.read_text(encoding="utf-8"))
    assert any(d.endswith(lone) for f in data["failures"] for d in f["drafts"])  # 读回来还是原样
    code, text = run("--json")  # 输出到标准输出的 JSON 同样写得出，而且合法
    assert code == 0
    printed = json.loads(text)
    drafts = [f["drafts"] for f in data["failures"]]
    assert printed["n"] == 12 and [f["drafts"] for f in printed["failures"]] == drafts


# ================================================================ 配置：RODIT_CONFIG、内置默认值


def test_env_mode_falls_back_to_the_config_named_by_rodit_config(tmp_path, monkeypatch):
    from rodit.core.config import config_fingerprint, load_config

    deployed = tmp_path / "deployed.yaml"
    deployed.write_text("decision:\n  vote_samples: 5\n", encoding="utf-8")
    monkeypatch.setenv("RODIT_CONFIG", f" {deployed} ")  # 首尾空白不算
    run = _run(tmp_path)
    assert run.code == 0, run.err
    assert run.data["llm"]["route_votes"] == 5 and len(run.cases_where(route_votes=5)) == 12
    expected = config_fingerprint(load_config(deployed))
    assert run.data["fingerprint"]["config_fingerprint"] == expected
    assert f"配置：没有给 --config，用环境变量 RODIT_CONFIG 指的 {deployed}" in run.err
    assert DEFAULTS_WARNING not in run.err
    # --config 优先
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("decision:\n  vote_samples: 1\n", encoding="utf-8")
    run = _run(tmp_path, None, "--config", str(explicit))
    assert run.data["llm"]["route_votes"] == 1 and "RODIT_CONFIG" not in run.err
    # 离线替身不读它：回归基线不随运行环境变
    out = io.StringIO()
    assert main(["--limit", "3", "--json"], out=out, err=io.StringIO()) == 0
    offline = json.loads(out.getvalue())["fingerprint"]["config_fingerprint"]
    assert offline == config_fingerprint(EngineConfig()) != expected
    # 指的文件读不了：不跑，一次调用都不发
    monkeypatch.setenv("RODIT_CONFIG", str(tmp_path / "missing.yaml"))
    client = FakeLLMClient(handler=Vendor(), model=MODEL)
    code, out_text, err = _main(WITH_PREFLIGHT, client)
    assert code == 2 and out_text == "" and "评测没有运行" in err and client.calls == []


def test_running_on_built_in_defaults_is_called_out(tmp_path):
    run = _run(tmp_path)
    assert "config/rodit.yaml" in DEFAULTS_WARNING and "不能当上线依据" in DEFAULTS_WARNING
    assert run.err.count(DEFAULTS_WARNING) == 2  # 开跑前一次，几百行进度之后结尾再一次
    assert run.err.index(DEFAULTS_WARNING) < run.err.index("[1/12]")
    assert run.err.rindex(DEFAULTS_WARNING) > run.err.index("[12/12]")
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    assert DEFAULTS_WARNING not in _run(tmp_path, None, "--config", str(config)).err
    err = io.StringIO()  # 离线替身的数字本来就不是上线依据：不提醒
    assert main(["--limit", "3"], out=io.StringIO(), err=err) == 0 and err.getvalue() == ""


def _request(purpose: str) -> LLMRequest:
    return LLMRequest(purpose=purpose, messages=[LLMMessage(role="user", content="合成的请求")])  # type: ignore[arg-type]


def test_env_client_sends_the_configured_extra_body(monkeypatch):
    """llm.extra_body（如关闭思考模式）两条路都带上：地址和模型来自环境变量，或写在配置里。"""
    monkeypatch.setenv("RODIT_LLM_BASE_URL", "https://llm.example.invalid/v1")
    monkeypatch.setenv("RODIT_LLM_MODEL", "synthetic-model")
    monkeypatch.setenv("RODIT_LLM_API_KEY", "sk-synthetic-not-a-real-key")  # 合成的假密钥
    request = _request("route")
    from_env = EngineConfig(llm=LLMConfig(extra_body={"thinking": {"type": "disabled"}}))
    payload = env_client(from_env).build_payload(request)
    assert payload["thinking"] == {"type": "disabled"} and payload["model"] == "synthetic-model"
    pinned = EngineConfig(
        llm=LLMConfig(
            base_url="https://pinned.example.invalid/v1",
            model="pinned-model",
            extra_body={"enable_thinking": False},
        )
    )
    payload = env_client(pinned).build_payload(request)
    assert payload["enable_thinking"] is False and payload["model"] == "pinned-model"
    plain = env_client(None).build_payload(request)  # 没配：请求里不多出字段
    assert "thinking" not in plain and "enable_thinking" not in plain


# ================================================================ 调用结果的记录（CallRecorder）


def test_call_recorder_counts_what_each_response_was_worth():
    intent = {Q_INTENT: {"choice": "greeting"}}
    cut = LLMResponse(text=json.dumps(intent), finish_reason="length")
    answers: list[tuple[str, Any]] = [
        ("route", intent),  # 只答了主要问题：这一票算数
        ("route", {Q_NEED_HUMAN: {"answer": "否", "p": 0.1}}),  # 没答主要问题：作废
        ("route", "好的，这位买家是在问物流。"),  # 不是 JSON：作废
        ("route", cut),  # 被 max_tokens 截断：作废，哪怕碰巧读得出
        ("route", LLMResponse(text="  ", finish_reason="stop")),  # 空：作废
        ("route", LLMTimeout("合成的故障：超时")),
        ("qa", dict(QA_PASS)),
        ("qa", {k: v for k, v in QA_PASS.items() if k != Q_TONE}),  # 少答一题：用不上
        ("generate", {"answer": "亲，在的", "evidence_ids": []}),
        ("generate", "抱歉，我没办法按这个格式回答。"),  # 解析不出回答
        ("generate", LLMResponse(text='{"answer": "亲，在的"}', finish_reason="length")),
        ("other", "OK"),
    ]
    fake = FakeLLMClient(responses=[answer for _, answer in answers], model=MODEL)
    recorder = CallRecorder(fake)
    for purpose, answer in answers:
        if isinstance(answer, Exception):
            with pytest.raises(LLMTimeout):  # 失败照常抛出
                recorder.complete(_request(purpose))
        else:
            recorder.complete(_request(purpose))
    stats = recorder.stats()
    assert stats["route"] == CallStats(
        calls=6, failed=1, truncated=1, empty=1, unusable=4, usable=1
    )
    assert stats["qa"] == CallStats(calls=2, unusable=1, usable=1)
    # 生成：被截断但解析得出回答的照常用得上（生成器也会用它）
    assert stats["generate"] == CallStats(calls=3, truncated=1, unusable=1, usable=2)
    assert stats["other"] == CallStats(calls=1, usable=1)
    assert recorder.model == MODEL and len(fake.calls) == len(answers)


def test_call_recorder_never_changes_the_outcome():
    class Odd:
        model = MODEL

        def complete(self, request: LLMRequest) -> Any:
            if request.purpose == "qa":
                raise ValueError("合成的缺陷")
            return SimpleNamespace(text=json.dumps({Q_INTENT: {"choice": "greeting"}}))

    recorder = CallRecorder(Odd())  # type: ignore[arg-type]
    response = recorder.complete(_request("route"))  # 响应对象没有 finish_reason 也照常
    assert json.loads(response.text) == {Q_INTENT: {"choice": "greeting"}}
    assert recorder.stats()["route"] == CallStats(calls=1, usable=1)
    with pytest.raises(ValueError):  # 不是 LLMError：照常抛出，不算模型服务的失败
        recorder.complete(_request("qa"))
    assert recorder.stats()["qa"] == CallStats(calls=1)


def test_llm_harness_records_calls_per_case():
    fake = FakeLLMClient(handler=Vendor(), model=MODEL)
    first, second = llm_harness(fake), llm_harness(fake)
    assert isinstance(first.calls, CallRecorder) and first.calls is not second.calls
    assert first.calls.stats() == {}


def test_health_error_labels_come_from_the_usage_detail():
    """PurposeHealth.errors 的键是 health.error_label 按 llm.usage 的 detail 给的：detail 带了
    状态码和代码就细分，没带就是异常类型名（rodit.llm.metering 写什么这里就读什么）。"""
    from rodit.eval.cli import _purpose_health

    details = [
        {"purpose": "route", "error": "LLMError", "kind": "rate_limit", "status": 429},
        {"purpose": "route", "error": "LLMError", "kind": "rate_limit", "status": 429},
        {
            "purpose": "route",
            "error": "LLMError",
            "kind": "bad_request",
            "status": 400,
            "code": "unsupported_parameter",
        },
        {"purpose": "route", "error": "LLMTimeout", "kind": "timeout"},
        {"purpose": "route", "error": "LLMTimeout"},  # 还没带 kind 的事件：按类型名认超时
        {"purpose": "route", "error": "LLMError", "kind": "network"},
        {"purpose": "route"},
        {"purpose": "qa", "error": "LLMError"},
    ]
    assert [error_label(d) for d in details[:6]] == [
        "http_429",
        "http_429",
        "http_400:unsupported_parameter",
        "timeout",
        "LLMTimeout",
        "network",
    ]
    trace = CaseTrace(
        llm_calls=details,
        metered=True,
        calls={"route": CallStats(calls=7, failed=6, truncated=1, unusable=1)},
    )
    health = _purpose_health([trace])
    route = health["route"]
    assert (route.calls, route.failed, route.timeouts) == (7, 6, 2)
    assert route.errors == {
        "http_429": 2,
        "http_400:unsupported_parameter": 1,
        "timeout": 1,
        "LLMTimeout": 1,
        "network": 1,
    }
    assert (route.truncated, route.empty, route.unusable) == (1, 0, 1)
    assert health["qa"].errors == {"LLMError": 1} and list(health) == ["route", "qa"]
