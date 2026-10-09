"""python -m rodit.eval --llm env：真实大模型的接线、计量、失败清单、回归比较。

对应方案 §13.2、§13.5、§12.1、§14。

不联网：大模型用 FakeLLMClient（注入客户端），或真实的 OpenAICompatibleClient 配假的 transport
（环境变量用 monkeypatch 设成合成值）。假模型按用途回答：路由答用例标注的类型，生成复述依据
（离线替身 EvidenceEchoLLM），质检按设定通过或不通过。
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from rodit.app.offline import EvidenceEchoLLM
from rodit.core.enums import Intent
from rodit.core.errors import LLMError
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
from rodit.decision.llm import SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT
from rodit.eval.cli import Failure, _render_failure, evaluate, main, render_text
from rodit.eval.dataset import CASES, cases
from rodit.eval.harness import UNPARSED_PREFIX, DraftRecorder, llm_harness
from rodit.eval.llm_env import PREFLIGHT_QUESTION, PreflightFailed, client_from_env
from rodit.eval.perf import Pricing
from rodit.testing.fakes import FakeLLMClient

EXPECTED = {mask_pii(c.text): c.expected_intent for c in CASES}
BY_ID = {c.id: c for c in CASES}
PHONE = "13812345678"  # 合成号码
FAKE_KEY = "sk-synthetic-eval-key-DO-NOT-LEAK"  # 合成的假密钥，只用来检查不外泄


def _state(request: LLMRequest) -> dict[str, Any]:
    user = request.messages[-1].content
    head = "## 状态（state）\n"
    return json.loads(user[user.index(head) + len(head) : user.index("\n\n## 问题")])


class ScriptedModel:
    """按 LLMRequest.purpose 回答的假模型（FakeLLMClient 的 handler）。

    - wrong：这些用例的路由答成 other；
    - grounded：质检"事实都有依据"的概率（< 0.8 就不通过）；
    - draft：生成请求一律回这句话（None 时复述依据）；raw 为真时原样输出（不是 JSON）。
    """

    def __init__(
        self,
        *,
        wrong: frozenset[str] = frozenset(),
        grounded: float = 0.95,
        draft: str | None = None,
        raw: bool = False,
    ) -> None:
        self.wrong_texts = {mask_pii(BY_ID[i].text) for i in wrong}
        self.grounded = grounded
        self.draft = draft
        self.raw = raw
        self.echo = EvidenceEchoLLM()

    def __call__(self, request: LLMRequest) -> str | dict | LLMResponse:
        if request.purpose == "other":  # 连通性检查
            return "OK"
        if request.purpose == "route":
            text = _state(request)["messages"][-1]["text"]
            intent = Intent.OTHER if text in self.wrong_texts else EXPECTED.get(text, Intent.OTHER)
            return {
                Q_INTENT: {"choice": intent.value},
                Q_NEED_HUMAN: {"answer": "否", "p": 0.05},
                Q_EMOTION: {"score": 0},
                Q_MULTI_INTENT: {"answer": "否", "p": 0.05},
            }
        if request.purpose == "qa":
            passed = self.grounded >= 0.8
            return {
                Q_GROUNDED: {"answer": "是" if passed else "否", "p": self.grounded},
                Q_ANSWERS_QUESTION: {"answer": "是", "p": 0.95},
                Q_PROMISE_RISK: {"answer": "否", "p": 0.05},
                Q_PLATFORM_VIOLATION: {"answer": "否", "p": 0.02},
                Q_TONE: {"score": 3},
            }
        if self.draft is not None:
            if self.raw:
                return self.draft
            return {"answer": self.draft, "evidence_ids": [], "need_clarify": False}
        return self.echo.complete(request)


def _fake(**kw: Any) -> FakeLLMClient:
    return FakeLLMClient(handler=ScriptedModel(**kw), model="synthetic-llm")


def _run(argv: list[str], **kw: Any) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err, **kw)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(autouse=True)
def _no_deployment_config(monkeypatch):
    """--llm env 没给 --config 时会读环境变量 RODIT_CONFIG 指的配置：这里的测试按内置默认值
    （或自己给的 --config）跑，不受运行环境影响。"""
    monkeypatch.delenv("RODIT_CONFIG", raising=False)


# ---------------------------------------------------------------- 接线与计量


def test_llm_mode_uses_model_for_route_generate_and_qa():
    fake = _fake()
    report = evaluate(
        "all",
        intent="product_spec",
        llm=fake,
        pricing=Pricing(input_per_1k=0.002, output_per_1k=0.006),
    )
    n = len([c for c in CASES if c.expected_intent == Intent.PRODUCT_SPEC])
    assert report.n == n and report.intent == "product_spec"
    assert report.llm.mode == "env" and report.llm.model == "synthetic-llm"
    assert report.llm.decision_provider == "incloud_llm" and report.llm.route_votes == 3
    # 路由走大模型（3 次投票），问题类型都按假模型答的来
    assert report.overall.intent_accuracy == 1.0
    assert not report.degraded_counts
    usage = report.usage
    assert usage is not None and usage.errors == 0
    assert usage.by_purpose["route"].calls == 3 * n
    assert usage.by_purpose["generate"].calls > 0 and usage.by_purpose["qa"].calls > 0
    # 每条用例的计量只算自己的调用；合计等于假模型实际收到的请求
    assert sum(c.usage.calls for c in report.cases) == usage.calls == len(fake.calls)
    route_prompts = [
        r for r in fake.calls if r.purpose == "route"
    ]  # FakeLLMClient 按字符数计 token
    assert usage.by_purpose["route"].input_tokens == sum(
        len("".join(m.content for m in r.messages)) for r in route_prompts
    )
    assert sum(abs(p.token_share) for p in usage.by_purpose.values()) == pytest.approx(1.0)
    expected_cost = usage.input_tokens / 1000 * 0.002 + usage.output_tokens / 1000 * 0.006
    assert usage.cost == pytest.approx(expected_cost)
    assert sum(c.usage.cost for c in report.cases) == pytest.approx(expected_cost)
    assert usage.projected_monthly_tokens_per_shop == pytest.approx(
        usage.total_tokens / n * 900 * 30
    )
    # 耗时：每条都有，各环节都有统计
    assert report.latency.n == n and set(report.latency.stages) >= {"route", "qa", "total"}
    assert all(c.latency_ms.get("total") is not None for c in report.cases)
    text = render_text(report)
    for section in ("== §12.1 耗时", "== §14 大模型用量", "路由：调用", "§14 示例口径", "合计"):
        assert section in text, section
    assert "单元：本地演示 + 大模型 synthetic-llm" in text


def test_llm_harness_keeps_usage_and_drafts_per_case():
    fake = _fake()
    first, second = llm_harness(fake), llm_harness(fake)
    assert first.usage is not second.usage and first.drafts is not second.drafts
    assert first.audit is not None and first.usage.events == []


def test_failures_show_model_drafts_and_qa_reasons_without_pii():
    """质检不通过：失败清单里有模型草稿（已脱敏）和质检原因。"""
    fake = _fake(grounded=0.1, draft=f"亲，这款是纯棉的，有问题打我电话{PHONE}")
    report = evaluate("all", intent="product_spec", llm=fake)
    failure = next(f for f in report.failures if f.drafts)  # 走大模型生成的用例
    assert failure.answer is None and failure.drafts and failure.qa_reasons
    assert all(PHONE not in d and "[手机号]" in d for d in failure.drafts)
    assert failure.no_suggestion_reason and "质检未通过" in failure.no_suggestion_reason
    text = render_text(report)
    assert "模型草稿 1：" in text
    assert all(reason in text for reason in failure.qa_reasons)  # 在"没有推荐"那行里，不重复列
    assert report.overall.suggestion_rate < 1.0
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False)
    assert PHONE not in text and PHONE not in payload


def test_failure_rendering_lists_qa_reasons_next_to_a_shown_answer():
    """有推荐（另一条草稿通过了）时，被拦下的草稿的质检原因单独列出。"""
    failure = Failure(
        id="x",
        split="dev",
        expected_intent=Intent.GREETING,
        predicted_intent=Intent.GREETING,
        action="answer",
        scores={"factual": 1, "on_topic": 2, "compliance": 2, "expression": 2},
        problems=["缺少要点：['在的']"],
        answer="亲，您好",
        no_suggestion_reason=None,
        drafts=["亲，您好", "亲，加我微信"],
        qa_reasons=["站外引流"],
        degraded=["decision:qa"],
    )
    lines: list[str] = []
    _render_failure(failure, lines.append)
    text = "\n".join(lines)
    assert "推荐：亲，您好" in text and "模型草稿 2：亲，加我微信" in text
    assert "质检原因：站外引流" in text and "降级：decision:qa" in text


def test_unparseable_generation_output_is_shown_truncated():
    recorder = DraftRecorder(_fake(draft="好的，这是回答：纯棉" + "啊" * 300, raw=True))
    request = LLMRequest(purpose="generate", messages=[LLMMessage(role="user", content="q")])
    recorder.complete(request)
    recorder.complete(request.model_copy(update={"purpose": "qa"}))  # 非生成请求不记
    (draft,) = recorder.drafts()
    assert draft.startswith(UNPARSED_PREFIX) and len(draft) <= len(UNPARSED_PREFIX) + 200
    assert recorder.model == "synthetic-llm"


def test_model_errors_are_counted_and_degrade():
    """大模型全部失败：路由退回关键词规则、生成失败都记成降级；用量里有失败类型。"""

    def broken(request: LLMRequest):
        return LLMError("合成的故障")

    report = evaluate("all", intent="greeting", llm=FakeLLMClient(handler=broken))
    usage = report.usage
    assert usage is not None and usage.errors == usage.calls > 0
    assert usage.error_types == {"LLMError": usage.calls}
    assert usage.input_tokens == 0
    assert report.degraded_counts.get("decision") == report.n
    assert "失败" in render_text(report)


# ---------------------------------------------------------------- 命令行：--llm env


def test_cli_env_mode_with_injected_client(tmp_path):
    path = tmp_path / "llm.json"
    code, out, err = _run(
        ["--llm", "env", "--limit", "4", "--seed", "7", "--json", str(path)],
        client_factory=lambda: _fake(),
    )
    assert code == 0, err
    assert "部分评测：随机抽 4 条（seed 7）" in out and "大模型用量" in out
    assert f"已写入 {path}" in out
    assert err.count("[") >= 4 and "[4/4]" in err  # 进度
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["n"] == 4 and data["limit"] == 4 and data["seed"] == 7
    assert data["llm"]["mode"] == "env" and data["usage"]["calls"] > 0
    # 运行健康度（rodit.eval.health）：接大模型时每次都有；假模型每次都答得上，这一轮算数
    health = data["health"]
    assert health["valid"] is True and health["reasons"] == [] and health["ceiling"] == 0.02
    assert health["overall"]["cases"] == 4 and health["overall"]["route_fallback"] == 0
    assert health["by_purpose"]["route"]["calls"] == 12 and health["routed_by"] == {
        "incloud_llm": 4
    }
    assert "运行健康度：有效" in out and "!!" not in out
    assert all(c["route_provider"] == "incloud_llm" for c in data["cases"])
    assert len(data["cases"]) == 4 and all(c["usage"]["calls"] > 0 for c in data["cases"])
    # 同一个 seed 抽到同样的用例
    _, out2, _ = _run(
        ["--llm", "env", "--limit", "4", "--seed", "7", "--json"], client_factory=lambda: _fake()
    )
    assert [c["id"] for c in json.loads(out2)["cases"]] == [c["id"] for c in data["cases"]]


def test_cli_env_mode_reads_environment_and_never_leaks_key(monkeypatch, tmp_path):
    """真实客户端 + 假 transport：地址、模型、超时来自环境变量；密钥只出现在请求头里，
    地址的路径不进报告。"""
    monkeypatch.setenv(
        "RODIT_LLM_BASE_URL", "https://llm.example.invalid:8443/v1/tenant-PATHSECRET"
    )
    monkeypatch.setenv("RODIT_LLM_MODEL", "synthetic-model")
    monkeypatch.setenv("RODIT_LLM_API_KEY", FAKE_KEY)
    monkeypatch.setenv("RODIT_LLM_TIMEOUT_S", "7")
    model = ScriptedModel()
    seen: list[tuple[str, str, float]] = []

    def transport(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float):
        seen.append((url, headers["Authorization"], timeout))
        messages = [LLMMessage(**m) for m in body["messages"]]
        if messages[0].content == JUDGE_SYSTEM_PROMPT:
            purpose = "route" if "### intent" in messages[-1].content else "qa"
        elif "连通性检查" in messages[-1].content:
            purpose = "other"
        else:
            purpose = "generate"
        answer = model(LLMRequest(purpose=purpose, messages=messages))
        if isinstance(answer, LLMResponse):
            text = answer.text
        else:
            text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return 200, {
            "model": "synthetic-model",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    path = tmp_path / "env.json"
    code, out, err = _run(
        [
            "--llm",
            "env",
            "--intent",
            "shipping_policy",
            "--json",
            str(path),
            "--price-input",
            "0.001",
            "--price-output",
            "0.002",
        ],
        client_factory=lambda: client_from_env(transport=transport),
    )
    assert code == 0, err
    assert seen and all(
        url == "https://llm.example.invalid:8443/v1/tenant-PATHSECRET/chat/completions"
        for url, _, _ in seen
    )
    assert all(auth == f"Bearer {FAKE_KEY}" for _, auth, _ in seen)
    # 连通性检查发的就是评测要发的请求，各带各的超时：路由 3 票、质检 1 次是 decision.timeout_s
    # （§12.1 ≤ 1 秒），生成 1 次是 generation.timeout_s；客户端的默认超时
    # （RODIT_LLM_TIMEOUT_S）用不上
    assert sorted(t for _, _, t in seen[:5]) == [1.0, 1.0, 1.0, 1.0, 2.5]
    assert 1.0 in {t for _, _, t in seen[5:]}  # 评测里的路由同样用 decision.timeout_s
    saved = path.read_text(encoding="utf-8")
    for text in (out, err, saved):
        assert FAKE_KEY not in text and "PATHSECRET" not in text
    data = json.loads(saved)
    assert data["llm"]["endpoint"] == "https://llm.example.invalid:8443"
    assert data["llm"]["model"] == "synthetic-model"
    calls = data["usage"]["calls"]
    assert calls == len(seen) - 5  # 连通性检查的 5 次请求不计入评测用量
    assert data["usage"]["input_tokens"] == 100 * calls
    assert data["usage"]["cost"] == pytest.approx(calls * (100 * 0.001 + 20 * 0.002) / 1000)
    assert "（https://llm.example.invalid:8443）" in out


def test_cli_env_mode_missing_configuration(monkeypatch):
    monkeypatch.setenv("RODIT_LLM_BASE_URL", "https://llm.example.invalid/v1")
    monkeypatch.setenv("RODIT_LLM_MODEL", "synthetic-model")
    monkeypatch.delenv("RODIT_LLM_API_KEY", raising=False)
    code, out, err = _run(["--llm", "env"])
    assert code == 2 and out == ""
    assert "RODIT_LLM_API_KEY 未设置" in err and "Traceback" not in err
    monkeypatch.delenv("RODIT_LLM_BASE_URL")
    code, _, err = _run(["--llm", "env"])
    assert code == 2 and "RODIT_LLM_BASE_URL" in err


def test_cli_env_mode_refuses_a_base_url_with_userinfo(monkeypatch):
    """地址里带用户名密码：客户端构造时就拒绝，报错里不带地址（以前要到发请求才出错，
    没写端口时密码还会被打出来）。"""
    monkeypatch.setenv("RODIT_LLM_BASE_URL", "https://user:pw@llm.example.invalid/v1")
    monkeypatch.setenv("RODIT_LLM_MODEL", "synthetic-model")
    monkeypatch.setenv("RODIT_LLM_API_KEY", FAKE_KEY)
    code, out, err = _run(["--llm", "env"])
    assert code == 2 and out == ""
    assert "base_url" in err and "Traceback" not in err
    for part in ("user:pw", "llm.example.invalid", FAKE_KEY):
        assert part not in err


def test_cli_env_mode_preflight_failure_stops_before_running():
    fake = FakeLLMClient(handler=lambda r: LLMError("大模型接口拒绝请求（HTTP 401）"))
    code, out, err = _run(["--llm", "env"], client_factory=lambda: fake)
    assert code == 2 and out == ""
    assert "评测没有运行：大模型连通性检查失败" in err and "HTTP 401" in err
    assert "路由、质检、生成没过" in err and "Traceback" not in err
    # 只发了检查用的 5 次请求（路由 3 票、质检、生成各 1 次），没有开跑，也没有重试
    assert sorted(r.purpose for r in fake.calls) == ["generate", "qa", "route", "route", "route"]
    assert all(PREFLIGHT_QUESTION in r.messages[-1].content for r in fake.calls)


def test_cli_skip_preflight_sends_no_check_request():
    """--skip-preflight（只在调试时用）：一条检查请求都不发，检查过不了的客户端也照样开跑。"""
    model = ScriptedModel()

    def refuses_the_check(request: LLMRequest):
        if PREFLIGHT_QUESTION in request.messages[-1].content:  # 连通性检查用的合成会话
            return LLMError("合成的故障：检查请求被拒")
        return model(request)

    argv = ["--llm", "env", "--limit", "2", "--seed", "7", "--json"]
    blocked = FakeLLMClient(handler=refuses_the_check, model="synthetic-llm")
    code, out, err = _run(argv, client_factory=lambda: blocked)
    assert code == 2 and out == "" and "连通性检查失败" in err  # 默认：检查没过就不跑

    skipped = FakeLLMClient(handler=refuses_the_check, model="synthetic-llm")
    code, out, err = _run([*argv, "--skip-preflight"], client_factory=lambda: skipped)
    assert code == 0, err
    assert "连通性检查" not in err
    assert not any(PREFLIGHT_QUESTION in r.messages[-1].content for r in skipped.calls)
    data = json.loads(out)
    assert data["n"] == 2 and len(skipped.calls) == data["usage"]["calls"] > 0  # 全是评测自己的请求


def test_cli_hands_the_config_to_preflight_and_prints_its_summary(monkeypatch, tmp_path):
    """main 把这次评测的配置交给 preflight，返回的每一行在开跑前打到标准错误；
    PreflightFailed 的说明原样报出（可以多行），退出码 2。"""
    seen: list[tuple[Any, Any]] = []

    def summarising(client, config=None):
        seen.append((client, config))
        return ["合成的摘要：路由 3 票 0.42 秒", "合成的摘要：生成 1.10 秒"]

    monkeypatch.setattr("rodit.eval.cli.preflight", summarising)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("decision:\n  timeout_s: 3.0\n", encoding="utf-8")
    with_config, without_config = _fake(), _fake()
    code, _, err = _run(
        ["--llm", "env", "--limit", "1", "--config", str(config_path)],
        client_factory=lambda: with_config,
    )
    assert code == 0, err
    assert "合成的摘要：路由 3 票 0.42 秒\n合成的摘要：生成 1.10 秒\n" in err
    assert err.index("合成的摘要：生成") < err.index("[1/1]")  # 摘要在第一条用例之前
    code, _, err = _run(["--llm", "env", "--limit", "1"], client_factory=lambda: without_config)
    assert code == 0, err
    (first_client, first_config), (second_client, second_config) = seen
    assert first_client is with_config and first_config.decision.timeout_s == 3.0
    assert second_client is without_config and second_config is None  # 没给 --config

    def failing(client, config=None):
        raise PreflightFailed("连通性检查没过（合成）：\n  路由：3 票里有 3 票超时（1 秒）")

    monkeypatch.setattr("rodit.eval.cli.preflight", failing)
    never_called = _fake()
    code, out, err = _run(["--llm", "env"], client_factory=lambda: never_called)
    assert code == 2 and out == "" and never_called.calls == []
    assert "连通性检查没过（合成）：\n  路由：3 票里有 3 票超时（1 秒）" in err
    assert "Traceback" not in err and "大模型连通性检查失败" not in err  # 不再包一层

    # 跳过检查时根本不调 preflight
    code, _, err = _run(
        ["--llm", "env", "--limit", "1", "--skip-preflight"], client_factory=lambda: _fake()
    )
    assert code == 0, err


# ---------------------------------------------------------------- §13.5 回归比较


def test_cli_baseline_detects_regression_on_common_cases(tmp_path):
    base = tmp_path / "base.json"
    code, _, err = _run(
        ["--llm", "env", "--intent", "size_advice", "--json", str(base)],
        client_factory=lambda: _fake(),
    )
    assert code == 0, err
    size = [c for c in cases() if c.expected_intent == Intent.SIZE_ADVICE]
    worse = frozenset([next(c.id for c in size if c.split == "dev")])  # 一条开发集用例答错
    code, out, err = _run(
        ["--llm", "env", "--intent", "size_advice", "--baseline", str(base)],
        client_factory=lambda: _fake(wrong=worse),
    )
    assert code == 1
    assert "== §13.5 回归比较（基线 base.json" in out and "↓回归" in out and "不能上线" in out
    # 退出码由进程在标准错误里说实际返回的那个；报告里不写（不算数的一轮返回的是 3）
    assert "退出码" not in out and "回归比较有指标下降超过容差（退出码 1）" in err
    # 容差放宽到超过一条用例的比例：不算回归
    code, out, _ = _run(
        ["--llm", "env", "--intent", "size_advice", "--baseline", str(base), "--tolerance", "0.5"],
        client_factory=lambda: _fake(wrong=worse),
    )
    assert code == 0 and "指标没有下降" in out
    # 基线的一部分：只比较共同用例
    code, out, _ = _run(
        [
            "--llm",
            "env",
            "--intent",
            "size_advice",
            "--limit",
            "2",
            "--baseline",
            str(base),
            "--json",
        ],
        client_factory=lambda: _fake(),
    )
    data = json.loads(out)
    assert code == 0 and data["baseline"]["common_cases"] == 2
    assert data["baseline"]["only_baseline"] == len(size) - 2 and data["baseline"]["passed"]
