"""评测的耗时（§12.1）、用量与成本（§14）、回归比较（§13.5）、部分评测，以及离线模式的命令行。"""

from __future__ import annotations

import io
import json

import pytest

from rodit.core.enums import Intent
from rodit.eval.baseline import compare_to_baseline, load_baseline
from rodit.eval.cli import evaluate, main, render_text
from rodit.eval.dataset import CASES, cases
from rodit.eval.llm_env import describe_endpoint, resolve_pricing
from rodit.eval.perf import (
    Pricing,
    calls_by_purpose,
    case_usage,
    is_timeout,
    latency_report,
    usage_report,
)
from rodit.eval.runner import select_cases


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------- §12.1 耗时


def test_latency_report_percentiles_and_targets():
    rows = [
        {
            "understand": 5,
            "route": 100 * i,
            "retrieve": 10,
            "generate": 0,
            "qa": 50,
            "total": 1000 * i,
        }
        for i in range(1, 11)
    ]
    report = latency_report([*rows, {}])  # 没有结果的用例（空）不计入
    assert report.n == 10
    route = report.stages["route"]
    assert (route.count, route.p50, route.p95, route.max) == (10, 500, 1000, 1000)  # 最近秩法
    assert route.target_ms == 1000 and route.within_target is True
    total = report.stages["total"]
    assert total.p95 == 10000 and total.within_target is False
    assert report.e2e_p95_within_target is False
    assert report.stages["understand"].target_ms is None
    assert report.stages["understand"].within_target is None
    assert report.stages["generate"].p95 == 0  # 没跑的环节记 0
    empty = latency_report([])
    assert empty.n == 0 and empty.stages["total"].p50 is None
    assert empty.e2e_p95_within_target is None
    # 没给调用记录、没有被截断的用例：和原来一样，没有"被截短"这回事
    assert not report.limited_by_budget and (report.budget_cut, report.llm_timeouts) == (0, 0)
    assert all(s.llm_calls == s.llm_timeouts == 0 for s in report.stages.values())


def test_latency_report_marks_values_cut_short_by_timeouts_or_the_budget():
    """单次调用到超时就断、整条消息到时间预算就返回：端到端超不过 5 秒的目标。有调用超时或有用例
    被截断时，没超标也不算达标（None）；分位数本身不改。"""
    rows = [{"route": 900, "generate": 2000, "qa": 800, "total": 4000}] * 20
    answered = [{"purpose": "route", "latency_ms": 300}] * 57 + [
        {"purpose": "generate", "latency_ms": 900}
    ] * 4
    failed = [
        {"purpose": "route", "latency_ms": 1005, "error": "LLMTimeout", "kind": "timeout"},
        {"purpose": "route", "latency_ms": 1005, "error": "LLMTimeout"},  # 还没带 kind 的事件
        {"purpose": "qa", "latency_ms": 20, "error": "LLMError", "kind": "rate_limit"},
    ]
    clean = latency_report(rows, answered)
    assert clean.e2e_p95_within_target is True and not clean.limited_by_budget
    assert clean.stages["route"].llm_calls == 57 and clean.stages["generate"].llm_calls == 4
    assert clean.stages["total"].llm_calls == clean.stages["retrieve"].llm_calls == 0
    report = latency_report(rows, [*answered, *failed])
    route = report.stages["route"]
    assert (route.llm_calls, route.llm_timeouts, route.within_target) == (59, 2, True)
    qa = report.stages["qa"]
    assert (qa.llm_calls, qa.llm_timeouts) == (1, 0)  # 被限流不是超时
    assert report.llm_timeouts == 2 and report.limited_by_budget and report.budget_cut == 0
    total = report.stages["total"]
    assert total.p95 == 4000 and total.within_target is None
    assert report.e2e_p95_within_target is None
    cut = latency_report(rows, budget_cut=3)  # 没有调用超时，但有用例被时间预算截断
    assert cut.limited_by_budget and cut.budget_cut == 3 and cut.e2e_p95_within_target is None
    over = latency_report([{"total": 6000}], budget_cut=1)  # 被截短的数都超过目标：就是超标
    assert over.e2e_p95_within_target is False
    assert is_timeout({"kind": "timeout"}) and is_timeout({"error": "LLMTimeout"})
    assert not is_timeout({"error": "LLMError", "kind": "rate_limit"}) and not is_timeout({})


def test_latency_report_withholds_the_verdict_when_the_model_was_not_evaluated():
    """这一轮没有有效评到模型（evaluated=False）：调大模型的环节和端到端量到的是兜底路径，
    没超标也不算达标（None）；不调大模型的环节、超标的照旧，分位数不改。"""
    rows = [{"understand": 1, "route": 3, "retrieve": 20, "generate": 0, "qa": 0, "total": 30}] * 20
    valid = latency_report(rows)
    assert not valid.not_evaluated and valid.e2e_p95_within_target is True
    assert all(valid.stages[k].within_target is True for k in ("route", "generate", "qa"))
    report = latency_report(rows, evaluated=False)
    assert report.not_evaluated and not report.limited_by_budget
    assert report.e2e_p95_within_target is None
    for key in ("route", "generate", "qa", "total"):
        assert report.stages[key].within_target is None and report.stages[key].target_ms
        assert report.stages[key].p95 == valid.stages[key].p95
    assert report.stages["retrieve"].within_target is True  # 检索不调大模型：照旧
    assert report.stages["understand"].within_target is None  # 本来就没有目标
    slow = latency_report([{"route": 1500, "retrieve": 900, "total": 6000}], evaluated=False)
    assert slow.stages["route"].within_target is False  # 超标就是超标
    assert slow.stages["retrieve"].within_target is False
    assert slow.e2e_p95_within_target is False
    assert latency_report([], evaluated=False).e2e_p95_within_target is None


# ---------------------------------------------------------------- §14 用量与成本


CALLS = [
    {"purpose": "route", "input_tokens": 600, "output_tokens": 40, "latency_ms": 300},
    {"purpose": "route", "input_tokens": 600, "output_tokens": 40, "latency_ms": 500},
    {
        "purpose": "route",
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": 1000,
        "error": "LLMTimeout",
    },
    {"purpose": "generate", "input_tokens": 2000, "output_tokens": 150, "latency_ms": 2000},
    {"purpose": "qa", "input_tokens": 1200, "output_tokens": 30, "latency_ms": 400},
]


def test_case_usage_and_cost():
    pricing = Pricing(input_per_1k=0.004, output_per_1k=0.016)
    usage = case_usage(CALLS, pricing)
    assert (usage.calls, usage.errors) == (5, 1)
    assert (usage.input_tokens, usage.output_tokens) == (4400, 260)
    assert usage.cost == pytest.approx(4.4 * 0.004 + 0.26 * 0.016)
    assert list(usage.by_purpose) == ["route", "generate", "qa"]
    assert usage.by_purpose["route"].calls == 3 and usage.by_purpose["route"].errors == 1
    assert case_usage(CALLS).cost is None  # 没有单价只报 token
    assert case_usage([]).calls == 0


def test_usage_report_totals_shares_and_projection():
    report = usage_report([CALLS, CALLS[:1], []], Pricing(input_per_1k=1.0, output_per_1k=2.0))
    assert report.cases == 3 and report.calls == 6 and report.errors == 1
    assert report.error_types == {"LLMTimeout": 1}
    assert report.input_tokens == 5000 and report.output_tokens == 300
    assert report.total_tokens == 5300
    route = report.by_purpose["route"]
    assert route.calls == 4 and route.token_share == pytest.approx((1800 + 120) / 5300)
    assert route.call_latency_ms.count == 4 and route.call_latency_ms.p95 == 1000
    assert sum(p.token_share for p in report.by_purpose.values()) == pytest.approx(1.0)
    assert report.cost == pytest.approx(5.0 + 0.6)
    assert report.mean_tokens_per_case == pytest.approx(5300 / 3)
    assert report.projected_monthly_tokens_per_shop == pytest.approx(5300 / 3 * 900 * 30)
    assert report.projected_monthly_cost_per_shop == pytest.approx(5.6 / 3 * 900 * 30)
    unpriced = usage_report([CALLS])
    assert unpriced.cost is None and unpriced.projected_monthly_cost_per_shop is None


def test_usage_report_says_what_the_totals_leave_out():
    """失败的调用按 0 token 计（另给估算）、成功却没有用量的调用、接口回显的模型名。"""
    served = "synthetic-model-2026-09-01"  # 配置的 synthetic-model 是别名
    calls = [
        {"purpose": "route", "model": served, "input_tokens": 600, "output_tokens": 40},
        {"purpose": "route", "model": served, "input_tokens": 800, "output_tokens": 60},
        {"purpose": "route", "model": "synthetic-model", "error": "LLMTimeout", "kind": "timeout"},
        {
            "purpose": "route",
            "model": "synthetic-model",
            "error": "LLMError",
            "kind": "rate_limit",
            "status": 429,
        },
        # 成功，但接口没有返回 usage
        {"purpose": "generate", "model": served, "input_tokens": 0, "output_tokens": 0},
        {
            "purpose": "qa",
            "model": "synthetic-model",
            "error": "LLMError",
            "kind": "bad_request",
            "status": 400,
            "code": "unsupported_parameter",
        },
    ]
    report = usage_report([calls])
    assert report.total_tokens == 1500  # 合计照旧只算计量到的
    route = report.by_purpose["route"]
    assert route.errors == 2 and route.zero_usage_calls == 0
    assert route.failed_tokens_estimate == pytest.approx(2 * 1500 / 2)  # 成功调用平均 750
    generate = report.by_purpose["generate"]
    assert generate.zero_usage_calls == 1 and generate.failed_tokens_estimate == 0
    qa = report.by_purpose["qa"]
    assert qa.errors == 1 and qa.failed_tokens_estimate is None  # 没有成功的调用可参照
    assert report.failed_tokens_estimate == pytest.approx(1500)  # 只加估得出的
    assert report.zero_usage_calls == 1
    assert report.served_models == {served: 3}  # 失败的调用没有回显，不算
    # 失败原因按 health.error_label 分开（带了 kind / status / code 才细分）
    assert report.error_types == {
        "timeout": 1,
        "http_429": 1,
        "http_400:unsupported_parameter": 1,
    }
    clean = usage_report([CALLS[:2]])
    assert clean.failed_tokens_estimate == 0 and clean.zero_usage_calls == 0
    assert clean.served_models == {}  # 事件里没有模型名
    assert usage_report([[calls[5]]]).failed_tokens_estimate is None
    # 成功的调用都没有用量时，失败的也估不出（不拿 0 当平均值）
    no_usage = usage_report([[calls[4], {"purpose": "generate", "error": "LLMError"}]])
    assert no_usage.by_purpose["generate"].failed_tokens_estimate is None
    assert list(calls_by_purpose(calls, also=["other"])) == ["route", "generate", "qa", "other"]


def test_resolve_pricing_from_flags_and_env():
    assert resolve_pricing(None, None, env={}) is None
    assert resolve_pricing(0.002, 0.008, env={}) == Pricing(input_per_1k=0.002, output_per_1k=0.008)
    env = {"RODIT_LLM_PRICE_INPUT_PER_1K": "0.001", "RODIT_LLM_PRICE_OUTPUT_PER_1K": " 0.003 "}
    assert resolve_pricing(None, None, env=env) == Pricing(input_per_1k=0.001, output_per_1k=0.003)
    assert resolve_pricing(0.5, None, env=env).input_per_1k == 0.5  # 命令行优先
    with pytest.raises(ValueError, match="一起给"):
        resolve_pricing(0.002, None, env={})
    with pytest.raises(ValueError, match="必须是数字"):
        resolve_pricing(None, None, env={**env, "RODIT_LLM_PRICE_INPUT_PER_1K": "abc"})
    with pytest.raises(ValueError, match="非负"):
        resolve_pricing(-1.0, 0.1, env={})
    with pytest.raises(ValueError, match="非负"):
        resolve_pricing(float("nan"), 0.1, env={})


def test_describe_endpoint_keeps_only_host():
    class Client:
        base_url = "https://user:secret@llm.example.invalid:8443/v1?token=abc"

    assert describe_endpoint(Client()) == "https://llm.example.invalid:8443"
    assert describe_endpoint(object()) is None


# ---------------------------------------------------------------- 部分评测


def test_select_cases_by_intent_limit_and_seed():
    all_cases = cases()
    spec = select_cases(all_cases, intent="product_spec")
    assert spec and all(c.expected_intent == Intent.PRODUCT_SPEC for c in spec)
    a = select_cases(all_cases, limit=10, seed=3)
    b = select_cases(all_cases, limit=10, seed=3)
    c = select_cases(all_cases, limit=10, seed=4)
    assert [x.id for x in a] == [x.id for x in b] != [x.id for x in c]
    order = {x.id: i for i, x in enumerate(all_cases)}
    assert [order[x.id] for x in a] == sorted(order[x.id] for x in a)  # 保持数据集顺序
    assert len({x.expected_intent for x in a}) > 1  # 随机抽，不是只取开头一类
    assert select_cases(all_cases, limit=len(all_cases) + 5) == all_cases
    with pytest.raises(ValueError):
        select_cases(all_cases, limit=0)


# ---------------------------------------------------------------- §13.5 回归比较


def _row(  # noqa: PLR0913
    id: str,  # noqa: A002
    split: str = "dev",
    *,
    intent_ok=True,
    action_ok=True,
    factual=2,
    compliance=2,
) -> dict:
    return {
        "id": id,
        "split": split,
        "intent_ok": intent_ok,
        "action_ok": action_ok,
        "scores": {"factual": factual, "on_topic": 2, "compliance": compliance, "expression": 2},
    }


def test_compare_to_baseline_on_common_cases():
    base = [_row("a"), _row("b"), _row("c", "test"), _row("d", "test"), _row("gone")]
    same = compare_to_baseline([_row("a"), _row("b"), _row("c", "test"), _row("new")], base)
    assert same.passed and same.common_cases == 3
    assert (same.only_current, same.only_baseline) == (1, 2)
    assert {d.scope for d in same.deltas} == {"all", "test"}
    worse = compare_to_baseline(
        [_row("a", factual=1), _row("b"), _row("c", "test", compliance=0), _row("d", "test")],
        base,
        source="/some/dir/base.json",
    )
    assert not worse.passed and worse.source == "base.json"
    regressed = {(d.scope, d.metric) for d in worse.regressions}
    assert regressed == {("all", "factual"), ("all", "compliance"), ("test", "compliance")}
    test_compliance = next(
        d for d in worse.deltas if d.scope == "test" and d.metric == "compliance"
    )
    assert (test_compliance.baseline, test_compliance.current, test_compliance.delta) == (
        1.0,
        0.5,
        -0.5,
    )
    tolerant = compare_to_baseline(
        [_row("a"), _row("b", intent_ok=False), _row("c", "test"), _row("d", "test")],
        base,
        tolerance=0.25,
    )
    assert tolerant.passed  # 类型准确率 100% → 75%，恰好在容差内
    better = compare_to_baseline([_row("a")], [_row("a", intent_ok=False, factual=0)])
    assert better.passed and all(d.delta >= 0 for d in better.deltas)
    with pytest.raises(ValueError, match="没有共同"):
        compare_to_baseline([_row("x")], base)
    with pytest.raises(ValueError, match="容差"):
        compare_to_baseline([_row("a")], base, tolerance=1.5)


def test_routing_regression_alone_fails_the_check():
    """升级判断层后不再提醒人工（human_priority 变成 answer）：类型、事实、合规都没变，
    只有路由动作退步——也要算回归（§13.5、§5.7）。"""
    base = [_row("human-01"), _row("emo-01", "test"), _row("ship-01", "test")]
    current = [
        _row("human-01", action_ok=False),
        _row("emo-01", "test", action_ok=False),
        _row("ship-01", "test"),
    ]
    check = compare_to_baseline(current, base)
    assert not check.passed
    assert {(d.scope, d.metric) for d in check.regressions} == {
        ("all", "action_accuracy"),
        ("test", "action_accuracy"),
    }


def test_load_baseline_errors(tmp_path):
    with pytest.raises(ValueError, match="读不了"):
        load_baseline(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="不是 JSON"):
        load_baseline(bad)
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"overall": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="cases"):
        load_baseline(old)
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"cases": [{"split": "dev"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="id"):
        load_baseline(broken)


# ---------------------------------------------------------------- 离线模式的命令行


def test_offline_report_has_cases_latency_and_no_usage():
    report = evaluate("test")
    assert report.llm.mode == "offline" and report.llm.decision_provider == "rules"
    assert report.usage is None and report.limit is None and report.seed is None
    assert [c.id for c in report.cases] == [c.id for c in cases("test")]
    assert all(c.usage is None and "total" in c.latency_ms for c in report.cases)
    assert report.latency.n == report.n
    failure_ids = {f.id for f in report.failures}
    records = {c.id: c for c in report.cases}
    assert all(min(records[i].scores.values()) < 2 for i in failure_ids)
    assert all(f.drafts == [] for f in report.failures)  # 离线替身不记草稿
    text = render_text(report)
    assert "离线替身不调用大模型" in text and "== §12.1 耗时" in text


def test_cli_offline_json_file_and_baseline_round_trip(tmp_path):
    path = tmp_path / "runs" / "offline.json"  # 目录不存在时自动创建
    code, out, err = _run("--split", "test", "--json", str(path))
    assert code == 0, err
    assert "== 总体" in out and f"已写入 {path}" in out  # 写文件时文本报告照常输出
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["n"] == len(cases("test")) == len(data["cases"])
    assert data["llm"]["mode"] == "offline" and data["usage"] is None
    code, out, _ = _run("--split", "test", "--baseline", str(path))
    assert code == 0 and "== §13.5 回归比较" in out and "指标没有下降" in out
    # 基线里的逐条结果更好（合成地改高）：本次就是回归
    for row in data["cases"]:
        row["intent_ok"] = True
        row["scores"]["factual"] = 2
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    code, out, _ = _run("--split", "test", "--baseline", str(path))
    assert code == 1 and "↓回归" in out


def test_cli_partial_runs_and_errors(tmp_path, monkeypatch):
    code, out, _ = _run("--intent", "greeting", "--json")
    data = json.loads(out)
    greeting = [c for c in CASES if c.expected_intent == Intent.GREETING]
    assert code == 0 and data["n"] == len(greeting) and data["intent"] == "greeting"
    code, out, _ = _run("--limit", "5", "--seed", "11")
    assert code == 0 and "随机抽 5 条（seed 11）" in out
    # 筛选后没有用例、单价只给一个、基线文件不存在、配置文件不对：退出码 2，不抛异常
    only_spec = [c for c in CASES if c.expected_intent == Intent.PRODUCT_SPEC]
    with monkeypatch.context() as m:
        m.setattr("rodit.eval.cli.cases", lambda split="all": only_spec)
        code, _, err = _run("--intent", "greeting")
    assert code == 2 and "为空" in err
    code, _, err = _run("--price-input", "0.002")
    assert code == 2 and "一起给" in err
    code, _, err = _run("--baseline", str(tmp_path / "missing.json"))
    assert code == 2 and "读不了" in err
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("decision: [", encoding="utf-8")
    code, _, err = _run("--config", str(bad_config))
    assert code == 2 and "评测没有运行" in err
    for argv in (["--limit", "0"], ["--tolerance", "1.5"], ["--llm", "cloud"]):
        with pytest.raises(SystemExit) as exc:
            main(argv, out=io.StringIO(), err=io.StringIO())
        assert exc.value.code == 2


def test_cli_config_file_is_used(tmp_path):
    config = tmp_path / "rodit.yaml"
    config.write_text("routing:\n  assist_threshold: 0.99\n", encoding="utf-8")
    code, out, _ = _run("--split", "test", "--config", str(config), "--json")
    strict = json.loads(out)
    code2, out2, _ = _run("--split", "test", "--json")
    default = json.loads(out2)
    assert code == code2 == 0
    # 阈值调到 0.99：关键词规则的置信度达不到，出推荐的比例下降
    assert strict["overall"]["suggestion_rate"] < default["overall"]["suggestion_rate"]
