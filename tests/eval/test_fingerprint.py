"""评测结论的指纹（方案 §13.5 换模型、改提示词之前先在测试集上跑回归）与上线门槛的测试集依据
（§13.2、§13.4）：--json 带上配置指纹、问题定义版本、代码版本；只有和线上配置一致、用真实大模型跑的
结论才能作为测试集依据。报告要跑完整个评测集；真实大模型跑的还要"评到了模型"（运行健康度
rodit.eval.health：路由退回关键词规则、质检判断层不可用、生成没有可用草稿……超过上限的不收），
结论里带上这些计数。数据全部合成，不联网。"""

from __future__ import annotations

import io
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.config import EngineConfig, config_fingerprint
from rodit.core.models import AuditEvent
from rodit.core.questions import QA_QUESTIONS_VER, ROUTER_QUESTIONS_VER
from rodit.eval.cli import main
from rodit.eval.dataset import cases
from rodit.eval.fingerprint import (
    GIT_SHA_ENV,
    EvalFingerprint,
    EvalVerdict,
    ModelNotEvaluated,
    current_fingerprint,
    describe_health,
    describe_verdict,
    git_revision,
    parse_verdicts,
    same_setup,
    test_set_evidence,
    verdict_from_report,
)
from rodit.eval.health import DEFAULT_CEILING, HealthCounts, RunHealth, assess

T0 = datetime(2026, 9, 25, 10, 0, tzinfo=CST)
# 线上的大模型（合成的地址和模型名）：线上的指纹要知道用的是哪个模型
LLM_ENV = {
    "RODIT_LLM_BASE_URL": "https://llm.example.invalid/v1",
    "RODIT_LLM_MODEL": "synthetic-model",
}
RUNNING = current_fingerprint(EngineConfig(), environ=LLM_ENV)


def other_config() -> EngineConfig:
    config = EngineConfig()
    config.routing.assist_threshold = 0.75
    return config


# ================================================================ 指纹


def test_fingerprint_names_the_config_and_question_versions():
    fp = current_fingerprint(EngineConfig(), environ=LLM_ENV)
    assert fp.config_fingerprint == config_fingerprint(EngineConfig())
    assert fp.router_questions_ver == ROUTER_QUESTIONS_VER
    assert fp.qa_questions_ver == QA_QUESTIONS_VER
    assert fp.git_sha is None and fp.classifier_sha256 is None  # 不查代码版本时不填
    changed = current_fingerprint(other_config(), environ=LLM_ENV)
    assert changed.config_fingerprint != fp.config_fingerprint
    assert not same_setup(changed, fp)
    assert same_setup(fp, fp.model_copy(update={"git_sha": "abc", "git_dirty": True}))
    assert not same_setup(fp, fp.model_copy(update={"qa_questions_ver": "other"}))


def test_git_revision_prefers_the_environment_then_the_repository(tmp_path: Path):
    assert git_revision(environ={GIT_SHA_ENV: " deadbeef "}) == ("deadbeef", None)
    assert git_revision(cwd=tmp_path, environ={}) == (None, None)  # 不是 git 仓库
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    git = ["git", "-C", str(tmp_path), "-c", "user.email=t@example.invalid", "-c", "user.name=t"]
    subprocess.run([*git, "add", "a.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "合成"], check=True)
    sha, dirty = git_revision(cwd=tmp_path, environ={})
    assert sha is not None and len(sha) == 40 and dirty is False
    (tmp_path / "a.txt").write_text("b", encoding="utf-8")
    assert git_revision(cwd=tmp_path, environ={}) == (sha, True)


def test_json_report_carries_the_fingerprint(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(GIT_SHA_ENV, "0123abcd")
    target = tmp_path / "e.json"
    out = io.StringIO()
    assert main(["--split", "test", "--json", str(target)], out=out) == 0
    data = json.loads(target.read_text(encoding="utf-8"))
    fp = data["fingerprint"]
    assert fp["config_fingerprint"] == config_fingerprint(EngineConfig())
    assert fp["router_questions_ver"] == ROUTER_QUESTIONS_VER
    assert fp["qa_questions_ver"] == QA_QUESTIONS_VER
    assert fp["git_sha"] == "0123abcd"
    assert "配置指纹" in out.getvalue()
    verdict = verdict_from_report(data)
    assert verdict.llm_mode == "offline" and verdict.split == "test"
    assert verdict.test_n == data["gate"]["n"] and verdict.gate_passes == data["gate"]["passes"]
    assert verdict.fingerprint == EvalFingerprint.model_validate(fp)
    assert len(verdict.report_sha256) == 64
    # 离线替身不调大模型：报告里没有运行健康度，结论里也没有
    assert data["health"] is None
    assert verdict.health_overall is None and verdict.health_test is None


def test_json_report_with_a_config_file_fingerprints_that_config(tmp_path: Path):
    config_path = tmp_path / "rodit.yaml"
    config_path.write_text("routing:\n  assist_threshold: 0.75\n", encoding="utf-8")
    target = tmp_path / "e.json"
    argv = ["--split", "test", "--limit", "3", "--config", str(config_path)]
    assert main([*argv, "--json", str(target)], out=io.StringIO()) == 0
    fp = json.loads(target.read_text(encoding="utf-8"))["fingerprint"]
    assert fp["config_fingerprint"] == config_fingerprint(other_config())


# ================================================================ 从报告到结论

# 一轮干净的全量评测：324 条里 233 条有草稿送检、250 条调过生成；测试集 66 条里 48、52 条
OVERALL = {"cases": 324, "qa_judged": 233, "generation_attempted": 250}
TEST = {"cases": 66, "qa_judged": 48, "generation_attempted": 52}
OFFLINE = {"mode": "offline", "model": "offline-evidence-echo", "decision_provider": "rules"}


def health(overall: dict | None = None, test: dict | None = None, **over) -> dict:
    """报告里的 health（rodit.eval.health.RunHealth 的 JSON；valid / reasons 像评测命令行那样按
    assess 填）。overall / test 改全部用例、测试集的计数，over 改这一块里别的键。"""
    all_counts = HealthCounts(**{**OVERALL, **(overall or {})})
    test_counts = HealthCounts(**{**TEST, **(test or {})})
    reasons = assess(all_counts, test_counts)
    block = RunHealth(
        ceiling=DEFAULT_CEILING,
        overall=all_counts,
        test=test_counts,
        valid=not reasons,
        reasons=reasons,
    ).model_dump(mode="json")
    block.update(over)
    return block


def report(**over) -> dict:
    data = {
        "split": "all",
        "n": 324,
        "intent": None,
        "limit": None,
        "gate": {
            "split": "test",
            "n": 66,
            "threshold": 0.95,
            "factual": 0.97,
            "compliance": 1.0,
            "passes": True,
        },
        "llm": {"mode": "env", "model": "synthetic-model", "decision_provider": "incloud_llm"},
        "health": health(),
        "baseline": None,
        "fingerprint": RUNNING.model_dump(mode="json"),
        "cases": [],
    }
    data.update(over)
    return data


def test_verdict_refuses_reports_that_cannot_back_the_test_set_gate():
    with pytest.raises(ValueError, match="配置指纹"):
        verdict_from_report(report(fingerprint=None))
    with pytest.raises(ValueError, match="部分评测"):
        verdict_from_report(report(limit=10))
    with pytest.raises(ValueError, match="部分评测"):
        verdict_from_report(report(intent="size_advice"))
    with pytest.raises(ValueError, match="测试集"):
        verdict_from_report(report(split="dev", gate=None))
    with pytest.raises(ValueError, match="不是评测报告"):
        verdict_from_report(["not", "a", "report"])
    v = verdict_from_report(report(baseline={"passed": False, "deltas": []}))
    assert v.baseline_passed is False and v.gate_passes and v.model == "synthetic-model"


@pytest.mark.parametrize(("split", "n"), [("all", 323), ("all", 66), ("test", 65), ("test", 324)])
def test_a_report_that_did_not_run_the_whole_split_is_refused(split: str, n: int):
    """条数对不上评测集（跑到一半停了、被打断）的报告不是完整的评测：不收，离线替身跑的也一样。"""
    size = len(cases(split))  # type: ignore[arg-type]
    for over in ({}, {"llm": OFFLINE, "health": None}):
        with pytest.raises(ValueError, match="报告不完整") as exc:
            verdict_from_report(report(split=split, n=n, **over))
        assert f"共 {size} 条" in str(exc.value) and f"报告里是 {n} 条" in str(exc.value)


def test_the_synthetic_reports_here_are_the_size_of_the_evaluation_set():
    assert (len(cases("all")), len(cases("test"))) == (OVERALL["cases"], TEST["cases"]) == (324, 66)
    only_test = report(split="test", n=66, health=health(overall=TEST))
    assert verdict_from_report(only_test).n == 66
    with pytest.raises(ValueError, match="字段不对：split"):  # 不认识的 split：没有对应的评测集
        verdict_from_report(report(split="holdout"))


# ================================================================ 运行健康度（rodit.eval.health）


def test_an_env_report_keeps_the_health_counts_in_the_verdict():
    v = verdict_from_report(report())
    assert v.llm_mode == "env"
    assert v.health_overall == HealthCounts(**OVERALL) and v.health_test == HealthCounts(**TEST)
    # 只跑测试集（--split test）：全部用例就是测试集，health 里没单独写 test 的也一样
    only_test = report(split="test", n=66, health=health(overall=TEST))
    v = verdict_from_report(only_test)
    assert v.health_overall == v.health_test == HealthCounts(**TEST)
    only_test["health"]["test"] = None
    assert verdict_from_report(only_test).health_test == HealthCounts(**TEST)


@pytest.mark.parametrize(
    ("group", "field", "limit"),
    [
        ("overall", "route_fallback", 6),
        ("overall", "qa_unavailable", 4),
        ("overall", "generation_unusable", 5),
        ("overall", "engine_errors", 6),
        ("overall", "route_lost_votes", 32),
        ("test", "route_fallback", 1),
        ("test", "qa_unavailable", 1),
        ("test", "generation_unusable", 1),
        ("test", "engine_errors", 1),
        ("test", "route_lost_votes", 6),
    ],
)
def test_each_health_rule_refuses_the_report_only_above_its_limit(
    group: str, field: str, limit: int
):
    """assess 的每条规则（全部用例、测试集各看一遍）：刚好在上限上的照收、计数记进结论，
    多一条就不收，原因写明条数和上限。"""

    def counted(count: int) -> dict:
        # 测试集是全部用例的一部分：测试集里出问题的用例也算在全部用例里
        test = {field: count} if group == "test" else None
        return report(health=health(overall={field: count}, test=test))

    v = verdict_from_report(counted(limit))
    kept = v.health_test if group == "test" else v.health_overall
    assert kept is not None and getattr(kept, field) == limit
    with pytest.raises(ModelNotEvaluated) as exc:
        verdict_from_report(counted(limit + 1))
    [reason] = exc.value.reasons
    assert reason.startswith("测试集：") == (group == "test")
    assert f"有 {limit + 1} 条" in reason and f"上限 {limit} 条" in reason
    assert reason in str(exc.value) and "没有有效评到大模型" in str(exc.value)


def test_a_run_that_did_not_evaluate_the_model_is_refused_with_every_reason():
    """路由的判断层大面积调不通、质检也跟着不可用：评测照样跑完、给出一组数字，但量的是兜底逻辑。"""
    degraded = health(
        overall={"route_fallback": 105, "qa_unavailable": 40}, test={"route_fallback": 20}
    )
    with pytest.raises(ModelNotEvaluated) as exc:
        verdict_from_report(report(health=degraded))
    assert exc.value.reasons == [
        "路由：324 条里有 105 条（32.4%）没用上判断层（退回关键词规则），上限 6 条。",
        "质检：有草稿送检的 233 条里有 40 条（17.2%）质检判断层不可用（只跑了硬规则），上限 4 条。",
        "测试集：路由：66 条里有 20 条（30.3%）没用上判断层（退回关键词规则），上限 1 条。",
    ]
    assert isinstance(exc.value, ValueError)  # 和别的不能用的报告一样：record-eval 退出码 2
    assert all(reason in str(exc.value) for reason in exc.value.reasons)


def test_the_health_verdict_is_recomputed_not_read_from_the_report():
    """报告自己说算数（valid、放宽了的 ceiling）不作数：固定按 DEFAULT_CEILING 重新算。"""
    lenient = health(overall={"route_fallback": 105}, valid=True, reasons=[], ceiling=1.0)
    with pytest.raises(ModelNotEvaluated, match="有 105 条"):
        verdict_from_report(report(health=lenient))
    # 反过来：评测时用了更严的上限、报告里写着不算数，按默认上限算数的照收
    strict = health(overall={"route_fallback": 3}, valid=False, reasons=["路由：…"], ceiling=0.0)
    v = verdict_from_report(report(health=strict))
    assert v.health_overall is not None and v.health_overall.route_fallback == 3
    # 时间预算截断（§12.1）是正常行为：照实记下，不算"模型没被评到"
    v = verdict_from_report(report(health=health(overall={"budget_cut": 200})))
    assert v.health_overall is not None and v.health_overall.budget_cut == 200


def test_health_fields_added_later_are_ignored():
    """评测命令行以后给 health 加的字段（docs/架构与分工.md §4.4：只加带默认值的）读报告时不管。"""
    block = health()
    block["aborted_after"] = None
    block["overall"]["vision_failed"] = 2
    block["test"]["vision_failed"] = 0
    block["by_purpose"] = {"route": {"calls": 972, "retried": 3}}
    v = verdict_from_report(report(health=block))
    assert v.health_overall == HealthCounts(**OVERALL) and v.health_test == HealthCounts(**TEST)
    assert "vision_failed" not in json.dumps(v.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ("degraded", "字段不对：health"),
        ({"valid": True}, "字段不对：health.overall"),
        ({"overall": {"route_fallback": 0}}, "字段不对：health.overall.cases"),
        ({"overall": {**OVERALL, "route_fallback": -1}, "test": TEST}, "route_fallback"),
        ({"overall": OVERALL, "test": {**TEST, "qa_unavailable": "很多"}}, "health.test"),
        # 计数说的不是这份报告：全部用例、测试集的条数要和报告一样，测试集的计数不能没有
        ({"overall": {**OVERALL, "cases": 300}, "test": TEST}, "对不上"),
        ({"overall": {"cases": 0}, "test": {"cases": 0}}, "对不上"),
        ({"overall": OVERALL, "test": {**TEST, "cases": 60}}, "对不上"),
        ({"overall": OVERALL, "test": None}, "对不上"),
        ({"overall": OVERALL}, "对不上"),
    ],
)
def test_a_health_block_that_cannot_be_read_or_is_not_about_this_report_is_refused(
    block: object, message: str
):
    with pytest.raises(ValueError, match=message) as exc:
        verdict_from_report(report(health=block))
    assert not isinstance(exc.value, ModelNotEvaluated)


def test_an_env_report_without_the_health_block_is_refused():
    """真实大模型跑的报告里没有 health（旧版本的评测命令行生成的，或者评测没有跑完）：看不出模型
    有没有被有效评到，不收，提示用本版本重新跑。"""
    without_key = report()
    del without_key["health"]
    for data in (report(health=None), without_key):
        with pytest.raises(ValueError, match="没有运行健康度") as exc:
            verdict_from_report(data)
        assert "重新跑" in str(exc.value) and not isinstance(exc.value, ModelNotEvaluated)


def test_offline_reports_need_no_health_block():
    """离线替身不调大模型，没有运行健康度可看：和以前一样照收（只作回归基线），结论里不带计数。"""
    for block in (None, "不看", health(overall={"route_fallback": 324})):
        v = verdict_from_report(report(llm=OFFLINE, health=block))
        assert v.llm_mode == "offline" and v.health_overall is None and v.health_test is None
    without_key = report(llm=OFFLINE)
    del without_key["health"]
    assert verdict_from_report(without_key).health_overall is None


def test_describe_verdict_mentions_the_health_counts_when_the_verdict_has_them():
    v = verdict_from_report(report())
    plain = (
        "2026-09-25 10:00 的评测（synthetic-model，测试集 66 条）："
        "事实正确 97.0%、合规 100.0%（门槛 ≥ 95.0%），达标"
    )
    counts = "路由退回 0/324、质检不可用 0/233、生成没有可用草稿 0/250"
    assert describe_verdict(T0, v) == f"{plain}；运行健康度：{counts}"
    without = v.model_copy(update={"health_overall": None, "health_test": None})
    assert describe_verdict(T0, without) == plain  # 没有计数的结论：和以前一样
    assert describe_health(HealthCounts(**TEST)) == (
        "路由退回 0/66、质检不可用 0/48、生成没有可用草稿 0/52"
    )
    # 丢票、引擎异常、被时间预算截断：有才写
    busy = HealthCounts(
        **OVERALL,
        route_fallback=2,
        qa_unavailable=1,
        generation_unusable=3,
        route_lost_votes=12,
        engine_errors=1,
        budget_cut=9,
    )
    assert describe_health(busy) == (
        "路由退回 2/324、质检不可用 1/233、生成没有可用草稿 3/250、"
        "路由丢票 12/324、引擎异常 1/324、被时间预算截断 9/324"
    )
    failing = {**report()["gate"], "factual": 0.9, "passes": False}
    text = describe_verdict(T0, verdict_from_report(report(gate=failing)))
    assert text.endswith(f"，没达标；运行健康度：{counts}")


# ================================================================ 测试集依据


def verdict_event(at: datetime, **over) -> AuditEvent:
    verdict = verdict_from_report(report(**over))
    return AuditEvent(
        shop_id=None,
        actor="system",
        action=AuditAction.EVAL_VERDICT.value,
        detail=verdict.model_dump(mode="json"),
        created_at=at,
    )


def test_no_verdict_leaves_the_test_set_unchecked_with_a_reason():
    ev = test_set_evidence([], RUNNING)
    assert ev.passed is None and "record-eval" in (ev.reason or "")
    assert ev.running_fingerprint == RUNNING.config_fingerprint


def test_a_matching_real_model_verdict_decides_the_test_set():
    ev = test_set_evidence([verdict_event(T0)], RUNNING)
    assert ev.passed is True and ev.reason is None and ev.verdict_at == T0
    assert "事实正确 97.0%" in ev.detail and "synthetic-model" in ev.detail
    assert "路由退回 0/324、质检不可用 0/233" in ev.detail  # 依据里看得到这一轮评到了模型
    failing = {**report()["gate"], "factual": 0.9, "passes": False}
    later = verdict_event(T0 + timedelta(hours=1), gate=failing)
    ev = test_set_evidence([verdict_event(T0), later], RUNNING)
    assert ev.passed is False and ev.verdict_at == T0 + timedelta(hours=1)  # 最近一次为准
    regressed = verdict_event(T0 + timedelta(hours=2), baseline={"passed": False})
    ev = test_set_evidence([verdict_event(T0), regressed], RUNNING)
    assert ev.passed is False and "下降" in ev.detail


def test_a_verdict_for_another_config_leaves_the_test_set_unchecked():
    stale = current_fingerprint(other_config()).model_dump(mode="json")
    ev = test_set_evidence([verdict_event(T0, fingerprint=stale)], RUNNING)
    assert ev.passed is None and "不一致" in (ev.reason or "") and "配置" in (ev.reason or "")
    assert ev.config_fingerprint == stale["config_fingerprint"]
    # 更早的一次和线上一致：以一致的那次为准
    ev = test_set_evidence(
        [verdict_event(T0), verdict_event(T0 + timedelta(hours=1), fingerprint=stale)], RUNNING
    )
    assert ev.passed is True and ev.verdict_at == T0
    questions = RUNNING.model_copy(update={"router_questions_ver": "old"}).model_dump(mode="json")
    ev = test_set_evidence([verdict_event(T0, fingerprint=questions)], RUNNING)
    assert ev.passed is None and "路由问题" in (ev.reason or "")


def test_offline_verdicts_are_regression_baselines_not_launch_evidence():
    offline = {"mode": "offline", "model": "offline-evidence-echo", "decision_provider": "rules"}
    ev = test_set_evidence([verdict_event(T0, llm=offline)], RUNNING)
    assert ev.passed is None and "离线替身" in (ev.reason or "")


def test_unreadable_or_shop_events_are_ignored():
    broken = AuditEvent(
        shop_id=None,
        actor="system",
        action=AuditAction.EVAL_VERDICT.value,
        detail={"fingerprint": "x"},
        created_at=T0,
    )
    shop_event = verdict_event(T0).model_copy(update={"shop_id": "shop_a"})
    assert test_set_evidence([broken, shop_event], RUNNING).passed is None
    assert EvalVerdict.model_validate(verdict_event(T0).detail).gate_passes


def test_verdicts_recorded_before_the_health_counts_still_count():
    """以前记下的 eval.verdict 没有 health_overall / health_test：照样读得出来、照样能当依据。"""
    detail = {
        "fingerprint": RUNNING.model_dump(mode="json"),
        "llm_mode": "env",
        "model": "synthetic-model",
        "decision_provider": "incloud_llm",
        "split": "all",
        "n": 324,
        "test_n": 66,
        "threshold": 0.95,
        "factual": 0.97,
        "compliance": 1.0,
        "gate_passes": True,
        "baseline_passed": None,
        "report_sha256": "0" * 64,
    }
    old = AuditEvent(
        shop_id=None,
        actor="system",
        action=AuditAction.EVAL_VERDICT.value,
        detail=detail,
        created_at=T0,
    )
    [(at, v)] = parse_verdicts([old])
    assert at == T0 and v.health_overall is None and v.health_test is None
    ev = test_set_evidence([old], RUNNING)
    assert ev.passed is True and "事实正确 97.0%" in ev.detail and "运行健康度" not in ev.detail
    # 现在记的结论带计数，存进事件再读出来还是这些计数
    [(_, new)] = parse_verdicts([verdict_event(T0)])
    assert new.health_overall == HealthCounts(**OVERALL) and new.health_test == HealthCounts(**TEST)
    assert set(verdict_event(T0).detail) == {*detail, "health_overall", "health_test"}
