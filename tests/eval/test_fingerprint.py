"""评测结论的指纹（方案 §13.5 换模型、改提示词之前先在测试集上跑回归）与上线门槛的测试集依据
（§13.2、§13.4）：--json 带上配置指纹、问题定义版本、代码版本；只有和线上配置一致、用真实大模型跑的
结论才能作为测试集依据。数据全部合成，不联网。"""

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
from rodit.eval.fingerprint import (
    GIT_SHA_ENV,
    EvalFingerprint,
    EvalVerdict,
    current_fingerprint,
    git_revision,
    same_setup,
    test_set_evidence,
    verdict_from_report,
)

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


def test_json_report_with_a_config_file_fingerprints_that_config(tmp_path: Path):
    config_path = tmp_path / "rodit.yaml"
    config_path.write_text("routing:\n  assist_threshold: 0.75\n", encoding="utf-8")
    target = tmp_path / "e.json"
    argv = ["--split", "test", "--limit", "3", "--config", str(config_path)]
    assert main([*argv, "--json", str(target)], out=io.StringIO()) == 0
    fp = json.loads(target.read_text(encoding="utf-8"))["fingerprint"]
    assert fp["config_fingerprint"] == config_fingerprint(other_config())


# ================================================================ 从报告到结论


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
