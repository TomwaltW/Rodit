"""评测结论的指纹与上线门槛的测试集依据（方案 §13.5 "换大模型、改提示词、升级判断层之前，
先在测试集上跑回归"、§13.2 测试集门槛、§13.4 影子 → 智能辅助）。

- 指纹（EvalFingerprint）：评测用的配置的 config_fingerprint（core.config）、路由 / 质检
  问题定义的版本（core.questions，改文字就变）、代码版本（git，拿得到时；部署镜像里没有 .git
  时由环境变量 RODIT_GIT_SHA 提供）、路由分类器模型文件的 SHA-256（--classifier 时）。
  python -m rodit.eval --json 的报告带上它。
- 评测结论（EvalVerdict）：python -m rodit.ops record-eval --report 报告.json 从报告里取出
  指纹、测试集判定（§13.2 事实正确、合规满分率 ≥ 95%）、回归比较结果、用的大模型，记成系统事件
  eval.verdict（shop_id 为空，不属于任何店铺）。只收完整的评测（不收 --intent / --limit 的
  部分评测，要有测试集）。
- 测试集依据（test_set_evidence）：上线状态、验收报告从 eval.verdict 里挑配置指纹、问题定义
  版本都和线上一致、用真实大模型（--llm env）跑的最近一次结论：测试集达标且回归没有下降算通过。
  离线替身的结论只作回归基线（rodit.eval.cli 的说明），不算上线依据；代码版本、分类器只作记录，
  不参与比较（线上不知道自己的代码版本）。没有能用的结论时写明原因（没有结论、指纹对不上、
  只有离线的）。

结论里只有指纹、计数、比例和模型名，没有买家数据、没有密钥（配置里只有环境变量名）。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.config import EngineConfig, config_fingerprint
from rodit.core.models import AuditEvent
from rodit.core.questions import QA_QUESTIONS_VER, ROUTER_QUESTIONS_VER
from rodit.feedback.rollout import TestSetEvidence

GIT_SHA_ENV = "RODIT_GIT_SHA"
"""部署镜像里没有 .git 时由构建流程写进来的代码版本。"""
GIT_TIMEOUT_S = 5
SHORT = 12  # 报告、原因里显示的指纹长度

HOW_TO_RECORD = (
    "用线上配置跑 python -m rodit.eval --llm env --config <部署配置> --json 报告.json，"
    "再 python -m rodit.ops record-eval --report 报告.json"
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalFingerprint(_Model):
    """一次评测对应的配置和代码。same_setup 只比较前三项。"""

    config_fingerprint: str
    router_questions_ver: str
    qa_questions_ver: str
    git_sha: str | None = None
    git_dirty: bool | None = None  # 有没有提交的改动（环境变量给的版本为 None）
    classifier_sha256: str | None = None  # --classifier 模型文件


def git_revision(
    cwd: str | Path | None = None, *, environ: Mapping[str, str] | None = None
) -> tuple[str | None, bool | None]:
    """(代码版本, 有没有未提交的改动)。先看 RODIT_GIT_SHA，再问 git；拿不到返回 (None, None)。"""
    env = os.environ if environ is None else environ
    fixed = env.get(GIT_SHA_ENV, "").strip()
    if fixed:
        return fixed, None
    where = str(cwd) if cwd is not None else str(Path(__file__).resolve().parent)
    try:
        head = subprocess.run(
            ["git", "-C", where, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
        if head.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "-C", where, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    sha = head.stdout.strip() or None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return sha, dirty


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_fingerprint(
    config: EngineConfig,
    *,
    git: bool = False,
    classifier: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> EvalFingerprint:
    """这套配置（和当前代码里的问题定义）的指纹。git=True 时带上代码版本；classifier 给了时
    带上模型文件的 SHA-256。线上比较用 git=False（same_setup 不比较代码版本）。"""
    sha, dirty = git_revision(environ=environ) if git else (None, None)
    return EvalFingerprint(
        config_fingerprint=config_fingerprint(config),
        router_questions_ver=ROUTER_QUESTIONS_VER,
        qa_questions_ver=QA_QUESTIONS_VER,
        git_sha=sha,
        git_dirty=dirty,
        classifier_sha256=file_sha256(classifier) if classifier else None,
    )


def same_setup(a: EvalFingerprint, b: EvalFingerprint) -> bool:
    """配置指纹、路由问题版本、质检问题版本都一样。"""
    return _differences(a, b) == []


def _differences(a: EvalFingerprint, b: EvalFingerprint) -> list[str]:
    out = []
    if a.config_fingerprint != b.config_fingerprint:
        out.append("配置指纹")
    if a.router_questions_ver != b.router_questions_ver:
        out.append("路由问题版本")
    if a.qa_questions_ver != b.qa_questions_ver:
        out.append("质检问题版本")
    return out


# ================================================================ 评测结论（eval.verdict）


class EvalVerdict(_Model):
    """一次完整评测的结论（eval.verdict 系统事件的 detail）。没有买家数据、没有密钥。"""

    fingerprint: EvalFingerprint
    llm_mode: Literal["offline", "env"]
    model: str | None = None
    decision_provider: str
    split: str
    n: int = Field(ge=0)
    test_n: int = Field(ge=0)
    threshold: float
    factual: float
    compliance: float
    gate_passes: bool
    baseline_passed: bool | None = None  # 没做回归比较为 None
    report_sha256: str


class _Digest(BaseModel):
    """评测报告（rodit.eval.cli.EvalCliReport 的 JSON）里 record-eval 用到的部分。"""

    model_config = ConfigDict(extra="ignore")


class _GateDigest(_Digest):
    n: int
    threshold: float
    factual: float
    compliance: float
    passes: bool


class _LLMDigest(_Digest):
    mode: Literal["offline", "env"]
    model: str | None = None
    decision_provider: str


class _BaselineDigest(_Digest):
    passed: bool


class _ReportDigest(_Digest):
    split: str
    n: int
    intent: str | None = None
    limit: int | None = None
    gate: _GateDigest | None = None
    llm: _LLMDigest
    baseline: _BaselineDigest | None = None
    fingerprint: EvalFingerprint | None = None


def verdict_from_report(data: Any, *, raw: bytes | None = None) -> EvalVerdict:
    """评测报告（--json 的内容，已解析）→ 结论。raw 是文件原文（算报告的 SHA-256；没给时按
    规范化的 JSON 算）。不能作为测试集依据的报告抛 ValueError（原因写明）。"""
    if not isinstance(data, Mapping):
        raise ValueError("不是评测报告（应为 python -m rodit.eval --json 的输出）")
    if not data.get("fingerprint"):
        raise ValueError("报告里没有配置指纹：用本版本的 python -m rodit.eval --json 重新生成")
    try:
        digest = _ReportDigest.model_validate(dict(data))
    except ValidationError as e:
        fields = "、".join(sorted({".".join(str(p) for p in err["loc"]) for err in e.errors()}))
        raise ValueError(f"不是完整的评测报告（字段不对：{fields}）") from None
    if digest.intent is not None or digest.limit is not None:
        raise ValueError("部分评测（--intent / --limit）不能作为上线结论：请跑完整的评测集")
    if digest.gate is None or digest.split == "dev":
        raise ValueError("报告里没有测试集判定：要用 --split test 或 all（§13.1 留出的测试集）")
    assert digest.fingerprint is not None
    if raw is None:
        raw = json_bytes(data)
    return EvalVerdict(
        fingerprint=digest.fingerprint,
        llm_mode=digest.llm.mode,
        model=digest.llm.model,
        decision_provider=digest.llm.decision_provider,
        split=digest.split,
        n=digest.n,
        test_n=digest.gate.n,
        threshold=digest.gate.threshold,
        factual=digest.gate.factual,
        compliance=digest.gate.compliance,
        gate_passes=digest.gate.passes,
        baseline_passed=digest.baseline.passed if digest.baseline is not None else None,
        report_sha256=hashlib.sha256(raw).hexdigest(),
    )


def json_bytes(data: Any) -> bytes:
    """规范化的 JSON（键排序、紧凑、UTF-8）。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def verdict_event(verdict: EvalVerdict, *, at: datetime, actor: str = "system") -> AuditEvent:
    """结论 → 系统事件（shop_id 为空）。"""
    return AuditEvent(
        shop_id=None,
        actor=actor,
        action=AuditAction.EVAL_VERDICT.value,
        detail=verdict.model_dump(mode="json"),
        created_at=at,
    )


def parse_verdicts(events: Iterable[AuditEvent]) -> list[tuple[datetime, EvalVerdict]]:
    """系统事件 → (记录时间, 结论)，按时间升序。店铺事件、字段不对的事件跳过。"""
    out: list[tuple[datetime, EvalVerdict]] = []
    for e in events:
        if e.shop_id is not None or e.action != AuditAction.EVAL_VERDICT.value:
            continue
        try:
            out.append((e.created_at, EvalVerdict.model_validate(e.detail)))
        except ValidationError:
            continue
    return sorted(out, key=lambda x: x[0])


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _when(at: datetime) -> str:
    return f"{at.astimezone(CST):%Y-%m-%d %H:%M}"


def describe_verdict(at: datetime, v: EvalVerdict) -> str:
    """结论的一句话说明（中文，没有买家数据）。"""
    model = v.model or "未知模型"
    text = (
        f"{_when(at)} 的评测（{model}，测试集 {v.test_n} 条）：事实正确 {_pct(v.factual)}、"
        f"合规 {_pct(v.compliance)}（门槛 ≥ {_pct(v.threshold)}）"
    )
    if v.baseline_passed is False:
        text += "；和基线比有指标下降（§13.5）"
    passed = v.gate_passes and v.baseline_passed is not False
    return text + ("，达标" if passed else "，没达标")


def test_set_evidence(events: Iterable[AuditEvent], running: EvalFingerprint) -> TestSetEvidence:
    """从 eval.verdict 系统事件里挑测试集依据（见模块说明）。events 一般是
    AuditLog.list_system_events(action="eval.verdict")。"""
    verdicts = parse_verdicts(events)
    base = {"running_fingerprint": running.config_fingerprint}
    if not verdicts:
        return TestSetEvidence(reason=f"还没有评测结论：{HOW_TO_RECORD}", **base)
    matching = [(at, v) for at, v in verdicts if same_setup(v.fingerprint, running)]
    if not matching:
        at, latest = verdicts[-1]
        diffs = "、".join(_differences(latest.fingerprint, running))
        return TestSetEvidence(
            reason=(
                f"最近的评测结论（{_when(at)}）和线上不一致（{diffs}；评测的配置指纹 "
                f"{latest.fingerprint.config_fingerprint[:SHORT]}，线上 "
                f"{running.config_fingerprint[:SHORT]}）：改过配置或问题定义之后要重新评测并记录"
            ),
            verdict_at=at,
            config_fingerprint=latest.fingerprint.config_fingerprint,
            **base,
        )
    real = [(at, v) for at, v in matching if v.llm_mode == "env"]
    if not real:
        at, latest = matching[-1]
        return TestSetEvidence(
            reason=(
                "和线上配置一致的评测结论都是离线替身跑的（只作回归基线，不是上线判断）："
                "用 --llm env 跑真实大模型之后再记录"
            ),
            verdict_at=at,
            config_fingerprint=latest.fingerprint.config_fingerprint,
            **base,
        )
    at, v = real[-1]
    return TestSetEvidence(
        passed=v.gate_passes and v.baseline_passed is not False,
        detail=describe_verdict(at, v),
        verdict_at=at,
        config_fingerprint=v.fingerprint.config_fingerprint,
        **base,
    )


test_set_evidence.__test__ = False  # type: ignore[attr-defined]  # 名字以 test_ 开头，不是 pytest 用例


__all__ = [
    "GIT_SHA_ENV",
    "HOW_TO_RECORD",
    "EvalFingerprint",
    "EvalVerdict",
    "current_fingerprint",
    "describe_verdict",
    "file_sha256",
    "git_revision",
    "parse_verdicts",
    "same_setup",
    "test_set_evidence",
    "verdict_event",
    "verdict_from_report",
]
