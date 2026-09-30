"""上线节奏的离线任务（rodit.ops.jobs.rollout）：record-eval 把评测结论记成 eval.verdict
（§13.5），acceptance-report 出 Phase 1 验收报告（§15，退出码 0 / 1），compliance-scan 写
compliance.flag（§13.4）。内存仓储 + 合成数据；@pytest.mark.pg 的用例在真实 PostgreSQL 上跑
（连不上时跳过）。经 serve.main 跑的任务输出到标准输出。"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rodit.app.demo import build_demo_unit
from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.clock import CST
from rodit.core.config import EngineConfig, config_fingerprint
from rodit.core.enums import (
    DraftKind,
    Intent,
    Platform,
    QualityReviewStatus,
    Role,
    SuggestionStatus,
)
from rodit.core.models import AuditEvent, QualityReview, QualityScores, SuggestionRecord
from rodit.eval.fingerprint import current_fingerprint, verdict_event, verdict_from_report
from rodit.ops import serve
from rodit.ops.jobs import rollout as rollout_jobs
from rodit.testing.fakes import FakeClock, make_message
from rodit.testing.fixtures import SHOP_A, SHOP_B

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=CST)  # W40 周三
W38 = datetime(2026, 9, 14, tzinfo=CST)
W39 = W38 + timedelta(days=7)
WEEKS = ("2026-W36", "2026-W37", "2026-W38", "2026-W39")
REVIEWER = "qa_reviewer_01"
FULL = QualityScores(factual=2, relevance=2, compliance=2, expression=2)
# 线上的大模型（合成的地址和模型名）：任务和线上服务读同样的环境变量
LLM_ENV = {
    "RODIT_LLM_BASE_URL": "https://llm.example.invalid/v1",
    "RODIT_LLM_MODEL": "synthetic-model",
}
_ids = iter(range(1, 100_000))


def parse(*argv: str):
    return serve._parser().parse_args(list(argv))  # noqa: SLF001 — 用部署入口同一个命令行


def eval_report(*, mode: str = "env", passes: bool = True, **over) -> dict:
    data = {
        "split": "all",
        "n": 324,
        "intent": None,
        "limit": None,
        "gate": {
            "split": "test",
            "n": 66,
            "threshold": 0.95,
            "factual": 0.97 if passes else 0.8,
            "compliance": 1.0,
            "passes": passes,
        },
        "llm": {"mode": mode, "model": "synthetic-model", "decision_provider": "incloud_llm"},
        "fingerprint": current_fingerprint(EngineConfig(), environ=LLM_ENV).model_dump(mode="json"),
        "cases": [],
    }
    data.update(over)
    return data


# ================================================================ record-eval


def test_record_eval_writes_a_system_event(tmp_path: Path):
    path = tmp_path / "e.json"
    path.write_text(json.dumps(eval_report(), ensure_ascii=False), encoding="utf-8")
    audit, out = InMemoryAuditSink(), io.StringIO()
    code = rollout_jobs.run(
        parse("record-eval", "--report", str(path)),
        LLM_ENV,
        audit=audit,
        clock=FakeClock(NOW),
        out=out,
    )
    assert code == 0
    [event] = audit.list_system_events(action=AuditAction.EVAL_VERDICT.value)
    assert event.shop_id is None and event.created_at == NOW and event.actor == "system"
    assert event.detail["fingerprint"]["config_fingerprint"] == config_fingerprint(EngineConfig())
    assert event.detail["gate_passes"] is True and event.detail["llm_mode"] == "env"
    assert "一致" in out.getvalue() and "达标" in out.getvalue()
    assert "不一致" not in out.getvalue()


def test_record_eval_warns_when_the_deployed_config_differs(tmp_path: Path):
    path = tmp_path / "e.json"
    path.write_text(json.dumps(eval_report()), encoding="utf-8")
    config = tmp_path / "rodit.yaml"
    config.write_text("routing:\n  assist_threshold: 0.75\n", encoding="utf-8")
    audit, out = InMemoryAuditSink(), io.StringIO()
    args = parse("record-eval", "--report", str(path))
    environ = {**LLM_ENV, "RODIT_CONFIG": str(config)}
    assert rollout_jobs.run(args, environ, audit=audit, out=out) == 0
    assert "不一致" in out.getvalue() and str(config) in out.getvalue()
    assert "配置指纹" in out.getvalue()
    assert len(audit.list_system_events()) == 1  # 照样记下（以后换回这套配置时还能用）


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (json.dumps(eval_report(limit=10)), "部分评测"),
        (json.dumps(eval_report(fingerprint=None)), "配置指纹"),
        (json.dumps(eval_report(split="dev", gate=None)), "测试集"),
        ("not json", "不是 JSON"),
        (None, "读不了"),
    ],
)
def test_record_eval_refuses_reports_that_cannot_back_a_launch(
    tmp_path: Path, caplog, content: str | None, message: str
):
    caplog.set_level(logging.INFO)
    path = tmp_path / "e.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    audit = InMemoryAuditSink()
    args = parse("record-eval", "--report", str(path))
    assert rollout_jobs.run(args, {}, audit=audit, out=io.StringIO()) == 2
    assert audit.list_system_events() == []
    assert message in caplog.text


def test_record_eval_through_the_ops_entry_point(tmp_path: Path, monkeypatch, ops_env):
    audit = InMemoryAuditSink()
    monkeypatch.setattr(rollout_jobs, "_pg_audit", lambda: audit)
    path = tmp_path / "e.json"
    path.write_text(json.dumps(eval_report()), encoding="utf-8")
    assert serve.main(["record-eval", "--report", str(path)], out=io.StringIO()) == 0
    assert len(audit.list_system_events(action="eval.verdict")) == 1


def test_job_commands_are_registered_and_other_commands_are_not_claimed():
    for argv in (["migrate"], ["serve"], ["issue-token", "--metrics"]):
        assert rollout_jobs.run(parse(*argv), {}) is None
    help_text = serve._parser().format_help()  # noqa: SLF001
    for command in ("record-eval", "acceptance-report", "compliance-scan"):
        assert command in help_text


# ================================================================ acceptance-report（§15）


def seed_shop(repos, shop_id: str, *, adopted: int, groups: int = 20) -> None:
    """一家店的验收数据：W38、W39 各 groups 组推荐（adopted 组原样发送）、推荐出现耗时、
    W36–W39 的平台违规登记（都是 0）、W39 的 25 条满分复核。全部合成。"""
    for start in (W38, W39):
        for i in range(groups):
            repos.suggestions.add(
                SuggestionRecord(
                    shop_id=shop_id,
                    conversation_id=f"conv_acc{i % 4}",
                    msg_id=f"acc_{shop_id}_{start:%m%d}_{i}",
                    intent=Intent.SHIPPING_POLICY,
                    text="亲，现货付款后 48 小时内发货哦~",
                    kind=DraftKind.LLM,
                    status=SuggestionStatus.SENT if i < adopted else SuggestionStatus.REJECTED,
                    agent_id="kf_a01",
                    latency_ms=1200,
                    created_at=start + timedelta(hours=2, minutes=i),
                )
            )
            repos.audit.record(
                AuditEvent(
                    shop_id=shop_id,
                    actor="engine",
                    action=AuditAction.RESULT.value,
                    detail={"suggestions": 1, "latency_ms": {"total": 1200}},
                    created_at=start + timedelta(hours=2, minutes=i),
                )
            )
    for week in WEEKS:
        repos.audit.record(
            AuditEvent(
                shop_id=shop_id,
                actor=REVIEWER,
                action=AuditAction.ROLLOUT_INCIDENT.value,
                detail={"kind": "violation", "week": week, "count": 0, "note": None},
                created_at=NOW - timedelta(days=1),
            )
        )
    sampled = repos.quality_reviews.add_many(
        [
            QualityReview(
                shop_id=shop_id,
                period="2026-W39",
                suggestion_id=next(_ids),
                intent=Intent.SHIPPING_POLICY,
                sampled_at=NOW - timedelta(days=2),
            )
            for _ in range(25)
        ]
    )
    for q in sampled:
        repos.quality_reviews.record(
            shop_id,
            q.id,
            status=QualityReviewStatus.SCORED,
            scores=FULL,
            reviewer=REVIEWER,
            note=None,
            reviewed_at=NOW - timedelta(days=1),
        )


@pytest.fixture
def repos():
    """演示单元的内存仓储（淘宝：店铺 A、B 各有一份合成授权）+ 一条和默认配置一致的评测结论。"""
    repos = build_demo_unit(Platform.TAOBAO).repos
    verdict = verdict_from_report(eval_report())
    repos.audit.record(verdict_event(verdict, at=NOW - timedelta(days=3)))
    return repos


def acceptance(repos, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    args = parse("acceptance-report", "--since", "2026-09-01", *argv)
    code = rollout_jobs.run(args, LLM_ENV, repos=repos, clock=FakeClock(NOW), out=out)
    return code, out.getvalue()


def test_acceptance_report_passes_when_every_criterion_is_met(repos):
    seed_shop(repos, SHOP_A, adopted=15)
    seed_shop(repos, SHOP_B, adopted=13)
    code, text = acceptance(repos, "--json")
    assert code == 0, text
    report = json.loads(text)
    assert report["passed"] is True and report["failing"] == [] and report["unchecked"] == []
    assert [s["shop_id"] for s in report["shops"]] == [SHOP_A, SHOP_B]
    assert {c["name"]: c["status"] for c in report["pooled"]} == {
        "test_set": "passed",
        "adoption": "passed",
        "latency_p95": "passed",
        "platform_violations": "passed",
        "compliance_scan": "passed",
        "weekly_review": "passed",
    }
    assert report["adoption_weeks"] == ["2026-W38", "2026-W39"]
    assert report["running_fingerprint"] == config_fingerprint(EngineConfig())
    assert "48 小时" not in text  # 没有推荐文字
    code, text = acceptance(repos)
    assert code == 0 and "结论：通过" in text


def test_acceptance_report_fails_and_names_the_criterion(repos):
    seed_shop(repos, SHOP_A, adopted=11)  # 11/20 = 55%
    seed_shop(repos, SHOP_B, adopted=11)
    code, text = acceptance(repos)
    assert code == 1
    last = text.strip().splitlines()[-1]
    assert "没通过" in last and "adoption=0.55" in last
    code, text = acceptance(repos, "--json")
    report = json.loads(text)
    assert code == 1 and report["failing"] == ["adoption"] and not report["passed"]
    adoption = next(c for c in report["pooled"] if c["name"] == "adoption")
    assert adoption["value"] == 0.55 and adoption["threshold"] == 0.6


def test_acceptance_report_counts_missing_data_as_not_passed(repos):
    seed_shop(repos, SHOP_A, adopted=15)
    code, text = acceptance(repos, "--json")
    report = json.loads(text)
    assert code == 1  # 店铺 B 没有数据：合在一起缺平台违规登记
    assert "platform_violations" in report["unchecked"]
    violations = next(c for c in report["pooled"] if c["name"] == "platform_violations")
    assert SHOP_B in violations["detail"]


def test_acceptance_report_rejects_a_bad_window(repos, caplog):
    caplog.set_level(logging.INFO)
    code, _ = acceptance(repos, "--until", "2026-08-01")
    assert code == 2 and "--since" in caplog.text


# ================================================================ compliance-scan（§13.4）


def test_compliance_scan_records_a_flag_per_shop_without_text(repos):
    at = W39 + timedelta(days=1)
    repos.conversations.add_message(
        make_message("有优惠吗", at=at, conversation_id="conv_cs", msg_id="b_cs")
    )
    repos.conversations.add_outgoing(
        shop_id=SHOP_A,
        conversation_id="conv_cs",
        msg_id="out_cs",
        role=Role.AGENT,
        text="加我微信 abc123，给您最低价",
        at=at + timedelta(minutes=1),
    )
    out = io.StringIO()
    args = parse("compliance-scan")
    assert rollout_jobs.run(args, {}, repos=repos, clock=FakeClock(NOW), out=out) == 0
    [flag_a] = repos.audit.list_events(SHOP_A, action=AuditAction.COMPLIANCE_FLAG.value)
    [flag_b] = repos.audit.list_events(SHOP_B, action=AuditAction.COMPLIANCE_FLAG.value)
    assert flag_a.detail["period"] == "2026-W39" and flag_a.detail["flagged"] == 1
    assert flag_a.detail["flags"] == [
        {
            "suggestion_id": None,
            "msg_id": "out_cs",
            "role": "agent",
            "rules": ["off_platform", "absolute_term"],
        }
    ]
    assert flag_b.detail["flagged"] == 0 and flag_b.detail["flags"] == []
    for text in (json.dumps(flag_a.detail, ensure_ascii=False), out.getvalue()):
        for word in ("微信", "abc123", "最低价", "有优惠吗"):
            assert word not in text
    assert "疑似违规 1 条" in out.getvalue()
    bad = rollout_jobs.run(
        parse("compliance-scan", "--week", "2026-W41"), {}, repos=repos, clock=FakeClock(NOW)
    )
    assert bad == 2


def test_compliance_scan_removes_its_flags_when_the_shop_is_purged_meanwhile(repos):
    """扫描期间 serve 清理了这家店（先删令牌、后删审计）：迟到的 compliance.flag 不留下。"""
    original = repos.audit.record

    def record_then_purge(event):  # type: ignore[no-untyped-def]
        original(event)
        if event.shop_id == SHOP_A:  # 写入之后 serve 才删令牌，但审计已经删过了
            repos.auth.delete(SHOP_A)

    repos.audit.purge_shop(SHOP_A)
    repos.audit.record = record_then_purge
    out = io.StringIO()
    assert (
        rollout_jobs.run(parse("compliance-scan"), {}, repos=repos, clock=FakeClock(NOW), out=out)
        == 0
    )
    assert repos.audit.list_events(SHOP_A, action=AuditAction.COMPLIANCE_FLAG.value) == []
    assert repos.audit.list_events(SHOP_B, action=AuditAction.COMPLIANCE_FLAG.value) != []
    assert f"{SHOP_A} 扫描期间被取消订购" in out.getvalue()


# ================================================================ PostgreSQL


@pytest.mark.pg
def test_record_eval_on_postgres(ops_env, ops_pg_dsn, monkeypatch, tmp_path: Path, capsys):
    """python -m rodit.eval --json 的真实输出 → record-eval → list_system_events 读得到。"""
    from rodit.eval.cli import main as eval_main
    from rodit.storage import Database, PgAuditLog

    Database(ops_pg_dsn).migrate()
    monkeypatch.setenv("RODIT_DATABASE_DSN", ops_pg_dsn)
    report = tmp_path / "e.json"
    assert eval_main(["--split", "test", "--json", str(report)], out=io.StringIO()) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["fingerprint"]
    capsys.readouterr()
    assert serve.main(["record-eval", "--report", str(report)]) == 0
    printed = capsys.readouterr().out
    [event] = PgAuditLog(Database(ops_pg_dsn)).list_system_events(
        action=AuditAction.EVAL_VERDICT.value
    )
    assert event.shop_id is None
    assert event.detail["fingerprint"]["config_fingerprint"] == config_fingerprint(EngineConfig())
    assert event.detail["llm_mode"] == "offline" and event.detail["split"] == "test"
    assert "离线替身" in printed


@pytest.mark.pg
def test_acceptance_report_on_postgres(ops_env, ops_pg_dsn, monkeypatch, capsys):
    """PostgreSQL 上列店铺（rodit_auth_expiring）、读各店数据、读系统事件：没有数据时退出码 1。"""
    from pydantic import SecretStr

    from rodit.app.production import postgres_repos
    from rodit.core.models import ShopAuth
    from rodit.core.scopes import MVP_SCOPES
    from rodit.storage import Database
    from rodit.testing.fixtures import build_demo_world

    db = Database(ops_pg_dsn)
    db.migrate()
    monkeypatch.setenv("RODIT_DATABASE_DSN", ops_pg_dsn)
    repos = postgres_repos(db, clock=FakeClock(NOW))
    world = build_demo_world()
    for shop_id in (SHOP_A, SHOP_B):
        repos.knowledge.upsert_shop(world.shop(shop_id))
        repos.auth.save(
            ShopAuth(
                shop_id=shop_id,
                access_token=SecretStr(f"synthetic-token-{shop_id}"),
                scopes=sorted(MVP_SCOPES),
                expires_at=NOW + timedelta(days=30),
            )
        )
    repos.audit.record(
        AuditEvent(
            shop_id=SHOP_A,
            actor=REVIEWER,
            action=AuditAction.ROLLOUT_INCIDENT.value,
            detail={"kind": "violation", "week": "2026-W39", "count": 1, "note": None},
            created_at=NOW - timedelta(days=1),
        )
    )
    argv = ["acceptance-report", "--since", "2026-09-01", "--until", "2026-09-30T12:00+08:00"]
    capsys.readouterr()
    assert serve.main([*argv, "--json"]) == 1
    printed = capsys.readouterr().out
    report = json.loads(printed)
    assert [s["shop_id"] for s in report["shops"]] == [SHOP_A, SHOP_B]
    assert "platform_violations" in report["failing"]  # A 登记了 1 条
    assert "synthetic-token" not in printed
