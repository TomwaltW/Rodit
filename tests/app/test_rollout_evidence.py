"""上线门槛的依据（方案 §13.4 "测试集和影子对比都达到 13.2 的门槛""平台违规为 0"
"没有因此产生的投诉"、§13.5 上线前回归）：测试集取自配置指纹和线上一致的评测结论，
平台违规 / 投诉取自复核人员的按周登记，另加实际发出去的文字的合规扫描。
演示单元 + 合成数据（演示时钟在 2026-09-28 周一 10:35，W40）。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.app.rollout import (
    RolloutGateNotPassed,
    compliance_weeks,
    eval_evidence,
    rollout_evidence,
)
from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import Intent, Platform, ShopStatus, SuggestionStatus
from rodit.core.errors import ShopInactive
from rodit.core.models import SuggestionRecord
from rodit.eval.fingerprint import current_fingerprint, verdict_event, verdict_from_report
from rodit.feedback import RolloutAdvice
from rodit.feedback.rollout import IncidentKind
from rodit.testing.fixtures import SHOP_A, SHOP_B

S = SuggestionStatus
REVIEWER = "qa_reviewer_01"
W39 = "2026-W39"
W39_START = datetime(2026, 9, 21, tzinfo=CST)
FULL = {"factual": 2, "relevance": 2, "compliance": 2, "expression": 2}


LLM_ENV = {
    "RODIT_LLM_BASE_URL": "https://llm.example.invalid/v1",
    "RODIT_LLM_MODEL": "synthetic-model",
}


@pytest.fixture(autouse=True)
def running_llm(monkeypatch):
    """线上的大模型（合成的地址和模型名，从环境变量读）：测试集依据要知道线上用的是哪个模型。"""
    for name, value in LLM_ENV.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def service() -> AssistService:
    return AssistService(build_demo_unit(Platform.TAOBAO))


def shadow_ready(service: AssistService, n: int = 25) -> None:
    """影子模式下 n 条推荐都抽中、都打满分：影子对比的门槛能过。"""
    repo = service.unit.repos.suggestions
    for i in range(n):
        repo.add(
            SuggestionRecord(
                shop_id=SHOP_A,
                conversation_id=f"conv_e{i % 5}",
                msg_id=f"e_{i}",
                question_text="什么时候发货",
                intent=Intent.SHIPPING_POLICY,
                text="亲，现货付款后 48 小时内发货哦~",
                status=S.SHADOW,
                created_at=W39_START + timedelta(hours=1, minutes=i),
            )
        )
    ro = service.rollout
    ro.sample(SHOP_A, reviewer=REVIEWER, period=W39)
    for q in service.unit.repos.quality_reviews.list(SHOP_A, period=W39):
        ro.score(SHOP_A, q.id, status="scored", scores=FULL, reviewer=REVIEWER)


def report(service: AssistService, *, passes: bool = True, mode: str = "env", **fp) -> dict:
    fingerprint = current_fingerprint(service.unit.config).model_dump(mode="json")
    fingerprint.update(fp)
    return {
        "split": "all",
        "n": 324,
        "gate": {
            "n": 66,
            "threshold": 0.95,
            "factual": 0.97 if passes else 0.8,
            "compliance": 1.0,
            "passes": passes,
        },
        "llm": {"mode": mode, "model": "synthetic-model", "decision_provider": "incloud_llm"},
        "fingerprint": fingerprint,
    }


def record_verdict(service: AssistService, data: dict, *, minutes_ago: int = 10) -> None:
    at = service.unit.clock.now() - timedelta(minutes=minutes_ago)
    service.unit.repos.audit.record(verdict_event(verdict_from_report(data), at=at))


# ================================================================ 测试集（§13.2、§13.5）


def test_the_running_config_fingerprint_decides_which_verdict_counts(service: AssistService):
    ev = eval_evidence(service.unit.repos.audit, service.unit.config)
    assert ev.passed is None and "还没有评测结论" in (ev.reason or "")
    # 演示单元改了判断层（关键词规则），默认配置跑的评测和它对不上
    record_verdict(service, report(service, config_fingerprint="d" * 64), minutes_ago=30)
    ev = eval_evidence(service.unit.repos.audit, service.unit.config)
    assert ev.passed is None and "不一致" in (ev.reason or "")
    record_verdict(service, report(service, mode="offline"), minutes_ago=20)
    ev = eval_evidence(service.unit.repos.audit, service.unit.config)
    assert ev.passed is None and "离线替身" in (ev.reason or "")
    record_verdict(service, report(service), minutes_ago=10)
    ev = eval_evidence(service.unit.repos.audit, service.unit.config)
    assert ev.passed is True and ev.reason is None


def test_a_model_swap_through_the_environment_needs_a_new_verdict(
    service: AssistService, monkeypatch
):
    """只改环境变量 RODIT_LLM_MODEL 换模型（配置没写 llm.model）：旧结论不再算测试集依据
    （§13.5 换大模型之前先跑回归）。"""
    ro = service.rollout
    ro.set_stage(SHOP_A, "shadow", operator=REVIEWER)
    shadow_ready(service)
    record_verdict(service, report(service))
    status = ro.status(SHOP_A)
    assert status.advice == RolloutAdvice.READY_FOR_ASSIST and status.unchecked == []
    monkeypatch.setenv("RODIT_LLM_MODEL", "synthetic-model-2")
    ev = eval_evidence(service.unit.repos.audit, service.unit.config)
    assert ev.passed is None and "大模型" in (ev.reason or "")
    assert "synthetic-model-2" in (ev.reason or "")
    status = ro.status(SHOP_A)
    [unchecked] = status.unchecked
    assert unchecked.name == "test_set" and "大模型" in (unchecked.reason or "")
    assert [c.name for c in status.shadow_gate.checks if c.name == "test_set"] == []


def test_shadow_switch_follows_the_test_set_verdict(service: AssistService):
    ro = service.rollout
    ro.set_stage(SHOP_A, "shadow", operator=REVIEWER)
    shadow_ready(service)
    record_verdict(service, report(service, passes=False))
    status = ro.status(SHOP_A)
    assert status.advice == RolloutAdvice.STAY_SHADOW and status.blocking == ["test_set"]
    assert status.unchecked == []
    with pytest.raises(RolloutGateNotPassed) as exc:
        ro.set_stage(SHOP_A, "assist", operator=REVIEWER)
    assert [c.name for c in exc.value.gate.checks if not c.passed] == ["test_set"]
    assert exc.value.gate == status.shadow_gate  # 切换和状态同一口径

    record_verdict(service, report(service, passes=True), minutes_ago=1)
    status = ro.status(SHOP_A)
    assert status.advice == RolloutAdvice.READY_FOR_ASSIST and status.shadow_gate.passed
    change = ro.set_stage(SHOP_A, "assist", operator=REVIEWER)
    assert change.changed and not change.forced and change.gate == status.shadow_gate
    event = service.unit.repos.audit.list_events(SHOP_A, action=AuditAction.ROLLOUT_STAGE.value)[-1]
    assert event.detail["gate"]["unchecked"] == [] and event.detail["gate"]["passed"] is True


# ================================================================ 平台违规、投诉登记（§13.4）


def test_record_incident_validates_and_audits(service: AssistService):
    ro = service.rollout
    rec = ro.record_incident(SHOP_A, kind="violation", week=W39, count=0, reviewer=REVIEWER)
    assert rec.current and rec.reviewer == REVIEWER and rec.note is None
    [event] = service.unit.repos.audit.list_events(
        SHOP_A, action=AuditAction.ROLLOUT_INCIDENT.value
    )
    assert event.detail == {"kind": "violation", "week": W39, "count": 0, "note": None}
    for bad in (
        {"week": "2026-W41"},  # 还没开始
        {"count": True},
        {"count": -1},
        {"reviewer": " "},
        {"kind": "refund"},
        {"week": "2026-40"},
    ):
        kwargs = {"kind": "violation", "week": W39, "count": 0, "reviewer": REVIEWER, **bad}
        with pytest.raises(ValueError):
            ro.record_incident(SHOP_A, **kwargs)
    repo = service.unit.repos.knowledge
    repo.upsert_shop(repo.get_shop(SHOP_A).model_copy(update={"status": ShopStatus.SUSPENDED}))
    with pytest.raises(ShopInactive):
        ro.record_incident(SHOP_A, kind="complaint", week=W39, count=1, reviewer=REVIEWER)
    assert len(service.unit.repos.audit.list_events(SHOP_A, action="shop.rollout_incident")) == 1


def test_incidents_of_another_shop_do_not_count(service: AssistService):
    ro = service.rollout
    for week in ("2026-W38", W39):
        ro.record_incident(SHOP_B, kind="violation", week=week, count=3, reviewer="qa_reviewer_02")
    status = ro.status(SHOP_A)
    assert [u.name for u in status.unchecked] == ["platform_violations"]
    assert status.evidence is not None and status.evidence.platform_violations.count is None
    assert ro.incidents(SHOP_A).items == []
    assert len(ro.incidents(SHOP_B).items) == 2


def test_evidence_windows(service: AssistService):
    ro = service.rollout
    ro.record_incident(SHOP_A, kind="complaint", week="2026-W35", count=5, reviewer=REVIEWER)
    for week in ("2026-W36", "2026-W37", "2026-W38", W39):
        ro.record_incident(SHOP_A, kind="complaint", week=week, count=0, reviewer=REVIEWER)
    shop = service.unit.repos.knowledge.get_shop(SHOP_A)
    ev = rollout_evidence(service.unit.repos, service.unit.config, shop, service.unit.clock.now())
    assert ev.complaints.kind == IncidentKind.COMPLAINT
    assert ev.complaints.weeks == ["2026-W36", "2026-W37", "2026-W38", W39]
    assert ev.complaints.count == 0  # W35 在 4 周窗口外
    assert ev.platform_violations.weeks == ["2026-W38", W39]
    assert ev.compliance.since == datetime(2026, 9, 14, tzinfo=CST)
    assert ev.compliance.until == service.unit.clock.now()


# ================================================================ 合规扫描（代理指标）


def test_an_edited_send_is_counted_once_even_with_its_write_back(service: AssistService):
    """客服编辑后发送：推荐的 edited_text 和写回会话的那条（agent:{推荐 id}）只算一次。"""
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="scan1"))
    assert result is not None and result.suggestions
    sid = result.suggestions[0].suggestion_id
    service.feedback_edited(SHOP_A, sid, agent_id="kf_a01", edited_text="亲，加我微信 abc123 发您")
    service.unit.clock.advance(minutes=5)
    now = service.unit.clock.now()
    shop = service.unit.repos.knowledge.get_shop(SHOP_A)
    weeks = compliance_weeks(
        service.unit.repos,
        service.unit.config,
        shop,
        since=now - timedelta(days=1),
        until=now,
    )
    flagged = [w for w in weeks if w.flagged]
    assert len(flagged) == 1 and flagged[0].flagged == 1 and flagged[0].duplicates == 1
    [flag] = flagged[0].flags
    assert flag.source == "edited_suggestion" and flag.suggestion_id == sid
    assert flag.rules == ["off_platform"]
    status = service.rollout.status(SHOP_A)
    assert status.evidence is not None and status.evidence.compliance.flagged == 1
    # 没确认之前只是候选（unchecked），复核人员确认是违规后才算没通过
    assert "compliance_scan" not in status.blocking
    assert "compliance_scan" in [u.name for u in status.unchecked]
    service.rollout.review_compliance(
        SHOP_A, suggestion_id=sid, decision="confirmed", reviewer=REVIEWER
    )
    status = service.rollout.status(SHOP_A)
    assert status.evidence.compliance.confirmed == 1 and "compliance_scan" in status.blocking
    assert "abc123" not in status.model_dump_json() and "微信" not in status.model_dump_json()
