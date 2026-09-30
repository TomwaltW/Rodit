"""合规扫描命中只是候选，由复核人员逐条确认（方案 §13.4 "平台违规为 0" 的代理指标、§15 验收）：
没确认的列为 unchecked（写明几条、哪些 ID），确认是违规的才算不通过，误报标记后不计。

典型误报：质检硬规则带着依据原文放行的成分标注（"面料：桑蚕丝 100%"），客服原样发出，
扫描没有依据原文，按绝对化用语命中。演示单元 + 合成数据（演示时钟在 2026-09-28 周一，W40）。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.app.rollout import acceptance_input, compliance_weeks
from rodit.core.audit import AuditAction
from rodit.core.enums import Platform, Role
from rodit.core.errors import ShopInactive
from rodit.core.models import AuditEvent
from rodit.feedback.acceptance import CriterionStatus, scan_criterion
from rodit.feedback.compliance import (
    COMPLIANCE_REVIEWED,
    ComplianceDecision,
    parse_reviews,
    summarize_scan,
)
from rodit.testing.fakes import make_message
from rodit.testing.fixtures import SHOP_A, SHOP_B

REVIEWER = "qa_reviewer_01"


@pytest.fixture
def service() -> AssistService:
    return AssistService(build_demo_unit(Platform.TAOBAO))


def send_silk_card(service: AssistService) -> int:
    """质检通过的成分标注卡片，客服原样发送（写回会话 agent:{推荐 id}）。返回推荐 ID。"""
    result = service.handle_event(
        "B", taobao_event("真丝的会不会皱", msg_id="fp1", item_id="i_888")
    )
    assert result is not None and result.suggestions
    card = result.suggestions[0]
    assert "100%" in card.text
    service.feedback_sent(SHOP_A, card.suggestion_id, agent_id="kf_a01")
    service.unit.clock.advance(minutes=5)
    return card.suggestion_id


def scan(service: AssistService, shop_id: str = SHOP_A):
    now = service.unit.clock.now()
    shop = service.unit.repos.knowledge.get_shop(shop_id)
    since = now - timedelta(days=1)
    return compliance_weeks(service.unit.repos, service.unit.config, shop, since=since, until=now)


def test_an_unreviewed_hit_is_unchecked_not_a_failure(service: AssistService):
    sid = send_silk_card(service)
    weeks = scan(service)
    [flag] = [f for w in weeks for f in w.flags]
    assert flag.rules == ["absolute_term"] and flag.msg_id == f"agent:{sid}"
    status = service.rollout.status(SHOP_A)
    assert "compliance_scan" not in status.blocking
    [unchecked] = [u for u in status.unchecked if u.name == "compliance_scan"]
    assert f"agent:{sid}" in (unchecked.reason or "") and "dismissed" in (unchecked.reason or "")
    summary = status.evidence.compliance
    assert (summary.flagged, summary.pending, summary.confirmed, summary.dismissed) == (1, 1, 0, 0)
    assert summary.pending_message_ids == [f"agent:{sid}"]
    criterion = scan_criterion(summarize_scan(weeks))
    assert criterion.status == CriterionStatus.UNCHECKED and f"agent:{sid}" in criterion.detail


def test_a_reviewer_dismisses_a_false_positive(service: AssistService):
    sid = send_silk_card(service)
    ro = service.rollout
    record = ro.review_compliance(
        SHOP_A, msg_id=f"agent:{sid}", decision="dismissed", reviewer=REVIEWER, note="依据里原样有"
    )
    assert record.decision == ComplianceDecision.DISMISSED and record.rules == ["absolute_term"]
    assert record.reviewer == REVIEWER and record.suggestion_id is None
    status = ro.status(SHOP_A)
    summary = status.evidence.compliance
    assert (summary.flagged, summary.pending, summary.dismissed) == (1, 0, 1)
    assert "compliance_scan" not in status.blocking
    assert "compliance_scan" not in [u.name for u in status.unchecked]
    assert "compliance_scan" in [c.name for c in status.assist_gate.checks]
    shop = service.unit.repos.knowledge.get_shop(SHOP_A)
    now = service.unit.clock.now()
    data = acceptance_input(
        service.unit.repos, service.unit.config, shop, since=now - timedelta(days=1), until=now
    )
    assert scan_criterion(data.compliance).status == CriterionStatus.PASSED
    [event] = service.unit.repos.audit.list_events(SHOP_A, action=COMPLIANCE_REVIEWED)
    assert event.actor == REVIEWER
    assert event.detail == {
        "suggestion_id": None,
        "msg_id": f"agent:{sid}",
        "decision": "dismissed",
        "rules": ["absolute_term"],
        "note": "依据里原样有",
    }
    assert "桑蚕丝" not in str(event.detail)  # 不记文字


def test_a_confirmed_violation_fails_and_the_latest_decision_wins(service: AssistService):
    ro = service.rollout
    at = service.unit.clock.now()
    conv = make_message("有纯棉的吗", at=at, conversation_id="conv_typed", msg_id="b_typed")
    service.unit.repos.conversations.add_message(conv)
    service.unit.repos.conversations.add_outgoing(
        shop_id=SHOP_A,
        conversation_id="conv_typed",
        msg_id="out_typed_1",
        role=Role.AGENT,
        text="亲，这款 100% 纯棉，全网最好",  # 客服自己打的字
        at=at,
    )
    service.unit.clock.advance(minutes=1)
    ro.review_compliance(SHOP_A, msg_id="out_typed_1", decision="dismissed", reviewer=REVIEWER)
    assert "compliance_scan" not in ro.status(SHOP_A).blocking
    service.unit.clock.advance(minutes=1)
    ro.review_compliance(SHOP_A, msg_id="out_typed_1", decision="confirmed", reviewer="qa_2")
    status = ro.status(SHOP_A)
    assert status.evidence.compliance.confirmed == 1 and "compliance_scan" in status.blocking
    check = {c.name: c for c in status.assist_gate.checks}["compliance_scan"]
    assert check.passed is False and check.value == 1
    assert scan_criterion(status.evidence.compliance).status == CriterionStatus.FAILED


def test_an_edited_send_is_reviewed_by_suggestion_id(service: AssistService):
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="ed1"))
    sid = result.suggestions[0].suggestion_id
    service.feedback_edited(SHOP_A, sid, agent_id="kf_a01", edited_text="亲，加我微信 abc123 发您")
    service.unit.clock.advance(minutes=5)
    ro = service.rollout
    assert ro.status(SHOP_A).evidence.compliance.pending_suggestion_ids == [sid]
    record = ro.review_compliance(
        SHOP_A, suggestion_id=sid, decision="confirmed", reviewer=REVIEWER
    )
    assert record.rules == ["off_platform"]
    assert "compliance_scan" in ro.status(SHOP_A).blocking


def test_only_recent_flags_of_this_shop_can_be_reviewed(service: AssistService):
    ro = service.rollout
    sid = send_silk_card(service)
    with pytest.raises(KeyError):  # 不是疑似违规
        ro.review_compliance(SHOP_A, msg_id="no_such_msg", decision="dismissed", reviewer=REVIEWER)
    with pytest.raises(KeyError):  # 别的店铺没有这条
        ro.review_compliance(SHOP_B, msg_id=f"agent:{sid}", decision="dismissed", reviewer=REVIEWER)
    for bad in (
        {"msg_id": f"agent:{sid}", "suggestion_id": sid},
        {},
        {"suggestion_id": 0},
        {"suggestion_id": True},
        {"msg_id": " "},
        {"msg_id": f"agent:{sid}", "reviewer": " "},
        {"msg_id": f"agent:{sid}", "decision": "maybe"},
    ):
        kwargs = {"decision": "dismissed", "reviewer": REVIEWER, **bad}
        with pytest.raises(ValueError):
            ro.review_compliance(SHOP_A, **kwargs)
    assert service.unit.repos.audit.list_events(SHOP_A, action=COMPLIANCE_REVIEWED) == []
    repo = service.unit.repos.knowledge
    repo.upsert_shop(repo.get_shop(SHOP_A).model_copy(update={"status": "suspended"}))
    with pytest.raises(ShopInactive):
        ro.review_compliance(SHOP_A, msg_id=f"agent:{sid}", decision="dismissed", reviewer=REVIEWER)


def test_reviews_of_another_shop_or_malformed_events_are_ignored():
    at = datetime(2026, 9, 28, 12, 0).astimezone()
    good = AuditEvent(
        shop_id=SHOP_A,
        actor=REVIEWER,
        action=COMPLIANCE_REVIEWED,
        detail={"suggestion_id": None, "msg_id": "m1", "decision": "dismissed", "note": None},
        created_at=at,
    )
    events = [
        good,
        good.model_copy(update={"shop_id": SHOP_B}),
        good.model_copy(update={"action": AuditAction.ROLLOUT_INCIDENT.value}),
        good.model_copy(update={"detail": {"msg_id": "m1", "decision": "maybe"}}),
        good.model_copy(
            update={"detail": {"suggestion_id": 3, "msg_id": "m1", "decision": "dismissed"}}
        ),
        good.model_copy(update={"detail": {"decision": "confirmed"}}),
    ]
    [entry] = parse_reviews(events, shop_id=SHOP_A)
    assert entry.msg_id == "m1" and entry.decision == ComplianceDecision.DISMISSED
