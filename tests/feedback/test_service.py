"""反馈闭环（方案 §5.8）：状态迁移、派生数据（标注集、知识缺失、候选知识）、商家审核、
非法迁移、串店、个人信息最小化、审计。"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from rodit.core import evidence as ev
from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.enums import (
    DraftKind,
    Intent,
    KnowledgeSource,
    KnowledgeStatus,
    RejectReason,
    SuggestionStatus,
)
from rodit.core.errors import ActionNotAllowed
from rodit.core.models import KnowledgeItem, SuggestionRecord
from rodit.core.pii import ADDRESS_MASK, PHONE_MASK
from rodit.feedback import (
    KNOWLEDGE_RETIRED,
    SKIP_NO_QUESTION,
    SKIP_REALTIME,
    FeedbackService,
)
from rodit.inmemory import (
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemorySuggestionRepository,
)
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import (
    AGENT,
    CONV,
    SHOP_A,
    SHOP_B,
    build_demo_world,
    seed_knowledge,
)

PHONE = "13812345678"
QUESTION = "165 110斤穿什么码"


class Env:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.world = build_demo_world()
        self.suggestions = InMemorySuggestionRepository()
        self.knowledge = InMemoryKnowledgeRepository(self.clock)
        self.faq_ids = seed_knowledge(self.knowledge, self.world)
        self.feedback = InMemoryFeedbackRepository()
        self.audit = InMemoryAuditSink()
        self.approved: list[KnowledgeItem] = []
        self.service = FeedbackService(
            self.suggestions,
            self.knowledge,
            self.feedback,
            self.audit,
            self.clock,
            on_approved=self.approved.append,
        )

    def add(self, **over) -> SuggestionRecord:
        fields = {
            "shop_id": SHOP_A,
            "conversation_id": CONV,
            "msg_id": "m_1",
            "question_text": QUESTION,
            "item_id": "i_555",
            "intent": Intent.SIZE_ADVICE,
            "intent_confidence": 0.72,
            "route_rule": 6,
            "text": "亲，建议您选 M 码哦~",
            "kind": DraftKind.LLM,
            "evidence_ids": [ev.item_size_advice("i_555"), ev.item_size_chart("i_555")],
            "qa_scores": {"grounded": 0.96},
            "status": SuggestionStatus.SHOWN,
            "agent_id": AGENT,
            "latency_ms": 1800,
            "created_at": self.clock.now(),
        }
        fields.update(over)
        return self.suggestions.add(SuggestionRecord(**fields))

    def events(self, action: str) -> list[dict]:
        return [e.detail for e in self.audit.events if e.action == action]


@pytest.fixture
def env() -> Env:
    return Env()


def _audit_dump(env: Env) -> str:
    return json.dumps([e.model_dump(mode="json") for e in env.audit.events], ensure_ascii=False)


# ================================================================ 原样发送


def test_record_sent(env):
    rec = env.add()
    env.clock.advance(seconds=30)
    out = env.service.record_sent(SHOP_A, rec.id, agent_id="kf_07")
    assert out.record.status == SuggestionStatus.SENT and out.record.agent_id == "kf_07"
    assert env.suggestions.get(SHOP_A, rec.id).status == SuggestionStatus.SENT
    [label] = env.feedback.list_labels(SHOP_A)
    assert label == out.label
    assert (label.text, label.predicted_intent, label.true_intent, label.source) == (
        QUESTION,
        Intent.SIZE_ADVICE,
        Intent.SIZE_ADVICE,
        "feedback_adopt",
    )
    assert label.predicted_confidence == 0.72  # §13.5 重算阈值的三元组要有置信度
    assert label.created_at == env.clock.now()
    [event] = [e for e in env.audit.events if e.action == AuditAction.FEEDBACK_SENT]
    assert event.actor == "kf_07" and event.shop_id == SHOP_A
    assert event.detail["suggestion_id"] == rec.id and event.detail["intent"] == "size_advice"
    assert event.detail["labeled"] is True and event.detail["edited_unchanged"] is False
    assert QUESTION not in _audit_dump(env) and rec.text not in _audit_dump(env)


def test_record_sent_masks_question_in_label(env):
    rec = env.add(question_text=f"我手机 {PHONE} 能改地址吗", intent=Intent.MODIFY_ORDER)
    out = env.service.record_sent(SHOP_A, rec.id, agent_id=AGENT)
    assert PHONE not in out.label.text and PHONE_MASK in out.label.text


def test_record_sent_without_question_has_no_label(env):
    rec = env.add(question_text=None)
    out = env.service.record_sent(SHOP_A, rec.id, agent_id=AGENT)
    assert out.label is None and env.feedback.list_labels(SHOP_A) == []
    assert env.events(AuditAction.FEEDBACK_SENT)[0]["labeled"] is False


def test_split_message_is_not_labeled_with_one_intent(env):
    """多问题分条（规则 5）：整条消息不只属于一个类型，不写单标签样本。"""
    ship = env.add(intent=Intent.SHIPPING_POLICY, question_text="发什么快递，有运费险吗")
    env.add(intent=Intent.INVOICE_GIFT, question_text="发什么快递，有运费险吗")
    out = env.service.record_sent(SHOP_A, ship.id, agent_id=AGENT)
    assert out.label is None
    # 同一类型的多条候选（一次推荐两条）不算分条
    first = env.add(msg_id="m_2")
    env.add(msg_id="m_2", text="亲，M 码更合适哦")
    assert env.service.record_sent(SHOP_A, first.id, agent_id=AGENT).label is not None


# ================================================================ 编辑后发送


def test_record_edited_creates_candidate_for_merchant_review(env):
    rec = env.add()
    edited = "亲，按您的身高体重建议选 M 码，版型偏小哦~"
    out = env.service.record_edited(SHOP_A, rec.id, agent_id=AGENT, edited_text=edited)
    assert out.record.status == SuggestionStatus.EDITED_SENT
    assert out.record.edited_text == edited
    cand = out.candidate
    assert cand is not None and cand.id is not None and out.candidate_skipped is None
    assert (cand.question, cand.answer, cand.intent, cand.item_id) == (
        QUESTION,
        edited,
        Intent.SIZE_ADVICE,
        "i_555",
    )
    assert (cand.source, cand.status, cand.approved_by) == (
        KnowledgeSource.ADOPTED,
        KnowledgeStatus.CANDIDATE,
        None,
    )
    # 候选知识在商家审核前不能用于回答
    approved = env.knowledge.list_knowledge(SHOP_A, item_id="i_555")
    assert cand.id not in {k.id for k in approved}
    assert cand.id in {k.id for k in env.service.pending_candidates(SHOP_A)}
    [edit_event] = env.events(AuditAction.FEEDBACK_EDITED)
    assert edit_event["candidate_id"] == cand.id and 0 < edit_event["edit_similarity"] < 1
    [cand_event] = env.events(AuditAction.KNOWLEDGE_CANDIDATE)
    assert cand_event["knowledge_id"] == cand.id and cand_event["item_id"] == "i_555"
    assert edited not in _audit_dump(env)
    # 编辑后发送：类型判断对了，记一条 feedback_edit 正样本（§5.8、§6.4），带置信度和路由规则
    [label] = env.feedback.list_labels(SHOP_A)
    assert label == out.label
    assert (label.predicted_intent, label.true_intent, label.source) == (
        Intent.SIZE_ADVICE,
        Intent.SIZE_ADVICE,
        "feedback_edit",
    )
    assert (label.predicted_confidence, label.route_rule) == (0.72, 6)


def test_record_edited_masks_personal_info(env):
    rec = env.add(intent=Intent.MODIFY_ORDER, evidence_ids=[ev.policy("modify_order")])
    edited = f"亲，已记下新地址 杭州市西湖区文三路 100 号 3 幢 502 室，电话 {PHONE}"
    out = env.service.record_edited(SHOP_A, rec.id, agent_id=AGENT, edited_text=edited)
    for text in (out.record.edited_text, out.candidate.answer):
        assert PHONE not in text and "文三路" not in text
        assert PHONE_MASK in text and ADDRESS_MASK in text
    assert PHONE not in _audit_dump(env) and "文三路" not in _audit_dump(env)
    stored = env.suggestions.get(SHOP_A, rec.id)
    assert PHONE not in stored.edited_text


def test_record_edited_unchanged_counts_as_sent(env):
    rec = env.add()
    out = env.service.record_edited(SHOP_A, rec.id, agent_id=AGENT, edited_text=f"  {rec.text} ")
    assert out.record.status == SuggestionStatus.SENT and out.record.edited_text is None
    assert out.candidate is None and out.label is not None
    assert env.events(AuditAction.FEEDBACK_SENT)[0]["edited_unchanged"] is True
    assert env.events(AuditAction.FEEDBACK_EDITED) == []


def test_record_edited_on_realtime_evidence_does_not_create_faq(env):
    rec = env.add(
        intent=Intent.LOGISTICS_TRACK,
        question_text="我的快递到哪了",
        item_id=None,
        evidence_ids=[ev.tool_order("o_1002"), ev.tool_logistics("o_1002")],
    )
    out = env.service.record_edited(
        SHOP_A, rec.id, agent_id=AGENT, edited_text="亲，截至 10:30 已到杭州转运中心啦"
    )
    assert out.record.status == SuggestionStatus.EDITED_SENT
    assert out.candidate is None and out.candidate_skipped == SKIP_REALTIME
    assert env.service.pending_candidates(SHOP_A) == [
        k for k in env.service.pending_candidates(SHOP_A) if k.question != "我的快递到哪了"
    ]
    assert env.events(AuditAction.FEEDBACK_EDITED)[0]["candidate_skipped"] == SKIP_REALTIME
    assert env.events(AuditAction.KNOWLEDGE_CANDIDATE) == []


def test_record_edited_without_question_or_duplicate(env):
    no_q = env.add(question_text=None)
    out = env.service.record_edited(SHOP_A, no_q.id, agent_id=AGENT, edited_text="亲，选 M 码")
    assert out.candidate is None and out.candidate_skipped == SKIP_NO_QUESTION
    first = env.add(msg_id="m_a")
    second = env.add(msg_id="m_b")
    edited = "亲，建议 M 码，偏小一码哦"
    made = env.service.record_edited(SHOP_A, first.id, agent_id=AGENT, edited_text=edited)
    assert made.candidate is not None
    dup = env.service.record_edited(SHOP_A, second.id, agent_id=AGENT, edited_text=edited)
    # 同样的候选已在等审核：不新建，这次反馈归到那一条上（审计里也记下它的 id）
    assert dup.candidate is not None and dup.candidate.id == made.candidate.id
    assert dup.candidate_skipped is None
    assert [k.answer for k in env.service.pending_candidates(SHOP_A)].count(edited) == 1
    assert dup.record.status == SuggestionStatus.EDITED_SENT


@pytest.mark.parametrize("text", ["", "   ", None])
def test_record_edited_requires_text(env, text):
    rec = env.add()
    with pytest.raises(ValueError):
        env.service.record_edited(SHOP_A, rec.id, agent_id=AGENT, edited_text=text)
    assert env.suggestions.get(SHOP_A, rec.id).status == SuggestionStatus.SHOWN


# ================================================================ 不合适（按原因分流）


def test_reject_missing_knowledge_creates_gap(env):
    rec = env.add(question_text=f"能开专票吗 电话 {PHONE}", intent=Intent.INVOICE_GIFT)
    env.clock.advance(minutes=1)
    out = env.service.record_rejected(
        SHOP_A, rec.id, agent_id=AGENT, reason=RejectReason.MISSING_KNOWLEDGE
    )
    assert out.record.status == SuggestionStatus.REJECTED
    assert out.record.reject_reason == RejectReason.MISSING_KNOWLEDGE
    [gap] = env.feedback.list_gaps(SHOP_A)
    assert gap == out.gap
    assert (gap.source, gap.intent, gap.missing) == (
        "feedback",
        Intent.INVOICE_GIFT,
        "item:i_555:invoice_gift",
    )
    assert PHONE not in gap.question_text and PHONE_MASK in gap.question_text
    assert gap.created_at == env.clock.now()
    no_item = env.add(msg_id="m_2", item_id=None, intent=Intent.INVOICE_GIFT)
    out2 = env.service.record_rejected(
        SHOP_A, no_item.id, agent_id=AGENT, reason=RejectReason.MISSING_KNOWLEDGE
    )
    assert out2.gap.missing == "intent:invoice_gift"
    assert env.events(AuditAction.FEEDBACK_REJECTED)[0]["gap_missing"] == gap.missing
    assert PHONE not in _audit_dump(env)


def test_reject_wrong_intent_creates_label(env):
    rec = env.add(question_text="这个会起球吗", intent=Intent.SIZE_ADVICE)
    out = env.service.record_rejected(
        SHOP_A,
        rec.id,
        agent_id=AGENT,
        reason=RejectReason.WRONG_INTENT,
        true_intent=Intent.PRODUCT_SPEC,
    )
    [label] = env.feedback.list_labels(SHOP_A)
    assert label == out.label
    assert (label.predicted_intent, label.true_intent, label.source) == (
        Intent.SIZE_ADVICE,
        Intent.PRODUCT_SPEC,
        "feedback_reject",
    )
    assert label.predicted_confidence == 0.72
    detail = env.events(AuditAction.FEEDBACK_REJECTED)[0]
    assert detail["reason"] == "wrong_intent" and detail["true_intent"] == "product_spec"
    # 只知道"不是这个类型"也记录
    other = env.add(msg_id="m_2")
    out2 = env.service.record_rejected(
        SHOP_A, other.id, agent_id=AGENT, reason=RejectReason.WRONG_INTENT
    )
    assert out2.label.true_intent is None


def test_reject_wrong_intent_cannot_confirm_same_intent(env):
    rec = env.add()
    with pytest.raises(ValueError):
        env.service.record_rejected(
            SHOP_A,
            rec.id,
            agent_id=AGENT,
            reason=RejectReason.WRONG_INTENT,
            true_intent=Intent.SIZE_ADVICE,
        )
    assert env.suggestions.get(SHOP_A, rec.id).status == SuggestionStatus.SHOWN
    assert env.feedback.list_labels(SHOP_A) == [] and env.audit.events == []


def test_reject_bad_wording_goes_to_tuning_queue(env):
    rec = env.add()
    out = env.service.record_rejected(
        SHOP_A, rec.id, agent_id=AGENT, reason="bad_wording", true_intent=Intent.OTHER
    )
    assert out.gap is None and out.label is None
    [tune] = env.events(AuditAction.FEEDBACK_BAD_WORDING)
    assert tune["suggestion_id"] == rec.id and tune["kind"] == "llm"
    assert tune["qa_scores"] == {"grounded": 0.96}
    # true_intent 只在理解错误时使用
    assert env.events(AuditAction.FEEDBACK_REJECTED)[0]["true_intent"] is None


@pytest.mark.parametrize("reason", [RejectReason.WRONG_FACT, RejectReason.OTHER])
def test_reject_other_reasons_only_audit(env, reason):
    rec = env.add()
    out = env.service.record_rejected(SHOP_A, rec.id, agent_id=AGENT, reason=reason)
    assert out.record.reject_reason == reason
    assert out.gap is None and out.label is None
    assert env.feedback.list_gaps(SHOP_A) == [] and env.feedback.list_labels(SHOP_A) == []
    detail = env.events(AuditAction.FEEDBACK_REJECTED)[0]
    assert detail["evidence_ids"] == rec.evidence_ids
    assert env.events(AuditAction.FEEDBACK_BAD_WORDING) == []


# ================================================================ 非法迁移、串店、操作人


def _all_ops(env: Env, shop_id: str, sid: int):
    return [
        lambda: env.service.record_sent(shop_id, sid, agent_id=AGENT),
        lambda: env.service.record_edited(shop_id, sid, agent_id=AGENT, edited_text="改过的"),
        lambda: env.service.record_rejected(
            shop_id, sid, agent_id=AGENT, reason=RejectReason.OTHER
        ),
    ]


@pytest.mark.parametrize(
    "status",
    [
        SuggestionStatus.SHADOW,
        SuggestionStatus.EXPIRED,
        SuggestionStatus.SENT,
        SuggestionStatus.EDITED_SENT,
        SuggestionStatus.REJECTED,
    ],
)
def test_only_shown_can_receive_feedback(env, status):
    rec = env.add(status=status)
    for op in _all_ops(env, SHOP_A, rec.id):
        with pytest.raises(ActionNotAllowed):
            op()
    assert env.suggestions.get(SHOP_A, rec.id).status == status
    assert env.audit.events == [] and env.feedback.list_labels(SHOP_A) == []


def test_double_click_is_rejected(env):
    rec = env.add()
    env.service.record_sent(SHOP_A, rec.id, agent_id=AGENT)
    with pytest.raises(ActionNotAllowed):
        env.service.record_rejected(SHOP_A, rec.id, agent_id=AGENT, reason=RejectReason.OTHER)
    assert len(env.feedback.list_labels(SHOP_A)) == 1


def test_other_shop_ids_are_treated_as_not_found(env):
    b_rec = env.add(shop_id=SHOP_B)
    for op in _all_ops(env, SHOP_A, b_rec.id):
        with pytest.raises(KeyError):
            op()
    for op in _all_ops(env, SHOP_A, 999_999):
        with pytest.raises(KeyError):
            op()
    assert env.suggestions.get(SHOP_B, b_rec.id).status == SuggestionStatus.SHOWN
    assert env.audit.events == []


@pytest.mark.parametrize("agent", ["", "  ", "system", "BOT", None])
def test_feedback_requires_a_human_agent(env, agent):
    rec = env.add()
    with pytest.raises(ValueError):
        env.service.record_sent(SHOP_A, rec.id, agent_id=agent)
    assert env.suggestions.get(SHOP_A, rec.id).status == SuggestionStatus.SHOWN


# ================================================================ 商家审核候选知识


def test_approve_candidate_becomes_faq(env):
    rec = env.add()
    cand = env.service.record_edited(
        SHOP_A, rec.id, agent_id=AGENT, edited_text="亲，建议 M 码，偏小一码哦"
    ).candidate
    env.clock.advance(hours=2)
    saved = env.service.approve_candidate(SHOP_A, cand.id, merchant="merchant_owner")
    assert saved.status == KnowledgeStatus.APPROVED and saved.approved_by == "merchant_owner"
    assert cand.id in {k.id for k in env.knowledge.list_knowledge(SHOP_A, item_id="i_555")}
    assert env.approved == [saved]
    [event] = [e for e in env.audit.events if e.action == AuditAction.KNOWLEDGE_APPROVED]
    assert event.actor == "merchant_owner" and event.created_at == env.clock.now()
    assert event.detail == {
        "knowledge_id": cand.id,
        "source": "adopted",
        "status_before": "candidate",
        "intent": "size_advice",
        "item_id": "i_555",
    }


def test_approve_seeded_candidate_and_guards(env):
    cand_id = env.faq_ids[(SHOP_A, "可以寄到新疆吗")]
    approved_faq = env.faq_ids[(SHOP_A, "发什么快递")]
    retired_faq = env.faq_ids[(SHOP_A, "双十一有活动吗")]
    b_faq = env.faq_ids[(SHOP_B, "发什么快递")]
    with pytest.raises(ActionNotAllowed):
        env.service.approve_candidate(SHOP_A, approved_faq, merchant="m")
    with pytest.raises(ActionNotAllowed):
        env.service.approve_candidate(SHOP_A, retired_faq, merchant="m")
    with pytest.raises(KeyError):
        env.service.approve_candidate(SHOP_A, b_faq, merchant="m")  # 别的店铺
    with pytest.raises(ValueError):
        env.service.approve_candidate(SHOP_A, cand_id, merchant="system")
    assert env.knowledge.get_knowledge(SHOP_A, cand_id).status == KnowledgeStatus.CANDIDATE
    assert env.approved == []
    env.service.approve_candidate(SHOP_A, cand_id, merchant="m")
    assert env.knowledge.get_knowledge(SHOP_A, cand_id).status == KnowledgeStatus.APPROVED


def test_retire_candidate_or_faq(env):
    cand_id = env.faq_ids[(SHOP_A, "可以寄到新疆吗")]
    faq_id = env.faq_ids[(SHOP_A, "有运费险吗")]
    assert env.service.retire(SHOP_A, cand_id, merchant="m").status == KnowledgeStatus.RETIRED
    assert env.service.retire(SHOP_A, faq_id, merchant="m").status == KnowledgeStatus.RETIRED
    assert faq_id not in {k.id for k in env.knowledge.list_knowledge(SHOP_A)}
    with pytest.raises(ActionNotAllowed):
        env.service.retire(SHOP_A, faq_id, merchant="m")
    with pytest.raises(ActionNotAllowed):
        env.service.approve_candidate(SHOP_A, cand_id, merchant="m")  # 驳回后不能再通过
    with pytest.raises(KeyError):
        env.service.retire(SHOP_A, env.faq_ids[(SHOP_B, "发什么快递")], merchant="m")
    assert [e["status_before"] for e in env.events(KNOWLEDGE_RETIRED)] == [
        "candidate",
        "approved",
    ]


def test_pending_candidates_are_per_shop(env):
    env.add(shop_id=SHOP_B, msg_id="b1")
    b_rec = env.suggestions.list(SHOP_B)[0]
    env.service.record_edited(SHOP_B, b_rec.id, agent_id=AGENT, edited_text="您好，建议 XL")
    assert {k.shop_id for k in env.service.pending_candidates(SHOP_A)} == {SHOP_A}
    assert [k.answer for k in env.service.pending_candidates(SHOP_B)] == ["您好，建议 XL"]


def test_feedback_timestamps_use_injected_clock(env):
    rec = env.add()
    env.clock.advance(days=1)
    env.service.record_rejected(
        SHOP_A, rec.id, agent_id=AGENT, reason=RejectReason.MISSING_KNOWLEDGE
    )
    assert env.audit.events[0].created_at == env.clock.now()
    assert env.feedback.list_gaps(SHOP_A)[0].created_at == rec.created_at + timedelta(days=1)
