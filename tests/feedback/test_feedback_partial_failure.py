"""反馈写到一半失败（§5.8）：推荐状态已经改了，候选知识 / 知识缺失 / 标注 / 审计却没写成。

原来的顺序是先把推荐记录改成 SENT / EDITED_SENT / REJECTED，再写派生数据和审计；
后面任何一步失败，调用方拿到错误，重试时推荐已经不是"展示中"被拒绝——派生数据和审计永久丢失。
现在：状态改了之后的步骤失败，就把推荐记录退回展示中（补偿），客服重试能完整走一遍。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from rodit.core import evidence as ev
from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.enums import (
    DraftKind,
    Intent,
    KnowledgeStatus,
    RejectReason,
    SuggestionStatus,
)
from rodit.core.errors import KnowledgeUnavailable, StorageUnavailable
from rodit.core.models import AuditEvent, SuggestionRecord
from rodit.feedback import FeedbackService
from rodit.inmemory import (
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemorySuggestionRepository,
)
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import AGENT, CONV, SHOP_A


class FailOnce:
    """包一层仓储：指定的方法第一次调用抛异常，之后照常。"""

    def __init__(self, inner: Any, method: str, exc: Exception) -> None:
        self._inner = inner
        self._method = method
        self._exc: Exception | None = exc
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name != self._method:
            return attr

        def wrapped(*args: Any, **kw: Any) -> Any:
            self.calls += 1
            if self._exc is not None:
                exc, self._exc = self._exc, None
                raise exc
            return attr(*args, **kw)

        return wrapped


class Env:
    def __init__(self, fail: str | None = None) -> None:
        self.clock = FakeClock()
        self.suggestions = InMemorySuggestionRepository()
        self.knowledge_store = InMemoryKnowledgeRepository(self.clock)
        self.feedback_store = InMemoryFeedbackRepository()
        self.audit_sink = InMemoryAuditSink()
        knowledge: Any = self.knowledge_store
        feedback: Any = self.feedback_store
        audit: Any = self.audit_sink
        if fail == "add_knowledge":
            knowledge = FailOnce(knowledge, fail, KnowledgeUnavailable("知识库暂时不可用"))
        elif fail in ("add_gap", "add_label"):
            feedback = FailOnce(feedback, fail, StorageUnavailable("数据库连接失败"))
        elif fail == "record":
            audit = FailOnce(audit, fail, StorageUnavailable("数据库连接失败"))
        self.service = FeedbackService(self.suggestions, knowledge, feedback, audit, self.clock)

    def add(self, **over: Any) -> SuggestionRecord:
        fields: dict[str, Any] = {
            "shop_id": SHOP_A,
            "conversation_id": CONV,
            "msg_id": "m_1",
            "question_text": "这款会起球吗",
            "item_id": "i_555",
            "intent": Intent.PRODUCT_SPEC,
            "text": "亲，不容易起球哦",
            "kind": DraftKind.LLM,
            "evidence_ids": [ev.item_attr("i_555", "面料")],
            "status": SuggestionStatus.SHOWN,
            "created_at": self.clock.now(),
        }
        fields.update(over)
        return self.suggestions.add(SuggestionRecord(**fields))

    def actions(self) -> list[str]:
        return [e.action for e in self.audit_sink.events]

    def status(self, rec: SuggestionRecord) -> SuggestionStatus:
        return self.suggestions.get(SHOP_A, rec.id).status


def _fails_then_completes(env: Env, op: Callable[[], Any], exc: type[Exception]) -> Any:
    rec = env.suggestions.list(SHOP_A)[0]
    with pytest.raises(exc):
        op()
    # 失败后推荐仍在展示中，没有留下"已操作"的半截状态
    got = env.suggestions.get(SHOP_A, rec.id)
    assert got.status == SuggestionStatus.SHOWN
    assert got.edited_text is None and got.reject_reason is None
    return op()  # 客服重试


def test_edited_retry_after_candidate_write_failure_completes():
    env = Env(fail="add_knowledge")
    rec = env.add()
    out = _fails_then_completes(
        env,
        lambda: env.service.record_edited(
            SHOP_A, rec.id, agent_id=AGENT, edited_text="亲，正常穿着不容易起球哦"
        ),
        KnowledgeUnavailable,
    )
    assert out.record.status == SuggestionStatus.EDITED_SENT == env.status(rec)
    assert out.candidate is not None
    pending = env.knowledge_store.list_knowledge(SHOP_A, status=KnowledgeStatus.CANDIDATE)
    assert [k.answer for k in pending] == ["亲，正常穿着不容易起球哦"]
    assert env.actions() == [
        AuditAction.FEEDBACK_EDITED.value,
        AuditAction.KNOWLEDGE_CANDIDATE.value,
    ]


def test_rejected_retry_after_gap_write_failure_completes():
    env = Env(fail="add_gap")
    rec = env.add()
    out = _fails_then_completes(
        env,
        lambda: env.service.record_rejected(
            SHOP_A, rec.id, agent_id=AGENT, reason=RejectReason.MISSING_KNOWLEDGE
        ),
        StorageUnavailable,
    )
    assert out.record.status == SuggestionStatus.REJECTED == env.status(rec)
    assert len(env.feedback_store.list_gaps(SHOP_A)) == 1
    assert env.actions() == [AuditAction.FEEDBACK_REJECTED.value]


def test_rejected_retry_after_label_write_failure_completes():
    env = Env(fail="add_label")
    rec = env.add()
    out = _fails_then_completes(
        env,
        lambda: env.service.record_rejected(
            SHOP_A,
            rec.id,
            agent_id=AGENT,
            reason=RejectReason.WRONG_INTENT,
            true_intent=Intent.SIZE_ADVICE,
        ),
        StorageUnavailable,
    )
    assert out.record.status == SuggestionStatus.REJECTED
    labels = env.feedback_store.list_labels(SHOP_A)
    assert [(s.predicted_intent, s.true_intent) for s in labels] == [
        (Intent.PRODUCT_SPEC, Intent.SIZE_ADVICE)
    ]
    assert env.actions() == [AuditAction.FEEDBACK_REJECTED.value]


def test_sent_retry_after_label_write_failure_completes():
    env = Env(fail="add_label")
    rec = env.add()
    out = _fails_then_completes(
        env, lambda: env.service.record_sent(SHOP_A, rec.id, agent_id=AGENT), StorageUnavailable
    )
    assert out.record.status == SuggestionStatus.SENT == env.status(rec)
    assert len(env.feedback_store.list_labels(SHOP_A)) == 1
    assert env.actions() == [AuditAction.FEEDBACK_SENT.value]


def test_sent_retry_after_audit_failure_completes():
    env = Env(fail="record")
    rec = env.add()
    out = _fails_then_completes(
        env, lambda: env.service.record_sent(SHOP_A, rec.id, agent_id=AGENT), StorageUnavailable
    )
    assert out.record.status == SuggestionStatus.SENT
    assert env.actions() == [AuditAction.FEEDBACK_SENT.value]
    assert env.feedback_store.list_labels(SHOP_A)  # 标注至少一条（重试可能再写一条）


def test_edited_retry_after_audit_failure_does_not_duplicate_candidate():
    env = Env(fail="record")
    rec = env.add()
    _fails_then_completes(
        env,
        lambda: env.service.record_edited(
            SHOP_A, rec.id, agent_id=AGENT, edited_text="亲，正常穿着不容易起球哦"
        ),
        StorageUnavailable,
    )
    assert env.status(rec) == SuggestionStatus.EDITED_SENT
    pending = env.knowledge_store.list_knowledge(SHOP_A, status=KnowledgeStatus.CANDIDATE)
    assert len(pending) == 1  # 第一次写进去的候选还在；重试按内容去重，不再写第二条
    assert AuditAction.FEEDBACK_EDITED.value in env.actions()


def test_successful_feedback_is_still_single_shot():
    """补偿只在失败时发生：成功之后再点仍然拒绝（防重复点击的语义不变）。"""
    env = Env()
    rec = env.add()
    env.service.record_sent(SHOP_A, rec.id, agent_id=AGENT)
    from rodit.core.errors import ActionNotAllowed

    with pytest.raises(ActionNotAllowed):
        env.service.record_sent(SHOP_A, rec.id, agent_id=AGENT)
    assert len(env.feedback_store.list_labels(SHOP_A)) == 1
    assert isinstance(env.audit_sink.events[0], AuditEvent)


def test_undo_failure_still_raises_original_error(caplog):
    """退回本身也失败（存储仍不可用）：抛原来的异常，日志只带推荐 ID 和异常类型。"""
    env = Env(fail="add_label")
    rec = env.add(question_text="我电话13812345678，这款会起球吗")
    calls = {"n": 0}
    real_update = env.suggestions.update

    def update(record: SuggestionRecord, **kw: Any) -> SuggestionRecord:
        calls["n"] += 1
        if calls["n"] == 2:  # 第 1 次是改状态，第 2 次是退回
            raise StorageUnavailable("数据库连接失败 13812345678")
        return real_update(record, **kw)

    env.suggestions.update = update  # type: ignore[method-assign]
    with caplog.at_level("WARNING"), pytest.raises(StorageUnavailable, match="数据库连接失败$"):
        env.service.record_sent(SHOP_A, rec.id, agent_id=AGENT)
    assert env.status(rec) == SuggestionStatus.SENT  # 退回失败：状态停在已发送
    assert str(rec.id) in caplog.text and "StorageUnavailable" in caplog.text
    assert "13812345678" not in caplog.text
