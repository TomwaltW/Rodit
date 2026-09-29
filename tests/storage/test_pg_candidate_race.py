"""候选知识去重跨实例在 PostgreSQL 上是原子的（§5.8）。

两个实例（各自一个 FeedbackService、各自一把进程内的锁）同时处理两条推荐的"编辑后发送"，
编辑后的文字、买家问题、商品、问题类型都一样：两边的"同样的候选已在等审核吗"检查都在对方写入之前
做完（用栅栏卡住），然后同时新建。部分唯一索引只让一条写进去，另一边接住冲突、归到这一条上。
全部是合成数据。
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from rodit.core import evidence as ev
from rodit.core.audit import AuditAction
from rodit.core.enums import DraftKind, Intent, KnowledgeStatus, SuggestionStatus
from rodit.core.models import KnowledgeItem, SuggestionRecord
from rodit.feedback import FeedbackService
from rodit.storage import (
    PgAuditLog,
    PgFeedbackRepository,
    PgKnowledgeRepository,
    PgSuggestionRepository,
)
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import AGENT, CONV, SHOP_A

pytestmark = pytest.mark.pg

EDITED = "亲，这款面料是95%棉加5%氨纶哦"


class _CheckThenWait:
    """每个实例第一次查等审核的候选之后，等另一个实例也查完，再往下走（都读到"没有"）。"""

    def __init__(self, inner: PgKnowledgeRepository, barrier: threading.Barrier) -> None:
        self._inner = inner
        self._barrier = barrier
        self._waited = False

    def list_knowledge(self, shop_id: str, **kw: Any) -> list[KnowledgeItem]:
        out = self._inner.list_knowledge(shop_id, **kw)
        if kw.get("status") == KnowledgeStatus.CANDIDATE and not self._waited:
            self._waited = True
            self._barrier.wait(10)
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_two_instances_editing_the_same_reply_create_one_candidate(db, truncate):
    clock = FakeClock()
    suggestions = PgSuggestionRepository(db)
    audit = PgAuditLog(db)
    records = [
        suggestions.add(
            SuggestionRecord(
                shop_id=SHOP_A,
                conversation_id=CONV,
                msg_id=msg_id,
                question_text="这款是什么面料",
                item_id="i_555",
                intent=Intent.PRODUCT_SPEC,
                text="亲，这款是棉的哦",
                kind=DraftKind.LLM,
                evidence_ids=[ev.item_attr("i_555", "面料")],
                status=SuggestionStatus.SHOWN,
                created_at=clock.now(),
            )
        )
        for msg_id in ("m_a", "m_b")
    ]
    barrier = threading.Barrier(2)
    services = [
        FeedbackService(
            suggestions,
            _CheckThenWait(PgKnowledgeRepository(db, clock=clock), barrier),
            PgFeedbackRepository(db),
            audit,
            clock,
        )
        for _ in records
    ]
    outcomes: dict[int, Any] = {}
    errors: list[BaseException] = []

    def edit(i: int) -> None:
        try:
            outcomes[i] = services[i].record_edited(
                SHOP_A, records[i].id, agent_id=AGENT, edited_text=EDITED
            )
        except BaseException as exc:  # noqa: BLE001 — 线程里的异常带回主线程断言
            errors.append(exc)

    threads = [threading.Thread(target=edit, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)

    assert errors == []
    pending = PgKnowledgeRepository(db).list_knowledge(SHOP_A, status=KnowledgeStatus.CANDIDATE)
    [candidate] = [k for k in pending if k.answer == EDITED]
    assert {outcomes[i].candidate.id for i in range(2)} == {candidate.id}
    assert all(outcomes[i].record.status == SuggestionStatus.EDITED_SENT for i in range(2))
    linked = audit.list_events(SHOP_A, action=AuditAction.KNOWLEDGE_CANDIDATE.value)
    assert [e.detail["knowledge_id"] for e in linked] == [candidate.id] * 2
