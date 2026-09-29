"""反馈学习闭环（方案 §5.8）：客服对推荐的每一次操作都变成数据。

    原样发送 → 正样本（计入采纳率；判断层标注集 feedback_adopt）
    编辑后发送 → 记录修改前后 → 候选知识 → 商家审核通过 → FAQ 与优质回答
    不合适 → 记录原因：知识缺失 → 提醒商家补知识；理解错误 → 判断层标注集（feedback_reject）；
             表达不好 → 调整提示词和语气设置（审计事件 feedback.bad_wording，作为调优队列）

状态迁移：只有 SHOWN 能变成 SENT / EDITED_SENT / REJECTED；其余（SHADOW 没展示过、EXPIRED 已被
买家新消息刷新、已经操作过的）一律 ActionNotAllowed。SHOWN → EXPIRED 由引擎做，不在这里。

店铺隔离：推荐记录、知识都按 shop_id 读取；用别的店铺的 ID 操作和"不存在"一样抛 KeyError，
不暴露别的店铺有没有这条记录（§9.2）。

留痕：每次操作写审计（actor = 客服子账号 / 商家），detail 只有 ID、类型、原因、相似度，
不放买家原文和回答原文（§4.1 原则 6、§12.3 日志脱敏）。写进标注集、知识缺失、候选知识的买家问题
和编辑后的回答都先做个人信息最小化（§5.3，见 minimize_text）。

写到一半失败：先用比较后更新把推荐改成 SENT / EDITED_SENT / REJECTED（防重复反馈的"占位"），
再写派生数据（标注、知识缺失、候选知识）和审计。后面任何一步失败，就把推荐记录退回展示中
再抛出异常（补偿），客服重试能完整走一遍，派生数据和审计不会因为状态已改而永久丢失。
重试时候选知识按内容去重（归到上次写进去的那条，审计照常补上）；标注、知识缺失在"派生数据写成、
审计失败"时可能多出一条（至少一次）。

多实例（§5.8 候选知识去重）：进程内的锁只管本实例。两个实例同时把同样的编辑回复变成候选知识时，
由存储保证只建一条（KnowledgeRepository.add_knowledge 对重复的等审核候选抛 ConcurrencyConflict，
PostgreSQL 是部分唯一索引），后到的一方重新读到先建的那条、归到它上面（见 _make_candidate）。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from rodit.core import evidence as ev
from rodit.core.audit import AuditAction
from rodit.core.clock import Clock
from rodit.core.enums import (
    Intent,
    KnowledgeSource,
    KnowledgeStatus,
    RejectReason,
    SuggestionStatus,
)
from rodit.core.errors import ActionNotAllowed, ConcurrencyConflict
from rodit.core.models import (
    AuditEvent,
    KnowledgeGap,
    KnowledgeItem,
    LabeledSample,
    SuggestionRecord,
    strip_control_chars,
)
from rodit.core.pii import contains_pii, mask_pii
from rodit.core.ports import (
    AuditSink,
    FeedbackRepository,
    KnowledgeRepository,
    SuggestionRepository,
)
from rodit.feedback.metrics import char_similarity

KNOWLEDGE_RETIRED = AuditAction.KNOWLEDGE_RETIRED.value  # 兼容旧名
"""商家下线 FAQ / 驳回候选知识的审计事件名（core.audit.AuditAction 里还没有，见回执）。"""

# 候选知识没有生成的原因（FeedbackOutcome.candidate_skipped）
SKIP_NO_QUESTION = "no_question"  # 推荐记录没有买家问题，生成不了 FAQ 的问题
SKIP_REALTIME = "realtime_evidence"  # 回答依据实时数据（订单、物流、库存、价格……），不能沉淀成 FAQ
# 同样的候选知识已在等商家审核。已不再产生：这种情况下这次反馈归到那条候选上（返回它、写审计），
# 见 FeedbackService._make_candidate；保留常量给旧调用方
SKIP_DUPLICATE = "duplicate"

_REALTIME_PREFIXES = tuple(
    build("")
    for build in (
        ev.tool_order,
        ev.tool_logistics,
        ev.tool_aftersale,
        ev.tool_return,
        ev.tool_stock,
        ev.tool_price,
    )
)
_NOT_HUMAN = frozenset({"system", "bot", "auto"})
_CANDIDATE_ATTEMPTS = 3  # 新建候选知识撞上别的实例刚建的同样候选时，最多重读几次

_log = logging.getLogger(__name__)


class FeedbackOutcome(BaseModel):
    """一次客服操作的结果：更新后的推荐记录 + 派生数据（没有时为 None）。"""

    model_config = ConfigDict(extra="forbid")

    record: SuggestionRecord
    label: LabeledSample | None = None
    gap: KnowledgeGap | None = None
    candidate: KnowledgeItem | None = None
    candidate_skipped: str | None = None


def minimize_text(text: str) -> str:
    """个人信息最小化：有个人信息时返回 mask_pii 的结果，没有时原样返回。

    mask_pii 会把整段文字做 NFKC 归一（全角逗号、问号变半角）；候选知识的回答以后要发给买家，
    没有个人信息时保留客服原来的标点（见回执 contract_issues）。
    """
    return mask_pii(text) if contains_pii(text) else text


def _require_human(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} 不能为空")
    if value.strip().lower() in _NOT_HUMAN:
        raise ValueError(f"{what} 必须是具体的人，不能是 {value!r}")
    return value.strip()


class FeedbackService:
    """反馈闭环（§5.8）。

    on_approved：候选知识审核通过后的回调（如知识中心给新 FAQ 算向量、刷新缓存）；
    回调抛出的异常向外传递，此时审核已经生效，可以重新触发索引。
    线程安全：同一进程内对推荐记录的"读—检查状态—更新"串行执行，避免重复点击产生两次反馈。
    """

    def __init__(
        self,
        suggestions: SuggestionRepository,
        knowledge: KnowledgeRepository,
        feedback: FeedbackRepository,
        audit: AuditSink,
        clock: Clock,
        *,
        on_approved: Callable[[KnowledgeItem], None] | None = None,
    ) -> None:
        self._suggestions = suggestions
        self._knowledge = knowledge
        self._feedback = feedback
        self._audit = audit
        self._clock = clock
        self._on_approved = on_approved
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 客服操作

    def record_sent(self, shop_id: str, suggestion_id: int, *, agent_id: str) -> FeedbackOutcome:
        """原样发送 → SENT，写一条正样本（true_intent = 推荐的类型）。

        多问题分条（同一条买家消息有不同类型的推荐）时不写标注：整条消息不只属于这一个类型。
        """
        actor = _require_human(agent_id, "agent_id")
        with self._lock:
            record = self._load_shown(shop_id, suggestion_id)
            return self._mark_sent(record, actor, edited_unchanged=False)

    def record_edited(
        self, shop_id: str, suggestion_id: int, *, agent_id: str, edited_text: str
    ) -> FeedbackOutcome:
        """编辑后发送 → EDITED_SENT，保存编辑后的文字（已脱敏），生成候选知识等商家审核；
        问题类型判断对了（客服只改了说法），记一条 feedback_edit 标注（§5.8、§6.4）。

        - 编辑后和推荐原文一样（去掉首尾空白）→ 按原样发送处理；
        - 候选知识：问题 = 推荐记录里的买家问题，回答 = 编辑后的文字（最小化），
          来源 ADOPTED、状态 CANDIDATE，限定在推荐时的商品上；同样的候选已在等审核时不新建，
          这次反馈归到那一条上（outcome.candidate 是它，审计照常记下它的 id）；
          依据里有实时数据（订单、物流、售后、退换资格、库存、价格）的不生成——这类回答因人因时而异，
          做成 FAQ 会把过期的实时数据当成标准答案（§5.2.1 实时数据优先）。
        """
        actor = _require_human(agent_id, "agent_id")
        if isinstance(edited_text, str):
            # 推荐记录用 model_copy 更新（不做校验）：先去掉 NUL 等控制字符，和模型边界的清洗一致
            edited_text = strip_control_chars(edited_text)
        if not isinstance(edited_text, str) or not edited_text.strip():
            raise ValueError("编辑后的文字不能为空")
        with self._lock:
            record = self._load_shown(shop_id, suggestion_id)
            if edited_text.strip() == record.text.strip():
                return self._mark_sent(record, actor, edited_unchanged=True)
            now = self._clock.now()
            answer = minimize_text(edited_text.strip())
            updated = self._update_shown(
                record.model_copy(
                    update={
                        "status": SuggestionStatus.EDITED_SENT,
                        "edited_text": answer,
                        "agent_id": actor,
                    }
                )
            )
            with self._undo_on_failure(record, updated):
                label = self._confirmed_label(record, "feedback_edit", now)
                candidate, skipped = self._make_candidate(record, answer, now)
                self._record(
                    AuditAction.FEEDBACK_EDITED,
                    actor,
                    record,
                    now,
                    edit_similarity=round(char_similarity(record.text, edited_text), 3),
                    candidate_id=candidate.id if candidate else None,
                    candidate_skipped=skipped,
                )
                if candidate is not None:
                    self._record(
                        AuditAction.KNOWLEDGE_CANDIDATE,
                        actor,
                        record,
                        now,
                        knowledge_id=candidate.id,
                        item_id=candidate.item_id,
                    )
            return FeedbackOutcome(
                record=updated, candidate=candidate, candidate_skipped=skipped, label=label
            )

    def record_rejected(
        self,
        shop_id: str,
        suggestion_id: int,
        *,
        agent_id: str,
        reason: RejectReason,
        true_intent: Intent | None = None,
    ) -> FeedbackOutcome:
        """不合适 → REJECTED，按原因分流（§5.8）：

        - 知识缺失 → KnowledgeGap(source="feedback")，missing 为 "item:{item_id}:{intent}"
          （有商品时）或 "intent:{intent}"，提醒商家补知识；
        - 理解错误 → LabeledSample(predicted=推荐的类型, true_intent, source="feedback_reject")；
          true_intent 可以不填（只知道不是这个类型），但不能等于推荐的类型；
        - 表达不好 → 审计 feedback.bad_wording（提示词 / 语气调优队列）；
        - 事实错误、其他 → 只记审计（detail 带依据 ID，方便复核用了哪条知识）。
        true_intent 只在理解错误时使用，其他原因忽略。
        """
        actor = _require_human(agent_id, "agent_id")
        reason = RejectReason(reason)
        true_intent = Intent(true_intent) if true_intent is not None else None
        with self._lock:
            record = self._load_shown(shop_id, suggestion_id)
            if reason == RejectReason.WRONG_INTENT and true_intent == record.intent:
                raise ValueError("理解错误时 true_intent 不能和推荐的类型相同")
            now = self._clock.now()
            updated = self._update_shown(
                record.model_copy(
                    update={
                        "status": SuggestionStatus.REJECTED,
                        "reject_reason": reason,
                        "agent_id": actor,
                    }
                )
            )
            with self._undo_on_failure(record, updated):
                question = minimize_text(record.question_text) if record.question_text else None
                gap: KnowledgeGap | None = None
                label: LabeledSample | None = None
                if reason == RejectReason.MISSING_KNOWLEDGE:
                    gap = KnowledgeGap(
                        shop_id=record.shop_id,
                        intent=record.intent,
                        missing=_gap_marker(record),
                        question_text=question,
                        source="feedback",
                        created_at=now,
                    )
                    self._feedback.add_gap(gap)
                elif reason == RejectReason.WRONG_INTENT and question:
                    label = LabeledSample(
                        shop_id=record.shop_id,
                        text=question,
                        predicted_intent=record.intent,
                        predicted_confidence=record.intent_confidence,
                        route_rule=record.route_rule,
                        true_intent=true_intent,
                        source="feedback_reject",
                        created_at=now,
                    )
                    self._feedback.add_label(label)
                self._record(
                    AuditAction.FEEDBACK_REJECTED,
                    actor,
                    record,
                    now,
                    reason=reason.value,
                    true_intent=(
                        true_intent.value
                        if true_intent and reason == RejectReason.WRONG_INTENT
                        else None
                    ),
                    evidence_ids=list(record.evidence_ids),
                    gap_missing=gap.missing if gap else None,
                    labeled=label is not None,
                )
                if reason == RejectReason.BAD_WORDING:
                    self._record(
                        AuditAction.FEEDBACK_BAD_WORDING,
                        actor,
                        record,
                        now,
                        qa_scores=dict(record.qa_scores or {}),
                    )
            return FeedbackOutcome(record=updated, gap=gap, label=label)

    # ------------------------------------------------------------ 商家审核候选知识

    def pending_candidates(self, shop_id: str) -> list[KnowledgeItem]:
        """等商家审核的候选知识（商家后台的审核队列），按 id 升序。"""
        return self._knowledge.list_knowledge(shop_id, status=KnowledgeStatus.CANDIDATE)

    def approve_candidate(self, shop_id: str, knowledge_id: int, *, merchant: str) -> KnowledgeItem:
        """商家审核通过：CANDIDATE → APPROVED，此后才能用于回答（§5.8 只有商家审核过的才入库）。

        不存在或不属于该店铺抛 KeyError；不是候选状态抛 ActionNotAllowed。
        """
        actor = _require_human(merchant, "merchant")
        with self._lock:
            item = self._knowledge.get_knowledge(shop_id, knowledge_id)
            if item is None:
                raise KeyError(f"店铺 {shop_id} 没有知识 {knowledge_id}")
            if item.status != KnowledgeStatus.CANDIDATE:
                raise ActionNotAllowed(
                    f"知识 {knowledge_id} 是 {item.status.value}，只有候选知识能审核通过"
                )
            saved = self._set_knowledge_status(item, KnowledgeStatus.APPROVED, approved_by=actor)
            self._record_knowledge(AuditAction.KNOWLEDGE_APPROVED.value, actor, item)
        if self._on_approved is not None:
            self._on_approved(saved)
        return saved

    def retire(self, shop_id: str, knowledge_id: int, *, merchant: str) -> KnowledgeItem:
        """商家驳回候选知识或下线 FAQ：CANDIDATE / APPROVED → RETIRED。

        不存在或不属于该店铺抛 KeyError；已经下线抛 ActionNotAllowed。
        """
        actor = _require_human(merchant, "merchant")
        with self._lock:
            item = self._knowledge.get_knowledge(shop_id, knowledge_id)
            if item is None:
                raise KeyError(f"店铺 {shop_id} 没有知识 {knowledge_id}")
            if item.status == KnowledgeStatus.RETIRED:
                raise ActionNotAllowed(f"知识 {knowledge_id} 已经下线")
            saved = self._set_knowledge_status(item, KnowledgeStatus.RETIRED)
            self._record_knowledge(KNOWLEDGE_RETIRED, actor, item)
            return saved

    # ------------------------------------------------------------ 内部

    def _set_knowledge_status(
        self, item: KnowledgeItem, status: KnowledgeStatus, *, approved_by: str | None = None
    ) -> KnowledgeItem:
        """比较后更新：只有状态仍是刚读到的那个才写入。进程内的锁管不到别的实例，
        多实例部署时商家审核通过和驳回同时发生，后写的不能覆盖先写的。"""
        assert item.id is not None
        try:
            return self._knowledge.set_knowledge_status(
                item.shop_id,
                item.id,
                status,
                approved_by=approved_by,
                expected_status=item.status,
            )
        except ConcurrencyConflict as exc:
            raise ActionNotAllowed(f"知识 {item.id} 的状态已经变化，请刷新后再操作") from exc

    def _load_shown(self, shop_id: str, suggestion_id: int) -> SuggestionRecord:
        record = self._suggestions.get(shop_id, suggestion_id)
        if record is None:
            raise KeyError(f"店铺 {shop_id} 没有推荐记录 {suggestion_id}")
        if record.status != SuggestionStatus.SHOWN:
            raise ActionNotAllowed(
                f"推荐 {suggestion_id} 的状态是 {record.status.value}，只有展示中的推荐能反馈"
            )
        return record

    def _update_shown(self, record: SuggestionRecord) -> SuggestionRecord:
        """比较后更新：只有仍是 SHOWN 才写入（多实例部署时防止重复反馈、或和过期撞车）。"""
        try:
            return self._suggestions.update(record, expected_status=SuggestionStatus.SHOWN)
        except ConcurrencyConflict as exc:
            raise ActionNotAllowed(f"推荐 {record.id} 的状态已经变化，请刷新后再操作") from exc

    @contextmanager
    def _undo_on_failure(
        self, original: SuggestionRecord, updated: SuggestionRecord
    ) -> Iterator[None]:
        """补偿：推荐状态已改、后面的派生数据或审计没写成时，把推荐记录整条退回原样（展示中）
        再抛出原来的异常，让客服重试能完整走一遍。退回也是比较后更新：期间状态又变了就不动；
        退回本身失败（如存储仍不可用）只记日志，抛原来的异常。日志只带推荐 ID 和异常类型：
        数据库异常的文字可能带着绑定参数（买家问题、回答原文）。"""
        try:
            yield
        except Exception:
            try:
                self._suggestions.update(original, expected_status=updated.status)
            except Exception as undo_error:
                _log.warning(
                    "反馈没写完，推荐 %s 退回展示中也失败了（%s）",
                    original.id,
                    type(undo_error).__name__,
                )
            raise

    def _mark_sent(
        self, record: SuggestionRecord, actor: str, *, edited_unchanged: bool
    ) -> FeedbackOutcome:
        now = self._clock.now()
        updated = self._update_shown(
            record.model_copy(update={"status": SuggestionStatus.SENT, "agent_id": actor})
        )
        with self._undo_on_failure(record, updated):
            label = self._confirmed_label(record, "feedback_adopt", now)
            self._record(
                AuditAction.FEEDBACK_SENT,
                actor,
                record,
                now,
                edited_unchanged=edited_unchanged,
                labeled=label is not None,
            )
        return FeedbackOutcome(record=updated, label=label)

    def _confirmed_label(
        self,
        record: SuggestionRecord,
        source: Literal["feedback_adopt", "feedback_edit"],
        now: datetime,
    ) -> LabeledSample | None:
        """客服发出去了（原样或编辑后）：问题类型判断对了 → 正样本（§5.8、§6.4）。
        多问题消息分条的推荐不记（整条消息的文字对不上单个类型）。"""
        if not record.question_text or self._is_split_message(record):
            return None
        label = LabeledSample(
            shop_id=record.shop_id,
            text=minimize_text(record.question_text),
            predicted_intent=record.intent,
            predicted_confidence=record.intent_confidence,
            route_rule=record.route_rule,
            true_intent=record.intent,
            source=source,
            created_at=now,
        )
        self._feedback.add_label(label)
        return label

    def _is_split_message(self, record: SuggestionRecord) -> bool:
        """同一条买家消息还有别的类型的推荐：多问题分条（§6.6 规则 5）。"""
        if record.msg_id is None:
            return False
        siblings = self._suggestions.list(record.shop_id, conversation_id=record.conversation_id)
        return any(s.msg_id == record.msg_id and s.intent != record.intent for s in siblings)

    def _make_candidate(
        self, record: SuggestionRecord, answer: str, now: datetime
    ) -> tuple[KnowledgeItem | None, str | None]:
        if not record.question_text:
            return None, SKIP_NO_QUESTION
        if any(e.startswith(_REALTIME_PREFIXES) for e in record.evidence_ids):
            return None, SKIP_REALTIME
        question = minimize_text(record.question_text)
        conflict: ConcurrencyConflict | None = None
        for _ in range(_CANDIDATE_ATTEMPTS):
            existing = self._pending_candidate(record, question, answer)
            if existing is not None:
                # 同样的候选已在等审核：这次反馈归到它上面，不再新建。它可能就是这次反馈上一次
                # 没写完（审计失败、推荐退回展示中）留下的——分不出来，所以一律当作这次反馈的
                # 候选，由调用方照常写 candidate_id 和 knowledge.candidate 审计，
                # 候选知识不会没有来历
                return existing, None
            try:
                saved = self._knowledge.add_knowledge(
                    KnowledgeItem(
                        shop_id=record.shop_id,
                        question=question,
                        answer=answer,
                        source=KnowledgeSource.ADOPTED,
                        status=KnowledgeStatus.CANDIDATE,
                        item_id=record.item_id,
                        intent=record.intent,
                        created_at=now,
                    )
                )
            except ConcurrencyConflict as exc:
                # 另一个实例刚建了同样的候选（上面的检查读到的"没有"已经过时，进程内的锁管不到
                # 别的实例）：存储挡下了第二条，重新读一次，归到先建的那条上
                conflict = exc
                continue
            return saved, None
        # 一直冲突又读不到那条（它在这期间又被审核掉了）：按写失败处理，推荐退回展示中，客服重试
        assert conflict is not None
        raise conflict

    def _pending_candidate(
        self, record: SuggestionRecord, question: str, answer: str
    ) -> KnowledgeItem | None:
        """等审核的同样候选（问题、回答、商品、问题类型都相同）。"""
        pending = self._knowledge.list_knowledge(
            record.shop_id, status=KnowledgeStatus.CANDIDATE, item_id=record.item_id
        )
        return next(
            (
                k
                for k in pending
                if k.question == question
                and k.answer == answer
                and k.item_id == record.item_id
                and k.intent == record.intent
            ),
            None,
        )

    def _record(
        self,
        action: AuditAction,
        actor: str,
        record: SuggestionRecord,
        now: datetime,
        **extra: Any,
    ) -> None:
        detail: dict[str, Any] = {
            "suggestion_id": record.id,
            "conversation_id": record.conversation_id,
            "msg_id": record.msg_id,
            "intent": record.intent.value,
            "kind": record.kind.value,
            **extra,
        }
        self._audit.record(
            AuditEvent(
                shop_id=record.shop_id,
                actor=actor,
                action=action.value,
                detail=detail,
                created_at=now,
            )
        )

    def _record_knowledge(self, action: str, actor: str, item: KnowledgeItem) -> None:
        self._audit.record(
            AuditEvent(
                shop_id=item.shop_id,
                actor=actor,
                action=action,
                detail={
                    "knowledge_id": item.id,
                    "source": item.source.value,
                    "status_before": item.status.value,
                    "intent": item.intent.value if item.intent else None,
                    "item_id": item.item_id,
                },
                created_at=self._clock.now(),
            )
        )


def _gap_marker(record: SuggestionRecord) -> str:
    """客服反馈的知识缺失：只知道问题类型（和商品），不知道具体缺哪一项。"""
    if record.item_id:
        return ev.missing_item_field(record.item_id, record.intent.value)
    return f"intent:{record.intent.value}"
