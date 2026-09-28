"""应用服务：把平台事件、客服操作、商家操作接到一个平台单元上（方案 §4.2、§5.6、§5.8、§7.3、§8）。

一条平台事件（§4.2、§8.2）：
    店铺授权检查（§8.4）→ 适配器 on_event：
    买家消息 → 按会话串行（§8.3 按 conversation_id 保证顺序；晚到的旧消息由引擎按顺序处理）
    → 问答引擎（去重、懂→找→写→判）→ 结果：
       影子模式：什么都不展示（§13.4）；
       否则：侧边栏展示推荐（模式 B，§5.6）；
       auto_reply：由适配器发送（只有模式 A 的全自动能发），发送成功记为已发送；
       发送失败（ok=False 或抛错）→ 转人工（§12.2 全自动模式下任何故障由客服接管）；
       transfer_to_human：请适配器转人工（§5.7）；转人工失败记错误日志（需要人工跟进）。
    客服 / 机器人回声 → 写进会话（§5.3 最近几轮对话；和反馈、自动回复写回的同一条回复不重复存）；
    撤回 → 从会话里去掉那条消息（不再进上下文，原文不留）。
客服操作（§5.8）：发送 / 编辑后发送 / 不合适 → 反馈闭环；发出去的文字写回会话，作为下一轮上下文。
取消订购（§8.4）与处理互斥：按店铺的读写闸，处理消息、客服 / 商家操作是读方，取消订购是写方——
等在途的处理做完、期间不开始新的，删完数据再放行（之后的事件授权检查不过）。
商家操作：审核候选知识、看缺失知识、填政策问卷（附录 C）、看报表（§12.4）。
售后审核（§7.3.2）：拉单、刷新、协商、客服确认后执行（写操作一律要人确认）。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from rodit.adapters import StandaloneAssistAdapter
from rodit.app.unit import Unit
from rodit.core.enums import (
    AfterSaleAction,
    Intent,
    Mode,
    RejectReason,
    Role,
    ShopStatus,
    SuggestionStatus,
)
from rodit.core.errors import (
    ActionNotAllowed,
    ConcurrencyConflict,
    ShopInactive,
    TenantIsolationError,
)
from rodit.core.models import (
    AfterSaleReview,
    EngineResult,
    HumanConfirmation,
    KnowledgeItem,
    StandardMessage,
    StoredMessage,
)
from rodit.core.ports import MessageRecall, PlatformAdapter
from rodit.feedback import (
    AdoptionReport,
    FeedbackOutcome,
    QAReport,
    adoption_report,
    qa_report,
)
from rodit.knowledge import apply_questionnaire, describe_gap, find_gaps

logger = logging.getLogger(__name__)

REASON_SEND_FAILED = "全自动回复发送失败，转人工（§12.2）"
# 同一次回复可能有两份记录（反馈 / 自动回复写回的、平台推回来的回声）：这个时间范围内文字相同的
# 客服 / 机器人消息算同一条
_OUTGOING_DEDUPE_WINDOW = timedelta(minutes=2)
_OUTGOING_DEDUPE_SCAN = 20


@dataclass
class _LockEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0  # 持有和等待这把锁的线程数


class _KeyedLocks:
    """按键的互斥锁（同一会话串行，§8.3）。没人持有、也没人在等的键立即删掉：
    占用的内存只和正在处理的会话数有关，不随见过的会话数增长。"""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[tuple[str, str], _LockEntry] = {}

    def __len__(self) -> int:
        with self._guard:
            return len(self._entries)

    @contextmanager
    def hold(self, key: tuple[str, str]) -> Iterator[None]:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = self._entries[key] = _LockEntry()
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    del self._entries[key]


class _ShopGate:
    """按店铺的读写闸（§8.4 取消订购后停止处理并删除数据）。

    processing（读方，可以并发）：处理平台事件、客服 / 商家操作；
    revoking（写方，独占）：等这家店在途的处理全部做完，期间新来的处理排队，写方做完再放行。
    有写方在等时新的读方也排队（写优先），繁忙的店铺也能取消订购。不可重入：持有读方时不能
    再取写方。没有进行中的处理时不占内存。
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._active: dict[str, int] = {}
        self._revoking: dict[str, int] = {}  # 正在或等着取消订购的次数
        self._exclusive: set[str] = set()

    @contextmanager
    def processing(self, shop_id: str) -> Iterator[None]:
        with self._cond:
            while self._revoking.get(shop_id):
                self._cond.wait()
            self._active[shop_id] = self._active.get(shop_id, 0) + 1
        try:
            yield
        finally:
            with self._cond:
                _decrement(self._active, shop_id)
                self._cond.notify_all()

    @contextmanager
    def revoking(self, shop_id: str) -> Iterator[None]:
        with self._cond:
            self._revoking[shop_id] = self._revoking.get(shop_id, 0) + 1
            while self._active.get(shop_id) or shop_id in self._exclusive:
                self._cond.wait()
            self._exclusive.add(shop_id)
        try:
            yield
        finally:
            with self._cond:
                self._exclusive.discard(shop_id)
                _decrement(self._revoking, shop_id)
                self._cond.notify_all()


def _decrement(counts: dict[str, int], key: str) -> None:
    n = counts.get(key, 0) - 1
    if n > 0:
        counts[key] = n
    else:
        counts.pop(key, None)


class AssistService:
    """一个平台单元的应用服务。线程安全：同一会话的消息串行处理，不同会话并行；
    取消订购等在途处理做完再删数据（见模块文档）。"""

    def __init__(self, unit: Unit) -> None:
        self.unit = unit
        self._locks = _KeyedLocks()
        self._gate = _ShopGate()

    # ================================================================ 买家消息

    def handle_event(
        self, access: str, raw: Mapping[str, Any], *, expected_shop_id: str | None = None
    ) -> EngineResult | None:
        """平台推送（模式 A）/ 插件上报（模式 B）的一条原始事件。不是买家消息、重复消息返回 None；
        客服 / 机器人回声写进会话、撤回从会话里去掉（都返回 None，§5.3）。

        先不改状态地看事件属于哪家店（peek_shop_id），核对 expected_shop_id（调用方身份）和授权，
        通过后才交给适配器处理——一个店铺的事件不能改动另一个店铺的状态（§9.2），
        授权过期的店铺连去重记录都不留（§8.4）。
        授权无效抛 AuthExpired；店铺不符抛 TenantIsolationError；
        引擎本身不会因 AI 故障抛错（§12.2）。
        """
        adapter = self.unit.adapters.get(self.unit.platform, access)
        event = dict(raw)
        shop_id = adapter.peek_shop_id(event)
        if shop_id is None:
            return None
        if expected_shop_id is not None and shop_id != expected_shop_id:
            raise TenantIsolationError("事件不属于调用方的店铺")
        with self._gate.processing(shop_id):
            self.unit.auth.require(shop_id)
            on_event = getattr(adapter, "on_event", None)  # 只实现了 on_message 的适配器照常可用
            parsed = on_event(event) if callable(on_event) else adapter.on_message(event)
            if parsed is None:
                return None
            if parsed.shop_id != shop_id:  # 适配器实现不一致时兜底
                raise TenantIsolationError("事件解析前后店铺不一致")
            if isinstance(parsed, StandardMessage):
                return self._process(adapter, parsed)
            if isinstance(parsed, StoredMessage):
                self._record_echo(parsed)
            elif isinstance(parsed, MessageRecall):
                self._record_recall(parsed)
            return None

    def _record_echo(self, echo: StoredMessage) -> None:
        """客服手打的回复（没用推荐）只能从回声里拿到：写进会话，作为最近几轮对话（§5.3）。"""
        if echo.role == Role.BUYER or not (echo.text or "").strip():
            return
        try:
            self._store_outgoing(
                shop_id=echo.shop_id,
                conversation_id=echo.conversation_id,
                msg_id=echo.msg_id,
                role=echo.role,
                text=echo.text or "",
                at=echo.created_at,
            )
        except Exception as exc:  # noqa: BLE001 — 存不进去只少一轮上下文，不影响平台推送
            logger.warning("客服回声写入会话失败（%s）", type(exc).__name__)

    def _record_recall(self, recall: MessageRecall) -> None:
        """买家撤回：那条消息不再进上下文，原文不留（§5.3、§12.3 最小化）。"""
        try:
            self.unit.repos.conversations.recall_message(
                recall.shop_id, recall.conversation_id, recall.recalled_msg_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("撤回写入会话失败（%s）", type(exc).__name__)

    def ask_standalone(
        self,
        *,
        shop_id: str,
        session_id: str,
        question: str,
        item_id: str | None = None,
        sku_id: str | None = None,
        agent_id: str | None = None,
        request_id: str | None = None,
    ) -> EngineResult | None:
        """模式 C 独立话术助手（§8.1）：客服粘贴买家问题，拿推荐回答。不需要平台授权。"""
        adapter = self.unit.adapters.get(self.unit.platform, "C")
        if not isinstance(adapter, StandaloneAssistAdapter):
            raise ActionNotAllowed("本单元没有配置模式 C 独立话术助手")
        with self._gate.processing(shop_id):
            shop = self.unit.repos.knowledge.get_shop(shop_id)
            if shop is None or shop.status == ShopStatus.SUSPENDED:
                raise ShopInactive(shop_id, shop.status if shop else None)
            raw: dict[str, Any] = {
                "shop_id": shop_id,
                "session_id": session_id,
                "question": question,
            }
            for key, value in (
                ("item_id", item_id),
                ("sku_id", sku_id),
                ("agent_id", agent_id),
                ("request_id", request_id),
            ):
                if value is not None:
                    raw[key] = value
            msg = adapter.on_message(raw)
            if msg is None:
                return None
            return self._process(adapter, msg)

    def _process(self, adapter: PlatformAdapter, msg: StandardMessage) -> EngineResult | None:
        """调用方已持有该店铺的处理闸（_gate.processing）。"""
        with self._locks.hold((msg.shop_id, msg.conversation_id)):
            result = self.unit.engine.handle(msg)
            if result is None:
                return None
            if result.mode != Mode.SHADOW:
                adapter.show_suggestions(msg.shop_id, msg.conversation_id, result.suggestions)
            reason = result.no_suggestion_reason or "转人工"
            if result.auto_reply is not None and result.mode == Mode.AUTO:
                if not self._send_auto_reply(adapter, msg, result):
                    # 买家由机器人接待，回复没发出去：不能既不回也不转（§12.2、§5.7）
                    reason = REASON_SEND_FAILED
                    result = result.model_copy(update={"transfer_to_human": True})
            if result.transfer_to_human:
                self._transfer(adapter, msg, reason)
            return result

    def _transfer(self, adapter: PlatformAdapter, msg: StandardMessage, reason: str) -> None:
        try:
            outcome = adapter.transfer_to_human(msg.shop_id, msg.conversation_id, reason)
        except ActionNotAllowed:
            return  # 该接入模式不支持转人工：客服本来就在接待（智能辅助）
        except Exception as exc:  # noqa: BLE001 — 不抛给平台：重推会被去重，更没人接
            logger.error(
                "转人工失败（%s）：店铺 %s 会话 %s 需要人工跟进",
                type(exc).__name__,
                msg.shop_id,
                msg.conversation_id,
            )
            return
        if not outcome.ok:
            logger.error(
                "转人工失败（平台返回失败）：店铺 %s 会话 %s 需要人工跟进",
                msg.shop_id,
                msg.conversation_id,
            )

    def _send_auto_reply(
        self, adapter: PlatformAdapter, msg: StandardMessage, result: EngineResult
    ) -> bool:
        """发送全自动回复；发出去了返回 True。发送失败（ok=False、接入模式不能发、抛错）返回
        False，推荐记录保持 SHOWN，留给接手的客服。"""
        reply = result.auto_reply
        assert reply is not None
        try:
            sent = adapter.send(msg.shop_id, msg.conversation_id, reply.text)
        except Exception as exc:  # noqa: BLE001 — 含 ActionNotAllowed（接入模式不能发送）
            logger.warning("全自动回复发送失败（%s），转人工", type(exc).__name__)
            return False
        if not sent.ok:
            logger.warning("全自动回复发送失败（平台返回失败），转人工")
            return False
        try:
            self._store_outgoing(
                shop_id=msg.shop_id,
                conversation_id=msg.conversation_id,
                msg_id=sent.platform_msg_id or f"bot:{reply.suggestion_id}",
                role=Role.BOT,
                text=reply.text,
                at=self.unit.clock.now(),
            )
            if reply.suggestion_id is not None:
                self._mark_sent_by_bot(msg.shop_id, reply.suggestion_id)
        except Exception as exc:  # noqa: BLE001 — 已经发出去了，只是记录没写上
            logger.warning("全自动回复已发送，记录写入失败（%s）", type(exc).__name__)
        return True

    def _mark_sent_by_bot(self, shop_id: str, suggestion_id: int) -> None:
        record = self.unit.repos.suggestions.get(shop_id, suggestion_id)
        if record is None:
            return
        try:
            self.unit.repos.suggestions.update(
                record.model_copy(update={"status": SuggestionStatus.SENT, "agent_id": "bot"}),
                expected_status=SuggestionStatus.SHOWN,
            )
        except ConcurrencyConflict:
            pass

    def _store_outgoing(
        self,
        *,
        shop_id: str,
        conversation_id: str,
        msg_id: str,
        role: Role,
        text: str,
        at: datetime,
    ) -> bool:
        """客服 / 机器人发出的文字写回会话（§5.3 多轮上下文）。存了返回 True。

        同一条回复可能有两份记录：客服点发送时的反馈（msg_id 为 agent:{推荐 id}）或全自动发送，
        和平台推回来的回声（平台消息 ID）。按 msg_id 去重之外，_OUTGOING_DEDUPE_WINDOW 内已有
        文字相同的客服 / 机器人消息时也不再存。
        """
        repo = self.unit.repos.conversations
        body = text.strip()
        recent = repo.recent_messages(shop_id, conversation_id, limit=_OUTGOING_DEDUPE_SCAN)
        for m in recent:
            if (
                m.role != Role.BUYER
                and (m.text or "").strip() == body
                and abs(m.created_at - at) <= _OUTGOING_DEDUPE_WINDOW
            ):
                return False
        return repo.add_outgoing(
            shop_id=shop_id,
            conversation_id=conversation_id,
            msg_id=msg_id,
            role=role,
            text=text,
            at=at,
        )

    # ================================================================ 客服反馈（§5.8）

    def _mirror_agent_reply(self, outcome: FeedbackOutcome, text: str) -> None:
        """客服发出的文字写回会话，作为下一轮的多轮上下文（§5.3）。回声已经先到时不重复存。"""
        record = outcome.record
        self._store_outgoing(
            shop_id=record.shop_id,
            conversation_id=record.conversation_id,
            msg_id=f"agent:{record.id}",
            role=Role.AGENT,
            text=text,
            at=self.unit.clock.now(),
        )

    def feedback_sent(self, shop_id: str, suggestion_id: int, *, agent_id: str) -> FeedbackOutcome:
        with self._gate.processing(shop_id):
            outcome = self.unit.feedback.record_sent(shop_id, suggestion_id, agent_id=agent_id)
            self._mirror_agent_reply(outcome, outcome.record.text)
            return outcome

    def feedback_edited(
        self, shop_id: str, suggestion_id: int, *, agent_id: str, edited_text: str
    ) -> FeedbackOutcome:
        with self._gate.processing(shop_id):
            outcome = self.unit.feedback.record_edited(
                shop_id, suggestion_id, agent_id=agent_id, edited_text=edited_text
            )
            sent_text = outcome.record.edited_text or outcome.record.text
            self._mirror_agent_reply(outcome, sent_text)
            return outcome

    def feedback_rejected(
        self,
        shop_id: str,
        suggestion_id: int,
        *,
        agent_id: str,
        reason: RejectReason,
        true_intent: Intent | None = None,
    ) -> FeedbackOutcome:
        with self._gate.processing(shop_id):
            return self.unit.feedback.record_rejected(
                shop_id, suggestion_id, agent_id=agent_id, reason=reason, true_intent=true_intent
            )

    # ================================================================ 商家（知识中心）

    def pending_candidates(self, shop_id: str) -> list[KnowledgeItem]:
        return self.unit.feedback.pending_candidates(shop_id)

    def approve_candidate(self, shop_id: str, knowledge_id: int, *, merchant: str) -> KnowledgeItem:
        with self._gate.processing(shop_id):
            return self.unit.feedback.approve_candidate(shop_id, knowledge_id, merchant=merchant)

    def retire_knowledge(self, shop_id: str, knowledge_id: int, *, merchant: str) -> KnowledgeItem:
        with self._gate.processing(shop_id):
            return self.unit.feedback.retire(shop_id, knowledge_id, merchant=merchant)

    def knowledge_gaps(self, shop_id: str) -> list[dict[str, str]]:
        """要提醒商家补的知识：冷启动检查（§5.2.3）+ 引擎和反馈记下的缺失（§5.7、§5.8），去重。"""
        markers = list(find_gaps(self.unit.repos.knowledge, shop_id))
        for gap in self.unit.repos.feedback.list_gaps(shop_id):
            if gap.missing not in markers:
                markers.append(gap.missing)
        return [{"marker": m, "description": describe_gap(m)} for m in markers]

    def apply_questionnaire(self, shop_id: str, answers: Mapping[int, str]) -> dict[str, Any]:
        """店铺政策问卷（附录 C）：写入政策，合并店铺配置覆盖项。"""
        repo = self.unit.repos.knowledge
        with self._gate.processing(shop_id):  # 读改写不能跨过取消订购（否则会把删掉的店铺写回来）
            shop = repo.get_shop(shop_id)
            if shop is None:
                raise KeyError(shop_id)
            entries, patch = apply_questionnaire(shop_id, dict(answers), self.unit.clock.now())
            for entry in entries:
                repo.upsert_policy(entry)
            if patch:
                repo.upsert_shop(
                    shop.model_copy(update={"config_overrides": {**shop.config_overrides, **patch}})
                )
            return {"policies": [e.key for e in entries], "config": sorted(patch)}

    # ================================================================ 报表（§12.4）

    def adoption(self, shop_id: str, *, since: datetime | None = None) -> AdoptionReport:
        records = self.unit.repos.suggestions.list(shop_id, since=since)
        return adoption_report(records, since=since)

    def qa(self, shop_id: str, *, since: datetime | None = None) -> QAReport:
        return qa_report(self.unit.repos.audit, shop_id, since)

    # ================================================================ 售后审核（§7.3.2）

    def review_aftersale(self, shop_id: str, aftersale_id: str) -> AfterSaleReview:
        with self._gate.processing(shop_id):
            existing = self.unit.repos.reviews.get(shop_id, aftersale_id)
            if existing is None:
                return self.unit.reviews.review(shop_id, aftersale_id)
            return self.unit.reviews.refresh(shop_id, aftersale_id)

    def negotiate_aftersale(
        self, shop_id: str, aftersale_id: str, *, operator: str, note: str | None = None
    ) -> AfterSaleReview:
        with self._gate.processing(shop_id):
            return self.unit.reviews.negotiate(shop_id, aftersale_id, operator=operator, note=note)

    def negotiation_done(
        self, shop_id: str, aftersale_id: str, *, operator: str, outcome: str | None = None
    ) -> AfterSaleReview:
        with self._gate.processing(shop_id):
            return self.unit.reviews.negotiation_done(
                shop_id, aftersale_id, operator=operator, outcome=outcome
            )

    def confirm_aftersale(
        self,
        shop_id: str,
        aftersale_id: str,
        action: AfterSaleAction,
        *,
        operator: str,
        expected_status_version: str,
        by_manager: bool = False,
        reason_code: str | None = None,
        reason_text: str | None = None,
    ) -> AfterSaleReview:
        """客服确认后执行（§7.3.2）：确认人、看到的平台状态版本都要带上，助手会重读再操作。"""
        confirmation = HumanConfirmation(
            operator=operator,
            confirmed_at=self.unit.clock.now(),
            expected_status_version=expected_status_version,
            by_manager=by_manager,
        )
        with self._gate.processing(shop_id):
            return self.unit.reviews.confirm(
                shop_id,
                aftersale_id,
                action,
                confirmation=confirmation,
                reason_code=reason_code,
                reason_text=reason_text,
            )

    # ================================================================ 店铺生命周期（§8.4）

    def revoke_shop(self, shop_id: str) -> dict[str, int]:
        """取消订购 / 授权过期：删令牌、停止处理，并按约定删除该店铺在本单元的全部数据。

        先等这家店在途的处理做完（期间新来的排队），再删令牌和数据，最后放行：在途处理写下的
        记录一并删掉，排队的事件授权检查不过（§8.4）。
        """
        purgers: list[object] = [*self.unit.repos.purgers(), self.unit.cache]
        for platform, access in self.unit.adapters.keys():
            adapter = self.unit.adapters.get(platform, access)
            if hasattr(adapter, "purge_shop"):
                purgers.append(adapter)
        purgers.extend(self.unit.extra_purgers)
        with self._gate.revoking(shop_id):
            return self.unit.auth.revoke(shop_id, purgers)


__all__ = ["AssistService"]
