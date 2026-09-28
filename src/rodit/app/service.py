"""应用服务：把平台事件、客服操作、商家操作接到一个平台单元上（方案 §4.2、§5.6、§5.8、§7.3、§8）。

一条买家消息（§4.2、§8.2）：
    适配器 on_message → 店铺授权检查（§8.4）→ 按会话串行（§8.3 按 conversation_id 保证顺序）
    → 问答引擎（去重、懂→找→写→判）→ 结果：
       影子模式：什么都不展示（§13.4）；
       否则：侧边栏展示推荐（模式 B，§5.6）；
       auto_reply：由适配器发送（只有模式 A 的全自动能发），发送成功记为已发送；
       transfer_to_human：请适配器转人工（§5.7）。
客服操作（§5.8）：发送 / 编辑后发送 / 不合适 → 反馈闭环；发出去的文字写回会话，作为下一轮上下文。
商家操作：审核候选知识、看缺失知识、填政策问卷（附录 C）、看报表（§12.4）。
售后审核（§7.3.2）：拉单、刷新、协商、客服确认后执行（写操作一律要人确认）。
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from rodit.adapters import StandaloneAssistAdapter
from rodit.app.unit import Unit
from rodit.core.enums import (
    AfterSaleAction,
    Intent,
    Mode,
    RejectReason,
    Role,
    SuggestionStatus,
)
from rodit.core.errors import ActionNotAllowed, ConcurrencyConflict
from rodit.core.models import (
    AfterSaleReview,
    EngineResult,
    HumanConfirmation,
    KnowledgeItem,
    StandardMessage,
)
from rodit.core.ports import PlatformAdapter
from rodit.feedback import (
    AdoptionReport,
    FeedbackOutcome,
    QAReport,
    adoption_report,
    qa_report,
)
from rodit.knowledge import apply_questionnaire, describe_gap, find_gaps


class AssistService:
    """一个平台单元的应用服务。线程安全：同一会话的消息串行处理，不同会话并行。"""

    def __init__(self, unit: Unit) -> None:
        self.unit = unit
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ================================================================ 买家消息

    def handle_event(self, access: str, raw: Mapping[str, Any]) -> EngineResult | None:
        """平台推送（模式 A）/ 插件上报（模式 B）的一条原始事件。不是买家消息、重复消息返回 None。

        授权无效抛 AuthExpired（接口层映射成 403）；引擎本身不会因为 AI 故障抛错（§12.2）。
        """
        adapter = self.unit.adapters.get(self.unit.platform, access)
        msg = adapter.on_message(dict(raw))
        if msg is None:
            return None
        self.unit.auth.require(msg.shop_id)
        return self._process(adapter, msg)

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
        raw: dict[str, Any] = {"shop_id": shop_id, "session_id": session_id, "question": question}
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

    def _lock_for(self, shop_id: str, conversation_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault((shop_id, conversation_id), threading.Lock())

    def _process(self, adapter: PlatformAdapter, msg: StandardMessage) -> EngineResult | None:
        with self._lock_for(msg.shop_id, msg.conversation_id):
            result = self.unit.engine.handle(msg)
            if result is None:
                return None
            if result.mode != Mode.SHADOW:
                adapter.show_suggestions(msg.shop_id, msg.conversation_id, result.suggestions)
            if result.auto_reply is not None and result.mode == Mode.AUTO:
                self._send_auto_reply(adapter, msg, result)
            if result.transfer_to_human:
                try:
                    adapter.transfer_to_human(
                        msg.shop_id, msg.conversation_id, result.no_suggestion_reason or "转人工"
                    )
                except ActionNotAllowed:
                    pass  # 该接入模式不支持转人工：客服本来就在接待（智能辅助）
            return result

    def _send_auto_reply(
        self, adapter: PlatformAdapter, msg: StandardMessage, result: EngineResult
    ) -> None:
        reply = result.auto_reply
        assert reply is not None
        try:
            sent = adapter.send(msg.shop_id, msg.conversation_id, reply.text)
        except ActionNotAllowed:
            return  # 接入模式不能发送：推荐照常展示，客服发送
        if not sent.ok:
            return
        now = self.unit.clock.now()
        self.unit.repos.conversations.add_outgoing(
            shop_id=msg.shop_id,
            conversation_id=msg.conversation_id,
            msg_id=sent.platform_msg_id or f"bot:{reply.suggestion_id}",
            role=Role.BOT,
            text=reply.text,
            at=now,
        )
        if reply.suggestion_id is None:
            return
        record = self.unit.repos.suggestions.get(msg.shop_id, reply.suggestion_id)
        if record is None:
            return
        try:
            self.unit.repos.suggestions.update(
                record.model_copy(update={"status": SuggestionStatus.SENT, "agent_id": "bot"}),
                expected_status=SuggestionStatus.SHOWN,
            )
        except ConcurrencyConflict:
            pass

    # ================================================================ 客服反馈（§5.8）

    def _mirror_agent_reply(self, outcome: FeedbackOutcome, text: str) -> None:
        """客服发出的文字写回会话，作为下一轮的多轮上下文（§5.3）。"""
        record = outcome.record
        self.unit.repos.conversations.add_outgoing(
            shop_id=record.shop_id,
            conversation_id=record.conversation_id,
            msg_id=f"agent:{record.id}",
            role=Role.AGENT,
            text=text,
            at=self.unit.clock.now(),
        )

    def feedback_sent(self, shop_id: str, suggestion_id: int, *, agent_id: str) -> FeedbackOutcome:
        outcome = self.unit.feedback.record_sent(shop_id, suggestion_id, agent_id=agent_id)
        self._mirror_agent_reply(outcome, outcome.record.text)
        return outcome

    def feedback_edited(
        self, shop_id: str, suggestion_id: int, *, agent_id: str, edited_text: str
    ) -> FeedbackOutcome:
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
        return self.unit.feedback.record_rejected(
            shop_id, suggestion_id, agent_id=agent_id, reason=reason, true_intent=true_intent
        )

    # ================================================================ 商家（知识中心）

    def pending_candidates(self, shop_id: str) -> list[KnowledgeItem]:
        return self.unit.feedback.pending_candidates(shop_id)

    def approve_candidate(self, shop_id: str, knowledge_id: int, *, merchant: str) -> KnowledgeItem:
        return self.unit.feedback.approve_candidate(shop_id, knowledge_id, merchant=merchant)

    def retire_knowledge(self, shop_id: str, knowledge_id: int, *, merchant: str) -> KnowledgeItem:
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
        existing = self.unit.repos.reviews.get(shop_id, aftersale_id)
        if existing is None:
            return self.unit.reviews.review(shop_id, aftersale_id)
        return self.unit.reviews.refresh(shop_id, aftersale_id)

    def negotiate_aftersale(
        self, shop_id: str, aftersale_id: str, *, operator: str, note: str | None = None
    ) -> AfterSaleReview:
        return self.unit.reviews.negotiate(shop_id, aftersale_id, operator=operator, note=note)

    def negotiation_done(
        self, shop_id: str, aftersale_id: str, *, operator: str, outcome: str | None = None
    ) -> AfterSaleReview:
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
        """取消订购 / 授权过期：删令牌、停止处理，并按约定删除该店铺在本单元的全部数据。"""
        purgers: list[object] = [*self.unit.repos.purgers(), self.unit.cache]
        for platform, access in self.unit.adapters.keys():
            adapter = self.unit.adapters.get(platform, access)
            if hasattr(adapter, "purge_shop"):
                purgers.append(adapter)
        purgers.extend(self.unit.extra_purgers)
        return self.unit.auth.revoke(shop_id, purgers)


__all__ = ["AssistService"]
