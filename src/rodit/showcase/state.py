"""演示站的运行状态：每个平台一个演示单元（方案 §9.1 一个平台云一个单元），用到时才组装。

PlatformDemo 是一个平台的演示：
- 单元用 rodit.app.demo.build_demo_unit 组装（内存仓储、模拟适配器、合成数据），生成换成演示站的
  ShowcaseLLM（见 showcase.llm），路由用关键词规则、质检用启发式判断（离线替身，不联网）；
- 单元的时钟是 FakeClock：组装时停在演示世界的"现在"（启动时间取整到分钟），回放历史时拨到过去，
  之后每次请求先拨到真实的当前时间（只往前走），推荐记录、审计的时间和真实时间一致；
- 同一个平台的请求串行处理（每个平台一把锁）：过程记录、按条数切审计都依赖这一点；
- 买家消息按平台走真实的接入方式：淘宝、抖店是模式 B 插件事件（模拟适配器的原始格式），
  拼多多是模式 C 独立话术助手（§8.1）。

ShowcaseHub 管三个平台：第一次用到某个平台时组装（各平台互不等待），重置时整个重建。
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rodit.aftersale import AFTERSALE_STATUS_LABELS, history, last_result, was_negotiated
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.app.offline import HeuristicQAJudge
from rodit.app.service import AssistService
from rodit.core.clock import CST, Clock, SystemClock
from rodit.core.config import EngineConfig
from rodit.core.enums import (
    AfterSaleAction,
    AfterSaleStatus,
    AfterSaleType,
    Intent,
    Platform,
    ProviderName,
    RejectReason,
    ReviewState,
)
from rodit.core.errors import ActionNotAllowed
from rodit.core.models import (
    AfterSaleInfo,
    AfterSaleReview,
    EngineResult,
    aftersale_idempotency_key,
)
from rodit.core.pii import mask_pii
from rodit.core.ports import LLMClient
from rodit.decision import RuleProvider
from rodit.feedback import REJECT_REASON_LABELS, FeedbackOutcome
from rodit.showcase.data import (
    AGENT_ID,
    OPERATOR,
    PROFILES,
    SHOP_SETTINGS,
    build_world,
    raw_buyer_id,
)
from rodit.showcase.labels import (
    AFTERSALE_RULE_TEXTS,
    REFUSE_REASON_CODES,
    REVIEW_ADVICE_LABELS,
    REVIEW_STATE_LABELS,
)
from rodit.showcase.llm import SIMULATED_NOTE, ShowcaseLLM
from rodit.showcase.recorder import Recorder
from rodit.showcase.scenarios import Scenario, get_scenario, scenarios_for
from rodit.showcase.seed import Feedback, HistoryMessage, HistorySummary, replay_history
from rodit.showcase.trace import (
    build_trace,
    card_view,
    degraded_label,
    gap_label,
    intent_label,
)
from rodit.tools.evidence import (
    AFTERSALE_REASON_LABELS,
    AFTERSALE_TYPE_LABELS,
    ORDER_STATUS_LABELS,
)
from rodit.tools.logistics import STATUS_LABELS as LOGISTICS_STATUS_LABELS

logger = logging.getLogger(__name__)

BUYER = "BUYER"
"""工作台里各场景的买家（合成；订单 o_1002 等都是这位买家的）。"""

AFTERSALE_WINDOW_DAYS = 7
"""售后审核页列出最近几天申请的售后单（更早的是 R7 用到的历史售后）。"""

ACTION_LABELS: dict[AfterSaleAction, str] = {
    AfterSaleAction.AGREE_REFUND: "同意退款",
    AfterSaleAction.AGREE_RETURN: "同意退货",
    AfterSaleAction.REFUSE: "拒绝售后",
}

FEEDBACK_LABELS: dict[str, str] = {
    "sent": "原样发送",
    "edited": "编辑后发送",
    "rejected": "不合适",
}

CANDIDATE_SKIPPED_LABELS: dict[str, str] = {
    "no_question": "推荐没有对应的买家问题",
    "realtime_evidence": "回答依据的是实时数据（订单、物流、库存、价格），不能沉淀成 FAQ",
    "duplicate": "同样的候选知识已在等商家审核",
}


class ShowcaseError(Exception):
    """演示站接口的错误：status 是 HTTP 状态码，message 给页面显示。"""

    status = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFound(ShowcaseError):
    status = 404


class Conflict(ShowcaseError):
    status = 409


class Invalid(ShowcaseError):
    status = 422


def demo_config() -> EngineConfig:
    """离线替身的配置：判断层用规则（和 rodit.app.demo 离线演示一致）。"""
    config = EngineConfig()
    return config.model_copy(
        update={"decision": config.decision.model_copy(update={"provider": ProviderName.RULES})}
    )


def minute_floor(dt: datetime) -> datetime:
    """取整到分钟（往下取，不会晚于真实时间）。"""
    return dt.astimezone(CST).replace(second=0, microsecond=0)


@dataclass
class ConversationState:
    """工作台里的一位买家（一个场景）。bubbles 是聊天窗口的消息，latest 是最近一条买家消息的
    处理结果。"""

    scenario: Scenario
    conversation_id: str
    bubbles: list[dict[str, Any]] = field(default_factory=list)
    latest: dict[str, Any] | None = None


class PlatformDemo:
    """一个平台的演示（见模块说明）。所有公开方法都线程安全。"""

    def __init__(
        self,
        platform: Platform,
        *,
        now: datetime,
        clock: Clock,
        llm: LLMClient | None = None,
        history_seed: int | None = 20260928,
    ) -> None:
        started = time.perf_counter()
        self.platform = platform
        self.profile = PROFILES[platform]
        self.shop_id = self.profile.shop_id
        self.now = now
        self._real_clock = clock
        self.world = build_world(platform, now)
        self.llm = ShowcaseLLM(llm)
        self.config = demo_config()
        self.unit = build_demo_unit(
            platform,
            world=self.world,
            config=self.config,
            llm=self.llm,
            route_provider=RuleProvider(),
            qa_provider=HeuristicQAJudge(),
        )
        self.clock = self.unit.clock  # FakeClock（build_demo_unit 组装的）
        self.recorder = Recorder()
        self.recorder.attach(self.unit)
        self.service = AssistService(self.unit)
        self.lock = threading.RLock()
        self._seq = itertools.count(1)
        self.conversations: dict[str, ConversationState] = {}
        self._update_shop(SHOP_SETTINGS)
        self.history = HistorySummary()
        if history_seed is not None:
            with self.llm.replay():
                self.history = replay_history(
                    platform=platform,
                    now=now,
                    seed=history_seed,
                    set_clock=self.clock.set,
                    set_shadow=lambda on: self._update_shop({"shadow_mode": on}),
                    send=self._send_history,
                    apply=self._apply_history,
                )
        # 抖店演示店在合成数据里是影子模式（§13.4）：演示里切到智能辅助，推荐才看得见
        self._update_shop({"shadow_mode": False})
        self.build_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "演示单元 %s 组装完成：回放 %d 条历史消息，用时 %d ms",
            platform.value,
            self.history.messages,
            self.build_ms,
        )

    # ================================================================ 内部

    def _update_shop(self, changes: dict[str, Any]) -> None:
        if self.unit.repos.knowledge.update_shop_config(self.shop_id, changes) is None:
            raise RuntimeError(f"演示店铺 {self.shop_id} 不在单元里")

    def tick(self) -> datetime:
        """把单元时钟拨到真实的当前时间（只往前走）。"""
        now = max(self.clock.now(), self._real_clock.now())
        self.clock.set(now)
        return now

    def _msg_id(self, prefix: str) -> str:
        return f"{prefix}-{self.platform.value}-{next(self._seq):06d}"

    def _send(
        self,
        conversation_id: str,
        text: str,
        *,
        item_id: str | None,
        order_id: str | None,
        buyer: str,
        msg_id: str,
    ) -> EngineResult | None:
        """按平台的接入方式交给 AssistService：模式 B 插件事件 / 模式 C 话术助手（§8.1）。"""
        at = self.clock.now()
        if self.platform == Platform.TAOBAO:
            raw = taobao_event(
                text,
                msg_id=msg_id,
                session_id=conversation_id,
                buyer=buyer,
                item_id=item_id,
                order_id=order_id,
                seller_id=self.shop_id,
                send_time_ms=int(at.timestamp() * 1000),
            )
            return self.service.handle_event("B", raw, expected_shop_id=self.shop_id)
        if self.platform == Platform.DOUYIN:
            content: dict[str, Any] = {"type": "text", "text": text}
            if item_id:
                content["product_id"] = item_id
            if order_id:
                content["order_id"] = order_id
            raw = {
                "shop_id": self.shop_id,
                "conversation_short_id": conversation_id,
                "server_message_id": msg_id,
                "sender": {
                    "role": "customer",
                    "uid": raw_buyer_id(self.platform, buyer),
                    "nickname": "合成昵称",  # 适配器丢弃（§5.3）
                },
                "content": content,
                "create_time": int(at.timestamp()),
                "service_uid": AGENT_ID,
                "source": "feige_plugin",
            }
            return self.service.handle_event("B", raw, expected_shop_id=self.shop_id)
        return self.service.ask_standalone(
            shop_id=self.shop_id,
            session_id=conversation_id,
            question=text,
            item_id=item_id,
            agent_id=AGENT_ID,
            request_id=msg_id,
        )

    def _send_history(self, conversation_id: str, msg: HistoryMessage) -> EngineResult | None:
        return self._send(
            conversation_id,
            msg.text,
            item_id=msg.item_id,
            order_id=msg.order_id,
            buyer=msg.buyer,
            msg_id=self._msg_id("hist"),
        )

    def _apply_history(self, fb: Feedback) -> None:
        self._feedback(fb.suggestion_id, fb.action, fb.text, fb.reason, fb.true_intent)

    def _feedback(
        self,
        suggestion_id: int,
        action: str,
        text: str | None,
        reason: RejectReason | None,
        true_intent: Intent | None = None,
    ) -> FeedbackOutcome:
        svc = self.service
        if action == "sent":
            return svc.feedback_sent(self.shop_id, suggestion_id, agent_id=AGENT_ID)
        if action == "edited":
            if not text or not text.strip():
                raise Invalid("编辑后发送需要填写发出去的文字")
            return svc.feedback_edited(
                self.shop_id, suggestion_id, agent_id=AGENT_ID, edited_text=text
            )
        if action == "rejected":
            if reason is None:
                raise Invalid("不合适需要选择原因")
            return svc.feedback_rejected(
                self.shop_id,
                suggestion_id,
                agent_id=AGENT_ID,
                reason=reason,
                true_intent=true_intent,
            )
        raise Invalid(f"不认识的操作：{action}")

    # ================================================================ 工作台

    def scenario_list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [self._scenario_view(s) for s in scenarios_for(self.platform)]

    def _card_view(self, scenario: Scenario) -> dict[str, Any] | None:
        """场景带的商品 / 订单卡片（客服工作台里显示的样子）。"""
        if scenario.order_id:
            order = next(
                (
                    o
                    for o in self.world.orders
                    if o.shop_id == self.shop_id and o.order_id == scenario.order_id
                ),
                None,
            )
            if order is None:
                return None
            trace = next(
                (
                    t
                    for t in self.world.traces
                    if t.shop_id == self.shop_id and t.order_id == order.order_id
                ),
                None,
            )
            return {
                "kind": "order",
                "id": order.order_id,
                "title": "、".join(line.title for line in order.lines),
                "status": ORDER_STATUS_LABELS[order.status],
                "amount": str(order.total_amount),
                "logistics": LOGISTICS_STATUS_LABELS[trace.status] if trace else None,
                "courier": order.courier,
            }
        if scenario.item_id:
            card = self.unit.repos.knowledge.get_item_card(self.shop_id, scenario.item_id)
            if card is None:
                return None
            price = next(
                (
                    p
                    for p in self.world.prices
                    if p.shop_id == self.shop_id and p.item_id == card.item_id
                ),
                None,
            )
            return {
                "kind": "item",
                "id": card.item_id,
                "title": card.title,
                "category": card.category,
                "price": str(price.price) if price else None,
                "attributes": dict(list(card.attributes.items())[:3]),
            }
        return None

    def _scenario_view(self, s: Scenario) -> dict[str, Any]:
        conv = self.conversations.get(s.id)
        return {
            "id": s.id,
            "title": s.title,
            "section": s.section,
            "text": s.text_for(self.platform),
            "intent": s.intent.value,
            "intent_label": intent_label(s.intent),
            "expect": s.expect_for(self.platform),
            "note": s.note,
            "card": self._card_view(s),
            "bubbles": list(conv.bubbles) if conv else [],
            "latest": conv.latest if conv else None,
        }

    def chat(self, scenario_id: str, text: str | None = None) -> dict[str, Any]:
        """买家在 scenario_id 这个会话里说一句话（text 为空时说场景预设的话），返回处理结果
        和过程。"""
        scenario = get_scenario(self.platform, scenario_id)
        if scenario is None:
            raise NotFound(f"{self.profile.workbench}没有这个演示会话：{scenario_id}")
        text = (text if text is not None else scenario.text_for(self.platform)).strip()
        if not text:
            raise Invalid("买家的话不能为空")
        with self.lock:
            now = self.tick()
            conv = self.conversations.get(scenario.id)
            if conv is None:
                conv = ConversationState(scenario, f"sc-{scenario.id}")
                self.conversations[scenario.id] = conv
            audit = self.unit.repos.audit
            decisions_before = len(audit.list_decisions(self.shop_id))
            events_before = len(audit.list_events(self.shop_id))
            scripted_before = self.llm.scripted
            msg_id = self._msg_id("live")
            with self.recorder.recording() as recording:
                result = self._send(
                    conv.conversation_id,
                    text,
                    item_id=scenario.item_id,
                    order_id=scenario.order_id,
                    buyer=BUYER,
                    msg_id=msg_id,
                )
            if result is None:
                raise Conflict("这条消息没有被处理（重复消息）")
            note = SIMULATED_NOTE if self.llm.scripted > scripted_before else None
            trace = build_trace(
                result,
                recording,
                config=self.config,
                knowledge=self.unit.repos.knowledge,
                decisions=audit.list_decisions(self.shop_id)[decisions_before:],
                events=audit.list_events(self.shop_id)[events_before:],
                simulated_note=note,
            )
            buyer_bubble = {"role": "buyer", "text": mask_pii(text), "at": now.isoformat()}
            view = {
                "scenario": scenario.id,
                "message": buyer_bubble,
                "result": self._result_view(result),
                "trace": trace,
            }
            conv.bubbles.append(buyer_bubble)
            conv.latest = view
            return view

    def _result_view(self, result: EngineResult) -> dict[str, Any]:
        route = result.route
        return {
            "mode": result.mode.value,
            "intent": route.intent.value if route else None,
            "intent_label": intent_label(route.intent) if route else None,
            "suggestions": [card_view(c) for c in result.suggestions],
            "alerts": list(result.alerts),
            "no_suggestion_reason": result.no_suggestion_reason,
            "missing_knowledge": [
                {"marker": m, "label": gap_label(m)} for m in result.missing_knowledge
            ],
            "transfer_to_human": result.transfer_to_human,
            "degraded": [degraded_label(m) for m in result.degraded],
        }

    def feedback(
        self,
        suggestion_id: int,
        action: str,
        *,
        text: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """客服对推荐的操作（§5.8）：发送 / 编辑后发送 / 不合适。"""
        reject_reason = None
        if action == "rejected":
            try:
                reject_reason = RejectReason(reason) if reason else None
            except ValueError:
                raise Invalid(f"不认识的原因：{reason}") from None
        with self.lock:
            now = self.tick()
            try:
                outcome = self._feedback(suggestion_id, action, text, reject_reason)
            except KeyError:
                raise NotFound("这条推荐不存在") from None
            except ActionNotAllowed as exc:
                raise Conflict(str(exc)) from None
            except ValueError as exc:
                raise Invalid(str(exc)) from None
            record = outcome.record
            sent_text = record.edited_text if action == "edited" else record.text
            if action == "edited" and text:
                sent_text = text.strip()
            bubble = None
            for conv in self.conversations.values():
                if conv.latest and any(
                    c["suggestion_id"] == suggestion_id
                    for c in conv.latest["result"]["suggestions"]
                ):
                    for c in conv.latest["result"]["suggestions"]:
                        if c["suggestion_id"] == suggestion_id:
                            c["status"] = action
                    if action in ("sent", "edited"):
                        bubble = {
                            "role": "agent",
                            "text": sent_text,
                            "at": now.isoformat(),
                            "suggestion_id": suggestion_id,
                            "edited": action == "edited",
                        }
                        conv.bubbles.append(bubble)
                    break
            return {
                "suggestion_id": suggestion_id,
                "action": action,
                "action_label": FEEDBACK_LABELS[action],
                "status": record.status.value,
                "bubble": bubble,
                "reason_label": REJECT_REASON_LABELS[reject_reason] if reject_reason else None,
                "candidate": None
                if outcome.candidate is None
                else {
                    "id": outcome.candidate.id,
                    "question": outcome.candidate.question,
                    "answer": outcome.candidate.answer,
                },
                "candidate_skipped": CANDIDATE_SKIPPED_LABELS.get(
                    outcome.candidate_skipped or "", outcome.candidate_skipped
                ),
                "gap": gap_label(outcome.gap.missing) if outcome.gap else None,
                "labeled": outcome.label is not None,
            }

    # ================================================================ 售后审核（§7.3.2）

    def _aftersale_info(self, aftersale_id: str) -> AfterSaleInfo | None:
        """平台上的售后单（不经缓存，和审核助手调用前重读一样）。"""
        return self.unit.tools.get_aftersale(self.shop_id, aftersale_id)

    def _aftersale_ids(self) -> list[str]:
        since = self.now.timestamp() - AFTERSALE_WINDOW_DAYS * 86400
        found = [
            a
            for a in self.world.aftersales
            if a.shop_id == self.shop_id and a.created_at.timestamp() >= since
        ]
        found.sort(key=lambda a: (a.created_at, a.aftersale_id), reverse=True)
        return [a.aftersale_id for a in found]

    def aftersales(self) -> dict[str, Any]:
        with self.lock:
            self.tick()
            items = []
            for aftersale_id in self._aftersale_ids():
                review = self.service.review_aftersale(self.shop_id, aftersale_id)
                info = self._aftersale_info(aftersale_id)
                items.append(self._aftersale_view(review, info))
            order = {ReviewState.CONFIRM: 0, ReviewState.VERIFY: 1, ReviewState.EVIDENCE: 2}
            items.sort(key=lambda v: order.get(ReviewState(v["review"]["state"]), 3))
            return {
                "items": items,
                "can_refuse": self.platform != Platform.PDD,
                "refuse_reasons": REFUSE_REASON_CODES,
                "operator": OPERATOR,
                "has_write_port": self.unit.action_port is not None,
            }

    def _actions(self, review: AfterSaleReview, info: AfterSaleInfo | None) -> list[dict]:
        if review.state != ReviewState.CONFIRM or info is None:
            return []
        out = []
        if info.status == AfterSaleStatus.PENDING_REVIEW and info.type != AfterSaleType.REFUND_ONLY:
            out.append(AfterSaleAction.AGREE_RETURN)
        if info.status == AfterSaleStatus.WAIT_SELLER_CONFIRM or (
            info.status == AfterSaleStatus.PENDING_REVIEW and info.type == AfterSaleType.REFUND_ONLY
        ):
            out.append(AfterSaleAction.AGREE_REFUND)
        if self.platform != Platform.PDD:
            out.append(AfterSaleAction.REFUSE)
        return [{"action": a.value, "label": ACTION_LABELS[a]} for a in out]

    def _aftersale_view(
        self, review: AfterSaleReview, info: AfterSaleInfo | None
    ) -> dict[str, Any]:
        order = None
        if info is not None:
            order = next(
                (
                    o
                    for o in self.world.orders
                    if o.shop_id == self.shop_id and o.order_id == info.order_id
                ),
                None,
            )
        result = last_result(review)
        return {
            "aftersale_id": review.aftersale_id,
            "order_id": info.order_id if info else None,
            "item_title": "、".join(line.title for line in order.lines) if order else None,
            "type": info.type.value if info else None,
            "type_label": AFTERSALE_TYPE_LABELS[info.type] if info else None,
            "reason": info.reason.value if info else None,
            "reason_label": AFTERSALE_REASON_LABELS[info.reason] if info else None,
            "amount": str(info.amount) if info else None,
            "created_at": info.created_at.isoformat() if info else None,
            "evidence_count": info.evidence_count if info else 0,
            "buyer_note": mask_pii(info.note) if info and info.note else None,
            "platform_status": review.platform_status.value,
            "platform_status_label": AFTERSALE_STATUS_LABELS[review.platform_status],
            "status_version": info.status_version if info else review.platform_status_version,
            "review": {
                "state": review.state.value,
                "state_label": REVIEW_STATE_LABELS[review.state],
                "rules": [
                    {"id": r, "text": AFTERSALE_RULE_TEXTS.get(r, r)} for r in review.matched_rules
                ],
                "suggestion": review.suggestion.value if review.suggestion else None,
                "suggestion_label": REVIEW_ADVICE_LABELS[review.suggestion]
                if review.suggestion
                else None,
                "reason": review.reason,
                "requires_manager": review.requires_manager,
                "risk_flags": list(review.risk_flags),
                "confirmed_by": review.confirmed_by,
                "negotiated": was_negotiated(review),
                "history": [
                    {
                        "from_state": h["from"],
                        "to_state": h["to"],
                        "from": REVIEW_STATE_LABELS[ReviewState(h["from"])],
                        "to": REVIEW_STATE_LABELS[ReviewState(h["to"])],
                        "at": h.get("at"),
                        "by": h.get("by"),
                    }
                    for h in history(review)
                ],
                "result": result.model_dump(mode="json") if result else None,
            },
            "actions": self._actions(review, info),
            "can_negotiate": review.state == ReviewState.CONFIRM,
            "can_finish_negotiation": review.state == ReviewState.NEGOTIATE,
        }

    def _status_view(self, info: AfterSaleInfo | None) -> dict[str, Any] | None:
        if info is None:
            return None
        return {
            "status": info.status.value,
            "label": AFTERSALE_STATUS_LABELS[info.status],
            "version": info.status_version,
        }

    def confirm(
        self,
        aftersale_id: str,
        action: str,
        *,
        expected_status_version: str,
        reason_code: str | None = None,
    ) -> dict[str, Any]:
        """店长确认后执行售后写操作（§7.3.2）：先重读平台状态，幂等键防止重复操作。

        同一张单、同一个操作、同一个平台状态版本已经成功执行过时（重复点击、网络重试），
        不再调平台，原样返回上次的结果（duplicate=True）。"""
        try:
            act = AfterSaleAction(action)
        except ValueError:
            raise Invalid(f"不认识的售后操作：{action}") from None
        if reason_code is not None and reason_code not in REFUSE_REASON_CODES:
            raise Invalid(f"不认识的拒绝原因码：{reason_code}")
        with self.lock:
            self.tick()
            before = self._aftersale_info(aftersale_id)
            current = self.unit.repos.reviews.get(self.shop_id, aftersale_id)
            if before is None or current is None:
                raise NotFound(f"售后单 {aftersale_id} 不存在或还没有审核建议")
            key = aftersale_idempotency_key(aftersale_id, act.value, expected_status_version)
            done = last_result(current)
            if done is not None and done.ok and done.idempotency_key == key:
                return {
                    "item": self._aftersale_view(current, before),
                    "reread": {"before": self._status_view(before), "after": None},
                    "idempotency_key": key,
                    "platform_result": done.model_dump(mode="json"),
                    "duplicate": True,
                    "operator": OPERATOR,
                }
            try:
                review = self.service.confirm_aftersale(
                    self.shop_id,
                    aftersale_id,
                    act,
                    operator=OPERATOR,
                    expected_status_version=expected_status_version,
                    by_manager=True,
                    reason_code=reason_code,
                    reason_text=REFUSE_REASON_CODES.get(reason_code or ""),
                )
            except ActionNotAllowed as exc:
                raise Conflict(str(exc)) from None
            after = self._aftersale_info(aftersale_id)
            result = last_result(review)
            return {
                "item": self._aftersale_view(review, after),
                "reread": {"before": self._status_view(before), "after": self._status_view(after)},
                "idempotency_key": result.idempotency_key if result else key,
                "platform_result": result.model_dump(mode="json") if result else None,
                "duplicate": bool(result and result.duplicate),
                "operator": OPERATOR,
            }

    def negotiate(self, aftersale_id: str, *, done: bool) -> dict[str, Any]:
        """客服选择协商（CONFIRM → NEGOTIATE），或协商有结果（NEGOTIATE → CONFIRM）。"""
        with self.lock:
            self.tick()
            if self.unit.repos.reviews.get(self.shop_id, aftersale_id) is None:
                raise NotFound(f"售后单 {aftersale_id} 不存在或还没有审核建议")
            try:
                if done:
                    review = self.service.negotiation_done(
                        self.shop_id, aftersale_id, operator=OPERATOR, outcome="协商有结果"
                    )
                else:
                    review = self.service.negotiate_aftersale(
                        self.shop_id, aftersale_id, operator=OPERATOR, note="店长选择先协商"
                    )
            except ActionNotAllowed as exc:
                raise Conflict(str(exc)) from None
            return {"item": self._aftersale_view(review, self._aftersale_info(aftersale_id))}


class ShowcaseHub:
    """三个平台的演示（见模块说明）。"""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        llm: LLMClient | None = None,
        history_seed: int | None = 20260928,
        default_platform: Platform = Platform.TAOBAO,
        factory: Callable[..., PlatformDemo] = PlatformDemo,
    ) -> None:
        self.clock = clock or SystemClock()
        self.llm = llm
        self.history_seed = history_seed
        self.default_platform = default_platform
        self._factory = factory
        self._demos: dict[Platform, PlatformDemo] = {}
        self._build_locks = {p: threading.Lock() for p in Platform}

    @property
    def offline(self) -> bool:
        return self.llm is None

    def _build(self, platform: Platform) -> PlatformDemo:
        return self._factory(
            platform,
            now=minute_floor(self.clock.now()),
            clock=self.clock,
            llm=self.llm,
            history_seed=self.history_seed,
        )

    def get(self, platform: Platform) -> PlatformDemo:
        demo = self._demos.get(platform)
        if demo is not None:
            return demo
        with self._build_locks[platform]:
            demo = self._demos.get(platform)
            if demo is None:
                demo = self._demos[platform] = self._build(platform)
            return demo

    def reset(self, platform: Platform) -> PlatformDemo:
        """重建这个平台的演示（新的演示世界、重新回放历史）。"""
        with self._build_locks[platform]:
            demo = self._demos[platform] = self._build(platform)
            return demo

    def built(self) -> list[Platform]:
        return [p for p in Platform if p in self._demos]


__all__ = [
    "Conflict",
    "ConversationState",
    "Invalid",
    "NotFound",
    "PlatformDemo",
    "ShowcaseError",
    "ShowcaseHub",
    "demo_config",
    "minute_floor",
]
