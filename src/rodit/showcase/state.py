"""演示站的运行状态：每个平台一个演示单元（方案 §9.1 一个平台云一个单元），用到时才组装。

PlatformDemo 是一个平台的演示：
- 单元用 rodit.app.demo.build_demo_unit 组装（内存仓储、模拟适配器、合成数据），生成换成演示站的
  ShowcaseLLM（见 showcase.llm）；默认路由用关键词规则、质检用启发式判断（离线替身，不联网）；
- 接真实大模型（llm 不为空）时生成走真实大模型；judge="llm" 时路由、质检也照生产组装
  （config.decision → build_decision_provider，大模型投票，见 showcase.realchain），回放历史期间
  一律离线（ReplaySwitch，判断层和生成共用一个开关）；每次真实调用、每次大模型判断都记进过程记录；
- 单元的时钟是 FakeClock：组装时停在演示世界的"现在"（启动时间取整到分钟），回放历史时拨到过去，
  之后每次请求先拨到真实的当前时间（只往前走），推荐记录、审计的时间和真实时间一致；
- 同一个平台的请求串行处理（每个平台一把锁）：过程记录、按条数切审计都依赖这一点；
- 买家消息按平台走真实的接入方式：淘宝、抖店是模式 B 插件事件（模拟适配器的原始格式，买家图片
  也按各平台的图片消息格式），拼多多是模式 C 独立话术助手（§8.1，只有文字）；
- 组装好之后（回放历史之后）照生产的路径做两件事：复核人员给演示店铺开半自动
  （rodit.app.rollout.set_semi_auto，门槛没过时强制开启并写明原因，留审计）；系统从平台指标接口
  （合成数据）同步最近 8 周的客服考核数据（rodit.app.platform_metrics，数据看板的接入前后对比用）。

每次组装有一个随机的编号 epoch（/scenarios、/chat、/aftersales、/reset 返回）：回放是确定的，重置或
服务重启后推荐编号会重复，页面在写操作（推荐反馈、售后确认、协商）里带上载入时的 epoch，对不上就
返回 409（stale_epoch，"演示已重置，请刷新页面"），重置前的卡片不会落到重置后同一个编号的推荐上。

ShowcaseHub 管三个平台：第一次用到某个平台时组装（各平台互不等待），重置时整个重建。
"""

from __future__ import annotations

import itertools
import logging
import secrets
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial
from typing import Any

from rodit.adapters.base import PREFILL_ACCESS
from rodit.aftersale import AFTERSALE_STATUS_LABELS, history, last_result, was_negotiated
from rodit.aftersale.evidence_vision import SUMMARY_PREFIX
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.app.offline import HeuristicQAJudge
from rodit.app.platform_metrics import PlatformMetricsService, SyncResult
from rodit.app.rollout import SemiAutoChange, SemiAutoGateNotPassed
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
    REVIEWER,
    SEMI_AUTO_INTENTS,
    SEMI_AUTO_NOTE,
    SHOP_SETTINGS,
    build_world,
    raw_buyer_id,
)
from rodit.showcase.images import DEMO_IMAGE_REFS, IMAGE_PLATFORMS, NO_IMAGE_REASON, tile_view
from rodit.showcase.labels import (
    AFTERSALE_RULE_TEXTS,
    REFUSE_REASON_CODES,
    REVIEW_ADVICE_LABELS,
    REVIEW_STATE_LABELS,
)
from rodit.showcase.llm import SIMULATED_NOTE, ShowcaseLLM
from rodit.showcase.realchain import (
    ChainInfo,
    JudgeMode,
    LatencyMode,
    ObservedLLM,
    ReplaySwitch,
    build_judges,
    chain_config,
    chain_info,
)
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

IMAGE_PLACEHOLDER = "[图片]"
"""淘宝图片消息里平台放的占位文字（接入层不当作买家说的话）。"""

CONSECUTIVE_GAP_MS = 200
"""淘宝一次发几张图（每张一条消息）时，相邻两条的发送时间间隔：远小于合并连发的窗口（§5.3），
先发的时间在前，合并后图片的顺序和买家发的一样。"""

STALE_EPOCH_MESSAGE = "演示已重置，请刷新页面"
"""写操作带的 epoch 和当前演示单元对不上（页面是重置或服务重启之前载入的）。"""

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
    """演示站接口的错误：status 是 HTTP 状态码，message 给页面显示；code 是错误代码（为空时按
    状态码取，见 showcase.app）。"""

    status = 400
    code: str | None = None

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFound(ShowcaseError):
    status = 404


class Conflict(ShowcaseError):
    status = 409


class StaleDemo(Conflict):
    """页面载入之后演示单元重建过（重置、服务重启）：epoch 对不上。"""

    code = "stale_epoch"


class Invalid(ShowcaseError):
    status = 422


def demo_config(base: EngineConfig | None = None) -> EngineConfig:
    """离线判断层的配置：判断层用规则（和 rodit.app.demo 离线演示一致）。"""
    config = base or EngineConfig()
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
        config: EngineConfig | None = None,
        judge: JudgeMode = "offline",
        latency: LatencyMode = "strict",
    ) -> None:
        started = time.perf_counter()
        if judge == "llm" and llm is None:
            raise ValueError("判断层用大模型投票（judge=llm）需要真实大模型（llm）")
        self.epoch = secrets.token_hex(8)
        self.platform = platform
        self.profile = PROFILES[platform]
        self.shop_id = self.profile.shop_id
        self.now = now
        self._real_clock = clock
        self.world = build_world(platform, now)
        self.judge: JudgeMode = judge
        self.config = chain_config(config, judge=judge, latency=latency)
        self.chain: ChainInfo = chain_info(llm, self.config, judge=judge, latency=latency)
        self.recorder = Recorder()
        self.switch = ReplaySwitch()  # 回放期间生成、路由、质检都离线（见模块说明）
        self.observed: ObservedLLM | None = (
            ObservedLLM(llm, self.switch, self.recorder.llm_call_sink) if llm is not None else None
        )
        self.llm = ShowcaseLLM(self.observed, switch=self.switch)
        route_provider, qa_provider = RuleProvider(), HeuristicQAJudge()
        if judge == "llm":
            assert self.observed is not None
            route_provider, qa_provider = build_judges(
                self.config,
                self.observed,
                self.switch,
                offline_route=route_provider,
                offline_qa=qa_provider,
                observer=self.recorder.add_judge,
            )
        self.unit = build_demo_unit(
            platform,
            world=self.world,
            config=self.config,
            llm=self.llm,
            route_provider=route_provider,
            qa_provider=qa_provider,
        )
        self.clock = self.unit.clock  # FakeClock（build_demo_unit 组装的）
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
        # 演示时的操作从这一刻起（含）：比回放历史的最后一刻晚 1 微秒，回放的审计都早于它；tick()
        # 不会把时钟拨到它之前。接真实大模型（--llm env）时看板的耗时只按这之后的消息算（回放用的是
        # 离线替身，毫秒级）
        self.live_since = self.clock.now() + timedelta(microseconds=1)
        # 抖店演示店在合成数据里是影子模式（§13.4）：演示里切到智能辅助，推荐才看得见
        self._update_shop({"shadow_mode": False})
        self.semi_auto = self._enable_semi_auto()
        self.metrics_sync = self._sync_platform_metrics()
        # 回放期间真实大模型一次都不该被调到（ObservedLLM 拦下的次数也算进来，应为 0）
        self.replay_llm_calls = (
            self.observed.calls + self.observed.blocked_during_replay if self.observed else 0
        )
        self.build_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "演示单元 %s 组装完成：回放 %d 条历史消息，用时 %d ms",
            platform.value,
            self.history.messages,
            self.build_ms,
        )

    def check_epoch(self, epoch: str) -> None:
        """写操作带回来的 epoch 要是这次组装的（见模块说明），否则 409。"""
        if epoch != self.epoch:
            raise StaleDemo(STALE_EPOCH_MESSAGE)

    # ================================================================ 内部

    def _update_shop(self, changes: dict[str, Any]) -> None:
        if self.unit.repos.knowledge.update_shop_config(self.shop_id, changes) is None:
            raise RuntimeError(f"演示店铺 {self.shop_id} 不在单元里")

    def _enable_semi_auto(self) -> dict[Intent, SemiAutoChange]:
        """复核人员给演示店铺开半自动（§5.6、§13.4；走 rodit.app.rollout.set_semi_auto，写审计）：
        先按门槛开；门槛没过（合成历史每类每周只有几组推荐）时强制开启，并在审计里写明原因。"""
        rollout = self.service.rollout
        out: dict[Intent, SemiAutoChange] = {}
        for intent in SEMI_AUTO_INTENTS:
            try:
                change = rollout.set_semi_auto(
                    self.shop_id, intent, True, operator=REVIEWER, role="reviewer"
                )
            except SemiAutoGateNotPassed:
                change = rollout.set_semi_auto(
                    self.shop_id,
                    intent,
                    True,
                    operator=REVIEWER,
                    role="reviewer",
                    force=True,
                    note=SEMI_AUTO_NOTE,
                )
            out[intent] = change
        return out

    def _sync_platform_metrics(self) -> SyncResult | None:
        """系统从平台指标接口同步最近 8 周的客服考核数据（§12.4；合成数据，操作人 system）。
        单元没有指标接口或仓储时不同步（看板显示没有数据）。"""
        if self.unit.metrics_source is None or self.unit.repos.platform_metrics is None:
            return None
        return self.metrics_service().sync(self.shop_id, actor="system")

    def metrics_service(self) -> PlatformMetricsService:
        return PlatformMetricsService(self.unit, self.service.processing)

    def tick(self) -> datetime:
        """把单元时钟拨到真实的当前时间（只往前走，不早于 live_since）。"""
        now = max(self.clock.now(), self._real_clock.now(), self.live_since)
        self.clock.set(now)
        return now

    def _msg_id(self, prefix: str) -> str:
        return f"{prefix}-{self.platform.value}-{next(self._seq):06d}"

    def _sends(
        self,
        conversation_id: str,
        text: str,
        *,
        images: Sequence[str] = (),
        item_id: str | None,
        order_id: str | None,
        buyer: str,
        msg_prefix: str,
    ) -> list[Callable[[], EngineResult | None]]:
        """一条买家消息按平台的接入方式（§8.1）变成交给 AssistService 的调用：按顺序调用，最后一个
        的结果就是这条消息的处理结果。

        - 淘宝（模式 B 千牛插件事件）：文字是一条 text 消息；每张图片各是一条 image 消息
          （pic_key），买家写的字作为最后一张图的附带文字，前几张是平台的占位文字"[图片]"。
          同一时刻连发的几条由引擎合并成一轮处理（§5.3），前几条各自的推荐在最后一条到达时过期；
        - 抖店（模式 B 飞鸽插件事件）：一条消息，图片在 content.images（只给 uri），文字在
          content.text；图片消息的商品 / 订单取插件看到的当前会话（context）；
        - 拼多多（模式 C 话术助手）：只有文字（发图片在 chat 里就拒绝了）。"""
        at = self.clock.now()
        if self.platform == Platform.TAOBAO:

            def taobao(content: str, pic: str | None = None, earlier: int = 0) -> dict[str, Any]:
                sent = at - timedelta(milliseconds=CONSECUTIVE_GAP_MS * earlier)
                raw = taobao_event(
                    content,
                    msg_id=self._msg_id(msg_prefix),
                    session_id=conversation_id,
                    buyer=buyer,
                    item_id=item_id,
                    order_id=order_id,
                    seller_id=self.shop_id,
                    send_time_ms=int(sent.timestamp() * 1000),
                )
                if pic is not None:
                    raw.update(msg_type="image", pic_key=pic)
                return raw

            if images:
                last = len(images) - 1
                raws = [
                    taobao(IMAGE_PLACEHOLDER, ref, last - i) for i, ref in enumerate(images[:-1])
                ]
                raws.append(taobao(text or IMAGE_PLACEHOLDER, images[-1]))
            else:
                raws = [taobao(text)]
            return [
                partial(self.service.handle_event, "B", raw, expected_shop_id=self.shop_id)
                for raw in raws
            ]
        if self.platform == Platform.DOUYIN:
            content: dict[str, Any] = {"type": "text", "text": text}
            context: dict[str, Any] = {}
            if images:
                content = {"type": "image", "text": text, "images": [{"uri": r} for r in images]}
                card = context  # 图片消息不带卡片：用插件看到的当前商品 / 订单
            else:
                card = content
            if item_id:
                card["product_id"] = item_id
            if order_id:
                card["order_id"] = order_id
            raw: dict[str, Any] = {
                "shop_id": self.shop_id,
                "conversation_short_id": conversation_id,
                "server_message_id": self._msg_id(msg_prefix),
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
            if context:
                raw["context"] = context
            return [partial(self.service.handle_event, "B", raw, expected_shop_id=self.shop_id)]
        if images:  # chat 已经挡掉了；这里再兜一次，不把图片悄悄丢掉
            raise Invalid(NO_IMAGE_REASON)
        return [
            partial(
                self.service.ask_standalone,
                shop_id=self.shop_id,
                session_id=conversation_id,
                question=text,
                item_id=item_id,
                agent_id=AGENT_ID,
                request_id=self._msg_id(msg_prefix),
            )
        ]

    def _send_history(self, conversation_id: str, msg: HistoryMessage) -> EngineResult | None:
        [send] = self._sends(
            conversation_id,
            msg.text,
            item_id=msg.item_id,
            order_id=msg.order_id,
            buyer=msg.buyer,
            msg_prefix="hist",
        )
        return send()

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
            "images": [tile_view(r) for r in s.images],
            "intent": s.intent.value,
            "intent_label": intent_label(s.intent),
            "expect": s.expect_for(self.platform),
            "note": s.note,
            "card": self._card_view(s),
            "bubbles": list(conv.bubbles) if conv else [],
            "latest": conv.latest if conv else None,
        }

    def chat(
        self,
        scenario_id: str,
        text: str | None = None,
        images: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """买家在 scenario_id 这个会话里发一条消息，返回处理结果和过程。text、images 都为空时发
        场景预设的话和图片；给了任何一个就只发给的（只发图片、不带文字也可以）。图片只能是固定的
        合成标识，只有模式 B 的平台能发（拼多多模式 C 拒绝，§8.1）。"""
        scenario = get_scenario(self.platform, scenario_id)
        if scenario is None:
            raise NotFound(f"{self.profile.workbench}没有这个演示会话：{scenario_id}")
        if text is None and images is None:
            text, images = scenario.text_for(self.platform), scenario.images
        text = (text or "").strip()
        refs = list(images or ())
        if refs and self.platform not in IMAGE_PLATFORMS:
            raise Invalid(NO_IMAGE_REASON)
        if any(r not in DEMO_IMAGE_REFS for r in refs) or len(set(refs)) != len(refs):
            raise Invalid("只能发演示用的合成图片，同一张不能重复")
        if not text and not refs:
            raise Invalid("买家的话不能为空")
        with self.lock:
            now = self.tick()
            conv = self.conversations.get(scenario.id)
            if conv is None:
                conv = ConversationState(scenario, f"sc-{scenario.id}")
                self.conversations[scenario.id] = conv
            sends = self._sends(
                conv.conversation_id,
                text,
                images=refs,
                item_id=scenario.item_id,
                order_id=scenario.order_id,
                buyer=scenario.buyer,
                msg_prefix="live",
            )
            for send in sends[:-1]:
                send()  # 淘宝的多张图片：前几张各是一条图片消息，和最后一条合并成一轮（见 _sends）
            audit = self.unit.repos.audit
            decisions_before = len(audit.list_decisions(self.shop_id))
            events_before = len(audit.list_events(self.shop_id))
            scripted_before = self.llm.scripted
            with self.recorder.recording() as recording:
                result = sends[-1]()
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
                chain=self.chain,
            )
            buyer_bubble = {
                "role": "buyer",
                "text": mask_pii(text) if text else "",
                "images": [tile_view(r) for r in refs],
                "at": now.isoformat(),
            }
            view = {
                "epoch": self.epoch,
                "scenario": scenario.id,
                "message": buyer_bubble,
                "result": self._result_view(result),
                "trace": trace,
            }
            conv.bubbles.append(buyer_bubble)
            conv.latest = view
            return view

    def _input_prefill(self, conversation_id: str) -> int | None:
        """模式 B 插件这一刻填进客服输入框的推荐编号（适配器的 sidebar_payload.input_prefill，
        §5.6 半自动）；模式 C 没有平台输入框，返回 None。"""
        if self.profile.access not in PREFILL_ACCESS:
            return None
        adapter = self.unit.adapters.get(self.platform, self.profile.access)
        payload = adapter.sidebar_payload(self.shop_id, conversation_id)
        prefill = payload.input_prefill if payload is not None else None
        return prefill.suggestion_id if prefill is not None else None

    def _result_view(self, result: EngineResult) -> dict[str, Any]:
        route = result.route
        return {
            "mode": result.mode.value,
            "intent": route.intent.value if route else None,
            "intent_label": intent_label(route.intent) if route else None,
            "suggestions": [card_view(c) for c in result.suggestions],
            # 半自动（§5.6）：模式 B 插件把带 prefill 的那条填进客服输入框（input）；
            # 模式 C 没有平台输入框，只标"可直接发送"（copy）
            "prefill_mode": "input" if self.profile.access in PREFILL_ACCESS else "copy",
            "input_prefill": self._input_prefill(result.conversation_id),
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
        # R5 的凭证图片理解（§15 Phase 3）接在理由后面（aftersale.evidence_vision）：页面上单独显示
        reason_main, sep, vision_summary = (review.reason or "").partition("；" + SUMMARY_PREFIX)
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
            "evidence_images": [tile_view(r) for r in info.evidence_images] if info else [],
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
                "reason_main": reason_main if sep else review.reason,
                "vision_summary": vision_summary if sep else None,
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
        config: EngineConfig | None = None,
        judge: JudgeMode = "offline",
        latency: LatencyMode = "strict",
    ) -> None:
        """llm 是真实大模型客户端（为空时离线替身）；config 是 --config 读进来的引擎配置（为空用
        默认值）；judge / latency 见 showcase.realchain。"""
        if judge == "llm" and llm is None:
            raise ValueError("判断层用大模型投票（judge=llm）需要真实大模型（llm）")
        self.clock = clock or SystemClock()
        self.llm = llm
        self.history_seed = history_seed
        self.default_platform = default_platform
        self._factory = factory
        self.base_config = config
        self.judge: JudgeMode = judge
        self.latency: LatencyMode = latency
        self.chain = chain_info(
            llm, chain_config(config, judge=judge, latency=latency), judge=judge, latency=latency
        )
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
            config=self.base_config,
            judge=self.judge,
            latency=self.latency,
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
    "STALE_EPOCH_MESSAGE",
    "ShowcaseError",
    "ShowcaseHub",
    "StaleDemo",
    "demo_config",
    "minute_floor",
]
