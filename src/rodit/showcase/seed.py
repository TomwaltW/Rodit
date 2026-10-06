"""演示站的历史数据：把合成的买家问题按过去 5 周的时间回放一遍，再模拟客服的操作，
让数据看板一打开就有内容（全部合成，页面上标"演示数据（合成）"）。

- 问题来自评测集（rodit.eval.dataset，§13.1 合成问题）：只取淘宝演示店（SHOP_A）的、不依赖
  历史对话的用例；拼多多是模式 C，去掉带订单卡片的用例（话术助手查不到订单，§8.1）。另外每周插几条
  "明天能到吗"（§11.5），让质检拦截有数据。
- 时间：从 35 天前的零点开始，每天 9:00–22:30 之间，每周的量逐渐增加。第 1 周店铺在影子模式
  （§13.4 影子模式 1–2 周：生成但不展示），fixtures.DEMO_GO_LIVE_DAYS_AGO 天前的零点（和平台
  指标的"接入日"同一天）切到智能辅助。店铺配置经 KnowledgeRepository.update_shop_config 改（§9.3）。
- 每条消息都走真实的问答引擎（离线替身）：引擎写推荐记录、审计、判断日志，和线上一样。
- 客服操作按固定的随机种子模拟（同样的输入每次结果一样）：采纳概率每周上升（总体约七成，
  按误差累积分配，每周的采纳率紧跟概率），FAQ / 模板更容易被采纳，高风险、带风险提示的更容易
  被弃用；弃用原因按知识缺失、理解错误、表达不好、事实错误、其他的比例抽。操作走 AssistService
  的反馈接口（§5.8 反馈闭环：编辑后发送会生成候选知识、知识缺失会提醒商家补知识）。
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

from rodit.core.clock import CST
from rodit.core.enums import DraftKind, Intent, Mode, Platform, RejectReason
from rodit.core.models import EngineResult, SuggestionCard
from rodit.eval.dataset import cases
from rodit.testing.fixtures import DEMO_GO_LIVE_DAYS_AGO, RAW_BUYERS

HISTORY_DAYS = 35
WEEKLY_VOLUME: tuple[int, ...] = (30, 40, 46, 52, 58)
"""每周的买家消息数（第 1 周是影子模式）。"""


def history_start(now: datetime) -> datetime:
    """回放从哪天零点开始（东八区）：今天往前 HISTORY_DAYS 天，正好 5 周。"""
    day = now.astimezone(CST).date() - timedelta(days=HISTORY_DAYS)
    return datetime.combine(day, time(0, 0), tzinfo=CST)


def go_live_at(now: datetime) -> datetime:
    """切到智能辅助的时间：DEMO_GO_LIVE_DAYS_AGO 天前的零点（和平台指标按天划分的"接入日"一致）。"""
    day = now.astimezone(CST).date() - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)
    return datetime.combine(day, time(0, 0), tzinfo=CST)


ARRIVAL_QUESTIONS: dict[Platform, tuple[str, str | None]] = {
    Platform.TAOBAO: ("催一下，明天能到吗？", "o_1002"),
    Platform.DOUYIN: ("催一下，明天能到吗？", "o_1002"),
    Platform.PDD: ("明天能到吗？", None),
}
ARRIVAL_PER_WEEK = 2

ADOPT_BASE = 0.62
ADOPT_WEEKLY_GAIN = 0.04
AS_IS_SHARE = 0.72  # 采纳里原样发送的比例
REJECT_SHARE = 0.8  # 没采纳的里面点了"不合适"的比例（其余没操作）
REJECT_WEIGHTS: dict[RejectReason, float] = {
    RejectReason.MISSING_KNOWLEDGE: 0.34,
    RejectReason.WRONG_INTENT: 0.22,
    RejectReason.BAD_WORDING: 0.24,
    RejectReason.WRONG_FACT: 0.08,
    RejectReason.OTHER: 0.12,
}
_EDIT_TAILS = ("，有其他问题随时找我哦~", "，您看看还有什么需要的~", "，感谢您的耐心~")


@dataclass(frozen=True)
class HistoryMessage:
    at: datetime
    text: str
    item_id: str | None
    order_id: str | None
    buyer: str  # fixtures.RAW_BUYERS 的键
    expected_intent: Intent | None


@dataclass
class HistorySummary:
    """回放的结果（测试、启动日志用）。"""

    messages: int = 0
    suggestions: int = 0
    actions: Counter[str] = field(default_factory=Counter)
    shadow_messages: int = 0


def _eligible(platform: Platform) -> list[tuple[str, str | None, str | None, str, Intent]]:
    out = []
    for c in cases("all"):
        if c.shop != "SHOP_A" or c.history or RAW_BUYERS[c.buyer][0] != "taobao":
            continue
        if platform == Platform.PDD and c.order_id:
            continue
        out.append((c.text, c.item_id, c.order_id, c.buyer, c.expected_intent))
    return out


def history_plan(platform: Platform, now: datetime, *, seed: int) -> list[HistoryMessage]:
    """要回放的消息（按时间升序）。同样的平台、时间、种子，计划完全一样。"""
    rng = random.Random(seed)
    pool = _eligible(platform)
    rng.shuffle(pool)
    start = history_start(now).date()
    plan: list[HistoryMessage] = []
    picked = 0
    arrival_text, arrival_order = ARRIVAL_QUESTIONS[platform]
    for week, volume in enumerate(WEEKLY_VOLUME):
        times = []
        for _ in range(volume):
            day = start + timedelta(days=week * 7 + rng.randrange(7))
            minute = rng.randrange(9 * 60, 22 * 60 + 30)
            times.append(
                datetime.combine(day, time(minute // 60, minute % 60), tzinfo=CST)
                + timedelta(seconds=rng.randrange(60))
            )
        times.sort()
        arrival_slots = set(rng.sample(range(volume), ARRIVAL_PER_WEEK))
        for i, at in enumerate(times):
            if at >= now:
                continue
            if i in arrival_slots:
                plan.append(
                    HistoryMessage(at, arrival_text, None, arrival_order, "BUYER", Intent.URGE_SHIP)
                )
                continue
            text, item_id, order_id, buyer, intent = pool[picked % len(pool)]
            picked += 1
            plan.append(HistoryMessage(at, text, item_id, order_id, buyer, intent))
    plan.sort(key=lambda m: m.at)
    return plan


def _groups(result: EngineResult) -> list[SuggestionCard]:
    """每组推荐（一个问题类型一组）的第一张卡片：客服对一组只操作一次。"""
    first: dict[Intent, SuggestionCard] = {}
    for card in result.suggestions:
        if card.suggestion_id is not None and card.intent not in first:
            first[card.intent] = card
    return list(first.values())


def _adopt_probability(card: SuggestionCard, assist_week: int) -> float:
    p = ADOPT_BASE + ADOPT_WEEKLY_GAIN * assist_week
    if card.kind in (DraftKind.FAQ, DraftKind.TEMPLATE):
        p += 0.08
    if card.risk_hints:
        p -= 0.12
    if card.intent == Intent.OTHER:
        p -= 0.2
    return min(0.95, max(0.2, p))


@dataclass
class Feedback:
    """模拟的一次客服操作。"""

    suggestion_id: int
    action: str  # sent / edited / rejected / ignored
    text: str | None = None
    reason: RejectReason | None = None
    true_intent: Intent | None = None


class AdoptionQuota:
    """按概率决定采纳与否，但用误差累积代替掷骰子：一段时间里采纳的比例紧跟概率，
    每周的采纳率不会因为样本少而大起大落（同样的输入结果完全一样）。"""

    def __init__(self) -> None:
        self._carry = 0.5

    def take(self, p: float) -> bool:
        self._carry += p
        if self._carry >= 1.0:
            self._carry -= 1.0
            return True
        return False


def simulate_feedback(
    result: EngineResult,
    msg: HistoryMessage,
    *,
    assist_week: int,
    rng: random.Random,
    quota: AdoptionQuota,
) -> list[Feedback]:
    out: list[Feedback] = []
    for card in _groups(result):
        assert card.suggestion_id is not None
        if quota.take(_adopt_probability(card, assist_week)):
            if rng.random() < AS_IS_SHARE:
                out.append(Feedback(card.suggestion_id, "sent"))
            else:
                tail = _EDIT_TAILS[rng.randrange(len(_EDIT_TAILS))]
                text = card.text.rstrip("。！!~～ ") + tail
                out.append(Feedback(card.suggestion_id, "edited", text=text))
        elif rng.random() < REJECT_SHARE:
            reasons, weights = zip(*REJECT_WEIGHTS.items(), strict=True)
            reason = rng.choices(reasons, weights=weights)[0]
            true_intent = None
            if reason == RejectReason.WRONG_INTENT and msg.expected_intent != card.intent:
                true_intent = msg.expected_intent
            out.append(
                Feedback(card.suggestion_id, "rejected", reason=reason, true_intent=true_intent)
            )
        else:
            out.append(Feedback(card.suggestion_id, "ignored"))
    return out


def replay_history(
    *,
    platform: Platform,
    now: datetime,
    seed: int,
    set_clock: Callable[[datetime], None],
    set_shadow: Callable[[bool], None],
    send: Callable[[str, HistoryMessage], EngineResult | None],
    apply: Callable[[Feedback], None],
) -> HistorySummary:
    """按计划回放（调用方提供发消息、写反馈、拨时钟、切影子模式的方法，见 showcase.state）。"""
    rng = random.Random(seed + 1)
    quota = AdoptionQuota()
    summary = HistorySummary()
    go_live = go_live_at(now)
    plan = history_plan(platform, now, seed=seed)
    shadow = True
    set_shadow(True)
    start = history_start(now)
    cursor = start

    def advance(to: datetime) -> None:  # 时钟只往前走
        nonlocal cursor
        cursor = max(cursor, to)
        set_clock(cursor)

    for n, msg in enumerate(plan):
        if shadow and msg.at >= go_live:
            shadow = False
            advance(go_live)
            set_shadow(False)
        advance(msg.at)
        result = send(f"h{n:04d}", msg)
        summary.messages += 1
        if result is None:
            continue
        if result.mode == Mode.SHADOW:
            summary.shadow_messages += 1
            continue
        summary.suggestions += len(result.suggestions)
        assist_week = max(0, (msg.at - start).days // 7 - 1)
        for fb in simulate_feedback(result, msg, assist_week=assist_week, rng=rng, quota=quota):
            summary.actions[fb.action] += 1
            if fb.action == "ignored":
                continue
            advance(msg.at + timedelta(seconds=rng.randrange(4, 75)))
            apply(fb)
    if shadow:
        set_shadow(False)
    set_clock(max(cursor, now))
    return summary


__all__ = [
    "ARRIVAL_QUESTIONS",
    "AdoptionQuota",
    "go_live_at",
    "history_start",
    "HISTORY_DAYS",
    "WEEKLY_VOLUME",
    "Feedback",
    "HistoryMessage",
    "HistorySummary",
    "history_plan",
    "replay_history",
    "simulate_feedback",
]
