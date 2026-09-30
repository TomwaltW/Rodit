"""采纳率、质检、影子对比报表与上线阶段门槛（方案 §1.3 指标、§12.4 可观测性、§13.4 上线节奏）。

口径（报表里的每个数都能按这里复算）：

- 推荐条数：SuggestionRecord 的条数，不含 SHADOW（影子模式没有展示，单独用 shadow_report），
  也不含全自动发出去的（agent_id="bot"，见 is_auto_sent）：采纳率是客服原样 / 小改后发送的比例
  （§1.3），全自动的回复单独列在 AdoptionReport.auto；有一张卡片被全自动发出去的推荐组整组算全自动。
  adoption_rate =（原样发送 + 编辑后发送）/ 推荐条数；as_is_rate = 原样发送 / 推荐条数；
  还没操作的（SHOWN）和被刷新掉的（EXPIRED）都算在分母里。
- 推荐组：同一条买家消息、同一个问题类型的推荐卡片（引擎一次给 1–3 条，客服最多发一条）。
  group_adoption_rate = 有卡片被发送（原样或编辑后）的组 / 组数。§1.3 "推荐采纳率 ≥ 60%"
  和 §13.4 的门槛按推荐组算：按卡片算时，一次给两条的店铺采纳率最高只有 50%。
- 问题类型占比 share：该类型的推荐组 / 全部推荐组（§12.4 各类问题的占比）。
- 耗时：SuggestionRecord.latency_ms，按推荐组取一个值（组内最大），分位数用最近秩法
  （p 分位 = 升序第 ⌈p·n⌉ 个）。整体耗时按买家消息去重（多问题分条的几组只算一次）。
- 质检报表读引擎每条消息写的 engine.result 审计事件，detail 的键见 RESULT_* 常量。
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.core import audit as _audit
from rodit.core.audit import AuditAction, ResultKey
from rodit.core.enums import INTENT_LABELS, Intent, RejectReason, Role, SuggestionStatus
from rodit.core.models import NON_BLOCKING_HARD_RULES, StoredMessage, SuggestionRecord
from rodit.core.ports import AuditLog

REJECT_REASON_LABELS: dict[RejectReason, str] = {
    RejectReason.MISSING_KNOWLEDGE: "知识缺失",
    RejectReason.WRONG_INTENT: "理解错误",
    RejectReason.BAD_WORDING: "表达不好",
    RejectReason.WRONG_FACT: "事实错误",
    RejectReason.OTHER: "其他",
}
"""弃用原因的中文名（§5.8、§12.4 弃用原因分布）。"""

E2E_P95_TARGET_MS = 5000  # 推荐出现（端到端）P95 ≤ 5 秒（§1.3、§12.1）

AUTO_AGENT_ID = Role.BOT.value
"""全自动发送成功的推荐记录的 agent_id：rodit.app.service 发送成功后把推荐记为 SENT、
agent_id="bot"。"""

_ADOPTED = frozenset({SuggestionStatus.SENT, SuggestionStatus.EDITED_SENT})


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ================================================================ 通用


def percentile(values: Sequence[float], q: float) -> float | None:
    """最近秩法分位数：升序第 ⌈q·n⌉ 个（至少第 1 个）；没有数据返回 None。"""
    if not 0 < q <= 1:
        raise ValueError(f"分位数必须在 (0, 1] 之间：{q}")
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _normalize(text: str) -> str:
    """NFKC 归一，去掉空白、标点、符号（~、emoji 等），只留文字和数字。"""
    norm = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in norm if unicodedata.category(ch)[0] in {"L", "N"})


def _grams(text: str, n: int) -> Counter[str]:
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def char_similarity(a: str, b: str) -> float:
    """字符重合度 ∈ [0, 1]：去掉空白和标点后，按字符二元组算 Dice 系数
    2·|A∩B| / (|A|+|B|)（多重集）；有一方只有一个字时退回单字。两边都为空算 1.0。"""
    x, y = _normalize(a), _normalize(b)
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    n = 2 if min(len(x), len(y)) >= 2 else 1
    ga, gb = _grams(x, n), _grams(y, n)
    overlap = sum((ga & gb).values())
    return 2 * overlap / (sum(ga.values()) + sum(gb.values()))


class LatencyStats(_Model):
    count: int = 0
    p50: float | None = None
    p95: float | None = None

    @classmethod
    def of(cls, values: Sequence[float]) -> LatencyStats:
        return cls(count=len(values), p50=percentile(values, 0.5), p95=percentile(values, 0.95))


# ================================================================ 采纳率（§1.3、§12.4）


class IntentStats(_Model):
    """一个问题类型（intent=None 表示全部）的采纳情况。rate 在分母为 0 时为 None。"""

    intent: Intent | None = None
    label: str
    total: int = 0  # 推荐条数（不含 SHADOW）
    shown: int = 0  # 还没操作
    sent: int = 0
    edited_sent: int = 0
    rejected: int = 0
    expired: int = 0
    adoption_rate: float | None = None
    as_is_rate: float | None = None
    edited_rate: float | None = None
    reject_rate: float | None = None
    expired_rate: float | None = None
    groups: int = 0  # 推荐组数
    adopted_groups: int = 0
    as_is_groups: int = 0
    group_adoption_rate: float | None = None
    group_as_is_rate: float | None = None
    share: float | None = None  # 该类型推荐组占全部推荐组的比例
    reject_reasons: dict[RejectReason, int] = Field(default_factory=dict)
    latency_ms: LatencyStats = Field(default_factory=LatencyStats)


class AutoReplyStats(_Model):
    """全自动发出去的回复（agent_id="bot"）：不算进采纳率，单独一行（§13.4 全自动按类型监控）。"""

    sent: int = 0  # 全自动发出去的推荐条数
    groups: int = 0  # 涉及的推荐组数（整组算全自动）
    by_intent: dict[Intent, int] = Field(default_factory=dict)  # 各类型的推荐组数
    latency_ms: LatencyStats = Field(default_factory=LatencyStats)  # 按买家消息去重


class AdoptionReport(_Model):
    """推荐采纳率报表（§12.4 给商家看的业务指标）。

    overall / by_intent 只含客服看到的推荐（不含影子模式、不含全自动发出去的推荐组）；
    by_intent 按 Intent 定义顺序，只含有数据的类型。全自动发出去的在 auto。
    """

    since: AwareDatetime | None = None
    until: AwareDatetime | None = None
    shop_ids: list[str] = Field(default_factory=list)
    shadow_excluded: int = 0
    overall: IntentStats
    by_intent: dict[Intent, IntentStats] = Field(default_factory=dict)
    auto: AutoReplyStats = Field(default_factory=AutoReplyStats)


def _group_key(r: SuggestionRecord) -> tuple[str, str, str, str]:
    return (r.shop_id, r.conversation_id, r.msg_id or f"#{r.id}", r.intent.value)


def is_auto_sent(r: SuggestionRecord) -> bool:
    """全自动发出去的推荐：SENT 且 agent_id 是 "bot"。全自动没发出去的仍是 SHOWN（留给客服）。"""
    return r.status == SuggestionStatus.SENT and r.agent_id == AUTO_AGENT_ID


def _message_latency(records: Iterable[SuggestionRecord]) -> list[float]:
    """按买家消息去重的耗时（多问题分条的几组共用一次处理耗时，取最大）。"""
    by_message: dict[tuple[str, str, str], list[int]] = {}
    for r in records:
        if r.latency_ms is not None:
            key = (r.shop_id, r.conversation_id, r.msg_id or f"#{r.id}")
            by_message.setdefault(key, []).append(r.latency_ms)
    return [float(max(v)) for v in by_message.values()]


def _stats(
    intent: Intent | None,
    records: list[SuggestionRecord],
    all_groups: int,
    message_latency: list[float] | None = None,
) -> IntentStats:
    counts = Counter(r.status for r in records)
    total = len(records)
    groups: dict[tuple[str, str, str, str], list[SuggestionRecord]] = {}
    for r in records:
        groups.setdefault(_group_key(r), []).append(r)
    adopted = sum(1 for g in groups.values() if any(r.status in _ADOPTED for r in g))
    as_is = sum(1 for g in groups.values() if any(r.status == SuggestionStatus.SENT for r in g))
    reasons = Counter(
        r.reject_reason or RejectReason.OTHER
        for r in records
        if r.status == SuggestionStatus.REJECTED
    )
    if message_latency is None:
        message_latency = [
            float(max(lat))
            for g in groups.values()
            if (lat := [r.latency_ms for r in g if r.latency_ms is not None])
        ]
    sent, edited = counts[SuggestionStatus.SENT], counts[SuggestionStatus.EDITED_SENT]
    return IntentStats(
        intent=intent,
        label=INTENT_LABELS[intent] if intent else "全部",
        total=total,
        shown=counts[SuggestionStatus.SHOWN],
        sent=sent,
        edited_sent=edited,
        rejected=counts[SuggestionStatus.REJECTED],
        expired=counts[SuggestionStatus.EXPIRED],
        adoption_rate=_rate(sent + edited, total),
        as_is_rate=_rate(sent, total),
        edited_rate=_rate(edited, total),
        reject_rate=_rate(counts[SuggestionStatus.REJECTED], total),
        expired_rate=_rate(counts[SuggestionStatus.EXPIRED], total),
        groups=len(groups),
        adopted_groups=adopted,
        as_is_groups=as_is,
        group_adoption_rate=_rate(adopted, len(groups)),
        group_as_is_rate=_rate(as_is, len(groups)),
        share=_rate(len(groups), all_groups),
        reject_reasons={reason: reasons.get(reason, 0) for reason in RejectReason},
        latency_ms=LatencyStats.of(message_latency),
    )


def adoption_report(
    records: Iterable[SuggestionRecord],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> AdoptionReport:
    """按问题类型和整体统计采纳情况。since 含、until 不含（按 created_at）。

    records 一般是 SuggestionRepository.list(shop_id, since=...) 的结果；给商家看的报表只传
    一个店铺的记录（shop_ids 会列出实际包含的店铺）。
    """
    picked = [
        r
        for r in records
        if (since is None or r.created_at >= since) and (until is None or r.created_at < until)
    ]
    visible = [r for r in picked if r.status != SuggestionStatus.SHADOW]
    # 全自动发出去的推荐组整组拿出来（组里其余卡片客服没看到，不压低客服的采纳率）
    auto_keys = {_group_key(r) for r in visible if is_auto_sent(r)}
    auto = [r for r in visible if _group_key(r) in auto_keys]
    shown = [r for r in visible if _group_key(r) not in auto_keys]
    all_groups = len({_group_key(r) for r in shown})
    # 整体耗时按买家消息去重：多问题分条（规则 5）的几组共用一次处理耗时
    overall = _stats(None, shown, all_groups, _message_latency(shown))
    by_intent: dict[Intent, IntentStats] = {}
    for intent in Intent:
        subset = [r for r in shown if r.intent == intent]
        if subset:
            by_intent[intent] = _stats(intent, subset, all_groups)
    auto_by_intent = Counter(key[3] for key in auto_keys)
    return AdoptionReport(
        since=since,
        until=until,
        shop_ids=sorted({r.shop_id for r in picked}),
        shadow_excluded=len(picked) - len(visible),
        overall=overall,
        by_intent=by_intent,
        auto=AutoReplyStats(
            sent=sum(1 for r in auto if is_auto_sent(r)),
            groups=len(auto_keys),
            by_intent={i: auto_by_intent[i.value] for i in Intent if auto_by_intent[i.value]},
            latency_ms=LatencyStats.of(_message_latency(r for r in auto if is_auto_sent(r))),
        ),
    )


# ================================================================ 质检与降级（§12.4 我们自己看）

# engine.result 审计事件 detail 的键：定义在 core.audit.ResultKey（引擎写、报表读同一份）
RESULT_DETAIL_KEYS = _audit.RESULT_DETAIL_KEYS  # 定义在 core.audit.ResultKey
RESULT_INTENT = ResultKey.INTENT.value
RESULT_ACTION = ResultKey.ACTION.value
RESULT_SUGGESTIONS = ResultKey.SUGGESTIONS.value
RESULT_NO_SUGGESTION = ResultKey.NO_SUGGESTION.value
RESULT_QA_CHECKED = ResultKey.QA_CHECKED.value
RESULT_QA_FAILED = ResultKey.QA_FAILED.value
RESULT_QA_REASONS = ResultKey.QA_REASONS.value
RESULT_HARD_HITS = ResultKey.HARD_HITS.value
RESULT_DEGRADED = ResultKey.DEGRADED.value
RESULT_TRANSFER = ResultKey.TRANSFER.value
RESULT_LATENCY = ResultKey.LATENCY.value
LATENCY_TOTAL = "total"


class QAReport(_Model):
    """质检拦截率、硬规则命中、降级、各环节耗时（§12.4）。

    - interception_rate：被拦截的草稿 / 质检次数；
    - message_interception_rate：有拦截（质检不通过或命中拦截类硬规则）的消息 / 跑过质检的消息；
    - hard_hit_counts 含自动修正类（format / length），blocking_hit_counts 不含。
    """

    shop_id: str
    since: AwareDatetime | None = None
    until: AwareDatetime | None = None
    messages: int = 0
    qa_checked: int = 0
    qa_failed: int = 0
    interception_rate: float | None = None
    messages_checked: int = 0
    messages_intercepted: int = 0
    message_interception_rate: float | None = None
    hard_hit_counts: dict[str, int] = Field(default_factory=dict)
    blocking_hit_counts: dict[str, int] = Field(default_factory=dict)
    qa_reason_counts: dict[str, int] = Field(default_factory=dict)
    degraded_messages: int = 0
    degraded_rate: float | None = None
    degraded_counts: dict[str, int] = Field(default_factory=dict)
    action_counts: dict[str, int] = Field(default_factory=dict)
    intent_counts: dict[str, int] = Field(default_factory=dict)
    no_suggestion: int = 0
    no_suggestion_reasons: dict[str, int] = Field(default_factory=dict)
    transfers: int = 0
    latency_ms: dict[str, LatencyStats] = Field(default_factory=dict)
    p95_within_target: bool | None = None  # 端到端 P95 ≤ 5 秒（§12.1）


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def _str_list(value: Any, *, key: str | None = None) -> list[str]:
    """list[str]；也接受单个字符串，或 list[dict]（取 key 字段，如 HardRuleHit.model_dump()
    的 "rule"）。其他类型当作空。"""
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list | tuple):
        return []
    out: list[str] = []
    for v in value:
        if isinstance(v, str):
            out.append(v)
        elif key and isinstance(v, Mapping) and isinstance(v.get(key), str):
            out.append(v[key])
    return out


def _latencies(value: Any) -> dict[str, float]:
    """dict[str, 数字]；单个数字当作端到端耗时。"""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return {LATENCY_TOTAL: float(value)}
    if not isinstance(value, Mapping):
        return {}
    return {
        str(k): float(v)
        for k, v in value.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }


def qa_report(
    audit: AuditLog,
    shop_id: str,
    since: datetime | None = None,
    *,
    until: datetime | None = None,
) -> QAReport:
    """从 engine.result 事件汇总（since 含、until 不含）。缺少的键按 0 / 空处理。"""
    events = [
        e
        for e in audit.list_events(shop_id, action=AuditAction.RESULT.value, since=since)
        if until is None or e.created_at < until
    ]
    hard, blocking, reasons = Counter[str](), Counter[str](), Counter[str]()
    degraded, actions, intents, no_sug_reasons = (
        Counter[str](),
        Counter[str](),
        Counter[str](),
        Counter[str](),
    )
    latencies: dict[str, list[float]] = {}
    qa_checked = qa_failed = checked_msgs = intercepted_msgs = degraded_msgs = 0
    no_suggestion = transfers = 0
    for e in events:
        d = e.detail
        checked, failed = _as_int(d.get(RESULT_QA_CHECKED)), _as_int(d.get(RESULT_QA_FAILED))
        hits = _str_list(d.get(RESULT_HARD_HITS), key="rule")
        blocking_hits = [h for h in hits if h not in NON_BLOCKING_HARD_RULES]
        qa_checked += checked
        qa_failed += failed
        hard.update(hits)
        blocking.update(blocking_hits)
        reasons.update(_str_list(d.get(RESULT_QA_REASONS)))
        if checked or hits:
            checked_msgs += 1
            if failed or blocking_hits:
                intercepted_msgs += 1
        deg = d.get(RESULT_DEGRADED)
        deg_list = ["unknown"] if deg is True else _str_list(deg)
        if deg_list:
            degraded_msgs += 1
            degraded.update(deg_list)
        if isinstance(d.get(RESULT_ACTION), str):
            actions[d[RESULT_ACTION]] += 1
        if isinstance(d.get(RESULT_INTENT), str):
            intents[d[RESULT_INTENT]] += 1
        no_sug = d.get(RESULT_NO_SUGGESTION)
        if isinstance(no_sug, str) and no_sug:
            no_suggestion += 1
            no_sug_reasons[no_sug] += 1
        elif RESULT_SUGGESTIONS in d and _as_int(d.get(RESULT_SUGGESTIONS)) == 0:
            no_suggestion += 1
        if d.get(RESULT_TRANSFER) is True:
            transfers += 1
        for stage, ms in _latencies(d.get(RESULT_LATENCY)).items():
            latencies.setdefault(stage, []).append(ms)
    latency = {stage: LatencyStats.of(v) for stage, v in sorted(latencies.items())}
    total_p95 = latency[LATENCY_TOTAL].p95 if LATENCY_TOTAL in latency else None
    return QAReport(
        shop_id=shop_id,
        since=since,
        until=until,
        messages=len(events),
        qa_checked=qa_checked,
        qa_failed=qa_failed,
        interception_rate=_rate(qa_failed, qa_checked),
        messages_checked=checked_msgs,
        messages_intercepted=intercepted_msgs,
        message_interception_rate=_rate(intercepted_msgs, checked_msgs),
        hard_hit_counts=dict(hard.most_common()),
        blocking_hit_counts=dict(blocking.most_common()),
        qa_reason_counts=dict(reasons.most_common()),
        degraded_messages=degraded_msgs,
        degraded_rate=_rate(degraded_msgs, len(events)),
        degraded_counts=dict(degraded.most_common()),
        action_counts=dict(actions.most_common()),
        intent_counts=dict(intents.most_common()),
        no_suggestion=no_suggestion,
        no_suggestion_reasons=dict(no_sug_reasons.most_common()),
        transfers=transfers,
        latency_ms=latency,
        p95_within_target=None if total_p95 is None else total_p95 <= E2E_P95_TARGET_MS,
    )


# ================================================================ 影子对比（§13.4 第一阶段）


class ShadowItem(_Model):
    """一个影子推荐组的对比结果（组内取最像客服实际回复的一条）。不含任何文字。"""

    suggestion_id: int | None = None
    msg_id: str | None = None
    intent: Intent
    similarity: float | None = None  # 客服没有回复时为 None


class ShadowIntentStats(_Model):
    groups: int = 0
    compared: int = 0
    mean_similarity: float | None = None
    match_rate: float | None = None


class ShadowReport(_Model):
    """影子模式：生成但不展示，事后和客服的实际回复对比（§13.4）。

    字符重合度只是粗略参考，§13.4 进入智能辅助的门槛（§13.2 事实正确、合规满分率 ≥ 95%）
    仍要人工评分；这里帮评审挑出差异大的样本。
    """

    groups: int = 0
    compared: int = 0
    no_reply: int = 0
    match_threshold: float
    mean_similarity: float | None = None
    median_similarity: float | None = None
    match_rate: float | None = None  # 相似度 ≥ match_threshold 的比例（在有回复的组里）
    by_intent: dict[Intent, ShadowIntentStats] = Field(default_factory=dict)
    items: list[ShadowItem] = Field(default_factory=list)


def agent_replies_by_msg(messages: Iterable[StoredMessage]) -> dict[str, list[str]]:
    """把客服的实际回复对到触发它的买家消息上（shadow_report 的 sent_texts）。

    messages 是同一会话按时间升序的消息（ConversationRepository.recent_messages 或
    PlatformAdapter.get_context 的 recent_messages）。连发的几条买家消息算一轮，之后到下一条买家
    消息之前的客服消息（role=agent；系统自动回复 bot 不算）都算这一轮每条买家消息的回复。
    """
    out: dict[str, list[str]] = {}
    run: list[str] = []
    replied = False
    for m in messages:
        if m.role == Role.BUYER:
            if replied:
                run, replied = [], False
            run.append(m.msg_id)
            out.setdefault(m.msg_id, [])
        elif m.role == Role.AGENT and m.text and run:
            for msg_id in run:
                out[msg_id].append(m.text)
            replied = True
    return out


def _best_similarity(text: str, replies: Sequence[str]) -> float | None:
    replies = [r for r in replies if r and r.strip()]
    if not replies:
        return None
    candidates = [*replies, "".join(replies)] if len(replies) > 1 else replies
    return max(char_similarity(text, r) for r in candidates)


def shadow_report(
    shadow_records: Iterable[SuggestionRecord],
    sent_texts: Mapping[str, str | Sequence[str]],
    *,
    match_threshold: float = 0.5,
) -> ShadowReport:
    """sent_texts：触发推荐的买家 msg_id → 客服实际发出的文字（一条或多条，多条时也和拼起来的
    整段比较）。只统计 SHADOW 记录；同一推荐组（同一消息、同一类型）取最高相似度。
    msg_id 在一个平台内唯一，按店铺分别调用。"""
    groups: dict[tuple[str, str, str, str], list[SuggestionRecord]] = {}
    for r in shadow_records:
        if r.status == SuggestionStatus.SHADOW:
            groups.setdefault(_group_key(r), []).append(r)
    items: list[ShadowItem] = []
    for records in groups.values():
        first = records[0]
        raw = sent_texts.get(first.msg_id or "") if first.msg_id else None
        replies = [raw] if isinstance(raw, str) else list(raw or [])
        best: tuple[float, SuggestionRecord] | None = None
        for r in records:
            score = _best_similarity(r.text, replies)
            if score is not None and (best is None or score > best[0]):
                best = (score, r)
        chosen = best[1] if best else first
        items.append(
            ShadowItem(
                suggestion_id=chosen.id,
                msg_id=chosen.msg_id,
                intent=chosen.intent,
                similarity=round(best[0], 4) if best else None,
            )
        )
    scores = [i.similarity for i in items if i.similarity is not None]
    by_intent: dict[Intent, ShadowIntentStats] = {}
    for intent in Intent:
        subset = [i for i in items if i.intent == intent]
        if not subset:
            continue
        s = [i.similarity for i in subset if i.similarity is not None]
        by_intent[intent] = ShadowIntentStats(
            groups=len(subset),
            compared=len(s),
            mean_similarity=sum(s) / len(s) if s else None,
            match_rate=_rate(sum(1 for x in s if x >= match_threshold), len(s)),
        )
    return ShadowReport(
        groups=len(items),
        compared=len(scores),
        no_reply=len(items) - len(scores),
        match_threshold=match_threshold,
        mean_similarity=sum(scores) / len(scores) if scores else None,
        median_similarity=percentile(scores, 0.5),
        match_rate=_rate(sum(1 for x in scores if x >= match_threshold), len(scores)),
        by_intent=by_intent,
        items=items,
    )


# ================================================================ 上线阶段门槛（§13.4）


class GateCheck(_Model):
    name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    detail: str = ""


class GateResult(_Model):
    """passed = 所有已检查的条件都通过；unchecked 列出没有数据、没法判断的条件（需要人工确认）。"""

    stage: str
    passed: bool
    checks: list[GateCheck] = Field(default_factory=list)
    unchecked: list[str] = Field(default_factory=list)


def _as_periods(reports: AdoptionReport | Sequence[AdoptionReport]) -> list[AdoptionReport]:
    return [reports] if isinstance(reports, AdoptionReport) else list(reports)


def _period_checks(
    periods: list[AdoptionReport], *, min_periods: int
) -> tuple[list[GateCheck], list[str]]:
    checks = [
        GateCheck(
            name="periods",
            passed=len(periods) >= min_periods,
            value=len(periods),
            threshold=min_periods,
            detail=f"需要连续 {min_periods} 个统计周期",
        )
    ]
    unchecked: list[str] = []
    if len(periods) >= 2:
        if all(p.since is not None and p.until is not None for p in periods):
            gaps = [i for i in range(len(periods) - 1) if periods[i].until != periods[i + 1].since]
            checks.append(
                GateCheck(
                    name="consecutive",
                    passed=not gaps,
                    detail="周期首尾相接" if not gaps else f"第 {gaps[0] + 1} 个周期之后不连续",
                )
            )
        else:
            unchecked.append("consecutive")
    return checks, unchecked


def _gate(
    stage: str,
    periods: list[AdoptionReport],
    pick: Callable[[AdoptionReport], IntentStats | None],
    rate_name: str,
    threshold: float,
    *,
    min_periods: int,
    min_samples: int,
    zero_counts: Mapping[str, int | None],
) -> GateResult:
    checks, unchecked = _period_checks(periods, min_periods=min_periods)
    for i, report in enumerate(periods, start=1):
        stats: IntentStats | None = pick(report)
        groups = stats.groups if stats else 0
        rate = getattr(stats, rate_name) if stats else None
        enough = groups >= max(1, min_samples)
        checks.append(
            GateCheck(
                name=f"period_{i}.{rate_name}",
                passed=enough and rate is not None and rate >= threshold,
                value=rate,
                threshold=threshold,
                detail=f"推荐组 {groups}（至少 {max(1, min_samples)}）",
            )
        )
    for name, count in zero_counts.items():
        if count is None:
            unchecked.append(name)
        else:
            checks.append(GateCheck(name=name, passed=count == 0, value=count, threshold=0))
    return GateResult(
        stage=stage,
        passed=bool(periods) and all(c.passed for c in checks),
        checks=checks,
        unchecked=unchecked,
    )


def phase_gate(
    reports: AdoptionReport | Sequence[AdoptionReport],
    *,
    min_adoption: float = 0.6,
    min_periods: int = 1,
    min_samples: int = 1,
    platform_violations: int | None = None,
) -> GateResult:
    """智能辅助阶段能否进入下一阶段（§13.4："连续 2 周采纳率 ≥ 60%，平台违规为 0"）。

    reports 是按时间顺序、首尾相接的周期报表（如每周一份）；§13.4 的判断用
    phase_gate(weekly, min_periods=2, platform_violations=n)。采纳率按推荐组算
    （group_adoption_rate）。platform_violations 不传时该条件列入 unchecked。
    Phase 0 验收（§15 模式 C 采纳率 ≥ 50%）用 min_adoption=0.5。
    """
    return _gate(
        "assist",
        _as_periods(reports),
        lambda r: r.overall,
        "group_adoption_rate",
        min_adoption,
        min_periods=min_periods,
        min_samples=min_samples,
        zero_counts={"platform_violations": platform_violations},
    )


def intent_gate(
    reports: AdoptionReport | Sequence[AdoptionReport],
    intent: Intent,
    *,
    min_as_is: float = 0.9,
    min_periods: int = 1,
    min_samples: int = 1,
    complaints: int | None = None,
) -> GateResult:
    """某个问题类型能否开半自动 / 全自动（§13.4）：按推荐组算的原样采纳率 ≥ 90%。

    半自动："该类型原样采纳率 ≥ 90%"；全自动："连续 4 周原样采纳率 ≥ 90%，且没有因此产生的
    投诉"，用 intent_gate(weekly, intent, min_periods=4, complaints=n)。
    """
    return _gate(
        f"intent:{Intent(intent).value}",
        _as_periods(reports),
        lambda r: r.by_intent.get(Intent(intent)),
        "group_as_is_rate",
        min_as_is,
        min_periods=min_periods,
        min_samples=min_samples,
        zero_counts={"complaints": complaints},
    )
