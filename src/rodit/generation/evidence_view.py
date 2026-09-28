"""按 core.evidence 约定的 data 结构读取证据（方案 §5.2、§7.2、§7.3.1）。

证据 ID 的前缀用 core.evidence 的构造函数算出来，不手拼字符串；data 里的模型对象用对应模型
model_validate，结构不对就当没有这条证据（交给大模型或别的模板），不抛异常。
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from rodit.core import evidence as E
from rodit.core.enums import EvidenceSource, Platform, PolicyKey
from rodit.core.models import (
    AfterSaleInfo,
    ConversationContext,
    Evidence,
    EvidenceBundle,
    LogisticsAnalysis,
    RefundProgress,
)

_LOGISTICS_PREFIX = E.tool_logistics("")
_AFTERSALE_PREFIX = E.tool_aftersale("")
_FAQ_PREFIX = E.faq(0)[: -len("0")]


def text_of(ev: Evidence, *keys: str) -> str:
    """证据的文字：优先取 data 里的 keys（如 value / text / answer），没有就用 content。"""
    for key in keys:
        value = ev.data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ev.content.strip()


def _validate(model: Any, raw: object) -> Any | None:
    if raw is None:
        return None
    try:
        return model.model_validate(raw)
    except ValidationError:
        return None


def policy_evidence(bundle: EvidenceBundle, key: PolicyKey | str) -> Evidence | None:
    return bundle.get(E.policy(str(key)))


def service_hours(bundle: EvidenceBundle) -> Evidence | None:
    return policy_evidence(bundle, PolicyKey.SERVICE_HOURS)


def platform_rule(
    bundle: EvidenceBundle, platform: Platform, key: E.PlatformRuleKey
) -> Evidence | None:
    return bundle.get(E.platform_rule(platform, key))


def logistics(
    bundle: EvidenceBundle, ctx: ConversationContext
) -> tuple[Evidence, LogisticsAnalysis] | None:
    """物流证据和分析结果。有多条时优先当前订单的，否则取第一条。"""
    found: list[tuple[Evidence, LogisticsAnalysis]] = []
    for ev in bundle.evidences:
        if not ev.id.startswith(_LOGISTICS_PREFIX):
            continue
        analysis = _validate(LogisticsAnalysis, ev.data.get("analysis"))
        if analysis is not None:
            found.append((ev, analysis))
    if not found:
        return None
    if ctx.order is not None:
        current = E.tool_logistics(ctx.order.order_id)
        for pair in found:
            if pair[0].id == current:
                return pair
    return found[0]


def refund_progress(
    bundle: EvidenceBundle, ctx: ConversationContext
) -> tuple[Evidence, RefundProgress] | None:
    """售后单证据和退款进度。有多条时优先当前订单的，其次最近更新的售后单。"""
    found: list[tuple[Evidence, RefundProgress, AfterSaleInfo | None]] = []
    for ev in bundle.evidences:
        if not ev.id.startswith(_AFTERSALE_PREFIX):
            continue
        progress = _validate(RefundProgress, ev.data.get("progress"))
        if progress is not None:
            found.append((ev, progress, _validate(AfterSaleInfo, ev.data.get("aftersale"))))
    if not found:
        return None
    if ctx.order is not None:
        for ev, progress, info in found:
            if info is not None and info.order_id == ctx.order.order_id:
                return ev, progress
    ev, progress, _ = max(
        found, key=lambda x: (x[2].updated_at if x[2] is not None else x[1].as_of, x[1].as_of)
    )
    return ev, progress


def faq_hits(bundle: EvidenceBundle, min_score: float) -> list[Evidence]:
    """相似度达到 min_score、可以直接用标准回答的 FAQ（§5.4），按分数降序。

    被部分覆盖的 FAQ（id 出现在 overridden_ids 里）不直接用：原文里有和高优先级依据冲突的句子。
    """
    overridden = set(bundle.overridden_ids)
    hits = [
        ev
        for ev in bundle.evidences
        if ev.source == EvidenceSource.FAQ
        and ev.id.startswith(_FAQ_PREFIX)
        and ev.score is not None
        and ev.score >= min_score
        and ev.id not in overridden
        and text_of(ev, "answer")
    ]
    return sorted(hits, key=lambda ev: -(ev.score or 0.0))


def faq_id_of(ev: Evidence) -> int | None:
    raw = ev.data.get("faq_id")
    if isinstance(raw, int):
        return raw
    tail = ev.id[len(_FAQ_PREFIX) :]
    return int(tail) if tail.isdigit() else None
