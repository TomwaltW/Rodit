"""确定性模板（方案 §5.4 "能不用大模型就不用"、§5.2.4、§5.7、§6.6、§7.2、§7.3.1、§11.2–§11.4、
§12.2）。

- 称呼和语气来自店铺生效配置（ctx.config.salutation / tone）：
  "亲" 开头用"亲，"，"您" 开头用"您好，"；语气亲切时句末带"哦"，专业、简洁不带。
- 事实一律来自证据，模板只负责措辞；返回的 Draft 带上用到的证据 ID，质检照常跑硬规则。
- 物流：始终写"截至 {时间}"；驿站 / 快递柜给取件指引；只有接口给出预计送达时间才提到货时间（§7.2）。
- 签收未收到（§7.3.1、§11.3）：物流进度（签收细分状态）+ 驿站 / 快递柜取件指引 + 已签收时
  "还是没找到就在订单页申请售后"（有平台申请售后规则的证据时，引用它）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from rodit.core.clock import CST, fmt_time
from rodit.core.enums import INTENT_LABELS, DraftKind, Intent, LogisticsStatus
from rodit.core.evidence import PlatformRuleKey
from rodit.core.models import (
    ConversationContext,
    Draft,
    Evidence,
    EvidenceBundle,
    LogisticsAnalysis,
    RefundProgress,
    RouteDecision,
    ShopConfig,
)
from rodit.generation import evidence_view as V

# ---------------------------------------------------------------- 模板 ID

GREETING = "greeting"
OFF_PLATFORM = "off_platform"
CLARIFY_ITEM = "clarify_item"
CLARIFY_ORDER = "clarify_order"
CLARIFY_INTENT = "clarify_intent"
CLARIFY = "clarify"  # 证据包给出的其他澄清问题
SOOTHE = "soothe"
TOOL_UNAVAILABLE = "tool_unavailable"
LOGISTICS = "logistics"
NOT_RECEIVED = "not_received"
REFUND_PROGRESS = "refund_progress"
APPLY_AFTERSALE = "apply_aftersale"

NOT_FOUND_APPLY = "如果还是没找到，您可以在订单页申请售后"
"""签收未收到的最后一步（§7.3.1 没找到就引导申请售后；平台只允许买家在订单页发起）。"""

Variant = Literal["direct", "with_followup", "clarify", "soothe", "script"]

CLARIFY_ITEM_TEXT = "请问您问的是哪一款呢？"
CLARIFY_ORDER_TEXT = "请问您问的是哪个订单呢？"


# ---------------------------------------------------------------- 称呼与语气


@dataclass(frozen=True)
class Style:
    opening: str  # 开头称呼
    particle: str  # 句末语气词


def style_for(config: ShopConfig) -> Style:
    opening = "亲，" if config.salutation == "亲" else "您好，"
    particle = "哦" if config.tone == "亲切" else ""
    return Style(opening=opening, particle=particle)


_TRAILING = "。．.！!？?；;，,、~ \t\n"


def _clause(text: str) -> str:
    """去掉句末标点，便于拼接。"""
    return (text or "").strip().rstrip(_TRAILING)


def _time_text(dt: datetime, now: datetime) -> str:
    """同一天只写时分（"10:30"），不是同一天带日期（"09-27 15:23"）。"""
    same_day = dt.astimezone(CST).date() == now.astimezone(CST).date()
    return fmt_time(dt, with_date=not same_day)


def _template(
    template_id: str,
    text: str,
    *,
    evidence_ids: list[str] | None = None,
    variant: Variant = "direct",
    need_clarify: bool = False,
) -> Draft:
    return Draft(
        text=text,
        evidence_ids=list(evidence_ids or []),
        kind=DraftKind.TEMPLATE,
        need_clarify=need_clarify,
        variant=variant,
        template_id=template_id,
    )


# ---------------------------------------------------------------- 站外交易（§11.4）


def off_platform_script(ctx: ConversationContext, evidence: EvidenceBundle | None = None) -> Draft:
    """平台规则标准话术（§11.4）。有平台站外交易规则的证据时引用它。"""
    s = style_for(ctx.config)
    text = (
        f"{s.opening}平台规定不能通过站外联系和交易{s.particle}，"
        "您直接在店铺下单就好，我帮您看看有没有可以用的优惠券。"
    )
    ids: list[str] = []
    if evidence is not None:
        rule = V.platform_rule(evidence, ctx.platform, PlatformRuleKey.OFF_PLATFORM)
        if rule is not None:
            ids.append(rule.id)
    return _template(OFF_PLATFORM, text, evidence_ids=ids, variant="script")


# ---------------------------------------------------------------- 问候


def greeting(ctx: ConversationContext, evidence: EvidenceBundle | None = None) -> Draft:
    """问候 / 催回复。有客服时间政策时带上并引用（附录 C 第 13 题）。"""
    s = style_for(ctx.config)
    text = f"{s.opening}在的{s.particle}，请问有什么可以帮您？"
    ids: list[str] = []
    hours = V.service_hours(evidence) if evidence is not None else None
    if hours is not None:
        value = _clause(V.text_of(hours, "value"))
        if value:
            if "时间" not in value:
                value = f"客服在线时间：{value}"
            text += f"{value}。"
            ids.append(hours.id)
    return _template(GREETING, text, evidence_ids=ids)


# ---------------------------------------------------------------- 澄清（§5.2.4、§6.6 规则 7）


def clarify_item(ctx: ConversationContext) -> Draft:
    s = style_for(ctx.config)
    return _template(
        CLARIFY_ITEM, f"{s.opening}{CLARIFY_ITEM_TEXT}", variant="clarify", need_clarify=True
    )


def clarify_order(ctx: ConversationContext) -> Draft:
    s = style_for(ctx.config)
    return _template(
        CLARIFY_ORDER, f"{s.opening}{CLARIFY_ORDER_TEXT}", variant="clarify", need_clarify=True
    )


def clarify_question(ctx: ConversationContext, question: str) -> Draft:
    """证据包给出的澄清问题（商品 / 订单不确定时）。没有称呼时补上称呼。"""
    q = question.strip()
    if "订单" in q:
        template_id = CLARIFY_ORDER
    elif "哪一款" in q or "哪款" in q or "商品" in q:
        template_id = CLARIFY_ITEM
    else:
        template_id = CLARIFY
    if not q.startswith(("亲", "您")):
        q = style_for(ctx.config).opening + q
    return _template(template_id, q, variant="clarify", need_clarify=True)


def _intent_label(intent: Intent) -> str:
    return INTENT_LABELS[intent].replace(" / ", "、")


def clarify_intent(ctx: ConversationContext, route: RouteDecision) -> Draft:
    """规则 7：置信度低、前两名接近 → "请问您是想了解 A 还是 B 呢？"（按 route.top_intents）。"""
    candidates = [intent for intent, _ in route.top_intents] or [route.intent]
    labels: list[str] = []
    for intent in candidates:
        intent = Intent(intent)
        if intent == Intent.OTHER:
            continue
        label = _intent_label(intent)
        if label not in labels:
            labels.append(label)
    s = style_for(ctx.config)
    if len(labels) >= 2:
        question = f"请问您是想了解{labels[0]}还是{labels[1]}呢？"
    elif labels:
        question = f"请问您是想了解{labels[0]}吗？"
    else:
        question = "请问您具体想咨询什么问题呢？"
    return _template(CLARIFY_INTENT, f"{s.opening}{question}", variant="clarify", need_clarify=True)


# ---------------------------------------------------------------- 安抚（§5.7、§6.6 规则 3）


def soothe(ctx: ConversationContext) -> Draft:
    s = style_for(ctx.config)
    please = "您" if ctx.config.salutation == "亲" else "请您"
    text = f"{s.opening}非常抱歉给您带来了不好的体验，{please}先别着急，我这边帮您看一下具体情况。"
    return _template(SOOTHE, text, variant="soothe")


# ---------------------------------------------------------------- 工具不可用（§12.2）


def tool_unavailable(ctx: ConversationContext) -> Draft:
    s = style_for(ctx.config)
    return _template(TOOL_UNAVAILABLE, f"{s.opening}我帮您查一下，请稍等{s.particle}。")


# ---------------------------------------------------------------- 物流（§7.2、§11.2、§11.3）


def logistics(ctx: ConversationContext, ev: Evidence, analysis: LogisticsAnalysis) -> Draft:
    """物流进度：截至时间 + 状态摘要 + 取件指引 + 异常时给买家的话。

    只有接口给出预计送达时间（analysis.eta）才提到货时间。
    """
    s = style_for(ctx.config)
    summary = _clause(analysis.summary)
    if "截至" in summary:
        text = f"{s.opening}{summary}。"
    else:
        text = f"{s.opening}截至 {_time_text(analysis.as_of, ctx.now)}，包裹最新状态：{summary}。"
    extras: list[str] = []
    if analysis.pickup_guidance:
        extras.append(_clause(analysis.pickup_guidance))
    for anomaly in analysis.anomalies:
        if anomaly.buyer_hint:
            extras.append(_clause(anomaly.buyer_hint))
    for extra in dict.fromkeys(e for e in extras if e):
        text += f"{extra}。"
    if analysis.eta is not None:
        text += f"平台显示预计 {fmt_time(analysis.eta)} 前送达，具体以物流为准。"
    return _template(LOGISTICS, text, evidence_ids=[ev.id])


def not_received(
    ctx: ConversationContext,
    ev: Evidence,
    analysis: LogisticsAnalysis,
    *,
    apply_rule: Evidence | None = None,
) -> Draft:
    """显示签收但没收到（§7.3.1、§11.3）：物流进度（签收细分状态：本人 / 驿站 / 快递柜代收）、
    驿站或快递柜的取件指引；已签收、且有平台申请售后规则的证据时，最后告诉买家"还是没找到就在
    订单页申请售后"（只能买家发起，客服不替买家申请，§1.4）。没签收时只说物流进度。"""
    s = style_for(ctx.config)
    base = logistics(ctx, ev, analysis)
    text, ids = base.text, list(base.evidence_ids)
    if analysis.status == LogisticsStatus.SIGNED and apply_rule is not None:
        text += f"{NOT_FOUND_APPLY}{s.particle}。"
        ids.append(apply_rule.id)
    return _template(NOT_RECEIVED, text, evidence_ids=ids)


# ---------------------------------------------------------------- 售后（§7.3.1）


def refund_progress(
    ctx: ConversationContext,
    ev: Evidence,
    progress: RefundProgress,
    timing: Evidence | None = None,
) -> Draft:
    """退款进度：当前进度（截至时间）+ 平台的一般到账规则（有证据时）。"""
    s = style_for(ctx.config)
    summary = _clause(progress.summary)
    if "截至" in summary:
        text = f"{s.opening}{summary}。"
    else:
        text = f"{s.opening}截至 {_time_text(progress.as_of, ctx.now)}，您的售后进度：{summary}。"
    ids = [ev.id]
    if timing is not None:
        rule = _clause(V.text_of(timing, "text"))
        if rule:
            text += f"{rule}。"
            ids.append(timing.id)
    return _template(REFUND_PROGRESS, text, evidence_ids=ids)


def apply_aftersale(ctx: ConversationContext, rule: Evidence) -> Draft:
    """ "怎么申请退款"：平台只允许买家在订单页发起，给出该平台的操作指引（§7.3.1）。"""
    s = style_for(ctx.config)
    text = f"{s.opening}申请售后的方法：{_clause(V.text_of(rule, 'text'))}。"
    return _template(APPLY_AFTERSALE, text, evidence_ids=[rule.id])


_HOW_TO_APPLY = (
    re.compile(
        r"(?:怎么|怎样|如何|在哪里?|哪里)(?:才能|可以|能|去|来|进行)?(?:申请|办理|发起)?"
        r"(?:退款|退货|售后|退换|换货|退)"
    ),
    re.compile(
        r"(?:退款|退货|售后|退换|换货)(?:要|该|应该|需要)?(?:怎么|怎样|如何|在哪里?)"
        r"(?:申请|办理|发起|弄|操作)"
    ),
)


def asks_how_to_apply(text: str) -> bool:
    """买家是不是在问"怎么申请退款 / 退货 / 售后"。"""
    return any(p.search(text or "") for p in _HOW_TO_APPLY)
