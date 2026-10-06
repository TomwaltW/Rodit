"""半自动（方案 §5.6 输出模式"高置信度的简单问题，推荐直接填进输入框，客服按发送即可"、§13.4
"该类型原样采纳率 ≥ 90%"）：哪张推荐卡片打上 SuggestionCard.prefill。

只在智能辅助（ASSIST）下考虑，影子模式、全自动都不打；最多一张——第一张卡片，第一张不满足就
一张都不打（不往后找）。下面全部满足才打：

1. 问题类型在店铺开了半自动的类型里（ShopConfig.semi_auto_intents，复核人员按门槛开启，§13.4）；
2. 路由是正常的规则 6 回答（ANSWER）：不是高风险（规则 4；routing.high_risk、投诉再查一次）、
   不是分条（规则 5）、不是澄清（规则 7）、不是站外话术或安抚（规则 1、3），判断层没有降级；
3. 路由的问题类型置信度 ≥ routing.semi_auto_threshold；
4. 卡片通过了质检、置信度为高、没有风险提示，草稿不是澄清问题；
5. 这条消息没有任何提醒（要人工、情绪激动、可能漏答、连续没听懂、证据包里的提醒……），
   也没有任何降级（engine 的降级标记）。

半自动不是自动发送：插件只把文字填进客服输入框，客服看过、按发送才发出；反馈和采纳率的口径
不变（§5.8，客服按发送就是"发送"）。
"""

from __future__ import annotations

from collections.abc import Sequence

from rodit.core.config import RoutingConfig
from rodit.core.enums import Confidence, Intent, Mode, RouteAction
from rodit.core.models import Draft, QAResult, RouteDecision, ShopConfig, SuggestionCard

EPS = 1e-9  # 和 routing.policy 一样："≥" 在相等时命中，容忍浮点误差

# prefill_blocker 的返回值：第一条不满足的条件（测试、日志用；不进审计正文以外的地方）
BLOCK_MODE = "mode"  # 不是智能辅助（影子、全自动）
BLOCK_INTENT = "intent"  # 店铺没给这个类型开半自动
BLOCK_ROUTE = "route"  # 不是规则 6 的正常回答（分条、澄清、站外话术、安抚……）
BLOCK_HIGH_RISK = "high_risk"  # 高风险类型（§6.6 规则 4）
BLOCK_CONFIDENCE = "confidence"  # 问题类型置信度没达到 semi_auto_threshold
BLOCK_QA = "qa"  # 没通过质检
BLOCK_CLARIFY = "clarify"  # 草稿是澄清问题
BLOCK_CARD_CONFIDENCE = "card_confidence"  # 卡片置信度不是高
BLOCK_RISK_HINTS = "risk_hints"  # 卡片有风险提示
BLOCK_ALERTS = "alerts"  # 这条消息有提醒
BLOCK_DEGRADED = "degraded"  # 有降级（判断层、质检、工具、存储……）


def prefill_blocker(
    *,
    mode: Mode,
    config: ShopConfig,
    routing: RoutingConfig,
    route: RouteDecision,
    card: SuggestionCard,
    draft: Draft,
    qa: QAResult,
    alerts: Sequence[str] = (),
    degraded: Sequence[str] = (),
) -> str | None:
    """第一张卡片能不能打 prefill：能 → None；不能 → 第一条不满足的条件（BLOCK_*）。"""
    if mode != Mode.ASSIST:
        return BLOCK_MODE
    if route.intent not in config.semi_auto_intents or card.intent != route.intent:
        return BLOCK_INTENT
    if route.action != RouteAction.ANSWER or route.rule != 6 or route.parts:
        return BLOCK_ROUTE
    if route.high_risk or route.intent in routing.high_risk or route.intent == Intent.COMPLAINT:
        return BLOCK_HIGH_RISK
    if route.degraded:
        return BLOCK_DEGRADED
    if route.intent_confidence < routing.semi_auto_threshold - EPS:
        return BLOCK_CONFIDENCE
    if not qa.passed:
        return BLOCK_QA
    if draft.need_clarify or draft.variant == "clarify":
        return BLOCK_CLARIFY
    if card.confidence != Confidence.HIGH:
        return BLOCK_CARD_CONFIDENCE
    if card.risk_hints:
        return BLOCK_RISK_HINTS
    if any(a for a in alerts):
        return BLOCK_ALERTS
    if degraded or qa.degraded:
        return BLOCK_DEGRADED
    return None


__all__ = [
    "BLOCK_ALERTS",
    "BLOCK_CARD_CONFIDENCE",
    "BLOCK_CLARIFY",
    "BLOCK_CONFIDENCE",
    "BLOCK_DEGRADED",
    "BLOCK_HIGH_RISK",
    "BLOCK_INTENT",
    "BLOCK_MODE",
    "BLOCK_QA",
    "BLOCK_RISK_HINTS",
    "BLOCK_ROUTE",
    "prefill_blocker",
]
