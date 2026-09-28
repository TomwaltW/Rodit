"""路由规则（方案 §6.6）：判断层答案 → RouteDecision。

按顺序判断，命中就停，rule 记下命中的规则序号：

1. P(off_platform) ≥ off_platform → 平台规则标准话术（优先级最高；即使店铺关了这一类也照样给）；
2. need_human ≥ need_human       → 辅助 / 影子：提醒客服优先处理；自动：转人工；
                                    自动模式另看关键词规则的"要求人工"信号（§5.7 兜底）；
3. emotion ≥ emotion_escalate    → 辅助 / 影子：安抚话术并标红；自动：转人工；
                                    安抚话术之外的回答和规则 4–8 用同样的闸门
                                    （core.pipeline.soothe_with_answer）：店铺关闭的类型、投诉、
                                    降级时的高风险类型不附回答；投诉另加提醒人工，附回答的
                                    高风险类型另加该类型的风险提示；
   （投诉照样按规则 4 提醒人工；其余店铺关闭的问题类型在这里拦下 → 不推荐，记为规则 8）
4. intent 属于高风险类型          → 投诉：提醒人工、不推荐（§5.1、§5.7）；
                                    其余：辅助推荐 + 风险提示，判断层降级时不推荐（§12.2）；
                                    自动：转人工；
5. multi_intent ≥ multi_intent   → 辅助：按问题分条推荐（概率 ≥ multi_intent_min_prob 的类型，
                                    降序最多 max_parts 个；不足两个就往下走，并提醒客服可能有
                                    漏答的问题）；自动：转人工；
6. intent 置信度 ≥ 辅助阈值        → 检索、生成、质检出推荐；达到该类型的自动阈值、店铺开了全自动、
                                    没降级时 auto_eligible；
7. 置信度低且前两名相差 < clarify_max_gap → 澄清；
8. 其他                            → 不推荐。

阈值比较："≥" 在相等时命中、"<" 在相等时不命中（容忍 1e-9 的浮点误差）。
是否题（need_human、multi_intent）的答案来自 n 票自一致性投票时只能取 0、1/n、…、1，
阈值先按 vote_threshold 换算到票数粒度（默认 3 票、0.7 → 2/3 票，否则 0.7 等于要求全票）。
"""

from __future__ import annotations

import math

from rodit.core.config import RoutingConfig
from rodit.core.decision import ChoiceAnswer, DecisionResponse, NoulAnswer, ScoreAnswer
from rodit.core.enums import Intent, Mode, RouteAction
from rodit.core.models import RouteDecision, RoutePart, ShopConfig
from rodit.core.pipeline import DEFAULT_RISK_HINT, RISK_HINTS, risk_hint, soothe_with_answer
from rodit.core.questions import Q_EMOTION, Q_INTENT, Q_MULTI_INTENT, Q_NEED_HUMAN

EPS = 1e-9

ALERT_NEED_HUMAN = "买家要求人工 / 投诉，请优先处理"
ALERT_EMOTION = "买家情绪激动，建议先安抚"
ALERT_COMPLAINT = "买家投诉 / 纠纷，请人工优先处理"
ALERT_MULTI_UNSPLIT = "买家消息里可能不止一个问题，没能分条推荐，请留意是否有漏答"

# RISK_HINTS / DEFAULT_RISK_HINT / risk_hint 定义在 core.pipeline（引擎给分条加提示也要用），
# 这里照旧导出
__all__ = [
    "ALERT_COMPLAINT",
    "ALERT_EMOTION",
    "ALERT_MULTI_UNSPLIT",
    "ALERT_NEED_HUMAN",
    "DEFAULT_RISK_HINT",
    "RISK_HINTS",
    "RoutingPolicy",
    "risk_hint",
    "vote_threshold",
]


def _ge(value: float, threshold: float) -> bool:
    return value >= threshold - EPS


def _lt(value: float, threshold: float) -> bool:
    return value < threshold - EPS


def _as_intent(value: str) -> Intent | None:
    try:
        return Intent(value)
    except ValueError:
        return None


def vote_threshold(threshold: float, votes: int | None) -> float:
    """是否题阈值换算到投票粒度（§6.4 自一致性投票）。

    n 票投票时"是"的占比只能取 0、1/n、…、1。阈值落在两个可取值之间、离下面那个更近
    （差不到半票）时取下面那个，但不低于过半数；其他情况原样返回。votes 为空或 1
    （分类器、关键词规则、只采样一次时用模型自报的 p）时不换算。
    默认 3 票、阈值 0.7 → 2/3（否则等于要求全票）；5 票时 0.7 仍要 4/5。
    """
    if votes is None or votes < 2:
        return threshold
    needed = math.ceil(threshold * votes - EPS)  # 原样比较需要的票数
    nearest = math.floor(threshold * votes + 0.5)  # 离阈值最近的票数
    if nearest < needed and nearest >= votes // 2 + 1:
        return nearest / votes
    return threshold


class RoutingPolicy:
    """§6.6 八条规则。纯函数式：不调判断层、不写日志（那是 DecisionRouter 的事）。"""

    def __init__(self, config: RoutingConfig) -> None:
        self.config = config

    def apply(
        self,
        answers: DecisionResponse,
        *,
        mode: Mode,
        shop_config: ShopConfig,
        degraded: bool,
        provider: str,
        vote_samples: int | None = None,
        keyword_need_human: float | None = None,
    ) -> RouteDecision:
        """vote_samples：是否题答案来自几票投票（None：不是投票得到的，见 vote_threshold）；
        keyword_need_human：关键词规则的"要求人工"概率，只在全自动模式参与规则 2（§5.7）。"""
        cfg = self.config
        auto = mode == Mode.AUTO
        high_risk = set(cfg.high_risk)
        disabled = set(shop_config.disabled_intents)

        # ---------------------------------------------------------------- 读答案
        intent_answer = answers.answers.get(Q_INTENT)
        intent, confidence = Intent.OTHER, 0.0
        probs: dict[Intent, float] = {}
        top: list[tuple[Intent, float]] = []
        if isinstance(intent_answer, ChoiceAnswer):
            intent = _as_intent(intent_answer.choice) or Intent.OTHER
            confidence = intent_answer.confidence
            for key, p in intent_answer.probabilities.items():
                if (i := _as_intent(key)) is not None:
                    probs[i] = p
            top = [(i, p) for k, p in intent_answer.top(2) if (i := _as_intent(k)) is not None]
        need_human = self._noul(answers, Q_NEED_HUMAN)
        emotion = self._score(answers, Q_EMOTION)
        multi = self._noul(answers, Q_MULTI_INTENT)
        wants_human = _ge(need_human, vote_threshold(cfg.need_human, vote_samples))
        if auto and keyword_need_human is not None:
            # 全自动没人把关：判断层没认出"要人工"时，关键词规则的信号也算（§5.7 立即转人工）
            wants_human = wants_human or _ge(keyword_need_human, cfg.need_human)
            need_human = max(need_human, keyword_need_human)

        base = {
            "intent": intent,
            "intent_confidence": confidence,
            "top_intents": top,
            "need_human": need_human,
            "emotion": emotion,
            "multi_intent": multi,
            "high_risk": intent in high_risk or intent == Intent.COMPLAINT,
            "degraded": degraded,
            "provider": provider,
            "answers": answers,
        }

        def decide(action: RouteAction, rule: int, **extra: object) -> RouteDecision:
            return RouteDecision.model_validate({**base, "action": action, "rule": rule, **extra})

        # 规则 1：站外交易请求（优先级最高，防止违规）
        p_off = probs.get(Intent.OFF_PLATFORM, 0.0)
        if intent == Intent.OFF_PLATFORM:
            p_off = max(p_off, confidence)
        if _ge(p_off, cfg.off_platform):
            return decide(
                RouteAction.OFF_PLATFORM_SCRIPT,
                1,
                intent=Intent.OFF_PLATFORM,
                intent_confidence=p_off,
                high_risk=False,
            )

        # 规则 2：要求人工 / 投诉
        if wants_human:
            action = RouteAction.TRANSFER_HUMAN if auto else RouteAction.HUMAN_PRIORITY
            return decide(action, 2, alerts=[ALERT_NEED_HUMAN])

        # 规则 3：情绪激烈
        if _ge(emotion, cfg.emotion_escalate):
            if auto:
                return decide(RouteAction.TRANSFER_HUMAN, 3, alerts=[ALERT_EMOTION])
            alerts = [ALERT_EMOTION]
            if intent == Intent.COMPLAINT:  # 投诉：安抚之外还要提醒人工（§5.7）
                alerts.append(ALERT_COMPLAINT)
            soothe = decide(RouteAction.SOOTHE, 3, alerts=alerts, risk_hints=[ALERT_EMOTION])
            if intent in high_risk and soothe_with_answer(soothe, disabled):
                # 附带的回答是高风险类型：推荐 + 风险提示（§5.1、§6.6 规则 4）
                hints = [ALERT_EMOTION, risk_hint(intent)]
                soothe = soothe.model_copy(update={"risk_hints": hints})
            return soothe

        # 投诉一律提醒人工（§5.1、§5.7）；店铺关了"投诉"这一类也一样——关的是推荐，不是提醒
        if intent == Intent.COMPLAINT:
            action = RouteAction.TRANSFER_HUMAN if auto else RouteAction.HUMAN_PRIORITY
            return decide(action, 4, alerts=[ALERT_COMPLAINT])

        # 店铺关闭的问题类型：不推荐（规则 1–3 和投诉仍然生效）
        if intent in disabled:
            return decide(RouteAction.NO_SUGGESTION, 8)

        # 规则 4：高风险类型
        if intent in high_risk:
            hints = [risk_hint(intent)]
            if auto:
                return decide(RouteAction.TRANSFER_HUMAN, 4, risk_hints=hints)
            if degraded:  # §12.2 判断层不可用：高风险类型一律不推荐
                return decide(RouteAction.NO_SUGGESTION, 4, risk_hints=hints)
            return decide(RouteAction.HIGH_RISK_ANSWER, 4, risk_hints=hints)

        # 规则 5：多个问题
        pending_alerts: list[str] = []  # 规则 5 分不出条时带到后面的规则
        if _ge(multi, vote_threshold(cfg.multi_intent, vote_samples)):
            if auto:
                return decide(RouteAction.TRANSFER_HUMAN, 5)
            parts, dropped_complaint = self._parts(probs, disabled, high_risk, degraded)
            alerts = [ALERT_COMPLAINT] if dropped_complaint else []
            if len(parts) >= 2:
                hints = [risk_hint(p.intent) for p in parts if p.intent in high_risk]
                return decide(
                    RouteAction.SPLIT_MULTI, 5, parts=parts, risk_hints=hints, alerts=alerts
                )
            # 判断层说有多个问题、却认不出第二个类型（投票时各票常常选同一个"主要"类型）：
            # 往下走只答一个问题，提醒客服别漏答；被去掉的投诉照样提醒人工
            pending_alerts = [ALERT_MULTI_UNSPLIT, *alerts]

        # 规则 6：置信度够 → 出推荐
        if _ge(confidence, cfg.assist_threshold):
            auto_threshold = cfg.auto_threshold.get(intent)
            eligible = (
                auto
                and shop_config.auto_reply_enabled
                and auto_threshold is not None
                and _ge(confidence, auto_threshold)
                and not degraded
            )
            return decide(RouteAction.ANSWER, 6, auto_eligible=eligible, alerts=pending_alerts)

        # 规则 7：置信度低、前两名接近 → 澄清
        top1 = top[0][1] if top else confidence
        top2 = top[1][1] if len(top) > 1 else 0.0
        if _lt(top1 - top2, cfg.clarify_max_gap):
            return decide(RouteAction.CLARIFY, 7, alerts=pending_alerts)

        # 规则 8：其他
        return decide(RouteAction.NO_SUGGESTION, 8, alerts=pending_alerts)

    # ---------------------------------------------------------------- 工具

    @staticmethod
    def _noul(answers: DecisionResponse, key: str) -> float:
        a = answers.answers.get(key)
        return a.noul if isinstance(a, NoulAnswer) else 0.0

    @staticmethod
    def _score(answers: DecisionResponse, key: str) -> float:
        a = answers.answers.get(key)
        return a.score if isinstance(a, ScoreAnswer) else 0.0

    def _parts(
        self,
        probs: dict[Intent, float],
        disabled: set[Intent],
        high_risk: set[Intent],
        degraded: bool,
    ) -> tuple[list[RoutePart], bool]:
        """规则 5 的分条：概率 ≥ multi_intent_min_prob，降序，最多 max_parts 个。

        不单独成条的：other（没有可答的内容）、投诉（只提醒人工）、店铺关闭的类型、
        判断层降级时的高风险类型（§12.2）。返回 (分条, 是否因投诉被去掉)。
        """
        cfg = self.config
        ranked = sorted(probs.items(), key=lambda kv: (-kv[1], list(Intent).index(kv[0])))
        parts: list[RoutePart] = []
        dropped_complaint = False
        for intent, p in ranked:
            if not _ge(p, cfg.multi_intent_min_prob):
                break
            if intent == Intent.COMPLAINT:
                dropped_complaint = True
                continue
            if intent == Intent.OTHER or intent in disabled:
                continue
            if degraded and intent in high_risk:
                continue
            if len(parts) < cfg.max_parts:
                parts.append(RoutePart(intent=intent, confidence=min(max(p, 0.0), 1.0)))
        return parts, dropped_complaint
