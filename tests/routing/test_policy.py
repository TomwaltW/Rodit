"""§6.6 八条路由规则：顺序、阈值边界、各模式、降级、店铺关闭的类型。"""

from __future__ import annotations

import pytest

from rodit.core.config import RoutingConfig
from rodit.core.decision import ChoiceAnswer, DecisionResponse
from rodit.core.enums import Intent, Mode, RouteAction
from rodit.core.models import RoutePart, ShopConfig
from rodit.core.questions import Q_INTENT, ROUTER_MODEL
from rodit.routing import (
    ALERT_COMPLAINT,
    ALERT_EMOTION,
    ALERT_NEED_HUMAN,
    DEFAULT_RISK_HINT,
    RISK_HINTS,
    RoutingPolicy,
)
from rodit.testing.fakes import router_answers

A = RouteAction
POLICY = RoutingPolicy(RoutingConfig())
SHOP = ShopConfig()
AUTO_SHOP = ShopConfig(auto_reply_enabled=True)


def _resp(intent: Intent | str = "size_advice", confidence: float = 0.92, **kw) -> DecisionResponse:
    return DecisionResponse(model=ROUTER_MODEL, answers=router_answers(intent, confidence, **kw))


def _route(
    resp: DecisionResponse,
    *,
    mode: Mode = Mode.ASSIST,
    shop: ShopConfig = SHOP,
    degraded: bool = False,
    policy: RoutingPolicy = POLICY,
):
    return policy.apply(resp, mode=mode, shop_config=shop, degraded=degraded, provider="fake")


# ---------------------------------------------------------------- 规则 1：站外


def test_rule1_off_platform_threshold_is_inclusive():
    d = _route(_resp("off_platform", 0.6))
    assert (d.action, d.rule, d.intent, d.intent_confidence) == (
        A.OFF_PLATFORM_SCRIPT,
        1,
        Intent.OFF_PLATFORM,
        0.6,
    )
    assert _route(_resp("off_platform", 0.59)).action != A.OFF_PLATFORM_SCRIPT


def test_rule1_reads_the_off_platform_probability_even_if_not_the_choice():
    probs = {"price_promo": 0.3, "off_platform": 0.65}
    d = _route(_resp("price_promo", 0.3, probabilities=probs))
    assert d.action == A.OFF_PLATFORM_SCRIPT and d.intent == Intent.OFF_PLATFORM
    assert d.intent_confidence == 0.65 and d.high_risk is False


def test_rule1_uses_confidence_when_probabilities_are_empty():
    resp = _resp("off_platform", 0.7)
    resp.answers[Q_INTENT] = ChoiceAnswer(choice="off_platform", confidence=0.7)
    assert _route(resp).action == A.OFF_PLATFORM_SCRIPT


@pytest.mark.parametrize("mode", list(Mode))
def test_rule1_beats_everything_and_ignores_disabled_intents(mode):
    resp = _resp("off_platform", 0.97, need_human=0.99, emotion=4.0, multi_intent=0.99)
    shop = ShopConfig(disabled_intents=[Intent.OFF_PLATFORM], auto_reply_enabled=True)
    d = _route(resp, mode=mode, shop=shop, degraded=True)
    assert (d.action, d.rule) == (A.OFF_PLATFORM_SCRIPT, 1)
    assert d.auto_eligible is False  # 只有规则 6 才会判全自动


# ---------------------------------------------------------------- 规则 2：要求人工


@pytest.mark.parametrize(
    ("mode", "action"),
    [
        (Mode.ASSIST, A.HUMAN_PRIORITY),
        (Mode.SHADOW, A.HUMAN_PRIORITY),
        (Mode.AUTO, A.TRANSFER_HUMAN),
    ],
)
def test_rule2_need_human(mode, action):
    d = _route(_resp(need_human=0.7, emotion=4.0), mode=mode)
    assert (d.action, d.rule, d.alerts) == (action, 2, [ALERT_NEED_HUMAN])
    assert d.need_human == 0.7
    assert _route(_resp(need_human=0.69), mode=mode).rule == 6


# ---------------------------------------------------------------- 规则 3：情绪


@pytest.mark.parametrize(
    ("mode", "action"),
    [(Mode.ASSIST, A.SOOTHE), (Mode.SHADOW, A.SOOTHE), (Mode.AUTO, A.TRANSFER_HUMAN)],
)
def test_rule3_emotion(mode, action):
    d = _route(_resp("price_promo", emotion=3.0, multi_intent=0.9), mode=mode)
    assert (d.action, d.rule) == (action, 3)
    assert d.alerts == [ALERT_EMOTION]
    if action == A.SOOTHE:
        # 推荐卡片上的风险提示（§5.6）；附带的价格回答还要带价格承诺提示（§6.6 规则 4）
        assert d.risk_hints == [ALERT_EMOTION, RISK_HINTS[Intent.PRICE_PROMO]]
    assert _route(_resp(emotion=2.99)).rule == 6


# ---------------------------------------------------------------- 店铺关闭的类型


def test_disabled_intent_gets_no_suggestion_after_rules_1_to_3():
    shop = ShopConfig(disabled_intents=[Intent.SIZE_ADVICE, Intent.PRICE_PROMO])
    d = _route(_resp("size_advice", 0.99), shop=shop)
    assert (d.action, d.rule) == (A.NO_SUGGESTION, 8)
    assert _route(_resp("price_promo", 0.99), shop=shop).action == A.NO_SUGGESTION  # 先于规则 4
    assert _route(_resp("size_advice", need_human=0.9), shop=shop).rule == 2
    assert _route(_resp("size_advice", emotion=3.5), shop=shop).rule == 3


# ---------------------------------------------------------------- 规则 4：高风险


@pytest.mark.parametrize("intent", sorted(RISK_HINTS))
def test_rule4_high_risk_answer_with_intent_specific_hint(intent):
    d = _route(_resp(intent, 0.9, multi_intent=0.95))
    assert (d.action, d.rule, d.high_risk) == (A.HIGH_RISK_ANSWER, 4, True)
    assert d.risk_hints == [RISK_HINTS[intent]]


def test_rule4_hint_texts():
    assert _route(_resp("price_promo")).risk_hints == ["涉及价格承诺，请确认"]
    assert _route(_resp("not_received")).risk_hints == ["请核查物流签收细分状态"]


def test_rule4_applies_even_at_low_confidence():
    """§6.6 规则 4 不看置信度：高风险类型排在置信度判断之前。"""
    assert _route(_resp("modify_order", 0.3)).action == A.HIGH_RISK_ANSWER


def test_rule4_auto_mode_transfers():
    d = _route(_resp("quality_issue", 0.99), mode=Mode.AUTO, shop=AUTO_SHOP)
    assert (d.action, d.rule, d.auto_eligible) == (A.TRANSFER_HUMAN, 4, False)


def test_rule4_degraded_high_risk_gets_no_suggestion():
    d = _route(_resp("price_promo", 0.9), degraded=True)
    assert (d.action, d.rule, d.degraded) == (A.NO_SUGGESTION, 4, True)
    assert d.risk_hints == [RISK_HINTS[Intent.PRICE_PROMO]]


@pytest.mark.parametrize(
    ("mode", "degraded", "action"),
    [
        (Mode.ASSIST, False, A.HUMAN_PRIORITY),
        (Mode.SHADOW, False, A.HUMAN_PRIORITY),
        (Mode.ASSIST, True, A.HUMAN_PRIORITY),  # 投诉本来就不推荐，只提醒人工
        (Mode.AUTO, False, A.TRANSFER_HUMAN),
    ],
)
def test_rule4_complaint_never_gets_a_suggestion(mode, degraded, action):
    d = _route(_resp("complaint", 0.8), mode=mode, degraded=degraded)
    assert (d.action, d.rule, d.alerts, d.high_risk) == (action, 4, [ALERT_COMPLAINT], True)


def test_complaint_is_handled_even_if_removed_from_high_risk_config():
    policy = RoutingPolicy(RoutingConfig(high_risk=[Intent.PRICE_PROMO]))
    assert _route(_resp("complaint"), policy=policy).action == A.HUMAN_PRIORITY
    assert _route(_resp("quality_issue"), policy=policy).action == A.ANSWER


def test_configured_high_risk_intent_without_specific_hint():
    policy = RoutingPolicy(RoutingConfig(high_risk=[Intent.STOCK]))
    d = _route(_resp("stock"), policy=policy)
    assert d.action == A.HIGH_RISK_ANSWER and d.risk_hints == [DEFAULT_RISK_HINT]


# ---------------------------------------------------------------- 规则 5：多问题


def _multi(probabilities: dict[str, float], multi: float = 0.8, **kw) -> DecisionResponse:
    top = max(probabilities, key=lambda k: probabilities[k])
    return _resp(top, probabilities[top], probabilities=probabilities, multi_intent=multi, **kw)


def test_rule5_split_parts_descending_and_other_excluded():
    d = _route(_multi({"size_advice": 0.45, "stock": 0.35, "other": 0.2}, multi=0.7))
    assert (d.action, d.rule) == (A.SPLIT_MULTI, 5)
    assert d.parts == [
        RoutePart(intent=Intent.SIZE_ADVICE, confidence=0.45),
        RoutePart(intent=Intent.STOCK, confidence=0.35),
    ]


def test_rule5_min_prob_is_inclusive_and_max_parts_caps():
    probs = {"size_advice": 0.3, "stock": 0.25, "usage": 0.2, "product_spec": 0.2, "other": 0.05}
    d = _route(_multi(probs))
    assert [p.intent for p in d.parts] == [Intent.SIZE_ADVICE, Intent.STOCK, Intent.PRODUCT_SPEC]
    policy = RoutingPolicy(RoutingConfig(max_parts=4))
    assert len(_route(_multi(probs), policy=policy).parts) == 4
    below = {"size_advice": 0.6, "stock": 0.19, "other": 0.21}
    assert _route(_multi(below)).action == A.ANSWER  # 只有一个类型达标：往下走到规则 6


def test_rule5_threshold_boundary():
    probs = {"size_advice": 0.5, "stock": 0.4, "other": 0.1}
    assert _route(_multi(probs, multi=0.7)).action == A.SPLIT_MULTI
    assert _route(_multi(probs, multi=0.69)).action == A.CLARIFY  # 0.5 - 0.4 < 0.2


def test_rule5_high_risk_part_gets_hint_and_is_dropped_when_degraded():
    probs = {"size_advice": 0.5, "price_promo": 0.4, "stock": 0.1}
    d = _route(_multi(probs))
    assert [p.intent for p in d.parts] == [Intent.SIZE_ADVICE, Intent.PRICE_PROMO]
    assert d.risk_hints == [RISK_HINTS[Intent.PRICE_PROMO]]
    degraded = _route(_multi(probs), degraded=True)
    assert degraded.action == A.CLARIFY  # 降级时价格那条不能单独推荐，只剩一条 → 往下走


def test_rule5_complaint_and_disabled_parts_are_not_split():
    probs = {"size_advice": 0.4, "complaint": 0.35, "stock": 0.25}
    d = _route(_multi(probs))
    assert [p.intent for p in d.parts] == [Intent.SIZE_ADVICE, Intent.STOCK]
    assert d.alerts == [ALERT_COMPLAINT]
    shop = ShopConfig(disabled_intents=[Intent.STOCK])
    assert _route(_multi(probs), shop=shop).action != A.SPLIT_MULTI


def test_rule5_auto_mode_transfers():
    d = _route(_multi({"size_advice": 0.5, "stock": 0.5}), mode=Mode.AUTO, shop=AUTO_SHOP)
    assert (d.action, d.rule, d.parts) == (A.TRANSFER_HUMAN, 5, [])


# ---------------------------------------------------------------- 规则 6：出推荐 / 全自动


def test_rule6_assist_threshold_is_inclusive():
    d = _route(_resp("size_advice", 0.6))
    assert (d.action, d.rule, d.auto_eligible) == (A.ANSWER, 6, False)


@pytest.mark.parametrize(
    ("mode", "shop", "intent", "confidence", "degraded", "eligible"),
    [
        (Mode.AUTO, AUTO_SHOP, "greeting", 0.95, False, True),  # 阈值相等也算
        (Mode.AUTO, AUTO_SHOP, "product_spec", 0.97, False, True),
        (Mode.AUTO, AUTO_SHOP, "product_spec", 0.96, False, False),  # 该类型阈值 0.97
        (Mode.AUTO, AUTO_SHOP, "greeting", 0.9499, False, False),
        (Mode.AUTO, AUTO_SHOP, "usage", 0.99, False, False),  # 没开放全自动的类型
        (Mode.AUTO, SHOP, "greeting", 0.99, False, False),  # 店铺没开全自动
        (Mode.AUTO, AUTO_SHOP, "greeting", 0.99, True, False),  # 降级
        (Mode.ASSIST, AUTO_SHOP, "greeting", 0.99, False, False),
        (Mode.SHADOW, AUTO_SHOP, "greeting", 0.99, False, False),
    ],
)
def test_rule6_auto_eligibility_matrix(mode, shop, intent, confidence, degraded, eligible):
    d = _route(_resp(intent, confidence), mode=mode, shop=shop, degraded=degraded)
    assert (d.action, d.rule) == (A.ANSWER, 6)
    assert d.auto_eligible is eligible


# ---------------------------------------------------------------- 规则 7、8


def test_rule7_clarify_between_close_candidates():
    probs = {"size_advice": 0.5, "product_spec": 0.35, "other": 0.15}
    d = _route(_resp("size_advice", 0.5, probabilities=probs))
    assert (d.action, d.rule) == (A.CLARIFY, 7)
    assert d.top_intents == [(Intent.SIZE_ADVICE, 0.5), (Intent.PRODUCT_SPEC, 0.35)]


def test_rule7_gap_equal_to_threshold_does_not_clarify():
    probs = {"size_advice": 0.55, "product_spec": 0.35, "other": 0.1}
    d = _route(_resp("size_advice", 0.55, probabilities=probs))
    assert (d.action, d.rule) == (A.NO_SUGGESTION, 8)


def test_rule8_low_confidence_single_candidate():
    d = _route(_resp("other", 0.4, probabilities={"other": 0.4}))
    assert (d.action, d.rule, d.intent) == (A.NO_SUGGESTION, 8, Intent.OTHER)


# ---------------------------------------------------------------- 结果字段


def test_decision_carries_all_judge_outputs():
    resp = _resp("size_advice", 0.92, need_human=0.1, emotion=1.5, multi_intent=0.3)
    d = RoutingPolicy(RoutingConfig()).apply(
        resp, mode=Mode.ASSIST, shop_config=SHOP, degraded=True, provider="rules"
    )
    assert d.intent == Intent.SIZE_ADVICE and d.intent_confidence == 0.92
    assert d.top_intents == [(Intent.SIZE_ADVICE, 0.92), (Intent.OTHER, 0.08)]
    assert (d.need_human, d.emotion, d.multi_intent) == (0.1, 1.5, 0.3)
    assert (d.degraded, d.provider, d.high_risk) == (True, "rules", False)
    assert d.answers == resp
    assert d.parts == [] and d.alerts == [] and d.risk_hints == []


def test_missing_answers_default_to_zero_and_unknown_intent_to_other():
    resp = DecisionResponse(
        model=ROUTER_MODEL,
        answers={Q_INTENT: ChoiceAnswer(choice="size_advice", confidence=0.8)},
    )
    d = _route(resp)
    assert (d.action, d.need_human, d.emotion, d.multi_intent) == (A.ANSWER, 0.0, 0.0, 0.0)
    odd = DecisionResponse(
        model=ROUTER_MODEL,
        answers={Q_INTENT: ChoiceAnswer(choice="mystery", confidence=0.9, probabilities={})},
    )
    assert _route(odd).intent == Intent.OTHER
