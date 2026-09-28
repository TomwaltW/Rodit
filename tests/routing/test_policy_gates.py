"""§6.6 规则 3 的回答部分、店铺关闭的投诉、投票粒度下的阈值、多问题分不出条（集成审查修复）。"""

from __future__ import annotations

import pytest

from rodit.core.config import RoutingConfig
from rodit.core.decision import DecisionResponse
from rodit.core.enums import Intent, Mode, RouteAction
from rodit.core.models import ShopConfig
from rodit.core.pipeline import soothe_with_answer
from rodit.core.questions import ROUTER_MODEL
from rodit.routing import (
    ALERT_COMPLAINT,
    ALERT_EMOTION,
    ALERT_MULTI_UNSPLIT,
    RISK_HINTS,
    RoutingPolicy,
    vote_threshold,
)
from rodit.testing.fakes import router_answers

A = RouteAction
POLICY = RoutingPolicy(RoutingConfig())
SHOP = ShopConfig()
AUTO_SHOP = ShopConfig(auto_reply_enabled=True)


def _resp(intent: Intent | str = "size_advice", confidence: float = 0.92, **kw) -> DecisionResponse:
    return DecisionResponse(model=ROUTER_MODEL, answers=router_answers(intent, confidence, **kw))


def _route(resp, *, mode=Mode.ASSIST, shop=SHOP, degraded=False, **kw):
    return POLICY.apply(resp, mode=mode, shop_config=shop, degraded=degraded, provider="fake", **kw)


# ---------------------------------------------------------------- 规则 3：安抚 + 回答的闸门


def test_soothe_high_risk_intent_carries_its_risk_hint():
    """§6.6 规则 4 / §5.1：情绪激动时照样附回答，高风险类型的回答要带风险提示。"""
    d = _route(_resp("price_promo", emotion=4.0))
    assert (d.action, d.rule) == (A.SOOTHE, 3)
    assert d.risk_hints == [ALERT_EMOTION, RISK_HINTS[Intent.PRICE_PROMO]]
    assert soothe_with_answer(d, SHOP.disabled_intents) is True
    calm = _route(_resp("size_advice", emotion=4.0))
    assert calm.risk_hints == [ALERT_EMOTION]


def test_soothe_complaint_alerts_human_and_attaches_no_answer():
    d = _route(_resp("complaint", emotion=4.0))
    assert (d.action, d.rule) == (A.SOOTHE, 3)
    assert d.alerts == [ALERT_EMOTION, ALERT_COMPLAINT]
    assert d.risk_hints == [ALERT_EMOTION]
    assert soothe_with_answer(d, SHOP.disabled_intents) is False


def test_soothe_degraded_high_risk_attaches_no_answer():
    """§12.2 判断层不可用：高风险类型一律不推荐，安抚话术照给。"""
    d = _route(_resp("price_promo", emotion=4.0), degraded=True)
    assert (d.action, d.rule, d.degraded) == (A.SOOTHE, 3, True)
    assert d.risk_hints == [ALERT_EMOTION]
    assert soothe_with_answer(d, SHOP.disabled_intents) is False
    assert soothe_with_answer(_route(_resp("size_advice", emotion=4.0), degraded=True), []) is True


def test_soothe_disabled_intent_attaches_no_answer():
    shop = ShopConfig(disabled_intents=[Intent.PRICE_PROMO])
    d = _route(_resp("price_promo", emotion=4.0), shop=shop)
    assert (d.action, d.rule) == (A.SOOTHE, 3)
    assert d.risk_hints == [ALERT_EMOTION]
    assert soothe_with_answer(d, shop.disabled_intents) is False


# ---------------------------------------------------------------- 店铺关闭投诉：照样提醒人工


@pytest.mark.parametrize(
    ("mode", "action"), [(Mode.ASSIST, A.HUMAN_PRIORITY), (Mode.AUTO, A.TRANSFER_HUMAN)]
)
def test_disabled_complaint_still_escalates(mode, action):
    shop = ShopConfig(disabled_intents=[Intent.COMPLAINT], auto_reply_enabled=True)
    d = _route(_resp("complaint", 0.9), shop=shop, mode=mode)
    assert (d.action, d.rule, d.alerts) == (action, 4, [ALERT_COMPLAINT])


# ---------------------------------------------------------------- 投票粒度（§6.4 自一致性 3 票）


@pytest.mark.parametrize(
    ("threshold", "votes", "expected"),
    [
        (0.7, None, 0.7),
        (0.7, 1, 0.7),
        (0.7, 3, 2 / 3),  # 默认：否则 0.7 等于要求全票
        (0.7, 2, 0.7),  # 2 票：过半数就是全票
        (0.7, 4, 0.7),  # 3/4 本来就够
        (0.7, 5, 0.7),  # 4/5 本来就够，3/5 不够
        (0.7, 6, 4 / 6),
        (0.9, 3, 0.9),  # 离 3/3 更近：仍要全票
        (0.3, 3, 0.3),  # 低于过半数的阈值不动
        (2 / 3, 3, 2 / 3),
    ],
)
def test_vote_threshold_snaps_to_nearest_reachable_share(threshold, votes, expected):
    assert vote_threshold(threshold, votes) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("mode", "action"), [(Mode.ASSIST, A.HUMAN_PRIORITY), (Mode.AUTO, A.TRANSFER_HUMAN)]
)
def test_need_human_two_of_three_votes_meets_the_threshold(mode, action):
    resp = _resp("greeting", 1.0, need_human=0.666667)
    d = _route(resp, mode=mode, shop=AUTO_SHOP, vote_samples=3)
    assert (d.action, d.rule) == (action, 2)
    assert _route(resp, mode=mode, shop=AUTO_SHOP, vote_samples=3).auto_eligible is False
    # 不是投票得到的概率（分类器、关键词）：照常按 0.7 比较
    assert _route(resp, mode=mode, shop=AUTO_SHOP).rule == 6
    one_vote = _resp("greeting", 1.0, need_human=0.333333)
    assert _route(one_vote, mode=mode, shop=AUTO_SHOP, vote_samples=3).rule == 6


def test_multi_intent_two_of_three_votes_splits():
    probs = {"product_spec": 0.666667, "shipping_policy": 0.333333}
    resp = _resp("product_spec", 0.666667, probabilities=probs, multi_intent=0.666667)
    d = _route(resp, vote_samples=3)
    assert (d.action, d.rule) == (A.SPLIT_MULTI, 5)
    assert [p.intent for p in d.parts] == [Intent.PRODUCT_SPEC, Intent.SHIPPING_POLICY]


def test_auto_keyword_need_human_signal_forces_transfer():
    """§5.7 全自动：买家要求人工 → 立即转人工；判断层没认出来时关键词规则兜底。"""
    resp = _resp("greeting", 1.0, need_human=0.333333)
    d = _route(resp, mode=Mode.AUTO, shop=AUTO_SHOP, vote_samples=3, keyword_need_human=0.92)
    assert (d.action, d.rule, d.need_human) == (A.TRANSFER_HUMAN, 2, 0.92)
    weak = _route(resp, mode=Mode.AUTO, shop=AUTO_SHOP, vote_samples=3, keyword_need_human=0.05)
    assert weak.rule == 6 and weak.need_human == 0.333333
    # 智能辅助有人把关，不用关键词信号覆盖判断层
    assisted = _route(resp, vote_samples=3, keyword_need_human=0.92)
    assert assisted.rule == 6


# ---------------------------------------------------------------- 规则 5：多问题分不出条


def test_multi_intent_without_two_parts_alerts_the_agent():
    """所有票都选了同一个主要类型：分不出第二条，推荐只回答一个问题，要提醒客服。"""
    resp = _resp("product_spec", 1.0, probabilities={"product_spec": 1.0}, multi_intent=1.0)
    d = _route(resp, vote_samples=3)
    assert (d.action, d.rule) == (A.ANSWER, 6)
    assert d.alerts == [ALERT_MULTI_UNSPLIT]
    assert _route(_resp("product_spec", 1.0, multi_intent=0.05)).alerts == []


def test_multi_intent_dropped_complaint_part_keeps_complaint_alert():
    probs = {"size_advice": 0.65, "complaint": 0.35}
    resp = _resp("size_advice", 0.65, probabilities=probs, multi_intent=0.9)
    d = _route(resp)
    assert (d.action, d.rule) == (A.ANSWER, 6)
    assert d.alerts == [ALERT_MULTI_UNSPLIT, ALERT_COMPLAINT]
