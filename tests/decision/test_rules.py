"""规则判断层（§6.4 规则与关键词、§12.2 降级路由）。"""

from __future__ import annotations

import pytest

from rodit.core.decision import (
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    ScoreQuestion,
    validate_response,
)
from rodit.core.enums import INTENT_EXAMPLES, Intent
from rodit.core.errors import DecisionError
from rodit.core.pii import mask_pii
from rodit.core.questions import (
    Q_EMOTION,
    Q_INTENT,
    Q_MULTI_INTENT,
    Q_NEED_HUMAN,
    QA_QUESTIONS,
    ROUTER_MODEL,
    ROUTER_QUESTIONS,
)
from rodit.core.states import build_router_state
from rodit.decision import rules as rules_module
from rodit.decision.rules import RuleProvider, analyze
from rodit.testing.fakes import make_context
from rodit.testing.fixtures import build_demo_world

W = build_demo_world()


def _state(text: str, *, history: list[tuple[str, str]] | None = None) -> dict:
    messages = [{"role": r, "text": mask_pii(t)} for r, t in history or []]
    messages.append({"role": "buyer", "text": mask_pii(text)})
    return {"platform": "taobao", "messages": messages, "item_card": None, "order_card": None}


def _ask(text: str, questions=None, **kw):
    request = DecisionRequest(
        model=ROUTER_MODEL, state=_state(text, **kw), questions=questions or ROUTER_QUESTIONS
    )
    return request, RuleProvider().decide(request)


def _top(text: str) -> Intent:
    return analyze(mask_pii(text)).choice


# ---------------------------------------------------------------- 准确率


def test_every_doc_example_is_recognised():
    """§5.1 的例句必须全部命中对应类型。"""
    for intent, texts in INTENT_EXAMPLES.items():
        for text in texts:
            assert _top(text) == intent, text


def test_top1_accuracy_on_synthetic_set(synthetic_intent_samples):
    covered = {i for _, i in synthetic_intent_samples}
    assert covered == set(Intent)  # 20 个类型都有样例
    assert len(synthetic_intent_samples) >= 80
    wrong = [(t, i, _top(t)) for t, i in synthetic_intent_samples if _top(t) != i]
    accuracy = 1 - len(wrong) / len(synthetic_intent_samples)
    assert accuracy >= 0.9, wrong


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("M码还有吗", Intent.STOCK),  # 尺码词被库存问法整体覆盖，不算尺码问题
        ("怎么用优惠券", Intent.PRICE_PROMO),
        ("新疆包邮吗", Intent.SHIPPING_POLICY),
        ("能寄到新疆吗", Intent.SHIPPING_POLICY),  # 动词在地区前面
        ("发不发西藏", Intent.SHIPPING_POLICY),
        ("送得到海南吗", Intent.SHIPPING_POLICY),
        ("可以邮寄到香港吗", Intent.SHIPPING_POLICY),
        ("什么时候能发货", Intent.SHIPPING_POLICY),
        ("退款什么时候到账", Intent.REFUND_PROGRESS),
        ("能退款吗", Intent.RETURN_POLICY),
        ("真丝的要怎么洗", Intent.USAGE),
        ("还没收到货，到哪了", Intent.LOGISTICS_TRACK),
        ("能改个手机号吗", Intent.MODIFY_ORDER),  # "手机号"被改单问法覆盖，不是站外请求
        ("快递员没打电话就放驿站了", Intent.LOGISTICS_TRACK),
        ("你们公司在哪个城市", Intent.OTHER),
    ],
)
def test_overlapping_keywords_prefer_the_specific_phrase(text, intent):
    assert _top(text) == intent


# ---------------------------------------------------------------- intent 概率


def test_probabilities_are_normalised_and_capped_below_auto_thresholds():
    _, resp = _ask("能开发票吗？")
    ans = resp.choice(Q_INTENT)
    assert ans.choice == "invoice_gift"
    assert ans.confidence == pytest.approx(0.9)
    assert sum(ans.probabilities.values()) == pytest.approx(1.0)
    assert ans.probabilities["other"] == pytest.approx(0.1)
    # 规则没有校准过的概率：最高 0.9，低于任何全自动阈值（0.95 / 0.97）
    assert rules_module.RULE_TOP_MASS < 0.95


def test_no_keyword_gives_low_confidence_other():
    _, resp = _ask("嗯")
    ans = resp.choice(Q_INTENT)
    assert (ans.choice, ans.confidence, ans.probabilities) == ("other", 0.4, {"other": 0.4})


def test_two_topics_split_probability_and_first_mentioned_wins_ties():
    a = analyze("有货吗能便宜点吗")
    assert a.probabilities[Intent.STOCK] == a.probabilities[Intent.PRICE_PROMO] == 0.45
    assert a.choice == Intent.STOCK
    assert a.ranking[:2] == (Intent.STOCK, Intent.PRICE_PROMO)


def test_same_topic_intents_merge_into_one():
    """ "什么时候发货？怎么还没发"是同一个话题：并成一个类型，不拆成两题。"""
    a = analyze("都三天了怎么还没发货，到底什么时候发")
    assert a.choice == Intent.URGE_SHIP
    assert a.confidence == pytest.approx(0.9)
    assert a.multi_intent < 0.7


def test_greeting_does_not_compete_with_a_real_question():
    a = analyze("在吗，这个有货吗")
    assert a.choice == Intent.STOCK and Intent.GREETING not in a.probabilities
    assert analyze("在吗？").choice == Intent.GREETING


def test_complaint_outweighs_other_topics():
    a = analyze("再不发货我就给差评")
    assert a.choice == Intent.COMPLAINT


def test_only_answers_criteria_options():
    q = ChoiceQuestion(instructions="x", criteria={"stock": None, "other": None})
    _, resp = _ask("有货吗能便宜点吗", questions={Q_INTENT: q})
    ans = resp.choice(Q_INTENT)
    assert set(ans.probabilities) <= {"stock", "other"} and ans.choice == "stock"
    q2 = ChoiceQuestion(instructions="x", criteria={"usage": None})
    with pytest.raises(DecisionError):
        _ask("有货吗", questions={Q_INTENT: q2})


# ---------------------------------------------------------------- 站外（规则 1）


def test_off_platform_request_dominates():
    _, resp = _ask("加个微信吧，转账便宜点")
    ans = resp.choice(Q_INTENT)
    assert ans.choice == "off_platform" and ans.probabilities["off_platform"] >= 0.6


def test_off_platform_uses_the_shared_lexicon(monkeypatch):
    """规则 1 的词表只有 core.lexicon 一份：替换它，规则判断跟着变。"""
    monkeypatch.setattr(rules_module, "find_off_platform_request", lambda text: [])
    assert analyze("加个微信吧").choice != Intent.OFF_PLATFORM
    monkeypatch.setattr(
        rules_module, "find_off_platform_request", lambda text: ["嘿"] if "嘿" in text else []
    )
    a = analyze("嘿嘿，有货吗")
    assert a.choice == Intent.OFF_PLATFORM and a.off_platform_strong


def test_masked_phone_is_only_a_weak_off_platform_signal():
    """买家改地址时留的电话被脱敏成 [手机号]：占位符不算买家要站外联系。"""
    text = mask_pii("地址写错了，改到杭州市西湖区文三路 188 号 3 单元 501 室，电话 13812345678")
    assert "[手机号]" in text and "[地址]" in text
    a = analyze(text)
    assert a.choice == Intent.MODIFY_ORDER
    assert not a.off_platform_strong
    assert a.probabilities.get(Intent.OFF_PLATFORM, 0) < 0.6


# ---------------------------------------------------------------- need_human / emotion / multi


@pytest.mark.parametrize(
    ("text", "low", "high"),
    [
        ("我要转人工", 0.7, 1.0),
        ("叫你们店长出来", 0.7, 1.0),
        ("我要打12315", 0.7, 1.0),
        ("别用机器人糊弄我", 0.7, 1.0),
        ("你是机器人吗", 0.0, 0.69),
        ("你是人工智能吗", 0.0, 0.69),
        ("你是真人吗", 0.0, 0.69),
        ("人工", 0.7, 1.0),
        ("有人工吗", 0.7, 1.0),
        ("我要人工，不要机器人", 0.7, 1.0),
        ("是真人试穿的吗", 0.0, 0.1),  # 商品问题里的"真人""人工"不是要人工
        ("纯人工缝制吗", 0.0, 0.1),
        ("不用转人工了，谢谢", 0.0, 0.2),
        ("这个有货吗", 0.0, 0.1),
    ],
)
def test_need_human(text, low, high):
    _, resp = _ask(text)
    assert low <= resp.noul(Q_NEED_HUMAN) <= high


@pytest.mark.parametrize(
    ("text", "low", "high"),
    [
        ("这个有货吗", 0, 0),
        ("滚筒洗衣机能洗吗", 0, 0),
        ("怎么还没发货", 1, 1),
        ("怎么还没发货？？", 2, 2),
        ("气死了！！", 3, 3),
        ("垃圾，骗子，我要给差评！！！", 4, 4),
        ("你们是不是傻逼", 3, 4),
    ],
)
def test_emotion_levels(text, low, high):
    _, resp = _ask(text)
    assert low <= resp.score(Q_EMOTION) <= high


def test_emotion_is_clipped_to_question_levels():
    q = ScoreQuestion(instructions="x", criteria=["平静", "不满", "愤怒"])
    _, resp = _ask("垃圾，骗子，我要给差评！！！", questions={Q_EMOTION: q})
    assert resp.score(Q_EMOTION) == 2


@pytest.mark.parametrize(
    ("text", "multi"),
    [
        ("有货吗？能便宜点吗？", True),
        ("有货吗能便宜点吗", True),  # 没标点也按"吗"切分句
        ("165 110斤穿什么码？是纯棉的吗？", True),
        ("这个是纯棉的吗", False),
        ("在吗，这个有货吗", False),  # 问候不算一个问题
        ("什么时候发货？发什么快递？", False),  # 同一话题
        ("尺寸偏大吗", False),  # 第二个话题只有泛泛的词
    ],
)
def test_multi_intent(text, multi):
    _, resp = _ask(text)
    assert (resp.noul(Q_MULTI_INTENT) >= 0.7) is multi


# ---------------------------------------------------------------- 请求处理


def test_reads_only_the_current_buyer_message():
    request, resp = _ask("在吗", history=[("buyer", "我要投诉"), ("agent", "亲，怎么了")])
    assert resp.choice(Q_INTENT).choice == "greeting"
    assert resp.noul(Q_NEED_HUMAN) < 0.7
    validate_response(request, resp)


def test_works_on_real_router_state():
    ctx = make_context("165 110斤穿什么码", W, item_id="i_555")
    request = DecisionRequest(
        model=ROUTER_MODEL, state=build_router_state(ctx), questions=ROUTER_QUESTIONS
    )
    resp = RuleProvider().decide(request)
    assert resp.model == ROUTER_MODEL
    assert set(resp.answers) == set(ROUTER_QUESTIONS)
    assert resp.choice(Q_INTENT).choice == "size_advice"


def test_empty_state_is_other():
    request = DecisionRequest(model=ROUTER_MODEL, state={}, questions=ROUTER_QUESTIONS)
    assert RuleProvider().decide(request).choice(Q_INTENT).choice == "other"


def test_answers_only_the_questions_asked():
    _, resp = _ask("有货吗", questions={Q_NEED_HUMAN: ROUTER_QUESTIONS[Q_NEED_HUMAN]})
    assert set(resp.answers) == {Q_NEED_HUMAN}


def test_rejects_unknown_questions_and_wrong_types():
    with pytest.raises(DecisionError):
        _ask("有货吗", questions=QA_QUESTIONS)
    with pytest.raises(DecisionError):
        _ask("有货吗", questions={Q_EMOTION: NoulQuestion(instructions="x")})


def test_provider_identity():
    p = RuleProvider()
    assert p.name == "rules" and p.accepts_buyer_data is True
    assert RuleProvider(name="kw").name == "kw"
