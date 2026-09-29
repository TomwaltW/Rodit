"""路由分类器（§6.4 数据攒够后的路由首选；这里是朴素贝叶斯占位实现）。"""

from __future__ import annotations

import pytest

from rodit.core.decision import ChoiceQuestion, DecisionRequest
from rodit.core.enums import INTENT_EXAMPLES, Intent
from rodit.core.errors import ComplianceError, DecisionError
from rodit.core.models import LabeledSample
from rodit.core.questions import (
    Q_EMOTION,
    Q_INTENT,
    Q_MULTI_INTENT,
    Q_NEED_HUMAN,
    ROUTER_MODEL,
    ROUTER_QUESTIONS,
)
from rodit.decision import ClassifierProvider, RuleProvider, char_ngrams
from rodit.testing.fakes import FakeDecisionProvider, router_answers
from rodit.testing.fixtures import DEFAULT_NOW, SHOP_A


def _request(text: str, questions=None) -> DecisionRequest:
    state = {"platform": "taobao", "messages": [{"role": "buyer", "text": text}]}
    return DecisionRequest(model=ROUTER_MODEL, state=state, questions=questions or ROUTER_QUESTIONS)


def _top(clf: ClassifierProvider, text: str) -> Intent:
    probs = clf.predict_proba(text)
    return max(probs, key=lambda i: probs[i])


def test_char_ngrams():
    assert char_ngrams("有 货吗？") == ["有", "货", "吗", "有货", "货吗"]
    assert char_ngrams("") == []


def test_untrained_classifier_refuses():
    clf = ClassifierProvider(RuleProvider())
    assert not clf.is_trained and clf.classes == ()
    with pytest.raises(DecisionError):
        clf.decide(_request("有货吗"))


def test_fit_and_predict(synthetic_intent_samples):
    clf = ClassifierProvider(RuleProvider()).fit(synthetic_intent_samples, seed_examples=True)
    assert clf.is_trained and set(clf.classes) == set(Intent)
    train_acc = sum(_top(clf, t) == i for t, i in synthetic_intent_samples) / len(
        synthetic_intent_samples
    )
    assert train_acc >= 0.95
    probs = clf.predict_proba("能开发票吗")
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_generalises_to_doc_examples(synthetic_intent_samples):
    """只用合成样例训练，§5.1 例句当留出集。"""
    clf = ClassifierProvider(RuleProvider()).fit(synthetic_intent_samples)
    examples = [(t, i) for i, texts in INTENT_EXAMPLES.items() for t in texts]
    accuracy = sum(_top(clf, t) == i for t, i in examples) / len(examples)
    assert accuracy >= 0.7


def test_decide_answers_intent_and_delegates_the_rest(synthetic_intent_samples):
    fallback = FakeDecisionProvider(
        router_answers("other", need_human=0.8, emotion=2.0, multi_intent=0.1), name="kw"
    )
    clf = ClassifierProvider(fallback).fit(synthetic_intent_samples, seed_examples=True)
    resp = clf.decide(_request("165 110斤穿什么码"))
    intent = resp.choice(Q_INTENT)
    assert intent.choice == "size_advice"
    assert sum(intent.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert set(intent.probabilities) <= {i.value for i in Intent}
    assert resp.noul(Q_NEED_HUMAN) == 0.8 and resp.score(Q_EMOTION) == 2.0
    assert resp.noul(Q_MULTI_INTENT) == 0.1
    (sub,) = fallback.requests
    assert set(sub.questions) == {Q_NEED_HUMAN, Q_EMOTION, Q_MULTI_INTENT}
    assert sub.state == _request("165 110斤穿什么码").state


def test_intent_only_request_skips_fallback(synthetic_intent_samples):
    fallback = FakeDecisionProvider(router_answers("other"))
    clf = ClassifierProvider(fallback).fit(synthetic_intent_samples)
    resp = clf.decide(_request("还有货吗", {Q_INTENT: ROUTER_QUESTIONS[Q_INTENT]}))
    assert set(resp.answers) == {Q_INTENT}
    assert fallback.requests == []


def test_restricts_to_criteria(synthetic_intent_samples):
    clf = ClassifierProvider(RuleProvider()).fit(synthetic_intent_samples)
    q = ChoiceQuestion(instructions="x", criteria={"stock": None, "greeting": None})
    ans = clf.decide(_request("还有货吗", {Q_INTENT: q})).choice(Q_INTENT)
    assert set(ans.probabilities) <= {"stock", "greeting"}
    assert sum(ans.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert ans.choice == "stock"
    q_none = ChoiceQuestion(instructions="x", criteria={"no_such": None})
    with pytest.raises(DecisionError):
        clf.decide(_request("还有货吗", {Q_INTENT: q_none}))


def test_fit_accepts_labeled_samples_and_skips_unlabelled():
    samples = [
        LabeledSample(
            shop_id=SHOP_A, text="还有货吗", true_intent=Intent.STOCK, created_at=DEFAULT_NOW
        ),
        LabeledSample(
            shop_id=SHOP_A,
            text="能便宜点吗",
            true_intent=Intent.PRICE_PROMO,
            created_at=DEFAULT_NOW,
        ),
        LabeledSample(
            shop_id=SHOP_A,
            text="随便说说",
            predicted_intent=Intent.GREETING,
            true_intent=None,
            created_at=DEFAULT_NOW,
        ),
    ]
    clf = ClassifierProvider(RuleProvider())
    assert clf.fit(samples) is clf
    assert set(clf.classes) == {Intent.STOCK, Intent.PRICE_PROMO}
    with pytest.raises(ValueError):
        ClassifierProvider(RuleProvider()).fit([samples[2]])


def test_training_text_is_masked_like_router_state():
    """训练文本先脱敏：带手机号的样例和线上 [手机号] 占位符是同一种特征。"""
    clf = ClassifierProvider(RuleProvider()).fit(
        [("我电话13812345678", Intent.MODIFY_ORDER), ("还有货吗", Intent.STOCK)]
    )
    assert _top(clf, "电话[手机号]") == Intent.MODIFY_ORDER


def test_temperature_softens_probabilities(synthetic_intent_samples):
    sharp = ClassifierProvider(RuleProvider()).fit(synthetic_intent_samples)
    soft = ClassifierProvider(RuleProvider(), temperature=5.0).fit(synthetic_intent_samples)
    assert max(soft.predict_proba("有货吗").values()) < max(sharp.predict_proba("有货吗").values())
    with pytest.raises(ValueError):
        ClassifierProvider(RuleProvider(), temperature=0)
    with pytest.raises(ValueError):
        ClassifierProvider(RuleProvider(), alpha=0)


def test_fallback_must_accept_buyer_data():
    with pytest.raises(ComplianceError):
        ClassifierProvider(FakeDecisionProvider(router_answers("other"), accepts_buyer_data=False))


def test_provider_identity():
    clf = ClassifierProvider(RuleProvider())
    assert clf.name == "classifier" and clf.accepts_buyer_data is True
