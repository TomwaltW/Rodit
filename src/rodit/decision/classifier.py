"""路由分类器 ClassifierProvider（方案 §6.4 "微调的中文小模型分类器：数据攒够后的路由首选"）。

占位实现，等真正的微调小模型：字符 1–2 元组的多项式朴素贝叶斯 + 拉普拉斯平滑，
在平台云内训练和推理（数据不出云），推理几乎不花钱。

- 只回答 intent（带归一化的各类型概率）；need_human / emotion / multi_intent 交给注入的
  fallback（通常是 RuleProvider）。
- 训练数据：反馈闭环的标注集（LabeledSample.true_intent，§5.8）或 (文本, 问题类型)；
  可以加上 §5.1 的例句。训练文本先过 mask_pii，和线上 state 的文字（已脱敏）保持一致。
- 朴素贝叶斯的概率偏"自信"，没有校准；上线前必须按 §13.3 重新校准阈值（temperature 可调软硬）。
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from rodit.core.decision import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionProvider,
    DecisionRequest,
    DecisionResponse,
    require_buyer_data_ok,
    validate_response,
)
from rodit.core.enums import INTENT_EXAMPLES, Intent
from rodit.core.errors import DecisionError
from rodit.core.models import LabeledSample
from rodit.core.pii import mask_pii
from rodit.core.questions import Q_INTENT
from rodit.decision._common import current_buyer_text, normalize

_SKIP = re.compile(r"[\s，,。.！!？?；;：:、~～…\"'“”‘’()（）【】\[\]]+")
ROUND_DIGITS = 6


def char_ngrams(text: str) -> list[str]:
    """字符一元 + 二元组（先归一、去掉空白和标点）。"""
    t = _SKIP.sub("", normalize(text))
    return list(t) + [t[i : i + 2] for i in range(len(t) - 1)]


@dataclass(frozen=True)
class _NaiveBayes:
    classes: tuple[Intent, ...]
    log_prior: dict[Intent, float]
    log_likelihood: dict[Intent, dict[str, float]]
    log_unseen: dict[Intent, float]  # 词表内、但该类型没出现过的特征
    vocab: frozenset[str]

    def scores(self, features: list[str]) -> dict[Intent, float]:
        counts = Counter(f for f in features if f in self.vocab)
        out: dict[Intent, float] = {}
        for c in self.classes:
            ll = self.log_likelihood[c]
            unseen = self.log_unseen[c]
            out[c] = self.log_prior[c] + sum(n * ll.get(f, unseen) for f, n in counts.items())
        return out


def _train(samples: list[tuple[str, Intent]], alpha: float) -> _NaiveBayes:
    doc_counts: Counter[Intent] = Counter()
    feature_counts: dict[Intent, Counter[str]] = {}
    for text, intent in samples:
        doc_counts[intent] += 1
        feature_counts.setdefault(intent, Counter()).update(char_ngrams(text))
    vocab = frozenset(f for counter in feature_counts.values() for f in counter)
    total_docs = sum(doc_counts.values())
    classes = tuple(i for i in Intent if i in doc_counts)
    log_prior: dict[Intent, float] = {}
    log_likelihood: dict[Intent, dict[str, float]] = {}
    log_unseen: dict[Intent, float] = {}
    for c in classes:
        counter = feature_counts[c]
        denom = sum(counter.values()) + alpha * len(vocab)
        log_prior[c] = math.log(doc_counts[c] / total_docs)
        log_likelihood[c] = {f: math.log((n + alpha) / denom) for f, n in counter.items()}
        log_unseen[c] = math.log(alpha / denom)
    return _NaiveBayes(classes, log_prior, log_likelihood, log_unseen, vocab)


def _as_pairs(
    samples: Iterable[LabeledSample] | Iterable[tuple[str, Intent | str]],
) -> list[tuple[str, Intent]]:
    pairs: list[tuple[str, Intent]] = []
    for s in samples:
        if isinstance(s, LabeledSample):
            if s.true_intent is None:
                continue  # 只有预测、没有人工确认的类型，不能当标签
            pairs.append((mask_pii(s.text), s.true_intent))
        else:
            text, intent = s
            pairs.append((mask_pii(text), Intent(intent)))
    return pairs


class ClassifierProvider:
    """朴素贝叶斯路由分类器（实现 core.decision.DecisionProvider）。

    fallback 回答 intent 以外的路由问题，也会拿到买家数据，所以构造时同样要求它能处理买家数据。
    """

    accepts_buyer_data = True

    def __init__(
        self,
        fallback: DecisionProvider,
        *,
        alpha: float = 1.0,
        temperature: float = 1.0,
        name: str = "classifier",
    ) -> None:
        require_buyer_data_ok(fallback)
        if alpha <= 0:
            raise ValueError("alpha（拉普拉斯平滑）必须大于 0")
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        self.fallback = fallback
        self.alpha = alpha
        self.temperature = temperature
        self.name = name
        self._model: _NaiveBayes | None = None
        self._lock = threading.Lock()

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def classes(self) -> tuple[Intent, ...]:
        return self._model.classes if self._model else ()

    def fit(
        self,
        samples: Iterable[LabeledSample] | Iterable[tuple[str, Intent | str]],
        *,
        seed_examples: bool = False,
    ) -> ClassifierProvider:
        """用标注数据（重新）训练；seed_examples=True 时加上 §5.1 的例句。返回自身。"""
        pairs = _as_pairs(samples)
        if seed_examples:
            pairs += [(mask_pii(t), i) for i, texts in INTENT_EXAMPLES.items() for t in texts]
        if not pairs:
            raise ValueError("没有可用的训练样本（LabeledSample 需要 true_intent）")
        model = _train(pairs, self.alpha)
        with self._lock:
            self._model = model
        return self

    def predict_proba(self, text: str) -> dict[Intent, float]:
        """各问题类型的概率（softmax(对数后验 / temperature)），未训练时抛 DecisionError。"""
        model = self._model
        if model is None:
            raise DecisionError("路由分类器还没有训练")
        scores = model.scores(char_ngrams(mask_pii(text)))
        top = max(scores.values())
        exp = {c: math.exp((s - top) / self.temperature) for c, s in scores.items()}
        total = sum(exp.values())
        return {c: v / total for c, v in exp.items()}

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        answers: dict[str, Answer] = {}
        rest = {k: q for k, q in request.questions.items() if k != Q_INTENT}
        if Q_INTENT in request.questions:
            question = request.questions[Q_INTENT]
            if not isinstance(question, ChoiceQuestion):
                raise DecisionError(f"问题 {Q_INTENT!r} 应为 choice 题")
            answers[Q_INTENT] = self._intent_answer(current_buyer_text(request.state), question)
        if rest:
            sub = DecisionRequest(model=request.model, state=request.state, questions=rest)
            answers.update(self.fallback.decide(sub).answers)
        response = DecisionResponse(model=request.model, answers=answers)
        validate_response(request, response)
        return response

    def _intent_answer(self, text: str, question: ChoiceQuestion) -> ChoiceAnswer:
        probs = {c.value: p for c, p in self.predict_proba(text).items() if c in question.criteria}
        total = sum(probs.values())
        if total <= 0:
            raise DecisionError("路由分类器认识的类型都不在问题选项里")
        order = question.options
        probs = {k: round(p / total, ROUND_DIGITS) for k, p in probs.items()}
        probs = {k: p for k, p in probs.items() if p > 0}
        choice = min(probs, key=lambda k: (-probs[k], order.index(k)))
        return ChoiceAnswer(choice=choice, confidence=probs[choice], probabilities=probs)
