"""路由分类器 ClassifierProvider（方案 §6.4 "微调的中文小模型分类器：数据攒够后的路由首选"）。

占位实现，等真正的微调小模型：字符 1–2 元组的多项式朴素贝叶斯 + 拉普拉斯平滑，
在平台云内训练和推理（数据不出云），推理几乎不花钱。

- 只回答 intent（带归一化的各类型概率）；need_human / emotion / multi_intent 交给注入的
  fallback（通常是 RuleProvider）。
- 训练数据：反馈闭环的标注集（LabeledSample.true_intent，§5.8）或 (文本, 问题类型)；
  可以加上 §5.1 的例句。训练文本先过 mask_pii，和线上 state 的文字（已脱敏）保持一致。
- 朴素贝叶斯的概率偏"自信"，没有校准；上线前必须按 §13.3 重新校准阈值（temperature 可调软硬）。

模型文件（§6.4 Phase 2：离线训练 → 版本化的文件 → 生产单元加载，见 rodit.eval.train）：

- JSON，不用 pickle（加载不执行任何代码）。顶层是 format / format_version / model / metadata /
  sha256；model 里存的是整数计数（各类型的样本数、各类型的特征计数）和 alpha、temperature，
  加载时按和 fit 完全相同的算法重建对数概率，所以加载后的预测和训练时逐位一致；
- metadata：训练时间、训练样本数、例句条数、路由问题定义的版本（core.questions）、评估指标；
- sha256 是除它以外全部内容的规范 JSON（键排序、紧凑、UTF-8）的摘要，只防损坏和误改，
  不防有意篡改：模型文件和标注数据一样只放在平台单元内、只有部署账号能写；
- 加载时校验格式名、版本、摘要、问题类型（不认识的类型直接拒绝：问题分类改过就要重新训练）、
  计数的结构和一致性，任何一项不对都抛 ClassifierArtifactError，不会带着半个模型上线。
- 模型里的 n 元组计数来自已脱敏（mask_pii）的买家问题，不含原句，但罕见的二元组可能是原文片段：
  模型文件只能留在平台单元内（§4.1 原则 4、§12.3），不要拷出平台云。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)

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
from rodit.core.questions import Q_INTENT, ROUTER_QUESTIONS_VER
from rodit.decision._common import current_buyer_text, normalize

_SKIP = re.compile(r"[\s，,。.！!？?；;：:、~～…\"'“”‘’()（）【】\[\]]+")
ROUND_DIGITS = 6

ARTIFACT_FORMAT = "rodit.route_classifier"
ARTIFACT_VERSION = 1
MODEL_KIND = "naive_bayes"
FEATURES = "char_1_2gram"  # char_ngrams 的特征定义；改特征提取时换名字并升格式版本
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024  # 远大于几千条标注训出来的模型；防止误读一个巨型文件


def char_ngrams(text: str) -> list[str]:
    """字符一元 + 二元组（先归一、去掉空白和标点）。"""
    t = _SKIP.sub("", normalize(text))
    return list(t) + [t[i : i + 2] for i in range(len(t) - 1)]


def feature_key(text: str) -> str:
    """特征意义上的"同一句话"：脱敏、归一、去掉空白和标点后的文字（训练去重用）。

    两句话的 feature_key 相同，char_ngrams 就相同，对朴素贝叶斯来说是同一条样本。
    """
    return _SKIP.sub("", normalize(mask_pii(text)))


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


@dataclass(frozen=True)
class _Counts:
    """训练的充分统计量：模型文件里存的就是它（整数，精确可复现）。"""

    doc_counts: dict[Intent, int]
    feature_counts: dict[Intent, dict[str, int]]


def _count(samples: list[tuple[str, Intent]]) -> _Counts:
    doc_counts: Counter[Intent] = Counter()
    feature_counts: dict[Intent, Counter[str]] = {}
    for text, intent in samples:
        doc_counts[intent] += 1
        feature_counts.setdefault(intent, Counter()).update(char_ngrams(text))
    classes = [i for i in Intent if i in doc_counts]
    return _Counts(
        doc_counts={c: doc_counts[c] for c in classes},
        feature_counts={c: dict(feature_counts.get(c, {})) for c in classes},
    )


def _build(counts: _Counts, alpha: float) -> _NaiveBayes:
    """从计数算对数概率。训练和加载模型文件走同一个函数，结果逐位一致。"""
    vocab = frozenset(f for counter in counts.feature_counts.values() for f in counter)
    total_docs = sum(counts.doc_counts.values())
    classes = tuple(i for i in Intent if i in counts.doc_counts)
    log_prior: dict[Intent, float] = {}
    log_likelihood: dict[Intent, dict[str, float]] = {}
    log_unseen: dict[Intent, float] = {}
    for c in classes:
        counter = counts.feature_counts.get(c, {})
        denom = sum(counter.values()) + alpha * len(vocab)
        log_prior[c] = math.log(counts.doc_counts[c] / total_docs)
        log_likelihood[c] = {f: math.log((n + alpha) / denom) for f, n in counter.items()}
        log_unseen[c] = math.log(alpha / denom)
    return _NaiveBayes(classes, log_prior, log_likelihood, log_unseen, vocab)


def _train(samples: list[tuple[str, Intent]], alpha: float) -> _NaiveBayes:
    return _build(_count(samples), alpha)


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


# ================================================================ 模型文件


class ClassifierArtifactError(ValueError):
    """模型文件不能用：不存在、读不出来、不是本格式、版本不支持、校验和不符、结构不对、
    有不认识的问题类型。消息里只有路径和原因，没有文件内容。"""


class ClassifierMeta(BaseModel):
    """模型文件的 metadata。metrics 是训练任务的评估结果（只有计数和比率，没有买家原文）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trained_at: AwareDatetime
    questions_version: str  # 训练时路由问题定义的版本（core.questions.ROUTER_QUESTIONS_VER）
    samples: int = Field(ge=0)  # 训练用的标注条数（去重后，不含例句）
    seed_examples: int = Field(ge=0)  # 加进训练的 §5.1 例句条数
    metrics: dict[str, Any] = Field(default_factory=dict)


_Count = Annotated[int, Field(strict=True, ge=1)]
_Feature = Annotated[str, StringConstraints(min_length=1, max_length=2)]


class _ModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["naive_bayes"]
    features: Literal["char_1_2gram"]
    alpha: float = Field(gt=0, allow_inf_nan=False)
    temperature: float = Field(gt=0, allow_inf_nan=False)
    classes: list[Intent] = Field(min_length=2)
    doc_counts: dict[Intent, _Count]
    feature_counts: dict[Intent, dict[_Feature, _Count]]


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["rodit.route_classifier"]
    format_version: Literal[1]
    model: _ModelParams
    metadata: ClassifierMeta
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _canonical(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def artifact_digest(artifact: Mapping[str, Any]) -> str:
    """模型文件的校验和：除 sha256 以外全部内容的规范 JSON 的 SHA-256。"""
    return hashlib.sha256(
        _canonical({k: v for k, v in artifact.items() if k != "sha256"})
    ).hexdigest()


def dump_artifact(artifact: Mapping[str, Any]) -> bytes:
    """模型文件的字节内容（规范 JSON + 换行）：同样的模型和 metadata 得到同样的字节。"""
    return _canonical(dict(artifact)) + b"\n"


def write_artifact(path: str | os.PathLike[str], artifact: Mapping[str, Any]) -> None:
    """原子写入：先写同目录的临时文件并落盘，再改名覆盖（读的一方只会看到旧文件或完整的新文件）。

    覆盖已有文件的策略由调用方决定（见 rodit.eval.train / rodit.ops 的 train-classifier）。"""
    target = Path(path)
    data = dump_artifact(artifact)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o640)  # 模型文件含脱敏文本的 n 元组统计：不给其他账号读
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def read_artifact(path: str | os.PathLike[str]) -> dict[str, Any]:
    """读模型文件并校验（格式、版本、校验和、结构、问题类型），返回原始 dict。"""
    p = Path(path)
    try:
        size = p.stat().st_size
    except FileNotFoundError:
        raise ClassifierArtifactError(f"模型文件不存在：{p}") from None
    except OSError as e:
        raise ClassifierArtifactError(f"模型文件读不了：{p}（{type(e).__name__}）") from None
    if size > MAX_ARTIFACT_BYTES:
        raise ClassifierArtifactError(f"模型文件过大：{p}（{size} 字节）")
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise ClassifierArtifactError(f"模型文件读不了：{p}（{type(e).__name__}）") from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ClassifierArtifactError(f"模型文件不是合法的 JSON：{p}") from None
    try:
        _validate_artifact(data)
    except ClassifierArtifactError as e:
        raise ClassifierArtifactError(f"{e}：{p}") from None
    return data


def _validate_artifact(data: Any) -> _Artifact:
    if not isinstance(data, dict):
        raise ClassifierArtifactError("模型文件不是 JSON 对象")
    if data.get("format") != ARTIFACT_FORMAT:
        raise ClassifierArtifactError("不是路由分类器的模型文件（format 不对）")
    version = data.get("format_version")
    if type(version) is not int or version != ARTIFACT_VERSION:
        raise ClassifierArtifactError(
            f"不支持的模型文件格式版本 {version!r}（本程序只认 {ARTIFACT_VERSION}）"
        )
    digest = data.get("sha256")
    try:
        actual = artifact_digest(data)
    except (TypeError, ValueError):
        raise ClassifierArtifactError("模型文件内容不是合法的 JSON 值") from None
    if not isinstance(digest, str) or digest != actual:
        raise ClassifierArtifactError("模型文件校验和不符：文件损坏或被改过")
    model = data.get("model")
    if isinstance(model, dict):
        named: list[Any] = []
        for key in ("classes", "doc_counts", "feature_counts"):
            value = model.get(key)
            named.extend(value if isinstance(value, list | dict) else [])
        known = {i.value for i in Intent}
        unknown = sorted({str(n) for n in named if not (isinstance(n, str) and n in known)})
        if unknown:
            raise ClassifierArtifactError(
                f"模型文件里有不认识的问题类型 {unknown}（问题分类改过？需要重新训练）"
            )
    try:
        parsed = _Artifact.model_validate(data)
    except ValidationError as e:
        fields = sorted({".".join(str(p) for p in err["loc"][:3]) for err in e.errors()})
        raise ClassifierArtifactError(f"模型文件结构不对：{', '.join(fields[:5])}") from None
    params = parsed.model
    if len(set(params.classes)) != len(params.classes):
        raise ClassifierArtifactError("模型文件的 classes 有重复")
    if not (set(params.classes) == set(params.doc_counts) == set(params.feature_counts)):
        raise ClassifierArtifactError("模型文件的 classes、doc_counts、feature_counts 对不上")
    if [c for c in Intent if c in set(params.classes)] != list(params.classes):
        raise ClassifierArtifactError("模型文件的 classes 顺序不对（应与问题分类表一致）")
    return parsed


# ================================================================ 判断层实现


@dataclass(frozen=True)
class _Fitted:
    model: _NaiveBayes
    counts: _Counts
    samples: int  # 标注样本条数（不含例句）
    seed_examples: int


class ClassifierProvider:
    """朴素贝叶斯路由分类器（实现 core.decision.DecisionProvider）。

    fallback 回答 intent 以外的路由问题，也会拿到买家数据，所以构造时同样要求它能处理买家数据。
    从模型文件加载的实例带 meta（ClassifierMeta）和 artifact_sha256，便于启动日志和留痕核对。
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
        if not alpha > 0 or not math.isfinite(alpha):
            raise ValueError("alpha（拉普拉斯平滑）必须大于 0")
        if not temperature > 0 or not math.isfinite(temperature):
            raise ValueError("temperature 必须大于 0")
        self.fallback = fallback
        self.alpha = alpha
        self.temperature = temperature
        self.name = name
        self.meta: ClassifierMeta | None = None
        self.artifact_sha256: str | None = None
        self._fitted: _Fitted | None = None
        self._lock = threading.Lock()

    @property
    def is_trained(self) -> bool:
        return self._fitted is not None

    @property
    def classes(self) -> tuple[Intent, ...]:
        fitted = self._fitted
        return fitted.model.classes if fitted else ()

    def fit(
        self,
        samples: Iterable[LabeledSample] | Iterable[tuple[str, Intent | str]],
        *,
        seed_examples: bool | Iterable[tuple[str, Intent | str]] = False,
    ) -> ClassifierProvider:
        """用标注数据（重新）训练，返回自身。

        seed_examples=True 时加上 §5.1 的例句（INTENT_EXAMPLES）；也可以直接给例句列表
        （训练任务去掉和评测集重复的例句后传进来）。例句条数单独记在模型文件的 metadata 里。"""
        pairs = _as_pairs(samples)
        n_samples = len(pairs)
        if seed_examples is True:
            pairs += [(mask_pii(t), i) for i, texts in INTENT_EXAMPLES.items() for t in texts]
        elif seed_examples is not False:
            pairs += _as_pairs(seed_examples)
        if not pairs:
            raise ValueError("没有可用的训练样本（LabeledSample 需要 true_intent）")
        counts = _count(pairs)
        fitted = _Fitted(_build(counts, self.alpha), counts, n_samples, len(pairs) - n_samples)
        with self._lock:
            self._fitted = fitted
            self.meta = None  # 新训练的模型还没有模型文件
            self.artifact_sha256 = None
        return self

    def predict_proba(self, text: str) -> dict[Intent, float]:
        """各问题类型的概率（softmax(对数后验 / temperature)），未训练时抛 DecisionError。"""
        fitted = self._fitted
        if fitted is None:
            raise DecisionError("路由分类器还没有训练")
        scores = fitted.model.scores(char_ngrams(mask_pii(text)))
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

    # ------------------------------------------------------------ 模型文件

    def to_artifact(
        self, *, trained_at: datetime, metrics: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """模型文件的内容（dict，带 sha256）。同样的训练数据、参数、trained_at 和 metrics
        得到同样的内容（计数是整数，键在序列化时排序）。"""
        fitted = self._fitted
        if fitted is None:
            raise ValueError("路由分类器还没有训练，不能导出模型文件")
        metrics = dict(metrics or {})
        try:  # 只收 JSON 原生值；NaN / 无穷大不静默变成 null
            _canonical(metrics)
        except (TypeError, ValueError) as e:
            raise ValueError(f"metrics 不能写进模型文件：{e}") from None
        meta = ClassifierMeta(
            trained_at=trained_at,
            questions_version=ROUTER_QUESTIONS_VER,
            samples=fitted.samples,
            seed_examples=fitted.seed_examples,
            metrics=metrics,
        )
        classes = list(fitted.model.classes)
        artifact: dict[str, Any] = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_VERSION,
            "model": {
                "kind": MODEL_KIND,
                "features": FEATURES,
                "alpha": self.alpha,
                "temperature": self.temperature,
                "classes": [c.value for c in classes],
                "doc_counts": {c.value: fitted.counts.doc_counts[c] for c in classes},
                "feature_counts": {
                    c.value: dict(sorted(fitted.counts.feature_counts[c].items())) for c in classes
                },
            },
            "metadata": meta.model_dump(mode="json"),
        }
        artifact["sha256"] = artifact_digest(artifact)
        return artifact

    @classmethod
    def from_artifact(
        cls,
        artifact: Mapping[str, Any],
        *,
        fallback: DecisionProvider,
        name: str = "classifier",
    ) -> ClassifierProvider:
        """从模型文件的内容重建（先完整校验，见模块说明）。不对抛 ClassifierArtifactError。"""
        parsed = _validate_artifact(dict(artifact))
        params = parsed.model
        counts = _Counts(
            doc_counts={c: params.doc_counts[c] for c in params.classes},
            feature_counts={c: dict(params.feature_counts[c]) for c in params.classes},
        )
        provider = cls(fallback, alpha=params.alpha, temperature=params.temperature, name=name)
        provider._fitted = _Fitted(
            _build(counts, params.alpha),
            counts,
            parsed.metadata.samples,
            parsed.metadata.seed_examples,
        )
        provider.meta = parsed.metadata
        provider.artifact_sha256 = parsed.sha256
        return provider

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        fallback: DecisionProvider,
        name: str = "classifier",
    ) -> ClassifierProvider:
        """读模型文件（read_artifact）并重建。不对抛 ClassifierArtifactError。"""
        return cls.from_artifact(read_artifact(path), fallback=fallback, name=name)

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        trained_at: datetime,
        metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """导出并原子写入模型文件，返回写入的内容。覆盖已有文件的策略由调用方决定。"""
        artifact = self.to_artifact(trained_at=trained_at, metrics=metrics)
        write_artifact(path, artifact)
        self.meta = ClassifierMeta.model_validate(artifact["metadata"])
        self.artifact_sha256 = artifact["sha256"]
        return artifact
