"""路由分类器的训练任务（方案 §6.4 Phase 2 "用反馈数据训练路由小模型分类器"、§13.3、§13.5）。

离线运行，结果由人确认后才上线（python -m rodit.ops train-classifier，见 rodit.ops.serve）：

1. 标注：FeedbackRepository.export_training_labels()（本单元所有店铺；PostgreSQL 上用
   rodit_trainer 角色只读 labeled_sample，这是唯一的跨店读路径）；只用有正确类型（true_intent）的，
   理解错误但没选正确类型的只知道"不是什么"，不能当标签；
2. 去重：特征相同的文本（decision.classifier.feature_key：脱敏、归一、去掉空白和标点）合成一条，
   标签取多数，多数打平的丢掉；和合成评测集测试集原句相同的不进训练（免得评测分数虚高）；
3. 分层留出：每个类型按 sha256(固定盐 + 文本) 排序，前 20% 留出、其余训练。不用全局随机数：
   同样的标注总是同样的切分，和导出的先后顺序无关；
4. 训练：训练集 + §5.1 例句（去掉和评测集测试集重复的）；
5. 评估：留出集和合成评测集的测试集（§13.1 留出的 20%，预测在 expected / acceptable intents 里
   算对），都和关键词规则（RuleProvider）比。两边都走线上同样的 decide()（Jev 格式请求，只问
   intent），置信度和线上路由看到的一致；
6. 校准：在留出集的预测上按 §13.3 扫描阈值（rodit.eval.recalibrate.plan_thresholds，
   ratchet=False：新模型对每条留出样本都给了置信度，没有"展示过才有标注"的选择偏差，
   所以可以下调，但仍不低于最低置信度），输出和阈值重算同样格式的配置片段，跟模型文件一起上线；
7. 模型文件：保存的就是在训练集上训出来、在留出集上评估和校准过的那个模型（不再用全部数据重训：
   重训后的概率和校准用的不是同一个模型）。metadata 里有训练时间、样本数、问题定义版本和本报告
   （只有计数和比率，不含任何文本）。

数据量：去重后的可用样本少于 2,000 条时警告（§6.4 "积累 2,000–5,000 条带标签数据后"），
少于下限（默认 200）拒绝训练，除非 force（只用于试验，不要上线）。

模型文件里是脱敏后买家问题的字符 1–2 元组计数（mask_pii 在 fit 里做）：不含原句，但罕见的
二元组可能是原文片段，所以模型文件只能留在平台单元内，不拷出平台云（§4.1 原则 4、§12.3）。
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.core.clock import CST
from rodit.core.config import RoutingConfig
from rodit.core.decision import DecisionProvider, DecisionRequest
from rodit.core.enums import INTENT_EXAMPLES, INTENT_LABELS, Intent, Role
from rodit.core.models import LabeledSample
from rodit.core.pii import mask_pii
from rodit.core.questions import Q_INTENT, ROUTER_MODEL, ROUTER_QUESTIONS
from rodit.decision.classifier import ClassifierProvider, feature_key
from rodit.decision.rules import RuleProvider
from rodit.eval.calibration import Sample, reliability
from rodit.eval.dataset import cases
from rodit.eval.recalibrate import (
    DEFAULT_MIN_SUPPORT,
    REGRESSION_REMINDER,
    ThresholdPlan,
    plan_thresholds,
    proposal_yaml,
    render_plan,
)

RECOMMENDED_SAMPLES = 2000  # §6.4 积累 2,000–5,000 条带标签数据
DEFAULT_MIN_SAMPLES = 200  # 低于这个数拒绝训练（除非 force）
HOLDOUT_FRACTION = 0.2
SPLIT_SALT = "rodit.route_classifier.split.v1"  # 改了盐切分就变：只在换切分规则时改
FEW_TRAINING_SAMPLES = 10  # 训练样本少于这个数的类型单独提醒


class InsufficientTrainingData(ValueError):
    """可用的标注不够，拒绝训练（数量见消息）。"""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSummary(_Model):
    labels_total: int  # 导出的全部标注
    unlabeled: int  # 没有正确类型（理解错误但没选），不能当标签
    usable: int  # 有正确类型的
    unique_texts: int  # 去重后的文本数（含下面丢掉 / 排除的）
    duplicates_merged: int  # 合进别的标注的重复条数
    conflicting_texts: int  # 同一句话被标成不同类型的文本数（取多数）
    dropped_ties: int  # 多数打平、丢掉的文本数
    excluded_eval_overlap: int  # 和评测集测试集原句相同、不进训练的文本数
    samples: int  # 最终参与训练 + 留出的文本数
    train: int
    holdout: int
    seed_examples: int  # 加进训练的 §5.1 例句条数
    train_by_intent: dict[str, int]
    holdout_by_intent: dict[str, int]


class IntentScore(_Model):
    n_true: int  # 真实为该类型的条数
    n_pred: int  # 预测为该类型的条数
    precision: float | None
    recall: float | None
    f1: float | None


class ProviderScore(_Model):
    n: int
    accuracy: float | None
    macro_f1: float | None
    ece: float | None  # 期望校准误差（rodit.eval.calibration.reliability，5 箱）
    per_intent: dict[str, IntentScore]


class Comparison(_Model):
    classifier: ProviderScore
    rules: ProviderScore  # 关键词规则基线（RuleProvider）


class TrainReport(_Model):
    trained_at: AwareDatetime
    alpha: float
    temperature: float
    min_samples: int
    forced: bool
    data: DataSummary
    holdout: Comparison
    eval_set: Comparison  # 合成评测集（rodit.eval.dataset）的测试集
    thresholds: ThresholdPlan
    thresholds_yaml: str
    warnings: list[str] = Field(default_factory=list)
    artifact_sha256: str | None = None


@dataclass
class TrainResult:
    provider: ClassifierProvider
    artifact: dict[str, Any]
    report: TrainReport


@dataclass(frozen=True)
class _Item:
    key: str  # feature_key
    text: str
    intent: Intent


# ================================================================ 数据准备


def _eval_keys() -> set[str]:
    return {feature_key(c.text) for c in cases("test")}


def dedupe(
    labels: Iterable[LabeledSample], *, exclude: set[str] | None = None
) -> tuple[list[_Item], dict[str, int]]:
    """有正确类型的标注按 feature_key 去重，标签取多数（打平的丢掉）；exclude 里的文本不要。

    代表文本取组里字典序最小的原文（组内各原文的特征完全相同，取哪条都一样，取最小的保证确定性）。
    返回按 key 排序的样本和计数（见 DataSummary 的同名字段）。"""
    exclude = exclude or set()
    groups: dict[str, list[tuple[str, Intent]]] = defaultdict(list)
    counts = Counter[str]()
    for label in labels:
        counts["labels_total"] += 1
        if label.true_intent is None:
            counts["unlabeled"] += 1
            continue
        counts["usable"] += 1
        key = feature_key(label.text)
        if not key:
            counts["unlabeled"] += 1  # 去掉标点后什么都不剩：没有特征，不能训练
            counts["usable"] -= 1
            continue
        groups[key].append((label.text, label.true_intent))
    items: list[_Item] = []
    for key in sorted(groups):
        members = groups[key]
        counts["duplicates_merged"] += len(members) - 1
        votes = Counter(intent for _, intent in members)
        ranked = votes.most_common()
        if len(ranked) > 1:
            counts["conflicting_texts"] += 1
            if ranked[0][1] == ranked[1][1]:
                counts["dropped_ties"] += 1
                continue
        if key in exclude:
            counts["excluded_eval_overlap"] += 1
            continue
        items.append(_Item(key=key, text=min(t for t, _ in members), intent=ranked[0][0]))
    counts["unique_texts"] = len(groups)
    return items, dict(counts)


def _rank(key: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}\x00{key}".encode()).hexdigest()


def split_holdout(
    items: Sequence[_Item], *, fraction: float = HOLDOUT_FRACTION, salt: str = SPLIT_SALT
) -> tuple[list[_Item], list[_Item]]:
    """分层留出：每个类型按 sha256(盐 + 文本) 排序，前 round(n × fraction) 条留出
    （至少给训练留 1 条；只有 1–2 条的类型不留出）。返回 (训练, 留出)，都按 key 排序。"""
    by_intent: dict[Intent, list[_Item]] = defaultdict(list)
    for item in items:
        by_intent[item.intent].append(item)
    train: list[_Item] = []
    holdout: list[_Item] = []
    for intent in Intent:
        members = sorted(by_intent.get(intent, []), key=lambda i: _rank(i.key, salt))
        n_hold = min(int(len(members) * fraction + 0.5), max(0, len(members) - 1))
        holdout += members[:n_hold]
        train += members[n_hold:]
    return sorted(train, key=lambda i: i.key), sorted(holdout, key=lambda i: i.key)


# ================================================================ 评估


def route_intent(provider: DecisionProvider, text: str) -> tuple[Intent, float]:
    """和线上路由一样问判断层（Jev 格式、只问 intent），返回预测类型和置信度。"""
    request = DecisionRequest(
        model=ROUTER_MODEL,
        state={"messages": [{"role": Role.BUYER.value, "text": mask_pii(text)}]},
        questions={Q_INTENT: ROUTER_QUESTIONS[Q_INTENT]},
    )
    answer = provider.decide(request).choice(Q_INTENT)
    return Intent(answer.choice), answer.confidence


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def score_provider(
    provider: DecisionProvider, rows: Sequence[tuple[str, Intent, frozenset[Intent]]]
) -> tuple[ProviderScore, list[Sample]]:
    """rows：(文本, 真实类型, 算对的类型)。返回得分和 (预测, 置信度, 真实) 三元组
    （预测在"算对的类型"里时，三元组的真实类型记成预测的类型）。"""
    samples: list[Sample] = []
    n_true = Counter[Intent]()
    n_pred = Counter[Intent]()
    hit_pred = Counter[Intent]()  # 预测为该类型且算对
    hit_true = Counter[Intent]()  # 真实为该类型且算对
    correct = 0
    for text, true, ok in rows:
        pred, conf = route_intent(provider, text)
        good = pred in ok
        correct += good
        n_true[true] += 1
        n_pred[pred] += 1
        if good:
            hit_pred[pred] += 1
            hit_true[true] += 1
        samples.append(
            Sample(predicted=pred.value, confidence=conf, true=(pred if good else true).value)
        )
    per_intent: dict[str, IntentScore] = {}
    f1s: list[float] = []
    for intent in Intent:
        if not (n_true[intent] or n_pred[intent]):
            continue
        precision = _ratio(hit_pred[intent], n_pred[intent])
        recall = _ratio(hit_true[intent], n_true[intent])
        f1 = None
        if precision is not None and recall is not None:
            f1 = (
                round(2 * precision * recall / (precision + recall), 4)
                if precision + recall
                else 0.0
            )
        f1s.append(f1 or 0.0)
        per_intent[intent.value] = IntentScore(
            n_true=n_true[intent], n_pred=n_pred[intent], precision=precision, recall=recall, f1=f1
        )
    ece = reliability(samples)[1] if samples else None
    score = ProviderScore(
        n=len(rows),
        accuracy=_ratio(correct, len(rows)),
        macro_f1=round(sum(f1s) / len(f1s), 4) if f1s else None,
        ece=ece,
        per_intent=per_intent,
    )
    return score, samples


def _compare(
    classifier: ClassifierProvider, rows: Sequence[tuple[str, Intent, frozenset[Intent]]]
) -> tuple[Comparison, list[Sample]]:
    clf_score, clf_samples = score_provider(classifier, rows)
    rules_score, _ = score_provider(RuleProvider(), rows)
    return Comparison(classifier=clf_score, rules=rules_score), clf_samples


# ================================================================ 训练


def train_classifier(
    labels: Iterable[LabeledSample],
    *,
    now: datetime,
    routing: RoutingConfig,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    force: bool = False,
    min_support: int = DEFAULT_MIN_SUPPORT,
    seed_examples: bool = True,
    alpha: float = 1.0,
    temperature: float = 1.0,
) -> TrainResult:
    """训练、评估、校准，返回模型、模型文件内容（未写盘）和报告。流程见模块说明。

    数据不够时抛 InsufficientTrainingData（force 只能越过下限，没有任何可用标注时仍然拒绝）。"""
    if now.tzinfo is None:
        raise ValueError("now 需要带时区")
    if min_samples < 1:
        raise ValueError("min_samples 必须 ≥ 1")
    eval_keys = _eval_keys()
    items, counts = dedupe(labels, exclude=eval_keys)
    n = len(items)
    if n == 0:
        raise InsufficientTrainingData("没有可用的标注（需要客服确认过正确类型的 true_intent）")
    warnings: list[str] = []
    if n < min_samples:
        if not force:
            raise InsufficientTrainingData(
                f"去重后可用的标注只有 {n} 条，少于下限 {min_samples} 条：继续积累反馈标注"
                "（§6.4 建议 2,000–5,000 条），或者加 --force 强行训练（只用于试验）"
            )
        warnings.append(
            f"可用标注 {n} 条，低于下限 {min_samples} 条，按 --force 强行训练：只用于试验，不要上线"
        )
    elif n < RECOMMENDED_SAMPLES:
        warnings.append(
            f"可用标注 {n} 条（去重后），少于 §6.4 建议的 2,000 条："
            "模型可能不稳定，上线前务必跑回归"
        )
    if counts.get("conflicting_texts"):
        warnings.append(
            f"{counts['conflicting_texts']} 条文本被标成了不同的类型（取多数，打平的丢掉）："
            "客服的标注口径可能不一致"
        )

    train, holdout = split_holdout(items)
    seeds = [
        (text, intent)
        for intent, texts in INTENT_EXAMPLES.items()
        for text in texts
        if feature_key(text) not in eval_keys
    ]
    provider = ClassifierProvider(RuleProvider(), alpha=alpha, temperature=temperature)
    provider.fit(
        [(i.text, i.intent) for i in train], seed_examples=seeds if seed_examples else False
    )
    if len(provider.classes) < 2:
        raise InsufficientTrainingData("训练数据只有一个问题类型，训不出分类器")
    train_by_intent = Counter(i.intent for i in train)
    few = [i.value for i in Intent if 0 < train_by_intent[i] < FEW_TRAINING_SAMPLES]
    if few:
        warnings.append(
            f"这些类型的训练样本不到 {FEW_TRAINING_SAMPLES} 条，预测不稳：{'、'.join(few)}"
        )

    holdout_rows = [(i.text, i.intent, frozenset({i.intent})) for i in holdout]
    holdout_cmp, holdout_samples = _compare(provider, holdout_rows)
    eval_rows = [
        (c.text, c.expected_intent, frozenset({c.expected_intent, *c.acceptable_intents}))
        for c in cases("test")
    ]
    eval_cmp, _ = _compare(provider, eval_rows)
    for name, cmp in (("留出集", holdout_cmp), ("合成评测集测试集", eval_cmp)):
        clf, rules = cmp.classifier.accuracy, cmp.rules.accuracy
        if clf is not None and rules is not None and clf < rules:
            warnings.append(f"{name}上分类器准确率 {clf:.1%} 低于关键词规则 {rules:.1%}")

    plan = plan_thresholds(holdout_samples, routing, min_support=min_support, ratchet=False)
    header = [
        f"路由分类器（训练时间 {now.astimezone(CST):%Y-%m-%d %H:%M}）"
        "在留出集上按 §13.3 校准的阈值：和模型文件一起上线（decision.provider: classifier）"
    ]
    data = DataSummary(
        labels_total=counts.get("labels_total", 0),
        unlabeled=counts.get("unlabeled", 0),
        usable=counts.get("usable", 0),
        unique_texts=counts.get("unique_texts", 0),
        duplicates_merged=counts.get("duplicates_merged", 0),
        conflicting_texts=counts.get("conflicting_texts", 0),
        dropped_ties=counts.get("dropped_ties", 0),
        excluded_eval_overlap=counts.get("excluded_eval_overlap", 0),
        samples=n,
        train=len(train),
        holdout=len(holdout),
        seed_examples=len(seeds) if seed_examples else 0,
        train_by_intent={i.value: train_by_intent[i] for i in Intent if train_by_intent[i]},
        holdout_by_intent={
            i.value: c for i in Intent if (c := sum(1 for h in holdout if h.intent == i))
        },
    )
    report = TrainReport(
        trained_at=now,
        alpha=alpha,
        temperature=temperature,
        min_samples=min_samples,
        forced=force and n < min_samples,
        data=data,
        holdout=holdout_cmp,
        eval_set=eval_cmp,
        thresholds=plan,
        thresholds_yaml=proposal_yaml(plan, routing, header=header),
        warnings=warnings,
    )
    metrics = report.model_dump(
        mode="json", exclude={"trained_at", "artifact_sha256", "thresholds_yaml"}
    )
    artifact = provider.to_artifact(trained_at=now, metrics=metrics)
    report.artifact_sha256 = artifact["sha256"]
    return TrainResult(provider=provider, artifact=artifact, report=report)


# ================================================================ 文字报告


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _score_line(name: str, s: ProviderScore) -> str:
    return (
        f"{name} 准确率 {_pct(s.accuracy)}、macro-F1 {_num(s.macro_f1)}、期望校准误差 {_num(s.ece)}"
    )


def render_text(report: TrainReport) -> str:
    """中文摘要：数据、切分、警告、和关键词规则的对比、按类型的表现、阈值建议和上线步骤。"""
    lines: list[str] = []
    add = lines.append
    d = report.data
    add("§6.4 路由分类器训练（朴素贝叶斯占位实现）：只生成模型文件，不改任何线上配置")
    add(f"训练时间：{report.trained_at.astimezone(CST):%Y-%m-%d %H:%M}（东八区）")
    add(
        f"标注：导出 {d.labels_total} 条，有正确类型 {d.usable} 条"
        f"（没有的 {d.unlabeled} 条不能当标签）；"
        f"去重后 {d.unique_texts} 条文本（合并重复 {d.duplicates_merged} 条；标注不一致 "
        f"{d.conflicting_texts} 条取多数，打平丢掉 {d.dropped_ties} 条；和评测集测试集相同 "
        f"{d.excluded_eval_overlap} 条不进训练）"
    )
    add(
        f"切分：训练 {d.train} 条 + §5.1 例句 {d.seed_examples} 条，留出 {d.holdout} 条"
        f"（每个类型 {HOLDOUT_FRACTION:.0%}，按固定盐的哈希排序，可复现）"
    )
    if report.warnings:
        add("警告：")
        for w in report.warnings:
            add(f"  - {w}")
    for title, cmp in (
        (f"留出集（{report.holdout.classifier.n} 条）", report.holdout),
        (f"合成评测集测试集（{report.eval_set.classifier.n} 条，§13.1）", report.eval_set),
    ):
        add(f"{title}：")
        add(f"  {_score_line('分类器', cmp.classifier)}")
        add(f"  {_score_line('关键词规则', cmp.rules)}")
    per_intent = report.holdout.classifier.per_intent
    if per_intent:
        add("各类型（留出集，分类器）：")
        for key, s in per_intent.items():
            intent = Intent(key)
            add(
                f"  - {key}（{INTENT_LABELS[intent]}）：真实 {s.n_true} 条、预测 {s.n_pred} 条，"
                f"准确率 {_pct(s.precision)}，召回率 {_pct(s.recall)}"
            )
    add("")
    render_plan(report.thresholds, add)
    add("")
    add("阈值配置片段（和模型文件一起上线）：")
    lines.extend(report.thresholds_yaml.rstrip("\n").splitlines())
    add("")
    if report.artifact_sha256:
        add(f"模型文件校验和：sha256 {report.artifact_sha256}")
    add(
        "上线：模型文件放在平台单元内（不拷出平台云），部署配置 decision.provider 改成 classifier、"
        "合并上面的阈值片段，环境变量 RODIT_CLASSIFIER_PATH 指向模型文件；质检仍用大模型（§6.4）"
    )
    add(REGRESSION_REMINDER)
    add(
        "注意：python -m rodit.eval 还不能加载模型文件跑整条链路（路由只有关键词规则或大模型），"
        "上面留出集和评测集的对比只覆盖问题类型识别"
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MIN_SAMPLES",
    "HOLDOUT_FRACTION",
    "RECOMMENDED_SAMPLES",
    "SPLIT_SALT",
    "Comparison",
    "DataSummary",
    "InsufficientTrainingData",
    "IntentScore",
    "ProviderScore",
    "TrainReport",
    "TrainResult",
    "dedupe",
    "render_text",
    "route_intent",
    "score_provider",
    "split_holdout",
    "train_classifier",
]
