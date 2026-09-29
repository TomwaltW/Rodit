"""路由分类器训练任务（§6.4 Phase 2）：去重、分层留出、和关键词规则对比、留出集校准、模型文件。

训练标注全部合成：合成评测集开发集（rodit.eval.dataset，dev）的问句加上固定的前后缀变体，
正确类型取用例的 expected_intent；评估用测试集，不参与训练。
"""

from __future__ import annotations

import json
import random
from datetime import timedelta

import pytest
import yaml

from rodit.core.config import RoutingConfig
from rodit.core.enums import INTENT_EXAMPLES, Intent
from rodit.core.models import LabeledSample
from rodit.decision import ClassifierProvider, RuleProvider
from rodit.decision.classifier import dump_artifact, feature_key
from rodit.eval.dataset import cases
from rodit.eval.recalibrate import REGRESSION_REMINDER, apply_patch
from rodit.eval.train import (
    RECOMMENDED_SAMPLES,
    InsufficientTrainingData,
    dedupe,
    render_text,
    route_intent,
    split_holdout,
    train_classifier,
)
from rodit.testing.fixtures import DEFAULT_NOW, SHOP_A, SHOP_B

NOW = DEFAULT_NOW
PREFIXES = ("", "请问", "亲，", "你好，")
SUFFIXES = ("", "呢", "呀")


def synthetic_labels(
    prefixes=PREFIXES, suffixes=SUFFIXES, *, limit: int | None = None
) -> list[LabeledSample]:
    out: list[LabeledSample] = []
    for case in cases("dev"):
        for p in prefixes:
            for s in suffixes:
                k = len(out)
                out.append(
                    LabeledSample(
                        shop_id=SHOP_A if k % 2 else SHOP_B,
                        text=f"{p}{case.text}{s}",
                        predicted_intent=case.expected_intent,
                        predicted_confidence=0.9,
                        true_intent=case.expected_intent,
                        source="feedback_adopt",
                        created_at=NOW - timedelta(minutes=k),
                    )
                )
    return out[:limit] if limit else out


def _label(text: str, true: Intent | None, *, minutes: int = 0) -> LabeledSample:
    return LabeledSample(
        shop_id=SHOP_A,
        text=text,
        predicted_intent=Intent.OTHER,
        predicted_confidence=0.7,
        true_intent=true,
        source="feedback_reject" if true != Intent.OTHER else "feedback_adopt",
        created_at=NOW - timedelta(minutes=minutes),
    )


@pytest.fixture(scope="module")
def labels() -> list[LabeledSample]:
    return synthetic_labels()


@pytest.fixture(scope="module")
def result(labels):
    return train_classifier(labels, now=NOW, routing=RoutingConfig())


# ================================================================ 去重


def test_dedupe_merges_same_features_and_takes_majority():
    test_text = cases("test")[0].text
    labels = [
        _label("还有货吗？", Intent.STOCK),
        _label("还有货吗", Intent.STOCK, minutes=1),
        _label("还 有 货 吗 ！", Intent.PRICE_PROMO, minutes=2),  # 同一句话，少数派
        _label("能便宜点吗", Intent.PRICE_PROMO),
        _label("能便宜点吗?", Intent.OTHER),  # 打平：丢掉
        _label("我要退货", None),  # 没有正确类型：不能当标签
        _label("？？？", Intent.OTHER),  # 去掉标点什么都不剩
        _label(test_text, Intent.OTHER),  # 和评测集测试集原句相同：不进训练
    ]
    items, counts = dedupe(labels, exclude={feature_key(test_text)})
    assert [(i.key, i.intent) for i in items] == [("还有货吗", Intent.STOCK)]
    assert items[0].text == min("还有货吗？", "还有货吗", "还 有 货 吗 ！")
    assert counts["labels_total"] == 8
    assert counts["unlabeled"] == 2 and counts["usable"] == 6
    assert counts["duplicates_merged"] == 3
    assert counts["conflicting_texts"] == 2 and counts["dropped_ties"] == 1
    assert counts["excluded_eval_overlap"] == 1
    assert counts["unique_texts"] == 3


def test_dedupe_is_order_independent(labels):
    shuffled = list(labels)
    random.Random(7).shuffle(shuffled)
    assert dedupe(labels) == dedupe(shuffled)


# ================================================================ 分层留出


def test_split_is_stratified_deterministic_and_disjoint(labels):
    items, _ = dedupe(labels)
    train, holdout = split_holdout(items)
    assert not {i.key for i in train} & {i.key for i in holdout}
    assert len(train) + len(holdout) == len(items)
    for intent in Intent:
        n = sum(1 for i in items if i.intent == intent)
        h = sum(1 for i in holdout if i.intent == intent)
        if n >= 5:
            assert abs(h - n * 0.2) <= 0.5 + 1e-9, intent
    shuffled = list(items)
    random.Random(3).shuffle(shuffled)
    assert split_holdout(shuffled) == (train, holdout)
    # 换盐切分就变（说明确实按哈希排序，而不是按输入顺序）
    assert split_holdout(items, salt="other-salt") != (train, holdout)


def test_tiny_classes_keep_their_only_samples_for_training():
    items, _ = dedupe([_label("在吗", Intent.GREETING), _label("有券吗", Intent.PRICE_PROMO)])
    items += dedupe([_label("还有货吗", Intent.STOCK), _label("有现货吗", Intent.STOCK)])[0]
    train, holdout = split_holdout(items)
    assert holdout == [] and len(train) == 4


# ================================================================ 训练与评估


def test_training_report_compares_with_rules(result, labels):
    report = result.report
    data = report.data
    assert data.labels_total == len(labels) and data.usable == len(labels)
    assert data.train + data.holdout == data.samples
    assert data.holdout == sum(data.holdout_by_intent.values())
    overlap = sum(
        1
        for texts in INTENT_EXAMPLES.values()
        for t in texts
        if feature_key(t) in {feature_key(c.text) for c in cases("test")}
    )
    assert data.seed_examples == sum(len(t) for t in INTENT_EXAMPLES.values()) - overlap
    for cmp, n in ((report.holdout, data.holdout), (report.eval_set, len(cases("test")))):
        assert cmp.classifier.n == cmp.rules.n == n
        assert 0 <= cmp.classifier.accuracy <= 1 and 0 <= cmp.rules.accuracy <= 1
        assert cmp.classifier.per_intent and cmp.rules.per_intent
    assert report.holdout.classifier.accuracy >= 0.9  # 合成变体：同一问句的变体容易认
    assert data.samples > RECOMMENDED_SAMPLES and not report.forced
    assert report.thresholds.ratchet is False  # 新模型的留出集没有选择偏差


def test_eval_set_scoring_accepts_acceptable_intents(result):
    """合成评测集：预测在 expected / acceptable intents 里就算对（和 rodit.eval.scoring 一致）。"""
    provider = result.provider
    test_cases = cases("test")
    expected = sum(
        route_intent(provider, c.text)[0] in {c.expected_intent, *c.acceptable_intents}
        for c in test_cases
    ) / len(test_cases)
    assert result.report.eval_set.classifier.accuracy == round(expected, 4)
    rules = sum(
        route_intent(RuleProvider(), c.text)[0] in {c.expected_intent, *c.acceptable_intents}
        for c in test_cases
    ) / len(test_cases)
    assert result.report.eval_set.rules.accuracy == round(rules, 4)


def test_artifact_loads_and_predicts_like_the_trained_model(result, tmp_path):
    loaded = ClassifierProvider.from_artifact(result.artifact, fallback=RuleProvider())
    assert loaded.artifact_sha256 == result.report.artifact_sha256
    for case in cases("test"):
        assert loaded.predict_proba(case.text) == result.provider.predict_proba(case.text)
    meta = loaded.meta
    assert meta is not None and meta.trained_at == NOW
    assert meta.samples == result.report.data.train  # 保存的就是训练集上训出来的模型
    metrics = meta.metrics
    assert metrics["holdout"]["classifier"]["accuracy"] == result.report.holdout.classifier.accuracy
    assert metrics["thresholds"]["ratchet"] is False
    assert "artifact_sha256" not in metrics and "thresholds_yaml" not in metrics


def test_artifact_metadata_has_no_label_text(result):
    meta = json.dumps(result.artifact["metadata"], ensure_ascii=False)
    for case in cases("dev")[:30]:
        if len(case.text) >= 6:
            assert case.text not in meta


def test_threshold_yaml_validates_against_routing(result):
    data = yaml.safe_load(result.report.thresholds_yaml)
    merged = apply_patch(RoutingConfig(), data)
    for p in result.report.thresholds.changes():
        if p.kind == "assist":
            assert merged.assist_threshold_for(p.intent) == p.proposed
        else:
            assert merged.auto_threshold[p.intent] == p.proposed
            assert p.proposed >= merged.assist_threshold_for(p.intent)


def test_render_text(result):
    text = render_text(result.report)
    assert "只生成模型文件" in text and "关键词规则" in text and "留出集" in text
    assert result.report.artifact_sha256 in text
    assert "RODIT_CLASSIFIER_PATH" in text
    assert REGRESSION_REMINDER in text


# ================================================================ 可复现


def test_training_is_deterministic(labels, result):
    shuffled = list(labels)
    random.Random(11).shuffle(shuffled)
    again = train_classifier(shuffled, now=NOW, routing=RoutingConfig())
    assert dump_artifact(again.artifact) == dump_artifact(result.artifact)
    later = train_classifier(labels, now=NOW + timedelta(days=1), routing=RoutingConfig())

    def strip(artifact: dict) -> dict:
        out = json.loads(dump_artifact(artifact))
        out.pop("sha256")
        out["metadata"].pop("trained_at")
        return out

    assert later.artifact["sha256"] != result.artifact["sha256"]
    assert strip(later.artifact) == strip(result.artifact)


# ================================================================ 数据量


def test_below_floor_refused_unless_forced():
    few = synthetic_labels(prefixes=("",), suffixes=("",), limit=150)
    with pytest.raises(InsufficientTrainingData, match="少于下限 200"):
        train_classifier(few, now=NOW, routing=RoutingConfig())
    forced = train_classifier(few, now=NOW, routing=RoutingConfig(), force=True)
    assert forced.report.forced
    assert any("--force" in w for w in forced.report.warnings)
    assert not any("2,000" in w for w in forced.report.warnings)  # 已经有更强的警告
    ok = train_classifier(few, now=NOW, routing=RoutingConfig(), min_samples=100)
    assert not ok.report.forced
    assert any("2,000" in w for w in ok.report.warnings)  # 低于 §6.4 建议量：警告


def test_no_usable_labels_refused_even_when_forced():
    unlabeled = [_label("我要退货", None), _label("？？", Intent.OTHER)]
    with pytest.raises(InsufficientTrainingData, match="没有可用的标注"):
        train_classifier(unlabeled, now=NOW, routing=RoutingConfig(), force=True)


def test_bad_arguments():
    with pytest.raises(ValueError, match="时区"):
        train_classifier([], now=NOW.replace(tzinfo=None), routing=RoutingConfig())
    with pytest.raises(ValueError, match="min_samples"):
        train_classifier([], now=NOW, routing=RoutingConfig(), min_samples=0)
