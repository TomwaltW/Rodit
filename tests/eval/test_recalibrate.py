"""§13.5 每月用采纳反馈重算阈值（§13.3 算法）。

标注全部是合成的：文本是编号的占位句（不含真实买家数据），置信度按固定种子生成。
"""

from __future__ import annotations

import random
from datetime import timedelta

import pytest
import yaml

from rodit.core.config import EngineConfig, RoutingConfig
from rodit.core.enums import Intent
from rodit.core.models import LabeledSample
from rodit.eval.calibration import Sample, pick_threshold
from rodit.eval.recalibrate import (
    REGRESSION_REMINDER,
    UNKNOWN_TRUE,
    ProposalStatus,
    apply_patch,
    build_samples,
    plan_thresholds,
    proposal_patch,
    proposal_yaml,
    recalibrate,
    render_text,
    wilson_lower,
)
from rodit.testing.fixtures import DEFAULT_NOW, SHOP_A, SHOP_B

X = Intent.SIZE_ADVICE  # 不是高风险、没开全自动
Y = Intent.STOCK
AUTO = Intent.SHIPPING_POLICY  # 默认开放全自动（0.95）
NOW = DEFAULT_NOW


def _label(
    predicted: Intent | None,
    confidence: float | None,
    *,
    true: Intent | None = None,
    source: str = "feedback_adopt",
    days_ago: float = 1,
    shop: str = SHOP_A,
    text: str = "合成问句",
) -> LabeledSample:
    if source == "feedback_adopt":
        true = predicted
    return LabeledSample(
        shop_id=shop,
        text=text,
        predicted_intent=predicted,
        predicted_confidence=confidence,
        true_intent=true,
        source=source,  # type: ignore[arg-type]
        created_at=NOW - timedelta(days=days_ago),
    )


def _samples(intent: Intent, rows: list[tuple[float, bool]]) -> list[Sample]:
    """(置信度, 是否理解对) → 三元组；理解错的记成"不知道正确类型"的误判。"""
    return [Sample(intent.value, c, intent.value if ok else UNKNOWN_TRUE) for c, ok in rows]


def _assist(plan, intent: Intent):
    return next(p for p in plan.assist if p.intent == intent)


def _auto(plan, intent: Intent):
    return next(p for p in plan.auto if p.intent == intent)


def _spread(n: int, lo: float, hi: float) -> list[float]:
    return [round(lo + (hi - lo) * i / max(1, n - 1), 6) for i in range(n)]


# ================================================================ 标注 → 三元组


def test_build_samples_follows_label_sources():
    labels = [
        _label(X, 0.8),  # 采纳 → (X, 0.8, X)
        _label(X, 0.7, true=Y, source="feedback_reject"),  # 理解错误，选了正确类型
        _label(X, 0.65, true=None, source="feedback_reject"),  # 理解错误，没选
        _label(Y, 0.9, true=Y, source="manual", shop=SHOP_B),  # 人工，带置信度
        _label(Y, 0.9, true=None, source="manual"),  # 人工但没有正确类型：跳过
        _label(X, None),  # 旧数据没有置信度：跳过
        _label(None, 0.9, true=X, source="manual"),  # 没有预测类型：跳过
        _label(X, 0.99, days_ago=31),  # 时间窗外
    ]
    samples, stats = build_samples(labels, since=NOW - timedelta(days=30), until=NOW)
    assert samples == [
        Sample("size_advice", 0.8, "size_advice"),
        Sample("size_advice", 0.7, "stock"),
        Sample("size_advice", 0.65, UNKNOWN_TRUE),
        Sample("stock", 0.9, "stock"),
    ]
    assert (stats.total, stats.in_window, stats.used) == (8, 7, 4)
    assert stats.by_source == {"feedback_adopt": 1, "feedback_reject": 2, "manual": 1}
    assert stats.reject_unknown_true == 1
    assert (stats.skipped_no_intent, stats.skipped_no_confidence) == (1, 1)
    assert stats.skipped_manual_unlabeled == 1
    assert stats.shops == 2


def test_edited_labels_count_as_correct_and_non_assist_rules_are_skipped():
    """编辑后发送 = 类型对了；记了路由规则的标注只用规则 6（安抚、分条不看辅助阈值）。"""
    edited = _label(X, 0.75, true=X, source="feedback_edit")
    soothe = _label(X, 0.3, true=None, source="feedback_reject").model_copy(
        update={"route_rule": 3}
    )
    split = _label(X, 0.4, true=Y, source="feedback_reject").model_copy(update={"route_rule": 5})
    rule6 = _label(X, 0.8).model_copy(update={"route_rule": 6})
    legacy = _label(X, 0.85)  # 旧标注没记规则：照常使用
    samples, stats = build_samples(
        [edited, soothe, split, rule6, legacy], since=NOW - timedelta(days=30), until=NOW
    )
    assert samples == [
        Sample("size_advice", 0.75, "size_advice"),
        Sample("size_advice", 0.8, "size_advice"),
        Sample("size_advice", 0.85, "size_advice"),
    ]
    assert stats.skipped_other_rule == 2 and stats.used == 3
    assert stats.by_source == {"feedback_adopt": 2, "feedback_edit": 1}


def test_window_is_half_open():
    since, until = NOW - timedelta(days=30), NOW
    at_since = _label(X, 0.8, days_ago=30)
    at_until = _label(X, 0.8, days_ago=0)
    samples, stats = build_samples([at_since, at_until], since=since, until=until)
    assert len(samples) == 1 and stats.in_window == 1


def test_unknown_true_reject_is_a_miss_without_support():
    rows = _samples(X, [(0.8, True)] * 27 + [(0.8, False)] * 3)
    plan = plan_thresholds(rows, RoutingConfig(), min_support=30)
    p = _assist(plan, X)
    assert (p.samples, p.support) == (30, 27)  # 误判不给任何类型加真实样本
    assert p.precision == 0.9 and p.status == ProposalStatus.KEEP
    assert all(q.support == 0 for q in plan.assist if q.intent != X)


def test_reject_with_true_intent_supports_the_true_type():
    rows = _samples(X, [(0.9, True)] * 30) + [Sample(X.value, 0.9, Y.value)] * 2
    plan = plan_thresholds(rows, RoutingConfig())
    assert _assist(plan, Y).support == 2 and _assist(plan, Y).samples == 0
    assert _assist(plan, X).precision == round(30 / 32, 4)


# ================================================================ 选择偏差


def test_naive_scan_would_propose_below_observed_data():
    """问题本身：所有样本都 ≥ 0.62（展示过才有标注），直接扫描会"建议"0.50。"""
    rows = _samples(X, [(c, True) for c in _spread(60, 0.62, 1.0)])
    naive = pick_threshold(rows, X.value, 0.90)
    assert naive is not None and naive.threshold == 0.5


def test_ratchet_keeps_current_threshold_when_data_starts_above_it():
    rows = _samples(X, [(c, True) for c in _spread(60, 0.62, 1.0)])
    p = _assist(plan_thresholds(rows, RoutingConfig()), X)
    assert p.status == ProposalStatus.KEEP and p.proposed == 0.6
    assert p.min_confidence == 0.62 and p.picked == 60 and p.precision == 1.0


def test_ratchet_never_lowers_even_with_sub_threshold_labels():
    """当前阈值以下只有安抚 / 分条等路径留下的少量标注：不据此下调。"""
    rows = _samples(X, [(c, True) for c in _spread(60, 0.7, 1.0)])
    rows += _samples(X, [(c, True) for c in _spread(40, 0.3, 0.69)])
    routing = RoutingConfig(assist_threshold_by_intent={X: 0.7})
    p = _assist(plan_thresholds(rows, routing), X)
    assert p.status == ProposalStatus.KEEP and p.proposed == 0.7
    assert p.min_confidence == 0.3
    # 同样的数据当作新模型的留出集预测（没有选择偏差）：可以下调，但不低于网格下限和最低置信度
    fresh = _assist(plan_thresholds(rows, routing, ratchet=False), X)
    assert fresh.status == ProposalStatus.CHANGE and fresh.note == "下调"
    assert fresh.proposed == 0.5


def test_ratchet_raises_when_precision_drops():
    rows = _samples(X, [(c, i % 3 != 0) for i, c in enumerate(_spread(60, 0.6, 0.79))])
    rows += _samples(X, [(c, True) for c in _spread(60, 0.8, 1.0)])
    p = _assist(plan_thresholds(rows, RoutingConfig()), X)
    assert p.status == ProposalStatus.CHANGE and p.note == "上调"
    assert p.proposed > 0.6 and p.precision >= 0.9 and p.picked >= 30
    assert p.current_precision is not None and p.current_precision < 0.9
    assert p.proposed >= p.min_confidence


@pytest.mark.parametrize("seed", range(12))
def test_invariants_on_random_feedback(seed):
    """随机合成的反馈：建议值从不低于该类型的最低置信度；辅助阈值只升不降；
    全自动阈值不低于辅助阈值；每个修改都达标且阈值以上样本够数。"""
    rng = random.Random(seed)
    current = rng.choice([0.55, 0.6, 0.65, 0.7])
    routing = RoutingConfig(
        assist_threshold=current,
        assist_threshold_by_intent={Y: round(current + 0.05, 2)},
        auto_threshold={AUTO: rng.choice([0.9, 0.95, 0.99]), X: 0.97},
    )
    rows: list[Sample] = []
    for intent in (X, Y, AUTO):
        floor = routing.assist_threshold_for(intent)
        n = rng.randint(20, 150)
        skill = rng.uniform(0.3, 1.2)
        for _ in range(n):
            c = round(rng.uniform(floor, 1.0), 6)
            ok = rng.random() < min(1.0, skill * c + 0.1)
            true = intent.value if ok else rng.choice([UNKNOWN_TRUE, Intent.OTHER.value])
            rows.append(Sample(intent.value, c, true))
        for _ in range(rng.randint(0, 10)):  # 规则 3 / 5 留下的低置信度标注
            rows.append(Sample(intent.value, round(rng.uniform(0.2, floor), 6), intent.value))
    for ratchet in (True, False):
        plan = plan_thresholds(rows, routing, ratchet=ratchet)
        for p in plan.assist + plan.auto:
            if p.changed:
                assert p.proposed >= p.min_confidence - 1e-9
                assert p.proposed >= 0.5
                assert p.precision is not None and p.precision >= p.target_precision - 1e-9
                assert p.picked >= plan.min_support
            if ratchet and p.kind == "assist":
                assert p.proposed >= p.current
        for a in plan.auto:
            assist = _assist(plan, a.intent)
            if a.changed:
                assert a.proposed >= assist.proposed


# ================================================================ 样本不足、达不到目标


def test_min_support_keeps_current():
    few = _samples(X, [(0.8, True)] * 29)
    p = _assist(plan_thresholds(few, RoutingConfig()), X)
    assert p.status == ProposalStatus.INSUFFICIENT and p.proposed == 0.6
    assert "29" in p.note
    enough = _samples(X, [(0.8, True)] * 30)
    assert _assist(plan_thresholds(enough, RoutingConfig()), X).status == ProposalStatus.KEEP
    assert (
        _assist(plan_thresholds(enough, RoutingConfig(), min_support=31), X).status
        == ProposalStatus.INSUFFICIENT
    )
    with pytest.raises(ValueError):
        plan_thresholds(enough, RoutingConfig(), min_support=0)


def test_unreachable_target_keeps_current():
    rows = _samples(X, [(c, i % 2 == 0) for i, c in enumerate(_spread(80, 0.6, 1.0))])
    p = _assist(plan_thresholds(rows, RoutingConfig()), X)
    assert p.status == ProposalStatus.UNREACHABLE and p.proposed == 0.6
    assert "达不到" in p.note


def test_target_reachable_only_with_too_few_samples_is_unreachable():
    rows = _samples(X, [(c, i % 2 == 0) for i, c in enumerate(_spread(60, 0.6, 0.9))])
    rows += _samples(X, [(0.99, True)] * 10)  # 0.99 以上全对，但只有 10 条
    p = _assist(plan_thresholds(rows, RoutingConfig()), X)
    assert p.status == ProposalStatus.UNREACHABLE and "样本太少" in p.note


def test_ties_go_to_the_lowest_threshold():
    """覆盖率相同取更低的阈值（§13.3）：按条数精确比较，1/3 这类四舍五入往下舍的覆盖率
    不会把并列判给更高的阈值（重算和 pick_threshold 一致）。"""
    rows = [Sample(X.value, 0.55, Y.value), Sample(X.value, 0.95, X.value)]
    rows += [Sample(Y.value, 0.9, X.value)] * 2
    routing = RoutingConfig(assist_threshold_by_intent={X: 0.99})
    p = _assist(plan_thresholds(rows, routing, min_support=1, ratchet=False), X)
    assert p.status == ProposalStatus.CHANGE and p.proposed == 0.56
    assert p.picked == 1 and p.coverage == round(1 / 3, 4)
    assert pick_threshold(rows, X.value, 0.9).threshold == 0.56


def test_equivalent_threshold_prefers_current():
    """数据上和当前阈值等价的候选不算修改（不做没有依据的改动）。"""
    rows = _samples(AUTO, [(c, True) for c in _spread(60, 0.96, 1.0)])
    p = _auto(plan_thresholds(rows, RoutingConfig()), AUTO)
    assert p.status == ProposalStatus.KEEP and p.proposed == 0.95


# ================================================================ 全自动阈值


def test_auto_threshold_can_move_down_to_observed_data():
    rows = _samples(AUTO, [(c, True) for c in _spread(60, 0.7, 1.0)])
    p = _auto(plan_thresholds(rows, RoutingConfig()), AUTO)
    assert p.status == ProposalStatus.CHANGE and p.note == "下调"
    assert p.proposed == 0.7  # 最低置信度本身（不外推到没有数据的 0.6–0.7）
    assert p.current_picked < p.picked == 60
    assert p.precision_lower is not None and p.precision_lower < 0.98  # 60 条的置信下限


def test_auto_never_below_assist():
    rows = _samples(AUTO, [(c, True) for c in _spread(200, 0.6, 1.0)])
    routing = RoutingConfig(assist_threshold_by_intent={AUTO: 0.9})
    plan = plan_thresholds(rows, routing)
    assert _assist(plan, AUTO).proposed == 0.9
    auto = _auto(plan, AUTO)
    assert auto.status == ProposalStatus.CHANGE and auto.proposed == 0.9
    # 辅助阈值被上调时，全自动阈值跟着不低于它
    noisy = _samples(AUTO, [(c, i % 2 == 0) for i, c in enumerate(_spread(100, 0.6, 0.89))])
    noisy += _samples(AUTO, [(c, True) for c in _spread(60, 0.9, 1.0)])
    routing = RoutingConfig(auto_threshold={AUTO: 0.8})  # 比上调后的辅助阈值还低
    plan = plan_thresholds(noisy, routing)
    assist, auto = _assist(plan, AUTO), _auto(plan, AUTO)
    assert assist.changed and assist.proposed > 0.8
    assert auto.changed and auto.proposed >= assist.proposed
    assert auto.floor is not None and auto.floor >= assist.proposed


def test_auto_only_for_listed_intents():
    rows = _samples(X, [(0.99, True)] * 50)
    plan = plan_thresholds(rows, RoutingConfig())
    assert {p.intent for p in plan.auto} == set(RoutingConfig().auto_threshold)
    assert X not in {p.intent for p in plan.auto}


def test_rule_governed_intents_not_applicable():
    routing = RoutingConfig(auto_threshold={Intent.PRICE_PROMO: 0.99, AUTO: 0.95})
    rows = []
    for intent in (Intent.PRICE_PROMO, Intent.COMPLAINT, Intent.OFF_PLATFORM):
        rows += _samples(intent, [(c, i % 2 == 0) for i, c in enumerate(_spread(60, 0.3, 1.0))])
    plan = plan_thresholds(rows, routing)
    for intent in (Intent.PRICE_PROMO, Intent.COMPLAINT, Intent.OFF_PLATFORM):
        p = _assist(plan, intent)
        assert p.status == ProposalStatus.NOT_APPLICABLE and p.proposed == p.current
        assert p.samples == 60
    assert _auto(plan, Intent.PRICE_PROMO).status == ProposalStatus.NOT_APPLICABLE
    assert not plan.changes()


def test_wilson_lower_bound():
    assert wilson_lower(0, 0) is None
    assert wilson_lower(60, 60) == pytest.approx(0.9398, abs=1e-4)
    assert wilson_lower(980, 1000) > 0.96
    assert wilson_lower(5, 10) < 0.5


# ================================================================ 配置片段


def _plan_with_changes():
    rows = _samples(X, [(c, i % 3 != 0) for i, c in enumerate(_spread(60, 0.6, 0.79))])
    rows += _samples(X, [(c, True) for c in _spread(60, 0.8, 1.0)])
    rows += _samples(AUTO, [(c, True) for c in _spread(60, 0.7, 1.0)])
    return plan_thresholds(rows, RoutingConfig())


def test_yaml_contains_only_changes_and_round_trips():
    routing = EngineConfig().routing
    plan = _plan_with_changes()
    text = proposal_yaml(plan, routing, header=["测试"])
    assert text.startswith("# 测试\n") and REGRESSION_REMINDER in text
    data = yaml.safe_load(text)
    assert data == proposal_patch(plan)
    section = data["routing"]
    assert set(section["assist_threshold_by_intent"]) == {X.value}
    assert set(section["auto_threshold"]) == {AUTO.value}
    merged = apply_patch(routing, data)
    assert isinstance(merged, RoutingConfig)
    assert merged.assist_threshold_for(X) == _assist(plan, X).proposed
    assert merged.auto_threshold[AUTO] == _auto(plan, AUTO).proposed
    # 逐项合并：没列出的类型保持原值（全自动的其他 5 个类型不会丢）
    for intent, value in routing.auto_threshold.items():
        if intent != AUTO:
            assert merged.auto_threshold[intent] == value
    assert merged.assist_threshold == routing.assist_threshold
    # 合并后的片段也能作为完整配置加载
    EngineConfig.model_validate({"routing": merged.model_dump(mode="json")})


def test_yaml_without_changes_is_a_no_op():
    rows = _samples(X, [(0.9, True)] * 40)
    plan = plan_thresholds(rows, RoutingConfig())
    text = proposal_yaml(plan, RoutingConfig())
    assert "没有需要修改的阈值" in text
    assert yaml.safe_load(text) == {"routing": {}}
    assert apply_patch(RoutingConfig(), yaml.safe_load(text)) == RoutingConfig()


def test_apply_patch_validates():
    with pytest.raises(ValueError):
        apply_patch(RoutingConfig(), {"routing": {"auto_threshold": {X.value: 1.5}}})
    with pytest.raises(ValueError):
        apply_patch(RoutingConfig(), {"routing": {"assist_threshold_by_intent": {"nope": 0.7}}})
    with pytest.raises(ValueError):
        apply_patch(RoutingConfig(), {"routing": ["not", "a", "mapping"]})


# ================================================================ 重算入口与文字报告


def _feedback(days_ago: float = 1) -> list[LabeledSample]:
    labels = []
    for i, c in enumerate(_spread(60, 0.6, 0.79)):
        if i % 3:
            labels.append(_label(X, c, days_ago=days_ago, text=f"合成问句{i}"))
        else:
            labels.append(_label(X, c, true=None, source="feedback_reject", days_ago=days_ago))
    labels += [_label(X, c, days_ago=days_ago) for c in _spread(60, 0.8, 1.0)]
    return labels


def test_recalibrate_default_window_is_last_30_days():
    report = recalibrate(_feedback(days_ago=31) + _feedback(days_ago=29), RoutingConfig(), now=NOW)
    assert report.until == NOW and report.since == NOW - timedelta(days=30)
    assert report.labels.total == 240 and report.labels.in_window == 120
    assert [p.intent for p in report.changes] == [X]
    custom = recalibrate(
        _feedback(days_ago=31),
        RoutingConfig(),
        now=NOW,
        since=NOW - timedelta(days=40),
        until=NOW - timedelta(days=20),
    )
    assert custom.labels.in_window == 120


def test_recalibrate_rejects_bad_window():
    with pytest.raises(ValueError, match="早于"):
        recalibrate([], RoutingConfig(), now=NOW, since=NOW, until=NOW)
    with pytest.raises(ValueError, match="时区"):
        recalibrate([], RoutingConfig(), now=NOW.replace(tzinfo=None))


def test_render_text_is_a_reviewable_summary():
    labels = _feedback()
    labels.append(_label(X, 0.9, text="我手机13812345678，地址北京市朝阳区建国路88号"))
    report = recalibrate(labels, RoutingConfig(), now=NOW)
    text = render_text(report, config_source="config/rodit.yaml")
    assert "只是建议" in text and "config/rodit.yaml" in text
    assert "size_advice" in text and "上调" in text
    assert REGRESSION_REMINDER in text
    assert "只维持或上调" in text
    # 摘要只有计数和比率，没有标注原文
    for leaked in ("13812345678", "建国路", "合成问句"):
        assert leaked not in text and leaked not in report.yaml
