"""python -m rodit.eval：在合成评测集上跑本地演示单元（§13），不崩、报告齐全、指标不低于基线。

下限比实测基线略低（留一两条用例的余量），用来发现回归（§13.5 指标不下降才上线），不是上线门槛。
实测基线（离线大模型 + 关键词规则路由 + 启发式质检；评测集扩充到 324 条后）：

| 评测集 | 条数 | 类型准确率 | 路由 | 出推荐 | 事实正确 | 切题 | 合规 | 表达 | 失败 | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| dev  | 258 | 78.7% | 83.7% | 79.5% | 58.5% | 76.7% | 100% | 85.7% | 118 | 0.056 |
| test | 66  | 81.8% | 84.8% | 83.3% | 65.2% | 78.8% | 100% | 90.9% | 30  | 0.037 |
| all  | 324 | 79.3% | 84.0% | 80.2% | 59.9% | 77.2% | 100% | 86.7% | 148 | 0.050 |

测试集门槛（事实正确、合规满分率 ≥ 95%）未通过：离线替身只会复述依据里最像的一句，
扩充的用例里多问题、依赖历史的追问、依据不足和售后细分场景更多，离线替身答不全。
（初版 81 条上的结果不变：dev 66 条、test 15 条的失败清单和扩充前一样。）
"""

from __future__ import annotations

import io
import json
import runpy
import sys

import pytest

from rodit.core.enums import Intent
from rodit.eval.cli import EvalCliReport, evaluate, main
from rodit.eval.dataset import cases

FLOORS: dict[str, dict[str, float]] = {
    "dev": {
        "intent_accuracy": 0.77,
        "action_accuracy": 0.82,
        "suggestion_rate": 0.78,
        "factual": 0.57,
        "on_topic": 0.75,
        "compliance": 1.0,
        "expression": 0.84,
    },
    "test": {
        "intent_accuracy": 0.79,
        "action_accuracy": 0.81,
        "suggestion_rate": 0.80,
        "factual": 0.62,
        "on_topic": 0.75,
        "compliance": 1.0,
        "expression": 0.87,
    },
    "all": {
        "intent_accuracy": 0.78,
        "action_accuracy": 0.83,
        "suggestion_rate": 0.79,
        "factual": 0.59,
        "on_topic": 0.76,
        "compliance": 1.0,
        "expression": 0.85,
    },
}
MAX_FAILURES = {"dev": 121, "test": 32, "all": 152}
MAX_ECE = 0.10  # 全部用例上的期望校准误差


@pytest.fixture(scope="module")
def report_all() -> EvalCliReport:
    return evaluate("all")


def _run(*argv: str) -> str:
    out = io.StringIO()
    assert main(list(argv), out=out) == 0
    return out.getvalue()


@pytest.mark.parametrize("split", ["dev", "test"])
def test_json_report_meets_floors(split: str):
    data = json.loads(_run("--split", split, "--json"))
    assert data["split"] == split and data["n"] == len(cases(split))
    overall = data["overall"]
    floors = FLOORS[split]
    for key in ("intent_accuracy", "action_accuracy", "suggestion_rate"):
        assert overall[key] >= floors[key], (split, key, overall[key])
    for dim in ("factual", "on_topic", "compliance", "expression"):
        assert overall["full_marks"][dim] >= floors[dim], (split, dim, overall["full_marks"][dim])
    assert len(data["failures"]) <= MAX_FAILURES[split]
    for failure in data["failures"]:
        assert failure["problems"] or min(failure["scores"].values()) < 2
    if split == "dev":
        assert data["gate"] is None  # 门槛只看测试集
        assert data["calibration"]["samples_split"] == "dev"
    else:
        gate = data["gate"]
        assert gate["n"] == len(cases("test"))
        assert gate["passes"] == (gate["factual"] >= 0.95 and gate["compliance"] >= 0.95)


def test_all_split_report(report_all: EvalCliReport):
    r = report_all
    assert r.n == len(cases("all")) == sum(r.split_sizes.values())
    assert set(r.by_split) == {"dev", "test"}
    for split in ("dev", "test", "all"):
        summary = r.overall if split == "all" else r.by_split[split]
        assert summary.intent_accuracy >= FLOORS[split]["intent_accuracy"]
        assert summary.full_marks["factual"] >= FLOORS[split]["factual"]
        assert summary.full_marks["compliance"] >= FLOORS[split]["compliance"]
    assert len(r.failures) <= MAX_FAILURES["all"]
    # 按问题类型：20 类都有用例
    assert set(r.by_intent) == {i.value for i in Intent}
    assert sum(s.n for s in r.by_intent.values()) == r.n
    # 门槛判定只看测试集
    assert r.gate is not None and r.gate.n == r.split_sizes["test"]
    assert r.gate.factual == r.by_split["test"].full_marks["factual"]
    # §13.3：只用开发集校准；给出的阈值都达到目标准确率
    cal = r.calibration
    assert cal.samples_split == "dev" and cal.target_precision == pytest.approx(0.90)
    assert cal.n_samples <= r.split_sizes["dev"]
    for pick in cal.thresholds.values():
        if pick is not None:
            assert pick.precision >= 0.90 and 0.5 <= pick.threshold < 1.0
    assert r.reliability.n_samples > 0 and 0.0 <= r.reliability.ece <= MAX_ECE
    assert sum(b.count for b in r.reliability.bins) == r.reliability.n_samples


def test_fixed_cases_stay_fixed(report_all: EvalCliReport):
    """端到端验证修掉的问题不再出现在失败清单里：
    refund-01（同一订单多张售后单时回答最近更新的那张）、inv-03（标注与演示世界的政策原文矛盾）。"""
    failed = {f.id for f in report_all.failures}
    assert "refund-01" not in failed
    assert "inv-03" not in failed


def test_text_report_sections():
    text = _run("--split", "all")
    for section in (
        "== 总体",
        "== §13.2 上线门槛",
        "== 按问题类型",
        "== §13.3 智能辅助阈值",
        "ECE =",
        "== 失败清单",
    ):
        assert section in text, section
    assert "测试集 n=" in text


def test_module_entry_point(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["python -m rodit.eval", "--split", "test"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("rodit.eval", run_name="__main__")
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "上线门槛" in out and "仅供参考" in out  # 只跑测试集时阈值用测试集算，要提示


# ================================================================ --classifier（§6.4、§13.5）


def _train_classifier(path) -> None:
    """用开发集的合成问句训练一个分类器（只为测试接线；真实训练见 rodit.eval.train）。"""
    from datetime import datetime

    from rodit.core.clock import CST
    from rodit.decision import ClassifierProvider, RuleProvider

    pairs = [(c.text, c.expected_intent) for c in cases("dev")]
    model = ClassifierProvider(RuleProvider()).fit(pairs, seed_examples=True)
    model.save(path, trained_at=datetime(2026, 9, 28, 10, 0, tzinfo=CST))


def test_classifier_option_routes_with_the_trained_model(tmp_path):
    target = tmp_path / "classifier.json"
    _train_classifier(target)
    out = io.StringIO()
    argv = ["--split", "test", "--limit", "12", "--classifier", str(target), "--json"]
    assert main(argv, out=out) == 0
    report = json.loads(out.getvalue())
    assert report["llm"]["decision_provider"] == "classifier"
    assert report["cases"] and all(c["predicted_intent"] for c in report["cases"])


def test_classifier_option_rejects_a_broken_model_file(tmp_path):
    target = tmp_path / "classifier.json"
    _train_classifier(target)
    target.write_text(target.read_text(encoding="utf-8").replace('"alpha"', '"alpha2"'))
    err = io.StringIO()
    assert main(["--classifier", str(target)], out=io.StringIO(), err=err) == 2
    assert "路由分类器模型文件不能用" in err.getvalue()
