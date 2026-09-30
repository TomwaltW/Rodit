"""上线状态的测试集依据按 serve 实际加载的路由分类器核对（方案 §13.5 "升级判断层之前先在测试集上
跑回归"、§13.4 影子 → 智能辅助）：模型文件启动后被原地换掉（train-classifier --overwrite 是原子
改名），不重启的话线上路由用的还是加载的那个——给新文件跑的评测不能让门槛显示通过，给加载的那个
跑的评测照旧算。别的进程（运维任务）看不到 serve 加载的是什么，照旧读文件。

演示单元 + 合成数据（演示时钟在 2026-09-28 周一 10:35，W40），不联网。"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit
from rodit.app.rollout import eval_evidence, unit_classifier_sha256
from rodit.core.clock import CST
from rodit.core.config import EngineConfig
from rodit.core.enums import Intent, Platform, ProviderName, SuggestionStatus
from rodit.core.models import SuggestionRecord
from rodit.decision import ClassifierProvider, RuleProvider
from rodit.eval.dataset import cases
from rodit.eval.fingerprint import (
    CLASSIFIER_PATH_ENV,
    EvalFingerprint,
    EvalVerdict,
    current_fingerprint,
    file_sha256,
    verdict_event,
)
from rodit.testing.fakes import FakeLLMClient
from rodit.testing.fixtures import SHOP_A

REVIEWER = "qa_reviewer_01"
W39 = "2026-W39"
W39_START = datetime(2026, 9, 21, tzinfo=CST)
TRAINED_AT = datetime(2026, 9, 25, 10, 0, tzinfo=CST)
FULL = {"factual": 2, "relevance": 2, "compliance": 2, "expression": 2}
LLM_ENV = {
    "RODIT_LLM_BASE_URL": "https://llm.example.invalid/v1",
    "RODIT_LLM_MODEL": "synthetic-model",
}
DEV = [(c.text, c.expected_intent) for c in cases("dev")]


@pytest.fixture(autouse=True)
def running_llm(monkeypatch):
    for name, value in LLM_ENV.items():
        monkeypatch.setenv(name, value)


def classifier_config() -> EngineConfig:
    config = EngineConfig()
    decision = config.decision.model_copy(update={"provider": ProviderName.CLASSIFIER})
    return config.model_copy(update={"decision": decision})


def train(path: Path, *, extra: int = 0, days: int = 0) -> ClassifierProvider:
    """训练并原子写入模型文件（和 train-classifier 同一个写法）。extra 条合成问句换一个模型。"""
    pairs = DEV + [(f"合成问句{i}号能不能便宜点", Intent.COMPLAINT) for i in range(extra)]
    model = ClassifierProvider(RuleProvider()).fit(pairs, seed_examples=True)
    model.save(path, trained_at=TRAINED_AT + timedelta(days=days))
    return model


@pytest.fixture
def deployed(tmp_path, monkeypatch):
    """serve 启动时加载了分类器 A（RODIT_CLASSIFIER_PATH），之后一直用它路由。"""
    path = tmp_path / "classifier.json"
    train(path)
    monkeypatch.setenv(CLASSIFIER_PATH_ENV, str(path))
    loaded = ClassifierProvider.load(path, fallback=RuleProvider())
    unit = build_demo_unit(
        Platform.TAOBAO,
        config=classifier_config(),
        llm=FakeLLMClient(handler=lambda request: "{}", model="synthetic-model"),
        route_provider=loaded,
    )
    assert unit.router.provider is loaded
    return AssistService(unit), path, loaded


def record(service: AssistService, fingerprint: EvalFingerprint, *, minutes_ago: int) -> None:
    verdict = EvalVerdict(
        fingerprint=fingerprint,
        llm_mode="env",
        model="synthetic-model",
        decision_provider="classifier",
        split="test",
        n=100,
        test_n=100,
        threshold=0.95,
        factual=0.97,
        compliance=0.99,
        gate_passes=True,
        report_sha256="0" * 64,
    )
    at = service.unit.clock.now() - timedelta(minutes=minutes_ago)
    service.unit.repos.audit.record(verdict_event(verdict, at=at))


def shadow_ready(service: AssistService, n: int = 25) -> None:
    """影子模式下 n 条推荐都抽中、都打满分：影子对比的门槛能过。"""
    repo = service.unit.repos.suggestions
    for i in range(n):
        repo.add(
            SuggestionRecord(
                shop_id=SHOP_A,
                conversation_id=f"conv_c{i % 5}",
                msg_id=f"c_{i}",
                question_text="什么时候发货",
                intent=Intent.SHIPPING_POLICY,
                text="亲，现货付款后 48 小时内发货哦~",
                status=SuggestionStatus.SHADOW,
                created_at=W39_START + timedelta(hours=1, minutes=i),
            )
        )
    ro = service.rollout
    ro.sample(SHOP_A, reviewer=REVIEWER, period=W39)
    for q in service.unit.repos.quality_reviews.list(SHOP_A, period=W39):
        ro.score(SHOP_A, q.id, status="scored", scores=FULL, reviewer=REVIEWER)


def test_the_gate_checks_the_classifier_serve_loaded_not_the_file_on_disk(deployed):
    service, path, loaded = deployed
    config = service.unit.config
    sha_a = loaded.file_sha256
    assert sha_a == file_sha256(path) and unit_classifier_sha256(service.unit) == sha_a
    ro = service.rollout
    ro.set_stage(SHOP_A, "shadow", operator=REVIEWER)
    shadow_ready(service)
    # 同一路径原地换成 B（不重启 serve）；运维给 B 跑评测、record-eval（别的进程，读文件）
    new = train(path, extra=40, days=1)
    fp_b = current_fingerprint(config)
    assert fp_b.classifier_sha256 == new.file_sha256 == file_sha256(path) != sha_a
    record(service, fp_b, minutes_ago=5)
    # 运维任务那一侧（不传加载的摘要）看到的是文件：B 的结论对得上
    assert eval_evidence(service.unit.repos.audit, config).passed is True
    # 线上状态按加载的 A 核对：B 的结论不算
    status = ro.status(SHOP_A)
    ev = status.evidence.test_set
    assert ev.passed is None and "路由分类器模型文件" in (ev.reason or "")
    assert f"线上 {sha_a[:12]}" in (ev.reason or "")
    assert "test_set" in [u.name for u in status.unchecked]
    # 切到智能辅助的门槛同一口径：测试集记为没核对，不是通过
    change = ro.set_stage(SHOP_A, "assist", operator=REVIEWER)
    assert change.gate == status.shadow_gate and change.gate.unchecked == ["test_set"]
    assert "test_set" not in [c.name for c in change.gate.checks]
    ro.set_stage(SHOP_A, "shadow", operator=REVIEWER)
    # 给实际加载的 A 跑的评测照旧算（文件已经换了也一样）
    record(service, fp_b.model_copy(update={"classifier_sha256": sha_a}), minutes_ago=1)
    status = ro.status(SHOP_A)
    assert status.evidence.test_set.passed is True and status.unchecked == []
    change = ro.set_stage(SHOP_A, "assist", operator=REVIEWER)
    assert change.gate == status.shadow_gate and change.gate.unchecked == []
    assert {c.name: c.passed for c in change.gate.checks}["test_set"] is True


def test_an_in_memory_classifier_falls_back_to_the_file(tmp_path, monkeypatch):
    """路由的分类器不是从模型文件加载的（现训的）：没有加载的摘要，照旧按 RODIT_CLASSIFIER_PATH。"""
    path = tmp_path / "classifier.json"
    train(path)
    monkeypatch.setenv(CLASSIFIER_PATH_ENV, str(path))
    fresh = ClassifierProvider(RuleProvider()).fit(DEV, seed_examples=True)
    unit = build_demo_unit(
        Platform.TAOBAO,
        config=classifier_config(),
        llm=FakeLLMClient(handler=lambda request: "{}", model="synthetic-model"),
        route_provider=fresh,
    )
    assert unit_classifier_sha256(unit) is None
    service = AssistService(unit)
    record(service, current_fingerprint(unit.config), minutes_ago=1)
    assert service.rollout.status(SHOP_A).evidence.test_set.passed is True
