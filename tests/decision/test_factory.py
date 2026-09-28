"""按配置组装判断层（§6.4、§6.6 decision 配置）。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rodit.core.config import DecisionConfig
from rodit.core.enums import ProviderName
from rodit.core.errors import ComplianceError
from rodit.decision import (
    ClassifierProvider,
    InCloudLLMProvider,
    JevConditions,
    JevProvider,
    RuleProvider,
    build_decision_provider,
)
from rodit.testing.fakes import FakeLLMClient

LLM = FakeLLMClient(lambda r: {})
ALL_MET = JevConditions(
    in_cloud_deployment=True,
    operator_and_dpa=True,
    platform_approved=True,
    chinese_eval_passed=True,
)


def _jev(purpose: str) -> JevProvider:
    return JevProvider(lambda p: {}, ALL_MET, purpose=purpose)  # type: ignore[arg-type]


def test_incloud_llm_for_routing_votes_three_times():
    p = build_decision_provider(DecisionConfig(), purpose="route", llm=LLM)
    assert isinstance(p, InCloudLLMProvider)
    assert (p.samples, p.temperature, p.timeout_s, p.purpose) == (3, 0.7, 1.0, "route")
    assert p.min_votes == 2  # 过半数


def test_incloud_llm_for_qa_samples_once_at_zero_temperature():
    p = build_decision_provider(DecisionConfig(), purpose="qa", llm=LLM)
    assert isinstance(p, InCloudLLMProvider)
    assert (p.samples, p.temperature, p.purpose, p.min_votes) == (1, 0.0, "qa", 1)
    cfg = DecisionConfig(qa_vote_samples=5, vote_temperature=0.5, timeout_s=2.0)
    p5 = build_decision_provider(cfg, purpose="qa", llm=LLM)
    assert (p5.samples, p5.temperature, p5.timeout_s, p5.min_votes) == (5, 0.5, 2.0, 3)


def test_incloud_llm_requires_a_client():
    with pytest.raises(ValueError):
        build_decision_provider(DecisionConfig(), purpose="route")


def test_rules_provider():
    cfg = DecisionConfig(provider=ProviderName.RULES)
    assert isinstance(build_decision_provider(cfg, purpose="route"), RuleProvider)
    # 规则答不了质检的问题：质检仍用大模型
    assert isinstance(build_decision_provider(cfg, purpose="qa", llm=LLM), InCloudLLMProvider)
    with pytest.raises(ValueError):
        build_decision_provider(cfg, purpose="qa")


def test_classifier_provider(synthetic_intent_samples):
    cfg = DecisionConfig(provider=ProviderName.CLASSIFIER)
    p = build_decision_provider(cfg, purpose="route", classifier_samples=synthetic_intent_samples)
    assert isinstance(p, ClassifierProvider) and p.is_trained
    assert isinstance(p.fallback, RuleProvider)
    with pytest.raises(ValueError):
        build_decision_provider(cfg, purpose="route")
    assert isinstance(build_decision_provider(cfg, purpose="qa", llm=LLM), InCloudLLMProvider)


def test_jev_needs_approval_and_an_injected_production_provider():
    with pytest.raises(ValidationError):
        DecisionConfig(provider=ProviderName.JEV)
    cfg = DecisionConfig(provider=ProviderName.JEV, jev_production_approved=True)
    with pytest.raises(ValueError):
        build_decision_provider(cfg, purpose="route")
    with pytest.raises(ComplianceError):
        build_decision_provider(cfg, purpose="route", jev=_jev("offline_eval"))
    prod = _jev("production")
    assert build_decision_provider(cfg, purpose="route", jev=prod) is prod
    assert build_decision_provider(cfg, purpose="qa", jev=prod) is prod
