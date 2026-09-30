"""评测结论的指纹要覆盖一切会改变回答的东西（方案 §13.5 "换大模型、改提示词、升级判断层之前，
先在测试集上跑回归"）：大模型（配置或环境变量给的）、路由判断层和分类器模型文件、生成环节的
提示词 / 模板；线上按部署规则算出同样的指纹来比较。分类器路由的评测不能给大模型路由的线上作依据，
反过来也不行。数据全部合成，不联网。"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rodit.app.offline import EvidenceEchoLLM
from rodit.core.clock import CST
from rodit.core.config import EngineConfig, load_config
from rodit.core.enums import ProviderName
from rodit.core.ports import LLMRequest
from rodit.decision import ClassifierProvider, RuleProvider
from rodit.eval import fingerprint as fp_module
from rodit.eval.cli import env_client, main
from rodit.eval.dataset import cases
from rodit.eval.fingerprint import (
    CLASSIFIER_PATH_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    EvalVerdict,
    current_fingerprint,
    endpoint_of,
    generation_version,
    running_llm,
    same_setup,
    test_set_evidence,
    verdict_event,
    verdict_from_report,
)
from rodit.eval.llm_env import describe_endpoint
from rodit.testing.fakes import FakeLLMClient, scripted_judge_llm

T0 = datetime(2026, 9, 25, 10, 0, tzinfo=CST)
BASE_URL = "https://llm.example.invalid/v1"
LLM_ENV = {LLM_BASE_URL_ENV: BASE_URL, LLM_MODEL_ENV: "synthetic-model"}
SHIPPED_CONFIG = Path(__file__).resolve().parents[2] / "config" / "rodit.yaml"


def with_provider(config: EngineConfig, provider: ProviderName) -> EngineConfig:
    decision = config.decision.model_copy(update={"provider": provider})
    return config.model_copy(update={"decision": decision})


def verdict_for(
    fingerprint, *, model: str = "synthetic-model", provider="incloud_llm"
) -> EvalVerdict:
    return EvalVerdict(
        fingerprint=fingerprint,
        llm_mode="env",
        model=model,
        decision_provider=provider,
        split="test",
        n=100,
        test_n=100,
        threshold=0.95,
        factual=0.97,
        compliance=0.99,
        gate_passes=True,
        report_sha256="0" * 64,
    )


def evidence(verdict: EvalVerdict, running):
    return test_set_evidence([verdict_event(verdict, at=T0)], running)


# ================================================================ 大模型（§13.5 换大模型）


def test_an_environment_only_model_swap_invalidates_the_verdict():
    """发布的配置里没写 llm.model：线上的模型来自 RODIT_LLM_MODEL，只改环境变量也要重新评测。"""
    cfg = load_config(SHIPPED_CONFIG)
    assert cfg.llm.model is None and cfg.llm.base_url is None
    env_a = {**LLM_ENV, LLM_MODEL_ENV: "model-a"}
    verdict = verdict_for(current_fingerprint(cfg, environ=env_a), model="model-a")
    assert evidence(verdict, current_fingerprint(cfg, environ=env_a)).passed is True
    env_b = {**LLM_ENV, LLM_MODEL_ENV: "model-b"}
    ev = evidence(verdict, current_fingerprint(cfg, environ=env_b))
    assert ev.passed is None and "大模型" in (ev.reason or "")
    assert "model-a" in (ev.reason or "") and "model-b" in (ev.reason or "")
    # 换了接口地址（同名模型换了服务商）也一样
    env_host = {**env_a, LLM_BASE_URL_ENV: "https://other-llm.example.invalid/v1"}
    ev = evidence(verdict, current_fingerprint(cfg, environ=env_host))
    assert ev.passed is None and "other-llm.example.invalid" in (ev.reason or "")


def test_an_unknown_running_model_matches_nothing():
    cfg = EngineConfig()
    verdict = verdict_for(current_fingerprint(cfg, environ=LLM_ENV))
    ev = evidence(verdict, current_fingerprint(cfg, environ={}))
    assert ev.passed is None and "未知" in (ev.reason or "") and LLM_MODEL_ENV in (ev.reason or "")
    # 以前记的结论没有大模型、判断层、生成环节：也不算
    old = verdict.fingerprint.model_copy(
        update={"llm_model": None, "llm_endpoint": None, "generation_ver": None}
    )
    ev = evidence(verdict_for(old), current_fingerprint(cfg, environ=LLM_ENV))
    assert ev.passed is None and "生成提示词" in (ev.reason or "") and "没记录" in (ev.reason or "")


def test_running_llm_follows_the_production_rule():
    """和 app.production.build_production_unit 一样：配置里地址和模型都写了用配置，
    否则都取环境变量。"""
    env = {LLM_BASE_URL_ENV: "https://env-llm.example.invalid:8443/v1", LLM_MODEL_ENV: "env-model"}
    pinned = EngineConfig.model_validate(
        {
            "llm": {
                "base_url": "https://user:pw@cfg-llm.example.invalid/v1?x=1",
                "model": "cfg-model",
            }
        }
    )
    assert running_llm(pinned, env) == ("cfg-model", "https://cfg-llm.example.invalid")
    only_model = EngineConfig.model_validate({"llm": {"model": "cfg-model"}})
    assert running_llm(only_model, env) == ("env-model", "https://env-llm.example.invalid:8443")
    assert running_llm(EngineConfig(), {}) == (None, None)
    for url in (BASE_URL, "http://10.0.0.8:8080/v1", "https://user:pw@h.example.invalid/v1"):
        assert endpoint_of(url) == describe_endpoint(type("C", (), {"base_url": url})())


def test_the_eval_client_follows_the_production_rule(monkeypatch):
    """配置钉了模型 B、评测机器的环境变量是模型 A：评测用 B（和线上一样），记下的也是 B。"""
    monkeypatch.setenv(LLM_BASE_URL_ENV, "https://llm-a.example.invalid/v1")
    monkeypatch.setenv(LLM_MODEL_ENV, "model-a")
    monkeypatch.setenv("RODIT_LLM_API_KEY", "sk-synthetic-not-a-real-key")
    pinned = EngineConfig.model_validate(
        {"llm": {"base_url": "https://llm-b.example.invalid/v1", "model": "model-b"}}
    )
    client = env_client(pinned)
    assert (client.model, client.base_url) == ("model-b", "https://llm-b.example.invalid/v1")
    assert env_client(None).model == "model-a"
    monkeypatch.delenv("RODIT_LLM_API_KEY")
    with pytest.raises(RuntimeError, match="RODIT_LLM_API_KEY"):
        env_client(pinned)


# ================================================================ 生成环节（§13.5 改提示词）


def test_generation_version_changes_when_the_generation_code_changes(tmp_path, monkeypatch):
    version = generation_version()
    assert version is not None and len(version) == 12
    assert current_fingerprint(EngineConfig(), environ=LLM_ENV).generation_ver == version
    package = tmp_path / "synthetic_generation_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "prompt.py").write_text('REQ = "先回答问题。"\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(fp_module, "GENERATION_PACKAGE", "synthetic_generation_pkg")
    generation_version.cache_clear()
    try:
        before = generation_version()
        (package / "prompt.py").write_text('REQ = "先回答问题，再追问一句。"\n', encoding="utf-8")
        generation_version.cache_clear()
        after = generation_version()
    finally:
        generation_version.cache_clear()
        sys.modules.pop("synthetic_generation_pkg", None)
    assert before and after and before != after
    running = current_fingerprint(EngineConfig(), environ=LLM_ENV)
    stale = verdict_for(running.model_copy(update={"generation_ver": before}))
    ev = evidence(stale, running)
    assert ev.passed is None and "生成提示词" in (ev.reason or "")


# ================================================================ 路由判断层和分类器（§13.5）


def _train(path: Path, *, trained_at: datetime = datetime(2026, 9, 28, 10, 0, tzinfo=CST)) -> None:
    pairs = [(c.text, c.expected_intent) for c in cases("dev")]
    model = ClassifierProvider(RuleProvider()).fit(pairs, seed_examples=True)
    model.save(path, trained_at=trained_at)


def _env_llm() -> FakeLLMClient:
    """--llm env 的替身：连通性检查回 OK，生成复述依据，路由 / 质检用脚本化的判断层。"""
    judge, echo = scripted_judge_llm(), EvidenceEchoLLM()

    def handler(request: LLMRequest):
        if request.purpose == "other":
            return "OK"
        return echo.complete(request) if request.purpose == "generate" else judge.complete(request)

    client = FakeLLMClient(handler=handler, model="synthetic-model")
    client.base_url = BASE_URL  # type: ignore[attr-defined]  # 和线上的环境变量同一个接口
    return client


def _eval(tmp_path: Path, *extra: str) -> EvalVerdict:
    target = tmp_path / f"report{len(list(tmp_path.glob('report*.json')))}.json"
    argv = ["--split", "test", "--llm", "env", *extra, "--json", str(target)]
    err = io.StringIO()
    assert main(argv, out=io.StringIO(), err=err, client_factory=_env_llm) == 0, err.getvalue()
    raw = target.read_bytes()
    return verdict_from_report(json.loads(raw), raw=raw)


@pytest.fixture(scope="module")
def classifier_eval(tmp_path_factory):
    """用分类器路由跑的完整测试集评测（配置是默认的 incloud_llm）和分类器文件。"""
    tmp = tmp_path_factory.mktemp("clf_eval")
    clf = tmp / "classifier.json"
    _train(clf)
    return _eval(tmp, "--classifier", str(clf)), clf


def test_a_classifier_routed_eval_is_not_evidence_for_an_llm_routed_unit(classifier_eval):
    verdict, clf = classifier_eval
    assert verdict.decision_provider == "classifier"
    assert verdict.fingerprint.decision_provider == "classifier"
    assert verdict.fingerprint.classifier_sha256 is not None
    llm_routed = current_fingerprint(EngineConfig(), environ=LLM_ENV)
    ev = evidence(verdict, llm_routed)
    assert ev.passed is None and ev.verdict_at == T0
    assert "路由判断层" in (ev.reason or "") and "classifier" in (ev.reason or "")


def test_a_classifier_routed_eval_backs_the_classifier_deployment_it_ran(classifier_eval, tmp_path):
    verdict, clf = classifier_eval
    deployed = with_provider(EngineConfig(), ProviderName.CLASSIFIER)
    running = current_fingerprint(deployed, environ={**LLM_ENV, CLASSIFIER_PATH_ENV: str(clf)})
    assert same_setup(verdict.fingerprint, running)
    ev = evidence(verdict, running)
    assert ev.reason is None and ev.passed is not None
    # 换了分类器文件（重新训练）：对不上
    retrained = tmp_path / "classifier_v2.json"
    _train(retrained, trained_at=datetime(2026, 9, 29, 10, 0, tzinfo=CST))
    other = current_fingerprint(deployed, environ={**LLM_ENV, CLASSIFIER_PATH_ENV: str(retrained)})
    ev = evidence(verdict, other)
    assert ev.passed is None and "路由分类器模型文件" in (ev.reason or "")
    # 线上没设分类器文件（或读不了）：不知道用的是哪个，也对不上
    unknown = current_fingerprint(deployed, environ=LLM_ENV)
    ev = evidence(verdict, unknown)
    assert ev.passed is None and CLASSIFIER_PATH_ENV in (ev.reason or "")


def test_an_llm_routed_eval_is_not_evidence_for_a_classifier_deployment(tmp_path):
    verdict = _eval(tmp_path)
    assert verdict.fingerprint.decision_provider == "incloud_llm"
    assert (verdict.fingerprint.llm_model, verdict.fingerprint.llm_endpoint) == (
        "synthetic-model",
        "https://llm.example.invalid",
    )
    assert evidence(verdict, current_fingerprint(EngineConfig(), environ=LLM_ENV)).reason is None
    clf = tmp_path / "classifier.json"
    _train(clf)
    deployed = with_provider(EngineConfig(), ProviderName.CLASSIFIER)
    running = current_fingerprint(deployed, environ={**LLM_ENV, CLASSIFIER_PATH_ENV: str(clf)})
    ev = evidence(verdict, running)
    assert ev.passed is None and "路由判断层" in (ev.reason or "")


def test_offline_report_fingerprint_names_the_stand_in(tmp_path):
    target = tmp_path / "offline.json"
    assert main(["--split", "test", "--limit", "3", "--json", str(target)], out=io.StringIO()) == 0
    fp = json.loads(target.read_text(encoding="utf-8"))["fingerprint"]
    assert fp["llm_model"] == EvidenceEchoLLM.model and fp["llm_endpoint"] is None
    assert fp["decision_provider"] == "incloud_llm" and fp["generation_ver"] == generation_version()


def test_a_later_offline_verdict_does_not_hide_a_matching_real_one():
    running = current_fingerprint(EngineConfig(), environ=LLM_ENV)
    real = verdict_event(verdict_for(running), at=T0)
    offline = verdict_for(running.model_copy(update={"llm_model": EvidenceEchoLLM.model}))
    offline = offline.model_copy(update={"llm_mode": "offline"})
    later = verdict_event(offline, at=T0 + timedelta(hours=1))
    ev = test_set_evidence([real, later], running)
    assert ev.passed is True and ev.verdict_at == T0
    ev = test_set_evidence([later], running)
    assert ev.passed is None and "离线替身" in (ev.reason or "")
