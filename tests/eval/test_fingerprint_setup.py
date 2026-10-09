"""评测结论的指纹要覆盖一切会改变回答的东西（方案 §13.5 "换大模型、改提示词、升级判断层之前，
先在测试集上跑回归"）：大模型（配置或环境变量给的）、路由判断层和分类器模型文件、生成环节的
提示词 / 模板、判断层的提示词和输出解析；线上按部署规则算出同样的指纹来比较。分类器路由的评测
不能给大模型路由的线上作依据，反过来也不行。数据全部合成，不联网。"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rodit
from rodit.app.offline import EvidenceEchoLLM
from rodit.core.clock import CST
from rodit.core.config import EngineConfig, load_config
from rodit.core.enums import Intent, ProviderName
from rodit.core.pii import mask_pii
from rodit.core.ports import LLMRequest
from rodit.core.questions import Q_INTENT
from rodit.decision import ClassifierProvider, RuleProvider
from rodit.eval import fingerprint as fp_module
from rodit.eval.cli import env_client, main
from rodit.eval.dataset import cases
from rodit.eval.fingerprint import (
    CLASSIFIER_PATH_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    EvalFingerprint,
    EvalVerdict,
    current_fingerprint,
    describe_verdict,
    endpoint_of,
    generation_version,
    judge_prompt_version,
    running_llm,
    same_setup,
    source_version,
    test_set_evidence,
    verdict_event,
    verdict_from_report,
)
from rodit.eval.llm_env import describe_endpoint
from rodit.testing.fakes import FakeLLMClient, judge_buyer_text, scripted_judge_llm

T0 = datetime(2026, 9, 25, 10, 0, tzinfo=CST)
BASE_URL = "https://llm.example.invalid/v1"
LLM_ENV = {LLM_BASE_URL_ENV: BASE_URL, LLM_MODEL_ENV: "synthetic-model"}
SHIPPED_CONFIG = Path(__file__).resolve().parents[2] / "config" / "rodit.yaml"
# 用例的买家消息（判断层 state 里是遮挡过个人信息的）→ 标注的问题类型
LABELLED = {mask_pii(c.text): c.expected_intent for c in cases("all")}


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
    for url in (
        BASE_URL,
        "http://10.0.0.8:8080/v1",
        "https://user:pw@h.example.invalid/v1",
        "https://h.example.invalid:443/v1",  # 默认端口：两处都不写
        "http://10.0.0.8:80/v1",
        "https://h.example.invalid:80/v1",  # 不是这个协议的默认端口：两处都保留
    ):
        assert endpoint_of(url) == describe_endpoint(type("C", (), {"base_url": url})())


@pytest.mark.parametrize(
    ("url", "endpoint"),
    [
        ("https://llm.example.invalid:443/v1", "https://llm.example.invalid"),
        ("https://llm.example.invalid/v1", "https://llm.example.invalid"),
        ("http://10.0.0.8:80/v1", "http://10.0.0.8"),
        ("http://10.0.0.8/v1", "http://10.0.0.8"),
        ("HTTPS://LLM.Example.invalid:443", "https://llm.example.invalid"),
        (" https://user:pw@llm.example.invalid:443/v1?x=1\n", "https://llm.example.invalid"),
        # 不是这个协议的默认端口：照写
        ("https://llm.example.invalid:8443/v1", "https://llm.example.invalid:8443"),
        ("https://llm.example.invalid:80/v1", "https://llm.example.invalid:80"),
        ("http://llm.example.invalid:443/v1", "http://llm.example.invalid:443"),
        ("http://10.0.0.8:8080/v1", "http://10.0.0.8:8080"),
    ],
)
def test_endpoint_of_leaves_out_the_default_port(url: str, endpoint: str):
    assert endpoint_of(url) == endpoint


def test_an_explicit_default_port_is_the_same_endpoint():
    """部署的地址写了 :443、评测时没写（或者反过来）：是同一个接口，评测结论照样算数；
    换了端口才是另一个接口。"""
    cfg = EngineConfig()
    plain = current_fingerprint(cfg, environ=LLM_ENV)
    explicit_url = "https://llm.example.invalid:443/v1"
    explicit = current_fingerprint(cfg, environ={**LLM_ENV, LLM_BASE_URL_ENV: explicit_url})
    assert plain.llm_endpoint == explicit.llm_endpoint == "https://llm.example.invalid"
    assert same_setup(plain, explicit)
    assert evidence(verdict_for(plain), explicit).passed is True
    assert evidence(verdict_for(explicit), plain).passed is True
    other_url = "https://llm.example.invalid:8443/v1"
    other = current_fingerprint(cfg, environ={**LLM_ENV, LLM_BASE_URL_ENV: other_url})
    ev = evidence(verdict_for(plain), other)
    assert ev.passed is None and "https://llm.example.invalid:8443" in (ev.reason or "")


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


# ================================================================ 提示词（§13.5 改提示词）


def test_source_version_changes_when_any_listed_module_changes(tmp_path, monkeypatch):
    version = generation_version()
    assert version is not None and len(version) == 12
    running = current_fingerprint(EngineConfig(), environ=LLM_ENV)
    assert running.generation_ver == version
    assert running.judge_prompt_ver == judge_prompt_version() != version
    package = tmp_path / "synthetic_generation_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "prompt.py").write_text('REQ = "先回答问题。"\n', encoding="utf-8")
    line = tmp_path / "synthetic_evidence_line.py"
    line.write_text('HEAD = "[{id}] {label}："\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    modules = ("synthetic_generation_pkg", "synthetic_evidence_line")
    monkeypatch.setattr(fp_module, "GENERATION_MODULES", modules)
    versions = []
    try:
        for path, text in (
            (None, ""),
            (package / "prompt.py", 'REQ = "先回答问题，再追问一句。"\n'),  # 包里的文件
            (line, 'HEAD = "<{id}> {label} -> "\n'),  # 单独的模块
        ):
            if path is not None:
                path.write_text(text, encoding="utf-8")
            source_version.cache_clear()
            versions.append(generation_version())
        assert source_version(("synthetic_evidence_line", "synthetic_no_such_module")) is None
    finally:
        source_version.cache_clear()
        for name in modules:
            sys.modules.pop(name, None)
    assert all(versions) and len(set(versions)) == 3
    stale = verdict_for(running.model_copy(update={"generation_ver": versions[0]}))
    ev = evidence(stale, running)
    assert ev.passed is None and "生成提示词" in (ev.reason or "")


def test_a_verdict_without_the_judge_prompt_version_does_not_count():
    """以前记的结论没有判断层提示词的版本：和线上对不上，要重新评测。"""
    running = current_fingerprint(EngineConfig(), environ=LLM_ENV)
    assert evidence(verdict_for(running), running).passed is True
    old = running.model_copy(update={"judge_prompt_ver": None})
    ev = evidence(verdict_for(old), running)
    assert ev.passed is None and "判断层提示词" in (ev.reason or "")


_FINGERPRINT_SCRIPT = """
import json, sys
import rodit
from rodit.core.config import EngineConfig
from rodit.eval.fingerprint import current_fingerprint
fp = current_fingerprint(EngineConfig(), environ=json.loads(sys.argv[1]))
print(json.dumps({"rodit": rodit.__file__, "fingerprint": fp.model_dump(mode="json")}))
"""

JUDGE_PROMPT = "判断层提示词"
GENERATION_PROMPT = "生成提示词"
PROMPT_EDITS = {
    # 判断层的 system 提示词（路由和质检的每一票都变）
    "judge_system_prompt": (
        "decision/llm.py",
        "1. 只根据 state 里的内容判断",
        "1. 宁可判为需要人工也不要漏判；只根据 state 里的内容判断",
        JUDGE_PROMPT,
    ),
    # 判断层输出的解析和计票
    "voting_parser": (
        "decision/voting.py",
        'nested = obj.get("answers")',
        'nested = obj.get("answer_map")',
        JUDGE_PROMPT,
    ),
    # 每条生成提示词里的依据行格式
    "evidence_line": (
        "core/evidence_line.py",
        'head = f"[{e.id}] {e.label}："',
        'head = f"<{e.id}> {e.label} -> "',
        GENERATION_PROMPT,
    ),
    # 提示词里的问题类型名
    "intent_labels": (
        "core/enums.py",
        'Intent.SIZE_ADVICE: "尺码 / 选型"',
        'Intent.SIZE_ADVICE: "尺码推荐"',
        GENERATION_PROMPT,
    ),
}


def _fingerprint_of_copy(tmp_path: Path, edit: tuple[str, str, str] | None) -> dict:
    """把 rodit 源码拷一份、改一处，在子进程里算线上指纹（配置和大模型都不变）。"""
    root = tmp_path / "src"
    shutil.copytree(
        Path(rodit.__file__).parent, root / "rodit", ignore=shutil.ignore_patterns("__pycache__")
    )
    if edit is not None:
        rel, old, new = edit
        target = root / "rodit" / rel
        text = target.read_text(encoding="utf-8")
        assert old in text, (rel, old)
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run(
        [sys.executable, "-c", _FINGERPRINT_SCRIPT, json.dumps(LLM_ENV)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert Path(out["rodit"]).resolve().is_relative_to(root.resolve())  # 用的是改过的拷贝
    return out["fingerprint"]


def test_an_unchanged_copy_gives_the_running_fingerprint(tmp_path):
    """对照：源码没改时，子进程算出来的和这里的一样（下面改一处就变的比较才有意义）。"""
    running = current_fingerprint(EngineConfig(), environ=LLM_ENV)
    assert _fingerprint_of_copy(tmp_path, None) == running.model_dump(mode="json")


@pytest.mark.parametrize("name", sorted(PROMPT_EDITS))
def test_a_prompt_change_outside_the_generation_package_needs_a_new_verdict(tmp_path, name):
    """改了判断层的提示词 / 输出解析、依据行格式、问题类型名：旧结论不再算测试集依据（§13.5）。"""
    rel, old, new, label = PROMPT_EDITS[name]
    running = current_fingerprint(EngineConfig(), environ=LLM_ENV)
    verdict = verdict_for(running)  # 改之前跑的评测
    assert evidence(verdict, running).passed is True
    changed = EvalFingerprint.model_validate(_fingerprint_of_copy(tmp_path, (rel, old, new)))
    assert changed.config_fingerprint == running.config_fingerprint
    assert not same_setup(verdict.fingerprint, changed)
    ev = evidence(verdict, changed)
    assert ev.passed is None and label in (ev.reason or "")


# ================================================================ 路由判断层和分类器（§13.5）


def _train(path: Path, *, trained_at: datetime = datetime(2026, 9, 28, 10, 0, tzinfo=CST)) -> None:
    pairs = [(c.text, c.expected_intent) for c in cases("dev")]
    model = ClassifierProvider(RuleProvider()).fit(pairs, seed_examples=True)
    model.save(path, trained_at=trained_at)


def _labelled_intent(request: LLMRequest) -> dict:
    """路由的每一票都答用例标注的问题类型。"""
    intent = LABELLED.get(judge_buyer_text(request), Intent.OTHER)
    return {Q_INTENT: {"choice": intent.value}}


def _env_llm() -> FakeLLMClient:
    """--llm env 的替身：生成复述依据，路由 / 质检用脚本化的判断层（路由答用例标注的类型，评测集
    以外的消息——连通性检查的合成会话——答 other；质检全部通过）。每次调用都成功、都读得出来：
    这一轮没有降级，评到的是"模型"，record-eval 才收。"""
    judge, echo = scripted_judge_llm(route=_labelled_intent), EvidenceEchoLLM()

    def handler(request: LLMRequest):
        return echo.complete(request) if request.purpose == "generate" else judge.complete(request)

    client = FakeLLMClient(handler=handler, model="synthetic-model")
    client.base_url = BASE_URL  # type: ignore[attr-defined]  # 和线上的环境变量同一个接口
    return client


def _eval(tmp_path: Path, *extra: str, split: str = "test") -> EvalVerdict:
    target = tmp_path / f"report{len(list(tmp_path.glob('report*.json')))}.json"
    argv = ["--split", split, "--llm", "env", *extra, "--json", str(target)]
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


def test_a_full_env_run_becomes_a_verdict_with_its_run_health(tmp_path):
    """评测命令行跑完整个评测集（替身大模型，没有降级）→ 结论：条数对得上评测集，带着运行健康度的
    计数（全部用例和测试集），能给同一套部署作测试集依据。"""
    verdict = _eval(tmp_path, split="all")
    assert (verdict.n, verdict.test_n) == (len(cases("all")), len(cases("test"))) == (324, 66)
    overall, test = verdict.health_overall, verdict.health_test
    assert overall is not None and test is not None
    assert (overall.cases, test.cases) == (324, 66)
    assert overall.route_fallback == overall.qa_unavailable == overall.generation_unusable == 0
    assert "运行健康度：路由退回 0/324" in describe_verdict(T0, verdict)
    ev = evidence(verdict, current_fingerprint(EngineConfig(), environ=LLM_ENV))
    assert ev.reason is None and "路由退回 0/324" in ev.detail


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
