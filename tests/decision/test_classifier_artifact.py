"""路由分类器的模型文件（§6.4 Phase 2：离线训练 → 版本化的 JSON 文件 → 生产单元加载）。

全部是合成数据（conftest 的合成问句、§5.1 例句），不联网。
"""

from __future__ import annotations

import copy
import json
import os
import stat
from datetime import timedelta

import pytest

from rodit.core.config import DecisionConfig
from rodit.core.decision import DecisionRequest
from rodit.core.enums import INTENT_EXAMPLES, Intent, ProviderName
from rodit.core.questions import Q_INTENT, ROUTER_MODEL, ROUTER_QUESTIONS, ROUTER_QUESTIONS_VER
from rodit.decision import (
    ClassifierArtifactError,
    ClassifierProvider,
    InCloudLLMProvider,
    RuleProvider,
    build_decision_provider,
    read_artifact,
    write_artifact,
)
from rodit.decision.classifier import (
    ARTIFACT_VERSION,
    artifact_digest,
    dump_artifact,
    feature_key,
    read_artifact_with_digest,
)
from rodit.eval.fingerprint import file_sha256
from rodit.testing.fakes import FakeLLMClient
from rodit.testing.fixtures import DEFAULT_NOW

PROBES = (
    "这件是纯棉的吗",
    "165 110斤穿什么码",
    "快递到哪了",
    "加我微信",
    "我电话13812345678，地址写错了",
    "完全没见过的一句话",
    "",
)


def _trained(samples) -> ClassifierProvider:
    return ClassifierProvider(RuleProvider(), temperature=1.5).fit(samples, seed_examples=True)


def _request(text: str) -> DecisionRequest:
    return DecisionRequest(
        model=ROUTER_MODEL,
        state={"messages": [{"role": "buyer", "text": text}]},
        questions=ROUTER_QUESTIONS,
    )


def _resign(artifact: dict) -> dict:
    """改了内容后重新算校验和（模拟"格式对、校验和也对，但内容不合法"的文件）。"""
    artifact = {k: v for k, v in artifact.items() if k != "sha256"}
    artifact["sha256"] = artifact_digest(artifact)
    return artifact


@pytest.fixture
def artifact(synthetic_intent_samples) -> dict:
    return _trained(synthetic_intent_samples).to_artifact(
        trained_at=DEFAULT_NOW, metrics={"holdout": {"accuracy": 0.9}}
    )


# ================================================================ 往返


def test_round_trip_gives_identical_predictions(synthetic_intent_samples, tmp_path):
    clf = _trained(synthetic_intent_samples)
    artifact = clf.to_artifact(trained_at=DEFAULT_NOW)
    path = tmp_path / "classifier.json"
    write_artifact(path, artifact)
    loaded = ClassifierProvider.load(path, fallback=RuleProvider())
    assert loaded.is_trained and loaded.classes == clf.classes
    assert (loaded.alpha, loaded.temperature) == (clf.alpha, clf.temperature)
    texts = [t for t, _ in synthetic_intent_samples] + list(PROBES)
    for text in texts:
        assert loaded.predict_proba(text) == clf.predict_proba(text)  # 逐位一致
        assert loaded.decide(_request(text)) == clf.decide(_request(text))


def test_loaded_provider_carries_metadata(synthetic_intent_samples, tmp_path):
    clf = _trained(synthetic_intent_samples)
    path = tmp_path / "classifier.json"
    saved = clf.save(path, trained_at=DEFAULT_NOW, metrics={"holdout": {"accuracy": 0.91}})
    assert clf.artifact_sha256 == saved["sha256"] and clf.meta is not None
    loaded = ClassifierProvider.load(path, fallback=RuleProvider(), name="clf-v1")
    meta = loaded.meta
    assert meta is not None and loaded.name == "clf-v1"
    assert meta.trained_at == DEFAULT_NOW
    assert meta.questions_version == ROUTER_QUESTIONS_VER
    seeds = sum(len(t) for t in INTENT_EXAMPLES.values())
    assert (meta.samples, meta.seed_examples) == (len(synthetic_intent_samples), seeds)
    assert meta.metrics == {"holdout": {"accuracy": 0.91}}
    assert loaded.artifact_sha256 == saved["sha256"] == artifact_digest(saved)
    # 重新训练就是新模型：不再挂着旧模型文件的 metadata
    loaded.fit(synthetic_intent_samples)
    assert loaded.meta is None and loaded.artifact_sha256 is None


def test_the_loaded_file_hash_stays_with_the_instance(synthetic_intent_samples, tmp_path):
    """加载时记下读到的字节的 SHA-256（和评测指纹的 file_sha256 同一个值）；同一路径之后被换成
    别的模型文件（train-classifier --overwrite 是原子改名），已加载的实例照旧是原来的那个。"""
    path = tmp_path / "classifier.json"
    a = _trained(synthetic_intent_samples)
    a.save(path, trained_at=DEFAULT_NOW)
    sha_a = file_sha256(path)
    assert a.file_sha256 == sha_a  # 写入的字节
    data, digest = read_artifact_with_digest(path)
    assert digest == sha_a and data == read_artifact(path)
    loaded = ClassifierProvider.load(path, fallback=RuleProvider())
    assert loaded.file_sha256 == sha_a
    b = _trained(synthetic_intent_samples[:-3])
    b.save(path, trained_at=DEFAULT_NOW + timedelta(days=1))
    assert file_sha256(path) == b.file_sha256 != sha_a
    assert loaded.file_sha256 == sha_a
    # 从内容重建、重新训练的实例不对应任何文件
    assert ClassifierProvider.from_artifact(data, fallback=RuleProvider()).file_sha256 is None
    loaded.fit(synthetic_intent_samples)
    assert loaded.file_sha256 is None


def test_artifact_is_plain_json_with_integer_counts(artifact):
    assert artifact["format"] == "rodit.route_classifier"
    assert artifact["format_version"] == ARTIFACT_VERSION
    model = artifact["model"]
    assert model["classes"] == [i.value for i in Intent if i.value in model["classes"]]
    assert set(model["doc_counts"]) == set(model["feature_counts"]) == set(model["classes"])
    for counts in model["feature_counts"].values():
        assert all(isinstance(n, int) and n >= 1 for n in counts.values())
        assert all(1 <= len(f) <= 2 for f in counts)
    # 规范 JSON：同样的内容总是同样的字节
    assert dump_artifact(artifact) == dump_artifact(copy.deepcopy(artifact))
    assert json.loads(dump_artifact(artifact)) == artifact


def test_same_training_gives_same_artifact(synthetic_intent_samples):
    a = _trained(synthetic_intent_samples).to_artifact(trained_at=DEFAULT_NOW)
    b = _trained(list(reversed(synthetic_intent_samples))).to_artifact(trained_at=DEFAULT_NOW)
    assert dump_artifact(a) == dump_artifact(b)
    c = _trained(synthetic_intent_samples).to_artifact(trained_at=DEFAULT_NOW + timedelta(days=1))
    assert c["sha256"] != a["sha256"]
    assert c["model"] == a["model"]


def test_training_texts_are_masked_before_counting():
    """模型文件里是脱敏后文本的 n 元组：手机号的数字不会出现在特征里。"""
    clf = ClassifierProvider(RuleProvider()).fit(
        [("我电话13812345678改地址", Intent.MODIFY_ORDER), ("还有货吗", Intent.STOCK)]
    )
    raw = dump_artifact(clf.to_artifact(trained_at=DEFAULT_NOW)).decode()
    assert "1381" not in raw and "5678" not in raw
    assert "手机" in raw  # 占位符 [手机号] 的字


def test_explicit_seed_list_is_counted_as_seeds(synthetic_intent_samples):
    seeds = [("在吗？", Intent.GREETING), ("我要投诉", Intent.COMPLAINT)]
    clf = ClassifierProvider(RuleProvider()).fit(synthetic_intent_samples, seed_examples=seeds)
    meta = clf.to_artifact(trained_at=DEFAULT_NOW)["metadata"]
    assert (meta["samples"], meta["seed_examples"]) == (len(synthetic_intent_samples), 2)


def test_untrained_or_unserialisable_metrics_refused(synthetic_intent_samples):
    with pytest.raises(ValueError, match="还没有训练"):
        ClassifierProvider(RuleProvider()).to_artifact(trained_at=DEFAULT_NOW)
    clf = _trained(synthetic_intent_samples)
    with pytest.raises(ValueError, match="metrics"):
        clf.to_artifact(trained_at=DEFAULT_NOW, metrics={"accuracy": float("nan")})


def test_feature_key_matches_ngram_identity():
    assert feature_key("有 货吗？") == feature_key("有货吗") == "有货吗"
    assert feature_key("ＡＢＣ！") == "abc"
    assert feature_key("电话13812345678") == feature_key("电话 [手机号]")


# ================================================================ 写文件


def test_write_is_atomic_and_private(artifact, tmp_path, monkeypatch):
    path = tmp_path / "classifier.json"
    path.write_text("旧内容", encoding="utf-8")
    write_artifact(path, artifact)
    assert read_artifact(path) == artifact
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ["classifier.json"]  # 没有临时文件

    # 改名前失败：原文件不动，临时文件清掉
    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    changed = _resign({**artifact, "metadata": {**artifact["metadata"], "samples": 1}})
    with pytest.raises(OSError):
        write_artifact(path, changed)
    assert read_artifact(path) == artifact
    assert sorted(p.name for p in tmp_path.iterdir()) == ["classifier.json"]


# ================================================================ 加载时的校验


def _write_raw(tmp_path, data) -> str:
    path = tmp_path / "bad.json"
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _load(path: str) -> ClassifierProvider:
    return ClassifierProvider.load(path, fallback=RuleProvider())


def test_missing_or_unreadable_file(tmp_path):
    with pytest.raises(ClassifierArtifactError, match="不存在"):
        _load(str(tmp_path / "absent.json"))
    with pytest.raises(ClassifierArtifactError, match="不是合法的 JSON"):
        _load(_write_raw(tmp_path, b"\xff\xfe not json"))
    with pytest.raises(ClassifierArtifactError, match="不是合法的 JSON"):
        _load(_write_raw(tmp_path, b'{"format": "rodit.route_classifier"'))  # 截断
    with pytest.raises(ClassifierArtifactError, match="不是 JSON 对象"):
        _load(_write_raw(tmp_path, [1, 2, 3]))


def test_tampered_artifact_rejected(artifact, tmp_path):
    tampered = copy.deepcopy(artifact)
    counts = tampered["model"]["doc_counts"]
    first = next(iter(counts))
    counts[first] += 1
    with pytest.raises(ClassifierArtifactError, match="校验和不符") as exc:
        _load(_write_raw(tmp_path, tampered))
    assert "bad.json" in str(exc.value)  # 消息里带路径，方便运维
    no_hash = {k: v for k, v in artifact.items() if k != "sha256"}
    with pytest.raises(ClassifierArtifactError, match="校验和不符"):
        _load(_write_raw(tmp_path, no_hash))
    meta_changed = copy.deepcopy(artifact)
    meta_changed["metadata"]["trained_at"] = "2030-01-01T00:00:00+08:00"
    with pytest.raises(ClassifierArtifactError, match="校验和不符"):
        ClassifierProvider.from_artifact(meta_changed, fallback=RuleProvider())


@pytest.mark.parametrize("version", [0, 2, "1", 1.0, True, None])
def test_wrong_format_version_rejected(artifact, tmp_path, version):
    changed = _resign({**artifact, "format_version": version})
    with pytest.raises(ClassifierArtifactError, match="格式版本"):
        _load(_write_raw(tmp_path, changed))


def test_wrong_format_name_rejected(artifact, tmp_path):
    with pytest.raises(ClassifierArtifactError, match="不是路由分类器"):
        _load(_write_raw(tmp_path, _resign({**artifact, "format": "pickle"})))


@pytest.mark.parametrize("where", ["classes", "doc_counts", "feature_counts"])
def test_unknown_intent_rejected(artifact, tmp_path, where):
    changed = copy.deepcopy(artifact)
    model = changed["model"]
    if where == "classes":
        model["classes"].append("refund_fast_track")
    else:
        model[where]["refund_fast_track"] = model[where][model["classes"][0]]
    with pytest.raises(ClassifierArtifactError, match="不认识的问题类型.*refund_fast_track"):
        _load(_write_raw(tmp_path, _resign(changed)))


def _mutate_model(artifact: dict, fn) -> dict:
    changed = copy.deepcopy(artifact)
    fn(changed["model"])
    return _resign(changed)


def _first(model: dict) -> str:
    return model["classes"][0]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda m: m["doc_counts"].__setitem__(_first(m), 0), "结构不对"),
        (lambda m: m["doc_counts"].__setitem__(_first(m), 2.5), "结构不对"),
        (lambda m: m["doc_counts"].__setitem__(_first(m), "3"), "结构不对"),
        (lambda m: m["feature_counts"][_first(m)].__setitem__("三个字", 1), "结构不对"),
        (lambda m: m["feature_counts"][_first(m)].__setitem__("", 1), "结构不对"),
        (lambda m: m.__setitem__("alpha", 0), "结构不对"),
        (lambda m: m.__setitem__("temperature", -1), "结构不对"),
        (lambda m: m.__setitem__("kind", "logistic"), "结构不对"),
        (lambda m: m.__setitem__("features", "word_unigram"), "结构不对"),
        (lambda m: m.__setitem__("extra", 1), "结构不对"),
        (lambda m: m.__setitem__("classes", m["classes"][:1]), "结构不对"),
        (lambda m: m["doc_counts"].pop(_first(m)), "对不上"),
        (lambda m: m.__setitem__("classes", [*m["classes"], m["classes"][0]]), "重复"),
        (lambda m: m.__setitem__("classes", list(reversed(m["classes"]))), "顺序"),
    ],
)
def test_malformed_model_rejected(artifact, tmp_path, mutate, match):
    with pytest.raises(ClassifierArtifactError, match=match):
        _load(_write_raw(tmp_path, _mutate_model(artifact, mutate)))


def test_oversized_file_rejected(artifact, tmp_path, monkeypatch):
    from rodit.decision import classifier as module

    path = tmp_path / "classifier.json"
    write_artifact(path, artifact)
    monkeypatch.setattr(module, "MAX_ARTIFACT_BYTES", 100)
    with pytest.raises(ClassifierArtifactError, match="过大"):
        _load(str(path))


# ================================================================ 组装


def test_factory_uses_loaded_classifier_for_routing_only(artifact):
    cfg = DecisionConfig(provider=ProviderName.CLASSIFIER)
    loaded = ClassifierProvider.from_artifact(artifact, fallback=RuleProvider())
    assert build_decision_provider(cfg, purpose="route", classifier=loaded) is loaded
    # 质检仍用大模型（§6.4 "质检用硬规则 + 大模型"）
    qa = build_decision_provider(
        cfg, purpose="qa", llm=FakeLLMClient(lambda r: {}), classifier=loaded
    )
    assert isinstance(qa, InCloudLLMProvider)
    with pytest.raises(ValueError, match="还没有训练"):
        build_decision_provider(cfg, purpose="route", classifier=ClassifierProvider(RuleProvider()))
    with pytest.raises(ValueError, match="RODIT_CLASSIFIER_PATH"):
        build_decision_provider(cfg, purpose="route")


def test_loaded_classifier_answers_router_questions(artifact):
    loaded = ClassifierProvider.from_artifact(artifact, fallback=RuleProvider())
    resp = loaded.decide(_request("加我微信聊"))
    assert set(resp.answers) == set(ROUTER_QUESTIONS)
    assert resp.choice(Q_INTENT).choice == Intent.OFF_PLATFORM.value
