"""部署入口的离线任务和路由分类器上线（方案 §13.5 每月重算阈值、§6.4 Phase 2 路由分类器）。

- recalibrate / train-classifier：标注来源注入（不连数据库），时钟注入；
- serve 在 decision.provider: classifier 时加载模型文件：缺失、损坏都拒绝启动（退出码 2）；
- 最后一条 @pytest.mark.pg：真实 PostgreSQL 上经 rodit_trainer 角色导出标注。
全部是合成数据；不联网（大模型地址是 .invalid，生产模式下不发业务请求）。
"""

from __future__ import annotations

import io
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
import yaml
from ops_support import (
    SYNTHETIC_LLM_KEY,
    SYNTHETIC_TOKEN_KEY,
    FakeReadinessDb,
    FakeServeLease,
    TestClient,
)

from rodit.app.demo import seed_unit
from rodit.app.unit import in_memory_repos
from rodit.core.config import load_config
from rodit.core.enums import DecisionStage, Intent
from rodit.core.errors import StorageUnavailable
from rodit.core.models import LabeledSample
from rodit.decision import (
    ClassifierProvider,
    InCloudLLMProvider,
    RuleProvider,
    read_artifact,
    write_artifact,
)
from rodit.eval.dataset import cases
from rodit.eval.recalibrate import (
    CLASSIFIER_REGRESSION_REMINDER,
    REGRESSION_REMINDER,
    apply_patch,
)
from rodit.ops import serve
from rodit.ops.serve import build_server, main
from rodit.ops.settings import ServerSettings
from rodit.storage import Database, PgFeedbackRepository
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import DEFAULT_NOW, SHOP_A, SHOP_B, build_demo_world

NOW = DEFAULT_NOW
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "rodit.yaml"


@pytest.fixture
def jobs_env(ops_env, monkeypatch):
    """ops_env 之外，再清掉本文件用到的变量（它们不在 ops_support 的清单里）。"""
    for name in ("RODIT_CLASSIFIER_PATH", "RODIT_TRAINER_DSN", "RODIT_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    return ops_env


# ================================================================ 合成标注


def feedback_labels(days_ago: float = 1) -> list[LabeledSample]:
    """尺码类：0.6–0.8 之间三条错一条，0.8 以上全对（辅助阈值应上调）。"""
    out: list[LabeledSample] = []
    for i in range(120):
        conf = round(0.6 + 0.4 * i / 119, 6)
        ok = conf >= 0.8 or i % 3
        out.append(
            LabeledSample(
                shop_id=SHOP_A if i % 2 else SHOP_B,
                text=f"合成尺码问句{i}",
                predicted_intent=Intent.SIZE_ADVICE,
                predicted_confidence=conf,
                true_intent=Intent.SIZE_ADVICE if ok else None,
                source="feedback_adopt" if ok else "feedback_reject",
                created_at=NOW - timedelta(days=days_ago, minutes=i),
            )
        )
    return out


def training_labels(n_prefix: int = 4) -> list[LabeledSample]:
    """合成评测集开发集的问句加前后缀变体（和测试集不重叠）。"""
    out: list[LabeledSample] = []
    for case in cases("dev"):
        for prefix in ("", "请问", "亲，", "你好，")[:n_prefix]:
            for suffix in ("", "呀"):
                out.append(
                    LabeledSample(
                        shop_id=SHOP_A,
                        text=f"{prefix}{case.text}{suffix}",
                        predicted_intent=case.expected_intent,
                        predicted_confidence=0.9,
                        true_intent=case.expected_intent,
                        source="feedback_adopt",
                        created_at=NOW - timedelta(minutes=len(out)),
                    )
                )
    return out


class Source:
    """注入的标注来源：记下被调用了几次。"""

    def __init__(self, labels: list[LabeledSample] | None = None, error: Exception | None = None):
        self.labels = labels or []
        self.error = error
        self.calls = 0
        self.windows: list[tuple[Any, Any]] = []

    def __call__(self, *, since=None, until=None) -> list[LabeledSample]:
        self.calls += 1
        self.windows.append((since, until))
        if self.error is not None:
            raise self.error
        return [
            label
            for label in self.labels
            if (since is None or label.created_at >= since)
            and (until is None or label.created_at < until)
        ]


def _run(argv: list[str], source: Source, *, now=NOW) -> tuple[int, str]:
    out = io.StringIO()
    code = main(argv, out=out, label_source=source, clock=FakeClock(now))
    return code, out.getvalue()


def _yaml_tail(text: str) -> dict[str, Any]:
    start = text.index("# §13.5 阈值重算建议")
    return yaml.safe_load(text[start:])


# ================================================================ recalibrate


def test_recalibrate_prints_summary_and_proposal(jobs_env, monkeypatch):
    monkeypatch.delenv("RODIT_PLATFORM")  # 离线任务不需要平台设置
    source = Source(feedback_labels())
    code, text = _run(["recalibrate"], source)
    assert code == 0 and source.calls == 1
    assert "只是建议" in text and "内置默认值" in text
    assert "size_advice" in text and "上调" in text
    assert REGRESSION_REMINDER in text
    proposal = _yaml_tail(text)
    assert set(proposal["routing"]["assist_threshold_by_intent"]) == {"size_advice"}
    merged = apply_patch(load_config(None).routing, proposal)
    assert merged.assist_threshold_for(Intent.SIZE_ADVICE) > 0.6
    assert "合成尺码问句" not in text  # 摘要里没有标注原文


def test_recalibrate_writes_proposal_file_against_deployed_config(jobs_env, tmp_path):
    target = tmp_path / "proposal.yaml"
    code, text = _run(
        ["recalibrate", "--config", str(REPO_CONFIG), "--out", str(target)],
        Source(feedback_labels()),
    )
    assert code == 0
    assert f"配置片段已写入：{target}" in text and str(REPO_CONFIG) in text
    assert "routing:" not in text  # 片段只写进文件
    deployed = load_config(REPO_CONFIG).routing
    merged = apply_patch(deployed, yaml.safe_load(target.read_text(encoding="utf-8")))
    assert merged.auto_threshold == deployed.auto_threshold
    assert merged.assist_threshold_for(Intent.SIZE_ADVICE) > deployed.assist_threshold


def test_recalibrate_config_from_env(jobs_env, monkeypatch):
    monkeypatch.setenv("RODIT_CONFIG", str(REPO_CONFIG))
    code, text = _run(["recalibrate"], Source(feedback_labels()))
    assert code == 0 and f"当前配置：{REPO_CONFIG}" in text


def test_recalibrate_window_arguments(jobs_env):
    source = Source(feedback_labels(days_ago=40))
    code, text = _run(["recalibrate"], source)
    assert code == 0 and "时间窗内 0 条" in text and "没有需要修改的阈值" in text
    code, text = _run(
        ["recalibrate", "--since", "2026-08-01", "--until", "2026-09-01T00:00+08:00"], source
    )
    assert code == 0
    assert "2026-08-01 00:00 ~ 2026-09-01 00:00" in text and "时间窗内 120 条" in text


def test_recalibrate_min_support(jobs_env):
    code, text = _run(["recalibrate", "--min-support", "500"], Source(feedback_labels()))
    assert code == 0 and "没有需要修改的阈值" in text and "少于 500 条" in text


@pytest.mark.parametrize(
    "argv",
    [
        ["recalibrate", "--since", "last tuesday"],
        ["recalibrate", "--min-support", "0"],
        ["recalibrate", "--min-support", "x"],
        ["train-classifier"],  # 缺 --out
        ["train-classifier", "--out", "x.json", "--min-samples", "-1"],
    ],
)
def test_bad_arguments_exit_2(jobs_env, argv):
    with pytest.raises(SystemExit) as exc:
        main(argv, out=io.StringIO(), label_source=Source())
    assert exc.value.code == 2


def test_recalibrate_bad_window_or_config(jobs_env, capsys, tmp_path):
    code, _ = _run(["recalibrate", "--since", "2026-09-10", "--until", "2026-09-01"], Source())
    assert code == serve.EXIT_STARTUP
    assert "早于" in capsys.readouterr().err
    code, _ = _run(["recalibrate", "--config", str(tmp_path / "absent.yaml")], Source())
    assert code == serve.EXIT_STARTUP
    bad = tmp_path / "bad.yaml"
    bad.write_text("routing:\n  auto_threshold:\n    greeting: 1.5\n", encoding="utf-8")
    code, _ = _run(["recalibrate", "--config", str(bad)], Source())
    assert code == serve.EXIT_STARTUP


@pytest.mark.parametrize(
    "error",
    [
        StorageUnavailable("数据库连接失败"),
        psycopg.OperationalError("connection to host secret-db.internal failed"),
    ],
)
def test_database_unavailable_exit_1(jobs_env, capsys, error):
    code, _ = _run(["recalibrate"], Source(error=error))
    assert code == serve.EXIT_FAILURE
    err = capsys.readouterr().err
    assert "数据库不可用" in err and "secret-db" not in err  # 只记类型


def test_missing_trainer_privilege_exit_1(jobs_env, capsys, tmp_path):
    error = psycopg.errors.InsufficientPrivilege("permission denied to set role")
    target = tmp_path / "never-written.json"
    code, _ = _run(["train-classifier", "--out", str(target)], Source(error=error))
    assert code == serve.EXIT_FAILURE
    assert "rodit_trainer" in capsys.readouterr().err
    assert not target.exists()


def test_default_label_source_needs_trainer_dsn(jobs_env, capsys):
    out = io.StringIO()
    assert main(["recalibrate"], out=out, clock=FakeClock(NOW)) == serve.EXIT_STARTUP
    err = capsys.readouterr().err
    assert "RODIT_TRAINER_DSN" in err and SYNTHETIC_TOKEN_KEY not in err


def test_default_label_source_unreachable_database(jobs_env, monkeypatch, capsys):
    monkeypatch.setenv("RODIT_TRAINER_DSN", "host=127.0.0.1 port=1 dbname=rodit_nowhere")
    out = io.StringIO()
    assert main(["recalibrate"], out=out, clock=FakeClock(NOW)) == serve.EXIT_FAILURE
    err = capsys.readouterr().err
    assert "数据库不可用" in err and "rodit_nowhere" not in err


# ================================================================ train-classifier


def _artifact_files(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def test_train_classifier_writes_loadable_artifact(jobs_env, tmp_path):
    target = tmp_path / "classifier-2026-09.json"
    code, text = _run(["train-classifier", "--out", str(target)], Source(training_labels()))
    assert code == 0, text
    assert _artifact_files(tmp_path) == [target.name]  # 原子写入，没有临时文件
    loaded = ClassifierProvider.load(target, fallback=RuleProvider())
    assert loaded.meta is not None and loaded.meta.trained_at == NOW
    assert f"已写入模型文件：{target}" in text and loaded.artifact_sha256 in text
    assert "关键词规则" in text and "留出集" in text
    assert CLASSIFIER_REGRESSION_REMINDER in text and "--classifier" in text


def test_train_classifier_refuses_existing_file_without_overwrite(jobs_env, tmp_path):
    target = tmp_path / "classifier.json"
    target.write_text("线上正在用的模型", encoding="utf-8")
    source = Source(training_labels())
    code, _ = _run(["train-classifier", "--out", str(target)], source)
    assert code == serve.EXIT_STARTUP
    assert source.calls == 0  # 先检查输出文件，不白跑导出
    assert target.read_text(encoding="utf-8") == "线上正在用的模型"


def test_train_classifier_overwrite_rules(jobs_env, tmp_path):
    target = tmp_path / "classifier.json"
    source = Source(training_labels(n_prefix=2))
    assert _run(["train-classifier", "--out", str(target)], source)[0] == 0
    first = read_artifact(target)

    # 已有的更新：不覆盖
    code, _ = _run(
        ["train-classifier", "--out", str(target), "--overwrite"], source, now=NOW - timedelta(1)
    )
    assert code == serve.EXIT_STARTUP and read_artifact(target) == first

    # 已有的更旧：覆盖
    later = NOW + timedelta(days=30)
    code, _ = _run(["train-classifier", "--out", str(target), "--overwrite"], source, now=later)
    assert code == 0
    assert read_artifact(target)["metadata"]["trained_at"] == later.isoformat()

    # 读不出来的文件：--overwrite 时照常覆盖
    target.write_text("{broken", encoding="utf-8")
    code, _ = _run(["train-classifier", "--out", str(target), "--overwrite"], source)
    assert code == 0 and read_artifact(target)["format"] == "rodit.route_classifier"
    assert _artifact_files(tmp_path) == ["classifier.json"]


def test_train_classifier_minimum_data(jobs_env, tmp_path, capsys):
    few = training_labels(n_prefix=1)[:150]
    target = tmp_path / "classifier.json"
    code, _ = _run(["train-classifier", "--out", str(target)], Source(few))
    assert code == serve.EXIT_STARTUP and not target.exists()
    assert "少于下限 200" in capsys.readouterr().err
    code, text = _run(["train-classifier", "--out", str(target), "--force"], Source(few))
    assert code == 0 and "--force" in text and target.exists()
    other = tmp_path / "classifier-2.json"
    code, _ = _run(["train-classifier", "--out", str(other), "--min-samples", "100"], Source(few))
    assert code == 0 and other.exists()


def test_train_classifier_missing_directory(jobs_env, tmp_path):
    code, _ = _run(
        ["train-classifier", "--out", str(tmp_path / "no" / "dir.json")],
        Source(training_labels()),
    )
    assert code == serve.EXIT_STARTUP


# ================================================================ serve：路由分类器上线


@pytest.fixture
def classifier_artifact(tmp_path) -> Path:
    samples = [(c.text, c.expected_intent) for c in cases("dev")]
    clf = ClassifierProvider(RuleProvider()).fit(samples, seed_examples=True)
    path = tmp_path / "models" / "classifier.json"
    path.parent.mkdir()
    clf.save(path, trained_at=NOW)
    return path


@pytest.fixture
def classifier_config(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "rodit.yaml"
    config = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
    config["decision"]["provider"] = "classifier"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("RODIT_CONFIG", str(path))
    return path


def _serve_fails(capsys) -> str:
    ran: list[Any] = []
    assert main(["serve"], run=lambda *a, **k: ran.append(a)) == serve.EXIT_STARTUP
    assert not ran
    err = capsys.readouterr().err
    record = json.loads(err.strip().splitlines()[-1])
    assert record["level"] == "ERROR" and "启动失败" in record["msg"]
    assert SYNTHETIC_TOKEN_KEY not in err and SYNTHETIC_LLM_KEY not in err
    return record["msg"]


def test_serve_with_classifier_requires_artifact_path(
    jobs_env, token_file, classifier_config, monkeypatch, capsys
):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    assert "RODIT_CLASSIFIER_PATH" in _serve_fails(capsys)


def test_serve_with_missing_artifact_refuses(
    jobs_env, token_file, classifier_config, monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_CLASSIFIER_PATH", str(tmp_path / "absent.json"))
    assert "不存在" in _serve_fails(capsys)


def test_serve_with_tampered_artifact_refuses(
    jobs_env, token_file, classifier_config, classifier_artifact, monkeypatch, capsys
):
    artifact = read_artifact(classifier_artifact)
    artifact["model"]["alpha"] = 0.5  # 改了参数没重算校验和
    classifier_artifact.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_CLASSIFIER_PATH", str(classifier_artifact))
    assert "校验和不符" in _serve_fails(capsys)


def test_serve_with_unknown_intent_artifact_refuses(
    jobs_env, token_file, classifier_config, classifier_artifact, monkeypatch, capsys
):
    from rodit.decision.classifier import artifact_digest

    artifact = read_artifact(classifier_artifact)
    artifact["model"]["classes"].append("refund_fast_track")
    artifact["sha256"] = artifact_digest(artifact)
    write_artifact(classifier_artifact, artifact)
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_CLASSIFIER_PATH", str(classifier_artifact))
    assert "不认识的问题类型" in _serve_fails(capsys)


def test_production_unit_routes_with_loaded_classifier(
    jobs_env, token_file, classifier_config, classifier_artifact, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="rodit.ops")
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_CLASSIFIER_PATH", str(classifier_artifact))
    server = build_server(ServerSettings.from_env(), database=FakeReadinessDb())
    unit = server.unit
    router_provider = unit.router.provider
    assert isinstance(router_provider, ClassifierProvider)
    assert router_provider.artifact_sha256 == read_artifact(classifier_artifact)["sha256"]
    assert isinstance(unit.router.fallback, RuleProvider)
    assert isinstance(unit.gate._provider, InCloudLLMProvider)  # type: ignore[attr-defined]  质检仍用大模型
    assert "路由分类器已加载" in caplog.text
    ran: list[Any] = []
    assert main(["serve"], run=lambda app, **kw: ran.append(app), lease_factory=FakeServeLease) == 0
    assert ran


def test_classifier_path_ignored_for_other_providers(
    jobs_env, token_file, classifier_artifact, monkeypatch, caplog
):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_CLASSIFIER_PATH", str(classifier_artifact))
    server = build_server(ServerSettings.from_env(), database=FakeReadinessDb())
    assert isinstance(server.unit.router.provider, InCloudLLMProvider)
    assert "不加载路由分类器" in caplog.text


def test_local_unit_answers_with_classifier_route(
    jobs_env, token_file, classifier_config, classifier_artifact, monkeypatch
):
    """本地试用（离线替身、内存仓储）：路由走分类器，decision_log 记下 provider=classifier。"""
    for name in ("RODIT_LLM_BASE_URL", "RODIT_LLM_MODEL", "RODIT_LLM_API_KEY"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("RODIT_ENV", "local")
    monkeypatch.setenv("RODIT_CLASSIFIER_PATH", str(classifier_artifact))
    world = build_demo_world()
    clock = FakeClock(world.now)
    server = build_server(
        ServerSettings.from_env(),
        clock=clock,
        repos=in_memory_repos(clock),
        database=FakeReadinessDb(),
    )
    seed_unit(server.unit, world)
    assert isinstance(server.unit.router.provider, ClassifierProvider)
    client = TestClient(server.app)
    ask = client.post(
        f"/v1/shops/{SHOP_A}/standalone/ask",
        json={"session_id": "web_clf", "question": "能开发票吗"},
        headers=token_file.auth("agent_a"),
    )
    assert ask.status_code == 200, ask.text
    assert ask.json()["suggestions"]
    (entry,) = server.unit.repos.audit.list_decisions(SHOP_A, stage=DecisionStage.ROUTE)
    assert entry.provider == "classifier"
    assert entry.answers["answers"]["intent"]["choice"] == Intent.INVOICE_GIFT.value


# ================================================================ PostgreSQL：rodit_trainer 导出


@pytest.mark.pg
def test_jobs_read_labels_through_trainer_role(jobs_env, ops_pg_dsn, monkeypatch, tmp_path):
    db = Database(ops_pg_dsn)
    db.migrate()
    repo = PgFeedbackRepository(db)  # 按店铺写入（rodit_app 角色，行级安全）
    for label in feedback_labels():
        repo.add_label(label)
    for label in training_labels(n_prefix=1)[:60]:
        repo.add_label(label)
    monkeypatch.setenv("RODIT_TRAINER_DSN", ops_pg_dsn)  # 测试库的管理员账号能 SET ROLE
    out = io.StringIO()
    assert main(["recalibrate"], out=out, clock=FakeClock(NOW)) == 0
    text = out.getvalue()
    # 库里按时间窗导出（[since, until)，created_at 正好等于 now 的那条不在窗里）
    assert "导出 179 条，时间窗内 179 条" in text and "来自 2 家店铺" in text and "上调" in text
    target = tmp_path / "classifier.json"
    out = io.StringIO()
    argv = ["train-classifier", "--out", str(target), "--min-samples", "20"]
    assert main(argv, out=out, clock=FakeClock(NOW)) == 0, out.getvalue()
    assert ClassifierProvider.load(target, fallback=RuleProvider()).is_trained
