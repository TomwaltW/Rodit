"""离线任务注册表（rodit.ops.jobs）：serve 的命令行加上各任务模块的子命令，main 把命令交给认领它的
模块，任务抛出的异常和内置命令一样映射退出码。任务模块加了命令之后，内置命令照旧都在。"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import types
from collections.abc import Mapping

import psycopg
import pytest

import rodit.ops.jobs as jobs
from rodit.core.errors import StorageUnavailable
from rodit.ops import serve
from rodit.ops.jobs import platform_metrics as metrics_jobs
from rodit.ops.jobs import rollout as rollout_jobs
from rodit.ops.jobs import tenancy as tenancy_jobs
from rodit.ops.settings import SettingsError

BUILTIN = {
    "serve",
    "migrate",
    "local-init",
    "healthcheck",
    "issue-token",
    "recalibrate",
    "train-classifier",
}


def _commands(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:  # noqa: SLF001 — 只读 argparse 的子命令表
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return set(action.choices)
    return set()


def test_registry_lists_the_job_modules():
    assert jobs.JOB_MODULES == (
        "rodit.ops.jobs.tenancy",
        "rodit.ops.jobs.rollout",
        "rodit.ops.jobs.platform_metrics",
    )
    assert jobs.modules() == [tenancy_jobs, rollout_jobs, metrics_jobs]


def test_builtin_commands_stay_with_job_modules_registered(monkeypatch):
    # 各任务模块加了自己的命令之后，内置命令照旧都在
    assert BUILTIN <= _commands(serve._parser())  # noqa: SLF001
    # 不注册任务模块时只有内置命令
    monkeypatch.setattr(jobs, "JOB_MODULES", ())
    assert _commands(serve._parser()) == BUILTIN  # noqa: SLF001


def test_job_modules_do_not_claim_builtin_commands():
    for argv in (["migrate"], ["serve"], ["issue-token", "--metrics"]):
        args = serve._parser().parse_args(argv)  # noqa: SLF001
        assert jobs.dispatch(args, {}) is None
        assert tenancy_jobs.run(args, {}) is None and rollout_jobs.run(args, {}) is None
        assert metrics_jobs.run(args, {}) is None


@pytest.fixture
def fake_job(monkeypatch):
    """一个临时的任务模块：注册 fake-job 命令，认领后返回 7。"""
    seen: list[tuple[argparse.Namespace, Mapping[str, str]]] = []
    module = types.ModuleType("rodit_test_fake_job")

    def register(subparsers) -> None:
        cmd = subparsers.add_parser("fake-job", help="测试用")
        cmd.add_argument("--flag", action="store_true")

    def run(args: argparse.Namespace, environ: Mapping[str, str]) -> int | None:
        if args.command != "fake-job":
            return None
        seen.append((args, environ))
        return 7

    module.register = register  # type: ignore[attr-defined]
    module.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(jobs, "JOB_MODULES", (*jobs.JOB_MODULES, module.__name__))
    return seen


def test_serve_parser_and_main_dispatch_to_job_modules(fake_job, ops_env):
    assert "fake-job" in _commands(serve._parser())  # noqa: SLF001
    assert serve.main(["fake-job", "--flag"]) == 7
    [(args, environ)] = fake_job
    assert args.flag is True and environ.get("RODIT_PLATFORM") == ops_env["RODIT_PLATFORM"]
    # 别的命令不交给它
    assert serve.main(["issue-token", "--metrics"], out=io.StringIO()) == 0
    assert len(fake_job) == 1


def test_broken_job_module_fails_loudly(monkeypatch):
    broken = types.ModuleType("rodit_test_broken_job")
    broken.register = lambda subparsers: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, broken.__name__, broken)
    with pytest.raises(TypeError, match="run"):
        jobs.modules((broken.__name__,))


def _raising_job(monkeypatch, exc: BaseException) -> None:
    """临时的任务模块：认领 boom-job，执行时抛 exc。"""
    module = types.ModuleType("rodit_test_raising_job")

    def register(subparsers) -> None:
        subparsers.add_parser("boom-job", help="测试用")

    def run(args: argparse.Namespace, environ: Mapping[str, str]) -> int | None:
        if args.command != "boom-job":
            return None
        raise exc

    module.register = register  # type: ignore[attr-defined]
    module.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(jobs, "JOB_MODULES", (*jobs.JOB_MODULES, module.__name__))


@pytest.mark.parametrize(
    "exc",
    [
        psycopg.OperationalError(
            'connection to server at "10.0.0.5", port 5432 failed: FATAL: '
            'password authentication failed for user "rodit_owner"'
        ),
        StorageUnavailable("数据库 10.0.0.5 连不上（rodit_owner）"),
    ],
)
def test_job_database_errors_map_to_failure_and_log_only_the_type(
    monkeypatch, ops_env, caplog, exc
):
    """任务里的数据库错误：退出码 EXIT_FAILURE，日志只有异常类型，没有主机名、账号。"""
    caplog.set_level(logging.INFO)
    _raising_job(monkeypatch, exc)
    assert serve.main(["boom-job"]) == serve.EXIT_FAILURE
    assert type(exc).__name__ in caplog.text
    assert "10.0.0.5" not in caplog.text and "rodit_owner" not in caplog.text


@pytest.mark.parametrize(
    "exc",
    [
        SettingsError("缺少环境变量 RODIT_X"),
        RuntimeError("缺少环境变量 RODIT_Y"),
        ValueError("--days 必须是正数"),
        OSError("配置文件不存在"),
    ],
)
def test_job_startup_errors_map_to_startup_exit_code(monkeypatch, ops_env, caplog, exc):
    caplog.set_level(logging.INFO)
    _raising_job(monkeypatch, exc)
    assert serve.main(["boom-job"]) == serve.EXIT_STARTUP
    assert str(exc) in caplog.text
