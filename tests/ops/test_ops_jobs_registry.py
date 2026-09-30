"""离线任务注册表（rodit.ops.jobs）：serve 的命令行加上各任务模块的子命令，main 把命令交给认领它的
模块。T0 的两个模块（tenancy、rollout）还没有命令：已有的命令行照旧。"""

from __future__ import annotations

import argparse
import io
import sys
import types
from collections.abc import Mapping

import pytest

import rodit.ops.jobs as jobs
from rodit.ops import serve
from rodit.ops.jobs import rollout as rollout_jobs
from rodit.ops.jobs import tenancy as tenancy_jobs

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


def test_registry_lists_both_job_modules():
    assert jobs.JOB_MODULES == ("rodit.ops.jobs.tenancy", "rodit.ops.jobs.rollout")
    assert jobs.modules() == [tenancy_jobs, rollout_jobs]


def test_stub_modules_register_nothing_and_claim_nothing():
    assert _commands(serve._parser()) == BUILTIN  # noqa: SLF001
    args = serve._parser().parse_args(["migrate"])  # noqa: SLF001
    assert jobs.dispatch(args, {}) is None
    assert tenancy_jobs.run(args, {}) is None and rollout_jobs.run(args, {}) is None


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
