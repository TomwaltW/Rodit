"""离线任务的注册表（python -m rodit.ops 的子命令，在平台单元内跑）。

每个任务模块提供两个函数：

    register(subparsers) -> None
        往 argparse 的子命令里加自己的命令（subparsers.add_parser(...)）；
    run(args, environ) -> int | None
        args.command 是自己的命令时执行并返回退出码；不是自己的命令返回 None。

rodit.ops.serve 的 _parser 对每个模块调 register，main 解析完参数后按顺序调 run，第一个
返回退出码的模块处理了这条命令（dispatch）。模块按 JOB_MODULES 的顺序导入，只在用到时导入
（serve 不会因为任务模块的依赖而多导入东西）。

run 抛出的异常由 main 按内置命令的约定映射退出码，任务模块不用各自再写一遍：
设置 / 启动 / 绑定错误、缺环境变量（RuntimeError）、配置值不对（ValueError、OSError、YAML）
→ EXIT_STARTUP，日志记异常说明；数据库错误（StorageUnavailable、psycopg.Error）→ EXIT_FAILURE，
日志只记异常类型（libpq 的说明里有主机名、账号）。所以异常说明里不能放令牌等机密；
需要更具体的退出码时 run 自己处理后返回。

- rodit.ops.jobs.tenancy：店铺生命周期（开通、续期、过期清理、取消订购，§8.4）；
- rodit.ops.jobs.rollout：上线节奏（评测结论、验收报告，§13.4、§13.5）。
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import Any, Protocol, cast

JOB_MODULES: tuple[str, ...] = ("rodit.ops.jobs.tenancy", "rodit.ops.jobs.rollout")


class JobModule(Protocol):
    """任务模块的约定（见模块说明）。"""

    def register(self, subparsers: Any) -> None: ...

    def run(self, args: argparse.Namespace, environ: Mapping[str, str]) -> int | None: ...


def modules(names: tuple[str, ...] | None = None) -> list[JobModule]:
    """按顺序导入任务模块（默认 JOB_MODULES，调用时读取）。缺了 register / run 的模块是
    写错了，直接报错（不静默跳过）。"""
    loaded: list[JobModule] = []
    for name in JOB_MODULES if names is None else names:
        module: ModuleType = importlib.import_module(name)
        for attr in ("register", "run"):
            if not callable(getattr(module, attr, None)):
                raise TypeError(f"任务模块 {name} 没有 {attr}()")
        loaded.append(cast(JobModule, module))
    return loaded


def register_all(subparsers: Any, names: tuple[str, ...] | None = None) -> None:
    """把各任务模块的子命令加进 serve 的命令行。"""
    for module in modules(names):
        module.register(subparsers)


def dispatch(
    args: argparse.Namespace,
    environ: Mapping[str, str],
    names: tuple[str, ...] | None = None,
) -> int | None:
    """交给第一个认领这条命令的任务模块执行，返回它的退出码；没有模块认领返回 None。"""
    for module in modules(names):
        code = module.run(args, environ)
        if code is not None:
            return code
    return None


__all__ = ["JOB_MODULES", "JobModule", "dispatch", "modules", "register_all"]
