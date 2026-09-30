"""店铺生命周期的离线任务（方案 §8.4）：onboard-shop、refresh-tokens、purge-expired、revoke。

T0 只放空的骨架（不注册任何命令），由 Track A 实现。约定见 rodit.ops.jobs：
register(subparsers) 加命令；run(args, environ) 处理自己的命令、返回退出码，别的命令返回 None。
令牌只从标准输入或文件读（--token-stdin），不进命令行参数和日志。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any


def register(subparsers: Any) -> None:
    """还没有命令。"""


def run(args: argparse.Namespace, environ: Mapping[str, str]) -> int | None:
    """还没有命令：一律不认领。"""
    return None
