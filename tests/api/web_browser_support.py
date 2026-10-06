"""真浏览器冒烟测试的公共部分（Chromium + Playwright）：在本进程里起一个只监听 127.0.0.1 随机端口的
服务器（合成演示数据、离线替身，不联网），用 node 跑冒烟脚本，读脚本最后一行的 JSON 结果。

没有 node / playwright 模块 / 浏览器时跳过（不新增 Python 依赖：用镜像里全局安装的 Node 版
Playwright）。脚本约定：退出码 0 通过、77 跳过、其他失败；标准输出最后一行是 {"ok": …, "steps": …}。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn

NODE = shutil.which("node")
SKIP_EXIT = 77
TIMEOUT_S = 180


def node_env() -> dict[str, str]:
    """全局安装的 node 模块（playwright）加进 NODE_PATH；
    浏览器位置沿用 PLAYWRIGHT_BROWSERS_PATH。"""
    env = dict(os.environ)
    assert NODE is not None
    global_modules = Path(NODE).resolve().parent.parent / "lib" / "node_modules"
    env["NODE_PATH"] = os.pathsep.join(p for p in (env.get("NODE_PATH"), str(global_modules)) if p)
    return env


@contextmanager
def serve(app) -> Iterator[str]:
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="warning", lifespan="off", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("测试服务器没有启动")
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def run_smoke(app, script: Path, **env_extra: str) -> dict:
    """起服务器、用 node 跑冒烟脚本，返回脚本最后一行的 JSON 结果（没有 Playwright 时跳过）。"""
    env = node_env()
    with serve(app) as base:
        env.update(RODIT_BASE_URL=base, **env_extra)
        proc = subprocess.run(
            [str(NODE), str(script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    if proc.returncode == SKIP_EXIT:
        pytest.skip(f"没有 Playwright 或浏览器起不来：{proc.stderr.strip()[:300]}")
    lines = proc.stdout.strip().splitlines()
    result = json.loads(lines[-1]) if lines else {"ok": False, "error": proc.stderr}
    assert proc.returncode == 0 and result["ok"], result.get("error") or proc.stderr
    return result


__all__ = ["NODE", "SKIP_EXIT", "TIMEOUT_S", "node_env", "run_smoke", "serve"]
