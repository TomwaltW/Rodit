"""模式 C 网页工具的真浏览器冒烟测试（Chromium + Playwright，§8.1）。

在本进程里起一个只监听 127.0.0.1 随机端口的服务器（合成演示数据、离线替身，不联网），
用 node 跑 web_smoke.cjs：登录、提问、三种反馈、过期、缺失知识、XSS、影子模式和 503 的显示、
手机宽度、退出、商家报表和问卷。没有 node / playwright 模块 / 浏览器时跳过（不新增 Python 依赖：
用镜像里全局安装的 Node 版 Playwright）。
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
from api_support import TOKENS, Api

from rodit.core.enums import SuggestionStatus
from rodit.testing.fixtures import AGENT, SHOP_A

NODE = shutil.which("node")
SCRIPT = Path(__file__).with_name("web_smoke.cjs")
SKIP_EXIT = 77
TIMEOUT_S = 180


def _node_env() -> dict[str, str]:
    """全局安装的 node 模块（playwright）加进 NODE_PATH；
    浏览器位置沿用 PLAYWRIGHT_BROWSERS_PATH。"""
    env = dict(os.environ)
    assert NODE is not None
    global_modules = Path(NODE).resolve().parent.parent / "lib" / "node_modules"
    env["NODE_PATH"] = os.pathsep.join(p for p in (env.get("NODE_PATH"), str(global_modules)) if p)
    return env


@contextmanager
def _serve(app) -> Iterator[str]:
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


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过真浏览器冒烟测试")
def test_assist_page_in_a_real_browser(api: Api):
    env = _node_env()
    tokens = {
        "agent": TOKENS["agent_a"],
        "merchant": TOKENS["merchant_a"],
        "integration": TOKENS["integration_a"],
    }
    with _serve(api.client.app) as base:
        env.update(RODIT_BASE_URL=base, RODIT_SMOKE_TOKENS=json.dumps(tokens))
        proc = subprocess.run(
            [str(NODE), str(SCRIPT)],
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
    assert result["steps"][-1] == "merchant"

    # 页面上的操作都落到了服务端（客服账号取自令牌对应的身份）
    records = api.unit.repos.suggestions.list(SHOP_A)
    statuses = [r.status for r in records]
    assert statuses.count(SuggestionStatus.SENT) == 1
    assert statuses.count(SuggestionStatus.EDITED_SENT) == 1
    assert statuses.count(SuggestionStatus.REJECTED) == 1
    assert SuggestionStatus.EXPIRED in statuses
    acted = [r for r in records if r.status != SuggestionStatus.EXPIRED and r.agent_id]
    assert acted and {r.agent_id for r in acted} == {AGENT}
    policies = api.unit.repos.knowledge.get_policies(SHOP_A)
    assert policies["warranty"].value == "保修一年，联系客服寄回维修"
