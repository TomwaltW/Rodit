"""客户演示站的真浏览器冒烟测试（Chromium + Playwright）。

在本进程里起一个只监听 127.0.0.1 随机端口的演示站（回放了合成历史、离线替身，不联网），用 node 跑
showcase_smoke.cjs：产品概览、工作台（推荐卡片、过程面板、发送 / 编辑后发送 / 不合适、质检拦截、
买家打字、上线节奏和半自动的"已填入输入框"、买家图片：场景里的破损照片和物流截图、自己选图发送）、
售后审核（确认框、重读状态、幂等、取消、凭证图片理解）、数据看板（图表、悬停提示、平台指标的总体
结论和按周序列）、合规页的打码演示、切换平台（拼多多不能发图片）、页面载入后演示被重置过时旧卡片的
操作被拒（409，提示并重新载入）、重置、手机宽度没有横向滚动，
全程没有控制台报错、没有 CSP 拦截、没有外部请求。
另一个用例（showcase_llm_smoke.cjs）让演示站接"真实大模型"——本机的 OpenAI 兼容替身
（llm_stub.py，真实的 OpenAICompatibleClient 走真实 HTTP）：徽标、判断层降级的说明、真实的票数。
设了 RODIT_SHOWCASE_SHOTS（目录）时把各页的整页截图存进去。
没有 node / playwright 模块 / 浏览器时跳过（不新增 Python 依赖：用镜像里全局安装的
Node 版 Playwright）。
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
from llm_stub import StubLLM
from showcase_support import NOW

from rodit.core.enums import AfterSaleStatus, Platform, SuggestionStatus
from rodit.eval.llm_env import client_from_env
from rodit.showcase.app import create_showcase_app
from rodit.showcase.state import ShowcaseHub
from rodit.testing.fakes import FakeClock

NODE = shutil.which("node")
SCRIPT = Path(__file__).with_name("showcase_smoke.cjs")
LLM_SCRIPT = Path(__file__).with_name("showcase_llm_smoke.cjs")
SKIP_EXIT = 77
TIMEOUT_S = 240


def _node_env() -> dict[str, str]:
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


def _run_node(script: Path, app, extra_env: dict[str, str] | None = None) -> dict:
    """起服务、跑 node 脚本，返回脚本最后一行的 JSON 结果（没有 Playwright 时跳过）。"""
    env = _node_env()
    env.update(extra_env or {})
    with _serve(app) as base:
        env["RODIT_BASE_URL"] = base
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


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过真浏览器冒烟测试")
def test_showcase_in_a_real_browser():
    hub = ShowcaseHub(clock=FakeClock(NOW))
    result = _run_node(SCRIPT, create_showcase_app(hub))
    assert result["steps"] == [
        "landing",
        "suggestion-trace",
        "send",
        "qa-block",
        "feedback",
        "typing",
        "prefill",
        "image",
        "aftersale",
        "dashboard",
        "compliance",
        "platform-switch",
        "stale-epoch",
        "reset",
        "phone",
    ]

    # 页面上的操作都落到了服务端
    taobao = hub.get(Platform.TAOBAO)
    live = [
        r
        for r in taobao.unit.repos.suggestions.list(taobao.shop_id)
        if r.conversation_id.startswith("sc-")
    ]
    statuses = {r.conversation_id: r.status for r in live if r.status != SuggestionStatus.SHOWN}
    assert statuses["sc-size"] == SuggestionStatus.SENT
    assert statuses["sc-invoice"] == SuggestionStatus.EDITED_SENT
    assert statuses["sc-price"] == SuggestionStatus.REJECTED
    assert statuses["sc-ship_quick"] == SuggestionStatus.SENT  # 半自动填进输入框，客服按的发送
    # 买家图片走了图片理解（合成数据的替身）：场景预设的、自己选的
    calls = {ref for _, ref in taobao.unit.vision.calls}
    assert {"img_buyer_damage_1", "img_buyer_logistics_1", "img_buyer_size_1"} <= calls
    info = taobao.unit.tools.get_aftersale(taobao.shop_id, "as_2001")
    assert info is not None and info.status == AfterSaleStatus.REFUNDING
    assert len(taobao.unit.action_port.calls) == 1  # 再提交一次没有重复操作（幂等）
    untouched = taobao.unit.tools.get_aftersale(taobao.shop_id, "as_2002")
    assert untouched is not None and untouched.status == AfterSaleStatus.PENDING_REVIEW


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过真浏览器冒烟测试")
def test_showcase_with_the_real_llm_chain_in_a_real_browser(stub_env: StubLLM):
    """--llm env --judge llm：徽标、降级说明、真实票数（大模型是本机的 OpenAI 兼容替身）。"""
    stub_env.fail("route", "429", times=3)  # 第一条消息的三票都被限流：页面要写明退回了关键词规则
    hub = ShowcaseHub(
        clock=FakeClock(NOW),
        llm=client_from_env(),
        judge="llm",
        latency="strict",
        history_seed=None,
    )
    result = _run_node(
        LLM_SCRIPT,
        create_showcase_app(hub),
        {
            "RODIT_LLM_MODEL": stub_env.model,
            "RODIT_EXPECT_HOST": stub_env.base_url.removesuffix("/v1"),
        },
    )
    assert result["steps"] == ["badge", "fallback", "votes", "phone"]
    # 页面上的操作真的走到了模型替身：每条消息路由 3 个请求，生成、质检都有
    assert stub_env.count("route") >= 9 and stub_env.count("generate") >= 1
    assert stub_env.count("qa") >= 1
