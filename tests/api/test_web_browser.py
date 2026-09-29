"""网页工具的真浏览器冒烟测试（Chromium + Playwright）：模式 C 话术助手（§8.1）和商家控制台
（§9.1、§5.2、§9.3、§13.4、§13.5）。

在本进程里起一个只监听 127.0.0.1 随机端口的服务器（合成演示数据、离线替身，不联网），用 node 跑：
- web_smoke.cjs：话术助手的登录、提问、三种反馈、过期、缺失知识、XSS、影子模式和 503 的显示、
  手机宽度、退出、商家报表和问卷；
- web_console_smoke.cjs：控制台的登录限制（平台对接、客服）、店长只读、商家的知识 / 政策 / 商品 /
  同步 / 设置 / 切回影子 / 报表、复核人员的抽样打分和强制切换、复核人员在话术助手页的提示；
- web_state_smoke.cjs（完整版 Chromium，打开往返缓存）：一个页面退出或换人登录后按「后退」，
  另一个页面不再用旧令牌；拒绝的角色能退出；写操作回来时原地刷新正显示的页签；晚回来的旧响应
  不覆盖新的；上一位没完成的同步不挡下一位；同步后保留还没保存的商家字段。
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
from api_support import TOKENS, Api

from rodit.core.audit import AuditAction
from rodit.core.enums import KnowledgeStatus, QualityReviewStatus, SuggestionStatus
from rodit.feedback.review import iso_period, previous_period
from rodit.testing.fixtures import AGENT, SHOP_A

NODE = shutil.which("node")
SCRIPT = Path(__file__).with_name("web_smoke.cjs")
CONSOLE_SCRIPT = Path(__file__).with_name("web_console_smoke.cjs")
STATE_SCRIPT = Path(__file__).with_name("web_state_smoke.cjs")
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


def _run_smoke(api: Api, script: Path, **env_extra: str) -> dict:
    """起服务器、用 node 跑冒烟脚本，返回脚本最后一行的 JSON 结果（没有 Playwright 时跳过）。"""
    env = _node_env()
    with _serve(api.client.app) as base:
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


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过真浏览器冒烟测试")
def test_assist_page_in_a_real_browser(api: Api):
    tokens = {
        "agent": TOKENS["agent_a"],
        "merchant": TOKENS["merchant_a"],
        "integration": TOKENS["integration_a"],
    }
    result = _run_smoke(api, SCRIPT, RODIT_SMOKE_TOKENS=json.dumps(tokens))
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


SEED_QUESTIONS = ("能开发票吗", "什么时候发货", "有运费险吗")


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过真浏览器冒烟测试")
def test_console_page_in_a_real_browser(api: Api):
    """控制台：店长只读；商家改知识、政策、商品、设置、切回影子；复核人员抽样打分、强制切到智能辅助。"""
    base = f"/v1/shops/{SHOP_A}"
    # 复核要有推荐可抽：按话术助手的调用方式提几个问题（本周生成，演示时钟固定）
    for i, question in enumerate(SEED_QUESTIONS):
        r = api.post(
            f"{base}/standalone/ask",
            "agent_a",
            json={"session_id": f"seed-{i}", "question": question, "request_id": f"seed-{i}"},
        )
        assert r.status_code == 200 and r.json()["suggestions"], r.text
    now = api.unit.clock.now()
    tokens = {
        "merchant": TOKENS["merchant_a"],
        "manager": TOKENS["manager_a"],
        "reviewer": TOKENS["reviewer_a"],
        "agent": TOKENS["agent_a"],
        "integration": TOKENS["integration_a"],
    }
    result = _run_smoke(
        api,
        CONSOLE_SCRIPT,
        RODIT_SMOKE_TOKENS=json.dumps(tokens),
        RODIT_SMOKE_PERIOD=iso_period(now),
        RODIT_SMOKE_DEFAULT_PERIOD=previous_period(now),
    )
    assert result["steps"] == [
        "login-guards",
        "manager",
        "knowledge",
        "policies",
        "items",
        "config",
        "merchant",
        "review",
        "stage",
        "assist-link",
    ]

    # 页面上的操作都落到了服务端，操作人取自令牌对应的身份
    repos = api.unit.repos
    faqs = repos.knowledge.list_knowledge(SHOP_A, status=None)
    by_id = {k.id: k for k in faqs}
    approved = [k for k in faqs if k.status == KnowledgeStatus.APPROVED]
    created = [k for k in approved if "发票抬头可以写公司吗" in k.question]
    assert len(created) == 1 and created[0].approved_by == "owner_a"
    assert created[0].intent is not None and created[0].intent.value == "invoice_gift"
    assert by_id[1].status == KnowledgeStatus.RETIRED  # 修改：旧条目下线
    assert any(k.answer == "亲，默认发中通快递，顺丰补 10 元哦~" for k in approved)
    assert by_id[6].status == KnowledgeStatus.RETIRED  # 审核时修改：候选下线、新条目上线
    assert any(k.answer == "亲，新疆可以发，补 15 元运费哦~" for k in approved)
    # 政策先填写、再清空（不再适用）：库里没有了，审计里两次都有
    assert "warranty" not in repos.knowledge.get_policies(SHOP_A)
    policy_events = repos.audit.list_events(SHOP_A, action="knowledge.policy_updated")
    assert [e.detail.get("key") for e in policy_events][-2:] == ["warranty", "warranty"]
    assert policy_events[-1].detail.get("cleared") is True
    card = repos.knowledge.get_item_card(SHOP_A, "i_777")
    assert card is not None and card.size_advice == "版型偏大，建议拍小一码"
    assert card.care == ["建议手洗", "不可暴晒"]
    assert repos.knowledge.get_item_card(SHOP_A, "i_999") is not None  # 同步新建的商品
    shop = repos.knowledge.get_shop(SHOP_A)
    assert shop is not None
    overrides = shop.config_overrides
    assert overrides["tone"] == "专业" and "suggestion_count" not in overrides
    assert overrides["disabled_intents"] == ["greeting"]
    assert overrides.get("shadow_mode") is False  # 商家切回影子 → 复核人员强制切到智能辅助
    stages = repos.audit.list_events(SHOP_A, action=AuditAction.ROLLOUT_STAGE.value)
    assert [(e.actor, e.detail["to"], e.detail["forced"]) for e in stages] == [
        ("owner_a", "shadow", False),
        ("qa_reviewer_01", "assist", True),
    ]
    assert stages[-1].detail["note"] == "测试集已人工确认达标"
    reviews = repos.quality_reviews.list(SHOP_A, period=iso_period(now))
    assert len(reviews) == result["samples"]
    done = {q.status: q for q in reviews if q.status != QualityReviewStatus.PENDING}
    assert set(done) == {QualityReviewStatus.SCORED, QualityReviewStatus.SKIPPED}
    scored = done[QualityReviewStatus.SCORED]
    assert scored.reviewer == "qa_reviewer_01" and scored.note == "表达略生硬"
    assert scored.scores is not None
    assert scored.scores.model_dump() == {
        "factual": 2,
        "relevance": 2,
        "compliance": 2,
        "expression": 1,
    }
    viewed = repos.audit.list_events(SHOP_A, action=AuditAction.QUALITY_VIEWED.value)
    assert viewed and {e.actor for e in viewed} == {"qa_reviewer_01"}  # 看原文留了访问记录


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过真浏览器冒烟测试")
def test_sign_in_state_and_request_order_in_a_real_browser(api: Api):
    """往返缓存恢复后不用旧令牌、拒绝的角色能退出、写后原地刷新、只用最近一次的响应、
    退出后不挡下一位、同步保留草稿（web_state_smoke.cjs，完整版 Chromium、打开往返缓存）。"""
    repos = api.unit.repos
    before = repos.knowledge.get_item_card(SHOP_A, "i_555")
    assert before is not None
    tokens = {
        "merchant": TOKENS["merchant_a"],
        "manager": TOKENS["manager_a"],
        "agent": TOKENS["agent_a"],
        "reviewer": TOKENS["reviewer_a"],
        "merchant_b": TOKENS["merchant_b"],
    }
    result = _run_smoke(api, STATE_SCRIPT, RODIT_SMOKE_TOKENS=json.dumps(tokens))
    assert result["steps"] == [
        "bfcache-console-signed-out",
        "bfcache-console-switched",
        "bfcache-assist-signed-out",
        "refused-role",
        "stale-current-tab",
        "latest-response",
        "busy-sign-out",
        "sync-keeps-drafts",
    ]

    # 慢请求放行后照常落库；修改面板里写了一半的回答、同步时的草稿都没有替商家保存
    card = repos.knowledge.get_item_card(SHOP_A, "i_777")
    assert card is not None and card.size_advice == "合成：版型偏大，建议拍小一码"
    after = repos.knowledge.get_item_card(SHOP_A, "i_555")
    assert after is not None and after.size_advice == before.size_advice
    assert "gift" not in repos.knowledge.get_policies(SHOP_A)
    faqs = repos.knowledge.list_knowledge(SHOP_A, status=None)
    assert not [k for k in faqs if "写了一半" in k.answer]
    assert {k.id: k.status for k in faqs}[1] == KnowledgeStatus.APPROVED
