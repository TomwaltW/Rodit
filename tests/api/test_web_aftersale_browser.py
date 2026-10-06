"""售后审核工作台的真浏览器冒烟测试（Chromium + Playwright，§7.3.2、§15 Phase 2）。

web_aftersale_smoke.cjs：登录限制（错误令牌、平台对接、复核人员去控制台）、客服的列表和概况、
拉取并跑规则、签收核查中的说明（R4）、R6 客服不能确认、买家说明 XSS、协商 → 协商有结果
（淘宝协商过才能拒绝）、状态已变化时提示并自动刷新、确认 as_2001 → 已同意、按状态筛选、
手机宽度没有横向滚动、退出、店长能确认 R6。
没有 node / playwright 模块 / 浏览器时跳过。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from api_support import TOKENS, Api
from web_browser_support import NODE, run_smoke

from rodit.core.audit import AuditAction
from rodit.core.enums import AfterSaleAction, ReviewState
from rodit.testing.fixtures import AGENT, SHOP_A

SCRIPT = Path(__file__).with_name("web_aftersale_smoke.cjs")
XSS = '<img src=x onerror="window.__xss=1">买家说明'


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过真浏览器冒烟测试")
def test_aftersale_desk_in_a_real_browser(api: Api):
    world = api.unit.tools.world
    i = next(i for i, a in enumerate(world.aftersales) if a.aftersale_id == "as_2007")
    world.aftersales[i] = world.aftersales[i].model_copy(update={"note": XSS})
    tokens = {
        "agent": TOKENS["agent_a"],
        "manager": TOKENS["manager_a"],
        "reviewer": TOKENS["reviewer_a"],
        "integration": TOKENS["integration_a"],
    }
    result = run_smoke(
        api.client.app, SCRIPT, RODIT_SMOKE_TOKENS=json.dumps(tokens), RODIT_SMOKE_XSS=XSS
    )
    assert result["steps"] == [
        "page",
        "login-guards",
        "queue",
        "run-rules",
        "verify",
        "manager-guard",
        "xss",
        "negotiate",
        "stale",
        "confirm",
        "filter",
        "phone-width",
        "manager",
    ]

    # 页面上的操作都落到了服务端：只执行了 as_2001 一次，确认人取自令牌对应的身份
    calls = api.unit.action_port.calls
    assert [(c["aftersale_id"], c["action"], c["operator"]) for c in calls] == [
        ("as_2001", AfterSaleAction.AGREE_REFUND, AGENT)
    ]
    reviews = api.unit.repos.reviews
    agreed = reviews.get(SHOP_A, "as_2001")
    assert agreed.state == ReviewState.AGREED and agreed.confirmed_by == AGENT
    # 状态已变化（拦截的 409）没有执行，自动刷新只是重读平台
    assert reviews.get(SHOP_A, "as_2002").state == ReviewState.CONFIRM
    # 店长只是打开看了，没有执行
    assert reviews.get(SHOP_A, "as_2005").state == ReviewState.CONFIRM
    # 协商留痕，备注打码
    negotiated = reviews.get(SHOP_A, "as_2007")
    assert negotiated.state == ReviewState.CONFIRM
    steps = negotiated.action_result["history"]
    assert [s["to"] for s in steps][-2:] == ["negotiate", "confirm"]
    assert "13812345678" not in json.dumps(steps, ensure_ascii=False)
    transitions = api.unit.repos.audit.list_events(SHOP_A, action=AuditAction.REVIEW_TRANSITION)
    assert any(e.detail.get("to") == "negotiate" and e.actor == AGENT for e in transitions)
