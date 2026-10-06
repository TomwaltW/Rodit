"""半自动与全自动生效时段的真浏览器冒烟测试（Chromium + Playwright，web_semiauto_smoke.cjs）：
话术助手页的"半自动：可直接发送"标记、控制台「上线」页的生效时段和半自动开关（§5.6、§13.4、
§15 Phase 3）。服务器、跳过规则同 test_web_browser.py（没有 node / playwright / 浏览器时跳过）。
数据全部合成。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from api_support import TOKENS, Api
from test_web_browser import NODE, _run_smoke

from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import Intent, SuggestionStatus
from rodit.core.models import SuggestionRecord
from rodit.testing.fixtures import AGENT, SHOP_A

SCRIPT = Path(__file__).with_name("web_semiauto_smoke.cjs")
W38_START = datetime(2026, 9, 14, tzinfo=CST)  # 演示时钟在 2026-09-28 10:35（周一，W40）
W39_START = datetime(2026, 9, 21, tzinfo=CST)


def _weeks(api: Api, intent: Intent, as_is: int) -> None:
    """最近两个完整周各 20 个推荐组，前 as_is 个原样发送。"""
    repo = api.unit.repos.suggestions
    for start in (W38_START, W39_START):
        for i in range(20):
            repo.add(
                SuggestionRecord(
                    shop_id=SHOP_A,
                    conversation_id=f"conv_{intent.value}_{i % 4}",
                    msg_id=f"sm_{intent.value}_{start:%d}_{i}",
                    question_text="合成问题",
                    intent=intent,
                    text="合成回答",
                    status=SuggestionStatus.SENT if i < as_is else SuggestionStatus.EDITED_SENT,
                    created_at=start + timedelta(hours=3, minutes=i),
                )
            )


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过真浏览器冒烟测试")
def test_semi_auto_and_schedule_in_a_real_browser(api: Api):
    repos = api.unit.repos
    repos.knowledge.update_shop_config(
        SHOP_A,
        {
            "semi_auto_intents": ["shipping_policy"],
            "auto_reply_enabled": True,  # 全自动已开（平台准入后由我们开启），现在是工作时间：暂停
            "service_hours": ["09:00-23:00"],
        },
    )
    _weeks(api, Intent.INVOICE_GIFT, 19)  # 95%：门槛达到
    _weeks(api, Intent.SIZE_ADVICE, 15)  # 75%：门槛没达到
    tokens = {
        "agent": TOKENS["agent_a"],
        "manager": TOKENS["manager_a"],
        "merchant": TOKENS["merchant_a"],
        "reviewer": TOKENS["reviewer_a"],
    }
    result = _run_smoke(api, SCRIPT, RODIT_SMOKE_TOKENS=json.dumps(tokens))
    assert result["steps"] == ["assist-badge", "manager", "service-hours", "merchant", "reviewer"]

    # 页面上的操作都落到了服务端，操作人取自令牌对应的身份
    shop = repos.knowledge.get_shop(SHOP_A)
    assert shop is not None
    assert shop.config_overrides["semi_auto_intents"] == ["size_advice", "invoice_gift"]
    assert shop.config_overrides["service_hours"] == ["09:00-12:00", "13:30-22:00"]
    events = repos.audit.list_events(SHOP_A, action=AuditAction.ROLLOUT_SEMI_AUTO.value)
    assert [
        (e.actor, e.detail["intent"], e.detail["enabled"], e.detail["forced"]) for e in events
    ] == [
        ("owner_a", "shipping_policy", False, False),
        ("qa_reviewer_01", "invoice_gift", True, False),
        ("qa_reviewer_01", "size_advice", True, True),
    ]
    assert events[-1].detail["note"] == "种子商家同意先试"
    # 话术助手上带半自动标记的那条：客服点了发送，照常记"原样发送"
    sent = [
        r
        for r in repos.suggestions.list(SHOP_A, status=SuggestionStatus.SENT)
        if r.agent_id == AGENT and r.intent == Intent.SHIPPING_POLICY
    ]
    assert len(sent) == 1
