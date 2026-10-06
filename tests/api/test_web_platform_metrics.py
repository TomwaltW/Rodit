"""商家控制台「报表」页的平台客服指标一节（§12.4 接入前后对比、§15 Phase 2 商家报表）。

- 静态检查：写进页面的指标定义和服务端一致；脚本只画本平台的指标、图表用 createElementNS、
  文字用 textContent；同步按钮给店长和商家、登记表单只给商家（data-write）；
- 真浏览器冒烟（web_platform_metrics_smoke.cjs，Chromium + Playwright）：接入日期未知时先按
  28 天前算并提示、同步平台数据后显示对比和按周的柱状图、换对比天数、填写接入日期、商家登记、
  手机宽度不横向滚动、退出后不留数据。没有 node / playwright / 浏览器时跳过。
全部是合成数据。
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

import pytest
from api_support import TOKENS, Api
from test_web_browser import NODE, _run_smoke
from test_web_page import _config

from rodit.api.web import CONSOLE_ASSET_PREFIX, CONSOLE_PAGE_PATHS, console_page_config
from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import MetricSource, Platform
from rodit.core.platform_metrics import metric_labels
from rodit.testing.fixtures import MANAGER, SHOP_A

PAGE = CONSOLE_PAGE_PATHS[0]
JS = CONSOLE_ASSET_PREFIX + "console.js"
SCRIPT = Path(__file__).with_name("web_platform_metrics_smoke.cjs")


def _section(js: str) -> str:
    start = js.index("// ===== 平台客服指标".replace("=====", "=" * 64))
    return js[start : js.index("// " + "=" * 64 + " 启动")]


def test_page_config_carries_the_metric_definitions(api: Api):
    config = _config(api.get(PAGE).text)
    assert config["platform_metrics"] == metric_labels()
    assert config["platform"] == "taobao"
    for platform in Platform:
        labels = console_page_config(platform)["platform_metrics"]
        mine = [k for k, v in labels.items() if v["platform"] == platform.value]
        assert mine, platform  # 每个平台单元的页面里都有本平台的指标


def test_script_section_builds_safe_dom_for_this_platform_only(api: Api):
    js = api.get(JS).text
    section = _section(js)
    assert "const PM_SPECS = CONFIG.platform_metrics || {};" in section
    assert "spec.platform === CONFIG.platform" in section  # 只列本平台的指标
    assert "document.createElementNS(SVG_NS, tag)" in section
    assert "document.createTextNode" in section and "innerHTML" not in section
    assert ".style" not in section  # 颜色、尺寸都在样式表里（CSP 不许内联样式）
    # 登记表单只给商家（enterApp 按 data-write 显示）；同步按钮页签可见的人都有
    assert "h('details', { id: 'pm-record', className: 'subpanel', 'data-write': true" in section
    assert "if (!isMerchant()) return;" in section
    # 内容区域退出时清空（data-dyn），同步 / 登记的按钮退出时复位（sharedCtx）
    assert (
        "h('div', { id: 'platform-metrics', className: 'pm-report', 'data-dyn': true })" in section
    )
    assert re.search(r"const pmSyncCtx = sharedCtx\('pm-sync'\);", section)
    # 接入日期：用户填的才发给接口
    assert "const typed = input.dataset.auto ? '' : input.value;" in section
    assert "initPlatformMetrics();" in js and "reports: loadReportsTab," in js


def test_no_button_in_the_section_is_named_like_the_report_refresh(api: Api):
    """报表页已有的「刷新」按钮按名字找（冒烟测试）：这一节的按钮不能也叫「刷新」。"""
    section = _section(api.get(JS).text)
    buttons = re.findall(r"h\('button', \{[^}]*\}, '([^']+)'\)", section)
    assert buttons == ["按接入日期对比", "同步平台数据", "登记"]
    assert not [b for b in buttons if "刷新" in b]


@pytest.mark.skipif(NODE is None, reason="没有 node，跳过真浏览器冒烟测试")
def test_platform_metrics_section_in_a_real_browser(api: Api):
    today = api.unit.clock.now().astimezone(CST).date()
    go_live = today - timedelta(days=28)
    record_day = today - timedelta(days=90)
    tokens = {"manager": TOKENS["manager_a"], "merchant": TOKENS["merchant_a"]}
    result = _run_smoke(
        api,
        SCRIPT,
        RODIT_SMOKE_TOKENS=json.dumps(tokens),
        RODIT_SMOKE_GO_LIVE=go_live.isoformat(),
        RODIT_SMOKE_RECORD_DAY=record_day.isoformat(),
    )
    assert result["steps"] == ["empty", "sync", "go-live", "record"]

    # 页面上的操作都落到了服务端，操作人取自令牌对应的身份
    repo = api.unit.repos.platform_metrics
    platform_rows = [r for r in repo.list(SHOP_A) if r.source == MetricSource.PLATFORM]
    assert len(platform_rows) == 56
    [merchant] = [r for r in repo.list(SHOP_A) if r.source == MetricSource.MERCHANT]
    assert (merchant.day, merchant.value, merchant.sample_size) == (record_day, 0.935, 240)
    assert merchant.recorded_by == "owner_a"
    audit = api.unit.repos.audit
    [synced] = audit.list_events(SHOP_A, action=AuditAction.PLATFORM_METRICS_SYNCED.value)
    assert synced.actor == MANAGER
    assert [
        e.actor
        for e in audit.list_events(SHOP_A, action=AuditAction.PLATFORM_METRICS_RECORDED.value)
    ] == ["owner_a"]  # 和已登记的一样的那次没有写
