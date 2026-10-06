"""售后审核工作台页面本身（§7.3.2、§15 Phase 2）：怎么提供、响应头、不引用外部地址、CSP 友好、
不用 innerHTML、令牌只进请求头、脚本调用的接口都真实存在且角色对得上、写进页面的配置和服务端
定义一致、无障碍的基本要求。页面行为在 test_web_aftersale_browser（真浏览器冒烟）里测。"""

from __future__ import annotations

import json
import re

import pytest
from api_support import CALLERS, SHOP_ROUTES, TOKENS, Api, TestClient
from test_web_page import _EXTERNAL, _config, _parse

from rodit.aftersale import AFTERSALE_STATUS_LABELS, TERMINAL_STATES
from rodit.api import StaticTokenAuthenticator, create_app
from rodit.api.auth import HUMAN_ROLES, PERSON_ROLES
from rodit.api.schemas import MAX_ID_CHARS, MAX_NOTE_CHARS
from rodit.api.web import (
    AFTERSALE_ASSET_PREFIX,
    AFTERSALE_ASSETS,
    AFTERSALE_CONFIRM_RULE,
    AFTERSALE_PAGE_PATHS,
    CONTENT_SECURITY_POLICY,
    aftersale_page_config,
    render_aftersale_page,
)
from rodit.app.aftersale_desk import (
    PLATFORM_NOTES,
    REVIEW_STATE_LABELS,
    RULES,
    STATE_FLOW,
)
from rodit.core.enums import AfterSaleAction, AfterSaleStatus, Platform, ReviewState
from rodit.testing.fixtures import SHOP_A

PAGE = AFTERSALE_PAGE_PATHS[0]
JS = AFTERSALE_ASSET_PREFIX + "aftersale.js"
CSS = AFTERSALE_ASSET_PREFIX + "aftersale.css"

# ================================================================ 提供页面


def test_page_is_served_without_auth_with_strict_headers(api: Api):
    r = api.get(PAGE)
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/html; charset=utf-8"
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"  # 退出后按后退不从缓存恢复（纵深防御）
    assert "set-cookie" not in r.headers
    assert r.text.startswith("<!doctype html>")


@pytest.mark.parametrize(
    ("path", "content_type"),
    [(JS, "text/javascript; charset=utf-8"), (CSS, "text/css; charset=utf-8")],
)
def test_assets_are_served_with_types_and_headers(api: Api, path: str, content_type: str):
    r = api.get(path)
    assert r.status_code == 200 and r.headers["content-type"] == content_type
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert r.headers["cache-control"] == "no-cache"


@pytest.mark.parametrize(
    "path",
    [
        "/aftersale/aftersale.html",
        "/aftersale/console.js",
        "/aftersale/assist.css",
        "/aftersale/../app.py",
        "/aftersale/%2e%2e/web.py",
        "/aftersale/static/aftersale.js",
        "/assist/aftersale.js",
    ],
)
def test_only_the_listed_assets_are_served(api: Api, path: str):
    r = api.get(path)
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_page_routes_are_not_in_the_api_schema(api: Api):
    paths = api.get("/openapi.json").json()["paths"]
    assert not [p for p in paths if p.startswith("/aftersale")]
    assert "/v1/shops/{shop_id}/aftersales" in paths


def test_page_can_be_turned_off(api: Api):
    app = create_app(
        api.service,
        authenticator=StaticTokenAuthenticator({TOKENS["agent_a"]: CALLERS["agent_a"]}),
        web_page=False,
    )
    client = TestClient(app)
    for path in (PAGE, JS, CSS):
        assert client.get(path).status_code == 404


def test_page_has_no_shop_data_or_secrets(api: Api):
    for path in (PAGE, JS, CSS):
        text = api.get(path).text
        assert SHOP_A not in text and "as_2001" not in text
        assert not any(token in text for token in TOKENS.values())


# ================================================================ 不引用外部地址、CSP 友好


def test_page_and_assets_reference_no_external_urls(api: Api):
    for path in (PAGE, JS, CSS):
        assert not _EXTERNAL.search(api.get(path).text), path
    css = api.get(CSS).text
    assert "@import" not in css and "url(" not in css


def test_page_uses_only_same_origin_relative_assets(api: Api):
    page = _parse(api.get(PAGE).text)
    assert [a["src"] for a in page.find("script") if a.get("src")] == ["aftersale/aftersale.js"]
    hrefs = {a["href"] for a in page.find("link")}
    assert hrefs == {"assist/assist.css", "aftersale/aftersale.css", "data:,"}  # 基础样式共用
    for asset in AFTERSALE_ASSETS:
        assert api.get(AFTERSALE_ASSET_PREFIX + asset).status_code == 200


def test_page_has_no_inline_code_or_styles(api: Api):
    html = api.get(PAGE).text
    page = _parse(html)
    for attrs, text in page.scripts:
        if attrs.get("src"):
            assert not text.strip()
        else:
            assert attrs.get("type") == "application/json"
    for tag, attrs in page.tags:
        assert tag != "style"
        assert "style" not in attrs, tag
        assert not [a for a in attrs if a.startswith("on")], (tag, attrs)
    assert "javascript:" not in html.lower()


def test_forms_cannot_leak_token_or_text_into_urls(api: Api):
    page = _parse(api.get(PAGE).text)
    for tag in ("input", "textarea", "select"):
        for attrs in page.find(tag):
            assert "name" not in attrs, attrs
    for attrs in page.find("form"):
        assert "action" not in attrs and "method" not in attrs
    [token] = [a for a in page.find("input") if a.get("id") == "token"]
    assert token["type"] == "password" and token["autocomplete"] == "off"


# ================================================================ 脚本：防 XSS、令牌只进请求头


def test_script_never_writes_html(api: Api):
    js = api.get(JS).text
    for banned in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        "createContextualFragment",
        "DOMParser",
        "srcdoc",
    ):
        assert banned not in js, banned
    assert "textContent" in js and "createTextNode" in js
    assert ".style" not in js  # 样式都在样式表里


def test_script_keeps_token_in_session_storage_and_headers_only(api: Api):
    js = api.get(JS).text
    assert "localStorage" not in js and "document.cookie" not in js
    assert "sessionStorage" in js
    assert not re.search(r"(?<![\w.])console\.", js)
    assert "STORAGE_TOKEN = 'rodit.assist.token'" in js  # 和另外两个页面同一个键
    uses = [line for line in js.splitlines() if "token" in line and "`" in line and "${" in line]
    assert uses == ["    if (token) headers.Authorization = `Bearer ${token}`;"]
    assert "const token = state.token;" in js
    assert "credentials: 'omit'" in js


_CALL = re.compile(r"\bapi\(\s*'(GET|POST)',\s*(?:'(v1/me)'|shopPath\((['`])(.*?)\3)")

DESK_CALLS = {
    ("GET", "v1/me"),
    ("GET", "aftersales"),
    ("GET", "aftersales/stats"),
    ("POST", "aftersales/review-pending"),
    ("GET", "aftersales/{}"),
    ("POST", "aftersales/{}/review"),
    ("POST", "aftersales/{}/negotiate"),
    ("POST", "aftersales/{}/negotiation-done"),
    ("POST", "aftersales/{}/confirm"),
}


def _script_calls(js: str) -> list[tuple[str, str]]:
    calls = []
    for method, me, _, path in _CALL.findall(js):
        if me:
            calls.append((method, me))
            continue
        assert "?" not in path, path  # 查询参数一律放 shopPath 的第二个参数
        calls.append((method, re.sub(r"\$\{[^}]*\}+\)*", "{}", path)))
    return calls


def test_script_calls_only_existing_endpoints_open_to_the_desk_roles(api: Api):
    js = api.get(JS).text
    calls = _script_calls(js)
    assert len(calls) == len(re.findall(r"\bapi\(", js)) - 1  # 减去 api 函数自己的定义
    assert set(calls) == DESK_CALLS
    spec = api.get("/openapi.json").json()["paths"]
    known = {(m.upper(), re.sub(r"\{[^}]+\}", "{}", p)) for p, ops in spec.items() for m in ops}
    desk = set(aftersale_page_config(Platform.TAOBAO)["role_groups"]["desk"])
    for method, path in DESK_CALLS:
        full = "/v1/me" if path == "v1/me" else f"/v1/shops/{{}}/{path}"
        assert (method, full) in known, (method, path)
        if path == "v1/me":
            continue
        pattern = re.compile(
            "^/v1/shops/\\{shop\\}/" + re.escape(path).replace(r"\{\}", "[^/]+") + "$"
        )
        roles = [r.roles for r in SHOP_ROUTES if r.method == method and pattern.match(r.path)]
        assert roles, (method, path)
        assert all(desk <= set(r) for r in roles), (method, path)  # 页面给的角色接口都放行


def test_script_confirms_only_from_the_dialog_with_the_status_version(api: Api):
    """写操作只在确认对话框里点「确认执行」后发（§4.1 原则 3），带详情里的平台状态版本；
    状态已变化时提示并自动刷新。"""
    js = api.get(JS).text
    assert js.count("/confirm`") == 1
    body = js.split("async function doConfirm(", 1)[1].split("\n  }\n", 1)[0]
    assert "expected_status_version: pending.version" in body
    assert "/confirm`" in body
    opener = js.split("function openConfirm(", 1)[1].split("\n  }\n", 1)[0]
    assert "version: d.platform.status_version" in opener and "showModal" in opener
    assert "$('confirm-go').addEventListener('click', doConfirm)" in js
    assert "售后单状态已变化，请刷新" in js and "STATUS_CHANGED" in js
    stale = js.split("async function staleRefresh(", 1)[1].split("\n  }\n", 1)[0]
    assert "/review`" in stale


def test_script_drops_responses_that_arrive_after_a_different_sign_in(api: Api):
    js = api.get(JS).text
    assert "state.me && state.token === token" in js  # 401 只登出发请求的那一位
    for fn, key in (("loadStats", "stats"), ("loadQueue", "queue"), ("loadDetail", "detail")):
        body = js.split(f"async function {fn}(", 1)[1].split("\n  }\n", 1)[0]
        assert "const me = state.me" in body and "state.me !== me" in body, fn
        assert f"const run = nextRun('{key}');" in body and f"latest('{key}', run)" in body, fn
    sign_out = js.split("function signOut(opts = {})", 1)[1].split("\n  }\n", 1)[0]
    assert "resetBusy();" in sign_out and "closeConfirm();" in sign_out
    assert "forgetToken(state.stored)" in sign_out
    assert "window.addEventListener('pageshow', onPageShow)" in js


def test_reviewers_are_sent_to_the_console(api: Api):
    """复核人员不做售后审核：只看到去控制台的提示（令牌留着，同一标签页打开控制台直接登录）。"""
    js = api.get(JS).text
    body = js.split("async function signIn(", 1)[1].split("\n  }\n", 1)[0]
    branch = body.split("if (!inGroup('desk', me.role))", 1)[1].split("return false;", 1)[0]
    assert "state.token = null;" in branch and "keepToken(token);" in branch
    assert "showIdentity(me);" in branch and "href: 'console'" in branch
    assert "enterApp" not in branch
    assert body.index("inGroup('desk'") < body.index("enterApp()")


# ================================================================ 写进页面的配置


def test_config_matches_server_definitions(api: Api):
    config = _config(api.get(PAGE).text)
    assert config == json.loads(
        json.dumps(aftersale_page_config(Platform.TAOBAO), ensure_ascii=False)
    )
    assert config["platform_name"] == "淘宝"
    assert config["review_states"] == {s.value: REVIEW_STATE_LABELS[s] for s in ReviewState}
    assert config["review_states"]["verify"] == "签收核查中"
    assert config["state_flow"] == [s.value for s in STATE_FLOW]
    assert config["terminal_states"] == sorted(s.value for s in TERMINAL_STATES)
    assert config["statuses"] == {s.value: AFTERSALE_STATUS_LABELS[s] for s in AfterSaleStatus}
    assert list(config["rules"]) == [f"R{i}" for i in range(1, 9)]
    assert config["rules"]["R4"] == RULES["R4"].model_dump()
    assert config["rules"]["R6"] == {
        "condition": "金额超过店铺设定的上限",
        "advice": "必须店长确认",
    }
    assert config["actions"] == {
        "agree_refund": "同意退款",
        "agree_return": "同意退货",
        "refuse": "拒绝售后",
    }
    assert config["role_groups"]["desk"] == sorted(HUMAN_ROLES)
    assert config["role_groups"]["person"] == sorted(PERSON_ROLES)
    assert config["unreviewed"] == "unreviewed" and "unreviewed" not in config["review_states"]
    assert (
        config["confirm_rule"] == AFTERSALE_CONFIRM_RULE
        and "必须由人确认" in AFTERSALE_CONFIRM_RULE
    )
    assert config["limits"] == {
        "note": MAX_NOTE_CHARS,
        "reason_code": MAX_ID_CHARS,
        "batch": 20,
        "batch_max": 50,
        "queue": 50,
    }


@pytest.mark.parametrize("platform", list(Platform))
def test_each_platform_unit_renders_its_own_rules(platform: Platform):
    config = _config(render_aftersale_page(platform).decode("utf-8"))
    assert config["platform"] == platform.value
    assert config["platform_note"] == PLATFORM_NOTES[platform]
    refuse = AfterSaleAction.REFUSE.value
    assert (refuse in config["platform_actions"]) == (platform != Platform.PDD)
    assert config["reason_code_required"] == (platform == Platform.DOUYIN)
    if platform == Platform.TAOBAO:
        assert "协商" in config["refuse_hint"]


# ================================================================ 页面结构、无障碍


def test_header_links_to_the_other_two_pages(api: Api):
    page = _parse(api.get(PAGE).text)
    links = [a["href"] for a in page.find("a") if a.get("class") == "page-link"]
    assert links == ["assist", "console"]


def test_every_form_control_has_a_label(api: Api):
    page = _parse(api.get(PAGE).text)
    labelled = {a.get("for") for a in page.find("label")}
    for tag in ("input", "textarea", "select"):
        for attrs in page.find(tag):
            assert attrs.get("id") in labelled or attrs.get("aria-label"), attrs
    [html] = page.find("html")
    assert html["lang"] == "zh-CN"
    [viewport] = [a for a in page.find("meta") if a.get("name") == "viewport"]
    assert "width=device-width" in viewport["content"]
    assert any(a.get("role") == "alert" for _, a in page.tags)
    assert any(a.get("aria-live") == "polite" for _, a in page.tags)


def test_confirmation_dialog_is_labelled(api: Api):
    page = _parse(api.get(PAGE).text)
    [dialog] = page.find("dialog")
    assert dialog["aria-labelledby"] == "confirm-title"
    assert dialog["aria-describedby"] == "confirm-rule"
    ids = {a.get("id") for _, a in page.tags}
    assert {"confirm-go", "confirm-cancel", "reason-code", "reason-text"} <= ids
    assert {"run-rules", "queue", "detail", "stats", "filter-state"} <= ids
