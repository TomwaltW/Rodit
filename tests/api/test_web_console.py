"""商家控制台页面本身（§9.1、§5.2、§9.3、§13.4、§13.5）：怎么提供、响应头、不引用外部地址、
CSP 友好、不用 innerHTML、令牌只进请求头、脚本调用的接口都真实存在且角色对得上、
写进页面的配置和服务端定义一致、无障碍的基本要求。
页面行为在 test_web_browser（真浏览器冒烟，web_console_smoke.cjs）里测。"""

from __future__ import annotations

import json
import re
import typing

import pytest
from api_support import CALLERS, SHOP_ROUTES, TOKENS, Api, TestClient
from test_web_page import _EXTERNAL, _config, _parse

from rodit.api import StaticTokenAuthenticator, create_app
from rodit.api.auth import CONSOLE_ROLES, MERCHANT_ROLES, PERSON_ROLES, REVIEWER_ROLES
from rodit.api.web import (
    CONSOLE_ASSET_PREFIX,
    CONSOLE_PAGE_PATHS,
    CONSOLE_TABS,
    CONTENT_SECURITY_POLICY,
    HARD_RULE_LABELS,
    config_fields,
    console_page_config,
    render_console_page,
)
from rodit.core.enums import INTENT_LABELS, POLICY_LABELS, Intent, Platform, PolicyKey
from rodit.core.models import HardRuleHit, QualityScores, ShopConfig
from rodit.feedback.review import DIMENSIONS, FULL_SCORE, LAUNCH_MIN_FULL_RATE
from rodit.knowledge.admin import (
    LOCKED_CONFIG_FIELDS,
    MAX_ANSWER_CHARS,
    MAX_QUESTION_CHARS,
    MERCHANT_CONFIG_FIELDS,
    MERCHANT_ITEM_FIELDS,
)
from rodit.testing.fixtures import SHOP_A

PAGE = CONSOLE_PAGE_PATHS[0]
JS = CONSOLE_ASSET_PREFIX + "console.js"
CSS = CONSOLE_ASSET_PREFIX + "console.css"

# ================================================================ 提供页面


def test_console_is_served_without_auth_with_strict_headers(api: Api):
    r = api.get(PAGE)
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/html; charset=utf-8"
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "set-cookie" not in r.headers
    assert r.text.startswith("<!doctype html>")


@pytest.mark.parametrize(
    ("path", "content_type"),
    [(JS, "text/javascript; charset=utf-8"), (CSS, "text/css; charset=utf-8")],
)
def test_console_assets_are_served_with_types_and_headers(api: Api, path: str, content_type: str):
    r = api.get(path)
    assert r.status_code == 200 and r.headers["content-type"] == content_type
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY


@pytest.mark.parametrize(
    "path",
    [
        "/console/assist.js",
        "/console/console.html",
        "/console/../app.py",
        "/console/%2e%2e/web.py",
        "/console/static/console.js",
        "/assist/console.js",
        "/static/console.js",
    ],
)
def test_only_the_listed_console_assets_are_served(api: Api, path: str):
    r = api.get(path)
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_console_routes_are_not_in_the_api_schema(api: Api):
    paths = api.get("/openapi.json").json()["paths"]
    assert not [p for p in paths if p == "/console" or p.startswith("/console/")]
    assert "/v1/shops/{shop_id}/console/faqs" in paths  # 控制台接口本身在 /v1 下


def test_console_has_no_shop_data_or_secrets(api: Api):
    """页面对所有人一样：不含店铺 ID、商品、FAQ、令牌；动态区域在静态页面里都是空的。"""
    html = api.get(PAGE).text
    assert SHOP_A not in html and "法式碎花连衣裙" not in html and "可以寄到新疆吗" not in html
    assert not any(token in html for token in TOKENS.values())
    for asset in (JS, CSS):
        text = api.get(asset).text
        assert SHOP_A not in text and not any(token in text for token in TOKENS.values())
    dynamic = re.findall(r"<(\w+)\b[^>]*\bdata-dyn\b[^>]*>(.*?)</\1>", html, re.S)
    assert len(dynamic) >= 15 and all(not content.strip() for _, content in dynamic)
    assert html.count("data-dyn") == len(dynamic)


# ================================================================ 不引用外部地址、CSP 友好


def test_console_references_no_external_urls(api: Api):
    for path in (PAGE, JS, CSS):
        assert not _EXTERNAL.search(api.get(path).text), path
    css = api.get(CSS).text
    assert "@import" not in css and "url(" not in css


def test_console_uses_only_same_origin_relative_assets(api: Api):
    page = _parse(api.get(PAGE).text)
    assert [a["src"] for a in page.find("script") if a.get("src")] == ["console/console.js"]
    # 基础样式沿用话术助手页的 assist.css，两个页面长得一样
    assert [a["href"] for a in page.find("link")] == [
        "data:,",
        "assist/assist.css",
        "console/console.css",
    ]
    for path in ("/assist/assist.css", CSS, JS):
        assert api.get(path).status_code == 200


def test_console_has_no_inline_code_or_styles(api: Api):
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


def test_console_forms_cannot_leak_into_urls(api: Api):
    page = _parse(api.get(PAGE).text)
    for tag in ("input", "textarea", "select"):
        for attrs in page.find(tag):
            assert "name" not in attrs, attrs
    for attrs in page.find("form"):
        assert "action" not in attrs and "method" not in attrs
    [token] = [a for a in page.find("input") if a.get("id") == "token"]
    assert token["type"] == "password" and token["autocomplete"] == "off"


# ================================================================ 脚本：防 XSS、令牌、身份守卫


def test_console_script_never_writes_html(api: Api):
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
        "window.open",
    ):
        assert banned not in js, banned
    assert "textContent" in js and "createTextNode" in js


def test_console_script_keeps_token_in_session_storage_and_headers_only(api: Api):
    js = api.get(JS).text
    assert "localStorage" not in js and "document.cookie" not in js
    assert "sessionStorage" in js
    assert not re.search(r"(?<![\w.])console\.", js)  # 不打日志（"rodit.console.tab" 是存储键）
    uses = [line for line in js.splitlines() if "token" in line and "`" in line and "${" in line]
    assert uses == ["    if (token) headers.Authorization = `Bearer ${token}`;"]
    assert "const token = state.token;" in js and "credentials: 'omit'" in js
    # 和话术助手页同一个存储键：同一标签页里两个页面互相跳转不用重新登录；退出时都清掉
    assert "const STORAGE_TOKEN = 'rodit.assist.token';" in js
    sign_out = js.split("function signOut()", 1)[1].split("\n  }\n", 1)[0]
    for key in ("STORAGE_TOKEN", "STORAGE_SESSION", "STORAGE_TAB"):
        assert f"storage.remove({key})" in sign_out
    assert "document.querySelectorAll('[data-dyn]')" in sign_out  # 退出后不留上一位的数据


def _functions(js: str, prefix: str) -> dict[str, str]:
    """顶层（IIFE 里缩进两格）的函数名 → 函数体。"""
    out = {}
    for match in re.finditer(rf"\n  {prefix}function (\w+)\(", js):
        body = js[match.end() :].split("\n  }\n", 1)[0]
        out[match.group(1)] = body
    return out


def test_console_drops_responses_that_arrive_after_a_different_sign_in(api: Api):
    """每个取数、写入的异步函数都在发请求前记下身份，响应回来时换了人就丢掉；
    401 只登出发请求的那一位。"""
    js = api.get(JS).text
    assert "state.me && state.token === token" in js
    functions = _functions(js, "async ")
    calling = {name: body for name, body in functions.items() if "api('" in body}
    assert len(calling) >= 15
    for name, body in calling.items():
        if name == "signIn":
            continue  # 登录时还没有身份
        assert "const me = state.me" in body, name
        assert "state.me !== me" in body or "state.me === me" in body, name
    # 不是 async 的操作把请求交给带身份守卫的 runEntryAction
    plain = _functions(js, "")
    for name in ("approveCandidate", "openRetire"):
        assert "runEntryAction(" in plain[name] and "api('" in plain[name], name


# ================================================================ 脚本调用的接口都存在，角色对得上

_CALL = re.compile(r"\bapi\('(GET|POST)', (?:'(v1/me)'|shopPath\((['`])(.*?)\3)")

TAB_READS: dict[str, set[tuple[str, str]]] = {
    "knowledge": {
        ("GET", "knowledge/gaps"),
        ("GET", "console/faqs"),
        ("GET", "console/items"),
    },
    "policies": {("GET", "console/policies")},
    "items": {("GET", "console/items"), ("GET", "console/items/{}")},
    "config": {("GET", "console/config")},
    "rollout": {("GET", "rollout/status"), ("GET", "rollout/shadow")},
    "review": {("GET", "quality/report"), ("GET", "quality/reviews")},
    "reports": {("GET", "reports/adoption"), ("GET", "reports/qa")},
}
"""各页签读的接口：页签对谁显示，这些接口就得允许谁读。"""

MERCHANT_WRITES = {
    ("POST", "console/faqs"),
    ("POST", "console/faqs/{}/edit"),
    ("POST", "console/candidates/{}/approve-edited"),
    ("POST", "knowledge/candidates/{}/approve"),
    ("POST", "knowledge/{}/retire"),
    ("POST", "console/policies/{}"),
    ("POST", "console/policies/{}/clear"),
    ("POST", "console/items/{}"),
    ("POST", "console/items/sync"),
    ("POST", "console/config"),
}
REVIEWER_WRITES = {("POST", "quality/sample"), ("POST", "quality/reviews/{}")}
STAGE_WRITE = ("POST", "rollout/stage")


def _script_calls(js: str) -> list[tuple[str, str]]:
    calls = []
    for method, me, _, path in _CALL.findall(js):
        if me:
            calls.append((method, me))
            continue
        assert "?" not in path, path  # 查询参数一律放 shopPath 的第二个参数
        calls.append((method, re.sub(r"\$\{[^}]*\}+\)*", "{}", path)))
    return calls


def _normalize(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path)


def test_console_script_calls_only_existing_endpoints(api: Api):
    """每一处 api(...) 调用都能解析出来；路径和方法都在应用的 OpenAPI 里。"""
    js = api.get(JS).text
    calls = _script_calls(js)
    assert len(calls) == len(re.findall(r"\bapi\('", js))  # 没有拼出来的、解析不了的地址
    spec = api.get("/openapi.json").json()["paths"]
    known = {(m.upper(), _normalize(p)) for p, ops in spec.items() for m in ops}
    for method, path in set(calls):
        full = "/v1/me" if path == "v1/me" else f"/v1/shops/{{}}/{path}"
        assert (method, full) in known, (method, path)
    used = set(calls)
    expected = (
        {("GET", "v1/me"), STAGE_WRITE}
        | set().union(*TAB_READS.values())
        | MERCHANT_WRITES
        | REVIEWER_WRITES
    )
    assert used == expected


def _route_roles(method: str, path: str) -> list[frozenset[str]]:
    pattern = re.compile("^/v1/shops/\\{shop\\}/" + re.escape(path).replace(r"\{\}", "[^/]+") + "$")
    roles = [r.roles for r in SHOP_ROUTES if r.method == method and pattern.match(r.path)]
    assert roles, (method, path)
    return roles


def test_console_tabs_and_write_buttons_match_the_endpoint_roles():
    """页签按角色显示、写按钮只给商家：和接口允许的角色一致（真正的检查在接口上，§9.2）。"""
    config = console_page_config(Platform.TAOBAO)
    assert set(config["tabs"]) == set(TAB_READS)
    for tab, reads in TAB_READS.items():
        shown = set(config["tabs"][tab])
        for method, path in reads:
            for roles in _route_roles(method, path):
                assert shown <= roles, (tab, method, path)
    assert CONSOLE_TABS["rollout"] == CONSOLE_ROLES | REVIEWER_ROLES
    assert CONSOLE_TABS["review"] == REVIEWER_ROLES
    groups = config["role_groups"]
    for method, path in MERCHANT_WRITES:
        assert all(roles == set(groups["merchant"]) for roles in _route_roles(method, path))
    for method, path in REVIEWER_WRITES:
        assert all(roles == set(groups["reviewer"]) for roles in _route_roles(method, path))
    [stage_roles] = _route_roles(*STAGE_WRITE)
    assert stage_roles == set(groups["stage_shadow"]) == MERCHANT_ROLES | REVIEWER_ROLES
    assert groups["stage_assist"] == ["reviewer"]  # 切到智能辅助：路由里再判，只给复核人员
    assert set(groups["person"]) == PERSON_ROLES
    # 客服没有任何页签（控制台提示去话术助手）；每个能登录控制台的人至少有一个页签
    assert not [t for t, roles in config["tabs"].items() if "agent" in roles]
    for role in ("manager", "merchant", "reviewer"):
        assert [t for t, roles in config["tabs"].items() if role in roles], role
    assert config["default_tab"] == {"reviewer": "review"}


# ================================================================ 写进页面的配置


def test_console_config_matches_server_definitions(api: Api):
    config = _config(api.get(PAGE).text)
    expected = console_page_config(Platform.TAOBAO)
    assert config == json.loads(json.dumps(expected, ensure_ascii=False))
    assert config["platform_name"] == "淘宝"
    assert config["roles"]["reviewer"] == "复核人员"
    assert config["intents"] == {i.value: INTENT_LABELS[i] for i in Intent}
    assert config["policies"] == {k.value: POLICY_LABELS[k] for k in PolicyKey}
    assert list(config["item_fields"]) == list(MERCHANT_ITEM_FIELDS)
    assert config["dimensions"] == DIMENSIONS
    assert set(DIMENSIONS) == set(QualityScores.model_fields)
    assert config["full_score"] == FULL_SCORE
    assert all(len(config["score_guide"][d]) == FULL_SCORE + 1 for d in DIMENSIONS)
    assert config["stages"] == {
        "shadow": "影子模式",
        "assist": "智能辅助",
        "auto": "全自动（已开启）",
    }
    assert config["limits"]["question"] == MAX_QUESTION_CHARS
    assert config["limits"]["answer"] == MAX_ANSWER_CHARS
    assert config["limits"]["launch_min_full_rate"] == LAUNCH_MIN_FULL_RATE
    assert re.fullmatch(config["period_pattern"], "2026-W40")


def test_config_fields_follow_the_shop_config_model():
    """每个店铺配置项都有中文名和合适的控件；能改的就是商家白名单，不能改的写明原因（§9.3、§13.4）。"""
    fields = {f["key"]: f for f in config_fields()}
    assert list(fields) == list(ShopConfig.model_fields)
    assert {k for k, f in fields.items() if f["editable"]} == set(MERCHANT_CONFIG_FIELDS)
    for key in LOCKED_CONFIG_FIELDS:
        assert not fields[key]["editable"] and fields[key]["locked_reason"]
        assert fields[key]["kind"] == "bool"
    assert fields["tone"]["kind"] == "select"
    assert fields["tone"]["options"] == ["亲切", "专业", "简洁"]
    assert fields["salutation"]["options"] == ["亲", "您"]
    assert (fields["max_answer_chars"]["min"], fields["max_answer_chars"]["max"]) == (20, 500)
    assert (fields["suggestion_count"]["min"], fields["suggestion_count"]["max"]) == (1, 3)
    assert fields["disabled_intents"]["kind"] == "intents"
    assert fields["banned_words"]["kind"] == fields["manager_ids"]["kind"] == "lines"
    assert fields["aftersale_amount_limit"]["kind"] == "money"
    assert fields["display_name"]["kind"] == "text"


def test_hard_rule_labels_cover_every_rule():
    rules = set(typing.get_args(HardRuleHit.model_fields["rule"].annotation))
    assert rules | {"policy_conflict", "duplicate_question"} == set(HARD_RULE_LABELS)


@pytest.mark.parametrize("platform", list(Platform))
def test_each_platform_unit_renders_its_own_console(platform: Platform):
    assert _config(render_console_page(platform).decode("utf-8"))["platform"] == platform.value


# ================================================================ 结构与无障碍


def test_console_tabs_follow_the_aria_tabs_pattern(api: Api):
    page = _parse(api.get(PAGE).text)
    [tablist] = [a for t, a in page.tags if a.get("role") == "tablist"]
    assert tablist.get("aria-label")
    tabs = [a for t, a in page.tags if a.get("role") == "tab"]
    panels = {a["id"]: a for t, a in page.tags if a.get("role") == "tabpanel"}
    assert [t["data-tab"] for t in tabs] == list(CONSOLE_TABS)
    for tab in tabs:
        panel = panels[tab["aria-controls"]]
        assert panel["aria-labelledby"] == tab["id"] == f"tab-{tab['data-tab']}"
        assert "hidden" in tab and "hidden" in panel  # 登录前什么都不显示
        assert tab["aria-selected"] == "false"
    assert len(panels) == len(tabs)


def test_console_controls_are_labelled(api: Api):
    page = _parse(api.get(PAGE).text)
    labelled = {a.get("for") for a in page.find("label")}
    for tag in ("input", "textarea", "select"):
        for attrs in page.find(tag):
            assert attrs.get("id") in labelled or attrs.get("aria-label"), attrs
    [html] = page.find("html")
    assert html["lang"] == "zh-CN"
    assert any(a.get("role") == "alert" for _, a in page.tags)
    assert any(a.get("aria-live") == "polite" for _, a in page.tags)
    # 两个页面互相有链接
    assert {"href": "assist", "class": "page-link"}.items() <= next(
        a for a in page.find("a") if a.get("class") == "page-link"
    ).items()


def test_review_panel_warns_that_viewing_is_logged(api: Api):
    html = api.get(PAGE).text
    panel = html.split('id="panel-review"', 1)[1].split("</section>", 1)[0]
    assert "访问日志" in panel and "§9.2" in panel and "已打码" in panel
    assert "记录访问" in panel  # 加载列表的按钮上也写明


def test_console_is_turned_off_with_the_web_page(api: Api):
    """只做模式 A / B 的单元关掉网页工具时，控制台页面也不挂（接口照常）。"""
    app = create_app(
        api.service,
        authenticator=StaticTokenAuthenticator({TOKENS["merchant_a"]: CALLERS["merchant_a"]}),
        web_page=False,
    )
    client = TestClient(app)
    for path in (PAGE, JS, CSS):
        assert client.get(path).status_code == 404
    r = client.get(
        f"/v1/shops/{SHOP_A}/console/config",
        headers={"Authorization": f"Bearer {TOKENS['merchant_a']}"},
    )
    assert r.status_code == 200
