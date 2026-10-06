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
    assert r.headers["cache-control"] == "no-store"  # 退出后按后退不从缓存恢复（纵深防御）
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
    assert r.headers["cache-control"] == "no-cache"  # 脚本、样式可以缓存，每次核对新旧


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


SVG_NAMESPACE = "'http://www.w3.org/2000/svg'"
"""createElementNS 要的 SVG 命名空间：是名字不是地址，浏览器不会去取它（报表页的按周柱状图）。"""


def test_console_references_no_external_urls(api: Api):
    for path in (PAGE, JS, CSS):
        text = api.get(path).text
        if path == JS:
            assert text.count(SVG_NAMESPACE) == 1  # 只有这一处，且只用作命名空间
            assert f"const SVG_NS = {SVG_NAMESPACE};" in text
            text = text.replace(SVG_NAMESPACE, "")
        assert not _EXTERNAL.search(text), path
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
    # （只清本页自己的令牌，见 test_pages_only_clear_their_own_stored_token）
    assert "const STORAGE_TOKEN = 'rodit.assist.token';" in js
    sign_out = _functions(js, "")["signOut"]
    assert "if (!opts.keepStorage) forgetToken(state.stored);" in sign_out
    forget = _functions(js, "")["forgetToken"]
    for key in ("STORAGE_TOKEN", "STORAGE_SESSION", "STORAGE_TAB"):
        assert f"storage.remove({key})" in forget
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


# ================================================================ 共用令牌：往返缓存、只清自己的

BOTH_SCRIPTS = pytest.mark.parametrize("path", ["/assist/assist.js", JS])


@BOTH_SCRIPTS
def test_pages_only_clear_their_own_stored_token(api: Api, path: str):
    """两个页面共用 sessionStorage 里的令牌：退出、401、令牌无效时，只在存着的还是本页的令牌时
    才清（同一标签页里另一个页面后来登录的人不被登出）；令牌只经 keepToken 写进去，记为本页的。"""
    js = api.get(path).text
    plain = _functions(js, "")
    forget = plain["forgetToken"]
    first = forget.split("{\n", 1)[1].strip().splitlines()[0]
    assert first == "if (!token || storage.get(STORAGE_TOKEN) !== token) return;"
    # 整个脚本里清存储只在 forgetToken 里（三个键）；写令牌只在 keepToken 里
    assert js.count("storage.remove(") == 3 == forget.count("storage.remove(")
    assert js.count("storage.set(STORAGE_TOKEN") == 1
    keep = plain["keepToken"]
    assert "storage.set(STORAGE_TOKEN, token);" in keep
    assert "state.stored = storage.get(STORAGE_TOKEN) === token ? token : null;" in keep
    # 退出：默认只清本页自己的令牌；keepStorage 时完全不动 sessionStorage
    sign_out = plain["signOut"]
    assert "function signOut(opts = {})" in js
    assert sign_out.count("forgetToken(") == 1
    assert "if (!opts.keepStorage) forgetToken(state.stored);" in sign_out
    assert "state.stored = null;" in sign_out and "signIns += 1;" in sign_out
    assert "dropRuns();" in sign_out
    # 登录失败（令牌无效、不是网页工具的角色）也只清这个令牌自己
    sign_in = _functions(js, "async ")["signIn"]
    assert sign_in.count("forgetToken(token);") == 2 and "storage.remove" not in sign_in
    assert "if (run !== signIns) return false;" in sign_in  # 期间退出或后退恢复：这次登录作废
    # 401 只登出发请求的那一位（signOut 里再核对存着的令牌）
    unauthorized = "if (resp.status === 401 && state.me && state.token === token) {"
    assert f"{unauthorized}\n        signOut();" in js


@BOTH_SCRIPTS
def test_pages_recheck_the_stored_token_when_restored_from_the_bfcache(api: Api, path: str):
    """从往返缓存恢复（pageshow persisted）：存着的令牌不是本页的（另一个页面退出或换人登录），
    先只清本页内存里的身份（不动 sessionStorage），有令牌再用它登录。"""
    js = api.get(path).text
    assert "window.addEventListener('pageshow', onPageShow);" in js
    handler = _functions(js, "")["onPageShow"].split("{\n", 1)[1]
    lines = [line.strip() for line in handler.splitlines() if line.strip()]
    assert lines[:4] == [
        "if (!event.persisted) return;",
        "const stored = storage.get(STORAGE_TOKEN);",
        "if (stored === state.stored) return;",
        "const wasSignedIn = Boolean(state.stored);",
    ]
    local = handler.index("signOut({ keepStorage: true });")
    assert local < handler.index("if (stored) signIn(stored);")
    for banned in ("signOut()", "forgetToken", "storage.remove"):
        assert banned not in handler, banned
    assert "已在另一个页面退出登录" in handler
    assert "unload" not in js  # unload 处理函数会让页面进不了往返缓存，也不可靠


@BOTH_SCRIPTS
def test_refused_roles_see_who_is_signed_in_and_can_sign_out(api: Api, path: str):
    """控制台拒绝客服、话术助手拒绝复核人员：令牌留给另一个页面（同一标签页跳过去直接登录），
    页头显示是谁、有「退出」（不然看起来没登录，下一位点页头链接就以这个身份登录了）。"""
    js = api.get(path).text
    sign_in = _functions(js, "async ")["signIn"]
    marker = "REVIEWER_ROLES.includes(me.role)" if "assist" in path else "!tabsFor(me.role).length"
    refused = sign_in.split(marker, 1)[1].split("return false;", 1)[0]
    assert "state.token = null;" in refused and "keepToken(token);" in refused
    assert "showIdentity(me);" in refused and "enterApp" not in refused
    assert "$('whoami').hidden = false;" in _functions(js, "")["showIdentity"]
    # 去售后审核工作台的链接：登录后只给客服、店长、商家显示（复核人员用不了），退出后恢复显示
    show = _functions(js, "")["showIdentity"]
    group = "!HUMAN_ROLES.includes(me.role)" if "assist" in path else "!inGroup('human', me.role)"
    assert f"$('aftersale-link').hidden = {group};" in show
    assert "$('aftersale-link').hidden = false;" in _functions(js, "")["signOut"]
    # 退出按钮同时清掉错误区（拒绝时的提示）
    click = js.split("$('sign-out').addEventListener('click', () => {", 1)[1].split("});", 1)[0]
    assert [line.strip() for line in click.strip().splitlines()] == [
        "signOut();",
        "clearError();",
        "announce('已退出');",
    ]


# ================================================================ 请求顺序、防重复提交、写后刷新

ASYNC_LOADERS = {
    "fetchItems",
    "loadKnowledge",
    "loadGaps",
    "loadPolicies",
    "loadItemsTab",
    "refreshItemTable",
    "loadItemDetail",
    "loadConfig",
    "refreshConfig",
    "loadRollout",
    "loadShadowDetail",
    "loadQualityReport",
    "loadQueue",
    "loadReports",
    "loadPlatformMetrics",
}
"""取数的异步函数：响应回来时既要核对身份（换人登录），也要核对是不是这块区域最近一次的请求。"""


def test_console_loaders_keep_only_the_latest_response(api: Api):
    """同一块区域（FAQ 列表、商品卡片、复核报表、报表……）只显示最近一次请求的结果：
    换筛选、换商品、换周期后，晚回来的旧响应不覆盖新的。"""
    js = api.get(JS).text
    functions = _functions(js, "async ")
    loaders = {n for n in functions if re.match(r"(load|refresh|fetch)[A-Z]", n)}
    assert loaders == ASYNC_LOADERS
    for name in loaders:
        body = functions[name]
        assert "state.me !== me" in body or "state.me === me" in body, name
        if name == "fetchItems":
            # 共用的商品列表缓存：比已经用上的那次新才用
            assert "const run = (itemRuns.sent += 1);" in body and "run > itemRuns.used" in body
            continue
        started = set(re.findall(r"nextRun\('(\w+)'\)", body))
        checked = set(re.findall(r"latest\('(\w+)', \w+\)", body))
        assert started and started == checked, (name, started, checked)
    # 收起商品卡片、退出时作废还没回来的加载
    assert "nextRun('item'); // 还没回来的商品卡片不再显示" in js
    assert "dropRuns();" in _functions(js, "")["signOut"]


def test_console_busy_contexts_belong_to_the_person_who_started_them(api: Api):
    """防重复提交的标记记下发起的身份和第几次：退出后不挡下一位，上一位的请求晚回来也不改下一位的按钮。"""
    js = api.get(JS).text
    plain = _functions(js, "")
    body = _functions(js, "async ")["busy"]
    assert "if (ctx.busy && ctx.owner === state.me) return undefined;" in body
    assert "ctx.owner = state.me;" in body
    assert "if (ctx.run === run) {" in body.split("finally", 1)[1]
    # 页面上固定的按钮共用的标记都登记过，退出时复位并重新启用按钮
    assert not re.search(r"\n  const \w+ = \{ busy: false \};", js)
    shared = dict(re.findall(r"\n  const (\w+Ctx) = sharedCtx\(([^)]*)\);", js))
    assert set(shared) == {
        "faqCreateCtx",
        "syncCtx",
        "sampleCtx",
        "queueCtx",
        "pmSyncCtx",
        "pmRecordCtx",
    }
    html = api.get(PAGE).text
    for ids in shared.values():
        for control in re.findall(r"'([\w-]+)'", ids):
            # 页面模板里的，或启动时由脚本建好的（报表页的平台客服指标一节）
            assert f'id="{control}"' in html or f"id: '{control}'" in js, control
    assert "resetBusy();" in plain["signOut"]
    assert "$(id).disabled = false;" in plain["resetBusy"]


def test_console_refreshes_the_tab_that_is_shown_without_closing_editors(api: Api):
    """写操作回来时用户已经切到受影响的页签：马上原地刷新（不再等下次切换）；
    知识页有打开的修改面板时只刷新「要补的知识」，设置页留着正在改的行。"""
    js = api.get(JS).text
    stale = _functions(js, "")["stale"]
    assert "if (id !== state.tab) state.loaded[id] = false;" in stale
    assert "(REFRESHERS[id] || LOADERS[id])();" in stale
    refreshers = js.split("const REFRESHERS = {", 1)[1].split("\n  };\n", 1)[0]
    assert "busyEditing('#candidates, #faqs') ? loadGaps() : loadKnowledge()" in refreshers
    assert "config: refreshConfig," in refreshers
    assert "loadRollout(busyEditing('#stage-controls'))" in refreshers
    assert "if (old && rowEdited(old)) continue;" in _functions(js, "async ")["refreshConfig"]
    # 每处 stale(...) 的页签都有加载函数
    tabs = set(re.findall(r"stale\('(\w+)'\)", js))
    assert tabs and tabs <= set(CONSOLE_TABS)


def test_console_sync_keeps_unsaved_item_edits(api: Api):
    """同步后重新显示打开的商品卡片（平台字段可能变了）：商家改过、还没保存的字段放回输入框并提示，
    不替商家保存，也不丢。"""
    js = api.get(JS).text
    sync = _functions(js, "async ")["syncItems"]
    assert "await loadItemDetail(shown, state.itemOpener, false, itemDrafts());" in sync
    drafts = _functions(js, "")["itemDrafts"]
    assert "area.value !== area.defaultValue" in drafts
    form = _functions(js, "")["itemForm"]
    assert "area.defaultValue = " in form and "area.value = drafts[f];" in form
    assert "还没保存" in form


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
    "reports": {
        ("GET", "reports/adoption"),
        ("GET", "reports/qa"),
        ("GET", "reports/platform-metrics"),
    },
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
    ("POST", "platform-metrics"),  # 登记平台后台的数据（§12.4）
}
CONSOLE_WRITES = {("POST", "platform-metrics/sync")}
"""店长、商家都能做的写操作：从平台接口同步客服考核数据（§12.4）。"""
REVIEWER_WRITES = {("POST", "quality/sample"), ("POST", "quality/reviews/{}")}
STAGE_WRITE = ("POST", "rollout/stage")
SEMI_AUTO_WRITE = ("POST", "rollout/semi-auto")  # 关：商家、复核；开：只有复核（路由内再判）


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
        {("GET", "v1/me"), STAGE_WRITE, SEMI_AUTO_WRITE}
        | set().union(*TAB_READS.values())
        | MERCHANT_WRITES
        | CONSOLE_WRITES
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
    for method, path in CONSOLE_WRITES:
        assert all(roles == set(groups["console"]) for roles in _route_roles(method, path))
    [stage_roles] = _route_roles(*STAGE_WRITE)
    assert stage_roles == set(groups["stage_shadow"]) == MERCHANT_ROLES | REVIEWER_ROLES
    assert groups["stage_assist"] == ["reviewer"]  # 切到智能辅助：路由里再判，只给复核人员
    [semi_auto_roles] = _route_roles(*SEMI_AUTO_WRITE)  # 开启半自动同样路由里再判，只给复核人员
    assert semi_auto_roles == stage_roles and set(groups["reviewer"]) == REVIEWER_ROLES
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
    assert fields["shadow_mode"]["kind"] == fields["auto_reply_enabled"]["kind"] == "bool"
    assert fields["auto_reply_schedule"]["kind"] == "select"
    assert fields["auto_reply_schedule"]["options"] == ["off_hours", "always"]
    assert fields["semi_auto_intents"]["kind"] == "intents"
    assert fields["service_hours"]["kind"] == "lines" and fields["service_hours"]["max_items"] == 6
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
    # 页头链接到话术助手和售后审核工作台（售后审核的链接登录后只给客服、店长、商家显示）
    links = [a for a in page.find("a") if a.get("class") == "page-link"]
    assert [a["href"] for a in links] == ["assist", "aftersale"]
    assert links[1]["id"] == "aftersale-link"


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
