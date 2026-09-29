"""模式 C 网页工具的页面本身（§8.1）：怎么提供、响应头、不引用外部地址、CSP 友好、不用 innerHTML、
无障碍的基本要求。页面行为在 test_web_browser（真浏览器冒烟）里测。"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

import pytest
from api_support import CALLERS, TOKENS, Api, TestClient

from rodit.api import StaticTokenAuthenticator, create_app
from rodit.api.schemas import MAX_TEXT_CHARS
from rodit.api.web import (
    ASSET_PREFIX,
    ASSETS,
    CONTENT_SECURITY_POLICY,
    PAGE_PATHS,
    json_for_html,
    page_config,
    render_page,
)
from rodit.core.enums import INTENT_LABELS, Intent, Platform, RejectReason
from rodit.feedback import REJECT_REASON_LABELS
from rodit.knowledge import QUESTIONNAIRE
from rodit.testing.fixtures import SHOP_A

JS = ASSET_PREFIX + "assist.js"
CSS = ASSET_PREFIX + "assist.css"


class _Page(HTMLParser):
    """收集页面里的标签、属性和 <script> 内容。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.scripts: list[tuple[dict[str, str | None], str]] = []
        self._in_script: dict[str, str | None] | None = None
        self._script_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.tags.append((tag, attributes))
        if tag == "script":
            self._in_script = attributes
            self._script_text = []

    def handle_endtag(self, tag):
        if tag == "script" and self._in_script is not None:
            self.scripts.append((self._in_script, "".join(self._script_text)))
            self._in_script = None

    def handle_data(self, data):
        if self._in_script is not None:
            self._script_text.append(data)

    def find(self, tag: str) -> list[dict[str, str | None]]:
        return [a for t, a in self.tags if t == tag]


def _parse(html: str) -> _Page:
    page = _Page()
    page.feed(html)
    return page


def _config(html: str) -> dict:
    page = _parse(html)
    [(attrs, text)] = [(a, t) for a, t in page.scripts if a.get("id") == "rodit-config"]
    assert attrs.get("type") == "application/json"
    return json.loads(text)


# ================================================================ 提供页面


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_page_is_served_without_auth_with_strict_headers(api: Api, path: str):
    r = api.get(path)
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/html; charset=utf-8"
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    # 页面不进缓存：退出后按后退不从缓存恢复上一位的界面（纵深防御；Chrome 的往返缓存照样可能
    # 恢复 no-store 的页面，页面脚本另外在 pageshow 时核对令牌）
    assert r.headers["cache-control"] == "no-store"
    assert "set-cookie" not in r.headers
    assert r.text.startswith("<!doctype html>")


def test_csp_forbids_inline_code_framing_and_form_posts():
    directives = dict(d.split(" ", 1) for d in CONTENT_SECURITY_POLICY.split("; "))
    assert directives["default-src"] == "'none'"
    assert directives["script-src"] == "'self'" and directives["style-src"] == "'self'"
    assert directives["connect-src"] == "'self'"
    assert directives["frame-ancestors"] == "'none'" and directives["form-action"] == "'none'"
    assert "unsafe" not in CONTENT_SECURITY_POLICY


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
        "/assist/nope.js",
        "/assist/assist.html",
        "/assist/../app.py",
        "/assist/%2e%2e/app.py",
        "/assist/static/assist.js",
        "/static/assist.js",
    ],
)
def test_only_the_listed_assets_are_served(api: Api, path: str):
    r = api.get(path)
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    assert "import" not in r.text


def test_page_routes_are_not_in_the_api_schema(api: Api):
    paths = api.get("/openapi.json").json()["paths"]
    assert not [p for p in paths if p == "/" or p.startswith("/assist")]
    assert "/v1/me" in paths and "/v1/shops/{shop_id}/items" in paths


def test_web_page_can_be_turned_off(api: Api):
    app = create_app(
        api.service,
        authenticator=StaticTokenAuthenticator({TOKENS["agent_a"]: CALLERS["agent_a"]}),
        web_page=False,
    )
    client = TestClient(app)
    for path in (*PAGE_PATHS, JS, CSS):
        assert client.get(path).status_code == 404
    assert client.get("/v1/me", headers={"Authorization": f"Bearer {TOKENS['agent_a']}"}).json()


def test_page_has_no_shop_data_or_secrets(api: Api):
    """页面对所有人一样：不含店铺 ID、商品、令牌。数据都要带令牌从接口取。"""
    html = api.get("/assist").text
    assert SHOP_A not in html and "法式碎花连衣裙" not in html
    assert not any(token in html for token in TOKENS.values())
    for asset in (JS, CSS):
        text = api.get(asset).text
        assert SHOP_A not in text and not any(token in text for token in TOKENS.values())


# ================================================================ 不引用外部地址、CSP 友好


_EXTERNAL = re.compile(r"(?i)\b(?:https?|ftp|wss?):|(?:src|href)\s*=\s*[\"']?//")


def test_page_and_assets_reference_no_external_urls(api: Api):
    for path in ("/assist", JS, CSS):
        text = api.get(path).text
        assert not _EXTERNAL.search(text), path
    css = api.get(CSS).text
    assert "@import" not in css and "url(" not in css  # 字体、图片都不从别处加载


def test_page_uses_only_same_origin_relative_assets(api: Api):
    page = _parse(api.get("/assist").text)
    srcs = [a["src"] for a in page.find("script") if a.get("src")]
    hrefs = [a["href"] for a in page.find("link")]
    assert srcs == ["assist/assist.js"]
    assert set(hrefs) == {"assist/assist.css", "data:,"}
    # 相对地址在 / 和 /assist 下都指向 /assist/...（资源确实在那儿）
    for asset in ASSETS:
        assert api.get(ASSET_PREFIX + asset).status_code == 200


def test_page_has_no_inline_code_or_styles(api: Api):
    """CSP 不放 'unsafe-inline'：没有内联脚本、事件属性、style 属性、<style>。"""
    html = api.get("/assist").text
    page = _parse(html)
    for attrs, text in page.scripts:
        if attrs.get("src"):
            assert not text.strip()
        else:
            assert attrs.get("type") == "application/json"  # 数据块，浏览器不执行
    for tag, attrs in page.tags:
        assert tag != "style"
        assert "style" not in attrs, tag
        assert not [a for a in attrs if a.startswith("on")], (tag, attrs)
    assert "javascript:" not in html.lower()


def test_forms_cannot_leak_token_or_question_into_urls(api: Api):
    """输入框都没有 name（表单即使被提交也不带值），表单没有 action；CSP 另外禁止了表单提交。"""
    page = _parse(api.get("/assist").text)
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


def test_script_keeps_token_in_session_storage_and_headers_only(api: Api):
    js = api.get(JS).text
    assert "localStorage" not in js and "document.cookie" not in js
    assert "sessionStorage" in js
    assert not re.search(r"(?<![\w.])console\.", js)  # 不打日志（"rodit.console.tab" 是存储键）
    assert "Authorization" in js and "Bearer ${token}" in js
    # 令牌不拼进地址：只在 Authorization 头里出现（api() 发请求时取一次 state.token）
    uses = [line for line in js.splitlines() if "token" in line and "`" in line and "${" in line]
    assert uses == ["    if (token) headers.Authorization = `Bearer ${token}`;"]
    assert "const token = state.token;" in js
    assert "credentials: 'omit'" in js


def test_script_calls_only_the_documented_endpoints(api: Api):
    js = api.get(JS).text
    # 第二个参数：字符串字面量，或 shopPath(...) / feedbackPath(...)（允许一层嵌套括号）
    paths = set(re.findall(r"api\('(?:GET|POST)', ('[^']*'|\w+\((?:[^()]|\([^()]*\))*\))", js))
    assert len(re.findall(r"\bapi\('", js)) == 9  # 每个接口只有一处调用
    assert paths == {
        "'v1/me'",
        "shopPath('items')",
        "shopPath('standalone/ask')",
        "feedbackPath(cs, 'sent')",
        "feedbackPath(cs, 'edited')",
        "feedbackPath(cs, 'rejected')",
        "shopPath(`reports/adoption${sinceParam()}`)",
        "shopPath('knowledge/gaps')",
        "shopPath('questionnaire')",
    }


# ================================================================ 写进页面的配置


def test_page_config_matches_server_definitions(api: Api):
    config = _config(api.get("/assist").text)
    assert config == json.loads(json.dumps(page_config(Platform.TAOBAO), ensure_ascii=False))
    assert config["platform_name"] == "淘宝"
    assert config["max_text_chars"] == MAX_TEXT_CHARS
    assert config["intents"] == {i.value: INTENT_LABELS[i] for i in Intent}
    assert list(config["reject_reasons"]) == [r.value for r in RejectReason]
    assert config["reject_reasons"]["missing_knowledge"] == "知识缺失"
    assert config["reject_reasons"] == {r.value: REJECT_REASON_LABELS[r] for r in RejectReason}
    assert [q["id"] for q in config["questionnaire"]] == [q.id for q in QUESTIONNAIRE]
    assert config["human_roles"] == ["agent", "manager", "merchant"]
    assert config["console_roles"] == ["manager", "merchant"]
    assert config["merchant_roles"] == ["merchant"]
    assert config["reviewer_roles"] == ["reviewer"]
    assert config["roles"]["reviewer"] == "复核人员"
    assert config["adoption_target"] == 0.5


def test_config_block_cannot_break_out_of_the_script_element():
    text = json_for_html({"x": "</script><script>alert(1)</script>", "y": "a&b<!--"})
    assert "<" not in text and ">" not in text and "&" not in text
    assert json.loads(text) == {"x": "</script><script>alert(1)</script>", "y": "a&b<!--"}


@pytest.mark.parametrize("platform", list(Platform))
def test_each_platform_unit_renders_its_own_page(platform: Platform):
    config = _config(render_page(platform).decode("utf-8"))
    assert config["platform"] == platform.value


# ================================================================ 合规提示、无障碍


def test_page_shows_compliance_notice(api: Api):
    html = api.get("/assist").text
    assert "个人信息委托处理协议" in html and "§8.1" in html and "建议法务确认" in html
    assert "个人信息最小化" in html and "调用大模型" in html
    assert "不做任何自动抓取" in html
    # 须知默认展开；登录后页面脚本收起，摘要（协议要求）一直可见
    page = _parse(html)
    [notice] = [a for t, a in page.tags if t == "details" and a.get("id") == "notice"]
    assert "open" in notice
    summary = html.split("<summary>", 1)[1].split("</summary>", 1)[0]
    assert "个人信息委托处理协议" in summary


def test_every_form_control_has_a_label(api: Api):
    page = _parse(api.get("/assist").text)
    labelled = {a.get("for") for a in page.find("label")}
    for tag in ("input", "textarea", "select"):
        for attrs in page.find(tag):
            assert attrs.get("id") in labelled or attrs.get("aria-label"), attrs
    [html] = page.find("html")
    assert html["lang"] == "zh-CN"
    [viewport] = [a for a in page.find("meta") if a.get("name") == "viewport"]
    assert "width=device-width" in viewport["content"]
    # 错误提示读屏会念出来；状态更新用 aria-live
    assert any(a.get("role") == "alert" for _, a in page.tags)
    assert any(a.get("aria-live") == "polite" for _, a in page.tags)


def test_script_drops_responses_that_arrive_after_a_different_sign_in(api: Api):
    """退出后换人登录：上一位的请求晚回来不写进下一位的界面，上一位的 401 也不把下一位登出。"""
    js = api.get(JS).text
    assert "state.me && state.token === token" in js  # 401 只登出发请求的那一位
    for fn in ("loadItems", "ask", "loadConsole"):
        body = js.split(f"async function {fn}(", 1)[1].split("\n  }\n", 1)[0]
        assert "const me = state.me" in body and "state.me !== me" in body, fn
    # 同一块区域只用最近一次请求的结果（换统计范围、提交问卷后重新加载时，晚回来的旧响应丢掉）
    for fn, key in (("loadItems", "items"), ("loadConsole", "console")):
        body = js.split(f"async function {fn}(", 1)[1].split("\n  }\n", 1)[0]
        assert f"const run = nextRun('{key}');" in body and f"latest('{key}', run)" in body, fn
    # 提问、问卷的按钮：上一位没回来的请求不改下一位的按钮（退出时复位）
    ask = js.split("async function ask(", 1)[1].split("\n  }\n", 1)[0]
    assert "if (state.me === me) setBusy(false);" in ask.split("finally", 1)[1]
    submit = js.split("async function submitQuestionnaire(", 1)[1].split("\n  }\n", 1)[0]
    assert "if (state.me === me) button.disabled = false;" in submit.split("finally", 1)[1]
    sign_out = js.split("function signOut(opts = {})", 1)[1].split("\n  }\n", 1)[0]
    assert "setBusy(false);" in sign_out
    assert "$('questionnaire-button').disabled = false;" in sign_out


def test_assist_page_links_to_the_console_and_sends_reviewers_there(api: Api):
    """页头有去商家控制台的链接；复核人员在这里登录只看到去控制台的提示，不显示话术助手
    （令牌留着，同一标签页打开控制台直接登录；页头显示是谁、有「退出」）。"""
    page = _parse(api.get("/assist").text)
    assert [a["href"] for a in page.find("a") if a.get("class") == "page-link"] == ["console"]
    js = api.get(JS).text
    body = js.split("async function signIn(", 1)[1].split("\n  }\n", 1)[0]
    reviewer = body.split("REVIEWER_ROLES.includes(me.role)", 1)[1].split("return false;", 1)[0]
    assert "state.token = null;" in reviewer and "keepToken(token);" in reviewer
    assert "showIdentity(me);" in reviewer
    assert "href: 'console'" in reviewer and "enterApp" not in reviewer
    assert body.index("REVIEWER_ROLES.includes") < body.index("enterApp()")
