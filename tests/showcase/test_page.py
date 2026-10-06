"""演示站页面本身：怎么提供、响应头和 CSP（和话术助手、商家控制台同一套）、不引用外部地址、
没有内联脚本和样式、脚本只用 textContent 写文字、页面里没有店铺数据。页面行为在
test_browser.py（真浏览器冒烟）里测。"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from importlib import resources

import pytest
from showcase_support import Showcase

from rodit.api import StaticTokenAuthenticator, create_app
from rodit.api.web import CONTENT_SECURITY_POLICY
from rodit.app.demo import build_demo_unit
from rodit.app.service import AssistService
from rodit.core.enums import Platform
from rodit.showcase.app import ASSET_PREFIX, ASSETS, FOOTER_TEXT, PAGE_PATHS

JS = ASSET_PREFIX + "showcase.js"
CSS = ASSET_PREFIX + "showcase.css"


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.scripts: list[tuple[dict[str, str | None], str]] = []
        self._in_script: dict[str, str | None] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.tags.append((tag, attributes))
        if tag == "script":
            self._in_script = attributes
            self._text = []

    def handle_endtag(self, tag):
        if tag == "script" and self._in_script is not None:
            self.scripts.append((self._in_script, "".join(self._text)))
            self._in_script = None

    def handle_data(self, data):
        if self._in_script is not None:
            self._text.append(data)


def _parse(html: str) -> _Page:
    page = _Page()
    page.feed(html)
    return page


def _config(html: str) -> dict:
    page = _parse(html)
    [(attrs, text)] = [(a, t) for a, t in page.scripts if a.get("id") == "showcase-config"]
    assert attrs.get("type") == "application/json"
    return json.loads(text)


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_page_is_served_with_strict_headers(showcase: Showcase, path: str):
    r = showcase.get(path)
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/html; charset=utf-8"
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"
    assert "set-cookie" not in r.headers
    assert r.text.startswith("<!doctype html>")


@pytest.mark.parametrize(
    ("path", "content_type"),
    [(JS, "text/javascript; charset=utf-8"), (CSS, "text/css; charset=utf-8")],
)
def test_assets_are_served_with_types_and_headers(showcase: Showcase, path, content_type):
    r = showcase.get(path)
    assert r.status_code == 200 and r.headers["content-type"] == content_type
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert r.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    "path",
    ["/showcase/showcase.html", "/showcase/../app.py", "/showcase/nope.js", "/static/showcase.js"],
)
def test_only_listed_assets_are_served(showcase: Showcase, path: str):
    r = showcase.get(path)
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_building_the_page_does_not_build_any_platform(showcase: Showcase):
    """打开页面不组装平台单元（第一次请求某个平台的数据时才组装）。"""
    showcase.get("/")
    assert showcase.hub.built() == []


_EXTERNAL = re.compile(r"(?i)\b(?:https?|ftp|wss?):|(?:src|href)\s*=\s*[\"']?//")


def test_page_and_assets_reference_no_external_urls(showcase: Showcase):
    for path in ("/", JS, CSS):
        text = showcase.get(path).text
        assert not _EXTERNAL.search(text), path
    css = showcase.get(CSS).text
    assert "@import" not in css and "url(" not in css  # 字体、图片都不从别处加载
    assert "@font-face" not in css


def test_page_uses_only_same_origin_relative_assets(showcase: Showcase):
    page = _parse(showcase.get("/").text)
    srcs = [a["src"] for t, a in page.tags if t == "script" and a.get("src")]
    hrefs = [a["href"] for t, a in page.tags if t == "link"]
    assert srcs == ["showcase/showcase.js"]
    assert set(hrefs) == {"showcase/showcase.css", "data:,"}
    for name in ASSETS:
        assert showcase.get(ASSET_PREFIX + name).status_code == 200


def test_page_has_no_inline_code_or_styles(showcase: Showcase):
    """CSP 不放 'unsafe-inline'：没有内联脚本、事件属性、style 属性、<style>。"""
    html = showcase.get("/").text
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


def test_script_writes_text_only_with_text_content():
    js = resources.files("rodit.showcase").joinpath("static", "showcase.js").read_text("utf-8")
    for banned in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "eval(",
        "new Function",
        "document.write",
        "setAttribute('style'",
        'setAttribute("style"',
    ):
        assert banned not in js, banned
    assert "localStorage" not in js  # 只用 sessionStorage 记平台和会话，不存令牌
    assert "Authorization" not in js and "Bearer" not in js


def test_page_config_has_no_shop_data(showcase: Showcase):
    html = showcase.get("/").text
    config = _config(html)
    assert config["default_platform"] == "taobao" and config["offline"] is True
    assert [p["id"] for p in config["platforms"]] == ["taobao", "douyin", "pdd"]
    assert [p["name"] for p in config["platforms"]] == ["淘宝·千牛", "抖店·飞鸽", "拼多多·多多客服"]
    assert {p["accent"] for p in config["platforms"]} == {"#ff5000", "#fe2c55", "#e02e24"}
    assert config["max_text_chars"] == 500
    assert config["footer"] == FOOTER_TEXT
    # 店铺名是页面常量；商品、店铺 ID、订单这些数据都要从接口取
    assert "法式碎花连衣裙" not in html and "s_1001" not in html and "as_2001" not in html


def test_page_has_navigation_and_labels(showcase: Showcase):
    html = showcase.get("/").text
    for label in (
        "产品概览",
        "智能辅助工作台",
        "售后审核助手",
        "数据看板",
        "合规与部署",
        "重置演示",
    ):
        assert label in html
    assert "淘宝 · 抖店 · 拼多多 商家 AI 客服" in html
    assert FOOTER_TEXT in html
    assert "售后写操作必须人确认" in html
    page = _parse(html)
    textareas = [a for t, a in page.tags if t == "textarea"]
    labels = {a.get("for") for t, a in page.tags if t == "label"}
    assert all(a.get("id") in labels or a.get("aria-label") for a in textareas)
    assert ("html", {"lang": "zh-CN"}) in page.tags


def test_css_supports_dark_mode_reduced_motion_and_phones():
    css = resources.files("rodit.showcase").joinpath("static", "showcase.css").read_text("utf-8")
    assert "prefers-color-scheme: dark" in css
    assert "prefers-reduced-motion: reduce" in css
    assert "max-width: 720px" in css
    assert '"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", system-ui' in css
    for accent in ("#ff5000", "#fe2c55", "#e02e24"):
        assert accent in css


def test_showcase_is_not_mounted_in_the_production_app():
    """演示站是独立应用：生产的 create_app 里没有 /showcase。"""
    app = create_app(
        AssistService(build_demo_unit(Platform.TAOBAO)),
        authenticator=StaticTokenAuthenticator({}),
    )
    paths = [getattr(r, "path", "") for r in app.routes]
    assert not [p for p in paths if p.startswith("/showcase")]
    for module in ("rodit.api.app", "rodit.ops.serve", "rodit.app.production"):
        source = resources.files(module.rsplit(".", 1)[0]).joinpath(
            module.rsplit(".", 1)[1] + ".py"
        )
        assert "showcase" not in source.read_text("utf-8"), module
