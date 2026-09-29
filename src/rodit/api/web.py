"""模式 C 独立话术助手网页（方案 §8.1 模式 C、§5.6 推荐卡片、§5.8 反馈、§12.4 商家报表）。

一个自带的页面：HTML + 原生 JS + CSS，没有构建步骤，不引用任何外部地址（CDN、字体）——要能在平台云 /
内网里直接用。文件在 rodit/api/static/，应用启动时读进内存。

- GET / 和 GET /assist：页面；GET /assist/assist.js、/assist/assist.css：脚本和样式。
  页面本身不含任何店铺数据，不需要鉴权；数据都由页面脚本带 Bearer 令牌调用 /v1 接口取
  （/v1/me 登录，/v1/shops/{shop_id}/... 提问、反馈、报表），鉴权和串店检查都在接口上（§9.2）。
- 令牌只放在 sessionStorage（关掉标签页就没了），不进 URL、不进日志；输入框没有 name，
  CSP 的 form-action 'none' 禁止表单提交，脚本没加载时令牌和买家问题也不会跑进网址。
- 响应头带严格的 CSP：只允许同源的脚本、样式和请求，不允许内联脚本、内联样式、被别的页面嵌入。
  页面脚本只用 textContent 写服务端和用户给的文字，不用 innerHTML（防 XSS）。
- 中文名（问题类型、弃用原因、置信度……）、问卷题目、文字长度上限在启动时写进页面里的 JSON 数据块
  （<script type="application/json">，浏览器不执行、不受 CSP 限制），和服务端用同一份定义。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib import resources
from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response

from rodit.api.auth import CONSOLE_ROLES, HUMAN_ROLES, MERCHANT_ROLES
from rodit.api.schemas import MAX_TEXT_CHARS
from rodit.core.enums import INTENT_LABELS, Confidence, DraftKind, Intent, Platform, RejectReason
from rodit.core.evidence import PLATFORM_NAMES
from rodit.feedback import REJECT_REASON_LABELS
from rodit.knowledge import QUESTIONNAIRE

PAGE_PATHS: tuple[str, ...] = ("/", "/assist")
"""页面地址。脚本和样式用相对地址引用，两个地址下都能找到。"""

ASSET_PREFIX = "/assist/"
ASSETS: dict[str, str] = {
    "assist.js": "text/javascript",
    "assist.css": "text/css",
}
"""静态文件 → 内容类型。只有这几个文件对外，不按请求路径读文件系统。"""

PAGE_FILE = "assist.html"
CONFIG_PLACEHOLDER = "__RODIT_PAGE_CONFIG__"

PHASE0_ADOPTION_TARGET = 0.5
"""Phase 0 验收：模式 C 下种子商家推荐采纳率 ≥ 50%（§15；按推荐组算，同 feedback.phase_gate）。"""

CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    )
)

SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}

CONFIDENCE_LABELS: dict[Confidence, str] = {Confidence.HIGH: "高", Confidence.MEDIUM: "中"}

DRAFT_KIND_LABELS: dict[DraftKind, str] = {
    DraftKind.FAQ: "FAQ",
    DraftKind.TEMPLATE: "模板",
    DraftKind.LLM: "AI 生成",
}

ROLE_LABELS: dict[str, str] = {
    "agent": "客服",
    "manager": "店长",
    "merchant": "商家",
    "integration": "平台对接",
}

DEGRADED_LABELS: dict[str, str] = {
    "storage": "存储",
    "knowledge": "知识库",
    "understanding": "会话理解",
    "router": "路由",
    "decision": "判断层（已退回关键词规则）",
    "decision:qa": "判断层质检（只跑了硬规则）",
    "retriever": "检索",
    "generator": "答案生成",
    "qa": "质检",
    "audit": "审计留痕",
    "engine": "问答引擎",
    "llm": "大模型",
    "tool": "平台接口",
}
"""降级标记（rodit.engine 模块文档）的中文名；"tool:<kind>" 按前缀 tool 显示。"""


def page_config(platform: Platform) -> dict[str, Any]:
    """写进页面的配置：中文名、问卷题目、长度上限、角色分组。都是服务端常量，没有店铺数据。"""
    return {
        "platform": platform.value,
        "platform_name": PLATFORM_NAMES.get(platform, platform.value),
        "max_text_chars": MAX_TEXT_CHARS,
        "adoption_target": PHASE0_ADOPTION_TARGET,
        "human_roles": sorted(HUMAN_ROLES),
        "console_roles": sorted(CONSOLE_ROLES),
        "merchant_roles": sorted(MERCHANT_ROLES),
        "roles": ROLE_LABELS,
        "intents": {i.value: INTENT_LABELS[i] for i in Intent},
        "reject_reasons": {r.value: REJECT_REASON_LABELS[r] for r in RejectReason},
        "confidence": {c.value: label for c, label in CONFIDENCE_LABELS.items()},
        "kinds": {k.value: label for k, label in DRAFT_KIND_LABELS.items()},
        "degraded": DEGRADED_LABELS,
        "questionnaire": [{"id": q.id, "text": q.text} for q in QUESTIONNAIRE],
    }


def json_for_html(data: Any) -> str:
    """JSON → 可以放进 <script type="application/json"> 的文字：转义 < > &，
    内容里出现 "</script>" 也不会提前结束数据块。"""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _static(name: str) -> bytes:
    return resources.files("rodit.api").joinpath("static", name).read_bytes()


def render_page(platform: Platform) -> bytes:
    html = _static(PAGE_FILE).decode("utf-8")
    if html.count(CONFIG_PLACEHOLDER) != 1:
        raise RuntimeError("页面模板里的配置占位符必须恰好出现一次")
    return html.replace(CONFIG_PLACEHOLDER, json_for_html(page_config(platform))).encode("utf-8")


def _responder(body: bytes, media_type: str) -> Callable[[], Response]:
    def serve() -> Response:
        return Response(content=body, media_type=media_type, headers=SECURITY_HEADERS)

    return serve


def build_web_router(platform: Platform) -> APIRouter:
    """网页工具的路由（不进 OpenAPI 文档）。文件在这里读一次，之后只从内存返回。"""
    router = APIRouter(include_in_schema=False)
    page = _responder(render_page(platform), "text/html")
    for path in PAGE_PATHS:
        router.add_api_route(path, page, methods=["GET"], name=f"web_page{path}")
    for name, media_type in ASSETS.items():
        router.add_api_route(
            ASSET_PREFIX + name,
            _responder(_static(name), media_type),
            methods=["GET"],
            name=f"web_asset_{name}",
        )
    return router


__all__ = [
    "ASSETS",
    "ASSET_PREFIX",
    "CONTENT_SECURITY_POLICY",
    "PAGE_PATHS",
    "PHASE0_ADOPTION_TARGET",
    "SECURITY_HEADERS",
    "build_web_router",
    "json_for_html",
    "page_config",
    "render_page",
]
