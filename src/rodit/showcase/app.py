"""演示站的 HTTP 应用（create_showcase_app）：一个页面 + /showcase/api/ 下的 JSON 接口。

- 独立的 FastAPI 应用，不挂进生产的 rodit.api.create_app，也不经 rodit.ops 部署；
- 页面和静态文件（rodit/showcase/static/）启动时读进内存，响应头和话术助手、商家控制台一样
  （rodit.api.web.SECURITY_HEADERS：严格 CSP，只允许同源脚本、样式和请求，不允许内联脚本、
  内联样式、被别的页面嵌入）；页面不引用任何外部地址（CDN、字体），平台云内网也能打开；
- 演示站没有登录：所有数据都是合成的，没有令牌、没有真实店铺；
- 接口的输入都按 showcase.schemas 校验（不认识的字段拒绝，文字最多 500 字，买家图片只收固定的
  合成标识、最多 3 张，请求体最大 16 KiB，不带 Content-Length 的分块上传拒绝）；
  不认识的平台返回 404，输入不合法返回 422，操作和当前状态冲突（如推荐已经处理过、售后单状态
  变了）返回 409，错误都是 {"error": 代码, "message": 中文说明}；
- 返回的数据里没有买家标识（buyer_ref）、买家昵称：工作台的买家只按场景称呼。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from importlib import resources
from typing import Any

from fastapi import APIRouter, FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from rodit.api.web import PAGE_HEADERS, SECURITY_HEADERS, json_for_html
from rodit.core.enums import Platform
from rodit.core.pii import (
    ADDRESS_MASK,
    CARD_MASK,
    CONTACT_MASK,
    EMAIL_MASK,
    ID_MASK,
    PHONE_MASK,
    mask_pii,
)
from rodit.feedback import REJECT_REASON_LABELS
from rodit.showcase.dashboard import SYNTHETIC_LABEL, dashboard, overview
from rodit.showcase.data import PROFILES
from rodit.showcase.images import (
    IMAGE_PLATFORMS,
    MAX_CHAT_IMAGES,
    NO_IMAGE_REASON,
    tiles_config,
)
from rodit.showcase.labels import (
    AFTERSALE_RULE_TEXTS,
    E2E_TARGET_MS,
    REVIEW_STATE_LABELS,
    STATE_STRIP,
)
from rodit.showcase.rollout import rollout_view
from rodit.showcase.schemas import (
    ID_PATTERN,
    MAX_BODY_BYTES,
    MAX_TEXT_CHARS,
    ChatRequest,
    ConfirmRequest,
    FeedbackRequest,
    MaskRequest,
    NegotiateRequest,
    ResetRequest,
)
from rodit.showcase.state import NotFound, PlatformDemo, ShowcaseError, ShowcaseHub

PAGE_PATHS: tuple[str, ...] = ("/", "/showcase")
ASSET_PREFIX = "/showcase/"
ASSETS: dict[str, str] = {
    "showcase.js": "text/javascript",
    "showcase.css": "text/css",
}
PAGE_FILE = "showcase.html"
CONFIG_PLACEHOLDER = "__SHOWCASE_CONFIG__"
API_PREFIX = "/showcase/api"

FOOTER_TEXT = "演示站使用合成数据和离线大模型替身；生产部署在各平台云内（方案 §9.1）"

PLATFORM_NAMES: dict[Platform, str] = {
    Platform.TAOBAO: "淘宝·千牛",
    Platform.DOUYIN: "抖店·飞鸽",
    Platform.PDD: "拼多多·多多客服",
}

SHORT_NAMES: dict[Platform, str] = {
    Platform.TAOBAO: "淘宝",
    Platform.DOUYIN: "抖店",
    Platform.PDD: "拼多多",
}

MASK_TOKENS: dict[str, str] = {
    PHONE_MASK: "手机号",
    ADDRESS_MASK: "收件地址",
    ID_MASK: "证件号",
    EMAIL_MASK: "邮箱",
    CARD_MASK: "银行卡号",
    CONTACT_MASK: "微信 / QQ 等联系方式",
}

API_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def page_config(hub: ShowcaseHub) -> dict[str, Any]:
    """写进页面数据块的配置：平台、中文名、长度上限。都是服务端常量，没有店铺数据。"""
    return {
        "default_platform": hub.default_platform.value,
        "offline": hub.offline,
        "max_text_chars": MAX_TEXT_CHARS,
        "max_chat_images": MAX_CHAT_IMAGES,
        "images": tiles_config(),
        "no_image_reason": NO_IMAGE_REASON,
        "e2e_target_ms": E2E_TARGET_MS,
        "synthetic_label": SYNTHETIC_LABEL,
        "footer": FOOTER_TEXT,
        "platforms": [
            {
                "id": p.value,
                "name": PLATFORM_NAMES[p],
                "short_name": SHORT_NAMES[p],
                "shop_name": PROFILES[p].shop_name,
                "workbench": PROFILES[p].workbench,
                "access": PROFILES[p].access,
                "access_label": PROFILES[p].access_label,
                "accent": PROFILES[p].accent,
                "can_send_images": p in IMAGE_PLATFORMS,
            }
            for p in Platform
        ],
        "reject_reasons": {r.value: label for r, label in REJECT_REASON_LABELS.items()},
        "review_states": [{"state": s.value, "label": REVIEW_STATE_LABELS[s]} for s in STATE_STRIP],
        "aftersale_rules": AFTERSALE_RULE_TEXTS,
    }


def _static(name: str) -> bytes:
    return resources.files("rodit.showcase").joinpath("static", name).read_bytes()


def render_page(hub: ShowcaseHub) -> bytes:
    html = _static(PAGE_FILE).decode("utf-8")
    if html.count(CONFIG_PLACEHOLDER) != 1:
        raise RuntimeError("页面模板里的配置占位符必须恰好出现一次")
    return html.replace(CONFIG_PLACEHOLDER, json_for_html(page_config(hub))).encode("utf-8")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": code, "message": message}, headers=API_HEADERS
    )


_ERROR_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    411: "length_required",
    413: "too_large",
    422: "invalid",
}


def platform_of(value: str) -> Platform:
    try:
        return Platform(value)
    except ValueError:
        raise NotFound(f"没有这个平台：{value}") from None


def mask_view(text: str) -> dict[str, Any]:
    """个人信息最小化的演示（§5.3）：打码后的文字 + 遮掉了哪几类。"""
    masked = mask_pii(text)
    found = []
    for token, label in MASK_TOKENS.items():
        count = masked.count(token) - text.count(token)
        if count > 0:
            found.append({"token": token, "label": label, "count": count})
    return {"masked": masked, "found": found, "changed": masked != text}


def build_api_router(hub: ShowcaseHub) -> APIRouter:
    router = APIRouter(prefix=API_PREFIX)
    platform_query = Query(pattern=r"^[a-z]{1,16}$")
    aftersale_path = Path(pattern=ID_PATTERN)

    def demo_for(platform: str) -> PlatformDemo:
        return hub.get(platform_of(platform))

    @router.get("/overview")
    def get_overview(platform: str = platform_query) -> dict[str, Any]:
        demo = demo_for(platform)
        return {**overview(demo), "build_ms": demo.build_ms}

    @router.get("/scenarios")
    def get_scenarios(platform: str = platform_query) -> dict[str, Any]:
        demo = demo_for(platform)
        can_send_images = demo.platform in IMAGE_PLATFORMS
        return {
            "platform": demo.platform.value,
            "access": demo.profile.access,
            "access_label": demo.profile.access_label,
            "shop_name": demo.profile.shop_name,
            "can_send_images": can_send_images,
            "no_image_reason": None if can_send_images else NO_IMAGE_REASON,
            "scenarios": demo.scenario_list(),
            "rollout": rollout_view(demo),
        }

    @router.post("/chat")
    def post_chat(body: ChatRequest) -> dict[str, Any]:
        return demo_for(body.platform).chat(body.scenario, body.text, body.images)

    @router.post("/feedback")
    def post_feedback(body: FeedbackRequest) -> dict[str, Any]:
        return demo_for(body.platform).feedback(
            body.suggestion_id,
            body.action,
            text=body.text,
            reason=body.reason.value if body.reason else None,
        )

    @router.get("/aftersales")
    def get_aftersales(platform: str = platform_query) -> dict[str, Any]:
        demo = demo_for(platform)
        return {"platform": demo.platform.value, **demo.aftersales()}

    @router.post("/aftersales/{aftersale_id}/confirm")
    def post_confirm(body: ConfirmRequest, aftersale_id: str = aftersale_path) -> dict[str, Any]:
        return demo_for(body.platform).confirm(
            aftersale_id,
            body.action.value,
            expected_status_version=body.expected_status_version,
            reason_code=body.reason_code,
        )

    @router.post("/aftersales/{aftersale_id}/negotiate")
    def post_negotiate(
        body: NegotiateRequest, aftersale_id: str = aftersale_path
    ) -> dict[str, Any]:
        return demo_for(body.platform).negotiate(aftersale_id, done=body.done)

    @router.get("/dashboard")
    def get_dashboard(platform: str = platform_query) -> dict[str, Any]:
        return dashboard(demo_for(platform))

    @router.post("/mask")
    def post_mask(body: MaskRequest) -> dict[str, Any]:
        return mask_view(body.text)

    @router.post("/reset")
    def post_reset(body: ResetRequest) -> dict[str, Any]:
        demo = hub.reset(platform_of(body.platform))
        return {
            "platform": demo.platform.value,
            "history_messages": demo.history.messages,
            "build_ms": demo.build_ms,
        }

    return router


def create_showcase_app(hub: ShowcaseHub | None = None) -> FastAPI:
    """演示站应用（见模块说明）。hub 为空时用离线替身、真实时钟。"""
    hub = hub or ShowcaseHub()
    app = FastAPI(
        title="Rodit 演示站",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.hub = hub

    @app.middleware("http")
    async def limit_and_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        length = request.headers.get("content-length")
        if length is None and "chunked" in request.headers.get("transfer-encoding", "").lower():
            # 分块上传量不出大小：一律要求带 Content-Length（浏览器的 fetch 都会带）
            return _error(411, "length_required", "请求要带 Content-Length")
        if length is not None:
            if not length.isdigit():
                return _error(400, "bad_request", "Content-Length 不合法")
            if int(length) > MAX_BODY_BYTES:
                return _error(413, "too_large", f"请求体不能超过 {MAX_BODY_BYTES} 字节")
        response = await call_next(request)
        if request.url.path.startswith(API_PREFIX):
            for key, value in API_HEADERS.items():
                response.headers.setdefault(key, value)
        return response

    @app.exception_handler(ShowcaseError)
    async def showcase_error(request: Request, exc: ShowcaseError) -> JSONResponse:
        return _error(exc.status, _ERROR_CODES.get(exc.status, "error"), exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _ERROR_CODES.get(exc.status_code, "error")
        message = "没有这个地址" if exc.status_code == 404 else str(exc.detail)
        return _error(exc.status_code, code, message)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            {"loc": [str(p) for p in e.get("loc", ())], "msg": str(e.get("msg", ""))}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": "invalid", "message": "输入不合法", "problems": problems},
            headers=API_HEADERS,
        )

    page = render_page(hub)
    for path in PAGE_PATHS:
        app.add_api_route(
            path,
            _responder(page, "text/html", PAGE_HEADERS),
            methods=["GET"],
            include_in_schema=False,
        )
    for name, media_type in ASSETS.items():
        app.add_api_route(
            ASSET_PREFIX + name,
            _responder(_static(name), media_type, SECURITY_HEADERS),
            methods=["GET"],
            include_in_schema=False,
        )
    app.include_router(build_api_router(hub))
    return app


def _responder(body: bytes, media_type: str, headers: dict[str, str]) -> Callable[[], Response]:
    def serve() -> Response:
        return Response(content=body, media_type=media_type, headers=headers)

    return serve


__all__ = [
    "API_PREFIX",
    "ASSETS",
    "ASSET_PREFIX",
    "FOOTER_TEXT",
    "PAGE_PATHS",
    "build_api_router",
    "create_showcase_app",
    "mask_view",
    "page_config",
    "platform_of",
    "render_page",
]
