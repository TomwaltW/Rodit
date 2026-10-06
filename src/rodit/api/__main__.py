"""本地演示：python -m rodit.api 在 127.0.0.1:8000 启动 HTTP 接口（仅本地演示，不用于生产）。

- 平台单元用 rodit.app.demo.build_demo_unit（淘宝，合成数据、离线大模型替身，不联网、不需要密钥）；
- 开发用令牌每次启动随机生成，打印在终端上，只在这个进程里有效；仓库里没有任何令牌；
- 只监听本机地址。生产部署在平台云内，由平台的登录体系实现 Authenticator（§9.1、§12.3）；
- 网页工具：模式 C 话术助手（§8.1）在 /assist，商家控制台在 /console，售后审核工作台（§7.3.2）在
  /aftersale，用下面打印的令牌登录；
- 启动前把合成的平台客服指标同步进本单元的仓储（每家演示店铺各一次，走 rodit.app.platform_metrics，
  操作人 system），控制台「报表」页一打开就有接入前后对比（§12.4）。
"""

from __future__ import annotations

import secrets
import sys
from collections.abc import Callable
from typing import Any, TextIO

import uvicorn
from fastapi import FastAPI

from rodit.api.app import create_app
from rodit.api.auth import Principal, StaticTokenAuthenticator
from rodit.app.demo import build_demo_unit
from rodit.app.platform_metrics import PlatformMetricsService
from rodit.app.service import AssistService
from rodit.core.enums import Platform
from rodit.testing.fixtures import AGENT, MANAGER, SHOP_A, SHOP_B

DEMO_HOST = "127.0.0.1"
DEMO_PORT = 8000
DEMO_PLATFORM = Platform.TAOBAO

DEMO_PRINCIPALS: tuple[Principal, ...] = (
    Principal(shop_id=SHOP_A, user_id=AGENT, role="agent"),
    Principal(shop_id=SHOP_A, user_id=MANAGER, role="manager"),
    Principal(shop_id=SHOP_A, user_id="owner_demo", role="merchant"),
    Principal(shop_id=SHOP_A, user_id="taobao_push_demo", role="integration"),
    Principal(shop_id=SHOP_B, user_id="kf_b01", role="agent"),  # 另一家店，用来试串店
)
"""演示账号（合成）：店铺 A 的客服、店长、商家、平台对接，外加店铺 B 的客服。"""

DEMO_PAGES: tuple[tuple[str, str], ...] = (
    ("assist", "话术助手（模式 C）：客服 / 店长 / 商家的令牌"),
    ("console", "商家控制台（报表页已有同步好的平台客服指标）：店长 / 商家的令牌"),
    ("aftersale", "售后审核工作台：客服 / 店长 / 商家的令牌"),
)
"""演示时打印的网页地址（页面名 → 说明、用哪个令牌登录）。"""

METRICS_ACTOR = "system"
"""同步演示平台指标的操作人：读平台数据，不是商家的决定（同 rodit.ops 的同步任务）。"""


def build_demo_app() -> tuple[FastAPI, dict[str, Principal]]:
    """演示应用 + 本次启动随机生成的开发令牌（令牌 → 身份）。平台客服指标已同步进仓储。"""
    service = AssistService(build_demo_unit(DEMO_PLATFORM))
    sync_demo_platform_metrics(service)
    tokens = {f"demo-{secrets.token_urlsafe(24)}": p for p in DEMO_PRINCIPALS}
    app = create_app(service, authenticator=StaticTokenAuthenticator(tokens))
    return app, tokens


def demo_shop_ids() -> list[str]:
    """演示单元里的店铺（演示账号所属的店铺，按出现顺序）。"""
    return list(dict.fromkeys(p.shop_id for p in DEMO_PRINCIPALS))


def sync_demo_platform_metrics(service: AssistService) -> dict[str, int]:
    """把合成的平台客服指标（单元的 metrics_source）同步进仓储：店铺 → 写入条数。

    走应用服务（授权检查、逐条核对、processing 闸门、审计 platform_metrics.synced），和线上的
    同步任务同一条路径；默认同步最近 56 天，够报表的接入前、接入后各 28 天。"""
    metrics = PlatformMetricsService(service.unit, service.processing)
    return {
        shop_id: metrics.sync(shop_id, actor=METRICS_ACTOR).written for shop_id in demo_shop_ids()
    }


def demo_page_url(page: str = "assist") -> str:
    """网页工具的本地地址（默认模式 C 话术助手）。"""
    return f"http://{DEMO_HOST}:{DEMO_PORT}/{page}"


def main(out: TextIO | None = None, *, run: Callable[..., Any] = uvicorn.run) -> int:
    out = out or sys.stdout
    app, tokens = build_demo_app()
    print("【仅本地演示】合成数据、离线大模型替身；令牌每次启动随机生成，不要用于生产。", file=out)
    print(f"接口地址：http://{DEMO_HOST}:{DEMO_PORT}（文档 /docs）", file=out)
    print("网页工具（用下面的令牌登录）：", file=out)
    for page, note in DEMO_PAGES:
        print(f"  {demo_page_url(page)}  {note}", file=out)
    print("开发令牌（只在这次启动的进程里有效）：", file=out)
    for token, p in tokens.items():
        print(f"  {p.role:<11} 店铺 {p.shop_id}  账号 {p.user_id:<16} Bearer {token}", file=out)
    print(
        "示例：curl -X POST -H 'Authorization: Bearer <客服令牌>' "
        "-H 'Content-Type: application/json' "
        f'-d \'{{"session_id":"web_1","question":"能开发票吗"}}\' '
        f"http://{DEMO_HOST}:{DEMO_PORT}/v1/shops/{SHOP_A}/standalone/ask",
        file=out,
    )
    out.flush()
    run(app, host=DEMO_HOST, port=DEMO_PORT)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
