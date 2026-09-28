"""本地演示：python -m rodit.api 在 127.0.0.1:8000 启动 HTTP 接口（仅本地演示，不用于生产）。

- 平台单元用 rodit.app.demo.build_demo_unit（淘宝，合成数据、离线大模型替身，不联网、不需要密钥）；
- 开发用令牌每次启动随机生成，打印在终端上，只在这个进程里有效；仓库里没有任何令牌；
- 只监听本机地址。生产部署在平台云内，由平台的登录体系实现 Authenticator（§9.1、§12.3）。
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
from rodit.app.service import AssistService
from rodit.core.enums import Platform
from rodit.testing.fixtures import AGENT, MANAGER, SHOP_A, SHOP_B

DEMO_HOST = "127.0.0.1"
DEMO_PORT = 8000

DEMO_PRINCIPALS: tuple[Principal, ...] = (
    Principal(shop_id=SHOP_A, user_id=AGENT, role="agent"),
    Principal(shop_id=SHOP_A, user_id=MANAGER, role="manager"),
    Principal(shop_id=SHOP_A, user_id="owner_demo", role="merchant"),
    Principal(shop_id=SHOP_A, user_id="taobao_push_demo", role="integration"),
    Principal(shop_id=SHOP_B, user_id="kf_b01", role="agent"),  # 另一家店，用来试串店
)
"""演示账号（合成）：店铺 A 的客服、店长、商家、平台对接，外加店铺 B 的客服。"""


def build_demo_app() -> tuple[FastAPI, dict[str, Principal]]:
    """演示应用 + 本次启动随机生成的开发令牌（令牌 → 身份）。"""
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    tokens = {f"demo-{secrets.token_urlsafe(24)}": p for p in DEMO_PRINCIPALS}
    app = create_app(service, authenticator=StaticTokenAuthenticator(tokens))
    return app, tokens


def main(out: TextIO | None = None, *, run: Callable[..., Any] = uvicorn.run) -> int:
    out = out or sys.stdout
    app, tokens = build_demo_app()
    print("【仅本地演示】合成数据、离线大模型替身；令牌每次启动随机生成，不要用于生产。", file=out)
    print(f"接口地址：http://{DEMO_HOST}:{DEMO_PORT}（文档 /docs）", file=out)
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
