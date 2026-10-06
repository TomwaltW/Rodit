"""启动客户演示站：python -m rodit.showcase [--host 127.0.0.1] [--port 8100]
[--platform taobao|douyin|pdd] [--llm offline|env]

- 默认只监听本机（127.0.0.1）。给客户投屏、在局域网里演示时可以 --host 0.0.0.0，会打印提醒：
  演示站没有登录，同一网络里的人都能打开、都能操作（数据全是合成的）；
- 只认监听地址对应的 Host 头（防 DNS 重绑定，见 showcase.app.allowed_hosts）：只监听本机时用
  127.0.0.1 / localhost / [::1] 打开，指定了别的地址时用这个地址打开；
- 默认用离线替身（不联网、不需要密钥），不读任何 RODIT_LLM_* 环境变量；
- --llm env：工作台里的回答改用真实的境内大模型生成（RODIT_LLM_BASE_URL / RODIT_LLM_MODEL /
  RODIT_LLM_API_KEY，只在本地开发机上这样做，密钥放本地 .env）；这时只允许监听本机地址，
  否则拒绝启动（退出码 2）：别人连进来演示会花你的大模型额度，买家问题也会发给模型服务。
  RODIT_LLM_* 没设或格式不对（超时不是正数、地址没有 http:// 或 https://）也拒绝启动（退出码 2，
  提示里只有变量名和原因，没有密钥）。路由、质检、历史数据回放仍用离线替身。
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
from collections.abc import Callable, Sequence
from typing import Any, TextIO

import uvicorn

from rodit.core.enums import Platform
from rodit.core.ports import LLMClient
from rodit.showcase.app import (
    DEFAULT_HOST,
    create_showcase_app,
    host_literal,
    is_loopback,
    is_unspecified,
)
from rodit.showcase.state import ShowcaseHub

DEFAULT_PORT = 8100
EXIT_USAGE = 2


def open_url(host: str, port: int) -> str:
    """启动后打印的网址：用监听的地址（IPv6 加方括号）；监听所有网卡（0.0.0.0 / ::）时给本机的
    回环地址——:: 只收 IPv6（asyncio 给 IPv6 套接字设了 IPV6_V6ONLY），所以是 [::1]。"""
    if is_unspecified(host):
        stripped = host.strip().strip("[]")
        ip = ipaddress.ip_address(stripped) if stripped else None
        shown = "[::1]" if ip is not None and ip.version == 6 else "127.0.0.1"
    else:
        shown = host_literal(host)
    return f"http://{shown}:{port}/"


def _env_llm() -> LLMClient:
    """--llm env 才调用：按 RODIT_LLM_* 组装真实大模型客户端（rodit.eval.llm_env 同一套检查）。"""
    from rodit.eval.llm_env import client_from_env

    return client_from_env()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rodit.showcase",
        description="Rodit 客户演示站（合成数据、离线大模型替身，仅演示）",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址（默认 127.0.0.1）")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"端口（默认 {DEFAULT_PORT}）"
    )
    parser.add_argument(
        "--platform",
        choices=[p.value for p in Platform],
        default=Platform.TAOBAO.value,
        help="页面打开时显示的平台（默认 taobao）",
    )
    parser.add_argument(
        "--llm",
        choices=["offline", "env"],
        default="offline",
        help="offline：离线替身（默认）；env：用 RODIT_LLM_* 配置的真实大模型生成回答",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    out: TextIO | None = None,
    *,
    run: Callable[..., Any] = uvicorn.run,
    llm_factory: Callable[[], LLMClient] = _env_llm,
) -> int:
    out = out or sys.stdout
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # --help 或参数错误
        return int(exc.code or 0)
    if not 0 <= args.port <= 65535:
        print(f"端口不合法：{args.port}", file=out)
        return EXIT_USAGE
    loopback = is_loopback(args.host)
    llm: LLMClient | None = None
    if args.llm == "env":
        if not loopback:
            print(
                "拒绝启动：--llm env 只能监听本机地址（127.0.0.1 / localhost），"
                f"现在是 {args.host}。"
                "别人连进来演示会用你的大模型额度，买家问题也会发给模型服务。",
                file=out,
            )
            return EXIT_USAGE
        try:
            llm = llm_factory()
        except RuntimeError as exc:  # 没设地址、模型或密钥（信息里只有变量名）
            print(f"拒绝启动：{exc}", file=out)
            return EXIT_USAGE
        except ValueError as exc:  # 格式不对：超时不是正数、地址没有协议（信息里没有密钥）
            print(f"拒绝启动：大模型配置（RODIT_LLM_*）不合法：{exc}", file=out)
            return EXIT_USAGE
    hub = ShowcaseHub(llm=llm, default_platform=Platform(args.platform))
    app = create_showcase_app(hub, host=args.host)
    print("【仅演示】Rodit 客户演示站", file=out)
    print("  · 所有店铺、买家、订单、指标都是合成数据，不含任何真实商家或买家信息", file=out)
    if llm is None:
        print(
            "  · 大模型用离线替身：不联网、不需要密钥（回答只复述依据，质量不代表真实大模型）",
            file=out,
        )
    else:
        print(
            f"  · 工作台的回答用真实大模型（{getattr(llm, 'model', '')}）；路由、质检仍是离线替身",
            file=out,
        )
    print(
        "  · 买家图片是合成的占位图块，图片理解用合成数据的替身（演示站没有任何真实图片）",
        file=out,
    )
    print("  · 售后写操作调用的是模拟的平台接口，不会动任何真实订单", file=out)
    if not loopback:
        print(
            f"  ! 注意：正在监听 {args.host}，同一网络里的人都能打开并操作演示站（没有登录）；"
            "演示完请关掉",
            file=out,
        )
    print(f"打开：{open_url(args.host, args.port)}", file=out)
    out.flush()
    run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
