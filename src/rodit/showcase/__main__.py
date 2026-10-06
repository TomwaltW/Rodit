"""启动客户演示站：python -m rodit.showcase [--host 127.0.0.1] [--port 8100]
[--platform taobao|douyin|pdd] [--llm offline|env] [--judge llm|offline] [--config YAML]
[--latency strict|relaxed]；自检：python -m rodit.showcase --selfcheck [--platform …|all]
[--llm env] [--judge …] [--latency …] [--json OUT]

- 默认只监听本机（127.0.0.1）。给客户投屏、在局域网里演示时可以 --host 0.0.0.0，会打印提醒：
  演示站没有登录，同一网络里的人都能打开、都能操作（数据全是合成的）；
- 只认监听地址对应的 Host 头（防 DNS 重绑定，见 showcase.app.allowed_hosts）：只监听本机时用
  127.0.0.1 / localhost / [::1] 打开，指定了别的地址时用这个地址打开；
- 默认用离线替身（不联网、不需要密钥），不读任何 RODIT_LLM_* 环境变量；
- --llm env：接真实的境内大模型（和评测 python -m rodit.eval --llm env 取同一个：--config 里
  llm.base_url 和 llm.model 都写了用配置，否则用 RODIT_LLM_BASE_URL / RODIT_LLM_MODEL /
  RODIT_LLM_TIMEOUT_S；模型服务的差异用 RODIT_LLM_EXTRA_BODY / RODIT_LLM_JSON_MODE /
  RODIT_LLM_LOGPROBS 调，配置文件里是 llm.extra_body / json_mode / logprobs；密钥
  RODIT_LLM_API_KEY 只放本地 .env 或环境的密钥设置）。
  起服务时只允许监听本机地址，否则拒绝启动（退出码 2）：别人连进来演示会花你的大模型额度，
  买家问题（合成数据）也会发给模型服务。配置不全、格式不对也拒绝启动（提示里只有变量名和原因）；
  启动前做一次能力检查（showcase.preflight，和评测同一个 rodit.eval.llm_env.probe_capabilities：
  普通调用、JSON 模式，路由用大模型投票时再加 logprobs，至多三个很小的请求），结果打在横幅里、
  自检报告里也有；没过就退出码 2，提示里写明是网络策略、密钥、地址 / 模型、参数（要改哪个开关）、
  限流、服务端、超时还是证书的问题，只显示主机名；
- --judge：llm（--llm env 时的默认）路由、质检照生产组装、用大模型投票（showcase.realchain）；
  offline（不接大模型时的默认、也是唯一选择）路由用关键词规则、质检用启发式判断。
  回放的合成历史一律用离线替身，不调用大模型；
- --config YAML：引擎配置（和评测同一个读取函数 rodit.core.config.load_config）；
- --latency：strict（默认）照配置的时间预算（§12.1：路由每一票 1 秒、整条消息 4.5 秒）；relaxed
  放宽到演示友好的值（showcase.realchain.RELAXED_*，配置里更宽的保留），页面和横幅都写明"演示放宽
  了时间预算"和实际生效的数字。时间预算是 strict 时启动时提醒：判断层用大模型时，路由每一票和质检
  都限时 decision.timeout_s（默认 1 秒），大多数托管模型答不完，很多消息的路由会退回关键词规则、
  大模型草稿的质检判为不可用（草稿不推荐）；生成超时时退回 FAQ / 模板（页面上都标为降级）；
  演示可以加 --latency relaxed；
- --selfcheck：不起服务，把每个预设场景在演示站里跑一遍并检查（showcase.selfcheck），--json OUT
  另把完整结果写成 JSON（OUT 为 - 时写到标准输出，这时标准输出只放 JSON：配置错误、启动检查的
  结果和"拒绝启动"都写到标准错误）。退出码 0 必查全部通过、1 有没通过的、2 配置错误。
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

import uvicorn
import yaml

from rodit.core.config import EngineConfig, llm_endpoint, load_config
from rodit.core.enums import Platform, ProviderName
from rodit.core.ports import LLMClient
from rodit.eval.llm_env import render_capabilities
from rodit.showcase.app import (
    DEFAULT_HOST,
    create_showcase_app,
    host_literal,
    is_loopback,
    is_unspecified,
)
from rodit.showcase.preflight import PreflightFailed, PreflightResult, run_preflight
from rodit.showcase.realchain import (
    RELAXED_NOTE,
    STRICT_OFFLINE_JUDGE_WARNING,
    STRICT_QA_ONLY_WARNING,
    STRICT_WARNING,
    ChainInfo,
    chain_config,
    chain_info,
    check_judge_config,
)
from rodit.showcase.state import ShowcaseHub

DEFAULT_PORT = 8100
EXIT_USAGE = 2
EXIT_SELFCHECK_FAILED = 1
ALL_PLATFORMS = "all"


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


def _env_llm(config: EngineConfig | None = None) -> LLMClient:
    """--llm env 才调用：和评测取同一个大模型客户端（rodit.eval.cli.env_client：配置里写全了
    llm.base_url / llm.model 用配置，否则按 RODIT_LLM_* 组装，rodit.eval.llm_env 同一套检查）。"""
    from rodit.eval.cli import env_client

    return env_client(config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rodit.showcase",
        description="Rodit 客户演示站（合成数据；默认离线替身，--llm env 接真实大模型；仅演示）",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址（默认 127.0.0.1）")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"端口（默认 {DEFAULT_PORT}）"
    )
    parser.add_argument(
        "--platform",
        choices=[*(p.value for p in Platform), ALL_PLATFORMS],
        default=None,
        help="页面打开时显示的平台（默认 taobao）；--selfcheck 时是要检查的平台（默认 all）",
    )
    parser.add_argument(
        "--llm",
        choices=["offline", "env"],
        default="offline",
        help="offline：离线替身（默认）；env：用 RODIT_LLM_*（或 --config 的 llm）配置的真实大模型",
    )
    parser.add_argument(
        "--judge",
        choices=["llm", "offline"],
        default=None,
        help="判断层（路由、质检）：llm 大模型投票（--llm env 时默认）；offline 离线规则",
    )
    parser.add_argument("--config", metavar="YAML", help="引擎配置（如 config/rodit.yaml）")
    parser.add_argument(
        "--latency",
        choices=["strict", "relaxed"],
        default="strict",
        help="时间预算：strict 按生产（§12.1，默认）；relaxed 演示用的宽松预算",
    )
    parser.add_argument(
        "--selfcheck", action="store_true", help="不起服务，把每个预设场景跑一遍并检查"
    )
    parser.add_argument(
        "--json", metavar="OUT", help="--selfcheck 的完整结果写成 JSON（- 表示标准输出）"
    )
    return parser


def _platforms(value: str | None, selfcheck: bool) -> list[Platform]:
    if value in (None, ALL_PLATFORMS):
        return list(Platform) if selfcheck else [Platform.TAOBAO]
    return [Platform(value)]


def _secrets(client: LLMClient | None) -> list[str]:
    """自检核对"页面数据里没有密钥"用的值（只在内存里比对，不打印、不写报告）。"""
    env = getattr(client, "api_key_env", None)
    value = os.environ.get(env, "").strip() if isinstance(env, str) else ""
    return [value] if value else []


def _banner(
    out: TextIO, chain: ChainInfo, check: PreflightResult | None, *, selfcheck: bool
) -> None:
    title = "【仅演示】Rodit 客户演示站" + ("自检" if selfcheck else "")
    print(title, file=out)
    print("  · 所有店铺、买家、订单、指标都是合成数据，不含任何真实商家或买家信息", file=out)
    if not chain.real:
        print(
            "  · 大模型用离线替身：不联网、不需要密钥（回答只复述依据，质量不代表真实大模型）",
            file=out,
        )
    else:
        print(
            f"  · 真实大模型：{chain.model}（{chain.host}）——工作台的回答由它生成；"
            "演示里买家的问题（合成数据）会发给这个接口",
            file=out,
        )
        if chain.judge_llm and chain.route_by_llm:
            print(
                f"  · 判断层：大模型投票（路由每条消息 {chain.vote_samples} 票、"
                f"质检 {chain.qa_vote_samples} 票，§6.4）；投票失败时路由退回关键词规则、"
                "质检判为不可用（§12.2，页面上标为降级）",
                file=out,
            )
        elif chain.judge_llm:
            print(
                "  · 判断层：路由用关键词规则（配置的 decision.provider = rules）、"
                f"质检用大模型（{chain.qa_vote_samples} 票）；质检失败时判为不可用（§12.2）",
                file=out,
            )
        else:
            print("  · 判断层：离线规则（路由用关键词规则、质检用启发式判断）", file=out)
        print(
            "  · 回放的合成历史用离线替身，不调用大模型；§11.5 质检拦截的第一稿是演示脚本写的",
            file=out,
        )
        if check is not None:
            _print_capabilities(out, check.lines(), head="  · 启动检查通过——", indent="    ")
    print(
        "  · 买家图片是合成的占位图块，图片理解用合成数据的替身（演示站没有任何真实图片）",
        file=out,
    )
    print("  · 售后写操作调用的是模拟的平台接口，不会动任何真实订单", file=out)
    if not chain.real:
        return
    # 数字都是实际生效的（relax_latency 保留配置里更宽的值；strict 照配置）
    numbers = {
        "route": chain.route_timeout_s,
        "generation": chain.generation_timeout_s,
        "budget": chain.message_budget_s,
    }
    if chain.relaxed:
        print(
            f"  ! {RELAXED_NOTE}：判断层每次调用 {chain.route_timeout_s:g} 秒、生成 "
            f"{chain.generation_timeout_s:g} 秒、整条消息 {chain.message_budget_s:g} 秒",
            file=out,
        )
    elif chain.judge_llm:
        warning = STRICT_WARNING if chain.route_by_llm else STRICT_QA_ONLY_WARNING
        print(
            "  ! "
            + warning.format(**numbers)
            + (
                f"（启动检查最慢一步就用了 {check.slowest_ms} ms）"
                if check is not None and check.slowest_ms > chain.route_timeout_s * 1000
                else ""
            ),
            file=out,
        )
    else:
        print("  ! " + STRICT_OFFLINE_JUDGE_WARNING.format(**numbers), file=out)


def _print_capabilities(
    out: TextIO, lines: Sequence[str], *, head: str = "", indent: str = ""
) -> None:
    """能力检查的文本（rodit.eval.llm_env.render_capabilities）：第一行接在 head 后面，
    其余每行前面加 indent。"""
    if not lines:
        return
    print(head + lines[0], file=out)
    for line in lines[1:]:
        print(indent + line, file=out)


def _write_json(target: str, payload: dict[str, Any], out: TextIO, diag: TextIO) -> bool:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if target == "-":
        print(text, file=out)
        return True
    try:
        path = Path(target).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"自检结果写不进 {target}：{type(exc).__name__}", file=diag)
        return False
    return True


def main(
    argv: Sequence[str] | None = None,
    out: TextIO | None = None,
    *,
    run: Callable[..., Any] = uvicorn.run,
    llm_factory: Callable[[], LLMClient] | None = None,
    hub_factory: Callable[..., ShowcaseHub] = ShowcaseHub,
    err: TextIO | None = None,
) -> int:
    """命令行入口（见模块说明）。llm_factory：--llm env 时构造大模型客户端（默认 _env_llm）；
    hub_factory：构造 ShowcaseHub（测试注入不回放历史的版本）；err：--selfcheck --json - 时
    诊断、拒绝启动这些非 JSON 的输出写到这里（默认标准错误），标准输出只放 JSON。"""
    out = out or sys.stdout
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # --help 或参数错误
        return int(exc.code or 0)
    if args.selfcheck and args.json == "-":  # 标准输出只放 JSON：其余的都写到标准错误
        json_out, out = out, err or sys.stderr
    else:
        json_out = out
    if not 0 <= args.port <= 65535:
        print(f"端口不合法：{args.port}", file=out)
        return EXIT_USAGE
    judge = args.judge or ("llm" if args.llm == "env" else "offline")
    if judge == "llm" and args.llm != "env":
        print("--judge llm 需要 --llm env（判断层要用真实大模型投票）", file=out)
        return EXIT_USAGE
    if args.platform == ALL_PLATFORMS and not args.selfcheck:
        print("--platform all 只用于 --selfcheck", file=out)
        return EXIT_USAGE
    if args.json and not args.selfcheck:
        print("--json 只用于 --selfcheck", file=out)
        return EXIT_USAGE
    try:
        config = load_config(args.config) if args.config else None
        effective = chain_config(config, judge=judge, latency=args.latency)
        if judge == "llm":
            check_judge_config(effective)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"拒绝启动：配置不能用：{exc}", file=out)
        return EXIT_USAGE
    loopback = is_loopback(args.host)
    llm: LLMClient | None = None
    check: PreflightResult | None = None
    if args.llm == "env":
        if not loopback and not args.selfcheck:
            print(
                "拒绝启动：--llm env 只能监听本机地址（127.0.0.1 / localhost），"
                f"现在是 {args.host}。"
                "别人连进来演示会用你的大模型额度，买家问题也会发给模型服务。",
                file=out,
            )
            return EXIT_USAGE
        try:
            llm = llm_factory() if llm_factory is not None else _env_llm(config)
        except RuntimeError as exc:  # 没设地址、模型或密钥（信息里只有变量名）
            print(f"拒绝启动：{exc}", file=out)
            return EXIT_USAGE
        except ValueError as exc:  # 格式不对：超时不是正数、地址没有协议（信息里没有密钥）
            print(f"拒绝启动：大模型配置（RODIT_LLM_*）不合法：{exc}", file=out)
            return EXIT_USAGE
        try:
            # 能力检查（和评测同一个）：路由用大模型投票时请求会带 logprobs（§6.4），一并检查
            votes = judge == "llm" and effective.decision.provider == ProviderName.INCLOUD_LLM
            check = run_preflight(
                llm,
                judge_llm=votes,
                vote_samples=effective.decision.vote_samples,
                # 接口写在配置文件里时诊断叫人改 llm.*（RODIT_LLM_* 在那一路不起作用）
                from_config=config is not None and llm_endpoint(config.llm)[2],
            )
        except PreflightFailed as exc:
            if exc.report is not None:
                _print_capabilities(out, render_capabilities(exc.report))
            print(f"拒绝启动：{exc}", file=out)
            return EXIT_USAGE
    platforms = _platforms(args.platform, args.selfcheck)
    hub = hub_factory(
        llm=llm,
        config=config,
        judge=judge,
        latency=args.latency,
        default_platform=platforms[0],
    )
    chain = chain_info(llm, effective, judge=judge, latency=args.latency)
    if not (args.selfcheck and args.json == "-"):  # --json - 时标准输出只放 JSON
        _banner(out, chain, check, selfcheck=args.selfcheck)
    if args.selfcheck:
        return _selfcheck(hub, platforms, llm, check, args.json, json_out, out)
    app = create_showcase_app(hub, host=args.host)
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


def _selfcheck(
    hub: ShowcaseHub,
    platforms: Sequence[Platform],
    llm: LLMClient | None,
    check: PreflightResult | None,
    json_target: str | None,
    out: TextIO,
    diag: TextIO,
) -> int:
    """跑自检。out 是标准输出（--json - 时只放 JSON）；diag 是文字报告、出错提示的去处（--json -
    时是标准错误，否则就是 out）。"""
    from rodit.showcase.selfcheck import render_summary, run_selfcheck

    quiet = json_target == "-"  # 标准输出只放 JSON，逐个场景的文字报告不打印

    def progress(text: str) -> None:
        if not quiet:
            print(text, file=out, flush=True)

    report = run_selfcheck(
        hub,
        platforms,
        secrets=_secrets(llm),
        progress=progress,
        capabilities=check.view() if check is not None else None,
    )
    if not quiet:
        print(render_summary(report), file=out)
    if json_target and not _write_json(json_target, report, out, diag):
        return EXIT_USAGE
    if json_target and not quiet:
        print(f"完整结果（JSON）已写入 {json_target}", file=out)
    return 0 if report["summary"]["ok"] else EXIT_SELFCHECK_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
