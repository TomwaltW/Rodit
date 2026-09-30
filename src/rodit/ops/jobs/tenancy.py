"""店铺生命周期的离线任务（方案 §8.4、§9.1 租户开通）：在平台单元内跑，和 serve 用同一套环境变量
（RODIT_PLATFORM、RODIT_DATABASE_DSN、RODIT_TOKEN_KEY、盐……，平台绑定同 serve）。

改数据的命令（下面除了 purge-expired --dry-run 都是）先拿本单元的单副本租约
（rodit.ops.lease.job_lease，和 serve 同一个键），做完释放。取消订购的闸门（等在途处理做完再删）、
工具缓存、适配器的去重记录都在 serve 进程内：另起一个进程删数据，serve 手上正在处理的消息照样把
推荐、审计写进来（再也没有人删），serve 的缓存里也还留着这家店的数据。所以 serve 在跑时这些命令
拒绝执行（退出码 2），日志写明改用什么：
    refresh-tokens / purge-expired → serve 自己按间隔跑（RODIT_TOKEN_REFRESH_INTERVAL_S /
        RODIT_PURGE_EXPIRED_INTERVAL_S，serve_jobs），不用另外排定时任务；
    revoke → 平台订购推送 POST /v1/platform/{access}/subscription（取消订购），
        或商家令牌 POST /v1/shops/{shop_id}/revoke；
    onboard-shop → 平台订购推送 POST /v1/platform/{access}/subscription（开通 / 续订）；
    set-status → serve 没有对应接口，停服务窗口里跑。
这些命令在跑时 serve 也起不来（退出码 2，任务做完释放后编排系统重启即可）。

    onboard-shop --shop-id ID --platform P --scopes a,b,c (--token-stdin | --token-file FILE)
                 [--category C] [--expires-at ISO | --expires-days N] [--refresh-expires-at ISO]
                 [--status trial|active]
        开通店铺：核对平台是本单元的平台、权限不超出 tenancy.onboarding_phase、不缺 MVP 必需的权限，
        写店铺（新店铺默认 trial）和加密的授权，审计 auth.granted。
        令牌只从标准输入或文件读（第一行 access_token，可选的第二行是续期令牌），不进命令行参数、
        不回显、不进日志；--expires-at / --expires-days 是 access_token 的到期时间（默认 30 天后；
        --expires-days 要在 (0, 3650] 之内）。
    refresh-tokens
        给 tenancy.refresh_ahead_hours 之内到期的店铺续期（rodit.app.token_lifecycle），只打印各结果
        的店铺数。
    purge-expired [--dry-run]
        授权过期超过 tenancy.purge_after_expiry_days 的店铺：取消订购、删除数据、留下系统事件
        shop.purged。只打印店铺数和删除条数；--dry-run 只数、不删。
    revoke --shop-id ID [--reason TEXT]
        取消订购（商家线下提出、平台推送丢了时用）：删令牌、删除该店铺在本单元的全部数据，
        留下 shop.purged（原因默认 ops）。令牌文件里这家店的访问令牌要另外删掉。
    set-status --shop-id ID --status trial|active|suspended
        切换店铺状态（试用 → 正式 → 停用 → 恢复），审计 shop.status_changed。

退出码：0 完成；1 数据库不可用、续期时平台云代码出错、清理没删完（日志里有要重试的店铺 ID）；
2 参数或设置不对、令牌输入不对、权限 / 平台 / 状态检查不通过、没接续期接口、serve 或另一个任务
正占着单副本租约。
约定见 rodit.ops.jobs：register(subparsers) 加命令；run(args, environ) 处理自己的命令。

serve_jobs(service, ...) 给 serve 用：进程内按间隔跑的续期、过期清理（rodit.ops.background），
用 serve 的同一个 AssistService；出错只记日志（不记令牌），下一轮再来。
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

from pydantic import SecretStr, ValidationError

from rodit.core.clock import CST
from rodit.core.enums import Platform, ShopStatus
from rodit.core.errors import ActionNotAllowed
from rodit.ops.background import PeriodicJob
from rodit.ops.lease import Lease, LeaseHeld

logger = logging.getLogger("rodit.ops")

COMMANDS: frozenset[str] = frozenset(
    {"onboard-shop", "refresh-tokens", "purge-expired", "revoke", "set-status"}
)
OPS_ACTOR = "ops"
DEFAULT_EXPIRES_DAYS = 30.0
MAX_EXPIRES_DAYS = 3650.0
EXIT_FAILURE = 1
EXIT_USAGE = 2
MAX_TOKEN_INPUT_BYTES = 16 * 1024


SERVE_ALTERNATIVES: dict[str, str] = {
    "onboard-shop": (
        "开通 / 续订改用平台订购推送 POST /v1/platform/{access}/subscription（平台对接身份）；"
        "一定要用这个命令就先停 serve"
    ),
    "refresh-tokens": (
        "serve 自己按 RODIT_TOKEN_REFRESH_INTERVAL_S（默认 3600 秒）续期（后台任务 refresh_tokens，"
        "结果看日志和 rodit_background_task_runs_total），不用另外跑；要马上续期就先停 serve"
    ),
    "purge-expired": (
        "serve 自己按 RODIT_PURGE_EXPIRED_INTERVAL_S（默认 21600 秒）清理（后台任务 purge_expired，"
        "结果看日志和 rodit_background_task_runs_total），不用另外跑；"
        "--dry-run 可以和 serve 同时跑；要马上清理就先停 serve"
    ),
    "revoke": (
        "取消订购改用 serve 的接口：平台对接身份推取消订购 POST /v1/platform/{access}/subscription"
        "（留下 shop.purged），或商家令牌 POST /v1/shops/{shop_id}/revoke；"
        "一定要用这个命令就先停 serve"
    ),
    "set-status": "serve 没有切换店铺状态的接口：停 serve 之后再跑（停服务窗口里做）",
}
"""serve 占着租约时，各命令改用什么（日志里原样给出）。"""


class TokenInputError(ValueError):
    """令牌输入不对（空的、行数不对）。消息里不带输入内容。"""


def register(subparsers: Any) -> None:
    onboard = subparsers.add_parser("onboard-shop", help="§8.4 开通店铺：写店铺和加密的授权")
    onboard.add_argument("--shop-id", required=True)
    onboard.add_argument(
        "--platform", required=True, choices=[p.value for p in Platform], help="必须是本单元的平台"
    )
    onboard.add_argument("--category", help="主营类目")
    onboard.add_argument(
        "--scopes", required=True, help="逗号分隔的权限，如 order.read,logistics.read,..."
    )
    source = onboard.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--token-stdin",
        action="store_true",
        help="从标准输入读令牌：第一行 access_token，可选的第二行是续期令牌",
    )
    source.add_argument("--token-file", metavar="FILE", help="从文件读令牌（格式同标准输入）")
    expiry = onboard.add_mutually_exclusive_group()
    expiry.add_argument("--expires-at", type=_iso_time, help="access_token 到期时间（ISO 8601）")
    expiry.add_argument(
        "--expires-days",
        type=float,
        default=None,
        help=(
            f"access_token 多少天后到期（默认 {DEFAULT_EXPIRES_DAYS:g}，最多 {MAX_EXPIRES_DAYS:g}）"
        ),
    )
    onboard.add_argument("--refresh-expires-at", type=_iso_time, help="续期令牌的到期时间")
    onboard.add_argument(
        "--status",
        choices=[ShopStatus.TRIAL.value, ShopStatus.ACTIVE.value],
        help="新店铺的状态（默认 trial）；已有店铺不给时保留原状态",
    )
    subparsers.add_parser("refresh-tokens", help="§8.4 给即将到期的店铺授权续期")
    purge = subparsers.add_parser("purge-expired", help="§8.4 删除授权过期超过宽限期的店铺数据")
    purge.add_argument("--dry-run", action="store_true", help="只数、不删")
    revoke = subparsers.add_parser("revoke", help="§8.4 取消订购：删令牌和该店铺的全部数据")
    revoke.add_argument("--shop-id", required=True)
    revoke.add_argument("--reason", default="ops", help="写进 shop.purged 的原因（默认 ops）")
    status = subparsers.add_parser("set-status", help="切换店铺状态（trial / active / suspended）")
    status.add_argument("--shop-id", required=True)
    status.add_argument("--status", required=True, choices=[s.value for s in ShopStatus])


def run(args: argparse.Namespace, environ: Mapping[str, str]) -> int | None:
    command = getattr(args, "command", None)
    if command not in COMMANDS:
        return None
    out = sys.stdout
    lease: Lease | None = None
    try:
        _check_args(args)  # 参数不对时不读令牌、不碰数据库
        # 先把令牌读出来再拿租约、组装单元：输入不对时不碰数据库
        tokens = _read_tokens(args) if command == "onboard-shop" else None
        if _mutating(args):
            candidate = job_lease(environ, command)
            try:
                candidate.acquire()
            except LeaseHeld as exc:
                logger.error("%s 没有执行：%s", command, _busy_message(command, exc))
                return EXIT_USAGE
            lease = candidate
        service = build_service(environ)
        if command == "onboard-shop":
            assert tokens is not None
            return _onboard(service, args, tokens, out)
        if command == "refresh-tokens":
            return _refresh(service, out)
        if command == "purge-expired":
            return _purge(service, args, out)
        if command == "revoke":
            return _revoke(service, args, out)
        return _set_status(service, args, out)
    except (ValueError, OverflowError, ValidationError, ActionNotAllowed, KeyError) as exc:
        # 业务检查不通过：说明里只有店铺 ID、权限名、平台、状态，没有令牌
        logger.error("%s 没有完成：%s", command, _describe(exc))
        return EXIT_USAGE
    finally:
        _release(lease)


def job_lease(environ: Mapping[str, str], command: str) -> Lease:
    """改数据的命令用的单副本租约（和 serve 同一个键，见模块说明）。测试替换这个函数。"""
    from rodit.ops.lease import job_lease as make_lease
    from rodit.ops.settings import ServerSettings

    settings = ServerSettings.from_env(environ)
    return make_lease(settings.platform, command, environ=environ)


def _mutating(args: argparse.Namespace) -> bool:
    """这条命令改数据（要拿租约）。只有 purge-expired --dry-run 只读。"""
    return not (args.command == "purge-expired" and getattr(args, "dry_run", False))


def _busy_message(command: str, exc: LeaseHeld) -> str:
    who = f"（占用者：{exc.holder}）" if exc.holder else ""
    if exc.held_by_job:
        return f"另一个运维任务正占着平台单元 {exc.platform.value} 的单副本租约{who}，等它做完再跑"
    return (
        f"平台单元 {exc.platform.value} 的 serve 正在运行{who}：改数据的命令不能和 serve 同时跑"
        "（取消订购的闸门、工具缓存都在 serve 进程内，另起进程改数据会和 serve 的在途处理冲突，"
        f"§8.4）。{SERVE_ALTERNATIVES[command]}"
    )


def _release(lease: Lease | None) -> None:
    if lease is None:
        return
    try:
        lease.release()
    except Exception as exc:  # noqa: BLE001  进程退出时连接断开，锁照样释放
        logger.warning("释放单副本租约失败：%s", type(exc).__name__)


def _check_args(args: argparse.Namespace) -> None:
    """不用连库就能判断的参数：--expires-days 要是 (0, MAX_EXPIRES_DAYS] 之内的有限数
    （inf、1e12 换成到期时间会溢出，nan 比较永远不成立）。"""
    days = getattr(args, "expires_days", None)
    if days is not None and not (math.isfinite(days) and 0 < days <= MAX_EXPIRES_DAYS):
        raise ValueError(f"--expires-days 必须大于 0、不超过 {MAX_EXPIRES_DAYS:g}")


def build_service(environ: Mapping[str, str]) -> Any:
    """和 serve 同样组装平台单元（rodit.ops.serve.compose_unit），返回 AssistService。
    测试替换这个函数注入内存单元。"""
    from rodit.app.service import AssistService
    from rodit.ops.serve import compose_unit
    from rodit.ops.settings import ServerSettings

    settings = ServerSettings.from_env(environ)
    return AssistService(compose_unit(settings, environ=environ).unit)


# ================================================================ 各命令


def _onboard(
    service: Any, args: argparse.Namespace, tokens: tuple[SecretStr, SecretStr | None], out: TextIO
) -> int:
    from rodit.core.models import SubscriptionEvent

    access_token, refresh_token = tokens
    now = service.unit.clock.now()
    if args.expires_at is not None:
        expires_at = args.expires_at
    else:
        days = DEFAULT_EXPIRES_DAYS if args.expires_days is None else args.expires_days
        expires_at = now + timedelta(days=days)  # 范围在 _check_args 里核对过
    if expires_at <= now:
        raise ValueError("access_token 的到期时间已经过了")
    event = SubscriptionEvent(
        kind="subscribed",
        shop_id=args.shop_id,
        platform=Platform(args.platform),
        at=now,
        scopes=_scopes(args.scopes),
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        refresh_expires_at=args.refresh_expires_at if refresh_token is not None else None,
        category=args.category,
    )
    status = ShopStatus(args.status) if args.status else None
    shop = service.tenancy.onboard(event, actor=OPS_ACTOR, status=status)
    print(
        f"已开通店铺 {shop.id}（{shop.platform.value}，状态 {shop.status.value}）："
        f"权限 {','.join(sorted(event.scopes))}；"
        f"授权到期 {expires_at.astimezone(CST).isoformat()}；"
        f"续期令牌：{'有' if refresh_token is not None else '无'}",
        file=out,
    )
    return 0


def _refresh(service: Any, out: TextIO) -> int:
    from rodit.app.token_lifecycle import RefreshOutcome, RefreshUnavailable

    try:
        report = service.tenancy.refresh_tokens()
    except RefreshUnavailable as exc:
        logger.error("refresh-tokens 没有完成：%s", exc)
        return EXIT_USAGE
    print(f"令牌续期：{_refresh_summary(report)}", file=out)
    return EXIT_FAILURE if report.counts()[RefreshOutcome.ERROR.value] else 0


def _refresh_summary(report: Any) -> str:
    """只有各结果的店铺数，没有店铺 ID 和令牌。"""
    from rodit.app.token_lifecycle import RefreshOutcome

    counts = report.counts()
    return f"检查 {report.checked} 家店铺，" + "，".join(
        f"{_OUTCOME_LABELS[o]} {counts[o.value]}" for o in RefreshOutcome
    )


_OUTCOME_LABELS: dict[str, str] = {
    "refreshed": "续期成功",
    "needs_reauth": "需要重新授权",
    "retry": "平台暂时不可用（下次再试）",
    "error": "出错",
    "no_refresh_token": "没有续期令牌",
    "refresh_expired": "续期令牌已过期",
    "already_needs_reauth": "已在等重新授权",
    "skipped": "跳过",
}


def _purge(service: Any, args: argparse.Namespace, out: TextIO) -> int:
    from rodit.app.tenancy import PurgeIncomplete

    days = service.unit.config.tenancy.purge_after_expiry_days
    if args.dry_run:
        due = service.tenancy.purge_expired(dry_run=True)
        print(
            f"授权过期超过 {days} 天、待清理的店铺：{len(due)} 家（--dry-run，没有删除）", file=out
        )
        return 0
    try:
        purged = service.tenancy.purge_expired()
    except PurgeIncomplete as exc:
        _print_purged(out, days, exc.purged)
        print(f"没有删完的店铺：{len(exc.failed)} 家（店铺 ID 见日志，用 revoke 重试）", file=out)
        logger.error("过期清理没有删完的店铺：%s", ", ".join(exc.failed))
        return EXIT_FAILURE
    _print_purged(out, days, purged)
    return 0


def _print_purged(out: TextIO, days: int, purged: Mapping[str, Mapping[str, int]]) -> None:
    total = sum(sum(c.values()) for c in purged.values())
    print(
        f"已清理授权过期超过 {days} 天的店铺：{len(purged)} 家，共删除 {total} 条记录",
        file=out,
    )


def _revoke(service: Any, args: argparse.Namespace, out: TextIO) -> int:
    counts = service.tenancy.revoke(args.shop_id, reason=args.reason, actor=OPS_ACTOR)
    print(
        f"已取消订购店铺 {args.shop_id}：共删除 {sum(counts.values())} 条记录"
        f"（令牌 {counts.get('auth', 0)} 条）。"
        "令牌文件（RODIT_TOKENS_FILE）里这家店的访问令牌请一并删掉",
        file=out,
    )
    return 0


def _set_status(service: Any, args: argparse.Namespace, out: TextIO) -> int:
    shop = service.tenancy.set_status(args.shop_id, ShopStatus(args.status), actor=OPS_ACTOR)
    print(f"店铺 {shop.id} 的状态：{shop.status.value}", file=out)
    return 0


# ================================================================ serve 里的定时任务


REFRESH_JOB = "refresh_tokens"
PURGE_JOB = "purge_expired"
SERVE_JOB_INITIAL_DELAY_S = 60.0
"""serve 起来之后多久跑第一轮续期 / 清理（先让服务稳定下来；重启多次也不会一上来就清理）。"""


def refresh_tokens_once(service: Any) -> bool:
    """serve 后台任务：令牌续期一轮。日志只有各结果的店铺数（没有令牌）；有出错的店铺返回 False。"""
    from rodit.app.token_lifecycle import RefreshOutcome

    report = service.tenancy.refresh_tokens()
    errors = report.counts()[RefreshOutcome.ERROR.value]
    log = logger.error if errors else logger.info
    log("令牌续期（serve 后台任务）：%s", _refresh_summary(report))
    return not errors


class ExpiredPurge:
    """serve 后台任务：过期清理一轮（TenancyService.purge_expired）。

    没删完的店铺令牌已经删了，purge_expired 下一轮不会再列出它们：这里记下来，下一轮先按
    取消订购重试（清理是幂等的），删完为止；进程重启会丢掉这份记录，所以日志里写明店铺 ID。"""

    def __init__(self, service: Any) -> None:
        self._service = service
        self._retry: set[str] = set()

    @property
    def pending(self) -> frozenset[str]:
        return frozenset(self._retry)

    def __call__(self) -> bool:
        from rodit.app.tenancy import PURGE_REASON_EXPIRED, SYSTEM_ACTOR, PurgeIncomplete

        tenancy = self._service.tenancy
        purged: dict[str, Mapping[str, int]] = {}
        for shop_id in sorted(self._retry):
            try:
                purged[shop_id] = tenancy.revoke(
                    shop_id, reason=PURGE_REASON_EXPIRED, actor=SYSTEM_ACTOR
                )
            except Exception:  # noqa: BLE001 — revoke 已记日志；留到下一轮
                continue
            self._retry.discard(shop_id)
        try:
            purged.update(tenancy.purge_expired())
        except PurgeIncomplete as exc:
            purged.update(exc.purged)
            self._retry.update(exc.failed)
        if purged:
            total = sum(sum(c.values()) for c in purged.values())
            logger.info(
                "过期清理（serve 后台任务）：%d 家店铺，共删除 %d 条记录", len(purged), total
            )
        if self._retry:
            logger.error(
                "过期清理（serve 后台任务）没有删完的店铺：%s（下一轮重试）",
                ", ".join(sorted(self._retry)),
            )
            return False
        return True


def serve_jobs(
    service: Any,
    *,
    refresh_interval_s: float,
    purge_interval_s: float,
    guard: Callable[[], bool] | None = None,
    initial_delay_s: float = SERVE_JOB_INITIAL_DELAY_S,
) -> list[PeriodicJob]:
    """serve 里按间隔跑的续期、过期清理（见模块说明）。没接续期接口（unit.token_refresher）时
    不排续期；间隔 0 的任务不跑。guard：每轮之前确认单副本租约还在本进程。"""
    jobs: list[PeriodicJob] = []
    if service.unit.token_refresher is not None:
        jobs.append(
            PeriodicJob(
                REFRESH_JOB,
                lambda: refresh_tokens_once(service),
                refresh_interval_s,
                initial_delay_s=min(initial_delay_s, refresh_interval_s),
                guard=guard,
            )
        )
    else:
        logger.info("平台绑定没有续期接口（token_refresher）：serve 不跑令牌续期")
    jobs.append(
        PeriodicJob(
            PURGE_JOB,
            ExpiredPurge(service),
            purge_interval_s,
            initial_delay_s=min(initial_delay_s, purge_interval_s),
            guard=guard,
        )
    )
    return jobs


# ================================================================ 输入


def _read_tokens(args: argparse.Namespace) -> tuple[SecretStr, SecretStr | None]:
    """令牌只从标准输入或文件读：第一行 access_token，可选的第二行是续期令牌。"""
    try:
        if args.token_stdin:
            text = sys.stdin.read(MAX_TOKEN_INPUT_BYTES + 1)
        else:
            path = Path(args.token_file)
            if path.stat().st_size > MAX_TOKEN_INPUT_BYTES:
                raise TokenInputError("令牌文件太大")
            text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TokenInputError(f"令牌读不出来（{type(exc).__name__}）") from None
    if len(text) > MAX_TOKEN_INPUT_BYTES:
        raise TokenInputError("令牌输入太长")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise TokenInputError("没有读到令牌（第一行应为 access_token）")
    if len(lines) > 2:
        raise TokenInputError("令牌输入最多两行：access_token 和可选的续期令牌")
    if any(ch.isspace() for line in lines for ch in line):
        raise TokenInputError("令牌里不能有空白字符")
    refresh = SecretStr(lines[1]) if len(lines) == 2 else None
    return SecretStr(lines[0]), refresh


def _scopes(text: str) -> list[str]:
    seen: list[str] = []
    for part in text.split(","):
        scope = part.strip()
        if scope and scope not in seen:
            seen.append(scope)
    return seen


def _iso_time(text: str) -> datetime:
    """ISO 8601 时间；没写时区的按东八区。"""
    try:
        value = datetime.fromisoformat(text.strip())
    except ValueError:
        raise argparse.ArgumentTypeError("应为 ISO 8601 时间，如 2026-10-30T00:00+08:00") from None
    return value if value.tzinfo is not None else value.replace(tzinfo=CST)


def _describe(exc: Exception) -> str:
    if isinstance(exc, KeyError):
        return f"店铺不存在：{exc.args[0] if exc.args else ''}"
    if isinstance(exc, ValidationError):
        fields = sorted({".".join(str(p) for p in e["loc"]) or "?" for e in exc.errors()})
        return f"参数不合法：{', '.join(fields)}"  # 不带输入值（可能是令牌）
    return str(exc)


__all__ = [
    "COMMANDS",
    "MAX_EXPIRES_DAYS",
    "PURGE_JOB",
    "REFRESH_JOB",
    "SERVE_ALTERNATIVES",
    "ExpiredPurge",
    "build_service",
    "job_lease",
    "refresh_tokens_once",
    "register",
    "run",
    "serve_jobs",
]
