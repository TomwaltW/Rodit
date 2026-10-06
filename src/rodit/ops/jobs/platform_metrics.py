"""平台客服指标的离线任务（方案 §12.4 "平台客服指标接入前后对比"、§15 Phase 2 商家报表）。在平台
单元内跑，约定见 rodit.ops.jobs。

    sync-platform-metrics [--days N] [--shop-id ID ...]
        从平台云内的平台指标接口（平台绑定 metrics_source）同步本单元每家有授权、没停用的店铺最近
        N 天（默认 56，1–400，到昨天为止）的客服考核数据（rodit.app.platform_metrics：先查授权、
        整段拉完核对后才写，按 source=platform 覆盖，商家登记的数不碰）。--shop-id 可以给多次，只
        同步这些店铺。每家店打印拉到 / 写入 / 跳过的条数，没有买家数据。每天排一次即可（平台会
        修正最近几天的数，日常 --days 7 就够；第一次接入时用默认的 56 天，报表的接入前后各 28 天
        才有数）。

不拿单副本租约，和 serve 同时跑：
- 写的只是平台给的数，按（店铺、指标、日期、来源）覆盖：和 serve 里的同步接口（POST
  platform-metrics/sync）同时写同一家店，结果一样；serve 进程里没有这张表的缓存，也没有别的
  进程内状态要和它对齐；
- 唯一的冲突是取消订购 / 过期清理（闸门在 serve 进程内，本任务看不见），和 compliance-scan
  一样处理：每家店写之前看任务开始之后（放 5 分钟时钟余量）有没有开始清理这家店（系统事件
  shop.purge_started），有就这次不写；写完再确认授权还在——不在了（期间被取消订购，清理可能已经
  删过这张表）就再删一次这家店的指标和审计（purge_shop 幂等，§8.4 不留迟到的数据）；授权还在但
  期间开始过清理（清理完又重新开通）就提示核对，这次写的是平台给这家店的数，重新同步一次即可。
- 拿租约的话：serve 一直占着它，这个任务在生产上永远跑不了（只能停服务窗口里跑）。

环境变量、平台绑定同 serve（RODIT_PLATFORM、RODIT_DATABASE_DSN、RODIT_TOKEN_KEY、盐……；绑定要有
metrics_source）。退出码：0 完成（没有店铺也是 0）；1 有店铺没同步成（平台接口不可用、接口返回了
别的店铺的数据、给的 --shop-id 没有授权或不存在；数据库错误按 rodit.ops.jobs 的约定）；2 参数不对、
平台绑定没有 metrics_source、单元没有指标存储。
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, TextIO

from rodit.core.audit import AuditAction
from rodit.core.enums import ShopStatus

logger = logging.getLogger("rodit.ops")

COMMAND = "sync-platform-metrics"
COMMANDS = (COMMAND,)
EXIT_FAILURE = 1
EXIT_BAD_INPUT = 2
OPS_ACTOR = "ops"
FAR_FUTURE = datetime(9999, 12, 31, tzinfo=UTC)
"""expiring_shop_ids 给一个很远的时间：本单元全部有授权的店铺（只有 ID 和到期时间，不读令牌）。"""
PURGE_CLOCK_MARGIN = timedelta(minutes=5)
"""找"任务期间开始的清理"时往前放的余量（serve 和本任务两台机器的时钟差，同 compliance-scan）。"""


def register(subparsers: Any) -> None:
    from rodit.app.platform_metrics import DEFAULT_SYNC_DAYS
    from rodit.feedback.platform_metrics import MAX_HISTORY_DAYS

    cmd = subparsers.add_parser(
        COMMAND, help="§12.4 从平台接口同步客服考核指标（接入前后对比报表用）"
    )
    cmd.add_argument(
        "--days",
        type=int,
        default=DEFAULT_SYNC_DAYS,
        help=f"同步最近几天，到昨天为止（默认 {DEFAULT_SYNC_DAYS}，1–{MAX_HISTORY_DAYS}）",
    )
    cmd.add_argument(
        "--shop-id",
        action="append",
        dest="shop_ids",
        metavar="ID",
        help="只同步这家店铺（可以给多次）；不给时同步本单元全部有授权的店铺",
    )


def run(
    args: argparse.Namespace,
    environ: Mapping[str, str],
    *,
    service: Any = None,
    out: TextIO | None = None,
) -> int | None:
    """处理 sync-platform-metrics，别的命令返回 None。service / out 给测试注入
    （默认和 serve 一样组装平台单元、输出到标准输出）。"""
    if getattr(args, "command", None) not in COMMANDS:
        return None
    out = out or sys.stdout
    try:
        days = _check_days(args.days)
        service = service if service is not None else build_service(environ)
        return sync_command(service, days=days, shop_ids=args.shop_ids or (), out=out)
    except ValueError as exc:
        logger.error("%s 没有完成：%s", COMMAND, exc)
        return EXIT_BAD_INPUT


def build_service(environ: Mapping[str, str]) -> Any:
    """和 serve 同样组装平台单元（rodit.ops.serve.compose_unit），返回 AssistService。
    测试替换这个函数注入内存单元。"""
    from rodit.app.service import AssistService
    from rodit.ops.serve import compose_unit
    from rodit.ops.settings import ServerSettings

    settings = ServerSettings.from_env(environ)
    return AssistService(compose_unit(settings, environ=environ).unit)


def _check_days(days: object) -> int:
    from rodit.feedback.platform_metrics import MAX_HISTORY_DAYS

    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= MAX_HISTORY_DAYS:
        raise ValueError(f"--days 要在 1–{MAX_HISTORY_DAYS} 之间")
    return days


def sync_command(service: Any, *, days: int, shop_ids: Sequence[str], out: TextIO) -> int:
    from rodit.app.platform_metrics import SKIP_LABELS, PlatformMetricsService
    from rodit.core.errors import AuthExpired, ShopInactive, TenantIsolationError, ToolUnavailable

    unit = service.unit
    repos = unit.repos
    if unit.metrics_source is None:
        raise ValueError("平台绑定没有 metrics_source（平台指标接口），只能在报表页手工登记")
    if repos.platform_metrics is None:
        raise ValueError("本单元没有平台指标存储（Repos.platform_metrics）")
    metrics = PlatformMetricsService(unit, service.processing)
    # 开始的时间：之后（放时钟余量）开始清理的店铺这次不写，见模块说明
    watch_from = unit.clock.now() - PURGE_CLOCK_MARGIN
    authorized = dict.fromkeys(sid for sid, _ in repos.auth.expiring_shop_ids(FAR_FUTURE))
    wanted = list(dict.fromkeys(shop_ids)) if shop_ids else sorted(authorized)
    failed = 0
    for shop_id in wanted:
        shop = repos.knowledge.get_shop(shop_id) if shop_id in authorized else None
        if shop is None:
            if shop_ids:  # 指名要同步的店铺不在：算没同步成
                print(f"{shop_id} 没有授权或不存在，没有同步", file=out)
                failed += 1
            continue
        if shop.status == ShopStatus.SUSPENDED:
            print(f"{shop_id} 已停用（§8.4），跳过", file=out)
            continue
        if _purge_started_since(repos, shop_id, watch_from):
            print(f"{shop_id} 任务期间开始了取消订购 / 清理，这次不写", file=out)
            continue
        try:
            result = metrics.sync(shop_id, actor=OPS_ACTOR, days=days)
        except AuthExpired:
            print(f"{shop_id} 店铺授权无效或缺权限（§8.4），没有调平台接口，跳过", file=out)
            continue
        except ShopInactive:
            print(f"{shop_id} 店铺不存在或已停用（§8.4），跳过", file=out)
            continue
        except ToolUnavailable:
            print(f"{shop_id} 平台指标接口暂时不可用，这家店这次什么都没写", file=out)
            failed += 1
            continue
        except TenantIsolationError:
            print(f"{shop_id} 平台指标接口返回了别的店铺的数据，已拒绝写入（§9.2）", file=out)
            failed += 1
            continue
        skipped = "、".join(f"{SKIP_LABELS[k]} {n}" for k, n in result.skipped.items())
        print(
            f"{shop_id} {result.since.isoformat()} – {result.until.isoformat()}："
            f"拉到 {result.fetched} 条，写入 {result.written} 条"
            + (f"，跳过 {sum(result.skipped.values())} 条（{skipped}）" if skipped else ""),
            file=out,
        )
        if repos.auth.stamp(shop_id) is None:  # 不解密令牌
            # 期间被取消订购：清理先删令牌、后删数据，这次写入可能晚于清理——再删一次（幂等）
            repos.platform_metrics.purge_shop(shop_id)
            repos.audit.purge_shop(shop_id)
            print(f"{shop_id} 同步期间被取消订购，已删除这家店的平台指标和审计", file=out)
        elif _purge_started_since(repos, shop_id, watch_from):
            print(
                f"警告：{shop_id} 同步期间开始了取消订购 / 清理（清理完又重新开通时，这次写入的是"
                "平台给这家店的数，请核对并重新同步一次）",
                file=out,
            )
            logger.warning("sync-platform-metrics：店铺 %s 同步期间开始了清理，请核对", shop_id)
    if not wanted:
        print("本单元没有有授权的店铺", file=out)
    return EXIT_FAILURE if failed else 0


def _purge_started_since(repos: Any, shop_id: str, since: datetime) -> bool:
    """since 之后（写入时间）有没有开始清理这家店（系统事件 shop.purge_started）。"""
    return any(
        e.detail.get("shop_id") == shop_id
        for e in repos.audit.list_system_events(
            action=AuditAction.SHOP_PURGE_STARTED.value, since=since
        )
    )


__all__ = ["COMMAND", "COMMANDS", "build_service", "register", "run", "sync_command"]
