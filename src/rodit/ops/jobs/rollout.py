"""上线节奏的离线任务（方案 §13.4 阶段门槛、§13.5 上线前回归、§15 Phase 1 验收）。在平台单元内跑，
只读业务数据；record-eval、compliance-scan 只写审计。约定见 rodit.ops.jobs。

    record-eval --report 报告.json [--config YAML]
        把 python -m rodit.eval --json 的完整评测报告记成 eval.verdict 系统事件（§13.5；
        rodit.eval.fingerprint）。部分评测（--intent / --limit）、没有测试集判定、没有配置指纹、
        条数和评测集对不上（没跑完）的报告不收（退出码 2，什么都不写）。真实大模型跑的报告还要
        "评到了模型"：按报告里的运行健康度（rodit.eval.health）用固定的上限重新判定，路由退回
        关键词规则、质检判断层不可用、生成没有可用草稿……超过上限的不收，原因逐条打出来（退出码
        同样是 2）；收下的结论带着这些计数，记录时打出来（全部用例和测试集）。
        顺带说明报告的指纹和线上（部署配置：--config，其次
        RODIT_CONFIG，都没有时是内置默认值；大模型、分类器按部署规则取配置和环境变量
        RODIT_LLM_BASE_URL / RODIT_LLM_MODEL / RODIT_CLASSIFIER_PATH）一致不一致、哪几项不一致：
        上线状态只认一致的、真实大模型跑的结论。
    acceptance-report --since ISO [--until ISO] [--json] [--config YAML]
        Phase 1 验收报告（feedback.acceptance，§15）：每家店一份 + 合在一起一份。店铺来自
        ShopAuthRepository.expiring_shop_ids(很远的将来)——本单元有授权的店铺（只有 ID 和到期时间，
        不读令牌）；取消订购后删掉的店铺跳过。--json 输出 JSON。
        退出码：0 合在一起的条件都通过；1 有条件没通过或缺数据（文本最后一行写明哪几条和取值）。
    compliance-scan [--week 2026-W39] [--config YAML]
        扫描一周实际发出去的文字（默认上一个完整周；feedback.compliance），每家店写一条
        compliance.flag 审计：计数、规则名、推荐 / 消息 ID，没有文字。命中的是疑似违规（候选），
        由复核人员逐条确认或标记误报（POST rollout/compliance-reviews），确认的才算门槛不通过。
        不拿租约、和 serve 同时跑：扫描开始之后开始清理（shop.purge_started）的店铺这次不写
        （之后重新扫描这一周），写完才发现令牌不在了的删掉这次的写入，写完才发现开始清理、令牌
        还在的打警告（清理完又重新开通时不能整家店删审计，请核对、重新扫描）。

数据库：RODIT_DATABASE_DSN（应用角色，行级安全按店铺隔离；系统事件在不设店铺范围的事务里读写）；
acceptance-report、compliance-scan 还要 RODIT_TOKEN_KEY（仓储组装需要，不解密任何令牌）。
record-eval、acceptance-report 判断测试集依据时要和线上服务同样的大模型、分类器环境变量
（RODIT_LLM_BASE_URL / RODIT_LLM_MODEL，配置里没写 llm.base_url 和 llm.model 时；
decision.provider 是 classifier 时 RODIT_CLASSIFIER_PATH）：不知道线上用的是什么，结论就对不上。
配置、报告文件、参数不对时退出码 2（日志写明原因）；数据库错误按 rodit.ops.jobs 的约定。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

import yaml

from rodit.core.audit import AuditAction
from rodit.core.clock import CST, Clock, SystemClock
from rodit.core.config import EngineConfig, config_fingerprint, load_config
from rodit.core.models import AuditEvent
from rodit.core.ports import AuditLog

logger = logging.getLogger("rodit.ops")

RECORD_EVAL = "record-eval"
ACCEPTANCE_REPORT = "acceptance-report"
COMPLIANCE_SCAN = "compliance-scan"
COMMANDS = (RECORD_EVAL, ACCEPTANCE_REPORT, COMPLIANCE_SCAN)
EXIT_NOT_PASSED = 1
EXIT_BAD_INPUT = 2
FAR_FUTURE = datetime(9999, 12, 31, tzinfo=UTC)
"""expiring_shop_ids 给一个很远的时间：本单元全部有授权的店铺。"""
SYSTEM_ACTOR = "system"
PURGE_CLOCK_MARGIN = timedelta(minutes=5)
"""compliance-scan 找"扫描期间开始的清理"时往前放的余量：开始清理的标记 shop.purge_started 按 serve
的时钟写，扫描开始的时间按本任务的时钟取，两台机器的时钟差在这之内（也盖住清理写了标记之后、
删令牌之前等在途处理的最多 30 秒）。"""


def _iso_time(text: str) -> datetime:
    """ISO 8601 时间；没写时区的按东八区。"""
    try:
        value = datetime.fromisoformat(text.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            "应为 ISO 8601 时间，如 2026-09-01 或 2026-09-01T00:00+08:00"
        ) from None
    return value if value.tzinfo is not None else value.replace(tzinfo=CST)


def register(subparsers: Any) -> None:
    config_help = "部署配置（默认 RODIT_CONFIG，都没有时用内置默认值）"
    rec = subparsers.add_parser(RECORD_EVAL, help="§13.5 把完整评测的结论记成 eval.verdict")
    rec.add_argument(
        "--report", required=True, metavar="FILE", help="python -m rodit.eval --json 的输出"
    )
    rec.add_argument("--config", metavar="YAML", help=config_help)
    acc = subparsers.add_parser(
        ACCEPTANCE_REPORT, help="§15 Phase 1 验收报告（退出码 0 通过，1 没通过或缺数据）"
    )
    acc.add_argument("--since", required=True, type=_iso_time, help="窗口起点（含）")
    acc.add_argument("--until", type=_iso_time, help="窗口终点（不含；默认现在）")
    acc.add_argument("--json", action="store_true", help="输出 JSON")
    acc.add_argument("--config", metavar="YAML", help=config_help)
    scan = subparsers.add_parser(
        COMPLIANCE_SCAN, help="§13.4 扫描一周实际发出去的文字，写 compliance.flag"
    )
    scan.add_argument("--week", metavar="ISO周", help="如 2026-W39（默认上一个完整周）")
    scan.add_argument("--config", metavar="YAML", help=config_help)


def _config(args: argparse.Namespace, environ: Mapping[str, str]) -> tuple[EngineConfig, str]:
    path = (getattr(args, "config", None) or environ.get("RODIT_CONFIG", "")).strip()
    if path:
        return load_config(path), path
    return load_config(None), "内置默认值（没有 --config，也没有 RODIT_CONFIG）"


def _pg_audit() -> AuditLog:
    from rodit.app.production import production_database
    from rodit.storage import PgAuditLog

    return PgAuditLog(production_database())


def _pg_repos(clock: Clock) -> Any:
    from rodit.app.production import postgres_repos, production_database

    return postgres_repos(production_database(), clock=clock)


def run(
    args: argparse.Namespace,
    environ: Mapping[str, str],
    *,
    repos: Any = None,
    audit: AuditLog | None = None,
    clock: Clock | None = None,
    out: TextIO | None = None,
) -> int | None:
    """处理 record-eval / acceptance-report / compliance-scan，别的命令返回 None。

    repos / audit / clock / out 给测试注入（默认连 RODIT_DATABASE_DSN、系统时间、标准输出）。"""
    command = getattr(args, "command", None)
    if command not in COMMANDS:
        return None
    out = out or sys.stdout
    clock = clock or SystemClock()
    try:
        config, source = _config(args, environ)
        if command == RECORD_EVAL:
            log = audit or (repos.audit if repos is not None else _pg_audit())
            return record_eval(
                args, out, audit=log, clock=clock, config=config, source=source, environ=environ
            )
        repos = repos if repos is not None else _pg_repos(clock)
        if command == ACCEPTANCE_REPORT:
            return acceptance_command(
                args, out, repos=repos, clock=clock, config=config, environ=environ
            )
        return compliance_command(args, out, repos=repos, clock=clock, config=config)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        logger.error("%s 没有完成：%s", command, exc)
        return EXIT_BAD_INPUT


# ================================================================ record-eval（§13.5）


def record_eval(
    args: argparse.Namespace,
    out: TextIO,
    *,
    audit: AuditLog,
    clock: Clock,
    config: EngineConfig,
    source: str,
    environ: Mapping[str, str] | None = None,
) -> int:
    from rodit.eval.fingerprint import (
        ModelNotEvaluated,
        current_fingerprint,
        describe_health,
        describe_verdict,
        differences,
        verdict_event,
        verdict_from_report,
    )

    path = Path(args.report).expanduser()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"读不了评测报告 {path}：{type(exc).__name__}") from None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError(f"评测报告 {path} 不是 JSON") from None
    try:
        verdict = verdict_from_report(data, raw=raw)
    except ModelNotEvaluated as exc:
        # 原因逐条打出来（日志里也有一份，是连成一句的）；照旧由 run 记日志、返回退出码 2
        print("没有记录评测结论：这一轮没有有效评到大模型（量到的是降级后的兜底逻辑）", file=out)
        for reason in exc.reasons:
            print(f"- {reason}", file=out)
        raise
    at = clock.now()
    audit.record(verdict_event(verdict, at=at, actor=SYSTEM_ACTOR))
    fp = verdict.fingerprint
    code = fp.git_sha[:12] if fp.git_sha else "未知"
    print(f"已记录评测结论（eval.verdict）：{describe_verdict(at, verdict)}", file=out)
    if verdict.health_test is not None:
        test = verdict.health_test
        print(f"其中测试集 {test.cases} 条：{describe_health(test)}", file=out)
    model = fp.llm_model or "未知"
    print(
        f"配置指纹 {fp.config_fingerprint[:12]}，路由问题 {fp.router_questions_ver}，"
        f"质检问题 {fp.qa_questions_ver}，生成环节 {fp.generation_ver or '未知'}，"
        f"判断层提示词 {fp.judge_prompt_ver or '未知'}，"
        f"判断层 {fp.decision_provider or '未知'}，大模型 {model}，代码 {code}",
        file=out,
    )
    diffs = differences(fp, current_fingerprint(config, environ=environ))
    if verdict.llm_mode != "env":
        print("离线替身跑的评测只作回归基线，上线状态不会用它判断测试集门槛", file=out)
    elif diffs:
        print(
            f"注意：和线上（部署配置 {source}）不一致（{'、'.join(diffs)}），"
            "上线状态不会用这次结论判断测试集门槛",
            file=out,
        )
    else:
        print(f"和线上（部署配置 {source}）一致：上线状态用这次结论判断测试集门槛", file=out)
    return 0


# ================================================================ acceptance-report（§15）


def _shops(repos: Any) -> list[Any]:
    """本单元有授权、还在的店铺（按到期时间、店铺 ID 的顺序去重）。"""
    shops = []
    for shop_id in dict.fromkeys(sid for sid, _ in repos.auth.expiring_shop_ids(FAR_FUTURE)):
        shop = repos.knowledge.get_shop(shop_id)
        if shop is not None:
            shops.append(shop)
    return sorted(shops, key=lambda s: s.id)


def acceptance_command(
    args: argparse.Namespace,
    out: TextIO,
    *,
    repos: Any,
    clock: Clock,
    config: EngineConfig,
    environ: Mapping[str, str] | None = None,
) -> int:
    from rodit.app.rollout import acceptance_input, eval_evidence
    from rodit.feedback.acceptance import acceptance_report, render_text

    now = clock.now()
    until = args.until or now
    since = args.since
    if since >= until:
        raise ValueError("--since 要早于 --until（默认现在）")
    inputs = [
        acceptance_input(repos, config, shop, since=since, until=until) for shop in _shops(repos)
    ]
    report = acceptance_report(
        inputs,
        test_set=eval_evidence(repos.audit, config, environ=environ),
        since=since,
        until=until,
        now=now,
        running_fingerprint=config_fingerprint(config),
    )
    if args.json:
        print(report.model_dump_json(indent=2), file=out)
    else:
        print(render_text(report), file=out)
    return 0 if report.passed else EXIT_NOT_PASSED


# ================================================================ compliance-scan（§13.4）


def compliance_command(
    args: argparse.Namespace, out: TextIO, *, repos: Any, clock: Clock, config: EngineConfig
) -> int:
    from rodit.app.rollout import compliance_weeks
    from rodit.feedback.compliance import flag_detail
    from rodit.feedback.review import check_period, period_range, previous_period

    now = clock.now()
    period = check_period(args.week) if args.week else previous_period(now)
    start, end = period_range(period)
    if start >= now:
        raise ValueError(f"{period} 还没开始，不能扫描")
    until = min(end, now)
    # 本任务不拿租约、和 serve 不共用闸门：扫描期间 serve 可能取消订购 / 清理某家店（清理先写开始
    # 清理的标记，再删令牌、删数据），甚至清理完之后又重新开通。每家店写之前、写之后各确认一次
    # 列出店铺之后（按开始扫描的时间，放时钟余量）有没有开始清理它。按整次扫描的开始时间、不按
    # 轮到这家店的时间：一家大店的清理可能跑好几分钟，轮到它时还在删、读到的是删了一半的数据；
    # 在这之前开始的清理，删令牌只要等在途处理（最多 30 秒），令牌删了就列不出这家店
    watch_from = now - PURGE_CLOCK_MARGIN
    shops = _shops(repos)
    for shop in shops:
        weeks = compliance_weeks(repos, config, shop, since=start, until=until)
        if weeks and _purge_started_since(repos, shop.id, watch_from):
            # 读到的可能是清理之前的数据（清理完又重新开通时就混进了新订购）：这次不写
            print(
                f"{shop.id} 扫描期间开始了取消订购 / 清理：这次不写记录（之后重新扫描这一周）",
                file=out,
            )
            continue
        for week in weeks:
            repos.audit.record(
                AuditEvent(
                    shop_id=shop.id,
                    actor=SYSTEM_ACTOR,
                    action=AuditAction.COMPLIANCE_FLAG.value,
                    detail=flag_detail(week),
                    created_at=now,
                )
            )
            rules = "、".join(f"{r} {n}" for r, n in week.rule_counts.items())
            print(
                f"{shop.id} {week.period}：编辑后发送的推荐 {week.scanned_suggestions} 条、"
                f"发出的消息 {week.scanned_messages} 条，疑似违规 {week.flagged} 条"
                + (f"（{rules}；等复核人员确认或标记误报）" if rules else ""),
                file=out,
            )
        if not weeks:
            continue
        if repos.auth.stamp(shop.id) is None:  # 不解密令牌
            # 扫描期间 serve 取消订购 / 过期清理了这家店（本任务和 serve 不共用闸门）：清理先删
            # 令牌、后删审计，令牌不在了说明这里的写入可能晚于清理——再删一次这家店的审计，
            # 不留下迟到的记录（§8.4；purge_shop 是幂等的）
            repos.audit.purge_shop(shop.id)
            print(f"{shop.id} 扫描期间被取消订购，已删除这次写入的记录", file=out)
        elif _purge_started_since(repos, shop.id, watch_from):
            # 写之前的确认之后才开始清理（写之前、写之后两次确认之间的很小的窗口；serve 和
            # 本任务的时钟差超过 PURGE_CLOCK_MARGIN 时也可能漏掉），令牌还在：要么清理还没删到
            # 令牌（它之后删审计，连这次的写入一起删掉），要么已经清理完又重新开通——这次写的
            # 记录可能来自已删除的数据，但不能整家店删审计（会删掉新订购的记录），只能提示
            # 核对、重新扫描
            print(
                f"警告：{shop.id} 扫描期间开始了取消订购 / 清理（清理完又重新开通时，这次写入的 "
                "compliance.flag 可能来自已删除的数据），请核对并重新扫描这一周",
                file=out,
            )
            logger.warning(
                "compliance-scan：店铺 %s 扫描期间开始了清理，这次的记录请核对、重新扫描",
                shop.id,
            )
    if not shops:
        print("本单元没有有授权的店铺", file=out)
    return 0


def _purge_started_since(repos: Any, shop_id: str, since: datetime) -> bool:
    """since 之后（写入时间）有没有开始清理这家店（系统事件 shop.purge_started）。"""
    return any(
        e.detail.get("shop_id") == shop_id
        for e in repos.audit.list_system_events(
            action=AuditAction.SHOP_PURGE_STARTED.value, since=since
        )
    )


__all__ = [
    "ACCEPTANCE_REPORT",
    "COMMANDS",
    "COMPLIANCE_SCAN",
    "RECORD_EVAL",
    "acceptance_command",
    "compliance_command",
    "record_eval",
    "register",
    "run",
]
