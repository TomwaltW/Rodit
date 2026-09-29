"""部署入口：一个平台单元的 HTTP 服务（方案 §9.1 单元化部署、§12.1–§12.4、§15 Phase 0）。

    python -m rodit.ops.serve [serve]     启动服务（默认命令）
    python -m rodit.ops.serve migrate     迁移数据库（属主账号；幂等，可以每次发布前跑）
    python -m rodit.ops.serve issue-token 签发访问令牌（打印一次原文 + 令牌文件条目）
    python -m rodit.ops.serve local-init  本地试用：迁移 + 写入合成演示数据 + 生成令牌文件

serve 的组装：
    rodit.app.production.build_production_unit（PostgreSQL、境内大模型，审计套上指标采集）
    + 平台绑定（平台云代码注入，见 rodit.ops.bindings；或 RODIT_MODE_C_ONLY=1 只部署模式 C）
    + rodit.api.create_app（鉴权用 TokenFileAuthenticator，令牌哈希来自挂载的密钥文件）
    + 运维接口 /healthz、/readyz、/metrics（rodit.ops.router）+ JSON 日志（rodit.ops.logconfig）。

生产模式（RODIT_ENV=production，默认）下拒绝启动的情况：没有平台绑定又没有明确只部署模式 C；
绑定里有模拟实现（合成数据）；缺少必需的环境变量（只报变量名）；令牌文件读不出来；
/metrics 既没有抓取令牌也没有明确允许公开。
本地试用（RODIT_ENV=local）允许合成数据的平台绑定，没配大模型时用离线替身（rodit.app.offline）。

一个进程一个平台单元，单进程运行：会话串行锁、适配器的去重记录、短时缓存都在进程内
（CLAUDE.md 选型记录：多实例部署时再换 Redis）。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, TextIO

import psycopg
from fastapi import FastAPI

from rodit.api import create_app
from rodit.api.auth import PrincipalRole
from rodit.app.production import build_production_unit, migrate, production_database
from rodit.app.service import AssistService
from rodit.app.unit import Repos, Unit, build_unit
from rodit.core.clock import Clock, SystemClock
from rodit.core.config import EngineConfig, load_config
from rodit.core.enums import ProviderName
from rodit.core.errors import StorageUnavailable
from rodit.core.pii import BuyerRefHasher
from rodit.core.ports import AuditLog
from rodit.llm import MeteredLLMClient
from rodit.ops.bindings import (
    BindingContext,
    BindingError,
    BindingsFactory,
    PlatformBindings,
    demo_bindings,
    load_factory,
    mode_c_bindings,
    validate_bindings,
)
from rodit.ops.health import (
    CheckResult,
    ReadinessChecker,
    SessionDatabase,
    database_probe,
    llm_probe,
    static_probe,
)
from rodit.ops.instrument import (
    AuthExpiryMonitor,
    MeteredToolGateway,
    ObservedAuditLog,
    RoditMetrics,
)
from rodit.ops.logconfig import configure_logging
from rodit.ops.router import MetricsGuard, install_ops
from rodit.ops.settings import ServerSettings, SettingsError
from rodit.ops.tokens import (
    IssuedToken,
    MetricsEntry,
    PrincipalEntry,
    TokenFile,
    TokenFileAuthenticator,
    TokenFileError,
    append_entries,
    issue_metrics_token,
    issue_principal_token,
    write_token_file,
)
from rodit.storage import APP_ROLE, Database

logger = logging.getLogger("rodit.ops")

LLM_ENV_PREFIX = "RODIT_LLM_"
MIGRATE_DSN_ENV = "RODIT_MIGRATE_DSN"
DATABASE_DSN_ENV = "RODIT_DATABASE_DSN"
TOKEN_KEY_ENV = "RODIT_TOKEN_KEY"
EXIT_FAILURE = 1
EXIT_STARTUP = 2


class StartupError(RuntimeError):
    """启动前检查不通过。消息里只有变量名、文件路径和原因，没有任何值。"""


@dataclass
class Server:
    """组装好的服务。app 交给 uvicorn；其余字段给测试和运维命令用。"""

    settings: ServerSettings
    app: FastAPI
    service: AssistService
    unit: Unit
    bindings: PlatformBindings
    metrics: RoditMetrics
    readiness: ReadinessChecker
    authenticator: TokenFileAuthenticator
    auth_monitor: AuthExpiryMonitor | None


# ================================================================ 启动前检查


def _llm_configured(config: EngineConfig, env: Mapping[str, str]) -> bool:
    base_url = config.llm.base_url or env.get(f"{LLM_ENV_PREFIX}BASE_URL", "")
    model = config.llm.model or env.get(f"{LLM_ENV_PREFIX}MODEL", "")
    return bool(base_url.strip() and model.strip())


def preflight(
    settings: ServerSettings, config: EngineConfig, environ: Mapping[str, str] | None = None
) -> None:
    """必需的环境变量都有值（只看有没有，报错只列变量名）。大模型密钥缺失不在这里拦：
    /readyz 报 missing_api_key，进程不接流量，日志里看得到原因。"""
    env = os.environ if environ is None else environ
    required = [DATABASE_DSN_ENV, TOKEN_KEY_ENV, config.privacy.buyer_salt_env]
    missing = [name for name in required if not env.get(name, "").strip()]
    if settings.is_production and not _llm_configured(config, env):
        missing.append(f"{LLM_ENV_PREFIX}BASE_URL / {LLM_ENV_PREFIX}MODEL（或配置文件 llm 段）")
    if missing:
        raise StartupError("缺少必需的环境变量：" + "、".join(missing))


def resolve_bindings(
    settings: ServerSettings, ctx: BindingContext, *, factory: BindingsFactory | None = None
) -> PlatformBindings:
    """平台绑定从哪来（见 rodit.ops.bindings）。生产模式下两样都没有就拒绝启动。"""
    if factory is None and settings.bindings:
        factory = load_factory(settings.bindings)
    if factory is not None and settings.mode_c_only:
        raise StartupError(
            "RODIT_MODE_C_ONLY=1 和平台绑定（RODIT_PLATFORM_BINDINGS）不能同时设置："
            "要同时提供模式 C，请在平台绑定里加上 StandaloneAssistAdapter"
        )
    if factory is not None:
        bindings = factory(ctx)
    elif settings.mode_c_only:
        bindings = mode_c_bindings(ctx)
    elif not settings.is_production:
        bindings = demo_bindings(ctx)
    else:
        raise StartupError(
            "生产模式必须由平台云代码提供平台只读工具、接入适配器（RODIT_PLATFORM_BINDINGS="
            "包.模块:函数）；只部署模式 C（Phase 0）时明确设置 RODIT_MODE_C_ONLY=1"
        )
    return validate_bindings(bindings, settings.platform, allow_mock=not settings.is_production)


# ================================================================ 组装


@dataclass
class _Composed:
    unit: Unit
    bindings: PlatformBindings
    metrics: RoditMetrics
    llm_probe: Callable[[], Mapping[str, CheckResult]]


def compose_unit(
    settings: ServerSettings,
    *,
    bindings_factory: BindingsFactory | None = None,
    clock: Clock | None = None,
    repos: Repos | None = None,
    environ: Mapping[str, str] | None = None,
) -> _Composed:
    """平台单元 + 指标采集（不含 HTTP 应用）。

    repos 只在本地试用、没配大模型（离线替身）时生效（测试用内存仓储）；
    其他情况一律 PostgreSQL（build_production_unit）。"""
    env = os.environ if environ is None else environ
    config = load_config(settings.config_path or None)
    preflight(settings, config, env)
    clock = clock or SystemClock()
    hasher = BuyerRefHasher.from_env(config.privacy.buyer_salt_env)
    ctx = BindingContext(platform=settings.platform, config=config, clock=clock, hasher=hasher)
    bindings = resolve_bindings(settings, ctx, factory=bindings_factory)
    metrics = RoditMetrics(platform=settings.platform, mode=settings.deploy_mode)
    tools = MeteredToolGateway(bindings.tools, metrics)

    def observe(audit: AuditLog) -> AuditLog:
        return ObservedAuditLog(audit, metrics)

    if settings.is_production or _llm_configured(config, env):
        unit = build_production_unit(
            settings.platform,
            config=config,
            tools=tools,
            adapters=list(bindings.adapters),
            action_port=bindings.action_port,
            clock=clock,
            audit_wrapper=observe,
        )
        probe = llm_probe(config.llm, environ=env, env_prefix=LLM_ENV_PREFIX)
    else:
        unit = _offline_unit(settings, config, clock, hasher, bindings, tools, observe, repos)
        probe = static_probe("llm", CheckResult(True, {"offline": True}))
    return _Composed(unit=unit, bindings=bindings, metrics=metrics, llm_probe=probe)


def _offline_unit(
    settings: ServerSettings,
    config: EngineConfig,
    clock: Clock,
    hasher: BuyerRefHasher,
    bindings: PlatformBindings,
    tools: MeteredToolGateway,
    observe: Callable[[AuditLog], AuditLog],
    repos: Repos | None,
) -> Unit:
    """本地试用、没配大模型：离线替身（关键词路由、复述依据、启发式质检），仓储默认 PostgreSQL。"""
    from rodit.app.offline import EvidenceEchoLLM, HeuristicQAJudge
    from rodit.app.production import postgres_repos
    from rodit.decision import RuleProvider

    logger.warning("本地试用：没有配置大模型，使用离线替身（回答只复述依据，仅供试用）")
    base = repos if repos is not None else postgres_repos(production_database(), clock=clock)
    repos = replace(base, audit=observe(base.audit))
    config = config.model_copy(
        update={"decision": config.decision.model_copy(update={"provider": ProviderName.RULES})}
    )
    return build_unit(
        platform=settings.platform,
        config=config,
        clock=clock,
        hasher=hasher,
        repos=repos,
        llm=MeteredLLMClient(EvidenceEchoLLM(), repos.audit, clock),
        tools=tools,
        adapters=list(bindings.adapters),
        action_port=bindings.action_port,
        route_provider=RuleProvider(),
        qa_provider=HeuristicQAJudge(),
    )


def _metrics_guard(settings: ServerSettings, auth: TokenFileAuthenticator) -> MetricsGuard | None:
    if settings.metrics_public:
        return None
    if auth.has_metrics_tokens:
        return auth.authenticate_metrics
    if not settings.is_production:
        return None
    raise StartupError(
        "生产模式下 /metrics 要抓取令牌：用 issue-token --metrics 签发并写进令牌文件的 metrics 段；"
        "网络策略已经限制了访问时可以设 RODIT_METRICS_PUBLIC=1"
    )


def build_server(
    settings: ServerSettings,
    *,
    bindings_factory: BindingsFactory | None = None,
    clock: Clock | None = None,
    repos: Repos | None = None,
    database: SessionDatabase | None = None,
    environ: Mapping[str, str] | None = None,
) -> Server:
    """组装完整的服务（见模块说明）。database 覆盖就绪检查用的数据库（测试用）。"""
    composed = compose_unit(
        settings, bindings_factory=bindings_factory, clock=clock, repos=repos, environ=environ
    )
    unit, metrics = composed.unit, composed.metrics
    authenticator = TokenFileAuthenticator(
        settings.tokens_file, reload_interval_s=settings.tokens_reload_s, clock=unit.clock
    )
    guard = _metrics_guard(settings, authenticator)
    readiness = ReadinessChecker(
        {
            "database": database_probe(database or production_database()),
            "llm": composed.llm_probe,
        },
        cache_s=settings.readiness_cache_s,
    )
    monitor: AuthExpiryMonitor | None = None
    if not composed.bindings.mode_c_only:  # 模式 C 不需要平台授权（§8.1），不告警
        platform_shops = composed.bindings.shop_ids

        def shop_ids() -> list[str]:
            shops = set(authenticator.shop_ids())
            if platform_shops is not None:
                shops.update(platform_shops())
            return sorted(shops)

        monitor = AuthExpiryMonitor(
            unit.auth,
            shop_ids,
            metrics,
            within=timedelta(days=settings.auth_expiry_warn_days),
            refresh_s=settings.auth_expiry_check_s,
            clock=unit.clock,
        )
        metrics.registry.before_render(monitor.refresh)
    service = AssistService(unit)
    app = create_app(service, authenticator=authenticator)
    install_ops(app, readiness=readiness, metrics=metrics, metrics_guard=guard)
    if not settings.is_production:
        logger.warning("本地试用模式（RODIT_ENV=local）：合成数据、模拟适配器，不能用于生产")
    access = sorted(str(getattr(a, "access", "?")) for a in composed.bindings.adapters)
    logger.info(
        "平台单元已组装：platform=%s mode=%s access=%s",
        settings.platform.value,
        settings.deploy_mode,
        ",".join(access),
    )
    return Server(
        settings=settings,
        app=app,
        service=service,
        unit=unit,
        bindings=composed.bindings,
        metrics=metrics,
        readiness=readiness,
        authenticator=authenticator,
        auth_monitor=monitor,
    )


# ================================================================ 迁移


def grant_readiness_access(db: Database, *, role: str = APP_ROLE) -> None:
    """让应用角色能读 schema_migrations（/readyz 核对迁移版本用；只读、幂等）。"""
    from psycopg import sql

    with db.session(None) as conn:
        conn.execute(
            sql.SQL("GRANT SELECT ON schema_migrations TO {}").format(sql.Identifier(role))
        )


def migrate_database() -> list[int]:
    """迁移（属主账号）：RODIT_MIGRATE_DSN，没设就用 RODIT_DATABASE_DSN。返回本次应用的版本号。
    连接串由 rodit.app.production.migrate 从环境变量读，这里只挑变量名。"""
    env = os.environ
    dsn_env = MIGRATE_DSN_ENV if env.get(MIGRATE_DSN_ENV, "").strip() else DATABASE_DSN_ENV
    if not env.get(dsn_env, "").strip():
        raise StartupError(f"缺少必需的环境变量：{MIGRATE_DSN_ENV} 或 {DATABASE_DSN_ENV}")
    applied = migrate(dsn_env)
    grant_readiness_access(Database.from_env(dsn_env, app_role=None))
    logger.info("数据库迁移完成：本次应用 %s", applied or "无（已是最新）")
    return applied


# ================================================================ 令牌


def _print_token(out: TextIO, issued: IssuedToken) -> None:
    entry = issued.entry
    if isinstance(entry, PrincipalEntry):
        who = f"{entry.role:<11} 店铺 {entry.shop_id}  账号 {entry.user_id}"
    else:
        who = f"{'metrics':<11} {entry.name}"
    print(f"  {who}  Bearer {issued.token}", file=out)


def issue_token_command(args: argparse.Namespace, out: TextIO) -> int:
    expires = datetime.now(UTC) + timedelta(days=args.expires_days) if args.expires_days else None
    if args.metrics:
        issued = issue_metrics_token(args.name or "prometheus", expires_at=expires)
    else:
        if not (args.shop_id and args.user_id and args.role):
            print("issue-token 需要 --shop-id、--user-id、--role（或 --metrics）", file=sys.stderr)
            return EXIT_STARTUP
        issued = issue_principal_token(args.shop_id, args.user_id, args.role, expires_at=expires)
    print("令牌原文只显示这一次，请直接交给使用方；令牌文件里只存它的哈希：", file=out)
    _print_token(out, issued)
    entry_json = issued.entry.model_dump_json(exclude_none=True)
    if args.append:
        append_entries(args.append, [issued.entry])
        print(f"已写入令牌文件：{args.append}", file=out)
    else:
        section = "metrics" if args.metrics else "principals"
        print(f"加进令牌文件的 {section} 段：{entry_json}", file=out)
    return 0


# ================================================================ 本地试用


LOCAL_USERS: tuple[tuple[str, PrincipalRole], ...] = (
    ("kf_02", "agent"),
    ("kf_manager", "manager"),
    ("owner_demo", "merchant"),
)


def local_init(settings: ServerSettings, out: TextIO, *, clock: Clock | None = None) -> int:
    """本地试用的初始化（只能在 RODIT_ENV=local 下跑）：迁移、写入合成演示数据、生成令牌文件。
    可以重复执行：店铺已经存在就不再写数据，令牌文件已经存在就不重新生成。"""
    from rodit.app.demo import seed_unit
    from rodit.testing.fixtures import build_demo_world

    if settings.is_production:
        raise StartupError("local-init 只用于本地试用（RODIT_ENV=local），不能在生产环境运行")
    migrate_database()
    composed = compose_unit(settings, clock=clock)
    unit = composed.unit
    world = build_demo_world(unit.clock.now())
    shops = sorted(s.id for s in world.shops.values() if s.platform == settings.platform)
    if any(unit.repos.knowledge.get_shop(s) is not None for s in shops):
        print(f"合成演示数据已存在（{', '.join(shops)}），跳过写入", file=out)
    else:
        seed_unit(unit, world)
        print(f"已写入合成演示数据：{settings.platform.value} 店铺 {', '.join(shops)}", file=out)
    if os.path.exists(settings.tokens_file):
        print(
            f"令牌文件已存在（{settings.tokens_file}），不重新生成；"
            "要重新生成请删除它（compose：docker compose down -v）",
            file=out,
        )
        return 0
    has_platform_events = any(
        getattr(a, "access", None) in {"A", "B"} for a in composed.bindings.adapters
    )
    issued: list[IssuedToken] = []
    for shop_id in shops:
        for user_id, role in LOCAL_USERS:
            issued.append(issue_principal_token(shop_id, user_id, role))
        if has_platform_events:  # 模式 A / B 的平台事件接口只给平台对接角色用
            push_user = f"{settings.platform.value}_push"
            issued.append(issue_principal_token(shop_id, push_user, "integration"))
    issued.append(issue_metrics_token("local"))
    write_token_file(
        settings.tokens_file,
        TokenFile(
            principals=[t.entry for t in issued if isinstance(t.entry, PrincipalEntry)],
            metrics=[t.entry for t in issued if isinstance(t.entry, MetricsEntry)],
        ),
    )
    print("【仅本地试用】合成数据；令牌原文只显示这一次（令牌文件里只存哈希）：", file=out)
    for token in issued:
        _print_token(out, token)
    return 0


# ================================================================ 容器健康检查


def healthcheck(
    environ: Mapping[str, str] | None = None,
    *,
    opener: Callable[..., Any] | None = None,
) -> int:
    """请求本机 http://127.0.0.1:$RODIT_PORT/healthz：200 返回 0，否则 1（Dockerfile HEALTHCHECK）。
    只查存活；就绪由编排系统的就绪探针查 /readyz。"""
    import urllib.request

    env = os.environ if environ is None else environ
    port = env.get("RODIT_PORT", "").strip() or "8000"
    if not port.isdigit():
        return EXIT_FAILURE
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(f"http://127.0.0.1:{port}/healthz", timeout=2) as resp:
            return 0 if getattr(resp, "status", None) == 200 else EXIT_FAILURE
    except Exception:  # noqa: BLE001
        return EXIT_FAILURE


# ================================================================ 命令行


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rodit.ops.serve", description="Rodit 平台单元的部署入口"
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="启动服务（默认）")
    sub.add_parser("migrate", help="迁移数据库（属主账号，幂等）")
    sub.add_parser("local-init", help="本地试用：迁移 + 合成演示数据 + 令牌文件")
    sub.add_parser("healthcheck", help="容器健康检查：请求本机 /healthz（HEALTHCHECK 用）")
    issue = sub.add_parser("issue-token", help="签发访问令牌")
    issue.add_argument("--shop-id")
    issue.add_argument("--user-id")
    issue.add_argument("--role", choices=["agent", "manager", "merchant", "integration"])
    issue.add_argument("--metrics", action="store_true", help="签发 /metrics 的抓取令牌")
    issue.add_argument("--name", help="抓取令牌的名字（--metrics 时）")
    issue.add_argument("--expires-days", type=float, default=0, help="有效天数（0 = 不过期）")
    issue.add_argument("--append", metavar="FILE", help="直接写进这个令牌文件（没有就新建）")
    return parser


def _early_logging(environ: Mapping[str, str]) -> None:
    """设置还没解析时先按环境变量配好日志（值不合法时退回默认），启动失败也有结构化日志。"""
    level = environ.get("RODIT_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    fmt = environ.get("RODIT_LOG_FORMAT", "json").strip().lower()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        level = "INFO"
    configure_logging(level, fmt if fmt in {"json", "text"} else "json")


def main(
    argv: Sequence[str] | None = None,
    *,
    run: Callable[..., Any] | None = None,
    out: TextIO | None = None,
) -> int:
    """命令行入口。run 默认是 uvicorn.run（测试注入假的）。返回进程退出码。"""
    out = out or sys.stdout
    args = _parser().parse_args(list(argv) if argv is not None else None)
    command = args.command or "serve"
    if command == "healthcheck":
        return healthcheck()
    _early_logging(os.environ)
    if command == "issue-token":
        return issue_token_command(args, out)
    try:
        if command == "migrate":
            migrate_database()
            return 0
        settings = ServerSettings.from_env()
        configure_logging(settings.log_level, settings.log_format)
        if command == "local-init":
            return local_init(settings, out)
        if settings.migrate_on_start:
            migrate_database()
        server = build_server(settings)
    except (SettingsError, StartupError, BindingError, TokenFileError) as exc:
        logger.error("启动失败：%s", exc)
        return EXIT_STARTUP
    except RuntimeError as exc:  # 其他模块缺环境变量时的报错（只含变量名）
        logger.error("启动失败：%s", exc)
        return EXIT_STARTUP
    except (StorageUnavailable, psycopg.Error) as exc:  # 只记类型：说明里可能有主机名、账号
        logger.error("数据库不可用：%s", type(exc).__name__)
        return EXIT_FAILURE
    if run is None:
        import uvicorn

        run = uvicorn.run
    run(
        server.app,
        host=settings.host,
        port=settings.port,
        log_config=None,  # 用 configure_logging 的 JSON 日志和过滤
        server_header=False,
        timeout_graceful_shutdown=20,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
