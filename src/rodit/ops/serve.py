"""部署入口：一个平台单元的 HTTP 服务（方案 §9.1 单元化部署、§12.1–§12.4、§15 Phase 0）。

    python -m rodit.ops.serve [serve]     启动服务（默认命令）
    python -m rodit.ops.serve migrate     迁移数据库（属主账号；幂等，可以每次发布前跑）
    python -m rodit.ops.serve issue-token 签发访问令牌（打印一次原文 + 令牌文件条目）
    python -m rodit.ops.serve local-init  本地试用：迁移 + 写入合成演示数据 + 生成令牌文件
    python -m rodit.ops.serve recalibrate 每月用采纳反馈重算阈值（§13.5，只输出建议）
    python -m rodit.ops.serve train-classifier --out FILE  训练路由分类器（§6.4 Phase 2）
    其余子命令由 rodit.ops.jobs 注册表里的任务模块提供（店铺生命周期、上线节奏）。

serve 的组装（顺序：平台绑定 → 平台单元 → 接上 ctx.tokens → 应用服务 → 接上 ctx.on_subscription）：
    rodit.app.production.build_production_unit（PostgreSQL、境内大模型，审计套上指标采集）
    + 平台绑定（平台云代码注入，见 rodit.ops.bindings；或 RODIT_MODE_C_ONLY=1 只部署模式 C）；
      工厂拿到的 ctx.tokens / ctx.on_subscription 是代理，单元和服务组装好之后才接上
    + rodit.api.create_app（鉴权用 TokenFileAuthenticator，令牌哈希来自挂载的密钥文件）
    + 运维接口 /healthz、/readyz、/metrics（rodit.ops.router）+ JSON 日志（rodit.ops.logconfig）。

生产模式（RODIT_ENV=production，默认）下拒绝启动的情况：没有平台绑定又没有明确只部署模式 C；
绑定里有模拟实现（合成数据）；缺少必需的环境变量（只报变量名）；令牌文件读不出来；
/metrics 既没有抓取令牌也没有明确允许公开；配置要求公示大模型（llm.require_disclosure）却没填
llm.public_name / llm.filing_no（§12.3 生成式人工智能服务要公示所用模型的名称和备案号）。
本地试用（RODIT_ENV=local）允许合成数据的平台绑定，没配大模型时用离线替身（rodit.app.offline）。
平台绑定给了 llm_transport（平台要求经它报备的通道调用大模型）时，大模型客户端经它发请求，
计量（llm.usage）和密钥缺失时的就绪语义不变。

decision.provider 是 classifier 时，serve 从 RODIT_CLASSIFIER_PATH 加载路由分类器的模型文件
（train-classifier 生成）当路由判断层，质检仍用大模型（§6.4）；没设、文件不存在或校验不过都
拒绝启动（退出码 2），日志里写明原因。

一个进程一个平台单元，单进程运行：会话串行锁、适配器的去重记录、短时缓存都在进程内
（CLAUDE.md 选型记录：多实例部署时再换 Redis）。serve 用单副本租约强制执行（rodit.ops.lease：
PostgreSQL 会话级 advisory lock，按平台单元取键）：组装好服务、通过启动前检查之后抢租约，另一个
进程占着就拒绝启动（退出码 2，日志写明占用者），连不上数据库退出码 1；RODIT_MIGRATE_ON_START 的
迁移在拿到租约之后跑；服务退出（包括异常退出）时释放。就绪检查多一项 lease（连接断了会重连再抢，
被别的进程抢走就一直不就绪）。

离线任务（在平台单元内跑，不需要 RODIT_PLATFORM；结果都由人确认后才上线）：

    recalibrate [--since ISO] [--until ISO] [--min-support N] [--out FILE] [--config YAML]
        读标注（默认最近 30 天）→ 中文摘要打到标准输出；只含要修改的类型的配置片段写到 --out
        （没给就接在摘要后面）。不改任何配置文件。
    train-classifier --out FILE [--min-samples N] [--force] [--overwrite] [--min-support N]
                     [--config YAML]
        读标注 → 训练、和关键词规则对比、留出集校准阈值 → 原子写入模型文件（临时文件 + 改名）。
        --out 已存在时拒绝（线上可能正在用它：重启就会加载一个没跑过回归的模型；请用带日期的
        新文件名，回归通过后再改 RODIT_CLASSIFIER_PATH）；--overwrite 才覆盖，而且已有文件的
        训练时间比这次新时仍然拒绝。可用标注少于 --min-samples（默认 200）时拒绝，除非 --force。

    标注从 FeedbackRepository.export_training_labels() 读（本单元所有店铺的 labeled_sample，
    唯一的跨店读路径）：连接串在 RODIT_TRAINER_DSN，这个登录账号要能 SET ROLE rodit_trainer
    （GRANT rodit_trainer TO 该账号），只属于训练任务，不是商家侧服务的 rodit_app 账号
    （见 rodit.storage.db）。--config 默认 $RODIT_CONFIG，都没有时用内置默认值。
    两个任务都提醒：上线前按 §13.5 先跑回归（python -m rodit.eval --baseline …）。

    退出码：0 完成；1 数据库不可用（连不上、没有 rodit_trainer 权限）；2 参数或配置不对、
    缺 RODIT_TRAINER_DSN、标注不够（train-classifier 没加 --force）、输出文件已存在 / 写不进去。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, TextIO, get_args

import psycopg
import yaml
from fastapi import FastAPI

import rodit.ops.jobs as jobs
from rodit.adapters import NullToolGateway
from rodit.api import create_app
from rodit.api.auth import PrincipalRole
from rodit.app.production import build_production_unit, migrate, production_database
from rodit.app.service import AssistService
from rodit.app.unit import Repos, Unit, build_unit
from rodit.core.clock import CST, Clock, SystemClock
from rodit.core.config import EngineConfig, load_config
from rodit.core.enums import ProviderName
from rodit.core.errors import StorageUnavailable
from rodit.core.models import LabeledSample
from rodit.core.pii import BuyerRefHasher
from rodit.core.ports import AuditLog, ToolGateway
from rodit.core.questions import ROUTER_QUESTIONS_VER
from rodit.decision import ClassifierArtifactError, ClassifierProvider, RuleProvider
from rodit.eval.recalibrate import DEFAULT_MIN_SUPPORT
from rodit.eval.train import DEFAULT_MIN_SAMPLES
from rodit.llm import MeteredLLMClient
from rodit.ops.bindings import (
    BindingContext,
    BindingError,
    BindingsFactory,
    PlatformBindings,
    bind_context,
    demo_bindings,
    load_factory,
    mode_c_bindings,
    validate_bindings,
)
from rodit.ops.health import (
    CheckResult,
    Probe,
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
    ObservedReadinessChecker,
    RoditMetrics,
)
from rodit.ops.lease import Lease, LeaseError, serve_lease
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
from rodit.storage import APP_ROLE, Database, PgFeedbackRepository

logger = logging.getLogger("rodit.ops")

LLM_ENV_PREFIX = "RODIT_LLM_"
MIGRATE_DSN_ENV = "RODIT_MIGRATE_DSN"
DATABASE_DSN_ENV = "RODIT_DATABASE_DSN"
TOKEN_KEY_ENV = "RODIT_TOKEN_KEY"
TRAINER_DSN_ENV = "RODIT_TRAINER_DSN"
CLASSIFIER_PATH_ENV = "RODIT_CLASSIFIER_PATH"
EXIT_FAILURE = 1
EXIT_STARTUP = 2


class LabelSource(Protocol):
    """离线任务的标注来源：默认 PostgreSQL 的 export_training_labels（trainer_label_source）；
    测试注入假的。since ≤ created_at < until；不给就是全部（训练用全部，重算只要时间窗里的）。"""

    def __call__(
        self, *, since: datetime | None = None, until: datetime | None = None
    ) -> Sequence[LabeledSample]: ...


class StartupError(RuntimeError):
    """启动前检查不通过。消息里只有变量名、文件路径和原因，没有任何值。"""


LeaseFactory = Callable[[ServerSettings], Lease]
"""设置 → 单副本租约（默认 rodit.ops.lease.serve_lease；测试注入假的）。"""


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
    /readyz 报 missing_api_key，进程不接流量，日志里看得到原因。

    生产模式下配置要求公示大模型（llm.require_disclosure）时，公示名称和备案号都要填
    （§12.3 生成式人工智能服务登记：公示所用模型的名称和备案号 / 上线编号）。"""
    env = os.environ if environ is None else environ
    required = [DATABASE_DSN_ENV, TOKEN_KEY_ENV, config.privacy.buyer_salt_env]
    missing = [name for name in required if not env.get(name, "").strip()]
    if settings.is_production and not _llm_configured(config, env):
        missing.append(f"{LLM_ENV_PREFIX}BASE_URL / {LLM_ENV_PREFIX}MODEL（或配置文件 llm 段）")
    if missing:
        raise StartupError("缺少必需的环境变量：" + "、".join(missing))
    if settings.is_production and config.llm.require_disclosure:
        blank = [
            f"llm.{name}"
            for name in ("public_name", "filing_no")
            if not (getattr(config.llm, name) or "").strip()
        ]
        if blank:
            raise StartupError(
                "配置 llm.require_disclosure 为 true，缺少 "
                + "、".join(blank)
                + "（§12.3 要公示所用大模型的名称和备案号，D5 选定后填进部署配置）"
            )


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


def load_route_classifier(
    settings: ServerSettings, config: EngineConfig
) -> ClassifierProvider | None:
    """decision.provider 是 classifier 时加载路由分类器的模型文件（§6.4 Phase 2），否则返回 None。

    没设 RODIT_CLASSIFIER_PATH、文件不存在、格式 / 版本 / 校验和 / 问题类型不对都拒绝启动
    （StartupError，消息里只有变量名、路径和原因）。intent 以外的路由问题交给关键词规则。"""
    if config.decision.provider != ProviderName.CLASSIFIER:
        if settings.classifier_path:
            logger.warning(
                "设置了 %s，但 decision.provider 是 %s：不加载路由分类器",
                CLASSIFIER_PATH_ENV,
                config.decision.provider.value,
            )
        return None
    if not settings.classifier_path:
        raise StartupError(
            f"decision.provider 是 classifier，缺少环境变量 {CLASSIFIER_PATH_ENV}"
            "（路由分类器模型文件，由 python -m rodit.ops train-classifier 生成）"
        )
    try:
        classifier = ClassifierProvider.load(settings.classifier_path, fallback=RuleProvider())
    except ClassifierArtifactError as exc:
        raise StartupError(f"路由分类器模型文件不能用（{CLASSIFIER_PATH_ENV}）：{exc}") from None
    meta = classifier.meta
    assert meta is not None
    logger.info(
        "路由分类器已加载：训练时间 %s，训练样本 %d 条，问题类型 %d 个，sha256 %s",
        meta.trained_at.astimezone(CST).isoformat(timespec="minutes"),
        meta.samples,
        len(classifier.classes),
        classifier.artifact_sha256,
    )
    if meta.questions_version != ROUTER_QUESTIONS_VER:
        logger.warning(
            "路由分类器训练时的路由问题定义版本 %s 和当前 %s 不同："
            "问题分类改过的话请重新训练并跑回归",
            meta.questions_version,
            ROUTER_QUESTIONS_VER,
        )
    return classifier


# ================================================================ 组装


@dataclass
class _Composed:
    unit: Unit
    bindings: PlatformBindings
    metrics: RoditMetrics
    llm_probe: Callable[[], Mapping[str, CheckResult]]
    ctx: BindingContext | None = None  # 交给绑定工厂的上下文（服务组装好后接上 on_subscription）


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
    其他情况一律 PostgreSQL（build_production_unit）。

    顺序：先调绑定工厂（它拿到的 ctx.tokens / ctx.on_subscription 是代理），再组装平台单元，
    把绑定里的 token_refresher / parse_subscription 挂到单元上，然后把 ctx.tokens 接到
    Unit.tokens；ctx.on_subscription 由 build_server 在服务组装好之后接上。"""
    env = os.environ if environ is None else environ
    config = load_config(settings.config_path or None)
    preflight(settings, config, env)
    classifier = load_route_classifier(settings, config)
    clock = clock or SystemClock()
    hasher = BuyerRefHasher.from_env(config.privacy.buyer_salt_env)
    ctx = BindingContext(platform=settings.platform, config=config, clock=clock, hasher=hasher)
    bindings = resolve_bindings(settings, ctx, factory=bindings_factory)
    metrics = RoditMetrics(platform=settings.platform, mode=settings.deploy_mode)
    # 模式 C 的空网关不调任何平台接口：不计量，免得平台调用量指标里出现根本没发生的调用（§12.4）
    tools: ToolGateway = (
        bindings.tools
        if isinstance(bindings.tools, NullToolGateway)
        else MeteredToolGateway(bindings.tools, metrics)
    )

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
            catalog=bindings.catalog,
            ocr=bindings.ocr,
            route_provider=classifier,
            llm_transport=bindings.llm_transport,  # 平台可报备的通道（§9.1、§12.3），没给用默认出口
        )
        probe = llm_probe(config.llm, environ=env, env_prefix=LLM_ENV_PREFIX)
    else:
        if bindings.llm_transport is not None:
            logger.warning("本地试用没有配置大模型（离线替身）：平台绑定的 llm_transport 用不到")
        unit = _offline_unit(
            settings, config, clock, hasher, bindings, tools, observe, repos, classifier
        )
        probe = static_probe("llm", CheckResult(True, {"offline": True}))
    # 平台云代码的续期接口、订购推送解析挂到单元上：店铺生命周期（续期任务、平台推送订购事件的
    # 接口）经 Unit / AssistService 取用
    unit.token_refresher = bindings.token_refresher
    unit.parse_subscription = bindings.parse_subscription
    bind_context(ctx, tokens=unit.tokens)  # 单元组装好了：平台云代码可以取店铺令牌了
    return _Composed(unit=unit, bindings=bindings, metrics=metrics, llm_probe=probe, ctx=ctx)


def _offline_unit(
    settings: ServerSettings,
    config: EngineConfig,
    clock: Clock,
    hasher: BuyerRefHasher,
    bindings: PlatformBindings,
    tools: ToolGateway,
    observe: Callable[[AuditLog], AuditLog],
    repos: Repos | None,
    classifier: ClassifierProvider | None = None,
) -> Unit:
    """本地试用、没配大模型：离线替身（关键词路由、复述依据、启发式质检），仓储默认 PostgreSQL。
    配置了路由分类器（decision.provider: classifier）时路由用它，否则用关键词规则。"""
    from rodit.app.offline import EvidenceEchoLLM, HeuristicQAJudge
    from rodit.app.production import postgres_repos

    logger.warning("本地试用：没有配置大模型，使用离线替身（回答只复述依据，仅供试用）")
    base = repos if repos is not None else postgres_repos(production_database(), clock=clock)
    repos = replace(base, audit=observe(base.audit))
    if classifier is None:
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
        route_provider=classifier or RuleProvider(),
        qa_provider=HeuristicQAJudge(),
        catalog=bindings.catalog,
        ocr=bindings.ocr,
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
    lease: Lease | None = None,
) -> Server:
    """组装完整的服务（见模块说明）。database 覆盖就绪检查用的数据库（测试用）。

    lease：单副本租约（main 在组装之后、跑服务之前 acquire），给了就在就绪检查里加一项 lease。
    /metrics 渲染前刷新就绪结果（rodit_ready）和缓存层的统计（缓存命中、过期兜底、串店）。"""
    composed = compose_unit(
        settings, bindings_factory=bindings_factory, clock=clock, repos=repos, environ=environ
    )
    unit, metrics = composed.unit, composed.metrics
    authenticator = TokenFileAuthenticator(
        settings.tokens_file, reload_interval_s=settings.tokens_reload_s, clock=unit.clock
    )
    guard = _metrics_guard(settings, authenticator)
    probes: dict[str, Probe] = {
        "database": database_probe(database or production_database()),
        "llm": composed.llm_probe,
    }
    if lease is not None:
        probes["lease"] = lease.probe
    readiness = ObservedReadinessChecker(probes, metrics, cache_s=settings.readiness_cache_s)
    # 抓取时就绪结果也是新的（结果有缓存，抓得再勤也不会打满数据库）
    metrics.registry.before_render(readiness.check)
    gateway = unit.gateway
    metrics.registry.before_render(lambda: metrics.observe_tool_cache(gateway.totals()))
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
    if composed.ctx is not None:  # 服务组装好了：平台推送的订购事件交给店铺生命周期处理
        bind_context(composed.ctx, on_subscription=service.tenancy.handle_subscription)
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
    entry_json = issued.entry.model_dump_json(exclude_none=True)
    if args.append:
        # 先写进文件再显示令牌：写不进去（文件格式不对、没有权限）时不显示一个用不了的令牌
        append_entries(args.append, [issued.entry])
    print("令牌原文只显示这一次，请直接交给使用方；令牌文件里只存它的哈希：", file=out)
    _print_token(out, issued)
    if args.append:
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


# ================================================================ 离线任务：阈值重算、分类器训练


def trainer_label_source(environ: Mapping[str, str] | None = None) -> LabelSource:
    """默认的标注来源：PostgreSQL 的 export_training_labels（本单元所有店铺，rodit_trainer 角色）。

    连接串在 RODIT_TRAINER_DSN：这个登录账号要能 SET ROLE rodit_trainer，只属于训练任务
    （见 rodit.storage.db）。离线任务不设请求级的语句超时。连接在调用时才建。"""
    env = os.environ if environ is None else environ
    dsn = env.get(TRAINER_DSN_ENV, "").strip()
    if not dsn:
        raise StartupError(
            f"缺少必需的环境变量：{TRAINER_DSN_ENV}"
            "（训练导出用的登录账号，要能 SET ROLE rodit_trainer）"
        )
    return PgFeedbackRepository(Database(dsn)).export_training_labels


def _job_config(args: argparse.Namespace) -> tuple[EngineConfig, str]:
    """离线任务对照的配置：--config，其次 RODIT_CONFIG，都没有时用内置默认值。"""
    path = (args.config or os.environ.get("RODIT_CONFIG", "")).strip()
    if path:
        return load_config(path), path
    return load_config(None), "内置默认值（没有 --config，也没有 RODIT_CONFIG）"


def recalibrate_command(
    args: argparse.Namespace, out: TextIO, *, labels: LabelSource, clock: Clock
) -> int:
    """§13.5 阈值重算：摘要打到 out，配置片段写到 --out（没给就接在摘要后面）。不改任何配置。"""
    from rodit.eval.recalibrate import recalibrate, render_text, resolve_window

    config, source = _job_config(args)
    now = clock.now()
    since, until = resolve_window(now, args.since, args.until)
    report = recalibrate(
        labels(since=since, until=until),  # 只导出时间窗里的标注（库里按 created_at 过滤）
        config.routing,
        now=now,
        since=since,
        until=until,
        min_support=args.min_support,
    )
    print(render_text(report, config_source=source), file=out)
    if args.out:
        Path(args.out).write_text(report.yaml, encoding="utf-8")
        print(f"配置片段已写入：{args.out}（只是建议，人工确认后再合并进部署配置）", file=out)
    else:
        print("", file=out)
        print(report.yaml, end="", file=out)
    return 0


def _refuse_newer_artifact(target: Path, trained_at: datetime) -> None:
    """--overwrite 时：已有的模型文件比这次训练的新就不覆盖；读不出来的文件照常覆盖。"""
    from rodit.decision import read_artifact

    try:
        existing = read_artifact(target)
    except ClassifierArtifactError as exc:
        logger.warning("已有的模型文件读不出来，按 --overwrite 覆盖：%s", exc)
        return
    previous = datetime.fromisoformat(existing["metadata"]["trained_at"])
    if previous > trained_at:
        raise StartupError(
            f"已有的模型文件更新（训练时间 {previous.astimezone(CST):%Y-%m-%d %H:%M}，"
            f"晚于这次的 {trained_at.astimezone(CST):%Y-%m-%d %H:%M}），不覆盖：{target}"
        )


def train_classifier_command(
    args: argparse.Namespace, out: TextIO, *, labels: LabelSource, clock: Clock
) -> int:
    """§6.4 Phase 2：训练路由分类器，原子写入模型文件（覆盖规则见模块说明），摘要打到 out。"""
    from rodit.decision import write_artifact
    from rodit.eval.train import render_text, train_classifier

    target = Path(args.out)
    if target.exists() and not args.overwrite:
        raise StartupError(
            f"模型文件已存在：{target}（线上可能正在用它；换一个带日期的文件名，"
            "回归通过后再改 RODIT_CLASSIFIER_PATH；确实要覆盖时加 --overwrite）"
        )
    if not target.parent.is_dir():
        raise StartupError(f"输出目录不存在：{target.parent}")
    config, _ = _job_config(args)
    now = clock.now()
    result = train_classifier(
        labels(),
        now=now,
        routing=config.routing,
        min_samples=args.min_samples,
        force=args.force,
        min_support=args.min_support,
    )
    if target.exists():
        _refuse_newer_artifact(target, now)
    write_artifact(target, result.artifact)
    print(render_text(result.report), file=out)
    print(f"已写入模型文件：{target}", file=out)
    return 0


def _run_job(
    command: str,
    args: argparse.Namespace,
    out: TextIO,
    *,
    label_source: LabelSource | None,
    clock: Clock | None,
) -> int:
    """跑离线任务，按模块说明映射退出码。数据库的错误只记类型（说明里可能有主机名、账号）。"""
    try:
        labels = label_source or trainer_label_source()
        run = recalibrate_command if command == "recalibrate" else train_classifier_command
        return run(args, out, labels=labels, clock=clock or SystemClock())
    except (StorageUnavailable, psycopg.Error) as exc:
        if isinstance(exc, psycopg.errors.InsufficientPrivilege):
            logger.error(
                "导出标注失败：没有权限（%s）；%s 的登录账号要能 SET ROLE rodit_trainer",
                type(exc).__name__,
                TRAINER_DSN_ENV,
            )
        else:
            logger.error("数据库不可用：%s", type(exc).__name__)
        return EXIT_FAILURE
    except (StartupError, ValueError, OSError, yaml.YAMLError) as exc:
        logger.error("%s 没有完成：%s", command, exc)
        return EXIT_STARTUP


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
    # 本机探测不走环境里的 HTTP 代理（HTTP_PROXY 没把 127.0.0.1 列进 NO_PROXY 时会发给代理）
    open_url = opener or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
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
    issue.add_argument("--role", choices=sorted(get_args(PrincipalRole)))
    issue.add_argument("--metrics", action="store_true", help="签发 /metrics 的抓取令牌")
    issue.add_argument("--name", help="抓取令牌的名字（--metrics 时）")
    issue.add_argument("--expires-days", type=float, default=0, help="有效天数（0 = 不过期）")
    issue.add_argument("--append", metavar="FILE", help="直接写进这个令牌文件（没有就新建）")
    config_help = "对照的部署配置（默认 RODIT_CONFIG，都没有时用内置默认值）"
    recal = sub.add_parser("recalibrate", help="§13.5 用采纳反馈重算阈值（只输出建议）")
    recal.add_argument("--since", type=_iso_time, help="标注时间窗起点（默认 --until 前 30 天）")
    recal.add_argument("--until", type=_iso_time, help="标注时间窗终点（不含；默认现在）")
    recal.add_argument(
        "--min-support",
        type=_positive_int,
        default=DEFAULT_MIN_SUPPORT,
        help=f"预测为该类型的标注少于这个数就不给建议（默认 {DEFAULT_MIN_SUPPORT}）",
    )
    recal.add_argument("--out", metavar="FILE", help="配置片段写到这个文件（默认接在摘要后面）")
    recal.add_argument("--config", metavar="YAML", help=config_help)
    train = sub.add_parser("train-classifier", help="§6.4 Phase 2 训练路由分类器，生成模型文件")
    train.add_argument("--out", metavar="FILE", required=True, help="模型文件（JSON）")
    train.add_argument(
        "--min-samples",
        type=_positive_int,
        default=DEFAULT_MIN_SAMPLES,
        help=f"去重后的可用标注少于这个数就拒绝训练（默认 {DEFAULT_MIN_SAMPLES}）",
    )
    train.add_argument("--force", action="store_true", help="标注不够也训练（只用于试验）")
    train.add_argument(
        "--overwrite", action="store_true", help="覆盖已有的模型文件（比这次新的仍不覆盖）"
    )
    train.add_argument(
        "--min-support",
        type=_positive_int,
        default=DEFAULT_MIN_SUPPORT,
        help=f"留出集上预测为该类型的样本少于这个数就不校准（默认 {DEFAULT_MIN_SUPPORT}）",
    )
    train.add_argument("--config", metavar="YAML", help=config_help)
    jobs.register_all(sub)  # 任务模块的子命令（rodit.ops.jobs）
    return parser


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("必须是正整数") from None
    if value < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def _iso_time(text: str) -> datetime:
    """ISO 8601 时间；没写时区的按东八区。"""
    try:
        value = datetime.fromisoformat(text.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            "应为 ISO 8601 时间，如 2026-09-01 或 2026-09-01T00:00+08:00"
        ) from None
    return value if value.tzinfo is not None else value.replace(tzinfo=CST)


def _early_logging(environ: Mapping[str, str]) -> None:
    """设置还没解析时先按环境变量配好日志（值不合法时退回默认），启动失败也有结构化日志。"""
    level = environ.get("RODIT_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    fmt = environ.get("RODIT_LOG_FORMAT", "json").strip().lower()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        level = "INFO"
    configure_logging(level, fmt if fmt in {"json", "text"} else "json")


def _default_lease(settings: ServerSettings) -> Lease:
    return serve_lease(settings.platform)


def _release(lease: Lease | None) -> None:
    """放租约（没拿到、已经放过都可以调用）。放不掉只记日志：进程退出时连接断开，锁照样释放。"""
    if lease is None:
        return
    try:
        lease.release()
    except Exception as exc:  # noqa: BLE001
        logger.warning("释放单副本租约失败：%s", type(exc).__name__)


def main(
    argv: Sequence[str] | None = None,
    *,
    run: Callable[..., Any] | None = None,
    out: TextIO | None = None,
    label_source: LabelSource | None = None,
    clock: Clock | None = None,
    lease_factory: LeaseFactory | None = None,
) -> int:
    """命令行入口。run 默认是 uvicorn.run（测试注入假的）。返回进程退出码。

    label_source / clock 只给离线任务（recalibrate、train-classifier）用：默认从 PostgreSQL 读
    标注（trainer_label_source）、用系统时间；测试注入假的，不连数据库。
    lease_factory 只给 serve 用：默认是 PostgreSQL 的单副本租约（rodit.ops.lease）。

    serve 的顺序：设置 → 组装服务（启动前检查，不连库）→ 抢租约 → 按需迁移 → 跑服务 → 放租约。"""
    out = out or sys.stdout
    args = _parser().parse_args(list(argv) if argv is not None else None)
    command = args.command or "serve"
    if command == "healthcheck":
        return healthcheck()
    _early_logging(os.environ)
    if command in ("recalibrate", "train-classifier"):
        return _run_job(command, args, out, label_source=label_source, clock=clock)
    lease: Lease | None = None
    try:
        # 任务模块认领的命令（rodit.ops.jobs）：抛出的异常和内置命令一样按下面映射退出码
        code = jobs.dispatch(args, os.environ)
        if code is not None:
            return code
        if command == "issue-token":
            return issue_token_command(args, out)
        if command == "migrate":
            migrate_database()
            return 0
        settings = ServerSettings.from_env()
        configure_logging(settings.log_level, settings.log_format)
        if command == "local-init":
            return local_init(settings, out)
        candidate = (lease_factory or _default_lease)(settings)
        server = build_server(settings, lease=candidate)
        lease = candidate  # 从这里起失败都要放租约（没拿到时放也没关系）
        lease.acquire()  # 另一个进程占着 → LeaseHeld：启动失败，退出码 2
        if settings.migrate_on_start:
            migrate_database()
    except (SettingsError, StartupError, BindingError, TokenFileError, LeaseError) as exc:
        _release(lease)
        logger.error("启动失败：%s", exc)
        return EXIT_STARTUP
    except RuntimeError as exc:  # 其他模块缺环境变量时的报错（只含变量名）
        _release(lease)
        logger.error("启动失败：%s", exc)
        return EXIT_STARTUP
    except (StorageUnavailable, psycopg.Error) as exc:  # 只记类型：说明里可能有主机名、账号
        _release(lease)
        logger.error("数据库不可用：%s", type(exc).__name__)
        return EXIT_FAILURE
    # 配置值不对（超时不是正整数、配置文件不存在、YAML 写错……）
    except (ValueError, OSError, yaml.YAMLError) as exc:
        _release(lease)
        logger.error("启动失败：%s", exc)
        return EXIT_STARTUP
    except BaseException:
        _release(lease)
        raise
    if run is None:
        import uvicorn

        run = uvicorn.run
    try:
        run(
            server.app,
            host=settings.host,
            port=settings.port,
            log_config=None,  # 用 configure_logging 的 JSON 日志和过滤
            server_header=False,
            timeout_graceful_shutdown=20,
        )
    finally:
        _release(lease)  # 正常退出、异常退出都放租约（被杀时连接断开，数据库自动释放）
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
