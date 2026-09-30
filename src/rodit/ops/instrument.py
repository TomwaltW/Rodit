"""运行指标的采集（方案 §12.4「质量与技术指标（我们自己看）」、§12.1 性能目标）。

不改业务代码，只在组装时套一层：

- ObservedAuditLog：套在单元的 AuditLog 外面。引擎每条消息写的 engine.result（意图、推荐数、
  质检、硬规则、降级、各环节耗时）和 MeteredLLMClient 写的 llm.usage（每个店铺的 token 数）
  都经过它，顺手记指标，再原样转给真正的审计。解析 engine.result 用 rodit.feedback.metrics
  的同一套规则（和给商家看的质检报表口径一致）。
- MeteredToolGateway：套在平台原始只读工具外面（缓存层之下），记平台接口的调用量、失败、耗时
  （§12.4 平台接口调用量、失败率）；失败按 ToolUnavailable.reason 分成 unavailable（故障）/
  throttled（限流，额度用完）/ timeout（超时）。
- observe_tool_cache：抓取时导出缓存层（rodit.tools.CachingToolGateway.totals()）自己记的统计：
  请求数、缓存命中、平台失败时用过期缓存兜底的次数、内层返回别家店铺数据的次数
  （按工具，不按店铺）。
- AuthExpiryMonitor：（serve 的后台线程按间隔）检查授权即将过期、已经过期的店铺
  （§12.4 授权过期告警）。
- ObservedReadinessChecker：就绪检查（/readyz、serve 的后台线程）的结果记成 rodit_ready；
  抓取 /metrics 时只输出上次的结果和它的年龄（publish），不查库，太旧了记成不就绪。
- HttpMetricsMiddleware：按路由模板（不是实际路径）记请求数、耗时和错误码。
- 后台任务（rodit.ops.background）每一轮的结果：rodit_background_task_runs_total{task,outcome}
  （标签不叫 job：job 是 Prometheus 抓取目标的标签，重名会被改成 exported_job）。

抓取 /metrics 时的钩子（before_render）只做内存里的事，不连数据库：数据库网络不通时连接要等
超时，抓取会超过 Prometheus 的抓取超时，结果是一个样本都没有、"不就绪"的告警反而发不出来。

串店（§9.2、§12.4 串店检查）：rodit_tenant_isolation_total{stage} 每次拦截记一次——
tenant.violation 审计事件按它 detail 里的 stage（引擎在哪一步发现的）；接口层按调用方身份拦下的
403 tenant_isolation（令牌不是这家店的）记 stage="http"；接口处理中服务端自己发现的串店
（TenantIsolationError：平台工具、知识仓储返回了别家店铺的数据等，响应同样是 403 tenant_isolation）
记 stage="server"（mark_server_tenant_isolation 给错误映射包一层、在请求状态里做标记）。
告警里 stage="http" 是 warning（调用方的问题），其余都是 critical（真的拿到了别家店铺的数据）。
同一次拦截在引擎的 engine.result 里还会有降级类型 tenant_isolation
（rodit_degradations_total{kind="tenant_isolation"}）、在缓存层还会记
rodit_tool_cache_tenant_violations_total，这两个是另外的指标，不重复计入
rodit_tenant_isolation_total；告警（deploy/prometheus/alerts.yml）三个都看。

端到端耗时：engine.result 的 latency_ms 每一项都进 rodit_engine_latency_seconds{stage}；
total 是引擎内的端到端，since_received（core.audit.LATENCY_SINCE_RECEIVED）从平台收到消息算起，
含排队、会话锁等待，更接近 §12.1 "推荐出现（端到端）P95 ≤ 5 秒" 的口径。

标签里不放买家数据：只有店铺 ID、问题类型、路由动作、规则名、降级类型、工具名、大模型用途、
路由模板这类代码；所有值再过 metrics.safe_label_value（中文、空格、个人信息一律变成 "other"）。
指标采集失败只记日志，绝不影响问答和审计本身。
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping
from datetime import datetime, timedelta
from importlib import metadata
from typing import Any

from starlette.routing import Match

from rodit.adapters import AuthGuard
from rodit.core.audit import DEGRADED_TENANT_ISOLATION, AuditAction
from rodit.core.clock import Clock, SystemClock
from rodit.core.enums import AfterSaleStatus, DecisionStage, Intent, Platform, RouteAction
from rodit.core.errors import TOOL_FAILURE_REASONS, TenantIsolationError, ToolUnavailable
from rodit.core.models import (
    NON_BLOCKING_HARD_RULES,
    AfterSaleInfo,
    AuditEvent,
    DecisionLogEntry,
    LogisticsTrace,
    OrderInfo,
    PriceInfo,
    StockInfo,
)
from rodit.core.ports import AuditLog, ToolGateway
from rodit.feedback import metrics as report
from rodit.ops.health import DEFAULT_CACHE_S, Probe, Readiness, ReadinessChecker
from rodit.ops.metrics import MetricsRegistry, safe_label_value

logger = logging.getLogger(__name__)

LATENCY_BUCKETS_S: tuple[float, ...] = (
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
    15.0,
    30.0,
)
"""端到端和各环节耗时的桶（秒）。§12.1 的目标值（0.5 / 1 / 2.5 / 5 秒）都是桶边界，
P95 ≤ 5 秒可以直接按桶判断。"""

TOOL_BUCKETS_S: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
"""平台接口耗时的桶（秒）。§12.1 检索与工具调用 ≤ 0.5 秒。"""

_CODE = re.compile(r"[a-z][a-z0-9_]{0,39}(?::[a-z0-9_]{1,40})?")
_ACTION = re.compile(r"[a-z][a-z0-9_]{0,30}(?:\.[a-z0-9_]{1,30}){0,2}")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_INTENTS = frozenset(i.value for i in Intent)
_ACTIONS = frozenset(a.value for a in RouteAction)

NO_SHOP = "none"  # 没有店铺归属的事件
DEFAULT_READINESS_STALE_S = 60.0
"""/metrics 输出的就绪结果最多多旧（秒）：再旧就记成不就绪（后台刷新卡住了）。"""
TENANT_STAGE_HTTP = "http"
"""接口层按调用方身份拦下的串店（403 tenant_isolation）：rodit_tenant_isolation_total 的 stage。"""
TENANT_STAGE_SERVER = "server"
"""接口处理中服务端发现的串店（TenantIsolationError → 403 tenant_isolation）的 stage。"""
TENANT_STAGE_STATE_KEY = "rodit_tenant_stage"
"""请求状态（scope["state"]）里的标记：服务端发现的串店在这里写上 stage，中间件按它记指标。"""
TENANT_STAGE_UNKNOWN = "unknown"
"""tenant.violation 事件没写 stage（或不是代码）时的 stage。"""
TENANT_ISOLATION_CODE = "tenant_isolation"  # 接口层的错误码（rodit.api.errors）
TOOL_CACHE_FIELDS: tuple[tuple[str, str], ...] = (
    ("requests", "tool_cache_requests"),
    ("cache_hits", "tool_cache_hits"),
    ("stale_served", "tool_cache_stale"),
    ("tenant_violations", "tool_cache_tenant"),
)
"""缓存层统计（rodit.tools.ToolCallStats 的字段）→ RoditMetrics 上的计数器。"""


def _code(value: object, *, default: str = "other") -> str:
    """规则名、降级类型、环节名这类代码：小写字母数字下划线（可带一个 ":子类"），否则 default。"""
    if isinstance(value, str) and _CODE.fullmatch(value):
        return value
    return default


def _enum_value(value: object, allowed: frozenset[str]) -> str:
    if value is None:
        return "none"
    text = str(value)
    return text if text in allowed else "other"


def package_version() -> str:
    try:
        return metadata.version("rodit")
    except metadata.PackageNotFoundError:  # pragma: no cover  源码目录直接运行
        return "unknown"


class RoditMetrics:
    """一个平台单元的全部运行指标（见模块说明）。线程安全。"""

    def __init__(
        self,
        *,
        platform: Platform | str,
        mode: str,
        registry: MetricsRegistry | None = None,
        version: str | None = None,
    ) -> None:
        r = registry or MetricsRegistry()
        self.registry = r
        self.build_info = r.gauge(
            "rodit_build_info", "构建信息（值恒为 1）", ["platform", "mode", "version"]
        )
        self.build_info.set(
            {
                "platform": Platform(platform).value,
                "mode": mode,
                "version": version or package_version(),
            },
            1,
        )
        # ---- 问答引擎（engine.result）
        self.messages = r.counter(
            "rodit_engine_messages_total",
            "引擎处理的买家消息数；outcome = suggested / no_suggestion / transfer_to_human",
            ["shop_id", "outcome"],
        )
        self.routes = r.counter(
            "rodit_engine_routes_total", "路由结果（问题类型、§6.6 动作）", ["intent", "action"]
        )
        self.suggestions = r.counter(
            "rodit_suggestions_total", "给出的推荐卡片数（含影子模式）", ["shop_id"]
        )
        self.stage_latency = r.histogram(
            "rodit_engine_latency_seconds",
            "问答引擎各环节耗时；stage=total 为端到端（§12.1 P95 ≤ 5 秒）",
            ["stage"],
            buckets=LATENCY_BUCKETS_S,
        )
        self.qa_checks = r.counter("rodit_qa_checks_total", "质检次数（含重新生成后的复检）")
        self.qa_failures = r.counter("rodit_qa_failures_total", "质检不通过次数")
        self.hard_rule_hits = r.counter(
            "rodit_qa_hard_rule_hits_total",
            "硬规则命中次数；blocking=false 为自动修正类（format / length）",
            ["rule", "blocking"],
        )
        self.degradations = r.counter(
            "rodit_degradations_total", "降级次数（§12.2），按降级类型", ["kind"]
        )
        # 预先建好 0 值序列：第一次串店也能被 increase() 看到（序列从无到 1 时 increase 是 0）
        self.degradations.inc({"kind": DEGRADED_TENANT_ISOLATION}, 0)
        self.tenant_isolation = r.counter(
            "rodit_tenant_isolation_total",
            "串店拦截次数（§9.2、§12.4 串店检查）；stage = 在哪一步拦下"
            "（tenant.violation 事件的 stage；http = 接口层按调用方身份拦下）",
            ["stage"],
        )
        self.tenant_isolation.inc({"stage": TENANT_STAGE_HTTP}, 0)
        self.tenant_isolation.inc({"stage": TENANT_STAGE_SERVER}, 0)
        # ---- 大模型（llm.usage）
        self.llm_calls = r.counter(
            "rodit_llm_calls_total",
            "大模型调用次数；outcome = ok 或异常类型",
            ["purpose", "outcome"],
        )
        self.llm_tokens = r.counter(
            "rodit_llm_tokens_total",
            "大模型 token 数（§12.4 每个店铺的模型成本）；direction = input / output",
            ["shop_id", "purpose", "direction"],
        )
        self.llm_latency = r.histogram(
            "rodit_llm_latency_seconds", "大模型调用耗时", ["purpose"], buckets=LATENCY_BUCKETS_S
        )
        # ---- 平台接口
        self.tool_calls = r.counter(
            "rodit_platform_tool_calls_total",
            "平台只读接口调用次数（缓存未命中才会调用）；outcome = ok / not_found / "
            "unavailable（故障）/ throttled（限流）/ timeout（超时）/ error（其他异常）",
            ["tool", "outcome"],
        )
        self.tool_latency = r.histogram(
            "rodit_platform_tool_latency_seconds",
            "平台只读接口耗时",
            ["tool"],
            buckets=TOOL_BUCKETS_S,
        )
        # ---- 缓存层（rodit.tools.CachingToolGateway 自己的统计，抓取时导出；按工具，各店相加）
        self.tool_cache_requests = r.counter(
            "rodit_tool_cache_requests_total",
            "问答链路查平台只读工具的次数（含命中缓存）",
            ["tool"],
        )
        self.tool_cache_hits = r.counter(
            "rodit_tool_cache_hits_total", "有效期内命中缓存、没有调用平台接口的次数", ["tool"]
        )
        self.tool_cache_stale = r.counter(
            "rodit_tool_cache_stale_served_total",
            "平台接口限流或故障时用过期缓存兜底的次数（§12.2 用缓存数据并注明时间）",
            ["tool"],
        )
        self.tool_cache_tenant = r.counter(
            "rodit_tool_cache_tenant_violations_total",
            "平台接口返回了别家店铺的数据、被缓存层拦下的次数（§9.2）",
            ["tool"],
        )
        # ---- 审计
        self.audit_events = r.counter("rodit_audit_events_total", "审计事件数", ["action"])
        self.audit_failures = r.counter(
            "rodit_audit_write_failures_total", "审计写入失败次数", ["kind"]
        )
        # ---- 店铺授权（§12.4 授权过期告警）
        self.auth_expiring = r.gauge(
            "rodit_auth_expiring_shops", "授权在告警窗口内到期（含已过期、没有授权）的店铺数"
        )
        self.auth_expired = r.gauge(
            "rodit_auth_expired_shops",
            "授权已经过期（含没有授权、需要重新授权、令牌读不出来）的店铺数：这些店铺已停止处理",
        )
        self.auth_expiring_shop = r.gauge(
            "rodit_shop_auth_expiring", "授权在告警窗口内到期的店铺（值恒为 1）", ["shop_id"]
        )
        self.auth_checked = r.gauge("rodit_auth_checked_shops", "上次授权检查覆盖的店铺数")
        self.auth_checked_at = r.gauge(
            "rodit_auth_expiry_last_check_timestamp_seconds", "上次授权检查成功的时间（Unix 秒）"
        )
        self.auth_check_errors = r.counter(
            "rodit_auth_expiry_check_errors_total", "授权检查失败次数"
        )
        self.auth_check_errors.inc(amount=0)  # 0 值序列：第一次失败也能被 increase() 看到
        # ---- 就绪（/readyz）
        self.ready = r.gauge(
            "rodit_ready",
            "就绪检查的结果：1 就绪，0 未就绪（不接流量；还没检查过、上次的结果太旧也是 0）",
        )
        self.readiness_age = r.gauge(
            "rodit_readiness_age_seconds",
            "上次就绪检查距今的秒数（/metrics 只输出上次的结果，不在抓取时查库）",
        )
        self.readiness_checks = r.gauge(
            "rodit_readiness_check_ok",
            "就绪检查各项的结果（1 通过，0 不通过）：database / migrations / llm / lease",
            ["check"],
        )
        # ---- HTTP
        self.http_requests = r.counter(
            "rodit_http_requests_total", "HTTP 请求数（按路由模板）", ["route", "method", "status"]
        )
        self.http_latency = r.histogram(
            "rodit_http_request_duration_seconds",
            "HTTP 请求耗时（按路由模板）",
            ["route", "method"],
            buckets=LATENCY_BUCKETS_S,
        )
        self.http_errors = r.counter(
            "rodit_http_errors_total",
            "错误响应的错误码（tenant_isolation = 串店拦截，§9.2）",
            ["route", "code"],
        )
        # ---- serve 的后台任务（rodit.ops.background：续期、过期清理、就绪和授权检查）
        self.background_runs = r.counter(
            "rodit_background_task_runs_total",
            "serve 后台任务每一轮的结果；outcome = ok / error（出错或没做完）/ "
            "skipped（前提不满足，如单副本租约不在本进程）",
            ["task", "outcome"],
        )
        self.background_last_ok = r.gauge(
            "rodit_background_task_last_success_timestamp_seconds",
            "serve 后台任务上次成功的时间（Unix 秒）",
            ["task"],
        )

    # ================================================================ 审计事件

    def observe_event(self, event: AuditEvent) -> None:
        """一条审计事件 → 指标。只读 action、shop_id 和 detail 里的代码与数字。"""
        self.audit_events.inc({"action": _code_action(event.action)})
        if event.action == AuditAction.RESULT.value:
            self.observe_engine_result(event.shop_id, event.detail)
        elif event.action == AuditAction.LLM_USAGE.value:
            self.observe_llm_usage(event.shop_id, event.detail)
        elif event.action == AuditAction.TENANT_VIOLATION.value:
            self.observe_tenant_violation(event.detail)

    def observe_tenant_violation(self, detail: Mapping[str, Any]) -> None:
        """tenant.violation 事件：只取 detail 里的 stage（代码），不看别的。"""
        stage = _code(detail.get("stage"), default=TENANT_STAGE_UNKNOWN)
        self.tenant_isolation.inc({"stage": stage})

    def observe_engine_result(self, shop_id: str | None, detail: Mapping[str, Any]) -> None:
        shop = shop_id or NO_SHOP
        suggestions = report._as_int(detail.get(report.RESULT_SUGGESTIONS))
        if detail.get(report.RESULT_TRANSFER) is True:
            outcome = "transfer_to_human"
        elif suggestions > 0:
            outcome = "suggested"
        else:
            outcome = "no_suggestion"
        self.messages.inc({"shop_id": shop, "outcome": outcome})
        self.routes.inc(
            {
                "intent": _enum_value(detail.get(report.RESULT_INTENT), _INTENTS),
                "action": _enum_value(detail.get(report.RESULT_ACTION), _ACTIONS),
            }
        )
        if suggestions:
            self.suggestions.inc({"shop_id": shop}, suggestions)
        checked = report._as_int(detail.get(report.RESULT_QA_CHECKED))
        failed = report._as_int(detail.get(report.RESULT_QA_FAILED))
        if checked:
            self.qa_checks.inc(amount=checked)
        if failed:
            self.qa_failures.inc(amount=failed)
        for rule in report._str_list(detail.get(report.RESULT_HARD_HITS), key="rule"):
            blocking = rule not in NON_BLOCKING_HARD_RULES
            self.hard_rule_hits.inc({"rule": _code(rule), "blocking": blocking})
        degraded = detail.get(report.RESULT_DEGRADED)
        kinds = ["unknown"] if degraded is True else report._str_list(degraded)
        for kind in kinds:
            self.degradations.inc({"kind": _code(kind)})
        for stage, ms in report._latencies(detail.get(report.RESULT_LATENCY)).items():
            if _CODE.fullmatch(stage) and ms >= 0:
                self.stage_latency.observe({"stage": stage}, ms / 1000.0)

    def observe_llm_usage(self, shop_id: str | None, detail: Mapping[str, Any]) -> None:
        purpose = _code(detail.get("purpose"))
        error = detail.get("error")
        outcome = "ok" if not error else _identifier(error)
        self.llm_calls.inc({"purpose": purpose, "outcome": outcome})
        shop = shop_id or NO_SHOP
        for direction, key in (("input", "input_tokens"), ("output", "output_tokens")):
            tokens = report._as_int(detail.get(key))
            if tokens:
                self.llm_tokens.inc(
                    {"shop_id": shop, "purpose": purpose, "direction": direction}, tokens
                )
        latency = detail.get("latency_ms")
        if isinstance(latency, int | float) and not isinstance(latency, bool) and latency >= 0:
            self.llm_latency.observe({"purpose": purpose}, latency / 1000.0)

    # ================================================================ 其他

    def observe_tool(self, tool: str, outcome: str, seconds: float) -> None:
        self.tool_calls.inc({"tool": tool, "outcome": outcome})
        self.tool_latency.observe({"tool": tool}, seconds)

    def observe_http(
        self,
        route: str,
        method: str,
        status: int,
        seconds: float,
        code: str | None,
        *,
        tenant_stage: str | None = None,
    ) -> None:
        """tenant_stage：错误码是 tenant_isolation 时记在哪个 stage（服务端发现的是 server，
        没有标记的是接口层按调用方身份拦下的 http）。"""
        self.http_requests.inc({"route": route, "method": method, "status": status})
        self.http_latency.observe({"route": route, "method": method}, seconds)
        if code is not None:
            self.http_errors.inc({"route": route, "code": code})
            if code == TENANT_ISOLATION_CODE:
                stage = _code(tenant_stage, default=TENANT_STAGE_HTTP)
                self.tenant_isolation.inc({"stage": stage})

    def observe_tool_cache(self, totals: Mapping[str, Any]) -> None:
        """缓存层的累计统计（{工具名: ToolCallStats}，rodit.tools.CachingToolGateway.totals()）
        → 按工具的计数器（整体替换）。"""
        for field, attr in TOOL_CACHE_FIELDS:
            counter = getattr(self, attr)
            counter.set_totals(
                ({"tool": _code(tool)}, report._as_int(getattr(stats, field, 0)))
                for tool, stats in totals.items()
            )

    def observe_readiness(self, readiness: Readiness) -> None:
        self.ready.set(value=1 if readiness.ready else 0)
        self.readiness_checks.replace(
            ({"check": _code(name)}, 1 if result.ok else 0)
            for name, result in readiness.checks.items()
        )

    def register_background(self, tasks: Iterable[str]) -> None:
        """预先建好各后台任务 error 的 0 值序列：第一次出错也能被 increase() 看到。"""
        for task in tasks:
            self.background_runs.inc({"task": _code(task), "outcome": "error"}, 0)

    def observe_background(self, task: str, outcome: str, *, at: float | None = None) -> None:
        """后台任务一轮的结果；成功时记下时间（Unix 秒，默认现在）。"""
        name = _code(task)
        self.background_runs.inc({"task": name, "outcome": _code(outcome)})
        if outcome == "ok":
            self.background_last_ok.set({"task": name}, time.time() if at is None else at)

    def forget_shop(self, shop_id: str) -> int:
        """取消订购（§8.4）后删掉该店铺的指标序列。"""
        return self.registry.forget("shop_id", shop_id)

    def render(self) -> str:
        return self.registry.render()


def _code_action(action: object) -> str:
    """审计 action（如 engine.result、aftersale.confirm）：小写字母、数字、下划线和点。"""
    if isinstance(action, str) and _ACTION.fullmatch(action):
        return action
    return "other"


def _identifier(value: object) -> str:
    """异常类型名（如 LLMTimeout）：ASCII 标识符，否则 "error"。"""
    if isinstance(value, str) and _IDENTIFIER.fullmatch(value):
        return value
    return "error"


# ================================================================ 审计装饰器


class ObservedAuditLog:
    """实现 core.ports.AuditLog：先转给真正的审计，再（无论成败）记指标。

    审计写入的异常照常抛出（调用方按原有规则降级，如引擎标记 "audit" 降级）；
    指标采集的异常只记日志。其余属性（实现特有的方法）原样转给 inner。
    """

    def __init__(self, inner: AuditLog, metrics: RoditMetrics) -> None:
        self._inner = inner
        self._metrics = metrics

    @property
    def inner(self) -> AuditLog:
        return self._inner

    def __repr__(self) -> str:
        return f"ObservedAuditLog(inner={self._inner!r})"

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name == "_inner":  # 构造完成前（copy 等）不要递归
            raise AttributeError(name)
        return getattr(self._inner, name)

    def record(self, event: AuditEvent) -> None:
        try:
            self._inner.record(event)
        except Exception:
            self._safely(lambda: self._metrics.audit_failures.inc({"kind": "event"}))
            raise
        finally:
            self._safely(lambda: self._metrics.observe_event(event))

    def log_decision(self, entry: DecisionLogEntry) -> None:
        try:
            self._inner.log_decision(entry)
        except Exception:
            self._safely(lambda: self._metrics.audit_failures.inc({"kind": "decision"}))
            raise

    def list_events(
        self, shop_id: str, *, action: str | None = None, since: datetime | None = None
    ) -> list[AuditEvent]:
        return self._inner.list_events(shop_id, action=action, since=since)

    def list_decisions(
        self,
        shop_id: str,
        *,
        stage: DecisionStage | None = None,
        since: datetime | None = None,
    ) -> list[DecisionLogEntry]:
        return self._inner.list_decisions(shop_id, stage=stage, since=since)

    def purge_shop(self, shop_id: str) -> int:
        count = self._inner.purge_shop(shop_id)
        self._safely(lambda: self._metrics.forget_shop(shop_id))
        return count

    @staticmethod
    def _safely(fn: Callable[[], object]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001  指标不能影响审计和问答
            logger.warning("指标采集失败：%s", type(exc).__name__)


# ================================================================ 平台工具装饰器


class MeteredToolGateway:
    """实现 core.ports.ToolGateway：转给平台原始工具，记调用次数、结果类别和耗时。

    结果类别：ok（有数据，列表可以为空）/ not_found（返回 None）/ ToolUnavailable 按它的 reason：
    unavailable（故障）、throttled（限流，额度用完）、timeout（超时）/ error（其他异常）。
    异常原样抛出，调用方（缓存层）按 §12.2 降级。
    """

    def __init__(
        self,
        inner: ToolGateway,
        metrics: RoditMetrics,
        *,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._inner = inner
        self._metrics = metrics
        self._timer = timer

    @property
    def platform(self) -> Platform:
        return self._inner.platform

    @property
    def inner(self) -> ToolGateway:
        return self._inner

    def __repr__(self) -> str:
        return f"MeteredToolGateway(inner={self._inner!r})"

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name == "_inner":  # 构造完成前（copy 等）不要递归
            raise AttributeError(name)
        return getattr(self._inner, name)

    def _call(self, tool: str, fn: Callable[[], Any]) -> Any:
        started = self._timer()
        outcome = "error"
        try:
            result = fn()
            outcome = "not_found" if result is None else "ok"
            return result
        except ToolUnavailable as exc:
            reason = getattr(exc, "reason", None)
            outcome = reason if reason in TOOL_FAILURE_REASONS else "unavailable"
            raise
        finally:
            elapsed = max(0.0, self._timer() - started)
            try:
                self._metrics.observe_tool(tool, outcome, elapsed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("指标采集失败：%s", type(exc).__name__)

    def get_order(self, shop_id: str, order_id: str) -> OrderInfo | None:
        return self._call("get_order", lambda: self._inner.get_order(shop_id, order_id))

    def list_recent_orders(
        self, shop_id: str, buyer_ref: str, *, limit: int = 5
    ) -> list[OrderInfo]:
        return self._call(
            "list_recent_orders",
            lambda: self._inner.list_recent_orders(shop_id, buyer_ref, limit=limit),
        )

    def get_logistics(self, shop_id: str, order_id: str) -> LogisticsTrace | None:
        return self._call("get_logistics", lambda: self._inner.get_logistics(shop_id, order_id))

    def get_aftersale(self, shop_id: str, aftersale_id: str) -> AfterSaleInfo | None:
        return self._call("get_aftersale", lambda: self._inner.get_aftersale(shop_id, aftersale_id))

    def list_aftersales(
        self,
        shop_id: str,
        *,
        order_id: str | None = None,
        buyer_ref: str | None = None,
        status: AfterSaleStatus | None = None,
        since: datetime | None = None,
    ) -> list[AfterSaleInfo]:
        return self._call(
            "list_aftersales",
            lambda: self._inner.list_aftersales(
                shop_id, order_id=order_id, buyer_ref=buyer_ref, status=status, since=since
            ),
        )

    def get_stock(self, shop_id: str, item_id: str, sku_id: str | None = None) -> StockInfo | None:
        return self._call("get_stock", lambda: self._inner.get_stock(shop_id, item_id, sku_id))

    def get_price(self, shop_id: str, item_id: str) -> PriceInfo | None:
        return self._call("get_price", lambda: self._inner.get_price(shop_id, item_id))


# ================================================================ 授权即将过期（§12.4）


class AuthExpiryMonitor:
    """按间隔检查本单元店铺的授权，写进 rodit_auth_expiring_shops / rodit_shop_auth_expiring，
    已经过期（到期时间已过、没有授权、需要重新授权）的店铺数写进 rodit_auth_expired_shops。

    - shop_ids：本单元服务的店铺（令牌文件里的店铺 + 平台云代码提供的订购列表，去掉已经取消订购
      清理过、没有再开通的店铺，见 rodit.ops.serve）；
    - serve 的后台线程按间隔调 refresh（rodit.ops.background），不挂在抓取 /metrics 的路径上
      （要查库）；距上次检查不到 refresh_s 秒时不查库（force 除外）；
    - 检查失败（数据库故障等）计入 rodit_auth_expiry_check_errors_total，保留上次的结果。
    """

    def __init__(
        self,
        guard: AuthGuard,
        shop_ids: Callable[[], Iterable[str]],
        metrics: RoditMetrics,
        *,
        within: timedelta = timedelta(days=7),
        refresh_s: float = 300.0,
        clock: Clock | None = None,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        if refresh_s < 0:
            raise ValueError("refresh_s 不能为负")
        self._guard = guard
        self._shop_ids = shop_ids
        self._metrics = metrics
        self._within = within
        self._refresh_s = refresh_s
        self._clock = clock or SystemClock()
        self._timer = timer
        self._lock = threading.Lock()
        self._last: float | None = None

    def refresh(self, *, force: bool = False) -> bool:
        """需要时检查一次；真的检查了返回 True。"""
        with self._lock:
            now = self._timer()
            if not force and self._last is not None and now - self._last < self._refresh_s:
                return False
            self._last = now
            try:
                shops = sorted({s for s in self._shop_ids() if isinstance(s, str) and s.strip()})
                expiring = self._guard.expiring(shops, within=self._within)
                # 已经过期的一定在即将过期的里面：只查这一小部分，不再把全部店铺查一遍
                expired = self._guard.expiring(expiring, within=timedelta(0)) if expiring else []
            except Exception as exc:  # noqa: BLE001  数据库故障等：保留上次结果
                self._metrics.auth_check_errors.inc()
                logger.warning("授权过期检查失败：%s", type(exc).__name__)
                return True
            self._metrics.auth_expiring.set(value=len(expiring))
            self._metrics.auth_expired.set(value=len(expired))
            self._metrics.auth_expiring_shop.replace(({"shop_id": s}, 1) for s in expiring)
            self._metrics.auth_checked.set(value=len(shops))
            self._metrics.auth_checked_at.set(value=self._clock.now().timestamp())
            return True


# ================================================================ 就绪（/readyz）


class ObservedReadinessChecker(ReadinessChecker):
    """就绪检查（rodit.ops.health.ReadinessChecker），每次的结果顺手记成 rodit_ready 和
    rodit_readiness_check_ok{check}。/readyz 和 serve 的后台线程刷新它（结果有缓存）。
    记指标失败只记日志，不影响就绪检查本身。

    publish 挂在 /metrics 渲染前：不跑探针（不查库），输出上次的结果和它的年龄
    （rodit_readiness_age_seconds）；还没检查过、或者上次的结果超过 stale_after_s 秒（后台刷新
    卡住了）时 rodit_ready 记 0。"""

    def __init__(
        self,
        probes: Mapping[str, Probe],
        metrics: RoditMetrics,
        *,
        cache_s: float = DEFAULT_CACHE_S,
        stale_after_s: float = DEFAULT_READINESS_STALE_S,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(probes, cache_s=cache_s, timer=timer)
        if stale_after_s <= 0:
            raise ValueError("stale_after_s 必须大于 0")
        self._metrics = metrics
        self._stale_after_s = stale_after_s
        self._last: tuple[float, Readiness] | None = None  # (检查完成的时刻, 结果)

    def check(self, *, fresh: bool = False) -> Readiness:
        result = super().check(fresh=fresh)
        cached = self._cached  # 父类的缓存：(检查完成的时刻, 结果)
        fresh_at = cached[0] if cached is not None and cached[1] is result else self._timer()
        self._last = (fresh_at, result)
        try:
            self._metrics.observe_readiness(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("指标采集失败：%s", type(exc).__name__)
        return result

    @property
    def last(self) -> Readiness | None:
        """上次检查的结果（还没检查过是 None）。不跑探针。"""
        last = self._last
        return None if last is None else last[1]

    def publish(self) -> None:
        """/metrics 渲染前：输出上次的结果和年龄，不跑探针（见类说明）。"""
        last = self._last
        if last is None:
            self._metrics.ready.set(value=0)
            return
        age = max(0.0, self._timer() - last[0])
        self._metrics.readiness_age.set(value=age)
        if age > self._stale_after_s:
            self._metrics.ready.set(value=0)
        else:
            self._metrics.observe_readiness(last[1])


# ================================================================ HTTP 中间件

ASGIApp = Callable[
    [MutableMapping[str, Any], Callable[[], Awaitable[Any]], Callable[[Any], Awaitable[None]]],
    Awaitable[None],
]

UNMATCHED_ROUTE = "unmatched"
_ERROR_BODY_LIMIT = 4096


class HttpMetricsMiddleware:
    """纯 ASGI 中间件：按路由模板记请求数、耗时、错误码。

    路由标签用路由模板（如 /v1/shops/{shop_id}/standalone/ask），不用实际路径：路径里有店铺 ID、
    售后单号，而且没匹配上的路径（扫描器）只记成 "unmatched"，标签值有界。
    状态码 ≥ 400 且是 JSON 时读响应体里的 "error" 字段（接口层统一的错误码），不读别的内容。
    """

    def __init__(self, app: ASGIApp, *, metrics: RoditMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[Any]],
        send: Callable[[Any], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        state: dict[str, Any] = {"status": 500, "json": False, "body": b"", "code": None}
        # 请求状态先建好：下层即使复制了 scope，拿到的也是同一个 dict（服务端串店的标记写在这里）
        if not isinstance(scope.get("state"), MutableMapping):
            scope["state"] = {}

        async def send_wrapper(message: Mapping[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                state["status"] = int(message.get("status", 500))
                headers = dict(message.get("headers") or [])
                content_type = headers.get(b"content-type", b"")
                state["json"] = content_type.startswith(b"application/json")
            elif (
                message.get("type") == "http.response.body"
                and state["status"] >= 400
                and state["json"]
                and len(state["body"]) < _ERROR_BODY_LIMIT
            ):
                state["body"] += bytes(message.get("body", b""))[:_ERROR_BODY_LIMIT]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # 未处理的异常：500 响应由外层的 ServerErrorMiddleware 生成，这里看不到响应体
            state["unhandled"] = True
            raise
        finally:
            try:
                self._observe(scope, state, time.perf_counter() - started)
            except Exception as exc:  # noqa: BLE001
                logger.warning("指标采集失败：%s", type(exc).__name__)

    def _observe(self, scope: Mapping[str, Any], state: Mapping[str, Any], seconds: float) -> None:
        route_label = _route_label(scope, state["status"])
        # 方法名来自客户端（h11 接受任意记号）：只认标准方法，其余归 other，标签值有界
        raw_method = str(scope.get("method", "")).upper()
        method = raw_method if raw_method in _HTTP_METHODS else "other"
        code = None
        if state.get("unhandled"):
            code = "internal_error"  # 和接口层对未预期异常的错误码一致
        elif state["status"] >= 400:
            code = _error_code(state["body"]) if state["json"] else None
            code = code or f"http_{state['status']}"
        request_state = scope.get("state")
        tenant_stage = None
        if isinstance(request_state, Mapping):
            tenant_stage = request_state.get(TENANT_STAGE_STATE_KEY)
        self.metrics.observe_http(
            route_label, method, state["status"], seconds, code, tenant_stage=tenant_stage
        )


_HTTP_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


def _route_label(scope: Mapping[str, Any], status: int) -> str:
    """路由模板作标签。FastAPI 只给 APIRoute 设 scope["route"]；它自己加的 /docs、/openapi.json
    这类 Starlette 路由匹配上了也没有，成功的请求按应用的路由表再匹配一次拿模板。
    没匹配上的（扫描器乱试的路径）一律记 unmatched。"""
    path = getattr(scope.get("route"), "path", None)
    if isinstance(path, str) and path:
        return path
    if status < 400:
        for candidate in getattr(scope.get("app"), "routes", ()):
            try:
                match, _ = candidate.matches(scope)
            except Exception:  # noqa: BLE001 — 匹配失败就当没匹配上
                continue
            template = getattr(candidate, "path", None)
            if match == Match.FULL and isinstance(template, str) and template:
                return template
    return UNMATCHED_ROUTE


def mark_server_tenant_isolation(app: Any) -> None:
    """给接口层 TenantIsolationError 的错误映射包一层（响应不变，仍是 403 tenant_isolation）：
    在请求状态里标上 stage=server，HttpMetricsMiddleware 据此把它记成
    rodit_tenant_isolation_total{stage="server"}（critical 告警），不和调用方令牌不对的
    stage="http"（warning）混在一起；日志记路由模板和路径里的店铺 ID（不记异常说明）。

    在接口层注册完错误映射之后、应用处理第一个请求之前调用（rodit.ops.serve.build_server）。"""
    original = app.exception_handlers.get(TenantIsolationError)
    if original is None:
        raise RuntimeError("接口层没有注册 TenantIsolationError 的错误映射")

    async def handler(request: Any, exc: Exception) -> Any:
        try:
            setattr(request.state, TENANT_STAGE_STATE_KEY, TENANT_STAGE_SERVER)
            route = getattr(request.scope.get("route"), "path", None) or UNMATCHED_ROUTE
            shop = safe_label_value(request.path_params.get("shop_id"))
            logger.error("接口处理中发现串店（§9.2，已拒绝）：route=%s shop_id=%s", route, shop)
        except Exception as err:  # noqa: BLE001  标记失败不影响响应
            logger.warning("指标采集失败：%s", type(err).__name__)
        response = original(request, exc)
        return await response if inspect.isawaitable(response) else response

    app.add_exception_handler(TenantIsolationError, handler)


def _error_code(body: bytes) -> str | None:
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    value = parsed.get("error") if isinstance(parsed, dict) else None
    return _code(value, default="") or None


__all__ = [
    "LATENCY_BUCKETS_S",
    "TENANT_STAGE_HTTP",
    "TENANT_STAGE_SERVER",
    "TENANT_STAGE_UNKNOWN",
    "TOOL_BUCKETS_S",
    "UNMATCHED_ROUTE",
    "AuthExpiryMonitor",
    "HttpMetricsMiddleware",
    "MeteredToolGateway",
    "ObservedAuditLog",
    "ObservedReadinessChecker",
    "RoditMetrics",
    "mark_server_tenant_isolation",
    "package_version",
]
