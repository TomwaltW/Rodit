"""串店、限流、缓存层统计、从收到消息算起的耗时、就绪、授权已过期的指标（方案 §9.2、§12.1、
§12.2、§12.4）。全部由合成审计事件和测试替身驱动，标签里没有买家文本。"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from ops_support import FakeReadinessDb, TestClient, label_values, parse_exposition, sample

from rodit.adapters import AuthGuard
from rodit.app.demo import seed_unit
from rodit.app.unit import in_memory_repos
from rodit.core.audit import (
    DEGRADED_TENANT_ISOLATION,
    LATENCY_SINCE_RECEIVED,
    AuditAction,
    InMemoryAuditSink,
    ResultKey,
)
from rodit.core.enums import Platform
from rodit.core.errors import TenantIsolationError, ToolUnavailable
from rodit.core.models import AuditEvent, ShopAuth
from rodit.inmemory import InMemoryCache, InMemoryShopAuthRepository
from rodit.ops.health import CheckResult, ReadinessChecker
from rodit.ops.instrument import (
    AuthExpiryMonitor,
    MeteredToolGateway,
    ObservedAuditLog,
    ObservedReadinessChecker,
    RoditMetrics,
)
from rodit.ops.serve import build_server
from rodit.ops.settings import ServerSettings
from rodit.storage.db import MIGRATIONS
from rodit.testing.fakes import FakeClock, InMemoryToolGateway
from rodit.testing.fixtures import SHOP_A, SHOP_B, build_demo_world
from rodit.tools import CachingToolGateway

PII_TEXT = "买家手机13812345678，寄到北京市朝阳区建国路88号"


@pytest.fixture
def metrics() -> RoditMetrics:
    return RoditMetrics(platform=Platform.TAOBAO, mode="platform", version="0.0.1")


def _samples(metrics: RoditMetrics) -> dict:
    return parse_exposition(metrics.render())


def _event(action: str, detail: dict[str, Any] | None = None) -> AuditEvent:
    return AuditEvent(
        shop_id=SHOP_A,
        actor="system",
        action=action,
        detail=detail or {},
        created_at=build_demo_world().now,
    )


def _violation(**detail: Any) -> AuditEvent:
    return _event(AuditAction.TENANT_VIOLATION.value, detail)


def _result(**detail: Any) -> AuditEvent:
    base: dict[str, Any] = {
        ResultKey.INTENT.value: "logistics",
        ResultKey.ACTION.value: "answer",
        ResultKey.SUGGESTIONS.value: 0,
        ResultKey.DEGRADED.value: [],
        ResultKey.TRANSFER.value: False,
        ResultKey.LATENCY.value: {"total": 1200},
    }
    base.update(detail)
    return _event(AuditAction.RESULT.value, base)


# ================================================================ 串店


def test_tenant_violation_events_count_by_stage(metrics):
    audit = ObservedAuditLog(InMemoryAuditSink(), metrics)
    before = _samples(metrics)
    # 0 值序列一开始就有：第一次串店也能被 increase() 看到
    assert sample(before, "rodit_tenant_isolation_total", stage="http") == 0
    assert sample(before, "rodit_degradations_total", kind=DEGRADED_TENANT_ISOLATION) == 0

    audit.record(_violation(stage="retrieve"))
    audit.record(_violation(stage="retrieve"))
    audit.record(_violation(stage="understand"))
    audit.record(_violation())  # 没写 stage
    audit.record(_violation(stage=PII_TEXT))  # 不是代码
    # 同一次拦截的 engine.result：降级类型另记，不重复计入串店次数
    audit.record(_result(**{ResultKey.DEGRADED.value: [DEGRADED_TENANT_ISOLATION]}))

    s = _samples(metrics)
    assert sample(s, "rodit_tenant_isolation_total", stage="retrieve") == 2
    assert sample(s, "rodit_tenant_isolation_total", stage="understand") == 1
    assert sample(s, "rodit_tenant_isolation_total", stage="unknown") == 2
    total = sum(v for (name, _), v in s.items() if name == "rodit_tenant_isolation_total")
    assert total == 5
    assert sample(s, "rodit_degradations_total", kind=DEGRADED_TENANT_ISOLATION) == 1
    assert sample(s, "rodit_audit_events_total", action="tenant.violation") == 5
    text = metrics.render()
    assert "13812345678" not in text and "建国路" not in text


def test_tenant_violation_is_counted_even_when_the_audit_write_fails(metrics):
    class Down(InMemoryAuditSink):
        def record(self, event: AuditEvent) -> None:
            raise OSError("审计库挂了")

    audit = ObservedAuditLog(Down(), metrics)
    with pytest.raises(OSError):
        audit.record(_violation(stage="retrieve"))
    s = _samples(metrics)
    assert sample(s, "rodit_tenant_isolation_total", stage="retrieve") == 1


def test_http_tenant_errors_also_count(metrics):
    metrics.observe_http(
        "/v1/shops/{shop_id}/standalone/ask", "POST", 403, 0.01, "tenant_isolation"
    )
    metrics.observe_http("/v1/shops/{shop_id}/standalone/ask", "POST", 403, 0.01, "forbidden_role")
    s = _samples(metrics)
    assert sample(s, "rodit_tenant_isolation_total", stage="http") == 1


# ================================================================ 平台接口：限流 / 故障 / 超时


class _Failing(InMemoryToolGateway):
    """按方法名抛指定原因的 ToolUnavailable。"""

    def __init__(self, world, reasons: dict[str, str | None]) -> None:
        super().__init__(world, Platform.TAOBAO)
        self.reasons = reasons

    def _enter(self, name: str, shop_id: str, *args: Any) -> None:
        super()._enter(name, shop_id, *args)
        if name in self.reasons:
            reason = self.reasons[name]
            if reason is None:
                raise ToolUnavailable(f"{name} 故障")  # 默认原因
            raise ToolUnavailable(f"{name} 失败", reason=reason)  # type: ignore[arg-type]


def test_tool_failures_are_split_by_reason(metrics):
    world = build_demo_world()
    inner = _Failing(
        world, {"get_order": "throttled", "get_logistics": "timeout", "get_price": None}
    )
    tools = MeteredToolGateway(inner, metrics)
    for call in (
        lambda: tools.get_order(SHOP_A, "o_1001"),
        lambda: tools.get_logistics(SHOP_A, "o_1002"),
        lambda: tools.get_price(SHOP_A, "i_555"),
    ):
        with pytest.raises(ToolUnavailable):
            call()
    s = _samples(metrics)
    tool_calls = "rodit_platform_tool_calls_total"
    assert sample(s, tool_calls, tool="get_order", outcome="throttled") == 1
    assert sample(s, tool_calls, tool="get_logistics", outcome="timeout") == 1
    assert sample(s, tool_calls, tool="get_price", outcome="unavailable") == 1
    assert sample(s, "rodit_platform_tool_latency_seconds_count", tool="get_order") == 1


# ================================================================ 缓存层的统计


class _Leaky(InMemoryToolGateway):
    """查店铺 A 的订单时返回店铺 B 的订单（模拟平台接口串店）。"""

    def get_order(self, shop_id: str, order_id: str):
        order = super().get_order(shop_id, order_id)
        if order_id == "o_leak":
            other = next(o for o in self.world.orders if o.shop_id == SHOP_B)
            return other
        return order


def test_caching_gateway_totals_become_counters(metrics):
    world = build_demo_world()
    clock = FakeClock(world.now)
    inner = _Leaky(world, Platform.TAOBAO)
    gateway = CachingToolGateway(
        MeteredToolGateway(inner, metrics), InMemoryCache(clock), clock, ttl_s=60
    )
    gateway.get_order(SHOP_A, "o_1001")  # 调平台
    gateway.get_order(SHOP_A, "o_1001")  # 命中缓存
    gateway.get_order(SHOP_B, "o_9001")  # 另一家店：按工具相加
    clock.advance(seconds=120)
    inner.fail_on.add("get_order")
    gateway.get_order(SHOP_A, "o_1001")  # 平台限流 → 过期缓存兜底
    inner.fail_on.clear()
    with pytest.raises(TenantIsolationError):
        gateway.get_order(SHOP_A, "o_leak")
    totals = gateway.totals()
    assert set(totals) == {"get_order"}
    assert (
        totals["get_order"].requests,
        totals["get_order"].cache_hits,
        totals["get_order"].stale_served,
        totals["get_order"].tenant_violations,
        totals["get_order"].failures,
    ) == (5, 1, 1, 1, 1)
    # 快照：改它不影响网关里的统计
    totals["get_order"].requests = 999
    assert gateway.totals()["get_order"].requests == 5

    metrics.observe_tool_cache(gateway.totals())
    metrics.observe_tool_cache(gateway.totals())  # 抓取两次：整体替换，不会翻倍
    s = _samples(metrics)
    assert sample(s, "rodit_tool_cache_requests_total", tool="get_order") == 5
    assert sample(s, "rodit_tool_cache_hits_total", tool="get_order") == 1
    assert sample(s, "rodit_tool_cache_stale_served_total", tool="get_order") == 1
    assert sample(s, "rodit_tool_cache_tenant_violations_total", tool="get_order") == 1
    assert sample(s, "rodit_platform_tool_calls_total", tool="get_order", outcome="throttled") == 1
    # 按工具，不按店铺：标签里没有店铺 ID
    assert SHOP_A not in label_values(
        {k: v for k, v in s.items() if k[0].startswith("rodit_tool_cache_")}
    )


# ================================================================ 从收到消息算起的耗时


def test_since_received_is_a_latency_stage(metrics):
    audit = ObservedAuditLog(InMemoryAuditSink(), metrics)
    latency = {"route": 300, "total": 1800, LATENCY_SINCE_RECEIVED: 6200}
    audit.record(_result(**{ResultKey.LATENCY.value: latency}))
    audit.record(_result(**{ResultKey.LATENCY.value: {"total": 900, LATENCY_SINCE_RECEIVED: -5}}))
    s = _samples(metrics)
    hist = "rodit_engine_latency_seconds"
    assert sample(s, f"{hist}_count", stage=LATENCY_SINCE_RECEIVED) == 1  # 负数（时钟偏差）不记
    assert sample(s, f"{hist}_sum", stage=LATENCY_SINCE_RECEIVED) == pytest.approx(6.2)
    assert sample(s, f"{hist}_bucket", stage=LATENCY_SINCE_RECEIVED, le="5") == 0  # 超过 5 秒
    assert sample(s, f"{hist}_bucket", stage=LATENCY_SINCE_RECEIVED, le="7.5") == 1
    assert sample(s, f"{hist}_count", stage="total") == 2


# ================================================================ 就绪、授权已过期


def test_readiness_results_become_gauges(metrics):
    state = {"ok": True}

    def probe() -> dict[str, CheckResult]:
        return {"database": CheckResult(state["ok"], {}), "llm": CheckResult(True, {})}

    checker = ObservedReadinessChecker({"db": probe}, metrics, cache_s=0)
    assert isinstance(checker, ReadinessChecker)
    assert checker.check().ready
    s = _samples(metrics)
    assert sample(s, "rodit_ready") == 1
    assert sample(s, "rodit_readiness_check_ok", check="database") == 1
    state["ok"] = False
    assert not checker.check().ready
    s = _samples(metrics)
    assert sample(s, "rodit_ready") == 0
    assert sample(s, "rodit_readiness_check_ok", check="database") == 0
    assert sample(s, "rodit_readiness_check_ok", check="llm") == 1


def test_auth_monitor_reports_expired_shops_separately(metrics):
    world = build_demo_world()
    clock = FakeClock(world.now)
    repo = InMemoryShopAuthRepository()
    repo.save(
        ShopAuth(
            shop_id=SHOP_A,
            access_token="synthetic-token",
            scopes=["order.read"],
            expires_at=world.now + timedelta(days=3),  # 即将过期
        )
    )
    repo.save(
        ShopAuth(
            shop_id=SHOP_B,
            access_token="synthetic-token",
            scopes=["order.read"],
            expires_at=world.now - timedelta(hours=1),  # 已经过期
        )
    )
    monitor = AuthExpiryMonitor(
        AuthGuard(repo, clock),
        lambda: [SHOP_A, SHOP_B, "s_9999"],  # s_9999 没有授权
        metrics,
        within=timedelta(days=7),
        refresh_s=0,
        clock=clock,
    )
    assert monitor.refresh()
    s = _samples(metrics)
    assert sample(s, "rodit_auth_expiring_shops") == 3
    assert sample(s, "rodit_auth_expired_shops") == 2
    clock.advance(days=4)
    monitor.refresh()
    assert sample(_samples(metrics), "rodit_auth_expired_shops") == 3


# ================================================================ 部署入口：/metrics 里都有


@pytest.fixture
def local_server(ops_env, token_file, monkeypatch):
    """本地试用模式：合成平台绑定 + 离线替身 + 内存仓储（不连数据库）。"""
    for name in ("RODIT_LLM_BASE_URL", "RODIT_LLM_MODEL", "RODIT_LLM_API_KEY"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("RODIT_ENV", "local")
    world = build_demo_world()
    clock = FakeClock(world.now)
    server = build_server(
        ServerSettings.from_env(),
        clock=clock,
        repos=in_memory_repos(clock),
        database=FakeReadinessDb(tuple(m.version for m in MIGRATIONS)),
    )
    seed_unit(server.unit, world)
    return server, TestClient(server.app), token_file


def test_served_metrics_include_tenant_quota_cache_and_readiness(local_server):
    server, client, tokens = local_server
    unit = server.unit
    unit.gateway.get_order(SHOP_A, "o_1001")
    unit.gateway.get_order(SHOP_A, "o_1001")
    unit.repos.audit.record(_violation(stage="retrieve"))
    r = client.post(
        f"/v1/shops/{SHOP_B}/standalone/ask",
        json={"session_id": "web_1", "question": PII_TEXT},
        headers=tokens.auth("agent_a"),
    )
    assert r.status_code == 403
    text = client.get("/metrics", headers=tokens.auth("metrics")).text
    s = parse_exposition(text)
    assert sample(s, "rodit_tenant_isolation_total", stage="retrieve") == 1
    assert sample(s, "rodit_tenant_isolation_total", stage="http") == 1
    assert sample(s, "rodit_tool_cache_requests_total", tool="get_order") == 2
    assert sample(s, "rodit_tool_cache_hits_total", tool="get_order") == 1
    assert sample(s, "rodit_ready") == 1
    assert sample(s, "rodit_readiness_check_ok", check="database") == 1
    assert sample(s, "rodit_auth_expired_shops") == 0
    for leaked in ("13812345678", "建国路", "web_1"):
        assert leaked not in text
