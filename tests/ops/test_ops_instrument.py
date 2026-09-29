"""运行指标的采集（方案 §12.4 质量与技术指标）：审计装饰器、平台工具装饰器、授权过期检查、
HTTP 中间件。标签里只有代码，没有买家数据。"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from ops_support import TestClient, label_values, parse_exposition, sample

from rodit.adapters import AuthGuard
from rodit.api.errors import install_error_handlers, tenant_violation
from rodit.app.offline import EvidenceEchoLLM, HeuristicQAJudge
from rodit.app.unit import build_unit, in_memory_repos
from rodit.core.audit import AuditAction, InMemoryAuditSink, ResultKey
from rodit.core.clock import SystemClock
from rodit.core.config import EngineConfig
from rodit.core.enums import DecisionStage, Platform
from rodit.core.errors import ToolUnavailable
from rodit.core.models import AuditEvent, DecisionLogEntry, ShopAuth
from rodit.core.pii import BuyerRefHasher
from rodit.core.ports import AuditLog, LLMRequest, LLMResponse, ToolGateway
from rodit.decision import RuleProvider
from rodit.inmemory import InMemoryShopAuthRepository
from rodit.llm import MeteredLLMClient
from rodit.ops.instrument import (
    AuthExpiryMonitor,
    HttpMetricsMiddleware,
    MeteredToolGateway,
    ObservedAuditLog,
    RoditMetrics,
)
from rodit.testing.fakes import FakeClock, InMemoryToolGateway
from rodit.testing.fixtures import SHOP_A, SHOP_B, TEST_SALT, build_demo_world


@pytest.fixture
def metrics() -> RoditMetrics:
    return RoditMetrics(platform=Platform.TAOBAO, mode="platform", version="0.0.1")


def _samples(metrics: RoditMetrics) -> dict:
    return parse_exposition(metrics.render())


def _result_event(shop_id: str | None = SHOP_A, **detail: Any) -> AuditEvent:
    base: dict[str, Any] = {
        ResultKey.INTENT.value: "product_spec",
        ResultKey.ACTION.value: "answer",
        ResultKey.SUGGESTIONS.value: 2,
        ResultKey.QA_CHECKED.value: 2,
        ResultKey.QA_FAILED.value: 1,
        ResultKey.HARD_HITS.value: ["time_promise", "length"],
        ResultKey.DEGRADED.value: ["decision", "tool:logistics"],
        ResultKey.TRANSFER.value: False,
        ResultKey.LATENCY.value: {"route": 300, "generate": 1200, "total": 1800},
    }
    base.update(detail)
    return AuditEvent(
        shop_id=shop_id,
        actor="system",
        action=AuditAction.RESULT.value,
        detail=base,
        created_at=build_demo_world().now,
    )


def test_build_info(metrics):
    s = _samples(metrics)
    assert sample(s, "rodit_build_info", platform="taobao", mode="platform", version="0.0.1") == 1


def test_engine_result_becomes_metrics(metrics):
    audit = ObservedAuditLog(InMemoryAuditSink(), metrics)
    audit.record(_result_event())
    audit.record(_result_event(**{ResultKey.SUGGESTIONS.value: 0, ResultKey.DEGRADED.value: True}))
    audit.record(_result_event(**{ResultKey.TRANSFER.value: True}))
    s = _samples(metrics)
    assert sample(s, "rodit_engine_messages_total", shop_id=SHOP_A, outcome="suggested") == 1
    assert sample(s, "rodit_engine_messages_total", shop_id=SHOP_A, outcome="no_suggestion") == 1
    assert (
        sample(s, "rodit_engine_messages_total", shop_id=SHOP_A, outcome="transfer_to_human") == 1
    )
    assert sample(s, "rodit_engine_routes_total", intent="product_spec", action="answer") == 3
    assert sample(s, "rodit_suggestions_total", shop_id=SHOP_A) == 4
    assert sample(s, "rodit_qa_checks_total") == 6
    assert sample(s, "rodit_qa_failures_total") == 3
    assert sample(s, "rodit_qa_hard_rule_hits_total", rule="time_promise", blocking="true") == 3
    assert sample(s, "rodit_qa_hard_rule_hits_total", rule="length", blocking="false") == 3
    assert sample(s, "rodit_degradations_total", kind="decision") == 2
    assert sample(s, "rodit_degradations_total", kind="tool:logistics") == 2
    assert sample(s, "rodit_degradations_total", kind="unknown") == 1
    assert sample(s, "rodit_engine_latency_seconds_count", stage="total") == 3
    assert sample(s, "rodit_engine_latency_seconds_bucket", stage="total", le="2") == 3
    assert sample(s, "rodit_engine_latency_seconds_bucket", stage="total", le="1.5") == 0
    assert sample(s, "rodit_engine_latency_seconds_sum", stage="generate") == pytest.approx(3.6)
    assert sample(s, "rodit_audit_events_total", action="engine.result") == 3


def test_free_text_in_result_detail_never_becomes_a_label(metrics):
    """质检原因、缺失知识是中文说明，不做标签；意图、规则、降级不认识的值记成 other。"""
    audit = ObservedAuditLog(InMemoryAuditSink(), metrics)
    audit.record(
        _result_event(
            **{
                ResultKey.INTENT.value: "买家说 13812345678",
                ResultKey.ACTION.value: "北京市朝阳区建国路88号",
                ResultKey.HARD_HITS.value: [{"rule": "加我微信 abc123456"}],
                ResultKey.DEGRADED.value: ["收件人 张三"],
                ResultKey.QA_REASONS.value: ["回答里承诺了明天到（买家 13812345678）"],
                ResultKey.MISSING.value: ["退货地址"],
                ResultKey.LATENCY.value: {"阶段": 10, "total": 20},
            }
        )
    )
    s = _samples(metrics)
    values = label_values(s)
    assert not any(ch > "\u007f" for v in values for ch in v)
    assert "13812345678" not in metrics.render()
    assert sample(s, "rodit_engine_routes_total", intent="other", action="other") == 1
    assert sample(s, "rodit_qa_hard_rule_hits_total", rule="other", blocking="true") == 1
    assert sample(s, "rodit_degradations_total", kind="other") == 1


def test_missing_keys_are_tolerated(metrics):
    audit = ObservedAuditLog(InMemoryAuditSink(), metrics)
    event = _result_event()
    audit.record(event.model_copy(update={"detail": {}, "shop_id": None}))
    s = _samples(metrics)
    assert sample(s, "rodit_engine_messages_total", shop_id="none", outcome="no_suggestion") == 1
    assert sample(s, "rodit_engine_routes_total", intent="none", action="none") == 1


def test_llm_usage_tokens_per_shop(metrics):
    audit = ObservedAuditLog(InMemoryAuditSink(), metrics)
    clock = FakeClock(build_demo_world().now)

    class Model:
        model = "synthetic"

        def __init__(self) -> None:
            self.fail = False

        def complete(self, request: LLMRequest) -> LLMResponse:
            if self.fail:
                from rodit.core.errors import LLMTimeout

                raise LLMTimeout("超时")
            return LLMResponse(text="{}", model="synthetic", input_tokens=120, output_tokens=30)

    inner = Model()
    llm = MeteredLLMClient(inner, audit, clock, timer=iter([0.0, 0.4, 1.0, 1.1, 2.0, 4.5]).__next__)
    msg = [{"role": "user", "content": "我的手机号 13812345678"}]
    llm.complete(LLMRequest(purpose="generate", messages=msg, shop_id=SHOP_A))
    llm.complete(LLMRequest(purpose="route", messages=msg, shop_id=SHOP_B))
    inner.fail = True
    with pytest.raises(Exception, match="超时"):
        llm.complete(LLMRequest(purpose="generate", messages=msg, shop_id=SHOP_A))
    s = _samples(metrics)
    assert (
        sample(s, "rodit_llm_tokens_total", shop_id=SHOP_A, purpose="generate", direction="input")
        == 120
    )
    assert (
        sample(s, "rodit_llm_tokens_total", shop_id=SHOP_A, purpose="generate", direction="output")
        == 30
    )
    assert (
        sample(s, "rodit_llm_tokens_total", shop_id=SHOP_B, purpose="route", direction="input")
        == 120
    )
    assert sample(s, "rodit_llm_calls_total", purpose="generate", outcome="ok") == 1
    assert sample(s, "rodit_llm_calls_total", purpose="generate", outcome="LLMTimeout") == 1
    assert sample(s, "rodit_llm_latency_seconds_count", purpose="generate") == 2
    assert "13812345678" not in metrics.render()


def test_audit_decorator_forwards_everything(metrics):
    inner = InMemoryAuditSink()
    audit = ObservedAuditLog(inner, metrics)
    assert isinstance(audit, AuditLog)
    event = _result_event()
    audit.record(event)
    entry = DecisionLogEntry(
        shop_id=SHOP_A,
        stage=DecisionStage.ROUTE,
        provider="rules",
        questions_ver="v1",
        answers={},
        created_at=event.created_at,
    )
    audit.log_decision(entry)
    assert audit.list_events(SHOP_A) == [event]
    assert audit.list_events(SHOP_A, action="other") == []
    assert audit.list_decisions(SHOP_A, stage=DecisionStage.ROUTE) == [entry]
    assert audit.actions(SHOP_A) == [AuditAction.RESULT.value]  # 实现特有的方法照样能用
    assert audit.inner is inner
    import copy

    assert copy.copy(audit).inner is inner  # 浅拷贝不会在 __getattr__ 里无限递归


def test_audit_write_failure_still_raises_and_is_counted(metrics):
    class Broken(InMemoryAuditSink):
        def record(self, event: AuditEvent) -> None:
            raise OSError("磁盘满")

        def log_decision(self, entry: DecisionLogEntry) -> None:
            raise OSError("磁盘满")

    audit = ObservedAuditLog(Broken(), metrics)
    with pytest.raises(OSError):
        audit.record(_result_event())
    s = _samples(metrics)
    assert sample(s, "rodit_audit_write_failures_total", kind="event") == 1
    # 事件确实发生了：指标照记
    assert sample(s, "rodit_engine_messages_total", shop_id=SHOP_A, outcome="suggested") == 1


def test_metrics_failure_never_breaks_audit(metrics, monkeypatch):
    inner = InMemoryAuditSink()
    audit = ObservedAuditLog(inner, metrics)

    def boom(event: AuditEvent) -> None:
        raise RuntimeError("指标坏了")

    monkeypatch.setattr(metrics, "observe_event", boom)
    audit.record(_result_event())
    assert len(inner.events) == 1


def test_purge_shop_also_forgets_shop_metrics(metrics):
    audit = ObservedAuditLog(InMemoryAuditSink(), metrics)
    audit.record(_result_event(SHOP_A))
    audit.record(_result_event(SHOP_B))
    assert audit.purge_shop(SHOP_A) == 1
    values = label_values(_samples(metrics))
    assert SHOP_A not in values and SHOP_B in values


# ================================================================ 平台工具


def test_tool_gateway_outcomes(metrics):
    world = build_demo_world()
    inner = InMemoryToolGateway(world, Platform.TAOBAO)
    tools = MeteredToolGateway(inner, metrics)
    assert isinstance(tools, ToolGateway)
    assert tools.platform == Platform.TAOBAO
    order = world.orders[0]
    assert tools.get_order(order.shop_id, order.order_id) is not None
    assert tools.get_order(SHOP_A, "o_missing") is None
    assert tools.list_aftersales(SHOP_A) is not None
    inner.fail_on.add("get_logistics")
    with pytest.raises(ToolUnavailable):
        tools.get_logistics(SHOP_A, order.order_id)
    s = _samples(metrics)
    assert sample(s, "rodit_platform_tool_calls_total", tool="get_order", outcome="ok") == 1
    assert sample(s, "rodit_platform_tool_calls_total", tool="get_order", outcome="not_found") == 1
    assert sample(s, "rodit_platform_tool_calls_total", tool="list_aftersales", outcome="ok") == 1
    assert (
        sample(s, "rodit_platform_tool_calls_total", tool="get_logistics", outcome="unavailable")
        == 1
    )
    assert sample(s, "rodit_platform_tool_latency_seconds_count", tool="get_order") == 2
    assert tools.calls  # 其余属性转给原始实现


def test_tool_gateway_unexpected_error(metrics):
    class Exploding(InMemoryToolGateway):
        def get_price(self, shop_id: str, item_id: str):
            raise KeyError(item_id)

    tools = MeteredToolGateway(Exploding(build_demo_world()), metrics)
    with pytest.raises(KeyError):
        tools.get_price(SHOP_A, "i_1")
    s = _samples(metrics)
    assert sample(s, "rodit_platform_tool_calls_total", tool="get_price", outcome="error") == 1


def test_metered_tools_plug_into_a_unit(metrics):
    """缓存层之下只记真正打到平台的调用：同一订单第二次查询走缓存，不计数。"""
    world = build_demo_world()
    clock = FakeClock(world.now)
    unit = build_unit(
        platform=Platform.TAOBAO,
        config=EngineConfig(),
        clock=clock,
        hasher=BuyerRefHasher(TEST_SALT),
        repos=in_memory_repos(clock),
        llm=EvidenceEchoLLM(),
        tools=MeteredToolGateway(InMemoryToolGateway(world, Platform.TAOBAO), metrics),
        adapters=[],
        route_provider=RuleProvider(),
        qa_provider=HeuristicQAJudge(),
    )
    order = world.orders[0]
    unit.gateway.get_order(order.shop_id, order.order_id)
    unit.gateway.get_order(order.shop_id, order.order_id)
    s = _samples(metrics)
    assert sample(s, "rodit_platform_tool_calls_total", tool="get_order", outcome="ok") == 1


# ================================================================ 授权即将过期


class _Timer:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_auth_expiry_monitor(metrics):
    world = build_demo_world()
    clock = FakeClock(world.now)
    repo = InMemoryShopAuthRepository()
    for shop_id, days in ((SHOP_A, 30), (SHOP_B, 3)):
        repo.save(
            ShopAuth(
                shop_id=shop_id,
                access_token="synthetic-token",
                scopes=["order.read"],
                expires_at=world.now + timedelta(days=days),
            )
        )
    timer = _Timer()
    shops = [SHOP_A, SHOP_B, "s_9999"]  # s_9999 没有授权
    monitor = AuthExpiryMonitor(
        AuthGuard(repo, clock),
        lambda: shops,
        metrics,
        within=timedelta(days=7),
        refresh_s=60,
        clock=clock,
        timer=timer,
    )
    metrics.registry.before_render(monitor.refresh)
    s = _samples(metrics)
    assert sample(s, "rodit_auth_expiring_shops") == 2
    assert sample(s, "rodit_shop_auth_expiring", shop_id=SHOP_B) == 1
    assert sample(s, "rodit_shop_auth_expiring", shop_id="s_9999") == 1
    assert sample(s, "rodit_shop_auth_expiring", shop_id=SHOP_A) is None
    assert sample(s, "rodit_auth_checked_shops") == 3
    assert sample(s, "rodit_auth_expiry_last_check_timestamp_seconds") == world.now.timestamp()

    # 间隔内不重新查库
    clock.advance(days=25)
    timer.now = 30
    assert monitor.refresh() is False
    assert sample(_samples(metrics), "rodit_auth_expiring_shops") == 2
    # 过了间隔：店铺 A 也进了告警窗口
    timer.now = 61
    s = _samples(metrics)
    assert sample(s, "rodit_auth_expiring_shops") == 3
    assert sample(s, "rodit_shop_auth_expiring", shop_id=SHOP_A) == 1


def test_auth_expiry_monitor_keeps_last_result_on_error(metrics):
    clock = SystemClock()

    class BrokenRepo(InMemoryShopAuthRepository):
        broken = False

        def get(self, shop_id: str):
            if self.broken:
                raise OSError("数据库挂了")
            return super().get(shop_id)

    repo = BrokenRepo()
    monitor = AuthExpiryMonitor(
        AuthGuard(repo, clock), lambda: [SHOP_A], metrics, refresh_s=0, clock=clock
    )
    assert monitor.refresh() is True
    repo.broken = True
    assert monitor.refresh() is True
    s = _samples(metrics)
    assert sample(s, "rodit_auth_expiring_shops") == 1  # 上次的结果
    assert sample(s, "rodit_auth_expiry_check_errors_total") == 1


# ================================================================ HTTP 中间件


def _http_app(metrics: RoditMetrics) -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/v1/shops/{shop_id}/things/{thing_id}")
    def thing(shop_id: str, thing_id: str) -> dict[str, str]:
        if shop_id == "s_other":
            raise tenant_violation()
        if thing_id == "boom":
            raise RuntimeError("内部错误 13812345678")
        return {"ok": "yes"}

    app.add_middleware(HttpMetricsMiddleware, metrics=metrics)
    return TestClient(app, raise_server_exceptions=False)


def test_http_metrics_use_route_templates_and_error_codes(metrics):
    client = _http_app(metrics)
    assert client.get("/v1/shops/s_1001/things/13812345678").status_code == 200
    assert client.get("/v1/shops/s_other/things/x").status_code == 403
    assert client.get("/v1/shops/s_1001/things/boom").status_code == 500
    assert client.get("/wp-admin/北京市朝阳区").status_code == 404
    s = _samples(metrics)
    route = "/v1/shops/{shop_id}/things/{thing_id}"
    assert sample(s, "rodit_http_requests_total", route=route, method="GET", status="200") == 1
    assert sample(s, "rodit_http_requests_total", route=route, method="GET", status="403") == 1
    assert sample(s, "rodit_http_errors_total", route=route, code="tenant_isolation") == 1
    assert sample(s, "rodit_http_errors_total", route=route, code="internal_error") == 1
    assert (
        sample(s, "rodit_http_requests_total", route="unmatched", method="GET", status="404") == 1
    )
    assert sample(s, "rodit_http_errors_total", route="unmatched", code="not_found") == 1
    assert sample(s, "rodit_http_request_duration_seconds_count", route=route, method="GET") == 3
    text = metrics.render()
    assert "s_1001" not in text and "13812345678" not in text and "北京" not in text
