"""端到端：5 秒预算（方案 §12.1 推荐出现 P95 ≤ 5 秒、§12.2 超时降级、§12.4 串店检查）。

用 build_demo_unit 组装真实的路由（关键词规则）、检索、生成、质检和问答引擎，只有大模型是慢替身：
生成每次 1.5 秒、判断层质检每次 0.8 秒（延迟在替身的锁外，并发调用互不排队）。

- 三个问题的消息并行分条：< 5.0 秒（顺序执行要 11.4 秒）；单个问题 < 2.6 秒（原来 3.8 秒）；
  卡片文字和顺序与零延迟替身完全一样；
- 生成要 10 秒时：handle() 在预算 + 0.3 秒内返回，降级带 budget，只剩 FAQ / 模板卡片，
  11 秒之后推荐记录和审计日志都没有变化（没有迟到的写入）；
- 平台工具返回别的店铺的数据（串店）：degraded == ["tenant_isolation"]，一条 tenant.violation；
- latency_ms["since_received"] ≥ latency_ms["total"]（假时钟）。
"""

from __future__ import annotations

import threading
import time
from typing import Any

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.app.offline import EvidenceEchoLLM
from rodit.core.audit import (
    DEGRADED_BUDGET,
    DEGRADED_TENANT_ISOLATION,
    LATENCY_SINCE_RECEIVED,
    RESULT_BUDGET_EXCEEDED,
    AuditAction,
)
from rodit.core.enums import DraftKind, Intent, RouteAction
from rodit.core.models import EngineResult
from rodit.core.ports import LLMRequest, LLMResponse
from rodit.decision import RuleProvider
from rodit.testing.fakes import scripted_judge_llm
from rodit.testing.fixtures import SHOP_A

THREE_PARTS = "这件165穿什么码？什么时候发货？能开发票吗？"
SINGLE = "这件165穿什么码"
WITH_TEMPLATE_PART = "这件165穿什么码？我的快递到哪了？"


class SlowLLM:
    """慢替身：生成按 EvidenceEchoLLM 复述依据，判断层质检全部通过；每次调用先睡一会儿。"""

    model = "slow-fake"

    def __init__(self, *, generate_s: float, qa_s: float) -> None:
        self.generate_s = generate_s
        self.qa_s = qa_s
        self.echo = EvidenceEchoLLM()
        self.judge = scripted_judge_llm()
        self.lock = threading.Lock()
        self.started = 0
        self.finished = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        with self.lock:
            self.started += 1
        time.sleep(self.generate_s if request.purpose == "generate" else self.qa_s)
        try:
            if request.purpose == "generate":
                return self.echo.complete(request)
            return self.judge.complete(request)
        finally:
            with self.lock:
                self.finished += 1


def service_with(llm: SlowLLM) -> AssistService:
    return AssistService(build_demo_unit(llm=llm, route_provider=RuleProvider()))


def say(service: AssistService, text: str, msg_id: str = "m1", **kw: Any) -> EngineResult:
    result = service.handle_event("B", taobao_event(text, msg_id=msg_id, **kw))
    assert isinstance(result, EngineResult)
    return result


def timed(service: AssistService, text: str, **kw: Any) -> tuple[EngineResult, float]:
    started = time.perf_counter()
    result = say(service, text, **kw)
    return result, time.perf_counter() - started


def cards(result: EngineResult) -> list[tuple[Intent, str]]:
    return [(c.intent, c.text) for c in result.suggestions]


# ================================================================ 并行后的耗时（§12.1）


def test_three_part_message_under_five_seconds_same_cards():
    slow, elapsed = timed(
        service_with(SlowLLM(generate_s=1.5, qa_s=0.8)), THREE_PARTS, item_id="i_555"
    )
    fast = say(service_with(SlowLLM(generate_s=0, qa_s=0)), THREE_PARTS, item_id="i_555")
    assert slow.route is not None and slow.route.action == RouteAction.SPLIT_MULTI
    assert len(slow.suggestions) == 3
    assert elapsed < 5.0, elapsed  # 顺序执行要 11.4 秒
    assert slow.degraded == [] and fast.degraded == []
    assert cards(slow) == cards(fast)  # 卡片文字和顺序与零延迟替身一样
    assert [c.kind for c in slow.suggestions] == [c.kind for c in fast.suggestions]


def test_single_question_under_2_6_seconds_same_cards():
    slow, elapsed = timed(service_with(SlowLLM(generate_s=1.5, qa_s=0.8)), SINGLE, item_id="i_555")
    fast = say(service_with(SlowLLM(generate_s=0, qa_s=0)), SINGLE, item_id="i_555")
    assert slow.suggestions and slow.suggestions[0].kind == DraftKind.LLM
    assert elapsed < 2.6, elapsed  # 两条候选并行生成 + 质检：原来 3.8 秒
    assert slow.degraded == []
    assert cards(slow) == cards(fast)


# ================================================================ 超过预算（§12.2）


def _snapshot(service: AssistService) -> tuple[Any, ...]:
    repos = service.unit.repos
    return (
        list(repos.audit.events),
        list(repos.audit.decisions),
        repos.suggestions.list(SHOP_A),
        repos.feedback.list_gaps(SHOP_A),
    )


def test_ten_second_generator_is_cut_at_the_budget_without_late_writes():
    budget_s = build_demo_unit().config.latency.message_budget_s
    runs: dict[str, dict[str, Any]] = {
        WITH_TEMPLATE_PART: {"kw": {"item_id": "i_555", "order_id": "o_1002"}},
        THREE_PARTS: {"kw": {"item_id": "i_555"}},
    }

    def run(text: str) -> None:
        llm = SlowLLM(generate_s=10, qa_s=0.8)
        service = service_with(llm)
        result, elapsed = timed(service, text, **runs[text]["kw"])
        runs[text].update(
            llm=llm,
            service=service,
            result=result,
            elapsed=elapsed,
            returned=time.monotonic(),
            snapshot=_snapshot(service),
        )

    threads = [threading.Thread(target=run, args=(text,)) for text in runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    for text, r in runs.items():
        result: EngineResult = r["result"]
        assert r["elapsed"] < budget_s + 0.3, (text, r["elapsed"])
        assert DEGRADED_BUDGET in result.degraded
        # 只剩 FAQ / 模板卡片：没做完的大模型草稿都丢掉了
        assert all(c.kind in (DraftKind.FAQ, DraftKind.TEMPLATE) for c in result.suggestions)
        [event] = r["service"].unit.repos.audit.list_events(SHOP_A, action=AuditAction.RESULT.value)
        assert event.detail[RESULT_BUDGET_EXCEEDED] is True

    # 有模板那条消息：物流进度模板保留，尺码那条丢掉并提醒客服
    kept = runs[WITH_TEMPLATE_PART]["result"]
    assert [(c.intent, c.kind) for c in kept.suggestions] == [
        (Intent.LOGISTICS_TRACK, DraftKind.TEMPLATE)
    ]
    assert any(a.startswith("「尺码 / 选型」没有推荐") for a in kept.alerts)

    # 等慢调用都返回（它们的线程跑完），并且离 handle() 返回已经过了 11 秒
    for r in runs.values():
        llm: SlowLLM = r["llm"]
        deadline = time.monotonic() + 15
        while llm.finished < llm.started and time.monotonic() < deadline:
            time.sleep(0.1)
        assert llm.finished == llm.started
    wait = max(r["returned"] for r in runs.values()) + 11 - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    for text, r in runs.items():
        assert _snapshot(r["service"]) == r["snapshot"], text  # 没有迟到的写入


# ================================================================ 串店检查（§9.2、§12.4）


def test_tool_returning_another_shops_data_is_a_tenant_violation():
    service = service_with(SlowLLM(generate_s=0, qa_s=0))
    tools = service.unit.tools
    original = tools.get_logistics

    def leaky(shop_id: str, order_id: str):  # type: ignore[no-untyped-def]
        trace = original(shop_id, order_id)
        return trace.model_copy(update={"shop_id": "s_someone_else"})  # 串店

    tools.get_logistics = leaky  # type: ignore[method-assign]
    result = say(service, "我的快递到哪了", order_id="o_1002")
    assert result.route is not None and result.route.intent == Intent.LOGISTICS_TRACK
    assert result.degraded == [DEGRADED_TENANT_ISOLATION]
    assert result.suggestions == []
    audit = service.unit.repos.audit
    [event] = audit.list_events(SHOP_A, action=AuditAction.TENANT_VIOLATION.value)
    assert event.detail == {"stage": "retrieve"}  # 不记任何数据
    assert "s_someone_else" not in str([e.detail for e in audit.events])


# ================================================================ 从收到消息算起（§12.1）


def test_since_received_is_at_least_total():
    service = service_with(SlowLLM(generate_s=0, qa_s=0))
    result = say(service, "什么时候发货")
    assert result.latency_ms[LATENCY_SINCE_RECEIVED] >= result.latency_ms["total"]
    [event] = service.unit.repos.audit.list_events(SHOP_A, action=AuditAction.RESULT.value)
    assert (
        event.detail["latency_ms"][LATENCY_SINCE_RECEIVED]
        == (result.latency_ms[LATENCY_SINCE_RECEIVED])
    )
