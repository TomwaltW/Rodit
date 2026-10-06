"""每条消息的时间预算（方案 §12.1 推荐出现 P95 ≤ 5 秒、§12.2 超时降级）、串店检查（§9.2、§12.4）、
从收到消息算起的耗时（latency_ms["since_received"]）。

多数用例注入假的单调时钟（AssistEngine(monotonic=...)），由替身在调用里拨时间，不用真的等；
"到点丢掉没做完的分条"要真等，用很短的预算。
"""

from __future__ import annotations

import threading
import time
from datetime import timedelta
from typing import Any

from engine_support import Kit, make_route, qa_pass

from rodit.core import budget
from rodit.core.audit import (
    DEGRADED_BUDGET,
    DEGRADED_TENANT_ISOLATION,
    LATENCY_SINCE_RECEIVED,
    RESULT_BUDGET_EXCEEDED,
    RESULT_DETAIL_KEYS,
    AuditAction,
)
from rodit.core.config import EngineConfig, LatencyConfig
from rodit.core.enums import DraftKind, Intent, Mode, RouteAction
from rodit.core.errors import TenantIsolationError
from rodit.core.models import Draft, EngineResult
from rodit.engine.engine import (
    LATENCY_KEYS,
    REASON_BUDGET,
    REASON_TENANT,
    TENANT_STAGE_RETRIEVE,
    TENANT_STAGE_UNDERSTAND,
)
from rodit.testing.fixtures import SHOP_A

SHIP = "亲，现货付款后 48 小时内发货哦~"


class Monotonic:
    """假的单调时钟（秒）：替身在调用里拨时间。"""

    def __init__(self) -> None:
        self.t = 5000.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            return self.t

    def advance(self, seconds: float) -> None:
        with self.lock:
            self.t += seconds


def llm(text: str) -> Draft:
    return Draft(text=text, kind=DraftKind.LLM, evidence_ids=["policy:ship_time"])


def template(text: str) -> Draft:
    return Draft(text=text, kind=DraftKind.TEMPLATE, template_id="tpl")


def faq_draft(text: str, faq_id: int = 3) -> Draft:
    return Draft(text=text, kind=DraftKind.FAQ, faq_id=faq_id, evidence_ids=[f"faq:{faq_id}"])


def handle(kit: Kit, text: str = "什么时候发货", **engine_kw: Any) -> EngineResult:
    result = kit.engine(**engine_kw).handle(kit.msg(text))
    assert result is not None
    return result


def result_detail(kit: Kit) -> dict[str, Any]:
    [event] = kit.result_events()
    return event.detail


def split_route(*intents: Intent):
    return make_route(
        intents[0],
        RouteAction.SPLIT_MULTI,
        rule=5,
        parts=[(i, 0.6) for i in intents],
    )


# ================================================================ 截止时间（§12.1）


def test_each_message_gets_its_own_deadline(kit: Kit) -> None:
    kit.config = EngineConfig(latency=LatencyConfig(message_budget_s=3.0, min_llm_remaining_s=0.7))
    seen: list[budget.Deadline | None] = []
    original = kit.generator.generate

    def generate(*args: Any, **kw: Any) -> list[Draft]:
        seen.append(budget.current_deadline())
        return original(*args, **kw)

    kit.generator.generate = generate  # type: ignore[method-assign]
    engine = kit.engine()
    engine.handle(kit.msg("什么时候发货"))
    engine.handle(kit.msg("发什么快递"))
    first, second = seen
    assert first is not None and second is not None and first is not second
    assert (first.budget_s, first.min_llm_remaining_s) == (3.0, 0.7)
    assert budget.current_deadline() is None  # 出了 handle 就没有截止时间


def test_expired_before_generation_skips_generation(kit: Kit) -> None:
    clock = Monotonic()
    original = kit.retriever.retrieve

    def slow_retrieve(ctx, route):  # type: ignore[no-untyped-def]
        clock.advance(10)  # 检索用掉了全部预算
        return original(ctx, route)

    fast = handle(kit, "发什么快递", monotonic=clock)  # 先来一条来得及的：不标记、没有可选键
    assert fast.degraded == [] and fast.suggestions
    assert tuple(result_detail(kit)) == RESULT_DETAIL_KEYS
    kit.audit.events.clear()
    kit.generator.calls.clear()

    kit.retriever.retrieve = slow_retrieve  # type: ignore[method-assign]
    result = handle(kit, monotonic=clock)
    assert kit.generator.calls == []
    assert result.suggestions == []
    assert result.degraded == [DEGRADED_BUDGET]
    assert result.no_suggestion_reason == REASON_BUDGET
    detail = result_detail(kit)
    assert detail[RESULT_BUDGET_EXCEEDED] is True
    assert detail["degraded"] == [DEGRADED_BUDGET]
    assert set(detail) == {*RESULT_DETAIL_KEYS, RESULT_BUDGET_EXCEEDED}


def test_llm_drafts_after_the_deadline_are_dropped_faq_kept(kit: Kit) -> None:
    """生成器到截止时间才返回：大模型草稿不判、丢掉；FAQ / 模板照常过硬规则、保留。"""
    clock = Monotonic()
    faq = faq_draft(SHIP)

    def late_generate(*args: Any, **kw: Any) -> list[Draft]:
        clock.advance(10)
        return [llm("亲，48 小时内发"), faq]

    kit.generator.generate = late_generate  # type: ignore[method-assign]
    result = handle(kit, monotonic=clock)
    assert [(c.kind, c.text) for c in result.suggestions] == [(DraftKind.FAQ, SHIP)]
    assert [d.text for d in kit.gate.calls] == [SHIP]  # 大模型草稿没有送质检
    assert result.degraded == [DEGRADED_BUDGET]
    assert result_detail(kit)[RESULT_BUDGET_EXCEEDED] is True
    assert result_detail(kit)["qa_failed"] == 0  # 丢掉的不算质检不通过
    [record] = kit.suggestions.list(SHOP_A)
    assert record.kind == DraftKind.FAQ


def test_llm_draft_is_not_sent_to_qa_without_time_for_it(kit: Kit) -> None:
    """还没到截止时间，但剩余时间不够调一次判断层：大模型草稿不判；模板照判。"""
    clock = Monotonic()
    budget_s = kit.config.latency.message_budget_s
    floor = kit.config.latency.min_llm_remaining_s

    def generate(*args: Any, **kw: Any) -> list[Draft]:
        clock.advance(budget_s - floor / 2)  # 还剩一半下限
        return [llm("亲，48 小时内发"), template("亲，我帮您查一下")]

    kit.generator.generate = generate  # type: ignore[method-assign]
    result = handle(kit, monotonic=clock)
    assert [d.text for d in kit.gate.calls] == ["亲，我帮您查一下"]
    assert [c.text for c in result.suggestions] == ["亲，我帮您查一下"]
    assert result.degraded == [DEGRADED_BUDGET]


def test_llm_qa_finished_after_the_deadline_is_not_used(kit: Kit) -> None:
    clock = Monotonic()

    def slow_judge(draft: Draft):  # type: ignore[no-untyped-def]
        clock.advance(10)  # 判断层答完时截止时间已过
        return qa_pass(draft)

    kit.gate.rule = slow_judge
    kit.generator.script = [[llm("亲，48 小时内发")]]
    result = handle(kit, monotonic=clock)
    assert result.suggestions == []
    assert result.degraded == [DEGRADED_BUDGET]
    assert result.no_suggestion_reason == REASON_BUDGET


def test_no_regeneration_when_time_is_short(kit: Kit) -> None:
    """质检不通过要重写，但剩余时间不够再调一次大模型：不重写（§12.2）。"""
    clock = Monotonic()
    budget_s = kit.config.latency.message_budget_s

    def judge(draft: Draft):  # type: ignore[no-untyped-def]
        clock.advance(budget_s - 0.1)  # 判完还没到截止时间，但只剩 0.1 秒
        return None  # 按默认规则："一定" → 绝对化用语，不通过

    kit.gate.rule = judge
    kit.generator.script = [[llm("亲，一定 48 小时内发")], [llm("亲，48 小时内发")]]
    result = handle(kit, monotonic=clock)
    assert len(kit.generator.calls) == 1  # 没有重写
    assert result.suggestions == []
    assert result.degraded == [DEGRADED_BUDGET]
    assert result.no_suggestion_reason is not None
    assert result.no_suggestion_reason.startswith("质检未通过")
    assert result_detail(kit)["qa_failed"] == 1


def test_regeneration_still_happens_with_time_left(kit: Kit) -> None:
    kit.generator.script = [[llm("亲，一定 48 小时内发")], [llm("亲，48 小时内发")]]
    result = handle(kit, monotonic=Monotonic())
    assert len(kit.generator.calls) == 2
    assert [c.text for c in result.suggestions] == ["亲，48 小时内发"]
    assert result.degraded == []


def test_budget_blocks_auto_reply_and_transfers(kit: Kit) -> None:
    """全自动下超时 = 故障：不自动回复，转人工（§12.2）。"""
    kit.set_shop_config(SHOP_A, auto_reply_enabled=True, auto_reply_schedule="always")
    kit.router.decide = make_route(Intent.SHIPPING_POLICY, confidence=0.97, auto_eligible=True)
    clock = Monotonic()
    faq = faq_draft(SHIP)

    def late_generate(*args: Any, **kw: Any) -> list[Draft]:
        clock.advance(10)
        return [faq]

    kit.generator.generate = late_generate  # type: ignore[method-assign]
    result = kit.engine(monotonic=clock).handle(kit.msg("什么时候发货", mode=Mode.AUTO))
    assert result is not None
    assert [c.text for c in result.suggestions] == [SHIP]
    assert result.auto_reply is None and result.transfer_to_human is True
    assert DEGRADED_BUDGET in result.degraded


def test_unfinished_part_is_abandoned_without_late_writes(kit: Kit) -> None:
    """真等：一条分条卡住，引擎到"截止时间 + 宽限"就返回、丢掉它；它的线程跑完后也不写东西。"""
    kit.config = EngineConfig(latency=LatencyConfig(message_budget_s=0.3, min_llm_remaining_s=0.1))
    kit.router.decide = split_route(Intent.SHIPPING_POLICY, Intent.INVOICE_GIFT)
    release = threading.Event()
    finished = threading.Event()
    original = kit.generator.generate

    def generate(ctx, route, evidence, **kw):  # type: ignore[no-untyped-def]
        if route.intent == Intent.INVOICE_GIFT:
            release.wait(5)  # 卡住，直到测试放行
            drafts = [llm("亲，可以开电子发票")]
            finished.set()
            return drafts
        return original(ctx, route, evidence, **kw)

    kit.generator.generate = generate  # type: ignore[method-assign]
    started = time.perf_counter()
    result = handle(kit, "什么时候发货？能开发票吗？")
    elapsed = time.perf_counter() - started
    assert elapsed < 0.3 + 0.3
    assert [c.intent for c in result.suggestions] == [Intent.SHIPPING_POLICY]
    assert result.degraded == [DEGRADED_BUDGET]
    assert f"「发票 / 赠品 / 保障」没有推荐：{REASON_BUDGET}" in result.alerts
    events = list(kit.audit.events)
    decisions = list(kit.audit.decisions)
    records = kit.suggestions.list(SHOP_A)
    gaps = kit.gaps.list_gaps(SHOP_A)
    gate_calls = len(kit.gate.calls)
    release.set()  # 放行卡住的分条，让它跑完
    assert finished.wait(5)
    time.sleep(0.2)
    assert list(kit.audit.events) == events
    assert list(kit.audit.decisions) == decisions
    assert kit.suggestions.list(SHOP_A) == records
    assert kit.gaps.list_gaps(SHOP_A) == gaps
    assert len(kit.gate.calls) == gate_calls  # 过了截止时间，丢掉的分条不再送质检


# ================================================================ 串店检查（§9.2、§12.4）


def test_tenant_violation_in_retrieve(kit: Kit) -> None:
    kit.retriever.error = TenantIsolationError("证据数据属于店铺 s_other，当前店铺是 s_1001")
    result = handle(kit)
    assert result.degraded == [DEGRADED_TENANT_ISOLATION]
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_TENANT
    [event] = kit.events(AuditAction.TENANT_VIOLATION)
    assert event.detail == {"stage": TENANT_STAGE_RETRIEVE}  # 只记在哪一步拦下的，不记数据
    assert event.actor == "system" and event.shop_id == SHOP_A
    assert kit.generator.calls == []
    assert result_detail(kit)["degraded"] == [DEGRADED_TENANT_ISOLATION]


def test_tenant_violation_in_one_part_keeps_the_others(kit: Kit) -> None:
    kit.router.decide = split_route(Intent.SHIPPING_POLICY, Intent.INVOICE_GIFT)

    def bundles(ctx, route):  # type: ignore[no-untyped-def]
        raise TenantIsolationError("串店")

    kit.retriever.bundles[Intent.INVOICE_GIFT] = bundles
    result = handle(kit, "什么时候发货？能开发票吗？")
    assert [c.intent for c in result.suggestions] == [Intent.SHIPPING_POLICY]
    assert result.degraded == [DEGRADED_TENANT_ISOLATION]
    assert len(kit.events(AuditAction.TENANT_VIOLATION)) == 1


def test_tenant_violation_in_understanding(kit: Kit, monkeypatch: Any) -> None:
    engine = kit.engine()

    def boom(*a: Any, **kw: Any) -> Any:
        raise TenantIsolationError("订单属于别的店铺")

    monkeypatch.setattr(engine.builder, "build_with_notes", boom)
    result = engine.handle(kit.msg("什么时候发货"))
    assert result is not None
    assert result.degraded == [DEGRADED_TENANT_ISOLATION]
    assert result.no_suggestion_reason == REASON_TENANT
    [event] = kit.events(AuditAction.TENANT_VIOLATION)
    assert event.detail == {"stage": TENANT_STAGE_UNDERSTAND}
    assert kit.router.calls == []


# ================================================================ 从收到消息算起的耗时（§12.1）


def test_since_received_is_at_least_total_under_fake_clock(kit: Kit) -> None:
    result = handle(kit)  # 假时钟不走：收到时间 = 现在
    latency = result.latency_ms
    assert tuple(latency) == LATENCY_KEYS
    assert latency[LATENCY_SINCE_RECEIVED] >= latency["total"]
    assert result_detail(kit)["latency_ms"] == latency


def test_since_received_counts_the_wait_before_processing(kit: Kit) -> None:
    msg = kit.msg("什么时候发货")
    kit.clock.advance(seconds=3)  # 消息在队列里等了 3 秒
    result = kit.engine().handle(msg)
    assert result is not None
    assert result.latency_ms[LATENCY_SINCE_RECEIVED] == 3000
    assert result.latency_ms["total"] < 3000


def test_since_received_never_below_total_with_clock_skew(kit: Kit) -> None:
    msg = kit.msg("什么时候发货")
    skewed = msg.model_copy(update={"received_at": msg.received_at + timedelta(seconds=30)})
    result = kit.engine().handle(skewed)  # 平台时间比我们快：差值是负的
    assert result is not None
    assert result.latency_ms[LATENCY_SINCE_RECEIVED] == result.latency_ms["total"]


def test_since_received_on_the_failure_path(kit: Kit) -> None:
    kit.router.decide = lambda ctx: "not a route"  # type: ignore[assignment,return-value]
    msg = kit.msg("什么时候发货")
    kit.clock.advance(seconds=2)
    result = kit.engine().handle(msg)
    assert result is not None and result.degraded == ["engine"]
    assert result.latency_ms[LATENCY_SINCE_RECEIVED] == 2000
