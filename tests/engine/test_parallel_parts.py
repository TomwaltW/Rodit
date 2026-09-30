"""多问题分条并行、草稿并行质检（方案 §6.6 规则 5、§12.1 推荐出现 P95 ≤ 5 秒）。

- 各条的"找 → 写 → 判"并行跑，同一条的几份大模型草稿并行质检；
- 结果按原顺序收回：卡片、提醒、降级、审计的先后都和顺序执行一样（后面的条先做完也一样）；
- 质检分批：判哪些草稿和顺序执行一样，不多调判断层；
- 审计、推荐记录只在调用方线程、收回结果之后写；工作线程带着店铺范围和截止时间。
"""

from __future__ import annotations

import threading
import time
from typing import Any

from engine_support import Kit, bundle_of, make_route, policy_evidence, qa_pass

from rodit.core import budget
from rodit.core import evidence as ev
from rodit.core.audit import AuditAction
from rodit.core.context import current_shop_id
from rodit.core.enums import DraftKind, Intent, RouteAction
from rodit.core.models import Draft, EngineResult, QAResult
from rodit.engine.engine import REASON_ENGINE
from rodit.testing.fixtures import SHOP_A

PARTS = (Intent.SHIPPING_POLICY, Intent.INVOICE_GIFT, Intent.PRODUCT_SPEC)
TEXT = "什么时候发货？能开发票吗？是纯棉的吗？"


def llm(text: str) -> Draft:
    return Draft(text=text, kind=DraftKind.LLM, evidence_ids=["policy:ship_time"])


def split_route(*intents: Intent):
    return make_route(
        intents[0], RouteAction.SPLIT_MULTI, rule=5, parts=[(i, 0.6) for i in intents]
    )


def handle(kit: Kit, text: str = TEXT, **engine_kw: Any) -> EngineResult:
    result = kit.engine(**engine_kw).handle(kit.msg(text))
    assert result is not None
    return result


def answer_of(intent: Intent) -> str:
    return f"亲，这是关于{intent.value}的回答"


class PartGenerator:
    """每条按问题类型给一份大模型草稿；delays[intent] 秒后返回，barrier 不为空时先等齐。"""

    def __init__(
        self,
        delays: dict[Intent, float] | None = None,
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.delays = delays or {}
        self.barrier = barrier
        self.finished: list[Intent] = []
        self.threads: set[int] = set()
        self.shop_ids: list[str | None] = []
        self.deadlines: list[budget.Deadline | None] = []
        self.lock = threading.Lock()

    def generate(self, ctx, route, evidence, *, retry_feedback=None):  # type: ignore[no-untyped-def]
        with self.lock:
            self.threads.add(threading.get_ident())
            self.shop_ids.append(current_shop_id())
            self.deadlines.append(budget.current_deadline())
        if self.barrier is not None:
            self.barrier.wait()  # 逐条执行时等不齐，抛 BrokenBarrierError → 生成失败
        time.sleep(self.delays.get(route.intent, 0.0))
        with self.lock:
            self.finished.append(route.intent)
        return [llm(answer_of(route.intent))]


# ================================================================ 并行


def test_parts_run_concurrently(kit: Kit) -> None:
    kit.router.decide = split_route(*PARTS)
    gen = PartGenerator(barrier=threading.Barrier(3, timeout=2))
    result = handle(kit, generator=gen)
    assert result.degraded == []
    assert [(c.intent, c.text) for c in result.suggestions] == [(i, answer_of(i)) for i in PARTS]
    assert len(gen.threads) == 3


def test_parts_overlap_in_time(kit: Kit) -> None:
    kit.router.decide = split_route(*PARTS)
    gen = PartGenerator(delays=dict.fromkeys(PARTS, 0.3))
    started = time.perf_counter()
    result = handle(kit, generator=gen)
    elapsed = time.perf_counter() - started
    assert len(result.suggestions) == 3
    assert elapsed < 0.6  # 顺序执行要 0.9 秒
    # 各环节耗时取各条里最长的（墙钟口径），不是三条相加
    assert 300 <= result.latency_ms["generate"] < 600
    assert result.latency_ms["total"] >= result.latency_ms["generate"]


def test_order_is_kept_when_later_parts_finish_first(kit: Kit) -> None:
    kit.router.decide = split_route(*PARTS)
    kit.retriever.bundles[Intent.SHIPPING_POLICY] = bundle_of(
        policy_evidence(), unavailable=[ev.unavailable("logistics")]
    )
    kit.retriever.bundles[Intent.PRODUCT_SPEC] = bundle_of(
        policy_evidence(), unavailable=[ev.unavailable("stock")]
    )
    gen = PartGenerator(
        delays={Intent.SHIPPING_POLICY: 0.3, Intent.INVOICE_GIFT: 0.15, Intent.PRODUCT_SPEC: 0.0}
    )
    result = handle(kit, generator=gen)
    assert gen.finished == list(reversed(PARTS))  # 后面的条先做完
    # 卡片、降级、推荐记录、审计都按原顺序
    assert [c.intent for c in result.suggestions] == list(PARTS)
    assert result.degraded == ["tool:logistics", "tool:stock"]
    assert [r.intent for r in kit.suggestions.list(SHOP_A)] == list(PARTS)
    stage_events = [
        (e.action, e.detail["intent"])
        for e in kit.audit.list_events(SHOP_A)
        if e.action in (AuditAction.RETRIEVE, AuditAction.GENERATE, AuditAction.QA)
    ]
    assert stage_events == [
        (action, str(intent))
        for intent in PARTS
        for action in (AuditAction.RETRIEVE, AuditAction.GENERATE, AuditAction.QA)
    ]
    actions = [e.action for e in kit.audit.list_events(SHOP_A)]
    assert actions[:2] == [AuditAction.UNDERSTAND, AuditAction.ROUTE]
    assert actions[-2:] == [AuditAction.SUGGEST, AuditAction.RESULT]


def test_workers_see_shop_scope_and_the_message_deadline(kit: Kit) -> None:
    kit.router.decide = split_route(*PARTS)
    gen = PartGenerator()
    handle(kit, generator=gen)
    assert gen.shop_ids == [SHOP_A] * 3  # 大模型成本按店铺归属（§12.4）
    deadlines = set(map(id, gen.deadlines))
    assert len(deadlines) == 1 and gen.deadlines[0] is not None  # 同一条消息同一个截止时间
    assert gen.deadlines[0].budget_s == kit.config.latency.message_budget_s


class ThreadRecordingAudit:
    """记下每次写审计的线程。"""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.threads: set[int] = set()

    def record(self, event: Any) -> None:
        self.threads.add(threading.get_ident())
        self.inner.record(event)

    def log_decision(self, entry: Any) -> None:
        self.threads.add(threading.get_ident())
        self.inner.log_decision(entry)


class ThreadRecordingSuggestions:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.threads: set[int] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def add(self, record: Any) -> Any:
        self.threads.add(threading.get_ident())
        return self.inner.add(record)


def test_writes_happen_on_the_calling_thread(kit: Kit) -> None:
    kit.router.decide = split_route(*PARTS)
    audit = ThreadRecordingAudit(kit.audit)
    suggestions = ThreadRecordingSuggestions(kit.suggestions)
    gen = PartGenerator(delays=dict.fromkeys(PARTS, 0.05))
    result = handle(kit, generator=gen, audit=audit, suggestions=suggestions)
    assert len(result.suggestions) == 3
    assert gen.threads and threading.get_ident() not in gen.threads  # 分条在工作线程里跑
    assert audit.threads == {threading.get_ident()}
    assert suggestions.threads == {threading.get_ident()}


def test_exception_in_a_part_matches_sequential_execution(kit: Kit) -> None:
    """第二条意外抛错：第一条和第二条抛错之前的记录照常写，第三条丢掉，交给兜底（和顺序执行一样）。"""
    kit.router.decide = split_route(*PARTS)
    kit.generator.script_for = {
        Intent.SHIPPING_POLICY: [[llm(answer_of(Intent.SHIPPING_POLICY))]],
        Intent.INVOICE_GIFT: [[None]],  # type: ignore[list-item]  # 写 GENERATE 审计时出错
    }
    result = handle(kit)
    assert result.suggestions == []
    assert result.degraded == ["engine"]
    assert result.no_suggestion_reason == REASON_ENGINE
    stage_events = [
        (e.action, e.detail["intent"])
        for e in kit.audit.list_events(SHOP_A)
        if e.action in (AuditAction.RETRIEVE, AuditAction.GENERATE, AuditAction.QA)
    ]
    assert stage_events == [
        (AuditAction.RETRIEVE, "shipping_policy"),
        (AuditAction.GENERATE, "shipping_policy"),
        (AuditAction.QA, "shipping_policy"),
        (AuditAction.RETRIEVE, "invoice_gift"),
    ]
    assert kit.suggestions.list(SHOP_A) == []
    assert len(kit.result_events()) == 1


# ================================================================ 草稿并行质检


class BarrierGate:
    """大模型草稿要等齐 parties 份才一起判完；FAQ / 模板直接通过。"""

    def __init__(self, parties: int) -> None:
        self.barrier = threading.Barrier(parties, timeout=2)
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def check(self, ctx, route, draft, evidence) -> QAResult:  # type: ignore[no-untyped-def]
        with self.lock:
            self.calls.append(draft.text)
        if draft.kind == DraftKind.LLM:
            self.barrier.wait()  # 逐份判时等不齐，抛错 → 质检失败
        return qa_pass(draft)


def test_llm_drafts_are_checked_concurrently(kit: Kit) -> None:
    kit.generator.script = [[llm("亲，48 小时内发货"), llm("亲，付款后 48 小时内发出")]]
    gate = BarrierGate(2)
    result = handle(kit, "什么时候发货", gate=gate)
    assert result.degraded == []
    assert [c.text for c in result.suggestions] == ["亲，48 小时内发货", "亲，付款后 48 小时内发出"]
    assert sorted(gate.calls) == sorted(["亲，48 小时内发货", "亲，付款后 48 小时内发出"])


def test_only_the_drafts_sequential_execution_would_check(kit: Kit) -> None:
    """每条最多一张卡片：第一份通过就不再判第二份（不多调判断层）。"""
    kit.router.decide = split_route(Intent.SHIPPING_POLICY, Intent.INVOICE_GIFT)
    kit.generator.script_for = {
        Intent.SHIPPING_POLICY: [[llm("发货一"), llm("发货二")]],
        Intent.INVOICE_GIFT: [[llm("发票一"), llm("发票二")]],
    }
    result = handle(kit, "什么时候发货？能开发票吗？")
    assert [c.text for c in result.suggestions] == ["发货一", "发票一"]
    assert sorted(d.text for d in kit.gate.calls) == ["发票一", "发货一"]


def test_failed_first_draft_checks_the_next_one(kit: Kit) -> None:
    kit.router.decide = split_route(Intent.SHIPPING_POLICY, Intent.INVOICE_GIFT)
    kit.generator.script_for = {
        Intent.SHIPPING_POLICY: [[llm("一定发"), llm("发货二")]],
    }
    result = handle(kit, "什么时候发货？能开发票吗？")
    assert [c.intent for c in result.suggestions] == [Intent.SHIPPING_POLICY, Intent.INVOICE_GIFT]
    assert result.suggestions[0].text == "发货二"
    shipping_checks = [d.text for d in kit.gate.calls if d.text in ("一定发", "发货二")]
    assert shipping_checks == ["一定发", "发货二"]  # 一份一份判（第二份是补的）
