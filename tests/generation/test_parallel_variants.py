"""生成器的并行与时间预算（方案 §5.4 "1–3 条候选"、§12.1 生成 ≤ 2.5 秒、§12.2 大模型超时退回模板）。

- 直接回答、附带追问两条大模型草稿并行生成，按原顺序收回（结果和逐条生成一样）；
- 剩余时间不够调一次大模型（core.budget.llm_allowed）→ 不调，退回 FAQ / 模板；
  本来要写大模型草稿才记一笔（core.budget.note_llm_skipped），FAQ / 模板占满名额的不记；
- 到截止时间还没返回的草稿丢掉，已经返回的照用；单次调用的超时不超过剩余时间；
- 工作线程带着调用方的店铺范围（大模型成本按店铺归属，§12.4）和截止时间。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from rodit.core import budget
from rodit.core.config import GenerationConfig
from rodit.core.context import current_shop_id, shop_scope
from rodit.core.enums import DraftKind, Intent, PolicyKey
from rodit.core.errors import LLMError
from rodit.core.evidence import PlatformRuleKey, unavailable
from rodit.core.models import EvidenceBundle
from rodit.generation import DefaultAnswerGenerator
from rodit.generation import templates as T
from rodit.testing.fakes import FakeLLMClient, make_context
from rodit.testing.fixtures import SHOP_A

CFG = GenerationConfig()
DIRECT = "亲，这款面料是棉 95%、氨纶 5% 哦"
FOLLOWUP = "亲，这款面料是棉 95%、氨纶 5%，您平时穿什么码呢？"


def is_followup(request) -> bool:
    return "追问一个问题" in request.messages[1].content


def reply(text: str) -> dict:
    return {"answer": text, "evidence_ids": ["item:i_555:attr:面料"], "need_clarify": False}


class SlowLLM:
    """按变体延迟的大模型替身。延迟在 FakeLLMClient 的锁外，并发调用互不排队。"""

    def __init__(self, *, direct_s: float = 0.0, followup_s: float = 0.0) -> None:
        self.delays = {False: direct_s, True: followup_s}
        self.inner = FakeLLMClient(lambda r: reply(FOLLOWUP if is_followup(r) else DIRECT))
        self.finished: list[str] = []
        self.shop_ids: list[str | None] = []
        self.deadlines: list[budget.Deadline | None] = []
        self.threads: set[int] = set()

    def complete(self, request):
        followup = is_followup(request)
        self.shop_ids.append(current_shop_id())
        self.deadlines.append(budget.current_deadline())
        self.threads.add(threading.get_ident())
        time.sleep(self.delays[followup])
        response = self.inner.complete(request)
        self.finished.append("with_followup" if followup else "direct")
        return response


class Monotonic:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def spec_context(world, **kw):
    return make_context("这个是纯棉的吗？", world, item_id="i_555", **kw)


def spec_bundle(ev) -> EvidenceBundle:
    return EvidenceBundle(evidences=[ev.item_attr("面料")])


def generate(llm, world, ev, route, **kw):
    return DefaultAnswerGenerator(llm, CFG).generate(
        spec_context(world, **kw), route(Intent.PRODUCT_SPEC), spec_bundle(ev)
    )


# ================================================================ 并行、按原顺序


class BarrierLLM:
    """两条请求都进了 complete 才一起返回；逐条调用时第一条等到栅栏超时，抛 LLMError。"""

    def __init__(self, parties: int = 2) -> None:
        self.barrier = threading.Barrier(parties, timeout=2)
        self.inner = FakeLLMClient(lambda r: reply(FOLLOWUP if is_followup(r) else DIRECT))

    def complete(self, request):
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError as exc:
            raise LLMError("没有并行调用") from exc
        return self.inner.complete(request)


def test_variants_are_generated_concurrently(world, ev, route):
    drafts = generate(BarrierLLM(), world, ev, route)
    assert [(d.variant, d.text) for d in drafts] == [
        ("direct", DIRECT),
        ("with_followup", FOLLOWUP),
    ]


def test_order_is_kept_when_the_second_returns_first(world, ev, route):
    llm = SlowLLM(direct_s=0.3)
    started = time.perf_counter()
    drafts = generate(llm, world, ev, route)
    elapsed = time.perf_counter() - started
    assert llm.finished == ["with_followup", "direct"]  # 追问那条先返回
    assert [d.variant for d in drafts] == ["direct", "with_followup"]  # 结果仍按原顺序
    assert all(d.kind == DraftKind.LLM for d in drafts)
    assert elapsed < 0.55  # 并行：不是 0.3 + 0


def test_workers_see_shop_scope_and_deadline(world, ev, route):
    llm = SlowLLM()
    with shop_scope(SHOP_A), budget.deadline_scope(5.0, min_llm_remaining_s=0.5) as deadline:
        generate(llm, world, ev, route)
    assert threading.get_ident() not in llm.threads  # 在工作线程里调
    assert llm.shop_ids == [SHOP_A, SHOP_A]  # llm.usage 计量按店铺归属（§12.4）
    assert llm.deadlines == [deadline, deadline]


# ================================================================ 时间预算


def test_no_llm_call_when_too_little_time_left(world, ev, route):
    """剩余时间不够调一次大模型：不调，退回模板（§12.2）。"""
    rule = ev.platform_rule(PlatformRuleKey.APPLY_AFTERSALE, "在订单页点「申请售后」发起")
    bundle = EvidenceBundle(
        evidences=[ev.policy(PolicyKey.SEVEN_DAY_RETURN), rule], unavailable=[unavailable("order")]
    )
    llm = FakeLLMClient(lambda r: reply(DIRECT))
    clock = Monotonic()
    with budget.deadline_scope(4.5, min_llm_remaining_s=0.8, now=clock):
        clock.t += 4.0  # 还剩 0.5 秒 < 0.8 秒
        assert not budget.llm_allowed()
        drafts = DefaultAnswerGenerator(llm, CFG).generate(
            make_context("能退吗", world), route(Intent.RETURN_POLICY), bundle
        )
    assert llm.calls == []
    assert [d.template_id for d in drafts] == [T.APPLY_AFTERSALE, T.TOOL_UNAVAILABLE]
    # 剩余时间正好等于下限：边界算来得及，照常调
    boundary = Monotonic()
    with budget.deadline_scope(4.5, min_llm_remaining_s=0.5, now=boundary):
        boundary.t += 4.0
        drafts = generate(llm, world, ev, route)
    assert [d.kind for d in drafts] == [DraftKind.LLM]
    assert len(llm.calls) == 2


def test_no_llm_call_without_fallback_gives_nothing(world, ev, route):
    llm = FakeLLMClient(lambda r: reply(DIRECT))
    clock = Monotonic()
    with budget.deadline_scope(4.5, min_llm_remaining_s=0.8, now=clock):
        clock.t += 10  # 截止时间已过
        drafts = generate(llm, world, ev, route)
    assert drafts == [] and llm.calls == []


def test_unfinished_variant_is_dropped_at_the_deadline(world, ev, route):
    """追问那条到截止时间还没返回：丢掉，直接回答照用；生成器在截止时间就返回，不等它。"""
    llm = SlowLLM(followup_s=1.5)
    started = time.perf_counter()
    with budget.deadline_scope(0.4):
        drafts = generate(llm, world, ev, route)
    elapsed = time.perf_counter() - started
    assert [(d.variant, d.text) for d in drafts] == [("direct", DIRECT)]
    assert elapsed < 0.4 + 0.25


def test_all_unfinished_falls_back_to_templates(world, ev, route):
    rule = ev.platform_rule(PlatformRuleKey.APPLY_AFTERSALE, "在订单页点「申请售后」发起")
    bundle = EvidenceBundle(
        evidences=[ev.policy(PolicyKey.SEVEN_DAY_RETURN), rule], unavailable=[unavailable("order")]
    )
    llm = SlowLLM(direct_s=1.5, followup_s=1.5)
    started = time.perf_counter()
    with budget.deadline_scope(0.4):
        drafts = DefaultAnswerGenerator(llm, CFG).generate(
            make_context("能退吗", world), route(Intent.RETURN_POLICY), bundle
        )
    assert time.perf_counter() - started < 0.4 + 0.25
    assert [d.template_id for d in drafts] == [T.APPLY_AFTERSALE, T.TOOL_UNAVAILABLE]
    assert llm.finished == []  # 返回时两条都还在调用中（结果不会再被用到）


def test_request_timeout_is_clamped_to_the_remaining_budget(world, ev, route):
    llm = FakeLLMClient(lambda r: reply(FOLLOWUP if is_followup(r) else DIRECT))
    generate(llm, world, ev, route)
    assert {c.timeout_s for c in llm.calls} == {CFG.timeout_s}  # 没有截止时间：照配置
    clock = Monotonic()
    llm2 = FakeLLMClient(lambda r: reply(FOLLOWUP if is_followup(r) else DIRECT))
    with budget.deadline_scope(4.5, min_llm_remaining_s=0.5, now=clock):
        clock.t += 3.25  # 还剩 1.25 秒 < timeout_s（2.5 秒）
        generate(llm2, world, ev, route)
    assert [c.timeout_s for c in llm2.calls] == [1.25, 1.25]


def test_unexpected_error_still_raises(world, ev, route):
    """意外错误（不是 LLMError）照常抛出，由引擎按生成失败降级（和逐条调用一样）。"""
    threads: set[int] = set()

    def boom(request):  # type: ignore[no-untyped-def]
        threads.add(threading.get_ident())
        return RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        generate(FakeLLMClient(boom), world, ev, route)
    assert threads and threading.get_ident() not in threads  # 错误出在工作线程里，照样抛给调用方


# ================================================================ 时间不够没调：告诉引擎


@contextmanager
def short_of_time(clock: Monotonic) -> Iterator[None]:
    """还剩 0.5 秒，下限 0.8 秒：没过期，但来不及调一次大模型。"""
    with budget.deadline_scope(4.5, min_llm_remaining_s=0.8, now=clock):
        clock.t += 4.0
        yield


def test_skip_is_noted_when_a_model_draft_was_wanted(world, ev, route):
    llm = FakeLLMClient(lambda r: reply(DIRECT))
    clock = Monotonic()
    with short_of_time(clock), budget.llm_skip_scope() as skip:
        assert not budget.llm_allowed() and not budget.expired()
        drafts = generate(llm, world, ev, route)
    assert drafts == [] and llm.calls == []
    assert skip.skipped is True


def test_skip_is_noted_when_templates_leave_a_slot_for_the_model(world, ev, route):
    """模板先占一个名额，剩下的名额本来要给大模型：没调也算（退回的模板照常返回）。"""
    rule = ev.platform_rule(PlatformRuleKey.APPLY_AFTERSALE, "在订单页点「申请售后」发起")
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.SEVEN_DAY_RETURN), rule])
    llm = FakeLLMClient(lambda r: reply(DIRECT))
    clock = Monotonic()
    with short_of_time(clock), budget.llm_skip_scope() as skip:
        drafts = DefaultAnswerGenerator(llm, CFG).generate(
            make_context("退货怎么申请", world), route(Intent.RETURN_POLICY), bundle
        )
    assert [d.template_id for d in drafts] == [T.APPLY_AFTERSALE] and llm.calls == []
    assert skip.skipped is True


def test_skip_is_noted_once_the_deadline_has_passed(world, ev, route):
    llm = FakeLLMClient(lambda r: reply(DIRECT))
    clock = Monotonic()
    with budget.deadline_scope(4.5, min_llm_remaining_s=0.8, now=clock):
        clock.t += 10
        with budget.llm_skip_scope() as skip:
            assert generate(llm, world, ev, route) == []
    assert skip.skipped is True and llm.calls == []


def test_no_skip_when_faq_fills_the_slots(world, ev, route):
    """FAQ 直答：本来就不调大模型，时间再紧也不算少了东西。"""
    faq = ev.faq("发什么快递", score=CFG.faq_direct_min_score, faq_id=7)
    llm = FakeLLMClient(responses=[])
    clock = Monotonic()
    with short_of_time(clock), budget.llm_skip_scope() as skip:
        drafts = DefaultAnswerGenerator(llm, CFG).generate(
            make_context("发什么快递呀", world),
            route(Intent.SHIPPING_POLICY),
            EvidenceBundle(evidences=[faq, ev.policy(PolicyKey.COURIER)]),
        )
    assert [d.kind for d in drafts] == [DraftKind.FAQ] and llm.calls == []
    assert skip.skipped is False


def test_no_skip_when_templates_answer(world, ev, route):
    llm = FakeLLMClient(responses=[])
    clock = Monotonic()
    with short_of_time(clock), budget.llm_skip_scope() as skip:
        greeting = DefaultAnswerGenerator(llm, CFG).generate(
            make_context("在吗", world), route(Intent.GREETING), EvidenceBundle()
        )
        clarify = DefaultAnswerGenerator(llm, CFG).generate(
            make_context("这个是纯棉的吗？", world),
            route(Intent.PRODUCT_SPEC),
            EvidenceBundle(clarify_question="请问您问的是哪一款呢？"),
        )
        nothing = DefaultAnswerGenerator(llm, CFG).generate(
            make_context("这个是纯棉的吗？", world), route(Intent.PRODUCT_SPEC), EvidenceBundle()
        )
    assert [d.kind for d in greeting] == [DraftKind.TEMPLATE]
    assert [d.kind for d in clarify] == [DraftKind.TEMPLATE]
    assert nothing == []  # 没有依据：不推荐，也不是时间的问题
    assert skip.skipped is False and llm.calls == []


def test_no_skip_when_a_template_takes_the_only_slot(world, ev, route):
    """只要一条推荐、模板已经占了：没有名额给大模型，不记。"""
    rule = ev.platform_rule(PlatformRuleKey.APPLY_AFTERSALE, "在订单页点「申请售后」发起")
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.SEVEN_DAY_RETURN), rule])
    llm = FakeLLMClient(responses=[])
    clock = Monotonic()
    with short_of_time(clock), budget.llm_skip_scope() as skip:
        drafts = DefaultAnswerGenerator(llm, CFG).generate(
            make_context("退货怎么申请", world, config_update={"suggestion_count": 1}),
            route(Intent.RETURN_POLICY),
            bundle,
        )
    assert [d.template_id for d in drafts] == [T.APPLY_AFTERSALE] and llm.calls == []
    assert skip.skipped is False


def test_no_skip_with_time_to_spare_or_without_a_deadline(world, ev, route):
    llm = FakeLLMClient(lambda r: reply(FOLLOWUP if is_followup(r) else DIRECT))
    with budget.llm_skip_scope() as no_deadline:
        assert len(generate(llm, world, ev, route)) == 2
    clock = Monotonic()
    with budget.deadline_scope(4.5, min_llm_remaining_s=0.8, now=clock):
        clock.t += 1.0
        with budget.llm_skip_scope() as in_time:
            assert len(generate(llm, world, ev, route)) == 2
    assert no_deadline.skipped is False and in_time.skipped is False
    assert len(llm.calls) == 4


def test_failed_or_unparseable_model_output_is_not_a_skip(world, ev, route):
    """调了但失败 / 读不出来不是"时间不够没调"：不记（引擎不记 budget）。"""
    clock = Monotonic()
    with budget.deadline_scope(4.5, min_llm_remaining_s=0.8, now=clock):
        with budget.llm_skip_scope() as failed:
            assert generate(FakeLLMClient(lambda r: LLMError("接口失败")), world, ev, route) == []
        with budget.llm_skip_scope() as garbled:
            assert generate(FakeLLMClient(lambda r: "不是 JSON"), world, ev, route) == []
    assert failed.skipped is False and garbled.skipped is False
