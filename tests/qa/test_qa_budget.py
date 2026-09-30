"""质检的时间预算（方案 §12.1 质检 ≤ 1 秒、§12.2）：剩余时间不够调判断层就不调，判断层答完时截止
时间已过的结论不用、也不写 decision_log（引擎已经不等了，不留迟到的写入）；FAQ / 模板只跑硬规则，
不受影响。没有截止时间时照常调。"""

from __future__ import annotations

import pytest

from rodit.core import budget
from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import QAGateConfig
from rodit.core.enums import DraftKind, Intent, PolicyKey
from rodit.core.models import Draft, EvidenceBundle
from rodit.qa import QA_OUT_OF_TIME, DefaultQualityGate
from rodit.testing.fakes import FakeClock, FakeDecisionProvider, make_context, qa_answers

TEXT = "亲，现货付款后 48 小时内发货哦"


class Monotonic:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def audit() -> InMemoryAuditSink:
    return InMemoryAuditSink()


def setup(world, ev, route, *, advance_in_judge: float = 0.0, clock: Monotonic | None = None):
    calls: list[str] = []

    def judge(request):  # type: ignore[no-untyped-def]
        calls.append(request.state["answer"])
        if clock is not None:
            clock.t += advance_in_judge
        return qa_answers()

    ctx = make_context("什么时候发货", world)
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.SHIP_TIME)])
    return FakeDecisionProvider(judge), calls, ctx, route(Intent.SHIPPING_POLICY), bundle


def llm_draft() -> Draft:
    return Draft(text=TEXT, kind=DraftKind.LLM, evidence_ids=["policy:ship_time"])


def test_no_judge_call_without_time_for_it(world, ev, route, audit):
    provider, calls, ctx, r, bundle = setup(world, ev, route)
    gate = DefaultQualityGate(provider, QAGateConfig(), audit, FakeClock(world.now))
    clock = Monotonic()
    with budget.deadline_scope(4.5, min_llm_remaining_s=0.8, now=clock):
        clock.t += 4.0  # 还剩 0.5 秒 < 0.8 秒
        result = gate.check(ctx, r, llm_draft(), bundle)
    assert calls == []
    assert not result.passed and result.reasons == [QA_OUT_OF_TIME]
    assert not result.degraded and not result.model_checked  # 不是判断层不可用
    assert audit.decisions == []


def test_answer_after_the_deadline_is_not_used_or_logged(world, ev, route, audit):
    clock = Monotonic()
    provider, calls, ctx, r, bundle = setup(world, ev, route, advance_in_judge=10, clock=clock)
    gate = DefaultQualityGate(provider, QAGateConfig(), audit, FakeClock(world.now))
    with budget.deadline_scope(4.5, min_llm_remaining_s=0.8, now=clock):
        result = gate.check(ctx, r, llm_draft(), bundle)
        # 截止时间过了，FAQ / 模板照常只跑硬规则
        faq = Draft(text=TEXT, kind=DraftKind.FAQ, faq_id=3, evidence_ids=["policy:ship_time"])
        faq_result = gate.check(ctx, r, faq, bundle)
    assert calls == [TEXT]  # 判断层调了（只调了大模型草稿那一次）
    assert not result.passed and QA_OUT_OF_TIME in result.reasons
    assert audit.decisions == []  # 截止之后不写 decision_log
    assert faq_result.passed


def test_judge_answer_in_time_is_used_and_logged(world, ev, route, audit):
    clock = Monotonic()
    provider, calls, ctx, r, bundle = setup(world, ev, route, advance_in_judge=1.0, clock=clock)
    gate = DefaultQualityGate(provider, QAGateConfig(), audit, FakeClock(world.now))
    with budget.deadline_scope(4.5, min_llm_remaining_s=0.8, now=clock):
        clock.t += 3.25  # 还剩 1.25 秒：够调判断层，判断层用 1 秒答完，还没到截止时间
        result = gate.check(ctx, r, llm_draft(), bundle)
        clock.t += 1.0  # 剩余时间不够了：下一份不再调
        late = gate.check(ctx, r, llm_draft(), bundle)
    assert result.passed and result.model_checked
    assert not late.passed and late.reasons == [QA_OUT_OF_TIME]
    assert calls == [TEXT] and len(audit.decisions) == 1
    # 没有截止时间（离线任务、测试）：照常调
    assert gate.check(ctx, r, llm_draft(), bundle).passed and calls == [TEXT, TEXT]
