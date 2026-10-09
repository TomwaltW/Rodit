"""在合成评测集上跑问答引擎（方案 §13.1–§13.3、§13.5 回归）。

每条用例用一套全新的仓储和演示世界（互不影响），先把多轮历史写进会话，再把当前问题交给引擎，
按 §13.2 打分；同时收集路由的 (预测类型, 置信度, 真实类型)，用于 §13.3 阈值校准。
换大模型、改提示词、升级判断层之前先跑这里的回归，指标不下降才上线（§13.5）。

每条用例另记一份过程数据（CaseTrace，不含买家原文）：各环节耗时（EngineResult.latency_ms，
和 engine.result 审计里的 latency_ms 是同一份，§12.1）、降级标记、质检不通过的原因
（engine.result 审计的 qa_reasons，已脱敏）、实际做出路由的判断层；harness 带计量和记录时
（接真实大模型，见 rodit.eval.harness），还有每次大模型调用的用量（llm.usage 审计的 detail，
§14）、模型草稿，和每个用途的调用结果（CallStats：失败、被截断、空、读不出，报告的运行健康度用）；
harness 带审计时还有判断层判过几份草稿（engine.qa 审计的 model_checked 之和）。

读计量之前先等引擎留在后台的调用做完（到截止时间被丢下、还在跑的，engine.wait_background）：
它们的 llm.usage 是 handle() 返回之后才写的，不等就漏了（§14 少算）；迟到的条数记在
CaseTrace.late_calls。等待在 handle() 之外，不计入耗时。

两个只给真实大模型用的开关（默认都关）：min_interval_s 让相邻两条用例的开始至少隔这么久
（限流时用；等待同样在 handle() 之外）；stop_after_failed 在连续这么多条用例的大模型调用没有一次
用得上时停下（EvalAborted）——调用失败，或者有响应但被截断、是空的、读不出（HTTP 200 的这几种同样算：
思考模型把 max_tokens 用完时每次调用都照常计费）。接口整个不可用时不必把剩下的几百条跑完。
"""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol

from rodit.core.audit import AuditAction, ResultKey
from rodit.core.enums import Intent, Role
from rodit.core.models import EngineResult, StandardMessage
from rodit.core.ports import ConversationRepository
from rodit.eval.calibration import Sample, ThresholdPick, calibrate
from rodit.eval.dataset import EvalCase
from rodit.eval.health import error_label
from rodit.eval.scoring import CaseScore, EvalReport, score_case, summarize
from rodit.testing import fixtures
from rodit.testing.fakes import make_message
from rodit.testing.fixtures import DemoWorld

CAUSE_TRUNCATED = "截断"  # 有响应但用不上的调用，原因分三种（一次调用只记一种，按这个先后）：
CAUSE_EMPTY = "空回答"  # 被 max_tokens 截断；文本是空的；
CAUSE_UNREADABLE = "读不出"  # 这个环节的解析器读不出要的东西（含一条回答里有几个 JSON 对象）
UNUSABLE_CAUSES = (CAUSE_TRUNCATED, CAUSE_EMPTY, CAUSE_UNREADABLE)
"""EvalAborted.labels 里除了失败原因的标签（rodit.eval.health.error_label）还会出现的键。"""

_sleep = time.sleep
"""min_interval_s 用它等。测试换的是这个名字，不是全进程的 time.sleep（子进程等待等别处也在用）。"""

BACKGROUND_WAIT_S = 10.0
"""一条用例跑完后最多等引擎的后台调用这么久（它们各自到单次调用的超时就结束，默认 1–2.5 秒）；
没有后台调用时不等。"""


class EngineLike(Protocol):
    def handle(self, msg: StandardMessage) -> EngineResult | None: ...


class EvalHarness(Protocol):
    """一套可以跑用例的引擎：引擎本身、它用的会话仓储、演示世界。"""

    engine: EngineLike
    conversations: ConversationRepository
    world: DemoWorld


@dataclass(frozen=True)
class CallStats:
    """一条用例里一个用途（route / qa / generate）的调用结果，单位是调用次数
    （rodit.eval.harness.CallRecorder 记的；路由、质检一次调用就是一票）。"""

    calls: int = 0
    failed: int = 0  # 调用抛了 LLMError / LLMTimeout
    truncated: int = 0  # 有响应，但 finish_reason 是 length（被 max_tokens 截断）
    empty: int = 0  # 有响应，但文本为空
    unusable: int = 0  # 有响应，但用不上（判断层：这一票作废；生成：解析不出回答）
    usable: int = 0  # 有响应，而且用得上

    @property
    def all_failed(self) -> bool:
        """调过，而且没有一次用得上：每一次不是失败，就是有响应但用不上（被截断、空、读不出）。
        调用抛了别的异常（代码缺陷，只记进 calls）的不算：那不是模型服务的问题。"""
        return self.calls > 0 and self.failed + self.unusable == self.calls


@dataclass
class CaseTrace:
    """一条用例的过程数据（见模块说明）。没有的项为空。"""

    latency_ms: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    qa_reasons: list[str] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    drafts: list[str] = field(default_factory=list)
    metered: bool = False  # harness 有没有计量大模型用量（离线替身不计量）
    # 实际做出路由的判断层（RouteDecision.provider：incloud_llm / rules / classifier / none）；
    # 没有路由结果时为 None
    route_provider: str | None = None
    calls: dict[str, CallStats] = field(default_factory=dict)  # 用途 → 调用结果；harness 没记时为空
    late_calls: int = 0  # handle() 返回之后才写进来的 llm.usage 条数（已经在 llm_calls 里）
    # 有响应但用不上的调用，按原因（UNUSABLE_CAUSES）计数，各用途合计；harness 没记时为空
    unusable_causes: dict[str, int] = field(default_factory=dict)
    # 这条用例里拿到质检判断层结论的草稿数（engine.qa 审计 model_checked 之和：判断层不可用的、
    # 到截止时间结论被丢掉的都不在内）；harness 没有审计时为 None
    judged_drafts: int | None = None

    def stats(self, purpose: str) -> CallStats:
        return self.calls.get(purpose, CallStats())

    @property
    def all_calls_failed(self) -> bool:
        """这条用例调过大模型，而且没有一次用得上（CallStats.all_failed；没调过大模型的不算）。"""
        made = [s for s in self.calls.values() if s.calls]
        return bool(made) and all(s.all_failed for s in made)


class EvalAborted(Exception):
    """连续多条用例的大模型调用没有一次用得上，评测提前停下（run_eval 的 stop_after_failed）。

    finished 是已经跑完的条数（含触发停下的那几条），labels 是这几条里每次调用没用上的原因：
    失败的按 rodit.eval.health.error_label，有响应但用不上的按 UNUSABLE_CAUSES（截断 / 空回答 /
    读不出）。消息是给操作的人看的一句话，不含提示词和模型输出。"""

    def __init__(self, *, finished: int, total: int, streak: int, labels: dict[str, int]) -> None:
        self.finished = finished
        self.total = total
        self.streak = streak
        self.labels = dict(labels)
        seen = "、".join(f"{label} ×{n}" for label, n in labels.items()) or "原因没有记下来"
        super().__init__(
            f"连续 {streak} 条用例的大模型调用没有一次用得上（{seen}），"
            f"跑到第 {finished} / {total} 条停下：没有生成报告，也没有保存结果"
        )


@dataclass
class EvalRun:
    scores: list[CaseScore]
    report: EvalReport
    samples: list[Sample]
    assist_thresholds: dict[str, ThresholdPick | None] = field(default_factory=dict)
    results: dict[str, EngineResult | None] = field(default_factory=dict)
    traces: dict[str, CaseTrace] = field(default_factory=dict)


Progress = Callable[[int, int, EvalCase, CaseTrace], None]
"""progress(第几条（从 1 数）, 总条数, 用例, 过程数据)：每跑完一条调用一次。

真实大模型跑一轮要几分钟，用来显示进度。"""


def select_cases(
    cases: Sequence[EvalCase],
    *,
    intent: Intent | str | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> list[EvalCase]:
    """部分评测（省钱、省时间）：先按期望的问题类型筛选；给了 limit 时再用 seed 随机抽 limit 条
    （同一个 seed 抽到的用例相同，可复现）。返回的用例保持数据集里的原顺序。"""
    if limit is not None and limit < 1:
        raise ValueError(f"limit 必须是正整数：{limit}")
    wanted = Intent(intent) if intent is not None else None
    picked = [c for c in cases if wanted is None or c.expected_intent == wanted]
    if limit is not None and limit < len(picked):
        keep = sorted(random.Random(seed).sample(range(len(picked)), limit))
        picked = [picked[i] for i in keep]
    return picked


def _seed_history(harness: EvalHarness, case: EvalCase, conv: str, shop_id: str, buyer: str):
    now = harness.world.now
    n = len(case.history)
    for i, (role, text) in enumerate(case.history):
        at = now - timedelta(minutes=(n - i) * 2)
        if role == Role.BUYER:
            harness.conversations.add_message(
                make_message(
                    text,
                    world=harness.world,
                    at=at,
                    shop_id=shop_id,
                    conversation_id=conv,
                    msg_id=f"{conv}_h{i}",
                    buyer_ref=buyer,
                    item_id=case.item_id,
                )
            )
        else:
            harness.conversations.add_outgoing(
                shop_id=shop_id,
                conversation_id=conv,
                msg_id=f"{conv}_h{i}",
                role=role,
                text=text,
                at=at,
            )


def _item_text(world: DemoWorld, case: EvalCase) -> str | None:
    """用例商品卡片上的文字（标题、属性、SKU、洗涤 / 使用、详情页），评分时核对成分标注。"""
    if case.item_id is None:
        return None
    shop_id = getattr(fixtures, case.shop)
    card = next(
        (i for i in world.items if i.shop_id == shop_id and i.item_id == case.item_id), None
    )
    if card is None:
        return None
    return card.model_dump_json(
        include={"title", "attributes", "skus", "care", "usage", "detail_text"}
    )


def run_case(harness: EvalHarness, case: EvalCase) -> EngineResult | None:
    shop_id = getattr(fixtures, case.shop)
    buyer = getattr(fixtures, case.buyer)
    conv = f"eval_{case.id}"
    _seed_history(harness, case, conv, shop_id, buyer)
    msg = make_message(
        case.text,
        world=harness.world,
        shop_id=shop_id,
        conversation_id=conv,
        msg_id=f"{conv}_q",
        buyer_ref=buyer,
        item_id=case.item_id,
        sku_id=case.sku_id,
        order_id=case.order_id,
    )
    return harness.engine.handle(msg)


def _usage_calls(usage: Any) -> list[dict[str, Any]]:
    events = list(usage.events)  # 后台线程可能还在写：先拷一份再读
    return [dict(e.detail) for e in events if e.action == AuditAction.LLM_USAGE.value]


def trace_case(harness: EvalHarness, case: EvalCase, result: EngineResult | None) -> CaseTrace:
    """收集一条用例的过程数据，紧接着 run_case 调（迟到的调用按这一刻之后才写进来的算）。

    harness 的 audit / usage / drafts / calls 都是可选的（见 rodit.eval.harness）。带计量时先等
    引擎的后台调用做完再读（最多 BACKGROUND_WAIT_S 秒；引擎没有 wait_background 就不等）。"""
    shop_id = getattr(fixtures, case.shop)
    usage = getattr(harness, "usage", None)
    late = 0
    if usage is not None:
        on_time = len(_usage_calls(usage))
        wait = getattr(harness.engine, "wait_background", None)
        if callable(wait):
            wait(shop_id, BACKGROUND_WAIT_S)
        late = len(_usage_calls(usage)) - on_time
    qa_reasons: list[str] = []
    judged_drafts: int | None = None
    audit = getattr(harness, "audit", None)
    if audit is not None:
        events = audit.list_events(shop_id, action=AuditAction.RESULT.value)
        if events:
            raw = events[-1].detail.get(ResultKey.QA_REASONS.value) or []
            qa_reasons = [str(r) for r in raw] if isinstance(raw, list) else []
        # 一条消息拆成几个问题时每个问题一条 engine.qa；每条用例一套全新的审计，全加起来就是这一条的
        checks = audit.list_events(shop_id, action=AuditAction.QA.value)
        judged_drafts = sum(_count(e.detail.get("model_checked")) for e in checks)
    recorder = getattr(harness, "drafts", None)
    call_recorder = getattr(harness, "calls", None)
    route = result.route if result is not None else None
    return CaseTrace(
        latency_ms=dict(result.latency_ms) if result is not None else {},
        degraded=list(result.degraded) if result is not None else [],
        qa_reasons=qa_reasons,
        llm_calls=_usage_calls(usage) if usage is not None else [],
        drafts=recorder.drafts() if recorder is not None else [],
        metered=usage is not None,
        route_provider=route.provider if route is not None else None,
        calls=call_recorder.stats() if call_recorder is not None else {},
        late_calls=max(0, late),
        unusable_causes=_unusable_causes(call_recorder),
        judged_drafts=judged_drafts,
    )


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _unusable_causes(call_recorder: Any) -> dict[str, int]:
    read = getattr(call_recorder, "unusable_causes", None)  # 别的记录器不一定有
    return dict(read()) if callable(read) else {}


def _failure_labels(traces: Sequence[CaseTrace]) -> dict[str, int]:
    """这几条用例里每次调用没用上的原因 → 次数：失败的调用，加上有响应但用不上的。"""
    labels = Counter(error_label(d) for t in traces for d in t.llm_calls if d.get("error"))
    for t in traces:
        labels.update(t.unusable_causes)
    return dict(labels.most_common())


def run_eval(
    cases: Sequence[EvalCase],
    make_harness: Callable[[], EvalHarness],
    *,
    max_chars: int = 120,
    gate: float = 0.95,
    assist_precision: float = 0.90,
    progress: Progress | None = None,
    min_interval_s: float = 0.0,
    stop_after_failed: int | None = None,
) -> EvalRun:
    """min_interval_s、stop_after_failed 见模块说明；后者触发时抛 EvalAborted。"""
    scores: list[CaseScore] = []
    samples: list[Sample] = []
    results: dict[str, EngineResult | None] = {}
    traces: dict[str, CaseTrace] = {}
    failed_streak: list[CaseTrace] = []
    started: float | None = None
    for index, case in enumerate(cases, start=1):
        if min_interval_s > 0 and started is not None:
            # 在 handle() 之外等：不计入这条或上一条的耗时（§12.1 的数字不受影响）
            wait = min_interval_s - (time.monotonic() - started)
            if wait > 0:
                _sleep(wait)
        started = time.monotonic()
        harness = make_harness()
        result = run_case(harness, case)
        results[case.id] = result
        traces[case.id] = trace = trace_case(harness, case, result)
        if progress is not None:
            progress(index, len(cases), case, trace)
        if stop_after_failed is not None:
            if trace.all_calls_failed:
                failed_streak.append(trace)
            elif any(s.calls for s in trace.calls.values()):
                failed_streak = []  # 没调过大模型的用例不打断连续计数，也不计入
            if len(failed_streak) >= stop_after_failed:
                raise EvalAborted(
                    finished=index,
                    total=len(cases),
                    streak=len(failed_streak),
                    labels=_failure_labels(failed_streak),
                )
        scores.append(
            score_case(
                case, result, max_chars=max_chars, evidence_text=_item_text(harness.world, case)
            )
        )
        if result is not None and result.route is not None:
            samples.append(
                Sample(
                    predicted=str(result.route.intent),
                    confidence=result.route.intent_confidence,
                    true=str(case.expected_intent),
                )
            )
    report = summarize(scores, cases, gate=gate)
    intents = sorted({str(c.expected_intent) for c in cases})
    return EvalRun(
        scores=scores,
        report=report,
        samples=samples,
        assist_thresholds=calibrate(samples, intents, target_precision=assist_precision),
        results=results,
        traces=traces,
    )
