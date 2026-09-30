"""在合成评测集上跑问答引擎（方案 §13.1–§13.3、§13.5 回归）。

每条用例用一套全新的仓储和演示世界（互不影响），先把多轮历史写进会话，再把当前问题交给引擎，
按 §13.2 打分；同时收集路由的 (预测类型, 置信度, 真实类型)，用于 §13.3 阈值校准。
换大模型、改提示词、升级判断层之前先跑这里的回归，指标不下降才上线（§13.5）。

每条用例另记一份过程数据（CaseTrace，不含买家原文）：各环节耗时（EngineResult.latency_ms，
和 engine.result 审计里的 latency_ms 是同一份，§12.1）、降级标记、质检不通过的原因
（engine.result 审计的 qa_reasons，已脱敏）；harness 带计量和草稿记录时（接真实大模型，
见 rodit.eval.harness），还有每次大模型调用的用量（llm.usage 审计的 detail，§14）和模型草稿。
"""

from __future__ import annotations

import random
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
from rodit.eval.scoring import CaseScore, EvalReport, score_case, summarize
from rodit.testing import fixtures
from rodit.testing.fakes import make_message
from rodit.testing.fixtures import DemoWorld


class EngineLike(Protocol):
    def handle(self, msg: StandardMessage) -> EngineResult | None: ...


class EvalHarness(Protocol):
    """一套可以跑用例的引擎：引擎本身、它用的会话仓储、演示世界。"""

    engine: EngineLike
    conversations: ConversationRepository
    world: DemoWorld


@dataclass
class CaseTrace:
    """一条用例的过程数据（见模块说明）。没有的项为空。"""

    latency_ms: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    qa_reasons: list[str] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    drafts: list[str] = field(default_factory=list)
    metered: bool = False  # harness 有没有计量大模型用量（离线替身不计量）


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


def trace_case(harness: EvalHarness, case: EvalCase, result: EngineResult | None) -> CaseTrace:
    """收集一条用例的过程数据。

    harness 的 audit / usage / drafts 都是可选的（见 rodit.eval.harness）。"""
    qa_reasons: list[str] = []
    audit = getattr(harness, "audit", None)
    if audit is not None:
        events = audit.list_events(getattr(fixtures, case.shop), action=AuditAction.RESULT.value)
        if events:
            raw = events[-1].detail.get(ResultKey.QA_REASONS.value) or []
            qa_reasons = [str(r) for r in raw] if isinstance(raw, list) else []
    usage = getattr(harness, "usage", None)
    calls = (
        [dict(e.detail) for e in usage.events if e.action == AuditAction.LLM_USAGE.value]
        if usage is not None
        else []
    )
    recorder = getattr(harness, "drafts", None)
    return CaseTrace(
        latency_ms=dict(result.latency_ms) if result is not None else {},
        degraded=list(result.degraded) if result is not None else [],
        qa_reasons=qa_reasons,
        llm_calls=calls,
        drafts=recorder.drafts() if recorder is not None else [],
        metered=usage is not None,
    )


def run_eval(
    cases: Sequence[EvalCase],
    make_harness: Callable[[], EvalHarness],
    *,
    max_chars: int = 120,
    gate: float = 0.95,
    assist_precision: float = 0.90,
    progress: Progress | None = None,
) -> EvalRun:
    scores: list[CaseScore] = []
    samples: list[Sample] = []
    results: dict[str, EngineResult | None] = {}
    traces: dict[str, CaseTrace] = {}
    for index, case in enumerate(cases, start=1):
        harness = make_harness()
        result = run_case(harness, case)
        results[case.id] = result
        traces[case.id] = trace_case(harness, case, result)
        if progress is not None:
            progress(index, len(cases), case, traces[case.id])
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
