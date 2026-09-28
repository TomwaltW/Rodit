"""在合成评测集上跑问答引擎（方案 §13.1–§13.3、§13.5 回归）。

每条用例用一套全新的仓储和演示世界（互不影响），先把多轮历史写进会话，再把当前问题交给引擎，
按 §13.2 打分；同时收集路由的 (预测类型, 置信度, 真实类型)，用于 §13.3 阈值校准。
换大模型、改提示词、升级判断层之前先跑这里的回归，指标不下降才上线（§13.5）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol

from rodit.core.enums import Role
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
class EvalRun:
    scores: list[CaseScore]
    report: EvalReport
    samples: list[Sample]
    assist_thresholds: dict[str, ThresholdPick | None] = field(default_factory=dict)
    results: dict[str, EngineResult | None] = field(default_factory=dict)


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


def run_eval(
    cases: Sequence[EvalCase],
    make_harness: Callable[[], EvalHarness],
    *,
    max_chars: int = 120,
    gate: float = 0.95,
    assist_precision: float = 0.90,
) -> EvalRun:
    scores: list[CaseScore] = []
    samples: list[Sample] = []
    results: dict[str, EngineResult | None] = {}
    for case in cases:
        harness = make_harness()
        result = run_case(harness, case)
        results[case.id] = result
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
    )
