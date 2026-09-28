"""判断层阈值校准（方案 §13.3）。

对每个问题类型扫描阈值，找出满足目标准确率时覆盖率最高的阈值：
- 智能辅助阈值：target_precision = 0.90（有人把关，追求覆盖率）；
- 全自动阈值：target_precision = 0.98（只对开放全自动的类型计算）。
另给出可靠性曲线（"模型说 0.9"是否真的约 90% 正确），用于判断概率是否需要再校准。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class Sample:
    """一条路由结果：(预测类型, 置信度, 真实类型)，与方案里的三元组一致。"""

    predicted: str
    confidence: float
    true: str


class ThresholdPick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    threshold: float
    precision: float
    coverage: float
    recall: float
    support: int  # 真实类型为该 intent 的样本数


def pick_threshold(
    samples: Sequence[Sample] | Sequence[tuple[str, float, str]],
    intent: str,
    target_precision: float,
    *,
    grid: Iterable[float] | None = None,
) -> ThresholdPick | None:
    """§13.3 的算法：在 0.50–0.99 上扫描，返回达标时覆盖率最高的阈值；达不到目标返回 None。

    与方案伪代码一致：覆盖率 = 预测为该类型且置信度 ≥ 阈值的样本数 / 真实为该类型的样本数；
    另外给出召回率（只数其中预测正确的），便于判断覆盖率里有多少是误判。
    同一覆盖率时取先扫到的更低阈值。
    """
    rows = [s if isinstance(s, Sample) else Sample(*s) for s in samples]
    support = sum(1 for s in rows if s.true == intent)
    if support == 0:
        return None
    best: ThresholdPick | None = None
    for t in grid if grid is not None else (x / 100 for x in range(50, 100)):
        picked = [s for s in rows if s.predicted == intent and s.confidence >= t]
        if not picked:
            continue
        correct = sum(1 for s in picked if s.true == intent)
        precision = correct / len(picked)
        coverage = len(picked) / support
        if precision >= target_precision and (best is None or coverage > best.coverage):
            best = ThresholdPick(
                intent=intent,
                threshold=t,
                precision=round(precision, 4),
                coverage=round(coverage, 4),
                recall=round(correct / support, 4),
                support=support,
            )
    return best


def calibrate(
    samples: Sequence[Sample],
    intents: Iterable[str],
    *,
    target_precision: float,
) -> dict[str, ThresholdPick | None]:
    """对多个类型分别校准。"""
    return {i: pick_threshold(samples, i, target_precision) for i in intents}


class ReliabilityBin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability(samples: Sequence[Sample], *, bins: int = 5) -> tuple[list[ReliabilityBin], float]:
    """可靠性分箱与期望校准误差（ECE）。置信度落在 [lower, upper) 的样本一箱，最后一箱含 1.0。"""
    edges = [i / bins for i in range(bins + 1)]
    out: list[ReliabilityBin] = []
    ece = 0.0
    total = len(samples)
    for lo, hi in zip(edges, edges[1:], strict=False):
        last = hi == edges[-1]
        members = [s for s in samples if lo <= s.confidence < hi or (last and s.confidence == hi)]
        if not members:
            continue
        mean_conf = sum(s.confidence for s in members) / len(members)
        acc = sum(1 for s in members if s.predicted == s.true) / len(members)
        out.append(
            ReliabilityBin(
                lower=lo,
                upper=hi,
                count=len(members),
                mean_confidence=round(mean_conf, 4),
                accuracy=round(acc, 4),
            )
        )
        ece += len(members) / total * abs(mean_conf - acc)
    return out, round(ece, 4)
