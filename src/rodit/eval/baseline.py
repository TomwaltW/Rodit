"""回归比较（方案 §13.5）。

§13.5：换大模型、改提示词、升级判断层之前，先在测试集上跑回归，指标不下降才上线。

基线是之前一次 `python -m rodit.eval --json out.json` 的完整输出，里面有逐条结果（cases）。
只比较两次都有的用例（按用例 ID）：基线可以是全量，本次用 --limit / --intent 只跑一部分，
比较的就是这一部分；两次的用例集合不同也不会把"换了题"当成"变好 / 变差"。

比较四项：问题类型准确率、路由准确率（动作对不对：该提醒人工的有没有提醒、该安抚的有没有安抚，
§5.7、§6.6——升级判断层时最容易退步的就是它）、事实正确满分率、合规满分率（§13.2 上线门槛的两项）。
范围：共同用例全部（all）；其中有测试集用例时另比测试集（test，§13.1 留出、§13.5 看这部分）。
任一项在任一范围里比基线低超过容差（tolerance，比例，0.02 = 2 个百分点）就算回归，
命令行以退出码 1 结束。默认容差 0：§13.5 要求指标不下降；真实大模型有采样波动，
可以按用例数给一点容差（一条用例在 n 条里是 1/n）。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

REGRESSION_METRICS = ("intent_accuracy", "action_accuracy", "factual", "compliance")
METRIC_LABELS = {
    "intent_accuracy": "类型准确率",
    "action_accuracy": "路由",
    "factual": "事实正确",
    "compliance": "合规",
}
_EPS = 1e-9


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricDelta(_Model):
    scope: str  # all / test
    metric: str
    n: int
    baseline: float
    current: float
    delta: float
    regressed: bool


class BaselineCheck(_Model):
    source: str  # 基线文件名（不含目录）
    tolerance: float
    common_cases: int
    only_current: int  # 本次有、基线没有的用例（不比较）
    only_baseline: int  # 基线有、本次没跑的用例（不比较）
    deltas: list[MetricDelta]
    passed: bool

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.regressed]


def _intent_ok(row: Mapping[str, Any]) -> bool:
    return row.get("intent_ok") is True


def _action_ok(row: Mapping[str, Any]) -> bool:
    return row.get("action_ok") is True


def _full(dim: str) -> Callable[[Mapping[str, Any]], bool]:
    def check(row: Mapping[str, Any]) -> bool:
        scores = row.get("scores")
        return isinstance(scores, Mapping) and scores.get(dim) == 2

    return check


_CHECKS: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "intent_accuracy": _intent_ok,
    "action_accuracy": _action_ok,
    "factual": _full("factual"),
    "compliance": _full("compliance"),
}


def _rate(rows: Sequence[Mapping[str, Any]], check: Callable[[Mapping[str, Any]], bool]) -> float:
    return sum(1 for r in rows if check(r)) / len(rows)


def _by_id(rows: Sequence[Mapping[str, Any]], what: str) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            raise ValueError(f"{what}的逐条结果格式不对：每条要有字符串 id")
        out[row["id"]] = row
    return out


def load_baseline(path: str | Path) -> list[dict[str, Any]]:
    """读基线文件里的逐条结果。文件不存在、不是 JSON、没有 cases 时抛 ValueError。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"读不了基线文件 {path}：{type(e).__name__}") from None
    except json.JSONDecodeError:
        raise ValueError(f"基线文件 {path} 不是 JSON") from None
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list):
        raise ValueError(f"基线文件 {path} 没有逐条结果 cases：请用本版本的 --json 输出做基线")
    _by_id(cases, "基线")
    return cases


def compare_to_baseline(
    current: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 0.0,
    source: str = "",
) -> BaselineCheck:
    """current / baseline：逐条结果（EvalCliReport.cases 的 JSON 形式）。

    没有共同用例时抛 ValueError。"""
    if not 0 <= tolerance < 1:
        raise ValueError(f"容差必须在 [0, 1) 之间：{tolerance}")
    now, before = _by_id(current, "本次"), _by_id(baseline, "基线")
    common = [i for i in now if i in before]
    if not common:
        raise ValueError("基线和本次没有共同的用例，无法比较")
    scopes = {"all": common, "test": [i for i in common if now[i].get("split") == "test"]}
    deltas: list[MetricDelta] = []
    for scope, ids in scopes.items():
        if not ids:
            continue
        for metric in REGRESSION_METRICS:
            check = _CHECKS[metric]
            b = _rate([before[i] for i in ids], check)
            c = _rate([now[i] for i in ids], check)
            deltas.append(
                MetricDelta(
                    scope=scope,
                    metric=metric,
                    n=len(ids),
                    baseline=round(b, 4),
                    current=round(c, 4),
                    delta=round(c - b, 4),
                    regressed=c < b - tolerance - _EPS,
                )
            )
    return BaselineCheck(
        source=Path(source).name if source else "",
        tolerance=tolerance,
        common_cases=len(common),
        only_current=len(now) - len(common),
        only_baseline=sum(1 for i in before if i not in now),
        deltas=deltas,
        passed=not any(d.regressed for d in deltas),
    )


__all__ = [
    "METRIC_LABELS",
    "REGRESSION_METRICS",
    "BaselineCheck",
    "MetricDelta",
    "compare_to_baseline",
    "load_baseline",
]
