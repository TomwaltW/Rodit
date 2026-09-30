"""评测里的耗时（方案 §12.1）和大模型用量、成本（§12.4、§14）。

耗时：每条用例取引擎的 EngineResult.latency_ms（和 engine.result 审计里的 latency_ms 同一份）：
understand / route / retrieve / generate / qa / total，毫秒；没跑的环节记 0，和
rodit.feedback.metrics.qa_report 同一口径（分位数也用那里的最近秩法 percentile）。
各环节的 P95 对照 §12.1 的目标：路由 ≤ 1 秒、检索与工具 ≤ 0.5 秒、生成 ≤ 2.5 秒、质检 ≤ 1 秒、
推荐出现（端到端）P95 ≤ 5 秒。§12.1 只对端到端写明 P95，各环节这里也按 P95 对照。
离线替身的耗时只说明代码路径，不代表真实大模型。

用量：每次大模型调用一条 llm.usage 审计（rodit.llm.MeteredLLMClient：purpose、model、
input_tokens、output_tokens、latency_ms，失败时另有 error = 异常类型名）。按用途（route / generate /
qa）汇总 token、调用次数、失败次数和单次调用耗时；§14 说"路由占了将近一半"，按用途的占比就是看这个。
成本 = 输入 token / 1000 × 输入单价 + 输出 token / 1000 × 输出单价；单价没给时只报 token
（§14：各项单价要在拿到报价后填入）。另按 §14 示例口径（每店每天 900 条消息、每月 30 天）
折算每店每月用量：评测集的问题类型分布和真实咨询不同，只作量级参考。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rodit.feedback.metrics import E2E_P95_TARGET_MS, LatencyStats, percentile

STAGES = ("understand", "route", "retrieve", "generate", "qa", "total")  # 同 engine.LATENCY_KEYS
STAGE_LABELS = {
    "understand": "会话理解",
    "route": "路由",
    "retrieve": "检索与工具",
    "generate": "生成",
    "qa": "质检",
    "total": "端到端",
}
STAGE_TARGETS_MS: dict[str, int | None] = {
    "understand": None,  # §12.1 没有单列
    "route": 1000,
    "retrieve": 500,
    "generate": 2500,
    "qa": 1000,
    "total": E2E_P95_TARGET_MS,
}
PURPOSES = ("route", "generate", "qa")  # 大模型调用的用途（LLMRequest.purpose）
PURPOSE_LABELS = {"route": "路由", "generate": "生成", "qa": "质检", "other": "其他"}
MESSAGES_PER_SHOP_DAY = 900  # §14 单店用量估算示例：每天 900 次推荐
DAYS_PER_MONTH = 30


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ================================================================ 耗时（§12.1）


class StageLatency(_Model):
    label: str
    count: int
    p50: float | None = None
    p95: float | None = None
    max: float | None = None
    target_ms: int | None = None  # §12.1 目标（对照 P95）；None 表示没有单列目标
    within_target: bool | None = None  # 没有目标或没有数据时为 None


class LatencyReport(_Model):
    n: int  # 有耗时数据的用例数
    stages: dict[str, StageLatency]
    e2e_p95_within_target: bool | None = None  # 推荐出现（端到端）P95 ≤ 5 秒


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def latency_report(per_case: Sequence[Mapping[str, Any]]) -> LatencyReport:
    """per_case：每条用例的 latency_ms（没有结果的用例传空 dict，不计入）。"""
    rows = [row for row in per_case if row]
    stages: dict[str, StageLatency] = {}
    for stage in STAGES:
        values = [v for row in rows if (v := _number(row.get(stage, 0))) is not None]
        target = STAGE_TARGETS_MS[stage]
        p95 = percentile(values, 0.95) if values else None
        stages[stage] = StageLatency(
            label=STAGE_LABELS[stage],
            count=len(values),
            p50=percentile(values, 0.5) if values else None,
            p95=p95,
            max=max(values) if values else None,
            target_ms=target,
            within_target=None if target is None or p95 is None else p95 <= target,
        )
    return LatencyReport(
        n=len(rows),
        stages=stages,
        e2e_p95_within_target=stages["total"].within_target,
    )


# ================================================================ 用量与成本（§14）


class Pricing(_Model):
    """单价：每 1000 个 token 的价格（输入、输出分开；单位随报价，如元）。"""

    input_per_1k: float = Field(ge=0)
    output_per_1k: float = Field(ge=0)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens / 1000 * self.input_per_1k + output_tokens / 1000 * self.output_per_1k


class PurposeTokens(_Model):
    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class CaseUsage(_Model):
    """一条用例（一条买家消息）的大模型用量。"""

    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None  # 没有单价时为 None
    by_purpose: dict[str, PurposeTokens] = Field(default_factory=dict)


class PurposeUsage(PurposeTokens):
    token_share: float | None = None  # 该用途的 token 占全部 token 的比例
    call_latency_ms: LatencyStats = Field(default_factory=LatencyStats)  # 单次调用耗时


class UsageReport(_Model):
    cases: int
    calls: int
    errors: int
    error_types: dict[str, int] = Field(default_factory=dict)
    input_tokens: int
    output_tokens: int
    total_tokens: int
    mean_tokens_per_case: float | None = None
    by_purpose: dict[str, PurposeUsage] = Field(default_factory=dict)
    pricing: Pricing | None = None
    cost: float | None = None
    mean_cost_per_case: float | None = None
    projected_monthly_tokens_per_shop: float | None = None  # §14 示例口径
    projected_monthly_cost_per_shop: float | None = None


def _tokens(detail: Mapping[str, Any], key: str) -> int:
    value = _number(detail.get(key))
    return max(0, int(value)) if value is not None else 0


def _purpose(detail: Mapping[str, Any]) -> str:
    purpose = detail.get("purpose")
    return purpose if isinstance(purpose, str) and purpose else "other"


def _ordered(keys: set[str]) -> list[str]:
    return [p for p in PURPOSES if p in keys] + sorted(keys - set(PURPOSES))


def case_usage(calls: Sequence[Mapping[str, Any]], pricing: Pricing | None = None) -> CaseUsage:
    """calls：本用例的 llm.usage 审计 detail。"""
    by: dict[str, PurposeTokens] = {}
    for d in calls:
        p = by.setdefault(_purpose(d), PurposeTokens())
        p.calls += 1
        p.errors += 1 if d.get("error") else 0
        p.input_tokens += _tokens(d, "input_tokens")
        p.output_tokens += _tokens(d, "output_tokens")
    input_tokens = sum(p.input_tokens for p in by.values())
    output_tokens = sum(p.output_tokens for p in by.values())
    return CaseUsage(
        calls=sum(p.calls for p in by.values()),
        errors=sum(p.errors for p in by.values()),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=pricing.cost(input_tokens, output_tokens) if pricing else None,
        by_purpose={k: by[k] for k in _ordered(set(by))},
    )


def usage_report(
    per_case: Sequence[Sequence[Mapping[str, Any]]], pricing: Pricing | None = None
) -> UsageReport:
    """per_case：每条用例的 llm.usage 审计 detail 列表（没调用大模型的用例传空列表）。"""
    calls = [d for case in per_case for d in case]
    totals = case_usage(calls, pricing)
    total_tokens = totals.input_tokens + totals.output_tokens
    latencies: dict[str, list[float]] = {}
    for d in calls:
        if (ms := _number(d.get("latency_ms"))) is not None:
            latencies.setdefault(_purpose(d), []).append(ms)
    by_purpose = {
        k: PurposeUsage(
            **p.model_dump(),
            token_share=(p.input_tokens + p.output_tokens) / total_tokens if total_tokens else None,
            call_latency_ms=LatencyStats.of(latencies.get(k, [])),
        )
        for k, p in totals.by_purpose.items()
    }
    n = len(per_case)
    mean_tokens = total_tokens / n if n else None
    mean_cost = totals.cost / n if n and totals.cost is not None else None
    month = MESSAGES_PER_SHOP_DAY * DAYS_PER_MONTH
    return UsageReport(
        cases=n,
        calls=totals.calls,
        errors=totals.errors,
        error_types=dict(Counter(str(d["error"]) for d in calls if d.get("error")).most_common()),
        input_tokens=totals.input_tokens,
        output_tokens=totals.output_tokens,
        total_tokens=total_tokens,
        mean_tokens_per_case=mean_tokens,
        by_purpose=by_purpose,
        pricing=pricing,
        cost=totals.cost,
        mean_cost_per_case=mean_cost,
        projected_monthly_tokens_per_shop=mean_tokens * month if mean_tokens is not None else None,
        projected_monthly_cost_per_shop=mean_cost * month if mean_cost is not None else None,
    )


__all__ = [
    "DAYS_PER_MONTH",
    "MESSAGES_PER_SHOP_DAY",
    "PURPOSE_LABELS",
    "STAGES",
    "STAGE_LABELS",
    "STAGE_TARGETS_MS",
    "CaseUsage",
    "LatencyReport",
    "Pricing",
    "PurposeTokens",
    "PurposeUsage",
    "StageLatency",
    "UsageReport",
    "case_usage",
    "latency_report",
    "usage_report",
]
