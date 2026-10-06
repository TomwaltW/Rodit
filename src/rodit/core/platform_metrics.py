"""平台客服考核指标的定义（方案 §1.3 目标"平台客服指标较接入前改善"、§2.1 客服考核、§12.4）。

每个平台单元只看本平台的指标（§9.1）；口径以平台为准，接入时按平台文档核对【待确认】。
报表的"接入前后对比"用这里的 unit / higher_is_better 判断变好还是变差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rodit.core.enums import Platform, PlatformMetric

MetricUnit = Literal["ratio", "seconds", "hours"]


@dataclass(frozen=True)
class PlatformMetricSpec:
    metric: PlatformMetric
    platform: Platform
    label: str  # 中文名，给商家看
    unit: MetricUnit  # ratio：0–1 的比例；seconds / hours：时长
    higher_is_better: bool
    note: str  # 平台口径说明（考核时段、哪些回复计入）


PLATFORM_METRIC_SPECS: dict[PlatformMetric, PlatformMetricSpec] = {
    spec.metric: spec
    for spec in (
        PlatformMetricSpec(
            PlatformMetric.TAOBAO_3MIN_RESPONSE_RATE,
            Platform.TAOBAO,
            "旺旺 3 分钟人工响应率",
            "ratio",
            True,
            "真实体验分的一项；全自动机器人的回复不计入，智能辅助下由客服发出的回复计入【较可信】",
        ),
        PlatformMetricSpec(
            PlatformMetric.DOUYIN_AVG_RESPONSE_S,
            Platform.DOUYIN,
            "飞鸽平均响应时长",
            "seconds",
            False,
            "商家体验分服务维度；智能辅助推荐由人工发出算人工回复【较可信】",
        ),
        PlatformMetricSpec(
            PlatformMetric.DOUYIN_DISSATISFACTION_RATE,
            Platform.DOUYIN,
            "会话不满意率",
            "ratio",
            False,
            "商家体验分服务维度【较可信】",
        ),
        PlatformMetricSpec(
            PlatformMetric.DOUYIN_HELP_RATE,
            Platform.DOUYIN,
            "平台求助率",
            "ratio",
            False,
            "商家体验分服务维度【较可信】",
        ),
        PlatformMetricSpec(
            PlatformMetric.DOUYIN_AFTERSALE_REVIEW_H,
            Platform.DOUYIN,
            "售后平均审核时长",
            "hours",
            False,
            "商家体验分服务维度；售后审核助手直接影响这一项（§7.3.2）【较可信】",
        ),
        PlatformMetricSpec(
            PlatformMetric.PDD_3MIN_REPLY_RATE,
            Platform.PDD,
            "3 分钟人工回复率",
            "ratio",
            True,
            "考核 8:00–23:00，机器人回复不计入【较可信】",
        ),
    )
}


def metrics_for(platform: Platform) -> list[PlatformMetric]:
    """本平台考核的指标，按上面的顺序。"""
    return [m for m, spec in PLATFORM_METRIC_SPECS.items() if spec.platform == platform]


def metric_labels() -> dict[str, dict[str, object]]:
    """给网页用的定义：指标 → 中文名、平台、单位、好坏方向、口径说明。"""
    return {
        m.value: {
            "label": s.label,
            "platform": s.platform.value,
            "unit": s.unit,
            "higher_is_better": s.higher_is_better,
            "note": s.note,
        }
        for m, s in PLATFORM_METRIC_SPECS.items()
    }


assert set(PLATFORM_METRIC_SPECS) == set(PlatformMetric)

__all__ = [
    "PLATFORM_METRIC_SPECS",
    "MetricUnit",
    "PlatformMetricSpec",
    "metric_labels",
    "metrics_for",
]
