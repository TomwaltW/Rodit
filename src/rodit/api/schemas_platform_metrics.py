"""平台客服指标接口的请求模型（方案 §12.4 平台客服指标接入前后对比、§15 Phase 2 商家报表）。

请求模型一律 extra="forbid"：登记人只取自调用方身份（Principal），请求体里带 recorded_by / source /
shop_id 等字段直接 422（§4.1 原则 3）。响应模型直接用 rodit.app.platform_metrics 和
rodit.feedback.platform_metrics 的模型（只有指标、日期、数值、条数，没有买家数据）。
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from rodit.core.enums import PlatformMetric
from rodit.feedback.platform_metrics import MAX_SAMPLE_SIZE


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricRecordRequest(_Request):
    """商家登记一天的一项平台指标（从平台后台抄录）。value 的单位见 core.platform_metrics：比例是
    0–1（91% 填 0.91），时长是秒或小时；sample_size 是平台后台显示的会话 / 售后单数（可不填）。
    同一指标同一天再登记覆盖之前登记的；平台同步的数据另存，不会被覆盖。"""

    metric: PlatformMetric
    day: date
    value: float = Field(ge=0, allow_inf_nan=False, strict=True)
    sample_size: int | None = Field(default=None, ge=0, le=MAX_SAMPLE_SIZE, strict=True)


__all__ = ["MetricRecordRequest"]
