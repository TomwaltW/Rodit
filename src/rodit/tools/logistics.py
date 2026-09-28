"""物流问答规则与异常识别（方案 §7.2、§11.2、§11.3）。

回答规则：
- 始终写明"截至 {时间}"：as_of 取最后一条轨迹的时间，没有轨迹时取获取时间（fetched_at）；
- 驿站或快递柜签收给出取件指引（只写驿站 / 快递柜名称，从不写取件码）；
- 不承诺到货时间：只有平台接口直接给出预计送达时间（estimated_delivery）时才有 eta。

异常识别（初始阈值见 LogisticsConfig，按实际数据调整）：
- 停滞：最后一条轨迹距今超过 stall_hours（严格大于），且未签收、未退回 → 提醒客服催查，
  推荐"已帮您催查"话术；没有任何轨迹时不判停滞（无从比较）；
- 疑难件：状态为 exception → 提醒客服跟进；
- 驿站 / 快递柜签收 → 推荐取件指引；
- 退回、拒签 → 提醒客服，并关联售后。
MVP 不做主动触达，异常只提醒客服（agent_alert），由工具证据并入 EvidenceBundle.alerts。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from rodit.core.clock import fmt_time
from rodit.core.config import LogisticsConfig
from rodit.core.enums import AnomalyKind, LogisticsStatus, SignedDetail
from rodit.core.models import LogisticsAnalysis, LogisticsAnomaly, LogisticsTrace
from rodit.tools.normalize import mask_free_text, scrub_pickup_code

STATUS_LABELS: dict[LogisticsStatus, str] = {
    LogisticsStatus.PICKED_UP: "已揽收",
    LogisticsStatus.IN_TRANSIT: "运输中",
    LogisticsStatus.OUT_FOR_DELIVERY: "派送中",
    LogisticsStatus.SIGNED: "已签收",
    LogisticsStatus.EXCEPTION: "疑难件",
    LogisticsStatus.RETURNING: "退回中",
}

SIGNED_LABELS: dict[SignedDetail, str] = {
    SignedDetail.SELF: "本人签收",
    SignedDetail.STATION: "驿站代收",
    SignedDetail.LOCKER: "快递柜代收",
    SignedDetail.OTHER: "他人代收",
}

_PICKUP_PLACES: dict[SignedDetail, str] = {
    SignedDetail.STATION: "驿站",
    SignedDetail.LOCKER: "快递柜",
}

STALL_BUYER_HINT = "已帮您催查"
EXCEPTION_ALERT = "物流显示疑难件，请联系快递跟进处理"
RETURNING_ALERT = "物流显示包裹退回或拒签，请核实原因，并关联售后处理"


def stall_alert(stall_hours: int) -> str:
    return f"物流超过 {stall_hours} 小时未更新，建议向快递催查"


def pickup_guidance(trace: LogisticsTrace) -> str | None:
    """驿站 / 快递柜签收的取件指引；不是驿站 / 快递柜签收返回 None。只写地点，不写取件码。"""
    if trace.status != LogisticsStatus.SIGNED or trace.signed_detail not in _PICKUP_PLACES:
        return None
    place = scrub_pickup_code(trace.pickup_info) or _PICKUP_PLACES[trace.signed_detail]
    return f"包裹已放在{place}，请凭取件码取件"


class DefaultLogisticsAdvisor:
    """实现 core.ports.LogisticsAdvisor。"""

    def __init__(self, config: LogisticsConfig | None = None) -> None:
        self._config = config or LogisticsConfig()

    def analyze(self, trace: LogisticsTrace, now: datetime) -> LogisticsAnalysis:
        last = trace.last_event
        as_of = last.time if last is not None else trace.fetched_at
        latest = (mask_free_text(last.description) or "") if last is not None else "暂无物流轨迹"

        label = STATUS_LABELS[trace.status]
        if trace.status == LogisticsStatus.SIGNED and trace.signed_detail is not None:
            label = f"{label}（{SIGNED_LABELS[trace.signed_detail]}）"
        summary = f"{label}，截至 {fmt_time(as_of)} {latest}".rstrip()

        anomalies: list[LogisticsAnomaly] = []
        stall = timedelta(hours=self._config.stall_hours)
        if (
            last is not None
            and trace.status not in (LogisticsStatus.SIGNED, LogisticsStatus.RETURNING)
            and now - last.time > stall
        ):
            hours = int((now - last.time).total_seconds() // 3600)
            anomalies.append(
                LogisticsAnomaly(
                    kind=AnomalyKind.STALLED,
                    detail=f"最后一条轨迹在 {fmt_time(last.time)}，已 {hours} 小时没有更新",
                    agent_alert=stall_alert(self._config.stall_hours),
                    buyer_hint=STALL_BUYER_HINT,
                )
            )
        if trace.status == LogisticsStatus.EXCEPTION:
            anomalies.append(
                LogisticsAnomaly(
                    kind=AnomalyKind.EXCEPTION,
                    detail=f"疑难件：{latest}",
                    agent_alert=EXCEPTION_ALERT,
                )
            )
        guidance = pickup_guidance(trace)
        if guidance is not None:
            assert trace.signed_detail is not None
            place = scrub_pickup_code(trace.pickup_info) or _PICKUP_PLACES[trace.signed_detail]
            anomalies.append(
                LogisticsAnomaly(
                    kind=AnomalyKind.STATION_OR_LOCKER,
                    detail=f"{SIGNED_LABELS[trace.signed_detail]}：{place}",
                    buyer_hint=guidance,
                )
            )
        if trace.status == LogisticsStatus.RETURNING:
            anomalies.append(
                LogisticsAnomaly(
                    kind=AnomalyKind.RETURNING,
                    detail=f"退回 / 拒签：{latest}",
                    agent_alert=RETURNING_ALERT,
                )
            )

        return LogisticsAnalysis(
            status=trace.status,
            as_of=as_of,
            summary=summary,
            anomalies=anomalies,
            pickup_guidance=guidance,
            eta=trace.estimated_delivery,
        )
