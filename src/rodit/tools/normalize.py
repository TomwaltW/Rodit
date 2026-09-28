"""平台原始数据归一与平台自由文本脱敏（方案 §7.1、§7.2 状态归一、§5.3 最小化个人信息）。

真实适配器（在平台云内）和 mock_platform.MockPlatformTools 共用这里的映射表和脱敏函数。

状态码映射
- 物流：各平台原始状态码 → (LogisticsStatus, SignedDetail | None)。三张表都是合成的，
  接入时按各开放平台文档填写并实测【待实测】。演示世界里的 raw_status 是淘宝风格
  （ACCEPT / TRANSPORT / DELIVERING / SIGN / SIGN_STATION / SIGN_LOCKER / FAILED / REJECT）。
- 未知物流状态码：不猜。轨迹状态取"最近一条能识别的轨迹"的状态（保持上一个已知状态），
  未知码记进 NormalizedStatus.unknown_codes，由调用方上报、补映射表；一条都识别不了（或没有轨迹）
  时按"运输中"处理——不说已签收、不报异常，也不承诺到货时间（§7.2 回答规则）。
- 订单 / 售后状态、售后类型：未知码抛 UnknownPlatformCode（ToolUnavailable 的子类）。
  售后状态猜错可能导致误操作，宁可按"工具暂时查不到"降级（§12.2）。
- 售后原因：未知原因归为 OTHER（审核规则落到 R8 人工决定，不会因此自动建议同意）。

平台自由文本
- TraceEvent.description / location、AfterSaleInfo.note 先过 core.pii.mask_pii（ToolGateway 契约），
  再遮掉取件码：取件码能直接取走包裹，不能出现在回答、提示词和日志里（§7.2 取件指引不含取件码）。
- 没有个人信息的文本保持原样（mask_pii 会做 NFKC 归一，把中文全角标点变成半角）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from rodit.core.enums import (
    AfterSaleReason,
    AfterSaleStatus,
    AfterSaleType,
    LogisticsStatus,
    OrderStatus,
    Platform,
    SignedDetail,
)
from rodit.core.errors import ToolUnavailable
from rodit.core.models import AfterSaleInfo, LogisticsTrace, TraceEvent
from rodit.core.pii import contains_pii, mask_pii


class UnknownPlatformCode(ToolUnavailable):
    """平台返回了映射表里没有的状态码 / 类型码（映射表待补【待实测】）。

    按工具不可用降级（§12.2）。"""


_P = Platform
_L = LogisticsStatus
_D = SignedDetail

# ---------------------------------------------------------------- 物流状态（§7.2）

LOGISTICS_CODES: dict[Platform, dict[str, tuple[LogisticsStatus, SignedDetail | None]]] = {
    # 【待实测】淘宝 taobao.logistics.trace.search 的 action 取值；同一状态有多个码时第一个是规范码
    _P.TAOBAO: {
        "ACCEPT": (_L.PICKED_UP, None),
        "GOT": (_L.PICKED_UP, None),
        "TRANSPORT": (_L.IN_TRANSIT, None),
        "ARRIVAL": (_L.IN_TRANSIT, None),
        "DEPARTURE": (_L.IN_TRANSIT, None),
        "DELIVERING": (_L.OUT_FOR_DELIVERY, None),
        "SIGN": (_L.SIGNED, _D.SELF),
        "SIGN_STATION": (_L.SIGNED, _D.STATION),
        "SIGN_LOCKER": (_L.SIGNED, _D.LOCKER),
        "SIGN_OTHER": (_L.SIGNED, _D.OTHER),
        "AGENT_SIGN": (_L.SIGNED, _D.OTHER),
        "FAILED": (_L.EXCEPTION, None),
        "REJECT": (_L.RETURNING, None),
        "RETURN": (_L.RETURNING, None),
    },
    # 【待实测】抖店 logistics/trackNoRouteDetail 的轨迹状态（合成的数字码）
    _P.DOUYIN: {
        "1": (_L.PICKED_UP, None),
        "2": (_L.IN_TRANSIT, None),
        "3": (_L.OUT_FOR_DELIVERY, None),
        "4": (_L.SIGNED, _D.SELF),
        "41": (_L.SIGNED, _D.STATION),
        "42": (_L.SIGNED, _D.LOCKER),
        "43": (_L.SIGNED, _D.OTHER),
        "5": (_L.EXCEPTION, None),
        "6": (_L.RETURNING, None),  # 退回
        "7": (_L.RETURNING, None),  # 拒签
    },
    # 【待实测】拼多多 pdd.logistics.ordertrace.get 的轨迹状态（合成）
    _P.PDD: {
        "GOT": (_L.PICKED_UP, None),
        "SEND": (_L.IN_TRANSIT, None),
        "ARRIVAL": (_L.IN_TRANSIT, None),
        "DISPATCH": (_L.OUT_FOR_DELIVERY, None),
        "SIGN": (_L.SIGNED, _D.SELF),
        "STA_SIGN": (_L.SIGNED, _D.STATION),
        "CAB_SIGN": (_L.SIGNED, _D.LOCKER),
        "PROXY_SIGN": (_L.SIGNED, _D.OTHER),
        "FAIL": (_L.EXCEPTION, None),
        "RETURN": (_L.RETURNING, None),
        "REJECTION": (_L.RETURNING, None),
    },
}

FALLBACK_LOGISTICS_STATUS = LogisticsStatus.IN_TRANSIT
"""一条轨迹都识别不了（或没有轨迹）时的状态：不说已签收、不报异常。"""


class NormalizedStatus(BaseModel):
    """一条物流轨迹归一后的状态。unknown_codes 非空说明映射表待补。"""

    model_config = ConfigDict(extra="forbid")

    status: LogisticsStatus
    signed_detail: SignedDetail | None = None
    unknown_codes: list[str] = Field(default_factory=list)


def normalize_logistics_code(
    platform: Platform, code: str | None
) -> tuple[LogisticsStatus, SignedDetail | None] | None:
    """单个物流状态码 → 内部状态；未知或为空返回 None。"""
    if code is None:
        return None
    return LOGISTICS_CODES[platform].get(str(code).strip())


def normalize_trace_status(platform: Platform, codes: Sequence[str | None]) -> NormalizedStatus:
    """整条轨迹的状态。codes 按时间升序（与 LogisticsTrace.events 一致）。

    取最近一条能识别的状态码；未知码去重后按出现顺序记进 unknown_codes。
    """
    unknown: list[str] = []
    for code in codes:
        if code is not None and normalize_logistics_code(platform, code) is None:
            if str(code) not in unknown:
                unknown.append(str(code))
    for code in reversed(codes):
        known = normalize_logistics_code(platform, code)
        if known is not None:
            status, detail = known
            return NormalizedStatus(status=status, signed_detail=detail, unknown_codes=unknown)
    return NormalizedStatus(status=FALLBACK_LOGISTICS_STATUS, unknown_codes=unknown)


# ---------------------------------------------------------------- 订单状态（§7.1 订单详情）

_O = OrderStatus
ORDER_STATUS_CODES: dict[Platform, dict[str, OrderStatus]] = {
    # 淘宝 taobao.trade.fullinfo.get 的 status。普通订单签收后仍是 WAIT_BUYER_CONFIRM_GOODS，
    # 是否签收看 sign_time（见 normalize_order_status 的 signed 参数）。
    _P.TAOBAO: {
        "WAIT_BUYER_PAY": _O.UNPAID,
        "WAIT_SELLER_SEND_GOODS": _O.PAID,
        "WAIT_BUYER_CONFIRM_GOODS": _O.SHIPPED,
        "SELLER_CONSIGNED_PART": _O.SHIPPED,
        "TRADE_BUYER_SIGNED": _O.SIGNED,
        "TRADE_FINISHED": _O.COMPLETED,
        "TRADE_CLOSED": _O.CLOSED,
        "TRADE_CLOSED_BY_TAOBAO": _O.CLOSED,
    },
    # 【待实测】抖店 order/orderDetail 的 order_status（合成）；签收看 receipt_time
    _P.DOUYIN: {"1": _O.UNPAID, "2": _O.PAID, "3": _O.SHIPPED, "4": _O.CLOSED, "5": _O.COMPLETED},
    # 【待实测】拼多多 order_status 只有 1 待发货 / 2 已发货待签收 / 3 已签收，交易成功、关闭要结合
    # confirm_status / refund_status 判断；这里合成成一个码
    _P.PDD: {
        "0": _O.UNPAID,
        "1": _O.PAID,
        "2": _O.SHIPPED,
        "3": _O.SIGNED,
        "4": _O.COMPLETED,
        "5": _O.CLOSED,
    },
}


def normalize_order_status(
    platform: Platform, code: object, *, signed: bool = False
) -> OrderStatus:
    """订单状态码 → OrderStatus。已发货且有签收时间时算已签收。未知码抛 UnknownPlatformCode。"""
    status = ORDER_STATUS_CODES[platform].get(str(code))
    if status is None:
        raise UnknownPlatformCode(f"{platform} 订单状态码 {code!r} 未映射【待实测】")
    if status == OrderStatus.SHIPPED and signed:
        return OrderStatus.SIGNED
    return status


# ---------------------------------------------------------------- 售后（§7.1 售后列表 / 详情）

_S = AfterSaleStatus
AFTERSALE_STATUS_CODES: dict[Platform, dict[str, AfterSaleStatus]] = {
    # 淘宝 taobao.refund.get 的 status；REFUNDING 是合成的（淘宝同意后很快变成 SUCCESS）
    _P.TAOBAO: {
        "WAIT_SELLER_AGREE": _S.PENDING_REVIEW,
        "WAIT_BUYER_RETURN_GOODS": _S.WAIT_BUYER_RETURN,
        "WAIT_SELLER_CONFIRM_GOODS": _S.WAIT_SELLER_CONFIRM,
        "REFUNDING": _S.REFUNDING,
        "SUCCESS": _S.REFUND_SUCCESS,
        "SELLER_REFUSE_BUYER": _S.REFUSED,
        "CLOSED": _S.CLOSED,
    },
    # 【待实测】抖店 afterSale/Detail 的 aftersale_status（合成的数字码）
    _P.DOUYIN: {
        "6": _S.PENDING_REVIEW,
        "7": _S.WAIT_BUYER_RETURN,
        "11": _S.WAIT_SELLER_CONFIRM,
        "12": _S.REFUNDING,
        "13": _S.REFUND_SUCCESS,
        "27": _S.REFUSED,
        "28": _S.CLOSED,
    },
    # 【待实测】拼多多 pdd.refund.information.get 的 after_sales_status（合成）
    _P.PDD: {
        "2": _S.PENDING_REVIEW,
        "3": _S.WAIT_BUYER_RETURN,
        "4": _S.WAIT_SELLER_CONFIRM,
        "5": _S.REFUNDING,
        "10": _S.REFUND_SUCCESS,
        "6": _S.REFUSED,
        "1": _S.CLOSED,
    },
}

_T = AfterSaleType
AFTERSALE_TYPE_CODES: dict[Platform, dict[str, AfterSaleType]] = {
    # 【待实测】淘宝用 has_good_return 区分，这里合成成一个类型码
    _P.TAOBAO: {
        "REFUND": _T.REFUND_ONLY,
        "RETURN_GOODS": _T.RETURN_REFUND,
        "EXCHANGE": _T.EXCHANGE,
    },
    # 【待实测】抖店：0 退货退款 / 1 已发货仅退款 / 2 未发货仅退款 / 3 换货
    _P.DOUYIN: {"0": _T.RETURN_REFUND, "1": _T.REFUND_ONLY, "2": _T.REFUND_ONLY, "3": _T.EXCHANGE},
    # 【待实测】拼多多 after_sales_type
    _P.PDD: {"1": _T.REFUND_ONLY, "2": _T.RETURN_REFUND, "3": _T.EXCHANGE},
}

_R = AfterSaleReason
AFTERSALE_REASON_CODES: dict[Platform, dict[str, AfterSaleReason]] = {
    # 淘宝返回原因文字【待实测】
    _P.TAOBAO: {
        "七天无理由退换货": _R.NO_REASON,
        "7天无理由退换货": _R.NO_REASON,
        "不想要了": _R.NO_REASON,
        "质量问题": _R.QUALITY,
        "商品破损": _R.DAMAGED,
        "卖家发错货": _R.WRONG_ITEM,
        "少件/漏发": _R.MISSING_ITEM,
        "未收到货": _R.NOT_RECEIVED,
        "其他": _R.OTHER,
    },
    # 【待实测】抖店原因码（合成）
    _P.DOUYIN: {
        "1": _R.NO_REASON,
        "2": _R.QUALITY,
        "3": _R.DAMAGED,
        "4": _R.WRONG_ITEM,
        "5": _R.MISSING_ITEM,
        "6": _R.NOT_RECEIVED,
        "99": _R.OTHER,
    },
    # 拼多多返回原因文字【待实测】
    _P.PDD: {
        "七天无理由": _R.NO_REASON,
        "质量问题": _R.QUALITY,
        "商品破损/变形": _R.DAMAGED,
        "发错货": _R.WRONG_ITEM,
        "少件/漏发": _R.MISSING_ITEM,
        "未收到货": _R.NOT_RECEIVED,
        "其他原因": _R.OTHER,
    },
}


def normalize_aftersale_status(platform: Platform, code: object) -> AfterSaleStatus:
    status = AFTERSALE_STATUS_CODES[platform].get(str(code))
    if status is None:
        raise UnknownPlatformCode(f"{platform} 售后状态码 {code!r} 未映射【待实测】")
    return status


def normalize_aftersale_type(platform: Platform, code: object) -> AfterSaleType:
    kind = AFTERSALE_TYPE_CODES[platform].get(str(code))
    if kind is None:
        raise UnknownPlatformCode(f"{platform} 售后类型码 {code!r} 未映射【待实测】")
    return kind


def normalize_aftersale_reason(platform: Platform, code: object) -> AfterSaleReason:
    """未知原因归为 OTHER（审核走 R8 人工决定）。"""
    return AFTERSALE_REASON_CODES[platform].get(str(code).strip(), AfterSaleReason.OTHER)


# ---------------------------------------------------------------- 平台自由文本脱敏

PICKUP_CODE_MASK = "[取件码]"
_PICKUP_CODE = re.compile(
    r"((?:取件码|取货码|提货码|取件号|开柜码)\s*(?:是|为)?\s*[:：]?\s*)[A-Za-z0-9][-A-Za-z0-9]{1,15}"
)


def scrub_pickup_code(text: str | None) -> str | None:
    """遮掉取件码的值，保留"取件码"字样（取件指引要提醒买家凭取件码取件）。"""
    if not text:
        return text
    return _PICKUP_CODE.sub(lambda m: m.group(1) + PICKUP_CODE_MASK, text)


def mask_free_text(text: str | None) -> str | None:
    """平台自由文本：个人信息（core.pii.mask_pii）+ 取件码。没有个人信息时保持原文。"""
    if not text:
        return text
    out = mask_pii(text) if contains_pii(text) else text
    return scrub_pickup_code(out)


_M = TypeVar("_M", TraceEvent, LogisticsTrace, AfterSaleInfo)


def mask_platform_text(obj: _M) -> _M:
    """对平台数据里的自由文本脱敏，返回副本（ToolGateway 契约：实现先过 mask_pii）。

    - TraceEvent：description、location；
    - LogisticsTrace：每条轨迹 + pickup_info（驿站 / 快递柜名称要用于取件指引，只遮取件码）；
    - AfterSaleInfo：note（买家填写的说明）。
    """
    if isinstance(obj, TraceEvent):
        return obj.model_copy(
            update={
                "description": mask_free_text(obj.description) or "",
                "location": mask_free_text(obj.location),
            }
        )
    if isinstance(obj, LogisticsTrace):
        return obj.model_copy(
            update={
                "events": [mask_platform_text(e) for e in obj.events],
                "pickup_info": scrub_pickup_code(obj.pickup_info),
            }
        )
    if isinstance(obj, AfterSaleInfo):
        return obj.model_copy(update={"note": mask_free_text(obj.note)})
    raise TypeError(f"不支持的平台数据类型：{type(obj).__name__}")
