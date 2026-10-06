"""模拟平台工具：演示数据 → 各平台"原始报文" → 按真实适配器的路径解析（方案 §7.1、§7.2、§8.2）。

CLAUDE.md：平台接口一律 mock。MockPlatformTools 演示真实适配器（在平台云内）要做的事：
1. 调平台接口拿到原始报文（这里由演示数据按平台格式编码：字段名、状态码、金额单位、
   时间格式各不相同）；
2. 解析：状态码经 tools.normalize 映射成内部状态，金额统一成元（Decimal），时间统一成东八区；
3. 最小化：收件人姓名、电话、地址（淘宝脱敏返回、抖店 / 拼多多密文）直接丢弃，不进内部模型（§5.3）；
4. 脱敏：轨迹描述、买家售后说明先过 mask_platform_text（ToolGateway 契约）。

报文字段和状态码都是合成的【待实测】，接入时以各开放平台文档为准。
买家：真实适配器在平台云内用同一个 BuyerRefHasher 把平台买家 ID 哈希成 buyer_ref；
演示数据里只有 buyer_ref，报文里直接带 buyer_ref。
演示用法：整个演示世界都按所选平台的格式编码（不按店铺的平台过滤），便于用店铺 A 的完整数据
测三个平台的解析；按 shop_id 隔离、排序规则和 InMemoryToolGateway 一致。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Protocol, TypeVar

from rodit.core.clock import CST, Clock
from rodit.core.enums import (
    AfterSaleStatus,
    LogisticsStatus,
    OrderStatus,
    Platform,
    SignedDetail,
)
from rodit.core.errors import ToolUnavailable
from rodit.core.models import (
    AfterSaleInfo,
    LogisticsTrace,
    OrderInfo,
    OrderLine,
    PriceInfo,
    StockInfo,
    TraceEvent,
)
from rodit.tools.normalize import (
    AFTERSALE_REASON_CODES,
    AFTERSALE_STATUS_CODES,
    AFTERSALE_TYPE_CODES,
    LOGISTICS_CODES,
    ORDER_STATUS_CODES,
    mask_platform_text,
    normalize_aftersale_reason,
    normalize_aftersale_status,
    normalize_aftersale_type,
    normalize_logistics_code,
    normalize_order_status,
    normalize_trace_status,
)

_PLACEHOLDER_IMAGE = re.compile(r"^(pic_\d+\.jpg|https://img\.example\.invalid/\d+\.jpg|img_\d+)$")
"""只有凭证张数、没有图片标识的合成售后单，编码时用的占位图名（解码时不当作图片标识）。"""


def _evidence_refs(a: AfterSaleInfo, placeholder: Callable[[int], str]) -> list[str]:
    """凭证图片在平台原始数据里的写法：有图片标识（AfterSaleInfo.evidence_images）用标识，
    张数比标识多的部分用占位图名。"""
    refs = list(a.evidence_images)
    refs += [placeholder(i) for i in range(len(refs), a.evidence_count)]
    return refs


def _image_refs(raw: object) -> list[str]:
    """原始数据里的凭证图片 → 图片标识（去掉占位图名）。"""
    if not isinstance(raw, list):
        return []
    return [r for r in raw if isinstance(r, str) and not _PLACEHOLDER_IMAGE.match(r)]


class WorldData(Protocol):
    """演示数据的形状（rodit.testing.fixtures.DemoWorld 满足；生产代码不导入 testing）。"""

    orders: list[OrderInfo]
    traces: list[LogisticsTrace]
    aftersales: list[AfterSaleInfo]
    stocks: list[StockInfo]
    prices: list[PriceInfo]


# ---------------------------------------------------------------- 编码辅助

_TIME_FMT = "%Y-%m-%d %H:%M:%S"
_K = TypeVar("_K")


def _fmt(dt: datetime | None) -> str | None:
    return dt.astimezone(CST).strftime(_TIME_FMT) if dt is not None else None


def _parse(s: str | None) -> datetime | None:
    return datetime.strptime(s, _TIME_FMT).replace(tzinfo=CST) if s else None


def _ts(dt: datetime | None) -> int | None:
    return int(dt.timestamp()) if dt is not None else None


def _from_ts(ts: int | None) -> datetime | None:
    return datetime.fromtimestamp(ts, CST) if ts is not None else None


def _fen(yuan: Decimal) -> int:
    return int((yuan * 100).to_integral_value(rounding=ROUND_HALF_UP))


def _yuan(fen: int) -> Decimal:
    return (Decimal(fen) / 100).quantize(Decimal("0.01"))


def _first_code(table: dict[str, _K], value: _K) -> str:
    """映射表里第一个映射到 value 的码（规范码）。"""
    for code, v in table.items():
        if v == value:
            return code
    raise KeyError(f"映射表里没有 {value!r}")


def _version(updated_at: datetime) -> str:
    """平台状态版本：用售后单修改时间（秒）表示，和演示数据一致。"""
    return str(int(updated_at.timestamp()))


# ---------------------------------------------------------------- 各平台编解码


class _Codec:
    """一个平台的报文格式。encode_* 只给 mock 用，decode_* 是真实适配器要做的解析。"""

    platform: Platform

    # ---- 物流状态码：演示数据的 raw_status 是淘宝风格，换成本平台的码
    def logistics_code(self, event: TraceEvent, trace: LogisticsTrace, *, last: bool) -> str:
        pair: tuple[LogisticsStatus, SignedDetail | None] | None
        if event.raw_status is not None:
            pair = normalize_logistics_code(Platform.TAOBAO, event.raw_status)
            if pair is None:
                return event.raw_status  # 未知码原样透传，演示"未知码"处理
        elif last:
            pair = (trace.status, trace.signed_detail)
        else:
            pair = (LogisticsStatus.IN_TRANSIT, None)
        return _first_code(LOGISTICS_CODES[self.platform], pair)

    signed_via_sign_time = True
    """普通订单签收后订单状态不变，靠签收时间区分（淘宝的 TRADE_BUYER_SIGNED 只用于货到付款）。"""

    def order_code(self, status: OrderStatus) -> str:
        if status == OrderStatus.SIGNED and self.signed_via_sign_time:
            status = OrderStatus.SHIPPED
        return _first_code(ORDER_STATUS_CODES[self.platform], status)

    def aftersale_codes(self, a: AfterSaleInfo) -> tuple[str, str, str]:
        p = self.platform
        return (
            _first_code(AFTERSALE_STATUS_CODES[p], a.status),
            _first_code(AFTERSALE_TYPE_CODES[p], a.type),
            _first_code(AFTERSALE_REASON_CODES[p], a.reason),
        )

    def encode_order(self, o: OrderInfo) -> dict[str, Any]:
        raise NotImplementedError

    def decode_order(self, raw: dict[str, Any]) -> OrderInfo:
        raise NotImplementedError

    def encode_trace(self, t: LogisticsTrace) -> dict[str, Any]:
        raise NotImplementedError

    def decode_trace(self, raw: dict[str, Any], fetched_at: datetime) -> LogisticsTrace:
        raise NotImplementedError

    def encode_aftersale(self, a: AfterSaleInfo) -> dict[str, Any]:
        raise NotImplementedError

    def decode_aftersale(self, raw: dict[str, Any]) -> AfterSaleInfo:
        raise NotImplementedError

    def encode_stock(self, s: StockInfo) -> dict[str, Any]:
        raise NotImplementedError

    def decode_stock(self, raw: dict[str, Any], fetched_at: datetime) -> StockInfo:
        raise NotImplementedError

    def encode_price(self, p: PriceInfo) -> dict[str, Any]:
        raise NotImplementedError

    def decode_price(self, raw: dict[str, Any], fetched_at: datetime) -> PriceInfo:
        raise NotImplementedError

    # ---- 解析后的公共收尾：状态归一 + 脱敏
    def _trace(
        self,
        *,
        shop_id: str,
        order_id: str,
        waybill_no: str,
        courier: str,
        events: list[TraceEvent],
        pickup_info: str | None,
        estimated_delivery: datetime | None,
        fetched_at: datetime,
    ) -> LogisticsTrace:
        events = sorted(events, key=lambda e: e.time)
        norm = normalize_trace_status(self.platform, [e.raw_status for e in events])
        trace = LogisticsTrace(
            shop_id=shop_id,
            order_id=order_id,
            waybill_no=waybill_no,
            courier=courier,
            status=norm.status,
            signed_detail=norm.signed_detail,
            pickup_info=pickup_info if norm.status == LogisticsStatus.SIGNED else None,
            events=events,
            estimated_delivery=estimated_delivery,
            fetched_at=fetched_at,
        )
        return mask_platform_text(trace)


class _TaobaoCodec(_Codec):
    """淘宝：金额是元（字符串），时间是 "YYYY-MM-DD HH:MM:SS"，收件信息脱敏返回。"""

    platform = Platform.TAOBAO

    def encode_order(self, o: OrderInfo) -> dict[str, Any]:
        return {
            "tid": o.order_id,
            "seller_shop_id": o.shop_id,
            "buyer_ref": o.buyer_ref,
            "status": self.order_code(o.status),
            "orders": [
                {
                    "num_iid": ln.item_id,
                    "sku_id": ln.sku_id,
                    "title": ln.title,
                    "num": ln.quantity,
                    "price": str(ln.price),
                }
                for ln in o.lines
            ],
            "payment": str(o.total_amount),
            "created": _fmt(o.created_at),
            "pay_time": _fmt(o.paid_at),
            "consign_time": _fmt(o.shipped_at),
            "sign_time": _fmt(o.signed_at),
            "out_sid": o.waybill_no,
            "company_name": o.courier,
            "seven_day_return": o.seven_day_return,
            "promise_ship_time": _fmt(o.promised_ship_by),
            # 收件信息：淘宝脱敏返回（合成占位），解析时丢弃
            "receiver_name": "测**",
            "receiver_mobile": "138****0000",
            "receiver_address": "浙江省杭州市西湖区******",
        }

    def decode_order(self, raw: dict[str, Any]) -> OrderInfo:
        return OrderInfo(
            shop_id=raw["seller_shop_id"],
            order_id=raw["tid"],
            buyer_ref=raw["buyer_ref"],
            status=normalize_order_status(
                self.platform, raw["status"], signed=raw.get("sign_time") is not None
            ),
            lines=[
                OrderLine(
                    item_id=ln["num_iid"],
                    sku_id=ln.get("sku_id"),
                    title=ln["title"],
                    quantity=ln["num"],
                    price=Decimal(ln["price"]),
                )
                for ln in raw["orders"]
            ],
            total_amount=Decimal(raw["payment"]),
            created_at=_parse(raw["created"]),
            paid_at=_parse(raw.get("pay_time")),
            shipped_at=_parse(raw.get("consign_time")),
            signed_at=_parse(raw.get("sign_time")),
            waybill_no=raw.get("out_sid"),
            courier=raw.get("company_name"),
            seven_day_return=bool(raw.get("seven_day_return", True)),
            promised_ship_by=_parse(raw.get("promise_ship_time")),
        )

    def encode_trace(self, t: LogisticsTrace) -> dict[str, Any]:
        n = len(t.events)
        return {
            "tid": t.order_id,
            "seller_shop_id": t.shop_id,
            "out_sid": t.waybill_no,
            "company_name": t.courier,
            "trace_list": [
                {
                    "action": self.logistics_code(e, t, last=i == n - 1),
                    "status_time": _fmt(e.time),
                    "status_desc": e.description,
                    "location": e.location,
                }
                for i, e in enumerate(t.events)
            ],
            "pickup_point": t.pickup_info,
            "estimated_delivery_time": _fmt(t.estimated_delivery),
        }

    def decode_trace(self, raw: dict[str, Any], fetched_at: datetime) -> LogisticsTrace:
        return self._trace(
            shop_id=raw["seller_shop_id"],
            order_id=raw["tid"],
            waybill_no=raw["out_sid"],
            courier=raw["company_name"],
            events=[
                TraceEvent(
                    time=_parse(e["status_time"]),
                    description=e["status_desc"],
                    location=e.get("location"),
                    raw_status=e.get("action"),
                )
                for e in raw["trace_list"]
            ],
            pickup_info=raw.get("pickup_point"),
            estimated_delivery=_parse(raw.get("estimated_delivery_time")),
            fetched_at=fetched_at,
        )

    def encode_aftersale(self, a: AfterSaleInfo) -> dict[str, Any]:
        status, kind, reason = self.aftersale_codes(a)
        return {
            "refund_id": a.aftersale_id,
            "tid": a.order_id,
            "seller_shop_id": a.shop_id,
            "buyer_ref": a.buyer_ref,
            "refund_type": kind,
            "reason": reason,
            "status": status,
            "refund_fee": str(a.amount),
            "created": _fmt(a.created_at),
            "modified": _fmt(a.updated_at),
            "evidence_pics": _evidence_refs(a, lambda i: f"pic_{i}.jpg"),
            "desc": a.note,
        }

    def decode_aftersale(self, raw: dict[str, Any]) -> AfterSaleInfo:
        updated = _parse(raw["modified"])
        assert updated is not None
        return mask_platform_text(
            AfterSaleInfo(
                shop_id=raw["seller_shop_id"],
                aftersale_id=raw["refund_id"],
                order_id=raw["tid"],
                buyer_ref=raw["buyer_ref"],
                type=normalize_aftersale_type(self.platform, raw["refund_type"]),
                reason=normalize_aftersale_reason(self.platform, raw["reason"]),
                status=normalize_aftersale_status(self.platform, raw["status"]),
                amount=Decimal(raw["refund_fee"]),
                created_at=_parse(raw["created"]),
                updated_at=updated,
                evidence_count=len(raw.get("evidence_pics") or []),
                evidence_images=_image_refs(raw.get("evidence_pics")),
                status_version=_version(updated),
                note=raw.get("desc"),
            )
        )

    def encode_stock(self, s: StockInfo) -> dict[str, Any]:
        return {
            "num_iid": s.item_id,
            "sku_id": s.sku_id,
            "seller_shop_id": s.shop_id,
            "quantity": s.quantity,
            "with_hold_quantity": 0,
            "in_stock": s.in_stock,
            "restock_note": s.restock_note,
        }

    def decode_stock(self, raw: dict[str, Any], fetched_at: datetime) -> StockInfo:
        return StockInfo(
            shop_id=raw["seller_shop_id"],
            item_id=raw["num_iid"],
            sku_id=raw.get("sku_id"),
            quantity=raw.get("quantity"),
            in_stock=bool(raw["in_stock"]),
            restock_note=raw.get("restock_note"),
            fetched_at=fetched_at,
        )

    def encode_price(self, p: PriceInfo) -> dict[str, Any]:
        return {
            "num_iid": p.item_id,
            "seller_shop_id": p.shop_id,
            "price": str(p.price),
            "promotions": list(p.promotions),
        }

    def decode_price(self, raw: dict[str, Any], fetched_at: datetime) -> PriceInfo:
        return PriceInfo(
            shop_id=raw["seller_shop_id"],
            item_id=raw["num_iid"],
            price=Decimal(raw["price"]),
            promotions=list(raw.get("promotions") or []),
            fetched_at=fetched_at,
        )


class _DouyinCodec(_Codec):
    """抖店：金额是分（整数），时间是 Unix 秒，收件信息是密文（需解密额度，问答不解密）。"""

    platform = Platform.DOUYIN

    def encode_order(self, o: OrderInfo) -> dict[str, Any]:
        return {
            "order_id": o.order_id,
            "shop_id": o.shop_id,
            "buyer_ref": o.buyer_ref,
            "order_status": int(self.order_code(o.status)),
            "sku_order_list": [
                {
                    "product_id": ln.item_id,
                    "sku_id": ln.sku_id,
                    "product_name": ln.title,
                    "item_num": ln.quantity,
                    "origin_amount": _fen(ln.price),
                }
                for ln in o.lines
            ],
            "pay_amount": _fen(o.total_amount),
            "create_time": _ts(o.created_at),
            "pay_time": _ts(o.paid_at),
            "ship_time": _ts(o.shipped_at),
            "receipt_time": _ts(o.signed_at),
            "logistics_code": o.waybill_no,
            "logistics_company": o.courier,
            "support_7day_return": 1 if o.seven_day_return else 0,
            "exp_ship_time": _ts(o.promised_ship_by),
            # 收件信息密文（合成占位），解析时丢弃，问答不解密（§7.1）
            "encrypt_post_receiver": "#synthetic-cipher-name#",
            "encrypt_post_tel": "$synthetic-cipher-tel$",
            "encrypt_post_addr": "#synthetic-cipher-addr#",
        }

    def decode_order(self, raw: dict[str, Any]) -> OrderInfo:
        return OrderInfo(
            shop_id=raw["shop_id"],
            order_id=raw["order_id"],
            buyer_ref=raw["buyer_ref"],
            status=normalize_order_status(
                self.platform, raw["order_status"], signed=raw.get("receipt_time") is not None
            ),
            lines=[
                OrderLine(
                    item_id=ln["product_id"],
                    sku_id=ln.get("sku_id"),
                    title=ln["product_name"],
                    quantity=ln["item_num"],
                    price=_yuan(ln["origin_amount"]),
                )
                for ln in raw["sku_order_list"]
            ],
            total_amount=_yuan(raw["pay_amount"]),
            created_at=_from_ts(raw["create_time"]),
            paid_at=_from_ts(raw.get("pay_time")),
            shipped_at=_from_ts(raw.get("ship_time")),
            signed_at=_from_ts(raw.get("receipt_time")),
            waybill_no=raw.get("logistics_code"),
            courier=raw.get("logistics_company"),
            seven_day_return=raw.get("support_7day_return", 1) == 1,
            promised_ship_by=_from_ts(raw.get("exp_ship_time")),
        )

    def encode_trace(self, t: LogisticsTrace) -> dict[str, Any]:
        n = len(t.events)
        return {
            "order_id": t.order_id,
            "shop_id": t.shop_id,
            "track_no": t.waybill_no,
            "company_name": t.courier,
            "route_list": [
                {
                    "state": self.logistics_code(e, t, last=i == n - 1),
                    "time": _ts(e.time),
                    "content": e.description,
                    "site": e.location,
                }
                for i, e in enumerate(t.events)
            ],
            "pickup_station": t.pickup_info,
            "expected_arrival": _ts(t.estimated_delivery),
        }

    def decode_trace(self, raw: dict[str, Any], fetched_at: datetime) -> LogisticsTrace:
        return self._trace(
            shop_id=raw["shop_id"],
            order_id=raw["order_id"],
            waybill_no=raw["track_no"],
            courier=raw["company_name"],
            events=[
                TraceEvent(
                    time=_from_ts(e["time"]),
                    description=e["content"],
                    location=e.get("site"),
                    raw_status=str(e["state"]) if e.get("state") is not None else None,
                )
                for e in raw["route_list"]
            ],
            pickup_info=raw.get("pickup_station"),
            estimated_delivery=_from_ts(raw.get("expected_arrival")),
            fetched_at=fetched_at,
        )

    def encode_aftersale(self, a: AfterSaleInfo) -> dict[str, Any]:
        status, kind, reason = self.aftersale_codes(a)
        return {
            "aftersale_id": a.aftersale_id,
            "order_id": a.order_id,
            "shop_id": a.shop_id,
            "buyer_ref": a.buyer_ref,
            "aftersale_type": int(kind),
            "reason_code": int(reason),
            "aftersale_status": int(status),
            "refund_amount": _fen(a.amount),
            "create_time": _ts(a.created_at),
            "update_time": _ts(a.updated_at),
            "evidence": _evidence_refs(a, lambda i: f"https://img.example.invalid/{i}.jpg"),
            "buyer_remark": a.note,
        }

    def decode_aftersale(self, raw: dict[str, Any]) -> AfterSaleInfo:
        updated = _from_ts(raw["update_time"])
        assert updated is not None
        return mask_platform_text(
            AfterSaleInfo(
                shop_id=raw["shop_id"],
                aftersale_id=raw["aftersale_id"],
                order_id=raw["order_id"],
                buyer_ref=raw["buyer_ref"],
                type=normalize_aftersale_type(self.platform, raw["aftersale_type"]),
                reason=normalize_aftersale_reason(self.platform, raw["reason_code"]),
                status=normalize_aftersale_status(self.platform, raw["aftersale_status"]),
                amount=_yuan(raw["refund_amount"]),
                created_at=_from_ts(raw["create_time"]),
                updated_at=updated,
                evidence_count=len(raw.get("evidence") or []),
                evidence_images=_image_refs(raw.get("evidence")),
                status_version=_version(updated),
                note=raw.get("buyer_remark"),
            )
        )

    def encode_stock(self, s: StockInfo) -> dict[str, Any]:
        return {
            "product_id": s.item_id,
            "sku_id": s.sku_id,
            "shop_id": s.shop_id,
            "stock_num": s.quantity,
            "in_stock": s.in_stock,
            "restock_note": s.restock_note,
        }

    def decode_stock(self, raw: dict[str, Any], fetched_at: datetime) -> StockInfo:
        return StockInfo(
            shop_id=raw["shop_id"],
            item_id=raw["product_id"],
            sku_id=raw.get("sku_id"),
            quantity=raw.get("stock_num"),
            in_stock=bool(raw["in_stock"]),
            restock_note=raw.get("restock_note"),
            fetched_at=fetched_at,
        )

    def encode_price(self, p: PriceInfo) -> dict[str, Any]:
        return {
            "product_id": p.item_id,
            "shop_id": p.shop_id,
            "price": _fen(p.price),
            "promotions": list(p.promotions),
        }

    def decode_price(self, raw: dict[str, Any], fetched_at: datetime) -> PriceInfo:
        return PriceInfo(
            shop_id=raw["shop_id"],
            item_id=raw["product_id"],
            price=_yuan(raw["price"]),
            promotions=list(raw.get("promotions") or []),
            fetched_at=fetched_at,
        )


class _PddCodec(_Codec):
    """拼多多：金额是分，时间是 "YYYY-MM-DD HH:MM:SS"，收件信息是密文。"""

    platform = Platform.PDD
    signed_via_sign_time = False  # order_status 3 即已签收

    def encode_order(self, o: OrderInfo) -> dict[str, Any]:
        return {
            "order_sn": o.order_id,
            "mall_id": o.shop_id,
            "buyer_ref": o.buyer_ref,
            "order_status": int(self.order_code(o.status)),
            "item_list": [
                {
                    "goods_id": ln.item_id,
                    "sku_id": ln.sku_id,
                    "goods_name": ln.title,
                    "goods_count": ln.quantity,
                    "goods_price": _fen(ln.price),
                }
                for ln in o.lines
            ],
            "pay_amount": _fen(o.total_amount),
            "created_time": _fmt(o.created_at),
            "pay_time": _fmt(o.paid_at),
            "shipping_time": _fmt(o.shipped_at),
            "receive_time": _fmt(o.signed_at),
            "tracking_number": o.waybill_no,
            "shipping_name": o.courier,
            "support_seven_day": o.seven_day_return,
            "last_ship_time": _fmt(o.promised_ship_by),
            # 收件信息密文（合成占位），解析时丢弃
            "receiver_name": "~synthetic-cipher-name~",
            "receiver_phone": "~synthetic-cipher-phone~",
            "address": "~synthetic-cipher-address~",
        }

    def decode_order(self, raw: dict[str, Any]) -> OrderInfo:
        return OrderInfo(
            shop_id=raw["mall_id"],
            order_id=raw["order_sn"],
            buyer_ref=raw["buyer_ref"],
            status=normalize_order_status(
                self.platform, raw["order_status"], signed=raw.get("receive_time") is not None
            ),
            lines=[
                OrderLine(
                    item_id=ln["goods_id"],
                    sku_id=ln.get("sku_id"),
                    title=ln["goods_name"],
                    quantity=ln["goods_count"],
                    price=_yuan(ln["goods_price"]),
                )
                for ln in raw["item_list"]
            ],
            total_amount=_yuan(raw["pay_amount"]),
            created_at=_parse(raw["created_time"]),
            paid_at=_parse(raw.get("pay_time")),
            shipped_at=_parse(raw.get("shipping_time")),
            signed_at=_parse(raw.get("receive_time")),
            waybill_no=raw.get("tracking_number"),
            courier=raw.get("shipping_name"),
            seven_day_return=bool(raw.get("support_seven_day", True)),
            promised_ship_by=_parse(raw.get("last_ship_time")),
        )

    def encode_trace(self, t: LogisticsTrace) -> dict[str, Any]:
        n = len(t.events)
        return {
            "order_sn": t.order_id,
            "mall_id": t.shop_id,
            "tracking_number": t.waybill_no,
            "shipping_name": t.courier,
            "traces": [
                {
                    "status": self.logistics_code(e, t, last=i == n - 1),
                    "time": _fmt(e.time),
                    "info": e.description,
                    "city": e.location,
                }
                for i, e in enumerate(t.events)
            ],
            "station_name": t.pickup_info,
            "predict_arrival_time": _fmt(t.estimated_delivery),
        }

    def decode_trace(self, raw: dict[str, Any], fetched_at: datetime) -> LogisticsTrace:
        return self._trace(
            shop_id=raw["mall_id"],
            order_id=raw["order_sn"],
            waybill_no=raw["tracking_number"],
            courier=raw["shipping_name"],
            events=[
                TraceEvent(
                    time=_parse(e["time"]),
                    description=e["info"],
                    location=e.get("city"),
                    raw_status=e.get("status"),
                )
                for e in raw["traces"]
            ],
            pickup_info=raw.get("station_name"),
            estimated_delivery=_parse(raw.get("predict_arrival_time")),
            fetched_at=fetched_at,
        )

    def encode_aftersale(self, a: AfterSaleInfo) -> dict[str, Any]:
        status, kind, reason = self.aftersale_codes(a)
        return {
            "id": a.aftersale_id,
            "order_sn": a.order_id,
            "mall_id": a.shop_id,
            "buyer_ref": a.buyer_ref,
            "after_sales_type": int(kind),
            "after_sales_reason": reason,
            "after_sales_status": int(status),
            "refund_amount": _fen(a.amount),
            "created_time": _fmt(a.created_at),
            "updated_time": _fmt(a.updated_at),
            "images": _evidence_refs(a, lambda i: f"img_{i}"),
            "remark": a.note,
        }

    def decode_aftersale(self, raw: dict[str, Any]) -> AfterSaleInfo:
        updated = _parse(raw["updated_time"])
        assert updated is not None
        return mask_platform_text(
            AfterSaleInfo(
                shop_id=raw["mall_id"],
                aftersale_id=raw["id"],
                order_id=raw["order_sn"],
                buyer_ref=raw["buyer_ref"],
                type=normalize_aftersale_type(self.platform, raw["after_sales_type"]),
                reason=normalize_aftersale_reason(self.platform, raw["after_sales_reason"]),
                status=normalize_aftersale_status(self.platform, raw["after_sales_status"]),
                amount=_yuan(raw["refund_amount"]),
                created_at=_parse(raw["created_time"]),
                updated_at=updated,
                evidence_count=len(raw.get("images") or []),
                evidence_images=_image_refs(raw.get("images")),
                status_version=_version(updated),
                note=raw.get("remark"),
            )
        )

    def encode_stock(self, s: StockInfo) -> dict[str, Any]:
        return {
            "goods_id": s.item_id,
            "sku_id": s.sku_id,
            "mall_id": s.shop_id,
            "quantity": s.quantity,  # 拼多多可能不给具体数量（None）
            "is_onsale": s.in_stock,
            "restock_note": s.restock_note,
        }

    def decode_stock(self, raw: dict[str, Any], fetched_at: datetime) -> StockInfo:
        return StockInfo(
            shop_id=raw["mall_id"],
            item_id=raw["goods_id"],
            sku_id=raw.get("sku_id"),
            quantity=raw.get("quantity"),
            in_stock=bool(raw["is_onsale"]),
            restock_note=raw.get("restock_note"),
            fetched_at=fetched_at,
        )

    def encode_price(self, p: PriceInfo) -> dict[str, Any]:
        return {
            "goods_id": p.item_id,
            "mall_id": p.shop_id,
            "price": _fen(p.price),
            "promotions": list(p.promotions),
        }

    def decode_price(self, raw: dict[str, Any], fetched_at: datetime) -> PriceInfo:
        return PriceInfo(
            shop_id=raw["mall_id"],
            item_id=raw["goods_id"],
            price=_yuan(raw["price"]),
            promotions=list(raw.get("promotions") or []),
            fetched_at=fetched_at,
        )


_CODECS: dict[Platform, type[_Codec]] = {
    Platform.TAOBAO: _TaobaoCodec,
    Platform.DOUYIN: _DouyinCodec,
    Platform.PDD: _PddCodec,
}


# ---------------------------------------------------------------- 网关


class MockPlatformTools:
    """实现 core.ports.ToolGateway：演示数据 → 平台原始报文 → 解析、归一、脱敏。

    - clock 为空时 fetched_at 沿用演示数据里的时间，给了 clock 就用获取时的 clock.now()；
    - fail_on：方法名集合，调用时抛 ToolUnavailable（模拟限流 / 故障，§12.2）；
    - calls：记录 (方法名, shop_id)；last_raw：最近一次的原始报文（排查、测试用）；
    - unknown_codes：遇到过的未映射物流状态码（映射表待补【待实测】）。
    """

    def __init__(self, world: WorldData, platform: Platform, *, clock: Clock | None = None) -> None:
        self.world = world
        self.platform = platform
        self._codec = _CODECS[platform]()
        self._clock = clock
        self.fail_on: set[str] = set()
        self.calls: list[tuple[str, str]] = []
        self.last_raw: Any = None
        self.unknown_codes: set[str] = set()

    def _enter(self, name: str, shop_id: str) -> None:
        self.calls.append((name, shop_id))
        if name in self.fail_on:
            raise ToolUnavailable(f"{self.platform} {name} 限流（模拟）")

    def _fetched_at(self, original: datetime) -> datetime:
        return self._clock.now() if self._clock is not None else original

    def _order(self, o: OrderInfo) -> OrderInfo:
        raw = self._codec.encode_order(o)
        self.last_raw = raw
        return self._codec.decode_order(raw)

    # ---------------------------------------------------------------- ToolGateway

    def get_order(self, shop_id: str, order_id: str) -> OrderInfo | None:
        self._enter("get_order", shop_id)
        o = next(
            (o for o in self.world.orders if o.shop_id == shop_id and o.order_id == order_id),
            None,
        )
        return self._order(o) if o is not None else None

    def list_recent_orders(
        self, shop_id: str, buyer_ref: str, *, limit: int = 5
    ) -> list[OrderInfo]:
        self._enter("list_recent_orders", shop_id)
        orders = [
            self._order(o)
            for o in self.world.orders
            if o.shop_id == shop_id and o.buyer_ref == buyer_ref
        ]
        orders.sort(key=lambda o: (o.created_at, o.order_id), reverse=True)
        return orders[:limit]

    def get_logistics(self, shop_id: str, order_id: str) -> LogisticsTrace | None:
        self._enter("get_logistics", shop_id)
        t = next(
            (t for t in self.world.traces if t.shop_id == shop_id and t.order_id == order_id),
            None,
        )
        if t is None:
            return None
        raw = self._codec.encode_trace(t)
        self.last_raw = raw
        trace = self._codec.decode_trace(raw, self._fetched_at(t.fetched_at))
        for e in trace.events:
            if normalize_logistics_code(self.platform, e.raw_status) is None and e.raw_status:
                self.unknown_codes.add(e.raw_status)
        return trace

    def _aftersale(self, a: AfterSaleInfo) -> AfterSaleInfo:
        raw = self._codec.encode_aftersale(a)
        self.last_raw = raw
        return self._codec.decode_aftersale(raw)

    def get_aftersale(self, shop_id: str, aftersale_id: str) -> AfterSaleInfo | None:
        self._enter("get_aftersale", shop_id)
        a = next(
            (
                a
                for a in self.world.aftersales
                if a.shop_id == shop_id and a.aftersale_id == aftersale_id
            ),
            None,
        )
        return self._aftersale(a) if a is not None else None

    def list_aftersales(
        self,
        shop_id: str,
        *,
        order_id: str | None = None,
        buyer_ref: str | None = None,
        status: AfterSaleStatus | None = None,
        since: datetime | None = None,
    ) -> list[AfterSaleInfo]:
        self._enter("list_aftersales", shop_id)
        decoded = [self._aftersale(a) for a in self.world.aftersales if a.shop_id == shop_id]
        out = [
            a
            for a in decoded
            if (order_id is None or a.order_id == order_id)
            and (buyer_ref is None or a.buyer_ref == buyer_ref)
            and (status is None or a.status == status)
            and (since is None or a.created_at >= since)
        ]
        out.sort(key=lambda a: (a.created_at, a.aftersale_id))
        return out

    def get_stock(self, shop_id: str, item_id: str, sku_id: str | None = None) -> StockInfo | None:
        self._enter("get_stock", shop_id)
        rows: list[StockInfo] = []
        for s in self.world.stocks:
            if s.shop_id == shop_id and s.item_id == item_id:
                raw = self._codec.encode_stock(s)
                self.last_raw = raw
                rows.append(self._codec.decode_stock(raw, self._fetched_at(s.fetched_at)))
        if sku_id is not None:
            return next((s for s in rows if s.sku_id == sku_id), None)
        if not rows:
            return None
        if len(rows) == 1 and rows[0].sku_id is None:
            return rows[0]
        # 商品级汇总（ToolGateway.get_stock 契约）
        in_stock = any(s.in_stock for s in rows)
        quantity = (
            None if any(s.quantity is None for s in rows) else sum(s.quantity or 0 for s in rows)
        )
        notes = [s.restock_note for s in rows if not s.in_stock and s.restock_note]
        return StockInfo(
            shop_id=shop_id,
            item_id=item_id,
            quantity=quantity,
            in_stock=in_stock,
            restock_note=None if in_stock else (notes[0] if notes else None),
            fetched_at=max(s.fetched_at for s in rows),
        )

    def get_price(self, shop_id: str, item_id: str) -> PriceInfo | None:
        self._enter("get_price", shop_id)
        p = next(
            (p for p in self.world.prices if p.shop_id == shop_id and p.item_id == item_id),
            None,
        )
        if p is None:
            return None
        raw = self._codec.encode_price(p)
        self.last_raw = raw
        return self._codec.decode_price(raw, self._fetched_at(p.fetched_at))
