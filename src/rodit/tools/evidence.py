"""实时工具数据 → 证据（方案 §5.1 答案依据、§5.2.1 实时接口优先级 1、§7、§12.2）。

DefaultToolEvidenceProvider 实现 core.ports.ToolEvidenceProvider，只处理 core.evidence.TOOL_NEEDS：
订单、物流、售后、退换资格、库存、价格。证据的 id / label / data 严格按 core.evidence 的约定
（模型对象 model_dump(mode="json")，读取时用对应模型 model_validate）。

找订单（ORDER / LOGISTICS / AFTERSALE / RETURN_ELIGIBILITY 都要先确定订单）：
1. 会话里有当前订单（ctx.order，指代理解的结果）就用它；
2. 否则从该买家在本店的最近订单里按问题类型挑（ctx.recent_orders；为空时查一次
   ToolGateway.list_recent_orders）：
   - logistics_track / not_received → 已发货或已签收的；urge_ship → 已付款未发货的；
   - refund_progress → 有售后单的（按该订单最近一张售后单的申请时间排）；
   - return_policy / quality_issue → 已签收或交易成功的；其他问题类型 → 全部最近订单；
   - return_policy / quality_issue 且会话里有当前商品（ctx.item）→ 只看含这件商品的订单；
     一单都没有就不找订单（买家问的是这件商品，按店铺政策回答，不拿别的订单的退换资格）；
   取最近的一单；如果还有别的候选和它的时间相差不超过 ambiguity_window（默认 24 小时），
   说明几单同样可能，给出澄清问题"请问您问的是哪个订单呢？"（不猜）；
3. 一单都没有 → bundle.missing 标 MISSING_ORDER。

冲突处理的主题（§5.2.1 实时接口优先级 1，覆盖同主题的政策 / 详情页 / FAQ）：
- 价格与活动：promotions（和店铺满减政策比较；一次只查当前商品，不会互相覆盖）；
- 退换资格：只有"不能无理由退货"时是 seven_day_return（否定说法，覆盖"支持七天无理由"）；
  可以退 / 需要人工判断时不带主题——内容里引用的政策原文可能带否定词，比较会误删一致的说法；
- 库存：topic_for_stock(item, sku)。

库存没定到 SKU（ctx.sku_id 为空，如"M 码"对应黑色 M、白色 M）且商品有多个 SKU、整款合计有货时，
逐个 SKU 再查（最多 MAX_SKU_STOCK_LOOKUPS 个），有缺货的规格就在证据里写清"部分规格有货"、
哪些规格缺货；data 多一个 "skus"（各 SKU 的 StockInfo）。查不了逐个 SKU 时写明"整款合计、
未区分规格"——整款合计的"有货"不能当成某个规格有货。

降级（§12.2）：工具抛 ToolUnavailable 时在 bundle.unavailable 标 core.evidence.unavailable(kind)，
不抛出（CachingToolGateway 已先用过期缓存兜底）。物流异常要提醒客服的话并入 bundle.alerts。
订单、售后单用数据自己的获取时间（CachingToolGateway 记下的 fetched_at）写"截至"；
比 stale_after 还旧，说明是平台故障时的过期缓存：同样在 bundle.unavailable 标该工具，
卡片提示客服核实（§12.2 用缓存数据并注明时间）。

个人信息（§5.3）：证据 content 只由结构化字段拼成，不放买家售后说明；轨迹描述、售后说明在进
证据前再过一次 mask_platform_text（防御：ToolGateway 实现本应已脱敏）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rodit.core import evidence as ev
from rodit.core.clock import Clock, fmt_time
from rodit.core.enums import (
    AfterSaleReason,
    AfterSaleType,
    EvidenceSource,
    Intent,
    OrderStatus,
    PolicyKey,
)
from rodit.core.errors import TenantIsolationError, ToolUnavailable
from rodit.core.evidence import Need
from rodit.core.models import (
    AfterSaleInfo,
    ConversationContext,
    Evidence,
    EvidenceBundle,
    ItemCard,
    LogisticsAnalysis,
    LogisticsTrace,
    OrderInfo,
    PolicyEntry,
    PriceInfo,
    RefundProgress,
    ReturnEligibility,
    Sku,
    StockInfo,
)
from rodit.core.ports import AfterSaleAdvisor, LogisticsAdvisor, ToolGateway
from rodit.tools.normalize import mask_platform_text

CLARIFY_ORDER_QUESTION = "请问您问的是哪个订单呢？"

ORDER_NEEDS = frozenset({Need.ORDER, Need.LOGISTICS, Need.AFTERSALE, Need.RETURN_ELIGIBILITY})
"""要先确定订单的依据。"""

_O = OrderStatus
INTENT_ORDER_STATUSES: dict[Intent, frozenset[OrderStatus]] = {
    Intent.LOGISTICS_TRACK: frozenset({_O.SHIPPED, _O.SIGNED}),
    Intent.NOT_RECEIVED: frozenset({_O.SHIPPED, _O.SIGNED}),
    Intent.URGE_SHIP: frozenset({_O.PAID}),
    Intent.RETURN_POLICY: frozenset({_O.SIGNED, _O.COMPLETED}),
    Intent.QUALITY_ISSUE: frozenset({_O.SIGNED, _O.COMPLETED}),
}
"""没有订单卡片时，各问题类型优先考虑的订单状态（refund_progress 按有没有售后单挑）。"""

ITEM_SCOPED_INTENTS = frozenset({Intent.RETURN_POLICY, Intent.QUALITY_ISSUE})
"""会话里有当前商品时，只从含这件商品的订单里找（问的是"这件"能不能退、有没有质量问题）。"""

MAX_SKU_STOCK_LOOKUPS = 10
"""没定到 SKU 时最多逐个查几个 SKU 的库存（平台接口有额度，§12.1）。"""

ORDER_STATUS_LABELS: dict[OrderStatus, str] = {
    OrderStatus.UNPAID: "待付款",
    OrderStatus.PAID: "已付款，待发货",
    OrderStatus.SHIPPED: "已发货",
    OrderStatus.SIGNED: "已签收",
    OrderStatus.COMPLETED: "交易成功",
    OrderStatus.CLOSED: "已关闭",
}

AFTERSALE_TYPE_LABELS: dict[AfterSaleType, str] = {
    AfterSaleType.REFUND_ONLY: "仅退款",
    AfterSaleType.RETURN_REFUND: "退货退款",
    AfterSaleType.EXCHANGE: "换货",
}

AFTERSALE_REASON_LABELS: dict[AfterSaleReason, str] = {
    AfterSaleReason.NO_REASON: "七天无理由",
    AfterSaleReason.QUALITY: "质量问题",
    AfterSaleReason.DAMAGED: "破损",
    AfterSaleReason.WRONG_ITEM: "错发",
    AfterSaleReason.MISSING_ITEM: "漏发",
    AfterSaleReason.NOT_RECEIVED: "未收到货",
    AfterSaleReason.OTHER: "其他",
}


# ---------------------------------------------------------------- 找订单


class OrderResolution(BaseModel):
    """找订单的结果：order_id 确定了一单；candidates 多于一单说明要澄清；都为空说明没有订单。"""

    model_config = ConfigDict(extra="forbid")

    order_id: str | None = None
    candidates: list[str] = Field(default_factory=list)

    @property
    def needs_clarify(self) -> bool:
        return self.order_id is None and len(self.candidates) > 1


def resolve_order(
    intent: Intent,
    orders: Sequence[OrderInfo],
    *,
    aftersales: Sequence[AfterSaleInfo] = (),
    window: timedelta = timedelta(hours=24),
) -> OrderResolution:
    """没有订单卡片时按问题类型从最近订单里挑一单（纯函数，规则见模块文档）。"""
    scored: list[tuple[datetime, str]] = []
    if intent == Intent.REFUND_PROGRESS:
        latest: dict[str, datetime] = {}
        for a in aftersales:
            if a.order_id not in latest or a.created_at > latest[a.order_id]:
                latest[a.order_id] = a.created_at
        # 有售后单的订单不一定在最近 5 单里：也算候选（由调用方按 ID 查订单）
        scored = [(t, oid) for oid, t in latest.items()]
    else:
        statuses = INTENT_ORDER_STATUSES.get(intent)
        scored = [
            (o.created_at, o.order_id) for o in orders if statuses is None or o.status in statuses
        ]
    if not scored:
        return OrderResolution()
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    top = scored[0][0]
    plausible = [oid for t, oid in scored if top - t <= window]
    if len(plausible) == 1:
        return OrderResolution(order_id=plausible[0], candidates=plausible)
    return OrderResolution(candidates=plausible)


# ---------------------------------------------------------------- 证据文字（不含个人信息）


def order_content(order: OrderInfo) -> str:
    lines = "、".join(f"{ln.title} × {ln.quantity}" for ln in order.lines)
    parts = [f"订单 {order.order_id}：{ORDER_STATUS_LABELS[order.status]}", f"商品：{lines}"]
    parts.append(f"实付 {order.total_amount} 元")
    if order.paid_at is not None:
        parts.append(f"付款时间 {fmt_time(order.paid_at)}")
    if order.status == OrderStatus.PAID and order.promised_ship_by is not None:
        parts.append(f"承诺最晚发货时间 {fmt_time(order.promised_ship_by)}")
    if order.shipped_at is not None:
        parts.append(f"发货时间 {fmt_time(order.shipped_at)}")
    if order.waybill_no:
        parts.append(f"快递 {order.courier or ''} {order.waybill_no}".replace("  ", " "))
    if order.signed_at is not None:
        parts.append(f"签收时间 {fmt_time(order.signed_at)}")
    if not order.seven_day_return:
        parts.append("该商品不支持七天无理由退换")
    return "；".join(parts)


def logistics_content(trace: LogisticsTrace, analysis: LogisticsAnalysis) -> str:
    parts = [f"{trace.courier} {trace.waybill_no}：{analysis.summary}"]
    if analysis.pickup_guidance:
        parts.append(f"取件指引：{analysis.pickup_guidance}")
    if analysis.eta is not None:
        parts.append(f"平台预计送达时间 {fmt_time(analysis.eta)}")
    else:
        parts.append("平台没有给出预计送达时间，不要承诺到货时间")
    hints = [
        a.buyer_hint
        for a in analysis.anomalies
        if a.buyer_hint and a.buyer_hint != analysis.pickup_guidance
    ]
    if hints:
        parts.append("可以告诉买家：" + "；".join(hints))
    return "；".join(parts)


def aftersale_content(info: AfterSaleInfo, progress: RefundProgress) -> str:
    parts = [
        f"售后单 {info.aftersale_id}（订单 {info.order_id}）："
        f"{AFTERSALE_TYPE_LABELS[info.type]}，原因：{AFTERSALE_REASON_LABELS[info.reason]}，"
        f"金额 {info.amount} 元",
        f"进度：{progress.summary}",
    ]
    if info.evidence_count:
        parts.append(f"买家已上传凭证 {info.evidence_count} 张")
    return "；".join(parts)


def return_content(eligibility: ReturnEligibility) -> str:
    verdict = {True: "可以无理由退货", False: "不能无理由退货", None: "需要人工判断"}[
        eligibility.eligible
    ]
    return f"订单 {eligibility.order_id} 退换资格：{verdict}。{eligibility.reason}"


def _item_name(item: ItemCard | None, item_id: str, sku_id: str | None) -> str:
    name = item.title if item is not None else f"商品 {item_id}"
    sku = item.find_sku(sku_id) if item is not None and sku_id else None
    if sku is not None and sku.props():
        name += "（" + "，".join(f"{k}：{v}" for k, v in sku.props().items()) + "）"
    return name


def stock_content(stock: StockInfo, item: ItemCard | None, sku_id: str | None) -> str:
    name = _item_name(item, stock.item_id, sku_id)
    if stock.in_stock:
        text = "有货" + (f"，库存 {stock.quantity} 件" if stock.quantity is not None else "")
    else:
        text = "暂时缺货" + (f"，{stock.restock_note}" if stock.restock_note else "")
    return f"{name}：{text}（截至 {fmt_time(stock.fetched_at)}）"


def _spec(sku: Sku) -> str:
    """规格的简称，如"黑色 / M"。"""
    return " / ".join(str(v) for v in sku.props().values()) or sku.sku_id


def sku_stock_content(
    item: ItemCard, per_sku: Sequence[tuple[Sku, StockInfo]], fetched_at: datetime
) -> str:
    """没定到 SKU、有的规格缺货：逐个规格写清有没有货（不给整款合计，免得被当成某个规格有货）。"""
    out: list[str] = []
    has: list[str] = []
    for sku, stock in per_sku:
        if stock.in_stock:
            qty = f" 库存 {stock.quantity} 件" if stock.quantity is not None else ""
            has.append(_spec(sku) + qty)
        else:
            out.append(_spec(sku) + (f"（{stock.restock_note}）" if stock.restock_note else ""))
    body = ["暂时缺货：" + "、".join(out)]
    if has:
        body.append("有货：" + "、".join(has))
    head = "部分规格有货" if has else "各规格都暂时缺货"
    return f"{item.title}：{head}。{'；'.join(body)}（截至 {fmt_time(fetched_at)}）"


def aggregate_stock_content(stock: StockInfo, item: ItemCard) -> str:
    """只拿到整款合计（逐个 SKU 查不了）：写明没有区分规格。"""
    qty = f"，库存 {stock.quantity} 件" if stock.quantity is not None else ""
    return (
        f"{item.title}：整款合计有货{qty}（未区分规格，具体规格有没有货以下单页为准）"
        f"（截至 {fmt_time(stock.fetched_at)}）"
    )


def price_content(price: PriceInfo, item: ItemCard | None) -> str:
    name = _item_name(item, price.item_id, None)
    promos = "；".join(price.promotions) if price.promotions else "当前没有活动"
    return f"{name}：售价 {price.price} 元；活动：{promos}（截至 {fmt_time(price.fetched_at)}）"


def _dump(obj: BaseModel) -> dict[str, Any]:
    return obj.model_dump(mode="json")


# ---------------------------------------------------------------- 证据提供者


class DefaultToolEvidenceProvider:
    """实现 core.ports.ToolEvidenceProvider。"""

    def __init__(
        self,
        gateway: ToolGateway,
        policies: Callable[[str], dict[str, PolicyEntry]],
        logistics: LogisticsAdvisor,
        aftersale: AfterSaleAdvisor,
        clock: Clock,
        *,
        ambiguity_window: timedelta = timedelta(hours=24),
        recent_limit: int = 5,
        stale_after: timedelta | None = None,
    ) -> None:
        """stale_after：订单、售后单的获取时间比这还旧，视为平台故障时的过期缓存（标降级）；
        用 CachingToolGateway 时传它的 TTL。None 表示不判断。"""
        self._gateway = gateway
        self._policies = policies
        self._logistics = logistics
        self._aftersale = aftersale
        self._clock = clock
        self._window = ambiguity_window
        self._recent_limit = recent_limit
        self._stale_after = stale_after

    def collect(
        self, ctx: ConversationContext, intent: Intent, needs: frozenset[Need]
    ) -> EvidenceBundle:
        needs = frozenset(needs) & ev.TOOL_NEEDS
        bundle = EvidenceBundle()
        if not needs:
            return bundle
        now = self._clock.now()
        run = _Run(ctx=ctx, bundle=bundle, now=now)

        if needs & ORDER_NEEDS:
            order = self._resolve(run, intent)
            if order is not None:
                if Need.ORDER in needs:
                    self._add_order(run, order)
                if Need.LOGISTICS in needs:
                    self._add_logistics(run, order)
                if Need.AFTERSALE in needs:
                    self._add_aftersales(run, order)
                if Need.RETURN_ELIGIBILITY in needs:
                    self._add_return(run, order)

        if needs & {Need.STOCK, Need.PRICE}:
            if ctx.item is None:
                _append_once(bundle.missing, ev.MISSING_ITEM)
            else:
                self._check_shop(ctx.shop_id, ctx.item)
                if Need.STOCK in needs:
                    self._add_stock(run, ctx.item)
                if Need.PRICE in needs:
                    self._add_price(run, ctx.item)
        return bundle

    # ---------------------------------------------------------------- 找订单

    def _resolve(self, run: _Run, intent: Intent) -> OrderInfo | None:
        ctx, bundle = run.ctx, run.bundle
        if ctx.order is not None:
            self._check_shop(ctx.shop_id, ctx.order)
            run.order_fetched_at = ctx.now
            return ctx.order

        if ctx.recent_orders:
            for o in ctx.recent_orders:
                self._check_shop(ctx.shop_id, o)
            orders = list(ctx.recent_orders)
            run.order_fetched_at = ctx.now
        else:
            try:
                orders = self._gateway.list_recent_orders(
                    ctx.shop_id, ctx.buyer_ref, limit=self._recent_limit
                )
            except ToolUnavailable:
                _append_once(bundle.unavailable, ev.unavailable("order"))
                return None
            for o in orders:
                self._check_shop(ctx.shop_id, o)
            run.order_fetched_at = run.now
        # 只考虑当前买家的订单（不回答别人的订单）
        orders = [o for o in orders if o.buyer_ref == ctx.buyer_ref]
        if intent in ITEM_SCOPED_INTENTS and ctx.item is not None:
            item_id = ctx.item.item_id
            orders = [o for o in orders if any(ln.item_id == item_id for ln in o.lines)]
            if not orders:
                return None  # 没买过当前商品：按店铺政策回答，不拿别的订单的退换资格

        aftersales: list[AfterSaleInfo] = []
        if intent == Intent.REFUND_PROGRESS:
            listed = self._list_aftersales(run, buyer_ref=ctx.buyer_ref)
            if listed is None:
                return None
            aftersales = [a for a in listed if a.buyer_ref == ctx.buyer_ref]
            run.buyer_aftersales = aftersales

        res = resolve_order(intent, orders, aftersales=aftersales, window=self._window)
        if res.needs_clarify:
            bundle.clarify_question = CLARIFY_ORDER_QUESTION
            return None
        if res.order_id is None:
            _append_once(bundle.missing, ev.MISSING_ORDER)
            return None
        found = next((o for o in orders if o.order_id == res.order_id), None)
        if found is not None:
            return found
        # 有售后单的订单不在最近订单里：按 ID 查
        try:
            fetched = self._gateway.get_order(ctx.shop_id, res.order_id)
        except ToolUnavailable:
            _append_once(bundle.unavailable, ev.unavailable("order"))
            return None
        if fetched is None:
            _append_once(bundle.missing, ev.MISSING_ORDER)
            return None
        self._check_shop(ctx.shop_id, fetched)
        run.order_fetched_at = run.now
        return fetched

    # ---------------------------------------------------------------- 获取时间与过期缓存

    def _as_of(
        self, run: _Run, fetched_at: datetime | None, fallback: datetime, kind: str
    ) -> datetime:
        """数据的获取时间（没有就用 fallback）；过期缓存（比 stale_after 还旧）标该工具降级。"""
        at = fetched_at or fallback
        if self._stale_after is not None and run.now - at >= self._stale_after:
            _append_once(run.bundle.unavailable, ev.unavailable(kind))
        return at

    def _order_as_of(self, run: _Run, order: OrderInfo) -> datetime:
        return self._as_of(run, order.fetched_at, run.order_fetched_at or run.now, "order")

    # ---------------------------------------------------------------- 各类证据

    def _add_order(self, run: _Run, order: OrderInfo) -> None:
        run.bundle.evidences.append(
            Evidence(
                id=ev.tool_order(order.order_id),
                source=EvidenceSource.REALTIME,
                label=ev.tool_label("order"),
                content=order_content(order),
                data={"order": _dump(order)},
                fetched_at=self._order_as_of(run, order),
            )
        )

    def _add_logistics(self, run: _Run, order: OrderInfo) -> None:
        ctx, bundle = run.ctx, run.bundle
        try:
            trace = self._gateway.get_logistics(ctx.shop_id, order.order_id)
        except ToolUnavailable:
            _append_once(bundle.unavailable, ev.unavailable("logistics"))
            return
        if trace is None:
            return  # 还没发货 / 没有运单：订单证据已经说明状态
        self._check_shop(ctx.shop_id, trace)
        trace = mask_platform_text(trace)
        analysis = self._logistics.analyze(trace, run.now)
        bundle.evidences.append(
            Evidence(
                id=ev.tool_logistics(order.order_id),
                source=EvidenceSource.REALTIME,
                label=ev.tool_label("logistics"),
                content=logistics_content(trace, analysis),
                data={"trace": _dump(trace), "analysis": _dump(analysis)},
                fetched_at=trace.fetched_at,
            )
        )
        for anomaly in analysis.anomalies:
            if anomaly.agent_alert:
                _append_once(bundle.alerts, anomaly.agent_alert)

    def _list_aftersales(self, run: _Run, **query: Any) -> list[AfterSaleInfo] | None:
        try:
            listed = self._gateway.list_aftersales(run.ctx.shop_id, **query)
        except ToolUnavailable:
            _append_once(run.bundle.unavailable, ev.unavailable("aftersale"))
            return None
        for a in listed:
            self._check_shop(run.ctx.shop_id, a)
        return listed

    def _add_aftersales(self, run: _Run, order: OrderInfo) -> None:
        if run.buyer_aftersales is not None:
            listed: list[AfterSaleInfo] | None = run.buyer_aftersales
        else:
            listed = self._list_aftersales(run, order_id=order.order_id)
        if listed is None:
            return
        for info in (a for a in listed if a.order_id == order.order_id):
            info = mask_platform_text(info)
            as_of = self._as_of(run, info.fetched_at, run.now, "aftersale")
            progress = self._aftersale.refund_progress(info, as_of)
            run.bundle.evidences.append(
                Evidence(
                    id=ev.tool_aftersale(info.aftersale_id),
                    source=EvidenceSource.REALTIME,
                    label=ev.tool_label("aftersale"),
                    content=aftersale_content(info, progress),
                    data={"aftersale": _dump(info), "progress": _dump(progress)},
                    fetched_at=as_of,
                )
            )

    def _add_return(self, run: _Run, order: OrderInfo) -> None:
        eligibility = self._aftersale.return_eligibility(
            order, self._policies(run.ctx.shop_id), run.now
        )
        run.bundle.evidences.append(
            Evidence(
                id=ev.tool_return(order.order_id),
                source=EvidenceSource.REALTIME,
                label=ev.tool_label("return"),
                content=return_content(eligibility),
                data={"eligibility": _dump(eligibility)},
                topic=(PolicyKey.SEVEN_DAY_RETURN.value if eligibility.eligible is False else None),
                fetched_at=self._order_as_of(run, order),
            )
        )

    def _add_stock(self, run: _Run, item: ItemCard) -> None:
        ctx, bundle = run.ctx, run.bundle
        sku_id = ctx.sku_id
        try:
            stock = self._gateway.get_stock(ctx.shop_id, item.item_id, sku_id)
        except ToolUnavailable:
            _append_once(bundle.unavailable, ev.unavailable("stock"))
            return
        if stock is None:
            return
        self._check_shop(ctx.shop_id, stock)
        content = stock_content(stock, item, sku_id)
        data: dict[str, Any] = {"stock": _dump(stock)}
        fetched_at = stock.fetched_at
        if sku_id is None and len(item.skus) > 1 and stock.in_stock:
            # 整款合计"有货"可能掩盖某些规格缺货（"M 码"对应的几个 SKU 都卖完了）
            per_sku = self._sku_stocks(run, item)
            if per_sku is None:
                content = aggregate_stock_content(stock, item)
            else:
                data["skus"] = [_dump(s) for _, s in per_sku]
                fetched_at = min([fetched_at, *(s.fetched_at for _, s in per_sku)])
                if any(not s.in_stock for _, s in per_sku):
                    content = sku_stock_content(item, per_sku, fetched_at)
        bundle.evidences.append(
            Evidence(
                id=ev.tool_stock(item.item_id, sku_id),
                source=EvidenceSource.REALTIME,
                label=ev.tool_label("stock"),
                content=content,
                data=data,
                topic=ev.topic_for_stock(item.item_id, sku_id),
                fetched_at=fetched_at,
            )
        )

    def _sku_stocks(self, run: _Run, item: ItemCard) -> list[tuple[Sku, StockInfo]] | None:
        """逐个 SKU 查库存；SKU 太多、查不了或一个都没查到时返回 None（只能用整款合计）。"""
        if len(item.skus) > MAX_SKU_STOCK_LOOKUPS:
            return None
        out: list[tuple[Sku, StockInfo]] = []
        for sku in item.skus:
            try:
                stock = self._gateway.get_stock(run.ctx.shop_id, item.item_id, sku.sku_id)
            except ToolUnavailable:
                return None
            if stock is None:
                continue
            self._check_shop(run.ctx.shop_id, stock)
            out.append((sku, stock))
        return out or None

    def _add_price(self, run: _Run, item: ItemCard) -> None:
        ctx, bundle = run.ctx, run.bundle
        try:
            price = self._gateway.get_price(ctx.shop_id, item.item_id)
        except ToolUnavailable:
            _append_once(bundle.unavailable, ev.unavailable("price"))
            return
        if price is None:
            return
        self._check_shop(ctx.shop_id, price)
        bundle.evidences.append(
            Evidence(
                id=ev.tool_price(item.item_id),
                source=EvidenceSource.REALTIME,
                label=ev.tool_label("price"),
                content=price_content(price, item),
                data={"price": _dump(price)},
                # 和店铺满减政策同主题：实时活动配置和政策冲突时覆盖政策（§5.2.1）
                topic=PolicyKey.PROMOTIONS.value,
                fetched_at=price.fetched_at,
            )
        )

    @staticmethod
    def _check_shop(shop_id: str, obj: Any) -> None:
        other = getattr(obj, "shop_id", shop_id)
        if other != shop_id:
            raise TenantIsolationError(f"证据数据属于店铺 {other}，当前店铺是 {shop_id}（§9.2）")


class _Run:
    """一次 collect 的中间状态。"""

    def __init__(self, *, ctx: ConversationContext, bundle: EvidenceBundle, now: datetime) -> None:
        self.ctx = ctx
        self.bundle = bundle
        self.now = now
        self.order_fetched_at: datetime | None = None
        self.buyer_aftersales: list[AfterSaleInfo] | None = None


def _append_once(target: list[str], value: str) -> None:
    if value not in target:
        target.append(value)
