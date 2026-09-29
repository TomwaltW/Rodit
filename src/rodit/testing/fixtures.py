"""合成的演示数据（CLAUDE.md：只用合成数据，真实商家、买家、订单数据不进仓库）。

一个"演示世界"覆盖方案里的主要场景：
- 店铺 A（s_1001，淘宝，女装）：完整的政策、商品、FAQ、订单、物流、售后；
- 店铺 B（s_2002，淘宝，男装）：和 A 在同一个平台单元（聚石塔）里，故意用相同的 item_id、
  同一个买家（同平台的 buyer_ref 相同）、相同的 FAQ 问题，但内容不同，用来做串店测试（§9.2）；
- 店铺 C（s_3003，拼多多）、店铺 D（s_4004，抖店）：最小数据，测平台差异和配置分层。

售后单覆盖 §7.3.2 的 R1–R8；订单和物流覆盖 §7.2 的异常识别和 §11 的核心流程。
店铺 A 的买家：BUYER 用于聊天场景（o_1001–o_1003、o_1006、o_1008–o_1010，售后只有
as_2001、as_2003 两单）；每个售后审核场景各用一个买家（BUYER_R3 等），只有
BUYER_RISKY 在 30 天内有 4 单售后（R7）。
所有时间相对 DEFAULT_NOW（2026-09-28 10:35，东八区）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from rodit.core.clock import CST
from rodit.core.enums import (
    AfterSaleReason,
    AfterSaleStatus,
    AfterSaleType,
    Intent,
    KnowledgeSource,
    KnowledgeStatus,
    LogisticsStatus,
    OrderStatus,
    Platform,
    PolicyKey,
    ShopStatus,
    SignedDetail,
)
from rodit.core.models import (
    AfterSaleInfo,
    ItemCard,
    KnowledgeItem,
    LogisticsTrace,
    OrderInfo,
    OrderLine,
    PolicyEntry,
    PriceInfo,
    Shop,
    Sku,
    StockInfo,
    TraceEvent,
)
from rodit.core.pii import hash_buyer_id

DEFAULT_NOW = datetime(2026, 9, 28, 10, 35, tzinfo=CST)

SHOP_A = "s_1001"
SHOP_B = "s_2002"
SHOP_C = "s_3003"
SHOP_D = "s_4004"
AGENT = "kf_02"
MANAGER = "kf_manager"
CONV = "conv_889"

TEST_SALT = "rodit-test-salt"  # 只用于测试；生产的盐从环境变量读（PrivacyConfig.buyer_salt_env）

# 平台原始买家 ID（合成）→ buyer_ref。接入层适配器用同一个盐哈希，才能和演示订单对上。
RAW_BUYERS: dict[str, tuple[str, str]] = {
    "BUYER": ("taobao", "tb_demo_buyer_001"),  # 普通买家（聊天场景）
    "BUYER_RISKY": ("taobao", "tb_demo_buyer_009"),  # 30 天内多次售后（R7）
    # 售后审核场景各用一个买家，避免同一买家售后次数叠加误触发 R7
    "BUYER_R3": ("taobao", "tb_demo_buyer_103"),
    "BUYER_R5": ("taobao", "tb_demo_buyer_105"),
    "BUYER_R6": ("taobao", "tb_demo_buyer_106"),
    "BUYER_R8": ("taobao", "tb_demo_buyer_108"),
    "BUYER_REFUND": ("taobao", "tb_demo_buyer_110"),  # 退款进度问答
    "BUYER_DOUYIN": ("douyin", "dy_demo_buyer_001"),
    "BUYER_PDD": ("pdd", "pdd_demo_buyer_001"),
}


def _ref(name: str) -> str:
    platform, raw = RAW_BUYERS[name]
    return hash_buyer_id(platform, raw, salt=TEST_SALT)


BUYER = _ref("BUYER")
BUYER_RISKY = _ref("BUYER_RISKY")
BUYER_R3 = _ref("BUYER_R3")
BUYER_R5 = _ref("BUYER_R5")
BUYER_R6 = _ref("BUYER_R6")
BUYER_R8 = _ref("BUYER_R8")
BUYER_REFUND = _ref("BUYER_REFUND")
BUYER_DOUYIN = _ref("BUYER_DOUYIN")
BUYER_PDD = _ref("BUYER_PDD")


def _t(days: float = 0, hours: float = 0, minutes: float = 0, *, now: datetime) -> datetime:
    """相对 now 往前推的时间。"""
    return now - timedelta(days=days, hours=hours, minutes=minutes)


@dataclass
class DemoWorld:
    now: datetime
    shops: dict[str, Shop] = field(default_factory=dict)
    policies: list[PolicyEntry] = field(default_factory=list)
    items: list[ItemCard] = field(default_factory=list)
    faqs: list[KnowledgeItem] = field(default_factory=list)
    orders: list[OrderInfo] = field(default_factory=list)
    traces: list[LogisticsTrace] = field(default_factory=list)
    aftersales: list[AfterSaleInfo] = field(default_factory=list)
    stocks: list[StockInfo] = field(default_factory=list)
    prices: list[PriceInfo] = field(default_factory=list)

    # ---- 便捷查找（测试用）
    def shop(self, shop_id: str) -> Shop:
        return self.shops[shop_id]

    def item(self, shop_id: str, item_id: str) -> ItemCard:
        return next(i for i in self.items if i.shop_id == shop_id and i.item_id == item_id)

    def order(self, shop_id: str, order_id: str) -> OrderInfo:
        return next(o for o in self.orders if o.shop_id == shop_id and o.order_id == order_id)

    def trace(self, shop_id: str, order_id: str) -> LogisticsTrace:
        return next(t for t in self.traces if t.shop_id == shop_id and t.order_id == order_id)

    def aftersale(self, shop_id: str, aftersale_id: str) -> AfterSaleInfo:
        return next(
            a for a in self.aftersales if a.shop_id == shop_id and a.aftersale_id == aftersale_id
        )

    def policy_map(self, shop_id: str) -> dict[str, PolicyEntry]:
        return {p.key: p for p in self.policies if p.shop_id == shop_id}


# ================================================================ 店铺


def _shops() -> dict[str, Shop]:
    # config_overrides 只写店铺自己改过的项；生效配置 = 平台默认 → 类目默认 → 覆盖项（§9.3）
    return {
        SHOP_A: Shop(
            id=SHOP_A,
            platform=Platform.TAOBAO,
            category="女装",
            status=ShopStatus.ACTIVE,
            config_overrides={
                "display_name": "小雅女装旗舰店",
                "tone": "亲切",
                "salutation": "亲",
                "max_answer_chars": 120,
                "banned_words": ["假一赔十", "全网最低"],
                "suggestion_count": 2,
                "aftersale_amount_limit": "500",
                "manager_ids": [MANAGER],
            },
        ),
        SHOP_B: Shop(
            id=SHOP_B,
            platform=Platform.TAOBAO,
            category="男装",
            status=ShopStatus.ACTIVE,
            config_overrides={"display_name": "木棉男装", "tone": "专业", "salutation": "您"},
        ),
        SHOP_C: Shop(
            id=SHOP_C,
            platform=Platform.PDD,
            category="女装",
            status=ShopStatus.TRIAL,
            # 没有覆盖 max_answer_chars：生效值来自拼多多平台默认（config/rodit.yaml 为 100）
            config_overrides={"display_name": "拼拼小铺", "tone": "简洁"},
        ),
        SHOP_D: Shop(
            id=SHOP_D,
            platform=Platform.DOUYIN,
            category="女装",
            status=ShopStatus.ACTIVE,
            config_overrides={"display_name": "抖抖女装", "shadow_mode": True},
        ),
    }


# ================================================================ 店铺政策（附录 C）

_POLICIES_A: dict[PolicyKey, str] = {
    PolicyKey.SHIP_TIME: "现货付款后 48 小时内发货；预售商品以商品页标注的发货时间为准",
    PolicyKey.PRESALE: "预售款会在商品标题注明【预售】，按页面标注时间发货",
    PolicyKey.COURIER: "默认发中通快递",
    PolicyKey.COURIER_CHOICE: "可以指定顺丰，需补运费差价 10 元",
    PolicyKey.REMOTE_AREAS: "新疆、西藏需补运费 15 元；港澳台及海外暂不发货",
    PolicyKey.SEVEN_DAY_RETURN: "支持七天无理由退换货，要求吊牌完好、未穿着、未洗涤",
    PolicyKey.NON_RETURNABLE: "定制款、贴身内衣不支持七天无理由退换",
    PolicyKey.RETURN_SHIPPING: "七天无理由退货的运费由买家承担；质量问题退货运费由店铺承担",
    PolicyKey.SHIPPING_INSURANCE: "所有订单都赠送运费险",
    PolicyKey.QUALITY_ISSUE: "收到质量问题商品，请在申请售后时上传清晰照片，核实后退款或补发",
    PolicyKey.INVOICE: "可以开具电子普通发票，确认收货后 3 个工作日内开出",
    PolicyKey.GIFT: "目前没有赠品活动",
    PolicyKey.MODIFY_ORDER: "发货前可以修改地址和规格，发货后无法修改",
    PolicyKey.PROMOTIONS: "店铺满 199 减 20，可以叠加平台券",
    PolicyKey.PROMISE_BOUNDARY: "客服不能承诺额外折扣、赠品或到货时间",
    PolicyKey.SIZE_TENDENCY: "版型整体偏小，建议拍大一码",
    PolicyKey.SERVICE_HOURS: "客服在线时间 9:00–23:00",
}

_POLICIES_B: dict[PolicyKey, str] = {
    PolicyKey.SHIP_TIME: "付款后 72 小时内发货",
    PolicyKey.COURIER: "默认发圆通快递",
    PolicyKey.SEVEN_DAY_RETURN: "支持七天无理由退货",
}

_POLICIES_C: dict[PolicyKey, str] = {
    PolicyKey.SHIP_TIME: "付款后 24 小时内发货",
}

_POLICIES_D: dict[PolicyKey, str] = {
    PolicyKey.SHIP_TIME: "付款后 48 小时内发货",
    PolicyKey.COURIER: "默认发韵达快递",
}


def _policies(now: datetime) -> list[PolicyEntry]:
    out: list[PolicyEntry] = []
    tables = (
        (SHOP_A, _POLICIES_A),
        (SHOP_B, _POLICIES_B),
        (SHOP_C, _POLICIES_C),
        (SHOP_D, _POLICIES_D),
    )
    for shop_id, table in tables:
        for key, value in table.items():
            out.append(
                PolicyEntry(shop_id=shop_id, key=key.value, value=value, updated_at=_t(3, now=now))
            )
    return out


# ================================================================ 商品知识（§5.2.2）


def _items(now: datetime) -> list[ItemCard]:
    dress = ItemCard(
        shop_id=SHOP_A,
        platform=Platform.TAOBAO,
        item_id="i_555",
        title="法式碎花连衣裙",
        category="女装/连衣裙",
        attributes={"面料": "棉 95%、氨纶 5%", "厚薄": "常规", "袖长": "短袖", "版型": "修身"},
        skus=[
            Sku(sku_id="k_1", 颜色="黑色", 尺码="M"),
            Sku(sku_id="k_2", 颜色="黑色", 尺码="L"),
            Sku(sku_id="k_3", 颜色="白色", 尺码="S"),
            Sku(sku_id="k_4", 颜色="白色", 尺码="M"),
        ],
        size_chart=[
            {"尺码": "S", "胸围cm": 84, "腰围cm": 66, "衣长cm": 98},
            {"尺码": "M", "胸围cm": 88, "腰围cm": 70, "衣长cm": 100},
            {"尺码": "L", "胸围cm": 92, "腰围cm": 74, "衣长cm": 102},
            {"尺码": "XL", "胸围cm": 96, "腰围cm": 78, "衣长cm": 104},
        ],
        size_advice=(
            "版型偏小，建议拍大一码；身高 160–165 cm、体重 100–115 斤建议 M；"
            "身高 165–170 cm、体重 115–125 斤建议 L"
        ),
        care=["建议手洗", "不可暴晒", "不可漂白"],
        forbidden_claims=["防晒", "抗菌"],
        # 故意和店铺政策（48 小时）冲突：政策优先级更高（§5.2.1）
        detail_text="法式复古碎花，收腰设计显瘦。现货付款后 24 小时内发货。面料亲肤透气。",
        updated_at=_t(2, now=now),
    )
    jeans = ItemCard(
        shop_id=SHOP_A,
        platform=Platform.TAOBAO,
        item_id="i_777",
        title="高腰阔腿牛仔裤",
        category="女装/牛仔裤",
        attributes={"面料": "棉 98%、氨纶 2%", "弹力": "微弹", "裤长": "长裤", "腰型": "高腰"},
        skus=[
            Sku(sku_id="k_71", 颜色="浅蓝", 尺码="26"),
            Sku(sku_id="k_72", 颜色="浅蓝", 尺码="27"),
            Sku(sku_id="k_73", 颜色="浅蓝", 尺码="28"),
        ],
        # 只有尺码表、没有商家尺码建议：推荐要更保守（§5.2.2）
        size_chart=[
            {"尺码": "26", "腰围cm": 66, "臀围cm": 90, "裤长cm": 100},
            {"尺码": "27", "腰围cm": 68, "臀围cm": 93, "裤长cm": 101},
            {"尺码": "28", "腰围cm": 71, "臀围cm": 96, "裤长cm": 102},
        ],
        care=["反面机洗", "首次洗涤会轻微掉色，建议单独洗"],
        updated_at=_t(5, now=now),
    )
    shirt = ItemCard(
        shop_id=SHOP_A,
        platform=Platform.TAOBAO,
        item_id="i_888",
        title="定制刺绣真丝衬衫",
        category="女装/衬衫",
        attributes={"面料": "桑蚕丝 100%", "定制": "可绣姓名首字母", "袖长": "长袖"},
        skus=[Sku(sku_id="k_81", 颜色="米白", 尺码="均码")],
        care=["专业干洗", "不可机洗"],
        forbidden_claims=["抗皱"],
        detail_text="定制款下单后 7 天内发货，定制商品不支持七天无理由退换。",
        updated_at=_t(10, now=now),
    )
    # 店铺 B：和 A 相同的 item_id，内容完全不同（串店测试）
    b_pants = ItemCard(
        shop_id=SHOP_B,
        platform=Platform.TAOBAO,
        item_id="i_555",
        title="男士休闲裤",
        category="男装/休闲裤",
        attributes={"面料": "涤纶 100%"},
        skus=[Sku(sku_id="k_1", 颜色="卡其", 尺码="XL")],
        size_advice="B 店专用尺码建议：正常码",
        updated_at=_t(1, now=now),
    )
    c_dress = ItemCard(
        shop_id=SHOP_C,
        platform=Platform.PDD,
        item_id="i_c1",
        title="纯色打底衫",
        category="女装/T恤",
        attributes={"面料": "棉 100%"},
        updated_at=_t(1, now=now),
    )
    d_skirt = ItemCard(
        shop_id=SHOP_D,
        platform=Platform.DOUYIN,
        item_id="i_d1",
        title="百褶半身裙",
        category="女装/半身裙",
        attributes={"面料": "聚酯纤维 100%"},
        updated_at=_t(1, now=now),
    )
    return [dress, jeans, shirt, b_pants, c_dress, d_skirt]


# ================================================================ FAQ（§5.2.1 优先级 4）


def _faqs(now: datetime) -> list[KnowledgeItem]:
    def faq(
        shop_id: str,
        q: str,
        a: str,
        *,
        intent: Intent | None = None,
        item_id: str | None = None,
        source: KnowledgeSource = KnowledgeSource.MERCHANT,
        status: KnowledgeStatus = KnowledgeStatus.APPROVED,
    ) -> KnowledgeItem:
        return KnowledgeItem(
            shop_id=shop_id,
            question=q,
            answer=a,
            source=source,
            status=status,
            intent=intent,
            item_id=item_id,
            approved_by="merchant" if status == KnowledgeStatus.APPROVED else None,
            created_at=_t(7, now=now),
        )

    return [
        faq(
            SHOP_A,
            "发什么快递",
            "亲，默认发中通快递哦，需要顺丰的话补 10 元运费差价就可以啦~",
            intent=Intent.SHIPPING_POLICY,
        ),
        faq(
            SHOP_A,
            "什么时候发货",
            "亲，现货付款后 48 小时内发货哦~",
            intent=Intent.SHIPPING_POLICY,
        ),
        faq(
            SHOP_A,
            "能开发票吗",
            "亲，可以开电子普通发票，确认收货后 3 个工作日内开出哦~",
            intent=Intent.INVOICE_GIFT,
        ),
        faq(SHOP_A, "有运费险吗", "亲，每个订单都送运费险的哦~", intent=Intent.INVOICE_GIFT),
        faq(
            SHOP_A,
            "这款会透吗",
            "亲，这款是常规厚度，白色款在强光下会略透，建议搭配浅色内搭哦~",
            intent=Intent.PRODUCT_SPEC,
            item_id="i_555",
        ),
        # 候选知识：客服采纳过、商家还没审核，不能用于回答
        faq(
            SHOP_A,
            "可以寄到新疆吗",
            "亲，新疆可以发，需要补 15 元运费哦~",
            intent=Intent.SHIPPING_POLICY,
            source=KnowledgeSource.ADOPTED,
            status=KnowledgeStatus.CANDIDATE,
        ),
        # 已下线
        faq(
            SHOP_A,
            "双十一有活动吗",
            "亲，双十一全场五折哦~",
            intent=Intent.PRICE_PROMO,
            status=KnowledgeStatus.RETIRED,
        ),
        # 店铺 B：同样的问题，不同的答案（串店测试）
        faq(SHOP_B, "发什么快递", "您好，默认发圆通快递。", intent=Intent.SHIPPING_POLICY),
    ]


# ================================================================ 订单、物流、售后（§7）


def _line(item_id: str, sku_id: str | None, title: str, price: str, qty: int = 1) -> OrderLine:
    return OrderLine(
        item_id=item_id, sku_id=sku_id, title=title, quantity=qty, price=Decimal(price)
    )


def _orders(now: datetime) -> list[OrderInfo]:
    dress, jeans, shirt = "法式碎花连衣裙", "高腰阔腿牛仔裤", "定制刺绣真丝衬衫"

    def order(
        order_id: str,
        status: OrderStatus,
        lines: list[OrderLine],
        *,
        shop_id: str = SHOP_A,
        buyer_ref: str = BUYER,
        paid_days: float,
        shipped_days: float | None = None,
        signed_days: float | None = None,
        waybill: str | None = None,
        courier: str | None = "中通快递",
        seven_day_return: bool = True,
        promised_ship_hours: float = 48,
    ) -> OrderInfo:
        paid_at = _t(paid_days, now=now)
        return OrderInfo(
            shop_id=shop_id,
            order_id=order_id,
            buyer_ref=buyer_ref,
            status=status,
            lines=lines,
            total_amount=sum((ln.price * ln.quantity for ln in lines), Decimal("0")),
            created_at=paid_at - timedelta(minutes=2),
            paid_at=paid_at,
            shipped_at=_t(shipped_days, now=now) if shipped_days is not None else None,
            signed_at=_t(signed_days, now=now) if signed_days is not None else None,
            waybill_no=waybill,
            courier=courier if waybill else None,
            seven_day_return=seven_day_return,
            promised_ship_by=paid_at + timedelta(hours=promised_ship_hours),
        )

    S = OrderStatus
    return [
        # 已付款未发货（R2、催发货）
        order("o_1001", S.PAID, [_line("i_555", "k_1", dress, "259.00")], paid_days=0.6),
        # 运输中（§11.2 物流查询、§11.5 质检拦截）
        order(
            "o_1002",
            S.SHIPPED,
            [_line("i_777", "k_72", jeans, "199.00")],
            paid_days=3,
            shipped_days=1.7,
            waybill="ZT7300000001",
        ),
        # 驿站签收（§11.3 显示签收但没收到、R4）
        order(
            "o_1003",
            S.SIGNED,
            [_line("i_555", "k_4", dress, "259.00")],
            paid_days=5,
            shipped_days=4,
            signed_days=0.8,
            waybill="ZT7300000002",
        ),
        # 本人签收 5 天，支持七天无理由（R3）
        order(
            "o_1004",
            S.SIGNED,
            [_line("i_777", "k_71", jeans, "199.00")],
            paid_days=9,
            shipped_days=8,
            signed_days=5,
            waybill="ZT7300000003",
            buyer_ref=BUYER_R3,
        ),
        # 签收 23 天、已确认收货（超过七天；R5 质量问题）
        order(
            "o_1005",
            S.COMPLETED,
            [_line("i_555", "k_2", dress, "259.00")],
            paid_days=27,
            shipped_days=26,
            signed_days=23,
            waybill="ZT7300000004",
            buyer_ref=BUYER_R5,
        ),
        # 物流停滞超过 48 小时（§7.2 异常）
        order(
            "o_1006",
            S.SHIPPED,
            [_line("i_555", "k_3", dress, "259.00")],
            paid_days=5.2,
            shipped_days=4,
            waybill="ZT7300000005",
        ),
        # 定制商品，不支持七天无理由（R8）
        order(
            "o_1007",
            S.SIGNED,
            [_line("i_888", "k_81", shirt, "459.00")],
            paid_days=10,
            shipped_days=4,
            signed_days=2,
            waybill="ZT7300000006",
            seven_day_return=False,
            promised_ship_hours=7 * 24,  # 定制款下单后 7 天内发货
            buyer_ref=BUYER_R8,
        ),
        # 疑难件（§7.2 异常）
        order(
            "o_1008",
            S.SHIPPED,
            [_line("i_777", "k_73", jeans, "199.00")],
            paid_days=4,
            shipped_days=3,
            waybill="ZT7300000007",
        ),
        # 拒签退回（§7.2 异常）
        order(
            "o_1009",
            S.SHIPPED,
            [_line("i_555", "k_1", dress, "259.00")],
            paid_days=8,
            shipped_days=7,
            waybill="ZT7300000008",
        ),
        # 快递柜签收（§7.2 取件指引）
        order(
            "o_1010",
            S.SIGNED,
            [_line("i_777", "k_72", jeans, "199.00")],
            paid_days=2.9,
            shipped_days=2,
            signed_days=0.06,
            waybill="SF1400000009",
            courier="顺丰速运",
        ),
        # 大额订单（R3 + R6 金额超过店铺上限）
        order(
            "o_1011",
            S.SIGNED,
            [_line("i_555", "k_1", dress, "259.00", qty=6)],
            paid_days=6,
            shipped_days=5,
            signed_days=3,
            waybill="ZT7300000010",
            buyer_ref=BUYER_R6,
        ),
        # 退款中（退款进度问答）
        order(
            "o_1012",
            S.COMPLETED,
            [_line("i_777", "k_71", jeans, "199.00")],
            paid_days=12,
            shipped_days=11,
            signed_days=8,
            waybill="ZT7300000011",
            buyer_ref=BUYER_REFUND,
        ),
        # 质量问题、凭证齐全（as_2009）
        order(
            "o_1013",
            S.SIGNED,
            [_line("i_777", "k_73", jeans, "199.00")],
            paid_days=7,
            shipped_days=6,
            signed_days=1.5,
            waybill="ZT7300000012",
            buyer_ref=BUYER_R5,
        ),
        # 高频售后买家（R7）：前两单已退货退款成功，订单关闭
        *[
            order(
                f"o_200{i}",
                S.CLOSED if i <= 2 else S.SIGNED,
                [_line("i_777", "k_72", jeans, "199.00")],
                buyer_ref=BUYER_RISKY,
                paid_days=days + 4,
                shipped_days=days + 3,
                signed_days=days,
                waybill=f"ZT73000001{i:02d}",
            )
            for i, days in ((1, 25), (2, 18), (3, 10), (4, 2))
        ],
        # 店铺 B：同一个买家、不同店铺（串店测试）
        order(
            "o_9001",
            S.PAID,
            [_line("i_555", "k_1", "男士休闲裤", "129.00")],
            shop_id=SHOP_B,
            paid_days=1,
        ),
        # 店铺 D（抖店）
        order(
            "o_d001",
            S.PAID,
            [_line("i_d1", None, "百褶半身裙", "89.00")],
            shop_id=SHOP_D,
            buyer_ref=BUYER_DOUYIN,
            paid_days=0.5,
        ),
    ]


def _ev(now: datetime, days: float, desc: str, loc: str | None, raw: str) -> TraceEvent:
    return TraceEvent(time=_t(days, now=now), description=desc, location=loc, raw_status=raw)


def _traces(now: datetime) -> list[LogisticsTrace]:
    L = LogisticsStatus

    def trace(
        order_id: str,
        waybill: str,
        status: LogisticsStatus,
        events: list[TraceEvent],
        *,
        courier: str = "中通快递",
        signed_detail: SignedDetail | None = None,
        pickup_info: str | None = None,
    ) -> LogisticsTrace:
        return LogisticsTrace(
            shop_id=SHOP_A,
            order_id=order_id,
            waybill_no=waybill,
            courier=courier,
            status=status,
            signed_detail=signed_detail,
            pickup_info=pickup_info,
            events=events,
            fetched_at=now,
        )

    last_event = now - timedelta(minutes=5)  # 默认 now 下是 10:30（§11.2 "截至 10:30"）
    return [
        trace(
            "o_1002",
            "ZT7300000001",
            L.IN_TRANSIT,
            [
                _ev(now, 1.7, "快件已揽收", "广州市", "ACCEPT"),
                _ev(now, 1.2, "已发往杭州转运中心", "广州转运中心", "TRANSPORT"),
                TraceEvent(
                    time=last_event,
                    description="已到达杭州转运中心",
                    location="杭州转运中心",
                    raw_status="TRANSPORT",
                ),
            ],
        ),
        trace(
            "o_1003",
            "ZT7300000002",
            L.SIGNED,
            [
                _ev(now, 4, "快件已揽收", "广州市", "ACCEPT"),
                _ev(now, 1, "派送中", "杭州市西湖区", "DELIVERING"),
                _ev(now, 0.8, "已签收，菜鸟驿站代收", "杭州市西湖区", "SIGN_STATION"),
            ],
            signed_detail=SignedDetail.STATION,
            pickup_info="菜鸟驿站（幸福里店）",
        ),
        trace(
            "o_1004",
            "ZT7300000003",
            L.SIGNED,
            [
                _ev(now, 8, "快件已揽收", "广州市", "ACCEPT"),
                _ev(now, 5, "已签收，本人签收", "杭州市西湖区", "SIGN"),
            ],
            signed_detail=SignedDetail.SELF,
        ),
        trace(
            "o_1006",
            "ZT7300000005",
            L.IN_TRANSIT,
            [
                _ev(now, 4, "快件已揽收", "广州市", "ACCEPT"),
                _ev(now, 3.1, "已到达武汉转运中心", "武汉转运中心", "TRANSPORT"),  # 约 74 小时前
            ],
        ),
        trace(
            "o_1008",
            "ZT7300000007",
            L.EXCEPTION,
            [
                _ev(now, 3, "快件已揽收", "广州市", "ACCEPT"),
                _ev(now, 0.5, "疑难件：地址不详，联系不上收件人", "杭州市", "FAILED"),
            ],
        ),
        trace(
            "o_1009",
            "ZT7300000008",
            L.RETURNING,
            [
                _ev(now, 7, "快件已揽收", "广州市", "ACCEPT"),
                _ev(now, 2, "收件人拒收，快件退回中", "杭州市", "REJECT"),
            ],
        ),
        trace(
            "o_1010",
            "SF1400000009",
            L.SIGNED,
            [
                _ev(now, 2, "快件已揽收", "深圳市", "ACCEPT"),
                _ev(now, 0.06, "已签收，快递柜代收", "杭州市西湖区", "SIGN_LOCKER"),
            ],
            courier="顺丰速运",
            signed_detail=SignedDetail.LOCKER,
            pickup_info="丰巢柜（幸福里 3 号楼）",
        ),
    ]


def _aftersales(now: datetime) -> list[AfterSaleInfo]:
    T, R, S = AfterSaleType, AfterSaleReason, AfterSaleStatus

    def aftersale(
        aftersale_id: str,
        order_id: str,
        type_: AfterSaleType,
        reason: AfterSaleReason,
        status: AfterSaleStatus,
        amount: str,
        *,
        days: float,
        buyer_ref: str = BUYER,
        shop_id: str = SHOP_A,
        evidence_count: int = 0,
        note: str | None = None,
        updated_days: float | None = None,
    ) -> AfterSaleInfo:
        created = _t(days, now=now)
        updated = _t(updated_days, now=now) if updated_days is not None else created
        return AfterSaleInfo(
            shop_id=shop_id,
            aftersale_id=aftersale_id,
            order_id=order_id,
            buyer_ref=buyer_ref,
            type=type_,
            reason=reason,
            status=status,
            amount=Decimal(amount),
            created_at=created,
            updated_at=updated,
            evidence_count=evidence_count,
            status_version=str(int(updated.timestamp())),
            note=note,
        )

    return [
        # R2：未发货，仅退款 → 建议同意
        aftersale(
            "as_2001", "o_1001", T.REFUND_ONLY, R.OTHER, S.PENDING_REVIEW, "259.00", days=0.1
        ),
        # R3：签收 5 天，七天无理由退货退款 → 建议同意退货
        aftersale(
            "as_2002",
            "o_1004",
            T.RETURN_REFUND,
            R.NO_REASON,
            S.PENDING_REVIEW,
            "199.00",
            days=0.2,
            buyer_ref=BUYER_R3,
        ),
        # R4：显示签收，买家称没收到 → 签收核查
        aftersale(
            "as_2003",
            "o_1003",
            T.REFUND_ONLY,
            R.NOT_RECEIVED,
            S.PENDING_REVIEW,
            "259.00",
            days=0.1,
            note="显示签收了但我根本没收到",
        ),
        # R5：质量问题，缺凭证 → 等待凭证
        aftersale(
            "as_2004",
            "o_1005",
            T.RETURN_REFUND,
            R.QUALITY,
            S.PENDING_REVIEW,
            "259.00",
            days=0.3,
            buyer_ref=BUYER_R5,
        ),
        # R3 + R6：金额 1554 超过上限 500 → 必须店长确认
        aftersale(
            "as_2005",
            "o_1011",
            T.RETURN_REFUND,
            R.NO_REASON,
            S.PENDING_REVIEW,
            "1554.00",
            days=0.2,
            buyer_ref=BUYER_R6,
        ),
        # R1：已不是待审核（买家撤销）
        aftersale(
            "as_2006",
            "o_1012",
            T.REFUND_ONLY,
            R.OTHER,
            S.CLOSED,
            "199.00",
            days=5,
            buyer_ref=BUYER_REFUND,
        ),
        # R8：定制商品申请七天无理由 → 人工决定
        aftersale(
            "as_2007",
            "o_1007",
            T.RETURN_REFUND,
            R.NO_REASON,
            S.PENDING_REVIEW,
            "459.00",
            days=0.4,
            buyer_ref=BUYER_R8,
        ),
        # 退款中（退款进度问答）
        aftersale(
            "as_2008",
            "o_1012",
            T.RETURN_REFUND,
            R.NO_REASON,
            S.REFUNDING,
            "199.00",
            days=4,
            updated_days=0.3,  # 仓库确认收货后同意退款，进入退款中
            buyer_ref=BUYER_REFUND,
        ),
        # 质量问题，凭证齐全（R5 → 客服查看后决定）
        aftersale(
            "as_2009",
            "o_1013",
            T.RETURN_REFUND,
            R.DAMAGED,
            S.PENDING_REVIEW,
            "259.00",
            days=0.05,
            evidence_count=3,
            buyer_ref=BUYER_R5,
        ),
        # R7：同一买家 30 天内售后 ≥ 3 次
        aftersale(
            "as_3001",
            "o_2001",
            T.RETURN_REFUND,
            R.NO_REASON,
            S.REFUND_SUCCESS,
            "199.00",
            days=23,
            buyer_ref=BUYER_RISKY,
        ),
        aftersale(
            "as_3002",
            "o_2002",
            T.RETURN_REFUND,
            R.NO_REASON,
            S.REFUND_SUCCESS,
            "199.00",
            days=16,
            buyer_ref=BUYER_RISKY,
        ),
        aftersale(
            "as_3003",
            "o_2003",
            T.REFUND_ONLY,
            R.OTHER,
            S.CLOSED,
            "199.00",
            days=8,
            buyer_ref=BUYER_RISKY,
        ),
        aftersale(
            "as_3004",
            "o_2004",
            T.RETURN_REFUND,
            R.NO_REASON,
            S.PENDING_REVIEW,
            "199.00",
            days=0.1,
            buyer_ref=BUYER_RISKY,
        ),
        # 店铺 B（串店测试）
        aftersale(
            "as_9001",
            "o_9001",
            T.REFUND_ONLY,
            R.OTHER,
            S.PENDING_REVIEW,
            "129.00",
            days=0.1,
            shop_id=SHOP_B,
        ),
    ]


def _stocks(now: datetime) -> list[StockInfo]:
    def stock(item_id: str, sku_id: str, qty: int, note: str | None = None) -> StockInfo:
        return StockInfo(
            shop_id=SHOP_A,
            item_id=item_id,
            sku_id=sku_id,
            quantity=qty,
            in_stock=qty > 0,
            restock_note=note,
            fetched_at=now,
        )

    return [
        stock("i_555", "k_1", 12),
        stock("i_555", "k_2", 3),
        stock("i_555", "k_3", 0, "预计 10 月 8 日补货"),
        stock("i_555", "k_4", 20),
        stock("i_777", "k_71", 8),
        stock("i_777", "k_72", 0),
        stock("i_777", "k_73", 5),
        stock("i_888", "k_81", 99),
        # 店铺 C：平台不给具体数量（quantity=None）
        StockInfo(shop_id=SHOP_C, item_id="i_c1", quantity=None, in_stock=True, fetched_at=now),
    ]


def _prices(now: datetime) -> list[PriceInfo]:
    promo = ["店铺满 199 减 20"]
    # i_888 是定制款，不参加店铺满减（实时活动配置优先于店铺政策，§5.2.1）
    return [
        PriceInfo(
            shop_id=SHOP_A,
            item_id="i_555",
            price=Decimal("259.00"),
            promotions=promo,
            fetched_at=now,
        ),
        PriceInfo(
            shop_id=SHOP_A,
            item_id="i_777",
            price=Decimal("199.00"),
            promotions=promo,
            fetched_at=now,
        ),
        PriceInfo(shop_id=SHOP_A, item_id="i_888", price=Decimal("459.00"), fetched_at=now),
    ]


def build_demo_world(now: datetime = DEFAULT_NOW) -> DemoWorld:
    """每次调用都返回一份新的数据，测试之间互不影响。"""
    return DemoWorld(
        now=now,
        shops=_shops(),
        policies=_policies(now),
        items=_items(now),
        faqs=_faqs(now),
        orders=_orders(now),
        traces=_traces(now),
        aftersales=_aftersales(now),
        stocks=_stocks(now),
        prices=_prices(now),
    )


def seed_knowledge(repo, world: DemoWorld) -> dict[tuple[str, str], int]:
    """把演示世界的店铺、政策、商品、FAQ 写进任意 KnowledgeRepository 实现。

    返回 {(shop_id, 问题): faq_id}，方便测试按问题找 FAQ。
    """
    for shop in world.shops.values():
        repo.upsert_shop(shop)
    for entry in world.policies:
        repo.upsert_policy(entry)
    for card in world.items:
        repo.upsert_item_card(card)
    ids: dict[tuple[str, str], int] = {}
    for item in world.faqs:
        saved = repo.add_knowledge(item)
        assert saved.id is not None
        ids[(saved.shop_id, saved.question)] = saved.id
    return ids
