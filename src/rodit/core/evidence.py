"""证据 ID 规则、问题类型所需依据（方案 §5.1 "答案依据" 列、§5.2、§7）。

证据 ID 是跨模块契约：检索 / 工具产出证据，生成引用 evidence_ids，质检逐条核对，
推荐记录落库。只能用这里的构造函数生成，不要手拼字符串。

各类证据的 data 结构（Evidence.data）：
- policy:{key}                      {"key", "value"}
- item:{item_id}:title              {"title"}
- item:{item_id}:attr:{name}        {"name", "value"}
- item:{item_id}:sku                {"skus": [Sku...]}
- item:{item_id}:size_chart         {"rows": [...]}
- item:{item_id}:size_advice        {"text"}
- item:{item_id}:care               {"care": [...]}
- item:{item_id}:usage              {"usage": [...]}
- item:{item_id}:detail             {"text"}
- faq:{id}                          {"question", "answer", "faq_id"}
- category:{key}                    {"text"}
- tool:order:{order_id}             {"order": OrderInfo}
- tool:logistics:{order_id}         {"trace": LogisticsTrace, "analysis": LogisticsAnalysis}
- tool:aftersale:{aftersale_id}     {"aftersale": AfterSaleInfo, "progress": RefundProgress}
- tool:return:{order_id}            {"eligibility": ReturnEligibility}
- tool:stock:{item_id}[:{sku_id}]   {"stock": StockInfo}
- tool:price:{item_id}              {"price": PriceInfo}
（模型对象都以 model_dump(mode="json") 存入，读取时用对应模型 model_validate。）
"""

from __future__ import annotations

from enum import StrEnum

from rodit.core.enums import POLICY_LABELS, Intent, PolicyKey


class Need(StrEnum):
    """回答某类问题需要的依据种类。"""

    ITEM_ATTRS = "item_attrs"  # 商品标题、属性、SKU
    SIZE = "size"  # 尺码表 + 尺码建议
    CARE_USAGE = "care_usage"  # 洗涤保养、使用安装
    ITEM_DETAIL = "item_detail"  # 详情页文字
    POLICY = "policy"  # 店铺政策（具体键见 INTENT_POLICY_KEYS）
    FAQ = "faq"  # FAQ 与优质回答（语义检索）
    CATEGORY = "category"  # 类目通用知识
    ORDER = "order"  # 订单工具
    LOGISTICS = "logistics"  # 物流工具
    AFTERSALE = "aftersale"  # 售后工具
    RETURN_ELIGIBILITY = "return_eligibility"  # 能不能退（订单状态 + 政策）
    STOCK = "stock"  # 实时库存
    PRICE = "price"  # 价格与活动配置


KNOWLEDGE_NEEDS = frozenset(
    {
        Need.ITEM_ATTRS,
        Need.SIZE,
        Need.CARE_USAGE,
        Need.ITEM_DETAIL,
        Need.POLICY,
        Need.FAQ,
        Need.CATEGORY,
    }
)
TOOL_NEEDS = frozenset(
    {
        Need.ORDER,
        Need.LOGISTICS,
        Need.AFTERSALE,
        Need.RETURN_ELIGIBILITY,
        Need.STOCK,
        Need.PRICE,
    }
)

_N = Need
INTENT_NEEDS: dict[Intent, frozenset[Need]] = {
    Intent.PRODUCT_SPEC: frozenset({_N.ITEM_ATTRS, _N.ITEM_DETAIL, _N.FAQ}),
    Intent.SIZE_ADVICE: frozenset({_N.SIZE, _N.ITEM_ATTRS, _N.POLICY, _N.FAQ}),
    Intent.USAGE: frozenset({_N.CARE_USAGE, _N.ITEM_ATTRS, _N.ITEM_DETAIL, _N.FAQ, _N.CATEGORY}),
    Intent.STOCK: frozenset({_N.STOCK, _N.ITEM_ATTRS, _N.POLICY}),
    Intent.PRICE_PROMO: frozenset({_N.PRICE, _N.POLICY}),
    Intent.SHIPPING_POLICY: frozenset({_N.POLICY, _N.FAQ}),
    Intent.INVOICE_GIFT: frozenset({_N.POLICY, _N.FAQ}),
    Intent.COMPARE_AUTHENTIC: frozenset({_N.ITEM_ATTRS, _N.ITEM_DETAIL, _N.FAQ}),
    Intent.LOGISTICS_TRACK: frozenset({_N.ORDER, _N.LOGISTICS}),
    Intent.URGE_SHIP: frozenset({_N.ORDER, _N.LOGISTICS, _N.POLICY}),
    Intent.MODIFY_ORDER: frozenset({_N.ORDER, _N.POLICY}),
    Intent.RETURN_POLICY: frozenset({_N.POLICY, _N.ORDER, _N.RETURN_ELIGIBILITY, _N.FAQ}),
    Intent.REFUND_PROGRESS: frozenset({_N.AFTERSALE, _N.ORDER}),
    Intent.QUALITY_ISSUE: frozenset({_N.AFTERSALE, _N.ORDER, _N.POLICY}),
    Intent.NOT_RECEIVED: frozenset({_N.LOGISTICS, _N.AFTERSALE, _N.ORDER}),
    Intent.MALFUNCTION: frozenset({_N.CARE_USAGE, _N.ITEM_ATTRS, _N.FAQ, _N.POLICY}),
    Intent.COMPLAINT: frozenset(),
    Intent.GREETING: frozenset({_N.POLICY}),
    Intent.OFF_PLATFORM: frozenset(),
    Intent.OTHER: frozenset({_N.FAQ}),
}

_P = PolicyKey
INTENT_POLICY_KEYS: dict[Intent, tuple[PolicyKey, ...]] = {
    Intent.SIZE_ADVICE: (_P.SIZE_TENDENCY,),
    Intent.STOCK: (_P.PRESALE, _P.SHIP_TIME),
    Intent.PRICE_PROMO: (_P.PROMOTIONS, _P.PROMISE_BOUNDARY),
    Intent.SHIPPING_POLICY: (
        _P.SHIP_TIME,
        _P.COURIER,
        _P.COURIER_CHOICE,
        _P.REMOTE_AREAS,
        _P.PRESALE,
    ),
    Intent.INVOICE_GIFT: (_P.INVOICE, _P.GIFT, _P.SHIPPING_INSURANCE, _P.WARRANTY),
    Intent.URGE_SHIP: (_P.SHIP_TIME, _P.PRESALE),
    Intent.MODIFY_ORDER: (_P.MODIFY_ORDER,),
    Intent.RETURN_POLICY: (
        _P.SEVEN_DAY_RETURN,
        _P.NON_RETURNABLE,
        _P.RETURN_SHIPPING,
        _P.SHIPPING_INSURANCE,
    ),
    Intent.QUALITY_ISSUE: (_P.QUALITY_ISSUE, _P.RETURN_SHIPPING),
    Intent.MALFUNCTION: (_P.WARRANTY, _P.QUALITY_ISSUE),
    Intent.GREETING: (_P.SERVICE_HOURS,),
}

# 不需要检索依据、直接用模板回答的问题类型（§5.4 "能不用大模型就不用"）
TEMPLATE_INTENTS = frozenset({Intent.GREETING, Intent.OFF_PLATFORM})


def needs_for(intent: Intent) -> frozenset[Need]:
    return INTENT_NEEDS.get(intent, frozenset())


def policy_keys_for(intent: Intent) -> tuple[PolicyKey, ...]:
    return INTENT_POLICY_KEYS.get(intent, ())


# ---------------------------------------------------------------- 证据 ID 构造


def policy(key: str) -> str:
    return f"policy:{key}"


def item_title(item_id: str) -> str:
    return f"item:{item_id}:title"


def item_attr(item_id: str, name: str) -> str:
    return f"item:{item_id}:attr:{name}"


def item_sku(item_id: str) -> str:
    return f"item:{item_id}:sku"


def item_size_chart(item_id: str) -> str:
    return f"item:{item_id}:size_chart"


def item_size_advice(item_id: str) -> str:
    return f"item:{item_id}:size_advice"


def item_care(item_id: str) -> str:
    return f"item:{item_id}:care"


def item_usage(item_id: str) -> str:
    return f"item:{item_id}:usage"


def item_detail(item_id: str) -> str:
    return f"item:{item_id}:detail"


def faq(faq_id: int) -> str:
    return f"faq:{faq_id}"


def category(key: str) -> str:
    return f"category:{key}"


def tool_order(order_id: str) -> str:
    return f"tool:order:{order_id}"


def tool_logistics(order_id: str) -> str:
    return f"tool:logistics:{order_id}"


def tool_aftersale(aftersale_id: str) -> str:
    return f"tool:aftersale:{aftersale_id}"


def tool_return(order_id: str) -> str:
    return f"tool:return:{order_id}"


def tool_stock(item_id: str, sku_id: str | None = None) -> str:
    return f"tool:stock:{item_id}" + (f":{sku_id}" if sku_id else "")


def tool_price(item_id: str) -> str:
    return f"tool:price:{item_id}"


# 工具不可用 / 知识缺失的标记（EvidenceBundle.unavailable / missing）
def unavailable(kind: str) -> str:
    """kind 取 order / logistics / aftersale / stock / price。"""
    return f"tool:{kind}"


def missing_policy(key: str) -> str:
    return f"policy:{key}"


def missing_item_field(item_id: str | None, field: str) -> str:
    """field 如 size_advice / size_chart / attr:面料；item_id 为空表示缺当前商品。"""
    return f"item:{item_id or '?'}:{field}"


MISSING_ITEM = "item:?"  # 需要商品知识但会话里没有当前商品
MISSING_ORDER = "order:?"  # 需要订单但会话里没有订单卡片、也查不到最近订单


# ---------------------------------------------------------------- 展示标签


def policy_label(key: str) -> str:
    try:
        return f"店铺政策 · {POLICY_LABELS[PolicyKey(key)]}"
    except ValueError:
        return f"店铺政策 · {key}"


def item_label(field: str) -> str:
    """field 如 "面料"、"尺码表"、"尺码建议"。"""
    return f"商品属性 · {field}"


TOOL_LABELS = {
    "order": "订单",
    "logistics": "物流轨迹",
    "aftersale": "售后单",
    "return": "退换资格",
    "stock": "实时库存",
    "price": "价格与活动",
}


def tool_label(kind: str) -> str:
    return f"实时数据 · {TOOL_LABELS.get(kind, kind)}"
