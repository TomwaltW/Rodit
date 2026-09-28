"""证据 ID 规则、问题类型所需依据、事实主题（方案 §5.1 "答案依据" 列、§5.2、§7）。

证据 ID 是跨模块契约：检索 / 工具产出证据，生成引用 evidence_ids，质检逐条核对，
推荐记录落库。只能用这里的构造函数生成，不要手拼字符串。

各类证据的 data 结构（Evidence.data）：
- platform:{platform}:{key}           {"platform", "key", "text"}
- policy:{key}                        {"key", "value"}
- item:{item_id}:title                {"title"}
- item:{item_id}:attr:{name}          {"name", "value"}
- item:{item_id}:sku                  {"skus": [Sku...]}
- item:{item_id}:size_chart           {"rows": [...], "conservative": bool}
- item:{item_id}:size_advice          {"text"}
- item:{item_id}:care                 {"care": [...]}
- item:{item_id}:usage                {"usage": [...]}
- item:{item_id}:detail[:{seg}]       {"text", "seg"}   详情页文字，按句切段，seg 从 0 开始
- faq:{id}                            {"question", "answer", "faq_id"}
- category:{key}                      {"text"}
- tool:order:{order_id}               {"order": OrderInfo}
- tool:logistics:{order_id}           {"trace": LogisticsTrace, "analysis": LogisticsAnalysis}
- tool:aftersale:{aftersale_id}       {"aftersale": AfterSaleInfo, "progress": RefundProgress}
- tool:return:{order_id}              {"eligibility": ReturnEligibility}
- tool:stock:{item_id}[:{sku_id}]     {"stock": StockInfo}
- tool:price:{item_id}                {"price": PriceInfo}
（模型对象都以 model_dump(mode="json") 存入，读取时用对应模型 model_validate。）

冲突处理（§5.2.1）：每条证据可以带一个 topic（Topic 的取值）。同一 topic 下只保留优先级最高
（EvidenceSource.priority 最小）的来源，其余从 EvidenceBundle.evidences 移除、
id 记进 overridden_ids。
自由文本（详情页、FAQ 回答）先按句切段，用 detect_topics 给每段打主题，只移除冲突的那一段。
"""

from __future__ import annotations

import re
from enum import StrEnum

from rodit.core.enums import POLICY_LABELS, Intent, Platform, PolicyKey


class Need(StrEnum):
    """回答某类问题需要的依据种类。"""

    ITEM_ATTRS = "item_attrs"  # 商品标题、属性、SKU
    SIZE = "size"  # 尺码表 + 尺码建议
    CARE_USAGE = "care_usage"  # 洗涤保养、使用安装
    ITEM_DETAIL = "item_detail"  # 详情页文字
    POLICY = "policy"  # 店铺政策（具体键见 INTENT_POLICY_KEYS）
    PLATFORM_RULE = "platform_rule"  # 平台规则（具体键见 INTENT_PLATFORM_RULES）
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
        Need.PLATFORM_RULE,
        Need.FAQ,
        Need.CATEGORY,
    }
)
"""知识中心负责的依据（EvidenceRetriever 自己处理）。"""

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
"""实时工具负责的依据（交给 ToolEvidenceProvider）。"""

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
    Intent.MODIFY_ORDER: frozenset({_N.ORDER, _N.POLICY, _N.PLATFORM_RULE}),
    Intent.RETURN_POLICY: frozenset(
        {_N.POLICY, _N.PLATFORM_RULE, _N.ORDER, _N.RETURN_ELIGIBILITY, _N.FAQ}
    ),
    Intent.REFUND_PROGRESS: frozenset({_N.AFTERSALE, _N.ORDER, _N.PLATFORM_RULE}),
    Intent.QUALITY_ISSUE: frozenset({_N.AFTERSALE, _N.ORDER, _N.POLICY, _N.PLATFORM_RULE}),
    Intent.NOT_RECEIVED: frozenset({_N.LOGISTICS, _N.AFTERSALE, _N.ORDER, _N.PLATFORM_RULE}),
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


class PlatformRuleKey(StrEnum):
    """平台规则（按平台维护，知识中心负责内容；§5.1、§7.3.1、§2.1）。"""

    APPLY_AFTERSALE = "apply_aftersale"  # 怎么申请售后：只能买家在订单页发起
    REFUND_TIMING = "refund_timing"  # 退款到账的一般规则（以平台为准）
    MODIFY_ORDER = "modify_order"  # 改地址 / 改规格 / 取消的平台规则
    OFF_PLATFORM = "off_platform"  # 禁止站外联系和交易


INTENT_PLATFORM_RULES: dict[Intent, tuple[PlatformRuleKey, ...]] = {
    Intent.MODIFY_ORDER: (PlatformRuleKey.MODIFY_ORDER,),
    Intent.RETURN_POLICY: (PlatformRuleKey.APPLY_AFTERSALE,),
    Intent.REFUND_PROGRESS: (PlatformRuleKey.REFUND_TIMING,),
    Intent.QUALITY_ISSUE: (PlatformRuleKey.APPLY_AFTERSALE,),
    Intent.NOT_RECEIVED: (PlatformRuleKey.APPLY_AFTERSALE,),
}

LLM_FREE_INTENTS = frozenset({Intent.GREETING, Intent.OFF_PLATFORM})
"""用模板回答、不调用大模型的问题类型（§5.4 "能不用大模型就不用"、§11.4）。
检索仍按 INTENT_NEEDS 进行（问候要引用客服时间政策）。"""


def needs_for(intent: Intent) -> frozenset[Need]:
    return INTENT_NEEDS.get(intent, frozenset())


def policy_keys_for(intent: Intent) -> tuple[PolicyKey, ...]:
    return INTENT_POLICY_KEYS.get(intent, ())


def platform_rules_for(intent: Intent) -> tuple[PlatformRuleKey, ...]:
    return INTENT_PLATFORM_RULES.get(intent, ())


# ---------------------------------------------------------------- 事实主题（冲突处理）


class Topic(StrEnum):
    """证据的事实主题。店铺政策的 topic 就是 PolicyKey 的值；这里补充政策之外的主题。

    实时数据的主题带上对象 ID（如 "stock:i_555:k_1"），用 topic_for_stock 等函数生成，
    避免不同商品互相覆盖。
    """

    PRICE = "price"
    STOCK = "stock"


def topic_for_policy(key: str) -> str:
    return key


def topic_for_stock(item_id: str, sku_id: str | None = None) -> str:
    return f"{Topic.STOCK}:{item_id}" + (f":{sku_id}" if sku_id else "")


def topic_for_price(item_id: str) -> str:
    return f"{Topic.PRICE}:{item_id}"


# 自由文本按句打主题（详情页、FAQ 回答）。只覆盖容易和店铺政策冲突的主题。
_TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        PolicyKey.SHIP_TIME,
        re.compile(r"(?:\d+\s*(?:小时|天|日)\s*(?:内)?\s*发货|发货时间|付款后.{0,6}发)"),
    ),
    (PolicyKey.PRESALE, re.compile(r"预售")),
    (
        PolicyKey.COURIER,
        re.compile(r"(?:默认|发).{0,4}(?:快递|顺丰|中通|圆通|申通|韵达|邮政|京东)"),
    ),
    (PolicyKey.REMOTE_AREAS, re.compile(r"(?:偏远|新疆|西藏|港澳台|海外).{0,8}(?:发|运费|不发)")),
    (PolicyKey.SEVEN_DAY_RETURN, re.compile(r"七天无理由|7\s*天无理由")),
    (PolicyKey.RETURN_SHIPPING, re.compile(r"退货.{0,4}运费|运费.{0,6}(?:承担|谁出)")),
    (PolicyKey.SHIPPING_INSURANCE, re.compile(r"运费险")),
    (PolicyKey.INVOICE, re.compile(r"发票")),
    (PolicyKey.GIFT, re.compile(r"赠品|送.{0,4}(?:小礼物|赠)")),
    (PolicyKey.WARRANTY, re.compile(r"保修|质保")),
    (PolicyKey.PROMOTIONS, re.compile(r"满\s*\d+\s*减|优惠券|折扣|活动价")),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def split_sentences(text: str) -> list[str]:
    """按中文句末标点切句，去掉空白句。"""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


def detect_topics(text: str) -> set[str]:
    """给一段自由文本打事实主题（PolicyKey 取值）。检索做冲突处理时用。"""
    return {str(topic) for topic, pattern in _TOPIC_PATTERNS if pattern.search(text or "")}


# ---------------------------------------------------------------- 证据 ID 构造


def platform_rule(platform: Platform | str, key: str) -> str:
    return f"platform:{platform}:{key}"


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


def item_detail(item_id: str, seg: int | None = None) -> str:
    """详情页文字；seg 是按句切段后的段号（从 0 开始），None 表示整段。"""
    return f"item:{item_id}:detail" + (f":{seg}" if seg is not None else "")


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


def missing_platform_rule(platform: Platform | str, key: str) -> str:
    return f"platform:{platform}:{key}"


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


PLATFORM_NAMES = {Platform.TAOBAO: "淘宝", Platform.DOUYIN: "抖店", Platform.PDD: "拼多多"}

PLATFORM_RULE_LABELS = {
    PlatformRuleKey.APPLY_AFTERSALE: "申请售后",
    PlatformRuleKey.REFUND_TIMING: "退款到账",
    PlatformRuleKey.MODIFY_ORDER: "改单规则",
    PlatformRuleKey.OFF_PLATFORM: "站外交易",
}


def platform_rule_label(platform: Platform | str, key: str) -> str:
    name = PLATFORM_NAMES.get(Platform(platform), str(platform))
    try:
        rule = PLATFORM_RULE_LABELS[PlatformRuleKey(key)]
    except ValueError:
        rule = key
    return f"平台规则 · {name}{rule}"


def item_label(field: str) -> str:
    """field 如 "面料"、"尺码表"、"尺码建议"。"""
    return f"商品属性 · {field}"


def faq_label(question: str) -> str:
    return f"FAQ · {question}"


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
