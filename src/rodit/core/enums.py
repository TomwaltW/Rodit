"""全局枚举（方案 §5.1 问题分类、§7 工具、§8.3 标准会话、§10 数据模型）。

所有模块共用这一份词表；新增取值要同步更新方案或在 docs/架构与分工.md 里记一笔。
"""

from __future__ import annotations

from enum import StrEnum


class Platform(StrEnum):
    TAOBAO = "taobao"
    DOUYIN = "douyin"
    PDD = "pdd"


class Mode(StrEnum):
    """输出模式（§5.6、§13.4）。"""

    ASSIST = "assist"  # 智能辅助：AI 起草、客服发送（MVP）
    AUTO = "auto"  # 全自动：高置信度、低风险直接发送（后期）
    SHADOW = "shadow"  # 影子模式：生成但不展示，事后对比


class Role(StrEnum):
    BUYER = "buyer"
    AGENT = "agent"  # 人工客服
    BOT = "bot"  # 系统自动回复


class Stage(StrEnum):
    PRE_SALE = "pre_sale"
    IN_SALE = "in_sale"
    AFTER_SALE = "after_sale"
    OTHER = "other"


class Intent(StrEnum):
    """问题类型（§5.1），顺序与方案表格一致。"""

    PRODUCT_SPEC = "product_spec"
    SIZE_ADVICE = "size_advice"
    USAGE = "usage"
    STOCK = "stock"
    PRICE_PROMO = "price_promo"
    SHIPPING_POLICY = "shipping_policy"
    INVOICE_GIFT = "invoice_gift"
    COMPARE_AUTHENTIC = "compare_authentic"
    LOGISTICS_TRACK = "logistics_track"
    URGE_SHIP = "urge_ship"
    MODIFY_ORDER = "modify_order"
    RETURN_POLICY = "return_policy"
    REFUND_PROGRESS = "refund_progress"
    QUALITY_ISSUE = "quality_issue"
    NOT_RECEIVED = "not_received"
    MALFUNCTION = "malfunction"
    COMPLAINT = "complaint"
    GREETING = "greeting"
    OFF_PLATFORM = "off_platform"
    OTHER = "other"


INTENT_STAGE: dict[Intent, Stage] = {
    Intent.PRODUCT_SPEC: Stage.PRE_SALE,
    Intent.SIZE_ADVICE: Stage.PRE_SALE,
    Intent.USAGE: Stage.PRE_SALE,
    Intent.STOCK: Stage.PRE_SALE,
    Intent.PRICE_PROMO: Stage.PRE_SALE,
    Intent.SHIPPING_POLICY: Stage.PRE_SALE,
    Intent.INVOICE_GIFT: Stage.PRE_SALE,
    Intent.COMPARE_AUTHENTIC: Stage.PRE_SALE,
    Intent.LOGISTICS_TRACK: Stage.IN_SALE,
    Intent.URGE_SHIP: Stage.IN_SALE,
    Intent.MODIFY_ORDER: Stage.IN_SALE,
    Intent.RETURN_POLICY: Stage.AFTER_SALE,
    Intent.REFUND_PROGRESS: Stage.AFTER_SALE,
    Intent.QUALITY_ISSUE: Stage.AFTER_SALE,
    Intent.NOT_RECEIVED: Stage.AFTER_SALE,
    Intent.MALFUNCTION: Stage.AFTER_SALE,
    Intent.COMPLAINT: Stage.AFTER_SALE,
    Intent.GREETING: Stage.OTHER,
    Intent.OFF_PLATFORM: Stage.OTHER,
    Intent.OTHER: Stage.OTHER,
}

# 中文名：用于推荐卡片、澄清问题、报表
INTENT_LABELS: dict[Intent, str] = {
    Intent.PRODUCT_SPEC: "商品参数",
    Intent.SIZE_ADVICE: "尺码 / 选型",
    Intent.USAGE: "使用方法",
    Intent.STOCK: "库存 / 补货",
    Intent.PRICE_PROMO: "价格 / 优惠",
    Intent.SHIPPING_POLICY: "发货与快递",
    Intent.INVOICE_GIFT: "发票 / 赠品 / 保障",
    Intent.COMPARE_AUTHENTIC: "对比 / 真伪",
    Intent.LOGISTICS_TRACK: "物流进度",
    Intent.URGE_SHIP: "催发货",
    Intent.MODIFY_ORDER: "改地址 / 改规格 / 取消",
    Intent.RETURN_POLICY: "退换政策",
    Intent.REFUND_PROGRESS: "退款进度",
    Intent.QUALITY_ISSUE: "质量 / 破损 / 错漏发",
    Intent.NOT_RECEIVED: "签收未收到",
    Intent.MALFUNCTION: "使用故障",
    Intent.COMPLAINT: "投诉 / 纠纷",
    Intent.GREETING: "问候 / 催回复",
    Intent.OFF_PLATFORM: "站外交易请求",
    Intent.OTHER: "其他",
}

# 例句（§5.1），用于路由问题的选项定义和合成评测集
INTENT_EXAMPLES: dict[Intent, tuple[str, ...]] = {
    Intent.PRODUCT_SPEC: ("这个是纯棉的吗？", "电池多大？"),
    Intent.SIZE_ADVICE: ("165 cm、110 斤穿什么码？",),
    Intent.USAGE: ("怎么安装？", "能机洗吗？"),
    Intent.STOCK: ("还有货吗？", "什么时候补？"),
    Intent.PRICE_PROMO: ("能便宜点吗？", "有券吗？"),
    Intent.SHIPPING_POLICY: ("什么时候发？", "发什么快递？", "新疆能到吗？"),
    Intent.INVOICE_GIFT: ("能开发票吗？", "有运费险吗？"),
    Intent.COMPARE_AUTHENTIC: ("是正品吗？", "和 A 款有什么区别？"),
    Intent.LOGISTICS_TRACK: ("我的快递到哪了？",),
    Intent.URGE_SHIP: ("怎么还没发货？",),
    Intent.MODIFY_ORDER: ("能改个地址吗？",),
    Intent.RETURN_POLICY: ("能退吗？", "运费谁出？"),
    Intent.REFUND_PROGRESS: ("退款什么时候到？",),
    Intent.QUALITY_ISSUE: ("收到的是坏的",),
    Intent.NOT_RECEIVED: ("显示签收了但我没收到",),
    Intent.MALFUNCTION: ("开不了机",),
    Intent.COMPLAINT: ("我要投诉",),
    Intent.GREETING: ("在吗？",),
    Intent.OFF_PLATFORM: ("加个微信吧，转账便宜点",),
    Intent.OTHER: (),
}


class ShopStatus(StrEnum):
    TRIAL = "trial"
    ACTIVE = "active"
    SUSPENDED = "suspended"


class PolicyKey(StrEnum):
    """店铺政策键（附录 C 问卷 1–13；14 禁用词、15 语气进 ShopConfig）。"""

    SHIP_TIME = "ship_time"  # 1 付款后多久发货
    PRESALE = "presale"  # 1 预售商品怎么说明
    COURIER = "courier"  # 2 默认快递
    COURIER_CHOICE = "courier_choice"  # 2 能否指定快递、是否加钱
    REMOTE_AREAS = "remote_areas"  # 3 不发货或需补运费的地区
    SEVEN_DAY_RETURN = "seven_day_return"  # 4 是否支持七天无理由
    NON_RETURNABLE = "non_returnable"  # 4 哪些商品不支持
    RETURN_SHIPPING = "return_shipping"  # 5 退货运费谁出
    SHIPPING_INSURANCE = "shipping_insurance"  # 5 运费险
    QUALITY_ISSUE = "quality_issue"  # 6 质量问题处理与凭证
    INVOICE = "invoice"  # 7 发票
    GIFT = "gift"  # 8 赠品
    MODIFY_ORDER = "modify_order"  # 9 改地址 / 改规格
    PROMOTIONS = "promotions"  # 10 目前的优惠
    PROMISE_BOUNDARY = "promise_boundary"  # 10 客服能承诺 / 不能承诺什么
    WARRANTY = "warranty"  # 11 保修
    SIZE_TENDENCY = "size_tendency"  # 12 尺码偏大偏小（服饰）
    SERVICE_HOURS = "service_hours"  # 13 客服工作时间


POLICY_LABELS: dict[PolicyKey, str] = {
    PolicyKey.SHIP_TIME: "发货时效",
    PolicyKey.PRESALE: "预售说明",
    PolicyKey.COURIER: "默认快递",
    PolicyKey.COURIER_CHOICE: "指定快递",
    PolicyKey.REMOTE_AREAS: "偏远地区",
    PolicyKey.SEVEN_DAY_RETURN: "七天无理由",
    PolicyKey.NON_RETURNABLE: "不支持退换的商品",
    PolicyKey.RETURN_SHIPPING: "退货运费",
    PolicyKey.SHIPPING_INSURANCE: "运费险",
    PolicyKey.QUALITY_ISSUE: "质量问题处理",
    PolicyKey.INVOICE: "发票",
    PolicyKey.GIFT: "赠品",
    PolicyKey.MODIFY_ORDER: "改地址 / 改规格",
    PolicyKey.PROMOTIONS: "优惠活动",
    PolicyKey.PROMISE_BOUNDARY: "承诺边界",
    PolicyKey.WARRANTY: "保修",
    PolicyKey.SIZE_TENDENCY: "尺码偏向",
    PolicyKey.SERVICE_HOURS: "客服时间",
}


class KnowledgeSource(StrEnum):
    MERCHANT = "merchant"  # 商家录入
    ADOPTED = "adopted"  # 客服采纳、商家审核通过的优质回答
    CATEGORY_TEMPLATE = "category_template"  # 类目通用知识（我们维护）


class KnowledgeStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    RETIRED = "retired"


class EvidenceSource(StrEnum):
    """证据来源；priority 越小优先级越高，冲突时高优先级覆盖低优先级（§5.2.1）。"""

    REALTIME = "realtime"  # 1 实时接口：价格、库存、订单、物流、售后
    POLICY = "policy"  # 2 店铺政策
    ITEM = "item"  # 3 商品知识
    FAQ = "faq"  # 4 FAQ 与优质回答
    CATEGORY = "category"  # 5 类目通用知识

    @property
    def priority(self) -> int:
        return _EVIDENCE_PRIORITY[self]


_EVIDENCE_PRIORITY = {
    EvidenceSource.REALTIME: 1,
    EvidenceSource.POLICY: 2,
    EvidenceSource.ITEM: 3,
    EvidenceSource.FAQ: 4,
    EvidenceSource.CATEGORY: 5,
}


class OrderStatus(StrEnum):
    UNPAID = "unpaid"
    PAID = "paid"  # 已付款未发货
    SHIPPED = "shipped"
    SIGNED = "signed"  # 物流已签收
    COMPLETED = "completed"  # 确认收货 / 交易成功
    CLOSED = "closed"  # 关闭或取消


class LogisticsStatus(StrEnum):
    """物流内部状态（§7.2），各平台状态在工具层映射到这里。"""

    PICKED_UP = "picked_up"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    SIGNED = "signed"
    EXCEPTION = "exception"  # 疑难件
    RETURNING = "returning"  # 退回、拒签


class SignedDetail(StrEnum):
    SELF = "self"  # 本人签收
    STATION = "station"  # 驿站
    LOCKER = "locker"  # 快递柜
    OTHER = "other"  # 他人代收等


class AnomalyKind(StrEnum):
    """物流异常（§7.2 异常识别）。"""

    STALLED = "stalled"  # 超过阈值没有新轨迹
    EXCEPTION = "exception"  # 疑难件
    STATION_OR_LOCKER = "station_or_locker"  # 驿站 / 快递柜签收
    RETURNING = "returning"  # 退回、拒签


class AfterSaleType(StrEnum):
    REFUND_ONLY = "refund_only"  # 仅退款
    RETURN_REFUND = "return_refund"  # 退货退款
    EXCHANGE = "exchange"  # 换货


class AfterSaleReason(StrEnum):
    NO_REASON = "no_reason"  # 七天无理由
    QUALITY = "quality"
    DAMAGED = "damaged"
    WRONG_ITEM = "wrong_item"  # 错发
    MISSING_ITEM = "missing_item"  # 漏发
    NOT_RECEIVED = "not_received"  # 签收未收到
    OTHER = "other"


class AfterSaleStatus(StrEnum):
    """售后单平台状态（归一后）。"""

    PENDING_REVIEW = "pending_review"  # 待商家处理
    WAIT_BUYER_RETURN = "wait_buyer_return"  # 已同意退货，等买家寄回
    WAIT_SELLER_CONFIRM = "wait_seller_confirm"  # 买家已寄回，等商家确认收货
    REFUNDING = "refunding"  # 退款中
    REFUND_SUCCESS = "refund_success"
    REFUSED = "refused"
    CLOSED = "closed"  # 买家撤销或平台关闭


class ReviewState(StrEnum):
    """售后审核状态机（§7.3.2）。"""

    NEW = "new"  # 待审核
    SUGGEST = "suggest"  # 规则建议
    VERIFY = "verify"  # 签收核查中
    EVIDENCE = "evidence"  # 等待凭证
    CONFIRM = "confirm"  # 待客服确认
    NEGOTIATE = "negotiate"  # 协商中
    AGREED = "agreed"
    REFUSED = "refused"
    PLATFORM = "platform"  # 平台已处理（自动处理或买家撤销）


class ReviewAdvice(StrEnum):
    """审核建议（§10 aftersale_review.suggestion）。"""

    AGREE = "agree"  # 同意（仅退款：同意退款）
    AGREE_RETURN = "agree_return"  # 同意退货，仓库确认收货后再同意退款（R3）
    REFUSE = "refuse"
    NEGOTIATE = "negotiate"
    MANUAL = "manual"  # 人工决定
    NONE = "none"  # 不操作，只同步状态（R1）


class AfterSaleAction(StrEnum):
    """售后写操作（只在审核助手里，由客服确认后执行）。"""

    AGREE_REFUND = "agree_refund"
    AGREE_RETURN = "agree_return"
    REFUSE = "refuse"


class RouteAction(StrEnum):
    """路由结果（§6.6 按顺序判断，命中就停）。"""

    OFF_PLATFORM_SCRIPT = "off_platform_script"  # 规则 1：平台规则标准话术
    HUMAN_PRIORITY = "human_priority"  # 规则 2 / 投诉：提醒客服优先处理，不给推荐
    SOOTHE = "soothe"  # 规则 3：安抚话术并标红
    HIGH_RISK_ANSWER = "high_risk_answer"  # 规则 4：推荐 + 风险提示
    SPLIT_MULTI = "split_multi"  # 规则 5：按问题分条推荐
    ANSWER = "answer"  # 规则 6：检索、生成、质检，出推荐
    CLARIFY = "clarify"  # 规则 7：推荐一句澄清问题
    NO_SUGGESTION = "no_suggestion"  # 规则 8：不推荐，客服自行处理
    TRANSFER_HUMAN = "transfer_human"  # 全自动模式下转人工


class DraftKind(StrEnum):
    FAQ = "faq"  # 商家审核过的标准回答
    TEMPLATE = "template"  # 模板（问候、站外话术、物流等）
    LLM = "llm"  # 大模型生成


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"


class SuggestionStatus(StrEnum):
    SHOWN = "shown"
    SENT = "sent"  # 原样发送
    EDITED_SENT = "edited_sent"
    REJECTED = "rejected"
    EXPIRED = "expired"  # 买家新消息到达后被刷新掉
    SHADOW = "shadow"  # 影子模式生成、未展示


class RejectReason(StrEnum):
    """客服点"不合适"时选择的原因（§5.8）。"""

    MISSING_KNOWLEDGE = "missing_knowledge"  # 知识缺失
    WRONG_INTENT = "wrong_intent"  # 理解错误
    BAD_WORDING = "bad_wording"  # 表达不好
    WRONG_FACT = "wrong_fact"  # 事实错误
    OTHER = "other"


class DecisionStage(StrEnum):
    ROUTE = "route"
    QA = "qa"


class ProviderName(StrEnum):
    INCLOUD_LLM = "incloud_llm"
    CLASSIFIER = "classifier"
    RULES = "rules"
    JEV = "jev"
