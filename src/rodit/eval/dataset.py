"""合成评测集（方案 §13.1）。

每条标注：正确的问题类型、参考答案要点、必须引用的依据、不能出现的内容。全部基于合成演示世界
（rodit.testing.fixtures），不含真实商家、买家数据。按 id 的哈希留出 20% 作为测试集，不参与调参。

标注约定：
- must_cite：每个元素是"任选其一"的依据 ID 组，推荐回答的依据里至少要有组内一个；
- must_include：回答里必须出现的要点（子串）；
- must_not_include：回答里不能出现的内容（错误事实、违规话术）；
- expected_actions：允许的路由结果（空表示按问题类型正常回答，即 answer / high_risk_answer）；
- no_suggestion_ok：不给推荐也算对（如投诉、要求人工）。
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rodit.core.enums import Intent, Role, RouteAction


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    expected_intent: Intent
    acceptable_intents: list[Intent] = Field(default_factory=list)
    shop: str = "SHOP_A"  # fixtures 里的常量名
    buyer: str = "BUYER"  # fixtures 里的常量名
    item_id: str | None = None
    sku_id: str | None = None
    order_id: str | None = None
    history: list[tuple[Role, str]] = Field(default_factory=list)
    expected_actions: list[RouteAction] = Field(default_factory=list)
    must_cite: list[list[str]] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    no_suggestion_ok: bool = False
    tags: list[str] = Field(default_factory=list)

    @property
    def split(self) -> Literal["dev", "test"]:
        """按 id 的哈希稳定分组：20% 测试集（§13.1 留出，不参与调参）。"""
        h = int(hashlib.sha256(self.id.encode()).hexdigest(), 16)
        return "test" if h % 5 == 0 else "dev"


I, A = Intent, RouteAction  # noqa: E741


def _c(id: str, text: str, intent: Intent, **kw) -> EvalCase:  # noqa: A002
    return EvalCase(id=id, text=text, expected_intent=intent, **kw)


CASES: list[EvalCase] = [
    # ---------------------------------------------------------------- 售前：商品参数
    _c(
        "spec-01",
        "这个是纯棉的吗？",
        I.PRODUCT_SPEC,
        item_id="i_555",
        must_cite=[["item:i_555:attr:面料"]],
        must_include=["95%"],
        must_not_include=["100%纯棉", "纯棉的哦"],
    ),
    _c(
        "spec-02",
        "面料是什么成分",
        I.PRODUCT_SPEC,
        item_id="i_555",
        must_cite=[["item:i_555:attr:面料"]],
        must_include=["氨纶"],
    ),
    _c(
        "spec-03",
        "这条裙子厚不厚",
        I.PRODUCT_SPEC,
        item_id="i_555",
        must_cite=[["item:i_555:attr:厚薄"]],
        must_include=["常规"],
    ),
    _c(
        "spec-04",
        "裤子有弹力吗",
        I.PRODUCT_SPEC,
        item_id="i_777",
        must_cite=[["item:i_777:attr:弹力"]],
        must_include=["微弹"],
    ),
    _c(
        "spec-05",
        "这款会透吗",
        I.PRODUCT_SPEC,
        item_id="i_555",
        must_cite=[["item:i_555:attr:厚薄", "faq:5"]],
    ),
    _c(
        "spec-06",
        "这件防晒效果好吗",
        I.PRODUCT_SPEC,
        item_id="i_555",
        must_not_include=["防晒效果好", "可以防晒"],
        tags=["forbidden_claim"],
    ),
    _c(
        "spec-07",
        "真丝的会不会皱",
        I.PRODUCT_SPEC,
        item_id="i_888",
        must_not_include=["抗皱", "不会皱"],
        tags=["forbidden_claim"],
    ),
    _c(
        "spec-08", "高腰的吗", I.PRODUCT_SPEC, item_id="i_777", must_cite=[["item:i_777:attr:腰型"]]
    ),
    # ---------------------------------------------------------------- 售前：尺码
    _c(
        "size-01",
        "165 110斤穿什么码",
        I.SIZE_ADVICE,
        item_id="i_555",
        must_cite=[["item:i_555:size_advice"]],
        must_include=["M"],
        tags=["§11.1"],
    ),
    _c(
        "size-02",
        "身高168体重120斤选哪个码",
        I.SIZE_ADVICE,
        item_id="i_555",
        must_cite=[["item:i_555:size_advice"]],
        must_include=["L"],
    ),
    _c(
        "size-03",
        "平时穿S，这款要拍大一码吗",
        I.SIZE_ADVICE,
        item_id="i_555",
        must_cite=[["item:i_555:size_advice", "policy:size_tendency"]],
    ),
    _c(
        "size-04",
        "腰围68穿几码",
        I.SIZE_ADVICE,
        item_id="i_777",
        must_cite=[["item:i_777:size_chart"]],
        must_include=["27"],
        tags=["conservative"],
    ),
    _c(
        "size-05",
        "胸围90应该选M还是L",
        I.SIZE_ADVICE,
        item_id="i_555",
        must_cite=[["item:i_555:size_chart", "item:i_555:size_advice"]],
    ),
    _c(
        "size-06",
        "码数偏大还是偏小",
        I.SIZE_ADVICE,
        item_id="i_555",
        acceptable_intents=[I.PRODUCT_SPEC],
        must_cite=[["policy:size_tendency", "item:i_555:size_advice"]],
        must_include=["偏小"],
    ),
    # ---------------------------------------------------------------- 售前：使用方法
    _c(
        "usage-01",
        "能机洗吗",
        I.USAGE,
        item_id="i_555",
        must_cite=[["item:i_555:care"]],
        must_include=["手洗"],
        must_not_include=["可以机洗"],
    ),
    _c(
        "usage-02",
        "牛仔裤会掉色吗",
        I.USAGE,
        item_id="i_777",
        acceptable_intents=[I.PRODUCT_SPEC],
        must_cite=[["item:i_777:care"]],
    ),
    _c(
        "usage-03",
        "真丝衬衫怎么洗",
        I.USAGE,
        item_id="i_888",
        must_cite=[["item:i_888:care"]],
        must_include=["干洗"],
    ),
    _c(
        "usage-04",
        "可以暴晒吗",
        I.USAGE,
        item_id="i_555",
        must_cite=[["item:i_555:care"]],
        must_not_include=["可以暴晒"],
    ),
    # ---------------------------------------------------------------- 售前：库存
    _c(
        "stock-01",
        "白色S码还有货吗",
        I.STOCK,
        item_id="i_555",
        sku_id="k_3",
        must_cite=[["tool:stock:i_555:k_3"]],
        must_not_include=["有货的哦", "现货"],
    ),
    _c(
        "stock-02",
        "黑色M有货吗",
        I.STOCK,
        item_id="i_555",
        sku_id="k_1",
        must_cite=[["tool:stock:i_555:k_1"]],
    ),
    _c(
        "stock-03",
        "什么时候补货",
        I.STOCK,
        item_id="i_555",
        sku_id="k_3",
        must_cite=[["tool:stock:i_555:k_3"]],
        must_include=["10 月 8 日"],
    ),
    _c("stock-04", "还有货吗", I.STOCK, item_id="i_777", must_cite=[["tool:stock:i_777"]]),
    # ---------------------------------------------------------------- 售前：价格优惠（高风险）
    _c(
        "promo-01",
        "能便宜点吗",
        I.PRICE_PROMO,
        item_id="i_555",
        expected_actions=[A.HIGH_RISK_ANSWER],
        must_not_include=["给您便宜", "打折给您", "包邮给您"],
        tags=["high_risk"],
    ),
    _c(
        "promo-02",
        "有优惠券吗",
        I.PRICE_PROMO,
        item_id="i_555",
        expected_actions=[A.HIGH_RISK_ANSWER],
        must_cite=[["policy:promotions", "tool:price:i_555"]],
        must_include=["199"],
    ),
    _c(
        "promo-03",
        "满减能叠加平台券吗",
        I.PRICE_PROMO,
        item_id="i_777",
        expected_actions=[A.HIGH_RISK_ANSWER],
        must_cite=[["policy:promotions"]],
    ),
    _c(
        "promo-04",
        "这件衬衫参加满减吗",
        I.PRICE_PROMO,
        item_id="i_888",
        expected_actions=[A.HIGH_RISK_ANSWER],
        must_cite=[["tool:price:i_888"]],
        tags=["realtime_over_policy"],
    ),
    # ---------------------------------------------------------------- 售前：发货与快递
    _c(
        "ship-01",
        "什么时候发货",
        I.SHIPPING_POLICY,
        must_cite=[["policy:ship_time", "faq:2"]],
        must_include=["48"],
        must_not_include=["24 小时", "24小时"],
        tags=["conflict"],
    ),
    _c(
        "ship-02",
        "发什么快递",
        I.SHIPPING_POLICY,
        must_cite=[["policy:courier", "faq:1"]],
        must_include=["中通"],
    ),
    _c(
        "ship-03",
        "能发顺丰吗",
        I.SHIPPING_POLICY,
        must_cite=[["policy:courier_choice", "faq:1"]],
        must_include=["10"],
    ),
    _c(
        "ship-04",
        "新疆能到吗",
        I.SHIPPING_POLICY,
        must_cite=[["policy:remote_areas"]],
        must_include=["15"],
    ),
    _c(
        "ship-05",
        "寄香港可以吗",
        I.SHIPPING_POLICY,
        must_cite=[["policy:remote_areas"]],
        must_include=["不发货"],
    ),
    _c(
        "ship-06",
        "这裙子下单了多久能发",
        I.SHIPPING_POLICY,
        item_id="i_555",
        acceptable_intents=[I.URGE_SHIP],
        must_cite=[["policy:ship_time", "faq:2"]],
        must_include=["48"],
        must_not_include=["24"],
        tags=["conflict"],
    ),
    _c(
        "ship-07",
        "定制款多久发货",
        I.SHIPPING_POLICY,
        item_id="i_888",
        acceptable_intents=[I.PRODUCT_SPEC],
    ),
    # ---------------------------------------------------------------- 售前：发票赠品
    _c(
        "inv-01",
        "能开发票吗",
        I.INVOICE_GIFT,
        must_cite=[["policy:invoice", "faq:3"]],
        must_include=["电子普通发票"],
    ),
    _c("inv-02", "有运费险吗", I.INVOICE_GIFT, must_cite=[["policy:shipping_insurance", "faq:4"]]),
    _c(
        "inv-03",
        "有赠品吗",
        I.INVOICE_GIFT,
        must_cite=[["policy:gift"]],
        must_not_include=["送您", "有赠品"],
    ),
    _c(
        "inv-04",
        "发票多久能开出来",
        I.INVOICE_GIFT,
        must_cite=[["policy:invoice", "faq:3"]],
        must_include=["3"],
    ),
    # ---------------------------------------------------------------- 售前：对比真伪（高风险）
    _c(
        "auth-01",
        "是正品吗",
        I.COMPARE_AUTHENTIC,
        item_id="i_888",
        expected_actions=[A.HIGH_RISK_ANSWER],
        must_not_include=["100%正品", "绝对正品", "假一赔十"],
        tags=["high_risk"],
    ),
    _c(
        "auth-02",
        "这条和那条牛仔裤有什么区别",
        I.COMPARE_AUTHENTIC,
        item_id="i_777",
        expected_actions=[A.HIGH_RISK_ANSWER, A.CLARIFY],
    ),
    # ---------------------------------------------------------------- 售中：物流
    _c(
        "logi-01",
        "我的快递到哪了",
        I.LOGISTICS_TRACK,
        order_id="o_1002",
        must_cite=[["tool:logistics:o_1002"]],
        must_include=["10:30", "杭州"],
        must_not_include=["明天", "一定"],
        tags=["§11.2"],
    ),
    _c(
        "logi-02",
        "明天能到吗",
        I.LOGISTICS_TRACK,
        order_id="o_1002",
        must_cite=[["tool:logistics:o_1002"]],
        must_include=["截至"],
        must_not_include=["明天一定", "明天能到", "明天就到"],
        tags=["§11.5"],
    ),
    _c(
        "logi-03",
        "快递怎么好几天不动了",
        I.LOGISTICS_TRACK,
        order_id="o_1006",
        acceptable_intents=[I.URGE_SHIP],
        must_cite=[["tool:logistics:o_1006"]],
        tags=["stalled"],
    ),
    _c(
        "logi-04",
        "物流显示疑难件是什么意思",
        I.LOGISTICS_TRACK,
        order_id="o_1008",
        must_cite=[["tool:logistics:o_1008"]],
    ),
    _c(
        "logi-05",
        "快递到哪了",
        I.LOGISTICS_TRACK,
        order_id="o_1010",
        must_cite=[["tool:logistics:o_1010"]],
        must_include=["丰巢"],
        tags=["locker"],
    ),
    _c(
        "logi-06",
        "查下物流",
        I.LOGISTICS_TRACK,
        order_id="o_1009",
        must_cite=[["tool:logistics:o_1009"]],
    ),
    # ---------------------------------------------------------------- 售中：催发货
    _c(
        "urge-01",
        "怎么还没发货",
        I.URGE_SHIP,
        order_id="o_1001",
        must_cite=[["tool:order:o_1001"]],
        must_include=["48"],
    ),
    _c(
        "urge-02",
        "拍了一天了还不发",
        I.URGE_SHIP,
        order_id="o_1001",
        must_cite=[["tool:order:o_1001", "policy:ship_time"]],
        must_not_include=["今天一定发", "马上发"],
    ),
    _c(
        "urge-03",
        "能不能今天发货",
        I.URGE_SHIP,
        order_id="o_1001",
        must_not_include=["今天一定", "保证今天"],
    ),
    # ---------------------------------------------------------------- 售中：改单（高风险）
    _c(
        "mod-01",
        "能改个地址吗",
        I.MODIFY_ORDER,
        order_id="o_1001",
        expected_actions=[A.HIGH_RISK_ANSWER],
        must_cite=[["policy:modify_order"]],
        tags=["high_risk"],
    ),
    _c(
        "mod-02",
        "发货了还能改地址吗",
        I.MODIFY_ORDER,
        order_id="o_1002",
        expected_actions=[A.HIGH_RISK_ANSWER],
        must_cite=[["policy:modify_order"]],
        must_include=["无法"],
    ),
    _c(
        "mod-03",
        "我要改成测试省示例市演示区样例路 8 号",
        I.MODIFY_ORDER,
        order_id="o_1001",
        expected_actions=[A.HIGH_RISK_ANSWER],
        must_not_include=["样例路"],
        tags=["pii"],
    ),
    _c(
        "mod-04",
        "能换成L码吗",
        I.MODIFY_ORDER,
        order_id="o_1001",
        expected_actions=[A.HIGH_RISK_ANSWER],
        acceptable_intents=[I.SIZE_ADVICE],
    ),
    # ---------------------------------------------------------------- 售后：退换政策
    _c(
        "ret-01",
        "能退吗",
        I.RETURN_POLICY,
        buyer="BUYER_R3",
        order_id="o_1004",
        must_cite=[["tool:return:o_1004", "policy:seven_day_return"]],
    ),
    _c(
        "ret-02",
        "退货运费谁出",
        I.RETURN_POLICY,
        must_cite=[["policy:return_shipping"]],
        must_include=["买家"],
    ),
    _c(
        "ret-03",
        "定制的衬衫能七天无理由吗",
        I.RETURN_POLICY,
        buyer="BUYER_R8",
        order_id="o_1007",
        must_cite=[["policy:non_returnable", "tool:return:o_1007"]],
        must_not_include=["可以七天无理由"],
    ),
    _c(
        "ret-04",
        "怎么申请退货",
        I.RETURN_POLICY,
        must_cite=[["platform:taobao:apply_aftersale"]],
        must_include=["申请售后"],
    ),
    _c(
        "ret-05",
        "买了二十多天了还能退吗",
        I.RETURN_POLICY,
        buyer="BUYER_R5",
        order_id="o_1005",
        must_cite=[["tool:return:o_1005", "policy:seven_day_return"]],
    ),
    # ---------------------------------------------------------------- 售后：退款进度
    _c(
        "refund-01",
        "退款什么时候到",
        I.REFUND_PROGRESS,
        buyer="BUYER_REFUND",
        order_id="o_1012",
        must_cite=[["tool:aftersale:as_2008"]],
        must_not_include=["今天到账", "马上到账"],
    ),
    _c(
        "refund-02",
        "钱怎么还没退回来",
        I.REFUND_PROGRESS,
        buyer="BUYER_REFUND",
        order_id="o_1012",
        must_cite=[["tool:aftersale:as_2008"]],
    ),
    # ---------------------------------------------------------------- 售后：质量问题（高风险）
    _c(
        "qual-01",
        "收到的是坏的",
        I.QUALITY_ISSUE,
        buyer="BUYER_R5",
        order_id="o_1005",
        expected_actions=[A.HIGH_RISK_ANSWER, A.SOOTHE],
        must_include=["照片"],
        must_not_include=["全额赔偿", "赔您"],
        tags=["high_risk"],
    ),
    _c(
        "qual-02",
        "衣服有个洞",
        I.QUALITY_ISSUE,
        buyer="BUYER_R5",
        order_id="o_1005",
        expected_actions=[A.HIGH_RISK_ANSWER, A.SOOTHE],
    ),
    _c(
        "qual-03",
        "少发了一条裤子",
        I.QUALITY_ISSUE,
        buyer="BUYER_R5",
        order_id="o_1013",
        expected_actions=[A.HIGH_RISK_ANSWER, A.SOOTHE],
    ),
    # ---------------------------------------------------------------- 售后：签收未收到（高风险）
    _c(
        "nr-01",
        "显示签收了但我根本没收到",
        I.NOT_RECEIVED,
        order_id="o_1003",
        expected_actions=[A.HIGH_RISK_ANSWER, A.SOOTHE],
        must_cite=[["tool:logistics:o_1003"]],
        must_include=["菜鸟驿站"],
        tags=["§11.3"],
    ),
    _c(
        "nr-02",
        "快递说放柜子里了我没看到",
        I.NOT_RECEIVED,
        order_id="o_1010",
        expected_actions=[A.HIGH_RISK_ANSWER, A.SOOTHE],
        must_include=["丰巢"],
    ),
    # ---------------------------------------------------------------- 售后：使用故障
    _c(
        "mal-01",
        "拉链拉不上了",
        I.MALFUNCTION,
        item_id="i_777",
        acceptable_intents=[I.QUALITY_ISSUE],
    ),
    # ---------------------------------------------------------------- 投诉 / 人工
    _c(
        "cmp-01",
        "我要投诉你们",
        I.COMPLAINT,
        expected_actions=[A.HUMAN_PRIORITY],
        no_suggestion_ok=True,
        tags=["human"],
    ),
    _c(
        "cmp-02",
        "再不处理我就打12315",
        I.COMPLAINT,
        expected_actions=[A.HUMAN_PRIORITY, A.SOOTHE],
        no_suggestion_ok=True,
    ),
    _c(
        "human-01",
        "转人工",
        I.OTHER,
        acceptable_intents=[I.COMPLAINT, I.GREETING],
        expected_actions=[A.HUMAN_PRIORITY],
        no_suggestion_ok=True,
    ),
    _c(
        "emo-01",
        "你们这什么垃圾东西！！！气死我了",
        I.QUALITY_ISSUE,
        acceptable_intents=[I.COMPLAINT],
        expected_actions=[A.SOOTHE, A.HUMAN_PRIORITY],
        no_suggestion_ok=True,
        tags=["emotion"],
    ),
    # ---------------------------------------------------------------- 其他
    _c("greet-01", "在吗", I.GREETING, must_not_include=["第一"]),
    _c("greet-02", "你好，有人吗", I.GREETING),
    _c(
        "off-01",
        "加个微信吧，转账给你便宜点",
        I.OFF_PLATFORM,
        expected_actions=[A.OFF_PLATFORM_SCRIPT],
        must_include=["站外"],
        must_not_include=["微信号", "可以加"],
        tags=["§11.4"],
    ),
    _c("off-02", "能支付宝直接转你吗", I.OFF_PLATFORM, expected_actions=[A.OFF_PLATFORM_SCRIPT]),
    _c(
        "off-03",
        "留个手机号吧",
        I.OFF_PLATFORM,
        expected_actions=[A.OFF_PLATFORM_SCRIPT],
        must_not_include=["手机号是", "电话是"],
    ),
    _c("off-04", "你们有qq吗", I.OFF_PLATFORM, expected_actions=[A.OFF_PLATFORM_SCRIPT]),
    _c(
        "other-01",
        "你们店在哪个城市",
        I.OTHER,
        no_suggestion_ok=True,
        expected_actions=[A.ANSWER, A.CLARIFY, A.NO_SUGGESTION],
    ),
    # ---------------------------------------------------------------- 多问题
    _c(
        "multi-01",
        "什么时候发货？能开发票吗？",
        I.SHIPPING_POLICY,
        acceptable_intents=[I.INVOICE_GIFT],
        expected_actions=[A.SPLIT_MULTI, A.ANSWER],
        tags=["multi"],
    ),
    _c(
        "multi-02",
        "这个纯棉吗，165穿什么码",
        I.PRODUCT_SPEC,
        item_id="i_555",
        acceptable_intents=[I.SIZE_ADVICE],
        expected_actions=[A.SPLIT_MULTI, A.ANSWER],
        tags=["multi"],
    ),
    # ---------------------------------------------------------------- 多轮 / 指代
    _c(
        "ctx-01",
        "那件黑的有L吗",
        I.STOCK,
        item_id="i_555",
        history=[(Role.BUYER, "这条裙子还有货吗"), (Role.AGENT, "亲，有的哦")],
        acceptable_intents=[I.SIZE_ADVICE],
        must_cite=[["tool:stock:i_555:k_2", "item:i_555:sku"]],
        tags=["reference"],
    ),
    _c(
        "ctx-02",
        "它能机洗吗",
        I.USAGE,
        item_id="i_777",
        history=[(Role.BUYER, "这条牛仔裤弹力怎么样"), (Role.AGENT, "亲，是微弹的哦")],
        must_cite=[["item:i_777:care"]],
        tags=["reference"],
    ),
    _c(
        "ctx-03",
        "我手机13812345678，快递到哪了",
        I.LOGISTICS_TRACK,
        order_id="o_1002",
        must_cite=[["tool:logistics:o_1002"]],
        must_not_include=["13812345678"],
        tags=["pii"],
    ),
]


def cases(split: Literal["dev", "test", "all"] = "all") -> list[EvalCase]:
    if split == "all":
        return list(CASES)
    return [c for c in CASES if c.split == split]
