"""演示站的预设场景：方案 §11 的核心流程和几个典型问题，每个场景是工作台里的一位买家。

每个场景写明对应的方案章节、买家说的话（和发的图片）、带的商品 / 订单卡片，以及"会看到什么"
（给讲解的人看）。买家的话和图片都是合成的；预期行为有测试逐条核对（tests/showcase/
test_scenarios.py、test_images.py、test_semi_auto.py），用的是真实的问答引擎和离线替身。
拼多多只能做模式 C（§8.1）：话术助手查不到订单、收不到图片，带订单卡片和图片的场景不出现，
质检拦截场景换成不带订单的问法。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rodit.core.enums import Intent, Platform
from rodit.showcase.images import IMAGE_PLATFORMS
from rodit.showcase.llm import SIMULATED_NOTE

ALL_PLATFORMS = frozenset(Platform)
ORDER_PLATFORMS = frozenset({Platform.TAOBAO, Platform.DOUYIN})
"""能带订单卡片的平台（模式 B 插件能读到当前订单；拼多多模式 C 没有订单数据）。"""


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    section: str  # 方案章节
    text: str  # 买家说的话（只发图片时为空）
    intent: Intent  # 列表上显示的问题类型标签（实际路由结果以引擎为准）
    expect: str  # 会看到什么
    item_id: str | None = None
    order_id: str | None = None
    note: str | None = None  # 演示说明（如模拟大模型的行为）
    platforms: frozenset[Platform] = ALL_PLATFORMS
    images: tuple[str, ...] = ()  # 买家发的图片（合成标识，showcase.images）
    buyer: str = "BUYER"  # 哪位合成买家（fixtures.RAW_BUYERS 的键；订单卡片要是这位买家的）
    texts: dict[Platform, str] = field(default_factory=dict)  # 按平台换问法
    expects: dict[Platform, str] = field(default_factory=dict)  # 按平台换说明

    def text_for(self, platform: Platform) -> str:
        return self.texts.get(platform, self.text)

    def expect_for(self, platform: Platform) -> str:
        return self.expects.get(platform, self.expect)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "size",
        "尺码咨询",
        "§11.1",
        "165 110斤穿什么码",
        Intent.SIZE_ADVICE,
        "按商家填的尺码建议推荐 M 码，依据标明「商品属性 · 尺码建议」",
        item_id="i_555",
    ),
    Scenario(
        "ship_time",
        "发货时效",
        "§5.2.1",
        "这条裙子什么面料？多久能发货？",
        Intent.SHIPPING_POLICY,
        "详情页写的发货时效和店铺政策不一样：以店铺政策为准，详情页那句被覆盖",
        item_id="i_555",
    ),
    Scenario(
        "logistics",
        "物流查询",
        "§11.2",
        "我的快递到哪了",
        Intent.LOGISTICS_TRACK,
        "查物流轨迹，用模板回答、写明「截至」时间，不调用大模型",
        order_id="o_1002",
        platforms=ORDER_PLATFORMS,
    ),
    Scenario(
        "not_received",
        "签收未收到",
        "§11.3",
        "显示签收了但我根本没收到",
        Intent.NOT_RECEIVED,
        "查到驿站代收，推荐取件指引并标记风险；还没找到就引导在订单页申请售后",
        order_id="o_1003",
        platforms=ORDER_PLATFORMS,
    ),
    Scenario(
        "off_platform",
        "站外交易",
        "§11.4",
        "加个微信吧，转账给你便宜点",
        Intent.OFF_PLATFORM,
        "命中 §6.6 规则 1：直接用审核过的平台规则标准话术，不调用大模型",
    ),
    Scenario(
        "qa_block",
        "质检拦截",
        "§11.5",
        "催一下，明天能到吗？",
        Intent.URGE_SHIP,
        "第一稿承诺「明天一定能到」，质检拦下；带着原因重写成只说物流事实的回答",
        order_id="o_1002",
        note=SIMULATED_NOTE,
        texts={Platform.PDD: "明天能到吗？"},
        expects={
            Platform.PDD: "第一稿承诺「明天一定能到」，质检拦下；模式 C 查不到订单，"
            "重写成只说发货时效的回答。发货类虽然开了半自动，但第一稿被拦过，重写稿通过了"
            "质检也不标「可直接发送」，由客服看过再发"
        },
    ),
    Scenario(
        "emotion",
        "情绪激动",
        "§5.7",
        "都等了一个星期了物流一直不动！！气死我了",
        Intent.LOGISTICS_TRACK,
        "情绪达到 §6.6 规则 3：先推荐安抚话术并标红；物流停滞超过 48 小时，提醒客服向快递催查",
        order_id="o_1006",
        texts={Platform.PDD: "都等了一个星期了怎么还没发货！！气死我了"},
        expects={
            Platform.PDD: "情绪达到 §6.6 规则 3：先推荐安抚话术并标红；模式 C 查不到订单，"
            "回答只说发货政策"
        },
    ),
    Scenario(
        "complaint",
        "投诉",
        "§5.7",
        "你们再不处理我就去投诉",
        Intent.COMPLAINT,
        "投诉不给推荐，提醒客服优先处理（§5.7 投诉、纠纷由人处理）",
    ),
    Scenario(
        "multi",
        "一条消息两个问题",
        "§6.6 规则 5",
        "这个会透吗？能开发票吗？",
        Intent.PRODUCT_SPEC,
        "拆成两条分别推荐：透不透按商品 FAQ 回答，发票按店铺政策和 FAQ 回答",
        item_id="i_555",
    ),
    Scenario(
        "invoice",
        "FAQ 直出",
        "§5.4",
        "能开发票吗",
        Intent.INVOICE_GIFT,
        "FAQ 命中，直接用商家审核过的标准回答：便宜、快、零编造；发票类开了半自动，"
        "推荐直接填进客服输入框",
        expects={
            Platform.PDD: "FAQ 命中，直接用商家审核过的标准回答：便宜、快、零编造；"
            "发票类开了半自动，推荐标「可直接发送」"
        },
    ),
    Scenario(
        "ship_quick",
        "发货时间（半自动）",
        "§5.6、§13.4",
        "什么时候发货呀？",
        Intent.SHIPPING_POLICY,
        "高置信度的简单问题、这一类开了半自动：推荐直接填进客服输入框，客服看一眼按发送即可，"
        "仍是人工回复",
        expects={
            Platform.PDD: "高置信度的简单问题、这一类开了半自动：模式 C 没有平台输入框，推荐标"
            "「可直接发送」，客服核对后复制回多多客服"
        },
    ),
    Scenario(
        "price",
        "价格优惠",
        "§5.1 高风险",
        "能便宜点吗？有券吗",
        Intent.PRICE_PROMO,
        "价格类是高风险类型（§6.6 规则 4）：推荐只说活动配置里有的优惠，"
        "并提示「涉及价格承诺，请确认」",
        item_id="i_555",
    ),
    Scenario(
        "photo_damage",
        "破损照片",
        "§15 Phase 3、§7.3.1",
        "收到货发现破了，能退吗？",
        Intent.QUALITY_ISSUE,
        "图片理解认出破损（仅供参考，以凭证为准）；第一条是确定性模板：照片看到了、请在订单页申请"
        "售后并上传照片，不承诺处理结果。这笔订单的售后单 as_2009 在售后审核助手里",
        order_id="o_1013",
        buyer="BUYER_R5",
        images=("img_buyer_damage_1",),
        platforms=IMAGE_PLATFORMS,
    ),
    Scenario(
        "photo_size",
        "尺码截图",
        "§15 Phase 3、§11.1",
        "",
        Intent.SIZE_ADVICE,
        "买家只发了一张身高体重截图，没打字：图中文字交给路由和生成，按商家填的尺码建议推荐 M 码",
        item_id="i_555",
        images=("img_buyer_size_1",),
        platforms=IMAGE_PLATFORMS,
    ),
    Scenario(
        "photo_logistics",
        "物流截图（个人信息打码）",
        "§15 Phase 3、§5.3",
        "",
        Intent.LOGISTICS_TRACK,
        "截图面单上的手机号在识别结果里就已打码，物流截图的图中文字也不交给大模型；"
        "回答按实时物流轨迹写明「截至」",
        order_id="o_1003",
        images=("img_buyer_logistics_1",),
        platforms=IMAGE_PLATFORMS,
    ),
)

SCENARIO_IDS = frozenset(s.id for s in SCENARIOS)


def scenarios_for(platform: Platform) -> list[Scenario]:
    return [s for s in SCENARIOS if platform in s.platforms]


def get_scenario(platform: Platform, scenario_id: str) -> Scenario | None:
    return next((s for s in scenarios_for(platform) if s.id == scenario_id), None)


__all__ = [
    "IMAGE_PLATFORMS",
    "ORDER_PLATFORMS",
    "SCENARIOS",
    "SCENARIO_IDS",
    "Scenario",
    "get_scenario",
    "scenarios_for",
]
