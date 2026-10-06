"""演示站的合成数据：每个平台单元一家演示店铺（方案 §9.1 一个平台一个单元）。

都从 rodit.testing.fixtures.build_demo_world 出发（全部合成，CLAUDE.md 只用合成数据）：

- 淘宝：小雅女装旗舰店（s_1001），演示世界原样；
- 抖店：抖抖女装（s_4004）。演示世界里这家店只有一件商品，这里把淘宝演示店的商品、政策、FAQ、
  订单、物流、售后、库存和价格照样复制一份过来（店铺 ID、平台、买家标识换成抖店的；快递换成店铺
  政策里的韵达），店铺自己已有的政策（发货时效、默认快递）保留；
- 拼多多：拼拼小铺（s_3003），只能做模式 C 独立话术助手（§8.1）。复制商品、政策、FAQ 给话术助手用；
  订单、物流、售后只给售后审核助手用（售后接口在多多云内调用，§2.1、§7.1），**不复制库存和价格**：
  模式 C 没有平台实时数据。

只改内存里的演示世界，不改 rodit.testing.fixtures。买家标识都是按合成的平台买家 ID 哈希出来的
（buyer_ref，§8.3），和接入层模拟适配器用同一个盐，订单才对得上；页面上从不显示它们。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rodit.core.enums import Intent, KnowledgeStatus, Platform, PolicyKey
from rodit.core.pii import hash_buyer_id
from rodit.testing.fixtures import (
    MANAGER,
    RAW_BUYERS,
    SHOP_A,
    SHOP_C,
    SHOP_D,
    TEST_SALT,
    DemoWorld,
    build_demo_world,
)


@dataclass(frozen=True)
class ShopProfile:
    """一个平台单元里的演示店铺。"""

    platform: Platform
    shop_id: str
    shop_name: str
    workbench: str  # 客服工作台（§2.1）
    access: str  # 接入模式（§8.1）：B 工作台插件 / C 独立话术助手
    access_label: str
    accent: str  # 页面上的平台主色
    courier: str
    waybill_prefix: str
    buyer_prefix: str  # 合成的平台买家 ID 前缀


PROFILES: dict[Platform, ShopProfile] = {
    Platform.TAOBAO: ShopProfile(
        Platform.TAOBAO,
        SHOP_A,
        "小雅女装旗舰店",
        "千牛",
        "B",
        "模式 B · 千牛客服面板插件：侧边栏推荐，客服点选发送",
        "#ff5000",
        "中通快递",
        "ZT",
        "tb_",
    ),
    Platform.DOUYIN: ShopProfile(
        Platform.DOUYIN,
        SHOP_D,
        "抖抖女装",
        "飞鸽",
        "B",
        "模式 B · 飞鸽客服工具插件：侧边栏推荐，客服点选发送",
        "#fe2c55",
        "韵达快递",
        "YD",
        "dy_",
    ),
    Platform.PDD: ShopProfile(
        Platform.PDD,
        SHOP_C,
        "拼拼小铺",
        "多多客服",
        "C",
        "模式 C · 独立话术助手：客服把买家的问题粘贴进来，拿到推荐后自己复制回多多客服；"
        "没有订单数据",
        "#e02e24",
        "圆通快递",
        "YT",
        "pdd_",
    ),
}

AGENT_ID = "kf_02"
"""演示里接待的客服子账号（合成）。"""

OPERATOR = MANAGER
"""售后审核里做确认的店长子账号（合成，ShopConfig.manager_ids 里的店长）。"""

SHOP_SETTINGS: dict[str, Any] = {
    "manager_ids": [MANAGER],
    "aftersale_amount_limit": "500",
    "suggestion_count": 2,
    # 客服工作时间（附录 C 第 13 题），和店铺政策"客服在线时间 9:00–23:00"一致；
    # 全自动开启后只在这之外生效（§15 Phase 3 先在非工作时间试行），演示店铺没开全自动
    "service_hours": ["09:00-23:00"],
}
"""三家演示店铺统一的店铺配置（经 KnowledgeRepository.update_shop_config 写入，§9.3）。"""

REVIEWER = "qa_reviewer"
"""开半自动的复核人员（合成账号；§13.4 半自动由复核人员按门槛开启）。"""

SEMI_AUTO_INTENTS: tuple[Intent, ...] = (Intent.SHIPPING_POLICY, Intent.INVOICE_GIFT)
"""演示店铺开了半自动的问题类型（§5.6）：店铺政策、FAQ 能答的低风险问题。"""

SEMI_AUTO_NOTE = (
    "演示站：为展示半自动直接开启；合成历史每类每周只有几组推荐，达不到 §13.4 的样本量和原样采纳率"
)
"""强制开启半自动时写进审计的原因（门槛没过时 rodit.app.rollout 要求写明）。"""

# 复制到抖店 / 拼多多时，快递政策和 FAQ 回答里要跟着换的文字（和各店自己的政策一致）
_FAQ_SWAPS: dict[Platform, tuple[tuple[str, str], ...]] = {
    Platform.TAOBAO: (),
    Platform.DOUYIN: (("中通", "韵达"),),
    Platform.PDD: (("中通", "圆通"), ("48 小时", "24 小时")),
}
# 商品详情页：淘宝、抖店的详情页写 24 小时发货，和店铺政策（48 小时）冲突；拼多多的政策本来就是
# 24 小时，详情页改成 12 小时，同样演示"以店铺政策为准"（§5.2.1）
_DETAIL_SWAPS: dict[Platform, tuple[tuple[str, str], ...]] = {
    Platform.TAOBAO: (),
    Platform.DOUYIN: (),
    Platform.PDD: (("24 小时", "12 小时"),),
}


def raw_buyer_id(platform: Platform, name: str) -> str:
    """合成的平台买家 ID：淘宝用演示世界的原值，其他平台换前缀（如 tb_demo_buyer_001 → dy_…）。"""
    _, raw = RAW_BUYERS[name]
    if platform == Platform.TAOBAO:
        return raw
    return PROFILES[platform].buyer_prefix + raw.removeprefix("tb_")


def buyer_ref(platform: Platform, name: str) -> str:
    return hash_buyer_id(platform.value, raw_buyer_id(platform, name), salt=TEST_SALT)


def _taobao_names() -> dict[str, str]:
    """淘宝演示世界里的 buyer_ref → 买家名（fixtures.RAW_BUYERS 的键）。"""
    return {
        hash_buyer_id("taobao", raw, salt=TEST_SALT): name
        for name, (platform, raw) in RAW_BUYERS.items()
        if platform == "taobao"
    }


def _swap(text: str | None, swaps: tuple[tuple[str, str], ...]) -> str | None:
    if text is None:
        return None
    for old, new in swaps:
        text = text.replace(old, new)
    return text


def _copy_shop_a(world: DemoWorld, platform: Platform, *, realtime: bool) -> None:
    """把淘宝演示店（s_1001）的数据复制给 platform 的演示店铺（见模块说明）。"""
    profile = PROFILES[platform]
    shop_id = profile.shop_id
    names = _taobao_names()

    def ref(old: str) -> str:
        return buyer_ref(platform, names[old])

    own_policies = {p.key for p in world.policies if p.shop_id == shop_id}
    for p in [p for p in world.policies if p.shop_id == SHOP_A and p.key not in own_policies]:
        value = p.value
        if p.key == PolicyKey.COURIER.value:
            value = _swap(value, _FAQ_SWAPS[platform]) or value
        world.policies.append(p.model_copy(update={"shop_id": shop_id, "value": value}))
    for card in [c for c in world.items if c.shop_id == SHOP_A]:
        world.items.append(
            card.model_copy(
                deep=True,
                update={
                    "shop_id": shop_id,
                    "platform": platform,
                    "detail_text": _swap(card.detail_text, _DETAIL_SWAPS[platform]),
                },
            )
        )
    for faq in [f for f in world.faqs if f.shop_id == SHOP_A]:
        if faq.status == KnowledgeStatus.RETIRED:
            continue
        world.faqs.append(
            faq.model_copy(
                update={"shop_id": shop_id, "answer": _swap(faq.answer, _FAQ_SWAPS[platform])}
            )
        )
    courier = profile.courier

    def waybill(no: str | None) -> str | None:
        return profile.waybill_prefix + no[2:] if no else None

    for order in [o for o in world.orders if o.shop_id == SHOP_A]:
        world.orders.append(
            order.model_copy(
                deep=True,
                update={
                    "shop_id": shop_id,
                    "buyer_ref": ref(order.buyer_ref),
                    "waybill_no": waybill(order.waybill_no),
                    "courier": courier if order.waybill_no else None,
                },
            )
        )
    for trace in [t for t in world.traces if t.shop_id == SHOP_A]:
        world.traces.append(
            trace.model_copy(
                deep=True,
                update={
                    "shop_id": shop_id,
                    "waybill_no": waybill(trace.waybill_no),
                    "courier": courier,
                },
            )
        )
    for aftersale in [a for a in world.aftersales if a.shop_id == SHOP_A]:
        world.aftersales.append(
            aftersale.model_copy(
                deep=True, update={"shop_id": shop_id, "buyer_ref": ref(aftersale.buyer_ref)}
            )
        )
    if realtime:
        for stock in [s for s in world.stocks if s.shop_id == SHOP_A]:
            world.stocks.append(stock.model_copy(update={"shop_id": shop_id}))
        for price in [p for p in world.prices if p.shop_id == SHOP_A]:
            world.prices.append(price.model_copy(deep=True, update={"shop_id": shop_id}))


def build_world(platform: Platform, now: datetime) -> DemoWorld:
    """platform 单元的演示世界（每次调用都是一份新的数据）。"""
    world = build_demo_world(now=now)
    if platform == Platform.DOUYIN:
        _copy_shop_a(world, platform, realtime=True)
    elif platform == Platform.PDD:
        _copy_shop_a(world, platform, realtime=False)
    return world


__all__ = [
    "AGENT_ID",
    "OPERATOR",
    "PROFILES",
    "REVIEWER",
    "SEMI_AUTO_INTENTS",
    "SEMI_AUTO_NOTE",
    "SHOP_SETTINGS",
    "ShopProfile",
    "build_world",
    "buyer_ref",
    "raw_buyer_id",
]
