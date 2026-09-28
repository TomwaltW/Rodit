"""抖店（飞鸽）模拟适配器（方案 §2.1、§8.1 模式 B）。

抖店没有查到公开的飞鸽消息收发接口，官方扩展方式是在飞鸽"客服工具"里启用服务市场插件，
头部服务商的抖音机器人也只做智能辅助（§2.1）。所以只允许模式 B：插件上报当前会话，
在侧边栏展示推荐，由客服发出；不能直接发送，也没有转人工接口。

原始消息格式是合成的（插件上报的当前会话），**全部【待确认】**（§8.1：飞鸽客服工具微应用
能否读取会话内容；附录 A）：

    {
      "shop_id": "s_4004",
      "conversation_short_id": "dyc_001",       会话 ID
      "server_message_id": "7290000000000001",  平台消息 ID
      "sender": {
        "role": "customer",                     customer 买家 / customer_service 客服 / system
        "uid": "dy_…",                          买家 ID → buyer_ref（哈希），原值不向后传
        "nickname": "…"                         丢弃（§5.3）
      },
      "content": {
        "type": "text",                         text / product / order；image、video 忽略；
                                                withdraw 撤回（withdraw_msg_id 是被撤回的消息）
        "text": "…",
        "product_id": "i_d1", "sku_id": null,   商品卡片
        "order_id": "o_d001",                   订单卡片
        "withdraw_msg_id": "…"
      },
      "context": {"product_id": "…", "sku_id": "…", "order_id": "…"},
                                                插件看到的当前商品 / 订单（消息没带卡片时用）
      "create_time": 1790563331,                秒级时间戳 → 东八区时间
      "service_uid": "kf_d01",                  当前接待客服 → agent_id
      "source": "feige_plugin"
    }
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rodit.adapters.base import (
    BaseMockAdapter,
    RawEvent,
    as_mapping,
    cst_from_epoch_s,
    opt_str,
)
from rodit.core.enums import Platform

_BUYER_TYPES = frozenset({"text", "product", "order"})


class DouyinMockAdapter(BaseMockAdapter):
    """飞鸽插件模拟适配器：只支持 access="B"（只展示推荐，不发送、不转人工）。"""

    platform = Platform.DOUYIN
    allowed_access = frozenset({"B"})
    extension_fields = frozenset({"content_type", "source"})

    def _parse(self, raw: Mapping[str, Any]) -> RawEvent | None:
        sender = as_mapping(raw.get("sender"))
        content = as_mapping(raw.get("content"))
        context = as_mapping(raw.get("context"))
        content_type = opt_str(content.get("type")) or "text"
        common = {
            "shop_id": opt_str(raw.get("shop_id")) or "",
            "conversation_id": opt_str(raw.get("conversation_short_id")) or "",
            "msg_id": opt_str(raw.get("server_message_id")) or "",
        }
        at = cst_from_epoch_s(raw["create_time"]) if "create_time" in raw else None
        if content_type == "withdraw":
            return RawEvent(
                kind="recall", recalled_msg_id=opt_str(content.get("withdraw_msg_id")), **common
            )
        if content_type not in _BUYER_TYPES:
            return None  # 图片、视频：MVP 不处理（§1.4）
        text_value = content.get("text")
        text = text_value if isinstance(text_value, str) else ""
        role = opt_str(sender.get("role"))
        if role == "customer_service":
            return RawEvent(kind="agent", text=text, at=at, **common)
        if role != "customer":
            return None  # 系统提示等
        # 消息里带的商品 / 订单卡片优先，其次是插件看到的当前商品 / 订单
        item_id = opt_str(content.get("product_id"))
        sku_id = opt_str(content.get("sku_id")) if item_id else None
        if item_id is None:
            item_id = opt_str(context.get("product_id"))
            sku_id = opt_str(context.get("sku_id"))
        order_id = opt_str(content.get("order_id")) or opt_str(context.get("order_id"))
        return RawEvent(
            kind="buyer",
            raw_buyer_id=opt_str(sender.get("uid")),
            text=text,
            item_id=item_id,
            sku_id=sku_id,
            order_id=order_id,
            agent_id=opt_str(raw.get("service_uid")),
            at=at,
            extension={"content_type": content_type, "source": opt_str(raw.get("source"))},
            **common,
        )
