"""淘宝 / 天猫（千牛 · 旺旺）模拟适配器（方案 §2.1、§8.1 模式 A 或 B）。

淘宝是三个平台里接入路径最明确的：模式 A（"客服机器人""AI 智能体-人机接待"类目，平台推送、
接口回复）或模式 B（"客服面板插件"，千牛侧边栏）。

原始消息格式是合成的，字段名参照淘宝开放平台的命名习惯，**全部【待确认】**（附录 A 第 2 题：
插件或接口能否读取当前会话的文字、商品卡片和订单卡片）：

    {
      "seller_id": "s_1001",              店铺 ID
      "session_id": "conv_889",           会话 ID
      "msg_id": "m_20260928_0001",        平台消息 ID（同一平台内唯一）
      "msg_type": "text",                 text / item / order / image；voice、video、system 忽略；
                                          recall 撤回（recall_msg_id 是被撤回的消息）
      "direction": "in",                  in 买家发来；out 客服或机器人发出（回声）
      "buyer_open_uid": "tb_…",           买家 ID → buyer_ref（哈希），原值不向后传
      "buyer_nick": "…",                  买家昵称：丢弃（§5.3）
      "content": "165 110斤穿什么码",
      "item_id": "i_555", "sku_id": "k_1",  当前商品（买家从哪个商品进来，或发来的商品卡片）
      "tid": "o_1001",                    订单卡片（淘宝订单号）
      "sub_account": "小雅女装旗舰店:kf_02",  当前接待的子账号，agent_id 取冒号后的部分
      "send_time_ms": 1790563331000,      毫秒时间戳 → 东八区时间
      "reception": "human",               human 人工接待 / robot 机器人接待（模式 A 全自动）
      "recall_msg_id": "…",
      "pic_key": "img_…",                 图片消息（msg_type=image）的平台图片标识：不透明 ID，
                                          平台云内的视觉模型凭它取图（§15 Phase 3）；
                                          没有或不合格（网址等，见 base.image_ref）时整条忽略
      "pic_url": "https://…"              图片的临时地址（可能带签名）：丢弃，不向后传
    }

订单卡片里常见的收件人字段（receiver_name / receiver_mobile / receiver_address）同样丢弃。
图片消息的 content 常是占位文字"[图片]"，不算买家说的话；买家随图写了字时照常作为 text。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rodit.adapters.base import (
    BaseMockAdapter,
    RawEvent,
    caption,
    cst_from_epoch_ms,
    image_ref,
    opt_str,
)
from rodit.core.enums import Platform

_BUYER_TYPES = frozenset({"text", "item", "order", "image"})


class TaobaoMockAdapter(BaseMockAdapter):
    """千牛 / 旺旺模拟适配器：access 为 "A"（可开全自动发送）或 "B"（侧边栏插件）。"""

    platform = Platform.TAOBAO
    allowed_access = frozenset({"A", "B"})
    extension_fields = frozenset({"msg_type", "reception"})

    def _parse(self, raw: Mapping[str, Any]) -> RawEvent | None:
        msg_type = opt_str(raw.get("msg_type")) or "text"
        common = {
            "shop_id": opt_str(raw.get("seller_id")) or "",
            "conversation_id": opt_str(raw.get("session_id")) or "",
            "msg_id": opt_str(raw.get("msg_id")) or "",
        }
        at = cst_from_epoch_ms(raw["send_time_ms"]) if "send_time_ms" in raw else None
        if msg_type == "recall":
            return RawEvent(
                kind="recall", recalled_msg_id=opt_str(raw.get("recall_msg_id")), **common
            )
        if msg_type not in _BUYER_TYPES:
            return None  # 语音、视频、系统提示：不处理
        content = raw.get("content")
        text = content if isinstance(content, str) else ""
        direction = opt_str(raw.get("direction")) or "in"
        images: tuple[object, ...] = ()
        if msg_type == "image":
            key = image_ref(raw.get("pic_key"))  # pic_url 不读（§15 Phase 3，见模块说明）
            if key is None:
                return None  # 没有可用的图片标识：照旧忽略
            images = (key,)
            text = caption(text)
        if direction == "out":
            if msg_type == "image":
                return None  # 客服发的图片：会话记录只记文字
            return RawEvent(kind="agent", text=text, at=at, **common)
        if direction != "in":
            return None
        return RawEvent(
            kind="buyer",
            raw_buyer_id=opt_str(raw.get("buyer_open_uid")),
            text=text,
            item_id=opt_str(raw.get("item_id")),
            sku_id=opt_str(raw.get("sku_id")),
            order_id=opt_str(raw.get("tid")),
            agent_id=_sub_account_to_agent(raw.get("sub_account")),
            at=at,
            extension={"msg_type": msg_type, "reception": opt_str(raw.get("reception"))},
            auto_requested=opt_str(raw.get("reception")) == "robot",
            images=images,
            **common,
        )


def _sub_account_to_agent(value: Any) -> str | None:
    """ "店铺名:子账号" → 子账号（旺旺子账号的常见写法【待确认】）；没有冒号时原样返回。"""
    sub = opt_str(value)
    if sub is None:
        return None
    return sub.split(":", 1)[-1].strip() or None
