"""拼多多（多多客服）模拟适配器（方案 §2.1、§8.1）。

拼多多没有查到面向商家服务商的客服消息开放接口，现有方案多依赖 RPA 或模拟商家后台——
这些我们一律不用（§2.4）。所以当前只允许模式 C 的能力：
- 没有平台工具数据（tools 必须是 NullToolGateway，只能答售前和政策类问题）；
- 不能发送、不能转人工，推荐只展示在我们自己的网页工具里，由客服复制回多多客服；
- 不做任何自动抓取：消息只来自客服主动提交（或平台将来开放的正式能力）。

下面的原始格式是按多多客服会话字段命名合成的，**全部【待确认】**（附录 A："拼多多：是否有面向
服务商的客服消息能力"）。它的作用是把拼多多的平台差异（数字消息类型码、秒级时间戳、
goods_id / order_sn 命名）固定在接入层：平台开放正式能力后，只需放开 allowed_access、
换成真实推送，引擎不用改。客服手工粘贴问题的通用入口见 StandaloneAssistAdapter。

    {
      "mall_id": "s_3003",                店铺 ID
      "conv_id": "pddc_001",              会话 ID
      "msg_id": "pdd_m_0001",             平台消息 ID
      "type": 1,                          1 文本、11 商品卡片、12 订单卡片；
                                          2 图片、31 系统提示：忽略；
                                          1002 撤回（recall_msg_id 是被撤回的消息）
      "from": {"role": "buyer", "uid": "pdd_…"},   buyer 买家 / mall_cs 客服 / system；
                                          uid → buyer_ref（哈希），原值不向后传
      "nickname": "…",                    丢弃（§5.3）
      "content": "…",
      "goods_id": "i_c1", "sku_id": null, 商品卡片
      "order_sn": "…",                    订单卡片
      "ts": 1790563331,                   秒级时间戳 → 东八区时间
      "cs_account": "kf_c01"              当前接待客服 → agent_id
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

TYPE_TEXT = 1
TYPE_IMAGE = 2
TYPE_GOODS = 11
TYPE_ORDER = 12
TYPE_SYSTEM = 31
TYPE_RECALL = 1002
_BUYER_TYPES = frozenset({TYPE_TEXT, TYPE_GOODS, TYPE_ORDER})


class PddMockAdapter(BaseMockAdapter):
    """多多客服模拟适配器：目前只支持 access="C"（见模块说明）。"""

    platform = Platform.PDD
    allowed_access = frozenset({"C"})
    extension_fields = frozenset({"msg_type_code"})

    def _parse(self, raw: Mapping[str, Any]) -> RawEvent | None:
        sender = as_mapping(raw.get("from"))
        msg_type = _type_code(raw.get("type", TYPE_TEXT))
        common = {
            "shop_id": opt_str(raw.get("mall_id")) or "",
            "conversation_id": opt_str(raw.get("conv_id")) or "",
            "msg_id": opt_str(raw.get("msg_id")) or "",
        }
        at = cst_from_epoch_s(raw["ts"]) if "ts" in raw else None
        if msg_type == TYPE_RECALL:
            return RawEvent(
                kind="recall", recalled_msg_id=opt_str(raw.get("recall_msg_id")), **common
            )
        if msg_type not in _BUYER_TYPES:
            return None  # 图片、系统提示等：MVP 不处理（§1.4）
        content = raw.get("content")
        text = content if isinstance(content, str) else ""
        role = opt_str(sender.get("role"))
        if role == "mall_cs":
            return RawEvent(kind="agent", text=text, at=at, **common)
        if role != "buyer":
            return None
        return RawEvent(
            kind="buyer",
            raw_buyer_id=opt_str(sender.get("uid")),
            text=text,
            item_id=opt_str(raw.get("goods_id")),
            sku_id=opt_str(raw.get("sku_id")),
            order_id=opt_str(raw.get("order_sn")),
            agent_id=opt_str(raw.get("cs_account")),
            at=at,
            extension={"msg_type_code": msg_type},
            **common,
        )


def _type_code(value: Any) -> int:
    """消息类型码：整数或数字字符串，其他抛 ValueError。"""
    if isinstance(value, bool):
        raise ValueError(f"消息类型码不对：{value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise ValueError(f"消息类型码不对：{value!r}")
