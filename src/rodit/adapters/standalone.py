"""独立话术助手（方案 §8.1 模式 C，过渡方案）。

客服把买家的问题复制到我们的网页工具（可以附上商品 ID），拿到推荐回答后自己复制回平台工作台。
不依赖任何平台准入，三个平台都能用，用来在 Phase 0 验证问答质量（§2.3 第 4 条、§15）。

- 没有平台数据：tools() 是 NullToolGateway，订单、物流、售后、库存、价格一律查不到，
  只能回答售前和政策类问题；也不接收订单号（查不了）。
- 没有平台买家 ID：buyer_ref 用网页工具的会话 ID 合成（哈希 "standalone:{session_id}"），
  同一个网页会话内多轮对话能对上，跨会话不关联任何买家。
- 客服粘贴的文字入口处就做个人信息最小化（mask_pii）：模式 C 用不到原文里的订单号去查询，
  手机号、地址这类个人信息没有理由进入系统（§5.3、§12.3）。
- 不做任何自动抓取（§2.4）：只处理客服主动提交的内容。
- 不能发送、不能转人工（ActionNotAllowed）；推荐放进网页工具的侧边栏（show_suggestions）。

网页工具提交的格式：

    {
      "shop_id": "s_3003",            客服登录网页工具时绑定的店铺
      "session_id": "web_7f3a",       网页工具的会话 ID（一次接待）
      "question": "这个是纯棉的吗",    客服粘贴的买家问题
      "item_id": "i_c1",              可选：商品 ID（客服从商品链接里复制）
      "sku_id": null,                 可选
      "agent_id": "kf_c01",           可选：登录的客服账号
      "request_id": "req_0001"        可选：网页端生成的请求 ID，同一会话内重试时用来去重
    }

msg_id = "standalone:{session_id}:req:{request_id}"；没有 request_id 时每次提交都是新消息。
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

from rodit.adapters.base import BaseMockAdapter, RawEvent, minimize, opt_str
from rodit.adapters.null_tools import NullToolGateway
from rodit.core.clock import Clock
from rodit.core.enums import Platform
from rodit.core.pii import BuyerRefHasher

STANDALONE_SOURCE = "standalone"


class StandaloneAssistAdapter(BaseMockAdapter):
    """模式 C 适配器（access 固定为 "C"）。platform 是店铺所在平台（StandardMessage.platform）。"""

    allowed_access = frozenset({"C"})
    extension_fields = frozenset({"source"})

    def __init__(
        self,
        platform: Platform,
        clock: Clock,
        hasher: BuyerRefHasher,
        *,
        dedup_capacity: int = 10_000,
        context_limit: int = 20,
        max_conversations: int = 10_000,
    ) -> None:
        self.platform = Platform(platform)
        self._request_seq = itertools.count(1)
        super().__init__(
            NullToolGateway(self.platform),
            clock,
            hasher,
            access="C",
            dedup_capacity=dedup_capacity,
            context_limit=context_limit,
            max_conversations=max_conversations,
        )

    @staticmethod
    def conversation_id_for(session_id: str) -> str:
        """网页工具会话 → 会话 ID（加前缀，不和平台会话 ID 冲突）。"""
        return f"{STANDALONE_SOURCE}:{session_id}"

    def _msg_id(self, session_id: str, request_id: str | None) -> str:
        """有 request_id 时用它（同一会话内重试去重）；没有时用提交时间 + 进程内序号，
        进程重启后也不会和已经落库的消息 ID 撞上（撞上会被会话仓储当成重复消息丢掉）。"""
        base = self.conversation_id_for(session_id)
        if request_id:
            return f"{base}:req:{request_id}"
        stamp = self._clock.now().strftime("%Y%m%d%H%M%S%f")
        return f"{base}:auto:{stamp}:{next(self._request_seq)}"

    def _parse(self, raw: Mapping[str, Any]) -> RawEvent | None:
        session_id = opt_str(raw.get("session_id"))
        question = raw.get("question")
        if session_id is None or not isinstance(question, str):
            return None
        msg_id = self._msg_id(session_id, opt_str(raw.get("request_id")))
        item_id = opt_str(raw.get("item_id"))
        return RawEvent(
            kind="buyer",
            shop_id=opt_str(raw.get("shop_id")) or "",
            conversation_id=self.conversation_id_for(session_id),
            msg_id=msg_id,
            raw_buyer_id=f"{STANDALONE_SOURCE}:{session_id}",
            text=minimize(question),
            item_id=item_id,
            sku_id=opt_str(raw.get("sku_id")) if item_id else None,
            agent_id=opt_str(raw.get("agent_id")),
            extension={"source": STANDALONE_SOURCE},
        )
