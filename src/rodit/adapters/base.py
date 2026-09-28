"""平台接入层的公共行为（方案 §8.1 三种接入模式、§8.2 适配器接口、§8.3 标准会话格式）。

各平台适配器只负责把平台原始消息解析成 RawEvent（字段名、消息类型码、时间格式这些平台差异）；
下面这些规则统一在 BaseMockAdapter 里执行，子类不能绕开：

- 去重：按 (shop_id, msg_id) 去重（§8.3 同一平台内按 msg_id 去重），每个店铺一个有界 LRU；
- 买家身份：平台买家 ID 只在这里用注入的 BuyerRefHasher 哈希成 buyer_ref，原值不向后传；
- 个人信息：昵称、手机号、收件地址等字段一律丢弃（§5.3）。extension 只保留子类白名单里的
  平台字段，白名单不能包含 PII_FIELD_NAMES，字符串取值含个人信息时也丢弃；
- 接入模式：A 官方机器人 / 消息接口（可全自动发送、转人工）；B 工作台插件（只展示推荐）；
  C 独立话术助手（没有平台数据，tools 必须是 NullToolGateway）。send 只在模式 A 且开了
  全自动时可用，transfer_to_human 只在模式 A 可用，其余抛 ActionNotAllowed（§8.2）；
- 会话上下文：本地记录买家消息、客服回声、发出的消息和最近的商品 / 订单卡片，供 get_context；
- 推荐展示：show_suggestions 把推荐卡片放进该会话的"侧边栏"（模式 B；模式 A 的辅助、模式 C 的
  网页工具同理），新的一批替换旧的。

所有状态按 shop_id 分开保存（§9.2）；线程安全（一把锁）。这是模拟实现：
真实适配器部署在对应平台云内（§9.1），发送、转人工调用平台接口。
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal, get_args

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from rodit.adapters.null_tools import NullToolGateway
from rodit.core.clock import CST, Clock
from rodit.core.enums import Mode, Platform, Role
from rodit.core.errors import ActionNotAllowed
from rodit.core.models import (
    ItemRef,
    OrderRef,
    PlatformContext,
    SendResult,
    StandardMessage,
    StoredMessage,
    SuggestionCard,
)
from rodit.core.pii import BuyerRefHasher, contains_pii, mask_pii
from rodit.core.ports import ToolGateway

AccessMode = Literal["A", "B", "C"]
"""接入模式（§8.1）：A 官方机器人 / 消息接口，B 工作台插件，C 独立话术助手。"""

ACCESS_MODES: tuple[AccessMode, ...] = get_args(AccessMode)

PII_FIELD_NAMES = frozenset(
    {
        # 买家身份与昵称
        "buyer_nick",
        "buyer_name",
        "nick",
        "nickname",
        "user_nick",
        "display_name",
        "avatar",
        # 手机号、收件信息
        "mobile",
        "phone",
        "telephone",
        "buyer_mobile",
        "buyer_phone",
        "receiver_name",
        "receiver_mobile",
        "receiver_phone",
        "receiver_address",
        "receiver_state",
        "receiver_city",
        "receiver_district",
        "address",
        "post_tel",
        "id_card",
        "email",
    }
)
"""平台原始消息里常见的个人信息字段名：不管哪个平台，一律不进 StandardMessage（§5.3）。"""

_MAX_TIMESTAMP_S = 4_102_444_800  # 2100-01-01，超过视为非法时间戳


def cst_from_epoch_ms(value: Any) -> datetime:
    """毫秒时间戳 → 东八区时间。不是数字或超出范围抛 ValueError。"""
    return cst_from_epoch_s(_number(value) / 1000)


def cst_from_epoch_s(value: Any) -> datetime:
    """秒级时间戳 → 东八区时间。不是数字或超出范围抛 ValueError。"""
    seconds = _number(value)
    if not 0 < seconds < _MAX_TIMESTAMP_S:
        raise ValueError(f"时间戳超出范围：{value!r}")
    return datetime.fromtimestamp(seconds, tz=CST)


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"不是数字：{value!r}")
    return float(value)


def opt_str(value: Any) -> str | None:
    """平台字段 → 非空字符串或 None（数字 ID 转成字符串；其他类型视为没有）。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def minimize(text: str) -> str:
    """个人信息最小化：有个人信息时返回 mask_pii 的结果，没有时原样返回。

    mask_pii 会把整段文字做 NFKC 归一（全角标点变半角）；没有个人信息时不必改动原文。
    """
    return mask_pii(text) if contains_pii(text) else text


@dataclass(frozen=True)
class RawEvent:
    """子类从平台原始消息解析出的中间结果（只在接入层内部使用）。

    kind：buyer 买家消息 → StandardMessage；agent 客服 / 机器人发出的消息回声 → 只记进本地上下文；
    recall 撤回 → 从本地上下文删除 recalled_msg_id。其他事件（系统提示、图片等）子类直接返回 None。
    raw_buyer_id 是平台原始买家 ID，只在 BaseMockAdapter 里哈希，不会出现在任何输出里。
    """

    kind: Literal["buyer", "agent", "recall"]
    shop_id: str
    conversation_id: str
    msg_id: str
    raw_buyer_id: str | None = None
    text: str = ""
    item_id: str | None = None
    sku_id: str | None = None
    order_id: str | None = None
    agent_id: str | None = None
    at: datetime | None = None
    extension: Mapping[str, Any] = field(default_factory=dict)
    auto_requested: bool = False  # 平台说明这条消息由机器人接待（模式 A 全自动）
    recalled_msg_id: str | None = None


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutgoingMessage(_Model):
    """send 发出的消息（模拟平台发送记录）。"""

    shop_id: str
    conversation_id: str
    platform_msg_id: str
    text: str
    at: AwareDatetime


class TransferRecord(_Model):
    """transfer_to_human 的记录（全自动模式转人工，§5.7）。reason 已做个人信息最小化。"""

    shop_id: str
    conversation_id: str
    reason: str
    at: AwareDatetime


class _LRU:
    """有界的"最近见过"集合：超出容量时淘汰最久没见过的。"""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._data: OrderedDict[str, None] = OrderedDict()

    def add(self, key: str) -> bool:
        """第一次见到返回 True；见过返回 False（并刷新为最近）。"""
        if key in self._data:
            self._data.move_to_end(key)
            return False
        self._data[key] = None
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)
        return True

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class _Conversation:
    messages: deque[StoredMessage]
    item: ItemRef | None = None
    order: OrderRef | None = None
    cards: list[SuggestionCard] = field(default_factory=list)
    transferred: bool = False


class BaseMockAdapter:
    """实现 core.ports.PlatformAdapter 的公共部分。子类设置 platform / allowed_access /
    extension_fields，并实现 _parse。

    - dedup_capacity：每个店铺记住的最近 msg_id 个数；
    - context_limit：每个会话在本地保留的最近消息条数（get_context 返回这些）；
    - max_conversations：每个店铺在本地保留的会话数，超出淘汰最久没有消息的会话。
    """

    platform: Platform
    allowed_access: ClassVar[frozenset[str]] = frozenset(ACCESS_MODES)
    extension_fields: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        leaked = cls.extension_fields & PII_FIELD_NAMES
        if leaked:
            raise TypeError(
                f"{cls.__name__}.extension_fields 不能包含个人信息字段：{sorted(leaked)}"
            )

    def __init__(
        self,
        tools: ToolGateway,
        clock: Clock,
        hasher: BuyerRefHasher,
        *,
        access: AccessMode,
        auto_send_enabled: bool = False,
        dedup_capacity: int = 10_000,
        context_limit: int = 20,
        max_conversations: int = 10_000,
    ) -> None:
        if access not in ACCESS_MODES:
            raise ValueError(f"未知的接入模式：{access!r}")
        if access not in self.allowed_access:
            raise ValueError(
                f"{self.platform.value} 目前只支持接入模式 {sorted(self.allowed_access)}"
                f"（方案 §8.1），不能用 {access}"
            )
        if auto_send_enabled and access != "A":
            raise ValueError("只有模式 A（官方机器人 / 消息接口）能全自动发送（方案 §8.1、§8.2）")
        if access == "C" and not isinstance(tools, NullToolGateway):
            raise ValueError("模式 C 没有平台数据（方案 §8.1）：tools 必须是 NullToolGateway")
        tools_platform = getattr(tools, "platform", None)
        if tools_platform != self.platform:
            raise ValueError(
                f"平台工具属于 {tools_platform}，适配器属于 {self.platform.value}："
                "每个平台一个部署单元，数据不跨云（方案 §2.2、§9.1）"
            )
        if min(dedup_capacity, context_limit, max_conversations) < 1:
            raise ValueError("dedup_capacity / context_limit / max_conversations 必须 ≥ 1")
        self.access: AccessMode = access
        self.auto_send_enabled = auto_send_enabled
        self._tools = tools
        self._clock = clock
        self._hasher = hasher
        self._dedup_capacity = dedup_capacity
        self._context_limit = context_limit
        self._max_conversations = max_conversations
        self._lock = threading.Lock()
        self._seen: dict[str, _LRU] = {}
        self._conversations: dict[str, OrderedDict[str, _Conversation]] = {}
        self._outbox: list[OutgoingMessage] = []
        self._transfers: list[TransferRecord] = []
        self._out_seq = itertools.count(1)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(platform={self.platform.value!r}, access={self.access!r})"

    # ------------------------------------------------------------ 子类实现

    def _parse(self, raw: Mapping[str, Any]) -> RawEvent | None:
        """平台原始消息 → RawEvent；不关心的事件返回 None。字段缺失 / 类型不对可以直接抛
        KeyError / TypeError / ValueError，on_message 会当作无法识别返回 None。"""
        raise NotImplementedError

    # ------------------------------------------------------------ PlatformAdapter

    def on_message(self, raw: dict) -> StandardMessage | None:
        """模式 A 平台推送 / 模式 B 插件上报 / 模式 C 网页工具提交 → 标准消息（§8.2、§8.3）。

        不是买家消息（客服回声、撤回、系统提示）、重复消息、无法识别时返回 None。
        """
        if not isinstance(raw, Mapping):
            return None
        try:
            event = self._parse(raw)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            return None
        if event is None or not all(
            isinstance(v, str) and v.strip()
            for v in (event.shop_id, event.conversation_id, event.msg_id)
        ):
            return None
        if event.kind == "buyer":
            return self._accept_buyer(event)
        if event.kind == "agent":
            self._accept_agent(event)
        elif event.kind == "recall":
            self._accept_recall(event)
        return None

    def get_context(self, shop_id: str, conversation_id: str) -> PlatformContext:
        """商品卡片、订单卡片、最近对话（本地记录，时间升序）。没见过的会话返回空上下文。"""
        with self._lock:
            conv = self._conversation(shop_id, conversation_id)
            if conv is None:
                return PlatformContext(shop_id=shop_id, conversation_id=conversation_id)
            return PlatformContext(
                shop_id=shop_id,
                conversation_id=conversation_id,
                item_card=conv.item.model_copy() if conv.item else None,
                order_card=conv.order.model_copy() if conv.order else None,
                # 平台推送可能乱序：按消息时间升序（同一时间保持到达顺序）
                recent_messages=[
                    m.model_copy() for m in sorted(conv.messages, key=lambda m: m.created_at)
                ],
            )

    def show_suggestions(
        self, shop_id: str, conversation_id: str, cards: list[SuggestionCard]
    ) -> None:
        """把推荐卡片放进该会话的侧边栏，替换上一批（§5.6、§8.2）；空列表表示清空。

        没收到过这个店铺这个会话的消息时不展示（插件只能在当前会话里展示）。
        """
        with self._lock:
            conv = self._conversation(shop_id, conversation_id)
            if conv is not None:
                conv.cards = [c.model_copy(deep=True) for c in cards]

    def send(self, shop_id: str, conversation_id: str, text: str) -> SendResult:
        """只有模式 A 且开了全自动时可用（text 已由引擎加好 AI 标识前缀，§12.3），其他情况抛
        ActionNotAllowed。空消息、会话不存在、会话已转人工时不发送，返回 ok=False。"""
        if self.access != "A" or not self.auto_send_enabled:
            raise ActionNotAllowed(
                f"{self.platform.value} 接入模式 {self.access}"
                f"{'' if self.access != 'A' else '（未开全自动）'}不能直接发送，"
                "推荐只能由客服发出（方案 §8.1、§8.2）"
            )
        if not isinstance(text, str) or not text.strip():
            return SendResult(ok=False, error="空消息不发送")
        with self._lock:
            conv = self._conversation(shop_id, conversation_id)
            if conv is None:
                return SendResult(ok=False, error="会话不存在")
            if conv.transferred:
                return SendResult(ok=False, error="会话已转人工，由客服接管（方案 §5.7）")
            now = self._clock.now()
            platform_msg_id = f"{self.platform.value}_out_{next(self._out_seq):06d}"
            conv.messages.append(
                StoredMessage(
                    msg_id=platform_msg_id,
                    shop_id=shop_id,
                    conversation_id=conversation_id,
                    role=Role.BOT,
                    text=text,
                    created_at=now,
                )
            )
            self._outbox.append(
                OutgoingMessage(
                    shop_id=shop_id,
                    conversation_id=conversation_id,
                    platform_msg_id=platform_msg_id,
                    text=text,
                    at=now,
                )
            )
        return SendResult(ok=True, platform_msg_id=platform_msg_id)

    def transfer_to_human(self, shop_id: str, conversation_id: str, reason: str) -> SendResult:
        """全自动模式下转人工（§5.7、GB/T 47746）。只有模式 A 支持；转过人工的会话，
        之后的买家消息不再标记为全自动。"""
        if self.access != "A":
            raise ActionNotAllowed(
                f"{self.platform.value} 接入模式 {self.access} 没有转人工接口（方案 §8.2）"
            )
        with self._lock:
            conv = self._conversation(shop_id, conversation_id)
            if conv is None:
                return SendResult(ok=False, error="会话不存在")
            conv.transferred = True
            self._transfers.append(
                TransferRecord(
                    shop_id=shop_id,
                    conversation_id=conversation_id,
                    reason=minimize(reason or ""),
                    at=self._clock.now(),
                )
            )
        return SendResult(ok=True)

    def tools(self) -> ToolGateway:
        return self._tools

    # ------------------------------------------------------------ 查询与清理（app 层、测试用）

    def sidebar(self, shop_id: str, conversation_id: str) -> list[SuggestionCard]:
        """当前展示在该会话侧边栏里的推荐卡片。"""
        with self._lock:
            conv = self._conversation(shop_id, conversation_id)
            return [c.model_copy(deep=True) for c in conv.cards] if conv else []

    def sent_messages(self, shop_id: str) -> list[OutgoingMessage]:
        with self._lock:
            return [m.model_copy() for m in self._outbox if m.shop_id == shop_id]

    def transfers(self, shop_id: str) -> list[TransferRecord]:
        with self._lock:
            return [t.model_copy() for t in self._transfers if t.shop_id == shop_id]

    def purge_shop(self, shop_id: str) -> int:
        """删除该店铺在适配器里的全部本地状态（§8.4 取消订购 / 授权过期），返回删除的条数
        （会话 + 发出的消息 + 转人工记录）。可以直接传给 AuthGuard.revoke。"""
        with self._lock:
            n = len(self._conversations.pop(shop_id, {}))
            self._seen.pop(shop_id, None)
            before = len(self._outbox) + len(self._transfers)
            self._outbox = [m for m in self._outbox if m.shop_id != shop_id]
            self._transfers = [t for t in self._transfers if t.shop_id != shop_id]
            return n + before - len(self._outbox) - len(self._transfers)

    # ------------------------------------------------------------ 内部

    def _conversation(self, shop_id: str, conversation_id: str) -> _Conversation | None:
        return self._conversations.get(shop_id, {}).get(conversation_id)

    def _touch_conversation(self, shop_id: str, conversation_id: str) -> _Conversation:
        convs = self._conversations.setdefault(shop_id, OrderedDict())
        conv = convs.get(conversation_id)
        if conv is None:
            conv = _Conversation(messages=deque(maxlen=self._context_limit))
            convs[conversation_id] = conv
            while len(convs) > self._max_conversations:
                convs.popitem(last=False)
        else:
            convs.move_to_end(conversation_id)
        return conv

    def _first_time(self, shop_id: str, msg_id: str) -> bool:
        seen = self._seen.get(shop_id)
        if seen is None:
            seen = self._seen[shop_id] = _LRU(self._dedup_capacity)
        return seen.add(msg_id)

    def _safe_extension(self, extension: Mapping[str, Any]) -> dict[str, Any]:
        """只保留白名单里的平台字段；取值只能是非空标量，字符串含个人信息时丢弃。"""
        out: dict[str, Any] = {}
        for key, value in extension.items():
            if key not in self.extension_fields or key in PII_FIELD_NAMES:
                continue
            if isinstance(value, bool | int | float):
                out[key] = value
            elif isinstance(value, str) and not contains_pii(value):
                out[key] = value
        return out

    def _mode_for(self, event: RawEvent, conv: _Conversation | None) -> Mode:
        """接入层只在模式 A、开了全自动、平台说明是机器人接待、会话没转过人工时标 AUTO；
        最终生效模式由引擎决定（ConversationContext.mode）。影子模式由店铺配置决定，不在这里。"""
        transferred = conv is not None and conv.transferred
        if self.access == "A" and self.auto_send_enabled and event.auto_requested:
            return Mode.ASSIST if transferred else Mode.AUTO
        return Mode.ASSIST

    def _accept_buyer(self, event: RawEvent) -> StandardMessage | None:
        raw_buyer_id = opt_str(event.raw_buyer_id)
        if raw_buyer_id is None:
            return None  # 没有买家 ID 无法生成 buyer_ref
        text = event.text.strip() if isinstance(event.text, str) else ""
        item = ItemRef(item_id=event.item_id, sku_id=event.sku_id) if event.item_id else None
        order = OrderRef(order_id=event.order_id) if event.order_id else None
        if not text and item is None and order is None:
            return None
        with self._lock:
            conv = self._conversation(event.shop_id, event.conversation_id)
            try:
                msg = StandardMessage(
                    platform=self.platform,
                    shop_id=event.shop_id,
                    conversation_id=event.conversation_id,
                    msg_id=event.msg_id,
                    buyer_ref=self._hasher(self.platform.value, raw_buyer_id),
                    text=text,
                    item_card=item,
                    order_card=order,
                    agent_id=event.agent_id,
                    mode=self._mode_for(event, conv),
                    received_at=event.at or self._clock.now(),
                    role=Role.BUYER,
                    extension=self._safe_extension(event.extension),
                )
            except ValidationError:
                return None
            if not self._first_time(event.shop_id, event.msg_id):
                return None
            conv = self._touch_conversation(event.shop_id, event.conversation_id)
            conv.messages.append(
                StoredMessage(
                    msg_id=msg.msg_id,
                    shop_id=msg.shop_id,
                    conversation_id=msg.conversation_id,
                    role=Role.BUYER,
                    text=msg.text,
                    item_id=item.item_id if item else None,
                    sku_id=item.sku_id if item else None,
                    order_ref=order.order_id if order else None,
                    created_at=msg.received_at,
                )
            )
            if item is not None:
                conv.item = item
            if order is not None:
                conv.order = order
        return msg

    def _accept_agent(self, event: RawEvent) -> None:
        text = event.text.strip() if isinstance(event.text, str) else ""
        if not text:
            return
        with self._lock:
            conv = self._conversation(event.shop_id, event.conversation_id)
            if conv is None or not self._first_time(event.shop_id, event.msg_id):
                return  # 没有买家消息的会话不记客服消息
            try:
                stored = StoredMessage(
                    msg_id=event.msg_id,
                    shop_id=event.shop_id,
                    conversation_id=event.conversation_id,
                    role=Role.AGENT,
                    text=text,
                    created_at=event.at or self._clock.now(),
                )
            except ValidationError:
                return
            conv.messages.append(stored)

    def _accept_recall(self, event: RawEvent) -> None:
        with self._lock:
            conv = self._conversation(event.shop_id, event.conversation_id)
            if conv is None or not event.recalled_msg_id:
                return
            if not self._first_time(event.shop_id, event.msg_id):
                return
            kept = [m for m in conv.messages if m.msg_id != event.recalled_msg_id]
            conv.messages = deque(kept, maxlen=self._context_limit)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """嵌套字段 → Mapping；没有时返回空 dict，类型不对抛 TypeError（on_message 当作无法识别）。"""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"期望对象，得到 {type(value).__name__}")
    return value
