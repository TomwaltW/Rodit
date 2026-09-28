"""会话理解（方案 §5.3）：合并连发、最近几轮、指代到商品 / SKU / 订单 → ConversationContext。

- 店铺：不存在或已停用（SUSPENDED）抛 ShopInactive（§8.4 停止处理）；生效配置用
  core.config.effective_shop_config（平台默认 → 类目默认 → 店铺覆盖，§9.3）。
- 生效模式：店铺开了影子模式 → SHADOW；消息是 AUTO 且店铺开了全自动 → AUTO；其余 ASSIST
  （引擎决定，不直接信任接入层，见 ConversationContext 文档）。
- 合并连发：同一角色、和上一条间隔不超过 conversation.merge_window_s 的消息合并成一个 Turn
  （按相邻两条的间隔判断，连续发的一串都会合并），文字用空格连接，Turn.at 取最后一条的时间。
  current 是末尾那个合并后的买家 Turn（一定包含本条消息）；history 保留最近
  generation.history_rounds 轮（一轮 = 一条买家 Turn 及其后的回复），不足时全部保留。
- 商品：本条消息带商品卡片 → 该商品（SKU 取卡片上的）；否则文字点名了会话里某一件
  （类目末级，如"牛仔裤"）→ 那件；否则买家用"这个 / 那件"之类指代、而会话里出现过两件及以上
  商品 → 先看有没有用规格值指代（"那件黑色的""L 码的那件"，只有一件候选有这个规格），
  有就用那件，否则置空（检索据此给澄清问题，§5.2.4）；"这个有 M 码吗"是在问尺码，
  不算用规格指代。其余情况取会话里最近出现的商品。候选 candidate_item_ids = 会话里出现过的商品，
  最近的在前。
- SKU：当前商品确定后，文字里提到的规格值（颜色 / 尺码……）唯一对上一个 SKU 就用它；
  对上多个时保留卡片上的 SKU（如果在其中），否则置空；没提规格值时沿用卡片上的 SKU。
- 订单：本条消息的订单卡片 → get_order；否则文字里写了订单号（最近订单里的 ID，或"订单号 xxx"
  且属于这个买家）→ 该订单；否则"刚买的那个 / 上次买的" → 最近订单的第一单；
  否则会话里最近出现的订单卡片。recent_orders = list_recent_orders(limit=5)。
  平台工具不可用（ToolUnavailable 等）时订单置空、recent_orders 为空，并报告降级 "tool:order"。
- unresolved_turns：从当前 Turn 往前数，连续表示"没听懂 / 答非所问"的买家 Turn 数（§5.7），
  中间客服的回复不打断计数。
- 指代识别全部用原文（订单号不能被脱敏掉）；ConversationContext 里的 Turn.text 也是原文，
  进提示词 / 判断层前由各步骤按约定脱敏。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from rodit.core.clock import Clock
from rodit.core.config import EngineConfig, effective_shop_config
from rodit.core.enums import Mode, Role, ShopStatus
from rodit.core.errors import (  # noqa: F401  (re-export)
    KnowledgeUnavailable,
    RoditError,
    ShopInactive,
)
from rodit.core.models import (
    ConversationContext,
    ItemCard,
    OrderInfo,
    Shop,
    ShopConfig,
    StandardMessage,
    StoredMessage,
    Turn,
)
from rodit.core.ports import ConversationRepository, KnowledgeRepository, ToolGateway
from rodit.engine.references import (
    expresses_confusion,
    find_known_order_id,
    has_demonstrative,
    hinted_order_ids,
    match_skus,
    names_item,
    refers_by_sku,
    refers_to_recent_order,
)

DEGRADED_TOOL_ORDER = "tool:order"
DEGRADED_STORAGE = "storage"
DEGRADED_KNOWLEDGE = "knowledge"

_MAX_ITEM_CANDIDATES = 5  # 消歧时最多读取的候选商品卡片数


class Understanding(BaseModel):
    """会话理解的结果：上下文 + 本步骤的降级标记（如 "tool:order"）。"""

    model_config = ConfigDict(extra="forbid")

    context: ConversationContext
    degraded: list[str] = Field(default_factory=list)


def effective_mode(msg_mode: Mode, config: ShopConfig) -> Mode:
    """本条消息的生效模式（ConversationContext.mode 的规则）。"""
    if config.shadow_mode:
        return Mode.SHADOW
    if msg_mode == Mode.AUTO and config.auto_reply_enabled:
        return Mode.AUTO
    return Mode.ASSIST


def merge_turns(messages: Sequence[StoredMessage], window_s: float) -> list[Turn]:
    """同一角色、与上一条间隔不超过 window_s 秒的连发消息合并成一个 Turn（§5.3）。"""
    turns: list[Turn] = []
    prev_at = None
    for m in messages:
        text = (m.text or "").strip()
        joinable = (
            turns
            and prev_at is not None
            and turns[-1].role == m.role
            and (m.created_at - prev_at).total_seconds() <= window_s
        )
        if joinable:
            last = turns[-1]
            merged = f"{last.text} {text}".strip() if text else last.text
            turns[-1] = last.model_copy(
                update={"text": merged, "at": m.created_at, "msg_ids": [*last.msg_ids, m.msg_id]}
            )
        else:
            turns.append(Turn(role=m.role, text=text, at=m.created_at, msg_ids=[m.msg_id]))
        prev_at = m.created_at
    return turns


def keep_rounds(turns: Sequence[Turn], rounds: int) -> list[Turn]:
    """保留最近 rounds 轮（一轮 = 一条买家 Turn 及其后的回复）；不足 rounds 轮时全部保留。

    与 core.states.recent_turns 的计法一致，所以后续按同样轮数截取不会再丢内容。
    """
    if rounds <= 0:
        return []
    buyer_seen = 0
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].role == Role.BUYER:
            buyer_seen += 1
            if buyer_seen >= rounds:
                return list(turns[i:])
    return list(turns)


def count_unresolved(turns: Sequence[Turn]) -> int:
    """从最后一个 Turn 往前，连续表示没听懂 / 答非所问的买家 Turn 数（§5.7）。"""
    count = 0
    for t in reversed(turns):
        if t.role != Role.BUYER:
            continue
        if not expresses_confusion(t.text):
            break
        count += 1
    return count


def _stored_from(msg: StandardMessage) -> StoredMessage:
    return StoredMessage(
        msg_id=msg.msg_id,
        shop_id=msg.shop_id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        text=msg.text,
        item_id=msg.item_card.item_id if msg.item_card else None,
        sku_id=msg.item_card.sku_id if msg.item_card else None,
        order_ref=msg.order_card.order_id if msg.order_card else None,
        created_at=msg.received_at,
    )


class _Notes:
    """一次 build 的降级标记（去重、保序）。"""

    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, code: str) -> None:
        if code not in self.items:
            self.items.append(code)


class ConversationBuilder:
    """会话理解（§5.3）：StandardMessage → ConversationContext。

    只读：不写会话、不写推荐（这些由引擎负责）；读会话记录、商品卡片、订单。
    """

    def __init__(
        self,
        knowledge: KnowledgeRepository,
        conversations: ConversationRepository,
        tools: ToolGateway,
        config: EngineConfig,
        clock: Clock,
        *,
        message_limit: int = 50,
        recent_order_limit: int = 5,
    ) -> None:
        self._knowledge = knowledge
        self._conversations = conversations
        self._tools = tools
        self._config = config
        self._clock = clock
        self._message_limit = message_limit
        self._recent_order_limit = recent_order_limit

    # ------------------------------------------------------------ 店铺与配置

    def load_shop(self, shop_id: str) -> Shop:
        """读店铺；不存在或已停用抛 ShopInactive。知识库故障时透传 KnowledgeUnavailable。"""
        shop = self._knowledge.get_shop(shop_id)
        if shop is None:
            raise ShopInactive(shop_id)
        if shop.status == ShopStatus.SUSPENDED:
            raise ShopInactive(shop_id, shop.status)
        return shop

    def shop_config(self, shop: Shop) -> ShopConfig:
        return effective_shop_config(self._config, shop)

    # ------------------------------------------------------------ 构建上下文

    def build(self, msg: StandardMessage, *, shop: Shop | None = None) -> ConversationContext:
        return self.build_with_notes(msg, shop=shop).context

    def build_with_notes(self, msg: StandardMessage, *, shop: Shop | None = None) -> Understanding:
        """构建上下文并返回降级标记。shop 为空时自己读（引擎已读过时传入，省一次查询）。"""
        if msg.role != Role.BUYER:
            raise ValueError("会话理解只处理买家消息")
        if shop is None:
            shop = self.load_shop(msg.shop_id)
        elif shop.id != msg.shop_id:
            raise ValueError("传入的店铺与消息的 shop_id 不一致")
        elif shop.status == ShopStatus.SUSPENDED:
            raise ShopInactive(shop.id, shop.status)
        notes = _Notes()
        config = self.shop_config(shop)

        messages = self._load_messages(msg, notes)
        turns = merge_turns(messages, self._config.conversation.merge_window_s)
        current = turns[-1]  # _load_messages 保证最后一条是本条买家消息
        history = keep_rounds(turns[:-1], self._config.generation.history_rounds)

        item, sku_id, candidates = self._resolve_item(msg, messages, current.text, notes)
        recent_orders = self._recent_orders(msg, notes)
        order = self._resolve_order(msg, messages, current.text, recent_orders, notes)

        ctx = ConversationContext(
            platform=shop.platform,
            shop=shop,
            config=config,
            conversation_id=msg.conversation_id,
            buyer_ref=msg.buyer_ref,
            mode=effective_mode(msg.mode, config),
            agent_id=msg.agent_id,
            current=current,
            history=history,
            item=item,
            sku_id=sku_id,
            candidate_item_ids=candidates,
            order=order,
            recent_orders=recent_orders,
            unresolved_turns=count_unresolved(turns),
            now=self._clock.now(),
        )
        return Understanding(context=ctx, degraded=notes.items)

    # ------------------------------------------------------------ 消息

    def _load_messages(self, msg: StandardMessage, notes: _Notes) -> list[StoredMessage]:
        """会话记录截到本条消息为止（本条之后写入的消息属于下一次处理）。

        本条消息不在记录里（会话存储故障，或调用方没先保存）时，取不晚于它的记录再补上本条。
        """
        try:
            stored = self._conversations.recent_messages(
                msg.shop_id, msg.conversation_id, limit=self._message_limit
            )
        except Exception:  # noqa: BLE001 — 会话存储故障不影响出推荐（§12.2）
            notes.add(DEGRADED_STORAGE)
            stored = []
        idx = next((i for i, m in enumerate(stored) if m.msg_id == msg.msg_id), None)
        if idx is not None:
            return list(stored[: idx + 1])
        earlier = [m for m in stored if m.created_at <= msg.received_at]
        return [*earlier, _stored_from(msg)]

    # ------------------------------------------------------------ 商品与 SKU

    def _card(
        self, shop_id: str, item_id: str, cache: dict[str, ItemCard | None], notes: _Notes
    ) -> ItemCard | None:
        if item_id not in cache:
            try:
                cache[item_id] = self._knowledge.get_item_card(shop_id, item_id)
            except KnowledgeUnavailable:
                notes.add(DEGRADED_KNOWLEDGE)
                cache[item_id] = None
        return cache[item_id]

    def _resolve_item(
        self,
        msg: StandardMessage,
        messages: Sequence[StoredMessage],
        text: str,
        notes: _Notes,
    ) -> tuple[ItemCard | None, str | None, list[str]]:
        candidates: list[str] = []
        for m in reversed(messages):
            if m.item_id and m.item_id not in candidates:
                candidates.append(m.item_id)
        cache: dict[str, ItemCard | None] = {}
        shop_id = msg.shop_id

        item_id: str | None
        if msg.item_card is not None:
            item_id = msg.item_card.item_id
            base_sku = msg.item_card.sku_id
        else:
            item_id = self._pick_item(shop_id, candidates, text, cache, notes)
            base_sku = next(
                (m.sku_id for m in reversed(messages) if item_id and m.item_id == item_id),
                None,
            )

        if item_id is None:
            return None, None, candidates
        item = self._card(shop_id, item_id, cache, notes)
        sku_id = base_sku
        if item is not None:
            matched = match_skus(item, text)
            if matched is not None:
                ids = [s.sku_id for s in matched]
                if len(ids) == 1:
                    sku_id = ids[0]
                elif base_sku not in ids:
                    sku_id = None
        return item, sku_id, candidates

    def _pick_item(
        self,
        shop_id: str,
        candidates: list[str],
        text: str,
        cache: dict[str, ItemCard | None],
        notes: _Notes,
    ) -> str | None:
        """本条消息没有商品卡片时，决定当前商品（见模块文档）。"""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        cards = [
            (cid, self._card(shop_id, cid, cache, notes))
            for cid in candidates[:_MAX_ITEM_CANDIDATES]
        ]
        named = [cid for cid, card in cards if card is not None and names_item(card, text)]
        if len(named) == 1:
            return named[0]
        if not has_demonstrative(text):
            return candidates[0]
        by_sku = [cid for cid, card in cards if card is not None and refers_by_sku(card, text)]
        if len(by_sku) == 1:
            return by_sku[0]
        return None  # 指代不明，交给检索给澄清问题（§5.2.4）

    # ------------------------------------------------------------ 订单

    def _recent_orders(self, msg: StandardMessage, notes: _Notes) -> list[OrderInfo]:
        try:
            orders = self._tools.list_recent_orders(
                msg.shop_id, msg.buyer_ref, limit=self._recent_order_limit
            )
        except Exception:  # noqa: BLE001 — ToolUnavailable 及其他故障一律降级（§12.2）
            notes.add(DEGRADED_TOOL_ORDER)
            return []
        # 防御：只保留本店、本买家的订单（§9.2）
        return [o for o in orders if o.shop_id == msg.shop_id and o.buyer_ref == msg.buyer_ref]

    def _get_order(
        self,
        msg: StandardMessage,
        order_id: str,
        recent: Sequence[OrderInfo],
        notes: _Notes,
        *,
        require_buyer: bool,
    ) -> OrderInfo | None:
        hit = next((o for o in recent if o.order_id == order_id), None)
        if hit is not None:
            return hit
        try:
            order = self._tools.get_order(msg.shop_id, order_id)
        except Exception:  # noqa: BLE001
            notes.add(DEGRADED_TOOL_ORDER)
            return None
        if order is None or order.shop_id != msg.shop_id:
            return None
        if require_buyer and order.buyer_ref != msg.buyer_ref:
            return None  # 买家打出来的订单号不是自己的：不查别人的订单
        return order

    def _resolve_order(
        self,
        msg: StandardMessage,
        messages: Sequence[StoredMessage],
        text: str,
        recent: list[OrderInfo],
        notes: _Notes,
    ) -> OrderInfo | None:
        if msg.order_card is not None:
            return self._get_order(msg, msg.order_card.order_id, recent, notes, require_buyer=False)
        known = find_known_order_id(text, (o.order_id for o in recent))
        if known is not None:
            return next(o for o in recent if o.order_id == known)
        for oid in hinted_order_ids(text)[:1]:  # 最多查一次，控制平台接口额度
            order = self._get_order(msg, oid, recent, notes, require_buyer=True)
            if order is not None:
                return order
        if refers_to_recent_order(text) and recent:
            return recent[0]
        ref = next((m.order_ref for m in reversed(messages) if m.order_ref), None)
        if ref is not None:
            return self._get_order(msg, ref, recent, notes, require_buyer=False)
        return None
