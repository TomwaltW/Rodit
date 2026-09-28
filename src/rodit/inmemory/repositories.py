"""仓储协议的内存实现（core.ports 第 2 组）。

用于单元测试、本地演示和模式 C 单进程部署；语义与 PostgreSQL 实现一致，两者都要通过
rodit.testing.contracts 的契约测试。所有读写按 shop_id 隔离（§9.2），返回深拷贝，
调用方改返回值不会影响存储。线程安全（一把锁）。
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Sequence
from datetime import datetime

from rodit.core.clock import Clock, SystemClock
from rodit.core.enums import KnowledgeStatus, ReviewState, Role, SuggestionStatus
from rodit.core.errors import ConcurrencyConflict
from rodit.core.models import (
    AfterSaleReview,
    Conversation,
    ItemCard,
    KnowledgeGap,
    KnowledgeItem,
    LabeledSample,
    PolicyEntry,
    Shop,
    ShopAuth,
    StandardMessage,
    StoredMessage,
    SuggestionRecord,
)
from rodit.inmemory.embedding import cosine


class InMemoryKnowledgeRepository:
    """实现 core.ports.KnowledgeRepository。"""

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._shops: dict[str, Shop] = {}
        self._policies: dict[str, dict[str, PolicyEntry]] = {}
        self._items: dict[tuple[str, str], ItemCard] = {}
        self._knowledge: dict[int, KnowledgeItem] = {}
        self._ids = itertools.count(1)

    # ---- 店铺
    def get_shop(self, shop_id: str) -> Shop | None:
        with self._lock:
            shop = self._shops.get(shop_id)
            return shop.model_copy(deep=True) if shop else None

    def upsert_shop(self, shop: Shop) -> None:
        with self._lock:
            self._shops[shop.id] = shop.model_copy(deep=True)

    # ---- 政策
    def get_policies(self, shop_id: str) -> dict[str, PolicyEntry]:
        with self._lock:
            return {k: v.model_copy() for k, v in self._policies.get(shop_id, {}).items()}

    def upsert_policy(self, entry: PolicyEntry) -> None:
        with self._lock:
            self._policies.setdefault(entry.shop_id, {})[entry.key] = entry.model_copy()

    # ---- 商品
    def get_item_card(self, shop_id: str, item_id: str) -> ItemCard | None:
        with self._lock:
            card = self._items.get((shop_id, item_id))
            return card.model_copy(deep=True) if card else None

    def list_item_cards(self, shop_id: str) -> list[ItemCard]:
        with self._lock:
            cards = [c for (s, _), c in self._items.items() if s == shop_id]
            return [c.model_copy(deep=True) for c in sorted(cards, key=lambda c: c.item_id)]

    def upsert_item_card(self, card: ItemCard) -> None:
        with self._lock:
            self._items[(card.shop_id, card.item_id)] = card.model_copy(deep=True)

    # ---- FAQ
    def add_knowledge(self, item: KnowledgeItem) -> KnowledgeItem:
        with self._lock:
            new_id = next(self._ids)
            saved = item.model_copy(
                deep=True,
                update={"id": new_id, "created_at": item.created_at or self._clock.now()},
            )
            self._knowledge[new_id] = saved
            return saved.model_copy(deep=True)

    def get_knowledge(self, shop_id: str, knowledge_id: int) -> KnowledgeItem | None:
        with self._lock:
            item = self._knowledge.get(knowledge_id)
            if item is None or item.shop_id != shop_id:
                return None
            return item.model_copy(deep=True)

    def _visible(self, item: KnowledgeItem, shop_id: str, status: KnowledgeStatus | None) -> bool:
        return item.shop_id == shop_id and (status is None or item.status == status)

    def list_knowledge(
        self,
        shop_id: str,
        *,
        status: KnowledgeStatus | None = KnowledgeStatus.APPROVED,
        item_id: str | None = None,
    ) -> list[KnowledgeItem]:
        with self._lock:
            out = [
                k
                for k in self._knowledge.values()
                if self._visible(k, shop_id, status)
                and (item_id is None or k.item_id is None or k.item_id == item_id)
            ]
            return [k.model_copy(deep=True) for k in sorted(out, key=lambda k: k.id or 0)]

    def search_knowledge(
        self,
        shop_id: str,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        item_id: str | None = None,
        status: KnowledgeStatus = KnowledgeStatus.APPROVED,
    ) -> list[tuple[KnowledgeItem, float]]:
        with self._lock:
            scored: list[tuple[KnowledgeItem, float]] = []
            for k in self._knowledge.values():
                if not self._visible(k, shop_id, status) or k.embedding is None:
                    continue
                if k.item_id is not None and k.item_id != item_id:
                    continue
                score = max(0.0, min(1.0, cosine(embedding, k.embedding)))
                scored.append((k, score))
            scored.sort(key=lambda pair: (-pair[1], pair[0].id or 0))
            return [(k.model_copy(deep=True), s) for k, s in scored[:top_k]]

    def set_knowledge_status(
        self,
        shop_id: str,
        knowledge_id: int,
        status: KnowledgeStatus,
        *,
        approved_by: str | None = None,
        expected_status: KnowledgeStatus | None = None,
    ) -> KnowledgeItem:
        with self._lock:
            item = self._knowledge.get(knowledge_id)
            if item is None or item.shop_id != shop_id:
                raise KeyError(knowledge_id)
            if expected_status is not None and item.status != expected_status:
                raise ConcurrencyConflict(
                    f"知识 {knowledge_id} 的状态已是 {item.status.value}，"
                    f"不是 {expected_status.value}"
                )
            update: dict[str, object] = {"status": status}
            if approved_by is not None:
                update["approved_by"] = approved_by
            saved = item.model_copy(update=update)
            self._knowledge[knowledge_id] = saved
            return saved.model_copy(deep=True)

    def set_knowledge_embedding(
        self, shop_id: str, knowledge_id: int, embedding: list[float]
    ) -> None:
        with self._lock:
            item = self._knowledge.get(knowledge_id)
            if item is None or item.shop_id != shop_id:
                raise KeyError(knowledge_id)
            self._knowledge[knowledge_id] = item.model_copy(update={"embedding": list(embedding)})

    def purge_shop(self, shop_id: str) -> int:
        with self._lock:
            n = 0
            if self._shops.pop(shop_id, None) is not None:
                n += 1
            n += len(self._policies.pop(shop_id, {}))
            for key in [k for k in self._items if k[0] == shop_id]:
                del self._items[key]
                n += 1
            for kid in [k for k, v in self._knowledge.items() if v.shop_id == shop_id]:
                del self._knowledge[kid]
                n += 1
            return n


class InMemoryConversationRepository:
    """实现 core.ports.ConversationRepository。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conversations: dict[tuple[str, str], Conversation] = {}
        self._messages: dict[tuple[str, str], list[StoredMessage]] = {}
        self._msg_ids: set[tuple[str, str]] = set()

    def _ensure_conversation(
        self, shop_id: str, conversation_id: str, buyer_ref: str, at: datetime
    ):
        key = (shop_id, conversation_id)
        if key not in self._conversations:
            self._conversations[key] = Conversation(
                id=conversation_id, shop_id=shop_id, buyer_ref=buyer_ref, started_at=at
            )
            self._messages[key] = []

    def _append(self, key: tuple[str, str], msg: StoredMessage) -> None:
        msgs = self._messages[key]
        msgs.append(msg)
        # 按时间升序；同一时间保持写入顺序（sort 是稳定的）
        msgs.sort(key=lambda m: m.created_at)

    def add_message(self, msg: StandardMessage) -> bool:
        with self._lock:
            if (msg.shop_id, msg.msg_id) in self._msg_ids:
                return False
            self._msg_ids.add((msg.shop_id, msg.msg_id))
            self._ensure_conversation(
                msg.shop_id, msg.conversation_id, msg.buyer_ref, msg.received_at
            )
            self._append(
                (msg.shop_id, msg.conversation_id),
                StoredMessage(
                    msg_id=msg.msg_id,
                    shop_id=msg.shop_id,
                    conversation_id=msg.conversation_id,
                    role=msg.role,
                    text=msg.text,
                    item_id=msg.item_card.item_id if msg.item_card else None,
                    sku_id=msg.item_card.sku_id if msg.item_card else None,
                    order_ref=msg.order_card.order_id if msg.order_card else None,
                    created_at=msg.received_at,
                ),
            )
            return True

    def add_outgoing(
        self,
        *,
        shop_id: str,
        conversation_id: str,
        msg_id: str,
        role: Role,
        text: str,
        at: datetime,
    ) -> bool:
        with self._lock:
            if (shop_id, msg_id) in self._msg_ids:
                return False
            key = (shop_id, conversation_id)
            if key not in self._conversations:
                return False  # 没有买家消息的会话不接受客服消息
            self._msg_ids.add((shop_id, msg_id))
            self._append(
                key,
                StoredMessage(
                    msg_id=msg_id,
                    shop_id=shop_id,
                    conversation_id=conversation_id,
                    role=role,
                    text=text,
                    created_at=at,
                ),
            )
            return True

    def get_conversation(self, shop_id: str, conversation_id: str) -> Conversation | None:
        with self._lock:
            conv = self._conversations.get((shop_id, conversation_id))
            return conv.model_copy() if conv else None

    def recent_messages(
        self, shop_id: str, conversation_id: str, *, limit: int = 20
    ) -> list[StoredMessage]:
        with self._lock:
            msgs = self._messages.get((shop_id, conversation_id), [])
            return [m.model_copy() for m in msgs[-limit:]] if limit > 0 else []

    def purge_shop(self, shop_id: str) -> int:
        with self._lock:
            n = 0
            for key in [k for k in self._conversations if k[0] == shop_id]:
                n += 1 + len(self._messages.pop(key, []))
                del self._conversations[key]
            self._msg_ids = {k for k in self._msg_ids if k[0] != shop_id}
            return n


class InMemorySuggestionRepository:
    """实现 core.ports.SuggestionRepository。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[int, SuggestionRecord] = {}
        self._ids = itertools.count(1)

    def add(self, record: SuggestionRecord) -> SuggestionRecord:
        with self._lock:
            new_id = next(self._ids)
            saved = record.model_copy(deep=True, update={"id": new_id})
            self._records[new_id] = saved
            return saved.model_copy(deep=True)

    def get(self, shop_id: str, suggestion_id: int) -> SuggestionRecord | None:
        with self._lock:
            rec = self._records.get(suggestion_id)
            if rec is None or rec.shop_id != shop_id:
                return None
            return rec.model_copy(deep=True)

    def update(
        self, record: SuggestionRecord, *, expected_status: SuggestionStatus | None = None
    ) -> SuggestionRecord:
        with self._lock:
            cur = self._records.get(record.id) if record.id is not None else None
            if cur is None or cur.shop_id != record.shop_id:
                raise KeyError(record.id)
            if expected_status is not None and cur.status != expected_status:
                raise ConcurrencyConflict(
                    f"推荐 {record.id} 的状态已是 {cur.status.value}，不是 {expected_status.value}"
                )
            self._records[record.id] = record.model_copy(deep=True)
            return record.model_copy(deep=True)

    def list(
        self,
        shop_id: str,
        *,
        conversation_id: str | None = None,
        status: SuggestionStatus | None = None,
        since: datetime | None = None,
    ) -> list[SuggestionRecord]:
        with self._lock:
            out = [
                r
                for r in self._records.values()
                if r.shop_id == shop_id
                and (conversation_id is None or r.conversation_id == conversation_id)
                and (status is None or r.status == status)
                and (since is None or r.created_at >= since)
            ]
            out.sort(key=lambda r: (r.created_at, r.id or 0))
            return [r.model_copy(deep=True) for r in out]

    def purge_shop(self, shop_id: str) -> int:
        with self._lock:
            ids = [i for i, r in self._records.items() if r.shop_id == shop_id]
            for i in ids:
                del self._records[i]
            return len(ids)


class InMemoryFeedbackRepository:
    """实现 core.ports.FeedbackRepository。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gaps: list[KnowledgeGap] = []
        self._labels: list[LabeledSample] = []

    def add_gap(self, gap: KnowledgeGap) -> None:
        with self._lock:
            self._gaps.append(gap.model_copy())

    def list_gaps(self, shop_id: str) -> list[KnowledgeGap]:
        with self._lock:
            out = [g for g in self._gaps if g.shop_id == shop_id]
            return [g.model_copy() for g in sorted(out, key=lambda g: g.created_at)]

    def add_label(self, sample: LabeledSample) -> None:
        with self._lock:
            self._labels.append(sample.model_copy())

    def list_labels(self, shop_id: str) -> list[LabeledSample]:
        with self._lock:
            out = [s for s in self._labels if s.shop_id == shop_id]
            return [s.model_copy() for s in sorted(out, key=lambda s: s.created_at)]

    def export_training_labels(self) -> list[LabeledSample]:
        with self._lock:
            return [s.model_copy() for s in sorted(self._labels, key=lambda s: s.created_at)]

    def purge_shop(self, shop_id: str) -> int:
        with self._lock:
            before = len(self._gaps) + len(self._labels)
            self._gaps = [g for g in self._gaps if g.shop_id != shop_id]
            self._labels = [s for s in self._labels if s.shop_id != shop_id]
            return before - len(self._gaps) - len(self._labels)


class InMemoryAfterSaleReviewRepository:
    """实现 core.ports.AfterSaleReviewRepository（乐观锁）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reviews: dict[tuple[str, str], AfterSaleReview] = {}

    def get(self, shop_id: str, aftersale_id: str) -> AfterSaleReview | None:
        with self._lock:
            r = self._reviews.get((shop_id, aftersale_id))
            return r.model_copy(deep=True) if r else None

    def save(self, review: AfterSaleReview, *, expected_version: int | None) -> AfterSaleReview:
        key = (review.shop_id, review.aftersale_id)
        with self._lock:
            cur = self._reviews.get(key)
            if expected_version is None:
                if cur is not None:
                    raise ConcurrencyConflict(f"售后审核 {review.aftersale_id} 已存在")
                new_version = 1
            else:
                if cur is None or cur.version != expected_version:
                    raise ConcurrencyConflict(
                        f"售后审核 {review.aftersale_id} 版本冲突："
                        f"期望 {expected_version}，实际 {cur.version if cur else '不存在'}"
                    )
                new_version = expected_version + 1
            saved = review.model_copy(deep=True, update={"version": new_version})
            self._reviews[key] = saved
            return saved.model_copy(deep=True)

    def list_by_state(
        self, shop_id: str, states: Sequence[ReviewState] | None = None
    ) -> list[AfterSaleReview]:
        with self._lock:
            out = [
                r
                for (s, _), r in self._reviews.items()
                if s == shop_id and (states is None or r.state in states)
            ]
            return [r.model_copy(deep=True) for r in sorted(out, key=lambda r: r.aftersale_id)]

    def purge_shop(self, shop_id: str) -> int:
        with self._lock:
            keys = [k for k in self._reviews if k[0] == shop_id]
            for k in keys:
                del self._reviews[k]
            return len(keys)


class InMemoryShopAuthRepository:
    """实现 core.ports.ShopAuthRepository（内存里不加密；PostgreSQL 实现加密落库）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._auth: dict[str, ShopAuth] = {}

    def get(self, shop_id: str) -> ShopAuth | None:
        with self._lock:
            a = self._auth.get(shop_id)
            return a.model_copy() if a else None

    def save(self, auth: ShopAuth) -> None:
        with self._lock:
            self._auth[auth.shop_id] = auth.model_copy()

    def delete(self, shop_id: str) -> None:
        with self._lock:
            self._auth.pop(shop_id, None)
