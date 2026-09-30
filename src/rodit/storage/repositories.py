"""仓储协议的 PostgreSQL 实现（core.ports 第 2 组 + AuditLog；方案 §9.2、§10）。

与内存实现（rodit.inmemory）语义一致，共用 rodit.testing.contracts 的契约测试。约定：
- 每个方法开一个 Database.session(shop_id)：行级安全按店铺限定，SQL 里也显式写
  WHERE shop_id = %s（两道防线，§9.2）。
- 数据进出都是 core.models 的 pydantic 模型；读出的时间统一转成东八区（core.clock.CST）。
- JSONB 列（card / config / qa_scores / answers / detail / action_result）按 pydantic 的
  JSON 模式序列化：Decimal 存成字符串、datetime 存成 ISO 字符串、枚举存取值。
  类型明确的字段（如 ShopConfig.aftersale_amount_limit）读回后由模型校验还原成 Decimal；
  dict[str, Any] 里的值读回就是 JSON 类型。
- 排序与内存实现一致：文本排序用 COLLATE "C"（按码点，与 Python 的 sorted 相同），
  同一时间按自增 id（写入顺序）。
- 知识库故障（连不上、语句超时）抛 core.errors.KnowledgeUnavailable（§12.2 知识库故障 → 不推荐）；
  其他仓储把数据库异常原样抛出（StorageUnavailable / psycopg.Error），由调用方降级。
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

import psycopg
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from psycopg.types.json import Jsonb
from pydantic import SecretStr, TypeAdapter

from rodit.core.clock import CST, Clock, SystemClock
from rodit.core.enums import (
    DecisionStage,
    KnowledgeStatus,
    QualityReviewStatus,
    ReviewState,
    Role,
    SuggestionStatus,
)
from rodit.core.errors import (
    ConcurrencyConflict,
    KnowledgeUnavailable,
    TokenDecryptionError,
)
from rodit.core.models import (
    AfterSaleReview,
    AuditEvent,
    Conversation,
    DecisionLogEntry,
    ItemCard,
    KnowledgeGap,
    KnowledgeItem,
    LabeledSample,
    PolicyEntry,
    QualityReview,
    QualityScores,
    Shop,
    ShopAuth,
    StandardMessage,
    StoredMessage,
    SuggestionRecord,
    merge_overrides,
    strip_control_chars,
)
from rodit.core.ports import RECALL_TOMBSTONE_TTL
from rodit.inmemory.embedding import cosine
from rodit.storage.db import Conn, Database, StorageUnavailable

Row = dict[str, Any]

_JSON = TypeAdapter(Any)


def _dumps(obj: Any) -> str:
    return json.dumps(_JSON.dump_python(obj, mode="json"), ensure_ascii=False)


def _jsonb(obj: Any) -> Jsonb | None:
    return None if obj is None else Jsonb(obj, dumps=_dumps)


def _cst(dt: datetime | None) -> datetime | None:
    return dt.astimezone(CST) if dt is not None else None


# TokenDecryptionError 定义在 core.errors（授权检查要认它），从这里导入的旧写法照常可用


# ================================================================ 知识（§5.2）

_K_COLS = (
    "id, shop_id, question, answer, source, status, item_id, intent, approved_by, "
    "created_at, embedding"
)
_CARD_COLUMNS = ("shop_id", "item_id", "updated_at")  # 在列上，card JSONB 里不重复存


def _knowledge(r: Row) -> KnowledgeItem:
    return KnowledgeItem(
        id=r["id"],
        shop_id=r["shop_id"],
        question=r["question"],
        answer=r["answer"],
        source=r["source"],
        status=r["status"],
        item_id=r["item_id"],
        intent=r["intent"],
        approved_by=r["approved_by"],
        created_at=_cst(r["created_at"]),
        embedding=list(r["embedding"]) if r["embedding"] is not None else None,
    )


def _shop(r: Row) -> Shop:
    return Shop(
        id=r["id"],
        platform=r["platform"],
        category=r["category"],
        status=r["status"],
        config_overrides=r["config"],
    )


def _item_card(r: Row) -> ItemCard:
    return ItemCard.model_validate(
        {
            **r["card"],
            "shop_id": r["shop_id"],
            "item_id": r["item_id"],
            "updated_at": _cst(r["updated_at"]),
        }
    )


class PgKnowledgeRepository:
    """实现 core.ports.KnowledgeRepository（shop / shop_policy / item_card / knowledge_item）。

    向量检索：取出本店、符合状态和商品范围、有 embedding 的候选，在应用内用
    rodit.inmemory.cosine 算余弦（单店 FAQ 量小，百条级）。平台 RDS 支持 pgvector 后的升级路径：
    embedding 列改成 VECTOR(维度)、建 HNSW 索引（vector_cosine_ops），查询改成
    ORDER BY embedding <=> %s, id LIMIT top_k，分数 = 1 − 距离后截断到 [0, 1]；接口不变。
    """

    def __init__(self, db: Database, *, clock: Clock | None = None) -> None:
        self._db = db
        self._clock = clock or SystemClock()

    @contextmanager
    def _session(self, shop_id: str) -> Iterator[Conn]:
        try:
            with self._db.session(shop_id) as conn:
                yield conn
        except (StorageUnavailable, psycopg.OperationalError) as e:
            raise KnowledgeUnavailable("知识库暂时不可用") from e

    # ---- 店铺
    def get_shop(self, shop_id: str) -> Shop | None:
        with self._session(shop_id) as conn:
            r = conn.execute(
                "SELECT id, platform, category, status, config FROM shop WHERE id = %s",
                (shop_id,),
            ).fetchone()
        return _shop(r) if r is not None else None

    def upsert_shop(self, shop: Shop) -> None:
        with self._session(shop.id) as conn:
            conn.execute(
                "INSERT INTO shop (id, platform, category, status, config)"
                " VALUES (%s, %s, %s, %s, %s)"
                " ON CONFLICT (id) DO UPDATE SET platform = EXCLUDED.platform,"
                " category = EXCLUDED.category, status = EXCLUDED.status,"
                " config = EXCLUDED.config",
                (shop.id, shop.platform, shop.category, shop.status, _jsonb(shop.config_overrides)),
            )

    def update_shop_config(self, shop_id: str, changes: Mapping[str, Any]) -> Shop | None:
        with self._session(shop_id) as conn:
            r = conn.execute(
                "SELECT id, platform, category, status, config FROM shop WHERE id = %s FOR UPDATE",
                (shop_id,),
            ).fetchone()
            if r is None:
                return None
            shop = _shop(r)
            updated = Shop.model_validate(
                {**shop.model_dump(), "config_overrides": merge_overrides(shop, changes)}
            )
            row = conn.execute(
                "UPDATE shop SET config = %s WHERE id = %s"
                " RETURNING id, platform, category, status, config",
                (_jsonb(updated.config_overrides), shop_id),
            ).fetchone()
        assert row is not None
        return _shop(row)

    # ---- 政策
    def get_policies(self, shop_id: str) -> dict[str, PolicyEntry]:
        with self._session(shop_id) as conn:
            rows = conn.execute(
                "SELECT shop_id, key, value, updated_at FROM shop_policy WHERE shop_id = %s"
                ' ORDER BY key COLLATE "C"',
                (shop_id,),
            ).fetchall()
        return {
            r["key"]: PolicyEntry(
                shop_id=r["shop_id"],
                key=r["key"],
                value=r["value"],
                updated_at=_cst(r["updated_at"]),
            )
            for r in rows
        }

    def upsert_policy(self, entry: PolicyEntry) -> None:
        with self._session(entry.shop_id) as conn:
            conn.execute(
                "INSERT INTO shop_policy (shop_id, key, value, updated_at)"
                " VALUES (%s, %s, %s, %s)"
                " ON CONFLICT (shop_id, key) DO UPDATE SET value = EXCLUDED.value,"
                " updated_at = EXCLUDED.updated_at",
                (entry.shop_id, entry.key, entry.value, entry.updated_at),
            )

    def delete_policy(self, shop_id: str, key: str) -> bool:
        with self._session(shop_id) as conn:
            deleted = conn.execute(
                "DELETE FROM shop_policy WHERE shop_id = %s AND key = %s", (shop_id, key)
            ).rowcount
        return deleted > 0

    # ---- 商品
    def get_item_card(self, shop_id: str, item_id: str) -> ItemCard | None:
        with self._session(shop_id) as conn:
            r = conn.execute(
                "SELECT shop_id, item_id, card, updated_at FROM item_card"
                " WHERE shop_id = %s AND item_id = %s",
                (shop_id, item_id),
            ).fetchone()
        return _item_card(r) if r else None

    def list_item_cards(self, shop_id: str) -> list[ItemCard]:
        with self._session(shop_id) as conn:
            rows = conn.execute(
                "SELECT shop_id, item_id, card, updated_at FROM item_card WHERE shop_id = %s"
                ' ORDER BY item_id COLLATE "C"',
                (shop_id,),
            ).fetchall()
        return [_item_card(r) for r in rows]

    def upsert_item_card(self, card: ItemCard) -> None:
        body = card.model_dump(mode="json", exclude=set(_CARD_COLUMNS))
        with self._session(card.shop_id) as conn:
            conn.execute(
                "INSERT INTO item_card (shop_id, item_id, card, updated_at)"
                " VALUES (%s, %s, %s, %s)"
                " ON CONFLICT (shop_id, item_id) DO UPDATE SET card = EXCLUDED.card,"
                " updated_at = EXCLUDED.updated_at",
                (card.shop_id, card.item_id, _jsonb(body), card.updated_at),
            )

    def update_item_card(
        self, shop_id: str, item_id: str, changes: Mapping[str, Any]
    ) -> ItemCard | None:
        with self._session(shop_id) as conn:
            r = conn.execute(
                "SELECT shop_id, item_id, card, updated_at FROM item_card"
                " WHERE shop_id = %s AND item_id = %s FOR UPDATE",
                (shop_id, item_id),
            ).fetchone()
            if r is None:
                return None
            updated = _item_card(r).patched(changes)
            body = updated.model_dump(mode="json", exclude=set(_CARD_COLUMNS))
            row = conn.execute(
                "UPDATE item_card SET card = %s, updated_at = %s"
                " WHERE shop_id = %s AND item_id = %s"
                " RETURNING shop_id, item_id, card, updated_at",
                (_jsonb(body), updated.updated_at, shop_id, item_id),
            ).fetchone()
        assert row is not None
        return _item_card(row)

    # ---- FAQ
    def add_knowledge(self, item: KnowledgeItem) -> KnowledgeItem:
        # 同样的候选已在等审核（部分唯一索引 knowledge_item_pending_candidate_uniq，迁移版本 3）：
        # ON CONFLICT 等并发写入的事务提交后再判断，不写入也不中断事务，转成 ConcurrencyConflict
        created_at = item.created_at or self._clock.now()
        with self._session(item.shop_id) as conn:
            r = conn.execute(
                "INSERT INTO knowledge_item (shop_id, question, answer, source, status, item_id,"
                " intent, approved_by, created_at, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                f" ON CONFLICT DO NOTHING RETURNING {_K_COLS}",
                (
                    item.shop_id,
                    item.question,
                    item.answer,
                    item.source,
                    item.status,
                    item.item_id,
                    item.intent,
                    item.approved_by,
                    created_at,
                    list(item.embedding) if item.embedding is not None else None,
                ),
            ).fetchone()
        if r is None:
            raise ConcurrencyConflict("同样的候选知识已在等商家审核")
        return _knowledge(r)

    def get_knowledge(self, shop_id: str, knowledge_id: int) -> KnowledgeItem | None:
        with self._session(shop_id) as conn:
            r = conn.execute(
                f"SELECT {_K_COLS} FROM knowledge_item WHERE shop_id = %s AND id = %s",
                (shop_id, knowledge_id),
            ).fetchone()
        return _knowledge(r) if r else None

    def list_knowledge(
        self,
        shop_id: str,
        *,
        status: KnowledgeStatus | None = KnowledgeStatus.APPROVED,
        item_id: str | None = None,
    ) -> list[KnowledgeItem]:
        conds, params = ["shop_id = %s"], [shop_id]
        if status is not None:
            conds.append("status = %s")
            params.append(status)
        if item_id is not None:  # 通用 FAQ + 该商品的 FAQ
            conds.append("(item_id IS NULL OR item_id = %s)")
            params.append(item_id)
        with self._session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_K_COLS} FROM knowledge_item WHERE {' AND '.join(conds)} ORDER BY id",
                params,
            ).fetchall()
        return [_knowledge(r) for r in rows]

    def search_knowledge(
        self,
        shop_id: str,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        item_id: str | None = None,
        status: KnowledgeStatus = KnowledgeStatus.APPROVED,
    ) -> list[tuple[KnowledgeItem, float]]:
        # item_id=None 时 "item_id = NULL" 不成立，只剩通用 FAQ
        with self._session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_K_COLS} FROM knowledge_item"
                " WHERE shop_id = %s AND status = %s AND embedding IS NOT NULL"
                " AND (item_id IS NULL OR item_id = %s)",
                (shop_id, status, item_id),
            ).fetchall()
        scored = [
            (item, max(0.0, min(1.0, cosine(embedding, item.embedding or []))))
            for item in map(_knowledge, rows)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].id or 0))
        return scored[:top_k] if top_k > 0 else []

    def set_knowledge_status(
        self,
        shop_id: str,
        knowledge_id: int,
        status: KnowledgeStatus,
        *,
        approved_by: str | None = None,
        expected_status: KnowledgeStatus | None = None,
    ) -> KnowledgeItem:
        # 比较后更新：WHERE 里带上期望状态。并发的另一个事务先提交时，READ COMMITTED 下
        # 这条 UPDATE 会按新行重新判断条件，状态已变就更新 0 行
        cond, params = "shop_id = %s AND id = %s", [shop_id, knowledge_id]
        if expected_status is not None:
            cond += " AND status = %s"
            params.append(expected_status)
        try:
            with self._session(shop_id) as conn:
                r = conn.execute(
                    "UPDATE knowledge_item SET status = %s, approved_by = COALESCE(%s, approved_by)"
                    f" WHERE {cond} RETURNING {_K_COLS}",
                    [status, approved_by, *params],
                ).fetchone()
                if r is None and expected_status is not None:
                    cur = conn.execute(
                        "SELECT status FROM knowledge_item WHERE shop_id = %s AND id = %s",
                        (shop_id, knowledge_id),
                    ).fetchone()
                    if cur is not None:
                        raise ConcurrencyConflict(
                            f"知识 {knowledge_id} 的状态已是 {cur['status']}，"
                            f"不是 {KnowledgeStatus(expected_status).value}"
                        )
        except psycopg.errors.UniqueViolation as e:
            # 改回候选时和等审核的同样候选重复（部分唯一索引）：不写入
            raise ConcurrencyConflict(f"知识 {knowledge_id} 和等审核的候选知识重复") from e
        if r is None:
            raise KeyError(knowledge_id)
        return _knowledge(r)

    def set_knowledge_embedding(
        self, shop_id: str, knowledge_id: int, embedding: list[float]
    ) -> None:
        with self._session(shop_id) as conn:
            cur = conn.execute(
                "UPDATE knowledge_item SET embedding = %s WHERE shop_id = %s AND id = %s",
                (list(embedding), shop_id, knowledge_id),
            )
            if cur.rowcount == 0:
                raise KeyError(knowledge_id)

    def purge_shop(self, shop_id: str) -> int:
        # §8.4 分批删、每批一个短事务（Database.purge_rows）；店铺本身最后删
        try:
            n = sum(
                self._db.purge_rows(shop_id, table)
                for table in ("shop_policy", "item_card", "knowledge_item")
            )
            return n + self._db.purge_rows(shop_id, "shop", column="id")
        except (StorageUnavailable, psycopg.OperationalError) as e:
            raise KnowledgeUnavailable("知识库暂时不可用") from e


# ================================================================ 会话（§5.3、§8.3）

_M_COLS = "msg_id, shop_id, conversation_id, role, text, item_id, sku_id, order_ref, created_at"


def _message(r: Row) -> StoredMessage:
    return StoredMessage(**{**r, "created_at": _cst(r["created_at"])})


_TOMBSTONE_CLEANUP_ROWS = 100
"""写撤回墓碑时顺带删掉的本店过期墓碑上限（每次一小批，数量随有效期有界）。"""


class PgConversationRepository:
    """实现 core.ports.ConversationRepository（conversation / message / message_recall）。

    消息按 (shop_id, msg_id) 去重（唯一约束 + ON CONFLICT DO NOTHING，并发重复推送也只存一条）。
    先写消息、再补会话：message 指向 conversation 的外键延迟到提交时检查，
    所以重复消息不会凭空建出会话。
    撤回的消息留一行"墓碑"继续占住 msg_id（去重），文字和卡片清空：text IS NULL 表示已撤回
    （正常写入的消息 text 都不是 NULL），recent_messages 不返回它。

    撤回先于消息到达（§8.3 平台推送乱序）：本店没见过的 msg_id 被撤回时在 message_recall 记一条
    墓碑（按店铺 + 会话，有效期 recall_ttl，按注入的时钟判断）。消息到达时删掉墓碑：还在有效期内
    就把这条消息存成已撤回的行（文字和卡片为 NULL，占住 msg_id 去重）、返回 False。记墓碑时顺带
    删掉本店过期的墓碑（每次最多 _TOMBSTONE_CLEANUP_ROWS 条），墓碑数量随有效期有界。

    撤回和消息写入按 (shop_id, msg_id) 取事务级 advisory lock 串行：多个实例同时处理撤回和这条消息
    时，后拿到锁的一方在 READ COMMITTED 下每条语句取新快照，一定看得到先提交的一方写下的消息或墓碑，
    不会出现"撤回没找到消息、消息也没看到墓碑"两边都落空、消息留下来的情况。
    """

    def __init__(
        self,
        db: Database,
        *,
        clock: Clock | None = None,
        recall_ttl: timedelta = RECALL_TOMBSTONE_TTL,
    ) -> None:
        if recall_ttl <= timedelta(0):
            raise ValueError("recall_ttl 必须为正")
        self._db = db
        self._clock = clock or SystemClock()
        self._recall_ttl = recall_ttl

    @staticmethod
    def _lock_msg_id(conn: Conn, shop_id: str, msg_id: str) -> None:
        """同一店铺同一 msg_id 的撤回和写入串行（事务结束自动释放）。"""
        conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"rodit.message:{shop_id}:{msg_id}",),
        )

    @staticmethod
    def _take_tombstone(
        conn: Conn,
        shop_id: str,
        conversation_id: str,
        msg_id: str,
        now: datetime,
        *,
        conversation_must_exist: bool = False,
    ) -> bool:
        """删掉这条消息的撤回墓碑（用掉或已过期都删）；墓碑还在有效期内返回 True。"""
        cond = "shop_id = %s AND conversation_id = %s AND msg_id = %s"
        params: list[object] = [shop_id, conversation_id, msg_id]
        if conversation_must_exist:
            cond += " AND EXISTS (SELECT 1 FROM conversation WHERE shop_id = %s AND id = %s)"
            params += [shop_id, conversation_id]
        row = conn.execute(
            f"DELETE FROM message_recall WHERE {cond} RETURNING expires_at", params
        ).fetchone()
        return row is not None and row["expires_at"] > now

    def add_message(self, msg: StandardMessage) -> bool:
        now = self._clock.now()
        with self._db.session(msg.shop_id) as conn:
            self._lock_msg_id(conn, msg.shop_id, msg.msg_id)
            # 先被撤回过（§8.3 撤回先到）：存成已撤回的行，只占住 msg_id，不留文字和卡片
            recalled = self._take_tombstone(conn, msg.shop_id, msg.conversation_id, msg.msg_id, now)
            kept = not recalled
            inserted = conn.execute(
                f"INSERT INTO message ({_M_COLS})"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (shop_id, msg_id) DO NOTHING RETURNING id",
                (
                    msg.msg_id,
                    msg.shop_id,
                    msg.conversation_id,
                    msg.role,
                    msg.text if kept else None,
                    msg.item_card.item_id if msg.item_card and kept else None,
                    msg.item_card.sku_id if msg.item_card and kept else None,
                    msg.order_card.order_id if msg.order_card and kept else None,
                    msg.received_at,
                ),
            ).fetchone()
            if inserted is None:
                return False
            conn.execute(
                "INSERT INTO conversation (shop_id, id, buyer_ref, started_at)"
                " VALUES (%s, %s, %s, %s) ON CONFLICT (shop_id, id) DO NOTHING",
                (msg.shop_id, msg.conversation_id, msg.buyer_ref, msg.received_at),
            )
            return kept

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
        now = self._clock.now()
        with self._db.session(shop_id) as conn:
            self._lock_msg_id(conn, shop_id, msg_id)
            # 会话不存在时不写、墓碑也留着（和内存实现一致）
            recalled = self._take_tombstone(
                conn, shop_id, conversation_id, msg_id, now, conversation_must_exist=True
            )
            inserted = conn.execute(
                "INSERT INTO message (msg_id, shop_id, conversation_id, role, text, created_at)"
                " SELECT %s, %s, %s, %s, %s, %s"
                " WHERE EXISTS (SELECT 1 FROM conversation WHERE shop_id = %s AND id = %s)"
                " ON CONFLICT (shop_id, msg_id) DO NOTHING RETURNING id",
                # 和内存实现（StoredMessage 构造时清洗）一致：去掉 NUL 等控制字符再写
                (
                    msg_id,
                    shop_id,
                    conversation_id,
                    role,
                    None if recalled else strip_control_chars(text),
                    at,
                    shop_id,
                    conversation_id,
                ),
            ).fetchone()
            return inserted is not None and not recalled

    def get_conversation(self, shop_id: str, conversation_id: str) -> Conversation | None:
        with self._db.session(shop_id) as conn:
            r = conn.execute(
                "SELECT id, shop_id, buyer_ref, started_at, closed_at FROM conversation"
                " WHERE shop_id = %s AND id = %s",
                (shop_id, conversation_id),
            ).fetchone()
        if r is None:
            return None
        return Conversation(
            id=r["id"],
            shop_id=r["shop_id"],
            buyer_ref=r["buyer_ref"],
            started_at=_cst(r["started_at"]),
            closed_at=_cst(r["closed_at"]),
        )

    def recent_messages(
        self, shop_id: str, conversation_id: str, *, limit: int = 20
    ) -> list[StoredMessage]:
        if limit <= 0:
            return []
        with self._db.session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_M_COLS} FROM ("
                f" SELECT id, {_M_COLS} FROM message"
                " WHERE shop_id = %s AND conversation_id = %s AND text IS NOT NULL"
                " ORDER BY created_at DESC, id DESC LIMIT %s"
                ") recent ORDER BY created_at, id",
                (shop_id, conversation_id, limit),
            ).fetchall()
        return [_message(r) for r in rows]

    def list_outgoing(
        self,
        shop_id: str,
        *,
        since: datetime,
        until: datetime,
        roles: Sequence[Role] = (Role.AGENT, Role.BOT),
    ) -> list[StoredMessage]:
        wanted = sorted({Role(r).value for r in roles})
        if not wanted:
            return []
        with self._db.session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_M_COLS} FROM message"
                " WHERE shop_id = %s AND role = ANY(%s) AND text IS NOT NULL"
                " AND created_at >= %s AND created_at < %s"
                " ORDER BY created_at, id",
                (shop_id, wanted, since, until),
            ).fetchall()
        return [_message(r) for r in rows]

    def recall_message(self, shop_id: str, conversation_id: str, msg_id: str) -> bool:
        now = self._clock.now()
        with self._db.session(shop_id) as conn:
            self._lock_msg_id(conn, shop_id, msg_id)
            updated = conn.execute(
                "UPDATE message SET text = NULL, item_id = NULL, sku_id = NULL, order_ref = NULL"
                " WHERE shop_id = %s AND conversation_id = %s AND msg_id = %s"
                " AND text IS NOT NULL RETURNING id",
                (shop_id, conversation_id, msg_id),
            ).fetchone()
            if updated is not None:
                return True
            # 本店没见过这个 msg_id：撤回先到了（§8.3），记墓碑。同一条的墓碑已过期时重新计时
            recorded = conn.execute(
                "INSERT INTO message_recall"
                " (shop_id, conversation_id, msg_id, recalled_at, expires_at)"
                " SELECT %s, %s, %s, %s, %s"
                " WHERE NOT EXISTS (SELECT 1 FROM message WHERE shop_id = %s AND msg_id = %s)"
                " ON CONFLICT (shop_id, conversation_id, msg_id) DO UPDATE"
                " SET recalled_at = EXCLUDED.recalled_at, expires_at = EXCLUDED.expires_at"
                " WHERE message_recall.expires_at <= EXCLUDED.recalled_at"
                " RETURNING 1",
                (
                    shop_id,
                    conversation_id,
                    msg_id,
                    now,
                    now + self._recall_ttl,
                    shop_id,
                    msg_id,
                ),
            ).fetchone()
            if recorded is not None:
                conn.execute(
                    "DELETE FROM message_recall WHERE ctid = ANY(ARRAY("
                    " SELECT ctid FROM message_recall"
                    " WHERE shop_id = %s AND expires_at <= %s LIMIT %s))",
                    (shop_id, now, _TOMBSTONE_CLEANUP_ROWS),
                )
            return False

    def purge_shop(self, shop_id: str) -> int:
        # §8.4 分批删、每批一个短事务（Database.purge_rows）。先删消息再删会话：
        # 会话的外键级联会把剩下的消息放进同一个事务里一起删
        return sum(
            self._db.purge_rows(shop_id, table)
            for table in ("message", "conversation", "message_recall")
        )


# ================================================================ 推荐（§5.6、§10 suggestion）

_S_FIELDS = (
    "shop_id",
    "conversation_id",
    "msg_id",
    "question_text",
    "item_id",
    "intent",
    "intent_confidence",
    "route_rule",
    "text",
    "kind",
    "evidence_ids",
    "qa_scores",
    "status",
    "edited_text",
    "reject_reason",
    "agent_id",
    "latency_ms",
    "created_at",
)
_S_COLS = "id, " + ", ".join(_S_FIELDS)


def _suggestion_values(record: SuggestionRecord) -> list[Any]:
    data = {f: getattr(record, f) for f in _S_FIELDS}
    data["evidence_ids"] = list(record.evidence_ids)
    data["qa_scores"] = _jsonb(record.qa_scores)
    return [data[f] for f in _S_FIELDS]


def _suggestion(r: Row) -> SuggestionRecord:
    return SuggestionRecord(**{**r, "created_at": _cst(r["created_at"])})


class PgSuggestionRepository:
    """实现 core.ports.SuggestionRepository。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, record: SuggestionRecord) -> SuggestionRecord:
        placeholders = ", ".join(["%s"] * len(_S_FIELDS))
        with self._db.session(record.shop_id) as conn:
            r = conn.execute(
                f"INSERT INTO suggestion ({', '.join(_S_FIELDS)}) VALUES ({placeholders})"
                f" RETURNING {_S_COLS}",
                _suggestion_values(record),
            ).fetchone()
        assert r is not None
        return _suggestion(r)

    def get(self, shop_id: str, suggestion_id: int) -> SuggestionRecord | None:
        with self._db.session(shop_id) as conn:
            r = conn.execute(
                f"SELECT {_S_COLS} FROM suggestion WHERE shop_id = %s AND id = %s",
                (shop_id, suggestion_id),
            ).fetchone()
        return _suggestion(r) if r else None

    def update(
        self, record: SuggestionRecord, *, expected_status: SuggestionStatus | None = None
    ) -> SuggestionRecord:
        if record.id is None:
            raise KeyError(None)
        fields = [f for f in _S_FIELDS if f != "shop_id"]
        values = dict(zip(_S_FIELDS, _suggestion_values(record), strict=True))
        assignments = ", ".join(f"{f} = %s" for f in fields)
        where = "shop_id = %s AND id = %s"
        params: list[object] = [*(values[f] for f in fields), record.shop_id, record.id]
        if expected_status is not None:
            where += " AND status = %s"
            params.append(expected_status.value)
        with self._db.session(record.shop_id) as conn:
            r = conn.execute(
                f"UPDATE suggestion SET {assignments} WHERE {where} RETURNING {_S_COLS}",
                params,
            ).fetchone()
            if r is None and expected_status is not None:
                exists = conn.execute(
                    "SELECT status FROM suggestion WHERE shop_id = %s AND id = %s",
                    (record.shop_id, record.id),
                ).fetchone()
                if exists is not None:
                    raise ConcurrencyConflict(
                        f"推荐 {record.id} 的状态已是 {exists['status']}，"
                        f"不是 {expected_status.value}"
                    )
        if r is None:
            raise KeyError(record.id)
        return _suggestion(r)

    def list(
        self,
        shop_id: str,
        *,
        conversation_id: str | None = None,
        status: SuggestionStatus | None = None,
        since: datetime | None = None,
    ) -> list[SuggestionRecord]:
        conds, params = ["shop_id = %s"], [shop_id]
        for column, value, op in (
            ("conversation_id", conversation_id, "="),
            ("status", status, "="),
            ("created_at", since, ">="),
        ):
            if value is not None:
                conds.append(f"{column} {op} %s")
                params.append(value)
        with self._db.session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_S_COLS} FROM suggestion WHERE {' AND '.join(conds)}"
                " ORDER BY created_at, id",
                params,
            ).fetchall()
        return [_suggestion(r) for r in rows]

    def purge_shop(self, shop_id: str) -> int:
        # §8.4 分批删、每批一个短事务（Database.purge_rows）
        return self._db.purge_rows(shop_id, "suggestion")


# ================================================================ 反馈（§5.8）

_GAP_COLS = "shop_id, intent, missing, question_text, source, created_at"
_LABEL_COLS = (
    "shop_id, text, predicted_intent, predicted_confidence, route_rule, true_intent, source,"
    " created_at"
)


def _gap(r: Row) -> KnowledgeGap:
    return KnowledgeGap(**{**r, "created_at": _cst(r["created_at"])})


def _label(r: Row) -> LabeledSample:
    return LabeledSample(**{**r, "created_at": _cst(r["created_at"])})


class PgFeedbackRepository:
    """实现 core.ports.FeedbackRepository（knowledge_gap / labeled_sample）。

    export_training_labels 用 rodit_trainer 角色读（只有 labeled_sample 的跨店只读策略，
    见 schema.sql），不经过商家侧的 rodit_app 权限。
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def add_gap(self, gap: KnowledgeGap) -> None:
        with self._db.session(gap.shop_id) as conn:
            conn.execute(
                f"INSERT INTO knowledge_gap ({_GAP_COLS}) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    gap.shop_id,
                    gap.intent,
                    gap.missing,
                    gap.question_text,
                    gap.source,
                    gap.created_at,
                ),
            )

    def list_gaps(self, shop_id: str) -> list[KnowledgeGap]:
        with self._db.session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_GAP_COLS} FROM knowledge_gap WHERE shop_id = %s ORDER BY created_at, id",
                (shop_id,),
            ).fetchall()
        return [_gap(r) for r in rows]

    def add_label(self, sample: LabeledSample) -> None:
        with self._db.session(sample.shop_id) as conn:
            conn.execute(
                f"INSERT INTO labeled_sample ({_LABEL_COLS})"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    sample.shop_id,
                    sample.text,
                    sample.predicted_intent,
                    sample.predicted_confidence,
                    sample.route_rule,
                    sample.true_intent,
                    sample.source,
                    sample.created_at,
                ),
            )

    def list_labels(self, shop_id: str) -> list[LabeledSample]:
        with self._db.session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_LABEL_COLS} FROM labeled_sample WHERE shop_id = %s"
                " ORDER BY created_at, id",
                (shop_id,),
            ).fetchall()
        return [_label(r) for r in rows]

    def export_training_labels(
        self, *, since: datetime | None = None, until: datetime | None = None
    ) -> list[LabeledSample]:
        conds: list[str] = []
        params: list[object] = []
        if since is not None:
            conds.append("created_at >= %s")
            params.append(since)
        if until is not None:
            conds.append("created_at < %s")
            params.append(until)
        where = f" WHERE {' AND '.join(conds)}" if conds else ""
        with self._db.session(None, trainer=True) as conn:
            rows = conn.execute(
                f"SELECT {_LABEL_COLS} FROM labeled_sample{where} ORDER BY created_at, id",
                params,
            ).fetchall()
        return [_label(r) for r in rows]

    def purge_shop(self, shop_id: str) -> int:
        # §8.4 分批删、每批一个短事务（Database.purge_rows）
        return self._db.purge_rows(shop_id, "knowledge_gap") + self._db.purge_rows(
            shop_id, "labeled_sample"
        )


# ================================================================ 每周人工复核（§13.5）

_Q_FIELDS = (
    "shop_id",
    "period",
    "suggestion_id",
    "intent",
    "status",
    "scores",
    "reviewer",
    "note",
    "sampled_at",
    "reviewed_at",
)
_Q_COLS = "id, " + ", ".join(_Q_FIELDS)


def _quality_review(r: Row) -> QualityReview:
    return QualityReview.model_validate(
        {**r, "sampled_at": _cst(r["sampled_at"]), "reviewed_at": _cst(r["reviewed_at"])}
    )


class PgQualityReviewRepository:
    """实现 core.ports.QualityReviewRepository（quality_review）。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add_many(self, reviews: Sequence[QualityReview]) -> list[QualityReview]:
        for r in reviews:
            if r.id is not None or r.status != QualityReviewStatus.PENDING:
                raise ValueError("新抽中的复核样本必须是 pending、且没有 id")
        by_shop: dict[str, list[int]] = {}
        for i, r in enumerate(reviews):
            by_shop.setdefault(r.shop_id, []).append(i)
        out: list[QualityReview | None] = [None] * len(reviews)
        placeholders = ", ".join(["%s"] * len(_Q_FIELDS))
        for shop_id, indexes in by_shop.items():
            with self._db.session(shop_id) as conn:
                for i in indexes:
                    r = reviews[i]
                    # ON CONFLICT DO NOTHING 不返回已有的行：冲突时再按唯一键读出来
                    row = conn.execute(
                        f"INSERT INTO quality_review ({', '.join(_Q_FIELDS)})"
                        f" VALUES ({placeholders})"
                        " ON CONFLICT (shop_id, period, suggestion_id) DO NOTHING"
                        f" RETURNING {_Q_COLS}",
                        (
                            r.shop_id,
                            r.period,
                            r.suggestion_id,
                            r.intent,
                            r.status,
                            None,
                            None,
                            r.note,
                            r.sampled_at,
                            None,
                        ),
                    ).fetchone()
                    if row is None:
                        row = conn.execute(
                            f"SELECT {_Q_COLS} FROM quality_review"
                            " WHERE shop_id = %s AND period = %s AND suggestion_id = %s",
                            (r.shop_id, r.period, r.suggestion_id),
                        ).fetchone()
                    assert row is not None
                    out[i] = _quality_review(row)
        return [r for r in out if r is not None]

    def get(self, shop_id: str, review_id: int) -> QualityReview | None:
        with self._db.session(shop_id) as conn:
            r = conn.execute(
                f"SELECT {_Q_COLS} FROM quality_review WHERE shop_id = %s AND id = %s",
                (shop_id, review_id),
            ).fetchone()
        return _quality_review(r) if r else None

    def list(
        self,
        shop_id: str,
        *,
        period: str | None = None,
        status: QualityReviewStatus | None = None,
    ) -> list[QualityReview]:
        conds, params = ["shop_id = %s"], [shop_id]
        for column, value in (("period", period), ("status", status)):
            if value is not None:
                conds.append(f"{column} = %s")
                params.append(value)
        with self._db.session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_Q_COLS} FROM quality_review WHERE {' AND '.join(conds)} ORDER BY id",
                params,
            ).fetchall()
        return [_quality_review(r) for r in rows]

    def record(
        self,
        shop_id: str,
        review_id: int,
        *,
        status: QualityReviewStatus,
        scores: QualityScores | None,
        reviewer: str,
        note: str | None,
        reviewed_at: datetime,
    ) -> QualityReview:
        if QualityReviewStatus(status) == QualityReviewStatus.PENDING:
            raise ValueError("复核结果不能是 pending")
        with self._db.session(shop_id) as conn:
            cur = conn.execute(
                f"SELECT {_Q_COLS} FROM quality_review WHERE shop_id = %s AND id = %s FOR UPDATE",
                (shop_id, review_id),
            ).fetchone()
            if cur is None:
                raise KeyError(review_id)
            # 先按模型校验（scored 要有 scores 等），再写库；note 在模型里做控制字符清洗
            updated = QualityReview.model_validate(
                {
                    **_quality_review(cur).model_dump(),
                    "status": status,
                    "scores": scores.model_dump() if scores is not None else None,
                    "reviewer": reviewer,
                    "note": note,
                    "reviewed_at": reviewed_at,
                }
            )
            row = conn.execute(
                "UPDATE quality_review SET status = %s, scores = %s, reviewer = %s, note = %s,"
                f" reviewed_at = %s WHERE shop_id = %s AND id = %s RETURNING {_Q_COLS}",
                (
                    updated.status,
                    _jsonb(updated.scores.model_dump() if updated.scores else None),
                    updated.reviewer,
                    updated.note,
                    updated.reviewed_at,
                    shop_id,
                    review_id,
                ),
            ).fetchone()
        assert row is not None
        return _quality_review(row)

    def purge_shop(self, shop_id: str) -> int:
        # §8.4 分批删、每批一个短事务（Database.purge_rows）
        return self._db.purge_rows(shop_id, "quality_review")


# ================================================================ 售后审核（§7.3.2）

_R_FIELDS = (
    "platform_status",
    "platform_status_version",
    "state",
    "matched_rule",
    "suggestion",
    "reason",
    "requires_manager",
    "risk_flags",
    "verify_started_at",
    "confirmed_by",
    "action_result",
    "updated_at",
)
_R_COLS = "shop_id, aftersale_id, version, " + ", ".join(_R_FIELDS)


def _review_values(review: AfterSaleReview) -> list[Any]:
    for rule in review.matched_rules:
        if not rule or "," in rule:  # matched_rule 列是逗号拼接，规则名本身不能含逗号
            raise ValueError(f"非法的审核规则名：{rule!r}")
    data: dict[str, Any] = {
        "platform_status": review.platform_status,
        "platform_status_version": review.platform_status_version,
        "state": review.state,
        "matched_rule": review.matched_rule,
        "suggestion": review.suggestion,
        "reason": review.reason,
        "requires_manager": review.requires_manager,
        "risk_flags": list(review.risk_flags),
        "verify_started_at": review.verify_started_at,
        "confirmed_by": review.confirmed_by,
        "action_result": _jsonb(review.action_result),
        "updated_at": review.updated_at,
    }
    return [data[f] for f in _R_FIELDS]


def _review(r: Row) -> AfterSaleReview:
    return AfterSaleReview(
        shop_id=r["shop_id"],
        aftersale_id=r["aftersale_id"],
        platform_status=r["platform_status"],
        platform_status_version=r["platform_status_version"],
        state=r["state"],
        matched_rules=r["matched_rule"].split(",") if r["matched_rule"] else [],
        suggestion=r["suggestion"],
        reason=r["reason"],
        requires_manager=r["requires_manager"],
        risk_flags=list(r["risk_flags"]),
        verify_started_at=_cst(r["verify_started_at"]),
        confirmed_by=r["confirmed_by"],
        action_result=r["action_result"],
        version=r["version"],
        updated_at=_cst(r["updated_at"]),
    )


class PgAfterSaleReviewRepository:
    """实现 core.ports.AfterSaleReviewRepository（乐观锁）。

    新建：INSERT … ON CONFLICT DO NOTHING，已存在 → ConcurrencyConflict，version = 1；
    更新：UPDATE … WHERE version = expected_version，0 行 → ConcurrencyConflict，version + 1。
    两个会话同时按同一版本更新时，后到的那个在行锁释放后重新检查条件，更新 0 行。
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, shop_id: str, aftersale_id: str) -> AfterSaleReview | None:
        with self._db.session(shop_id) as conn:
            r = conn.execute(
                f"SELECT {_R_COLS} FROM aftersale_review WHERE shop_id = %s AND aftersale_id = %s",
                (shop_id, aftersale_id),
            ).fetchone()
        return _review(r) if r else None

    def save(self, review: AfterSaleReview, *, expected_version: int | None) -> AfterSaleReview:
        values = _review_values(review)
        key = (review.shop_id, review.aftersale_id)
        with self._db.session(review.shop_id) as conn:
            if expected_version is None:
                placeholders = ", ".join(["%s"] * len(_R_FIELDS))
                r = conn.execute(
                    f"INSERT INTO aftersale_review (shop_id, aftersale_id, version,"
                    f" {', '.join(_R_FIELDS)}) VALUES (%s, %s, 1, {placeholders})"
                    f" ON CONFLICT (shop_id, aftersale_id) DO NOTHING RETURNING {_R_COLS}",
                    [*key, *values],
                ).fetchone()
                if r is None:
                    raise ConcurrencyConflict(f"售后审核 {review.aftersale_id} 已存在")
            else:
                assignments = ", ".join(f"{f} = %s" for f in _R_FIELDS)
                r = conn.execute(
                    f"UPDATE aftersale_review SET {assignments}, version = version + 1"
                    " WHERE shop_id = %s AND aftersale_id = %s AND version = %s"
                    f" RETURNING {_R_COLS}",
                    [*values, *key, expected_version],
                ).fetchone()
                if r is None:
                    cur = conn.execute(
                        "SELECT version FROM aftersale_review"
                        " WHERE shop_id = %s AND aftersale_id = %s",
                        key,
                    ).fetchone()
                    raise ConcurrencyConflict(
                        f"售后审核 {review.aftersale_id} 版本冲突：期望 {expected_version}，"
                        f"实际 {cur['version'] if cur else '不存在'}"
                    )
        return _review(r)

    def list_by_state(
        self, shop_id: str, states: Sequence[ReviewState] | None = None
    ) -> list[AfterSaleReview]:
        conds, params = ["shop_id = %s"], [shop_id]
        if states is not None:
            conds.append("state = ANY(%s)")
            params.append([str(s) for s in states])
        with self._db.session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_R_COLS} FROM aftersale_review WHERE {' AND '.join(conds)}"
                ' ORDER BY aftersale_id COLLATE "C"',
                params,
            ).fetchall()
        return [_review(r) for r in rows]

    def purge_shop(self, shop_id: str) -> int:
        # §8.4 分批删、每批一个短事务（Database.purge_rows）
        return self._db.purge_rows(shop_id, "aftersale_review")


# ================================================================ 授权（§8.4）


_TOKEN_FORMAT = b"RDT1"  # 密文格式标记（版本 1：AES-256-GCM）
_TOKEN_NONCE_BYTES = 12
_TOKEN_TAG_BYTES = 16
_TOKEN_KDF_INFO = b"rodit shop_auth access_token v1"
# 续期令牌的附加认证数据 = 店铺 ID + NUL + "refresh"。店铺 ID 存在 TEXT 列里，不会有 NUL，
# 所以它和任何店铺 access_token 的附加认证数据（店铺 ID 本身）都不相同；用 ":refresh" 的话，
# 店铺 X 的续期令牌密文能当成店铺 "X:refresh" 的 access_token 解开
_REFRESH_AAD_SUFFIX = "\x00refresh"
_AUTH_COLS = (
    "shop_id, access_token, scopes, expires_at, refresh_token, refresh_expires_at,"
    " refreshed_at, needs_reauth"
)


def _token_cipher(encryption_key: str) -> AESGCM:
    """从配置的密钥串派生 256 位 AES 密钥（HKDF-SHA256）。"""
    kdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_TOKEN_KDF_INFO)
    return AESGCM(kdf.derive(encryption_key.encode("utf-8")))


class PgShopAuthRepository:
    """实现 core.ports.ShopAuthRepository。access_token 在应用里用 AES-256-GCM 加密后落库。

    - encryption_key 由应用层从环境变量读（默认 RODIT_TOKEN_KEY，见 from_env），
      仓库里没有真实密钥；只有测试里写死合成的密钥。
    - 加解密都在应用进程里做，数据库只收到、只存密文（BYTEA）：密钥和明文令牌从不作为
      SQL 参数发给数据库——库开了慢查询日志（log_min_duration_statement）或 auto_explain 时，
      绑定参数会原样写进数据库日志。
    - 密文格式：b"RDT1" + 12 字节随机 nonce + 密文 + 16 字节认证标签；店铺 ID 作为附加认证数据，
      把一个店铺的密文挪到别的店铺名下解不出来。续期令牌（refresh_token，迁移版本 5）同样加密，
      附加认证数据是"店铺 ID + NUL 字节 + refresh"：和 access_token 的密文互换、挪到任何店铺
      （包括名字是"店铺 ID:refresh"这样的店铺）的 access_token 列都解不出来。
    - expiring_shop_ids 经 SECURITY DEFINER 函数 rodit_auth_expiring 跨店读（只有店铺 ID 和到期
      时间，不解密）；mark_needs_reauth 只改标记、不解密。
    - 密钥不对、密文损坏、旧格式（v0.x 用库内 pgcrypto 加密的令牌）都抛 TokenDecryptionError，
      需要重新授权写入；repr、异常消息里都不出现密钥和令牌（令牌进出都是 SecretStr）。
    - 换密钥：用旧密钥读出、新密钥写回（MVP 未提供批量工具）。
    """

    MIN_KEY_LENGTH = 16

    def __init__(self, db: Database, *, encryption_key: str) -> None:
        if len(encryption_key or "") < self.MIN_KEY_LENGTH:
            raise ValueError(f"令牌加密密钥至少 {self.MIN_KEY_LENGTH} 个字符")
        self._db = db
        self._cipher = _token_cipher(encryption_key)

    def __repr__(self) -> str:
        return "PgShopAuthRepository(encryption_key=***)"

    @classmethod
    def from_env(cls, db: Database, env_name: str = "RODIT_TOKEN_KEY") -> PgShopAuthRepository:
        key = os.environ.get(env_name, "")
        if not key:
            raise RuntimeError(f"环境变量 {env_name} 未设置：店铺令牌必须加密存储（方案 §8.4）")
        return cls(db, encryption_key=key)

    @staticmethod
    def _aad(shop_id: str, *, refresh: bool) -> bytes:
        if "\x00" in shop_id:  # 库里存不下；也保证和续期令牌的附加认证数据分得开
            raise ValueError("店铺 ID 不能含 NUL 字符")
        return (shop_id + _REFRESH_AAD_SUFFIX if refresh else shop_id).encode("utf-8")

    def _encrypt(self, shop_id: str, token: str, *, refresh: bool = False) -> bytes:
        nonce = os.urandom(_TOKEN_NONCE_BYTES)
        sealed = self._cipher.encrypt(
            nonce, token.encode("utf-8"), self._aad(shop_id, refresh=refresh)
        )
        return _TOKEN_FORMAT + nonce + sealed

    def _decrypt(self, shop_id: str, blob: bytes, *, refresh: bool = False) -> str:
        head = len(_TOKEN_FORMAT)
        if not blob.startswith(_TOKEN_FORMAT) or len(blob) < (
            head + _TOKEN_NONCE_BYTES + _TOKEN_TAG_BYTES
        ):
            raise TokenDecryptionError(
                f"店铺 {shop_id} 的令牌格式不对（可能是旧版加密或数据损坏）：请重新授权写入"
            )
        nonce = blob[head : head + _TOKEN_NONCE_BYTES]
        try:
            plain = self._cipher.decrypt(
                nonce, blob[head + _TOKEN_NONCE_BYTES :], self._aad(shop_id, refresh=refresh)
            )
            return plain.decode("utf-8")
        except (InvalidTag, UnicodeDecodeError) as e:
            raise TokenDecryptionError(
                f"店铺 {shop_id} 的令牌无法解密：请检查令牌加密密钥是否与写入时一致"
            ) from e

    def get(self, shop_id: str) -> ShopAuth | None:
        with self._db.session(shop_id) as conn:
            r = conn.execute(
                f"SELECT {_AUTH_COLS} FROM shop_auth WHERE shop_id = %s", (shop_id,)
            ).fetchone()
        if r is None:
            return None
        refresh = r["refresh_token"]
        return ShopAuth(
            shop_id=r["shop_id"],
            access_token=SecretStr(self._decrypt(r["shop_id"], bytes(r["access_token"]))),
            scopes=list(r["scopes"]),
            expires_at=_cst(r["expires_at"]),
            refresh_token=(
                SecretStr(self._decrypt(r["shop_id"], bytes(refresh), refresh=True))
                if refresh is not None
                else None
            ),
            refresh_expires_at=_cst(r["refresh_expires_at"]),
            refreshed_at=_cst(r["refreshed_at"]),
            needs_reauth=bool(r["needs_reauth"]),
        )

    def save(self, auth: ShopAuth) -> None:
        sealed = self._encrypt(auth.shop_id, auth.access_token.get_secret_value())
        sealed_refresh = (
            self._encrypt(auth.shop_id, auth.refresh_token.get_secret_value(), refresh=True)
            if auth.refresh_token is not None
            else None
        )
        with self._db.session(auth.shop_id) as conn:
            conn.execute(
                f"INSERT INTO shop_auth ({_AUTH_COLS})"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (shop_id) DO UPDATE SET access_token = EXCLUDED.access_token,"
                " scopes = EXCLUDED.scopes, expires_at = EXCLUDED.expires_at,"
                " refresh_token = EXCLUDED.refresh_token,"
                " refresh_expires_at = EXCLUDED.refresh_expires_at,"
                " refreshed_at = EXCLUDED.refreshed_at, needs_reauth = EXCLUDED.needs_reauth",
                (
                    auth.shop_id,
                    sealed,
                    list(auth.scopes),
                    auth.expires_at,
                    sealed_refresh,
                    auth.refresh_expires_at,
                    auth.refreshed_at,
                    auth.needs_reauth,
                ),
            )

    def expiring_shop_ids(self, before: datetime) -> list[tuple[str, datetime]]:
        """跨店：不设店铺范围，经 SECURITY DEFINER 函数读（只有店铺 ID 和到期时间）。"""
        with self._db.session(None) as conn:
            rows = conn.execute(
                "SELECT shop_id, expires_at FROM rodit_auth_expiring(%s)"
                ' ORDER BY expires_at, shop_id COLLATE "C"',
                (before,),
            ).fetchall()
        return [(r["shop_id"], _cst(r["expires_at"])) for r in rows]

    def mark_needs_reauth(self, shop_id: str) -> bool:
        with self._db.session(shop_id) as conn:
            return (
                conn.execute(
                    "UPDATE shop_auth SET needs_reauth = true WHERE shop_id = %s", (shop_id,)
                ).rowcount
                > 0
            )

    def delete(self, shop_id: str) -> bool:
        """只删除、不解密：旧格式或密钥不对的令牌也能删掉（取消订购，§8.4）。"""
        return self.purge_shop(shop_id) > 0

    def purge_shop(self, shop_id: str) -> int:
        with self._db.session(shop_id) as conn:
            return conn.execute("DELETE FROM shop_auth WHERE shop_id = %s", (shop_id,)).rowcount


# ================================================================ 审计（§4.1 原则 6、§12.4）

_E_COLS = "shop_id, actor, action, detail, created_at"
_D_COLS = (
    "shop_id, conversation_id, stage, provider, questions_ver, answers, route_result, "
    "latency_ms, created_at"
)


class PgAuditLog:
    """实现 core.ports.AuditLog（audit_log / decision_log）。按店铺查询，结果按时间升序。

    shop_id 为空的系统事件（如没有店铺归属的大模型调用、shop.purged、eval.verdict）在不设店铺范围
    的事务里写入，也只能在不设店铺范围的事务里读（list_system_events；迁移版本 5 的
    system_event_select 策略）：设了店铺范围的事务读不到任何系统事件，list_events 只返回本店的。
    detail / answers 不放买家原文（调用方负责最小化，§5.3）。
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def record(self, event: AuditEvent) -> None:
        with self._db.session(event.shop_id) as conn:
            conn.execute(  # 不用 RETURNING：无店铺事件对应用不可见
                f"INSERT INTO audit_log ({_E_COLS}) VALUES (%s, %s, %s, %s, %s)",
                (event.shop_id, event.actor, event.action, _jsonb(event.detail), event.created_at),
            )

    def log_decision(self, entry: DecisionLogEntry) -> None:
        with self._db.session(entry.shop_id) as conn:
            conn.execute(
                f"INSERT INTO decision_log ({_D_COLS}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    entry.shop_id,
                    entry.conversation_id,
                    entry.stage,
                    entry.provider,
                    entry.questions_ver,
                    _jsonb(entry.answers),
                    entry.route_result,
                    entry.latency_ms,
                    entry.created_at,
                ),
            )

    def list_events(
        self, shop_id: str, *, action: str | None = None, since: datetime | None = None
    ) -> list[AuditEvent]:
        conds, params = ["shop_id = %s"], [shop_id]
        if action is not None:
            conds.append("action = %s")
            params.append(action)
        if since is not None:
            conds.append("created_at >= %s")
            params.append(since)
        with self._db.session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_E_COLS} FROM audit_log WHERE {' AND '.join(conds)}"
                " ORDER BY created_at, id",
                params,
            ).fetchall()
        return [AuditEvent(**{**r, "created_at": _cst(r["created_at"])}) for r in rows]

    def list_system_events(
        self, *, action: str | None = None, since: datetime | None = None
    ) -> list[AuditEvent]:
        conds: list[str] = ["shop_id IS NULL"]
        params: list[Any] = []
        if action is not None:
            conds.append("action = %s")
            params.append(action)
        if since is not None:
            conds.append("created_at >= %s")
            params.append(since)
        with self._db.session(None) as conn:
            rows = conn.execute(
                f"SELECT {_E_COLS} FROM audit_log WHERE {' AND '.join(conds)}"
                " ORDER BY created_at, id",
                params,
            ).fetchall()
        return [AuditEvent(**{**r, "created_at": _cst(r["created_at"])}) for r in rows]

    def list_decisions(
        self,
        shop_id: str,
        *,
        stage: DecisionStage | None = None,
        since: datetime | None = None,
    ) -> list[DecisionLogEntry]:
        conds, params = ["shop_id = %s"], [shop_id]
        if stage is not None:
            conds.append("stage = %s")
            params.append(stage)
        if since is not None:
            conds.append("created_at >= %s")
            params.append(since)
        with self._db.session(shop_id) as conn:
            rows = conn.execute(
                f"SELECT {_D_COLS} FROM decision_log WHERE {' AND '.join(conds)}"
                " ORDER BY created_at, id",
                params,
            ).fetchall()
        return [DecisionLogEntry(**{**r, "created_at": _cst(r["created_at"])}) for r in rows]

    def purge_shop(self, shop_id: str) -> int:
        """删除该店铺的审计和判断记录（§8.4 按约定删除数据时调用；是否保留由委托协议决定）。

        审计记录量最大：分批删、每批一个短事务，不受请求级语句 / 等锁超时约束
        （Database.purge_rows）。
        """
        return self._db.purge_rows(shop_id, "audit_log") + self._db.purge_rows(
            shop_id, "decision_log"
        )
