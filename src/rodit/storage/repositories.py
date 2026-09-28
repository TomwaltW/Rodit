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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from pydantic import SecretStr, TypeAdapter

from rodit.core.clock import CST, Clock, SystemClock
from rodit.core.enums import DecisionStage, KnowledgeStatus, ReviewState, Role, SuggestionStatus
from rodit.core.errors import ConcurrencyConflict, KnowledgeUnavailable, RoditError
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
    Shop,
    ShopAuth,
    StandardMessage,
    StoredMessage,
    SuggestionRecord,
)
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


class TokenDecryptionError(RoditError):
    """店铺令牌解密失败：密钥与写入时不一致，或数据损坏。消息里不带密钥和令牌。"""


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
        if r is None:
            return None
        return Shop(
            id=r["id"],
            platform=r["platform"],
            category=r["category"],
            status=r["status"],
            config_overrides=r["config"],
        )

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

    # ---- FAQ
    def add_knowledge(self, item: KnowledgeItem) -> KnowledgeItem:
        created_at = item.created_at or self._clock.now()
        with self._session(item.shop_id) as conn:
            r = conn.execute(
                "INSERT INTO knowledge_item (shop_id, question, answer, source, status, item_id,"
                " intent, approved_by, created_at, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                f" RETURNING {_K_COLS}",
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
        assert r is not None
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
    ) -> KnowledgeItem:
        with self._session(shop_id) as conn:
            r = conn.execute(
                "UPDATE knowledge_item SET status = %s, approved_by = COALESCE(%s, approved_by)"
                f" WHERE shop_id = %s AND id = %s RETURNING {_K_COLS}",
                (status, approved_by, shop_id, knowledge_id),
            ).fetchone()
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
        with self._session(shop_id) as conn:
            n = 0
            for table in ("shop_policy", "item_card", "knowledge_item"):
                n += conn.execute(f"DELETE FROM {table} WHERE shop_id = %s", (shop_id,)).rowcount
            n += conn.execute("DELETE FROM shop WHERE id = %s", (shop_id,)).rowcount
            return n


# ================================================================ 会话（§5.3、§8.3）

_M_COLS = "msg_id, shop_id, conversation_id, role, text, item_id, sku_id, order_ref, created_at"


def _message(r: Row) -> StoredMessage:
    return StoredMessage(**{**r, "created_at": _cst(r["created_at"])})


class PgConversationRepository:
    """实现 core.ports.ConversationRepository（conversation / message）。

    消息按 (shop_id, msg_id) 去重（唯一约束 + ON CONFLICT DO NOTHING，并发重复推送也只存一条）。
    先写消息、再补会话：message 指向 conversation 的外键延迟到提交时检查，
    所以重复消息不会凭空建出会话。
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def add_message(self, msg: StandardMessage) -> bool:
        with self._db.session(msg.shop_id) as conn:
            inserted = conn.execute(
                f"INSERT INTO message ({_M_COLS})"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (shop_id, msg_id) DO NOTHING RETURNING id",
                (
                    msg.msg_id,
                    msg.shop_id,
                    msg.conversation_id,
                    msg.role,
                    msg.text,
                    msg.item_card.item_id if msg.item_card else None,
                    msg.item_card.sku_id if msg.item_card else None,
                    msg.order_card.order_id if msg.order_card else None,
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
        with self._db.session(shop_id) as conn:
            inserted = conn.execute(
                "INSERT INTO message (msg_id, shop_id, conversation_id, role, text, created_at)"
                " SELECT %s, %s, %s, %s, %s, %s"
                " WHERE EXISTS (SELECT 1 FROM conversation WHERE shop_id = %s AND id = %s)"
                " ON CONFLICT (shop_id, msg_id) DO NOTHING RETURNING id",
                (msg_id, shop_id, conversation_id, role, text, at, shop_id, conversation_id),
            ).fetchone()
            return inserted is not None

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
                " WHERE shop_id = %s AND conversation_id = %s"
                " ORDER BY created_at DESC, id DESC LIMIT %s"
                ") recent ORDER BY created_at, id",
                (shop_id, conversation_id, limit),
            ).fetchall()
        return [_message(r) for r in rows]

    def purge_shop(self, shop_id: str) -> int:
        with self._db.session(shop_id) as conn:
            n = conn.execute("DELETE FROM message WHERE shop_id = %s", (shop_id,)).rowcount
            n += conn.execute("DELETE FROM conversation WHERE shop_id = %s", (shop_id,)).rowcount
            return n


# ================================================================ 推荐（§5.6、§10 suggestion）

_S_FIELDS = (
    "shop_id",
    "conversation_id",
    "msg_id",
    "question_text",
    "item_id",
    "intent",
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
        with self._db.session(shop_id) as conn:
            return conn.execute("DELETE FROM suggestion WHERE shop_id = %s", (shop_id,)).rowcount


# ================================================================ 反馈（§5.8）

_GAP_COLS = "shop_id, intent, missing, question_text, source, created_at"
_LABEL_COLS = "shop_id, text, predicted_intent, true_intent, source, created_at"


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
                f"INSERT INTO labeled_sample ({_LABEL_COLS}) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    sample.shop_id,
                    sample.text,
                    sample.predicted_intent,
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

    def export_training_labels(self) -> list[LabeledSample]:
        with self._db.session(None, trainer=True) as conn:
            rows = conn.execute(
                f"SELECT {_LABEL_COLS} FROM labeled_sample ORDER BY created_at, id"
            ).fetchall()
        return [_label(r) for r in rows]

    def purge_shop(self, shop_id: str) -> int:
        with self._db.session(shop_id) as conn:
            n = conn.execute("DELETE FROM knowledge_gap WHERE shop_id = %s", (shop_id,)).rowcount
            n += conn.execute("DELETE FROM labeled_sample WHERE shop_id = %s", (shop_id,)).rowcount
            return n


# ================================================================ 售后审核（§7.3.2）

_R_FIELDS = (
    "platform_status",
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
        with self._db.session(shop_id) as conn:
            return conn.execute(
                "DELETE FROM aftersale_review WHERE shop_id = %s", (shop_id,)
            ).rowcount


# ================================================================ 授权（§8.4）


class PgShopAuthRepository:
    """实现 core.ports.ShopAuthRepository。access_token 用 pgcrypto 对称加密（AES-256）后落库。

    - encryption_key 由应用层从环境变量读（默认 RODIT_TOKEN_KEY，见 from_env），
      仓库里没有真实密钥；只有测试里写死合成的密钥。
    - 密钥作为查询参数传给 pgp_sym_encrypt / pgp_sym_decrypt，不拼进 SQL；
      repr、异常消息里都不出现密钥和令牌（令牌进出都是 SecretStr）。
    - 换密钥：用旧密钥读出、新密钥写回（MVP 未提供批量工具）。
    """

    MIN_KEY_LENGTH = 16

    def __init__(self, db: Database, *, encryption_key: str) -> None:
        if len(encryption_key or "") < self.MIN_KEY_LENGTH:
            raise ValueError(f"令牌加密密钥至少 {self.MIN_KEY_LENGTH} 个字符")
        self._db = db
        self._key = SecretStr(encryption_key)

    def __repr__(self) -> str:
        return "PgShopAuthRepository(encryption_key=***)"

    @classmethod
    def from_env(cls, db: Database, env_name: str = "RODIT_TOKEN_KEY") -> PgShopAuthRepository:
        key = os.environ.get(env_name, "")
        if not key:
            raise RuntimeError(f"环境变量 {env_name} 未设置：店铺令牌必须加密存储（方案 §8.4）")
        return cls(db, encryption_key=key)

    def get(self, shop_id: str) -> ShopAuth | None:
        with self._db.session(shop_id) as conn:
            try:
                r = conn.execute(
                    "SELECT shop_id, pgp_sym_decrypt(access_token, %s) AS token, scopes,"
                    " expires_at FROM shop_auth WHERE shop_id = %s",
                    (self._key.get_secret_value(), shop_id),
                ).fetchone()
            except psycopg.errors.ExternalRoutineInvocationException as e:
                raise TokenDecryptionError(
                    f"店铺 {shop_id} 的令牌无法解密：请检查令牌加密密钥是否与写入时一致"
                ) from e
        if r is None:
            return None
        return ShopAuth(
            shop_id=r["shop_id"],
            access_token=SecretStr(r["token"]),
            scopes=list(r["scopes"]),
            expires_at=_cst(r["expires_at"]),
        )

    def save(self, auth: ShopAuth) -> None:
        with self._db.session(auth.shop_id) as conn:
            conn.execute(
                "INSERT INTO shop_auth (shop_id, access_token, scopes, expires_at)"
                " VALUES (%s, pgp_sym_encrypt(%s, %s, 'cipher-algo=aes256'), %s, %s)"
                " ON CONFLICT (shop_id) DO UPDATE SET access_token = EXCLUDED.access_token,"
                " scopes = EXCLUDED.scopes, expires_at = EXCLUDED.expires_at",
                (
                    auth.shop_id,
                    auth.access_token.get_secret_value(),
                    self._key.get_secret_value(),
                    list(auth.scopes),
                    auth.expires_at,
                ),
            )

    def delete(self, shop_id: str) -> None:
        self.purge_shop(shop_id)

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

    shop_id 为空的事件（如没有店铺归属的大模型调用）在不设店铺范围的事务里写入，
    应用侧读不到（list_events 必须带店铺）；表是 FORCE 行级安全，连表的所有者也读不到，
    只有超级用户或带 BYPASSRLS 的运维角色能查看。
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
        """删除该店铺的审计和判断记录（§8.4 按约定删除数据时调用；是否保留由委托协议决定）。"""
        with self._db.session(shop_id) as conn:
            n = conn.execute("DELETE FROM audit_log WHERE shop_id = %s", (shop_id,)).rowcount
            n += conn.execute("DELETE FROM decision_log WHERE shop_id = %s", (shop_id,)).rowcount
            return n
