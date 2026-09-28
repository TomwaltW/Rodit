"""FAQ 语义检索（方案 §5.2.4 "语义检索补充"、§5.4 FAQ 直答、§9.2 向量索引按店铺隔离）。

- 只检索商家审核通过（APPROVED）的 FAQ；候选（客服采纳、商家未审核）和已下线的不参与（§5.8）。
- 商品专属 FAQ 只在问到该商品时出现：item_id=None 只找通用 FAQ，item_id=X 找通用 + 商品 X 的
  （与 KnowledgeRepository.search_knowledge 的范围一致）。
- 向量由注入的 Embedder 对 FAQ 的"问题"计算，存进仓储（set_knowledge_embedding）；每次检索前
  先给缺向量（或维度和当前 Embedder 不一致）的已审核 FAQ 补向量（懒索引），新审核通过的 FAQ
  下一次检索就能找到。
- 查询先脱敏（§5.3，knowledge.text.mask_text = mask_pii + 本地补丁）：向量服务也是外部依赖，
  买家打出来的手机号、地址不发出去。

分数语义（Evidence.score，生成模块拿它和 GenerationConfig.faq_direct_min_score 比较，
决定能否直接用商家审核过的标准回答）：

- 归一化后与 FAQ 问题完全相同 → EXACT_MATCH_SCORE（1.0）。归一化 = 脱敏、NFKC、小写、去掉
  开头的称呼寒暄（亲、你好、在吗、请问……）、去掉标点空白、去掉句末语气词（吗、呢、呀……）。
  例："亲，什么时候发货呀？" 与 "什么时候发货" 相同。
- 其余 → 向量余弦相似度，截断到 [0, SEMANTIC_SCORE_CAP]（0.99）。**不做任何字面加分**：
  包含关系（"什么时候发货到新疆" 包含 "什么时候发货"）、关键词重合都不加分。所以换一种说法的
  问题能不能达到直答阈值只取决于向量相似度，1.0 只留给字面相同的问题。

降级（§12.2）：向量服务报错时退回只做字面匹配（完全相同的问题仍然命中，其余不返回），
不影响结构化检索；仓储报错照常抛出（由检索器转成 KnowledgeUnavailable）。
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence

from rodit.core.enums import KnowledgeStatus
from rodit.core.errors import TenantIsolationError
from rodit.core.models import KnowledgeItem
from rodit.core.ports import Embedder, KnowledgeRepository
from rodit.knowledge.text import mask_text

logger = logging.getLogger(__name__)

EXACT_MATCH_SCORE = 1.0
"""归一化后问题完全相同的分数。"""

SEMANTIC_SCORE_CAP = 0.99
"""向量相似度的上限：1.0 只留给字面相同的问题。"""

# 开头的称呼、寒暄：称呼类后面必须跟标点或空白（避免把"亲子装"的"亲"去掉）
_LEADING_FILLER = re.compile(
    r"^(?:(?:亲爱的|亲们|亲|你好|您好|在吗|在不在|在么|hello|hi|嗨|哈喽)[\s,，.。!！~～、?？]+"
    r"|请问一?下?|麻烦问一?下|想问一?下|问一?下)"
)
_TRAILING_PARTICLES = re.compile(r"(?:吗|呢|呀|啊|哈|哦|噢|喔|吖|嘛|啦|哇)+$")
_NON_WORD = re.compile(r"[\W_]+")


def _own_approved(shop_id: str, items: Sequence[KnowledgeItem]) -> list[KnowledgeItem]:
    """防御性检查（§9.2）：仓储实现返回了别家店的 FAQ 就抛 TenantIsolationError（宁可不推荐）；
    顺带去掉非 APPROVED 的记录。"""
    for k in items:
        if k.shop_id != shop_id:
            raise TenantIsolationError(
                f"知识仓储返回了店铺 {k.shop_id} 的 FAQ（查询店铺 {shop_id}）"
            )
    return [k for k in items if k.status == KnowledgeStatus.APPROVED]


def normalize_question(text: str) -> str:
    """问题的字面归一化（判断"完全相同"用）。先脱敏，保证比较时不依赖个人信息。"""
    s = unicodedata.normalize("NFKC", mask_text(text or "")).lower().strip()
    while True:
        stripped = _LEADING_FILLER.sub("", s, count=1).lstrip()
        if stripped == s:
            break
        s = stripped
    s = _NON_WORD.sub("", s)
    return _TRAILING_PARTICLES.sub("", s)


class FaqSearcher:
    """FAQ 检索：懒索引 + 向量检索 + 字面完全匹配。线程安全（无内部可变状态）。"""

    def __init__(self, repo: KnowledgeRepository, embedder: Embedder) -> None:
        self._repo = repo
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    # ---------------------------------------------------------------- 索引

    def index(self, shop_id: str) -> int:
        """给该店缺向量的已审核 FAQ 计算并保存向量（对"问题"文字），返回新写入的条数。

        已有向量但维度和当前 Embedder 不一致的（换了向量模型）也重新计算。
        向量服务或仓储报错时直接抛出。
        """
        items = _own_approved(
            shop_id, self._repo.list_knowledge(shop_id, status=KnowledgeStatus.APPROVED)
        )
        todo = self._needs_embedding(items)
        if not todo:
            return 0
        vectors = self._embedder.embed([mask_text(k.question) for k in todo])
        if len(vectors) != len(todo):
            raise ValueError(f"Embedder 返回 {len(vectors)} 条向量，期望 {len(todo)} 条")
        self._store(shop_id, todo, vectors)
        return len(todo)

    def _needs_embedding(self, items: Sequence[KnowledgeItem]) -> list[KnowledgeItem]:
        dim = self._embedder.dim
        return [
            k
            for k in items
            if k.id is not None and (k.embedding is None or len(k.embedding) != dim)
        ]

    def _store(
        self, shop_id: str, items: Sequence[KnowledgeItem], vectors: Sequence[Sequence[float]]
    ) -> None:
        for item, vec in zip(items, vectors, strict=True):
            assert item.id is not None
            self._repo.set_knowledge_embedding(shop_id, item.id, list(vec))

    # ---------------------------------------------------------------- 检索

    def search(
        self,
        shop_id: str,
        query: str,
        *,
        item_id: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[KnowledgeItem, float]]:
        """返回 [(FAQ, 分数)]，分数降序（同分按 id 升序），只含分数 ≥ min_score 的前 top_k 条。

        分数语义见模块文档：字面相同 1.0，其余是向量相似度（≤ 0.99）。
        """
        if top_k <= 0:
            return []
        masked = mask_text(query or "")
        norm = normalize_question(masked)
        if not norm:
            return []

        items = _own_approved(
            shop_id, self._repo.list_knowledge(shop_id, status=KnowledgeStatus.APPROVED)
        )
        scored: dict[int, tuple[KnowledgeItem, float]] = {}
        for k in items:
            in_scope = k.item_id is None or k.item_id == item_id
            if in_scope and k.id is not None and normalize_question(k.question) == norm:
                scored[k.id] = (k, EXACT_MATCH_SCORE)

        todo = self._needs_embedding(items)
        vectors = self._embed_or_none([*(mask_text(k.question) for k in todo), masked])
        if vectors is not None:
            self._store(shop_id, todo, vectors[:-1])
            hits = self._repo.search_knowledge(shop_id, vectors[-1], top_k=top_k, item_id=item_id)
            approved = {id(k) for k in _own_approved(shop_id, [k for k, _ in hits])}
            for k, score in hits:
                if id(k) not in approved or k.id is None or k.id in scored:
                    continue
                if k.item_id is not None and k.item_id != item_id:
                    continue
                scored[k.id] = (k, min(max(score, 0.0), SEMANTIC_SCORE_CAP))

        ranked = sorted(scored.values(), key=lambda pair: (-pair[1], pair[0].id or 0))
        return [(k, s) for k, s in ranked if s >= min_score][:top_k]

    def _embed_or_none(self, texts: list[str]) -> list[list[float]] | None:
        """向量服务故障时返回 None（退回字面匹配）；日志里不记查询原文。"""
        try:
            vectors = self._embedder.embed(texts)
        except Exception:
            logger.warning("向量服务不可用，FAQ 检索退回字面匹配", exc_info=True)
            return None
        if len(vectors) != len(texts):
            logger.warning(
                "向量服务返回条数不对（%d / %d），FAQ 检索退回字面匹配", len(vectors), len(texts)
            )
            return None
        return vectors
