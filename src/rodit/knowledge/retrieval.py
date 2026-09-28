"""找依据（方案 §5.2.4 检索策略、§5.2.1 冲突处理、§12.2 知识库故障降级）：
实现 core.ports.EvidenceRetriever。

流程：
1. 按 core.evidence.needs_for 决定找什么；多问题（route.parts）取各问题所需依据的并集。
2. 知识依据（KNOWLEDGE_NEEDS）在这里处理，结构化优先：
   - 商品字段（标题、相关属性、SKU、尺码表 + 尺码建议、洗涤 / 使用、详情页逐句）；
     需要商品但会话里没有当前商品：候选商品多于一个 → 澄清"请问您问的是哪一款呢？"，
     否则记缺失 MISSING_ITEM；
   - 店铺政策（policy_keys_for，没填的记 policy:{key}）、平台规则（platform_rules_for，按平台）；
   - FAQ 语义检索补充（前 faq_top_k 条、分数下限 faq_min_score）；类目通用知识兜底。
3. 实时依据（TOOL_NEEDS）交给 ToolEvidenceProvider，结果合并（证据、缺失、不可用、澄清、提醒）；
   没有注入 provider 时把每种工具记为不可用（core.evidence.unavailable）。
4. 冲突处理：同一主题只保留优先级最高的来源，其余移除并记进 overridden_ids（§5.2.1）；
   详情页句子、FAQ 回答涉及店铺政策主题、而本次没取这条政策时（如商品参数类问题的详情页里写着
   旧的发货时效），把同主题政策取来只当裁判：参与冲突判断，但不作为依据输出；
   再按 GenerationConfig.max_evidence_chars 控制长度（§14）；证据按优先级排序输出。

店铺隔离（§9.2）：所有仓储调用都带 ctx.shop_id；会话里的商品卡片、仓储返回的政策 / FAQ
不属于当前店铺时抛 TenantIsolationError（宁可不推荐，也不拿别家店的知识回答）。
知识库（仓储）任何报错都转成 KnowledgeUnavailable，引擎据此不推荐（§12.2）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from rodit.core import evidence as ev
from rodit.core.config import GenerationConfig
from rodit.core.enums import POLICY_LABELS, EvidenceSource, Intent, PolicyKey
from rodit.core.errors import KnowledgeUnavailable, TenantIsolationError, ToolError
from rodit.core.evidence import KNOWLEDGE_NEEDS, TOOL_NEEDS, Need
from rodit.core.models import (
    ConversationContext,
    Evidence,
    EvidenceBundle,
    ItemCard,
    KnowledgeItem,
    PolicyEntry,
    RouteDecision,
)
from rodit.core.ports import KnowledgeRepository, ToolEvidenceProvider
from rodit.knowledge import items as item_ev
from rodit.knowledge.category import category_group, category_tip_evidence, category_tips
from rodit.knowledge.conflict import (
    Candidate,
    dedupe,
    ordered_topics,
    resolve_conflicts,
    trim_to_budget,
)
from rodit.knowledge.platform_rules import find_platform_rule, platform_rule_evidence
from rodit.knowledge.search import FaqSearcher
from rodit.knowledge.text import for_matching, minimized

_POLICY_TOPICS = frozenset(ev.topic_for_policy(k.value) for k in PolicyKey)

CLARIFY_ITEM_QUESTION = "请问您问的是哪一款呢？"
"""会话里有多个商品、又判断不出指哪个时的澄清问题（§5.2.4）。"""

DEFAULT_FAQ_MIN_SCORE = 0.3
"""FAQ 分数下限：低于它的检索结果和问题基本无关，不作为依据。"""

ITEM_NEEDS = frozenset({Need.ITEM_ATTRS, Need.SIZE, Need.CARE_USAGE, Need.ITEM_DETAIL})
"""需要"当前商品"的依据。"""

TOOL_KINDS: dict[Need, str] = {
    Need.ORDER: "order",
    Need.LOGISTICS: "logistics",
    Need.AFTERSALE: "aftersale",
    Need.RETURN_ELIGIBILITY: "return",
    Need.STOCK: "stock",
    Need.PRICE: "price",
}
"""工具依据 → core.evidence.unavailable(kind) 的 kind（与 TOOL_LABELS 的键一致）。"""

_NEED_ORDER = {n: i for i, n in enumerate(Need)}
_T = TypeVar("_T")


def route_intents(route: RouteDecision) -> list[Intent]:
    """本次检索覆盖的问题类型：route.intent + route.parts（多问题），去重保序。"""
    out: list[Intent] = []
    for intent in (route.intent, *(p.intent for p in route.parts)):
        if intent not in out:
            out.append(intent)
    return out


def _sorted_needs(needs: Iterable[Need]) -> list[Need]:
    return sorted(needs, key=_NEED_ORDER.__getitem__)


def _extend_unique(target: list[str], values: Iterable[str]) -> None:
    for v in values:
        if v not in target:
            target.append(v)


def _unique(values: Iterable[_T]) -> list[_T]:
    out: list[_T] = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


class KnowledgeRetriever:
    """知识中心的 EvidenceRetriever 实现。

    - repo：知识仓储（政策；商品卡片由会话理解放进 ctx.item）；
    - searcher：FAQ 检索（同一个仓储）；
    - config：生成配置（faq_top_k、max_evidence_chars）；
    - tool_provider：实时工具证据；为空时工具依据一律记为不可用；
    - faq_min_score：FAQ 分数下限。
    """

    def __init__(
        self,
        repo: KnowledgeRepository,
        searcher: FaqSearcher,
        config: GenerationConfig,
        *,
        tool_provider: ToolEvidenceProvider | None = None,
        faq_min_score: float = DEFAULT_FAQ_MIN_SCORE,
    ) -> None:
        self._repo = repo
        self._searcher = searcher
        self._config = config
        self._tools = tool_provider
        self._faq_min_score = faq_min_score

    # ================================================================ 入口

    def retrieve(self, ctx: ConversationContext, route: RouteDecision) -> EvidenceBundle:
        intents = route_intents(route)
        needs: set[Need] = set()
        for intent in intents:
            needs |= ev.needs_for(intent)

        cands: list[Candidate] = []
        missing: list[str] = []
        unavailable: list[str] = []
        overridden: list[str] = []
        alerts: list[str] = []
        clarify: str | None = None

        # 知识在前：知识库故障时直接抛出，不再白白调用平台接口
        if needs & KNOWLEDGE_NEEDS:
            clarify = self._collect_knowledge(ctx, intents, needs, cands, missing)

        tool_needs = needs & TOOL_NEEDS
        if tool_needs:
            if self._tools is None:
                _extend_unique(
                    unavailable, (ev.unavailable(TOOL_KINDS[n]) for n in _sorted_needs(tool_needs))
                )
            else:
                for intent in intents:
                    per_intent = ev.needs_for(intent) & TOOL_NEEDS
                    if not per_intent:
                        continue
                    try:
                        bundle = self._tools.collect(ctx, intent, frozenset(per_intent))
                    except ToolError:
                        # 协议要求 provider 自己标记不可用；万一抛出，按不可用处理（§12.2）
                        _extend_unique(
                            unavailable,
                            (ev.unavailable(TOOL_KINDS[n]) for n in _sorted_needs(per_intent)),
                        )
                        continue
                    cands.extend(Candidate.of(e) for e in bundle.evidences)
                    _extend_unique(missing, bundle.missing)
                    _extend_unique(unavailable, bundle.unavailable)
                    _extend_unique(overridden, bundle.overridden_ids)
                    _extend_unique(alerts, bundle.alerts)
                    if clarify is None:
                        clarify = bundle.clarify_question

        cands = dedupe(cands)
        referees = self._policy_referees(ctx, cands) if needs & KNOWLEDGE_NEEDS else []
        kept, lost = resolve_conflicts([*cands, *referees])
        referee_ids = {c.evidence.id for c in referees}
        kept = [c for c in kept if c.evidence.id not in referee_ids]
        _extend_unique(overridden, lost)
        kept = trim_to_budget(kept, self._config.max_evidence_chars)
        kept.sort(key=lambda c: c.evidence.priority)  # 稳定排序：同优先级保持检索顺序
        return EvidenceBundle(
            evidences=[c.evidence for c in kept],
            missing=missing,
            unavailable=unavailable,
            overridden_ids=overridden,
            clarify_question=clarify,
            alerts=alerts,
        )

    # ================================================================ 知识依据

    def _collect_knowledge(
        self,
        ctx: ConversationContext,
        intents: list[Intent],
        needs: set[Need],
        cands: list[Candidate],
        missing: list[str],
    ) -> str | None:
        """按结构化优先的顺序收集知识证据；返回商品澄清问题（没有则 None）。"""
        question = ctx.current.text
        card = ctx.item
        if card is not None and card.shop_id != ctx.shop_id:
            raise TenantIsolationError(
                f"会话商品 {card.item_id} 属于店铺 {card.shop_id}，不属于 {ctx.shop_id}"
            )
        clarify: str | None = None

        if needs & ITEM_NEEDS:
            if card is not None:
                self._item_evidence(card, ctx.sku_id, question, needs, cands, missing)
            elif len(set(ctx.candidate_item_ids)) > 1:
                clarify = CLARIFY_ITEM_QUESTION
            else:
                _extend_unique(missing, [ev.MISSING_ITEM])

        if Need.PLATFORM_RULE in needs:
            keys = _unique(k for i in intents for k in ev.platform_rules_for(i))
            for key in keys:
                if find_platform_rule(ctx.platform, key) is None:
                    _extend_unique(missing, [ev.missing_platform_rule(ctx.platform, key)])
                else:
                    cands.append(Candidate(platform_rule_evidence(ctx.platform, key)))

        if Need.POLICY in needs:
            keys = _unique(k for i in intents for k in ev.policy_keys_for(i))
            if keys:
                policies = self._repo_call(self._repo.get_policies, ctx.shop_id)
                for key in keys:
                    entry = policies.get(key.value)
                    if entry is not None and entry.shop_id != ctx.shop_id:
                        raise TenantIsolationError(
                            f"知识仓储返回了店铺 {entry.shop_id} 的政策（查询店铺 {ctx.shop_id}）"
                        )
                    if entry is None or not entry.value.strip():
                        _extend_unique(missing, [ev.missing_policy(key.value)])
                    else:
                        cands.append(Candidate(_policy_evidence(entry)))

        if Need.FAQ in needs:
            hits = self._repo_call(
                self._searcher.search,
                ctx.shop_id,
                question,
                item_id=card.item_id if card is not None else None,
                top_k=self._config.faq_top_k,
                min_score=self._faq_min_score,
            )
            cands.extend(_faq_candidate(k, score) for k, score in hits)

        if Need.CATEGORY in needs:
            group = (category_group(card.category) if card is not None else None) or (
                category_group(ctx.shop.category)
            )
            for tip in category_tips(group, for_matching(question)):
                cands.append(Candidate.of(category_tip_evidence(tip), rank=0.0))

        return clarify

    def _item_evidence(
        self,
        card: ItemCard,
        sku_id: str | None,
        question: str,
        needs: set[Need],
        cands: list[Candidate],
        missing: list[str],
    ) -> None:
        if Need.ITEM_ATTRS in needs:
            cands.append(item_ev.title_candidate(card))
            attrs, lost = item_ev.attribute_candidates(card, question)
            cands.extend(attrs)
            _extend_unique(missing, lost)
            sku = item_ev.sku_candidate(card, sku_id)
            if sku is not None:
                cands.append(sku)
        if Need.SIZE in needs:
            sized, lost = item_ev.size_candidates(card)
            cands.extend(sized)
            _extend_unique(missing, lost)
        if Need.CARE_USAGE in needs:
            cared, lost = item_ev.care_usage_candidates(card, question)
            cands.extend(cared)
            _extend_unique(missing, lost)
        if Need.ITEM_DETAIL in needs:
            cands.extend(item_ev.detail_candidates(card, question))

    def _policy_referees(self, ctx: ConversationContext, cands: list[Candidate]) -> list[Candidate]:
        """低优先级证据（详情页句子、FAQ 回答）涉及的店铺政策主题、本次又没取这条政策时，
        取同主题政策当裁判（§5.2.1 以店铺政策为准）。政策没填不记缺失（本次问题不需要它）。"""
        fetched = {c.evidence.id for c in cands}
        policy_priority = EvidenceSource.POLICY.priority
        topics = {
            t for c in cands if c.evidence.priority > policy_priority for t in c.topics
        } & _POLICY_TOPICS
        keys = [t for t in ordered_topics(topics) if ev.policy(t) not in fetched]
        if not keys:
            return []
        policies = self._repo_call(self._repo.get_policies, ctx.shop_id)
        referees: list[Candidate] = []
        for key in keys:
            entry = policies.get(key)
            if entry is None or not entry.value.strip():
                continue
            if entry.shop_id != ctx.shop_id:
                raise TenantIsolationError(
                    f"知识仓储返回了店铺 {entry.shop_id} 的政策（查询店铺 {ctx.shop_id}）"
                )
            referees.append(Candidate(_policy_evidence(entry)))
        return referees

    @staticmethod
    def _repo_call(fn: Callable[..., _T], *args: object, **kwargs: object) -> _T:
        """仓储（含 FAQ 检索里的仓储读写）任何报错 → KnowledgeUnavailable（§12.2 不推荐）。"""
        try:
            return fn(*args, **kwargs)
        except (KnowledgeUnavailable, TenantIsolationError):
            raise
        except Exception as e:
            raise KnowledgeUnavailable(f"知识库不可用：{type(e).__name__}") from e


# ================================================================ 证据构造


def _policy_evidence(entry: PolicyEntry) -> Evidence:
    """policy:{key}，data={"key","value"}，topic=key。

    与详情页、FAQ 回答里同主题的句子冲突时政策优先（§5.2.1）。
    """
    try:
        name = POLICY_LABELS[PolicyKey(entry.key)]
    except ValueError:
        name = entry.key
    return Evidence(
        id=ev.policy(entry.key),
        source=EvidenceSource.POLICY,
        label=ev.policy_label(entry.key),
        content=minimized(f"{name}：{entry.value}"),
        data={"key": entry.key, "value": entry.value},
        topic=ev.topic_for_policy(entry.key),
    )


def _faq_candidate(item: KnowledgeItem, score: float) -> Candidate:
    """faq:{id}，data={"question","answer","faq_id"}，score=检索分数。

    topic 取回答里检测到的主题（detect_topics）：回答涉及发货时效、运费险等政策主题时，
    会被同主题的店铺政策覆盖——政策改了而 FAQ 没改时，不会用过时的 FAQ 回答（§5.2.1）。
    """
    assert item.id is not None
    topics = ordered_topics(ev.detect_topics(item.answer))
    evidence = Evidence(
        id=ev.faq(item.id),
        source=EvidenceSource.FAQ,
        label=ev.faq_label(item.question),
        content=minimized(f"问：{item.question}\n答：{item.answer}"),
        data={"question": item.question, "answer": item.answer, "faq_id": item.id},
        topic=topics[0] if topics else None,
        score=score,
    )
    return Candidate(evidence, topics=frozenset(topics), rank=score)
