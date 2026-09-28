"""证据冲突处理与长度预算（方案 §5.2.1 "高优先级覆盖低优先级"、§14 "控制证据长度"）。

冲突（core.evidence 模块文档）：同一 topic 下，低优先级证据和更高优先级依据内容冲突时
（core.evidence.contradicts）从证据里移除、id 记进 overridden_ids；内容一致时两者都保留
（FAQ 标准回答仍可直接用，§5.4）。类目通用知识遇到同主题的更具体来源一律让位。
同一来源（同优先级）之间不互相覆盖。
自由文本（详情页一句、FAQ 回答）可能同时涉及几个主题（如"24 小时内发货，默认发中通"）：
内部用 Candidate.topics 记全部主题，任一主题被更高优先级的来源占据就移除这一句
（宁可少说，不说和高优先级依据矛盾的话）；对外的 Evidence.topic 只写第一个主题。

长度预算：证据在提示词【依据】里占的字数（core.evidence_line：每行 "[id] 标签：内容"，
含获取时间和行间换行，和 generation.prompt 按同一个字符串计长）超过 max_chars 时，
按"优先级低的先删；同优先级里 rank 低的先删；再相同则靠后的先删"逐条删除，
实时数据、平台规则、店铺政策优先级最高，最后才删；
至少保留一条（单条超长时宁可超预算也不交空证据）。
这样检索按相关度留下的依据，提示词都放得下，不会再按位置截掉（例如长详情页里最相关的那一句）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rodit.core import evidence as ev
from rodit.core.enums import EvidenceSource, PolicyKey
from rodit.core.evidence_line import evidence_line_chars
from rodit.core.models import Evidence

STRUCTURED_RANK = 2.0
"""结构化证据（商品字段、政策、平台规则、实时数据）的 rank：高于任何相似度分数，
同优先级里比详情页文字、FAQ 更晚被删。"""

_POLICY_ORDER = {key.value: i for i, key in enumerate(PolicyKey)}


def ordered_topics(topics: set[str] | frozenset[str]) -> list[str]:
    """主题按 PolicyKey 的声明顺序排列（其余主题按字母序排在后面），保证结果确定。"""
    return sorted(topics, key=lambda t: (_POLICY_ORDER.get(t, len(_POLICY_ORDER)), t))


@dataclass(frozen=True)
class Candidate:
    """检索过程中的一条候选证据。

    - topics：用于冲突处理的全部主题（Evidence.topic 只能放一个）；evidence.topic 总会被并进来，
      构造时只需要传额外的主题；
    - rank：同优先级内的保留强度，越大越晚被预算删掉（FAQ 用相似度，详情页用与问题的字面相关度）。
    """

    evidence: Evidence
    topics: frozenset[str] = frozenset()
    rank: float = STRUCTURED_RANK

    def __post_init__(self) -> None:
        if self.evidence.topic and self.evidence.topic not in self.topics:
            object.__setattr__(self, "topics", self.topics | {self.evidence.topic})

    @classmethod
    def of(cls, evidence: Evidence, *, rank: float | None = None) -> Candidate:
        """外部来的证据（如工具证据）：topics 取 evidence.topic，rank 默认取 score。"""
        if rank is None:
            rank = evidence.score if evidence.score is not None else STRUCTURED_RANK
        return cls(evidence=evidence, rank=rank)


def dedupe(cands: Sequence[Candidate]) -> list[Candidate]:
    """同一证据 ID 只保留第一次出现的。"""
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in cands:
        if c.evidence.id not in seen:
            seen.add(c.evidence.id)
            out.append(c)
    return out


def resolve_conflicts(cands: Sequence[Candidate]) -> tuple[list[Candidate], list[str]]:
    """返回 (保留的候选，保持原顺序；被覆盖的证据 ID，按原顺序)。"""
    best: dict[str, int] = {}
    for c in cands:
        for topic in c.topics:
            p = c.evidence.priority
            if topic not in best or p < best[topic]:
                best[topic] = p
    kept: list[Candidate] = []
    overridden: list[str] = []
    for c in cands:
        losing = {t for t in c.topics if best[t] < c.evidence.priority}
        if not losing:
            kept.append(c)
            continue
        # 类目通用知识是泛泛的建议：同主题有更具体的来源就不用它
        if c.evidence.source == EvidenceSource.CATEGORY:
            overridden.append(c.evidence.id)
            continue
        # 其余来源：只有内容冲突才覆盖（core.evidence.contradicts），一致时两者都保留
        higher = [o for o in cands if o.evidence.priority < c.evidence.priority]
        same_topic = [o.evidence.content for o in higher if o.topics & losing]
        if ev.contradicts(
            c.evidence.content,
            reference=[o.evidence.content for o in higher],
            same_topic=same_topic,
        ):
            overridden.append(c.evidence.id)
        else:
            kept.append(c)
    return kept, overridden


def content_chars(cands: Sequence[Candidate]) -> int:
    return sum(len(c.evidence.content) for c in cands)


def line_chars(cands: Sequence[Candidate]) -> int:
    """这些证据在提示词【依据】里占的字数：各行字数 + 行间换行（和 generation.prompt 一致）。"""
    if not cands:
        return 0
    return sum(evidence_line_chars(c.evidence) for c in cands) + len(cands) - 1


def trim_to_budget(cands: Sequence[Candidate], max_chars: int) -> list[Candidate]:
    """按预算删证据，返回保留的候选（保持原顺序）。预算按提示词里的整行计（line_chars）。"""
    # 每行连同它前面的换行一起算，总数比 line_chars 多 1
    costs = [evidence_line_chars(c.evidence) + 1 for c in cands]
    total = sum(costs)
    limit = max_chars + 1
    if total <= limit:
        return list(cands)
    drop_order = sorted(
        range(len(cands)),
        key=lambda i: (-cands[i].evidence.priority, cands[i].rank, -i),
    )
    dropped: set[int] = set()
    for i in drop_order:
        if total <= limit or len(cands) - len(dropped) <= 1:
            break
        dropped.add(i)
        total -= costs[i]
    return [c for i, c in enumerate(cands) if i not in dropped]
