"""推荐卡片（方案 §5.6）：回答、依据来源、置信度（高 / 中）、风险提示。

- text 用质检清洗后的文本（QAResult.text），不是草稿原文；
- 依据来源按草稿引用的 evidence_ids 从证据包里取标签（"商品属性 · 面料""店铺政策 · 发货时效"）；
- 置信度：FAQ / 模板，或大模型回答的 grounded ≥ 0.9，并且问题类型置信度 ≥ 0.85 → 高，其余中；
- 风险提示 = 路由的风险提示 + 证据包的提醒（物流停滞等）+ 降级说明（去重保序）。

推荐记录 → 卡片（card_from_record）只在撤回后刷新侧边栏时用（§5.3）：记录里没有问题类型置信度、
依据标签和风险提示，置信度按"中"、依据来源只列 ID，风险提示换成"请核实"。
"""

from __future__ import annotations

from collections.abc import Iterable

from rodit.core.enums import Confidence, DraftKind, Intent
from rodit.core.evidence import TOOL_LABELS
from rodit.core.models import (
    NON_BLOCKING_HARD_RULES,
    Draft,
    EvidenceBundle,
    EvidenceRefView,
    QAResult,
    RouteDecision,
    SuggestionCard,
    SuggestionRecord,
)
from rodit.core.questions import Q_GROUNDED

HIGH_GROUNDED_MIN = 0.9  # 大模型回答 grounded 达到这个值才算"高"
HIGH_INTENT_MIN = 0.85  # 问题类型置信度达到这个值才算"高"

HARD_RULE_LABELS: dict[str, str] = {
    "off_platform": "站外引流",
    "absolute_term": "绝对化用语",
    "banned_word": "商家禁用词",
    "forbidden_claim": "不能说的卖点",
    "number_mismatch": "数字与依据不一致",
    "time_promise": "时效承诺",
    "length": "超长",
    "format": "格式",
    "unknown_evidence": "引用了不存在的依据",
}

_DEGRADATION_NOTES: dict[str, str] = {
    "decision": "判断层暂不可用，问题类型按关键词规则判断，请核实",
    "decision:qa": "判断层质检暂不可用，只做了硬规则检查，请核实",
    "storage": "会话记录暂时读不到，上下文可能不全，请核实",
    "knowledge": "知识库暂时读不全，请核实",
    "vision": "买家发的图片没能识别，推荐没有参考图片内容，请先看图再发送",
}


def dedupe(values: Iterable[str]) -> list[str]:
    """去重保序，去掉空字符串。"""
    out: list[str] = []
    for v in values:
        if v and v not in out:
            out.append(v)
    return out


def degradation_note(code: str) -> str | None:
    """降级标记 → 给客服看的风险提示；和卡片内容无关的标记（如 generator、audit）返回 None。"""
    if code.startswith("tool:"):
        kind = code.split(":", 1)[1]
        return f"实时数据暂时查不到（{TOOL_LABELS.get(kind, kind)}），请核实后再发送"
    return _DEGRADATION_NOTES.get(code)


def card_confidence(kind: DraftKind, qa: QAResult, intent_confidence: float) -> Confidence:
    """§5.6 置信度：高 / 中。"""
    trusted = kind in (DraftKind.FAQ, DraftKind.TEMPLATE) or (
        qa.scores.get(Q_GROUNDED, 0.0) >= HIGH_GROUNDED_MIN
    )
    if trusted and intent_confidence >= HIGH_INTENT_MIN:
        return Confidence.HIGH
    return Confidence.MEDIUM


def evidence_views(evidence_ids: Iterable[str], bundle: EvidenceBundle) -> list[EvidenceRefView]:
    """草稿引用的依据 → 展示给客服的来源（找不到标签时用 ID 本身）。"""
    views: list[EvidenceRefView] = []
    for eid in dedupe(evidence_ids):
        ev = bundle.get(eid)
        views.append(EvidenceRefView(id=eid, label=ev.label if ev is not None else eid))
    return views


def build_card(
    *,
    draft: Draft,
    qa: QAResult,
    bundle: EvidenceBundle,
    route: RouteDecision,
    intent: Intent,
    intent_confidence: float,
    degraded: Iterable[str],
) -> SuggestionCard:
    notes = [degradation_note(c) for c in degraded]
    return SuggestionCard(
        text=qa.text,
        intent=intent,
        kind=draft.kind,
        evidence=evidence_views(draft.evidence_ids, bundle),
        confidence=card_confidence(draft.kind, qa, intent_confidence),
        risk_hints=dedupe([*route.risk_hints, *bundle.alerts, *(n for n in notes if n)]),
        qa_scores=dict(qa.scores),
    )


RECALL_REFRESH_HINT = "买家撤回了一条消息，推荐已刷新，请核实后再发送"


def card_from_record(record: SuggestionRecord) -> SuggestionCard:
    """仍在展示中的推荐记录 → 侧边栏卡片（撤回后刷新，§5.3）。记录里没存的信息不替它补：
    置信度按"中"（没法复核的卡片不声称高）、依据来源只列 ID、风险提示提醒客服核实。"""
    return SuggestionCard(
        suggestion_id=record.id,
        text=record.text,
        intent=record.intent,
        kind=record.kind,
        evidence=[EvidenceRefView(id=e, label=e) for e in dedupe(record.evidence_ids)],
        confidence=Confidence.MEDIUM,
        risk_hints=[RECALL_REFRESH_HINT],
        qa_scores=dict(record.qa_scores or {}),
    )


def blocking_rules(qa: QAResult) -> list[str]:
    """拦截性的硬规则命中（format / length 是自动修正，不算）。"""
    return dedupe(h.rule for h in qa.hard_hits if h.rule not in NON_BLOCKING_HARD_RULES)


def dropped_draft_alert(draft: Draft, qa: QAResult) -> str:
    """FAQ / 模板没过质检时给客服的提醒：标准回答本身有问题，要改知识库，重写也没用。"""
    if draft.kind == DraftKind.FAQ:
        source = f"FAQ #{draft.faq_id}" if draft.faq_id is not None else "FAQ 标准回答"
    else:
        source = f"模板 {draft.template_id}" if draft.template_id else "标准话术"
    rules = [HARD_RULE_LABELS.get(r, r) for r in blocking_rules(qa)]
    why = "、".join(rules) or "；".join(dedupe(qa.reasons)) or "质检未通过"
    return f"{source}未通过质检（{why}），本次不推荐，请检查并修改知识库"
