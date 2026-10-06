"""一条买家消息的处理过程（"AI 是怎么想的"）：懂 → 找 → 写 → 判 → 兜底，外加各环节耗时。

来源：
- EngineResult：路由结果（问题类型、概率、需要人工、情绪、多问题、命中的 §6.6 规则）、推荐、
  提醒、转人工、缺失知识、降级、各环节耗时；
- 判断日志（AuditLog.list_decisions，这条消息新写的）：路由、质检用的判断层实现和题目版本；
- 审计（engine.result，这条消息新写的）：硬规则命中和质检原因的汇总；
- 审计（engine.understand，这条消息新写的）：买家图片的张数、识别失败 / 没识别的张数；
- 过程记录（showcase.recorder.Recording）：路由看到的图片识别结果和描述、每条依据的内容、每份草稿、
  每份草稿的质检结果。

只放给讲解用的字段：不放 buyer_ref、买家昵称、会话里的其他买家消息；依据内容是引擎给大模型看的
那份（已做个人信息最小化，§5.3）；图片的识别结果（摘要、图中文字）是图片理解打过码的，这里
再过一次 mask_pii。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rodit.api.web import CONFIDENCE_LABELS, DEGRADED_LABELS, DRAFT_KIND_LABELS, HARD_RULE_LABELS
from rodit.core import evidence as ev
from rodit.core.audit import AuditAction
from rodit.core.clock import fmt_time
from rodit.core.config import EngineConfig
from rodit.core.enums import IMAGE_KIND_LABELS, INTENT_LABELS, DecisionStage, Intent, RouteAction
from rodit.core.models import (
    NON_BLOCKING_HARD_RULES,
    AuditEvent,
    DecisionLogEntry,
    EngineResult,
    Evidence,
    QAResult,
    SuggestionCard,
)
from rodit.core.pii import mask_pii
from rodit.core.ports import KnowledgeRepository
from rodit.engine.images import ALERT_VISION_FAILED, DEGRADED_VISION, ROUTING_MIN_CONFIDENCE
from rodit.knowledge import describe_gap
from rodit.showcase.images import tile_view
from rodit.showcase.labels import (
    E2E_TARGET_MS,
    LATENCY_STAGES,
    PROVIDER_LABELS,
    QA_QUESTION_LABELS,
    ROUTE_ACTION_LABELS,
    SOURCE_LABELS,
    emotion_label,
    qa_thresholds,
    rule_texts,
)
from rodit.showcase.recorder import Recording

_INTENT_VALUES = frozenset(i.value for i in Intent)

CONTENT_CHARS = 90
"""依据内容在页面上最多显示多少字。"""


def _short(text: str, limit: int = CONTENT_CHARS) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def intent_label(intent: Intent | str | None) -> str:
    if intent is None:
        return "—"
    try:
        return INTENT_LABELS[Intent(intent)]
    except ValueError:
        return str(intent)


def gap_label(marker: str) -> str:
    """缺失知识 / 上下文的说明（knowledge.describe_gap，另补"会话里没有订单"）。"""
    if marker == ev.MISSING_ORDER:
        return "会话里没有订单（模式 C 话术助手查不到订单）"
    head, _, rest = marker.partition(":")
    if head in ("intent", "knowledge") and rest:
        return f"{intent_label(rest)}类问题的知识"
    label = describe_gap(marker)
    # 客服反馈"知识缺失"时只知道商品和问题类型（feedback.service：item:{商品}:{问题类型}）；
    # describe_gap 不认识的字段名原样返回，这时按问题类型说
    item_id, _, field = rest.partition(":")
    if head == "item" and field in _INTENT_VALUES and label.endswith(f" · {field}"):
        return f"商品 {item_id} · {intent_label(field)}类问题的知识"
    return label


EXTRA_DEGRADED_LABELS: dict[str, str] = {DEGRADED_VISION: "图片理解（没能识别，请客服自己看图）"}
"""商家控制台的降级中文名里还没有的（图片理解是 §15 Phase 3 新加的）。"""


def degraded_label(marker: str) -> str:
    if marker.startswith("tool:"):
        kind = marker.removeprefix("tool:")
        return f"{DEGRADED_LABELS['tool']}（{ev.TOOL_LABELS.get(kind, kind)}）"
    return DEGRADED_LABELS.get(marker) or EXTRA_DEGRADED_LABELS.get(marker, marker)


def card_view(card: SuggestionCard) -> dict[str, Any]:
    """推荐卡片（§5.6）：可编辑的回答、依据来源、置信度、风险提示、草稿类型。"""
    return {
        "suggestion_id": card.suggestion_id,
        "text": card.text,
        "intent": card.intent.value,
        "intent_label": intent_label(card.intent),
        "kind": card.kind.value,
        "kind_label": DRAFT_KIND_LABELS[card.kind],
        "confidence": card.confidence.value,
        "confidence_label": CONFIDENCE_LABELS[card.confidence],
        "evidence": [{"id": e.id, "label": e.label} for e in card.evidence],
        "risk_hints": list(card.risk_hints),
        "qa_scores": dict(card.qa_scores),
        "prefill": card.prefill,  # 半自动（§5.6）：可以直接填进客服输入框
        "status": "shown",
    }


def is_image_alert(alert: str) -> bool:
    """引擎给客服的图片提醒（engine.images：图片摘要、没能识别、没开图片理解、只识别了前几张）。"""
    return alert.startswith("买家发来") or alert == ALERT_VISION_FAILED


def images_view(
    result: EngineResult, recording: Recording, events: Sequence[AuditEvent]
) -> dict[str, Any] | None:
    """买家图片的识别（§15 Phase 3）：几张、每张的类型 / 置信度 / 摘要 / 图中文字（已打码）、
    接在买家消息后面给路由和生成看的描述、识别失败的降级和给客服的提醒。没有图片时返回 None。"""
    understand = next(
        (e for e in reversed(events) if e.action == AuditAction.UNDERSTAND.value), None
    )
    detail = understand.detail if understand is not None else {}
    total = int(detail.get("images") or 0)
    if not total:
        return None
    routed = recording.routed[0] if recording.routed else None
    findings = routed.image_findings if routed else []
    return {
        "total": total,
        "failed": int(detail.get("images_failed") or 0),
        "skipped": int(detail.get("images_skipped") or 0),
        "findings": [
            {
                **tile_view(f.ref),
                "kind": f.kind.value,
                "kind_label": IMAGE_KIND_LABELS[f.kind],
                "confidence": round(f.confidence, 3),
                "summary": mask_pii(f.summary) if f.summary else "",
                "extracted_text": mask_pii(f.extracted_text) if f.extracted_text else None,
                "routed": f.confidence >= ROUTING_MIN_CONFIDENCE,
            }
            for f in findings
        ],
        "min_confidence": ROUTING_MIN_CONFIDENCE,
        "description": mask_pii(routed.image_description)
        if routed and routed.image_description
        else None,
        "degraded": DEGRADED_VISION in result.degraded,
        "degraded_label": degraded_label(DEGRADED_VISION),
        "alerts": [a for a in result.alerts if is_image_alert(a)],
    }


def describe_evidence(
    knowledge: KnowledgeRepository, shop_id: str, evidence_id: str
) -> tuple[str, str] | None:
    """被高优先级覆盖、已经从证据包里拿掉的依据：按 ID 找回它的标签和内容（只认详情页句子、
    商品属性、店铺政策、FAQ 四种；其余返回 None）。"""
    parts = evidence_id.split(":")
    if parts[0] == "item" and len(parts) >= 3:
        card = knowledge.get_item_card(shop_id, parts[1])
        if card is None:
            return None
        if parts[2] == "detail" and len(parts) == 4 and parts[3].isdigit():
            sentences = ev.split_sentences(card.detail_text or "")
            seg = int(parts[3])
            return ("商品详情 · 详情页", sentences[seg]) if seg < len(sentences) else None
        if parts[2] == "attr" and len(parts) == 4 and parts[3] in card.attributes:
            return f"商品属性 · {parts[3]}", card.attributes[parts[3]]
        return None
    if parts[0] == "policy" and len(parts) == 2:
        entry = knowledge.get_policies(shop_id).get(parts[1])
        return (ev.policy_label(parts[1]), entry.value) if entry else None
    if parts[0] == "faq" and len(parts) == 2 and parts[1].isdigit():
        item = knowledge.get_knowledge(shop_id, int(parts[1]))
        return (ev.faq_label(item.question), item.answer) if item else None
    return None


def _evidence_view(e: Evidence, cited: set[str]) -> dict[str, Any]:
    return {
        "id": e.id,
        "label": e.label,
        "source": e.source.value,
        "source_label": SOURCE_LABELS[e.source],
        "priority": e.priority,
        "content": _short(e.content),
        "as_of": f"截至 {fmt_time(e.fetched_at)}" if e.fetched_at else None,
        "cited": e.id in cited,
    }


def _qa_view(result: QAResult, kind: str, text: str, attempt: int, cfg: EngineConfig) -> dict:
    thresholds = qa_thresholds(cfg.qa_gate)
    scores = []
    for key, value in result.scores.items():
        limit = thresholds.get(key, {})
        ok = True
        if "min" in limit:
            ok = value >= float(limit["min"])
        elif "max" in limit:
            ok = value <= float(limit["max"])
        scores.append(
            {
                "key": key,
                "label": QA_QUESTION_LABELS.get(key, key),
                "value": round(value, 3),
                "min": limit.get("min"),
                "max": limit.get("max"),
                "scale": 4 if key == "tone" else 1,
                "ok": ok,
            }
        )
    return {
        "attempt": attempt,
        "text": text,
        "kind": kind,
        "kind_label": DRAFT_KIND_LABELS.get(kind, kind),  # type: ignore[call-overload]
        "passed": result.passed,
        "hard_hits": [
            {
                "rule": h.rule,
                "label": HARD_RULE_LABELS.get(h.rule, h.rule),
                "detail": h.detail,
                "blocking": h.rule not in NON_BLOCKING_HARD_RULES,
            }
            for h in result.hard_hits
        ],
        "scores": scores,
        "model_checked": result.model_checked,
        "provider": result.provider,
        "provider_label": PROVIDER_LABELS.get(
            result.provider or "", result.provider or "只跑硬规则"
        ),
        "degraded": result.degraded,
        "reasons": list(result.reasons),
    }


def _understand(result: EngineResult, cfg: EngineConfig, decisions: Sequence[DecisionLogEntry]):
    route = result.route
    if route is None:
        return None
    rules = rule_texts(cfg.routing)
    route_log = next((d for d in decisions if d.stage == DecisionStage.ROUTE), None)
    return {
        "intent": route.intent.value,
        "intent_label": intent_label(route.intent),
        "confidence": route.intent_confidence,
        "top_intents": [
            {"intent": i.value, "label": intent_label(i), "p": round(p, 3)}
            for i, p in route.top_intents
        ],
        "parts": [
            {"intent": p.intent.value, "label": intent_label(p.intent), "p": round(p.confidence, 3)}
            for p in route.parts
        ],
        "need_human": route.need_human,
        "need_human_threshold": cfg.routing.need_human,
        "emotion": route.emotion,
        "emotion_label": emotion_label(route.emotion),
        "emotion_threshold": cfg.routing.emotion_escalate,
        "multi_intent": route.multi_intent,
        "multi_intent_threshold": cfg.routing.multi_intent,
        "assist_threshold": cfg.routing.assist_threshold,
        "high_risk": route.high_risk,
        "rule": route.rule,
        "rule_text": rules[route.rule],
        "action": route.action.value,
        "action_label": ROUTE_ACTION_LABELS[route.action],
        "provider": route.provider,
        "provider_label": PROVIDER_LABELS.get(route.provider, route.provider),
        "questions_ver": route_log.questions_ver if route_log else None,
        "degraded": route.degraded,
        "risk_hints": list(route.risk_hints),
        "alerts": list(route.alerts),
    }


def build_trace(
    result: EngineResult,
    recording: Recording,
    *,
    config: EngineConfig,
    knowledge: KnowledgeRepository,
    decisions: Sequence[DecisionLogEntry],
    events: Sequence[AuditEvent],
    simulated_note: str | None = None,
) -> dict[str, Any]:
    """页面上"AI 是怎么想的"面板的数据（见模块说明）。"""
    shop_id = result.shop_id
    cited = {e.id for card in result.suggestions for e in card.evidence}
    intents: list[Intent] = []
    for item in [*recording.retrievals, *recording.generations, *recording.checks]:
        if item.intent not in intents:
            intents.append(item.intent)
    parts = []
    for intent in intents:
        find = [r for r in recording.retrievals if r.intent == intent]
        bundle = find[0].bundle if find else None
        overridden = []
        if bundle is not None:
            for eid in bundle.overridden_ids:
                found = describe_evidence(knowledge, shop_id, eid)
                overridden.append(
                    {
                        "id": eid,
                        "label": found[0] if found else eid,
                        "content": _short(found[1]) if found else None,
                    }
                )
        attempts = []
        for n, g in enumerate([g for g in recording.generations if g.intent == intent], 1):
            attempts.append(
                {
                    "attempt": n,
                    "retry_feedback": list(g.retry_feedback),
                    "drafts": [
                        {
                            "text": d.text,
                            "kind": d.kind.value,
                            "kind_label": DRAFT_KIND_LABELS[d.kind],
                            "variant": d.variant,
                            "evidence_ids": list(d.evidence_ids),
                        }
                        for d in g.drafts
                    ],
                }
            )
        checks = []
        attempt_of: dict[str, int] = {}
        for a in attempts:
            for d in a["drafts"]:
                attempt_of.setdefault(d["text"], a["attempt"])
        for c in [c for c in recording.checks if c.intent == intent]:
            checks.append(
                _qa_view(
                    c.result,
                    c.draft.kind.value,
                    c.draft.text,
                    attempt_of.get(c.draft.text, 1),
                    config,
                )
            )
        parts.append(
            {
                "intent": intent.value,
                "intent_label": intent_label(intent),
                "find": None
                if bundle is None
                else {
                    "evidence": [_evidence_view(e, cited) for e in bundle.evidences],
                    "overridden": overridden,
                    "missing": [{"marker": m, "label": gap_label(m)} for m in bundle.missing],
                    "unavailable": [
                        {"marker": m, "label": degraded_label(m)} for m in bundle.unavailable
                    ],
                    "clarify": bundle.clarify_question,
                    "alerts": list(bundle.alerts),
                },
                "write": attempts,
                "judge": checks,
            }
        )
    summary = next((e for e in reversed(events) if e.action == "engine.result"), None)
    hard_hits = list(summary.detail.get("hard_hits", [])) if summary else []
    qa_logs = [d for d in decisions if d.stage == DecisionStage.QA]
    latency = result.latency_ms
    total = int(latency.get("total", 0))
    since = int(latency.get("since_received", total))
    route = result.route
    blocked = sum(1 for p in parts for c in p["judge"] if not c["passed"])
    return {
        "understand": _understand(result, config, decisions),
        "images": images_view(result, recording, events),
        "parts": parts,
        "fallback": {
            "alerts": list(result.alerts),
            "transfer_to_human": result.transfer_to_human,
            "no_suggestion_reason": result.no_suggestion_reason,
            "missing_knowledge": [
                {"marker": m, "label": gap_label(m)} for m in result.missing_knowledge
            ],
            "degraded": [{"marker": m, "label": degraded_label(m)} for m in result.degraded],
            "human_only": route is not None
            and route.action in (RouteAction.HUMAN_PRIORITY, RouteAction.NO_SUGGESTION),
        },
        "qa_summary": {
            "checked": sum(len(p["judge"]) for p in parts),
            "blocked": blocked,
            "hard_hits": [{"rule": h, "label": HARD_RULE_LABELS.get(h, h)} for h in hard_hits if h],
            "model_checks": len(qa_logs),
        },
        "latency": {
            "stages": [
                {"key": k, "label": label, "ms": int(latency.get(k, 0))}
                for k, label in LATENCY_STAGES.items()
            ],
            "total_ms": total,
            "since_received_ms": max(since, total),
            "target_ms": E2E_TARGET_MS,
            "within_target": max(since, total) <= E2E_TARGET_MS,
        },
        "simulated_note": simulated_note,
    }


__all__ = [
    "build_trace",
    "card_view",
    "degraded_label",
    "describe_evidence",
    "gap_label",
    "images_view",
    "intent_label",
    "is_image_alert",
]
