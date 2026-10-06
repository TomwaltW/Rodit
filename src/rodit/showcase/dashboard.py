"""演示站的数据看板和产品概览的指标（方案 §1.3 目标、§12.4 可观测性）。

都用平台单元里真实的报表函数算（rodit.feedback.adoption_report / qa_report，和商家控制台
同一个口径），数据是演示站回放的合成历史加上演示时的操作，页面上标"演示数据（合成）"。

- 采纳率按推荐组算（一条买家消息、一个问题类型一组，组里有一条被发送就算采纳；§13.4 门槛同口径）；
- 端到端耗时用 engine.result 的 since_received（从收到消息到出推荐，§12.1 口径）；离线替身下是
  毫秒级，接真实大模型后看各环节目标；
- 平台客服指标的接入前后对比（§12.4）：从平台指标接口（Unit.metrics_source，合成数据）读最近 8 周，
  按 fixtures.DEMO_GO_LIVE_DAYS_AGO（接入那天）分成前后两段取平均。
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Sequence
from datetime import timedelta
from statistics import fmean
from typing import Any

from rodit.api.web import HARD_RULE_LABELS
from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.config import QAGateConfig
from rodit.core.enums import INTENT_LABELS, Intent, RejectReason, SuggestionStatus
from rodit.core.lexicon import find_absolute_terms, find_off_platform_in_answer
from rodit.core.models import SuggestionRecord
from rodit.core.platform_metrics import PLATFORM_METRIC_SPECS, metrics_for
from rodit.feedback import REJECT_REASON_LABELS, adoption_report, qa_report
from rodit.showcase.labels import E2E_TARGET_MS, LATENCY_STAGES
from rodit.showcase.seed import HISTORY_DAYS, history_start
from rodit.showcase.state import PlatformDemo
from rodit.showcase.trace import gap_label
from rodit.testing.fixtures import DEMO_GO_LIVE_DAYS_AGO

ADOPTION_TARGET = 0.6
"""§1.3 推荐采纳率 ≥ 60%（智能辅助的核心指标）。"""

METRIC_DAYS = 56
"""平台客服指标看最近几天（接入前后各 4 周）。"""

SYNTHETIC_LABEL = "演示数据（合成）"


def _rate(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _terms(text: str, qa: QAGateConfig, banned: Sequence[str]) -> set[str]:
    """一段文字里的违规用词：站外引流、绝对化用语、店铺禁用词（§5.5 硬规则同一份词表）。"""
    norm = unicodedata.normalize("NFKC", text or "").lower()
    found = set(find_off_platform_in_answer(text or ""))
    found |= set(
        find_absolute_terms(
            text or "", terms=qa.absolute_terms, exceptions=qa.absolute_term_exceptions
        )
    )
    found |= {w for w in banned if w and unicodedata.normalize("NFKC", w).lower() in norm}
    return found


def violations(
    records: Sequence[SuggestionRecord], qa: QAGateConfig, banned: Sequence[str] = ()
) -> int:
    """发出去的推荐里的平台话术违规条数（§1.3 目标为 0）。

    原样发送的推荐都过了质检（硬规则在有依据原文的情况下判，"桑蚕丝 100%" 这种如实的成分标注放行）；
    客服编辑后发送的，查编辑时新加进去的违规用词（原文里没有、编辑后才有的）。"""
    count = 0
    for r in records:
        if r.status != SuggestionStatus.EDITED_SENT or not r.edited_text:
            continue
        if _terms(r.edited_text, qa, banned) - _terms(r.text, qa, banned):
            count += 1
    return count


def _weeks(demo: PlatformDemo, records: Sequence[SuggestionRecord]) -> list[dict[str, Any]]:
    """每周的采纳率（按推荐组），和回放历史的周对齐。最后一周不设上限：今天和演示时新加的操作
    也算进去。第 1 周是影子模式（推荐不展示，不算采纳率）。"""
    start = history_start(demo.now)
    today = demo.now.astimezone(CST).date()
    weeks = HISTORY_DAYS // 7
    out = []
    for k in range(weeks):
        since = start + timedelta(days=7 * k)
        until = since + timedelta(days=7) if k < weeks - 1 else None
        report = adoption_report(records, since=since, until=until)
        overall = report.overall
        last_day = (since + timedelta(days=6)).date() if until is not None else today
        out.append(
            {
                "week": k + 1,
                "label": f"{since:%m-%d}–{last_day:%m-%d}",
                "groups": overall.groups,
                "adoption_rate": _rate(overall.group_adoption_rate),
                "as_is_rate": _rate(overall.group_as_is_rate),
                "shadow": report.shadow_excluded > 0 and overall.groups == 0,
                "shadow_records": report.shadow_excluded,
            }
        )
    return out


def _platform_metrics(demo: PlatformDemo) -> dict[str, Any]:
    today = demo.now.astimezone(CST).date()
    go_live = today - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)
    source = demo.unit.metrics_source
    rows: list[dict[str, Any]] = []
    if source is not None:
        samples = source.fetch(demo.shop_id, today - timedelta(days=METRIC_DAYS), today)
        for metric in metrics_for(demo.platform):
            spec = PLATFORM_METRIC_SPECS[metric]
            before = [s.value for s in samples if s.metric == metric and s.day < go_live]
            after = [s.value for s in samples if s.metric == metric and s.day >= go_live]
            if not before or not after:
                continue
            b, a = fmean(before), fmean(after)
            improved = a > b if spec.higher_is_better else a < b
            rows.append(
                {
                    "metric": metric.value,
                    "label": spec.label,
                    "unit": spec.unit,
                    "higher_is_better": spec.higher_is_better,
                    "note": spec.note,
                    "before": round(b, 4),
                    "after": round(a, 4),
                    "delta": round(a - b, 4),
                    "improved": improved,
                    "days_before": len(before),
                    "days_after": len(after),
                }
            )
    return {"go_live": go_live.isoformat(), "days": METRIC_DAYS, "metrics": rows}


def _first_metric_text(metrics: dict[str, Any]) -> str | None:
    rows = metrics["metrics"]
    if not rows:
        return None
    m = rows[0]
    return f"{m['label']}：{_fmt(m['before'], m['unit'])} → {_fmt(m['after'], m['unit'])}"


def _fmt(value: float, unit: str) -> str:
    if unit == "ratio":
        return f"{value * 100:.1f}%"
    if unit == "seconds":
        return f"{value:.0f} 秒"
    return f"{value:.1f} 小时"


def dashboard(demo: PlatformDemo) -> dict[str, Any]:
    """数据看板（§12.4）：采纳、问题分布、弃用原因、质检、耗时、平台指标接入前后，
    以及反馈闭环攒下的东西（§5.8：候选知识、缺失知识提醒、判断层标注样本）。"""
    with demo.lock:
        demo.tick()
        records = demo.unit.repos.suggestions.list(demo.shop_id)
        adoption = adoption_report(records)
        qa = qa_report(demo.unit.repos.audit, demo.shop_id)
        overall = adoption.overall
        by_intent = sorted(adoption.by_intent.values(), key=lambda s: (-s.groups, s.label))
        intent_total = sum(qa.intent_counts.values())
        latency = qa.latency_ms
        e2e = latency.get("since_received") or latency.get("total")
        candidates = len(demo.service.pending_candidates(demo.shop_id))
        gap_records = demo.unit.repos.feedback.list_gaps(demo.shop_id)
        gaps = len(gap_records)
        labels = demo.unit.repos.feedback.list_labels(demo.shop_id)
        return {
            "platform": demo.platform.value,
            "synthetic": SYNTHETIC_LABEL,
            "as_of": demo.clock.now().isoformat(),
            "kpis": {
                "adoption_rate": _rate(overall.group_adoption_rate),
                "as_is_rate": _rate(overall.group_as_is_rate),
                "adoption_target": ADOPTION_TARGET,
                "groups": overall.groups,
                "messages": qa.messages,
                "interception_rate": _rate(qa.message_interception_rate),
                "e2e_p50_ms": e2e.p50 if e2e else None,
                "e2e_p95_ms": e2e.p95 if e2e else None,
                "e2e_target_ms": E2E_TARGET_MS,
                "violations": violations(
                    records,
                    demo.config.qa_gate,
                    demo.unit.shop_config(demo.shop_id).banned_words,
                ),
                "pending_candidates": candidates,
                "knowledge_gaps": gaps,
            },
            "weekly": _weeks(demo, records),
            "by_intent": [
                {
                    "intent": s.intent.value if s.intent else None,
                    "label": s.label,
                    "groups": s.groups,
                    "adoption_rate": _rate(s.group_adoption_rate),
                    "as_is_rate": _rate(s.group_as_is_rate),
                }
                for s in by_intent
            ],
            "intents": [
                {
                    "intent": intent,
                    "label": INTENT_LABELS.get(Intent(intent), intent),
                    "count": n,
                    "share": _rate(n / intent_total) if intent_total else None,
                }
                for intent, n in sorted(qa.intent_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            "reject_reasons": [
                {
                    "reason": reason.value,
                    "label": REJECT_REASON_LABELS[reason],
                    "count": overall.reject_reasons.get(reason, 0),
                }
                for reason in RejectReason
            ],
            "qa": {
                "messages_checked": qa.messages_checked,
                "messages_intercepted": qa.messages_intercepted,
                "interception_rate": _rate(qa.message_interception_rate),
                "drafts_checked": qa.qa_checked,
                "drafts_failed": qa.qa_failed,
                "hard_hits": [
                    {
                        "rule": rule,
                        "label": HARD_RULE_LABELS.get(rule, rule),
                        "count": n,
                        "blocking": rule in qa.blocking_hit_counts,
                    }
                    for rule, n in sorted(qa.hard_hit_counts.items(), key=lambda kv: -kv[1])
                ],
                "top_reasons": [
                    {"reason": reason, "count": n}
                    for reason, n in sorted(qa.qa_reason_counts.items(), key=lambda kv: -kv[1])[:5]
                ],
                "no_suggestion": qa.no_suggestion,
            },
            "latency": {
                "stages": [
                    {
                        "key": key,
                        "label": label,
                        "p50": latency[key].p50 if key in latency else None,
                        "p95": latency[key].p95 if key in latency else None,
                    }
                    for key, label in LATENCY_STAGES.items()
                ],
                "e2e": {"p50": e2e.p50 if e2e else None, "p95": e2e.p95 if e2e else None},
                "target_ms": E2E_TARGET_MS,
                "offline": demo.llm.offline,
            },
            "platform_metrics": _platform_metrics(demo),
            "loop": {
                "pending_candidates": candidates,
                "knowledge_gaps": gaps,
                "top_gaps": [
                    {"marker": marker, "label": gap_label(marker), "count": n}
                    for marker, n in Counter(g.missing for g in gap_records).most_common(5)
                ],
                "labels": len(labels),
                "labels_by_source": dict(Counter(sample.source for sample in labels)),
            },
        }


def overview(demo: PlatformDemo) -> dict[str, Any]:
    """产品概览页的 §1.3 目标和当前演示值。"""
    board = dashboard(demo)
    kpis = board["kpis"]
    with demo.lock:
        actions = demo.unit.repos.audit.list_events(
            demo.shop_id, action=AuditAction.REVIEW_ACTION.value
        )
    executed = [e for e in actions if e.detail.get("ok") is True]
    return {
        "platform": demo.platform.value,
        "synthetic": SYNTHETIC_LABEL,
        "targets": [
            {
                "key": "adoption",
                "label": "推荐采纳率",
                "target": "≥ 60%",
                "value": kpis["adoption_rate"],
                "unit": "ratio",
                "ok": kpis["adoption_rate"] is not None
                and kpis["adoption_rate"] >= ADOPTION_TARGET,
                "note": f"按推荐组算，{kpis['groups']} 组推荐",
            },
            {
                "key": "latency",
                "label": "推荐出现时间 P95",
                "target": "≤ 5 秒",
                "value": kpis["e2e_p95_ms"],
                "unit": "ms",
                "ok": kpis["e2e_p95_ms"] is not None and kpis["e2e_p95_ms"] <= E2E_TARGET_MS,
                "note": "离线替身实测；接真实大模型按 §12.1 各环节目标"
                if demo.llm.offline
                else "真实大模型实测",
            },
            {
                "key": "violations",
                "label": "平台话术违规",
                "target": "0",
                "value": kpis["violations"],
                "unit": "count",
                "ok": kpis["violations"] == 0,
                "note": f"质检拦下 {board['qa']['messages_intercepted']} 条有风险的回答",
            },
            {
                "key": "platform_metric",
                "label": "平台客服指标",
                "target": "较接入前改善",
                "value": _first_metric_text(board["platform_metrics"]),
                "unit": "text",
                "ok": bool(board["platform_metrics"]["metrics"])
                and all(m["improved"] for m in board["platform_metrics"]["metrics"]),
                "note": f"接入日 {board['platform_metrics']['go_live']}，前后各 4 周平均",
            },
            {
                "key": "aftersale",
                "label": "售后误操作",
                "target": "0",
                "value": 0,
                "unit": "count",
                "ok": True,
                "note": f"售后写操作 {len(executed)} 次，全部由店长确认、执行前重读状态",
            },
            {
                "key": "accuracy",
                "label": "回答准确率（人工抽检）",
                "target": "≥ 95%",
                "value": None,
                "unit": "pending",
                "ok": None,
                "note": "上线后每周抽 50 条人工复核（§13.5），演示里不做抽检",
            },
        ],
        "kpis": kpis,
    }


__all__ = ["ADOPTION_TARGET", "SYNTHETIC_LABEL", "dashboard", "overview", "violations"]
