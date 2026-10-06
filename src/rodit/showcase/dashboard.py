"""演示站的数据看板和产品概览的指标（方案 §1.3 目标、§12.4 可观测性）。

都用平台单元里真实的报表函数算（rodit.feedback.adoption_report / qa_report，和商家控制台
同一个口径），数据是演示站回放的合成历史加上演示时的操作，页面上标"演示数据（合成）"。

- 采纳率按推荐组算（一条买家消息、一个问题类型一组，组里有一条被发送就算采纳；§13.4 门槛同口径）；
- 端到端耗时用 engine.result 的 since_received（从收到消息到出推荐，§12.1 口径）；离线替身下是
  毫秒级（含回放的合成历史），接真实大模型（--llm env）时只算演示时的消息（PlatformDemo.live_since
  之后；回放历史用的是离线替身，不算"真实大模型实测"），还没有时显示"暂无真实大模型数据"；
- 售后误操作（§1.3"不错误地同意或拒绝售后"）对错要人工复核，审计里算不出来：概览只写执行了几次
  写操作和防护措施，状态是"后续评估"，不写死 0 / 达标；
- 平台客服指标的接入前后对比（§12.4）：用商家报表同一个服务（rodit.app.platform_metrics 的
  PlatformMetricsService.report，口径见 rodit.feedback.platform_metrics）。数据是演示单元组装时
  从平台指标接口（合成数据）同步进单元仓储的最近 8 周（showcase.state）；接入日按演示设定填写：
  fixtures.DEMO_GO_LIVE_DAYS_AGO 天前（和回放历史切到智能辅助是同一天）。每项指标给出接入前后
  均值、变化、是否变好、按周序列，另有总体结论（§1.3 "较接入前改善"）。
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from rodit.api.web import HARD_RULE_LABELS
from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import INTENT_LABELS, Intent, RejectReason, SuggestionStatus
from rodit.core.lexicon import find_absolute_terms, find_off_platform_in_answer
from rodit.core.models import SuggestionRecord
from rodit.core.pii import CONTACT_MASK, EMAIL_MASK, PHONE_MASK, mask_pii
from rodit.feedback import REJECT_REASON_LABELS, adoption_report, qa_report
from rodit.feedback.compliance import (
    RULE_ABSOLUTE,
    RULE_BANNED,
    RULE_OFF_PLATFORM,
    ComplianceRules,
    scan_text,
)
from rodit.feedback.platform_metrics import MetricComparison, PlatformMetricsReport
from rodit.showcase.labels import E2E_TARGET_MS, LATENCY_STAGES
from rodit.showcase.seed import HISTORY_DAYS, history_start
from rodit.showcase.state import PlatformDemo
from rodit.showcase.trace import gap_label
from rodit.testing.fixtures import DEMO_GO_LIVE_DAYS_AGO

ADOPTION_TARGET = 0.6
"""§1.3 推荐采纳率 ≥ 60%（智能辅助的核心指标）。"""

SYNTHETIC_LABEL = "演示数据（合成）"


def _rate(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


_CONTACT_PLACEHOLDERS = (PHONE_MASK, CONTACT_MASK, EMAIL_MASK)


def _hits(text: str, rules: ComplianceRules) -> set[str]:
    """一段文字的违规命中：生产的合规扫描（feedback.compliance.scan_text）命中的规则名，再细到具体
    用词（规则名:用词）。站外引流、绝对化用语、店铺禁用词和质检硬规则同一份词表；存下来的文字做过
    个人信息最小化，遮住的手机号、联系方式、邮箱占位符也算站外引流（和 scan_text 同一条规则）。"""
    text = text or ""
    norm = unicodedata.normalize("NFKC", text).lower()
    found = set(scan_text(text, rules))
    found |= {f"{RULE_OFF_PLATFORM}:{t}" for t in find_off_platform_in_answer(text)}
    found |= {f"{RULE_OFF_PLATFORM}:{p}" for p in _CONTACT_PLACEHOLDERS if p in text}
    found |= {
        f"{RULE_ABSOLUTE}:{t}"
        for t in find_absolute_terms(
            text, terms=rules.absolute_terms, exceptions=rules.absolute_term_exceptions
        )
    }
    found |= {
        f"{RULE_BANNED}:{w}"
        for w in rules.banned_words
        if unicodedata.normalize("NFKC", w).lower() in norm
    }
    return found


def violations(records: Sequence[SuggestionRecord], rules: ComplianceRules) -> int:
    """发出去的推荐里的平台话术违规条数（§1.3 目标为 0）。rules 用 ComplianceRules.from_config
    （EngineConfig.qa_gate + 店铺的生效配置），和上线门槛的合规扫描同一份词表。

    原样发送的推荐都过了质检（硬规则在有依据原文的情况下判，"桑蚕丝 100%" 这种如实的成分标注放行）；
    客服编辑后发送的，查编辑时新加进去的违规（原文里没有、编辑后才有的规则或用词）。edited_text
    存的是个人信息最小化之后的文字（客服加的手机号、邮箱变成了占位符），原文也按同样的方式遮一遍
    再比。"""
    count = 0
    for r in records:
        if r.status != SuggestionStatus.EDITED_SENT or not r.edited_text:
            continue
        if _hits(r.edited_text, rules) - _hits(mask_pii(r.text), rules):
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


def _metric_view(m: MetricComparison) -> dict[str, Any]:
    return {
        "metric": m.metric.value,
        "label": m.label,
        "unit": m.unit,
        "higher_is_better": m.higher_is_better,
        "note": m.note,
        "before": None if m.before.mean is None else round(m.before.mean, 4),
        "after": None if m.after.mean is None else round(m.after.mean, 4),
        "delta": m.delta,
        "relative_change": m.relative_change,
        "direction": m.direction,
        "direction_label": m.direction_label,
        "improved": m.direction == "improved",
        "days_before": m.before.days,
        "days_after": m.after.days,
        "sample_before": m.before.sample_size,
        "sample_after": m.after.sample_size,
        "weighted": m.before.weighted and m.after.weighted,
        "reliable": m.reliable,
        "flag_notes": list(m.flag_notes),
        "weekly": [
            {
                "index": w.index,
                "phase": w.phase,
                "start": w.start.isoformat(),
                "end": w.end.isoformat(),
                "days": w.days,
                "mean": None if w.mean is None else round(w.mean, 4),
            }
            for w in m.weekly
        ],
    }


def platform_report(demo: PlatformDemo) -> PlatformMetricsReport:
    """接入前后对比报表（商家报表同一个服务；接入日按演示设定填写，见模块说明）。"""
    today = demo.clock.now().astimezone(CST).date()
    go_live = demo.now.astimezone(CST).date() - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)
    return demo.metrics_service().report(demo.shop_id, go_live=min(go_live, today))


def _platform_metrics(demo: PlatformDemo) -> dict[str, Any]:
    if demo.unit.repos.platform_metrics is None:
        return {
            "go_live": None,
            "metrics": [],
            "verdict": "insufficient",
            "verdict_label": "本单元没有平台指标数据",
            "notes": [],
        }
    report = platform_report(demo)
    sync = demo.metrics_sync
    return {
        "go_live": report.go_live.isoformat() if report.go_live else None,
        "go_live_label": "演示设定的接入日（和回放历史切到智能辅助是同一天）",
        "window_days": report.window_days,
        "before": [d.isoformat() if d else None for d in (report.before_start, report.before_end)],
        "after": [d.isoformat() if d else None for d in (report.after_start, report.after_end)],
        "verdict": report.verdict,
        "verdict_label": report.verdict_label,
        "reliable": report.reliable,
        "counts": dict(report.counts),
        "notes": list(report.notes),
        "metrics": [_metric_view(m) for m in report.metrics],
        "synced": None
        if sync is None
        else {
            "written": sync.written,
            "days": sync.days,
            "since": sync.since.isoformat(),
            "until": sync.until.isoformat(),
            "at": sync.synced_at.isoformat(),
        },
    }


def _first_metric_text(metrics: dict[str, Any]) -> str | None:
    rows = [m for m in metrics["metrics"] if m["before"] is not None and m["after"] is not None]
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
        live_only = not demo.llm.offline
        timed = (
            qa_report(demo.unit.repos.audit, demo.shop_id, since=demo.live_since)
            if live_only
            else qa
        )
        latency = timed.latency_ms
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
                "e2e_note": _latency_note(live_only, timed.messages),
                "violations": violations(
                    records,
                    ComplianceRules.from_config(
                        demo.config.qa_gate, demo.unit.shop_config(demo.shop_id)
                    ),
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
                "live_only": live_only,
                "messages": timed.messages,
                "note": _latency_note(live_only, timed.messages),
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


def _latency_note(live_only: bool, messages: int) -> str:
    """耗时的口径说明（看板和概览共用）。"""
    if not live_only:
        return "离线替身实测（含回放的合成历史）；接真实大模型按 §12.1 各环节目标"
    if not messages:
        return (
            "暂无真实大模型数据：在工作台以买家身份发几条消息后再看（回放的历史用离线替身，不计入）"
        )
    return (
        f"真实大模型实测：演示时的 {messages} 条消息（回放的历史用离线替身，不计入；"
        "路由、质检仍是离线替身）"
    )


def _metric_note(metrics: dict[str, Any]) -> str:
    if not metrics.get("go_live"):
        return metrics["verdict_label"]
    days = metrics["window_days"]
    return f"{metrics['verdict_label']} · 接入日 {metrics['go_live']}，前后各 {days} 天"


def overview(demo: PlatformDemo) -> dict[str, Any]:
    """产品概览页的 §1.3 目标和当前演示值。"""
    board = dashboard(demo)
    kpis = board["kpis"]
    with demo.lock:
        actions = demo.unit.repos.audit.list_events(
            demo.shop_id, action=AuditAction.REVIEW_ACTION.value
        )
    executed = [e for e in actions if e.detail.get("ok") is True and not e.detail.get("duplicate")]
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
                "ok": None if kpis["e2e_p95_ms"] is None else kpis["e2e_p95_ms"] <= E2E_TARGET_MS,
                "note": kpis["e2e_note"],
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
                "ok": board["platform_metrics"]["verdict"] == "improved",
                "note": _metric_note(board["platform_metrics"]),
            },
            {
                "key": "aftersale",
                "label": "售后误操作",
                "target": "0",
                # 同意 / 拒绝得对不对要人工复核（上线后），审计里算不出来：不写死 0 / 达标
                "value": f"已执行 {len(executed)} 次写操作",
                "unit": "text",
                "ok": None,
                "note": "同意 / 拒绝得对不对要人工复核，演示里不做复核；防护：只给建议、"
                "店长确认才执行、执行前重读平台状态、幂等键防重复（§7.3.2）",
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


__all__ = [
    "ADOPTION_TARGET",
    "SYNTHETIC_LABEL",
    "dashboard",
    "overview",
    "platform_report",
    "violations",
]
