"""每月用采纳反馈重新计算阈值（方案 §13.5），按 §13.3 的算法给出建议，人工确认后才生效。

输入是反馈闭环的判断层标注集（LabeledSample，§5.8）：
- feedback_adopt（原样发送）、feedback_edit（编辑后发送）→ (预测类型, 置信度, 预测类型)：
  类型判断对了；
- feedback_reject（理解错误）且客服选了正确类型 → (预测类型, 置信度, 正确类型)；
- feedback_reject 没选正确类型 → (预测类型, 置信度, "<unknown>")：算作预测类型的一次误判，
  不给任何类型增加真实样本（覆盖率的分母）；
- manual（人工标注）带预测类型、置信度、正确类型时同样使用；
- 没有预测类型或置信度（旧数据、手工标注）的跳过并计数；记了路由规则、但不是规则 6
  （安抚、分条等不看辅助阈值的路径）的跳过并计数；时间窗外的不用。

对每个问题类型，按 §13.3 扫描阈值：智能辅助 target_precision = 0.90；全自动 0.98，只对
routing.auto_threshold 里列出的类型计算。和 rodit.eval.calibration.pick_threshold 的规则相同
（0.50–0.99 的网格，达标时覆盖率最高、同覆盖率取更低的阈值），另外：

1. 选择偏差（关键）：只有展示过的推荐才会有客服反馈，而路由只在置信度 ≥ 当前辅助阈值时出推荐
   （§6.6 规则 6）。当前阈值以下只有安抚（规则 3）、分条（规则 5）这些少数路径留下的标注，
   分条推荐还只会留下"理解错误"（原样发送的不写标注，见 FeedbackService），这部分既不完整也
   没有代表性（记了路由规则的标注按规则 6 过滤掉它们，旧标注分不开）。直接扫描的话，阈值往下挪时
   选中的样本不变、准确率和覆盖率一样，并列取低值，就会"建议"一个比当前阈值低、却没有任何
   数据支撑的阈值。所以：
   - 反馈数据的辅助阈值只维持或上调（ratchet）：扫描从当前阈值开始；要下调，先用影子对比或
     人工标注补上低置信度样本，或者换判断层后用 train-classifier 在留出集上重新校准；
   - 任何建议值都不低于该类型实际出现过的最低置信度（没有数据的区间不外推），也不低于网格下限
     0.50；达到同样效果时优先保留当前阈值（不做没有依据的改动）；
   - 全自动阈值不低于（建议后的）辅助阈值；辅助阈值以上的推荐都展示给了客服，这个区间的数据
     有代表性，全自动阈值可以下调。
2. 样本数：预测为该类型的标注少于 min_support 条时保持不变（样本不足）；建议的阈值以上也要有
   至少 min_support 条，否则按"达不到目标"处理（几条样本的 100% 不算数）。
3. 路由不看辅助阈值的类型不给建议：高风险类型走规则 4（推荐 + 风险提示）、投诉提醒人工、
   站外交易请求走规则 1（routing.off_platform）。
4. 覆盖率按条数精确比较（和 pick_threshold 一样，并列取更低的阈值），这里另外实现扫描，
   因为要加上面的下限和样本数规则，只复用它的结果类型。

输出（RecalibrationReport）：每个类型的当前阈值、建议阈值、样本数、准确率、覆盖率、状态，
中文摘要（render_text），以及只含要修改的类型的配置片段（proposal_yaml，合并进当前配置后按
RoutingConfig 校验过）。这里不写任何配置文件：建议由人确认，上线前按 §13.5 先跑回归。

另外要知道的数据局限：因知识缺失 / 表达不好等原因弃用的推荐不写标注，客服没操作就过期的推荐
也没有标注；全自动发出去的回复没有客服反馈，全自动阈值主要靠智能辅助
模式下的采纳数据。
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

import yaml
from pydantic import AwareDatetime, BaseModel, ConfigDict

from rodit.core.clock import CST
from rodit.core.config import RoutingConfig
from rodit.core.enums import INTENT_LABELS, Intent
from rodit.core.models import LabeledSample
from rodit.eval.calibration import Sample

ASSIST_PRECISION = 0.90  # §13.3 智能辅助：有人把关，追求覆盖率
AUTO_PRECISION = 0.98  # §13.3 全自动
DEFAULT_WINDOW_DAYS = 30  # §13.5 每月一次
DEFAULT_MIN_SUPPORT = 30
UNKNOWN_TRUE = "<unknown>"  # 理解错误但没选正确类型
ASSIST_RULE = 6  # §6.6 规则 6：置信度 ≥ 辅助阈值 → 出推荐（重算的就是这个阈值）
GRID: tuple[float, ...] = tuple(x / 100 for x in range(50, 100))  # 与 §13.3 一致
EPS = 1e-9

REGRESSION_REMINDER = (
    "上线前按 §13.5 先跑回归：python -m rodit.eval --config <新配置> --baseline <基线.json>，"
    "指标不下降才上线"
)
CLASSIFIER_REGRESSION_REMINDER = (
    "上线前按 §13.5 先带模型文件跑整条链路的回归：python -m rodit.eval --config <新配置> "
    "--classifier <模型文件> --baseline <基线.json>，指标不下降才上线"
)

Kind = Literal["assist", "auto"]


class ProposalStatus(StrEnum):
    CHANGE = "change"  # 建议修改
    KEEP = "keep"  # 当前阈值已经最好（或和最好的在数据上等价）
    INSUFFICIENT = "insufficient"  # 保持：样本不足
    UNREACHABLE = "unreachable"  # 保持：达不到目标准确率
    NOT_APPLICABLE = "not_applicable"  # 路由不看这个阈值


STATUS_LABELS: dict[ProposalStatus, str] = {
    ProposalStatus.CHANGE: "修改",
    ProposalStatus.KEEP: "保持（已是最好）",
    ProposalStatus.INSUFFICIENT: "保持（样本不足）",
    ProposalStatus.UNREACHABLE: "保持（达不到目标）",
    ProposalStatus.NOT_APPLICABLE: "不适用",
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ThresholdProposal(_Model):
    """一个问题类型、一种阈值（辅助 / 全自动）的建议。

    samples：预测为该类型的样本数（准确率的分母，样本不足按它判断）；
    support：真实为该类型的样本数（覆盖率、召回率的分母，同 ThresholdPick.support）；
    picked / precision / coverage / recall：建议阈值（保持时是当前阈值）上的表现，
    coverage 按 §13.3 的口径（阈值以上的条数 / support，有误判时可以超过 100%）；
    precision_lower：准确率的 95% 置信下限（Wilson），只作参考；current_*：当前阈值上的表现。
    """

    kind: Kind
    intent: Intent
    status: ProposalStatus
    note: str = ""
    target_precision: float
    current: float
    proposed: float  # 修改时是建议值，否则等于 current
    samples: int
    support: int
    min_confidence: float | None
    floor: float | None  # 这次扫描允许的最低阈值
    picked: int
    precision: float | None
    precision_lower: float | None
    coverage: float | None
    recall: float | None
    current_picked: int
    current_precision: float | None
    current_precision_lower: float | None
    current_coverage: float | None
    current_recall: float | None

    @property
    def changed(self) -> bool:
        return self.status == ProposalStatus.CHANGE


class ThresholdPlan(_Model):
    min_support: int
    assist_precision: float
    auto_precision: float
    ratchet: bool  # True：辅助阈值只维持或上调（反馈数据有选择偏差）
    assist: list[ThresholdProposal]
    auto: list[ThresholdProposal]

    def changes(self) -> list[ThresholdProposal]:
        return [p for p in (*self.assist, *self.auto) if p.changed]


class LabelStats(_Model):
    total: int  # 导出的全部标注
    in_window: int
    used: int
    by_source: dict[str, int]  # 用上的标注按来源
    reject_unknown_true: int  # 理解错误、没选正确类型的
    skipped_no_intent: int
    skipped_no_confidence: int
    skipped_manual_unlabeled: int
    skipped_other_rule: int = 0  # 不是规则 6 出的推荐（记了路由规则的标注才分得出）
    shops: int  # 用上的标注来自几家店铺


class RecalibrationReport(_Model):
    generated_at: AwareDatetime
    since: AwareDatetime
    until: AwareDatetime
    labels: LabelStats
    plan: ThresholdPlan
    yaml: str

    @property
    def changes(self) -> list[ThresholdProposal]:
        return self.plan.changes()


# ================================================================ 标注 → 三元组


def build_samples(
    labels: Iterable[LabeledSample], *, since: datetime, until: datetime
) -> tuple[list[Sample], LabelStats]:
    """时间窗 [since, until) 内的标注 → §13.3 三元组（规则见模块说明）。"""
    total = in_window = 0
    skipped = Counter[str]()
    by_source = Counter[str]()
    shops: set[str] = set()
    unknown = 0
    out: list[Sample] = []
    for label in labels:
        total += 1
        if not since <= label.created_at < until:
            continue
        in_window += 1
        if label.predicted_intent is None:
            skipped["no_intent"] += 1
            continue
        if label.predicted_confidence is None:
            skipped["no_confidence"] += 1
            continue
        if label.route_rule is not None and label.route_rule != ASSIST_RULE:
            skipped["other_rule"] += 1  # 安抚（规则 3）、分条（规则 5）等不看辅助阈值
            continue
        predicted = label.predicted_intent.value
        if label.source in ("feedback_adopt", "feedback_edit"):
            true = predicted  # 原样或编辑后发送：类型判断对了
        elif label.true_intent is not None:
            true = label.true_intent.value
        elif label.source == "feedback_reject":
            true = UNKNOWN_TRUE  # 只知道不是预测的类型
            unknown += 1
        else:
            skipped["manual_unlabeled"] += 1
            continue
        out.append(Sample(predicted=predicted, confidence=label.predicted_confidence, true=true))
        by_source[label.source] += 1
        shops.add(label.shop_id)
    stats = LabelStats(
        total=total,
        in_window=in_window,
        used=len(out),
        by_source=dict(sorted(by_source.items())),
        reject_unknown_true=unknown,
        skipped_no_intent=skipped["no_intent"],
        skipped_no_confidence=skipped["no_confidence"],
        skipped_manual_unlabeled=skipped["manual_unlabeled"],
        skipped_other_rule=skipped["other_rule"],
        shops=len(shops),
    )
    return out, stats


# ================================================================ 扫描


def not_applicable_reason(intent: Intent, routing: RoutingConfig) -> str | None:
    """路由不看辅助 / 全自动阈值的类型（§6.6），返回原因；其他类型返回 None。"""
    if intent == Intent.COMPLAINT:
        return "投诉走规则 4：提醒人工、不推荐"
    if intent in routing.high_risk:
        return "高风险类型走规则 4：推荐 + 风险提示，不看辅助阈值，也不全自动"
    if intent == Intent.OFF_PLATFORM:
        return "站外交易请求走规则 1（routing.off_platform）"
    return None


def wilson_lower(correct: int, n: int, *, z: float = 1.96) -> float | None:
    """准确率的 95% 置信下限（Wilson 区间）。只作参考，不参与 §13.3 的选择：
    几十条样本上的 100%，下限可能远低于 98%（全自动阈值尤其要看它）。"""
    if n <= 0:
        return None
    p = correct / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return round(max(0.0, (centre - margin) / (1 + z2 / n)), 4)


class _Rows:
    """一个类型的样本：按置信度统计选中数和正确数。"""

    def __init__(self, samples: Sequence[Sample], intent: str) -> None:
        self.predicted = [s for s in samples if s.predicted == intent]
        self.support = sum(1 for s in samples if s.true == intent)
        self.intent = intent
        ranked = sorted((s.confidence, s.true == intent) for s in self.predicted)
        self._confs = [c for c, _ in ranked]  # 升序
        self._correct_from = [0] * (len(ranked) + 1)  # 下标 i 起（置信度更高的一侧）正确的条数
        for i in range(len(ranked) - 1, -1, -1):
            self._correct_from[i] = self._correct_from[i + 1] + ranked[i][1]
        self.min_confidence = self._confs[0] if self._confs else None

    def at(self, t: float) -> tuple[int, int]:
        """阈值 t 上选中的样本数、其中正确的条数（置信度 ≥ t，容忍浮点误差，同路由的 _ge）。"""
        i = bisect_left(self._confs, t - EPS)
        return len(self._confs) - i, self._correct_from[i]

    def metrics(self, t: float, prefix: str = "") -> dict[str, Any]:
        """阈值 t 上的表现（ThresholdProposal 的字段；prefix="current_" 给当前阈值用）。"""
        picked, correct = self.at(t)
        out = {
            "picked": picked,
            "precision": round(correct / picked, 4) if picked else None,
            "precision_lower": wilson_lower(correct, picked),
            "coverage": round(picked / self.support, 4) if self.support else None,
            "recall": round(correct / self.support, 4) if self.support else None,
        }
        return {f"{prefix}{k}": v for k, v in out.items()}

    def base(self, intent: Intent, kind: Kind, target: float, current: float) -> dict[str, Any]:
        return {
            "kind": kind,
            "intent": intent,
            "target_precision": target,
            "current": current,
            "samples": len(self.predicted),
            "support": self.support,
            "min_confidence": self.min_confidence,
            **self.metrics(current, "current_"),
        }


def _meets(correct: int, picked: int, target: float) -> bool:
    return picked > 0 and correct / picked >= target - EPS


def _propose(
    rows: _Rows,
    intent: Intent,
    *,
    kind: Kind,
    target: float,
    current: float,
    policy_floor: float,
    min_support: int,
) -> ThresholdProposal:
    """一个类型、一种阈值的 §13.3 扫描（规则见模块说明）。

    候选：当前阈值（不低于 policy_floor 时，排第一个，并列时胜出）+ 网格里不低于 floor 的值 + floor
    本身，floor = max(policy_floor, 该类型最低置信度, 0.50)。只看选中样本 ≥ min_support 的候选；
    达标的候选里选中最多的胜出（覆盖率的分母不变，按条数比较是精确的），并列取先出现的。
    """
    base = rows.base(intent, kind, target, current)

    def result(
        status: ProposalStatus, note: str, proposed: float, floor: float | None
    ) -> ThresholdProposal:
        return ThresholdProposal(
            **base,
            status=status,
            note=note,
            proposed=proposed,
            floor=floor,
            **rows.metrics(proposed),
        )

    if rows.min_confidence is None or len(rows.predicted) < min_support:
        note = f"预测为该类型的标注 {len(rows.predicted)} 条，少于 {min_support} 条"
        return result(ProposalStatus.INSUFFICIENT, note, current, None)
    floor = max(policy_floor, rows.min_confidence, GRID[0])
    candidates: list[float] = []
    if current >= policy_floor - EPS:
        candidates.append(current)
    candidates += sorted({floor, *(t for t in GRID if t >= floor - EPS)})
    best: tuple[float, int] | None = None  # (阈值, 选中数)
    reachable_small = False
    for t in candidates:
        picked, correct = rows.at(t)
        if not _meets(correct, picked, target):
            continue
        if picked < min_support:
            reachable_small = True
            continue
        if best is None or picked > best[1]:
            best = (t, picked)
    if best is None:
        if reachable_small:
            note = f"只有阈值以上不到 {min_support} 条的高阈值才达标，样本太少不作数"
        else:
            note = f"扫描到的阈值都达不到 {target:.0%}"
        return result(ProposalStatus.UNREACHABLE, note, current, floor)
    threshold = best[0]
    if threshold == current:
        return result(ProposalStatus.KEEP, "当前阈值达标，且覆盖率已是最高", current, floor)
    return result(
        ProposalStatus.CHANGE, "上调" if threshold > current else "下调", threshold, floor
    )


def plan_thresholds(
    samples: Sequence[Sample],
    routing: RoutingConfig,
    *,
    min_support: int = DEFAULT_MIN_SUPPORT,
    assist_precision: float = ASSIST_PRECISION,
    auto_precision: float = AUTO_PRECISION,
    ratchet: bool = True,
) -> ThresholdPlan:
    """每个问题类型的辅助阈值建议，和 auto_threshold 里列出的类型的全自动阈值建议。

    ratchet=True（反馈数据，默认）：辅助阈值只维持或上调；ratchet=False（新模型在留出集上的
    预测，每个样本都有模型给的置信度，没有展示与否的筛选）：可以下调，但仍不低于最低置信度。
    """
    if min_support < 1:
        raise ValueError("min_support 必须 ≥ 1")
    for name, target in (
        ("assist_precision", assist_precision),
        ("auto_precision", auto_precision),
    ):
        if not 0 < target <= 1:
            raise ValueError(f"{name} 必须在 (0, 1] 之间")
    assist: list[ThresholdProposal] = []
    auto: list[ThresholdProposal] = []
    for intent in Intent:
        rows = _Rows(samples, intent.value)
        current = routing.assist_threshold_for(intent)
        reason = not_applicable_reason(intent, routing)
        if reason is not None:
            assist.append(
                _not_applicable(rows, intent, "assist", assist_precision, current, reason)
            )
        else:
            assist.append(
                _propose(
                    rows,
                    intent,
                    kind="assist",
                    target=assist_precision,
                    current=current,
                    policy_floor=current if ratchet else 0.0,
                    min_support=min_support,
                )
            )
        current_auto = routing.auto_threshold.get(intent)
        if current_auto is None:
            continue
        if reason is not None:
            auto.append(_not_applicable(rows, intent, "auto", auto_precision, current_auto, reason))
            continue
        auto.append(
            _propose(
                rows,
                intent,
                kind="auto",
                target=auto_precision,
                current=current_auto,
                policy_floor=assist[-1].proposed,  # 全自动阈值不低于（建议后的）辅助阈值
                min_support=min_support,
            )
        )
    return ThresholdPlan(
        min_support=min_support,
        assist_precision=assist_precision,
        auto_precision=auto_precision,
        ratchet=ratchet,
        assist=assist,
        auto=auto,
    )


def _not_applicable(
    rows: _Rows, intent: Intent, kind: Kind, target: float, current: float, reason: str
) -> ThresholdProposal:
    return ThresholdProposal(
        **rows.base(intent, kind, target, current),
        status=ProposalStatus.NOT_APPLICABLE,
        note=reason,
        proposed=current,
        floor=None,
        **rows.metrics(current),
    )


# ================================================================ 配置片段


def proposal_patch(plan: ThresholdPlan) -> dict[str, Any]:
    """只含要修改的类型的 routing 片段：{"routing": {"assist_threshold_by_intent": …,
    "auto_threshold": …}}（没有修改时 routing 为空）。"""
    routing: dict[str, Any] = {}
    assist = {p.intent.value: p.proposed for p in plan.assist if p.changed}
    auto = {p.intent.value: p.proposed for p in plan.auto if p.changed}
    if assist:
        routing["assist_threshold_by_intent"] = assist
    if auto:
        routing["auto_threshold"] = auto
    return {"routing": routing}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_patch(routing: RoutingConfig, patch: dict[str, Any]) -> RoutingConfig:
    """把配置片段逐项合并进当前的 routing（字典逐键合并，不整段替换）并按 RoutingConfig 校验。"""
    section = patch.get("routing") or {}
    if not isinstance(section, dict):
        raise ValueError("配置片段的 routing 应为映射")
    return RoutingConfig.model_validate(_merge(routing.model_dump(mode="json"), section))


def proposal_yaml(
    plan: ThresholdPlan, routing: RoutingConfig, *, header: Sequence[str] = ()
) -> str:
    """建议的配置片段（YAML）。先解析回来、合并进当前配置、按 RoutingConfig 校验，再返回；
    合并后的阈值必须正好是建议值（校验不过抛 ValueError，不会给出一个用不了的片段）。"""
    patch = proposal_patch(plan)
    lines = [f"# {h}" for h in header]
    lines += [
        "# 只列出要修改的问题类型：逐项合并进部署配置（config/rodit.yaml）的 routing 段，",
        "# 没列出的类型保持原值。不要整段替换 auto_threshold（没列出的类型会失去全自动资格）。",
        f"# {REGRESSION_REMINDER}",
    ]
    if not patch["routing"]:
        lines.append("# 没有需要修改的阈值")
    body = yaml.safe_dump(patch, allow_unicode=True, sort_keys=False, default_flow_style=False)
    text = "\n".join(lines) + "\n" + body
    merged = apply_patch(routing, yaml.safe_load(text) or {})
    for p in plan.changes():
        value = (
            merged.assist_threshold_for(p.intent)
            if p.kind == "assist"
            else merged.auto_threshold.get(p.intent)
        )
        if value != p.proposed:
            raise ValueError(f"配置片段合并后 {p.kind}.{p.intent.value} 不是建议值")
    return text


# ================================================================ 重算


def resolve_window(
    now: datetime,
    since: datetime | None = None,
    until: datetime | None = None,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> tuple[datetime, datetime]:
    """标注时间窗 [since, until)：默认 now 之前 window_days 天。时间都要带时区。"""
    if now.tzinfo is None:
        raise ValueError("now 需要带时区")
    until = until or now
    since = since or until - timedelta(days=window_days)
    if since.tzinfo is None or until.tzinfo is None:
        raise ValueError("since / until 需要带时区")
    if since >= until:
        raise ValueError("since 必须早于 until")
    return since, until


def recalibrate(
    labels: Iterable[LabeledSample],
    routing: RoutingConfig,
    *,
    now: datetime,
    since: datetime | None = None,
    until: datetime | None = None,
    min_support: int = DEFAULT_MIN_SUPPORT,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> RecalibrationReport:
    """§13.5 每月重算：默认用 now 之前 30 天的标注。只返回建议，不改任何配置。"""
    since, until = resolve_window(now, since, until, window_days=window_days)
    samples, stats = build_samples(labels, since=since, until=until)
    plan = plan_thresholds(samples, routing, min_support=min_support, ratchet=True)
    header = [f"§13.5 阈值重算建议（标注时间窗 {_fmt(since)} ~ {_fmt(until)}，东八区）"]
    return RecalibrationReport(
        generated_at=now,
        since=since,
        until=until,
        labels=stats,
        plan=plan,
        yaml=proposal_yaml(plan, routing, header=header),
    )


# ================================================================ 文字报告


def _fmt(dt: datetime) -> str:
    return dt.astimezone(CST).strftime("%Y-%m-%d %H:%M")


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}".rstrip("0").rstrip(".")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _intent_name(intent: Intent) -> str:
    return f"{intent.value}（{INTENT_LABELS[intent]}）"


def _performance(p: ThresholdProposal) -> str:
    return (
        f"阈值以上 {p.picked} 条，准确率 {_pct(p.precision)}（95% 置信下限 "
        f"{_pct(p.precision_lower)}），覆盖率 {_pct(p.coverage)}，召回率 {_pct(p.recall)}"
    )


def render_plan(plan: ThresholdPlan, add: Callable[[str], None]) -> None:
    """阈值建议的文字部分（重算和训练任务共用）。"""
    for kind, rows, target, title in (
        ("assist", plan.assist, plan.assist_precision, "智能辅助阈值"),
        ("auto", plan.auto, plan.auto_precision, "全自动阈值"),
    ):
        extra = "，只算已开放全自动的类型" if kind == "auto" else ""
        if kind == "assist" and plan.ratchet:
            extra = "，反馈数据只维持或上调"
        add(f"{title}（目标准确率 {target:.0%}{extra}）：")
        for p in rows:
            if p.status == ProposalStatus.NOT_APPLICABLE or not p.samples:
                continue
            if p.changed:
                head = f"{_num(p.current)} → {_num(p.proposed)}（{p.note}）"
            else:
                head = f"{_num(p.current)}，{STATUS_LABELS[p.status]}"
            line = f"  - {_intent_name(p.intent)}：{head}；样本 {p.samples} 条，{_performance(p)}"
            if p.changed:
                line += (
                    f"；当前阈值上 {p.current_picked} 条，准确率 {_pct(p.current_precision)}，"
                    f"覆盖率 {_pct(p.current_coverage)}"
                )
                if p.precision_lower is not None and p.precision_lower < p.target_precision:
                    line += "；注意：样本量下置信下限低于目标，确认前多看一眼"
            elif p.status in (ProposalStatus.UNREACHABLE, ProposalStatus.INSUFFICIENT):
                line += f"；{p.note}"
            add(line)
        empty = [p.intent.value for p in rows if not p.samples]
        if empty:
            add(f"  没有标注的类型：{'、'.join(empty)}")
        na = [
            p.intent.value for p in rows if p.status == ProposalStatus.NOT_APPLICABLE and p.samples
        ]
        if na:
            add(f"  不适用（路由不看这个阈值：规则 1 / 规则 4）：{'、'.join(na)}")
        if not rows:
            add("  （auto_threshold 没有列出任何类型）")


def render_text(report: RecalibrationReport, *, config_source: str | None = None) -> str:
    """中文摘要：时间窗、标注统计、每个类型的建议、注意事项和回归提醒。"""
    lines: list[str] = []
    add = lines.append
    s = report.labels
    add("§13.5 阈值重算（§13.3 算法）：只是建议，不会改任何配置；人工确认后再合并进部署配置")
    add(f"时间窗：{_fmt(report.since)} ~ {_fmt(report.until)}（东八区）")
    if config_source:
        add(f"当前配置：{config_source}")
    sources = s.by_source
    add(
        f"标注：导出 {s.total} 条，时间窗内 {s.in_window} 条，用上 {s.used} 条"
        f"（采纳 {sources.get('feedback_adopt', 0)}、编辑后发送 {sources.get('feedback_edit', 0)}、"
        f"理解错误 {sources.get('feedback_reject', 0)}"
        f"（其中没选正确类型 {s.reject_unknown_true}）、人工 {sources.get('manual', 0)}），"
        f"来自 {s.shops} 家店铺"
    )
    skipped = (
        s.skipped_no_intent
        + s.skipped_no_confidence
        + s.skipped_manual_unlabeled
        + s.skipped_other_rule
    )
    if skipped:
        add(
            f"跳过 {skipped} 条：没有预测类型 {s.skipped_no_intent}、没有置信度 "
            f"{s.skipped_no_confidence}、人工标注没有正确类型 {s.skipped_manual_unlabeled}、"
            f"不是规则 6 的推荐 {s.skipped_other_rule}"
        )
    add(f"样本门槛：预测为该类型的标注 ≥ {report.plan.min_support} 条才给建议")
    add("")
    render_plan(report.plan, add)
    add("")
    changes = report.changes
    n_assist = sum(1 for p in changes if p.kind == "assist")
    n_auto = sum(1 for p in changes if p.kind == "auto")
    if changes:
        add(f"建议修改：辅助阈值 {n_assist} 个类型、全自动阈值 {n_auto} 个类型（配置片段见下）")
    else:
        add("没有需要修改的阈值")
    add("说明：")
    add(
        "- 只有展示过的推荐才有标注：当前辅助阈值以下没有有代表性的数据，所以辅助阈值只维持或上调；"
        "要下调，先用影子对比或人工标注补上低置信度的样本，或者换判断层后用 train-classifier 的"
        "留出集重新校准"
    )
    add("- 建议值不低于该类型实际出现过的最低置信度；全自动阈值不低于辅助阈值")
    add(
        "- 因其他原因弃用、没人操作就过期的推荐没有标注；"
        "全自动发出去的回复也没有客服反馈，全自动阈值主要靠智能辅助模式下的采纳数据"
    )
    add(REGRESSION_REMINDER)
    return "\n".join(lines)


__all__ = [
    "ASSIST_PRECISION",
    "AUTO_PRECISION",
    "DEFAULT_MIN_SUPPORT",
    "DEFAULT_WINDOW_DAYS",
    "GRID",
    "CLASSIFIER_REGRESSION_REMINDER",
    "REGRESSION_REMINDER",
    "STATUS_LABELS",
    "UNKNOWN_TRUE",
    "resolve_window",
    "LabelStats",
    "ProposalStatus",
    "RecalibrationReport",
    "ThresholdPlan",
    "ThresholdProposal",
    "apply_patch",
    "build_samples",
    "not_applicable_reason",
    "plan_thresholds",
    "proposal_patch",
    "proposal_yaml",
    "recalibrate",
    "render_plan",
    "render_text",
    "wilson_lower",
]
