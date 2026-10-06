"""售后审核时长（方案 §2.1 抖店体验分"售后平均审核时长"、§7.4 审核助手直接改善这个指标、§12.4）。

纯函数，不读仓储、不读时钟：输入审核记录的迁移历史、审计事件、平台售后单的申请时间，输出时长和汇总。
售后审核工作台（rodit.app.aftersale_desk）用它给商家看"售后平均审核时长"的近似值。

口径：
- 审核时长 = 平台售后单的申请时间（AfterSaleInfo.created_at）→ 第一次由人确认执行的时间。
  申请时间取审核助手确认执行时写进审计的 aftersale_created_at（CREATED_AT_KEY，执行前重读平台
  拿到的），统计时不读平台；加这个字段之前确认的单没有，算"时长未知"。
  "由人确认"指客服在工作台里确认、平台写操作成功：审计 aftersale.action（ok=True，不含幂等键重复的
  那次）或审核记录迁移历史里第一条 待客服确认 → 已同意 / 已拒绝 / 退货中、带操作人的迁移，两者取早。
  R3 同意退货之后再同意退款是同一张售后单的第二次操作，不另算。
- 商家直接在平台后台处理的售后单，工作台看不到确认时间（只能看到"平台已处理"），不计入。
- 时钟误差导致确认时间早于申请时间时按 0 计，不出现负数。
- 汇总给条数、平均、中位数、P90（最近秩法）和最大值，单位秒；没有样本时都为空。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict

from rodit.core.audit import AuditAction
from rodit.core.enums import ReviewState
from rodit.core.models import AuditEvent

DECISION_STATES: frozenset[ReviewState] = frozenset(
    {ReviewState.AGREED, ReviewState.REFUSED, ReviewState.WAIT_RETURN}
)
"""客服确认执行后进入的状态（core.enums.REVIEW_TRANSITIONS：只能从待客服确认进入）。"""

CREATED_AT_KEY = "aftersale_created_at"
"""审核助手确认执行时写进 aftersale.confirm / aftersale.action 审计 detail 的平台申请时间
（AfterSaleInfo.created_at，ISO 格式带时区）：统计审核时长不再为每张单读平台。"""

_NOT_HUMAN = frozenset({"system", "bot", "auto"})


class DurationSummary(BaseModel):
    """一组时长的汇总（秒）。count 为 0 时其余都为空。"""

    model_config = ConfigDict(extra="forbid")

    count: int = 0
    average_s: float | None = None
    median_s: float | None = None
    p90_s: float | None = None
    max_s: float | None = None


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else None
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _is_person(actor: Any) -> bool:
    if not isinstance(actor, str) or not actor.strip():
        return False
    return actor.strip().lower() not in _NOT_HUMAN


def first_decision(history: Sequence[Mapping[str, Any]]) -> datetime | None:
    """迁移历史（rodit.aftersale.state_machine 的 history 条目）里第一次由人确认执行的时间。

    只认 待客服确认 → 已同意 / 已拒绝 / 退货中、带操作人（by 是具体的人）的迁移；系统发现商家在平台
    后台同意了退货（没有 by）不算。缺键、时间格式不对、不带时区的条目跳过。"""
    found: datetime | None = None
    for step in history:
        if not isinstance(step, Mapping):
            continue
        if step.get("from") != ReviewState.CONFIRM.value:
            continue
        if step.get("to") not in {s.value for s in DECISION_STATES}:
            continue
        if not _is_person(step.get("by")):
            continue
        at = _parse_time(step.get("at"))
        if at is not None and (found is None or at < found):
            found = at
    return found


def decision_times(events: Iterable[AuditEvent]) -> dict[str, datetime]:
    """审计事件 → {售后单 ID: 第一次平台写操作成功的时间}。

    只看 aftersale.action、ok=True、不是幂等键重复（duplicate）、操作人是具体的人的事件；
    其他事件忽略（调用方可以直接传某店铺的全部审计事件）。"""
    out: dict[str, datetime] = {}
    for event in events:
        if event.action != AuditAction.REVIEW_ACTION.value:
            continue
        detail = event.detail if isinstance(event.detail, Mapping) else {}
        if detail.get("ok") is not True or detail.get("duplicate") is True:
            continue
        if not _is_person(event.actor):
            continue
        aftersale_id = detail.get("aftersale_id")
        if not isinstance(aftersale_id, str) or not aftersale_id:
            continue
        current = out.get(aftersale_id)
        if current is None or event.created_at < current:
            out[aftersale_id] = event.created_at
    return out


def application_times(events: Iterable[AuditEvent]) -> dict[str, datetime]:
    """审计事件 → {售后单 ID: 平台申请时间}（aftersale.confirm / aftersale.action 的
    CREATED_AT_KEY）。没有这个字段、格式不对、不带时区的跳过；其他事件忽略。"""
    actions = {AuditAction.REVIEW_ACTION.value, AuditAction.REVIEW_CONFIRM.value}
    out: dict[str, datetime] = {}
    for event in events:
        if event.action not in actions:
            continue
        detail = event.detail if isinstance(event.detail, Mapping) else {}
        aftersale_id = detail.get("aftersale_id")
        created = _parse_time(detail.get(CREATED_AT_KEY))
        if isinstance(aftersale_id, str) and aftersale_id and created is not None:
            out.setdefault(aftersale_id, created)
    return out


def merge_decisions(*sources: Mapping[str, datetime]) -> dict[str, datetime]:
    """几份 {售后单 ID: 确认时间} 合在一起，同一张售后单取最早的。"""
    out: dict[str, datetime] = {}
    for source in sources:
        for aftersale_id, at in source.items():
            if aftersale_id not in out or at < out[aftersale_id]:
                out[aftersale_id] = at
    return out


def review_duration(created_at: datetime, decided_at: datetime) -> timedelta:
    """申请 → 确认执行的时长；确认时间早于申请时间（时钟误差）时按 0 计。"""
    return max(decided_at - created_at, timedelta(0))


def waiting_seconds(created_at: datetime, now: datetime) -> int:
    """还没处理的售后单已经等了多久（秒，向下取整，不小于 0）。"""
    return max(int((now - created_at).total_seconds()), 0)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """最近秩法：第 ceil(q × n) 个（从 1 数）。"""
    rank = max(math.ceil(q * len(sorted_values)), 1)
    return sorted_values[rank - 1]


def summarize(durations: Iterable[timedelta]) -> DurationSummary:
    """时长 → 条数、平均、中位数、P90、最大值（秒）。"""
    values = sorted(d.total_seconds() for d in durations)
    if not values:
        return DurationSummary()
    n = len(values)
    middle = n // 2
    median = values[middle] if n % 2 else (values[middle - 1] + values[middle]) / 2
    return DurationSummary(
        count=n,
        average_s=sum(values) / n,
        median_s=median,
        p90_s=_percentile(values, 0.9),
        max_s=values[-1],
    )


__all__ = [
    "CREATED_AT_KEY",
    "DECISION_STATES",
    "DurationSummary",
    "application_times",
    "decision_times",
    "first_decision",
    "merge_decisions",
    "review_duration",
    "summarize",
    "waiting_seconds",
]
