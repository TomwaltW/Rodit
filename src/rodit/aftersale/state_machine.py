"""售后审核状态机（方案 §7.3.2；合法迁移见 core.enums.REVIEW_TRANSITIONS）。

AfterSaleReview 没有"历史"字段。本模块把审核流程的元数据放在 action_result（§10 JSONB）里，
键固定如下（其余模块只通过本模块的函数读写）：

- "history"：[{"from", "to", "at", "by"?, "note"?}]，每次迁移追加一条，按时间顺序；
- "negotiated"：进入过协商（NEGOTIATE）就为 true。淘宝 2025-10-15 起部分场景要先协商才能拒绝，
  确认拒绝时据此判断；
- "result"：最近一次平台写操作的结果（core.models.ActionResult 的 model_dump(mode="json")）。

迁移同时写审计（aftersale.transition，由审核助手负责），历史既在记录里也在审计日志里。
进入 VERIFY 时记下 verify_started_at（R4 签收核查开始时间）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rodit.core.enums import REVIEW_TRANSITIONS, ReviewState
from rodit.core.errors import ActionNotAllowed
from rodit.core.models import ActionResult, AfterSaleReview, strip_control_chars

HISTORY_KEY = "history"
NEGOTIATED_KEY = "negotiated"
RESULT_KEY = "result"

TERMINAL_STATES = frozenset(s for s, targets in REVIEW_TRANSITIONS.items() if not targets)


def can_transition(frm: ReviewState, to: ReviewState) -> bool:
    return to in REVIEW_TRANSITIONS[frm]


def _meta(review: AfterSaleReview) -> dict[str, Any]:
    return dict(review.action_result or {})


def transition(
    review: AfterSaleReview,
    to: ReviewState,
    *,
    at: datetime,
    by: str | None = None,
    note: str | None = None,
) -> AfterSaleReview:
    """按状态机迁移，返回新副本；非法迁移抛 ActionNotAllowed（不修改原记录）。"""
    if not can_transition(review.state, to):
        raise ActionNotAllowed(f"售后审核不能从 {review.state} 迁移到 {to}（§7.3.2 状态机）")
    meta = _meta(review)
    entry: dict[str, Any] = {"from": str(review.state), "to": str(to), "at": at.isoformat()}
    # 新记录用 model_copy 生成、不走模型边界的清洗：自由文本先去掉 NUL 等控制字符，
    # 否则 PostgreSQL 的 JSONB 写不进去（内存实现照存，两种实现不一致）
    if by is not None:
        entry["by"] = strip_control_chars(by)
    if note is not None:
        entry["note"] = strip_control_chars(note)
    meta[HISTORY_KEY] = [*meta.get(HISTORY_KEY, []), entry]
    if to == ReviewState.NEGOTIATE:
        meta[NEGOTIATED_KEY] = True
    update: dict[str, Any] = {"state": to, "updated_at": at, "action_result": meta}
    if to == ReviewState.VERIFY:
        update["verify_started_at"] = at
    return review.model_copy(update=update, deep=True)


def history(review: AfterSaleReview) -> list[dict[str, Any]]:
    """迁移历史（按时间顺序）。"""
    return list(_meta(review).get(HISTORY_KEY, []))


def previous_states(review: AfterSaleReview) -> list[ReviewState]:
    """经过的状态（不含当前状态），按时间顺序。"""
    return [ReviewState(h["from"]) for h in history(review)]


def was_negotiated(review: AfterSaleReview) -> bool:
    return bool(_meta(review).get(NEGOTIATED_KEY, False))


def with_result(review: AfterSaleReview, result: ActionResult) -> AfterSaleReview:
    """记下平台写操作结果（保留历史等其他元数据）。"""
    meta = _meta(review)
    meta[RESULT_KEY] = result.model_dump(mode="json")
    return review.model_copy(update={"action_result": meta}, deep=True)


def last_result(review: AfterSaleReview) -> ActionResult | None:
    raw = _meta(review).get(RESULT_KEY)
    return ActionResult.model_validate(raw) if raw is not None else None
