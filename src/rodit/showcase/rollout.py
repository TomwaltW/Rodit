"""演示站的"上线节奏"说明（方案 §13.4 影子 → 智能辅助 → 半自动 → 全自动、§15 Phase 3 全自动先在
非工作时间试行、§5.6 半自动）：演示店铺现在在哪一步、下一步要什么条件。

口径都取生产代码，不另算：
- 当前阶段：feedback.rollout.stage_of（生效配置）；
- 半自动：feedback.rollout.semi_auto_status（开了的类型、各类型 §13.4 门槛：最近 2 个完整周、
  每周至少 10 组推荐、原样采纳率 ≥ 90%），开启的记录是演示单元组装时复核人员调用
  rodit.app.rollout.set_semi_auto 的结果（门槛没过时强制开启、写明原因，见 showcase.state）；
- 全自动生效时段：rodit.app.rollout.auto_schedule（和引擎 engine.modes 同一条规则：只在客服工作
  时间以外生效）；客服工作时间是店铺配置（showcase.data.SHOP_SETTINGS）。

不调 RolloutService.status：它的测试集条件要读线上的大模型配置（环境变量），演示站离线运行时
不读这些。影子、智能辅助的日期取回放的合成历史（showcase.seed）。
"""

from __future__ import annotations

from typing import Any

from rodit.app.rollout import auto_schedule
from rodit.core.clock import CST
from rodit.core.enums import INTENT_LABELS, Platform
from rodit.feedback.rollout import (
    SEMI_AUTO_PERIODS,
    STAGE_LABELS,
    RolloutStage,
    semi_auto_status,
    stage_of,
    week_windows,
    weekly_adoption,
)
from rodit.showcase.seed import go_live_at, history_start
from rodit.showcase.state import PlatformDemo

AUTO_NOTES: dict[Platform, str] = {
    Platform.TAOBAO: "要平台准入（模式 A：客服机器人 / AI 智能体-人机接待）；"
    "开启后只对开放的问题类型、高置信度生效，先在客服工作时间以外试行（§15 Phase 3）",
    Platform.DOUYIN: "飞鸽没有公开的消息收发接口，抖店只做智能辅助（§2.1），不开全自动",
    Platform.PDD: "模式 C 话术助手没有发送接口，拼多多不能全自动（§8.1）",
}
"""各平台全自动的可能性（§2.1 调研结论）。"""

AUTO_AVAILABLE: frozenset[Platform] = frozenset({Platform.TAOBAO})
"""有机会开全自动的平台（要平台准入）。"""


def _fmt(dt: Any) -> str | None:
    return dt.astimezone(CST).strftime("%m-%d %H:%M") if dt is not None else None


def rollout_view(demo: PlatformDemo) -> dict[str, Any]:
    """工作台上方"上线节奏"条的数据（见模块说明）。只读。"""
    with demo.lock:
        now = demo.tick()  # "现在是不是工作时间"按真实的当前时间
        unit = demo.unit
        config = unit.shop_config(demo.shop_id)
        stage = stage_of(config)
        windows = week_windows(now, SEMI_AUTO_PERIODS)
        records = [
            r
            for r in unit.repos.suggestions.list(demo.shop_id, since=windows[0][1])
            if r.shop_id == demo.shop_id
        ]
        weekly = weekly_adoption(records, now, weeks=SEMI_AUTO_PERIODS)
        semi = semi_auto_status(weekly, config, unit.config.routing)
        schedule = auto_schedule(config, now)
        replayed = demo.history.messages > 0
        shadow_since = history_start(demo.now) if replayed else None
        assist_since = go_live_at(demo.now) if replayed else None

        intents = []
        for row in semi.intents:
            if not row.enabled:
                continue
            change = demo.semi_auto.get(row.intent)
            weeks = []
            for (period, _, _), report in zip(windows, weekly, strict=True):
                stats = report.by_intent.get(row.intent)
                weeks.append(
                    {
                        "period": period,
                        "groups": stats.groups if stats else 0,
                        "as_is_rate": stats.group_as_is_rate if stats else None,
                    }
                )
            intents.append(
                {
                    "intent": row.intent.value,
                    "label": INTENT_LABELS[row.intent],
                    "gate_passed": bool(row.gate and row.gate.passed),
                    "forced": bool(change and change.forced),
                    "weeks": weeks,
                }
            )

    steps = [
        {
            "key": RolloutStage.SHADOW.value,
            "label": STAGE_LABELS[RolloutStage.SHADOW],
            "state": "current" if stage == RolloutStage.SHADOW else "done",
            "text": "生成推荐但不展示，和客服的实际回复对比（§13.4，1–2 周）"
            + (f"；本店 {shadow_since:%m-%d} 起的第 1 周" if shadow_since else ""),
        },
        {
            "key": RolloutStage.ASSIST.value,
            "label": STAGE_LABELS[RolloutStage.ASSIST],
            "state": {RolloutStage.SHADOW: "next", RolloutStage.ASSIST: "current"}.get(
                stage, "done"
            ),
            "text": "AI 起草、客服确认后发送，计入人工回复"
            + (f"；本店 {assist_since:%m-%d} 起" if assist_since else ""),
        },
        {
            "key": "semi_auto",
            "label": "半自动",
            "state": "partial" if semi.enabled else "next",
            "text": "按问题类型开：高置信度的简单问题直接填进客服输入框，客服按发送（§5.6）；"
            f"门槛是该类型连续 {semi.periods} 周原样采纳率 ≥ {semi.min_as_is:.0%}（§13.4）",
        },
        {
            "key": RolloutStage.AUTO.value,
            "label": "全自动",
            "state": "current"
            if stage == RolloutStage.AUTO
            else ("next" if demo.platform in AUTO_AVAILABLE else "unavailable"),
            "text": AUTO_NOTES[demo.platform],
        },
    ]
    return {
        "stage": stage.value,
        "stage_label": STAGE_LABELS[stage],
        "steps": steps,
        "semi_auto": {
            "threshold": semi.threshold,
            "min_as_is": semi.min_as_is,
            "periods": semi.periods,
            "min_groups": semi.min_groups,
            "intents": intents,
        },
        "auto": {
            "available": demo.platform in AUTO_AVAILABLE,
            "enabled": schedule.auto_reply_enabled,
            "schedule": schedule.auto_reply_schedule,
            "service_hours": list(schedule.service_hours),
            "timezone": schedule.timezone,
            "in_service_hours": schedule.in_service_hours,
            "active_now": schedule.auto_active_now,
            "next_switch_at": _fmt(schedule.next_switch_at),
        },
    }


__all__ = ["AUTO_AVAILABLE", "AUTO_NOTES", "rollout_view"]
