"""本条消息的生效模式（方案 §5.6 输出模式、§13.4 上线节奏、§15 Phase 3 "全自动……先在非工作时间
试行"、§4.1 原则 3 人在回路）。ConversationContext.mode 的规则，引擎决定，不直接信任接入层：

1. 店铺开了影子模式 → SHADOW（§13.4 上线第一阶段，优先于一切）；
2. 消息是 AUTO（接入层：模式 A、开了全自动发送、平台说明这条由机器人接待）且店铺开了全自动
   （auto_reply_enabled）→ 再看全自动生效时段（ShopConfig.auto_reply_schedule）：
   - always：AUTO；
   - off_hours（默认）：只在客服工作时间（ShopConfig.service_hours，北京时间，core.schedule）以外
     才是 AUTO。工作时间内 → ASSIST，拦下原因 "service_hours"（客服在岗）；没填工作时间 →
     ASSIST，拦下原因 "no_service_hours"（不知道客服在不在，不知道就不自动）；不知道处理时刻
     （调用方没给时间）→ ASSIST，拦下原因 "unknown_time"。工作时间的边界：开始那一分钟算工作时间，
     结束那一分钟不算（"09:00-23:00" 的 23:00 起是非工作时间），跨午夜的时段照常；
3. 其余 ASSIST（店铺没开全自动时，平台说是机器人接待也只按智能辅助处理，§3 D2）。

被拦下（auto_gate 不为空）的消息由客服接管（§12.2 全自动 → 切回智能辅助、由客服接管；§5.7）：
引擎照常出推荐，并转人工——模式 A 的会话在平台上是机器人接待，不转过去就没人回。
engine.result 的 detail 另加可选键 auto_gate（core.audit.RESULT_AUTO_GATE，在
OPTIONAL_RESULT_DETAIL_KEYS 里；只在被拦下时写，取值是上面的原因，不含买家原文）。
路由的 auto_eligible 按生效模式算（routing.policy：模式不是 AUTO 时为 False），两边总是一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rodit.core.audit import RESULT_AUTO_GATE  # 契约在 core；这里照旧导出，原来的引用不用改
from rodit.core.enums import Mode
from rodit.core.models import ShopConfig
from rodit.core.schedule import within_service_hours

AUTO_GATE_SERVICE_HOURS = "service_hours"
"""全自动被拦下：现在是客服工作时间（客服在岗，§15 Phase 3 先在非工作时间试行）。"""
AUTO_GATE_NO_SERVICE_HOURS = "no_service_hours"
"""全自动被拦下：店铺没填客服工作时间（附录 C 第 13 题），不知道客服在不在，按不自动处理。"""
AUTO_GATE_UNKNOWN_TIME = "unknown_time"
"""全自动被拦下：调用方没给处理时刻，没法判断是不是工作时间。"""

AUTO_GATES: tuple[str, ...] = (
    AUTO_GATE_SERVICE_HOURS,
    AUTO_GATE_NO_SERVICE_HOURS,
    AUTO_GATE_UNKNOWN_TIME,
)


@dataclass(frozen=True)
class ModeDecision:
    """生效模式 + 全自动被拦下的原因（没被拦下时为 None）。"""

    mode: Mode
    auto_gate: str | None = None


def auto_gate(config: ShopConfig, at: datetime | None) -> str | None:
    """全自动此刻能不能生效（只看生效时段，不看总开关）：能 → None，不能 → 拦下原因。"""
    if config.auto_reply_schedule == "always":
        return None
    if not config.service_hours:
        return AUTO_GATE_NO_SERVICE_HOURS
    if at is None:
        return AUTO_GATE_UNKNOWN_TIME
    inside = within_service_hours(config.service_hours, at)
    if inside is None:  # ShopConfig 已校验过时段，走不到这里；兜底按不知道处理
        return AUTO_GATE_NO_SERVICE_HOURS
    return AUTO_GATE_SERVICE_HOURS if inside else None


def resolve_mode(msg_mode: Mode, config: ShopConfig, at: datetime | None) -> ModeDecision:
    """本条消息的生效模式（规则见模块文档）。at 是处理这条消息的时刻（带时区）。"""
    if config.shadow_mode:
        return ModeDecision(Mode.SHADOW)
    if msg_mode != Mode.AUTO or not config.auto_reply_enabled:
        return ModeDecision(Mode.ASSIST)
    gate = auto_gate(config, at)
    if gate is not None:
        return ModeDecision(Mode.ASSIST, gate)
    return ModeDecision(Mode.AUTO)


__all__ = [
    "AUTO_GATES",
    "AUTO_GATE_NO_SERVICE_HOURS",
    "AUTO_GATE_SERVICE_HOURS",
    "AUTO_GATE_UNKNOWN_TIME",
    "RESULT_AUTO_GATE",
    "ModeDecision",
    "auto_gate",
    "resolve_mode",
]
