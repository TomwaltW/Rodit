"""售后（方案 §7.3）：买家侧问答依据 + 商家侧审核助手（R1–R8 规则、状态机、确认后执行）。

负责：T3。只依赖 rodit.core。
"""

from rodit.aftersale.advisor import (
    AFTERSALE_STATUS_LABELS,
    DefaultAfterSaleAdvisor,
    days_since_signed,
)
from rodit.aftersale.assistant import ReviewAssistant
from rodit.aftersale.rules import ReviewSuggestion, evaluate, repeat_count
from rodit.aftersale.state_machine import (
    TERMINAL_STATES,
    can_transition,
    history,
    last_result,
    previous_states,
    transition,
    was_negotiated,
)

__all__ = [
    "AFTERSALE_STATUS_LABELS",
    "TERMINAL_STATES",
    "DefaultAfterSaleAdvisor",
    "ReviewAssistant",
    "ReviewSuggestion",
    "can_transition",
    "days_since_signed",
    "evaluate",
    "history",
    "last_result",
    "previous_states",
    "repeat_count",
    "transition",
    "was_negotiated",
]
