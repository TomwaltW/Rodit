"""路由（方案 §6.5、§6.6）：构造路由请求、调判断层、按顺序套规则得到 RouteDecision。

负责：T1。只依赖 rodit.core 和 rodit.decision。
"""

from rodit.routing.policy import (
    ALERT_COMPLAINT,
    ALERT_EMOTION,
    ALERT_NEED_HUMAN,
    DEFAULT_RISK_HINT,
    RISK_HINTS,
    RoutingPolicy,
    risk_hint,
)
from rodit.routing.router import NO_PROVIDER, DecisionRouter

__all__ = [
    "ALERT_COMPLAINT",
    "ALERT_EMOTION",
    "ALERT_NEED_HUMAN",
    "DEFAULT_RISK_HINT",
    "NO_PROVIDER",
    "RISK_HINTS",
    "DecisionRouter",
    "RoutingPolicy",
    "risk_hint",
]
