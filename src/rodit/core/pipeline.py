"""每种路由结果要跑哪些步骤（方案 §4.2、§5.4、§5.5、§6.6、§11.2、§11.4）。

引擎（T6）按这张表编排；生成（T4）、质检（T4）据此知道自己会在什么情况下被调用。
另有路由、引擎、生成共用的两条约定：高风险类型的风险提示（RISK_HINTS / risk_hint），
和规则 3（安抚）什么时候在安抚话术之外附一条回答（soothe_with_answer）。
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from rodit.core.enums import Intent, RouteAction
from rodit.core.models import RouteDecision


@dataclass(frozen=True)
class ActionPlan:
    retrieve: bool  # 调 EvidenceRetriever
    generate: bool  # 调 AnswerGenerator
    check: bool  # 调 QualityGate（FAQ / 模板草稿只跑硬规则，由 QualityGate 自己按 draft.kind 区分）
    # 约定：generate=True 时 check 一定为 True——任何要给客服 / 买家看的文字都要过质检（§5.5）
    per_part: bool = False  # 按 RouteDecision.parts 分条执行（规则 5）


ACTION_PLANS: dict[RouteAction, ActionPlan] = {
    # 规则 1：平台规则标准话术，不用大模型，只跑硬规则（§11.4）；检索只取平台的站外交易规则
    # （core.evidence.INTENT_NEEDS，静态文字），卡片据此显示依据"平台规则 · 站外交易"（§5.1、§5.6）
    RouteAction.OFF_PLATFORM_SCRIPT: ActionPlan(retrieve=True, generate=True, check=True),
    # 规则 2 / 投诉：只提醒客服，不给推荐（§5.1、§5.7）
    RouteAction.HUMAN_PRIORITY: ActionPlan(retrieve=False, generate=False, check=False),
    # 规则 3：安抚话术；有依据、且 soothe_with_answer 为真时再附一条回答
    RouteAction.SOOTHE: ActionPlan(retrieve=True, generate=True, check=True),
    # 规则 4：推荐 + 风险提示
    RouteAction.HIGH_RISK_ANSWER: ActionPlan(retrieve=True, generate=True, check=True),
    # 规则 5：按问题分条推荐
    RouteAction.SPLIT_MULTI: ActionPlan(retrieve=True, generate=True, check=True, per_part=True),
    # 规则 6：检索、生成、质检
    RouteAction.ANSWER: ActionPlan(retrieve=True, generate=True, check=True),
    # 规则 7：澄清问题（模板），不检索
    RouteAction.CLARIFY: ActionPlan(retrieve=False, generate=True, check=True),
    # 规则 8 / 全自动转人工：什么都不做
    RouteAction.NO_SUGGESTION: ActionPlan(retrieve=False, generate=False, check=False),
    RouteAction.TRANSFER_HUMAN: ActionPlan(retrieve=False, generate=False, check=False),
}


# ---------------------------------------------------------------- 风险提示（§5.1、§5.6、§6.6）

RISK_HINTS: dict[Intent, str] = {
    Intent.PRICE_PROMO: "涉及价格承诺，请确认",
    Intent.COMPARE_AUTHENTIC: "涉及真伪 / 对比，避免绝对化承诺",
    Intent.MODIFY_ORDER: "需要客服在后台操作改单",
    Intent.QUALITY_ISSUE: "请买家在申请售后时上传照片凭证",
    Intent.NOT_RECEIVED: "请核查物流签收细分状态",
}
"""高风险类型的风险提示（§5.1 "推荐 + 风险提示"、§5.6 推荐卡片）。
路由给整条消息加；引擎给多问题分条的每一条只加该条自己的（§6.6 规则 5）。"""

DEFAULT_RISK_HINT = "高风险问题，请核实后再发送"
"""配置里新增的高风险类型没有专门提示时用这一句。"""


def risk_hint(intent: Intent) -> str:
    return RISK_HINTS.get(intent, DEFAULT_RISK_HINT)


# ---------------------------------------------------------------- 规则 3 的回答部分


def soothe_with_answer(route: RouteDecision, disabled: Collection[Intent]) -> bool:
    """规则 3（安抚）在安抚话术之外能不能再附一条回答。

    安抚话术总是给（§5.7）；回答部分和规则 4–8 用同样的闸门：
    店铺关闭的类型不答（§9.3）、投诉不答只提醒人工（§5.1、§5.7）、
    判断层降级时高风险类型不答（§12.2）。route.high_risk 由路由按 routing.high_risk 填写。
    """
    if route.intent == Intent.COMPLAINT or route.intent in disabled:
        return False
    return not (route.degraded and route.high_risk)
