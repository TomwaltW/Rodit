"""每种路由结果要跑哪些步骤（方案 §4.2、§5.4、§5.5、§6.6、§11.2、§11.4）。

引擎（T6）按这张表编排；生成（T4）、质检（T4）据此知道自己会在什么情况下被调用。
"""

from __future__ import annotations

from dataclasses import dataclass

from rodit.core.enums import RouteAction


@dataclass(frozen=True)
class ActionPlan:
    retrieve: bool  # 调 EvidenceRetriever
    generate: bool  # 调 AnswerGenerator
    check: bool  # 调 QualityGate（FAQ / 模板草稿只跑硬规则，由 QualityGate 自己按 draft.kind 区分）
    # 约定：generate=True 时 check 一定为 True——任何要给客服 / 买家看的文字都要过质检（§5.5）
    per_part: bool = False  # 按 RouteDecision.parts 分条执行（规则 5）


ACTION_PLANS: dict[RouteAction, ActionPlan] = {
    # 规则 1：平台规则标准话术，不检索、不用大模型，只跑硬规则（§11.4）
    RouteAction.OFF_PLATFORM_SCRIPT: ActionPlan(retrieve=False, generate=True, check=True),
    # 规则 2 / 投诉：只提醒客服，不给推荐（§5.1、§5.7）
    RouteAction.HUMAN_PRIORITY: ActionPlan(retrieve=False, generate=False, check=False),
    # 规则 3：安抚话术；有依据时再附一条回答
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
