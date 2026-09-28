"""路由器 DecisionRouter（实现 core.ports.Router；方案 §6.5、§6.6、§12.2）。

一次路由：
1. 用 core.states.build_router_state 构造 state（唯一入口，已脱敏），
   问题取 core.questions.ROUTER_QUESTIONS；
2. 调主判断层；判断层不可用（DecisionError / LLMError）时退回 fallback（关键词规则），
   degraded=True；
   fallback 也失败 → NO_SUGGESTION（规则 8、provider="none"），不向外抛；
3. RoutingPolicy 按 §6.6 套规则；
4. 写一条 decision_log（stage=route，实际用的实现、问题版本、原始答案、路由结果、耗时）。
   audit_log 由引擎写（见 core.ports 模块文档字符串的留痕分工），这里不写。
"""

from __future__ import annotations

from datetime import datetime

from rodit.core.clock import Clock
from rodit.core.context import shop_scope
from rodit.core.decision import (
    DecisionProvider,
    DecisionRequest,
    DecisionResponse,
    require_buyer_data_ok,
    validate_response,
)
from rodit.core.enums import DecisionStage, Intent, RouteAction
from rodit.core.errors import DecisionError, LLMError
from rodit.core.models import ConversationContext, DecisionLogEntry, RouteDecision
from rodit.core.ports import AuditSink
from rodit.core.questions import ROUTER_MODEL, ROUTER_QUESTIONS, ROUTER_QUESTIONS_VER
from rodit.core.states import build_router_state
from rodit.routing.policy import RoutingPolicy

NO_PROVIDER = "none"


class DecisionRouter:
    """判断层 + §6.6 规则 → RouteDecision。

    provider 是主判断层（MVP 为 incloud_llm），fallback 是判断层不可用时的兜底（RuleProvider）。
    两者都会拿到买家数据，构造时用 require_buyer_data_ok 拒绝不能处理买家数据的实现（§6.2、§6.3）。
    """

    def __init__(
        self,
        provider: DecisionProvider,
        fallback: DecisionProvider,
        policy: RoutingPolicy,
        audit: AuditSink,
        clock: Clock,
        *,
        history_rounds: int = 5,
    ) -> None:
        require_buyer_data_ok(provider)
        require_buyer_data_ok(fallback)
        self.provider = provider
        self.fallback = fallback
        self.policy = policy
        self.audit = audit
        self.clock = clock
        self.history_rounds = history_rounds

    def build_request(self, ctx: ConversationContext) -> DecisionRequest:
        """§6.5 路由请求。"""
        return DecisionRequest(
            model=ROUTER_MODEL,
            state=build_router_state(ctx, history_rounds=self.history_rounds),
            questions=ROUTER_QUESTIONS,
        )

    @staticmethod
    def _ask(provider: DecisionProvider, request: DecisionRequest) -> DecisionResponse:
        response = provider.decide(request)
        validate_response(request, response)  # 实现有漏答 / 错题型也按不可用处理
        return response

    def route(self, ctx: ConversationContext) -> RouteDecision:
        started = self.clock.now()
        request = self.build_request(ctx)
        response: DecisionResponse | None = None
        used = NO_PROVIDER
        degraded = False
        with shop_scope(ctx.shop_id):  # 大模型成本按店铺归属（§12.4）
            try:
                response = self._ask(self.provider, request)
                used = self.provider.name
            except (DecisionError, LLMError):
                degraded = True
                try:
                    response = self._ask(self.fallback, request)
                    used = self.fallback.name
                except (DecisionError, LLMError):
                    response = None
        if response is None:
            decision = RouteDecision(
                action=RouteAction.NO_SUGGESTION,
                rule=8,
                intent=Intent.OTHER,
                intent_confidence=0.0,
                degraded=True,
                provider=NO_PROVIDER,
            )
        else:
            decision = self.policy.apply(
                response,
                mode=ctx.mode,
                shop_config=ctx.config,
                degraded=degraded,
                provider=used,
            )
        self._log(ctx, response, decision, started)
        return decision

    def _log(
        self,
        ctx: ConversationContext,
        response: DecisionResponse | None,
        decision: RouteDecision,
        started: datetime,
    ) -> None:
        now = self.clock.now()
        latency_ms = max(0, round((now - started).total_seconds() * 1000))
        self.audit.log_decision(
            DecisionLogEntry(
                shop_id=ctx.shop_id,
                conversation_id=ctx.conversation_id,
                stage=DecisionStage.ROUTE,
                provider=decision.provider,
                questions_ver=ROUTER_QUESTIONS_VER,
                answers=response.model_dump(mode="json") if response is not None else {},
                route_result=str(decision.action),
                latency_ms=latency_ms,
                created_at=now,
            )
        )
