"""路由器 DecisionRouter（实现 core.ports.Router；方案 §6.5、§6.6、§12.2）。

一次路由：
1. 用 core.states.build_router_state 构造 state（唯一入口，已脱敏），
   问题取 core.questions.ROUTER_QUESTIONS；主判断层是投票制的 InCloudLLMProvider 时换成
   ROUTER_VOTE_QUESTIONS（多一道追问 second_intent，同一个请求里答，§6.4 一票一个请求）：
   一票只能选一个 intent，认不出第二个问题；规则、分类器、Jev 的 intent 概率本来就分摊到
   各个类型上，照旧只问四个必答问题（兜底的关键词规则也一样）；
2. 调主判断层；判断层不可用（DecisionError / LLMError）时退回 fallback（关键词规则），
   degraded=True；
   fallback 也失败 → NO_SUGGESTION（规则 8、provider="none"），不向外抛；
   全自动模式下主判断层答出来了，也再问一次 fallback 的"要求人工"（关键词，不联网），
   交给 RoutingPolicy 当规则 2 的兜底（§5.7 买家要求人工 → 全自动立即转人工）；
3. RoutingPolicy 按 §6.6 套规则；主判断层是多票投票的 InCloudLLMProvider 时带上票数，
   是否题阈值按票数粒度比较（routing.policy.vote_threshold）；
4. 写一条 decision_log（stage=route，实际用的实现、问题版本、原始答案、路由结果、耗时）。
   写失败只记警告（只记异常类型），照样返回路由结果：留痕故障不能让推荐全部消失（§12.2）。
   audit_log 由引擎写（见 core.ports 模块文档字符串的留痕分工），这里不写。
"""

from __future__ import annotations

import logging
from datetime import datetime

from rodit.core.clock import Clock
from rodit.core.context import shop_scope
from rodit.core.decision import (
    DecisionProvider,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    require_buyer_data_ok,
    validate_response,
)
from rodit.core.enums import DecisionStage, Intent, Mode, RouteAction
from rodit.core.errors import DecisionError, LLMError
from rodit.core.models import ConversationContext, DecisionLogEntry, RouteDecision
from rodit.core.ports import AuditSink
from rodit.core.questions import (
    Q_NEED_HUMAN,
    ROUTER_MODEL,
    ROUTER_QUESTIONS,
    ROUTER_QUESTIONS_VER,
    with_followups,
)
from rodit.core.states import build_router_state
from rodit.decision.llm import InCloudLLMProvider
from rodit.routing.policy import RoutingPolicy

logger = logging.getLogger(__name__)

NO_PROVIDER = "none"


def vote_samples_of(provider: DecisionProvider) -> int | None:
    """是否题答案来自几票投票：多次采样的 InCloudLLMProvider 返回采样次数，其他实现返回 None
    （分类器、关键词给的是连续概率；只采样一次时用模型自报的 p）。"""
    if isinstance(provider, InCloudLLMProvider) and provider.samples > 1:
        return provider.samples
    return None


def asks_followups(provider: DecisionProvider) -> bool:
    """这个判断层要不要追问 second_intent：投票制的大模型判断层要（只采样一次也一样，
    一票只选一个 intent）；给连续概率的实现（分类器、关键词规则、Jev）不要。"""
    return isinstance(provider, InCloudLLMProvider)


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
        """§6.5 路由请求（四个必答问题）。"""
        return DecisionRequest(
            model=ROUTER_MODEL,
            state=build_router_state(ctx, history_rounds=self.history_rounds),
            questions=ROUTER_QUESTIONS,
        )

    @staticmethod
    def request_for(provider: DecisionProvider, request: DecisionRequest) -> DecisionRequest:
        """给这个判断层的请求：投票制的大模型判断层加上追问（同一个 state；路由问题即
        ROUTER_VOTE_QUESTIONS），其他照旧。"""
        if not asks_followups(provider):
            return request
        return request.model_copy(update={"questions": with_followups(request.questions)})

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
        keyword_need_human: float | None = None
        with shop_scope(ctx.shop_id):  # 大模型成本按店铺归属（§12.4）
            try:
                response = self._ask(self.provider, self.request_for(self.provider, request))
                used = self.provider.name
            except (DecisionError, LLMError):
                degraded = True
                try:
                    response = self._ask(self.fallback, self.request_for(self.fallback, request))
                    used = self.fallback.name
                except (DecisionError, LLMError):
                    response = None
            if response is not None and not degraded and ctx.mode == Mode.AUTO:
                keyword_need_human = self._fallback_need_human(
                    self.request_for(self.fallback, request)
                )
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
                vote_samples=None if degraded else vote_samples_of(self.provider),
                keyword_need_human=keyword_need_human,
            )
        self._log(ctx, response, decision, started)
        return decision

    def _fallback_need_human(self, request: DecisionRequest) -> float | None:
        """fallback（关键词规则）对"要求人工"的判断；答不出来时返回 None（不影响主判断）。"""
        try:
            answer = self._ask(self.fallback, request).answers.get(Q_NEED_HUMAN)
        except (DecisionError, LLMError):
            return None
        return answer.noul if isinstance(answer, NoulAnswer) else None

    def _log(
        self,
        ctx: ConversationContext,
        response: DecisionResponse | None,
        decision: RouteDecision,
        started: datetime,
    ) -> None:
        now = self.clock.now()
        latency_ms = max(0, round((now - started).total_seconds() * 1000))
        try:
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
        except Exception as exc:  # noqa: BLE001 — 留痕失败不影响路由结果（§12.2）
            # 只记异常类型，不记异常消息（可能带买家原文，§12.3 日志脱敏）
            logger.warning("decision_log（route）写入失败（%s）", type(exc).__name__)
