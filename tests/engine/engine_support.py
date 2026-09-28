"""问答引擎测试的替身与组装（conftest.py 把本目录加进 sys.path，测试里直接 import）。

路由、检索、生成、质检的真实实现由其他模块并行开发，这里用按脚本回答的替身；
仓储、审计、平台工具用 rodit.inmemory / rodit.testing 的内存实现和合成数据。
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rodit.core import evidence as ev
from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.config import EngineConfig
from rodit.core.context import current_shop_id
from rodit.core.enums import (
    INTENT_LABELS,
    DraftKind,
    EvidenceSource,
    Intent,
    Role,
    RouteAction,
)
from rodit.core.models import (
    AuditEvent,
    ConversationContext,
    Draft,
    Evidence,
    EvidenceBundle,
    HardRuleHit,
    QAResult,
    RouteDecision,
    RoutePart,
    StandardMessage,
)
from rodit.engine import AssistEngine
from rodit.inmemory import (
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemorySuggestionRepository,
)
from rodit.testing.fakes import FakeClock, InMemoryToolGateway, make_message
from rodit.testing.fixtures import CONV, SHOP_A, DemoWorld, build_demo_world, seed_knowledge

CLARIFY_ITEM = "请问您问的是哪一款呢？"
OFF_PLATFORM_SCRIPT = "亲，平台规定不能通过站外联系和交易哦，您直接在店铺下单就好~"
SOOTHE_SCRIPT = "亲，非常抱歉给您带来不好的体验，我马上帮您处理~"
CLARIFY_INTENT = "亲，您是想问发货时间还是物流进度呢？"
CHECK_FOR_YOU = "亲，我帮您查一下，请稍等~"

_out_seq = itertools.count(1)


# ================================================================ 构造路由结果 / 证据 / 质检结果


def make_route(
    intent: Intent,
    action: RouteAction = RouteAction.ANSWER,
    *,
    rule: int = 6,
    confidence: float = 0.92,
    parts: list[tuple[Intent, float]] | None = None,
    **kw: Any,
) -> RouteDecision:
    kw.setdefault("provider", "stub")
    return RouteDecision(
        action=action,
        rule=rule,
        intent=intent,
        intent_confidence=confidence,
        top_intents=[(intent, confidence)],
        parts=[RoutePart(intent=i, confidence=c) for i, c in (parts or [])],
        **kw,
    )


def policy_evidence(key: str = "ship_time", content: str = "现货付款后 48 小时内发货") -> Evidence:
    return Evidence(
        id=ev.policy(key),
        source=EvidenceSource.POLICY,
        label=ev.policy_label(key),
        content=content,
        data={"key": key, "value": content},
        topic=key,
    )


def bundle_of(*evidences: Evidence, **kw: Any) -> EvidenceBundle:
    return EvidenceBundle(evidences=list(evidences), **kw)


def qa_pass(draft: Draft, *, grounded: float = 0.95, **kw: Any) -> QAResult:
    llm = draft.kind == DraftKind.LLM
    return QAResult(
        passed=True,
        text=draft.text,
        scores={"grounded": grounded, "promise_risk": 0.05} if llm else {},
        model_checked=llm,
        provider="stub" if llm else None,
        **kw,
    )


def qa_fail(
    draft: Draft,
    reasons: list[str],
    *,
    rule: str | None = "absolute_term",
    **kw: Any,
) -> QAResult:
    hits = [HardRuleHit(rule=rule, detail=reasons[0] if reasons else rule)] if rule else []
    return QAResult(passed=False, text=draft.text, hard_hits=hits, reasons=reasons, **kw)


# ================================================================ 替身


class StubRouter:
    """按脚本路由：decide 是 RouteDecision 或 ctx → RouteDecision；error 不为空时抛出。"""

    def __init__(self) -> None:
        self.decide: RouteDecision | Callable[[ConversationContext], RouteDecision] = make_route(
            Intent.SHIPPING_POLICY
        )
        self.error: Exception | None = None
        self.calls: list[ConversationContext] = []
        self.shop_ids: list[str | None] = []

    def route(self, ctx: ConversationContext) -> RouteDecision:
        self.calls.append(ctx)
        self.shop_ids.append(current_shop_id())
        if self.error is not None:
            raise self.error
        return self.decide(ctx) if callable(self.decide) else self.decide


class StubRetriever:
    """按问题类型给证据包；默认给一条店铺政策。商品不确定时给澄清问题（§5.2.4）。"""

    def __init__(self) -> None:
        self.bundles: dict[Intent, EvidenceBundle | Callable[..., EvidenceBundle]] = {}
        self.error: Exception | None = None
        self.calls: list[tuple[ConversationContext, RouteDecision]] = []

    def retrieve(self, ctx: ConversationContext, route: RouteDecision) -> EvidenceBundle:
        self.calls.append((ctx, route))
        if self.error is not None:
            raise self.error
        if ctx.item is None and len(ctx.candidate_item_ids) >= 2:
            return EvidenceBundle(missing=[ev.MISSING_ITEM], clarify_question=CLARIFY_ITEM)
        scripted = self.bundles.get(route.intent)
        if callable(scripted):
            return scripted(ctx, route)
        if scripted is not None:
            return scripted.model_copy(deep=True)
        return bundle_of(policy_evidence())


class StubGenerator:
    """script 里的每一项按调用顺序依次返回（列表或异常）；用完后按默认规则生成。

    默认：澄清问题 / 站外话术 / 安抚 / 查不到时"我帮您查一下"用模板，
    其余用"大模型"根据第一条依据写。
    """

    def __init__(self) -> None:
        self.script: list[list[Draft] | Exception] = []
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        ctx: ConversationContext,
        route: RouteDecision,
        evidence: EvidenceBundle,
        *,
        retry_feedback: list[str] | None = None,
    ) -> list[Draft]:
        self.calls.append(
            {
                "intent": route.intent,
                "action": route.action,
                "retry_feedback": retry_feedback,
                "evidence_ids": evidence.ids(),
            }
        )
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return default_drafts(route, evidence)


def default_drafts(route: RouteDecision, evidence: EvidenceBundle) -> list[Draft]:
    def template(text: str, variant: str, template_id: str) -> list[Draft]:
        return [Draft(text=text, kind=DraftKind.TEMPLATE, variant=variant, template_id=template_id)]

    if evidence.clarify_question:
        return [
            Draft(
                text=evidence.clarify_question,
                kind=DraftKind.TEMPLATE,
                need_clarify=True,
                variant="clarify",
                template_id="clarify_item",
            )
        ]
    if route.action == RouteAction.OFF_PLATFORM_SCRIPT or route.intent == Intent.OFF_PLATFORM:
        return template(OFF_PLATFORM_SCRIPT, "script", "off_platform")
    if route.action == RouteAction.CLARIFY:
        return template(CLARIFY_INTENT, "clarify", "clarify_intent")
    if route.action == RouteAction.SOOTHE:
        return template(SOOTHE_SCRIPT, "soothe", "soothe")
    if evidence.is_empty and evidence.unavailable:
        return template(CHECK_FOR_YOU, "direct", "check_for_you")
    first = evidence.evidences[0] if evidence.evidences else None
    text = f"亲，{INTENT_LABELS[route.intent]}：{first.content if first else '这个我帮您确认一下'}"
    return [Draft(text=text, kind=DraftKind.LLM, evidence_ids=[first.id] if first else [])]


class StubGate:
    """默认：文字里有"一定"就按绝对化用语拦截，其余通过。rule 可以按草稿返回自定义结果。"""

    def __init__(self) -> None:
        self.rule: Callable[[Draft], QAResult | None] | None = None
        self.error: Exception | None = None
        self.error_on: Callable[[Draft], bool] | None = None
        self.calls: list[Draft] = []

    def check(
        self,
        ctx: ConversationContext,
        route: RouteDecision,
        draft: Draft,
        evidence: EvidenceBundle,
    ) -> QAResult:
        self.calls.append(draft)
        if self.error is not None and (self.error_on is None or self.error_on(draft)):
            raise self.error
        if self.rule is not None:
            result = self.rule(draft)
            if result is not None:
                return result
        if "一定" in draft.text:
            return qa_fail(draft, ["绝对化用语：一定"])
        return qa_pass(draft)


# ================================================================ 组装


@dataclass
class Kit:
    world: DemoWorld
    clock: FakeClock
    config: EngineConfig
    knowledge: InMemoryKnowledgeRepository
    conversations: InMemoryConversationRepository
    suggestions: InMemorySuggestionRepository
    gaps: InMemoryFeedbackRepository
    tools: InMemoryToolGateway
    audit: InMemoryAuditSink
    router: StubRouter
    retriever: StubRetriever
    generator: StubGenerator
    gate: StubGate
    faq_ids: dict[tuple[str, str], int] = field(default_factory=dict)

    def engine(self, *, with_gaps: bool = True, **overrides: Any) -> AssistEngine:
        deps: dict[str, Any] = {
            "config": self.config,
            "knowledge": self.knowledge,
            "conversations": self.conversations,
            "suggestions": self.suggestions,
            "tools": self.tools,
            "router": self.router,
            "retriever": self.retriever,
            "generator": self.generator,
            "gate": self.gate,
            "audit": self.audit,
            "clock": self.clock,
            "gaps": self.gaps if with_gaps else None,
        }
        deps.update(overrides)
        return AssistEngine(**deps)

    def msg(self, text: str, *, after_s: float = 60, **kw: Any) -> StandardMessage:
        """一条买家消息：先把时钟往后拨 after_s 秒（默认 60 秒，不会和上一条合并）。"""
        self.clock.advance(seconds=after_s)
        kw.setdefault("at", self.clock.now())
        return make_message(text, world=self.world, **kw)

    def agent_says(
        self,
        text: str,
        *,
        after_s: float = 30,
        shop_id: str = SHOP_A,
        conversation_id: str = CONV,
        role: Role = Role.AGENT,
    ) -> None:
        self.clock.advance(seconds=after_s)
        ok = self.conversations.add_outgoing(
            shop_id=shop_id,
            conversation_id=conversation_id,
            msg_id=f"out_{next(_out_seq):05d}",
            role=role,
            text=text,
            at=self.clock.now(),
        )
        assert ok

    def set_shop_config(self, shop_id: str, **overrides: Any) -> None:
        shop = self.knowledge.get_shop(shop_id)
        assert shop is not None
        merged = {**shop.config_overrides, **overrides}
        self.knowledge.upsert_shop(shop.model_copy(update={"config_overrides": merged}))

    def events(self, action: AuditAction, shop_id: str = SHOP_A) -> list[AuditEvent]:
        return self.audit.list_events(shop_id, action=action.value)

    def result_events(self, shop_id: str = SHOP_A) -> list[AuditEvent]:
        return self.events(AuditAction.RESULT, shop_id)


def build_kit(config: EngineConfig | None = None, *, now: datetime | None = None) -> Kit:
    world = build_demo_world(now) if now else build_demo_world()
    clock = FakeClock(world.now)
    knowledge = InMemoryKnowledgeRepository(clock)
    faq_ids = seed_knowledge(knowledge, world)
    return Kit(
        world=world,
        clock=clock,
        config=config or EngineConfig(),
        knowledge=knowledge,
        conversations=InMemoryConversationRepository(),
        suggestions=InMemorySuggestionRepository(),
        gaps=InMemoryFeedbackRepository(),
        tools=InMemoryToolGateway(world),
        audit=InMemoryAuditSink(),
        router=StubRouter(),
        retriever=StubRetriever(),
        generator=StubGenerator(),
        gate=StubGate(),
        faq_ids=faq_ids,
    )
