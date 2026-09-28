"""答案生成器（方案 §5.4、§5.7、§6.6、§7.2、§7.3.1、§11.2–§11.5、§12.2）。

顺序（§5.4 "能不用大模型就不用"），命中就不再往下：
1. OFF_PLATFORM_SCRIPT（或 intent=off_platform）→ 平台规则标准话术（§11.4）；
2. CLARIFY → 按 route.top_intents 的澄清问题（规则 7）；
3. 证据包给出 clarify_question → 该澄清问题（商品 / 订单不确定，§5.2.4）；
4. greeting → 问候模板（引用客服时间政策）；
5. logistics_track 且有物流证据 → 物流模板，不调大模型（§11.2）；
6. refund_progress 且有退款进度 → 退款进度模板（§7.3.1）；
7. 没有任何证据：工具不可用 → "我帮您查一下"（§12.2）；缺订单 / 缺商品 → 澄清；否则不推荐（§5.7）；
8. FAQ 相似度 ≥ faq_direct_min_score → 商家审核过的标准回答原文；
9. 其余交给大模型：一条直接回答，名额够时再加一条附带追问（§5.4 "1–3 条候选"）。

SOOTHE（规则 3）先放一条安抚话术，剩下的名额按上面的顺序出回答（没有依据就只有安抚话术）。
大模型失败（LLMError，含输出无法解析）时退回已有的 FAQ / 模板草稿，都没有就返回空列表（§12.2）。
返回的草稿最多 ctx.config.suggestion_count 条；不加全自动的 AI 标识前缀（引擎加）。
"""

from __future__ import annotations

from rodit.core.config import GenerationConfig
from rodit.core.enums import DraftKind, Intent, RouteAction
from rodit.core.errors import LLMError
from rodit.core.evidence import LLM_FREE_INTENTS, MISSING_ITEM, MISSING_ORDER, PlatformRuleKey
from rodit.core.models import ConversationContext, Draft, Evidence, EvidenceBundle, RouteDecision
from rodit.core.ports import LLMClient
from rodit.generation import evidence_view as V
from rodit.generation import templates as T
from rodit.generation.parse import parse_generation_output
from rodit.generation.prompt import PromptVariant, build_generation_request

_NO_GENERATION = frozenset(
    {RouteAction.HUMAN_PRIORITY, RouteAction.NO_SUGGESTION, RouteAction.TRANSFER_HUMAN}
)
_APPLY_INTENTS = frozenset(
    {Intent.RETURN_POLICY, Intent.REFUND_PROGRESS, Intent.QUALITY_ISSUE, Intent.NOT_RECEIVED}
)
_LLM_VARIANTS: tuple[PromptVariant, ...] = ("direct", "with_followup")


def _dedupe(drafts: list[Draft]) -> list[Draft]:
    seen: set[str] = set()
    out: list[Draft] = []
    for d in drafts:
        key = d.text.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


class DefaultAnswerGenerator:
    """实现 core.ports.AnswerGenerator。

    llm_alongside_faq：FAQ 直答命中后是否还用大模型补满推荐名额。默认 False，
    按 §5.4 "FAQ 命中就直接用商家审核过的标准回答，不走大模型"。
    """

    def __init__(
        self,
        llm: LLMClient,
        config: GenerationConfig,
        *,
        llm_alongside_faq: bool = False,
    ) -> None:
        self._llm = llm
        self._config = config
        self._llm_alongside_faq = llm_alongside_faq

    def generate(
        self,
        ctx: ConversationContext,
        route: RouteDecision,
        evidence: EvidenceBundle,
        *,
        retry_feedback: list[str] | None = None,
    ) -> list[Draft]:
        limit = ctx.config.suggestion_count
        if route.action in _NO_GENERATION:
            return []
        if route.action == RouteAction.OFF_PLATFORM_SCRIPT or route.intent == Intent.OFF_PLATFORM:
            return [T.off_platform_script(ctx, evidence)]
        if route.action == RouteAction.CLARIFY:
            return [T.clarify_intent(ctx, route)]
        drafts: list[Draft] = []
        if route.action == RouteAction.SOOTHE:
            drafts.append(T.soothe(ctx))
        slots = limit - len(drafts)
        if slots > 0:
            drafts.extend(self._answer(ctx, route, evidence, slots, retry_feedback))
        return _dedupe(drafts)[:limit]

    # ---- 回答（FAQ → 模板 → 大模型）
    def _answer(
        self,
        ctx: ConversationContext,
        route: RouteDecision,
        evidence: EvidenceBundle,
        slots: int,
        retry_feedback: list[str] | None,
    ) -> list[Draft]:
        if evidence.clarify_question:
            return [T.clarify_question(ctx, evidence.clarify_question)]
        if route.intent == Intent.GREETING:
            return [T.greeting(ctx, evidence)]
        if route.intent in LLM_FREE_INTENTS:  # 只用模板回答的其他类型
            return []
        logistics = V.logistics(evidence, ctx)
        if route.intent == Intent.LOGISTICS_TRACK and logistics is not None:
            return [T.logistics(ctx, *logistics)]
        progress = V.refund_progress(evidence, ctx)
        timing = V.platform_rule(evidence, ctx.platform, PlatformRuleKey.REFUND_TIMING)
        if route.intent == Intent.REFUND_PROGRESS and progress is not None:
            return [T.refund_progress(ctx, *progress, timing=timing)]
        if evidence.is_empty:
            return self._no_evidence(ctx, evidence)

        faq = [
            self._faq_draft(ev) for ev in V.faq_hits(evidence, self._config.faq_direct_min_score)
        ]
        faq = _dedupe(faq)
        if faq and not self._llm_alongside_faq:
            return faq[:slots]

        primary: list[Draft] = list(faq)
        apply_rule = V.platform_rule(evidence, ctx.platform, PlatformRuleKey.APPLY_AFTERSALE)
        apply_draft = T.apply_aftersale(ctx, apply_rule) if apply_rule is not None else None
        if (
            apply_draft is not None
            and route.intent in _APPLY_INTENTS
            and T.asks_how_to_apply(ctx.current.text)
        ):
            primary.append(apply_draft)
        primary = _dedupe(primary)[:slots]

        llm_drafts = self._llm_drafts(ctx, route, evidence, slots - len(primary), retry_feedback)
        if llm_drafts or len(primary) >= slots:
            return _dedupe([*primary, *llm_drafts])[:slots]

        # 大模型没有给出可用回答：退回手上的模板（§12.2）
        fallback: list[Draft] = []
        if apply_draft is not None and route.intent in _APPLY_INTENTS:
            fallback.append(apply_draft)
        if logistics is not None:
            fallback.append(T.logistics(ctx, *logistics))
        if progress is not None:
            fallback.append(T.refund_progress(ctx, *progress, timing=timing))
        if evidence.unavailable:
            fallback.append(T.tool_unavailable(ctx))
        return _dedupe([*primary, *fallback])[:slots]

    def _no_evidence(self, ctx: ConversationContext, evidence: EvidenceBundle) -> list[Draft]:
        """没有任何依据：能查的查不到就"我帮您查一下"，缺订单 / 商品就问，其他不推荐（§5.7）。"""
        if evidence.unavailable:
            return [T.tool_unavailable(ctx)]
        if MISSING_ORDER in evidence.missing:
            return [T.clarify_order(ctx)]
        if MISSING_ITEM in evidence.missing:
            return [T.clarify_item(ctx)]
        return []

    @staticmethod
    def _faq_draft(ev: Evidence) -> Draft:
        return Draft(
            text=V.text_of(ev, "answer"),
            evidence_ids=[ev.id],
            kind=DraftKind.FAQ,
            variant="direct",
            faq_id=V.faq_id_of(ev),
        )

    # ---- 大模型
    def _llm_drafts(
        self,
        ctx: ConversationContext,
        route: RouteDecision,
        evidence: EvidenceBundle,
        slots: int,
        retry_feedback: list[str] | None,
    ) -> list[Draft]:
        drafts: list[Draft] = []
        for variant in _LLM_VARIANTS[: max(slots, 0)]:
            request = build_generation_request(
                ctx,
                route,
                evidence,
                variant=variant,
                config=self._config,
                retry_feedback=retry_feedback,
            )
            try:
                response = self._llm.complete(request)
            except LLMError:
                break  # 调用失败 / 超时：不再重复调用，退回模板
            try:
                output = parse_generation_output(response.text)
            except LLMError:
                continue  # 这一条格式不对，下一条照常生成
            drafts.append(
                Draft(
                    text=output.answer,
                    evidence_ids=list(output.evidence_ids),
                    kind=DraftKind.LLM,
                    need_clarify=output.need_clarify,
                    variant="clarify" if output.need_clarify else variant,
                )
            )
            if output.need_clarify:
                break  # 依据不够、要先问买家：再生成一条也是追问
        return _dedupe(drafts)
