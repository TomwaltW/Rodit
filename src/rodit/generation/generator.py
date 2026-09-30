"""答案生成器（方案 §5.4、§5.7、§6.6、§7.2、§7.3.1、§11.2–§11.5、§12.2）。

顺序（§5.4 "能不用大模型就不用"），命中就不再往下：
1. OFF_PLATFORM_SCRIPT（或 intent=off_platform）→ 平台规则标准话术（§11.4）；
2. CLARIFY → 按 route.top_intents 的澄清问题（规则 7）；
3. 证据包给出 clarify_question → 该澄清问题（商品 / 订单不确定，§5.2.4）；
4. greeting → 问候模板（引用客服时间政策）；
5. logistics_track 且有物流证据 → 物流模板，不调大模型（§11.2）；
6. refund_progress 且有退款进度 → 退款进度模板（§7.3.1）；
7. 没有任何证据：工具不可用 → "我帮您查一下"（§12.2）；缺订单 / 缺商品 → 澄清；否则不推荐（§5.7）；
8. not_received 且物流显示已签收 → 签收未收到模板放第一条（§7.3.1、§11.3：签收细分状态、
   驿站 / 快递柜的取件指引、还是没找到就在订单页申请售后），其余名额照常出 FAQ / 大模型；
9. FAQ 相似度 ≥ faq_direct_min_score → 商家审核过的标准回答原文；
10. 其余交给大模型：一条直接回答，名额够时再加一条附带追问（§5.4 "1–3 条候选"）。

SOOTHE（规则 3）先放一条安抚话术，剩下的名额按上面的顺序出回答（没有依据就只有安抚话术）；
回答部分和规则 4–8 用同样的闸门（core.pipeline.soothe_with_answer）：店铺关闭的类型、投诉、
判断层降级时的高风险类型只给安抚话术，不调大模型（§9.3、§5.7、§12.2）。
大模型失败（LLMError，含输出无法解析）时退回已有的 FAQ / 模板草稿，都没有就返回空列表（§12.2）。
返回的草稿最多 ctx.config.suggestion_count 条；不加全自动的 AI 标识前缀（引擎加）。

时间预算（§12.1、§12.2，core.budget；引擎给每条消息设截止时间）：
- 几条大模型草稿（直接回答、附带追问）并行生成，按原顺序收回：和逐条调用的结果一样——
  前面一条调用失败（LLMError）就不用后面的，前面一条是追问（need_clarify）也不用后面的；
- 剩余时间不够调一次大模型（core.budget.llm_allowed() 为 False）→ 不调，退回 FAQ / 模板；
- 最多等到截止时间：到点还没返回的草稿丢掉（线程在后台跑完，结果不用），照常退回模板；
- 单次调用的超时不超过剩余时间（prompt.build_generation_request 里 clamp_timeout）。
没有截止时间（离线任务、测试）时等全部返回。
"""

from __future__ import annotations

import concurrent.futures as cf

from rodit.core import budget
from rodit.core.config import GenerationConfig
from rodit.core.enums import DraftKind, Intent, LogisticsStatus, RouteAction
from rodit.core.errors import LLMError
from rodit.core.evidence import LLM_FREE_INTENTS, MISSING_ITEM, MISSING_ORDER, PlatformRuleKey
from rodit.core.models import ConversationContext, Draft, Evidence, EvidenceBundle, RouteDecision
from rodit.core.pipeline import soothe_with_answer
from rodit.core.ports import LLMClient, LLMRequest, LLMResponse
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


class _Unfinished:
    """到截止时间还没返回的大模型调用（结果丢掉）。"""


_UNFINISHED = _Unfinished()

_CallResult = LLMResponse | BaseException | _Unfinished


def _complete_concurrently(llm: LLMClient, requests: list[LLMRequest]) -> list[_CallResult]:
    """并行调用大模型，按原顺序返回：LLMResponse、调用抛出的异常，或 _UNFINISHED（到截止时间
    还没返回）。任务带着调用方的 contextvars（店铺范围、截止时间，core.budget.submit_in_context）；
    在引擎的 background_scope 里时，到点没返回、留在后台的调用记在 BackgroundWork 里（取消订购
    等它们做完再删数据）。没有截止时间、只有一条请求时直接在调用方线程里调。"""
    if not requests:
        return []
    deadline = budget.current_deadline()
    if len(requests) == 1 and deadline is None:
        try:
            return [llm.complete(requests[0])]
        except Exception as exc:  # noqa: BLE001 — 交给调用方按顺序处理
            return [exc]
    pool = cf.ThreadPoolExecutor(max_workers=len(requests), thread_name_prefix="rodit-generate")
    try:
        futures = [budget.submit_in_context(pool, llm.complete, r) for r in requests]
    finally:
        pool.shutdown(wait=False)  # 到点没返回的调用留在后台跑完，不等
    cf.wait(futures, timeout=None if deadline is None else deadline.remaining())
    out: list[_CallResult] = []
    for future in futures:
        if not future.done():
            out.append(_UNFINISHED)
            continue
        error = future.exception()
        out.append(error if error is not None else future.result())
    return out


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
            if not soothe_with_answer(route, ctx.config.disabled_intents):
                return drafts  # 店铺关闭 / 投诉 / 降级时的高风险类型：只给安抚话术
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

        apply_rule = V.platform_rule(evidence, ctx.platform, PlatformRuleKey.APPLY_AFTERSALE)
        # 签收未收到（§7.3.1、§11.3）：物流显示已签收时，确定性模板放第一条（还没签收的照常回答）
        lost = (
            [T.not_received(ctx, *logistics, apply_rule=apply_rule)]
            if route.intent == Intent.NOT_RECEIVED
            and logistics is not None
            and logistics[1].status == LogisticsStatus.SIGNED
            else []
        )
        faq = [
            self._faq_draft(ev) for ev in V.faq_hits(evidence, self._config.faq_direct_min_score)
        ]
        faq = _dedupe(faq)
        if faq and not self._llm_alongside_faq:
            return _dedupe([*lost, *faq])[:slots]

        primary: list[Draft] = [*lost, *faq]
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
        if logistics is not None and not lost:  # 签收未收到模板里已经有物流进度
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
        variants = _LLM_VARIANTS[: max(slots, 0)]
        if not variants or not budget.llm_allowed():
            return []  # 没有名额，或剩余时间不够调一次大模型：退回 FAQ / 模板（§12.2）
        requests = [
            build_generation_request(
                ctx,
                route,
                evidence,
                variant=variant,
                config=self._config,
                retry_feedback=retry_feedback,
            )
            for variant in variants
        ]
        drafts: list[Draft] = []
        # 并行调用，按原顺序处理：结果和逐条调用一样（见模块说明）
        for variant, result in zip(
            variants, _complete_concurrently(self._llm, requests), strict=True
        ):
            if isinstance(result, _Unfinished):
                continue  # 到截止时间还没返回：丢掉（§12.2）
            if isinstance(result, LLMError):
                break  # 调用失败 / 超时：后面的不用，退回模板
            if isinstance(result, BaseException):
                raise result  # 意外错误照常抛出，引擎按生成失败降级
            try:
                output = parse_generation_output(result.text)
            except LLMError:
                continue  # 这一条格式不对，下一条照常用
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
