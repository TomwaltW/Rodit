"""质检把关（方案 §5.5）：第一道硬规则，第二道判断层质检，按放行规则决定推不推荐。

- FAQ / 模板草稿只跑硬规则（商家审核过的标准回答和确定性模板，§11.2、§11.4），不调判断层；
- 大模型草稿：两道都跑，原因合在一起（§11.5 "promise_risk 0.86，而且'明天''一定'在证据里
  找不到 → 拦截"）。硬规则命中照样拦截；判断层的分数也记下来，重新生成时原因更全，
  也是 §13.3 校准阈值要用的数据。阈值见 QAGateConfig，边界值算通过；
- 判断层不可用（未配置、DecisionError / LLMError、答案不完整）→ degraded=True，大模型草稿不放行
  （§12.2：质检不可用时不推荐未经质检的生成内容），原因里加"质检不可用"；
- 跑了判断层质检时写一条 decision_log（stage=qa），不写 audit_log（引擎写）；
  写失败只记警告（只记异常类型），质检结论照常返回；
- 进判断层的回答先遮个人信息（回答可能复述了买家的手机号、地址，§5.3）；
  QAResult.text 不遮，它是要展示给客服、发给这位买家的文本。
"""

from __future__ import annotations

import logging
import time

from rodit.core.clock import Clock
from rodit.core.config import QAGateConfig
from rodit.core.decision import (
    DecisionProvider,
    DecisionRequest,
    DecisionResponse,
    require_buyer_data_ok,
    validate_response,
)
from rodit.core.enums import DecisionStage, DraftKind
from rodit.core.errors import DecisionError, LLMError
from rodit.core.models import (
    ConversationContext,
    DecisionLogEntry,
    Draft,
    EvidenceBundle,
    HardRuleHit,
    QAResult,
    RouteDecision,
)
from rodit.core.ports import AuditSink
from rodit.core.questions import (
    Q_ANSWERS_QUESTION,
    Q_GROUNDED,
    Q_PLATFORM_VIOLATION,
    Q_PROMISE_RISK,
    Q_TONE,
    QA_MODEL,
    QA_QUESTIONS,
    QA_QUESTIONS_VER,
)
from rodit.core.states import build_qa_state
from rodit.qa.hard_rules import blocking_hits, check_hard_rules
from rodit.qa.masking import mask_text, masked_context, masked_evidence

logger = logging.getLogger(__name__)

QA_UNAVAILABLE = "质检不可用"
EMPTY_ANSWER = "回答为空"


def model_reasons(
    scores: dict[str, float], config: QAGateConfig, *, clarify: bool = False
) -> list[str]:
    """§5.5 放行规则：返回不通过的原因（空列表表示通过）。边界值算通过。

    clarify=True（大模型判断依据不够、回答是一句追问，§5.4 need_clarify）时不看 answers_question：
    追问本来就不直接回答问题。分数照常记录。
    """
    reasons: list[str] = []
    grounded = scores[Q_GROUNDED]
    if grounded < config.grounded_min:
        reasons.append(
            f"回答里有依据中找不到的内容（grounded {grounded:.2f}，"
            f"要求不低于 {config.grounded_min}）"
        )
    answers = scores[Q_ANSWERS_QUESTION]
    if not clarify and answers < config.answers_question_min:
        reasons.append(
            f"没有回应买家的问题（answers_question {answers:.2f}，"
            f"要求不低于 {config.answers_question_min}）"
        )
    promise = scores[Q_PROMISE_RISK]
    if promise > config.promise_risk_max:
        reasons.append(
            f"包含依据之外的承诺（价格、赠品、时效或赔偿；promise_risk {promise:.2f}，"
            f"要求不高于 {config.promise_risk_max}）"
        )
    violation = scores[Q_PLATFORM_VIOLATION]
    if violation > config.platform_violation_max:
        reasons.append(
            f"可能包含或引导站外联系、站外交易（platform_violation {violation:.2f}，"
            f"要求不高于 {config.platform_violation_max}）"
        )
    return reasons


def _scores(response: DecisionResponse) -> dict[str, float]:
    return {
        Q_GROUNDED: response.noul(Q_GROUNDED),
        Q_ANSWERS_QUESTION: response.noul(Q_ANSWERS_QUESTION),
        Q_PROMISE_RISK: response.noul(Q_PROMISE_RISK),
        Q_PLATFORM_VIOLATION: response.noul(Q_PLATFORM_VIOLATION),
        Q_TONE: response.score(Q_TONE),
    }


class DefaultQualityGate:
    """实现 core.ports.QualityGate。

    qa_provider 为 None 表示没有配置判断层质检：FAQ / 模板照常只跑硬规则，大模型草稿一律
    degraded、不放行。构造时拒绝不能处理买家数据的判断层实现（§6.2、§6.3）。
    """

    def __init__(
        self,
        qa_provider: DecisionProvider | None,
        config: QAGateConfig,
        audit: AuditSink,
        clock: Clock,
    ) -> None:
        if qa_provider is not None:
            require_buyer_data_ok(qa_provider)
        self._provider = qa_provider
        self._config = config
        self._audit = audit
        self._clock = clock

    def check(
        self,
        ctx: ConversationContext,
        route: RouteDecision,
        draft: Draft,
        evidence: EvidenceBundle,
    ) -> QAResult:
        text, hits = check_hard_rules(
            draft.text, ctx=ctx, draft=draft, evidence=evidence, config=self._config
        )
        reasons = [h.detail for h in blocking_hits(hits)]
        if not text.strip():
            reasons.append(EMPTY_ANSWER)
        if draft.kind != DraftKind.LLM or not text.strip():
            # FAQ / 模板只跑硬规则
            return QAResult(passed=not reasons, text=text, hard_hits=hits, reasons=reasons)
        if self._provider is None:
            return self._degraded(text, hits, reasons, provider=None)
        return self._model_check(ctx, draft, evidence, text, hits, reasons)

    # ---- 第二道：判断层质检
    def _model_check(
        self,
        ctx: ConversationContext,
        draft: Draft,
        evidence: EvidenceBundle,
        text: str,
        hits: list[HardRuleHit],
        hard_reasons: list[str],
    ) -> QAResult:
        assert self._provider is not None
        provider = self._provider
        # 对话、依据、回答（可能复述了买家的手机号、地址）进判断层前都遮掉（§5.3）
        state = build_qa_state(
            masked_context(ctx),
            draft,
            masked_evidence(evidence),
            answer_text=mask_text(text),
        )
        request = DecisionRequest(model=QA_MODEL, state=state, questions=dict(QA_QUESTIONS))
        started = time.perf_counter()
        try:
            response = provider.decide(request)
            validate_response(request, response)
            scores = _scores(response)
        except (DecisionError, LLMError, KeyError, TypeError):
            return self._degraded(text, hits, hard_reasons, provider=provider.name)
        latency_ms = int((time.perf_counter() - started) * 1000)
        clarify = draft.need_clarify or draft.variant == "clarify"
        reasons = [*hard_reasons, *model_reasons(scores, self._config, clarify=clarify)]
        try:
            self._audit.log_decision(
                DecisionLogEntry(
                    shop_id=ctx.shop_id,
                    conversation_id=ctx.conversation_id,
                    stage=DecisionStage.QA,
                    provider=provider.name,
                    questions_ver=QA_QUESTIONS_VER,
                    answers={k: a.model_dump(mode="json") for k, a in response.answers.items()},
                    route_result="block" if reasons else "pass",
                    latency_ms=latency_ms,
                    created_at=self._clock.now(),
                )
            )
        except Exception as exc:  # noqa: BLE001 — 留痕失败不影响质检结论（§12.2）
            # 只记异常类型，不记异常消息（可能带买家原文，§12.3 日志脱敏）
            logger.warning("decision_log（qa）写入失败（%s）", type(exc).__name__)
        return QAResult(
            passed=not reasons,
            text=text,
            hard_hits=hits,
            scores=scores,
            reasons=reasons,
            model_checked=True,
            provider=provider.name,
            answers=response,
        )

    def _degraded(
        self,
        text: str,
        hits: list[HardRuleHit],
        hard_reasons: list[str],
        *,
        provider: str | None,
    ) -> QAResult:
        return QAResult(
            passed=False,
            text=text,
            hard_hits=hits,
            reasons=[*hard_reasons, QA_UNAVAILABLE],
            degraded=True,
            provider=provider,
        )
