"""回答质量评分（方案 §13.2）与上线门槛。

§13.2 的四个维度（0 / 1 / 2 分）在合成评测集上按标注自动打分，是人工抽检之前的代理指标：
- 事实正确：出现 must_not_include 里的错误事实 → 0；
  must_include 要点和 must_cite 依据全满足 → 2；否则 1；
- 切题：问题类型（或允许的类型）和路由结果都对且有推荐 → 2；有推荐但类型不对 → 1；该答没答 → 0；
- 合规：回答里有站外引流或绝对化用语 → 0（core.lexicon，与质检硬规则同一份词表；
  商品卡片上原样有的成分标注如"桑蚕丝 100%"不算绝对化用语）；否则 2；
- 表达：没有推荐 → 0；超长或带 Markdown → 1；否则 2。
上线门槛：测试集上"事实正确"和"合规"两项的满分率都 ≥ 95%（§13.2）。
不该给推荐的用例（no_suggestion_ok）没给推荐时，四项都记满分。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from rodit.core.enums import Intent, RouteAction
from rodit.core.lexicon import find_absolute_terms, find_off_platform_in_answer
from rodit.core.models import EngineResult
from rodit.eval.dataset import EvalCase

_MARKDOWN = re.compile(r"\*\*|__|^#+\s|`|\[[^\]]+\]\([^)]+\)", re.M)
_ANSWER_ACTIONS = {RouteAction.ANSWER, RouteAction.HIGH_RISK_ANSWER, RouteAction.SPLIT_MULTI}


class CaseScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    split: str
    intent_ok: bool
    action_ok: bool
    predicted_intent: Intent | None
    intent_confidence: float | None
    action: RouteAction | None
    has_suggestion: bool
    factual: int = Field(ge=0, le=2)
    on_topic: int = Field(ge=0, le=2)
    compliance: int = Field(ge=0, le=2)
    expression: int = Field(ge=0, le=2)
    problems: list[str] = Field(default_factory=list)


def score_case(
    case: EvalCase,
    result: EngineResult | None,
    *,
    max_chars: int = 120,
    evidence_text: str | None = None,
) -> CaseScore:
    """evidence_text：用例涉及的商品卡片文字（合成数据），判断成分标注是否如实。"""
    route = result.route if result else None
    predicted = route.intent if route else None
    ok_intents = {case.expected_intent, *case.acceptable_intents}
    intent_ok = predicted in ok_intents
    action = route.action if route else None
    expected_actions = set(case.expected_actions) or _ANSWER_ACTIONS
    action_ok = action in expected_actions
    cards = result.suggestions if result else []
    has = bool(cards)
    problems: list[str] = []

    if not has and case.no_suggestion_ok:
        return CaseScore(
            case_id=case.id,
            split=case.split,
            intent_ok=intent_ok,
            action_ok=action_ok,
            predicted_intent=predicted,
            intent_confidence=route.intent_confidence if route else None,
            action=action,
            has_suggestion=False,
            factual=2,
            on_topic=2 if action_ok else 1,
            compliance=2,
            expression=2,
        )

    text = cards[0].text if has else ""
    all_text = "\n".join(c.text for c in cards)
    cited = {e.id for c in cards for e in c.evidence}

    # 事实正确
    wrong = [s for s in case.must_not_include if s in all_text]
    missing_points = [s for s in case.must_include if s not in all_text]
    missing_cites = [group for group in case.must_cite if not cited.intersection(group)]
    if wrong:
        factual = 0
        problems.append(f"出现不该有的内容：{wrong}")
    elif not has:
        factual = 0
        problems.append("没有推荐")
    elif missing_points or missing_cites:
        factual = 1
        if missing_points:
            problems.append(f"缺少要点：{missing_points}")
        if missing_cites:
            problems.append(f"缺少依据：{missing_cites}")
    else:
        factual = 2

    # 切题
    if has and intent_ok and action_ok:
        on_topic = 2
    elif has:
        on_topic = 1
        problems.append(f"类型 / 路由不符：{predicted} / {action}")
    else:
        on_topic = 0

    # 合规（所有候选都要合规：客服可能点任何一条）
    violations = [
        *(f"站外：{h}" for c in cards for h in find_off_platform_in_answer(c.text)),
        *(
            f"绝对化：{h}"
            for c in cards
            for h in find_absolute_terms(c.text, evidence_text=evidence_text)
        ),
    ]
    compliance = 0 if violations else 2
    problems.extend(violations)

    # 表达
    if not has:
        expression = 0
    elif len(text) > max_chars or _MARKDOWN.search(text):
        expression = 1
        problems.append("超长或带格式")
    else:
        expression = 2

    return CaseScore(
        case_id=case.id,
        split=case.split,
        intent_ok=intent_ok,
        action_ok=action_ok,
        predicted_intent=predicted,
        intent_confidence=route.intent_confidence if route else None,
        action=action,
        has_suggestion=has,
        factual=factual,
        on_topic=on_topic,
        compliance=compliance,
        expression=expression,
        problems=problems,
    )


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    intent_accuracy: float
    action_accuracy: float
    suggestion_rate: float
    full_marks: dict[str, float]  # 各维度满分率
    mean_scores: dict[str, float]
    confusion: dict[str, dict[str, int]]  # 真实类型 → 预测类型 → 次数
    passes_gate: bool  # 事实正确、合规满分率都 ≥ gate
    gate: float
    failures: list[CaseScore]


DIMENSIONS = ("factual", "on_topic", "compliance", "expression")


def summarize(
    scores: Sequence[CaseScore], cases: Sequence[EvalCase], *, gate: float = 0.95
) -> EvalReport:
    n = len(scores)
    if n == 0:
        raise ValueError("没有评测结果")
    by_id = {c.id: c for c in cases}
    confusion: dict[str, Counter[str]] = {}
    for s in scores:
        true = str(by_id[s.case_id].expected_intent)
        confusion.setdefault(true, Counter())[str(s.predicted_intent)] += 1
    full = {d: sum(1 for s in scores if getattr(s, d) == 2) / n for d in DIMENSIONS}
    mean = {d: sum(getattr(s, d) for s in scores) / n for d in DIMENSIONS}
    return EvalReport(
        n=n,
        intent_accuracy=sum(s.intent_ok for s in scores) / n,
        action_accuracy=sum(s.action_ok for s in scores) / n,
        suggestion_rate=sum(s.has_suggestion for s in scores) / n,
        full_marks={k: round(v, 4) for k, v in full.items()},
        mean_scores={k: round(v, 4) for k, v in mean.items()},
        confusion={k: dict(v) for k, v in confusion.items()},
        passes_gate=full["factual"] >= gate and full["compliance"] >= gate,
        gate=gate,
        failures=[s for s in scores if min(s.factual, s.on_topic, s.compliance, s.expression) < 2],
    )
