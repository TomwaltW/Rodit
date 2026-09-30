"""规则判断层 RuleProvider（方案 §6.4 "规则与关键词：兜底与硬规则"、§12.2 降级路由）。

判断层不可用时路由退回它（§12.2）。只回答 §6.5 的四个路由问题
（intent / need_human / emotion / multi_intent），别的问题一律 DecisionError。
追问 second_intent（core.questions.followup_questions）只问投票制的大模型判断层：
这里 intent 的概率本来就分摊到消息里认出的各个类型上，规则 5 直接用它分条，路由器不问规则追问。

- intent：关键词命中 → 按权重归一成概率（命中类型共分 RULE_TOP_MASS，其余给 other）；
  规则没有校准过的概率，最高置信度 0.9，低于任何全自动阈值（§6.6），规则永远不会触发全自动发送。
  都没命中 → other 0.4，路由落到规则 7 / 8（澄清或不推荐）。
- off_platform：用 core.lexicon.find_off_platform_request（与质检同一份词表），命中时 P = 0.9，
  保证 §6.6 规则 1 优先；只由脱敏占位符（如"[手机号]"）引起的命中算弱信号，
  被其他问题的具体问法完整覆盖的命中（如"改个手机号"）不算站外请求。
- need_human / emotion / multi_intent：见 keywords 模块的词表和 analyze() 的说明。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rodit.core.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
    ScoreQuestion,
    validate_response,
)
from rodit.core.enums import Intent
from rodit.core.errors import DecisionError
from rodit.core.lexicon import find_off_platform_request
from rodit.core.pii import (
    ADDRESS_MASK,
    CARD_MASK,
    CONTACT_MASK,
    EMAIL_MASK,
    ID_MASK,
    PHONE_MASK,
)
from rodit.core.questions import Q_EMOTION, Q_INTENT, Q_MULTI_INTENT, Q_NEED_HUMAN
from rodit.decision._common import current_buyer_text, normalize
from rodit.decision.keywords import (
    EMOTION_PUNCT,
    EMOTION_RULES,
    INTENT_GROUPS,
    INTENT_KEYWORDS,
    NEED_HUMAN_NEGATED,
    NEED_HUMAN_P_NEGATED,
    NEED_HUMAN_P_NONE,
    NEED_HUMAN_P_STRONG,
    NEED_HUMAN_P_WEAK,
    NEED_HUMAN_STRONG,
    NEED_HUMAN_WEAK,
    TIE_BREAK,
)

ROUTER_QUESTION_TYPES: dict[str, str] = {
    Q_INTENT: "choice",
    Q_NEED_HUMAN: "noul",
    Q_EMOTION: "score",
    Q_MULTI_INTENT: "noul",
}
"""规则判断层能回答的问题及题型（§6.5 路由请求）。"""

RULE_TOP_MASS = 0.9  # 命中的问题类型共分这么多概率，剩下的给 other
OFF_PLATFORM_P = 0.9  # 明确的站外联系 / 交易请求
NO_MATCH_OTHER_P = 0.4  # 都没命中：other 低置信度，路由落到澄清 / 不推荐
WEAK_OFF_PLATFORM_SCORE = 2  # 只由脱敏占位符引起的站外命中，按普通关键词记分

MULTI_P_CROSS_CLAUSE = 0.9  # 不同分句问了不同话题
MULTI_P_SAME_CLAUSE = 0.75  # 同一句里两个话题都有明确问法
MULTI_P_WEAK = 0.3  # 有第二个话题的弱信号，或多个问句里有没认出来的
MULTI_P_NONE = 0.05

_MASKS = (PHONE_MASK, ID_MASK, EMAIL_MASK, CARD_MASK, ADDRESS_MASK, CONTACT_MASK)
_TIE_INDEX = {intent: i for i, intent in enumerate(TIE_BREAK)}
_BOUNDARY = re.compile(r"[，,。.！!？?；;\n~～…\s]+|(?<=[吗呢嘛])(?![，,。.！!？?；;\n~～…\s])")
_INTERROGATIVE = re.compile(
    r"什么|怎么|咋|多少|几|哪|能不能|可不可以|有没有|是不是|会不会|多久|多大|多高|多重|[吗呢嘛么]$"
)


@dataclass(frozen=True)
class KeywordMatch:
    """一次被采纳的关键词命中（片段不与其他采纳的命中重叠）。"""

    start: int
    end: int
    intent: Intent
    weight: int


@dataclass(frozen=True)
class RuleAnalysis:
    """一条买家消息的规则分析结果（RuleProvider 据此作答，也便于测试和排查）。"""

    text: str  # 归一后的文本（已脱敏）
    matches: tuple[KeywordMatch, ...]
    scores: dict[Intent, float]  # 合并同话题之后的得分
    probabilities: dict[Intent, float]
    ranking: tuple[Intent, ...]  # 概率降序；同分先按在消息里出现的先后，再按 TIE_BREAK
    choice: Intent
    confidence: float
    off_platform_hits: tuple[str, ...]
    off_platform_strong: bool
    need_human: float
    emotion: float
    multi_intent: float


def _tie(intent: Intent) -> int:
    return _TIE_INDEX.get(intent, len(_TIE_INDEX))


def _strip_masks(text: str) -> str:
    """把脱敏占位符换成空格：占位符里的字（"手机号""地址"）不该被当成买家说的话。"""
    for mask in _MASKS:
        text = text.replace(normalize(mask), " ")
    return text


def _match_keywords(text: str) -> tuple[KeywordMatch, ...]:
    candidates: list[KeywordMatch] = []
    for intent, rules in INTENT_KEYWORDS.items():
        for pattern, weight in rules:
            for m in pattern.finditer(text):
                if m.end() > m.start():
                    candidates.append(KeywordMatch(m.start(), m.end(), intent, weight))
    # 权重高、片段长的先占位；同样的再按出现位置、类型优先级
    candidates.sort(key=lambda c: (-c.weight, -(c.end - c.start), c.start, _tie(c.intent)))
    occupied = [False] * len(text)
    accepted: list[KeywordMatch] = []
    for c in candidates:
        if any(occupied[c.start : c.end]):
            continue
        for i in range(c.start, c.end):
            occupied[i] = True
        accepted.append(c)
    return tuple(sorted(accepted, key=lambda c: c.start))


def _occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = text.find(needle)
    while needle and start >= 0:
        spans.append((start, start + len(needle)))
        start = text.find(needle, start + 1)
    return spans


def _off_platform(
    masked: str, clean: str, matches: tuple[KeywordMatch, ...]
) -> tuple[tuple[str, ...], bool, bool]:
    """返回 (命中片段, 是否强信号, 是否弱信号)。"""
    covered = [(m.start, m.end) for m in matches if m.weight >= 3]
    clean_hits = find_off_platform_request(clean)
    strong = False
    for hit in clean_hits:
        spans = _occurrences(clean, normalize(hit))
        # 定位不到的命中按强信号处理（宁可按站外请求回标准话术）
        if not spans or any(not any(cs <= s and e <= ce for cs, ce in covered) for s, e in spans):
            strong = True
    if clean_hits:
        return tuple(clean_hits), strong, False
    masked_hits = find_off_platform_request(masked)
    return tuple(masked_hits), False, bool(masked_hits)


def _merge_scores(
    matches: tuple[KeywordMatch, ...], weak_off: bool
) -> tuple[dict[Intent, float], dict[Intent, int]]:
    """返回 (合并后的得分, 每个类型第一次出现的位置)。"""
    raw: dict[Intent, float] = {}
    first: dict[Intent, int] = {}
    for m in matches:
        raw[m.intent] = raw.get(m.intent, 0.0) + m.weight
        first[m.intent] = min(first.get(m.intent, m.start), m.start)
    if weak_off:
        raw[Intent.OFF_PLATFORM] = raw.get(Intent.OFF_PLATFORM, 0.0) + WEAK_OFF_PLATFORM_SCORE
    # 问候只是客套：同时问了别的问题时不算
    if any(i != Intent.GREETING for i in raw):
        raw.pop(Intent.GREETING, None)
    # 同一话题只保留得分最高的类型，其余分数并入它（"什么时候发？怎么还没发"是一个问题）
    merged: dict[Intent, float] = {}
    by_group: dict[str, list[Intent]] = {}
    for intent in raw:
        group = INTENT_GROUPS.get(intent)
        if group is None:
            merged[intent] = raw[intent]
        else:
            by_group.setdefault(group, []).append(intent)
    for intents in by_group.values():
        winner = min(intents, key=lambda i: (-raw[i], first[i], _tie(i)))
        merged[winner] = sum(raw[i] for i in intents)
        first[winner] = min(first.get(i, 0) for i in intents)
    return merged, first


def _probabilities(scores: dict[Intent, float], strong_off: bool) -> dict[Intent, float]:
    if strong_off:
        others = {i: s for i, s in scores.items() if i != Intent.OFF_PLATFORM}
        probs = {Intent.OFF_PLATFORM: OFF_PLATFORM_P}
        rest = 1.0 - OFF_PLATFORM_P
        if others:
            total = sum(others.values())
            for i, s in others.items():
                probs[i] = rest * 0.8 * s / total
            probs[Intent.OTHER] = rest * 0.2
        else:
            probs[Intent.OTHER] = rest
    elif scores:
        total = sum(scores.values())
        probs = {i: RULE_TOP_MASS * s / total for i, s in scores.items()}
        probs[Intent.OTHER] = probs.get(Intent.OTHER, 0.0) + (1.0 - RULE_TOP_MASS)
    else:
        probs = {Intent.OTHER: NO_MATCH_OTHER_P}
    return {i: round(p, 4) for i, p in probs.items()}


def _need_human(text: str) -> float:
    negated = NEED_HUMAN_NEGATED.search(text) is not None
    rest = NEED_HUMAN_NEGATED.sub(" ", text)
    if NEED_HUMAN_STRONG.search(rest):
        return NEED_HUMAN_P_STRONG
    if negated:
        return NEED_HUMAN_P_NEGATED
    if NEED_HUMAN_WEAK.search(rest):
        return NEED_HUMAN_P_WEAK
    return NEED_HUMAN_P_NONE


def _emotion(clean: str, masked: str) -> float:
    score = 0
    for pattern, weight in EMOTION_RULES:
        score += weight * len({m.group(0) for m in pattern.finditer(clean)})
    if EMOTION_PUNCT.search(masked):
        score += 1
    return float(min(score, 4))


def _clauses(text: str) -> list[tuple[int, int, bool]]:
    """按标点、空白和"吗 / 呢 / 嘛"切分句，返回 (起, 止, 是否问句)。"""
    out: list[tuple[int, int, bool]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        if m.start() > start:
            seg = text[start : m.start()]
            is_q = "?" in m.group(0) or bool(_INTERROGATIVE.search(seg))
            out.append((start, m.start(), is_q))
        start = m.end()
    if start < len(text):
        seg = text[start:]
        out.append((start, len(text), bool(_INTERROGATIVE.search(seg))))
    return out


def _multi_intent(text: str, matches: tuple[KeywordMatch, ...]) -> float:
    countable = [m for m in matches if m.intent in INTENT_GROUPS]
    all_groups = {INTENT_GROUPS[m.intent] for m in countable}
    strong_groups = {INTENT_GROUPS[m.intent] for m in countable if m.weight >= 2}
    clause_groups: set[str] = set()
    unmatched_questions = 0
    questions = 0
    for start, end, is_q in _clauses(text):
        inside = [m for m in countable if start <= m.start < end]
        questions += is_q
        if is_q and not inside:
            unmatched_questions += 1
        strong = [m for m in inside if m.weight >= 2]
        if strong:
            best = max(strong, key=lambda m: (m.weight, m.end - m.start, -_tie(m.intent)))
            clause_groups.add(INTENT_GROUPS[best.intent])
    if len(clause_groups) >= 2:
        return MULTI_P_CROSS_CLAUSE
    if len(strong_groups) >= 2:
        return MULTI_P_SAME_CLAUSE
    if len(all_groups) >= 2 or (all_groups and questions >= 2 and unmatched_questions):
        return MULTI_P_WEAK
    return MULTI_P_NONE


def analyze(text: str) -> RuleAnalysis:
    """规则分析一条（已脱敏的）买家消息。"""
    masked = normalize(text)
    clean = _strip_masks(masked)
    matches = _match_keywords(clean)
    hits, strong_off, weak_off = _off_platform(masked, clean, matches)
    scores, first = _merge_scores(matches, weak_off)
    probs = _probabilities(scores, strong_off)
    ranking = tuple(sorted(probs, key=lambda i: (-probs[i], first.get(i, len(clean)), _tie(i))))
    return RuleAnalysis(
        text=masked,
        matches=matches,
        scores=scores,
        probabilities=probs,
        ranking=ranking,
        choice=ranking[0],
        confidence=probs[ranking[0]],
        off_platform_hits=hits,
        off_platform_strong=strong_off,
        need_human=_need_human(clean),
        emotion=_emotion(clean, masked),
        multi_intent=_multi_intent(clean, matches),
    )


class RuleProvider:
    """关键词规则判断层（实现 core.decision.DecisionProvider）。

    在平台云内运行、不调用外部服务，可以处理买家数据（accepts_buyer_data=True）。
    只看当前买家消息（state.messages 最后一条买家消息，已由 core.states 脱敏）。
    """

    accepts_buyer_data = True

    def __init__(self, *, name: str = "rules") -> None:
        self.name = name

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        unknown = sorted(set(request.questions) - set(ROUTER_QUESTION_TYPES))
        if unknown:
            raise DecisionError(f"规则判断层只回答路由问题（§6.5），不认识：{unknown}")
        for key, question in request.questions.items():
            if question.type != ROUTER_QUESTION_TYPES[key]:
                raise DecisionError(
                    f"问题 {key!r} 应为 {ROUTER_QUESTION_TYPES[key]} 题，实际是 {question.type}"
                )
        result = analyze(current_buyer_text(request.state))
        answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {}
        for key, question in request.questions.items():
            if key == Q_INTENT:
                assert isinstance(question, ChoiceQuestion)
                answers[key] = _intent_answer(result, question)
            elif key == Q_NEED_HUMAN:
                answers[key] = NoulAnswer(noul=result.need_human)
            elif key == Q_MULTI_INTENT:
                answers[key] = NoulAnswer(noul=result.multi_intent)
            else:
                assert isinstance(question, ScoreQuestion)
                answers[key] = ScoreAnswer(score=min(result.emotion, float(question.max_score)))
        response = DecisionResponse(model=request.model, answers=answers)
        validate_response(request, response)
        return response


def _intent_answer(result: RuleAnalysis, question: ChoiceQuestion) -> ChoiceAnswer:
    ranked = [i.value for i in result.ranking if i.value in question.criteria]
    if not ranked:
        raise DecisionError("规则判断层的候选类型都不在问题选项里")
    probs = {k: result.probabilities[Intent(k)] for k in ranked}
    return ChoiceAnswer(choice=ranked[0], confidence=probs[ranked[0]], probabilities=probs)
