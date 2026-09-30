"""大模型判断输出的解析与自一致性投票（方案 §6.4 "大模型怎么给出概率"）。

- 解析尽量宽容：去掉 ``` 代码块围栏、取第一个完整的 {...}；
  是否题认 是 / 否 / yes / no / true / false；
  单选题认选项键（大小写、首尾空白、后面跟的中文说明都容忍）；打分题认数字或等级名称。
  解析不了的票按问题丢弃（同一次采样里别的问题照样计票）。
- 投票：单选题的概率 = 各选项在有效票里的占比，置信度 = 多数选项的占比，同票按选项定义顺序取前者；
  接口给了选项概率（LLMResponse.logprobs）时直接用它（多次采样取平均）；
  是否题 = 有效票里"是"的占比（只采样一次时用模型自报的 p）；打分题 = 有效票的平均等级。
- 追问 second_intent（core.questions.followup_questions）按票配对：同一次采样里，追问的类型和主要
  问题（intent）不同、而且这一票说有多个问题（multi_intent 是）才算这一票的第二个问题，否则这一票
  算"无"；没答或答不出来也算"无"（追问答不好只是不分条，不让整次路由降级）；
  主要问题都没答出来的采样不参与追问的计票。这样同一票里主要问题和追问不会重复，
  mention_probabilities 把两者的占比直接相加，就是每个类型"被问到"的票数占比（§6.6 规则 5 分条用）。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from rodit.core.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from rodit.core.errors import DecisionError
from rodit.core.questions import NO_SECOND_INTENT, Q_INTENT, Q_MULTI_INTENT, Q_SECOND_INTENT

_FENCE = re.compile(r"```[A-Za-z]*\s*(.*?)```", re.DOTALL)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUOTES = "\"'“”‘’「」『』`"
_TRAILING = "。.！!？?，,；;：:"

YES_WORDS = frozenset({"是", "是的", "对", "有", "会", "yes", "y", "true", "t"})
NO_WORDS = frozenset({"否", "不是", "不", "没有", "无", "不会", "no", "n", "false", "f"})

_NOUL_ANSWER_KEYS = ("answer", "noul", "value", "result")
_NOUL_P_KEYS = ("p", "prob", "probability", "confidence")
_CHOICE_KEYS = ("choice", "answer", "option", "value")
_SCORE_KEYS = ("score", "answer", "level", "value")

ROUND_DIGITS = 6


@dataclass(frozen=True)
class NoulVote:
    """是否题的一票：yes 是答案，p 是模型自报的"是"的概率（没有时为 None）。"""

    yes: bool
    p: float | None = None


Vote = NoulVote | str | float  # 是否题 / 单选题（选项键）/ 打分题（等级）


# ---------------------------------------------------------------- JSON 提取


def _first_braced(text: str) -> str | None:
    """第一个括号配平的 {...} 片段（跳过字符串里的括号）。"""
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _loads_dict(candidate: str | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
    except (ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出里取出 JSON 对象：去掉代码块围栏，整段解析不了就取第一个 {...}。"""
    if not text:
        return None
    fence = _FENCE.search(text)
    bodies = [fence.group(1), text] if fence else [text]
    for body in bodies:
        for candidate in (body.strip(), _first_braced(body)):
            obj = _loads_dict(candidate)
            if obj is not None:
                return obj
    return None


# ---------------------------------------------------------------- 单票解析


def _first_key(raw: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in raw:
            return raw[k]
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        num = float(value)
    elif isinstance(value, str):
        try:
            num = float(value.strip().rstrip("%")) / (100.0 if value.strip().endswith("%") else 1.0)
        except ValueError:
            return None
    else:
        return None
    return num if math.isfinite(num) else None


def _clip01(num: float | None) -> float | None:
    return None if num is None else min(max(num, 0.0), 1.0)


def _as_yes(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().strip(_QUOTES).rstrip(_TRAILING).strip().lower()
        if word in YES_WORDS:
            return True
        if word in NO_WORDS:
            return False
    return None


def parse_noul_vote(raw: Any) -> NoulVote | None:
    """是否题的一票。p 定义为"是"的概率；p 与答案矛盾（答"否"却给 0.9）时，
    视为模型报的是对自己答案的把握，换算成 1 - p。"""
    answer, p_raw = raw, None
    if isinstance(raw, dict):
        answer = _first_key(raw, _NOUL_ANSWER_KEYS)
        p_raw = _first_key(raw, _NOUL_P_KEYS)
    p = _clip01(_as_number(p_raw))
    yes = _as_yes(answer)
    if yes is None:
        num = _clip01(_as_number(answer))
        if num is not None:
            p = num if p is None else p
            yes = num >= 0.5
        elif p is not None:
            yes = p >= 0.5
        else:
            return None
    if p is not None and ((yes and p < 0.5) or (not yes and p > 0.5)):
        p = 1.0 - p
    return NoulVote(yes=yes, p=p)


def match_option(question: ChoiceQuestion, value: Any) -> str | None:
    """把模型给的选项文字对到 criteria 的键上；对不上返回 None。"""
    if not isinstance(value, str):
        return None
    options = {opt.lower(): opt for opt in question.criteria}
    word = value.strip().strip(_QUOTES).strip().lower()
    if word in options:
        return options[word]
    lead = _IDENT.match(word)
    if lead and lead.group(0) in options:
        return options[lead.group(0)]
    return None


def parse_choice_vote(question: ChoiceQuestion, raw: Any) -> str | None:
    value = _first_key(raw, _CHOICE_KEYS) if isinstance(raw, dict) else raw
    return match_option(question, value)


def parse_score_vote(question: ScoreQuestion, raw: Any) -> float | None:
    """打分题的一票：数字（超出范围的截到 [0, 最高等级]）或等级名称。"""
    value = _first_key(raw, _SCORE_KEYS) if isinstance(raw, dict) else raw
    num = _as_number(value)
    if num is None and isinstance(value, str):
        label = value.strip().strip(_QUOTES).strip()
        if label in question.criteria:
            num = float(question.criteria.index(label))
    if num is None:
        return None
    return min(max(num, 0.0), float(question.max_score))


def parse_vote(question: Question, raw: Any) -> Vote | None:
    if isinstance(question, NoulQuestion):
        return parse_noul_vote(raw)
    if isinstance(question, ChoiceQuestion):
        return parse_choice_vote(question, raw)
    return parse_score_vote(question, raw)


def parse_sample(questions: dict[str, Question], text: str) -> dict[str, Vote]:
    """解析一次采样的输出，返回 {问题: 票}；解析不了的问题不出现在结果里。"""
    obj = extract_json_object(text)
    if obj is None:
        return {}
    nested = obj.get("answers")
    if isinstance(nested, dict) and not (set(obj) & set(questions)):
        obj = nested
    lowered = {k.strip().lower(): v for k, v in obj.items() if isinstance(k, str)}
    votes: dict[str, Vote] = {}
    for key, question in questions.items():
        raw = obj[key] if key in obj else lowered.get(key.lower())
        if raw is None:
            continue
        vote = parse_vote(question, raw)
        if vote is not None:
            votes[key] = vote
    return votes


def choice_distribution(
    question: ChoiceQuestion, logprobs: dict[str, float] | None
) -> dict[str, float] | None:
    """LLMResponse.logprobs → 选项概率分布（归一化）。

    契约里写的是"选项 → 概率"，但字段名是 logprobs：有负数时按对数概率处理（取 exp），
    全部在 [0, 1] 时按概率处理；对不上选项的键忽略；没有可用值时返回 None。
    """
    if not logprobs:
        return None
    pairs = [
        (opt, v)
        for k, v in logprobs.items()
        if (opt := match_option(question, k)) is not None
        and isinstance(v, int | float)
        and not isinstance(v, bool)
        and math.isfinite(v)
    ]
    if not pairs:
        return None
    if any(v < 0 for _, v in pairs):
        if any(v > 0 for _, v in pairs):
            return None
        probs = [(opt, math.exp(v)) for opt, v in pairs]
    elif all(v <= 1.0 for _, v in pairs):
        probs = [(opt, float(v)) for opt, v in pairs]
    else:
        return None
    dist: dict[str, float] = {}
    for opt, p in probs:
        dist[opt] = dist.get(opt, 0.0) + p
    total = sum(dist.values())
    if total <= 0:
        return None
    return {opt: p / total for opt, p in dist.items()}


# ---------------------------------------------------------------- 聚合


def aggregate_choice(
    key: str,
    question: ChoiceQuestion,
    votes: list[str],
    distributions: list[dict[str, float]] | None = None,
) -> ChoiceAnswer:
    """单选题：有选项概率就取平均，否则按票数占比；同分按 criteria 顺序取前者。"""
    order = question.options
    if distributions:
        probs = {o: sum(d.get(o, 0.0) for d in distributions) / len(distributions) for o in order}
    elif votes:
        probs = {o: votes.count(o) / len(votes) for o in order}
    else:
        raise DecisionError(f"问题 {key!r} 没有有效的投票")
    probs = {o: round(p, ROUND_DIGITS) for o, p in probs.items() if p > 0}
    if not probs:
        raise DecisionError(f"问题 {key!r} 的选项概率全为 0")
    choice = min(probs, key=lambda o: (-probs[o], order.index(o)))
    return ChoiceAnswer(choice=choice, confidence=probs[choice], probabilities=probs)


def aggregate_noul(key: str, votes: list[NoulVote], *, use_reported_p: bool) -> NoulAnswer:
    """是否题：多票取"是"的占比；只采样一次时用模型自报的 p（没有就按答案取 1 / 0）。"""
    if not votes:
        raise DecisionError(f"问题 {key!r} 没有有效的投票")
    if use_reported_p and len(votes) == 1:
        v = votes[0]
        noul = v.p if v.p is not None else (1.0 if v.yes else 0.0)
    else:
        noul = sum(1 for v in votes if v.yes) / len(votes)
    return NoulAnswer(noul=round(noul, ROUND_DIGITS))


def aggregate_score(key: str, question: ScoreQuestion, votes: list[float]) -> ScoreAnswer:
    """打分题：有效票的平均等级，截到 [0, 最高等级]。"""
    if not votes:
        raise DecisionError(f"问题 {key!r} 没有有效的投票")
    mean = sum(votes) / len(votes)
    return ScoreAnswer(score=round(min(max(mean, 0.0), float(question.max_score)), ROUND_DIGITS))


# ---------------------------------------------------------------- 追问 second_intent（§6.6 规则 5）


def second_intent_votes(samples: Iterable[dict[str, Vote]]) -> list[str]:
    """各次采样（parse_sample 的结果）里追问 second_intent 的票，按模块说明和主要问题配对。

    主要问题（intent）没有有效票的采样不计；其余每次采样出一票：另一个问题的选项键，或"无"。
    """
    votes: list[str] = []
    for sample in samples:
        primary = sample.get(Q_INTENT)
        if not isinstance(primary, str):
            continue
        second = sample.get(Q_SECOND_INTENT)
        multi = sample.get(Q_MULTI_INTENT)
        if (
            isinstance(second, str)
            and second not in (NO_SECOND_INTENT, primary)
            and isinstance(multi, NoulVote)
            and multi.yes
        ):
            votes.append(second)
        else:
            votes.append(NO_SECOND_INTENT)
    return votes


def aggregate_second_intent(key: str, question: ChoiceQuestion, votes: list[str]) -> ChoiceAnswer:
    """追问的答案：各选项的票数占比；一票都没有时答"无"（不分条）。"""
    if not votes:
        return ChoiceAnswer(
            choice=NO_SECOND_INTENT, confidence=1.0, probabilities={NO_SECOND_INTENT: 1.0}
        )
    return aggregate_choice(key, question, votes)


def _shares(answer: ChoiceAnswer | None) -> dict[str, float]:
    if answer is None:
        return {}
    return dict(answer.probabilities) or {answer.choice: answer.confidence}


def mention_probabilities(
    intent: ChoiceAnswer | None, second: ChoiceAnswer | None
) -> dict[str, float]:
    """每个选项"被问到"的概率：主要问题的概率 + 追问的占比（去掉"无"），截到 1。

    追问按票和主要问题配对（second_intent_votes），同一票里两者不同，直接相加不会重复计数。
    路由规则 5 在 multi_intent 达到阈值时用它分条（routing.policy）；没有追问时就是 intent 的概率。
    """
    merged = _shares(intent)
    for option, p in _shares(second).items():
        if option == NO_SECOND_INTENT:
            continue
        merged[option] = min(1.0, round(merged.get(option, 0.0) + p, ROUND_DIGITS))
    return {option: p for option, p in merged.items() if p > 0}
