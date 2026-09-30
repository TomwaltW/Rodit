"""判断层的问题定义（方案 §6.5 路由、§5.5 质检）。

问题定义是跨模块契约：路由（routing）和质检（qa）都从这里取，decision_log 记录
questions_ver。改动任何问题的文字都会改变版本号，改之前先在评测集上跑回归（§13.5）。

追问（followup_questions）：路由的 multi_intent 后面跟一道 second_intent（intent 的选项 + 无），
问"主要问题以外的另一个问题是哪一类"。一票只能选一个 intent，各票常常都选同一个"主要"类型，
规则 5 认不出第二个问题；所以投票制的大模型判断层在同一个请求里一并回答追问（§6.4 一票一个请求），
路由按票数占比合并后分条（§6.6 规则 5）。规则、分类器、Jev 给的 intent 概率本来就分摊到各个类型上，
不问追问——ROUTER_QUESTIONS 是每个路由判断层都要回答的四个问题（§6.5 请求示例），
ROUTER_VOTE_QUESTIONS 是投票制判断层的完整请求。追问的文字同样算进版本号（questions_version）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from rodit.core.decision import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion
from rodit.core.enums import INTENT_EXAMPLES, INTENT_LABELS, Intent


def _intent_definition(intent: Intent) -> str:
    examples = "；".join(INTENT_EXAMPLES[intent])
    label = INTENT_LABELS[intent]
    return f"{label}。例：{examples}" if examples else label


# ---------------------------------------------------------------- 路由（§6.5）

Q_INTENT = "intent"
Q_NEED_HUMAN = "need_human"
Q_EMOTION = "emotion"
Q_MULTI_INTENT = "multi_intent"

EMOTION_LEVELS = ["平静", "略有不满", "不满", "愤怒", "极度愤怒"]

ROUTER_QUESTIONS: dict[str, Question] = {
    Q_INTENT: ChoiceQuestion(
        instructions="买家这条消息主要在问什么？各选项定义见问题分类表。",
        criteria={i.value: _intent_definition(i) for i in Intent},
    ),
    Q_NEED_HUMAN: NoulQuestion(instructions="买家是否明确要求人工客服，或表示要投诉？"),
    Q_EMOTION: ScoreQuestion(instructions="买家当前情绪的激烈程度。", criteria=EMOTION_LEVELS),
    Q_MULTI_INTENT: NoulQuestion(instructions="这条消息是否包含两个及以上不同的问题？"),
}

# ---------------------------------------------------------------- 路由的追问（§6.4、§6.6 规则 5）

Q_SECOND_INTENT = "second_intent"
NO_SECOND_INTENT = "无"  # 追问的额外选项：只有一个问题

SECOND_INTENT_INSTRUCTIONS = (
    "multi_intent 答“是”时：除了 intent 选的那个主要问题，消息里另一个问题属于哪一类？"
    "选项含义同 intent，不能和 intent 选同一个；另外的问题不止一个时选最重要的。"
    "multi_intent 答“否”时选 无。"
)
SECOND_INTENT_NONE_DEFINITION = "只有一个问题（multi_intent 答“否”）"


def second_intent_question(intent: ChoiceQuestion) -> ChoiceQuestion:
    """追问 second_intent：选项是 intent 的全部选项（定义同 intent，不再重复）加上"无"。"""
    criteria: dict[str, str | None] = {option: None for option in intent.criteria}
    criteria[NO_SECOND_INTENT] = SECOND_INTENT_NONE_DEFINITION
    return ChoiceQuestion(instructions=SECOND_INTENT_INSTRUCTIONS, criteria=criteria)


def followup_questions(questions: Mapping[str, Question]) -> dict[str, Question]:
    """questions 需要的追问：同时有 intent（单选）和 multi_intent（是否）时追问 second_intent。

    intent 的选项里已经有"无"时分不清"没有第二个问题"，不追问。"""
    intent = questions.get(Q_INTENT)
    multi = questions.get(Q_MULTI_INTENT)
    if (
        isinstance(intent, ChoiceQuestion)
        and isinstance(multi, NoulQuestion)
        and NO_SECOND_INTENT not in intent.criteria
    ):
        return {Q_SECOND_INTENT: second_intent_question(intent)}
    return {}


def with_followups(questions: Mapping[str, Question]) -> dict[str, Question]:
    """questions 加上它的追问（排在最后；调用方已经给了同名问题的，照用调用方的）。"""
    extra = {k: q for k, q in followup_questions(questions).items() if k not in questions}
    return {**questions, **extra}


ROUTER_VOTE_QUESTIONS: dict[str, Question] = with_followups(ROUTER_QUESTIONS)
"""投票制大模型判断层的路由请求：四个必答问题 + 追问 second_intent（一票一个请求）。"""

# ---------------------------------------------------------------- 质检（§5.5 第二道）

Q_GROUNDED = "grounded"
Q_ANSWERS_QUESTION = "answers_question"
Q_PROMISE_RISK = "promise_risk"
Q_PLATFORM_VIOLATION = "platform_violation"
Q_TONE = "tone"

TONE_LEVELS = ["生硬", "偏生硬", "一般", "礼貌", "亲切得体"]

QA_QUESTIONS: dict[str, Question] = {
    Q_GROUNDED: NoulQuestion(instructions="回答里的每个事实，是否都能在依据中找到？"),
    Q_ANSWERS_QUESTION: NoulQuestion(instructions="回答是否回应了买家的问题？"),
    Q_PROMISE_RISK: NoulQuestion(instructions="是否包含依据之外的承诺（价格、赠品、时效、赔偿）？"),
    Q_PLATFORM_VIOLATION: NoulQuestion(instructions="是否包含或引导站外联系、站外交易？"),
    Q_TONE: ScoreQuestion(instructions="礼貌程度，0 生硬 → 4 亲切得体。", criteria=TONE_LEVELS),
}

ROUTER_MODEL = "router-v1"
QA_MODEL = "qa-v1"


def _digest(questions: Mapping[str, Question]) -> str:
    payload = {k: q.model_dump(mode="json") for k, q in sorted(questions.items())}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def questions_version(questions: Mapping[str, Question]) -> str:
    """问题定义的版本号（写入 decision_log.questions_ver）。

    连同追问一起算（with_followups）：路由问题带不带追问是同一个版本，追问的文字改了版本号也变。"""
    return _digest(with_followups(questions))


ROUTER_QUESTIONS_VER = questions_version(ROUTER_QUESTIONS)
QA_QUESTIONS_VER = questions_version(QA_QUESTIONS)
