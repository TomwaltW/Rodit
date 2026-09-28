"""判断层的问题定义（方案 §6.5 路由、§5.5 质检）。

问题定义是跨模块契约：路由（routing）和质检（qa）都从这里取，decision_log 记录
questions_ver。改动任何问题的文字都会改变版本号，改之前先在评测集上跑回归（§13.5）。
"""

from __future__ import annotations

import hashlib
import json

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


def _digest(questions: dict[str, Question]) -> str:
    payload = {k: q.model_dump(mode="json") for k, q in sorted(questions.items())}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def questions_version(questions: dict[str, Question]) -> str:
    """问题定义的版本号（写入 decision_log.questions_ver）。"""
    return _digest(questions)


ROUTER_QUESTIONS_VER = questions_version(ROUTER_QUESTIONS)
QA_QUESTIONS_VER = questions_version(QA_QUESTIONS)
