"""判断层请求 / 响应格式（方案 §6.5），与 Jev 的 /api/v1/systemone 保持一致。

题型：
- noul：是否题，答案 noul ∈ [0, 1] 是"是"的概率；
- choice：多选一，criteria 是 {选项: 定义或 None}，答案带 choice / confidence / probabilities；
- score：打分，criteria 是等级列表（下标 0..n-1），答案 score ∈ [0, n-1] 是期望等级。

编排、阈值、审计日志只认这里的格式；换判断层实现（incloud_llm / classifier / rules / jev）
只换 DecisionProvider，不改其他模块。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoulQuestion(_Model):
    type: Literal["noul"] = "noul"
    instructions: str


class ChoiceQuestion(_Model):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str | None]

    @property
    def options(self) -> list[str]:
        return list(self.criteria)


class ScoreQuestion(_Model):
    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]

    @property
    def max_score(self) -> int:
        return len(self.criteria) - 1


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionRequest(_Model):
    model: str
    state: dict[str, Any]
    questions: dict[str, Question]


class NoulAnswer(_Model):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(_Model):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ChoiceAnswer:
        for k, v in self.probabilities.items():
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"probability of {k!r} out of range: {v}")
        return self

    def top(self, n: int = 2) -> list[tuple[str, float]]:
        """按概率降序取前 n 个选项；probabilities 为空时只返回 choice。"""
        if not self.probabilities:
            return [(self.choice, self.confidence)]
        return sorted(self.probabilities.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


class ScoreAnswer(_Model):
    type: Literal["score"] = "score"
    score: float = Field(ge=0.0)


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class DecisionResponse(_Model):
    model: str
    answers: dict[str, Answer]

    def noul(self, key: str) -> float:
        a = self.answers[key]
        if not isinstance(a, NoulAnswer):
            raise TypeError(f"answer {key!r} is {a.type}, not noul")
        return a.noul

    def choice(self, key: str) -> ChoiceAnswer:
        a = self.answers[key]
        if not isinstance(a, ChoiceAnswer):
            raise TypeError(f"answer {key!r} is {a.type}, not choice")
        return a

    def score(self, key: str) -> float:
        a = self.answers[key]
        if not isinstance(a, ScoreAnswer):
            raise TypeError(f"answer {key!r} is {a.type}, not score")
        return a.score


def validate_response(request: DecisionRequest, response: DecisionResponse) -> None:
    """检查响应是否完整回答了请求里的每个问题、题型一致、取值在范围内。

    不满足时抛 DecisionError。所有 DecisionProvider 返回前都应调用。
    """
    from rodit.core.errors import DecisionError

    missing = set(request.questions) - set(response.answers)
    if missing:
        raise DecisionError(f"判断层缺少答案：{sorted(missing)}")
    for key, q in request.questions.items():
        a = response.answers[key]
        if a.type != q.type:
            raise DecisionError(f"问题 {key!r} 题型不一致：{q.type} vs {a.type}")
        if isinstance(q, ChoiceQuestion):
            assert isinstance(a, ChoiceAnswer)
            if a.choice not in q.criteria:
                raise DecisionError(f"问题 {key!r} 的选项 {a.choice!r} 不在 criteria 里")
            unknown = set(a.probabilities) - set(q.criteria)
            if unknown:
                raise DecisionError(f"问题 {key!r} 的概率里有未知选项：{sorted(unknown)}")
        elif isinstance(q, ScoreQuestion):
            assert isinstance(a, ScoreAnswer)
            if a.score > q.max_score:
                raise DecisionError(f"问题 {key!r} 的分数 {a.score} 超过上限 {q.max_score}")


@runtime_checkable
class DecisionProvider(Protocol):
    """判断层实现（§6.5）：incloud_llm / classifier / rules / jev。

    - 只做判断，不写回答，也不触发任何操作（§6.1）。
    - 必须回答请求里的全部问题，否则抛 DecisionError（调用方据此降级）。
    - state 里的文本在进入本接口前已经做过个人信息最小化（core.pii.mask_pii）。
    """

    name: str

    def decide(self, request: DecisionRequest) -> DecisionResponse: ...
