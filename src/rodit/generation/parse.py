"""解析大模型的生成输出（方案 §5.4 输出 JSON：{"answer", "evidence_ids", "need_clarify"}）。

模型常见的"不规范"：包在 ```json 代码块里、前面带一句"好的，以下是回答"、后面带解释。
这里依次尝试整段、代码块、从每个 "{" 开始解码，取第一个带 answer 的 JSON 对象；
仍然拿不到或字段不合法就抛 LLMError（调用方退回 FAQ / 模板，§12.2）。
错误信息里不放模型原文（可能复述了买家的个人信息）。
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rodit.core.errors import LLMError

_FENCE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\n?(.*?)```", re.S)


class GenerationOutput(BaseModel):
    """生成结果。多余字段忽略；evidence_ids 原样保留（未知 ID 由质检 unknown_evidence 拦截）。"""

    model_config = ConfigDict(extra="ignore")

    answer: str
    evidence_ids: list[str] = Field(default_factory=list)
    need_clarify: bool = False

    @field_validator("answer")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("answer 为空")
        return v

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v


def _candidates(text: str) -> list[str]:
    return [text.strip(), *(m.group(1).strip() for m in _FENCE.finditer(text))]


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出里取出 JSON 对象；优先带 "answer" 键的。找不到抛 LLMError。"""
    if not text or not text.strip():
        raise LLMError("大模型输出为空")
    first: dict[str, Any] | None = None
    for candidate in _candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if "answer" in obj:
                return obj
            first = first or obj
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if "answer" in obj:
                return obj
            first = first or obj
    if first is not None:
        return first
    raise LLMError("大模型输出里找不到 JSON 对象")


def parse_generation_output(text: str) -> GenerationOutput:
    """解析并校验生成输出，失败抛 LLMError。"""
    obj = extract_json_object(text)
    try:
        return GenerationOutput.model_validate(obj)
    except ValidationError as e:
        fields = sorted({".".join(str(p) for p in err["loc"]) or "?" for err in e.errors()})
        raise LLMError(f"大模型输出格式不对：{', '.join(fields)}") from e
