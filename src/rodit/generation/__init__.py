"""答案生成（方案 §5.4）：提示词、FAQ / 模板直答、大模型生成。

负责：T4。只依赖 rodit.core。
"""

from rodit.generation import templates
from rodit.generation.generator import DefaultAnswerGenerator
from rodit.generation.parse import GenerationOutput, extract_json_object, parse_generation_output
from rodit.generation.prompt import (
    SPLIT_MULTI_RULE,
    PromptVariant,
    answer_char_limit,
    build_generation_messages,
    build_generation_request,
    split_focus_intent,
)

__all__ = [
    "SPLIT_MULTI_RULE",
    "DefaultAnswerGenerator",
    "GenerationOutput",
    "PromptVariant",
    "answer_char_limit",
    "build_generation_messages",
    "build_generation_request",
    "extract_json_object",
    "parse_generation_output",
    "split_focus_intent",
    "templates",
]
