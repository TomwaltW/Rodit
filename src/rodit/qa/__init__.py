"""答案质检（方案 §5.5）：硬规则 + 判断层质检 + 放行规则。

负责：T4。只依赖 rodit.core。
"""

from rodit.qa.gate import (
    EMPTY_ANSWER,
    QA_OUT_OF_TIME,
    QA_UNAVAILABLE,
    DefaultQualityGate,
    model_reasons,
)
from rodit.qa.hard_rules import EvidenceCorpus, blocking_hits, check_hard_rules
from rodit.qa.numbers import NumberSet, NumberToken, extract_numbers, unsupported_numbers
from rodit.qa.text import strip_markdown, truncate

__all__ = [
    "EMPTY_ANSWER",
    "QA_OUT_OF_TIME",
    "QA_UNAVAILABLE",
    "DefaultQualityGate",
    "EvidenceCorpus",
    "NumberSet",
    "NumberToken",
    "blocking_hits",
    "check_hard_rules",
    "extract_numbers",
    "model_reasons",
    "strip_markdown",
    "truncate",
    "unsupported_numbers",
]
