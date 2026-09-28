"""判断层实现（方案 §6）：RuleProvider / InCloudLLMProvider / ClassifierProvider / JevProvider。

负责：T1。只依赖 rodit.core。
"""

from rodit.decision.classifier import ClassifierProvider, char_ngrams
from rodit.decision.factory import build_decision_provider, build_llm_provider
from rodit.decision.jev import JevConditions, JevProvider
from rodit.decision.llm import SYSTEM_PROMPT, InCloudLLMProvider, build_judge_messages
from rodit.decision.rules import ROUTER_QUESTION_TYPES, RuleAnalysis, RuleProvider, analyze
from rodit.decision.voting import extract_json_object, parse_sample

__all__ = [
    "ROUTER_QUESTION_TYPES",
    "SYSTEM_PROMPT",
    "ClassifierProvider",
    "InCloudLLMProvider",
    "JevConditions",
    "JevProvider",
    "RuleAnalysis",
    "RuleProvider",
    "analyze",
    "build_decision_provider",
    "build_judge_messages",
    "build_llm_provider",
    "char_ngrams",
    "extract_json_object",
    "parse_sample",
]
