"""知识中心（方案 §5.2）：检索（结构化优先 + 语义检索补充）、平台规则、冷启动问卷与缺失检查。

负责：T2。只依赖 rodit.core。
"""

from rodit.knowledge.category import (
    CATEGORY_TIPS,
    CategoryTip,
    category_group,
    category_tip_evidence,
    category_tips,
)
from rodit.knowledge.coverage import (
    APPAREL_CHECKLIST,
    GENERIC_CHECKLIST,
    ChecklistEntry,
    checklist_for,
    describe_gap,
    find_gaps,
)
from rodit.knowledge.platform_rules import (
    PLATFORM_RULES,
    PlatformRuleText,
    find_platform_rule,
    pending_platform_rules,
    platform_rule_evidence,
    platform_rule_text,
)
from rodit.knowledge.questionnaire import (
    QUESTIONNAIRE,
    QUESTIONNAIRE_BY_ID,
    PolicyTarget,
    QuestionnaireItem,
    apply_questionnaire,
    parse_banned_words,
    parse_tone,
    question_for_policy,
    split_policy_answer,
)
from rodit.knowledge.retrieval import (
    CLARIFY_ITEM_QUESTION,
    DEFAULT_FAQ_MIN_SCORE,
    KnowledgeRetriever,
    route_intents,
)
from rodit.knowledge.search import (
    EXACT_MATCH_SCORE,
    SEMANTIC_SCORE_CAP,
    FaqSearcher,
    normalize_question,
)

__all__ = [
    "APPAREL_CHECKLIST",
    "CATEGORY_TIPS",
    "CLARIFY_ITEM_QUESTION",
    "DEFAULT_FAQ_MIN_SCORE",
    "EXACT_MATCH_SCORE",
    "GENERIC_CHECKLIST",
    "PLATFORM_RULES",
    "QUESTIONNAIRE",
    "QUESTIONNAIRE_BY_ID",
    "SEMANTIC_SCORE_CAP",
    "CategoryTip",
    "ChecklistEntry",
    "FaqSearcher",
    "KnowledgeRetriever",
    "PlatformRuleText",
    "PolicyTarget",
    "QuestionnaireItem",
    "apply_questionnaire",
    "category_group",
    "category_tip_evidence",
    "category_tips",
    "checklist_for",
    "describe_gap",
    "find_gaps",
    "find_platform_rule",
    "normalize_question",
    "parse_banned_words",
    "parse_tone",
    "pending_platform_rules",
    "platform_rule_evidence",
    "platform_rule_text",
    "question_for_policy",
    "route_intents",
    "split_policy_answer",
]
