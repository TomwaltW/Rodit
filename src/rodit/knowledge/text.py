"""知识中心共用的文字处理：个人信息最小化、关键词匹配用的归一化、字面相关度。"""

from __future__ import annotations

import re
import unicodedata

from rodit.core.pii import mask_pii

_NON_WORD = re.compile(r"[\W_]+")


def mask_text(text: str) -> str:
    """买家文字进向量服务、提示词之前的个人信息最小化（§5.3）。"""
    return mask_pii(text or "")


def minimized(text: str) -> str:
    """进提示词的证据文字（Evidence.content）：商家录入的政策、详情页 OCR 里偶尔会有电话、地址，
    这里兜底遮掉；没有个人信息时原样返回（mask_pii 只替换命中的片段，§5.3）。"""
    return mask_pii(text or "")


def for_matching(text: str) -> str:
    """关键词匹配用：NFKC + 小写（"V领" 与 "v领"、全角数字与半角数字一致）。"""
    return unicodedata.normalize("NFKC", text or "").lower()


def _bigrams(text: str) -> set[str]:
    s = _NON_WORD.sub("", for_matching(text))
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def relevance(question: str, text: str) -> float:
    """问题与一段文字的字面相关度 ∈ [0, 1]：问题的字符二元组有多少出现在文字里。

    只用于长度预算里决定详情页哪几句先删，不作为 Evidence.score（score 只给 FAQ）。
    """
    q = _bigrams(question)
    if not q:
        return 0.0
    return len(q & _bigrams(text)) / len(q)
