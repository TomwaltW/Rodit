"""知识中心共用的文字处理：个人信息最小化、关键词匹配用的归一化、字面相关度。"""

from __future__ import annotations

import re
import unicodedata

from rodit.core.pii import ADDRESS_MASK, contains_pii, mask_pii

_NON_WORD = re.compile(r"[\W_]+")

# 本地补丁（core.pii 的地址遮挡漏洞，已在回执 contract_issues 里报告，集成方修好 core 后可删）：
# 1. "住幸福路 18 号 3 栋 201"：地址中间有空格时，core 把空格当成引导词边界，只遮掉后半段，
#    留下"幸福路 18"。先去掉汉字与数字之间的空白再遮挡（数字与数字之间的空白保留，
#    "165 110斤"不受影响）。
# 2. "阳光小区 5 栋 302"：栋号后面直接跟房号时，房号没有被遮掉。遮挡后再吃掉紧跟在
#    [地址] 后面的 3–4 位房号。
_CJK_DIGIT_GAP = re.compile(r"(?<=[一-鿿])\s+(?=\d)|(?<=\d)\s+(?=[一-鿿])")
_ROOM_AFTER_ADDRESS = re.compile(re.escape(ADDRESS_MASK) + r"\s*\d{3,4}(?!\d)")


def _collapse(text: str) -> str:
    return _CJK_DIGIT_GAP.sub("", text or "")


def mask_text(text: str) -> str:
    """core.pii.mask_pii 加上本地补丁：买家文字进向量服务、提示词之前都用它（§5.3）。"""
    return _ROOM_AFTER_ADDRESS.sub(ADDRESS_MASK, mask_pii(_collapse(text)))


def minimized(text: str) -> str:
    """进提示词的证据文字（Evidence.content）：有个人信息才遮挡（§5.3）。

    没有个人信息时原样返回，避免 mask_pii 的 NFKC 把中文全角标点改成半角。
    商家录入的政策、详情页 OCR 里偶尔会有电话、地址，这里兜底遮掉。
    """
    return mask_text(text) if contains_pii(_collapse(text)) else text


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
