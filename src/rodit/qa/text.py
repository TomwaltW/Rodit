"""回答文本清洗：去掉 Markdown、超长截断（方案 §5.5 第一道"长度与格式"）。

两者都是自动修正（NON_BLOCKING_HARD_RULES），只改写文本、不拦截。平台聊天窗口不渲染 Markdown，
链接本身属于站外引流，去掉链接语法之前由硬规则先在原文上检查。
"""

from __future__ import annotations

import re

# 图片 / 链接：![alt](url) → alt，[文字](url) → 文字
_LINK = re.compile(r"!?\[([^\]\n]*)\]\([^)\n]*\)")
_FENCE = re.compile(r"```[A-Za-z0-9_-]*")
_BOLD = re.compile(r"(\*\*|__)(.+?)\1", re.S)
_STRIKE = re.compile(r"~~(.+?)~~", re.S)
_ITALIC = re.compile(r"(?<![*A-Za-z0-9_])\*(?![\s*])([^*\n]+?)(?<!\s)\*(?![*A-Za-z0-9_])")
_INLINE_CODE = re.compile(r"`+([^`\n]*)`+")
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.M)
_QUOTE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.M)
_BULLET = re.compile(r"^[ \t]*(?:[-*+•]|\d{1,2}[.)])[ \t]+", re.M)
_RULE_LINE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.M)
_HTML = re.compile(r"</?(?:b|strong|i|em|u|br|p|span|div|a)\b[^>]*>", re.I)

_MARKDOWN_STEPS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_LINK, r"\1"),
    (_FENCE, ""),
    (_BOLD, r"\2"),
    (_STRIKE, r"\1"),
    (_ITALIC, r"\1"),
    (_INLINE_CODE, r"\1"),
    (_RULE_LINE, ""),
    (_HEADING, ""),
    (_QUOTE, ""),
    (_BULLET, ""),
    (_HTML, ""),
)

_BLANK_LINES = re.compile(r"\n\s*\n+")


def strip_markdown(text: str) -> tuple[str, bool]:
    """去掉 Markdown（**、__、#、`、[x](url)→x、列表符号……）。返回 (新文本, 是否去掉了格式)。

    只整理空白（首尾空白、多余空行）不算"去掉了格式"。
    """
    out = text or ""
    changed = False
    for pattern, repl in _MARKDOWN_STEPS:
        out, n = pattern.subn(repl, out)
        changed = changed or n > 0
    lines = [line.strip() for line in out.replace("\r\n", "\n").split("\n")]
    out = _BLANK_LINES.sub("\n", "\n".join(lines)).strip()
    return out, changed


_SENTENCE_END = frozenset("。！？!?；;…~\n")
_CLAUSE_END = frozenset("，,、：:")
_ELLIPSIS = "…"


def truncate(text: str, limit: int) -> str:
    """超过 limit 字时在句子边界截断（§5.5 超长截断），结果不超过 limit 字。

    依次尝试：整句（保留部分至少占 limit 的三分之一）→ 分句（逗号等，末尾加"…"）→ 硬截断加"…"。
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    min_keep = max(1, limit // 3)
    head = text[:limit]
    cut = max((i for i, ch in enumerate(head) if ch in _SENTENCE_END), default=-1)
    if cut + 1 >= min_keep:
        kept = head[: cut + 1].rstrip()
        if kept:
            return kept
    clause = max((i for i, ch in enumerate(head[: limit - 1]) if ch in _CLAUSE_END), default=-1)
    if clause >= min_keep:
        return head[:clause] + _ELLIPSIS
    return text[: limit - 1] + _ELLIPSIS
