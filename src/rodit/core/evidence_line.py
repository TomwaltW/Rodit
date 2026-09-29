"""提示词【依据】里一条依据的写法（方案 §5.4 提示词骨架、§5.3 最小化个人信息、§14 控制证据长度）。

一行是 "[{id}] {label}：{content}"：content 先遮个人信息、压成一行，实时数据补"（数据截至 …）"。
检索的长度预算（knowledge.conflict.trim_to_budget）和提示词（generation.prompt）按同一个字符串计长，
检索按相关度留下的依据，提示词一定放得下，不会再按位置丢一轮。
"""

from __future__ import annotations

import re

from rodit.core.clock import fmt_time
from rodit.core.models import Evidence
from rodit.core.pii import mask_pii

_WHITESPACE = re.compile(r"\s+")


def evidence_line_parts(e: Evidence) -> tuple[str, str]:
    """(行首 "[id] 标签：", 内容)。内容已遮个人信息、压成一行、带获取时间。"""
    head = f"[{e.id}] {e.label}："
    content = _WHITESPACE.sub(" ", mask_pii(e.content) if e.content else "").strip()
    if e.fetched_at is not None and "截至" not in content:
        content += f"（数据截至 {fmt_time(e.fetched_at)}）"
    return head, content


def evidence_line(e: Evidence) -> str:
    head, content = evidence_line_parts(e)
    return head + content


def evidence_line_chars(e: Evidence) -> int:
    """这一行的字数（不含换行）。几行合在一起的字数 = 各行字数之和 + 行数 - 1。"""
    return len(evidence_line(e))
