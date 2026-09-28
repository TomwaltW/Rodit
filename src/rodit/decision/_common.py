"""判断层实现共用的小工具：从 state 里取当前买家消息、遍历 state 里的字符串、文本归一。

state 只由 core.states 构造（§6.5），这里只读，不拼 state。
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from typing import Any

from rodit.core.enums import Role


def normalize(text: str) -> str:
    """NFKC 归一 + 小写（全角转半角，便于关键词和 n-gram 匹配）。"""
    return unicodedata.normalize("NFKC", text or "").lower()


def current_buyer_text(state: dict[str, Any]) -> str:
    """路由 state 里"当前"买家消息的文字（已脱敏）。

    core.states.build_router_state 把当前消息放在 messages 最后；这里取最后一条买家消息，
    没有买家消息时返回空串。
    """
    messages = state.get("messages") or []
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == Role.BUYER.value:
            text = m.get("text")
            return text if isinstance(text, str) else ""
    return ""


def iter_strings(obj: Any) -> Iterator[str]:
    """递归遍历 dict / list / tuple 里的全部字符串（含 dict 的键）。"""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from iter_strings(v)
    elif isinstance(obj, list | tuple | set | frozenset):
        for v in obj:
            yield from iter_strings(v)
