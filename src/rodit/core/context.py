"""请求级上下文：当前处理的店铺（用于大模型成本按店铺归属，§12.4）。

引擎处理每条消息时用 shop_scope(shop_id) 包住；线程池里的任务要用
contextvars.copy_context().run(...) 提交，才能继承这个值。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current_shop: ContextVar[str | None] = ContextVar("rodit_current_shop", default=None)


def current_shop_id() -> str | None:
    return _current_shop.get()


@contextmanager
def shop_scope(shop_id: str) -> Iterator[None]:
    token = _current_shop.set(shop_id)
    try:
        yield
    finally:
        _current_shop.reset(token)
