"""短时缓存的进程内实现（实现 core.ports.Cache）。多实例部署时换 Redis（见 CLAUDE.md 选型记录）。"""

from __future__ import annotations

import threading
from datetime import datetime

from rodit.core.clock import Clock, SystemClock


class InMemoryCache:
    """按店铺分命名空间；只记写入时间，过期由调用方判断。max_entries 超出时淘汰最早写入的。"""

    def __init__(self, clock: Clock | None = None, *, max_entries: int = 10_000) -> None:
        self._clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._data: dict[tuple[str, str], tuple[object, datetime]] = {}
        self._max = max_entries

    def get(self, shop_id: str, key: str) -> tuple[object, datetime] | None:
        with self._lock:
            return self._data.get((shop_id, key))

    def set(self, shop_id: str, key: str, value: object) -> None:
        with self._lock:
            self._data.pop((shop_id, key), None)  # 重新插入，保持按写入时间的顺序
            self._data[(shop_id, key)] = (value, self._clock.now())
            while len(self._data) > self._max:
                self._data.pop(next(iter(self._data)))

    def clear_shop(self, shop_id: str) -> None:
        with self._lock:
            for k in [k for k in self._data if k[0] == shop_id]:
                del self._data[k]
