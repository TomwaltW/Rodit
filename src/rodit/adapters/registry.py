"""适配器注册表：(平台, 接入模式) → 适配器（方案 §8.1、§9.1）。

app 层按店铺的平台和该店铺开通的接入模式取适配器；一个平台单元里每种接入模式只注册一个实例。
"""

from __future__ import annotations

import threading

from rodit.adapters.base import ACCESS_MODES, AccessMode
from rodit.core.enums import Platform
from rodit.core.ports import PlatformAdapter

_PREFERENCE: tuple[AccessMode, ...] = ("A", "B", "C")  # 能力从强到弱（§8.1）


class AdapterRegistry:
    """(Platform, AccessMode) → PlatformAdapter。线程安全。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._adapters: dict[tuple[Platform, AccessMode], PlatformAdapter] = {}

    def register(self, adapter: PlatformAdapter, *, access: AccessMode | None = None) -> None:
        """注册适配器。access 不传时读 adapter.access（BaseMockAdapter 都有）。
        同一 (平台, 接入模式) 重复注册抛 ValueError。"""
        mode = access if access is not None else getattr(adapter, "access", None)
        if mode not in ACCESS_MODES:
            raise ValueError(f"适配器 {adapter!r} 没有合法的接入模式：{mode!r}")
        adapter_access = getattr(adapter, "access", mode)
        if adapter_access != mode:
            raise ValueError(f"适配器的接入模式是 {adapter_access}，不能注册成 {mode}")
        key = (Platform(adapter.platform), mode)
        with self._lock:
            if key in self._adapters:
                raise ValueError(
                    f"{key[0].value} 的接入模式 {mode} 已注册：{self._adapters[key]!r}"
                )
            self._adapters[key] = adapter

    def get(self, platform: Platform, access: AccessMode) -> PlatformAdapter:
        """没有注册时抛 KeyError。"""
        adapter = self.find(platform, access)
        if adapter is None:
            raise KeyError(f"{Platform(platform).value} 没有注册接入模式 {access} 的适配器")
        return adapter

    def find(self, platform: Platform, access: AccessMode) -> PlatformAdapter | None:
        with self._lock:
            return self._adapters.get((Platform(platform), access))

    def preferred(self, platform: Platform) -> PlatformAdapter:
        """该平台能力最强的已注册适配器（A > B > C）；一个都没有时抛 KeyError。"""
        for mode in _PREFERENCE:
            adapter = self.find(platform, mode)
            if adapter is not None:
                return adapter
        raise KeyError(f"{Platform(platform).value} 没有注册任何适配器")

    def for_platform(self, platform: Platform) -> dict[AccessMode, PlatformAdapter]:
        """该平台已注册的适配器，按 A、B、C 排序。"""
        with self._lock:
            return {
                mode: self._adapters[(Platform(platform), mode)]
                for mode in _PREFERENCE
                if (Platform(platform), mode) in self._adapters
            }

    def keys(self) -> list[tuple[Platform, AccessMode]]:
        with self._lock:
            return sorted(self._adapters, key=lambda k: (k[0].value, k[1]))

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._adapters

    def __len__(self) -> int:
        with self._lock:
            return len(self._adapters)
