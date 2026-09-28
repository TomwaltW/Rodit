"""适配器注册表：(平台, 接入模式) → 适配器。"""

from __future__ import annotations

import pytest

from rodit.adapters import (
    AdapterRegistry,
    DouyinMockAdapter,
    NullToolGateway,
    PddMockAdapter,
    StandaloneAssistAdapter,
    TaobaoMockAdapter,
)
from rodit.core.enums import Platform
from rodit.testing.fakes import InMemoryToolGateway


@pytest.fixture
def adapters(world, clock, hasher):
    return {
        "tb_a": TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="A"),
        "tb_b": TaobaoMockAdapter(InMemoryToolGateway(world), clock, hasher, access="B"),
        "tb_c": StandaloneAssistAdapter(Platform.TAOBAO, clock, hasher),
        "dy_b": DouyinMockAdapter(
            InMemoryToolGateway(world, Platform.DOUYIN), clock, hasher, access="B"
        ),
        "pdd_c": PddMockAdapter(NullToolGateway(Platform.PDD), clock, hasher, access="C"),
    }


def test_register_and_lookup(adapters):
    reg = AdapterRegistry()
    for adapter in adapters.values():
        reg.register(adapter)
    assert len(reg) == 5
    assert reg.get(Platform.TAOBAO, "A") is adapters["tb_a"]
    assert reg.get(Platform.TAOBAO, "C") is adapters["tb_c"]
    assert reg.get("douyin", "B") is adapters["dy_b"]  # type: ignore[arg-type]
    assert reg.find(Platform.DOUYIN, "A") is None
    with pytest.raises(KeyError):
        reg.get(Platform.PDD, "A")
    assert (Platform.PDD, "C") in reg
    assert reg.keys() == [
        (Platform.DOUYIN, "B"),
        (Platform.PDD, "C"),
        (Platform.TAOBAO, "A"),
        (Platform.TAOBAO, "B"),
        (Platform.TAOBAO, "C"),
    ]
    assert list(reg.for_platform(Platform.TAOBAO)) == ["A", "B", "C"]
    assert reg.for_platform(Platform.PDD) == {"C": adapters["pdd_c"]}


def test_preferred_picks_strongest_access(adapters):
    reg = AdapterRegistry()
    reg.register(adapters["tb_c"])
    assert reg.preferred(Platform.TAOBAO) is adapters["tb_c"]
    reg.register(adapters["tb_b"])
    assert reg.preferred(Platform.TAOBAO) is adapters["tb_b"]
    reg.register(adapters["tb_a"])
    assert reg.preferred(Platform.TAOBAO) is adapters["tb_a"]
    with pytest.raises(KeyError):
        reg.preferred(Platform.DOUYIN)


def test_duplicate_and_mismatched_registration(adapters, world, clock, hasher):
    reg = AdapterRegistry()
    reg.register(adapters["pdd_c"])
    with pytest.raises(ValueError, match="已注册"):
        reg.register(StandaloneAssistAdapter(Platform.PDD, clock, hasher))
    with pytest.raises(ValueError, match="不能注册成"):
        reg.register(adapters["tb_b"], access="A")

    class _NoAccess:
        platform = Platform.TAOBAO

    with pytest.raises(ValueError, match="接入模式"):
        reg.register(_NoAccess())  # type: ignore[arg-type]
