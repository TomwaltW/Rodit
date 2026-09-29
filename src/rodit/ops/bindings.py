"""平台绑定：平台只读工具、接入适配器、售后写接口从哪来（方案 §8.2、§8.4、§9.1、§15 Phase 0）。

这三样必须由平台云内的代码实现并注入（开放平台接口只能在平台云里调，本仓库只有模拟实现）。
部署时用环境变量 RODIT_PLATFORM_BINDINGS 指定一个工厂（``包.模块:函数``），入口导入并调用它：

    def build(ctx: rodit.ops.BindingContext) -> rodit.ops.PlatformBindings: ...

ctx 里有平台、生效配置、时钟和 buyer_ref 哈希器（适配器要用同一个哈希器把平台买家 ID 变成
buyer_ref，§8.3）。

- 模式 C 单独部署（Phase 0，§8.1 不需要任何平台数据）：不给工厂，设 RODIT_MODE_C_ONLY=1，
  入口用 mode_c_bindings（NullToolGateway + StandaloneAssistAdapter，没有写接口）；
- 生产模式拒绝模拟实现（rodit.testing、rodit.tools.mock_platform、各平台的 *MockAdapter）：
  那是合成数据，接进生产等于给商家看假订单；模式 C 的适配器不算模拟（它本来就不接平台）；
- 本地试用（RODIT_ENV=local）没给工厂时用 demo_bindings：合成演示世界 + 模拟适配器。
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from rodit.adapters import (
    BaseMockAdapter,
    DouyinMockAdapter,
    NullToolGateway,
    StandaloneAssistAdapter,
    TaobaoMockAdapter,
)
from rodit.core.clock import Clock
from rodit.core.config import EngineConfig
from rodit.core.enums import Platform
from rodit.core.pii import BuyerRefHasher
from rodit.core.ports import (
    AfterSaleActionPort,
    ImageTextExtractor,
    ItemCatalog,
    PlatformAdapter,
    ToolGateway,
)

MOCK_MODULE_PREFIXES: tuple[str, ...] = ("rodit.testing", "rodit.tools.mock_platform")


class BindingError(RuntimeError):
    """平台绑定缺失、不合法或在生产模式下用了模拟实现。"""


@dataclass(frozen=True)
class BindingContext:
    """交给平台绑定工厂的上下文。"""

    platform: Platform
    config: EngineConfig
    clock: Clock
    hasher: BuyerRefHasher


@dataclass
class PlatformBindings:
    """平台云代码提供的实现。

    - tools：平台只读工具（订单、物流、售后、库存、价格，§7.1），platform 必须是本单元的平台；
    - adapters：接入适配器（模式 A / B，需要时也可以带上模式 C 的 StandaloneAssistAdapter）；
    - action_port：售后写接口（二期才申请写权限，§8.4；MVP 为 None，审核只给建议）；
    - shop_ids：本单元服务的店铺（订购列表），授权即将过期告警用（§12.4）；不给时只看令牌文件。
    - catalog / ocr：平台商品接口和详情图 OCR（§5.2.3 商品自动同步），不给时商品同步不可用。
    """

    tools: ToolGateway
    adapters: Sequence[PlatformAdapter]
    action_port: AfterSaleActionPort | None = None
    catalog: ItemCatalog | None = None
    ocr: ImageTextExtractor | None = None
    shop_ids: Callable[[], Iterable[str]] | None = None
    mode_c_only: bool = field(default=False)


BindingsFactory = Callable[[BindingContext], PlatformBindings]


def load_factory(spec: str) -> BindingsFactory:
    """``包.模块:函数`` → 工厂函数。"""
    module_name, sep, attr = spec.strip().partition(":")
    if not sep or not module_name or not attr:
        raise BindingError(f"RODIT_PLATFORM_BINDINGS 格式应为 包.模块:函数，现在是 {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise BindingError(f"找不到平台绑定模块 {module_name!r}（{type(e).__name__}）") from e
    target: object = module
    for part in attr.split("."):
        target = getattr(target, part, None)
        if target is None:
            raise BindingError(f"平台绑定模块 {module_name!r} 里没有 {attr!r}")
    if not callable(target):
        raise BindingError(f"{spec!r} 不是可调用对象")
    return target  # type: ignore[return-value]


def mode_c_bindings(ctx: BindingContext) -> PlatformBindings:
    """模式 C 独立话术助手单独部署（§8.1）：没有平台数据、没有写接口。"""
    return PlatformBindings(
        tools=NullToolGateway(ctx.platform),
        adapters=[StandaloneAssistAdapter(ctx.platform, ctx.clock, ctx.hasher)],
        action_port=None,
        mode_c_only=True,
    )


def validate_bindings(
    bindings: object, platform: Platform, *, allow_mock: bool
) -> PlatformBindings:
    """核对工厂返回的绑定：类型、平台一致（§9.1 数据不跨云）；生产模式下不许有模拟实现。"""
    if not isinstance(bindings, PlatformBindings):
        raise BindingError("平台绑定工厂必须返回 rodit.ops.PlatformBindings")
    if getattr(bindings.tools, "platform", None) != platform:
        raise BindingError(f"平台工具不属于本单元的平台 {platform.value}")
    adapters = list(bindings.adapters)
    if not adapters:
        raise BindingError("至少要有一个接入适配器")
    for adapter in adapters:
        if getattr(adapter, "platform", None) != platform:
            raise BindingError(f"适配器 {type(adapter).__name__} 不属于本单元的平台")
    if isinstance(bindings.tools, NullToolGateway) and not bindings.mode_c_only:
        raise BindingError("NullToolGateway 只能用于模式 C 单独部署（RODIT_MODE_C_ONLY=1）")
    if not allow_mock:
        mocks = [
            type(obj).__name__
            for obj in (bindings.tools, *adapters, bindings.action_port)
            if obj is not None and is_mock(obj)
        ]
        if mocks:
            raise BindingError(
                "生产模式不能用模拟实现（合成数据）：" + "、".join(sorted(set(mocks)))
            )
    return bindings


def is_mock(obj: object) -> bool:
    """本仓库的模拟实现（合成数据）。模式 C 的 StandaloneAssistAdapter 不算。"""
    if isinstance(obj, StandaloneAssistAdapter):
        return False
    if isinstance(obj, BaseMockAdapter):
        return True
    module = type(obj).__module__ or ""
    return any(module == p or module.startswith(p + ".") for p in MOCK_MODULE_PREFIXES)


# ================================================================ 本地试用（合成数据）


def demo_bindings(ctx: BindingContext) -> PlatformBindings:
    """本地试用（RODIT_ENV=local）：合成演示世界的平台数据 + 模拟适配器（模式 B）+ 模式 C。
    售后写接口也是模拟的（FakeActionPort，只改合成数据）。生产模式会拒绝这些实现。"""
    from rodit.app.demo import SUPPORTED_ACTIONS
    from rodit.testing.fakes import FakeActionPort, InMemoryToolGateway
    from rodit.testing.fixtures import build_demo_world

    world = build_demo_world(ctx.clock.now())
    tools = InMemoryToolGateway(world, ctx.platform)
    adapters: list[PlatformAdapter] = [StandaloneAssistAdapter(ctx.platform, ctx.clock, ctx.hasher)]
    # 拼多多没有查到官方接入能力（§8.1），只有模式 C
    if ctx.platform == Platform.TAOBAO:
        adapters.append(TaobaoMockAdapter(tools, ctx.clock, ctx.hasher, access="B"))
    elif ctx.platform == Platform.DOUYIN:
        adapters.append(DouyinMockAdapter(tools, ctx.clock, ctx.hasher, access="B"))
    action_port = FakeActionPort(
        ctx.platform, supported=SUPPORTED_ACTIONS[ctx.platform], gateway=tools, clock=ctx.clock
    )
    shops = sorted(s.id for s in world.shops.values() if s.platform == ctx.platform)
    return PlatformBindings(
        tools=tools, adapters=adapters, action_port=action_port, shop_ids=lambda: shops
    )


__all__ = [
    "MOCK_MODULE_PREFIXES",
    "BindingContext",
    "BindingError",
    "BindingsFactory",
    "PlatformBindings",
    "demo_bindings",
    "is_mock",
    "load_factory",
    "mode_c_bindings",
    "validate_bindings",
]
