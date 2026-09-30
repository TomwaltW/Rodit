"""平台绑定：平台只读工具、接入适配器、售后写接口从哪来（方案 §8.2、§8.4、§9.1、§15 Phase 0）。

这三样必须由平台云内的代码实现并注入（开放平台接口只能在平台云里调，本仓库只有模拟实现）。
部署时用环境变量 RODIT_PLATFORM_BINDINGS 指定一个工厂（``包.模块:函数``），入口导入并调用它：

    def build(ctx: rodit.ops.BindingContext) -> rodit.ops.PlatformBindings: ...

ctx 里有平台、生效配置、时钟和 buyer_ref 哈希器（适配器要用同一个哈希器把平台买家 ID 变成
buyer_ref，§8.3），还有两个"先交出、后接上"的入口（工厂被调用时平台单元还没组装好）：
- ctx.tokens（core.ports.TokenProvider）：平台只读工具、适配器调用开放平台接口时取店铺令牌；
  单元组装好之后接到 Unit.tokens（AuthGuardTokenProvider），之前调用抛 RuntimeError；
- ctx.on_subscription：平台推送的订购事件（SubscriptionEvent）交给它；服务组装好之后接到
  AssistService.tenancy.handle_subscription。
平台云代码还可以提供：令牌续期接口（token_refresher）、订购推送解析（parse_subscription）、
大模型的出口（llm_transport，签名同 rodit.llm.openai_compat.Transport，§12.3 经平台可报备的通道）。

- 模式 C 单独部署（Phase 0，§8.1 不需要任何平台数据）：不给工厂，设 RODIT_MODE_C_ONLY=1，
  入口用 mode_c_bindings（NullToolGateway + StandaloneAssistAdapter，没有写接口）；
- 生产模式拒绝模拟实现（rodit.testing、rodit.tools.mock_platform、各平台的 *MockAdapter）：
  那是合成数据，接进生产等于给商家看假订单；模式 C 的适配器不算模拟（它本来就不接平台）；
- 本地试用（RODIT_ENV=local）没给工厂时用 demo_bindings：合成演示世界 + 模拟适配器。
"""

from __future__ import annotations

import importlib
import inspect
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import SecretStr

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
from rodit.core.models import SubscriptionEvent
from rodit.core.pii import BuyerRefHasher
from rodit.core.ports import (
    AfterSaleActionPort,
    ImageTextExtractor,
    ItemCatalog,
    PlatformAdapter,
    TokenProvider,
    TokenRefreshPort,
    ToolGateway,
)
from rodit.core.scopes import MVP_SCOPES
from rodit.llm.openai_compat import Transport

MOCK_MODULE_PREFIXES: tuple[str, ...] = ("rodit.testing", "rodit.tools.mock_platform")


class BindingError(RuntimeError):
    """平台绑定缺失、不合法或在生产模式下用了模拟实现。"""


class LazyTokenProvider:
    """BindingContext.tokens：绑定工厂被调用时平台单元还没组装好，先交出这个代理，
    组装好之后 bind 到 Unit.tokens。没绑定时调用抛 RuntimeError（不会静默放行）。只能绑定一次。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target: TokenProvider | None = None

    def __repr__(self) -> str:
        return f"LazyTokenProvider(bound={self.bound})"

    @property
    def bound(self) -> bool:
        return self._target is not None

    def bind(self, provider: TokenProvider) -> None:
        with self._lock:
            if self._target is not None and self._target is not provider:
                raise RuntimeError("BindingContext.tokens 已经绑定过")
            self._target = provider

    def _get(self) -> TokenProvider:
        target = self._target
        if target is None:
            raise RuntimeError("平台单元还没组装好：BindingContext.tokens 还不能用")
        return target

    def access_token(self, shop_id: str, scopes: Iterable[str] = MVP_SCOPES) -> SecretStr:
        return self._get().access_token(shop_id, scopes)

    def report_rejected(self, shop_id: str) -> None:
        self._get().report_rejected(shop_id)


SubscriptionHandler = Callable[[SubscriptionEvent], None]


class LazySubscriptionHandler:
    """BindingContext.on_subscription：服务组装好之后 bind 到
    AssistService.tenancy.handle_subscription。没绑定时调用抛 RuntimeError（平台代码应当重推，
    而不是当作处理过了）。只能绑定一次。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target: SubscriptionHandler | None = None

    def __repr__(self) -> str:
        return f"LazySubscriptionHandler(bound={self.bound})"

    @property
    def bound(self) -> bool:
        return self._target is not None

    def bind(self, handler: SubscriptionHandler) -> None:
        with self._lock:
            if self._target is not None and self._target != handler:
                raise RuntimeError("BindingContext.on_subscription 已经绑定过")
            self._target = handler

    def __call__(self, event: SubscriptionEvent) -> None:
        target = self._target
        if target is None:
            raise RuntimeError("服务还没组装好：BindingContext.on_subscription 还不能用")
        target(event)


@dataclass(frozen=True)
class BindingContext:
    """交给平台绑定工厂的上下文。tokens / on_subscription 见模块说明（先交出、后接上）。"""

    platform: Platform
    config: EngineConfig
    clock: Clock
    hasher: BuyerRefHasher
    tokens: TokenProvider = field(default_factory=LazyTokenProvider)
    on_subscription: SubscriptionHandler = field(default_factory=LazySubscriptionHandler)


def bind_context(
    ctx: BindingContext,
    *,
    tokens: TokenProvider | None = None,
    on_subscription: SubscriptionHandler | None = None,
) -> None:
    """把 ctx 里的代理接到真正的实现上（serve 组装时调用）。ctx 里不是代理（调用方自己给了
    实现）时不动它。"""
    if tokens is not None and isinstance(ctx.tokens, LazyTokenProvider):
        ctx.tokens.bind(tokens)
    if on_subscription is not None and isinstance(ctx.on_subscription, LazySubscriptionHandler):
        ctx.on_subscription.bind(on_subscription)


@dataclass
class PlatformBindings:
    """平台云代码提供的实现。

    - tools：平台只读工具（订单、物流、售后、库存、价格，§7.1），platform 必须是本单元的平台；
    - adapters：接入适配器（模式 A / B，需要时也可以带上模式 C 的 StandaloneAssistAdapter）；
    - action_port：售后写接口（二期才申请写权限，§8.4；MVP 为 None，审核只给建议）；
    - shop_ids：本单元服务的店铺（订购列表），授权即将过期告警用（§12.4）；不给时只看令牌文件。
    - catalog / ocr：平台商品接口和详情图 OCR（§5.2.3 商品自动同步），不给时商品同步不可用。
    - token_refresher：平台的令牌续期接口（§8.4），不给时续期任务不可用（只能等商家重新授权）。
    - parse_subscription：平台订购推送（原始 dict）→ SubscriptionEvent；不认识的推送返回 None。
    - llm_transport：大模型的出口（签名同 rodit.llm.openai_compat.Transport），平台要求经它报备的
      通道调用时注入；不给时用默认的 urllib 出口。
    """

    tools: ToolGateway
    adapters: Sequence[PlatformAdapter]
    action_port: AfterSaleActionPort | None = None
    catalog: ItemCatalog | None = None
    ocr: ImageTextExtractor | None = None
    shop_ids: Callable[[], Iterable[str]] | None = None
    mode_c_only: bool = field(default=False)
    token_refresher: TokenRefreshPort | None = None
    parse_subscription: Callable[[dict[str, Any]], SubscriptionEvent | None] | None = None
    llm_transport: Transport | None = None


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
    """核对工厂返回的绑定：类型、平台一致（§9.1 数据不跨云）、适配器的 show_suggestions 签名；
    生产模式下不许有模拟实现。"""
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
        _check_show_suggestions(adapter)
    if isinstance(bindings.tools, NullToolGateway) and not bindings.mode_c_only:
        raise BindingError("NullToolGateway 只能用于模式 C 单独部署（RODIT_MODE_C_ONLY=1）")
    if bindings.catalog is not None and getattr(bindings.catalog, "platform", None) != platform:
        raise BindingError(f"商品接口不属于本单元的平台 {platform.value}")
    refresher = bindings.token_refresher
    if refresher is not None and getattr(refresher, "platform", None) != platform:
        raise BindingError(f"令牌续期接口不属于本单元的平台 {platform.value}")
    for name in ("parse_subscription", "llm_transport"):
        hook = getattr(bindings, name)
        if hook is not None and not callable(hook):
            raise BindingError(f"平台绑定的 {name} 必须是可调用对象")
    if not allow_mock:
        mocks = [
            _label(obj)
            for obj in (
                bindings.tools,
                *adapters,
                bindings.action_port,
                bindings.catalog,
                bindings.ocr,
                bindings.token_refresher,
                bindings.parse_subscription,
                bindings.llm_transport,
            )
            if obj is not None and is_mock(obj)
        ]
        if mocks:
            raise BindingError(
                "生产模式不能用模拟实现（合成数据）：" + "、".join(sorted(set(mocks)))
            )
    return bindings


SHOW_SUGGESTION_KEYWORDS: tuple[str, ...] = ("alerts", "notice")
"""PlatformAdapter.show_suggestions 必须接受的关键字参数（标红提醒、没有推荐的原因）。"""


def _check_show_suggestions(adapter: object) -> None:
    """适配器的 show_suggestions 要能接受 alerts / notice 关键字参数（core.ports.PlatformAdapter）。

    服务每次展示推荐都带上这两个参数；按旧签名（只有 shop_id、conversation_id、cards）写的适配器
    每次都会 TypeError，而展示失败只记一条警告（不抛给平台）：侧边栏从此看不到任何推荐。
    runtime_checkable 协议只核对属性名，查不出来，所以启动时按签名核对，不对就拒绝启动。
    看不到签名（如 C 实现）时不拦。"""
    name = type(adapter).__name__
    show = getattr(adapter, "show_suggestions", None)
    if not callable(show):
        raise BindingError(f"适配器 {name} 没有 show_suggestions")
    try:
        params = inspect.signature(show).parameters
    except (TypeError, ValueError):
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    keyword_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    missing = [
        kw
        for kw in SHOW_SUGGESTION_KEYWORDS
        if kw not in params or params[kw].kind not in keyword_kinds
    ]
    if missing:
        raise BindingError(
            f"适配器 {name}.show_suggestions 要接受关键字参数 "
            + " / ".join(missing)
            + "（core.ports.PlatformAdapter：标红提醒和没有推荐的原因）"
        )


def is_mock(obj: object) -> bool:
    """本仓库的模拟实现（合成数据）。模式 C 的 StandaloneAssistAdapter 不算。
    函数（如 parse_subscription、llm_transport）按定义它的模块判断。"""
    if isinstance(obj, StandaloneAssistAdapter):
        return False
    if isinstance(obj, BaseMockAdapter):
        return True
    modules = {type(obj).__module__ or "", getattr(obj, "__module__", None) or ""}
    return any(
        module == prefix or module.startswith(prefix + ".")
        for module in modules
        for prefix in MOCK_MODULE_PREFIXES
    )


def _label(obj: object) -> str:
    """报错里用的名字：函数用限定名，对象用类名。"""
    name = getattr(obj, "__qualname__", None)
    return name if isinstance(name, str) and name else type(obj).__name__


# ================================================================ 本地试用（合成数据）


def demo_bindings(ctx: BindingContext) -> PlatformBindings:
    """本地试用（RODIT_ENV=local）：合成演示世界的平台数据 + 模拟适配器（模式 B）+ 模式 C。
    售后写接口也是模拟的（FakeActionPort，只改合成数据）。生产模式会拒绝这些实现。"""
    from rodit.app.demo import SUPPORTED_ACTIONS
    from rodit.testing.fakes import (
        FakeActionPort,
        FakeSubscriptionParser,
        FakeTokenRefresher,
        InMemoryToolGateway,
    )
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
        tools=tools,
        adapters=adapters,
        action_port=action_port,
        shop_ids=lambda: shops,
        token_refresher=FakeTokenRefresher(ctx.platform, clock=ctx.clock),
        parse_subscription=FakeSubscriptionParser(ctx.platform),
    )


__all__ = [
    "MOCK_MODULE_PREFIXES",
    "SHOW_SUGGESTION_KEYWORDS",
    "BindingContext",
    "BindingError",
    "BindingsFactory",
    "LazySubscriptionHandler",
    "LazyTokenProvider",
    "PlatformBindings",
    "SubscriptionHandler",
    "bind_context",
    "demo_bindings",
    "is_mock",
    "load_factory",
    "mode_c_bindings",
    "validate_bindings",
]
