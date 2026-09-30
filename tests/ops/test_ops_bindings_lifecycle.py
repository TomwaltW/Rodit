"""平台绑定里的店铺生命周期入口（方案 §8.4、§9.1）：

- BindingContext.tokens / on_subscription 在工厂被调用时是代理，平台单元、应用服务组装好之后才接上；
- PlatformBindings 的令牌续期接口、订购推送解析、大模型出口：平台一致、生产模式不许是模拟实现。
不连数据库、不调大模型（本地试用 + 内存仓储 + 离线替身）。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest
from ops_support import FakeReadinessDb, cloud_bindings
from pydantic import SecretStr

from rodit.adapters import AuthGuardTokenProvider
from rodit.app.unit import in_memory_repos
from rodit.core.config import EngineConfig
from rodit.core.enums import Platform
from rodit.core.errors import AuthExpired
from rodit.core.models import ShopAuth
from rodit.core.pii import BuyerRefHasher
from rodit.ops.bindings import (
    BindingContext,
    BindingError,
    LazySubscriptionHandler,
    LazyTokenProvider,
    PlatformBindings,
    bind_context,
    demo_bindings,
    is_mock,
    validate_bindings,
)
from rodit.ops.serve import build_server
from rodit.ops.settings import ServerSettings
from rodit.storage.db import MIGRATIONS
from rodit.testing.fakes import (
    FakeClock,
    FakeSubscriptionParser,
    FakeTokenRefresher,
    parse_subscription,
    subscription_push,
)
from rodit.testing.fixtures import SHOP_A, TEST_SALT, build_demo_world

ALL_VERSIONS = tuple(m.version for m in MIGRATIONS)


def _ctx(platform: Platform = Platform.TAOBAO) -> BindingContext:
    return BindingContext(
        platform=platform,
        config=EngineConfig(),
        clock=FakeClock(),
        hasher=BuyerRefHasher(TEST_SALT),
    )


# ================================================================ 代理


def test_context_starts_with_unbound_proxies():
    ctx = _ctx()
    assert isinstance(ctx.tokens, LazyTokenProvider) and not ctx.tokens.bound
    assert isinstance(ctx.on_subscription, LazySubscriptionHandler)
    assert not ctx.on_subscription.bound
    with pytest.raises(RuntimeError, match="还没组装好"):
        ctx.tokens.access_token(SHOP_A)
    with pytest.raises(RuntimeError, match="还没组装好"):
        ctx.tokens.report_rejected(SHOP_A)
    event = parse_subscription(subscription_push("cancelled", SHOP_A))
    assert event is not None
    with pytest.raises(RuntimeError, match="还没组装好"):
        ctx.on_subscription(event)
    assert _ctx().tokens is not ctx.tokens  # 每个上下文一份


def test_bind_context_binds_once():
    ctx = _ctx()
    seen: list[Any] = []

    class Provider:
        def access_token(self, shop_id, scopes=()):
            return SecretStr(f"tok-{shop_id}")

        def report_rejected(self, shop_id):
            seen.append(("rejected", shop_id))

    provider = Provider()
    bind_context(ctx, tokens=provider, on_subscription=seen.append)
    assert ctx.tokens.access_token(SHOP_A).get_secret_value() == f"tok-{SHOP_A}"
    ctx.tokens.report_rejected(SHOP_A)
    event = parse_subscription(subscription_push("cancelled", SHOP_A))
    ctx.on_subscription(event)
    assert seen == [("rejected", SHOP_A), event]
    bind_context(ctx, tokens=provider)  # 同一个实现：幂等
    with pytest.raises(RuntimeError, match="已经绑定过"):
        bind_context(ctx, tokens=Provider())
    with pytest.raises(RuntimeError, match="已经绑定过"):
        bind_context(ctx, on_subscription=lambda e: None)
    # 调用方自己给了实现（不是代理）：不动它
    own = replace(ctx, tokens=provider)
    bind_context(own, tokens=Provider())
    assert own.tokens is provider


@pytest.fixture
def local_env(ops_env, token_file, monkeypatch):
    for name in ("RODIT_LLM_BASE_URL", "RODIT_LLM_MODEL", "RODIT_LLM_API_KEY"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("RODIT_ENV", "local")
    monkeypatch.setenv("RODIT_AUTH_EXPIRY_CHECK_S", "0")


def test_serve_binds_tokens_to_the_unit_and_subscriptions_to_tenancy(local_env):
    """组装顺序：工厂拿到代理 → 单元组装好后 tokens 接到 Unit.tokens → 服务组装好后
    on_subscription 接到 AssistService.tenancy.handle_subscription。"""
    captured: list[BindingContext] = []
    world = build_demo_world()
    clock = FakeClock(world.now)

    def factory(ctx: BindingContext) -> PlatformBindings:
        captured.append(ctx)
        assert not ctx.tokens.bound and not ctx.on_subscription.bound  # 工厂被调用时还没接上
        return demo_bindings(ctx)

    server = build_server(
        ServerSettings.from_env(),
        bindings_factory=factory,
        clock=clock,
        repos=in_memory_repos(clock),
        database=FakeReadinessDb(ALL_VERSIONS),
    )
    [ctx] = captured
    assert ctx.tokens.bound and ctx.on_subscription.bound
    assert isinstance(server.unit.tokens, AuthGuardTokenProvider)
    # tokens 走的是单元的授权检查
    with pytest.raises(AuthExpired):
        ctx.tokens.access_token(SHOP_A)
    server.unit.repos.auth.save(
        ShopAuth(
            shop_id=SHOP_A,
            access_token=SecretStr("tok-synthetic-bound"),
            scopes=["order.read", "logistics.read", "refund.read", "item.read"],
            expires_at=clock.now() + timedelta(days=1),
        )
    )
    assert ctx.tokens.access_token(SHOP_A).get_secret_value() == "tok-synthetic-bound"
    ctx.tokens.report_rejected(SHOP_A)
    assert server.unit.repos.auth.get(SHOP_A).needs_reauth is True
    # on_subscription 交给店铺生命周期（AssistService.tenancy.handle_subscription）
    assert ctx.on_subscription._target == server.service.tenancy.handle_subscription  # noqa: SLF001


# ================================================================ 新的绑定字段


def test_demo_bindings_provide_lifecycle_fakes():
    ctx = _ctx(Platform.DOUYIN)
    bindings = demo_bindings(ctx)
    assert isinstance(bindings.token_refresher, FakeTokenRefresher)
    assert bindings.token_refresher.platform == Platform.DOUYIN
    assert isinstance(bindings.parse_subscription, FakeSubscriptionParser)
    assert bindings.llm_transport is None
    assert validate_bindings(bindings, Platform.DOUYIN, allow_mock=True) is bindings
    with pytest.raises(BindingError, match="FakeSubscriptionParser|FakeTokenRefresher"):
        validate_bindings(bindings, Platform.DOUYIN, allow_mock=False)


class CloudRefresher:
    """模拟平台云代码的续期接口（不是 rodit.testing 的类）。"""

    def __init__(self, platform: Platform) -> None:
        self.platform = platform

    def refresh(self, shop_id: str, refresh_token: SecretStr) -> Any:
        raise AssertionError("测试里不会调用")


def cloud_parse(raw: dict) -> Any:
    return None


def cloud_transport(url: str, headers: dict, body: dict, timeout_s: float) -> tuple[int, dict]:
    return 200, {}


def test_cloud_lifecycle_bindings_are_accepted_in_production():
    ctx = _ctx()
    bindings = replace(
        cloud_bindings(ctx),
        token_refresher=CloudRefresher(Platform.TAOBAO),
        parse_subscription=cloud_parse,
        llm_transport=cloud_transport,
    )
    assert validate_bindings(bindings, Platform.TAOBAO, allow_mock=False) is bindings


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("token_refresher", CloudRefresher(Platform.PDD), "令牌续期接口不属于本单元"),
        ("parse_subscription", "不是函数", "parse_subscription"),
        ("llm_transport", 42, "llm_transport"),
    ],
)
def test_lifecycle_bindings_are_checked(field, value, match):
    bindings = replace(cloud_bindings(_ctx()), **{field: value})
    with pytest.raises(BindingError, match=match):
        validate_bindings(bindings, Platform.TAOBAO, allow_mock=True)


@pytest.mark.parametrize(
    "field, value",
    [
        ("token_refresher", FakeTokenRefresher(Platform.TAOBAO)),
        ("parse_subscription", parse_subscription),  # 函数：按定义它的模块判断
        ("parse_subscription", FakeSubscriptionParser(Platform.TAOBAO)),
    ],
)
def test_production_rejects_mock_lifecycle_bindings(field, value):
    assert is_mock(value)
    bindings = replace(cloud_bindings(_ctx()), **{field: value})
    with pytest.raises(BindingError, match="模拟实现"):
        validate_bindings(bindings, Platform.TAOBAO, allow_mock=False)
    assert validate_bindings(bindings, Platform.TAOBAO, allow_mock=True) is bindings


def test_is_mock_for_functions_and_methods():
    assert not is_mock(cloud_parse) and not is_mock(cloud_transport)
    assert is_mock(FakeTokenRefresher().refresh)  # 模拟类的方法也算
    assert not is_mock(CloudRefresher(Platform.TAOBAO).refresh)
