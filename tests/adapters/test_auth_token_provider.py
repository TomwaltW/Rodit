"""店铺授权（§8.4）：需要重新授权的店铺停止处理；平台云代码经 TokenProvider 取令牌。"""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import SecretStr

from rodit.adapters import MVP_SCOPES, SCOPE_REFUND_WRITE, AuthGuard, AuthGuardTokenProvider
from rodit.core.errors import AuthExpired
from rodit.core.models import ShopAuth
from rodit.core.ports import TokenProvider
from rodit.inmemory import InMemoryShopAuthRepository
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A, SHOP_B

TOKEN = "tok-synthetic-provider-0001"


@pytest.fixture
def setup():
    clock = FakeClock()
    repo = InMemoryShopAuthRepository()
    repo.save(
        ShopAuth(
            shop_id=SHOP_A,
            access_token=SecretStr(TOKEN),
            scopes=sorted(MVP_SCOPES),
            expires_at=clock.now() + timedelta(days=10),
        )
    )
    guard = AuthGuard(repo, clock)
    return clock, repo, guard, AuthGuardTokenProvider(guard)


def test_needs_reauth_blocks_processing_and_shows_in_expiring(setup):
    clock, repo, guard, _ = setup
    assert guard.is_authorized(SHOP_A)
    assert guard.expiring([SHOP_A], within=timedelta(days=1)) == []
    assert guard.mark_needs_reauth(SHOP_A) is True
    with pytest.raises(AuthExpired, match="重新授权") as ei:
        guard.require(SHOP_A)
    assert TOKEN not in str(ei.value)
    assert not guard.is_authorized(SHOP_A)
    # 授权还有 10 天才到期，但需要重新授权：列进到期告警
    assert guard.expiring([SHOP_A], within=timedelta(days=1)) == [SHOP_A]
    assert guard.mark_needs_reauth(SHOP_B) is False  # 没有授权
    # 商家重新授权（整条保存）后恢复
    auth = repo.get(SHOP_A)
    assert auth is not None
    repo.save(auth.model_copy(update={"needs_reauth": False}))
    assert guard.require(SHOP_A).access_token.get_secret_value() == TOKEN


def test_token_provider_returns_valid_tokens_only(setup):
    clock, repo, guard, provider = setup
    assert isinstance(provider, TokenProvider)
    token = provider.access_token(SHOP_A)
    assert isinstance(token, SecretStr) and token.get_secret_value() == TOKEN
    assert TOKEN not in repr(provider)
    with pytest.raises(AuthExpired):
        provider.access_token(SHOP_B)  # 没有授权
    with pytest.raises(AuthExpired, match="refund.write"):
        provider.access_token(SHOP_A, MVP_SCOPES | {SCOPE_REFUND_WRITE})  # 权限不够
    clock.advance(days=11)
    with pytest.raises(AuthExpired, match="过期"):
        provider.access_token(SHOP_A)


def test_report_rejected_marks_the_shop_for_reauthorization(setup):
    _, repo, guard, provider = setup
    provider.report_rejected(SHOP_A)
    auth = repo.get(SHOP_A)
    assert auth is not None and auth.needs_reauth is True
    assert auth.access_token.get_secret_value() == TOKEN  # 只改标记
    with pytest.raises(AuthExpired) as ei:
        provider.access_token(SHOP_A)
    assert TOKEN not in str(ei.value)
    provider.report_rejected(SHOP_B)  # 没有授权的店铺：不报错
    assert repo.get(SHOP_B) is None
