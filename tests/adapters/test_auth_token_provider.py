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


def test_report_rejected_with_the_current_token_marks_the_shop(setup):
    _, repo, guard, provider = setup
    token = provider.access_token(SHOP_A)
    provider.report_rejected(SHOP_A, token)
    assert repo.get(SHOP_A).needs_reauth is True
    assert not guard.is_authorized(SHOP_A)


def test_report_rejected_with_a_replaced_token_leaves_the_new_grant_alone(setup):
    """在途调用拿着旧令牌被平台拒绝时，续期 / 重新授权已经换上了新令牌：不标记新授权。"""
    _, repo, guard, provider = setup
    old = provider.access_token(SHOP_A)
    auth = repo.get(SHOP_A)
    repo.save(auth.model_copy(update={"access_token": SecretStr("tok-synthetic-provider-0002")}))
    provider.report_rejected(SHOP_A, old)
    assert repo.get(SHOP_A).needs_reauth is False and guard.is_authorized(SHOP_A)
    assert provider.access_token(SHOP_A).get_secret_value() == "tok-synthetic-provider-0002"
    provider.report_rejected(SHOP_B, old)  # 没有授权的店铺：不报错
    assert repo.get(SHOP_B) is None


def test_report_rejected_does_not_mark_a_grant_replaced_between_check_and_mark(setup):
    """比对令牌之后、标记之前换了新授权（另一个进程的续期）：按读到的版本比较并交换，
    不标记新授权。"""
    _, repo, guard, provider = setup
    old = provider.access_token(SHOP_A)
    real_mark = repo.mark_needs_reauth

    def mark(shop_id: str, *, expected_version: int | None = None) -> bool:
        current = repo.get(shop_id)
        repo.save(current.model_copy(update={"access_token": SecretStr("tok-renewed-0003")}))
        return real_mark(shop_id, expected_version=expected_version)

    repo.mark_needs_reauth = mark  # type: ignore[method-assign]
    provider.report_rejected(SHOP_A, old)
    repo.mark_needs_reauth = real_mark  # type: ignore[method-assign]
    got = repo.get(SHOP_A)
    assert got.needs_reauth is False and got.access_token.get_secret_value() == "tok-renewed-0003"
