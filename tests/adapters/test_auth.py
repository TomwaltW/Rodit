"""店铺授权（方案 §8.4）：过期、权限不足、最小权限、取消订购后删除数据、过期告警。"""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import SecretStr

from rodit.adapters import (
    MVP_SCOPES,
    PHASE2_SCOPES,
    SCOPE_REFUND_WRITE,
    AuthGuard,
    StandaloneAssistAdapter,
    excess_scopes,
    scopes_for_phase,
)
from rodit.core.enums import KnowledgeStatus, Platform
from rodit.core.errors import AuthExpired
from rodit.core.models import KnowledgeGap, ShopAuth
from rodit.inmemory import (
    InMemoryConversationRepository,
    InMemoryFeedbackRepository,
    InMemoryKnowledgeRepository,
    InMemoryShopAuthRepository,
)
from rodit.testing.fakes import make_message
from rodit.testing.fixtures import SHOP_A, SHOP_B, seed_knowledge

TOKEN = "tok-synthetic-not-real"


def _auth(clock, shop_id=SHOP_A, *, scopes=MVP_SCOPES, days: float = 30) -> ShopAuth:
    return ShopAuth(
        shop_id=shop_id,
        access_token=SecretStr(TOKEN),
        scopes=sorted(scopes),
        expires_at=clock.now() + timedelta(days=days),
    )


@pytest.fixture
def repo() -> InMemoryShopAuthRepository:
    return InMemoryShopAuthRepository()


@pytest.fixture
def guard(repo, clock) -> AuthGuard:
    return AuthGuard(repo, clock)


def test_scope_constants_follow_minimum_privilege():
    assert MVP_SCOPES == {"order.read", "logistics.read", "refund.read", "item.read"}
    assert all(s.endswith(".read") for s in MVP_SCOPES)
    assert PHASE2_SCOPES == MVP_SCOPES | {SCOPE_REFUND_WRITE}
    assert scopes_for_phase(1) == MVP_SCOPES and scopes_for_phase(2) == PHASE2_SCOPES
    assert excess_scopes(["order.read", "refund.write"]) == {"refund.write"}
    assert excess_scopes(PHASE2_SCOPES, phase=2) == frozenset()
    with pytest.raises(ValueError):
        scopes_for_phase(3)  # type: ignore[arg-type]


def test_require_returns_valid_auth(repo, guard, clock):
    repo.save(_auth(clock))
    auth = guard.require(SHOP_A)
    assert auth.shop_id == SHOP_A and set(auth.scopes) == MVP_SCOPES
    assert guard.is_authorized(SHOP_A)


def test_require_missing_auth(guard):
    with pytest.raises(AuthExpired, match="没有授权"):
        guard.require(SHOP_A)
    assert not guard.is_authorized(SHOP_A)


def test_require_expiry_boundary(repo, guard, clock):
    repo.save(_auth(clock, days=1))
    clock.advance(days=1, seconds=-1)
    assert guard.is_authorized(SHOP_A)  # 到期前 1 秒仍有效
    clock.advance(seconds=1)
    with pytest.raises(AuthExpired, match="过期") as exc:
        guard.require(SHOP_A)  # expires_at == 现在：已过期
    assert TOKEN not in str(exc.value)


def test_require_insufficient_scope(repo, guard, clock):
    repo.save(_auth(clock, scopes={"order.read", "logistics.read"}))
    with pytest.raises(AuthExpired, match="缺少权限") as exc:
        guard.require(SHOP_A)
    assert "refund.read" in str(exc.value) and "item.read" in str(exc.value)
    assert guard.is_authorized(SHOP_A, {"order.read"})


def test_write_scope_only_in_phase2(repo, guard, clock):
    repo.save(_auth(clock))
    with pytest.raises(AuthExpired, match="refund.write"):
        guard.require(SHOP_A, PHASE2_SCOPES)
    repo.save(_auth(clock, scopes=PHASE2_SCOPES))
    guard.require(SHOP_A, PHASE2_SCOPES)


def test_auth_is_per_shop(repo, guard, clock):
    repo.save(_auth(clock, SHOP_B))
    with pytest.raises(AuthExpired):
        guard.require(SHOP_A)
    guard.require(SHOP_B)


def test_expiring_alerts(repo, guard, clock):
    repo.save(_auth(clock, SHOP_A, days=2))
    repo.save(_auth(clock, SHOP_B, days=30))
    shops = [SHOP_B, SHOP_A, "s_no_auth"]
    assert guard.expiring(shops, within=timedelta(days=3)) == [SHOP_A, "s_no_auth"]
    assert guard.expiring(shops, within=timedelta(days=31)) == shops
    assert guard.expiring(shops, within=timedelta(0)) == ["s_no_auth"]


def test_revoke_deletes_token_and_purges_every_repo(world, repo, guard, clock, hasher):
    knowledge = InMemoryKnowledgeRepository(clock)
    seed_knowledge(knowledge, world)
    conversations = InMemoryConversationRepository()
    conversations.add_message(make_message("在吗", world=world, msg_id="a1"))
    conversations.add_message(make_message("在吗", world=world, shop_id=SHOP_B, msg_id="b1"))
    feedback = InMemoryFeedbackRepository()
    feedback.add_gap(KnowledgeGap(shop_id=SHOP_A, missing="policy:gift", created_at=clock.now()))
    feedback.add_gap(KnowledgeGap(shop_id=SHOP_B, missing="policy:gift", created_at=clock.now()))
    adapter = StandaloneAssistAdapter(Platform.TAOBAO, clock, hasher)
    adapter.on_message({"shop_id": SHOP_A, "session_id": "w", "question": "在吗"})
    repo.save(_auth(clock, SHOP_A))
    repo.save(_auth(clock, SHOP_B))
    a_knowledge = len(knowledge.list_knowledge(SHOP_A, status=None))

    counts = guard.revoke(SHOP_A, [knowledge, conversations, feedback, adapter])

    assert counts["auth"] == 1
    assert counts["InMemoryKnowledgeRepository"] > a_knowledge  # 店铺 + 政策 + 商品 + FAQ
    assert counts["InMemoryConversationRepository"] == 2  # 会话 + 消息
    assert counts["InMemoryFeedbackRepository"] == 1
    assert counts["StandaloneAssistAdapter"] == 1
    assert repo.get(SHOP_A) is None
    with pytest.raises(AuthExpired):
        guard.require(SHOP_A)
    assert knowledge.get_shop(SHOP_A) is None
    assert knowledge.list_knowledge(SHOP_A, status=None) == []
    assert conversations.get_conversation(SHOP_A, "conv_889") is None
    # 其他店铺不受影响
    guard.require(SHOP_B)
    assert knowledge.list_knowledge(SHOP_B, status=KnowledgeStatus.APPROVED)
    assert conversations.get_conversation(SHOP_B, "conv_889") is not None
    assert len(feedback.list_gaps(SHOP_B)) == 1
    # 重复撤销是幂等的
    again = guard.revoke(SHOP_A, [knowledge, conversations, feedback])
    assert again == {
        "auth": 0,
        "InMemoryKnowledgeRepository": 0,
        "InMemoryConversationRepository": 0,
        "InMemoryFeedbackRepository": 0,
    }


def test_revoke_accepts_callables_and_names_duplicates(repo, guard, clock):
    repo.save(_auth(clock))
    calls: list[str] = []

    def purge_cache(shop_id: str) -> int:
        calls.append(shop_id)
        return 7

    assert guard.revoke(SHOP_A, purge_cache) == {"auth": 1, "purge_cache": 7}
    counts = guard.revoke(
        SHOP_A, [InMemoryFeedbackRepository(), InMemoryFeedbackRepository(), purge_cache]
    )
    assert counts == {
        "auth": 0,
        "InMemoryFeedbackRepository": 0,
        "InMemoryFeedbackRepository#2": 0,
        "purge_cache": 7,
    }
    assert calls == [SHOP_A, SHOP_A]
    assert guard.revoke(SHOP_A) == {"auth": 0}


class _UndecryptableAuthRepository(InMemoryShopAuthRepository):
    """令牌读不出来（旧版加密、密钥不对、数据损坏）：get 抛错，删除照常。"""

    def get(self, shop_id: str) -> ShopAuth | None:
        raise RuntimeError(f"店铺 {shop_id} 的令牌无法解密")


def test_revoke_does_not_need_to_decrypt_the_token(world, clock):
    """取消订购只删除，用不着令牌明文：令牌解不出来也要删掉令牌、清理店铺数据（§8.4）。"""
    repo = _UndecryptableAuthRepository()
    repo.save(_auth(clock, SHOP_A))
    conversations = InMemoryConversationRepository()
    conversations.add_message(make_message("在吗", world=world, msg_id="a1"))
    counts = AuthGuard(repo, clock).revoke(SHOP_A, [conversations])
    assert counts == {"auth": 1, "InMemoryConversationRepository": 2}
    assert repo.delete(SHOP_A) is False  # 已经删掉了
    assert conversations.get_conversation(SHOP_A, "conv_889") is None
