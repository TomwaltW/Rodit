"""取消订购不需要解密令牌（§8.4）。

升级前用库内 pgcrypto 加密的令牌、换过密钥或数据损坏时 PgShopAuthRepository.get 抛
TokenDecryptionError。取消订购只删除：令牌和店铺数据照样删掉，不能因为读不出令牌卡住
（取消订购的商家不会再来重新授权）。全部合成数据；密钥只用于测试。
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from rodit.adapters import AuthGuard
from rodit.storage import Database, PgConversationRepository, PgShopAuthRepository
from rodit.storage.repositories import TokenDecryptionError
from rodit.testing.fakes import FakeClock, make_message
from rodit.testing.fixtures import SHOP_A, SHOP_B, build_demo_world

pytestmark = pytest.mark.pg

WORLD = build_demo_world()


def _count(admin_db: Database, table: str, shop_id: str) -> int:
    with admin_db.session(None) as conn:
        row = conn.execute(
            f"SELECT count(*) AS n FROM {table} WHERE shop_id = %s",  # noqa: S608 — 表名是常量
            (shop_id,),
        ).fetchone()
    return int(row["n"])


def test_revoke_deletes_legacy_pgcrypto_token_and_purges_shop_data(
    pg_dsn, admin_db, truncate, token_key
):
    with admin_db.session(SHOP_A) as conn:  # 升级前 save() 的写法：库内 pgcrypto 加密
        conn.execute(
            "INSERT INTO shop_auth (shop_id, access_token, scopes, expires_at)"
            " VALUES (%s, pgp_sym_encrypt(%s, %s, 'cipher-algo=aes256'), %s, %s)",
            (
                SHOP_A,
                "tok-synthetic-legacy",
                token_key,
                ["order.read"],
                WORLD.now + timedelta(days=30),
            ),
        )
    db = Database(pg_dsn)
    repo = PgShopAuthRepository(db, encryption_key=token_key)
    with pytest.raises(TokenDecryptionError):
        repo.get(SHOP_A)
    conversations = PgConversationRepository(db)
    conversations.add_message(make_message("在吗", world=WORLD, msg_id="m_legacy"))

    counts = AuthGuard(repo, FakeClock(WORLD.now)).revoke(SHOP_A, [conversations])

    assert counts == {"auth": 1, "PgConversationRepository": 2}
    assert _count(admin_db, "shop_auth", SHOP_A) == 0
    assert _count(admin_db, "message", SHOP_A) == 0
    assert repo.get(SHOP_A) is None
    assert repo.delete(SHOP_A) is False
    assert repo.delete(SHOP_B) is False
