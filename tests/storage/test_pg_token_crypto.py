"""店铺令牌加密（§8.4）：密钥和明文令牌不能作为 SQL 参数发给 PostgreSQL。

数据库开了 log_min_duration_statement / auto_explain 时，慢查询日志会连同绑定参数一起记下
（log_parameter_max_length 默认不截断），拿到数据库日志的人就拿到了每个店铺令牌的密钥。
所以加解密在应用里做，库里只存密文。这里包一层 Database，记下仓储发出的每条 SQL 和参数来核对。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import pytest
from pydantic import SecretStr

from rodit.core.models import ShopAuth
from rodit.storage import Database, PgShopAuthRepository
from rodit.storage.repositories import TokenDecryptionError
from rodit.testing.fixtures import SHOP_A, SHOP_B, build_demo_world

pytestmark = pytest.mark.pg

TOKEN = "tok-synthetic-plaintext-7788"
NOW = build_demo_world().now


class _RecordingConn:
    def __init__(self, conn: Any, sent: list[tuple[str, Any]]) -> None:
        self._conn = conn
        self._sent = sent

    def execute(self, query: Any, params: Any = None, **kw: Any) -> Any:
        self._sent.append((str(query), params))
        return self._conn.execute(query, params, **kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class RecordingDatabase(Database):
    """和 Database 一样，另外记下每条 SQL 和它的绑定参数。"""

    def __init__(self, dsn: str) -> None:
        super().__init__(dsn)
        self.sent: list[tuple[str, Any]] = []

    @contextmanager
    def session(self, shop_id: str | None, *, trainer: bool = False) -> Iterator[Any]:
        with super().session(shop_id, trainer=trainer) as conn:
            yield _RecordingConn(conn, self.sent)


def _auth(shop_id: str = SHOP_A, token: str = TOKEN) -> ShopAuth:
    return ShopAuth(
        shop_id=shop_id,
        access_token=SecretStr(token),
        scopes=["order.read"],
        expires_at=NOW + timedelta(days=30),
    )


def _flatten(params: Any) -> str:
    if params is None:
        return ""
    if isinstance(params, dict):
        params = list(params.values())
    return " ".join(repr(p) for p in params)


def test_key_and_plaintext_token_never_reach_the_database(pg_dsn, truncate, token_key):
    db = RecordingDatabase(pg_dsn)
    repo = PgShopAuthRepository(db, encryption_key=token_key)
    repo.save(_auth())
    got = repo.get(SHOP_A)
    assert got is not None and got.access_token.get_secret_value() == TOKEN
    repo.save(_auth(token="tok-synthetic-rotated-0001"))
    assert repo.get(SHOP_A).access_token.get_secret_value() == "tok-synthetic-rotated-0001"
    assert db.sent, "没有记录到 SQL"
    for query, params in db.sent:
        text = query + " " + _flatten(params)
        for secret in (token_key, TOKEN, "tok-synthetic-rotated-0001"):
            assert secret not in text, f"密钥或明文令牌发给了数据库：{query}"
        assert "pgp_sym" not in query  # 不再在库里加解密


def test_ciphertext_is_bound_to_its_shop(pg_dsn, admin_db, truncate, token_key):
    """把 A 店铺的密文挪到 B 店铺名下解不出来（店铺 ID 作为附加认证数据）。"""
    db = Database(pg_dsn)
    repo = PgShopAuthRepository(db, encryption_key=token_key)
    repo.save(_auth(SHOP_A, "tok-synthetic-a"))
    repo.save(_auth(SHOP_B, "tok-synthetic-b"))
    with admin_db.session(None) as conn:
        conn.execute(
            "UPDATE shop_auth SET access_token ="
            " (SELECT access_token FROM shop_auth WHERE shop_id = %s) WHERE shop_id = %s",
            (SHOP_A, SHOP_B),
        )
    assert repo.get(SHOP_A).access_token.get_secret_value() == "tok-synthetic-a"
    with pytest.raises(TokenDecryptionError) as ei:
        repo.get(SHOP_B)
    text = str(ei.value) + repr(ei.value) + str(ei.value.__cause__)
    assert token_key not in text and "tok-synthetic" not in text


def test_corrupted_ciphertext_fails_without_leaking(pg_dsn, admin_db, truncate, token_key):
    repo = PgShopAuthRepository(Database(pg_dsn), encryption_key=token_key)
    repo.save(_auth())
    for garbage in (b"", b"\x00\x01", b"not-a-token-ciphertext-at-all-0000000000"):
        with admin_db.session(None) as conn:
            conn.execute("UPDATE shop_auth SET access_token = %s", (garbage,))
        with pytest.raises(TokenDecryptionError) as ei:
            repo.get(SHOP_A)
        text = str(ei.value) + repr(ei.value) + str(ei.value.__cause__)
        assert token_key not in text and TOKEN not in text


# ================================================================ 续期令牌（迁移版本 5）

REFRESH = "tok-synthetic-refresh-9911"


def _with_refresh(shop_id: str = SHOP_A) -> ShopAuth:
    return _auth(shop_id).model_copy(
        update={
            "refresh_token": SecretStr(REFRESH),
            "refresh_expires_at": NOW + timedelta(days=90),
            "refreshed_at": NOW,
        }
    )


def test_refresh_token_never_reaches_the_database_in_plaintext(pg_dsn, truncate, token_key):
    db = RecordingDatabase(pg_dsn)
    repo = PgShopAuthRepository(db, encryption_key=token_key)
    repo.save(_with_refresh())
    got = repo.get(SHOP_A)
    assert got is not None and got.refresh_token is not None
    assert got.refresh_token.get_secret_value() == REFRESH
    assert repo.mark_needs_reauth(SHOP_A) is True
    assert [s for s, _ in repo.expiring_shop_ids(NOW + timedelta(days=365))] == [SHOP_A]
    for query, params in db.sent:
        text = query + " " + _flatten(params)
        for secret in (token_key, TOKEN, REFRESH):
            assert secret not in text, f"密钥或明文令牌发给了数据库：{query}"


def test_refresh_ciphertext_is_bound_to_its_shop_and_role(pg_dsn, admin_db, truncate, token_key):
    """续期令牌的附加认证数据是"店铺 ID + NUL + refresh"：
    和 access_token 的密文互换、挪到别的店铺都解不开。"""
    repo = PgShopAuthRepository(Database(pg_dsn), encryption_key=token_key)
    repo.save(_with_refresh(SHOP_A))
    repo.save(_with_refresh(SHOP_B))
    with admin_db.session(None) as conn:  # A：两个密文互换
        conn.execute(
            "UPDATE shop_auth SET access_token = refresh_token, refresh_token = access_token"
            " WHERE shop_id = %s",
            (SHOP_A,),
        )
    with pytest.raises(TokenDecryptionError) as ei:
        repo.get(SHOP_A)
    text = str(ei.value) + repr(ei.value) + str(ei.value.__cause__)
    assert token_key not in text and REFRESH not in text and TOKEN not in text
    with admin_db.session(None) as conn:  # B：换成 A 的续期令牌密文（A 此时的 access_token 列）
        conn.execute(
            "UPDATE shop_auth SET refresh_token ="
            " (SELECT access_token FROM shop_auth WHERE shop_id = %s) WHERE shop_id = %s",
            (SHOP_A, SHOP_B),
        )
    with pytest.raises(TokenDecryptionError):
        repo.get(SHOP_B)


def test_refresh_ciphertext_cannot_pose_as_the_access_token_of_a_lookalike_shop(
    pg_dsn, admin_db, truncate, token_key
):
    """店铺 ID 可以含冒号：店铺 X 的续期令牌密文挪到店铺 "X:refresh" 的 access_token 列，
    也解不开（附加认证数据用 NUL 分隔，TEXT 里不会有 NUL）。"""
    lookalike = SHOP_A + ":refresh"
    repo = PgShopAuthRepository(Database(pg_dsn), encryption_key=token_key)
    repo.save(_with_refresh(SHOP_A))
    repo.save(_auth(lookalike, "tok-synthetic-lookalike"))
    assert repo.get(lookalike).access_token.get_secret_value() == "tok-synthetic-lookalike"
    with admin_db.session(None) as conn:
        conn.execute(
            "UPDATE shop_auth SET access_token ="
            " (SELECT refresh_token FROM shop_auth WHERE shop_id = %s) WHERE shop_id = %s",
            (SHOP_A, lookalike),
        )
    with pytest.raises(TokenDecryptionError):
        repo.get(lookalike)
    # 反过来：假冒店铺的 access_token 密文当成 X 的续期令牌也解不开
    repo.save(_auth(lookalike, "tok-synthetic-lookalike"))
    with admin_db.session(None) as conn:
        conn.execute(
            "UPDATE shop_auth SET refresh_token ="
            " (SELECT access_token FROM shop_auth WHERE shop_id = %s) WHERE shop_id = %s",
            (lookalike, SHOP_A),
        )
    with pytest.raises(TokenDecryptionError):
        repo.get(SHOP_A)
    with pytest.raises(ValueError, match="NUL"):
        repo.save(_auth(SHOP_A + "\x00x"))


def test_unreadable_tokens_can_still_be_listed_and_marked(pg_dsn, admin_db, truncate, token_key):
    """令牌读不出来（换过密钥、数据损坏）的店铺照样列进快到期的店铺、照样能标记需要重新授权：
    这两个操作都不解密。"""
    repo = PgShopAuthRepository(Database(pg_dsn), encryption_key=token_key)
    repo.save(_with_refresh())
    with admin_db.session(None) as conn:
        conn.execute("UPDATE shop_auth SET access_token = 'garbage'::bytea")
    with pytest.raises(TokenDecryptionError):
        repo.get(SHOP_A)
    assert repo.expiring_shop_ids(NOW + timedelta(days=31)) == [(SHOP_A, NOW + timedelta(days=30))]
    assert repo.mark_needs_reauth(SHOP_A) is True
    with admin_db.session(None) as conn:
        flag = conn.execute("SELECT needs_reauth FROM shop_auth").fetchone()["needs_reauth"]
    assert flag is True
