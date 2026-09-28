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
