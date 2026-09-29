"""生产用的令牌文件鉴权（方案 §8.4、§9.2、§12.3）：文件里只有哈希，令牌原文不落盘、不进日志。"""

from __future__ import annotations

import json
import logging
import os
import stat
from datetime import timedelta
from pathlib import Path

import pytest

from rodit.api import Authenticator, Principal
from rodit.ops.tokens import (
    TOKEN_PREFIX,
    MetricsEntry,
    PrincipalEntry,
    TokenFile,
    TokenFileAuthenticator,
    TokenFileError,
    append_entries,
    hash_token,
    issue_metrics_token,
    issue_principal_token,
    new_token,
    write_token_file,
)
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import AGENT, SHOP_A, SHOP_B, build_demo_world


class _Timer:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _write(path: Path, *issued) -> None:
    write_token_file(
        path,
        TokenFile(
            principals=[i.entry for i in issued if isinstance(i.entry, PrincipalEntry)],
            metrics=[i.entry for i in issued if isinstance(i.entry, MetricsEntry)],
        ),
    )


def test_new_tokens_are_random_and_prefixed():
    a, b = new_token(), new_token()
    assert a != b and a.startswith(TOKEN_PREFIX) and len(a) > 40
    assert hash_token(a) == hash_token(a) != hash_token(b)
    assert len(hash_token(a)) == 64


def test_authenticate_by_hash(tmp_path):
    agent = issue_principal_token(SHOP_A, AGENT, "agent")
    other = issue_principal_token(SHOP_B, "kf_b01", "manager")
    path = tmp_path / "tokens.json"
    _write(path, agent, other)
    auth = TokenFileAuthenticator(path)
    assert isinstance(auth, Authenticator)
    assert auth.authenticate(agent.token) == Principal(shop_id=SHOP_A, user_id=AGENT, role="agent")
    assert auth.authenticate(other.token).shop_id == SHOP_B
    for bad in ("", "short", new_token(), agent.token + "x", "x" * 1000, None, 123):
        assert auth.authenticate(bad) is None  # type: ignore[arg-type]
    assert auth.shop_ids() == [SHOP_A, SHOP_B]


def test_file_contains_only_hashes_with_owner_only_permissions(tmp_path):
    agent = issue_principal_token(SHOP_A, AGENT, "agent")
    path = tmp_path / "tokens.json"
    _write(path, agent)
    content = path.read_text(encoding="utf-8")
    assert agent.token not in content and hash_token(agent.token) in content
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_tokens_never_appear_in_repr_or_logs(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    agent = issue_principal_token(SHOP_A, AGENT, "agent")
    path = tmp_path / "tokens.json"
    _write(path, agent)
    auth = TokenFileAuthenticator(path, reload_interval_s=0)
    auth.authenticate(agent.token)
    auth.authenticate("rdt_wrong_token_0123456789")
    path.write_text("{broken", encoding="utf-8")
    auth.authenticate(agent.token)
    assert agent.token not in repr(auth) and agent.token not in repr(agent)
    assert agent.token not in caplog.text
    assert "rdt_" not in caplog.text


def test_expired_entries_are_rejected(tmp_path):
    now = build_demo_world().now
    clock = FakeClock(now)
    soon = issue_principal_token(SHOP_A, AGENT, "agent", expires_at=now + timedelta(hours=1))
    metrics = issue_metrics_token("prometheus", expires_at=now + timedelta(hours=1))
    path = tmp_path / "tokens.json"
    _write(path, soon, metrics)
    auth = TokenFileAuthenticator(path, clock=clock)
    assert auth.authenticate(soon.token) is not None
    assert auth.authenticate_metrics(metrics.token)
    clock.advance(hours=2)
    assert auth.authenticate(soon.token) is None
    assert not auth.authenticate_metrics(metrics.token)


def test_metrics_tokens_are_separate_from_principals(tmp_path):
    agent = issue_principal_token(SHOP_A, AGENT, "agent")
    metrics = issue_metrics_token("prometheus")
    path = tmp_path / "tokens.json"
    _write(path, agent, metrics)
    auth = TokenFileAuthenticator(path)
    assert auth.has_metrics_tokens
    assert auth.authenticate_metrics(metrics.token)
    assert not auth.authenticate_metrics(agent.token)
    assert not auth.authenticate_metrics(None)
    assert auth.authenticate(metrics.token) is None  # 抓取令牌不能调业务接口


def test_reload_on_change_and_keep_last_good_version(tmp_path):
    first = issue_principal_token(SHOP_A, AGENT, "agent")
    second = issue_principal_token(SHOP_A, "kf_03", "agent")
    path = tmp_path / "tokens.json"
    _write(path, first)
    timer = _Timer()
    auth = TokenFileAuthenticator(path, reload_interval_s=30, timer=timer)
    _write(path, second)  # 轮换：旧令牌作废，新令牌生效
    assert auth.authenticate(second.token) is None  # 还没到检查间隔
    timer.now = 31
    assert auth.authenticate(second.token) is not None
    assert auth.authenticate(first.token) is None
    # 新文件坏了：继续用上一版
    path.write_text('{"version": 1, "principals": [{"sha256": "zz"}]}', encoding="utf-8")
    timer.now = 62
    assert auth.authenticate(second.token) is not None
    assert auth.reload() is False
    # 文件被删：同样保留上一版
    path.unlink()
    assert auth.reload() is False
    assert auth.authenticate(second.token) is not None


def test_symlink_rotation_is_picked_up(tmp_path):
    """K8s Secret 轮换：挂载点的软链接换指向新文件。"""
    first = issue_principal_token(SHOP_A, AGENT, "agent")
    second = issue_principal_token(SHOP_A, "kf_03", "agent")
    v1, v2 = tmp_path / "v1.json", tmp_path / "v2.json"
    _write(v1, first)
    _write(v2, second)
    link = tmp_path / "rodit_tokens.json"
    link.symlink_to(v1)
    auth = TokenFileAuthenticator(link, reload_interval_s=0)
    assert auth.authenticate(first.token) is not None
    tmp = tmp_path / "next"
    tmp.symlink_to(v2)
    tmp.replace(link)
    assert auth.authenticate(second.token) is not None
    assert auth.authenticate(first.token) is None


@pytest.mark.parametrize(
    "content, reason",
    [
        (None, "不存在"),
        ("not json", "格式不对"),
        ('{"version": 2}', "格式不对"),
        (
            '{"principals": [{"sha256": "abc", "shop_id": "s", "user_id": "u", "role": "agent"}]}',
            "格式不对",
        ),
        ('{"principals": [], "extra": 1}', "格式不对"),
    ],
)
def test_bad_files_fail_at_startup(tmp_path, content, reason):
    path = tmp_path / "tokens.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(TokenFileError, match=reason) as exc:
        TokenFileAuthenticator(path)
    if content:
        assert content not in str(exc.value)  # 不回显文件内容


def test_invalid_principals_rejected(tmp_path):
    digest = hash_token(new_token())
    for entry in (
        {"sha256": digest, "shop_id": SHOP_A, "user_id": "system", "role": "agent"},
        {"sha256": digest, "shop_id": SHOP_A, "user_id": "kf", "role": "admin"},
        {"sha256": digest.upper(), "shop_id": SHOP_A, "user_id": "kf", "role": "agent"},
    ):
        path = tmp_path / "tokens.json"
        path.write_text(json.dumps({"principals": [entry]}), encoding="utf-8")
        with pytest.raises(TokenFileError):
            TokenFileAuthenticator(path)


def test_duplicate_hashes_rejected(tmp_path):
    agent = issue_principal_token(SHOP_A, AGENT, "agent")
    data = {
        "principals": [agent.entry.model_dump(mode="json")] * 2,
    }
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TokenFileError):
        TokenFileAuthenticator(path)


def test_append_entries_creates_and_extends(tmp_path):
    path = tmp_path / "deep" / "tokens.json"
    first = issue_principal_token(SHOP_A, AGENT, "agent")
    append_entries(path, [first.entry])
    metrics = issue_metrics_token("prometheus")
    data = append_entries(path, [metrics.entry])
    assert [p.sha256 for p in data.principals] == [first.entry.sha256]
    assert [m.name for m in data.metrics] == ["prometheus"]
    auth = TokenFileAuthenticator(path)
    assert auth.authenticate(first.token) is not None
    assert auth.authenticate_metrics(metrics.token)
    assert not list(path.parent.glob(".*.tmp"))  # 原子写入，不留临时文件
