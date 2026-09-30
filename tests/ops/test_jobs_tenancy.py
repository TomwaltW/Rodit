"""店铺生命周期的离线任务（python -m rodit.ops onboard-shop / refresh-tokens / purge-expired /
revoke / set-status，方案 §8.4、§9.1），以及 serve 组装好后平台代码经 BindingContext.on_subscription
推订购事件。

- 命令行（内存单元）：build_service 换成演示单元，不连数据库；令牌只从标准输入 / 文件读；
- @pytest.mark.pg：真的起子进程跑 python -m rodit.ops，连临时库（本地试用 + 离线替身 + 合成
  令牌密钥），核对加密落库、续期、过期清理和清理之后留下的系统事件。
全部是合成数据和合成令牌。
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from ops_support import FakeReadinessDb
from pydantic import SecretStr

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit
from rodit.app.unit import in_memory_repos
from rodit.core.audit import AuditAction
from rodit.core.enums import Platform, ShopStatus
from rodit.core.models import ShopAuth
from rodit.core.scopes import MVP_SCOPES
from rodit.ops import serve
from rodit.ops.bindings import BindingContext, PlatformBindings, demo_bindings
from rodit.ops.jobs import tenancy as tenancy_jobs
from rodit.ops.serve import build_server
from rodit.ops.settings import ServerSettings
from rodit.storage.db import MIGRATIONS
from rodit.testing.fakes import FakeClock, FakeTokenRefresher, parse_subscription, subscription_push
from rodit.testing.fixtures import SHOP_A, SHOP_B, build_demo_world

NEW = "s_new"
TOKEN = "tok-cli-access-synthetic-6d2e"
REFRESH = "tok-cli-refresh-synthetic-8a4f"
SCOPES = "order.read,logistics.read,refund.read,item.read"
ONBOARD = [
    "onboard-shop",
    "--shop-id",
    NEW,
    "--platform",
    "taobao",
    "--category",
    "apparel",
    "--scopes",
    SCOPES,
]


# ================================================================ 命令行（内存单元）


@pytest.fixture
def cli(monkeypatch):
    """命令行跑在演示单元上（带假的续期接口）；返回应用服务和 build_service 的调用次数。"""
    unit = build_demo_unit(Platform.TAOBAO)
    unit.token_refresher = FakeTokenRefresher(Platform.TAOBAO, clock=unit.clock)
    service = AssistService(unit)
    built: list[dict] = []

    def build(environ):
        built.append(dict(environ))
        return service

    monkeypatch.setattr(tenancy_jobs, "build_service", build)
    return service, built


def _stdin(monkeypatch, text: str) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))


def test_commands_are_registered():
    commands = set(serve._parser()._subparsers._group_actions[0].choices)  # noqa: SLF001
    assert tenancy_jobs.COMMANDS <= commands


def test_onboard_shop_reads_the_token_from_stdin(cli, monkeypatch, capsys, caplog):
    service, built = cli
    _stdin(monkeypatch, f"{TOKEN}\n")
    with caplog.at_level(logging.DEBUG):
        code = serve.main([*ONBOARD, "--token-stdin"])
    assert code == 0 and len(built) == 1
    out, err = capsys.readouterr()
    assert f"已开通店铺 {NEW}" in out and "trial" in out and "续期令牌：无" in out
    for text in (out, err, caplog.text):
        assert TOKEN not in text
    unit = service.unit
    shop = unit.repos.knowledge.get_shop(NEW)
    assert shop.status == ShopStatus.TRIAL and shop.category == "apparel"
    auth = unit.repos.auth.get(NEW)
    assert auth.access_token.get_secret_value() == TOKEN
    assert set(auth.scopes) == MVP_SCOPES and auth.refresh_token is None
    assert auth.expires_at == unit.clock.now() + timedelta(days=30)
    [granted] = unit.repos.audit.list_events(NEW, action=AuditAction.AUTH_GRANTED)
    assert granted.actor == "ops"


def test_onboard_shop_reads_access_and_refresh_token_from_a_file(cli, tmp_path, capsys):
    service, _ = cli
    token_file = tmp_path / "grant.txt"
    token_file.write_text(f"{TOKEN}\n{REFRESH}\n", encoding="utf-8")
    code = serve.main(
        [
            *ONBOARD,
            "--token-file",
            str(token_file),
            "--expires-days",
            "2",
            "--refresh-expires-at",
            "2027-01-01T00:00+08:00",
            "--status",
            "active",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "续期令牌：有" in out and TOKEN not in out and REFRESH not in out
    auth = service.unit.repos.auth.get(NEW)
    assert auth.refresh_token.get_secret_value() == REFRESH
    assert auth.expires_at == service.unit.clock.now() + timedelta(days=2)
    assert auth.refresh_expires_at.isoformat() == "2027-01-01T00:00:00+08:00"
    assert service.unit.repos.knowledge.get_shop(NEW).status == ShopStatus.ACTIVE


def test_onboard_shop_rejects_refund_write_in_phase_one(cli, monkeypatch, caplog):
    service, _ = cli
    _stdin(monkeypatch, TOKEN)
    with caplog.at_level(logging.ERROR):
        code = serve.main([*ONBOARD[:-1], SCOPES + ",refund.write", "--token-stdin"])
    assert code == 2
    assert "refund.write" in caplog.text and TOKEN not in caplog.text
    assert service.unit.repos.knowledge.get_shop(NEW) is None
    assert service.unit.repos.auth.get(NEW) is None


def test_onboard_shop_rejects_another_platform(cli, monkeypatch, caplog):
    service, _ = cli
    _stdin(monkeypatch, TOKEN)
    argv = [*ONBOARD, "--token-stdin"]
    argv[argv.index("taobao")] = "douyin"
    with caplog.at_level(logging.ERROR):
        assert serve.main(argv) == 2
    assert "douyin" in caplog.text
    assert service.unit.repos.knowledge.get_shop(NEW) is None


@pytest.mark.parametrize(
    "text",
    ["", "\n\n", "a b\n", "one\ntwo\nthree\n", "x" * (tenancy_jobs.MAX_TOKEN_INPUT_BYTES + 5)],
)
def test_bad_token_input_is_refused_before_touching_the_unit(cli, monkeypatch, caplog, text):
    service, built = cli
    _stdin(monkeypatch, text)
    with caplog.at_level(logging.ERROR):
        assert serve.main([*ONBOARD, "--token-stdin"]) == 2
    assert built == []  # 没组装单元、没连数据库
    assert service.unit.repos.knowledge.get_shop(NEW) is None
    if text.strip():
        assert text.strip()[:20] not in caplog.text  # 不回显输入


def test_missing_token_file_is_refused(cli, tmp_path):
    _, built = cli
    assert serve.main([*ONBOARD, "--token-file", str(tmp_path / "nope.txt")]) == 2
    assert built == []


def test_token_is_never_a_command_line_argument(cli):
    for argv in (
        [*ONBOARD],  # 没说令牌从哪来
        [*ONBOARD, "--token-stdin", "--token-file", "x"],  # 只能选一个
        [*ONBOARD, "--token", TOKEN],  # 没有这个参数
    ):
        with pytest.raises(SystemExit) as err:
            serve.main(argv)
        assert err.value.code == 2


def test_onboard_shop_rejects_past_expiry(cli, monkeypatch):
    service, _ = cli
    _stdin(monkeypatch, TOKEN)
    assert serve.main([*ONBOARD, "--token-stdin", "--expires-at", "2020-01-01"]) == 2
    _stdin(monkeypatch, TOKEN)
    assert serve.main([*ONBOARD, "--token-stdin", "--expires-days", "0"]) == 2
    assert service.unit.repos.auth.get(NEW) is None


def _grant(service: AssistService, shop_id: str, *, hours: float, refresh: bool = True) -> None:
    unit = service.unit
    unit.repos.auth.save(
        ShopAuth(
            shop_id=shop_id,
            access_token=SecretStr(f"tok-cli-old-{shop_id}"),
            scopes=sorted(MVP_SCOPES),
            expires_at=unit.clock.now() + timedelta(hours=hours),
            refresh_token=SecretStr(f"tok-cli-old-refresh-{shop_id}") if refresh else None,
        )
    )


def test_refresh_tokens_prints_counts_only(cli, capsys):
    service, _ = cli
    _grant(service, SHOP_A, hours=3)
    _grant(service, SHOP_B, hours=4, refresh=False)
    assert serve.main(["refresh-tokens"]) == 0
    out = capsys.readouterr().out
    assert "检查 2 家店铺" in out and "续期成功 1" in out and "没有续期令牌 1" in out
    assert SHOP_A not in out and SHOP_B not in out and "tok-" not in out
    assert service.unit.repos.auth.get(SHOP_A).refreshed_at == service.unit.clock.now()


def test_refresh_tokens_exit_codes(cli, capsys):
    service, _ = cli
    _grant(service, SHOP_A, hours=3)
    service.unit.token_refresher.script(SHOP_A, RuntimeError("平台云代码出错（模拟）"))
    assert serve.main(["refresh-tokens"]) == 1  # 意外错误：要人看
    assert "出错 1" in capsys.readouterr().out
    service.unit.token_refresher = None
    assert serve.main(["refresh-tokens"]) == 2  # 没接续期接口


def _expire(service: AssistService, shop_id: str, days_ago: float) -> None:
    auth = service.unit.repos.auth.get(shop_id)
    expires = service.unit.clock.now() - timedelta(days=days_ago)
    service.unit.repos.auth.save(auth.model_copy(update={"expires_at": expires}))


def test_purge_expired_dry_run_then_purge(cli, capsys):
    service, _ = cli
    unit = service.unit
    _expire(service, SHOP_A, 45)
    _expire(service, SHOP_B, 10)
    assert serve.main(["purge-expired", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "待清理的店铺：1 家" in out and "没有删除" in out and SHOP_A not in out
    assert unit.repos.knowledge.get_shop(SHOP_A) is not None

    assert serve.main(["purge-expired"]) == 0
    out = capsys.readouterr().out
    assert "已清理授权过期超过 30 天的店铺：1 家" in out and SHOP_A not in out
    assert unit.repos.knowledge.get_shop(SHOP_A) is None
    assert unit.repos.knowledge.get_shop(SHOP_B) is not None
    [record] = unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.detail["shop_id"] == SHOP_A and record.detail["reason"] == "expired"


def test_purge_expired_failure_exits_1_and_logs_the_shop(cli, capsys, caplog):
    service, _ = cli
    _expire(service, SHOP_A, 45)

    class Broken:
        def purge_shop(self, shop_id: str) -> int:
            raise RuntimeError("模拟清理失败")

    service.unit.extra_purgers.append(Broken())
    with caplog.at_level(logging.ERROR):
        assert serve.main(["purge-expired"]) == 1
    assert "没有删完的店铺：1 家" in capsys.readouterr().out
    assert SHOP_A in caplog.text  # 日志里有要重试的店铺


def test_revoke_command_leaves_a_system_record(cli, capsys):
    service, _ = cli
    assert serve.main(["revoke", "--shop-id", SHOP_A, "--reason", "merchant_request"]) == 0
    out = capsys.readouterr().out
    assert f"已取消订购店铺 {SHOP_A}" in out and "令牌 1 条" in out
    assert service.unit.repos.auth.get(SHOP_A) is None
    [record] = service.unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.actor == "ops" and record.detail["reason"] == "merchant_request"
    assert serve.main(["revoke", "--shop-id", SHOP_B]) == 0
    records = service.unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert records[-1].detail["reason"] == "ops"


def test_set_status_command(cli, capsys):
    service, _ = cli
    assert serve.main(["set-status", "--shop-id", SHOP_A, "--status", "suspended"]) == 0
    assert service.unit.repos.knowledge.get_shop(SHOP_A).status == ShopStatus.SUSPENDED
    assert "suspended" in capsys.readouterr().out
    assert serve.main(["set-status", "--shop-id", "s_nope", "--status", "active"]) == 2


def test_set_status_rejects_bad_transition_and_bad_value(cli):
    service, _ = cli
    service.tenancy.set_status(SHOP_A, ShopStatus.ACTIVE, actor="ops")
    assert serve.main(["set-status", "--shop-id", SHOP_A, "--status", "trial"]) == 2
    with pytest.raises(SystemExit):
        serve.main(["set-status", "--shop-id", SHOP_A, "--status", "bogus"])


def test_job_database_errors_map_to_exit_1(cli, monkeypatch, caplog):
    from rodit.core.errors import StorageUnavailable

    def down(environ):
        raise StorageUnavailable("连不上 host=db.internal user=rodit（模拟）")

    monkeypatch.setattr(tenancy_jobs, "build_service", down)
    with caplog.at_level(logging.ERROR):
        assert serve.main(["refresh-tokens"]) == 1
    assert "db.internal" not in caplog.text


# ================================================================ serve：平台代码推订购事件


@pytest.fixture
def local_env(ops_env, token_file, monkeypatch):
    for name in ("RODIT_LLM_BASE_URL", "RODIT_LLM_MODEL", "RODIT_LLM_API_KEY"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("RODIT_ENV", "local")
    monkeypatch.setenv("RODIT_AUTH_EXPIRY_CHECK_S", "0")


def test_cancel_pushed_through_the_binding_context_revokes_in_process(local_env):
    """平台云代码在进程内拿到订购推送：ctx.on_subscription(event) → 店铺生命周期取消订购。"""
    captured: list[BindingContext] = []
    world = build_demo_world()
    clock = FakeClock(world.now)

    def factory(ctx: BindingContext) -> PlatformBindings:
        captured.append(ctx)
        return demo_bindings(ctx)

    server = build_server(
        ServerSettings.from_env(),
        bindings_factory=factory,
        clock=clock,
        repos=in_memory_repos(clock),
        database=FakeReadinessDb(tuple(m.version for m in MIGRATIONS)),
    )
    [ctx] = captured
    unit = server.unit
    tenancy = server.service.tenancy
    grant = parse_subscription(
        subscription_push("subscribed", SHOP_A, at=clock.now(), access_token=TOKEN)
    )
    tenancy.onboard(grant)
    assert unit.auth.is_authorized(SHOP_A)

    clock.advance(minutes=5)  # 取消晚于开通
    event = unit.parse_subscription(subscription_push("cancelled", SHOP_A, at=clock.now()))
    ctx.on_subscription(event)
    assert unit.repos.auth.get(SHOP_A) is None
    assert unit.repos.knowledge.get_shop(SHOP_A) is None
    [record] = unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.detail["shop_id"] == SHOP_A and record.detail["reason"] == "cancelled"
    assert record.actor == "platform"


# ================================================================ PostgreSQL：真的跑命令行


TOKEN_KEY = "rodit-cli-token-key-synthetic-only"


@pytest.fixture
def pg_cli(ops_pg_dsn, tmp_path):
    """迁移好的临时库 + 子进程的环境变量（本地试用、离线替身、合成密钥）。"""
    from rodit.storage import Database

    Database(ops_pg_dsn).migrate()
    env = {k: v for k, v in os.environ.items() if not k.startswith("RODIT_")}
    env.update(
        {
            "RODIT_ENV": "local",
            "RODIT_PLATFORM": "taobao",
            "RODIT_DATABASE_DSN": ops_pg_dsn,
            "RODIT_TOKEN_KEY": TOKEN_KEY,
            "RODIT_BUYER_SALT": "salt-cli-synthetic-only",
            "RODIT_TOKENS_FILE": str(tmp_path / "tokens.json"),
            "RODIT_LOG_FORMAT": "text",
        }
    )
    return ops_pg_dsn, env


def _run(env: dict[str, str], *args: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rodit.ops", *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        cwd=Path(__file__).resolve().parents[2],
    )


def _pg_repos(dsn: str, monkeypatch):
    from rodit.core.clock import SystemClock
    from rodit.storage import Database, PgAuditLog, PgKnowledgeRepository, PgShopAuthRepository

    db = Database(dsn)
    monkeypatch.setenv("RODIT_TOKEN_KEY", TOKEN_KEY)
    return (
        PgKnowledgeRepository(db, clock=SystemClock()),
        PgShopAuthRepository.from_env(db, env_name="RODIT_TOKEN_KEY"),
        PgAuditLog(db),
    )


@pytest.mark.pg
def test_cli_onboard_refresh_and_purge_on_postgres(pg_cli, monkeypatch):
    import psycopg

    dsn, env = pg_cli
    argv = [
        "onboard-shop",
        "--shop-id",
        NEW,
        "--platform",
        "taobao",
        "--category",
        "apparel",
        "--scopes",
        SCOPES,
        "--token-stdin",
    ]
    assert TOKEN not in " ".join(argv)
    done = _run(env, *argv, stdin=f"{TOKEN}\n{REFRESH}\n")
    assert done.returncode == 0, done.stderr
    assert f"已开通店铺 {NEW}" in done.stdout
    for text in (done.stdout, done.stderr):
        assert TOKEN not in text and REFRESH not in text

    knowledge, auth_repo, audit = _pg_repos(dsn, monkeypatch)
    shop = knowledge.get_shop(NEW)
    assert shop is not None and shop.status == ShopStatus.TRIAL and shop.category == "apparel"
    auth = auth_repo.get(NEW)
    assert auth.access_token.get_secret_value() == TOKEN
    assert auth.refresh_token.get_secret_value() == REFRESH
    assert set(auth.scopes) == MVP_SCOPES
    # 库里是密文：两个令牌的原文都不在表里
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "SELECT access_token, refresh_token FROM shop_auth WHERE shop_id = %s", (NEW,)
        ).fetchone()
    assert TOKEN.encode() not in bytes(row[0]) and REFRESH.encode() not in bytes(row[1])
    [granted] = audit.list_events(NEW, action=AuditAction.AUTH_GRANTED)
    assert granted.actor == "ops" and TOKEN not in str(granted.detail)

    # 快到期了 → refresh-tokens 续期（本地试用用合成的续期接口），只打印条数
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "UPDATE shop_auth SET expires_at = now() + interval '2 hours' WHERE shop_id = %s",
            (NEW,),
        )
    done = _run(env, "refresh-tokens")
    assert done.returncode == 0, done.stderr
    assert "检查 1 家店铺" in done.stdout and "续期成功 1" in done.stdout
    assert NEW not in done.stdout and "tok-" not in done.stdout + done.stderr
    refreshed = auth_repo.get(NEW)
    assert refreshed.refreshed_at is not None
    assert refreshed.expires_at > auth.expires_at - timedelta(days=1)
    assert refreshed.access_token.get_secret_value() != TOKEN
    assert audit.list_events(NEW, action=AuditAction.AUTH_REFRESHED)

    # 过期 40 天 → purge-expired --dry-run 只数不删；purge-expired 删掉，系统事件留下来
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "UPDATE shop_auth SET expires_at = now() - interval '40 days' WHERE shop_id = %s",
            (NEW,),
        )
    done = _run(env, "purge-expired", "--dry-run")
    assert done.returncode == 0, done.stderr
    assert "待清理的店铺：1 家" in done.stdout and NEW not in done.stdout
    assert knowledge.get_shop(NEW) is not None and auth_repo.get(NEW) is not None

    done = _run(env, "purge-expired")
    assert done.returncode == 0, done.stderr
    assert "已清理授权过期超过 30 天的店铺：1 家" in done.stdout
    assert knowledge.get_shop(NEW) is None and auth_repo.get(NEW) is None
    assert audit.list_events(NEW) == []
    [record] = audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.shop_id is None and record.actor == "system"
    assert record.detail["shop_id"] == NEW and record.detail["reason"] == "expired"
    assert set(record.detail) == {"shop_id", "reason", "counts", "event_at"}
    assert record.detail["counts"]["auth"] == 1


@pytest.mark.pg
def test_cli_onboard_rejections_write_nothing_on_postgres(pg_cli, monkeypatch):
    dsn, env = pg_cli
    base = ["onboard-shop", "--shop-id", NEW, "--platform", "taobao", "--token-stdin"]
    done = _run(env, *base, "--scopes", SCOPES + ",refund.write", stdin=TOKEN)
    assert done.returncode == 2 and "refund.write" in done.stderr
    assert TOKEN not in done.stdout + done.stderr
    done = _run(
        env,
        "onboard-shop",
        "--shop-id",
        NEW,
        "--platform",
        "pdd",
        "--scopes",
        SCOPES,
        "--token-stdin",
        stdin=TOKEN,
    )
    assert done.returncode == 2 and "pdd" in done.stderr
    knowledge, auth_repo, _ = _pg_repos(dsn, monkeypatch)
    assert knowledge.get_shop(NEW) is None and auth_repo.get(NEW) is None
