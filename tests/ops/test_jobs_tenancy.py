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
from rodit.ops.lease import LeaseHeld
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


class FakeJobLease:
    """改数据的命令拿的单副本租约替身（不连数据库）；held_by 不为空时模拟 serve 或另一个任务
    占着。"""

    def __init__(self, command: str, *, held_by: str | None = None) -> None:
        self.command = command
        self.held_by = held_by
        self.events: list[str] = []

    def acquire(self) -> FakeJobLease:
        self.events.append("acquire")
        if self.held_by is not None:
            raise LeaseHeld(Platform.TAOBAO, holder=self.held_by)
        return self

    def release(self) -> None:
        self.events.append("release")

    def probe(self) -> dict:
        from rodit.ops.health import CheckResult

        return {"lease": CheckResult(self.events[-1:] == ["acquire"])}


@pytest.fixture
def job_leases(monkeypatch):
    """替换 tenancy.job_lease：返回（造出来的租约列表, {"held_by": 占用者}）。"""
    made: list[FakeJobLease] = []
    state: dict[str, str | None] = {"held_by": None}

    def factory(environ, command):
        made.append(FakeJobLease(command, held_by=state["held_by"]))
        return made[-1]

    monkeypatch.setattr(tenancy_jobs, "job_lease", factory)
    return made, state


@pytest.fixture
def cli(monkeypatch, job_leases):
    """命令行跑在演示单元上（带假的续期接口）；返回应用服务和 build_service 的调用次数。
    单副本租约换成替身（job_leases）。"""
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


def test_onboard_shop_does_not_stamp_local_time_as_the_platform_grant_time(cli, monkeypatch):
    """复现（修复前）：onboard-shop 把本机时间 N 记成这份授权的平台开通时间。商家在 N 之前
    （停服务窗口里）取消订购，推送在 serve 重启之后才到：按"旧事件"忽略，数据留到过期清理
    （约 60 天）。现在不给 --granted-at 时不记开通时间，这条取消照常删除。"""
    service, _ = cli
    unit = service.unit
    _stdin(monkeypatch, f"{TOKEN}\n")
    assert serve.main([*ONBOARD, "--token-stdin"]) == 0
    assert unit.repos.auth.stamp(NEW).granted_at is None
    cancel = parse_subscription(
        subscription_push("cancelled", NEW, at=unit.clock.now() - timedelta(minutes=10))
    )
    assert service.tenancy.handle_subscription(cancel).action == "revoked"
    assert unit.repos.auth.get(NEW) is None and unit.repos.knowledge.get_shop(NEW) is None


def test_onboard_shop_records_the_given_platform_grant_time(cli, monkeypatch, caplog):
    """--granted-at（服务市场订单上的开通时间）记进授权：比它早的订购推送才算旧事件；
    晚于现在的不收（退出码 2，什么都不写）。"""
    service, _ = cli
    unit = service.unit
    p0 = unit.clock.now() - timedelta(hours=2)
    _stdin(monkeypatch, f"{TOKEN}\n")
    assert serve.main([*ONBOARD, "--token-stdin", "--granted-at", p0.isoformat()]) == 0
    assert unit.repos.auth.stamp(NEW).granted_at == p0
    [granted] = unit.repos.audit.list_events(NEW, action=AuditAction.AUTH_GRANTED)
    assert granted.detail["event_at"] == p0.isoformat()
    stale = parse_subscription(subscription_push("cancelled", NEW, at=p0 - timedelta(seconds=1)))
    assert service.tenancy.handle_subscription(stale).action == "ignored"
    assert unit.auth.is_authorized(NEW)

    future = (unit.clock.now() + timedelta(hours=1)).isoformat()
    _stdin(monkeypatch, f"{TOKEN}\n")
    argv = [*ONBOARD, "--token-stdin", "--granted-at", future]
    argv[argv.index(NEW)] = "s_new_future"
    with caplog.at_level(logging.ERROR):
        assert serve.main(argv) == 2
    assert "--granted-at" in caplog.text and TOKEN not in caplog.text
    assert unit.repos.auth.get("s_new_future") is None


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


def test_purge_expired_finishes_what_an_earlier_run_left(cli, capsys, caplog):
    """复现（修复前）：清理删了令牌之后出错，下一次 purge-expired 再也列不出这家店（令牌没了），
    数据一直留着，只能靠日志里的店铺 ID 手动 revoke。现在下一次接着删完（开始清理的标记）。"""
    service, _ = cli
    unit = service.unit
    _expire(service, SHOP_A, 45)

    class FailsOnce:
        failed = False

        def purge_shop(self, shop_id: str) -> int:
            if not self.failed:
                self.failed = True
                raise RuntimeError("模拟清理失败")
            return 0

    unit.extra_purgers.append(FailsOnce())
    with caplog.at_level(logging.ERROR):
        assert serve.main(["purge-expired"]) == 1
    assert "没有删完的店铺：1 家" in capsys.readouterr().out
    assert unit.repos.auth.get(SHOP_A) is None

    assert serve.main(["purge-expired", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "待清理的店铺：0 家" in out and "上次没删完、要接着删的店铺：1 家" in out
    assert SHOP_A not in out
    assert unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED) == []

    assert serve.main(["purge-expired"]) == 0
    assert "已清理授权过期超过 30 天的店铺：1 家" in capsys.readouterr().out
    [record] = unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.detail["shop_id"] == SHOP_A and record.detail["reason"] == "expired"
    assert serve.main(["purge-expired", "--dry-run"]) == 0
    assert "上次没删完" not in capsys.readouterr().out


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


# ================================================================ 单副本租约：不和 serve 同时改数据


SERVE_HOLDER = "rodit-serve/taobao/host-1/42，自 2026-09-30 08:00:00 起"
MUTATING = {
    "onboard-shop": [*ONBOARD, "--token-stdin"],
    "refresh-tokens": ["refresh-tokens"],
    "purge-expired": ["purge-expired"],
    "revoke": ["revoke", "--shop-id", SHOP_A],
    "set-status": ["set-status", "--shop-id", SHOP_A, "--status", "suspended"],
}
ALTERNATIVE = {
    "onboard-shop": "POST /v1/platform/{access}/subscription",
    "refresh-tokens": "RODIT_TOKEN_REFRESH_INTERVAL_S",
    "purge-expired": "RODIT_PURGE_EXPIRED_INTERVAL_S",
    "revoke": "POST /v1/shops/{shop_id}/revoke",
    "set-status": "停 serve",
}


@pytest.mark.parametrize("command", sorted(MUTATING))
def test_mutating_commands_refuse_while_serve_holds_the_lease(
    cli, job_leases, monkeypatch, caplog, command
):
    """取消订购的闸门、缓存都在 serve 进程内：serve 在跑时改数据的命令一律不执行（退出码 2），
    日志写明改用哪个接口或 serve 里的哪个定时任务，什么都不改、不组装单元。"""
    service, built = cli
    leases, state = job_leases
    state["held_by"] = SERVE_HOLDER
    unit = service.unit
    _grant(service, SHOP_A, hours=3)
    _expire(service, SHOP_B, 45)
    before = (unit.repos.auth.get(SHOP_A), unit.repos.auth.get(SHOP_B))
    status = unit.repos.knowledge.get_shop(SHOP_A).status
    _stdin(monkeypatch, f"{TOKEN}\n")
    with caplog.at_level(logging.ERROR):
        assert serve.main(MUTATING[command]) == 2
    assert built == []  # 没组装单元
    assert [lease.command for lease in leases] == [command]
    assert leases[0].events == ["acquire"]  # 没拿到，不用放
    assert "serve 正在运行" in caplog.text and "rodit-serve/taobao/host-1/42" in caplog.text
    assert ALTERNATIVE[command] in caplog.text
    assert TOKEN not in caplog.text
    # 什么都没改
    assert (unit.repos.auth.get(SHOP_A), unit.repos.auth.get(SHOP_B)) == before
    assert unit.repos.knowledge.get_shop(SHOP_A).status == status
    assert unit.repos.knowledge.get_shop(NEW) is None
    assert unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED) == []


@pytest.mark.parametrize("command", sorted(MUTATING))
def test_mutating_commands_hold_the_lease_while_they_run(cli, job_leases, monkeypatch, command):
    service, built = cli
    leases, _ = job_leases
    _grant(service, SHOP_A, hours=3)
    seen: list[list[str]] = []
    real = tenancy_jobs.build_service

    def build(environ):
        seen.append(list(leases[0].events))
        return real(environ)

    monkeypatch.setattr(tenancy_jobs, "build_service", build)
    _stdin(monkeypatch, f"{TOKEN}\n")
    assert serve.main(MUTATING[command]) == 0
    assert seen == [["acquire"]]  # 组装单元、改数据时已经拿着租约
    assert leases[0].events == ["acquire", "release"]
    assert leases[0].command == command


def test_lease_is_released_when_the_job_fails(cli, job_leases, caplog):
    leases, _ = job_leases
    with caplog.at_level(logging.ERROR):
        assert serve.main(["set-status", "--shop-id", "s_nope", "--status", "active"]) == 2
    assert leases[0].events == ["acquire", "release"]


def test_dry_run_is_read_only_and_can_run_next_to_serve(cli, job_leases, capsys):
    service, built = cli
    leases, state = job_leases
    state["held_by"] = SERVE_HOLDER
    _expire(service, SHOP_A, 45)
    assert serve.main(["purge-expired", "--dry-run"]) == 0
    assert "待清理的店铺：1 家" in capsys.readouterr().out
    assert leases == [] and len(built) == 1
    assert service.unit.repos.auth.get(SHOP_A) is not None


def test_another_job_holding_the_lease_is_named(cli, job_leases, caplog):
    service, built = cli
    leases, state = job_leases
    state["held_by"] = "rodit-job/taobao/purge-expired/host-2/7"
    with caplog.at_level(logging.ERROR):
        assert serve.main(["revoke", "--shop-id", SHOP_A]) == 2
    assert "另一个运维任务" in caplog.text and "rodit-job/taobao/purge-expired" in caplog.text
    assert built == [] and service.unit.repos.auth.get(SHOP_A) is not None


def test_revoke_output_reminds_to_remove_the_token_file_entries(cli, capsys):
    assert serve.main(["revoke", "--shop-id", SHOP_A]) == 0
    assert "令牌文件" in capsys.readouterr().out


def test_job_lease_uses_the_unit_platform_and_names_the_job(monkeypatch):
    from rodit.ops.lease import ServeLease, lease_key

    lease = tenancy_jobs.job_lease(
        {"RODIT_PLATFORM": "douyin", "RODIT_DATABASE_DSN": "dbname=never_connected"}, "revoke"
    )
    assert isinstance(lease, ServeLease) and not lease.held  # 构造时不连库
    assert lease.key == lease_key(Platform.DOUYIN)  # 和 serve 同一个键
    assert lease.application_name.startswith("rodit-job/douyin/revoke/")
    assert "never_connected" not in repr(lease)


# ================================================================ 参数：--expires-days


@pytest.mark.parametrize("days", ["inf", "-inf", "nan", "1e12", "4000000", "3650.5", "-1", "0"])
def test_bad_expires_days_is_a_usage_error_before_reading_the_token(
    cli, job_leases, monkeypatch, caplog, days
):
    """inf、1e12 换成到期时间会溢出（原来是没有说明的崩溃、退出码 1）：一律退出码 2，
    令牌还没读、单元还没组装、租约还没拿。"""
    service, built = cli
    leases, _ = job_leases
    stdin = io.StringIO(f"{TOKEN}\n")
    monkeypatch.setattr(sys, "stdin", stdin)
    with caplog.at_level(logging.ERROR):
        assert serve.main([*ONBOARD, "--token-stdin", f"--expires-days={days}"]) == 2
    assert "--expires-days" in caplog.text and TOKEN not in caplog.text
    assert stdin.tell() == 0  # 没读令牌
    assert built == [] and leases == []
    assert service.unit.repos.auth.get(NEW) is None


def test_expires_days_upper_bound_is_accepted(cli, monkeypatch):
    service, _ = cli
    _stdin(monkeypatch, f"{TOKEN}\n")
    assert serve.main([*ONBOARD, "--token-stdin", "--expires-days", "3650"]) == 0
    auth = service.unit.repos.auth.get(NEW)
    assert auth.expires_at == service.unit.clock.now() + timedelta(days=3650)


def test_overflow_inside_a_job_is_a_usage_error_not_a_traceback(cli, monkeypatch, caplog):
    """兜底：任务里别的地方溢出（OverflowError 不是 ValueError）也按参数不对处理。"""
    service, _ = cli

    def overflow(*args, **kwargs):
        raise OverflowError("date value out of range")

    monkeypatch.setattr(service.tenancy, "set_status", overflow)
    with caplog.at_level(logging.ERROR):
        assert serve.main(["set-status", "--shop-id", SHOP_A, "--status", "active"]) == 2
    assert "out of range" in caplog.text


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
def test_cli_refuses_to_purge_while_serve_holds_the_lease_on_postgres(pg_cli, monkeypatch):
    """复现（修复前）：serve 在跑时 python -m rodit.ops revoke 另起进程直接删数据，serve 手上的
    消息之后照样写进推荐和审计，没有人再删。现在命令先拿单副本租约：serve 占着就退出码 2，
    什么都不删；serve 退出之后照常执行。"""
    from rodit.ops.lease import ServeLease

    dsn, env = pg_cli
    done = _run(env, *ONBOARD, "--token-stdin", stdin=f"{TOKEN}\n")
    assert done.returncode == 0, done.stderr
    knowledge, auth_repo, audit = _pg_repos(dsn, monkeypatch)
    serve_lease = ServeLease(
        Platform.TAOBAO, dsn=dsn, application_name="rodit-serve/taobao/test-host/1"
    ).acquire()
    try:
        done = _run(env, "revoke", "--shop-id", NEW)
        assert done.returncode == 2, done.stdout + done.stderr
        assert "serve 正在运行" in done.stderr and "rodit-serve/taobao/test-host/1" in done.stderr
        assert "POST /v1/platform/{access}/subscription" in done.stderr
        assert "rodit_ops_" not in done.stderr  # 不带连接串（库名）
        assert knowledge.get_shop(NEW) is not None and auth_repo.get(NEW) is not None
        assert audit.list_system_events(action=AuditAction.SHOP_PURGED) == []
        # 只读的 --dry-run 可以和 serve 同时跑
        done = _run(env, "purge-expired", "--dry-run")
        assert done.returncode == 0, done.stderr
    finally:
        serve_lease.release()
    done = _run(env, "revoke", "--shop-id", NEW)
    assert done.returncode == 0, done.stderr
    assert knowledge.get_shop(NEW) is None and auth_repo.get(NEW) is None
    # 任务做完释放了租约：serve 能拿到
    ServeLease(Platform.TAOBAO, dsn=dsn).acquire().release()


@pytest.mark.pg
def test_serve_refuses_to_start_while_a_job_holds_the_lease_on_postgres(
    pg_cli, monkeypatch, capsys
):
    """反过来：任务在跑时 serve 起不来（退出码 2），日志说明是运维任务占着、做完再启动。"""
    import json

    from rodit.ops.lease import job_lease

    dsn, env = pg_cli
    for name, value in env.items():
        if name.startswith("RODIT_"):
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("RODIT_LOG_FORMAT", "json")
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_METRICS_PUBLIC", "1")
    Path(env["RODIT_TOKENS_FILE"]).write_text('{"principals": [], "metrics": []}', "utf-8")
    job = job_lease(Platform.TAOBAO, "purge-expired", environ=env).acquire()
    ran: list[object] = []
    try:
        assert serve.main(["serve"], run=lambda *a, **k: ran.append(a)) == serve.EXIT_STARTUP
    finally:
        job.release()
    assert not ran
    records = [json.loads(line) for line in capsys.readouterr().err.strip().splitlines()]
    (record,) = [r for r in records if r["level"] == "ERROR"]
    assert "运维任务" in record["msg"] and "rodit-job/taobao/purge-expired/" in record["msg"]


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
