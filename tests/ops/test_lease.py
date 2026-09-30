"""单副本租约（方案 §9.1 一个平台单元一个进程、§16 运维负担）：serve 启动时拿 PostgreSQL 会话级
advisory lock，拿不到（另一个进程占着）就拒绝启动（退出码 2），退出时释放。

不连数据库的部分用假租约测 serve 的流程；标 @pytest.mark.pg 的部分用临时空库测真的锁。
全部是合成数据，不联网（大模型地址是 .invalid，测试里不会调用）。
"""

from __future__ import annotations

import json
from typing import Any

import psycopg
import pytest
from ops_support import FakeReadinessDb, TestClient

from rodit.core.enums import Platform
from rodit.ops import serve
from rodit.ops.health import CheckResult
from rodit.ops.lease import (
    LEASE_NAMESPACE,
    LeaseError,
    LeaseHeld,
    ServeLease,
    lease_key,
)
from rodit.ops.serve import build_server, main
from rodit.ops.settings import ServerSettings

INT4 = range(-(2**31), 2**31)


# ================================================================ 键和消息（不连数据库）


def test_lease_key_is_per_platform_and_stable():
    keys = {p: lease_key(p) for p in Platform}
    assert len(set(keys.values())) == len(Platform)  # 三个平台单元互不影响
    for namespace, objid in keys.values():
        assert namespace == LEASE_NAMESPACE and namespace in INT4 and objid in INT4
    # 跨进程、跨版本稳定（不用 Python 的 hash，那个每个进程都不一样）
    assert lease_key("taobao") == lease_key(Platform.TAOBAO) == keys[Platform.TAOBAO]
    assert keys[Platform.TAOBAO] == (0x526F6469, 1475453331)
    assert keys[Platform.PDD] == (0x526F6469, -506701283)  # crc32 超过 int4 上限的折成负数


def test_lease_held_message_is_clear():
    err = LeaseHeld(Platform.DOUYIN, holder="rodit-serve/douyin/host-1/42")
    text = str(err)
    assert "douyin" in text and "rodit-serve/douyin/host-1/42" in text
    assert "只能" in text and "进程" in text  # 说清楚为什么不能再起一个
    assert isinstance(err, LeaseError) and isinstance(err, RuntimeError)
    assert "占用者" not in str(LeaseHeld(Platform.PDD, holder=None))


def test_lease_without_dsn_refuses_to_acquire(monkeypatch):
    monkeypatch.delenv("RODIT_DATABASE_DSN", raising=False)
    lease = ServeLease(Platform.TAOBAO)  # 构造时不读环境变量、不连库
    with pytest.raises(LeaseError, match="RODIT_DATABASE_DSN"):
        lease.acquire()
    assert not lease.held
    assert lease.probe() == {"lease": CheckResult(False, {"reason": "not_acquired"})}
    lease.release()  # 没拿到也可以调用


def test_lease_repr_hides_dsn():
    lease = ServeLease(Platform.TAOBAO, dsn="postgresql://u:secret-pw@db/rodit")
    assert "secret-pw" not in repr(lease) and "taobao" in repr(lease)


# ================================================================ serve 的流程（假租约）


class FakeLease:
    """记录 serve 对租约的调用；held_by 不为空时模拟"另一个进程占着"。"""

    instances: list[FakeLease] = []

    def __init__(self, settings: ServerSettings, *, held_by: str | None = None) -> None:
        self.platform = settings.platform
        self.held_by = held_by
        self.events: list[str] = []
        FakeLease.instances.append(self)

    def acquire(self) -> FakeLease:
        self.events.append("acquire")
        if self.held_by is not None:
            raise LeaseHeld(self.platform, holder=self.held_by)
        return self

    def release(self) -> None:
        self.events.append("release")

    def probe(self) -> dict[str, CheckResult]:
        return {"lease": CheckResult("acquire" in self.events and "release" not in self.events)}


@pytest.fixture
def fake_leases():
    FakeLease.instances = []
    return FakeLease.instances


@pytest.fixture
def mode_c(ops_env, token_file, monkeypatch):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")


def test_serve_acquires_before_run_and_releases_after(mode_c, fake_leases):
    seen: list[list[str]] = []

    def run(app, **kw):
        seen.append(list(fake_leases[0].events))  # 跑服务时已经拿到租约

    assert main(["serve"], run=run, lease_factory=FakeLease) == 0
    assert seen == [["acquire"]]
    assert fake_leases[0].events == ["acquire", "release"]
    assert fake_leases[0].platform == Platform.TAOBAO  # 按本单元的平台取键


def test_serve_releases_even_when_the_server_crashes(mode_c, fake_leases):
    def run(app, **kw):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        main(["serve"], run=run, lease_factory=FakeLease)
    assert fake_leases[0].events == ["acquire", "release"]


def test_serve_refuses_when_lease_is_held(mode_c, fake_leases, capsys):
    ran: list[Any] = []

    def held(settings):
        return FakeLease(settings, held_by="rodit-serve/taobao/old-host/7")

    assert main(["serve"], run=lambda *a, **k: ran.append(a), lease_factory=held) == 2
    assert not ran
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["level"] == "ERROR" and "启动失败" in record["msg"]
    assert "rodit-serve/taobao/old-host/7" in record["msg"] and "taobao" in record["msg"]


def test_startup_errors_come_before_touching_the_lease(ops_env, token_file, fake_leases):
    """配置不对（这里：没有平台绑定）先报出来，不去连数据库抢租约。"""
    assert main(["serve"], run=lambda *a, **k: None, lease_factory=FakeLease) == 2
    assert all(not lease.events for lease in fake_leases)


def test_database_down_while_acquiring_is_exit_1(mode_c, capsys):
    from rodit.core.errors import StorageUnavailable

    class Down(FakeLease):
        def acquire(self):
            raise StorageUnavailable("数据库连接失败")

    assert main(["serve"], run=lambda *a, **k: None, lease_factory=Down) == 1
    assert "数据库不可用" in capsys.readouterr().err


def test_migrate_on_start_runs_under_the_lease(mode_c, fake_leases, monkeypatch):
    monkeypatch.setenv("RODIT_MIGRATE_ON_START", "1")
    order: list[str] = []
    monkeypatch.setattr(serve, "migrate", lambda env: order.append("migrate") or [])
    monkeypatch.setattr(serve, "grant_readiness_access", lambda db: None)
    assert main(["serve"], run=lambda *a, **k: None, lease_factory=FakeLease) == 0
    assert fake_leases[0].events == ["acquire", "release"]
    assert order == ["migrate"]


def test_migration_failure_releases_the_lease(mode_c, fake_leases, monkeypatch):
    monkeypatch.setenv("RODIT_MIGRATE_ON_START", "1")

    def broken(env):
        raise psycopg.OperationalError("synthetic")

    monkeypatch.setattr(serve, "migrate", broken)
    assert main(["serve"], run=lambda *a, **k: None, lease_factory=FakeLease) == 1
    assert fake_leases[0].events == ["acquire", "release"]


def test_readiness_includes_the_lease(mode_c, fake_leases):
    lease = FakeLease(ServerSettings.from_env())
    server = build_server(ServerSettings.from_env(), database=FakeReadinessDb(), lease=lease)
    client = TestClient(server.app)
    body = client.get("/readyz").json()
    assert body["status"] == "not_ready" and body["checks"]["lease"] == {"ok": False}
    lease.acquire()
    server.readiness.check(fresh=True)
    body = client.get("/readyz").json()
    assert body["status"] == "ready" and body["checks"]["lease"] == {"ok": True}
    # 指标里也有这一项（告警 RoditNotReady 看 rodit_ready）
    metrics = server.metrics.render()
    assert 'rodit_readiness_check_ok{check="lease"} 1' in metrics
    assert "rodit_ready 1" in metrics


def test_default_lease_is_postgres_keyed_on_the_unit_platform(ops_env, monkeypatch):
    monkeypatch.setenv("RODIT_PLATFORM", "douyin")
    lease = serve._default_lease(ServerSettings.from_env())
    assert isinstance(lease, ServeLease) and not lease.held  # 构造时不连库
    assert lease.key == lease_key(Platform.DOUYIN)
    assert lease.application_name.startswith("rodit-serve/douyin/")


# ================================================================ 真的锁（PostgreSQL）


@pytest.mark.pg
def test_second_lease_is_refused_until_the_first_is_released(ops_pg_dsn):
    first = ServeLease(Platform.TAOBAO, dsn=ops_pg_dsn).acquire()
    try:
        assert first.held
        assert first.probe() == {"lease": CheckResult(True, {})}
        second = ServeLease(Platform.TAOBAO, dsn=ops_pg_dsn)
        with pytest.raises(LeaseHeld) as exc:
            second.acquire()
        assert "taobao" in str(exc.value) and "rodit-serve/taobao/" in str(exc.value)
        assert not second.held
        # 另一个平台单元的租约不受影响（同一个库里也可以各跑一个）
        with ServeLease(Platform.DOUYIN, dsn=ops_pg_dsn):
            pass
    finally:
        first.release()
    assert not first.held
    first.release()  # 幂等
    again = ServeLease(Platform.TAOBAO, dsn=ops_pg_dsn).acquire()
    again.release()


@pytest.mark.pg
def test_serve_exits_2_while_another_process_holds_the_lease(
    ops_env, token_file, ops_pg_dsn, monkeypatch, capsys
):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_DATABASE_DSN", ops_pg_dsn)
    other = ServeLease(Platform.TAOBAO, dsn=ops_pg_dsn).acquire()
    ran: list[Any] = []
    try:
        assert main(["serve"], run=lambda *a, **k: ran.append(a)) == serve.EXIT_STARTUP
    finally:
        other.release()
    assert not ran
    err = capsys.readouterr().err
    records = [json.loads(line) for line in err.strip().splitlines()]
    (record,) = [r for r in records if r["level"] == "ERROR"]
    assert "启动失败" in record["msg"] and "已有一个 serve 进程在运行" in record["msg"]
    assert "rodit-serve/taobao/" in record["msg"]
    assert "rodit_ops_" not in err  # 不带连接串（库名）

    # 占着的进程退出后，serve 能拿到租约；运行期间别人拿不到；退出后释放
    during: list[bool] = []

    def run(app, **kw):
        try:
            ServeLease(Platform.TAOBAO, dsn=ops_pg_dsn).acquire().release()
            during.append(True)
        except LeaseHeld:
            during.append(False)

    assert main(["serve"], run=run) == 0
    assert during == [False]
    ServeLease(Platform.TAOBAO, dsn=ops_pg_dsn).acquire().release()


def _terminate(admin_dsn: str, application_name_prefix: str) -> int:
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        rows = admin.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE application_name LIKE %s",
            (application_name_prefix + "%",),
        ).fetchall()
    return len(rows)


@pytest.mark.pg
def test_lost_connection_is_detected_and_reacquired(ops_pg_dsn, ops_pg_admin):
    """数据库重启 / 网络断开：锁随连接一起没了。就绪检查发现后重连再抢，抢到了照常就绪。"""
    lease = ServeLease(
        Platform.PDD, dsn=ops_pg_dsn, application_name="rodit-test-lease-a"
    ).acquire()
    try:
        assert _terminate(ops_pg_admin, "rodit-test-lease-a") == 1
        assert lease.probe() == {"lease": CheckResult(True, {"reacquired": True})}
        assert lease.held and lease.probe() == {"lease": CheckResult(True, {})}
    finally:
        lease.release()


@pytest.mark.pg
def test_lease_taken_by_another_process_after_a_drop_means_not_ready(ops_pg_dsn, ops_pg_admin):
    """连接断了之后锁被别的进程抢走：这边一直不就绪（摘流量），不会两个进程同时就绪。"""
    lease = ServeLease(
        Platform.PDD, dsn=ops_pg_dsn, application_name="rodit-test-lease-b"
    ).acquire()
    try:
        assert _terminate(ops_pg_admin, "rodit-test-lease-b") == 1
        with ServeLease(Platform.PDD, dsn=ops_pg_dsn):
            assert lease.probe() == {"lease": CheckResult(False, {"reason": "held_elsewhere"})}
            assert not lease.held
        # 那个进程退出了：下一次检查重新拿到
        assert lease.probe() == {"lease": CheckResult(True, {"reacquired": True})}
    finally:
        lease.release()
    assert lease.probe() == {"lease": CheckResult(False, {"reason": "released"})}
