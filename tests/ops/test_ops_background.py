"""serve 的后台任务（rodit.ops.background，方案 §8.4 续期 / 过期清理、§12.2 就绪、§12.4 告警）：

- BackgroundRunner：按间隔跑、出错只记日志和指标（异常类型，不记说明）、guard 不满足就跳过、
  stop 等手上这一轮做完；
- 抓取 /metrics 不跑就绪探针、不查授权（数据库网络不通时抓取会超时，rodit_ready 一个样本都没有）：
  只输出后台刷新的上次结果和年龄，太旧了记成不就绪；
- 续期、过期清理在 serve 里跑（同一个 AssistService：同一把取消订购的闸门、同一份缓存），
  每轮之前确认单副本租约还在本进程；main 拿到租约之后启动、退出时停下；
- 授权过期检查不再把取消订购清理过、没有再开通的店铺算成"已过期"（令牌文件里还有它也一样）。

全部是合成数据；不连数据库（就绪检查用假数据库、内存仓储）、不联网。
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import timedelta
from typing import Any

import pytest
from ops_support import FakeReadinessDb, FakeServeLease, TestClient, parse_exposition, sample
from pydantic import SecretStr

from rodit.app.demo import seed_unit
from rodit.app.unit import in_memory_repos
from rodit.core.audit import AuditAction
from rodit.core.enums import Platform
from rodit.core.models import ShopAuth
from rodit.core.scopes import MVP_SCOPES
from rodit.ops import serve
from rodit.ops.background import BackgroundRunner, PeriodicJob
from rodit.ops.health import CheckResult
from rodit.ops.instrument import ObservedReadinessChecker, RoditMetrics
from rodit.ops.jobs import tenancy as tenancy_jobs
from rodit.ops.serve import build_server, main
from rodit.ops.settings import ServerSettings
from rodit.storage.db import MIGRATIONS
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A, SHOP_B, build_demo_world

ALL_VERSIONS = tuple(m.version for m in MIGRATIONS)


# ================================================================ BackgroundRunner


def test_runner_runs_jobs_on_their_interval_until_stopped():
    ran = threading.Event()
    calls: list[str] = []
    results: list[tuple[str, str]] = []

    def job() -> None:
        calls.append("x")
        if len(calls) >= 3:
            ran.set()

    runner = BackgroundRunner(
        [PeriodicJob("tick", job, 0.01), PeriodicJob("off", lambda: None, 0)],
        on_result=lambda name, outcome: results.append((name, outcome)),
    )
    runner.start()
    runner.start()  # 重复调用不会多起线程
    assert ran.wait(5)
    runner.stop()
    assert not runner.running
    stopped_at = len(calls)
    assert stopped_at >= 3
    assert {name for name, _ in results} == {"tick"}  # 间隔 0 的任务不跑
    assert all(outcome == "ok" for _, outcome in results)
    runner.start()  # 停下之后不再启动
    assert not runner.running and len(calls) == stopped_at


def test_runner_logs_only_the_error_type_and_keeps_going(caplog):
    secret = "tok-synthetic-should-not-be-logged"
    outcomes: list[str] = []

    def broken() -> None:
        raise RuntimeError(f"平台云代码出错：{secret} host=db.internal")

    runner = BackgroundRunner(
        [PeriodicJob("broken", broken, 60), PeriodicJob("partial", lambda: False, 60)],
        on_result=lambda name, outcome: outcomes.append(outcome),
    )
    with caplog.at_level(logging.DEBUG):
        assert runner.run_once("broken") == "error"
        assert runner.run_once("partial") == "error"  # 返回 False：没做完
    assert outcomes == ["error", "error"]
    assert "RuntimeError" in caplog.text and "broken" in caplog.text
    assert secret not in caplog.text and "db.internal" not in caplog.text


def test_runner_skips_when_the_guard_fails():
    calls: list[int] = []
    allowed = {"ok": False}
    runner = BackgroundRunner(
        [PeriodicJob("guarded", lambda: calls.append(1), 60, guard=lambda: allowed["ok"])]
    )
    assert runner.run_once("guarded") == "skipped" and calls == []
    allowed["ok"] = True
    assert runner.run_once("guarded") == "ok" and calls == [1]


def test_runner_rejects_duplicate_names_and_negative_intervals():
    with pytest.raises(ValueError, match="重名"):
        BackgroundRunner([PeriodicJob("a", lambda: None, 1), PeriodicJob("a", lambda: None, 1)])
    with pytest.raises(ValueError, match="不能为负"):
        BackgroundRunner([PeriodicJob("a", lambda: None, -1)])


def test_stop_waits_for_the_round_in_progress():
    started, release = threading.Event(), threading.Event()
    finished: list[bool] = []

    def slow() -> None:
        started.set()
        release.wait(5)
        finished.append(True)

    runner = BackgroundRunner([PeriodicJob("slow", slow, 60)])
    runner.start()
    assert started.wait(5)
    threading.Timer(0.05, release.set).start()
    runner.stop(timeout_s=5)
    assert finished == [True] and not runner.running


# ================================================================ /metrics 不跑探针


class Timer:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_readiness_publish_never_runs_probes_and_goes_stale():
    """抓取时只输出上次的结果和年龄：探针再慢也不会拖住抓取；结果太旧记成不就绪。"""
    metrics = RoditMetrics(platform=Platform.TAOBAO, mode="platform", version="0.0.1")
    timer = Timer()
    calls: list[int] = []

    def probe() -> dict[str, CheckResult]:
        calls.append(1)
        return {"database": CheckResult(True, {})}

    checker = ObservedReadinessChecker(
        {"db": probe}, metrics, cache_s=0, stale_after_s=60, timer=timer
    )
    metrics.registry.before_render(checker.publish)
    s = parse_exposition(metrics.render())
    assert calls == [] and sample(s, "rodit_ready") == 0  # 还没检查过：不算就绪
    assert sample(s, "rodit_readiness_age_seconds") is None

    assert checker.check().ready and calls == [1]
    timer.now += 30
    s = parse_exposition(metrics.render())
    assert calls == [1]  # 抓取没有跑探针
    assert sample(s, "rodit_ready") == 1 and sample(s, "rodit_readiness_age_seconds") == 30
    assert sample(s, "rodit_readiness_check_ok", check="database") == 1

    timer.now += 31  # 后台刷新卡住了：上次的结果超过 60 秒
    s = parse_exposition(metrics.render())
    assert calls == [1] and sample(s, "rodit_ready") == 0
    assert sample(s, "rodit_readiness_age_seconds") == 61
    checker.check()
    assert sample(parse_exposition(metrics.render()), "rodit_ready") == 1


@pytest.fixture
def local_server(ops_env, token_file, monkeypatch):
    """本地试用模式：合成平台绑定（带合成的续期接口）+ 离线替身 + 内存仓储（不连数据库）。"""
    for name in ("RODIT_LLM_BASE_URL", "RODIT_LLM_MODEL", "RODIT_LLM_API_KEY"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("RODIT_ENV", "local")
    world = build_demo_world()
    clock = FakeClock(world.now)
    db = FakeReadinessDb(ALL_VERSIONS)
    lease = FakeServeLease()
    server = build_server(
        ServerSettings.from_env(),
        clock=clock,
        repos=in_memory_repos(clock),
        database=db,
        lease=lease,
    )
    seed_unit(server.unit, world)
    return server, TestClient(server.app), token_file, clock, db, lease


def test_scrape_does_not_touch_the_database(local_server):
    """数据库网络不通时就绪探针、授权检查每次都要等连接超时；抓取路径上一个都不能有。"""
    server, client, tokens, _, db, lease = local_server
    lease.acquire()
    auth_calls: list[str] = []
    auth = server.unit.auth
    real_expiring = auth.expiring

    def counting(*args: Any, **kw: Any) -> Any:
        auth_calls.append("expiring")
        return real_expiring(*args, **kw)

    auth.expiring = counting  # type: ignore[method-assign]
    for _ in range(3):
        assert client.get("/metrics", headers=tokens.auth("metrics")).status_code == 200
    assert db.sessions == 0 and auth_calls == []  # 抓取时没查库
    s = parse_exposition(client.get("/metrics", headers=tokens.auth("metrics")).text)
    assert sample(s, "rodit_ready") == 0  # 还没检查过

    # 后台任务（或 /readyz）刷新之后，抓取输出上次的结果
    assert server.background.run_once("readiness") == "ok"
    assert server.background.run_once("auth_expiry") == "ok"
    assert db.sessions == 1 and auth_calls
    s = parse_exposition(client.get("/metrics", headers=tokens.auth("metrics")).text)
    assert sample(s, "rodit_ready") == 1
    assert sample(s, "rodit_readiness_check_ok", check="lease") == 1
    assert sample(s, "rodit_readiness_age_seconds") is not None
    assert sample(s, "rodit_auth_checked_shops") == 2
    assert sample(s, "rodit_background_task_runs_total", task="readiness", outcome="ok") == 1
    assert sample(s, "rodit_background_task_runs_total", task="purge_expired", outcome="error") == 0
    assert db.sessions == 1  # 抓取还是没查库


def test_serve_has_the_expected_background_jobs(local_server):
    jobs = local_server[0].background.jobs
    assert set(jobs) == {"readiness", "auth_expiry", "refresh_tokens", "purge_expired"}
    assert jobs["readiness"].interval_s == 10
    assert jobs["auth_expiry"].interval_s == 300
    assert jobs["refresh_tokens"].interval_s == 3600
    assert jobs["purge_expired"].interval_s == 21600
    assert jobs["refresh_tokens"].initial_delay_s == 60


def test_mode_c_has_no_auth_or_refresh_jobs(ops_env, token_file, monkeypatch):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_PURGE_EXPIRED_INTERVAL_S", "0")
    server = build_server(ServerSettings.from_env(), database=FakeReadinessDb(ALL_VERSIONS))
    assert set(server.background.jobs) == {"readiness", "purge_expired"}
    assert not server.background.jobs["purge_expired"].enabled  # 间隔 0：不跑


# ================================================================ 续期、过期清理在 serve 里跑


def _expire(server: Any, shop_id: str, days_ago: float) -> None:
    unit = server.unit
    auth = unit.repos.auth.get(shop_id)
    expires = unit.clock.now() - timedelta(days=days_ago)
    unit.repos.auth.save(auth.model_copy(update={"expires_at": expires}))


def test_purge_expired_runs_inside_serve_and_clears_serves_cache(local_server):
    """过期清理用 serve 的 AssistService：serve 进程里的工具缓存一起清掉（另起进程清不到）。"""
    server, _, _, _, _, lease = local_server
    lease.acquire()
    unit = server.unit
    unit.gateway.get_order(SHOP_A, "o_1001")  # serve 的缓存里有这家店的订单
    _expire(server, SHOP_A, 45)
    _expire(server, SHOP_B, 10)
    assert server.background.run_once("purge_expired") == "ok"
    assert unit.repos.auth.get(SHOP_A) is None and unit.repos.knowledge.get_shop(SHOP_A) is None
    assert unit.repos.knowledge.get_shop(SHOP_B) is not None
    [record] = unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.detail["shop_id"] == SHOP_A and record.detail["reason"] == "expired"
    assert record.actor == "system"
    # 清掉的是 serve 自己的缓存（另起进程清的是它自己那份空缓存）
    assert record.detail["counts"]["InMemoryCache"] == 1
    assert unit.cache.purge_shop(SHOP_A) == 0


def test_purge_job_retries_shops_that_were_not_fully_purged(local_server, caplog):
    """没删完的店铺令牌已经删了，purge_expired 下一轮不会再列出：后台任务记着、下一轮重试。"""
    server, _, _, _, _, lease = local_server
    lease.acquire()
    unit = server.unit
    _expire(server, SHOP_A, 45)
    broken = {"on": True}

    class Flaky:
        def purge_shop(self, shop_id: str) -> int:
            if broken["on"]:
                raise RuntimeError("模拟清理失败")
            return 0

    unit.extra_purgers.append(Flaky())
    with caplog.at_level(logging.ERROR):
        assert server.background.run_once("purge_expired") == "error"
    assert SHOP_A in caplog.text  # 日志里有要重试的店铺
    assert unit.repos.auth.get(SHOP_A) is None  # 令牌已删：purge_expired 不会再列出它
    assert unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED) == []
    broken["on"] = False
    assert server.background.run_once("purge_expired") == "ok"
    [record] = unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.detail["shop_id"] == SHOP_A
    assert server.background.run_once("purge_expired") == "ok"  # 不再重试
    assert len(unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)) == 1


def test_refresh_tokens_runs_inside_serve(local_server, caplog):
    server, _, _, _, _, lease = local_server
    lease.acquire()
    unit = server.unit
    unit.repos.auth.save(
        ShopAuth(
            shop_id=SHOP_A,
            access_token=SecretStr("tok-bg-old-synthetic"),
            scopes=sorted(MVP_SCOPES),
            expires_at=unit.clock.now() + timedelta(hours=3),
            refresh_token=SecretStr("tok-bg-refresh-synthetic"),
        )
    )
    with caplog.at_level(logging.DEBUG):
        assert server.background.run_once("refresh_tokens") == "ok"
    refreshed = unit.repos.auth.get(SHOP_A)
    assert refreshed.refreshed_at == unit.clock.now()
    assert refreshed.access_token.get_secret_value() != "tok-bg-old-synthetic"
    assert "续期成功 1" in caplog.text
    assert "tok-" not in caplog.text  # 日志里没有令牌


def test_lifecycle_jobs_skip_when_the_lease_is_not_held(local_server):
    """租约不在本进程（连接断了被别的进程抢走）：续期、清理这一轮跳过，不和别的进程抢着改数据。"""
    server, _, _, _, _, lease = local_server
    assert not lease.probe()["lease"].ok  # 还没拿到
    _expire(server, SHOP_A, 45)
    assert server.background.run_once("purge_expired") == "skipped"
    assert server.background.run_once("refresh_tokens") == "skipped"
    assert server.unit.repos.auth.get(SHOP_A) is not None
    assert server.background.run_once("readiness") == "ok"  # 就绪检查照常（报 lease 不就绪）
    assert not server.readiness.last.ready


def test_main_starts_background_jobs_after_the_lease_and_stops_them(
    ops_env, token_file, monkeypatch
):
    monkeypatch.setenv("RODIT_MODE_C_ONLY", "1")
    monkeypatch.setenv("RODIT_READINESS_REFRESH_S", "0.01")
    leases: list[FakeServeLease] = []

    def factory(settings: ServerSettings) -> FakeServeLease:
        leases.append(FakeServeLease(settings))
        return leases[-1]

    built: list[Any] = []
    real_build = serve.build_server

    def build(*args: Any, **kw: Any) -> Any:
        built.append(real_build(*args, **kw))
        return built[-1]

    monkeypatch.setattr(serve, "build_server", build)
    seen: dict[str, Any] = {}

    def run(app: Any, **kw: Any) -> None:
        [server] = built
        seen["lease"] = list(leases[0].events)
        seen["running"] = server.background.running
        deadline = time.monotonic() + 5
        while server.readiness.last is None and time.monotonic() < deadline:
            time.sleep(0.01)
        seen["checked"] = server.readiness.last is not None  # 后台线程刷新过就绪检查

    assert main(["serve"], run=run, lease_factory=factory) == 0
    assert seen == {"lease": ["acquire"], "running": True, "checked": True}
    assert not built[0].background.running  # 退出时停下
    assert leases[0].events == ["acquire", "release"]  # 先停后台任务，再放租约


# ============================================================ 授权过期检查：清理过的店铺不再算过期


def test_purged_shop_is_not_counted_as_expired_forever(local_server):
    """取消订购 / 过期清理只删库里的数据，令牌文件里这家店的令牌还在：不能一直算"已过期"
    （critical 的 RoditShopAuthExpired 按说明处理完也消不掉）。再开通之后回到检查里。"""
    server, client, tokens, clock, _, _ = local_server
    run = server.background.run_once

    def scrape() -> dict:
        return parse_exposition(client.get("/metrics", headers=tokens.auth("metrics")).text)

    assert run("auth_expiry") == "ok"
    s = scrape()
    assert sample(s, "rodit_auth_expired_shops") == 0
    assert sample(s, "rodit_auth_checked_shops") == 2

    server.service.tenancy.revoke(SHOP_A, reason="ops", actor="ops")
    assert run("auth_expiry") == "ok"
    s = scrape()
    assert sample(s, "rodit_auth_expired_shops") == 0
    assert sample(s, "rodit_auth_expiring_shops") == 0
    assert sample(s, "rodit_shop_auth_expiring", shop_id=SHOP_A) is None
    assert sample(s, "rodit_auth_checked_shops") == 1
    assert SHOP_A in server.authenticator.shop_ids()  # 令牌文件里还有它

    # 再开通（授权记录回来了）：回到检查里，过期了照样告警
    server.unit.repos.auth.save(
        ShopAuth(
            shop_id=SHOP_A,
            access_token=SecretStr("tok-bg-again-synthetic"),
            scopes=sorted(MVP_SCOPES),
            expires_at=clock.now() - timedelta(hours=1),
        )
    )
    assert run("auth_expiry") == "ok"
    s = scrape()
    assert sample(s, "rodit_auth_checked_shops") == 2
    assert sample(s, "rodit_auth_expired_shops") == 1
    assert sample(s, "rodit_shop_auth_expiring", shop_id=SHOP_A) == 1


def test_shop_without_auth_that_was_never_purged_still_counts(local_server):
    """没有授权、也没有清理记录的店铺（开通漏了）照旧算"已过期"。"""
    server, client, tokens, _, _, _ = local_server
    server.unit.repos.auth.delete(SHOP_B)
    assert server.background.run_once("auth_expiry") == "ok"
    s = parse_exposition(client.get("/metrics", headers=tokens.auth("metrics")).text)
    assert sample(s, "rodit_auth_expired_shops") == 1
    assert sample(s, "rodit_shop_auth_expiring", shop_id=SHOP_B) == 1


def test_serve_jobs_without_a_refresher_only_purge():
    class Unit:
        token_refresher = None

    class Service:
        unit = Unit()

    jobs = tenancy_jobs.serve_jobs(Service(), refresh_interval_s=10, purge_interval_s=20)
    assert [j.name for j in jobs] == ["purge_expired"]
    assert jobs[0].initial_delay_s == 20  # 不超过间隔
