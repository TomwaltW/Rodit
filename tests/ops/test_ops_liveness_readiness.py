"""存活、就绪、后台任务结果的几处边界（方案 §12.2 可用性、§12.4 可观测性）：

- /metrics、/healthz 不排在业务请求后面：数据库网络不通时同步的业务接口占满线程池（每个都在等
  连接超时），抓取和容器健康检查照样马上回答；
- serve 的后台就绪刷新每轮都真的跑探针（不用 /readyz 的缓存）：RODIT_READINESS_CACHE_S 调大了，
  健康的服务也不会因为"结果太旧"被记成不就绪；只由 /readyz 刷新（间隔 0）时过期阈值算上缓存期；
- 授权过期检查出错时后台任务这一轮记 error（RoditBackgroundTaskFailing 能看到），上次成功的
  时刻不往前走。

全部是合成数据；不连数据库（就绪检查用假数据库、内存仓储）、不联网。
"""

from __future__ import annotations

import threading
from typing import Any

import anyio
import anyio.to_thread
import httpx
import pytest
from ops_support import FakeReadinessDb, TestClient, parse_exposition, sample

from rodit.app.demo import seed_unit
from rodit.app.unit import in_memory_repos
from rodit.ops import serve
from rodit.ops.serve import build_server
from rodit.ops.settings import ServerSettings
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A, build_demo_world


class Timer:
    """就绪检查用的单调时钟替身（秒）。"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _local_server(monkeypatch: pytest.MonkeyPatch, **env: str) -> tuple[Any, FakeReadinessDb]:
    """本地试用模式：合成平台绑定 + 离线替身 + 内存仓储 + 假数据库（不连数据库）。"""
    for name in ("RODIT_LLM_BASE_URL", "RODIT_LLM_MODEL", "RODIT_LLM_API_KEY"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("RODIT_ENV", "local")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    world = build_demo_world()
    clock = FakeClock(world.now)
    db = FakeReadinessDb()
    server = build_server(
        ServerSettings.from_env(), clock=clock, repos=in_memory_repos(clock), database=db
    )
    seed_unit(server.unit, world)
    return server, db


def _scrape(client: TestClient, tokens: Any) -> dict[Any, float]:
    resp = client.get("/metrics", headers=tokens.auth("metrics"))
    assert resp.status_code == 200
    return parse_exposition(resp.text)


# ================================================================ /metrics、/healthz 不排队


def test_metrics_and_healthz_answer_while_business_requests_fill_the_thread_pool(
    ops_env, token_file, monkeypatch
):
    """业务接口是同步函数，共用 AnyIO 的线程池；数据库网络不通时它们都卡在连接超时上。
    这里把线程池缩到 2 个线程、让 4 个业务请求卡住：/metrics 和 /healthz 仍要马上回答。"""
    server, _ = _local_server(monkeypatch)
    assert server.background.run_once("readiness") == "ok"
    release = threading.Event()
    entered: list[int] = []
    knowledge = server.unit.repos.knowledge
    real_list = knowledge.list_item_cards

    def stuck(*args: Any, **kw: Any) -> Any:  # 模拟数据库连接超时：占着线程不放
        entered.append(1)
        release.wait(10)
        return real_list(*args, **kw)

    knowledge.list_item_cards = stuck  # type: ignore[method-assign]
    business: list[int] = []
    timings: dict[str, int] = {}

    async def scenario() -> None:
        anyio.to_thread.current_default_thread_limiter().total_tokens = 2
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://rodit.test") as client:

            async def call_business() -> None:
                resp = await client.get(
                    f"/v1/shops/{SHOP_A}/items", headers=token_file.auth("agent_a")
                )
                business.append(resp.status_code)

            async with anyio.create_task_group() as tg:
                for _ in range(4):
                    tg.start_soon(call_business)
                try:
                    with anyio.fail_after(5):
                        while len(entered) < 2:  # 两个线程都被业务请求占住了
                            await anyio.sleep(0.01)
                    with anyio.fail_after(2):  # 旧代码：排在卡住的业务请求后面，这里超时
                        metrics = await client.get("/metrics", headers=token_file.auth("metrics"))
                        timings["metrics"] = metrics.status_code
                        assert "rodit_ready 1" in metrics.text.splitlines()
                        healthz = await client.get("/healthz")
                        timings["healthz"] = healthz.status_code
                    assert len(entered) == 2 and business == []  # 业务请求还卡着
                finally:
                    release.set()

    anyio.run(scenario)
    assert timings == {"metrics": 200, "healthz": 200}
    assert business == [200, 200, 200, 200]  # 线程放开之后业务请求照常完成


# ================================================================ 后台就绪刷新不用缓存


def test_background_readiness_refresh_runs_the_probes_even_with_a_long_cache(
    ops_env, token_file, monkeypatch
):
    """RODIT_READINESS_CACHE_S 比过期阈值（60 秒）长：后台刷新拿缓存的话，结果的年龄一直从
    第一次检查算起，60 秒后抓取就把健康的服务记成不就绪（RoditNotReady 是 critical）。"""
    server, db = _local_server(monkeypatch, RODIT_READINESS_CACHE_S="600")
    timer = Timer()
    # build_server 不接受时钟参数：换掉就绪检查用的单调时钟（缓存和年龄都按它算）
    monkeypatch.setattr(server.readiness, "_timer", timer)
    client = TestClient(server.app)

    assert server.background.run_once("readiness") == "ok"
    assert db.sessions == 1
    for _ in range(12):  # 后台每 10 秒刷新一次，共 120 秒
        timer.now += 10
        assert server.background.run_once("readiness") == "ok"
        s = _scrape(client, token_file)
        assert sample(s, "rodit_ready") == 1
        assert sample(s, "rodit_readiness_age_seconds") == 0
    assert db.sessions == 13  # 每轮都真的查了库

    # /readyz 照样用缓存：两次后台刷新之间不再查库
    assert client.get("/readyz").status_code == 200
    assert db.sessions == 13


def test_readyz_only_refresh_counts_the_cache_in_the_staleness_threshold(
    ops_env, token_file, monkeypatch
):
    """间隔为 0：只由 /readyz 刷新，/readyz 在缓存期内拿缓存的结果（年龄从真的检查那一刻算）。
    阈值没算上缓存期时，缓存 600 秒、每 10 秒一次 /readyz，健康的服务 60 秒后就被记成不就绪。"""
    server, db = _local_server(
        monkeypatch, RODIT_READINESS_REFRESH_S="0", RODIT_READINESS_CACHE_S="600"
    )
    timer = Timer()
    monkeypatch.setattr(server.readiness, "_timer", timer)
    client = TestClient(server.app)

    for step in range(71):  # 0 … 700 秒，每 10 秒一次 /readyz（编排系统的就绪探针）
        if step:
            timer.now += 10
        assert client.get("/readyz").status_code == 200
        s = _scrape(client, token_file)
        assert sample(s, "rodit_ready") == 1, f"第 {step * 10} 秒"
    assert db.sessions == 2  # 0 秒、600 秒各真的查了一次

    # 没有人再刷新：超过"缓存期 + 60 秒"照样记成不就绪（刷新卡住了还是看得出来）
    timer.now = 1000.0 + 600 + 660 + 1
    assert sample(_scrape(client, token_file), "rodit_ready") == 0


@pytest.mark.parametrize(
    ("refresh_s", "cache_s", "expected"),
    [
        (10.0, 2.0, 60.0),  # 默认值：后台每轮都真的检查，缓存不影响
        (10.0, 600.0, 60.0),
        (30.0, 2.0, 90.0),
        (0.0, 2.0, 62.0),  # 只由 /readyz 刷新：加上缓存期
        (0.0, 600.0, 660.0),
    ],
)
def test_readiness_stale_threshold(refresh_s, cache_s, expected):
    settings = ServerSettings(
        platform="taobao", readiness_refresh_s=refresh_s, readiness_cache_s=cache_s
    )
    assert serve.readiness_stale_after_s(settings) == expected


# ================================================================ 授权检查出错记 error


def test_failed_auth_expiry_check_is_a_background_error(ops_env, token_file, monkeypatch):
    """数据库故障时授权检查出错：后台任务这一轮记 error（RoditBackgroundTaskFailing），
    上次成功的时刻不往前走；授权告警保留上次的结果。"""
    server, _ = _local_server(monkeypatch)
    client = TestClient(server.app)
    assert server.background.run_once("auth_expiry") == "ok"
    s = _scrape(client, token_file)
    succeeded_at = sample(
        s, "rodit_background_task_last_success_timestamp_seconds", task="auth_expiry"
    )
    assert succeeded_at is not None
    checked = sample(s, "rodit_auth_checked_shops")

    def broken(*args: Any, **kw: Any) -> Any:
        raise ConnectionError("合成的数据库故障")

    server.unit.auth.expiring = broken  # type: ignore[method-assign]
    assert server.background.run_once("auth_expiry") == "error"
    s = _scrape(client, token_file)
    assert sample(s, "rodit_auth_expiry_check_errors_total") == 1
    assert sample(s, "rodit_background_task_runs_total", task="auth_expiry", outcome="error") == 1
    assert sample(s, "rodit_background_task_runs_total", task="auth_expiry", outcome="ok") == 1
    assert (
        sample(s, "rodit_background_task_last_success_timestamp_seconds", task="auth_expiry")
        == succeeded_at
    )
    assert sample(s, "rodit_auth_checked_shops") == checked  # 上次的结果
