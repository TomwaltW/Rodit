"""平台客服指标同步的离线任务（python -m rodit.ops sync-platform-metrics，方案 §12.4、§8.4、§9.2）。

命令行跑在内存演示单元上（build_service 换成演示单元，不连数据库）：每家有授权的店铺同步一次、
只同步指名的店铺、参数和绑定不对时退出码 2、平台接口不可用 / 串店退出码 1；不拿单副本租约，
和 serve 同时跑时按取消订购的系统事件跳过、写完发现授权没了就删掉这次的写入。全部是合成数据。
"""

from __future__ import annotations

import io
import logging

import pytest

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit
from rodit.app.platform_metrics import PLATFORM_METRICS_SYNCED
from rodit.core.audit import AuditAction
from rodit.core.enums import MetricSource, Platform, ShopStatus
from rodit.core.errors import StorageUnavailable
from rodit.core.models import AuditEvent
from rodit.ops import serve
from rodit.ops.jobs import platform_metrics as metrics_jobs
from rodit.testing.fixtures import SHOP_A, SHOP_B


@pytest.fixture
def cli(monkeypatch):
    """命令行跑在淘宝演示单元上（店铺 A、B，带平台指标接口替身）；返回应用服务和 build_service
    的调用次数。"""
    service = AssistService(build_demo_unit(Platform.TAOBAO))
    built: list[dict] = []

    def build(environ):
        built.append(dict(environ))
        return service

    monkeypatch.setattr(metrics_jobs, "build_service", build)
    return service, built


def run(*argv: str, service: AssistService | None = None) -> tuple[int | None, str]:
    """经部署入口同一个命令行解析，直接调任务模块（输出收进字符串）。"""
    out = io.StringIO()
    args = serve._parser().parse_args(list(argv))  # noqa: SLF001
    return metrics_jobs.run(args, {}, service=service, out=out), out.getvalue()


def rows(service: AssistService, shop_id: str) -> list:
    return service.unit.repos.platform_metrics.list(shop_id)


def test_command_is_registered_and_only_claims_itself():
    commands = set(serve._parser()._subparsers._group_actions[0].choices)  # noqa: SLF001
    assert metrics_jobs.COMMAND in commands  # 注册进了部署入口
    args = serve._parser().parse_args(["refresh-tokens"])  # noqa: SLF001
    assert metrics_jobs.run(args, {}) is None


def test_sync_every_authorized_shop_through_serve_main(cli, capsys):
    service, built = cli
    assert serve.main(["sync-platform-metrics", "--days", "7"]) == 0
    assert len(built) == 1
    assert len(rows(service, SHOP_A)) == len(rows(service, SHOP_B)) == 7
    assert {r.source for r in rows(service, SHOP_A)} == {MetricSource.PLATFORM}
    out = capsys.readouterr().out
    assert f"{SHOP_A} " in out and f"{SHOP_B} " in out and "写入 7 条" in out
    for shop in (SHOP_A, SHOP_B):
        [event] = service.unit.repos.audit.list_events(shop, action=PLATFORM_METRICS_SYNCED)
        assert event.actor == "ops"


def test_default_days_cover_both_report_windows(cli):
    service, _ = cli
    code, out = run("sync-platform-metrics", service=service)
    assert code == 0 and len(rows(service, SHOP_A)) == 56


def test_only_the_named_shops(cli):
    service, _ = cli
    code, out = run("sync-platform-metrics", "--days", "3", "--shop-id", SHOP_B, service=service)
    assert code == 0
    assert rows(service, SHOP_A) == [] and len(rows(service, SHOP_B)) == 3
    assert SHOP_A not in out


def test_named_shop_without_authorization_counts_as_not_synced(cli):
    service, _ = cli
    service.unit.repos.auth.delete(SHOP_B)
    code, out = run(
        "sync-platform-metrics",
        "--shop-id",
        SHOP_A,
        "--shop-id",
        "s_nope",
        "--shop-id",
        SHOP_B,
        service=service,
    )
    assert code == 1
    assert "s_nope 没有授权或不存在" in out and f"{SHOP_B} 没有授权或不存在" in out
    assert len(rows(service, SHOP_A)) == 56 and rows(service, SHOP_B) == []


def test_shops_without_authorization_are_not_listed(cli):
    service, _ = cli
    service.unit.repos.auth.delete(SHOP_B)
    code, out = run("sync-platform-metrics", "--days", "2", service=service)
    assert code == 0 and SHOP_B not in out and rows(service, SHOP_B) == []


@pytest.mark.parametrize("days", ["0", "401", "-3"])
def test_bad_days_exit_2_without_building_the_unit(cli, caplog, days):
    service, built = cli
    with caplog.at_level(logging.ERROR):
        assert serve.main(["sync-platform-metrics", "--days", days]) == 2
    assert built == [] and "--days" in caplog.text


def test_unit_without_metrics_source_or_storage_exits_2(cli, caplog):
    service, _ = cli
    source = service.unit.metrics_source
    service.unit.metrics_source = None
    with caplog.at_level(logging.ERROR):
        assert run("sync-platform-metrics", service=service)[0] == 2
    assert "metrics_source" in caplog.text
    service.unit.metrics_source = source
    service.unit.repos.platform_metrics = None
    assert run("sync-platform-metrics", service=service)[0] == 2


def test_platform_outage_exits_1_and_writes_nothing(cli):
    service, _ = cli
    service.unit.metrics_source.fail = True
    code, out = run("sync-platform-metrics", service=service)
    assert code == 1 and out.count("平台指标接口暂时不可用") == 2
    assert rows(service, SHOP_A) == [] and rows(service, SHOP_B) == []


def test_rows_of_another_shop_exit_1(cli):
    service, _ = cli
    source = service.unit.metrics_source
    original = source.fetch

    def leaky(shop_id, since, until):
        found = original(shop_id, since, until)
        return [*found, found[0].model_copy(update={"shop_id": "s_9999"})]

    source.fetch = leaky
    code, out = run("sync-platform-metrics", "--days", "2", service=service)
    assert code == 1 and "别的店铺的数据" in out
    assert rows(service, SHOP_A) == [] and rows(service, SHOP_B) == []


def test_suspended_and_unauthorized_shops_are_skipped(cli):
    service, _ = cli
    repo = service.unit.repos.knowledge
    repo.upsert_shop(repo.get_shop(SHOP_A).model_copy(update={"status": ShopStatus.SUSPENDED}))
    auth = service.unit.repos.auth.get(SHOP_B)
    service.unit.repos.auth.save(auth.model_copy(update={"scopes": ["order.read"]}))
    code, out = run("sync-platform-metrics", "--days", "2", service=service)
    assert code == 0
    assert f"{SHOP_A} 已停用" in out and f"{SHOP_B} 店铺授权无效或缺权限" in out
    assert rows(service, SHOP_A) == [] and rows(service, SHOP_B) == []
    assert service.unit.metrics_source.calls == []  # 一次都没调平台接口


def _purge_started(service: AssistService, shop_id: str) -> None:
    service.unit.repos.audit.record(
        AuditEvent(
            shop_id=None,
            actor="system",
            action=AuditAction.SHOP_PURGE_STARTED.value,
            detail={"shop_id": shop_id, "reason": "unsubscribed"},
            created_at=service.unit.clock.now(),
        )
    )


def test_shop_whose_purge_started_during_the_job_is_not_written(cli):
    service, _ = cli
    _purge_started(service, SHOP_A)  # serve 在任务开始之后开始清理店铺 A
    code, out = run("sync-platform-metrics", "--days", "2", service=service)
    assert code == 0 and f"{SHOP_A} 任务期间开始了取消订购" in out
    assert rows(service, SHOP_A) == [] and len(rows(service, SHOP_B)) == 2


def test_shop_revoked_while_syncing_has_the_late_writes_removed(cli):
    """serve 在拉数据期间取消订购了店铺 A（先删令牌、再删数据）：本任务写完发现授权没了，
    再删一次这家店的指标和审计，不留迟到的数据（§8.4）。"""
    service, _ = cli
    source = service.unit.metrics_source
    original = source.fetch

    def revoked_meanwhile(shop_id, since, until):
        found = original(shop_id, since, until)
        if shop_id == SHOP_A:
            service.unit.repos.auth.delete(SHOP_A)
        return found

    source.fetch = revoked_meanwhile
    code, out = run("sync-platform-metrics", "--days", "2", service=service)
    assert code == 0 and f"{SHOP_A} 同步期间被取消订购" in out
    assert rows(service, SHOP_A) == []
    assert service.unit.repos.audit.list_events(SHOP_A) == []
    assert len(rows(service, SHOP_B)) == 2


def test_purge_started_and_reonboarded_while_syncing_warns(cli, caplog):
    service, _ = cli
    source = service.unit.metrics_source
    original = source.fetch

    def purge_meanwhile(shop_id, since, until):
        if shop_id == SHOP_A:
            _purge_started(service, SHOP_A)  # 期间开始清理、令牌还在（又重新开通了）
        return original(shop_id, since, until)

    source.fetch = purge_meanwhile
    with caplog.at_level(logging.WARNING):
        code, out = run("sync-platform-metrics", "--days", "2", service=service)
    assert code == 0 and f"警告：{SHOP_A} 同步期间开始了取消订购" in out
    assert len(rows(service, SHOP_A)) == 2  # 写的是平台给这家店的数，提示核对
    assert "sync-platform-metrics" in caplog.text


def test_no_authorized_shops(cli):
    service, _ = cli
    for shop in (SHOP_A, SHOP_B):
        service.unit.repos.auth.delete(shop)
    code, out = run("sync-platform-metrics", service=service)
    assert code == 0 and "本单元没有有授权的店铺" in out


def test_job_does_not_take_the_single_replica_lease(cli, monkeypatch):
    """不拿租约（serve 一直占着，拿了就永远跑不了）：租约模块被调用就失败。"""
    import rodit.ops.lease as lease_module

    def refuse(*args, **kwargs):
        raise AssertionError("sync-platform-metrics 不应该拿单副本租约")

    monkeypatch.setattr(lease_module, "job_lease", refuse)
    service, _ = cli
    assert run("sync-platform-metrics", "--days", "1", service=service)[0] == 0


def test_database_errors_map_to_exit_1_without_leaking_the_dsn(cli, monkeypatch, caplog):
    def down(environ):
        raise StorageUnavailable("连不上 host=db.internal user=rodit（模拟）")

    monkeypatch.setattr(metrics_jobs, "build_service", down)
    with caplog.at_level(logging.ERROR):
        assert serve.main(["sync-platform-metrics"]) == 1
    assert "db.internal" not in caplog.text
