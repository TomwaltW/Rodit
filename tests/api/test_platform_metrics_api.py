"""平台客服指标接口（rodit.api.routes_platform_metrics；方案 §12.4、§15 Phase 2、§9.2、§8.4）：
角色、请求校验、店铺隔离、停用店铺、没有指标存储 / 没接平台接口 / 平台接口故障时的错误码、
登记人取自身份、报表的接入日期参数。串店和 401 由 test_api_auth 对全部路由统一覆盖
（路由清单 api_routes_platform_metrics.py）。淘宝演示单元 + 合成数据，不联网。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from api_support import CALLERS, Api

from rodit.app.platform_metrics import PLATFORM_METRICS_RECORDED, PLATFORM_METRICS_SYNCED
from rodit.core.clock import CST
from rodit.core.enums import MetricSource, PlatformMetric, ShopStatus
from rodit.core.models import PlatformMetricSample
from rodit.testing.fixtures import DEMO_GO_LIVE_DAYS_AGO, SHOP_A, SHOP_B

BASE = f"/v1/shops/{SHOP_A}"
REPORT = f"{BASE}/reports/platform-metrics"
ROWS = f"{BASE}/platform-metrics"
SYNC = f"{BASE}/platform-metrics/sync"


def today(api: Api) -> date:
    return api.unit.clock.now().astimezone(CST).date()


def go_live(api: Api) -> str:
    return (today(api) - timedelta(days=DEMO_GO_LIVE_DAYS_AGO)).isoformat()


def record_body(api: Api, **over) -> dict:
    return {
        "metric": "taobao_3min_response_rate",
        "day": (today(api) - timedelta(days=1)).isoformat(),
        "value": 0.93,
        "sample_size": 250,
        **over,
    }


def rows(api: Api, shop_id: str = SHOP_A) -> list[PlatformMetricSample]:
    return api.unit.repos.platform_metrics.list(shop_id)


# ================================================================ 同步


@pytest.mark.parametrize("who", ["manager_a", "merchant_a"])
def test_manager_and_merchant_can_sync(api: Api, who: str):
    r = api.post(SYNC, who, params={"days": 7})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["written"] == body["fetched"] == 7 and body["skipped"] == {}
    assert body["metrics"] == ["taobao_3min_response_rate"]
    assert body["until"] == (today(api) - timedelta(days=1)).isoformat()
    assert len(rows(api)) == 7 and rows(api, SHOP_B) == []
    [event] = api.unit.repos.audit.list_events(SHOP_A, action=PLATFORM_METRICS_SYNCED)
    assert event.actor == CALLERS[who].user_id  # 操作人取自令牌对应的身份


def test_sync_actor_is_the_caller(api: Api):
    assert api.post(SYNC, "merchant_a").status_code == 200
    [event] = api.unit.repos.audit.list_events(SHOP_A, action=PLATFORM_METRICS_SYNCED)
    assert event.actor == "owner_a" and event.detail["written"] == 56


@pytest.mark.parametrize("days", [0, 401, "x"])
def test_sync_days_are_validated(api: Api, days):
    r = api.post(SYNC, "merchant_a", params={"days": days})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert rows(api) == []


def test_sync_body_must_be_empty(api: Api):
    r = api.post(SYNC, "merchant_a", json={"actor": "someone_else"})
    assert r.status_code == 422
    assert api.post(SYNC, "merchant_a", json={}).status_code == 200


def test_sync_without_metrics_source_is_503_with_a_clear_code(api: Api):
    api.unit.metrics_source = None
    r = api.post(SYNC, "merchant_a")
    assert r.status_code == 503
    assert r.json()["error"] == "metrics_source_not_configured"
    assert "手工登记" in r.json()["message"]


def test_sync_when_the_platform_is_down_writes_nothing(api: Api):
    api.unit.metrics_source.fail = True
    r = api.post(SYNC, "manager_a")
    assert r.status_code == 503 and r.json()["error"] == "metrics_source_unavailable"
    assert "什么都没有写" in r.json()["message"]
    assert rows(api) == []
    assert api.unit.repos.audit.list_events(SHOP_A, action=PLATFORM_METRICS_SYNCED) == []


def test_sync_without_authorization_is_refused(api: Api):
    api.unit.repos.auth.delete(SHOP_A)
    r = api.post(SYNC, "merchant_a")
    assert r.status_code == 403 and r.json()["error"] == "shop_not_authorized"
    assert api.unit.metrics_source.calls == [] and rows(api) == []
    # 只读的报表照常能看
    assert api.get(REPORT, "merchant_a").status_code == 200


def test_sync_rows_of_another_shop_are_refused_as_tenant_violation(api: Api):
    source = api.unit.metrics_source
    stray = source.samples[0].model_copy(update={"shop_id": SHOP_B})
    # 替身按店铺过滤：把 B 店的数改成"返回给 A 店"模拟平台云代码的串店错误
    source.samples = [s for s in source.samples if s.shop_id == SHOP_A]
    original = source.fetch
    source.fetch = lambda shop_id, since, until: [*original(shop_id, since, until), stray]
    r = api.post(SYNC, "merchant_a", params={"days": 7})
    assert r.status_code == 403 and r.json()["error"] == "tenant_isolation"
    assert rows(api) == [] and rows(api, SHOP_B) == []


# ================================================================ 商家登记


def test_merchant_records_a_value_and_the_recorder_comes_from_the_token(api: Api):
    r = api.post(ROWS, "merchant_a", json=record_body(api))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] is True and body["replaced"] is False
    assert body["sample"]["recorded_by"] == "owner_a" and body["sample"]["source"] == "merchant"
    [stored] = rows(api)
    assert stored.value == 0.93 and stored.sample_size == 250
    assert stored.source == MetricSource.MERCHANT and stored.recorded_by == "owner_a"
    [event] = api.unit.repos.audit.list_events(SHOP_A, action=PLATFORM_METRICS_RECORDED)
    assert event.actor == "owner_a" and event.detail["value"] == 0.93
    again = api.post(ROWS, "merchant_a", json=record_body(api)).json()
    assert again["changed"] is False


@pytest.mark.parametrize(
    "over",
    [
        {"value": 1.5},  # 比例超过 1
        {"value": -0.1},
        {"value": "0.9"},  # 字符串不收
        {"value": True},
        {"metric": "douyin_help_rate"},  # 不是本平台的指标
        {"metric": "nope"},
        {"day": "2099-01-01"},  # 晚于今天
        {"day": "2020-01-01"},  # 早于 400 天
        {"day": "昨天"},
        {"sample_size": -1},
        {"sample_size": 1.5},
        {"recorded_by": "someone_else"},  # 登记人只取自身份
        {"source": "platform"},  # 不能冒充平台数据
        {"shop_id": SHOP_B},
    ],
)
def test_record_validation_is_422_and_writes_nothing(api: Api, over):
    r = api.post(ROWS, "merchant_a", json=record_body(api, **over))
    assert r.status_code == 422, r.text
    assert r.json()["error"] == "validation_error"
    assert rows(api) == []


def test_record_rejects_nan_and_infinity(api: Api):
    for token in ("NaN", "Infinity"):
        day = (today(api) - timedelta(days=1)).isoformat()
        raw = f'{{"metric": "taobao_3min_response_rate", "day": "{day}", "value": {token}}}'
        r = api.post(ROWS, "merchant_a", content=raw, headers={"Content-Type": "application/json"})
        assert r.status_code == 422, (token, r.text)
    assert rows(api) == []


def test_only_the_merchant_records(api: Api):
    for who in ("manager_a", "reviewer_a", "agent_a"):
        r = api.post(ROWS, who, json=record_body(api))
        assert r.status_code == 403 and r.json()["error"] == "forbidden_role"
    assert rows(api) == []


# ================================================================ 报表、原始数据


@pytest.mark.parametrize("who", ["manager_a", "merchant_a", "reviewer_a"])
def test_report_after_sync_shows_the_comparison(api: Api, who: str):
    assert api.post(SYNC, "merchant_a").status_code == 200
    r = api.get(REPORT, who, params={"go_live": go_live(api)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "improved" and body["verdict_label"] == "较接入前改善"
    assert body["go_live"] == go_live(api) and body["go_live_source"] == "query"
    assert body["window_days"] == 28 and body["reliable"] is True
    [metric] = body["metrics"]
    assert metric["metric"] == "taobao_3min_response_rate" and metric["unit"] == "ratio"
    assert metric["direction"] == "improved" and metric["higher_is_better"] is True
    assert metric["before"]["days"] == metric["after"]["days"] == 28
    assert len(metric["weekly"]) == 8 and metric["weekly"][4]["start"] == go_live(api)
    assert metric["note"]  # 口径说明


def test_report_without_go_live_asks_for_one(api: Api):
    body = api.get(REPORT, "merchant_a").json()
    assert body["verdict"] == "no_go_live" and body["go_live"] is None
    assert body["go_live_source"] == "unknown" and body["metrics"] == []


@pytest.mark.parametrize(
    "params",
    [
        {"go_live": "2099-01-01"},
        {"go_live": "2026/09/01"},
        {"go_live": "2020-01-01"},
        {"window_days": 6},
        {"window_days": 91},
    ],
)
def test_report_parameters_are_validated(api: Api, params):
    r = api.get(REPORT, "merchant_a", params=params)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_report_window_days(api: Api):
    api.post(SYNC, "merchant_a")
    body = api.get(REPORT, "merchant_a", params={"go_live": go_live(api), "window_days": 7}).json()
    assert body["window_days"] == 7
    assert body["metrics"][0]["before"]["days"] == 7 and len(body["metrics"][0]["weekly"]) == 2
    # 只有 7 天：可靠性提示不影响结论
    assert body["verdict"] == "improved"


def test_list_rows_with_filters(api: Api):
    api.post(SYNC, "merchant_a", params={"days": 10})
    api.post(ROWS, "merchant_a", json=record_body(api))
    since = (today(api) - timedelta(days=2)).isoformat()
    r = api.get(ROWS, "reviewer_a", params={"since": since})
    assert r.status_code == 200
    body = r.json()
    assert body["since"] == since and body["until"] == today(api).isoformat()
    assert [(s["day"], s["source"]) for s in body["samples"]] == [
        (since, "platform"),
        ((today(api) - timedelta(days=1)).isoformat(), "merchant"),
        ((today(api) - timedelta(days=1)).isoformat(), "platform"),
    ]
    only = api.get(ROWS, "manager_a", params={"metric": "taobao_3min_response_rate"}).json()
    assert len(only["samples"]) == 11
    for params in (
        {"metric": "douyin_help_rate"},
        {"metric": "nope"},
        {"since": today(api).isoformat(), "until": (today(api) - timedelta(days=1)).isoformat()},
        {"since": "2025-01-01", "until": today(api).isoformat()},
    ):
        r = api.get(ROWS, "merchant_a", params=params)
        assert r.status_code == 422, params


def test_shops_only_see_their_own_rows(api: Api):
    api.post(f"/v1/shops/{SHOP_B}/platform-metrics/sync", "merchant_b", params={"days": 7})
    assert len(rows(api, SHOP_B)) == 7
    assert api.get(ROWS, "merchant_a").json()["samples"] == []
    report = api.get(REPORT, "merchant_a", params={"go_live": go_live(api)}).json()
    assert report["verdict"] == "insufficient"


# ================================================================ 停用店铺、没有存储


def test_suspended_shop_gets_403_on_every_endpoint(api: Api):
    repo = api.unit.repos.knowledge
    repo.upsert_shop(repo.get_shop(SHOP_A).model_copy(update={"status": ShopStatus.SUSPENDED}))
    calls = [
        ("GET", REPORT, "merchant_a", None),
        ("GET", ROWS, "reviewer_a", None),
        ("POST", ROWS, "merchant_a", record_body(api)),
        ("POST", SYNC, "manager_a", None),
    ]
    for method, path, who, body in calls:
        r = api.call(method, path, who, json=body)
        assert r.status_code == 403 and r.json()["error"] == "shop_inactive", path
    assert rows(api) == []


def test_unit_without_metrics_storage_is_503_on_every_endpoint(api: Api):
    api.unit.repos.platform_metrics = None
    calls = [
        ("GET", REPORT, "merchant_a", None),
        ("GET", ROWS, "reviewer_a", None),
        ("POST", ROWS, "merchant_a", record_body(api)),
        ("POST", SYNC, "manager_a", None),
    ]
    for method, path, who, body in calls:
        r = api.call(method, path, who, json=body)
        assert r.status_code == 503, (path, r.text)
        assert r.json()["error"] == "platform_metrics_unavailable"


def test_routes_are_in_the_api_schema_with_their_tag(api: Api):
    spec = api.get("/openapi.json").json()
    paths = spec["paths"]
    for path, method in (
        ("/v1/shops/{shop_id}/reports/platform-metrics", "get"),
        ("/v1/shops/{shop_id}/platform-metrics", "get"),
        ("/v1/shops/{shop_id}/platform-metrics", "post"),
        ("/v1/shops/{shop_id}/platform-metrics/sync", "post"),
    ):
        assert paths[path][method]["tags"] == ["平台客服指标"]
    assert PlatformMetric.TAOBAO_3MIN_RESPONSE_RATE.value in str(spec["components"])  # 指标枚举
