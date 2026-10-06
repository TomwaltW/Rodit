"""商家在设置页填客服工作时间（附录 C 第 13 题；§15 Phase 3 全自动先在非工作时间试行）：HTTP 接口。
数据全部合成。"""

from __future__ import annotations

import pytest
from api_support import Api, shop_snapshot

from rodit.testing.fixtures import SHOP_A

CONFIG = f"/v1/shops/{SHOP_A}/console/config"


def test_merchant_sets_service_hours_and_sees_them_normalized(api: Api):
    r = api.post(
        CONFIG,
        "merchant_a",
        json={"changes": {"service_hours": ["9:00-12:00", "下午1点半到晚10点"]}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] == ["service_hours"]
    assert body["effective"]["service_hours"] == ["09:00-12:00", "13:30-22:00"]
    assert "service_hours" in body["editable"] and "service_hours" not in body["locked"]
    status = api.get(f"/v1/shops/{SHOP_A}/rollout/status", "merchant_a").json()
    assert status["auto_schedule"]["service_hours"] == ["09:00-12:00", "13:30-22:00"]


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        (["9点到11点"], "上午还是晚上"),
        (["周一至周五 9:00-18:00"], "星期"),
        (["7:00-8:00"] * 1 + [f"{h}:00-{h}:30" for h in range(9, 15)], "最多 6"),
    ],
)
def test_unclear_service_hours_are_422_with_a_message(api: Api, value, fragment: str):
    before = shop_snapshot(api, SHOP_A)
    r = api.post(CONFIG, "merchant_a", json={"changes": {"service_hours": value}})
    assert r.status_code == 422 and fragment in r.json()["message"]
    assert shop_snapshot(api, SHOP_A) == before


@pytest.mark.parametrize("caller", ["manager_a", "agent_a", "reviewer_a"])
def test_only_the_merchant_edits_service_hours(api: Api, caller: str):
    r = api.post(CONFIG, caller, json={"changes": {"service_hours": ["9:00-23:00"]}})
    assert r.status_code == 403
