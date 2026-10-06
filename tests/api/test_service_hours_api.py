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


def test_off_duty_wording_is_422_with_how_to_write_it(api: Api):
    """设置页和问卷第 13 题同一个解析：写了不在岗的时段（午休、机器人值班……）不猜，说明怎么写。"""
    before = shop_snapshot(api, SHOP_A)
    r = api.post(
        CONFIG, "merchant_a", json={"changes": {"service_hours": ["全天在线，12:00-13:30午休"]}}
    )
    assert r.status_code == 422 and "09:00-23:00" in r.json()["message"]
    assert shop_snapshot(api, SHOP_A) == before


def test_questionnaire_with_unclear_hours_clears_them_and_warns(api: Api):
    """重新答第 13 题、写成看不懂的：旧时段清空（不知道客服在不在就不全自动），
    响应里提醒去设置页填。"""
    r = api.post(CONFIG, "merchant_a", json={"changes": {"service_hours": ["9:00-23:00"]}})
    assert r.status_code == 200, r.text
    r = api.post(
        f"/v1/shops/{SHOP_A}/questionnaire",
        "merchant_a",
        json={"answers": {"13": "9:00-23:00，周末也是"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["policies"] == ["service_hours"] and body["config"] == ["service_hours"]
    [warning] = body["warnings"]
    assert "设置" in warning and "09:00-23:00" in warning
    status = api.get(f"/v1/shops/{SHOP_A}/rollout/status", "merchant_a").json()
    assert status["auto_schedule"]["service_hours"] == []
