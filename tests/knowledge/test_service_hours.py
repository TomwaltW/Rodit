"""客服工作时间（附录 C 第 13 题；§15 Phase 3 全自动先在非工作时间试行）：问卷回答和设置页的写法
→ ShopConfig.service_hours（北京时间 "HH:MM-HH:MM"）。商家能改；看不懂时不猜（猜错了全自动会在
客服在岗时回复）。全部合成数据，内存仓储，不联网。"""

from __future__ import annotations

from typing import Any

import pytest

from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.config import EngineConfig
from rodit.core.errors import ActionNotAllowed
from rodit.inmemory import InMemoryKnowledgeRepository
from rodit.knowledge.admin import (
    LOCKED_CONFIG_FIELDS,
    MERCHANT_CONFIG_FIELDS,
    KnowledgeAdmin,
)
from rodit.knowledge.questionnaire import (
    QUESTIONNAIRE_BY_ID,
    apply_questionnaire,
    parse_service_hours_text,
)
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import DEFAULT_NOW, SHOP_A, SHOP_B, DemoWorld

MERCHANT = "owner_a"


# ================================================================ 写法


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("9:00-23:00", ["09:00-23:00"]),
        ("9:00–23:00", ["09:00-23:00"]),  # 问卷里常见的长横线
        ("09：00～23：00", ["09:00-23:00"]),  # 全角
        ("9:30至21:30", ["09:30-21:30"]),
        ("9 点到 22 点", ["09:00-22:00"]),
        ("9-23", ["09:00-23:00"]),
        ("8点-24点", ["08:00-24:00"]),
        ("18点到0点", ["18:00-24:00"]),
        ("早9点到晚11点", ["09:00-23:00"]),
        ("早上九点到晚上十一点半", ["09:00-23:30"]),
        ("早8点半到晚10点", ["08:30-22:00"]),
        ("上午9点到下午6点", ["09:00-18:00"]),
        ("下午1点到5点", ["13:00-17:00"]),  # 结束跟着开始算下午
        ("中午12点到晚上12点", ["12:00-24:00"]),
        ("晚上10点到凌晨2点", ["22:00-02:00"]),  # 跨午夜
        ("22:00-02:00", ["22:00-02:00"]),
        ("早9晚11", ["09:00-23:00"]),
        ("8:00-12:00，13:30-22:00", ["08:00-12:00", "13:30-22:00"]),
        ("8:00-12:00、8:00-12:00", ["08:00-12:00"]),  # 去重
        ("9:00-23:00 左右", ["09:00-23:00"]),
        ("全天在线", ["00:00-24:00"]),
        ("24 小时", ["00:00-24:00"]),
    ],
)
def test_parse_common_forms(text: str, expected: list[str]) -> None:
    assert parse_service_hours_text(text) == expected


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("9点到11点", "上午还是晚上"),  # 11 点是上午还是晚上？不猜
        ("早9点到11点", "上午还是晚上"),
        ("周一至周五 9:00-18:00", "星期"),
        ("工作日 9:00-18:00，节假日休息", "星期"),
        ("看情况", "没有认出"),
        ("", "没有填写"),
        ("   ", "没有填写"),
        ("25:00-26:00", "时间不对"),
        ("9:00-9:00", "不能相同"),
        ("9:75-23:00", "时间不对"),
        ("9:00-23:00 " + "备注" * 150, "最多"),
        ("1:00-2:00,3:00-4:00,5:00-6:00,7:00-8:00,9:00-10:00,11:00-12:00,13:00-14:00", "最多 6"),
    ],
)
def test_unclear_forms_are_refused_with_a_reason(text: str, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        parse_service_hours_text(text)


# ================================================================ 问卷第 13 题


def test_question_13_writes_policy_and_service_hours() -> None:
    assert QUESTIONNAIRE_BY_ID[13].config_fields == ("service_hours",)
    entries, patch = apply_questionnaire("s_new", {13: "早上9点到晚上11点"}, DEFAULT_NOW)
    assert [(e.key, e.value) for e in entries] == [("service_hours", "早上9点到晚上11点")]
    assert patch == {"service_hours": ["09:00-23:00"]}


@pytest.mark.parametrize("answer", ["周一到周五 9:00-18:00", "9点到11点", "看客服心情"])
def test_question_13_keeps_only_the_policy_text_when_unclear(answer: str) -> None:
    entries, patch = apply_questionnaire("s_new", {13: answer}, DEFAULT_NOW)
    assert [(e.key, e.value) for e in entries] == [("service_hours", answer)]
    assert patch == {}


# ================================================================ 设置页（商家能改）


class Kit:
    def __init__(self, repo: InMemoryKnowledgeRepository, world: DemoWorld) -> None:
        self.repo = repo
        self.audit = InMemoryAuditSink()
        self.admin = KnowledgeAdmin(repo, self.audit, FakeClock(world.now), config=EngineConfig())

    def events(self) -> list:
        return self.audit.list_events(SHOP_A, action=AuditAction.CONFIG_UPDATED.value)

    def stored(self, shop_id: str = SHOP_A) -> dict[str, Any]:
        shop = self.repo.get_shop(shop_id)
        assert shop is not None
        return dict(shop.config_overrides)


@pytest.fixture
def kit(repo: InMemoryKnowledgeRepository, world: DemoWorld) -> Kit:
    return Kit(repo, world)


def test_service_hours_are_merchant_editable_but_the_auto_switches_are_not() -> None:
    assert "service_hours" in MERCHANT_CONFIG_FIELDS
    assert {"auto_reply_enabled", "auto_reply_schedule", "semi_auto_intents"} <= set(
        LOCKED_CONFIG_FIELDS
    )


def test_lines_are_parsed_and_normalized(kit: Kit) -> None:
    snapshot = kit.admin.update_config(
        SHOP_A,
        {"service_hours": ["9:00-12:00", " 下午1点半到晚10点 ", "", "09:00-12:00"]},
        merchant=MERCHANT,
    )
    assert snapshot.changed == ["service_hours"]
    assert snapshot.effective.service_hours == ["09:00-12:00", "13:30-22:00"]
    assert kit.stored()["service_hours"] == ["09:00-12:00", "13:30-22:00"]
    [event] = kit.events()
    assert event.detail == {"changed": ["service_hours"], "removed": []}  # 不记取值
    # 写法不同、结果一样：没有实际改动，不写、不记审计
    again = kit.admin.update_config(
        SHOP_A, {"service_hours": ["09:00-12:00", "13:30-22:00"]}, merchant=MERCHANT
    )
    assert again.changed == [] and len(kit.events()) == 1
    assert "service_hours" not in kit.stored(SHOP_B)  # 别的店铺不受影响


def test_a_paragraph_is_accepted_like_the_questionnaire(kit: Kit) -> None:
    snapshot = kit.admin.update_config(
        SHOP_A, {"service_hours": "早9点到晚11点"}, merchant=MERCHANT
    )
    assert snapshot.effective.service_hours == ["09:00-23:00"]


def test_clearing_and_removing(kit: Kit) -> None:
    kit.admin.update_config(SHOP_A, {"service_hours": ["9:00-23:00"]}, merchant=MERCHANT)
    snapshot = kit.admin.update_config(SHOP_A, {"service_hours": []}, merchant=MERCHANT)
    assert snapshot.effective.service_hours == [] and kit.stored()["service_hours"] == []
    snapshot = kit.admin.update_config(SHOP_A, {"service_hours": None}, merchant=MERCHANT)
    assert snapshot.removed == ["service_hours"] and "service_hours" not in kit.stored()
    blank = kit.admin.update_config(SHOP_A, {"service_hours": "  "}, merchant=MERCHANT)
    assert blank.effective.service_hours == []


@pytest.mark.parametrize(
    ("value", "why"),
    [
        (["9点到11点"], "「9点到11点」.*上午还是晚上"),
        (["周一到周五 9-18"], "星期"),
        (["9:00-9:00"], "不能相同"),
        (["亲，早上九点到晚上十一点半都有客服在线回复哦"], "每一条最多 20 个字"),
        ([f"{h}:00-{h}:30" for h in range(1, 8)], "最多 6"),
        ("看情况", "没有认出"),
        ([9], "文字列表"),
        (True, "取值不合法"),
        (17, "文字列表"),
    ],
)
def test_bad_values_are_refused_with_a_clear_message(kit: Kit, value: Any, why: str) -> None:
    before = kit.stored()
    with pytest.raises(ValueError, match=why):
        kit.admin.update_config(SHOP_A, {"service_hours": value}, merchant=MERCHANT)
    assert kit.stored() == before and kit.events() == []


def test_locked_keys_in_the_same_request_refuse_everything(kit: Kit) -> None:
    before = kit.stored()
    for locked in ("auto_reply_schedule", "semi_auto_intents"):
        value = "always" if locked == "auto_reply_schedule" else ["shipping_policy"]
        with pytest.raises(ActionNotAllowed):
            kit.admin.update_config(
                SHOP_A, {"service_hours": ["9:00-23:00"], locked: value}, merchant=MERCHANT
            )
    assert kit.stored() == before and kit.events() == []
