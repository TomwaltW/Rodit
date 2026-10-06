"""上线节奏里的半自动开关和全自动生效时段（方案 §13.4 半自动 "该类型原样采纳率 ≥ 90%"、§5.6、
§15 Phase 3 先在非工作时间试行、§8.4 取消订购与在途操作互斥、§9.2 店铺隔离）。

复核人员按门槛开启（没过要强制并写原因），高风险类型永远不能开；关闭随时可以；只写
semi_auto_intents 一个键，审计记操作人（取自调用方）、门槛摘要、是否强制、原因。
数据全部合成（演示单元 + 手造推荐记录）。
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Any

import pytest

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit
from rodit.app.rollout import SemiAutoGateNotPassed, auto_schedule
from rodit.core.clock import CST
from rodit.core.enums import Intent, Platform, SuggestionStatus
from rodit.core.errors import ActionNotAllowed, ShopInactive
from rodit.core.models import ShopConfig, SuggestionRecord
from rodit.feedback.rollout import SEMI_AUTO_CHANGED
from rodit.testing.fixtures import SHOP_A, SHOP_B

S = SuggestionStatus
SHIP, INVOICE, PRICE = Intent.SHIPPING_POLICY, Intent.INVOICE_GIFT, Intent.PRICE_PROMO
REVIEWER = "qa_reviewer_01"
MERCHANT = "owner_a"
W38_START = datetime(2026, 9, 14, tzinfo=CST)  # 演示时钟在 2026-09-28（周一，W40）
W39_START = datetime(2026, 9, 21, tzinfo=CST)
PHONE = "13812345678"


@pytest.fixture
def service() -> AssistService:
    return AssistService(build_demo_unit(Platform.TAOBAO))


def add(
    service: AssistService,
    n: int,
    as_is: int,
    *,
    at: datetime,
    intent: Intent = SHIP,
    shop_id: str = SHOP_A,
) -> None:
    """n 个推荐组（每组一张卡片），前 as_is 个原样发送，其余编辑后发送。"""
    repo = service.unit.repos.suggestions
    for i in range(n):
        repo.add(
            SuggestionRecord(
                shop_id=shop_id,
                conversation_id=f"conv_s{i % 5}",
                msg_id=f"s_{intent.value}_{at:%m%d}_{shop_id}_{i}",
                question_text="什么时候发货",
                intent=intent,
                intent_confidence=0.95,
                text="亲，现货付款后 48 小时内发货哦~",
                status=S.SENT if i < as_is else S.EDITED_SENT,
                created_at=at + timedelta(hours=2, minutes=i),
            )
        )


def ready(service: AssistService, intent: Intent = SHIP, shop_id: str = SHOP_A) -> None:
    """该类型最近两个完整周都是 95% 原样采纳、每周 20 组：半自动门槛能过。"""
    add(service, 20, 19, at=W38_START, intent=intent, shop_id=shop_id)
    add(service, 20, 19, at=W39_START, intent=intent, shop_id=shop_id)


def overrides(service: AssistService, shop_id: str = SHOP_A) -> dict[str, Any]:
    shop = service.unit.repos.knowledge.get_shop(shop_id)
    assert shop is not None
    return dict(shop.config_overrides)


def events(service: AssistService, shop_id: str = SHOP_A):
    return service.unit.repos.audit.list_events(shop_id, action=SEMI_AUTO_CHANGED)


# ================================================================ 开启（复核人员，按门槛）


def test_enable_when_the_gate_passes(service: AssistService):
    ready(service)
    before = overrides(service)
    change = service.rollout.set_semi_auto(
        SHOP_A, "shipping_policy", True, operator=REVIEWER, role="reviewer"
    )
    assert change.changed and not change.forced and change.enabled
    assert change.gate is not None and change.gate.passed
    assert change.semi_auto_intents == [SHIP]
    after = overrides(service)
    assert after == {**before, "semi_auto_intents": ["shipping_policy"]}  # 只写了这一个键
    [event] = events(service)
    assert event.actor == REVIEWER
    assert event.detail == {
        "intent": "shipping_policy",
        "enabled": True,
        "role": "reviewer",
        "forced": False,
        "force_requested": False,
        "gate": {
            "stage": "semi_auto:shipping_policy",
            "passed": True,
            "failed": [],
            "unchecked": [],
            "values": {
                "periods": 2,
                "period_1.group_as_is_rate": 0.95,
                "period_2.group_as_is_rate": 0.95,
            },
        },
        "note": None,
        "intents": ["shipping_policy"],
    }
    # 开了以后，这家店这类问题的推荐填进输入框（半自动）
    result = service.ask_standalone(
        shop_id=SHOP_A, session_id="w1", question="什么时候发货", request_id="q1"
    )
    assert result is not None and [c.prefill for c in result.suggestions] == [True]


def test_gate_not_passed_is_refused_without_force(service: AssistService):
    add(service, 20, 17, at=W38_START)  # 85%
    add(service, 20, 20, at=W39_START)
    before = overrides(service)
    with pytest.raises(SemiAutoGateNotPassed) as info:
        service.rollout.set_semi_auto(SHOP_A, SHIP, True, operator=REVIEWER)
    gate = info.value.gate
    assert not gate.passed
    assert [c.name for c in gate.checks if not c.passed] == ["period_1.group_as_is_rate"]
    assert overrides(service) == before and events(service) == []


def test_force_needs_a_note_and_is_checked_before_the_gate(service: AssistService):
    for note in (None, "", "   "):
        with pytest.raises(ValueError, match="写明原因"):
            service.rollout.set_semi_auto(
                SHOP_A, SHIP, True, operator=REVIEWER, force=True, note=note
            )
    assert "semi_auto_intents" not in overrides(service) and events(service) == []


def test_forced_enable_records_the_minimized_reason(service: AssistService):
    change = service.rollout.set_semi_auto(
        SHOP_A,
        SHIP,
        True,
        operator=REVIEWER,
        role="reviewer",
        force=True,
        note=f"种子商家要求先试，联系人 {PHONE}",
    )
    assert change.changed and change.forced and change.gate is not None
    assert not change.gate.passed
    [event] = events(service)
    assert event.detail["forced"] is True and event.detail["force_requested"] is True
    assert event.detail["gate"]["passed"] is False
    assert PHONE not in event.detail["note"] and "种子商家要求先试" in event.detail["note"]


def test_force_when_the_gate_passes_is_not_recorded_as_forced(service: AssistService):
    ready(service)
    change = service.rollout.set_semi_auto(
        SHOP_A, SHIP, True, operator=REVIEWER, force=True, note="顺手"
    )
    assert change.changed and not change.forced
    assert events(service)[0].detail["force_requested"] is True


@pytest.mark.parametrize("intent", [PRICE, Intent.COMPLAINT, Intent.QUALITY_ISSUE])
def test_high_risk_intents_can_never_be_enabled(service: AssistService, intent: Intent):
    ready(service, intent)
    for force in (False, True):
        with pytest.raises(ActionNotAllowed, match="高风险"):
            service.rollout.set_semi_auto(
                SHOP_A, intent, True, operator=REVIEWER, force=force, note="x"
            )
    assert "semi_auto_intents" not in overrides(service) and events(service) == []


@pytest.mark.parametrize("intent", [Intent.OFF_PLATFORM, Intent.OTHER])
def test_non_answer_intents_cannot_be_enabled(service: AssistService, intent: Intent):
    with pytest.raises(ActionNotAllowed, match="规则 6"):
        service.rollout.set_semi_auto(SHOP_A, intent, True, operator=REVIEWER, force=True, note="x")


def test_intents_are_kept_in_intent_order(service: AssistService):
    ready(service, INVOICE)
    ready(service, SHIP)
    ro = service.rollout
    ro.set_semi_auto(SHOP_A, INVOICE, True, operator=REVIEWER)
    change = ro.set_semi_auto(SHOP_A, SHIP, True, operator=REVIEWER)
    assert change.semi_auto_intents == [SHIP, INVOICE]
    assert overrides(service)["semi_auto_intents"] == ["shipping_policy", "invoice_gift"]


def test_bad_input(service: AssistService):
    ro = service.rollout
    with pytest.raises(ValueError):
        ro.set_semi_auto(SHOP_A, "nope", True, operator=REVIEWER)
    with pytest.raises(ValueError, match="操作人"):
        ro.set_semi_auto(SHOP_A, SHIP, True, operator="  ")
    with pytest.raises(ValueError, match="enabled"):
        ro.set_semi_auto(SHOP_A, SHIP, "yes", operator=REVIEWER)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="最多"):
        ro.set_semi_auto(SHOP_A, SHIP, True, operator=REVIEWER, force=True, note="长" * 501)


# ================================================================ 关闭（随时可以）


def test_disable_is_always_allowed_and_noops_write_nothing(service: AssistService):
    ro = service.rollout
    noop = ro.set_semi_auto(SHOP_A, SHIP, False, operator=MERCHANT, role="merchant")
    assert not noop.changed and noop.semi_auto_intents == [] and events(service) == []
    # 配置里被直接写进了高风险类型：关闭照样可以（不看门槛、不看类型）
    service.unit.repos.knowledge.update_shop_config(
        SHOP_A, {"semi_auto_intents": ["price_promo", "shipping_policy"]}
    )
    change = ro.set_semi_auto(SHOP_A, PRICE, False, operator=MERCHANT, role="merchant", force=True)
    assert change.changed and change.gate is None and not change.forced
    assert change.semi_auto_intents == [SHIP]
    [event] = events(service)
    assert event.actor == MERCHANT
    assert event.detail["enabled"] is False and event.detail["force_requested"] is False
    assert event.detail["intents"] == ["shipping_policy"]
    again = ro.set_semi_auto(SHOP_A, SHIP, True, operator=REVIEWER)
    assert not again.changed and len(events(service)) == 1  # 已经开着：不检查门槛、不写
    last = ro.set_semi_auto(SHOP_A, SHIP, False, operator=REVIEWER)
    assert last.semi_auto_intents == [] and overrides(service)["semi_auto_intents"] == []


# ================================================================ 店铺隔离（§9.2）


def test_other_shops_records_do_not_count_and_changes_stay_in_the_shop(service: AssistService):
    ready(service, shop_id=SHOP_B)  # 店铺 B 达标，店铺 A 没有数据
    with pytest.raises(SemiAutoGateNotPassed):
        service.rollout.set_semi_auto(SHOP_A, SHIP, True, operator=REVIEWER)
    change = service.rollout.set_semi_auto(SHOP_B, SHIP, True, operator="qa_reviewer_02")
    assert change.changed and not change.forced
    assert "semi_auto_intents" not in overrides(service, SHOP_A)
    assert events(service) == [] and len(events(service, SHOP_B)) == 1
    status_a = service.rollout.status(SHOP_A)
    assert status_a.semi_auto is not None and status_a.semi_auto.enabled == []


# ================================================================ 上线状态


def test_status_reports_semi_auto_gates_and_the_schedule(service: AssistService):
    ready(service)
    status = service.rollout.status(SHOP_A)
    assert status.semi_auto is not None
    rows = {r.intent: r for r in status.semi_auto.intents}
    assert rows[SHIP].gate is not None and rows[SHIP].gate.passed and not rows[SHIP].enabled
    assert not rows[PRICE].allowed and rows[PRICE].gate is None
    schedule = status.auto_schedule
    assert schedule is not None
    assert schedule.model_dump() == {
        "auto_reply_enabled": False,
        "auto_reply_schedule": "off_hours",
        "service_hours": [],
        "timezone": "北京时间（UTC+8）",
        "in_service_hours": None,
        "auto_active_now": False,
        "auto_gate": None,
        "next_switch_at": None,
    }
    service.unit.repos.knowledge.update_shop_config(
        SHOP_A, {"auto_reply_enabled": True, "service_hours": ["09:00-23:00"]}
    )
    schedule = service.rollout.status(SHOP_A).auto_schedule
    assert schedule is not None
    assert schedule.in_service_hours is True and schedule.auto_active_now is False
    assert schedule.auto_gate == "service_hours"
    assert schedule.next_switch_at == datetime(2026, 9, 28, 23, 0, tzinfo=CST)


@pytest.mark.parametrize(
    ("kw", "now", "active", "gate"),
    [
        ({"service_hours": ["09:00-23:00"]}, datetime(2026, 9, 28, 23, 30, tzinfo=CST), True, None),
        ({}, datetime(2026, 9, 28, 23, 30, tzinfo=CST), False, "no_service_hours"),
        ({"auto_reply_schedule": "always"}, datetime(2026, 9, 28, 10, tzinfo=CST), True, None),
        (
            {"auto_reply_enabled": False, "service_hours": ["09:00-23:00"]},
            datetime(2026, 9, 28, 23, 30, tzinfo=CST),
            False,
            None,
        ),
    ],
)
def test_auto_schedule_follows_the_engine_rule(kw, now, active, gate):
    kw = {"auto_reply_enabled": True, **kw}
    schedule = auto_schedule(ShopConfig(**kw), now)
    assert schedule.auto_active_now is active and schedule.auto_gate == gate


# ================================================================ 取消订购（§8.4）


def test_semi_auto_after_revoke_does_not_recreate_the_shop(service: AssistService):
    service.revoke_shop(SHOP_A)
    with pytest.raises(ShopInactive):
        service.rollout.set_semi_auto(SHOP_A, SHIP, False, operator=MERCHANT)
    with pytest.raises(ShopInactive):
        service.rollout.set_semi_auto(SHOP_A, SHIP, True, operator=REVIEWER, force=True, note="x")
    assert service.unit.repos.knowledge.get_shop(SHOP_A) is None
    assert service.unit.repos.audit.list_events(SHOP_A) == []


def test_revoke_waits_for_an_in_flight_semi_auto_change(service: AssistService):
    repo = service.unit.repos.knowledge
    inflight, revoke_started, revoke_done = (threading.Event() for _ in range(3))
    original = repo.update_shop_config
    seen: list[bool] = []

    def slow(*args: Any, **kwargs: Any) -> Any:
        inflight.set()
        assert revoke_started.wait(5)
        revoke_done.wait(0.3)  # 取消订购要等在途操作做完：这里等不到，超时后继续
        seen.append(revoke_done.is_set())
        return original(*args, **kwargs)

    repo.update_shop_config = slow  # type: ignore[method-assign]
    errors: list[BaseException] = []

    def write() -> None:
        try:
            service.rollout.set_semi_auto(
                SHOP_A, SHIP, True, operator=REVIEWER, force=True, note="试行"
            )
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def revoke() -> None:
        assert inflight.wait(5)
        revoke_started.set()
        service.revoke_shop(SHOP_A)
        revoke_done.set()

    threads = [threading.Thread(target=write), threading.Thread(target=revoke)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert revoke_done.is_set() and not errors and seen == [False]
    assert repo.get_shop(SHOP_A) is None  # 在途的配置写入没有把删掉的店铺写回来
    assert service.unit.repos.audit.list_events(SHOP_A) == []
    assert repo.get_shop(SHOP_B) is not None


def test_concurrent_enables_of_different_intents_both_stick(service: AssistService):
    """两个复核人员同时开不同的类型：读改写串行，两个都留下。"""
    repo = service.unit.repos.knowledge
    original = repo.update_shop_config
    first_in = threading.Event()
    release = threading.Event()

    def slow(shop_id: str, changes: Any) -> Any:
        if not first_in.is_set():
            first_in.set()
            release.wait(2)
        return original(shop_id, changes)

    repo.update_shop_config = slow  # type: ignore[method-assign]
    ro = service.rollout
    t1 = threading.Thread(
        target=lambda: ro.set_semi_auto(SHOP_A, SHIP, True, operator=REVIEWER, force=True, note="a")
    )
    t2 = threading.Thread(
        target=lambda: ro.set_semi_auto(
            SHOP_A, INVOICE, True, operator="qa_reviewer_02", force=True, note="b"
        )
    )
    t1.start()
    assert first_in.wait(5)
    t2.start()
    t2.join(0.3)  # 没有串行的话，t2 这时已经读到旧列表、写完了，t1 再写就把它覆盖掉
    release.set()
    t1.join(10)
    t2.join(10)
    assert overrides(service)["semi_auto_intents"] == ["shipping_policy", "invoice_gift"]
    assert len(events(service)) == 2


def test_change_is_undone_when_its_audit_cannot_be_written(service: AssistService, monkeypatch):
    """开关必须留痕：审计写不进去就撤回（覆盖项恢复原样），重试走完整流程、写上审计。"""
    audit = service.unit.repos.audit
    original = audit.record
    calls = {"n": 0}

    def flaky(event):
        if event.action == SEMI_AUTO_CHANGED and calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("审计库连不上")
        return original(event)

    monkeypatch.setattr(audit, "record", flaky)
    ro = service.rollout
    with pytest.raises(RuntimeError):
        ro.set_semi_auto(SHOP_A, SHIP, True, operator=REVIEWER, force=True, note="试行")
    assert "semi_auto_intents" not in overrides(service)  # 键删掉了，回到默认
    change = ro.set_semi_auto(SHOP_A, SHIP, True, operator=REVIEWER, force=True, note="试行")
    assert change.changed and overrides(service)["semi_auto_intents"] == ["shipping_policy"]
    assert len(events(service)) == 1
