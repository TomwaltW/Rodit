"""上线节奏与每周复核的应用服务（方案 §13.4 影子对比与阶段切换、§13.5 每周抽样复核、§9.2 查看
原文留访问记录、§8.4 取消订购与在途操作互斥）。数据全部合成（演示单元 + 手造推荐记录）。"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Any

import pytest

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.app.rollout import RolloutGateNotPassed
from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import (
    Intent,
    Mode,
    Platform,
    QualityReviewStatus,
    SuggestionStatus,
)
from rodit.core.errors import ActionNotAllowed, ShopInactive
from rodit.core.models import QualityScores, SuggestionRecord
from rodit.feedback import GateStatus, RolloutAdvice, RolloutStage
from rodit.testing.fixtures import SHOP_A, SHOP_B, SHOP_D

S = SuggestionStatus
Q = QualityReviewStatus
SHIP, SIZE, INVOICE = Intent.SHIPPING_POLICY, Intent.SIZE_ADVICE, Intent.INVOICE_GIFT
REVIEWER = "qa_reviewer_01"
W39 = "2026-W39"
W39_START = datetime(2026, 9, 21, tzinfo=CST)  # 演示时钟在 2026-09-28（周一，W40）
PHONE = "13812345678"
FULL = {"factual": 2, "relevance": 2, "compliance": 2, "expression": 2}


@pytest.fixture
def service() -> AssistService:
    return AssistService(build_demo_unit(Platform.TAOBAO))


def add(
    service: AssistService,
    n: int,
    *,
    intent: Intent = SHIP,
    status: SuggestionStatus = S.SHOWN,
    at: datetime = W39_START,
    shop_id: str = SHOP_A,
    question: str = "什么时候发货",
    text: str = "亲，现货付款后 48 小时内发货哦~",
) -> list[SuggestionRecord]:
    """直接用构造函数写推荐记录（绕过引擎，所以文字没有经过写入时的最小化）。"""
    repo = service.unit.repos.suggestions
    return [
        repo.add(
            SuggestionRecord(
                shop_id=shop_id,
                conversation_id=f"conv_r{i % 7}",
                msg_id=f"r_{intent.value}_{status.value}_{at:%m%d%H}_{i}",
                question_text=question,
                intent=intent,
                intent_confidence=0.9,
                text=text,
                evidence_ids=["faq:2"],
                status=status,
                created_at=at + timedelta(hours=1, minutes=i),
            )
        )
        for i in range(n)
    ]


def events(service: AssistService, action: AuditAction, shop_id: str = SHOP_A):
    return service.unit.repos.audit.list_events(shop_id, action=action.value)


def overrides(service: AssistService, shop_id: str = SHOP_A) -> dict[str, Any]:
    shop = service.unit.repos.knowledge.get_shop(shop_id)
    assert shop is not None
    return dict(shop.config_overrides)


def shadow_ready(service: AssistService, n: int = 25) -> None:
    """影子模式下 n 条推荐都抽中、都打满分：影子 → 智能辅助的门槛能过。"""
    add(service, n, status=S.SHADOW)
    ro = service.rollout
    ro.sample(SHOP_A, reviewer=REVIEWER, period=W39)
    for q in service.unit.repos.quality_reviews.list(SHOP_A, period=W39):
        ro.score(SHOP_A, q.id, status="scored", scores=FULL, reviewer=REVIEWER)


# ================================================================ 每周抽样（§13.5）


def test_sample_defaults_to_previous_week_and_audits_counts_only(service: AssistService):
    add(service, 40, intent=SHIP)
    add(service, 20, intent=SIZE, status=S.SENT)
    add(service, 10, intent=INVOICE, status=S.EXPIRED)
    add(service, 5, at=W39_START + timedelta(days=7))  # 本周（W40）
    add(service, 5, shop_id=SHOP_B)
    result = service.rollout.sample(SHOP_A, reviewer=REVIEWER)
    assert result.period == W39 and (result.since, result.until) == (
        W39_START,
        W39_START + timedelta(days=7),
    )
    assert (result.eligible, result.existing, result.added, result.total) == (70, 0, 50, 50)
    assert result.by_intent == {SIZE: 14, SHIP: 29, INVOICE: 7}
    reviews = service.unit.repos.quality_reviews.list(SHOP_A, period=W39)
    assert len(reviews) == 50 and {q.status for q in reviews} == {Q.PENDING}
    assert service.unit.repos.quality_reviews.list(SHOP_B) == []
    [event] = events(service, AuditAction.QUALITY_SAMPLED)
    assert event.actor == REVIEWER
    assert event.detail == {
        "period": W39,
        "size": 50,
        "eligible": 70,
        "existing": 0,
        "added": 50,
        "total": 50,
        "by_intent": {"size_advice": 14, "shipping_policy": 29, "invoice_gift": 7},
    }


def test_resample_is_idempotent_and_tops_up_later_suggestions(service: AssistService):
    ro, repo = service.rollout, service.unit.repos.quality_reviews
    add(service, 3, intent=SHIP)
    first = ro.sample(SHOP_A, reviewer=REVIEWER, period=W39, size=10)
    assert (first.added, first.total) == (3, 3)
    before = [(q.id, q.suggestion_id) for q in repo.list(SHOP_A, period=W39)]
    again = ro.sample(SHOP_A, reviewer="qa_reviewer_02", period=W39, size=10)
    assert (again.existing, again.added, again.total) == (3, 0, 3)
    assert [(q.id, q.suggestion_id) for q in repo.list(SHOP_A, period=W39)] == before
    assert len(events(service, AuditAction.QUALITY_SAMPLED)) == 1  # 没抽到新的不留痕
    # 同一周又有推荐进来（晚到的平台消息）：补到 10 条，原来的一条不少
    add(service, 20, intent=SIZE, at=W39_START + timedelta(days=2))
    top_up = ro.sample(SHOP_A, reviewer=REVIEWER, period=W39, size=10)
    assert (top_up.existing, top_up.added, top_up.total) == (3, 7, 10)
    after = [(q.id, q.suggestion_id) for q in repo.list(SHOP_A, period=W39)]
    assert after[:3] == before
    # 已打分的样本在补抽时原样保留
    ro.score(SHOP_A, before[0][0], status="scored", scores=FULL, reviewer=REVIEWER)
    ro.sample(SHOP_A, reviewer=REVIEWER, period=W39, size=12)
    assert repo.get(SHOP_A, before[0][0]).status == Q.SCORED


def test_sample_rejects_bad_periods_and_sizes(service: AssistService):
    ro = service.rollout
    for kwargs in (
        {"period": "2026-W41"},  # 还没开始
        {"period": "2025-W53"},  # 这一年没有第 53 周
        {"period": "2026-40"},
        {"size": 0},
        {"size": 201},
    ):
        with pytest.raises(ValueError):
            ro.sample(SHOP_A, reviewer=REVIEWER, **kwargs)
    # 本周（还没结束）可以先抽
    add(service, 2, at=W39_START + timedelta(days=7))
    assert ro.sample(SHOP_A, reviewer=REVIEWER, period="2026-W40").added == 2
    assert service.unit.repos.audit.list_events(SHOP_B) == []


# ================================================================ 复核列表（§9.2 访问记录）


def test_review_list_returns_masked_texts_and_writes_an_access_log(service: AssistService):
    """推荐写入时本应已做最小化；这里用构造函数写入带手机号的问题，列表照样打码。"""
    [record] = add(service, 1, question=f"我手机 {PHONE}，什么时候发货")
    add(service, 2, intent=SIZE, status=S.REJECTED, question="165 穿什么码")
    ro = service.rollout
    assert ro.list_reviews(SHOP_A, viewer=REVIEWER).count == 0
    assert events(service, AuditAction.QUALITY_VIEWED) == []  # 没返回内容，不记
    ro.sample(SHOP_A, reviewer=REVIEWER)
    listed = ro.list_reviews(SHOP_A, viewer=REVIEWER)
    assert listed.period == W39 and listed.count == 3
    item = next(i for i in listed.items if i.suggestion_id == record.id)
    assert item.suggestion is not None
    assert item.suggestion.question_text == "我手机 [手机号]，什么时候发货"
    assert item.suggestion.text == record.text and item.suggestion.evidence_ids == ["faq:2"]
    assert item.suggestion.status == S.SHOWN and item.suggestion.intent_confidence == 0.9
    assert PHONE not in listed.model_dump_json()
    [viewed] = events(service, AuditAction.QUALITY_VIEWED)
    assert viewed.actor == REVIEWER
    assert viewed.detail == {
        "period": W39,
        "status": None,
        "count": 3,
        "review_ids": [i.id for i in listed.items],
    }
    # 每次返回内容都留一条访问记录；按状态筛选
    ro.score(SHOP_A, listed.items[0].id, status="skipped", reviewer=REVIEWER)
    pending = ro.list_reviews(SHOP_A, viewer="qa_reviewer_02", status="pending")
    assert pending.count == 2 and {i.status for i in pending.items} == {Q.PENDING}
    views = events(service, AuditAction.QUALITY_VIEWED)
    assert len(views) == 2 and views[1].actor == "qa_reviewer_02"
    assert views[1].detail["status"] == "pending" and views[1].detail["count"] == 2
    for e in service.unit.repos.audit.list_events(SHOP_A):
        assert "什么时候发货" not in str(e.detail) and PHONE not in str(e.detail)


# ================================================================ 打分（§13.2）


def test_score_minimizes_the_note_and_audits_scores_only(service: AssistService):
    add(service, 2)
    ro, repo = service.rollout, service.unit.repos.quality_reviews
    ro.sample(SHOP_A, reviewer=REVIEWER, period=W39)
    first, second = repo.list(SHOP_A, period=W39)
    done = ro.score(
        SHOP_A,
        first.id,
        status="scored",
        scores=QualityScores(factual=2, relevance=1, compliance=2, expression=1),
        note=f"  回答对了，买家留的电话 {PHONE} 不该出现在推荐里 \x00 ",
        reviewer=REVIEWER,
    )
    assert done.status == Q.SCORED and done.reviewer == REVIEWER
    assert done.reviewed_at == service.unit.clock.now()
    assert done.note == "回答对了，买家留的电话 [手机号] 不该出现在推荐里"
    assert repo.get(SHOP_A, first.id).note == done.note
    [event] = events(service, AuditAction.QUALITY_SCORED)
    assert event.actor == REVIEWER
    assert event.detail == {
        "review_id": first.id,
        "suggestion_id": first.suggestion_id,
        "period": W39,
        "intent": "shipping_policy",
        "status": "scored",
        "previous_status": "pending",
        "scores": {"factual": 2, "relevance": 1, "compliance": 2, "expression": 1},
        "has_note": True,
    }
    # 改判：后一次覆盖前一次，历史在审计里
    redo = ro.score(SHOP_A, first.id, status="skipped", reviewer="qa_reviewer_02", note="  ")
    assert redo.status == Q.SKIPPED and redo.scores is None and redo.note is None
    last = events(service, AuditAction.QUALITY_SCORED)[-1]
    assert last.detail["previous_status"] == "scored" and last.detail["has_note"] is False
    # 不合法的结果
    for kwargs in (
        {"status": "pending"},
        {"status": "scored"},  # 没有分数
        {"status": "skipped", "scores": FULL},
        {"status": "scored", "scores": {**FULL, "factual": 3}},
        {"status": "scored", "scores": FULL, "note": "x" * 501},
    ):
        with pytest.raises(ValueError):
            ro.score(SHOP_A, second.id, reviewer=REVIEWER, **kwargs)
    assert repo.get(SHOP_A, second.id).status == Q.PENDING


def test_review_of_another_shop_is_not_found(service: AssistService):
    """§9.2：店铺 A 用店铺 B 的样本 ID 打分、列表里也看不到。"""
    add(service, 1, shop_id=SHOP_B)
    service.rollout.sample(SHOP_B, reviewer="qa_reviewer_02", period=W39)
    [b_review] = service.unit.repos.quality_reviews.list(SHOP_B)
    with pytest.raises(KeyError):
        service.rollout.score(SHOP_A, b_review.id, status="skipped", reviewer=REVIEWER)
    assert service.unit.repos.quality_reviews.get(SHOP_B, b_review.id).status == Q.PENDING
    assert service.rollout.list_reviews(SHOP_A, viewer=REVIEWER, period=W39).count == 0
    assert events(service, AuditAction.QUALITY_SCORED) == []
    assert service.rollout.report(SHOP_A, period=W39).sampled == 0


def test_report_after_scoring(service: AssistService):
    add(service, 20)
    add(service, 5, intent=SIZE)
    ro = service.rollout
    ro.sample(SHOP_A, reviewer=REVIEWER)
    reviews = service.unit.repos.quality_reviews.list(SHOP_A, period=W39)
    for q in reviews[:21]:
        ro.score(SHOP_A, q.id, status="scored", scores=FULL, reviewer=REVIEWER)
    ro.score(SHOP_A, reviews[21].id, status="skipped", reviewer=REVIEWER, note=f"电话 {PHONE}")
    report = ro.report(SHOP_A)
    assert report.period == W39
    assert (report.sampled, report.scored, report.skipped, report.pending) == (25, 21, 1, 3)
    assert report.gate.status == GateStatus.PASSED
    assert report.dimensions["factual"].full_rate == 1.0
    assert set(report.by_intent) == {SIZE, SHIP}
    assert PHONE not in report.model_dump_json()
    assert ro.report(SHOP_A, period="2026-W38").gate.status == GateStatus.INSUFFICIENT


# ================================================================ 阶段切换（§13.4）


def test_switch_to_shadow_is_always_allowed_and_noop_is_not_audited(service: AssistService):
    ro = service.rollout
    change = ro.set_stage(SHOP_A, "shadow", operator="owner_a", role="merchant")
    assert change.changed and change.previous == RolloutStage.ASSIST
    assert change.stage == RolloutStage.SHADOW and change.gate is None
    assert overrides(service)["shadow_mode"] is True
    assert overrides(service)["display_name"] == "小雅女装旗舰店"  # 其他覆盖项不动
    [event] = events(service, AuditAction.ROLLOUT_STAGE)
    assert event.actor == "owner_a"
    assert event.detail == {
        "from": "assist",
        "to": "shadow",
        "role": "merchant",
        "forced": False,
        "force_requested": False,
        "gate": None,
        "note": None,
    }
    again = ro.set_stage(SHOP_A, "shadow", operator="owner_a")
    assert not again.changed and again.stage == RolloutStage.SHADOW
    assert len(events(service, AuditAction.ROLLOUT_STAGE)) == 1
    # 引擎按新配置走：生成但不展示（§13.4）
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="st1"))
    assert result is not None and result.mode == Mode.SHADOW
    assert {r.status for r in service.unit.repos.suggestions.list(SHOP_A)} == {S.SHADOW}


def test_force_does_not_matter_when_switching_to_shadow(service: AssistService):
    """切回影子是安全方向，随时可以：带 force 也不要求写原因，也不算强制。"""
    change = service.rollout.set_stage(SHOP_A, "shadow", operator=REVIEWER, force=True)
    assert change.changed and not change.forced
    [event] = events(service, AuditAction.ROLLOUT_STAGE)
    assert event.detail["forced"] is False and event.detail["force_requested"] is False


def test_shadow_to_assist_is_refused_until_the_gate_passes(service: AssistService):
    ro = service.rollout
    ro.set_stage(SHOP_A, "shadow", operator=REVIEWER)
    before = overrides(service)
    with pytest.raises(RolloutGateNotPassed) as exc:
        ro.set_stage(SHOP_A, "assist", operator=REVIEWER)
    gate = exc.value.gate
    assert not gate.passed and "shadow_review.scored" in [c.name for c in gate.checks]
    assert gate.unchecked == ["test_set"]
    assert isinstance(exc.value, ActionNotAllowed)  # 接口层映射成 409
    # 强制切换必须写原因
    with pytest.raises(ValueError):
        ro.set_stage(SHOP_A, "assist", operator=REVIEWER, force=True)
    with pytest.raises(ValueError):
        ro.set_stage(SHOP_A, "assist", operator=REVIEWER, force=True, note=" \n ")
    assert overrides(service) == before
    assert len(events(service, AuditAction.ROLLOUT_STAGE)) == 1  # 只有切到影子那一条

    change = ro.set_stage(
        SHOP_A, "assist", operator=REVIEWER, force=True, note=f"测试集已过，负责人 {PHONE}"
    )
    assert change.changed and change.forced and change.stage == RolloutStage.ASSIST
    assert overrides(service)["shadow_mode"] is False
    event = events(service, AuditAction.ROLLOUT_STAGE)[-1]
    assert event.detail["forced"] is True and event.detail["force_requested"] is True
    assert event.detail["note"] == "测试集已过，负责人 [手机号]"
    assert event.detail["gate"]["passed"] is False
    assert event.detail["gate"]["failed"] == [  # 一条都还没打分
        "shadow_review.scored",
        "shadow_review.factual_full_rate",
        "shadow_review.compliance_full_rate",
    ]
    assert event.detail["gate"]["unchecked"] == ["test_set"]
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="st2"))
    assert result is not None and result.mode == Mode.ASSIST and result.suggestions


def test_shadow_to_assist_without_force_when_reviews_pass(service: AssistService):
    ro = service.rollout
    ro.set_stage(SHOP_A, "shadow", operator=REVIEWER)
    shadow_ready(service)
    status = ro.status(SHOP_A)
    assert status.advice == RolloutAdvice.READY_FOR_ASSIST
    assert status.shadow_quality.scored == 25 and status.shadow_gate.passed
    change = ro.set_stage(SHOP_A, "assist", operator=REVIEWER, role="reviewer")
    assert change.changed and not change.forced and change.gate == status.shadow_gate
    event = events(service, AuditAction.ROLLOUT_STAGE)[-1]
    assert event.detail["gate"]["passed"] is True and event.detail["forced"] is False
    assert event.detail["gate"]["values"]["shadow_review.scored"] == 25
    assert ro.status(SHOP_A).stage == RolloutStage.ASSIST


def test_shadow_reviews_of_a_bad_week_block_the_switch(service: AssistService):
    ro = service.rollout
    ro.set_stage(SHOP_A, "shadow", operator=REVIEWER)
    add(service, 25, status=S.SHADOW)
    ro.sample(SHOP_A, reviewer=REVIEWER, period=W39)
    for i, q in enumerate(service.unit.repos.quality_reviews.list(SHOP_A, period=W39)):
        scores = {**FULL, "compliance": 1} if i < 2 else FULL  # 23/25 = 92%
        ro.score(SHOP_A, q.id, status="scored", scores=scores, reviewer=REVIEWER)
    with pytest.raises(RolloutGateNotPassed) as exc:
        ro.set_stage(SHOP_A, "assist", operator=REVIEWER)
    assert [c.name for c in exc.value.gate.checks if not c.passed] == [
        "shadow_review.compliance_full_rate"
    ]
    assert ro.status(SHOP_A).blocking == ["shadow_review.compliance_full_rate"]


def test_auto_reply_shop_can_only_go_back_to_shadow(service: AssistService):
    repo = service.unit.repos.knowledge
    shop = repo.get_shop(SHOP_A)
    repo.upsert_shop(
        shop.model_copy(
            update={"config_overrides": {**shop.config_overrides, "auto_reply_enabled": True}}
        )
    )
    ro = service.rollout
    assert ro.status(SHOP_A).stage == RolloutStage.AUTO
    with pytest.raises(ValueError):
        ro.set_stage(SHOP_A, "auto", operator=REVIEWER)
    assert ro.set_stage(SHOP_A, "shadow", operator=REVIEWER).previous == RolloutStage.AUTO
    shadow_ready(service)
    # 关掉影子模式会直接进全自动：不在这里做
    with pytest.raises(ActionNotAllowed):
        ro.set_stage(SHOP_A, "assist", operator=REVIEWER, force=True, note="试试")
    assert overrides(service)["shadow_mode"] is True


# ================================================================ 上线状态与影子对比（演示单元）


def _echo(text: str, msg_id: str, send_time_ms: int) -> dict:
    """客服发出的消息回声（淘宝 direction=out，合成）。"""
    return {
        "seller_id": SHOP_A,
        "session_id": "conv_demo",
        "msg_id": msg_id,
        "msg_type": "text",
        "direction": "out",
        "content": text,
        "send_time_ms": send_time_ms,
    }


def test_status_compares_shadow_suggestions_with_agent_replies(service: AssistService):
    ro = service.rollout
    ro.set_stage(SHOP_A, "shadow", operator=REVIEWER)
    t0 = int(service.unit.clock.now().timestamp() * 1000)
    result = service.handle_event("B", taobao_event("什么时候发货", msg_id="sh1", send_time_ms=t0))
    assert result is not None and result.mode == Mode.SHADOW
    service.handle_event("B", _echo("亲，现货付款后 48 小时内发货哦~", "echo1", t0 + 5_000))
    service.handle_event("B", taobao_event("能开发票吗", msg_id="sh2", send_time_ms=t0 + 9_000))
    status = ro.status(SHOP_A)
    assert status.stage == RolloutStage.SHADOW and status.advice == RolloutAdvice.STAY_SHADOW
    assert status.shadow.groups == 2 and status.shadow.compared == 1
    assert status.shadow.no_reply == 1 and status.shadow.match_rate == 1.0
    assert status.shadow.until == service.unit.clock.now()
    detail = ro.shadow(SHOP_A)
    by_msg = {i.msg_id: i.similarity for i in detail.report.items}
    assert by_msg == {"sh1": 1.0, "sh2": None}
    # 只读：没有写审计、没有改配置
    assert events(service, AuditAction.QUALITY_VIEWED) == []
    assert len(events(service, AuditAction.ROLLOUT_STAGE)) == 1
    # 窗口可以指定：窗口里没有影子推荐
    since = service.unit.clock.now() - timedelta(days=30)
    empty = ro.shadow(SHOP_A, since=since, until=since + timedelta(days=1))
    assert empty.report.groups == 0
    with pytest.raises(ValueError):
        ro.shadow(SHOP_A, since=since, until=since)


def test_douyin_shop_configured_in_shadow_mode():
    """抖店演示店铺 D 由店铺配置开了影子模式（§13.4），没有切换记录也认得出来。"""
    service = AssistService(build_demo_unit(Platform.DOUYIN))
    status = service.rollout.status(SHOP_D)
    assert status.stage == RolloutStage.SHADOW and status.shadow.groups == 0
    assert status.latest_quality is None
    assert [u.name for u in status.unchecked] == ["test_set"]


def test_latest_quality_is_the_most_recent_period_with_samples(service: AssistService):
    add(service, 3)
    service.rollout.sample(SHOP_A, reviewer=REVIEWER, period=W39)
    latest = service.rollout.status(SHOP_A).latest_quality
    assert latest is not None and latest.period == W39 and latest.pending == 3


# ================================================================ 取消订购（§8.4）


def test_writes_after_revoke_do_not_recreate_anything(service: AssistService):
    add(service, 3)
    ro = service.rollout
    ro.sample(SHOP_A, reviewer=REVIEWER, period=W39)
    [review_id, *_] = [q.id for q in service.unit.repos.quality_reviews.list(SHOP_A)]
    service.revoke_shop(SHOP_A)
    calls = (
        lambda: ro.sample(SHOP_A, reviewer=REVIEWER, period=W39),
        lambda: ro.score(SHOP_A, review_id, status="skipped", reviewer=REVIEWER),
        lambda: ro.list_reviews(SHOP_A, viewer=REVIEWER, period=W39),
        lambda: ro.set_stage(SHOP_A, "shadow", operator=REVIEWER),
        lambda: ro.status(SHOP_A),
        lambda: ro.report(SHOP_A),
        lambda: ro.shadow(SHOP_A),
    )
    for call in calls:
        with pytest.raises(ShopInactive):
            call()
    repos = service.unit.repos
    assert repos.knowledge.get_shop(SHOP_A) is None
    assert repos.quality_reviews.list(SHOP_A) == []
    assert repos.audit.list_events(SHOP_A) == []


def _race(service: AssistService, write, patch_target, attr: str) -> tuple[list, list]:
    """write 在 patch_target.attr 里停住时商家取消订购：取消订购要等写完再删（§8.4）。
    返回 (写线程的异常, 写入那一刻取消订购是否已经做完)。"""
    inflight, revoke_started, revoke_done = (threading.Event() for _ in range(3))
    original = getattr(patch_target, attr)
    seen: list[bool] = []

    def slow(*args: Any, **kwargs: Any) -> Any:
        inflight.set()
        assert revoke_started.wait(5)
        revoke_done.wait(0.3)  # 取消订购要等在途操作做完：这里等不到，超时后继续
        seen.append(revoke_done.is_set())
        return original(*args, **kwargs)

    setattr(patch_target, attr, slow)
    errors: list[BaseException] = []

    def run_write() -> None:
        try:
            write()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def revoke() -> None:
        assert inflight.wait(5)
        revoke_started.set()
        service.revoke_shop(SHOP_A)
        revoke_done.set()

    threads = [threading.Thread(target=run_write), threading.Thread(target=revoke)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert revoke_done.is_set()
    return errors, seen


def test_revoke_waits_for_in_flight_sample_and_leaves_no_rows(service: AssistService):
    add(service, 5)
    repo = service.unit.repos.quality_reviews
    errors, seen = _race(
        service,
        lambda: service.rollout.sample(SHOP_A, reviewer=REVIEWER, period=W39),
        repo,
        "add_many",
    )
    assert not errors and seen == [False]
    assert repo.list(SHOP_A) == []
    assert service.unit.repos.audit.list_events(SHOP_A) == []


def test_revoke_waits_for_in_flight_stage_switch_and_the_shop_stays_deleted(
    service: AssistService,
):
    repo = service.unit.repos.knowledge
    errors, seen = _race(
        service,
        lambda: service.rollout.set_stage(SHOP_A, "shadow", operator=REVIEWER),
        repo,
        "update_shop_config",
    )
    assert not errors and seen == [False]
    assert repo.get_shop(SHOP_A) is None  # 在途的配置写入没有把删掉的店铺写回来
    assert service.unit.repos.audit.list_events(SHOP_A) == []
    assert repo.get_shop(SHOP_B) is not None


def test_revoke_waits_for_in_flight_score(service: AssistService):
    add(service, 2)
    service.rollout.sample(SHOP_A, reviewer=REVIEWER, period=W39)
    repo = service.unit.repos.quality_reviews
    [first, _] = repo.list(SHOP_A)
    errors, seen = _race(
        service,
        lambda: service.rollout.score(SHOP_A, first.id, status="skipped", reviewer=REVIEWER),
        repo,
        "record",
    )
    assert not errors and seen == [False]
    assert repo.list(SHOP_A) == [] and service.unit.repos.audit.list_events(SHOP_A) == []
