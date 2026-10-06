"""售后审核工作台的判断（rodit.app.aftersale_desk 的纯函数，§7.3.2）：下一步、平台状态取哪份、
执行后审核记录里的平台状态、买家说明打码截短、建议对应的操作、工作台的中文名覆盖完整。"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from rodit.aftersale import TERMINAL_STATES
from rodit.aftersale.state_machine import RESULT_KEY
from rodit.app.aftersale_desk import (
    ADVICE_LABELS,
    AFTER_ACTION_STATUS,
    MAX_NOTE_CHARS,
    NEEDS_ACTION,
    NEXT_STEP_LABELS,
    OPEN_STATES,
    PLATFORM_ACTIONS,
    PLATFORM_NOTES,
    QUEUE_FILTERS,
    REFUSE_HINTS,
    REVIEW_STATE_LABELS,
    RULES,
    STATE_FLOW,
    UNREVIEWED,
    NextStep,
    believed_status,
    effective_status,
    masked_note,
    next_step,
    platform_changed,
    recommended_action,
    review_is_newer,
    verify_due,
)
from rodit.core.enums import (
    AfterSaleAction,
    AfterSaleReason,
    AfterSaleStatus,
    AfterSaleType,
    Platform,
    ReviewAdvice,
    ReviewState,
)
from rodit.core.models import AfterSaleInfo, AfterSaleReview
from rodit.testing.fixtures import DEFAULT_NOW, SHOP_A

NOW = DEFAULT_NOW
VERIFY_HOURS = 24


def _review(state: ReviewState, **update) -> AfterSaleReview:
    base = {
        "shop_id": SHOP_A,
        "aftersale_id": "as_x",
        "platform_status": AfterSaleStatus.PENDING_REVIEW,
        "platform_status_version": "100",
        "state": state,
        "updated_at": NOW - timedelta(minutes=30),
    }
    return AfterSaleReview(**{**base, **update})


def _info(status: AfterSaleStatus = AfterSaleStatus.PENDING_REVIEW, **update) -> AfterSaleInfo:
    base = {
        "shop_id": SHOP_A,
        "aftersale_id": "as_x",
        "order_id": "o_x",
        "buyer_ref": "buyer_hash",
        "type": AfterSaleType.REFUND_ONLY,
        "reason": AfterSaleReason.OTHER,
        "status": status,
        "amount": Decimal("99.00"),
        "created_at": NOW - timedelta(hours=5),
        "updated_at": NOW - timedelta(hours=5),
        "status_version": "100",
        "fetched_at": NOW - timedelta(minutes=10),
    }
    return AfterSaleInfo(**{**base, **update})


def _acted(state: ReviewState, action: AfterSaleAction, ok: bool = True) -> AfterSaleReview:
    """客服确认执行过的审核记录：平台状态还是执行前重读的（审核助手执行后不再重读）。"""
    result = {
        "ok": ok,
        "action": action.value,
        "aftersale_id": "as_x",
        "idempotency_key": f"as_x:{action.value}:100",
        "platform_code": "0" if ok else "PLATFORM_ERROR",
        "message": None,
        "duplicate": False,
    }
    return _review(state, confirmed_by="kf_02", action_result={RESULT_KEY: result})


def _step(review: AfterSaleReview | None, info: AfterSaleInfo | None = None, **kw) -> NextStep:
    return next_step(review, info, now=kw.pop("now", NOW), verify_hours=VERIFY_HOURS, **kw)


# ================================================================ 平台状态取哪份


def test_review_is_newer_compares_update_and_fetch_times():
    review = _review(ReviewState.CONFIRM)
    assert review_is_newer(review, _info(fetched_at=NOW - timedelta(hours=1)))
    assert review_is_newer(review, _info(fetched_at=review.updated_at))  # 同一时刻：审核记录为准
    assert not review_is_newer(review, _info(fetched_at=NOW))
    assert not review_is_newer(review, _info(fetched_at=None))  # 没经过缓存：刚读到的
    assert not review_is_newer(_review(ReviewState.CONFIRM, updated_at=None), _info())


def test_platform_changed_only_when_newer_and_different():
    review = _review(ReviewState.CONFIRM)
    assert not platform_changed(review, _info())
    assert platform_changed(review, _info(status_version="101"))
    assert platform_changed(review, _info(AfterSaleStatus.CLOSED))
    old = _info(AfterSaleStatus.CLOSED, fetched_at=NOW - timedelta(hours=2))
    assert not platform_changed(review, old)  # 缓存比审核记录旧：不算变化
    assert effective_status(review, old) == AfterSaleStatus.PENDING_REVIEW
    assert effective_status(review, _info(AfterSaleStatus.CLOSED)) == AfterSaleStatus.CLOSED
    assert effective_status(None, _info(AfterSaleStatus.CLOSED)) == AfterSaleStatus.CLOSED
    assert effective_status(review, None) == AfterSaleStatus.PENDING_REVIEW
    assert effective_status(None, None) is None


@pytest.mark.parametrize(
    ("state", "action"),
    [
        (ReviewState.AGREED, AfterSaleAction.AGREE_REFUND),
        (ReviewState.WAIT_RETURN, AfterSaleAction.AGREE_RETURN),
        (ReviewState.REFUSED, AfterSaleAction.REFUSE),
    ],
)
def test_after_a_confirmed_action_the_review_believes_the_post_action_status(state, action):
    review = _acted(state, action)
    status, version = believed_status(review)
    assert status == AFTER_ACTION_STATUS[action] and version is None
    # 比审核记录旧的缓存（执行前的待审核）不会让刚执行完的单看起来"平台变了"
    assert effective_status(review, _info(fetched_at=NOW - timedelta(hours=1))) == status
    # 比审核记录新的平台数据：状态和执行后的一致就不算变化（状态版本执行后本来就变了）
    fresh = _info(status, status_version="999", fetched_at=NOW)
    assert not platform_changed(review, fresh)


def test_believed_status_keeps_the_synced_status_otherwise():
    synced = (AfterSaleStatus.PENDING_REVIEW, "100")
    assert believed_status(_review(ReviewState.CONFIRM)) == synced
    assert believed_status(_acted(ReviewState.CONFIRM, AfterSaleAction.AGREE_REFUND)) == synced
    failed = _acted(ReviewState.AGREED, AfterSaleAction.AGREE_REFUND, ok=False)
    assert believed_status(failed) == synced
    # 商家在平台后台同意了退货：系统同步进退货中，没有确认人，用同步到的状态
    system = _review(
        ReviewState.WAIT_RETURN,
        platform_status=AfterSaleStatus.WAIT_BUYER_RETURN,
        platform_status_version="200",
    )
    assert believed_status(system) == (AfterSaleStatus.WAIT_BUYER_RETURN, "200")


# ================================================================ 下一步


def test_next_step_without_review_is_running_the_rules():
    assert _step(None, _info()) == NextStep.RUN_RULES
    assert _step(None, None) == NextStep.RUN_RULES


@pytest.mark.parametrize("state", sorted(TERMINAL_STATES))
def test_finished_reviews_are_done(state):
    assert _step(_review(state), _info(AfterSaleStatus.CLOSED, fetched_at=NOW)) == NextStep.DONE


def test_confirm_needs_a_refresh_when_the_platform_moved():
    review = _review(ReviewState.CONFIRM)
    assert _step(review, _info()) == NextStep.CONFIRM
    assert _step(review, None) == NextStep.CONFIRM
    assert _step(review, _info(status_version="101", fetched_at=NOW)) == NextStep.REFRESH
    assert _step(review, _info(AfterSaleStatus.CLOSED, fetched_at=NOW)) == NextStep.REFRESH
    assert _step(review, _info(), platform_missing=True) == NextStep.REFRESH


def test_verify_waits_until_the_hours_are_up():
    started = NOW - timedelta(hours=VERIFY_HOURS - 1)
    review = _review(ReviewState.VERIFY, verify_started_at=started)
    assert _step(review, _info()) == NextStep.WAIT_VERIFY
    assert _step(review, _info(), now=started + timedelta(hours=VERIFY_HOURS)) == NextStep.REFRESH
    assert verify_due(review, started + timedelta(hours=VERIFY_HOURS), VERIFY_HOURS)
    assert not verify_due(_review(ReviewState.VERIFY), NOW, VERIFY_HOURS)  # 没有开始时间
    # 核查期间买家撤销了：刷新同步成平台已处理
    closed = _info(AfterSaleStatus.CLOSED, fetched_at=NOW)
    assert _step(review, closed) == NextStep.REFRESH


def test_evidence_waits_for_photos():
    review = _review(ReviewState.EVIDENCE)
    assert _step(review, _info()) == NextStep.WAIT_EVIDENCE
    arrived = _info(evidence_count=2, status_version="101", fetched_at=NOW)
    assert _step(review, arrived) == NextStep.REFRESH
    # 缓存比审核记录旧：看不出凭证到了没有，照样等
    stale = _info(evidence_count=2, status_version="101", fetched_at=NOW - timedelta(hours=1))
    assert _step(review, stale) == NextStep.WAIT_EVIDENCE


def test_wait_return_until_the_buyer_ships_back():
    review = _acted(ReviewState.WAIT_RETURN, AfterSaleAction.AGREE_RETURN)
    assert _step(review, _info(fetched_at=NOW - timedelta(hours=1))) == NextStep.WAIT_RETURN
    waiting = _info(AfterSaleStatus.WAIT_BUYER_RETURN, status_version="300", fetched_at=NOW)
    assert _step(review, waiting) == NextStep.WAIT_RETURN
    shipped = _info(AfterSaleStatus.WAIT_SELLER_CONFIRM, status_version="301", fetched_at=NOW)
    assert _step(review, shipped) == NextStep.REFRESH
    refunded = _info(AfterSaleStatus.REFUND_SUCCESS, status_version="302", fetched_at=NOW)
    assert _step(review, refunded) == NextStep.REFRESH


def test_negotiation_and_transient_states():
    assert _step(_review(ReviewState.NEGOTIATE), _info()) == NextStep.NEGOTIATING
    closed = _info(AfterSaleStatus.CLOSED, fetched_at=NOW)
    assert _step(_review(ReviewState.NEGOTIATE), closed) == NextStep.REFRESH
    for state in (ReviewState.NEW, ReviewState.SUGGEST):
        assert _step(_review(state), _info()) == NextStep.REFRESH


def test_needs_action_is_what_the_agent_can_push_forward():
    assert NEEDS_ACTION == {NextStep.RUN_RULES, NextStep.CONFIRM, NextStep.REFRESH}


# ================================================================ 其他


def test_masked_note_masks_strips_and_truncates():
    assert masked_note(None) is None and masked_note("") is None and masked_note("  ") is None
    note = masked_note("电话 13812345678\x00，地址浙江省杭州市西湖区文三路 100 号")
    assert note is not None and "13812345678" not in note and "\x00" not in note
    long = masked_note("好" * (MAX_NOTE_CHARS + 50))
    assert long is not None and len(long) == MAX_NOTE_CHARS and long.endswith("…")
    assert (
        masked_note("<script>x</script>") == "<script>x</script>"
    )  # 只当文字，由页面 textContent 显示


def test_recommended_action_follows_the_advice_only_when_confirming():
    confirm = _review(ReviewState.CONFIRM, suggestion=ReviewAdvice.AGREE_RETURN)
    assert recommended_action(confirm) == AfterSaleAction.AGREE_RETURN
    agree = _review(ReviewState.CONFIRM, suggestion=ReviewAdvice.AGREE)
    assert recommended_action(agree) == AfterSaleAction.AGREE_REFUND
    for advice in (ReviewAdvice.MANUAL, ReviewAdvice.NEGOTIATE, ReviewAdvice.NONE, None):
        assert recommended_action(_review(ReviewState.CONFIRM, suggestion=advice)) is None
    assert recommended_action(_review(ReviewState.VERIFY, suggestion=ReviewAdvice.AGREE)) is None
    assert recommended_action(None) is None


def test_labels_cover_every_value():
    assert set(REVIEW_STATE_LABELS) == set(ReviewState) == set(STATE_FLOW)
    assert len(STATE_FLOW) == len(set(STATE_FLOW))
    assert set(ADVICE_LABELS) == set(ReviewAdvice)
    assert set(NEXT_STEP_LABELS) == set(NextStep)
    assert list(RULES) == [f"R{i}" for i in range(1, 9)]
    assert set(PLATFORM_ACTIONS) == set(PLATFORM_NOTES) == set(REFUSE_HINTS) == set(Platform)
    assert AfterSaleAction.REFUSE not in PLATFORM_ACTIONS[Platform.PDD]  # §7.1 没查到拒绝接口
    assert set(OPEN_STATES) == set(ReviewState) - TERMINAL_STATES
    assert QUEUE_FILTERS == (UNREVIEWED, *(s.value for s in ReviewState))
    assert set(AFTER_ACTION_STATUS) == set(AfterSaleAction)
