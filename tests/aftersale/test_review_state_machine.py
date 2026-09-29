"""T3 售后审核状态机（方案 §7.3.2；合法迁移见 core.enums.REVIEW_TRANSITIONS）。"""

from __future__ import annotations

import itertools
from datetime import timedelta

import pytest

from rodit.aftersale.state_machine import (
    TERMINAL_STATES,
    can_transition,
    history,
    last_result,
    previous_states,
    transition,
    was_negotiated,
    with_result,
)
from rodit.core.enums import (
    REVIEW_TRANSITIONS,
    AfterSaleAction,
    AfterSaleStatus,
    ReviewState,
)
from rodit.core.errors import ActionNotAllowed
from rodit.core.models import ActionResult, AfterSaleReview
from rodit.testing.fixtures import DEFAULT_NOW, SHOP_A

S = ReviewState


def _review(state: ReviewState = S.NEW) -> AfterSaleReview:
    return AfterSaleReview(
        shop_id=SHOP_A,
        aftersale_id="as_2001",
        platform_status=AfterSaleStatus.PENDING_REVIEW,
        state=state,
    )


@pytest.mark.parametrize(("frm", "to"), list(itertools.product(ReviewState, ReviewState)))
def test_transition_matrix(frm, to):
    legal = to in REVIEW_TRANSITIONS[frm]
    assert can_transition(frm, to) is legal
    review = _review(frm)
    if legal:
        moved = transition(review, to, at=DEFAULT_NOW)
        assert moved.state == to and moved.updated_at == DEFAULT_NOW
    else:
        with pytest.raises(ActionNotAllowed):
            transition(review, to, at=DEFAULT_NOW)


def test_terminal_states():
    assert TERMINAL_STATES == {S.AGREED, S.REFUSED, S.PLATFORM}
    # 非终态都能因平台侧变化进入 PLATFORM
    for state in set(ReviewState) - TERMINAL_STATES:
        assert can_transition(state, S.PLATFORM)


def test_history_and_previous_states():
    t = DEFAULT_NOW
    r = transition(_review(), S.SUGGEST, at=t)
    r = transition(r, S.CONFIRM, at=t + timedelta(seconds=1))
    r = transition(r, S.NEGOTIATE, at=t + timedelta(minutes=5), by="kf_02", note="约定补发")
    r = transition(r, S.CONFIRM, at=t + timedelta(hours=1), by="kf_02")
    assert previous_states(r) == [S.NEW, S.SUGGEST, S.CONFIRM, S.NEGOTIATE]
    h = history(r)
    assert [x["to"] for x in h] == ["suggest", "confirm", "negotiate", "confirm"]
    assert h[2] == {
        "from": "confirm",
        "to": "negotiate",
        "at": (t + timedelta(minutes=5)).isoformat(),
        "by": "kf_02",
        "note": "约定补发",
    }
    assert was_negotiated(r)


def test_not_negotiated_by_default():
    r = transition(transition(_review(), S.SUGGEST, at=DEFAULT_NOW), S.CONFIRM, at=DEFAULT_NOW)
    assert not was_negotiated(r)


def test_verify_sets_verify_started_at():
    r = transition(_review(), S.SUGGEST, at=DEFAULT_NOW)
    later = DEFAULT_NOW + timedelta(seconds=3)
    assert transition(r, S.VERIFY, at=later).verify_started_at == later
    assert transition(r, S.CONFIRM, at=later).verify_started_at is None


def test_transition_does_not_mutate_input():
    original = _review()
    moved = transition(original, S.SUGGEST, at=DEFAULT_NOW)
    assert original.state == S.NEW and original.action_result is None
    assert moved is not original


def test_with_result_keeps_history():
    r = transition(transition(_review(), S.SUGGEST, at=DEFAULT_NOW), S.CONFIRM, at=DEFAULT_NOW)
    result = ActionResult(
        ok=True,
        action=AfterSaleAction.AGREE_REFUND,
        aftersale_id="as_2001",
        idempotency_key="as_2001:agree_refund:1",
        platform_code="0",
    )
    r = with_result(r, result)
    assert last_result(r) == result
    assert len(history(r)) == 2
    assert last_result(_review()) is None
