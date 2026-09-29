"""售后审核的客服备注写进 action_result（§10 JSONB）前去掉 NUL 等控制字符。

迁移用 model_copy(update=...) 生成新记录，不走模型边界的清洗：备注里的 NUL 内存实现照存，
PostgreSQL 的 JSONB 写不进去（协商接口 500）。NUL 夹在手机号中间时，先去掉再脱敏，号码照样遮住。
"""

from __future__ import annotations

from rodit.aftersale.state_machine import history, transition
from rodit.app.demo import build_demo_unit
from rodit.core.enums import ReviewState
from rodit.testing.fixtures import AGENT, SHOP_A


def test_negotiate_note_is_cleaned_before_it_is_stored():
    unit = build_demo_unit()
    unit.reviews.review(SHOP_A, "as_2007")
    saved = unit.reviews.negotiate(
        SHOP_A, "as_2007", operator=AGENT, note="买家同意换货\x00，明天寄回，电话138\x0012345678"
    )
    assert saved.state == ReviewState.NEGOTIATE
    note = history(saved)[-1]["note"]
    assert "\x00" not in note
    assert note.startswith("买家同意换货，明天寄回")
    assert "12345678" not in note  # 去掉 NUL 后的手机号照样脱敏
    stored = unit.repos.reviews.get(SHOP_A, "as_2007")
    assert stored is not None and "\x00" not in history(stored)[-1]["note"]


def test_transition_cleans_note_and_operator():
    unit = build_demo_unit()
    review = unit.reviews.review(SHOP_A, "as_2007")
    assert review.state == ReviewState.CONFIRM
    moved = transition(
        review, ReviewState.NEGOTIATE, at=unit.clock.now(), by="kf_02\x00", note="协商\x00中"
    )
    entry = history(moved)[-1]
    assert entry["by"] == "kf_02" and entry["note"] == "协商中"
