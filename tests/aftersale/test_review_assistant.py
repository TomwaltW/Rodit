"""T3 售后审核助手：建议、状态推进、确认后执行、平台差异、防误操作、留痕（方案 §7.3.2、§11.3）。"""

from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest

from rodit.aftersale.assistant import (
    MSG_DOUYIN_REFUSE,
    MSG_MANAGER,
    MSG_NO_PORT,
    MSG_PDD_REFUSE,
    MSG_STALE,
    MSG_TAOBAO_REFUSE,
    ReviewAssistant,
)
from rodit.aftersale.state_machine import history, last_result, was_negotiated
from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.config import AfterSaleConfig, EngineConfig, effective_shop_config
from rodit.core.enums import (
    AfterSaleAction,
    AfterSaleStatus,
    Platform,
    ReviewAdvice,
    ReviewState,
)
from rodit.core.errors import (
    ActionNotAllowed,
    ConcurrencyConflict,
    TenantIsolationError,
    ToolUnavailable,
)
from rodit.core.models import AfterSaleReview, aftersale_idempotency_key
from rodit.inmemory import InMemoryAfterSaleReviewRepository, InMemoryCache
from rodit.testing.fakes import FakeActionPort, FakeClock, InMemoryToolGateway, confirmation_for
from rodit.testing.fixtures import AGENT, MANAGER, SHOP_A, SHOP_B, build_demo_world
from rodit.tools.gateway import CachingToolGateway

S, A, ACT = ReviewState, ReviewAdvice, AfterSaleAction
RS = AfterSaleStatus


class ConflictOnce(InMemoryAfterSaleReviewRepository):
    """下一次保存抛 ConcurrencyConflict（模拟另一个客服同时改了这条审核记录）。"""

    def __init__(self) -> None:
        super().__init__()
        self.conflict_next = False

    def save(self, review: AfterSaleReview, *, expected_version: int | None) -> AfterSaleReview:
        if self.conflict_next:
            self.conflict_next = False
            raise ConcurrencyConflict("模拟并发修改")
        return super().save(review, expected_version=expected_version)


class RacingPort(FakeActionPort):
    """平台状态在"我们重读之后、平台操作之前"变化（重读与写之间的竞态）。"""

    flip_to: AfterSaleStatus = RS.CLOSED

    def agree_refund(self, shop_id: str, aftersale_id: str, **kw: Any):
        assert self.gateway is not None
        self.gateway.set_aftersale_status(
            shop_id, aftersale_id, self.flip_to, at=self.clock.now() + timedelta(seconds=30)
        )
        return super().agree_refund(shop_id, aftersale_id, **kw)


class Env:
    def __init__(
        self,
        platform: Platform = Platform.TAOBAO,
        *,
        port: bool = True,
        supported: set[AfterSaleAction] | None = None,
        port_gateway: bool = True,
        port_cls: type[FakeActionPort] = FakeActionPort,
        cached: bool = False,
    ) -> None:
        self.w = build_demo_world()
        self.clock = FakeClock(self.w.now)
        self.inner = InMemoryToolGateway(self.w, platform)
        self.gw = (
            CachingToolGateway(self.inner, InMemoryCache(self.clock), self.clock, ttl_s=3600)
            if cached
            else self.inner
        )
        self.port = (
            port_cls(
                platform,
                supported=supported,
                gateway=self.inner if port_gateway else None,
                clock=self.clock,
            )
            if port
            else None
        )
        self.repo = ConflictOnce()
        self.audit = InMemoryAuditSink()
        engine = EngineConfig()
        self.asst = ReviewAssistant(
            self.gw,
            self.repo,
            self.audit,
            self.clock,
            AfterSaleConfig(),
            shop_config=lambda shop_id: effective_shop_config(engine, self.w.shop(shop_id)),
            action_port=self.port,
        )

    def conf(self, aftersale_id: str, **kw: Any):
        """客服按当前平台数据确认。"""
        return confirmation_for(self.w.aftersale(SHOP_A, aftersale_id), at=self.clock.now(), **kw)

    def set_status(self, aftersale_id: str, status: AfterSaleStatus, **kw: Any):
        return self.inner.set_aftersale_status(
            SHOP_A, aftersale_id, status, at=self.clock.now() + timedelta(seconds=1), **kw
        )

    def events(self, action: AuditAction):
        return self.audit.list_events(SHOP_A, action=str(action))


@pytest.fixture
def env():
    return Env()


# ================================================================ review


@pytest.mark.parametrize(
    ("aftersale_id", "state", "advice", "rules"),
    [
        ("as_2001", S.CONFIRM, A.AGREE, ["R2"]),
        ("as_2002", S.CONFIRM, A.AGREE_RETURN, ["R3"]),
        ("as_2003", S.VERIFY, A.NEGOTIATE, ["R4"]),
        ("as_2004", S.EVIDENCE, A.MANUAL, ["R5"]),
        ("as_2005", S.CONFIRM, A.AGREE_RETURN, ["R3", "R6"]),
        ("as_2006", S.PLATFORM, A.NONE, ["R1"]),
        ("as_2007", S.CONFIRM, A.MANUAL, ["R8"]),
        ("as_2008", S.PLATFORM, A.NONE, ["R1"]),
        ("as_2009", S.CONFIRM, A.MANUAL, ["R5"]),
        ("as_3004", S.CONFIRM, A.MANUAL, ["R3", "R7"]),
    ],
)
def test_review_demo_world(env, aftersale_id, state, advice, rules):
    r = env.asst.review(SHOP_A, aftersale_id)
    assert (r.state, r.suggestion, r.matched_rules) == (state, advice, rules)
    assert r.version == 1 and r.updated_at == env.w.now
    assert r.platform_status == env.w.aftersale(SHOP_A, aftersale_id).status
    assert env.repo.get(SHOP_A, aftersale_id) == r
    assert [h["to"] for h in history(r)] == ["suggest", str(state)]
    assert env.audit.actions(SHOP_A) == [
        AuditAction.REVIEW_TRANSITION,
        AuditAction.REVIEW_SUGGEST,
        AuditAction.REVIEW_TRANSITION,
    ]
    assert env.port.calls == []  # 只给建议，不操作平台


def test_review_flags_r6_and_r7(env):
    assert env.asst.review(SHOP_A, "as_2005").requires_manager
    risky = env.asst.review(SHOP_A, "as_3004")
    assert risky.risk_flags == ["30 天内售后 4 次（≥ 3）"]


def test_review_skips_order_lookups_for_r1(env):
    env.asst.review(SHOP_A, "as_2006")
    assert [c[0] for c in env.inner.calls] == ["get_aftersale"]


def test_review_is_idempotent(env):
    first = env.asst.review(SHOP_A, "as_2001")
    again = env.asst.review(SHOP_A, "as_2001")
    assert again == first and again.version == 1
    assert len(env.events(AuditAction.REVIEW_SUGGEST)) == 1


def test_review_unknown_or_other_shop(env):
    with pytest.raises(KeyError):
        env.asst.review(SHOP_A, "as_nope")
    with pytest.raises(KeyError):
        env.asst.review(SHOP_B, "as_2001")  # A 店的售后单用 B 店身份查不到
    with pytest.raises(KeyError):
        env.asst.refresh(SHOP_B, "as_2001")  # 没有记录的 refresh 等同 review
    b = env.asst.review(SHOP_B, "as_9001")
    assert b.shop_id == SHOP_B and env.repo.get(SHOP_A, "as_9001") is None


def test_review_degrades_when_tools_unavailable(env):
    env.inner.fail_on = {"get_order", "get_logistics", "list_aftersales"}
    r = env.asst.review(SHOP_A, "as_2001")
    assert r.matched_rules == ["R8"] and r.suggestion == A.MANUAL and r.state == S.CONFIRM
    assert any("订单暂时查不到" in f for f in r.risk_flags)
    assert any("未检查 R7" in f for f in r.risk_flags)


def test_review_raises_when_aftersale_unavailable(env):
    env.inner.fail_on = {"get_aftersale"}
    with pytest.raises(ToolUnavailable):
        env.asst.review(SHOP_A, "as_2001")
    assert env.repo.get(SHOP_A, "as_2001") is None


class _ForeignGateway(InMemoryToolGateway):
    def get_aftersale(self, shop_id: str, aftersale_id: str):
        return super().get_aftersale(SHOP_B, "as_9001")


def test_review_rejects_cross_shop_data():
    w = build_demo_world()
    clock = FakeClock(w.now)
    asst = ReviewAssistant(
        _ForeignGateway(w),
        InMemoryAfterSaleReviewRepository(),
        InMemoryAuditSink(),
        clock,
        AfterSaleConfig(),
        shop_config=lambda s: effective_shop_config(EngineConfig(), w.shop(s)),
        action_port=None,
    )
    with pytest.raises(TenantIsolationError):
        asst.review(SHOP_A, "as_2001")


# ================================================================ confirm


def test_confirm_happy_path(env):
    env.asst.review(SHOP_A, "as_2001")
    info = env.w.aftersale(SHOP_A, "as_2001")
    r = env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    assert r.state == S.AGREED and r.confirmed_by == AGENT and r.version == 2
    key = aftersale_idempotency_key("as_2001", "agree_refund", info.status_version)
    assert [c["key"] for c in env.port.calls] == [key]
    result = last_result(r)
    assert result.ok and result.idempotency_key == key and not result.duplicate
    assert env.w.aftersale(SHOP_A, "as_2001").status == RS.REFUNDING
    # 留痕：谁确认的、依据哪条规则、平台返回结果
    confirm_ev, action_ev = (
        env.events(AuditAction.REVIEW_CONFIRM),
        env.events(AuditAction.REVIEW_ACTION),
    )
    assert len(confirm_ev) == 1 and len(action_ev) == 1
    assert confirm_ev[0].actor == AGENT and action_ev[0].actor == AGENT
    assert confirm_ev[0].detail["rules"] == ["R2"] and confirm_ev[0].detail["suggestion"] == "agree"
    assert confirm_ev[0].detail["expected_status_version"] == info.status_version
    assert action_ev[0].detail["ok"] is True and action_ev[0].detail["platform_code"] == "0"
    assert action_ev[0].detail["idempotency_key"] == key
    assert env.audit.actions(SHOP_A)[-3:] == [
        AuditAction.REVIEW_CONFIRM,
        AuditAction.REVIEW_ACTION,
        AuditAction.REVIEW_TRANSITION,
    ]


def test_r3_two_step_flow(env):
    r = env.asst.review(SHOP_A, "as_2002")
    assert (r.state, r.suggestion) == (S.CONFIRM, A.AGREE_RETURN)
    r = env.asst.confirm(SHOP_A, "as_2002", ACT.AGREE_RETURN, confirmation=env.conf("as_2002"))
    assert r.state == S.WAIT_RETURN
    assert env.w.aftersale(SHOP_A, "as_2002").status == RS.WAIT_BUYER_RETURN

    # 买家还没寄回：保持退货中
    env.clock.advance(hours=12)
    assert env.asst.refresh(SHOP_A, "as_2002").state == S.WAIT_RETURN

    # 买家寄回、平台状态变成等商家确认收货 → 回到待确认，建议同意退款
    env.clock.advance(days=1)
    env.set_status("as_2002", RS.WAIT_SELLER_CONFIRM)
    r = env.asst.refresh(SHOP_A, "as_2002")
    assert (r.state, r.suggestion) == (S.CONFIRM, A.AGREE)
    assert r.platform_status == RS.WAIT_SELLER_CONFIRM
    assert "仓库确认收货后同意退款" in r.reason

    # 同意退货不能再点一次
    with pytest.raises(ActionNotAllowed):
        env.asst.confirm(SHOP_A, "as_2002", ACT.AGREE_RETURN, confirmation=env.conf("as_2002"))
    r = env.asst.confirm(SHOP_A, "as_2002", ACT.AGREE_REFUND, confirmation=env.conf("as_2002"))
    assert r.state == S.AGREED
    assert [c["action"] for c in env.port.calls] == [ACT.AGREE_RETURN, ACT.AGREE_REFUND]
    assert [h["to"] for h in history(r)] == [
        "suggest",
        "confirm",
        "wait_return",
        "confirm",
        "agreed",
    ]
    assert env.w.aftersale(SHOP_A, "as_2002").status == RS.REFUNDING


def test_reread_race_platform_closed_no_port_call(env):
    env.asst.review(SHOP_A, "as_2001")
    conf = env.conf("as_2001")
    env.set_status("as_2001", RS.CLOSED)  # 买家在客服确认前撤销了
    r = env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=conf)
    assert r.state == S.PLATFORM and r.suggestion == A.NONE and r.matched_rules == ["R1"]
    assert r.platform_status == RS.CLOSED
    assert env.port.calls == []
    assert env.events(AuditAction.REVIEW_ACTION) == []


def test_stale_confirmation_version(env):
    env.asst.review(SHOP_A, "as_2001")
    conf = env.conf("as_2001")
    env.set_status("as_2001", RS.PENDING_REVIEW)  # 状态没变，但版本变了（买家改了申请）
    with pytest.raises(ActionNotAllowed, match=MSG_STALE):
        env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=conf)
    assert env.port.calls == []
    assert env.repo.get(SHOP_A, "as_2001").state == S.CONFIRM
    # 客服拿到新版本、但建议还是按旧版本给的：也要先刷新（建议可能已经过期）
    with pytest.raises(ActionNotAllowed, match=MSG_STALE):
        env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    assert env.port.calls == []
    # 刷新后建议按新版本重新生成，再按新版本确认即可
    refreshed = env.asst.refresh(SHOP_A, "as_2001")
    latest = env.inner.get_aftersale(SHOP_A, "as_2001")
    assert refreshed.state == S.CONFIRM
    assert refreshed.platform_status_version == latest.status_version
    r = env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    assert r.state == S.AGREED


def test_double_confirm_sequential_one_platform_call(env):
    env.asst.review(SHOP_A, "as_2001")
    conf = env.conf("as_2001")
    env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=conf)
    with pytest.raises(ActionNotAllowed):
        env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=conf)
    assert len(env.port.calls) == 1


def test_double_confirm_concurrent_uses_idempotency_key():
    env = Env(port_gateway=False)  # 平台状态不变：两次确认算出同一个幂等键
    env.asst.review(SHOP_A, "as_2001")
    before = env.repo.get(SHOP_A, "as_2001")
    conf = env.conf("as_2001")
    first = env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=conf)
    # 另一个客服端还停在确认前的记录上
    env.repo.save(before, expected_version=first.version)
    second = env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=conf)
    assert len(env.port.calls) == 1  # 平台只操作一次
    assert last_result(second).duplicate is True and second.state == S.AGREED


def test_concurrency_conflict_propagates_and_retry_is_safe(env):
    env.asst.review(SHOP_A, "as_2001")
    env.repo.conflict_next = True
    with pytest.raises(ConcurrencyConflict):
        env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    assert len(env.port.calls) == 1  # 平台已经操作
    assert len(env.events(AuditAction.REVIEW_ACTION)) == 1  # 平台结果已留痕
    assert env.repo.get(SHOP_A, "as_2001").state == S.CONFIRM
    # 重试：重读发现平台已在退款中 → PLATFORM，不再调平台
    r = env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    assert r.state == S.PLATFORM and len(env.port.calls) == 1


def test_manager_required_for_r6(env):
    env.asst.review(SHOP_A, "as_2005")
    with pytest.raises(ActionNotAllowed, match="店长"):
        env.asst.confirm(SHOP_A, "as_2005", ACT.AGREE_RETURN, confirmation=env.conf("as_2005"))
    # 勾了店长确认，但操作人不是店长
    fake_manager = env.conf("as_2005", manager=True)
    with pytest.raises(ActionNotAllowed, match=MSG_MANAGER):
        env.asst.confirm(SHOP_A, "as_2005", ACT.AGREE_RETURN, confirmation=fake_manager)
    # 店长本人但没勾店长确认
    with pytest.raises(ActionNotAllowed):
        env.asst.confirm(
            SHOP_A, "as_2005", ACT.AGREE_RETURN, confirmation=env.conf("as_2005", operator=MANAGER)
        )
    assert env.port.calls == []
    ok = env.conf("as_2005", operator=MANAGER, manager=True)
    r = env.asst.confirm(SHOP_A, "as_2005", ACT.AGREE_RETURN, confirmation=ok)
    assert r.state == S.WAIT_RETURN and r.confirmed_by == MANAGER
    assert env.events(AuditAction.REVIEW_CONFIRM)[-1].detail["by_manager"] is True


def test_manager_required_when_amount_raised_after_review(env):
    r = env.asst.review(SHOP_A, "as_2001")
    assert not r.requires_manager
    # 买家把申请金额改到上限以上：状态不变、版本变了
    a = env.w.aftersale(SHOP_A, "as_2001")
    env.w.aftersales[env.w.aftersales.index(a)] = a.model_copy(
        update={"amount": Decimal("800.00"), "status_version": "9999999999"}
    )
    # 建议是按改金额之前的售后单给的：先要求刷新
    with pytest.raises(ActionNotAllowed, match=MSG_STALE):
        env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    refreshed = env.asst.refresh(SHOP_A, "as_2001")
    assert refreshed.requires_manager and "R6" in refreshed.matched_rules
    with pytest.raises(ActionNotAllowed, match=MSG_MANAGER):
        env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    assert env.port.calls == []
    ok = env.conf("as_2001", operator=MANAGER, manager=True)
    assert env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=ok).state == S.AGREED


def test_taobao_refuse_requires_negotiation(env):
    env.asst.review(SHOP_A, "as_2007")  # R8 人工决定
    with pytest.raises(ActionNotAllowed, match=MSG_TAOBAO_REFUSE):
        env.asst.confirm(SHOP_A, "as_2007", ACT.REFUSE, confirmation=env.conf("as_2007"))
    r = env.asst.negotiate(SHOP_A, "as_2007", operator=AGENT, note="建议买家换货")
    assert r.state == S.NEGOTIATE and was_negotiated(r)
    with pytest.raises(ActionNotAllowed):  # 协商中不能确认操作
        env.asst.confirm(SHOP_A, "as_2007", ACT.REFUSE, confirmation=env.conf("as_2007"))
    r = env.asst.negotiation_done(SHOP_A, "as_2007", operator=AGENT, outcome="买家坚持退货")
    assert r.state == S.CONFIRM and was_negotiated(r)
    assert env.port.calls == []
    r = env.asst.confirm(
        SHOP_A,
        "as_2007",
        ACT.REFUSE,
        confirmation=env.conf("as_2007"),
        reason_text="定制商品不支持七天无理由",
    )
    assert r.state == S.REFUSED
    assert [c["action"] for c in env.port.calls] == [ACT.REFUSE]
    assert [h["to"] for h in history(r)] == [
        "suggest",
        "confirm",
        "negotiate",
        "confirm",
        "refused",
    ]


def test_douyin_refuse_requires_reason_code():
    env = Env(Platform.DOUYIN)
    env.asst.review(SHOP_A, "as_2007")
    with pytest.raises(ActionNotAllowed, match=MSG_DOUYIN_REFUSE):
        env.asst.confirm(SHOP_A, "as_2007", ACT.REFUSE, confirmation=env.conf("as_2007"))
    assert env.port.calls == []
    r = env.asst.confirm(
        SHOP_A, "as_2007", ACT.REFUSE, confirmation=env.conf("as_2007"), reason_code="R_CUSTOM"
    )
    assert r.state == S.REFUSED  # 抖店不要求先协商
    assert env.port.calls[0]["reason_code"] == "R_CUSTOM"


def test_pdd_refuse_not_supported():
    env = Env(Platform.PDD, supported={ACT.AGREE_REFUND, ACT.AGREE_RETURN})
    env.asst.review(SHOP_A, "as_2007")
    with pytest.raises(ActionNotAllowed, match=MSG_PDD_REFUSE):
        env.asst.confirm(
            SHOP_A, "as_2007", ACT.REFUSE, confirmation=env.conf("as_2007"), reason_code="x"
        )
    assert "请在商家后台手动操作" in MSG_PDD_REFUSE
    assert env.port.calls == []
    r = env.asst.confirm(SHOP_A, "as_2007", ACT.AGREE_RETURN, confirmation=env.conf("as_2007"))
    assert r.state == S.WAIT_RETURN


def test_unsupported_action_generic_message():
    env = Env(supported={ACT.AGREE_REFUND})
    env.asst.review(SHOP_A, "as_2002")
    with pytest.raises(ActionNotAllowed, match="淘宝不支持同意退货"):
        env.asst.confirm(SHOP_A, "as_2002", ACT.AGREE_RETURN, confirmation=env.conf("as_2002"))


def test_no_action_port_only_suggests():
    env = Env(port=False)
    r = env.asst.review(SHOP_A, "as_2001")
    assert r.state == S.CONFIRM and r.suggestion == A.AGREE
    with pytest.raises(ActionNotAllowed, match=MSG_NO_PORT):
        env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    assert env.repo.get(SHOP_A, "as_2001").state == S.CONFIRM


def test_confirm_requires_confirm_state_and_matching_status(env):
    env.asst.review(SHOP_A, "as_2003")  # VERIFY
    with pytest.raises(ActionNotAllowed, match="不在待客服确认状态"):
        env.asst.confirm(SHOP_A, "as_2003", ACT.AGREE_REFUND, confirmation=env.conf("as_2003"))
    with pytest.raises(ActionNotAllowed, match="还没有审核记录"):
        env.asst.confirm(SHOP_A, "as_2009", ACT.AGREE_REFUND, confirmation=env.conf("as_2009"))
    env.asst.review(SHOP_A, "as_2001")
    with pytest.raises(ActionNotAllowed, match="仅退款"):
        env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_RETURN, confirmation=env.conf("as_2001"))
    assert env.port.calls == []


def test_platform_error_keeps_confirm_and_retry_works(env):
    env.asst.review(SHOP_A, "as_2001")
    env.port.fail_next = True
    r = env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    assert r.state == S.CONFIRM
    assert last_result(r).ok is False and last_result(r).platform_code == "PLATFORM_ERROR"
    assert env.events(AuditAction.REVIEW_ACTION)[-1].detail["ok"] is False
    r = env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    assert r.state == S.AGREED and len(env.port.calls) == 2


def test_port_status_changed_to_closed_goes_platform():
    env = Env(port_cls=RacingPort)
    env.asst.review(SHOP_A, "as_2001")
    r = env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    assert last_result(r).platform_code == "STATUS_CHANGED"
    assert r.state == S.PLATFORM and r.platform_status == RS.CLOSED


def test_port_status_changed_but_still_actionable_stays_confirm():
    class Bump(RacingPort):
        flip_to = RS.PENDING_REVIEW

    env = Env(port_cls=Bump)
    env.asst.review(SHOP_A, "as_2001")
    r = env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    assert r.state == S.CONFIRM and r.reason.startswith(MSG_STALE)
    assert r.suggestion == A.AGREE


def test_confirm_bypasses_tool_cache():
    env = Env(cached=True)
    env.asst.review(SHOP_A, "as_2001")
    conf = env.conf("as_2001")
    assert env.gw.get_aftersale(SHOP_A, "as_2001").status == RS.PENDING_REVIEW  # 缓存已热
    env.set_status("as_2001", RS.CLOSED)
    assert env.gw.get_aftersale(SHOP_A, "as_2001").status == RS.PENDING_REVIEW  # 缓存还是旧的
    r = env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=conf)
    assert r.state == S.PLATFORM and env.port.calls == []


def test_port_exception_is_audited_and_raised():
    env = Env(Platform.DOUYIN)

    def boom(*_: Any, **__: Any):
        raise ToolUnavailable("平台限流")

    env.asst.review(SHOP_A, "as_2001")
    env.port.agree_refund = boom  # type: ignore[method-assign]
    with pytest.raises(ToolUnavailable):
        env.asst.confirm(SHOP_A, "as_2001", ACT.AGREE_REFUND, confirmation=env.conf("as_2001"))
    detail = env.events(AuditAction.REVIEW_ACTION)[-1].detail
    assert detail["ok"] is False and "平台限流" in detail["error"]
    assert env.repo.get(SHOP_A, "as_2001").state == S.CONFIRM


# ================================================================ refresh


def test_verify_24h_timeout_suggests_agree(env):
    r = env.asst.review(SHOP_A, "as_2003")
    assert r.state == S.VERIFY and r.verify_started_at == env.w.now
    env.clock.advance(hours=23, minutes=59)
    same = env.asst.refresh(SHOP_A, "as_2003")
    assert same.state == S.VERIFY and same.version == r.version
    env.clock.advance(minutes=1)
    r = env.asst.refresh(SHOP_A, "as_2003")
    assert (r.state, r.suggestion) == (S.CONFIRM, A.AGREE)
    assert "24 小时后仍未找到，建议同意" in r.reason
    r = env.asst.confirm(SHOP_A, "as_2003", ACT.AGREE_REFUND, confirmation=env.conf("as_2003"))
    assert r.state == S.AGREED


def test_verify_buyer_found_parcel_and_cancelled(env):
    env.asst.review(SHOP_A, "as_2003")
    env.clock.advance(hours=3)
    env.set_status("as_2003", RS.CLOSED)
    r = env.asst.refresh(SHOP_A, "as_2003")
    assert r.state == S.PLATFORM and r.suggestion == A.NONE


def test_verify_timeout_keeps_r7_manual(env):
    risky = env.w.aftersale(SHOP_A, "as_3004").model_copy(
        update={"aftersale_id": "as_3005", "reason": "not_received", "type": "refund_only"}
    )
    env.w.aftersales.append(risky)
    r = env.asst.review(SHOP_A, "as_3005")
    assert r.state == S.VERIFY and r.matched_rules == ["R4", "R7"]
    env.clock.advance(hours=24)
    r = env.asst.refresh(SHOP_A, "as_3005")
    assert (r.state, r.suggestion) == (S.CONFIRM, A.MANUAL)
    assert r.risk_flags


def test_evidence_uploaded_moves_to_confirm(env):
    r = env.asst.review(SHOP_A, "as_2004")
    assert r.state == S.EVIDENCE
    env.clock.advance(hours=2)
    unchanged = env.asst.refresh(SHOP_A, "as_2004")
    assert unchanged.version == r.version
    env.set_status("as_2004", RS.PENDING_REVIEW, evidence_count=2)
    r = env.asst.refresh(SHOP_A, "as_2004")
    assert (r.state, r.suggestion, r.matched_rules) == (S.CONFIRM, A.MANUAL, ["R5"])
    assert "客服查看凭证后决定" in r.reason


def test_confirm_state_resuggests_when_platform_status_changes(env):
    env.asst.review(SHOP_A, "as_2002")
    env.set_status("as_2002", RS.WAIT_SELLER_CONFIRM)  # 商家在后台同意了退货，买家已寄回
    r = env.asst.refresh(SHOP_A, "as_2002")
    assert (r.state, r.suggestion) == (S.CONFIRM, A.AGREE)
    assert len(env.events(AuditAction.REVIEW_SUGGEST)) == 2


def test_refresh_terminal_does_not_call_platform(env):
    env.asst.review(SHOP_A, "as_2006")
    env.inner.calls.clear()
    assert env.asst.refresh(SHOP_A, "as_2006").state == S.PLATFORM
    assert env.inner.calls == []


def test_refresh_without_record_reviews(env):
    assert env.asst.refresh(SHOP_A, "as_2001").state == S.CONFIRM


def test_refresh_aftersale_disappeared(env):
    env.asst.review(SHOP_A, "as_2001")
    env.w.aftersales = [a for a in env.w.aftersales if a.aftersale_id != "as_2001"]
    r = env.asst.refresh(SHOP_A, "as_2001")
    assert r.state == S.PLATFORM and "查不到" in r.reason


# ================================================================ 协商


def test_negotiate_rules(env):
    env.asst.review(SHOP_A, "as_2003")  # VERIFY
    with pytest.raises(ActionNotAllowed):
        env.asst.negotiate(SHOP_A, "as_2003", operator=AGENT)
    env.asst.review(SHOP_A, "as_2001")
    for robot in ("system", "BOT", " auto ", ""):
        with pytest.raises(ActionNotAllowed):
            env.asst.negotiate(SHOP_A, "as_2001", operator=robot)
    with pytest.raises(ActionNotAllowed):
        env.asst.negotiation_done(SHOP_A, "as_2001", operator=AGENT)
    r = env.asst.negotiate(SHOP_A, "as_2001", operator=AGENT)
    assert env.events(AuditAction.REVIEW_TRANSITION)[-1].actor == AGENT
    assert r.state == S.NEGOTIATE


def test_buyer_cancels_during_negotiation(env):
    env.asst.review(SHOP_A, "as_2001")
    env.asst.negotiate(SHOP_A, "as_2001", operator=AGENT)
    env.set_status("as_2001", RS.CLOSED)
    r = env.asst.negotiation_done(SHOP_A, "as_2001", operator=AGENT, outcome="买家撤销")
    assert r.state == S.PLATFORM


# ================================================================ 留痕不含买家原文


def test_audit_details_contain_no_buyer_text(env):
    a = env.w.aftersale(SHOP_A, "as_2003")
    env.w.aftersales[env.w.aftersales.index(a)] = a.model_copy(
        update={"note": "我电话13812345678，显示签收了但我根本没收到"}
    )
    env.asst.review(SHOP_A, "as_2003")
    env.clock.advance(hours=24)
    env.asst.refresh(SHOP_A, "as_2003")
    env.asst.negotiate(SHOP_A, "as_2003", operator=AGENT, note="买家说电话13812345678打不通")
    env.asst.negotiation_done(SHOP_A, "as_2003", operator=AGENT)
    env.asst.confirm(
        SHOP_A,
        "as_2003",
        ACT.REFUSE,
        confirmation=env.conf("as_2003"),
        reason_text="已联系买家 13812345678 确认收到",
    )
    blob = json.dumps([e.detail for e in env.audit.events], ensure_ascii=False)
    assert "根本没收到" not in blob
    assert "13812345678" not in blob
    stored = json.dumps(env.repo.get(SHOP_A, "as_2003").model_dump(mode="json"), ensure_ascii=False)
    assert "13812345678" not in stored


# ================================================================ 平台后台同意退货（R3 第二步）


def _second_step_after_buyer_ships_back(env, aftersale_id: str) -> AfterSaleReview:
    """买家寄回（WAIT_SELLER_CONFIRM）→ 刷新应回到待确认，建议同意退款（R3 第二步）。"""
    env.clock.advance(days=1)
    env.set_status(aftersale_id, RS.WAIT_SELLER_CONFIRM)
    r = env.asst.refresh(SHOP_A, aftersale_id)
    assert (r.state, r.suggestion) == (S.CONFIRM, A.AGREE)
    assert "R3" in r.matched_rules and "仓库确认收货后同意退款" in (r.reason or "")
    assert r.platform_status == RS.WAIT_SELLER_CONFIRM
    return r


def test_backend_agree_return_while_confirming_waits_for_return(env):
    """待确认时商家在平台后台同意了退货（淘宝常见，短信验证）：审核进入等待退货，
    不能进"平台已处理"终态——买家寄回后还要给"同意退款"的建议（§7.3.2 R3）。"""
    r = env.asst.review(SHOP_A, "as_2002")
    assert (r.state, r.suggestion) == (S.CONFIRM, A.AGREE_RETURN)
    env.set_status("as_2002", RS.WAIT_BUYER_RETURN)
    r = env.asst.refresh(SHOP_A, "as_2002")
    assert r.state == S.WAIT_RETURN
    assert r.platform_status == RS.WAIT_BUYER_RETURN
    assert r.platform_status_version == env.w.aftersale(SHOP_A, "as_2002").status_version
    # 买家还没寄回：保持等待退货
    env.clock.advance(hours=12)
    assert env.asst.refresh(SHOP_A, "as_2002").state == S.WAIT_RETURN

    r = _second_step_after_buyer_ships_back(env, "as_2002")
    r = env.asst.confirm(SHOP_A, "as_2002", ACT.AGREE_REFUND, confirmation=env.conf("as_2002"))
    assert r.state == S.AGREED
    assert [c["action"] for c in env.port.calls] == [ACT.AGREE_REFUND]
    assert [h["to"] for h in history(r)] == [
        "suggest",
        "confirm",
        "wait_return",
        "confirm",
        "agreed",
    ]


@pytest.mark.parametrize("path", ["evidence", "negotiate", "confirm", "new"])
def test_backend_agree_return_from_other_states_waits_for_return(env, path):
    """证据待补、协商中、客服确认时重读、首次审核时，平台已是等待买家退货：都进等待退货。"""
    if path == "evidence":
        env.asst.review(SHOP_A, "as_2004")  # R5 缺凭证 → EVIDENCE
        env.set_status("as_2004", RS.WAIT_BUYER_RETURN)
        r = env.asst.refresh(SHOP_A, "as_2004")
    elif path == "negotiate":
        env.asst.review(SHOP_A, "as_2004")
        env.set_status("as_2004", RS.PENDING_REVIEW, evidence_count=2)
        env.asst.refresh(SHOP_A, "as_2004")
        env.asst.negotiate(SHOP_A, "as_2004", operator=AGENT)
        env.set_status("as_2004", RS.WAIT_BUYER_RETURN)
        r = env.asst.negotiation_done(SHOP_A, "as_2004", operator=AGENT, outcome="同意退货")
    elif path == "confirm":
        env.asst.review(SHOP_A, "as_2004")
        env.set_status("as_2004", RS.PENDING_REVIEW, evidence_count=2)
        env.asst.refresh(SHOP_A, "as_2004")
        conf = env.conf("as_2004")
        env.set_status("as_2004", RS.WAIT_BUYER_RETURN)
        r = env.asst.confirm(SHOP_A, "as_2004", ACT.AGREE_RETURN, confirmation=conf)
        assert env.port.calls == []  # 重读发现平台已处理：不调写接口
    else:
        env.set_status("as_2004", RS.WAIT_BUYER_RETURN)
        r = env.asst.review(SHOP_A, "as_2004")
    assert r.state == S.WAIT_RETURN, path
    assert r.platform_status == RS.WAIT_BUYER_RETURN
    assert history(r)[-1]["to"] == "wait_return"
    env.set_status("as_2004", RS.WAIT_SELLER_CONFIRM)
    env.clock.advance(hours=1)
    r = env.asst.refresh(SHOP_A, "as_2004")
    assert (r.state, r.suggestion, r.matched_rules[0]) == (S.CONFIRM, A.AGREE, "R3")
