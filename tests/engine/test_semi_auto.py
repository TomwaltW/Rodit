"""半自动（方案 §5.6 "高置信度的简单问题，推荐直接填进输入框，客服按发送即可"、§13.4）：
推荐卡片的 prefill。

只在智能辅助下、最多一张（第一张），店铺开了这个类型、规则 6 的正常回答、置信度 ≥
routing.semi_auto_threshold、卡片通过质检且为高、没有风险提示、提醒和降级才打；每一条不满足都不打。
客服仍要按发送：推荐记录和反馈口径不变。
"""

from __future__ import annotations

from typing import Any

import pytest
from engine_support import (
    Kit,
    bundle_of,
    make_route,
    policy_evidence,
    qa_fail,
    qa_pass,
)

from rodit.core.audit import AuditAction
from rodit.core.config import RoutingConfig
from rodit.core.enums import Confidence, DraftKind, Intent, Mode, RouteAction, SuggestionStatus
from rodit.core.models import Draft, EngineResult, QAResult, ShopConfig, SuggestionCard
from rodit.engine.semi_auto import (
    BLOCK_ALERTS,
    BLOCK_CARD_CONFIDENCE,
    BLOCK_CLARIFY,
    BLOCK_CONFIDENCE,
    BLOCK_DEGRADED,
    BLOCK_HIGH_RISK,
    BLOCK_INTENT,
    BLOCK_MODE,
    BLOCK_QA,
    BLOCK_RISK_HINTS,
    BLOCK_ROUTE,
    prefill_blocker,
)
from rodit.routing.policy import ALERT_EMOTION, ALERT_MULTI_UNSPLIT
from rodit.testing.fixtures import BUYER_DOUYIN, SHOP_A, SHOP_B, SHOP_D

SHIP = Intent.SHIPPING_POLICY


def semi(kit: Kit, *intents: Intent, shop_id: str = SHOP_A, **overrides: Any) -> None:
    kit.set_shop_config(
        shop_id, semi_auto_intents=[i.value for i in (intents or (SHIP,))], **overrides
    )


def ask(kit: Kit, text: str = "什么时候发货", **kw: Any) -> EngineResult:
    result = kit.engine().handle(kit.msg(text, **kw))
    assert result is not None
    return result


def prefills(result: EngineResult) -> list[bool]:
    return [c.prefill for c in result.suggestions]


# ================================================================ 打上 prefill


def test_top_card_is_prefilled_and_audited(kit: Kit) -> None:
    semi(kit)
    result = ask(kit)
    assert result.mode == Mode.ASSIST
    assert prefills(result) == [True]
    card = result.suggestions[0]
    assert card.confidence == Confidence.HIGH and card.suggestion_id is not None
    [suggest] = kit.events(AuditAction.SUGGEST)
    assert suggest.detail["prefill"] == [card.suggestion_id]
    # 客服仍要按发送：推荐记录照常是"已展示"，没有自动发送，也不转人工
    record = kit.suggestions.get(SHOP_A, card.suggestion_id)
    assert record is not None and record.status == SuggestionStatus.SHOWN
    assert result.auto_reply is None and result.transfer_to_human is False


def test_threshold_boundary(kit: Kit) -> None:
    semi(kit)
    kit.router.decide = make_route(SHIP, confidence=0.9)  # 刚好等于 semi_auto_threshold
    assert prefills(ask(kit)) == [True]
    kit.router.decide = make_route(SHIP, confidence=0.899)
    assert prefills(ask(kit)) == [False]


def test_threshold_comes_from_routing_config(kit: Kit) -> None:
    semi(kit)
    kit.config = kit.config.model_copy(update={"routing": RoutingConfig(semi_auto_threshold=0.95)})
    kit.router.decide = make_route(SHIP, confidence=0.93)
    assert prefills(ask(kit)) == [False]
    kit.router.decide = make_route(SHIP, confidence=0.96)
    assert prefills(ask(kit)) == [True]


def test_faq_and_template_cards_can_be_prefilled(kit: Kit) -> None:
    semi(kit, Intent.GREETING, SHIP)
    kit.generator.script = [[Draft(text="亲，默认发中通快递哦~", kind=DraftKind.FAQ, faq_id=1)]]
    assert prefills(ask(kit, "发什么快递")) == [True]
    kit.router.decide = make_route(Intent.GREETING, confidence=0.95)
    kit.generator.script = [
        [Draft(text="亲，在的~", kind=DraftKind.TEMPLATE, template_id="greeting")]
    ]
    assert prefills(ask(kit, "在吗")) == [True]


def test_only_the_top_card_and_never_a_lower_one(kit: Kit) -> None:
    semi(kit, suggestion_count=2)
    first = Draft(
        text="亲，48 小时内发货哦~", kind=DraftKind.LLM, evidence_ids=["policy:ship_time"]
    )
    second = Draft(
        text="亲，付款后两天内发出~", kind=DraftKind.LLM, evidence_ids=["policy:ship_time"]
    )
    kit.generator.script = [[first, second]]
    result = ask(kit)
    assert prefills(result) == [True, False]
    # 第一张不满足（置信度为中）：第二张满足也不往后找
    kit.generator.script = [[first, second]]
    kit.gate.rule = lambda d: qa_pass(d, grounded=0.8) if d.text == first.text else None
    assert prefills(ask(kit)) == [False, False]
    suggest = kit.events(AuditAction.SUGGEST)[-1]
    assert "prefill" not in suggest.detail


# ================================================================ 每一条不满足都不打


def _shadow(kit: Kit) -> dict[str, Any]:
    semi(kit, shop_id=SHOP_D)
    return {"shop_id": SHOP_D, "buyer_ref": BUYER_DOUYIN}


def _auto(kit: Kit) -> dict[str, Any]:
    semi(kit, auto_reply_enabled=True, auto_reply_schedule="always")
    kit.router.decide = make_route(SHIP, confidence=0.97, auto_eligible=True)
    return {"mode": Mode.AUTO}


def _intent_not_enabled(kit: Kit) -> dict[str, Any]:
    semi(kit, Intent.INVOICE_GIFT)
    return {}


def _clarify_route(kit: Kit) -> dict[str, Any]:
    semi(kit)
    kit.router.decide = make_route(SHIP, RouteAction.CLARIFY, rule=7, confidence=0.95)
    return {}


def _clarify_draft(kit: Kit) -> dict[str, Any]:
    semi(kit)
    kit.retriever.bundles[SHIP] = bundle_of(policy_evidence(), clarify_question="请问是哪一款呢？")
    return {}


def _high_risk(kit: Kit) -> dict[str, Any]:
    # 配置里被直接写进了高风险类型（接口不让开）：引擎照样不打
    semi(kit, Intent.PRICE_PROMO)
    kit.router.decide = make_route(Intent.PRICE_PROMO, confidence=0.95)
    return {}


def _high_risk_rule4(kit: Kit) -> dict[str, Any]:
    semi(kit, Intent.PRICE_PROMO)
    kit.router.decide = make_route(
        Intent.PRICE_PROMO,
        RouteAction.HIGH_RISK_ANSWER,
        rule=4,
        confidence=0.95,
        high_risk=True,
        risk_hints=["涉及价格承诺，请确认"],
    )
    return {}


def _low_confidence(kit: Kit) -> dict[str, Any]:
    semi(kit)
    kit.router.decide = make_route(SHIP, confidence=0.86)  # 卡片仍为高（≥ 0.85）
    return {}


def _medium_card(kit: Kit) -> dict[str, Any]:
    semi(kit)
    kit.gate.rule = lambda d: qa_pass(d, grounded=0.85)
    return {}


def _risk_hint(kit: Kit) -> dict[str, Any]:
    semi(kit)
    kit.retriever.bundles[SHIP] = bundle_of(policy_evidence(), alerts=["物流停滞超过 48 小时"])
    return {}


def _route_alert(kit: Kit) -> dict[str, Any]:
    semi(kit)
    kit.router.decide = make_route(SHIP, confidence=0.95, alerts=[ALERT_MULTI_UNSPLIT])
    return {}


def _soothe(kit: Kit) -> dict[str, Any]:
    semi(kit)
    kit.router.decide = make_route(
        SHIP,
        RouteAction.SOOTHE,
        rule=3,
        confidence=0.95,
        emotion=3.5,
        alerts=[ALERT_EMOTION],
        risk_hints=[ALERT_EMOTION],
    )
    return {}


def _unresolved(kit: Kit) -> dict[str, Any]:
    semi(kit)
    kit.conversations.add_message(kit.msg("没听懂"))
    return {"text": "你答非所问"}  # 连续两轮没听懂：提醒客服接管（§5.7）


def _split(kit: Kit) -> dict[str, Any]:
    semi(kit, SHIP, Intent.INVOICE_GIFT)
    kit.router.decide = make_route(
        SHIP,
        RouteAction.SPLIT_MULTI,
        rule=5,
        confidence=0.95,
        parts=[(SHIP, 0.95), (Intent.INVOICE_GIFT, 0.92)],
    )
    return {}


def _route_degraded(kit: Kit) -> dict[str, Any]:
    semi(kit)
    kit.router.decide = make_route(SHIP, confidence=0.95, degraded=True, provider="rules")
    return {}


def _qa_degraded(kit: Kit) -> dict[str, Any]:
    semi(kit)
    kit.gate.rule = lambda d: qa_pass(d, degraded=True)
    return {}


def _tool_degraded(kit: Kit) -> dict[str, Any]:
    semi(kit)
    kit.tools.fail_on = {"list_recent_orders"}
    return {}


def _dropped_faq(kit: Kit) -> dict[str, Any]:
    """FAQ 标准回答没过质检（提醒客服改知识库）：后面的大模型草稿通过了也不填。"""
    semi(kit)
    faq = Draft(text="亲，一定 48 小时内发货", kind=DraftKind.FAQ, faq_id=2)
    llm = Draft(text="亲，48 小时内发货哦~", kind=DraftKind.LLM, evidence_ids=["policy:ship_time"])
    kit.generator.script = [[faq, llm]]
    return {}


BLOCKERS = {
    "shadow": _shadow,
    "auto": _auto,
    "intent_not_enabled": _intent_not_enabled,
    "clarify_route": _clarify_route,
    "clarify_draft": _clarify_draft,
    "high_risk": _high_risk,
    "high_risk_rule4": _high_risk_rule4,
    "low_confidence": _low_confidence,
    "medium_card": _medium_card,
    "risk_hint": _risk_hint,
    "route_alert": _route_alert,
    "soothe": _soothe,
    "unresolved": _unresolved,
    "split": _split,
    "route_degraded": _route_degraded,
    "qa_degraded": _qa_degraded,
    "tool_degraded": _tool_degraded,
    "dropped_faq": _dropped_faq,
}


@pytest.mark.parametrize("name", list(BLOCKERS))
def test_each_condition_blocks_prefill(kit: Kit, name: str) -> None:
    kw = BLOCKERS[name](kit)
    result = ask(kit, **kw)
    assert result.suggestions, name  # 推荐照常给（只是不填进输入框）
    assert not any(prefills(result)), name
    assert all(
        "prefill" not in e.detail
        for e in kit.events(AuditAction.SUGGEST, kw.get("shop_id", SHOP_A))
    )


def test_qa_failure_gives_no_card_at_all(kit: Kit) -> None:
    semi(kit)
    kit.gate.rule = lambda d: qa_fail(d, ["绝对化用语"])
    result = ask(kit)
    assert result.suggestions == []


def test_storage_failure_drops_prefill(kit: Kit) -> None:
    """推荐记录写不进去（没有推荐编号）：客服按发送后记不了反馈，不填进输入框。"""
    semi(kit)

    class Broken:
        def __getattr__(self, name: str) -> Any:
            return getattr(kit.suggestions, name)

        def add(self, record: Any) -> Any:
            raise RuntimeError("db down")

    result = kit.engine(suggestions=Broken()).handle(kit.msg("什么时候发货"))
    assert result is not None and result.suggestions
    assert result.suggestions[0].suggestion_id is None
    assert prefills(result) == [False]
    assert "storage" in result.degraded
    [suggest] = kit.events(AuditAction.SUGGEST)
    assert "prefill" not in suggest.detail


def test_semi_auto_is_per_shop(kit: Kit) -> None:
    """店铺 B 给发货开了半自动，店铺 A 没开：A 的推荐不填（§9.2 配置按店铺）。"""
    semi(kit, shop_id=SHOP_B)
    result = ask(kit)
    assert prefills(result) == [False]


def test_gated_auto_message_in_service_hours_can_be_prefilled(kit: Kit) -> None:
    """机器人接待的消息在客服工作时间里退回智能辅助、转人工：接手的客服照样能用半自动。"""
    semi(kit, auto_reply_enabled=True, service_hours=["00:00-24:00"])
    result = ask(kit, mode=Mode.AUTO)
    assert result.mode == Mode.ASSIST and result.transfer_to_human is True
    assert prefills(result) == [True]


# ================================================================ 规则本身（prefill_blocker）


def _inputs(**kw: Any) -> dict[str, Any]:
    route = kw.pop("route", make_route(SHIP, confidence=0.95))
    draft = kw.pop("draft", Draft(text="亲，48 小时内发货", kind=DraftKind.LLM))
    base: dict[str, Any] = {
        "mode": Mode.ASSIST,
        "config": ShopConfig(semi_auto_intents=[SHIP]),
        "routing": RoutingConfig(),
        "route": route,
        "card": SuggestionCard(
            suggestion_id=1,
            text=draft.text,
            intent=route.intent,
            kind=draft.kind,
            confidence=Confidence.HIGH,
        ),
        "draft": draft,
        "qa": QAResult(passed=True, text=draft.text),
        "alerts": [],
        "degraded": [],
    }
    base.update(kw)
    return base


def test_prefill_blocker_ok() -> None:
    assert prefill_blocker(**_inputs()) is None


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({"mode": Mode.SHADOW}, BLOCK_MODE),
        ({"mode": Mode.AUTO}, BLOCK_MODE),
        ({"config": ShopConfig()}, BLOCK_INTENT),
        ({"route": make_route(SHIP, RouteAction.CLARIFY, rule=7, confidence=0.95)}, BLOCK_ROUTE),
        (
            {"route": make_route(SHIP, RouteAction.OFF_PLATFORM_SCRIPT, rule=1, confidence=0.95)},
            BLOCK_ROUTE,
        ),
        (
            {"route": make_route(SHIP, rule=6, confidence=0.95, parts=[(SHIP, 0.9)])},
            BLOCK_ROUTE,
        ),
        ({"route": make_route(SHIP, confidence=0.95, high_risk=True)}, BLOCK_HIGH_RISK),
        (
            {
                "route": make_route(Intent.PRICE_PROMO, confidence=0.95),
                "config": ShopConfig(semi_auto_intents=[Intent.PRICE_PROMO]),
            },
            BLOCK_HIGH_RISK,
        ),
        (
            {
                "route": make_route(Intent.COMPLAINT, confidence=0.95),
                "config": ShopConfig(semi_auto_intents=[Intent.COMPLAINT]),
                "routing": RoutingConfig(high_risk=[]),
            },
            BLOCK_HIGH_RISK,
        ),
        ({"route": make_route(SHIP, confidence=0.95, degraded=True)}, BLOCK_DEGRADED),
        ({"route": make_route(SHIP, confidence=0.8)}, BLOCK_CONFIDENCE),
        ({"qa": QAResult(passed=False, text="x")}, BLOCK_QA),
        (
            {
                "draft": Draft(
                    text="请问哪一款？",
                    kind=DraftKind.TEMPLATE,
                    need_clarify=True,
                    variant="clarify",
                )
            },
            BLOCK_CLARIFY,
        ),
        ({"alerts": ["买家情绪激动"]}, BLOCK_ALERTS),
        ({"degraded": ["audit"]}, BLOCK_DEGRADED),
        ({"qa": QAResult(passed=True, text="x", degraded=True)}, BLOCK_DEGRADED),
    ],
)
def test_prefill_blocker_conditions(kw: dict[str, Any], expected: str) -> None:
    assert prefill_blocker(**_inputs(**kw)) == expected


def test_prefill_blocker_card_conditions() -> None:
    base = _inputs()
    medium = base["card"].model_copy(update={"confidence": Confidence.MEDIUM})
    assert prefill_blocker(**{**base, "card": medium}) == BLOCK_CARD_CONFIDENCE
    risky = base["card"].model_copy(update={"risk_hints": ["请核实"]})
    assert prefill_blocker(**{**base, "card": risky}) == BLOCK_RISK_HINTS
    other = base["card"].model_copy(update={"intent": Intent.INVOICE_GIFT})
    assert prefill_blocker(**{**base, "card": other}) == BLOCK_INTENT
