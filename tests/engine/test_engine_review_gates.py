"""引擎：分条只带自己的风险提示、规则 3 回答部分的闸门、店铺关闭的原因、站外话术依据、
decision_log 写失败（集成审查修复）。

前半部分用 engine_support 的替身测 _plan_units；后半部分用演示单元（真实路由、检索、生成、质检，
离线大模型替身）复现审查里的场景。
"""

from __future__ import annotations

from engine_support import SOOTHE_SCRIPT, Kit, make_route

from rodit.app.demo import build_demo_unit, taobao_event
from rodit.app.service import AssistService
from rodit.core.enums import DraftKind, Intent, RouteAction
from rodit.core.errors import DecisionError, StorageUnavailable
from rodit.core.models import Draft
from rodit.engine.engine import REASON_DISABLED
from rodit.routing import ALERT_EMOTION, RISK_HINTS
from rodit.testing.fakes import FakeDecisionProvider
from rodit.testing.fixtures import SHOP_A

PRICE_HINT = RISK_HINTS[Intent.PRICE_PROMO]


def _soothe_and_answer(text: str) -> list[Draft]:
    return [
        Draft(text=SOOTHE_SCRIPT, kind=DraftKind.TEMPLATE, variant="soothe", template_id="soothe"),
        Draft(text=text, kind=DraftKind.LLM, evidence_ids=["policy:ship_time"]),
    ]


# ================================================================ _plan_units（替身）


def test_split_parts_carry_only_their_own_risk_hint(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.SIZE_ADVICE,
        RouteAction.SPLIT_MULTI,
        rule=5,
        parts=[(Intent.SIZE_ADVICE, 0.6), (Intent.PRICE_PROMO, 0.3)],
        risk_hints=[PRICE_HINT],
    )
    result = kit.engine().handle(kit.msg("穿什么码合适？能便宜点吗"))
    assert result is not None
    part_routes = [route for _, route in kit.retriever.calls]
    assert [(r.intent, r.risk_hints, r.high_risk) for r in part_routes] == [
        (Intent.SIZE_ADVICE, [], False),
        (Intent.PRICE_PROMO, [PRICE_HINT], True),
    ]
    assert [(c.intent, c.risk_hints) for c in result.suggestions] == [
        (Intent.SIZE_ADVICE, []),
        (Intent.PRICE_PROMO, [PRICE_HINT]),
    ]


def test_soothe_on_a_disabled_intent_keeps_only_the_script(kit: Kit) -> None:
    kit.set_shop_config(SHOP_A, disabled_intents=["price_promo"])
    kit.router.decide = make_route(
        Intent.PRICE_PROMO,
        RouteAction.SOOTHE,
        rule=3,
        high_risk=True,
        alerts=[ALERT_EMOTION],
        risk_hints=[ALERT_EMOTION],
    )
    kit.generator.script = [_soothe_and_answer("亲，店铺满 199 减 20 哦")]
    result = kit.engine().handle(kit.msg("太差了！！有券吗，垃圾"))
    assert result is not None
    assert [c.text for c in result.suggestions] == [SOOTHE_SCRIPT]


def test_soothe_degraded_high_risk_keeps_only_the_script(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.PRICE_PROMO, RouteAction.SOOTHE, rule=3, high_risk=True, degraded=True
    )
    kit.generator.script = [_soothe_and_answer("亲，店铺满 199 减 20 哦")]
    result = kit.engine().handle(kit.msg("太差了！！有券吗，垃圾"))
    assert result is not None
    assert [c.text for c in result.suggestions] == [SOOTHE_SCRIPT]


def test_gated_soothe_does_not_retrieve(kit: Kit) -> None:
    """规则 3 不附回答时只出安抚话术：不检索、不调平台工具、不给关闭的类型记知识缺失。"""
    kit.set_shop_config(SHOP_A, disabled_intents=["price_promo"])
    kit.router.decide = make_route(
        Intent.PRICE_PROMO, RouteAction.SOOTHE, rule=3, high_risk=True, alerts=[ALERT_EMOTION]
    )
    kit.generator.script = [_soothe_and_answer("亲，店铺满 199 减 20 哦")]
    result = kit.engine().handle(kit.msg("太差了！！有券吗，垃圾"))
    assert result is not None
    assert kit.retriever.calls == []
    assert [c.text for c in result.suggestions] == [SOOTHE_SCRIPT]
    assert result.missing_knowledge == []


def test_off_platform_script_survives_retrieval_failure(kit: Kit) -> None:
    """站外交易标准话术不依赖依据（§11.4）：检索失败照常出话术。"""
    kit.router.decide = make_route(
        Intent.OFF_PLATFORM, RouteAction.OFF_PLATFORM_SCRIPT, rule=1, confidence=0.97
    )
    kit.retriever.error = RuntimeError("boom")
    result = kit.engine().handle(kit.msg("加个微信吧，转账给你便宜点"))
    assert result is not None
    [card] = result.suggestions
    assert card.kind == DraftKind.TEMPLATE and "站外" in card.text
    assert result.no_suggestion_reason is None


def test_soothe_open_intent_keeps_script_and_answer(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.SHIPPING_POLICY, RouteAction.SOOTHE, rule=3)
    kit.generator.script = [_soothe_and_answer("亲，现货 48 小时内发货哦")]
    result = kit.engine().handle(kit.msg("太慢了！！什么时候发货"))
    assert result is not None
    assert [c.text for c in result.suggestions] == [SOOTHE_SCRIPT, "亲，现货 48 小时内发货哦"]


def test_disabled_gate_in_the_policy_reports_the_disabled_reason(kit: Kit) -> None:
    kit.set_shop_config(SHOP_A, disabled_intents=["invoice_gift"])
    kit.router.decide = make_route(Intent.INVOICE_GIFT, RouteAction.NO_SUGGESTION, rule=8)
    result = kit.engine().handle(kit.msg("能开发票吗"))
    assert result is not None
    assert result.no_suggestion_reason == REASON_DISABLED


# ================================================================ 演示单元（审查复现）

ANGRY_PRICE = "太差了！！有没有优惠券啊，别家便宜多了，垃圾"


def _say(service: AssistService, text: str, n: int, **kw):
    return service.handle_event(
        "B", taobao_event(text, msg_id=f"a2_{n}", session_id=f"a2_c{n}", **kw)
    )


def _cards(result):
    return [(c.kind, c.risk_hints) for c in result.suggestions]


def test_demo_angry_price_answer_carries_the_price_hint() -> None:
    result = _say(AssistService(build_demo_unit()), ANGRY_PRICE, 1, item_id="i_555")
    assert (result.route.action, result.route.rule) == (RouteAction.SOOTHE, 3)
    assert len(result.suggestions) == 2
    assert all(PRICE_HINT in c.risk_hints for c in result.suggestions)


def test_demo_degraded_angry_price_gets_only_the_soothe_script() -> None:
    down = FakeDecisionProvider({}, name="incloud_llm", fail=DecisionError("判断层不可用"))
    result = _say(
        AssistService(build_demo_unit(route_provider=down)), ANGRY_PRICE, 2, item_id="i_555"
    )
    assert result.route.degraded and result.route.action == RouteAction.SOOTHE
    assert [c.kind for c in result.suggestions] == [DraftKind.TEMPLATE]
    assert result.suggestions[0].text.startswith("亲，非常抱歉")


def test_demo_disabled_angry_price_gets_only_the_soothe_script() -> None:
    unit = build_demo_unit()
    shop = unit.repos.knowledge.get_shop(SHOP_A)
    overrides = {**shop.config_overrides, "disabled_intents": ["price_promo"]}
    unit.repos.knowledge.upsert_shop(shop.model_copy(update={"config_overrides": overrides}))
    service = AssistService(unit)
    angry = _say(service, ANGRY_PRICE, 3, item_id="i_555")
    assert [c.kind for c in angry.suggestions] == [DraftKind.TEMPLATE]
    calm = _say(service, "有没有优惠券啊，别家便宜多了", 4, item_id="i_555")
    assert calm.suggestions == [] and calm.no_suggestion_reason == REASON_DISABLED


def test_demo_disabled_complaint_still_alerts_a_human() -> None:
    unit = build_demo_unit()
    shop = unit.repos.knowledge.get_shop(SHOP_A)
    overrides = {**shop.config_overrides, "disabled_intents": ["complaint"]}
    unit.repos.knowledge.upsert_shop(shop.model_copy(update={"config_overrides": overrides}))
    result = _say(AssistService(unit), "你们这是虚假宣传，黑店", 5)
    assert result.route.action == RouteAction.HUMAN_PRIORITY
    assert "买家投诉 / 纠纷，请人工优先处理" in result.alerts


def test_demo_split_parts_do_not_share_risk_hints() -> None:
    result = _say(
        AssistService(build_demo_unit()),
        "穿什么码合适，身高165体重110斤？能便宜点吗",
        6,
        item_id="i_555",
    )
    assert result.route.action == RouteAction.SPLIT_MULTI
    hints = {c.intent: c.risk_hints for c in result.suggestions}
    assert hints[Intent.SIZE_ADVICE] == []
    assert hints[Intent.PRICE_PROMO] == [PRICE_HINT]


def test_demo_off_platform_card_cites_the_platform_rule() -> None:
    result = _say(AssistService(build_demo_unit()), "加个微信吧，转账给你便宜点", 7)
    assert result.route.action == RouteAction.OFF_PLATFORM_SCRIPT
    [card] = result.suggestions
    assert [e.label for e in card.evidence] == ["平台规则 · 淘宝站外交易"]


def test_demo_decision_log_failure_still_recommends() -> None:
    unit = build_demo_unit()

    def broken(entry) -> None:  # noqa: ANN001
        raise StorageUnavailable("decision_log 写入失败")

    unit.repos.audit.log_decision = broken  # type: ignore[method-assign]
    result = _say(AssistService(unit), "什么时候发货", 8)
    assert result.route is not None and "router" not in result.degraded
    assert len(result.suggestions) >= 1
