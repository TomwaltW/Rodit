"""生成：规则 3 回答部分的闸门、站外话术引用平台规则、质量问题先致歉安抚（集成审查修复）。"""

from __future__ import annotations

import pytest

from rodit.core import evidence as E
from rodit.core.config import GenerationConfig
from rodit.core.enums import DraftKind, Intent, PolicyKey, RouteAction
from rodit.core.evidence import Need, PlatformRuleKey
from rodit.core.models import EvidenceBundle, RoutePart
from rodit.core.pipeline import ACTION_PLANS
from rodit.generation import DefaultAnswerGenerator, build_generation_messages
from rodit.generation import templates as T
from rodit.testing.fakes import FakeLLMClient, make_context

CFG = GenerationConfig()
APOLOGY = "先用一句道歉安抚的话开头"


def _reply(answer: str, *ids: str) -> dict:
    return {"answer": answer, "evidence_ids": list(ids), "need_clarify": False}


# ---------------------------------------------------------------- 规则 3：安抚 + 回答的闸门


def _soothe_route(route, intent: Intent, *, degraded: bool = False, high_risk: bool = False):
    r = route(Intent.PRICE_PROMO, RouteAction.SOOTHE, rule=3, high_risk=high_risk)
    return r.model_copy(update={"intent": intent, "degraded": degraded})


@pytest.mark.parametrize(
    ("intent", "degraded", "high_risk", "config_update"),
    [
        (Intent.PRICE_PROMO, False, True, {"disabled_intents": ["price_promo"]}),  # 店铺关了
        (Intent.PRICE_PROMO, True, True, {}),  # §12.2 降级时高风险类型不推荐
        (Intent.COMPLAINT, False, True, {}),  # 投诉只提醒人工
    ],
)
def test_soothe_gives_only_the_script_when_the_answer_is_gated(
    world, ev, route, intent, degraded, high_risk, config_update
):
    llm = FakeLLMClient(responses=[])
    ctx = make_context("太差了！！有没有优惠券啊，垃圾", world, config_update=config_update)
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.PROMOTIONS)])
    r = _soothe_route(route, intent, degraded=degraded, high_risk=high_risk)
    drafts = DefaultAnswerGenerator(llm, CFG).generate(ctx, r, bundle)
    assert [d.template_id for d in drafts] == [T.SOOTHE]
    assert llm.calls == []


def test_soothe_still_answers_an_open_intent(world, ev, route):
    llm = FakeLLMClient(lambda r: _reply("亲，店铺满 199 减 20 哦", "policy:promotions"))
    ctx = make_context("太差了！！有没有优惠券啊，垃圾", world)
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.PROMOTIONS)])
    r = _soothe_route(route, Intent.PRICE_PROMO, high_risk=True)
    drafts = DefaultAnswerGenerator(llm, CFG).generate(ctx, r, bundle)
    assert [d.template_id for d in drafts][:2] == [T.SOOTHE, None]
    assert drafts[1].kind == DraftKind.LLM


# ---------------------------------------------------------------- 站外话术的依据（§5.1、§5.6）


def test_off_platform_route_retrieves_the_platform_rule():
    assert ACTION_PLANS[RouteAction.OFF_PLATFORM_SCRIPT].retrieve
    assert E.needs_for(Intent.OFF_PLATFORM) == frozenset({Need.PLATFORM_RULE})
    assert E.platform_rules_for(Intent.OFF_PLATFORM) == (PlatformRuleKey.OFF_PLATFORM,)


def test_off_platform_script_cites_the_retrieved_rule(world, ev, route):
    rule = ev.platform_rule(PlatformRuleKey.OFF_PLATFORM, "平台禁止站外联系和交易")
    llm = FakeLLMClient(responses=[])
    drafts = DefaultAnswerGenerator(llm, CFG).generate(
        make_context("加个微信吧", world),
        route(Intent.OFF_PLATFORM, RouteAction.OFF_PLATFORM_SCRIPT, rule=1),
        EvidenceBundle(evidences=[rule]),
    )
    assert [d.evidence_ids for d in drafts] == [[rule.id]] and llm.calls == []


# ---------------------------------------------------------------- 收到的东西坏了（§7.3.1）


def test_quality_issue_prompt_asks_for_an_apology_first(world, ev, route):
    ctx = make_context("收到的衣服是坏的，破了个洞", world, order_id="o_1003")
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.QUALITY_ISSUE)])
    calm = route(Intent.QUALITY_ISSUE, RouteAction.HIGH_RISK_ANSWER, rule=4, high_risk=True)
    user = build_generation_messages(ctx, calm, bundle, config=CFG)[1].content
    assert APOLOGY in user
    # 分条推荐里的质量问题那一条也要
    split = route(
        Intent.QUALITY_ISSUE,
        RouteAction.HIGH_RISK_ANSWER,
        rule=5,
        parts=[],
        high_risk=True,
    )
    assert APOLOGY in build_generation_messages(ctx, split, bundle, config=CFG)[1].content
    # 安抚路由已经有"道歉或安抚"那一句，不重复
    angry = route(Intent.QUALITY_ISSUE, RouteAction.SOOTHE, rule=3, high_risk=True)
    angry_user = build_generation_messages(ctx, angry, bundle, config=CFG)[1].content
    assert "先用一句道歉或安抚的话开头" in angry_user and APOLOGY not in angry_user
    # 其他售后类型（§11.3 签收未收到）不加
    lost = route(Intent.NOT_RECEIVED, RouteAction.HIGH_RISK_ANSWER, rule=4, high_risk=True)
    assert APOLOGY not in build_generation_messages(ctx, lost, bundle, config=CFG)[1].content
    other = route(
        Intent.SHIPPING_POLICY,
        RouteAction.SPLIT_MULTI,
        rule=5,
        parts=[RoutePart(intent=Intent.SHIPPING_POLICY, confidence=0.6)],
    )
    assert APOLOGY not in build_generation_messages(ctx, other, bundle, config=CFG)[1].content
