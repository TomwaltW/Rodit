"""显示签收但没收到（方案 §7.3.1、§11.3）：物流有结果时，第一条推荐是确定性模板——
签收细分状态 + 驿站 / 快递柜取件指引 + 还是没找到就在订单页申请售后（平台规则 APPLY_AFTERSALE）；
大模型提示词里也加一条对应的要求。

证据用工具层的真实分析（DefaultLogisticsAdvisor + logistics_content）和知识中心的平台规则文字，
和线上检索给出的一样；数据全部来自合成演示世界。
"""

from __future__ import annotations

from rodit.core import evidence as E
from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import GenerationConfig, QAGateConfig
from rodit.core.enums import DraftKind, EvidenceSource, Intent, LogisticsStatus, RouteAction
from rodit.core.errors import LLMTimeout
from rodit.core.evidence import PlatformRuleKey
from rodit.core.models import Evidence, EvidenceBundle, LogisticsAnalysis
from rodit.generation import DefaultAnswerGenerator
from rodit.generation import templates as T
from rodit.generation.prompt import NOT_RECEIVED_REQUIREMENT, build_generation_messages
from rodit.knowledge.platform_rules import platform_rule_evidence
from rodit.qa import DefaultQualityGate
from rodit.testing.fakes import FakeClock, FakeLLMClient, make_context
from rodit.testing.fixtures import SHOP_A, DemoWorld
from rodit.tools import DefaultLogisticsAdvisor
from rodit.tools.evidence import logistics_content

CFG = GenerationConfig()
LLM_TEXT = "亲，物流显示包裹已签收，麻烦您先看看代收点哦"


def logistics_evidence(world: DemoWorld, order_id: str) -> tuple[Evidence, LogisticsAnalysis]:
    trace = world.trace(SHOP_A, order_id)
    analysis = DefaultLogisticsAdvisor().analyze(trace, world.now)
    evidence = Evidence(
        id=E.tool_logistics(order_id),
        source=EvidenceSource.REALTIME,
        label=E.tool_label("logistics"),
        content=logistics_content(trace, analysis),
        data={
            "trace": trace.model_dump(mode="json"),
            "analysis": analysis.model_dump(mode="json"),
        },
        fetched_at=trace.fetched_at,
    )
    return evidence, analysis


def apply_rule() -> Evidence:
    return platform_rule_evidence("taobao", PlatformRuleKey.APPLY_AFTERSALE)


def llm_reply(request) -> dict:
    return {"answer": LLM_TEXT, "evidence_ids": [], "need_clarify": False}


def not_received_route(route):
    return route(Intent.NOT_RECEIVED, RouteAction.HIGH_RISK_ANSWER, rule=4, high_risk=True)


def generate(world, route, order_id: str, *, text: str = "显示签收了但我根本没收到", **kw):
    ev, _ = logistics_evidence(world, order_id)
    bundle = EvidenceBundle(evidences=[ev, apply_rule()])
    llm = kw.pop("llm", None) or FakeLLMClient(llm_reply)
    ctx = make_context(text, world, order_id=order_id, **kw)
    drafts = DefaultAnswerGenerator(llm, CFG).generate(ctx, not_received_route(route), bundle)
    return drafts, llm, ctx, bundle


# ================================================================ 模板


def test_station_signed_gives_pickup_guidance_and_apply_step(world, route):
    """o_1003：菜鸟驿站（幸福里店）代收。"""
    ev, analysis = logistics_evidence(world, "o_1003")
    assert analysis.status == LogisticsStatus.SIGNED and analysis.pickup_guidance
    draft = T.not_received(
        make_context("没收到", world, order_id="o_1003"), ev, analysis, apply_rule=apply_rule()
    )
    assert draft.kind == DraftKind.TEMPLATE and draft.template_id == T.NOT_RECEIVED
    assert draft.text.startswith("亲，")
    assert "驿站代收" in draft.text and "截至" in draft.text  # 签收细分状态 + 截至时间（§7.2）
    assert "菜鸟驿站" in draft.text
    assert analysis.pickup_guidance in draft.text  # 取件指引原样给出（只有地点，不含取件码）
    assert draft.text.endswith(f"{T.NOT_FOUND_APPLY}哦。")
    assert "申请售后" in draft.text
    assert draft.evidence_ids == [ev.id, "platform:taobao:apply_aftersale"]


def test_locker_signed_names_the_locker(world, route):
    """o_1010：丰巢快递柜代收。"""
    ev, analysis = logistics_evidence(world, "o_1010")
    draft = T.not_received(
        make_context("没收到", world, order_id="o_1010"), ev, analysis, apply_rule=apply_rule()
    )
    assert "丰巢" in draft.text and "快递柜代收" in draft.text
    assert analysis.pickup_guidance in draft.text
    assert "申请售后" in draft.text


def test_self_signed_has_no_pickup_place(world, route):
    """o_1004：本人签收，没有驿站 / 快递柜，不能套别的订单的取件点。"""
    ev, analysis = logistics_evidence(world, "o_1004")
    assert analysis.pickup_guidance is None
    draft = T.not_received(
        make_context("没收到", world, order_id="o_1004"), ev, analysis, apply_rule=apply_rule()
    )
    assert "本人签收" in draft.text
    assert "驿站" not in draft.text and "丰巢" not in draft.text and "取件" not in draft.text
    assert "申请售后" in draft.text


def test_apply_step_needs_the_platform_rule(world):
    ev, analysis = logistics_evidence(world, "o_1003")
    draft = T.not_received(make_context("没收到", world, order_id="o_1003"), ev, analysis)
    assert "菜鸟驿站" in draft.text
    assert "申请售后" not in draft.text  # 没有平台规则作依据就不说
    assert draft.evidence_ids == [ev.id]


def test_not_signed_only_reports_progress(world):
    """还没签收（o_1002 运输中）：只说物流进度，不引导申请售后。"""
    ev, analysis = logistics_evidence(world, "o_1002")
    assert analysis.status != LogisticsStatus.SIGNED
    draft = T.not_received(
        make_context("没收到", world, order_id="o_1002"), ev, analysis, apply_rule=apply_rule()
    )
    assert "截至" in draft.text and "申请售后" not in draft.text
    assert draft.evidence_ids == [ev.id]


def test_style_follows_shop_config(world):
    ev, analysis = logistics_evidence(world, "o_1003")
    ctx = make_context(
        "没收到", world, order_id="o_1003", config_update={"salutation": "您", "tone": "专业"}
    )
    draft = T.not_received(ctx, ev, analysis, apply_rule=apply_rule())
    assert draft.text.startswith("您好，")
    assert draft.text.endswith(f"{T.NOT_FOUND_APPLY}。")


def test_template_passes_hard_rules(world, route):
    """模板只跑硬规则（§5.5）：数字、时间都来自证据，不超长，没有承诺。"""
    gate = DefaultQualityGate(None, QAGateConfig(), InMemoryAuditSink(), FakeClock(world.now))
    for order_id in ("o_1003", "o_1010", "o_1004"):
        drafts, _, ctx, bundle = generate(world, route, order_id)
        result = gate.check(ctx, not_received_route(route), drafts[0], bundle)
        assert result.passed, (order_id, result.reasons)
        assert len(result.text) <= ctx.config.max_answer_chars


# ================================================================ 生成器


def test_generator_puts_the_template_first(world, route):
    drafts, llm, _, _ = generate(world, route, "o_1003")
    assert [d.kind for d in drafts] == [DraftKind.TEMPLATE, DraftKind.LLM]
    assert drafts[0].template_id == T.NOT_RECEIVED
    assert "菜鸟驿站" in drafts[0].text and "申请售后" in drafts[0].text
    assert drafts[1].text == LLM_TEXT
    # 模板占了一个名额：大模型只补一条
    assert len(llm.calls_for("generate")) == 1


def test_template_only_for_signed_not_received(world, route):
    """还没签收（o_1002 运输中）照常交给大模型；别的问题类型（物流查询）也不用这个模板。"""
    drafts, llm, _, _ = generate(world, route, "o_1002", config_update={"suggestion_count": 1})
    assert [d.kind for d in drafts] == [DraftKind.LLM]
    assert len(llm.calls_for("generate")) == 1
    ev, _ = logistics_evidence(world, "o_1003")
    tracking = DefaultAnswerGenerator(FakeLLMClient(llm_reply), CFG).generate(
        make_context("我的快递到哪了", world, order_id="o_1003"),
        route(Intent.LOGISTICS_TRACK),
        EvidenceBundle(evidences=[ev, apply_rule()]),
    )
    assert [d.template_id for d in tracking] == [T.LOGISTICS]
    assert "申请售后" not in tracking[0].text
    # 签收了的同一张单，问"没收到"就用签收未收到模板
    signed, _, _, _ = generate(world, route, "o_1003", config_update={"suggestion_count": 1})
    assert [d.template_id for d in signed] == [T.NOT_RECEIVED]


def test_one_slot_uses_only_the_template(world, route):
    drafts, llm, _, _ = generate(world, route, "o_1010", config_update={"suggestion_count": 1})
    assert [d.template_id for d in drafts] == [T.NOT_RECEIVED]
    assert "丰巢" in drafts[0].text
    assert llm.calls == []


def test_llm_failure_keeps_the_template_without_a_duplicate(world, route):
    """大模型超时（§12.2）：退回模板，不再补一条只有物流进度的重复模板。"""
    drafts, _, _, _ = generate(
        world, route, "o_1003", llm=FakeLLMClient(lambda r: LLMTimeout("超时"))
    )
    assert drafts[0].template_id == T.NOT_RECEIVED
    assert T.LOGISTICS not in [d.template_id for d in drafts]


# ================================================================ 提示词


def test_prompt_has_the_not_received_requirement(world, route):
    ev, _ = logistics_evidence(world, "o_1003")
    ctx = make_context("显示签收了但我没收到", world, order_id="o_1003")
    bundle = EvidenceBundle(evidences=[ev, apply_rule()])
    user = build_generation_messages(ctx, not_received_route(route), bundle, config=CFG)[1]
    assert f"- {NOT_RECEIVED_REQUIREMENT}" in user.content
    assert "驿站或快递柜" in NOT_RECEIVED_REQUIREMENT and "申请售后" in NOT_RECEIVED_REQUIREMENT
    # 安抚路由照样加（要求管内容，安抚管语气）
    soothe = route(Intent.NOT_RECEIVED, RouteAction.SOOTHE, rule=3, high_risk=True)
    angry = build_generation_messages(ctx, soothe, bundle, config=CFG)[1].content
    assert NOT_RECEIVED_REQUIREMENT in angry and "先用一句道歉或安抚的话开头" in angry
    # 其他类型不加
    other = build_generation_messages(ctx, route(Intent.LOGISTICS_TRACK), bundle, config=CFG)[1]
    assert NOT_RECEIVED_REQUIREMENT not in other.content
