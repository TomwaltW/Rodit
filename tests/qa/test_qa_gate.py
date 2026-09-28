"""质检把关（方案 §5.5 两道关 + 放行规则、§11.5 带原因重新生成、§12.2 降级）。"""

from __future__ import annotations

import json

import pytest

from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import GenerationConfig, QAGateConfig
from rodit.core.decision import DecisionResponse, NoulAnswer
from rodit.core.enums import DecisionStage, DraftKind, Intent, PolicyKey, Role, RouteAction
from rodit.core.errors import ComplianceError, DecisionError, LLMError
from rodit.core.models import Draft, EvidenceBundle, LogisticsAnalysis
from rodit.core.ports import QualityGate
from rodit.core.questions import QA_MODEL, QA_QUESTIONS, QA_QUESTIONS_VER
from rodit.generation import DefaultAnswerGenerator
from rodit.generation import templates as T
from rodit.qa import QA_UNAVAILABLE, DefaultQualityGate
from rodit.testing.fakes import (
    FakeClock,
    FakeDecisionProvider,
    FakeLLMClient,
    make_context,
    qa_answers,
)

CFG = QAGateConfig()


@pytest.fixture
def audit() -> InMemoryAuditSink:
    return InMemoryAuditSink()


def make_gate(provider, audit, config=CFG) -> DefaultQualityGate:
    return DefaultQualityGate(provider, config, audit, FakeClock())


def llm_draft(text: str, *ids: str) -> Draft:
    return Draft(text=text, kind=DraftKind.LLM, evidence_ids=list(ids))


def test_implements_protocol(audit):
    assert isinstance(make_gate(None, audit), QualityGate)


def test_rejects_provider_that_cannot_take_buyer_data(audit):
    jev_like = FakeDecisionProvider(qa_answers(), name="jev", accepts_buyer_data=False)
    with pytest.raises(ComplianceError):
        make_gate(jev_like, audit)


def test_llm_draft_passes_both_gates(world, ev, route, audit):
    provider = FakeDecisionProvider(qa_answers())
    gate = make_gate(provider, audit)
    ctx = make_context("这个是纯棉的吗？", world, item_id="i_555")
    bundle = EvidenceBundle(evidences=[ev.item_attr("面料")])
    draft = llm_draft("**亲**，这款面料是棉 95%、氨纶 5% 哦", "item:i_555:attr:面料")
    result = gate.check(ctx, route(Intent.PRODUCT_SPEC), draft, bundle)
    assert result.passed and result.model_checked and not result.degraded
    assert result.text == "亲，这款面料是棉 95%、氨纶 5% 哦"
    assert [h.rule for h in result.hard_hits] == ["format"]
    assert result.reasons == []
    assert result.provider == "fake" and result.answers is not None
    assert result.scores == {
        "grounded": 0.96,
        "answers_question": 0.95,
        "promise_risk": 0.05,
        "platform_violation": 0.02,
        "tone": 3.5,
    }
    # 判断层请求：QA 模型、QA 问题、state 里是清洗后的文本
    (request,) = provider.requests
    assert request.model == QA_MODEL
    assert set(request.questions) == set(QA_QUESTIONS)
    assert request.state["answer"] == result.text
    assert request.state["evidence_ids"] == ["item:i_555:attr:面料"]
    # 一条 decision_log（stage=qa），不写 audit_log
    (entry,) = audit.decisions
    assert entry.stage == DecisionStage.QA and entry.questions_ver == QA_QUESTIONS_VER
    assert entry.shop_id == ctx.shop_id and entry.conversation_id == ctx.conversation_id
    assert entry.provider == "fake" and entry.route_result == "pass"
    assert entry.answers["grounded"] == {"type": "noul", "noul": 0.96}
    assert audit.events == []


@pytest.mark.parametrize(
    ("scores", "passed", "reason"),
    [
        ({"grounded": 0.8}, True, None),
        ({"grounded": 0.79}, False, "回答里有依据中找不到的内容（grounded 0.79"),
        ({"promise_risk": 0.3}, True, None),
        ({"promise_risk": 0.31}, False, "包含依据之外的承诺"),
        ({"platform_violation": 0.2}, True, None),
        ({"platform_violation": 0.21}, False, "可能包含或引导站外联系、站外交易"),
        ({"answers_question": 0.5}, True, None),
        ({"answers_question": 0.49}, False, "没有回应买家的问题"),
        ({"tone": 0.0}, True, None),  # 语气只打分，不拦截
    ],
)
def test_threshold_boundaries(world, ev, route, audit, scores, passed, reason):
    gate = make_gate(FakeDecisionProvider(qa_answers(**scores)), audit)
    ctx = make_context("纯棉吗", world, item_id="i_555")
    bundle = EvidenceBundle(evidences=[ev.item_attr("面料")])
    result = gate.check(ctx, route(Intent.PRODUCT_SPEC), llm_draft("亲，是棉 95% 的哦"), bundle)
    assert result.passed is passed
    assert result.model_checked
    if reason:
        assert len(result.reasons) == 1 and result.reasons[0].startswith(reason)
        assert audit.decisions[0].route_result == "block"
    else:
        assert result.reasons == []


def test_clarify_draft_is_not_blocked_for_not_answering(world, ev, route, audit):
    """§5.4 need_clarify：追问本来就不直接回答问题，不看 answers_question。"""
    gate = make_gate(FakeDecisionProvider(qa_answers(answers_question=0.1)), audit)
    ctx = make_context("我穿哪个码", world, item_id="i_777")
    bundle = EvidenceBundle(evidences=[ev.item_attr("面料", item_id="i_777")])
    ask = Draft(
        text="亲，请问您平时穿什么码呢？", kind=DraftKind.LLM, need_clarify=True, variant="clarify"
    )
    result = gate.check(ctx, route(Intent.SIZE_ADVICE), ask, bundle)
    assert result.passed and result.scores["answers_question"] == 0.1
    answer = llm_draft("亲，这款是微弹的")
    assert not gate.check(ctx, route(Intent.SIZE_ADVICE), answer, bundle).passed


def test_multiple_model_reasons(world, ev, route, audit):
    gate = make_gate(FakeDecisionProvider(qa_answers(grounded=0.5, promise_risk=0.86)), audit)
    ctx = make_context("明天能到吗", world, order_id="o_1002")
    result = gate.check(
        ctx,
        route(Intent.LOGISTICS_TRACK),
        llm_draft("亲，运输中哦"),
        EvidenceBundle(evidences=[ev.o_1002()]),
    )
    assert not result.passed and len(result.reasons) == 2
    assert "promise_risk 0.86" in result.reasons[1]


@pytest.mark.parametrize(
    "draft",
    [
        Draft(text="亲，默认发中通快递哦", kind=DraftKind.FAQ, evidence_ids=["faq:1"], faq_id=1),
        Draft(text="亲，我帮您查一下，请稍等哦。", kind=DraftKind.TEMPLATE, template_id="x"),
    ],
)
def test_faq_and_template_skip_model_check(world, ev, route, audit, draft):
    provider = FakeDecisionProvider(qa_answers(grounded=0.0))
    gate = make_gate(provider, audit)
    ctx = make_context("发什么快递", world)
    bundle = EvidenceBundle(evidences=[ev.faq("发什么快递", 0.95, faq_id=1)])
    result = gate.check(ctx, route(Intent.SHIPPING_POLICY), draft, bundle)
    assert result.passed and not result.model_checked and not result.degraded
    assert result.provider is None and result.scores == {}
    assert provider.requests == [] and audit.decisions == []


def test_template_still_runs_hard_rules(world, route, audit):
    gate = make_gate(None, audit)
    ctx = make_context("是正品吗", world)
    draft = Draft(text="亲，假一赔十哦", kind=DraftKind.TEMPLATE, template_id="x")
    result = gate.check(ctx, route(Intent.COMPARE_AUTHENTIC), draft, EvidenceBundle())
    assert not result.passed and result.reasons == ["包含商家禁用词：假一赔十"]


def test_hard_rule_block_and_model_reasons_combined(world, ev, route, audit):
    """§11.5：promise_risk 0.86，而且"明天""一定"在证据里找不到 → 拦截，原因合在一起。"""
    provider = FakeDecisionProvider(qa_answers(promise_risk=0.86))
    gate = make_gate(provider, audit)
    ctx = make_context("明天能到吗？", world, order_id="o_1002")
    result = gate.check(
        ctx,
        route(Intent.LOGISTICS_TRACK),
        llm_draft("亲，明天一定能到哦！", "tool:logistics:o_1002"),
        EvidenceBundle(evidences=[ev.o_1002()]),
    )
    assert not result.passed and result.model_checked
    assert result.reasons[:2] == ["包含绝对化用语：一定", "包含依据中没有的时效承诺：明天"]
    assert len(result.reasons) == 3 and "promise_risk 0.86" in result.reasons[2]
    assert result.scores["promise_risk"] == 0.86
    (entry,) = audit.decisions
    assert entry.route_result == "block"
    # 判断层说没问题，硬规则命中也照样拦截
    ok_model = make_gate(FakeDecisionProvider(qa_answers()), InMemoryAuditSink())
    blocked = ok_model.check(
        ctx,
        route(Intent.LOGISTICS_TRACK),
        llm_draft("亲，明天一定能到哦！"),
        EvidenceBundle(evidences=[ev.o_1002()]),
    )
    assert not blocked.passed and blocked.model_checked
    assert blocked.reasons == ["包含绝对化用语：一定", "包含依据中没有的时效承诺：明天"]


def test_degraded_keeps_hard_reasons(world, ev, route, audit):
    gate = make_gate(None, audit)
    ctx = make_context("明天能到吗？", world, order_id="o_1002")
    result = gate.check(
        ctx,
        route(Intent.LOGISTICS_TRACK),
        llm_draft("亲，明天能到哦"),
        EvidenceBundle(evidences=[ev.o_1002()]),
    )
    assert result.degraded and not result.passed
    assert result.reasons == ["包含依据中没有的时效承诺：明天", QA_UNAVAILABLE]


def test_no_provider_degrades_llm_drafts(world, ev, route, audit):
    gate = make_gate(None, audit)
    ctx = make_context("纯棉吗", world, item_id="i_555")
    result = gate.check(
        ctx,
        route(Intent.PRODUCT_SPEC),
        llm_draft("亲，是棉 95% 的"),
        EvidenceBundle(evidences=[ev.item_attr("面料")]),
    )
    assert not result.passed and result.degraded and not result.model_checked
    assert result.reasons == [QA_UNAVAILABLE]
    assert audit.decisions == []


@pytest.mark.parametrize(
    "failure", [DecisionError("判断层超时"), LLMError("大模型限流")], ids=["decision", "llm"]
)
def test_provider_failure_degrades(world, ev, route, audit, failure):
    provider = FakeDecisionProvider(qa_answers(), fail=failure)
    gate = make_gate(provider, audit)
    ctx = make_context("纯棉吗", world, item_id="i_555")
    result = gate.check(
        ctx,
        route(Intent.PRODUCT_SPEC),
        llm_draft("亲，是棉 95% 的"),
        EvidenceBundle(evidences=[ev.item_attr("面料")]),
    )
    assert not result.passed and result.degraded
    assert result.reasons == [QA_UNAVAILABLE]
    assert len(provider.requests) == 1
    assert audit.decisions == []


def test_incomplete_answers_degrade(world, ev, route, audit):
    """判断层实现没有校验就返回了不完整答案：gate 自己再校验一次。"""

    class Sloppy:
        name = "sloppy"
        accepts_buyer_data = True

        def decide(self, request):

            return DecisionResponse(model=request.model, answers={"grounded": NoulAnswer(noul=1)})

    gate = make_gate(Sloppy(), audit)
    ctx = make_context("纯棉吗", world, item_id="i_555")
    result = gate.check(
        ctx,
        route(Intent.PRODUCT_SPEC),
        llm_draft("亲，是棉 95% 的"),
        EvidenceBundle(evidences=[ev.item_attr("面料")]),
    )
    assert result.degraded and not result.passed


def test_empty_answer_is_blocked(world, route, audit):
    gate = make_gate(FakeDecisionProvider(qa_answers()), audit)
    ctx = make_context("在吗", world)
    draft = Draft(text="```\n```", kind=DraftKind.TEMPLATE, template_id="x")
    result = gate.check(ctx, route(Intent.GREETING), draft, EvidenceBundle())
    assert not result.passed and result.reasons == ["回答为空"]


def test_no_pii_reaches_decision_request_or_log(world, ev, route, audit):
    provider = FakeDecisionProvider(qa_answers())
    gate = make_gate(provider, audit)
    ctx = make_context(
        "我手机 13812345678，改寄到杭州市西湖区文三路 100 号可以吗",
        world,
        order_id="o_1001",
        history=[(Role.BUYER, "我住幸福小区 3 栋 501")],
    )
    draft = llm_draft(
        "亲，您是想改寄到文三路 100 号对吗？发货前可以修改地址哦", "policy:modify_order"
    )
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.MODIFY_ORDER)])
    result = gate.check(ctx, route(Intent.MODIFY_ORDER), draft, bundle)
    assert result.model_checked
    payload = json.dumps(provider.requests[0].model_dump(), ensure_ascii=False)
    payload += json.dumps([d.model_dump(mode="json") for d in audit.decisions], ensure_ascii=False)
    assert "13812345678" not in payload and "文三路" not in payload
    assert "幸福小区" not in payload
    assert ctx.buyer_ref not in payload
    # 展示给客服的文本不遮（是给这位买家看的回答）
    assert result.text.startswith("亲，您是想改寄到文三路 100 号")
    # 回答复述了手机号：硬规则拦截，判断层请求和原因里都没有号码原文
    leaky = gate.check(
        ctx, route(Intent.MODIFY_ORDER), llm_draft("亲，已记下您的手机 13812345678"), bundle
    )
    assert not leaky.passed and leaky.model_checked
    assert "13812345678" not in json.dumps(provider.requests[1].model_dump(), ensure_ascii=False)
    assert "13812345678" not in "".join(leaky.reasons)


def test_generated_templates_pass_the_gate(world, ev, route, audit):
    """模板和标准话术要能过自己的硬规则（§11.2、§11.4）。"""
    gate = make_gate(None, audit)
    cases = [
        (make_context("加个微信", world), T.off_platform_script(make_context("加", world)), None),
        (
            make_context("在吗", world),
            T.greeting(
                make_context("在吗", world),
                EvidenceBundle(evidences=[ev.policy(PolicyKey.SERVICE_HOURS)]),
            ),
            EvidenceBundle(evidences=[ev.policy(PolicyKey.SERVICE_HOURS)]),
        ),
        (make_context("?", world), T.soothe(make_context("?", world)), None),
        (make_context("?", world), T.tool_unavailable(make_context("?", world)), None),
        (make_context("?", world), T.clarify_item(make_context("?", world)), None),
        (make_context("?", world), T.clarify_order(make_context("?", world)), None),
    ]
    for ctx, draft, bundle in cases:
        result = gate.check(ctx, route(Intent.OTHER), draft, bundle or EvidenceBundle())
        assert result.passed, (draft.text, result.reasons)
    for order_id, evidence in (
        ("o_1002", ev.o_1002()),
        ("o_1003", ev.o_1003()),
        ("o_1006", ev.o_1006_stalled()),
    ):
        ctx = make_context("到哪了", world, order_id=order_id)
        bundle = EvidenceBundle(evidences=[evidence])

        draft = T.logistics(
            ctx, evidence, LogisticsAnalysis.model_validate(evidence.data["analysis"])
        )
        result = gate.check(ctx, route(Intent.LOGISTICS_TRACK), draft, bundle)
        assert result.passed, (draft.text, result.reasons)


def test_11_5_regenerate_with_reasons(world, ev, route, audit):
    """§11.5 全流程：生成"明天一定能到"→ 拦截 → 带原因重新生成 → 通过。"""
    ctx = make_context("明天能到吗？", world, order_id="o_1002")
    r = route(Intent.NOT_RECEIVED, RouteAction.HIGH_RISK_ANSWER, rule=4, high_risk=True)
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    good = (
        "亲，包裹目前在运输中，截至 10:30 已到达杭州转运中心，"
        "我们会持续关注，有更新第一时间告诉您。"
    )
    llm = FakeLLMClient(
        responses=[
            {"answer": "亲，明天一定能到哦！", "evidence_ids": ["tool:logistics:o_1002"]},
            {"answer": good, "evidence_ids": ["tool:logistics:o_1002"]},
        ]
    )
    generator = DefaultAnswerGenerator(llm, GenerationConfig())

    def judge(request):
        risky = "明天" in request.state["answer"]
        return qa_answers(promise_risk=0.86 if risky else 0.05)

    gate = make_gate(FakeDecisionProvider(judge), audit)
    ctx1 = ctx.model_copy(update={"config": ctx.config.model_copy(update={"suggestion_count": 1})})

    first = generator.generate(ctx1, r, bundle)
    result = gate.check(ctx1, r, first[0], bundle)
    assert not result.passed
    assert "包含依据中没有的时效承诺：明天" in result.reasons

    second = generator.generate(ctx1, r, bundle, retry_feedback=result.reasons)
    retry_prompt = llm.calls[1].messages[1].content
    assert (
        "上次回答未通过质检：包含绝对化用语：一定；包含依据中没有的时效承诺：明天；"
        "包含依据之外的承诺（价格、赠品、时效或赔偿；promise_risk 0.86"
    ) in retry_prompt
    assert retry_prompt.rstrip().count("请修改。") == 1
    final = gate.check(ctx1, r, second[0], bundle)
    assert final.passed and final.model_checked and final.text == good
    assert [d.route_result for d in audit.decisions] == ["block", "pass"]
