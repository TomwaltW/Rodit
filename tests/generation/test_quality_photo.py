"""买家发来质量问题的照片（方案 §7.3.1 "收到的东西坏了：请买家在申请售后时上传照片"、
§15 Phase 3 图片理解、§5.5 不承诺）：第一条推荐是确定性模板，大模型提示词里加图片只供参考的要求。
"""

from __future__ import annotations

import pytest

from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import GenerationConfig, QAGateConfig
from rodit.core.enums import DraftKind, ImageKind, Intent, RouteAction
from rodit.core.evidence import PlatformRuleKey
from rodit.core.lexicon import find_absolute_terms
from rodit.core.models import ConversationContext, EvidenceBundle, ImageFinding
from rodit.generation import DefaultAnswerGenerator
from rodit.generation import templates as T
from rodit.generation.prompt import IMAGE_REQUIREMENT, build_generation_messages
from rodit.knowledge.platform_rules import platform_rule_evidence
from rodit.qa import DefaultQualityGate
from rodit.testing.fakes import FakeClock, FakeLLMClient, make_context

CFG = GenerationConfig()
DESCRIPTION = "【买家发来图片：破损（外包装一角压扁，裙摆有一处约 3 厘米的撕裂）】"
LLM_TEXT = "亲，非常抱歉，麻烦您申请售后时上传照片哦"


def finding(kind: ImageKind = ImageKind.DAMAGE, conf: float = 0.93) -> ImageFinding:
    return ImageFinding(ref="img_1", kind=kind, confidence=conf, summary="外包装一角压扁")


def photo_ctx(world, *findings: ImageFinding, text: str = DESCRIPTION, **kw) -> ConversationContext:
    ctx = make_context(text, world, order_id="o_1013", **kw)
    return ctx.model_copy(update={"image_findings": list(findings)})


def quality_route(route):
    return route(Intent.QUALITY_ISSUE, RouteAction.HIGH_RISK_ANSWER, rule=4, high_risk=True)


def bundle(ev) -> EvidenceBundle:
    rule = platform_rule_evidence("taobao", PlatformRuleKey.APPLY_AFTERSALE)
    return EvidenceBundle(evidences=[ev.policy("quality_issue"), rule])


def llm() -> FakeLLMClient:
    return FakeLLMClient(lambda request: {"answer": LLM_TEXT, "evidence_ids": []})


# ================================================================ 模板


@pytest.mark.parametrize("kind", [ImageKind.DAMAGE, ImageKind.DEFECT, ImageKind.WRONG_ITEM])
def test_problem_photo_kinds(world, kind):
    assert T.has_problem_photo(photo_ctx(world, finding(kind)))


@pytest.mark.parametrize(
    "found",
    [
        [],
        [finding(ImageKind.SIZE_REFERENCE)],
        [finding(ImageKind.LOGISTICS_SCREENSHOT)],
        [finding(ImageKind.PRODUCT_PHOTO)],
        [finding(ImageKind.OTHER)],
        [finding(ImageKind.DAMAGE, conf=0.4)],  # 看不太清：不当作质量问题的照片
    ],
)
def test_not_problem_photos(world, found):
    assert not T.has_problem_photo(photo_ctx(world, *found))


def test_quality_photo_template_wording(world, ev):
    rule = platform_rule_evidence("taobao", PlatformRuleKey.APPLY_AFTERSALE)
    draft = T.quality_photo(photo_ctx(world, finding()), rule)
    assert draft.kind == DraftKind.TEMPLATE and draft.template_id == T.QUALITY_PHOTO
    assert draft.text == (
        "亲，非常抱歉给您带来不好的体验，您发来的照片我们看到了。"
        "麻烦您在订单页申请售后，申请时把照片一起上传，我们核实后会跟您说明处理方式哦。"
    )
    assert draft.evidence_ids == [rule.id]
    # 不复述图片理解的结论，不承诺处理结果、不用绝对化用语（§5.5）
    for word in ("破损", "退款", "补发", "赔", "换新", "保证", "一定", "马上"):
        assert word not in draft.text
    assert find_absolute_terms(draft.text) == []


def test_quality_photo_template_without_rule_and_professional_style(world):
    ctx = photo_ctx(world, finding(), config_update={"salutation": "您", "tone": "专业"})
    draft = T.quality_photo(ctx)
    assert draft.text.startswith("您好，") and draft.text.endswith("处理方式。")
    assert draft.evidence_ids == []


@pytest.mark.parametrize("platform_limit", [100, 120])
def test_quality_photo_template_passes_hard_rules(world, ev, route, platform_limit):
    """模板只跑硬规则（§5.5）：没有数字、承诺、站外引流，不超长（拼多多默认上限 100 字）。"""
    gate = DefaultQualityGate(None, QAGateConfig(), InMemoryAuditSink(), FakeClock(world.now))
    ctx = photo_ctx(world, finding(), config_update={"max_answer_chars": platform_limit})
    b = bundle(ev)
    draft = T.quality_photo(ctx, b.evidences[1])
    result = gate.check(ctx, quality_route(route), draft, b)
    assert result.passed, result.reasons
    assert result.text == draft.text and len(draft.text) <= platform_limit


# ================================================================ 生成器


def test_generator_puts_photo_template_first(world, ev, route):
    model = llm()
    ctx = photo_ctx(world, finding())
    drafts = DefaultAnswerGenerator(model, CFG).generate(ctx, quality_route(route), bundle(ev))
    assert [d.kind for d in drafts] == [DraftKind.TEMPLATE, DraftKind.LLM]
    assert drafts[0].template_id == T.QUALITY_PHOTO
    assert drafts[1].text == LLM_TEXT
    # 大模型提示词里有"图片只供参考"的要求和图片描述
    [request] = model.calls
    prompt = "\n".join(m.content for m in request.messages)
    assert IMAGE_REQUIREMENT in prompt and DESCRIPTION in prompt


def test_generator_without_photo_has_no_template(world, ev, route):
    ctx = photo_ctx(world, text="收到的裙子是坏的")
    drafts = DefaultAnswerGenerator(llm(), CFG).generate(ctx, quality_route(route), bundle(ev))
    assert all(d.template_id != T.QUALITY_PHOTO for d in drafts)


@pytest.mark.parametrize("intent", [Intent.RETURN_POLICY, Intent.NOT_RECEIVED, Intent.SIZE_ADVICE])
def test_photo_template_only_for_quality_issue(world, ev, route, intent):
    ctx = photo_ctx(world, finding())
    drafts = DefaultAnswerGenerator(llm(), CFG).generate(ctx, route(intent), bundle(ev))
    assert all(d.template_id != T.QUALITY_PHOTO for d in drafts)


def test_photo_template_survives_llm_failure(world, ev, route):
    from rodit.core.errors import LLMTimeout

    failing = FakeLLMClient(lambda request: LLMTimeout("超时"))
    ctx = photo_ctx(world, finding())
    drafts = DefaultAnswerGenerator(failing, CFG).generate(ctx, quality_route(route), bundle(ev))
    assert drafts and drafts[0].template_id == T.QUALITY_PHOTO


def test_single_slot_keeps_only_the_template(world, ev, route):
    model = llm()
    ctx = photo_ctx(world, finding(), config_update={"suggestion_count": 1})
    drafts = DefaultAnswerGenerator(model, CFG).generate(ctx, quality_route(route), bundle(ev))
    assert [d.template_id for d in drafts] == [T.QUALITY_PHOTO]
    assert model.calls == []  # 名额已满，不调大模型（§5.4 能不用就不用）


# ================================================================ 提示词


def test_image_requirement_only_with_findings(world, ev, route):
    plain = build_generation_messages(
        photo_ctx(world, text="收到的裙子是坏的"), quality_route(route), bundle(ev), config=CFG
    )
    assert IMAGE_REQUIREMENT not in plain[1].content
    size = build_generation_messages(
        photo_ctx(world, finding(ImageKind.SIZE_REFERENCE)),
        route(Intent.SIZE_ADVICE),
        bundle(ev),
        config=CFG,
    )
    assert IMAGE_REQUIREMENT in size[1].content
    # 要求在输出格式之前，和其他要求放在一起
    body = size[1].content
    assert body.index(IMAGE_REQUIREMENT) < body.index("请只输出 JSON")
