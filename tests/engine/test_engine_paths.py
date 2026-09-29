"""问答引擎各条路径（方案 §5.4–§5.7、§6.6、§11、§13.4）：每种路由结果、去重、刷新推荐、
重新生成、FAQ / 模板拦截、澄清、缺失知识、分条推荐、影子模式、店铺停用。"""

from __future__ import annotations

from engine_support import (
    CHECK_FOR_YOU,
    CLARIFY_INTENT,
    CLARIFY_ITEM,
    OFF_PLATFORM_SCRIPT,
    SOOTHE_SCRIPT,
    Kit,
    bundle_of,
    make_route,
    policy_evidence,
    qa_fail,
    qa_pass,
)

from rodit.core import evidence as ev
from rodit.core.audit import AuditAction
from rodit.core.config import EngineConfig
from rodit.core.enums import (
    Confidence,
    DraftKind,
    Intent,
    Mode,
    Platform,
    RouteAction,
    SuggestionStatus,
)
from rodit.core.models import Draft, EvidenceBundle, EvidenceRefView
from rodit.engine import AssistEngine
from rodit.engine.engine import (
    REASON_DISABLED,
    REASON_HUMAN_PRIORITY,
    REASON_NO_EVIDENCE,
    REASON_NO_SUGGESTION,
    REASON_NO_SUGGESTION_DEGRADED,
    REASON_QA_FAILED,
    REASON_SHOP_INACTIVE,
    REASON_TRANSFER,
)
from rodit.testing.fixtures import BUYER_DOUYIN, CONV, SHOP_A, SHOP_B, SHOP_D

SHIP = "亲，发货与快递：现货付款后 48 小时内发货"


def llm(text: str, *ids: str) -> Draft:
    return Draft(text=text, kind=DraftKind.LLM, evidence_ids=list(ids) or [ev.policy("ship_time")])


def faq_draft(text: str, faq_id: int = 3) -> Draft:
    return Draft(text=text, kind=DraftKind.FAQ, faq_id=faq_id, evidence_ids=[ev.faq(faq_id)])


# ================================================================ 规则 6：检索、生成、质检


def test_answer_path_produces_card_and_record(kit: Kit) -> None:
    engine = kit.engine()
    msg = kit.msg("什么时候发货")
    result = engine.handle(msg)
    assert result is not None
    assert result.mode == Mode.ASSIST
    assert result.msg_ids == [msg.msg_id]
    assert result.route is not None and result.route.intent == Intent.SHIPPING_POLICY
    assert result.no_suggestion_reason is None
    assert result.degraded == []
    assert result.transfer_to_human is False and result.auto_reply is None
    [card] = result.suggestions
    assert card.text == SHIP
    assert card.intent == Intent.SHIPPING_POLICY
    assert card.kind == DraftKind.LLM
    assert card.evidence == [EvidenceRefView(id="policy:ship_time", label="店铺政策 · 发货时效")]
    assert card.confidence == Confidence.HIGH
    assert card.risk_hints == []
    assert card.qa_scores["grounded"] == 0.95

    record = kit.suggestions.get(SHOP_A, card.suggestion_id or -1)
    assert record is not None
    assert record.status == SuggestionStatus.SHOWN
    assert record.msg_id == msg.msg_id
    assert record.question_text == "什么时候发货"
    assert record.text == SHIP
    assert record.evidence_ids == ["policy:ship_time"]
    assert record.kind == DraftKind.LLM
    assert record.agent_id == msg.agent_id
    assert record.created_at == kit.clock.now()
    assert isinstance(record.latency_ms, int)
    assert kit.audit.actions(SHOP_A) == [
        AuditAction.UNDERSTAND,
        AuditAction.ROUTE,
        AuditAction.RETRIEVE,
        AuditAction.GENERATE,
        AuditAction.QA,
        AuditAction.SUGGEST,
        AuditAction.RESULT,
    ]


def test_card_records_current_item(kit: Kit) -> None:
    result = kit.engine().handle(kit.msg("这个是纯棉的吗", item_id="i_555"))
    assert result is not None
    record = kit.suggestions.get(SHOP_A, result.suggestions[0].suggestion_id or -1)
    assert record is not None and record.item_id == "i_555"


def test_suggestion_count_caps_cards_and_stops_checking(kit: Kit) -> None:
    kit.generator.script = [
        [faq_draft("亲，现货付款后 48 小时内发货哦~"), llm("回答二"), llm("回答三")]
    ]
    result = kit.engine().handle(kit.msg("什么时候发货"))
    assert result is not None
    assert [c.text for c in result.suggestions] == ["亲，现货付款后 48 小时内发货哦~", "回答二"]
    assert [c.kind for c in result.suggestions] == [DraftKind.FAQ, DraftKind.LLM]
    assert len(kit.gate.calls) == 2  # 够了就不再质检第三条


def test_suggestion_count_follows_shop_config(kit: Kit) -> None:
    kit.set_shop_config(SHOP_A, suggestion_count=1)
    kit.generator.script = [[llm("回答一"), llm("回答二")]]
    result = kit.engine().handle(kit.msg("什么时候发货"))
    assert result is not None and [c.text for c in result.suggestions] == ["回答一"]


def test_card_text_is_cleaned_qa_text(kit: Kit) -> None:
    kit.gate.rule = lambda d: qa_pass(d).model_copy(update={"text": "清洗后的文本"})
    result = kit.engine().handle(kit.msg("什么时候发货"))
    assert result is not None and result.suggestions[0].text == "清洗后的文本"
    record = kit.suggestions.get(SHOP_A, result.suggestions[0].suggestion_id or -1)
    assert record is not None and record.text == "清洗后的文本"


def test_confidence_rules(kit: Kit) -> None:
    # 大模型 grounded < 0.9 → 中
    kit.gate.rule = lambda d: qa_pass(d, grounded=0.85)
    result = kit.engine().handle(kit.msg("什么时候发货"))
    assert result is not None and result.suggestions[0].confidence == Confidence.MEDIUM
    # 问题类型置信度 < 0.85 → 中（即使是模板）
    kit.gate.rule = None
    kit.router.decide = make_route(Intent.SHIPPING_POLICY, confidence=0.8)
    kit.generator.script = [[faq_draft("亲，48 小时内发货哦~")]]
    result = kit.engine().handle(kit.msg("什么时候发货呀"))
    assert result is not None and result.suggestions[0].confidence == Confidence.MEDIUM
    # 模板 + 置信度 0.85 → 高
    kit.router.decide = make_route(Intent.SHIPPING_POLICY, confidence=0.85)
    kit.generator.script = [[faq_draft("亲，48 小时内发货哦~")]]
    result = kit.engine().handle(kit.msg("啥时候发"))
    assert result is not None and result.suggestions[0].confidence == Confidence.HIGH


def test_unknown_evidence_label_falls_back_to_id(kit: Kit) -> None:
    kit.generator.script = [[llm("回答", "policy:ship_time", "policy:ghost")]]
    result = kit.engine().handle(kit.msg("什么时候发货"))
    assert result is not None
    assert result.suggestions[0].evidence == [
        EvidenceRefView(id="policy:ship_time", label="店铺政策 · 发货时效"),
        EvidenceRefView(id="policy:ghost", label="policy:ghost"),
    ]


def test_work_runs_inside_shop_scope(kit: Kit) -> None:
    kit.engine().handle(kit.msg("什么时候发货"))
    assert kit.router.shop_ids == [SHOP_A]


# ================================================================ 规则 4：推荐 + 风险提示


def test_high_risk_answer_carries_risk_hints_and_bundle_alerts(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.PRICE_PROMO,
        RouteAction.HIGH_RISK_ANSWER,
        rule=4,
        high_risk=True,
        risk_hints=["涉及价格承诺，请确认"],
    )
    kit.retriever.bundles[Intent.PRICE_PROMO] = bundle_of(
        policy_evidence("promotions", "店铺满 199 减 20"), alerts=["活动今晚结束，请确认"]
    )
    result = kit.engine().handle(kit.msg("能便宜点吗"))
    assert result is not None
    [card] = result.suggestions
    assert card.risk_hints == ["涉及价格承诺，请确认", "活动今晚结束，请确认"]
    assert result.alerts == []  # 证据包的提醒进卡片的风险提示，不重复标红


# ================================================================ 规则 1、3、7：不检索 / 模板


def test_off_platform_script_retrieves_only_its_platform_rule(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.OFF_PLATFORM, RouteAction.OFF_PLATFORM_SCRIPT, rule=1, confidence=0.97
    )
    kit.retriever.bundles[Intent.OFF_PLATFORM] = EvidenceBundle()
    result = kit.engine().handle(kit.msg("加个微信吧，转账给你便宜点"))
    assert result is not None
    # 检索只为给卡片找依据（平台站外交易规则，§5.1、§5.6），证据包为空也照给标准话术
    assert [route.action for _, route in kit.retriever.calls] == [RouteAction.OFF_PLATFORM_SCRIPT]
    assert kit.generator.calls[0]["evidence_ids"] == []
    [card] = result.suggestions
    assert card.text == OFF_PLATFORM_SCRIPT
    assert card.kind == DraftKind.TEMPLATE
    assert card.confidence == Confidence.HIGH
    assert len(kit.gate.calls) == 1  # 模板也要过质检（硬规则）


def test_clarify_route_skips_retrieval(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.SHIPPING_POLICY, RouteAction.CLARIFY, rule=7, confidence=0.5
    )
    result = kit.engine().handle(kit.msg("那个呢"))
    assert result is not None
    assert kit.retriever.calls == []
    assert [c.text for c in result.suggestions] == [CLARIFY_INTENT]
    assert result.suggestions[0].confidence == Confidence.MEDIUM


def test_soothe_without_evidence_still_suggests(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.QUALITY_ISSUE, RouteAction.SOOTHE, rule=3, alerts=["买家情绪激动，建议先安抚"]
    )
    kit.retriever.bundles[Intent.QUALITY_ISSUE] = EvidenceBundle()
    result = kit.engine().handle(kit.msg("气死我了！收到的是坏的！"))
    assert result is not None
    assert [c.text for c in result.suggestions] == [SOOTHE_SCRIPT]
    assert result.alerts == ["买家情绪激动，建议先安抚"]
    assert len(kit.retriever.calls) == 1


# ================================================================ 规则 2、8、转人工：不推荐


def test_human_priority_gives_alerts_only(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.COMPLAINT, RouteAction.HUMAN_PRIORITY, rule=2, alerts=["买家要求人工，请优先处理"]
    )
    result = kit.engine().handle(kit.msg("我要投诉，叫你们人工来"))
    assert result is not None
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_HUMAN_PRIORITY
    assert result.alerts == ["买家要求人工，请优先处理"]
    assert result.transfer_to_human is False
    assert kit.retriever.calls == [] and kit.generator.calls == [] and kit.gate.calls == []
    assert kit.suggestions.list(SHOP_A) == []


def test_human_priority_without_router_alert_gets_default_alert(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.COMPLAINT, RouteAction.HUMAN_PRIORITY, rule=2)
    result = kit.engine().handle(kit.msg("我要投诉"))
    assert result is not None and result.alerts == ["买家要求人工或在投诉，请优先处理"]


def test_no_suggestion_route(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.OTHER, RouteAction.NO_SUGGESTION, rule=8, confidence=0.3)
    result = kit.engine().handle(kit.msg("嗯嗯"))
    assert result is not None
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_NO_SUGGESTION
    assert kit.generator.calls == []


def test_no_suggestion_route_when_decision_degraded(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.PRICE_PROMO, RouteAction.NO_SUGGESTION, rule=8, degraded=True, provider="rules"
    )
    result = kit.engine().handle(kit.msg("能便宜点吗"))
    assert result is not None
    assert result.no_suggestion_reason == REASON_NO_SUGGESTION_DEGRADED
    assert result.degraded == ["decision"]


def test_transfer_human_route(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.COMPLAINT, RouteAction.TRANSFER_HUMAN, rule=2)
    result = kit.engine().handle(kit.msg("转人工"))
    assert result is not None
    assert result.transfer_to_human is True
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_TRANSFER


# ================================================================ 去重、刷新推荐（§5.3、§8.3）


def test_duplicate_message_returns_none(kit: Kit) -> None:
    engine = kit.engine()
    msg = kit.msg("什么时候发货")
    assert engine.handle(msg) is not None
    assert engine.handle(msg) is None
    assert len(kit.router.calls) == 1
    assert len(kit.result_events()) == 1
    assert len(kit.suggestions.list(SHOP_A)) == 1


def test_non_buyer_message_is_ignored(kit: Kit) -> None:
    from rodit.core.enums import Role

    msg = kit.msg("在的亲").model_copy(update={"role": Role.AGENT})
    assert kit.engine().handle(msg) is None
    assert kit.audit.events == []
    assert kit.conversations.recent_messages(SHOP_A, CONV) == []


def test_new_message_expires_previous_shown_suggestions(kit: Kit) -> None:
    engine = kit.engine()
    kit.generator.script = [[llm("回答一"), llm("回答二")]]
    first = engine.handle(kit.msg("什么时候发货"))
    other = engine.handle(kit.msg("在吗", conversation_id="conv_other"))
    assert first is not None and other is not None
    second = engine.handle(kit.msg("发什么快递"))
    assert second is not None
    statuses = {r.id: r.status for r in kit.suggestions.list(SHOP_A, conversation_id=CONV)}
    for card in first.suggestions:
        assert statuses[card.suggestion_id] == SuggestionStatus.EXPIRED
    assert statuses[second.suggestions[0].suggestion_id] == SuggestionStatus.SHOWN
    [other_record] = kit.suggestions.list(SHOP_A, conversation_id="conv_other")
    assert other_record.status == SuggestionStatus.SHOWN  # 别的会话不动


def test_merged_follow_up_refreshes_and_answers_merged_turn(kit: Kit) -> None:
    engine = kit.engine()
    m1 = kit.msg("165")
    first = engine.handle(m1)
    m2 = kit.msg("110斤穿什么码", after_s=1.0)
    second = engine.handle(m2)
    assert first is not None and second is not None
    assert second.msg_ids == [m1.msg_id, m2.msg_id]
    assert kit.router.calls[-1].current.text == "165 110斤穿什么码"
    record = kit.suggestions.get(SHOP_A, first.suggestions[0].suggestion_id or -1)
    assert record is not None and record.status == SuggestionStatus.EXPIRED


# ================================================================ 质检不通过则重写（§5.5、§11.5）


def test_regenerates_once_with_reasons_then_succeeds(kit: Kit) -> None:
    kit.generator.script = [
        [llm("亲，明天一定能到哦！")],
        [llm("亲，包裹目前在运输中，截至 10:30 已到达杭州转运中心")],
    ]
    result = kit.engine().handle(kit.msg("明天能到吗"))
    assert result is not None
    assert [c.text for c in result.suggestions] == [
        "亲，包裹目前在运输中，截至 10:30 已到达杭州转运中心"
    ]
    assert [c["retry_feedback"] for c in kit.generator.calls] == [None, ["绝对化用语：一定"]]
    [result_event] = kit.result_events()
    assert result_event.detail["qa_failed"] == 1
    assert result_event.detail["hard_hits"] == ["absolute_term"]


def test_regeneration_gives_up_after_one_retry_with_reasons(kit: Kit) -> None:
    kit.gate.rule = lambda d: qa_fail(
        d, ["承诺了依据之外的到货时间"], rule="time_promise", scores={"promise_risk": 0.86}
    )
    kit.generator.script = [[llm("亲，明天能到")], [llm("亲，明天下午能到")], [llm("不该调用")]]
    result = kit.engine().handle(kit.msg("明天能到吗"))
    assert result is not None
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_QA_FAILED + "承诺了依据之外的到货时间"
    assert len(kit.generator.calls) == 2
    assert kit.generator.calls[1]["retry_feedback"] == ["承诺了依据之外的到货时间"]
    assert kit.result_events()[0].detail["qa_failed"] == 2
    assert kit.suggestions.list(SHOP_A) == []


def test_regenerated_identical_text_is_not_rechecked(kit: Kit) -> None:
    """生成器忽略了重写原因、给出同样的文字：已经判过不通过，不再质检、不推荐。"""
    kit.generator.script = [[llm("亲，明天一定能到哦！")], [llm("亲，明天一定能到哦！")]]
    result = kit.engine().handle(kit.msg("明天能到吗"))
    assert result is not None and result.suggestions == []
    assert len(kit.generator.calls) == 2
    assert len(kit.gate.calls) == 1
    assert kit.result_events()[0].detail["qa_failed"] == 1


def test_max_regenerations_zero_means_no_retry(kit: Kit) -> None:
    kit.config = EngineConfig.model_validate({"qa_gate": {"max_regenerations": 0}})
    kit.generator.script = [[llm("亲，明天一定能到哦！")]]
    result = kit.engine().handle(kit.msg("明天能到吗"))
    assert result is not None and result.suggestions == []
    assert len(kit.generator.calls) == 1


def test_failed_draft_without_reasons_uses_rule_labels(kit: Kit) -> None:
    kit.gate.rule = lambda d: qa_fail(d, [], rule="off_platform")
    result = kit.engine().handle(kit.msg("什么时候发货"))
    assert result is not None
    assert result.no_suggestion_reason == REASON_QA_FAILED + "站外引流"
    assert kit.generator.calls[1]["retry_feedback"] == ["站外引流"]


def test_regeneration_does_not_recheck_same_faq(kit: Kit) -> None:
    faq = faq_draft("亲，现货付款后 48 小时内发货哦~")
    kit.generator.script = [[faq, llm("一定发")], [faq, llm("48 小时内发")]]
    result = kit.engine().handle(kit.msg("什么时候发货"))
    assert result is not None
    assert [c.text for c in result.suggestions] == [faq.text, "48 小时内发"]
    assert [d.text for d in kit.gate.calls] == [faq.text, "一定发", "48 小时内发"]


def test_no_regeneration_when_enough_passed(kit: Kit) -> None:
    kit.set_shop_config(SHOP_A, suggestion_count=1)
    kit.generator.script = [[llm("48 小时内发"), llm("一定发")]]
    result = kit.engine().handle(kit.msg("什么时候发货"))
    assert result is not None and [c.text for c in result.suggestions] == ["48 小时内发"]
    assert len(kit.generator.calls) == 1


# ================================================================ FAQ / 模板没过硬规则：丢弃并提醒


def test_faq_failing_hard_rule_is_dropped_with_alert(kit: Kit) -> None:
    bad_faq = faq_draft("亲，全网最低价哦", faq_id=7)
    kit.generator.script = [[bad_faq, llm("亲，店铺满 199 减 20")]]
    kit.gate.rule = lambda d: (
        qa_fail(d, ["商家禁用词：全网最低"], rule="banned_word")
        if d.kind == DraftKind.FAQ
        else None
    )
    result = kit.engine().handle(kit.msg("能便宜点吗"))
    assert result is not None
    assert [c.text for c in result.suggestions] == ["亲，店铺满 199 减 20"]
    assert result.alerts == ["FAQ #7未通过质检（商家禁用词），本次不推荐，请检查并修改知识库"]
    assert len(kit.generator.calls) == 1  # FAQ 不重写


def test_template_only_failure_gives_no_suggestion_with_reason(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.OFF_PLATFORM, RouteAction.OFF_PLATFORM_SCRIPT, rule=1, confidence=0.97
    )
    kit.gate.rule = lambda d: qa_fail(d, ["站外引流：微信"], rule="off_platform")
    result = kit.engine().handle(kit.msg("加个微信"))
    assert result is not None
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_QA_FAILED + "站外引流：微信"
    assert result.alerts == [
        "模板 off_platform未通过质检（站外引流），本次不推荐，请检查并修改知识库"
    ]
    assert len(kit.generator.calls) == 1


# ================================================================ 澄清（§5.2.4）


def test_clarify_when_item_is_ambiguous(kit: Kit) -> None:
    engine = kit.engine()
    engine.handle(kit.msg("看看这个", item_id="i_555"))
    engine.handle(kit.msg("还有这个", item_id="i_777"))
    kit.router.decide = make_route(Intent.PRODUCT_SPEC)
    result = engine.handle(kit.msg("这个是纯棉的吗"))
    assert result is not None
    ctx = kit.router.calls[-1]
    assert ctx.item is None and ctx.candidate_item_ids == ["i_777", "i_555"]
    [card] = result.suggestions
    assert card.text == CLARIFY_ITEM
    assert card.kind == DraftKind.TEMPLATE
    assert result.missing_knowledge == [ev.MISSING_ITEM]
    assert kit.gaps.list_gaps(SHOP_A) == []  # 商品不确定不是知识缺失


# ================================================================ 依据不足 → 缺失知识（§5.7、§5.8）


def test_missing_knowledge_gives_no_suggestion_and_records_gap(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.INVOICE_GIFT)
    kit.retriever.bundles[Intent.INVOICE_GIFT] = EvidenceBundle(missing=["policy:invoice"])
    result = kit.engine().handle(kit.msg("能开发票吗？我电话 13812345678"))
    assert result is not None
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_NO_EVIDENCE
    assert result.missing_knowledge == ["policy:invoice"]
    assert kit.generator.calls == []
    [gap] = kit.gaps.list_gaps(SHOP_A)
    assert gap.missing == "policy:invoice"
    assert gap.intent == Intent.INVOICE_GIFT
    assert gap.source == "engine"
    assert gap.question_text == "能开发票吗？我电话 [手机号]"  # 只遮个人信息，原文标点保留
    assert gap.created_at == kit.clock.now()
    assert kit.result_events()[0].detail["missing"] == ["policy:invoice"]


def test_missing_knowledge_without_details_uses_intent_marker(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.OTHER)
    kit.retriever.bundles[Intent.OTHER] = EvidenceBundle()
    result = kit.engine(with_gaps=False).handle(kit.msg("你们老板是谁"))
    assert result is not None
    assert result.missing_knowledge == ["knowledge:other"]
    assert kit.gaps.list_gaps(SHOP_A) == []  # 没注入 gaps：不记


def test_missing_order_is_not_a_knowledge_gap(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.LOGISTICS_TRACK)
    kit.retriever.bundles[Intent.LOGISTICS_TRACK] = EvidenceBundle(missing=[ev.MISSING_ORDER])
    result = kit.engine().handle(kit.msg("我的快递到哪了"))
    assert result is not None
    assert result.no_suggestion_reason == REASON_NO_EVIDENCE
    assert result.missing_knowledge == [ev.MISSING_ORDER]
    assert kit.gaps.list_gaps(SHOP_A) == []


def test_partial_missing_still_suggests_and_records_gap(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.SIZE_ADVICE)
    kit.retriever.bundles[Intent.SIZE_ADVICE] = bundle_of(
        policy_evidence("size_tendency", "版型整体偏小，建议拍大一码"),
        missing=["item:i_777:size_advice"],
    )
    result = kit.engine().handle(kit.msg("165 110斤穿什么码", item_id="i_777"))
    assert result is not None
    assert len(result.suggestions) == 1
    assert result.missing_knowledge == ["item:i_777:size_advice"]
    assert [g.missing for g in kit.gaps.list_gaps(SHOP_A)] == ["item:i_777:size_advice"]


def test_unavailable_tool_still_generates_check_for_you(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.LOGISTICS_TRACK)
    kit.retriever.bundles[Intent.LOGISTICS_TRACK] = EvidenceBundle(unavailable=["tool:logistics"])
    result = kit.engine().handle(kit.msg("我的快递到哪了", order_id="o_1002"))
    assert result is not None
    [card] = result.suggestions
    assert card.text == CHECK_FOR_YOU
    assert card.risk_hints == ["实时数据暂时查不到（物流轨迹），请核实后再发送"]
    assert result.degraded == ["tool:logistics"]


def test_template_intent_without_evidence_still_generates(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.GREETING)
    kit.retriever.bundles[Intent.GREETING] = EvidenceBundle(missing=["policy:service_hours"])
    kit.generator.script = [
        [Draft(text="亲，在的~", kind=DraftKind.TEMPLATE, template_id="greeting")]
    ]
    result = kit.engine().handle(kit.msg("在吗"))
    assert result is not None
    assert [c.text for c in result.suggestions] == ["亲，在的~"]
    assert result.missing_knowledge == ["policy:service_hours"]


def test_bundle_alerts_surface_when_no_card(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.LOGISTICS_TRACK)
    kit.retriever.bundles[Intent.LOGISTICS_TRACK] = bundle_of(
        policy_evidence(), alerts=["物流超过 48 小时没有更新，建议向快递催查"]
    )
    kit.gate.rule = lambda d: qa_fail(d, ["依据不足"], rule=None)
    result = kit.engine().handle(kit.msg("怎么还没到"))
    assert result is not None
    assert result.suggestions == []
    assert "物流超过 48 小时没有更新，建议向快递催查" in result.alerts


# ================================================================ 规则 5：按问题分条（§6.6）


def test_split_multi_one_card_per_part(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.SHIPPING_POLICY,
        RouteAction.SPLIT_MULTI,
        rule=5,
        confidence=0.55,
        parts=[(Intent.SHIPPING_POLICY, 0.55), (Intent.PRICE_PROMO, 0.9)],
        multi_intent=0.85,
    )
    kit.generator.script = [[llm("发货一"), llm("发货二")], [llm("优惠一")]]
    result = kit.engine().handle(kit.msg("什么时候发货？能便宜点吗？"))
    assert result is not None
    assert [(c.intent, c.text) for c in result.suggestions] == [
        (Intent.SHIPPING_POLICY, "发货一"),
        (Intent.PRICE_PROMO, "优惠一"),
    ]
    part_routes = [route for _, route in kit.retriever.calls]
    assert [(r.intent, r.action, r.parts) for r in part_routes] == [
        (Intent.SHIPPING_POLICY, RouteAction.ANSWER, []),
        (Intent.PRICE_PROMO, RouteAction.HIGH_RISK_ANSWER, []),
    ]
    assert result.route is not None and result.route.action == RouteAction.SPLIT_MULTI
    # 分条的置信度按该条的概率算
    assert [c.confidence for c in result.suggestions] == [Confidence.MEDIUM, Confidence.HIGH]
    records = kit.suggestions.list(SHOP_A)
    assert [r.intent for r in records] == [Intent.SHIPPING_POLICY, Intent.PRICE_PROMO]


def test_split_multi_partial_failure_is_flagged(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.SHIPPING_POLICY,
        RouteAction.SPLIT_MULTI,
        rule=5,
        parts=[(Intent.SHIPPING_POLICY, 0.6), (Intent.INVOICE_GIFT, 0.3)],
    )
    kit.retriever.bundles[Intent.INVOICE_GIFT] = EvidenceBundle(missing=["policy:invoice"])
    result = kit.engine().handle(kit.msg("什么时候发货？能开发票吗？"))
    assert result is not None
    assert [c.intent for c in result.suggestions] == [Intent.SHIPPING_POLICY]
    assert result.no_suggestion_reason is None
    assert result.missing_knowledge == ["policy:invoice"]
    assert result.alerts == [f"「发票 / 赠品 / 保障」没有推荐：{REASON_NO_EVIDENCE}"]


def test_split_multi_complaint_part_is_not_answered(kit: Kit) -> None:
    kit.router.decide = make_route(
        Intent.LOGISTICS_TRACK,
        RouteAction.SPLIT_MULTI,
        rule=5,
        parts=[(Intent.LOGISTICS_TRACK, 0.6), (Intent.COMPLAINT, 0.3)],
    )
    result = kit.engine().handle(kit.msg("快递到哪了？再不到我就投诉"))
    assert result is not None
    assert [c.intent for c in result.suggestions] == [Intent.LOGISTICS_TRACK]
    assert "买家消息里有投诉，请人工优先处理" in result.alerts
    assert [route.intent for _, route in kit.retriever.calls] == [Intent.LOGISTICS_TRACK]


def test_split_multi_without_parts_falls_back_to_route_intent(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.PRICE_PROMO, RouteAction.SPLIT_MULTI, rule=5)
    result = kit.engine().handle(kit.msg("能便宜点吗？有券吗？"))
    assert result is not None
    assert [c.intent for c in result.suggestions] == [Intent.PRICE_PROMO]
    assert kit.retriever.calls[0][1].action == RouteAction.HIGH_RISK_ANSWER


# ================================================================ 店铺关闭的问题类型


def test_disabled_intent_gives_no_suggestion(kit: Kit) -> None:
    kit.set_shop_config(SHOP_A, disabled_intents=["invoice_gift"])
    kit.router.decide = make_route(Intent.INVOICE_GIFT)
    result = kit.engine().handle(kit.msg("能开发票吗"))
    assert result is not None
    assert result.no_suggestion_reason == REASON_DISABLED
    assert kit.retriever.calls == []


def test_disabled_part_is_skipped(kit: Kit) -> None:
    kit.set_shop_config(SHOP_A, disabled_intents=["invoice_gift"])
    kit.router.decide = make_route(
        Intent.SHIPPING_POLICY,
        RouteAction.SPLIT_MULTI,
        rule=5,
        parts=[(Intent.SHIPPING_POLICY, 0.6), (Intent.INVOICE_GIFT, 0.3)],
    )
    result = kit.engine().handle(kit.msg("什么时候发货？能开发票吗？"))
    assert result is not None
    assert [c.intent for c in result.suggestions] == [Intent.SHIPPING_POLICY]


# ================================================================ 影子模式（§13.4）


def test_shadow_mode_generates_but_marks_shadow(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.SHIPPING_POLICY, confidence=0.97, auto_eligible=True)
    engine = kit.engine()
    msg = kit.msg("什么时候发货", shop_id=SHOP_D, buyer_ref=BUYER_DOUYIN, mode=Mode.AUTO)
    result = engine.handle(msg)
    assert result is not None
    assert result.mode == Mode.SHADOW
    assert len(result.suggestions) == 1
    assert result.auto_reply is None
    assert result.transfer_to_human is False
    [record] = kit.suggestions.list(SHOP_D)
    assert record.status == SuggestionStatus.SHADOW
    assert kit.events(AuditAction.SUGGEST, SHOP_D)[0].detail["status"] == "shadow"
    # 下一条消息不会把影子记录改成过期
    engine.handle(kit.msg("发什么快递", shop_id=SHOP_D, buyer_ref=BUYER_DOUYIN))
    assert kit.suggestions.get(SHOP_D, record.id or -1).status == SuggestionStatus.SHADOW  # type: ignore[union-attr]


def test_shadow_mode_never_transfers(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.COMPLAINT, RouteAction.TRANSFER_HUMAN, rule=2)
    result = kit.engine().handle(kit.msg("转人工", shop_id=SHOP_D, buyer_ref=BUYER_DOUYIN))
    assert result is not None
    assert result.mode == Mode.SHADOW
    assert result.transfer_to_human is False


def test_shadow_mode_failure_result_is_still_shadow(kit: Kit) -> None:
    kit.router.error = RuntimeError("boom")
    result = kit.engine().handle(kit.msg("在吗", shop_id=SHOP_D, buyer_ref=BUYER_DOUYIN))
    assert result is not None and result.mode == Mode.SHADOW


# ================================================================ 店铺停用、店铺隔离（§8.4、§9.2）


def test_inactive_shop_is_not_processed(kit: Kit) -> None:
    from rodit.core.enums import ShopStatus

    shop = kit.knowledge.get_shop(SHOP_B)
    assert shop is not None
    kit.knowledge.upsert_shop(shop.model_copy(update={"status": ShopStatus.SUSPENDED}))
    engine = kit.engine()
    msg = kit.msg("在吗", shop_id=SHOP_B)
    result = engine.handle(msg)
    assert result is not None
    assert result.no_suggestion_reason == REASON_SHOP_INACTIVE
    assert result.suggestions == []
    assert kit.conversations.recent_messages(SHOP_B, CONV) == []  # 不保存消息
    assert kit.router.calls == []
    assert len(kit.result_events(SHOP_B)) == 1
    unknown = engine.handle(kit.msg("在吗", shop_id="s_unknown", platform=Platform.TAOBAO))
    assert unknown is not None and unknown.no_suggestion_reason == REASON_SHOP_INACTIVE


def test_same_conversation_id_in_two_shops_does_not_cross(kit: Kit) -> None:
    engine = kit.engine()
    a = engine.handle(kit.msg("什么时候发货"))
    b = engine.handle(kit.msg("什么时候发货", shop_id=SHOP_B))
    assert a is not None and b is not None
    [record_a] = kit.suggestions.list(SHOP_A)
    assert record_a.status == SuggestionStatus.SHOWN  # B 店的新消息不刷新 A 店的推荐
    assert kit.router.calls[-1].shop_id == SHOP_B
    assert kit.router.calls[-1].history == []
    assert len(kit.result_events(SHOP_A)) == 1 and len(kit.result_events(SHOP_B)) == 1


def test_engine_accepts_protocol_implementations(kit: Kit) -> None:
    engine = kit.engine()
    assert isinstance(engine, AssistEngine)
    assert engine.builder is not None
