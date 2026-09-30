"""方案 §11 的五个核心流程，端到端跑在演示单元上。

AssistService → 适配器 → 引擎 → 反馈 / 售后审核。路由、检索、生成、质检、引擎、反馈、
售后审核都是真实组件；只有大模型（生成）和判断层是脚本化的，见 e2e_support。全部合成数据。
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from e2e_support import (
    RETRY_MARK,
    Chat,
    demo_service,
    enable_auto,
    generate_calls,
    llm_answer,
    prompt_text,
    route,
    scripted_llm,
    scripted_qa,
    scripted_router,
)

from rodit.core.audit import AuditAction
from rodit.core.enums import (
    AfterSaleAction,
    AfterSaleStatus,
    DecisionStage,
    DraftKind,
    Intent,
    Mode,
    ReviewAdvice,
    ReviewState,
    Role,
    RouteAction,
    SuggestionStatus,
)
from rodit.core.errors import ActionNotAllowed
from rodit.core.models import aftersale_idempotency_key
from rodit.core.questions import Q_GROUNDED, Q_PROMISE_RISK
from rodit.core.scopes import PHASE2_SCOPES
from rodit.testing.fakes import qa_answers
from rodit.testing.fixtures import AGENT, SHOP_A

# ================================================================ §11.1 售前：尺码咨询


SIZE_ANSWER = (
    "亲，您身高 165、体重 110 斤，这款建议选 M 码哦，版型偏小，"
    "身高 160–165 cm、体重 100–115 斤穿 M 正合适~"
)


def test_11_1_size_advice_with_item_card_is_adopted_as_is():
    llm = scripted_llm({"穿什么码": llm_answer(SIZE_ANSWER, "item:i_555:size_advice")})
    service = demo_service(
        llm=llm,
        router=scripted_router([("穿什么码", route(Intent.SIZE_ADVICE, 0.92))]),
        qa=scripted_qa(),
    )
    unit = service.unit
    chat = Chat(service, "c_11_1")

    result = chat.say("165 110斤穿什么码", item_id="i_555")

    # 路由：size_advice 0.92 → 规则 6
    assert result.route.intent == Intent.SIZE_ADVICE
    assert result.route.intent_confidence == pytest.approx(0.92)
    assert result.route.action == RouteAction.ANSWER and result.route.rule == 6
    # 知识中心取尺码建议，大模型生成回答并附依据
    [card] = result.suggestions  # 两个候选文字相同，去重后一条
    assert card.kind == DraftKind.LLM
    assert "M" in card.text
    assert [e.id for e in card.evidence] == ["item:i_555:size_advice"]
    [first, *_] = generate_calls(llm)
    assert "[item:i_555:size_advice]" in prompt_text(first)
    # 质检：有依据 0.96，无违规 → 推荐
    assert card.qa_scores[Q_GROUNDED] == pytest.approx(0.96)
    qa_logs = unit.repos.audit.list_decisions(SHOP_A, stage=DecisionStage.QA)
    assert qa_logs and {d.route_result for d in qa_logs} == {"pass"}
    # 推荐出现在客服工作台侧边栏
    sidebar = unit.adapters.get(unit.platform, "B").sidebar(SHOP_A, "c_11_1")
    assert [c.text for c in sidebar] == [card.text]

    # 客服点击发送 → 反馈为原样采纳（§5.8 正样本，计入采纳率）
    outcome = service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    assert outcome.record.status == SuggestionStatus.SENT
    report = service.adoption(SHOP_A)
    assert report.overall.sent == 1 and report.overall.as_is_rate == pytest.approx(1.0)
    size_stats = report.by_intent[Intent.SIZE_ADVICE]
    assert size_stats.sent == 1 and size_stats.adoption_rate == pytest.approx(1.0)
    assert unit.repos.audit.list_events(SHOP_A, action=AuditAction.FEEDBACK_SENT)
    # 发出去的文字写回会话，作为下一轮上下文（§5.3）
    last = unit.repos.conversations.recent_messages(SHOP_A, "c_11_1")[-1]
    assert last.role == Role.AGENT and last.text == card.text


# ================================================================ §11.2 售中：物流查询


def test_11_2_logistics_with_order_card_uses_template_without_llm():
    llm = scripted_llm({})  # 任何生成调用都会被当成错误
    qa_judge = scripted_qa()
    service = demo_service(
        llm=llm,
        router=scripted_router([("快递到哪了", route(Intent.LOGISTICS_TRACK, 0.95))]),
        qa=qa_judge,
    )

    result = Chat(service, "c_11_2").say("我的快递到哪了", order_id="o_1002")

    assert (
        result.route.intent == Intent.LOGISTICS_TRACK and result.route.action == RouteAction.ANSWER
    )
    [card] = result.suggestions
    assert card.kind == DraftKind.TEMPLATE
    assert "截至" in card.text and "10:30" in card.text and "杭州转运中心" in card.text
    assert [e.id for e in card.evidence] == ["tool:logistics:o_1002"]
    # 模板回答不调大模型；质检只跑硬规则（不调判断层、不写 qa 的 decision_log）
    assert llm.calls == []
    assert qa_judge.requests == []
    assert service.unit.repos.audit.list_decisions(SHOP_A, stage=DecisionStage.QA) == []
    [qa_event] = service.unit.repos.audit.list_events(SHOP_A, action=AuditAction.QA)
    assert qa_event.detail["model_checked"] == 0 and qa_event.detail["passed"] == 1


def test_11_5_doc_question_on_logistics_intent_never_promises_arrival():
    """§11.5 的原题"明天能到吗"（物流运输中、没有预计送达时间）：物流问题走模板（§11.2），
    只写截至时间和最新状态，不承诺到货时间（§7.2），根本不给大模型编"明天一定到"的机会。"""
    llm = scripted_llm({})
    service = demo_service(
        llm=llm,
        router=scripted_router([("明天能到吗", route(Intent.LOGISTICS_TRACK, 0.95))]),
        qa=scripted_qa(),
    )
    result = Chat(service, "c_11_5a").say("明天能到吗", order_id="o_1002")
    [card] = result.suggestions
    assert card.kind == DraftKind.TEMPLATE and "截至" in card.text
    assert "明天" not in card.text and "一定" not in card.text
    assert llm.calls == []


# ================================================================ §11.3 售后：显示签收但没收到

NR_ANSWER = (
    "亲，非常抱歉让您着急了，物流显示包裹已由菜鸟驿站（幸福里店）代收，麻烦您先去驿站问一下哦；"
    "如果还是找不到，可以在订单页申请售后，我们帮您核查。"
)


def _grant_refund_write(service, shop_id: str = SHOP_A) -> None:
    """这家店申请了写售后权限（二期，§8.4）：客服确认后执行售后写操作要有 refund.write。"""
    auth = service.unit.repos.auth.get(shop_id)
    service.unit.repos.auth.save(auth.model_copy(update={"scopes": sorted(PHASE2_SCOPES)}))


def _nr_service():
    llm = scripted_llm({"没收到": llm_answer(NR_ANSWER, "tool:logistics:o_1003")})
    service = demo_service(
        llm=llm,
        router=scripted_router([("没收到", route(Intent.NOT_RECEIVED, 0.93))]),
        qa=scripted_qa(),
    )
    _grant_refund_write(service)  # 这些流程一直走到客服确认后执行写操作
    return service


def test_11_3_signed_not_received_pickup_guidance_then_r4_review_and_confirm():
    service = _nr_service()
    unit = service.unit

    # 买家问 → 查物流细分状态（驿站代收）→ 推荐驿站信息和取件指引，并标记风险
    result = Chat(service, "c_11_3").say("显示签收了但我根本没收到", order_id="o_1003")
    assert result.route.intent == Intent.NOT_RECEIVED
    assert result.route.action == RouteAction.HIGH_RISK_ANSWER and result.route.rule == 4
    # 第一条是确定性模板：签收细分状态 + 驿站取件指引 + 没找到就在订单页申请售后（§7.3.1）；
    # 第二条是大模型的回答
    card, llm_card = result.suggestions
    assert card.kind == DraftKind.TEMPLATE
    assert "菜鸟驿站" in card.text and "请凭取件码取件" in card.text and "申请售后" in card.text
    assert [e.id for e in card.evidence] == [
        "tool:logistics:o_1003",
        "platform:taobao:apply_aftersale",
    ]
    assert "请核查物流签收细分状态" in card.risk_hints
    assert llm_card.kind == DraftKind.LLM and llm_card.text == NR_ANSWER
    assert [e.id for e in llm_card.evidence] == ["tool:logistics:o_1003"]
    assert "请核查物流签收细分状态" in llm_card.risk_hints
    service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)  # 客服发送取件指引

    # 买家没找到，在订单页申请仅退款（as_2003）→ 审核助手命中 R4，进入签收核查
    review = service.review_aftersale(SHOP_A, "as_2003")
    assert review.matched_rules == ["R4"]
    assert review.state == ReviewState.VERIFY and review.suggestion == ReviewAdvice.NEGOTIATE
    assert "菜鸟驿站" in (review.reason or "")
    # 不到 24 小时刷新：仍在核查，不能确认
    unit.clock.set(unit.clock.now() + timedelta(hours=23))
    assert service.review_aftersale(SHOP_A, "as_2003").state == ReviewState.VERIFY
    info = unit.tools.get_aftersale(SHOP_A, "as_2003")
    with pytest.raises(ActionNotAllowed):
        service.confirm_aftersale(
            SHOP_A,
            "as_2003",
            AfterSaleAction.AGREE_REFUND,
            operator=AGENT,
            expected_status_version=info.status_version,
        )
    # 24 小时后仍未找到 → 建议同意
    unit.clock.set(unit.clock.now() + timedelta(hours=1, minutes=1))
    review = service.review_aftersale(SHOP_A, "as_2003")
    assert review.state == ReviewState.CONFIRM and review.suggestion == ReviewAdvice.AGREE
    assert unit.action_port.calls == []  # 到这里为止没有任何写操作

    # 客服确认同意 → 重读状态后调用平台同意接口（带幂等键），只调一次
    info = unit.tools.get_aftersale(SHOP_A, "as_2003")
    done = service.confirm_aftersale(
        SHOP_A,
        "as_2003",
        AfterSaleAction.AGREE_REFUND,
        operator=AGENT,
        expected_status_version=info.status_version,
    )
    assert done.state == ReviewState.AGREED and done.confirmed_by == AGENT
    [call] = unit.action_port.calls
    assert call["action"] == AfterSaleAction.AGREE_REFUND and call["operator"] == AGENT
    assert call["key"] == aftersale_idempotency_key(
        "as_2003", AfterSaleAction.AGREE_REFUND, info.status_version
    )
    assert unit.tools.get_aftersale(SHOP_A, "as_2003").status == AfterSaleStatus.REFUNDING
    # 重复点击：已经是终态，不会再调平台
    with pytest.raises(ActionNotAllowed):
        service.confirm_aftersale(
            SHOP_A,
            "as_2003",
            AfterSaleAction.AGREE_REFUND,
            operator=AGENT,
            expected_status_version=info.status_version,
        )
    assert len(unit.action_port.calls) == 1
    # 留痕：谁确认的、依据哪条规则、平台返回结果（§7.3.2）
    audit = unit.repos.audit
    [confirm] = audit.list_events(SHOP_A, action=AuditAction.REVIEW_CONFIRM)
    assert confirm.actor == AGENT and confirm.detail["rules"] == ["R4"]
    [action] = audit.list_events(SHOP_A, action=AuditAction.REVIEW_ACTION)
    assert action.detail["ok"] is True and action.detail["idempotency_key"] == call["key"]


def _to_confirm(service, aftersale_id: str) -> None:
    unit = service.unit
    service.review_aftersale(SHOP_A, aftersale_id)
    unit.clock.set(unit.clock.now() + timedelta(hours=25))
    assert service.review_aftersale(SHOP_A, aftersale_id).state == ReviewState.CONFIRM


def test_11_3_reread_race_buyer_withdraws_before_confirm():
    """客服看到"建议同意"后、点确认前，买家找到包裹撤销了售后：重读发现已不可操作 → 平台已处理，
    不调写接口（§7.3.2 调用前重新读取一次售后单状态）。"""
    service = _nr_service()
    unit = service.unit
    _to_confirm(service, "as_2003")
    seen = unit.tools.get_aftersale(SHOP_A, "as_2003")  # 客服确认时看到的版本
    unit.tools.set_aftersale_status(SHOP_A, "as_2003", AfterSaleStatus.CLOSED, at=unit.clock.now())
    done = service.confirm_aftersale(
        SHOP_A,
        "as_2003",
        AfterSaleAction.AGREE_REFUND,
        operator=AGENT,
        expected_status_version=seen.status_version,
    )
    assert done.state == ReviewState.PLATFORM and done.matched_rules == ["R1"]
    assert unit.action_port.calls == []


def test_11_3_reread_race_status_version_changed():
    """状态没变但版本变了（买家补充了凭证）：客服看到的是旧版本 → 不执行；
    建议也是按旧版本给的 → 刷新后（签收核查的结论不丢）按新版本重新确认。"""
    service = _nr_service()
    unit = service.unit
    _to_confirm(service, "as_2003")
    seen = unit.tools.get_aftersale(SHOP_A, "as_2003")
    unit.tools.set_aftersale_status(
        SHOP_A, "as_2003", AfterSaleStatus.PENDING_REVIEW, at=unit.clock.now(), evidence_count=1
    )
    with pytest.raises(ActionNotAllowed, match="状态已变化"):
        service.confirm_aftersale(
            SHOP_A,
            "as_2003",
            AfterSaleAction.AGREE_REFUND,
            operator=AGENT,
            expected_status_version=seen.status_version,
        )
    assert unit.action_port.calls == []
    fresh = unit.tools.get_aftersale(SHOP_A, "as_2003")
    assert fresh.status_version != seen.status_version
    with pytest.raises(ActionNotAllowed, match="状态已变化"):
        service.confirm_aftersale(
            SHOP_A,
            "as_2003",
            AfterSaleAction.AGREE_REFUND,
            operator=AGENT,
            expected_status_version=fresh.status_version,
        )
    assert unit.action_port.calls == []
    refreshed = service.review_aftersale(SHOP_A, "as_2003")
    assert refreshed.state == ReviewState.CONFIRM
    assert refreshed.suggestion == ReviewAdvice.AGREE  # 核查已满 24 小时，不回到"先协商"
    assert refreshed.platform_status_version == fresh.status_version
    done = service.confirm_aftersale(
        SHOP_A,
        "as_2003",
        AfterSaleAction.AGREE_REFUND,
        operator=AGENT,
        expected_status_version=fresh.status_version,
    )
    assert done.state == ReviewState.AGREED
    [call] = unit.action_port.calls
    assert call["key"] == aftersale_idempotency_key(
        "as_2003", AfterSaleAction.AGREE_REFUND, fresh.status_version
    )


# ================================================================ R3 两步（退货 → 退款）


def test_r3_two_step_return_then_refund():
    service = demo_service()
    _grant_refund_write(service)
    unit = service.unit
    review = service.review_aftersale(SHOP_A, "as_2002")
    assert review.matched_rules == ["R3"]
    assert review.state == ReviewState.CONFIRM and review.suggestion == ReviewAdvice.AGREE_RETURN

    v1 = unit.tools.get_aftersale(SHOP_A, "as_2002").status_version
    step1 = service.confirm_aftersale(
        SHOP_A, "as_2002", AfterSaleAction.AGREE_RETURN, operator=AGENT, expected_status_version=v1
    )
    assert step1.state == ReviewState.WAIT_RETURN
    assert unit.tools.get_aftersale(SHOP_A, "as_2002").status == AfterSaleStatus.WAIT_BUYER_RETURN
    # 买家还没寄回：刷新不变
    unit.clock.set(unit.clock.now() + timedelta(hours=2))
    assert service.review_aftersale(SHOP_A, "as_2002").state == ReviewState.WAIT_RETURN

    # 买家寄回，平台变成"等商家确认收货" → 刷新回到待客服确认，建议同意退款
    unit.clock.set(unit.clock.now() + timedelta(days=2))
    unit.tools.set_aftersale_status(
        SHOP_A, "as_2002", AfterSaleStatus.WAIT_SELLER_CONFIRM, at=unit.clock.now()
    )
    review = service.review_aftersale(SHOP_A, "as_2002")
    assert review.state == ReviewState.CONFIRM and review.suggestion == ReviewAdvice.AGREE
    assert review.platform_status == AfterSaleStatus.WAIT_SELLER_CONFIRM

    v2 = unit.tools.get_aftersale(SHOP_A, "as_2002").status_version
    step2 = service.confirm_aftersale(
        SHOP_A, "as_2002", AfterSaleAction.AGREE_REFUND, operator=AGENT, expected_status_version=v2
    )
    assert step2.state == ReviewState.AGREED
    assert unit.tools.get_aftersale(SHOP_A, "as_2002").status == AfterSaleStatus.REFUNDING
    calls = unit.action_port.calls
    assert [c["action"] for c in calls] == [
        AfterSaleAction.AGREE_RETURN,
        AfterSaleAction.AGREE_REFUND,
    ]
    assert len({c["key"] for c in calls}) == 2  # 两步各自一个幂等键


# ================================================================ §11.4 买家要求站外交易


def test_11_4_off_platform_request_gets_standard_script_without_any_llm_call():
    llm = scripted_llm({})
    qa_judge = scripted_qa()
    service = demo_service(
        llm=llm,
        router=scripted_router([("微信", route(Intent.OFF_PLATFORM, 0.97))]),
        qa=qa_judge,
    )
    result = Chat(service, "c_11_4").say("加个微信吧，转账给你便宜点")
    assert result.route.action == RouteAction.OFF_PLATFORM_SCRIPT and result.route.rule == 1
    [card] = result.suggestions
    assert card.kind == DraftKind.TEMPLATE
    assert card.text == (
        "亲，平台规定不能通过站外联系和交易哦，您直接在店铺下单就好，"
        "我帮您看看有没有可以用的优惠券。"
    )
    assert llm.calls == [] and qa_judge.requests == []


# ================================================================ §11.5 质检拦截，带原因重写

PROMISE = "亲，明天一定能到哦！"
GROUNDED = "亲，现货付款后 48 小时内发货，默认发中通快递，到货时间以物流信息为准哦~"


def test_11_5_blocked_promise_is_regenerated_once_with_reasons():
    def generate(request):
        # 第一次写出依据外的时效承诺；带着质检原因重写时只用依据
        text = prompt_text(request)
        return llm_answer(GROUNDED if RETRY_MARK in text else PROMISE, "policy:ship_time")

    llm = scripted_llm({"明天能到吗": generate})
    qa_judge = scripted_qa([("明天", qa_answers(grounded=0.35, promise_risk=0.86))])
    service = demo_service(
        llm=llm,
        router=scripted_router([("明天能到吗", route(Intent.SHIPPING_POLICY, 0.9))]),
        qa=qa_judge,
    )
    result = Chat(service, "c_11_5").say("今天下单，明天能到吗")

    [card] = result.suggestions
    assert card.text == GROUNDED and card.kind == DraftKind.LLM
    assert card.qa_scores[Q_PROMISE_RISK] == pytest.approx(0.05)

    calls = generate_calls(llm)
    first = [c for c in calls if RETRY_MARK not in prompt_text(c)]
    retries = [c for c in calls if RETRY_MARK in prompt_text(c)]
    assert first and retries  # 生成 → 拦截 → 带原因重新生成（只重写一轮）
    assert calls.index(retries[0]) > calls.index(first[-1])
    for retry in retries:
        text = prompt_text(retry)
        assert "时效承诺：明天" in text  # 硬规则：证据里找不到"明天"
        assert "绝对化用语：一定" in text  # 硬规则：绝对化用语
        assert "promise_risk 0.86" in text  # 判断层质检
    # decision_log：第一次拦截、第二次放行
    results = [
        d.route_result
        for d in service.unit.repos.audit.list_decisions(SHOP_A, stage=DecisionStage.QA)
    ]
    assert results == ["block", "pass"]
    [qa_event] = service.unit.repos.audit.list_events(SHOP_A, action=AuditAction.QA)
    assert qa_event.detail["regenerations"] == 1
    [summary] = service.unit.repos.audit.list_events(SHOP_A, action=AuditAction.RESULT)
    assert {"time_promise", "absolute_term"} <= set(summary.detail["hard_hits"])
    assert summary.detail["qa_failed"] == 1


# ================================================================ 全自动（§5.6、§5.7、§12.3）


def _auto_service():
    service = demo_service(
        router=scripted_router(
            [
                ("在吗", route(Intent.GREETING, 0.98)),
                ("什么时候发货", route(Intent.SHIPPING_POLICY, 0.97)),
                ("便宜点", route(Intent.PRICE_PROMO, 0.97)),
            ]
        ),
    )
    return service, enable_auto(service)


@pytest.mark.parametrize(
    ("text", "intent", "kind"),
    [
        ("在吗", Intent.GREETING, DraftKind.TEMPLATE),
        ("什么时候发货", Intent.SHIPPING_POLICY, DraftKind.FAQ),
    ],
)
def test_auto_mode_sends_eligible_answer_once_with_ai_prefix(text, intent, kind):
    service, adapter = _auto_service()
    unit = service.unit
    prefix = unit.config.generation.auto_reply_prefix
    assert prefix == "【智能客服】"

    result = Chat(service, "c_auto", access="A", robot=True).say(text)

    assert result.mode == Mode.AUTO
    assert result.route.intent == intent and result.route.auto_eligible
    assert result.auto_reply is not None and not result.transfer_to_human
    [card] = result.suggestions
    assert card.kind == kind and not card.text.startswith(prefix)
    # 平台只收到一条，开头恰好一个 AI 生成内容标识
    [sent] = adapter.sent_messages(SHOP_A)
    assert sent.text == prefix + card.text and sent.text.count(prefix) == 1
    assert adapter.transfers(SHOP_A) == []
    # 推荐记录标为已发送（机器人发出），发出的文字写回会话
    record = unit.repos.suggestions.get(SHOP_A, card.suggestion_id)
    assert record.status == SuggestionStatus.SENT and record.agent_id == "bot"
    last = unit.repos.conversations.recent_messages(SHOP_A, "c_auto")[-1]
    assert last.role == Role.BOT and last.text == sent.text
    assert unit.repos.audit.list_events(SHOP_A, action=AuditAction.AUTO_REPLY)


def test_auto_mode_high_risk_question_transfers_to_human():
    service, adapter = _auto_service()
    result = Chat(service, "c_auto_risk", access="A", robot=True).say("能便宜点吗")
    assert result.mode == Mode.AUTO
    assert result.route.action == RouteAction.TRANSFER_HUMAN and result.route.rule == 4
    assert result.transfer_to_human and result.auto_reply is None
    assert adapter.sent_messages(SHOP_A) == []
    [transfer] = adapter.transfers(SHOP_A)
    assert transfer.conversation_id == "c_auto_risk"
    # 转过人工的会话，后面的消息不再全自动（由客服接管）
    follow = Chat(service, "c_auto_risk", access="A", robot=True)
    follow.n = 1
    assert follow.say("在吗").mode == Mode.ASSIST
    assert adapter.sent_messages(SHOP_A) == []


def test_auto_mode_requires_shop_switch():
    """店铺没开全自动总开关时，平台说是机器人接待也只按智能辅助处理（§3 D2、§5.6）。"""
    service, adapter = _auto_service()
    repo = service.unit.repos.knowledge
    shop = repo.get_shop(SHOP_A)
    repo.upsert_shop(
        shop.model_copy(
            update={"config_overrides": {**shop.config_overrides, "auto_reply_enabled": False}}
        )
    )
    result = Chat(service, "c_auto_off", access="A", robot=True).say("在吗")
    assert result.mode == Mode.ASSIST and result.auto_reply is None
    assert result.suggestions and adapter.sent_messages(SHOP_A) == []
