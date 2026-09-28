"""T0 契约自检：模型、判断层格式、配置分层、证据规则、测试替身、合成数据的一致性。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from rodit.core import evidence as ev
from rodit.core.clock import CST, fmt_time
from rodit.core.config import (
    DecisionConfig,
    EngineConfig,
    effective_shop_config,
    load_config,
    resolve_shop_config,
)
from rodit.core.decision import (
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    require_buyer_data_ok,
    validate_response,
)
from rodit.core.enums import (
    ACTIONABLE_STATUSES,
    INTENT_EXAMPLES,
    INTENT_LABELS,
    INTENT_STAGE,
    REVIEW_TRANSITIONS,
    AfterSaleAction,
    AfterSaleStatus,
    EvidenceSource,
    Intent,
    Platform,
    ProviderName,
    ReviewState,
    Role,
    RouteAction,
    is_actionable,
)
from rodit.core.errors import ActionNotAllowed, ComplianceError, DecisionError, ToolUnavailable
from rodit.core.models import (
    HumanConfirmation,
    ItemCard,
    Shop,
    StandardMessage,
    aftersale_idempotency_key,
)
from rodit.core.pipeline import ACTION_PLANS
from rodit.core.questions import (
    QA_QUESTIONS,
    QA_QUESTIONS_VER,
    ROUTER_QUESTIONS,
    ROUTER_QUESTIONS_VER,
    questions_version,
)
from rodit.core.states import build_qa_state, build_router_state, recent_turns
from rodit.testing.fakes import (
    FakeActionPort,
    FakeClock,
    FakeDecisionProvider,
    FakeLLMClient,
    InMemoryToolGateway,
    confirmation_for,
    make_context,
    make_message,
    qa_answers,
    router_answers,
)
from rodit.testing.fixtures import (
    BUYER,
    BUYER_RISKY,
    DEFAULT_NOW,
    RAW_BUYERS,
    SHOP_A,
    SHOP_B,
    SHOP_C,
    SHOP_D,
    TEST_SALT,
    build_demo_world,
)

ROOT = Path(__file__).resolve().parents[2]
YAML = ROOT / "config" / "rodit.yaml"


# ---------------------------------------------------------------- 词表与证据规则


def test_intent_tables_cover_all_intents():
    for table in (INTENT_STAGE, INTENT_LABELS, INTENT_EXAMPLES, ev.INTENT_NEEDS):
        assert set(table) == set(Intent)
    assert set(ev.INTENT_POLICY_KEYS) <= set(Intent)
    assert set(ev.INTENT_PLATFORM_RULES) <= set(Intent)
    assert ev.KNOWLEDGE_NEEDS.isdisjoint(ev.TOOL_NEEDS)
    assert ev.KNOWLEDGE_NEEDS | ev.TOOL_NEEDS == set(ev.Need)
    # 有平台规则键的问题类型必须声明 PLATFORM_RULE 依据，反之亦然
    for intent in Intent:
        assert bool(ev.platform_rules_for(intent)) == (
            ev.Need.PLATFORM_RULE in ev.needs_for(intent)
        )
        assert bool(ev.policy_keys_for(intent)) == (ev.Need.POLICY in ev.needs_for(intent))


def test_evidence_priority_order():
    order = sorted(EvidenceSource, key=lambda s: s.priority)
    assert order == [
        EvidenceSource.REALTIME,
        EvidenceSource.PLATFORM,
        EvidenceSource.POLICY,
        EvidenceSource.ITEM,
        EvidenceSource.FAQ,
        EvidenceSource.CATEGORY,
    ]


def test_evidence_ids_and_labels():
    assert ev.item_detail("i_555") == "item:i_555:detail"
    assert ev.item_detail("i_555", 2) == "item:i_555:detail:2"
    assert ev.platform_rule(Platform.TAOBAO, "apply_aftersale") == "platform:taobao:apply_aftersale"
    assert ev.platform_rule_label("taobao", "apply_aftersale") == "平台规则 · 淘宝申请售后"
    assert ev.policy_label("ship_time") == "店铺政策 · 发货时效"
    assert ev.tool_stock("i_555", "k_1") == "tool:stock:i_555:k_1"
    assert ev.topic_for_stock("i_555", "k_1") != ev.topic_for_stock("i_777", "k_1")


def test_detect_topics_on_detail_text_conflict():
    """§5.2.1：详情页"24 小时发货"这句和店铺政策 ship_time 同主题，其他句子不受影响。"""
    world = build_demo_world()
    sentences = ev.split_sentences(world.item(SHOP_A, "i_555").detail_text)
    tagged = {s: ev.detect_topics(s) for s in sentences}
    ship = [s for s, t in tagged.items() if "ship_time" in t]
    assert ship == ["现货付款后 24 小时内发货。"]
    assert all(not t for s, t in tagged.items() if s not in ship)
    assert "ship_time" in ev.detect_topics(world.policy_map(SHOP_A)["ship_time"].value)
    assert "promotions" in ev.detect_topics("店铺满 199 减 20")
    assert ev.detect_topics("面料亲肤透气") == set()


def test_action_plans_cover_all_actions():
    assert set(ACTION_PLANS) == set(RouteAction)
    # 站外话术只取平台站外交易规则当依据（§5.1、§5.6），不用大模型
    assert ACTION_PLANS[RouteAction.OFF_PLATFORM_SCRIPT].retrieve
    assert not ACTION_PLANS[RouteAction.HUMAN_PRIORITY].generate
    assert ACTION_PLANS[RouteAction.SPLIT_MULTI].per_part


# ---------------------------------------------------------------- 判断层格式


def test_router_questions_match_doc_shape():
    intent_q = ROUTER_QUESTIONS["intent"]
    assert intent_q.type == "choice"
    assert list(intent_q.criteria) == [i.value for i in Intent]
    assert ROUTER_QUESTIONS["emotion"].criteria == ["平静", "略有不满", "不满", "愤怒", "极度愤怒"]
    assert set(QA_QUESTIONS) == {
        "grounded",
        "answers_question",
        "promise_risk",
        "platform_violation",
        "tone",
    }
    assert ROUTER_QUESTIONS_VER == questions_version(ROUTER_QUESTIONS)
    assert ROUTER_QUESTIONS_VER != QA_QUESTIONS_VER


def test_decision_request_roundtrip_jev_format():
    """§6.5 的路由请求 / 响应示例可以原样解析。"""
    raw = {
        "model": "router-v1",
        "state": {
            "platform": "douyin",
            "messages": [{"role": "buyer", "text": "165 110斤穿什么码"}],
            "item_card": {"item_id": "i_555", "category": "女装/连衣裙"},
            "order_card": None,
        },
        "questions": {k: q.model_dump() for k, q in ROUTER_QUESTIONS.items()},
    }
    req = DecisionRequest.model_validate(raw)
    resp = DecisionResponse.model_validate(
        {
            "model": "router-v1",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": "size_advice",
                    "confidence": 0.92,
                    "probabilities": {"size_advice": 0.92, "product_spec": 0.05, "other": 0.03},
                },
                "need_human": {"type": "noul", "noul": 0.03},
                "emotion": {"type": "score", "score": 0.2},
                "multi_intent": {"type": "noul", "noul": 0.04},
            },
        }
    )
    validate_response(req, resp)
    assert resp.choice("intent").top(2) == [("size_advice", 0.92), ("product_spec", 0.05)]
    assert resp.noul("need_human") == 0.03
    assert json.loads(resp.model_dump_json())["answers"]["emotion"] == {
        "type": "score",
        "score": 0.2,
    }


def test_validate_response_rejects_incomplete_or_invalid():
    req = DecisionRequest(model="qa-v1", state={}, questions=dict(QA_QUESTIONS))
    with pytest.raises(DecisionError):
        validate_response(req, DecisionResponse(model="qa-v1", answers={}))
    bad = dict(qa_answers())
    bad["grounded"] = ChoiceAnswer(choice="x", confidence=0.5)
    with pytest.raises(DecisionError):
        validate_response(req, DecisionResponse(model="qa-v1", answers=bad))
    over = dict(qa_answers(tone=9))
    with pytest.raises(DecisionError):
        validate_response(req, DecisionResponse(model="qa-v1", answers=over))
    with pytest.raises(ValidationError):
        NoulAnswer(noul=1.5)


def test_fake_decision_provider_and_buyer_data_guard():
    provider = FakeDecisionProvider({**router_answers("size_advice"), **qa_answers()})
    req = DecisionRequest(model="router-v1", state={}, questions=dict(ROUTER_QUESTIONS))
    resp = provider.decide(req)
    assert set(resp.answers) == set(ROUTER_QUESTIONS)
    assert resp.choice("intent").choice == "size_advice"
    require_buyer_data_ok(provider)
    with pytest.raises(ComplianceError):
        require_buyer_data_ok(FakeDecisionProvider({}, name="jev", accepts_buyer_data=False))
    down = FakeDecisionProvider({}, fail=DecisionError("down"))
    with pytest.raises(DecisionError):
        down.decide(req)


def test_state_builders_follow_doc_shape_and_mask_pii():
    world = build_demo_world()
    ctx = make_context(
        "我手机 13812345678，165 110斤穿什么码",
        world,
        item_id="i_555",
        order_id="o_1001",
        history=[(Role.BUYER, "在吗"), (Role.AGENT, "亲，在的")],
    )
    state = build_router_state(ctx)
    assert set(state) == {"platform", "messages", "item_card", "order_card"}
    assert state["item_card"] == {
        "item_id": "i_555",
        "category": "女装/连衣裙",
        "title": "法式碎花连衣裙",
    }
    assert state["order_card"] == {"order_id": "o_1001", "status": "paid"}
    assert state["messages"][-1] == {"role": "buyer", "text": "我手机 [手机号]，165 110斤穿什么码"}
    assert "13812345678" not in json.dumps(state, ensure_ascii=False)
    assert BUYER not in json.dumps(state, ensure_ascii=False)
    from rodit.core.enums import DraftKind
    from rodit.core.models import Draft, Evidence, EvidenceBundle

    draft = Draft(
        text="亲，建议选 M 码哦", evidence_ids=["item:i_555:size_advice"], kind=DraftKind.LLM
    )
    bundle = EvidenceBundle(
        evidences=[
            Evidence(
                id="item:i_555:size_advice",
                source=EvidenceSource.ITEM,
                label="商品属性 · 尺码建议",
                content="身高 160–165 cm、体重 100–115 斤建议 M",
            )
        ]
    )
    qa_state = build_qa_state(ctx, draft, bundle)
    assert qa_state["answer"] == "亲，建议选 M 码哦"
    assert qa_state["evidence"][0]["id"] == "item:i_555:size_advice"
    assert "13812345678" not in json.dumps(qa_state, ensure_ascii=False)


def test_recent_turns_counts_buyer_rounds():
    world = build_demo_world()
    history = []
    for i in range(7):
        history += [(Role.BUYER, f"问题{i}"), (Role.AGENT, f"回答{i}")]
    ctx = make_context("最新问题", world, history=history)
    turns = recent_turns(ctx, history_rounds=5)
    assert [t.text for t in turns][0] == "问题2"
    assert turns[-1].text == "最新问题"
    assert [t.text for t in recent_turns(ctx, history_rounds=0)] == ["最新问题"]


# ---------------------------------------------------------------- 配置


def test_config_yaml_matches_defaults():
    cfg = load_config(YAML)
    default = EngineConfig()
    assert cfg.routing == default.routing
    assert cfg.qa_gate.grounded_min == 0.8
    assert cfg.routing.auto_threshold[Intent.RETURN_POLICY] == 0.97
    assert Intent.COMPLAINT in cfg.routing.high_risk
    assert cfg.decision.provider == ProviderName.INCLOUD_LLM


def test_jev_cannot_be_configured_for_production():
    with pytest.raises(ValidationError):
        DecisionConfig(provider="jev")
    assert DecisionConfig(provider="jev", jev_production_approved=True).provider == "jev"


def test_shop_config_layering_survives_json_roundtrip():
    """§9.3：SHOP_C 没有覆盖 max_answer_chars，生效值来自拼多多平台默认；落库再读回也一样。"""
    cfg = load_config(YAML)
    world = build_demo_world()
    shop_c = world.shop(SHOP_C)
    assert effective_shop_config(cfg, shop_c).max_answer_chars == 100
    roundtrip = Shop.model_validate_json(shop_c.model_dump_json())
    assert effective_shop_config(cfg, roundtrip).max_answer_chars == 100
    assert effective_shop_config(cfg, roundtrip).tone == "简洁"  # 店铺覆盖
    a = effective_shop_config(cfg, world.shop(SHOP_A))
    assert a.manager_ids == ["kf_manager"] and a.aftersale_amount_limit == Decimal("500")
    assert effective_shop_config(cfg, world.shop(SHOP_D)).shadow_mode is True
    sc = resolve_shop_config(
        cfg, platform=Platform.PDD, category="女装", shop_override={"salutation": "您"}
    )
    assert sc.max_answer_chars == 100 and sc.tone == "亲切" and sc.salutation == "您"


def test_shop_overrides_are_validated():
    with pytest.raises(ValidationError):
        Shop(id="s", platform=Platform.TAOBAO, config_overrides={"no_such_key": 1})
    with pytest.raises(ValidationError):
        Shop(id="s", platform=Platform.TAOBAO, config_overrides={"suggestion_count": 9})


# ---------------------------------------------------------------- 售后契约


def test_review_transitions_and_actionable_statuses():
    assert ReviewState.WAIT_RETURN in REVIEW_TRANSITIONS[ReviewState.CONFIRM]
    assert REVIEW_TRANSITIONS[ReviewState.WAIT_RETURN] == {
        ReviewState.CONFIRM,
        ReviewState.PLATFORM,
    }
    for terminal in (ReviewState.AGREED, ReviewState.REFUSED, ReviewState.PLATFORM):
        assert not REVIEW_TRANSITIONS[terminal]
    assert set(REVIEW_TRANSITIONS) == set(ReviewState)
    assert AfterSaleStatus.WAIT_SELLER_CONFIRM in ACTIONABLE_STATUSES[AfterSaleAction.AGREE_REFUND]
    assert (
        AfterSaleStatus.WAIT_SELLER_CONFIRM not in ACTIONABLE_STATUSES[AfterSaleAction.AGREE_RETURN]
    )
    assert not is_actionable(AfterSaleStatus.CLOSED) and is_actionable(
        AfterSaleStatus.PENDING_REVIEW
    )


def test_human_confirmation_rejects_system_operator():
    with pytest.raises(ValidationError):
        HumanConfirmation(operator="system", confirmed_at=DEFAULT_NOW, expected_status_version="1")
    assert aftersale_idempotency_key("as_1", "agree_refund", "123") == "as_1:agree_refund:123"


def test_fake_action_port_follows_protocol():
    w = build_demo_world()
    clock = FakeClock(w.now)
    gw = InMemoryToolGateway(w)
    port = FakeActionPort(gateway=gw, clock=clock)
    info = gw.get_aftersale(SHOP_A, "as_2001")
    key = aftersale_idempotency_key("as_2001", "agree_refund", info.status_version)
    conf = confirmation_for(info)
    r1 = port.agree_refund(SHOP_A, "as_2001", confirmation=conf, idempotency_key=key)
    r2 = port.agree_refund(SHOP_A, "as_2001", confirmation=conf, idempotency_key=key)
    assert r1.ok and not r1.duplicate and r2.duplicate
    assert len(port.calls) == 1
    assert gw.get_aftersale(SHOP_A, "as_2001").status == AfterSaleStatus.REFUNDING
    # 已不是待审核：不改平台状态，返回 STATUS_CHANGED
    closed = gw.get_aftersale(SHOP_A, "as_2006")
    res = port.agree_refund(
        SHOP_A, "as_2006", confirmation=confirmation_for(closed), idempotency_key="k_closed"
    )
    assert not res.ok and res.platform_code == "STATUS_CHANGED"
    assert gw.get_aftersale(SHOP_A, "as_2006").status == AfterSaleStatus.CLOSED
    # 客服确认后平台状态变了（版本不一致）
    info2 = gw.get_aftersale(SHOP_A, "as_2002")
    stale = confirmation_for(info2)
    gw.set_aftersale_status(
        SHOP_A, "as_2002", AfterSaleStatus.PENDING_REVIEW, at=w.now + timedelta(minutes=1)
    )
    res = port.agree_return(SHOP_A, "as_2002", confirmation=stale, idempotency_key="k_stale")
    assert not res.ok and res.platform_code == "STATUS_CHANGED"
    # 不存在 / 跨店
    with pytest.raises(ActionNotAllowed):
        port.agree_refund(SHOP_B, "as_2001", confirmation=conf, idempotency_key="k_x")
    # 平台差异
    pdd = FakeActionPort(Platform.PDD, supported={AfterSaleAction.AGREE_REFUND})
    with pytest.raises(ActionNotAllowed):
        pdd.refuse(SHOP_A, "as_2001", confirmation=conf, idempotency_key="k")
    douyin = FakeActionPort(Platform.DOUYIN)
    with pytest.raises(ActionNotAllowed):
        douyin.refuse(SHOP_A, "as_2001", confirmation=conf, idempotency_key="k")
    port.fail_next = True
    info3 = gw.get_aftersale(SHOP_A, "as_2005")
    res = port.agree_return(
        SHOP_A, "as_2005", confirmation=confirmation_for(info3), idempotency_key="k_fail"
    )
    assert not res.ok and res.platform_code == "PLATFORM_ERROR"


# ---------------------------------------------------------------- 合成数据


def test_models_require_timezone():
    with pytest.raises(ValidationError):
        StandardMessage(
            platform=Platform.TAOBAO,
            shop_id=SHOP_A,
            conversation_id="c",
            msg_id="m",
            buyer_ref=BUYER,
            text="hi",
            received_at=datetime(2026, 9, 28, 10, 0),
        )


def test_buyer_refs_derive_from_raw_ids():
    from rodit.core.pii import hash_buyer_id

    platform, raw = RAW_BUYERS["BUYER"]
    assert BUYER == hash_buyer_id(platform, raw, salt=TEST_SALT)
    assert len(BUYER) == 26 and raw not in BUYER


def test_demo_world_is_consistent():
    w = build_demo_world()
    assert w.now == DEFAULT_NOW
    card = w.item(SHOP_A, "i_555")
    assert isinstance(card, ItemCard)
    assert card.find_sku("k_1").props() == {"颜色": "黑色", "尺码": "M"}
    # 串店数据：同一平台、同一个 item_id，不同店铺内容不同
    assert w.shop(SHOP_B).platform == w.shop(SHOP_A).platform
    assert w.item(SHOP_B, "i_555").title != card.title
    for o in w.orders:
        assert o.total_amount == sum((ln.price * ln.quantity for ln in o.lines), Decimal("0"))
        if o.shipped_at and o.promised_ship_by:
            assert o.shipped_at <= o.promised_ship_by, o.order_id
    orders = {(o.shop_id, o.order_id): o for o in w.orders}
    for a in w.aftersales:
        order = orders[(a.shop_id, a.order_id)]
        assert order.buyer_ref == a.buyer_ref
        assert a.updated_at >= a.created_at
        if order.signed_at and a.reason in ("no_reason", "quality", "damaged"):
            assert a.created_at >= order.signed_at, a.aftersale_id
    # 每个订单最多一张待处理售后单
    pending = [(a.shop_id, a.order_id) for a in w.aftersales if a.status == "pending_review"]
    assert len(pending) == len(set(pending))
    # 只有高频买家在 30 天内有 ≥ 3 单售后（R7）
    since = w.now - timedelta(days=30)
    counts: dict[str, int] = {}
    for a in w.aftersales:
        if a.shop_id == SHOP_A and a.created_at >= since:
            counts[a.buyer_ref] = counts.get(a.buyer_ref, 0) + 1
    assert {b for b, n in counts.items() if n >= 3} == {BUYER_RISKY}
    # 物流轨迹按时间升序，最后一条不晚于 now
    for t in w.traces:
        times = [e.time for e in t.events]
        assert times == sorted(times) and times[-1] <= w.now
    # 同一买家的订单下单时间不重复（最近订单排序确定）
    by_buyer: dict[tuple[str, str], list[datetime]] = {}
    for o in w.orders:
        by_buyer.setdefault((o.shop_id, o.buyer_ref), []).append(o.created_at)
    for times in by_buyer.values():
        assert len(times) == len(set(times))
    assert w.shop(SHOP_D).platform == Platform.DOUYIN


def test_tool_gateway_isolation_ordering_and_failures():
    w = build_demo_world()
    gw = InMemoryToolGateway(w)
    assert gw.get_order(SHOP_A, "o_1001") is not None
    assert gw.get_order(SHOP_B, "o_1001") is None  # 不跨店
    assert [o.order_id for o in gw.list_recent_orders(SHOP_B, BUYER)] == ["o_9001"]
    recent = gw.list_recent_orders(SHOP_A, BUYER)
    assert [o.created_at for o in recent] == sorted((o.created_at for o in recent), reverse=True)
    assert recent[0].order_id == "o_1001"
    assert gw.get_stock(SHOP_A, "i_555", "k_3").in_stock is False
    item_level = gw.get_stock(SHOP_A, "i_555")
    assert item_level.in_stock and item_level.quantity == 35
    unknown_qty = gw.get_stock(SHOP_C, "i_c1")
    assert unknown_qty.in_stock and unknown_qty.quantity is None
    risky = gw.list_aftersales(SHOP_A, buyer_ref=BUYER_RISKY)
    assert [a.aftersale_id for a in risky] == ["as_3001", "as_3002", "as_3003", "as_3004"]
    gw.fail_on.add("get_logistics")
    with pytest.raises(ToolUnavailable):
        gw.get_logistics(SHOP_A, "o_1002")


def test_fake_llm_client_modes():
    llm = FakeLLMClient(responses=['{"answer": "好的"}', RuntimeError("boom")])
    from rodit.core.ports import LLMMessage, LLMRequest

    req = LLMRequest(purpose="generate", messages=[LLMMessage(role="user", content="hi")])
    assert llm.complete(req).text == '{"answer": "好的"}'
    with pytest.raises(RuntimeError):
        llm.complete(req)
    assert len(llm.calls_for("generate")) == 2


def test_make_message_and_context():
    w = build_demo_world()
    msg = make_message("165 110斤穿什么码", world=w, item_id="i_555")
    assert msg.platform == Platform.TAOBAO and msg.item_card.item_id == "i_555"
    assert make_message("在吗").received_at == DEFAULT_NOW  # 不依赖机器时区
    ctx = make_context(
        "那件黑的有 L 吗",
        w,
        item_id="i_555",
        history=[(Role.BUYER, "在吗"), (Role.AGENT, "亲，在的")],
        with_recent_orders=True,
        config_update={"suggestion_count": 3},
    )
    assert ctx.item.title == "法式碎花连衣裙"
    assert [t.role for t in ctx.history] == [Role.BUYER, Role.AGENT]
    assert ctx.recent_orders and all(o.buyer_ref == BUYER for o in ctx.recent_orders)
    assert ctx.config.suggestion_count == 3 and ctx.config.display_name == "小雅女装旗舰店"


def test_fmt_time():
    assert fmt_time(datetime(2026, 9, 28, 10, 30, tzinfo=CST)) == "09-28 10:30"
    assert fmt_time(datetime(2026, 9, 28, 10, 30, tzinfo=CST), with_date=False) == "10:30"
