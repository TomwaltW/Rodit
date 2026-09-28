"""T0 契约自检：模型、判断层格式、配置、个人信息最小化、测试替身。"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from rodit.core import evidence as ev
from rodit.core.clock import CST, fmt_time
from rodit.core.config import EngineConfig, load_config, resolve_shop_config
from rodit.core.decision import (
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    validate_response,
)
from rodit.core.enums import (
    INTENT_EXAMPLES,
    INTENT_LABELS,
    INTENT_STAGE,
    AfterSaleAction,
    AfterSaleStatus,
    EvidenceSource,
    Intent,
    Platform,
    Role,
)
from rodit.core.errors import ActionNotAllowed, DecisionError, ToolUnavailable
from rodit.core.models import ItemCard, StandardMessage
from rodit.core.pii import contains_pii, hash_buyer_id, mask_pii
from rodit.core.questions import (
    QA_QUESTIONS,
    QA_QUESTIONS_VER,
    ROUTER_QUESTIONS,
    ROUTER_QUESTIONS_VER,
    questions_version,
)
from rodit.testing.fakes import (
    FakeActionPort,
    FakeClock,
    FakeDecisionProvider,
    FakeLLMClient,
    InMemoryToolGateway,
    make_context,
    make_message,
    qa_answers,
    router_answers,
)
from rodit.testing.fixtures import (
    BUYER,
    BUYER_RISKY,
    DEFAULT_NOW,
    SHOP_A,
    SHOP_B,
    build_demo_world,
)

ROOT = Path(__file__).resolve().parents[2]


def test_intent_tables_cover_all_intents():
    for table in (INTENT_STAGE, INTENT_LABELS, INTENT_EXAMPLES, ev.INTENT_NEEDS):
        assert set(table) == set(Intent)
    assert set(ev.INTENT_POLICY_KEYS) <= set(Intent)
    assert ev.KNOWLEDGE_NEEDS.isdisjoint(ev.TOOL_NEEDS)
    assert ev.KNOWLEDGE_NEEDS | ev.TOOL_NEEDS == set(ev.Need)


def test_evidence_priority_order():
    order = sorted(EvidenceSource, key=lambda s: s.priority)
    assert order == [
        EvidenceSource.REALTIME,
        EvidenceSource.POLICY,
        EvidenceSource.ITEM,
        EvidenceSource.FAQ,
        EvidenceSource.CATEGORY,
    ]


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
    """§6.5 的路由请求示例可以原样解析。"""
    raw = {
        "model": "router-v1",
        "state": {
            "platform": "douyin",
            "messages": [{"role": "buyer", "text": "165 110斤穿什么码"}],
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


def test_fake_decision_provider_answers_only_asked_questions():
    provider = FakeDecisionProvider({**router_answers("size_advice"), **qa_answers()})
    req = DecisionRequest(model="router-v1", state={}, questions=dict(ROUTER_QUESTIONS))
    resp = provider.decide(req)
    assert set(resp.answers) == set(ROUTER_QUESTIONS)
    assert resp.choice("intent").choice == "size_advice"
    down = FakeDecisionProvider({}, fail=DecisionError("down"))
    with pytest.raises(DecisionError):
        down.decide(req)


def test_config_yaml_matches_defaults():
    cfg = load_config(ROOT / "config" / "rodit.yaml")
    default = EngineConfig()
    assert cfg.routing == default.routing
    assert cfg.qa_gate.grounded_min == 0.8
    assert cfg.routing.auto_threshold[Intent.RETURN_POLICY] == 0.97
    assert Intent.COMPLAINT in cfg.routing.high_risk


def test_shop_config_layering():
    cfg = load_config(ROOT / "config" / "rodit.yaml")
    sc = resolve_shop_config(
        cfg, platform=Platform.PDD, category="女装", shop_override={"salutation": "您"}
    )
    assert sc.max_answer_chars == 100  # 平台默认
    assert sc.tone == "亲切"  # 类目默认
    assert sc.salutation == "您"  # 店铺覆盖


@pytest.mark.parametrize(
    "text,masked_token",
    [
        ("我手机 13812345678 你打给我", "[手机号]"),
        ("电话 +86 138-1234-5678", "[手机号]"),
        ("座机 0571-88886666", "[手机号]"),
        ("身份证 33010619900101123X", "[证件号]"),
        ("邮箱 abc.def@example.com", "[邮箱]"),
        ("卡号 6222021234567890123", "[卡号]"),
        ("寄到浙江省杭州市西湖区文三路 100 号 3 栋", "[地址]"),
    ],
)
def test_mask_pii(text, masked_token):
    out = mask_pii(text)
    assert masked_token in out
    assert contains_pii(text)


def test_mask_pii_keeps_normal_text():
    for text in ("165 110斤穿什么码", "快递单号 ZT7300000001 到哪了", "满 199 减 20 吗"):
        assert mask_pii(text) == text


def test_hash_buyer_id_is_stable_and_platform_scoped():
    a = hash_buyer_id("taobao", "tb_buyer_1")
    assert a == hash_buyer_id("taobao", "tb_buyer_1")
    assert a != hash_buyer_id("douyin", "tb_buyer_1")
    assert a.startswith("b_") and "tb_buyer_1" not in a


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


def test_demo_world_is_consistent():
    w = build_demo_world()
    assert w.now == DEFAULT_NOW
    card = w.item(SHOP_A, "i_555")
    assert isinstance(card, ItemCard)
    assert card.find_sku("k_1").props() == {"颜色": "黑色", "尺码": "M"}
    # 串店数据：同一个 item_id，不同店铺内容不同
    assert w.item(SHOP_B, "i_555").title != card.title
    # 每个订单的总额 = 明细之和；每个售后单都能找到订单
    for o in w.orders:
        assert o.total_amount == sum((ln.price * ln.quantity for ln in o.lines), Decimal("0"))
    order_keys = {(o.shop_id, o.order_id) for o in w.orders}
    for a in w.aftersales:
        assert (a.shop_id, a.order_id) in order_keys
        buyer = next(
            o.buyer_ref for o in w.orders if (o.shop_id, o.order_id) == (a.shop_id, a.order_id)
        )
        assert buyer == a.buyer_ref
    # 只有高频买家在 30 天内有 ≥ 3 单售后（R7）
    since = w.now - timedelta(days=30)
    counts: dict[str, int] = {}
    for a in w.aftersales:
        if a.shop_id == SHOP_A and a.created_at >= since:
            counts[a.buyer_ref] = counts.get(a.buyer_ref, 0) + 1
    assert {b for b, n in counts.items() if n >= 3} == {BUYER_RISKY}
    # 物流轨迹按时间升序
    for t in w.traces:
        times = [e.time for e in t.events]
        assert times == sorted(times)


def test_tool_gateway_isolation_and_failures():
    w = build_demo_world()
    gw = InMemoryToolGateway(w)
    assert gw.get_order(SHOP_A, "o_1001") is not None
    assert gw.get_order(SHOP_B, "o_1001") is None  # 不跨店
    assert [o.order_id for o in gw.list_recent_orders(SHOP_B, BUYER)] == ["o_9001"]
    assert gw.get_stock(SHOP_A, "i_555", "k_3").in_stock is False
    gw.fail_on.add("get_logistics")
    with pytest.raises(ToolUnavailable):
        gw.get_logistics(SHOP_A, "o_1002")


def test_fake_action_port_idempotent_and_updates_platform_state():
    w = build_demo_world()
    clock = FakeClock(w.now)
    gw = InMemoryToolGateway(w)
    port = FakeActionPort(gateway=gw, clock=clock)
    r1 = port.agree_refund(SHOP_A, "as_2001", operator="kf_02", idempotency_key="as_2001:agree:0")
    r2 = port.agree_refund(SHOP_A, "as_2001", operator="kf_02", idempotency_key="as_2001:agree:0")
    assert r1.ok and not r1.duplicate and r2.duplicate
    assert len(port.calls) == 1
    assert gw.get_aftersale(SHOP_A, "as_2001").status == AfterSaleStatus.REFUNDING
    pdd = FakeActionPort(Platform.PDD, supported={AfterSaleAction.AGREE_REFUND})
    with pytest.raises(ActionNotAllowed):
        pdd.refuse(SHOP_A, "as_2001", operator="kf_02", idempotency_key="k")


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
    ctx = make_context(
        "那件黑的有 L 吗",
        w,
        item_id="i_555",
        history=[(Role.BUYER, "在吗"), (Role.AGENT, "亲，在的")],
        with_recent_orders=True,
    )
    assert ctx.item.title == "法式碎花连衣裙"
    assert [t.role for t in ctx.history] == [Role.BUYER, Role.AGENT]
    assert ctx.recent_orders and all(o.buyer_ref == BUYER for o in ctx.recent_orders)


def test_fmt_time():
    assert fmt_time(datetime(2026, 9, 28, 10, 30, tzinfo=CST)) == "09-28 10:30"
    assert fmt_time(datetime(2026, 9, 28, 10, 30, tzinfo=CST), with_date=False) == "10:30"
