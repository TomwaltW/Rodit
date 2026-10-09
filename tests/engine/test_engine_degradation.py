"""降级与留痕（方案 §12.2、§4.1 原则 6、§5.3 / §12.3 个人信息最小化、§12.1 / §12.4 耗时）。

handle() 对合法消息永不抛错；每条处理过的消息恰好一条 engine.result 审计；
审计和推荐记录里没有买家原文里的个人信息。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from engine_support import Kit, make_route, qa_pass

from rodit.core.audit import AuditAction
from rodit.core.config import DecisionConfig, QAGateConfig
from rodit.core.enums import DraftKind, Intent, Mode
from rodit.core.errors import DecisionError, KnowledgeUnavailable, LLMError, LLMTimeout
from rodit.core.models import Draft, EngineResult
from rodit.core.ports import LLMResponse
from rodit.decision import build_llm_provider
from rodit.engine import LATENCY_KEYS, RESULT_DETAIL_KEYS
from rodit.engine.engine import (
    REASON_ENGINE,
    REASON_GENERATOR,
    REASON_KNOWLEDGE,
    REASON_QA_ERROR,
    REASON_QA_FAILED,
    REASON_RETRIEVER,
    REASON_ROUTER,
    REASON_UNDERSTANDING,
)
from rodit.qa import QA_UNAVAILABLE, DefaultQualityGate
from rodit.testing.fakes import FakeLLMClient
from rodit.testing.fixtures import CONV, SHOP_A

SHIP = "亲，发货与快递：现货付款后 48 小时内发货"


def handle(kit: Kit, text: str = "什么时候发货", **kw: Any) -> EngineResult:
    engine = kw.pop("engine", None) or kit.engine()
    result = engine.handle(kit.msg(text, **kw))
    assert result is not None
    return result


def assert_single_result_event(kit: Kit, result: EngineResult) -> dict[str, Any]:
    [event] = kit.result_events(result.shop_id)
    assert tuple(event.detail) == RESULT_DETAIL_KEYS
    assert event.detail["degraded"] == result.degraded
    assert event.detail["latency_ms"] == result.latency_ms
    assert event.detail["missing"] == result.missing_knowledge
    assert event.detail["suggestions"] == len(result.suggestions)
    assert event.actor == "system"
    return event.detail


# ================================================================ 各环节故障（§12.2）


def test_router_raising_unexpectedly(kit: Kit) -> None:
    kit.router.error = RuntimeError("boom")
    result = handle(kit)
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_ROUTER
    assert result.degraded == ["router"]
    assert result.route is None
    detail = assert_single_result_event(kit, result)
    assert detail["intent"] is None and detail["action"] is None and detail["rule"] is None


def test_router_raising_decision_error_is_still_caught(kit: Kit) -> None:
    kit.router.error = DecisionError("judge down")
    result = handle(kit)
    assert result.degraded == ["router"]


def test_retriever_knowledge_unavailable(kit: Kit) -> None:
    kit.retriever.error = KnowledgeUnavailable("db down")
    result = handle(kit)
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_KNOWLEDGE
    assert result.degraded == ["knowledge"]
    assert kit.generator.calls == []
    assert_single_result_event(kit, result)


def test_retriever_unexpected_exception(kit: Kit) -> None:
    kit.retriever.error = ValueError("bad")
    result = handle(kit)
    assert result.no_suggestion_reason == REASON_RETRIEVER
    assert result.degraded == ["retriever"]


def test_generator_exception(kit: Kit) -> None:
    kit.generator.script = [LLMError("llm down")]
    result = handle(kit)
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_GENERATOR
    assert result.degraded == ["generator"]
    assert_single_result_event(kit, result)


def test_generator_exception_on_regeneration_keeps_passed(kit: Kit) -> None:
    faq = Draft(text="亲，现货付款后 48 小时内发货哦~", kind=DraftKind.FAQ, faq_id=2)
    bad = Draft(text="一定发", kind=DraftKind.LLM)
    kit.generator.script = [[faq, bad], RuntimeError("llm down")]
    result = handle(kit)
    assert [c.text for c in result.suggestions] == [faq.text]
    assert result.degraded == ["generator"]


def test_gate_exception_on_every_draft(kit: Kit) -> None:
    kit.gate.error = RuntimeError("qa down")
    result = handle(kit)
    assert result.suggestions == []  # 没验证过的文字不推荐
    assert result.no_suggestion_reason == REASON_QA_ERROR
    assert result.degraded == ["qa"]
    assert_single_result_event(kit, result)


def test_gate_exception_on_one_draft_keeps_others(kit: Kit) -> None:
    kit.generator.script = [
        [
            Draft(text="甲", kind=DraftKind.LLM, evidence_ids=["policy:ship_time"]),
            Draft(text="乙", kind=DraftKind.LLM, evidence_ids=["policy:ship_time"]),
        ]
    ]
    kit.gate.error = RuntimeError("qa flaky")
    kit.gate.error_on = lambda d: d.text == "甲"
    result = handle(kit)
    assert [c.text for c in result.suggestions] == ["乙"]
    assert result.degraded == ["qa"]


def test_qa_degraded_is_reported_and_hinted(kit: Kit) -> None:
    kit.gate.rule = lambda d: qa_pass(d, degraded=True)
    result = handle(kit)
    assert result.degraded == ["decision:qa"]
    assert result.suggestions[0].risk_hints == ["判断层质检暂不可用，只做了硬规则检查，请核实"]


@pytest.mark.parametrize(
    ("answer", "why"),
    [
        (LLMTimeout("大模型调用超时（1 秒）"), "大模型判断全部失败（1 次：超时 1）"),
        (
            LLMResponse(text='{"grounded": {"answer": "是"', finish_reason="length"),
            "大模型判断全部失败（1 次：截断 1）",
        ),
        (LLMResponse(text=""), "大模型判断全部失败（1 次：空回答 1）"),
    ],
    ids=["timeout", "truncated", "blank"],
)
def test_real_gate_with_an_unusable_judge_blocks_without_a_rewrite(
    kit: Kit, caplog: pytest.LogCaptureFixture, answer: Any, why: str
) -> None:
    """真的质检 + 真的大模型判断层，判断层每次都答不出（超时、被截断、空回答）：草稿不放行、
    原因照旧是"质检不可用"，不重写——判断层只调一次，生成也只调一次；原因进了日志。"""
    judge = FakeLLMClient(lambda request: answer)
    provider = build_llm_provider(DecisionConfig(), purpose="qa", llm=judge)
    gate = DefaultQualityGate(provider, QAGateConfig(), kit.audit, kit.clock)
    kit.generator.script = [
        [Draft(text=SHIP, kind=DraftKind.LLM, evidence_ids=["policy:ship_time"])],
        [Draft(text="不该调用", kind=DraftKind.LLM)],
    ]
    with caplog.at_level(logging.WARNING):
        result = handle(kit, engine=kit.engine(gate=gate))
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_QA_FAILED + QA_UNAVAILABLE
    assert result.degraded == ["decision:qa"]
    assert [c["retry_feedback"] for c in kit.generator.calls] == [None]
    assert len(judge.calls_for("qa")) == 1
    assert why in caplog.text
    detail = assert_single_result_event(kit, result)
    assert detail["qa_failed"] == 1 and detail["qa_reasons"] == [QA_UNAVAILABLE]


def test_route_degraded_is_reported_and_hinted(kit: Kit) -> None:
    kit.router.decide = make_route(Intent.SHIPPING_POLICY, degraded=True, provider="rules")
    result = handle(kit)
    assert result.degraded == ["decision"]
    assert result.suggestions[0].risk_hints == ["判断层暂不可用，问题类型按关键词规则判断，请核实"]


def test_tool_failure_during_understanding(kit: Kit) -> None:
    kit.tools.fail_on = {"list_recent_orders", "get_order"}
    result = handle(kit, order_id="o_1002")
    assert len(result.suggestions) == 1
    assert result.degraded == ["tool:order"]
    assert result.suggestions[0].risk_hints == ["实时数据暂时查不到（订单），请核实后再发送"]
    ctx = kit.router.calls[-1]
    assert ctx.order is None and ctx.recent_orders == []
    [understand] = kit.events(AuditAction.UNDERSTAND)
    assert understand.detail["degraded"] == ["tool:order"]


def test_shop_lookup_failure_still_stores_message(kit: Kit) -> None:
    class Broken:
        def __getattr__(self, name: str) -> Any:
            raise KnowledgeUnavailable("down")

    engine = kit.engine(knowledge=Broken())
    msg = kit.msg("什么时候发货")
    result = engine.handle(msg)
    assert result is not None
    assert result.no_suggestion_reason == REASON_KNOWLEDGE
    assert result.degraded == ["knowledge"]
    assert [m.msg_id for m in kit.conversations.recent_messages(SHOP_A, CONV)] == [msg.msg_id]
    assert engine.handle(msg) is None  # 仍然去重
    assert_single_result_event(kit, result)


def test_conversation_store_failure_does_not_block(kit: Kit) -> None:
    class Broken:
        def add_message(self, msg: Any) -> bool:
            raise RuntimeError("db down")

        def recent_messages(self, *a: Any, **kw: Any) -> list[Any]:
            raise RuntimeError("db down")

    result = handle(kit, engine=kit.engine(conversations=Broken()))
    assert [c.text for c in result.suggestions] == [SHIP]
    assert result.degraded == ["storage"]
    assert result.suggestions[0].risk_hints == ["会话记录暂时读不到，上下文可能不全，请核实"]


def test_suggestion_store_failure_still_returns_cards(kit: Kit) -> None:
    class Broken:
        def list(self, *a: Any, **kw: Any) -> list[Any]:
            return []

        def add(self, record: Any) -> Any:
            raise RuntimeError("db down")

    result = handle(kit, engine=kit.engine(suggestions=Broken()))
    assert [c.text for c in result.suggestions] == [SHIP]
    assert result.suggestions[0].suggestion_id is None
    assert result.degraded == ["storage"]


def test_expire_failure_is_degraded_not_fatal(kit: Kit) -> None:
    class Broken:
        def list(self, *a: Any, **kw: Any) -> list[Any]:
            raise RuntimeError("db down")

        def add(self, record: Any) -> Any:
            return record.model_copy(update={"id": 1})

    result = handle(kit, engine=kit.engine(suggestions=Broken()))
    assert len(result.suggestions) == 1
    assert result.degraded == ["storage"]


def test_gap_store_failure_is_degraded(kit: Kit) -> None:
    from rodit.core.models import EvidenceBundle

    class Broken:
        def add_gap(self, gap: Any) -> None:
            raise RuntimeError("db down")

    kit.retriever.bundles[Intent.SHIPPING_POLICY] = EvidenceBundle(missing=["policy:ship_time"])
    result = handle(kit, engine=kit.engine(gaps=Broken()))
    assert result.missing_knowledge == ["policy:ship_time"]
    assert result.degraded == ["storage"]


def test_audit_failure_never_raises(kit: Kit) -> None:
    class Broken:
        def record(self, event: Any) -> None:
            raise RuntimeError("audit down")

        def log_decision(self, entry: Any) -> None:
            raise RuntimeError("audit down")

    result = handle(kit, engine=kit.engine(audit=Broken()))
    assert len(result.suggestions) == 1
    assert result.degraded == ["audit"]


def test_understanding_unexpected_failure(kit: Kit, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = kit.engine()

    def boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(engine.builder, "build_with_notes", boom)
    result = handle(kit, engine=engine)
    assert result.no_suggestion_reason == REASON_UNDERSTANDING
    assert result.degraded == ["understanding"]
    assert_single_result_event(kit, result)


def test_understanding_knowledge_failure(kit: Kit, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = kit.engine()

    def boom(*a: Any, **kw: Any) -> Any:
        raise KnowledgeUnavailable("down")

    monkeypatch.setattr(engine.builder, "build_with_notes", boom)
    result = handle(kit, engine=engine)
    assert result.no_suggestion_reason == REASON_KNOWLEDGE
    assert result.degraded == ["knowledge"]


def test_any_unexpected_exception_becomes_result(kit: Kit) -> None:
    kit.router.decide = lambda ctx: "not a route"  # type: ignore[assignment,return-value]
    result = handle(kit)
    assert result.suggestions == []
    assert result.no_suggestion_reason == REASON_ENGINE
    assert result.degraded == ["engine"]
    assert set(result.latency_ms) == set(LATENCY_KEYS)
    assert_single_result_event(kit, result)


def test_garbage_drafts_do_not_escape(kit: Kit) -> None:
    kit.generator.script = [[None]]  # type: ignore[list-item]
    result = handle(kit)
    assert result.suggestions == []
    assert result.degraded  # 不管在哪一步兜住，都不抛
    assert_single_result_event(kit, result)


# ================================================================ 审计与耗时（§4.1 原则 6、§12.4）


def test_exactly_one_result_event_per_message(kit: Kit) -> None:
    engine = kit.engine()
    for text in ("在吗", "什么时候发货", "发什么快递"):
        engine.handle(kit.msg(text))
    assert len(kit.result_events()) == 3


def test_result_detail_contents(kit: Kit) -> None:
    kit.generator.script = [
        [Draft(text="亲，一定 48 小时发", kind=DraftKind.LLM)],
        [Draft(text=SHIP, kind=DraftKind.LLM, evidence_ids=["policy:ship_time"])],
    ]
    result = handle(kit)
    detail = assert_single_result_event(kit, result)
    assert detail["intent"] == "shipping_policy"
    assert detail["action"] == "answer"
    assert detail["rule"] == 6
    assert detail["suggestions"] == 1
    assert detail["hard_hits"] == ["absolute_term"]
    assert detail["qa_failed"] == 1
    assert detail["degraded"] == []
    assert detail["missing"] == []
    json.dumps(detail)  # 能直接写进 JSONB


def test_latency_keys_present_on_every_path(kit: Kit) -> None:
    engine = kit.engine()
    ok = engine.handle(kit.msg("什么时候发货"))
    kit.router.error = RuntimeError("boom")
    failed = engine.handle(kit.msg("什么时候发货"))
    for result in (ok, failed):
        assert result is not None
        assert tuple(result.latency_ms) == LATENCY_KEYS
        assert all(isinstance(v, int) and v >= 0 for v in result.latency_ms.values())
        assert result.latency_ms["total"] >= result.latency_ms["route"]
    assert failed is not None and failed.latency_ms["retrieve"] == 0


def test_stage_events_have_minimal_detail(kit: Kit) -> None:
    handle(kit, item_id="i_555", order_id="o_1002")
    [understand] = kit.events(AuditAction.UNDERSTAND)
    assert understand.detail["item_id"] == "i_555"
    assert understand.detail["order_id"] == "o_1002"
    assert understand.detail["mode"] == "assist"
    [route] = kit.events(AuditAction.ROUTE)
    assert route.detail["intent"] == "shipping_policy" and route.detail["provider"] == "stub"
    [retrieve] = kit.events(AuditAction.RETRIEVE)
    assert retrieve.detail["evidence_ids"] == ["policy:ship_time"]
    [qa] = kit.events(AuditAction.QA)
    assert qa.detail["checked"] == 1 and qa.detail["passed"] == 1
    [suggest] = kit.events(AuditAction.SUGGEST)
    assert suggest.detail["count"] == 1 and suggest.detail["status"] == "shown"


# ================================================================ 个人信息最小化（§5.3、§12.3）


PII_TEXT = "我手机 13812345678，寄到杭州市西湖区文三路 100 号 3 单元 501，什么时候发货"


def test_pii_never_in_audit_or_records(kit: Kit) -> None:
    from rodit.core.models import EvidenceBundle

    engine = kit.engine()
    result = handle(kit, PII_TEXT, engine=engine)
    [card] = result.suggestions
    record = kit.suggestions.get(SHOP_A, card.suggestion_id or -1)
    assert record is not None and record.question_text is not None
    assert "13812345678" not in record.question_text
    assert "文三路" not in record.question_text
    assert "[手机号]" in record.question_text and "[地址]" in record.question_text

    kit.retriever.bundles[Intent.SHIPPING_POLICY] = EvidenceBundle(missing=["policy:ship_time"])
    handle(kit, PII_TEXT, engine=engine)
    [gap] = kit.gaps.list_gaps(SHOP_A)
    assert gap.question_text is not None and "13812345678" not in gap.question_text

    dumped = json.dumps([e.detail for e in kit.audit.events], ensure_ascii=False)
    assert "13812345678" not in dumped
    assert "文三路" not in dumped
    assert "什么时候发货" not in dumped  # 审计里没有买家原文


def test_context_keeps_raw_text_for_downstream(kit: Kit) -> None:
    """ConversationContext 里是原文（订单号识别要用），脱敏由各步骤进提示词前做。"""
    handle(kit, "订单号 o_1002 到哪了")
    ctx = kit.router.calls[-1]
    assert ctx.current.text == "订单号 o_1002 到哪了"
    assert ctx.order is not None and ctx.order.order_id == "o_1002"


def test_result_mode_is_assist_for_default_shop(kit: Kit) -> None:
    assert handle(kit).mode == Mode.ASSIST
