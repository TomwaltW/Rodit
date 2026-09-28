"""全自动模式（方案 §5.6、§5.7、§6.6 规则 6、§12.2、§12.3 AI 生成内容标识）。

条件全部满足才出 auto_reply（带前缀）；任何一条不满足都退回智能辅助，并转人工。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from engine_support import Kit, bundle_of, make_route, policy_evidence, qa_fail, qa_pass

from rodit.core.audit import AuditAction
from rodit.core.enums import DraftKind, Intent, Mode, RouteAction
from rodit.core.models import Draft, EngineResult
from rodit.testing.fixtures import BUYER_DOUYIN, SHOP_A, SHOP_D

PREFIX = "【智能客服】"
SHIP = "亲，发货与快递：现货付款后 48 小时内发货"


def auto_route(**kw: Any) -> Any:
    kw.setdefault("confidence", 0.97)
    kw.setdefault("auto_eligible", True)
    return make_route(Intent.SHIPPING_POLICY, **kw)


def enable_auto(kit: Kit, **overrides: Any) -> None:
    kit.set_shop_config(SHOP_A, auto_reply_enabled=True, **overrides)
    kit.router.decide = auto_route()


def run_auto(kit: Kit, text: str = "什么时候发货", **kw: Any) -> EngineResult:
    kw.setdefault("mode", Mode.AUTO)
    result = kit.engine().handle(kit.msg(text, **kw))
    assert result is not None
    return result


def test_auto_reply_happy_path_adds_prefix_once(kit: Kit) -> None:
    enable_auto(kit)
    result = run_auto(kit)
    assert result.mode == Mode.AUTO
    assert result.auto_reply is not None
    assert result.auto_reply.text == PREFIX + SHIP
    assert result.auto_reply.suggestion_id == result.suggestions[0].suggestion_id
    assert result.suggestions[0].text == SHIP  # 推荐卡片本身不带前缀
    assert result.transfer_to_human is False
    record = kit.suggestions.get(SHOP_A, result.auto_reply.suggestion_id or -1)
    assert record is not None and record.text == SHIP
    [event] = kit.events(AuditAction.AUTO_REPLY)
    assert event.detail == {
        "suggestion_id": result.auto_reply.suggestion_id,
        "intent": "shipping_policy",
        "kind": "llm",
        "chars": len(PREFIX + SHIP),
    }


def test_auto_reply_for_faq_answer(kit: Kit) -> None:
    enable_auto(kit)
    kit.generator.script = [[Draft(text="亲，默认发中通快递哦~", kind=DraftKind.FAQ, faq_id=1)]]
    result = run_auto(kit, "发什么快递")
    assert result.auto_reply is not None
    assert result.auto_reply.text == PREFIX + "亲，默认发中通快递哦~"


def test_prefix_length_boundary(kit: Kit) -> None:
    enable_auto(kit, max_answer_chars=len(PREFIX + SHIP))
    assert run_auto(kit).auto_reply is not None  # 刚好等于上限：可以


def _block_assist_mode(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    return {"mode": Mode.ASSIST}


def _block_shop_disabled(kit: Kit) -> dict[str, Any]:
    kit.router.decide = auto_route()
    return {}


def _block_not_eligible(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.router.decide = auto_route(auto_eligible=False)
    return {}


def _block_route_degraded(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.router.decide = auto_route(degraded=True, provider="rules")
    return {}


def _block_tool_degraded(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.tools.fail_on = {"list_recent_orders"}
    return {}


def _block_bundle_unavailable(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.retriever.bundles[Intent.SHIPPING_POLICY] = bundle_of(
        policy_evidence(), unavailable=["tool:order"]
    )
    return {}


def _block_prefix_too_long(kit: Kit) -> dict[str, Any]:
    enable_auto(kit, max_answer_chars=len(PREFIX + SHIP) - 1)
    return {}


def _block_qa_failed(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.gate.rule = lambda d: qa_fail(d, ["依据不足"], rule=None)
    return {}


def _block_medium_confidence(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.gate.rule = lambda d: qa_pass(d, grounded=0.85)
    return {}


def _block_qa_degraded(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.gate.rule = lambda d: qa_pass(d, degraded=True)
    return {}


def _block_risk_hint(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.router.decide = auto_route(risk_hints=["涉及时效承诺，请确认"])
    return {}


def _block_route_alert(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.router.decide = auto_route(alerts=["买家情绪激动"])
    return {}


def _block_unresolved(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.conversations.add_message(kit.msg("没听懂"))
    return {"text": "你答非所问"}


def _block_split(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.router.decide = auto_route(
        action=RouteAction.SPLIT_MULTI,
        rule=5,
        parts=[(Intent.SHIPPING_POLICY, 0.97)],
    )
    return {}


def _block_dropped_faq(kit: Kit) -> dict[str, Any]:
    enable_auto(kit)
    kit.generator.script = [
        [
            Draft(text="亲，全网最低", kind=DraftKind.FAQ, faq_id=9),
            Draft(text=SHIP, kind=DraftKind.LLM, evidence_ids=["policy:ship_time"]),
        ]
    ]
    kit.gate.rule = lambda d: (
        qa_fail(d, ["商家禁用词"], rule="banned_word") if d.kind == DraftKind.FAQ else None
    )
    return {}


BLOCKERS: dict[str, Callable[[Kit], dict[str, Any]]] = {
    "shop_disabled": _block_shop_disabled,
    "not_eligible": _block_not_eligible,
    "route_degraded": _block_route_degraded,
    "tool_degraded": _block_tool_degraded,
    "bundle_unavailable": _block_bundle_unavailable,
    "prefix_too_long": _block_prefix_too_long,
    "qa_failed": _block_qa_failed,
    "medium_confidence": _block_medium_confidence,
    "qa_degraded": _block_qa_degraded,
    "risk_hint": _block_risk_hint,
    "route_alert": _block_route_alert,
    "unresolved": _block_unresolved,
    "split": _block_split,
    "dropped_faq": _block_dropped_faq,
}


@pytest.mark.parametrize("name", sorted(BLOCKERS))
def test_auto_reply_blocked(kit: Kit, name: str) -> None:
    overrides = BLOCKERS[name](kit)
    result = run_auto(kit, **overrides)
    assert result.auto_reply is None
    assert kit.events(AuditAction.AUTO_REPLY) == []
    for card in result.suggestions:
        assert not card.text.startswith(PREFIX)  # 前缀只在 auto_reply 上
    if result.mode == Mode.AUTO:
        # 全自动模式下不能自动发送：退回智能辅助，转人工由客服接管（§5.7、§12.2）
        assert result.transfer_to_human is True
    else:
        assert name == "shop_disabled"
        assert result.mode == Mode.ASSIST
        assert result.transfer_to_human is False


def test_assist_mode_message_never_auto_replies(kit: Kit) -> None:
    overrides = _block_assist_mode(kit)
    result = run_auto(kit, **overrides)
    assert result.mode == Mode.ASSIST
    assert result.auto_reply is None
    assert result.transfer_to_human is False
    assert len(result.suggestions) == 1


def test_blocked_auto_still_gives_assist_suggestions(kit: Kit) -> None:
    _block_not_eligible(kit)
    result = run_auto(kit)
    assert [c.text for c in result.suggestions] == [SHIP]
    assert result.transfer_to_human is True


def test_auto_mode_failure_transfers_to_human(kit: Kit) -> None:
    enable_auto(kit)
    kit.router.error = RuntimeError("boom")
    result = run_auto(kit)
    assert result.mode == Mode.AUTO
    assert result.suggestions == []
    assert result.transfer_to_human is True


def test_shadow_shop_never_auto_replies(kit: Kit) -> None:
    kit.set_shop_config(SHOP_D, auto_reply_enabled=True)
    kit.router.decide = auto_route()
    result = kit.engine().handle(
        kit.msg("什么时候发货", shop_id=SHOP_D, buyer_ref=BUYER_DOUYIN, mode=Mode.AUTO)
    )
    assert result is not None
    assert result.mode == Mode.SHADOW
    assert result.auto_reply is None
    assert result.transfer_to_human is False
