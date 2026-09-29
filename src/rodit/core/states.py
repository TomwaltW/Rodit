"""判断层请求的 state 构造（方案 §6.5 路由、§5.5 质检）。

路由和质检的 DecisionRequest.state 只能由这里生成：一是保证所有实现（incloud_llm / rules /
classifier / jev）读到同样的键；二是个人信息在这里统一最小化（§5.3）——只放对话文字（已脱敏）、
商品和订单的少量字段，不放 buyer_ref、昵称、手机号、地址、金额明细。
"""

from __future__ import annotations

from typing import Any

from rodit.core.models import ConversationContext, Draft, EvidenceBundle, Turn
from rodit.core.pii import mask_pii


def _messages(turns: list[Turn]) -> list[dict[str, str]]:
    return [{"role": str(t.role), "text": mask_pii(t.text)} for t in turns]


def recent_turns(ctx: ConversationContext, *, history_rounds: int = 5) -> list[Turn]:
    """最近 history_rounds 轮历史 + 当前消息，时间升序。一轮 = 一条买家消息及其之后的回复。"""
    history = list(ctx.history)
    # 从后往前数买家轮次，保留最近 history_rounds 个买家轮次及其回复
    buyer_seen = 0
    start = len(history)
    for i in range(len(history) - 1, -1, -1):
        start = i
        if history[i].role == "buyer":
            buyer_seen += 1
            if buyer_seen >= history_rounds:
                break
    kept = history[start:] if history_rounds > 0 else []
    return [*kept, ctx.current]


def build_router_state(ctx: ConversationContext, *, history_rounds: int = 5) -> dict[str, Any]:
    """§6.5 路由请求的 state。"""
    item_card = None
    if ctx.item is not None:
        item_card = {
            "item_id": ctx.item.item_id,
            "category": ctx.item.category,
            "title": ctx.item.title,
        }
    order_card = None
    if ctx.order is not None:
        order_card = {"order_id": ctx.order.order_id, "status": str(ctx.order.status)}
    return {
        "platform": str(ctx.platform),
        "messages": _messages(recent_turns(ctx, history_rounds=history_rounds)),
        "item_card": item_card,
        "order_card": order_card,
    }


def build_qa_state(
    ctx: ConversationContext,
    draft: Draft,
    evidence: EvidenceBundle,
    *,
    answer_text: str | None = None,
    history_rounds: int = 2,
) -> dict[str, Any]:
    """§5.5 第二道质检的 state：买家问题、最近对话、待检查的回答、依据列表。

    answer_text 传清洗后的文本（硬规则去掉 Markdown、截断之后），默认用 draft.text。
    """
    return {
        "platform": str(ctx.platform),
        "question": mask_pii(ctx.current.text),
        "dialogue": _messages(recent_turns(ctx, history_rounds=history_rounds)),
        "answer": answer_text if answer_text is not None else draft.text,
        "evidence_ids": list(draft.evidence_ids),
        "evidence": [
            {"id": e.id, "label": e.label, "content": mask_pii(e.content)}
            for e in evidence.evidences
        ],
    }
