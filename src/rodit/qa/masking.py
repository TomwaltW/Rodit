"""进判断层请求、质检原因前的个人信息最小化（方案 §5.3、§12.3）：core.pii.mask_pii 的薄包装。"""

from __future__ import annotations

from rodit.core.models import ConversationContext, EvidenceBundle
from rodit.core.pii import mask_pii


def mask_text(text: str) -> str:
    """遮掉手机号、证件号、微信 / QQ 号、收件地址等；没有个人信息时原样返回。"""
    return mask_pii(text) if text else text


def masked_context(ctx: ConversationContext) -> ConversationContext:
    """对话文字已遮个人信息的上下文副本（只用于构造判断层请求）。"""
    return ctx.model_copy(
        update={
            "current": ctx.current.model_copy(update={"text": mask_text(ctx.current.text)}),
            "history": [t.model_copy(update={"text": mask_text(t.text)}) for t in ctx.history],
        }
    )


def masked_evidence(evidence: EvidenceBundle) -> EvidenceBundle:
    """依据内容已遮个人信息的证据包副本（只用于构造判断层请求）。"""
    return evidence.model_copy(
        update={
            "evidences": [
                e.model_copy(update={"content": mask_text(e.content)}) for e in evidence.evidences
            ]
        }
    )
