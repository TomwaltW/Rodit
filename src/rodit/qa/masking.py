"""进判断层请求、质检原因前的个人信息最小化（方案 §5.3、§12.3）：core.pii.mask_pii 的本地包装。

- 没有要遮的内容时返回原文：mask_pii 会顺带做 NFKC 归一，把中文全角标点改成半角，
  判断层请求里没必要改动；
- 地址关键字（路、小区……）、门牌数字、楼栋单位之间有空格时（"文三路 100 号""幸福小区 3 栋"），
  core.pii 的地址识别会把空格当成引导词、只遮住后半截，路名 / 小区名漏出来。
  这里先去掉这类空格再遮（问题已写进回执，core 修复后可以去掉这一步）。
- core.states.build_qa_state 内部用 mask_pii，有同样的问题：进判断层前先用
  masked_context / masked_evidence 把对话和依据遮一遍再交给它。
"""

from __future__ import annotations

import re

from rodit.core.models import ConversationContext, EvidenceBundle
from rodit.core.pii import contains_pii, mask_pii

_ADDRESS_GAP = re.compile(
    r"(?<=路|街|巷|弄|道|同|区|园|寓|厦|村|苑|场)\s+(?=\d)"
    r"|(?<=\d)\s+(?=栋|幢|号|座|单元|层|楼|室)"
)


def mask_text(text: str) -> str:
    """遮掉手机号、证件号、微信 / QQ 号、收件地址等；没有个人信息时原样返回。"""
    if not text:
        return text
    prepared = _ADDRESS_GAP.sub("", text)
    if not contains_pii(prepared):
        return text
    return mask_pii(prepared)


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
