"""售后凭证的图片理解（方案 §7.3.2 R5 "质量问题、破损、错漏发：需要凭证；客服查看后决定"、
§15 Phase 3 图片理解、§12.2 降级）。

R5 命中、售后单带凭证图片标识（AfterSaleInfo.evidence_images）、单元注入了 ImageUnderstanding 时，
审核助手识别最多 MAX_EVIDENCE_IMAGES 张凭证，在建议理由后面加一句摘要，例如
"图片理解（仅供参考）：3 张凭证中 2 张显示破损（裙摆撕裂，线头外露；外包装压痕明显）"。

- 只是参考：不改命中的规则、建议和下一个状态（仍是客服查看凭证后决定），不据此自动同意或拒绝；
- 并行识别（core.budget.submit_in_context，带上店铺范围），最多等 EVIDENCE_VISION_TIMEOUT_S 秒；
  识别失败、超时当作没有这张图的理解：全部失败时理由里写"凭证图片未能识别，请客服查看原图"，
  部分失败写"另有 n 张未能识别"（§12.2）；
- 置信度低于 MIN_CONFIDENCE 的结果算"看不清"，不计入"显示破损"；
- 个人信息（§5.3）：识别结果再过一次 mask_pii；审计只记张数和类型（EvidenceReading.audit），
  不记识别出的文字。
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from rodit.core.budget import submit_in_context
from rodit.core.enums import IMAGE_KIND_LABELS, ImageKind
from rodit.core.models import ImageFinding, ImageRef
from rodit.core.pii import mask_pii
from rodit.core.ports import ImageUnderstanding

logger = logging.getLogger(__name__)

MAX_EVIDENCE_IMAGES = 6
"""一张售后单最多识别几张凭证（视觉模型有成本和耗时，§14）。"""

EVIDENCE_VISION_TIMEOUT_S = 8.0
"""识别凭证最多等多久（秒）。审核助手不在买家消息的 5 秒预算里，但客服在等建议。"""

MIN_CONFIDENCE = 0.5
"""置信度低于它的识别结果算"看不清"。"""

SUMMARY_PREFIX = "图片理解（仅供参考）："
NOTE_FAILED = "凭证图片未能识别，请客服查看原图"

PROBLEM_ORDER: tuple[ImageKind, ...] = (ImageKind.DAMAGE, ImageKind.DEFECT, ImageKind.WRONG_ITEM)
"""R5 关心的问题类型（破损、质量瑕疵、错发漏发），摘要按这个顺序写。"""

_EXAMPLES = 2  # 每种问题最多举几张图的摘要
_EXAMPLE_CHARS = 20  # 每张图的摘要最多字数


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _sanitize(finding: ImageFinding, ref: str) -> ImageFinding:
    """防御：识别结果的文字再打一次码（§5.3），图片标识以请求的为准。"""
    extracted = mask_pii(finding.extracted_text) if finding.extracted_text else None
    return ImageFinding(
        ref=ref,
        kind=finding.kind,
        confidence=finding.confidence,
        summary=mask_pii(finding.summary) if finding.summary else "",
        extracted_text=extracted,
        provider=finding.provider,
    )


@dataclass(frozen=True)
class EvidenceReading:
    """一张售后单的凭证识别结果。"""

    total: int  # 售后单上的凭证图片张数
    findings: tuple[ImageFinding, ...]  # 识别成功的（已打码），按凭证顺序
    failed: int  # 识别失败、超时（含不合法的图片标识）的张数
    skipped: int  # 超过 MAX_EVIDENCE_IMAGES 没有识别的张数

    @property
    def analyzed(self) -> int:
        return self.total - self.skipped

    def summary(self) -> str:
        """接在 R5 建议理由后面的一句摘要（见模块说明）。"""
        if not self.findings:
            return SUMMARY_PREFIX + NOTE_FAILED
        clear = [f for f in self.findings if f.confidence >= MIN_CONFIDENCE]
        unclear = len(self.findings) - len(clear)
        head = f"{self.total} 张凭证"
        if self.skipped:
            head += f"（识别了前 {self.analyzed} 张）"
        groups: list[str] = []
        for kind in PROBLEM_ORDER:
            hits = [f for f in clear if f.kind == kind]
            if not hits:
                continue
            examples = "；".join(
                _clip(f.summary, _EXAMPLE_CHARS) for f in hits[:_EXAMPLES] if f.summary
            )
            label = IMAGE_KIND_LABELS[kind]
            groups.append(f"{len(hits)} 张显示{label}" + (f"（{examples}）" if examples else ""))
        if groups:
            main = f"{head}中 " + "，".join(groups)
        elif clear:
            main = f"{head}未识别出破损、质量瑕疵或错发漏发"
        else:
            main = head
        tails: list[str] = []
        if unclear:
            tails.append(f"{unclear} 张看不清")
        if self.failed:
            tails.append(f"另有 {self.failed} 张未能识别")
        return SUMMARY_PREFIX + "；".join([main, *tails])

    def audit(self) -> dict[str, Any]:
        """审计 detail：只记张数和类型，不记识别出的文字（§5.3）。"""
        return {
            "images": self.total,
            "analyzed": self.analyzed,
            "kinds": [str(f.kind) for f in self.findings],
            "failed": self.failed,
        }


def read_evidence(
    vision: ImageUnderstanding,
    shop_id: str,
    refs: Sequence[str],
    *,
    max_images: int = MAX_EVIDENCE_IMAGES,
    timeout_s: float = EVIDENCE_VISION_TIMEOUT_S,
) -> EvidenceReading:
    """并行识别售后单的凭证图片（只看 shop_id 这家店的，§9.2）。不抛错：失败按张数记下。"""
    unique = list(dict.fromkeys(r for r in refs if isinstance(r, str)))
    targets = unique[:max_images]
    skipped = len(unique) - len(targets)
    images: list[ImageRef | None] = []
    for ref in targets:
        try:
            images.append(ImageRef(ref=ref))
        except ValidationError:
            images.append(None)  # 不合法的图片标识：不交给视觉模型
    valid = [img for img in images if img is not None]
    if not valid:
        return EvidenceReading(total=len(unique), findings=(), failed=len(targets), skipped=skipped)
    pool = cf.ThreadPoolExecutor(max_workers=len(valid), thread_name_prefix="rodit-evidence")
    try:
        futures = {img.ref: submit_in_context(pool, vision.analyze, shop_id, img) for img in valid}
    finally:
        pool.shutdown(wait=False)  # 超时的识别留在后台跑完，结果不用
    cf.wait(futures.values(), timeout=timeout_s)
    findings: list[ImageFinding] = []
    failed = len(targets) - len(valid)
    for img in valid:
        future = futures[img.ref]
        if not future.done():
            logger.warning("凭证图片识别超时，当作没有识别结果")
            failed += 1
            continue
        error = future.exception()
        result = future.result() if error is None else None
        if not isinstance(result, ImageFinding):
            # 只记异常类型，不记异常消息（§12.3 日志脱敏）
            logger.warning("凭证图片识别失败（%s）", type(error).__name__ if error else "结果不对")
            failed += 1
            continue
        findings.append(_sanitize(result, img.ref))
    return EvidenceReading(
        total=len(unique), findings=tuple(findings), failed=failed, skipped=skipped
    )


__all__ = [
    "EVIDENCE_VISION_TIMEOUT_S",
    "MAX_EVIDENCE_IMAGES",
    "MIN_CONFIDENCE",
    "NOTE_FAILED",
    "SUMMARY_PREFIX",
    "EvidenceReading",
    "read_evidence",
]
