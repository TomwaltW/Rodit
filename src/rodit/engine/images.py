"""买家图片理解（方案 §15 Phase 3 "图片理解：破损照片、尺码截图"、§5.3、§12.2）。

问答引擎在会话理解之后、路由之前调 ImageReader.read，把当前买家消息里的图片读成文字：

- 哪些图片：当前买家消息（合并连发之后的 Turn，§5.3）里每条消息的图片。会话记录
  （StoredMessage）没有图片字段，所以消息一到就按 msg_id 把图片标识记进短时缓存（core.ports.Cache，
  按店铺分开，取消订购随单元缓存一并删除，§8.4），同一 Turn 里后到的文字消息也能看到前面发的图；
  没有缓存时只看本条消息自带的图片。最多看 MAX_IMAGES 张，其余提醒客服自己看。
- 怎么读：并行调用 ImageUnderstanding.analyze（core.budget.submit_in_context，在引擎的
  background_scope 里记进 BackgroundWork），最多等 VISION_BUDGET_S 秒（不超过整条消息的截止时间）。
  识别失败、超时一律当作没有这张图的理解：记降级 "vision"，提醒客服自己看图（§12.2），照常出推荐。
  同一张图的识别结果短时缓存（超时被丢下的识别做完时也记进去），同一 Turn 后到的文字消息不再
  重复识别；识别失败的不记，下一条消息再试。
  没注入 ImageUnderstanding 时不调用，只提醒客服买家发了图片（不算降级）。
- 识别结果写进 ConversationContext.image_findings（按图片顺序），并拼成一段文字描述接在当前买家消息
  后面，例如"【买家发来图片：破损（外包装一角压扁…）】"：路由（判断层、关键词规则）、检索、生成、
  质检看到的是同一份。描述用图片类型的中文名（core.enums.IMAGE_KIND_LABELS）：破损、质量瑕疵、
  错发漏发命中质量问题的关键词，尺码截图命中尺码；尺码截图另带图中文字（"身高 165cm 体重 110 斤"），
  尺码建议据此回答。置信度低于 ROUTING_MIN_CONFIDENCE 的结果不进描述（算"未能识别"），只给客服看。
- 历史轮次（§5.3 最近 5 轮）里缓存还记得的图片同样补上描述，只用缓存里的识别结果，不再调用。
- 个人信息（§5.3）：图片标识是平台给的不透明 ID（接入层已经挡掉带签名的网址）；识别出的文字
  再过一次 mask_pii（实现本应已经打码）。图中文字只有尺码截图的进描述——物流截图、面单上有手机号、
  地址，不往后传。物流 / 订单截图、其他图片（PERSONAL_KINDS）的内容本来就可能是个人信息
  （收件人姓名、聊天记录……，mask_pii 遮不掉姓名），描述里只写类型，摘要也不写：描述会进路由、
  生成、质检的提示词和推荐记录（question_text，还会进知识候选）；摘要只在给客服的提醒里。
  审计只记张数和类型（describe_for_audit），不记识别出的文字。
- 给客服的提醒："买家发来 1 张图片：破损（93%）——外包装一角压扁…；仅供参考，以凭证为准"。
  图片理解只是参考，不据此承诺任何处理结果（§7.3.2 R5 由客服看凭证决定）；有提醒的消息不出全自动
  回复（引擎的全自动条件，§5.6），全自动模式下因此转人工。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any

from pydantic import ValidationError

from rodit.core import budget
from rodit.core.clock import Clock
from rodit.core.enums import IMAGE_KIND_LABELS, ImageKind, Role
from rodit.core.models import ConversationContext, ImageFinding, ImageRef, StandardMessage, Turn
from rodit.core.pii import mask_pii
from rodit.core.ports import Cache, ImageUnderstanding
from rodit.engine.concurrency import Outcome, run_ordered

logger = logging.getLogger(__name__)

DEGRADED_VISION = "vision"
"""图片识别失败或超时（EngineResult.degraded）。"""

MAX_IMAGES = 3
"""当前买家消息里最多识别几张图（视觉模型有成本和耗时，§12.1、§14）。"""

VISION_BUDGET_S = 1.5
"""图片识别最多等多久（秒）；同时不超过整条消息的截止时间（§12.1 端到端 ≤ 5 秒）。"""

MEMORY_TTL_S = 1800
"""消息里的图片标识、图片的识别结果在缓存里记多久（秒）：覆盖合并连发和最近几轮对话。"""

ROUTING_MIN_CONFIDENCE = 0.5
"""识别结果的置信度低于它时不进描述（不影响路由和回答），只在提醒里给客服看。"""

PROBLEM_KINDS = frozenset({ImageKind.DAMAGE, ImageKind.DEFECT, ImageKind.WRONG_ITEM})
"""质量问题类的图片（§7.3.1 收到的东西坏了、§7.3.2 R5）。"""

PERSONAL_KINDS = frozenset({ImageKind.LOGISTICS_SCREENSHOT, ImageKind.OTHER})
"""内容本来就可能是个人信息的图片（物流 / 订单页面上的收件人、聊天截图……）：描述只写类型，
不写摘要和图中文字；摘要只在给客服的提醒里（§5.3，见模块说明）。"""

DESCRIPTION_CHARS = 60  # 描述里每张图的摘要 / 图中文字最多字数
ALERT_CHARS = 40  # 提醒里每张图的摘要最多字数

ALERT_VISION_FAILED = "图片没能识别，请客服自己查看图片"
ALERT_NO_VISION = "买家发来 {n} 张图片，没有开通图片理解，请客服自己查看图片"
ALERT_SKIPPED = "买家发来 {n} 张图片，只识别了前 {k} 张，其余请客服自己查看"

_MSG_KEY = "vision_msg|{conversation_id}|{msg_id}"
_REF_KEY = "vision_ref|{ref}"


@dataclass(frozen=True)
class ImageReading:
    """一条买家消息的图片识别结果（引擎据此记降级、提醒客服、写审计）。"""

    total: int = 0  # 当前买家消息（合并后）里的图片张数
    findings: tuple[ImageFinding, ...] = ()  # 识别成功的（已打码），按图片顺序
    failed: int = 0  # 识别失败或超时的张数
    skipped: int = 0  # 超过 MAX_IMAGES 没有识别的张数
    vision: bool = False  # 单元有没有图片理解
    description: str = ""  # 接在当前买家消息后面的描述
    alerts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def degraded(self) -> bool:
        return self.failed > 0


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _mask(text: str | None) -> str:
    return mask_pii(text) if text else ""


def sanitize(finding: ImageFinding, ref: str) -> ImageFinding:
    """防御：识别结果的文字再打一次码（§5.3），图片标识以请求的为准。"""
    extracted = _mask(finding.extracted_text) or None
    return ImageFinding(
        ref=ref,
        kind=finding.kind,
        confidence=finding.confidence,
        summary=_mask(finding.summary),
        extracted_text=extracted,
        provider=finding.provider,
    )


def _finding_text(finding: ImageFinding) -> str:
    """描述里的一张图：类型（摘要；图中文字）。图中文字只有尺码截图的才写；物流截图、其他图片
    只写类型（PERSONAL_KINDS，见模块说明）。"""
    label = IMAGE_KIND_LABELS[finding.kind]
    if finding.kind in PERSONAL_KINDS:
        return label
    parts: list[str] = []
    summary = _clip(_mask(finding.summary), DESCRIPTION_CHARS)
    if summary:
        parts.append(summary)
    if finding.kind == ImageKind.SIZE_REFERENCE and finding.extracted_text:
        parts.append(f"图中文字：{_clip(_mask(finding.extracted_text), DESCRIPTION_CHARS)}")
    return f"{label}（{'；'.join(parts)}）" if parts else label


def describe(total: int, findings: Sequence[ImageFinding], *, vision: bool) -> str:
    """当前买家消息里图片的文字描述（接在买家消息后面，进路由、检索、生成、质检）。"""
    if total <= 0:
        return ""
    if not vision:
        return f"【买家发来 {total} 张图片】"
    usable = [f for f in findings if f.confidence >= ROUTING_MIN_CONFIDENCE]
    rest = total - len(usable)
    if not usable:
        return f"【买家发来 {total} 张图片，未能识别】"
    parts = [_finding_text(f) for f in usable]
    if total == 1:
        return f"【买家发来图片：{parts[0]}】"
    if rest:
        parts.append(f"另有 {rest} 张未能识别")
    return f"【买家发来 {total} 张图片：{'；'.join(parts)}】"


def summary_alert(total: int, findings: Sequence[ImageFinding]) -> str | None:
    """给客服的图片摘要，如"买家发来 1 张图片：破损（93%）——外包装一角压扁…；
    仅供参考，以凭证为准"。"""
    if not findings:
        return None
    parts: list[str] = []
    for f in findings:
        head = f"{IMAGE_KIND_LABELS[f.kind]}（{round(f.confidence * 100)}%）"
        summary = _clip(_mask(f.summary), ALERT_CHARS)
        parts.append(f"{head}——{summary}" if summary else head)
    tail = "以凭证为准" if any(f.kind in PROBLEM_KINDS for f in findings) else "请以图片为准"
    return f"买家发来 {total} 张图片：{'；'.join(parts)}；仅供参考，{tail}"


def describe_for_audit(reading: ImageReading) -> dict[str, Any]:
    """审计 detail 里的图片信息：只有张数和类型，不记识别出的文字（§5.3）。"""
    return {
        "images": reading.total,
        "image_kinds": [str(f.kind) for f in reading.findings],
        "images_failed": reading.failed,
        "images_skipped": reading.skipped,
    }


def _join(text: str, description: str) -> str:
    return f"{text} {description}".strip() if description else text


class ImageReader:
    """问答引擎的图片识别步骤（见模块说明）。线程安全（状态都在 Cache 里）。

    - vision：图片理解的实现，为空时不调用；
    - cache：记消息里的图片标识和识别结果，为空时只看本条消息自带的图片、不复用识别结果；
    - monotonic：时间预算用的单调时钟，和引擎的截止时间同一个。
    """

    def __init__(
        self,
        vision: ImageUnderstanding | None,
        cache: Cache | None,
        clock: Clock,
        *,
        max_images: int = MAX_IMAGES,
        budget_s: float = VISION_BUDGET_S,
        memory_ttl_s: float = MEMORY_TTL_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_images < 1:
            raise ValueError("max_images 至少为 1")
        if budget_s <= 0:
            raise ValueError("图片识别的时间预算必须大于 0")
        self._vision = vision
        self._cache = cache
        self._clock = clock
        self._max_images = max_images
        self._budget_s = budget_s
        self._ttl = timedelta(seconds=memory_ttl_s)
        self._monotonic = monotonic

    @property
    def enabled(self) -> bool:
        return self._vision is not None

    # ------------------------------------------------------------ 消息里的图片

    def remember(self, msg: StandardMessage) -> None:
        """消息到达时记下它的图片标识（同一 Turn 里后到的消息要用）。缓存故障只记日志。"""
        if not msg.images or self._cache is None:
            return
        refs = list(dict.fromkeys(i.ref for i in msg.images))
        key = _MSG_KEY.format(conversation_id=msg.conversation_id, msg_id=msg.msg_id)
        self._cache_set(msg.shop_id, key, refs)

    def forget(self, shop_id: str, conversation_id: str, msg_id: str) -> None:
        """撤回的消息（§5.3）：不再记它的图片。"""
        if self._cache is None:
            return
        key = _MSG_KEY.format(conversation_id=conversation_id, msg_id=msg_id)
        if self._cache_get(shop_id, key) is not None:
            self._cache_set(shop_id, key, [])

    def _refs_of(self, shop_id: str, conversation_id: str, msg_id: str) -> list[str]:
        key = _MSG_KEY.format(conversation_id=conversation_id, msg_id=msg_id)
        value = self._cache_get(shop_id, key)
        if not isinstance(value, list):
            return []
        return [r for r in value if isinstance(r, str) and r]

    def _turn_refs(
        self,
        shop_id: str,
        conversation_id: str,
        msg_ids: Sequence[str],
        incoming: StandardMessage | None,
    ) -> list[str]:
        refs: list[str] = []
        for msg_id in msg_ids:
            found = self._refs_of(shop_id, conversation_id, msg_id)
            if not found and incoming is not None and incoming.msg_id == msg_id:
                found = [i.ref for i in incoming.images]  # 缓存不可用：本条消息自带的
            for ref in found:
                if ref not in refs:
                    refs.append(ref)
        return refs

    # ------------------------------------------------------------ 识别结果缓存

    def _cached_finding(self, shop_id: str, ref: str) -> ImageFinding | None:
        value = self._cache_get(shop_id, _REF_KEY.format(ref=ref))
        if not isinstance(value, dict):
            return None
        try:
            finding = ImageFinding.model_validate(value)
        except ValidationError:
            return None
        return finding if finding.ref == ref else None

    def _store_finding(self, shop_id: str, finding: ImageFinding) -> None:
        self._cache_set(shop_id, _REF_KEY.format(ref=finding.ref), finding.model_dump(mode="json"))

    def _cache_get(self, shop_id: str, key: str) -> object | None:
        if self._cache is None:
            return None
        try:
            hit = self._cache.get(shop_id, key)
        except Exception as exc:  # noqa: BLE001 — 缓存故障：当作没记过
            logger.warning("图片缓存读取失败（%s）", type(exc).__name__)
            return None
        if hit is None:
            return None
        value, written_at = hit
        if self._clock.now() - written_at > self._ttl:
            return None
        return value

    def _cache_set(self, shop_id: str, key: str, value: object) -> None:
        if self._cache is None:
            return
        try:
            self._cache.set(shop_id, key, value)
        except Exception as exc:  # noqa: BLE001 — 缓存故障不影响出推荐
            logger.warning("图片缓存写入失败（%s）", type(exc).__name__)

    # ------------------------------------------------------------ 识别

    def _analyze(self, shop_id: str, ref: str) -> ImageFinding:
        """在工作线程里跑：识别一张图并记进缓存。超时被丢下的任务做完时照样记进缓存
        （同一 Turn 后到的消息用得上；任务记在 BackgroundWork 里，取消订购先等它做完再删缓存）。"""
        assert self._vision is not None
        finding = self._vision.analyze(shop_id, ImageRef(ref=ref))
        if not isinstance(finding, ImageFinding):
            raise TypeError("图片理解返回的不是 ImageFinding")
        finding = sanitize(finding, ref)
        self._store_finding(shop_id, finding)
        return finding

    def _analyze_all(self, shop_id: str, refs: list[str]) -> tuple[dict[str, ImageFinding], int]:
        """识别 refs（先查缓存），返回 (ref → 识别结果, 失败或超时的张数)。"""
        results: dict[str, ImageFinding] = {}
        todo: list[str] = []
        for ref in refs:
            cached = self._cached_finding(shop_id, ref)
            if cached is not None:
                results[ref] = sanitize(cached, ref)
            else:
                todo.append(ref)
        if not todo:
            return results, 0
        if budget.expired():  # 整条消息的时间已经用完：不再识别（§12.2）
            return results, len(todo)
        tasks = [partial(self._analyze, shop_id, ref) for ref in todo]
        with budget.deadline_scope(self._budget_s, now=self._monotonic):
            outcomes: list[Outcome[ImageFinding]] = run_ordered(
                tasks, grace_s=0.0, name="rodit-vision"
            )
        failed = 0
        for ref, outcome in zip(todo, outcomes, strict=True):
            if outcome.finished and outcome.error is None and outcome.value is not None:
                results[ref] = outcome.value
                continue
            failed += 1
            if not outcome.finished:
                logger.warning("图片识别超时，当作没有识别结果")
            else:
                # 只记异常类型，不记异常消息（§12.3 日志脱敏）
                logger.warning("图片识别失败（%s）", type(outcome.error).__name__)
        return results, failed

    # ------------------------------------------------------------ 入口

    def read(
        self, ctx: ConversationContext, incoming: StandardMessage | None = None
    ) -> tuple[ConversationContext, ImageReading]:
        """识别当前买家消息里的图片，返回 (补上图片描述和识别结果的上下文, 识别结果)。

        incoming 是交给引擎的那条消息（缓存不可用时用它自带的图片）。没有图片时上下文原样返回
        （历史轮次里记得的图片照样补上描述）。识别失败不抛错（见模块说明）。"""
        history = self._annotate_history(ctx)
        refs = self._turn_refs(ctx.shop_id, ctx.conversation_id, ctx.current.msg_ids, incoming)
        if not refs:
            if history is None:
                return ctx, ImageReading(vision=self.enabled)
            return ctx.model_copy(update={"history": history}), ImageReading(vision=self.enabled)

        total = len(refs)
        targets = refs[: self._max_images]
        skipped = total - len(targets)
        findings: list[ImageFinding] = []
        failed = 0
        alerts: list[str] = []
        if self._vision is None:
            alerts.append(ALERT_NO_VISION.format(n=total))
        else:
            results, failed = self._analyze_all(ctx.shop_id, targets)
            findings = [results[r] for r in targets if r in results]
            summary = summary_alert(total, findings)
            if summary is not None:
                alerts.append(summary)
            if failed:
                alerts.append(ALERT_VISION_FAILED)
            if skipped:
                alerts.append(ALERT_SKIPPED.format(n=total, k=len(targets)))
        description = describe(total, findings, vision=self.enabled)
        current = ctx.current.model_copy(update={"text": _join(ctx.current.text, description)})
        update: dict[str, Any] = {"current": current, "image_findings": findings}
        if history is not None:
            update["history"] = history
        reading = ImageReading(
            total=total,
            findings=tuple(findings),
            failed=failed,
            skipped=skipped,
            vision=self.enabled,
            description=description,
            alerts=tuple(alerts),
        )
        return ctx.model_copy(update=update), reading

    def _annotate_history(self, ctx: ConversationContext) -> list[Turn] | None:
        """历史买家轮次里缓存还记得的图片补上描述（只用缓存里的识别结果）；没有改动返回 None。"""
        if self._cache is None:
            return None
        out: list[Turn] = []
        changed = False
        for turn in ctx.history:
            if turn.role != Role.BUYER:
                out.append(turn)
                continue
            refs = self._turn_refs(ctx.shop_id, ctx.conversation_id, turn.msg_ids, None)
            if not refs:
                out.append(turn)
                continue
            findings = [
                f
                for f in (self._cached_finding(ctx.shop_id, r) for r in refs[: self._max_images])
                if f is not None
            ]
            findings = [sanitize(f, f.ref) for f in findings]
            description = describe(len(refs), findings, vision=self.enabled)
            out.append(turn.model_copy(update={"text": _join(turn.text, description)}))
            changed = True
        return out if changed else None


__all__ = [
    "ALERT_NO_VISION",
    "ALERT_SKIPPED",
    "ALERT_VISION_FAILED",
    "DEGRADED_VISION",
    "MAX_IMAGES",
    "MEMORY_TTL_S",
    "PERSONAL_KINDS",
    "PROBLEM_KINDS",
    "ROUTING_MIN_CONFIDENCE",
    "VISION_BUDGET_S",
    "ImageReader",
    "ImageReading",
    "describe",
    "describe_for_audit",
    "sanitize",
    "summary_alert",
]
