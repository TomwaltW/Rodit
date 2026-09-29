"""平台商品同步（方案 §5.2.3 冷启动第 2 步"商品自动同步"：拉商品标题、属性、SKU；详情图做 OCR
提取文字；§5.2.1 商品知识"商品变更时同步"）。只依赖 rodit.core。

- 新商品建卡片；已有卡片只更新平台维护的字段（标题、类目、属性、SKU、尺码表、详情文字），
  商家填写的字段（size_advice、care、usage、forbidden_claims）原样保留；内容没变的卡片不写，
  updated_at 不变（重复同步是空操作）。
- 详情图逐张交给 ImageTextExtractor（在平台云内识别）。合并规则（确定、可重复）：
  detail_text = 平台详情文字 + 每张图识别出的文字，各占一段，用换行分开（检索按句切段时各成一句）；
  图片文字按图片顺序，去掉空行和首尾空白；和平台详情文字重复、或和前面某张图完全相同的不再加。
  每次同步都从平台详情文字重新合成，不在已有的 detail_text 上追加，所以不会越同步越长。
- 某张图识别失败：记进报告（ocr_failures），不影响同步；这时如果本地卡片的详情文字仍以平台详情文字
  开头（平台文字没变），保留本地的详情文字（里面有上次识别出的图片文字），不因为一次识别失败丢字，
  也不来回改卡片；平台文字变了就用新的平台文字 + 这次识别出的文字。
- 已下架（on_sale=False）的商品照常同步（买家可能还会问买过的商品），列进报告 off_sale；
  本地有卡片、平台上已经没有的商品只列进报告 missing_on_platform，不删卡片（§5.2.3 不自动删知识）。
- 纵深防御（§9.1、§9.2）：商品接口返回的商品必须属于这家店铺、本单元的平台，否则不写入，
  只计数（rejected，报告里不列别的店铺的商品 ID）。
- 商品接口故障（ToolUnavailable）照常抛出；没有接入商品接口或 OCR 时抛 ItemSyncUnavailable。
- 审计 knowledge.item_sync：只记各类条数，不记商品内容。

调用方负责和取消订购互斥（app.console 在 processing(shop_id) 里调用），并在写之前确认店铺还在。
"""

from __future__ import annotations

import logging
import threading
import unicodedata
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from rodit.core.audit import AuditAction
from rodit.core.clock import Clock
from rodit.core.enums import Platform
from rodit.core.errors import RoditError, TenantIsolationError
from rodit.core.models import AuditEvent, ItemCard, PlatformItem, strip_control_chars
from rodit.core.ports import AuditSink, ImageTextExtractor, ItemCatalog, KnowledgeRepository

logger = logging.getLogger(__name__)

MAX_OCR_CHARS = 1000
"""一张详情图最多取这么多字（详情图偶尔是整页长图；证据还有长度预算，§14）。"""

MSG_SYNC_UNAVAILABLE = "本单元没有接入平台商品接口或详情图识别，商品同步不可用（§5.2.3）"


class ItemSyncUnavailable(RoditError):
    """本单元没有注入平台商品接口（ItemCatalog）或详情图 OCR（ImageTextExtractor）：商品同步不可用。
    接口层映射为 503（和接口故障一样，商家稍后再试或联系我们接入）。"""

    def __init__(self, message: str = MSG_SYNC_UNAVAILABLE) -> None:
        super().__init__(message)


class SyncReport(BaseModel):
    """一次同步的结果。各列表是商品 ID（按 item_id 升序）；rejected 只有条数。"""

    model_config = ConfigDict(extra="forbid")

    shop_id: str
    scope: Literal["shop", "item"]
    item_id: str | None = None  # 只同步一个商品时
    created: list[str] = []
    updated: list[str] = []
    unchanged: list[str] = []
    off_sale: list[str] = []  # 平台上已下架（卡片照常同步，不删）
    missing_on_platform: list[str] = []  # 本地有卡片、平台上没有了（不删）
    ocr_failed_items: list[str] = []  # 有详情图识别失败的商品
    ocr_failures: int = 0  # 识别失败的图片张数
    rejected: int = 0  # 不属于这家店铺 / 本单元平台、没有写入的商品数
    synced_at: AwareDatetime

    def counts(self) -> dict[str, int]:
        return {
            "created": len(self.created),
            "updated": len(self.updated),
            "unchanged": len(self.unchanged),
            "off_sale": len(self.off_sale),
            "missing_on_platform": len(self.missing_on_platform),
            "ocr_failed_items": len(self.ocr_failed_items),
            "ocr_failures": self.ocr_failures,
            "rejected": self.rejected,
        }


def _clean_ocr(text: object) -> str:
    """识别结果：去掉控制字符、空行和每行首尾空白；过长截断。"""
    if not isinstance(text, str):
        return ""
    lines = [line.strip() for line in strip_control_chars(text).splitlines()]
    return "\n".join(line for line in lines if line)[:MAX_OCR_CHARS].strip()


def _key(text: str) -> str:
    """判断重复用：NFKC、去掉全部空白。"""
    return "".join(unicodedata.normalize("NFKC", text).split())


def merge_detail_text(platform_text: str | None, ocr_texts: Sequence[str]) -> str | None:
    """平台详情文字 + 详情图文字（规则见模块文档）。确定：同样的输入总是同样的输出。"""
    base = (platform_text or "").strip()
    parts = [base] if base else []
    base_key = _key(base)
    seen: set[str] = set()
    for text in ocr_texts:
        key = _key(text)
        if not key or key in seen or key in base_key:
            continue
        seen.add(key)
        parts.append(text)
    return "\n".join(parts) or None


class ItemSync:
    """平台商品 → 商品知识卡片。

    lock：卡片"读—合并—写"用的锁（和商家改卡片共用，同一进程内不互相覆盖）；
    识别详情图在锁外做（可能比较慢）。
    """

    def __init__(
        self,
        repo: KnowledgeRepository,
        catalog: ItemCatalog | None,
        ocr: ImageTextExtractor | None,
        audit: AuditSink,
        clock: Clock,
        *,
        platform: Platform,
        lock: threading.Lock | None = None,
    ) -> None:
        if catalog is None or ocr is None:
            raise ItemSyncUnavailable()
        if catalog.platform != platform:
            raise TenantIsolationError(
                f"商品接口属于 {catalog.platform}，单元是 {platform}（§9.1 数据不跨云）"
            )
        self._repo = repo
        self._catalog = catalog
        self._ocr = ocr
        self._audit = audit
        self._clock = clock
        self._platform = platform
        self._lock = lock

    # ================================================================ 入口

    def sync_shop(self, shop_id: str, *, operator: str) -> SyncReport:
        """同步这家店铺在平台上的全部商品（含已下架的）。"""
        items = self._catalog.list_items(shop_id)
        report = self._run(shop_id, items, scope="shop")
        platform_ids = {i.item_id for i in items if self._owned(shop_id, i)}
        report.missing_on_platform = sorted(
            c.item_id
            for c in self._repo.list_item_cards(shop_id)
            if c.shop_id == shop_id and c.item_id not in platform_ids
        )
        self._record(shop_id, operator, report)
        return report

    def sync_item(self, shop_id: str, item_id: str, *, operator: str) -> SyncReport:
        """只同步一个商品。平台上和本地都没有这个商品抛 KeyError；平台上没有、本地有卡片的
        列进 missing_on_platform（不删）。"""
        item = self._catalog.get_item(shop_id, item_id)
        if item is not None and item.item_id != item_id:
            raise TenantIsolationError("商品接口返回的商品和请求的不一致")
        report = self._run(shop_id, [item] if item is not None else [], scope="item")
        report.item_id = item_id
        if item is None:
            if self._repo.get_item_card(shop_id, item_id) is None:
                raise KeyError(f"店铺 {shop_id} 没有商品 {item_id}")
            report.missing_on_platform = [item_id]
        self._record(shop_id, operator, report)
        return report

    # ================================================================ 内部

    def _owned(self, shop_id: str, item: PlatformItem) -> bool:
        return item.shop_id == shop_id and item.platform == self._platform

    def _run(
        self, shop_id: str, items: Sequence[PlatformItem], *, scope: Literal["shop", "item"]
    ) -> SyncReport:
        report = SyncReport(shop_id=shop_id, scope=scope, synced_at=self._clock.now())
        seen: set[str] = set()
        for item in sorted(items, key=lambda i: i.item_id):
            if not self._owned(shop_id, item):
                report.rejected += 1
                logger.error("商品接口返回了不属于店铺 / 本单元平台的商品，已跳过（§9.2）")
                continue
            if item.item_id in seen:  # 接口重复返回同一个商品：只同步一次
                continue
            seen.add(item.item_id)
            self._sync_one(shop_id, item, report)
        return report

    def _sync_one(self, shop_id: str, item: PlatformItem, report: SyncReport) -> None:
        texts, failures = self._extract(shop_id, item)
        if failures:
            report.ocr_failures += failures
            report.ocr_failed_items.append(item.item_id)
        with self._lock if self._lock is not None else nullcontext():
            existing = self._repo.get_item_card(shop_id, item.item_id)
            if existing is not None and existing.shop_id != shop_id:
                raise TenantIsolationError("知识仓储返回了别的店铺的商品")
            card = self._merge(shop_id, existing, item, texts, partial=failures > 0)
            if existing is None:
                self._repo.upsert_item_card(card)
                report.created.append(item.item_id)
            elif _same(existing, card):
                report.unchanged.append(item.item_id)
            else:
                # 只写平台维护的字段（仓储里原子地读改写）：商家同时在别的实例上改尺码建议等
                # 自己填的字段，两边的修改都留下
                saved = self._repo.update_item_card(
                    shop_id,
                    item.item_id,
                    {f: getattr(card, f) for f in (*PLATFORM_ITEM_FIELDS, "updated_at")},
                )
                if saved is None:  # 同步过程中卡片被删了（取消订购清理）：不重建
                    return
                report.updated.append(item.item_id)
        if not item.on_sale:
            report.off_sale.append(item.item_id)

    def _extract(self, shop_id: str, item: PlatformItem) -> tuple[list[str], int]:
        """逐张识别详情图，返回 (识别出的文字，按图片顺序；失败张数)。一张失败不影响其他。"""
        texts: list[str] = []
        failures = 0
        for ref in dict.fromkeys(item.detail_images):  # 同一张图只识别一次
            try:
                text = self._ocr.extract_text(shop_id, ref)
            except Exception as exc:  # noqa: BLE001 — 协议是 ToolUnavailable；其余异常同样只算这张失败
                failures += 1
                logger.warning("详情图识别失败（%s），商品同步照常进行", type(exc).__name__)
                continue
            cleaned = _clean_ocr(text)
            if cleaned:
                texts.append(cleaned)
        return texts, failures

    def _merge(
        self,
        shop_id: str,
        existing: ItemCard | None,
        item: PlatformItem,
        texts: list[str],
        *,
        partial: bool,
    ) -> ItemCard:
        detail = merge_detail_text(item.detail_text, texts)
        if partial and existing is not None and existing.detail_text:
            base = (item.detail_text or "").strip()
            if existing.detail_text.startswith(base):
                detail = existing.detail_text  # 平台文字没变：保留上次合并的结果（见模块文档）
        return ItemCard(
            shop_id=shop_id,
            platform=self._platform,
            item_id=item.item_id,
            title=item.title,
            category=item.category,
            attributes=dict(item.attributes),
            skus=[s.model_copy(deep=True) for s in item.skus],
            size_chart=[dict(row) for row in item.size_chart],
            size_advice=existing.size_advice if existing else None,
            care=list(existing.care) if existing else [],
            usage=list(existing.usage) if existing else [],
            forbidden_claims=list(existing.forbidden_claims) if existing else [],
            detail_text=detail,
            updated_at=self._clock.now(),
        )

    def _record(self, shop_id: str, operator: str, report: SyncReport) -> None:
        self._audit.record(
            AuditEvent(
                shop_id=shop_id,
                actor=operator,
                action=AuditAction.ITEM_SYNC.value,
                detail={"scope": report.scope, **report.counts()},
                created_at=self._clock.now(),
            )
        )


PLATFORM_ITEM_FIELDS: tuple[str, ...] = (
    "title",
    "category",
    "attributes",
    "skus",
    "size_chart",
    "detail_text",
)
"""商品卡片上由平台商品接口维护的字段，同步只改这些（商家填的见 admin.MERCHANT_ITEM_FIELDS）。"""


def _same(a: ItemCard, b: ItemCard) -> bool:
    """除 updated_at 之外完全相同。"""
    exclude = {"updated_at"}
    return a.model_dump(mode="json", exclude=exclude) == b.model_dump(mode="json", exclude=exclude)


__all__ = [
    "MAX_OCR_CHARS",
    "MSG_SYNC_UNAVAILABLE",
    "ItemSync",
    "ItemSyncUnavailable",
    "SyncReport",
    "merge_detail_text",
]
