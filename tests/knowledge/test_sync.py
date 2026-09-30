"""平台商品同步 + 详情图 OCR（§5.2.3 冷启动第 2 步、§5.2.1 商品变更时同步）。

平台商品接口和 OCR 都是合成替身（InMemoryItemCatalog、FakeImageTextExtractor），不联网。
演示数据见 rodit.testing.fixtures.demo_platform_items：i_555 标题改了、多一张详情图；i_777 没变；
i_888 已下架；i_999 是新商品（一张图能识别、一张识别失败）。
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.enums import Platform
from rodit.core.errors import TenantIsolationError, ToolUnavailable
from rodit.core.models import PlatformItem
from rodit.inmemory import InMemoryKnowledgeRepository
from rodit.knowledge.sync import (
    MAX_OCR_CHARS,
    ItemSync,
    ItemSyncUnavailable,
    SyncReport,
    merge_detail_text,
)
from rodit.testing.fakes import FakeClock, FakeImageTextExtractor, InMemoryItemCatalog
from rodit.testing.fixtures import (
    DEFAULT_NOW,
    DEMO_OCR_TEXTS,
    SHOP_A,
    SHOP_B,
    DemoWorld,
    build_demo_world,
    demo_platform_items,
    seed_knowledge,
)

OPERATOR = "owner_a"


class CountingRepo:
    """包一层内存仓储，记下写商品卡片的次数（重复同步不应该写）。"""

    def __init__(self, inner: InMemoryKnowledgeRepository) -> None:
        self._inner = inner
        self.card_writes: list[str] = []
        self.bookkeeping: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def upsert_item_card(self, card) -> None:
        self.card_writes.append(card.item_id)
        self._inner.upsert_item_card(card)

    def update_item_card(self, shop_id: str, item_id: str, changes):
        if set(changes) == {"platform_detail_text"}:
            self.bookkeeping.append(item_id)  # 只补记平台原文（升级前同步的卡片）
        else:
            self.card_writes.append(item_id)
        return self._inner.update_item_card(shop_id, item_id, changes)


class Kit:
    def __init__(self, repo: InMemoryKnowledgeRepository, world: DemoWorld) -> None:
        self.inner = repo
        self.repo = CountingRepo(repo)
        self.world = world
        self.clock = FakeClock(world.now)
        self.audit = InMemoryAuditSink()
        self.catalog = InMemoryItemCatalog(demo_platform_items(world), Platform.TAOBAO)
        self.ocr = FakeImageTextExtractor(DEMO_OCR_TEXTS)
        self.sync = ItemSync(
            self.repo,  # type: ignore[arg-type]
            self.catalog,
            self.ocr,
            self.audit,
            self.clock,
            platform=Platform.TAOBAO,
        )

    def run(self) -> SyncReport:
        return self.sync.sync_shop(SHOP_A, operator=OPERATOR)

    def platform_item(self, item_id: str, shop_id: str = SHOP_A) -> PlatformItem:
        return next(i for i in self.catalog.items if i.item_id == item_id and i.shop_id == shop_id)

    def replace(self, item: PlatformItem) -> None:
        self.catalog.items = [
            item if (i.shop_id, i.item_id) == (item.shop_id, item.item_id) else i
            for i in self.catalog.items
        ]


@pytest.fixture
def kit(repo: InMemoryKnowledgeRepository, world: DemoWorld) -> Kit:
    return Kit(repo, world)


# ================================================================ 整店同步


def test_first_sync_creates_updates_and_keeps_merchant_fields(kit: Kit):
    before = {c.item_id: c for c in kit.inner.list_item_cards(SHOP_A)}
    kit.clock.advance(minutes=30)
    report = kit.run()
    assert report.created == ["i_999"]
    assert report.updated == ["i_555"]
    assert report.unchanged == ["i_777", "i_888"]
    assert report.off_sale == ["i_888"]
    assert report.missing_on_platform == []
    assert report.ocr_failed_items == ["i_999"] and report.ocr_failures == 1
    assert report.rejected == 0

    dress = kit.inner.get_item_card(SHOP_A, "i_555")
    old = before["i_555"]
    assert dress.title == "法式碎花连衣裙（新款）"  # 平台字段跟着平台
    # 商家填的字段原样保留
    assert dress.size_advice == old.size_advice and dress.care == old.care
    assert dress.forbidden_claims == old.forbidden_claims and dress.usage == old.usage
    assert dress.detail_text == old.detail_text + "\n" + DEMO_OCR_TEXTS["img_555_detail_1"]
    assert dress.updated_at == kit.clock.now()
    # 没变的商品不写，updated_at 不变
    assert kit.inner.get_item_card(SHOP_A, "i_777").updated_at == before["i_777"].updated_at
    assert sorted(kit.repo.card_writes) == ["i_555", "i_999"]
    # 没变的卡片只补记平台原文（不动内容、不动 updated_at），再同步就不写了
    assert sorted(kit.repo.bookkeeping) == ["i_777", "i_888"]

    new = kit.inner.get_item_card(SHOP_A, "i_999")
    assert new.platform == Platform.TAOBAO and new.title == "宽松针织开衫"
    assert new.detail_text == "慵懒宽松版型，春秋百搭。\n" + DEMO_OCR_TEXTS["img_999_detail_1"]
    assert new.size_advice is None and new.care == [] and new.forbidden_claims == []
    assert [s.sku_id for s in new.skus] == ["k_91", "k_92"]


def test_resync_is_a_no_op(kit: Kit):
    kit.run()
    cards = {c.item_id: c for c in kit.inner.list_item_cards(SHOP_A)}
    kit.repo.card_writes.clear()
    kit.repo.bookkeeping.clear()
    kit.clock.advance(hours=1)
    report = kit.run()
    assert report.created == [] and report.updated == []
    assert report.unchanged == ["i_555", "i_777", "i_888", "i_999"]
    assert kit.repo.card_writes == [] and kit.repo.bookkeeping == []
    assert {c.item_id: c for c in kit.inner.list_item_cards(SHOP_A)} == cards
    dress = kit.inner.get_item_card(SHOP_A, "i_555")
    assert dress.detail_text.count(DEMO_OCR_TEXTS["img_555_detail_1"]) == 1  # 没有越同步越长


def test_merchant_edits_between_syncs_survive(kit: Kit):
    kit.run()
    card = kit.inner.get_item_card(SHOP_A, "i_999")
    kit.inner.upsert_item_card(
        card.model_copy(update={"size_advice": "均码，160–170 cm 都能穿", "care": ["手洗"]})
    )
    kit.replace(kit.platform_item("i_999").model_copy(update={"title": "宽松针织开衫（加厚）"}))
    report = kit.run()
    assert report.updated == ["i_999"]
    card = kit.inner.get_item_card(SHOP_A, "i_999")
    assert card.title == "宽松针织开衫（加厚）"
    assert card.size_advice == "均码，160–170 cm 都能穿" and card.care == ["手洗"]


def test_audit_has_counts_only(kit: Kit):
    kit.run()
    [event] = kit.audit.list_events(SHOP_A, action=AuditAction.ITEM_SYNC.value)
    assert event.actor == OPERATOR
    assert event.detail == {
        "scope": "shop",
        "created": 1,
        "updated": 1,
        "unchanged": 2,
        "off_sale": 1,
        "missing_on_platform": 0,
        "ocr_failed_items": 1,
        "ocr_failures": 1,
        "rejected": 0,
    }
    text = json.dumps(event.detail, ensure_ascii=False)
    assert "i_999" not in text and "针织" not in text


# ================================================================ 详情图 OCR


def test_ocr_failure_does_not_lose_previously_extracted_text(kit: Kit):
    kit.run()
    dress_before = kit.inner.get_item_card(SHOP_A, "i_555")
    del kit.ocr.texts["img_555_detail_1"]  # 这次识别失败（上次成功过）
    report = kit.run()
    assert report.ocr_failed_items == ["i_555", "i_999"] and report.ocr_failures == 2
    assert "i_555" in report.unchanged  # 平台文字没变：保留上次合并的结果，不来回改卡片
    assert kit.inner.get_item_card(SHOP_A, "i_555") == dress_before


def test_ocr_failure_with_new_platform_text_uses_the_new_text(kit: Kit):
    kit.run()
    del kit.ocr.texts["img_555_detail_1"]
    kit.replace(kit.platform_item("i_555").model_copy(update={"detail_text": "法式复古碎花。"}))
    report = kit.run()
    assert report.updated == ["i_555"]
    assert kit.inner.get_item_card(SHOP_A, "i_555").detail_text == "法式复古碎花。"


def test_any_ocr_exception_only_fails_that_image(kit: Kit):
    class Exploding(FakeImageTextExtractor):
        def extract_text(self, shop_id: str, image_ref: str) -> str:
            if image_ref == "img_999_detail_1":
                raise RuntimeError("OCR SDK 超时（模拟）")
            return super().extract_text(shop_id, image_ref)

    kit.sync._ocr = Exploding(DEMO_OCR_TEXTS)
    report = kit.run()
    assert report.ocr_failures == 2 and "i_999" in report.created
    assert kit.inner.get_item_card(SHOP_A, "i_999").detail_text == "慵懒宽松版型，春秋百搭。"


def test_each_image_is_read_once_and_blank_or_duplicate_text_is_dropped(kit: Kit):
    kit.ocr.texts.update(
        {"img_a": "  \n", "img_b": "慵懒宽松版型，春秋百搭。", "img_c": "毛衣\x00针织"}
    )
    item = kit.platform_item("i_999").model_copy(
        update={"detail_images": ["img_a", "img_b", "img_c", "img_c", "img_999_detail_1"]}
    )
    kit.replace(item)
    kit.run()
    assert [ref for _, ref in kit.ocr.calls if ref.startswith("img_")].count("img_c") == 1
    assert kit.inner.get_item_card(SHOP_A, "i_999").detail_text == (
        "慵懒宽松版型，春秋百搭。\n毛衣针织\n" + DEMO_OCR_TEXTS["img_999_detail_1"]
    )


def test_merge_detail_text_is_deterministic():
    assert merge_detail_text(None, []) is None
    assert merge_detail_text("  ", ["  "]) is None
    assert merge_detail_text(None, ["图一", "图二"]) == "图一\n图二"
    # 和平台文字重复（含空白差异）、和前面的图重复的不再加
    assert merge_detail_text("领口 包边。", ["领口包边。", "腰部松紧", "腰部 松紧"]) == (
        "领口 包边。\n腰部松紧"
    )
    assert merge_detail_text("A", ["B"]) == merge_detail_text("A", ["B"])


def test_long_ocr_text_is_truncated(kit: Kit):
    kit.ocr.texts["img_999_detail_1"] = "长" * (MAX_OCR_CHARS + 50)
    kit.run()
    detail = kit.inner.get_item_card(SHOP_A, "i_999").detail_text
    assert detail.endswith("长" * MAX_OCR_CHARS) and "长" * (MAX_OCR_CHARS + 1) not in detail


# ================================================================ 下架、平台上没有了


def test_off_sale_and_missing_items_are_reported_not_deleted(kit: Kit):
    kit.catalog.items = [
        i for i in kit.catalog.items if not (i.shop_id == SHOP_A and i.item_id == "i_777")
    ]
    report = kit.run()
    assert report.missing_on_platform == ["i_777"]
    assert report.off_sale == ["i_888"]
    assert kit.inner.get_item_card(SHOP_A, "i_777") is not None
    assert kit.inner.get_item_card(SHOP_A, "i_888") is not None


# ================================================================ 单个商品


def test_sync_one_item(kit: Kit):
    report = kit.sync.sync_item(SHOP_A, "i_555", operator=OPERATOR)
    assert report.scope == "item" and report.item_id == "i_555"
    assert report.updated == ["i_555"] and report.created == []
    assert [c for c, _ in kit.catalog.calls] == ["get_item"]
    assert kit.inner.get_item_card(SHOP_A, "i_999") is None  # 别的商品不动


def test_sync_one_item_that_is_gone_from_the_platform(kit: Kit):
    kit.catalog.items = [i for i in kit.catalog.items if i.item_id != "i_777"]
    report = kit.sync.sync_item(SHOP_A, "i_777", operator=OPERATOR)
    assert report.missing_on_platform == ["i_777"]
    assert kit.inner.get_item_card(SHOP_A, "i_777") is not None
    with pytest.raises(KeyError):
        kit.sync.sync_item(SHOP_A, "i_nope", operator=OPERATOR)


def test_another_shops_item_cannot_be_synced_into_this_shop(kit: Kit, world: DemoWorld):
    b_only = PlatformItem(
        shop_id=SHOP_B,
        platform=Platform.TAOBAO,
        item_id="i_b_only",
        title="男士衬衫",
        updated_at=world.now,
    )
    kit.catalog.items.append(b_only)
    with pytest.raises(KeyError):
        kit.sync.sync_item(SHOP_A, "i_b_only", operator=OPERATOR)
    assert kit.inner.get_item_card(SHOP_A, "i_b_only") is None
    assert kit.inner.get_item_card(SHOP_B, "i_b_only") is None


# ================================================================ 故障与纵深防御


def test_catalog_failure_propagates_and_writes_nothing(kit: Kit):
    kit.catalog.fail_on.add("list_items")
    before = kit.inner.list_item_cards(SHOP_A)
    with pytest.raises(ToolUnavailable):
        kit.run()
    assert kit.inner.list_item_cards(SHOP_A) == before
    assert kit.audit.events == [] and kit.ocr.calls == []


def test_missing_catalog_or_ocr_makes_sync_unavailable(kit: Kit):
    for catalog, ocr in ((None, kit.ocr), (kit.catalog, None), (None, None)):
        with pytest.raises(ItemSyncUnavailable):
            ItemSync(kit.inner, catalog, ocr, kit.audit, kit.clock, platform=Platform.TAOBAO)
    with pytest.raises(TenantIsolationError):
        ItemSync(kit.inner, kit.catalog, kit.ocr, kit.audit, kit.clock, platform=Platform.PDD)


class LeakyCatalog(InMemoryItemCatalog):
    """有 bug 的商品接口：不按店铺过滤。"""

    def list_items(self, shop_id: str) -> list[PlatformItem]:
        return sorted(self.items, key=lambda i: (i.item_id, i.shop_id))


def test_items_of_other_shops_or_platforms_are_rejected(kit: Kit, world: DemoWorld):
    leaky = LeakyCatalog(demo_platform_items(world), Platform.TAOBAO)
    leaky.items.append(
        PlatformItem(
            shop_id=SHOP_A,
            platform=Platform.PDD,  # 平台不对（数据不跨云，§9.1）
            item_id="i_pdd",
            title="别的平台的商品",
            updated_at=world.now,
        )
    )
    kit.sync._catalog = leaky
    b_before = kit.inner.list_item_cards(SHOP_B)
    report = kit.run()
    assert report.rejected == 2  # 店铺 B 的 i_555、平台不对的 i_pdd
    assert "i_pdd" not in json.dumps(report.model_dump(mode="json"))
    assert kit.inner.get_item_card(SHOP_A, "i_pdd") is None
    assert kit.inner.list_item_cards(SHOP_B) == b_before
    assert kit.inner.get_item_card(SHOP_A, "i_555").title == "法式碎花连衣裙（新款）"


def test_duplicate_items_from_the_catalog_are_synced_once(kit: Kit):
    kit.catalog.items.append(kit.platform_item("i_999"))
    report = kit.run()
    assert report.created == ["i_999"]


def test_updated_at_uses_the_injected_clock(kit: Kit):
    kit.clock.set(kit.world.now + timedelta(days=1))
    kit.run()
    assert kit.inner.get_item_card(SHOP_A, "i_999").updated_at == kit.world.now + timedelta(days=1)


# ================================================================ 识别失败：平台原文没变才保留


def _one_item(detail: str | None, images: list[str]) -> PlatformItem:
    return PlatformItem(
        shop_id=SHOP_A,
        platform=Platform.TAOBAO,
        item_id="i_coat",
        title="羊毛大衣",
        detail_text=detail,
        detail_images=images,
        updated_at=DEFAULT_NOW,
    )


@pytest.mark.parametrize(
    "new_detail",
    [
        "含羊毛 30%。",  # 删掉了后半句：新文字是旧合并结果的前缀
        None,  # 平台清空了详情文字
        "含羊毛 30%。\n支持七天无理由退换",  # 多行：删掉了中间一句后仍像是前缀
    ],
)
def test_partial_ocr_failure_never_keeps_text_the_platform_withdrew(new_detail):
    clock = FakeClock()
    repo = InMemoryKnowledgeRepository(clock)
    seed_knowledge(repo, build_demo_world())
    old_detail = "含羊毛 30%。\n支持七天无理由退换\n可机洗"
    catalog = InMemoryItemCatalog([_one_item(old_detail, ["img1"])], Platform.TAOBAO)
    sync = ItemSync(
        repo,
        catalog,
        FakeImageTextExtractor({"img1": "图一：水洗标"}),
        InMemoryAuditSink(),
        clock,
        platform=Platform.TAOBAO,
    )
    sync.sync_shop(SHOP_A, operator="owner_a")
    catalog.items = [_one_item(new_detail, ["img2"])]  # 新图识别失败
    clock.advance(hours=1)
    report = sync.sync_shop(SHOP_A, operator="owner_a")
    card = repo.get_item_card(SHOP_A, "i_coat")
    assert report.updated == ["i_coat"]
    assert "可机洗" not in (card.detail_text or "")
    assert card.detail_text == ((new_detail or "").strip() or None)


def test_partial_ocr_failure_keeps_last_merge_when_platform_text_is_unchanged():
    clock = FakeClock()
    repo = InMemoryKnowledgeRepository(clock)
    seed_knowledge(repo, build_demo_world())
    ocr = FakeImageTextExtractor({"img1": "图一：水洗标"})
    catalog = InMemoryItemCatalog([_one_item("含羊毛 30%。", ["img1"])], Platform.TAOBAO)
    sync = ItemSync(repo, catalog, ocr, InMemoryAuditSink(), clock, platform=Platform.TAOBAO)
    sync.sync_shop(SHOP_A, operator="owner_a")
    first = repo.get_item_card(SHOP_A, "i_coat")
    assert first.detail_text == "含羊毛 30%。\n图一：水洗标"
    del ocr.texts["img1"]  # 这次识别失败
    report = sync.sync_shop(SHOP_A, operator="owner_a")
    assert "i_coat" in report.unchanged
    assert repo.get_item_card(SHOP_A, "i_coat") == first
