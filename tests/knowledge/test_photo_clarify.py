"""买家发来商品图、当前商品又不确定（方案 §5.2.4 商品不确定时先澄清、§15 Phase 3 图片理解）：
图片认不出是店里哪件商品，问"请问您图片里的是哪一款呢？"，不拿别的商品回答。
"""

from __future__ import annotations

import pytest

from rodit.core import evidence as ev
from rodit.core.enums import ImageKind, Intent
from rodit.core.evidence import Need
from rodit.core.models import EvidenceBundle, ImageFinding
from rodit.knowledge import CLARIFY_ITEM_QUESTION
from rodit.knowledge.retrieval import CLARIFY_PHOTO_ITEM_QUESTION, photo_item_clarify
from rodit.testing.fakes import make_context

PHOTO = "这款还有货吗 【买家发来图片：商品图（一件碎花连衣裙的商品图）】"


def photo(kind: ImageKind = ImageKind.PRODUCT_PHOTO, conf: float = 0.72) -> ImageFinding:
    return ImageFinding(ref="img_p", kind=kind, confidence=conf, summary="一件碎花连衣裙")


def ctx_with(world, *findings: ImageFinding, text: str = PHOTO, **kw):
    ctx = make_context(text, world, **kw)
    return ctx.model_copy(update={"image_findings": list(findings)})


class ToolSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[Intent, frozenset[Need]]] = []

    def collect(self, ctx, intent, needs):
        self.calls.append((intent, needs))
        return EvidenceBundle(missing=[ev.MISSING_ITEM])


@pytest.mark.parametrize(
    "intent",
    [Intent.STOCK, Intent.PRODUCT_SPEC, Intent.SIZE_ADVICE, Intent.PRICE_PROMO, Intent.USAGE],
)
def test_product_photo_without_item_asks_which_one(world, make_retriever, route, intent):
    b = make_retriever(tool_provider=ToolSpy()).retrieve(ctx_with(world, photo()), route(intent))
    assert b.clarify_question == CLARIFY_PHOTO_ITEM_QUESTION == "请问您图片里的是哪一款呢？"
    assert not any(e.id.startswith("item:") for e in b.evidences)
    if intent not in (Intent.STOCK, Intent.PRICE_PROMO):  # 库存、价格的缺失由工具层记
        assert ev.MISSING_ITEM not in b.missing


def test_product_photo_with_several_candidates_asks_about_the_photo(world, make_retriever, route):
    ctx = ctx_with(world, photo(), candidate_item_ids=["i_555", "i_777"])
    b = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))
    assert b.clarify_question == CLARIFY_PHOTO_ITEM_QUESTION


def test_known_item_is_kept(world, make_retriever, route):
    """会话里已经确定了当前商品（商品卡片）：照常按这件商品回答，图片只给客服参考。"""
    ctx = ctx_with(world, photo(), item_id="i_555")
    b = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))
    assert b.clarify_question is None
    assert any(e.id.startswith("item:i_555") for e in b.evidences)


@pytest.mark.parametrize(
    "found",
    [
        [],
        [photo(conf=0.3)],  # 看不太清
        [photo(ImageKind.DAMAGE)],
        [photo(ImageKind.SIZE_REFERENCE)],
    ],
)
def test_without_clear_product_photo_behaviour_is_unchanged(world, make_retriever, route, found):
    ctx = ctx_with(world, *found, candidate_item_ids=[])
    ctx = ctx.model_copy(update={"item": None, "candidate_item_ids": []})
    b = make_retriever().retrieve(ctx, route(Intent.PRODUCT_SPEC))
    assert b.clarify_question is None
    assert ev.MISSING_ITEM in b.missing
    two = ctx_with(world, *found, candidate_item_ids=["i_555", "i_777"])
    assert make_retriever().retrieve(two, route(Intent.PRODUCT_SPEC)).clarify_question == (
        CLARIFY_ITEM_QUESTION
    )


@pytest.mark.parametrize("intent", [Intent.SHIPPING_POLICY, Intent.INVOICE_GIFT, Intent.GREETING])
def test_photo_does_not_force_clarify_for_shop_level_questions(
    world, make_retriever, route, intent
):
    """店铺层面的问题（发货、发票、问候）不需要确定商品：商品图不影响。"""
    b = make_retriever().retrieve(ctx_with(world, photo()), route(intent))
    assert b.clarify_question is None


def test_photo_item_clarify_helper(world):
    assert photo_item_clarify(ctx_with(world, photo())) == CLARIFY_PHOTO_ITEM_QUESTION
    assert photo_item_clarify(ctx_with(world, photo(), item_id="i_555")) is None
    assert photo_item_clarify(ctx_with(world)) is None
