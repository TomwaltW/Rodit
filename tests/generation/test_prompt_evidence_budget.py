"""检索保留的依据，提示词一条都不能丢（§14 控制证据长度：两处按同一个字符串计长）。"""

from __future__ import annotations

from rodit.core.config import GenerationConfig
from rodit.core.enums import EvidenceSource, Intent
from rodit.core.models import Evidence, EvidenceBundle
from rodit.generation import build_generation_messages
from rodit.inmemory import HashingEmbedder, InMemoryKnowledgeRepository
from rodit.knowledge import FaqSearcher, KnowledgeRetriever
from rodit.testing.fakes import FakeClock, make_context
from rodit.testing.fixtures import SHOP_A, seed_knowledge

CFG = GenerationConfig()
RELEVANT = "这款连衣裙的内衬是涤纶材质，透气不闷热。"


def _evidence_block(user: str) -> str:
    return user.split("【依据】\n")[1].split("\n【对话】")[0]


def test_prompt_keeps_every_evidence_the_retriever_kept(world, route):
    """审查复现：长详情页的第 69 句和问题最相关，检索留下了它，提示词却按位置截掉了。"""
    base = world.item(SHOP_A, "i_555")
    filler = "".join(f"第{n}段说明文字，版型宽松日常百搭。" for n in range(69))
    world.items.append(
        base.model_copy(update={"item_id": "i_long", "detail_text": filler + RELEVANT})
    )
    repo = InMemoryKnowledgeRepository(FakeClock(world.now))
    seed_knowledge(repo, world)
    retriever = KnowledgeRetriever(repo, FaqSearcher(repo, HashingEmbedder(256)), CFG)
    ctx = make_context("内衬是什么材质的", world, item_id="i_long")
    bundle = retriever.retrieve(ctx, route(Intent.PRODUCT_SPEC))
    assert "item:i_long:detail:69" in bundle.ids()

    user = build_generation_messages(ctx, route(Intent.PRODUCT_SPEC), bundle, config=CFG)[1]
    block = _evidence_block(user.content)
    for eid in bundle.ids():
        assert f"[{eid}]" in block, eid
    assert RELEVANT in block
    assert len(block) <= CFG.max_evidence_chars


def test_line_that_does_not_fit_is_skipped_not_the_rest(world, route):
    """单条放不下时跳过它，后面放得下的依据照常放（不在第一条放不下处整体截断）。"""
    long_one = Evidence(
        id="item:i_555:detail:0", source=EvidenceSource.ITEM, label="商品详情", content="长" * 150
    )
    short = Evidence(
        id="item:i_555:detail:1", source=EvidenceSource.ITEM, label="商品详情", content="面料亲肤"
    )
    first = Evidence(
        id="item:i_555:attr:面料", source=EvidenceSource.ITEM, label="商品属性", content="棉"
    )
    bundle = EvidenceBundle(evidences=[first, long_one, short])
    ctx = make_context("面料怎么样", world, item_id="i_555")
    small = GenerationConfig(max_evidence_chars=80)
    user = build_generation_messages(ctx, route(Intent.PRODUCT_SPEC), bundle, config=small)[1]
    block = _evidence_block(user.content)
    assert "[item:i_555:detail:1]" in block
    assert len(block) <= 80
