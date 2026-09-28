"""仓储协议的契约测试（core.ports 第 2 组）。

内存实现（rodit.inmemory）和 PostgreSQL 实现（rodit.storage）跑同一套检查，保证语义一致；
check_cross_shop_isolation 是方案 §9.2 要求的"串店测试"：
用 A 店铺的身份尝试读取 B 店铺的所有数据类型。

用法：在测试里传入"新建一个空仓储"的工厂函数，例如
    check_knowledge_repository(lambda: InMemoryKnowledgeRepository(FakeClock()))
每个 check_* 自己造数据、自己断言，失败时抛 AssertionError。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from pydantic import SecretStr

from rodit.core.enums import (
    AfterSaleStatus,
    DraftKind,
    Intent,
    KnowledgeSource,
    KnowledgeStatus,
    ReviewAdvice,
    ReviewState,
    Role,
    SuggestionStatus,
)
from rodit.core.errors import ConcurrencyConflict
from rodit.core.models import (
    AfterSaleReview,
    KnowledgeGap,
    KnowledgeItem,
    LabeledSample,
    ShopAuth,
    SuggestionRecord,
)
from rodit.core.ports import (
    AfterSaleReviewRepository,
    ConversationRepository,
    FeedbackRepository,
    KnowledgeRepository,
    ShopAuthRepository,
    SuggestionRepository,
)
from rodit.testing.fakes import make_message
from rodit.testing.fixtures import (
    BUYER,
    SHOP_A,
    SHOP_B,
    build_demo_world,
    seed_knowledge,
)

_W = build_demo_world()
NOW = _W.now


def _unit(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


# ---------------------------------------------------------------- 知识


def check_knowledge_repository(make: Callable[[], KnowledgeRepository]) -> None:
    repo = make()
    world = build_demo_world()
    ids = seed_knowledge(repo, world)

    # 店铺、政策、商品原样读回，按店隔离
    shop_a = repo.get_shop(SHOP_A)
    assert shop_a == world.shop(SHOP_A)
    assert repo.get_shop("s_nope") is None
    pol = repo.get_policies(SHOP_A)
    assert pol["ship_time"].value.startswith("现货付款后 48 小时")
    assert repo.get_policies(SHOP_B)["ship_time"].value == "付款后 72 小时内发货"
    card = repo.get_item_card(SHOP_A, "i_555")
    assert card == world.item(SHOP_A, "i_555")
    assert card.find_sku("k_1").props() == {"颜色": "黑色", "尺码": "M"}
    assert repo.get_item_card(SHOP_B, "i_555").title == "男士休闲裤"
    assert [c.item_id for c in repo.list_item_cards(SHOP_A)] == ["i_555", "i_777", "i_888"]

    # 返回的是副本
    card.title = "被改掉"
    assert repo.get_item_card(SHOP_A, "i_555").title == "法式碎花连衣裙"

    # FAQ：状态过滤、商品范围
    approved = repo.list_knowledge(SHOP_A)
    assert all(k.status == KnowledgeStatus.APPROVED for k in approved)
    questions = [k.question for k in approved]
    assert "这款会透吗" in questions  # item_id=None 返回全部（含商品专属）
    assert "可以寄到新疆吗" not in questions  # 候选
    assert "双十一有活动吗" not in questions  # 已下线
    assert [k.id for k in approved] == sorted(k.id for k in approved)
    for_777 = [k.question for k in repo.list_knowledge(SHOP_A, item_id="i_777")]
    assert "这款会透吗" not in for_777 and "发什么快递" in for_777
    for_555 = [k.question for k in repo.list_knowledge(SHOP_A, item_id="i_555")]
    assert "这款会透吗" in for_555
    everything = repo.list_knowledge(SHOP_A, status=None)
    assert len(everything) == 7
    assert all(k.shop_id == SHOP_A for k in everything)
    assert [k.answer for k in repo.list_knowledge(SHOP_B)] == ["您好，默认发圆通快递。"]

    # 新增：忽略传入 id，补 created_at
    added = repo.add_knowledge(
        KnowledgeItem(
            id=999,
            shop_id=SHOP_A,
            question="能改地址吗",
            answer="亲，发货前可以改哦~",
            source=KnowledgeSource.ADOPTED,
            status=KnowledgeStatus.CANDIDATE,
        )
    )
    assert added.id is not None and added.id != 999 and added.created_at is not None
    assert repo.get_knowledge(SHOP_A, added.id).question == "能改地址吗"
    assert repo.get_knowledge(SHOP_B, added.id) is None

    # 审核通过
    ok = repo.set_knowledge_status(
        SHOP_A, added.id, KnowledgeStatus.APPROVED, approved_by="merchant"
    )
    assert ok.status == KnowledgeStatus.APPROVED and ok.approved_by == "merchant"
    try:
        repo.set_knowledge_status(SHOP_B, added.id, KnowledgeStatus.RETIRED)
    except KeyError:
        pass
    else:
        raise AssertionError("跨店修改 FAQ 状态应抛 KeyError")

    # 向量检索：只在有 embedding 的记录里找，item 范围、状态、分数排序
    dim = 8
    ship_q = ids[(SHOP_A, "发什么快递")]
    see_q = ids[(SHOP_A, "这款会透吗")]
    cand_q = ids[(SHOP_A, "可以寄到新疆吗")]
    b_q = ids[(SHOP_B, "发什么快递")]
    v_ship = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    v_see = _unit([0.9, 0.1, 0, 0, 0, 0, 0, 0])
    repo.set_knowledge_embedding(SHOP_A, ship_q, v_ship)
    repo.set_knowledge_embedding(SHOP_A, see_q, v_see)
    repo.set_knowledge_embedding(SHOP_A, cand_q, v_ship)
    repo.set_knowledge_embedding(SHOP_B, b_q, v_ship)
    try:
        repo.set_knowledge_embedding(SHOP_B, ship_q, v_ship)
    except KeyError:
        pass
    else:
        raise AssertionError("跨店写 embedding 应抛 KeyError")
    hits = repo.search_knowledge(SHOP_A, v_ship, top_k=5)
    assert [k.id for k, _ in hits] == [ship_q]  # 通用、已审核、有 embedding
    assert abs(hits[0][1] - 1.0) < 1e-6
    hits_555 = repo.search_knowledge(SHOP_A, v_ship, top_k=5, item_id="i_555")
    assert [k.id for k, _ in hits_555] == [ship_q, see_q]
    assert all(0.0 <= s <= 1.0 for _, s in hits_555)
    assert hits_555[0][1] >= hits_555[1][1]
    assert repo.search_knowledge(SHOP_A, v_ship, top_k=1, item_id="i_555")[0][0].id == ship_q
    cand_hits = repo.search_knowledge(SHOP_A, v_ship, status=KnowledgeStatus.CANDIDATE)
    assert [k.id for k, _ in cand_hits] == [cand_q]
    opposite = _unit([-1, 0, 0, 0, 0, 0, 0, 0])
    assert repo.search_knowledge(SHOP_A, opposite)[0][1] == 0.0  # 截断到 [0, 1]
    assert [k.shop_id for k, _ in repo.search_knowledge(SHOP_B, v_ship)] == [SHOP_B]
    assert len(v_ship) == dim

    # 删除店铺数据
    removed = repo.purge_shop(SHOP_B)
    assert removed > 0
    assert repo.get_shop(SHOP_B) is None and repo.list_knowledge(SHOP_B, status=None) == []
    assert repo.get_item_card(SHOP_B, "i_555") is None and repo.get_policies(SHOP_B) == {}
    assert repo.get_shop(SHOP_A) is not None


# ---------------------------------------------------------------- 会话


def check_conversation_repository(make: Callable[[], ConversationRepository]) -> None:
    repo = make()
    world = build_demo_world()
    m1 = make_message("在吗", world=world, msg_id="m_c1", at=NOW - timedelta(seconds=30))
    m2 = make_message(
        "这件黑的有 L 吗",
        world=world,
        msg_id="m_c2",
        item_id="i_555",
        sku_id="k_1",
        at=NOW - timedelta(seconds=10),
    )
    assert repo.add_message(m1) is True
    assert repo.add_message(m1) is False  # 按 (shop_id, msg_id) 去重
    assert repo.add_message(m2) is True
    assert repo.add_outgoing(
        shop_id=SHOP_A,
        conversation_id=m1.conversation_id,
        msg_id="m_c_agent",
        role=Role.AGENT,
        text="亲，在的",
        at=NOW - timedelta(seconds=20),
    )
    assert not repo.add_outgoing(
        shop_id=SHOP_A,
        conversation_id=m1.conversation_id,
        msg_id="m_c_agent",
        role=Role.AGENT,
        text="重复",
        at=NOW,
    )
    conv = repo.get_conversation(SHOP_A, m1.conversation_id)
    assert conv is not None and conv.buyer_ref == BUYER and conv.started_at == m1.received_at
    msgs = repo.recent_messages(SHOP_A, m1.conversation_id)
    assert [m.msg_id for m in msgs] == ["m_c1", "m_c_agent", "m_c2"]  # 时间升序
    assert msgs[2].item_id == "i_555" and msgs[2].sku_id == "k_1"
    assert msgs[1].role == Role.AGENT
    assert [m.msg_id for m in repo.recent_messages(SHOP_A, m1.conversation_id, limit=2)] == [
        "m_c_agent",
        "m_c2",
    ]
    # 同一 msg_id 在另一家店是另一条消息
    other = make_message("在吗", world=world, shop_id=SHOP_B, msg_id="m_c1")
    assert repo.add_message(other) is True
    assert repo.get_conversation(SHOP_B, m1.conversation_id) is not None
    assert [m.text for m in repo.recent_messages(SHOP_B, m1.conversation_id)] == ["在吗"]
    assert repo.purge_shop(SHOP_B) > 0
    assert repo.get_conversation(SHOP_B, m1.conversation_id) is None
    assert len(repo.recent_messages(SHOP_A, m1.conversation_id)) == 3


# ---------------------------------------------------------------- 推荐


def _suggestion(shop_id: str, conv: str, *, minutes: float, text: str = "推荐") -> SuggestionRecord:
    return SuggestionRecord(
        shop_id=shop_id,
        conversation_id=conv,
        msg_id="m_s",
        question_text="发什么快递",
        item_id="i_555",
        intent=Intent.SHIPPING_POLICY,
        text=text,
        kind=DraftKind.FAQ,
        evidence_ids=["faq:1"],
        qa_scores={"grounded": 0.96},
        status=SuggestionStatus.SHOWN,
        agent_id="kf_02",
        latency_ms=800,
        created_at=NOW - timedelta(minutes=minutes),
    )


def check_suggestion_repository(make: Callable[[], SuggestionRepository]) -> None:
    repo = make()
    a1 = repo.add(_suggestion(SHOP_A, "c1", minutes=3, text="a1").model_copy(update={"id": 77}))
    a2 = repo.add(_suggestion(SHOP_A, "c1", minutes=1, text="a2"))
    a3 = repo.add(_suggestion(SHOP_A, "c2", minutes=2, text="a3"))
    b1 = repo.add(_suggestion(SHOP_B, "c1", minutes=1, text="b1"))
    assert len({a1.id, a2.id, a3.id, b1.id}) == 4 and a1.id != 77
    assert repo.get(SHOP_A, a1.id) == a1
    assert repo.get(SHOP_B, a1.id) is None
    assert [r.text for r in repo.list(SHOP_A)] == ["a1", "a3", "a2"]  # created_at 升序
    assert [r.text for r in repo.list(SHOP_A, conversation_id="c1")] == ["a1", "a2"]
    assert [r.text for r in repo.list(SHOP_A, since=NOW - timedelta(minutes=2.5))] == ["a3", "a2"]
    updated = repo.update(
        a1.model_copy(update={"status": SuggestionStatus.EDITED_SENT, "edited_text": "改过"})
    )
    assert updated.status == SuggestionStatus.EDITED_SENT
    assert repo.get(SHOP_A, a1.id).edited_text == "改过"
    assert [r.text for r in repo.list(SHOP_A, status=SuggestionStatus.SHOWN)] == ["a3", "a2"]
    try:
        repo.update(b1.model_copy(update={"shop_id": SHOP_A}))
    except KeyError:
        pass
    else:
        raise AssertionError("用别的店铺身份更新推荐记录应抛 KeyError")
    assert repo.get(SHOP_B, b1.id).shop_id == SHOP_B
    assert repo.purge_shop(SHOP_B) == 1 and repo.list(SHOP_B) == []


# ---------------------------------------------------------------- 反馈


def check_feedback_repository(make: Callable[[], FeedbackRepository]) -> None:
    repo = make()
    repo.add_gap(
        KnowledgeGap(
            shop_id=SHOP_A, missing="policy:invoice", created_at=NOW - timedelta(minutes=1)
        )
    )
    repo.add_gap(
        KnowledgeGap(
            shop_id=SHOP_A,
            intent=Intent.SIZE_ADVICE,
            missing="item:i_777:size_advice",
            source="feedback",
            created_at=NOW - timedelta(minutes=2),
        )
    )
    repo.add_gap(KnowledgeGap(shop_id=SHOP_B, missing="policy:gift", created_at=NOW))
    assert [g.missing for g in repo.list_gaps(SHOP_A)] == [
        "item:i_777:size_advice",
        "policy:invoice",
    ]
    repo.add_label(
        LabeledSample(
            shop_id=SHOP_A,
            text="发啥快递",
            predicted_intent=Intent.OTHER,
            true_intent=Intent.SHIPPING_POLICY,
            source="feedback_reject",
            created_at=NOW,
        )
    )
    repo.add_label(
        LabeledSample(shop_id=SHOP_B, text="在吗", true_intent=Intent.GREETING, created_at=NOW)
    )
    assert [s.text for s in repo.list_labels(SHOP_A)] == ["发啥快递"]
    assert {s.shop_id for s in repo.export_training_labels()} == {SHOP_A, SHOP_B}
    assert repo.purge_shop(SHOP_B) == 2
    assert repo.list_gaps(SHOP_B) == [] and repo.list_labels(SHOP_B) == []
    assert len(repo.list_gaps(SHOP_A)) == 2


# ---------------------------------------------------------------- 售后审核


def check_aftersale_review_repository(make: Callable[[], AfterSaleReviewRepository]) -> None:
    repo = make()
    new = AfterSaleReview(
        shop_id=SHOP_A,
        aftersale_id="as_2002",
        platform_status=AfterSaleStatus.PENDING_REVIEW,
        state=ReviewState.NEW,
        version=42,  # 传入的 version 被忽略
    )
    saved = repo.save(new, expected_version=None)
    assert saved.version == 1
    try:
        repo.save(new, expected_version=None)
    except ConcurrencyConflict:
        pass
    else:
        raise AssertionError("重复新建应抛 ConcurrencyConflict")
    moved = repo.save(
        saved.model_copy(
            update={
                "state": ReviewState.CONFIRM,
                "matched_rules": ["R3", "R6"],
                "suggestion": ReviewAdvice.AGREE_RETURN,
                "requires_manager": True,
                "risk_flags": ["金额超过上限"],
                "reason": "签收 5 天，支持七天无理由",
                "verify_started_at": NOW,
                "action_result": {"ok": True},
                "updated_at": NOW,
            }
        ),
        expected_version=1,
    )
    assert moved.version == 2
    got = repo.get(SHOP_A, "as_2002")
    assert got == moved and got.matched_rule == "R3,R6"
    try:
        repo.save(moved, expected_version=1)  # 过期版本
    except ConcurrencyConflict:
        pass
    else:
        raise AssertionError("过期版本写入应抛 ConcurrencyConflict")
    try:
        repo.save(
            AfterSaleReview(
                shop_id=SHOP_A,
                aftersale_id="as_nope",
                platform_status=AfterSaleStatus.PENDING_REVIEW,
                state=ReviewState.NEW,
            ),
            expected_version=3,
        )
    except ConcurrencyConflict:
        pass
    else:
        raise AssertionError("更新不存在的记录应抛 ConcurrencyConflict")
    repo.save(
        AfterSaleReview(
            shop_id=SHOP_A,
            aftersale_id="as_2001",
            platform_status=AfterSaleStatus.PENDING_REVIEW,
            state=ReviewState.SUGGEST,
        ),
        expected_version=None,
    )
    repo.save(
        AfterSaleReview(
            shop_id=SHOP_B,
            aftersale_id="as_2002",
            platform_status=AfterSaleStatus.PENDING_REVIEW,
            state=ReviewState.CONFIRM,
        ),
        expected_version=None,
    )
    assert [r.aftersale_id for r in repo.list_by_state(SHOP_A)] == ["as_2001", "as_2002"]
    assert [r.aftersale_id for r in repo.list_by_state(SHOP_A, [ReviewState.CONFIRM])] == [
        "as_2002"
    ]
    assert repo.get(SHOP_B, "as_2002").state == ReviewState.CONFIRM
    assert repo.get(SHOP_A, "as_2002").state == ReviewState.CONFIRM
    assert repo.get(SHOP_B, "as_2002").version == 1
    assert repo.purge_shop(SHOP_B) == 1 and repo.get(SHOP_B, "as_2002") is None


# ---------------------------------------------------------------- 授权


def check_shop_auth_repository(make: Callable[[], ShopAuthRepository]) -> None:
    repo = make()
    auth = ShopAuth(
        shop_id=SHOP_A,
        access_token=SecretStr("tok-synthetic-not-real"),
        scopes=["order.read", "logistics.read"],
        expires_at=NOW + timedelta(days=30),
    )
    repo.save(auth)
    got = repo.get(SHOP_A)
    assert got is not None and got.access_token.get_secret_value() == "tok-synthetic-not-real"
    assert "tok-synthetic" not in repr(got)
    assert got.scopes == ["order.read", "logistics.read"]
    assert repo.get(SHOP_B) is None
    repo.save(auth.model_copy(update={"access_token": SecretStr("tok-2")}))
    assert repo.get(SHOP_A).access_token.get_secret_value() == "tok-2"
    repo.delete(SHOP_A)
    assert repo.get(SHOP_A) is None
    repo.delete(SHOP_A)  # 重复删除不报错


# ---------------------------------------------------------------- 串店测试（§9.2）


@dataclass
class RepoSet:
    knowledge: KnowledgeRepository
    conversations: ConversationRepository
    suggestions: SuggestionRepository
    feedback: FeedbackRepository
    reviews: AfterSaleReviewRepository
    auth: ShopAuthRepository


def check_cross_shop_isolation(make: Callable[[], RepoSet]) -> None:
    """写入 B 店铺的每一种数据，再用 A 店铺的身份逐一尝试读取：必须读不到任何 B 的数据。"""
    repos = make()
    world = build_demo_world()
    ids = seed_knowledge(repos.knowledge, world)
    b_faq = ids[(SHOP_B, "发什么快递")]
    repos.knowledge.set_knowledge_embedding(SHOP_B, b_faq, _unit([1, 0, 0, 0]))
    repos.conversations.add_message(
        make_message(
            "B 店的消息", world=world, shop_id=SHOP_B, msg_id="m_b1", conversation_id="conv_b"
        )
    )
    b_sug = repos.suggestions.add(_suggestion(SHOP_B, "conv_b", minutes=1, text="B 店推荐"))
    repos.feedback.add_gap(KnowledgeGap(shop_id=SHOP_B, missing="policy:gift", created_at=NOW))
    repos.feedback.add_label(LabeledSample(shop_id=SHOP_B, text="B", created_at=NOW))
    repos.reviews.save(
        AfterSaleReview(
            shop_id=SHOP_B,
            aftersale_id="as_9001",
            platform_status=AfterSaleStatus.PENDING_REVIEW,
            state=ReviewState.NEW,
        ),
        expected_version=None,
    )
    repos.auth.save(
        ShopAuth(
            shop_id=SHOP_B,
            access_token=SecretStr("tok-b"),
            scopes=["order.read"],
            expires_at=NOW + timedelta(days=1),
        )
    )

    k = repos.knowledge
    assert k.get_knowledge(SHOP_A, b_faq) is None
    assert all(x.shop_id == SHOP_A for x in k.list_knowledge(SHOP_A, status=None))
    assert all(x.shop_id == SHOP_A for x, _ in k.search_knowledge(SHOP_A, _unit([1, 0, 0, 0])))
    assert all(p.shop_id == SHOP_A for p in k.get_policies(SHOP_A).values())
    assert k.get_item_card(SHOP_A, "i_555").title == "法式碎花连衣裙"  # 同 item_id，读到的是 A 的
    assert all(c.shop_id == SHOP_A for c in k.list_item_cards(SHOP_A))
    assert repos.conversations.get_conversation(SHOP_A, "conv_b") is None
    assert repos.conversations.recent_messages(SHOP_A, "conv_b") == []
    assert repos.suggestions.get(SHOP_A, b_sug.id) is None
    assert repos.suggestions.list(SHOP_A) == []
    assert repos.feedback.list_gaps(SHOP_A) == [] and repos.feedback.list_labels(SHOP_A) == []
    assert repos.reviews.get(SHOP_A, "as_9001") is None
    assert repos.reviews.list_by_state(SHOP_A) == []
    assert repos.auth.get(SHOP_A) is None
    # B 自己读得到（证明数据确实写进去了）
    assert repos.suggestions.get(SHOP_B, b_sug.id) is not None
    assert repos.reviews.get(SHOP_B, "as_9001") is not None
