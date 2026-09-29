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
    QualityReviewStatus,
    ReviewAdvice,
    ReviewState,
    Role,
    SuggestionStatus,
)
from rodit.core.errors import ConcurrencyConflict
from rodit.core.models import (
    AfterSaleReview,
    ItemCard,
    KnowledgeGap,
    KnowledgeItem,
    LabeledSample,
    PolicyEntry,
    QualityReview,
    QualityScores,
    ShopAuth,
    SuggestionRecord,
)
from rodit.core.ports import (
    RECALL_TOMBSTONE_TTL,
    AfterSaleReviewRepository,
    ConversationRepository,
    FeedbackRepository,
    KnowledgeRepository,
    QualityReviewRepository,
    ShopAuthRepository,
    SuggestionRepository,
)
from rodit.testing.fakes import FakeClock, make_message
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


def check_knowledge_status_compare_and_set(make: Callable[[], KnowledgeRepository]) -> None:
    """set_knowledge_status 的 expected_status：当前状态不等于它就抛 ConcurrencyConflict、不写入。

    多实例部署时商家子账号 A 审核通过、子账号 B 同时驳回同一条候选知识：只能有一个生效，
    被驳回的候选不能因为"后写的赢"又变成已审核、拿去回答买家。
    """
    repo = make()

    def candidate(question: str) -> KnowledgeItem:
        return repo.add_knowledge(
            KnowledgeItem(
                shop_id=SHOP_A,
                question=question,
                answer="亲，发货前可以改哦~",
                source=KnowledgeSource.ADOPTED,
                status=KnowledgeStatus.CANDIDATE,
            )
        )

    k = candidate("能改地址吗")
    ok = repo.set_knowledge_status(
        SHOP_A,
        k.id,
        KnowledgeStatus.APPROVED,
        approved_by="merchant_a",
        expected_status=KnowledgeStatus.CANDIDATE,
    )
    assert ok.status == KnowledgeStatus.APPROVED and ok.approved_by == "merchant_a"
    try:  # 另一个实例还以为它是候选，要驳回
        repo.set_knowledge_status(
            SHOP_A, k.id, KnowledgeStatus.RETIRED, expected_status=KnowledgeStatus.CANDIDATE
        )
    except ConcurrencyConflict:
        pass
    else:
        raise AssertionError("状态已变化时比较后更新应抛 ConcurrencyConflict")
    got = repo.get_knowledge(SHOP_A, k.id)
    assert got.status == KnowledgeStatus.APPROVED and got.approved_by == "merchant_a"

    # 期望状态对得上：照常更新
    retired = repo.set_knowledge_status(
        SHOP_A, k.id, KnowledgeStatus.RETIRED, expected_status=KnowledgeStatus.APPROVED
    )
    assert retired.status == KnowledgeStatus.RETIRED

    # 不存在、或属于别的店铺：仍是 KeyError（不暴露别的店铺有没有这条），不是 ConcurrencyConflict
    other = candidate("发什么快递")
    for shop_id, kid in ((SHOP_B, other.id), (SHOP_A, 987654)):
        try:
            repo.set_knowledge_status(
                shop_id, kid, KnowledgeStatus.APPROVED, expected_status=KnowledgeStatus.CANDIDATE
            )
        except KeyError:
            pass
        else:
            raise AssertionError("不存在或跨店的比较后更新应抛 KeyError")
    assert repo.get_knowledge(SHOP_A, other.id).status == KnowledgeStatus.CANDIDATE

    # 不带 expected_status：无条件更新（原有语义）
    assert (
        repo.set_knowledge_status(SHOP_A, other.id, KnowledgeStatus.RETIRED).status
        == KnowledgeStatus.RETIRED
    )


def check_candidate_uniqueness(make: Callable[[], KnowledgeRepository]) -> None:
    """等商家审核的候选知识不重复（§5.8）：同一店铺、同样的问题和回答、同一商品、同一问题类型的
    CANDIDATE 只能有一条。多个实例同时把同一条编辑回复变成候选时由存储保证（PostgreSQL 部分唯一
    索引，内存实现在同一把锁里检查）：后到的 add_knowledge 抛 ConcurrencyConflict、不写入。"""
    repo = make()

    def candidate(**kw: object) -> KnowledgeItem:
        base: dict[str, object] = {
            "shop_id": SHOP_A,
            "question": "能改地址吗",
            "answer": "亲，发货前可以改哦~",
            "source": KnowledgeSource.ADOPTED,
            "status": KnowledgeStatus.CANDIDATE,
            "item_id": "i_555",
            "intent": Intent.MODIFY_ORDER,
            "created_at": NOW,
        }
        base.update(kw)
        return KnowledgeItem.model_validate(base)

    def conflicts(item: KnowledgeItem) -> bool:
        try:
            repo.add_knowledge(item)
        except ConcurrencyConflict:
            return True
        return False

    first = repo.add_knowledge(candidate())
    assert conflicts(candidate())  # 同样的候选已在等审核
    assert [k.id for k in repo.list_knowledge(SHOP_A, status=KnowledgeStatus.CANDIDATE)] == [
        first.id
    ]
    # 任何一项不同都是另一条候选
    for other in (
        candidate(answer="亲，发货前可以改的"),
        candidate(question="地址能改吗"),
        candidate(item_id="i_777"),
        candidate(item_id=None),
        candidate(intent=Intent.SHIPPING_POLICY),
        candidate(intent=None),
        candidate(shop_id=SHOP_B),
    ):
        assert not conflicts(other), other
    assert conflicts(candidate(item_id=None))  # 没有商品、没有类型的也按"相同"去重
    assert conflicts(candidate(intent=None))
    # 只管等审核的：已通过 / 已下线的同样内容不挡新候选，商家自己录的同样 FAQ 也不挡
    repo.set_knowledge_status(SHOP_A, first.id, KnowledgeStatus.APPROVED, approved_by="merchant")
    second = repo.add_knowledge(candidate())
    assert not conflicts(candidate(status=KnowledgeStatus.APPROVED))
    # 改回候选时撞上等审核的同样内容：比较后更新不写入
    try:
        repo.set_knowledge_status(SHOP_A, first.id, KnowledgeStatus.CANDIDATE)
    except ConcurrencyConflict:
        pass
    else:
        raise AssertionError("改回候选会和等审核的同样候选重复，应抛 ConcurrencyConflict")
    assert repo.get_knowledge(SHOP_A, first.id).status == KnowledgeStatus.APPROVED
    repo.set_knowledge_status(SHOP_A, second.id, KnowledgeStatus.RETIRED)
    assert repo.set_knowledge_status(SHOP_A, first.id, KnowledgeStatus.CANDIDATE).status == (
        KnowledgeStatus.CANDIDATE
    )


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
    check_conversation_recall(make)


def check_conversation_recall(make: Callable[[], ConversationRepository]) -> None:
    """撤回（§5.3）：消息不再出现在最近几轮里、原文不留；msg_id 仍然去重；按店铺、会话隔离。"""
    repo = make()
    world = build_demo_world()
    conv = "conv_recall"
    m1 = make_message(
        "我的手机号13812345678，这件有M码吗",
        world=world,
        msg_id="m_r1",
        conversation_id=conv,
        item_id="i_555",
        at=NOW - timedelta(seconds=30),
    )
    m2 = make_message("那黑色的吧", world=world, msg_id="m_r2", conversation_id=conv, at=NOW)
    other_shop = make_message(
        "在吗", world=world, shop_id=SHOP_B, msg_id="m_r1", conversation_id=conv
    )
    assert repo.add_message(m1) and repo.add_message(m2) and repo.add_message(other_shop)
    # 别的会话、别的店铺撤回不了这条
    assert repo.recall_message(SHOP_A, "conv_other", "m_r1") is False
    assert repo.recall_message(SHOP_A, conv, "m_missing") is False
    assert repo.recall_message(SHOP_A, conv, "m_r1") is True
    assert repo.recall_message(SHOP_A, conv, "m_r1") is False  # 已经撤回过
    msgs = repo.recent_messages(SHOP_A, conv)
    assert [m.msg_id for m in msgs] == ["m_r2"]
    assert all("13812345678" not in (m.text or "") for m in msgs)
    assert [m.msg_id for m in repo.recent_messages(SHOP_A, conv, limit=1)] == ["m_r2"]
    assert repo.add_message(m1) is False  # 平台重推被撤回的消息：仍然去重
    assert [m.msg_id for m in repo.recent_messages(SHOP_B, conv)] == ["m_r1"]
    assert repo.purge_shop(SHOP_A) > 0
    assert repo.recent_messages(SHOP_A, conv) == []


def check_recall_before_message(make: Callable[[FakeClock], ConversationRepository]) -> None:
    """撤回先于被撤回的消息到达（§8.3 平台推送乱序、§5.3 撤回的消息不再进上下文）。

    本店没见过的 msg_id 被撤回时记一条墓碑（按店铺 + 会话），recall_message 仍返回 False；
    墓碑有效期（RECALL_TOMBSTONE_TTL）内这条消息到达时丢弃：不保存、add_message / add_outgoing
    返回 False（引擎当作重复消息，不回答），之后按撤回过的消息去重。过期的墓碑不再生效；
    取消订购时和会话一起删除。make 用传入的时钟判断过期。
    """
    clock = FakeClock(NOW)
    repo = make(clock)
    world = build_demo_world()
    conv = "conv_early_recall"

    def buyer(msg_id: str, text: str = "在吗", **kw: object) -> object:
        kw.setdefault("conversation_id", conv)
        kw.setdefault("at", clock.now())
        return make_message(text, world=world, msg_id=msg_id, **kw)  # type: ignore[arg-type]

    assert repo.recall_message(SHOP_A, conv, "m_e1") is False  # 没有消息被撤回
    assert repo.recall_message(SHOP_A, conv, "m_e1") is False  # 平台重推的撤回：幂等
    early = buyer("m_e1", "我的手机号13812345678，这件有M码吗")
    assert repo.add_message(early) is False  # 到达时丢弃：不保存、不回答
    assert repo.recent_messages(SHOP_A, conv) == []
    assert repo.get_conversation(SHOP_A, conv) is not None  # 会话照常建立
    assert repo.add_message(early) is False  # 平台重推：仍然去重
    clock.advance(seconds=5)
    assert repo.add_message(buyer("m_e2", "那黑色的吧")) is True
    assert [m.msg_id for m in repo.recent_messages(SHOP_A, conv)] == ["m_e2"]

    # 客服 / 机器人发出的消息也可能先收到撤回
    assert repo.recall_message(SHOP_A, conv, "out_e1") is False
    for _ in range(2):  # 第二次是平台重推
        assert not repo.add_outgoing(
            shop_id=SHOP_A,
            conversation_id=conv,
            msg_id="out_e1",
            role=Role.AGENT,
            text="有的亲",
            at=clock.now(),
        )
    assert [m.msg_id for m in repo.recent_messages(SHOP_A, conv)] == ["m_e2"]

    # 按店铺 + 会话：别的会话、别的店铺同样的 msg_id 照常保存
    assert repo.recall_message(SHOP_A, conv, "m_e3") is False
    assert repo.add_message(buyer("m_e3", conversation_id="conv_early_other")) is True
    assert repo.add_message(buyer("m_e3", shop_id=SHOP_B)) is True
    assert [m.msg_id for m in repo.recent_messages(SHOP_B, conv)] == ["m_e3"]
    # 本店已经见过的 msg_id（在别的会话里）：不记墓碑，也不影响那条消息
    assert repo.recall_message(SHOP_A, "conv_early_third", "m_e3") is False
    assert [m.msg_id for m in repo.recent_messages(SHOP_A, "conv_early_other")] == ["m_e3"]

    # 墓碑过期后到达的消息照常保存
    assert repo.recall_message(SHOP_A, conv, "m_e4") is False
    clock.advance(seconds=RECALL_TOMBSTONE_TTL.total_seconds() + 1)
    assert repo.add_message(buyer("m_e4", "什么时候发货")) is True
    assert [m.msg_id for m in repo.recent_messages(SHOP_A, conv)] == ["m_e2", "m_e4"]

    # 取消订购：墓碑一起删掉（之后同一个 msg_id 是新消息）
    assert repo.recall_message(SHOP_A, conv, "m_e5") is False
    assert repo.purge_shop(SHOP_A) > 0
    assert repo.add_message(buyer("m_e5")) is True
    assert [m.msg_id for m in repo.recent_messages(SHOP_B, conv)] == ["m_e3"]


# ---------------------------------------------------------------- 推荐


def _suggestion(shop_id: str, conv: str, *, minutes: float, text: str = "推荐") -> SuggestionRecord:
    return SuggestionRecord(
        shop_id=shop_id,
        conversation_id=conv,
        msg_id="m_s",
        question_text="发什么快递",
        item_id="i_555",
        intent=Intent.SHIPPING_POLICY,
        intent_confidence=0.875,
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
    assert a1.intent_confidence == 0.875  # §13.5 重算阈值用的置信度原样存取
    old = _suggestion(SHOP_B, "c9", minutes=9, text="旧")
    no_conf = repo.add(old.model_copy(update={"intent_confidence": None}))
    assert repo.get(SHOP_B, no_conf.id).intent_confidence is None
    repo.purge_shop(SHOP_B)
    b1 = repo.add(_suggestion(SHOP_B, "c1", minutes=1, text="b1"))
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
    # 比较后更新：状态已不是期望值时不写入
    try:
        repo.update(
            a1.model_copy(update={"status": SuggestionStatus.EXPIRED}),
            expected_status=SuggestionStatus.SHOWN,
        )
    except ConcurrencyConflict:
        pass
    else:
        raise AssertionError("状态已变化时比较后更新应抛 ConcurrencyConflict")
    assert repo.get(SHOP_A, a1.id).status == SuggestionStatus.EDITED_SENT
    ok = repo.update(
        a2.model_copy(update={"status": SuggestionStatus.SENT}),
        expected_status=SuggestionStatus.SHOWN,
    )
    assert ok.status == SuggestionStatus.SENT
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
            predicted_confidence=0.625,
            true_intent=Intent.SHIPPING_POLICY,
            source="feedback_reject",
            created_at=NOW,
        )
    )
    repo.add_label(
        LabeledSample(shop_id=SHOP_B, text="在吗", true_intent=Intent.GREETING, created_at=NOW)
    )
    assert [s.text for s in repo.list_labels(SHOP_A)] == ["发啥快递"]
    assert repo.list_labels(SHOP_A)[0].predicted_confidence == 0.625
    exported = {s.shop_id: s for s in repo.export_training_labels()}
    assert set(exported) == {SHOP_A, SHOP_B}
    assert exported[SHOP_A].predicted_confidence == 0.625
    assert exported[SHOP_B].predicted_confidence is None
    assert repo.purge_shop(SHOP_B) == 2
    assert repo.list_gaps(SHOP_B) == [] and repo.list_labels(SHOP_B) == []
    assert len(repo.list_gaps(SHOP_A)) == 2


# ---------------------------------------------------------------- 每周人工复核（§13.5）


def _quality(shop_id: str, suggestion_id: int, *, period: str = "2026-W40") -> QualityReview:
    return QualityReview(
        shop_id=shop_id,
        period=period,
        suggestion_id=suggestion_id,
        intent=Intent.SHIPPING_POLICY,
        sampled_at=NOW,
    )


def check_quality_review_repository(make: Callable[[], QualityReviewRepository]) -> None:
    repo = make()
    a1, a2, b1 = repo.add_many([_quality(SHOP_A, 11), _quality(SHOP_A, 12), _quality(SHOP_B, 11)])
    assert len({a1.id, a2.id, b1.id}) == 3 and a1.status == QualityReviewStatus.PENDING
    assert repo.get(SHOP_A, a1.id) == a1 and repo.get(SHOP_B, a1.id) is None
    # 幂等：同一店铺、周期、推荐不重复写，返回已有的那条；新周期是新样本
    again, other_week = repo.add_many(
        [_quality(SHOP_A, 11), _quality(SHOP_A, 11, period="2026-W41")]
    )
    assert again == a1 and other_week.id not in {a1.id, a2.id, b1.id}
    assert [r.id for r in repo.list(SHOP_A, period="2026-W40")] == [a1.id, a2.id]
    assert len(repo.list(SHOP_A)) == 3
    for bad in (_quality(SHOP_A, 13).model_copy(update={"id": 5}),):
        try:
            repo.add_many([bad])
        except ValueError:
            pass
        else:
            raise AssertionError("带 id 的样本不能当新样本写入")
    scores = QualityScores(factual=2, relevance=2, compliance=2, expression=1)
    done = repo.record(
        SHOP_A,
        a1.id,
        status=QualityReviewStatus.SCORED,
        scores=scores,
        reviewer="qa_01",
        note="语气可以更自然",
        reviewed_at=NOW + timedelta(hours=1),
    )
    assert done.status == QualityReviewStatus.SCORED and done.scores == scores
    assert repo.get(SHOP_A, a1.id) == done
    assert repo.get(SHOP_A, a1.id).reviewed_at == NOW + timedelta(hours=1)
    # 已复核的再抽中也不会被重置成 pending
    assert repo.add_many([_quality(SHOP_A, 11)])[0].status == QualityReviewStatus.SCORED
    # 改判：后一次覆盖
    skipped = repo.record(
        SHOP_A,
        a1.id,
        status=QualityReviewStatus.SKIPPED,
        scores=None,
        reviewer="qa_02",
        note=None,
        reviewed_at=NOW + timedelta(hours=2),
    )
    assert skipped.status == QualityReviewStatus.SKIPPED and skipped.scores is None
    assert skipped.reviewer == "qa_02"
    assert [r.id for r in repo.list(SHOP_A, status=QualityReviewStatus.PENDING)] == [
        a2.id,
        other_week.id,
    ]
    for kwargs in (
        {"status": QualityReviewStatus.SCORED, "scores": None},  # scored 必须有分数
        {"status": QualityReviewStatus.PENDING, "scores": None},  # 不能改回 pending
    ):
        try:
            repo.record(SHOP_A, a2.id, reviewer="qa_01", note=None, reviewed_at=NOW, **kwargs)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法的复核结果应抛 ValueError：{kwargs}")
    assert repo.get(SHOP_A, a2.id).status == QualityReviewStatus.PENDING
    try:  # 别的店铺的样本：当作不存在
        repo.record(
            SHOP_A,
            b1.id,
            status=QualityReviewStatus.SKIPPED,
            scores=None,
            reviewer="qa_01",
            note=None,
            reviewed_at=NOW,
        )
    except KeyError:
        pass
    else:
        raise AssertionError("用别的店铺身份复核应抛 KeyError")
    assert repo.get(SHOP_B, b1.id).status == QualityReviewStatus.PENDING
    assert repo.purge_shop(SHOP_B) == 1 and repo.list(SHOP_B) == []
    assert repo.purge_shop(SHOP_B) == 0 and len(repo.list(SHOP_A)) == 3


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
                "platform_status_version": "1700000000123",
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
    assert repo.delete(SHOP_A) is True
    assert repo.get(SHOP_A) is None
    assert repo.delete(SHOP_A) is False  # 重复删除不报错，返回没有删到
    assert repo.delete(SHOP_B) is False


# ---------------------------------------------------------------- 串店测试（§9.2）


@dataclass
class RepoSet:
    knowledge: KnowledgeRepository
    conversations: ConversationRepository
    suggestions: SuggestionRepository
    feedback: FeedbackRepository
    reviews: AfterSaleReviewRepository
    auth: ShopAuthRepository
    quality_reviews: QualityReviewRepository


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
    (b_quality,) = repos.quality_reviews.add_many([_quality(SHOP_B, b_sug.id or 1)])

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
    assert repos.quality_reviews.get(SHOP_A, b_quality.id) is None
    assert repos.quality_reviews.list(SHOP_A) == []
    # B 自己读得到（证明数据确实写进去了）
    assert repos.suggestions.get(SHOP_B, b_sug.id) is not None
    assert repos.reviews.get(SHOP_B, "as_9001") is not None
    assert repos.quality_reviews.get(SHOP_B, b_quality.id) is not None


# ---------------------------------------------------------------- 取消订购后的清理（§8.4）


def check_purge_is_idempotent(make: Callable[[], RepoSet]) -> None:
    """purge_shop 删掉这家店铺的全部数据、返回删除条数，别的店铺一行不动；再调一次返回 0
    （AuthGuard.revoke 中途失败时可以重试）。两种实现对同样的数据返回同样的条数。"""
    repos = make()
    world = build_demo_world()
    seed_knowledge(repos.knowledge, world)
    for shop in (SHOP_A, SHOP_B):
        for i in (1, 2):
            repos.conversations.add_message(
                make_message(
                    "在吗",
                    world=world,
                    shop_id=shop,
                    msg_id=f"m_p{i}",
                    conversation_id="conv_purge",
                    at=NOW + timedelta(seconds=i),
                )
            )
        repos.conversations.recall_message(shop, "conv_purge", "m_p_later")  # 撤回墓碑
        for minutes in (1, 2):
            repos.suggestions.add(_suggestion(shop, "conv_purge", minutes=minutes))
        repos.feedback.add_gap(KnowledgeGap(shop_id=shop, missing="policy:gift", created_at=NOW))
        repos.feedback.add_label(LabeledSample(shop_id=shop, text="在吗", created_at=NOW))
        repos.reviews.save(
            AfterSaleReview(
                shop_id=shop,
                aftersale_id="as_purge",
                platform_status=AfterSaleStatus.PENDING_REVIEW,
                state=ReviewState.NEW,
            ),
            expected_version=None,
        )
        repos.quality_reviews.add_many([_quality(shop, 1), _quality(shop, 2)])
    k = repos.knowledge
    knowledge_rows = (
        1
        + len(k.get_policies(SHOP_A))
        + len(k.list_item_cards(SHOP_A))
        + len(k.list_knowledge(SHOP_A, status=None))
    )
    purgers = {
        "knowledge": (repos.knowledge, knowledge_rows),
        "conversations": (repos.conversations, 4),  # 2 条消息 + 会话 + 撤回墓碑
        "suggestions": (repos.suggestions, 2),
        "feedback": (repos.feedback, 2),
        "reviews": (repos.reviews, 1),
        "quality_reviews": (repos.quality_reviews, 2),
    }
    for name, (repo, expected) in purgers.items():
        assert repo.purge_shop(SHOP_A) == expected, name  # type: ignore[attr-defined]
    for name, (repo, _) in purgers.items():
        assert repo.purge_shop(SHOP_A) == 0, name  # type: ignore[attr-defined]
    assert k.get_shop(SHOP_A) is None and k.list_knowledge(SHOP_A, status=None) == []
    assert repos.conversations.recent_messages(SHOP_A, "conv_purge") == []
    assert repos.suggestions.list(SHOP_A) == [] and repos.reviews.get(SHOP_A, "as_purge") is None
    assert repos.feedback.list_gaps(SHOP_A) == [] and repos.feedback.list_labels(SHOP_A) == []
    # 别的店铺不受影响（墓碑也还在：B 的 m_p_later 到达时照样丢弃）
    assert k.get_shop(SHOP_B) is not None
    assert len(repos.conversations.recent_messages(SHOP_B, "conv_purge")) == 2
    assert (
        repos.conversations.add_message(
            make_message(
                "在吗",
                world=world,
                shop_id=SHOP_B,
                msg_id="m_p_later",
                conversation_id="conv_purge",
            )
        )
        is False
    )
    assert len(repos.suggestions.list(SHOP_B)) == 2 and repos.reviews.get(SHOP_B, "as_purge")
    assert len(repos.feedback.list_gaps(SHOP_B)) == 1
    assert repos.quality_reviews.list(SHOP_A) == []
    assert len(repos.quality_reviews.list(SHOP_B)) == 2


# ---------------------------------------------------------------- 文本清洗（NUL 等控制字符）

DIRTY_TEXT = "这件\x00是纯棉\x07的吗\x1b\n尺码\t偏小吗\r"
"""带 NUL 和其他 C0 控制字符的文字（换行、制表符要保留）。"""
CLEAN_TEXT = "这件是纯棉的吗\n尺码\t偏小吗"


def check_text_sanitation(make: Callable[[], RepoSet]) -> None:
    """NUL 和其他 C0 控制字符（换行、制表符除外）在模型边界去掉（core.models.strip_control_chars）。

    PostgreSQL 的 TEXT / JSONB 不接受 NUL，内存实现照单全收；两种实现对同一段文字的写入结果
    必须一致：都写得进去，读回的都是去掉控制字符后的文字。模型用构造函数或 model_validate
    创建时清洗（model_copy 不做校验，调用方自己用构造函数）。
    """
    repos = make()
    world = build_demo_world()

    msg = make_message(DIRTY_TEXT, world=world, msg_id="m_nul")
    assert msg.text == CLEAN_TEXT
    assert repos.conversations.add_message(msg) is True
    stored = repos.conversations.recent_messages(SHOP_A, msg.conversation_id)
    assert [m.text for m in stored] == [CLEAN_TEXT]
    # 客服 / 机器人发出的文字（反馈、平台回声）同样清洗
    assert repos.conversations.add_outgoing(
        shop_id=SHOP_A,
        conversation_id=msg.conversation_id,
        msg_id="agent_nul",
        role=Role.AGENT,
        text="有的\x00亲",
        at=msg.received_at + timedelta(seconds=1),
    )
    stored = repos.conversations.recent_messages(SHOP_A, msg.conversation_id)
    assert [m.text for m in stored] == [CLEAN_TEXT, "有的亲"]

    k = repos.knowledge.add_knowledge(
        KnowledgeItem(
            shop_id=SHOP_A,
            question=DIRTY_TEXT,
            answer="亲\x00，是纯棉的哦",
            source=KnowledgeSource.ADOPTED,
            status=KnowledgeStatus.CANDIDATE,
        )
    )
    got = repos.knowledge.get_knowledge(SHOP_A, k.id)
    assert got is not None and got.question == CLEAN_TEXT and got.answer == "亲，是纯棉的哦"

    repos.knowledge.upsert_policy(
        PolicyEntry(shop_id=SHOP_A, key="ship_time", value="付款后\x00 48 小时发货", updated_at=NOW)
    )
    assert repos.knowledge.get_policies(SHOP_A)["ship_time"].value == "付款后 48 小时发货"
    card = ItemCard(
        shop_id=SHOP_A,
        platform=world.shop(SHOP_A).platform,
        item_id="i_nul",
        title="碎花\x00连衣裙",
        attributes={"面料": "棉\x0095%"},
        size_advice="偏小\x00一码",
        care=["手洗\x00"],
        detail_text=DIRTY_TEXT,
        updated_at=NOW,
    )
    assert card.title == "碎花连衣裙" and card.detail_text == CLEAN_TEXT
    assert card.attributes == {"面料": "棉95%"} and card.care == ["手洗"]
    repos.knowledge.upsert_item_card(card)
    assert repos.knowledge.get_item_card(SHOP_A, "i_nul") == card

    dirty_record = {
        **_suggestion(SHOP_A, "c_nul", minutes=1).model_dump(),
        "question_text": DIRTY_TEXT,
        "text": "亲\x00，是的",
    }
    rec = repos.suggestions.add(SuggestionRecord.model_validate(dirty_record))
    assert rec.question_text == CLEAN_TEXT and rec.text == "亲，是的"
    edited = SuggestionRecord.model_validate(
        {**rec.model_dump(), "status": SuggestionStatus.EDITED_SENT, "edited_text": "亲\x00，是"}
    )
    repos.suggestions.update(edited)
    got_rec = repos.suggestions.get(SHOP_A, rec.id)
    assert got_rec is not None and got_rec.edited_text == "亲，是"
    assert got_rec.question_text == CLEAN_TEXT and got_rec.text == "亲，是的"

    repos.feedback.add_gap(
        KnowledgeGap(
            shop_id=SHOP_A, missing="policy:gift", question_text=DIRTY_TEXT, created_at=NOW
        )
    )
    repos.feedback.add_label(LabeledSample(shop_id=SHOP_A, text=DIRTY_TEXT, created_at=NOW))
    assert [g.question_text for g in repos.feedback.list_gaps(SHOP_A)] == [CLEAN_TEXT]
    assert [s.text for s in repos.feedback.list_labels(SHOP_A)] == [CLEAN_TEXT]

    saved = repos.reviews.save(
        AfterSaleReview(
            shop_id=SHOP_A,
            aftersale_id="as_nul",
            platform_status=AfterSaleStatus.PENDING_REVIEW,
            state=ReviewState.NEW,
            reason="买家说\x00没收到",
            action_result={"note": "已\x00处理", "steps": ["a\x00b"]},
        ),
        expected_version=None,
    )
    got_review = repos.reviews.get(SHOP_A, "as_nul")
    assert got_review is not None and got_review == saved
    assert got_review.reason == "买家说没收到"
    assert got_review.action_result == {"note": "已处理", "steps": ["ab"]}

    (quality,) = repos.quality_reviews.add_many([_quality(SHOP_A, rec.id or 1)])
    scored = repos.quality_reviews.record(
        SHOP_A,
        quality.id or 0,
        status=QualityReviewStatus.SKIPPED,
        scores=None,
        reviewer="qa_01",
        note=DIRTY_TEXT,
        reviewed_at=NOW,
    )
    assert scored.note == CLEAN_TEXT
    got_quality = repos.quality_reviews.get(SHOP_A, quality.id or 0)
    assert got_quality is not None and got_quality.note == CLEAN_TEXT
