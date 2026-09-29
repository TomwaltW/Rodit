"""商家控制台的知识编辑与店铺配置（§5.2 知识中心、§5.2.1、§5.2.2、§9.3）：领域逻辑，不经过 HTTP。

全部合成数据（演示世界），内存仓储，不联网。
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.config import EngineConfig
from rodit.core.enums import Intent, KnowledgeSource, KnowledgeStatus, PolicyKey, ShopStatus
from rodit.core.errors import ActionNotAllowed, ConcurrencyConflict, ShopInactive
from rodit.core.models import ItemCard, Platform
from rodit.inmemory import InMemoryKnowledgeRepository
from rodit.knowledge.admin import (
    LOCKED_CONFIG_FIELDS,
    MAX_ANSWER_CHARS,
    MERCHANT_CONFIG_FIELDS,
    FaqChanges,
    FaqCheck,
    FaqWarning,
    KnowledgeAdmin,
)
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import SHOP_A, SHOP_B, DemoWorld

MERCHANT = "owner_a"


class Kit:
    def __init__(
        self,
        repo: InMemoryKnowledgeRepository,
        world: DemoWorld,
        *,
        config: EngineConfig | None = None,
        checker: Any = None,
    ) -> None:
        self.repo = repo
        self.world = world
        self.clock = FakeClock(world.now)
        self.audit = InMemoryAuditSink()
        self.indexed: list[str] = []
        self.checks: list[FaqCheck] = []
        self.checker_result: list[FaqWarning] = []

        def default_checker(check: FaqCheck) -> list[FaqWarning]:
            self.checks.append(check)
            return list(self.checker_result)

        self.admin = KnowledgeAdmin(
            repo,
            self.audit,
            self.clock,
            config=config or EngineConfig(),
            on_faq_changed=self.indexed.append,
            answer_checker=checker or default_checker,
        )

    def events(self, action: AuditAction) -> list:
        return self.audit.list_events(SHOP_A, action=action.value)


@pytest.fixture
def kit(repo: InMemoryKnowledgeRepository, world: DemoWorld) -> Kit:
    return Kit(repo, world)


def _b_only_card(world: DemoWorld) -> ItemCard:
    return ItemCard(
        shop_id=SHOP_B,
        platform=Platform.TAOBAO,
        item_id="i_b_only",
        title="男士衬衫（只在店铺 B）",
        updated_at=world.now,
    )


def _all_text(kit: Kit) -> str:
    return json.dumps([e.detail for e in kit.audit.events], ensure_ascii=False)


# ================================================================ FAQ：录入


def test_create_faq_is_approved_by_the_merchant_and_indexed(kit: Kit):
    kit.clock.advance(minutes=5)
    change = kit.admin.create_faq(
        SHOP_A,
        question="  可以用花呗吗\x00 ",
        answer="亲，支持花呗分期付款哦~",
        intent=Intent.OTHER,
        merchant=MERCHANT,
    )
    item = change.item
    assert item.id is not None and change.retired is None
    assert item.question == "可以用花呗吗" and item.answer == "亲，支持花呗分期付款哦~"
    assert item.source == KnowledgeSource.MERCHANT and item.status == KnowledgeStatus.APPROVED
    assert item.approved_by == MERCHANT and item.created_at == kit.clock.now()
    assert kit.repo.get_knowledge(SHOP_A, item.id) == item
    assert kit.indexed == [SHOP_A]
    [event] = kit.events(AuditAction.KNOWLEDGE_CREATED)
    assert event.actor == MERCHANT
    assert event.detail["knowledge_id"] == item.id and event.detail["intent"] == "other"
    assert "花呗" not in _all_text(kit)  # 审计不放问答原文


@pytest.mark.parametrize(
    ("question", "answer", "message"),
    [
        ("", "亲，可以的", "问题不能为空"),
        (" \x00\t ", "亲，可以的", "问题不能为空"),
        ("可以吗", "  ", "回答不能为空"),
        ("可以吗", "好" * (MAX_ANSWER_CHARS + 1), "最多"),
        (None, "亲，可以的", "问题不能为空"),
    ],
)
def test_create_faq_rejects_empty_or_too_long_text(kit: Kit, question, answer, message):
    before = kit.repo.list_knowledge(SHOP_A, status=None)
    with pytest.raises(ValueError, match=message):
        kit.admin.create_faq(SHOP_A, question=question, answer=answer, merchant=MERCHANT)
    assert kit.repo.list_knowledge(SHOP_A, status=None) == before
    assert kit.audit.events == [] and kit.indexed == []


@pytest.mark.parametrize("merchant", ["", "  ", "system", "Bot", "auto"])
def test_create_faq_needs_a_real_person(kit: Kit, merchant: str):
    with pytest.raises(ValueError):
        kit.admin.create_faq(SHOP_A, question="在吗", answer="亲，在的", merchant=merchant)


def test_item_scoped_faq_needs_an_item_card_of_this_shop(kit: Kit, world: DemoWorld):
    kit.repo.upsert_item_card(_b_only_card(world))
    before = kit.repo.list_knowledge(SHOP_A, status=None)
    for item_id in ("i_b_only", "i_nope"):  # 别的店铺的商品和不存在一样
        with pytest.raises(ValueError, match="本店没有这个商品"):
            kit.admin.create_faq(
                SHOP_A,
                question="会起球吗",
                answer="亲，不容易起球",
                item_id=item_id,
                merchant=MERCHANT,
            )
    assert kit.repo.list_knowledge(SHOP_A, status=None) == before
    change = kit.admin.create_faq(
        SHOP_A, question="会起球吗", answer="亲，不容易起球哦", item_id="i_777", merchant=MERCHANT
    )
    assert change.item.item_id == "i_777"
    assert kit.checks[-1].item is not None and kit.checks[-1].item.item_id == "i_777"


def test_writes_to_a_deleted_or_suspended_shop_are_refused(kit: Kit):
    shop = kit.repo.get_shop(SHOP_A)
    kit.repo.upsert_shop(shop.model_copy(update={"status": ShopStatus.SUSPENDED}))
    with pytest.raises(ShopInactive):
        kit.admin.create_faq(SHOP_A, question="在吗", answer="亲，在的", merchant=MERCHANT)
    kit.repo.purge_shop(SHOP_A)
    calls: list = [
        lambda: kit.admin.create_faq(SHOP_A, question="在吗", answer="亲，在的", merchant=MERCHANT),
        lambda: kit.admin.set_policy(SHOP_A, "invoice", "可以开", merchant=MERCHANT),
        lambda: kit.admin.update_config(SHOP_A, {"tone": "专业"}, merchant=MERCHANT),
        lambda: kit.admin.update_item_card(SHOP_A, "i_555", {"care": []}, merchant=MERCHANT),
    ]
    for call in calls:
        with pytest.raises(ShopInactive):
            call()
    assert kit.repo.get_shop(SHOP_A) is None
    assert kit.repo.get_policies(SHOP_A) == {}
    assert kit.repo.list_item_cards(SHOP_A) == []
    assert kit.repo.list_knowledge(SHOP_A, status=None) == []
    assert kit.audit.events == []


# ================================================================ FAQ：修改


def test_edit_retires_the_old_item_and_creates_a_new_one(kit: Kit, faq_ids):
    old_id = faq_ids[(SHOP_A, "这款会透吗")]
    old = kit.repo.get_knowledge(SHOP_A, old_id)
    kit.clock.advance(hours=1)
    change = kit.admin.edit_faq(
        SHOP_A,
        old_id,
        FaqChanges.of(answer="亲，白色款强光下略透，建议搭配肤色内搭哦~"),
        merchant=MERCHANT,
    )
    new = change.item
    assert new.id != old_id and change.retired is not None and change.retired.id == old_id
    # 旧条目只改状态，内容不变：推荐记录里的 faq:{旧 id} 仍指向买家当时看到的话
    retired = kit.repo.get_knowledge(SHOP_A, old_id)
    assert retired.status == KnowledgeStatus.RETIRED
    assert (retired.question, retired.answer) == (old.question, old.answer)
    # 没改的字段保持原样（商品范围、问题类型、来源），审核人是这次修改的商家
    assert new.status == KnowledgeStatus.APPROVED and new.question == old.question
    assert new.item_id == "i_555" and new.intent == Intent.PRODUCT_SPEC
    assert new.source == old.source and new.approved_by == MERCHANT
    assert new.created_at == kit.clock.now()
    assert kit.indexed == [SHOP_A]
    [event] = kit.events(AuditAction.KNOWLEDGE_EDITED)
    assert event.detail["old_knowledge_id"] == old_id
    assert event.detail["knowledge_id"] == new.id
    assert event.detail["changed"] == ["answer"]
    assert "内搭" not in _all_text(kit)


def test_edit_can_clear_item_scope_and_intent(kit: Kit, faq_ids):
    old_id = faq_ids[(SHOP_A, "这款会透吗")]
    change = kit.admin.edit_faq(
        SHOP_A, old_id, FaqChanges.of(item_id=None, intent=None), merchant=MERCHANT
    )
    assert change.item.item_id is None and change.item.intent is None
    assert kit.events(AuditAction.KNOWLEDGE_EDITED)[0].detail["changed"] == ["item_id", "intent"]


def test_edit_refuses_non_approved_missing_and_foreign_items(kit: Kit, faq_ids):
    candidate = faq_ids[(SHOP_A, "可以寄到新疆吗")]
    retired = faq_ids[(SHOP_A, "双十一有活动吗")]
    foreign = faq_ids[(SHOP_B, "发什么快递")]
    changes = FaqChanges.of(answer="亲，改一下")
    for kid in (candidate, retired):
        with pytest.raises(ActionNotAllowed):
            kit.admin.edit_faq(SHOP_A, kid, changes, merchant=MERCHANT)
    for kid in (foreign, 9999):
        with pytest.raises(KeyError):
            kit.admin.edit_faq(SHOP_A, kid, changes, merchant=MERCHANT)
    assert kit.repo.get_knowledge(SHOP_B, foreign).status == KnowledgeStatus.APPROVED
    same = kit.repo.get_knowledge(SHOP_A, faq_ids[(SHOP_A, "发什么快递")])
    with pytest.raises(ValueError, match="没有变化"):
        kit.admin.edit_faq(SHOP_A, same.id, FaqChanges.of(answer=same.answer), merchant=MERCHANT)
    assert kit.repo.get_knowledge(SHOP_A, same.id).status == KnowledgeStatus.APPROVED
    with pytest.raises(ValueError):
        FaqChanges.of(merchant="someone")
    assert kit.audit.events == []


def test_edit_with_a_foreign_item_id_changes_nothing(kit: Kit, faq_ids, world: DemoWorld):
    kit.repo.upsert_item_card(_b_only_card(world))
    kid = faq_ids[(SHOP_A, "发什么快递")]
    with pytest.raises(ValueError, match="本店没有这个商品"):
        kit.admin.edit_faq(SHOP_A, kid, FaqChanges.of(item_id="i_b_only"), merchant=MERCHANT)
    assert kit.repo.get_knowledge(SHOP_A, kid).status == KnowledgeStatus.APPROVED


class _RacingRepo:
    """包一层内存仓储：set_knowledge_status 之前别的实例先把这条下线了（多实例并发修改）。"""

    def __init__(self, inner: InMemoryKnowledgeRepository) -> None:
        self._inner = inner
        self.race = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def set_knowledge_status(self, shop_id, knowledge_id, status, **kw):
        if self.race:
            self.race = False
            self._inner.set_knowledge_status(shop_id, knowledge_id, KnowledgeStatus.RETIRED)
        return self._inner.set_knowledge_status(shop_id, knowledge_id, status, **kw)


def test_concurrent_edit_loses_the_compare_and_set(repo, world: DemoWorld, faq_ids):
    racing = _RacingRepo(repo)
    kit = Kit(racing, world)  # type: ignore[arg-type]
    kid = faq_ids[(SHOP_A, "发什么快递")]
    before = len(repo.list_knowledge(SHOP_A, status=None))
    with pytest.raises(ActionNotAllowed, match="刷新"):
        kit.admin.edit_faq(SHOP_A, kid, FaqChanges.of(answer="亲，默认发中通哦"), merchant=MERCHANT)
    assert len(repo.list_knowledge(SHOP_A, status=None)) == before  # 没有多出一条
    assert kit.audit.events == []


class _FailingAddRepo:
    def __init__(self, inner: InMemoryKnowledgeRepository) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def add_knowledge(self, item):
        raise ConnectionError("数据库连接断开（模拟）")


def test_edit_restores_the_old_item_when_the_new_one_cannot_be_written(
    repo, world: DemoWorld, faq_ids
):
    kit = Kit(_FailingAddRepo(repo), world)  # type: ignore[arg-type]
    kid = faq_ids[(SHOP_A, "发什么快递")]
    with pytest.raises(ConnectionError):
        kit.admin.edit_faq(SHOP_A, kid, FaqChanges.of(answer="亲，默认发中通哦"), merchant=MERCHANT)
    assert repo.get_knowledge(SHOP_A, kid).status == KnowledgeStatus.APPROVED
    assert kit.audit.events == []


# ================================================================ FAQ：审核候选时修改


def test_approve_edited_without_changes_is_a_plain_approval(kit: Kit, faq_ids):
    kid = faq_ids[(SHOP_A, "可以寄到新疆吗")]
    candidate = kit.repo.get_knowledge(SHOP_A, kid)
    change = kit.admin.approve_edited(
        SHOP_A, kid, FaqChanges.of(answer=candidate.answer), merchant=MERCHANT
    )
    assert change.item.id == kid and change.retired is None
    assert change.item.status == KnowledgeStatus.APPROVED and change.item.approved_by == MERCHANT
    [event] = kit.events(AuditAction.KNOWLEDGE_APPROVED)
    assert event.detail["knowledge_id"] == kid and event.detail["edited"] is False
    assert kit.indexed == [SHOP_A]


def test_approve_edited_retires_the_candidate_and_approves_the_edited_copy(kit: Kit, faq_ids):
    kid = faq_ids[(SHOP_A, "可以寄到新疆吗")]
    candidate = kit.repo.get_knowledge(SHOP_A, kid)
    change = kit.admin.approve_edited(
        SHOP_A,
        kid,
        FaqChanges.of(question="新疆能发货吗", answer="亲，新疆可以发，需补运费 15 元哦~"),
        merchant=MERCHANT,
    )
    new = change.item
    assert new.id != kid and new.status == KnowledgeStatus.APPROVED
    assert new.source == KnowledgeSource.ADOPTED  # 来源渠道不变
    assert new.approved_by == MERCHANT and new.intent == candidate.intent
    old = kit.repo.get_knowledge(SHOP_A, kid)
    assert old.status == KnowledgeStatus.RETIRED and old.answer == candidate.answer
    [event] = kit.events(AuditAction.KNOWLEDGE_APPROVED)
    assert event.detail["knowledge_id"] == kid and event.detail["new_knowledge_id"] == new.id
    assert event.detail["edited"] is True and event.detail["changed"] == ["question", "answer"]
    assert "新疆" not in _all_text(kit)  # 候选的问题是买家原话（已最小化），也不进审计


def test_approve_edited_only_accepts_candidates(kit: Kit, faq_ids):
    for question in ("发什么快递", "双十一有活动吗"):
        with pytest.raises(ActionNotAllowed, match="只有候选知识"):
            kit.admin.approve_edited(
                SHOP_A, faq_ids[(SHOP_A, question)], FaqChanges.of(answer="x"), merchant=MERCHANT
            )
    with pytest.raises(KeyError):
        kit.admin.approve_edited(SHOP_A, 9999, FaqChanges.of(answer="x"), merchant=MERCHANT)


# ================================================================ FAQ：保存前检查


def test_policy_conflict_is_reported_as_blocking(kit: Kit):
    conflict = kit.admin.create_faq(
        SHOP_A, question="几天发货", answer="亲，付款后 24 小时内发货哦~", merchant=MERCHANT
    )
    [warning] = [w for w in conflict.warnings if w.rule == "policy_conflict"]
    assert warning.blocks_answer and "发货时效" in warning.detail
    consistent = kit.admin.create_faq(
        SHOP_A, question="多久能发货", answer="亲，现货付款后 48 小时内发货哦~", merchant=MERCHANT
    )
    assert not [w for w in consistent.warnings if w.rule == "policy_conflict"]
    assert kit.events(AuditAction.KNOWLEDGE_CREATED)[0].detail["warnings"] == ["policy_conflict"]


def test_duplicate_question_is_reported(kit: Kit, faq_ids):
    change = kit.admin.create_faq(
        SHOP_A,
        question="亲，什么时候发货呀？",
        answer="亲，现货付款后 48 小时内发货哦~",
        merchant=MERCHANT,
    )
    [dup] = [w for w in change.warnings if w.rule == "duplicate_question"]
    assert str(faq_ids[(SHOP_A, "什么时候发货")]) in dup.detail and not dup.blocks_answer
    # 修改自己不算重复
    edited = kit.admin.edit_faq(
        SHOP_A,
        faq_ids[(SHOP_A, "什么时候发货")],
        FaqChanges.of(answer="亲，现货付款后 48 小时内发出哦~"),
        merchant=MERCHANT,
    )
    assert [w.rule for w in edited.warnings if w.rule == "duplicate_question"] == [
        "duplicate_question"  # 刚才录入的那条还在
    ]


def test_checker_gets_the_answer_time_materials(kit: Kit):
    kit.checker_result = [
        FaqWarning(rule="banned_word", detail="包含商家禁用词", blocks_answer=True)
    ]
    change = kit.admin.create_faq(
        SHOP_A, question="防晒吗", answer="亲，这款不防晒哦", item_id="i_555", merchant=MERCHANT
    )
    assert change.warnings[0].rule == "banned_word"
    [check] = kit.checks
    assert check.config.banned_words == ["假一赔十", "全网最低"]  # 店铺生效配置
    assert check.item is not None and check.item.forbidden_claims == ["防晒", "抗菌"]
    assert check.evidence.id == "faq:0" and "这款不防晒" in check.evidence.content
    reference = " ".join(e.content for e in check.reference)
    assert "48 小时内发货" in reference and "版型偏小" in reference
    assert check.now == kit.clock.now()


# ================================================================ FAQ：列表


def test_list_faqs_filters_by_status_and_item(kit: Kit):
    all_items = kit.admin.list_faqs(SHOP_A)
    assert {k.status for k in all_items} == set(KnowledgeStatus)
    approved = kit.admin.list_faqs(SHOP_A, status=KnowledgeStatus.APPROVED)
    assert approved and all(k.status == KnowledgeStatus.APPROVED for k in approved)
    scoped = kit.admin.list_faqs(SHOP_A, item_id="i_555")
    assert [k.question for k in scoped] == ["这款会透吗"]  # 只列这个商品专属的
    assert all(k.shop_id == SHOP_A for k in all_items)


# ================================================================ 店铺政策


def test_list_policies_shows_every_known_key_with_label_and_question(kit: Kit):
    rows = kit.admin.list_policies(SHOP_A)
    assert [r.key for r in rows] == [k.value for k in PolicyKey]
    ship = rows[0]
    assert ship.label == "发货时效" and ship.questionnaire_id == 1 and ship.value
    warranty = next(r for r in rows if r.key == "warranty")
    assert warranty.value is None and warranty.updated_at is None
    assert warranty.questionnaire_id == 11 and warranty.known


def test_set_policy_writes_and_audits_without_the_value(kit: Kit):
    kit.clock.advance(minutes=3)
    row = kit.admin.set_policy(SHOP_A, "warranty", "  整机保修一年\x00 ", merchant=MERCHANT)
    assert row.value == "整机保修一年" and row.updated_at == kit.clock.now()
    assert kit.repo.get_policies(SHOP_A)["warranty"].value == "整机保修一年"
    [event] = kit.events(AuditAction.POLICY_UPDATED)
    assert event.detail == {"key": "warranty", "created": True} and event.actor == MERCHANT
    # 同样的内容再写一次：不写库、不记审计
    kit.clock.advance(minutes=3)
    again = kit.admin.set_policy(SHOP_A, "warranty", "整机保修一年", merchant=MERCHANT)
    assert again.updated_at == row.updated_at
    assert len(kit.events(AuditAction.POLICY_UPDATED)) == 1


@pytest.mark.parametrize(("key", "value"), [("nope", "x"), ("warranty", "  "), ("warranty", "")])
def test_set_policy_rejects_unknown_keys_and_empty_values(kit: Kit, key: str, value: str):
    with pytest.raises(ValueError):
        kit.admin.set_policy(SHOP_A, key, value, merchant=MERCHANT)
    assert "warranty" not in kit.repo.get_policies(SHOP_A)


# ================================================================ 商品卡片


def test_update_item_card_changes_only_merchant_fields(kit: Kit):
    before = kit.repo.get_item_card(SHOP_A, "i_555")
    kit.clock.advance(minutes=10)
    card, changed = kit.admin.update_item_card(
        SHOP_A,
        "i_555",
        {
            "size_advice": "  偏小一码，建议拍大一码 ",
            "care": ["手洗", " 手洗 ", "", "不可暴晒\x00"],
            "usage": None,
        },
        merchant=MERCHANT,
    )
    assert changed == ["size_advice", "care"]  # usage 本来就是空的
    assert card.size_advice == "偏小一码，建议拍大一码" and card.care == ["手洗", "不可暴晒"]
    assert card.title == before.title and card.skus == before.skus
    assert card.updated_at == kit.clock.now()
    assert kit.repo.get_item_card(SHOP_A, "i_555") == card
    [event] = kit.events(AuditAction.ITEM_CARD_UPDATED)
    assert event.detail == {"item_id": "i_555", "fields": ["size_advice", "care"]}


def test_update_item_card_can_clear_and_is_a_no_op_without_changes(kit: Kit):
    card, changed = kit.admin.update_item_card(
        SHOP_A, "i_555", {"size_advice": " ", "forbidden_claims": None}, merchant=MERCHANT
    )
    assert changed == ["size_advice", "forbidden_claims"]
    assert card.size_advice is None and card.forbidden_claims == []
    kit.clock.advance(minutes=1)
    same, changed = kit.admin.update_item_card(
        SHOP_A, "i_555", {"size_advice": None}, merchant=MERCHANT
    )
    assert changed == [] and same.updated_at == card.updated_at
    assert len(kit.events(AuditAction.ITEM_CARD_UPDATED)) == 1


def test_update_item_card_rejects_platform_fields_and_foreign_items(kit: Kit, world: DemoWorld):
    kit.repo.upsert_item_card(_b_only_card(world))
    with pytest.raises(ValueError, match="商品同步"):
        kit.admin.update_item_card(SHOP_A, "i_555", {"title": "改标题"}, merchant=MERCHANT)
    with pytest.raises(ValueError):
        kit.admin.update_item_card(SHOP_A, "i_555", {}, merchant=MERCHANT)
    with pytest.raises(ValueError, match="最多"):
        kit.admin.update_item_card(
            SHOP_A, "i_555", {"care": [f"洗法 {i}" for i in range(31)]}, merchant=MERCHANT
        )
    for item_id in ("i_b_only", "i_nope"):
        with pytest.raises(KeyError):
            kit.admin.update_item_card(SHOP_A, item_id, {"care": ["手洗"]}, merchant=MERCHANT)
    assert kit.repo.get_item_card(SHOP_B, "i_b_only").care == []
    assert kit.audit.events == []


# ================================================================ 店铺配置（§9.3）


def test_get_config_shows_effective_defaults_and_overrides(kit: Kit):
    snapshot = kit.admin.get_config(SHOP_A)
    assert snapshot.effective.display_name == "小雅女装旗舰店"
    assert snapshot.defaults.display_name == "" and snapshot.overrides["tone"] == "亲切"
    assert snapshot.editable == list(MERCHANT_CONFIG_FIELDS)
    assert snapshot.locked == list(LOCKED_CONFIG_FIELDS)


def test_update_config_sets_and_removes_overrides(repo, world: DemoWorld):
    config = EngineConfig(shop_defaults_by_category={"女装": {"max_answer_chars": 100}})
    kit = Kit(repo, world, config=config)
    snapshot = kit.admin.update_config(
        SHOP_A,
        {
            "tone": "专业",
            "max_answer_chars": None,  # 删掉覆盖项：回到类目默认
            "aftersale_amount_limit": 800,
            "banned_words": "最便宜、假一赔十\x00，最便宜",
            "disabled_intents": ["invoice_gift", "invoice_gift"],
        },
        merchant=MERCHANT,
    )
    assert snapshot.changed == [
        "aftersale_amount_limit",
        "banned_words",
        "disabled_intents",
        "tone",
    ]
    assert snapshot.removed == ["max_answer_chars"]
    assert snapshot.effective.tone == "专业" and snapshot.effective.max_answer_chars == 100
    stored = kit.repo.get_shop(SHOP_A).config_overrides
    assert "max_answer_chars" not in stored
    assert stored["aftersale_amount_limit"] == "800"  # JSON 原生值（Decimal 写成字符串）
    assert stored["banned_words"] == ["最便宜", "假一赔十"]
    assert stored["disabled_intents"] == ["invoice_gift"]
    [event] = kit.events(AuditAction.CONFIG_UPDATED)
    assert event.detail == {
        "changed": ["aftersale_amount_limit", "banned_words", "disabled_intents", "tone"],
        "removed": ["max_answer_chars"],
    }
    assert "最便宜" not in _all_text(kit)
    # 没有实际改动：不写、不记审计
    again = kit.admin.update_config(SHOP_A, {"tone": "专业"}, merchant=MERCHANT)
    assert again.changed == [] and len(kit.events(AuditAction.CONFIG_UPDATED)) == 1


@pytest.mark.parametrize("key", ["shadow_mode", "auto_reply_enabled"])
def test_rollout_switches_are_not_merchant_editable(kit: Kit, key: str):
    before = kit.repo.get_shop(SHOP_A)
    with pytest.raises(ActionNotAllowed, match="上线节奏"):
        kit.admin.update_config(SHOP_A, {key: True, "tone": "专业"}, merchant=MERCHANT)
    assert kit.repo.get_shop(SHOP_A) == before and kit.audit.events == []


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"nope": 1},
        {"max_answer_chars": 5},
        {"tone": "热情"},
        {"salutation": "你"},
        {"suggestion_count": 9},
        {"aftersale_amount_limit": -1},
        {"aftersale_amount_limit": "很多"},
        {"aftersale_amount_limit": True},
        {"manager_ids": ["system"]},
        {"manager_ids": "kf_manager"},
        {"disabled_intents": ["no_such_intent"]},
        {"disabled_intents": [{"a": 1}]},  # 不可哈希的取值也是 422，不是 500
        {"disabled_intents": "invoice_gift"},
        {"suggestion_count": True},
        {"display_name": 123},
        {"banned_words": ["词" * 31]},
        {"banned_words": {"最便宜": 1}},
    ],
)
def test_invalid_config_values_are_rejected(kit: Kit, changes: dict):
    before = kit.repo.get_shop(SHOP_A)
    with pytest.raises(ValueError):
        kit.admin.update_config(SHOP_A, changes, merchant=MERCHANT)
    assert kit.repo.get_shop(SHOP_A) == before and kit.audit.events == []


def test_indexing_failure_does_not_undo_the_write(repo, world: DemoWorld):
    kit = Kit(repo, world)

    def broken(shop_id: str) -> None:
        raise RuntimeError("向量服务不可用（模拟）")

    kit.admin._on_faq_changed = broken
    change = kit.admin.create_faq(SHOP_A, question="在吗", answer="亲，在的哦", merchant=MERCHANT)
    assert repo.get_knowledge(SHOP_A, change.item.id).status == KnowledgeStatus.APPROVED


def test_created_at_follows_the_injected_clock(kit: Kit):
    kit.clock.set(kit.world.now + timedelta(days=2))
    change = kit.admin.create_faq(SHOP_A, question="在吗", answer="亲，在的哦", merchant=MERCHANT)
    assert change.item.created_at == kit.world.now + timedelta(days=2)


def test_concurrency_conflict_type_is_not_leaked(kit: Kit, faq_ids):
    """仓储的 ConcurrencyConflict 转成 ActionNotAllowed（接口层 409，提示刷新）。"""
    kid = faq_ids[(SHOP_A, "发什么快递")]
    kit.repo.set_knowledge_status(SHOP_A, kid, KnowledgeStatus.RETIRED)
    with pytest.raises(ActionNotAllowed):
        kit.admin.edit_faq(SHOP_A, kid, FaqChanges.of(answer="x"), merchant=MERCHANT)
    assert not issubclass(ActionNotAllowed, ConcurrencyConflict)
