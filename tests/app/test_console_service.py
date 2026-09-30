"""商家控制台的应用服务（§9.1 知识编辑跟着单元走、§5.2.1 改动即时生效、§5.5 保存前质检提醒、
§8.4 取消订购与控制台写操作互斥）。演示单元，全部合成数据，不联网。"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from rodit.app import AssistService
from rodit.app.demo import build_demo_unit
from rodit.core.audit import AuditAction
from rodit.core.enums import Intent, KnowledgeStatus, Platform
from rodit.core.errors import ShopInactive, ToolUnavailable
from rodit.knowledge.admin import FaqChanges
from rodit.knowledge.sync import ItemSyncUnavailable
from rodit.testing.fixtures import SHOP_A, SHOP_B

MERCHANT = "owner_a"
QUESTION = "发票抬头可以写公司吗"


@pytest.fixture
def service() -> AssistService:
    return AssistService(build_demo_unit(Platform.TAOBAO))


def _ask(service: AssistService, question: str, session: str, **kw: Any):
    result = service.ask_standalone(shop_id=SHOP_A, session_id=session, question=question, **kw)
    assert result is not None
    return result


def _evidence_ids(result) -> list[str]:
    return [e.id for card in result.suggestions for e in card.evidence]


# ================================================================ 改了立刻用于回答


def test_new_faq_is_used_by_the_next_answer(service: AssistService):
    before = _ask(service, QUESTION, "w1")
    change = service.console.create_faq(
        SHOP_A,
        question=QUESTION,
        answer="亲，可以开公司抬头的电子普通发票哦~",
        intent=Intent.INVOICE_GIFT,
        merchant=MERCHANT,
    )
    faq_id = f"faq:{change.item.id}"
    assert faq_id not in _evidence_ids(before)
    after = _ask(service, QUESTION, "w2")
    assert _evidence_ids(after) == [faq_id]
    assert after.suggestions[0].text == "亲，可以开公司抬头的电子普通发票哦~"
    # 立刻建了索引（不用等下一次检索懒索引）
    assert service.unit.repos.knowledge.get_knowledge(SHOP_A, change.item.id).embedding


def test_edited_faq_answers_with_the_new_id_and_keeps_the_old_one_intact(service: AssistService):
    created = service.console.create_faq(
        SHOP_A,
        question=QUESTION,
        answer="亲，可以开公司抬头的电子普通发票哦~",
        intent=Intent.INVOICE_GIFT,
        merchant=MERCHANT,
    )
    first = _ask(service, QUESTION, "w1")
    old_id = created.item.id
    assert _evidence_ids(first) == [f"faq:{old_id}"]
    shown = service.unit.repos.suggestions.get(SHOP_A, first.suggestions[0].suggestion_id)

    edited = service.console.edit_faq(
        SHOP_A,
        old_id,
        FaqChanges.of(answer="亲，可以开公司抬头的电子普通发票，确认收货后 3 个工作日内开出哦~"),
        merchant=MERCHANT,
    )
    new_id = edited.item.id
    assert new_id != old_id and edited.retired.status == KnowledgeStatus.RETIRED
    second = _ask(service, QUESTION, "w2")
    assert _evidence_ids(second) == [f"faq:{new_id}"]
    assert "3 个工作日" in second.suggestions[0].text
    # 旧推荐的证据 faq:{旧 id} 仍指向买家当时看到的那段话
    assert shown.evidence_ids == [f"faq:{old_id}"]
    old = service.unit.repos.knowledge.get_knowledge(SHOP_A, old_id)
    assert old.answer == shown.text and old.status == KnowledgeStatus.RETIRED


def test_policy_change_is_effective_immediately(service: AssistService):
    before = _ask(service, "什么时候发货", "w1")
    assert "48" in before.suggestions[0].text
    service.console.set_policy(SHOP_A, "ship_time", "现货付款后 72 小时内发货", merchant=MERCHANT)
    after = _ask(service, "什么时候发货", "w2")
    assert "72" in after.suggestions[0].text and "48" not in after.suggestions[0].text
    assert "policy:ship_time" in _evidence_ids(after)
    assert "faq:2" not in _evidence_ids(after)  # 和新政策冲突的 FAQ 被覆盖（§5.2.1）


def test_item_card_edit_is_effective_immediately(service: AssistService):
    service.console.update_item(
        SHOP_A,
        "i_555",
        {"size_advice": "版型偏大，建议拍小一码；身高 160–165 cm、体重 100–115 斤建议 S"},
        merchant=MERCHANT,
    )
    result = _ask(service, "165 110斤穿什么码", "w1", item_id="i_555")
    text = result.suggestions[0].text
    assert "建议 S" in text and "建议 M" not in text


def test_config_change_is_effective_immediately(service: AssistService):
    assert _ask(service, "能开发票吗", "w1").suggestions
    snapshot = service.console.update_config(
        SHOP_A, {"disabled_intents": ["invoice_gift"]}, merchant=MERCHANT
    )
    assert snapshot.effective.disabled_intents == [Intent.INVOICE_GIFT]
    assert _ask(service, "能开发票吗", "w2").suggestions == []
    service.console.update_config(SHOP_A, {"disabled_intents": None}, merchant=MERCHANT)
    assert _ask(service, "能开发票吗", "w3").suggestions


# ================================================================ 保存前质检提醒（§5.5）


@pytest.mark.parametrize(
    ("answer", "item_id", "rule", "blocks"),
    [
        ("亲，加我微信聊哦~", None, "off_platform", True),
        ("亲，这是全店最好的发票服务哦~", None, "absolute_term", True),
        ("亲，我们假一赔十，放心开票~", None, "banned_word", True),  # 店铺禁用词
        ("亲，这款面料防晒效果很好哦~", "i_555", "forbidden_claim", True),  # 商家标注的卖点
        ("亲，**可以开**发票哦~", None, "format", False),
        ("亲，" + "可以开电子普通发票" * 20, None, "length", False),
        ("亲，今天下单明天就能到哦~", None, "time_promise", False),
    ],
)
def test_faq_warnings_predict_answer_time_quality_checks(
    service: AssistService, answer: str, item_id: str | None, rule: str, blocks: bool
):
    change = service.console.create_faq(
        SHOP_A, question="开票的问题", answer=answer, item_id=item_id, merchant=MERCHANT
    )
    matching = [w for w in change.warnings if w.rule == rule]
    assert matching, change.warnings
    assert all(w.blocks_answer is blocks for w in matching)
    assert change.item.status == KnowledgeStatus.APPROVED  # 只提醒，不拦截保存


def test_clean_faq_has_no_warnings(service: AssistService):
    change = service.console.create_faq(
        SHOP_A,
        question=QUESTION,
        answer="亲，可以开公司抬头的电子普通发票哦~",
        merchant=MERCHANT,
    )
    assert change.warnings == []


def test_blocking_warning_matches_what_happens_at_answer_time(service: AssistService):
    """提醒说会被拦截的 FAQ，回答时确实用不上（和质检同一套规则）。"""
    change = service.console.create_faq(
        SHOP_A,
        question=QUESTION,
        answer="亲，全网最低价开发票哦~",
        intent=Intent.INVOICE_GIFT,
        merchant=MERCHANT,
    )
    assert "banned_word" in {w.rule for w in change.warnings if w.blocks_answer}
    result = _ask(service, QUESTION, "w1")
    assert all("全网最低" not in card.text for card in result.suggestions)


# ================================================================ 商品同步


def test_sync_through_the_service(service: AssistService):
    report = service.console.sync_items(SHOP_A, merchant=MERCHANT)
    assert report.created == ["i_999"] and report.off_sale == ["i_888"]
    [event] = service.unit.repos.audit.list_events(SHOP_A, action=AuditAction.ITEM_SYNC.value)
    assert event.actor == MERCHANT
    one = service.console.sync_items(SHOP_A, item_id="i_555", merchant=MERCHANT)
    assert one.unchanged == ["i_555"]
    # 新同步的商品立刻能用于回答
    result = _ask(service, "这件是什么面料", "w1", item_id="i_999")
    assert "腈纶" in result.suggestions[0].text


def test_sync_unavailable_without_catalog_or_ocr(service: AssistService):
    service.unit.catalog = None
    with pytest.raises(ItemSyncUnavailable):
        service.console.sync_items(SHOP_A, merchant=MERCHANT)
    service.unit.catalog = build_demo_unit(Platform.TAOBAO).catalog
    service.unit.ocr = None
    with pytest.raises(ItemSyncUnavailable):
        service.console.sync_items(SHOP_A, merchant=MERCHANT)


def test_sync_propagates_catalog_failure(service: AssistService):
    service.unit.catalog.fail_on.add("list_items")
    with pytest.raises(ToolUnavailable):
        service.console.sync_items(SHOP_A, merchant=MERCHANT)


# ================================================================ 审计没有买家数据


def test_console_audit_has_no_buyer_or_answer_text(service: AssistService):
    candidates = service.console.list_faqs(SHOP_A, status=KnowledgeStatus.CANDIDATE)
    [candidate] = candidates  # 客服采纳的回答（问题是买家原话，已最小化）
    service.console.approve_edited(
        SHOP_A,
        candidate.id,
        FaqChanges.of(answer="亲，新疆可以发，运费补 15 元哦~"),
        merchant=MERCHANT,
    )
    service.console.create_faq(SHOP_A, question="在吗", answer="亲，在的哦~", merchant=MERCHANT)
    service.console.set_policy(SHOP_A, "warranty", "整机保修一年", merchant=MERCHANT)
    service.console.update_item(SHOP_A, "i_555", {"care": ["冷水手洗"]}, merchant=MERCHANT)
    service.console.update_config(SHOP_A, {"banned_words": ["最便宜"]}, merchant=MERCHANT)
    service.console.sync_items(SHOP_A, merchant=MERCHANT)
    console_actions = {
        AuditAction.KNOWLEDGE_APPROVED.value,
        AuditAction.KNOWLEDGE_CREATED.value,
        AuditAction.POLICY_UPDATED.value,
        AuditAction.ITEM_CARD_UPDATED.value,
        AuditAction.CONFIG_UPDATED.value,
        AuditAction.ITEM_SYNC.value,
    }
    events = [
        e for e in service.unit.repos.audit.list_events(SHOP_A) if e.action in console_actions
    ]
    assert {e.action for e in events} == console_actions
    assert all(e.actor == MERCHANT for e in events)
    text = json.dumps([e.detail for e in events], ensure_ascii=False)
    for secret in (candidate.question, "新疆", "在的", "保修", "冷水", "最便宜", "针织"):
        assert secret not in text


# ================================================================ 取消订购与控制台写操作（§8.4）


def _writes(service: AssistService) -> list[Callable[[], Any]]:
    c = service.console
    return [
        lambda: c.create_faq(SHOP_A, question="在吗", answer="亲，在的哦", merchant=MERCHANT),
        lambda: c.edit_faq(SHOP_A, 1, FaqChanges.of(answer="亲，发中通"), merchant=MERCHANT),
        lambda: c.approve_edited(SHOP_A, 6, FaqChanges.of(answer="亲，可以"), merchant=MERCHANT),
        lambda: c.set_policy(SHOP_A, "invoice", "可以开电子发票", merchant=MERCHANT),
        lambda: c.update_item(SHOP_A, "i_555", {"care": ["手洗"]}, merchant=MERCHANT),
        lambda: c.update_config(SHOP_A, {"tone": "专业"}, merchant=MERCHANT),
        lambda: c.sync_items(SHOP_A, merchant=MERCHANT),
        lambda: c.sync_items(SHOP_A, item_id="i_999", merchant=MERCHANT),
    ]


def _assert_shop_gone(service: AssistService) -> None:
    repos = service.unit.repos
    assert repos.knowledge.get_shop(SHOP_A) is None
    assert repos.knowledge.get_policies(SHOP_A) == {}
    assert repos.knowledge.list_item_cards(SHOP_A) == []
    assert repos.knowledge.list_knowledge(SHOP_A, status=None) == []
    assert repos.audit.list_events(SHOP_A) == []
    assert repos.knowledge.get_shop(SHOP_B) is not None  # 别的店铺不受影响


def test_writes_after_revoke_do_not_recreate_the_shop(service: AssistService):
    service.revoke_shop(SHOP_A)
    for write in _writes(service):
        with pytest.raises(ShopInactive):
            write()
    _assert_shop_gone(service)


@pytest.mark.parametrize("index", range(8))
def test_revoke_waits_for_an_in_flight_console_write(index: int):
    """控制台写到一半时商家取消订购：取消订购等写完，再把写下的也删掉（§8.4）。"""
    unit = build_demo_unit(Platform.TAOBAO)
    service = AssistService(unit)
    repo = unit.repos.knowledge
    inflight = threading.Event()
    revoke_started = threading.Event()
    revoke_done = threading.Event()
    original = repo.get_shop

    def slow_get_shop(shop_id: str):
        shop = original(shop_id)
        if threading.current_thread().name == "console" and not inflight.is_set():
            inflight.set()
            assert revoke_started.wait(5)
            revoke_done.wait(0.3)  # 取消订购要等在途的写操作做完：这里等不到，超时后继续
        return shop

    repo.get_shop = slow_get_shop  # type: ignore[method-assign]
    errors: list[BaseException] = []
    write = _writes(service)[index]

    def console() -> None:
        try:
            write()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def revoke() -> None:
        assert inflight.wait(5)
        revoke_started.set()
        service.revoke_shop(SHOP_A)
        revoke_done.set()

    threads = [threading.Thread(target=console, name="console"), threading.Thread(target=revoke)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors
    assert revoke_done.is_set()
    _assert_shop_gone(service)


def test_console_write_waiting_behind_revoke_does_not_write_the_shop_back():
    """取消订购进行中（删数据）时来的控制台写操作：排队等删完，然后发现店铺没了，什么都不写。"""
    unit = build_demo_unit(Platform.TAOBAO)
    service = AssistService(unit)
    repo = unit.repos.knowledge
    purging = threading.Event()
    release = threading.Event()
    original = repo.purge_shop

    def slow_purge(shop_id: str) -> int:
        purging.set()
        assert release.wait(5)
        return original(shop_id)

    repo.purge_shop = slow_purge  # type: ignore[method-assign]
    results: list[BaseException | object] = []

    def revoke() -> None:
        service.revoke_shop(SHOP_A)

    def console() -> None:
        assert purging.wait(5)
        try:
            results.append(
                service.console.set_policy(SHOP_A, "invoice", "可以开电子发票", merchant=MERCHANT)
            )
        except BaseException as e:  # noqa: BLE001
            results.append(e)

    threads = [threading.Thread(target=revoke), threading.Thread(target=console)]
    for t in threads:
        t.start()
    assert purging.wait(5)
    release.set()
    for t in threads:
        t.join(10)
    [outcome] = results
    assert isinstance(outcome, ShopInactive)
    _assert_shop_gone(service)


# ================================================================ PostgreSQL（§10 JSONB 往返）

PG_ADMIN_DSN = os.environ.get("RODIT_TEST_PG_DSN", "dbname=postgres")


@pytest.fixture
def pg_service(monkeypatch: pytest.MonkeyPatch) -> Iterator[AssistService]:
    """临时库上的演示单元（应用角色，受行级安全约束）；PostgreSQL 不可用时跳过。"""
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from rodit.adapters import StandaloneAssistAdapter
    from rodit.app import build_unit
    from rodit.app.demo import seed_unit
    from rodit.app.offline import EvidenceEchoLLM, HeuristicQAJudge
    from rodit.app.production import postgres_repos
    from rodit.core.config import EngineConfig
    from rodit.core.pii import BuyerRefHasher
    from rodit.decision import RuleProvider
    from rodit.storage import MIGRATIONS, Database
    from rodit.testing.fakes import (
        FakeClock,
        FakeImageTextExtractor,
        InMemoryItemCatalog,
        InMemoryToolGateway,
    )
    from rodit.testing.fixtures import (
        DEMO_OCR_TEXTS,
        TEST_SALT,
        build_demo_world,
        demo_platform_items,
    )

    name = f"rodit_console_{uuid.uuid4().hex[:12]}"
    try:
        with psycopg.connect(PG_ADMIN_DSN, autocommit=True, connect_timeout=3) as admin:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    except psycopg.Error as e:
        pytest.skip(f"PostgreSQL 不可用或没有建库权限（{type(e).__name__}），跳过")
    dsn = make_conninfo(PG_ADMIN_DSN, dbname=name)
    try:
        assert Database(dsn).migrate() == [m.version for m in MIGRATIONS]
        monkeypatch.setenv("RODIT_TOKEN_KEY", "rodit-console-token-key-synthetic-only")
        world = build_demo_world()
        clock = FakeClock(world.now)
        hasher = BuyerRefHasher(TEST_SALT)
        tools = InMemoryToolGateway(world, Platform.TAOBAO)
        unit = build_unit(
            platform=Platform.TAOBAO,
            config=EngineConfig(),
            clock=clock,
            hasher=hasher,
            repos=postgres_repos(Database(dsn), clock=clock),
            llm=EvidenceEchoLLM(),
            tools=tools,
            adapters=[StandaloneAssistAdapter(Platform.TAOBAO, clock, hasher)],
            route_provider=RuleProvider(),
            qa_provider=HeuristicQAJudge(),
            catalog=InMemoryItemCatalog(demo_platform_items(world), Platform.TAOBAO),
            ocr=FakeImageTextExtractor(DEMO_OCR_TEXTS),
        )
        seed_unit(unit, world)
        yield AssistService(unit)
    finally:
        with psycopg.connect(PG_ADMIN_DSN, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.mark.pg
def test_console_on_postgres(pg_service: AssistService):
    service = pg_service
    repo = service.unit.repos.knowledge
    first = service.console.sync_items(SHOP_A, merchant=MERCHANT)
    assert first.created == ["i_999"] and first.updated == ["i_555"]
    cards = {c.item_id: c for c in repo.list_item_cards(SHOP_A)}
    service.unit.clock.advance(hours=1)
    again = service.console.sync_items(SHOP_A, merchant=MERCHANT)
    assert again.created == [] and again.updated == []  # 从 JSONB 读回来比较，仍然没变
    assert {c.item_id: c for c in repo.list_item_cards(SHOP_A)} == cards

    created = service.console.create_faq(
        SHOP_A,
        question=QUESTION,
        answer="亲，可以开公司抬头的电子普通发票哦~",
        intent=Intent.INVOICE_GIFT,
        merchant=MERCHANT,
    )
    edited = service.console.edit_faq(
        SHOP_A,
        created.item.id,
        FaqChanges.of(answer="亲，可以开公司抬头的发票哦~"),
        merchant=MERCHANT,
    )
    assert repo.get_knowledge(SHOP_A, created.item.id).status == KnowledgeStatus.RETIRED
    assert _evidence_ids(_ask(service, QUESTION, "w1")) == [f"faq:{edited.item.id}"]

    service.console.update_config(
        SHOP_A, {"aftersale_amount_limit": 800, "banned_words": None}, merchant=MERCHANT
    )
    overrides = repo.get_shop(SHOP_A).config_overrides
    assert overrides["aftersale_amount_limit"] == "800" and "banned_words" not in overrides
    service.console.update_item(SHOP_A, "i_999", {"care": ["冷水手洗"]}, merchant=MERCHANT)
    assert service.console.sync_items(SHOP_A, merchant=MERCHANT).updated == []
    assert repo.get_item_card(SHOP_A, "i_999").care == ["冷水手洗"]
    service.revoke_shop(SHOP_A)
    with pytest.raises(ShopInactive):
        service.console.set_policy(SHOP_A, "invoice", "可以开", merchant=MERCHANT)
    assert repo.get_policies(SHOP_A) == {} and repo.get_shop(SHOP_A) is None
