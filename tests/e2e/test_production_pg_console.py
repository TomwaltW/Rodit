"""生产组装跑在 PostgreSQL 上：商家控制台、上线节奏与每周复核、阈值重算的数据链路
（§5.2.3、§9.1、§9.3、§13.4、§13.5、§8.4）。

和内存实现同样的行为要在 PostgreSQL 上成立：
- 商家录入 FAQ 后下一次提问就用上；改 FAQ 是旧条目下线、新条目上线；
- 商品同步保留商家填的字段，再同步一次什么都不写；商家改卡片和同步各改各的字段；
- 商家改配置和复核人员切换影子模式同时发生，两边的修改都留下；
- 影子模式的推荐带路由置信度和规则，每周抽样、复核打分、报表、切到智能辅助（强制要写原因）；
- 客服发送后的标注带置信度和规则，训练导出角色按时间窗读得到；
- 取消订购删掉复核样本在内的全部数据。
大模型用离线替身，平台接口是模拟实现，全部是合成数据。
"""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest
from psycopg import sql

from rodit.adapters import StandaloneAssistAdapter
from rodit.app import AssistService, build_unit
from rodit.app.demo import seed_unit
from rodit.app.offline import EvidenceEchoLLM, HeuristicQAJudge
from rodit.app.production import postgres_repos
from rodit.app.rollout import RolloutGateNotPassed
from rodit.core.audit import AuditAction
from rodit.core.config import EngineConfig
from rodit.core.enums import Intent, Platform, QualityReviewStatus, SuggestionStatus
from rodit.core.pii import BuyerRefHasher
from rodit.decision import RuleProvider
from rodit.feedback.review import iso_period
from rodit.knowledge.admin import FaqChanges
from rodit.storage import BUSINESS_TABLES, Database, PgFeedbackRepository
from rodit.testing.fakes import (
    FakeClock,
    FakeImageTextExtractor,
    InMemoryItemCatalog,
    InMemoryToolGateway,
)
from rodit.testing.fixtures import (
    DEMO_OCR_TEXTS,
    SHOP_A,
    SHOP_B,
    TEST_SALT,
    build_demo_world,
    demo_platform_items,
)

pytestmark = pytest.mark.pg

TOKEN_KEY = "rodit-e2e-token-key-synthetic-only"  # 只用于测试的合成密钥
MERCHANT = "owner_a"
REVIEWER = "qa_reviewer_01"
AGENT = "kf_02"


def _admin(dsn: str) -> Database:
    return Database(dsn, app_role=None)


@pytest.fixture
def pg_service(e2e_pg_dsn: str, monkeypatch: pytest.MonkeyPatch) -> AssistService:
    with _admin(e2e_pg_dsn).session(None) as conn:
        tables = sql.SQL(", ").join(sql.Identifier(t) for t in BUSINESS_TABLES)
        conn.execute(sql.SQL("TRUNCATE {} RESTART IDENTITY").format(tables))
    monkeypatch.setenv("RODIT_TOKEN_KEY", TOKEN_KEY)
    world = build_demo_world()
    clock = FakeClock(world.now)
    hasher = BuyerRefHasher(TEST_SALT)
    tools = InMemoryToolGateway(world, Platform.TAOBAO)
    unit = build_unit(
        platform=Platform.TAOBAO,
        config=EngineConfig(),
        clock=clock,
        hasher=hasher,
        repos=postgres_repos(Database(e2e_pg_dsn, purge_batch_rows=3), clock=clock),
        llm=EvidenceEchoLLM(),
        tools=tools,
        adapters=[StandaloneAssistAdapter(Platform.TAOBAO, clock, hasher)],
        route_provider=RuleProvider(),
        qa_provider=HeuristicQAJudge(),
        catalog=InMemoryItemCatalog(demo_platform_items(world), Platform.TAOBAO),
        ocr=FakeImageTextExtractor(DEMO_OCR_TEXTS),
    )
    seed_unit(unit, world)
    return AssistService(unit)


def _ask(service: AssistService, question: str, session: str, item_id: str | None = None):
    result = service.ask_standalone(
        shop_id=SHOP_A, session_id=session, question=question, item_id=item_id, agent_id=AGENT
    )
    assert result is not None
    return result


# ================================================================ 商家控制台（§9.1、§5.2）


def test_faq_create_edit_and_answer_on_postgres(pg_service: AssistService):
    console = pg_service.console
    created = console.create_faq(
        SHOP_A,
        question="能开公司抬头的发票吗",
        answer="亲，可以开公司抬头的电子普通发票哦~",
        intent=Intent.INVOICE_GIFT,
        merchant=MERCHANT,
    )
    assert created.item.id is not None and created.item.approved_by == MERCHANT
    result = _ask(pg_service, "能开公司抬头的发票吗", "w_faq")
    cited = {e.id for card in result.suggestions for e in card.evidence}
    assert f"faq:{created.item.id}" in cited  # 录入后立刻检索得到

    edited = console.edit_faq(
        SHOP_A,
        created.item.id,
        FaqChanges.of(answer="亲，可以开公司抬头的电子普通发票，下单时备注税号哦~"),
        merchant=MERCHANT,
    )
    assert edited.retired is not None and edited.retired.id == created.item.id
    assert edited.item.id != created.item.id
    repo = pg_service.unit.repos.knowledge
    assert repo.get_knowledge(SHOP_A, created.item.id).status == "retired"
    assert repo.get_knowledge(SHOP_A, edited.item.id).status == "approved"
    actions = [e.action for e in pg_service.unit.repos.audit.list_events(SHOP_A)]
    assert AuditAction.KNOWLEDGE_CREATED in actions and AuditAction.KNOWLEDGE_EDITED in actions


def test_item_sync_keeps_merchant_fields_and_is_idempotent_on_postgres(
    pg_service: AssistService,
):
    console = pg_service.console
    repo = pg_service.unit.repos.knowledge
    before = repo.get_item_card(SHOP_A, "i_555")
    report = console.sync_items(SHOP_A, merchant=MERCHANT)
    assert report.created == ["i_999"] and report.updated == ["i_555"]
    assert report.off_sale == ["i_888"] and report.ocr_failed_items == ["i_999"]
    dress = repo.get_item_card(SHOP_A, "i_555")
    assert dress.title == "法式碎花连衣裙（新款）"
    assert dress.size_advice == before.size_advice and dress.care == before.care
    assert DEMO_OCR_TEXTS["img_555_detail_1"] in (dress.detail_text or "")
    cards = {c.item_id: c for c in repo.list_item_cards(SHOP_A)}

    pg_service.unit.clock.advance(hours=1)  # type: ignore[attr-defined]
    again = console.sync_items(SHOP_A, merchant=MERCHANT)
    assert again.created == [] and again.updated == []
    assert {c.item_id: c for c in repo.list_item_cards(SHOP_A)} == cards  # 什么都没写

    card, changed = console.update_item(
        SHOP_A, "i_999", {"size_advice": "宽松版型，平时码即可"}, merchant=MERCHANT
    )
    assert changed == ["size_advice"] and card.title == "宽松针织开衫"
    assert repo.get_item_card(SHOP_B, "i_555").title == "男士休闲裤"  # 别的店铺不受影响


def test_concurrent_config_edit_and_stage_switch_both_survive_on_postgres(
    pg_service: AssistService,
):
    """商家改配置、复核人员切回影子模式同时发生：各写各的键，两边都留下（原子部分修改）。"""
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run(fn) -> None:
        try:
            barrier.wait(timeout=10)
            fn()
        except BaseException as e:  # noqa: BLE001 — 带回主线程断言
            errors.append(e)

    threads = [
        threading.Thread(
            target=run,
            args=(
                lambda: pg_service.console.update_config(
                    SHOP_A, {"tone": "专业", "suggestion_count": 1}, merchant=MERCHANT
                ),
            ),
        ),
        threading.Thread(
            target=run,
            args=(lambda: pg_service.rollout.set_stage(SHOP_A, "shadow", operator=REVIEWER),),
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []
    overrides = pg_service.unit.repos.knowledge.get_shop(SHOP_A).config_overrides
    assert overrides["tone"] == "专业" and overrides["suggestion_count"] == 1
    assert overrides["shadow_mode"] is True


# ================================================================ 上线节奏与复核（§13.4、§13.5）


def test_shadow_review_and_forced_switch_on_postgres(pg_service: AssistService):
    rollout = pg_service.rollout
    rollout.set_stage(SHOP_A, "shadow", operator=REVIEWER)
    for i, question in enumerate(("什么时候发货", "能开发票吗", "发什么快递", "这款会起球吗")):
        result = _ask(pg_service, question, f"w_shadow_{i}", item_id="i_555")
        assert result.suggestions == [] or all(s.suggestion_id for s in result.suggestions)
    records = pg_service.unit.repos.suggestions.list(SHOP_A, status=SuggestionStatus.SHADOW)
    assert records, "影子模式要留下 SHADOW 记录"
    assert all(r.intent_confidence is not None and r.route_rule is not None for r in records)

    period = iso_period(pg_service.unit.clock.now())
    sampled = rollout.sample(SHOP_A, reviewer=REVIEWER, period=period, size=50)
    assert sampled.total == len(records)
    again = rollout.sample(SHOP_A, reviewer=REVIEWER, period=period, size=50)
    assert again.added == 0 and again.total == sampled.total  # 重跑不重复抽

    listing = rollout.list_reviews(SHOP_A, viewer=REVIEWER, period=period)
    assert listing.count == len(records)
    assert all(i.suggestion is not None and i.suggestion.text for i in listing.items)
    viewed = pg_service.unit.repos.audit.list_events(SHOP_A, action=AuditAction.QUALITY_VIEWED)
    assert len(viewed) == 1 and viewed[0].actor == REVIEWER

    first = listing.items[0]
    scored = rollout.score(
        SHOP_A,
        first.id,
        status="scored",
        reviewer=REVIEWER,
        scores={"factual": 2, "relevance": 2, "compliance": 2, "expression": 1},
        note="语气还可以更自然",
    )
    assert scored.status == QualityReviewStatus.SCORED
    report = rollout.report(SHOP_A, period=period)
    assert report.scored == 1 and report.pending == len(records) - 1

    with pytest.raises(RolloutGateNotPassed):  # 复核样本不够：门槛没过
        rollout.set_stage(SHOP_A, "assist", operator=REVIEWER)
    assert pg_service.unit.repos.knowledge.get_shop(SHOP_A).config_overrides["shadow_mode"]
    change = rollout.set_stage(
        SHOP_A, "assist", operator=REVIEWER, force=True, note="种子商家试点，人工确认过影子对比"
    )
    assert change.changed and change.forced
    assert pg_service.unit.repos.knowledge.get_shop(SHOP_A).config_overrides["shadow_mode"] is False

    # 取消订购：复核样本随店铺一起删掉
    pg_service.revoke_shop(SHOP_A)
    assert pg_service.unit.repos.quality_reviews.list(SHOP_A) == []


# ================================================================ 标注与训练导出（§13.5、§6.4）


def test_labels_carry_confidence_and_rule_and_export_by_window(
    pg_service: AssistService, e2e_pg_dsn: str
):
    result = _ask(pg_service, "什么时候发货", "w_label")
    card = result.suggestions[0]
    pg_service.feedback_sent(SHOP_A, card.suggestion_id, agent_id=AGENT)
    [label] = pg_service.unit.repos.feedback.list_labels(SHOP_A)
    assert label.source == "feedback_adopt"
    assert label.predicted_confidence is not None and label.route_rule == 6

    trainer = PgFeedbackRepository(Database(e2e_pg_dsn))  # 测试库的管理员账号能 SET ROLE
    now = pg_service.unit.clock.now()
    assert [s.text for s in trainer.export_training_labels(since=now - timedelta(days=30))] == [
        label.text
    ]
    assert trainer.export_training_labels(until=now - timedelta(days=1)) == []
