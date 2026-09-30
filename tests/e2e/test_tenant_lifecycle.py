"""店铺生命周期端到端（方案 §8.4 订购 → 授权 → 开通 → 续期 → 停用 / 恢复 → 取消订购 / 过期清理，
§9.1 租户开通、§9.2 串店、§10 shop / shop_auth）。

- 演示单元（内存仓储）+ HTTP 接口：平台对接推订购事件开通 → 出推荐 → 停用 / 恢复 → 续期被拒后
  403 → 续订恢复 → 平台代码在进程内推取消订购 → 数据删除、系统事件留下 → 重新开通；
- 读者探针（502379f 时授权过期照样读平台、只读权限照样写售后）现在都在调平台之前停下；
- @pytest.mark.pg：同样的生命周期跑在 PostgreSQL（行级安全、令牌加密、跨店只经
  rodit_auth_expiring 列出到期店铺、清理之后系统事件还读得到）。
全部是合成数据和合成令牌；平台接口、续期接口、订购推送都是 mock。
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from psycopg import sql
from pydantic import SecretStr

from rodit.adapters import StandaloneAssistAdapter, TaobaoMockAdapter
from rodit.api import Principal, StaticTokenAuthenticator, create_app
from rodit.app import AssistService, build_unit
from rodit.app.demo import SUPPORTED_ACTIONS, build_demo_unit, seed_unit, taobao_event
from rodit.app.offline import EvidenceEchoLLM, HeuristicQAJudge
from rodit.app.production import postgres_repos
from rodit.app.token_lifecycle import RefreshOutcome
from rodit.core.audit import AuditAction
from rodit.core.config import EngineConfig
from rodit.core.enums import AfterSaleAction, Platform, ReviewState, ShopStatus
from rodit.core.errors import AuthExpired
from rodit.core.models import ShopAuth
from rodit.core.pii import BuyerRefHasher
from rodit.core.scopes import MVP_SCOPES
from rodit.decision import RuleProvider
from rodit.engine.engine import REASON_SHOP_INACTIVE
from rodit.ops.bindings import LazySubscriptionHandler
from rodit.storage import BUSINESS_TABLES, Database
from rodit.testing.fakes import (
    FakeActionPort,
    FakeClock,
    FakeSubscriptionParser,
    FakeTokenRefresher,
    InMemoryToolGateway,
    subscription_push,
)
from rodit.testing.fixtures import AGENT, SHOP_A, SHOP_B, TEST_SALT, build_demo_world

NEW = "s_new"
GRANT = "tok-e2e-grant-synthetic-1b7c"
GRANT_REFRESH = "tok-e2e-refresh-synthetic-2d8e"
MERCHANT = "owner_new"
FAQ_Q = "能开发票吗"
FAQ_A = "亲，支持开具电子发票，确认收货后在订单详情里申请就可以啦~"

CALLERS = {
    "push_new": Principal(shop_id=NEW, user_id="tb_push_new", role="integration"),
    "push_a": Principal(shop_id=SHOP_A, user_id="tb_push_a", role="integration"),
}
TOKENS = {name: f"test-token-e2e-{name}-0123456789" for name in CALLERS}


def _client(service: AssistService):
    import warnings

    from starlette.exceptions import StarletteDeprecationWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StarletteDeprecationWarning)
        from fastapi.testclient import TestClient
    authenticator = StaticTokenAuthenticator({TOKENS[n]: p for n, p in CALLERS.items()})
    return TestClient(create_app(service, authenticator=authenticator))


def _auth(who: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKENS[who]}"}


def _secrets_absent(service: AssistService, *secrets: str) -> None:
    dumped = json.dumps(
        [e.model_dump(mode="json") for e in service.unit.repos.audit.events], ensure_ascii=False
    )
    for secret in secrets:
        assert secret not in dumped


# ================================================================ 内存单元 + HTTP


def test_tenant_lifecycle_end_to_end():
    unit = build_demo_unit(Platform.TAOBAO)
    clock = unit.clock
    refresher = FakeTokenRefresher(Platform.TAOBAO, clock=clock)
    unit.token_refresher = refresher
    unit.parse_subscription = FakeSubscriptionParser(Platform.TAOBAO)
    service = AssistService(unit)
    client = _client(service)
    # 平台云代码在进程内拿到的订购入口（serve 里由 bind_context 接上，这里手动接）
    on_subscription = LazySubscriptionHandler()
    on_subscription.bind(service.tenancy.handle_subscription)

    def event(text: str, msg_id: str):
        return client.post(
            "/v1/platform/B/events",
            json=taobao_event(text, msg_id=msg_id, seller_id=NEW),
            headers=_auth("push_new"),
        )

    # 1. 订购开通（平台推送，HTTP）：店铺 trial + 加密授权（内存里不加密）+ auth.granted
    grant = subscription_push(
        "subscribed",
        NEW,
        at=clock.now(),
        access_token=GRANT,
        refresh_token=GRANT_REFRESH,
        expires_at=clock.now() + timedelta(hours=20),  # 快到期了：续期任务会续
        refresh_expires_at=clock.now() + timedelta(days=60),
        category="女装",
        buyer_nick="合成昵称",
    )
    r = client.post("/v1/platform/B/subscription", json=grant, headers=_auth("push_new"))
    assert r.status_code == 200 and r.json()["action"] == "granted", r.text
    assert r.json()["status"] == "trial"
    service.console.create_faq(NEW, question=FAQ_Q, answer=FAQ_A, merchant=MERCHANT)

    # 2. 开通后出推荐
    r = event(FAQ_Q, "e1")
    assert r.status_code == 200, r.text
    assert r.json()["suggestions"][0]["text"] == FAQ_A

    # 3. 转正、停用（停止处理）、恢复
    service.tenancy.set_status(NEW, ShopStatus.ACTIVE, actor="ops")
    service.tenancy.set_status(NEW, ShopStatus.SUSPENDED, actor="ops")
    r = event(FAQ_Q, "e2")
    assert r.status_code == 403 and r.json()["error"] == "shop_inactive"
    in_process = service.handle_event("B", taobao_event(FAQ_Q, msg_id="e2b", seller_id=NEW))
    assert in_process.suggestions == [] and in_process.no_suggestion_reason == REASON_SHOP_INACTIVE
    service.tenancy.set_status(NEW, ShopStatus.ACTIVE, actor="ops")
    assert event(FAQ_Q, "e3").json()["suggestions"]

    # 4. 续期：成功（新令牌）→ 下一轮被拒（需要重新授权）→ 平台事件 403 shop_not_authorized
    report = service.tenancy.refresh_tokens()
    assert report.of(RefreshOutcome.REFRESHED) == [NEW]
    assert unit.repos.auth.get(NEW).access_token.get_secret_value() != GRANT
    clock.advance(days=29, hours=12)  # 新令牌也快到期了
    refresher.always(NEW, AuthExpired("续期令牌被撤销（模拟）"))
    assert service.tenancy.refresh_tokens().of(RefreshOutcome.NEEDS_REAUTH) == [NEW]
    r = event(FAQ_Q, "e4")
    assert r.status_code == 403 and r.json()["error"] == "shop_not_authorized"

    # 5. 商家重新授权（续订推送）：恢复处理
    renewed = subscription_push(
        "renewed",
        NEW,
        at=clock.now(),
        access_token="tok-e2e-renewed-synthetic",
        expires_at=clock.now() + timedelta(days=365),
    )
    r = client.post("/v1/platform/B/subscription", json=renewed, headers=_auth("push_new"))
    assert r.status_code == 200 and r.json()["status"] == "active"
    assert unit.repos.auth.get(NEW).needs_reauth is False
    assert event(FAQ_Q, "e5").json()["suggestions"]

    # 6. 取消订购（平台代码在进程内推）：停止处理、删除数据、系统事件留下
    cancel = unit.parse_subscription(subscription_push("cancelled", NEW, at=clock.now()))
    on_subscription(cancel)
    assert unit.repos.auth.get(NEW) is None and unit.repos.knowledge.get_shop(NEW) is None
    assert unit.repos.audit.list_events(NEW) == []
    assert unit.repos.suggestions.list(NEW) == []
    assert event(FAQ_Q, "e6").status_code == 403
    [record] = unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.detail["shop_id"] == NEW and record.detail["reason"] == "cancelled"
    assert set(record.detail) == {"shop_id", "reason", "counts"}
    # 别的店不受影响
    assert unit.repos.knowledge.get_shop(SHOP_A) is not None and unit.auth.is_authorized(SHOP_A)

    # 7. 重新订购：一家全新的店铺
    clock.advance(minutes=10)
    again = subscription_push("subscribed", NEW, at=clock.now(), access_token="tok-e2e-again")
    r = client.post("/v1/platform/B/subscription", json=again, headers=_auth("push_new"))
    assert r.status_code == 200 and r.json()["status"] == "trial"
    assert event(FAQ_Q, "e7").json()["suggestions"] == []  # 知识随取消订购删了
    service.console.create_faq(NEW, question=FAQ_Q, answer=FAQ_A, merchant=MERCHANT)
    assert event(FAQ_Q, "e8").json()["suggestions"]

    # 串店：别的店的对接令牌推这家店的取消 → 403，什么都不改
    r = client.post(
        "/v1/platform/B/subscription",
        json=subscription_push("cancelled", NEW, at=clock.now()),
        headers=_auth("push_a"),
    )
    assert r.status_code == 403 and r.json()["error"] == "tenant_isolation"
    assert unit.repos.auth.get(NEW) is not None

    _secrets_absent(service, GRANT, GRANT_REFRESH, "tok-e2e-renewed-synthetic", "tok-refreshed-")


# ================================================================ 读者探针（502379f）


def test_reader_probe_now_fails_closed():
    """502379f 时：授权昨天过期，review_aftersale 照样读平台给出待确认；只有 MVP 只读权限，
    confirm_aftersale(同意退款) 照样调写接口成功。现在两者都在调平台之前停下。"""
    unit = build_demo_unit()
    service = AssistService(unit)
    now = unit.clock.now()
    unit.repos.auth.save(
        ShopAuth(
            shop_id="s_1001",
            access_token=SecretStr("x"),
            scopes=sorted(MVP_SCOPES),
            expires_at=now - timedelta(days=1),
        )
    )
    assert not unit.auth.is_authorized("s_1001")
    with pytest.raises(AuthExpired):
        service.review_aftersale("s_1001", "as_2001")
    assert unit.tools.calls == []  # 一次平台只读接口都没调

    unit.repos.auth.save(
        ShopAuth(
            shop_id="s_1001",
            access_token=SecretStr("x"),
            scopes=sorted(MVP_SCOPES),
            expires_at=now + timedelta(days=10),
        )
    )
    review = service.review_aftersale("s_1001", "as_2001")
    assert review.state == ReviewState.CONFIRM
    with pytest.raises(AuthExpired, match="refund.write"):
        service.confirm_aftersale(
            "s_1001",
            "as_2001",
            AfterSaleAction.AGREE_REFUND,
            operator="kf_02",
            expected_status_version=review.platform_status_version,
        )
    assert unit.action_port.calls == []  # 写接口一次都没调
    assert unit.repos.reviews.get("s_1001", "as_2001").state == ReviewState.CONFIRM


# ================================================================ PostgreSQL


TOKEN_KEY = "rodit-e2e-tenancy-token-key-synthetic"


@pytest.fixture
def pg_service(e2e_pg_dsn: str, monkeypatch: pytest.MonkeyPatch) -> AssistService:
    admin = Database(e2e_pg_dsn, app_role=None)
    with admin.session(None) as conn:
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
        repos=postgres_repos(Database(e2e_pg_dsn), clock=clock),  # 应用角色，受行级安全约束
        llm=EvidenceEchoLLM(),
        tools=tools,
        adapters=[
            TaobaoMockAdapter(tools, clock, hasher, access="B"),
            StandaloneAssistAdapter(Platform.TAOBAO, clock, hasher),
        ],
        action_port=FakeActionPort(
            Platform.TAOBAO,
            supported=SUPPORTED_ACTIONS[Platform.TAOBAO],
            gateway=tools,
            clock=clock,
        ),
        route_provider=RuleProvider(),
        qa_provider=HeuristicQAJudge(),
        token_refresher=FakeTokenRefresher(Platform.TAOBAO, clock=clock),
        parse_subscription=FakeSubscriptionParser(Platform.TAOBAO),
    )
    seed_unit(unit, world)  # 店铺 A、B
    return AssistService(unit)


def _rows(dsn: str, shop_id: str) -> dict[str, int]:
    """超级用户直接数这家店在每张业务表里的行数（绕过行级安全核对清理结果）。"""
    admin = Database(dsn, app_role=None)
    counts: dict[str, int] = {}
    with admin.session(None) as conn:
        for table in BUSINESS_TABLES:
            column = "id" if table == "shop" else "shop_id"
            row = conn.execute(
                sql.SQL("SELECT count(*) AS n FROM {} WHERE {} = %s").format(
                    sql.Identifier(table), sql.Identifier(column)
                ),
                (shop_id,),
            ).fetchone()
            counts[table] = row["n"]
    return counts


@pytest.mark.pg
def test_tenant_lifecycle_on_postgres(pg_service: AssistService, e2e_pg_dsn: str):
    service = pg_service
    unit = service.unit
    clock = unit.clock
    now = clock.now()

    # 开通（加密落库）→ 出推荐
    event = unit.parse_subscription(
        subscription_push(
            "subscribed",
            NEW,
            at=now,
            access_token=GRANT,
            refresh_token=GRANT_REFRESH,
            expires_at=now + timedelta(hours=6),
            refresh_expires_at=now + timedelta(days=60),
        )
    )
    outcome = service.tenancy.handle_subscription(event)
    assert outcome.action == "granted"
    admin = Database(e2e_pg_dsn, app_role=None)
    with admin.session(None) as conn:
        row = conn.execute(
            "SELECT access_token, refresh_token FROM shop_auth WHERE shop_id = %s", (NEW,)
        ).fetchone()
    assert GRANT.encode() not in bytes(row["access_token"])
    assert GRANT_REFRESH.encode() not in bytes(row["refresh_token"])
    service.console.create_faq(NEW, question=FAQ_Q, answer=FAQ_A, merchant=MERCHANT)
    result = service.handle_event("B", taobao_event(FAQ_Q, msg_id="pg1", seller_id=NEW))
    assert result is not None and result.suggestions

    # 续期：跨店列出到期店铺只经 rodit_auth_expiring（只有 ID 和到期时间）
    assert [s for s, _ in unit.repos.auth.expiring_shop_ids(now + timedelta(hours=24))] == [NEW]
    report = service.tenancy.refresh_tokens()
    assert report.of(RefreshOutcome.REFRESHED) == [NEW]
    refreshed = unit.repos.auth.get(NEW)
    assert refreshed.refreshed_at == now and refreshed.expires_at == now + timedelta(days=30)
    assert unit.repos.audit.list_events(NEW, action=AuditAction.AUTH_REFRESHED)

    # 过期超过宽限期：只清理这家；清理之后系统事件还在，店铺 A、B 不受影响
    for shop_id in (SHOP_A, SHOP_B):  # A、B 续订了一年
        auth = unit.repos.auth.get(shop_id)
        unit.repos.auth.save(auth.model_copy(update={"expires_at": now + timedelta(days=365)}))
    clock.advance(days=61)  # 新令牌 30 天后到期 → 已过期 31 天，超过 30 天宽限期
    assert service.tenancy.purge_expired(dry_run=True) == {NEW: {}}
    purged = service.tenancy.purge_expired()
    assert list(purged) == [NEW]
    assert sum(_rows(e2e_pg_dsn, NEW).values()) == 0
    assert _rows(e2e_pg_dsn, SHOP_A)["shop"] == 1 and _rows(e2e_pg_dsn, SHOP_B)["shop"] == 1
    [record] = unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.shop_id is None and record.detail["shop_id"] == NEW
    assert record.detail["reason"] == "expired" and record.detail["counts"] == purged[NEW]
    # 店铺范围的查询看不到系统事件（行级安全）
    assert unit.repos.audit.list_events(NEW) == []

    # 重新开通
    again = unit.parse_subscription(
        subscription_push("subscribed", NEW, at=clock.now(), access_token="tok-pg-again")
    )
    assert service.tenancy.handle_subscription(again).action == "granted"
    assert unit.auth.is_authorized(NEW)
    assert unit.repos.knowledge.get_shop(NEW).status == ShopStatus.TRIAL


@pytest.mark.pg
def test_cancel_and_auth_checks_on_postgres(pg_service: AssistService):
    service = pg_service
    unit = service.unit
    # 只有 MVP 只读权限（演示单元带写接口时默认给了写售后权限，这里收回）：写售后在调写接口之前被拒
    auth = unit.repos.auth.get(SHOP_A)
    unit.repos.auth.save(auth.model_copy(update={"scopes": sorted(MVP_SCOPES)}))
    review = service.review_aftersale(SHOP_A, "as_2001")
    with pytest.raises(AuthExpired, match="refund.write"):
        service.confirm_aftersale(
            SHOP_A,
            "as_2001",
            AfterSaleAction.AGREE_REFUND,
            operator=AGENT,
            expected_status_version=review.platform_status_version,
        )
    assert unit.action_port.calls == []

    cancel = unit.parse_subscription(subscription_push("auth_revoked", SHOP_A, at=unit.clock.now()))
    outcome = service.tenancy.handle_subscription(cancel)
    assert outcome.action == "revoked" and outcome.counts["auth"] == 1
    with pytest.raises(AuthExpired):
        service.review_aftersale(SHOP_A, "as_2001")
    [record] = unit.repos.audit.list_system_events(action=AuditAction.SHOP_PURGED)
    assert record.detail == {"shop_id": SHOP_A, "reason": "auth_revoked", "counts": outcome.counts}
    assert unit.repos.knowledge.get_shop(SHOP_B) is not None
