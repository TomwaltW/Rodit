"""售后审核工作台接口（§7.3.2 商家侧审核助手、§15 Phase 2、§2.1 售后平均审核时长）。

- 待审核列表：平台待审核 + 本地还没结束的审核记录（平台列表里没有的退货中等），排序、筛选、
  平台变化提示、平台列表读不到时降级（§12.2）；GET 不写任何东西；不返回 buyer_ref、说明打码；
- 批量拉取并跑规则：和单张一样留痕，单张失败只记在那一行，授权失效时停下；
- 详情：各平台能做的操作和不能做的原因（拼多多没有拒绝、淘宝拒绝先协商、抖店拒绝要原因码、
  MVP 没有写接口 / 写权限、R6 店长、状态版本变了要刷新），重读平台失败时退回缓存；
- 统计：各状态单数、规则分布、平均审核时长（审计 + 审核记录），since 过滤；
- 鉴权：角色、串店、店铺停用、授权过期（一次平台接口都不调）、取消订购；仓储 / 网关串店的纵深防御。
数据全部来自合成演示世界，不联网。
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from api_support import CALLERS, TOKENS, Api, TestClient, build_api, shop_snapshot

from rodit.aftersale.assistant import (
    MSG_DOUYIN_REFUSE,
    MSG_MANAGER,
    MSG_NO_PORT,
    MSG_PDD_REFUSE,
    MSG_STALE,
    MSG_TAOBAO_REFUSE,
)
from rodit.api import StaticTokenAuthenticator, create_app
from rodit.app import AssistService
from rodit.app.aftersale_desk import (
    MSG_NO_WRITE_SCOPE,
    MSG_NOT_REVIEWED,
    MSG_PLATFORM_UNAVAILABLE,
    MSG_RETURN_FOR_REFUND_ONLY,
    PLATFORM_NOTES,
)
from rodit.app.demo import build_demo_unit
from rodit.core.audit import AuditAction
from rodit.core.enums import (
    AfterSaleAction,
    AfterSaleReason,
    AfterSaleStatus,
    AfterSaleType,
    Platform,
    ReviewState,
)
from rodit.core.errors import ConcurrencyConflict, ToolUnavailable
from rodit.core.models import AfterSaleInfo
from rodit.core.scopes import MVP_SCOPES
from rodit.testing.fixtures import (
    AGENT,
    BUYER_DOUYIN,
    BUYER_PDD,
    MANAGER,
    SHOP_A,
    SHOP_B,
    SHOP_C,
    SHOP_D,
)

A = f"/v1/shops/{SHOP_A}/aftersales"
PENDING_A = ["as_2007", "as_2004", "as_2002", "as_2005", "as_2001", "as_2003", "as_3004", "as_2009"]
"""店铺 A 平台上待审核的售后单，按申请时间从早到晚（同一时间按 ID）。"""

DESK_PATHS = (A, f"{A}/stats", f"{A}/as_2001")


def _queue(api: Api, who: str = "agent_a", base: str = A, **params) -> dict:
    r = api.get(base, who, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _ids(queue: dict) -> list[str]:
    return [row["aftersale_id"] for row in queue["rows"]]


def _rows(queue: dict) -> dict[str, dict]:
    return {row["aftersale_id"]: row for row in queue["rows"]}


def _run(api: Api, who: str = "agent_a", base: str = A, **body) -> dict:
    r = api.post(f"{base}/review-pending", who, json=body or None)
    assert r.status_code == 200, r.text
    return r.json()


def _detail(api: Api, aftersale_id: str, who: str = "agent_a", base: str = A) -> dict:
    r = api.get(f"{base}/{aftersale_id}", who)
    assert r.status_code == 200, r.text
    return r.json()


def _actions(detail: dict) -> dict[str, dict]:
    return {a["action"]: a for a in detail["actions"]}


def _confirm(
    api: Api, aftersale_id: str, action: str, who: str = "agent_a", base: str = A, **extra
):
    version = _detail(api, aftersale_id, who, base)["platform"]["status_version"]
    body = {"action": action, "expected_status_version": version, **extra}
    return api.post(f"{base}/{aftersale_id}/confirm", who, json=body)


def _expire_cache(api: Api, minutes: int = 10) -> None:
    """往后拨时钟，让平台数据的短时缓存过期（TTL 5 分钟，§12.1）。"""
    api.unit.clock.advance(minutes=minutes)


STALE_LIMIT_MINUTES = 61
"""平台接口故障时网关最多用 1 小时内的缓存兜底（rodit.app.unit.TOOL_CACHE_MAX_STALE_S）：
拨过这个时间，故障才会让调用方看到。"""


def _api_for(unit) -> Api:
    service = AssistService(unit)
    authenticator = StaticTokenAuthenticator({TOKENS[n]: p for n, p in CALLERS.items()})
    return Api(TestClient(create_app(service, authenticator=authenticator)), service)


def _add_aftersale(api: Api, shop_id: str, aftersale_id: str, buyer_ref: str, **fields) -> None:
    now = api.unit.clock.now()
    created = now - timedelta(hours=2)
    info = AfterSaleInfo(
        shop_id=shop_id,
        aftersale_id=aftersale_id,
        order_id=fields.pop("order_id", "o_x1"),
        buyer_ref=buyer_ref,
        type=fields.pop("type", AfterSaleType.REFUND_ONLY),
        reason=fields.pop("reason", AfterSaleReason.OTHER),
        status=AfterSaleStatus.PENDING_REVIEW,
        amount=Decimal(fields.pop("amount", "89.00")),
        created_at=created,
        updated_at=created,
        status_version=str(int(created.timestamp())),
        **fields,
    )
    api.unit.tools.world.aftersales.append(info)


# ================================================================ 待审核列表


def test_queue_lists_platform_pending_as_unreviewed_oldest_first(api: Api):
    q = _queue(api)
    assert _ids(q) == PENDING_A
    assert (q["total"], q["needs_action"], q["unreviewed"]) == (8, 8, 8)
    assert q["platform_unavailable"] is False and q["as_of"] == api.unit.clock.now().isoformat()
    row = _rows(q)["as_2001"]
    assert row == {
        **row,
        "order_id": "o_1001",
        "type": "refund_only",
        "reason": "other",
        "platform_status": "pending_review",
        "amount": "259.00",
        "waiting_s": int(0.1 * 86400),
        "evidence_count": 0,
        "reviewed": False,
        "state": None,
        "suggestion": None,
        "matched_rules": [],
        "next_step": "run_rules",
        "needs_action": True,
        "platform_changed": False,
    }
    assert _rows(q)["as_2003"]["note"] == "显示签收了但我根本没收到"
    assert _rows(q)["as_2009"]["evidence_count"] == 3
    # 别的店铺、已经不是待审核的售后单不在列表里
    assert not {"as_9001", "as_2006", "as_2008", "as_3001"} & set(_ids(q))


def test_queue_never_exposes_buyer_ref_and_masks_notes_defensively(api: Api, monkeypatch):
    """网关协议要求说明已打码；网关实现漏了也不往外给手机号（§5.3），也不返回 buyer_ref。"""
    tools = api.unit.tools
    original = tools.list_aftersales

    def leaky(shop_id, **kwargs):
        out = original(shop_id, **kwargs)
        return [
            a.model_copy(update={"note": "我的手机 13812345678，<b>快点</b>"})
            if a.aftersale_id == "as_2001"
            else a
            for a in out
        ]

    monkeypatch.setattr(tools, "list_aftersales", leaky)
    r = api.get(A, "agent_a")
    assert r.status_code == 200
    assert "13812345678" not in r.text
    note = _rows(r.json())["as_2001"]["note"]
    assert "138" not in note and "<b>快点</b>" in note  # 原样当文字，不做 HTML 处理
    buyer_refs = {a.buyer_ref for a in tools.world.aftersales}
    assert not any(ref in r.text for ref in buyer_refs)
    assert all("buyer_ref" not in row for row in r.json()["rows"])
    _run(api)
    for path in (*DESK_PATHS, f"{A}/as_2003"):
        text = api.get(path, "agent_a").text
        assert "buyer_ref" not in text and not any(ref in text for ref in buyer_refs), path


def test_long_notes_are_truncated(api: Api):
    world = api.unit.tools.world
    i = next(i for i, a in enumerate(world.aftersales) if a.aftersale_id == "as_2007")
    world.aftersales[i] = world.aftersales[i].model_copy(update={"note": "太大了" * 200})
    note = _rows(_queue(api))["as_2007"]["note"]
    assert len(note) == 200 and note.endswith("…")
    assert _detail(api, "as_2007")["ticket"]["note"] == note


def test_get_endpoints_have_no_side_effects(api: Api):
    """GET 列表、统计、详情不建审核记录、不推进状态机、不写审计、不调写接口（§7.3.2 只读）。"""
    _run(api)
    _expire_cache(api, minutes=60 * 25)  # 签收核查已满时：刷新才会推进，GET 不推进
    before = shop_snapshot(api, SHOP_A)
    events = len(api.unit.repos.audit.list_events(SHOP_A))
    for path in (*DESK_PATHS, f"{A}/as_2003", f"{A}/as_2005"):
        assert api.get(path, "agent_a").status_code == 200, path
    for params in ({"state": "unreviewed"}, {"state": "agreed"}, {"rule": "R4"}):
        assert api.get(A, "manager_a", params=params).status_code == 200
    assert shop_snapshot(api, SHOP_A) == before
    assert len(api.unit.repos.audit.list_events(SHOP_A)) == events
    assert api.unit.repos.reviews.get(SHOP_A, "as_2003").state == ReviewState.VERIFY
    assert _rows(_queue(api))["as_2003"]["next_step"] == "refresh"


def test_unreviewed_detail_does_not_create_a_review(api: Api):
    d = _detail(api, "as_2007")
    assert d["review"] is None and d["next_step"] == "run_rules" and d["can_run_rules"] is True
    assert api.unit.repos.reviews.get(SHOP_A, "as_2007") is None
    assert api.unit.repos.audit.list_events(SHOP_A) == []


# ---------------------------------------------------------------- 合并与排序


def test_queue_after_rules_sorts_actionable_first_then_oldest(api: Api):
    _run(api)
    q = _queue(api)
    assert _ids(q) == [
        # 待客服确认（要处理）按申请时间从早到晚
        "as_2007",
        "as_2002",
        "as_2005",
        "as_2001",
        "as_3004",
        "as_2009",
        # 等凭证、等签收核查（在等别人）
        "as_2004",
        "as_2003",
    ]
    rows = _rows(q)
    assert (q["total"], q["needs_action"], q["unreviewed"]) == (8, 6, 0)
    assert (
        rows["as_2001"]
        | {
            "state": "confirm",
            "suggestion": "agree",
            "matched_rules": ["R2"],
            "next_step": "confirm",
            "needs_action": True,
        }
        == rows["as_2001"]
    )
    assert rows["as_2005"]["requires_manager"] is True
    assert rows["as_2005"]["matched_rules"] == ["R3", "R6"]
    assert rows["as_3004"]["risk_flags"] and rows["as_3004"]["suggestion"] == "manual"
    assert (rows["as_2003"]["state"], rows["as_2003"]["next_step"]) == ("verify", "wait_verify")
    assert (rows["as_2004"]["state"], rows["as_2004"]["next_step"]) == (
        "evidence",
        "wait_evidence",
    )
    assert "驿站" in rows["as_2003"]["reason_text"] and "24 小时" in rows["as_2003"]["reason_text"]


def test_queue_keeps_open_reviews_that_left_the_platform_pending_list(api: Api):
    """退货中（R3 同意退货后）平台状态是等买家寄回，不在平台待审核列表里，工作台照样列出；
    买家寄回后提示刷新，刷新给出同意退款的建议（R3 第二步）。"""
    _run(api)
    assert _confirm(api, "as_2002", "agree_return").json()["state"] == "wait_return"
    row = _rows(_queue(api))["as_2002"]  # 平台列表还是缓存里的旧数据：审核记录更新，以它为准
    assert (row["next_step"], row["platform_status"], row["needs_action"]) == (
        "wait_return",
        "wait_buyer_return",
        False,
    )
    _expire_cache(api)
    row = _rows(_queue(api))["as_2002"]  # 平台列表里已经没有它：从本地审核记录补上
    assert (row["next_step"], row["platform_status"], row["order_id"]) == (
        "wait_return",
        "wait_buyer_return",
        "o_1004",
    )
    now = api.unit.clock.now()
    api.unit.tools.set_aftersale_status(
        SHOP_A, "as_2002", AfterSaleStatus.WAIT_SELLER_CONFIRM, at=now
    )
    _expire_cache(api)
    row = _rows(_queue(api))["as_2002"]
    assert (row["next_step"], row["platform_changed"], row["needs_action"]) == (
        "refresh",
        True,
        True,
    )
    assert row["platform_status"] == "wait_seller_confirm"
    r = api.post(f"{A}/as_2002/review", "agent_a")
    assert (r.json()["state"], r.json()["suggestion"]) == ("confirm", "agree")
    row = _rows(_queue(api))["as_2002"]
    assert (row["next_step"], row["platform_changed"]) == ("confirm", False)
    d = _detail(api, "as_2002")
    assert _actions(d)["agree_refund"]["available"] and _actions(d)["agree_refund"]["recommended"]


def test_finished_reviews_leave_the_default_queue_and_show_under_their_state(api: Api):
    _run(api)
    r = _confirm(api, "as_2001", "agree_refund")
    assert r.status_code == 200 and r.json()["state"] == "agreed"
    # 缓存里的平台列表还有它（待审核）：审核记录已结束，不再列出
    assert "as_2001" not in _ids(_queue(api))
    agreed = _queue(api, state="agreed")
    assert _ids(agreed) == ["as_2001"] and agreed["total"] == 1
    row = agreed["rows"][0]
    assert (row["next_step"], row["needs_action"], row["waiting_s"]) == ("done", False, None)
    assert row["platform_status"] == "refunding" and row["platform_changed"] is False
    assert _queue(api, state="agreed", rule="R7")["rows"] == []
    assert _queue(api, state="agreed", status="refunding")["total"] == 1
    assert _queue(api, state="refused")["total"] == 0


def test_queue_flags_platform_changes_and_refresh_closes_them(api: Api):
    """待确认期间买家撤销：缓存过期后提示刷新；刷新进入"平台已处理"（R1），离开列表。"""
    _run(api)
    api.unit.tools.set_aftersale_status(
        SHOP_A, "as_2001", AfterSaleStatus.CLOSED, at=api.unit.clock.now()
    )
    assert _rows(_queue(api))["as_2001"]["next_step"] == "confirm"  # 缓存还没过期
    _expire_cache(api)
    row = _rows(_queue(api))["as_2001"]
    assert (row["next_step"], row["platform_changed"], row["platform_status"]) == (
        "refresh",
        True,
        "closed",
    )
    d = _detail(api, "as_2001")  # 详情重读平台（不走缓存）
    assert d["next_step"] == "refresh"
    assert _actions(d)["agree_refund"]["blocked_reason"] == "售后单当前是「已关闭」，不能同意退款"
    assert api.post(f"{A}/as_2001/review", "agent_a").json()["state"] == "platform"
    assert "as_2001" not in _ids(_queue(api))
    assert _ids(_queue(api, state="platform")) == ["as_2001"]


def test_verify_timeout_and_new_evidence_ask_for_a_refresh(api: Api):
    _run(api)
    api.unit.tools.set_aftersale_status(
        SHOP_A, "as_2004", AfterSaleStatus.PENDING_REVIEW, at=api.unit.clock.now(), evidence_count=2
    )
    _expire_cache(api)
    rows = _rows(_queue(api))
    assert (rows["as_2004"]["next_step"], rows["as_2004"]["evidence_count"]) == ("refresh", 2)
    assert rows["as_2003"]["next_step"] == "wait_verify"
    _expire_cache(api, minutes=60 * 24)
    assert _rows(_queue(api))["as_2003"]["next_step"] == "refresh"  # 签收核查满 24 小时
    assert api.post(f"{A}/as_2003/review", "agent_a").json()["state"] == "confirm"
    assert api.post(f"{A}/as_2004/review", "agent_a").json()["state"] == "confirm"
    rows = _rows(_queue(api))
    assert rows["as_2003"]["next_step"] == rows["as_2004"]["next_step"] == "confirm"


def test_queue_filters_and_limit(api: Api):
    _run(api, limit=4)  # as_2007、as_2004、as_2002、as_2005
    assert _ids(_queue(api, state="unreviewed")) == ["as_2001", "as_2003", "as_3004", "as_2009"]
    assert _ids(_queue(api, state="confirm")) == ["as_2007", "as_2002", "as_2005"]
    assert _ids(_queue(api, state="evidence")) == ["as_2004"]
    assert _ids(_queue(api, rule="R6")) == ["as_2005"]
    assert _ids(_queue(api, rule="R3")) == ["as_2002", "as_2005"]
    assert _queue(api, status="pending_review")["total"] == 8
    assert _queue(api, status="refunding")["total"] == 0
    limited = _queue(api, limit=2)
    assert len(limited["rows"]) == 2 and limited["total"] == 8
    assert limited["needs_action"] == 7  # 筛选后、截断前的条数


@pytest.mark.parametrize(
    "params",
    [
        {"state": "bogus"},
        {"state": "CONFIRM"},
        {"rule": "R9"},
        {"rule": "r1"},
        {"status": "bogus"},
        {"limit": 0},
        {"limit": 201},
    ],
)
def test_queue_rejects_bad_filters(api: Api, params):
    r = api.get(A, "agent_a", params=params)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_queue_degrades_to_local_reviews_when_the_platform_is_down(api: Api):
    """平台待审核列表读不到（§12.2）：只列本地还没结束的审核记录，并标明；不报 503。"""
    _run(api)
    api.unit.tools.fail_on |= {"list_aftersales", "get_aftersale"}
    _expire_cache(api)
    assert _queue(api)["platform_unavailable"] is False  # 1 小时内：网关用缓存兜底
    _expire_cache(api, minutes=STALE_LIMIT_MINUTES)
    q = _queue(api)
    assert q["platform_unavailable"] is True
    assert sorted(_ids(q)) == sorted(PENDING_A)
    row = _rows(q)["as_2001"]
    assert row["platform_unavailable"] is True and row["state"] == "confirm"
    assert row["platform_status"] == "pending_review" and row["order_id"] is None
    stats = api.get(f"{A}/stats", "agent_a").json()
    assert stats["platform_unavailable"] is True and stats["open_total"] == 8


def test_queue_defends_against_repositories_and_gateways_leaking_other_shops(api: Api, monkeypatch):
    """§9.2 纵深防御：仓储、网关返回了别的店铺的数据也不往外给。"""
    assert api.post(f"/v1/shops/{SHOP_B}/aftersales/as_9001/review", "agent_b").status_code == 200
    repo = api.unit.repos.reviews
    original_list = repo.list_by_state
    leaked = repo.get(SHOP_B, "as_9001")
    monkeypatch.setattr(
        repo,
        "list_by_state",
        lambda shop_id, states=None: [*original_list(shop_id, states), leaked],
    )
    tools = api.unit.tools
    original = tools.list_aftersales
    other = next(a for a in tools.world.aftersales if a.aftersale_id == "as_9001")
    monkeypatch.setattr(
        tools, "list_aftersales", lambda shop_id, **kw: [*original(shop_id, **kw), other]
    )
    # 缓存网关自己就会拒绝串店数据（TenantIsolationError → 403）；这里绕过它看工作台本身
    monkeypatch.setattr(
        api.unit.gateway, "list_aftersales", lambda shop_id, **kw: [*original(shop_id, **kw), other]
    )
    r = api.get(A, "agent_a")
    assert r.status_code == 200 and "as_9001" not in r.text
    stats = api.get(f"{A}/stats", "agent_a").json()
    assert stats["open_total"] == 8 and stats["by_state"]["confirm"] == 0


def test_each_shop_sees_only_its_own_queue(api: Api):
    q = _queue(api, "agent_b", base=f"/v1/shops/{SHOP_B}/aftersales")
    assert _ids(q) == ["as_9001"]
    assert not set(PENDING_A) & set(_ids(q))
    r = api.get(f"/v1/shops/{SHOP_B}/aftersales/as_2001", "agent_b")
    assert r.status_code == 404 and r.json()["error"] == "not_found"  # 店铺 A 的单在 B 查不到


# ================================================================ 批量拉取并跑规则


def test_batch_review_runs_rules_like_the_single_endpoint(api: Api):
    result = _run(api)
    assert (result["attempted"], result["succeeded"], result["failed"]) == (8, 8, 0)
    assert result["remaining"] == 0 and result["stopped"] is None
    rows = {r["aftersale_id"]: r for r in result["rows"]}
    assert [r["aftersale_id"] for r in result["rows"]] == PENDING_A  # 从最早的开始
    assert rows["as_2001"] == {
        "aftersale_id": "as_2001",
        "ok": True,
        "state": "confirm",
        "suggestion": "agree",
        "matched_rules": ["R2"],
        "requires_manager": False,
        "error": None,
        "message": None,
    }
    assert (rows["as_2003"]["state"], rows["as_2003"]["matched_rules"]) == ("verify", ["R4"])
    assert rows["as_2005"]["requires_manager"] is True
    # 留痕和单张接口一样：每张一条规则建议，状态迁移都有审计
    audit = api.unit.repos.audit
    suggested = {
        e.detail["aftersale_id"] for e in audit.list_events(SHOP_A, action="aftersale.suggest")
    }
    assert suggested == set(PENDING_A)
    assert len(audit.list_events(SHOP_A, action=AuditAction.REVIEW_TRANSITION.value)) >= 16
    assert api.unit.action_port.calls == []  # 只读：不调写接口
    # 再跑一次：都有建议了，没有要跑的
    again = _run(api)
    assert (again["attempted"], again["remaining"]) == (0, 0)


def test_batch_review_respects_the_limit(api: Api):
    first = _run(api, limit=3)
    assert [r["aftersale_id"] for r in first["rows"]] == PENDING_A[:3]
    assert first["remaining"] == 5
    second = _run(api, limit=3)
    assert [r["aftersale_id"] for r in second["rows"]] == PENDING_A[3:6]
    assert second["remaining"] == 2


@pytest.mark.parametrize(
    "body", [{"limit": 0}, {"limit": 51}, {"operator": "kf_02"}, {"limit": "x"}]
)
def test_batch_review_rejects_bad_bodies(api: Api, body):
    r = api.post(f"{A}/review-pending", "agent_a", json=body)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert api.unit.repos.reviews.list_by_state(SHOP_A) == []


def test_batch_review_reports_failures_per_row(api: Api, monkeypatch):
    """一张失败（平台接口故障、并发冲突）只记在这一行，其他照常，整个请求 200。"""
    tools = api.unit.tools
    original_get = tools.get_aftersale

    def flaky_get(shop_id, aftersale_id):
        if aftersale_id == "as_2004":
            raise ToolUnavailable("as_2004 限流（模拟）", reason="throttled")
        return original_get(shop_id, aftersale_id)

    repo = api.unit.repos.reviews
    original_save = repo.save

    def racing_save(review, *, expected_version):
        if review.aftersale_id == "as_2002":
            raise ConcurrencyConflict("另一位客服刚刚处理了这张售后单，手机 13812345678")
        return original_save(review, expected_version=expected_version)

    monkeypatch.setattr(tools, "get_aftersale", flaky_get)
    monkeypatch.setattr(repo, "save", racing_save)
    result = _run(api)
    assert (result["attempted"], result["succeeded"], result["failed"]) == (8, 6, 2)
    rows = {r["aftersale_id"]: r for r in result["rows"]}
    assert rows["as_2004"] | {"ok": False, "error": "tool_unavailable"} == rows["as_2004"]
    assert rows["as_2004"]["message"] == "平台接口暂时不可用，请稍后重试"  # 固定说明
    assert (
        rows["as_2002"]["error"] == "conflict" and "13812345678" not in rows["as_2002"]["message"]
    )
    assert rows["as_2002"]["state"] is None
    assert rows["as_2001"]["ok"] is True
    assert repo.get(SHOP_A, "as_2004") is None and repo.get(SHOP_A, "as_2002") is None
    # 故障恢复后再跑：只跑刚才失败的两张
    monkeypatch.undo()
    again = _run(api)
    assert sorted(r["aftersale_id"] for r in again["rows"]) == ["as_2002", "as_2004"]
    assert again["succeeded"] == 2


def test_batch_review_stops_when_authorization_is_lost_midway(api: Api, monkeypatch):
    service = api.service
    real = service.review_aftersale
    calls: list[str] = []

    def review_then_revoke_auth(shop_id: str, aftersale_id: str):
        calls.append(aftersale_id)
        if len(calls) == 2:
            api.unit.repos.auth.delete(shop_id)  # 商家撤销了授权（§8.4）
        return real(shop_id, aftersale_id)

    monkeypatch.setattr(service, "review_aftersale", review_then_revoke_auth)
    result = _run(api)
    assert calls == PENDING_A[:2]  # 停下之后不再跑
    assert [r["ok"] for r in result["rows"]] == [True, False]
    assert result["rows"][1]["error"] == "shop_not_authorized"
    assert result["stopped"] and "授权" in result["stopped"]
    assert result["remaining"] == 6


def test_batch_review_needs_the_platform_list(api: Api):
    api.unit.tools.fail_on.add("list_aftersales")
    r = api.post(f"{A}/review-pending", "agent_a")
    assert r.status_code == 503 and r.json()["error"] == "tool_unavailable"
    assert api.unit.repos.reviews.list_by_state(SHOP_A) == []


# ================================================================ 详情


def test_detail_for_r2_on_taobao(api: Api):
    _run(api)
    d = _detail(api, "as_2001")
    info = api.unit.tools.world.aftersale(SHOP_A, "as_2001")
    assert d["platform"] == {
        "status": "pending_review",
        "status_version": info.status_version,
        "type": "refund_only",
        "reason": "other",
        "amount": "259.00",
        "evidence_count": 0,
    }
    assert d["ticket"]["order_id"] == "o_1001" and d["ticket"]["waiting_s"] == int(0.1 * 86400)
    review = d["review"]
    assert (review["state"], review["suggestion"], review["matched_rules"]) == (
        "confirm",
        "agree",
        ["R2"],
    )
    assert review["platform"]["status_version"] == info.status_version  # 待确认时附上当前状态
    assert review["needs_refresh"] is False
    assert [h["to_state"] for h in review["history"]] == ["suggest", "confirm"]
    actions = _actions(d)
    assert list(actions) == ["agree_refund", "agree_return", "refuse"]
    assert actions["agree_refund"] == {
        "action": "agree_refund",
        "label": "同意退款",
        "available": True,
        "recommended": True,
        "blocked_reason": None,
        "needs_reason_code": False,
    }
    assert actions["agree_return"]["blocked_reason"] == MSG_RETURN_FOR_REFUND_ONLY
    assert actions["refuse"]["blocked_reason"] == MSG_TAOBAO_REFUSE  # 淘宝拒绝要先协商
    assert d["recommended_action"] == "agree_refund" and d["next_step"] == "confirm"
    assert (d["write_enabled"], d["write_scope"], d["manager_required"]) == (True, True, False)
    assert (d["can_negotiate"], d["can_finish_negotiation"], d["can_refresh"]) == (
        True,
        False,
        True,
    )
    assert d["can_run_rules"] is False and d["negotiated"] is False
    assert d["supported_actions"] == ["agree_refund", "agree_return", "refuse"]
    assert d["platform_note"] == PLATFORM_NOTES[Platform.TAOBAO] and "协商" in d["refuse_hint"]


def test_detail_before_rules_offers_to_run_them(api: Api):
    d = _detail(api, "as_2007")
    assert d["review"] is None and d["can_run_rules"] is True and d["can_refresh"] is False
    assert {a["blocked_reason"] for a in d["actions"]} == {MSG_NOT_REVIEWED}
    assert d["platform"]["status"] == "pending_review" and d["next_step"] == "run_rules"
    r = api.post(f"{A}/as_2007/review", "agent_a")
    assert r.json()["state"] == "confirm"
    d = _detail(api, "as_2007")
    assert d["can_run_rules"] is False and d["review"]["matched_rules"] == ["R8"]


def test_detail_r6_needs_a_listed_manager(api: Api):
    """R6：客服、不在店长名单里的"店长"都不能确认，按钮附原因；名单里的店长可以。"""
    _run(api)
    agent = _detail(api, "as_2005")
    assert agent["manager_required"] is True and agent["caller_is_manager"] is False
    assert {a["blocked_reason"] for a in agent["actions"] if a["action"] != "refuse"} == {
        MSG_MANAGER
    }
    assert _actions(agent)["agree_return"]["recommended"] is True
    fake = _detail(api, "as_2005", "fake_manager_a")
    assert fake["caller_is_manager"] is False and not _actions(fake)["agree_return"]["available"]
    manager = _detail(api, "as_2005", "manager_a")
    assert manager["caller_is_manager"] is True
    assert _actions(manager)["agree_return"]["available"] is True
    assert _actions(manager)["agree_refund"]["available"] is True
    # 页面按钮和接口一致：客服确认 403，店长确认成功
    r = _confirm(api, "as_2005", "agree_return")
    assert r.status_code == 403 and r.json()["error"] == "manager_required"
    r = _confirm(api, "as_2005", "agree_return", "manager_a")
    assert r.status_code == 200 and r.json()["state"] == "wait_return"
    assert api.unit.action_port.calls[-1]["operator"] == MANAGER


def test_detail_in_verify_explains_why_nothing_can_be_done_yet(api: Api):
    _run(api)
    d = _detail(api, "as_2003")
    assert d["review"]["state"] == "verify" and d["review"]["verify_started_at"]
    assert {a["blocked_reason"] for a in d["actions"]} == {
        "当前是「签收核查中」，只有待客服确认时才能执行"
    }
    assert d["can_refresh"] is True and d["can_negotiate"] is False
    assert d["next_step"] == "wait_verify" and d["ticket"]["note"] == "显示签收了但我根本没收到"


def test_taobao_refuse_opens_up_after_negotiation(api: Api):
    _run(api)
    assert api.post(f"{A}/as_2007/negotiate", "agent_a", json={"note": "先协商"}).status_code == 200
    d = _detail(api, "as_2007")
    assert d["can_finish_negotiation"] is True and d["next_step"] == "negotiating"
    assert d["negotiated"] is True
    r = api.post(f"{A}/as_2007/negotiation-done", "agent_a", json={"outcome": "买家坚持退货"})
    assert r.json()["state"] == "confirm"
    d = _detail(api, "as_2007")
    assert _actions(d)["refuse"]["available"] is True
    r = _confirm(api, "as_2007", "refuse", reason_text="定制商品不支持七天无理由")
    assert r.status_code == 200 and r.json()["state"] == "refused"


def test_detail_blocks_confirmation_when_the_platform_version_moved(api: Api):
    """建议是按旧版本的售后单给的（§7.3.2 调用前重新读取）：先刷新，再确认。"""
    _run(api)
    api.unit.tools.set_aftersale_status(
        SHOP_A, "as_2001", AfterSaleStatus.PENDING_REVIEW, at=api.unit.clock.now()
    )
    d = _detail(api, "as_2001")
    assert d["review"]["needs_refresh"] is True and d["next_step"] == "refresh"
    assert _actions(d)["agree_refund"]["blocked_reason"] == MSG_STALE
    r = _confirm(api, "as_2001", "agree_refund")  # 绕过页面直接确认：审核助手照样拒绝
    assert r.status_code == 409 and r.json()["error"] == "action_not_allowed"
    assert "状态已变化" in r.json()["message"]
    assert api.unit.action_port.calls == []
    api.post(f"{A}/as_2001/review", "agent_a")
    d = _detail(api, "as_2001")
    assert d["review"]["needs_refresh"] is False and _actions(d)["agree_refund"]["available"]


def test_detail_unknown_aftersale_is_404(api: Api):
    for aftersale_id in ("as_nope", "as_9001"):  # as_9001 是店铺 B 的
        r = api.get(f"{A}/{aftersale_id}", "agent_a")
        assert r.status_code == 404 and r.json()["error"] == "not_found"
        assert "129.00" not in r.text


def test_detail_falls_back_to_cache_when_the_platform_read_fails(api: Api):
    _run(api)
    api.unit.gateway.get_aftersale(SHOP_A, "as_2001")  # 缓存里有一份（其他页面读过）
    api.unit.tools.fail_on.add("get_aftersale")
    d = _detail(api, "as_2001")
    assert d["platform_unavailable"] is True
    assert d["platform"]["status"] == "pending_review"  # 缓存数据，只用来显示
    assert d["ticket"]["fetched_at"] == api.unit.clock.now().isoformat()
    assert d["review"]["platform"] is None  # 不拿缓存冒充平台当前状态
    assert {a["blocked_reason"] for a in d["actions"] if a["action"] == "agree_refund"} == {
        MSG_PLATFORM_UNAVAILABLE
    }
    # 没有缓存：有审核记录时照样返回，平台部分为空
    d = _detail(api, "as_2002")
    assert d["platform"] is None and d["ticket"] is None and d["review"]["state"] == "confirm"
    assert not any(a["available"] for a in d["actions"])
    # 没有缓存也没有审核记录：不知道有没有这张单 → 503
    api.unit.repos.reviews.purge_shop(SHOP_A)
    r = api.get(f"{A}/as_2007", "agent_a")
    assert r.status_code == 503 and r.json()["error"] == "tool_unavailable"


def test_detail_after_confirmation_is_final(api: Api):
    _run(api)
    assert _confirm(api, "as_2001", "agree_refund").json()["state"] == "agreed"
    d = _detail(api, "as_2001")
    review = d["review"]
    assert review["state"] == "agreed" and review["confirmed_by"] == AGENT
    assert review["last_result"]["ok"] is True and review["last_result"]["action"] == "agree_refund"
    assert (
        review["history"][-1] | {"from_state": "confirm", "to_state": "agreed", "by": AGENT}
        == (review["history"][-1])
    )
    assert d["platform"]["status"] == "refunding" and d["next_step"] == "done"
    assert (d["can_refresh"], d["can_negotiate"], d["ticket"]["waiting_s"]) == (False, False, None)
    assert {a["blocked_reason"] for a in d["actions"]} == {
        "当前是「已同意」，只有待客服确认时才能执行"
    }


# ---------------------------------------------------------------- 平台差异（§7.1、§7.3.2）


def test_pdd_unit_offers_no_refuse():
    api = build_api(Platform.PDD)
    base = f"/v1/shops/{SHOP_C}/aftersales"
    _add_aftersale(api, SHOP_C, "as_pdd1", BUYER_PDD)
    assert _ids(_queue(api, "agent_c", base=base)) == ["as_pdd1"]
    assert _run(api, "agent_c", base=base)["succeeded"] == 1
    d = _detail(api, "as_pdd1", "agent_c", base)
    assert d["supported_actions"] == ["agree_refund", "agree_return"]
    assert _actions(d)["refuse"]["blocked_reason"] == MSG_PDD_REFUSE
    assert _actions(d)["agree_refund"]["available"] is True
    assert d["platform_note"] == PLATFORM_NOTES[Platform.PDD] and d["refuse_hint"] == MSG_PDD_REFUSE
    r = _confirm(api, "as_pdd1", "agree_refund", "agent_c", base)
    assert r.status_code == 200 and r.json()["state"] == "agreed"
    assert api.unit.action_port.platform == Platform.PDD
    assert api.unit.action_port.calls[-1]["aftersale_id"] == "as_pdd1"


def test_douyin_unit_refuse_needs_a_reason_code(douyin_api: Api):
    api = douyin_api
    base = f"/v1/shops/{SHOP_D}/aftersales"
    _add_aftersale(api, SHOP_D, "as_dy1", BUYER_DOUYIN)
    _run(api, "agent_d", base=base)
    d = _detail(api, "as_dy1", "agent_d", base)
    refuse = _actions(d)["refuse"]
    assert refuse["available"] is True and refuse["needs_reason_code"] is True  # 抖店不要求先协商
    assert d["platform_note"] == PLATFORM_NOTES[Platform.DOUYIN]
    r = _confirm(api, "as_dy1", "refuse", "agent_d", base)
    assert r.status_code == 409 and r.json()["message"] == MSG_DOUYIN_REFUSE
    r = _confirm(api, "as_dy1", "refuse", "agent_d", base, reason_code="DY_R01")
    assert r.status_code == 200 and r.json()["state"] == "refused"
    call = api.unit.action_port.calls[-1]
    assert (call["action"], call["reason_code"], call["operator"]) == (
        AfterSaleAction.REFUSE,
        "DY_R01",
        "kf_d01",
    )


def test_mvp_unit_without_write_port_only_suggests():
    """MVP 没申请写售后权限（§8.4、D6）：只给建议，按钮说明去平台后台操作。"""
    api = _api_for(build_demo_unit(Platform.TAOBAO, with_action_port=False))
    _run(api)
    d = _detail(api, "as_2001")
    assert (d["write_enabled"], d["write_scope"]) == (False, False)
    assert {a["blocked_reason"] for a in d["actions"]} == {MSG_NO_PORT}
    assert d["review"]["suggestion"] == "agree"


def test_missing_write_scope_blocks_actions(api: Api):
    auth = api.unit.repos.auth
    auth.save(auth.get(SHOP_A).model_copy(update={"scopes": sorted(MVP_SCOPES)}))
    _run(api)
    d = _detail(api, "as_2001")
    assert (d["write_enabled"], d["write_scope"]) == (True, False)
    assert {a["blocked_reason"] for a in d["actions"]} == {MSG_NO_WRITE_SCOPE}
    r = _confirm(api, "as_2001", "agree_refund")
    assert r.status_code == 403 and r.json()["error"] == "shop_not_authorized"
    assert api.unit.action_port.calls == []


# ================================================================ 统计


def test_stats_count_the_workload_and_rules(api: Api):
    s = api.get(f"{A}/stats", "agent_a").json()
    assert s["by_state"]["unreviewed"] == 8 and s["open_total"] == 8 and s["needs_action"] == 8
    assert s["review_duration"] == {
        "count": 0,
        "average_s": None,
        "median_s": None,
        "p90_s": None,
        "max_s": None,
    }
    assert s["waiting"]["count"] == 8 and s["waiting"]["max_s"] == 0.4 * 86400
    _run(api)
    s = api.get(f"{A}/stats", "agent_a").json()
    by = s["by_state"]
    assert set(by) == {"unreviewed", *(st.value for st in ReviewState)}
    assert (by["unreviewed"], by["confirm"], by["verify"], by["evidence"]) == (0, 6, 1, 1)
    assert s["needs_action"] == 6
    assert s["pending_by_rule"] == {
        "R1": 0,
        "R2": 1,
        "R3": 3,
        "R4": 1,
        "R5": 2,
        "R6": 1,
        "R7": 1,
        "R8": 1,
    }


def test_stats_average_review_duration_from_confirmations(api: Api):
    """申请 → 第一次确认执行（§2.1 售后平均审核时长）；since 只看之后确认的。"""
    _run(api)
    t0 = api.unit.clock.now()
    assert _confirm(api, "as_2001", "agree_refund").status_code == 200  # 申请 0.1 天前
    api.unit.clock.advance(hours=1)
    assert _confirm(api, "as_2009", "agree_refund").status_code == 200  # 申请 0.05 天前
    s = api.get(f"{A}/stats", "agent_a").json()
    first, second = 0.1 * 86400, 0.05 * 86400 + 3600
    assert s["decided"] == 2 and s["duration_unknown"] == 0
    d = s["review_duration"]
    assert d["count"] == 2 and d["average_s"] == pytest.approx((first + second) / 2)
    assert d["max_s"] == pytest.approx(first)
    assert s["by_state"]["agreed"] == 2 and s["open_total"] == 6
    since = (t0 + timedelta(minutes=30)).isoformat()
    s = api.get(f"{A}/stats", "agent_a", params={"since": since}).json()
    assert s["decided"] == 1 and s["review_duration"]["average_s"] == pytest.approx(second)
    assert s["by_state"]["agreed"] == 1 and s["since"] == since
    assert s["by_state"]["confirm"] == 4  # 还没结束的不受 since 影响


def test_stats_use_review_history_when_audit_events_are_missing(api: Api, monkeypatch):
    _run(api)
    _confirm(api, "as_2001", "agree_refund")
    monkeypatch.setattr(api.unit.repos.audit, "list_events", lambda *a, **k: [])
    s = api.get(f"{A}/stats", "agent_a").json()
    assert s["decided"] == 1 and s["review_duration"]["average_s"] == pytest.approx(0.1 * 86400)


def test_stats_count_unknown_durations_when_the_platform_is_down(api: Api):
    _run(api)
    _confirm(api, "as_2001", "agree_refund")
    _expire_cache(api, minutes=STALE_LIMIT_MINUTES)
    api.unit.tools.fail_on |= {"get_aftersale", "list_aftersales"}
    s = api.get(f"{A}/stats", "agent_a").json()
    assert s["decided"] == 1 and s["duration_unknown"] == 1
    assert s["review_duration"]["count"] == 0 and s["platform_unavailable"] is True


def test_stats_since_needs_a_timezone(api: Api):
    r = api.get(f"{A}/stats", "agent_a", params={"since": "2026-09-01T00:00:00"})
    assert r.status_code == 422


# ================================================================ 鉴权（§8.4、§9.2）


@pytest.mark.parametrize("caller", ["agent_a", "manager_a", "merchant_a"])
def test_desk_is_open_to_the_aftersale_roles(api: Api, caller: str):
    for path in DESK_PATHS:
        assert api.get(path, caller).status_code == 200, path
    assert api.post(f"{A}/review-pending", caller, json={"limit": 1}).status_code == 200


@pytest.mark.parametrize("caller", ["reviewer_a", "integration_a"])
def test_desk_rejects_other_roles(api: Api, caller: str):
    for path in DESK_PATHS:
        r = api.get(path, caller)
        assert r.status_code == 403 and r.json()["error"] == "forbidden_role", path
    r = api.post(f"{A}/review-pending", caller)
    assert r.status_code == 403 and r.json()["error"] == "forbidden_role"
    assert api.unit.repos.reviews.list_by_state(SHOP_A) == []


def test_desk_rejects_other_shops_without_leaking(api: Api):
    _run(api)
    for caller in ("agent_b", "manager_b", "merchant_b"):
        for path in DESK_PATHS:
            r = api.get(path, caller)
            assert r.status_code == 403 and r.json()["error"] == "tenant_isolation"
            assert "as_2001" not in r.text and "259" not in r.text
        r = api.post(f"{A}/review-pending", caller)
        assert r.status_code == 403 and r.json()["error"] == "tenant_isolation"


def _all_desk_calls(api: Api, who: str = "agent_a"):
    return [
        *(api.get(path, who) for path in DESK_PATHS),
        api.post(f"{A}/review-pending", who),
    ]


def test_suspended_shop_is_stopped(api: Api):
    repo = api.unit.repos.knowledge
    repo.upsert_shop(repo.get_shop(SHOP_A).model_copy(update={"status": "suspended"}))
    calls = len(api.unit.tools.calls)
    for r in _all_desk_calls(api):
        assert r.status_code == 403 and r.json()["error"] == "shop_inactive"
    assert len(api.unit.tools.calls) == calls


def test_expired_authorization_stops_before_any_platform_call(api: Api):
    """§8.4 授权过期时停止处理：一次平台接口都不调（缓存也不读）。"""
    _queue(api)  # 缓存里已有平台列表
    auth = api.unit.repos.auth
    auth.save(
        auth.get(SHOP_A).model_copy(update={"expires_at": api.unit.clock.now() - timedelta(1)})
    )
    calls = len(api.unit.tools.calls)
    stats = api.unit.gateway.stats(SHOP_A)
    for r in _all_desk_calls(api):
        assert r.status_code == 403 and r.json()["error"] == "shop_not_authorized"
    assert len(api.unit.tools.calls) == calls
    assert api.unit.gateway.stats(SHOP_A) == stats
    assert api.unit.repos.reviews.list_by_state(SHOP_A) == []


def test_revoked_shop_is_rejected(api: Api):
    _run(api)
    assert api.post(f"/v1/shops/{SHOP_A}/revoke", "merchant_a").status_code == 200
    for r in _all_desk_calls(api):
        assert r.status_code == 403 and r.json()["error"] == "shop_inactive"


# ================================================================ 端到端：页面的调用顺序


def test_page_flow_queue_rules_detail_confirm_stats(api: Api):
    """页面脚本的调用：登录 → 概况、列表 → 拉取并跑规则 → 打开详情 → 确认对话框（带详情里的
    平台状态版本）→ 已同意、写接口被调用一次 → 列表、概况更新。"""
    me = api.get("/v1/me", "agent_a").json()
    assert me["role"] == "agent"
    assert api.get(f"{A}/stats", "agent_a").json()["by_state"]["unreviewed"] == 8
    assert _run(api, limit=20)["succeeded"] == 8
    d = _detail(api, "as_2001")
    assert _actions(d)["agree_refund"]["available"]
    body = {"action": "agree_refund", "expected_status_version": d["platform"]["status_version"]}
    r = api.post(f"{A}/as_2001/confirm", "agent_a", json=body)
    assert r.status_code == 200 and r.json()["state"] == "agreed"
    # 重复点击（同一状态版本）：不会重复调平台
    r = api.post(f"{A}/as_2001/confirm", "agent_a", json=body)
    assert r.status_code == 409
    calls = api.unit.action_port.calls
    assert [(c["aftersale_id"], c["action"], c["operator"]) for c in calls] == [
        ("as_2001", AfterSaleAction.AGREE_REFUND, AGENT)
    ]
    assert "as_2001" not in _ids(_queue(api))
    s = api.get(f"{A}/stats", "agent_a").json()
    assert s["decided"] == 1 and s["by_state"]["agreed"] == 1
