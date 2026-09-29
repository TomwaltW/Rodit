"""模式 C 网页工具用到的接口（§8.1 模式 C、§5.3、§5.6、§5.8、§12.4）。

- /v1/me：令牌 → 身份（话术助手和商家控制台登录；具体的人都能用，含复核人员，平台对接不行）；
  /v1/shops/{shop_id}/items：选商品（只读、按店铺隔离）；
  这两个接口的 401 / 串店 403 / 角色 403 也在 test_api_auth 的全路由矩阵里
  （api_support.SHOP_ROUTES）。
- 端到端：按页面脚本的调用方式（同样的路径、请求体）走一遍
  提问 → 发送 / 编辑后发送 / 不合适 → 报表。
"""

from __future__ import annotations

import pytest
from api_support import CALLERS, TOKENS, Api, TestClient, auth

from rodit.api import StaticTokenAuthenticator, create_app
from rodit.api.schemas import EngineResultView, describe_missing
from rodit.app import AssistService
from rodit.app.demo import build_demo_unit
from rodit.core.audit import AuditAction
from rodit.core.enums import KnowledgeStatus, Mode, Platform, SuggestionStatus
from rodit.core.models import EngineResult
from rodit.testing.fixtures import AGENT, SHOP_A, SHOP_B, SHOP_D

A = f"/v1/shops/{SHOP_A}"


def _ask(api: Api, question: str, session: str = "web-e2e", who: str = "agent_a", **extra):
    body = {"session_id": session, "question": question, **extra}
    return api.post(f"{A}/standalone/ask", who, json=body)


# ================================================================ /v1/me


@pytest.mark.parametrize(
    "caller", ["agent_a", "manager_a", "merchant_a", "agent_b", "reviewer_a", "reviewer_b"]
)
def test_me_returns_the_callers_own_identity(api: Api, caller: str):
    r = api.get("/v1/me", caller)
    assert r.status_code == 200, r.text
    p = CALLERS[caller]
    assert r.json() == {
        "shop_id": p.shop_id,
        "user_id": p.user_id,
        "role": p.role,
        "platform": "taobao",
    }
    assert TOKENS[caller] not in r.text


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer not-a-known-token-0000000000"}, {"Authorization": "Basic x"}],
    ids=["missing", "unknown", "basic"],
)
def test_me_rejects_missing_or_unknown_token(api: Api, headers):
    r = api.get("/v1/me", headers=headers)
    assert r.status_code == 401 and r.json()["error"] == "unauthorized"
    assert r.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("caller", ["integration_a", "integration_b", "integration_d"])
def test_me_is_for_people_not_platform_integration(api: Api, caller: str):
    r = api.get("/v1/me", caller)
    assert r.status_code == 403 and r.json()["error"] == "forbidden_role"
    assert CALLERS[caller].shop_id not in r.text and TOKENS[caller] not in r.text


def test_reviewer_signs_in_but_only_reaches_reviewer_endpoints(api: Api):
    """复核人员能用 /v1/me 登录控制台；登录不扩大权限：模式 C 提问、商家控制台的读写照样 403。"""
    me = api.get("/v1/me", "reviewer_a").json()
    assert me["role"] == "reviewer" and me["shop_id"] == SHOP_A
    for method, path in (
        ("POST", f"{A}/standalone/ask"),
        ("GET", f"{A}/items"),
        ("GET", f"{A}/console/faqs"),
        ("GET", f"{A}/reports/adoption"),
    ):
        r = api.call(method, path, "reviewer_a", json={"session_id": "r", "question": "在吗"})
        assert r.status_code == 403 and r.json()["error"] == "forbidden_role", path
    assert api.get(f"{A}/quality/report", "reviewer_a").status_code == 200
    assert api.get(f"{A}/rollout/status", "reviewer_a").status_code == 200


def test_me_does_not_read_shop_state(api: Api):
    """店铺停用时 /v1/me 照常返回身份（页面据此登录），之后的店铺接口是 403 shop_inactive。"""
    repo = api.unit.repos.knowledge
    repo.upsert_shop(repo.get_shop(SHOP_A).model_copy(update={"status": "suspended"}))
    assert api.get("/v1/me", "agent_a").status_code == 200
    r = api.get(f"{A}/items", "agent_a")
    assert r.status_code == 403 and r.json()["error"] == "shop_inactive"


# ================================================================ 商品列表


def test_items_lists_only_the_callers_shop(api: Api):
    """两家店都有 i_555（标题不同）：各自只看到自己的。"""
    r = api.get(f"{A}/items", "agent_a")
    assert r.status_code == 200, r.text
    assert r.json() == [
        {"item_id": "i_555", "title": "法式碎花连衣裙"},
        {"item_id": "i_777", "title": "高腰阔腿牛仔裤"},
        {"item_id": "i_888", "title": "定制刺绣真丝衬衫"},
    ]
    r = api.get(f"/v1/shops/{SHOP_B}/items", "agent_b")
    assert r.json() == [{"item_id": "i_555", "title": "男士休闲裤"}]


@pytest.mark.parametrize("caller", ["agent_a", "manager_a", "merchant_a"])
def test_items_is_open_to_every_human_role(api: Api, caller: str):
    assert api.get(f"{A}/items", caller).status_code == 200


def test_items_cross_shop_and_integration_are_rejected(api: Api):
    for caller in ("agent_b", "manager_b", "merchant_b", "integration_b"):
        r = api.get(f"{A}/items", caller)
        assert r.status_code == 403 and r.json()["error"] == "tenant_isolation"
        assert "法式碎花连衣裙" not in r.text
    r = api.get(f"{A}/items", "integration_a")
    assert r.status_code == 403 and r.json()["error"] == "forbidden_role"


def test_items_defends_against_a_repository_leaking_other_shops(api: Api, monkeypatch):
    repo = api.unit.repos.knowledge
    leaked = repo.list_item_cards(SHOP_A) + repo.list_item_cards(SHOP_B)
    monkeypatch.setattr(repo, "list_item_cards", lambda shop_id: leaked)
    r = api.get(f"{A}/items", "agent_a")
    assert r.status_code == 200
    assert "男士休闲裤" not in r.text and len(r.json()) == 3


def test_items_after_revoke_is_rejected(api: Api):
    assert api.post(f"{A}/revoke", "merchant_a").status_code == 200
    r = api.get(f"{A}/items", "agent_a")
    assert r.status_code == 403 and r.json()["error"] == "shop_inactive"


# ================================================================ 缺失知识的说明（§5.7）


def test_missing_knowledge_comes_with_descriptions(api: Api):
    r = _ask(api, "这个面料是什么", session="web-miss")
    view = r.json()
    assert view["suggestions"] == [] and view["no_suggestion_reason"]
    assert view["missing_knowledge"] == ["item:?"]
    assert view["missing_knowledge_details"] == [
        {"marker": "item:?", "description": "会话里没有识别到当前商品"}
    ]
    # 选上商品再问：有依据，能回答
    r = _ask(api, "这个面料是什么", session="web-miss", item_id="i_777", request_id="r2")
    view = r.json()
    assert "棉" in view["suggestions"][0]["text"] and view["missing_knowledge_details"] == []


@pytest.mark.parametrize(
    ("marker", "description"),
    [
        ("policy:invoice", "店铺政策 · 发票（问卷第 7 题）"),
        ("item:i_888:size_advice", "商品 i_888 · 尺码建议"),
        ("item:?", "会话里没有识别到当前商品"),
        ("order:?", "会话里没有订单信息"),
        ("knowledge:size_advice", "「尺码 / 选型」相关的知识"),
        ("knowledge:not_an_intent", "knowledge:not_an_intent"),
        ("platform:taobao:apply_aftersale", "平台规则 · 淘宝申请售后"),
        ("platform:nowhere:x", "platform:nowhere:x"),
        ("something:else", "something:else"),
    ],
)
def test_describe_missing(marker: str, description: str):
    assert describe_missing(marker) == description


def test_shadow_view_has_no_missing_knowledge_details():
    result = EngineResult(
        shop_id=SHOP_A,
        conversation_id="c",
        msg_ids=["m"],
        mode=Mode.SHADOW,
        missing_knowledge=["policy:invoice"],
    )
    assert EngineResultView.from_result(result).missing_knowledge_details == []


def test_shadow_shop_shows_nothing_to_the_page(douyin_api: Api):
    r = douyin_api.post(
        f"/v1/shops/{SHOP_D}/standalone/ask",
        "agent_d",
        json={"session_id": "web-shadow", "question": "这个面料是什么", "request_id": "r1"},
    )
    view = r.json()
    assert view["shadow"] is True and view["suggestions"] == []
    assert view["missing_knowledge"] == [] and view["missing_knowledge_details"] == []


# ================================================================ 端到端：页面的调用顺序


def test_page_flow_ask_send_edit_reject_then_report(api: Api):
    """页面脚本的调用：登录 → 商品 → 提问（同一会话、每次新的 request_id）→ 三种反馈 → 报表。"""
    me = api.get("/v1/me", "agent_a").json()
    assert me["shop_id"] == SHOP_A
    items = api.get(f"{A}/items", "agent_a").json()
    assert "i_555" in {i["item_id"] for i in items}

    # 1) 提问 → 复制并标记已发送（页面发 {}）
    r = _ask(api, "能开发票吗", request_id="req-1")
    first = r.json()["suggestions"][0]
    r = api.post(f"{A}/suggestions/{first['suggestion_id']}/sent", "agent_a", json={})
    assert r.status_code == 200 and r.json()["status"] == "sent"

    # 2) 追问（带商品）：上一条已处理，不受影响；新推荐 → 编辑后发送 → 候选知识
    r = _ask(api, "165 110斤穿什么码", request_id="req-2", item_id="i_555")
    second = r.json()["suggestions"][0]
    assert "M" in second["text"]
    edited = "亲，165cm/110 斤建议选 M 码，喜欢宽松可以选 L 哦~"
    r = api.post(
        f"{A}/suggestions/{second['suggestion_id']}/edited",
        "agent_a",
        json={"edited_text": edited},
    )
    assert r.status_code == 200, r.text
    fb = r.json()
    assert fb["status"] == "edited_sent" and fb["candidate_knowledge_id"] is not None
    candidate = api.unit.repos.knowledge.get_knowledge(SHOP_A, fb["candidate_knowledge_id"])
    assert candidate.status == KnowledgeStatus.CANDIDATE and candidate.item_id == "i_555"

    # 3) 再问一条，先不处理；再追问一条 → 前一条过期，页面上的按钮对应 409
    #    （客服没回复时连着的两条买家消息会合并理解，第二次可能出多组推荐，§5.3）
    r = _ask(api, "什么时候发货", request_id="req-3")
    stale = r.json()["suggestions"][0]
    r = _ask(api, "能便宜点吗", request_id="req-4")
    fresh = r.json()["suggestions"][0]
    assert "涉及价格承诺，请确认" in fresh["risk_hints"]
    r = api.post(f"{A}/suggestions/{stale['suggestion_id']}/sent", "agent_a", json={})
    assert r.status_code == 409 and r.json()["error"] == "action_not_allowed"

    # 4) 不合适：理解错误 + 真实问题类型 → 标注集
    r = api.post(
        f"{A}/suggestions/{fresh['suggestion_id']}/rejected",
        "agent_a",
        json={"reason": "wrong_intent", "true_intent": "shipping_policy"},
    )
    assert r.status_code == 200 and r.json()["status"] == "rejected" and r.json()["labeled"]

    # 同一个 request_id 重试：204，页面显示"没有处理"
    r = _ask(api, "能便宜点吗", request_id="req-4")
    assert r.status_code == 204

    # 服务端记录：客服账号取自身份
    records = {r.id: r for r in api.unit.repos.suggestions.list(SHOP_A)}
    assert records[first["suggestion_id"]].status == SuggestionStatus.SENT
    assert records[second["suggestion_id"]].status == SuggestionStatus.EDITED_SENT
    assert records[stale["suggestion_id"]].status == SuggestionStatus.EXPIRED
    assert records[fresh["suggestion_id"]].status == SuggestionStatus.REJECTED
    assert {records[s["suggestion_id"]].agent_id for s in (first, second, fresh)} == {AGENT}
    # 发出去的文字写回了会话（追问有上下文，§5.3）
    msgs = api.unit.repos.conversations.recent_messages(SHOP_A, "standalone:web-e2e", limit=20)
    assert edited in [m.text for m in msgs]
    assert api.unit.repos.audit.list_events(SHOP_A, action=AuditAction.FEEDBACK_EDITED)

    # 5) 店长 / 商家看报表和缺失知识（页面的"店铺数据"）
    report = api.get(f"{A}/reports/adoption", "manager_a").json()
    overall = report["overall"]
    assert (overall["sent"], overall["edited_sent"], overall["rejected"]) == (1, 1, 1)
    assert overall["expired"] >= 1 and overall["adopted_groups"] == 2
    assert overall["group_adoption_rate"] == 2 / overall["groups"]
    assert overall["total"] == len(records)
    assert report["by_intent"]["invoice_gift"]["sent"] == 1
    since = "2000-01-01T00:00:00+08:00"
    r = api.get(f"{A}/reports/adoption", "merchant_a", params={"since": since})
    assert r.status_code == 200 and r.json()["overall"] == overall
    gaps = api.get(f"{A}/knowledge/gaps", "merchant_a").json()
    assert all(set(g) == {"marker", "description"} for g in gaps)
    # 客服看不了报表（页面不显示"店铺数据"）
    assert api.get(f"{A}/reports/adoption", "agent_a").status_code == 403


def test_page_flow_reject_missing_knowledge_reminds_merchant(api: Api):
    r = _ask(api, "有赠品吗", session="web-gap")
    card = r.json()["suggestions"][0]
    r = api.post(
        f"{A}/suggestions/{card['suggestion_id']}/rejected",
        "agent_a",
        json={"reason": "missing_knowledge"},
    )
    assert r.status_code == 200 and r.json()["knowledge_gap"]
    markers = {g["marker"] for g in api.get(f"{A}/knowledge/gaps", "manager_a").json()}
    assert r.json()["knowledge_gap"] in markers


def test_page_flow_questionnaire_answers_use_string_keys(api: Api):
    """页面按题号（字符串键）提交问卷，只提交填写了的题目。"""
    assert "warranty" not in api.unit.repos.knowledge.get_policies(SHOP_A)
    r = api.post(
        f"{A}/questionnaire",
        "merchant_a",
        json={"answers": {"11": "保修一年，联系客服寄回维修"}},
    )
    assert r.status_code == 200 and r.json() == {"policies": ["warranty"], "config": []}
    policy = api.unit.repos.knowledge.get_policies(SHOP_A)["warranty"]
    assert policy.value == "保修一年，联系客服寄回维修"
    # 店长能看缺失知识，但不能填问卷（页面只给商家显示问卷）
    r = api.post(f"{A}/questionnaire", "manager_a", json={"answers": {"11": "x"}})
    assert r.status_code == 403 and r.json()["error"] == "forbidden_role"


def test_other_platform_units_report_their_platform():
    service = AssistService(build_demo_unit(Platform.PDD))
    app = create_app(
        service, authenticator=StaticTokenAuthenticator({TOKENS["agent_a"]: CALLERS["agent_a"]})
    )
    r = TestClient(app).get("/v1/me", headers=auth("agent_a"))
    assert r.json()["platform"] == "pdd"
