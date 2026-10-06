"""演示站的接口：每个接口的正常结果、输入校验（422）、不认识的平台（404）、状态冲突（409）、
请求体大小（413）、响应头，以及任何响应里都没有买家标识（buyer_ref）和昵称（§5.3、§8.3）。"""

from __future__ import annotations

import re
import threading

import pytest
from showcase_support import BUYER_REF, NOW, RAW_BUYER, Showcase, build_showcase, json_text

from rodit.core.config import EngineConfig
from rodit.core.enums import AfterSaleStatus, Intent, Platform, SuggestionStatus
from rodit.core.models import SuggestionRecord
from rodit.showcase.app import API_PREFIX
from rodit.showcase.dashboard import violations
from rodit.showcase.schemas import MAX_BODY_BYTES, MAX_TEXT_CHARS

API = API_PREFIX
PII_SAMPLE = "我的手机号 13812345678，地址杭州市西湖区文三路 100 号 3 幢 2 单元 502，订单 o_1002"
"""合成的买家消息（手机号、地址都是编的）。"""


# ================================================================ 平台、输入校验


@pytest.mark.parametrize("path", ["overview", "scenarios", "aftersales", "dashboard"])
def test_unknown_platform_is_404(showcase: Showcase, path: str):
    r = showcase.get(f"{API}/{path}", platform="jd")
    assert r.status_code == 404
    assert r.json() == {"error": "not_found", "message": "没有这个平台：jd"}


@pytest.mark.parametrize("path", ["overview", "scenarios", "aftersales", "dashboard"])
def test_platform_query_is_validated(showcase: Showcase, path: str):
    assert showcase.get(f"{API}/{path}").status_code == 422  # 缺平台
    r = showcase.get(f"{API}/{path}", platform="<script>")
    assert r.status_code == 422 and r.json()["error"] == "invalid"


@pytest.mark.parametrize(
    "body",
    [
        {"platform": "taobao", "scenario": "size", "text": "x" * (MAX_TEXT_CHARS + 1)},
        {"platform": "taobao", "scenario": "size", "text": "   "},
        {"platform": "taobao", "scenario": "Size!"},
        {"platform": "taobao", "scenario": "size", "extra": 1},
        {"platform": "taobao"},
        {"scenario": "size"},
        {"platform": 1, "scenario": "size"},
    ],
)
def test_chat_input_is_validated(showcase: Showcase, body: dict):
    r = showcase.post(f"{API}/chat", body)
    assert r.status_code == 422 and r.json()["error"] == "invalid"
    assert r.json()["problems"]


def test_chat_unknown_platform_or_scenario_is_404(showcase: Showcase):
    r = showcase.post(f"{API}/chat", {"platform": "jd", "scenario": "size"})
    assert r.status_code == 404
    r = showcase.post(f"{API}/chat", {"platform": "taobao", "scenario": "nope"})
    assert r.status_code == 404 and "nope" in r.json()["message"]
    # 拼多多是模式 C，没有带订单卡片的物流场景
    r = showcase.post(f"{API}/chat", {"platform": "pdd", "scenario": "logistics"})
    assert r.status_code == 404


def test_text_at_the_limit_is_accepted(showcase: Showcase):
    view = showcase.chat("taobao", "size", "码" * MAX_TEXT_CHARS)
    assert view["message"]["text"]


def test_oversized_body_is_413(showcase: Showcase):
    body = '{"text": "' + "x" * (MAX_BODY_BYTES + 10) + '"}'
    r = showcase.client.post(
        f"{API}/mask", content=body, headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 413 and r.json()["error"] == "too_large"


def test_chunked_body_without_length_is_refused(showcase: Showcase):
    """分块上传量不出大小：要求带 Content-Length（411），不读进内存。"""

    def chunks():
        yield b'{"text": "'
        yield b"x" * 10
        yield b'"}'

    r = showcase.client.post(
        f"{API}/mask", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 411 and r.json()["error"] == "length_required"
    r = showcase.client.post(
        f"{API}/mask", content=b'{"text": "x"}', headers={"Content-Length": "abc"}
    )
    assert r.status_code == 400


def test_unknown_path_and_method_use_the_same_error_shape(showcase: Showcase):
    r = showcase.get(f"{API}/nope")
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    r = showcase.client.delete(f"{API}/reset")
    assert r.status_code == 405 and r.json()["error"] == "method_not_allowed"


def test_api_responses_carry_security_headers(showcase: Showcase):
    r = showcase.get(f"{API}/scenarios", platform="taobao")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "set-cookie" not in r.headers


def test_no_api_docs_are_exposed(showcase: Showcase):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert showcase.get(path).status_code == 404


# ================================================================ 工作台


def test_scenarios_describe_the_cards_and_access_mode(showcase: Showcase):
    taobao = showcase.get(f"{API}/scenarios", platform="taobao").json()
    assert taobao["access"] == "B" and taobao["shop_name"] == "小雅女装旗舰店"
    by_id = {s["id"]: s for s in taobao["scenarios"]}
    assert by_id["size"]["card"]["kind"] == "item" and by_id["size"]["card"]["id"] == "i_555"
    assert by_id["size"]["card"]["price"] == "259.00"
    logistics = by_id["logistics"]["card"]
    assert logistics["kind"] == "order" and logistics["status"] == "已发货"
    assert logistics["logistics"] == "运输中"
    assert by_id["off_platform"]["card"] is None
    assert by_id["qa_block"]["note"] == "演示：模拟大模型先写出没有依据的承诺"
    assert all(s["bubbles"] == [] and s["latest"] is None for s in taobao["scenarios"])
    pdd = showcase.get(f"{API}/scenarios", platform="pdd").json()
    assert pdd["access"] == "C" and "模式 C" in pdd["access_label"]
    pdd_ids = {s["id"]: s for s in pdd["scenarios"]}
    assert pdd_ids["size"]["card"]["price"] is None  # 模式 C 没有平台实时数据
    assert "模式 C" in pdd_ids["qa_block"]["expect"] and "模式 C" in pdd_ids["emotion"]["expect"]
    assert "模式 C" not in by_id["qa_block"]["expect"]
    douyin = showcase.get(f"{API}/scenarios", platform="douyin").json()
    assert douyin["shop_name"] == "抖抖女装"
    assert next(s for s in douyin["scenarios"] if s["id"] == "logistics")["card"]["courier"] == (
        "韵达快递"
    )


def test_chat_result_is_kept_for_the_page(showcase: Showcase):
    view = showcase.chat("taobao", "invoice")
    scenarios = showcase.get(f"{API}/scenarios", platform="taobao").json()["scenarios"]
    invoice = next(s for s in scenarios if s["id"] == "invoice")
    assert invoice["bubbles"] == [view["message"]]
    assert invoice["latest"]["result"] == view["result"]
    # 别的平台互不影响（§9.1 每个平台一个单元）
    douyin = showcase.get(f"{API}/scenarios", platform="douyin").json()["scenarios"]
    assert all(s["latest"] is None for s in douyin)


def test_feedback_sent_adds_an_agent_bubble_and_counts_as_adopted(showcase: Showcase):
    view = showcase.chat("taobao", "size")
    [card] = view["result"]["suggestions"]
    r = showcase.feedback("taobao", card["suggestion_id"], "sent")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "sent" and out["action_label"] == "原样发送"
    assert out["bubble"]["role"] == "agent" and out["bubble"]["text"] == card["text"]
    assert out["labeled"] is True  # 原样采纳是判断层的正样本（§5.8）
    demo = showcase.hub.get(Platform.TAOBAO)
    record = demo.unit.repos.suggestions.get(demo.shop_id, card["suggestion_id"])
    assert record is not None and record.status == SuggestionStatus.SENT
    assert record.agent_id == "kf_02"
    scenarios = showcase.get(f"{API}/scenarios", platform="taobao").json()["scenarios"]
    size = next(s for s in scenarios if s["id"] == "size")
    assert [b["role"] for b in size["bubbles"]] == ["buyer", "agent"]
    assert size["latest"]["result"]["suggestions"][0]["status"] == "sent"
    # 已经处理过的推荐不能再操作
    again = showcase.feedback("taobao", card["suggestion_id"], "rejected", reason="other")
    assert again.status_code == 409 and again.json()["error"] == "conflict"


def test_feedback_edited_creates_candidate_knowledge(showcase: Showcase):
    """§5.8：编辑后发送 → 记录修改前后 → 候选知识（等商家审核）。"""
    view = showcase.chat("douyin", "size")
    [card] = view["result"]["suggestions"]
    edited = card["text"] + "，有其他问题随时找我哦~"
    out = showcase.feedback("douyin", card["suggestion_id"], "edited", text=edited).json()
    assert out["status"] == "edited_sent"
    assert out["bubble"]["text"] == edited and out["bubble"]["edited"] is True
    assert out["candidate"]["answer"] == edited
    assert out["candidate"]["question"] == "165 110斤穿什么码"
    demo = showcase.hub.get(Platform.DOUYIN)
    pending = demo.service.pending_candidates(demo.shop_id)
    assert any(k.answer == edited for k in pending)


def test_feedback_edited_on_realtime_evidence_skips_candidate(showcase: Showcase):
    view = showcase.chat("taobao", "logistics")
    [card] = view["result"]["suggestions"]
    out = showcase.feedback("taobao", card["suggestion_id"], "edited", text="亲，在路上了").json()
    assert out["candidate"] is None
    assert "实时数据" in out["candidate_skipped"]


def test_feedback_rejected_records_reason_and_gap(showcase: Showcase):
    view = showcase.chat("pdd", "invoice")
    [card] = view["result"]["suggestions"]
    out = showcase.feedback(
        "pdd", card["suggestion_id"], "rejected", reason="missing_knowledge"
    ).json()
    assert out["status"] == "rejected" and out["reason_label"] == "知识缺失"
    assert out["bubble"] is None and out["gap"]  # 提醒商家补知识
    demo = showcase.hub.get(Platform.PDD)
    record = demo.unit.repos.suggestions.get(demo.shop_id, card["suggestion_id"])
    assert record is not None and record.reject_reason is not None


@pytest.mark.parametrize(
    ("action", "extra"),
    [("edited", {}), ("rejected", {}), ("edited", {"text": " "}), ("rejected", {"reason": "x"})],
)
def test_feedback_input_is_validated(showcase: Showcase, action: str, extra: dict):
    view = showcase.chat("taobao", "invoice")
    [card] = view["result"]["suggestions"]
    r = showcase.feedback("taobao", card["suggestion_id"], action, **extra)
    assert r.status_code == 422, r.text


def test_feedback_unknown_suggestion_is_404(showcase: Showcase):
    assert showcase.feedback("taobao", 999999, "sent").status_code == 404
    r = showcase.post(
        f"{API}/feedback", {"platform": "taobao", "suggestion_id": 0, "action": "sent"}
    )
    assert r.status_code == 422


def test_feedback_changes_the_dashboard(showcase: Showcase):
    before = showcase.dashboard("taobao")
    assert before["kpis"]["groups"] == 0 and before["kpis"]["adoption_rate"] is None
    view = showcase.chat("taobao", "invoice")
    [card] = view["result"]["suggestions"]
    showcase.feedback("taobao", card["suggestion_id"], "sent")
    after = showcase.dashboard("taobao")
    assert after["kpis"]["groups"] == 1 and after["kpis"]["adoption_rate"] == 1.0
    assert after["kpis"]["messages"] == before["kpis"]["messages"] + 1
    assert after["weekly"][-1]["groups"] == 1
    assert {i["intent"]: i["count"] for i in after["intents"]}["invoice_gift"] == 1
    view = showcase.chat("taobao", "price")
    [card] = view["result"]["suggestions"]
    showcase.feedback("taobao", card["suggestion_id"], "rejected", reason="bad_wording")
    third = showcase.dashboard("taobao")
    assert third["kpis"]["adoption_rate"] == 0.5
    reasons = {r["reason"]: r["count"] for r in third["reject_reasons"]}
    assert reasons["bad_wording"] == 1


def test_qa_interception_shows_on_the_dashboard(showcase: Showcase):
    showcase.chat("taobao", "qa_block")
    qa = showcase.dashboard("taobao")["qa"]
    assert qa["messages_intercepted"] == 1 and qa["drafts_failed"] == 1
    rules = {h["rule"]: h for h in qa["hard_hits"]}
    assert rules["absolute_term"]["blocking"] and rules["time_promise"]["count"] == 1


# ================================================================ 售后审核（§7.3.2）


def test_aftersales_list_with_rules_and_suggestions(showcase: Showcase):
    data = showcase.aftersales("taobao")
    assert data["can_refuse"] is True and data["operator"] == "kf_manager"
    assert data["has_write_port"] is True and data["refuse_reasons"]
    items = {i["aftersale_id"]: i for i in data["items"]}
    # 最近 7 天的售后单；30 天内的历史售后（R7 用）不列出
    assert "as_3001" not in items and {"as_2001", "as_2003", "as_2005", "as_3004"} <= set(items)
    r2 = items["as_2001"]
    assert [r["id"] for r in r2["review"]["rules"]] == ["R2"]
    assert r2["review"]["suggestion"] == "agree" and r2["review"]["state"] == "confirm"
    assert [a["action"] for a in r2["actions"]] == ["agree_refund", "refuse"]
    r6 = items["as_2005"]
    assert r6["review"]["requires_manager"] is True
    assert {r["id"] for r in r6["review"]["rules"]} == {"R3", "R6"}
    assert items["as_3004"]["review"]["risk_flags"]  # R7
    r4 = items["as_2003"]
    assert r4["review"]["state"] == "verify" and r4["actions"] == []
    assert r4["buyer_note"] == "显示签收了但我根本没收到"
    assert items["as_2006"]["review"]["state"] == "platform"  # R1
    # 待确认的排在前面
    states = [i["review"]["state"] for i in data["items"]]
    assert states.index("confirm") < states.index("platform")


def test_pdd_has_no_refuse_action(showcase: Showcase):
    """§7.1：拼多多没有查到拒绝售后接口，只能同意。"""
    data = showcase.aftersales("pdd")
    assert data["can_refuse"] is False
    actions = {a["action"] for i in data["items"] for a in i["actions"]}
    assert actions and "refuse" not in actions
    item = next(i for i in data["items"] if i["aftersale_id"] == "as_2002")
    r = showcase.post(
        f"{API}/aftersales/as_2002/confirm",
        {"platform": "pdd", "action": "refuse", "expected_status_version": item["status_version"]},
    )
    assert r.status_code == 409 and "拼多多" in r.json()["message"]


def test_confirm_rereads_status_and_is_idempotent(showcase: Showcase):
    item = showcase.aftersale("taobao", "as_2001")
    body = {
        "platform": "taobao",
        "action": "agree_refund",
        "expected_status_version": item["status_version"],
    }
    r = showcase.post(f"{API}/aftersales/as_2001/confirm", body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["operator"] == "kf_manager" and out["duplicate"] is False
    assert out["idempotency_key"] == f"as_2001:agree_refund:{item['status_version']}"
    assert out["reread"]["before"]["status"] == "pending_review"
    assert out["reread"]["after"]["status"] == "refunding"
    assert out["platform_result"]["ok"] is True
    assert out["item"]["review"]["state"] == "agreed"
    assert out["item"]["review"]["confirmed_by"] == "kf_manager"
    demo = showcase.hub.get(Platform.TAOBAO)
    port = demo.unit.action_port
    assert len(port.calls) == 1
    # 重复点击（同一张单、同一个操作、同一个状态版本）：不再调平台
    again = showcase.post(f"{API}/aftersales/as_2001/confirm", body).json()
    assert again["duplicate"] is True and again["idempotency_key"] == out["idempotency_key"]
    assert again["reread"]["after"] is None
    assert len(port.calls) == 1
    info = demo.unit.tools.get_aftersale(demo.shop_id, "as_2001")
    assert info.status == AfterSaleStatus.REFUNDING
    events = demo.unit.repos.audit.list_events(demo.shop_id, action="aftersale.confirm")
    assert [e.actor for e in events] == ["kf_manager"]


def test_confirm_with_a_stale_version_is_refused(showcase: Showcase):
    showcase.aftersale("taobao", "as_2002")
    r = showcase.post(
        f"{API}/aftersales/as_2002/confirm",
        {"platform": "taobao", "action": "agree_return", "expected_status_version": "1"},
    )
    assert r.status_code == 409 and "刷新" in r.json()["message"]
    demo = showcase.hub.get(Platform.TAOBAO)
    assert demo.unit.action_port.calls == []


def test_r6_amount_over_limit_is_confirmed_by_the_manager(showcase: Showcase):
    item = showcase.aftersale("taobao", "as_2005")
    out = showcase.post(
        f"{API}/aftersales/as_2005/confirm",
        {
            "platform": "taobao",
            "action": "agree_return",
            "expected_status_version": item["status_version"],
        },
    ).json()
    assert out["platform_result"]["ok"] is True
    assert out["item"]["review"]["state"] == "wait_return"
    assert out["reread"]["after"]["status"] == "wait_buyer_return"


def test_confirm_in_a_non_confirm_state_is_conflict(showcase: Showcase):
    item = showcase.aftersale("taobao", "as_2003")  # R4 签收核查中
    r = showcase.post(
        f"{API}/aftersales/as_2003/confirm",
        {
            "platform": "taobao",
            "action": "agree_refund",
            "expected_status_version": item["status_version"],
        },
    )
    assert r.status_code == 409


def test_taobao_refuse_needs_negotiation_first(showcase: Showcase):
    item = showcase.aftersale("taobao", "as_2007")
    body = {
        "platform": "taobao",
        "action": "refuse",
        "expected_status_version": item["status_version"],
    }
    r = showcase.post(f"{API}/aftersales/as_2007/confirm", body)
    assert r.status_code == 409 and "协商" in r.json()["message"]
    out = showcase.post(f"{API}/aftersales/as_2007/negotiate", {"platform": "taobao"}).json()
    assert out["item"]["review"]["state"] == "negotiate"
    assert out["item"]["can_finish_negotiation"] is True and out["item"]["actions"] == []
    out = showcase.post(
        f"{API}/aftersales/as_2007/negotiate", {"platform": "taobao", "done": True}
    ).json()
    assert out["item"]["review"]["state"] == "confirm"
    assert out["item"]["review"]["negotiated"] is True
    assert [h["to_state"] for h in out["item"]["review"]["history"]][-2:] == [
        "negotiate",
        "confirm",
    ]
    done = showcase.post(f"{API}/aftersales/as_2007/confirm", body).json()
    assert done["platform_result"]["ok"] is True
    assert done["item"]["review"]["state"] == "refused"


def test_douyin_refuse_needs_a_reason_code(showcase: Showcase):
    item = showcase.aftersale("douyin", "as_2007")
    body = {
        "platform": "douyin",
        "action": "refuse",
        "expected_status_version": item["status_version"],
    }
    r = showcase.post(f"{API}/aftersales/as_2007/confirm", body)
    assert r.status_code == 409 and "原因码" in r.json()["message"]
    out = showcase.post(
        f"{API}/aftersales/as_2007/confirm", {**body, "reason_code": "goods_used"}
    ).json()
    assert out["platform_result"]["ok"] is True
    demo = showcase.hub.get(Platform.DOUYIN)
    assert demo.unit.action_port.calls[-1]["reason_code"] == "goods_used"


_V1 = {"expected_status_version": "1"}


@pytest.mark.parametrize(
    ("path", "body", "status"),
    [
        ("/aftersales/as_9999/confirm", {"action": "agree_refund", **_V1}, 404),
        ("/aftersales/as_9001/confirm", {"action": "agree_refund", **_V1}, 404),  # 店铺 B 的单
        ("/aftersales/as_2001/confirm", {"action": "approve", **_V1}, 422),
        ("/aftersales/as_2001/confirm", {"action": "agree_refund"}, 422),
        (
            "/aftersales/as_2001/confirm",
            {"action": "agree_refund", "expected_status_version": "1 OR 1"},
            422,
        ),
        ("/aftersales/as_2001/confirm", {"action": "refuse", **_V1, "reason_code": "x" * 40}, 422),
        ("/aftersales/as%202001/confirm", {"action": "agree_refund", **_V1}, 422),
        ("/aftersales/as_9999/negotiate", {}, 404),
    ],
)
def test_aftersale_input_is_validated(showcase: Showcase, path: str, body: dict, status: int):
    showcase.aftersales("taobao")
    r = showcase.post(f"{API}{path}", {"platform": "taobao", **body})
    assert r.status_code == status, r.text


def test_unknown_reason_code_is_rejected(showcase: Showcase):
    item = showcase.aftersale("douyin", "as_2007")
    r = showcase.post(
        f"{API}/aftersales/as_2007/confirm",
        {
            "platform": "douyin",
            "action": "refuse",
            "expected_status_version": item["status_version"],
            "reason_code": "made_up",
        },
    )
    assert r.status_code == 422


def test_other_shop_in_the_same_unit_is_unreachable(showcase: Showcase):
    """淘宝单元里还有店铺 B（s_2002，§9.2 串店测试用）：演示站只用小雅女装旗舰店，B 店的数据
    （售后单 as_9001、男装商品）从任何接口都拿不到。"""
    showcase.aftersales("taobao")
    for scenario in ("size", "invoice", "multi"):
        showcase.chat("taobao", scenario)
    showcase.dashboard("taobao")
    text = showcase.all_text()
    assert "as_9001" not in text and "木棉男装" not in text and "男士休闲裤" not in text
    assert not re.search(r"\bs_2002\b", text)  # 店铺 ID（as_2002 是 A 店的售后单）


# ================================================================ 看板、概览、打码、重置


def test_dashboard_shape(showcase: Showcase):
    d = showcase.dashboard("douyin")
    assert d["platform"] == "douyin" and d["synthetic"] == "演示数据（合成）"
    assert set(d) >= {"kpis", "weekly", "by_intent", "intents", "reject_reasons", "qa", "latency"}
    assert [w["week"] for w in d["weekly"]] == [1, 2, 3, 4, 5]
    assert [r["reason"] for r in d["reject_reasons"]] == [
        "missing_knowledge",
        "wrong_intent",
        "bad_wording",
        "wrong_fact",
        "other",
    ]
    metrics = d["platform_metrics"]["metrics"]
    assert [m["metric"] for m in metrics] == [
        "douyin_avg_response_s",
        "douyin_dissatisfaction_rate",
        "douyin_help_rate",
        "douyin_aftersale_review_h",
    ]
    assert all(m["improved"] for m in metrics)
    response = metrics[0]
    assert response["before"] > response["after"] and response["higher_is_better"] is False


def test_overview_targets(showcase: Showcase):
    r = showcase.get(f"{API}/overview", platform="pdd")
    assert r.status_code == 200
    data = r.json()
    keys = [t["key"] for t in data["targets"]]
    assert keys == ["adoption", "latency", "violations", "platform_metric", "aftersale", "accuracy"]
    by_key = {t["key"]: t for t in data["targets"]}
    assert by_key["violations"]["value"] == 0 and by_key["violations"]["ok"] is True
    assert "3 分钟人工回复率" in by_key["platform_metric"]["value"]
    assert by_key["accuracy"]["ok"] is None  # 演示里不做人工抽检，不冒充达标
    assert data["build_ms"] >= 0


def test_mask_demo(showcase: Showcase):
    r = showcase.post(
        f"{API}/mask",
        {"text": PII_SAMPLE},
    )
    assert r.status_code == 200
    out = r.json()
    assert "13812345678" not in out["masked"] and "文三路" not in out["masked"]
    assert "o_1002" in out["masked"]  # 平台订单号保留（回答要用）
    assert {f["token"] for f in out["found"]} == {"[手机号]", "[地址]"}
    assert out["changed"] is True
    plain = showcase.post(f"{API}/mask", {"text": "这个裙子能寄到新疆吗"}).json()
    assert plain == {"masked": "这个裙子能寄到新疆吗", "found": [], "changed": False}
    assert showcase.post(f"{API}/mask", {"text": "x" * 501}).status_code == 422
    assert showcase.post(f"{API}/mask", {"text": ""}).status_code == 422


def test_reset_rebuilds_the_platform(showcase: Showcase):
    view = showcase.chat("taobao", "size")
    [card] = view["result"]["suggestions"]
    showcase.feedback("taobao", card["suggestion_id"], "sent")
    showcase.chat("douyin", "size")
    old = showcase.hub.get(Platform.TAOBAO)
    r = showcase.post(f"{API}/reset", {"platform": "taobao"})
    assert r.status_code == 200 and r.json()["platform"] == "taobao"
    assert showcase.hub.get(Platform.TAOBAO) is not old
    scenarios = showcase.get(f"{API}/scenarios", platform="taobao").json()["scenarios"]
    assert all(s["bubbles"] == [] for s in scenarios)
    assert showcase.dashboard("taobao")["kpis"]["groups"] == 0
    # 只重置这个平台
    douyin = showcase.get(f"{API}/scenarios", platform="douyin").json()["scenarios"]
    assert next(s for s in douyin if s["id"] == "size")["bubbles"]
    assert showcase.post(f"{API}/reset", {"platform": "jd"}).status_code == 404


def test_platforms_are_built_lazily_and_once(showcase: Showcase):
    hub = showcase.hub
    assert hub.built() == []
    showcase.get(f"{API}/scenarios", platform="douyin")
    assert hub.built() == [Platform.DOUYIN]
    results: list[object] = []
    threads = [
        threading.Thread(target=lambda: results.append(hub.get(Platform.PDD))) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(r) for r in results}) == 1


def test_no_buyer_identifiers_in_any_response():
    """buyer_ref（平台买家 ID 的哈希）、平台买家 ID、昵称都不出现在任何响应里（§5.3、§8.3）。"""
    sc = build_showcase()
    for platform in Platform:
        p = str(platform)
        sc.get(f"{API}/scenarios", platform=p)
        scenarios = sc.get(f"{API}/scenarios", platform=p).json()["scenarios"]
        for s in scenarios:
            view = sc.chat(p, s["id"])
            for card in view["result"]["suggestions"][:1]:
                sc.feedback(p, card["suggestion_id"], "sent")
        data = sc.aftersales(p)
        for item in data["items"]:
            if item["actions"]:
                action = item["actions"][0]["action"]
                body = {
                    "platform": p,
                    "action": action,
                    "expected_status_version": item["status_version"],
                }
                if action == "refuse":
                    body["reason_code"] = "goods_used"
                sc.post(f"{API}/aftersales/{item['aftersale_id']}/confirm", body)
        sc.dashboard(p)
        sc.get(f"{API}/overview", platform=p)
    text = sc.all_text()
    assert not BUYER_REF.search(text)
    assert not RAW_BUYER.search(text)
    assert "合成昵称" not in text and "buyer_ref" not in text and "nickname" not in text
    assert json_text(sc.hub.built())  # 三个平台都组装过


def test_violations_count_what_the_agent_added_when_editing(showcase: Showcase):
    """§1.3 平台话术违规：原样发送的都过了质检；编辑后发送时客服新加进去的站外引流算违规。"""
    view = showcase.chat("taobao", "invoice")
    [card] = view["result"]["suggestions"]
    assert showcase.dashboard("taobao")["kpis"]["violations"] == 0
    showcase.feedback(
        "taobao", card["suggestion_id"], "edited", text=card["text"] + " 加我微信 abc12345 细聊"
    )
    assert showcase.dashboard("taobao")["kpis"]["violations"] == 1
    overview = showcase.get(f"{API}/overview", platform="taobao").json()
    target = next(t for t in overview["targets"] if t["key"] == "violations")
    assert target["value"] == 1 and target["ok"] is False


def test_composition_label_is_not_a_violation():
    """推荐原文里如实的成分标注（"桑蚕丝 100%"，质检有依据时放行）不算；
    编辑时加的"最"、店铺禁用词算。"""
    qa = EngineConfig().qa_gate
    base = {
        "shop_id": "s_1001",
        "conversation_id": "c",
        "intent": Intent.PRODUCT_SPEC,
        "created_at": NOW,
    }
    silk = "亲，面料是桑蚕丝 100%"
    edited = SuggestionStatus.EDITED_SENT
    records = [
        SuggestionRecord(text=silk, status=SuggestionStatus.SENT, **base),
        SuggestionRecord(text=silk, edited_text=silk + "，手感很好", status=edited, **base),
        SuggestionRecord(text=silk, edited_text=silk + "，全网最好", status=edited, **base),
        SuggestionRecord(text=silk, edited_text=silk + "，假一赔十", status=edited, **base),
    ]
    assert violations(records, qa) == 1  # 只有"最"是编辑时加进去的
    assert violations(records, qa, banned=["假一赔十"]) == 2  # 店铺禁用词
