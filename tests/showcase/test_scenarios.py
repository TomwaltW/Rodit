"""演示站预设场景的行为（方案 §11 核心流程、§5.2.1、§5.7、§6.6 规则 1–6、§11.5）：
每个场景在每个平台上都走真实的问答引擎（离线替身），页面上写的"会看到什么"要真的发生。"""

from __future__ import annotations

import pytest
from showcase_support import Showcase

from rodit.core.enums import Platform
from rodit.showcase.llm import SCRIPTED_PROMISE, SIMULATED_NOTE
from rodit.showcase.scenarios import ORDER_PLATFORMS, SCENARIOS, scenarios_for
from rodit.showcase.trace import gap_label

ALL = [(p, s.id) for p in Platform for s in scenarios_for(p)]


def _cards(view: dict) -> list[dict]:
    return view["result"]["suggestions"]


def _evidence_ids(view: dict) -> set[str]:
    return {
        e["id"] for part in view["trace"]["parts"] if part["find"] for e in part["find"]["evidence"]
    }


def test_scenario_sets_per_platform():
    ids = {p: [s.id for s in scenarios_for(p)] for p in Platform}
    assert len(SCENARIOS) == 11
    assert ids[Platform.TAOBAO] == ids[Platform.DOUYIN] == [s.id for s in SCENARIOS]
    # 拼多多只能做模式 C（§8.1）：查不到订单，带订单卡片的物流、签收场景不出现
    assert "logistics" not in ids[Platform.PDD] and "not_received" not in ids[Platform.PDD]
    assert ORDER_PLATFORMS == {Platform.TAOBAO, Platform.DOUYIN}
    for s in SCENARIOS:
        assert s.section.startswith("§") and s.expect and s.title


@pytest.mark.parametrize(("platform", "scenario"), ALL)
def test_every_scenario_runs_through_the_engine(showcase: Showcase, platform, scenario):
    view = showcase.chat(platform, scenario)
    trace = view["trace"]
    assert view["scenario"] == scenario
    assert view["result"]["mode"] == "assist"  # 抖店演示店的影子模式已切回智能辅助
    assert trace["understand"] is not None and 1 <= trace["understand"]["rule"] <= 8
    assert trace["understand"]["rule_text"]
    assert not view["result"]["degraded"]
    latency = trace["latency"]
    assert [s["key"] for s in latency["stages"]] == [
        "understand",
        "route",
        "retrieve",
        "generate",
        "qa",
    ]
    assert latency["target_ms"] == 5000 and latency["within_target"] is True
    assert view["message"]["text"]


@pytest.mark.parametrize("platform", list(Platform))
def test_size_advice_uses_the_merchant_size_advice(showcase: Showcase, platform):
    """§11.1：165 cm、110 斤 → 按商家填的尺码建议推荐 M，依据标明来源。"""
    view = showcase.chat(platform, "size")
    u = view["trace"]["understand"]
    assert u["intent"] == "size_advice" and u["rule"] == 6 and u["action"] == "answer"
    [card] = _cards(view)
    assert "M" in card["text"] and "160–165" in card["text"]
    assert [e["label"] for e in card["evidence"]] == ["商品属性 · 尺码建议"]
    assert {"item:i_555:size_advice", "item:i_555:size_chart"} <= _evidence_ids(view)
    [part] = view["trace"]["parts"]
    [check] = part["judge"]
    assert check["passed"] and check["model_checked"]
    assert {s["key"] for s in check["scores"]} >= {"grounded", "promise_risk", "platform_violation"}


@pytest.mark.parametrize(
    ("platform", "policy", "detail"),
    [
        (Platform.TAOBAO, "48 小时", "24 小时"),
        (Platform.DOUYIN, "48 小时", "24 小时"),
        (Platform.PDD, "24 小时", "12 小时"),
    ],
)
def test_ship_time_policy_overrides_the_detail_page(showcase: Showcase, platform, policy, detail):
    """§5.2.1：详情页写的发货时效和店铺政策冲突时以政策为准，详情页那句被覆盖。"""
    view = showcase.chat(platform, "ship_time")
    assert view["trace"]["understand"]["rule"] == 5
    by_intent = {c["intent"]: c for c in _cards(view)}
    assert set(by_intent) == {"product_spec", "shipping_policy"}
    assert policy in by_intent["shipping_policy"]["text"]
    assert detail not in by_intent["shipping_policy"]["text"]
    assert "棉 95%" in by_intent["product_spec"]["text"]
    overridden = [o for p in view["trace"]["parts"] if p["find"] for o in p["find"]["overridden"]]
    assert overridden and all(o["label"] == "商品详情 · 详情页" for o in overridden)
    assert any(detail in o["content"] for o in overridden)


@pytest.mark.parametrize("platform", sorted(ORDER_PLATFORMS))
def test_logistics_uses_the_template_with_as_of_time(showcase: Showcase, platform):
    """§11.2：物流问题用模板回答、写明"截至"，不调用大模型。"""
    view = showcase.chat(platform, "logistics")
    assert view["trace"]["understand"]["intent"] == "logistics_track"
    [card] = _cards(view)
    assert card["kind"] == "template" and "截至" in card["text"] and "运输中" in card["text"]
    [part] = view["trace"]["parts"]
    assert [d["kind"] for a in part["write"] for d in a["drafts"]] == ["template"]
    assert all(not c["model_checked"] for c in part["judge"])  # 模板只跑硬规则
    logistics = [e for e in part["find"]["evidence"] if e["id"] == "tool:logistics:o_1002"]
    assert logistics and logistics[0]["as_of"].startswith("截至") and logistics[0]["priority"] == 1
    assert view["trace"]["latency"]["stages"][3]["key"] == "generate"


@pytest.mark.parametrize("platform", sorted(ORDER_PLATFORMS))
def test_not_received_gives_pickup_guidance_and_risk(showcase: Showcase, platform):
    """§11.3：显示签收但没收到 → 驿站取件指引 + 风险提示（高风险类型，规则 4）。"""
    view = showcase.chat(platform, "not_received")
    u = view["trace"]["understand"]
    assert u["intent"] == "not_received" and u["rule"] == 4 and u["high_risk"]
    first = _cards(view)[0]
    assert first["kind"] == "template" and "驿站" in first["text"] and "申请售后" in first["text"]
    assert first["risk_hints"]


@pytest.mark.parametrize("platform", list(Platform))
def test_off_platform_request_gets_the_platform_script(showcase: Showcase, platform):
    """§11.4：站外交易 → §6.6 规则 1，审核过的标准话术，不调用大模型。"""
    view = showcase.chat(platform, "off_platform")
    u = view["trace"]["understand"]
    assert u["intent"] == "off_platform" and u["rule"] == 1 and u["action"] == "off_platform_script"
    [card] = _cards(view)
    assert card["kind"] == "template" and "平台规定不能通过站外联系和交易" in card["text"]
    assert "微信" not in card["text"]
    assert all(
        d["kind"] == "template"
        for p in view["trace"]["parts"]
        for a in p["write"]
        for d in a["drafts"]
    )


@pytest.mark.parametrize("platform", list(Platform))
def test_qa_blocks_the_unfounded_promise_then_passes_a_rewrite(showcase: Showcase, platform):
    """§11.5：模拟大模型先写"明天一定能到"，真实质检拦下，带原因重写后通过。"""
    view = showcase.chat(platform, "qa_block")
    trace = view["trace"]
    assert trace["simulated_note"] == SIMULATED_NOTE
    [part] = trace["parts"]
    attempts = part["write"]
    assert [a["attempt"] for a in attempts] == [1, 2]
    assert attempts[0]["drafts"][0]["text"] == SCRIPTED_PROMISE
    assert attempts[1]["retry_feedback"]  # 带着质检原因重写
    first, second = part["judge"]
    assert not first["passed"] and first["text"] == SCRIPTED_PROMISE
    rules = {h["rule"] for h in first["hard_hits"]}
    assert {"absolute_term", "time_promise"} <= rules
    assert any(s["key"] == "promise_risk" and not s["ok"] for s in first["scores"])
    assert second["passed"] and second["attempt"] == 2
    [card] = _cards(view)
    assert card["text"] == second["text"]
    assert "一定" not in card["text"] and "明天" not in card["text"]
    assert trace["qa_summary"]["blocked"] == 1
    if platform in ORDER_PLATFORMS:  # 有订单卡片：只说物流事实（§11.5 第 5 步）
        assert "运输中" in card["text"] and "截至" in card["text"]
        assert [e["id"] for e in card["evidence"]] == ["tool:logistics:o_1002"]
    else:  # 模式 C 查不到订单：说发货时效
        assert "24 小时" in card["text"]


@pytest.mark.parametrize("platform", list(Platform))
def test_emotional_buyer_gets_soothing_first_and_an_alert(showcase: Showcase, platform):
    """§5.7、§6.6 规则 3：情绪激烈 → 先推荐安抚话术并标红提醒。"""
    view = showcase.chat(platform, "emotion")
    u = view["trace"]["understand"]
    assert u["rule"] == 3 and u["action"] == "soothe" and u["emotion"] >= 3
    assert "买家情绪激动，建议先安抚" in view["result"]["alerts"]
    cards = _cards(view)
    assert cards[0]["kind"] == "template" and "抱歉" in cards[0]["text"]
    assert len(cards) == 2  # 安抚话术之后照常给回答
    if platform in ORDER_PLATFORMS:  # 物流停滞超过 48 小时：提醒客服向快递催查（§7.2）
        assert any("催查" in h for h in cards[1]["risk_hints"])


@pytest.mark.parametrize("platform", list(Platform))
def test_complaint_is_left_to_a_human(showcase: Showcase, platform):
    """§5.7：投诉不推荐，提醒客服优先处理。"""
    view = showcase.chat(platform, "complaint")
    u = view["trace"]["understand"]
    assert u["intent"] == "complaint" and u["rule"] == 2 and u["action"] == "human_priority"
    assert _cards(view) == []
    assert view["result"]["alerts"] and view["result"]["no_suggestion_reason"]
    assert view["trace"]["fallback"]["human_only"] is True
    assert view["trace"]["parts"] == []  # 不检索、不生成


@pytest.mark.parametrize("platform", list(Platform))
def test_two_questions_are_answered_separately(showcase: Showcase, platform):
    """§6.6 规则 5：一条消息两个问题 → 按问题分条推荐，各自有依据。"""
    view = showcase.chat(platform, "multi")
    u = view["trace"]["understand"]
    assert u["rule"] == 5 and u["action"] == "split_multi"
    assert {p["intent"] for p in u["parts"]} == {"product_spec", "invoice_gift"}
    by_intent = {c["intent"]: c for c in _cards(view)}
    assert set(by_intent) == {"product_spec", "invoice_gift"}
    assert "透" in by_intent["product_spec"]["text"]
    assert "发票" in by_intent["invoice_gift"]["text"]
    assert len(view["trace"]["parts"]) == 2


@pytest.mark.parametrize("platform", list(Platform))
def test_invoice_question_uses_the_faq_answer_directly(showcase: Showcase, platform):
    """§5.4：FAQ 命中就直接用商家审核过的标准回答，不调用大模型。"""
    view = showcase.chat(platform, "invoice")
    [card] = _cards(view)
    assert card["kind"] == "faq" and card["kind_label"] == "FAQ"
    assert "电子普通发票" in card["text"]
    assert card["evidence"][0]["label"].startswith("FAQ · ")


@pytest.mark.parametrize("platform", list(Platform))
def test_price_question_is_high_risk_with_a_hint(showcase: Showcase, platform):
    """§5.1、§6.6 规则 4：价格类推荐 + 风险提示，回答只说活动配置里有的优惠。"""
    view = showcase.chat(platform, "price")
    u = view["trace"]["understand"]
    assert u["intent"] == "price_promo" and u["rule"] == 4 and u["action"] == "high_risk_answer"
    [card] = _cards(view)
    assert "涉及价格承诺，请确认" in card["risk_hints"]
    assert "满 199 减 20" in card["text"]


def test_typed_buyer_text_goes_to_the_same_conversation(showcase: Showcase):
    """演示者在输入框里以买家身份打字：同一个会话的多轮对话，之前的推荐过期（§5.3）。"""
    first = showcase.chat("taobao", "size")
    second = showcase.chat("taobao", "size", "那 170 120斤呢")
    assert second["message"]["text"] == "那 170 120斤呢"
    assert second["trace"]["understand"]["intent"] == "size_advice"
    assert "L" in _cards(second)[0]["text"]
    scenarios = showcase.get("/showcase/api/scenarios", platform="taobao").json()["scenarios"]
    size = next(s for s in scenarios if s["id"] == "size")
    assert [b["text"] for b in size["bubbles"]] == ["165 110斤穿什么码", "那 170 120斤呢"]
    assert size["latest"]["message"]["text"] == "那 170 120斤呢"
    # 上一条的推荐已经过期，不能再发送
    old = _cards(first)[0]["suggestion_id"]
    r = showcase.feedback("taobao", old, "sent")
    assert r.status_code == 409


def test_buyer_text_is_minimized_before_it_is_shown_back(showcase: Showcase):
    """买家打出来的手机号在会话气泡里也打码（§5.3；合成号码）。"""
    view = showcase.chat("douyin", "size", "我电话 13812345678，165 110斤穿什么码")
    assert "13812345678" not in view["message"]["text"]
    assert "[手机号]" in view["message"]["text"]


@pytest.mark.parametrize(
    ("marker", "label"),
    [
        ("order:?", "会话里没有订单（模式 C 话术助手查不到订单）"),
        ("intent:return_policy", "退换政策类问题的知识"),
        ("knowledge:invoice_gift", "发票 / 赠品 / 保障类问题的知识"),
        ("item:i_555:product_spec", "商品 i_555 · 商品参数类问题的知识"),
        ("item:i_777:size_advice", "商品 i_777 · 尺码建议"),  # 商品字段（引擎记的缺失）
        ("policy:warranty", "店铺政策 · 保修（问卷第 11 题）"),
    ],
)
def test_gap_labels_are_plain_chinese(marker, label):
    assert gap_label(marker) == label
