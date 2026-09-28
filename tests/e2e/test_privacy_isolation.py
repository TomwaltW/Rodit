"""个人信息最小化（§5.3、§12.3）与租户隔离（§9.2），端到端。

- 买家消息里的手机号、收件地址、平台昵称、原始买家 ID 不出现在任何大模型提示词（路由、质检、生成）、
  审计 detail、decision_log、推荐记录的 question_text、缺失知识和标注样本里；
- 同一个平台单元里的两家店（同一个 item_id、同一个买家）互相看不到对方的知识、订单、推荐和审核。
"""

from __future__ import annotations

import pytest
from e2e_support import (
    Chat,
    demo_service,
    llm_answer,
    production_like_llm,
    stored_texts,
)

from rodit.app.offline import EvidenceEchoLLM
from rodit.core.enums import Intent, RejectReason, RouteAction
from rodit.core.pii import ADDRESS_MASK, PHONE_MASK, mask_pii
from rodit.testing.fixtures import AGENT, RAW_BUYERS, SHOP_A, SHOP_B

PHONE_1 = "13812345678"  # 合成号码
PHONE_2 = "13912345678"
ADDRESS = "测试省示例市演示区样例路 8 号"
NICK = "合成昵称"  # rodit.app.demo.taobao_event 放进原始事件的买家昵称
RAW_BUYER_ID = RAW_BUYERS["BUYER"][1]
SECRETS = (PHONE_1, PHONE_2, "样例路 8 号", "样例路8号", NICK, RAW_BUYER_ID)


def _leaks(text: str) -> list[str]:
    return [s for s in SECRETS if s in text]


def test_synthetic_pii_is_recognised_by_the_masker():
    """前提：这几条合成个人信息本身能被 mask_pii 识别（否则下面的检查没有意义）。"""
    masked = mask_pii(f"我手机{PHONE_1}，改到{ADDRESS}")
    assert PHONE_1 not in masked and "样例路" not in masked
    assert PHONE_MASK in masked and ADDRESS_MASK in masked


def test_buyer_pii_never_reaches_llm_prompts_logs_or_records():
    llm = production_like_llm(
        routes=[
            ("快递到哪了", Intent.LOGISTICS_TRACK),
            ("改地址", Intent.MODIFY_ORDER),
            ("穿什么码", Intent.SIZE_ADVICE),
        ],
        generate={
            "改地址": llm_answer(
                "亲，订单还没发货，可以修改地址，请您在订单页修改，或者由客服在后台帮您修改哦~",
                "policy:modify_order",
            ),
            "穿什么码": llm_answer(
                "亲，身高 160–165 cm、体重 100–115 斤建议选 M 码哦~", "item:i_555:size_advice"
            ),
        },
    )
    service = demo_service(llm=llm)  # 默认配置：路由、质检都是 incloud_llm，和生产一样
    unit = service.unit
    chat = Chat(service, "c_pii")

    logistics = chat.say(f"我手机{PHONE_1}，快递到哪了", order_id="o_1002")
    modify = chat.say(f"能改地址吗，我要改成{ADDRESS}", order_id="o_1001")
    size = chat.say(f"我165 110斤，电话{PHONE_2}，穿什么码", item_id="i_555")

    # 三条都正常处理：路由、质检、生成都真的调了大模型
    assert logistics.route.intent == Intent.LOGISTICS_TRACK and logistics.suggestions
    assert modify.route.action == RouteAction.HIGH_RISK_ANSWER and modify.suggestions
    assert size.route.intent == Intent.SIZE_ADVICE and size.suggestions
    assert not any(r.route.degraded for r in (logistics, modify, size))
    for purpose in ("route", "qa", "generate"):
        assert llm.calls_for(purpose), purpose

    # 客服把尺码推荐标为"不合适"，进入标注集和缺失知识（反馈闭环也不能带出个人信息）
    service.feedback_rejected(
        SHOP_A,
        size.suggestions[0].suggestion_id,
        agent_id=AGENT,
        reason=RejectReason.WRONG_INTENT,
        true_intent=Intent.PRODUCT_SPEC,
    )

    prompts = llm.all_prompt_text()
    assert _leaks(prompts) == []
    assert PHONE_MASK in prompts and ADDRESS_MASK in prompts  # 遮盖后照样进了提示词
    for where, text in stored_texts(unit, SHOP_A).items():
        assert _leaks(text) == [], where
    questions = stored_texts(unit, SHOP_A)["suggestion.question_text"]
    assert PHONE_MASK in questions and ADDRESS_MASK in questions


# ================================================================ 租户隔离（§9.2）

A_ONLY = ("法式碎花连衣裙", "小雅女装旗舰店", "48 小时", "中通", "杭州转运中心", "160–165")


def test_same_item_and_buyer_in_two_shops_never_cross():
    """店铺 A、B 在同一个平台单元：B 用同样的 item_id（i_555）、同一个买家问同样的问题，
    推荐、依据、大模型提示词里都只有 B 自己的数据。"""
    service = demo_service()  # 离线替身：大模型只复述依据，依据串了店一眼就能看出来
    unit = service.unit
    llm = unit.llm
    assert isinstance(llm, EvidenceEchoLLM)

    # A 先正常用一轮，并把客服改过的回答审核入库（A 的新 FAQ）
    a_chat = Chat(service, "c_iso", seller_id=SHOP_A)
    a_size = a_chat.say("165 110斤穿什么码", item_id="i_555")
    assert "160–165" in a_size.suggestions[0].text
    a_ship = a_chat.say("什么时候发货")
    assert "48" in a_ship.suggestions[0].text
    remote = a_chat.say("新疆能到吗")
    edited = "亲，新疆可以发，补 15 元运费哦~"
    service.feedback_edited(
        SHOP_A, remote.suggestions[0].suggestion_id, agent_id=AGENT, edited_text=edited
    )
    [candidate] = [k for k in service.pending_candidates(SHOP_A) if k.answer == edited]
    service.approve_candidate(SHOP_A, candidate.id, merchant="店长")
    a_calls = len(llm.calls)

    # B：同一个会话 ID、同一个买家、同一个 item_id
    b_chat = Chat(service, "c_iso", seller_id=SHOP_B)
    b_size = b_chat.say("165 110斤穿什么码", item_id="i_555")
    b_ship = b_chat.say("什么时候发货")
    b_courier = b_chat.say("发什么快递")
    b_remote = b_chat.say("新疆能到吗")
    b_order = b_chat.say("我的快递到哪了", order_id="o_1002")  # A 的订单号

    assert b_size.route.intent == Intent.SIZE_ADVICE
    assert "72" in b_ship.suggestions[0].text  # B 的发货政策
    assert "圆通" in b_courier.suggestions[0].text  # B 的 FAQ
    b_results = (b_size, b_ship, b_courier, b_remote, b_order)
    for result in b_results:
        assert result.shop_id == SHOP_B
        for card in result.suggestions:
            assert not [s for s in A_ONLY if s in card.text], card.text
            assert "15 元" not in card.text  # A 刚审核入库的 FAQ
            assert "tool:logistics:o_1002" not in {e.id for e in card.evidence}
    # 给 B 的大模型提示词里没有 A 的任何知识
    b_prompts = "\n".join(
        m.content for c in llm.calls[a_calls:] if c.shop_id == SHOP_B for m in c.messages
    )
    assert b_prompts and not [s for s in A_ONLY if s in b_prompts]
    assert "B 店专用尺码建议" in b_prompts and "木棉男装" in b_prompts  # B 自己的商品、店名
    assert all(c.shop_id == SHOP_B for c in llm.calls[a_calls:])

    # 推荐记录、会话、审计、审核各归各店
    repos = unit.repos
    a_ids = {r.id for r in repos.suggestions.list(SHOP_A)}
    b_ids = {r.id for r in repos.suggestions.list(SHOP_B)}
    assert a_ids and b_ids and not a_ids & b_ids
    assert all(repos.suggestions.get(SHOP_B, i) is None for i in a_ids)
    a_msgs = repos.conversations.recent_messages(SHOP_A, "c_iso")
    b_msgs = repos.conversations.recent_messages(SHOP_B, "c_iso")
    assert {m.shop_id for m in a_msgs} == {SHOP_A} and {m.shop_id for m in b_msgs} == {SHOP_B}
    assert edited not in {m.text for m in b_msgs}
    assert all(e.shop_id == SHOP_B for e in repos.audit.list_events(SHOP_B))
    b_texts = stored_texts(unit, SHOP_B)
    for where, text in b_texts.items():
        assert not [s for s in A_ONLY if s in text], where
    # B 不能审核 A 的售后单；B 的侧边栏只有 B 的推荐
    with pytest.raises(KeyError):
        service.review_aftersale(SHOP_B, "as_2003")
    sidebar = unit.adapters.get(unit.platform, "B")
    assert sidebar.sidebar(SHOP_B, "c_iso") == b_order.suggestions
    assert sidebar.sidebar(SHOP_A, "c_iso") == remote.suggestions
