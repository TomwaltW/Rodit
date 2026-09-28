"""提示词（方案 §5.4 骨架、§5.3 最小化个人信息、§6.6 规则 3–5、§5.5 重新生成）。"""

from __future__ import annotations

from rodit.core.config import GenerationConfig
from rodit.core.enums import Intent, Mode, PolicyKey, Role, RouteAction
from rodit.core.models import EvidenceBundle, RoutePart
from rodit.generation import (
    answer_char_limit,
    build_generation_messages,
    build_generation_request,
    split_focus_intent,
)
from rodit.testing.fakes import make_context
from rodit.testing.fixtures import SHOP_B, SHOP_C

CFG = GenerationConfig()


def _text(messages) -> str:
    return "\n".join(m.content for m in messages)


def test_skeleton_follows_design_doc(world, ev, route):
    ctx = make_context(
        "这个是纯棉的吗？",
        world,
        item_id="i_555",
        history=[(Role.BUYER, "在吗"), (Role.AGENT, "亲，在的哦")],
    )
    bundle = EvidenceBundle(evidences=[ev.item_attr("面料"), ev.policy(PolicyKey.SHIP_TIME)])
    messages = build_generation_messages(ctx, route(Intent.PRODUCT_SPEC), bundle, config=CFG)
    assert [m.role for m in messages] == ["system", "user"]
    system, user = messages[0].content, messages[1].content
    assert system.startswith("你是小雅女装旗舰店的客服，正在淘宝上回答买家的咨询。")
    # 五条规则
    assert "1. 只使用【依据】中的信息回答" in system and "这个我帮您确认一下" in system
    assert "2. 不承诺依据之外的任何事情，包括价格、赠品、到货时间、赔偿。" in system
    assert "3. 不提供、也不索要任何站外联系方式、链接或收款方式。" in system
    assert "4. 不使用" in system and "绝对化用语" in system and "功效或医疗" in system
    assert "5. 语气：亲切" in system and "“亲”" in system
    assert "长度不超过 120 字" in system and "一次最多问买家一个问题" in system
    # 依据：带来源 ID，按优先级（政策在商品属性前面）
    assert "【依据】\n[policy:ship_time] 店铺政策 · 发货时效：现货付款后 48 小时内发货" in user
    assert "[item:i_555:attr:面料] 商品属性 · 面料：面料：棉 95%、氨纶 5%" in user
    assert user.index("[policy:ship_time]") < user.index("[item:i_555:attr:面料]")
    # 对话：带角色，当前消息在最后
    dialogue = user.split("【对话】\n")[1].split("\n【要求】")[0].splitlines()
    assert dialogue == ["买家：在吗", "客服：亲，在的哦", "买家（当前消息）：这个是纯棉的吗？"]
    assert "买家的问题类型：商品参数" in user
    assert '"answer"' in user and '"evidence_ids"' in user and '"need_clarify"' in user


def test_request_fields(world, ev, route):
    ctx = make_context("纯棉吗", world, item_id="i_555")
    req = build_generation_request(
        ctx,
        route(Intent.PRODUCT_SPEC),
        EvidenceBundle(evidences=[ev.item_attr("面料")]),
        config=CFG,
    )
    assert req.purpose == "generate"
    assert req.json_mode is True
    assert req.shop_id == ctx.shop_id
    assert req.temperature == CFG.temperature and req.max_tokens == CFG.max_tokens


def test_shop_config_drives_tone_salutation_and_limit(world, route):
    ctx_b = make_context("有什么颜色", world, shop_id=SHOP_B)
    system = build_generation_messages(
        ctx_b, route(Intent.PRODUCT_SPEC), EvidenceBundle(), config=CFG
    )[0].content
    assert system.startswith("你是木棉男装的客服，正在淘宝上")
    assert "语气：专业" in system and "“您”" in system
    assert "小雅女装" not in system

    ctx_c = make_context("有货吗", world, shop_id=SHOP_C, config_update={"max_answer_chars": 60})
    system_c = build_generation_messages(ctx_c, route(Intent.STOCK), EvidenceBundle(), config=CFG)[
        0
    ].content
    assert "你是拼拼小铺的客服，正在拼多多上" in system_c
    assert "语气：简洁" in system_c and "长度不超过 60 字" in system_c


def test_empty_display_name_falls_back(world, route):
    ctx = make_context("在吗", world, config_update={"display_name": ""})
    system = build_generation_messages(ctx, route(Intent.OTHER), EvidenceBundle(), config=CFG)[0]
    assert system.content.startswith("你是本店的客服")


def test_auto_mode_reserves_room_for_ai_prefix(world, route):
    ctx = make_context("在吗", world, mode=Mode.AUTO)
    limit = 120 - len(CFG.auto_reply_prefix)
    assert answer_char_limit(ctx, CFG) == limit
    system = build_generation_messages(ctx, route(Intent.OTHER), EvidenceBundle(), config=CFG)[0]
    assert f"长度不超过 {limit} 字" in system.content
    # 前缀本身不进提示词（前缀只由引擎添加）
    assert CFG.auto_reply_prefix not in system.content


def test_evidence_budget_is_respected(world, ev, route):
    ctx = make_context("介绍一下", world, item_id="i_555")
    evidences = [ev.policy(k) for k in world.policy_map(ctx.shop_id)]
    evidences += [ev.item_attr("面料"), ev.size_advice()]
    bundle = EvidenceBundle(evidences=evidences)
    small = GenerationConfig(max_evidence_chars=200)
    user = build_generation_messages(ctx, route(Intent.OTHER), bundle, config=small)[1].content
    block = user.split("【依据】\n")[1].split("\n【对话】")[0]
    assert len(block) <= 200
    assert block.startswith("[policy:")  # 优先级高的先放
    assert "[item:i_555:size_advice]" not in block  # 优先级低的被挤掉
    full = build_generation_messages(ctx, route(Intent.OTHER), bundle, config=CFG)[1].content
    assert "[item:i_555:size_advice]" in full


def test_no_evidence_is_explicit(world, route):
    ctx = make_context("这个怎么样", world)
    user = build_generation_messages(ctx, route(Intent.OTHER), EvidenceBundle(), config=CFG)[1]
    assert "【依据】\n（没有找到依据）" in user.content


def test_realtime_evidence_carries_as_of(world, ev, route):
    ctx = make_context("到哪了", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    user = build_generation_messages(ctx, route(Intent.NOT_RECEIVED), bundle, config=CFG)[1]
    assert "（数据截至 09-28 10:35）" in user.content


def test_pii_is_masked_everywhere(world, ev, route):
    """§5.3：提示词里没有买家手机号、地址、buyer_ref；客服历史消息和质检原因也遮掉。"""
    phone, street = "13812345678", "文三路 100 号"
    ctx = make_context(
        f"我手机{phone}，地址杭州市西湖区{street} 3 单元 501，明天能到吗",
        world,
        order_id="o_1002",
        history=[
            (Role.BUYER, "我的微信号是 abc_12345 加一下"),
            (Role.AGENT, f"亲，是寄到{street}吗"),
        ],
    )
    msgs = build_generation_messages(
        ctx,
        route(Intent.LOGISTICS_TRACK),
        EvidenceBundle(evidences=[ev.o_1002()]),
        config=CFG,
        retry_feedback=[f"包含站外联系或站外交易内容：{phone}"],
    )
    text = _text(msgs)
    assert phone not in text
    assert "文三路" not in text
    assert "abc_12345" not in text
    assert ctx.buyer_ref not in text
    assert "[手机号]" in text and "[地址]" in text and "[联系方式]" in text


def test_variants(world, ev, route):
    ctx = make_context("纯棉吗", world, item_id="i_555")
    bundle = EvidenceBundle(evidences=[ev.item_attr("面料")])
    direct = build_generation_messages(ctx, route(Intent.PRODUCT_SPEC), bundle, config=CFG)
    follow = build_generation_messages(
        ctx, route(Intent.PRODUCT_SPEC), bundle, config=CFG, variant="with_followup"
    )
    assert "直接回答买家的问题。" in direct[1].content
    assert "追问一个问题（只问一个）" not in direct[1].content
    assert "追问一个问题（只问一个）" in follow[1].content


def test_split_multi_part_focus(world, ev, route):
    """规则 5：引擎按 part 复制路由（action=ANSWER、parts=[]、rule 仍是 5）。"""
    ctx = make_context("纯棉吗？什么时候发货", world, item_id="i_555")
    original = route(
        Intent.PRODUCT_SPEC,
        RouteAction.SPLIT_MULTI,
        rule=5,
        parts=[
            RoutePart(intent=Intent.PRODUCT_SPEC, confidence=0.6),
            RoutePart(intent=Intent.SHIPPING_POLICY, confidence=0.4),
        ],
    )
    part = original.model_copy(
        update={"intent": Intent.SHIPPING_POLICY, "parts": [], "action": RouteAction.ANSWER}
    )
    assert split_focus_intent(part) == Intent.SHIPPING_POLICY
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.SHIP_TIME)])
    user = build_generation_messages(ctx, part, bundle, config=CFG)[1].content
    assert "买家这条消息包含多个问题，本条只回答关于发货与快递的部分" in user
    # 显式指定 focus_intent
    user2 = build_generation_messages(
        ctx, route(Intent.PRODUCT_SPEC), bundle, config=CFG, focus_intent=Intent.PRODUCT_SPEC
    )[1].content
    assert "本条只回答关于商品参数的部分" in user2
    # 普通路由没有这句
    plain = build_generation_messages(ctx, route(Intent.PRODUCT_SPEC), bundle, config=CFG)
    assert "包含多个问题" not in plain[1].content
    assert split_focus_intent(route(Intent.PRODUCT_SPEC)) is None


def test_soothe_and_high_risk_instructions(world, ev, route):
    ctx = make_context("东西坏了！！", world, order_id="o_1003")
    bundle = EvidenceBundle(evidences=[ev.policy(PolicyKey.QUALITY_ISSUE)])
    soothe = build_generation_messages(
        ctx, route(Intent.QUALITY_ISSUE, RouteAction.SOOTHE, rule=3), bundle, config=CFG
    )[1].content
    assert "先用一句道歉或安抚的话开头" in soothe
    risky = build_generation_messages(
        ctx,
        route(
            Intent.PRICE_PROMO,
            RouteAction.HIGH_RISK_ANSWER,
            rule=4,
            high_risk=True,
            risk_hints=["涉及价格承诺，请确认"],
        ),
        bundle,
        config=CFG,
    )[1].content
    assert "这是高风险问题" in risky and "风险提示：涉及价格承诺，请确认。" in risky
    plain = build_generation_messages(ctx, route(Intent.QUALITY_ISSUE), bundle, config=CFG)
    assert "这是高风险问题" not in plain[1].content
    assert "道歉或安抚" not in plain[1].content


def test_retry_feedback_is_appended(world, ev, route):
    ctx = make_context("明天能到吗", world, order_id="o_1002")
    user = build_generation_messages(
        ctx,
        route(Intent.LOGISTICS_TRACK),
        EvidenceBundle(evidences=[ev.o_1002()]),
        config=CFG,
        retry_feedback=["包含依据中没有的时效承诺：明天", "包含绝对化用语：一定"],
    )[1].content
    assert (
        "上次回答未通过质检：包含依据中没有的时效承诺：明天；包含绝对化用语：一定，请修改。" in user
    )
    without = build_generation_messages(
        ctx, route(Intent.LOGISTICS_TRACK), EvidenceBundle(), config=CFG
    )[1].content
    assert "未通过质检" not in without


def test_history_rounds_limit(world, route):
    history = []
    for i in range(8):
        history += [(Role.BUYER, f"问题{i}"), (Role.AGENT, f"回答{i}")]
    ctx = make_context("最后的问题", world, history=history)
    user = build_generation_messages(
        ctx, route(Intent.OTHER), EvidenceBundle(), config=GenerationConfig(history_rounds=2)
    )[1].content
    assert "问题5" not in user and "问题6" in user and "问题7" in user
