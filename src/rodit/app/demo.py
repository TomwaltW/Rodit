"""本地演示：用合成数据组装一个平台单元，跑方案 §11 的核心流程（python -m rodit.app.demo）。

- 只用合成演示世界（rodit.testing.fixtures），不联网、不需要任何密钥（CLAUDE.md 云端开发硬规则）；
- 大模型默认用离线替身（rodit.app.offline）：生成只复述依据，质检用启发式判断，路由用关键词规则。
  设了 RODIT_LLM_BASE_URL / RODIT_LLM_MODEL / RODIT_LLM_API_KEY 时，演示改用真实的境内模型
  （只在本地开发机上这样做；云端沙箱只用合成数据，数据也不出境）；
- 单元只装本平台的店铺（§9.1 每个平台云各部署一份）。
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from typing import TextIO

from pydantic import SecretStr

from rodit.adapters import (
    MVP_SCOPES,
    DouyinMockAdapter,
    StandaloneAssistAdapter,
    TaobaoMockAdapter,
)
from rodit.app.offline import EvidenceEchoLLM, HeuristicQAJudge
from rodit.app.service import AssistService
from rodit.app.unit import Unit, build_unit, in_memory_repos
from rodit.core.config import EngineConfig
from rodit.core.decision import DecisionProvider
from rodit.core.enums import AfterSaleAction, Platform, ProviderName
from rodit.core.models import EngineResult, ShopAuth
from rodit.core.pii import BuyerRefHasher
from rodit.core.ports import LLMClient
from rodit.decision import RuleProvider
from rodit.testing.fakes import (
    FakeActionPort,
    FakeClock,
    FakeImageTextExtractor,
    InMemoryItemCatalog,
    InMemoryToolGateway,
)
from rodit.testing.fixtures import (
    DEMO_OCR_TEXTS,
    RAW_BUYERS,
    TEST_SALT,
    DemoWorld,
    build_demo_world,
    demo_platform_items,
)

# 演示用写接口支持的售后操作；拼多多没有查到拒绝售后接口（§7.1），只支持同意
SUPPORTED_ACTIONS = {
    Platform.TAOBAO: frozenset(AfterSaleAction),
    Platform.DOUYIN: frozenset(AfterSaleAction),
    Platform.PDD: frozenset({AfterSaleAction.AGREE_REFUND, AfterSaleAction.AGREE_RETURN}),
}


def seed_unit(unit: Unit, world: DemoWorld) -> None:
    """只把本平台的店铺、政策、商品、FAQ 写进本单元（§9.1），并给每家店铺一份演示授权。"""
    repo = unit.repos.knowledge
    shop_ids = {s.id for s in world.shops.values() if s.platform == unit.platform}
    for shop_id in sorted(shop_ids):
        repo.upsert_shop(world.shop(shop_id))
        unit.repos.auth.save(
            ShopAuth(
                shop_id=shop_id,
                access_token=SecretStr(f"demo-token-{shop_id}"),  # 合成，不是真实令牌
                scopes=sorted(MVP_SCOPES),
                expires_at=world.now + timedelta(days=30),
            )
        )
    for entry in world.policies:
        if entry.shop_id in shop_ids:
            repo.upsert_policy(entry)
    for card in world.items:
        if card.shop_id in shop_ids:
            repo.upsert_item_card(card)
    for item in world.faqs:
        if item.shop_id in shop_ids:
            repo.add_knowledge(item)
    for shop_id in shop_ids:
        unit.searcher.index(shop_id)


def build_demo_unit(
    platform: Platform = Platform.TAOBAO,
    *,
    world: DemoWorld | None = None,
    config: EngineConfig | None = None,
    llm: LLMClient | None = None,
    route_provider: DecisionProvider | None = None,
    qa_provider: DecisionProvider | None = None,
    with_action_port: bool = True,
) -> Unit:
    """演示用的平台单元：内存仓储 + 合成平台数据 + 模拟适配器（模式 B / C）。

    llm 为空时用离线替身（EvidenceEchoLLM），此时路由默认关键词规则、质检默认启发式判断；
    传入真实大模型时，路由和质检按 config.decision 组装。
    """
    world = world or build_demo_world()
    clock = FakeClock(world.now)  # 演示时间固定在合成数据的"现在"，场景可复现
    offline = llm is None
    config = config or EngineConfig()
    if offline:
        config = config.model_copy(
            update={"decision": config.decision.model_copy(update={"provider": ProviderName.RULES})}
        )
        llm = EvidenceEchoLLM()
        route_provider = route_provider or RuleProvider()
        qa_provider = qa_provider or HeuristicQAJudge()
    hasher = BuyerRefHasher(TEST_SALT)
    tools = InMemoryToolGateway(world, platform)
    adapters = [StandaloneAssistAdapter(platform, clock, hasher)]
    if platform == Platform.TAOBAO:
        adapters.append(TaobaoMockAdapter(tools, clock, hasher, access="B"))
    elif platform == Platform.DOUYIN:
        adapters.append(DouyinMockAdapter(tools, clock, hasher, access="B"))
    action_port = (
        FakeActionPort(platform, supported=SUPPORTED_ACTIONS[platform], gateway=tools, clock=clock)
        if with_action_port
        else None
    )
    unit = build_unit(
        platform=platform,
        config=config,
        clock=clock,
        hasher=hasher,
        repos=in_memory_repos(clock),
        llm=llm,
        tools=tools,
        adapters=adapters,
        action_port=action_port,
        route_provider=route_provider,
        qa_provider=qa_provider,
        catalog=InMemoryItemCatalog(demo_platform_items(world), platform),
        ocr=FakeImageTextExtractor(DEMO_OCR_TEXTS),
    )
    seed_unit(unit, world)
    return unit


def taobao_event(
    text: str,
    *,
    msg_id: str,
    session_id: str = "conv_demo",
    buyer: str = "BUYER",
    item_id: str | None = None,
    order_id: str | None = None,
    seller_id: str = "s_1001",
    send_time_ms: int | None = None,
) -> dict:
    """构造一条合成的千牛插件事件（字段见 rodit.adapters.taobao，【待确认】）。"""
    _, raw_buyer = RAW_BUYERS[buyer]
    raw: dict = {
        "seller_id": seller_id,
        "session_id": session_id,
        "msg_id": msg_id,
        "buyer_open_uid": raw_buyer,
        "buyer_nick": "合成昵称",  # 适配器会丢掉，不向后传
        "content": text,
        "msg_type": "text",
        "direction": "in",
        "sub_account": "小雅女装旗舰店:kf_02",
    }
    if item_id:
        raw["item_id"] = item_id
    if order_id:
        raw["tid"] = order_id
    if send_time_ms is not None:
        raw["send_time_ms"] = send_time_ms
    return raw


def _print_result(title: str, result: EngineResult | None, out: TextIO) -> None:
    print(f"\n=== {title}", file=out)
    if result is None:
        print("（重复或非买家消息，未处理）", file=out)
        return
    route = result.route
    if route is not None:
        print(
            f"路由：{route.intent}（{route.intent_confidence:.2f}） → {route.action}"
            f"（规则 {route.rule}）{' [降级]' if route.degraded else ''}",
            file=out,
        )
    for i, card in enumerate(result.suggestions, 1):
        refs = "、".join(e.label for e in card.evidence) or "无"
        print(f"推荐 {i}［{card.kind}/{card.confidence}］：{card.text}", file=out)
        print(f"   依据：{refs}", file=out)
        for hint in card.risk_hints:
            print(f"   风险提示：{hint}", file=out)
    for alert in result.alerts:
        print(f"提醒客服：{alert}", file=out)
    if result.no_suggestion_reason:
        print(f"没有推荐：{result.no_suggestion_reason}", file=out)
    if result.missing_knowledge:
        print(f"缺少知识：{'、'.join(result.missing_knowledge)}", file=out)


def main(out: TextIO | None = None) -> int:
    out = out or sys.stdout
    llm: LLMClient | None = None
    if os.environ.get("RODIT_LLM_BASE_URL") and os.environ.get("RODIT_LLM_MODEL"):
        from rodit.llm import OpenAICompatibleClient

        llm = OpenAICompatibleClient.from_env()
        print("使用真实大模型（环境变量 RODIT_LLM_*）", file=out)
    else:
        print("使用离线替身：生成只复述依据、质检用启发式判断、路由用关键词规则", file=out)
    unit = build_demo_unit(Platform.TAOBAO, llm=llm)
    service = AssistService(unit)
    t0 = int(unit.clock.now().timestamp() * 1000)
    scenarios = [
        ("§11.1 售前尺码咨询", taobao_event("165 110斤穿什么码", msg_id="d1", item_id="i_555")),
        ("§5.2.1 发货时效（政策优先于详情页）", taobao_event("什么时候发货", msg_id="d2")),
        (
            "§11.2 物流查询",
            taobao_event("我的快递到哪了", msg_id="d3", order_id="o_1002", session_id="c3"),
        ),
        (
            "§11.3 显示签收但没收到",
            taobao_event(
                "显示签收了但我根本没收到", msg_id="d4", order_id="o_1003", session_id="c4"
            ),
        ),
        (
            "§11.4 站外交易请求",
            taobao_event("加个微信吧，转账给你便宜点", msg_id="d5", session_id="c5"),
        ),
        ("§5.7 投诉", taobao_event("我要投诉你们", msg_id="d6", session_id="c6")),
    ]
    for i, (title, raw) in enumerate(scenarios):
        raw["send_time_ms"] = t0 + i * 60_000
        unit.clock.set(unit.clock.now() + timedelta(minutes=1))
        _print_result(title, service.handle_event("B", raw), out)

    print("\n=== 模式 C 独立话术助手", file=out)
    _print_result(
        "客服粘贴：能开发票吗",
        service.ask_standalone(shop_id="s_1001", session_id="web_1", question="能开发票吗"),
        out,
    )

    print("\n=== §7.3.2 售后审核助手（只给建议，客服确认后才执行）", file=out)
    for aftersale_id in ("as_2001", "as_2002", "as_2003", "as_2005", "as_3004"):
        review = service.review_aftersale("s_1001", aftersale_id)
        flags = f"；风险：{'、'.join(review.risk_flags)}" if review.risk_flags else ""
        manager = "；须店长确认" if review.requires_manager else ""
        print(
            f"{aftersale_id}：{review.matched_rule} → 建议 {review.suggestion}，"
            f"状态 {review.state}{manager}{flags}\n   理由：{review.reason}",
            file=out,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
