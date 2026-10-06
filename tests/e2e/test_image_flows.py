"""端到端：买家图片（方案 §15 Phase 3 "图片理解：破损照片、尺码截图"、§7.3.1、§11、§12.2）。

模式 B 插件上报的原始图片消息 → 接入层（只收不透明的图片标识）→ 问答引擎识别图片 → 路由、检索、
生成、质检 → 侧边栏（推荐 + 图片摘要提醒）。组件都是真的（rodit.app.demo.build_demo_unit），
视觉模型用 FakeImageUnderstanding（合成数据 DEMO_IMAGE_FINDINGS），不联网。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from e2e_support import (
    demo_service,
    enable_auto,
    judge_state,
    llm_answer,
    production_like_llm,
    route,
    scripted_router,
)

from rodit.app import AssistService
from rodit.app.demo import taobao_event
from rodit.core.audit import AuditAction
from rodit.core.enums import DraftKind, Intent, Mode, Platform, RouteAction
from rodit.core.models import EngineResult
from rodit.engine.images import ALERT_VISION_FAILED, DEGRADED_VISION
from rodit.generation import templates as T
from rodit.knowledge.retrieval import CLARIFY_PHOTO_ITEM_QUESTION
from rodit.testing.fakes import make_context
from rodit.testing.fixtures import RAW_BUYERS, SHOP_A, SHOP_D, build_demo_world

DAMAGE = "【买家发来图片：破损（外包装一角压扁，裙摆有一处约 3 厘米的撕裂）】"
DAMAGE_ALERT = (
    "买家发来 1 张图片：破损（93%）——外包装一角压扁，裙摆有一处约 3 厘米的撕裂；"
    "仅供参考，以凭证为准"
)
SIGNED_URL = "https://img.alicdn.example.invalid/x.jpg?Signature=tok_secret_1"


class Shop:
    """一个淘宝会话（模式 B 插件）：每条消息前时钟往后拨（默认 1 分钟，不合并连发）。"""

    def __init__(self, service: AssistService, session_id: str = "c_img") -> None:
        self.service = service
        self.session_id = session_id
        self.n = 0

    @property
    def unit(self):
        return self.service.unit

    def _raw(self, text: str, after_s: float, **kw: Any) -> dict[str, Any]:
        self.n += 1
        clock = self.unit.clock
        clock.set(clock.now() + timedelta(seconds=after_s))
        return taobao_event(
            text,
            msg_id=kw.pop("msg_id", f"{self.session_id}_{self.n}"),
            session_id=self.session_id,
            send_time_ms=int(clock.now().timestamp() * 1000),
            **kw,
        )

    def image(
        self, pic: str, *, caption: str = "[图片]", after_s: float = 60, robot: bool = False, **kw
    ) -> EngineResult | None:
        raw = self._raw(caption, after_s, **kw)
        raw.update(msg_type="image", pic_key=pic, pic_url=SIGNED_URL)
        if robot:
            raw["reception"] = "robot"
        return self.service.handle_event("A" if robot else "B", raw)

    def say(self, text: str, *, after_s: float = 60, **kw: Any) -> EngineResult | None:
        return self.service.handle_event("B", self._raw(text, after_s, **kw))

    def sidebar_alerts(self) -> list[str]:
        adapter = self.unit.adapters.get(self.unit.platform, "B")
        return adapter.sidebar_alerts(SHOP_A, self.session_id)


@pytest.fixture
def shop() -> Shop:
    return Shop(demo_service())


# ================================================================ 关键词规则判断层（演示默认）


def test_damage_photo_gets_photo_template_and_alert(shop: Shop):
    result = shop.image("img_buyer_damage_1", order_id="o_1013")
    assert result is not None and result.route is not None
    assert result.route.intent == Intent.QUALITY_ISSUE
    assert result.route.action == RouteAction.HIGH_RISK_ANSWER and result.route.rule == 4
    first = result.suggestions[0]
    assert first.kind == DraftKind.TEMPLATE and "您发来的照片我们看到了" in first.text
    assert "申请售后" in first.text and "上传" in first.text
    assert [e.id for e in first.evidence] == ["platform:taobao:apply_aftersale"]
    assert "请买家在申请售后时上传照片凭证" in first.risk_hints  # 高风险类型的风险提示
    assert DAMAGE_ALERT in result.alerts and DAMAGE_ALERT in shop.sidebar_alerts()
    assert result.degraded == []
    assert shop.unit.vision.calls == [(SHOP_A, "img_buyer_damage_1")]
    # 推荐记录里的买家问题是图片描述（消息本身只有图片，没有文字）
    record = shop.unit.repos.suggestions.get(SHOP_A, first.suggestion_id)
    assert record.question_text == DAMAGE
    [msg] = shop.unit.repos.conversations.recent_messages(SHOP_A, "c_img")
    assert (msg.text or "") == ""  # 会话里存的是买家原文：只发图时为空


def test_wrong_item_photo_routes_to_quality_issue(shop: Shop):
    result = shop.image("img_buyer_wrong_1", order_id="o_1013")
    assert result.route.intent == Intent.QUALITY_ISSUE
    assert result.suggestions[0].kind == DraftKind.TEMPLATE
    assert "您发来的照片我们看到了" in result.suggestions[0].text
    assert any(a.startswith("买家发来 1 张图片：错发漏发（86%）") for a in result.alerts)


def test_size_screenshot_answers_from_the_size_chart(shop: Shop):
    result = shop.image("img_buyer_size_1", item_id="i_555")
    assert result.route.intent == Intent.SIZE_ADVICE and result.route.rule == 6
    [card] = result.suggestions
    assert [e.id for e in card.evidence] == ["item:i_555:size_advice"]
    assert "M" in card.text and "165" in card.text  # 截图里的身高体重对上了尺码建议
    assert any(a.startswith("买家发来 1 张图片：尺码截图（95%）") for a in result.alerts)


def test_logistics_screenshot_uses_logistics_tools(shop: Shop):
    result = shop.image("img_buyer_logistics_1", order_id="o_1003")
    assert result.route.intent in (Intent.LOGISTICS_TRACK, Intent.NOT_RECEIVED)
    [card, *_] = result.suggestions
    assert any(e.id == "tool:logistics:o_1003" for e in card.evidence)
    assert "截至" in card.text and "驿站" in card.text


def test_product_photo_with_question_asks_which_item(shop: Shop):
    result = shop.image("img_buyer_product_1", caption="这款还有货吗")
    assert result.route.intent == Intent.STOCK
    [card] = result.suggestions
    assert card.text == f"亲，{CLARIFY_PHOTO_ITEM_QUESTION}"


def test_image_only_product_photo_tells_agent(shop: Shop):
    result = shop.image("img_buyer_product_1")
    assert result.suggestions == []
    assert any(a.startswith("买家发来 1 张图片：商品图（72%）") for a in result.alerts)
    assert result.no_suggestion_reason


def test_vision_failure_degrades_and_still_answers(shop: Shop):
    shop.unit.vision.fail = True
    result = shop.image("img_buyer_damage_1", caption="什么时候发货")
    assert result.route.intent == Intent.SHIPPING_POLICY
    assert result.suggestions
    assert result.degraded == [DEGRADED_VISION]
    assert ALERT_VISION_FAILED in result.alerts and ALERT_VISION_FAILED in shop.sidebar_alerts()
    hints = result.suggestions[0].risk_hints
    assert any("图片没能识别" in h for h in hints)
    [event] = [
        e for e in shop.unit.repos.audit.list_events(SHOP_A) if e.action == AuditAction.RESULT.value
    ]
    assert DEGRADED_VISION in event.detail["degraded"]


def test_unreadable_image_only_message(shop: Shop):
    result = shop.image("img_unreadable")
    assert result.degraded == [DEGRADED_VISION] and ALERT_VISION_FAILED in result.alerts
    assert result.suggestions == []  # 不知道买家想问什么：不推荐，请客服看图


def test_text_after_image_merges_into_one_turn(shop: Shop):
    first = shop.image("img_buyer_damage_1", order_id="o_1013")
    second = shop.say("收到就是这样的", after_s=1)
    assert first is not None and second is not None
    assert second.msg_ids == [first.msg_ids[0], "c_img_2"]
    assert second.route.intent == Intent.QUALITY_ISSUE
    assert "您发来的照片我们看到了" in second.suggestions[0].text
    assert len(shop.unit.vision.calls) == 1  # 识别结果复用
    record = shop.unit.repos.suggestions.get(SHOP_A, second.suggestions[0].suggestion_id)
    assert record.question_text == f"收到就是这样的 {DAMAGE}"


def test_image_recall_clears_sidebar(shop: Shop):
    result = shop.image("img_buyer_damage_1", order_id="o_1013", msg_id="p_recall")
    assert result.suggestions
    recall = taobao_event("", msg_id="r_1", session_id="c_img")
    recall.update(msg_type="recall", recall_msg_id="p_recall")
    assert shop.service.handle_event("B", recall) is None
    adapter = shop.unit.adapters.get(Platform.TAOBAO, "B")
    assert adapter.sidebar(SHOP_A, "c_img") == []


def test_revoke_purges_image_memory(shop: Shop):
    """取消订购（§8.4）：图片标识和识别结果随单元缓存一并删除。"""
    shop.image("img_buyer_damage_1", order_id="o_1013", msg_id="p_rev")
    cache = shop.unit.cache
    assert cache.get(SHOP_A, "vision_msg|c_img|p_rev") is not None
    assert cache.get(SHOP_A, "vision_ref|img_buyer_damage_1") is not None
    counts = shop.service.revoke_shop(SHOP_A)
    assert counts is not None
    assert cache.get(SHOP_A, "vision_msg|c_img|p_rev") is None
    assert cache.get(SHOP_A, "vision_ref|img_buyer_damage_1") is None


def test_duplicate_image_push_is_ignored(shop: Shop):
    assert shop.image("img_buyer_damage_1", msg_id="dup_1") is not None
    assert shop.image("img_buyer_damage_1", msg_id="dup_1") is None
    assert len(shop.unit.vision.calls) == 1


def test_image_message_with_signed_url_only_is_ignored(shop: Shop):
    raw = taobao_event("[图片]", msg_id="u_1", session_id="c_img")
    raw.update(msg_type="image", pic_url=SIGNED_URL)  # 没有图片标识，只有带签名的网址
    assert shop.service.handle_event("B", raw) is None
    assert shop.unit.vision.calls == []


# ================================================================ 大模型判断层（生产组装）


def test_llm_judge_sees_image_description():
    routes = [("破损", Intent.QUALITY_ISSUE), ("尺码截图", Intent.SIZE_ADVICE)]
    generate = {
        "破损": llm_answer("亲，非常抱歉，麻烦您申请售后时上传照片哦"),
        "尺码截图": llm_answer("亲，身高 165、体重 110 斤建议选 M 码哦", "item:i_555:size_advice"),
    }
    llm = production_like_llm(routes, generate)
    shop = Shop(demo_service(llm=llm))
    damage = shop.image("img_buyer_damage_1", order_id="o_1013")
    assert damage.route.intent == Intent.QUALITY_ISSUE
    assert damage.route.provider != "rules" and not damage.route.degraded
    state = judge_state(llm.calls_for("route")[0])
    assert state["messages"][-1]["text"] == DAMAGE
    size = shop.image("img_buyer_size_1", item_id="i_555")
    assert size.route.intent == Intent.SIZE_ADVICE
    gens = [r for r in llm.calls_for("generate") if "尺码截图" in r.messages[1].content]
    assert gens  # 直接回答 + 附带追问两份
    for gen in gens:
        assert "图中文字：身高 165cm 体重 110 斤" in gen.messages[1].content
        assert "对话里【买家发来图片：…】是系统对图片内容的识别" in gen.messages[1].content


# ================================================================ 全自动（§5.6）、抖店、影子模式


def test_auto_mode_with_image_transfers_to_human():
    """图片理解只是参考：带图片的消息不出全自动回复，全自动模式下转人工（§5.6、§5.7）。"""
    service = demo_service(
        router=scripted_router([("什么时候发货", route(Intent.SHIPPING_POLICY, 0.97))])
    )
    adapter = enable_auto(service)
    # 对照：同样的问题不带图片时自动回复
    plain = Shop(service, "c_plain").service.handle_event(
        "A",
        {
            **taobao_event("什么时候发货", msg_id="auto_0", session_id="c_plain"),
            "reception": "robot",
        },
    )
    assert plain is not None and plain.mode == Mode.AUTO and plain.auto_reply is not None
    shop = Shop(service, "c_auto")
    result = shop.image("img_buyer_size_1", caption="什么时候发货", robot=True)
    assert result.mode == Mode.AUTO and result.route.intent == Intent.SHIPPING_POLICY
    assert result.suggestions  # 推荐照常给接手的客服
    assert result.auto_reply is None and result.transfer_to_human is True
    assert [t.conversation_id for t in adapter.transfers(SHOP_A)] == ["c_auto"]
    assert [m.conversation_id for m in adapter.sent_messages(SHOP_A)] == ["c_plain"]


def test_douyin_image_event_in_shadow_shop():
    service = demo_service(Platform.DOUYIN)
    clock = service.unit.clock
    raw = {
        "shop_id": SHOP_D,
        "conversation_short_id": "dyc_img",
        "server_message_id": "dy_img_1",
        "sender": {"role": "customer", "uid": RAW_BUYERS["BUYER_DOUYIN"][1], "nickname": "x"},
        "content": {
            "type": "image",
            "text": "",
            "images": [{"uri": "img_buyer_damage_1", "url": SIGNED_URL}],
        },
        "create_time": int(clock.now().timestamp()),
        "service_uid": "kf_d01",
    }
    result = service.handle_event("B", raw)
    assert result is not None and result.mode == Mode.SHADOW  # 店铺 D 在影子模式（§13.4）
    assert result.route.intent == Intent.QUALITY_ISSUE
    assert any(c.text.startswith("亲，非常抱歉") for c in result.suggestions)
    adapter = service.unit.adapters.get(Platform.DOUYIN, "B")
    assert adapter.sidebar(SHOP_D, "dyc_img") == []  # 影子模式不展示
    assert service.unit.vision.calls == [(SHOP_D, "img_buyer_damage_1")]


def test_photo_template_is_recorded_verbatim(shop: Shop):
    """推荐记录里是模板原文（质检硬规则没有改写它），复核和反馈按它统计。"""
    result = shop.image("img_buyer_damage_1", order_id="o_1013")
    record = shop.unit.repos.suggestions.get(SHOP_A, result.suggestions[0].suggestion_id)
    assert record.kind == DraftKind.TEMPLATE
    assert record.text == T.quality_photo(make_context("", build_demo_world())).text
