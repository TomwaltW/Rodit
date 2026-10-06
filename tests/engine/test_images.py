"""问答引擎的买家图片理解（方案 §15 Phase 3 图片理解、§5.3 合并连发与最小化个人信息、§12.2 降级）。

引擎在会话理解之后识别当前买家消息里的图片（engine.images）：结果进 ctx.image_findings，
描述接在当前买家消息后面（路由、检索、生成、质检都看得到），给客服一条摘要提醒；
识别失败、超时记降级 "vision"，照常出推荐。
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from datetime import timedelta
from typing import Any

import pytest
from engine_support import Kit, make_route

from rodit.core.audit import AuditAction
from rodit.core.config import EngineConfig
from rodit.core.enums import ImageKind, Intent, Mode
from rodit.core.models import ConversationContext, ImageFinding, ImageRef, StandardMessage
from rodit.decision import RuleProvider
from rodit.decision.rules import analyze
from rodit.engine.cards import degradation_note
from rodit.engine.images import (
    ALERT_NO_VISION,
    ALERT_SKIPPED,
    ALERT_VISION_FAILED,
    DEGRADED_VISION,
    ImageReader,
    describe,
    summary_alert,
)
from rodit.inmemory import InMemoryCache
from rodit.routing import DecisionRouter, RoutingPolicy
from rodit.testing.fakes import FakeImageUnderstanding
from rodit.testing.fixtures import CONV, DEMO_IMAGE_FINDINGS, SHOP_A, SHOP_B

PHONE = "13812345678"
DAMAGE = "【买家发来图片：破损（外包装一角压扁，裤腿有一处约 3 厘米的撕裂）】"
DAMAGE_ALERT = (
    "买家发来 1 张图片：破损（93%）——外包装一角压扁，裤腿有一处约 3 厘米的撕裂；"
    "仅供参考，以凭证为准"
)


def with_images(msg: StandardMessage, *refs: str) -> StandardMessage:
    return msg.model_copy(update={"images": [ImageRef(ref=r) for r in refs]})


class Rig:
    """一个带图片理解的引擎：kit 的替身 + FakeImageUnderstanding + 单元缓存。"""

    def __init__(self, kit: Kit, *, vision: Any = "default", cache: Any = "default", **kw: Any):
        self.kit = kit
        self.vision = FakeImageUnderstanding(DEMO_IMAGE_FINDINGS) if vision == "default" else vision
        self.cache = InMemoryCache(kit.clock) if cache == "default" else cache
        self.engine = kit.engine(vision=self.vision, image_cache=self.cache, **kw)

    def send(self, text: str, *refs: str, after_s: float = 60, **kw: Any):
        msg = with_images(self.kit.msg(text, after_s=after_s, **kw), *refs)
        return self.engine.handle(msg)

    @property
    def routed(self) -> ConversationContext:
        return self.kit.router.calls[-1]


@pytest.fixture
def rig(kit: Kit) -> Rig:
    return Rig(kit)


# ================================================================ 描述与提醒（纯函数）


def _finding(kind: ImageKind, conf: float = 0.9, summary: str = "", extracted: str | None = None):
    return ImageFinding(
        ref="img_x", kind=kind, confidence=conf, summary=summary, extracted_text=extracted
    )


def test_describe_variants():
    damage = _finding(ImageKind.DAMAGE, summary="外包装压扁")
    size = _finding(
        ImageKind.SIZE_REFERENCE, summary="身高体重截图", extracted="身高 165cm 体重 110 斤"
    )
    blurry = _finding(ImageKind.DEFECT, conf=0.3, summary="看不太清")
    assert describe(0, [], vision=True) == ""
    assert describe(2, [], vision=False) == "【买家发来 2 张图片】"
    assert describe(1, [], vision=True) == "【买家发来 1 张图片，未能识别】"
    assert describe(1, [damage], vision=True) == "【买家发来图片：破损（外包装压扁）】"
    assert describe(1, [size], vision=True) == (
        "【买家发来图片：尺码截图（身高体重截图；图中文字：身高 165cm 体重 110 斤）】"
    )
    # 置信度太低的不进描述，算未能识别
    assert describe(1, [blurry], vision=True) == "【买家发来 1 张图片，未能识别】"
    assert describe(3, [damage, blurry], vision=True) == (
        "【买家发来 3 张图片：破损（外包装压扁）；另有 2 张未能识别】"
    )
    assert describe(1, [_finding(ImageKind.OTHER)], vision=True) == "【买家发来图片：其他图片】"


def test_describe_only_size_screenshots_carry_extracted_text():
    logistics = _finding(
        ImageKind.LOGISTICS_SCREENSHOT, summary="物流页面截图", extracted=f"联系电话 {PHONE}"
    )
    text = describe(1, [logistics], vision=True)
    assert text == "【买家发来图片：物流 / 订单截图】"
    assert "电话" not in text


@pytest.mark.parametrize("kind", [ImageKind.LOGISTICS_SCREENSHOT, ImageKind.OTHER])
def test_describe_personal_kinds_carry_only_the_label(kind: ImageKind):
    """物流截图、其他图片的摘要可能带收件人姓名（mask_pii 遮不掉）：描述只写类型；
    给客服的提醒里照常有摘要（客服本来就能看图）。"""
    named = _finding(kind, summary="物流页面截图：收件人王小明，已签收", extracted="收件人 王小明")
    damage = _finding(ImageKind.DAMAGE, summary="外包装压扁")
    text = describe(2, [named, damage], vision=True)
    assert "王小明" not in text and "签收" not in text
    assert text.startswith("【买家发来 2 张图片：") and "破损（外包装压扁）" in text
    alert = summary_alert(2, [named, damage])
    assert alert is not None and "王小明" in alert


def test_history_description_of_personal_kinds_carries_only_the_label(rig: Rig):
    """历史轮次里缓存记得的物流截图同样只写类型（engine.images._annotate_history）。"""
    rig.vision.findings["img_named"] = {
        "kind": ImageKind.LOGISTICS_SCREENSHOT,
        "confidence": 0.9,
        "summary": "物流页面截图：收件人王小明，已签收",
    }
    rig.send("", "img_named")
    assert "王小明" not in rig.routed.current.text
    rig.send("那怎么办")
    history = [t.text for t in rig.routed.history]
    assert any("物流 / 订单截图" in t for t in history)
    assert not any("王小明" in t for t in history)


def test_describe_masks_and_clips():
    leaky = _finding(ImageKind.DAMAGE, summary=f"面单上写着 {PHONE}，" + "很长的描述" * 30)
    text = describe(1, [leaky], vision=True)
    assert PHONE not in text and "[手机号]" in text
    assert text.endswith("…）】") and len(text) < 90


def test_summary_alert_tail_depends_on_kind():
    damage = _finding(ImageKind.DAMAGE, conf=0.93, summary="外包装压扁")
    size = _finding(ImageKind.SIZE_REFERENCE, conf=0.951, summary="")
    assert summary_alert(1, []) is None
    assert (
        summary_alert(1, [damage])
        == "买家发来 1 张图片：破损（93%）——外包装压扁；仅供参考，以凭证为准"
    )
    assert summary_alert(2, [size]) == "买家发来 2 张图片：尺码截图（95%）；仅供参考，请以图片为准"


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("img_buyer_damage_1", {Intent.QUALITY_ISSUE}),
        ("img_buyer_wrong_1", {Intent.QUALITY_ISSUE}),
        ("img_buyer_size_1", {Intent.SIZE_ADVICE}),
        ("img_buyer_logistics_1", {Intent.LOGISTICS_TRACK, Intent.NOT_RECEIVED}),
    ],
)
def test_keyword_rules_route_image_descriptions(ref: str, expected: set[Intent]):
    """关键词规则（判断层不可用时的兜底，§12.2）靠描述里的图片类型名认出问题类型。"""
    finding = FakeImageUnderstanding(DEMO_IMAGE_FINDINGS).analyze(SHOP_A, ImageRef(ref=ref))
    for text in ("", "你们看看"):
        result = analyze(f"{text} {describe(1, [finding], vision=True)}".strip())
        assert result.choice in expected


def test_keyword_rules_route_defect_photo_to_quality():
    defect = _finding(ImageKind.DEFECT, summary="领口开线，有一处污渍")
    assert analyze(describe(1, [defect], vision=True)).choice == Intent.QUALITY_ISSUE


@pytest.mark.parametrize(
    ("ref", "intent"),
    [
        ("img_buyer_damage_1", Intent.QUALITY_ISSUE),
        ("img_defect", Intent.QUALITY_ISSUE),
        ("img_buyer_wrong_1", Intent.QUALITY_ISSUE),
        ("img_buyer_size_1", Intent.SIZE_ADVICE),
    ],
)
def test_rule_router_through_engine_routes_image_only_messages(kit: Kit, ref: str, intent: Intent):
    """只发图的消息经引擎走真实的路由（关键词规则判断层）：路由看到的是图片描述。"""
    findings = {
        **DEMO_IMAGE_FINDINGS,
        "img_defect": {"kind": ImageKind.DEFECT, "confidence": 0.88, "summary": "领口开线"},
    }
    router = DecisionRouter(
        RuleProvider(), RuleProvider(), RoutingPolicy(kit.config.routing), kit.audit, kit.clock
    )
    rig = Rig(kit, vision=FakeImageUnderstanding(findings), router=router)
    result = rig.send("", ref, item_id="i_555")
    assert result.route is not None and result.route.intent == intent
    assert result.route.degraded is False
    if intent == Intent.QUALITY_ISSUE:  # 高风险类型：推荐 + 风险提示（§6.6 规则 4）
        assert result.route.rule == 4 and result.route.risk_hints


def test_reader_rejects_bad_settings(kit: Kit):
    with pytest.raises(ValueError):
        ImageReader(None, None, kit.clock, max_images=0)
    with pytest.raises(ValueError):
        ImageReader(None, None, kit.clock, budget_s=0)


# ================================================================ 识别成功


def test_damage_photo_reaches_router_findings_and_alerts(rig: Rig):
    result = rig.send("", "img_buyer_damage_1")
    assert result is not None
    ctx = rig.routed
    assert ctx.current.text == DAMAGE
    [finding] = ctx.image_findings
    assert (finding.ref, finding.kind, finding.confidence) == (
        "img_buyer_damage_1",
        ImageKind.DAMAGE,
        0.93,
    )
    assert DAMAGE_ALERT in result.alerts
    assert result.degraded == [] and result.suggestions  # 照常出推荐
    assert rig.vision.calls == [(SHOP_A, "img_buyer_damage_1")]
    # 推荐记录的买家问题带上图片描述（打过码），复核时看得到买家发了什么图
    [record] = rig.kit.suggestions.list(SHOP_A)
    assert record.question_text == DAMAGE


def test_caption_and_image_in_one_message(rig: Rig):
    rig.send("收到就是这样的", "img_buyer_damage_1")
    assert rig.routed.current.text == f"收到就是这样的 {DAMAGE}"


def test_size_screenshot_carries_extracted_text(rig: Rig):
    rig.send("", "img_buyer_size_1", item_id="i_555")
    assert rig.routed.current.text == (
        "【买家发来图片：尺码截图（买家发来的身高体重截图；图中文字：身高 165cm 体重 110 斤）】"
    )


def test_understand_audit_records_counts_and_kinds_only(rig: Rig):
    rig.send("", "img_buyer_damage_1", "img_buyer_logistics_1")
    [event] = rig.kit.events(AuditAction.UNDERSTAND)
    assert event.detail["images"] == 2
    assert event.detail["image_kinds"] == ["damage", "logistics_screenshot"]
    assert event.detail["images_failed"] == 0 and event.detail["images_skipped"] == 0
    dumped = json.dumps([e.detail for e in rig.kit.audit.list_events(SHOP_A)], ensure_ascii=False)
    for text in ("压扁", "撕裂", "物流页面截图", "img_buyer", PHONE, "[手机号]"):
        assert text not in dumped


def test_messages_without_images_keep_audit_unchanged(rig: Rig):
    rig.send("什么时候发货")
    [event] = rig.kit.events(AuditAction.UNDERSTAND)
    assert "images" not in event.detail
    assert rig.routed.image_findings == [] and rig.vision.calls == []


def test_multiple_images_cap(rig: Rig):
    refs = [
        "img_buyer_damage_1",
        "img_buyer_size_1",
        "img_buyer_wrong_1",
        "img_buyer_logistics_1",
        "img_buyer_product_1",
    ]
    result = rig.send("", *refs)
    # 三张图并行识别，调用的先后不固定：只核对识别的是前三张
    assert sorted(r for _, r in rig.vision.calls) == sorted(refs[:3])
    assert [f.ref for f in rig.routed.image_findings] == refs[:3]
    assert ALERT_SKIPPED.format(n=5, k=3) in result.alerts
    assert rig.routed.current.text.startswith("【买家发来 5 张图片：破损（")
    assert rig.routed.current.text.endswith("；另有 2 张未能识别】")
    assert DEGRADED_VISION not in result.degraded  # 没看的图不算故障
    [event] = rig.kit.events(AuditAction.UNDERSTAND)
    assert event.detail["images_skipped"] == 2


def test_engine_max_images_is_configurable(kit: Kit):
    rig = Rig(kit, max_images=1)
    rig.send("", "img_buyer_damage_1", "img_buyer_size_1")
    assert len(rig.vision.calls) == 1


def test_duplicate_message_does_nothing(rig: Rig):
    msg = with_images(rig.kit.msg(""), "img_buyer_damage_1")
    assert rig.engine.handle(msg) is not None
    assert rig.engine.handle(msg) is None
    assert len(rig.vision.calls) == 1


# ================================================================ 合并连发、历史、乱序、撤回


def test_text_after_image_within_merge_window_reuses_finding(rig: Rig):
    rig.send("", "img_buyer_damage_1")
    rig.send("收到就是这样的", after_s=1)
    ctx = rig.routed
    assert len(ctx.current.msg_ids) == 2
    assert ctx.current.text == f"收到就是这样的 {DAMAGE}"
    assert [f.ref for f in ctx.image_findings] == ["img_buyer_damage_1"]
    assert len(rig.vision.calls) == 1  # 识别结果复用，不再调用视觉模型


def test_without_cache_only_the_own_images_are_seen(kit: Kit):
    rig = Rig(kit, cache=None)
    rig.send("", "img_buyer_damage_1")
    assert rig.routed.current.text == DAMAGE
    rig.send("收到就是这样的", after_s=1)
    assert rig.routed.current.text == "收到就是这样的"  # 前一条的图片没有地方记
    assert rig.routed.image_findings == []


def test_history_turn_gets_description_from_cache(rig: Rig):
    rig.send("", "img_buyer_damage_1")
    rig.kit.agent_says("亲，抱歉，我看一下")
    rig.send("那怎么办")
    ctx = rig.routed
    assert ctx.current.text == "那怎么办"
    assert ctx.image_findings == []
    buyer_turns = [t.text for t in ctx.history if t.role == "buyer"]
    assert buyer_turns == [DAMAGE]
    assert len(rig.vision.calls) == 1


def test_image_memory_expires(rig: Rig):
    rig.send("", "img_buyer_damage_1")
    rig.kit.clock.advance(seconds=1801)
    rig.send("那怎么办", after_s=1)
    assert all("买家发来" not in t.text for t in rig.routed.history)


def test_late_image_message_is_merged_into_latest_turn(rig: Rig):
    rig.vision.findings["img_x"] = dict(DEMO_IMAGE_FINDINGS["img_buyer_damage_1"])
    text = rig.kit.msg("收到就是这样的")
    # 图片比文字早 1 秒发出，却晚到（§8.3 乱序）：重新处理最新那条，图片按 msg_id 从缓存里找
    image = with_images(rig.kit.msg("", after_s=0, at=text.received_at - timedelta(seconds=1)))
    image = with_images(image, "img_x")
    assert rig.engine.handle(text) is not None
    result = rig.engine.handle(image)
    assert result is not None and result.msg_ids == [image.msg_id, text.msg_id]
    assert rig.routed.current.text == f"收到就是这样的 {DAMAGE}"
    assert [f.ref for f in rig.routed.image_findings] == ["img_x"]


def test_recall_forgets_images(rig: Rig):
    image = with_images(rig.kit.msg(""), "img_buyer_damage_1")
    rig.engine.handle(image)
    assert rig.kit.conversations.recall_message(SHOP_A, CONV, image.msg_id)
    rig.engine.on_recall(SHOP_A, CONV, image.msg_id)
    rig.send("收到就是这样的", after_s=1)
    assert rig.routed.current.text == "收到就是这样的"
    assert rig.routed.image_findings == []


def test_purged_shop_cache_forgets_images(rig: Rig):
    rig.send("", "img_buyer_damage_1")
    assert rig.cache.purge_shop(SHOP_A) >= 2  # 图片标识 + 识别结果
    rig.send("收到就是这样的", after_s=1)
    assert "买家发来" not in rig.routed.current.text


# ================================================================ 降级（§12.2）


def test_vision_failure_degrades_but_still_answers(rig: Rig):
    rig.vision.fail = True
    result = rig.send("", "img_buyer_damage_1")
    assert result.degraded == [DEGRADED_VISION]
    assert ALERT_VISION_FAILED in result.alerts
    assert rig.routed.current.text == "【买家发来 1 张图片，未能识别】"
    assert rig.routed.image_findings == []
    [card] = result.suggestions[:1]
    assert degradation_note(DEGRADED_VISION) in card.risk_hints
    [event] = rig.kit.result_events()
    assert DEGRADED_VISION in event.detail["degraded"]


def test_unreadable_image_counts_as_failure(rig: Rig):
    result = rig.send("", "img_buyer_damage_1", "img_unreadable")
    assert result.degraded == [DEGRADED_VISION]
    assert rig.routed.current.text == (
        "【买家发来 2 张图片：破损（外包装一角压扁，裤腿有一处约 3 厘米的撕裂）；"
        "另有 1 张未能识别】"
    )
    assert result.alerts[:2] == [
        "买家发来 2 张图片：破损（93%）——外包装一角压扁，裤腿有一处约 3 厘米的撕裂；"
        "仅供参考，以凭证为准",
        ALERT_VISION_FAILED,
    ]
    [event] = rig.kit.events(AuditAction.UNDERSTAND)
    assert event.detail["images_failed"] == 1


def test_failed_images_are_retried_on_next_message(rig: Rig):
    rig.vision.fail = True
    rig.send("", "img_buyer_damage_1")
    rig.vision.fail = False
    result = rig.send("收到就是这样的", after_s=1)
    assert result.degraded == []
    assert rig.routed.current.text == f"收到就是这样的 {DAMAGE}"
    assert len(rig.vision.calls) == 2


class _Odd:
    provider = "odd"

    def __init__(self, behaviour: str) -> None:
        self.behaviour = behaviour
        self.calls: list[tuple[str, str]] = []

    def analyze(self, shop_id: str, image: ImageRef) -> Any:
        self.calls.append((shop_id, image.ref))
        if self.behaviour == "crash":
            raise RuntimeError(f"视觉模型内部错误 {PHONE}")
        return {"kind": "damage"}  # 不是 ImageFinding


@pytest.mark.parametrize("behaviour", ["crash", "wrong_type"])
def test_unexpected_vision_errors_are_failures(kit: Kit, behaviour: str):
    rig = Rig(kit, vision=_Odd(behaviour))
    result = rig.send("什么时候发货", "img_1")
    assert result.degraded == [DEGRADED_VISION]
    assert result.suggestions
    assert rig.routed.current.text == "什么时候发货 【买家发来 1 张图片，未能识别】"


def test_reader_crash_is_contained(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(rig.engine._images, "read", boom)
    result = rig.send("什么时候发货", "img_buyer_damage_1")
    assert result.degraded == [DEGRADED_VISION]
    assert ALERT_VISION_FAILED in result.alerts and result.suggestions
    assert rig.routed.current.text == "什么时候发货"


class _BrokenCache:
    def get(self, shop_id: str, key: str) -> Any:
        raise OSError("cache down")

    def set(self, shop_id: str, key: str, value: object) -> None:
        raise OSError("cache down")

    def purge_shop(self, shop_id: str) -> int:
        return 0


def test_cache_failure_falls_back_to_own_images(kit: Kit):
    rig = Rig(kit, cache=_BrokenCache())
    result = rig.send("", "img_buyer_damage_1")
    assert rig.routed.current.text == DAMAGE
    assert result.degraded == []  # 缓存故障不算图片识别失败


def test_vision_absent_tells_agent_without_calls(kit: Kit):
    rig = Rig(kit, vision=None)
    result = rig.send("", "img_buyer_damage_1", "img_buyer_size_1")
    assert ALERT_NO_VISION.format(n=2) in result.alerts
    assert result.degraded == []
    assert rig.routed.current.text == "【买家发来 2 张图片】"
    assert rig.routed.image_findings == []


class _BlockingVision(FakeImageUnderstanding):
    def __init__(self) -> None:
        super().__init__(DEMO_IMAGE_FINDINGS)
        self.release = threading.Event()

    def analyze(self, shop_id: str, image: ImageRef) -> ImageFinding:
        self.release.wait(5)
        return super().analyze(shop_id, image)


def test_slow_vision_exceeds_budget_and_is_skipped(kit: Kit):
    vision = _BlockingVision()
    rig = Rig(kit, vision=vision, vision_budget_s=0.05)
    started = time.perf_counter()
    try:
        result = rig.send("什么时候发货", "img_buyer_damage_1")
        assert time.perf_counter() - started < 2
        assert result.degraded == [DEGRADED_VISION]
        assert ALERT_VISION_FAILED in result.alerts and result.suggestions
        assert rig.routed.current.text == "什么时候发货 【买家发来 1 张图片，未能识别】"
        assert rig.engine._background.pending(SHOP_A) == 1  # 还在后台跑
    finally:
        vision.release.set()
    assert rig.engine.wait_background(SHOP_A, timeout=5)  # 取消订购前等得到它
    # 迟到的识别结果写进了缓存，下一条消息就用上了，不再调用
    rig.send("收到就是这样的", after_s=1)
    assert rig.routed.current.text.endswith(DAMAGE)
    assert len(vision.calls) == 1


def test_expired_message_budget_skips_vision(kit: Kit):
    ticks = itertools.count(0, 100)  # 每读一次单调时钟过 100 秒：一开始就超时
    rig = Rig(kit, monotonic=lambda: float(next(ticks)))
    result = rig.send("什么时候发货", "img_buyer_damage_1")
    assert rig.vision.calls == []
    assert DEGRADED_VISION in result.degraded and "budget" in result.degraded


# ================================================================ 全自动（§5.6）


def test_images_block_auto_reply_and_transfer(kit: Kit):
    # 全自动要在生效时段内：这里全天生效（engine.modes，非工作时间试行见 test_modes）
    kit.set_shop_config(SHOP_A, auto_reply_enabled=True, auto_reply_schedule="always")
    kit.router.decide = make_route(Intent.SHIPPING_POLICY, confidence=0.97, auto_eligible=True)
    rig = Rig(kit)
    plain = rig.send("什么时候发货", mode=Mode.AUTO)
    assert plain.auto_reply is not None  # 对照：没有图片时自动回复
    result = rig.send("什么时候发货", "img_buyer_size_1", mode=Mode.AUTO)
    assert result.mode == Mode.AUTO
    assert result.auto_reply is None and result.transfer_to_human is True


# ================================================================ 店铺隔离与个人信息


def test_image_memory_is_per_shop(kit: Kit):
    vision = FakeImageUnderstanding(DEMO_IMAGE_FINDINGS, shop_ids={SHOP_A})
    rig = Rig(kit, vision=vision)
    rig.send("", "img_buyer_damage_1", msg_id="m_same")
    # 另一家店、同样的会话 ID 和消息 ID：A 店记下的 m_same 的图片，B 店看不到
    rig.send("还在吗", after_s=1, shop_id=SHOP_B, msg_id="m_same")
    assert rig.routed.shop_id == SHOP_B
    assert rig.routed.current.text == "还在吗" and rig.routed.image_findings == []
    assert vision.calls == [(SHOP_A, "img_buyer_damage_1")]
    # 同样的图片标识：B 店重新识别（只看 B 店的图），不复用 A 店缓存的识别结果
    b = rig.send("这个怎么办", "img_buyer_damage_1", shop_id=SHOP_B, msg_id="m_b")
    assert vision.calls[-1] == (SHOP_B, "img_buyer_damage_1")
    assert b.degraded == [DEGRADED_VISION]
    assert "破损" not in rig.routed.current.text


class _LeakyVision(FakeImageUnderstanding):
    """没打码的实现（违反 core.ports.ImageUnderstanding 的约定）：引擎要再打一次码。"""

    def analyze(self, shop_id: str, image: ImageRef) -> ImageFinding:
        self.calls.append((shop_id, image.ref))
        return ImageFinding(
            ref="something_else",
            kind=ImageKind.DAMAGE,
            confidence=0.9,
            summary=f"面单上写着 {PHONE}，外包装压扁",
            extracted_text=f"收件人电话 {PHONE}",
        )


def test_leaky_vision_output_is_masked_everywhere(kit: Kit):
    rig = Rig(kit, vision=_LeakyVision({}))
    result = rig.send("", "img_1")
    [finding] = rig.routed.image_findings
    assert finding.ref == "img_1"  # 以请求的图片为准
    assert PHONE not in finding.summary and PHONE not in (finding.extracted_text or "")
    blobs = [
        rig.routed.model_dump_json(),
        result.model_dump_json(),
        json.dumps([e.detail for e in kit.audit.list_events(SHOP_A)], ensure_ascii=False),
        json.dumps([r.model_dump(mode="json") for r in kit.suggestions.list(SHOP_A)]),
    ]
    for blob in blobs:
        assert PHONE not in blob


def test_reader_without_turn_images_returns_context_unchanged(kit: Kit):
    reader = ImageReader(FakeImageUnderstanding(DEMO_IMAGE_FINDINGS), None, kit.clock)
    from rodit.testing.fakes import make_context

    ctx = make_context("什么时候发货", kit.world)
    same, reading = reader.read(ctx)
    assert same is ctx and reading.total == 0 and reading.vision is True


def test_default_engine_config_budget_contains_vision_budget():
    assert EngineConfig().latency.message_budget_s > 1.5
