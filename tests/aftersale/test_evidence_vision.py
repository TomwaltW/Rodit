"""售后审核助手：R5 凭证的图片理解（方案 §7.3.2 R5 "需要凭证；客服查看后决定"、§15 Phase 3、
§12.2 降级、§5.3 最小化个人信息）。

R5 命中、售后单有凭证图片、注入了 vision 时，建议理由后面加一句"图片理解（仅供参考）：…"；
规则、建议、状态都不变，从不据此自动同意或拒绝。
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from rodit.aftersale.assistant import ReviewAssistant
from rodit.aftersale.evidence_vision import (
    MAX_EVIDENCE_IMAGES,
    NOTE_FAILED,
    SUMMARY_PREFIX,
    EvidenceReading,
    evidence_batch,
    read_evidence,
)
from rodit.app import AssistService
from rodit.app.aftersale_desk import AfterSaleDesk
from rodit.app.demo import build_demo_unit
from rodit.core.audit import AuditAction, InMemoryAuditSink
from rodit.core.config import AfterSaleConfig, EngineConfig, effective_shop_config
from rodit.core.enums import AfterSaleReason, AfterSaleStatus, ImageKind, ReviewAdvice, ReviewState
from rodit.core.models import ImageFinding, ImageRef
from rodit.inmemory import InMemoryAfterSaleReviewRepository
from rodit.testing.fakes import (
    FakeActionPort,
    FakeClock,
    FakeImageUnderstanding,
    InMemoryToolGateway,
)
from rodit.testing.fixtures import DEMO_IMAGE_FINDINGS, SHOP_A, SHOP_B, build_demo_world

PHONE = "13812345678"
AS2009 = (
    "质量 / 破损 / 错漏发，买家已上传 3 张凭证：客服查看凭证后决定；"
    "图片理解（仅供参考）：3 张凭证中 2 张显示破损（裤腿撕裂，线头外露；外包装压痕明显）"
)


class Env:
    def __init__(self, vision: Any = "default", **kw: Any) -> None:
        self.w = build_demo_world()
        self.clock = FakeClock(self.w.now)
        self.gw = InMemoryToolGateway(self.w)
        self.vision = FakeImageUnderstanding(DEMO_IMAGE_FINDINGS) if vision == "default" else vision
        self.audit = InMemoryAuditSink()
        self.repo = InMemoryAfterSaleReviewRepository()
        self.port = FakeActionPort(self.w.shop(SHOP_A).platform, gateway=self.gw, clock=self.clock)
        engine = EngineConfig()
        self.asst = ReviewAssistant(
            self.gw,
            self.repo,
            self.audit,
            self.clock,
            AfterSaleConfig(),
            shop_config=lambda shop_id: effective_shop_config(engine, self.w.shop(shop_id)),
            action_port=self.port,
            vision=self.vision,
            **kw,
        )

    def suggest_events(self) -> list[dict[str, Any]]:
        return [
            e.detail for e in self.audit.list_events(SHOP_A, action=str(AuditAction.REVIEW_SUGGEST))
        ]


def _with_images(env: Env, aftersale_id: str, refs: list[str], **update: Any) -> None:
    info = env.w.aftersale(SHOP_A, aftersale_id)
    patched = info.model_copy(
        update={"evidence_images": refs, "evidence_count": len(refs), **update}
    )
    env.w.aftersales = [patched if a is info else a for a in env.w.aftersales]


# ================================================================ R5 + 凭证


def test_r5_reason_gets_evidence_summary():
    env = Env()
    review = env.asst.review(SHOP_A, "as_2009")
    assert review.matched_rules == ["R5"]
    assert review.suggestion == ReviewAdvice.MANUAL and review.state == ReviewState.CONFIRM
    assert review.reason == AS2009
    # 并行识别：调用顺序不固定，摘要按凭证顺序
    assert sorted(env.vision.calls) == [
        (SHOP_A, "img_as2009_1"),
        (SHOP_A, "img_as2009_2"),
        (SHOP_A, "img_as2009_3"),
    ]
    # 审计只记张数和类型，不记识别出的文字
    [detail] = env.suggest_events()
    assert detail["evidence_vision"] == {
        "images": 3,
        "analyzed": 3,
        "kinds": ["damage", "damage", "other"],
        "failed": 0,
    }
    dumped = json.dumps([e.detail for e in env.audit.list_events(SHOP_A)], ensure_ascii=False)
    assert "撕裂" not in dumped and "img_as2009" not in dumped
    assert env.port.calls == []  # 只给建议，不操作平台


def test_vision_never_changes_rules_or_advice():
    with_vision = Env().asst.review(SHOP_A, "as_2009")
    without = Env(vision=None).asst.review(SHOP_A, "as_2009")
    assert without.reason == "质量 / 破损 / 错漏发，买家已上传 3 张凭证：客服查看凭证后决定"
    for field in ("matched_rules", "suggestion", "state", "requires_manager", "risk_flags"):
        assert getattr(with_vision, field) == getattr(without, field)


def test_vision_failure_notes_unreadable_evidence():
    env = Env()
    env.vision.fail = True
    review = env.asst.review(SHOP_A, "as_2009")
    assert review.reason.endswith(f"；{SUMMARY_PREFIX}{NOTE_FAILED}")
    assert review.suggestion == ReviewAdvice.MANUAL and review.state == ReviewState.CONFIRM
    [detail] = env.suggest_events()
    assert detail["evidence_vision"]["failed"] == 3 and detail["evidence_vision"]["kinds"] == []


def test_partial_failure_and_unclear_images():
    findings = {
        **DEMO_IMAGE_FINDINGS,
        "img_blurry": {"kind": ImageKind.DEFECT, "confidence": 0.3, "summary": "看不清"},
        "img_wrong": {"kind": ImageKind.WRONG_ITEM, "confidence": 0.8, "summary": "颜色不对"},
    }
    env = Env(vision=FakeImageUnderstanding(findings))
    _with_images(env, "as_2009", ["img_as2009_1", "img_blurry", "img_unreadable", "img_wrong"])
    review = env.asst.review(SHOP_A, "as_2009")
    assert review.reason.endswith(
        "图片理解（仅供参考）：4 张凭证中 1 张显示破损（裤腿撕裂，线头外露），"
        "1 张显示错发漏发（颜色不对）；1 张看不清；另有 1 张未能识别"
    )


def test_no_problem_seen_says_so():
    findings = {"img_ok": {"kind": ImageKind.OTHER, "confidence": 0.9, "summary": "吊牌特写"}}
    env = Env(vision=FakeImageUnderstanding(findings))
    _with_images(env, "as_2009", ["img_ok"])
    review = env.asst.review(SHOP_A, "as_2009")
    assert review.reason.endswith("图片理解（仅供参考）：1 张凭证未识别出破损、质量瑕疵或错发漏发")


def test_at_most_six_images_are_analyzed():
    env = Env()
    refs = [f"img_e{i}" for i in range(8)]
    for ref in refs:
        env.vision.findings[ref] = dict(DEMO_IMAGE_FINDINGS["img_as2009_1"])
    _with_images(env, "as_2009", refs)
    review = env.asst.review(SHOP_A, "as_2009")
    assert len(env.vision.calls) == MAX_EVIDENCE_IMAGES == 6
    assert "8 张凭证（识别了前 6 张）中 6 张显示破损" in review.reason
    [detail] = env.suggest_events()
    assert detail["evidence_vision"]["images"] == 8 and detail["evidence_vision"]["analyzed"] == 6


@pytest.mark.parametrize("aftersale_id", ["as_2001", "as_2002", "as_2003", "as_2005", "as_2007"])
def test_non_r5_reviews_do_not_call_vision(aftersale_id: str):
    env = Env()
    _with_images(env, aftersale_id, ["img_as2009_1"])
    review = env.asst.review(SHOP_A, aftersale_id)
    assert "R5" not in review.matched_rules
    assert env.vision.calls == []
    assert SUMMARY_PREFIX not in (review.reason or "")
    assert "evidence_vision" not in env.suggest_events()[0]


def test_r5_without_evidence_images_waits_for_evidence():
    """as_2004：质量问题、没有凭证 → 等待凭证，不调用视觉模型。"""
    env = Env()
    review = env.asst.review(SHOP_A, "as_2004")
    assert review.state == ReviewState.EVIDENCE and env.vision.calls == []
    assert SUMMARY_PREFIX not in (review.reason or "")


def test_evidence_added_later_is_analyzed_on_refresh():
    env = Env()
    env.asst.review(SHOP_A, "as_2004")
    _with_images(
        env,
        "as_2004",
        ["img_as2009_1", "img_as2009_2"],
        status_version="v2",
        reason=AfterSaleReason.QUALITY,
    )
    review = env.asst.refresh(SHOP_A, "as_2004")
    assert review.state == ReviewState.CONFIRM
    assert review.reason.endswith(
        "图片理解（仅供参考）：2 张凭证中 2 张显示破损（裤腿撕裂，线头外露；外包装压痕明显）"
    )


def test_other_shop_evidence_is_not_readable():
    """vision 只认 A 店的图（§9.2）：B 店身份识别不了，按未能识别处理。"""
    vision = FakeImageUnderstanding(DEMO_IMAGE_FINDINGS, shop_ids={SHOP_B})
    env = Env(vision=vision)
    review = env.asst.review(SHOP_A, "as_2009")
    assert {s for s, _ in vision.calls} == {SHOP_A}
    assert review.reason.endswith(NOTE_FAILED)


# ================================================================ read_evidence


class _Leaky:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def analyze(self, shop_id: str, image: ImageRef) -> ImageFinding:
        self.calls.append(image.ref)
        return ImageFinding(
            ref="whatever",
            kind=ImageKind.DAMAGE,
            confidence=0.9,
            summary=f"面单 {PHONE} 旁边破了",
            extracted_text=f"电话 {PHONE}",
        )


def test_read_evidence_masks_and_keeps_requested_ref():
    reading = read_evidence(_Leaky(), SHOP_A, ["img_1"])
    [finding] = reading.findings
    assert finding.ref == "img_1"
    assert PHONE not in finding.summary and PHONE not in (finding.extracted_text or "")
    assert PHONE not in reading.summary()


def test_read_evidence_rejects_invalid_refs_and_dedupes():
    vision = FakeImageUnderstanding(DEMO_IMAGE_FINDINGS)
    reading = read_evidence(vision, SHOP_A, ["img_as2009_1", "img_as2009_1", "", "x" * 201])
    assert vision.calls == [(SHOP_A, "img_as2009_1")]
    assert reading.total == 3 and reading.failed == 2 and len(reading.findings) == 1


def test_read_evidence_only_invalid_refs():
    reading = read_evidence(FakeImageUnderstanding({}), SHOP_A, [""])
    assert reading == EvidenceReading(total=1, findings=(), failed=1, skipped=0)
    assert reading.summary() == SUMMARY_PREFIX + NOTE_FAILED


class _Slow(FakeImageUnderstanding):
    def __init__(self) -> None:
        super().__init__(DEMO_IMAGE_FINDINGS)
        self.release = threading.Event()

    def analyze(self, shop_id: str, image: ImageRef) -> ImageFinding:
        if image.ref == "img_as2009_2":
            self.release.wait(5)
        return super().analyze(shop_id, image)


def test_read_evidence_timeout_counts_as_failed():
    vision = _Slow()
    try:
        reading = read_evidence(vision, SHOP_A, ["img_as2009_1", "img_as2009_2"], timeout_s=0.05)
    finally:
        vision.release.set()
    assert reading.failed == 1 and [f.ref for f in reading.findings] == ["img_as2009_1"]
    assert reading.summary().endswith("；另有 1 张未能识别")


def test_review_with_timeout_still_saves():
    vision = _Slow()
    env = Env(vision=vision, vision_timeout_s=0.05)
    try:
        review = env.asst.review(SHOP_A, "as_2009")
    finally:
        vision.release.set()
    assert review.version == 1 and review.state == ReviewState.CONFIRM
    assert "另有 1 张未能识别" in review.reason


def test_status_changes_do_not_skip_vision_on_reevaluate():
    """待确认期间平台上的售后单变了（如补了凭证）：重新跑规则，图片摘要按新凭证给。"""
    env = Env()
    env.asst.review(SHOP_A, "as_2009")
    _with_images(env, "as_2009", ["img_as2009_3"], status_version="v9")
    review = env.asst.refresh(SHOP_A, "as_2009")
    assert review.reason.endswith("图片理解（仅供参考）：1 张凭证未识别出破损、质量瑕疵或错发漏发")
    assert review.platform_status == AfterSaleStatus.PENDING_REVIEW


# ================================================================ 批量审核：共用预算和熔断


class _Hung(FakeImageUnderstanding):
    """视觉模型挂住：analyze 一直等到 release（refs 记下开始识别的图，finished 记做完的张数）。"""

    def __init__(self) -> None:
        super().__init__(DEMO_IMAGE_FINDINGS)
        self.release = threading.Event()
        self.refs: list[str] = []
        self.finished = 0
        self._lock = threading.Lock()

    @property
    def started(self) -> int:
        return len(self.refs)

    def analyze(self, shop_id: str, image: ImageRef) -> ImageFinding:
        with self._lock:
            self.refs.append(image.ref)
        try:
            self.release.wait(10)
            return super().analyze(shop_id, image)
        finally:
            with self._lock:
                self.finished += 1


def test_batch_skips_vision_after_one_review_timed_out():
    """一批里有一张单的识别超时（视觉模型可能挂了）：这一批剩下的单不再识别，理由里写未能识别；
    单张审核（不在批量里）照旧每张单都识别。"""
    vision = _Hung()
    env = Env(vision=vision, vision_timeout_s=0.05)
    _with_images(env, "as_2004", ["img_as2009_1"], reason=AfterSaleReason.QUALITY)
    try:
        with evidence_batch():
            first = env.asst.review(SHOP_A, "as_2009")
            calls_after_first = vision.started
            second = env.asst.review(SHOP_A, "as_2004")
        assert calls_after_first == 3 and vision.started == 3  # 第二张单没有再调用
        assert first.reason.endswith(NOTE_FAILED) and second.reason.endswith(NOTE_FAILED)
        assert second.state == ReviewState.CONFIRM and second.matched_rules == ["R5"]
        # 批量之外：照旧识别（熔断只管这一批）
        _with_images(env, "as_2004", ["img_as2009_2"], status_version="v2")
        env.asst.refresh(SHOP_A, "as_2004")
        assert vision.refs[3:] == ["img_as2009_2"]
    finally:
        vision.release.set()


def test_batch_budget_caps_the_total_wait():
    """一批的识别加起来不超过预算：预算用完后的单不再识别（不用等到某张单超时）。"""
    clock = [0.0]
    env = Env()
    _with_images(env, "as_2004", ["img_as2009_1"], reason=AfterSaleReason.QUALITY)
    with evidence_batch(10.0, now=lambda: clock[0]):
        first = env.asst.review(SHOP_A, "as_2009")
        assert len(env.vision.calls) == 3 and NOTE_FAILED not in first.reason
        clock[0] = 10.0  # 预算用完
        second = env.asst.review(SHOP_A, "as_2004")
    assert len(env.vision.calls) == 3
    assert second.reason.endswith(f"{SUMMARY_PREFIX}{NOTE_FAILED}")


def test_evidence_batch_rejects_bad_budget():
    with pytest.raises(ValueError), evidence_batch(0):
        pass


def test_timed_out_vision_is_tracked_until_it_finishes():
    """超时被丢下的识别记在审核助手的后台任务里（取消订购删数据之前等它们做完）。"""
    vision = _Hung()
    env = Env(vision=vision, vision_timeout_s=0.05)
    try:
        env.asst.review(SHOP_A, "as_2009")
        assert vision.finished < vision.started
        assert not env.asst.wait_background(SHOP_A, timeout=0)
        assert env.asst.wait_background(SHOP_B, timeout=0)  # 别的店铺不受影响
    finally:
        vision.release.set()
    assert env.asst.wait_background(SHOP_A, timeout=5)
    assert vision.finished == vision.started == 3


# ================================================================ 售后工作台批量跑规则


def _desk_service(vision: FakeImageUnderstanding) -> AssistService:
    world = build_demo_world()
    info = world.aftersale(SHOP_A, "as_2004")  # 质量问题补了凭证：和 as_2009 一样命中 R5
    patched = info.model_copy(update={"evidence_images": ["img_as2009_1"], "evidence_count": 1})
    world.aftersales = [patched if a is info else a for a in world.aftersales]
    unit = build_demo_unit(world.shop(SHOP_A).platform, world=world)
    unit.vision.analyze = vision.analyze  # type: ignore[method-assign]  # 引擎和审核助手同一个实例
    return AssistService(unit)


def test_desk_batch_stops_waiting_for_a_hung_vision_model():
    """售后工作台一次跑一批（最多 50 张）：视觉模型挂住时，第一张 R5 单等到这一批的预算就熔断，
    后面的 R5 单不再识别——一次请求不会因为每张单各等 8 秒而拖到几分钟。"""
    vision = _Hung()
    service = _desk_service(vision)
    desk = AfterSaleDesk(service.unit, service.review_aftersale, vision_budget_s=0.3)
    started = time.monotonic()
    try:
        result = desk.review_pending(SHOP_A, limit=50)
        elapsed = time.monotonic() - started
    finally:
        vision.release.set()
    reviews = {o.aftersale_id: o.review for o in result.outcomes if o.review is not None}
    assert {"as_2004", "as_2009"} <= set(reviews)
    assert reviews["as_2004"].reason.endswith(NOTE_FAILED)
    assert reviews["as_2009"].reason.endswith(NOTE_FAILED)
    assert vision.refs == ["img_as2009_1"]  # 只有第一张 R5 单调用了
    assert elapsed < 0.3 + 2.0


def test_revoke_waits_for_vision_left_running_by_a_review():
    """取消订购：先等审核助手留在后台的识别做完再删数据（迟到的写入一并删掉，§8.4）。"""
    vision = _Hung()
    service = _desk_service(vision)
    AfterSaleDesk(service.unit, service.review_aftersale, vision_budget_s=0.1).review_pending(
        SHOP_A, limit=50
    )
    assert vision.finished < vision.started
    service.background_drain_s = 5.0
    timer = threading.Timer(0.2, vision.release.set)
    timer.start()
    try:
        counts = service.revoke_shop(SHOP_A)
    finally:
        timer.cancel()
        vision.release.set()
    assert counts is not None
    assert vision.finished == vision.started == 1
