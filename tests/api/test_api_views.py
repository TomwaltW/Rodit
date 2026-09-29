"""响应视图与平台事件的店铺核对细节（§5.6、§13.4、§8.2、§9.2）。"""

from __future__ import annotations

from datetime import datetime

from api_support import EVENTS_PATH, Api

from rodit.api.schemas import AfterSaleReviewView, EngineResultView
from rodit.app.demo import taobao_event
from rodit.core.clock import CST
from rodit.core.enums import (
    AfterSaleStatus,
    Confidence,
    DraftKind,
    Intent,
    Mode,
    ReviewState,
)
from rodit.core.models import AfterSaleReview, EngineResult, EvidenceRefView, SuggestionCard
from rodit.testing.fixtures import SHOP_A, SHOP_B

_CARD = SuggestionCard(
    suggestion_id=7,
    text="亲，现货付款后 48 小时内发货哦~",
    intent=Intent.SHIPPING_POLICY,
    kind=DraftKind.FAQ,
    evidence=[EvidenceRefView(id="faq:2", label="FAQ · 什么时候发货")],
    confidence=Confidence.HIGH,
    qa_scores={"grounded": 0.9},
)


def _result(mode: Mode, **kw) -> EngineResult:
    return EngineResult(
        shop_id=SHOP_A,
        conversation_id="c1",
        msg_ids=["m1"],
        mode=mode,
        suggestions=[_CARD],
        alerts=["买家要求人工，请优先处理"],
        missing_knowledge=["policy:invoice"],
        latency_ms={"route": 3},
        **kw,
    )


def test_view_keeps_card_contents_but_drops_internal_fields():
    view = EngineResultView.from_result(_result(Mode.ASSIST, auto_reply=_CARD))
    [card] = view.suggestions
    assert card.text == _CARD.text and card.evidence[0].label == "FAQ · 什么时候发货"
    assert view.alerts and view.missing_knowledge == ["policy:invoice"]
    assert view.auto_reply is None  # 不是全自动：不出现自动回复
    dumped = view.model_dump()
    assert "qa_scores" not in dumped["suggestions"][0] and "latency_ms" not in dumped
    assert "msg_ids" not in dumped


def test_view_shows_auto_reply_only_in_auto_mode():
    view = EngineResultView.from_result(_result(Mode.AUTO, auto_reply=_CARD))
    assert view.auto_reply is not None and view.auto_reply.suggestion_id == 7


def test_shadow_view_shows_nothing():
    view = EngineResultView.from_result(
        _result(Mode.SHADOW, auto_reply=_CARD, no_suggestion_reason="x")
    )
    assert view.shadow is True
    assert view.suggestions == [] and view.auto_reply is None
    assert view.alerts == [] and view.missing_knowledge == [] and view.no_suggestion_reason is None


class _OpaqueAdapter:
    """不是模拟适配器、也没有 peek_shop_id 的适配器：无法在处理前核对店铺。"""

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def on_message(self, raw):
        self.seen.append(raw)
        return None


class _PeekAdapter(_OpaqueAdapter):
    def __init__(self, shop_id: str | None) -> None:
        super().__init__()
        self.shop_id = shop_id

    def peek_shop_id(self, raw):
        return self.shop_id


def test_events_fail_closed_when_the_adapter_cannot_be_peeked(api: Api, monkeypatch):
    opaque = _OpaqueAdapter()
    monkeypatch.setattr(api.unit.adapters, "get", lambda platform, access: opaque)
    r = api.post(EVENTS_PATH, "integration_a", json=taobao_event("在吗", msg_id="o1"))
    assert r.status_code == 403 and r.json()["error"] == "tenant_isolation"
    assert opaque.seen == []  # 适配器没有碰到这条事件


def test_events_use_adapter_peek_when_available(api: Api, monkeypatch):
    other = _PeekAdapter(SHOP_B)
    monkeypatch.setattr(api.unit.adapters, "get", lambda platform, access: other)
    r = api.post(EVENTS_PATH, "integration_a", json={"anything": 1})
    assert r.status_code == 403 and other.seen == []

    unknown = _PeekAdapter(None)
    monkeypatch.setattr(api.unit.adapters, "get", lambda platform, access: unknown)
    r = api.post(EVENTS_PATH, "integration_a", json={"anything": 1})
    assert r.status_code == 204 and unknown.seen == []


def test_openapi_schema_is_generated_without_auth(api: Api):
    r = api.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/v1/shops/{shop_id}/aftersales/{aftersale_id}/confirm" in paths
    assert "/v1/platform/{access}/events" in paths
    confirm = r.json()["components"]["schemas"]["ConfirmRequest"]
    assert set(confirm["properties"]) == {
        "action",
        "expected_status_version",
        "reason_code",
        "reason_text",
    }


def test_review_view_tolerates_incomplete_metadata():
    review = AfterSaleReview(
        shop_id=SHOP_A,
        aftersale_id="as_x",
        platform_status=AfterSaleStatus.PENDING_REVIEW,
        state=ReviewState.CONFIRM,
        action_result={"history": [{"from": "new"}, "junk"], "result": {"ok": True}},
    )
    view = AfterSaleReviewView.from_review(review)
    assert view.history == [] and view.last_result is None and view.platform is None


def test_review_view_timestamps_are_timezone_aware(api: Api):
    r = api.post(f"/v1/shops/{SHOP_A}/aftersales/as_2001/review", "agent_a")
    updated = datetime.fromisoformat(r.json()["updated_at"])
    assert updated.utcoffset() == CST.utcoffset(None)
