"""测试替身（CLAUDE.md：平台接口一律 mock；大模型测试用假 provider；单元测试不依赖网络）。

各模块的单元测试共用这里的替身，保证大家对契约的理解一致。
"""

from __future__ import annotations

import itertools
import json
import threading
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any

from rodit.core.decision import (
    Answer,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
    validate_response,
)
from rodit.core.enums import (
    AfterSaleAction,
    AfterSaleStatus,
    Intent,
    Mode,
    Platform,
    Role,
)
from rodit.core.errors import ActionNotAllowed, DecisionError, ToolUnavailable
from rodit.core.models import (
    ActionResult,
    AfterSaleInfo,
    ConversationContext,
    ItemRef,
    LogisticsTrace,
    OrderInfo,
    OrderRef,
    PriceInfo,
    StandardMessage,
    StockInfo,
    Turn,
)
from rodit.core.ports import LLMRequest, LLMResponse
from rodit.core.questions import (
    Q_ANSWERS_QUESTION,
    Q_EMOTION,
    Q_GROUNDED,
    Q_INTENT,
    Q_MULTI_INTENT,
    Q_NEED_HUMAN,
    Q_PLATFORM_VIOLATION,
    Q_PROMISE_RISK,
    Q_TONE,
)
from rodit.testing.fixtures import AGENT, BUYER, CONV, SHOP_A, DemoWorld

# ================================================================ 时间


class FakeClock:
    def __init__(self, now: datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("FakeClock 需要带时区的时间")
        self._now = now

    def now(self) -> datetime:
        return self._now

    def set(self, now: datetime) -> None:
        self._now = now

    def advance(self, **kwargs: float) -> datetime:
        self._now = self._now + timedelta(**kwargs)
        return self._now


# ================================================================ 大模型


LLMHandler = Callable[[LLMRequest], "str | LLMResponse | Exception | dict[str, Any]"]


class FakeLLMClient:
    """假大模型。两种用法：

    - handler(request) → str / dict（自动转 JSON）/ LLMResponse / Exception（会被抛出）；
    - responses=[...]：按顺序依次返回，用完再调用抛 AssertionError。

    calls 记录所有请求，测试可以断言提示词里没有个人信息等。
    """

    def __init__(
        self,
        handler: LLMHandler | None = None,
        *,
        responses: Iterable[str | LLMResponse | Exception | dict[str, Any]] | None = None,
        model: str = "fake-llm",
    ) -> None:
        if (handler is None) == (responses is None):
            raise ValueError("handler 和 responses 必须且只能给一个")
        self._handler = handler
        self._queue = list(responses) if responses is not None else None
        self._lock = threading.Lock()
        self.model = model
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            self.calls.append(request)
            if self._handler is not None:
                result = self._handler(request)
            else:
                assert self._queue is not None
                if not self._queue:
                    raise AssertionError("FakeLLMClient 的预设回复已用完")
                result = self._queue.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, LLMResponse):
            return result
        text = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else result
        prompt_chars = sum(len(m.content) for m in request.messages)
        return LLMResponse(
            text=text, model=self.model, input_tokens=prompt_chars, output_tokens=len(text)
        )

    def calls_for(self, purpose: str) -> list[LLMRequest]:
        return [c for c in self.calls if c.purpose == purpose]

    def all_prompt_text(self) -> str:
        return "\n".join(m.content for c in self.calls for m in c.messages)


# ================================================================ 判断层


def router_answers(
    intent: Intent | str,
    confidence: float = 0.92,
    *,
    probabilities: Mapping[str, float] | None = None,
    need_human: float = 0.03,
    emotion: float = 0.2,
    multi_intent: float = 0.04,
) -> dict[str, Answer]:
    """构造 §6.5 路由响应的 answers。probabilities 为空时剩余概率给 other。"""
    intent = Intent(intent)
    if probabilities is None:
        rest = round(1.0 - confidence, 6)
        probabilities = {intent.value: confidence}
        if rest > 0 and intent != Intent.OTHER:
            probabilities[Intent.OTHER.value] = rest
    return {
        Q_INTENT: ChoiceAnswer(
            choice=intent.value, confidence=confidence, probabilities=dict(probabilities)
        ),
        Q_NEED_HUMAN: NoulAnswer(noul=need_human),
        Q_EMOTION: ScoreAnswer(score=emotion),
        Q_MULTI_INTENT: NoulAnswer(noul=multi_intent),
    }


def qa_answers(
    grounded: float = 0.96,
    answers_question: float = 0.95,
    promise_risk: float = 0.05,
    platform_violation: float = 0.02,
    tone: float = 3.5,
) -> dict[str, Answer]:
    """构造 §5.5 质检响应的 answers。默认全部通过。"""
    return {
        Q_GROUNDED: NoulAnswer(noul=grounded),
        Q_ANSWERS_QUESTION: NoulAnswer(noul=answers_question),
        Q_PROMISE_RISK: NoulAnswer(noul=promise_risk),
        Q_PLATFORM_VIOLATION: NoulAnswer(noul=platform_violation),
        Q_TONE: ScoreAnswer(score=tone),
    }


class FakeDecisionProvider:
    """假判断层：按预设 answers（或函数）回答，只返回请求里问到的问题。

    answers 可以是 dict（问题 → Answer）或 callable(request) → dict。
    fail 不为空时每次都抛出（模拟判断层不可用）。
    """

    def __init__(
        self,
        answers: Mapping[str, Answer] | Callable[[DecisionRequest], Mapping[str, Answer]],
        *,
        name: str = "fake",
        fail: Exception | None = None,
    ) -> None:
        self.name = name
        self._answers = answers
        self.fail = fail
        self.requests: list[DecisionRequest] = []

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        if self.fail is not None:
            raise self.fail
        answers = self._answers(request) if callable(self._answers) else self._answers
        try:
            picked = {k: answers[k] for k in request.questions}
        except KeyError as e:
            raise DecisionError(f"FakeDecisionProvider 没有问题 {e} 的预设答案") from e
        response = DecisionResponse(model=request.model, answers=picked)
        validate_response(request, response)
        return response


# ================================================================ 平台工具（只读）


class InMemoryToolGateway:
    """基于演示世界的只读工具（§7.1）。按 shop_id 隔离。

    - fail_on：方法名集合，调用时抛 ToolUnavailable（模拟限流 / 故障，§12.2）。
    - calls：记录 (方法名, shop_id, 参数)。
    """

    def __init__(self, world: DemoWorld, platform: Platform = Platform.TAOBAO) -> None:
        self.world = world
        self.platform = platform
        self.fail_on: set[str] = set()
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []

    def _enter(self, name: str, shop_id: str, *args: Any) -> None:
        self.calls.append((name, shop_id, args))
        if name in self.fail_on:
            raise ToolUnavailable(f"{name} 限流（模拟）")

    def get_order(self, shop_id: str, order_id: str) -> OrderInfo | None:
        self._enter("get_order", shop_id, order_id)
        return next(
            (
                o.model_copy(deep=True)
                for o in self.world.orders
                if o.shop_id == shop_id and o.order_id == order_id
            ),
            None,
        )

    def list_recent_orders(
        self, shop_id: str, buyer_ref: str, *, limit: int = 5
    ) -> list[OrderInfo]:
        self._enter("list_recent_orders", shop_id, buyer_ref)
        orders = [o for o in self.world.orders if o.shop_id == shop_id and o.buyer_ref == buyer_ref]
        orders.sort(key=lambda o: o.created_at, reverse=True)
        return [o.model_copy(deep=True) for o in orders[:limit]]

    def get_logistics(self, shop_id: str, order_id: str) -> LogisticsTrace | None:
        self._enter("get_logistics", shop_id, order_id)
        return next(
            (
                t.model_copy(deep=True)
                for t in self.world.traces
                if t.shop_id == shop_id and t.order_id == order_id
            ),
            None,
        )

    def get_aftersale(self, shop_id: str, aftersale_id: str) -> AfterSaleInfo | None:
        self._enter("get_aftersale", shop_id, aftersale_id)
        return next(
            (
                a.model_copy(deep=True)
                for a in self.world.aftersales
                if a.shop_id == shop_id and a.aftersale_id == aftersale_id
            ),
            None,
        )

    def list_aftersales(
        self,
        shop_id: str,
        *,
        order_id: str | None = None,
        buyer_ref: str | None = None,
        status: AfterSaleStatus | None = None,
        since: datetime | None = None,
    ) -> list[AfterSaleInfo]:
        self._enter("list_aftersales", shop_id, order_id, buyer_ref, status, since)
        out = [
            a.model_copy(deep=True)
            for a in self.world.aftersales
            if a.shop_id == shop_id
            and (order_id is None or a.order_id == order_id)
            and (buyer_ref is None or a.buyer_ref == buyer_ref)
            and (status is None or a.status == status)
            and (since is None or a.created_at >= since)
        ]
        out.sort(key=lambda a: a.created_at)
        return out

    def get_stock(self, shop_id: str, item_id: str, sku_id: str | None = None) -> StockInfo | None:
        self._enter("get_stock", shop_id, item_id, sku_id)
        rows = [s for s in self.world.stocks if s.shop_id == shop_id and s.item_id == item_id]
        if sku_id is not None:
            return next((s.model_copy() for s in rows if s.sku_id == sku_id), None)
        if not rows:
            return None
        total = sum(s.quantity or 0 for s in rows)
        return StockInfo(
            shop_id=shop_id,
            item_id=item_id,
            quantity=total,
            in_stock=total > 0,
            fetched_at=rows[0].fetched_at,
        )

    def get_price(self, shop_id: str, item_id: str) -> PriceInfo | None:
        self._enter("get_price", shop_id, item_id)
        return next(
            (
                p.model_copy(deep=True)
                for p in self.world.prices
                if p.shop_id == shop_id and p.item_id == item_id
            ),
            None,
        )

    # ---- 测试辅助：模拟平台侧状态变化
    def set_aftersale_status(
        self, shop_id: str, aftersale_id: str, status: AfterSaleStatus, *, at: datetime
    ) -> AfterSaleInfo:
        for i, a in enumerate(self.world.aftersales):
            if a.shop_id == shop_id and a.aftersale_id == aftersale_id:
                updated = a.model_copy(
                    update={
                        "status": status,
                        "updated_at": at,
                        "status_version": str(int(at.timestamp())),
                    }
                )
                self.world.aftersales[i] = updated
                return updated
        raise KeyError(aftersale_id)


_NEXT_STATUS = {
    AfterSaleAction.AGREE_REFUND: AfterSaleStatus.REFUNDING,
    AfterSaleAction.AGREE_RETURN: AfterSaleStatus.WAIT_BUYER_RETURN,
    AfterSaleAction.REFUSE: AfterSaleStatus.REFUSED,
}


class FakeActionPort:
    """假的售后写接口（§7.1）。记录调用，按幂等键去重；给了 gateway 时同步修改平台状态。"""

    def __init__(
        self,
        platform: Platform = Platform.TAOBAO,
        *,
        supported: Iterable[AfterSaleAction] | None = None,
        gateway: InMemoryToolGateway | None = None,
        clock: FakeClock | None = None,
    ) -> None:
        self.platform = platform
        self.supported = frozenset(supported if supported is not None else AfterSaleAction)
        self.gateway = gateway
        self.clock = clock
        self.calls: list[dict[str, Any]] = []
        self._done: dict[str, ActionResult] = {}

    def _do(
        self, action: AfterSaleAction, shop_id: str, aftersale_id: str, key: str, **extra: Any
    ) -> ActionResult:
        if action not in self.supported:
            raise ActionNotAllowed(f"{self.platform} 不支持 {action}")
        if key in self._done:
            return self._done[key].model_copy(update={"duplicate": True})
        self.calls.append(
            {
                "action": action,
                "shop_id": shop_id,
                "aftersale_id": aftersale_id,
                "key": key,
                **extra,
            }
        )
        if self.gateway is not None:
            at = self.clock.now() if self.clock else datetime.now().astimezone()
            self.gateway.set_aftersale_status(shop_id, aftersale_id, _NEXT_STATUS[action], at=at)
        result = ActionResult(
            ok=True,
            action=action,
            aftersale_id=aftersale_id,
            idempotency_key=key,
            platform_code="0",
        )
        self._done[key] = result
        return result

    def agree_refund(
        self, shop_id: str, aftersale_id: str, *, operator: str, idempotency_key: str
    ) -> ActionResult:
        return self._do(
            AfterSaleAction.AGREE_REFUND, shop_id, aftersale_id, idempotency_key, operator=operator
        )

    def agree_return(
        self, shop_id: str, aftersale_id: str, *, operator: str, idempotency_key: str
    ) -> ActionResult:
        return self._do(
            AfterSaleAction.AGREE_RETURN, shop_id, aftersale_id, idempotency_key, operator=operator
        )

    def refuse(
        self,
        shop_id: str,
        aftersale_id: str,
        *,
        operator: str,
        idempotency_key: str,
        reason_code: str | None = None,
        reason_text: str | None = None,
    ) -> ActionResult:
        return self._do(
            AfterSaleAction.REFUSE,
            shop_id,
            aftersale_id,
            idempotency_key,
            operator=operator,
            reason_code=reason_code,
            reason_text=reason_text,
        )


# ================================================================ 构造消息与上下文

_msg_seq = itertools.count(1)


def make_message(
    text: str,
    *,
    world: DemoWorld | None = None,
    at: datetime | None = None,
    shop_id: str = SHOP_A,
    platform: Platform | None = None,
    conversation_id: str = CONV,
    msg_id: str | None = None,
    buyer_ref: str = BUYER,
    item_id: str | None = None,
    sku_id: str | None = None,
    order_id: str | None = None,
    agent_id: str | None = AGENT,
    mode: Mode = Mode.ASSIST,
) -> StandardMessage:
    """构造一条标准买家消息（§8.3）。平台默认取演示世界里店铺的平台。"""
    if platform is None:
        platform = world.shop(shop_id).platform if world else Platform.TAOBAO
    if at is None:
        at = world.now if world else datetime(2026, 9, 28, 10, 35).astimezone()
    return StandardMessage(
        platform=platform,
        shop_id=shop_id,
        conversation_id=conversation_id,
        msg_id=msg_id or f"m_test_{next(_msg_seq):05d}",
        buyer_ref=buyer_ref,
        text=text,
        item_card=ItemRef(item_id=item_id, sku_id=sku_id) if item_id else None,
        order_card=OrderRef(order_id=order_id) if order_id else None,
        agent_id=agent_id,
        mode=mode,
        received_at=at,
    )


def make_context(
    text: str,
    world: DemoWorld,
    *,
    shop_id: str = SHOP_A,
    item_id: str | None = None,
    sku_id: str | None = None,
    order_id: str | None = None,
    candidate_item_ids: list[str] | None = None,
    history: list[tuple[Role, str]] | None = None,
    buyer_ref: str = BUYER,
    mode: Mode = Mode.ASSIST,
    unresolved_turns: int = 0,
    with_recent_orders: bool = False,
) -> ConversationContext:
    """直接构造"会话理解之后"的上下文，供路由、检索、生成、质检的单元测试使用。"""
    shop = world.shop(shop_id)
    now = world.now
    turns = [
        Turn(role=role, text=t, at=now - timedelta(minutes=len(history or []) - i))
        for i, (role, t) in enumerate(history or [])
    ]
    item = world.item(shop_id, item_id) if item_id else None
    order = world.order(shop_id, order_id) if order_id else None
    recent: list[OrderInfo] = []
    if with_recent_orders:
        recent = sorted(
            (o for o in world.orders if o.shop_id == shop_id and o.buyer_ref == buyer_ref),
            key=lambda o: o.created_at,
            reverse=True,
        )[:5]
    return ConversationContext(
        platform=shop.platform,
        shop=shop,
        conversation_id=CONV,
        buyer_ref=buyer_ref,
        mode=mode,
        agent_id=AGENT,
        current=Turn(role=Role.BUYER, text=text, at=now, msg_ids=["m_ctx"]),
        history=turns,
        item=item,
        sku_id=sku_id,
        candidate_item_ids=candidate_item_ids or ([item_id] if item_id else []),
        order=order,
        recent_orders=recent,
        unresolved_turns=unresolved_turns,
        now=now,
    )
