"""测试替身（CLAUDE.md：平台接口一律 mock；大模型测试用假 provider；单元测试不依赖网络）。

各模块的单元测试共用这里的替身，保证大家对契约的理解一致。仓储的内存实现在 rodit.inmemory。
"""

from __future__ import annotations

import itertools
import json
import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from pydantic import SecretStr, ValidationError

from rodit.core.clock import Clock
from rodit.core.config import EngineConfig, effective_shop_config
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
    ACTIONABLE_STATUSES,
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
    HumanConfirmation,
    ItemRef,
    LogisticsTrace,
    OrderInfo,
    OrderRef,
    PlatformItem,
    PriceInfo,
    RefreshedToken,
    StandardMessage,
    StockInfo,
    SubscriptionEvent,
    Turn,
)
from rodit.core.pii import mask_pii
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
from rodit.core.scopes import MVP_SCOPES
from rodit.testing.fixtures import AGENT, BUYER, CONV, DEFAULT_NOW, SHOP_A, DemoWorld

# ================================================================ 时间


class FakeClock:
    def __init__(self, now: datetime = DEFAULT_NOW) -> None:
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

    calls 记录所有请求，测试可以断言提示词里没有个人信息等。线程安全（投票会并发调用）。
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
        accepts_buyer_data: bool = True,
    ) -> None:
        self.name = name
        self.accepts_buyer_data = accepts_buyer_data
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


# ---------------------------------------------------------------- 判断层走大模型时的脚本

_JUDGE_STATE = re.compile(r"## 状态（state）\n(?P<state>.*?)\n\n## 问题", re.S)
_JUDGE_QUESTION = re.compile(r"^### (?P<key>\S+)（(?P<type>是否题|单选题|打分题)）", re.M)
_JUDGE_TYPES = {"是否题": "noul", "单选题": "choice", "打分题": "score"}

JUDGE_DEFAULT_ANSWERS: dict[str, dict[str, Any]] = {
    # 路由（§6.5）：不要人工、情绪平稳、只问了一件事
    Q_NEED_HUMAN: {"answer": "否", "p": 0.02},
    Q_EMOTION: {"score": 0},
    Q_MULTI_INTENT: {"answer": "否", "p": 0.05},
    # 质检（§5.5）：全部通过
    Q_GROUNDED: {"answer": "是", "p": 0.95},
    Q_ANSWERS_QUESTION: {"answer": "是", "p": 0.95},
    Q_PROMISE_RISK: {"answer": "否", "p": 0.05},
    Q_PLATFORM_VIOLATION: {"answer": "否", "p": 0.02},
    Q_TONE: {"score": 3},
}
"""判断层提示词里的问题没有写进脚本时的默认作答（大模型输出格式，见 decision.llm）。"""


def judge_questions(request: LLMRequest) -> dict[str, str]:
    """判断层提示词（decision.llm.build_judge_messages）里问到的问题：
    问题名 → noul / choice / score。"""
    text = "\n".join(m.content for m in request.messages)
    return {m["key"]: _JUDGE_TYPES[m["type"]] for m in _JUDGE_QUESTION.finditer(text)}


def judge_state(request: LLMRequest) -> dict[str, Any]:
    """判断层提示词里的 state（已脱敏的 JSON）。不是判断层提示词时返回空 dict。"""
    m = _JUDGE_STATE.search("\n".join(m.content for m in request.messages))
    return json.loads(m.group("state")) if m else {}


def judge_buyer_text(request: LLMRequest) -> str:
    """路由 state 里的当前买家消息（最后一条买家消息，已脱敏）；质检 state 里的买家问题。"""
    state = judge_state(request)
    for message in reversed(state.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == Role.BUYER.value:
            return str(message.get("text") or "")
    return str(state.get("question") or "")


JudgeAnswers = Mapping[str, Mapping[str, Any]]
JudgeScript = Sequence[tuple[str, JudgeAnswers]] | Callable[[LLMRequest], JudgeAnswers]


def _judge_answers(script: JudgeScript | JudgeAnswers | None, request: LLMRequest) -> JudgeAnswers:
    if script is None:
        return {}
    if callable(script):
        return script(request)
    if isinstance(script, Mapping):
        return script
    text = judge_buyer_text(request)
    for keyword, answers in script:
        if keyword in text:
            return answers
    raise AssertionError(f"判断层脚本里没有这条消息：{text}")


def judge_output(
    request: LLMRequest,
    answers: JudgeAnswers,
    *,
    defaults: JudgeAnswers = JUDGE_DEFAULT_ANSWERS,
) -> dict[str, Any]:
    """按提示词问到的问题逐题作答（一次请求答全部问题，§6.4 一次投票一个请求）：脚本里有的用脚本，
    没有的用 defaults；是否题 / 打分题再没有就答"否" / 0；单选题必须写进脚本。
    脚本里有、提示词没问的问题丢掉（路由问题增减时脚本不用跟着改）。"""
    asked = judge_questions(request)
    if not asked:
        raise AssertionError("不是判断层提示词：没有找到问题")
    out: dict[str, Any] = {}
    for key, kind in asked.items():
        if key in answers:
            out[key] = dict(answers[key])
        elif key in defaults:
            out[key] = dict(defaults[key])
        elif kind == "noul":
            out[key] = {"answer": "否", "p": 0.05}
        elif kind == "score":
            out[key] = {"score": 0}
        else:
            raise AssertionError(f"判断层脚本里没有单选题 {key} 的答案")
    return out


def scripted_judge_llm(
    *,
    route: JudgeScript | JudgeAnswers | None = None,
    qa: JudgeScript | JudgeAnswers | None = None,
    generate: Callable[[LLMRequest], Any] | None = None,
    defaults: JudgeAnswers = JUDGE_DEFAULT_ANSWERS,
    model: str = "fake-judge-llm",
) -> FakeLLMClient:
    """路由、质检（、生成）共用一个假大模型，判断层按提示词里的问题逐题作答（多个问题一次答完）：

    - route：[(买家消息里的关键词, 答案)] 按当前买家消息查表，或 dict（每次都一样），
      或 callable(request) → 答案；答案的格式同大模型输出，如
      {"intent": {"choice": "size_advice"}, "multi_intent": {"answer": "是", "p": 0.9}}；
    - qa：同上（默认全部通过）；generate：生成请求的 handler（默认抛 AssertionError）。
    投票时每一票都是一次请求，调用次数 = 投票次数（FakeLLMClient.calls_for("route")）。
    """

    def handler(request: LLMRequest) -> Any:
        if request.purpose in ("route", "qa"):
            script = route if request.purpose == "route" else qa
            try:
                answers = _judge_answers(script, request)
                return judge_output(request, answers, defaults=defaults)
            except AssertionError as exc:
                return exc
        if generate is None:
            return AssertionError(f"这个假大模型不处理 {request.purpose} 请求")
        return generate(request)

    return FakeLLMClient(handler, model=model)


# ================================================================ 店铺授权（§8.4）


class FakeTokenRefresher:
    """实现 core.ports.TokenRefreshPort 的假实现（平台续期接口在平台云内，沙箱里一律 mock）。

    - 默认每次续期成功：新令牌 tok-refreshed-{店铺}-{序号}（合成的），有效期 = 现在 + ttl；
      rotate_refresh=True 时同时换一个续期令牌（有效期 refresh_ttl），否则 refresh_token 为空；
    - script(shop_id, *results)：这家店接下来几次的结果（RefreshedToken 或异常，按顺序用完为止）；
      always(shop_id, result)：这家店之后每次都是这个结果（如 AuthExpired、ToolUnavailable）；
    - calls：[(店铺, 传入的续期令牌)]（SecretStr，repr 里看不到原文）。线程安全。
    """

    def __init__(
        self,
        platform: Platform = Platform.TAOBAO,
        *,
        clock: Clock | None = None,
        ttl: timedelta = timedelta(days=30),
        refresh_ttl: timedelta = timedelta(days=90),
        rotate_refresh: bool = True,
    ) -> None:
        self.platform = Platform(platform)
        self._clock: Clock = clock or FakeClock()
        self._ttl = ttl
        self._refresh_ttl = refresh_ttl
        self._rotate = rotate_refresh
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._queued: dict[str, list[RefreshedToken | Exception]] = {}
        self._always: dict[str, RefreshedToken | Exception] = {}
        self.calls: list[tuple[str, SecretStr]] = []

    def script(self, shop_id: str, *results: RefreshedToken | Exception) -> None:
        with self._lock:
            self._queued.setdefault(shop_id, []).extend(results)

    def always(self, shop_id: str, result: RefreshedToken | Exception | None) -> None:
        """result=None 表示恢复默认（续期成功）。"""
        with self._lock:
            if result is None:
                self._always.pop(shop_id, None)
            else:
                self._always[shop_id] = result

    def calls_for(self, shop_id: str) -> int:
        with self._lock:
            return sum(1 for shop, _ in self.calls if shop == shop_id)

    def refresh(self, shop_id: str, refresh_token: SecretStr) -> RefreshedToken:
        with self._lock:
            self.calls.append((shop_id, refresh_token))
            queued = self._queued.get(shop_id)
            if queued:
                result: RefreshedToken | Exception | None = queued.pop(0)
            else:
                result = self._always.get(shop_id)
            n = next(self._seq)
        if isinstance(result, Exception):
            raise result
        if result is not None:
            return result.model_copy()
        now = self._clock.now()
        return RefreshedToken(
            access_token=SecretStr(f"tok-refreshed-{shop_id}-{n:03d}"),
            expires_at=now + self._ttl,
            refresh_token=SecretStr(f"tok-refresh-{shop_id}-{n:03d}") if self._rotate else None,
            refresh_expires_at=now + self._refresh_ttl if self._rotate else None,
        )


SUBSCRIPTION_EVENT_TYPE = "subscription"
"""合成的订购推送里的 type（模拟平台服务市场的消息；真实格式由平台云代码解析）。"""


def subscription_push(
    kind: str,
    shop_id: str,
    *,
    platform: Platform = Platform.TAOBAO,
    at: datetime | None = None,
    scopes: Iterable[str] | None = None,
    access_token: str | None = None,
    refresh_token: str | None = None,
    expires_at: datetime | None = None,
    refresh_expires_at: datetime | None = None,
    category: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """构造一条合成的订购推送（平台原始格式的模拟，全部是假数据）。

    subscribed / renewed 没给令牌时自动造一个合成令牌、权限默认 MVP 只读、有效期 30 天；
    extra 里的字段原样放进去（如 buyer_nick 之类，用来测试解析时丢弃不认识的字段）。
    """
    at = at or DEFAULT_NOW
    raw: dict[str, Any] = {
        "type": SUBSCRIPTION_EVENT_TYPE,
        "event": kind,
        "shop_id": shop_id,
        "platform": Platform(platform).value,
        "at": at.isoformat(),
    }
    if kind in ("subscribed", "renewed"):  # 开通 / 续订带令牌：没给就造合成的
        scopes = MVP_SCOPES if scopes is None else scopes
        access_token = access_token or f"tok-sub-{shop_id}-synthetic"
        expires_at = expires_at or at + timedelta(days=30)
    for key, value in (
        ("scopes", sorted(scopes) if scopes is not None else None),
        ("access_token", access_token),
        ("expires_at", expires_at.isoformat() if expires_at else None),
        ("refresh_token", refresh_token),
        ("refresh_expires_at", refresh_expires_at.isoformat() if refresh_expires_at else None),
        ("category", category),
    ):
        if value is not None:
            raw[key] = value
    return {**raw, **extra}


_SUBSCRIPTION_FIELDS = (
    "shop_id",
    "platform",
    "at",
    "scopes",
    "access_token",
    "refresh_token",
    "expires_at",
    "refresh_expires_at",
    "category",
)


def parse_subscription(raw: Mapping[str, Any]) -> SubscriptionEvent | None:
    """合成订购推送 → SubscriptionEvent（PlatformBindings.parse_subscription 的假实现）。

    不是订购推送、字段不合法时返回 None（和适配器的 on_event 一样，无法识别的事件不处理）；
    只读认识的字段，其余（昵称之类）一律丢弃。scopes 可以是列表或逗号分隔的字符串。
    """
    if not isinstance(raw, Mapping) or raw.get("type") != SUBSCRIPTION_EVENT_TYPE:
        return None
    data: dict[str, Any] = {k: raw[k] for k in _SUBSCRIPTION_FIELDS if raw.get(k) is not None}
    data["kind"] = raw.get("event")
    scopes = data.get("scopes")
    if isinstance(scopes, str):
        data["scopes"] = [s.strip() for s in scopes.split(",") if s.strip()]
    try:
        return SubscriptionEvent.model_validate(data)
    except ValidationError:
        return None


class FakeSubscriptionParser:
    """按平台解析合成订购推送（模拟适配器那一侧）：别的平台的推送返回 None（§9.1 数据不跨云）。
    是 rodit.testing 的类，生产模式的绑定检查会拒绝它。"""

    def __init__(self, platform: Platform) -> None:
        self.platform = Platform(platform)

    def __call__(self, raw: Mapping[str, Any]) -> SubscriptionEvent | None:
        event = parse_subscription(raw)
        if event is None or event.platform != self.platform:
            return None
        return event


# ================================================================ 平台工具（只读）


class InMemoryToolGateway:
    """基于演示世界的只读工具（§7.1）。按 shop_id 隔离，排序遵守 ToolGateway 协议。

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
            raise ToolUnavailable(f"{name} 限流（模拟）", reason="throttled")

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
        orders.sort(key=lambda o: (o.created_at, o.order_id), reverse=True)
        return [o.model_copy(deep=True) for o in orders[:limit]]

    def get_logistics(self, shop_id: str, order_id: str) -> LogisticsTrace | None:
        self._enter("get_logistics", shop_id, order_id)
        trace = next(
            (t for t in self.world.traces if t.shop_id == shop_id and t.order_id == order_id),
            None,
        )
        if trace is None:
            return None
        # ToolGateway 协议：平台自由文本由实现先脱敏
        events = [
            e.model_copy(update={"description": mask_pii(e.description)}) for e in trace.events
        ]
        return trace.model_copy(deep=True, update={"events": events})

    def get_aftersale(self, shop_id: str, aftersale_id: str) -> AfterSaleInfo | None:
        self._enter("get_aftersale", shop_id, aftersale_id)
        return next(
            (
                _masked(a)
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
            _masked(a)
            for a in self.world.aftersales
            if a.shop_id == shop_id
            and (order_id is None or a.order_id == order_id)
            and (buyer_ref is None or a.buyer_ref == buyer_ref)
            and (status is None or a.status == status)
            and (since is None or a.created_at >= since)
        ]
        out.sort(key=lambda a: (a.created_at, a.aftersale_id))
        return out

    def get_stock(self, shop_id: str, item_id: str, sku_id: str | None = None) -> StockInfo | None:
        self._enter("get_stock", shop_id, item_id, sku_id)
        rows = [s for s in self.world.stocks if s.shop_id == shop_id and s.item_id == item_id]
        if sku_id is not None:
            return next((s.model_copy() for s in rows if s.sku_id == sku_id), None)
        if not rows:
            return None
        if len(rows) == 1 and rows[0].sku_id is None:
            return rows[0].model_copy()
        in_stock = any(s.in_stock for s in rows)
        quantity = (
            None if any(s.quantity is None for s in rows) else sum(s.quantity or 0 for s in rows)
        )
        notes = [s.restock_note for s in rows if not s.in_stock and s.restock_note]
        return StockInfo(
            shop_id=shop_id,
            item_id=item_id,
            quantity=quantity,
            in_stock=in_stock,
            restock_note=None if in_stock else (notes[0] if notes else None),
            fetched_at=max(s.fetched_at for s in rows),
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
        self,
        shop_id: str,
        aftersale_id: str,
        status: AfterSaleStatus,
        *,
        at: datetime,
        evidence_count: int | None = None,
    ) -> AfterSaleInfo:
        for i, a in enumerate(self.world.aftersales):
            if a.shop_id == shop_id and a.aftersale_id == aftersale_id:
                # 版本号用毫秒并保证递增：同一秒内的两次变化也能区分
                version = max(int(at.timestamp() * 1000), int(a.status_version) + 1)
                update: dict[str, Any] = {
                    "status": status,
                    "updated_at": at,
                    "status_version": str(version),
                }
                if evidence_count is not None:
                    update["evidence_count"] = evidence_count
                updated = a.model_copy(update=update)
                self.world.aftersales[i] = updated
                return updated
        raise ActionNotAllowed(f"售后单 {aftersale_id} 不存在或不属于店铺 {shop_id}")


class InMemoryItemCatalog:
    """平台商品接口的替身（实现 core.ports.ItemCatalog，§5.2.3）。按 shop_id 隔离。

    - items：平台上的商品（如 fixtures.demo_platform_items(world)）；可以直接改列表模拟平台变化；
    - fail_on：方法名集合，调用时抛 ToolUnavailable（模拟限流 / 故障）；
    - calls：记录 (方法名, shop_id)。
    """

    def __init__(self, items: Iterable[PlatformItem], platform: Platform = Platform.TAOBAO) -> None:
        self.platform = platform
        self.items: list[PlatformItem] = [i for i in items if i.platform == platform]
        self.fail_on: set[str] = set()
        self.calls: list[tuple[str, str]] = []

    def _enter(self, name: str, shop_id: str) -> None:
        self.calls.append((name, shop_id))
        if name in self.fail_on:
            raise ToolUnavailable(f"{name} 限流（模拟）", reason="throttled")

    def list_items(self, shop_id: str) -> list[PlatformItem]:
        self._enter("list_items", shop_id)
        out = [i for i in self.items if i.shop_id == shop_id]
        return [i.model_copy(deep=True) for i in sorted(out, key=lambda i: i.item_id)]

    def get_item(self, shop_id: str, item_id: str) -> PlatformItem | None:
        self._enter("get_item", shop_id)
        return next(
            (
                i.model_copy(deep=True)
                for i in self.items
                if i.shop_id == shop_id and i.item_id == item_id
            ),
            None,
        )


class FakeImageTextExtractor:
    """详情图 OCR 的替身（实现 core.ports.ImageTextExtractor）：按图片引用查表，
    表里没有的图抛 ToolUnavailable（模拟识别失败）。calls 记录 (shop_id, image_ref)。"""

    def __init__(self, texts: Mapping[str, str]) -> None:
        self.texts = dict(texts)
        self.calls: list[tuple[str, str]] = []

    def extract_text(self, shop_id: str, image_ref: str) -> str:
        self.calls.append((shop_id, image_ref))
        if image_ref not in self.texts:
            raise ToolUnavailable(f"图片 {image_ref} 识别失败（模拟）")
        return self.texts[image_ref]


def _masked(info: AfterSaleInfo) -> AfterSaleInfo:
    return info.model_copy(deep=True, update={"note": mask_pii(info.note) if info.note else None})


_NEXT_STATUS = {
    AfterSaleAction.AGREE_REFUND: AfterSaleStatus.REFUNDING,
    AfterSaleAction.AGREE_RETURN: AfterSaleStatus.WAIT_BUYER_RETURN,
    AfterSaleAction.REFUSE: AfterSaleStatus.REFUSED,
}


class FakeActionPort:
    """假的售后写接口（§7.1），行为遵守 AfterSaleActionPort 协议：

    - 不支持的操作 / 抖店拒绝缺原因码 / 售后单不存在或不属于该店 → ActionNotAllowed；
    - 重读状态：不在 ACTIONABLE_STATUSES 里，或 status_version 和客服确认时看到的不一致
      → ok=False、platform_code="STATUS_CHANGED"，不改平台状态；
    - 同一幂等键只操作一次，重复调用返回 duplicate=True；
    - fail_next=True 时下一次调用返回 ok=False、platform_code="PLATFORM_ERROR"（模拟平台报错）。
    给了 gateway 时成功操作会同步修改演示世界里的平台状态。
    """

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
        self.clock = clock or FakeClock()
        self.fail_next = False
        self.calls: list[dict[str, Any]] = []
        self._done: dict[str, ActionResult] = {}

    def _do(
        self,
        action: AfterSaleAction,
        shop_id: str,
        aftersale_id: str,
        key: str,
        confirmation: HumanConfirmation,
        **extra: Any,
    ) -> ActionResult:
        if action not in self.supported:
            raise ActionNotAllowed(f"{self.platform} 不支持 {action}")
        if (
            action == AfterSaleAction.REFUSE
            and self.platform == Platform.DOUYIN
            and not extra.get("reason_code")
        ):
            raise ActionNotAllowed("抖店拒绝售后必须传拒绝原因码")
        if key in self._done:
            return self._done[key].model_copy(update={"duplicate": True})
        current = None
        if self.gateway is not None:
            current = next(
                (
                    a
                    for a in self.gateway.world.aftersales
                    if a.shop_id == shop_id and a.aftersale_id == aftersale_id
                ),
                None,
            )
            if current is None:
                raise ActionNotAllowed(f"售后单 {aftersale_id} 不存在或不属于店铺 {shop_id}")
        self.calls.append(
            {
                "action": action,
                "shop_id": shop_id,
                "aftersale_id": aftersale_id,
                "key": key,
                "operator": confirmation.operator,
                **extra,
            }
        )
        base = {"action": action, "aftersale_id": aftersale_id, "idempotency_key": key}
        if self.fail_next:
            self.fail_next = False
            return ActionResult(
                ok=False, platform_code="PLATFORM_ERROR", message="平台报错（模拟）", **base
            )
        if current is not None and (
            current.status not in ACTIONABLE_STATUSES[action]
            or current.status_version != confirmation.expected_status_version
        ):
            return ActionResult(
                ok=False, platform_code="STATUS_CHANGED", message="售后单状态已变化", **base
            )
        if self.gateway is not None:
            self.gateway.set_aftersale_status(
                shop_id, aftersale_id, _NEXT_STATUS[action], at=self.clock.now()
            )
        result = ActionResult(ok=True, platform_code="0", **base)
        self._done[key] = result
        return result

    def agree_refund(
        self,
        shop_id: str,
        aftersale_id: str,
        *,
        confirmation: HumanConfirmation,
        idempotency_key: str,
    ) -> ActionResult:
        return self._do(
            AfterSaleAction.AGREE_REFUND, shop_id, aftersale_id, idempotency_key, confirmation
        )

    def agree_return(
        self,
        shop_id: str,
        aftersale_id: str,
        *,
        confirmation: HumanConfirmation,
        idempotency_key: str,
    ) -> ActionResult:
        return self._do(
            AfterSaleAction.AGREE_RETURN, shop_id, aftersale_id, idempotency_key, confirmation
        )

    def refuse(
        self,
        shop_id: str,
        aftersale_id: str,
        *,
        confirmation: HumanConfirmation,
        idempotency_key: str,
        reason_code: str | None = None,
        reason_text: str | None = None,
    ) -> ActionResult:
        return self._do(
            AfterSaleAction.REFUSE,
            shop_id,
            aftersale_id,
            idempotency_key,
            confirmation,
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
    """构造一条标准买家消息（§8.3）。平台默认取演示世界里店铺的平台，时间默认 DEFAULT_NOW。"""
    if platform is None:
        platform = world.shop(shop_id).platform if world else Platform.TAOBAO
    if at is None:
        at = world.now if world else DEFAULT_NOW
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
    engine_config: EngineConfig | None = None,
    config_update: Mapping[str, Any] | None = None,
) -> ConversationContext:
    """直接构造"会话理解之后"的上下文，供路由、检索、生成、质检的单元测试使用。

    config 是生效配置：engine_config（默认 EngineConfig()）的分层 + 店铺覆盖项，
    再叠加 config_update（测试里临时改配置用）。
    """
    shop = world.shop(shop_id)
    now = world.now
    config = effective_shop_config(engine_config or EngineConfig(), shop)
    if config_update:
        config = config.model_validate({**config.model_dump(), **dict(config_update)})
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
            key=lambda o: (o.created_at, o.order_id),
            reverse=True,
        )[:5]
    return ConversationContext(
        platform=shop.platform,
        shop=shop,
        config=config,
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


def confirmation_for(
    info: AfterSaleInfo, *, operator: str = AGENT, at: datetime = DEFAULT_NOW, manager: bool = False
) -> HumanConfirmation:
    """按客服看到的售后单构造确认（测试用）。"""
    return HumanConfirmation(
        operator=operator,
        confirmed_at=at,
        expected_status_version=info.status_version,
        by_manager=manager,
    )
