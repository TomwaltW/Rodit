"""演示站接真实大模型（--llm env）时的整条链路：判断层、调用记录、时间预算、页面上的说明。

- 判断层（--judge llm）：路由、质检照生产的组装方式来——config.decision 交给
  rodit.decision.build_decision_provider（purpose=route / qa；默认 incloud_llm：路由自一致性投票
  vote_samples 次、质检 qa_vote_samples 次，§6.4），Router / QualityGate 的买家数据检查不变
  （require_buyer_data_ok）。--judge offline 时路由用关键词规则、质检用启发式判断（离线替身）。
- 回放历史（组装演示单元时的两百多条合成消息）一律离线：ReplaySwitch 打开期间，
  SwitchableLLMProvider 把判断交给离线替身（路由：关键词规则；质检：启发式判断），
  ShowcaseLLM 的生成也走离线写法，ObservedLLM 拒绝任何真实调用并计数（双保险，计数应为 0）。
  回放只在组装时进行，那时演示单元还没有对外，切换用锁保护，任何线程读到的都是同一个状态。
- SwitchableLLMProvider 是 InCloudLLMProvider 的子类：路由器按类型认出"投票制大模型判断层"
  （一并问追问 second_intent、按票数粒度比较阈值，rodit.routing.router），回放期间 name /
  samples 报离线替身的（判断日志记的是实际回答的实现，阈值不按票数换算），追问由这里答"无"
  （和离线替身不问追问的结果一样）。accepts_buyer_data = 真实实现和离线替身都允许才允许。
- 每次判断留一条 JudgeRecord（每票解析出来的答案、失败原因、耗时、token），每次大模型调用留
  一条 LLMCall（用途、成败、耗时、token），交给过程记录（showcase.recorder），页面上画
  "大模型投票几票、每票选了什么、哪几票超时"，自检报告按用途汇总调用次数和 token。
  记录里没有提示词和输出原文，失败原因只有分类后的短说明（showcase.preflight.short_label）。
- 时间预算（--latency）：strict 照配置（默认就是 §12.1：路由单次 1 秒、整条消息 4.5 秒）；
  relaxed 放宽到演示友好的值（RELAXED_*，见 relax_latency），页面和启动横幅都写明
  "演示放宽了时间预算"。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from rodit.core.config import EngineConfig, LLMRequestOptions
from rodit.core.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionProvider,
    DecisionRequest,
    DecisionResponse,
    validate_response,
)
from rodit.core.enums import ProviderName
from rodit.core.errors import DecisionError, LLMError
from rodit.core.ports import LLMClient, LLMRequest, LLMResponse
from rodit.core.questions import NO_SECOND_INTENT, Q_SECOND_INTENT
from rodit.decision import build_decision_provider
from rodit.decision.llm import InCloudLLMProvider, logprob_target
from rodit.decision.voting import choice_distribution, parse_sample
from rodit.eval.llm_env import client_options, describe_endpoint
from rodit.showcase.preflight import short_label

JudgeMode = Literal["llm", "offline"]
LatencyMode = Literal["strict", "relaxed"]

RELAXED_DECISION_TIMEOUT_S = 8.0
"""放宽后判断层（路由每一票、质检）单次调用的超时（生产 1 秒，§12.1）。"""
RELAXED_GENERATION_TIMEOUT_S = 15.0
"""放宽后生成单次调用的超时（生产 2.5 秒，§12.1）。"""
RELAXED_MESSAGE_BUDGET_S = 30.0
"""放宽后每条消息的时间预算（生产 4.5 秒；LatencyConfig 允许的上限是 30 秒）。"""
RELAXED_MIN_LLM_REMAINING_S = 3.0
"""放宽后剩余时间少于这个值就不再调大模型（生产 0.8 秒）：托管模型一次调用常要两三秒。"""

RELAXED_NOTE = "演示放宽了时间预算（生产按 §12.1：路由 ≤ 1 秒、端到端 P95 ≤ 5 秒）"
"""放宽时间预算时页面、横幅、自检报告上的说明。"""

STRICT_WARNING = (
    "判断层用大模型投票、时间预算按生产（§12.1：路由每一票限时 {route:g} 秒、整条消息 "
    "{budget:g} 秒）：大多数托管大模型 1 秒内答不完，很多消息的路由会超时、退回关键词规则"
    "（页面上标为降级）。演示时可以加 --latency relaxed"
)

JUDGE_LABELS: dict[str, str] = {
    "llm": "大模型投票",
    "llm_qa_only": "关键词路由 + 大模型质检",
    "offline": "离线规则",
}
"""判断层的说明。--judge llm 但配置的 decision.provider 是 rules 时，路由用关键词规则、
质检仍用大模型（rodit.decision.build_decision_provider：质检只有大模型能答，§6.4）。"""

JUDGE_SENTENCES: dict[str, str] = {
    "llm": "路由、质检用大模型投票",
    "llm_qa_only": "路由用关键词规则、质检用大模型",
    "offline": "路由、质检仍是离线替身",
}
LATENCY_LABELS: dict[str, str] = {"strict": "严格", "relaxed": "放宽"}


def relax_latency(config: EngineConfig) -> EngineConfig:
    """--latency relaxed：判断层、生成的单次超时和每条消息的预算放宽到 RELAXED_*（配置里本来
    更宽的保持不变）。阈值、投票次数等其他配置不动。"""
    decision = config.decision.model_copy(
        update={"timeout_s": max(config.decision.timeout_s, RELAXED_DECISION_TIMEOUT_S)}
    )
    generation = config.generation.model_copy(
        update={"timeout_s": max(config.generation.timeout_s, RELAXED_GENERATION_TIMEOUT_S)}
    )
    latency = config.latency.model_copy(
        update={
            "message_budget_s": max(config.latency.message_budget_s, RELAXED_MESSAGE_BUDGET_S),
            "min_llm_remaining_s": max(
                config.latency.min_llm_remaining_s, RELAXED_MIN_LLM_REMAINING_S
            ),
        }
    )
    return config.model_copy(
        update={"decision": decision, "generation": generation, "latency": latency}
    )


def offline_judges(config: EngineConfig) -> EngineConfig:
    """--judge offline：判断层换成关键词规则（和 rodit.app.demo 的离线演示一致）。"""
    return config.model_copy(
        update={"decision": config.decision.model_copy(update={"provider": ProviderName.RULES})}
    )


def chain_config(
    base: EngineConfig | None, *, judge: JudgeMode, latency: LatencyMode
) -> EngineConfig:
    """演示单元用的配置：--config 读进来的（没有就是默认值），按 --judge / --latency 调整。"""
    config = base or EngineConfig()
    if judge == "offline":
        config = offline_judges(config)
    if latency == "relaxed":
        config = relax_latency(config)
    return config


SUPPORTED_JUDGE_PROVIDERS = frozenset({ProviderName.INCLOUD_LLM, ProviderName.RULES})
"""--judge llm 支持的 decision.provider：分类器要模型文件、Jev 不能处理买家数据（§6.2），
演示站都不接。"""


def check_judge_config(config: EngineConfig) -> None:
    """--judge llm 前检查配置：不支持的 decision.provider 抛 ValueError（启动时报配置错误）。"""
    provider = config.decision.provider
    if provider not in SUPPORTED_JUDGE_PROVIDERS:
        raise ValueError(
            f"演示站的 --judge llm 只支持 decision.provider = incloud_llm 或 rules，配置里是 "
            f"{provider.value}（分类器要模型文件、Jev 不能处理买家数据，演示站都不接）"
        )


# ================================================================ 回放开关


class ReplaySwitch:
    """回放历史期间打开（可以嵌套）。线程安全：读写都在锁里（见模块说明）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0

    @property
    def on(self) -> bool:
        with self._lock:
            return self._depth > 0

    @contextmanager
    def replaying(self) -> Iterator[None]:
        with self._lock:
            self._depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1


# ================================================================ 调用记录


@dataclass(frozen=True)
class LLMCall:
    """一次真实大模型调用（没有提示词和输出原文）。error 是分类后的短说明，成功时为 None。"""

    purpose: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    timed_out: bool = False


REPLAY_BLOCKED = "回放历史期间不调用真实大模型"


class ObservedLLM:
    """实现 core.ports.LLMClient：转发给真实客户端，每次调用留一条 LLMCall。

    sink_for 在每次调用开始时给出这次调用的记录去处（过程记录绑定到当时正在处理的那条消息）：
    到截止时间还没返回、在后台跑完的调用（§12.2 生成不等）不会记到下一条消息上。
    回放开关打开时拒绝调用（抛 LLMError，计入 blocked_during_replay）：回放应该一次都调不到
    这里（见模块说明），计数是给测试和自检报告核对的。线程安全（投票会并发调用）。"""

    def __init__(
        self,
        inner: LLMClient,
        switch: ReplaySwitch,
        sink_for: Callable[[], Callable[[LLMCall], None] | None] | None = None,
    ) -> None:
        self.inner = inner
        self.switch = switch
        self.sink_for = sink_for
        self._lock = threading.Lock()
        self.calls = 0
        self.blocked_during_replay = 0

    @property
    def model(self) -> str:
        return str(getattr(self.inner, "model", "") or "")

    @property
    def base_url(self) -> str | None:
        return getattr(self.inner, "base_url", None)

    @property
    def options(self) -> LLMRequestOptions:
        """inner 生效的请求参数（OpenAICompatibleClient.options：JSON 模式、logprobs 开关等）。"""
        return client_options(self.inner)

    def __repr__(self) -> str:
        return f"ObservedLLM(inner={self.inner!r})"

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.switch.on:
            with self._lock:
                self.blocked_during_replay += 1
            raise LLMError(REPLAY_BLOCKED)
        with self._lock:
            self.calls += 1
        sink = self.sink_for() if self.sink_for is not None else None
        started = time.perf_counter()
        try:
            response = self.inner.complete(request)
        except LLMError as exc:
            if sink is not None:
                label = short_label(exc, timeout_s=request.timeout_s)
                sink(
                    LLMCall(
                        purpose=request.purpose,
                        latency_ms=_ms_since(started),
                        error=label,
                        timed_out=label.startswith("超时"),
                    )
                )
            raise
        if sink is not None:
            sink(
                LLMCall(
                    purpose=request.purpose,
                    latency_ms=_ms_since(started),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
            )
        return response


def _ms_since(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


# ================================================================ 判断层


@dataclass
class JudgeRecord:
    """一次大模型判断（路由或质检）：每票的答案、失败原因、汇总（见模块说明）。

    - samples / min_votes / timeout_s：采样次数、每题至少几张有效票、单次超时；
    - votes：每次成功返回的采样解析出来的答案（{问题: 票}，解析不了的问题不在里面）；
    - errors：失败的采样（分类后的短说明，如"超时（限时 1 秒）""限流或额度用完（HTTP 429）"）；
    - failed：整次判断不可用的原因（DecisionError 的说明），路由据此退回关键词规则、质检判为不可用；
    - logprobs：有没有用上接口给的选项概率（没有时概率按票数占比算）。"""

    purpose: str
    provider: str
    samples: int
    min_votes: int
    timeout_s: float | None
    votes: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    failed: str | None = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    logprobs: bool = False
    answers: dict[str, Any] = field(default_factory=dict)

    @property
    def all_timed_out(self) -> bool:
        return (
            bool(self.errors) and not self.votes and all(e.startswith("超时") for e in self.errors)
        )


class SwitchableLLMProvider(InCloudLLMProvider):
    """回放期间离线、之后用真实大模型投票的判断层（见模块说明）。

    real：按配置组装好的 InCloudLLMProvider（真实客户端）；offline：回放期间用的离线替身；
    observer：每次真实判断后收到一条 JudgeRecord（过程记录用）。"""

    def __init__(
        self,
        real: InCloudLLMProvider,
        offline: DecisionProvider,
        switch: ReplaySwitch,
        observer: Callable[[JudgeRecord], None] | None = None,
    ) -> None:
        self._real_name = real.name
        self._real_samples = real.samples
        self.offline = offline
        self.switch = switch
        self.observer = observer
        self._local = threading.local()
        super().__init__(
            real.llm,
            samples=real.samples,
            temperature=real.temperature,
            purpose=real.purpose,
            timeout_s=real.timeout_s,
            name=real.name,
            min_votes=real.min_votes,
            max_tokens=real.max_tokens,
        )
        # 两边都允许处理买家数据才允许（Router / QualityGate 构造时据此检查，§6.2、§6.3）
        self.accepts_buyer_data = bool(getattr(real, "accepts_buyer_data", False)) and bool(
            getattr(offline, "accepts_buyer_data", False)
        )

    # name / samples：回放期间报离线替身的（见模块说明）
    @property  # type: ignore[override]
    def name(self) -> str:
        return str(getattr(self.offline, "name", "offline")) if self.switch.on else self._real_name

    @name.setter
    def name(self, value: str) -> None:
        self._real_name = value

    @property  # type: ignore[override]
    def samples(self) -> int:
        return 1 if self.switch.on else self._real_samples

    @samples.setter
    def samples(self, value: int) -> None:
        self._real_samples = value

    # ---------------------------------------------------------------- 判断

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        if self.switch.on:
            return self._decide_offline(request)
        self._local.results = None
        started = time.perf_counter()
        try:
            response = super().decide(request)
        except DecisionError as exc:
            self._report(request, None, started, failed=str(exc))
            raise
        self._report(request, response, started, failed=None)
        return response

    def _sample(self, llm_request: LLMRequest) -> list[LLMResponse | LLMError]:
        results = super()._sample(llm_request)
        self._local.results = (llm_request, results)
        return results

    def _decide_offline(self, request: DecisionRequest) -> DecisionResponse:
        """离线替身只答它认识的问题；追问 second_intent 答"无"（离线替身本来不问追问）。"""
        asked = {k: q for k, q in request.questions.items() if k != Q_SECOND_INTENT}
        response = self.offline.decide(request.model_copy(update={"questions": asked}))
        answers = dict(response.answers)
        if Q_SECOND_INTENT in request.questions:
            answers[Q_SECOND_INTENT] = ChoiceAnswer(
                choice=NO_SECOND_INTENT, confidence=1.0, probabilities={NO_SECOND_INTENT: 1.0}
            )
        out = DecisionResponse(model=response.model, answers=answers)
        validate_response(request, out)
        return out

    def _report(
        self,
        request: DecisionRequest,
        response: DecisionResponse | None,
        started: float,
        *,
        failed: str | None,
    ) -> None:
        if self.observer is None:
            return
        sampled = getattr(self._local, "results", None)
        self._local.results = None
        llm_request, results = sampled if sampled is not None else (None, [])
        timeout_s = llm_request.timeout_s if llm_request is not None else self.timeout_s
        record = JudgeRecord(
            purpose=str(self.purpose),
            provider=self._real_name,
            samples=self._real_samples,
            min_votes=self.min_votes,
            timeout_s=timeout_s,
            failed=failed,
            latency_ms=_ms_since(started),
            answers=(
                {k: a.model_dump(mode="json") for k, a in response.answers.items()}
                if response is not None
                else {}
            ),
        )
        target = logprob_target(request.questions)
        for r in results:
            if isinstance(r, LLMError):
                record.errors.append(short_label(r, timeout_s=timeout_s))
                continue
            record.input_tokens += r.input_tokens
            record.output_tokens += r.output_tokens
            record.votes.append(parse_sample(request.questions, r.text))
            question = request.questions.get(target) if target else None
            if (
                isinstance(question, ChoiceQuestion)
                and choice_distribution(question, r.logprobs) is not None
            ):
                record.logprobs = True
        try:
            self.observer(record)
        except Exception:  # noqa: BLE001 — 过程记录出错不能影响判断结果
            pass


def build_judges(
    config: EngineConfig,
    llm: LLMClient,
    switch: ReplaySwitch,
    *,
    offline_route: DecisionProvider,
    offline_qa: DecisionProvider,
    observer: Callable[[JudgeRecord], None] | None = None,
) -> tuple[DecisionProvider, DecisionProvider]:
    """--judge llm 的路由、质检判断层：照生产组装（build_decision_provider），大模型判断层包成
    回放期间离线的 SwitchableLLMProvider；不调大模型的实现（关键词规则）原样用。"""
    check_judge_config(config)

    def wrap(real: DecisionProvider, offline: DecisionProvider) -> DecisionProvider:
        if isinstance(real, InCloudLLMProvider):
            return SwitchableLLMProvider(real, offline, switch, observer)
        return real

    route = build_decision_provider(config.decision, purpose="route", llm=llm)
    qa = build_decision_provider(config.decision, purpose="qa", llm=llm)
    return wrap(route, offline_route), wrap(qa, offline_qa)


# ================================================================ 页面、横幅上的说明


@dataclass(frozen=True)
class ChainInfo:
    """这次演示用的是什么：生成（真实大模型 / 离线替身）、判断层、时间预算。"""

    real: bool
    model: str | None
    host: str | None
    judge: JudgeMode
    latency: LatencyMode
    route_timeout_s: float
    message_budget_s: float
    generation_timeout_s: float
    vote_samples: int
    qa_vote_samples: int
    route_by_llm: bool = (
        True  # --judge llm 时路由是不是大模型投票（配置的 provider 是 rules 就不是）
    )
    json_mode: bool = True  # 请求带不带 response_format（RODIT_LLM_JSON_MODE / llm.json_mode）
    logprobs: bool = True  # 路由请求带不带 logprobs（RODIT_LLM_LOGPROBS / llm.logprobs）
    extra_body_keys: tuple[str, ...] = ()  # 模型服务特有参数的键名（RODIT_LLM_EXTRA_BODY）

    @property
    def relaxed(self) -> bool:
        return self.latency == "relaxed"

    @property
    def judge_llm(self) -> bool:
        return self.judge == "llm"

    @property
    def judge_key(self) -> str:
        if self.judge == "llm" and not self.route_by_llm:
            return "llm_qa_only"
        return self.judge

    @property
    def judge_label(self) -> str:
        return JUDGE_LABELS[self.judge_key]

    @property
    def judge_sentence(self) -> str:
        """一句话说判断层用的是什么（页脚、看板的耗时说明用）。"""
        return JUDGE_SENTENCES[self.judge_key]

    def badge(self) -> str:
        """页眉、页脚的徽标："真实大模型：<模型>（<主机>）· 判断层：… · 时间预算：…"。"""
        if not self.real:
            return "离线替身（不联网）· 判断层：离线规则"
        return (
            f"真实大模型：{self.model}（{self.host}）· 判断层：{self.judge_label}"
            f" · 时间预算：{LATENCY_LABELS[self.latency]}"
        )

    def footer(self) -> str:
        if not self.real:
            return "演示站使用合成数据和离线大模型替身；生产部署在各平台云内（方案 §9.1）"
        parts = [
            f"演示站使用合成数据；回答由真实大模型 {self.model} 生成",
            self.judge_sentence,
            "图片理解、平台接口仍是合成数据的替身",
            "生产部署在各平台云内（方案 §9.1）",
        ]
        if self.relaxed:
            parts.append(RELAXED_NOTE)
        return "；".join(parts)

    def view(self) -> dict[str, Any]:
        return {
            "real": self.real,
            "model": self.model,
            "host": self.host,
            "judge": self.judge,
            "judge_label": self.judge_label,
            "route_by_llm": self.real and self.judge_llm and self.route_by_llm,
            "latency": self.latency,
            "latency_label": LATENCY_LABELS[self.latency],
            "relaxed_note": RELAXED_NOTE if self.real and self.relaxed else None,
            "route_timeout_s": self.route_timeout_s,
            "message_budget_s": self.message_budget_s,
            "generation_timeout_s": self.generation_timeout_s,
            "vote_samples": self.vote_samples,
            "qa_vote_samples": self.qa_vote_samples,
            "request_options": {
                "json_mode": self.json_mode,
                "logprobs": self.logprobs,
                "extra_body_keys": list(self.extra_body_keys),
            }
            if self.real
            else None,
            "badge": self.badge(),
            "footer": self.footer(),
        }


def chain_info(
    llm: LLMClient | None, config: EngineConfig, *, judge: JudgeMode, latency: LatencyMode
) -> ChainInfo:
    options = client_options(llm)
    return ChainInfo(
        real=llm is not None,
        model=(str(getattr(llm, "model", "") or "") or None) if llm is not None else None,
        host=describe_endpoint(llm) if llm is not None else None,
        judge=judge,
        latency=latency,
        route_timeout_s=config.decision.timeout_s,
        message_budget_s=config.latency.message_budget_s,
        generation_timeout_s=config.generation.timeout_s,
        vote_samples=config.decision.vote_samples,
        qa_vote_samples=config.decision.qa_vote_samples,
        route_by_llm=config.decision.provider == ProviderName.INCLOUD_LLM,
        json_mode=options.json_mode,
        logprobs=options.logprobs,
        extra_body_keys=tuple(sorted(options.extra_body)),
    )


__all__ = [
    "JUDGE_LABELS",
    "LATENCY_LABELS",
    "RELAXED_DECISION_TIMEOUT_S",
    "RELAXED_GENERATION_TIMEOUT_S",
    "RELAXED_MESSAGE_BUDGET_S",
    "RELAXED_MIN_LLM_REMAINING_S",
    "RELAXED_NOTE",
    "REPLAY_BLOCKED",
    "STRICT_WARNING",
    "SUPPORTED_JUDGE_PROVIDERS",
    "ChainInfo",
    "JudgeMode",
    "JudgeRecord",
    "LLMCall",
    "LatencyMode",
    "ObservedLLM",
    "ReplaySwitch",
    "SwitchableLLMProvider",
    "build_judges",
    "chain_config",
    "chain_info",
    "check_judge_config",
    "offline_judges",
    "relax_latency",
]
