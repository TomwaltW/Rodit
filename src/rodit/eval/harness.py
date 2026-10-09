"""评测用的平台单元（方案 §13）：离线演示单元，或接真实大模型的演示单元。

每条用例一套全新的单元（仓储、缓存、演示世界互不影响），都在合成演示世界上跑（rodit.testing.fixtures）。

- demo_harness()：离线替身（rodit.app.demo.build_demo_unit 不传 llm）——生成只复述依据、
  路由用关键词规则、质检用启发式判断。不联网，是默认模式和回归基线。
- llm_harness(client)：同一个演示单元，大模型换成 client（真实的 OpenAI 兼容接口，或测试里的
  FakeLLMClient）。接线和 build_demo_unit 传入 llm 时一样：路由、质检按 config.decision 组装
  （默认 incloud_llm：路由自一致性投票 3 次、质检 1 次，§6.4），生成走 DefaultAnswerGenerator。
  client 外面包三层（都不改变调用结果）：
  - DraftRecorder：记下生成请求的输出（模型草稿），失败清单里给人看；
  - CallRecorder：记下每次调用的结果——失败、被截断、空、读不出（只有计数，不留提示词和输出）。
    有响应不等于用得上：路由少了一票、生成解析不出回答，链路都不报错也不降级，只有在这里看得到，
    报告的运行健康度（rodit.eval.health）靠它；
  - MeteredLLMClient：每次调用写一条 llm.usage 审计（token、耗时、失败类型，§12.4、§14），
    写进本用例自己的审计（harness.usage），不和别的用例混在一起。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace

from rodit.app.demo import build_demo_unit
from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import EngineConfig
from rodit.core.decision import DecisionProvider
from rodit.core.enums import Platform
from rodit.core.errors import LLMError
from rodit.core.pii import mask_pii
from rodit.core.ports import AuditLog, ConversationRepository, LLMClient, LLMRequest, LLMResponse
from rodit.core.questions import QA_QUESTIONS, ROUTER_QUESTIONS, ROUTER_VOTE_QUESTIONS
from rodit.decision.voting import parse_sample
from rodit.eval.health import json_object_count
from rodit.eval.runner import (
    CAUSE_EMPTY,
    CAUSE_TRUNCATED,
    CAUSE_UNREADABLE,
    CallStats,
    EngineLike,
)
from rodit.generation import parse_generation_output
from rodit.llm import MeteredLLMClient
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import DemoWorld, build_demo_world

DEMO_PLATFORM = Platform.TAOBAO  # 评测集用店铺 A（淘宝）
UNPARSED_PREFIX = "[无法解析的输出] "
UNPARSED_MAX_CHARS = 200
FINISH_TRUNCATED = "length"  # LLMResponse.finish_reason：输出被 max_tokens 截断


class DraftRecorder:
    """实现 core.ports.LLMClient：原样转发给 inner，记下生成请求（purpose="generate"）的草稿。

    - 能解析成 §5.4 的 JSON 时记 answer；解析不了记原文前 200 字（加前缀标明），便于排查提示词；
    - 草稿一律先过 mask_pii：提示词本来已经脱敏（§5.3），这里再兜一层，报告里不出现手机号、地址；
    - 调用失败（LLMError）不记，失败由 MeteredLLMClient 记在 llm.usage 里。
    线程安全（路由投票会并发调用同一个客户端）。
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._drafts: list[str] = []

    @property
    def model(self) -> str:
        return str(getattr(self._inner, "model", "") or "")

    def __repr__(self) -> str:
        return f"DraftRecorder(inner={self._inner!r})"

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._inner.complete(request)
        if request.purpose == "generate":
            self._record(response.text)
        return response

    def _record(self, text: str) -> None:
        try:
            draft = parse_generation_output(text).answer
        except LLMError:
            draft = UNPARSED_PREFIX + text.strip()[:UNPARSED_MAX_CHARS]
        with self._lock:
            self._drafts.append(mask_pii(draft))

    def drafts(self) -> list[str]:
        with self._lock:
            return list(self._drafts)


def _readable(purpose: str, text: str) -> bool:
    """这个环节的解析器读不读得出东西：路由要每个必答的问题（ROUTER_QUESTIONS：类型、要不要转
    人工、情绪、是不是多个问题）都读得出——判断层每个问题都要有效票，哪一个不够都会退回关键词规则，
    和连通性检查（llm_env._vote_problem）一条规则；追问（second_intent）答没答不看。质检五个问题都
    要读得出（少一个，判断层就答不全）；生成要解析得出回答（§5.4 的 JSON）。别的用途不判断。
    路由、质检、生成的回答里顶层的 JSON 对象不止一个（比如先写了一份草稿、后面才是最终答案）时一律
    算读不出：解析器取到的未必是模型最后给的那个（health.json_object_count）。"""
    if purpose in ("route", "qa", "generate") and json_object_count(text) > 1:
        return False
    if purpose == "route":
        votes = parse_sample(ROUTER_VOTE_QUESTIONS, text)
        return all(key in votes for key in ROUTER_QUESTIONS)
    if purpose == "qa":
        return len(parse_sample(QA_QUESTIONS, text)) == len(QA_QUESTIONS)
    if purpose == "generate":
        try:
            parse_generation_output(text)
        except LLMError:
            return False
    return True


def _worth(purpose: str, response: LLMResponse) -> tuple[dict[str, int], str | None]:
    """一次有响应的调用记哪几项（runner.CallStats 的字段），和用不上时的原因（只记一种：截断 >
    空回答 > 读不出；用得上时为 None）。finish_reason 用 getattr 读：别的 LLMClient 实现返回的对象
    不一定有这个字段。"""
    truncated = getattr(response, "finish_reason", None) == FINISH_TRUNCATED
    empty = not response.text.strip()
    usable = not empty and _readable(purpose, response.text)
    if purpose in ("route", "qa"):  # 判断层的一票：被截断的不算数，哪怕碰巧读得出
        usable = usable and not truncated
    counts = {
        "truncated": int(truncated),
        "empty": int(empty),
        "usable": int(usable),
        "unusable": int(not usable),
    }
    if usable:
        return counts, None
    return counts, CAUSE_TRUNCATED if truncated else CAUSE_EMPTY if empty else CAUSE_UNREADABLE


class CallRecorder:
    """实现 core.ports.LLMClient：原样转发给 inner，按用途记下每次调用的结果（runner.CallStats）。

    - 调用抛 LLMError / LLMTimeout：failed（别的异常是代码缺陷，照常抛出，只记进 calls）；
    - 有响应：finish_reason 是 length 记 truncated，文本为空记 empty（两样可以同时有）；
      再看用不用得上——路由、质检的一票被截断、为空或读不出（_readable）都算作废，生成只看解析不
      解析得出回答——用得上记 usable，否则记 unusable，另按原因记一笔（unusable_causes：
      截断 / 空回答 / 读不出，一次调用只记一种；连续失败提前停下时报给操作的人）。
    只有计数，不留提示词和输出。线程安全（路由投票会并发调用同一个客户端）。
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._stats: dict[str, CallStats] = {}
        self._causes: dict[str, int] = {}

    @property
    def model(self) -> str:
        return str(getattr(self._inner, "model", "") or "")

    def __repr__(self) -> str:
        return f"CallRecorder(inner={self._inner!r})"

    def complete(self, request: LLMRequest) -> LLMResponse:
        purpose = request.purpose
        self._add(purpose, calls=1)
        try:
            response = self._inner.complete(request)
        except LLMError:
            self._add(purpose, failed=1)
            raise
        try:
            counts, cause = _worth(purpose, response)
        except Exception:  # noqa: BLE001 — 记录不能改变调用结果：判断不了就当用不上
            counts, cause = {"unusable": 1}, CAUSE_UNREADABLE
        self._add(purpose, **counts)
        if cause is not None:
            with self._lock:
                self._causes[cause] = self._causes.get(cause, 0) + 1
        return response

    def _add(self, purpose: str, **counts: int) -> None:
        with self._lock:
            now = self._stats.get(purpose, CallStats())
            changes = {name: getattr(now, name) + n for name, n in counts.items()}
            self._stats[purpose] = replace(now, **changes)

    def stats(self) -> dict[str, CallStats]:
        with self._lock:
            return dict(self._stats)

    def unusable_causes(self) -> dict[str, int]:
        """有响应但用不上的调用，原因 → 次数（各用途合计）。"""
        with self._lock:
            return dict(self._causes)


@dataclass
class DemoHarness:
    """runner.EvalHarness 的实现：一个全新的演示单元（每条用例一套，互不影响）。

    - audit：单元自己的审计，读 engine.result（质检原因等）和 engine.qa（判断层判过几份草稿）；
    - usage：只有接大模型时有，本用例的 llm.usage 事件；
    - drafts：只有接大模型时有，本用例的模型草稿；
    - calls：只有接大模型时有，本用例每个用途的调用结果。
    """

    engine: EngineLike
    conversations: ConversationRepository
    world: DemoWorld
    audit: AuditLog | None = None
    usage: InMemoryAuditSink | None = None
    drafts: DraftRecorder | None = None
    calls: CallRecorder | None = None


def demo_harness(
    config: EngineConfig | None = None, route_provider: DecisionProvider | None = None
) -> DemoHarness:
    """离线替身单元（见模块说明）。

    config 里的 decision.provider 会被换成 rules（build_demo_unit 的离线接线）；route_provider
    给了就用它路由（如训练好的分类器，§6.4），质检照旧。"""
    world = build_demo_world()
    unit = build_demo_unit(  # 淘宝单元：店铺 A、B
        DEMO_PLATFORM, world=world, config=config, route_provider=route_provider
    )
    return DemoHarness(
        engine=unit.engine,
        conversations=unit.repos.conversations,
        world=world,
        audit=unit.repos.audit,
    )


def llm_harness(
    client: LLMClient,
    config: EngineConfig | None = None,
    route_provider: DecisionProvider | None = None,
) -> DemoHarness:
    """接大模型的单元（见模块说明）。client 在各用例之间共用，计量、草稿、调用结果按用例分开。
    route_provider 给了就用它路由（如训练好的分类器），生成和质检照旧用大模型。"""
    world = build_demo_world()
    usage = InMemoryAuditSink()
    recorder = DraftRecorder(client)
    calls = CallRecorder(recorder)
    metered = MeteredLLMClient(calls, usage, FakeClock(world.now))
    unit = build_demo_unit(
        DEMO_PLATFORM, world=world, config=config, llm=metered, route_provider=route_provider
    )
    return DemoHarness(
        engine=unit.engine,
        conversations=unit.repos.conversations,
        world=world,
        audit=unit.repos.audit,
        usage=usage,
        drafts=recorder,
        calls=calls,
    )


__all__ = [
    "DEMO_PLATFORM",
    "CallRecorder",
    "DemoHarness",
    "DraftRecorder",
    "demo_harness",
    "llm_harness",
]
