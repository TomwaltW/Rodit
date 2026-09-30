"""评测用的平台单元（方案 §13）：离线演示单元，或接真实大模型的演示单元。

每条用例一套全新的单元（仓储、缓存、演示世界互不影响），都在合成演示世界上跑（rodit.testing.fixtures）。

- demo_harness()：离线替身（rodit.app.demo.build_demo_unit 不传 llm）——生成只复述依据、
  路由用关键词规则、质检用启发式判断。不联网，是默认模式和回归基线。
- llm_harness(client)：同一个演示单元，大模型换成 client（真实的 OpenAI 兼容接口，或测试里的
  FakeLLMClient）。接线和 build_demo_unit 传入 llm 时一样：路由、质检按 config.decision 组装
  （默认 incloud_llm：路由自一致性投票 3 次、质检 1 次，§6.4），生成走 DefaultAnswerGenerator。
  client 外面包两层（都不改变调用结果）：
  - DraftRecorder：记下生成请求的输出（模型草稿），失败清单里给人看；
  - MeteredLLMClient：每次调用写一条 llm.usage 审计（token、耗时、失败类型，§12.4、§14），
    写进本用例自己的审计（harness.usage），不和别的用例混在一起。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from rodit.app.demo import build_demo_unit
from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import EngineConfig
from rodit.core.decision import DecisionProvider
from rodit.core.enums import Platform
from rodit.core.errors import LLMError
from rodit.core.pii import mask_pii
from rodit.core.ports import AuditLog, ConversationRepository, LLMClient, LLMRequest, LLMResponse
from rodit.eval.runner import EngineLike
from rodit.generation import parse_generation_output
from rodit.llm import MeteredLLMClient
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import DemoWorld, build_demo_world

DEMO_PLATFORM = Platform.TAOBAO  # 评测集用店铺 A（淘宝）
UNPARSED_PREFIX = "[无法解析的输出] "
UNPARSED_MAX_CHARS = 200


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


@dataclass
class DemoHarness:
    """runner.EvalHarness 的实现：一个全新的演示单元（每条用例一套，互不影响）。

    - audit：单元自己的审计，读 engine.result（质检原因等）；
    - usage：只有接大模型时有，本用例的 llm.usage 事件；
    - drafts：只有接大模型时有，本用例的模型草稿。
    """

    engine: EngineLike
    conversations: ConversationRepository
    world: DemoWorld
    audit: AuditLog | None = None
    usage: InMemoryAuditSink | None = None
    drafts: DraftRecorder | None = None


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
    """接大模型的单元（见模块说明）。client 在各用例之间共用，计量和草稿按用例分开。
    route_provider 给了就用它路由（如训练好的分类器），生成和质检照旧用大模型。"""
    world = build_demo_world()
    usage = InMemoryAuditSink()
    recorder = DraftRecorder(client)
    metered = MeteredLLMClient(recorder, usage, FakeClock(world.now))
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
    )


__all__ = ["DEMO_PLATFORM", "DemoHarness", "DraftRecorder", "demo_harness", "llm_harness"]
