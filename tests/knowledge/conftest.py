"""知识中心测试的共用夹具：演示世界（全部合成数据）、内存仓储、可观察的向量、可注入故障的仓储。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from rodit.core.config import GenerationConfig
from rodit.core.enums import Intent, RouteAction
from rodit.core.models import RouteDecision, RoutePart
from rodit.inmemory import HashingEmbedder, InMemoryKnowledgeRepository
from rodit.knowledge import FaqSearcher, KnowledgeRetriever
from rodit.testing.fakes import FakeClock
from rodit.testing.fixtures import DemoWorld, build_demo_world, seed_knowledge


class SpyEmbedder:
    """记录发给向量服务的全部文字；fail=True 时模拟向量服务故障。"""

    def __init__(self, dim: int = 256) -> None:
        self._inner = HashingEmbedder(dim)
        self.dim = dim
        self.texts: list[str] = []
        self.fail = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        if self.fail:
            raise RuntimeError("向量服务不可用（模拟）")
        return self._inner.embed(texts)


class FlakyRepo:
    """包一层内存仓储：fail_on 里的方法调用时抛错（模拟知识库故障）；calls 记录调用的方法名。"""

    def __init__(self, inner: InMemoryKnowledgeRepository) -> None:
        self._inner = inner
        self.fail_on: set[str] = set()
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            if name in self.fail_on or "*" in self.fail_on:
                raise ConnectionError(f"{name}：数据库连接断开（模拟）")
            return attr(*args, **kwargs)

        return wrapper


@pytest.fixture
def world() -> DemoWorld:
    return build_demo_world()


@pytest.fixture
def seeded(world: DemoWorld) -> tuple[InMemoryKnowledgeRepository, dict[tuple[str, str], int]]:
    repo = InMemoryKnowledgeRepository(FakeClock(world.now))
    ids = seed_knowledge(repo, world)
    return repo, ids


@pytest.fixture
def repo(seeded) -> InMemoryKnowledgeRepository:
    return seeded[0]


@pytest.fixture
def faq_ids(seeded) -> dict[tuple[str, str], int]:
    return seeded[1]


@pytest.fixture
def spy() -> SpyEmbedder:
    return SpyEmbedder()


@pytest.fixture
def flaky(repo) -> FlakyRepo:
    return FlakyRepo(repo)


@pytest.fixture
def make_retriever(repo, spy) -> Callable[..., KnowledgeRetriever]:
    """make_retriever(repo=..., config=..., tool_provider=...)：默认用种好数据的内存仓储。"""

    def make(
        *,
        repo_: Any = None,
        config: GenerationConfig | None = None,
        tool_provider: Any = None,
        embedder: Any = None,
    ) -> KnowledgeRetriever:
        r = repo_ if repo_ is not None else repo
        return KnowledgeRetriever(
            r,
            FaqSearcher(r, embedder if embedder is not None else spy),
            config or GenerationConfig(),
            tool_provider=tool_provider,
        )

    return make


def _route(intent: Intent, *parts: Intent, action: RouteAction | None = None) -> RouteDecision:
    return RouteDecision(
        action=action or (RouteAction.SPLIT_MULTI if parts else RouteAction.ANSWER),
        rule=5 if parts else 6,
        intent=intent,
        intent_confidence=0.9,
        parts=[RoutePart(intent=p, confidence=0.45) for p in parts],
        provider="test",
    )


@pytest.fixture
def route() -> Callable[..., RouteDecision]:
    """route(intent) 或 route(主 intent, part1, part2)（多问题 SPLIT_MULTI）。"""
    return _route
