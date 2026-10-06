"""演示站的过程记录：把一条买家消息在"找、写、判"三步里的中间结果记下来，给页面画
"AI 是怎么想的"（方案 §4.1 原则 2 懂找写判分开、原则 6 一切留痕）。

审计和判断日志按设计不放回答原文和依据内容（§12.3 日志脱敏），页面要展示的依据内容、草稿文字、
每份草稿的质检结果、买家图片的识别结果只能在处理当时拿到。所以在单元组装好之后，给单元里的路由、
检索、生成、质检四个组件（Unit.router / retriever / generator / gate，和引擎用的是同一个对象）的
route / retrieve / generate / check 套一层：照原样调用、照原样返回（异常照原样抛），顺手把输入
输出记进当前的 Recording。路由只记它看到的图片识别结果（ctx.image_findings，已打码）和接在买家消息
后面的图片描述（§15 Phase 3），不记买家原文。
不改业务代码，也不改变引擎的任何行为；没有正在记录的消息时（如历史数据回放）什么都不记。

只在演示站里用。记下来的东西里有买家原文所在的上下文对象，不往外传：trace.py 只取需要展示的字段。
"""

from __future__ import annotations

import functools
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from rodit.core.enums import Intent
from rodit.core.models import Draft, EvidenceBundle, ImageFinding, QAResult

_IMAGE_DESCRIPTION = re.compile(r"【买家发来[^】]*】$")
"""引擎接在买家消息后面的图片描述（engine.images.describe 的写法）。"""


@dataclass
class Retrieval:
    intent: Intent
    bundle: EvidenceBundle


@dataclass
class Generation:
    intent: Intent
    retry_feedback: list[str]
    drafts: list[Draft]


@dataclass
class Check:
    intent: Intent
    draft: Draft
    result: QAResult


@dataclass
class Routed:
    """路由看到的买家图片：识别结果（按图片顺序，已打码）和接在买家消息后面的描述。"""

    image_findings: list[ImageFinding]
    image_description: str | None


@dataclass
class Recording:
    """一条买家消息的中间结果（按发生顺序；分条并行时各条交错，按 intent 区分）。"""

    routed: list[Routed] = field(default_factory=list)
    retrievals: list[Retrieval] = field(default_factory=list)
    generations: list[Generation] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)


class Recorder:
    """套在一个单元的检索、生成、质检组件上（见模块说明）。线程安全：分条并行时多个线程同时写。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Recording | None = None

    @contextmanager
    def recording(self) -> Iterator[Recording]:
        """在这个范围里处理的消息都记进返回的 Recording。调用方保证同一时间只处理一条
        （演示站按平台串行处理）。"""
        rec = Recording()
        with self._lock:
            self._current = rec
        try:
            yield rec
        finally:
            with self._lock:
                self._current = None

    def _add(self, kind: str, item: Any) -> None:
        with self._lock:
            if self._current is not None:
                getattr(self._current, kind).append(item)

    def attach(self, unit: Any) -> None:
        """给单元的 router / retriever / generator / gate 套上记录（只套一次）。"""
        if getattr(unit, "_showcase_recorder", None) is not None:
            raise RuntimeError("这个单元已经套过过程记录")
        router, retriever, generator, gate = unit.router, unit.retriever, unit.generator, unit.gate

        def on_route(result: Any, ctx: Any, *args: Any, **kwargs: Any) -> None:
            text = getattr(getattr(ctx, "current", None), "text", "") or ""
            found = _IMAGE_DESCRIPTION.search(text)
            self._add(
                "routed",
                Routed(
                    list(getattr(ctx, "image_findings", []) or []),
                    found.group(0) if found else None,
                ),
            )

        def on_retrieve(result: EvidenceBundle, ctx: Any, route: Any) -> None:
            self._add("retrievals", Retrieval(route.intent, result))

        def on_generate(
            result: list[Draft],
            ctx: Any,
            route: Any,
            evidence: Any,
            *,
            retry_feedback: list[str] | None = None,
        ) -> None:
            self._add(
                "generations", Generation(route.intent, list(retry_feedback or []), list(result))
            )

        def on_check(result: QAResult, ctx: Any, route: Any, draft: Draft, evidence: Any) -> None:
            self._add("checks", Check(route.intent, draft, result))

        _wrap(router, "route", on_route)
        _wrap(retriever, "retrieve", on_retrieve)
        _wrap(generator, "generate", on_generate)
        _wrap(gate, "check", on_check)
        unit._showcase_recorder = self


def _wrap(obj: Any, name: str, observe: Callable[..., None]) -> None:
    original = getattr(obj, name)

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        observe(result, *args, **kwargs)
        return result

    setattr(obj, name, wrapped)


__all__ = ["Check", "Generation", "Recorder", "Recording", "Retrieval", "Routed"]
