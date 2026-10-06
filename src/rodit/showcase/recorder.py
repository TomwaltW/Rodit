"""演示站的过程记录：把一条买家消息在"找、写、判"三步里的中间结果记下来，给页面画
"AI 是怎么想的"（方案 §4.1 原则 2 懂找写判分开、原则 6 一切留痕）。

审计和判断日志按设计不放回答原文和依据内容（§12.3 日志脱敏），页面要展示的依据内容、草稿文字、
每份草稿的质检结果、买家图片的识别结果只能在处理当时拿到。所以在单元组装好之后，给单元里的路由、
检索、生成、质检四个组件（Unit.router / retriever / generator / gate，和引擎用的是同一个对象）的
route / retrieve / generate / check 套一层：照原样调用、照原样返回（异常照原样抛），顺手把输入
输出记进当前的 Recording。路由只记它看到的图片识别结果（ctx.image_findings，已打码）和接在买家消息
后面的图片描述（§15 Phase 3），不记买家原文。
不改业务代码，也不改变引擎的任何行为；没有正在记录的消息时（如历史数据回放）什么都不记。

接真实大模型时另记两样（showcase.realchain）：每次大模型判断的 JudgeRecord（每票的答案、失败原因），
挂到发起它的那次路由 / 质检上（判断在发起它的线程里做完，用线程局部变量对上）；每次真实大模型调用的
LLMCall（用途、成败、耗时、token；投票、分条生成在线程池里调用，按消息汇总，不分到哪一步）。
四个组件的调用、判断、大模型调用都在开始时绑定到当时正在处理的消息，结束时那条消息已经结束就不记：
到截止时间引擎不等了、在后台跑完的质检 / 生成 / 调用不会记到下一条消息上。

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
from rodit.showcase.realchain import JudgeRecord, LLMCall

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
    judge: JudgeRecord | None = None  # 这次质检的大模型判断（接真实大模型时）


@dataclass
class Routed:
    """路由看到的买家图片：识别结果（按图片顺序，已打码）和接在买家消息后面的描述；
    judge 是这次路由的大模型投票（接真实大模型时；判断层不调大模型时为 None）。"""

    image_findings: list[ImageFinding]
    image_description: str | None
    judge: JudgeRecord | None = None


@dataclass
class Recording:
    """一条买家消息的中间结果（按发生顺序；分条并行时各条交错，按 intent 区分）。"""

    routed: list[Routed] = field(default_factory=list)
    retrievals: list[Retrieval] = field(default_factory=list)
    generations: list[Generation] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    judges: list[JudgeRecord] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)


class Recorder:
    """套在一个单元的检索、生成、质检组件上（见模块说明）。线程安全：分条并行时多个线程同时写。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Recording | None = None
        self._local = threading.local()  # 当前线程最近一次大模型判断（挂到路由 / 质检上）

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

    def _open(self) -> Recording | None:
        with self._lock:
            return self._current

    def _add_to(self, recording: Recording | None, kind: str, item: Any) -> None:
        """记进调用开始时绑定的那条消息，而且只在它还开着时记：到截止时间引擎不等了、在后台做完的
        检索 / 生成 / 质检 / 判断，结果不会记到下一条消息上（那条消息的引擎根本没收到它）。"""
        with self._lock:
            if recording is not None and recording is self._current:
                getattr(recording, kind).append(item)

    def add_judge(self, record: JudgeRecord) -> None:
        """SwitchableLLMProvider 的 observer：记下判断，并留给本线程正在做的路由 / 质检。
        判断在发起它的路由 / 质检线程里做完，记到那次调用开始时绑定的消息上。"""
        self._local.judge = record
        bound = getattr(self._local, "recording", None)
        self._add_to(bound if bound is not None else self._open(), "judges", record)

    def llm_call_sink(self) -> Callable[[LLMCall], None] | None:
        """ObservedLLM 每次调用开始时取：绑定到这时正在记录的消息（没有时返回 None，不记）。"""
        with self._lock:
            recording = self._current
        if recording is None:
            return None

        def sink(call: LLMCall) -> None:
            with self._lock:
                recording.llm_calls.append(call)

        return sink

    def _take_judge(self) -> JudgeRecord | None:
        record = getattr(self._local, "judge", None)
        self._local.judge = None
        return record

    def attach(self, unit: Any) -> None:
        """给单元的 router / retriever / generator / gate 套上记录（只套一次）。"""
        if getattr(unit, "_showcase_recorder", None) is not None:
            raise RuntimeError("这个单元已经套过过程记录")
        router, retriever, generator, gate = unit.router, unit.retriever, unit.generator, unit.gate

        def on_route(rec: Recording | None, result: Any, ctx: Any, *args: Any, **kw: Any) -> None:
            text = getattr(getattr(ctx, "current", None), "text", "") or ""
            found = _IMAGE_DESCRIPTION.search(text)
            self._add_to(
                rec,
                "routed",
                Routed(
                    list(getattr(ctx, "image_findings", []) or []),
                    found.group(0) if found else None,
                    self._take_judge(),
                ),
            )

        def on_retrieve(
            rec: Recording | None, result: EvidenceBundle, ctx: Any, route: Any
        ) -> None:
            self._add_to(rec, "retrievals", Retrieval(route.intent, result))

        def on_generate(
            rec: Recording | None,
            result: list[Draft],
            ctx: Any,
            route: Any,
            evidence: Any,
            *,
            retry_feedback: list[str] | None = None,
        ) -> None:
            generation = Generation(route.intent, list(retry_feedback or []), list(result))
            self._add_to(rec, "generations", generation)

        def on_check(
            rec: Recording | None,
            result: QAResult,
            ctx: Any,
            route: Any,
            draft: Draft,
            evidence: Any,
        ) -> None:
            self._add_to(rec, "checks", Check(route.intent, draft, result, self._take_judge()))

        self._wrap(router, "route", on_route, judged=True)
        self._wrap(retriever, "retrieve", on_retrieve)
        self._wrap(generator, "generate", on_generate)
        self._wrap(gate, "check", on_check, judged=True)
        unit._showcase_recorder = self

    def _wrap(
        self, obj: Any, name: str, observe: Callable[..., None], *, judged: bool = False
    ) -> None:
        """照原样调用、照原样返回；调用开始时绑定当时正在记录的消息（线程局部，判断的 observer
        据此记），结束后交给 observe 记进这条消息（_add_to：消息已经结束就不记）。"""
        original = getattr(obj, name)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            recording = self._open()
            outer = getattr(self._local, "recording", None)
            self._local.recording = recording
            if judged:
                self._take_judge()  # 清掉本线程上一次留下的判断，免得挂错
            try:
                result = original(*args, **kwargs)
            finally:
                self._local.recording = outer
            observe(recording, result, *args, **kwargs)
            return result

        setattr(obj, name, wrapped)


__all__ = ["Check", "Generation", "Recorder", "Recording", "Retrieval", "Routed"]
