"""每条消息的时间预算（方案 §12.1 端到端 ≤ 5 秒、§12.2 超时降级）。

引擎处理一条买家消息时用

    deadline_scope(latency.message_budget_s, min_llm_remaining_s=latency.min_llm_remaining_s)

包住（latency 是 EngineConfig.latency），各步骤用 remaining() / expired() 判断还来不来得及；
生成、质检调大模型之前用 llm_allowed() 判断：剩余时间少于 min_llm_remaining_s 就不调，
退回 FAQ / 模板。这个下限跟着截止时间走（contextvar），生成器、质检不用另外拿配置。
生成器本来要写大模型草稿、因为剩余时间不够而没调时用 note_llm_skipped() 记一笔；引擎用
llm_skip_scope() 包住这一次生成，据此给这一条记 budget 降级（没有名额、FAQ / 模板已经
够了而不调大模型的不记）。

- 截止时间放在 contextvar 里：同一线程里的调用都看得到；线程池里的任务要用 submit_in_context
  提交（contextvars.copy_context().run），才能看到提交时的截止时间（和 core.context.shop_scope
  一样的做法）。
- 嵌套：内层的截止时间不会晚于外层（取两者中较早的），外层的预算不会被内层"续命"；
  调大模型的剩余时间下限也只会更严（取两者中较大的），内层没给时沿用外层的。
- 用单调时钟（time.monotonic），不受系统时间调整影响；测试可以注入假的 now 函数。
  嵌套的几层要用同一个时钟。
- 没有截止时间时 remaining() 返回 None、expired() 返回 False、llm_allowed() 返回 True：
  不在引擎里的调用（离线任务、控制台）不受影响。
- 到截止时间没做完的线程池任务不等（Python 线程不能强行停止），它们会在后台跑完、照常留下
  副作用（工具结果写缓存、大模型用量计量）。在 background_scope(work, shop_id) 里经
  submit_in_context 提交的任务都按店铺记在 BackgroundWork 里（含任务里再提交的嵌套任务）：
  取消订购删数据之前等它们做完，迟到的写入一并删掉（§8.4）。
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

T = TypeVar("T")

MonotonicNow = Callable[[], float]


def _check_floor(min_llm_remaining_s: float) -> float:
    if min_llm_remaining_s < 0:
        raise ValueError("调大模型的剩余时间下限不能是负数")
    return float(min_llm_remaining_s)


class Deadline:
    """一个截止时间：从创建时起 seconds 秒（seconds 必须 ≥ 0）。

    min_llm_remaining_s：剩余时间少于它就不再调大模型（LatencyConfig.min_llm_remaining_s，
    默认 0 = 没过期就可以调），见 llm_allowed。"""

    __slots__ = ("_now", "budget_s", "expires_at", "min_llm_remaining_s", "started_at")

    def __init__(
        self,
        seconds: float,
        *,
        now: MonotonicNow = time.monotonic,
        min_llm_remaining_s: float = 0.0,
    ) -> None:
        if seconds < 0:
            raise ValueError("时间预算不能是负数")
        self._now = now
        self.budget_s = float(seconds)
        self.min_llm_remaining_s = _check_floor(min_llm_remaining_s)
        self.started_at = now()
        self.expires_at = self.started_at + self.budget_s

    @classmethod
    def until(
        cls,
        expires_at: float,
        *,
        now: MonotonicNow = time.monotonic,
        min_llm_remaining_s: float = 0.0,
    ) -> Deadline:
        """截止到单调时钟的 expires_at（已经过了也可以，此时立即过期）。"""
        deadline = cls(0.0, now=now, min_llm_remaining_s=min_llm_remaining_s)
        deadline.expires_at = expires_at
        deadline.budget_s = max(0.0, expires_at - deadline.started_at)
        return deadline

    def with_llm_floor(self, min_llm_remaining_s: float) -> Deadline:
        """同一个截止时间（开始时间、预算、时钟都不变），换一个调大模型的剩余时间下限。"""
        floor = _check_floor(min_llm_remaining_s)
        if floor == self.min_llm_remaining_s:
            return self
        copy = Deadline.__new__(Deadline)
        copy._now = self._now
        copy.budget_s = self.budget_s
        copy.started_at = self.started_at
        copy.expires_at = self.expires_at
        copy.min_llm_remaining_s = floor
        return copy

    def remaining(self) -> float:
        """还剩多少秒（不会小于 0）。"""
        return max(0.0, self.expires_at - self._now())

    def elapsed(self) -> float:
        return self._now() - self.started_at

    def expired(self) -> bool:
        return self._now() >= self.expires_at

    def llm_allowed(self) -> bool:
        """还来得及调一次大模型：没过期，且剩余时间不少于 min_llm_remaining_s。"""
        left = self.expires_at - self._now()
        return left > 0 and left >= self.min_llm_remaining_s

    def __repr__(self) -> str:
        return (
            f"Deadline(budget_s={self.budget_s:.3f}, remaining={self.remaining():.3f},"
            f" min_llm_remaining_s={self.min_llm_remaining_s:.3f})"
        )


_current: ContextVar[Deadline | None] = ContextVar("rodit_deadline", default=None)


def current_deadline() -> Deadline | None:
    """当前生效的截止时间；没有返回 None。"""
    return _current.get()


@contextmanager
def deadline_scope(
    seconds: float | None,
    *,
    min_llm_remaining_s: float | None = None,
    now: MonotonicNow = time.monotonic,
) -> Iterator[Deadline | None]:
    """在这段代码里设置截止时间（seconds 秒后），返回生效的 Deadline。

    外层已经有更早的截止时间时沿用外层的（内层不能延长外层的预算）。seconds 为 None 表示不设新的
    截止时间：沿用外层的（没有外层就是没有截止时间，返回 None）。退出时恢复外层的。

    min_llm_remaining_s：调大模型的剩余时间下限（见 llm_allowed）。为 None 时沿用外层的（没有外层
    就是 0）；给了时取它和外层的较大值（内层不能放宽外层的下限）。
    """
    outer = _current.get()
    outer_floor = outer.min_llm_remaining_s if outer is not None else 0.0
    floor = (
        outer_floor
        if min_llm_remaining_s is None
        else max(outer_floor, _check_floor(min_llm_remaining_s))
    )
    effective: Deadline | None
    if seconds is None:
        effective = outer.with_llm_floor(floor) if outer is not None else None
    else:
        inner = Deadline(seconds, now=now, min_llm_remaining_s=floor)
        if outer is not None and outer.expires_at <= inner.expires_at:
            effective = outer.with_llm_floor(floor)
        else:
            effective = inner
    token = _current.set(effective)
    try:
        yield effective
    finally:
        _current.reset(token)


def remaining() -> float | None:
    """当前截止时间还剩多少秒；没有截止时间返回 None。"""
    deadline = _current.get()
    return None if deadline is None else deadline.remaining()


def expired() -> bool:
    """当前截止时间已经过了；没有截止时间返回 False。"""
    deadline = _current.get()
    return deadline is not None and deadline.expired()


def llm_allowed() -> bool:
    """还来得及调一次大模型（见 Deadline.llm_allowed）；没有截止时间返回 True。

    生成、质检调大模型之前判断：返回 False 时不调，退回 FAQ / 模板（§12.2）。"""
    deadline = _current.get()
    return deadline is None or deadline.llm_allowed()


class LLMSkip:
    """llm_skip_scope 里的记录：skipped = 这段代码里有大模型调用因为剩余时间不够而没发。"""

    __slots__ = ("skipped",)

    def __init__(self) -> None:
        self.skipped = False

    def __repr__(self) -> str:
        return f"LLMSkip(skipped={self.skipped})"


_llm_skip: ContextVar[LLMSkip | None] = ContextVar("rodit_llm_skip", default=None)


@contextmanager
def llm_skip_scope() -> Iterator[LLMSkip]:
    """这段代码里的 note_llm_skipped() 记到返回的 LLMSkip 上（每次进入一份新的；嵌套时只记到
    最里面一层）。引擎包住一次生成：

        with llm_skip_scope() as skip:
            drafts = generator.generate(ctx, route, evidence)
        if skip.skipped: ...  # 本来要调大模型，时间不够没调

    记录跟着 contextvar 走：并行的几条分条各在自己的线程里进入，互不影响。"""
    record = LLMSkip()
    token = _llm_skip.set(record)
    try:
        yield record
    finally:
        _llm_skip.reset(token)


def note_llm_skipped() -> None:
    """本来要调大模型、因为 llm_allowed() 为 False 没调时记一笔。只在确实有调用要发的时候记
    （没有名额、不需要大模型的不算）；不在 llm_skip_scope 里时什么都不做。"""
    record = _llm_skip.get()
    if record is not None:
        record.skipped = True


def clamp_timeout(timeout_s: float) -> float:
    """单次调用的超时不超过剩余预算：min(timeout_s, remaining())；没有截止时间时原样返回。"""
    left = remaining()
    return timeout_s if left is None else min(timeout_s, left)


class BackgroundWork:
    """按店铺记着交给线程池、还没做完的任务（含到截止时间被丢下、还在后台跑的）。线程安全。

    track 在提交之后马上调（任务已经做完也没关系：做完的回调立即执行）；wait_idle 等这家店
    的任务全部做完。没有进行中的任务时不占内存。"""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending: dict[str, int] = {}

    def track(self, shop_id: str, future: Future[Any]) -> None:
        with self._cond:
            self._pending[shop_id] = self._pending.get(shop_id, 0) + 1
        future.add_done_callback(lambda _f: self._done(shop_id))

    def _done(self, shop_id: str) -> None:
        with self._cond:
            left = self._pending.get(shop_id, 0) - 1
            if left > 0:
                self._pending[shop_id] = left
            else:
                self._pending.pop(shop_id, None)
            self._cond.notify_all()

    def pending(self, shop_id: str) -> int:
        with self._cond:
            return self._pending.get(shop_id, 0)

    def wait_idle(self, shop_id: str, timeout: float | None = None) -> bool:
        """等这家店的任务全部做完；timeout 秒之内做完（或本来就没有）返回 True。"""
        limit = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._cond:
            while self._pending.get(shop_id):
                if limit is None:
                    self._cond.wait()
                    continue
                left = limit - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
            return True


_background: ContextVar[tuple[BackgroundWork, str] | None] = ContextVar(
    "rodit_background_work", default=None
)


@contextmanager
def background_scope(work: BackgroundWork, shop_id: str) -> Iterator[None]:
    """这段代码里（含它提交到线程池的任务里）submit_in_context 提交的任务都记到 work 的
    shop_id 名下。"""
    token = _background.set((work, shop_id))
    try:
        yield
    finally:
        _background.reset(token)


def submit_in_context(pool: Executor, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
    """把任务提交到线程池，并带上当前的 contextvars（截止时间、店铺范围等）：

        future = submit_in_context(pool, generate, ctx, route)

    线程池的工作线程不会继承提交方的 contextvar；这里复制一份上下文、在里面运行 fn。
    每个任务一份拷贝，任务里对 contextvar 的修改不影响提交方和其他任务。
    在 background_scope 里提交时，任务记到那个 BackgroundWork 的店铺名下（做完才减）；
    嵌套提交在外层任务还没做完时就已经记上，计数不会中途归零。
    """
    context = contextvars.copy_context()
    future = pool.submit(context.run, fn, *args, **kwargs)
    scope = _background.get()
    if scope is not None:
        work, shop_id = scope
        work.track(shop_id, future)
    return future


__all__ = [
    "BackgroundWork",
    "Deadline",
    "LLMSkip",
    "background_scope",
    "clamp_timeout",
    "current_deadline",
    "deadline_scope",
    "expired",
    "llm_allowed",
    "llm_skip_scope",
    "note_llm_skipped",
    "remaining",
    "submit_in_context",
]
