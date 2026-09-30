"""引擎的并发执行（方案 §12.1 推荐出现 P95 ≤ 5 秒、§12.2 超时降级）。

多问题消息的各条（§6.6 规则 5）、同一条的几份草稿的质检并行跑，结果按原来的顺序收回：
输出和顺序执行一样，只是更快。

- 任务用 core.budget.submit_in_context 提交：工作线程看得到调用方的截止时间、店铺范围
  （core.context.shop_scope，大模型成本按店铺归属）等 contextvar；
- 有截止时间时最多等到"截止时间 + grace_s"：截止时刻生成器刚退回模板、只剩硬规则质检的任务
  还能收尾。到点没做完的任务不再等——线程会继续跑完（Python 线程不能强行停止），结果丢掉。
  任务不写审计、推荐记录这类引擎自己的记录（都在调用方线程、收回结果之后写），但它调用的平台
  工具、大模型会照常留下副作用：工具结果写进缓存（tools.CachingToolGateway）、大模型用量计量
  （llm.MeteredLLMClient 的 llm.usage）；
- 所以在 background_scope(work, shop_id) 里提交的任务都记在 BackgroundWork 里（按店铺计数，
  做完才减；两者定义在 core.budget，submit_in_context 提交时记上，这里转出）：取消订购删数据
  之前 wait_idle 等它们做完，它们迟到的写入一并删掉（§8.4）。嵌套的并行（分条里的质检、生成器
  并行生成几份草稿、路由投票）在外层任务的线程里提交，外层还没做完时就已经记上，计数不会中途
  归零；
- 没有截止时间时等全部做完（离线任务、控制台、测试）；只有一个任务时直接在调用方线程里跑；
- 任务里抛出的异常放进 Outcome.error，由调用方按原顺序处理（和顺序执行一样：前面的先生效）。
"""

from __future__ import annotations

import concurrent.futures as cf
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from rodit.core.budget import (
    BackgroundWork,
    background_scope,
    current_deadline,
    submit_in_context,
)

T = TypeVar("T")

JOIN_GRACE_S = 0.15
"""等分条的宽限：截止时间之后最多再等这么久（生成器在截止时刻退回模板，硬规则质检是毫秒级）。
handle() 在截止时间 + 这个宽限 + 收尾（写推荐记录、审计）之内返回。"""

NESTED_GRACE_S = 0.05
"""分条内部（多份草稿并行质检）的宽限：比 JOIN_GRACE_S 短，分条自己先收尾，引擎才等得到它。"""


@dataclass(frozen=True)
class Outcome(Generic[T]):
    """一个任务的结果：finished=False 表示到点没做完（结果丢掉）；error 是任务抛出的异常。"""

    finished: bool
    value: T | None = None
    error: BaseException | None = None


def run_ordered(
    tasks: Sequence[Callable[[], T]],
    *,
    grace_s: float = JOIN_GRACE_S,
    name: str = "rodit-engine",
) -> list[Outcome[T]]:
    """并发运行 tasks，按原顺序返回每个任务的 Outcome（见模块说明）。"""
    if not tasks:
        return []
    deadline = current_deadline()
    if len(tasks) == 1 and deadline is None:
        return [run_inline(tasks[0])]
    pool = cf.ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix=name)
    try:
        # 在 background_scope 里时 submit_in_context 把任务记进 BackgroundWork
        futures = [submit_in_context(pool, task) for task in tasks]
    finally:
        # 不等：到点没做完的任务留在后台跑完，线程随后退出（BackgroundWork 记着它们）
        pool.shutdown(wait=False)
    timeout = None if deadline is None else deadline.remaining() + grace_s
    cf.wait(futures, timeout=timeout)
    return [_collect(f) for f in futures]


def run_inline(task: Callable[[], T]) -> Outcome[T]:
    """在调用方线程里直接跑一个任务，结果包成 Outcome（异常不抛出）。"""
    try:
        return Outcome(finished=True, value=task())
    except Exception as exc:  # noqa: BLE001 — 交给调用方按顺序处理
        return Outcome(finished=True, error=exc)


def _collect(future: cf.Future[T]) -> Outcome[T]:
    if not future.done():
        return Outcome(finished=False)
    error = future.exception()
    if error is not None:
        return Outcome(finished=True, error=error)
    return Outcome(finished=True, value=future.result())


__all__ = [
    "JOIN_GRACE_S",
    "NESTED_GRACE_S",
    "BackgroundWork",
    "Outcome",
    "background_scope",
    "run_inline",
    "run_ordered",
]
