"""serve 进程里的后台定时任务（方案 §8.4 续期 / 过期清理、§12.2 就绪、§12.4 授权过期告警）。

为什么放进 serve：取消订购的闸门（AssistService 按店铺的读写闸）、工具缓存、适配器的去重记录
都在 serve 进程内。续期、过期清理放在另一个进程里跑，就和 serve 的在途处理互不相知：清理完之后
serve 手上的消息照样把推荐、审计写进来（没有人再删），serve 的缓存里也还留着已清理店铺的数据。
所以它们在 serve 里按间隔跑，用 serve 的同一个 AssistService（同一把闸、同一份缓存）；命令行的
refresh-tokens / purge-expired 等改数据的任务要先拿单副本租约，serve 在跑时拒绝
（rodit.ops.jobs.tenancy）。

就绪检查、授权过期检查也在这里刷新：抓取 /metrics 时不再查库（数据库网络不通时一次检查要十几秒，
超过 Prometheus 的抓取超时，rodit_ready 反而一个样本都没有），只输出上次的结果和它的年龄。

- 每个任务一条守护线程，按间隔跑：上一轮跑完才开始计时，不会叠起来；间隔 0 = 不跑；
- guard：每轮跑之前再确认一次（如单副本租约还在本进程），不满足就跳过这一轮（skipped）；
- 任务出错只记日志和指标，不影响 serve 和其他任务。日志只记异常类型，不记说明
  （说明里可能有连接串；令牌从不进异常说明，这里也不冒这个险）；
- 任务函数返回 False 表示这一轮没做完（它自己已经记了日志），和出错一样记 error；
- stop 通知所有线程退出，等它们跑完手上这一轮（有上限）。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

logger = logging.getLogger("rodit.ops")

OUTCOME_OK = "ok"
OUTCOME_ERROR = "error"
OUTCOME_SKIPPED = "skipped"

DEFAULT_STOP_TIMEOUT_S = 10.0

ResultHook = Callable[[str, str], None]
"""(任务名, 结果 ok / error / skipped) → 记指标。出错只记日志。"""


@dataclass(frozen=True)
class PeriodicJob:
    """一个后台任务。fn 返回 False 表示没做完（记 error）；其余返回值都算完成。

    guard 返回 False 时跳过这一轮；initial_delay_s 是启动后第一轮之前等多久。"""

    name: str
    fn: Callable[[], object]
    interval_s: float
    initial_delay_s: float = 0.0
    guard: Callable[[], bool] | None = None

    @property
    def enabled(self) -> bool:
        return self.interval_s > 0


class BackgroundRunner:
    """按间隔跑一组 PeriodicJob（见模块说明）。start / stop 都可以重复调用；线程安全。"""

    def __init__(self, jobs: Iterable[PeriodicJob], *, on_result: ResultHook | None = None) -> None:
        self._jobs: dict[str, PeriodicJob] = {}
        for job in jobs:
            if job.name in self._jobs:
                raise ValueError(f"后台任务重名：{job.name}")
            if job.interval_s < 0 or job.initial_delay_s < 0:
                raise ValueError(f"后台任务 {job.name} 的间隔不能为负")
            self._jobs[job.name] = job
        self._on_result = on_result
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    @property
    def jobs(self) -> dict[str, PeriodicJob]:
        return dict(self._jobs)

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def start(self) -> None:
        """给每个启用的任务起一条守护线程（已经起过就什么也不做）。"""
        with self._lock:
            if self._threads or self._stop.is_set():
                return
            for job in self._jobs.values():
                if not job.enabled:
                    logger.info("后台任务 %s 没有启用（间隔为 0）", job.name)
                    continue
                thread = threading.Thread(
                    target=self._loop, args=(job,), name=f"rodit-bg-{job.name}", daemon=True
                )
                self._threads.append(thread)
                thread.start()
            if self._threads:
                logger.info("后台任务已启动：%s", ",".join(t.name[8:] for t in self._threads))

    def stop(self, timeout_s: float = DEFAULT_STOP_TIMEOUT_S) -> None:
        """通知线程退出，最多等 timeout_s 秒让它们跑完手上这一轮。没退出的是守护线程，
        进程退出时一起结束：续期可以重跑；清理删令牌之前写了开始清理的标记
        shop.purge_started，删到一半被打断的店铺下一轮过期清理接着删完（重启之后也一样）。"""
        self._stop.set()
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        alive = [t.name for t in threads if t.is_alive()]
        if alive:
            logger.warning("后台任务没有在 %.0f 秒内结束：%s", timeout_s, ",".join(alive))

    def run_once(self, name: str) -> str:
        """马上在当前线程跑一轮（测试、运维排查用），返回 ok / error / skipped。"""
        return self._run(self._jobs[name])

    # ---------------------------------------------------------------- 内部

    def _loop(self, job: PeriodicJob) -> None:
        if self._stop.wait(job.initial_delay_s):
            return
        while True:
            self._run(job)
            if self._stop.wait(job.interval_s):
                return

    def _run(self, job: PeriodicJob) -> str:
        outcome = OUTCOME_OK
        try:
            if job.guard is not None and not job.guard():
                logger.warning("后台任务 %s 跳过这一轮（前提条件不满足）", job.name)
                outcome = OUTCOME_SKIPPED
            elif job.fn() is False:
                outcome = OUTCOME_ERROR
        except Exception as exc:  # noqa: BLE001  后台任务出错不能拖垮 serve
            logger.error("后台任务 %s 出错：%s", job.name, type(exc).__name__)
            outcome = OUTCOME_ERROR
        if self._on_result is not None:
            try:
                self._on_result(job.name, outcome)
            except Exception as exc:  # noqa: BLE001
                logger.warning("指标采集失败：%s", type(exc).__name__)
        return outcome


__all__ = [
    "OUTCOME_ERROR",
    "OUTCOME_OK",
    "OUTCOME_SKIPPED",
    "BackgroundRunner",
    "PeriodicJob",
]
