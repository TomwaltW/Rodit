"""引擎留在后台的任务（engine.concurrency.BackgroundWork，方案 §8.4、§12.2）。

到截止时间被丢下的任务还在后台跑，它调用的平台工具、大模型会迟到地写缓存和用量记录；
在 background_scope 里提交的任务按店铺记下来，取消订购删数据之前等它们做完。
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
import time

from rodit.core import budget
from rodit.engine.concurrency import BackgroundWork, background_scope, run_ordered

SHOP = "s_bg_1"
OTHER = "s_bg_2"


def test_background_work_counts_per_shop_until_the_tasks_finish():
    work = BackgroundWork()
    assert work.pending(SHOP) == 0 and work.wait_idle(SHOP, timeout=0)
    first: cf.Future[int] = cf.Future()
    second: cf.Future[int] = cf.Future()
    work.track(SHOP, first)
    work.track(SHOP, second)
    work.track(OTHER, cf.Future())
    assert work.pending(SHOP) == 2 and work.pending(OTHER) == 1
    assert work.wait_idle(SHOP, timeout=0.05) is False  # 还没做完：到点返回 False

    first.set_result(1)
    assert work.pending(SHOP) == 1
    threading.Timer(0.1, lambda: second.set_exception(RuntimeError("任务失败也算做完"))).start()
    assert work.wait_idle(SHOP, timeout=5) is True
    assert work.pending(SHOP) == 0 and work.pending(OTHER) == 1  # 别的店铺不受影响

    done: cf.Future[int] = cf.Future()
    done.set_result(0)
    work.track(SHOP, done)  # 已经做完的：立即减掉
    assert work.pending(SHOP) == 0


def test_run_ordered_records_abandoned_tasks_under_the_scope_shop():
    work = BackgroundWork()
    release = threading.Event()
    nested_started = threading.Event()

    def slow() -> str:
        release.wait(5)
        return "late"

    def with_nested() -> list[str | None]:
        # 分条里再并行（质检）：嵌套提交时外层还没做完，计数不会中途归零
        outcomes = run_ordered([slow, slow], grace_s=0.0, name="rodit-test-nested")
        nested_started.set()
        return [o.value for o in outcomes]

    with (
        background_scope(work, SHOP),
        budget.deadline_scope(0.1),
    ):
        outcomes = run_ordered([slow, with_nested], grace_s=0.05, name="rodit-test")
    assert outcomes[0].finished is False  # 到点丢下
    assert nested_started.wait(5)
    # 丢下的外层任务 + 嵌套的两个都还记着（带嵌套的那个外层任务自己收尾之后减掉）
    limit = time.monotonic() + 2
    while work.pending(SHOP) > 3 and time.monotonic() < limit:
        time.sleep(0.01)
    assert work.pending(SHOP) == 3 and work.pending(OTHER) == 0
    assert work.wait_idle(SHOP, timeout=0.05) is False

    started = time.monotonic()
    threading.Timer(0.2, release.set).start()
    assert work.wait_idle(SHOP, timeout=5) is True  # 外层、嵌套的都做完了
    assert time.monotonic() - started >= 0.15
    assert work.pending(SHOP) == 0


def test_run_ordered_outside_a_scope_records_nothing():
    work = BackgroundWork()
    outcomes = run_ordered([lambda: 1, lambda: 2])
    assert [o.value for o in outcomes] == [1, 2]
    assert work.pending(SHOP) == 0
