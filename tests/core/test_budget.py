"""每条消息的时间预算（core.budget，方案 §12.1 端到端 ≤ 5 秒、§12.2 超时降级）。

时间用假的单调时钟推进，不 sleep；线程池只用来验证截止时间能跟着 submit_in_context 进工作线程。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from rodit.core import budget
from rodit.core.budget import (
    Deadline,
    clamp_timeout,
    current_deadline,
    deadline_scope,
    expired,
    remaining,
    submit_in_context,
)
from rodit.core.context import current_shop_id, shop_scope


class Mono:
    """假的单调时钟。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_no_deadline_by_default():
    assert current_deadline() is None
    assert remaining() is None and expired() is False
    assert clamp_timeout(2.5) == 2.5


def test_deadline_counts_down_and_expires():
    mono = Mono()
    d = Deadline(4.5, now=mono)
    assert d.budget_s == 4.5 and d.remaining() == pytest.approx(4.5)
    mono.advance(4.0)
    assert d.remaining() == pytest.approx(0.5) and not d.expired()
    assert d.elapsed() == pytest.approx(4.0)
    mono.advance(0.5)
    assert d.expired() and d.remaining() == 0.0
    mono.advance(10)
    assert d.remaining() == 0.0  # 不会小于 0
    assert "Deadline(" in repr(d)


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        Deadline(-0.1)


def test_until_builds_an_absolute_deadline():
    mono = Mono()
    d = Deadline.until(1003.0, now=mono)
    assert d.remaining() == pytest.approx(3.0)
    past = Deadline.until(900.0, now=mono)
    assert past.expired() and past.remaining() == 0.0 and past.budget_s == 0.0


def test_scope_sets_and_restores():
    mono = Mono()
    with deadline_scope(2.0, now=mono) as d:
        assert current_deadline() is d
        assert remaining() == pytest.approx(2.0)
        assert clamp_timeout(2.5) == pytest.approx(2.0)
        assert clamp_timeout(1.0) == 1.0
        mono.advance(2.5)
        assert expired() and remaining() == 0.0 and clamp_timeout(1.0) == 0.0
    assert current_deadline() is None and not expired()


def test_nested_scope_never_extends_the_outer_budget():
    mono = Mono()
    with deadline_scope(1.0, now=mono) as outer:
        with deadline_scope(5.0, now=mono) as inner:  # 更长：沿用外层
            assert inner is outer
            assert remaining() == pytest.approx(1.0)
        with deadline_scope(0.3, now=mono) as tighter:  # 更短：内层生效
            assert tighter is not outer
            assert remaining() == pytest.approx(0.3)
            mono.advance(0.4)
            assert expired()
        assert current_deadline() is outer  # 退出后恢复外层
        assert not expired() and remaining() == pytest.approx(0.6)
        with deadline_scope(None, now=mono) as same:  # None：沿用外层
            assert same is outer


def test_none_scope_without_outer_means_no_deadline():
    with deadline_scope(None) as d:
        assert d is None and remaining() is None and not expired()


def test_scope_restores_on_exception():
    with pytest.raises(RuntimeError):
        with deadline_scope(1.0):
            raise RuntimeError("boom")
    assert current_deadline() is None


def test_submit_in_context_propagates_deadline_and_shop_to_workers():
    mono = Mono()
    seen: list[tuple[float | None, bool, str | None, str]] = []
    lock = threading.Lock()

    def probe(tag: str) -> str:
        with lock:
            seen.append((remaining(), expired(), current_shop_id(), tag))
        return tag

    with ThreadPoolExecutor(max_workers=3) as pool:
        with shop_scope("s_1001"), deadline_scope(4.5, now=mono):
            mono.advance(1.0)
            futures = [submit_in_context(pool, probe, f"part{i}") for i in range(3)]
            assert [f.result() for f in futures] == ["part0", "part1", "part2"]
            plain = pool.submit(remaining).result()  # 不用 submit_in_context：工作线程看不到
    assert plain is None
    assert sorted(tag for *_, tag in seen) == ["part0", "part1", "part2"]
    for left, is_expired, shop, _ in seen:
        assert left == pytest.approx(3.5) and is_expired is False and shop == "s_1001"


def test_submit_in_context_passes_args_and_isolates_changes():
    mono = Mono()

    def worker(a: int, *, b: int) -> float | None:
        with deadline_scope(0.1, now=mono):  # 任务里改截止时间，不影响提交方
            return remaining()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with deadline_scope(3.0, now=mono) as outer:
            inner_left = submit_in_context(pool, worker, 1, b=2).result()
            assert current_deadline() is outer and remaining() == pytest.approx(3.0)
    assert inner_left == pytest.approx(0.1)


def test_expired_deadline_is_visible_in_workers():
    mono = Mono()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with deadline_scope(0.5, now=mono):
            mono.advance(0.6)
            assert submit_in_context(pool, expired).result() is True
            assert submit_in_context(pool, remaining).result() == 0.0


def test_public_names():
    assert set(budget.__all__) == {
        "Deadline",
        "clamp_timeout",
        "current_deadline",
        "deadline_scope",
        "expired",
        "remaining",
        "submit_in_context",
    }
