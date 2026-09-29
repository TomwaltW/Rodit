"""Prometheus 文本格式（0.0.4）的指标登记表（方案 §12.4 可观测性）。手写实现，不引入新依赖。

- Counter / Gauge / Histogram 三种类型，按标签值分序列；线程安全（每个指标一把锁）。
- 标签值一律过 safe_label_value：只允许 ASCII 字母数字和 ``_ . : / { } + -``，最长 128 个字符，
  并且不能被 core.pii.contains_pii 识别为个人信息；不合格的值换成 "other"。中文、空格、
  手机号这类值因此进不了标签——标签里只能是店铺 ID、问题类型、规则名这类代码（§5.3、§12.3）。
- 每个指标的序列数有上限（max_series）：超出后新的标签组合记到 "_overflow" 序列，
  标签值再多也不会把内存和 Prometheus 撑爆。
- forget(label, value)：删除某个标签值的全部序列（取消订购后删掉该店铺的指标，§8.4）。
- 登记表可以挂渲染前的钩子（before_render），抓取时刷新按需计算的量（如授权即将过期的店铺数）；
  钩子出错只记日志，不影响其余指标输出。
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from rodit.core.pii import contains_pii

logger = logging.getLogger(__name__)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
"""Prometheus 文本格式的 Content-Type。"""

OTHER = "other"
"""不合格的标签值统一换成这个。"""
OVERFLOW = "_overflow"
"""超出序列上限后新的标签组合都记到这个值上。"""

DEFAULT_MAX_SERIES = 10_000

_METRIC_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_SAFE_VALUE = re.compile(r"[A-Za-z0-9_.:/{}+\-]{1,128}")

LabelKey = tuple[str, ...]


def safe_label_value(value: object) -> str:
    """标签值 → 可以对外暴露的形式。None 记作 "none"；不合格（含中文、空格、个人信息等）记作
    "other"。只接受字符串、整数、枚举这类代码值，其他类型一律 "other"。"""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        return OTHER
    text = value.strip()
    if not _SAFE_VALUE.fullmatch(text) or contains_pii(text):
        return OTHER
    return text


def format_value(value: float) -> str:
    """数值的文本形式：整数不带小数点，特殊值用 +Inf / -Inf / NaN。"""
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value.is_integer() and abs(value) < 2**53:
        return str(int(value))
    return repr(value)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _escape_help(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _labels_text(names: Sequence[str], values: Sequence[str]) -> str:
    if not names:
        return ""
    pairs = ",".join(f'{n}="{_escape_label(v)}"' for n, v in zip(names, values, strict=True))
    return "{" + pairs + "}"


class _Metric:
    kind = "untyped"

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        *,
        max_series: int = DEFAULT_MAX_SERIES,
    ) -> None:
        if not _METRIC_NAME.fullmatch(name):
            raise ValueError(f"指标名不合法：{name!r}")
        for label in labelnames:
            if not _LABEL_NAME.fullmatch(label) or label.startswith("__") or label == "le":
                raise ValueError(f"标签名不合法：{label!r}")
        if len(set(labelnames)) != len(labelnames):
            raise ValueError(f"标签名重复：{labelnames!r}")
        if max_series < 1:
            raise ValueError("max_series 至少为 1")
        self.name = name
        self.help = help_text
        self.labelnames: tuple[str, ...] = tuple(labelnames)
        self._max_series = max_series
        self._lock = threading.Lock()
        self._series: dict[LabelKey, Any] = {}

    # ---------------------------------------------------------------- 序列

    def _key(self, labels: Mapping[str, object] | None) -> LabelKey:
        labels = labels or {}
        if set(labels) != set(self.labelnames):
            raise ValueError(
                f"{self.name} 的标签是 {sorted(self.labelnames)}，传入的是 {sorted(labels)}"
            )
        return tuple(safe_label_value(labels[n]) for n in self.labelnames)

    def _slot(self, key: LabelKey, factory: Callable[[], Any]) -> Any:
        """调用方持有 self._lock。"""
        series = self._series.get(key)
        if series is None:
            if len(self._series) >= self._max_series:
                key = (OVERFLOW,) * len(self.labelnames)
                series = self._series.get(key)
            if series is None:
                series = self._series[key] = factory()
        return series

    def forget(self, label: str, value: str) -> int:
        """删除 label=value 的全部序列，返回删除的条数（标签不存在时为 0）。"""
        if label not in self.labelnames:
            return 0
        idx = self.labelnames.index(label)
        target = safe_label_value(value)
        with self._lock:
            doomed = [k for k in self._series if k[idx] == target]
            for k in doomed:
                del self._series[k]
        return len(doomed)

    def series_count(self) -> int:
        with self._lock:
            return len(self._series)

    # ---------------------------------------------------------------- 输出

    def _lines(self) -> list[str]:
        raise NotImplementedError

    def render(self) -> str:
        head = [f"# HELP {self.name} {_escape_help(self.help)}", f"# TYPE {self.name} {self.kind}"]
        return "\n".join(head + self._lines())


class Counter(_Metric):
    """只增不减的计数（名字按惯例以 _total 结尾）。"""

    kind = "counter"

    def inc(self, labels: Mapping[str, object] | None = None, amount: float = 1.0) -> None:
        if amount < 0 or math.isnan(amount):
            raise ValueError("计数只能增加")
        key = self._key(labels)
        with self._lock:
            box = self._slot(key, lambda: [0.0])
            box[0] += amount

    def value(self, labels: Mapping[str, object] | None = None) -> float:
        key = self._key(labels)
        with self._lock:
            box = self._series.get(key)
            return box[0] if box else 0.0

    def _lines(self) -> list[str]:
        with self._lock:
            items = sorted(self._series.items())
        return [
            f"{self.name}{_labels_text(self.labelnames, k)} {format_value(v[0])}" for k, v in items
        ]


class Gauge(_Metric):
    """可增可减的当前值。replace 用一组新值整体替换（抓取时重新计算的量）。"""

    kind = "gauge"

    def set(self, labels: Mapping[str, object] | None = None, value: float = 0.0) -> None:
        key = self._key(labels)
        with self._lock:
            box = self._slot(key, lambda: [0.0])
            box[0] = float(value)

    def inc(self, labels: Mapping[str, object] | None = None, amount: float = 1.0) -> None:
        key = self._key(labels)
        with self._lock:
            box = self._slot(key, lambda: [0.0])
            box[0] += amount

    def replace(self, values: Iterable[tuple[Mapping[str, object], float]]) -> None:
        fresh: dict[LabelKey, list[float]] = {}
        for labels, value in values:
            key = self._key(labels)
            if key not in fresh and len(fresh) >= self._max_series:
                key = (OVERFLOW,) * len(self.labelnames)
            fresh.setdefault(key, [0.0])[0] = float(value)
        with self._lock:
            self._series = dict(fresh)

    def value(self, labels: Mapping[str, object] | None = None) -> float | None:
        key = self._key(labels)
        with self._lock:
            box = self._series.get(key)
            return box[0] if box else None

    def _lines(self) -> list[str]:
        with self._lock:
            items = sorted(self._series.items())
        return [
            f"{self.name}{_labels_text(self.labelnames, k)} {format_value(v[0])}" for k, v in items
        ]


class _HistogramSeries:
    __slots__ = ("buckets", "count", "sum")

    def __init__(self, n: int) -> None:
        self.buckets = [0] * n  # 每个上界各自的计数（输出时再累加）
        self.count = 0
        self.sum = 0.0


class Histogram(_Metric):
    """分桶直方图：输出 _bucket（累计，含 +Inf）、_sum、_count。
    P50 / P95 用 PromQL 的 histogram_quantile(0.95, sum by (le) (rate(x_bucket[5m]))) 算。"""

    kind = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        *,
        buckets: Sequence[float],
        max_series: int = DEFAULT_MAX_SERIES,
    ) -> None:
        super().__init__(name, help_text, labelnames, max_series=max_series)
        bounds = sorted({float(b) for b in buckets if not math.isinf(float(b))})
        if not bounds:
            raise ValueError("至少要有一个有限的桶上界")
        self.bounds: tuple[float, ...] = (*bounds, math.inf)

    def observe(self, labels: Mapping[str, object] | None = None, value: float = 0.0) -> None:
        value = float(value)
        if math.isnan(value):
            return
        key = self._key(labels)
        n = len(self.bounds)
        with self._lock:
            series = self._slot(key, lambda: _HistogramSeries(n))
            for i, bound in enumerate(self.bounds):
                if value <= bound:
                    series.buckets[i] += 1
                    break
            series.count += 1
            series.sum += value

    def snapshot(self, labels: Mapping[str, object] | None = None) -> tuple[list[int], int, float]:
        """(累计桶计数, 总数, 总和)；没有这个序列时全是 0。"""
        key = self._key(labels)
        with self._lock:
            series = self._series.get(key)
            if series is None:
                return [0] * len(self.bounds), 0, 0.0
            raw, count, total = list(series.buckets), series.count, series.sum
        cumulative, running = [], 0
        for c in raw:
            running += c
            cumulative.append(running)
        return cumulative, count, total

    def _lines(self) -> list[str]:
        with self._lock:
            items = sorted((k, (list(s.buckets), s.count, s.sum)) for k, s in self._series.items())
        lines: list[str] = []
        names_le = (*self.labelnames, "le")
        for key, (raw, count, total) in items:
            running = 0
            for bound, c in zip(self.bounds, raw, strict=True):
                running += c
                labels = _labels_text(names_le, (*key, format_value(bound)))
                lines.append(f"{self.name}_bucket{labels} {running}")
            labels = _labels_text(self.labelnames, key)
            lines.append(f"{self.name}_sum{labels} {format_value(total)}")
            lines.append(f"{self.name}_count{labels} {count}")
        return lines


class MetricsRegistry:
    """一个进程（一个平台单元）的全部指标。按登记顺序输出。"""

    def __init__(self, *, max_series: int = DEFAULT_MAX_SERIES) -> None:
        self._max_series = max_series
        self._lock = threading.Lock()
        self._metrics: dict[str, _Metric] = {}
        self._hooks: list[Callable[[], None]] = []

    def _add(self, metric: _Metric) -> Any:
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"指标重复登记：{metric.name}")
            self._metrics[metric.name] = metric
        return metric

    def counter(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> Counter:
        return self._add(Counter(name, help_text, labelnames, max_series=self._max_series))

    def gauge(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> Gauge:
        return self._add(Gauge(name, help_text, labelnames, max_series=self._max_series))

    def histogram(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        *,
        buckets: Sequence[float],
    ) -> Histogram:
        return self._add(
            Histogram(name, help_text, labelnames, buckets=buckets, max_series=self._max_series)
        )

    def get(self, name: str) -> _Metric | None:
        with self._lock:
            return self._metrics.get(name)

    def before_render(self, hook: Callable[[], None]) -> None:
        """登记渲染前的钩子（抓取时刷新按需计算的指标）。"""
        with self._lock:
            self._hooks.append(hook)

    def forget(self, label: str, value: str) -> int:
        """所有指标里删除 label=value 的序列，返回删除的总条数。"""
        with self._lock:
            metrics = list(self._metrics.values())
        return sum(m.forget(label, value) for m in metrics)

    def render(self) -> str:
        with self._lock:
            hooks = list(self._hooks)
            metrics = list(self._metrics.values())
        for hook in hooks:
            try:
                hook()
            except Exception as exc:  # noqa: BLE001  钩子失败不影响其余指标
                logger.warning("指标刷新失败：%s", type(exc).__name__)
        return "\n".join(m.render() for m in metrics) + "\n"


__all__ = [
    "CONTENT_TYPE",
    "DEFAULT_MAX_SERIES",
    "OTHER",
    "OVERFLOW",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "format_value",
    "safe_label_value",
]
