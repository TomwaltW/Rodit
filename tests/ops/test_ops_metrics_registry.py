"""Prometheus 文本格式的指标登记表（方案 §12.4）：格式、标签值清理、序列上限、按店铺删除。"""

from __future__ import annotations

import threading

import pytest
from ops_support import label_values, parse_exposition, sample

from rodit.ops.metrics import (
    CONTENT_TYPE,
    OTHER,
    OVERFLOW,
    MetricsRegistry,
    format_value,
    safe_label_value,
)


def test_counter_gauge_histogram_exposition():
    r = MetricsRegistry()
    c = r.counter("rodit_things_total", "东西的数量", ["shop_id"])
    g = r.gauge("rodit_level", "当前水位")
    h = r.histogram("rodit_latency_seconds", "耗时", ["stage"], buckets=[0.5, 1, 5])
    c.inc({"shop_id": "s_1001"})
    c.inc({"shop_id": "s_1001"}, 2)
    c.inc({"shop_id": "s_2002"})
    g.set(value=3.5)
    for v in (0.2, 0.7, 0.9, 4.0, 12.0):
        h.observe({"stage": "total"}, v)

    text = r.render()
    assert text.endswith("\n")
    assert "# HELP rodit_things_total 东西的数量" in text
    assert "# TYPE rodit_things_total counter" in text
    assert "# TYPE rodit_level gauge" in text
    assert "# TYPE rodit_latency_seconds histogram" in text
    s = parse_exposition(text)
    assert sample(s, "rodit_things_total", shop_id="s_1001") == 3
    assert sample(s, "rodit_things_total", shop_id="s_2002") == 1
    assert sample(s, "rodit_level") == 3.5
    # 桶是累计的，最后一个是 +Inf，等于 _count
    buckets = {
        le: sample(s, "rodit_latency_seconds_bucket", stage="total", le=le)
        for le in ("0.5", "1", "5", "+Inf")
    }
    assert buckets == {"0.5": 1, "1": 3, "5": 4, "+Inf": 5}
    assert sample(s, "rodit_latency_seconds_count", stage="total") == 5
    assert sample(s, "rodit_latency_seconds_sum", stage="total") == pytest.approx(17.8)
    assert CONTENT_TYPE.startswith("text/plain; version=0.0.4")


def test_histogram_snapshot_supports_percentiles():
    """P50 / P95 可以从桶里推出来（PromQL histogram_quantile 的做法）。"""
    r = MetricsRegistry()
    h = r.histogram("x_seconds", "x", buckets=[1, 2, 5])
    for v in [0.5] * 50 + [1.5] * 45 + [4.0] * 5:
        h.observe(value=v)
    cumulative, count, _ = h.snapshot()
    assert count == 100
    p50_bucket = next(b for b, c in zip(h.bounds, cumulative, strict=True) if c >= 50)
    p95_bucket = next(b for b, c in zip(h.bounds, cumulative, strict=True) if c >= 95)
    assert (p50_bucket, p95_bucket) == (1, 2)


@pytest.mark.parametrize(
    "value",
    [
        "北京市朝阳区建国路88号",  # 中文
        "13812345678",  # 手机号
        "has space",
        'quote"inside',
        "line\nbreak",
        "a" * 200,
        "user@example.com",
        "",
    ],
)
def test_unsafe_label_values_become_other(value):
    assert safe_label_value(value) == OTHER


@pytest.mark.parametrize(
    "value",
    ["s_1001", "product_spec", "tool:logistics", "/v1/shops/{shop_id}/standalone/ask", "200"],
)
def test_code_like_label_values_are_kept(value):
    assert safe_label_value(value) == value


def test_non_string_label_values():
    assert safe_label_value(None) == "none"
    assert safe_label_value(True) == "true"
    assert safe_label_value(404) == "404"
    assert safe_label_value(1.5) == OTHER
    assert safe_label_value({"a": 1}) == OTHER


def test_label_values_are_sanitized_in_output():
    r = MetricsRegistry()
    c = r.counter("rodit_x_total", "x", ["intent"])
    c.inc({"intent": "我想问一下 13812345678"})
    s = parse_exposition(r.render())
    assert label_values(s) == {OTHER}


def test_labels_must_match_declaration():
    r = MetricsRegistry()
    c = r.counter("rodit_x_total", "x", ["shop_id"])
    with pytest.raises(ValueError, match="标签"):
        c.inc({"shop": "s_1"})
    with pytest.raises(ValueError, match="只能增加"):
        c.inc({"shop_id": "s_1"}, -1)


@pytest.mark.parametrize("name", ["1bad", "has-dash", "", "trailing_newline\n"])
def test_invalid_metric_names(name):
    with pytest.raises(ValueError):
        MetricsRegistry().counter(name, "x")


@pytest.mark.parametrize("label", ["le", "__x", "bad-label"])
def test_invalid_label_names(label):
    with pytest.raises(ValueError):
        MetricsRegistry().counter("rodit_x_total", "x", [label])


def test_duplicate_registration_rejected():
    r = MetricsRegistry()
    r.counter("rodit_x_total", "x")
    with pytest.raises(ValueError, match="重复"):
        r.gauge("rodit_x_total", "x")


def test_series_cap_folds_into_overflow():
    r = MetricsRegistry(max_series=3)
    c = r.counter("rodit_x_total", "x", ["shop_id"])
    for i in range(10):
        c.inc({"shop_id": f"s_{i}"})
    assert c.series_count() == 4  # 3 个正常序列 + _overflow
    assert c.value({"shop_id": OVERFLOW}) == 7


def test_forget_removes_a_shops_series_everywhere():
    r = MetricsRegistry()
    a = r.counter("rodit_a_total", "a", ["shop_id", "kind"])
    b = r.gauge("rodit_b", "b", ["shop_id"])
    c = r.counter("rodit_c_total", "c", ["kind"])
    a.inc({"shop_id": "s_1", "kind": "x"})
    a.inc({"shop_id": "s_2", "kind": "x"})
    b.set({"shop_id": "s_1"}, 1)
    c.inc({"kind": "x"})
    assert r.forget("shop_id", "s_1") == 2
    s = parse_exposition(r.render())
    assert "s_1" not in label_values(s)
    assert sample(s, "rodit_a_total", shop_id="s_2", kind="x") == 1
    assert sample(s, "rodit_c_total", kind="x") == 1


def test_gauge_replace_drops_stale_series():
    r = MetricsRegistry()
    g = r.gauge("rodit_expiring", "x", ["shop_id"])
    g.replace([({"shop_id": "s_1"}, 1), ({"shop_id": "s_2"}, 1)])
    g.replace([({"shop_id": "s_2"}, 1)])
    s = parse_exposition(r.render())
    assert label_values(s) == {"s_2"}


def test_before_render_hook_failure_does_not_break_output(caplog):
    r = MetricsRegistry()
    g = r.gauge("rodit_x", "x")
    calls: list[int] = []

    def ok_hook() -> None:
        calls.append(1)
        g.set(value=len(calls))

    def bad_hook() -> None:
        raise RuntimeError("连接串 postgresql://u:p@h/db 不该出现")

    r.before_render(bad_hook)
    r.before_render(ok_hook)
    s = parse_exposition(r.render())
    assert sample(s, "rodit_x") == 1
    assert "postgresql" not in caplog.text  # 只记异常类型


def test_escaping_in_help_and_labels():
    r = MetricsRegistry()
    r.counter("rodit_x_total", "第一行\n第二行 \\ 反斜杠")
    text = r.render()
    assert "# HELP rodit_x_total 第一行\\n第二行 \\\\ 反斜杠" in text


def test_format_value():
    assert format_value(3) == "3"
    assert format_value(2.5) == "2.5"
    assert format_value(float("inf")) == "+Inf"
    assert format_value(float("-inf")) == "-Inf"
    assert format_value(float("nan")) == "NaN"


def test_concurrent_increments_are_not_lost():
    r = MetricsRegistry()
    c = r.counter("rodit_x_total", "x", ["shop_id"])
    h = r.histogram("rodit_y_seconds", "y", buckets=[1])

    def work() -> None:
        for _ in range(1000):
            c.inc({"shop_id": "s_1"})
            h.observe(value=0.5)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.value({"shop_id": "s_1"}) == 8000
    assert h.snapshot()[1] == 8000
