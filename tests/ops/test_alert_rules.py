"""Prometheus 告警和记录规则（deploy/prometheus，方案 §12.4 可观测性、§16 统一监控）的静态检查：

- 两个 YAML 都能解析，结构对（groups → rules，每条要么 record 要么 alert）；
- 必需的告警都在：授权即将过期 / 已过期、串店、端到端 P95 > 5 秒、平台接口和大模型失败率、
  降级激增、不就绪、授权检查出错；每条都有 severity 和中文说明；
- 表达式里引用的每个指标都登记在 rodit.ops.metrics 的登记表里（RoditMetrics 实际输出的样本名，
  直方图含 _bucket / _sum / _count），标签选择器里的标签名是该指标声明过的；引用的记录规则都有定义；
  除了指标名、记录规则名，其余标识符只能是 PromQL 的函数和关键字（写错一个字母就报出来）；
- 机器上有 promtool 时再跑一遍 promtool check rules（没有就跳过这一项）。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from rodit.core.enums import Platform
from rodit.core.errors import TOOL_FAILURE_REASONS
from rodit.ops.instrument import RoditMetrics

ROOT = Path(__file__).resolve().parents[2]
PROM_DIR = ROOT / "deploy" / "prometheus"
ALERTS = PROM_DIR / "alerts.yml"
RECORDING = PROM_DIR / "recording.yml"

REQUIRED_ALERTS = {
    "RoditShopAuthExpiring",  # 授权即将过期
    "RoditShopAuthExpired",  # 授权已过期
    "RoditTenantIsolationBreach",  # 任何串店
    "RoditTenantIsolationRejected",
    "RoditEndToEndLatencyHigh",  # 端到端 P95 > 5 秒
    "RoditPlatformToolFailureRateHigh",  # 平台接口失败率
    "RoditPlatformToolThrottled",
    "RoditLLMFailureRateHigh",  # 大模型失败率
    "RoditDegradationSpike",  # 降级激增
    "RoditNotReady",  # 就绪失败
    "RoditScrapeDown",  # 抓不到指标（进程一直重启、/metrics 超时）
    "RoditAuthCheckErrors",  # 授权检查出错
    "RoditBackgroundTaskFailing",  # serve 的后台任务（续期、过期清理……）出错
}

PROMQL_FUNCTIONS = {
    "abs", "absent", "absent_over_time", "avg", "avg_over_time", "ceil", "changes", "clamp",
    "clamp_max", "clamp_min", "count", "count_over_time", "delta", "deriv", "floor",
    "histogram_quantile", "increase", "irate", "label_replace", "max", "max_over_time", "min",
    "min_over_time", "quantile", "quantile_over_time", "rate", "resets", "round", "scalar",
    "sort", "sum", "sum_over_time", "time", "topk", "bottomk", "vector",
}  # fmt: skip
PROMQL_KEYWORDS = {
    "and", "or", "unless", "offset", "bool", "by", "without", "on", "ignoring", "group_left",
    "group_right",
}  # fmt: skip
GROUPING = re.compile(r"\b(by|without|on|ignoring|group_left|group_right)\s*\(([^)]*)\)")
SELECTOR = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)\s*\{([^}]*)\}")
MATCHER = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(=~|!~|!=|=)\s*"((?:[^"\\]|\\.)*)"\s*,?')
IDENT = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
TARGET_LABELS = {"job", "instance"}  # Prometheus 抓取时加的目标标签
TARGET_METRICS = {"up"}  # Prometheus 自己给每个抓取目标生成的序列（只有目标标签）


def _load(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and set(data) == {"groups"}, path
    rules: list[dict[str, Any]] = []
    for group in data["groups"]:
        assert set(group) <= {"name", "interval", "rules"} and group["name"].startswith("rodit.")
        assert group["rules"], group["name"]
        rules.extend(group["rules"])
    return rules


@pytest.fixture(scope="module")
def alerts() -> list[dict[str, Any]]:
    return _load(ALERTS)


@pytest.fixture(scope="module")
def recordings() -> list[dict[str, Any]]:
    return _load(RECORDING)


@pytest.fixture(scope="module")
def registered() -> dict[str, tuple[str, ...]]:
    """rodit.ops.metrics 登记表里的样本名 → 标签名（部署入口用的就是这一份）。"""
    metrics = RoditMetrics(platform=Platform.TAOBAO, mode="platform", version="0.0.1")
    return metrics.registry.sample_names()


def _references(expr: str) -> tuple[list[tuple[str, dict[str, str]]], set[str], set[str]]:
    """PromQL 表达式 → ([(选择器里的指标名, {标签: 值})], 其余标识符, 聚合 / 匹配用到的标签)。"""
    selectors = [
        (name, {m.group(1): m.group(3) for m in MATCHER.finditer(body)})
        for name, body in SELECTOR.findall(expr)
    ]
    for name, body in SELECTOR.findall(expr):
        consumed = "".join(m.group(0) for m in MATCHER.finditer(body))
        assert consumed.strip() == body.strip(), f"标签选择器写得不对：{name}{{{body}}}"
    grouping = {
        label.strip()
        for _, labels in GROUPING.findall(expr)
        for label in labels.split(",")
        if label.strip()
    }
    stripped = SELECTOR.sub(lambda m: f" {m.group(1)} ", expr)
    stripped = GROUPING.sub(" ", stripped)
    stripped = re.sub(r"\[[^\]]*\]", " ", stripped)  # 时间范围 [5m]
    stripped = re.sub(r'"[^"]*"', " ", stripped)
    stripped = re.sub(r"\b\d+(?:\.\d+)?[smhdwy]?\b", " ", stripped)  # 数字、offset 的时长
    idents = set(IDENT.findall(stripped))
    return selectors, idents, grouping


def _check_expr(
    expr: str, registered: dict[str, tuple[str, ...]], records: set[str], where: str
) -> set[str]:
    """核对一条表达式，返回它引用的 rodit_* 指标名。"""
    selectors, idents, grouping = _references(expr)
    used: set[str] = set()
    for name, matchers in selectors:
        if name.startswith("rodit_"):
            assert name in registered, f"{where}：{name} 没有登记"
            unknown = set(matchers) - set(registered[name])
            assert not unknown, f"{where}：{name} 没有标签 {sorted(unknown)}"
        elif name in TARGET_METRICS:
            unknown = set(matchers) - TARGET_LABELS
            assert not unknown, f"{where}：{name} 没有标签 {sorted(unknown)}"
        else:
            assert name in records, f"{where}：{name} 不是登记的指标，也不是记录规则"
    for ident in idents:
        if ident in PROMQL_KEYWORDS:
            continue
        # 函数调用：f(...)；聚合还可以写成 sum by (标签) (...)
        followed_by_call = re.search(
            rf"(?<![\w:]){re.escape(ident)}\s*(?:\(|(?:by|without)\s*\()", expr
        )
        if ident in PROMQL_FUNCTIONS and followed_by_call:
            continue
        if ident.startswith("rodit_"):
            assert ident in registered, f"{where}：{ident} 没有登记"
            used.add(ident)
        elif ident in TARGET_METRICS:
            continue
        elif ident.startswith("rodit:"):
            assert ident in records, f"{where}：记录规则 {ident} 没有定义"
        else:
            raise AssertionError(f"{where}：不认识的标识符 {ident!r}")
    used |= {name for name, _ in selectors if name.startswith("rodit_")}
    all_labels = {label for labels in registered.values() for label in labels} | TARGET_LABELS
    assert grouping <= all_labels, f"{where}：聚合用到了没有的标签 {sorted(grouping - all_labels)}"
    return used


# ================================================================ 结构


def test_required_alerts_exist_with_severity_and_description(alerts):
    names = [r["alert"] for r in alerts]
    assert len(names) == len(set(names)), "告警名重复"
    missing = REQUIRED_ALERTS - set(names)
    assert not missing, f"缺少告警：{sorted(missing)}"
    for rule in alerts:
        assert set(rule) <= {"alert", "expr", "for", "labels", "annotations"}, rule["alert"]
        assert rule["labels"]["severity"] in {"critical", "warning"}, rule["alert"]
        assert rule["annotations"]["summary"].strip(), rule["alert"]
        assert rule["annotations"]["description"].strip(), rule["alert"]
        assert re.search(r"[一-鿿]", rule["annotations"]["summary"]), "说明用中文"
        if "for" in rule:
            assert re.fullmatch(r"\d+[smh]", rule["for"]), rule["alert"]
    critical = {r["alert"] for r in alerts if r["labels"]["severity"] == "critical"}
    assert {
        "RoditTenantIsolationBreach",
        "RoditShopAuthExpired",
        "RoditNotReady",
        "RoditScrapeDown",
    } <= critical


def test_recording_rule_names_follow_the_convention(recordings):
    names = [r["record"] for r in recordings]
    assert len(names) == len(set(names))
    for rule in recordings:
        assert set(rule) == {"record", "expr"}, rule
        assert re.fullmatch(r"rodit:[a-z0-9_]+:[a-z0-9_]+", rule["record"]), rule["record"]


# ================================================================ 引用的指标都已登记


def test_every_referenced_metric_is_registered(alerts, recordings, registered):
    records = {r["record"] for r in recordings}
    used: set[str] = set()
    for rule in recordings:
        used |= _check_expr(rule["expr"], registered, records, rule["record"])
    for rule in alerts:
        used |= _check_expr(rule["expr"], registered, records, rule["alert"])
    # 本轨新增的指标都有规则用到
    for name in (
        "rodit_tenant_isolation_total",
        "rodit_tool_cache_tenant_violations_total",
        "rodit_tool_cache_hits_total",
        "rodit_tool_cache_stale_served_total",
        "rodit_auth_expired_shops",
        "rodit_ready",
        "rodit_engine_latency_seconds_bucket",
        "rodit_platform_tool_calls_total",
        "rodit_llm_calls_total",
        "rodit_degradations_total",
        "rodit_auth_expiry_check_errors_total",
        "rodit_background_task_runs_total",
    ):
        assert name in used, name


def test_the_checker_catches_typos(registered):
    """核对本身要能报错：写错指标名、标签名、函数名都不能放过。"""
    with pytest.raises(AssertionError, match="没有登记"):
        _check_expr("rate(rodit_llm_call_total[5m])", registered, set(), "typo")
    with pytest.raises(AssertionError, match="没有标签"):
        _check_expr('rodit_llm_calls_total{shop="x"}', registered, set(), "label")
    with pytest.raises(AssertionError, match="不认识"):
        _check_expr("rat(rodit_llm_calls_total[5m])", registered, set(), "func")
    with pytest.raises(AssertionError, match="没有定义"):
        _check_expr("rodit:nope:rate5m > 1", registered, set(), "record")


def test_tool_failure_rule_covers_every_failure_reason(recordings):
    """失败率的 outcome 正好是 ToolUnavailable 的全部原因加上 error（其他异常）。"""
    (rule,) = [r for r in recordings if r["record"] == "rodit:platform_tool_failure_ratio:rate10m"]
    (pattern,) = re.findall(r'outcome=~"([^"]+)"', rule["expr"])
    assert set(pattern.split("|")) == {*TOOL_FAILURE_REASONS, "error"}


def test_latency_alert_uses_the_5s_target_and_since_received(alerts, recordings):
    (rule,) = [r for r in alerts if r["alert"] == "RoditEndToEndLatencyHigh"]
    assert rule["expr"].strip().endswith("> 5")
    assert "since_received" in rule["expr"]
    (p95,) = [r for r in recordings if r["record"] == "rodit:engine_latency_seconds:p95_5m"]
    assert "histogram_quantile(0.95" in p95["expr"]


# ================================================================ promtool


@pytest.mark.skipif(shutil.which("promtool") is None, reason="没有 promtool，只做静态检查")
def test_promtool_check_rules():
    promtool = shutil.which("promtool")
    assert promtool is not None
    proc = subprocess.run(
        [promtool, "check", "rules", str(RECORDING), str(ALERTS)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


RULE_TESTS = Path(__file__).with_name("prometheus_rules_test.yml")
"""promtool 的告警单元测试：合成的指标序列 → 某个时刻正在触发的告警（ALERTS）。"""


def test_rule_unit_tests_cover_every_required_alert():
    """不管有没有 promtool，单元测试文件本身要覆盖每个必需的告警，也要有"一切正常没有告警"。"""
    data = yaml.safe_load(RULE_TESTS.read_text(encoding="utf-8"))
    assert data["rule_files"] == [
        "../../deploy/prometheus/recording.yml",
        "../../deploy/prometheus/alerts.yml",
    ]
    fired: set[str] = set()
    quiet = 0
    for case in data["tests"]:
        for check in case["promql_expr_test"]:
            samples = check["exp_samples"]
            quiet += not samples
            for s in samples:
                fired |= set(re.findall(r'alertname="([A-Za-z]+)"', s["labels"]))
    assert REQUIRED_ALERTS <= fired, sorted(REQUIRED_ALERTS - fired)
    assert quiet >= 2


@pytest.mark.skipif(shutil.which("promtool") is None, reason="没有 promtool，只做静态检查")
def test_promtool_rule_unit_tests():
    promtool = shutil.which("promtool")
    assert promtool is not None
    proc = subprocess.run(
        [promtool, "test", "rules", str(RULE_TESTS)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
