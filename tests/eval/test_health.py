"""评测的运行健康度（rodit.eval.health，docs/架构与分工.md §4.4）：报告里 health 这一块的结构，
和判定"这一轮算不算评到了模型"的规则（allowed / assess / error_label）。

纯数据和纯函数：不跑评测、不联网，计数都是手造的（324 = 评测集全部，66 = 测试集）。
并行期间别的任务只能给这几个模型加带默认值的字段，所以这里按字段核对，不比较整个 dump。
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from rodit.eval import health
from rodit.eval.health import (
    DEFAULT_CEILING,
    LOST_VOTE_CEILING,
    HealthCounts,
    PurposeHealth,
    RunHealth,
    allowed,
    assess,
    error_label,
    json_object_count,
)

ROUTE_EXAMPLE = "路由：324 条里有 105 条（32.4%）没用上判断层（退回关键词规则），上限 6 条。"


def _counts(**fields: int) -> HealthCounts:
    """全量评测的一组计数：324 条，153 条送过质检，304 条调过生成。"""
    return HealthCounts(**{"cases": 324, "qa_judged": 153, "generation_attempted": 304, **fields})


def _has(model: BaseModel, expected: dict[str, Any]) -> bool:
    dumped = model.model_dump()
    return {key: dumped.get(key) for key in expected} == expected


# ================================================================ 模块和数据结构


def test_module_depends_only_on_pydantic_and_the_standard_library():
    tree = ast.parse(Path(health.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module is not None  # 不用相对导入
            roots.add(node.module.split(".")[0])
    assert "pydantic" in roots and "rodit" not in roots
    assert roots - {"pydantic"} <= sys.stdlib_module_names


def test_ceilings():
    assert DEFAULT_CEILING == 0.02 and LOST_VOTE_CEILING == 0.10


def test_models_have_the_agreed_fields_and_defaults():
    assert _has(
        PurposeHealth(),
        {
            "calls": 0,
            "failed": 0,
            "timeouts": 0,
            "truncated": 0,
            "empty": 0,
            "unusable": 0,
            "errors": {},
        },
    )
    assert _has(
        HealthCounts(cases=324),
        {
            "cases": 324,
            "route_fallback": 0,
            "route_lost_votes": 0,
            "qa_judged": 0,
            "qa_unavailable": 0,
            "generation_attempted": 0,
            "generation_unusable": 0,
            "engine_errors": 0,
            "budget_cut": 0,
            "budget_unjudged": 0,
        },
    )
    minimal = RunHealth(ceiling=DEFAULT_CEILING, overall=HealthCounts(cases=324), valid=True)
    assert _has(
        minimal,
        {"ceiling": 0.02, "test": None, "by_purpose": {}, "routed_by": {}, "late_calls": 0},
    )
    assert minimal.valid is True and minimal.reasons == []
    assert minimal.overall == HealthCounts(cases=324)
    # 可变默认值不共用
    other = RunHealth(ceiling=DEFAULT_CEILING, overall=HealthCounts(cases=324), valid=True)
    assert PurposeHealth().errors is not PurposeHealth().errors
    assert minimal.by_purpose is not other.by_purpose and minimal.reasons is not other.reasons


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (HealthCounts, {}),  # cases 必填
        (RunHealth, {"overall": {"cases": 1}, "valid": True}),  # ceiling 必填
        (RunHealth, {"ceiling": 0.02, "valid": True}),  # overall 必填
        (RunHealth, {"ceiling": 0.02, "overall": {"cases": 1}}),  # valid 必填
    ],
)
def test_required_fields(model: type[BaseModel], data: dict[str, Any]):
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (PurposeHealth, {"calls": 1, "not_a_field": 1}),
        (HealthCounts, {"cases": 1, "not_a_field": 1}),
        (
            RunHealth,
            {"ceiling": 0.02, "overall": {"cases": 1}, "valid": True, "not_a_field": 1},
        ),
        (RunHealth, {"ceiling": 0.02, "overall": {"cases": 1, "not_a_field": 1}, "valid": True}),
    ],
)
def test_unknown_fields_are_rejected(model: type[BaseModel], data: dict[str, Any]):
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (PurposeHealth, {"failed": -1}),
        (HealthCounts, {"cases": -1}),
        (HealthCounts, {"cases": 1, "route_fallback": -1}),
        (RunHealth, {"ceiling": -0.01, "overall": {"cases": 1}, "valid": True}),
        (RunHealth, {"ceiling": 1.01, "overall": {"cases": 1}, "valid": True}),  # 比例，不是百分数
        (RunHealth, {"ceiling": float("nan"), "overall": {"cases": 1}, "valid": True}),
        (RunHealth, {"ceiling": 0.02, "overall": {"cases": 1}, "valid": True, "late_calls": -1}),
    ],
)
def test_counts_cannot_be_negative_and_the_ceiling_is_a_ratio(
    model: type[BaseModel], data: dict[str, Any]
):
    with pytest.raises(ValidationError):
        model.model_validate(data)


def test_run_health_round_trips_through_report_json():
    overall = _counts(route_fallback=105, route_lost_votes=40, budget_cut=3)
    test = HealthCounts(cases=66, route_fallback=20, qa_judged=31, generation_attempted=60)
    reasons = assess(overall, test)
    run = RunHealth(
        ceiling=DEFAULT_CEILING,
        overall=overall,
        test=test,
        by_purpose={
            "route": PurposeHealth(
                calls=972,
                failed=345,
                timeouts=300,
                truncated=4,
                empty=2,
                unusable=6,
                errors={"timeout": 300, "http_429": 45},
            ),
            "qa": PurposeHealth(calls=153),
            "generate": PurposeHealth(calls=462, unusable=3),
        },
        routed_by={"incloud_llm": 219, "rules": 104, "none": 1},
        late_calls=12,
        valid=not reasons,
        reasons=reasons,
    )
    assert run.valid is False and run.reasons[0] == ROUTE_EXAMPLE
    payload = json.dumps(run.model_dump(mode="json"), ensure_ascii=False)
    assert RunHealth.model_validate(json.loads(payload)) == run
    # 读报告的一方（record-eval）拿同样的计数重新算，结论一样
    again = RunHealth.model_validate(json.loads(payload))
    assert assess(again.overall, again.test) == run.reasons


# ================================================================ allowed


@pytest.mark.parametrize(
    ("n", "ceiling", "expected"),
    [
        (0, 0.02, 0),  # 没有用例：一条都不容许
        (-3, 0.02, 0),
        (1, 0.02, 1),  # 至少容许 1 条
        (49, 0.02, 1),
        (50, 0.02, 1),
        (99, 0.02, 1),
        (100, 0.02, 2),
        (149, 0.02, 2),
        (150, 0.02, 3),
        (66, 0.02, 1),  # 测试集
        (258, 0.02, 5),  # 开发集
        (324, 0.02, 6),  # 全部
        (9, 0.10, 1),
        (10, 0.10, 1),
        (20, 0.10, 2),
        (66, 0.10, 6),
        (324, 0.10, 32),
        (324, 0.0, 1),  # 上限是 0 也容许 1 条
        (5, 0.5, 2),
        (324, 1.0, 324),
    ],
)
def test_allowed(n: int, ceiling: float, expected: int):
    assert allowed(n, ceiling) == expected


def test_allowed_is_exact_for_the_two_ceilings_in_use():
    """浮点乘法不会让默认的两个上限少算一条（0.02、0.10 的二进制近似值都略大于本身）。"""
    for n in range(1, 20001):
        assert allowed(n, DEFAULT_CEILING) == max(1, n * 2 // 100), n
        assert allowed(n, LOST_VOTE_CEILING) == max(1, n * 10 // 100), n


# ================================================================ assess


def test_a_clean_run_is_valid():
    assert assess(_counts()) == []
    assert assess(_counts(), None) == []
    assert assess(_counts(), HealthCounts(cases=66, qa_judged=31, generation_attempted=60)) == []


def test_the_documented_example_sentence():
    assert assess(HealthCounts(cases=324, route_fallback=105)) == [ROUTE_EXAMPLE]


@pytest.mark.parametrize(
    ("field", "limit", "sentence"),
    [
        (
            "route_fallback",
            6,
            "路由：324 条里有 7 条（2.2%）没用上判断层（退回关键词规则），上限 6 条。",
        ),
        (
            "qa_unavailable",
            3,
            "质检：有草稿送检的 153 条里有 4 条（2.6%）"
            "质检判断层不可用（只跑了硬规则），上限 3 条。",
        ),
        (
            "generation_unusable",
            6,
            "生成：调过大模型生成的 304 条里有 7 条（2.3%）"
            "没有大模型草稿送到质检（生成调用失败，或输出解析不了），上限 6 条。",
        ),
        ("engine_errors", 6, "引擎：324 条里有 7 条（2.2%）引擎异常，上限 6 条。"),
        (
            "budget_unjudged",
            6,
            "时间预算：324 条里有 7 条（2.2%）被时间预算截断、没有大模型草稿过完质检，上限 6 条。",
        ),
        (
            "route_lost_votes",
            32,
            "路由：324 条里有 33 条（10.2%）"
            "至少有一票没算上（调用失败、被截断、读不出），上限 32 条。",
        ),
    ],
)
def test_each_rule_fires_just_above_its_ceiling(field: str, limit: int, sentence: str):
    assert assess(_counts(**{field: limit})) == []  # 正好在上限：不算问题
    assert assess(_counts(**{field: limit + 1})) == [sentence]


def test_reasons_come_in_a_fixed_order():
    reasons = assess(
        _counts(
            route_lost_votes=40,
            budget_unjudged=8,
            engine_errors=9,
            generation_unusable=30,
            qa_unavailable=20,
            route_fallback=105,
        )
    )
    assert [r.split("（")[0] for r in reasons] == [
        "路由：324 条里有 105 条",
        "质检：有草稿送检的 153 条里有 20 条",
        "生成：调过大模型生成的 304 条里有 30 条",
        "引擎：324 条里有 9 条",
        "时间预算：324 条里有 8 条",
        "路由：324 条里有 40 条",
    ]
    assert all(r.endswith(" 条。") and "上限 " in r and "%）" in r for r in reasons)


def test_the_test_split_is_judged_separately_and_prefixed():
    overall = _counts(route_fallback=6)  # 全部用例没超上限
    test = HealthCounts(cases=66, route_fallback=2, qa_judged=31, generation_attempted=60)
    in_test = "测试集：路由：66 条里有 2 条（3.0%）没用上判断层（退回关键词规则），上限 1 条。"
    assert assess(overall, test) == [in_test]
    assert assess(overall) == []
    # 两边都超：先全部、后测试集
    assert assess(_counts(route_fallback=105), test) == [ROUTE_EXAMPLE, in_test]
    # 测试集每条规则都看
    every = HealthCounts(
        cases=66,
        route_fallback=2,
        route_lost_votes=7,
        qa_judged=31,
        qa_unavailable=2,
        generation_attempted=60,
        generation_unusable=2,
        engine_errors=2,
        budget_unjudged=2,
    )
    reasons = assess(_counts(), every)
    assert len(reasons) == 6 and all(r.startswith("测试集：") for r in reasons)
    assert reasons[1] == (
        "测试集：质检：有草稿送检的 31 条里有 2 条（6.5%）"
        "质检判断层不可用（只跑了硬规则），上限 1 条。"
    )
    assert reasons[4] == (
        "测试集：时间预算：66 条里有 2 条（3.0%）"
        "被时间预算截断、没有大模型草稿过完质检，上限 1 条。"
    )
    assert reasons[5] == (
        "测试集：路由：66 条里有 7 条（10.6%）"
        "至少有一票没算上（调用失败、被截断、读不出），上限 6 条。"
    )


def test_lost_votes_have_their_own_looser_ceiling():
    assert assess(_counts(route_lost_votes=32)) == []  # 退回关键词规则的话 7 条就超了
    assert len(assess(_counts(route_fallback=32))) == 1
    # 两个上限各管各的
    assert assess(_counts(route_fallback=16), ceiling=0.05) == []
    assert len(assess(_counts(route_fallback=17), ceiling=0.05)) == 1
    assert len(assess(_counts(route_lost_votes=33), ceiling=0.5)) == 1
    assert len(assess(_counts(route_fallback=7), lost_vote_ceiling=0.5)) == 1
    assert assess(_counts(route_lost_votes=6), lost_vote_ceiling=0.02) == []
    assert assess(_counts(route_lost_votes=7), lost_vote_ceiling=0.02) == [
        "路由：324 条里有 7 条（2.2%）至少有一票没算上（调用失败、被截断、读不出），上限 6 条。"
    ]
    assert assess(_counts(), HealthCounts(cases=66, route_lost_votes=6)) == []


def test_small_runs_tolerate_one_case():
    assert assess(HealthCounts(cases=4, route_fallback=1, engine_errors=1)) == []
    assert assess(HealthCounts(cases=4, route_fallback=2)) == [
        "路由：4 条里有 2 条（50.0%）没用上判断层（退回关键词规则），上限 1 条。"
    ]


def test_zero_denominators():
    # 没有草稿送检、没调过生成：分母是 0、计数也是 0，不算问题
    assert assess(HealthCounts(cases=10)) == []
    assert assess(HealthCounts(cases=0)) == []
    assert assess(HealthCounts(cases=10), HealthCounts(cases=0)) == []
    # 分母是 0、计数却不是 0：一条都不容许，照样报出来，占比写不出就写"—"
    assert assess(HealthCounts(cases=10, qa_unavailable=2)) == [
        "质检：有草稿送检的 0 条里有 2 条（—）质检判断层不可用（只跑了硬规则），上限 0 条。"
    ]
    assert assess(HealthCounts(cases=10, generation_unusable=1)) == [
        "生成：调过大模型生成的 0 条里有 1 条（—）"
        "没有大模型草稿送到质检（生成调用失败，或输出解析不了），上限 0 条。"
    ]
    assert assess(HealthCounts(cases=0, route_fallback=1, engine_errors=1, route_lost_votes=1)) == [
        "路由：0 条里有 1 条（—）没用上判断层（退回关键词规则），上限 0 条。",
        "引擎：0 条里有 1 条（—）引擎异常，上限 0 条。",
        "路由：0 条里有 1 条（—）至少有一票没算上（调用失败、被截断、读不出），上限 0 条。",
    ]


def test_budget_cuts_alone_never_invalidate_a_run():
    """时间预算截断是 §12.1 的正常行为：照实展示，本身不算"模型没被评到"。"""
    assert assess(_counts(budget_cut=324), HealthCounts(cases=66, budget_cut=66)) == []
    assert assess(_counts(budget_cut=324), ceiling=0.0, lost_vote_ceiling=0.0) == []


def test_budget_cuts_before_any_judged_draft_do():
    """截在生成或质检完成之前、没有一条大模型草稿判完的用例，量到的不是模型的回答：超过上限不算数。
    没调过生成、没有草稿送检时别的分母都是 0，只有这一条规则看得见。"""
    silent = _counts(budget_cut=324, budget_unjudged=324, qa_judged=0, generation_attempted=0)
    assert assess(silent) == [
        "时间预算：324 条里有 324 条（100.0%）被时间预算截断、没有大模型草稿过完质检，上限 6 条。"
    ]
    assert assess(_counts(budget_cut=40, budget_unjudged=6)) == []  # 其余是重写被截断：不算
    test = HealthCounts(cases=66, budget_cut=3, budget_unjudged=2)
    assert assess(_counts(), test) == [
        "测试集：时间预算：66 条里有 2 条（3.0%）"
        "被时间预算截断、没有大模型草稿过完质检，上限 1 条。"
    ]


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ('{"intent": {"choice": "stock"}, "emotion": {"score": 0}}', 1),  # 嵌套的不单独算
        ('好的，结果如下：\n```json\n{"answer": "亲，有货"}\n```', 1),
        ('{"answer": "尺码表里写着 {M: 100-115 斤}"}', 1),  # 字符串里的花括号不算
        ('格式是 {问题名: {...}}，答案：{"answer": "好"}', 1),  # 解析不了的花括号跳过
        ('{"intent": {"choice": "other"}}\n最终答案：{"intent": {"choice": "stock"}}', 2),
        ('<think>先写个草稿 {"answer": "草稿"}</think>{"answer": "亲，有货"}', 2),
        ('[{"answer": "一"}, {"answer": "二"}]', 2),
        ("{} 和 {}", 0),  # 空对象不算
        ("只有文字，没有 JSON", 0),
        ("", 0),
        ('{"answer": "没写完', 0),
    ],
)
def test_json_object_count(text: str, count: int):
    assert json_object_count(text) == count


# ================================================================ error_label


@pytest.mark.parametrize(
    ("detail", "label"),
    [
        # 超时：优先，不管有没有状态码
        ({"error": "LLMTimeout", "kind": "timeout"}, "timeout"),
        ({"error": "LLMTimeout", "kind": "timeout", "status": 504, "code": "gateway"}, "timeout"),
        # 有状态码：http_<状态码>，有 code 再接上；状态码优先于别的 kind
        ({"error": "LLMError", "kind": "rate_limit", "status": 429}, "http_429"),
        (
            {"error": "LLMError", "kind": "bad_request", "status": 400, "code": "bad_param"},
            "http_400:bad_param",
        ),
        ({"error": "LLMError", "status": 402, "code": 1113}, "http_402:1113"),  # 数字代码
        ({"error": "LLMError", "status": 500, "code": "  "}, "http_500"),
        ({"error": "LLMError", "status": 503, "code": None}, "http_503"),
        # 状态码是 2xx 的失败（HTTP 200 但读不出回答、或响应体里带着错误对象）：按 kind 标，
        # 有 code 接上；连 kind 都没有才退回 http_<状态码>
        ({"error": "LLMError", "kind": "malformed", "status": 200}, "malformed"),
        (
            {"error": "LLMError", "kind": "malformed", "status": 200, "code": "1301"},
            "malformed:1301",
        ),
        ({"error": "LLMError", "kind": "content_filter", "status": 200}, "content_filter"),
        ({"error": "LLMError", "status": 200}, "http_200"),
        ({"error": "LLMError", "status": 204, "code": "x"}, "http_204:x"),
        # 没有状态码：别的 kind（code 不单独成键）
        ({"error": "LLMError", "kind": "network"}, "network"),
        ({"error": "LLMError", "kind": "malformed", "code": "no_choices"}, "malformed"),
        ({"error": "LLMError", "kind": "config"}, "config"),
        # 什么原因都没带（还没有 kind / status 的旧事件）：异常类型名
        ({"error": "LLMError"}, "LLMError"),
        ({"error": "LLMTimeout"}, "LLMTimeout"),
        ({"error": "LLMError", "status": True}, "LLMError"),  # 布尔不是状态码
        ({"error": "LLMError", "status": "429"}, "LLMError"),  # 字符串也不是
        # 连异常类型名都没有
        ({"purpose": "route", "latency_ms": 12}, "unknown"),
        ({"error": ""}, "unknown"),
        ({"error": None, "kind": None, "status": None, "code": None}, "unknown"),
        ({}, "unknown"),
    ],
)
def test_error_label(detail: dict[str, Any], label: str):
    assert error_label(detail) == label
