"""评测跑完之后交出去的东西（python -m rodit.eval）：不算数的一轮不覆盖基线和已有的报告、
基线自己要是有效评到模型的一轮、和基线没有共同用例在开跑前就拦下、中止和退出时的说明按失败原因给。

模型一律是脚本化的假模型（FakeLLMClient），全部合成数据，不联网。
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import rodit.eval.cli as cli
from rodit.app.offline import EvidenceEchoLLM
from rodit.core.enums import Intent
from rodit.core.errors import LLMError
from rodit.core.pii import mask_pii
from rodit.core.ports import LLMRequest
from rodit.core.questions import (
    Q_ANSWERS_QUESTION,
    Q_EMOTION,
    Q_GROUNDED,
    Q_INTENT,
    Q_MULTI_INTENT,
    Q_NEED_HUMAN,
    Q_PLATFORM_VIOLATION,
    Q_PROMISE_RISK,
    Q_TONE,
)
from rodit.eval.cli import (
    DEFAULTS_WARNING,
    STOP_AFTER_FAILED_CASES,
    _abort_hint,
    _label_kinds,
    build_parser,
    main,
)
from rodit.eval.dataset import CASES, cases
from rodit.eval.runner import select_cases
from rodit.testing.fakes import FakeLLMClient

MODEL = "synthetic-llm"
SAMPLE = ["--limit", "12", "--seed", "3"]
ARGV = ["--llm", "env", "--skip-preflight", *SAMPLE]
WITH_PREFLIGHT = ["--llm", "env", *SAMPLE]
PICKED = select_cases(cases("all"), limit=12, seed=3)
EXPECTED = {mask_pii(c.text): c.expected_intent for c in CASES}

Fault = Callable[[LLMRequest, int], Any]
"""fault(请求, 这是第几次路由请求（从 1 数；不是路由请求时为 0）) → 代替正常回答的结果，None 照常。
一条用例发 3 次路由请求、用例一条接一条跑，所以"前 9 次路由请求"就是前 3 条用例。"""


class Vendor:
    """脚本化的模型服务：路由答用例标注的类型、质检全部通过、生成复述依据。"""

    def __init__(self, fault: Fault | None = None) -> None:
        self.fault = fault
        self.echo = EvidenceEchoLLM()
        self.routes = 0

    def __call__(self, request: LLMRequest) -> Any:
        if request.purpose == "other":  # 连通性检查
            return "OK"
        nth = 0
        if request.purpose == "route":
            self.routes += 1
            nth = self.routes
        if self.fault is not None:
            result = self.fault(request, nth)
            if result is not None:
                return result
        if request.purpose == "route":
            user = request.messages[-1].content
            head = "## 状态（state）\n"
            state = json.loads(user[user.index(head) + len(head) : user.index("\n\n## 问题")])
            intent = EXPECTED.get(state["messages"][-1]["text"], Intent.OTHER)
            return {
                Q_INTENT: {"choice": intent.value},
                Q_NEED_HUMAN: {"answer": "否", "p": 0.05},
                Q_EMOTION: {"score": 0},
                Q_MULTI_INTENT: {"answer": "否", "p": 0.05},
            }
        if request.purpose == "qa":
            return {
                Q_GROUNDED: {"answer": "是", "p": 0.95},
                Q_ANSWERS_QUESTION: {"answer": "是", "p": 0.95},
                Q_PROMISE_RISK: {"answer": "否", "p": 0.05},
                Q_PLATFORM_VIOLATION: {"answer": "否", "p": 0.02},
                Q_TONE: {"score": 3},
            }
        return self.echo.complete(request).text


def _rejected(status: int, kind: str) -> LLMError:
    return LLMError(f"合成的故障：HTTP {status}", status=status, kind=kind)  # type: ignore[arg-type]


def _routes_rejected(status: int, kind: str, first: int | None = None) -> Fault:
    """路由请求被拒：first 给了只拒前 first 次（3 次一条用例），没给就全拒。"""

    def fault(request: LLMRequest, nth: int):
        if request.purpose == "route" and (first is None or nth <= first):
            return _rejected(status, kind)
        return None

    return fault


def _run(argv: list[str], fault: Fault | None = None) -> tuple[int, str, str, FakeLLMClient]:
    client = FakeLLMClient(handler=Vendor(fault), model=MODEL)
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err, client_factory=lambda: client)
    return code, out.getvalue(), err.getvalue(), client


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_report(path: Path) -> str:
    """一份算数的 --llm env 报告，返回文件内容。"""
    code, _, err, _ = _run([*ARGV, "--json", str(path)])
    assert code == 0, err
    assert _load(path)["health"]["valid"] is True
    return path.read_text(encoding="utf-8")


# ================================================================ 不算数的一轮不覆盖已有的文件


def test_an_invalid_run_does_not_replace_the_baseline_it_was_compared_with(tmp_path):
    target = tmp_path / "llm.json"
    sibling = tmp_path / "llm.invalid.json"
    before = _valid_report(target)
    argv = [*ARGV, "--baseline", str(target), "--json", str(target)]

    code, out, err, _ = _run(argv, _routes_rejected(400, "bad_request"))
    assert code == 3
    assert target.read_text(encoding="utf-8") == before  # 基线一个字节都没动
    data = _load(sibling)
    assert data["health"]["valid"] is False and data["health"]["overall"]["route_fallback"] == 12
    assert data["baseline"]["common_cases"] == 12  # 比较照常做了，结果在旁边那份里
    assert f"不覆盖 {target}（它是这次的基线），完整结果改写到旁边的 {sibling}" in err
    last = out.strip().splitlines()[-1]
    assert last.startswith(f"完整结果（JSON）已写入 {sibling}（") and "没有覆盖" in last
    assert f"已写入 {target}" not in out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["llm.invalid.json", "llm.json"]

    # --allow-degraded 只改退出码：这份结果照样不算数，照样不覆盖
    sibling.unlink()
    code, _, err, _ = _run([*argv, "--allow-degraded"], _routes_rejected(400, "bad_request"))
    assert code == 1  # 路由全退回关键词规则，类型准确率下降
    assert (
        target.read_text(encoding="utf-8") == before and _load(sibling)["health"]["valid"] is False
    )
    assert f"完整结果改写到旁边的 {sibling}" in err

    # 算数的一轮照常把基线换成新的（旁边那份留着，不碰）
    code, out, _, _ = _run(argv)
    assert code == 0 and f"完整结果（JSON）已写入 {target}" in out and "没有覆盖" not in out
    fresh = _load(target)
    assert fresh["health"]["valid"] is True and fresh["baseline"]["passed"] is True
    assert sibling.exists()


def test_the_baseline_is_recognised_behind_another_spelling_of_its_path(tmp_path):
    """--json 写的是符号链接指的那个文件：它就是基线时一样不覆盖。"""
    real = tmp_path / "data" / "llm.json"
    real.parent.mkdir()
    before = _valid_report(real)
    link = tmp_path / "latest.json"
    link.symlink_to(real)
    code, _, err, _ = _run(
        [*ARGV, "--baseline", str(real), "--json", str(link)], _routes_rejected(400, "bad_request")
    )
    assert code == 3 and real.read_text(encoding="utf-8") == before and link.is_symlink()
    assert "它是这次的基线" in err
    assert _load(real.parent / "llm.invalid.json")["health"]["valid"] is False


def test_an_invalid_run_does_not_replace_an_existing_file(tmp_path):
    existing = tmp_path / "report.json"
    before = _valid_report(existing)
    code, out, err, _ = _run([*ARGV, "--json", str(existing)], _routes_rejected(400, "bad_request"))
    sibling = tmp_path / "report.invalid.json"
    assert code == 3 and existing.read_text(encoding="utf-8") == before
    assert _load(sibling)["health"]["valid"] is False
    assert f"不覆盖 {existing}（它已经存在），完整结果改写到旁边的 {sibling}" in err
    assert f"完整结果（JSON）已写入 {sibling}" in out
    # 没有后缀的文件名：后面接 .invalid
    bare = tmp_path / "result"
    bare.write_text("合成的旧内容", encoding="utf-8")
    code, _, _, _ = _run([*ARGV, "--json", str(bare)], _routes_rejected(400, "bad_request"))
    assert code == 3 and bare.read_text(encoding="utf-8") == "合成的旧内容"
    assert _load(tmp_path / "result.invalid")["health"]["valid"] is False


def test_an_invalid_run_still_writes_a_new_file_and_stdout(tmp_path):
    """没有东西可覆盖时照旧：新文件直接写，--json 不带路径照旧打到标准输出。"""
    fresh = tmp_path / "first.json"
    code, out, err, _ = _run([*ARGV, "--json", str(fresh)], _routes_rejected(400, "bad_request"))
    assert code == 3 and _load(fresh)["health"]["valid"] is False
    assert f"完整结果（JSON）已写入 {fresh}" in out and "不覆盖" not in err
    assert [p.name for p in tmp_path.iterdir()] == ["first.json"]
    code, out, err, _ = _run([*ARGV, "--json"], _routes_rejected(400, "bad_request"))
    assert code == 3 and json.loads(out)["health"]["valid"] is False and "不覆盖" not in err


def test_a_sibling_that_cannot_be_written_is_reported(tmp_path):
    target = tmp_path / "llm.json"
    before = _valid_report(target)
    (tmp_path / "llm.invalid.json").mkdir()  # 旁边那个名字被一个目录占了
    code, out, err, _ = _run([*ARGV, "--json", str(target)], _routes_rejected(400, "bad_request"))
    assert code == 2 and f"完整结果写不进 {tmp_path / 'llm.invalid.json'}：" in err
    assert target.read_text(encoding="utf-8") == before
    assert "== 总体" in out and "已写入" not in out  # 文本报告照样给出来


# ================================================================ 基线自己要是有效的一轮


def test_an_invalid_env_baseline_is_refused_before_any_call(tmp_path):
    base = tmp_path / "base.json"
    code, _, _, _ = _run(
        [*ARGV, "--json", str(base), "--allow-degraded"], _routes_rejected(400, "bad_request")
    )
    assert code == 0 and _load(base)["health"]["valid"] is False  # 兜底逻辑的数字，退出码 0
    code, out, err, client = _run([*WITH_PREFLIGHT, "--baseline", str(base)])
    assert code == 2 and out == "" and client.calls == []  # 连连通性检查都没有发
    assert f"评测没有运行：基线 {base} 不是一轮有效评到模型的结果" in err
    assert "路由：12 条里有 12 条（100.0%）没用上判断层（退回关键词规则），上限 1 条" in err
    assert "测试集：路由：" in err and "Traceback" not in err
    # 离线替身的这一轮同样不拿它当基线
    out, err = io.StringIO(), io.StringIO()
    assert main([*SAMPLE, "--baseline", str(base)], out=out, err=err) == 2
    assert out.getvalue() == "" and "不能当回归基线" in err.getvalue()


def test_an_env_baseline_without_a_health_block_is_refused(tmp_path):
    """以前的版本写的报告没有 health：看不出那一轮有多少用例走了兜底逻辑。"""
    base = tmp_path / "old.json"
    _valid_report(base)
    data = _load(base)
    del data["health"]
    base.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    code, out, err, client = _run([*WITH_PREFLIGHT, "--baseline", str(base)])
    assert code == 2 and out == "" and client.calls == []
    assert "没有运行健康度（health）" in err


def test_a_baseline_valid_only_under_a_raised_ceiling_is_refused(tmp_path):
    """基线那一轮用 --max-degraded 放宽了上限才算数：按默认上限重新判定，不信它的 valid。"""
    base = tmp_path / "loose.json"
    code, _, err, _ = _run(
        [*ARGV, "--json", str(base), "--max-degraded", "0.5"],
        _routes_rejected(400, "bad_request", first=9),  # 前 3 条用例退回关键词规则
    )
    assert code == 0, err
    health = _load(base)["health"]
    assert health["valid"] is True and health["overall"]["route_fallback"] == 3
    code, out, err, client = _run([*WITH_PREFLIGHT, "--baseline", str(base)])
    assert code == 2 and out == "" and client.calls == []
    assert "路由：12 条里有 3 条（25.0%）没用上判断层（退回关键词规则），上限 1 条" in err


def test_valid_env_and_offline_baselines_are_accepted(tmp_path):
    env_base = tmp_path / "env.json"
    _valid_report(env_base)
    code, out, err, _ = _run([*WITH_PREFLIGHT, "--baseline", str(env_base)])
    assert code == 0, err
    assert "== §13.5 回归比较（基线 env.json" in out and "→ 指标没有下降" in out
    # 以后的版本在计数里多写了字段：不因此读不了
    data = _load(env_base)
    data["health"]["overall"]["a_later_counter"] = 4
    env_base.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert _run([*ARGV, "--baseline", str(env_base)])[0] == 0
    # 离线替身的基线没有 health，照常接受
    offline = tmp_path / "offline.json"
    assert main([*SAMPLE, "--json", str(offline)], out=io.StringIO(), err=io.StringIO()) == 0
    assert _load(offline)["health"] is None
    code, out, err, _ = _run([*ARGV, "--baseline", str(offline), "--tolerance", "0.99"])
    assert code == 0, err
    assert "== §13.5 回归比较（基线 offline.json" in out


# ================================================================ 和基线没有共同用例：开跑前拦下


def test_a_baseline_with_no_common_case_is_refused_before_any_call(tmp_path):
    base = tmp_path / "greeting.json"
    out = io.StringIO()
    assert main(["--intent", "greeting", "--json", str(base)], out=out, err=io.StringIO()) == 0
    code, text, err, client = _run(
        ["--llm", "env", "--intent", "size_advice", "--baseline", str(base)]
    )
    assert code == 2 and text == "" and client.calls == []
    size = sum(1 for c in CASES if c.expected_intent == Intent.SIZE_ADVICE)
    assert f"评测没有运行：基线 {base} 和这次选出的 {size} 条用例没有共同的用例，无法比较" in err


def test_the_check_uses_the_same_selection_as_the_run(tmp_path):
    """开跑前的检查和 evaluate 选的是同一批用例（--limit / --seed 抽样）：基线里只有抽中的一条时
    照常跑、比这一条；基线里是其余所有用例时不跑。"""
    picked = [c.id for c in select_cases(cases("all"), limit=3, seed=5)]
    full = tmp_path / "full.json"
    argv = ["--limit", "3", "--seed", "5"]
    assert main(["--json", str(full)], out=io.StringIO(), err=io.StringIO()) == 0
    data = _load(full)

    def baseline_of(keep: Callable[[str], bool], name: str) -> Path:
        path = tmp_path / name
        rows = [row for row in data["cases"] if keep(row["id"])]
        path.write_text(json.dumps({**data, "cases": rows}, ensure_ascii=False), encoding="utf-8")
        return path

    one = baseline_of(lambda i: i == picked[1], "one.json")
    out, err = io.StringIO(), io.StringIO()
    assert main([*argv, "--baseline", str(one), "--json"], out=out, err=err) == 0, err.getvalue()
    report = json.loads(out.getvalue())
    assert [c["id"] for c in report["cases"]] == picked
    assert report["baseline"]["common_cases"] == 1 and report["baseline"]["only_current"] == 2

    rest = baseline_of(lambda i: i not in picked, "rest.json")
    out, err = io.StringIO(), io.StringIO()
    assert main([*argv, "--baseline", str(rest)], out=out, err=err) == 2
    assert out.getvalue() == "" and "和这次选出的 3 条用例没有共同的用例" in err.getvalue()


def test_a_comparison_that_fails_after_the_run_keeps_the_report(tmp_path, monkeypatch):
    """跑完之后比较才出岔子（正常到不了：开跑前查过）：报告和 JSON 照常给出，退出码 2。"""
    base = tmp_path / "base.json"
    _valid_report(base)

    def broken(*args: Any, **kw: Any):
        raise ValueError("合成的故障：比较出错")

    monkeypatch.setattr(cli, "compare_to_baseline", broken)
    target = tmp_path / "out.json"
    code, out, err, _ = _run([*ARGV, "--baseline", str(base), "--json", str(target)])
    assert code == 2
    data = _load(target)
    assert data["n"] == 12 and data["baseline"] is None and data["health"]["valid"] is True
    assert "== 总体" in out and "§13.5 回归比较" not in out
    last = err.strip().splitlines()[-1]
    assert last.startswith("§13.5 回归比较没有做成：合成的故障：比较出错。评测已经跑完")
    assert last.endswith("（退出码 2）") and "评测没有运行" not in err
    # 这一轮本来就不算数：退出码照旧是 3
    code, _, err, _ = _run(
        [*ARGV, "--baseline", str(base), "--json"], _routes_rejected(400, "bad_request")
    )
    assert code == 3 and err.strip().splitlines()[-1].endswith("（退出码 3）")


# ================================================================ 中止、退出时的说明


def test_label_kinds_group_the_health_labels():
    assert _label_kinds(["http_429", "http_429:rate_limit_exceeded"]) == {"rate_limit"}
    assert _label_kinds({"http_402:insufficient_balance": 30}) == {"billing"}
    assert _label_kinds(["http_401", "http_403:forbidden"]) == {"auth"}
    assert _label_kinds(["http_400:invalid_request", "http_404"]) == {"bad_request"}
    assert _label_kinds(["http_503", "timeout", "malformed:oops", "LLMError"]) == {
        "server",
        "timeout",
        "malformed",
        "LLMError",
    }
    assert _label_kinds(["rate_limit", "http_x", "http_302"]) == {
        "rate_limit",
        "http_x",
        "http_302",
    }
    for nothing in (None, {}, [], 7):  # 没有标签、读不懂的：当作没有
        assert _label_kinds(nothing) == set()


def test_the_abort_hint_follows_the_failure_labels():
    slow = _abort_hint({"http_429": 24})
    assert "--min-interval" in slow and "充值" not in slow and "--skip-preflight" not in slow
    paid = _abort_hint({"http_402:insufficient_balance": 30})
    assert "充值" in paid and "--min-interval" not in paid
    for labels in (
        {"http_400:invalid_request": 9},
        {"malformed": 9},
        {"http_401": 9},
        {"读不出": 9},
    ):
        hint = _abort_hint(labels)
        assert "不带 --skip-preflight 重跑一次" in hint and "连通性检查" in hint
        assert "--min-interval" not in hint
    for labels in ({"truncated": 8}, {"empty": 8}, {"截断": 8}, {"空回答": 8}):
        hint = _abort_hint(labels)
        assert "思考模式" in hint and "llm.extra_body" in hint and "--min-interval" not in hint
    both = _abort_hint({"http_429": 3, "http_402": 21})
    assert "--min-interval" in both and "充值" in both
    for labels in ({"timeout": 24}, {"http_503": 24}, {}, None):  # 没有对得上的：不乱给建议
        assert _abort_hint(labels) == "先解决失败的原因再重跑"


@pytest.mark.parametrize(
    ("status", "kind", "expected", "absent"),
    [
        (429, "rate_limit", "--min-interval", "充值"),
        (402, "billing", "充值", "--min-interval"),
        (400, "bad_request", "不带 --skip-preflight 重跑一次", "--min-interval"),
    ],
)
def test_the_abort_message_names_the_remedy_for_what_failed(status, kind, expected, absent):
    def outage(request: LLMRequest, nth: int):
        return _rejected(status, kind)

    code, out, err, _ = _run(ARGV, outage)
    assert code == 2 and out == ""
    last = err.strip().splitlines()[-1]
    head = f"评测中止：连续 {STOP_AFTER_FAILED_CASES} 条用例的大模型调用没有一次用得上（"
    assert last.startswith(head)
    assert f"http_{status} ×" in last and expected in last and absent not in last
    assert last.endswith("确实要跑完、看兜底逻辑的数字时加 --allow-degraded")


def test_an_abort_without_labels_still_gets_a_message(monkeypatch):
    """中止的异常上没有 labels（另一条改动会换停下的规则）：照样给出说明，不抛异常。"""

    class Bare(cli.EvalAborted):
        def __init__(self) -> None:
            Exception.__init__(self, "合成的中止")

    def aborted(*args: Any, **kw: Any):
        raise Bare()

    monkeypatch.setattr(cli, "evaluate", aborted)
    code, out, err, _ = _run(ARGV)
    assert code == 2 and out == ""
    assert err.strip().splitlines()[-1] == (
        "评测中止：合成的中止。先解决失败的原因再重跑；"
        "确实要跑完、看兜底逻辑的数字时加 --allow-degraded"
    )


def test_the_exit_3_line_mentions_min_interval_only_for_rate_limits():
    limited = _routes_rejected(429, "rate_limit", first=9)  # 3 条用例被限流，不到提前停下的条数
    code, _, err, _ = _run(ARGV, limited)
    last = err.strip().splitlines()[-1]
    assert code == 3 and last.startswith("这一轮没有有效评到模型（退出码 3）：路由：12 条里有 3 条")
    assert "被限流（HTTP 429）的调用，重跑时加 --min-interval 放慢" in last
    assert last.endswith("确认就是要这份结果时加 --allow-degraded")
    code, _, err, _ = _run(ARGV, _routes_rejected(400, "bad_request", first=9))
    assert code == 3 and "--min-interval" not in err


def test_the_regression_line_does_not_claim_an_exit_code_the_process_does_not_return(tmp_path):
    base = tmp_path / "base.json"
    _valid_report(base)
    argv = [*ARGV, "--baseline", str(base)]
    # 不算数又有回归：进程返回 3，哪里都不写"退出码 1"
    code, out, err, _ = _run(argv, _routes_rejected(400, "bad_request"))
    assert code == 3 and "↓回归" in out
    assert "→ 有指标下降超过容差：不能上线" in out and "退出码" not in out
    assert "退出码 1" not in err and "退出码 3" in err
    assert "回归比较也有指标下降，但兜底逻辑的数字比不出回归，退出码仍是 3" in err
    # 真的返回 1 时才说 1
    code, out, err, _ = _run([*argv, "--allow-degraded"], _routes_rejected(400, "bad_request"))
    assert code == 1 and "退出码" not in out
    assert err.strip().splitlines()[-1] == "§13.5 回归比较有指标下降超过容差（退出码 1）"


# ================================================================ 帮助、配置的提醒


def test_help_states_the_fixed_lost_vote_ceiling_and_what_happens_to_the_files():
    text = "".join(build_parser().format_help().split())  # argparse 会折行：去掉空白再比
    assert "路由丢票的上限固定是10%，不随它变" in text
    assert "不覆盖已有的文件（包括--baseline那个）：改写到旁边的<文件名>.invalid.json" in text
    assert "--llmenv跑出来的基线要是有效评到模型的一轮（按默认上限）" in text
    assert "基线读不了或不能当基线" in text and "跑完了但回归比较没有做成" in text
    doc = "".join((cli.__doc__ or "").split())
    assert "路由丢票的上限固定是10%" in doc and "<文件名>.invalid.json" in doc
    assert "这一轮不算数时是3" in doc and "报告和JSON照常写出" not in doc


def test_the_config_note_is_repeated_after_the_progress_lines(tmp_path, monkeypatch):
    """用的是 RODIT_CONFIG 指的配置：和"按内置默认值跑"的提醒一样，开跑前一次、结尾再一次。"""
    deployed = tmp_path / "deployed.yaml"
    deployed.write_text("decision:\n  vote_samples: 3\n", encoding="utf-8")
    monkeypatch.setenv("RODIT_CONFIG", str(deployed))
    code, _, err, _ = _run(ARGV)
    note = f"配置：没有给 --config，用环境变量 RODIT_CONFIG 指的 {deployed}"
    assert code == 0 and err.count(note) == 2 and DEFAULTS_WARNING not in err
    assert err.index(note) < err.index("[1/12]") and err.rindex(note) > err.index("[12/12]")
    code, _, err, _ = _run([*ARGV, "--config", str(deployed)])  # 给了 --config：没有这句
    assert code == 0 and "RODIT_CONFIG" not in err
