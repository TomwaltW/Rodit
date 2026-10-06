"""python -m rodit.showcase --selfcheck：每个预设场景跑一遍、逐个打印、汇总；必查全过退出码 0，
有没过的 1，配置错误 2。离线替身、本机的模型替身（llm_stub.py，走真实 HTTP）各跑一遍；
再用一个"把什么都认成问候"的假模型确认必查真的会失败。"""

from __future__ import annotations

import io
import json
import re
from functools import partial
from pathlib import Path

import pytest
from llm_stub import StubLLM, qa_answer, route_answer
from showcase_support import NOW

from rodit.showcase.__main__ import EXIT_SELFCHECK_FAILED, EXIT_USAGE, main
from rodit.showcase.llm import ShowcaseLLM
from rodit.showcase.realchain import RELAXED_NOTE
from rodit.showcase.state import ShowcaseHub
from rodit.testing.fakes import FakeClock, FakeLLMClient

FAST_HUB = partial(ShowcaseHub, clock=FakeClock(NOW), history_seed=None)
"""不回放历史的演示站（快）；回放历史的另有一个测试。"""


def _no_server(*args, **kwargs):
    raise AssertionError("--selfcheck 不起服务")


def _selfcheck(argv: list[str], **kwargs) -> tuple[int, str]:
    out = io.StringIO()
    kwargs.setdefault("hub_factory", FAST_HUB)
    code = main(["--selfcheck", *argv], out, run=_no_server, **kwargs)
    return code, out.getvalue()


def test_offline_selfcheck_passes_and_prints_every_scenario(tmp_path: Path):
    target = tmp_path / "report" / "selfcheck.json"
    code, text = _selfcheck(["--platform", "pdd", "--json", str(target)])
    assert code == 0, text
    assert "【仅演示】Rodit 客户演示站自检" in text and "离线替身" in text
    assert "能力检查" not in text  # 离线不调大模型，没有能力检查
    assert "[拼多多] off_platform · 站外交易（§11.4）" in text
    assert "✓ 必查 站外交易命中 §6.6 规则 1" in text
    assert "耗时（P50 / P95，对照 §12.1）" in text and "自检通过" in text
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["summary"]["ok"] is True and report["summary"]["must"]["failed"] == 0
    assert report["chain"]["real"] is False and report["capabilities"] is None
    assert report["chain"]["request_options"] is None
    assert {s["platform"] for s in report["scenarios"]} == {"pdd"}
    assert len(report["scenarios"]) == 10
    off = next(s for s in report["scenarios"] if s["scenario"] == "off_platform")
    assert off["route"]["rule"] == 1 and off["suggestions"][0]["kind"] == "template"
    assert set(off["latency_ms"]) >= {"route", "generate", "qa", "total"}
    assert off["llm"] is None  # 离线：没有真实调用


def test_json_to_stdout_is_only_json():
    code, text = _selfcheck(["--platform", "taobao", "--json", "-"])
    assert code == 0
    report = json.loads(text)
    assert len(report["scenarios"]) == 15 and report["summary"]["ok"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--llm", "env", "--platform", "taobao"],  # 启动检查没过（密钥被拒）
        ["--config", "/nonexistent/rodit.yaml"],  # 配置不能用
    ],
)
def test_json_to_stdout_stays_json_when_startup_fails(stub_env: StubLLM, argv):
    """--json -：标准输出只放 JSON。启动就失败时标准输出是空的，诊断和"拒绝启动"都写到标准错误。"""
    stub_env.fail("preflight", "401")
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["--selfcheck", *argv, "--json", "-"], out, err=err, run=_no_server, hub_factory=FAST_HUB
    )
    assert code == EXIT_USAGE
    assert out.getvalue() == ""
    assert "拒绝启动" in err.getvalue()


def test_real_chain_selfcheck_against_the_stub(stub_env: StubLLM):
    code, text = _selfcheck(["--llm", "env", "--platform", "douyin", "--json", "-"])
    assert code == 0, text
    report = json.loads(text)
    chain = report["chain"]
    assert chain["real"] and chain["judge"] == "llm" and chain["latency"] == "strict"
    assert chain["model"] == "stub-chat" and chain["host"] == stub_env.base_url.removesuffix("/v1")
    for s in report["scenarios"]:
        assert s["route"]["provider"] == "incloud_llm" and not s["route"]["degraded"]
        assert s["route"]["votes"]["answered"] == 3
        calls = {row["purpose"]: row["calls"] for row in s["llm"]["by_purpose"]}
        assert calls["route"] == 3
    summary = report["summary"]
    assert summary["llm"]["calls"]["route"] == 3 * len(report["scenarios"])
    assert summary["llm"]["input_tokens"] > 0 and summary["degradations"]["route_fallback"] == 0
    assert [c["id"] for c in report["platform_checks"]] == ["replay_offline"]
    # 启动时的能力检查（和评测同一个）原样进报告
    caps = report["capabilities"]
    assert [(c["name"], c["status"]) for c in caps["checks"]] == [
        ("plain", "ok"),
        ("json_mode", "ok"),
        ("logprobs", "ok"),
    ]
    assert caps["endpoint"] == chain["host"] and caps["model"] == "stub-chat"
    assert caps["lines"][0].startswith("大模型能力检查（stub-chat @ ")
    assert chain["request_options"] == {"json_mode": True, "logprobs": True, "extra_body_keys": []}
    assert stub_env.api_key not in text


def test_strict_selfcheck_reports_fallbacks_and_suggests_relaxed(stub_env: StubLLM, tmp_path):
    """路由每一票都超时：退回关键词规则，降级记清楚；正确的降级不算必查失败。"""
    config = tmp_path / "rodit.yaml"
    config.write_text("decision:\n  timeout_s: 0.2\n", encoding="utf-8")
    stub_env.delay("route", 0.4)
    code, text = _selfcheck(["--llm", "env", "--platform", "pdd", "--config", str(config)])
    assert code == 0, text
    assert "大模型 3 次调用全部超时（每次限时 0.2 秒） → 这条消息改用关键词规则路由" in text
    assert "--latency relaxed" in text
    assert "降级：路由退回关键词规则 10 次" in text
    assert "不适用" in text  # 降级时高风险类型不推荐：价格类的必查记为不适用


def test_relaxed_banner(stub_env: StubLLM):
    code, text = _selfcheck(["--llm", "env", "--platform", "pdd", "--latency", "relaxed"])
    assert code == 0 and RELAXED_NOTE in text
    # 横幅里是能力检查的每一步，汇总里一行列出
    assert "启动检查通过——大模型能力检查（stub-chat @ http://127.0.0.1:" in text
    assert "      JSON 模式：通过" in text
    assert "启动时的大模型能力检查：普通调用 通过（" in text and "logprobs 通过" in text
    assert "提示（logprobs：可以省掉这个参数：设 RODIT_LLM_LOGPROBS=off" in text
    assert "判断层每次调用 8 秒、生成 15 秒、整条消息 30 秒" in text
    assert "大多数托管大模型 1 秒内答不完" not in text


def test_history_replay_is_checked(stub_env: StubLLM):
    code, text = _selfcheck(
        ["--llm", "env", "--platform", "taobao"],
        hub_factory=partial(ShowcaseHub, clock=FakeClock(NOW)),
    )
    assert code == 0, text
    assert "✓ 必查 回放历史 226 条消息，没有调用真实大模型" in text


def _confused(request):
    """把什么都认成问候的假模型：路由永远答 greeting、不要人工、情绪 0。"""
    text = "\n".join(m.content for m in request.messages)
    if request.purpose == "route":
        answer = {
            "intent": {"choice": "greeting"},
            "need_human": {"answer": "否", "p": 0.0},
            "emotion": {"score": 0},
            "multi_intent": {"answer": "否", "p": 0.0},
            "second_intent": {"choice": "无"},
        }
        return answer
    if request.purpose == "qa":
        return {
            "grounded": {"answer": "是", "p": 0.95},
            "answers_question": {"answer": "是", "p": 0.9},
            "promise_risk": {"answer": "否", "p": 0.05},
            "platform_violation": {"answer": "否", "p": 0.02},
            "tone": {"score": 3},
        }
    if request.purpose == "generate":
        return {"answer": "亲，在的", "evidence_ids": [], "need_clarify": False}
    return '{"ok": true}' if "JSON" in text else "OK"


def test_must_failures_exit_1():
    client = FakeLLMClient(handler=_confused, model="confused-model")
    code, text = _selfcheck(["--llm", "env", "--platform", "taobao"], llm_factory=lambda: client)
    assert code == EXIT_SELFCHECK_FAILED
    assert "✗ 必查 站外交易命中 §6.6 规则 1" in text
    assert "✗ 必查 投诉：不推荐、提醒客服优先处理（§5.7）" in text
    assert "自检没通过：有必查项失败" in text


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--judge", "llm"], "--judge llm 需要 --llm env"),
        (["--config", "/nonexistent/rodit.yaml"], "配置不能用"),
    ],
)
def test_config_errors_exit_2(argv, message):
    code, text = _selfcheck(argv)
    assert code == EXIT_USAGE and message in text


def test_unsupported_judge_provider_exits_2(stub_env: StubLLM, tmp_path):
    config = tmp_path / "rodit.yaml"
    config.write_text("decision:\n  provider: classifier\n", encoding="utf-8")
    code, text = _selfcheck(["--llm", "env", "--config", str(config)])
    assert code == EXIT_USAGE and "incloud_llm 或 rules" in text
    assert stub_env.count() == 0  # 配置错误在调用模型之前就拦下


def test_bad_yaml_exits_2(tmp_path):
    config = tmp_path / "rodit.yaml"
    config.write_text("decision: [unclosed\n", encoding="utf-8")
    code, text = _selfcheck(["--config", str(config)])
    assert code == EXIT_USAGE and "配置不能用" in text


def test_options_that_only_make_sense_with_selfcheck():
    out = io.StringIO()
    assert main(["--platform", "all"], out, run=_no_server) == EXIT_USAGE
    assert main(["--json", "x.json"], out, run=_no_server) == EXIT_USAGE
    assert "只用于 --selfcheck" in out.getvalue()


def test_unwritable_json_target_exits_2(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    code, text = _selfcheck(["--platform", "pdd", "--json", str(blocker / "out.json")])
    assert code == EXIT_USAGE and "写不进" in text


def test_selfcheck_does_not_need_a_loopback_host(stub_env: StubLLM):
    """--selfcheck 不起服务：--host 不影响（只有起服务时才要求 --llm env 只监听本机）。"""
    code, _ = _selfcheck(["--llm", "env", "--platform", "pdd", "--host", "0.0.0.0"])
    assert code == 0


def test_phone_check_ignores_hex_epochs():
    """页面数据里的手机号检查：中文里夹的号码要认出来，epoch 这类十六进制串里的数字不算。"""
    from rodit.showcase.selfcheck import PHONE

    assert PHONE.search("联系电话13812345678，谢谢")
    assert PHONE.search('"text": "13812345678"')
    assert not PHONE.search('"epoch": "9a13812345678abc"')
    assert not PHONE.search("订单 o_1002、运单 ZT7300000001")


# ================================================================ §12.2 降级不算必查失败


def _report(argv: list[str], **kwargs) -> tuple[int, dict]:
    code, text = _selfcheck([*argv, "--json", "-"], **kwargs)
    return code, json.loads(text)


def _check(report: dict, scenario: str, cid: str) -> dict:
    s = next(s for s in report["scenarios"] if s["scenario"] == scenario)
    return next(c for c in s["checks"] if c["id"] == cid)


def _must_failures(report: dict) -> list[tuple[str, str, str]]:
    return [
        (s["scenario"], c["id"], c["detail"])
        for s in report["scenarios"]
        for c in s["checks"]
        if c["must"] and c["status"] == "fail"
    ]


def test_qa_judge_timeouts_under_strict_are_not_must_failures(stub_env: StubLLM, tmp_path):
    """路由投票、质检判断层每次都超时（同一个 decision.timeout_s）：路由退回关键词规则，大模型草稿
    质检不可用、不推荐（§12.2）。分条那一项前提不成立（不适用），不是必查失败；汇总的提醒里写明
    质检也超时了。"""
    config = tmp_path / "rodit.yaml"
    config.write_text("decision:\n  timeout_s: 0.2\n", encoding="utf-8")
    stub_env.delay("route", 0.4)
    stub_env.delay("qa", 0.4)
    code, text = _selfcheck(["--llm", "env", "--platform", "taobao", "--config", str(config)])
    assert code == 0, text
    assert "大模型草稿的质检不可用" in text and "--latency relaxed" in text
    code, report = _report(["--llm", "env", "--platform", "taobao", "--config", str(config)])
    assert code == 0, _must_failures(report)
    two = _check(report, "multi", "two_parts")
    assert two["status"] == "skip" and "§12.2" in two["detail"]
    deg = report["summary"]["degradations"]
    assert deg["qa_unavailable"] > 0 and deg["route_fallback"] == 15


@pytest.mark.parametrize("platform", ["taobao", "pdd"])
def test_generation_timeouts_are_not_must_failures(stub_env: StubLLM, tmp_path, platform):
    """生成每次都超时：重写稿、分条都没有大模型草稿，退回 FAQ / 模板（§12.2）。重写、分条两项记为
    不适用（原因写生成超时），汇总里数出生成超时。"""
    config = tmp_path / "rodit.yaml"
    config.write_text("generation:\n  timeout_s: 0.2\n", encoding="utf-8")
    stub_env.delay("generate", 0.4)
    code, report = _report(["--llm", "env", "--platform", platform, "--config", str(config)])
    assert code == 0, _must_failures(report)
    rewrite = _check(report, "qa_block", "rewrite_passed")
    assert rewrite["status"] == "skip" and "没有给出" in rewrite["detail"]
    assert _check(report, "qa_block", "first_blocked")["status"] == "pass"
    two = _check(report, "multi", "two_parts")
    assert two["status"] == "skip" and "生成" in two["detail"]
    deg = report["summary"]["degradations"]
    assert deg["generate_failed"] > 0 and deg["generate_timeouts"] == deg["generate_failed"]
    qa_block = next(s for s in report["scenarios"] if s["scenario"] == "qa_block")
    assert any("生成" in label for label in qa_block["degraded"])


def test_generation_timeouts_in_text_summary(stub_env: StubLLM, tmp_path):
    """文字汇总里有生成超时的次数，并提示 --latency relaxed。"""
    config = tmp_path / "rodit.yaml"
    config.write_text("generation:\n  timeout_s: 0.2\n", encoding="utf-8")
    stub_env.delay("generate", 0.4)
    code, text = _selfcheck(["--llm", "env", "--platform", "taobao", "--config", str(config)])
    assert code == 0, text
    assert re.search(r"大模型生成失败 \d+ 次（其中超时 \d+ 次）", text)
    assert "--latency relaxed" in text
    assert "自检通过" in text


def _emotion_below_threshold() -> FakeLLMClient:
    """路由照关键词规则答，只是情绪激动的消息三票打 3、3、2（平均 2.67，没到阈值 3）。"""
    scores = iter([3, 3, 2] * 1000)

    def handler(request):
        user = request.messages[-1].content
        if request.purpose == "route":
            answer = route_answer(user)
            if answer.get("emotion", {}).get("score", 0) >= 3:
                answer["emotion"] = {"score": next(scores)}
            return answer
        if request.purpose == "qa":
            return qa_answer(user)
        if request.purpose == "generate":
            return ShowcaseLLM().complete(request).text
        text = "\n".join(m.content for m in request.messages)
        return '{"ok": true}' if "JSON" in text else "OK"

    return FakeLLMClient(handler=handler, model="calm-model")


def test_soothe_is_not_applicable_when_the_model_scores_emotion_below_threshold():
    """情绪分是模型投出来的：没到阈值时规则 6 是对的，安抚那一项记为不适用（不是必查失败）。"""
    client = _emotion_below_threshold()
    code, report = _report(["--llm", "env", "--platform", "taobao"], llm_factory=lambda: client)
    soothe = _check(report, "emotion", "soothe")
    assert soothe["status"] == "skip" and "没到阈值" in soothe["detail"], soothe
    assert code == 0, _must_failures(report)
