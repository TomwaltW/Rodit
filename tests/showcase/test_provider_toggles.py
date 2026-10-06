"""模型服务不支持某些请求参数时，用开关让演示站照样能用（端到端，不联网）。

真实的 OpenAICompatibleClient（rodit.eval.llm_env.client_from_env，演示站 --llm env 用的就是它）
走真实的 HTTP 调本机的模型替身（llm_stub.py）；替身按"服务的能力"拒绝参数，不分用途：

- 不支持 JSON 模式（response_format）：启动检查拒绝，提示 RODIT_LLM_JSON_MODE=off；关掉后整条链路
  （路由投票、质检、生成）只靠提示词出 JSON，一个请求都不再带 response_format；
- 不支持 logprobs：启动检查拒绝，提示 RODIT_LLM_LOGPROBS=off（或 --judge offline）；关掉后路由照样
  用大模型投票，置信度按票数算；
- 默认开思考模式：开着时 JSON 模式被拒绝，提示 RODIT_LLM_EXTRA_BODY；用它关掉思考后通过，每个
  请求都带这个参数。只关 JSON 模式不关思考时，回答前面有 <think>…</think>（里面还有一个 JSON
  草稿），客户端去掉后照样能用。
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from llm_stub import StubLLM
from showcase_support import NOW, build_real_showcase, json_text

from rodit.showcase.__main__ import EXIT_USAGE, main
from rodit.showcase.state import ShowcaseHub
from rodit.testing.fakes import FakeClock

FAST_HUB = partial(ShowcaseHub, clock=FakeClock(NOW), history_seed=None)
"""不回放历史的演示站（快）。"""

THINKING_OFF = '{"thinking": {"type": "disabled"}}'

Start = Callable[..., StubLLM]


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[Start]:
    """start(**能力) 起一个有这些能力的模型替身，RODIT_LLM_* 指过去（三个开关先清空）。"""
    with ExitStack() as stack:

        def start(**capabilities: Any) -> StubLLM:
            stub = stack.enter_context(StubLLM(**capabilities))
            monkeypatch.setenv("RODIT_LLM_BASE_URL", stub.base_url)
            monkeypatch.setenv("RODIT_LLM_MODEL", stub.model)
            monkeypatch.setenv("RODIT_LLM_API_KEY", stub.api_key)
            for name in ("TIMEOUT_S", "EXTRA_BODY", "JSON_MODE", "LOGPROBS"):
                monkeypatch.delenv(f"RODIT_LLM_{name}", raising=False)
            return stub

        yield start


def _no_server(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("--selfcheck 不起服务")


def selfcheck(*argv: str, json_out: Path | None = None) -> tuple[int, str, dict | None]:
    """python -m rodit.showcase --selfcheck --llm env …；给了 json_out 时一并读回完整报告。"""
    out = io.StringIO()
    extra = ["--json", str(json_out)] if json_out is not None else []
    code = main(
        ["--selfcheck", "--llm", "env", *argv, *extra], out, run=_no_server, hub_factory=FAST_HUB
    )
    report = None
    if json_out is not None and json_out.exists():
        report = json.loads(json_out.read_text(encoding="utf-8"))
    return code, out.getvalue(), report


def assert_full_llm_chain(report: dict, stub: StubLLM) -> None:
    """每个场景的路由都是大模型投票（3 票都有回答、没有降级），质检、生成也真的调了模型。"""
    assert report["summary"]["ok"] is True
    assert report["summary"]["errors"] == 0
    degradations = report["summary"]["degradations"]
    assert degradations["route_fallback"] == 0 and degradations["qa_unavailable"] == 0
    for s in report["scenarios"]:
        route = s["route"]
        assert route["provider"] == "incloud_llm" and not route["degraded"], s["scenario"]
        assert route["votes"]["answered"] == 3 and not route["votes"]["errors"], s["scenario"]
    calls = report["summary"]["llm"]["calls"]
    assert calls["route"] == 3 * len(report["scenarios"])
    assert calls.get("qa", 0) > 0 and calls.get("generate", 0) > 0
    assert not any(report["summary"]["llm"]["failed"].values())
    assert stub.rejected == 0
    assert stub.api_key not in json.dumps(report, ensure_ascii=False)


def capability(report: dict, name: str) -> dict:
    return next(c for c in report["capabilities"]["checks"] if c["name"] == name)


# ================================================================ 不支持 JSON 模式


def test_provider_without_json_mode_needs_the_json_toggle(
    provider: Start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stub = provider(accept_response_format=False)
    code, text, _ = selfcheck("--platform", "pdd")
    assert code == EXIT_USAGE
    assert "JSON 模式：被拒绝" in text and "response_format_unsupported" in text
    assert "拒绝启动：大模型 JSON 模式检查没通过" in text and "RODIT_LLM_JSON_MODE=off" in text
    assert stub.count("route") == 0 and stub.api_key not in text  # 没过就不跑场景

    monkeypatch.setenv("RODIT_LLM_JSON_MODE", "off")
    stub.reset()
    code, text, report = selfcheck("--platform", "pdd", json_out=tmp_path / "r.json")
    assert code == 0, text
    assert report is not None
    assert_full_llm_chain(report, stub)
    # 一个请求都不再带 response_format：JSON 全靠提示词，解析照样能用
    assert stub.count() > 0 and all("response_format" not in b for b in stub.bodies())
    json_check = capability(report, "json_mode")
    assert json_check["status"] == "ok" and json_check["output_ok"] is True
    assert "RODIT_LLM_JSON_MODE=off" in json_check["note"]
    assert report["chain"]["request_options"]["json_mode"] is False
    assert '"json_mode":false' in text and "启动时的大模型能力检查：普通调用 通过" in text


# ================================================================ 不支持 logprobs


def test_provider_without_logprobs_needs_the_logprobs_toggle(
    provider: Start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stub = provider(accept_logprobs=False)
    code, text, _ = selfcheck("--platform", "douyin")
    assert code == EXIT_USAGE
    assert "拒绝启动：大模型 logprobs 检查没通过" in text
    assert "RODIT_LLM_LOGPROBS=off" in text and "--judge offline" in text
    assert stub.count("route") == 0

    # 路由不用大模型投票时根本不请求 logprobs：不用开关也能跑
    stub.reset()
    code, text, _ = selfcheck("--platform", "douyin", "--judge", "offline")
    assert code == 0, text
    assert stub.count("probe_logprobs") == 0 and stub.count("route") == 0 and stub.rejected == 0

    monkeypatch.setenv("RODIT_LLM_LOGPROBS", "off")
    stub.reset()
    code, text, report = selfcheck("--platform", "douyin", json_out=tmp_path / "r.json")
    assert code == 0, text
    assert report is not None
    assert_full_llm_chain(report, stub)
    assert all("logprobs" not in b and "top_logprobs" not in b for b in stub.bodies())
    assert all(s["route"]["votes"]["probability_source"] == "votes" for s in report["scenarios"])
    assert capability(report, "logprobs")["status"] == "off"
    assert report["chain"]["request_options"]["logprobs"] is False


# ================================================================ 默认开思考模式


def test_thinking_provider_needs_extra_body(
    provider: Start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    stub = provider(thinking=True)
    code, text, _ = selfcheck("--platform", "taobao")
    assert code == EXIT_USAGE
    assert "response_format_requires_thinking_disabled" in text
    assert "拒绝启动：大模型 JSON 模式检查没通过" in text and "RODIT_LLM_EXTRA_BODY" in text

    monkeypatch.setenv("RODIT_LLM_EXTRA_BODY", THINKING_OFF)
    stub.reset()
    code, text, report = selfcheck("--platform", "taobao", json_out=tmp_path / "r.json")
    assert code == 0, text
    assert report is not None
    assert_full_llm_chain(report, stub)
    # 每个请求（启动检查、路由、质检、生成）都带着关思考的参数，标准字段照旧
    bodies = stub.bodies()
    assert bodies and all(b["thinking"] == {"type": "disabled"} for b in bodies)
    assert any(b.get("response_format") == {"type": "json_object"} for b in bodies)
    assert report["chain"]["request_options"]["extra_body_keys"] == ["thinking"]
    # 报告里只有 extra_body 的键名（值可能是网关凭据，见 core.config.LLMRequestOptions.summary）
    assert '"extra_body_keys":["thinking"]' in report["capabilities"]["options"]
    assert "disabled" not in report["capabilities"]["options"]


def test_thinking_output_is_stripped_when_only_json_mode_is_off(
    provider: Start, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """只关 JSON 模式、不关思考：每个回答前面都有 <think>…</think>（里面还有一个 JSON 草稿），
    客户端去掉它，判断层的票、生成的回答都取后面的正文。"""
    stub = provider(thinking=True)
    monkeypatch.setenv("RODIT_LLM_JSON_MODE", "off")
    code, text, report = selfcheck("--platform", "taobao", json_out=tmp_path / "r.json")
    assert code == 0, text
    assert report is not None
    assert_full_llm_chain(report, stub)
    # 思考模式真的一直开着（每个回答都带了 <think>…</think>），页面数据和报告里看不到它
    assert stub.count() > 0 and all(stub.thinking_on(b) for b in stub.bodies())
    shown = json.dumps(report, ensure_ascii=False)
    assert "思考里的草稿" not in shown and "<think>" not in shown


# ================================================================ 页面：三个开关一起


def test_page_works_against_a_provider_that_needs_all_three_toggles(
    provider: Start, monkeypatch: pytest.MonkeyPatch
):
    """不支持 JSON 模式、不支持 logprobs、默认开思考的服务：三个开关都设好，页面（HTTP 接口）
    整条链路照样是真实大模型，页面配置里写明生效的请求参数。"""
    stub = provider(accept_response_format=False, accept_logprobs=False, thinking=True)
    monkeypatch.setenv("RODIT_LLM_JSON_MODE", "off")
    monkeypatch.setenv("RODIT_LLM_LOGPROBS", "off")
    monkeypatch.setenv("RODIT_LLM_EXTRA_BODY", THINKING_OFF)
    sc = build_real_showcase()
    view = sc.chat("taobao", "size")
    u = view["trace"]["understand"]
    assert u["provider"] == "incloud_llm" and not u["degraded"]
    judge = u["judge"]
    assert judge["answered"] == 3 and not judge["errors"]
    assert judge["probability_source"] == "votes"
    [card] = view["result"]["suggestions"]
    assert card["kind"] == "llm" and not view["result"]["degraded"]
    assert stub.rejected == 0 and stub.count("generate") >= 1 and stub.count("qa") >= 1
    found = re.search(
        r'<script id="showcase-config" type="application/json">(.*?)</script>',
        sc.client.get("/").text,
        re.S,
    )
    assert found is not None
    chain = json.loads(found.group(1))["chain"]
    assert chain["request_options"] == {
        "json_mode": False,
        "logprobs": False,
        "extra_body_keys": ["thinking"],
    }
    assert stub.api_key not in json_text(view)


# ================================================================ 开关的值不对


@pytest.mark.parametrize(
    ("name", "value", "phrase"),
    [
        ("EXTRA_BODY", '{"api_key": "x"}', "RODIT_LLM_EXTRA_BODY"),
        ("EXTRA_BODY", '{"response_format": {"type": "text"}}', "response_format"),
        ("EXTRA_BODY", "not json", "RODIT_LLM_EXTRA_BODY"),
        ("JSON_MODE", "maybe", "RODIT_LLM_JSON_MODE"),
        ("LOGPROBS", "2", "RODIT_LLM_LOGPROBS"),
    ],
)
def test_bad_toggle_values_refuse_to_start_before_any_request(
    provider: Start, monkeypatch: pytest.MonkeyPatch, name: str, value: str, phrase: str
):
    stub = provider()
    monkeypatch.setenv(f"RODIT_LLM_{name}", value)
    out = io.StringIO()
    runs: list[Any] = []
    code = main(["--llm", "env"], out, run=lambda app, **kw: runs.append(app))
    text = out.getvalue()
    assert code == EXIT_USAGE and runs == [] and stub.count() == 0
    assert "拒绝启动" in text and phrase in text
    assert stub.api_key not in text
