"""大模型能力检查（eval.llm_env.probe_capabilities）和请求参数进评测指纹（方案 §13.5、D5）。

preflight 只发最简单的请求；路由、质检、生成还会带 response_format（JSON 模式）和
logprobs——各家模型服务支持得不一样。能力检查至多再发两个小请求，被拒绝时说清楚改哪个开关；
--llm env 跑之前先做，被拒绝就不白跑一整轮。请求参数（extra_body、两个开关）会改变回答，
记进评测指纹，线上按同一条规则算出来比较。
真实客户端 + 假传输层，合成数据和合成密钥，不联网。
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from rodit.core.config import EngineConfig, LLMRequestOptions
from rodit.core.errors import LLMError, LLMTimeout
from rodit.core.ports import LLMRequest
from rodit.eval.cli import _run_fingerprint, env_client, main
from rodit.eval.fingerprint import (
    CLASSIFIER_PATH_ENV,
    current_fingerprint,
    differences,
    running_llm_options,
    same_setup,
)
from rodit.eval.llm_env import (
    JSON_PROBE_PROMPT,
    LOGPROBS_PROBE_PROMPT,
    client_from_env,
    describe_options,
    preflight,
    probe_capabilities,
    render_capabilities,
)
from rodit.llm import OpenAICompatibleClient
from rodit.testing.fakes import FakeLLMClient

KEY = "sk-synthetic-capability-probe-0001"  # 合成密钥，只用于测试
BASE = "https://user:pw@llm.example.invalid:8443/v1"
ENV = {
    "RODIT_LLM_BASE_URL": BASE,
    "RODIT_LLM_MODEL": "synthetic-model",
    "RODIT_LLM_API_KEY": KEY,
}
THINKING_OFF = '{"thinking": {"type": "disabled"}}'


@pytest.fixture
def llm_env(monkeypatch):
    for name in ("EXTRA_BODY", "JSON_MODE", "LOGPROBS", "TIMEOUT_S"):
        monkeypatch.delenv(f"RODIT_LLM_{name}", raising=False)
    monkeypatch.delenv(CLASSIFIER_PATH_ENV, raising=False)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def ok(text: str, *, logprobs: Any = None) -> tuple[int, dict[str, Any]]:
    choice: dict[str, Any] = {"message": {"content": text}, "finish_reason": "stop"}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return 200, {"choices": [choice], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}


LETTER_LOGPROBS = {
    "content": [
        {
            "token": "A",
            "logprob": -0.01,
            "top_logprobs": [{"token": "A", "logprob": -0.01}, {"token": "B", "logprob": -5.0}],
        }
    ]
}


class Provider:
    """假的模型服务：按请求内容回答；json_status / logprobs_status 不是 200 时拒绝对应参数。"""

    def __init__(
        self,
        *,
        plain: tuple[int, dict[str, Any]] | Exception | None = None,
        json_status: int = 200,
        logprobs_status: int = 200,
        json_text: str = '{"ok": true}',
        return_logprobs: bool = True,
        json_error: Exception | None = None,
    ) -> None:
        self.plain = plain
        self.json_status = json_status
        self.logprobs_status = logprobs_status
        self.json_text = json_text
        self.return_logprobs = return_logprobs
        self.json_error = json_error
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, url, headers, body, timeout_s):
        self.bodies.append(body)
        prompt = body["messages"][-1]["content"]
        if prompt == JSON_PROBE_PROMPT:
            if self.json_error is not None:
                raise self.json_error
            if "response_format" in body and self.json_status != 200:
                return self.json_status, {"error": {"type": "invalid_request_error"}}
            return ok(self.json_text)
        if prompt == LOGPROBS_PROBE_PROMPT:
            if "logprobs" in body and self.logprobs_status != 200:
                return self.logprobs_status, {"error": {"code": "unsupported_parameter"}}
            return ok("A", logprobs=LETTER_LOGPROBS if self.return_logprobs else None)
        if isinstance(self.plain, Exception):
            raise self.plain
        return self.plain or ok("OK")


def client(provider: Provider, **kw: Any) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(base_url=BASE, model="synthetic-model", transport=provider, **kw)


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("RODIT_LLM_API_KEY", KEY)


def no_secrets(report) -> None:
    text = json.dumps(report.model_dump(mode="json"), ensure_ascii=False) + "\n".join(
        render_capabilities(report)
    )
    assert KEY not in text and "user:pw" not in text


# ================================================================ 能力检查


def test_everything_supported(api_key):
    provider = Provider()
    report = probe_capabilities(client(provider, extra_body={"enable_thinking": False}))
    assert [(c.name, c.status) for c in report.checks] == [
        ("plain", "ok"),
        ("json_mode", "ok"),
        ("logprobs", "ok"),
    ]
    assert report.reachable and report.blocking == [] and report.hints == []
    assert report.endpoint == "https://llm.example.invalid:8443"
    assert report.options == describe_options(
        client(provider, extra_body={"enable_thinking": False})
    )
    # 三个小请求：第一个就是 preflight 的请求；后两个各只带一个要检查的参数；extra_body 每个都带
    assert len(provider.bodies) == 3
    plain, as_json, with_lp = provider.bodies
    assert "response_format" not in plain and "logprobs" not in plain
    assert as_json["response_format"] == {"type": "json_object"} and "logprobs" not in as_json
    assert with_lp["logprobs"] is True and "response_format" not in with_lp
    assert all(b["enable_thinking"] is False for b in provider.bodies)
    assert all(b["max_tokens"] <= 32 for b in provider.bodies)
    no_secrets(report)


def test_preflight_request_is_unchanged(api_key):
    provider = Provider()
    preflight(client(provider))
    probe_capabilities(client(provider))
    assert provider.bodies[0] == provider.bodies[1]


def test_rejected_json_mode_blocks_and_names_the_switch(api_key):
    report = probe_capabilities(client(Provider(json_status=400)))
    check = report.check("json_mode")
    assert (check.status, check.http_status) == ("rejected", 400)
    assert "invalid_request_error" in (check.error or "")
    assert report.blocking == [check]
    assert any("RODIT_LLM_JSON_MODE=off" in h for h in report.hints)
    assert any("RODIT_LLM_EXTRA_BODY" in h for h in report.hints)  # 思考模式下不支持 JSON 的情况
    assert report.check("logprobs").status == "ok"  # 后面的照样检查
    no_secrets(report)


def test_rejected_logprobs_blocks_and_names_the_switch(api_key):
    report = probe_capabilities(client(Provider(logprobs_status=422)))
    check = report.check("logprobs")
    assert (check.status, check.http_status) == ("rejected", 422)
    assert report.blocking == [check]
    assert check.hint and "RODIT_LLM_LOGPROBS=off" in check.hint


def test_logprobs_accepted_but_not_returned_is_only_a_note(api_key):
    report = probe_capabilities(client(Provider(return_logprobs=False)))
    check = report.check("logprobs")
    assert check.status == "ok" and check.note and "没返回" in check.note
    assert check.hint and "RODIT_LLM_LOGPROBS=off" in check.hint
    assert report.blocking == []


def test_switched_off_capabilities(api_key):
    """开关关着：JSON 只靠提示词（请求里没有 response_format），logprobs 不检查（只发两个请求）。"""
    provider = Provider(json_status=400, logprobs_status=400)
    report = probe_capabilities(client(provider, json_mode=False, logprobs=False))
    assert [(c.name, c.status) for c in report.checks] == [
        ("plain", "ok"),
        ("json_mode", "ok"),
        ("logprobs", "off"),
    ]
    assert "RODIT_LLM_JSON_MODE=off" in (report.check("json_mode").note or "")
    assert len(provider.bodies) == 2 and "response_format" not in provider.bodies[1]
    assert report.blocking == []


def test_json_output_that_does_not_parse_gets_a_hint(api_key):
    report = probe_capabilities(client(Provider(json_text="好的，没问题")))
    check = report.check("json_mode")
    assert check.status == "ok" and "找不到 JSON" in (check.note or "")
    assert "RODIT_LLM_EXTRA_BODY" in (check.hint or "")
    assert report.blocking == []
    # 代码块、前后带话的 JSON 照样算能解析（判断层的规则）
    fenced = probe_capabilities(client(Provider(json_text='好的：\n```json\n{"ok": true}\n```')))
    assert fenced.check("json_mode").note is None


@pytest.mark.parametrize(
    ("status", "words"),
    [(401, "密钥"), (403, "密钥"), (404, "RODIT_LLM_BASE_URL"), (400, "RODIT_LLM_EXTRA_BODY")],
)
def test_unreachable_stops_after_one_request(api_key, status, words):
    provider = Provider(plain=(status, {"error": {"type": "x", "message": f"key {KEY}"}}))
    report = probe_capabilities(client(provider))
    plain = report.check("plain")
    assert (plain.status, plain.http_status) == ("rejected", status)
    assert not report.reachable and report.blocking == [plain]
    assert plain.hint and words in plain.hint
    assert [c.status for c in report.checks[1:]] == ["skipped", "skipped"]
    assert len(provider.bodies) == 1
    no_secrets(report)


def test_output_ok_records_whether_the_answer_is_usable(api_key):
    """output_ok：JSON 模式找得到 JSON 对象、logprobs 有返回（演示站据此拦下启动 / 提示）。"""
    good = probe_capabilities(client(Provider()))
    assert good.check("json_mode").output_ok is True and good.check("logprobs").output_ok is True
    assert good.check("plain").output_ok is None
    bad = probe_capabilities(client(Provider(json_text="好的，没问题", return_logprobs=False)))
    assert bad.check("json_mode").output_ok is False and bad.check("logprobs").output_ok is False
    rejected = probe_capabilities(client(Provider(json_status=400)))
    assert rejected.check("json_mode").output_ok is None  # 没有结果


def test_logprobs_check_can_be_skipped(api_key):
    """调用方这次不会请求 logprobs（演示站路由不用大模型投票）：不发第三个请求，记为没检查。"""
    provider = Provider(logprobs_status=400)
    report = probe_capabilities(client(provider), check_logprobs=False)
    check = report.check("logprobs")
    assert check.status == "skipped" and check.note and report.blocking == []
    assert len(provider.bodies) == 2
    # 开关关着时仍记为"已关闭"（说的是请求参数，不是这次检没检查）
    off = probe_capabilities(client(Provider(), logprobs=False), check_logprobs=False)
    assert off.check("logprobs").status == "off"


def test_timeouts_and_rate_limits_are_not_rejections(api_key):
    slow = probe_capabilities(client(Provider(json_error=TimeoutError())))
    check = slow.check("json_mode")
    assert check.status == "failed" and check.http_status is None and check.timed_out
    assert "RODIT_LLM_TIMEOUT_S" in (check.hint or "") and "思考" in (check.hint or "")
    assert slow.blocking == []
    limited = probe_capabilities(client(Provider(logprobs_status=429)))
    assert limited.check("logprobs").status == "failed" and limited.blocking == []
    assert not limited.check("logprobs").timed_out
    assert "限流" in (limited.check("logprobs").hint or "")


def test_other_clients_are_probed_with_default_options_and_scrubbed(api_key):
    """测试替身等不暴露 options 的客户端：按默认（两个参数都开）检查；错误信息再抹一次密钥。"""

    def handler(request: LLMRequest):
        if request.json_mode:
            return LLMError(f"大模型接口拒绝请求（HTTP 400）：{KEY}")
        if request.want_logprobs:
            return LLMTimeout("大模型调用超时（1 秒）")
        return "OK"

    fake = FakeLLMClient(handler=handler, model="fake-model")
    report = probe_capabilities(fake)
    assert report.options == LLMRequestOptions().describe() == describe_options(fake)
    assert report.check("json_mode").status == "rejected"
    assert report.check("logprobs").status == "failed"
    assert [r.purpose for r in fake.calls] == ["other"] * 3
    no_secrets(report)


# ================================================================ 命令行 --llm env


def _run(argv: list[str], **kw: Any) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, out=out, err=err, **kw)
    return code, out.getvalue(), err.getvalue()


def test_cli_stops_when_an_enabled_capability_is_rejected(llm_env):
    provider = Provider(logprobs_status=400)
    code, out, err = _run(
        ["--llm", "env", "--limit", "2"],
        client_factory=lambda: client_from_env(transport=provider),
    )
    assert code == 2 and out == ""
    assert "大模型能力检查（synthetic-model @ https://llm.example.invalid:8443" in err
    assert "大模型能力检查没通过" in err and "RODIT_LLM_LOGPROBS=off" in err
    assert len(provider.bodies) == 3  # 只发了检查请求，没跑评测
    assert KEY not in err and "user:pw" not in err


def test_cli_runs_with_switches_from_the_environment_and_records_them(llm_env, tmp_path):
    """RODIT_LLM_LOGPROBS=off、关思考模式：检查通过，评测的每个请求都照这个发，指纹和线上一致。"""
    llm_env.setenv("RODIT_LLM_EXTRA_BODY", THINKING_OFF)
    llm_env.setenv("RODIT_LLM_LOGPROBS", "off")
    provider = Provider(logprobs_status=400)
    path = tmp_path / "report.json"
    code, _, err = _run(
        ["--llm", "env", "--intent", "shipping_policy", "--limit", "2", "--json", str(path)],
        client_factory=lambda: client_from_env(transport=provider),
    )
    # 评测里的路由 / 生成请求由假服务回 OK（解析不了就降级），这里只看请求参数和指纹
    assert code == 0, err
    assert "logprobs：已关闭" in err
    assert len(provider.bodies) > 2
    assert all(b["thinking"] == {"type": "disabled"} for b in provider.bodies)
    assert all("logprobs" not in b for b in provider.bodies)
    fp = json.loads(path.read_text(encoding="utf-8"))["fingerprint"]
    assert fp["llm_options"] == running_llm_options(EngineConfig())
    assert json.loads(fp["llm_options"]) == {
        "extra_body": {"thinking": {"type": "disabled"}},
        "json_mode": True,
        "logprobs": False,
    }


# ================================================================ 指纹：请求参数也要一致


def test_eval_and_running_fingerprints_agree_on_request_options(llm_env):
    llm_env.setenv("RODIT_LLM_EXTRA_BODY", f" {THINKING_OFF}\n")
    llm_env.setenv("RODIT_LLM_JSON_MODE", "OFF")
    config = EngineConfig()
    evaluated = _run_fingerprint(env_client(config), config, None, None)
    running = current_fingerprint(config)
    assert evaluated.llm_options == running.llm_options is not None
    assert differences(evaluated, running) == []


def test_changed_request_options_need_a_new_verdict(llm_env):
    config = EngineConfig()
    verdict = current_fingerprint(config)
    llm_env.setenv("RODIT_LLM_EXTRA_BODY", THINKING_OFF)
    changed = current_fingerprint(config)
    assert not same_setup(verdict, changed)
    (diff,) = differences(verdict, changed)
    assert diff.startswith("大模型请求参数") and "disabled" in diff
    # 旧结论没记请求参数：对不上
    old = verdict.model_copy(update={"llm_options": None})
    assert any("没记录" in d for d in differences(old, verdict))
    # 线上的环境变量不合法：不知道线上发什么，对不上
    llm_env.setenv("RODIT_LLM_EXTRA_BODY", '{"api_key": "x"}')
    broken = current_fingerprint(config)
    assert broken.llm_options is None
    assert any("不合法" in d for d in differences(verdict, broken))


def test_config_pinned_endpoint_brings_its_own_options(llm_env):
    """配置钉了接口：评测和线上都用配置的 llm 段，环境变量的请求参数两边都不看。"""
    llm_env.setenv("RODIT_LLM_EXTRA_BODY", '{"enable_thinking": false}')
    config = EngineConfig.model_validate(
        {
            "llm": {
                "base_url": "https://cfg.example.invalid/v1",
                "model": "cfg-model",
                "extra_body": {"thinking": {"type": "disabled"}},
                "logprobs": False,
            }
        }
    )
    client = env_client(config)
    assert client.options == LLMRequestOptions({"thinking": {"type": "disabled"}}, True, False)
    evaluated = _run_fingerprint(client, config, None, None)
    running = current_fingerprint(config)
    assert evaluated.llm_options == running.llm_options == client.options.describe()
    assert differences(evaluated, running) == []


def test_offline_runs_record_no_request_options():
    assert describe_options(None) is None
    assert _run_fingerprint(None, None, None, None).llm_options is None
