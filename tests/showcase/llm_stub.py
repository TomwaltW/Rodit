"""本地的 OpenAI 兼容大模型替身（只给测试用，不联网）。

StubLLM 在 127.0.0.1 的随机端口上起一个 ThreadingHTTPServer，实现 POST {base}/chat/completions：
检查 Bearer 密钥，按请求内容认出用途，回一个"像样"的回答，格式就是 rodit 的解析器要的那种：

- 判断层（system 是 rodit.decision.llm.SYSTEM_PROMPT）：从提示词里取出 state（JSON）和问的题，
  - 路由（有 intent 题）：用关键词规则（rodit.decision.RuleProvider）认问题类型，每一票选概率最高的
    类型；多个问题时追问 second_intent 答第二名；需要人工、多问题按 0.5 答"是 / 否"并给 p；
  - 质检（有 grounded 题）：用启发式判断（rodit.app.offline.HeuristicQAJudge）给每题答"是 / 否"
    和 p，语气打 3 分；
- 生成（用户消息里有【依据】）：{"answer": 依据里的一句话, "evidence_ids": [...],
  "need_clarify": false}（复述依据，写法同演示站离线替身 rodit.showcase.llm）；
- 能力检查（rodit.eval.llm_env.probe_capabilities，演示站启动检查也是它）：普通调用（preflight）
  回 OK，JSON 模式检查（probe_json）回 {"ok": true}，logprobs 检查（probe_logprobs）回 A。

usage 按字数估 token。logprobs：
- logprobs="tokens"：把输出切成小段当 token（JSON 模式下第一个有内容的 token 是键名，选项概率
  用不上——真实接口的样子）；
- logprobs="option"：路由请求的第一个 token 就是选中的问题类型，候选里还有第二名（让选项概率真的
  用上）；
- 默认（None）：不返回 logprobs。

模型服务的能力差异（整个服务的属性，不分用途；和下面按用途注入的失败不同）：
- accept_response_format=False：带 response_format 的请求一律 HTTP 400（不支持 JSON 模式的服务，
  要 RODIT_LLM_JSON_MODE=off）；
- accept_logprobs=False：带 logprobs 的请求一律 HTTP 400（要 RODIT_LLM_LOGPROBS=off）；
- thinking=True：默认开思考模式的模型——请求里没有 {"thinking": {"type": "disabled"}} 或
  {"enable_thinking": false}（RODIT_LLM_EXTRA_BODY）时，带 response_format 的请求 HTTP 400
  （思考模式下不支持 JSON 模式），不带的回答前面多一段 <think>…</think>（里面还有一个 JSON 草稿，
  客户端要去掉它）。thinking_style 改思考过程的样子：closed（默认，上面那样）；truncated（请求的
  token 上限 ≤ 64 时 </think> 来不及输出，只有思考的开头）；reasoning（思考放在 reasoning_content
  字段里，token 上限 ≤ 64 时正文为空）。
rejected 是因为这些被拒绝的请求数。

可以按用途（route / qa / generate / preflight / probe_json / probe_logprobs / "*"）设延迟（delay）
和注入失败（fail；另有 generate_retry：只对带着质检原因重写的那次生成生效）：
401 / 403 / 404 / 400 / 400_response_format（请求带 response_format 才拒）/ 400_logprobs（请求带
logprobs 才拒）/ 429 / 500 / timeout（睡够 timeout_sleep_s 秒再回）/ malformed（200 但响应体不是
JSON）/ not_json（200，内容不是 JSON）/ content_filter（finish_reason=content_filter）/ no_choices。
times 为空时一直失败，否则失败这么多次后恢复。

requests 记下每个请求（用途、状态码、请求体），测试据此核对：回放历史没有请求、路由一次请求
vote_samples 次、请求里没有密钥以外的认证信息等。
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from rodit.app.offline import HeuristicQAJudge
from rodit.core.decision import ChoiceAnswer, DecisionRequest, NoulAnswer, ScoreAnswer
from rodit.core.ports import LLMMessage, LLMRequest
from rodit.core.questions import (
    NO_SECOND_INTENT,
    QA_MODEL,
    QA_QUESTIONS,
    ROUTER_MODEL,
    ROUTER_QUESTIONS,
)
from rodit.decision import RuleProvider
from rodit.decision.llm import SYSTEM_PROMPT
from rodit.eval.llm_env import JSON_PROBE_PROMPT, LOGPROBS_PROBE_PROMPT
from rodit.showcase.llm import RETRY_MARK, ShowcaseLLM

STUB_KEY = "stub-test-key-0000"
"""测试用的假密钥（不是任何服务的真实密钥）。"""

_STATE = re.compile(r"## 状态（state）\n(?P<state>.*?)\n\n## 问题\n", re.S)
_QUESTION = re.compile(r"^### (?P<key>[a-z_]+)（", re.M)

PURPOSES = ("route", "qa", "generate", "preflight", "probe_json", "probe_logprobs", "other")

THINKING_PREFIX = '<think>先想一想：买家问的是什么？草稿 {"answer": "思考里的草稿"}</think>\n'
"""thinking=True 且没关思考时，回答前面多出来的思考过程（有的模型把它放在正文里）。"""
THINKING_TRUNCATED = "<think>\n嗯，用户让我只回复"
"""thinking_style="truncated"：请求的 token 上限小（≤ THINKING_CUTOFF_TOKENS）时
只来得及输出的思考开头。"""
THINKING_CUTOFF_TOKENS = 64


@dataclass
class StubRequest:
    purpose: str
    status: int
    body: dict[str, Any]
    authorization: str | None
    unsupported: bool = False  # 因为服务不支持的参数被拒绝


@dataclass
class _Failure:
    kind: str
    times: int | None  # None：一直失败


@dataclass
class StubLLM:
    """见模块说明。用 with StubLLM() as stub: 起停；stub.base_url 是 http://127.0.0.1:<端口>/v1。"""

    api_key: str = STUB_KEY
    model: str = "stub-chat"
    base_path: str = "/v1"
    logprobs: str | None = None
    timeout_sleep_s: float = 3.0
    accept_response_format: bool = True
    accept_logprobs: bool = True
    thinking: bool = False
    thinking_style: str = "closed"  # closed / truncated / reasoning，见模块说明
    requests: list[StubRequest] = field(default_factory=list)
    _delays: dict[str, float] = field(default_factory=dict)
    _failures: dict[str, list[_Failure]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _server: ThreadingHTTPServer | None = None
    _thread: threading.Thread | None = None

    # ---------------------------------------------------------------- 起停

    def start(self) -> StubLLM:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                stub._handle(self)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        self._server = server
        # 轮询间隔短一点：测试结束 shutdown 不用等默认的 0.5 秒
        self._thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> StubLLM:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def port(self) -> int:
        assert self._server is not None, "替身还没有启动"
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self.base_path}"

    # ---------------------------------------------------------------- 配置

    def delay(self, purpose: str, seconds: float) -> None:
        with self._lock:
            self._delays[purpose] = seconds

    def fail(self, purpose: str, kind: str, times: int | None = None) -> None:
        with self._lock:
            self._failures.setdefault(purpose, []).append(_Failure(kind, times))

    def reset(self) -> None:
        with self._lock:
            self._delays.clear()
            self._failures.clear()
            self.requests.clear()

    def count(self, purpose: str | None = None) -> int:
        with self._lock:
            return sum(1 for r in self.requests if purpose is None or r.purpose == purpose)

    def bodies(self, purpose: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [r.body for r in self.requests if purpose is None or r.purpose == purpose]

    @property
    def rejected(self) -> int:
        """因为服务不支持的参数被拒绝的请求数（response_format、logprobs、思考模式下的
        JSON 模式）。"""
        with self._lock:
            return sum(1 for r in self.requests if r.unsupported)

    def thinking_on(self, body: dict[str, Any]) -> bool:
        """这次请求思考模式开着：thinking=True 且请求没用服务特有的参数关掉它。"""
        if not self.thinking:
            return False
        disabled = (
            body.get("thinking") == {"type": "disabled"} or body.get("enable_thinking") is False
        )
        return not disabled

    def _unsupported(self, body: dict[str, Any]) -> str | None:
        """请求里有这个服务不支持的参数：返回错误体的键（_FAILURE_BODIES），否则 None。"""
        if "response_format" in body:
            if not self.accept_response_format:
                return "400_response_format"
            if self.thinking_on(body):
                return "400_thinking_json"
        if body.get("logprobs") and not self.accept_logprobs:
            return "400_logprobs"
        return None

    # ---------------------------------------------------------------- 处理

    def _failure_for(self, purpose: str, body: dict[str, Any]) -> str | None:
        keys = [purpose, "*"]
        if purpose == "generate" and RETRY_MARK in _user(body.get("messages", [])):
            keys.insert(0, "generate_retry")  # 带着质检原因重写的那次生成（§5.5）
        with self._lock:
            for key in keys:
                for f in self._failures.get(key, []):
                    if f.times is not None and f.times <= 0:
                        continue
                    if f.kind == "400_response_format" and "response_format" not in body:
                        continue
                    if f.kind == "400_logprobs" and not body.get("logprobs"):
                        continue
                    if f.times is not None:
                        f.times -= 1
                    return f.kind
        return None

    def _record(
        self,
        purpose: str,
        status: int,
        body: dict[str, Any],
        auth: str | None,
        *,
        unsupported: bool = False,
    ) -> None:
        with self._lock:
            self.requests.append(StubRequest(purpose, status, body, auth, unsupported))

    def _handle(self, h: BaseHTTPRequestHandler) -> None:
        try:
            length = int(h.headers.get("Content-Length") or 0)
            raw = h.rfile.read(length)
            auth = h.headers.get("Authorization")
            if h.path != f"{self.base_path}/chat/completions":
                self._record("other", 404, {}, auth)
                return _send(h, 404, {"error": {"type": "not_found", "code": "unknown_url"}})
            try:
                body = json.loads(raw.decode("utf-8"))
            except ValueError:
                self._record("other", 400, {}, auth)
                return _send(h, 400, {"error": {"type": "invalid_request_error"}})
            purpose = purpose_of(body)
            if auth != f"Bearer {self.api_key}":
                self._record(purpose, 401, body, auth)
                # 真实服务的错误消息里常带密钥片段：客户端不能把 message 带出去
                return _send(
                    h,
                    401,
                    {
                        "error": {
                            "type": "invalid_request_error",
                            "code": "invalid_api_key",
                            "message": f"Incorrect API key provided: {(auth or '')[-8:]}",
                        }
                    },
                )
            unsupported = self._unsupported(body)
            if unsupported is not None:
                self._record(purpose, 400, body, auth, unsupported=True)
                return _send(h, 400, _FAILURE_BODIES[unsupported])
            failure = self._failure_for(purpose, body)
            with self._lock:
                delay = self._delays.get(purpose, self._delays.get("*", 0.0))
            if delay:
                time.sleep(delay)
            status = _FAILURE_STATUS.get(failure or "", 200)
            self._record(purpose, status, body, auth)
            if failure == "timeout":
                time.sleep(self.timeout_sleep_s)
            elif failure == "malformed":
                return _send_raw(h, 200, b"<html>not json</html>", "text/html")
            elif failure in _FAILURE_BODIES:
                return _send(h, status, _FAILURE_BODIES[failure])
            text = self._answer(purpose, body)
            if failure == "not_json":
                text = "我认为这是尺码问题，建议选 M 码。"
            reasoning: str | None = None
            if self.thinking_on(body):
                short = int(body.get("max_tokens") or 0) <= THINKING_CUTOFF_TOKENS
                if self.thinking_style == "truncated" and short:
                    text = THINKING_TRUNCATED  # token 上限小：</think> 来不及输出
                elif self.thinking_style == "reasoning":
                    reasoning = THINKING_TRUNCATED.removeprefix("<think>\n")
                    text = None if short else text  # 思考在单独字段里；上限小时正文为空
                else:
                    text = THINKING_PREFIX + text
            return _send(h, 200, self._completion(purpose, body, text, failure, reasoning))
        except (BrokenPipeError, ConnectionResetError):  # 客户端已经超时走了
            return None

    def _completion(
        self,
        purpose: str,
        body: dict[str, Any],
        text: str | None,
        failure: str | None,
        reasoning: str | None = None,
    ) -> dict[str, Any]:
        if failure == "no_choices":
            return {"object": "chat.completion", "model": self.model}
        prompt_chars = sum(len(str(m.get("content", ""))) for m in body.get("messages", []))
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        choice: dict[str, Any] = {
            "index": 0,
            "message": message,
            "finish_reason": "content_filter" if failure == "content_filter" else "stop",
        }
        text = text or ""
        if body.get("logprobs") and self.logprobs:
            choice["logprobs"] = self._logprobs(purpose, text)
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "model": self.model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": max(1, prompt_chars // 2),
                "completion_tokens": max(1, len(text) // 2),
                "total_tokens": max(1, prompt_chars // 2) + max(1, len(text) // 2),
            },
        }

    def _logprobs(self, purpose: str, text: str) -> dict[str, Any]:
        if self.logprobs == "option" and purpose == "route":
            first = _route_ranking(text)
            top = [{"token": opt, "logprob": lp} for opt, lp in first]
            return {
                "content": [{"token": first[0][0], "logprob": first[0][1], "top_logprobs": top}]
            }
        tokens = [text[i : i + 3] for i in range(0, len(text), 3)]
        return {
            "content": [
                {"token": t, "logprob": -0.05, "top_logprobs": [{"token": t, "logprob": -0.05}]}
                for t in tokens
            ]
        }

    # ---------------------------------------------------------------- 回答

    def _answer(self, purpose: str, body: dict[str, Any]) -> str:
        messages = body.get("messages", [])
        if purpose == "preflight":
            return "OK"
        if purpose == "probe_json":
            return '{"ok": true}'
        if purpose == "probe_logprobs":
            return "A"
        if purpose == "route":
            return json.dumps(route_answer(_user(messages)), ensure_ascii=False)
        if purpose == "qa":
            return json.dumps(qa_answer(_user(messages)), ensure_ascii=False)
        if purpose == "generate":
            request = LLMRequest(
                purpose="generate",
                messages=[LLMMessage(role=m["role"], content=m["content"]) for m in messages],
            )
            return ShowcaseLLM().complete(request).text
        return "好的"


_FAILURE_STATUS = {
    "401": 401,
    "403": 403,
    "404": 404,
    "400": 400,
    "400_response_format": 400,
    "400_logprobs": 400,
    "429": 429,
    "500": 500,
}
_FAILURE_BODIES: dict[str, dict[str, Any]] = {
    "401": {"error": {"type": "authentication_error", "code": "invalid_api_key"}},
    "403": {"error": {"type": "permission_denied", "code": "model_not_allowed"}},
    "404": {"error": {"type": "not_found", "code": "model_not_found"}},
    "400": {"error": {"type": "invalid_request_error", "code": "invalid_parameter"}},
    "400_response_format": {
        "error": {"type": "invalid_request_error", "code": "response_format_unsupported"}
    },
    "400_logprobs": {"error": {"type": "invalid_request_error", "code": "logprobs_unsupported"}},
    "400_thinking_json": {
        "error": {
            "type": "invalid_request_error",
            "code": "response_format_requires_thinking_disabled",
        }
    },
    "429": {"error": {"type": "rate_limit_error", "code": "rate_limit_exceeded"}},
    "500": {"error": {"type": "server_error", "code": "internal_error"}},
}


def _send(h: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    _send_raw(
        h, status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json"
    )


def _send_raw(h: BaseHTTPRequestHandler, status: int, data: bytes, content_type: str) -> None:
    h.send_response(status)
    h.send_header("Content-Type", content_type)
    h.send_header("Content-Length", str(len(data)))
    h.end_headers()
    h.wfile.write(data)


def _user(messages: list[dict[str, Any]]) -> str:
    users = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
    return users[-1] if users else ""


def purpose_of(body: dict[str, Any]) -> str:
    """按请求内容认用途（rodit 不把 purpose 发给模型服务）。"""
    messages = body.get("messages") or []
    text = "\n".join(str(m.get("content", "")) for m in messages)
    system = next((str(m.get("content", "")) for m in messages if m.get("role") == "system"), "")
    if JSON_PROBE_PROMPT in text:
        return "probe_json"
    if LOGPROBS_PROBE_PROMPT in text:
        return "probe_logprobs"
    if "连通性检查" in text:
        return "preflight"
    if system == SYSTEM_PROMPT:
        user = _user(messages)
        if "### intent（" in user:
            return "route"
        if "### grounded（" in user:
            return "qa"
        return "other"
    if "【依据】" in text:
        return "generate"
    return "other"


def _state_and_questions(user: str) -> tuple[dict[str, Any], list[str]]:
    m = _STATE.search(user)
    state = json.loads(m.group("state")) if m else {}
    return state, _QUESTION.findall(user)


def _yes_no(p: float) -> dict[str, Any]:
    return {"answer": "是" if p > 0.5 else "否", "p": round(p, 2)}


def route_answer(user: str) -> dict[str, Any]:
    """一票路由答案：关键词规则认类型，每题照规则的结果答（见模块说明）。"""
    state, asked = _state_and_questions(user)
    request = DecisionRequest(model=ROUTER_MODEL, state=state, questions=ROUTER_QUESTIONS)
    answers = RuleProvider().decide(request).answers
    out: dict[str, Any] = {}
    intent = answers["intent"]
    assert isinstance(intent, ChoiceAnswer)
    ranked = sorted(intent.probabilities.items(), key=lambda kv: -kv[1])
    multi = answers["multi_intent"]
    assert isinstance(multi, NoulAnswer)
    for key in asked:
        if key == "intent":
            out[key] = {"choice": intent.choice}
        elif key == "second_intent":
            second = next(
                (o for o, p in ranked if o != intent.choice and p >= 0.15), NO_SECOND_INTENT
            )
            out[key] = {"choice": second if multi.noul > 0.5 else NO_SECOND_INTENT}
        elif isinstance(a := answers.get(key), NoulAnswer):
            out[key] = _yes_no(a.noul)
        elif isinstance(a, ScoreAnswer):
            out[key] = {"score": round(a.score)}
    return out


def _route_ranking(text: str) -> list[tuple[str, float]]:
    """logprobs="option"：选中的类型 logprob -0.2，再给一个候选 -1.8（让概率和票数不一样）。"""
    chosen = json.loads(text).get("intent", {}).get("choice", "other")
    other = "other" if chosen != "other" else "greeting"
    return [(chosen, -0.2), (other, -1.8)]


def qa_answer(user: str) -> dict[str, Any]:
    """一票质检答案：启发式判断（数字核对、依据外的承诺、站外引流），语气 3 分。"""
    state, asked = _state_and_questions(user)
    request = DecisionRequest(model=QA_MODEL, state=state, questions=dict(QA_QUESTIONS))
    answers = HeuristicQAJudge().decide(request).answers
    out: dict[str, Any] = {}
    for key in asked:
        a = answers.get(key)
        if isinstance(a, NoulAnswer):
            out[key] = _yes_no(a.noul)
        elif isinstance(a, ScoreAnswer):
            out[key] = {"score": round(a.score)}
    return out


__all__ = [
    "PURPOSES",
    "STUB_KEY",
    "THINKING_PREFIX",
    "StubLLM",
    "StubRequest",
    "purpose_of",
    "qa_answer",
]
