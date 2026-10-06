"""接真实大模型（--llm env）前的启动检查和故障诊断。

启动检查（run_preflight）：
1. rodit.eval.llm_env.preflight：几个 token 的连通性检查（地址、模型名、密钥不对时立刻报错）；
2. 一次 JSON 模式探测（json_mode，和生成、判断层的请求一样要求只输出 JSON）：判断层投票
   （--judge llm）时同时请求选项概率（want_logprobs，路由请求就是这样问的，§6.4）；
   探测被 HTTP 400 拒绝时，去掉 logprobs 再探一次，分清是 JSON 模式还是 logprobs 不被接受。
任何一步失败都抛 PreflightFailed，带一句中文诊断（classify）；诊断里只有接口主机名，
没有密钥、路径、查询参数，也没有模型服务返回的错误原文（客户端的错误信息本来就只带状态码和
错误类型 / 代码，见 rodit.llm.openai_compat）。

故障分类（classify / short_label）按客户端的错误信息认：
- 出口被网络策略拦下（代理对 CONNECT 回 403 / 407："Tunnel connection failed: 403"）；
- HTTP 401 密钥无效、403 没有权限、404 地址或模型 ID 不对、400 参数不被接受（JSON 模式、
  logprobs、思考模式这类专有参数）、429 限流或额度、5xx 服务端错误、3xx 重定向（不跟随）；
- 超时（LLMTimeout）、TLS 证书校验失败、域名解析失败、连接被拒绝、响应不是 OpenAI 兼容格式、
  内容安全拦截、没有设置密钥。
short_label 是同一套分类的短说明，给工作台过程面板里每一票的失败原因、自检报告用。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from rodit.core.errors import LLMError, LLMTimeout
from rodit.core.ports import LLMClient, LLMMessage, LLMRequest, LLMResponse
from rodit.decision.voting import extract_json_object
from rodit.eval.llm_env import describe_endpoint, preflight

Kind = Literal[
    "egress",
    "auth",
    "forbidden",
    "not_found",
    "bad_param",
    "rate_limit",
    "server",
    "redirect",
    "timeout",
    "tls",
    "dns",
    "refused",
    "format",
    "filtered",
    "no_key",
    "json",
    "other",
]

PROBE_MARK = "演示站启动前的接口检查"
"""JSON 模式探测请求的标记（测试用的本地模型替身据此认出探测请求）。"""

PROBE_MESSAGES = (
    LLMMessage(
        role="system",
        content=f"这是{PROBE_MARK}。只输出一个 JSON 对象，不要输出 JSON 以外的任何文字。",
    ),
    LLMMessage(role="user", content='请原样输出这个 JSON：{"ok": true}'),
)

_HTTP_STATUS = re.compile(r"HTTP (\d{3})")
_EGRESS = re.compile(r"Tunnel connection failed|ProxyError|Proxy Authentication Required", re.I)
_TLS = re.compile(r"CERTIFICATE_VERIFY_FAILED|certificate verify failed|SSLCertVerification", re.I)
_TLS_OTHER = re.compile(r"\[SSL[:\]]|SSLError|WRONG_VERSION_NUMBER", re.I)
_DNS = re.compile(
    r"Name or service not known|nodename nor servname|getaddrinfo failed|"
    r"Temporary failure in name resolution|No address associated with hostname",
    re.I,
)
_REFUSED = re.compile(r"Connection refused|ConnectionRefusedError|Errno 111|Errno 61", re.I)
_FORMAT = re.compile(r"响应格式不对|JSONDecodeError|不是 JSON 对象|Expecting value", re.I)
_NO_KEY = "没有大模型 API key"
_FILTERED = "内容安全策略拦截"


def endpoint_host(client: object) -> str:
    """诊断里显示的接口主机（协议 + 主机 + 端口，没有路径）；客户端没有地址时返回"大模型接口"。"""
    return describe_endpoint(client) or "大模型接口"


def _status(message: str) -> int | None:
    m = _HTTP_STATUS.search(message)
    return int(m.group(1)) if m else None


def kind_of(exc: BaseException) -> Kind:
    """按客户端的错误信息分类（见模块说明）。"""
    if isinstance(exc, LLMTimeout):
        return "timeout"
    message = str(exc)
    if _NO_KEY in message:
        return "no_key"
    if _EGRESS.search(message):
        return "egress"
    status = _status(message)
    if status is not None:
        if status == 401:
            return "auth"
        if status == 403:
            return "forbidden"
        if status == 404:
            return "not_found"
        if status == 429:
            return "rate_limit"
        if status >= 500:
            return "server"
        if 300 <= status < 400:
            return "redirect"
        if status in (400, 413, 415, 422):
            return "bad_param"
        return "other"
    if _TLS.search(message) or _TLS_OTHER.search(message):
        return "tls"
    if _DNS.search(message):
        return "dns"
    if _REFUSED.search(message):
        return "refused"
    if "timed out" in message.lower():
        return "timeout"
    if _FILTERED in message:
        return "filtered"
    if _FORMAT.search(message):
        return "format"
    return "other"


SHORT_LABELS: dict[Kind, str] = {
    "egress": "网络策略拦截",
    "auth": "密钥无效（HTTP 401）",
    "forbidden": "没有权限（HTTP 403）",
    "not_found": "地址或模型不对（HTTP 404）",
    "bad_param": "参数不被接受（HTTP 400）",
    "rate_limit": "限流或额度用完（HTTP 429）",
    "server": "服务端错误（HTTP 5xx）",
    "redirect": "接口返回重定向",
    "timeout": "超时",
    "tls": "TLS 证书校验失败",
    "dns": "域名解析失败",
    "refused": "连接被拒绝",
    "format": "响应格式不对",
    "filtered": "内容安全拦截",
    "no_key": "没有设置密钥",
    "json": "没有按 JSON 输出",
    "other": "调用失败",
}


def short_label(exc: BaseException, *, timeout_s: float | None = None) -> str:
    """一次失败调用的短说明（不含错误原文）：超时带上限时秒数，HTTP 错误带状态码。"""
    kind = kind_of(exc)
    if kind == "timeout" and timeout_s is not None:
        return f"超时（限时 {timeout_s:g} 秒）"
    status = _status(str(exc))
    if kind in ("server", "bad_param", "redirect", "other") and status is not None:
        return f"{SHORT_LABELS[kind].split('（')[0]}（HTTP {status}）"
    return SHORT_LABELS[kind]


def classify(
    exc: BaseException,
    host: str,
    *,
    timeout_s: float | None = None,
    logprobs_rejected: bool = False,
    vote_samples: int | None = None,
) -> str:
    """启动检查失败时给人看的中文诊断（只含主机名，见模块说明）。"""
    kind = kind_of(exc)
    status = _status(str(exc))
    if kind == "egress":
        return (
            f"这个环境的网络策略没放行 {host}：在云端环境设置的网络访问里把它加进允许的域名"
            "（改完要新开一个会话才生效）；在本地开发机上运行时检查公司代理或防火墙"
        )
    if kind == "no_key":
        return f"{exc}"
    if kind == "auth":
        return (
            f"{host} 拒绝了密钥（HTTP 401）：检查 RODIT_LLM_API_KEY 是不是这个服务的有效密钥"
            "（密钥只放本地 .env 或环境的密钥设置，不要提交进仓库）"
        )
    if kind == "forbidden":
        return (
            f"{host} 拒绝访问（HTTP 403）：密钥没有这个模型的权限、账号没开通或欠费，"
            "或者服务限制了调用来源；到模型服务的控制台确认密钥权限和模型开通状态"
        )
    if kind == "not_found":
        return (
            f"{host} 找不到接口或模型（HTTP 404）：检查 RODIT_LLM_BASE_URL（OpenAI 兼容地址，"
            "一般以 /v1 或服务文档写的兼容路径结尾，不含 /chat/completions）和 RODIT_LLM_MODEL"
            "（模型 ID 要和控制台里的一致）"
        )
    if kind == "bad_param":
        if logprobs_rejected:
            return (
                f"{host} 不接受选项概率参数 logprobs（HTTP {status}）：判断层投票的路由请求会带"
                " logprobs（§6.4），这样每次路由都会失败、退回关键词规则。换一个支持 logprobs 的"
                "模型，或用 --judge offline（路由、质检用离线规则，只有生成走真实大模型）"
            )
        return (
            f"{host} 不接受请求参数（HTTP {status}）：演示站的请求要求 JSON 模式"
            "（response_format: json_object）；有的模型不支持 JSON 模式，或者要求关闭思考模式"
            "这类专有参数。换一个支持 JSON 模式的模型（或它的非思考版本）"
        )
    if kind == "rate_limit":
        votes = f"（路由投票一次并发 {vote_samples} 个请求）" if vote_samples else ""
        return (
            f"{host} 限流或额度用完（HTTP 429）：稍等再试，或在模型服务的控制台检查账户余额、"
            f"调用频率和并发上限{votes}"
        )
    if kind == "server":
        return f"{host} 服务端出错（HTTP {status}）：模型服务的问题，稍后再试"
    if kind == "redirect":
        return (
            f"{host} 返回了重定向（HTTP {status}）：检查 RODIT_LLM_BASE_URL"
            "（演示站不跟随重定向，避免把密钥带到别的地址）"
        )
    if kind == "timeout":
        limit = f" {timeout_s:g} 秒" if timeout_s is not None else "限定时间"
        return (
            f"{host} 在{limit}内没有响应：网络慢或者模型服务繁忙；"
            "可以设 RODIT_LLM_TIMEOUT_S 调大单次调用的超时再试"
        )
    if kind == "tls":
        return (
            f"和 {host} 的 TLS 证书校验失败：走企业代理时要把代理的根证书加进信任"
            "（如 SSL_CERT_FILE），不要关闭证书校验"
        )
    if kind == "dns":
        return f"解析不了 {host} 的域名：检查 RODIT_LLM_BASE_URL 的主机名和本机 DNS"
    if kind == "refused":
        return f"{host} 拒绝连接：地址或端口不对，或者服务没有启动"
    if kind == "format":
        return (
            f"{host} 的响应不是 OpenAI 兼容格式：检查 RODIT_LLM_BASE_URL 是不是 OpenAI 兼容接口"
            "（/chat/completions 的上一级）"
        )
    if kind == "filtered":
        return f"{host} 的内容安全策略拦截了探测请求的输出：换个模型或稍后再试"
    return f"调用 {host} 失败：{short_label(exc, timeout_s=timeout_s)}"


JSON_REQUIRED = (
    "模型没有按 JSON 输出（JSON 模式探测）：判断层和生成都要求只输出 JSON。"
    "换一个支持 JSON 模式的模型，或者用它的非思考 / 非推理版本"
)


class PreflightFailed(RuntimeError):
    """启动检查没过：message 是给人看的中文诊断（见 classify）。"""

    def __init__(self, stage: str, message: str, kind: Kind) -> None:
        super().__init__(f"{stage}没通过：{message}")
        self.stage = stage
        self.kind = kind


@dataclass
class PreflightResult:
    """启动检查的结果：接口主机、模型名、每步耗时（毫秒）、探测有没有返回选项概率。"""

    host: str
    model: str
    steps_ms: dict[str, int] = field(default_factory=dict)
    logprobs: bool | None = None  # 没请求时为 None

    @property
    def slowest_ms(self) -> int:
        return max(self.steps_ms.values(), default=0)


def _probe_request(*, logprobs: bool) -> LLMRequest:
    return LLMRequest(
        purpose="other",
        messages=list(PROBE_MESSAGES),
        temperature=0.0,
        max_tokens=32,
        json_mode=True,
        want_logprobs=logprobs,
    )


def _timed(fn, *args):  # noqa: ANN001, ANN202
    started = time.perf_counter()
    result = fn(*args)
    return result, int((time.perf_counter() - started) * 1000)


def run_preflight(
    client: LLMClient, *, judge_llm: bool, vote_samples: int | None = None
) -> PreflightResult:
    """启动检查（见模块说明）。失败抛 PreflightFailed。"""
    host = endpoint_host(client)
    timeout_s = getattr(client, "timeout_s", None)
    result = PreflightResult(host=host, model=str(getattr(client, "model", "") or ""))
    stage = "连通性检查"
    try:
        _, result.steps_ms["connect"] = _timed(preflight, client)
    except LLMError as exc:
        raise PreflightFailed(
            stage,
            classify(exc, host, timeout_s=timeout_s, vote_samples=vote_samples),
            kind_of(exc),
        ) from None
    stage = "JSON 模式探测"
    response: LLMResponse
    try:
        response, result.steps_ms["json"] = _timed(
            client.complete, _probe_request(logprobs=judge_llm)
        )
    except LLMError as exc:
        logprobs_rejected = False
        if judge_llm and kind_of(exc) == "bad_param":
            # 分清是 JSON 模式还是 logprobs 不被接受：去掉 logprobs 再探一次
            try:
                client.complete(_probe_request(logprobs=False))
                logprobs_rejected = True
            except LLMError:
                logprobs_rejected = False
        raise PreflightFailed(
            stage,
            classify(
                exc,
                host,
                timeout_s=timeout_s,
                logprobs_rejected=logprobs_rejected,
                vote_samples=vote_samples,
            ),
            kind_of(exc),
        ) from None
    if extract_json_object(response.text) is None:
        raise PreflightFailed(stage, JSON_REQUIRED, "json")
    if judge_llm:
        result.logprobs = response.logprobs is not None
    return result


__all__ = [
    "JSON_REQUIRED",
    "PROBE_MARK",
    "SHORT_LABELS",
    "PreflightFailed",
    "PreflightResult",
    "classify",
    "endpoint_host",
    "kind_of",
    "run_preflight",
    "short_label",
]
