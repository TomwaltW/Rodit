"""接真实大模型（--llm env）前的启动检查和故障诊断。

启动检查（run_preflight）就是评测 --llm env 用的能力检查 rodit.eval.llm_env.probe_capabilities，
至多三个很小的请求，不另发探测：
1. 普通调用（连通性：地址、模型名、密钥）；
2. JSON 模式：生成和判断层的请求都要求只输出 JSON（json_mode）；RODIT_LLM_JSON_MODE=off 时请求
   不带 response_format，只靠提示词，照样检查输出里找不找得到 JSON 对象；
3. logprobs：只在判断层用大模型投票路由时检查（路由请求会带 logprobs，§6.4）；
   RODIT_LLM_LOGPROBS=off 时不发。
演示站比评测严（每条消息都要用 JSON）：普通调用、JSON 模式任何一步没有结果，或者输出里找不到
JSON 对象，都拒绝启动；logprobs 被拒绝（4xx）也拒绝启动；logprobs 超时、限流、没返回只提示。
没过就抛 PreflightFailed，带中文诊断：连通性问题按故障类型分（classify），参数被拒绝时写明要改
哪个开关或参数（RODIT_LLM_JSON_MODE / RODIT_LLM_LOGPROBS / RODIT_LLM_EXTRA_BODY，配置文件里是
llm.json_mode / llm.logprobs / llm.extra_body）；几项都没过时一次说全。诊断里只有接口主机名，
没有密钥、路径、查询参数，也没有模型服务返回的错误原文（客户端的错误信息本来就只带状态码和
错误类型 / 代码，见 rodit.llm.openai_compat）。

故障分类（classify / short_label）按客户端的错误信息认：
- 思考模式没关：模型只输出了思考过程（<think> 没结束，或正文为空、只有 reasoning_content；普通调用
  的 token 上限很小，默认开思考的模型多半在这一步就被截断），提示在 extra_body 里关闭思考模式；
- 出口被网络策略拦下（代理对 CONNECT 回 403 / 407："Tunnel connection failed: 403"），诊断里写
  要放行的域名（不带协议、端口），重试没有用；
- HTTP 401 密钥无效、403 没有权限、404 地址或模型 ID 不对、400 参数不被接受（JSON 模式、
  logprobs、思考模式这类专有参数）、429 限流或额度、5xx 服务端错误、3xx 重定向（不跟随）；
- 超时（LLMTimeout）、TLS 证书校验失败、域名解析失败、连接被拒绝、响应不是 OpenAI 兼容格式、
  内容安全拦截、没有设置密钥。
short_label 是同一套分类的短说明，给工作台过程面板里每一票的失败原因、自检报告用。
诊断里要改的设置名跟着接口的来源走：环境变量那一路是 RODIT_LLM_*，配置文件里写全了 llm.base_url 和
llm.model 时是 llm.*（RODIT_LLM_TIMEOUT_S 这些在那一路不起作用）；密钥是 llm.api_key_env 指的变量。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from rodit.core.config import LLM_ENV_PREFIX, LLMRequestOptions
from rodit.core.errors import LLMError, LLMTimeout
from rodit.core.ports import LLMClient
from rodit.eval.llm_env import (
    JSON_OFF_HINT,
    LOGPROBS_OFF_HINT,
    THINKING_HINT,
    CapabilityCheck,
    CapabilityReport,
    CheckName,
    SettingNames,
    client_options,
    describe_endpoint,
    error_kind,
    probe_capabilities,
    render_capabilities,
)

Kind = Literal[
    "thinking",
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

STAGES: dict[CheckName, str] = {
    "plain": "大模型连通性检查",
    "json_mode": "大模型 JSON 模式检查",
    "logprobs": "大模型 logprobs 检查",
}
"""启动检查各步的名字（"拒绝启动：<名字>没通过：……"）。"""

_HTTP_STATUS = re.compile(r"HTTP (\d{3})")
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
    network = error_kind(message)  # 思考模式没关、出口拦截、证书、域名、连接被拒（eval.llm_env）
    if network in ("thinking", "egress"):
        return network
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
    if network is not None:
        return network
    if "timed out" in message.lower():
        return "timeout"
    if _FILTERED in message:
        return "filtered"
    if _FORMAT.search(message):
        return "format"
    return "other"


SHORT_LABELS: dict[Kind, str] = {
    "thinking": "思考模式没关",
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


def _param_rejected(
    check: CheckName,
    host: str,
    status: int | None,
    options: LLMRequestOptions,
    names: SettingNames | None = None,
) -> str:
    """参数不被接受（HTTP 400 等）时的诊断：写明要改哪个开关或参数（见模块说明）。"""
    code = f"HTTP {status}" if status is not None else "被拒绝"
    if check == "json_mode":
        return (
            f"{host} 不接受 JSON 模式参数 response_format（{code}）：{JSON_OFF_HINT}；"
            f"如果是开着思考模式时不支持 JSON 模式，{THINKING_HINT}"
        )
    if check == "logprobs":
        return (
            f"{host} 不接受选项概率参数 logprobs（{code}）：判断层投票的路由请求会带 logprobs"
            f"（§6.4），这样每次路由都会失败、退回关键词规则。{LOGPROBS_OFF_HINT}；"
            "或者用 --judge offline（路由、质检用离线规则，只有生成走真实大模型）"
        )
    extra = (
        f"；{LLM_ENV_PREFIX}EXTRA_BODY（或 llm.extra_body）里的参数"
        f"（{'、'.join(sorted(options.extra_body))}）要是这个服务认的"
        if options.extra_body
        else ""
    )
    return (
        f"{host} 不接受请求参数（{code}）：检查 {(names or SettingNames()).model}"
        f"（模型 ID 要和控制台里的一致）；{THINKING_HINT}{extra}；"
        "不接受 temperature=0 的服务（质检、启动检查用 0）暂不支持"
    )


def classify(
    exc: BaseException,
    host: str,
    *,
    timeout_s: float | None = None,
    vote_samples: int | None = None,
    check: CheckName = "plain",
    options: LLMRequestOptions | None = None,
    names: SettingNames | None = None,
) -> str:
    """启动检查失败时给人看的中文诊断（只含主机名，见模块说明）。check 是哪一步的请求失败了
    （参数不被接受时据此说是哪个参数）；options 是客户端生效的请求参数；names 是要改的设置名
    （接口来自环境变量时是 RODIT_LLM_*，写在配置文件里时是 llm.*，
    见 eval.llm_env.SettingNames）。"""
    names = names or SettingNames()
    kind = kind_of(exc)
    status = _status(str(exc))
    if kind == "thinking":
        return f"{host} 的模型开着思考模式（只输出了思考过程、没有回答）：{THINKING_HINT}"
    if kind == "egress":
        domain = urlsplit(host).hostname if "://" in host else host
        return (
            f"这个环境的网络策略没放行 {host}：在云端环境设置的网络访问里把域名 {domain} "
            "加进允许的域名（改完要新开一个会话才生效，重试没有用）；在本地开发机上运行时"
            "检查公司代理或防火墙"
        )
    if kind == "no_key":
        return f"{exc}"
    if kind == "auth":
        return (
            f"{host} 拒绝了密钥（HTTP 401）：检查 {names.api_key} 是不是这个服务的有效密钥"
            "（密钥只放本地 .env 或环境的密钥设置，不要提交进仓库）"
        )
    if kind == "forbidden":
        return (
            f"{host} 拒绝访问（HTTP 403）：密钥没有这个模型的权限、账号没开通或欠费，"
            "或者服务限制了调用来源；到模型服务的控制台确认密钥权限和模型开通状态"
        )
    if kind == "not_found":
        return (
            f"{host} 找不到接口或模型（HTTP 404）：检查 {names.base_url}（OpenAI 兼容地址，"
            f"一般以 /v1 或服务文档写的兼容路径结尾，不含 /chat/completions）和 {names.model}"
            "（模型 ID 要和控制台里的一致）"
        )
    if kind == "bad_param":
        return _param_rejected(check, host, status, options or LLMRequestOptions(), names)
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
            f"{host} 返回了重定向（HTTP {status}）：检查 {names.base_url}"
            "（演示站不跟随重定向，避免把密钥带到别的地址）"
        )
    if kind == "timeout":
        limit = f" {timeout_s:g} 秒" if timeout_s is not None else "限定时间"
        return (
            f"{host} 在{limit}内没有响应：网络慢或者模型服务繁忙；"
            f"可以把 {names.timeout} 调大（单次调用的超时）再试"
        )
    if kind == "tls":
        return (
            f"和 {host} 的 TLS 证书校验失败：走企业代理时要把代理的根证书加进信任"
            "（如 SSL_CERT_FILE），不要关闭证书校验"
        )
    if kind == "dns":
        return f"解析不了 {host} 的域名：检查 {names.base_url} 的主机名和本机 DNS"
    if kind == "refused":
        return f"{host} 拒绝连接：地址或端口不对（{names.base_url}），或者服务没有启动"
    if kind == "format":
        return (
            f"{host} 的响应不是 OpenAI 兼容格式：检查 {names.base_url} 是不是 OpenAI 兼容接口"
            "（/chat/completions 的上一级）"
        )
    if kind == "filtered":
        return f"{host} 的内容安全策略拦截了探测请求的输出：换个模型或稍后再试"
    return f"调用 {host} 失败：{short_label(exc, timeout_s=timeout_s)}"


JSON_REQUIRED = "模型没有按 JSON 输出（JSON 模式检查）：判断层和生成都要求只输出 JSON"
"""输出里找不到 JSON 对象时诊断的开头（json_required 补上要改什么）。"""


def json_required(options: LLMRequestOptions) -> str:
    """接受了 JSON 模式检查的请求，但输出里找不到 JSON 对象：多半是思考模式没关。"""
    text = f"{JSON_REQUIRED}。{THINKING_HINT}"
    if not options.json_mode:
        text += (
            f"；这个服务支持 JSON 模式的话去掉 {LLM_ENV_PREFIX}JSON_MODE=off"
            "（或 llm.json_mode: false），让请求带上 response_format"
        )
    return text + "；还不行就换一个能按要求只输出 JSON 的模型"


class PreflightFailed(RuntimeError):
    """启动检查没过：message 是给人看的中文诊断（见 classify）；report 是能力检查的完整结果
    （命令行把它逐行打出来，再报拒绝启动）。"""

    def __init__(
        self, stage: str, message: str, kind: Kind, report: CapabilityReport | None = None
    ) -> None:
        super().__init__(f"{stage}没通过：{message}")
        self.stage = stage
        self.kind = kind
        self.report = report


@dataclass
class PreflightResult:
    """启动检查通过：接口主机、模型名和能力检查的完整结果（每步状态、耗时、提示）。"""

    host: str
    model: str
    report: CapabilityReport

    @property
    def steps_ms(self) -> dict[str, int]:
        return {c.name: c.latency_ms for c in self.report.checks if c.latency_ms is not None}

    @property
    def slowest_ms(self) -> int:
        return max(self.steps_ms.values(), default=0)

    @property
    def logprobs(self) -> bool | None:
        """路由投票要的选项概率接口给不给：检查通过时看有没有返回；
        没检查、已关闭、没有结果时为 None。"""
        check = self.report.check("logprobs")
        return check.output_ok if check.status == "ok" else None

    def lines(self) -> list[str]:
        """能力检查的文本（rodit.eval.llm_env.render_capabilities，和评测打的一样）。"""
        return render_capabilities(self.report)

    def view(self) -> dict[str, Any]:
        """自检报告（JSON）里的启动检查：能力检查的每一步、提示和文本。没有密钥。"""
        return {
            **self.report.model_dump(mode="json"),
            "hints": self.report.hints,
            "lines": self.lines(),
        }


def _as_error(check: CapabilityCheck) -> LLMError:
    """能力检查记下的错误（客户端的错误信息，已抹掉密钥）还原成异常，交给 classify 分类。"""
    message = check.error or "调用失败"
    return LLMTimeout(message) if check.timed_out else LLMError(message)


def _diagnose(
    check: CapabilityCheck,
    host: str,
    options: LLMRequestOptions,
    *,
    timeout_s: float | None,
    vote_samples: int | None,
    names: SettingNames,
) -> tuple[str, Kind] | None:
    """一步检查要不要拦下启动；要拦时返回 (诊断, 故障类型)（规则见模块说明）。"""
    if check.status in ("off", "skipped"):
        return None
    if check.status == "ok":
        if check.name == "json_mode" and check.output_ok is False:
            return json_required(options), "json"
        return None
    if check.name == "logprobs" and check.status != "rejected":
        return None  # 超时、限流、服务端错误：只提示，不拦
    if check.name != "plain" and check.status == "rejected":
        # 普通调用过了、同一个密钥和地址带上这个参数就被拒绝：就是这个参数的问题
        return _param_rejected(check.name, host, check.http_status, options, names), "bad_param"
    exc = _as_error(check)
    return (
        classify(
            exc,
            host,
            timeout_s=timeout_s,
            vote_samples=vote_samples,
            check=check.name,
            options=options,
            names=names,
        ),
        kind_of(exc),
    )


def run_preflight(
    client: LLMClient,
    *,
    judge_llm: bool,
    vote_samples: int | None = None,
    from_config: bool = False,
) -> PreflightResult:
    """启动检查（见模块说明）：judge_llm 是判断层用大模型投票路由（路由请求带 logprobs，要检查）；
    from_config 是接口写在配置文件里（llm.base_url / llm.model，诊断叫人改 llm.*，不是
    RODIT_LLM_*）。没过抛 PreflightFailed（第一步没过的名字 + 所有拦下启动的诊断）。"""
    host = endpoint_host(client)
    timeout_s = getattr(client, "timeout_s", None)
    options = client_options(client)
    names = SettingNames.of(client, from_config=from_config)
    report = probe_capabilities(client, check_logprobs=judge_llm, names=names)
    failures: list[tuple[CheckName, str, Kind]] = []
    for check in report.checks:
        found = _diagnose(
            check, host, options, timeout_s=timeout_s, vote_samples=vote_samples, names=names
        )
        if found is not None:
            failures.append((check.name, *found))
    if failures:
        name, message, kind = failures[0]
        for other, more, _ in failures[1:]:
            message += f"。另外{STAGES[other]}也没通过：{more}"
        raise PreflightFailed(STAGES[name], message, kind, report)
    return PreflightResult(host=host, model=str(getattr(client, "model", "") or ""), report=report)


__all__ = [
    "JSON_REQUIRED",
    "SHORT_LABELS",
    "STAGES",
    "PreflightFailed",
    "PreflightResult",
    "classify",
    "endpoint_host",
    "json_required",
    "kind_of",
    "run_preflight",
    "short_label",
]
