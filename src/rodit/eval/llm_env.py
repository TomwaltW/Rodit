"""用真实大模型跑评测时的环境配置（`python -m rodit.eval --llm env`）。

大模型：rodit.llm.OpenAICompatibleClient.from_env——RODIT_LLM_BASE_URL、RODIT_LLM_MODEL（必填）、
RODIT_LLM_TIMEOUT_S（可选）；模型服务特有的请求参数 RODIT_LLM_EXTRA_BODY（JSON 对象，如关闭
思考模式）和能力开关 RODIT_LLM_JSON_MODE / RODIT_LLM_LOGPROBS（on / off，默认 on），规则见
core.config.env_request_options；密钥 RODIT_LLM_API_KEY 由客户端在每次调用时自己读。
部署配置里写了 llm.base_url 和 llm.model 时整组改用配置（client_from_config，和线上同一条规则）。
这里只检查密钥"有没有设置"，不读出、不打印、不写进报告；报告里的接口地址只留主机名
（URL 里的用户名密码、路径、查询参数都不要）。密钥只放本地 .env（已 gitignore）或
云端环境的密钥设置，不进仓库（CLAUDE.md）。

单价（§14，不是密钥）：命令行 --price-input / --price-output 优先，其次环境变量
RODIT_LLM_PRICE_INPUT_PER_1K / RODIT_LLM_PRICE_OUTPUT_PER_1K（每 1000 token 的价格）；
两个都没有时只报 token，只给了一个视为配置错误。

跑之前先做一次连通性检查（preflight）：一个几个 token 的请求，地址、模型名、密钥不对时立刻报错
（客户端的错误信息里已经抹掉了密钥），不白跑一整轮、也不让整轮结果全是降级。
能力检查（probe_capabilities）在它之后再发至多两个很小的请求：JSON 模式（response_format）
和 logprobs 接不接受，被拒绝时给出要改哪个开关 / 参数的提示——不同模型服务对这两个参数的
支持不一样，preflight 的请求里没有它们，过了也不代表路由、质检、生成的请求能过。
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from rodit.core.config import LLM_ENV_PREFIX, LLMConfig, LLMRequestOptions
from rodit.core.errors import LLMError, LLMTimeout
from rodit.core.ports import LLMClient, LLMMessage, LLMRequest, LLMResponse
from rodit.decision.voting import extract_json_object
from rodit.eval.perf import Pricing
from rodit.llm import OpenAICompatibleClient

ENV_PREFIX = LLM_ENV_PREFIX
PRICE_INPUT_ENV = "RODIT_LLM_PRICE_INPUT_PER_1K"
PRICE_OUTPUT_ENV = "RODIT_LLM_PRICE_OUTPUT_PER_1K"
KEY_HINT = "密钥只放本地 .env 或环境的密钥设置，不要提交进仓库"


def _require_key(client: OpenAICompatibleClient) -> OpenAICompatibleClient:
    if not os.environ.get(client.api_key_env, "").strip():
        raise RuntimeError(
            f"环境变量 {client.api_key_env} 未设置：没有大模型 API key（{KEY_HINT}）"
        )
    return client


def client_from_env(*, prefix: str = ENV_PREFIX, **kwargs: Any) -> OpenAICompatibleClient:
    """按环境变量构造客户端（kwargs 原样交给 OpenAICompatibleClient.from_env，测试用来注入
    transport）。地址或模型没设、密钥没设时抛 RuntimeError（信息里只有变量名）；请求参数
    （RODIT_LLM_EXTRA_BODY / JSON_MODE / LOGPROBS）不合法抛 ValueError。"""
    return _require_key(OpenAICompatibleClient.from_env(prefix=prefix, **kwargs))


def client_from_config(config: LLMConfig, **kwargs: Any) -> OpenAICompatibleClient:
    """和线上单元同一条规则构造客户端（OpenAICompatibleClient.from_config：配置里 llm.base_url 和
    llm.model 都写了整组用配置，否则整组取环境变量）。错误同 client_from_env。"""
    return _require_key(OpenAICompatibleClient.from_config(config, **kwargs))


def describe_endpoint(client: object) -> str | None:
    """报告里显示的接口地址：只留协议和主机（端口），没有 base_url 的客户端返回 None。"""
    base_url = getattr(client, "base_url", None)
    if not isinstance(base_url, str) or not base_url:
        return None
    parts = urlsplit(base_url)
    if not parts.hostname:
        return None
    try:
        port = f":{parts.port}" if parts.port else ""
    except ValueError:  # 端口不是数字：不显示
        port = ""
    return f"{parts.scheme}://{parts.hostname}{port}"


def describe_model(client: object) -> str | None:
    model = getattr(client, "model", None)
    return model if isinstance(model, str) and model else None


def client_options(client: object) -> LLMRequestOptions:
    """客户端实际发的请求参数（OpenAICompatibleClient.options；MeteredLLMClient 转发 inner 的）。
    不暴露请求参数的客户端（测试替身）按默认值算：不带特有参数，JSON 模式和 logprobs 照请求方的
    要求。"""
    options = getattr(client, "options", None)
    return options if isinstance(options, LLMRequestOptions) else LLMRequestOptions()


def describe_options(client: object) -> str | None:
    """评测指纹里的请求参数（LLMRequestOptions.describe）；没有客户端（离线替身）返回 None。"""
    return None if client is None else client_options(client).describe()


def _preflight_request() -> LLMRequest:
    return LLMRequest(
        purpose="other",
        messages=[LLMMessage(role="user", content="连通性检查，只回复：OK")],
        temperature=0.0,
        max_tokens=8,
    )


def preflight(client: LLMClient) -> None:
    """连通性检查：失败时照常抛 LLMError / LLMTimeout（调用方报错退出）。"""
    client.complete(_preflight_request())


# ================================================================ 能力检查

CheckName = Literal["plain", "json_mode", "logprobs"]
CheckStatus = Literal["ok", "rejected", "failed", "off", "skipped"]

CHECK_LABELS: dict[str, str] = {
    "plain": "普通调用",
    "json_mode": "JSON 模式",
    "logprobs": "logprobs",
}
STATUS_LABELS: dict[str, str] = {
    "ok": "通过",
    "rejected": "被拒绝",
    "failed": "没有结果",
    "off": "已关闭",
    "skipped": "没检查",
}
_HTTP_STATUS = re.compile(r"HTTP (\d{3})")
_THINKING_HINT = (
    f"默认开思考模式的模型要在 {ENV_PREFIX}EXTRA_BODY（或 llm.extra_body）里关闭，如 "
    '{"thinking":{"type":"disabled"}} 或 {"enable_thinking":false}，按服务商的写法'
)
JSON_OFF_HINT = (
    f"设 {ENV_PREFIX}JSON_MODE=off（或 llm.json_mode: false）：提示词本来就要求只输出 JSON，"
    "解析能容忍代码块和前后的文字"
)
LOGPROBS_OFF_HINT = (
    f"设 {ENV_PREFIX}LOGPROBS=off（或 llm.logprobs: false）：路由的置信度改按投票占比算"
)
JSON_PROBE_PROMPT = '能力检查：只输出这个 JSON 对象，不要别的文字：{"ok": true}'
LOGPROBS_PROBE_PROMPT = "能力检查：只回复一个大写字母 A，不要别的文字"


class CapabilityCheck(BaseModel):
    """一项能力检查的结果。error 是客户端的错误信息（只有状态码和错误类型 / 代码，密钥已抹掉）。"""

    model_config = ConfigDict(extra="forbid")

    name: CheckName
    status: CheckStatus
    latency_ms: int | None = None
    http_status: int | None = None
    error: str | None = None
    note: str | None = None  # 通过但有要注意的（输出不是 JSON、没返回 logprobs……）
    hint: str | None = None  # 要改哪个开关 / 参数


class CapabilityReport(BaseModel):
    """能力检查报告（probe_capabilities）。没有密钥：接口只留协议和主机。"""

    model_config = ConfigDict(extra="forbid")

    endpoint: str | None = None
    model: str | None = None
    options: str  # 生效的请求参数（LLMRequestOptions.describe）
    checks: list[CapabilityCheck]

    def check(self, name: CheckName) -> CapabilityCheck:
        return next(c for c in self.checks if c.name == name)

    @property
    def reachable(self) -> bool:
        """普通调用通过（地址、模型名、密钥都对）。"""
        return self.check("plain").status == "ok"

    @property
    def blocking(self) -> list[CapabilityCheck]:
        """会让整轮请求都失败的问题：普通调用不通，或开着的 JSON 模式 / logprobs 被拒绝。"""
        return [
            c
            for c in self.checks
            if (c.name == "plain" and c.status != "ok")
            or (c.name != "plain" and c.status == "rejected")
        ]

    @property
    def hints(self) -> list[str]:
        return [f"{CHECK_LABELS[c.name]}：{c.hint}" for c in self.checks if c.hint]


def _http_status(error: LLMError) -> int | None:
    status = getattr(error, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    match = _HTTP_STATUS.search(str(error))
    return int(match.group(1)) if match else None


def _is_rejection(status: int | None) -> bool:
    """服务明确拒绝了这个请求（4xx；超时 408、限流 429 不算：换个时间可能就过了）。"""
    return status is not None and 400 <= status < 500 and status not in (408, 429)


def _failure_hint(name: CheckName, status: int | None, timeout: bool) -> str:
    if timeout or status == 408:
        return (
            f"超时：先把 {ENV_PREFIX}TIMEOUT_S 调大量一次延迟（路由 decision.timeout_s、"
            f"生成 generation.timeout_s 是各环节自己的上限）；{_THINKING_HINT}"
        )
    if status == 429:
        return "限流：稍后再试或提高并发额度（路由每条消息并发 decision.vote_samples 次）"
    if name == "json_mode" and _is_rejection(status):
        return (
            f"该服务 / 模型不接受 response_format：{JSON_OFF_HINT}；"
            f"若是思考模式下不支持 JSON 模式，{_THINKING_HINT}"
        )
    if name == "logprobs" and _is_rejection(status):
        return f"该服务 / 模型不接受 logprobs：{LOGPROBS_OFF_HINT}"
    if status in (401, 403):
        return (
            "密钥不对或没有权限：检查 API key 环境变量"
            "（llm.api_key_env，默认 RODIT_LLM_API_KEY）的值"
        )
    if status == 404:
        return (
            f"接口地址或模型名不对：{ENV_PREFIX}BASE_URL 写到 /v1 这一级（客户端自己加 "
            f"/chat/completions），{ENV_PREFIX}MODEL 用服务商当前的模型 ID"
        )
    if _is_rejection(status):
        return (
            f"请求被拒绝：先确认 {ENV_PREFIX}MODEL；{_THINKING_HINT}；"
            "不接受 temperature=0 的服务（质检用 0）暂不支持"
        )
    return "服务端错误或网络不通：稍后再试；网络错误时检查代理和出网策略"


def _scrub(text: str, client: object) -> str:
    """错误信息里抹掉密钥（OpenAICompatibleClient 已经抹过；别的客户端再保险一次）。"""
    key_env = getattr(client, "api_key_env", None) or f"{ENV_PREFIX}API_KEY"
    key = os.environ.get(str(key_env), "").strip()
    return text.replace(key, "***") if len(key) >= 4 else text


def _attempt(
    client: LLMClient,
    name: CheckName,
    request: LLMRequest,
    timer: Callable[[], float],
) -> tuple[CapabilityCheck, LLMResponse | None]:
    started = timer()
    try:
        response = client.complete(request)
    except LLMError as e:
        latency_ms = max(0, round((timer() - started) * 1000))
        timeout = isinstance(e, LLMTimeout)
        status = None if timeout else _http_status(e)
        return (
            CapabilityCheck(
                name=name,
                status="rejected" if _is_rejection(status) else "failed",
                latency_ms=latency_ms,
                http_status=status,
                error=_scrub(str(e), client)[:200],
                hint=_failure_hint(name, status, timeout),
            ),
            None,
        )
    latency_ms = max(0, round((timer() - started) * 1000))
    return CapabilityCheck(name=name, status="ok", latency_ms=latency_ms), response


def probe_capabilities(
    client: LLMClient, *, timer: Callable[[], float] = time.perf_counter
) -> CapabilityReport:
    """检查这个模型服务接不接受我们会发的参数，至多三个很小的请求（purpose="other"）：

    1. 普通调用：就是 preflight 的请求（地址、模型名、密钥；不通就不再发后面的）；
    2. JSON 模式：要求只输出一个 JSON 对象（json_mode 开着时带 response_format，关着时只靠提示词），
       看接不接受、输出能不能解析（判断层的解析规则：容忍代码块和前后文字）；
    3. logprobs：开着时带 logprobs + top_logprobs，看接不接受、有没有返回；关着时不发。
    被拒绝（4xx，限流除外）、超时都记下状态码 / 错误类型和要改哪个开关、参数的提示；不抛异常。
    结果里没有密钥：接口只留协议和主机，错误信息只有状态码和错误类型 / 代码。"""
    options = client_options(client)
    report = CapabilityReport(
        endpoint=describe_endpoint(client),
        model=describe_model(client),
        options=options.describe(),
        checks=[],
    )
    plain, _ = _attempt(client, "plain", _preflight_request(), timer)
    report.checks.append(plain)
    if plain.status != "ok":
        report.checks.extend(
            CapabilityCheck(name=name, status="skipped", note="普通调用没通过，不再检查")
            for name in ("json_mode", "logprobs")
        )
        return report

    json_check, response = _attempt(
        client,
        "json_mode",
        LLMRequest(
            purpose="other",
            messages=[LLMMessage(role="user", content=JSON_PROBE_PROMPT)],
            temperature=0.0,
            max_tokens=32,
            json_mode=True,
        ),
        timer,
    )
    if response is not None and extract_json_object(response.text) is None:
        json_check.note = "接受了请求，但输出里找不到 JSON 对象"
        json_check.hint = (
            f"确认思考模式已关闭（{ENV_PREFIX}EXTRA_BODY），或换一个能按要求只输出 JSON 的模型"
        )
    if not options.json_mode:
        json_check.note = f"{ENV_PREFIX}JSON_MODE=off：不发 response_format，只靠提示词" + (
            f"；{json_check.note}" if json_check.note else ""
        )
    report.checks.append(json_check)

    if not options.logprobs:
        report.checks.append(
            CapabilityCheck(
                name="logprobs",
                status="off",
                note="不发 logprobs：路由的置信度按投票占比算",
            )
        )
        return report
    lp_check, response = _attempt(
        client,
        "logprobs",
        LLMRequest(
            purpose="other",
            messages=[LLMMessage(role="user", content=LOGPROBS_PROBE_PROMPT)],
            temperature=0.0,
            max_tokens=8,
            want_logprobs=True,
        ),
        timer,
    )
    if response is not None and not response.logprobs:
        lp_check.note = "接受了参数，但没返回选项概率"
        lp_check.hint = f"可以省掉这个参数：{LOGPROBS_OFF_HINT}"
    report.checks.append(lp_check)
    return report


def render_capabilities(report: CapabilityReport) -> list[str]:
    """能力检查的文本（逐行；命令行打到标准错误）。"""
    where = report.endpoint or "未知地址"
    model = report.model or "未知模型"
    lines = [f"大模型能力检查（{model} @ {where}；请求参数 {report.options}）："]
    for c in report.checks:
        parts = [STATUS_LABELS[c.status]]
        if c.latency_ms is not None:
            parts.append(f"{c.latency_ms} 毫秒")
        if c.error:
            parts.append(c.error)
        if c.note:
            parts.append(c.note)
        lines.append(f"  {CHECK_LABELS[c.name]}：" + "，".join(parts))
    lines.extend(f"  提示（{hint}）" for hint in report.hints)
    return lines


def _price(flag: float | None, env_name: str, env: Mapping[str, str]) -> float | None:
    if flag is not None:
        value = flag
    else:
        raw = env.get(env_name, "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"环境变量 {env_name} 必须是数字（每 1000 token 的价格）") from None
    if not 0 <= value < float("inf"):  # 也挡住 NaN、无穷大
        raise ValueError(f"单价必须是非负的有限数字（{env_name} 或对应的命令行参数）")
    return value


def resolve_pricing(
    price_input: float | None,
    price_output: float | None,
    env: Mapping[str, str] | None = None,
) -> Pricing | None:
    """命令行优先，其次环境变量；都没有返回 None（只报 token）。"""
    env = os.environ if env is None else env
    pi = _price(price_input, PRICE_INPUT_ENV, env)
    po = _price(price_output, PRICE_OUTPUT_ENV, env)
    if pi is None and po is None:
        return None
    if pi is None or po is None:
        raise ValueError(
            "输入、输出单价要一起给（--price-input / --price-output，或环境变量 "
            f"{PRICE_INPUT_ENV} / {PRICE_OUTPUT_ENV}）"
        )
    return Pricing(input_per_1k=pi, output_per_1k=po)


__all__ = [
    "PRICE_INPUT_ENV",
    "PRICE_OUTPUT_ENV",
    "CapabilityCheck",
    "CapabilityReport",
    "client_from_config",
    "client_from_env",
    "client_options",
    "describe_endpoint",
    "describe_model",
    "describe_options",
    "preflight",
    "probe_capabilities",
    "render_capabilities",
    "resolve_pricing",
]
