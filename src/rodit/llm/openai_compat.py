"""OpenAI 兼容 /chat/completions 客户端（D5 境内已备案模型；方案 §4.4、§6.4、§12.1、§12.2）。

很多境内已备案模型都提供 OpenAI 兼容接口；具体用哪个模型是 D5 待决策项，
base_url / model 由部署配置决定，代码里不写死。实现 core.ports.LLMClient：

- 密钥：每次调用时从环境变量读（默认 RODIT_LLM_API_KEY），不保存在对象上；缺失、或者取值
  中间有换行、有非 ASCII 字符（多行的 .env 取值、带了中文引号）时抛 LLMError，消息里只有变量名。
  密钥不出现在 repr、异常消息、日志里：传输层自己抛的 LLMError 先抹掉密钥；别的异常只报类型名，
  不带异常的文字——里面可能有转义过的密钥（http.client 报请求头不合法时把整个头值打出来，
  按原文抹不掉）、接口地址的路径和查询参数、代理地址。
- 超时：request.timeout_s 优先，否则用构造时的 timeout_s（路由 ≤ 1 秒由调用方传入，§12.1）。
- 错误（§12.2 调用方据此退回 FAQ / 模板）：超时 → LLMTimeout；HTTP 429 / 5xx / 其他非 2xx、
  网络错误、响应格式不对 → LLMError。不重试：延迟预算紧，降级比重试更可控。
- 失败原因（core.errors.LLMError 的 status / kind / code；调用方、计量、评测报告只读这三项，
  不解析消息）：
  - kind：超时 timeout；429 rate_limit；401 / 403 auth；402 billing；其他 4xx bad_request；
    5xx server；3xx redirect；连不上 network；响应读不出来（不是 JSON 对象、没有 choices、
    没有文本）malformed；finish_reason 是 content_filter / sensitive → content_filter；
    密钥没设或不能用 → config。
  - status：HTTP 状态码。非 2xx 时带；响应读不出来（malformed）时也照带，哪怕是 200。
    没收到响应（超时、连不上、密钥不能用）就没有；内容安全拦截也不带，评测按 kind 归类。
  - code：响应体里的 error.code，没有就用 error.type；抹掉密钥、最多 64 个字符，
    不取 error.message（原文可能回显提示词里的买家文字）。
  传输层自己抛的 LLMError 按原类型重抛，这三项照带（code 同样抹掉密钥）。
- finish_reason：接口返回什么就填进 LLMResponse.finish_reason（没给是 None）。被 max_tokens 截断
  （length）、文本是空的都照常返回，不抛异常：token 已经花了要照常计量，算不算数由调用方定。
- json_mode → response_format {"type": "json_object"}；usage → input_tokens / output_tokens。
- extra_body：模型服务特有的请求参数（如关闭思考模式），并入每次请求，标准字段优先。取值来自
  配置的 llm.extra_body（core.config.LLMConfig：算进配置指纹，不放密钥），由组装客户端的地方
  在构造时传进来（线上是 app.production；评测见 docs/架构与分工.md §4.4）。from_env 不从
  环境变量读它：那样给的值不在指纹里。
- want_logprobs → 请求 logprobs + top_logprobs，读出"选项 → 概率"放进 LLMResponse.logprobs（§6.4）：
  取第一个有内容的输出位置（跳过只有空白、引号、括号、冒号、逗号的 token），
  把该位置的候选 token 去掉这些字符后作为选项，概率 = exp(logprob)，同名选项相加（上限 1）。
  适用于"先输出选项标签"的提示词；JSON 模式下第一个有内容的 token 往往是键名，不适合用。
  接口不返回 logprobs 时为 None。
- request.purpose / shop_id 只用于本地计量（MeteredLLMClient），不发给模型服务。

默认传输层用标准库 urllib（不引入新依赖）；测试注入假的 transport，不访问网络。
默认传输层不跟随重定向：urllib 跟随 3xx 时会把 Authorization 原样带到新地址（别的主机、
甚至明文 HTTP），密钥就泄露了。模型接口的 POST 本来就不需要重定向，3xx 按非 2xx 报 LLMError。
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from rodit.core.errors import LLMError, LLMErrorKind, LLMTimeout
from rodit.core.ports import LLMRequest, LLMResponse

Transport = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, dict[str, Any]]]
"""transport(url, headers, body, timeout_s) → (HTTP 状态码, 解析后的 JSON 响应体)。

超时抛 TimeoutError；网络错误抛 OSError 等；2xx 但响应体不是 JSON 对象时抛 ValueError；
非 2xx 返回状态码和能解析出来的响应体（解析不了就是空 dict）。"""

_SKIP_CHARS = " \t\r\n\"'`{}[]:,"
_FILTERED = frozenset({"content_filter", "sensitive"})  # 模型服务的内容安全拦截


def _parse_body(raw: bytes) -> dict[str, Any]:
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("响应体不是 JSON 对象")
    return parsed


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """不跟随任何重定向：3xx 作为 HTTPError 交回调用方，Authorization 不会被带到别处。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _urlopen(request: urllib.request.Request, timeout: float) -> Any:
    # 每次新建 opener：代理设置按调用时的环境变量读取（和 urllib.request.urlopen 一样走代理）
    return urllib.request.build_opener(_RefuseRedirects).open(request, timeout=timeout)


def urllib_transport(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float
) -> tuple[int, dict[str, Any]]:
    """默认传输层（标准库 urllib，走环境里配置的 HTTP(S) 代理，不跟随重定向）。"""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with _urlopen(req, timeout=timeout_s) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:  # 非 2xx：尽量带回错误体，解析不了就算了
        try:
            raw_err = e.read()
        except OSError:
            raw_err = b""
        finally:
            e.close()
        try:
            return e.code, _parse_body(raw_err)
        except ValueError:
            return e.code, {}
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise TimeoutError("连接超时") from e
        raise
    return status, _parse_body(raw)


def _scrub(text: str, secret: str, *, limit: int = 200) -> str:
    return (text.replace(secret, "***") if secret else text)[:limit]


def _short_code(value: Any, key: str) -> str | None:
    """错误代码整理成短字符串（LLMError.code）：抹掉密钥，最多 64 个字符；不是字符串 / 整数、
    或者是空的，返回 None。"""
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    return _scrub(str(value).strip(), key, limit=64) or None


def _error_code(body: Mapping[str, Any], key: str) -> str | None:
    """模型服务给的错误代码：响应体里的 error.code，没有就用 error.type。
    不取 error.message：原文可能回显提示词里的买家文字。"""
    err = body.get("error")
    if not isinstance(err, dict):
        return None
    return _short_code(err.get("code"), key) or _short_code(err.get("type"), key)


def _http_kind(status: int) -> LLMErrorKind:
    """非 2xx 的状态码 → 失败原因（LLMError.kind）。"""
    if 300 <= status < 400:
        return "redirect"
    if status == 429:
        return "rate_limit"
    if status in (401, 403):
        return "auth"
    if status == 402:
        return "billing"
    if status >= 500:
        return "server"
    return "bad_request"


def _exception_types(e: BaseException) -> str:
    """异常的类型名，不带异常的文字。urllib 的 URLError 把真正的原因放在 reason 里，
    那一层的类型名也带上（"URLError（ConnectionRefusedError）"），不然域名解析不了、连接被拒、
    证书不对看起来都一样。"""
    name = type(e).__name__
    reason = getattr(e, "reason", None)
    if isinstance(reason, BaseException):
        name += f"（{type(reason).__name__}）"
    return name


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def _option_probabilities(logprobs: Any) -> dict[str, float] | None:
    """OpenAI 格式的 choices[0].logprobs → {选项: 概率}（规则见模块文档字符串）。"""
    if not isinstance(logprobs, dict) or not isinstance(logprobs.get("content"), list):
        return None
    for position in logprobs["content"]:
        if not isinstance(position, dict):
            continue
        token = position.get("token")
        if not isinstance(token, str) or not token.strip(_SKIP_CHARS):
            continue
        candidates = position.get("top_logprobs")
        if not isinstance(candidates, list) or not candidates:
            candidates = [position]
        probs: dict[str, float] = {}
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            t, lp = cand.get("token"), cand.get("logprob")
            if not isinstance(t, str) or isinstance(lp, bool) or not isinstance(lp, int | float):
                continue
            if math.isnan(lp):
                continue
            option = t.strip(_SKIP_CHARS)
            if option:
                probs[option] = min(1.0, probs.get(option, 0.0) + math.exp(min(0.0, lp)))
        chosen, chosen_lp = token.strip(_SKIP_CHARS), position.get("logprob")
        if (
            chosen not in probs
            and isinstance(chosen_lp, int | float)
            and not isinstance(chosen_lp, bool)
            and not math.isnan(chosen_lp)
        ):
            probs[chosen] = math.exp(min(0.0, chosen_lp))
        return probs or None
    return None


def _text_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # 少数服务返回分段内容 [{"type": "text", "text": ...}]
        parts = [p.get("text") for p in content if isinstance(p, dict)]
        if parts and all(isinstance(p, str) for p in parts):
            return "".join(parts)
    return None


class OpenAICompatibleClient:
    """实现 core.ports.LLMClient，调用 OpenAI 兼容的 {base_url}/chat/completions。

    - top_logprobs：want_logprobs 时每个位置返回的候选数。
    - extra_body：模型服务特有的参数（如关闭思考模式），会被标准字段覆盖；组装时传配置里的
      llm.extra_body。
    线程安全（无可变状态），可以给自一致性投票并发调用。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "RODIT_LLM_API_KEY",
        timeout_s: float = 10.0,
        transport: Transport | None = None,
        top_logprobs: int = 5,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("base_url 必须以 http:// 或 https:// 开头")
        if not model:
            raise ValueError("model 不能为空")
        if timeout_s <= 0:
            raise ValueError("timeout_s 必须为正数")
        if not 1 <= top_logprobs <= 20:
            raise ValueError("top_logprobs 取值 1–20")
        if not api_key_env:
            raise ValueError("api_key_env 不能为空")
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.top_logprobs = top_logprobs
        self._extra_body = dict(extra_body or {})
        self._transport: Transport = transport or urllib_transport

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleClient(base_url={self.base_url!r}, model={self.model!r}, "
            f"api_key_env={self.api_key_env!r})"
        )

    @classmethod
    def from_env(cls, *, prefix: str = "RODIT_LLM_", **kwargs: Any) -> OpenAICompatibleClient:
        """从环境变量构造：{prefix}BASE_URL、{prefix}MODEL（必填），{prefix}TIMEOUT_S（可选）；
        密钥仍在每次调用时从 {prefix}API_KEY 读。

        值去掉首尾空白（从密钥文件、挂载文件注入时常带换行），和 core.config.llm_endpoint 同一条
        规则：启动前检查、就绪检查、评测指纹描述的就是这里造出来的客户端（模型名多一个空格，
        评测记下的和线上算出来的就对不上）。"""
        base_url = os.environ.get(f"{prefix}BASE_URL", "").strip()
        model = os.environ.get(f"{prefix}MODEL", "").strip()
        if not base_url or not model:
            raise RuntimeError(
                f"环境变量 {prefix}BASE_URL 和 {prefix}MODEL 必须设置（D5 所选模型）"
            )
        timeout = os.environ.get(f"{prefix}TIMEOUT_S", "").strip()
        if timeout:
            kwargs.setdefault("timeout_s", float(timeout))
        kwargs.setdefault("api_key_env", f"{prefix}API_KEY")
        return cls(base_url=base_url, model=model, **kwargs)

    # ---------------------------------------------------------------- 请求

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **self._extra_body,
            "model": self.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if request.want_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self.top_logprobs
        return payload

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise LLMError(f"环境变量 {self.api_key_env} 未设置：没有大模型 API key", kind="config")
        if "\r" in key or "\n" in key or not key.isascii():
            # 不能放进请求头。在这里拦下，消息里只有变量名：交给 http.client 的话，它报错时
            # 会把整个头值（也就是密钥）打出来
            raise LLMError(
                f"环境变量 {self.api_key_env} 的取值不能当 API key 用：中间有换行或非 ASCII 字符",
                kind="config",
            )
        timeout = request.timeout_s if request.timeout_s is not None else self.timeout_s
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            status, body = self._transport(
                self.endpoint, headers, self.build_payload(request), timeout
            )
        except LLMError as e:  # 传输层自己抛的：抹掉密钥后按原类型重抛，失败原因照带
            raise type(e)(
                _scrub(str(e), key), status=e.status, kind=e.kind, code=_short_code(e.code, key)
            ) from None
        except TimeoutError:
            raise LLMTimeout(f"大模型调用超时（{timeout:g} 秒）") from None
        except Exception as e:
            # 网络错误、响应不是 JSON 等：统一成 LLMError。只报异常类型，不带异常的文字
            # （见模块说明）。按传输层的约定，ValueError 是 2xx 但响应体不是 JSON 对象
            raise LLMError(
                f"大模型接口调用失败：{_exception_types(e)}",
                kind="malformed" if isinstance(e, ValueError) else "network",
            ) from None
        return self._parse(status, body, request, key)

    # ---------------------------------------------------------------- 响应

    def _parse(self, status: int, body: Any, request: LLMRequest, key: str) -> LLMResponse:
        if not isinstance(body, dict):
            raise LLMError("大模型响应格式不对：不是 JSON 对象", status=status, kind="malformed")
        if not 200 <= status < 300:
            raise LLMError(
                self._http_error(status, body, key),
                status=status,
                kind=_http_kind(status),
                code=_error_code(body, key),
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            # 有的服务出错时也回 HTTP 200，错误放在响应体的 error 里：代码照样带上
            raise LLMError(
                "大模型响应格式不对：没有 choices",
                status=status,
                kind="malformed",
                code=_error_code(body, key),
            )
        choice = choices[0]
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason in _FILTERED:
            raise LLMError("大模型输出被内容安全策略拦截", kind="content_filter")
        message = choice.get("message")
        text = _text_content(message.get("content")) if isinstance(message, dict) else None
        if text is None:
            raise LLMError("大模型响应格式不对：没有文本内容", status=status, kind="malformed")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        model = body.get("model")
        return LLMResponse(
            text=text,
            model=model if isinstance(model, str) and model else self.model,
            input_tokens=_non_negative_int(usage.get("prompt_tokens")),
            output_tokens=_non_negative_int(usage.get("completion_tokens")),
            logprobs=_option_probabilities(choice.get("logprobs"))
            if request.want_logprobs
            else None,
            # 照实填，被截断（length）、文本是空的也不抛异常：算不算数由调用方定
            finish_reason=reason if isinstance(reason, str) and reason else None,
        )

    @staticmethod
    def _http_error(status: int, body: dict[str, Any], key: str) -> str:
        if 300 <= status < 400:
            head = f"大模型接口返回重定向（HTTP {status}），不跟随：请检查 base_url"
        elif status == 429:
            head = "大模型接口限流（HTTP 429）"
        elif status >= 500:
            head = f"大模型服务错误（HTTP {status}）"
        else:
            head = f"大模型接口拒绝请求（HTTP {status}）"
        # 只带错误类型 / 代码，不带错误消息原文（可能回显提示词里的买家文字或密钥片段）
        err = body.get("error")
        if isinstance(err, dict):
            tags = [
                _scrub(str(err[k]), key, limit=64)
                for k in ("type", "code")
                if isinstance(err.get(k), str | int) and not isinstance(err.get(k), bool)
            ]
            if tags:
                head += "：" + " / ".join(tags)
        return head
