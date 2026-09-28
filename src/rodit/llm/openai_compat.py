"""OpenAI 兼容 /chat/completions 客户端（D5 境内已备案模型；方案 §4.4、§6.4、§12.1、§12.2）。

很多境内已备案模型都提供 OpenAI 兼容接口；具体用哪个模型是 D5 待决策项，
base_url / model 由部署配置决定，代码里不写死。实现 core.ports.LLMClient：

- 密钥：每次调用时从环境变量读（默认 RODIT_LLM_API_KEY），不保存在对象上；
  缺失时抛 LLMError。密钥不出现在 repr、异常消息、日志里（传输层异常的文字也会先抹掉密钥）。
- 超时：request.timeout_s 优先，否则用构造时的 timeout_s（路由 ≤ 1 秒由调用方传入，§12.1）。
- 错误（§12.2 调用方据此退回 FAQ / 模板）：超时 → LLMTimeout；HTTP 429 / 5xx / 其他非 2xx、
  网络错误、响应格式不对 → LLMError。不重试：延迟预算紧，降级比重试更可控。
- json_mode → response_format {"type": "json_object"}；usage → input_tokens / output_tokens。
- want_logprobs → 请求 logprobs + top_logprobs，读出"选项 → 概率"放进 LLMResponse.logprobs（§6.4）：
  取第一个有内容的输出位置（跳过只有空白、引号、括号、冒号、逗号的 token），
  把该位置的候选 token 去掉这些字符后作为选项，概率 = exp(logprob)，同名选项相加（上限 1）。
  适用于"先输出选项标签"的提示词；JSON 模式下第一个有内容的 token 往往是键名，不适合用。
  接口不返回 logprobs 时为 None。
- request.purpose / shop_id 只用于本地计量（MeteredLLMClient），不发给模型服务。

默认传输层用标准库 urllib（不引入新依赖）；测试注入假的 transport，不访问网络。
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from rodit.core.errors import LLMError, LLMTimeout
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


def urllib_transport(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float
) -> tuple[int, dict[str, Any]]:
    """默认传输层（标准库 urllib，走环境里配置的 HTTP(S) 代理）。"""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
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
    - extra_body：模型服务特有的参数（如关闭思考模式），会被标准字段覆盖。
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
        密钥仍在每次调用时从 {prefix}API_KEY 读。"""
        base_url = os.environ.get(f"{prefix}BASE_URL", "")
        model = os.environ.get(f"{prefix}MODEL", "")
        if not base_url or not model:
            raise RuntimeError(
                f"环境变量 {prefix}BASE_URL 和 {prefix}MODEL 必须设置（D5 所选模型）"
            )
        timeout = os.environ.get(f"{prefix}TIMEOUT_S")
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
            raise LLMError(f"环境变量 {self.api_key_env} 未设置：没有大模型 API key")
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
        except LLMError as e:
            raise type(e)(_scrub(str(e), key)) from None
        except TimeoutError:
            raise LLMTimeout(f"大模型调用超时（{timeout:g} 秒）") from None
        except Exception as e:  # 网络错误、响应不是 JSON 等：统一成 LLMError，抹掉可能带出的密钥
            raise LLMError(
                f"大模型接口调用失败：{type(e).__name__}: {_scrub(str(e), key)}"
            ) from None
        return self._parse(status, body, request, key)

    # ---------------------------------------------------------------- 响应

    def _parse(self, status: int, body: Any, request: LLMRequest, key: str) -> LLMResponse:
        if not isinstance(body, dict):
            raise LLMError("大模型响应格式不对：不是 JSON 对象")
        if not 200 <= status < 300:
            raise LLMError(self._http_error(status, body, key))
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMError("大模型响应格式不对：没有 choices")
        choice = choices[0]
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason in _FILTERED:
            raise LLMError("大模型输出被内容安全策略拦截")
        message = choice.get("message")
        text = _text_content(message.get("content")) if isinstance(message, dict) else None
        if text is None:
            raise LLMError("大模型响应格式不对：没有文本内容")
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
        )

    @staticmethod
    def _http_error(status: int, body: dict[str, Any], key: str) -> str:
        if status == 429:
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
