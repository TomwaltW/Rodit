"""Jev 的 HTTP 传输层，只给离线对比评测用（方案 §6.3 "离线评测：用我们自己编写的合成问题"、§6.5）。

JevProvider（rodit.decision.jev）里没有 HTTP 代码，transport 由调用方注入（dict → dict）；这里是
`python -m rodit.eval.compare --jev env` 注入的那个实现：

- POST {RODIT_JEV_BASE_URL}/api/v1/systemone，请求体就是 JevProvider 给的 Jev 格式 JSON
  （§6.5：model、state、questions），响应体原样交回 JevProvider 校验；
- 密钥：每次调用时从环境变量 RODIT_JEV_API_KEY 读（Authorization: Bearer），不保存在对象上，
  不出现在 repr、异常消息、日志里——本模块不写日志，异常消息只有状态码和异常类型名，
  不带响应体原文（可能回显请求内容）；
- 超时 RODIT_JEV_TIMEOUT_S（默认 10 秒）；不跟随重定向（urllib 跟随 3xx 时会把 Authorization
  原样带到新地址），3xx 按失败处理；地址只接受 https（密钥不走明文），不能带用户名密码、
  查询参数；
- 失败一律抛 DecisionError 的子类（JevTransportError，超时是 JevTimeout）：HTTP 非 2xx、
  网络错误、超时、响应不是 JSON 对象。JevProvider 再包一层 DecisionError（按"判断层不可用"
  处理，§12.2），原异常留在 __cause__ 里，对比评测据此区分超时和其他失败；
- 用量：响应里带 usage（input_tokens / output_tokens，或 prompt_tokens / completion_tokens）
  时取出来累计（usage_totals），再把 usage 从响应里拿掉——DecisionResponse 不收这个字段；
  其余字段原样交回，格式变了照样由 JevProvider 报错。

只发合成数据：这个传输层只由对比评测在合成评测集上构造；JevProvider 的 offline_eval 护栏
（state.synthetic 必须为 True、state 里不能有个人信息）在调用这里之前检查（§6.2、§6.3）。
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

from rodit.core.errors import DecisionError

ENV_PREFIX = "RODIT_JEV_"
BASE_URL_ENV = "RODIT_JEV_BASE_URL"
API_KEY_ENV = "RODIT_JEV_API_KEY"
TIMEOUT_ENV = "RODIT_JEV_TIMEOUT_S"
MODEL_ENV = "RODIT_JEV_MODEL"  # 可选：默认用 JevProvider 的模型名（jev-1.13.0）
JEV_PATH = "/api/v1/systemone"
DEFAULT_TIMEOUT_S = 10.0
KEY_HINT = "密钥只放本地 .env 或环境的密钥设置，不要提交进仓库"

Opener = Callable[[urllib.request.Request, float], Any]
"""opener(request, timeout_s) → 带 status、read() 的响应（上下文管理器），同 urllib 的 urlopen。"""


class JevTransportError(DecisionError):
    """Jev 调用失败（HTTP 非 2xx、网络错误、响应不是 JSON 对象）。status 是 HTTP 状态码（有的话）。

    消息里只有状态码、异常类型名，没有密钥和响应体原文。"""

    def __init__(self, msg: str, *, status: int | None = None) -> None:
        super().__init__(msg)
        self.status = status


class JevTimeout(JevTransportError):
    """Jev 调用超时（连接或读响应）。"""


class JevConfigError(ValueError):
    """--jev env 的环境变量没设或不合法（消息里只有变量名，没有取值）。"""


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """不跟随任何重定向：3xx 作为 HTTPError 交回调用方，Authorization 不会被带到别处。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _urlopen(request: urllib.request.Request, timeout: float) -> Any:
    # 每次新建 opener：代理设置按调用时的环境变量读取（和 urllib.request.urlopen 一样走代理）
    return urllib.request.build_opener(_RefuseRedirects).open(request, timeout=timeout)


def _check_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise JevConfigError(f"{BASE_URL_ENV} 必须以 https:// 开头（密钥不走明文）")
    if not parts.hostname:
        raise JevConfigError(f"{BASE_URL_ENV} 没有主机名")
    if parts.username is not None or parts.password is not None:
        raise JevConfigError(f"{BASE_URL_ENV} 里不能带用户名密码（密钥用 {API_KEY_ENV}）")
    if parts.query or parts.fragment:
        raise JevConfigError(f"{BASE_URL_ENV} 不能带查询参数或 #片段")
    if parts.path.rstrip("/").endswith(JEV_PATH):
        raise JevConfigError(f"{BASE_URL_ENV} 只写到主机（和路径前缀），不含 {JEV_PATH}")
    try:
        parts.port  # noqa: B018 — 端口不是数字时这里抛 ValueError
    except ValueError:
        raise JevConfigError(f"{BASE_URL_ENV} 的端口不是数字") from None
    return url


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return max(0, int(value))


def _http_message(status: int) -> str:
    if 300 <= status < 400:
        return f"Jev 接口返回重定向（HTTP {status}），不跟随：请检查 {BASE_URL_ENV}"
    if status in (401, 403):
        return f"Jev 拒绝了请求的密钥（HTTP {status}）：请检查 {API_KEY_ENV}"
    if status == 429:
        return "Jev 接口限流（HTTP 429）"
    if status >= 500:
        return f"Jev 服务错误（HTTP {status}）"
    return f"Jev 接口拒绝请求（HTTP {status}）"


class JevHttpTransport:
    """实现 rodit.decision.jev.JevTransport（dict → dict），见模块说明。线程安全。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str = API_KEY_ENV,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        opener: Opener | None = None,
    ) -> None:
        if not api_key_env:
            raise JevConfigError("api_key_env 不能为空")
        if not 0 < timeout_s < float("inf"):  # 也挡住 NaN
            raise JevConfigError(f"{TIMEOUT_ENV} 必须是正数（秒）")
        self.base_url = _check_base_url(base_url)
        self.endpoint = f"{self.base_url}{JEV_PATH}"
        self.api_key_env = api_key_env
        self.timeout_s = float(timeout_s)
        self._opener = opener
        self._lock = threading.Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._reports = 0  # 响应里带了用量的次数

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **kwargs: Any) -> JevHttpTransport:
        """按环境变量构造（见模块说明）。地址或密钥没设、超时不是正数时抛 JevConfigError，
        消息里只有变量名。密钥这里只检查"有没有设置"，调用时再读。"""
        env = os.environ if env is None else env
        base_url = env.get(BASE_URL_ENV, "").strip()
        missing = [name for name in (BASE_URL_ENV, API_KEY_ENV) if not env.get(name, "").strip()]
        if missing:
            raise JevConfigError(f"--jev env 需要环境变量 {' 和 '.join(missing)}（{KEY_HINT}）")
        raw_timeout = env.get(TIMEOUT_ENV, "").strip()
        if raw_timeout:
            try:
                kwargs.setdefault("timeout_s", float(raw_timeout))
            except ValueError:
                raise JevConfigError(f"{TIMEOUT_ENV} 必须是数字（秒）") from None
        return cls(base_url=base_url, api_key_env=API_KEY_ENV, **kwargs)

    def __repr__(self) -> str:
        return (
            f"JevHttpTransport(base_url={self.base_url!r}, api_key_env={self.api_key_env!r}, "
            f"timeout_s={self.timeout_s:g})"
        )

    # ---------------------------------------------------------------- 用量

    def usage_totals(self) -> tuple[int, int, int]:
        """累计 (输入 token, 输出 token, 带用量的响应数)。Jev 不报用量时都是 0。"""
        with self._lock:
            return self._input_tokens, self._output_tokens, self._reports

    def _record_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        inp = _non_negative_int(usage.get("input_tokens", usage.get("prompt_tokens")))
        out = _non_negative_int(usage.get("output_tokens", usage.get("completion_tokens")))
        if inp is None and out is None:
            return
        with self._lock:
            self._input_tokens += inp or 0
            self._output_tokens += out or 0
            self._reports += 1

    # ---------------------------------------------------------------- 调用

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise JevTransportError(f"环境变量 {self.api_key_env} 未设置：没有 Jev API key")
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        opener = self._opener or _urlopen
        timeout = f"{self.timeout_s:g}"
        try:
            with opener(request, self.timeout_s) as resp:
                status, raw = int(resp.status), resp.read()
        except urllib.error.HTTPError as e:  # 非 2xx（含不跟随的 3xx）：不读、不带响应体
            e.close()
            raise JevTransportError(_http_message(e.code), status=e.code) from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise JevTimeout(f"Jev 调用超时（{timeout} 秒）") from None
            raise JevTransportError(f"Jev 网络错误：{type(e.reason).__name__}") from None
        except TimeoutError:  # 连接后读响应超时（socket.timeout 是它的别名）
            raise JevTimeout(f"Jev 调用超时（{timeout} 秒）") from None
        except (OSError, http.client.HTTPException) as e:  # 连接被断开、响应不完整等
            raise JevTransportError(f"Jev 网络错误：{type(e).__name__}") from None
        if not 200 <= status < 300:
            raise JevTransportError(_http_message(status), status=status)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise JevTransportError("Jev 响应不是 JSON", status=status) from None
        if not isinstance(body, dict):
            raise JevTransportError("Jev 响应不是 JSON 对象", status=status)
        self._record_usage(body.pop("usage", None))
        return body


def transport_from_env(env: Mapping[str, str] | None = None) -> tuple[JevHttpTransport, str | None]:
    """--jev env：(传输层, 模型名)。模型名取 RODIT_JEV_MODEL，没设时为 None
    （用 JevProvider 的默认模型名）。"""
    env = os.environ if env is None else env
    transport = JevHttpTransport.from_env(env)
    model = env.get(MODEL_ENV, "").strip() or None
    return transport, model


__all__ = [
    "API_KEY_ENV",
    "BASE_URL_ENV",
    "DEFAULT_TIMEOUT_S",
    "JEV_PATH",
    "MODEL_ENV",
    "TIMEOUT_ENV",
    "JevConfigError",
    "JevHttpTransport",
    "JevTimeout",
    "JevTransportError",
    "transport_from_env",
]
