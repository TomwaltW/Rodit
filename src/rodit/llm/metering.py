"""大模型用量计量（方案 §12.4「每个店铺的模型成本（token 数）」、§14 成本估算）。

MeteredLLMClient 包在任何 LLMClient 外面，每次调用后写一条审计事件：
  action = core.audit.AuditAction.LLM_USAGE（"llm.usage"）
  shop_id = request.shop_id，没填时取 core.context.current_shop_id()（引擎处理消息时设置）
  detail = {purpose, model, input_tokens, output_tokens, latency_ms}，失败时另加 error（异常类型名）
可选的键，有才写（正常的成功调用、没带原因的失败，detail 和上面一样）：
  失败：kind、status、code——LLMError 带的失败原因（core.errors），哪项知道写哪项；
  成功：finish_reason——不是 stop 才写（length 是被 max_tokens 截断）；empty: true——文本是空的。
detail 里没有提示词和输出原文（§5.3）。审计写入失败不影响调用结果（只记一条警告日志），
计量不能把问答链路拖垮；原调用的异常照常抛出，调用方按 §12.2 降级。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from rodit.core.audit import AuditAction
from rodit.core.clock import Clock
from rodit.core.context import current_shop_id
from rodit.core.errors import LLMError
from rodit.core.models import AuditEvent
from rodit.core.ports import AuditSink, LLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


def _cause(error: Exception) -> dict[str, Any]:
    """失败原因（LLMError 的 kind / status / code）里知道的那几项。类型不对的不写：
    status 要是整数（评测按它分类），kind、code 要是非空的字符串（code 是整数时转成字符串）。"""
    if not isinstance(error, LLMError):
        return {}
    # getattr：子类没调基类的构造函数时也不能让计量把原来的异常换掉
    kind, status, code = (getattr(error, name, None) for name in ("kind", "status", "code"))
    cause: dict[str, Any] = {}
    if isinstance(kind, str) and kind:
        cause["kind"] = kind
    if isinstance(status, int) and not isinstance(status, bool):
        cause["status"] = status
    if isinstance(code, int) and not isinstance(code, bool):
        code = str(code)
    if isinstance(code, str) and code.strip():
        cause["code"] = code
    return cause


class MeteredLLMClient:
    """实现 core.ports.LLMClient：转发给 inner，并按店铺记录用量。

    - timer：计算耗时用的单调时钟（秒），测试里注入假的；业务时间（created_at）用 clock。
    - actor：审计事件的 actor，默认 system。
    """

    def __init__(
        self,
        inner: LLMClient,
        audit: AuditSink,
        clock: Clock,
        *,
        timer: Callable[[], float] = time.perf_counter,
        actor: str = "system",
    ) -> None:
        self._inner = inner
        self._audit = audit
        self._clock = clock
        self._timer = timer
        self._actor = actor

    @property
    def model(self) -> str:
        return str(getattr(self._inner, "model", "") or "")

    def __repr__(self) -> str:
        return f"MeteredLLMClient(inner={self._inner!r})"

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = self._timer()
        try:
            response = self._inner.complete(request)
        except Exception as e:
            self._record(request, started, response=None, error=e)
            raise
        self._record(request, started, response=response, error=None)
        return response

    def _record(
        self,
        request: LLMRequest,
        started: float,
        *,
        response: LLMResponse | None,
        error: Exception | None,
    ) -> None:
        latency_ms = max(0, round((self._timer() - started) * 1000))
        detail: dict[str, Any] = {
            "purpose": request.purpose,
            "model": (response.model if response is not None else "") or self.model,
            "input_tokens": response.input_tokens if response is not None else 0,
            "output_tokens": response.output_tokens if response is not None else 0,
            "latency_ms": latency_ms,
        }
        if response is not None:
            if response.finish_reason and response.finish_reason != "stop":
                detail["finish_reason"] = response.finish_reason
            if not response.text.strip():
                detail["empty"] = True
        if error is not None:
            detail["error"] = type(error).__name__
            detail.update(_cause(error))
        try:
            self._audit.record(
                AuditEvent(
                    shop_id=request.shop_id or current_shop_id(),
                    actor=self._actor,
                    action=AuditAction.LLM_USAGE.value,
                    detail=detail,
                    created_at=self._clock.now(),
                )
            )
        except Exception as e:  # 计量失败不影响问答；日志里只有用途和异常类型
            logger.warning(
                "大模型用量记录失败：purpose=%s error=%s", request.purpose, type(e).__name__
            )
