"""大模型用量计量（方案 §12.4「每个店铺的模型成本（token 数）」、§14 成本估算）。

MeteredLLMClient 包在任何 LLMClient 外面，每次调用后写一条审计事件：
  action = core.audit.AuditAction.LLM_USAGE（"llm.usage"）
  shop_id = request.shop_id，没填时取 core.context.current_shop_id()（引擎处理消息时设置）
  detail = {purpose, model, input_tokens, output_tokens, latency_ms}，失败时另加 error（异常类型名）
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
from rodit.core.models import AuditEvent
from rodit.core.ports import AuditSink, LLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


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

    @property
    def options(self) -> Any:
        """inner 的请求参数（OpenAICompatibleClient.options，能力检查、指纹用）；没有时为 None。"""
        return getattr(self._inner, "options", None)

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
        if error is not None:
            detail["error"] = type(error).__name__
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
