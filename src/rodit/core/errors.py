"""异常类型。降级逻辑（§12.2）按异常类型分支，不要用裸 Exception 传递业务错误。"""

from __future__ import annotations

from typing import Literal, get_args

ToolFailureReason = Literal["unavailable", "throttled", "timeout"]
"""平台接口失败的原因（ToolUnavailable.reason）：故障、限流、超时。指标按它分开统计（§12.4）。"""

TOOL_FAILURE_REASONS: tuple[ToolFailureReason, ...] = get_args(ToolFailureReason)


class RoditError(Exception):
    """所有业务异常的基类。"""


class LLMError(RoditError):
    """大模型调用失败：报错、限流、输出无法解析。调用方退回 FAQ / 模板（§12.2）。"""


class LLMTimeout(LLMError):
    """大模型调用超时。"""


class DecisionError(RoditError):
    """判断层不可用，或答案不完整 / 不合法。路由退回关键词规则（§12.2）。"""


class ToolError(RoditError):
    """平台工具调用失败的基类。"""


class ToolUnavailable(ToolError):
    """平台接口限流或故障。有缓存用缓存并注明时间，没有就推荐"我帮您查一下"（§12.2）。

    reason 区分故障（unavailable，默认）、限流（throttled）、超时（timeout），降级处理一样，
    只是指标分开统计；不认识的取值抛 ValueError。消息里不放令牌和买家信息。
    """

    def __init__(self, msg: str = "", *, reason: ToolFailureReason = "unavailable") -> None:
        if reason not in TOOL_FAILURE_REASONS:
            raise ValueError(f"未知的平台接口失败原因：{reason!r}")
        super().__init__(msg)
        self.reason: ToolFailureReason = reason


class KnowledgeUnavailable(RoditError):
    """知识库故障。此时不推荐（§12.2）。"""


class StorageUnavailable(RoditError):
    """数据库连不上或超时。消息里不带连接串（可能含密码）。

    降级（§12.2）：问答链路不推荐（客服照常工作）；售后审核、反馈不落库时如实报错，不假装成功。
    """


class ShopInactive(RoditError):
    """店铺不存在或已停用（SUSPENDED）：不处理这条消息（§8.4 取消订购 / 授权过期后停止处理）。"""

    def __init__(self, shop_id: str, status: object | None = None) -> None:
        self.shop_id = shop_id
        self.status = status
        state = "不存在" if status is None else f"状态为 {status}"
        super().__init__(f"店铺 {shop_id} {state}，不处理")


class ReadOnlyViolation(RoditError):
    """问答链路尝试调用写接口。问答引擎只调用只读工具（§7）。"""


class ActionNotAllowed(RoditError):
    """售后写操作不被允许：未经人确认、状态不对、平台不支持、权限不足等（§7.3.2）。"""


class ConcurrencyConflict(RoditError):
    """乐观锁版本冲突（aftersale_review.version）。"""


class TenantIsolationError(RoditError):
    """跨店铺访问（§9.2 串店）。"""


class ComplianceError(RoditError):
    """违反合规约束，例如在不满足 §6.3 条件时把买家数据发给 Jev。"""


class AuthExpired(RoditError):
    """店铺授权过期或已取消订购：停止处理（§8.4）。"""


class TokenDecryptionError(RoditError):
    """店铺令牌解密失败：密钥与写入时不一致、旧版加密格式，或数据损坏。消息里不带密钥和令牌。

    授权检查（adapters.auth.AuthGuard）把它当作"需要重新授权"：停止处理、列进到期告警。
    """
