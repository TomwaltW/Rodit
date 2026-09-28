"""异常类型。降级逻辑（§12.2）按异常类型分支，不要用裸 Exception 传递业务错误。"""

from __future__ import annotations


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
    """平台接口限流或故障。有缓存用缓存并注明时间，没有就推荐"我帮您查一下"（§12.2）。"""


class KnowledgeUnavailable(RoditError):
    """知识库故障。此时不推荐（§12.2）。"""


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
