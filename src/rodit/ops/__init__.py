"""部署与运维：一个平台单元的容器化运行（方案 §9.1 单元化部署、§12.1–§12.4）。

- serve：部署入口（python -m rodit.ops.serve / python -m rodit.ops）：serve、migrate、
  issue-token、local-init；
- bindings：平台云代码注入平台只读工具、接入适配器、售后写接口的约定（PlatformBindings）；
- tokens：生产用的调用方鉴权，令牌哈希来自挂载的密钥文件（TokenFileAuthenticator）；
- health / router：/healthz、/readyz、/metrics；
- metrics / instrument：Prometheus 文本格式的指标，和不改业务代码的采集（审计、平台工具、HTTP）；
- logconfig：结构化 JSON 日志，个人信息和密钥过滤。

负责：部署打包（成员 D）。和 rodit.app、rodit.api 一样可以依赖所有子包。
"""

from rodit.ops.bindings import (
    BindingContext,
    BindingError,
    PlatformBindings,
    demo_bindings,
    mode_c_bindings,
)
from rodit.ops.health import CheckResult, Readiness, ReadinessChecker
from rodit.ops.instrument import (
    AuthExpiryMonitor,
    HttpMetricsMiddleware,
    MeteredToolGateway,
    ObservedAuditLog,
    RoditMetrics,
)
from rodit.ops.metrics import MetricsRegistry
from rodit.ops.router import install_ops
from rodit.ops.serve import Server, StartupError, build_server, main
from rodit.ops.settings import ServerSettings, SettingsError
from rodit.ops.tokens import TokenFileAuthenticator, TokenFileError

__all__ = [
    "AuthExpiryMonitor",
    "BindingContext",
    "BindingError",
    "CheckResult",
    "HttpMetricsMiddleware",
    "MeteredToolGateway",
    "MetricsRegistry",
    "ObservedAuditLog",
    "PlatformBindings",
    "Readiness",
    "ReadinessChecker",
    "RoditMetrics",
    "Server",
    "ServerSettings",
    "SettingsError",
    "StartupError",
    "TokenFileAuthenticator",
    "TokenFileError",
    "build_server",
    "demo_bindings",
    "install_ops",
    "main",
    "mode_c_bindings",
]
