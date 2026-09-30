"""部署入口的设置（环境变量 → ServerSettings）。

这里只放"怎么跑"的设置，不放任何密钥：数据库连接串、令牌加密密钥、buyer_ref 的盐、大模型密钥
仍由原来的代码在用到时直接从环境变量读（rodit.app.production、rodit.llm），ServerSettings 的
字段和 repr 里都不会出现它们的值。

| 变量 | 默认 | 说明 |
|---|---|---|
| RODIT_ENV | production | production / local（本地试用：允许合成数据和离线大模型替身） |
| RODIT_PLATFORM | （必填） | taobao / douyin / pdd：本单元所在的平台云（§9.1） |
| RODIT_PLATFORM_BINDINGS | | 平台云代码的绑定工厂 ``包.模块:函数``（见 rodit.ops.bindings） |
| RODIT_MODE_C_ONLY | false | 只部署模式 C（Phase 0）：不需要平台绑定 |
| RODIT_CONFIG | | 引擎配置 YAML（镜像里是 /app/config/rodit.yaml）；为空用默认值 |
| RODIT_TOKENS_FILE | /run/secrets/rodit_tokens.json | 访问令牌哈希文件（rodit.ops.tokens） |
| RODIT_TOKENS_RELOAD_S | 30 | 令牌文件变化检查间隔（秒） |
| RODIT_METRICS_PUBLIC | false | /metrics 不要令牌（只在网络策略已限制访问时用） |
| RODIT_HOST / RODIT_PORT | 127.0.0.1 / 8000 | 监听地址（镜像里 HOST 是 0.0.0.0） |
| RODIT_LOG_LEVEL / RODIT_LOG_FORMAT | INFO / json | 日志级别、格式（json / text） |
| RODIT_MIGRATE_ON_START | false | serve 启动前先迁移（需要属主权限，一般用单独的 migrate 命令） |
| RODIT_AUTH_EXPIRY_WARN_DAYS | 7 | 授权在这么多天内到期就告警（§12.4） |
| RODIT_AUTH_EXPIRY_CHECK_S | 300 | 授权检查的最短间隔（秒） |
| RODIT_READINESS_CACHE_S | 2 | /readyz 结果缓存（秒） |
| RODIT_CLASSIFIER_PATH | | 路由分类器模型文件（provider 是 classifier 时必填） |

离线任务 recalibrate / train-classifier 不用 ServerSettings（不需要 RODIT_PLATFORM），标注从
RODIT_TRAINER_DSN 读（只属于 rodit_trainer 的登录账号，见 rodit.ops.serve）。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rodit.core.enums import Platform

DEFAULT_TOKENS_FILE = "/run/secrets/rodit_tokens.json"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})


class SettingsError(RuntimeError):
    """设置不合法。消息里只有变量名和原因，不回显值以外的环境内容。"""


def _bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise SettingsError(f"环境变量 {name} 应为 1/0、true/false，现在是 {raw!r}")


def _number(env: Mapping[str, str], name: str, default: float, *, integer: bool = False) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        raise SettingsError(f"环境变量 {name} 必须是数字，现在是 {raw!r}") from None
    if value < 0:
        raise SettingsError(f"环境变量 {name} 不能为负，现在是 {raw!r}")
    return value


class ServerSettings(BaseModel):
    """见模块说明。用 from_env 构造；测试里可以直接传字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    env: Literal["production", "local"] = "production"
    platform: Platform
    bindings: str = ""
    mode_c_only: bool = False
    config_path: str = ""
    tokens_file: str = DEFAULT_TOKENS_FILE
    tokens_reload_s: float = Field(default=30.0, ge=0)
    metrics_public: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    migrate_on_start: bool = False
    auth_expiry_warn_days: float = Field(default=7.0, gt=0)
    auth_expiry_check_s: float = Field(default=300.0, ge=0)
    readiness_cache_s: float = Field(default=2.0, ge=0)
    classifier_path: str = ""  # 路由分类器模型文件；只在 decision.provider: classifier 时加载

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def deploy_mode(self) -> str:
        """rodit_build_info 的 mode 标签：mode_c_only / platform / local。"""
        if self.env == "local":
            return "local"
        return "mode_c_only" if self.mode_c_only else "platform"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServerSettings:
        env = os.environ if environ is None else environ
        platform = env.get("RODIT_PLATFORM", "").strip().lower()
        if not platform:
            raise SettingsError("环境变量 RODIT_PLATFORM 必须设置（taobao / douyin / pdd，§9.1）")
        try:
            return cls(
                env=env.get("RODIT_ENV", "production").strip().lower() or "production",
                platform=platform,
                bindings=env.get("RODIT_PLATFORM_BINDINGS", "").strip(),
                mode_c_only=_bool(env, "RODIT_MODE_C_ONLY"),
                config_path=env.get("RODIT_CONFIG", "").strip(),
                tokens_file=env.get("RODIT_TOKENS_FILE", "").strip() or DEFAULT_TOKENS_FILE,
                tokens_reload_s=_number(env, "RODIT_TOKENS_RELOAD_S", 30.0),
                metrics_public=_bool(env, "RODIT_METRICS_PUBLIC"),
                host=env.get("RODIT_HOST", "").strip() or "127.0.0.1",
                port=int(_number(env, "RODIT_PORT", 8000, integer=True)),
                log_level=env.get("RODIT_LOG_LEVEL", "INFO").strip().upper() or "INFO",
                log_format=env.get("RODIT_LOG_FORMAT", "json").strip().lower() or "json",
                migrate_on_start=_bool(env, "RODIT_MIGRATE_ON_START"),
                auth_expiry_warn_days=_number(env, "RODIT_AUTH_EXPIRY_WARN_DAYS", 7.0),
                auth_expiry_check_s=_number(env, "RODIT_AUTH_EXPIRY_CHECK_S", 300.0),
                readiness_cache_s=_number(env, "RODIT_READINESS_CACHE_S", 2.0),
                classifier_path=env.get("RODIT_CLASSIFIER_PATH", "").strip(),
            )
        except ValidationError as e:
            fields = sorted({".".join(str(p) for p in err["loc"]) for err in e.errors()})
            raise SettingsError(f"部署设置不合法：{', '.join(fields)}") from None


__all__ = ["DEFAULT_TOKENS_FILE", "ServerSettings", "SettingsError"]
