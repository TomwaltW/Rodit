"""配置（方案 §6.6 阈值、§5.5 放行规则、§7.2 异常阈值、§7.3.2 审核规则、§9.3 配置分层）。

- 阈值由我们统一校准（§13.3），不开放给商家；商家能改的只有 ShopConfig（core.models）。
- 默认值与方案一致；部署时用 config/rodit.yaml 覆盖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rodit.core.enums import Intent, Platform
from rodit.core.models import ShopConfig


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionConfig(_Model):
    provider: str = "incloud_llm"  # incloud_llm / classifier / rules / jev
    vote_samples: int = Field(default=3, ge=1, le=7)  # 路由自一致性投票次数（§6.4）
    qa_vote_samples: int = Field(default=1, ge=1, le=7)  # 质检投票次数
    vote_temperature: float = 0.7
    timeout_s: float = 1.0  # 路由 ≤ 1 秒（§12.1）


class RoutingConfig(_Model):
    off_platform: float = 0.6
    need_human: float = 0.7
    emotion_escalate: float = 3.0
    multi_intent: float = 0.7
    clarify_max_gap: float = 0.2
    assist_threshold: float = 0.6  # 智能辅助有人把关，阈值可以低
    auto_threshold: dict[Intent, float] = Field(  # 全自动只对列出的类型生效
        default_factory=lambda: {
            Intent.GREETING: 0.95,
            Intent.SHIPPING_POLICY: 0.95,
            Intent.LOGISTICS_TRACK: 0.95,
            Intent.INVOICE_GIFT: 0.95,
            Intent.RETURN_POLICY: 0.97,
            Intent.PRODUCT_SPEC: 0.97,
        }
    )
    high_risk: list[Intent] = Field(
        default_factory=lambda: [
            Intent.PRICE_PROMO,
            Intent.COMPARE_AUTHENTIC,
            Intent.MODIFY_ORDER,
            Intent.QUALITY_ISSUE,
            Intent.NOT_RECEIVED,
            Intent.COMPLAINT,
        ]
    )


class QAGateConfig(_Model):
    grounded_min: float = 0.8
    promise_risk_max: float = 0.3
    platform_violation_max: float = 0.2
    answers_question_min: float = 0.5
    max_regenerations: int = 1  # 硬规则命中 → 带原因重新生成一次（§5.5）
    absolute_terms: list[str] = Field(  # 绝对化用语（§5.5），商家可在 banned_words 里补充
        default_factory=lambda: [
            "最",
            "第一",
            "绝对",
            "100%",
            "百分百",
            "顶级",
            "极致",
            "唯一",
            "万能",
            "永久",
            "保证",
            "一定",
            "肯定",
        ]
    )


class GenerationConfig(_Model):
    history_rounds: int = 5  # 最近 5 轮对话（§5.3）
    faq_direct_min_score: float = 0.85  # FAQ 相似度达到这个值才直接用标准回答
    faq_top_k: int = 5  # 语义检索取前 3–5 条（§5.2.4）
    max_evidence_chars: int = 1500  # 控制证据长度（§14 降本）
    temperature: float = 0.3
    max_tokens: int = 400
    auto_reply_prefix: str = "【智能客服】"  # 全自动回复的 AI 生成内容标识（§12.3）


class ConversationConfig(_Model):
    merge_window_s: float = 2.0  # 1–2 秒内连发的消息合并处理（§5.3）
    unresolved_turns_alert: int = 2  # 连续两轮没听懂 → 提示客服接管（§5.7）


class LogisticsConfig(_Model):
    stall_hours: int = 48  # 超过 48 小时没有新轨迹 → 停滞（§7.2）
    cache_ttl_s: int = 300  # 物流、订单短时缓存


class AfterSaleConfig(_Model):
    seven_day_days: int = 7  # R3
    verify_hours: int = 24  # R4 签收核查等待
    repeat_window_days: int = 30  # R7
    repeat_threshold: int = 3  # R7


class EngineConfig(_Model):
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    qa_gate: QAGateConfig = Field(default_factory=QAGateConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    logistics: LogisticsConfig = Field(default_factory=LogisticsConfig)
    aftersale: AfterSaleConfig = Field(default_factory=AfterSaleConfig)
    # 配置分层（§9.3）：平台默认 → 类目默认 → 店铺配置，下层覆盖上层
    shop_defaults_by_platform: dict[Platform, dict[str, Any]] = Field(default_factory=dict)
    shop_defaults_by_category: dict[str, dict[str, Any]] = Field(default_factory=dict)


def load_config(path: str | Path | None = None) -> EngineConfig:
    """读取 YAML 配置；path 为空时返回默认值。YAML 顶层键与 EngineConfig 字段一一对应。"""
    if path is None:
        return EngineConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return EngineConfig.model_validate(data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_shop_config(
    config: EngineConfig,
    *,
    platform: Platform,
    category: str | None,
    shop_override: dict[str, Any] | None = None,
) -> ShopConfig:
    """平台默认 → 类目默认 → 店铺配置，下层覆盖上层（§9.3）。"""
    merged: dict[str, Any] = {}
    merged = _deep_merge(merged, config.shop_defaults_by_platform.get(platform, {}))
    if category:
        merged = _deep_merge(merged, config.shop_defaults_by_category.get(category, {}))
    merged = _deep_merge(merged, shop_override or {})
    return ShopConfig.model_validate(merged)
