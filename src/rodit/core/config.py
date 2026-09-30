"""配置（方案 §6.6 阈值、§5.5 放行规则、§7.2 异常阈值、§7.3.2 审核规则、§9.3 配置分层）。

- 阈值由我们统一校准（§13.3），不开放给商家；商家能改的只有 ShopConfig（core.models）。
- 默认值与方案一致；部署时用 config/rodit.yaml 覆盖。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rodit.core.enums import Intent, Platform, ProviderName
from rodit.core.lexicon import ABSOLUTE_TERM_EXCEPTIONS, ABSOLUTE_TERMS
from rodit.core.models import Shop, ShopConfig


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionConfig(_Model):
    provider: ProviderName = ProviderName.INCLOUD_LLM
    vote_samples: int = Field(default=3, ge=1, le=7)  # 路由自一致性投票次数（§6.4）
    qa_vote_samples: int = Field(default=1, ge=1, le=7)  # 质检投票次数
    vote_temperature: float = 0.7
    timeout_s: float = 1.0  # 路由 ≤ 1 秒（§12.1）
    # §6.3 四个条件全部满足、经法务确认后才能改成 True；否则 provider 不能是 jev（§6.2）
    jev_production_approved: bool = False

    @model_validator(mode="after")
    def _jev_gate(self) -> DecisionConfig:
        if self.provider == ProviderName.JEV and not self.jev_production_approved:
            raise ValueError("Jev 暂不进入生产链路（方案 §6.2、§6.3）：provider 不能配置为 jev")
        return self


class RoutingConfig(_Model):
    off_platform: float = 0.6
    need_human: float = 0.7
    emotion_escalate: float = 3.0
    multi_intent: float = 0.7
    clarify_max_gap: float = 0.2
    multi_intent_min_prob: float = 0.2  # 规则 5 分条推荐：概率达到这个值的问题类型才单独成条
    max_parts: int = Field(default=3, ge=2, le=5)
    assist_threshold: float = 0.6  # 智能辅助有人把关，阈值可以低
    # 按问题类型校准过的智能辅助阈值（§13.3 按类型扫描、§13.5 每月重算）；没列出的类型用
    # assist_threshold。由我们统一校准，不开放给商家（§9.3）
    assist_threshold_by_intent: dict[Intent, float] = Field(default_factory=dict)
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

    @field_validator("assist_threshold_by_intent", "auto_threshold")
    @classmethod
    def _unit_interval(cls, v: dict[Intent, float]) -> dict[Intent, float]:
        bad = sorted(str(k) for k, t in v.items() if not 0 < t <= 1)
        if bad:
            raise ValueError(f"阈值必须在 (0, 1] 之间：{bad}")
        return v

    def assist_threshold_for(self, intent: Intent) -> float:
        """该问题类型的智能辅助阈值（校准过的优先，否则用统一值）。"""
        return self.assist_threshold_by_intent.get(intent, self.assist_threshold)


class QAGateConfig(_Model):
    grounded_min: float = 0.8
    promise_risk_max: float = 0.3
    platform_violation_max: float = 0.2
    answers_question_min: float = 0.5
    max_regenerations: int = 1  # 质检不通过 → 带原因重新生成一次（§5.5、§11.5）
    # 绝对化用语（§5.5），匹配规则见 core.lexicon.find_absolute_terms；商家补充的词在 banned_words
    absolute_terms: list[str] = Field(default_factory=lambda: list(ABSOLUTE_TERMS))
    absolute_term_exceptions: list[str] = Field(
        default_factory=lambda: list(ABSOLUTE_TERM_EXCEPTIONS)
    )


class GenerationConfig(_Model):
    history_rounds: int = 5  # 最近 5 轮对话（§5.3）
    faq_direct_min_score: float = 0.85  # FAQ 相似度达到这个值才直接用标准回答
    faq_top_k: int = 5  # 语义检索取前 3–5 条（§5.2.4）
    max_evidence_chars: int = 1500  # 控制证据长度（§14 降本）
    temperature: float = 0.3
    max_tokens: int = 400
    timeout_s: float = 2.5  # 生成 ≤ 2.5 秒（§12.1）
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


class LLMConfig(_Model):
    """境内已备案大模型（D5）的调用配置。密钥不在这里：只写环境变量名，调用时再读。

    public_name / filing_no：对外公示的模型名称和备案号（§12.3 生成式人工智能服务登记 / 备案），
    网页工具和 /v1/me 展示；require_disclosure 为 True 时生产部署缺这两项拒绝启动。
    """

    base_url: str | None = None  # OpenAI 兼容接口地址；为空时从环境变量 RODIT_LLM_BASE_URL 读
    model: str | None = None  # 为空时从 RODIT_LLM_MODEL 读
    api_key_env: str = "RODIT_LLM_API_KEY"
    timeout_s: float = 10.0  # 单次调用的上限；各环节自己的预算见 decision / generation
    public_name: str | None = None  # 公示的模型名称【待定：D5 选定后填】
    filing_no: str | None = None  # 备案号【待定：D5 选定后填】
    require_disclosure: bool = False


class PrivacyConfig(_Model):
    # buyer_ref 哈希的盐从这个环境变量读（§8.3）；生产部署必须设置，仓库里没有真实值
    buyer_salt_env: str = "RODIT_BUYER_SALT"


class LatencyConfig(_Model):
    """每条消息的时间预算（§12.1 端到端 ≤ 5 秒）。引擎用 core.budget.deadline_scope 执行
    （两个值都交给它）；生成、质检用 core.budget.llm_allowed() 判断还能不能调大模型。"""

    # 从引擎开始处理到出结果的上限：留出接入层、展示的余量（5 秒 − 0.5 秒）
    message_budget_s: float = Field(default=4.5, gt=0, le=30)
    # 剩余时间少于这个值就不再调大模型（一次生成 / 质检来不及），退回 FAQ / 模板
    min_llm_remaining_s: float = Field(default=0.8, ge=0, le=30)

    @model_validator(mode="after")
    def _llm_fits_budget(self) -> LatencyConfig:
        if self.min_llm_remaining_s >= self.message_budget_s:
            raise ValueError("min_llm_remaining_s 必须小于 message_budget_s（否则永远不调大模型）")
        return self


class TenancyConfig(_Model):
    """店铺生命周期（§8.4 开通、续期、取消订购 / 授权过期后按约定删除数据）。"""

    refresh_ahead_hours: float = Field(default=24, gt=0)  # 到期前多久续期
    # 授权过期多少天后删除该店铺的数据【待定：委托处理协议（§8.4 按约定删除），法务确认后改】
    purge_after_expiry_days: int = Field(default=30, ge=0)
    # 开通时允许申请的权限阶段（adapters.auth.scopes_for_phase）：1 = MVP 只读，2 = 加上写售后
    onboarding_phase: Literal[1, 2] = 1


class EngineConfig(_Model):
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    qa_gate: QAGateConfig = Field(default_factory=QAGateConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    logistics: LogisticsConfig = Field(default_factory=LogisticsConfig)
    aftersale: AfterSaleConfig = Field(default_factory=AfterSaleConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    tenancy: TenancyConfig = Field(default_factory=TenancyConfig)
    # 配置分层（§9.3）：平台默认 → 类目默认 → 店铺配置，下层覆盖上层
    shop_defaults_by_platform: dict[Platform, dict[str, Any]] = Field(default_factory=dict)
    shop_defaults_by_category: dict[str, dict[str, Any]] = Field(default_factory=dict)


def load_config(path: str | Path | None = None) -> EngineConfig:
    """读取 YAML 配置；path 为空时返回默认值。YAML 顶层键与 EngineConfig 字段一一对应。"""
    if path is None:
        return EngineConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return EngineConfig.model_validate(data)


def config_fingerprint(config: EngineConfig) -> str:
    """生效配置的指纹：规范化 JSON（键排序、紧凑、UTF-8）的 SHA-256 十六进制串。

    同样的配置（不管来自默认值还是 YAML、键的顺序如何）指纹相同；任何一个取值变了指纹就变。
    §13.5 上线前回归：评测结论记下指纹，线上配置的指纹和它一致才算"这套配置跑过回归"。
    配置里没有密钥（只有环境变量名），指纹可以写进审计和报告。
    """
    canonical = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


def effective_shop_config(config: EngineConfig, shop: Shop) -> ShopConfig:
    """店铺的生效配置：平台默认 → 类目默认 → shop.config_overrides（§9.3）。

    会话理解（engine）构建 ConversationContext.config 时调用；售后审核等不经过会话的地方也用它。
    """
    return resolve_shop_config(
        config, platform=shop.platform, category=shop.category, shop_override=shop.config_overrides
    )
