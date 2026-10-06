"""配置（方案 §6.6 阈值、§5.5 放行规则、§7.2 异常阈值、§7.3.2 审核规则、§9.3 配置分层）。

- 阈值由我们统一校准（§13.3），不开放给商家；商家能改的只有 ShopConfig（core.models）。
- 默认值与方案一致；部署时用 config/rodit.yaml 覆盖。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rodit.core.enums import Intent, Platform, ProviderName
from rodit.core.lexicon import ABSOLUTE_TERM_EXCEPTIONS, ABSOLUTE_TERMS
from rodit.core.models import Shop, ShopConfig


class _Model(BaseModel):
    # 校验报错不带输入值：配置里误写的密钥（如 llm.api_key）、extra_body 的值都不会被打出来。
    # 要设在每一层（EngineConfig 和各段都继承这里）：只设在内层，外层的报错照样带值
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


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
    # 半自动（§5.6、§13.4）：店铺开了半自动的类型，置信度达到这个值的推荐直接填进客服输入框
    semi_auto_threshold: float = Field(default=0.9, gt=0, le=1)
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


LLM_ENV_PREFIX = "RODIT_LLM_"
"""配置里没写全 llm.base_url / llm.model 时读的环境变量前缀：{前缀}BASE_URL、{前缀}MODEL，
以及请求参数 {前缀}EXTRA_BODY、{前缀}JSON_MODE、{前缀}LOGPROBS
（OpenAICompatibleClient.from_env）。"""

LLM_EXTRA_BODY_MAX_CHARS = 2048
"""extra_body 规范化 JSON 的长度上限（字符）：关思考模式这类参数只有几十个字符。"""
LLM_EXTRA_BODY_MAX_DEPTH = 6
LLM_CONTROLLED_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "response_format",
        "logprobs",
        "top_logprobs",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "n",
    }
)
"""客户端自己控制的请求字段，extra_body 不能设置：模型名、提示词、非流式调用、JSON 模式和
logprobs（用开关 json_mode / logprobs）、各环节自己的温度和 token 上限（路由投票、质检、生成
各不相同；max_completion_tokens 和客户端发的 max_tokens 同时出现时有的服务直接报错）、
只取第一个候选。"""

# 看起来像凭据的键（按词比较：max_tokens、budget_tokens 这类计数不算）和值的开头
_SECRET_WORDS = frozenset(
    {
        "token",
        "auth",
        "authorization",
        "secret",
        "password",
        "passwd",
        "pwd",
        "credential",
        "credentials",
        "cookie",
        "signature",
        "sign",
        "bearer",
        "apikey",
        "key",  # app_key、appKey、x_key……（已知不是凭据的见 _BENIGN_KEYS）
        "ak",
        "sk",
        "jwt",
        "passphrase",
    }
)
_SECRET_FRAGMENTS = (
    "apikey",
    "accesskey",
    "secretkey",
    "privatekey",
    "secret",
    "password",
    "passwd",
    "passphrase",
    "credential",
    "authorization",
)
_BENIGN_KEYS = frozenset({"prompt_cache_key"})
"""带 key 这个词、但不是凭据的已知参数（规范化的蛇形写法）。"""
# 值的开头：Bearer、sk-、百度 bce-v3/、PEM，JWT（eyJ）、AWS 风格的 AccessKey（AKIA）
_SECRET_VALUE_PREFIXES = ("bearer ", "sk-", "bce-v3/", "-----begin", "eyj", "akia")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def _key_words(key: str) -> tuple[str, list[str]]:
    """键名 → (规范化的蛇形写法, 词)：apiKey、X-Api-Key、api_key 都是 api_key。"""
    norm = _NON_WORD.sub("_", _CAMEL.sub("_", key).lower()).strip("_")
    return norm, [w for w in norm.split("_") if w]


def _looks_like_secret_key(key: str) -> bool:
    norm, words = _key_words(key)
    if norm in _BENIGN_KEYS:
        return False
    compact = norm.replace("_", "")
    return any(w in _SECRET_WORDS for w in words) or any(f in compact for f in _SECRET_FRAGMENTS)


def _key_path(path: str, key: str) -> str:
    shown = key if len(key) <= 40 else key[:40] + "…"
    return f"{path}.{shown}" if path else shown


def _walk_extra_body(value: Any, *, path: str, depth: int, source: str) -> None:
    """逐层检查：键是字符串、不像凭据，值是 JSON 值、字符串不像密钥，嵌套不太深。
    报错只带键名路径，不带值（值可能就是密钥）。"""
    if depth > LLM_EXTRA_BODY_MAX_DEPTH:
        raise ValueError(f"{source} 嵌套太深（超过 {LLM_EXTRA_BODY_MAX_DEPTH} 层）")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{source} 的键必须是字符串（{path or '顶层'}）")
            where = _key_path(path, key)
            if _looks_like_secret_key(key):
                raise ValueError(
                    f"{source} 里的键 {where} 看起来像凭据：密钥只放 API key 的环境变量"
                    "（llm.api_key_env，默认 RODIT_LLM_API_KEY），不放请求参数"
                )
            _walk_extra_body(item, path=where, depth=depth + 1, source=source)
    elif isinstance(value, list | tuple):
        for i, item in enumerate(value):
            _walk_extra_body(item, path=f"{path}[{i}]", depth=depth + 1, source=source)
    elif isinstance(value, str):
        if value.strip().lower().startswith(_SECRET_VALUE_PREFIXES):
            raise ValueError(
                f"{source} 里 {path or '顶层'} 的值看起来像密钥（以 sk- / Bearer 等开头）："
                "密钥只放 API key 的环境变量，不放请求参数"
            )
    elif value is not None and not isinstance(value, bool | int | float):
        raise ValueError(f"{source} 只能包含 JSON 值（{path or '顶层'} 不是）")


def check_extra_body(value: Any, *, source: str = "llm.extra_body") -> dict[str, Any]:
    """校验模型服务特有的请求参数（OpenAI 兼容接口之外的字段，如关闭思考模式），返回规范化的副本。

    - 必须是 JSON 对象；规范化 JSON 不超过 LLM_EXTRA_BODY_MAX_CHARS 个字符、嵌套不超过
      LLM_EXTRA_BODY_MAX_DEPTH 层；数字不能是 NaN / 无穷大；
    - 不能设置客户端控制的字段（LLM_CONTROLLED_FIELDS）；
    - 任何一层的键都不能像凭据（api_key、app_key、ak / sk、authorization、token、jwt、sign、
      secret、password……按词比较，max_tokens 这类计数、prompt_cache_key 不算），字符串值不能像
      密钥（sk-、Bearer、JWT 的 eyJ、AKIA 开头……）：凭据只放 API key 的环境变量。
      挡不全的（比如网关自己起名的键）也不会被打出来：给人看的输出只有键名，评测指纹里是值的哈希
      （LLMRequestOptions.summary / describe）。
    报错（ValueError）只带键名路径，不带值。"""
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} 必须是 JSON 对象（键值对）")
    _walk_extra_body(value, path="", depth=1, source=source)
    try:
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        raise ValueError(f"{source} 只能包含 JSON 值（数字不能是 NaN / 无穷大）") from None
    if len(text) > LLM_EXTRA_BODY_MAX_CHARS:
        raise ValueError(
            f"{source} 太大（{len(text)} 个字符，上限 {LLM_EXTRA_BODY_MAX_CHARS}）："
            "只放模型服务特有的少量参数"
        )
    controlled = sorted(k for k in value if _key_words(k)[0] in LLM_CONTROLLED_FIELDS)
    if controlled:
        raise ValueError(
            f"{source} 不能设置 {'、'.join(controlled)}：这些字段由客户端控制（JSON 模式、logprobs "
            "用开关 RODIT_LLM_JSON_MODE / RODIT_LLM_LOGPROBS 或 llm.json_mode / llm.logprobs；"
            "温度和 token 上限由路由、质检、生成各自的配置决定）"
        )
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    return parsed


def parse_extra_body(raw: str | None, *, source: str) -> dict[str, Any]:
    """环境变量里的 extra_body（JSON 对象文本）→ 校验过的 dict；没设或只有空白是 {}。"""
    text = (raw or "").strip()
    if not text:
        return {}
    if len(text) > 4 * LLM_EXTRA_BODY_MAX_CHARS:
        raise ValueError(f"{source} 太长（上限 {LLM_EXTRA_BODY_MAX_CHARS} 个字符的 JSON 对象）")
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        raise ValueError(
            f'{source} 不是合法的 JSON（应为 JSON 对象，如 {{"thinking": {{"type": "disabled"}}}}）'
        ) from None
    return check_extra_body(value, source=source)


_ON = frozenset({"on", "true", "1", "yes"})
_OFF = frozenset({"off", "false", "0", "no"})


def parse_switch(raw: str | None, *, source: str) -> bool | None:
    """on / off 开关（也认 true / false、1 / 0、yes / no，大小写、首尾空白不计）；没设返回 None。"""
    word = (raw or "").strip().lower()
    if not word:
        return None
    if word in _ON:
        return True
    if word in _OFF:
        return False
    raise ValueError(f"{source} 只能是 on 或 off")


class _Hidden:
    """没通过校验的 llm.extra_body：pydantic 的错误信息只显示原因，不显示原值（可能是密钥）。"""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __repr__(self) -> str:
        return "（已隐藏）"


_REQUEST_OPTION_DEFAULTS: dict[str, Any] = {"extra_body": {}, "json_mode": True, "logprobs": True}


class LLMConfig(_Model):
    """境内已备案大模型（D5）的调用配置。密钥不在这里：只写环境变量名，调用时再读。

    public_name / filing_no：对外公示的模型名称和备案号（§12.3 生成式人工智能服务登记 / 备案），
    网页工具和 /v1/me 展示；require_disclosure 为 True 时生产部署缺这两项拒绝启动。

    请求参数（extra_body、json_mode、logprobs）跟着接口走（llm_endpoint 的规则）：配置里
    base_url 和 model 都写了才用这里的，否则整组取环境变量 RODIT_LLM_EXTRA_BODY /
    RODIT_LLM_JSON_MODE / RODIT_LLM_LOGPROBS。只写了请求参数、没写接口的配置不会生效，直接报错。
    """

    base_url: str | None = None  # OpenAI 兼容接口地址；为空时从环境变量 RODIT_LLM_BASE_URL 读
    model: str | None = None  # 为空时从 RODIT_LLM_MODEL 读
    api_key_env: str = "RODIT_LLM_API_KEY"
    timeout_s: float = 10.0  # 单次调用的上限；各环节自己的预算见 decision / generation
    # 模型服务特有的请求参数（如关闭思考模式），校验规则见 check_extra_body
    extra_body: dict[str, Any] | None = None
    json_mode: bool = True  # 关掉时不发 response_format（服务 / 模型不支持 JSON 模式时）
    logprobs: bool = True  # 关掉时不发 logprobs / top_logprobs（不支持时；路由按投票占比算置信度）
    public_name: str | None = None  # 公示的模型名称【待定：D5 选定后填】
    filing_no: str | None = None  # 备案号【待定：D5 选定后填】
    require_disclosure: bool = False

    @model_validator(mode="before")
    @classmethod
    def _check_extra_body_quietly(cls, data: Any) -> Any:
        # 在字段校验之前查：不合格的原值换成 _Hidden，报错里不出现原值
        if isinstance(data, Mapping) and data.get("extra_body") is not None:
            try:
                checked: Any = check_extra_body(data["extra_body"])
            except ValueError as e:
                checked = _Hidden(str(e))
            data = {**data, "extra_body": checked}
        return data

    @field_validator("extra_body", mode="before")
    @classmethod
    def _reject_hidden(cls, v: Any) -> Any:
        if isinstance(v, _Hidden):
            raise ValueError(v.reason)
        return v

    @field_validator("base_url", "model", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        # 和环境变量同一条规则：去掉首尾空白，只有空白等于没写
        if isinstance(v, str):
            return v.strip() or None
        return v

    @model_validator(mode="after")
    def _request_options_need_endpoint(self) -> LLMConfig:
        # 只看和默认值不同的（model_dump 再读回来照样通过）：这些值在环境变量那一路不会生效
        values = {
            "extra_body": self.extra_body or {},
            "json_mode": self.json_mode,
            "logprobs": self.logprobs,
        }
        explicit = [name for name, v in values.items() if v != _REQUEST_OPTION_DEFAULTS[name]]
        if explicit and not (self.base_url and self.model):
            fields = "、".join(f"llm.{name}" for name in explicit)
            raise ValueError(
                f"{fields} 只在配置里同时写了 llm.base_url 和 llm.model 时生效"
                "（否则整组取环境变量）：要么把接口也写进配置，要么改用环境变量 "
                f"{LLM_ENV_PREFIX}EXTRA_BODY / "
                f"{LLM_ENV_PREFIX}JSON_MODE / {LLM_ENV_PREFIX}LOGPROBS"
            )
        return self


def llm_endpoint(
    config: LLMConfig, environ: Mapping[str, str] | None = None, *, prefix: str = LLM_ENV_PREFIX
) -> tuple[str, str, bool]:
    """线上单元实际用的大模型：（接口地址，模型名，是否来自配置文件）。

    规则：配置里 llm.base_url 和 llm.model 都写了用配置；否则两项都取环境变量
    {prefix}BASE_URL / {prefix}MODEL（只写了一项的配置不生效，不和环境变量拼）。
    拿不到的项是空字符串。
    组装客户端（app.production）、启动前检查和就绪检查（ops.serve、ops.health）、评测指纹
    （eval.fingerprint）都用它，几处不会各算各的。只读变量，不读密钥。"""
    if config.base_url and config.model:
        return config.base_url, config.model, True
    env = os.environ if environ is None else environ
    return env.get(f"{prefix}BASE_URL", "").strip(), env.get(f"{prefix}MODEL", "").strip(), False


@dataclass(frozen=True)
class LLMRequestOptions:
    """发给模型服务的请求参数（llm_request_options 解析，OpenAICompatibleClient.options 带着）。

    - extra_body：模型服务特有的参数（check_extra_body 校验过），标准字段优先；
    - json_mode：False 时不发 response_format（提示词本来就要求只输出 JSON）；
    - logprobs：False 时不发 logprobs / top_logprobs（路由的置信度按投票占比算）。
    extra_body 的值不出现在任何输出里（有的网关要在请求体里放凭据，键名校验挡不全）：
    summary() 是给人看、写进报告的（只有键名）；describe() 记进评测指纹（会改变回答的东西），
    键名加上值的规范化 JSON 的 SHA-256——值一改指纹就变，指纹里看不到值。"""

    extra_body: dict[str, Any] = field(default_factory=dict, hash=False)
    json_mode: bool = True
    logprobs: bool = True

    @staticmethod
    def _json(data: dict[str, Any]) -> str:
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _public(self) -> dict[str, Any]:
        return {
            "extra_body_keys": sorted(self.extra_body),
            "json_mode": self.json_mode,
            "logprobs": self.logprobs,
        }

    def summary(self) -> str:
        """给人看的（横幅、能力检查、自检报告）：extra_body 只有键名。"""
        return self._json(self._public())

    def describe(self) -> str:
        """评测指纹里的：键名 + extra_body 规范化 JSON 的 SHA-256（没有 extra_body 时不带哈希）。"""
        data = self._public()
        if self.extra_body:
            canonical = self._json(self.extra_body)
            data["extra_body_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self._json(data)


def _reject_api_key_inside(
    extra_body: Mapping[str, Any], key: str | None, *, source: str, api_key_env: str
) -> None:
    """extra_body 里出现了密钥原文（换了个不像凭据的键名放进去）：拒绝，否则会进指纹和报告。"""
    secret = (key or "").strip()
    if len(secret) >= 8 and secret in json.dumps(extra_body, ensure_ascii=False):
        raise ValueError(f"{source} 里不能放大模型密钥（{api_key_env} 的值）")


def env_request_options(
    environ: Mapping[str, str] | None = None,
    *,
    prefix: str = LLM_ENV_PREFIX,
    api_key_env: str | None = None,
) -> LLMRequestOptions:
    """环境变量给的请求参数：{prefix}EXTRA_BODY（JSON 对象）、{prefix}JSON_MODE、{prefix}LOGPROBS
    （on / off，默认 on）。值去掉首尾空白（和 BASE_URL / MODEL 一样）；不合法抛 ValueError
    （信息里只有变量名和原因，没有值）。"""
    env = os.environ if environ is None else environ
    key_env = api_key_env or f"{prefix}API_KEY"
    source = f"环境变量 {prefix}EXTRA_BODY"
    extra = parse_extra_body(env.get(f"{prefix}EXTRA_BODY"), source=source)
    _reject_api_key_inside(extra, env.get(key_env), source=source, api_key_env=key_env)
    json_mode = parse_switch(env.get(f"{prefix}JSON_MODE"), source=f"环境变量 {prefix}JSON_MODE")
    logprobs = parse_switch(env.get(f"{prefix}LOGPROBS"), source=f"环境变量 {prefix}LOGPROBS")
    return LLMRequestOptions(
        extra_body=extra,
        json_mode=True if json_mode is None else json_mode,
        logprobs=True if logprobs is None else logprobs,
    )


def llm_request_options(
    config: LLMConfig, environ: Mapping[str, str] | None = None, *, prefix: str = LLM_ENV_PREFIX
) -> LLMRequestOptions:
    """线上单元实际发的请求参数，和 llm_endpoint 同一条规则：配置里 llm.base_url 和 llm.model
    都写了用配置的 llm.extra_body / json_mode / logprobs；否则整组取环境变量
    （env_request_options）。
    组装客户端（OpenAICompatibleClient.from_config）、启动前检查和就绪检查、评测指纹都用它。
    不合法抛 ValueError。"""
    env = os.environ if environ is None else environ
    _, _, from_config = llm_endpoint(config, env, prefix=prefix)
    if not from_config:
        return env_request_options(env, prefix=prefix, api_key_env=config.api_key_env)
    options = LLMRequestOptions(
        extra_body=dict(config.extra_body or {}),
        json_mode=config.json_mode,
        logprobs=config.logprobs,
    )
    _reject_api_key_inside(
        options.extra_body,
        env.get(config.api_key_env),
        source="llm.extra_body",
        api_key_env=config.api_key_env,
    )
    return options


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
