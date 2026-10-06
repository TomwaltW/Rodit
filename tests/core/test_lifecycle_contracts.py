"""T0 契约：店铺授权续期、订购事件、平台接口失败原因、审计事件名、时间预算与店铺生命周期配置、
配置指纹、端口协议，以及对应的测试替身（全部合成数据，不联网）。"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

import rodit.adapters.auth as adapters_auth
from rodit.core import scopes
from rodit.core.audit import (
    DEGRADED_BUDGET,
    DEGRADED_TENANT_ISOLATION,
    LATENCY_SINCE_RECEIVED,
    OPTIONAL_RESULT_DETAIL_KEYS,
    RESULT_AUTO_GATE,
    RESULT_BUDGET_EXCEEDED,
    RESULT_DETAIL_KEYS,
    SYSTEM_EVENT_ACTIONS,
    AuditAction,
    InMemoryAuditSink,
)
from rodit.core.config import (
    EngineConfig,
    LatencyConfig,
    LLMConfig,
    TenancyConfig,
    config_fingerprint,
    llm_endpoint,
    load_config,
)
from rodit.core.decision import DecisionRequest
from rodit.core.enums import Intent, Platform
from rodit.core.errors import AuthExpired, ToolError, ToolUnavailable
from rodit.core.models import RefreshedToken, ShopAuth, SubscriptionEvent
from rodit.core.ports import (
    AuditLog,
    ConversationRepository,
    ShopAuthRepository,
    TokenProvider,
    TokenRefreshPort,
)
from rodit.core.questions import QA_QUESTIONS, ROUTER_QUESTIONS
from rodit.decision.llm import InCloudLLMProvider
from rodit.inmemory import InMemoryConversationRepository, InMemoryShopAuthRepository
from rodit.testing.fakes import (
    FakeClock,
    FakeSubscriptionParser,
    FakeTokenRefresher,
    judge_output,
    parse_subscription,
    scripted_judge_llm,
    subscription_push,
)
from rodit.testing.fixtures import DEFAULT_NOW, SHOP_A, SHOP_B

YAML = Path(__file__).resolve().parents[2] / "config" / "rodit.yaml"
NOW = DEFAULT_NOW


# ================================================================ 模型


def test_shop_auth_new_fields_are_optional_and_secret():
    auth = ShopAuth(shop_id=SHOP_A, access_token=SecretStr("tok-a"), scopes=[], expires_at=NOW)
    assert auth.refresh_token is None and auth.refresh_expires_at is None
    assert auth.refreshed_at is None and auth.needs_reauth is False
    full = auth.model_copy(update={"refresh_token": SecretStr("tok-refresh-secret")})
    assert "tok-refresh-secret" not in repr(full) and "tok-refresh-secret" not in str(full)
    with pytest.raises(ValidationError):
        ShopAuth(
            shop_id=SHOP_A,
            access_token=SecretStr("x"),
            scopes=[],
            expires_at=NOW,
            refreshed_at=NOW.replace(tzinfo=None),  # 时间必须带时区
        )


def test_subscription_event_contract():
    granted = SubscriptionEvent(
        kind="subscribed",
        shop_id=SHOP_A,
        platform=Platform.TAOBAO,
        at=NOW,
        scopes=sorted(scopes.MVP_SCOPES),
        access_token=SecretStr("tok-sub-secret"),
        expires_at=NOW + timedelta(days=30),
        category="女装",
    )
    assert "tok-sub-secret" not in repr(granted)
    cancelled = SubscriptionEvent(kind="cancelled", shop_id=SHOP_A, platform="taobao", at=NOW)
    assert cancelled.access_token is None and cancelled.scopes == []
    for bad in (
        {"kind": "subscribed"},  # 开通没有令牌
        {"kind": "renewed", "access_token": SecretStr("t")},  # 续订没有到期时间
        {"kind": "paused"},  # 不认识的事件
        {"kind": "cancelled", "shop_id": ""},
        {"kind": "cancelled", "extra": 1},  # 不认识的字段
    ):
        data = {"shop_id": SHOP_A, "platform": "taobao", "at": NOW, **bad}
        with pytest.raises(ValidationError):
            SubscriptionEvent.model_validate(data)


def test_refreshed_token_contract():
    t = RefreshedToken(access_token=SecretStr("tok-new"), expires_at=NOW)
    assert t.refresh_token is None and t.refresh_expires_at is None
    assert "tok-new" not in repr(t)


# ================================================================ 异常


def test_tool_unavailable_reason():
    plain = ToolUnavailable("订单接口故障")
    assert plain.reason == "unavailable" and str(plain) == "订单接口故障"
    assert isinstance(plain, ToolError)
    assert ToolUnavailable().reason == "unavailable"
    assert ToolUnavailable("限流", reason="throttled").reason == "throttled"
    assert ToolUnavailable("超时", reason="timeout").reason == "timeout"
    with pytest.raises(ValueError):
        ToolUnavailable("x", reason="slow")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ToolUnavailable("x", "throttled")  # type: ignore[misc]  # reason 只能按关键字传


# ================================================================ 审计事件名与常量


def test_new_audit_actions_and_constants():
    expected = {
        "AUTH_GRANTED": "auth.granted",
        "AUTH_REFRESHED": "auth.refreshed",
        "AUTH_REFRESH_FAILED": "auth.refresh_failed",
        "SHOP_STATUS_CHANGED": "shop.status_changed",
        "SHOP_PURGED": "shop.purged",
        "SHOP_PURGE_STARTED": "shop.purge_started",
        "TENANT_VIOLATION": "tenant.violation",
        "ROLLOUT_INCIDENT": "shop.rollout_incident",
        "COMPLIANCE_FLAG": "compliance.flag",
        "COMPLIANCE_REVIEWED": "compliance.reviewed",
        "EVAL_VERDICT": "eval.verdict",
        # Phase 2 / 3（原来定义在 app.platform_metrics、feedback.rollout；取值不变，存量审计照读）
        "PLATFORM_METRICS_SYNCED": "platform_metrics.synced",
        "PLATFORM_METRICS_RECORDED": "platform_metrics.recorded",
        "ROLLOUT_SEMI_AUTO": "shop.rollout_semi_auto",
    }
    for name, value in expected.items():
        assert AuditAction[name].value == value
    assert len({a.value for a in AuditAction}) == len(AuditAction)  # 取值不重复
    assert SYSTEM_EVENT_ACTIONS == {"shop.purged", "shop.purge_started", "eval.verdict"}
    assert (DEGRADED_TENANT_ISOLATION, DEGRADED_BUDGET, LATENCY_SINCE_RECEIVED) == (
        "tenant_isolation",
        "budget",
        "since_received",
    )
    # 可选键（超时、全自动被生效时段拦下）：不在每次都有的键里（正常的消息 detail 仍然正好是
    # RESULT_DETAIL_KEYS）
    assert (RESULT_BUDGET_EXCEEDED, RESULT_AUTO_GATE) == ("budget_exceeded", "auto_gate")
    assert OPTIONAL_RESULT_DETAIL_KEYS == (RESULT_BUDGET_EXCEEDED, RESULT_AUTO_GATE)
    assert not set(OPTIONAL_RESULT_DETAIL_KEYS) & set(RESULT_DETAIL_KEYS)


# ================================================================ 配置


def test_latency_and_tenancy_defaults():
    cfg = EngineConfig()
    assert cfg.latency == LatencyConfig(message_budget_s=4.5, min_llm_remaining_s=0.8)
    assert cfg.tenancy == TenancyConfig(
        refresh_ahead_hours=24, purge_after_expiry_days=30, onboarding_phase=1
    )
    assert cfg.llm.public_name is None and cfg.llm.filing_no is None
    assert cfg.llm.require_disclosure is False
    with pytest.raises(ValidationError):
        LatencyConfig(message_budget_s=1.0, min_llm_remaining_s=1.0)
    with pytest.raises(ValidationError):
        LatencyConfig(message_budget_s=0)
    with pytest.raises(ValidationError):
        TenancyConfig(onboarding_phase=3)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TenancyConfig(purge_after_expiry_days=-1)
    disclosed = LLMConfig(public_name="某模型", filing_no="备案-合成-0001", require_disclosure=True)
    assert disclosed.public_name == "某模型"


def test_llm_endpoint_uses_the_file_only_when_both_fields_are_set():
    env = {"RODIT_LLM_BASE_URL": " https://env.example.invalid/v1 ", "RODIT_LLM_MODEL": "env-m"}
    full = LLMConfig(base_url="https://file.example.invalid/v1", model="file-m")
    assert llm_endpoint(full, env) == ("https://file.example.invalid/v1", "file-m", True)
    half = LLMConfig(base_url="https://file.example.invalid/v1")  # 只写一项：不生效
    assert llm_endpoint(half, env) == ("https://env.example.invalid/v1", "env-m", False)
    assert llm_endpoint(LLMConfig(), {}) == ("", "", False)
    assert llm_endpoint(LLMConfig(), {"X_BASE_URL": "u", "X_MODEL": "m"}, prefix="X_")[:2] == (
        "u",
        "m",
    )


def test_yaml_matches_new_defaults():
    cfg = load_config(YAML)
    default = EngineConfig()
    assert cfg.latency == default.latency
    assert cfg.tenancy == default.tenancy
    assert cfg.llm.require_disclosure is False and cfg.llm.public_name is None


def test_config_fingerprint_is_stable_and_sensitive():
    base = EngineConfig()
    fp = config_fingerprint(base)
    assert len(fp) == 64 and all(c in "0123456789abcdef" for c in fp)
    assert config_fingerprint(EngineConfig()) == fp
    # 同样的配置，不管字典的键顺序如何，指纹一样
    reordered = EngineConfig.model_validate(
        dict(reversed(list(base.model_dump(mode="json").items())))
    )
    assert config_fingerprint(reordered) == fp
    # 任何一个取值变了，指纹就变
    changed = base.model_copy(
        update={"routing": base.routing.model_copy(update={"assist_threshold": 0.61})}
    )
    assert config_fingerprint(changed) != fp
    by_intent = base.model_copy(
        update={
            "routing": base.routing.model_copy(
                update={"assist_threshold_by_intent": {Intent.SIZE_ADVICE: 0.7}}
            )
        }
    )
    assert config_fingerprint(by_intent) not in {fp, config_fingerprint(changed)}
    # YAML 读出来的配置：读两次一样；和内置默认值不同（YAML 里有平台 / 类目默认）
    assert config_fingerprint(load_config(YAML)) == config_fingerprint(load_config(YAML))
    assert config_fingerprint(load_config(YAML)) != fp


# ================================================================ 权限名与端口


def test_scopes_live_in_core_and_are_reexported():
    assert adapters_auth.MVP_SCOPES is scopes.MVP_SCOPES
    assert adapters_auth.PHASE2_SCOPES is scopes.PHASE2_SCOPES
    assert adapters_auth.SCOPE_REFUND_WRITE == scopes.SCOPE_REFUND_WRITE == "refund.write"
    assert scopes.SCOPE_REFUND_WRITE not in scopes.MVP_SCOPES


def test_in_memory_implementations_satisfy_the_extended_ports():
    assert isinstance(InMemoryShopAuthRepository(), ShopAuthRepository)
    assert isinstance(InMemoryConversationRepository(), ConversationRepository)
    assert isinstance(InMemoryAuditSink(), AuditLog)
    assert isinstance(FakeTokenRefresher(), TokenRefreshPort)
    guard = adapters_auth.AuthGuard(InMemoryShopAuthRepository(), FakeClock())
    assert isinstance(adapters_auth.AuthGuardTokenProvider(guard), TokenProvider)


# ================================================================ 测试替身：令牌续期


def test_fake_token_refresher_defaults_scripts_and_records():
    clock = FakeClock(NOW)
    fake = FakeTokenRefresher(Platform.DOUYIN, clock=clock)
    assert fake.platform == Platform.DOUYIN
    first = fake.refresh(SHOP_A, SecretStr("tok-old-refresh"))
    assert first.expires_at == NOW + timedelta(days=30)
    assert first.refresh_token is not None and first.refresh_expires_at == NOW + timedelta(90)
    second = fake.refresh(SHOP_A, SecretStr("tok-old-refresh"))
    assert second.access_token.get_secret_value() != first.access_token.get_secret_value()
    scripted = RefreshedToken(access_token=SecretStr("tok-scripted"), expires_at=NOW)
    fake.script(SHOP_B, scripted, AuthExpired("续期令牌已失效"))
    assert fake.refresh(SHOP_B, SecretStr("r")).access_token.get_secret_value() == "tok-scripted"
    with pytest.raises(AuthExpired):
        fake.refresh(SHOP_B, SecretStr("r"))
    fake.always(SHOP_B, ToolUnavailable("限流", reason="throttled"))
    for _ in range(2):
        with pytest.raises(ToolUnavailable):
            fake.refresh(SHOP_B, SecretStr("r"))
    fake.always(SHOP_B, None)
    assert fake.refresh(SHOP_B, SecretStr("r")).access_token is not None
    assert fake.calls_for(SHOP_A) == 2 and fake.calls_for(SHOP_B) == 5
    assert "tok-old-refresh" not in repr(fake.calls)
    no_rotate = FakeTokenRefresher(clock=clock, rotate_refresh=False).refresh(
        SHOP_A, SecretStr("r")
    )
    assert no_rotate.refresh_token is None and no_rotate.refresh_expires_at is None


# ================================================================ 测试替身：订购推送


def test_parse_subscription_reads_only_known_fields():
    raw = subscription_push(
        "subscribed",
        SHOP_A,
        refresh_token="tok-refresh-synthetic",
        category="女装",
        buyer_nick="昵称不该出现",
        mobile="13800000000",
    )
    event = parse_subscription(raw)
    assert event is not None and event.kind == "subscribed" and event.shop_id == SHOP_A
    assert set(event.scopes) == scopes.MVP_SCOPES and event.category == "女装"
    assert event.access_token is not None and event.refresh_token is not None
    assert event.expires_at == NOW + timedelta(days=30)
    dumped = event.model_dump_json()
    assert "昵称" not in dumped and "13800000000" not in dumped
    csv = parse_subscription({**raw, "scopes": "order.read, item.read"})
    assert csv is not None and csv.scopes == ["order.read", "item.read"]
    cancelled = parse_subscription(subscription_push("auth_revoked", SHOP_A))
    assert cancelled is not None and cancelled.kind == "auth_revoked"
    assert cancelled.access_token is None


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "message", "shop_id": SHOP_A},
        {"shop_id": SHOP_A},
        {"type": "subscription", "event": "subscribed", "shop_id": SHOP_A, "platform": "taobao"},
        {
            "type": "subscription",
            "event": "exploded",
            "shop_id": SHOP_A,
            "platform": "taobao",
            "at": NOW.isoformat(),
        },
        {
            "type": "subscription",
            "event": "cancelled",
            "shop_id": SHOP_A,
            "platform": "jd",
            "at": NOW.isoformat(),
        },
        "not a mapping",
    ],
)
def test_parse_subscription_rejects_unrecognized(raw):
    assert parse_subscription(raw) is None  # type: ignore[arg-type]


def test_platform_bound_subscription_parser():
    raw = subscription_push("cancelled", SHOP_A, platform=Platform.DOUYIN)
    assert FakeSubscriptionParser(Platform.DOUYIN)(raw) is not None
    assert FakeSubscriptionParser(Platform.TAOBAO)(raw) is None  # 别的平台的推送不处理


# ================================================================ 测试替身：判断层大模型脚本


def _route_request(text: str) -> DecisionRequest:
    return DecisionRequest(
        model="incloud_llm",
        state={"platform": "taobao", "messages": [{"role": "buyer", "text": text}]},
        questions=ROUTER_QUESTIONS,
    )


def test_scripted_judge_llm_answers_every_asked_question_in_one_request():
    llm = scripted_judge_llm(
        route=[
            (
                "穿什么码",
                {
                    "intent": {"choice": "size_advice"},
                    "multi_intent": {"answer": "是", "p": 0.9},
                    "second_intent": {"choice": "shipping_policy"},  # 没问到的问题：丢掉
                },
            )
        ]
    )
    provider = InCloudLLMProvider(llm, samples=3, temperature=0.7, purpose="route")
    response = provider.decide(_route_request("165 110斤穿什么码？什么时候发货？"))
    assert set(response.answers) == set(ROUTER_QUESTIONS)
    assert response.answers["intent"].choice == "size_advice"
    assert response.answers["multi_intent"].noul == pytest.approx(1.0)  # 三票都是"是"
    assert response.answers["need_human"].noul == 0.0  # 默认：不要人工
    assert len(llm.calls_for("route")) == 3  # 一票一个请求
    with pytest.raises(AssertionError):
        judge_output(llm.calls[0], {}, defaults={})  # 单选题必须写进脚本


def test_scripted_judge_llm_qa_defaults_pass_and_unknown_messages_fail_loudly():
    llm = scripted_judge_llm(route={"intent": {"choice": "greeting"}})
    qa = InCloudLLMProvider(llm, samples=1, temperature=0.0, purpose="qa")
    request = DecisionRequest(
        model="incloud_llm",
        state={"platform": "taobao", "question": "在吗", "answer": "亲，在的"},
        questions=QA_QUESTIONS,
    )
    answers = qa.decide(request).answers
    assert answers["grounded"].noul > 0.8 and answers["promise_risk"].noul < 0.3
    keyed = scripted_judge_llm(route=[("发货", {"intent": {"choice": "shipping_policy"}})])
    with pytest.raises(AssertionError, match="判断层脚本里没有这条消息"):
        InCloudLLMProvider(keyed, samples=1, temperature=0.0, purpose="route").decide(
            _route_request("在吗")
        )
    gen = scripted_judge_llm()
    with pytest.raises(AssertionError):
        from rodit.core.ports import LLMMessage, LLMRequest

        gen.complete(
            LLMRequest(purpose="generate", messages=[LLMMessage(role="user", content="x")])
        )
