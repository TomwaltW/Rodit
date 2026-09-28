"""错误映射（§12.2 降级、§12.3 不回显令牌、连接串、个人信息）。"""

from __future__ import annotations

import pytest
from api_support import TOKENS, Api, build_api
from pydantic import BaseModel, Field

from rodit.core.errors import (
    ActionNotAllowed,
    AuthExpired,
    ComplianceError,
    ConcurrencyConflict,
    DecisionError,
    KnowledgeUnavailable,
    LLMError,
    ReadOnlyViolation,
    ShopInactive,
    StorageUnavailable,
    TenantIsolationError,
    ToolUnavailable,
)
from rodit.testing.fixtures import SHOP_A

REPORT = f"/v1/shops/{SHOP_A}/reports/adoption"
DSN = "postgresql://rodit:s3cr3t-pw@db.internal:5432/rodit"
PHONE = "13812345678"


def _raise(exc: BaseException):
    def boom(*args, **kwargs):
        raise exc

    return boom


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (KeyError("s_1001 没有这条记录"), 404, "not_found"),
        (ActionNotAllowed("售后单状态已变化，请刷新后重新确认"), 409, "action_not_allowed"),
        (ConcurrencyConflict("审核记录版本冲突"), 409, "conflict"),
        (ValueError("问卷没有这些题号：[99]"), 422, "validation_error"),
        (AuthExpired("店铺 s_1001 的授权已过期"), 403, "shop_not_authorized"),
        (ShopInactive(SHOP_A), 403, "shop_inactive"),
        (TenantIsolationError("售后单 as_9 不属于店铺 s_2002"), 403, "tenant_isolation"),
        (ComplianceError("Jev 不能处理买家数据"), 403, "compliance"),
        (StorageUnavailable(f"连不上 {DSN}"), 503, "storage_unavailable"),
        (ToolUnavailable(f"售后接口限流 {DSN}"), 503, "tool_unavailable"),
        (KnowledgeUnavailable(f"知识库超时 {DSN}"), 503, "knowledge_unavailable"),
        (LLMError(f"大模型报错 key=sk-{DSN}"), 500, "internal_error"),
        (DecisionError("判断层不可用"), 500, "internal_error"),
        (ReadOnlyViolation("问答链路调用了写接口"), 500, "internal_error"),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else str(v),
)
def test_service_exceptions_map_to_status_codes(api: Api, monkeypatch, exc, status, code):
    monkeypatch.setattr(api.service, "adoption", _raise(exc))
    r = api.get(REPORT, "manager_a")
    assert r.status_code == status
    body = r.json()
    assert set(body) == {"error", "message"} and body["error"] == code
    assert body["message"]
    for secret in (DSN, "s3cr3t", "sk-", TOKENS["manager_a"]):
        assert secret not in body["message"]


def test_business_messages_are_passed_through_when_safe(api: Api, monkeypatch):
    monkeypatch.setattr(api.service, "adoption", _raise(ActionNotAllowed("请先生成审核建议")))
    assert api.get(REPORT, "manager_a").json()["message"] == "请先生成审核建议"
    # 跨店异常不透露别的店铺的数据
    monkeypatch.setattr(
        api.service, "adoption", _raise(TenantIsolationError("售后单 as_9 不属于店铺 s_2002"))
    )
    message = api.get(REPORT, "manager_a").json()["message"]
    assert "as_9" not in message and "s_2002" not in message


def test_messages_are_masked_and_truncated(api: Api, monkeypatch):
    monkeypatch.setattr(
        api.service, "adoption", _raise(ActionNotAllowed(f"买家 {PHONE} 的售后单不能操作" * 50))
    )
    message = api.get(REPORT, "manager_a").json()["message"]
    assert PHONE not in message and len(message) <= 300


def test_pydantic_validation_errors_from_the_service_do_not_echo_input(api: Api, monkeypatch):
    class Probe(BaseModel):
        phone: str = Field(max_length=3)

    try:
        Probe(phone=PHONE)
    except ValueError as exc:
        error = exc
    monkeypatch.setattr(api.service, "adoption", _raise(error))
    r = api.get(REPORT, "manager_a")
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert "phone" in r.json()["message"] and PHONE not in r.text


def test_request_validation_errors_do_not_echo_the_body(api: Api):
    r = api.post(
        f"/v1/shops/{SHOP_A}/suggestions/1/edited",
        "agent_a",
        json={"edited_text": f"买家手机 {PHONE} " * 200},
    )
    assert r.status_code == 422
    assert set(r.json()) == {"error", "message"} and PHONE not in r.text
    assert "edited_text" in r.json()["message"]
    r = api.post(
        f"/v1/shops/{SHOP_A}/standalone/ask",
        "agent_a",
        content=b'{"session_id": "w", "question": "' + PHONE.encode() + b'"',
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422 and PHONE not in r.text


def test_unknown_routes_and_methods_use_the_same_error_shape(api: Api):
    r = api.get("/v1/nope")
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    r = api.get(f"/v1/shops/{SHOP_A}/revoke", "merchant_a")
    assert r.status_code == 405 and r.json()["error"] == "method_not_allowed"


def test_unexpected_exceptions_become_generic_500(monkeypatch):
    api = build_api(raise_server_exceptions=False)
    monkeypatch.setattr(api.service, "adoption", _raise(RuntimeError(f"boom {DSN}")))
    r = api.get(REPORT, "manager_a")
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "message": "服务内部错误，请稍后重试"}


def test_error_logs_do_not_contain_exception_messages(api: Api, monkeypatch, caplog):
    monkeypatch.setattr(api.service, "adoption", _raise(StorageUnavailable(f"连不上 {DSN}")))
    with caplog.at_level("ERROR", logger="rodit.api"):
        assert api.get(REPORT, "manager_a").status_code == 503
    assert caplog.records and DSN not in caplog.text and TOKENS["manager_a"] not in caplog.text
