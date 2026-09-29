"""Jev 判断层的合规护栏（§6.2 结论、§6.3 离线评测与重新引入的条件、§6.5 格式）。"""

from __future__ import annotations

import inspect

import pytest

from rodit.core.decision import DecisionRequest, require_buyer_data_ok
from rodit.core.errors import ComplianceError, DecisionError
from rodit.core.questions import Q_INTENT, ROUTER_MODEL, ROUTER_QUESTIONS
from rodit.decision import JevConditions, JevProvider
from rodit.decision import jev as jev_module
from rodit.testing.fakes import router_answers

ALL_MET = JevConditions(
    in_cloud_deployment=True,
    operator_and_dpa=True,
    platform_approved=True,
    chinese_eval_passed=True,
)


def _synthetic_state(text: str = "165 110斤穿什么码", **extra) -> dict:
    return {
        "synthetic": True,
        "platform": "douyin",
        "messages": [{"role": "buyer", "text": text}],
        "item_card": {"item_id": "i_555", "category": "女装/连衣裙"},
        "order_card": None,
        **extra,
    }


def _request(state: dict) -> DecisionRequest:
    return DecisionRequest(model=ROUTER_MODEL, state=state, questions=ROUTER_QUESTIONS)


class _Transport:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.payloads: list[dict] = []
        self.response = response
        self.error = error

    def __call__(self, payload: dict) -> dict:
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response  # type: ignore[return-value]
        answers = router_answers("size_advice", 0.92)
        return {
            "model": payload["model"],
            "answers": {k: a.model_dump(mode="json") for k, a in answers.items()},
        }


def test_conditions_default_to_unmet():
    c = JevConditions()
    assert not c.all_met
    assert c.unmet() == [
        "in_cloud_deployment",
        "operator_and_dpa",
        "platform_approved",
        "chinese_eval_passed",
    ]
    assert not JevConditions(in_cloud_deployment=True, operator_and_dpa=True).all_met
    assert ALL_MET.all_met and ALL_MET.unmet() == []


def test_production_requires_all_conditions():
    with pytest.raises(ComplianceError, match="chinese_eval_passed"):
        JevProvider(
            _Transport(),
            JevConditions(in_cloud_deployment=True, operator_and_dpa=True, platform_approved=True),
            purpose="production",
        )
    jev = JevProvider(_Transport(), ALL_MET, purpose="production")
    assert jev.accepts_buyer_data is True
    require_buyer_data_ok(jev)


def test_offline_eval_never_accepts_buyer_data():
    jev = JevProvider(_Transport(), ALL_MET, purpose="offline_eval")
    assert jev.accepts_buyer_data is False
    with pytest.raises(ComplianceError):
        require_buyer_data_ok(jev)


def test_offline_eval_passes_jev_format_through():
    transport = _Transport()
    jev = JevProvider(transport, JevConditions(), purpose="offline_eval")
    request = _request(_synthetic_state())
    resp = jev.decide(request)
    (payload,) = transport.payloads
    assert set(payload) == {"model", "state", "questions"}
    assert payload["model"] == "jev-1.13.0"
    assert payload["state"] == request.state
    assert payload["questions"] == request.model_dump(mode="json")["questions"]
    assert payload["questions"][Q_INTENT]["type"] == "choice"
    assert resp.model == "jev-1.13.0"
    assert resp.choice(Q_INTENT).choice == "size_advice"


def test_offline_eval_requires_the_synthetic_flag():
    transport = _Transport()
    jev = JevProvider(transport, JevConditions(), purpose="offline_eval")
    for flag in (None, False, "true", 1):
        state = _synthetic_state()
        if flag is None:
            del state["synthetic"]
        else:
            state["synthetic"] = flag
        with pytest.raises(ComplianceError):
            jev.decide(_request(state))
    assert transport.payloads == []


@pytest.mark.parametrize(
    "state",
    [
        _synthetic_state("我手机 13812345678，加急发"),
        _synthetic_state("寄到杭州市西湖区文三路 188 号 3 单元 501 室"),
        _synthetic_state("有货吗", order_card={"note": "联系邮箱 buyer@example.com"}),
        _synthetic_state("有货吗", extra={"13812345678": "键里也不行"}),
    ],
)
def test_offline_eval_rejects_any_pii_anywhere_in_state(state):
    transport = _Transport()
    jev = JevProvider(transport, JevConditions(), purpose="offline_eval")
    with pytest.raises(ComplianceError):
        jev.decide(_request(state))
    assert transport.payloads == []


def test_transport_failures_and_bad_responses_are_decision_errors():
    request = _request(_synthetic_state())
    bad_responses = [
        "not a dict",
        {"model": "jev-1.13.0"},
        {"model": "jev-1.13.0", "answers": {Q_INTENT: {"type": "choice", "choice": "size_advice"}}},
        {"model": "jev-1.13.0", "answers": {"intent": {"type": "noul", "noul": 2}}},
    ]
    for response in bad_responses:
        jev = JevProvider(_Transport(response=response), JevConditions(), purpose="offline_eval")
        with pytest.raises(DecisionError):
            jev.decide(request)
    jev = JevProvider(
        _Transport(error=TimeoutError("超时")), JevConditions(), purpose="offline_eval"
    )
    with pytest.raises(DecisionError):
        jev.decide(request)


def test_production_jev_sends_router_state():
    transport = _Transport()
    jev = JevProvider(transport, ALL_MET, purpose="production", model="jev-laya")
    resp = jev.decide(_request({"platform": "taobao", "messages": []}))
    assert transport.payloads[0]["model"] == "jev-laya"
    assert resp.choice(Q_INTENT).choice == "size_advice"


def test_module_has_no_endpoint_or_credentials():
    source = inspect.getsource(jev_module).lower()
    for forbidden in ("http://", "https://", "api_key", "apikey", "authorization", "bearer"):
        assert forbidden not in source


def test_provider_name():
    assert JevProvider(_Transport(), JevConditions(), purpose="offline_eval").name == "jev"
