"""DecisionRouter：投票粒度、全自动的关键词"要求人工"兜底、decision_log 写失败不影响路由。

（集成审查修复）
"""

from __future__ import annotations

import logging

from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import DecisionConfig, RoutingConfig
from rodit.core.enums import Intent, Mode, RouteAction
from rodit.core.errors import StorageUnavailable
from rodit.decision import RuleProvider, build_llm_provider
from rodit.routing import ALERT_MULTI_UNSPLIT, DecisionRouter, RoutingPolicy
from rodit.testing.fakes import (
    FakeClock,
    FakeDecisionProvider,
    FakeLLMClient,
    make_context,
    router_answers,
)
from rodit.testing.fixtures import build_demo_world

W = build_demo_world()
A = RouteAction
AUTO_CFG = {"auto_reply_enabled": True}


def _router(provider, *, audit=None):
    audit = audit if audit is not None else InMemoryAuditSink()
    router = DecisionRouter(
        provider, RuleProvider(), RoutingPolicy(RoutingConfig()), audit, FakeClock(W.now)
    )
    return router, audit


def _vote(intent: str, *, need_human: bool = False, multi: bool = False) -> dict:
    yes_no = {True: {"answer": "是", "p": 0.9}, False: {"answer": "否", "p": 0.1}}
    return {
        "intent": {"choice": intent},
        "need_human": yes_no[need_human],
        "emotion": {"score": 0},
        "multi_intent": yes_no[multi],
    }


def _default_route_provider(votes: list[dict] | None = None, *, every: dict | None = None):
    """默认配置（vote_samples=3、过半数有效票）的路由判断层：按顺序给出采样，或每次都给 every。"""
    llm = FakeLLMClient(lambda r: every) if every is not None else FakeLLMClient(responses=votes)
    return build_llm_provider(DecisionConfig(), purpose="route", llm=llm)


def test_two_of_three_votes_for_a_human_transfer_in_auto_mode():
    """审查复现：need_human 2/3 票，默认阈值 0.7 → 以前落到规则 6 自动回复。"""
    provider = _default_route_provider(
        [
            _vote("greeting", need_human=True),
            _vote("greeting", need_human=True),
            _vote("greeting"),
        ]
    )
    router, _ = _router(provider)
    ctx = make_context("你好，在吗", W, mode=Mode.AUTO, config_update=AUTO_CFG)
    d = router.route(ctx)
    assert (d.action, d.rule, d.auto_eligible) == (A.TRANSFER_HUMAN, 2, False)


def test_auto_mode_keyword_need_human_overrides_a_missed_judgement():
    provider = _default_route_provider(every=_vote("greeting"))
    router, _ = _router(provider)
    ctx = make_context("你好，我要找人工客服", W, mode=Mode.AUTO, config_update=AUTO_CFG)
    d = router.route(ctx)
    assert (d.action, d.rule, d.degraded, d.provider) == (A.TRANSFER_HUMAN, 2, False, "incloud_llm")
    # 智能辅助：以判断层为准（有人把关），不额外看关键词
    assisted = router.route(make_context("你好，我要找人工客服", W))
    assert (assisted.action, assisted.rule, assisted.need_human) == (A.ANSWER, 6, 0.0)
    # 全自动、没提人工：照常自动回复
    calm = router.route(make_context("你好，在吗", W, mode=Mode.AUTO, config_update=AUTO_CFG))
    assert (calm.rule, calm.auto_eligible) == (6, True)


def test_multi_intent_with_unanimous_main_intent_alerts_the_agent():
    provider = _default_route_provider(every=_vote("product_spec", multi=True))
    router, _ = _router(provider)
    d = router.route(make_context("这个是纯棉的吗？另外什么时候发货？", W))
    assert (d.action, d.rule, d.multi_intent) == (A.ANSWER, 6, 1.0)
    assert ALERT_MULTI_UNSPLIT in d.alerts


class _BrokenAudit(InMemoryAuditSink):
    def log_decision(self, entry) -> None:  # noqa: ANN001
        raise StorageUnavailable("decision_log 写入失败：连接数已满 13812345678")


def test_decision_log_failure_does_not_block_routing(caplog):
    provider = FakeDecisionProvider(router_answers("shipping_policy", 0.92), name="incloud_llm")
    router, _ = _router(provider, audit=_BrokenAudit())
    with caplog.at_level(logging.WARNING):
        d = router.route(make_context("什么时候发货", W))
    assert (d.action, d.intent, d.degraded) == (A.ANSWER, Intent.SHIPPING_POLICY, False)
    assert "StorageUnavailable" in caplog.text
    assert "13812345678" not in caplog.text  # 只记异常类型，不记消息
