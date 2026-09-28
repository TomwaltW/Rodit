"""端到端测试的组装代码和脚本化替身（全部合成数据，不联网）。

- 组件都是真的：路由（DecisionRouter + RoutingPolicy）、检索（KnowledgeRetriever + 工具证据）、
  生成（DefaultAnswerGenerator）、质检（DefaultQualityGate）、问答引擎、反馈、售后审核助手，
  由 rodit.app.demo.build_demo_unit 组装，经 AssistService 走接入层适配器进来；
- 只有"模型"是脚本化的：生成用 FakeLLMClient（按当前买家消息给出预设回答），判断层用
  FakeDecisionProvider（按当前买家消息 / 待检查的回答给出预设答案），或者整条链路共用一个
  FakeLLMClient（路由、质检、生成都走大模型，和生产的组装方式一样）。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import Any

from rodit.adapters import TaobaoMockAdapter
from rodit.app import AssistService, Unit
from rodit.app.demo import build_demo_unit, taobao_event
from rodit.core.decision import Answer, DecisionRequest
from rodit.core.enums import Intent, Platform, Role
from rodit.core.models import EngineResult
from rodit.core.ports import LLMRequest
from rodit.testing.fakes import FakeDecisionProvider, FakeLLMClient, qa_answers, router_answers
from rodit.testing.fixtures import SHOP_A, DemoWorld, build_demo_world

GENERATE = "generate"
RETRY_MARK = "上次回答未通过质检"
_CURRENT = re.compile(r"（当前消息）：(?P<text>.*)$", re.M)


# ================================================================ 大模型（生成）


def prompt_text(request: LLMRequest) -> str:
    return "\n".join(m.content for m in request.messages)


def current_message(request: LLMRequest) -> str:
    """生成提示词【对话】里的当前买家消息（已脱敏）。"""
    found = _CURRENT.findall(prompt_text(request))
    return found[-1] if found else ""


def llm_answer(text: str, *evidence_ids: str, need_clarify: bool = False) -> dict[str, Any]:
    """§5.4 规定的生成输出 JSON。"""
    return {"answer": text, "evidence_ids": list(evidence_ids), "need_clarify": need_clarify}


GenScript = Mapping[str, "dict[str, Any] | Exception | Callable[[LLMRequest], Any]"]


def _generation_handler(script: GenScript) -> Callable[[LLMRequest], Any]:
    def handler(request: LLMRequest) -> Any:
        if request.purpose != GENERATE:
            return AssertionError(f"这个测试的大模型只处理生成请求，收到 {request.purpose}")
        current = current_message(request)
        for key, value in script.items():
            if key in current:
                return value(request) if callable(value) else value
        return AssertionError(f"生成脚本里没有这条消息：{current}")

    return handler


def scripted_llm(script: GenScript) -> FakeLLMClient:
    """生成请求：按当前买家消息里出现的关键词取预设回答（dict / 异常 / 以请求为参数的函数）。

    路由、质检请求不应该走到这个大模型（测试里判断层都另外注入），走到了就是组装错了。
    """
    return FakeLLMClient(_generation_handler(script))


_STATE = re.compile(r"## 状态（state）\n(?P<state>.*?)\n\n## 问题", re.S)

QA_PASS_JSON: dict[str, Any] = {
    "grounded": {"answer": "是", "p": 0.95},
    "answers_question": {"answer": "是", "p": 0.95},
    "promise_risk": {"answer": "否", "p": 0.05},
    "platform_violation": {"answer": "否", "p": 0.02},
    "tone": {"score": 3},
}


def judge_state(request: LLMRequest) -> dict[str, Any]:
    """判断层提示词（rodit.decision.llm.build_judge_messages）里的 state JSON。"""
    m = _STATE.search(prompt_text(request))
    assert m is not None, "不是判断层提示词"
    return json.loads(m.group("state"))


def route_json(intent: Intent) -> dict[str, Any]:
    """大模型判断层对路由四个问题的一次作答（§6.5）。"""
    return {
        "intent": {"choice": intent.value},
        "need_human": {"answer": "否", "p": 0.02},
        "emotion": {"score": 0},
        "multi_intent": {"answer": "否", "p": 0.05},
    }


def production_like_llm(routes: Sequence[tuple[str, Intent]], generate: GenScript) -> FakeLLMClient:
    """路由、质检、生成共用一个大模型（和生产组装一样，判断层是 incloud_llm）：
    路由按 state 里的当前买家消息查表，质检一律通过，生成按 scripted_llm 的规则。"""
    gen = _generation_handler(generate)

    def handler(request: LLMRequest) -> Any:
        if request.purpose == "route":
            text = _buyer_text(judge_state(request))
            for key, intent in routes:
                if key in text:
                    return route_json(intent)
            return AssertionError(f"路由脚本里没有这条消息：{text}")
        if request.purpose == "qa":
            judge_state(request)
            return QA_PASS_JSON
        return gen(request)

    return FakeLLMClient(handler)


def failing_llm(error: Exception) -> FakeLLMClient:
    """所有请求（路由、质检、生成）都抛同一个错误：大模型超时或报错（§12.2）。"""
    return FakeLLMClient(lambda request: error)


# ================================================================ 判断层


def _buyer_text(state: Mapping[str, Any]) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == Role.BUYER.value:
            return str(m.get("text") or "")
    return ""


def scripted_router(table: Sequence[tuple[str, Mapping[str, Answer]]]) -> FakeDecisionProvider:
    """路由判断层：按当前买家消息里的关键词给出预设答案（router_answers）。没写到的消息抛
    AssertionError（引擎会记成 router 降级，测试断言随之失败，不会被悄悄兜过去）。"""

    def answers(request: DecisionRequest) -> Mapping[str, Answer]:
        text = _buyer_text(request.state)
        for key, value in table:
            if key in text:
                return value
        raise AssertionError(f"路由脚本里没有这条消息：{text}")

    return FakeDecisionProvider(answers, name="scripted_router")


def route(intent: Intent, confidence: float = 0.92, **kw: float) -> dict[str, Answer]:
    return router_answers(intent, confidence, **kw)


def scripted_qa(
    rules: Sequence[tuple[str, Mapping[str, Answer]]] = (),
) -> FakeDecisionProvider:
    """质检判断层：回答里出现关键词时用预设答案，否则全部通过（qa_answers 默认值）。"""

    def answers(request: DecisionRequest) -> Mapping[str, Answer]:
        answer = str(request.state.get("answer") or "")
        for key, value in rules:
            if key in answer:
                return value
        return qa_answers()

    return FakeDecisionProvider(answers, name="scripted_qa")


# ================================================================ 组装


def demo_service(
    platform: Platform = Platform.TAOBAO,
    *,
    world: DemoWorld | None = None,
    llm: Any = None,
    router: Any = None,
    qa: Any = None,
) -> AssistService:
    unit = build_demo_unit(
        platform, world=world or build_demo_world(), llm=llm, route_provider=router, qa_provider=qa
    )
    return AssistService(unit)


def enable_auto(service: AssistService, shop_id: str = SHOP_A) -> TaobaoMockAdapter:
    """给单元加一个淘宝模式 A 适配器（开了全自动发送），并打开店铺的全自动总开关。"""
    unit = service.unit
    adapter = TaobaoMockAdapter(
        unit.tools, unit.clock, unit.hasher, access="A", auto_send_enabled=True
    )
    unit.adapters.register(adapter)
    repo = unit.repos.knowledge
    shop = repo.get_shop(shop_id)
    assert shop is not None
    overrides = {**shop.config_overrides, "auto_reply_enabled": True}
    repo.upsert_shop(shop.model_copy(update={"config_overrides": overrides}))
    return adapter


class Chat:
    """一个淘宝会话：每条消息前时钟走一分钟（避免被当成 1–2 秒内的连发合并，§5.3）。"""

    def __init__(
        self,
        service: AssistService,
        session_id: str,
        *,
        seller_id: str = SHOP_A,
        buyer: str = "BUYER",
        access: str = "B",
        robot: bool = False,
    ) -> None:
        self.service = service
        self.session_id = session_id
        self.seller_id = seller_id
        self.buyer = buyer
        self.access = access
        self.robot = robot
        self.n = 0

    def say(
        self, text: str, *, item_id: str | None = None, order_id: str | None = None
    ) -> EngineResult:
        self.n += 1
        clock = self.service.unit.clock
        clock.set(clock.now() + timedelta(minutes=1))
        raw = taobao_event(
            text,
            msg_id=f"{self.session_id}_{self.n}",
            session_id=self.session_id,
            buyer=self.buyer,
            item_id=item_id,
            order_id=order_id,
            seller_id=self.seller_id,
        )
        if self.robot:
            raw["reception"] = "robot"
        result = self.service.handle_event(self.access, raw)
        assert result is not None
        return result


# ================================================================ 检查


def generate_calls(llm: FakeLLMClient) -> list[LLMRequest]:
    return llm.calls_for(GENERATE)


def stored_texts(unit: Unit, shop_id: str) -> dict[str, str]:
    """一个店铺在单元里留下的全部文字（审计、decision_log、推荐记录、缺失知识、标注样本），
    按来源分开，便于检查个人信息和串店。"""
    repos = unit.repos
    audit = repos.audit
    events = audit.list_events(shop_id)
    decisions = audit.list_decisions(shop_id)
    records = repos.suggestions.list(shop_id)
    return {
        "audit": json.dumps([e.detail for e in events], ensure_ascii=False, default=str),
        "decision_log": json.dumps([d.answers for d in decisions], ensure_ascii=False, default=str),
        "suggestion.question_text": "\n".join(r.question_text or "" for r in records),
        "suggestion.text": "\n".join(r.text for r in records),
        "knowledge_gap": "\n".join(
            g.question_text or "" for g in repos.feedback.list_gaps(shop_id)
        ),
        "labeled_sample": "\n".join(s.text for s in repos.feedback.list_labels(shop_id)),
    }
