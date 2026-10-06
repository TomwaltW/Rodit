"""演示站接"真实大模型"的整条链路（--llm env --judge llm）：真实的 OpenAICompatibleClient 走真实的
HTTP 调本机的 OpenAI 兼容替身（llm_stub.py），不联网。

核对：每个预设场景都能走通（路由大模型投票 vote_samples 票、质检大模型判断、生成真实调用）；
回放历史一次都不调模型；投票失败、超时、限流、格式不对按 §12.2 降级并在过程面板上写明原因；
时间预算放宽 / 严格；页面配置里的徽标；判断层的买家数据检查不变。
"""

from __future__ import annotations

import json
import re
import threading

import pytest
from llm_stub import StubLLM
from showcase_support import build_real_showcase, json_text

from rodit.app.offline import HeuristicQAJudge
from rodit.core.config import EngineConfig
from rodit.core.decision import DecisionRequest
from rodit.core.enums import Platform
from rodit.core.errors import ComplianceError
from rodit.core.questions import ROUTER_MODEL, ROUTER_VOTE_QUESTIONS
from rodit.decision import RuleProvider, build_llm_provider
from rodit.routing.policy import RoutingPolicy
from rodit.routing.router import DecisionRouter, asks_followups, vote_samples_of
from rodit.showcase.llm import RETRY_MARK, SCRIPTED_PROMISE, SIMULATED_NOTE, ShowcaseLLM
from rodit.showcase.realchain import (
    RELAXED_DECISION_TIMEOUT_S,
    RELAXED_GENERATION_TIMEOUT_S,
    RELAXED_MESSAGE_BUDGET_S,
    RELAXED_NOTE,
    REPLAY_BLOCKED,
    ObservedLLM,
    ReplaySwitch,
    SwitchableLLMProvider,
    chain_config,
)
from rodit.showcase.scenarios import scenarios_for
from rodit.testing.fakes import FakeClock, FakeLLMClient
from rodit.testing.fixtures import DEFAULT_NOW

ALL = [(p, s.id) for p in Platform for s in scenarios_for(p)]


def _config(**decision) -> EngineConfig:
    base = EngineConfig()
    return base.model_copy(update={"decision": base.decision.model_copy(update=decision)})


def _route_judge(view: dict) -> dict:
    judge = view["trace"]["understand"]["judge"]
    assert judge is not None
    return judge


def _question(judge: dict, key: str) -> dict:
    return next(q for q in judge["questions"] if q["key"] == key)


# ================================================================ 每个场景都走真实链路


@pytest.mark.parametrize(("platform", "scenario"), ALL)
def test_every_scenario_runs_through_the_real_chain(stub_env: StubLLM, platform, scenario):
    sc = build_real_showcase()
    routes_before = stub_env.count("route")
    view = sc.chat(platform, scenario)
    u = view["trace"]["understand"]
    # 路由：生产组装的大模型投票，一条消息 vote_samples（默认 3）个请求，没有降级
    assert u["provider"] == "incloud_llm" and not u["degraded"] and u["fallback"] is None
    assert stub_env.count("route") - routes_before == 3
    judge = _route_judge(view)
    assert judge["samples"] == 3 and judge["answered"] == 3 and not judge["errors"]
    assert sum(c["votes"] for c in _question(judge, "intent")["counts"]) == 3
    assert u["provider_label"] == "大模型投票（stub-chat）"
    # 调用记录：每次真实调用都在，路由 3 次
    llm = view["trace"]["llm"]
    by = {row["purpose"]: row for row in llm["by_purpose"]}
    assert by["route"]["calls"] == 3 and llm["failed"] == 0 and llm["input_tokens"] > 0
    # 用了大模型草稿的地方质检也是大模型判断的
    for part in view["trace"]["parts"]:
        for check in part["judge"]:
            if check["kind"] == "llm" and check["text"] != SCRIPTED_PROMISE:
                assert check["model_checked"] and check["provider"] == "incloud_llm"
                assert check["judge"]["samples"] == 1
    assert not view["result"]["degraded"]
    assert stub_env.api_key not in json_text(view)


def test_size_scenario_votes_and_real_generation(stub_env: StubLLM):
    """§11.1：路由三票都选尺码，生成请求真的发给了模型，推荐引用尺码建议，质检过了。"""
    sc = build_real_showcase()
    view = sc.chat("taobao", "size")
    judge = _route_judge(view)
    assert _question(judge, "intent")["counts"] == [
        {"option": "size_advice", "label": "尺码 / 选型", "votes": 3}
    ]
    assert _question(judge, "need_human")["no"] == 3
    assert _question(judge, "second_intent")["counts"][0]["option"] == "无"
    assert judge["probability_source"] == "votes"  # 替身默认不返回 logprobs
    assert stub_env.count("generate") >= 1
    [card] = view["result"]["suggestions"]
    assert card["kind"] == "llm" and "M" in card["text"]
    assert [e["label"] for e in card["evidence"]] == ["商品属性 · 尺码建议"]
    [part] = view["trace"]["parts"]
    [check] = part["judge"]
    assert check["passed"] and check["judge"]["answered"] == 1
    grounded = _question(check["judge"], "grounded")
    assert grounded["yes"] == 1 and grounded["p"] == [0.95]


def test_multi_question_split_uses_the_followup_votes(stub_env: StubLLM):
    """§6.6 规则 5：一票只能选一个主要类型，追问 second_intent 的票把第二个问题认出来。"""
    sc = build_real_showcase()
    view = sc.chat("douyin", "multi")
    u = view["trace"]["understand"]
    assert u["action"] == "split_multi" and u["rule"] == 5
    judge = _route_judge(view)
    assert _question(judge, "second_intent")["counts"] == [
        {"option": "invoice_gift", "label": "发票 / 赠品 / 保障", "votes": 3}
    ]
    assert {c["intent"] for c in view["result"]["suggestions"]} == {"product_spec", "invoice_gift"}


def test_scripted_first_draft_is_blocked_and_the_real_model_rewrites(stub_env: StubLLM):
    """§11.5：第一稿是演示脚本写的承诺（不发给模型），硬规则 + 大模型质检都拦；重写交给真实模型。"""
    sc = build_real_showcase()
    view = sc.chat("taobao", "qa_block")
    trace = view["trace"]
    assert trace["simulated_note"] == SIMULATED_NOTE and trace["simulated_rewrite_passed"] is True
    first, second = trace["parts"][0]["judge"]
    assert first["text"] == SCRIPTED_PROMISE and not first["passed"]
    assert first["model_checked"] and first["judge"] is not None
    assert any(s["key"] == "promise_risk" and not s["ok"] for s in first["scores"])
    assert second["passed"] and second["attempt"] == 2
    prompts = [b["messages"][-1]["content"] for b in stub_env.bodies("generate")]
    assert prompts and all(RETRY_MARK in p for p in prompts)  # 只有重写稿发给了模型
    assert all(SCRIPTED_PROMISE not in c["text"] for c in view["result"]["suggestions"])
    assert trace["simulated_rewrite"] == "passed" and second["kind"] == "llm"
    assert trace["fallback"]["generation"] is None


def _slow_generation(stub: StubLLM) -> EngineConfig:
    """生成每次都超时（限时 0.2 秒，替身 0.4 秒才答）。"""
    stub.delay("generate", 0.4)
    base = EngineConfig()
    return base.model_copy(
        update={"generation": base.generation.model_copy(update={"timeout_s": 0.2})}
    )


@pytest.mark.parametrize("platform", ["taobao", "pdd"])
def test_rewrite_that_times_out_is_not_called_a_passed_rewrite(stub_env: StubLLM, platform):
    """§11.5：重写稿的生成超时、退回模板（或什么都没有）：模板过了质检不算"重写后通过"，
    页面写明没有大模型重写稿、生成超时退回 FAQ / 模板（§12.2）。"""
    sc = build_real_showcase(config=_slow_generation(stub_env))
    view = sc.chat(platform, "qa_block")
    trace = view["trace"]
    assert trace["simulated_note"] == SIMULATED_NOTE
    assert trace["simulated_rewrite"] == "none" and trace["simulated_rewrite_passed"] is False
    assert all(c["kind"] != "llm" for c in view["result"]["suggestions"])
    note = trace["fallback"]["generation"]
    assert note is not None and "超时（限时 0.2 秒）" in note and "§12.2" in note
    by = {row["purpose"]: row for row in trace["llm"]["by_purpose"]}
    assert by["generate"]["timeouts"] == by["generate"]["failed"] > 0


def test_generation_timeout_is_marked_on_the_page(stub_env: StubLLM):
    """生成超时：引擎退回 FAQ / 模板（这里什么都没有），不记降级；过程面板要写明是生成超时。"""
    sc = build_real_showcase(judge="offline", config=_slow_generation(stub_env))
    view = sc.chat("taobao", "size")
    assert view["result"]["suggestions"] == []
    note = view["trace"]["fallback"]["generation"]
    assert note is not None and note.startswith("大模型生成有 ") and "超时（限时 0.2 秒）" in note


def test_offline_trace_has_no_generation_note():
    from showcase_support import build_showcase

    view = build_showcase().chat("taobao", "qa_block")
    assert view["trace"]["fallback"]["generation"] is None
    assert view["trace"]["simulated_rewrite"] == "passed"


# ================================================================ 回放历史不调模型


def test_history_replay_makes_zero_model_calls(stub_env: StubLLM):
    sc = build_real_showcase(history=True)
    demo = sc.hub.get(Platform.PDD)
    assert demo.history.messages > 200
    assert stub_env.count() == 0 and demo.replay_llm_calls == 0
    assert demo.observed is not None and demo.observed.blocked_during_replay == 0
    # 回放的判断日志记的是实际回答的实现（离线替身），不是大模型
    providers = {d.provider for d in demo.unit.repos.audit.list_decisions(demo.shop_id)}
    assert "incloud_llm" not in providers and "rules" in providers
    sc.chat("pdd", "size")
    assert stub_env.count("route") == 3  # 回放之后才真的调


def test_observed_client_refuses_calls_while_replaying():
    real = FakeLLMClient(handler=lambda r: "{}")
    switch = ReplaySwitch()
    observed = ObservedLLM(real, switch)
    llm = ShowcaseLLM(observed, switch=switch)
    with llm.replay():
        assert switch.on
        with pytest.raises(Exception, match=REPLAY_BLOCKED):
            from rodit.core.ports import LLMMessage, LLMRequest

            observed.complete(
                LLMRequest(purpose="route", messages=[LLMMessage(role="user", content="x")])
            )
    assert not switch.on and real.calls == [] and observed.blocked_during_replay == 1


# ================================================================ 投票、概率


def test_vote_samples_from_config_are_the_requests_per_route(stub_env: StubLLM):
    sc = build_real_showcase(config=_config(vote_samples=5))
    view = sc.chat("taobao", "invoice")
    assert stub_env.count("route") == 5
    judge = _route_judge(view)
    assert judge["samples"] == 5 and judge["answered"] == 5 and judge["min_votes"] == 3


def test_option_logprobs_are_used_when_the_api_gives_them(monkeypatch):
    with StubLLM(logprobs="option") as stub:
        monkeypatch.setenv("RODIT_LLM_BASE_URL", stub.base_url)
        monkeypatch.setenv("RODIT_LLM_MODEL", stub.model)
        monkeypatch.setenv("RODIT_LLM_API_KEY", stub.api_key)
        sc = build_real_showcase()
        view = sc.chat("taobao", "size")
    judge = _route_judge(view)
    assert judge["probability_source"] == "logprobs"
    top = view["trace"]["understand"]["top_intents"]
    # 概率取接口给的选项概率（-0.2 vs -1.8 → 约 0.83），不是三票全中的 1.0
    assert top[0]["intent"] == "size_advice" and 0.8 < top[0]["p"] < 0.86


# ================================================================ 降级（§12.2）


def test_route_timeout_falls_back_to_keywords_and_says_so(stub_env: StubLLM):
    stub_env.delay("route", 0.5)
    sc = build_real_showcase(config=_config(timeout_s=0.2))
    view = sc.chat("taobao", "size")
    u = view["trace"]["understand"]
    assert u["degraded"] and u["provider"] == "rules"
    assert u["provider_label"] == "关键词规则（大模型判断层不可用时的兜底）"
    assert u["fallback"] == (
        "大模型 3 次调用全部超时（每次限时 0.2 秒） → 这条消息改用关键词规则路由（降级，§12.2）"
    )
    judge = _route_judge(view)
    assert judge["failed"] and judge["answered"] == 0
    assert judge["errors"] == [{"label": "超时（限时 0.2 秒）", "count": 3}]
    by = {row["purpose"]: row for row in view["trace"]["llm"]["by_purpose"]}
    assert by["route"]["timeouts"] == 3
    assert view["result"]["degraded"]  # 页面上标为降级
    assert view["result"]["suggestions"]  # 尺码不是高风险类型：关键词路由照样出推荐


def test_relaxed_latency_lets_slow_routing_finish(stub_env: StubLLM):
    stub_env.delay("route", 0.5)
    sc = build_real_showcase(latency="relaxed", config=_config(timeout_s=0.2))
    view = sc.chat("taobao", "size")
    u = view["trace"]["understand"]
    assert not u["degraded"] and u["provider"] == "incloud_llm"
    assert _route_judge(view)["timeout_s"] == RELAXED_DECISION_TIMEOUT_S
    lat = view["trace"]["latency"]
    assert lat["relaxed"] is True and lat["budget_s"] == RELAXED_MESSAGE_BUDGET_S


def test_one_failed_vote_still_has_a_quorum(stub_env: StubLLM):
    stub_env.fail("route", "500", times=1)
    sc = build_real_showcase()
    view = sc.chat("taobao", "size")
    judge = _route_judge(view)
    assert not view["trace"]["understand"]["degraded"]
    assert judge["answered"] == 2 and judge["errors"] == [
        {"label": "服务端错误（HTTP 500）", "count": 1}
    ]


def test_too_few_votes_fall_back(stub_env: StubLLM):
    stub_env.fail("route", "429", times=2)
    sc = build_real_showcase()
    view = sc.chat("taobao", "size")
    u = view["trace"]["understand"]
    assert u["degraded"] and u["provider"] == "rules"
    assert u["fallback"].startswith(
        "有效票不够：1/3 次有回答（另有 限流或额度用完（HTTP 429） ×2）"
    )


def test_unparseable_votes_fall_back(stub_env: StubLLM):
    stub_env.fail("route", "not_json")
    sc = build_real_showcase()
    view = sc.chat("taobao", "invoice")
    u = view["trace"]["understand"]
    assert u["degraded"] and "解析不出" in u["fallback"]
    assert view["result"]["suggestions"][0]["kind"] == "faq"  # FAQ 不靠判断层


def test_late_qa_is_not_recorded_on_the_next_message(stub_env: StubLLM):
    """质检到截止时间还没做完（引擎不等了，线程在后台跑完）：它的结果和判断不能记到下一条消息的
    过程面板上。下一条自己的质检也没做完，所以下一条的"判"里什么都没有。"""
    stub_env.delay("qa", 0.85)
    config = EngineConfig.model_validate(
        {
            "decision": {"timeout_s": 3.0},
            "latency": {"message_budget_s": 0.6, "min_llm_remaining_s": 0.1},
        }
    )
    sc = build_real_showcase(config=config)
    first = sc.chat("taobao", "size")
    second = sc.chat("taobao", "size")
    for view in (first, second):
        assert [c for p in view["trace"]["parts"] for c in p["judge"]] == []
    # 第一条的质检确实在后台跑完了（替身答完才记请求），只是没记到第二条上
    assert stub_env.count("qa") >= 1


def test_high_risk_question_gets_no_answer_when_routing_degraded(stub_env: StubLLM):
    """§12.2：判断层不可用 → 关键词路由，高风险类型一律不推荐。"""
    stub_env.fail("route", "401")
    sc = build_real_showcase()
    view = sc.chat("taobao", "price")
    u = view["trace"]["understand"]
    assert u["degraded"] and u["high_risk"]
    assert "密钥无效（HTTP 401）" in u["fallback"]
    assert view["result"]["suggestions"] == []


def test_qa_unavailable_means_no_model_answer(stub_env: StubLLM):
    """§12.2：质检判断层不可用时不推荐未经质检的生成内容。"""
    stub_env.fail("qa", "500")
    sc = build_real_showcase()
    view = sc.chat("taobao", "size")
    [part] = view["trace"]["parts"]
    check = part["judge"][0]
    assert not check["passed"] and check["degraded"]
    assert check["unavailable"] == (
        "大模型 1 次调用全部失败（服务端错误（HTTP 500）） → 质检不可用：大模型草稿不推荐（§12.2）"
    )
    assert all(c["kind"] != "llm" for c in view["result"]["suggestions"])
    assert view["trace"]["qa_summary"]["unavailable"] >= 1


@pytest.mark.parametrize("kind", ["429", "content_filter", "malformed", "timeout"])
def test_generation_failure_falls_back_to_templates(stub_env: StubLLM, kind: str):
    stub_env.timeout_sleep_s = 1.0
    stub_env.fail("generate", kind)
    config = EngineConfig()
    config = config.model_copy(
        update={"generation": config.generation.model_copy(update={"timeout_s": 0.3})}
    )
    sc = build_real_showcase(config=config)
    view = sc.chat("taobao", "size")
    assert all(c["kind"] != "llm" for c in view["result"]["suggestions"])
    by = {row["purpose"]: row for row in view["trace"]["llm"]["by_purpose"]}
    assert by["generate"]["failed"] >= 1


# ================================================================ 判断层的买家数据检查、线程安全


class _NoBuyerData:
    name = "external"
    accepts_buyer_data = False

    def decide(self, request):  # pragma: no cover - 不会被调到
        raise AssertionError


def test_switchable_provider_keeps_buyer_data_semantics():
    real = build_llm_provider(
        EngineConfig().decision, purpose="route", llm=FakeLLMClient(handler=lambda r: "{}")
    )
    ok = SwitchableLLMProvider(real, RuleProvider(), ReplaySwitch())
    assert ok.accepts_buyer_data is True
    bad = SwitchableLLMProvider(real, _NoBuyerData(), ReplaySwitch())
    assert bad.accepts_buyer_data is False
    with pytest.raises(ComplianceError):
        DecisionRouter(bad, RuleProvider(), RoutingPolicy(EngineConfig().routing), None, None)  # type: ignore[arg-type]
    # 路由器仍按"投票制大模型判断层"对待它（追问 second_intent、按票数粒度比较阈值）
    assert asks_followups(ok) and vote_samples_of(ok) == 3


def test_switchable_provider_is_offline_while_replaying():
    real_llm = FakeLLMClient(handler=lambda r: "{}")
    real = build_llm_provider(EngineConfig().decision, purpose="route", llm=real_llm)
    switch = ReplaySwitch()
    provider = SwitchableLLMProvider(real, RuleProvider(), switch)
    request = DecisionRequest(
        model=ROUTER_MODEL,
        state={"platform": "taobao", "messages": [{"role": "buyer", "text": "能开发票吗"}]},
        questions=ROUTER_VOTE_QUESTIONS,
    )
    with switch.replaying():
        assert provider.name == "rules" and provider.samples == 1
        response = provider.decide(request)
        assert vote_samples_of(provider) is None
    assert response.choice("intent").choice == "invoice_gift"
    assert response.choice("second_intent").choice == "无"
    assert real_llm.calls == []
    assert provider.name == "incloud_llm" and provider.samples == 3


def test_replay_switch_is_thread_safe():
    switch = ReplaySwitch()
    seen: list[bool] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        for _ in range(200):
            with switch.replaying():
                seen.append(switch.on)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(seen) and len(seen) == 1600 and not switch.on


def test_qa_judge_offline_counterpart_is_the_heuristic(stub_env: StubLLM):
    sc = build_real_showcase()
    demo = sc.hub.get(Platform.TAOBAO)
    gate_provider = demo.unit.gate._provider
    assert isinstance(gate_provider, SwitchableLLMProvider)
    assert isinstance(gate_provider.offline, HeuristicQAJudge)
    assert gate_provider.purpose == "qa" and gate_provider.samples == 1


# ================================================================ 时间预算、页面上的说明


def test_relaxed_profile_values():
    relaxed = chain_config(None, judge="llm", latency="relaxed")
    assert relaxed.decision.timeout_s == RELAXED_DECISION_TIMEOUT_S
    assert relaxed.generation.timeout_s == RELAXED_GENERATION_TIMEOUT_S
    assert relaxed.latency.message_budget_s == RELAXED_MESSAGE_BUDGET_S
    strict = chain_config(None, judge="llm", latency="strict")
    assert strict.decision.timeout_s == 1.0 and strict.latency.message_budget_s == 4.5
    assert strict.decision.provider.value == "incloud_llm"
    assert chain_config(None, judge="offline", latency="strict").decision.provider.value == "rules"


def _page_config(page: str) -> dict:
    found = re.search(r'<script id="showcase-config" type="application/json">(.*?)</script>', page)
    assert found is not None
    return json.loads(found.group(1))


def test_page_badge_and_footer(stub_env: StubLLM):
    sc = build_real_showcase(latency="relaxed")
    page = sc.client.get("/").text
    config = _page_config(page)
    chain = config["chain"]
    host = stub_env.base_url.removesuffix("/v1")
    assert chain["badge"] == f"真实大模型：stub-chat（{host}）· 判断层：大模型投票 · 时间预算：放宽"
    assert chain["host"] == host  # 只有协议和主机，没有路径
    assert chain["relaxed_note"] == RELAXED_NOTE and chain["route_timeout_s"] == 8.0
    assert config["offline"] is False
    assert (
        "回答由真实大模型 stub-chat 生成" in config["footer"] and RELAXED_NOTE in config["footer"]
    )
    assert stub_env.api_key not in page
    strict = build_real_showcase(judge="offline")
    chain = _page_config(strict.client.get("/").text)["chain"]
    assert chain["badge"].endswith("判断层：离线规则 · 时间预算：严格")
    assert chain["relaxed_note"] is None


def test_dashboard_note_says_what_is_real(stub_env: StubLLM):
    sc = build_real_showcase()
    sc.chat("taobao", "size")
    note = sc.dashboard("taobao")["latency"]["note"]
    assert "真实大模型实测" in note and "路由、质检用大模型投票" in note
    offline_judges = build_real_showcase(judge="offline")
    offline_judges.chat("taobao", "size")
    note = offline_judges.dashboard("taobao")["latency"]["note"]
    assert "路由、质检仍是离线替身" in note


def test_judge_offline_only_generation_is_real(stub_env: StubLLM):
    sc = build_real_showcase(judge="offline")
    view = sc.chat("taobao", "size")
    u = view["trace"]["understand"]
    assert u["provider"] == "rules" and u["judge"] is None
    assert u["provider_label"] == "关键词规则（离线替身）"
    assert stub_env.count("route") == 0 and stub_env.count("qa") == 0
    assert stub_env.count("generate") >= 1


def test_judge_llm_requires_a_real_llm():
    from rodit.showcase.state import ShowcaseHub

    with pytest.raises(ValueError, match="需要真实大模型"):
        ShowcaseHub(clock=FakeClock(DEFAULT_NOW), judge="llm")


def test_configured_rules_routing_with_llm_qa(stub_env: StubLLM):
    """配置的 decision.provider 是 rules 时（--judge llm）：路由用关键词规则（不是降级、也不调
    模型），质检照生产仍用大模型（§6.4），徽标和过程面板都照实写。"""
    sc = build_real_showcase(config=_config(provider="rules"))
    view = sc.chat("taobao", "size")
    u = view["trace"]["understand"]
    assert u["provider"] == "rules" and not u["degraded"] and u["judge"] is None
    assert u["provider_label"] == "关键词规则（配置的 decision.provider = rules）"
    assert stub_env.count("route") == 0 and stub_env.count("qa") >= 1
    [check] = view["trace"]["parts"][0]["judge"]
    assert check["provider"] == "incloud_llm" and check["judge"]["samples"] == 1
    chain = sc.hub.chain
    assert chain.judge_label == "关键词路由 + 大模型质检" and not chain.route_by_llm
    assert "判断层：关键词路由 + 大模型质检" in chain.badge()
    assert "路由用关键词规则、质检用大模型" in chain.footer()
