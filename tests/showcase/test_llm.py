"""演示站的大模型替身（showcase.llm）：离线时只复述依据；§11.5 的"先写出没有依据的承诺"只在
第一次生成时出现；接真实大模型（--llm env）时交给它，回放历史时不调它。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from showcase_support import NOW

from rodit.core.enums import Intent, Platform
from rodit.core.ports import LLMMessage, LLMRequest
from rodit.showcase.app import create_showcase_app
from rodit.showcase.llm import (
    CONFIRM_PHRASE,
    RETRY_MARK,
    SCRIPTED_PROMISE,
    ShowcaseLLM,
    current_message,
    focus_clause,
    offline_answer,
)
from rodit.showcase.state import ShowcaseHub
from rodit.testing.fakes import FakeClock, FakeLLMClient


def _prompt(evidence: list[str], question: str, extra: str = "") -> str:
    return "\n".join(
        [
            "【依据】",
            *evidence,
            "【对话】",
            f"买家（当前消息）：{question}",
            "【要求】",
            "- 直接回答买家的问题。",
            extra,
        ]
    )


def _request(prompt: str, purpose: str = "generate") -> LLMRequest:
    return LLMRequest(
        purpose=purpose,  # type: ignore[arg-type]
        messages=[
            LLMMessage(role="system", content="你是小雅女装旗舰店的客服。称呼买家为“亲”"),
            LLMMessage(role="user", content=prompt),
        ],
    )


def _answer(llm: ShowcaseLLM, prompt: str) -> dict:
    return json.loads(llm.complete(_request(prompt)).text)


def test_current_message_and_focus_clause():
    assert current_message(_prompt([], "这个是纯棉的吗")) == "这个是纯棉的吗"
    assert current_message("没有对话") == ""
    q = "这个会透吗？能开发票吗？"
    assert focus_clause(q, Intent.PRODUCT_SPEC) == "这个会透吗"
    assert focus_clause(q, Intent.INVOICE_GIFT) == "能开发票吗"
    assert focus_clause(q, Intent.STOCK) is None


def test_offline_answer_only_repeats_evidence():
    evidence = [
        "[item:i_555:title] 商品属性 · 标题：法式碎花连衣裙",
        "[item:i_555:attr:面料] 商品属性 · 面料：面料：棉 95%、氨纶 5%",
        "[policy:ship_time] 店铺政策 · 发货时效：发货时效：现货付款后 48 小时内发货",
    ]
    text, ids = offline_answer(_prompt(evidence, "什么面料"), "什么面料")
    assert text == "亲，面料是棉 95%、氨纶 5%" and ids == ["item:i_555:attr:面料"]
    text, ids = offline_answer(_prompt(evidence, "多久发货"), "多久发货")
    assert "48 小时" in text and ids == ["policy:ship_time"]
    # 没有依据：不猜（§5.4 提示词第 1 条）
    assert offline_answer(_prompt([], "能便宜吗"), "能便宜吗") == (CONFIRM_PHRASE, [])
    # 只有标题时才拿标题
    text, ids = offline_answer(_prompt(evidence[:1], "这是什么"), "这是什么")
    assert ids == ["item:i_555:title"]


def test_offline_answer_prefers_a_matching_faq():
    evidence = [
        "[item:i_555:attr:厚薄] 商品属性 · 厚薄：厚薄：常规",
        "[faq:5] FAQ · 这款会透吗：问：这款会透吗 答：亲，这款是常规厚度，白色款在强光下会略透哦~",
    ]
    text, ids = offline_answer(_prompt(evidence, "这个会透吗"), "这个会透吗")
    assert ids == ["faq:5"] and text.startswith("亲，这款是常规厚度")
    assert "问：" not in text and "亲，亲" not in text


def test_salutation_follows_the_shop_setting():
    evidence = ["[policy:invoice] 店铺政策 · 发票：可以开具电子普通发票"]
    prompt = _prompt(evidence, "能开发票吗") + "\n称呼买家为“您”"
    text, _ = offline_answer(prompt, "能开发票吗")
    assert text.startswith("您，")


def test_arrival_question_gets_the_scripted_promise_only_first_time():
    llm = ShowcaseLLM()
    evidence = [
        "[tool:order:o_1002] 实时数据 · 订单：订单 o_1002：已发货",
        "[tool:logistics:o_1002] 实时数据 · 物流轨迹：中通快递 ZT7300000001：运输中，截至 09-28 "
        "10:30 已到达杭州转运中心（数据截至 09-28 10:35）",
    ]
    first = _answer(llm, _prompt(evidence, "催一下，明天能到吗？"))
    assert first["answer"] == SCRIPTED_PROMISE and llm.scripted == 1
    retry = _answer(llm, _prompt(evidence, "催一下，明天能到吗？", f"- {RETRY_MARK}：一定"))
    assert retry["evidence_ids"] == ["tool:logistics:o_1002"]
    assert "运输中，截至 09-28 10:30 已到达杭州转运中心" in retry["answer"]
    assert "明天" not in retry["answer"] and "ZT73" not in retry["answer"]
    assert llm.scripted == 1
    # 没有物流依据时说发货时效
    policy = ["[policy:ship_time] 店铺政策 · 发货时效：付款后 24 小时内发货"]
    retry = _answer(llm, _prompt(policy, "明天能到吗", f"- {RETRY_MARK}"))
    assert "24 小时" in retry["answer"] and retry["evidence_ids"] == ["policy:ship_time"]
    # 分条里的到货问题、不是问到货的问题都不写承诺
    split = _prompt(policy, "明天能到吗？能开发票吗", "- 本条只回答关于发票 / 赠品 / 保障的部分。")
    assert _answer(llm, split)["answer"] != SCRIPTED_PROMISE
    assert _answer(llm, _prompt(policy, "什么时候发货"))["answer"] != SCRIPTED_PROMISE
    assert llm.scripted == 1


def test_offline_refuses_route_and_qa_requests():
    llm = ShowcaseLLM()
    with pytest.raises(ValueError):
        llm.complete(_request("x", purpose="route"))


def test_real_llm_is_used_live_but_not_for_history():
    real = FakeLLMClient(
        handler=lambda req: {
            "answer": "亲，真实模型的回答",
            "evidence_ids": [],
            "need_clarify": False,
        },
        model="domestic-model",
    )
    llm = ShowcaseLLM(real)
    assert llm.offline is False and "domestic-model" in llm.model
    with llm.replay():
        replayed = _answer(
            llm, _prompt(["[policy:invoice] 店铺政策 · 发票：可以开发票"], "能开发票吗")
        )
    assert replayed["answer"] == "亲，可以开发票"
    assert real.calls == []
    live = _answer(llm, _prompt(["[policy:invoice] 店铺政策 · 发票：可以开发票"], "能开发票吗"))
    assert live["answer"] == "亲，真实模型的回答" and len(real.calls) == 1
    # §11.5 的模拟承诺照样出现（页面上标明是演示），重写交给真实模型
    assert _answer(llm, _prompt([], "明天能到吗"))["answer"] == SCRIPTED_PROMISE
    _answer(llm, _prompt([], "明天能到吗", f"- {RETRY_MARK}"))
    assert len(real.calls) == 2
    llm.complete(_request("路由", purpose="route"))  # 非生成请求原样交给真实模型
    assert real.calls[-1].purpose == "route"


def test_hub_with_a_real_llm_replays_history_offline():
    """--llm env：工作台的回答用真实模型，回放的几百条历史不调模型（不花额度）。"""
    real = FakeLLMClient(
        handler=lambda req: {
            "answer": "亲，尺码建议 M 哦",
            "evidence_ids": [],
            "need_clarify": False,
        }
    )
    hub = ShowcaseHub(clock=FakeClock(NOW), llm=real)
    assert hub.offline is False
    demo = hub.get(Platform.TAOBAO)
    assert demo.history.messages > 200 and real.calls == []
    client = TestClient(create_showcase_app(hub), base_url="http://127.0.0.1")
    view = client.post("/showcase/api/chat", json={"platform": "taobao", "scenario": "size"}).json()
    assert real.calls and all(c.purpose == "generate" for c in real.calls)
    assert view["result"]["suggestions"][0]["text"] == "亲，尺码建议 M 哦"
    page = client.get("/").text
    assert '"offline":false' in page


def _target(client: TestClient, key: str) -> dict:
    data = client.get("/showcase/api/overview", params={"platform": "taobao"}).json()
    return next(t for t in data["targets"] if t["key"] == key)


def test_real_llm_latency_counts_only_live_messages():
    """--llm env：回放的几百条历史用离线替身（毫秒级），不能算进"真实大模型实测"的 P95。
    还没有演示时的消息时显示"暂无真实大模型数据"；有了以后只按演示时的消息算。"""
    real = FakeLLMClient(
        handler=lambda req: {
            "answer": "亲，尺码建议 M 哦",
            "evidence_ids": [],
            "need_clarify": False,
        }
    )
    hub = ShowcaseHub(clock=FakeClock(NOW), llm=real)
    client = TestClient(create_showcase_app(hub), base_url="http://127.0.0.1")
    latency = _target(client, "latency")
    assert latency["value"] is None and latency["ok"] is None
    assert "暂无真实大模型数据" in latency["note"]
    board = client.get("/showcase/api/dashboard", params={"platform": "taobao"}).json()
    assert board["kpis"]["e2e_p95_ms"] is None and board["kpis"]["messages"] > 200
    assert board["latency"]["live_only"] is True and board["latency"]["messages"] == 0
    view = client.post("/showcase/api/chat", json={"platform": "taobao", "scenario": "size"}).json()
    e2e = view["trace"]["latency"]["since_received_ms"]
    latency = _target(client, "latency")
    assert latency["value"] == e2e and latency["ok"] is True
    assert "真实大模型实测" in latency["note"] and "1 条" in latency["note"]
    board = client.get("/showcase/api/dashboard", params={"platform": "taobao"}).json()
    assert board["latency"]["messages"] == 1 and board["latency"]["e2e"]["p95"] == e2e


def test_offline_latency_keeps_the_offline_caveat():
    hub = ShowcaseHub(clock=FakeClock(NOW))
    client = TestClient(create_showcase_app(hub), base_url="http://127.0.0.1")
    latency = _target(client, "latency")
    assert latency["value"] is not None and "离线替身" in latency["note"]
    assert "真实大模型实测" not in latency["note"]
    board = client.get("/showcase/api/dashboard", params={"platform": "taobao"}).json()
    assert board["latency"]["live_only"] is False
