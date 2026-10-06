"""演示站的大模型替身（只在演示站里用，生产和评测不用）。

ShowcaseLLM 是演示单元的生成客户端，只处理生成请求（purpose="generate"）：

- 离线（默认）：自己按【依据】写回答，原则和 rodit.app.offline.EvidenceEchoLLM 一样——只复述依据里的
  一句话，不编造；挑哪一句的办法更适合演示：FAQ 的问题和买家的话很像时直接用 FAQ 的回答，否则按
  字词、数字的重合挑最接近的一句（标题只是复述商品名，不拿来回答）；多问题分条（§6.6 规则 5）时
  只拿买家消息里属于这一条的那一句去挑（用路由的关键词表认）。
- `--llm env`：交给真实的境内大模型；回放历史数据期间（replay）仍用离线的写法，不为几百条历史消息
  调模型。
- §11.5 质检拦截的演示（离线和真实大模型都一样）：买家问"明天 / 今天能到吗"这类到货时间、而且是
  第一次生成（提示词里没有"上次回答未通过质检"）时，故意写出一句没有依据的承诺"亲，明天一定能到哦！"。
  这句话交给真实的质检（硬规则 + 判断层）去拦，引擎带着原因重新生成一次（§5.5）；页面上标明
  "演示：模拟大模型先写出没有依据的承诺"。离线时重新生成按物流轨迹依据写一句有依据的话，不承诺到货
  时间（§7.2）。

它只决定"写什么"，不碰路由、检索和质检：拦不拦、怎么判都是引擎里真实的组件做的。
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from rodit.core.enums import INTENT_LABELS, Intent
from rodit.core.ports import LLMClient, LLMRequest, LLMResponse
from rodit.decision.keywords import INTENT_KEYWORDS
from rodit.showcase.realchain import ReplaySwitch

SCRIPTED_PROMISE = "亲，明天一定能到哦！"
"""§11.5 第 3 步：大模型写出的、依据里没有的承诺（演示用）。"""

SIMULATED_NOTE = "演示：模拟大模型先写出没有依据的承诺"
"""页面上给这次生成加的说明（§11.5）。"""

RETRY_MARK = "上次回答未通过质检"
"""生成提示词里重新生成的标记（rodit.generation.prompt 的写法）。"""

CONFIRM_PHRASE = "这个我帮您确认一下"
"""没有依据时的回答（§5.4 提示词骨架第 1 条）。"""

OFFLINE_MODEL = "showcase-offline"

ARRIVAL_QUESTION = re.compile(r"(?:明天|今天|后天|今晚)[^，,。？?！!]{0,6}(?:到|送达|收到)")
"""买家问到货时间（§11.5"明天能到吗"）。"""

FAQ_MATCH = 0.5
"""FAQ 的问题和买家的话（字的二元组）重合到这个比例，直接用 FAQ 的回答。"""

_CURRENT = re.compile(r"^.*（当前消息）：(?P<text>.*)$", re.MULTILINE)
_EVIDENCE = re.compile(r"^\[(?P<id>[^\]]+)\]\s*(?P<label>[^：\n]*)：(?P<content>.*)$", re.MULTILINE)
_FOCUS = re.compile(r"本条只回答关于(?P<label>.+?)的部分")
_SALUTATION = re.compile(r"称呼买家为“(?P<s>亲|您)”")
_FAQ_CONTENT = re.compile(r"^问：(?P<q>.*?)\s*答：(?P<a>.*)$")
_LEADING_SALUTATION = re.compile(r"^(?:亲亲|亲爱的|亲|您好)[，,！!~～]*")
_DATA_AS_OF = re.compile(r"（数据截至[^）]*）")
_CLAUSES = re.compile(r"[^，,。？?！!；;\n]+")
_SENTENCES = re.compile(r"[^。；;！!？?\n]+")
_FIRST_CLAUSE = re.compile(r"^[^。；;（(]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SHORT_HEAD = re.compile(r"^(?P<head>[^\s\d：]{1,6})：(?P<rest>.+)$")
_PLAIN_HEADS = frozenset({"详情页", "说明"})
_STOP_CHARS = frozenset("的吗了呢吧啊呀么是有这那个款件条也都就还和与在我你您亲哦嘛不没能会要")
_LABEL_TO_INTENT = {label: intent for intent, label in INTENT_LABELS.items()}


def _prompt(request: LLMRequest) -> str:
    return "\n".join(m.content for m in request.messages)


def current_message(prompt: str) -> str:
    """提示词【对话】里买家的当前消息（已脱敏）。没有时返回空字符串。"""
    m = _CURRENT.search(prompt)
    return m.group("text").strip() if m else ""


def focus_clause(question: str, intent: Intent) -> str | None:
    """多问题分条时，买家消息里属于这一条（intent）的那一句：用路由的关键词表认（同一份词表，
    rodit.decision.keywords）。认不出来返回 None。"""
    rules = INTENT_KEYWORDS.get(intent, ())
    for clause in _CLAUSES.findall(question):
        clause = clause.strip()
        if clause and any(pattern.search(clause.lower()) for pattern, _ in rules):
            return clause
    return None


def _norm(text: str) -> str:
    return re.sub(r"\W+", "", unicodedata.normalize("NFKC", text or "").lower())


def _bigrams(text: str) -> set[str]:
    s = _norm(text)
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _chars(text: str) -> set[str]:
    return {c for c in _norm(text) if not c.isdigit() and c not in _STOP_CHARS}


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(unicodedata.normalize("NFKC", text or "")))


@dataclass(frozen=True)
class _Evidence:
    id: str
    content: str


def _evidences(prompt: str) -> list[_Evidence]:
    return [
        _Evidence(m.group("id"), _DATA_AS_OF.sub("", m.group("content")).strip())
        for m in _EVIDENCE.finditer(prompt)
    ]


def _phrase(clause: str) -> str:
    """ "面料：棉 95%" 这样的属性写法改成一句话（"面料是棉 95%"）。"""
    clause = clause.strip(" ，,：:")
    m = _SHORT_HEAD.match(clause)
    if m is None:
        return clause
    if m.group("head") in _PLAIN_HEADS:
        return m.group("rest").strip()
    return f"{m.group('head')}是{m.group('rest').strip()}"


def offline_answer(prompt: str, question: str) -> tuple[str, list[str]]:
    """离线写回答：只复述依据里的一句（见模块说明）。返回 (回答, 依据 ID)。"""
    m = _SALUTATION.search(prompt)
    salutation = m.group("s") if m else "亲"
    evidences = _evidences(prompt)
    if not evidences:
        return CONFIRM_PHRASE, []
    q_bigrams, q_chars, q_numbers = _bigrams(question), _chars(question), _numbers(question)
    best_faq: tuple[float, _Evidence, str] | None = None
    candidates: list[tuple[_Evidence, str]] = []
    for e in evidences:
        faq = _FAQ_CONTENT.match(e.content) if e.id.startswith("faq:") else None
        if faq is not None:
            answer = _LEADING_SALUTATION.sub("", faq.group("a").strip())
            fq = _bigrams(faq.group("q"))
            score = len(q_bigrams & fq) / len(fq) if fq else 0.0
            if score >= FAQ_MATCH and (best_faq is None or score > best_faq[0]):
                best_faq = (score, e, answer)
            candidates.append((e, answer))
        elif not e.id.endswith(":title"):
            candidates.append((e, e.content))
    if best_faq is not None:
        return f"{salutation}，{best_faq[2]}", [best_faq[1].id]
    if not candidates:  # 只有标题
        candidates = [(e, e.content) for e in evidences]
    best: tuple[int, _Evidence, str] | None = None
    for e, content in candidates:
        for sentence in _SENTENCES.findall(content):
            sentence = sentence.strip(" ，,：:")
            if not sentence:
                continue
            score = (
                3 * len(q_bigrams & _bigrams(sentence))
                + len(q_chars & _chars(sentence))
                + 2 * len(q_numbers & _numbers(sentence))
            )
            if best is None or score > best[0]:  # 同分取前面的（依据按优先级排好了）
                best = (score, e, sentence)
    if best is None:
        return CONFIRM_PHRASE, []
    return f"{salutation}，{_phrase(best[2])}", [best[1].id]


def _arrival_retry(prompt: str) -> tuple[str, list[str]] | None:
    """离线重新生成（§11.5 第 5 步）：有物流轨迹依据时说物流现状，没有时说发货时效；
    都不承诺到货时间（§7.2）。两种依据都没有时返回 None（按一般问题处理）。"""
    evidences = _evidences(prompt)
    for e in evidences:
        if not e.id.startswith("tool:logistics:"):
            continue
        # 物流依据写成"快递公司 运单号：状态……"，回答里只说状态
        content = e.content.split("：", 1)[1] if "：" in e.content else e.content
        first = _FIRST_CLAUSE.match(content.strip())
        if first is None:
            continue
        status = first.group(0).strip(" ，,")
        return f"亲，包裹目前{status}，我们会持续关注物流，有新进展会及时告诉您哦~", [e.id]
    for e in evidences:
        if e.id == "policy:ship_time":
            first = _FIRST_CLAUSE.match(e.content.strip())
            if first is not None:
                rule = _phrase(first.group(0))
                return f"亲，{rule}，发货后可以在订单页查看物流进度哦~", [e.id]
    return None


class ShowcaseLLM:
    """演示站的生成客户端（见模块说明）。inner 为空时离线。线程安全：分条并行生成时会被同时调用。

    switch 是回放开关（showcase.realchain.ReplaySwitch），和判断层共用一个：回放期间生成、
    路由、质检都离线。"""

    def __init__(
        self, inner: LLMClient | None = None, *, switch: ReplaySwitch | None = None
    ) -> None:
        self.inner = inner
        self.offline = inner is None
        self.model = OFFLINE_MODEL if inner is None else f"showcase+{getattr(inner, 'model', '')}"
        self.switch = switch or ReplaySwitch()
        self._lock = threading.Lock()
        self.scripted = 0  # 写过几次没有依据的承诺（页面据此加说明）

    @contextmanager
    def replay(self) -> Iterator[None]:
        """回放历史数据期间一律离线（只在组装演示单元时用，那时还没有别的请求）。"""
        with self.switch.replaying():
            yield

    def complete(self, request: LLMRequest) -> LLMResponse:
        offline = self.inner is None or self.switch.on
        if request.purpose != "generate":
            if offline:
                raise ValueError("演示站的离线替身只处理生成请求（路由、质检用规则和启发式判断）")
            assert self.inner is not None
            return self.inner.complete(request)
        prompt = _prompt(request)
        question = current_message(prompt)
        focus = _FOCUS.search(prompt)
        if focus is None and ARRIVAL_QUESTION.search(question):
            if RETRY_MARK not in prompt:
                with self._lock:
                    self.scripted += 1
                ids = [e.id for e in _evidences(prompt)][:1]
                return self._response(SCRIPTED_PROMISE, ids, prompt)
            if offline:
                grounded = _arrival_retry(prompt)
                if grounded is not None:
                    return self._response(*grounded, prompt)
        if not offline:
            assert self.inner is not None
            return self.inner.complete(request)
        if focus is not None:
            intent = _LABEL_TO_INTENT.get(focus.group("label"))
            question = (focus_clause(question, intent) if intent else None) or question
        return self._response(*offline_answer(prompt, question), prompt)

    def _response(self, text: str, evidence_ids: list[str], prompt: str) -> LLMResponse:
        body = json.dumps(
            {"answer": text, "evidence_ids": evidence_ids, "need_clarify": False},
            ensure_ascii=False,
        )
        return LLMResponse(
            text=body, model=self.model, input_tokens=len(prompt), output_tokens=len(body)
        )


__all__ = [
    "ARRIVAL_QUESTION",
    "CONFIRM_PHRASE",
    "OFFLINE_MODEL",
    "RETRY_MARK",
    "SCRIPTED_PROMISE",
    "SIMULATED_NOTE",
    "ShowcaseLLM",
    "current_message",
    "focus_clause",
    "offline_answer",
]
