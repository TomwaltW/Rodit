"""离线演示 / 评测用的大模型与判断层替身（不联网、不需要密钥）。

- EvidenceEchoLLM：只处理生成请求（purpose="generate"）。从提示词的【依据】里挑和买家问题最相关的
  一条，按店铺称呼复述成一句回答，输出 §5.4 规定的 JSON。它只会复述依据，不会编造，
  所以能用来跑通"懂→找→写→判"全链路、做回归评测（§13.5），但回答质量远不如真实大模型。
- HeuristicQAJudge：质检（§5.5 第二道）的启发式判断：数字是否都在依据里、有没有依据外的时效承诺、
  有没有站外引流。只用于离线演示和评测，不能替代判断层（§6.4）。
  数字核对和硬规则同一套（qa.numbers）：买家自己说的数字只放行身高体重、尺码，价格、时长必须在依据里。

生产环境用 rodit.llm.OpenAICompatibleClient（境内已备案模型，D5）和 rodit.decision 的判断层。
"""

from __future__ import annotations

import json
import re
import unicodedata

from rodit.core.decision import (
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
    validate_response,
)
from rodit.core.errors import DecisionError, LLMError
from rodit.core.lexicon import find_off_platform_in_answer
from rodit.core.ports import LLMRequest, LLMResponse
from rodit.core.questions import (
    Q_ANSWERS_QUESTION,
    Q_GROUNDED,
    Q_PLATFORM_VIOLATION,
    Q_PROMISE_RISK,
    Q_TONE,
)
from rodit.qa.numbers import NumberSet, unsupported_numbers

_EVIDENCE_LINE = re.compile(r"^\[(?P<id>[^\]]+)\]\s*(?P<label>[^：]*)：(?P<content>.*)$")
_CURRENT = re.compile(r"（当前消息）：(?P<text>.*)$")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_PROMISE = re.compile(r"明天|今天|后天|今晚|马上到|当天到|一定|保证|包赔|全额赔")
_SALUTATION = re.compile(r"称呼(?:用|买家为)?[“\"「]?(亲|您)")
_CLAUSE = re.compile(r"[。；;！!？?\n]")


def _bigrams(text: str) -> set[str]:
    s = re.sub(r"\W+", "", unicodedata.normalize("NFKC", text or "").lower())
    return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s} - {""}


def _prompt_text(request: LLMRequest) -> str:
    return "\n".join(m.content for m in request.messages)


class EvidenceEchoLLM:
    """只复述依据的"大模型"（离线演示 / 评测用，见模块说明）。"""

    model = "offline-evidence-echo"

    def __init__(self, *, max_chars: int = 80) -> None:
        self.max_chars = max_chars
        self.calls: list[LLMRequest] = []

    @staticmethod
    def _best_clause(question: str, evidences: list[tuple[str, str]]) -> tuple[str, str]:
        """在所有依据的所有分句里，挑和买家问题字面最接近的一句（数字重合权重更高）。"""
        q = _bigrams(question)
        q_numbers = set(_NUMBER.findall(unicodedata.normalize("NFKC", question)))
        best: tuple[float, int, str, str] | None = None
        order = 0
        # 标题只是在复述商品名：有别的依据时不拿它来回答
        candidates = [e for e in evidences if not e[0].endswith(":title")] or evidences
        for eid, content in candidates:
            for clause in _CLAUSE.split(content):
                clause = clause.strip(" ，,：:")
                if not clause:
                    continue
                numbers = set(_NUMBER.findall(unicodedata.normalize("NFKC", clause)))
                score = len(q & _bigrams(clause)) + 2 * len(q_numbers & numbers)
                key = (score, -order, eid, clause)
                if best is None or key[:2] > best[:2]:
                    best = key
                order += 1
        assert best is not None
        return best[2], best[3]

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if request.purpose != "generate":
            raise LLMError("离线大模型只处理生成请求；路由和质检请配置规则 / 启发式判断")
        prompt = _prompt_text(request)
        evidences = []
        question = ""
        for line in prompt.splitlines():
            line = line.strip()
            m = _EVIDENCE_LINE.match(line)
            if m:
                evidences.append((m.group("id"), m.group("content").strip()))
                continue
            cur = _CURRENT.search(line)
            if cur:
                question = cur.group("text")
        salutation = "您" if "称呼买家为“您”" in prompt or "称呼用“您”" in prompt else "亲"
        m = _SALUTATION.search(prompt)
        if m:
            salutation = m.group(1)
        if not evidences:
            answer = {"answer": "这个我帮您确认一下", "evidence_ids": [], "need_clarify": False}
        else:
            eid, clause = self._best_clause(question, evidences)
            text = f"{salutation}，{clause}"[: self.max_chars]
            answer = {"answer": text, "evidence_ids": [eid], "need_clarify": False}
        text = json.dumps(answer, ensure_ascii=False)
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=len(prompt),
            output_tokens=len(text),
        )


class HeuristicQAJudge:
    """质检判断的启发式实现（离线演示 / 评测用，见模块说明）。

    实现 core.decision.DecisionProvider。"""

    name = "offline_heuristic"
    accepts_buyer_data = True  # 本地计算，不外传

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        state = request.state
        if "answer" not in state or "evidence" not in state:
            raise DecisionError("启发式判断只支持质检请求（core.states.build_qa_state）")
        answer = str(state.get("answer") or "")
        question = str(state.get("question") or "")
        evidence = "\n".join(str(e.get("content", "")) for e in state.get("evidence", []))
        corpus = evidence + "\n" + question
        # 数字核对和硬规则同一套（qa.numbers）：买家说过的身高体重、尺码可以复述，
        # 买家报的价格、时长不算依据（§5.5）
        support = NumberSet()
        support.add_text(evidence)
        support.add_buyer_text(question)
        unsupported = unsupported_numbers(answer, support)
        overlap = len(_bigrams(answer) & _bigrams(corpus)) / max(1, len(_bigrams(answer)))
        grounded = 0.2 if unsupported else (0.95 if overlap >= 0.3 or not answer else 0.6)
        promises = [p for p in _PROMISE.findall(answer) if p not in evidence]
        answers: dict[str, NoulAnswer | ScoreAnswer] = {
            Q_GROUNDED: NoulAnswer(noul=grounded),
            Q_ANSWERS_QUESTION: NoulAnswer(noul=0.9 if answer.strip() else 0.1),
            Q_PROMISE_RISK: NoulAnswer(noul=0.85 if promises else 0.05),
            Q_PLATFORM_VIOLATION: NoulAnswer(
                noul=0.9 if find_off_platform_in_answer(answer) else 0.02
            ),
            Q_TONE: ScoreAnswer(score=3.0),
        }
        response = DecisionResponse(
            model=request.model, answers={k: answers[k] for k in request.questions}
        )
        validate_response(request, response)
        return response
