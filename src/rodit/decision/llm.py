"""平台云内大模型判断层 InCloudLLMProvider（方案 §6.4 "MVP 路由与质检"、D4 / D5）。

- 模型是境内已备案、部署在平台云内的大模型（经 core.ports.LLMClient 调用），可以处理买家数据。
- 提示词：system 说明判断层的职责（只判断、不写回答，§6.1）；user 给出 state（JSON）和每个问题的
  题型、说明、选项 / 等级，要求只输出 JSON：{问题名: {...}}。
  state 由 core.states 构造、已做个人信息最小化（§5.3），这里原样放进提示词、不再拼别的买家信息。
- 自一致性投票（§6.4）：同一请求采样 samples 次；多于一次时在线程池里并行调用
  （§12.1 路由 ≤ 1 秒），用 core.budget.submit_in_context 提交，
  保证 core.context.current_shop_id 在线程里可见（§12.4 成本归属）。
  解析与计票见 decision.voting。
- 多问题（§6.6 规则 5）：请求里带了追问 second_intent（core.questions.ROUTER_VOTE_QUESTIONS，
  由路由器在同一个请求里一并问，不多跑一轮）时，追问的票按采样和主要问题配对
  （voting.second_intent_votes）；追问没答、答不出来都按"无"计，不影响其他问题，也不算有效票不足。
- 选项概率（LLMResponse.logprobs）：不请求（want_logprobs 一律是 False）。判断层的请求都是
  JSON 模式，接口读的是输出里第一个有内容的 token，这时它是键名（如 intent）、不是选项，
  读不出选项概率；不少境内模型不支持 logprobs，带上这个参数整个请求可能被拒掉；接口照给的话，
  键名那个位置的候选里只要混进一个和选项同名的 token，归一化之后就是 1.0，一张票能盖过全部投票。
  响应里自己带了选项概率时（现在只有测试替身会带）照旧使用，只能记到一个单选题上：
  输出的第一个问题是单选题时记给它，否则只有一个单选题时记给那一个（选项名对得上才用）；
  路由请求里 intent 排第一，追问和它同一套选项、排在最后，只看票。
- 失败：单次调用抛 LLMError 的那一票作废；回答被 max_tokens 截断（finish_reason 是 length）
  或者文本为空的那一票也作废、不解析（截到一半的 JSON 不能当答案）。所有采样都作废，或某个问题的
  有效票数少于 min_votes，抛 DecisionError（调用方据此降级，§12.2）；消息里写明一共采样几次、
  没算上的票各是什么原因、各几票，如"3 次：超时 2、截断 1"（回答完整但这道题读不出有效票的
  算"读不出"）。原因只取类别和 LLMError 上的 kind / status / code，不带异常消息和模型输出的文字。
  返回前一律 validate_response。
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from rodit.core.budget import submit_in_context
from rodit.core.context import current_shop_id
from rodit.core.decision import (
    Answer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulQuestion,
    Question,
    ScoreQuestion,
    validate_response,
)
from rodit.core.errors import DecisionError, LLMError, LLMTimeout
from rodit.core.ports import LLMClient, LLMMessage, LLMRequest, LLMResponse
from rodit.core.questions import Q_INTENT, Q_SECOND_INTENT
from rodit.decision.voting import (
    NoulVote,
    Vote,
    aggregate_choice,
    aggregate_noul,
    aggregate_score,
    aggregate_second_intent,
    choice_distribution,
    parse_sample,
    second_intent_votes,
)

JudgePurpose = Literal["route", "qa"]

SYSTEM_PROMPT = (
    "你是电商平台客服系统里的“判断层”。你只做判断：阅读给定的状态（state）和问题，逐题作答。\n"
    "规则：\n"
    "1. 只根据 state 里的内容判断，不要编造；不要写给买家的回复，不要给出任何操作建议。\n"
    "2. state 里的对话已做个人信息脱敏（如 [手机号]、[地址]），不要猜测被遮掉的内容。\n"
    "3. 只输出一个 JSON 对象，键是问题名，不要输出 JSON 以外的任何文字、解释或代码块标记。\n"
    '4. 是否题回答 {"answer": "是" 或 "否", "p": 答案为“是”的概率（0 到 1）}；'
    '单选题回答 {"choice": 选项键}，只能从给出的选项键里选一个；'
    '打分题回答 {"score": 等级序号（整数）}。'
)

_TYPE_LABELS = {"noul": "是否题", "choice": "单选题", "score": "打分题"}

ERROR_KIND_LABELS: dict[str, str] = {
    "timeout": "超时",
    "rate_limit": "限流",
    "auth": "密钥无效或没有权限",
    "billing": "余额或额度用完",
    "bad_request": "请求被拒",
    "content_filter": "内容安全拦截",
    "server": "模型服务出错",
    "redirect": "重定向",
    "network": "连不上",
    "malformed": "响应格式不对",
    "config": "本地配置不对",
}
"""LLMError.kind（core.errors.LLMErrorKind）在报错消息里的说法。"""

# 一票没算上的其他原因（和上面的说法一起写进 DecisionError 的消息）
LOST_TRUNCATED = "截断"  # 回答被 max_tokens 截断（finish_reason 是 length），文本空不空都算
LOST_EMPTY = "空回答"  # 没被截断，但文本为空
LOST_UNREADABLE = "读不出"  # 回答完整，但这道题解析不出有效票

_CODE_CHARS = 64  # 错误代码写进消息时最多留这么长（客户端给的本来就是短字符串，这里再兜一层）


def _short(value: object) -> str:
    """kind / code 写进消息前：压成一行、截短；不是字符串或整数的（没填）给空串。"""
    if isinstance(value, bool) or not isinstance(value, str | int):
        return ""
    return " ".join(str(value).split())[:_CODE_CHARS]


def _error_cause(error: LLMError) -> str:
    """一次失败调用的原因，如"超时""限流（HTTP 429，rate_limit_exceeded）"。

    只读 LLMError 上的 kind / status / code（core.errors；没有这几项的替身按没填算），不读异常消息
    ——里面可能有模型服务返回的文字。都没填时只有异常类型。
    """
    kind = _short(getattr(error, "kind", None))
    code = _short(getattr(error, "code", None))
    status = getattr(error, "status", None)
    http = f"HTTP {status}" if isinstance(status, int) and not isinstance(status, bool) else ""
    if not kind and isinstance(error, LLMTimeout):
        kind = "timeout"
    if kind:
        label, detail = ERROR_KIND_LABELS.get(kind, kind), [http, code]
    elif http:
        label, detail = http, [code]
    else:
        label, detail = type(error).__name__, [code]
    detail = [d for d in detail if d]
    return f"{label}（{'，'.join(detail)}）" if detail else label


def _vote_counts(samples: int, lost: Iterable[str]) -> str:
    """一共采样几次，没算上的票各是什么原因、各几票，如"3 次：超时 2、截断 1"
    （多的在前，票数一样按原因名排，写法固定）。"""
    counts = sorted(Counter(lost).items(), key=lambda item: (-item[1], item[0]))
    if not counts:
        return f"{samples} 次"
    return f"{samples} 次：" + "、".join(f"{cause} {n}" for cause, n in counts)


def _question_block(key: str, question: Question) -> str:
    lines = [f"### {key}（{_TYPE_LABELS[question.type]}）", question.instructions]
    if isinstance(question, ChoiceQuestion):
        lines.append("选项（只能填下面的选项键）：")
        for option, definition in question.criteria.items():
            lines.append(f"- {option}：{definition}" if definition else f"- {option}")
        lines.append('回答格式：{"choice": "<选项键>"}')
    elif isinstance(question, ScoreQuestion):
        levels = "；".join(f"{i} = {label}" for i, label in enumerate(question.criteria))
        lines.append(f"等级：{levels}")
        lines.append(f'回答格式：{{"score": <0 到 {question.max_score} 的整数>}}')
    else:
        lines.append('回答格式：{"answer": "是" 或 "否", "p": <“是”的概率，0 到 1>}')
    return "\n".join(lines)


def _output_template(questions: dict[str, Question]) -> str:
    parts = []
    for key, question in questions.items():
        if isinstance(question, ChoiceQuestion):
            parts.append(f'"{key}": {{"choice": "<选项键>"}}')
        elif isinstance(question, ScoreQuestion):
            parts.append(f'"{key}": {{"score": <整数>}}')
        else:
            parts.append(f'"{key}": {{"answer": "<是/否>", "p": <0 到 1>}}')
    return "{" + ", ".join(parts) + "}"


def logprob_target(questions: dict[str, Question]) -> str | None:
    """响应里带了选项概率时记给哪个单选题（见模块说明）；没有能记的返回 None。"""
    first = next(iter(questions.items()), None)
    if first is not None and isinstance(first[1], ChoiceQuestion):
        return first[0]
    choices = [k for k, q in questions.items() if isinstance(q, ChoiceQuestion)]
    return choices[0] if len(choices) == 1 else None


def _paired_second_intent(questions: dict[str, Question]) -> bool:
    """请求里的 second_intent 是跟着 intent 的追问（按票配对计票）。"""
    return isinstance(questions.get(Q_SECOND_INTENT), ChoiceQuestion) and isinstance(
        questions.get(Q_INTENT), ChoiceQuestion
    )


def build_judge_messages(request: DecisionRequest) -> list[LLMMessage]:
    """判断层请求（§6.5 Jev 格式）→ 大模型提示词。state 原样序列化（已由 core.states 脱敏）。"""
    state_json = json.dumps(request.state, ensure_ascii=False, indent=2, default=str)
    blocks = "\n\n".join(_question_block(k, q) for k, q in request.questions.items())
    user = (
        f"## 状态（state）\n{state_json}\n\n"
        f"## 问题\n{blocks}\n\n"
        f"## 输出\n只输出一个 JSON 对象，格式如下：\n{_output_template(request.questions)}"
    )
    return [LLMMessage(role="system", content=SYSTEM_PROMPT), LLMMessage(role="user", content=user)]


def _json_objects(text: str) -> int:
    """回答里顶层的 JSON 对象有几个（能解析、不是空对象的才算）。和 rodit.eval.health 的
    json_object_count 同一口径：判断层不依赖评测包，这里自己数。要求只输出一个 JSON 对象，
    多于一个的回答不计票。"""
    decoder, count, i = json.JSONDecoder(), 0, 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return count
        try:
            value, end = decoder.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(value, dict) and value:
            count += 1
        i = end


class InCloudLLMProvider:
    """平台云内大模型 + 自一致性投票（实现 core.decision.DecisionProvider）。

    - samples：采样次数（路由默认 3，质检默认 1，见 DecisionConfig）；
    - temperature：采样温度（投票需要一定随机性；只采样一次时建议 0）；
    - purpose：写进 LLMRequest.purpose（成本统计、测试替身分流）；
    - timeout_s：单次调用超时（路由 ≤ 1 秒，§12.1），交给 LLMClient 执行；
    - min_votes：每个问题至少要有几张有效票，不够抛 DecisionError（默认 1）。
    """

    accepts_buyer_data = True

    def __init__(
        self,
        llm: LLMClient,
        *,
        samples: int,
        temperature: float,
        purpose: JudgePurpose,
        timeout_s: float | None = None,
        name: str = "incloud_llm",
        min_votes: int = 1,
        max_tokens: int | None = None,
    ) -> None:
        if samples < 1:
            raise ValueError("samples 至少为 1")
        if not 1 <= min_votes <= samples:
            raise ValueError("min_votes 必须在 1 到 samples 之间")
        self.llm = llm
        self.samples = samples
        self.temperature = temperature
        self.purpose: JudgePurpose = purpose
        self.timeout_s = timeout_s
        self.name = name
        self.min_votes = min_votes
        self.max_tokens = max_tokens

    # ---------------------------------------------------------------- 调用

    def _llm_request(self, request: DecisionRequest) -> LLMRequest:
        return LLMRequest(
            purpose=self.purpose,
            messages=build_judge_messages(request),
            temperature=self.temperature,
            max_tokens=self.max_tokens or max(128, 64 + 48 * len(request.questions)),
            json_mode=True,
            want_logprobs=False,  # JSON 模式下读不出选项概率，还可能被接口拒掉（见模块说明）
            timeout_s=self.timeout_s,
            shop_id=current_shop_id(),
        )

    def _call(self, llm_request: LLMRequest) -> LLMResponse | LLMError:
        try:
            return self.llm.complete(llm_request)
        except LLMError as e:
            return e

    def _sample(self, llm_request: LLMRequest) -> list[LLMResponse | LLMError]:
        if self.samples == 1:
            return [self._call(llm_request)]
        with ThreadPoolExecutor(max_workers=self.samples, thread_name_prefix="rodit-vote") as pool:
            # 每次提交都复制一份当前上下文（同一个 Context 不能被多个线程同时进入）；
            # 在引擎的 background_scope 里时还记进 BackgroundWork（取消订购等它们做完）
            futures = [
                submit_in_context(pool, self._call, llm_request) for _ in range(self.samples)
            ]
            return [f.result() for f in futures]

    # ---------------------------------------------------------------- 判断

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        results = self._sample(self._llm_request(request))
        errors = [r for r in results if isinstance(r, LLMError)]
        lost = [_error_cause(e) for e in errors]  # 没算上的票各是什么原因
        responses: list[LLMResponse] = []
        for r in results:
            if not isinstance(r, LLMResponse):
                continue
            if getattr(r, "finish_reason", None) == "length":
                lost.append(LOST_TRUNCATED)  # 截到一半的回答不解析：里面凑得出的片段不能当答案
            elif not r.text.strip():
                lost.append(LOST_EMPTY)
            elif _json_objects(r.text) > 1:
                # 先写了一份草稿、后面才是答案：分不清哪个算数，这一票不要（取第一个会按草稿判）
                lost.append(LOST_UNREADABLE)
            else:
                responses.append(r)
        if not responses:
            cause = errors[-1] if errors else None  # 都是截断、空回答时没有调用异常可挂
            raise DecisionError(
                f"大模型判断全部失败（{_vote_counts(self.samples, lost)}）"
            ) from cause
        parsed = [parse_sample(request.questions, r.text) for r in responses]
        target = logprob_target(request.questions)
        paired = _paired_second_intent(request.questions)
        answers: dict[str, Answer] = {}
        for key, question in request.questions.items():
            votes: list[Vote] = [p[key] for p in parsed if key in p]
            if paired and key == Q_SECOND_INTENT and isinstance(question, ChoiceQuestion):
                # 追问：按采样和主要问题配对，答不出来算"无"（不做有效票数检查）
                answers[key] = aggregate_second_intent(key, question, second_intent_votes(parsed))
            elif isinstance(question, ChoiceQuestion):
                dists = []
                if key == target:
                    dists = [
                        d
                        for r in responses
                        if (d := choice_distribution(question, r.logprobs)) is not None
                    ]
                self._check_quorum(key, max(len(votes), len(dists)), len(responses), lost)
                answers[key] = aggregate_choice(
                    key, question, [v for v in votes if isinstance(v, str)], dists
                )
            elif isinstance(question, NoulQuestion):
                self._check_quorum(key, len(votes), len(responses), lost)
                answers[key] = aggregate_noul(
                    key,
                    [v for v in votes if isinstance(v, NoulVote)],
                    use_reported_p=self.samples == 1,
                )
            else:
                self._check_quorum(key, len(votes), len(responses), lost)
                answers[key] = aggregate_score(
                    key, question, [float(v) for v in votes if isinstance(v, float)]
                )
        response = DecisionResponse(model=request.model, answers=answers)
        validate_response(request, response)
        return response

    def _check_quorum(self, key: str, valid: int, usable: int, lost: list[str]) -> None:
        """这道题的有效票不够就抛 DecisionError。usable 是拿去解析的回答数（其中没给出有效票的
        算"读不出"），lost 是没算上的票各自的原因。"""
        if valid >= self.min_votes:
            return
        counts = _vote_counts(self.samples, [*lost, *[LOST_UNREADABLE] * (usable - valid)])
        if valid == 0:
            raise DecisionError(f"问题 {key!r} 的所有投票都无效（{counts}）")
        raise DecisionError(
            f"问题 {key!r} 只有 {valid} 张有效票，少于 {self.min_votes}（{counts}）"
        )
