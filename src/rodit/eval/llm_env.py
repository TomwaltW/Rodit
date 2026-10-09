"""用真实大模型跑评测时的环境配置（`python -m rodit.eval --llm env`）。

大模型：rodit.llm.OpenAICompatibleClient.from_env——RODIT_LLM_BASE_URL、RODIT_LLM_MODEL（必填）、
RODIT_LLM_TIMEOUT_S（可选）；密钥 RODIT_LLM_API_KEY 由客户端在每次调用时自己读。
这里只检查密钥"有没有设置"，不读出、不打印、不写进报告；报告里的接口地址只留主机名
（URL 里的用户名密码、路径、查询参数都不要）。密钥只放本地 .env（已 gitignore）或
云端环境的密钥设置，不进仓库（CLAUDE.md）。

单价（§14，不是密钥）：命令行 --price-input / --price-output 优先，其次环境变量
RODIT_LLM_PRICE_INPUT_PER_1K / RODIT_LLM_PRICE_OUTPUT_PER_1K（每 1000 token 的价格）；
两个都没有时只报 token，只给了一个视为配置错误。

跑之前先做连通性检查（preflight）：把这一轮评测要发的几种请求按原样各试一次，整轮每次调用都会
失败的问题在这里用几次调用就暴露出来，不白跑一整轮、也不让整轮结果全是降级：
- 路由：按 config.decision 组装的判断层回答真实的路由问题——投票次数、并发、温度、max_tokens、
  decision.timeout_s 都和评测时一样；路由不用大模型判断层（decision.provider 是 classifier / rules）
  时不查，摘要里说明；
- 质检：按 config.decision 组装的质检判断层回答真实的质检问题；
- 生成：真实的生成提示词（generation.temperature / max_tokens / timeout_s），输出用生成环节自己的
  解析检查。
请求里的买家消息、商品、依据来自合成演示世界（rodit.testing.fixtures，全部手造），大小和评测时相当，
没有个人信息。一共 vote_samples + qa_vote_samples + 1 次调用（默认 5 次），不重试，不计入评测用量。
哪一次调用失败、回了空文本、被 max_tokens 截断，或者回答用该环节自己的解析读不出，都算没过：抛
PreflightFailed，说明里有是哪种请求、失败原因（HTTP 状态码和错误代码读 LLMError 的 status / kind /
code，不解析消息）、实测耗时和配置的超时、每种原因的下一步。通过时返回摘要：每种请求的调用次数、
耗时、超时（耗时超过超时的 70% 时提醒），和网络路径（via_proxy：请求是不是经环境里配置的代理
发出去的，§12.1 的耗时要看这条路径）。python -m rodit.eval --skip-preflight 跳过检查（只在调试时
用）；python -m rodit.eval.compare 只调路由判断层，用 run_preflight(route_request=…) 只查路由。
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from rodit.core import evidence as ev
from rodit.core.config import DecisionConfig, EngineConfig
from rodit.core.decision import DecisionProvider, DecisionRequest, validate_response
from rodit.core.enums import DraftKind, Intent, ProviderName, RouteAction
from rodit.core.errors import LLM_ERROR_KINDS, DecisionError, LLMError, LLMTimeout
from rodit.core.models import Draft, RouteDecision
from rodit.core.pii import mask_pii
from rodit.core.ports import LLMClient, LLMRequest, LLMResponse
from rodit.core.questions import QA_MODEL, QA_QUESTIONS
from rodit.core.states import build_qa_state
from rodit.decision import build_decision_provider, build_llm_provider, parse_sample
from rodit.eval.perf import Pricing
from rodit.generation import build_generation_request, parse_generation_output
from rodit.llm import OpenAICompatibleClient, urllib_transport
from rodit.qa.masking import mask_text, masked_context, masked_evidence
from rodit.routing.router import DecisionRouter

ENV_PREFIX = "RODIT_LLM_"
PRICE_INPUT_ENV = "RODIT_LLM_PRICE_INPUT_PER_1K"
PRICE_OUTPUT_ENV = "RODIT_LLM_PRICE_OUTPUT_PER_1K"
KEY_HINT = "密钥只放本地 .env 或环境的密钥设置，不要提交进仓库"
DEFAULT_PORTS = {"https": 443, "http": 80}

# 连通性检查用的合成会话：演示世界里店铺 A 的一条售前尺码咨询（方案 §11.1 的场景），没有个人信息。
# 评测集里没有这句话：模型服务那边的日志、测试替身都能把检查请求和评测请求分开。
PREFLIGHT_QUESTION = "身高 168、体重 118 斤，这条裙子拍 M 还是 L 合适？"
PREFLIGHT_ITEM_ID = "i_555"
PREFLIGHT_ANSWER = (
    "亲，按您 168cm、118 斤的身材建议拍 L 码哦，这款版型偏小，可以再对照尺码表确认一下呢"
)

SLOW_SHARE = 0.7  # 耗时超过单次超时的这个比例就提醒：整轮评测里这种请求会间歇性超时
ERROR_TEXT_CHARS = 200
ERROR_CODE_CHARS = 64
ROUTE, QA, GENERATE = "route", "qa", "generate"
_LABELS = {ROUTE: "路由", QA: "质检", GENERATE: "生成"}
_KNOBS = {ROUTE: "decision.timeout_s", QA: "decision.timeout_s", GENERATE: "generation.timeout_s"}
_SLOW_EFFECTS = {
    ROUTE: "超时的那一票作废，票不够就退回关键词规则",
    QA: "质检不可用，模型草稿不放行",
    GENERATE: "这条草稿作废，退回 FAQ / 模板",
}
# 没过的原因：LLMErrorKind 里的各类，加上这里自己判的四种
UNKNOWN, EMPTY, TRUNCATED, UNPARSEABLE = "error", "empty", "truncated", "unparseable"
_CAUSE_LABELS = {
    "timeout": "超时",
    "rate_limit": "被限流",
    "auth": "密钥无效或没有权限",
    "billing": "余额或额度用完",
    "bad_request": "请求被拒",
    "content_filter": "被内容安全拦截",
    "server": "模型服务出错",
    "redirect": "返回了重定向",
    "network": "连不上",
    "malformed": "响应读不出来",
    "config": "本地配置不对",
    UNKNOWN: "调用失败",
}
_THINKING_HINT = (
    "在 --config 的 llm.extra_body 里关掉思考模式"
    "（thinking: {type: disabled} 或 enable_thinking: false，例子见 config/rodit.yaml）"
)
_NOT_EVIDENCE = "这样的配置指纹和线上不同，结论不能当上线依据"
_MAYBE_OCCASIONAL = (
    "  - 同一种请求里有的调用过了、有的没过：也可能是偶发的，可以再试一次；反复出现的话，"
    "整轮评测里这种请求也会按这个比例失败。"
)


class PreflightFailed(RuntimeError):
    """连通性检查没过。消息是给操作的人看的完整说明（可以多行），不含密钥。"""


def client_from_env(*, prefix: str = ENV_PREFIX, **kwargs: Any) -> OpenAICompatibleClient:
    """按环境变量构造客户端（kwargs 原样交给 OpenAICompatibleClient.from_env，测试用来注入
    transport）。地址或模型没设、密钥没设时抛 RuntimeError（信息里只有变量名）。"""
    client = OpenAICompatibleClient.from_env(prefix=prefix, **kwargs)
    if not os.environ.get(client.api_key_env, "").strip():
        raise RuntimeError(
            f"环境变量 {client.api_key_env} 未设置：没有大模型 API key（{KEY_HINT}）"
        )
    return client


def describe_endpoint(client: object) -> str | None:
    """报告里显示的接口地址：只留协议和主机（端口），没有 base_url 的客户端返回 None。

    端口是协议的默认端口（https 的 443、http 的 80）时不写：https://host 和 https://host:443
    是同一个接口，评测记下的和线上算出来的要一样（eval.fingerprint.endpoint_of 同一口径）。"""
    base_url = getattr(client, "base_url", None)
    if not isinstance(base_url, str) or not base_url:
        return None
    parts = urlsplit(base_url)
    if not parts.hostname:
        return None
    try:
        port = parts.port
    except ValueError:  # 端口不是数字：不显示
        port = None
    shown = f":{port}" if port and port != DEFAULT_PORTS.get(parts.scheme) else ""
    return f"{parts.scheme}://{parts.hostname}{shown}"


def describe_model(client: object) -> str | None:
    model = getattr(client, "model", None)
    return model if isinstance(model, str) and model else None


def via_proxy(client: object) -> bool | None:
    """这个客户端的请求是不是经环境里配置的代理发出去的：是 True、直连 False；不知道
    （不是 HTTP 客户端、注入了 transport）是 None。只回答是不是，不返回代理地址
    （地址里可能有用户名密码）。

    按 urllib 自己的规则判断（默认传输层每次调用都按当时的环境新建 opener）：这个协议配了代理
    （urllib.request.getproxies：环境变量，macOS 上没有环境变量时读系统设置），而且主机不在
    免代理名单里（proxy_bypass：NO_PROXY，小写的 no_proxy 优先）。说的是调用这一刻的环境。"""
    if not isinstance(client, OpenAICompatibleClient):
        return None
    if getattr(client, "_transport", None) is not urllib_transport:
        return None  # 注入的 transport 走不走代理这里看不出来
    try:
        request = urllib.request.Request(client.base_url)  # 只按 urllib 的规则拆地址，不发请求
    except ValueError:
        return None
    if not request.host:
        return None
    if request.type not in urllib.request.getproxies():
        return False
    return not urllib.request.proxy_bypass(request.host)


# ================================================================ 连通性检查


@dataclass(frozen=True)
class _Call:
    """一次调用：请求、耗时（秒）、结果（response 和 error 有且只有一个）。"""

    request: LLMRequest
    seconds: float
    response: LLMResponse | None = None
    error: LLMError | None = None


@dataclass(frozen=True)
class _Problem:
    """没过的原因。cause 决定给哪条"下一步"，text 是给人看的说明。"""

    cause: str
    text: str


@dataclass
class _Shape:
    """一种请求的检查结果。

    problems 是逐次调用的问题；verdict 是每次调用都正常、这个环节自己却得不出结果时的说明；
    skipped 不为空表示没有检查（说明原因）。"""

    name: str
    calls: list[_Call] = field(default_factory=list)
    problems: list[_Problem] = field(default_factory=list)
    verdict: _Problem | None = None
    skipped: str | None = None

    @property
    def failed(self) -> bool:
        return bool(self.problems) or self.verdict is not None

    @property
    def timeout_s(self) -> float | None:
        """这种请求自带的单次超时（请求里没有设时为 None）。"""
        timeouts = (c.request.timeout_s for c in self.calls)
        return next((t for t in timeouts if t is not None), None)


class _Probe:
    """包在大模型客户端外面（实现 core.ports.LLMClient）：原样转发，记下每次调用的请求、耗时和
    结果，判断层把单票的失败吞掉之后这里还看得到。线程安全（路由投票并发调用同一个客户端）。
    只记 LLMError；别的异常是程序错误，照常往外抛。"""

    def __init__(self, inner: LLMClient, timer: Callable[[], float]) -> None:
        self._inner = inner
        self._timer = timer
        self._lock = threading.Lock()
        self._calls: list[_Call] = []

    @property
    def model(self) -> str:
        return str(getattr(self._inner, "model", "") or "")

    def __repr__(self) -> str:
        return f"_Probe(inner={self._inner!r})"

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = self._timer()
        try:
            response = self._inner.complete(request)
        except LLMError as e:
            self._add(_Call(request, self._timer() - started, error=e))
            raise
        self._add(_Call(request, self._timer() - started, response=response))
        return response

    def _add(self, call: _Call) -> None:
        with self._lock:
            self._calls.append(call)

    def count(self) -> int:
        with self._lock:
            return len(self._calls)

    def since(self, start: int) -> list[_Call]:
        with self._lock:
            return self._calls[start:]


def _status_of(error: LLMError) -> int | None:
    status = getattr(error, "status", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _cause_of(error: LLMError) -> str:
    """失败原因：抛出的地方给了 kind 就用它（core.errors.LLMErrorKind），否则超时看异常类型、
    其余按 HTTP 状态码归类；都没有就是 UNKNOWN。只读 status / kind，不解析异常消息。"""
    kind = getattr(error, "kind", None)
    if isinstance(kind, str) and kind in LLM_ERROR_KINDS:
        return kind
    if isinstance(error, LLMTimeout):
        return "timeout"
    status = _status_of(error)
    if status is None:
        return UNKNOWN
    if status in (401, 403):
        return "auth"
    if status == 402:
        return "billing"
    if status == 429:
        return "rate_limit"
    if 300 <= status < 400:
        return "redirect"
    if 400 <= status < 500:
        return "bad_request"
    return "server" if status >= 500 else UNKNOWN


def _error_text(error: LLMError) -> str:
    """一次失败的说明，如"请求被拒（HTTP 400，错误代码 invalid_parameter）——客户端给的说明"。
    客户端给的说明先过 mask_pii、截到 200 字（它本来就抹掉了密钥、不带响应原文，这里再兜一层）。"""
    tags: list[str] = []
    status = _status_of(error)
    if status is not None:
        tags.append(f"HTTP {status}")
    code = getattr(error, "code", None)
    if isinstance(code, str | int) and not isinstance(code, bool) and str(code).strip():
        tags.append(f"错误代码 {str(code).strip()[:ERROR_CODE_CHARS]}")
    head = _CAUSE_LABELS[_cause_of(error)] + (f"（{'，'.join(tags)}）" if tags else "")
    detail = mask_pii(str(error)).strip()[:ERROR_TEXT_CHARS]
    return f"{head}——{detail}" if detail else head


def _call_problem(call: _Call) -> _Problem | None:
    """调用层面的问题：失败、被 max_tokens 截断、回了空文本。"""
    if call.error is not None:
        return _Problem(_cause_of(call.error), _error_text(call.error))
    assert call.response is not None
    blank = not call.response.text.strip()
    if call.response.finish_reason == "length":
        tail = "，正文是空的" if blank else ""
        return _Problem(TRUNCATED, f"输出被 max_tokens 截断（finish_reason 是 length）{tail}")
    if blank:
        return _Problem(EMPTY, "调用成功，但回的正文是空的")
    return None


def _vote_problem(
    call: _Call, request: DecisionRequest, optional: Collection[str]
) -> _Problem | None:
    """这一票用判断层自己的解析（decision.voting.parse_sample）读不读得出每个必答问题。
    optional 里的问题（路由的追问）答不出来按"无"算，不要求。"""
    assert call.response is not None
    votes = parse_sample(request.questions, call.response.text)
    if not votes:
        return _Problem(UNPARSEABLE, "回答里读不出要的 JSON 答案")
    missing = [k for k in request.questions if k not in votes and k not in optional]
    if missing:
        return _Problem(
            UNPARSEABLE, f"没有答出 {'、'.join(missing)}（没答、读不出，或者不在给的选项里）"
        )
    return None


def _check_judge(
    name: str,
    probe: _Probe,
    provider: DecisionProvider,
    request: DecisionRequest,
    optional: Collection[str] = (),
) -> _Shape:
    """用评测时的判断层答一次评测时的问题（和路由器、质检一样：decide 之后 validate_response），
    再逐票看：调用有没有失败、截断、空文本，这一票读不读得出来。"""
    start = probe.count()
    verdict: _Problem | None = None
    try:
        validate_response(request, provider.decide(request))
    except (DecisionError, LLMError) as e:
        text = mask_pii(str(e)).strip()[:ERROR_TEXT_CHARS]
        verdict = _Problem(UNPARSEABLE, f"判断层没能得出答案——{text}")
    shape = _Shape(name, calls=probe.since(start))
    for call in shape.calls:
        problem = _call_problem(call) or _vote_problem(call, request, optional)
        if problem is not None:
            shape.problems.append(problem)
    if not shape.problems:  # 逐票都正常：整体的结论才是原因
        shape.verdict = verdict
    return shape


def _check_generation(probe: _Probe, request: LLMRequest) -> _Shape:
    """发一次评测时的生成请求，输出用生成环节自己的解析（parse_generation_output）检查。"""
    start = probe.count()
    try:
        probe.complete(request)
    except LLMError:
        pass  # 记在 probe 里了
    shape = _Shape(GENERATE, calls=probe.since(start))
    for call in shape.calls:
        problem = _call_problem(call)
        if problem is None:
            assert call.response is not None
            try:
                parse_generation_output(call.response.text)
            except LLMError as e:  # 解析的错误信息里没有模型原文
                problem = _Problem(UNPARSEABLE, f"生成的输出解析不了——{e}")
        if problem is not None:
            shape.problems.append(problem)
    return shape


def _judged_by_llm(decision: DecisionConfig, purpose: str) -> bool:
    """这份配置下这个环节的判断层是不是大模型（decision.factory.build_decision_provider 的规则：
    质检除了 jev 都用大模型，路由只有 incloud_llm 用）。"""
    if decision.provider == ProviderName.JEV:
        return False
    return purpose == QA or decision.provider == ProviderName.INCLOUD_LLM


def _synthetic_requests(
    config: EngineConfig,
) -> tuple[DecisionRequest, DecisionRequest, LLMRequest]:
    """检查用的三份请求（§6.5 路由请求、§5.5 质检请求、§5.4 生成请求）。

    和评测走同一条路造出来：离线演示单元（不调大模型）对一条合成的买家消息做会话理解，单元的
    DecisionRouter.build_request 造路由请求，单元的检索给出依据，生成请求用真实的提示词构造，
    质检请求和 qa.gate 一样先遮个人信息再用 core.states.build_qa_state。"""
    # 用到时才导入：rodit.app 会把整个应用层带进来（app.rollout 又依赖 rodit.eval.fingerprint）
    from rodit.app.demo import build_demo_unit
    from rodit.testing.fakes import make_message
    from rodit.testing.fixtures import SHOP_A, build_demo_world

    world = build_demo_world()
    unit = build_demo_unit(world.shop(SHOP_A).platform, world=world, config=config)
    message = make_message(
        PREFLIGHT_QUESTION,
        world=world,
        shop_id=SHOP_A,
        conversation_id="preflight",
        msg_id="preflight_q",
        item_id=PREFLIGHT_ITEM_ID,
    )
    ctx = unit.engine.builder.build(message)
    route = RouteDecision(
        action=RouteAction.ANSWER,
        rule=6,
        intent=Intent.SIZE_ADVICE,
        intent_confidence=1.0,
        provider=ProviderName.INCLOUD_LLM.value,
    )
    evidence = unit.retriever.retrieve(ctx, route)
    cited = (ev.item_size_advice(PREFLIGHT_ITEM_ID), ev.item_size_chart(PREFLIGHT_ITEM_ID))
    draft = Draft(
        text=PREFLIGHT_ANSWER,
        evidence_ids=[i for i in cited if evidence.get(i) is not None],
        kind=DraftKind.LLM,
    )
    qa_state = build_qa_state(
        masked_context(ctx), draft, masked_evidence(evidence), answer_text=mask_text(draft.text)
    )
    return (
        unit.router.build_request(ctx),
        DecisionRequest(model=QA_MODEL, state=qa_state, questions=dict(QA_QUESTIONS)),
        build_generation_request(ctx, route, evidence, config=config.generation),
    )


def _check_all(probe: _Probe, config: EngineConfig) -> list[_Shape]:
    """python -m rodit.eval 的三种请求：路由、质检、生成（一种没过也接着查完，一次看全）。"""
    decision = config.decision
    route_request, qa_request, generation_request = _synthetic_requests(config)
    not_llm = f"不走大模型判断层（decision.provider 是 {decision.provider.value}）"
    shapes: list[_Shape] = []
    if _judged_by_llm(decision, ROUTE):
        provider = build_decision_provider(decision, purpose="route", llm=probe)
        request = DecisionRouter.request_for(provider, route_request)  # 投票制多一道追问
        followups = set(request.questions) - set(route_request.questions)
        shapes.append(_check_judge(ROUTE, probe, provider, request, followups))
    else:
        shapes.append(_Shape(ROUTE, skipped=f"这份配置的路由{not_llm}"))
    if _judged_by_llm(decision, QA):
        provider = build_decision_provider(decision, purpose="qa", llm=probe)
        shapes.append(_check_judge(QA, probe, provider, qa_request))
    else:
        shapes.append(_Shape(QA, skipped=f"这份配置的质检{not_llm}"))
    shapes.append(_check_generation(probe, generation_request))
    return shapes


def _unit(shape: _Shape) -> str:
    return "票" if shape.name != GENERATE and len(shape.calls) > 1 else "次"


def _shape_lines(shape: _Shape, *, show_result: bool) -> list[str]:
    """一种请求的说明：调用次数、耗时、超时；没过时逐项列原因；有调用成功但偏慢时提醒。"""
    label = _LABELS[shape.name]
    if shape.skipped is not None:
        return [f"  {label}：没有检查——{shape.skipped}"]
    unit = _unit(shape)
    count = f"{len(shape.calls)} {unit}" + ("并发" if unit == "票" else "")
    timeout, knob = shape.timeout_s, _KNOBS[shape.name]
    parts: list[str] = []
    if show_result:
        parts.append("没过" if shape.failed else "通过")
    if shape.calls:
        spent = " / ".join(f"{c.seconds:.2f}" for c in sorted(shape.calls, key=lambda c: c.seconds))
        parts.append(f"耗时 {spent} 秒")
    if timeout is not None:
        parts.append(f"单次超时 {timeout:g} 秒（{knob}）")
    lines = [f"  {label}（{count}）：{'，'.join(parts)}"]
    for problem, n in Counter(shape.problems).items():
        lines.append(f"    - {n} {unit}：{problem.text}")
    if shape.verdict is not None:
        lines.append(f"    - {shape.verdict.text}")
    answered = [c.seconds for c in shape.calls if c.error is None]
    if timeout is not None and answered and max(answered) > SLOW_SHARE * timeout:
        lines.append(
            f"    注意：最慢的一{unit}用了 {max(answered):.2f} 秒，超过单次超时 {timeout:g} 秒的 "
            f"{SLOW_SHARE:.0%}——整轮评测里这种请求会间歇性超时（{_SLOW_EFFECTS[shape.name]}）。"
            f"要放宽就另写 --config 调大 {knob}（{_NOT_EVIDENCE}）。"
        )
    return lines


def _path_line(client: object) -> str:
    """网络路径（§12.1 的耗时要看这条路径）。只说经没经代理，不带代理地址。"""
    endpoint = describe_endpoint(client) or "大模型接口"
    proxied = via_proxy(client)
    if proxied is None:
        return "  网络路径：不确定经没经代理（不是直接走 HTTP 的客户端，或者注入了 transport）"
    if proxied:
        return (
            f"  网络路径：经环境里配置的代理访问 {endpoint}，上面的耗时含代理这一段；"
            "要直连就把这个主机加进 NO_PROXY（同时有小写的 no_proxy 时以小写的为准）"
        )
    return f"  网络路径：直连 {endpoint}（没有经代理）"


def _next_step(cause: str, client: object, config: EngineConfig) -> str:
    """每种原因的下一步（一行）。"""
    key_env = getattr(client, "api_key_env", None)
    key = f"环境变量 {key_env} " if isinstance(key_env, str) and key_env else "密钥的环境变量"
    votes = config.decision.vote_samples
    steps = {
        "timeout": (
            "超时：从这台机器调这个接口，单次请求比配置的超时慢。想先探一探模型，就另写一份把 "
            "decision.timeout_s / generation.timeout_s（和 latency.message_budget_s）调大的 YAML，"
            f"用 --config 跑——{_NOT_EVIDENCE}；再看上面的网络路径，请求是不是绕了代理。"
            "环境变量 RODIT_LLM_TIMEOUT_S 管不到这些请求。"
        ),
        "rate_limit": (
            f"被限流（HTTP 429）：账号的并发或速率上限低于一条消息要的量（路由 {votes} 票同时发）。"
            "提高限额，或用 --config 把 decision.vote_samples 调小"
            f"（{_NOT_EVIDENCE}）；有的服务额度用完也回 429，先看一眼余额。"
        ),
        "auth": f"密钥无效或没有权限：检查{key}里的密钥是不是这个接口、这个模型的。",
        "billing": "余额或额度用完：充值或换一个有额度的账号，再调也是一样。",
        "bad_request": (
            "请求被拒：这几种请求都带 response_format json_object（JSON 模式），模型或接口可能"
            "不支持；也可能是模型名不对，或者不接受这次的 temperature / max_tokens——对照上面的"
            "错误代码，检查模型名（RODIT_LLM_MODEL 或 llm.model）和 --config 里的 "
            "decision.vote_temperature、decision.qa_vote_samples（只投一票时温度是 0）、"
            "generation.temperature。"
        ),
        "content_filter": (
            "被模型服务的内容安全拦截：检查用的是合成的客服对话，换一个模型或问服务方。"
        ),
        "server": "模型服务出错：稍后再试，一直这样就换接口。",
        "redirect": (
            "接口返回了重定向（不跟随）：检查 RODIT_LLM_BASE_URL（或配置的 llm.base_url）"
            "是不是最终地址。"
        ),
        "network": (
            "连不上接口（DNS、TLS、代理、连接被拒）：检查地址和网络；看上面的网络路径，代理没开、"
            "或者这个主机不该走代理时把它加进 NO_PROXY。"
        ),
        "malformed": (
            "响应读不出来（不是 JSON、没有 choices、没有文本）：确认接口是 OpenAI 兼容的 "
            f"/chat/completions；开着思考模式的模型也可能只回思考内容——{_THINKING_HINT}。"
        ),
        "config": "本地配置不对，请求没有发出去：按上面的说明补上（比如密钥的环境变量）。",
        UNKNOWN: (
            "看上面的说明——客户端没有给出 HTTP 状态码或错误代码，这里分不出是哪一类：先核对接口"
            "地址、模型名、密钥，再看接口接不接受这些请求参数（JSON 模式、temperature、"
            "max_tokens）。"
        ),
        EMPTY: (
            "回了空的正文：模型很可能开着思考模式，思考内容把 max_tokens 用完了——"
            f"{_THINKING_HINT}。"
        ),
        TRUNCATED: (
            "输出被 max_tokens 截断：模型很可能开着思考模式，思考内容也占 max_tokens——"
            f"{_THINKING_HINT}；不是思考模式的话就是回答比预期长"
            "（生成可以调 generation.max_tokens，判断层的 max_tokens 不能配置）。"
        ),
        UNPARSEABLE: (
            "回答读不出来：确认模型支持 JSON 模式、照提示词只输出 JSON；开着思考模式时也常这样——"
            f"{_THINKING_HINT}。"
        ),
    }
    return steps[cause]


def run_preflight(
    client: LLMClient,
    config: EngineConfig | None = None,
    *,
    route_request: DecisionRequest | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> list[str]:
    """preflight 的实现（见模块说明），多两个参数：

    - route_request：给了就只查路由判断层，原样发这份请求（python -m rodit.eval.compare：它只调
      路由判断层，--llm env 一律用 build_llm_provider 按 config.decision 组装，传这一轮要发的
      第一条请求）；
    - timer：量耗时用的单调时钟（秒），测试里注入假的。
    没过抛 PreflightFailed；通过返回摘要，一行一条。"""
    config = config or EngineConfig()
    probe = _Probe(client, timer)
    if route_request is None:
        shapes = _check_all(probe, config)
    else:
        provider = build_llm_provider(config.decision, purpose="route", llm=probe)
        shapes = [_check_judge(ROUTE, probe, provider, route_request)]
    calls = probe.count()
    failed = [shape for shape in shapes if shape.failed]
    body = [line for shape in shapes for line in _shape_lines(shape, show_result=bool(failed))]
    if not failed:
        return [
            f"大模型连通性检查通过：按这次评测要发的请求试了 {calls} 次调用"
            "（合成数据，不计入评测用量）",
            *body,
            _path_line(client),
        ]
    checked = sum(1 for shape in shapes if shape.skipped is None)
    names = "、".join(_LABELS[shape.name] for shape in failed)
    problems = [p for shape in failed for p in (*shape.problems, shape.verdict) if p is not None]
    causes = list(dict.fromkeys(p.cause for p in problems))  # 每种原因只说一次，按出现的先后
    steps = [f"  - {_next_step(cause, client, config)}" for cause in causes]
    if any(0 < len(shape.problems) < len(shape.calls) for shape in failed):
        steps.append(_MAYBE_OCCASIONAL)
    raise PreflightFailed(
        "\n".join(
            [
                f"大模型连通性检查失败：按这次评测要发的请求试了 {checked} 种、{calls} 次调用"
                f"（合成数据），{names}没过，评测没有开跑。",
                *body,
                _path_line(client),
                "下一步：",
                *steps,
            ]
        )
    )


def preflight(client: LLMClient, config: EngineConfig | None = None) -> list[str]:
    """连通性检查：没过抛 PreflightFailed（调用方报错退出）。

    config 是这次评测用的配置（None 表示内置默认值）。返回给人看的检查摘要，一行一条，
    命令行打到标准错误。做法见模块说明和 run_preflight。"""
    return run_preflight(client, config)


def _price(flag: float | None, env_name: str, env: Mapping[str, str]) -> float | None:
    if flag is not None:
        value = flag
    else:
        raw = env.get(env_name, "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"环境变量 {env_name} 必须是数字（每 1000 token 的价格）") from None
    if not 0 <= value < float("inf"):  # 也挡住 NaN、无穷大
        raise ValueError(f"单价必须是非负的有限数字（{env_name} 或对应的命令行参数）")
    return value


def resolve_pricing(
    price_input: float | None,
    price_output: float | None,
    env: Mapping[str, str] | None = None,
) -> Pricing | None:
    """命令行优先，其次环境变量；都没有返回 None（只报 token）。"""
    env = os.environ if env is None else env
    pi = _price(price_input, PRICE_INPUT_ENV, env)
    po = _price(price_output, PRICE_OUTPUT_ENV, env)
    if pi is None and po is None:
        return None
    if pi is None or po is None:
        raise ValueError(
            "输入、输出单价要一起给（--price-input / --price-output，或环境变量 "
            f"{PRICE_INPUT_ENV} / {PRICE_OUTPUT_ENV}）"
        )
    return Pricing(input_per_1k=pi, output_per_1k=po)


__all__ = [
    "PREFLIGHT_QUESTION",
    "PRICE_INPUT_ENV",
    "PRICE_OUTPUT_ENV",
    "PreflightFailed",
    "client_from_env",
    "describe_endpoint",
    "describe_model",
    "preflight",
    "resolve_pricing",
    "run_preflight",
    "via_proxy",
]
