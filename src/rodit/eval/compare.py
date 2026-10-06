"""判断层离线对比评测（方案 §6.3、§6.4、§13.3；§15 Phase 0 "对比判断层方案（含 Jev 在合成数据上的
表现）"、Phase 3 "视条件重新评估 Jev"）：python -m rodit.eval.compare [选项]

只比路由的判断（§6.5 的四个必答问题：intent、need_human、emotion、multi_intent），不跑生成和质检：

- 请求：合成评测集（rodit.eval.dataset）的每条用例在离线演示单元上做会话理解（写入多轮历史、
  指代到商品 / 订单），再由线上同一个 DecisionRouter.build_request 构造 §6.5 路由请求——state 由
  core.states 构造（已脱敏），问题取 core.questions.ROUTER_QUESTIONS。请求只构造一次，每个判断层
  都直接调 DecisionProvider.decide、收到同一份请求（深拷贝）：投票制大模型判断层线上还会多问一道
  追问 second_intent（只影响规则 5 分条），这里不问，各家比的是同样的四个问题。
- 判断层：rules（关键词规则，总是可用）；classifier（--classifier 模型文件）；incloud_llm
  （--llm env，和 python -m rodit.eval --llm env 同一套环境变量和连通性检查，投票次数、超时按
  --config 的 decision）；jev（--jev env，见 rodit.eval.jev_transport）。
- Jev 只收合成数据（§6.2、§6.3）：JevProvider 用 purpose="offline_eval"，state 另带
  synthetic=True，且任何字符串都不能含个人信息（JevProvider 自己检查，命中就拒绝发送、记为
  "拒绝发送"）；用例只能来自合成评测集（build_compare_cases 逐条核对），不是的话整轮拒绝运行。
  这个命令里没有任何读取真实买家数据的路径。

每个判断层的指标：
- 问题类型准确率：预测在期望类型或可接受的备选类型里算对（和 rodit.eval.scoring 一致）；
  调用失败按错算（另给"答出的准确率"）；总体、dev / test、按问题类型；macro-F1（同样按"算对"
  口径：答了可接受的备选类型时真实类型记成预测的类型）；主要误判；
- §6.6 规则 2 / 规则 3 与用例路由标注的一致率：评测集没有逐条的"要人工""情绪等级"标注，这里用
  用例允许的路由结果推出代理标注（need_human_label / emotion_label），只取说得清的用例；
  要人工按 routing.need_human（投票制按票数粒度换算，同线上）、情绪按 routing.emotion_escalate；
- 概率校准：10 个等宽分箱的可靠性表、期望校准误差（ECE）、所选类型的 Brier 分数
  （(置信度 − 是否答对)² 的均值），只算答出的用例；
- §13.3 阈值：智能辅助 target_precision = 0.90（每个类型），全自动 0.98（只算
  routing.auto_threshold 里的类型），用 rodit.eval.calibration.pick_threshold；有开发集时只用开发集
  校准（§13.1 测试集不参与调参），再在测试集上看这个阈值的准确率和覆盖率；
- 耗时 P50 / P95（每次 decide 的墙钟时间）、失败 / 超时 / 拒绝发送、token 用量（判断层报告了才有）。

§6.3 结论：Jev 对比本次跑的境内路由方案（§6.4 的境内大模型、路由分类器；关键词规则只是兜底，
不作对照）里准确率最高的那个（同分取 ECE 更低的），只看测试集：
条件 4"中文测试集上准确率和概率校准达到境内方案的水平"满足 = Jev 准确率 ≥ 境内 − 容差
（--accuracy-tolerance，默认 1 个百分点）且 ECE ≤ 境内 + 容差（--ece-tolerance，默认 0.02）；
境内只跑了关键词规则（默认的 --jev env 没加 --llm env / --classifier）、或 --limit 抽样时，
条件 4 判"无法判定"，数字照列只作参考；
条件 1–3（境内部署、运营主体和数据处理协议、平台报备）这里判断不了，写"需人工确认"。
本命令只发送合成数据，结论只作选型依据，不授权 Jev 进入生产链路（§6.2）。

输出：中文文本报告；--json 输出完整结果（不含买家原文：逐条只有用例 ID 和判断结果；接口地址只留
协议和主机，没有密钥）。退出码：0 正常；2 配置错误（环境变量没设、连通性检查失败、模型文件不能用、
筛选后没有用例等）。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rodit.app.demo import build_demo_unit
from rodit.core.config import EngineConfig, RoutingConfig, load_config
from rodit.core.context import shop_scope
from rodit.core.decision import (
    DecisionProvider,
    DecisionRequest,
    NoulAnswer,
    ScoreAnswer,
    validate_response,
)
from rodit.core.enums import INTENT_LABELS, Intent, Role, RouteAction
from rodit.core.errors import ComplianceError, DecisionError, LLMError, LLMTimeout, RoditError
from rodit.core.pii import mask_pii
from rodit.core.ports import LLMClient, LLMRequest, LLMResponse
from rodit.core.questions import (
    Q_EMOTION,
    Q_INTENT,
    Q_MULTI_INTENT,
    Q_NEED_HUMAN,
    ROUTER_MODEL,
    ROUTER_QUESTIONS,
    ROUTER_QUESTIONS_VER,
)
from rodit.decision import (
    ClassifierArtifactError,
    ClassifierProvider,
    JevConditions,
    JevProvider,
    RuleProvider,
    build_llm_provider,
)
from rodit.eval.calibration import (
    ReliabilityBin,
    Sample,
    ThresholdPick,
    pick_threshold,
    reliability,
)
from rodit.eval.cases_base import EvalCase
from rodit.eval.cli import env_client
from rodit.eval.dataset import CASES, cases
from rodit.eval.fingerprint import git_revision, judge_prompt_version
from rodit.eval.harness import DEMO_PLATFORM, DemoHarness
from rodit.eval.jev_transport import (
    JevConfigError,
    JevTimeout,
    transport_from_env,
)
from rodit.eval.llm_env import describe_endpoint, describe_model, preflight
from rodit.eval.runner import _seed_history, select_cases
from rodit.feedback.metrics import percentile
from rodit.routing.policy import vote_threshold
from rodit.routing.router import vote_samples_of
from rodit.testing import fixtures
from rodit.testing.fakes import make_message
from rodit.testing.fixtures import build_demo_world

Split = Literal["dev", "test", "all"]
Kind = Literal["domestic", "jev"]
ErrorKind = Literal["error", "timeout", "refused"]
Add = Callable[[str], None]

RULES, CLASSIFIER, INCLOUD_LLM, JEV = "rules", "classifier", "incloud_llm", "jev"
PROVIDER_ORDER = (RULES, CLASSIFIER, INCLOUD_LLM, JEV)
SLOW_PROVIDERS = frozenset({INCLOUD_LLM, JEV})  # 联网的判断层：逐条显示进度
# §6.3 条件 4 的"境内方案"：§6.4 的境内路由方案（MVP 的平台云内大模型、数据攒够后的路由分类器）；
# 关键词规则只是兜底、置信度没有校准过（rodit.decision.rules），不作对照
VERDICT_BASELINES = (CLASSIFIER, INCLOUD_LLM)
BASELINE_HINT = "条件 4 需要对照境内大模型或分类器（--llm env / --classifier）"
ASSIST_PRECISION = 0.90  # §13.3 智能辅助
AUTO_PRECISION = 0.98  # §13.3 全自动
ACCURACY_TOLERANCE = 0.01  # §6.3 条件 4：准确率不低于境内方案 1 个百分点以上
ECE_TOLERANCE = 0.02  # §6.3 条件 4：ECE 不高于境内方案 0.02 以上
ECE_BINS = 10
TOP_CONFUSIONS = 10  # JSON 里每个判断层的主要误判条数（文本报告列前 5 个）
TOP_CONFUSIONS_SHOWN = 5
FAILED_CASES_KEPT = 50  # JSON 里每个判断层列出的失败用例条数上限
FAILED_CASES_SHOWN = 5
ERROR_MESSAGE_CHARS = 200
EPS = 1e-9  # 阈值比较的浮点容差（同 routing.policy）

PROVIDER_LABELS = {
    RULES: "关键词规则（平台云内，线上兜底）",
    CLASSIFIER: "路由分类器（平台云内，§6.4 数据攒够后的路由首选）",
    INCLOUD_LLM: "境内大模型（平台云内，§6.4 MVP 路由）",
    JEV: "Jev（境外处理，只用于离线评测，§6.2、§6.3）",
}
CONDITION_LABELS = {  # 键与 JevConditions 的字段一一对应（§6.3 重新引入的四个条件）
    "in_cloud_deployment": "境内部署或私有化部署，能部署在平台云内",
    "operator_and_dpa": "公开运营主体，数据处理协议写明保留期限、承诺不用于训练",
    "platform_approved": "通过平台的报备或审核",
    "chinese_eval_passed": "中文测试集上准确率和概率校准达到境内方案的水平",
}
VERDICT_LABELS = {
    "met": "满足",
    "unmet": "未满足",
    "undetermined": "无法判定",
    "not_run": "本次没有跑 Jev",
    "manual": "需人工确认",
}
SYNTHETIC_NOTICE = (
    "本次发给各判断层的只有合成评测集的路由请求（state 由 core.states 构造、已脱敏；"
    "发给 Jev 的另带 synthetic=True 并通过个人信息检查），没有任何真实买家、商家、订单数据"
)
NOT_AUTHORIZATION = (
    "本结论只作选型依据，不授权 Jev 进入生产链路：§6.2 结论是买家消息不能发给 Jev；"
    "§6.3 四个条件要同时满足，并由人确认后才能改接线（配置里 decision.provider 不能是 jev）"
)
LABELS_NOTE = (
    "代理标注：评测集没有逐条的“要人工”“情绪等级”标注，按用例允许的路由结果推出——"
    "要人工：投诉类、标了 human、或只允许“提醒人工”的算该触发，不允许“提醒人工”、"
    "也不可能是投诉的算不该触发；"
    "情绪：标了 emotion 且只允许“安抚 / 提醒人工”的算到了安抚线，不允许安抚和提醒人工、也没标 "
    "emotion 的算没到；说不清的不计"
)
JEV_PREFLIGHT_STATE: dict[str, Any] = {  # 连通性检查：一条手写的合成问候，没有任何个人信息
    "synthetic": True,
    "platform": DEMO_PLATFORM.value,
    "messages": [{"role": Role.BUYER.value, "text": "在吗"}],
    "item_card": None,
    "order_card": None,
}
_DATASET_BY_ID = {c.id: c for c in CASES}


# ================================================================ 请求（所有判断层共用）


@dataclass(frozen=True)
class CompareCase:
    """一条用例的路由请求和标注。request 不含 synthetic 标记（发给 Jev 时由 jev_request 加上）。

    synthetic 只有用例确实来自合成评测集（rodit.eval.dataset.CASES 里同一条）时才是 True。"""

    id: str
    split: str
    expected: Intent
    ok_intents: frozenset[Intent]
    shop_id: str
    request: DecisionRequest
    need_human_label: bool | None
    emotion_label: bool | None
    synthetic: bool


def need_human_label(case: EvalCase) -> bool | None:
    """规则 2（要人工）的代理标注（见 LABELS_NOTE）：True 该触发、False 不该触发、None 说不清。"""
    allowed = set(case.expected_actions)
    if (
        case.expected_intent == Intent.COMPLAINT
        or "human" in case.tags
        or allowed == {RouteAction.HUMAN_PRIORITY}
    ):
        return True
    if (
        RouteAction.HUMAN_PRIORITY not in allowed
        and Intent.COMPLAINT not in case.acceptable_intents
    ):
        return False
    return None


def emotion_label(case: EvalCase) -> bool | None:
    """规则 3（情绪 ≥ 安抚线）的代理标注（见 LABELS_NOTE）：True 到了、False 没到、None 说不清。"""
    allowed = set(case.expected_actions)
    calm_ok = {RouteAction.SOOTHE, RouteAction.HUMAN_PRIORITY}
    if "emotion" in case.tags:
        if RouteAction.SOOTHE in allowed and allowed <= calm_ok:
            return True
        return None
    if allowed.isdisjoint(calm_ok) and case.expected_intent != Intent.COMPLAINT:
        return False
    return None


def _from_dataset(case: EvalCase) -> bool:
    return _DATASET_BY_ID.get(case.id) == case


def build_compare_cases(
    selected: Sequence[EvalCase], config: EngineConfig | None = None
) -> list[CompareCase]:
    """每条用例的 §6.5 路由请求：离线演示单元上做会话理解，再用单元的 DecisionRouter.build_request
    构造（和线上路由同一个入口）。历史写法、消息字段和 python -m rodit.eval 相同
    （rodit.eval.runner），会话 ID 每条不同、互不影响。"""
    world = build_demo_world()
    unit = build_demo_unit(DEMO_PLATFORM, world=world, config=config)
    harness = DemoHarness(engine=unit.engine, conversations=unit.repos.conversations, world=world)
    out: list[CompareCase] = []
    for case in selected:
        shop_id = getattr(fixtures, case.shop)
        if world.shop(shop_id).platform != DEMO_PLATFORM:
            raise ValueError(f"用例 {case.id} 的店铺不在演示单元的平台 {DEMO_PLATFORM} 上")
        buyer = getattr(fixtures, case.buyer)
        conv = f"cmp_{case.id}"
        _seed_history(harness, case, conv, shop_id, buyer)
        msg = make_message(
            case.text,
            world=world,
            shop_id=shop_id,
            conversation_id=conv,
            msg_id=f"{conv}_q",
            buyer_ref=buyer,
            item_id=case.item_id,
            sku_id=case.sku_id,
            order_id=case.order_id,
        )
        ctx = unit.engine.builder.build(msg)
        out.append(
            CompareCase(
                id=case.id,
                split=case.split,
                expected=case.expected_intent,
                ok_intents=frozenset({case.expected_intent, *case.acceptable_intents}),
                shop_id=shop_id,
                request=unit.router.build_request(ctx),
                need_human_label=need_human_label(case),
                emotion_label=emotion_label(case),
                synthetic=_from_dataset(case),
            )
        )
    return out


def require_synthetic(compare_cases: Sequence[CompareCase]) -> None:
    """Jev 只收合成评测集的用例（§6.2、§6.3）：有一条不是就整轮拒绝，一条也不发。"""
    bad = [c.id for c in compare_cases if c.synthetic is not True]
    if bad:
        raise ComplianceError(
            f"Jev 离线评测只能用合成评测集的用例（方案 §6.3），这些不是：{bad[:5]}"
        )


def jev_request(case: CompareCase) -> DecisionRequest:
    """发给 Jev 的请求：同一份请求（深拷贝），state 另带 synthetic=True（JevProvider 的
    offline_eval 护栏要求）。用例不是来自合成评测集时抛 ComplianceError。"""
    require_synthetic([case])
    state = copy.deepcopy(case.request.state)
    state["synthetic"] = True
    return case.request.model_copy(deep=True, update={"state": state})


# ================================================================ 判断层


@runtime_checkable
class UsageSource(Protocol):
    def usage_totals(self) -> tuple[int, int, int]:
        """累计 (输入 token, 输出 token, 报告了用量的次数)。"""
        ...


class TokenMeter:
    """包在 LLMClient 外面累计 token（实现 core.ports.LLMClient，线程安全，投票会并发调用）。"""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._totals = [0, 0, 0]

    @property
    def model(self) -> str:
        return str(getattr(self._inner, "model", "") or "")

    def __repr__(self) -> str:
        return f"TokenMeter(inner={self._inner!r})"

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._inner.complete(request)
        with self._lock:
            self._totals[0] += response.input_tokens
            self._totals[1] += response.output_tokens
            self._totals[2] += 1
        return response

    def usage_totals(self) -> tuple[int, int, int]:
        with self._lock:
            return self._totals[0], self._totals[1], self._totals[2]


class ProviderInfo(BaseModel):
    """一个判断层的说明（报告用；接口地址只留协议和主机，没有密钥）。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    label: str
    kind: Kind
    model: str | None = None
    endpoint: str | None = None
    vote_samples: int | None = None
    timeout_s: float | None = None
    classifier_sha256: str | None = None


@dataclass
class ProviderSpec:
    """参加对比的一个判断层。usage 给了就按它统计 token。"""

    info: ProviderInfo
    provider: DecisionProvider
    usage: UsageSource | None = None

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def is_jev(self) -> bool:
        return self.info.kind == "jev"


def rules_spec() -> ProviderSpec:
    return ProviderSpec(
        ProviderInfo(name=RULES, label=PROVIDER_LABELS[RULES], kind="domestic"), RuleProvider()
    )


def classifier_spec(path: str) -> ProviderSpec:
    """--classifier：和 python -m rodit.eval 一样加载（兜底关键词规则）。

    模型文件不能用时抛 ValueError。"""
    try:
        provider = ClassifierProvider.load(path, fallback=RuleProvider())
    except ClassifierArtifactError as e:
        raise ValueError(f"路由分类器模型文件不能用：{e}") from None
    return ProviderSpec(
        ProviderInfo(
            name=CLASSIFIER,
            label=PROVIDER_LABELS[CLASSIFIER],
            kind="domestic",
            classifier_sha256=provider.file_sha256,
        ),
        provider,
    )


def llm_spec(llm: LLMClient, config: EngineConfig | None = None) -> ProviderSpec:
    """--llm env：境内大模型判断层，投票次数、温度、超时按 config.decision（和线上路由一样）。"""
    decision = (config or EngineConfig()).decision
    meter = TokenMeter(llm)
    provider = build_llm_provider(decision, purpose="route", llm=meter)
    return ProviderSpec(
        ProviderInfo(
            name=INCLOUD_LLM,
            label=PROVIDER_LABELS[INCLOUD_LLM],
            kind="domestic",
            model=describe_model(llm),
            endpoint=describe_endpoint(llm),
            vote_samples=provider.samples,
            timeout_s=provider.timeout_s,
        ),
        provider,
        usage=meter,
    )


def jev_spec(
    transport: Callable[[dict[str, Any]], dict[str, Any]], *, model: str | None = None
) -> ProviderSpec:
    """--jev env：JevProvider（purpose="offline_eval"，条件默认都不满足）。transport 是
    rodit.eval.jev_transport.JevHttpTransport（或测试替身）；报告用量的话按它统计 token。"""
    kwargs = {"model": model} if model else {}
    provider = JevProvider(transport, JevConditions(), purpose="offline_eval", **kwargs)
    usage = transport if isinstance(transport, UsageSource) else None
    return ProviderSpec(
        ProviderInfo(
            name=JEV,
            label=PROVIDER_LABELS[JEV],
            kind="jev",
            model=provider.model,
            endpoint=describe_endpoint(transport),
            timeout_s=getattr(transport, "timeout_s", None),
        ),
        provider,
        usage=usage,
    )


def jev_preflight(spec: ProviderSpec) -> None:
    """Jev 连通性检查：发一条手写的合成问候（JEV_PREFLIGHT_STATE），失败抛 RuntimeError
    （地址、密钥不对时立刻报错，不白跑一整轮）。"""
    request = DecisionRequest(
        model=ROUTER_MODEL, state=copy.deepcopy(JEV_PREFLIGHT_STATE), questions=ROUTER_QUESTIONS
    )
    try:
        spec.provider.decide(request)
    except (DecisionError, LLMError) as e:
        _, message = _describe_error(e)
        raise RuntimeError(f"Jev 连通性检查失败：{message}") from None


# ================================================================ 逐条运行


@dataclass(frozen=True)
class CaseOutcome:
    """一个判断层在一条用例上的结果。latency_ms 为 None 表示没有发出（拒绝发送）。"""

    case_id: str
    intent: Intent | None = None
    confidence: float | None = None
    need_human: float | None = None
    emotion: float | None = None
    multi_intent: float | None = None
    latency_ms: float | None = None
    error_kind: ErrorKind | None = None
    error_type: str | None = None
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def answered(self) -> bool:
        return self.intent is not None


Progress = Callable[[str, int, int, CompareCase, CaseOutcome], None]
"""progress(判断层, 第几条（从 1 数）, 总条数, 用例, 结果)：每跑完一条调用一次。"""


def _chain(e: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    cur: BaseException | None = e
    while cur is not None and all(cur is not s for s in seen):
        seen.append(cur)
        cur = cur.__cause__ or cur.__context__
    return seen


def _is_timeout(e: BaseException) -> bool:
    return any(isinstance(x, TimeoutError | LLMTimeout | JevTimeout) for x in _chain(e))


def _describe_error(e: BaseException) -> tuple[str, str]:
    """(异常类型名, 消息)：取异常链里最里层的业务异常（如 JevTimeout、LLMTimeout）；
    消息过 mask_pii、截到 200 字（异常消息本来就不带密钥和响应原文，这里再兜一层）。"""
    business = [x for x in _chain(e) if isinstance(x, RoditError)]
    root = business[-1] if business else e
    return type(root).__name__, mask_pii(str(root))[:ERROR_MESSAGE_CHARS]


def _noul(response: Any, key: str) -> float | None:
    answer = response.answers.get(key)
    return answer.noul if isinstance(answer, NoulAnswer) else None


def _score(response: Any, key: str) -> float | None:
    answer = response.answers.get(key)
    return answer.score if isinstance(answer, ScoreAnswer) else None


def run_provider(
    spec: ProviderSpec,
    compare_cases: Sequence[CompareCase],
    *,
    timer: Callable[[], float] = time.perf_counter,
    progress: Progress | None = None,
) -> list[CaseOutcome]:
    """逐条调 spec.provider.decide（同一份请求的深拷贝；Jev 另带 synthetic=True），记下结果、
    耗时、失败和 token。判断层不可用（DecisionError / LLMError）记失败，拒绝发送
    （ComplianceError）记 refused，都不中断整轮。"""
    if spec.is_jev:
        require_synthetic(compare_cases)  # 一条都不发之前先整轮检查
    outcomes: list[CaseOutcome] = []
    total = len(compare_cases)
    for index, case in enumerate(compare_cases, start=1):
        request = jev_request(case) if spec.is_jev else case.request.model_copy(deep=True)
        before = spec.usage.usage_totals() if spec.usage is not None else None
        started = timer()
        outcome: CaseOutcome
        try:
            with shop_scope(case.shop_id):  # 大模型成本按店铺归属，同线上路由（§12.4）
                response = spec.provider.decide(request)
            validate_response(request, response)  # 漏答 / 错题型按不可用处理，同线上路由
            choice = response.choice(Q_INTENT)
        except ComplianceError as e:
            kind, message = _describe_error(e)
            outcome = CaseOutcome(
                case_id=case.id, error_kind="refused", error_type=kind, error=message
            )
        except (DecisionError, LLMError) as e:
            elapsed = (timer() - started) * 1000
            kind, message = _describe_error(e)
            outcome = CaseOutcome(
                case_id=case.id,
                latency_ms=round(elapsed, 1),
                error_kind="timeout" if _is_timeout(e) else "error",
                error_type=kind,
                error=message,
            )
        else:
            elapsed = (timer() - started) * 1000
            outcome = CaseOutcome(
                case_id=case.id,
                intent=Intent(choice.choice),
                confidence=choice.confidence,
                need_human=_noul(response, Q_NEED_HUMAN),
                emotion=_score(response, Q_EMOTION),
                multi_intent=_noul(response, Q_MULTI_INTENT),
                latency_ms=round(elapsed, 1),
            )
        if before is not None and spec.usage is not None:
            after = spec.usage.usage_totals()
            if after[2] > before[2]:
                outcome = _with_tokens(outcome, after[0] - before[0], after[1] - before[1])
        outcomes.append(outcome)
        if progress is not None:
            progress(spec.name, index, total, case, outcome)
    return outcomes


def _with_tokens(outcome: CaseOutcome, inp: int, out: int) -> CaseOutcome:
    return replace(outcome, input_tokens=inp, output_tokens=out)


# ================================================================ 指标


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntentRow(_Model):
    """按真实类型：n 条里答对 correct 条（accuracy，失败按错算）；预测为该类型 n_pred 条，
    其中答对的比例 precision；f1 = 2·P·R/(P+R)，R 按"算对"口径（见 intent_scores；没有答成
    备选类型时即 accuracy），这个类型的用例全都答成了备选类型时 f1 为 None。"""

    label: str
    n: int
    correct: int
    accuracy: float | None
    n_pred: int
    precision: float | None
    f1: float | None


class SplitScore(_Model):
    n: int
    answered: int
    correct: int
    accuracy: float | None  # 失败按错算
    answered_accuracy: float | None  # 只算答出的
    macro_f1: float | None
    ece: float | None
    brier: float | None


class ConfusionRow(_Model):
    true: str
    predicted: str
    count: int


class Agreement(_Model):
    """是否题和代理标注的一致率：positives 条该触发、命中 hits；
    negatives 条不该触发、误报 false_alarms。"""

    threshold: float
    n: int
    agree: int
    rate: float | None
    positives: int
    hits: int
    negatives: int
    false_alarms: int


class CalibrationReport(_Model):
    n: int  # 答出的条数
    ece: float | None
    brier: float | None
    mean_confidence: float | None
    accuracy: float | None
    bins: list[ReliabilityBin] = Field(default_factory=list)


class ThresholdRow(_Model):
    """一个类型的 §13.3 阈值：pick 在校准集上选出（None 表示达不到目标或没有样本）；
    holdout_* 是这个阈值在验证集（测试集）上的表现。"""

    intent: str
    label: str
    configured: float | None = None  # 配置里的全自动阈值（全自动才有）
    support: int  # 校准集里真实为该类型的条数
    pick: ThresholdPick | None
    holdout_support: int | None = None
    holdout_picked: int | None = None
    holdout_precision: float | None = None
    holdout_coverage: float | None = None


class ThresholdReport(_Model):
    calibrated_on: str
    validated_on: str | None
    n_calibration: int
    n_validation: int
    assist_precision: float
    auto_precision: float
    assist: list[ThresholdRow]
    auto: list[ThresholdRow]


class LatencySummary(_Model):
    n: int
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None


class FailedCase(_Model):
    id: str
    kind: ErrorKind
    error_type: str
    message: str


class FailureSummary(_Model):
    failures: int  # 调用失败（含超时）
    timeouts: int
    refused: int  # 拒绝发送（个人信息检查等，§6.3）
    by_type: dict[str, int]
    cases: list[FailedCase]


class TokenSummary(_Model):
    reported_cases: int
    input_tokens: int
    output_tokens: int
    mean_per_case: float | None


class ProviderReport(_Model):
    info: ProviderInfo
    overall: SplitScore
    by_split: dict[str, SplitScore]
    by_intent: dict[str, IntentRow]
    top_confusions: list[ConfusionRow]
    need_human: Agreement | None
    emotion: Agreement | None
    calibration: CalibrationReport
    thresholds: ThresholdReport
    latency: LatencySummary
    failures: FailureSummary
    tokens: TokenSummary | None = None


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def samples_of(
    compare_cases: Sequence[CompareCase], outcomes: Sequence[CaseOutcome]
) -> list[Sample]:
    """答出的用例 → (预测, 置信度, 真实) 三元组；预测在可接受的类型里时真实类型记成预测的类型
    （和 rodit.eval.train.score_provider 一样，§13.3 的准确率、覆盖率按"算对"计）。"""
    out: list[Sample] = []
    for case, o in zip(compare_cases, outcomes, strict=True):
        if o.intent is None or o.confidence is None:
            continue
        ok = o.intent in case.ok_intents
        out.append(
            Sample(
                predicted=o.intent.value,
                confidence=o.confidence,
                true=(o.intent if ok else case.expected).value,
            )
        )
    return out


def brier_score(samples: Sequence[Sample]) -> float | None:
    """所选类型的 Brier 分数：(置信度 − 是否答对)² 的均值；没有样本返回 None。"""
    if not samples:
        return None
    total = sum((s.confidence - (1.0 if s.predicted == s.true else 0.0)) ** 2 for s in samples)
    return round(total / len(samples), 4)


def calibration_of(samples: Sequence[Sample], *, bins: int = ECE_BINS) -> CalibrationReport:
    """可靠性表和 ECE（calibration.reliability，等宽分箱）+ Brier。"""
    if not samples:
        return CalibrationReport(n=0, ece=None, brier=None, mean_confidence=None, accuracy=None)
    table, ece = reliability(samples, bins=bins)
    return CalibrationReport(
        n=len(samples),
        ece=ece,
        brier=brier_score(samples),
        mean_confidence=round(sum(s.confidence for s in samples) / len(samples), 4),
        accuracy=_ratio(sum(1 for s in samples if s.predicted == s.true), len(samples)),
        bins=table,
    )


def intent_scores(
    rows: Sequence[tuple[Intent, frozenset[Intent], Intent | None]],
) -> tuple[dict[str, IntentRow], float | None]:
    """rows：(真实类型, 算对的类型, 预测类型或 None)。返回按类型的表现（枚举顺序）和 macro-F1。

    准确率按用例的真实类型；F1 按"算对"口径：预测在可接受的类型里时真实类型记成预测的类型
    （同 samples_of），一条答对的用例只给预测的类型记一次命中，不会因为答的是备选类型而同时
    拉低真实类型（有真实没预测）和预测类型（有预测没真实）的 F1。macro-F1 是这个口径下真实或
    预测里出现过的类型的 F1 平均（只出现在一边的 F1 记 0）；用例全都答成备选类型的类型不计 F1。"""
    n_true = Counter[Intent]()  # 按用例的真实类型（准确率）
    hit_true = Counter[Intent]()
    n_eff = Counter[Intent]()  # 按算对口径的真实类型（F1 的召回）
    n_pred = Counter[Intent]()
    hits = Counter[Intent]()  # 预测为该类型且算对
    for true, ok, pred in rows:
        n_true[true] += 1
        if pred is not None:
            n_pred[pred] += 1
        if pred is not None and pred in ok:
            hit_true[true] += 1
            n_eff[pred] += 1
            hits[pred] += 1
        else:
            n_eff[true] += 1
    table: dict[str, IntentRow] = {}
    f1s: list[float] = []
    for intent in Intent:
        if not (n_true[intent] or n_pred[intent]):
            continue
        accuracy = hit_true[intent] / n_true[intent] if n_true[intent] else None
        precision = hits[intent] / n_pred[intent] if n_pred[intent] else None
        f1: float | None = None
        if n_eff[intent] or n_pred[intent]:
            recall = hits[intent] / n_eff[intent] if n_eff[intent] else None
            if recall is None or precision is None or precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1s.append(f1)
        table[intent.value] = IntentRow(
            label=INTENT_LABELS[intent],
            n=n_true[intent],
            correct=hit_true[intent],
            accuracy=None if accuracy is None else round(accuracy, 4),
            n_pred=n_pred[intent],
            precision=None if precision is None else round(precision, 4),
            f1=None if f1 is None else round(f1, 4),
        )
    return table, (round(sum(f1s) / len(f1s), 4) if f1s else None)


def split_score(
    compare_cases: Sequence[CompareCase], outcomes: Sequence[CaseOutcome]
) -> SplitScore:
    rows = [
        (c.expected, c.ok_intents, o.intent) for c, o in zip(compare_cases, outcomes, strict=True)
    ]
    correct = sum(1 for _, ok, pred in rows if pred is not None and pred in ok)
    answered = sum(1 for o in outcomes if o.answered)
    _, macro = intent_scores(rows)
    samples = samples_of(compare_cases, outcomes)
    cal = calibration_of(samples)
    return SplitScore(
        n=len(rows),
        answered=answered,
        correct=correct,
        accuracy=_ratio(correct, len(rows)),
        answered_accuracy=_ratio(correct, answered),
        macro_f1=macro,
        ece=cal.ece,
        brier=cal.brier,
    )


def top_confusions(
    compare_cases: Sequence[CompareCase],
    outcomes: Sequence[CaseOutcome],
    *,
    limit: int = TOP_CONFUSIONS,
) -> list[ConfusionRow]:
    """误判（预测不在算对的类型里），按次数降序、同次数按类型名；失败不算误判。"""
    counts = Counter[tuple[str, str]]()
    for case, o in zip(compare_cases, outcomes, strict=True):
        if o.intent is not None and o.intent not in case.ok_intents:
            counts[(case.expected.value, o.intent.value)] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [ConfusionRow(true=t, predicted=p, count=n) for (t, p), n in ranked]


def agreement(pairs: Sequence[tuple[bool, float]], threshold: float) -> Agreement | None:
    """pairs：(代理标注, 判断层的值)；值 ≥ 阈值（容忍 1e-9）算触发。没有标注返回 None。"""
    if not pairs:
        return None
    positives = [v for label, v in pairs if label]
    negatives = [v for label, v in pairs if not label]
    hits = sum(1 for v in positives if v >= threshold - EPS)
    false_alarms = sum(1 for v in negatives if v >= threshold - EPS)
    agree = hits + len(negatives) - false_alarms
    return Agreement(
        threshold=threshold,
        n=len(pairs),
        agree=agree,
        rate=_ratio(agree, len(pairs)),
        positives=len(positives),
        hits=hits,
        negatives=len(negatives),
        false_alarms=false_alarms,
    )


def _labeled(
    compare_cases: Sequence[CompareCase],
    outcomes: Sequence[CaseOutcome],
    label: Callable[[CompareCase], bool | None],
    value: Callable[[CaseOutcome], float | None],
) -> list[tuple[bool, float]]:
    out: list[tuple[bool, float]] = []
    for case, o in zip(compare_cases, outcomes, strict=True):
        lab, val = label(case), value(o)
        if lab is not None and val is not None:
            out.append((lab, val))
    return out


def threshold_rows(
    calib: Sequence[Sample],
    holdout: Sequence[Sample] | None,
    intents: Sequence[Intent],
    target: float,
    *,
    configured: Mapping[Intent, float] | None = None,
) -> list[ThresholdRow]:
    """§13.3：每个类型在 calib 上用 pick_threshold 选阈值，再看它在 holdout 上的准确率和覆盖率
    （覆盖率口径同 pick_threshold：阈值以上的条数 / 真实为该类型的条数）。"""
    rows: list[ThresholdRow] = []
    for intent in intents:
        pick = pick_threshold(calib, intent.value, target)
        row = ThresholdRow(
            intent=intent.value,
            label=INTENT_LABELS[intent],
            configured=(configured or {}).get(intent),
            support=sum(1 for s in calib if s.true == intent.value),
            pick=pick,
        )
        if holdout is not None:
            support = sum(1 for s in holdout if s.true == intent.value)
            row.holdout_support = support
            if pick is not None:
                picked = [
                    s
                    for s in holdout
                    if s.predicted == intent.value and s.confidence >= pick.threshold
                ]
                row.holdout_picked = len(picked)
                row.holdout_precision = _ratio(
                    sum(1 for s in picked if s.true == intent.value), len(picked)
                )
                row.holdout_coverage = _ratio(len(picked), support)
        rows.append(row)
    return rows


def _intents_in(compare_cases: Sequence[CompareCase]) -> list[Intent]:
    present = {c.expected for c in compare_cases}
    return [i for i in Intent if i in present]


def thresholds_of(
    compare_cases: Sequence[CompareCase],
    outcomes: Sequence[CaseOutcome],
    routing: RoutingConfig,
    *,
    assist_precision: float,
    auto_precision: float,
) -> ThresholdReport:
    """有开发集时只用开发集校准、测试集验证（§13.1）；否则用本次跑的全部用例校准、不验证。"""
    pairs = list(zip(compare_cases, outcomes, strict=True))
    dev = [(c, o) for c, o in pairs if c.split == "dev"]
    test = [(c, o) for c, o in pairs if c.split == "test"]
    if dev:
        calib_pairs, calibrated_on = dev, "dev"
        holdout_pairs, validated_on = (test, "test") if test else (None, None)
    else:
        calib_pairs, calibrated_on = pairs, "test" if test else "all"
        holdout_pairs, validated_on = None, None
    calib_cases = [c for c, _ in calib_pairs]
    calib = samples_of(calib_cases, [o for _, o in calib_pairs])
    holdout = (
        samples_of([c for c, _ in holdout_pairs], [o for _, o in holdout_pairs])
        if holdout_pairs is not None
        else None
    )
    auto_intents = [i for i in Intent if i in routing.auto_threshold]
    return ThresholdReport(
        calibrated_on=calibrated_on,
        validated_on=validated_on,
        n_calibration=len(calib),
        n_validation=len(holdout) if holdout is not None else 0,
        assist_precision=assist_precision,
        auto_precision=auto_precision,
        assist=threshold_rows(calib, holdout, _intents_in(calib_cases), assist_precision),
        auto=threshold_rows(
            calib, holdout, auto_intents, auto_precision, configured=routing.auto_threshold
        ),
    )


def latency_of(outcomes: Sequence[CaseOutcome]) -> LatencySummary:
    values = [o.latency_ms for o in outcomes if o.latency_ms is not None]
    return LatencySummary(
        n=len(values),
        p50_ms=percentile(values, 0.5) if values else None,
        p95_ms=percentile(values, 0.95) if values else None,
        max_ms=max(values) if values else None,
    )


def failures_of(outcomes: Sequence[CaseOutcome]) -> FailureSummary:
    failed = [o for o in outcomes if o.error_kind is not None]
    return FailureSummary(
        failures=sum(1 for o in failed if o.error_kind in ("error", "timeout")),
        timeouts=sum(1 for o in failed if o.error_kind == "timeout"),
        refused=sum(1 for o in failed if o.error_kind == "refused"),
        by_type=dict(Counter(o.error_type or "?" for o in failed).most_common()),
        cases=[
            FailedCase(
                id=o.case_id,
                kind=o.error_kind,  # type: ignore[arg-type]
                error_type=o.error_type or "?",
                message=o.error or "",
            )
            for o in failed[:FAILED_CASES_KEPT]
        ],
    )


def tokens_of(outcomes: Sequence[CaseOutcome]) -> TokenSummary | None:
    reported = [o for o in outcomes if o.input_tokens is not None]
    if not reported:
        return None
    inp = sum(o.input_tokens or 0 for o in reported)
    out = sum(o.output_tokens or 0 for o in reported)
    return TokenSummary(
        reported_cases=len(reported),
        input_tokens=inp,
        output_tokens=out,
        mean_per_case=round((inp + out) / len(reported), 1),
    )


def provider_report(
    spec: ProviderSpec,
    compare_cases: Sequence[CompareCase],
    outcomes: Sequence[CaseOutcome],
    routing: RoutingConfig,
    *,
    assist_precision: float = ASSIST_PRECISION,
    auto_precision: float = AUTO_PRECISION,
) -> ProviderReport:
    """一个判断层的全部指标（见模块说明）。"""
    if len(compare_cases) != len(outcomes):
        raise ValueError("用例和结果条数不一致")
    rows = [
        (c.expected, c.ok_intents, o.intent) for c, o in zip(compare_cases, outcomes, strict=True)
    ]
    by_intent, _ = intent_scores(rows)
    by_split: dict[str, SplitScore] = {}
    for name in ("dev", "test"):
        idx = [i for i, c in enumerate(compare_cases) if c.split == name]
        if idx:
            by_split[name] = split_score(
                [compare_cases[i] for i in idx], [outcomes[i] for i in idx]
            )
    votes = vote_samples_of(spec.provider)
    need_human = agreement(
        _labeled(compare_cases, outcomes, lambda c: c.need_human_label, lambda o: o.need_human),
        vote_threshold(routing.need_human, votes),
    )
    emotion = agreement(
        _labeled(compare_cases, outcomes, lambda c: c.emotion_label, lambda o: o.emotion),
        routing.emotion_escalate,
    )
    return ProviderReport(
        info=spec.info,
        overall=split_score(compare_cases, outcomes),
        by_split=by_split,
        by_intent=by_intent,
        top_confusions=top_confusions(compare_cases, outcomes),
        need_human=need_human,
        emotion=emotion,
        calibration=calibration_of(samples_of(compare_cases, outcomes)),
        thresholds=thresholds_of(
            compare_cases,
            outcomes,
            routing,
            assist_precision=assist_precision,
            auto_precision=auto_precision,
        ),
        latency=latency_of(outcomes),
        failures=failures_of(outcomes),
        tokens=tokens_of(outcomes),
    )


# ================================================================ §6.3 结论


class ConditionStatus(_Model):
    name: str  # JevConditions 的字段名
    label: str
    status: Literal["met", "unmet", "undetermined", "not_run", "manual"]


class JevVerdict(_Model):
    """§6.3 条件 4 的判定（见模块说明）和四个条件的状态。"""

    status: Literal["met", "unmet", "undetermined", "not_run"]
    split: str | None = None  # 判定用的数据（test；没有测试集时为本次跑的部分，只作参考）
    n: int = 0
    jev: str | None = None
    domestic: str | None = None  # 对照的境内方案（本次准确率最高的；只有规则时是规则，只作参考）
    jev_accuracy: float | None = None
    domestic_accuracy: float | None = None
    jev_ece: float | None = None
    domestic_ece: float | None = None
    accuracy_tolerance: float
    ece_tolerance: float
    accuracy_ok: bool | None = None
    ece_ok: bool | None = None
    partial: bool = False  # --limit 抽样：不下结论
    reasons: list[str] = Field(default_factory=list)
    conditions: list[ConditionStatus] = Field(default_factory=list)
    synthetic_only: str = SYNTHETIC_NOTICE
    not_authorization: str = NOT_AUTHORIZATION


def _conditions(status4: str) -> list[ConditionStatus]:
    out = []
    for name in JevConditions.model_fields:
        status = status4 if name == "chinese_eval_passed" else "manual"
        out.append(ConditionStatus(name=name, label=CONDITION_LABELS[name], status=status))  # type: ignore[arg-type]
    return out


def _verdict_score(report: ProviderReport, split: str) -> SplitScore:
    return report.by_split[split] if split in report.by_split else report.overall


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def jev_verdict(
    reports: Sequence[ProviderReport],
    *,
    accuracy_tolerance: float = ACCURACY_TOLERANCE,
    ece_tolerance: float = ECE_TOLERANCE,
    partial: bool = False,
) -> JevVerdict:
    """§6.3 条件 4：Jev 对比本次准确率最高的境内路由方案（VERDICT_BASELINES，同分取 ECE 低的），
    只看测试集。只对照了关键词规则、或 partial（--limit 抽样）时判"无法判定"，数字和
    accuracy_ok / ece_ok 照列，只作参考。"""
    base: dict[str, Any] = {
        "accuracy_tolerance": accuracy_tolerance,
        "ece_tolerance": ece_tolerance,
        "partial": partial,
    }
    jev = next((r for r in reports if r.info.kind == "jev"), None)
    if jev is None:
        return JevVerdict(
            status="not_run",
            reasons=["本次没有跑 Jev（--jev env，并且 --providers 里有 jev）"],
            conditions=_conditions("not_run"),
            **base,
        )
    domestic = [r for r in reports if r.info.kind == "domestic"]
    baselines = [r for r in domestic if r.info.name in VERDICT_BASELINES]
    base["jev"] = jev.info.name
    if not domestic:
        return JevVerdict(
            status="undetermined",
            reasons=[f"没有境内方案对照：{BASELINE_HINT}"],
            conditions=_conditions("undetermined"),
            **base,
        )
    if "test" not in jev.by_split:
        split = next(iter(jev.by_split), "all")
        score = _verdict_score(jev, split)
        return JevVerdict(
            status="undetermined",
            split=split,
            n=score.n,
            jev_accuracy=score.accuracy,
            jev_ece=score.ece,
            reasons=["本次没有测试集用例：条件 4 只看测试集（--split test 或 all 才判定，§13.1）"],
            conditions=_conditions("undetermined"),
            **base,
        )

    def rank(r: ProviderReport) -> tuple[float, float]:
        s = _verdict_score(r, "test")
        return (
            s.accuracy if s.accuracy is not None else -1.0,
            -(s.ece if s.ece is not None else math.inf),
        )

    # 对照 §6.4 的境内路由方案；只跑了关键词规则时拿规则算出数字作参考，但不下结论
    best = max(baselines or domestic, key=rank)  # 同分时保持 PROVIDER_ORDER 里靠前的
    js, ds = _verdict_score(jev, "test"), _verdict_score(best, "test")
    base.update(
        split="test",
        n=js.n,
        domestic=best.info.name,
        jev_accuracy=js.accuracy,
        domestic_accuracy=ds.accuracy,
        jev_ece=js.ece,
        domestic_ece=ds.ece,
    )
    inconclusive: list[str] = []  # 不下结论的原因（放在最前面）；数字照列，只作参考
    if not baselines:
        inconclusive.append(
            f"本次只有关键词规则作境内对照：规则只是兜底、置信度没有校准过（§6.4），{BASELINE_HINT}；"
            f"下面对照 {best.info.name} 的数字只作参考"
        )
    if partial:
        inconclusive.append("部分评测（--limit 抽样）：测试集不全，条件 4 不下结论，数字只作参考")
    reasons: list[str] = []
    if jev.failures.failures or jev.failures.refused:
        reasons.append(
            f"Jev 有 {jev.failures.failures} 条调用失败（超时 {jev.failures.timeouts}）、"
            f"{jev.failures.refused} 条拒绝发送，都按答错计"
        )
    status: Literal["met", "unmet", "undetermined"]
    accuracy_ok: bool | None
    ece_ok: bool | None
    if js.answered == 0 or js.ece is None:
        reasons.insert(0, "Jev 在测试集上没有答出任何用例，校准无从比较")
        status, accuracy_ok, ece_ok = "unmet", False, False
    elif ds.accuracy is None or ds.ece is None:
        reasons.insert(0, f"境内方案 {best.info.name} 在测试集上没有答出任何用例，无从对照")
        status, accuracy_ok, ece_ok = "undetermined", None, None
    else:
        assert js.accuracy is not None
        accuracy_ok = js.accuracy >= ds.accuracy - accuracy_tolerance - EPS
        ece_ok = js.ece <= ds.ece + ece_tolerance + EPS
        acc_line = (
            f"准确率 Jev {_pct(js.accuracy)}，境内最好的 {best.info.name} {_pct(ds.accuracy)}"
            f"（容差 {accuracy_tolerance * 100:g} 个百分点）→ {'达到' if accuracy_ok else '没达到'}"
        )
        ece_line = (
            f"ECE Jev {_num(js.ece)}，{best.info.name} {_num(ds.ece)}（容差 {ece_tolerance:g}）"
            f"→ {'达到' if ece_ok else '没达到'}"
        )
        reasons[:0] = [acc_line, ece_line]
        status = "met" if accuracy_ok and ece_ok else "unmet"
    if inconclusive:
        status = "undetermined"
    return JevVerdict(
        status=status,
        accuracy_ok=accuracy_ok,
        ece_ok=ece_ok,
        reasons=inconclusive + reasons,
        conditions=_conditions(status),
        **base,
    )


# ================================================================ 整体报告


class CaseResult(_Model):
    intent: str | None
    confidence: float | None
    ok: bool
    need_human: float | None = None
    emotion: float | None = None
    multi_intent: float | None = None
    latency_ms: float | None = None
    error: str | None = None  # 失败时：异常类型名


class CaseRow(_Model):
    """逐条结果。不含买家原文，只有用例 ID 和各判断层的判断。"""

    id: str
    split: str
    expected_intent: str
    acceptable_intents: list[str]
    need_human_label: bool | None
    emotion_label: bool | None
    results: dict[str, CaseResult]


class CompareReport(_Model):
    split: Split
    n: int
    split_sizes: dict[str, int]
    limit: int | None = None
    seed: int | None = None
    router_questions_ver: str
    judge_prompt_ver: str | None = None  # 跑了境内大模型时才有
    git_sha: str | None = None
    git_dirty: bool | None = None
    data_notice: str = SYNTHETIC_NOTICE
    labels_note: str = LABELS_NOTE
    providers: list[ProviderReport]
    verdict: JevVerdict
    cases: list[CaseRow] = Field(default_factory=list)


def _case_rows(
    compare_cases: Sequence[CompareCase], runs: Mapping[str, Sequence[CaseOutcome]]
) -> list[CaseRow]:
    rows: list[CaseRow] = []
    for i, case in enumerate(compare_cases):
        results: dict[str, CaseResult] = {}
        for name, outcomes in runs.items():
            o = outcomes[i]
            results[name] = CaseResult(
                intent=o.intent.value if o.intent is not None else None,
                confidence=o.confidence,
                ok=o.intent is not None and o.intent in case.ok_intents,
                need_human=o.need_human,
                emotion=o.emotion,
                multi_intent=o.multi_intent,
                latency_ms=o.latency_ms,
                error=o.error_type,
            )
        rows.append(
            CaseRow(
                id=case.id,
                split=case.split,
                expected_intent=case.expected.value,
                acceptable_intents=sorted(i.value for i in case.ok_intents - {case.expected}),
                need_human_label=case.need_human_label,
                emotion_label=case.emotion_label,
                results=results,
            )
        )
    return rows


def compare(
    compare_cases: Sequence[CompareCase],
    specs: Sequence[ProviderSpec],
    *,
    split: Split = "all",
    routing: RoutingConfig | None = None,
    assist_precision: float = ASSIST_PRECISION,
    auto_precision: float = AUTO_PRECISION,
    accuracy_tolerance: float = ACCURACY_TOLERANCE,
    ece_tolerance: float = ECE_TOLERANCE,
    limit: int | None = None,
    seed: int | None = None,
    timer: Callable[[], float] = time.perf_counter,
    progress: Progress | None = None,
    git: bool = False,
) -> CompareReport:
    """在同一批请求上逐个跑判断层，汇总成报告（见模块说明）。specs 按 PROVIDER_ORDER 排序。"""
    if not compare_cases:
        raise ValueError("没有用例")
    if not specs:
        raise ValueError("没有判断层")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"判断层重名：{names}")
    if any(s.is_jev for s in specs):
        require_synthetic(compare_cases)
    routing = routing or RoutingConfig()
    order = {n: i for i, n in enumerate(PROVIDER_ORDER)}
    ordered = sorted(specs, key=lambda s: order.get(s.name, len(order)))
    runs: dict[str, list[CaseOutcome]] = {}
    reports: list[ProviderReport] = []
    for spec in ordered:
        outcomes = run_provider(spec, compare_cases, timer=timer, progress=progress)
        runs[spec.name] = outcomes
        reports.append(
            provider_report(
                spec,
                compare_cases,
                outcomes,
                routing,
                assist_precision=assist_precision,
                auto_precision=auto_precision,
            )
        )
    sha, dirty = git_revision() if git else (None, None)
    partial = limit is not None and limit < len(cases(split))
    return CompareReport(
        split=split,
        n=len(compare_cases),
        split_sizes={n: sum(1 for c in compare_cases if c.split == n) for n in ("dev", "test")},
        limit=limit,
        seed=seed if limit is not None else None,
        router_questions_ver=ROUTER_QUESTIONS_VER,
        judge_prompt_ver=judge_prompt_version() if INCLOUD_LLM in runs else None,
        git_sha=sha,
        git_dirty=dirty,
        providers=reports,
        verdict=jev_verdict(
            reports,
            accuracy_tolerance=accuracy_tolerance,
            ece_tolerance=ece_tolerance,
            partial=partial,
        ),
        cases=_case_rows(compare_cases, runs),
    )


# ================================================================ 文本报告


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def _provider_line(info: ProviderInfo) -> str:
    parts = [info.label]
    if info.model:
        parts.append(f"模型 {info.model}")
    if info.endpoint:
        parts.append(f"接口 {info.endpoint}")
    if info.vote_samples is not None:
        parts.append(f"投票 {info.vote_samples} 次")
    if info.timeout_s is not None:
        parts.append(f"单次超时 {info.timeout_s:g} 秒")
    if info.classifier_sha256:
        parts.append(f"模型文件 sha256 {info.classifier_sha256[:12]}")
    return f"  - {info.name}：" + "，".join(parts)


def _render_scores(report: CompareReport, add: Add) -> None:
    add("== 问题类型（预测在期望类型或可接受的备选类型里算对；调用失败按错算）")
    add(
        f"{'判断层':<12}{'答出/总数':>10}  {'准确率':>7}  {'dev':>7}  {'test':>7}"
        f"  {'macro-F1':>8}  {'ECE':>7}  {'Brier':>7}"
    )
    for r in report.providers:
        s = r.overall
        dev = r.by_split.get("dev")
        test = r.by_split.get("test")
        add(
            f"{r.info.name:<12}{f'{s.answered}/{s.n}':>10}  {_pct(s.accuracy):>7}"
            f"  {_pct(dev.accuracy if dev else None):>7}"
            f"  {_pct(test.accuracy if test else None):>7}"
            f"  {_num(s.macro_f1, 3):>8}  {_num(s.ece):>7}  {_num(s.brier):>7}"
        )
    add("")
    add("== 按问题类型的准确率（真实类型 n 条里答对的比例）")
    names = [r.info.name for r in report.providers]
    add(f"{'类型':<18}{'n':>4}  " + "  ".join(f"{n:>11}" for n in names))
    intents = [i.value for i in Intent if any(i.value in r.by_intent for r in report.providers)]
    for intent in intents:
        n = next((r.by_intent[intent].n for r in report.providers if intent in r.by_intent), 0)
        if n == 0:
            continue
        cells = "  ".join(
            f"{_pct(r.by_intent[intent].accuracy if intent in r.by_intent else None):>11}"
            for r in report.providers
        )
        add(f"{intent:<18}{n:>4}  {cells}")
    add("")
    add("== 主要误判（真实 → 预测 ×次数）")
    for r in report.providers:
        shown = r.top_confusions[:TOP_CONFUSIONS_SHOWN]
        text = "；".join(f"{c.true} → {c.predicted} ×{c.count}" for c in shown) or "无"
        add(f"{r.info.name}：{text}")


def _agreement_cell(a: Agreement | None) -> str:
    if a is None:
        return "无标注"
    return (
        f"一致 {_pct(a.rate)}（n={a.n}；该触发 {a.positives} 条命中 {a.hits}，"
        f"不该触发 {a.negatives} 条误报 {a.false_alarms}）"
    )


def _render_agreement(report: CompareReport, add: Add) -> None:
    add("== §6.6 规则 2（要人工）/ 规则 3（情绪到安抚线）与用例路由标注的一致率")
    add(report.labels_note)
    for r in report.providers:
        nh = r.need_human
        threshold = f"≥ {nh.threshold:.2f}" if nh is not None else ""
        add(f"{r.info.name}：要人工 {threshold} {_agreement_cell(nh)}")
        emo = r.emotion
        threshold = f"≥ {emo.threshold:g}" if emo is not None else ""
        add(f"{' ' * len(r.info.name)}  情绪 {threshold} {_agreement_cell(emo)}")


def _render_calibration(report: CompareReport, add: Add) -> None:
    add(f"== 概率校准（只算答出的；{ECE_BINS} 个等宽分箱；Brier 按所选类型）")
    for r in report.providers:
        c = r.calibration
        if c.n == 0:
            add(f"{r.info.name}：没有答出的用例")
            continue
        add(
            f"{r.info.name}：n={c.n}  ECE {_num(c.ece)}  Brier {_num(c.brier)}"
            f"  平均置信度 {_num(c.mean_confidence, 2)}  实际准确率 {_pct(c.accuracy)}"
        )
        for b in c.bins:
            add(
                f"  [{b.lower:.1f}, {b.upper:.1f}{']' if b.upper == 1.0 else ')'}  n={b.count:<4}"
                f"平均置信度 {b.mean_confidence:.2f}  实际准确率 {_pct(b.accuracy)}"
            )


def _threshold_cell(row: ThresholdRow) -> str:
    if row.support == 0:
        return "无样本"
    if row.pick is None:
        return "达不到"
    cell = f"{row.pick.threshold:.2f}/{_pct(row.pick.coverage)}"
    if row.holdout_support is not None:  # 验证集上阈值以上一条都没有时是 "→—"
        cell += f"→{_pct(row.holdout_precision)}"
    return cell


def _render_thresholds(report: CompareReport, add: Add) -> None:
    if not report.providers:
        return
    t0 = report.providers[0].thresholds
    where = f"用 {t0.calibrated_on} 校准"
    if t0.validated_on:
        where += f"，{t0.validated_on} 上验证"
    add(f"== §13.3 阈值（{where}）")
    if t0.calibrated_on == "test":
        add("注意：只跑了测试集，阈值是用测试集算的，仅供参考；正式阈值用开发集校准（§13.1）")
    add(
        "格式：阈值/校准集覆盖率→验证集准确率；“达不到”是校准集上达不到目标准确率，"
        "“无样本”是校准集没有这个类型"
    )
    names = [r.info.name for r in report.providers]
    for kind, title, target in (
        ("assist", "智能辅助", t0.assist_precision),
        (
            "auto",
            "全自动（只算 routing.auto_threshold 里的类型，括号里是配置的阈值）",
            t0.auto_precision,
        ),
    ):
        add(f"{title}，target_precision = {target:.2f}：")
        add(f"  {'类型':<24}" + "  ".join(f"{n:>20}" for n in names))
        rows_by_provider = [getattr(r.thresholds, kind) for r in report.providers]
        for i, row in enumerate(rows_by_provider[0]):
            name = row.intent if row.configured is None else f"{row.intent}（{row.configured:g}）"
            cells = "  ".join(f"{_threshold_cell(rows[i]):>20}" for rows in rows_by_provider)
            add(f"  {name:<24}{cells}")


def _render_runtime(report: CompareReport, add: Add) -> None:
    add("== 耗时（每次判断的墙钟时间，毫秒）、失败与用量")
    for r in report.providers:
        lat, f = r.latency, r.failures
        line = (
            f"{r.info.name}：P50 {_ms(lat.p50_ms)}  P95 {_ms(lat.p95_ms)}  最大 {_ms(lat.max_ms)}"
            f"；失败 {f.failures}（超时 {f.timeouts}）、拒绝发送 {f.refused}"
        )
        if f.by_type:
            line += "（" + "、".join(f"{k} ×{v}" for k, v in f.by_type.items()) + "）"
        if r.tokens is not None:
            t = r.tokens
            line += (
                f"；token 输入 {t.input_tokens:,}、输出 {t.output_tokens:,}"
                f"（{t.reported_cases} 条报告了用量，平均每条 {t.mean_per_case:g}）"
            )
        elif r.info.kind == "jev" or r.info.name == INCLOUD_LLM:
            line += "；没有报告 token 用量"
        add(line)
        for c in f.cases[:FAILED_CASES_SHOWN]:
            add(f"    {c.id}：{c.error_type}：{c.message}")


def _render_verdict(v: JevVerdict, add: Add) -> None:
    add("== §6.3 Jev 重新引入生产链路的条件（需要同时满足）")
    add(v.synthetic_only)
    for i, c in enumerate(v.conditions, start=1):
        add(f"{i}. {c.name}（{c.label}）：{VERDICT_LABELS[c.status]}")
    if v.status != "not_run" and v.split:
        add(f"条件 4 的依据（{v.split}，n={v.n}）：")
    for reason in v.reasons:
        add(f"  - {reason}")
    add(v.not_authorization)


def render_text(report: CompareReport) -> str:
    lines: list[str] = []
    add = lines.append
    sizes = "、".join(f"{k} {v} 条" for k, v in report.split_sizes.items() if v)
    add(
        f"Rodit 判断层离线对比（§6.3、§6.4、§13.3）  "
        f"评测集：{report.split}（{report.n} 条：{sizes}）"
    )
    if report.limit is not None:
        add(f"部分评测：随机抽 {report.limit} 条（seed {report.seed}）")
    add(
        f"请求：§6.5 路由的四个必答问题（问题版本 {report.router_questions_ver}），"
        "每个判断层收到同一份请求"
    )
    add("判断层：")
    for r in report.providers:
        add(_provider_line(r.info))
    add("")
    _render_scores(report, add)
    add("")
    _render_agreement(report, add)
    add("")
    _render_calibration(report, add)
    add("")
    _render_thresholds(report, add)
    add("")
    _render_runtime(report, add)
    add("")
    _render_verdict(report.verdict, add)
    if report.git_sha:
        code = report.git_sha[:12] + ("（有未提交的改动）" if report.git_dirty else "")
        add(f"代码版本 {code}")
    return "\n".join(lines)


# ================================================================ 命令行


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def _precision(text: str) -> float:
    value = float(text)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("必须在 (0, 1] 之间")
    return value


def _tolerance(text: str) -> float:
    value = float(text)
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError("必须在 [0, 1) 之间")
    return value


def _providers(text: str) -> list[str]:
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in PROVIDER_ORDER]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"不认识的判断层 {unknown}：可选 {','.join(PROVIDER_ORDER)}"
        )
    return list(dict.fromkeys(names))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rodit.eval.compare",
        description=(
            "在合成评测集上对比判断层的路由判断（§6.3、§6.4、§13.3）：问题类型准确率、macro-F1、"
            "概率校准（ECE、Brier）、§13.3 阈值、耗时和用量，以及 §6.3 Jev 条件 4 的判定。"
            "只发送合成数据"
        ),
    )
    parser.add_argument(
        "--providers",
        type=_providers,
        metavar="LIST",
        help=(
            "逗号分隔，可选 rules,classifier,incloud_llm,jev；默认 rules，"
            "再按 --classifier / --llm env / --jev env 加上对应的判断层"
        ),
    )
    parser.add_argument(
        "--classifier",
        metavar="PATH",
        help="路由分类器模型文件（python -m rodit.ops train-classifier 的输出）",
    )
    parser.add_argument(
        "--llm",
        choices=["env"],
        help="env：环境变量 RODIT_LLM_*（或 --config 的 llm）配置的境内大模型判断层",
    )
    parser.add_argument(
        "--jev",
        choices=["env"],
        help="env：环境变量 RODIT_JEV_BASE_URL / RODIT_JEV_API_KEY（RODIT_JEV_TIMEOUT_S、"
        "RODIT_JEV_MODEL 可选）配置的 Jev，只发合成数据；§6.3 条件 4 要同时给 --llm env 或 "
        "--classifier 作境内对照",
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--limit", type=_positive_int, help="随机抽 N 条（按 --seed，可复现）")
    parser.add_argument("--seed", type=int, default=0, help="--limit 抽样的随机种子（默认 0）")
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="完整结果（JSON）：不带 PATH 输出到标准输出；带 PATH 写文件，文本报告照常输出",
    )
    parser.add_argument(
        "--target-assist",
        type=_precision,
        default=ASSIST_PRECISION,
        help="§13.3 智能辅助阈值的目标准确率（默认 0.90）",
    )
    parser.add_argument(
        "--target-auto",
        type=_precision,
        default=AUTO_PRECISION,
        help="§13.3 全自动阈值的目标准确率（默认 0.98）",
    )
    parser.add_argument(
        "--accuracy-tolerance",
        type=_tolerance,
        default=ACCURACY_TOLERANCE,
        help="§6.3 条件 4：Jev 准确率可以比境内方案低多少（比例，默认 0.01 = 1 个百分点）",
    )
    parser.add_argument(
        "--ece-tolerance",
        type=_tolerance,
        default=ECE_TOLERANCE,
        help="§6.3 条件 4：Jev 的 ECE 可以比境内方案高多少（默认 0.02）",
    )
    parser.add_argument(
        "--config",
        metavar="YAML",
        help="引擎配置：大模型判断层的投票和超时、路由阈值、会话轮数（默认内置配置）",
    )
    return parser


def resolve_providers(args: argparse.Namespace) -> list[str]:
    """要跑的判断层（PROVIDER_ORDER 顺序）。缺少所需参数时抛 ValueError。"""
    if args.providers is None:
        names = [RULES]
        if args.classifier:
            names.append(CLASSIFIER)
        if args.llm == "env":
            names.append(INCLOUD_LLM)
        if args.jev == "env":
            names.append(JEV)
    else:
        names = list(args.providers)
    if CLASSIFIER in names and not args.classifier:
        raise ValueError("classifier 需要 --classifier 指定模型文件")
    if INCLOUD_LLM in names and args.llm != "env":
        raise ValueError("incloud_llm 需要 --llm env（环境变量 RODIT_LLM_*）")
    if JEV in names and args.jev != "env":
        raise ValueError("jev 需要 --jev env（环境变量 RODIT_JEV_BASE_URL / RODIT_JEV_API_KEY）")
    return [n for n in PROVIDER_ORDER if n in names]


def _progress_printer(err: TextIO) -> Progress:
    def show(name: str, index: int, total: int, case: CompareCase, o: CaseOutcome) -> None:
        if name not in SLOW_PROVIDERS:
            return
        tail = f"，失败：{o.error_type}" if o.error_kind is not None else ""
        print(
            f"[{name} {index}/{total}] {case.id} {_ms(o.latency_ms)} 毫秒{tail}",
            file=err,
            flush=True,
        )

    return show


JevFactory = Callable[[], tuple[Callable[[dict[str, Any]], dict[str, Any]], str | None]]


def main(
    argv: Sequence[str] | None = None,
    out: TextIO | None = None,
    *,
    err: TextIO | None = None,
    client_factory: Callable[[], LLMClient] | None = None,
    jev_factory: JevFactory | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> int:
    """命令行入口，返回退出码（见模块说明）。

    client_factory：--llm env 时构造大模型客户端（默认 rodit.eval.cli.env_client，和
    python -m rodit.eval --llm env 同一条规则）；jev_factory：--jev env 时构造 (transport, 模型名)
    （默认 jev_transport.transport_from_env）。测试注入替身。"""
    args = build_parser().parse_args(argv)
    out = out or sys.stdout
    err = err or sys.stderr
    try:
        config = load_config(args.config) if args.config else None
        names = resolve_providers(args)
        selected = select_cases(cases(args.split), limit=args.limit, seed=args.seed)
        if not selected:
            raise ValueError(f"评测集 {args.split} 为空")
        compare_cases = build_compare_cases(selected, config)
        specs: list[ProviderSpec] = []
        for name in names:
            if name == RULES:
                specs.append(rules_spec())
            elif name == CLASSIFIER:
                specs.append(classifier_spec(args.classifier))
            elif name == INCLOUD_LLM:
                llm = client_factory() if client_factory is not None else env_client(config)
                try:
                    preflight(llm)
                except LLMError as e:  # 客户端的错误信息里已经抹掉了密钥
                    raise RuntimeError(f"大模型连通性检查失败：{e}") from None
                specs.append(llm_spec(llm, config))
            else:
                transport, model = (jev_factory or transport_from_env)()
                spec = jev_spec(transport, model=model)
                jev_preflight(spec)
                specs.append(spec)
        report = compare(
            compare_cases,
            specs,
            split=args.split,
            routing=(config or EngineConfig()).routing,
            assist_precision=args.target_assist,
            auto_precision=args.target_auto,
            accuracy_tolerance=args.accuracy_tolerance,
            ece_tolerance=args.ece_tolerance,
            limit=args.limit,
            seed=args.seed,
            timer=timer,
            progress=_progress_printer(err),
            git=True,
        )
    except (
        ValueError,
        RuntimeError,
        OSError,
        yaml.YAMLError,
        JevConfigError,
        ComplianceError,
    ) as e:
        print(f"对比评测没有运行：{e}", file=err)
        return 2
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if args.json == "-":
        print(payload, file=out)
        return 0
    print(render_text(report), file=out)
    if args.json:
        try:
            target = Path(args.json).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload + "\n", encoding="utf-8")
        except OSError as e:
            print(f"完整结果写不进 {args.json}：{type(e).__name__}", file=err)
            return 2
        print(f"\n完整结果（JSON）已写入 {args.json}", file=out)
    return 0


__all__ = [
    "ACCURACY_TOLERANCE",
    "AUTO_PRECISION",
    "ASSIST_PRECISION",
    "CONDITION_LABELS",
    "ECE_TOLERANCE",
    "PROVIDER_ORDER",
    "Agreement",
    "CaseOutcome",
    "CompareCase",
    "CompareReport",
    "JevVerdict",
    "ProviderInfo",
    "ProviderReport",
    "ProviderSpec",
    "TokenMeter",
    "agreement",
    "brier_score",
    "build_compare_cases",
    "build_parser",
    "calibration_of",
    "classifier_spec",
    "compare",
    "emotion_label",
    "intent_scores",
    "jev_preflight",
    "jev_request",
    "jev_spec",
    "jev_verdict",
    "llm_spec",
    "main",
    "need_human_label",
    "provider_report",
    "render_text",
    "require_synthetic",
    "rules_spec",
    "run_provider",
    "samples_of",
    "threshold_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())
