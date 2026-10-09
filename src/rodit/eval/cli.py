"""评测命令行（方案 §13）：python -m rodit.eval [选项]

在合成评测集（rodit.eval.dataset）上跑演示单元（rodit.eval.harness），输出：

- 问题类型准确率：总体 + 按问题类型；路由结果准确率、出推荐的比例；
- §13.2 四个维度（事实正确、切题、合规、表达）的满分率；
- §13.2 上线门槛：只看测试集（§13.1 留出的 20%）——事实正确、合规满分率都 ≥ 95%；
- §13.3 智能辅助阈值（target_precision = 0.90）：只用开发集的路由结果校准（测试集不参与调参），
  另给出可靠性分箱和期望校准误差（ECE）。只用配置的判断层做出的路由：退回关键词规则的用例不参与
  （两边的置信度不是一个刻度），报告里写明排除了几条；
- §12.1 耗时：逐条和汇总（各环节、端到端的 P50 / P95，对照 §12.1 目标），最慢的几条；各环节有多少
  调用到超时才结束；有用例被时间预算截断或有调用超时时，端到端不写"达标"（那个数被截短了）；
- §14 大模型用量：逐条和汇总的 token、调用次数、失败次数，按用途（路由 / 生成 / 质检）的占比；
  给了单价时估算成本。失败的调用不在合计里，另给估算；接口没返回用量、回显的模型名和配置的不一样
  都会写明；
- 降级统计（§12.2）；失败清单：每条用例的问题、实际推荐；接大模型时还有模型草稿和质检原因；
- §13.5 回归比较（--baseline）：指标比基线下降超过容差时退出码为 1（这一轮不算数时是 3，见下）。
  基线开跑前就检查：--llm env 跑出来的基线自己要是有效评到模型的一轮（有运行健康度，而且按默认
  上限算数），和这次选出的用例要有共同的——不然不跑（退出码 2）。

两种模式（--llm）：
- offline（默认）：离线替身——生成只复述依据、关键词规则路由、启发式质检，全部合成数据、不联网。
  离线替身的回答质量远不如真实大模型，这里的数字是回归基线（§13.5），不是上线判断；
- env：真实的 OpenAI 兼容大模型，和线上取同一个（app.production.build_production_unit 的规则）：
  --config 里 llm.base_url 和 llm.model 都写了用配置，否则用环境变量（rodit.eval.llm_env：
  RODIT_LLM_BASE_URL / RODIT_LLM_MODEL / RODIT_LLM_TIMEOUT_S）；密钥从 llm.api_key_env 指的
  环境变量读（默认 RODIT_LLM_API_KEY）；配置里的 llm.extra_body（如关闭思考模式）并入每次请求。
  路由和质检按 EngineConfig.decision 组装（默认 incloud_llm）。
  没给 --config 时用环境变量 RODIT_CONFIG 指的文件（和 python -m rodit.ops 的任务一样）；两样都
  没有就按内置默认值跑，并提醒：配置指纹和用 config/rodit.yaml 的部署对不上，结果不能当上线依据。
  开跑前先做连通性检查（rodit.eval.llm_env.preflight，没过就不跑；--skip-preflight 跳过，
  只在调试时用）。
  只在本地开发机上这样做：密钥放本地 .env，不进仓库；评测集全是合成数据。

运行健康度（只有 --llm env；rodit.eval.health）：调用失败、被截断、读不出时链路按 §12.2 降级，评测
照样跑完，但那组数字量的是兜底逻辑，不是模型。报告里的 health 记下：路由退回关键词规则、路由丢票
（判断层照常给出结果，但有一票没算上）、质检判断层不可用、生成没有可用草稿、引擎异常各有几条，
每个用途的调用里失败（按原因分开）、被截断、为空、用不上的各有几次，实际做出路由的判断层。
超过上限（默认 2%，至少容许 1 条；--max-degraded 改这一次的上限，路由丢票的上限固定是 10%、
不随它变）这一轮就不算数：文本报告开头和上线门槛旁边写明原因，退出码 3。--allow-degraded 时
不因此返回 3（没有回归就是 0；JSON 里的 valid 照旧是 false）。连续 8 条用例的大模型调用全部失败
（抛 LLMError / LLMTimeout）时提前停下（退出码 2，不出报告；标准错误按失败原因给下一步）；
--allow-degraded 时不停。
--min-interval 秒数：相邻两条用例的开始至少隔这么久（限流时用，等待不计入耗时）。

部分评测：--intent 只跑某个问题类型，--limit N 用 --seed 随机抽 N 条（可复现）。
路由分类器（§6.4）：--classifier 模型文件（python -m rodit.ops train-classifier 的输出）代替路由的
判断层，两种模式都可以；上线前和 --baseline 一起跑回归（§13.5）。
完整结果：--json 输出 JSON 到标准输出；--json out.json 写文件（文本报告照常输出）。写文件的路径
开跑前先验证写得进去；跑完先写 JSON（临时文件写好再换名）再出文本报告。
不算数的一轮不覆盖已有的文件：--json 指的文件已经存在（或者就是 --baseline 那个）时，这一轮的
JSON 写到旁边的 <文件名>.invalid.json（out.json → out.invalid.json），标准错误和文本报告末尾
写明；--allow-degraded 也一样。
报告带指纹（rodit.eval.fingerprint）：按实际跑的记——配置的 config_fingerprint（--classifier 时
判断层记为 classifier）、路由 / 质检问题定义版本、生成环节和判断层提示词的版本、用的大模型
（模型名、接口主机）、--classifier 模型文件的 SHA-256、代码版本；
python -m rodit.ops record-eval --report out.json
把完整评测的结论记成 eval.verdict，上线状态只认指纹和线上一致的结论（§13.4、§13.5）。
退出码：0 正常；1 回归比较有指标下降；2 没有跑或没有跑完——配置错误（环境变量、基线文件读不了
或不能当基线、筛选后没有用例、--json 的路径写不进去等）、连通性检查没过、连续多条用例调用全部
失败而停下、最后写不进 JSON、跑完了但回归比较没有做成（报告照常给出）；3 这一轮没有有效评到
模型（报告照常给出，JSON 不覆盖已有的文件；同时有指标下降也是 3——兜底逻辑的数字比不出回归）；
130 被中断（Ctrl-C：结果没有保存）。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, TextIO

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rodit.app.offline import EvidenceEchoLLM
from rodit.core.audit import DEGRADED_BUDGET, DEGRADED_TENANT_ISOLATION
from rodit.core.config import EngineConfig, load_config
from rodit.core.decision import DecisionProvider
from rodit.core.enums import INTENT_LABELS, Intent, ProviderName
from rodit.core.errors import LLMError
from rodit.core.models import EngineResult
from rodit.core.ports import LLMClient
from rodit.decision import ClassifierArtifactError, ClassifierProvider, RuleProvider
from rodit.engine import (
    DEGRADED_AUDIT,
    DEGRADED_DECISION,
    DEGRADED_DECISION_QA,
    DEGRADED_ENGINE,
    DEGRADED_GENERATOR,
    DEGRADED_KNOWLEDGE,
    DEGRADED_QA,
    DEGRADED_RETRIEVER,
    DEGRADED_ROUTER,
    DEGRADED_STORAGE,
    DEGRADED_UNDERSTANDING,
)
from rodit.eval.baseline import (
    METRIC_LABELS,
    BaselineCheck,
    baseline_problems,
    compare_to_baseline,
    read_baseline,
)
from rodit.eval.calibration import (
    ReliabilityBin,
    Sample,
    ThresholdPick,
    calibrate,
    reliability,
)
from rodit.eval.dataset import EvalCase, cases
from rodit.eval.fingerprint import EvalFingerprint, file_sha256, fingerprint_of
from rodit.eval.harness import DEMO_PLATFORM, DemoHarness, demo_harness, llm_harness
from rodit.eval.health import (
    DEFAULT_CEILING,
    LOST_VOTE_CEILING,
    HealthCounts,
    PurposeHealth,
    RunHealth,
    assess,
    error_label,
)
from rodit.eval.llm_env import (
    PRICE_INPUT_ENV,
    PRICE_OUTPUT_ENV,
    client_from_env,
    describe_endpoint,
    describe_model,
    preflight,
    resolve_pricing,
    via_proxy,
)
from rodit.eval.perf import (
    DAYS_PER_MONTH,
    MESSAGES_PER_SHOP_DAY,
    MODEL_STAGES,
    PURPOSE_LABELS,
    CaseUsage,
    LatencyReport,
    Pricing,
    StageLatency,
    UsageReport,
    calls_by_purpose,
    case_usage,
    is_timeout,
    latency_report,
    usage_report,
)
from rodit.eval.runner import CaseTrace, EvalAborted, Progress, run_eval, select_cases
from rodit.eval.scoring import DIMENSIONS, CaseScore, summarize
from rodit.llm import OpenAICompatibleClient
from rodit.routing import NO_PROVIDER
from rodit.testing import fixtures
from rodit.testing.fixtures import build_demo_world

Split = Literal["dev", "test", "all"]
LLMMode = Literal["offline", "env"]
Add = Callable[[str], None]  # 文本报告逐行追加

ASSIST_PRECISION = 0.90  # §13.3 智能辅助阈值
GATE = 0.95  # §13.2 上线门槛
SLOWEST_SHOWN = 5  # 文本报告里列出最慢的几条
LOST_VOTES_SHOWN = 20  # 文本报告里列出路由丢票的用例，最多这么多条
STOP_AFTER_FAILED_CASES = 8  # 连续这么多条用例的大模型调用全部失败就停下（--allow-degraded 时不停）
CONFIG_ENV = "RODIT_CONFIG"  # 没给 --config 时 --llm env 读它指的配置文件（同 python -m rodit.ops）
EXIT_REGRESSION, EXIT_NOT_RUN, EXIT_DEGRADED, EXIT_INTERRUPTED = 1, 2, 3, 130
BANNER = "!! "  # 这一轮不算数时提示行的开头
DEFAULTS_WARNING = (
    f"注意：没有 --config，也没有环境变量 {CONFIG_ENV}，按内置默认值跑。配置指纹和用 "
    "config/rodit.yaml 的部署对不上，这次的结果不能当上线依据（record-eval 会说和线上不一致）；"
    "要当依据，加 --config config/rodit.yaml 重跑。"
)
STAGE_SHORT = {"route": "路由", "retrieve": "检索", "generate": "生成", "qa": "质检"}
OFFLINE_UNIT_LINE = (
    "单元：本地演示（离线大模型只复述依据、关键词规则路由、启发式质检；合成数据，不联网）"
)
DIMENSION_LABELS = {
    "factual": "事实正确",
    "on_topic": "切题",
    "compliance": "合规",
    "expression": "表达",
}


def _check_cases(selected: Sequence[EvalCase]) -> None:
    """评测集的店铺都要在演示单元的平台上（§9.1 每个平台一个单元）。"""
    world = build_demo_world()
    for case in selected:
        shop = world.shop(getattr(fixtures, case.shop))
        if shop.platform != DEMO_PLATFORM:
            raise ValueError(
                f"用例 {case.id} 的店铺属于 {shop.platform}，演示单元是 {DEMO_PLATFORM}"
            )


# ================================================================ 报告


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SplitSummary(_Model):
    n: int
    intent_accuracy: float
    action_accuracy: float
    suggestion_rate: float
    full_marks: dict[str, float]
    mean_scores: dict[str, float]


class IntentSummary(_Model):
    label: str
    n: int
    intent_accuracy: float
    action_accuracy: float
    suggestion_rate: float
    full_marks: dict[str, float]


class GateVerdict(_Model):
    split: Literal["test"] = "test"
    n: int
    threshold: float
    factual: float
    compliance: float
    passes: bool


class Calibration(_Model):
    target_precision: float
    samples_split: str  # 校准用的是哪部分数据
    n_samples: int  # 参与校准的条数：只算配置的判断层做出的路由
    thresholds: dict[str, ThresholdPick | None]
    # 真实类型 → 参与校准的条数。阈值是 None 时靠它分清"没有样本"（0 条：这个类型的用例都没参与
    # 校准）和"达不到目标准确率"；旧报告没有这一项，为 None
    support: dict[str, int] | None = None
    # 这部分数据里没参与校准的：退回了关键词规则的（置信度不是一个刻度）、没有路由结果的
    left_out_fallback: int = 0
    left_out_no_route: int = 0


class Reliability(_Model):
    n_samples: int
    bins: list[ReliabilityBin]
    ece: float | None  # 期望校准误差；一条样本都没有时为 None（算不出来，不是 0）
    left_out_fallback: int = 0  # 同 Calibration，范围是全部用例
    left_out_no_route: int = 0


class LLMInfo(_Model):
    """这次评测用的大模型和判断层（不含密钥；接口地址只留主机）。"""

    mode: LLMMode
    model: str | None = None
    endpoint: str | None = None
    decision_provider: str  # 路由的判断层（离线模式固定为关键词规则）
    route_votes: int | None = None  # 路由自一致性投票次数（§6.4）
    qa_votes: int | None = None
    route_timeout_s: float | None = None
    # 请求是不是经环境里配置的代理发出去的（rodit.eval.llm_env.via_proxy）：耗时是从这台机器量的，
    # 要看这条路径（§12.1）。不知道（离线替身、注入的客户端）时为 None；不记代理地址
    via_proxy: bool | None = None


class CaseRecord(_Model):
    """逐条结果（--json 的完整结果、--baseline 的比较对象）。不含买家原文和回答文字。"""

    id: str
    split: str
    expected_intent: Intent
    predicted_intent: Intent | None
    intent_confidence: float | None
    intent_ok: bool
    action: str | None
    action_ok: bool
    has_suggestion: bool
    scores: dict[str, int]
    latency_ms: dict[str, int] = Field(default_factory=dict)
    degraded: list[str] = Field(default_factory=list)
    usage: CaseUsage | None = None  # 离线模式不计量
    # 实际做出路由的判断层（incloud_llm / rules / classifier / none）；没有路由结果时为 None。
    # 配置的是 incloud_llm、这里却是 rules，就是退回了关键词规则（degraded 里有 decision）
    route_provider: str | None = None
    # 路由发出的票数（一次调用一票）和其中算得上的；离线模式、不用大模型路由时为 None / 0
    route_votes: int | None = None
    route_votes_usable: int | None = None


class Failure(_Model):
    id: str
    split: str
    expected_intent: Intent
    predicted_intent: Intent | None
    action: str | None
    scores: dict[str, int]
    problems: list[str]
    answer: str | None
    no_suggestion_reason: str | None
    # 模型草稿（含质检不通过后重写的；已脱敏、去重；只有接大模型时有）
    drafts: list[str] = Field(default_factory=list)
    # 有模型草稿，但这条用例没有一份草稿拿到质检判断层的结论（生成的第一种话术调用失败、后面的被
    # 生成器丢掉；判断层质检不可用；被时间预算截断……）：列出来的草稿没有用上
    drafts_unused: bool = False
    qa_reasons: list[str] = Field(default_factory=list)  # 质检不通过的原因（已脱敏）
    degraded: list[str] = Field(default_factory=list)
    latency_ms: dict[str, int] = Field(default_factory=dict)


class EvalCliReport(_Model):
    split: Split
    n: int
    split_sizes: dict[str, int]
    overall: SplitSummary
    by_split: dict[str, SplitSummary]
    by_intent: dict[str, IntentSummary]
    gate: GateVerdict | None
    calibration: Calibration
    reliability: Reliability
    confusion: dict[str, dict[str, int]]
    failures: list[Failure]
    intent: str | None = None  # --intent 筛选
    limit: int | None = None  # --limit 抽样
    seed: int | None = None  # 抽样用的种子（只在 limit 生效时有）
    llm: LLMInfo
    cases: list[CaseRecord] = Field(default_factory=list)
    latency: LatencyReport
    usage: UsageReport | None = None
    degraded_counts: dict[str, int] = Field(default_factory=dict)
    # 运行健康度（rodit.eval.health）：这一轮算不算"评到了模型"；离线替身不调大模型，为 None
    health: RunHealth | None = None
    baseline: BaselineCheck | None = None
    fingerprint: EvalFingerprint | None = None  # 评测用的配置、问题定义、代码版本（§13.5）


def _rate(values: Sequence[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _split_summary(scores: Sequence[CaseScore], selected: Sequence[EvalCase]) -> SplitSummary:
    report = summarize(scores, selected, gate=GATE)
    return SplitSummary(
        n=report.n,
        intent_accuracy=round(report.intent_accuracy, 4),
        action_accuracy=round(report.action_accuracy, 4),
        suggestion_rate=round(report.suggestion_rate, 4),
        full_marks=report.full_marks,
        mean_scores=report.mean_scores,
    )


def _by_intent(scores: Sequence[CaseScore], by_id: dict[str, EvalCase]) -> dict[str, IntentSummary]:
    groups: dict[Intent, list[CaseScore]] = {}
    for s in scores:
        groups.setdefault(by_id[s.case_id].expected_intent, []).append(s)
    out: dict[str, IntentSummary] = {}
    for intent in Intent:  # 按枚举顺序，和方案 §5.1 的表一致
        group = groups.get(intent)
        if not group:
            continue
        out[intent.value] = IntentSummary(
            label=INTENT_LABELS[intent],
            n=len(group),
            intent_accuracy=_rate([s.intent_ok for s in group]),
            action_accuracy=_rate([s.action_ok for s in group]),
            suggestion_rate=_rate([s.has_suggestion for s in group]),
            full_marks={d: _rate([getattr(s, d) == 2 for s in group]) for d in DIMENSIONS},
        )
    return out


def _judge_routed(provider: str | None, degraded: Sequence[str]) -> bool:
    """这条用例的路由是不是配置的判断层做出来的：有路由结果，而且没有退回关键词规则。"""
    return provider is not None and DEGRADED_DECISION not in degraded


def _samples(
    scores: Sequence[CaseScore], by_id: dict[str, EvalCase], traces: Mapping[str, CaseTrace]
) -> tuple[list[Sample], int, int]:
    """(预测类型, 置信度, 真实类型)，和没算进去的两个条数（退回关键词规则的、没有路由结果的）。

    §13.3 校准的是配置的判断层：退回关键词规则的用例不算——关键词规则的置信度（0.4–0.9）和判断层的
    （投票占比）不是一个刻度，混在一起扫出来的阈值两边都不适用；没有路由结果的也不算。"""
    samples: list[Sample] = []
    fallback = no_route = 0
    for s in scores:
        if DEGRADED_DECISION in traces[s.case_id].degraded:
            fallback += 1
        elif s.predicted_intent is None or s.intent_confidence is None:
            no_route += 1
        else:
            samples.append(
                Sample(
                    predicted=str(s.predicted_intent),
                    confidence=s.intent_confidence,
                    true=str(by_id[s.case_id].expected_intent),
                )
            )
    return samples, fallback, no_route


def _failure(
    score: CaseScore, case: EvalCase, result: EngineResult | None, trace: CaseTrace
) -> Failure:
    cards = result.suggestions if result is not None else []
    return Failure(
        id=score.case_id,
        split=score.split,
        expected_intent=case.expected_intent,
        predicted_intent=score.predicted_intent,
        action=str(score.action) if score.action is not None else None,
        scores={d: getattr(score, d) for d in DIMENSIONS},
        problems=list(score.problems),
        answer=cards[0].text if cards else None,
        no_suggestion_reason=result.no_suggestion_reason if result is not None else None,
        drafts=list(dict.fromkeys(trace.drafts)),  # 一次生成的几条候选常常相同，只列一次
        drafts_unused=bool(trace.drafts) and _no_judged_draft(trace),
        qa_reasons=list(trace.qa_reasons),
        degraded=list(trace.degraded),
        latency_ms=dict(trace.latency_ms),
    )


def _case_record(
    score: CaseScore, case: EvalCase, trace: CaseTrace, pricing: Pricing | None
) -> CaseRecord:
    votes = trace.stats("route") if trace.metered else None
    return CaseRecord(
        id=score.case_id,
        split=score.split,
        expected_intent=case.expected_intent,
        predicted_intent=score.predicted_intent,
        intent_confidence=score.intent_confidence,
        intent_ok=score.intent_ok,
        action=str(score.action) if score.action is not None else None,
        action_ok=score.action_ok,
        has_suggestion=score.has_suggestion,
        scores={d: getattr(score, d) for d in DIMENSIONS},
        latency_ms=dict(trace.latency_ms),
        degraded=list(trace.degraded),
        usage=case_usage(trace.llm_calls, pricing) if trace.metered else None,
        route_provider=trace.route_provider,
        route_votes=votes.calls if votes is not None else None,
        route_votes_usable=votes.usable if votes is not None else None,
    )


def _llm_info(
    llm: LLMClient | None, config: EngineConfig | None, *, classifier: bool = False
) -> LLMInfo:
    if llm is None:
        provider = ProviderName.CLASSIFIER if classifier else ProviderName.RULES
        return LLMInfo(
            mode="offline", model=EvidenceEchoLLM.model, decision_provider=provider.value
        )
    decision = (config or EngineConfig()).decision
    return LLMInfo(
        mode="env",
        model=describe_model(llm),
        endpoint=describe_endpoint(llm),
        decision_provider=ProviderName.CLASSIFIER.value if classifier else decision.provider.value,
        route_votes=decision.vote_samples,
        qa_votes=decision.qa_vote_samples,
        route_timeout_s=decision.timeout_s,
        via_proxy=via_proxy(llm),
    )


# ================================================================ 运行健康度（rodit.eval.health）


def _lost_votes(trace: CaseTrace) -> int:
    """路由用了判断层、却没算上的票数（调用失败、被截断、为空、有必答的问题读不出，见
    harness._readable）。退回关键词规则的用例不在这里数（那是 route_fallback）。"""
    if not _judge_routed(trace.route_provider, trace.degraded):
        return 0
    votes = trace.stats("route")
    return votes.calls - votes.usable


PIPELINE_ERROR_CODES = frozenset(
    {
        DEGRADED_ENGINE,
        DEGRADED_ROUTER,
        DEGRADED_UNDERSTANDING,
        DEGRADED_RETRIEVER,
        DEGRADED_GENERATOR,
        DEGRADED_QA,
        DEGRADED_STORAGE,
        DEGRADED_KNOWLEDGE,
        DEGRADED_AUDIT,
        DEGRADED_TENANT_ISOLATION,
    }
)
"""带这些降级标记的用例，链路上有一环出了意外（抛了不该抛的异常、存储或知识库不可用……），
这一条量到的不是模型：计入 HealthCounts.engine_errors。评测跑在内存里的演示单元上，正常的一轮
一条也不该有（离线替身的基线就是 0）。判断层不可用（decision / decision:qa）另有各自的计数，
时间预算截断（budget）、平台工具降级（tool:…）不在这里。图片理解出意外的标记（vision，
engine.images）也不在：评测集没有带图片的用例，一轮评测里出不了这个标记。"""


def _no_judged_draft(trace: CaseTrace) -> bool:
    """这条用例没有一份草稿拿到质检判断层的结论。按引擎自己的记录（engine.qa 审计的
    model_checked，runner.trace_case 读进 judged_drafts）：CallRecorder 分不出来——质检调用可以在
    截止时间之后才答完，那个结论引擎没有用。harness 没有审计时才退回去看质检调用有没有读得出的。"""
    if trace.judged_drafts is not None:
        return trace.judged_drafts == 0
    return not trace.stats("qa").usable


def _generation_unusable(trace: CaseTrace) -> bool:
    """调了生成，却没有大模型草稿送到质检（HealthCounts.generation_unusable）：
    - 生成的回答没有一条解析得出来（或者调用全失败）；
    - 或者有调用失败 / 用不上，而且这条用例一次质检判断层都没调——生成器按话术的先后处理并行的
      几次调用，前面一种调用失败就不用后面的（generation.generator），后一种答得再好也被丢掉，
      CallRecorder 却记它"用得上"。只有后一种话术失败、重写失败的用例都调过质检，不算；
      被时间预算截断的也不算（那是 budget_unjudged）。
    一条消息拆成几个问题时按整条用例看：只丢了其中一个问题的草稿，这里看不出来。"""
    generate = trace.stats("generate")
    if not generate.calls:
        return False
    if not generate.usable:
        return True
    lost = generate.failed or generate.unusable
    return bool(lost) and not trace.stats("qa").calls and DEGRADED_BUDGET not in trace.degraded


def _health_counts(traces: Sequence[CaseTrace]) -> HealthCounts:
    """一组用例的计数（各项的含义见 rodit.eval.health.HealthCounts）。降级标记是引擎给的，
    丢票、有没有送检、有没有可用草稿是 harness 的 CallRecorder 记的，判断层判过几份草稿是引擎的
    engine.qa 审计记的。"""
    attempted = [t for t in traces if t.stats("generate").calls]
    return HealthCounts(
        cases=len(traces),
        route_fallback=sum(1 for t in traces if DEGRADED_DECISION in t.degraded),
        route_lost_votes=sum(1 for t in traces if _lost_votes(t) > 0),
        # 判断层质检不可用的用例也算送过检（保证分子不超过分母）
        qa_judged=sum(
            1 for t in traces if t.stats("qa").calls or DEGRADED_DECISION_QA in t.degraded
        ),
        # 送检的草稿一条也没拿到判断层的结论才算：两份草稿并行送检、只有一份没判成的，另一份是
        # 判断层判过的，不算（那一次失败在 by_purpose 的 failed 里看得到）
        qa_unavailable=sum(
            1 for t in traces if DEGRADED_DECISION_QA in t.degraded and not t.stats("qa").usable
        ),
        generation_attempted=len(attempted),
        generation_unusable=sum(1 for t in attempted if _generation_unusable(t)),
        engine_errors=sum(1 for t in traces if PIPELINE_ERROR_CODES.intersection(t.degraded)),
        budget_cut=sum(1 for t in traces if DEGRADED_BUDGET in t.degraded),
        # 已经有草稿判过、只是重写被截断的不在这里
        budget_unjudged=sum(
            1 for t in traces if DEGRADED_BUDGET in t.degraded and _no_judged_draft(t)
        ),
    )


def _purpose_health(traces: Sequence[CaseTrace]) -> dict[str, PurposeHealth]:
    """每个用途的调用情况：次数、失败和失败原因来自 llm.usage 的 detail（原因的标签由
    health.error_label 给），被截断、为空、用不上来自 CallRecorder（它看得到响应本身）。"""
    recorded = sorted({purpose for t in traces for purpose in t.calls})
    grouped = calls_by_purpose([d for t in traces for d in t.llm_calls], also=recorded)
    out: dict[str, PurposeHealth] = {}
    for purpose, calls in grouped.items():
        failed = [d for d in calls if d.get("error")]
        stats = [t.stats(purpose) for t in traces]
        out[purpose] = PurposeHealth(
            calls=len(calls),
            failed=len(failed),
            timeouts=sum(1 for d in failed if is_timeout(d)),
            truncated=sum(s.truncated for s in stats),
            empty=sum(s.empty for s in stats),
            unusable=sum(s.unusable for s in stats),
            errors=dict(Counter(error_label(d) for d in failed).most_common()),
        )
    return out


def _run_health(
    traces: Sequence[CaseTrace], test: Sequence[CaseTrace], ceiling: float
) -> RunHealth:
    """这一轮的运行健康度。算不算数只有 health.assess 这一条规则（ceiling 是这次用的上限）。"""
    overall = _health_counts(traces)
    test_counts = _health_counts(test) if test else None
    reasons = assess(overall, test_counts, ceiling=ceiling)
    return RunHealth(
        ceiling=ceiling,
        overall=overall,
        test=test_counts,
        by_purpose=_purpose_health(traces),
        routed_by=dict(Counter(t.route_provider or NO_PROVIDER for t in traces).most_common()),
        late_calls=sum(t.late_calls for t in traces),
        valid=not reasons,
        reasons=reasons,
    )


def evaluate(
    split: Split = "all",
    *,
    target_precision: float = ASSIST_PRECISION,
    intent: Intent | str | None = None,
    limit: int | None = None,
    seed: int = 0,
    llm: LLMClient | None = None,
    config: EngineConfig | None = None,
    pricing: Pricing | None = None,
    progress: Progress | None = None,
    route_provider: DecisionProvider | None = None,
    classifier_sha256: str | None = None,
    ceiling: float = DEFAULT_CEILING,
    min_interval_s: float = 0.0,
    stop_after_failed: int | None = None,
) -> EvalCliReport:
    """在演示单元上跑评测集的一部分，汇总成报告。

    llm 为空用离线替身；给了就用它（真实模型或测试替身），路由和质检按 config.decision 组装。
    route_provider 给了就用它路由（--classifier：训练好的分类器，§6.4 上线前的 §13.5 回归），
    classifier_sha256 是它的模型文件的 SHA-256（记进指纹）。
    intent / limit / seed 见 runner.select_cases；pricing 为空时只报 token。
    ceiling：运行健康度这次用的上限（health.assess；只有给了 llm 才有 health）。
    min_interval_s / stop_after_failed 见 runner.run_eval（后者触发时抛 runner.EvalAborted）。
    """
    if not 0 <= ceiling <= 1:  # 开跑前就拦下：不然跑完一整轮才在组装 health 时报错
        raise ValueError(f"运行健康度的上限必须在 [0, 1] 之间（比例，不是百分数）：{ceiling}")
    wanted = Intent(intent) if intent is not None else None
    selected = select_cases(cases(split), intent=wanted, limit=limit, seed=seed)
    if not selected:
        where = f"（问题类型 {wanted.value}）" if wanted is not None else ""
        raise ValueError(f"评测集 {split}{where} 为空")
    _check_cases(selected)
    if llm is None:
        make_harness = partial(demo_harness, config, route_provider)
    else:
        make_harness = partial(llm_harness, llm, config, route_provider)
    run = run_eval(
        selected,
        make_harness,
        gate=GATE,
        assist_precision=target_precision,
        progress=progress,
        min_interval_s=min_interval_s,
        stop_after_failed=stop_after_failed,
    )
    by_id = {c.id: c for c in selected}
    scores = run.scores

    by_split: dict[str, SplitSummary] = {}
    for name in ("dev", "test"):
        part = [s for s in scores if s.split == name]
        if part:
            by_split[name] = _split_summary(part, selected)

    test_scores = [s for s in scores if s.split == "test"]
    gate: GateVerdict | None = None
    if test_scores:
        test_report = summarize(test_scores, selected, gate=GATE)
        gate = GateVerdict(
            n=test_report.n,
            threshold=GATE,
            factual=test_report.full_marks["factual"],
            compliance=test_report.full_marks["compliance"],
            passes=test_report.passes_gate,
        )

    # §13.1：测试集不参与调参——有开发集时只用开发集校准
    dev_scores = [s for s in scores if s.split == "dev"]
    calib_scores, calib_split = (dev_scores, "dev") if dev_scores else (scores, split)
    calib_samples, calib_fallback, calib_no_route = _samples(calib_scores, by_id, run.traces)
    intents = sorted({str(by_id[s.case_id].expected_intent) for s in calib_scores})
    thresholds = calibrate(calib_samples, intents, target_precision=target_precision)
    support = Counter(sample.true for sample in calib_samples)

    all_samples, all_fallback, all_no_route = _samples(scores, by_id, run.traces)
    # 一条样本都没有时 ECE 算不出来：记 None，不记 0（0 读起来像"校准得很好"）
    bins, ece = reliability(all_samples) if all_samples else ([], None)
    report = summarize(scores, selected, gate=GATE)
    traces = [run.traces[s.case_id] for s in scores]
    metered = any(t.metered for t in traces)
    health: RunHealth | None = None
    if llm is not None:  # 离线替身不调大模型：没有"评没评到模型"这回事
        test_traces = [run.traces[s.case_id] for s in test_scores]
        health = _run_health(traces, test_traces, ceiling)
    return EvalCliReport(
        split=split,
        n=len(scores),
        split_sizes={name: sum(1 for s in scores if s.split == name) for name in ("dev", "test")},
        overall=_split_summary(scores, selected),
        by_split=by_split,
        by_intent=_by_intent(scores, by_id),
        gate=gate,
        calibration=Calibration(
            target_precision=target_precision,
            samples_split=calib_split,
            n_samples=len(calib_samples),
            thresholds=thresholds,
            support={intent: support.get(intent, 0) for intent in intents},
            left_out_fallback=calib_fallback,
            left_out_no_route=calib_no_route,
        ),
        reliability=Reliability(
            n_samples=len(all_samples),
            bins=bins,
            ece=ece,
            left_out_fallback=all_fallback,
            left_out_no_route=all_no_route,
        ),
        confusion=report.confusion,
        failures=[
            _failure(s, by_id[s.case_id], run.results.get(s.case_id), run.traces[s.case_id])
            for s in report.failures
        ],
        intent=wanted.value if wanted is not None else None,
        limit=limit,
        seed=seed if limit is not None else None,
        llm=_llm_info(llm, config, classifier=route_provider is not None),
        cases=[_case_record(s, by_id[s.case_id], run.traces[s.case_id], pricing) for s in scores],
        latency=latency_report(
            [t.latency_ms for t in traces],
            [d for t in traces for d in t.llm_calls],
            budget_cut=sum(1 for t in traces if DEGRADED_BUDGET in t.degraded),
            evaluated=health is None or health.valid,
        ),
        usage=usage_report([t.llm_calls for t in traces], pricing) if metered else None,
        degraded_counts=dict(Counter(d for t in traces for d in t.degraded).most_common()),
        health=health,
        fingerprint=_run_fingerprint(llm, config, route_provider, classifier_sha256),
    )


def _run_fingerprint(
    llm: LLMClient | None,
    config: EngineConfig | None,
    route_provider: DecisionProvider | None,
    classifier_sha256: str | None,
) -> EvalFingerprint:
    """这次评测实际跑的指纹：用分类器路由时判断层记为 classifier（配置里写的是别的也一样——
    分类器跑出来的结论只能给分类器路由的部署作依据，反过来也一样）；大模型记实际用的客户端
    （离线替身记替身的名字，没有接口地址）。"""
    effective = config or EngineConfig()
    if route_provider is not None:
        decision = effective.decision.model_copy(update={"provider": ProviderName.CLASSIFIER})
        effective = effective.model_copy(update={"decision": decision})
    if llm is None:
        model, endpoint = EvidenceEchoLLM.model, None
    else:
        model, endpoint = describe_model(llm), describe_endpoint(llm)
    return fingerprint_of(
        effective,
        llm_model=model,
        llm_endpoint=endpoint,
        classifier_sha256=classifier_sha256 if route_provider is not None else None,
        git=True,
    )


# ================================================================ 输出


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def _num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def _summary_line(name: str, s: SplitSummary) -> str:
    marks = "  ".join(f"{DIMENSION_LABELS[d]} {_pct(s.full_marks[d])}" for d in DIMENSIONS)
    return (
        f"{name:<6} n={s.n:<3} 类型准确率 {_pct(s.intent_accuracy)}  路由 {_pct(s.action_accuracy)}"
        f"  出推荐 {_pct(s.suggestion_rate)}  |  满分率：{marks}"
    )


def _unit_line(info: LLMInfo) -> str:
    if info.mode == "offline":
        return OFFLINE_UNIT_LINE
    where = f"（{info.endpoint}）" if info.endpoint else ""
    return (
        f"单元：本地演示 + 大模型 {info.model or '未知'}{where}；"
        f"路由判断层 {info.decision_provider}"
        f"（投票 {info.route_votes} 次，单次超时 {info.route_timeout_s:g} 秒），"
        f"质检投票 {info.qa_votes} 次；合成数据"
    )


def _stage_breakdown(latency: dict[str, int]) -> str:
    parts = [f"{label} {latency[k]}" for k, label in STAGE_SHORT.items() if latency.get(k)]
    return f"（{' / '.join(parts)}）" if parts else ""


VIA_PROXY_LABELS = {True: "经本机代理", False: "直连", None: "未知"}
LIMITED_BY_BUDGET = "受时间预算限制"
NOT_EVALUATED = "不算数（见运行健康度）"


def _share(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%" if total else "—"


def _stage_verdict(key: str, stage: StageLatency, lat: LatencyReport) -> str:
    if stage.target_ms is None:
        return "—"
    if stage.within_target is None:
        # 没有数据；或者这一轮没有有效评到模型（调大模型的环节、端到端量到的是兜底路径）；或者
        # 端到端那个数被时间预算截短了——后两种都不能写"达标"
        if stage.p95 is None:
            return f"≤ {stage.target_ms}"
        if lat.not_evaluated and key in MODEL_STAGES:
            return f"≤ {stage.target_ms} {NOT_EVALUATED}"
        limited = key == "total" and lat.limited_by_budget
        return f"≤ {stage.target_ms}{' ' + LIMITED_BY_BUDGET if limited else ''}"
    return f"≤ {stage.target_ms} {'达标' if stage.within_target else '超标'}"


def _render_latency(report: EvalCliReport, add: Add) -> None:
    lat = report.latency
    add(f"== §12.1 耗时（毫秒，{lat.n} 条；没跑的环节记 0；目标对照 P95）")
    if report.llm.mode == "offline":
        add("离线替身的耗时只说明代码路径，不代表真实大模型")
    else:
        add(
            f"网络路径：{VIA_PROXY_LABELS[report.llm.via_proxy]}。耗时是从这台机器量的"
            "（含到模型服务的网络），不是平台云内的耗时"
        )
    with_calls = any(stage.llm_calls for stage in lat.stages.values())
    head = f"{'环节':<8}{'n':>4}  {'P50':>7}  {'P95':>7}  {'最大':>7}  目标"
    add(head + ("  ｜到超时才结束的调用" if with_calls else ""))
    for key, stage in lat.stages.items():
        row = (
            f"{stage.label:<8}{stage.count:>4}  {_ms(stage.p50):>7}  {_ms(stage.p95):>7}"
            f"  {_ms(stage.max):>7}  {_stage_verdict(key, stage, lat)}"
        )
        if stage.llm_calls:
            row += (
                f"  ｜{stage.llm_timeouts} / {stage.llm_calls}"
                f"（{_share(stage.llm_timeouts, stage.llm_calls)}）"
            )
        add(row)
    if lat.not_evaluated:
        add(
            "这一轮没有有效评到模型（见运行健康度）：路由、生成、质检和端到端的耗时量到的是兜底路径"
            "（调用失败得快，没调成的环节记 0），不是模型正常工作时的耗时，没超过目标也不能算达标"
        )
    if lat.limited_by_budget:
        add(
            f"端到端{LIMITED_BY_BUDGET}：有 {lat.budget_cut} 条用例被时间预算截断、"
            f"{lat.llm_timeouts} 次调用到超时才结束。单次调用到超时就断、整条消息到时间预算就返回，"
            "所以端到端超不过目标：上面的数是被截短的，不是模型服务的真实耗时，不能算达标"
            "（有调用超时的环节，耗时同样只是下限）"
        )
    slowest = sorted(
        (c for c in report.cases if c.latency_ms.get("total") is not None),
        key=lambda c: c.latency_ms["total"],
        reverse=True,
    )[:SLOWEST_SHOWN]
    if slowest:
        add(
            "最慢的几条："
            + "；".join(
                f"{c.id} {c.latency_ms['total']}{_stage_breakdown(c.latency_ms)}" for c in slowest
            )
        )


def _render_usage(report: EvalCliReport, add: Add) -> None:
    add("== §14 大模型用量（token）")
    usage = report.usage
    if usage is None:
        add("离线替身不调用大模型，不统计用量（--llm env 才有）")
        return
    errors = ""
    if usage.errors:
        kinds = "、".join(f"{k} ×{v}" for k, v in usage.error_types.items())
        errors = f"（失败 {usage.errors}：{kinds}）"
    add(
        f"调用 {usage.calls} 次{errors}；"
        f"输入 {usage.input_tokens:,}、输出 {usage.output_tokens:,}，"
        f"合计 {usage.total_tokens:,}；平均每条 {_num(usage.mean_tokens_per_case, 0)}"
    )
    for purpose, p in usage.by_purpose.items():
        share = _pct(p.token_share) if p.token_share is not None else "—"
        lat = p.call_latency_ms
        add(
            f"  {PURPOSE_LABELS.get(purpose, purpose)}：调用 {p.calls}（失败 {p.errors}），"
            f"输入 {p.input_tokens:,}、输出 {p.output_tokens:,}，占 {share}；"
            f"单次耗时 P50 {_ms(lat.p50)} / P95 {_ms(lat.p95)} 毫秒"
        )
    _render_usage_caveats(report, usage, add)
    if usage.pricing is None:
        add(
            "成本：没有单价，只报 token（--price-input / --price-output，"
            f"或环境变量 {PRICE_INPUT_ENV} / {PRICE_OUTPUT_ENV}，每 1000 token 的价格）"
        )
    else:
        add(
            f"成本：单价 输入 {usage.pricing.input_per_1k:g} / 输出 {usage.pricing.output_per_1k:g}"
            f"（每 1000 token）→ 合计 {_num(usage.cost)}，平均每条 {_num(usage.mean_cost_per_case)}"
        )
    monthly = f"每店每月 ≈ {_num(usage.projected_monthly_tokens_per_shop, 0)} token"
    if usage.projected_monthly_cost_per_shop is not None:
        monthly += f"，成本 ≈ {_num(usage.projected_monthly_cost_per_shop, 2)}"
    add(
        f"§14 示例口径（每店每天 {MESSAGES_PER_SHOP_DAY} 条消息 × {DAYS_PER_MONTH} 天）：{monthly}"
        "（评测集的问题类型分布和真实咨询不同，只作量级参考）"
    )


def _render_usage_caveats(report: EvalCliReport, usage: UsageReport, add: Add) -> None:
    """合计里少了什么：失败的调用、没返回用量的调用；实际应答的是哪个模型。"""
    if usage.errors:
        add(
            f"失败的 {usage.errors} 次调用按 0 token 计，上面的合计、成本和每店每月都不含它们"
            "（请求已经发出去，模型服务可能照常计费）。按同用途成功调用的平均值估算（只是估算，不是计量）："
        )
        for purpose, p in usage.by_purpose.items():
            if not p.errors:
                continue
            estimate = (
                f"估算 ≈ {_num(p.failed_tokens_estimate, 0)} token"
                if p.failed_tokens_estimate is not None
                else "估不出（这个用途没有带用量的成功调用可参照）"
            )
            add(f"  {PURPOSE_LABELS.get(purpose, purpose)}：失败 {p.errors} 次，{estimate}")
        if usage.failed_tokens_estimate is not None:
            add(f"  估得出的合计 ≈ {_num(usage.failed_tokens_estimate, 0)} token（估算）")
    if usage.zero_usage_calls:
        add(
            f"注意：有 {usage.zero_usage_calls} 次成功的调用 token 数是 0——接口没有返回用量"
            "（usage），上面的 token 和成本少算了，不是真的为 0"
        )
    if usage.served_models:
        names = "、".join(f"{name} ×{n}" for name, n in usage.served_models.items())
        # 接口不回显模型名时客户端填的是配置的名字：和配置的一样不等于模型服务确认过这个名字
        add(f"计量记下的模型名（成功的调用；接口不回显时记的是配置的名字）：{names}")
        configured = report.llm.model
        if configured and any(name != configured for name in usage.served_models):
            add(
                f"注意：接口回显的模型名和配置的 {configured} 不一样（配置的可能只是别名）："
                "这次量的是回显的那个版本，请记下来——别名以后改指向别的版本，配置指纹看不出来"
            )


def _render_failure(f: Failure, add: Add) -> None:
    dims = " ".join(f"{DIMENSION_LABELS[d]}{f.scores[d]}" for d in DIMENSIONS)
    predicted = f.predicted_intent.value if f.predicted_intent else "—"
    add(
        f"- {f.id}（{f.split}）期望 {f.expected_intent.value}，实际 {predicted} / {f.action}"
        f"  [{dims}]"
    )
    for p in f.problems:
        add(f"    问题：{p}")
    if f.answer is not None:
        add(f"    推荐：{f.answer}")
    elif f.no_suggestion_reason:
        add(f"    没有推荐：{f.no_suggestion_reason}")
    unused = "（没有用上：没有拿到质检判断层的结论）" if f.drafts_unused else ""
    for i, draft in enumerate(f.drafts, 1):
        add(f"    模型草稿 {i}{unused}：{draft}")
    shown = f.no_suggestion_reason if f.answer is None else None
    for reason in f.qa_reasons:
        if not shown or reason not in shown:  # "没有推荐"里已经写了的原因不重复
            add(f"    质检原因：{reason}")
    if f.degraded:
        add(f"    降级：{'、'.join(f.degraded)}")


def _render_baseline(check: BaselineCheck, add: Add) -> None:
    add(f"== §13.5 回归比较（基线 {check.source or '—'}，容差 {check.tolerance * 100:g} 个百分点）")
    add(
        f"共同用例 {check.common_cases} 条（只在本次 {check.only_current} 条、"
        f"只在基线 {check.only_baseline} 条，不比较）"
    )
    for scope in ("all", "test"):
        rows = [d for d in check.deltas if d.scope == scope]
        if not rows:
            continue
        name = "全部" if scope == "all" else "测试集"
        cells = "  ".join(
            f"{METRIC_LABELS[d.metric]} {_pct(d.baseline)} → {_pct(d.current)}"
            f"（{d.delta * 100:+.1f}）{' ↓回归' if d.regressed else ''}"
            for d in rows
        )
        add(f"{name:<6} n={rows[0].n:<3} {cells}")
    # 退出码不写在这里：这一轮同时不算数时进程返回的是 3，由 main 在标准错误里说实际的那个
    add("→ 指标没有下降" if check.passed else "→ 有指标下降超过容差：不能上线")


def _left_out_note(fallback: int, no_route: int) -> str:
    """没参与 §13.3 校准的用例：几条、为什么（都是 0 时返回空串）。"""
    parts = []
    if fallback:
        parts.append(
            f"{fallback} 条退回了关键词规则（它的置信度和判断层的不是一个刻度，不能混在一起）"
        )
    if no_route:
        parts.append(f"{no_route} 条没有路由结果")
    return "、".join(parts)


def _ceiling_note(health: RunHealth) -> str:
    note = f"上限 {health.ceiling * 100:g}%"
    if health.ceiling != DEFAULT_CEILING:
        default = f"{DEFAULT_CEILING * 100:g}%"
        note += f"，--max-degraded 给的；默认 {default}，record-eval 按默认上限重新判定"
    return note


def _health_banner(health: RunHealth) -> list[str]:
    """这一轮不算数时的提示：报告开头和上线门槛旁边各放一次，原因逐条列出。"""
    return [
        f"{BANNER}这一轮没有有效评到模型（{_ceiling_note(health)}）：有超过上限的用例走了兜底逻辑"
        "（关键词规则、只跑硬规则的质检、模板）或者缺了票，下面的数字不是模型正常工作时的成绩，"
        "不能当成模型的评测结果。原因：",
        *(f"{BANNER}  {reason}" for reason in health.reasons),
    ]


def _counts_line(name: str, c: HealthCounts) -> str:
    return (
        f"{name} {c.cases} 条：路由退回关键词规则 {c.route_fallback}、"
        f"路由丢票 {c.route_lost_votes}、"
        f"引擎异常 {c.engine_errors}、被时间预算截断 {c.budget_cut}"
        f"（其中没有草稿过完质检判断层的 {c.budget_unjudged}）；"
        f"质检判断层不可用 {c.qa_unavailable} / {c.qa_judged}（有草稿送检的）；"
        f"生成没有可用草稿 {c.generation_unusable} / {c.generation_attempted}（调过大模型生成的）"
    )


def _accuracy_of(rows: Sequence[CaseRecord]) -> str:
    if not rows:
        return "0 条"
    return f"{len(rows)} 条 {_pct(sum(1 for c in rows if c.intent_ok) / len(rows))}"


def _render_health(report: EvalCliReport, health: RunHealth, add: Add) -> None:
    add(
        f"== 运行健康度（这一轮算不算评到了模型：{_ceiling_note(health)}，"
        f"路由丢票上限 {LOST_VOTE_CEILING * 100:g}%，各项至少容许 1 条）"
    )
    add(_counts_line("全部", health.overall))
    if health.test is not None:
        add(_counts_line("测试集", health.test))
    add(
        "大模型调用（失败原因分开列：timeout 超时，http_<状态码>[:模型服务的错误代码]，"
        "其余是类别或异常类型名）："
    )
    for purpose, p in health.by_purpose.items():
        causes = "、".join(f"{label} ×{n}" for label, n in p.errors.items())
        add(
            f"  {PURPOSE_LABELS.get(purpose, purpose)}：调用 {p.calls}，"
            f"失败 {p.failed}{f'（{causes}）' if causes else ''}；"
            f"有响应的里面被截断 {p.truncated}、为空 {p.empty}、用不上 {p.unusable}"
        )
    if health.late_calls:
        add(
            f"  handle() 返回之后才结束的调用 {health.late_calls} 次"
            "（到时间预算被丢下的，已经计入上面的次数和 §14 的用量）"
        )
    routed = "、".join(f"{name} {n} 条" for name, n in health.routed_by.items())
    add(f"实际做出路由的判断层：{routed}（{NO_PROVIDER} = 没有路由结果或判断层和兜底都没答出来）")
    judge = [c for c in report.cases if _judge_routed(c.route_provider, c.degraded)]
    fallback = [c for c in report.cases if DEGRADED_DECISION in c.degraded]
    rest = len(report.cases) - len(judge) - len(fallback)
    add(
        f"类型准确率分开看：判断层路由 {_accuracy_of(judge)}；"
        f"退回关键词规则 {_accuracy_of(fallback)}；没有路由结果 {rest} 条"
    )
    lost = [c for c in judge if (c.route_votes or 0) > (c.route_votes_usable or 0)]
    if lost:
        shown = "、".join(
            f"{c.id} {c.route_votes_usable}/{c.route_votes}" for c in lost[:LOST_VOTES_SHOWN]
        )
        more = f"……（只列前 {LOST_VOTES_SHOWN} 条）" if len(lost) > LOST_VOTES_SHOWN else ""
        add(
            f"路由丢票的用例 {len(lost)} 条（算得上的票 / 发出的票；票少了，置信度的刻度就粗了）："
            f"{shown}{more}"
        )


def render_text(report: EvalCliReport) -> str:
    lines: list[str] = []
    add = lines.append
    health = report.health
    sizes = "、".join(f"{k} {v} 条" for k, v in report.split_sizes.items() if v)
    add(f"Rodit 评测（§13）  评测集：{report.split}（{report.n} 条：{sizes}）")
    add(_unit_line(report.llm))
    picks = []
    if report.intent:
        picks.append(f"只跑问题类型 {report.intent}")
    if report.limit is not None:
        picks.append(f"随机抽 {report.limit} 条（seed {report.seed}）")
    if picks:
        add("部分评测：" + "；".join(picks))
    if health is not None and not health.valid:
        lines.extend(_health_banner(health))
    elif health is not None:
        add(f"运行健康度：有效（{_ceiling_note(health)}）。{_counts_line('全部', health.overall)}")
    add("")
    add("== 总体（§13.2 四个维度的满分率）")
    add(_summary_line("全部", report.overall))
    if len(report.by_split) > 1:
        for name, summary in report.by_split.items():
            add(_summary_line(name, summary))
    add("")
    add("== §13.2 上线门槛（只看测试集：事实正确、合规满分率都 ≥ 95%）")
    gate = report.gate
    if gate is None:
        add("本次没有测试集用例（--split test 或 all 才做判定）")
    else:
        verdict = "通过" if gate.passes else "未通过"
        add(
            f"测试集 n={gate.n}：事实正确 {_pct(gate.factual)}，合规 {_pct(gate.compliance)}"
            f" → {verdict}"
        )
    if health is not None and not health.valid:
        lines.extend(_health_banner(health))  # 判定旁边再说一次：这个"通过 / 未通过"不算数
    if health is not None:
        add("")
        _render_health(report, health, add)
    add("")
    add("== 按问题类型")
    add(f"{'类型':<18}{'n':>3}  {'类型准确率':>8}  {'出推荐':>6}  {'事实':>6}  {'切题':>6}")
    for intent, s in report.by_intent.items():
        add(
            f"{intent:<18}{s.n:>3}  {_pct(s.intent_accuracy):>8}  {_pct(s.suggestion_rate):>6}"
            f"  {_pct(s.full_marks['factual']):>6}  {_pct(s.full_marks['on_topic']):>6}"
        )
    wrong = [
        f"{true} → {pred} ×{n}"
        for true, preds in sorted(report.confusion.items())
        for pred, n in sorted(preds.items())
        if pred != true
    ]
    if wrong:
        add("误判（真实 → 预测，未计入可接受的备选类型）：" + "；".join(wrong))
    add("")
    cal = report.calibration
    add(
        f"== §13.3 智能辅助阈值（target_precision = {cal.target_precision:.2f}，"
        f"用 {cal.samples_split} 的 {cal.n_samples} 条路由结果校准）"
    )
    if cal.samples_split == "test":
        add("注意：只跑了测试集，阈值是用测试集算的，仅供参考；正式阈值用开发集校准（§13.1）")
    left_out = _left_out_note(cal.left_out_fallback, cal.left_out_no_route)
    if left_out:
        add(f"只用配置的判断层做出的路由：{cal.samples_split} 里另有 {left_out}，没有参与校准")
    if health is not None and health.overall.route_lost_votes:
        add(
            f"注意：有 {health.overall.route_lost_votes} 条用例路由丢了票（见运行健康度），"
            "它们的置信度是按更少的票算的，刻度更粗；丢票多的话这里的阈值要打折扣看"
        )
    for intent, pick in cal.thresholds.items():
        if pick is None and cal.support is not None and not cal.support.get(intent):
            add(f"{intent:<18}没有样本（这个类型的用例都没参与校准）")
        elif pick is None:
            add(f"{intent:<18}达不到目标准确率")
        else:
            add(
                f"{intent:<18}阈值 {pick.threshold:.2f}  准确率 {_pct(pick.precision)}"
                f"  覆盖率 {_pct(pick.coverage)}  召回 {_pct(pick.recall)}  (n={pick.support})"
            )
    rel = report.reliability
    rel_left_out = rel.left_out_fallback + rel.left_out_no_route
    without = f"，不含没参与的 {rel_left_out} 条" if rel_left_out else ""
    add(f"可靠性（{rel.n_samples} 条{without}）：ECE = {_num(rel.ece)}")
    for b in rel.bins:
        add(
            f"  [{b.lower:.1f}, {b.upper:.1f}{']' if b.upper == 1.0 else ')'}  n={b.count:<3}"
            f" 平均置信度 {b.mean_confidence:.2f}  实际准确率 {_pct(b.accuracy)}"
        )
    add("")
    _render_latency(report, add)
    add("")
    _render_usage(report, add)
    if report.degraded_counts:
        add("")
        add(
            "== 降级（§12.2）：" + "、".join(f"{k} ×{v}" for k, v in report.degraded_counts.items())
        )
    add("")
    add(f"== 失败清单（{len(report.failures)} 条：任一维度不是满分）")
    for f in report.failures:
        _render_failure(f, add)
    if report.baseline is not None:
        add("")
        _render_baseline(report.baseline, add)
    if report.fingerprint is not None:
        add("")
        _render_fingerprint(report.fingerprint, add)
    return "\n".join(lines)


def _render_fingerprint(fp: EvalFingerprint, add: Add) -> None:
    code = fp.git_sha[:12] if fp.git_sha else "未知"
    if fp.git_dirty:
        code += "（有未提交的改动）"
    classifier = f"，分类器 {fp.classifier_sha256[:12]}" if fp.classifier_sha256 else ""
    model = fp.llm_model or "未知"
    where = f" @ {fp.llm_endpoint}" if fp.llm_endpoint else ""
    add(
        f"== 配置指纹 {fp.config_fingerprint[:12]}（路由问题 {fp.router_questions_ver}，"
        f"质检问题 {fp.qa_questions_ver}，生成环节 {fp.generation_ver or '未知'}，"
        f"判断层提示词 {fp.judge_prompt_ver or '未知'}，"
        f"判断层 {fp.decision_provider or '未知'}{classifier}，大模型 {model}{where}，代码 {code}）"
    )
    add("记录这次的结论（完整评测才行）：python -m rodit.ops record-eval --report <--json 的文件>")


# ================================================================ 命令行


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def _tolerance(text: str) -> float:
    value = float(text)
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError("必须在 [0, 1) 之间（0.02 = 2 个百分点）")
    return value


def _ratio(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:  # NaN 也过不了这个比较
        raise argparse.ArgumentTypeError("必须在 [0, 1] 之间（比例，不是百分数：0.02 = 2%）")
    return value


def _seconds(text: str) -> float:
    value = float(text)
    if not 0 <= value < float("inf"):
        raise argparse.ArgumentTypeError("必须是非负的秒数")
    return value


EXIT_CODES_HELP = (
    "退出码：0 正常；1 回归比较有指标下降；2 没有跑或没有跑完（配置错误、基线读不了或不能当基线、"
    "--json 的路径写不进去、连通性检查没过、连续多条用例的大模型调用全部失败而停下、最后写不进 "
    "JSON、跑完了但回归比较没有做成）；"
    "3 这一轮没有有效评到模型（--llm env：调用失败、被截断、读不出的用例超过上限，报告照常给出，"
    "JSON 不覆盖已有的文件、改写到旁边的 .invalid.json，同时有指标下降也是 3；"
    "--allow-degraded 时不返回 3）；130 被中断（结果没有保存）"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rodit.eval",
        description=(
            "在合成评测集上跑演示单元（离线替身或真实大模型），输出 §13.2 评分、上线门槛、"
            "§13.3 阈值校准、§12.1 耗时、§14 用量，可与基线做 §13.5 回归比较"
        ),
        epilog=EXIT_CODES_HELP,
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help=(
            "完整结果（JSON）：不带 PATH 输出到标准输出；带 PATH 写文件，文本报告照常输出"
            "（路径开跑前先验证写得进去）。这一轮没有有效评到模型时不覆盖已有的文件"
            "（包括 --baseline 那个）：改写到旁边的 <文件名>.invalid.json"
        ),
    )
    parser.add_argument(
        "--llm",
        choices=["offline", "env"],
        default="offline",
        help="offline：离线替身（默认，不联网）；env：环境变量 RODIT_LLM_* 配置的真实大模型",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="--llm env 时跳过开跑前的连通性检查（只在调试时用）",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help=(
            "--llm env 时这一轮没有有效评到模型也不返回 3（JSON 里 health.valid 照旧是 false），"
            f"连续 {STOP_AFTER_FAILED_CASES} 条用例调用全部失败时也不提前停下"
        ),
    )
    parser.add_argument(
        "--max-degraded",
        type=_ratio,
        default=DEFAULT_CEILING,
        metavar="FLOAT",
        help=(
            "运行健康度这一次用的上限（比例，默认 "
            f"{DEFAULT_CEILING:g} = {DEFAULT_CEILING * 100:g}%%）：出问题的用例占比超过它就不算"
            "评到了模型。记在 health.ceiling 里，只影响这次的判定和退出码（record-eval、--baseline "
            "对基线的检查都按默认上限重新判定）；路由丢票的上限固定是 "
            f"{LOST_VOTE_CEILING * 100:g}%%，不随它变"
        ),
    )
    parser.add_argument(
        "--min-interval",
        type=_seconds,
        default=0.0,
        metavar="SECONDS",
        help="相邻两条用例的开始至少隔这么多秒（被限流时用；等待不计入耗时，默认 0）",
    )
    parser.add_argument("--intent", choices=[i.value for i in Intent], help="只跑某个问题类型")
    parser.add_argument("--limit", type=_positive_int, help="随机抽 N 条（按 --seed，可复现）")
    parser.add_argument("--seed", type=int, default=0, help="--limit 抽样的随机种子（默认 0）")
    parser.add_argument(
        "--config",
        metavar="YAML",
        help=(
            f"引擎配置（如 config/rodit.yaml）；--llm env 没给时用环境变量 {CONFIG_ENV} 指的文件，"
            "都没有就按内置默认值跑（指纹和部署对不上，不能当上线依据）"
        ),
    )
    parser.add_argument(
        "--price-input", type=float, metavar="PRICE", help="输入单价：每 1000 token 的价格"
    )
    parser.add_argument(
        "--price-output", type=float, metavar="PRICE", help="输出单价：每 1000 token 的价格"
    )
    parser.add_argument(
        "--classifier",
        metavar="PATH",
        help="用训练好的路由分类器（python -m rodit.ops train-classifier 的输出）代替路由判断层",
    )
    parser.add_argument(
        "--baseline",
        metavar="PATH",
        help=(
            "基线（之前一次 --json 的输出）。开跑前检查：--llm env 跑出来的基线要是有效评到模型的"
            "一轮（按默认上限），和这次选出的用例要有共同的，否则不跑（退出码 2）"
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=_tolerance,
        default=0.0,
        help="回归容差（比例，默认 0：§13.5 指标不下降）",
    )
    return parser


def env_client(config: EngineConfig | None) -> OpenAICompatibleClient:
    """--llm env 的大模型客户端，和 app.production.build_production_unit 取同一个：配置里
    llm.base_url 和 llm.model 都写了用配置（超时 llm.timeout_s），否则用环境变量 RODIT_LLM_*
    （llm_env.client_from_env）。密钥从 llm.api_key_env 指的环境变量读，没设时抛 RuntimeError
    （信息里只有变量名）。评测记下的大模型（指纹）就是它。
    两条路都带上配置里的 llm.extra_body（模型服务特有的请求参数，如关闭思考模式）：它算在配置
    指纹里，评测发出去的请求要和这份配置的部署发出去的一样。"""
    llm = (config or EngineConfig()).llm
    if not (llm.base_url and llm.model):
        return client_from_env(api_key_env=llm.api_key_env, extra_body=llm.extra_body)
    client = OpenAICompatibleClient(
        base_url=llm.base_url,
        model=llm.model,
        api_key_env=llm.api_key_env,
        timeout_s=llm.timeout_s,
        extra_body=llm.extra_body,
    )
    if not os.environ.get(client.api_key_env, "").strip():
        raise RuntimeError(f"环境变量 {client.api_key_env} 未设置：没有大模型 API key")
    return client


def _progress_printer(err: TextIO) -> Progress:
    def show(index: int, total: int, case: EvalCase, trace: CaseTrace) -> None:
        ms = trace.latency_ms.get("total")
        failed = sum(1 for c in trace.llm_calls if c.get("error"))
        calls = f"，大模型调用 {len(trace.llm_calls)} 次" + (f"（失败 {failed}）" if failed else "")
        # 丢票和降级标记也打出来：有响应却用不上（被截断、读不出）时没有失败的调用，丢一票连降级
        # 都没有——几百条跑完才在报告里发现就晚了
        votes = trace.stats("route")
        lost = f"，路由只有 {votes.usable}/{votes.calls} 票算数" if _lost_votes(trace) else ""
        degraded = f"，降级：{'、'.join(trace.degraded)}" if trace.degraded else ""
        line = f"[{index}/{total}] {case.id} {_ms(ms)} 毫秒{calls}{lost}{degraded}"
        print(line, file=err, flush=True)

    return show


class _Tracker:
    """进度：记下跑完了几条（被中断时报出来）；show 给了就每跑完一条转给它（接大模型时打进度）。"""

    def __init__(self, show: Progress | None = None) -> None:
        self._show = show
        self.done = 0
        self.total: int | None = None

    def __call__(self, index: int, total: int, case: EvalCase, trace: CaseTrace) -> None:
        self.done, self.total = index, total
        if self._show is not None:
            self._show(index, total, case, trace)


def _run_config(args: argparse.Namespace) -> tuple[EngineConfig | None, str | None]:
    """这次评测的配置，和要告诉操作的人的话（没有就是 None）。

    --config 优先。--llm env 没给 --config 时用环境变量 RODIT_CONFIG 指的文件（部署和
    python -m rodit.ops 的任务都读它），连它也没有就是内置默认值（返回 None），并给出提醒。
    离线替身不读 RODIT_CONFIG：它的数字只是回归基线，不随运行环境变。"""
    if args.config:
        return load_config(args.config), None
    if args.llm != "env":
        return None, None
    path = os.environ.get(CONFIG_ENV, "").strip()
    if path:
        return load_config(path), f"配置：没有给 --config，用环境变量 {CONFIG_ENV} 指的 {path}"
    return None, DEFAULTS_WARNING


def _checked_baseline(args: argparse.Namespace) -> list[dict[str, Any]]:
    """--baseline 的逐条结果。能不能当基线在开跑前（第一次调用大模型之前）就查清，不能用抛
    ValueError——不然跑完一整轮才发现比不了，或者和一份兜底逻辑的数字比出"没有下降"（§13.5）：
    - 基线自己要是有效评到模型的一轮（baseline.baseline_problems）；
    - 和这次选出的用例（和 evaluate 同一个 select_cases）至少有一条共同的。"""
    data = read_baseline(args.baseline)
    problems = baseline_problems(data)
    if problems:
        raise ValueError(
            f"基线 {args.baseline} 不是一轮有效评到模型的结果，不能当回归基线（§13.5）："
            + "；".join(p.rstrip("。") for p in problems)
            + "。先重跑出一份有效的基线（--llm env 跑完退出码不是 3 的那种）"
        )
    rows: list[dict[str, Any]] = data["cases"]
    selected = select_cases(cases(args.split), intent=args.intent, limit=args.limit, seed=args.seed)
    known = {row["id"] for row in rows}
    if selected and not any(c.id in known for c in selected):  # 没有用例时由 evaluate 报"为空"
        raise ValueError(
            f"基线 {args.baseline} 和这次选出的 {len(selected)} 条用例没有共同的用例，无法比较"
            "（--split / --intent / --limit / --seed 选出的要和基线里的有重合）"
        )
    return rows


def _json_target(arg: str | None) -> Path | None:
    """--json PATH 的目标文件。开跑前就确认写得进去：建好上级目录，在旁边写一个临时文件再删掉——
    不然跑完一整轮（真实大模型是几千次付费调用）才发现路径不对，结果就丢了。
    不碰目标文件本身：--baseline 和 --json 可以是同一个文件（算数的一轮跑完才换成新的；不算数的
    一轮不覆盖它，见 _keep_reason）。写不进去抛 ValueError。
    符号链接先解析到它指的文件（换名换的是那个文件，链接留着）；设备、管道这类不是普通文件的目标
    （如 /dev/stdout）没法先写旁边再换名，不验证，跑完直接写。"""
    if arg is None or arg == "-":
        return None
    target = Path(arg).expanduser()
    try:
        if target.is_symlink():
            target = target.resolve()
        if target.is_dir():
            raise IsADirectoryError(str(target))
        if _special(target):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        probe = _temp_beside(target)
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        raise ValueError(
            f"完整结果写不进 {arg}（{type(e).__name__}）：换一个写得进去的 --json 路径再跑"
        ) from None
    return target


def _special(target: Path) -> bool:
    """已经存在、但不是普通文件（设备、管道）。"""
    return target.exists() and not target.is_file()


def _temp_beside(target: Path) -> Path:
    return target.with_name(f".{target.name}.{os.getpid()}.tmp")


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:  # 有一个不存在：按解析后的路径比
        return os.path.realpath(a) == os.path.realpath(b)


def _keep_reason(target: Path, baseline: str | None) -> str | None:
    """不算数的一轮为什么不能覆盖 --json 的目标文件（可以覆盖时返回 None）。

    目标就是这次的 --baseline，或者已经存在（多半是之前一轮的结果）：换成兜底逻辑的数字，原来那份
    就没了——下一次回归比的就是它。设备、管道这类目标谈不上覆盖。"""
    if _special(target):
        return None
    if baseline and _same_file(target, Path(baseline).expanduser()):
        return "它是这次的基线"
    if target.exists():
        return "它已经存在"
    return None


def _invalid_sibling(target: Path) -> Path:
    """不算数的一轮改写到的文件：和目标同一个目录（开跑前验证过写得进去），llm.json →
    llm.invalid.json。之前不算数的一轮留在那里的同名文件会被换掉。"""
    return target.with_name(f"{target.stem}.invalid{target.suffix}")


def _write_json(target: Path, payload: str) -> None:
    """先写旁边的临时文件再换名（os.replace）：写到一半出错，原来的文件（可能就是基线）不会坏。
    落单的代理项（模型文字里偶尔有，UTF-8 写不出）写成 \\uXXXX，仍然是合法的 JSON。"""
    if _special(target):
        target.write_text(payload + "\n", encoding="utf-8", errors="backslashreplace")
        return
    temp = _temp_beside(target)
    try:
        temp.write_text(payload + "\n", encoding="utf-8", errors="backslashreplace")
        os.replace(temp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            temp.unlink()
        raise


def _emit(text: str, stream: TextIO) -> None:
    """打印一段文字。流的编码写不出的字符（如模型文字里落单的代理项）换成反斜杠转义：
    不能因为一个字符，把跑完的一整轮结果丢在最后一步。"""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        safe = text.encode(encoding, errors="backslashreplace").decode(encoding)
    except LookupError:  # 流报了一个不认识的编码名
        safe = text.encode("utf-8", errors="backslashreplace").decode("utf-8")
    print(safe, file=stream)


THINKING_HINT = (
    "多半是模型开着思考模式，在 --config 的 llm.extra_body 里关掉（例子见 config/rodit.yaml）"
)
_LABEL_KINDS = {"429": "rate_limit", "402": "billing", "401": "auth", "403": "auth"}


def _label_kinds(labels: Any) -> set[str]:
    """失败原因的标签（health.error_label：http_<状态码>[:代码]、<类别>[:代码]、异常类型名）
    归成类别：429 → rate_limit，402 → billing，401 / 403 → auth，其他 4xx → bad_request，
    5xx → server；不是 HTTP 状态码的取冒号前面那一段。读不懂的输入当作没有标签。"""
    kinds: set[str] = set()
    try:
        names = [str(label) for label in labels or ()]
    except TypeError:
        return kinds
    for name in names:
        head = name.split(":", 1)[0].strip()
        status = head.removeprefix("http_")
        if status == head or not status.isdigit():
            kinds.add(head)
        elif status in _LABEL_KINDS:
            kinds.add(_LABEL_KINDS[status])
        elif status.startswith(("4", "5")):
            kinds.add("bad_request" if status.startswith("4") else "server")
        else:
            kinds.add(head)
    return kinds


def _abort_hint(labels: Any) -> str:
    """评测中止之后该做什么：按这几条用例的失败原因给，不是什么原因都说"加 --min-interval"。"""
    kinds = _label_kinds(labels)
    hints: list[str] = []
    if "rate_limit" in kinds:
        hints.append(
            "被限流（HTTP 429）加 --min-interval 放慢，或提高限额（有的服务额度用完也回 429，"
            "先看一眼余额）"
        )
    if "billing" in kinds:
        hints.append("余额或额度用完（HTTP 402）要充值或换有额度的账号，放慢重跑没有用")
    if kinds & {"auth", "bad_request", "malformed", "unusable"}:
        hints.append(
            "请求被拒或响应读不出来的，不带 --skip-preflight 重跑一次——连通性检查会先说明是"
            "哪一种请求、什么原因"
        )
    if kinds & {"truncated", "empty", "length"}:
        hints.append(f"回答被截断或为空的，{THINKING_HINT}")
    if not hints:
        return "先解决失败的原因再重跑"
    return "先解决失败的原因再重跑：" + "；".join(hints)


def main(
    argv: Sequence[str] | None = None,
    out: TextIO | None = None,
    *,
    err: TextIO | None = None,
    client_factory: Callable[[], LLMClient] | None = None,
) -> int:
    """命令行入口，返回退出码（见模块说明）。

    client_factory：--llm env 时构造大模型客户端（默认 env_client：和线上同一条规则，
    先看 --config 的 llm，再看环境变量；测试注入替身）。
    """
    args = build_parser().parse_args(argv)
    out = out or sys.stdout
    err = err or sys.stderr
    tracker = _Tracker(_progress_printer(err) if args.llm == "env" else None)
    config_note: str | None = None
    try:
        pricing = resolve_pricing(args.price_input, args.price_output)
        config, config_note = _run_config(args)
        if config_note:
            print(config_note, file=err, flush=True)
        if (
            config is not None
            and config.decision.provider == ProviderName.CLASSIFIER
            and not args.classifier
        ):
            # 不然路由悄悄换成关键词规则 / 大模型，回归测的不是要上线的分类器（§13.5）
            raise ValueError("配置的 decision.provider 是 classifier：用 --classifier 指定模型文件")
        baseline = _checked_baseline(args) if args.baseline else None  # 同样在开跑前
        json_target = _json_target(args.json)  # 第一次调用大模型之前就验证
        route_provider: DecisionProvider | None = None
        if args.classifier:
            try:
                route_provider = ClassifierProvider.load(args.classifier, fallback=RuleProvider())
            except ClassifierArtifactError as e:
                raise ValueError(f"路由分类器模型文件不能用：{e}") from None
        llm: LLMClient | None = None
        if args.llm == "env":
            llm = client_factory() if client_factory is not None else env_client(config)
            if not args.skip_preflight:
                checked = config
                if route_provider is not None:  # --classifier：路由不走大模型判断层，不查路由
                    base = config or EngineConfig()
                    decision = base.decision.model_copy(
                        update={"provider": ProviderName.CLASSIFIER}
                    )
                    checked = base.model_copy(update={"decision": decision})
                try:
                    summary = preflight(llm, checked)
                except LLMError as e:  # 客户端的错误信息里已经抹掉了密钥
                    raise RuntimeError(f"大模型连通性检查失败：{e}") from None
                for line in summary:  # 检查摘要；PreflightFailed 由下面的 RuntimeError 分支报出
                    print(line, file=err, flush=True)
        stop_early = llm is not None and not args.allow_degraded
        report = evaluate(
            args.split,
            intent=args.intent,
            limit=args.limit,
            seed=args.seed,
            llm=llm,
            config=config,
            pricing=pricing,
            progress=tracker,
            route_provider=route_provider,
            classifier_sha256=file_sha256(args.classifier) if args.classifier else None,
            ceiling=args.max_degraded,
            min_interval_s=args.min_interval,
            stop_after_failed=STOP_AFTER_FAILED_CASES if stop_early else None,
        )
    except EvalAborted as e:
        print(
            f"评测中止：{e}。{_abort_hint(getattr(e, 'labels', None))}；"
            "确实要跑完、看兜底逻辑的数字时加 --allow-degraded",
            file=err,
        )
        return EXIT_NOT_RUN
    except KeyboardInterrupt:
        done = "还没有跑完一条"
        if tracker.total:
            done = f"已经跑完 {tracker.done} / {tracker.total} 条"
        print(
            f"\n评测被中断：{done}，结果没有保存（没有报告，也没有 JSON）；重跑会从头开始",
            file=err,
        )
        return EXIT_INTERRUPTED
    except (ValueError, RuntimeError, OSError, yaml.YAMLError) as e:
        print(f"评测没有运行：{e}", file=err)
        return EXIT_NOT_RUN
    not_compared: str | None = None
    if baseline is not None:
        # 评测已经跑完：比较出了岔子（开跑前查过共同用例，正常到不了这里）也不能把报告丢掉
        try:
            check = compare_to_baseline(
                [c.model_dump(mode="json") for c in report.cases],
                baseline,
                tolerance=args.tolerance,
                source=args.baseline,
            )
            report = report.model_copy(update={"baseline": check})
        except ValueError as e:
            not_compared = (
                f"§13.5 回归比较没有做成：{e}。评测已经跑完，报告和完整结果照常给出"
                "（里面没有回归比较）"
            )
    code = _deliver(report, args, json_target, out, err, config_note)
    if not_compared is not None:
        print(f"{not_compared}（退出码 {code or EXIT_NOT_RUN}）", file=err)
        return code or EXIT_NOT_RUN
    return code


def _json_payload(report: EvalCliReport, stream: TextIO | None = None) -> str:
    """报告的 JSON。给了 stream（输出到标准输出）而它的编码写不出原文时，改成全 ASCII 转义：
    哪种编码都写得出，而且仍然是合法的 JSON。"""
    data = report.model_dump(mode="json")
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    if stream is not None:
        try:
            payload.encode(getattr(stream, "encoding", None) or "utf-8")
        except (UnicodeEncodeError, LookupError):
            return json.dumps(data, ensure_ascii=True, indent=2)
    return payload


def _deliver(
    report: EvalCliReport,
    args: argparse.Namespace,
    json_target: Path | None,
    out: TextIO,
    err: TextIO,
    config_note: str | None,
) -> int:
    """跑完之后：先把完整结果写出去（JSON 先于文本报告：渲染文本出错也不丢结果），再定退出码。

    不算数的一轮（health.valid 为 False，加没加 --allow-degraded 都一样）不覆盖已有的文件：
    --json 的目标是这次的基线或者已经存在时，写到旁边的 _invalid_sibling，并在标准错误和文本
    报告末尾写明。"""
    saved = True
    health = report.health
    invalid = health is not None and not health.valid
    regressed = report.baseline is not None and not report.baseline.passed
    if args.json == "-":
        print(_json_payload(report, out), file=out)
    else:
        written = args.json  # 实际写到哪里（给人看的）
        note = ""
        if json_target is not None:
            keep = _keep_reason(json_target, args.baseline) if invalid else None
            if keep is not None:
                json_target = _invalid_sibling(json_target)
                written = str(json_target)
                note = f"（这一轮没有有效评到模型，没有覆盖 {args.json}：{keep}）"
                print(
                    f"这一轮没有有效评到模型：不覆盖 {args.json}（{keep}），"
                    f"完整结果改写到旁边的 {written}",
                    file=err,
                )
            try:
                _write_json(json_target, _json_payload(report))
            except OSError as e:
                saved = False
                print(f"完整结果写不进 {written}：{type(e).__name__}", file=err)
        _emit(render_text(report), out)
        if json_target is not None and saved:
            _emit(f"\n完整结果（JSON）已写入 {written}{note}", out)
    if config_note:  # 进度刷了几百行，结尾再说一次这次用的是哪份配置
        print(config_note, file=err)
    if not saved:
        return EXIT_NOT_RUN
    if health is not None and invalid:
        if not args.allow_degraded:
            labels = [label for p in health.by_purpose.values() for label in p.errors]
            slow_down = (
                "；被限流（HTTP 429）的调用，重跑时加 --min-interval 放慢"
                if "rate_limit" in _label_kinds(labels)
                else ""
            )
            also = (
                f"。回归比较也有指标下降，但兜底逻辑的数字比不出回归，退出码仍是 {EXIT_DEGRADED}"
                if regressed
                else ""
            )
            print(
                f"这一轮没有有效评到模型（退出码 {EXIT_DEGRADED}）："
                + "；".join(r.rstrip("。") for r in health.reasons)
                + f"{slow_down}{also}。确认就是要这份结果时加 --allow-degraded",
                file=err,
            )
            return EXIT_DEGRADED
        print(
            "这一轮没有有效评到模型（原因见报告开头）；加了 --allow-degraded，退出码不因此改变",
            file=err,
        )
    if regressed:
        print(f"§13.5 回归比较有指标下降超过容差（退出码 {EXIT_REGRESSION}）", file=err)
        return EXIT_REGRESSION
    return 0


__all__ = [
    "CaseRecord",
    "DemoHarness",
    "EvalCliReport",
    "LLMInfo",
    "build_parser",
    "demo_harness",
    "env_client",
    "evaluate",
    "main",
    "render_text",
]
