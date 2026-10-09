"""评测结论的指纹与上线门槛的测试集依据（方案 §13.5 "换大模型、改提示词、升级判断层之前，
先在测试集上跑回归"、§13.2 测试集门槛、§13.4 影子 → 智能辅助）。

- 指纹（EvalFingerprint）：配置、提示词、判断层、分类器、大模型——
  - 配置的 config_fingerprint（core.config：阈值、判断层、生成参数……都在配置里）；
  - 路由 / 质检问题定义的版本（core.questions，改文字就变）；
  - 生成环节的版本（generation_version：GENERATION_MODULES 这几个模块的源码——
    rodit.generation 整个包（提示词骨架、要求、模板话术、依据的排法、输出解析），加上文字会进
    生成提示词和模板的 core 模块：依据行的格式 core.evidence_line、类型名 core.enums、平台名和
    缺依据的说法 core.evidence、最近几轮对话 core.states、个人信息遮挡 core.pii、
    时间写法 core.clock；改了任何一处就变）；
  - 判断层提示词的版本（judge_prompt_version：JUDGE_PROMPT_MODULES 的源码——大模型判断层的
    system 提示词、user 模板和回答格式 decision.llm、输出解析和计票 decision.voting、
    题型和回答校验 core.decision、路由 / 质检的 state（core.states、core.pii、qa.masking）。
    质检一直用大模型判断层，分类器路由时也比较这一项）；
  - 路由判断层（decision_provider）和路由分类器模型文件的 SHA-256（classifier 路由时）；
  - 大模型：模型名和接口地址（只留协议和主机；端口是协议的默认端口时不写，https://主机 和
    https://主机:443 是同一个接口）。
  模块按名字排序、按 importlib 找到源文件来算。列表以外的代码（阈值判断、质检硬规则……）
  改了指纹不变：它们的改动靠代码版本留痕（git，拿得到时；部署镜像里没有 .git 时由环境变量
  RODIT_GIT_SHA 提供），代码版本只作记录，same_setup 不比较。
  python -m rodit.eval --json 的报告带上它：评测按实际跑的记——用的大模型客户端、--classifier 时
  判断层是 classifier（配置里写的是别的也一样）、分类器文件的 SHA-256。
- 线上的指纹（current_fingerprint）按部署的规则解析，和线上单元实际用的一致：大模型和
  app.production.build_production_unit 同一条规则（配置里 llm.base_url、llm.model 都写了用配置，
  否则都取环境变量 RODIT_LLM_BASE_URL / RODIT_LLM_MODEL）；decision.provider 是 classifier 时
  分类器用 serve 启动时实际加载的模型文件的 SHA-256：同一进程里的调用方（上线状态、切换阶段的
  门槛）传 loaded_classifier_sha256，取自 ClassifierProvider.file_sha256，文件启动后被换掉也照旧
  是加载的那个；别的进程（运维任务 record-eval、acceptance-report）看不到 serve 加载的是什么，
  读环境变量 RODIT_CLASSIFIER_PATH 指向的文件（下次重启 serve 加载的就是它）。拿不到的项是 None，
  None 和什么都对不上（不知道线上用的是什么，就不能说跑过回归）。
- 评测结论（EvalVerdict）：python -m rodit.ops record-eval --report 报告.json 从报告里取出
  指纹、测试集判定（§13.2 事实正确、合规满分率 ≥ 95%）、回归比较结果、用的大模型，记成系统事件
  eval.verdict（shop_id 为空，不属于任何店铺）。只收完整的评测（不收 --intent / --limit 的
  部分评测，要有测试集；条数要等于评测集这个 split 的条数，跑到一半停下的不收）。
- 运行健康度（rodit.eval.health）：真实大模型跑的评测，调用失败、被截断、读不出时链路按 §12.2
  降级，照样跑完、给出一组数字，量的却是兜底逻辑。这样的报告不收：按报告 health 里的计数用固定的
  DEFAULT_CEILING / LOST_VOTE_CEILING 重新算一遍 assess（不看报告自己写的 valid、reasons、
  ceiling），给出原因的抛 ModelNotEvaluated（被时间预算截断本身不算，截断时没有一条大模型草稿
  过完质检的 budget_unjudged 超过上限才算）；报告里没有 health（旧版本生成的、没跑完的）、
  计数和报告对不上的也不收。收下的结论带着这些计数（EvalVerdict.health_overall / health_test），
  describe_verdict 写出来；以前记下的结论里没有后加的计数（如 budget_unjudged）时按 0 读。
  离线替身不调大模型，没有健康度可看，照旧。
- 测试集依据（test_set_evidence）：上线状态、验收报告从 eval.verdict 里挑指纹和线上一致
  （same_setup：除代码版本外每一项都一样）、用真实大模型（--llm env）跑的最近一次结论：测试集
  达标且回归没有下降算通过。离线替身的结论只作回归基线（rodit.eval.cli 的说明），不算上线
  依据。没有能用的结论时写明原因（没有结论、哪几项和线上对不上、只有离线的）。

结论里只有指纹、计数、比例、模型名和接口主机，没有买家数据、没有密钥（配置里只有环境变量名，
接口地址去掉了用户名密码、路径和查询参数）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.config import EngineConfig, config_fingerprint, llm_endpoint
from rodit.core.enums import ProviderName
from rodit.core.models import AuditEvent
from rodit.core.questions import QA_QUESTIONS_VER, ROUTER_QUESTIONS_VER
from rodit.eval.health import DEFAULT_CEILING, LOST_VOTE_CEILING, HealthCounts, assess
from rodit.feedback.rollout import TestSetEvidence

GIT_SHA_ENV = "RODIT_GIT_SHA"
"""部署镜像里没有 .git 时由构建流程写进来的代码版本。"""
LLM_BASE_URL_ENV = "RODIT_LLM_BASE_URL"
LLM_MODEL_ENV = "RODIT_LLM_MODEL"
"""配置里没写 llm.base_url / llm.model 时线上用的大模型（OpenAICompatibleClient.from_env）。"""
CLASSIFIER_PATH_ENV = "RODIT_CLASSIFIER_PATH"
"""decision.provider 是 classifier 时 serve 加载的路由分类器模型文件（rodit.ops.serve）。"""
GENERATION_MODULES: tuple[str, ...] = (
    "rodit.generation",  # 整个包：提示词骨架和要求、模板话术、依据的排法、输出解析
    "rodit.core.clock",  # fmt_time：依据行、模板里的时间写法
    "rodit.core.enums",  # INTENT_LABELS、POLICY_LABELS：提示词和模板里的类型名
    "rodit.core.evidence",  # PLATFORM_NAMES、缺依据时的说法
    "rodit.core.evidence_line",  # 依据行 "[id] 标签：内容" 的格式
    "rodit.core.pii",  # 放进提示词之前的个人信息遮挡
    "rodit.core.states",  # 最近几轮对话（recent_turns）
)
"""生成环节的版本（generation_version）算哪些模块的源码：包算包里各 .py（不含子包）。"""
JUDGE_PROMPT_MODULES: tuple[str, ...] = (
    "rodit.decision.llm",  # 判断层的 system 提示词、user 模板、回答格式
    "rodit.decision.voting",  # 输出解析和计票
    "rodit.core.decision",  # 题型（Jev 格式）和回答校验
    "rodit.core.states",  # 路由 / 质检的 state
    "rodit.core.pii",  # state 里的个人信息遮挡
    "rodit.qa.masking",  # 质检 state 里的对话、依据、回答的遮挡
)
"""判断层提示词的版本（judge_prompt_version）算哪些模块的源码。"""
GIT_TIMEOUT_S = 5
SHORT = 12  # 报告、原因里显示的指纹长度
DEFAULT_PORTS = {"https": 443, "http": 80}
"""协议的默认端口：接口地址里写不写都是同一个接口（endpoint_of 不写）。"""

HOW_TO_RECORD = (
    "用线上配置和线上的大模型跑 python -m rodit.eval --llm env --config <部署配置> --json 报告.json"
    "（环境变量 RODIT_LLM_* 和部署一致；分类器路由时加 --classifier <线上的模型文件>），"
    "再 python -m rodit.ops record-eval --report 报告.json"
    "（record-eval 也要带部署的 --config / RODIT_CONFIG 和 RODIT_LLM_*：它拿自己这个进程的配置当"
    "线上来比，没给时比的是内置默认值）"
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalFingerprint(_Model):
    """一次评测（或线上）对应的配置、生成环节、判断层和大模型。same_setup 比较除代码版本以外的
    各项；以前记的结论没有后加的几项（None），和线上对不上，要重新评测。"""

    config_fingerprint: str
    router_questions_ver: str
    qa_questions_ver: str
    generation_ver: str | None = None  # 生成环节的版本（generation_version）
    judge_prompt_ver: str | None = None  # 判断层提示词的版本（judge_prompt_version）
    decision_provider: str | None = None  # 路由判断层（ProviderName 的值）
    llm_model: str | None = None  # 大模型的模型名
    # 大模型接口的协议和主机（不是默认端口时带端口），没有路径、用户名密码
    llm_endpoint: str | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None  # 有没有提交的改动（环境变量给的版本为 None）
    classifier_sha256: str | None = None  # 路由分类器模型文件（classifier 路由时）


def git_revision(
    cwd: str | Path | None = None, *, environ: Mapping[str, str] | None = None
) -> tuple[str | None, bool | None]:
    """(代码版本, 有没有未提交的改动)。先看 RODIT_GIT_SHA，再问 git；拿不到返回 (None, None)。"""
    env = os.environ if environ is None else environ
    fixed = env.get(GIT_SHA_ENV, "").strip()
    if fixed:
        return fixed, None
    where = str(cwd) if cwd is not None else str(Path(__file__).resolve().parent)
    try:
        head = subprocess.run(
            ["git", "-C", where, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
        if head.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "-C", where, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    sha = head.stdout.strip() or None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return sha, dirty


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_FILE_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def _cached_file_sha256(path: str) -> str | None:
    """线上每次看上线状态都要算分类器文件的 SHA-256：按（路径、修改时间、大小）缓存最近一个。
    读不了返回 None。"""
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
        cached = _FILE_SHA_CACHE.get(key)
        if cached is None:
            cached = file_sha256(path)
            _FILE_SHA_CACHE.clear()  # 线上只有一个分类器文件
            _FILE_SHA_CACHE[key] = cached
        return cached
    except OSError:
        return None


def _module_sources(name: str) -> list[tuple[str, bytes]]:
    """一个模块的源码：[(标识, 内容)]。包是包里各 .py（按文件名排序，不含子包），
    标识 "包名/文件名"；普通模块是它的 .py，标识是模块名。找不到、不是 .py 源文件抛
    OSError / ImportError。"""
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ModuleNotFoundError(name)
    if spec.submodule_search_locations is not None:
        package = resources.files(name)
        files = sorted(
            (f for f in package.iterdir() if f.is_file() and f.name.endswith(".py")),
            key=lambda f: f.name,
        )
        if not files:
            raise FileNotFoundError(name)
        return [(f"{name}/{f.name}", f.read_bytes()) for f in files]
    if not spec.origin or not spec.origin.endswith(".py"):
        raise FileNotFoundError(name)
    return [(name, Path(spec.origin).read_bytes())]


@lru_cache(maxsize=8)
def source_version(modules: tuple[str, ...]) -> str | None:
    """几个模块源码的版本：按模块名排序，逐个文件摘标识、内容（换行统一成 LF，各跟一个 NUL
    分隔），SHA-256 前 12 位。改了任何一处版本就变（§13.5 改提示词之前先在测试集上跑回归）。
    任何一个读不到返回 None（和什么都对不上）。进程里按模块列表缓存：源码在进程运行期间不变。"""
    try:
        digest = hashlib.sha256()
        for name in sorted(set(modules)):
            for label, data in _module_sources(name):
                digest.update(label.encode())
                digest.update(b"\0")
                digest.update(data.replace(b"\r\n", b"\n"))
                digest.update(b"\0")
    except (OSError, ImportError, ValueError):
        return None
    return digest.hexdigest()[:SHORT]


def generation_version() -> str | None:
    """生成环节的版本：GENERATION_MODULES 的源码（source_version）。"""
    return source_version(GENERATION_MODULES)


def judge_prompt_version() -> str | None:
    """判断层提示词的版本：JUDGE_PROMPT_MODULES 的源码（source_version）。"""
    return source_version(JUDGE_PROMPT_MODULES)


def endpoint_of(base_url: str | None) -> str | None:
    """接口地址只留协议和主机（端口）：去掉用户名密码、路径、查询参数（和
    eval.llm_env.describe_endpoint 同一口径）。端口是这个协议的默认端口（DEFAULT_PORTS：https 的
    443、http 的 80）时不写：https://主机 和 https://主机:443 是同一个接口，比较时要相等。
    没有主机时返回 None。"""
    if not base_url:
        return None
    parts = urlsplit(base_url.strip())
    if not parts.hostname:
        return None
    try:
        number = parts.port
    except ValueError:  # 端口不是数字：不显示
        number = None
    port = f":{number}" if number and number != DEFAULT_PORTS.get(parts.scheme) else ""
    return f"{parts.scheme}://{parts.hostname}{port}"


def running_llm(
    config: EngineConfig, environ: Mapping[str, str] | None = None
) -> tuple[str | None, str | None]:
    """线上单元实际用的大模型：（模型名，接口的协议和主机）。和 app.production.build_production_unit
    同一条规则（core.config.llm_endpoint）：配置里 llm.base_url 和 llm.model 都写了用配置，否则
    两项都取环境变量 RODIT_LLM_BASE_URL / RODIT_LLM_MODEL。拿不到的项是 None。"""
    base_url, model, _ = llm_endpoint(config.llm, environ)
    return (model or None), endpoint_of(base_url)


def running_classifier_sha256(
    config: EngineConfig,
    environ: Mapping[str, str] | None = None,
    *,
    loaded: str | None = None,
) -> str | None:
    """decision.provider 是 classifier 时线上路由分类器模型文件的 SHA-256：loaded 是同一进程里
    serve 实际加载的（ClassifierProvider.file_sha256），给了就用它；没给（别的进程）按
    RODIT_CLASSIFIER_PATH 读文件——下次重启 serve 加载的那个。不是 classifier、没设、读不了都
    返回 None。"""
    if config.decision.provider != ProviderName.CLASSIFIER:
        return None
    if loaded:
        return loaded
    env = os.environ if environ is None else environ
    path = env.get(CLASSIFIER_PATH_ENV, "").strip()
    return _cached_file_sha256(path) if path else None


def fingerprint_of(
    config: EngineConfig,
    *,
    llm_model: str | None,
    llm_endpoint: str | None,
    classifier_sha256: str | None = None,
    git: bool = False,
    environ: Mapping[str, str] | None = None,
) -> EvalFingerprint:
    """给定大模型和分类器的指纹（评测按实际跑的填）。判断层取 config.decision.provider：
    评测用 --classifier 时调用方先把它换成 classifier。git=True 时带上代码版本。"""
    sha, dirty = git_revision(environ=environ) if git else (None, None)
    return EvalFingerprint(
        config_fingerprint=config_fingerprint(config),
        router_questions_ver=ROUTER_QUESTIONS_VER,
        qa_questions_ver=QA_QUESTIONS_VER,
        generation_ver=generation_version(),
        judge_prompt_ver=judge_prompt_version(),
        decision_provider=config.decision.provider.value,
        llm_model=llm_model,
        llm_endpoint=llm_endpoint,
        git_sha=sha,
        git_dirty=dirty,
        classifier_sha256=classifier_sha256,
    )


def current_fingerprint(
    config: EngineConfig,
    *,
    git: bool = False,
    classifier: str | Path | None = None,
    loaded_classifier_sha256: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> EvalFingerprint:
    """线上这套部署的指纹：大模型按部署规则解析（running_llm）；分类器按 running_classifier_sha256——
    同一进程里的调用方传 serve 实际加载的 loaded_classifier_sha256，否则读 RODIT_CLASSIFIER_PATH
    （classifier 给了就算这个文件）。线上比较用 git=False（same_setup 不比较代码版本）。
    environ 默认 os.environ：线上服务、运维任务和线上单元读的是同一份环境变量。"""
    model, endpoint = running_llm(config, environ)
    if classifier:
        classifier_sha = file_sha256(classifier)
    else:
        classifier_sha = running_classifier_sha256(config, environ, loaded=loaded_classifier_sha256)
    return fingerprint_of(
        config,
        llm_model=model,
        llm_endpoint=endpoint,
        classifier_sha256=classifier_sha,
        git=git,
        environ=environ,
    )


def same_setup(a: EvalFingerprint, b: EvalFingerprint) -> bool:
    """除代码版本外每一项都一样（见 differences）。"""
    return differences(a, b) == []


def _known_and_equal(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and a == b


def _classifier_routed(fp: EvalFingerprint) -> bool:
    return fp.decision_provider == ProviderName.CLASSIFIER.value or fp.classifier_sha256 is not None


def _llm_label(fp: EvalFingerprint, *, running: bool = False) -> str:
    if fp.llm_model is None and fp.llm_endpoint is None:
        if running:
            return (
                "未知：配置里没写 llm.base_url 和 llm.model，环境变量 "
                f"{LLM_BASE_URL_ENV} / {LLM_MODEL_ENV} 也没设"
            )
        return "没记录"
    return f"{fp.llm_model or '未知模型'} @ {fp.llm_endpoint or '未知地址'}"


def differences(verdict: EvalFingerprint, running: EvalFingerprint) -> list[str]:
    """评测结论（verdict）和线上（running）对不上的项，中文说明；空表示一致。后加的几项有一边
    不知道（None）也算对不上：旧结论没记、线上解析不出来，都不能说"这套跑过回归"。"""
    a, b = verdict, running
    out = []
    if a.config_fingerprint != b.config_fingerprint:
        out.append("配置指纹")
    if a.router_questions_ver != b.router_questions_ver:
        out.append("路由问题版本")
    if a.qa_questions_ver != b.qa_questions_ver:
        out.append("质检问题版本")
    if not _known_and_equal(a.generation_ver, b.generation_ver):
        out.append("生成提示词 / 模板的版本")
    if not _known_and_equal(a.judge_prompt_ver, b.judge_prompt_ver):
        out.append("判断层提示词 / 输出解析的版本")
    if not _known_and_equal(a.decision_provider, b.decision_provider):
        evaluated, deployed = a.decision_provider or "没记录", b.decision_provider or "未知"
        out.append(f"路由判断层（评测 {evaluated}，线上 {deployed}）")
    if (_classifier_routed(a) or _classifier_routed(b)) and not _known_and_equal(
        a.classifier_sha256, b.classifier_sha256
    ):
        evaluated = a.classifier_sha256[:SHORT] if a.classifier_sha256 else "没记录"
        deployed = (
            b.classifier_sha256[:SHORT] if b.classifier_sha256 else f"未知（{CLASSIFIER_PATH_ENV}）"
        )
        out.append(f"路由分类器模型文件（评测 {evaluated}，线上 {deployed}）")
    if not (
        _known_and_equal(a.llm_model, b.llm_model)
        and _known_and_equal(a.llm_endpoint, b.llm_endpoint)
    ):
        out.append(f"大模型（评测 {_llm_label(a)}，线上 {_llm_label(b, running=True)}）")
    return out


# ================================================================ 评测结论（eval.verdict）


class EvalVerdict(_Model):
    """一次完整评测的结论（eval.verdict 系统事件的 detail）。没有买家数据、没有密钥。"""

    fingerprint: EvalFingerprint
    llm_mode: Literal["offline", "env"]
    model: str | None = None
    decision_provider: str
    split: str
    n: int = Field(ge=0)
    test_n: int = Field(ge=0)
    threshold: float
    factual: float
    compliance: float
    gate_passes: bool
    baseline_passed: bool | None = None  # 没做回归比较为 None
    report_sha256: str
    # 运行健康度的计数（rodit.eval.health）：全部用例、测试集。真实大模型跑的才有；离线替身跑的、
    # 以前记下的结论没有（None）
    health_overall: HealthCounts | None = None
    health_test: HealthCounts | None = None


class ModelNotEvaluated(ValueError):
    """报告里的这一轮没有有效评到大模型（rodit.eval.health.assess 给出了原因），不能记成评测结论。
    reasons 是 assess 的原因，一句一条；异常说明里把它们都写上。"""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(reasons)
        super().__init__(self.reasons)

    def __str__(self) -> str:
        head = "这一轮没有有效评到大模型（量到的是降级后的兜底逻辑），不能记成评测结论："
        return head + "".join(self.reasons)


class _Digest(BaseModel):
    """评测报告（rodit.eval.cli.EvalCliReport 的 JSON）里 record-eval 用到的部分。"""

    model_config = ConfigDict(extra="ignore")


class _GateDigest(_Digest):
    n: int
    threshold: float
    factual: float
    compliance: float
    passes: bool


class _LLMDigest(_Digest):
    mode: Literal["offline", "env"]
    model: str | None = None
    decision_provider: str


class _BaselineDigest(_Digest):
    passed: bool


class _CountsDigest(HealthCounts):
    """报告的 health 里一组计数：字段照 HealthCounts，多出来的键不管。"""

    model_config = ConfigDict(extra="ignore")


class _HealthDigest(_Digest):
    """报告的 health（rodit.eval.health.RunHealth 的 JSON）里 record-eval 用到的部分：只要计数。
    valid、reasons、ceiling 不读（_run_health 重新算）；评测命令行以后加的字段不管。"""

    overall: _CountsDigest
    test: _CountsDigest | None = None


class _ReportDigest(_Digest):
    split: Literal["dev", "test", "all"]
    n: int
    intent: str | None = None
    limit: int | None = None
    gate: _GateDigest | None = None
    llm: _LLMDigest
    health: Any = None  # 运行健康度：只有真实大模型跑的报告才看（_run_health）
    baseline: _BaselineDigest | None = None
    fingerprint: EvalFingerprint | None = None


def _bad_fields(e: ValidationError, *prefix: str) -> str:
    """校验没过的字段（a.b.c 的写法，去重、排序），prefix 是它们在报告里的上一级。"""
    return "、".join(
        sorted({".".join(str(p) for p in (*prefix, *err["loc"])) for err in e.errors()})
    )


def _run_health(
    health: Any, *, split: str, n: int, test_n: int
) -> tuple[HealthCounts, HealthCounts]:
    """真实大模型跑的报告的 health →（全部用例的计数，测试集的计数）。split、n、test_n 是这份报告的
    split、条数和测试集条数。没有这一块、读不出来、和报告对不上的抛 ValueError；这一轮没有有效评到
    大模型的抛 ModelNotEvaluated——按固定的 DEFAULT_CEILING / LOST_VOTE_CEILING 重新算 assess，
    报告里的 valid、reasons、ceiling 不作数（评测时可以放宽上限，记成上线依据时不行）。"""
    if health is None:
        raise ValueError(
            "报告里没有运行健康度（health）：是旧版本生成的，或者评测没有跑完，看不出大模型有没有被"
            "有效评到。用本版本的 python -m rodit.eval --llm env --json 重新跑"
        )
    try:
        block = _HealthDigest.model_validate(health)
    except ValidationError as e:
        fields = _bad_fields(e, "health")
        raise ValueError(f"不是完整的评测报告（字段不对：{fields}）") from None
    overall, test = block.overall, block.test
    if test is None and split == "test":
        test = overall  # 只跑测试集：全部用例就是测试集
    if overall.cases != n or test is None or test.cases != test_n:
        seen = "没有计数" if test is None else f" {test.cases} 条"
        raise ValueError(
            f"报告里的运行健康度（health）和这份报告对不上（health：全部 {overall.cases} 条、"
            f"测试集{seen}；报告：全部 {n} 条、测试集 {test_n} 条）："
            "用本版本的 python -m rodit.eval --llm env --json 重新跑"
        )
    reasons = assess(overall, test, ceiling=DEFAULT_CEILING, lost_vote_ceiling=LOST_VOTE_CEILING)
    if reasons:
        raise ModelNotEvaluated(reasons)
    return HealthCounts(**overall.model_dump()), HealthCounts(**test.model_dump())


def verdict_from_report(data: Any, *, raw: bytes | None = None) -> EvalVerdict:
    """评测报告（--json 的内容，已解析）→ 结论。raw 是文件原文（算报告的 SHA-256；没给时按
    规范化的 JSON 算）。不能作为测试集依据的报告抛 ValueError（原因写明）：不是完整的评测
    （部分评测、没有测试集、条数和评测集对不上），或者真实大模型跑的报告里没有运行健康度、
    那一轮没有有效评到模型（ModelNotEvaluated，见 _run_health）。离线替身跑的不看运行健康度。"""
    # 只有这里用评测集：线上服务 import 本模块（上线状态）时不必加载它
    from rodit.eval.dataset import cases

    if not isinstance(data, Mapping):
        raise ValueError("不是评测报告（应为 python -m rodit.eval --json 的输出）")
    if not data.get("fingerprint"):
        raise ValueError("报告里没有配置指纹：用本版本的 python -m rodit.eval --json 重新生成")
    try:
        digest = _ReportDigest.model_validate(dict(data))
    except ValidationError as e:
        raise ValueError(f"不是完整的评测报告（字段不对：{_bad_fields(e)}）") from None
    if digest.intent is not None or digest.limit is not None:
        raise ValueError("部分评测（--intent / --limit）不能作为上线结论：请跑完整的评测集")
    if digest.gate is None or digest.split == "dev":
        raise ValueError("报告里没有测试集判定：要用 --split test 或 all（§13.1 留出的测试集）")
    size = len(cases(digest.split))
    if digest.n != size:
        raise ValueError(
            f"报告不完整：评测集 {digest.split} 共 {size} 条，报告里是 {digest.n} 条"
            "（评测没有跑完，或者跑的不是这一版评测集）：请把完整的评测集跑完"
        )
    assert digest.fingerprint is not None
    health_overall = health_test = None
    if digest.llm.mode == "env":
        health_overall, health_test = _run_health(
            digest.health, split=digest.split, n=digest.n, test_n=digest.gate.n
        )
    if raw is None:
        raw = json_bytes(data)
    return EvalVerdict(
        fingerprint=digest.fingerprint,
        llm_mode=digest.llm.mode,
        model=digest.llm.model,
        decision_provider=digest.llm.decision_provider,
        split=digest.split,
        n=digest.n,
        test_n=digest.gate.n,
        threshold=digest.gate.threshold,
        factual=digest.gate.factual,
        compliance=digest.gate.compliance,
        gate_passes=digest.gate.passes,
        baseline_passed=digest.baseline.passed if digest.baseline is not None else None,
        report_sha256=hashlib.sha256(raw).hexdigest(),
        health_overall=health_overall,
        health_test=health_test,
    )


def json_bytes(data: Any) -> bytes:
    """规范化的 JSON（键排序、紧凑、UTF-8）。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def verdict_event(verdict: EvalVerdict, *, at: datetime, actor: str = "system") -> AuditEvent:
    """结论 → 系统事件（shop_id 为空）。"""
    return AuditEvent(
        shop_id=None,
        actor=actor,
        action=AuditAction.EVAL_VERDICT.value,
        detail=verdict.model_dump(mode="json"),
        created_at=at,
    )


def parse_verdicts(events: Iterable[AuditEvent]) -> list[tuple[datetime, EvalVerdict]]:
    """系统事件 → (记录时间, 结论)，按时间升序。店铺事件、字段不对的事件跳过。"""
    out: list[tuple[datetime, EvalVerdict]] = []
    for e in events:
        if e.shop_id is not None or e.action != AuditAction.EVAL_VERDICT.value:
            continue
        try:
            out.append((e.created_at, EvalVerdict.model_validate(e.detail)))
        except ValidationError:
            continue
    return sorted(out, key=lambda x: x[0])


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _when(at: datetime) -> str:
    return f"{at.astimezone(CST):%Y-%m-%d %H:%M}"


def describe_health(c: HealthCounts) -> str:
    """一组运行健康度计数的说明，每项是"出问题的条数/分母"：路由退回（关键词规则）、质检不可用、
    生成没有可用草稿这三项总是写；路由丢票、引擎异常、被时间预算截断、截断时没有草稿过完质检
    （budget_unjudged，被时间预算截断的一部分）有才写。"""
    parts = [
        f"路由退回 {c.route_fallback}/{c.cases}",
        f"质检不可用 {c.qa_unavailable}/{c.qa_judged}",
        f"生成没有可用草稿 {c.generation_unusable}/{c.generation_attempted}",
    ]
    rarer = (
        ("路由丢票", c.route_lost_votes),
        ("引擎异常", c.engine_errors),
        ("被时间预算截断", c.budget_cut),
        ("截断时没有草稿过完质检", c.budget_unjudged),
    )
    parts += [f"{label} {count}/{c.cases}" for label, count in rarer if count]
    return "、".join(parts)


def describe_verdict(at: datetime, v: EvalVerdict) -> str:
    """结论的一句话说明（中文，没有买家数据）。带运行健康度的结论（真实大模型跑的）最后写上
    全部用例的计数（describe_health）。"""
    model = v.model or "未知模型"
    text = (
        f"{_when(at)} 的评测（{model}，测试集 {v.test_n} 条）：事实正确 {_pct(v.factual)}、"
        f"合规 {_pct(v.compliance)}（门槛 ≥ {_pct(v.threshold)}）"
    )
    if v.baseline_passed is False:
        text += "；和基线比有指标下降（§13.5）"
    passed = v.gate_passes and v.baseline_passed is not False
    text += "，达标" if passed else "，没达标"
    if v.health_overall is not None:
        text += f"；运行健康度：{describe_health(v.health_overall)}"
    return text


def test_set_evidence(events: Iterable[AuditEvent], running: EvalFingerprint) -> TestSetEvidence:
    """从 eval.verdict 系统事件里挑测试集依据（见模块说明）。events 一般是
    AuditLog.list_system_events(action="eval.verdict")；running 是线上的指纹
    （current_fingerprint）。"""
    verdicts = parse_verdicts(events)
    base = {"running_fingerprint": running.config_fingerprint}
    if not verdicts:
        return TestSetEvidence(reason=f"还没有评测结论：{HOW_TO_RECORD}", **base)
    real = [(at, v) for at, v in verdicts if v.llm_mode == "env"]
    matching = [(at, v) for at, v in real if same_setup(v.fingerprint, running)]
    if not matching:
        at, latest = verdicts[-1]
        if latest.llm_mode != "env":
            reason = (
                f"最近的评测结论（{_when(at)}）是离线替身跑的（只作回归基线，不是上线判断），"
                f"也没有和线上一致的真实大模型结论：{HOW_TO_RECORD}"
            )
        else:
            diffs = "、".join(differences(latest.fingerprint, running))
            reason = (
                f"最近的评测结论（{_when(at)}）和线上不一致（{diffs}；评测的配置指纹 "
                f"{latest.fingerprint.config_fingerprint[:SHORT]}，线上 "
                f"{running.config_fingerprint[:SHORT]}）：换了配置、问题定义、提示词、判断层或"
                "大模型之后要重新评测并记录"
            )
        return TestSetEvidence(
            reason=reason,
            verdict_at=at,
            config_fingerprint=latest.fingerprint.config_fingerprint,
            **base,
        )
    at, v = matching[-1]
    return TestSetEvidence(
        passed=v.gate_passes and v.baseline_passed is not False,
        detail=describe_verdict(at, v),
        verdict_at=at,
        config_fingerprint=v.fingerprint.config_fingerprint,
        **base,
    )


test_set_evidence.__test__ = False  # type: ignore[attr-defined]  # 名字以 test_ 开头，不是 pytest 用例


__all__ = [
    "CLASSIFIER_PATH_ENV",
    "GENERATION_MODULES",
    "GIT_SHA_ENV",
    "HOW_TO_RECORD",
    "JUDGE_PROMPT_MODULES",
    "LLM_BASE_URL_ENV",
    "LLM_MODEL_ENV",
    "EvalFingerprint",
    "EvalVerdict",
    "ModelNotEvaluated",
    "current_fingerprint",
    "describe_health",
    "describe_verdict",
    "differences",
    "endpoint_of",
    "file_sha256",
    "fingerprint_of",
    "generation_version",
    "git_revision",
    "judge_prompt_version",
    "parse_verdicts",
    "running_classifier_sha256",
    "running_llm",
    "same_setup",
    "source_version",
    "test_set_evidence",
    "verdict_event",
    "verdict_from_report",
]
