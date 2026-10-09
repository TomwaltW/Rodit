"""问答引擎编排（方案 §4.2、§5.3–§5.7、§11、§12.1–§12.3、§13.4）：懂 → 找 → 写 → 判。

AssistEngine.handle(msg) 处理一条买家消息，所有工作都在 core.context.shop_scope(msg.shop_id) 里
（大模型成本按店铺归属，§12.4）。顺序：

1. 店铺检查：不存在或已停用 → 不处理、不保存消息，返回带 no_suggestion_reason 的结果（§8.4）；
2. 去重：ConversationRepository.add_message 返回 False → 返回 None（重复消息什么都不做、不留痕）；
   不是买家消息（接入层本不该送来）也返回 None；
   乱序（§8.3 按会话保证顺序）：会话里已经有比本条晚的买家消息（平台推送晚到）→ 本条照常存进
   会话，但不单独回答旧问题，改为重新处理会话里最新的那条买家消息（本条按时间落在它的 Turn 里
   或历史里），结果等同按顺序到达；重新处理时不出 auto_reply（最新那条可能已经自动回复过，
   全自动下由此转人工）；
3. 刷新推荐：同一会话之前 SHOWN 的推荐记录改成 EXPIRED（§5.3 新消息到达时刷新推荐）；
   撤回（on_recall，应用层在撤回事件里调用）：被撤回的那条消息触发的 SHOWN 推荐改成 EXPIRED，
   返回会话里仍在展示的推荐卡片给侧边栏刷新（§5.3 撤回的消息不再进上下文）；
4. 会话理解（engine.understanding.ConversationBuilder）→ ConversationContext；
   买家图片（§15 Phase 3，engine.images.ImageReader）：消息一到就记下它的图片标识，会话理解之后识别
   当前买家消息里的图片，结果放进 ctx.image_findings、描述接在当前买家消息后面（路由、检索、生成、
   质检都看得到），给客服一条图片摘要提醒；识别失败、超时记降级 "vision"，照常往下走；
5. 路由 → RouteDecision，按 core.pipeline.ACTION_PLANS[route.action] 执行：
   - 检索（KnowledgeUnavailable → 不推荐，§12.2）；
   - ANSWER / HIGH_RISK_ANSWER：证据包没有依据、没有澄清问题、也没有查不到的工具 → 不生成，
     不推荐并给出缺失知识（§5.7 "依据不足、也无法澄清 → 提示缺哪类知识"），
     注入了 FeedbackRepository 时记 KnowledgeGap。模板类问题（core.evidence.LLM_FREE_INTENTS）
     不需要依据，不走这个判断；
   - 生成 → 每条草稿过质检 → 大模型草稿不通过时，带 QAResult.reasons 重新生成
     （最多 qa_gate.max_regenerations 次，§5.5、§11.5）→ 保留通过的草稿；
     判断层质检不可用（QAResult.degraded）的草稿照旧不放行、原因照记，但不算重写的理由，
     它的原因也不带进重写；
     FAQ / 模板草稿不通过说明标准回答本身有问题，不重写，丢弃并提醒客服（改知识库）；
     一条都没通过 → 不推荐，no_suggestion_reason 写质检原因（§5.5 给客服显示原因）；
     质检永远要跑（§5.5 每条推荐都要过两道关），计划里的 check 只是说明；
   - SPLIT_MULTI 按 route.parts 分条：每条用
     route.model_copy(update={"intent": 该条, "parts": [], "action": HIGH_RISK_ANSWER 或 ANSWER})
     单独检索、生成、质检，每条最多一张卡片（其余路由结果最多 ShopConfig.suggestion_count 张）；
     投诉类的分条不生成，改为提醒人工；
   - 并行（§12.1，engine.concurrency）：各条的"找 → 写 → 判"并行跑，同一条的几份草稿并行质检；
     结果按原顺序收回，输出和顺序执行一致。工作线程只把要写的审计、降级记在这一条自己的记录里，
     审计、推荐记录、缺失知识都在调用方线程、收回结果之后按原顺序写。
     质检按"还差几张卡片就先判几份"分批：判哪些草稿和顺序执行时一样，不多调判断层；
   - HUMAN_PRIORITY / NO_SUGGESTION：不推荐，只给提醒；TRANSFER_HUMAN：transfer_to_human=True；
   - 店铺在 ShopConfig.disabled_intents 里关闭的问题类型，回答类路由（含分条）不推荐。
6. 推荐卡片（engine.cards，§5.6）写 SuggestionRecord（SHOWN；影子模式 SHADOW），
   question_text = mask_pii(当前买家消息)，回填 SuggestionCard.suggestion_id；
   半自动（engine.semi_auto，§5.6、§13.4）：智能辅助下第一张卡片满足条件时 prefill=True
   （店铺开了这个类型的半自动、规则 6 的正常回答、置信度够、卡片为高、这一条没有草稿被质检拦下过
   ——重写后才通过的也不填，没有提醒和降级），
   客服仍要按发送；记录写不进去（没有推荐编号）时去掉 prefill。SUGGEST 审计另加 prefill
   （打了 prefill 的推荐编号，只在有时写）；
7. 全自动（§5.6、§12.2、§12.3）：以下全部满足才出 auto_reply——
   ctx.mode == AUTO、ctx.config.auto_reply_enabled、route.auto_eligible、不是分条推荐、
   第一张卡片通过质检且置信度为高、卡片没有风险提示、结果没有提醒（含连续没听懂）、
   本条消息没有任何降级、len(前缀 + 文本) ≤ ctx.config.max_answer_chars。
   auto_reply = 第一张卡片、文本前加 generation.auto_reply_prefix（AI 生成内容标识，
   整个系统只在这里加）。不满足时退回智能辅助（照常给推荐），并且 transfer_to_human=True——
   §5.7 全自动模式下质检不通过、依据不足、投诉等都要转人工，§12.2 全自动出故障由客服接管。
   全自动生效时段（engine.modes，§15 Phase 3 先在非工作时间试行）：消息是 AUTO、店铺开了全自动，
   但现在是客服工作时间（或没填工作时间）→ 生效模式是 ASSIST（路由也按智能辅助走），照常给推荐，
   并且 transfer_to_human=True（机器人接待的会话交给在岗客服，§12.2），engine.result 另加可选键
   auto_gate（拦下原因，core.audit.RESULT_AUTO_GATE）。
   影子模式（§13.4）：mode=SHADOW，推荐只用于事后对比，app 层不得展示；不出 auto_reply、不转人工。
   影子记录每轮一套：乱序时重算最新一轮（run.late），这一轮（按最新买家消息的 msg_id）已经有
   SHADOW 记录就不再写第二套（事后对比不重复计数），结果照常返回（卡片没有 suggestion_id），
   SUGGEST 审计带 shadow_duplicate=True。
   店铺或配置读不出来（知识库故障等，不是已知停用）而消息是 AUTO 时，生效模式无从判断：
   按全自动的兜底处理，transfer_to_human=True（§12.2 全自动模式下任何故障 → 客服接管）。

时间预算（§12.1、§12.2，core.budget）：每条消息在 deadline_scope(latency.message_budget_s,
min_llm_remaining_s=latency.min_llm_remaining_s) 里处理（EngineConfig.latency）。
- 生成器、质检自己判断还能不能调大模型（core.budget.llm_allowed），来不及就退回 FAQ / 模板；
  生成器本来要写大模型草稿（FAQ / 模板之外还有名额）、因为剩余时间不够没调时告诉引擎
  （core.budget.note_llm_skipped，引擎用 llm_skip_scope 包住每次生成），这一条和"大模型草稿
  来不及送判断层质检"一样记 budget；FAQ / 模板已经占满名额、本来就不调大模型的不记；
- 截止时间过了：不再检索、生成、重写；没做完的大模型草稿丢掉（生成器不再等它，引擎也不判它；
  截止之后才判完的大模型草稿同样丢掉），已经通过质检的 FAQ / 模板卡片保留；
  剩余时间不够再调一次大模型时，同样不重写、不送判断层质检；
- 引擎最多等到截止时间 + concurrency.JOIN_GRACE_S；还没做完的分条整条丢掉。它的线程会在后台
  跑完，引擎和质检自己的记录都不在后台写：handle() 返回之后没有迟到的审计、推荐记录、
  decision_log（质检截止之后不写）。后台线程调用的平台工具、大模型照常留下副作用：工具结果写进
  缓存、大模型用量计量（llm.MeteredLLMClient 的 llm.usage，调用确实发生了，成本要算到店铺上，
  §12.4）。handle 里提交到线程池的任务（分条、质检、生成器并行生成的几份草稿、路由投票）都经
  core.budget.submit_in_context 记在引擎的 BackgroundWork 里（按店铺，handle 里的
  background_scope）：取消订购删数据之前 AssistService.revoke_shop 用 wait_background 等它们
  做完，迟到的写入一并删掉（§8.4）；
- 以上任何一种"时间不够、丢了东西"都记降级 "budget"，engine.result 另加
  budget_exceeded=True（core.audit.RESULT_BUDGET_EXCEEDED）。

降级（§12.2）：每个环节单独兜住，handle() 对合法消息永不抛错。降级标记（EngineResult.degraded）：
- "storage"      会话 / 推荐 / 缺失知识记录读写失败（仍然出推荐，缺 suggestion_id）
- "knowledge"    知识库不可用（KnowledgeUnavailable）→ 不推荐
- "understanding" 会话理解意外失败 → 不推荐
- "router"       Router 意外抛错（正常情况下 Router 自己降级、不抛）→ 不推荐
- "decision"     判断层不可用，路由退回关键词规则（RouteDecision.degraded）
- "decision:qa"  判断层质检不可用，只跑了硬规则（QAResult.degraded）
- "retriever"    检索意外抛错（非 KnowledgeUnavailable、非串店）→ 该条不推荐
- "tenant_isolation" 会话理解或检索发现串店（TenantIsolationError，§9.2、§12.4 串店检查）→ 不推荐
                 （检索时是该条不推荐），另记一条 tenant.violation 审计，detail 只有 stage
                 （understand / retrieve），不记任何数据
- "generator"    生成抛错 → 保留已通过的草稿
- "qa"           质检抛错 → 该草稿不推荐（没有验证过的文字不能推荐），保留已通过的
- "budget"       时间预算不够，有步骤没做或草稿被丢掉（见上）
- "tool:<kind>"  平台工具不可用：会话理解查订单（tool:order），或 EvidenceBundle.unavailable
- "vision"       买家图片识别失败或超时（engine.images）→ 照常出推荐，提醒客服自己看图
- "audit"        审计写入失败
- "engine"       其他意外异常 → 不推荐
和卡片内容有关的降级（decision、decision:qa、tool:*、storage、knowledge、vision）会写进卡片的风险提示。

缺失知识（EngineResult.missing_knowledge）= 各证据包的 missing；没有依据、证据包又没说缺什么时记
"knowledge:{intent}"。KnowledgeGap 只记知识类缺失（不记 core.evidence.MISSING_ORDER /
MISSING_ITEM 这类"会话里没有订单 / 商品"的上下文缺失），有推荐时缺的部分也记（提醒商家补齐）。

审计（core.audit.AuditAction，actor="system"，detail 里没有买家原文）：
UNDERSTAND、ROUTE 各一条；RETRIEVE、GENERATE、QA 每次调用 / 每条分条一条；有卡片时 SUGGEST 一条；
出 auto_reply 时 AUTO_REPLY 一条；每条处理过的消息（重复消息除外）恰好一条 RESULT，detail 键固定为：
- "intent"      str | None  路由的问题类型（没有路由结果时为 None）
- "action"      str | None  路由结果（RouteAction 取值）
- "rule"        int | None  命中的 §6.6 规则序号
- "suggestions" int         推荐卡片数（影子模式也算）
- "hard_hits"   list[str]   本条消息全部质检结果里的硬规则命中（规则名，一次命中一项），
                            含 format / length 这类自动修正（core.models.NON_BLOCKING_HARD_RULES）
- "qa_failed"   int         质检不通过的草稿数（含重新生成的；质检抛错不算）
- "degraded"    list[str]   同 EngineResult.degraded
- "latency_ms"  dict[str, int]  同 EngineResult.latency_ms
- "missing"     list[str]   同 EngineResult.missing_knowledge
另有 "no_suggestion_reason"、"qa_checked"、"qa_reasons"、"transfer_to_human"；
键的完整定义见 core.audit.ResultKey（常量 RESULT_DETAIL_KEYS，报表按同一份读）。
超过时间预算时另加可选键 "budget_exceeded"（True，core.audit.OPTIONAL_RESULT_DETAIL_KEYS）；
全自动不在生效时段被拦下时另加可选键 "auto_gate"（core.audit.RESULT_AUTO_GATE，同在
OPTIONAL_RESULT_DETAIL_KEYS 里）。

耗时（EngineResult.latency_ms，§12.1、§12.4）：键固定为 understand / route / retrieve / generate /
qa / total / since_received（LATENCY_KEYS），毫秒整数，没跑的环节为 0。分条并行时同一环节取各条里
最长的（墙钟口径，和 total 对得上）；同一条里重写的几轮累加。各环节用 time.perf_counter 量墙钟耗时
（单调时钟），不用注入的 Clock——Clock 只管业务时间戳（记录的 created_at 等）。
understand 含店铺检查、会话理解、图片识别；total 是整条消息从进入 handle 到出结果；since_received
（core.audit.LATENCY_SINCE_RECEIVED）是从平台收到消息（交给 handle 的那条消息的 received_at）到
出结果：Clock.now() − received_at，含排队、会话锁等待；时钟精度不够或两边时间对不齐时不小于 total。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Any

from rodit.core import audit as _audit
from rodit.core import budget
from rodit.core.audit import (
    DEGRADED_BUDGET,
    DEGRADED_TENANT_ISOLATION,
    LATENCY_SINCE_RECEIVED,
    RESULT_AUTO_GATE,
    RESULT_BUDGET_EXCEEDED,
    AuditAction,
    ResultKey,
)
from rodit.core.clock import Clock
from rodit.core.config import EngineConfig
from rodit.core.context import shop_scope
from rodit.core.enums import (
    INTENT_LABELS,
    Confidence,
    DraftKind,
    Intent,
    Mode,
    Role,
    RouteAction,
    SuggestionStatus,
)
from rodit.core.errors import ConcurrencyConflict, KnowledgeUnavailable, TenantIsolationError
from rodit.core.evidence import LLM_FREE_INTENTS, MISSING_ITEM, MISSING_ORDER, missing_any
from rodit.core.models import (
    AuditEvent,
    ConversationContext,
    Draft,
    EngineResult,
    EvidenceBundle,
    ItemRef,
    KnowledgeGap,
    OrderRef,
    QAResult,
    RouteDecision,
    Shop,
    StandardMessage,
    SuggestionCard,
    SuggestionRecord,
)
from rodit.core.pii import mask_pii
from rodit.core.pipeline import ACTION_PLANS, ActionPlan, risk_hint, soothe_with_answer
from rodit.core.ports import (
    AnswerGenerator,
    AuditSink,
    Cache,
    ConversationRepository,
    EvidenceRetriever,
    FeedbackRepository,
    ImageUnderstanding,
    KnowledgeRepository,
    QualityGate,
    Router,
    SuggestionRepository,
    ToolGateway,
)
from rodit.engine.cards import (
    HARD_RULE_LABELS,
    blocking_rules,
    build_card,
    card_from_record,
    dedupe,
    dropped_draft_alert,
)
from rodit.engine.concurrency import (
    NESTED_GRACE_S,
    BackgroundWork,
    Outcome,
    background_scope,
    run_inline,
    run_ordered,
)
from rodit.engine.images import (
    ALERT_VISION_FAILED,
    DEGRADED_VISION,
    MAX_IMAGES,
    VISION_BUDGET_S,
    ImageReader,
    ImageReading,
    describe_for_audit,
)
from rodit.engine.modes import resolve_mode
from rodit.engine.semi_auto import prefill_blocker
from rodit.engine.understanding import ConversationBuilder, ShopInactive

logger = logging.getLogger(__name__)

ACTOR = "system"

# ---------------------------------------------------------------- 降级标记

DEGRADED_STORAGE = "storage"
DEGRADED_KNOWLEDGE = "knowledge"
DEGRADED_UNDERSTANDING = "understanding"
DEGRADED_ROUTER = "router"
DEGRADED_DECISION = "decision"
DEGRADED_DECISION_QA = "decision:qa"
DEGRADED_RETRIEVER = "retriever"
DEGRADED_GENERATOR = "generator"
DEGRADED_QA = "qa"
DEGRADED_AUDIT = "audit"
DEGRADED_ENGINE = "engine"
# DEGRADED_TENANT_ISOLATION（"tenant_isolation"）、DEGRADED_BUDGET（"budget"）定义在 core.audit


RESULT_DETAIL_KEYS = _audit.RESULT_DETAIL_KEYS  # 定义在 core.audit.ResultKey
_STAGES = ("understand", "route", "retrieve", "generate", "qa")
LATENCY_KEYS = (*_STAGES, "total", LATENCY_SINCE_RECEIVED)

# 串店审计（tenant.violation）的 stage：只记在哪一步拦下的，不记任何数据（§9.2、§12.4）
TENANT_STAGE_UNDERSTAND = "understand"
TENANT_STAGE_RETRIEVE = "retrieve"

# ---------------------------------------------------------------- 给客服看的原因与提醒

REASON_SHOP_INACTIVE = "店铺未开通或已停用，不处理（§8.4）"
REASON_KNOWLEDGE = "知识库暂不可用，本条不推荐（§12.2）"
REASON_UNDERSTANDING = "会话理解失败，本条不推荐，请客服自行处理"
REASON_ROUTER = "问题类型判断失败，本条不推荐，请客服自行处理"
REASON_RETRIEVER = "查找依据失败，本条不推荐，请客服自行处理"
REASON_NO_EVIDENCE = "依据不足、也无法澄清，本条不推荐，请补充相关知识（§5.7）"
REASON_GENERATOR = "回答生成失败，本条不推荐，请客服自行处理（§12.2）"
REASON_QA_ERROR = "质检暂不可用，未经质检的回答不推荐（§5.5）"
REASON_NO_DRAFT = "没有可推荐的回答，请客服自行处理"
REASON_QA_FAILED = "质检未通过："
REASON_HUMAN_PRIORITY = "买家要求人工或在投诉，不推荐，请人工优先处理（§5.7）"
REASON_NO_SUGGESTION = "没有把握判断问题类型，不推荐，请客服自行处理（§6.6 规则 8）"
REASON_NO_SUGGESTION_DEGRADED = "判断层暂不可用，本条不推荐，请客服自行处理（§12.2）"
REASON_TRANSFER = "已转人工（§5.7）"
REASON_DISABLED = "店铺关闭了这类问题的推荐"
REASON_ENGINE = "问答引擎内部错误，本条不推荐，请客服自行处理（§12.2）"
REASON_TENANT = "检测到跨店铺数据，本条不推荐，请客服自行处理（§9.2）"
REASON_BUDGET = "处理超时，本条不推荐，请客服自行处理（§12.1）"

ALERT_HUMAN_PRIORITY = "买家要求人工或在投诉，请优先处理"
ALERT_COMPLAINT_PART = "买家消息里有投诉，请人工优先处理"
ALERT_UNRESOLVED = "买家连续{n}轮表示没听懂或答非所问，建议客服接管"

_ANSWER_ACTIONS = frozenset({RouteAction.ANSWER, RouteAction.HIGH_RISK_ANSWER})
_CONTEXT_MISSING = frozenset({MISSING_ORDER, MISSING_ITEM})
_LATE_CHECK_LIMIT = 20  # 判断乱序时读的最近消息条数


class _NotInTime:
    """质检结果的占位：这份大模型草稿来不及判（时间预算），丢掉，不算质检不通过。"""

    def __repr__(self) -> str:
        return "<not in time>"


_NOT_IN_TIME: Any = _NotInTime()


def missing_for_intent(intent: Intent) -> str:
    """没有依据、也说不清缺哪条时的缺失标记（core.evidence.missing_any）。"""
    return missing_any(intent)


# ---------------------------------------------------------------- 一条消息的处理状态


_Op = tuple[str, Any]  # ("degrade", 代码) / ("audit", (AuditAction, detail))


@dataclass
class _Unit:
    """一次"找 → 写 → 判"：一个路由结果，或多问题消息里的一条（§6.6 规则 5）。

    在工作线程里跑（concurrency.run_ordered），只改自己：要记的降级、要写的审计按发生顺序记在
    ops 里，耗时、质检计数也记在这里；主线程收回之后按原顺序并入 _Run（_merge_unit），
    所以审计的先后和顺序执行时一样。没按时做完的分条，主线程不再读它（换成一个空的替身）。
    """

    route: RouteDecision
    intent_confidence: float
    limit: int  # 最多几张卡片
    bundle: EvidenceBundle = field(default_factory=EvidenceBundle)
    passed: list[tuple[Draft, QAResult]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    reason: str | None = None
    ops: list[_Op] = field(default_factory=list)
    latency: dict[str, float] = field(default_factory=lambda: dict.fromkeys(_STAGES, 0.0))
    hard_hits: list[str] = field(default_factory=list)
    qa_checked: int = 0
    qa_failed: int = 0
    qa_reasons: list[str] = field(default_factory=list)
    budget_cut: bool = False  # 时间预算不够，有步骤没做或草稿被丢掉

    @property
    def intent(self) -> Intent:
        return self.route.intent

    def degrade(self, code: str) -> None:
        self.ops.append(("degrade", code))

    def audit(self, action: AuditAction, detail: dict[str, Any]) -> None:
        self.ops.append(("audit", (action, detail)))

    def add_latency(self, stage: str, since: float) -> None:
        self.latency[stage] += time.perf_counter() - since

    def cut(self) -> None:
        """时间预算不够：记 budget 降级（每条只记一次）。"""
        if not self.budget_cut:
            self.budget_cut = True
            self.degrade(DEGRADED_BUDGET)

    def placeholder(self, reason: str) -> _Unit:
        """同一条的空替身（原来的对象还在工作线程里跑，主线程不能再读它）。"""
        return _Unit(self.route, self.intent_confidence, self.limit, reason=reason)


@dataclass
class _Run:
    """一条消息的处理状态。每次 handle 新建一个，引擎本身无状态，可以并发调用。
    只在调用方线程里读写（分条的工作线程不碰它）。"""

    msg: StandardMessage
    started: float
    received_at: datetime | None = None  # 交给 handle 的那条消息（乱序时 msg 会换成最新那条）
    mode: Mode = Mode.ASSIST
    # 消息是 AUTO、但店铺 / 配置读不出来、生效模式无从判断：按全自动兜底转人工（§12.2）
    auto_unverified: bool = False
    # 消息是 AUTO、店铺开了全自动，但不在全自动生效时段（engine.modes）：退回智能辅助并转人工
    auto_gate: str | None = None
    # 乱序：本条晚到，改为重新处理会话里最新的买家消息（run.msg 已换成那条），不出 auto_reply
    late: bool = False
    ctx: ConversationContext | None = None
    route: RouteDecision | None = None
    degraded: list[str] = field(default_factory=list)
    context_notes: list[str] = field(default_factory=list)  # 影响所有卡片的降级（理解、路由）
    latency: dict[str, float] = field(default_factory=lambda: dict.fromkeys(_STAGES, 0.0))
    alerts: list[str] = field(default_factory=list)
    hard_hits: list[str] = field(default_factory=list)
    qa_checked: int = 0
    qa_failed: int = 0
    qa_reasons: list[str] = field(default_factory=list)
    result: EngineResult | None = None

    def degrade(self, code: str) -> None:
        if code not in self.degraded:
            self.degraded.append(code)

    def add_latency(self, stage: str, since: float) -> None:
        self.latency[stage] += time.perf_counter() - since

    def elapsed_ms(self) -> int:
        return round((time.perf_counter() - self.started) * 1000)

    def latency_ms(self, now: datetime | None = None) -> dict[str, int]:
        """各环节耗时 + total + since_received（now 是 Clock.now()；拿不到时按 total 算）。"""
        out = {k: round(v * 1000) for k, v in self.latency.items()}
        total = self.elapsed_ms()
        out["total"] = total
        since = total
        if now is not None and self.received_at is not None:
            since = max(total, round((now - self.received_at).total_seconds() * 1000))
        out[LATENCY_SINCE_RECEIVED] = since
        return out


# ---------------------------------------------------------------- 引擎


class AssistEngine:
    """问答引擎（§5）：一条买家消息 → 推荐卡片 / 提醒 / 转人工 / 缺失知识（EngineResult）。

    各步骤只通过 core.ports 的协议注入；引擎不直接读平台写接口（问答链路只读，§7）。
    gaps 为空时不记知识缺失提醒（结果里仍有 missing_knowledge）。
    monotonic：时间预算用的单调时钟（秒，core.budget.Deadline 的 now），测试可以注入假的；
    外层已经有截止时间时（调用方设的）要用同一个时钟。
    交给线程池的任务按店铺记在 BackgroundWork 里，wait_background 等某家店的全部做完（取消订购用）。
    vision / image_cache：买家图片理解和记图片标识、识别结果的短时缓存（engine.images；单元组装时
    传单元缓存，取消订购一并删除）；vision 为空时不识别，只提醒客服买家发了图片。
    """

    def __init__(
        self,
        *,
        config: EngineConfig,
        knowledge: KnowledgeRepository,
        conversations: ConversationRepository,
        suggestions: SuggestionRepository,
        tools: ToolGateway,
        router: Router,
        retriever: EvidenceRetriever,
        generator: AnswerGenerator,
        gate: QualityGate,
        audit: AuditSink,
        clock: Clock,
        gaps: FeedbackRepository | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        vision: ImageUnderstanding | None = None,
        image_cache: Cache | None = None,
        vision_budget_s: float = VISION_BUDGET_S,
        max_images: int = MAX_IMAGES,
    ) -> None:
        self._config = config
        self._monotonic = monotonic
        self._conversations = conversations
        self._suggestions = suggestions
        self._router = router
        self._retriever = retriever
        self._generator = generator
        self._gate = gate
        self._audit_sink = audit
        self._clock = clock
        self._gaps = gaps
        self._builder = ConversationBuilder(knowledge, conversations, tools, config, clock)
        self._background = BackgroundWork()
        self._images = ImageReader(
            vision,
            image_cache,
            clock,
            max_images=max_images,
            budget_s=vision_budget_s,
            monotonic=monotonic,
        )

    @property
    def builder(self) -> ConversationBuilder:
        return self._builder

    def wait_background(self, shop_id: str, timeout: float | None = None) -> bool:
        """等这家店留在后台的任务（到截止时间被丢下、还在跑的分条和质检）全部做完；
        timeout 秒之内做完（或本来就没有）返回 True。取消订购删数据之前调
        （AssistService.revoke_shop），调用方要保证期间不会有这家店的新消息（持有写方闸门）。"""
        return self._background.wait_idle(shop_id, timeout)

    # ------------------------------------------------------------ 入口

    def handle(self, msg: StandardMessage) -> EngineResult | None:
        """处理一条买家消息。重复消息（或非买家消息）返回 None；对合法消息永不抛错（§12.2）。

        整条消息在时间预算（EngineConfig.latency）里处理，见模块说明。"""
        run = _Run(msg=msg, started=time.perf_counter(), received_at=msg.received_at)
        latency = self._config.latency
        with (
            shop_scope(msg.shop_id),
            background_scope(self._background, msg.shop_id),
            budget.deadline_scope(
                latency.message_budget_s,
                min_llm_remaining_s=latency.min_llm_remaining_s,
                now=self._monotonic,
            ),
        ):
            try:
                return self._handle(run)
            except Exception as exc:  # noqa: BLE001 — 最后一道兜底
                self._warn("handle", exc)
                if run.result is not None:
                    return run.result
                run.degrade(DEGRADED_ENGINE)
                return self._finish_safely(run)

    # ------------------------------------------------------------ 主流程

    def _handle(self, run: _Run) -> EngineResult | None:
        msg = incoming = run.msg
        if msg.role != Role.BUYER:
            return None

        # 1. 店铺
        t = time.perf_counter()
        shop: Shop | None = None
        shop_error: str | None = None
        try:
            shop = self._builder.load_shop(msg.shop_id)
        except ShopInactive:
            shop_error = REASON_SHOP_INACTIVE
        except Exception as exc:  # noqa: BLE001
            self._warn("shop", exc)
            run.degrade(DEGRADED_KNOWLEDGE)
            shop, shop_error = None, REASON_KNOWLEDGE
            run.auto_unverified = msg.mode == Mode.AUTO
        if shop is not None:
            # 先定生效模式：后面任何一步失败，影子模式的结果也不能被展示
            try:
                decided = resolve_mode(msg.mode, self._builder.shop_config(shop), self._clock.now())
                run.mode, run.auto_gate = decided.mode, decided.auto_gate
            except Exception as exc:  # noqa: BLE001 — 配置算不出来时会话理解会再失败一次
                self._warn("shop_config", exc)
                run.auto_unverified = msg.mode == Mode.AUTO
        run.add_latency("understand", t)
        if shop_error == REASON_SHOP_INACTIVE:
            return self._finish(run, reason=shop_error)

        # 2. 去重
        try:
            added = self._conversations.add_message(msg)
        except Exception as exc:  # noqa: BLE001 — 存不进去也照常出推荐（§12.2）
            self._warn("add_message", exc)
            run.degrade(DEGRADED_STORAGE)
            added = True
        if not added:
            return None
        self._images.remember(msg)  # 同一 Turn 里后到的消息要用到这条的图片（engine.images）
        if shop is None:
            return self._finish(run, reason=shop_error)
        latest = self._newer_buyer_message(msg)
        if latest is not None:  # 乱序：本条已存进会话，改为重新处理最新那条（§8.3）
            logger.info("消息晚到，重新处理会话里最新的买家消息")
            run.msg = msg = latest
            run.late = True

        # 3. 新消息到达：刷新推荐
        self._expire_shown(run)

        # 4. 会话理解
        t = time.perf_counter()
        ctx: ConversationContext | None = None
        reason: str | None = None
        try:
            understanding = self._builder.build_with_notes(msg, shop=shop)
            ctx = understanding.context
            run.auto_gate = understanding.auto_gate  # 和 ctx.mode 同一时刻判断的
            for code in understanding.degraded:
                run.degrade(code)
                run.context_notes.append(code)
        except KnowledgeUnavailable:
            run.degrade(DEGRADED_KNOWLEDGE)
            reason = REASON_KNOWLEDGE
        except TenantIsolationError as exc:  # 串店（§9.2）：不推荐，只记在哪一步拦下的
            self._warn("understand", exc)
            run.degrade(DEGRADED_TENANT_ISOLATION)
            self._audit(run, AuditAction.TENANT_VIOLATION, _tenant_detail(TENANT_STAGE_UNDERSTAND))
            reason = REASON_TENANT
        except Exception as exc:  # noqa: BLE001
            self._warn("understand", exc)
            run.degrade(DEGRADED_UNDERSTANDING)
            reason = REASON_UNDERSTANDING
        run.add_latency("understand", t)
        if ctx is None:
            return self._finish(run, reason=reason)
        ctx, images = self._read_images(run, ctx, incoming)
        run.ctx = ctx
        run.mode = ctx.mode
        self._audit(run, AuditAction.UNDERSTAND, _understand_detail(ctx, run.context_notes, images))
        threshold = self._config.conversation.unresolved_turns_alert
        if ctx.unresolved_turns >= threshold:
            run.alerts.append(ALERT_UNRESOLVED.format(n=ctx.unresolved_turns))

        # 5. 路由
        t = time.perf_counter()
        route: RouteDecision | None = None
        try:
            route = self._router.route(ctx)
        except Exception as exc:  # noqa: BLE001 — Router 本应自己降级，这里再兜一次
            self._warn("route", exc)
        run.add_latency("route", t)
        if route is None:
            run.degrade(DEGRADED_ROUTER)
            return self._finish(run, reason=REASON_ROUTER)
        run.route = route
        if route.degraded:
            run.degrade(DEGRADED_DECISION)
            run.context_notes.append(DEGRADED_DECISION)
        run.alerts[:0] = route.alerts
        self._audit(run, AuditAction.ROUTE, _route_detail(route))

        # 6. 按执行计划跑
        plan = ACTION_PLANS[route.action]
        if not plan.generate:
            soothe = self._transfer_soothe(run, ctx, route)
            return self._finish(
                run,
                reason=self._no_generate_reason(run, route),
                cards=[soothe[0]] if soothe else None,
                auto_reply=soothe[1] if soothe else None,
            )
        units, plan_reason = self._plan_units(run, ctx, route, plan)
        units = self._run_units(run, ctx, units, plan)
        self._record_gaps(run, ctx, units)

        scored = self._semi_auto(run, ctx, route, plan, units, self._cards(run, units))
        cards = self._persist(
            run, ctx, [(c, d) for c, d, _ in scored], confidences=[x for _, _, x in scored]
        )
        auto_reply = self._auto_reply(run, ctx, route, plan, units, cards)
        return self._finish(
            run,
            reason=None if cards else self._units_reason(units, plan_reason),
            cards=[c for c, _ in cards],
            units=units,
            auto_reply=auto_reply,
        )

    # ------------------------------------------------------------ 买家图片（§15 Phase 3）

    def _read_images(
        self, run: _Run, ctx: ConversationContext, incoming: StandardMessage
    ) -> tuple[ConversationContext, ImageReading | None]:
        """识别当前买家消息里的图片（engine.images），耗时记在 understand 里。

        识别失败、超时：记 vision 降级（写进卡片的风险提示），提醒客服自己看图，照常往下走；
        这一步意外抛错也一样（上下文保持原样）。"""
        t = time.perf_counter()
        try:
            ctx, reading = self._images.read(ctx, incoming)
        except Exception as exc:  # noqa: BLE001 — 图片理解出任何问题都不影响出推荐（§12.2）
            self._warn("vision", exc)
            run.degrade(DEGRADED_VISION)
            run.context_notes.append(DEGRADED_VISION)
            run.alerts.append(ALERT_VISION_FAILED)
            return ctx, None
        finally:
            run.add_latency("understand", t)
        if reading.degraded:
            run.degrade(DEGRADED_VISION)
            run.context_notes.append(DEGRADED_VISION)
        run.alerts.extend(reading.alerts)
        return ctx, reading

    # ------------------------------------------------------------ 乱序（§8.3）

    def _newer_buyer_message(self, msg: StandardMessage) -> StandardMessage | None:
        """会话里比 msg 晚的最新一条买家消息（还原成 StandardMessage）；没有返回 None。

        平台推送可能乱序：旧消息晚到时不能单独回答旧问题、把新消息的推荐顶掉。时间相同不算晚。
        记录里没有的字段（buyer_ref、接待客服、模式、平台扩展字段）沿用 msg 的——同一会话。
        """
        try:
            stored = self._conversations.recent_messages(
                msg.shop_id, msg.conversation_id, limit=_LATE_CHECK_LIMIT
            )
        except Exception:  # noqa: BLE001 — 读不到就按本条处理（会话理解会记 storage 降级）
            return None
        newer = [m for m in stored if m.role == Role.BUYER and m.created_at > msg.received_at]
        if not newer:
            return None
        last = newer[-1]
        return msg.model_copy(
            update={
                "msg_id": last.msg_id,
                "text": last.text or "",
                "item_card": ItemRef(item_id=last.item_id, sku_id=last.sku_id)
                if last.item_id
                else None,
                "order_card": OrderRef(order_id=last.order_ref) if last.order_ref else None,
                # 记录里没有图片：那条消息的图片由 engine.images 按 msg_id 从缓存里找，不沿用 msg 的
                "images": [],
                "received_at": last.created_at,
            }
        )

    # ------------------------------------------------------------ 刷新推荐（§5.3）

    def _expire_shown(self, run: _Run) -> None:
        msg = run.msg
        try:
            shown = self._suggestions.list(
                msg.shop_id, conversation_id=msg.conversation_id, status=SuggestionStatus.SHOWN
            )
            for record in shown:
                try:
                    self._suggestions.update(
                        record.model_copy(update={"status": SuggestionStatus.EXPIRED}),
                        expected_status=SuggestionStatus.SHOWN,
                    )
                except ConcurrencyConflict:
                    continue  # 客服刚好发送 / 反馈了这条：以客服的操作为准
        except Exception as exc:  # noqa: BLE001
            self._warn("expire", exc)
            run.degrade(DEGRADED_STORAGE)

    def on_recall(
        self, shop_id: str, conversation_id: str, msg_id: str
    ) -> list[SuggestionCard] | None:
        """撤回（§5.3）：被撤回的消息触发的、还在展示中的推荐改成 EXPIRED——撤回的消息不再进
        上下文，照着它写的推荐也不能再发。

        有推荐被过期时返回会话里仍在展示中的推荐卡片（侧边栏按它刷新；通常为空，即清空），
        没有改动返回 None（不用刷新）。客服刚好发送 / 反馈了的那条以客服的操作为准；
        影子记录（SHADOW）没展示过，不改状态。
        所有状态的推荐记录都去掉抄下来的买家原文（question_text）：撤回的消息原文不留（§5.3），
        每周复核（§13.5）也看不到。存储失败向外抛：应用层让平台重推撤回，两步都是幂等的。
        """
        expired = False
        shown = self._suggestions.list(
            shop_id, conversation_id=conversation_id, status=SuggestionStatus.SHOWN
        )
        # 剩下的卡片从过期之前读到的列表里算，不再读第二次：过期已经提交后第二次读失败的话，
        # 平台重推撤回时已经没有可过期的记录，侧边栏就再也刷新不了
        remaining = [r for r in shown if r.msg_id != msg_id]
        for record in shown:
            if record.msg_id != msg_id:
                continue
            try:
                self._suggestions.update(
                    record.model_copy(
                        update={"status": SuggestionStatus.EXPIRED, "question_text": None}
                    ),
                    expected_status=SuggestionStatus.SHOWN,
                )
            except ConcurrencyConflict:
                continue
            expired = True
        self._forget_recalled_text(shop_id, conversation_id, msg_id)
        self._images.forget(shop_id, conversation_id, msg_id)
        if not expired:
            return None
        return [card_from_record(r) for r in remaining]

    def _forget_recalled_text(self, shop_id: str, conversation_id: str, msg_id: str) -> None:
        """撤回的消息触发的推荐记录（任何状态）去掉买家原文。和客服操作并发时按新状态再试。"""
        for record in self._suggestions.list(shop_id, conversation_id=conversation_id):
            if record.msg_id != msg_id or record.question_text is None:
                continue
            current: SuggestionRecord | None = record
            for _ in range(3):
                if current is None or current.question_text is None:
                    break
                try:
                    self._suggestions.update(
                        current.model_copy(update={"question_text": None}),
                        expected_status=current.status,
                    )
                    break
                except ConcurrencyConflict:
                    assert current.id is not None
                    current = self._suggestions.get(shop_id, current.id)

    # ------------------------------------------------------------ 执行计划

    def _no_generate_reason(self, run: _Run, route: RouteDecision) -> str:
        if route.action == RouteAction.HUMAN_PRIORITY:
            if not route.alerts:
                run.alerts.append(ALERT_HUMAN_PRIORITY)
            return REASON_HUMAN_PRIORITY
        if route.action == RouteAction.TRANSFER_HUMAN:
            return REASON_TRANSFER
        if run.ctx is not None and route.intent in run.ctx.config.disabled_intents:
            return REASON_DISABLED  # 路由按店铺配置拦下的类型（§9.3），不是"没把握判断"
        return REASON_NO_SUGGESTION_DEGRADED if route.degraded else REASON_NO_SUGGESTION

    def _plan_units(
        self, run: _Run, ctx: ConversationContext, route: RouteDecision, plan: ActionPlan
    ) -> tuple[list[_Unit], str | None]:
        disabled = set(ctx.config.disabled_intents)
        if not plan.per_part:
            if route.action in _ANSWER_ACTIONS and route.intent in disabled:
                return [], REASON_DISABLED
            limit = ctx.config.suggestion_count
            if route.action == RouteAction.SOOTHE and not soothe_with_answer(route, disabled):
                limit = 1  # 规则 3 不附回答（店铺关闭 / 投诉 / 降级的高风险）：只留安抚话术
            unit = _Unit(route, route.intent_confidence, limit)
            return [unit], None
        parts = [(p.intent, p.confidence) for p in route.parts] or [
            (route.intent, route.intent_confidence)
        ]
        units: list[_Unit] = []
        reason: str | None = None
        high_risk = set(self._config.routing.high_risk)
        for intent, confidence in parts:
            if intent == Intent.COMPLAINT:  # 投诉不给推荐，提醒人工（§5.1、§5.7）
                run.alerts.append(ALERT_COMPLAINT_PART)
                reason = REASON_HUMAN_PRIORITY
                continue
            if intent in disabled:
                reason = reason or REASON_DISABLED
                continue
            risky = intent in high_risk
            # 每条只带自己的风险提示（§5.6），不继承整条消息的（那里有别的分条的提示）
            part_route = route.model_copy(
                update={
                    "intent": intent,
                    "parts": [],
                    "action": RouteAction.HIGH_RISK_ANSWER if risky else RouteAction.ANSWER,
                    "high_risk": risky,
                    "risk_hints": [risk_hint(intent)] if risky else [],
                }
            )
            units.append(_Unit(part_route, confidence, 1))
        return units, reason

    def _run_units(
        self, run: _Run, ctx: ConversationContext, units: list[_Unit], plan: ActionPlan
    ) -> list[_Unit]:
        """各条并行跑"找 → 写 → 判"，按原顺序收回并入 run（§12.1）。

        到"截止时间 + 宽限"还没做完的分条换成空的替身（该条不推荐，记 budget）；工作线程
        不写引擎自己的记录，丢掉的分条不会在 handle() 返回之后留下审计、推荐记录；它调用的平台
        工具、大模型迟到的缓存、用量写入由 BackgroundWork 兜住（取消订购等它们做完再删）。
        某一条意外抛错时和顺序执行一样：前面各条已经并入，这个异常交给 handle 的兜底，后面各条丢掉。
        """
        tasks = [partial(self._run_unit, ctx, unit, plan) for unit in units]
        outcomes: list[Outcome[_Unit]] = run_ordered(tasks)
        done: list[_Unit] = []
        for unit, outcome in zip(units, outcomes, strict=True):
            if not outcome.finished:
                logger.warning("问答引擎降级：分条超过时间预算，丢弃（%s）", unit.intent)
                late = unit.placeholder(REASON_BUDGET)
                late.budget_cut = True
                run.degrade(DEGRADED_BUDGET)
                done.append(late)
                continue
            if outcome.error is not None:
                # 这一条已经跑完（抛错结束），可以读它：抛错之前记下的照常并入，和顺序执行一样
                self._merge_unit(run, unit)
                self._merge_latency(run, [*done, unit])
                raise outcome.error
            assert outcome.value is not None
            self._merge_unit(run, outcome.value)
            done.append(outcome.value)
        self._merge_latency(run, done)
        return done

    def _merge_unit(self, run: _Run, unit: _Unit) -> None:
        """把一条的记录按发生顺序并入 run：降级、审计（在调用方线程写）、质检计数。"""
        for kind, value in unit.ops:
            if kind == "degrade":
                run.degrade(value)
            else:
                action, detail = value
                self._audit(run, action, detail)
        run.hard_hits.extend(unit.hard_hits)
        run.qa_checked += unit.qa_checked
        run.qa_failed += unit.qa_failed
        run.qa_reasons.extend(r for r in unit.qa_reasons if r not in run.qa_reasons)

    @staticmethod
    def _merge_latency(run: _Run, units: list[_Unit]) -> None:
        """分条并行：每个环节取各条里最长的（墙钟口径）。"""
        for stage in ("retrieve", "generate", "qa"):
            run.latency[stage] += max((u.latency[stage] for u in units), default=0.0)

    def _run_unit(self, ctx: ConversationContext, unit: _Unit, plan: ActionPlan) -> _Unit:
        """一条的"找 → 写 → 判"（工作线程里跑，只改 unit，见 _Unit）。"""
        route = unit.route
        # 规则 3 不附回答时（关闭的类型、投诉、降级时的高风险类型）只出安抚话术：不检索，
        # 不调平台工具，也不给关闭的类型记知识缺失
        soothe_only = route.action == RouteAction.SOOTHE and not soothe_with_answer(
            route, ctx.config.disabled_intents
        )
        if plan.retrieve and not soothe_only:
            if budget.expired():  # 截止时间已过：不再检索
                unit.cut()
                unit.reason = REASON_BUDGET
                return unit
            t = time.perf_counter()
            bundle: EvidenceBundle | None = None
            try:
                bundle = self._retriever.retrieve(ctx, route)
            except KnowledgeUnavailable:
                unit.degrade(DEGRADED_KNOWLEDGE)
                unit.reason = REASON_KNOWLEDGE
            except TenantIsolationError as exc:  # 串店（§9.2）：该条不推荐，只记在哪一步拦下的
                self._warn("retrieve", exc)
                unit.degrade(DEGRADED_TENANT_ISOLATION)
                unit.audit(AuditAction.TENANT_VIOLATION, _tenant_detail(TENANT_STAGE_RETRIEVE))
                unit.reason = REASON_TENANT
                unit.add_latency("retrieve", t)
                return unit
            except Exception as exc:  # noqa: BLE001
                self._warn("retrieve", exc)
                unit.degrade(DEGRADED_RETRIEVER)
                unit.reason = REASON_RETRIEVER
            unit.add_latency("retrieve", t)
            if bundle is None:
                if route.action != RouteAction.OFF_PLATFORM_SCRIPT:
                    return unit
                # 站外交易标准话术不依赖依据（§11.4）：检索失败照常出话术，只是不带依据
                bundle, unit.reason = EvidenceBundle(), None
            else:
                unit.audit(AuditAction.RETRIEVE, _retrieve_detail(route, bundle))
            unit.bundle = bundle
        bundle = unit.bundle
        for code in bundle.unavailable:
            unit.degrade(code)
        unit.missing = list(bundle.missing)
        if (
            route.action in _ANSWER_ACTIONS
            and route.intent not in LLM_FREE_INTENTS
            and bundle.is_empty
            and not bundle.clarify_question
            and not bundle.unavailable
        ):
            unit.reason = REASON_NO_EVIDENCE
            unit.missing = unit.missing or [missing_for_intent(route.intent)]
            return unit
        self._generate_and_check(ctx, unit)
        return unit

    def _generate_and_check(self, ctx: ConversationContext, unit: _Unit) -> None:
        """生成 → 质检 → 大模型草稿不通过带原因重写（§5.5、§11.5），受时间预算约束（§12.2）；
        判断层质检不可用的草稿不触发重写。

        生成器本来要写大模型草稿、因为剩余时间不够（少于 latency.min_llm_remaining_s）没调时，
        这一条记 budget（和来不及送判断层的草稿一样）；没有名额要写大模型草稿的不记。

        质检分批：还差 n 张卡片就先判接下来的 n 份草稿（并行），判哪些草稿和顺序执行一样。
        """
        route, bundle = unit.route, unit.bundle
        max_regen = self._config.qa_gate.max_regenerations
        checked_texts: set[str] = set()
        results: list[QAResult] = []
        failed_reasons: list[str] = []
        feedback: list[str] | None = None
        regenerations = 0
        generator_error = qa_error = False
        while True:
            if budget.expired():  # 截止时间已过：不再生成、重写
                unit.cut()
                break
            t = time.perf_counter()
            drafts: list[Draft] | None = None
            with budget.llm_skip_scope() as skip:
                try:
                    drafts = list(
                        self._generator.generate(ctx, route, bundle, retry_feedback=feedback)
                    )
                except Exception as exc:  # noqa: BLE001 — 生成失败：保留已通过的（§12.2）
                    self._warn("generate", exc)
            unit.add_latency("generate", t)
            if drafts is None:
                unit.degrade(DEGRADED_GENERATOR)
                generator_error = True
                break
            unit.audit(AuditAction.GENERATE, _generate_detail(route, drafts, regenerations))
            if skip.skipped:  # 生成器要写大模型草稿、剩余时间不够没调：这一条少了大模型草稿
                unit.cut()
            if budget.expired():  # 生成器到截止时间才返回：没做完的大模型草稿已经丢掉
                unit.cut()

            llm_failed: list[QAResult] = []
            queue = list(drafts)
            while queue and len(unit.passed) < unit.limit:
                wave: list[Draft] = []
                while queue and len(wave) < unit.limit - len(unit.passed):
                    draft = queue.pop(0)
                    if draft.text in checked_texts:
                        continue  # 同样的文字已经判过（重写时生成器又给了一遍 FAQ / 模板）
                    checked_texts.add(draft.text)
                    wave.append(draft)
                for draft, qa in zip(wave, self._check_wave(ctx, unit, wave), strict=True):
                    if qa is _NOT_IN_TIME:  # 来不及判的大模型草稿：丢掉，不算质检不通过
                        unit.cut()
                        continue
                    if qa is None:
                        unit.degrade(DEGRADED_QA)
                        qa_error = True
                        continue
                    results.append(qa)
                    unit.qa_checked += 1
                    unit.hard_hits.extend(h.rule for h in qa.hard_hits)
                    if qa.degraded:
                        unit.degrade(DEGRADED_DECISION_QA)
                    if qa.passed:
                        unit.passed.append((draft, qa))
                        continue
                    unit.qa_failed += 1
                    why = _qa_reasons(qa)
                    failed_reasons.extend(why)
                    unit.qa_reasons.extend(r for r in why if r not in unit.qa_reasons)
                    if draft.kind != DraftKind.LLM:  # FAQ / 模板：标准回答本身有问题，重写没用
                        unit.alerts.append(dropped_draft_alert(draft, qa))
                    elif not qa.degraded:
                        # 判断层质检不可用（qa.degraded）的草稿不算重写的理由：不是回答写得不对，
                        # 重写出来的一样没有判断层可判，只会在接口出故障的时候多调一轮大模型
                        llm_failed.append(qa)
            if len(unit.passed) >= unit.limit or not llm_failed or regenerations >= max_regen:
                break
            if not budget.llm_allowed():  # 来不及再调一次大模型：不重写，保留已通过的
                unit.cut()
                break
            regenerations += 1
            feedback = dedupe(r for qa in llm_failed for r in _qa_reasons(qa))

        if results or qa_error:
            unit.audit(AuditAction.QA, _qa_detail(route, results, unit, regenerations))
        if unit.passed:
            return
        if failed_reasons:
            unit.reason = REASON_QA_FAILED + "；".join(dedupe(failed_reasons))
        elif qa_error:
            unit.reason = REASON_QA_ERROR
        elif generator_error:
            unit.reason = REASON_GENERATOR
        elif unit.budget_cut:
            unit.reason = REASON_BUDGET
        else:
            unit.reason = REASON_NO_DRAFT

    def _check_wave(self, ctx: ConversationContext, unit: _Unit, wave: list[Draft]) -> list[Any]:
        """并行质检一批草稿，按原顺序返回：QAResult；None = 质检抛错；_NOT_IN_TIME = 大模型
        草稿来不及判（剩余时间不够调判断层、到点没判完，或截止之后才判完——结论不用）。

        FAQ / 模板只跑硬规则（§5.5，毫秒级），截止时间过了也照常判完。要调判断层的大模型草稿
        有两份以上时一起并行判；只有一份时按原顺序逐份判（没有可并行的，调用顺序和顺序执行一样），
        大模型草稿仍在单独的线程里判，到点不再等它。
        """
        if not wave:
            return []
        t = time.perf_counter()
        out: list[Any] = [None] * len(wave)
        pending: list[int] = []
        for i, draft in enumerate(wave):
            if draft.kind == DraftKind.LLM and not budget.llm_allowed():
                out[i] = _NOT_IN_TIME  # 不送判断层（§12.2 来不及调大模型）
            else:
                pending.append(i)
        judged = [i for i in pending if wave[i].kind == DraftKind.LLM]
        batches = [pending] if len(judged) >= 2 else [[i] for i in pending]
        check = partial(self._gate.check, ctx, unit.route)
        for batch in batches:
            tasks = [partial(check, wave[i], unit.bundle) for i in batch]
            if len(batch) == 1 and wave[batch[0]].kind != DraftKind.LLM:
                outcomes = [run_inline(tasks[0])]  # 只跑硬规则：直接判
            else:
                outcomes = run_ordered(tasks, grace_s=NESTED_GRACE_S, name="rodit-qa")
            for i, outcome in zip(batch, outcomes, strict=True):
                out[i] = self._wave_result(wave[i], outcome)
        unit.add_latency("qa", t)
        return out

    def _wave_result(self, draft: Draft, outcome: Outcome[QAResult]) -> Any:
        if not outcome.finished:
            return _NOT_IN_TIME
        if outcome.error is not None:
            self._warn("qa", outcome.error)  # 没验证过的文字不推荐
            return None
        if draft.kind == DraftKind.LLM and budget.expired():
            return _NOT_IN_TIME  # 截止之后才判完：引擎已经不等它了
        return outcome.value

    # ------------------------------------------------------------ 缺失知识（§5.7、§5.8）

    def _record_gaps(self, run: _Run, ctx: ConversationContext, units: list[_Unit]) -> None:
        if self._gaps is None:
            return
        seen: set[tuple[Intent, str]] = set()
        question = mask_pii(ctx.current.text)
        now = self._clock.now()
        try:
            for unit in units:
                for missing in unit.missing:
                    key = (unit.intent, missing)
                    if missing in _CONTEXT_MISSING or key in seen:
                        continue
                    seen.add(key)
                    self._gaps.add_gap(
                        KnowledgeGap(
                            shop_id=ctx.shop_id,
                            intent=unit.intent,
                            missing=missing,
                            question_text=question,
                            source="engine",
                            created_at=now,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            self._warn("gaps", exc)
            run.degrade(DEGRADED_STORAGE)

    # ------------------------------------------------------------ 卡片与记录（§5.6、§10）

    def _cards(self, run: _Run, units: list[_Unit]) -> list[tuple[SuggestionCard, Draft, float]]:
        """每张卡片带上它那一条的路由置信度（写进 SuggestionRecord.intent_confidence）。"""
        out: list[tuple[SuggestionCard, Draft, float]] = []
        for unit in units:
            for draft, qa in unit.passed[: unit.limit]:
                notes = [
                    *run.context_notes,
                    *unit.bundle.unavailable,
                    *([DEGRADED_DECISION_QA] if qa.degraded else []),
                ]
                card = build_card(
                    draft=draft,
                    qa=qa,
                    bundle=unit.bundle,
                    route=unit.route,
                    intent=unit.intent,
                    intent_confidence=unit.intent_confidence,
                    degraded=notes,
                )
                out.append((card, draft, unit.intent_confidence))
        return out

    def _persist(
        self,
        run: _Run,
        ctx: ConversationContext,
        cards: list[tuple[SuggestionCard, Draft]],
        *,
        confidences: Sequence[float],
    ) -> list[tuple[SuggestionCard, Draft]]:
        if not cards:
            return cards
        status = SuggestionStatus.SHADOW if ctx.mode == Mode.SHADOW else SuggestionStatus.SHOWN
        if status == SuggestionStatus.SHADOW and run.late and self._shadow_turn_recorded(run):
            # 影子模式每轮一套记录（§13.4 事后对比不重复计数）：这一轮第一次处理时已经记过
            self._audit(
                run,
                AuditAction.SUGGEST,
                {
                    "suggestion_ids": [None] * len(cards),
                    "count": len(cards),
                    "status": str(status),
                    "intents": [str(c.intent) for c, _ in cards],
                    "kinds": [str(c.kind) for c, _ in cards],
                    "shadow_duplicate": True,
                },
            )
            return cards
        question = mask_pii(ctx.current.text)
        item_id = ctx.item.item_id if ctx.item is not None else None
        now = self._clock.now()
        latency = run.elapsed_ms()
        out: list[tuple[SuggestionCard, Draft]] = []
        for (card, draft), confidence in zip(cards, confidences, strict=True):
            record = SuggestionRecord(
                shop_id=ctx.shop_id,
                conversation_id=ctx.conversation_id,
                msg_id=run.msg.msg_id,
                question_text=question,
                item_id=item_id,
                intent=card.intent,
                intent_confidence=min(1.0, max(0.0, confidence)),
                route_rule=run.route.rule if run.route is not None else None,
                text=card.text,
                kind=card.kind,
                evidence_ids=dedupe(draft.evidence_ids),
                qa_scores=dict(card.qa_scores) or None,
                status=status,
                agent_id=ctx.agent_id,
                latency_ms=latency,
                created_at=now,
            )
            try:
                saved = self._suggestions.add(record)
                card = card.model_copy(update={"suggestion_id": saved.id})
            except Exception as exc:  # noqa: BLE001 — 记录失败也照常展示推荐
                self._warn("suggestion", exc)
                run.degrade(DEGRADED_STORAGE)
                # 半自动：没有推荐编号，客服按发送后记不了反馈——不填进输入框
                card = card.model_copy(update={"prefill": False})
            out.append((card, draft))
        detail: dict[str, Any] = {
            "suggestion_ids": [c.suggestion_id for c, _ in out],
            "count": len(out),
            "status": str(status),
            "intents": [str(c.intent) for c, _ in out],
            "kinds": [str(c.kind) for c, _ in out],
        }
        prefilled = [c.suggestion_id for c, _ in out if c.prefill]
        if prefilled:  # 可选键：半自动填进输入框的推荐（§5.6）
            detail["prefill"] = prefilled
        self._audit(run, AuditAction.SUGGEST, detail)
        return out

    def _shadow_turn_recorded(self, run: _Run) -> bool:
        """这一轮（最新买家消息）已经有影子记录。读不到时按没有处理（照常写，写不进去会降级）。"""
        msg = run.msg
        try:
            records = self._suggestions.list(
                msg.shop_id, conversation_id=msg.conversation_id, status=SuggestionStatus.SHADOW
            )
        except Exception as exc:  # noqa: BLE001
            self._warn("shadow_check", exc)
            return False
        return any(r.msg_id == msg.msg_id for r in records)

    # ------------------------------------------------------------ 半自动（§5.6、§13.4）

    def _semi_auto(
        self,
        run: _Run,
        ctx: ConversationContext,
        route: RouteDecision,
        plan: ActionPlan,
        units: list[_Unit],
        scored: list[tuple[SuggestionCard, Draft, float]],
    ) -> list[tuple[SuggestionCard, Draft, float]]:
        """第一张卡片满足半自动的条件（engine.semi_auto）时打上 prefill，最多一张。
        分条推荐（每条一张卡片）不打。"""
        if not scored or plan.per_part or len(units) != 1 or not units[0].passed:
            return scored
        unit = units[0]
        card, draft, confidence = scored[0]
        passed_draft, qa = unit.passed[0]
        if passed_draft is not draft:  # 第一张卡片就是这一条第一份通过的草稿
            return scored
        blocker = prefill_blocker(
            mode=ctx.mode,
            config=ctx.config,
            routing=self._config.routing,
            route=route,
            card=card,
            draft=draft,
            qa=qa,
            alerts=[*run.alerts, *unit.alerts, *unit.bundle.alerts],
            degraded=run.degraded,
            qa_blocked=unit.qa_failed,  # 第一稿被拦、重写后才通过的也不填（§5.6 宁可保守）
        )
        if blocker is not None:
            return scored
        return [(card.model_copy(update={"prefill": True}), draft, confidence), *scored[1:]]

    # ------------------------------------------------------------ 全自动（§5.6、§12.2、§12.3）

    def _transfer_soothe(
        self, run: _Run, ctx: ConversationContext, route: RouteDecision
    ) -> tuple[SuggestionCard, SuggestionCard] | None:
        """全自动模式下因情绪激烈转人工时，附一句安抚话术（§5.7 "转人工，附安抚话术"）。

        只用生成器的安抚模板（不调大模型），过质检硬规则，加 AI 标识前缀后仍不超长才发；
        任何一步不满足就只转人工。返回 (记录下来的卡片, 带前缀的自动回复)。
        """
        if route.action != RouteAction.TRANSFER_HUMAN or route.rule != 3:
            return None
        if ctx.mode != Mode.AUTO or not ctx.config.auto_reply_enabled or run.late:
            return None
        soothe_route = route.model_copy(update={"action": RouteAction.SOOTHE})
        bundle = EvidenceBundle()
        try:
            drafts = self._generator.generate(ctx, soothe_route, bundle)
            draft = next(
                (d for d in drafts if d.kind == DraftKind.TEMPLATE and d.variant == "soothe"),
                None,
            )
            if draft is None:
                return None
            qa = self._gate.check(ctx, soothe_route, draft, bundle)
        except Exception as exc:  # noqa: BLE001 — 安抚话术出不来就只转人工
            self._warn("transfer_soothe", exc)
            return None
        if not qa.passed:
            return None
        text = self._config.generation.auto_reply_prefix + qa.text
        if len(text) > ctx.config.max_answer_chars:
            return None
        card = build_card(
            draft=draft,
            qa=qa,
            bundle=bundle,
            route=soothe_route,
            intent=route.intent,
            intent_confidence=route.intent_confidence,
            degraded=[],
        )
        persisted = self._persist(run, ctx, [(card, draft)], confidences=[route.intent_confidence])
        stored = persisted[0][0] if persisted else card
        return stored, stored.model_copy(update={"text": text})

    def _auto_reply(
        self,
        run: _Run,
        ctx: ConversationContext,
        route: RouteDecision,
        plan: ActionPlan,
        units: list[_Unit],
        cards: list[tuple[SuggestionCard, Draft]],
    ) -> SuggestionCard | None:
        if ctx.mode != Mode.AUTO or not ctx.config.auto_reply_enabled:
            return None
        if run.late:  # 重新处理最新那条：它可能已经自动回复过，不再发一遍
            return None
        if not route.auto_eligible or plan.per_part or not cards:
            return None
        top = cards[0][0]
        if top.confidence != Confidence.HIGH or top.risk_hints:
            return None
        if run.degraded or run.alerts or any(u.alerts for u in units):
            return None
        if ctx.unresolved_turns >= self._config.conversation.unresolved_turns_alert:
            return None
        text = _with_ai_prefix(self._config.generation.auto_reply_prefix, top.text)
        if len(text) > ctx.config.max_answer_chars:
            return None
        return top.model_copy(update={"text": text})

    # ------------------------------------------------------------ 收尾

    def _units_reason(self, units: list[_Unit], plan_reason: str | None) -> str:
        reasons = dedupe(u.reason for u in units if u.reason)
        if plan_reason:
            reasons = dedupe([*reasons, plan_reason])
        return "；".join(reasons) or REASON_NO_DRAFT

    def _finish(
        self,
        run: _Run,
        *,
        reason: str | None,
        cards: list[SuggestionCard] | None = None,
        units: list[_Unit] | None = None,
        auto_reply: SuggestionCard | None = None,
    ) -> EngineResult:
        cards = cards or []
        units = units or []
        alerts = list(run.alerts)
        for unit in units:
            alerts.extend(unit.alerts)
            if not cards:  # 没有卡片时，证据包里的提醒（物流停滞等）直接提醒客服
                alerts.extend(unit.bundle.alerts)
            elif unit.reason and not unit.passed:
                alerts.append(f"「{INTENT_LABELS[unit.intent]}」没有推荐：{unit.reason}")
        route = run.route
        if auto_reply is not None:
            self._audit(
                run,
                AuditAction.AUTO_REPLY,
                {
                    "suggestion_id": auto_reply.suggestion_id,
                    "intent": str(auto_reply.intent),
                    "kind": str(auto_reply.kind),
                    "chars": len(auto_reply.text),
                },
            )
        shadow = run.mode == Mode.SHADOW
        auto = run.mode == Mode.AUTO or run.auto_unverified
        transfer = not shadow and (
            (route is not None and route.action == RouteAction.TRANSFER_HUMAN)
            or (auto and auto_reply is None)
            or run.auto_gate is not None  # 不在全自动生效时段：交给在岗客服（engine.modes）
        )
        ctx = run.ctx
        result = EngineResult(
            shop_id=run.msg.shop_id,
            conversation_id=run.msg.conversation_id,
            msg_ids=list(ctx.current.msg_ids) if ctx is not None else [run.msg.msg_id],
            mode=run.mode,
            route=route,
            suggestions=cards,
            alerts=dedupe(alerts),
            no_suggestion_reason=None if cards else (reason or REASON_NO_DRAFT),
            missing_knowledge=dedupe(m for u in units for m in u.missing),
            auto_reply=None if shadow else auto_reply,
            transfer_to_human=transfer,
            degraded=list(run.degraded),
            latency_ms=run.latency_ms(self._clock.now()),
        )
        run.result = result
        self._write_result(run, result)
        return result

    def _finish_safely(self, run: _Run) -> EngineResult:
        """意外异常后的收尾：尽量给出完整结果，再不行给最小结果。"""
        try:
            return self._finish(run, reason=REASON_ENGINE)
        except Exception as exc:  # noqa: BLE001
            self._warn("finish", exc)
        if run.result is not None:
            return run.result
        try:
            now: datetime | None = self._clock.now()
        except Exception:  # noqa: BLE001 — 时钟都读不出来：since_received 按 total 算
            now = None
        result = EngineResult(
            shop_id=run.msg.shop_id,
            conversation_id=run.msg.conversation_id,
            msg_ids=[run.msg.msg_id],
            mode=run.mode,
            no_suggestion_reason=REASON_ENGINE,
            transfer_to_human=(
                run.mode == Mode.AUTO or run.auto_unverified or run.auto_gate is not None
            ),
            degraded=list(run.degraded),
            latency_ms=run.latency_ms(now),
        )
        run.result = result
        self._write_result(run, result)
        return result

    def _write_result(self, run: _Run, result: EngineResult) -> None:
        route = result.route
        detail: dict[str, Any] = {
            ResultKey.INTENT: str(route.intent) if route else None,
            ResultKey.ACTION: str(route.action) if route else None,
            ResultKey.RULE: route.rule if route else None,
            ResultKey.SUGGESTIONS: len(result.suggestions),
            ResultKey.NO_SUGGESTION: result.no_suggestion_reason,
            ResultKey.QA_CHECKED: run.qa_checked,
            ResultKey.QA_FAILED: run.qa_failed,
            ResultKey.QA_REASONS: [mask_pii(r) for r in run.qa_reasons],
            ResultKey.HARD_HITS: list(run.hard_hits),
            ResultKey.DEGRADED: list(result.degraded),
            ResultKey.TRANSFER: result.transfer_to_human,
            ResultKey.LATENCY: dict(result.latency_ms),
            ResultKey.MISSING: list(result.missing_knowledge),
        }
        detail = {str(k): v for k, v in detail.items()}
        if DEGRADED_BUDGET in result.degraded:  # 可选键：只在超过时间预算时写
            detail[RESULT_BUDGET_EXCEEDED] = True
        if run.auto_gate is not None:  # 可选键：全自动不在生效时段，退回智能辅助（engine.modes）
            detail[RESULT_AUTO_GATE] = run.auto_gate
        self._audit(run, AuditAction.RESULT, detail)

    # ------------------------------------------------------------ 留痕

    def _audit(self, run: _Run, action: AuditAction, detail: dict[str, Any]) -> None:
        try:
            self._audit_sink.record(
                AuditEvent(
                    shop_id=run.msg.shop_id,
                    actor=ACTOR,
                    action=action.value,
                    detail=detail,
                    created_at=self._clock.now(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._warn("audit", exc)
            run.degrade(DEGRADED_AUDIT)

    @staticmethod
    def _warn(stage: str, exc: BaseException) -> None:
        # 只记异常类型，不记异常消息（可能带买家原文，§12.3 日志脱敏）
        logger.warning("问答引擎降级：%s 失败（%s）", stage, type(exc).__name__)


def _with_ai_prefix(prefix: str, text: str) -> str:
    """全自动回复开头加且只加一次 AI 生成内容标识（§12.3）。会话历史里的自动回复带着前缀，
    大模型照着写时回答开头已经有了：先去掉再加。"""
    body = text.lstrip()
    while prefix and body.startswith(prefix):
        body = body[len(prefix) :].lstrip()
    return prefix + body


# ---------------------------------------------------------------- 审计 detail（不放买家原文）


def _tenant_detail(stage: str) -> dict[str, Any]:
    """串店审计：只记在哪一步拦下的，不记店铺、订单或任何数据（§9.2、§12.4）。"""
    return {"stage": stage}


def _qa_reasons(qa: QAResult) -> list[str]:
    if qa.reasons:
        return list(qa.reasons)
    rules = [HARD_RULE_LABELS.get(r, r) for r in blocking_rules(qa)]
    return rules or ["质检未通过"]


def _understand_detail(
    ctx: ConversationContext, degraded: list[str], images: ImageReading | None = None
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "msg_ids": list(ctx.current.msg_ids),
        "mode": str(ctx.mode),
        "history_turns": len(ctx.history),
        "item_id": ctx.item.item_id if ctx.item is not None else None,
        "sku_id": ctx.sku_id,
        "candidate_items": len(ctx.candidate_item_ids),
        "order_id": ctx.order.order_id if ctx.order is not None else None,
        "recent_orders": len(ctx.recent_orders),
        "unresolved_turns": ctx.unresolved_turns,
        "degraded": list(degraded),
    }
    if images is not None and images.total:
        # 买家图片（§15 Phase 3）：只记张数和类型，不记识别出的文字（§5.3）
        detail.update(describe_for_audit(images))
    return detail


def _route_detail(route: RouteDecision) -> dict[str, Any]:
    return {
        "intent": str(route.intent),
        "action": str(route.action),
        "rule": route.rule,
        "confidence": route.intent_confidence,
        "parts": [str(p.intent) for p in route.parts],
        "high_risk": route.high_risk,
        "auto_eligible": route.auto_eligible,
        "degraded": route.degraded,
        "provider": route.provider,
    }


def _retrieve_detail(route: RouteDecision, bundle: EvidenceBundle) -> dict[str, Any]:
    return {
        "intent": str(route.intent),
        "evidence_ids": bundle.ids(),
        "overridden_ids": list(bundle.overridden_ids),
        "missing": list(bundle.missing),
        "unavailable": list(bundle.unavailable),
        "clarify": bundle.clarify_question is not None,
        "alerts": len(bundle.alerts),
    }


def _generate_detail(route: RouteDecision, drafts: list[Draft], retry: int) -> dict[str, Any]:
    return {
        "intent": str(route.intent),
        "drafts": len(drafts),
        "kinds": [str(d.kind) for d in drafts],
        "retry": retry,
    }


def _qa_detail(
    route: RouteDecision, results: list[QAResult], unit: _Unit, regenerations: int
) -> dict[str, Any]:
    return {
        "intent": str(route.intent),
        "checked": len(results),
        "passed": len(unit.passed),
        "hard_hits": [h.rule for r in results for h in r.hard_hits],
        "model_checked": sum(1 for r in results if r.model_checked),
        "degraded": any(r.degraded for r in results),
        "regenerations": regenerations,
    }
