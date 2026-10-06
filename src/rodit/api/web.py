"""网页工具：模式 C 独立话术助手（方案 §8.1 模式 C、§5.6 推荐卡片、§5.8 反馈、§12.4 商家报表）
和商家控制台（§9.1 "知识编辑、配置、报表"、§5.2 知识中心、§9.3 配置分层、§13.4 上线节奏、
§13.5 每周复核）。

三个自带的页面：HTML + 原生 JS + CSS，没有构建步骤，不引用任何外部地址（CDN、字体）——要能在平台云 /
内网里直接用。文件在 rodit/api/static/，应用启动时读进内存。

- 话术助手：GET / 和 GET /assist；GET /assist/assist.js、/assist/assist.css：脚本和样式。
- 商家控制台：GET /console；GET /console/console.js、/console/console.css（基础样式沿用
  /assist/assist.css，两个页面长得一样）。店长、商家看知识、政策、商品、设置、上线状态和报表（写操作
  只给商家），复核人员做每周复核、切换上线阶段；客服在控制台只看到去话术助手的提示。
- 售后审核工作台（§7.3.2、§15 Phase 2）：GET /aftersale；GET /aftersale/aftersale.js、
  /aftersale/aftersale.css（基础样式沿用 /assist/assist.css）。客服、店长、商家看待审核列表、
  批量跑规则、看详情和规则说明，确认执行（R6 店长确认、平台差异由接口把关）；
  复核人员只看到去控制台的提示。
- 页面本身不含任何店铺数据，不需要鉴权；数据都由页面脚本带 Bearer 令牌调用 /v1 接口取
  （/v1/me 登录，/v1/shops/{shop_id}/... 取数、写入），鉴权、角色和串店检查都在接口上（§9.2）。
- 令牌只放在 sessionStorage（关掉标签页就没了），不进 URL、不进日志；两个页面用同一个键，
  同一标签页里互相跳转不用重新登录。输入框没有 name，CSP 的 form-action 'none' 禁止表单提交，
  脚本没加载时令牌和输入的文字也不会跑进网址。
- 响应头带严格的 CSP：只允许同源的脚本、样式和请求，不允许内联脚本、内联样式、被别的页面嵌入。
  页面脚本只用 textContent 写服务端和用户给的文字，不用 innerHTML（防 XSS）。
- 中文名（问题类型、弃用原因、政策项、配置项、上线阶段、复核维度……）、问卷题目、文字长度上限在启动时
  写进页面里的 JSON 数据块（<script type="application/json">，浏览器不执行、不受 CSP 限制），
  和服务端用同一份定义。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from importlib import resources
from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response

from rodit.aftersale import AFTERSALE_STATUS_LABELS, TERMINAL_STATES
from rodit.aftersale.assistant import ACTION_LABELS as AFTERSALE_ACTION_LABELS
from rodit.api.auth import (
    CONSOLE_ROLES,
    HUMAN_ROLES,
    MERCHANT_ROLES,
    PERSON_ROLES,
    REVIEWER_ROLES,
)
from rodit.api.schemas import MAX_ID_CHARS, MAX_NOTE_CHARS, MAX_TEXT_CHARS
from rodit.api.schemas_console import MAX_LIST_ITEMS
from rodit.app.aftersale_desk import (
    ADVICE_LABELS,
    DEFAULT_BATCH,
    DEFAULT_QUEUE_LIMIT,
    MAX_BATCH,
    NEEDS_ACTION,
    NEXT_STEP_LABELS,
    PLATFORM_ACTIONS,
    PLATFORM_NOTES,
    REFUSE_HINTS,
    REVIEW_STATE_LABELS,
    RULES,
    STATE_FLOW,
    UNREVIEWED,
    NextStep,
)
from rodit.app.rollout import MAX_NOTE_CHARS as MAX_REVIEW_NOTE_CHARS
from rodit.core.enums import (
    INTENT_LABELS,
    POLICY_LABELS,
    AfterSaleAction,
    AfterSaleReason,
    AfterSaleStatus,
    AfterSaleType,
    Confidence,
    DraftKind,
    Intent,
    KnowledgeSource,
    KnowledgeStatus,
    Platform,
    PolicyKey,
    QualityReviewStatus,
    RejectReason,
    ReviewAdvice,
    ReviewState,
    SuggestionStatus,
)
from rodit.core.evidence import PLATFORM_NAMES
from rodit.core.models import QUALITY_PERIOD_PATTERN, ShopConfig
from rodit.core.platform_metrics import metric_labels
from rodit.core.schedule import MAX_SERVICE_WINDOWS
from rodit.feedback import REJECT_REASON_LABELS
from rodit.feedback.review import (
    DEFAULT_MIN_SCORED,
    DEFAULT_SAMPLE_SIZE,
    DIMENSIONS,
    FULL_SCORE,
    LAUNCH_MIN_FULL_RATE,
    MAX_SAMPLE_SIZE,
    GateStatus,
)
from rodit.feedback.rollout import STAGE_LABELS, RolloutStage
from rodit.knowledge import QUESTIONNAIRE
from rodit.knowledge.admin import (
    LOCKED_CONFIG_FIELDS,
    MAX_ACCOUNT_CHARS,
    MAX_ANSWER_CHARS,
    MAX_BANNED_WORD_CHARS,
    MAX_BANNED_WORDS,
    MAX_CLAIM_CHARS,
    MAX_DISPLAY_NAME_CHARS,
    MAX_ITEM_LINE_CHARS,
    MAX_MANAGER_IDS,
    MAX_POLICY_CHARS,
    MAX_QUESTION_CHARS,
    MAX_SIZE_ADVICE_CHARS,
    MERCHANT_CONFIG_FIELDS,
    MERCHANT_ITEM_FIELDS,
)
from rodit.tools.evidence import AFTERSALE_REASON_LABELS, AFTERSALE_TYPE_LABELS

PAGE_PATHS: tuple[str, ...] = ("/", "/assist")
"""话术助手页面地址。脚本和样式用相对地址引用，两个地址下都能找到。"""

ASSET_PREFIX = "/assist/"
ASSETS: dict[str, str] = {
    "assist.js": "text/javascript",
    "assist.css": "text/css",
}
"""静态文件 → 内容类型。只有这几个文件对外，不按请求路径读文件系统。"""

PAGE_FILE = "assist.html"

CONSOLE_PAGE_PATHS: tuple[str, ...] = ("/console",)
"""商家控制台页面地址（相对地址 console/console.js、assist/assist.css 都从根目录算起）。"""

CONSOLE_ASSET_PREFIX = "/console/"
CONSOLE_ASSETS: dict[str, str] = {
    "console.js": "text/javascript",
    "console.css": "text/css",
}

CONSOLE_PAGE_FILE = "console.html"
CONFIG_PLACEHOLDER = "__RODIT_PAGE_CONFIG__"

SERVICE_HOURS_LINE_CHARS = 20
"""一行工作时段（"09:00-23:00"）的输入长度上限。"""

PHASE0_ADOPTION_TARGET = 0.5
"""Phase 0 验收：模式 C 下种子商家推荐采纳率 ≥ 50%（§15；按推荐组算，同 feedback.phase_gate）。"""

CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    )
)

SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}

PAGE_HEADERS: dict[str, str] = {**SECURITY_HEADERS, "Cache-Control": "no-store"}
"""两个页面本身不进缓存：退出后按后退不从缓存恢复出上一位的界面（纵深防御；Chrome 的往返缓存
对 no-store 的页面照样可能恢复，所以页面脚本另外在 pageshow 时核对 sessionStorage 里的令牌）。"""

CONFIDENCE_LABELS: dict[Confidence, str] = {Confidence.HIGH: "高", Confidence.MEDIUM: "中"}

DRAFT_KIND_LABELS: dict[DraftKind, str] = {
    DraftKind.FAQ: "FAQ",
    DraftKind.TEMPLATE: "模板",
    DraftKind.LLM: "AI 生成",
}

ROLE_LABELS: dict[str, str] = {
    "agent": "客服",
    "manager": "店长",
    "merchant": "商家",
    "reviewer": "复核人员",
    "integration": "平台对接",
}

DEGRADED_LABELS: dict[str, str] = {
    "storage": "存储",
    "knowledge": "知识库",
    "understanding": "会话理解",
    "router": "路由",
    "decision": "判断层（已退回关键词规则）",
    "decision:qa": "判断层质检（只跑了硬规则）",
    "retriever": "检索",
    "generator": "答案生成",
    "qa": "质检",
    "audit": "审计留痕",
    "engine": "问答引擎",
    "llm": "大模型",
    "tool": "平台接口",
}
"""降级标记（rodit.engine 模块文档）的中文名；"tool:<kind>" 按前缀 tool 显示。"""

# ================================================================ 商家控制台的中文名

CONSOLE_TABS: dict[str, frozenset[str]] = {
    "knowledge": CONSOLE_ROLES,
    "policies": CONSOLE_ROLES,
    "items": CONSOLE_ROLES,
    "config": CONSOLE_ROLES,
    "rollout": CONSOLE_ROLES | REVIEWER_ROLES,
    "review": REVIEWER_ROLES,
    "reports": CONSOLE_ROLES,
}
"""控制台各页签 → 能看的角色（和各页签用到的读接口的角色一致，§9.2 由接口把关）。
写操作另看 role_groups：知识、政策、商品、设置只给商家；切回影子给商家和复核人员，切到智能辅助只给
复核人员（§13.4）。"""

KNOWLEDGE_STATUS_LABELS: dict[KnowledgeStatus, str] = {
    KnowledgeStatus.APPROVED: "已上线",
    KnowledgeStatus.CANDIDATE: "候选（待审核）",
    KnowledgeStatus.RETIRED: "已下线",
}

KNOWLEDGE_SOURCE_LABELS: dict[KnowledgeSource, str] = {
    KnowledgeSource.MERCHANT: "商家录入",
    KnowledgeSource.ADOPTED: "客服采纳",
    KnowledgeSource.CATEGORY_TEMPLATE: "类目通用",
}

ITEM_FIELD_LABELS: dict[str, str] = {
    "size_advice": "尺码建议",
    "care": "洗涤保养",
    "usage": "使用说明",
    "forbidden_claims": "不能说的卖点",
}
"""商品卡片上商家填写的字段（§5.2.2，knowledge.admin.MERCHANT_ITEM_FIELDS）。"""

ITEM_FIELD_LIMITS: dict[str, dict[str, int]] = {
    "size_advice": {"max_chars": MAX_SIZE_ADVICE_CHARS},
    "care": {"max_chars": MAX_ITEM_LINE_CHARS, "max_items": MAX_LIST_ITEMS},
    "usage": {"max_chars": MAX_ITEM_LINE_CHARS, "max_items": MAX_LIST_ITEMS},
    "forbidden_claims": {"max_chars": MAX_CLAIM_CHARS, "max_items": MAX_LIST_ITEMS},
}

SYNC_FIELD_LABELS: dict[str, str] = {
    "created": "新建",
    "updated": "更新",
    "unchanged": "没有变化",
    "off_sale": "平台上已下架",
    "missing_on_platform": "平台上已没有",
    "ocr_failed_items": "详情图识别失败",
}
"""商品同步报告（knowledge.sync.SyncReport）里各类商品 ID 列表的中文名。"""

CONFIG_FIELD_LABELS: dict[str, str] = {
    "display_name": "店铺名",
    "tone": "语气",
    "salutation": "称呼",
    "max_answer_chars": "回答最多字数",
    "banned_words": "禁用词",
    "disabled_intents": "关闭的问题类型",
    "suggestion_count": "推荐条数",
    "aftersale_amount_limit": "售后金额上限（元）",
    "manager_ids": "店长子账号",
    "shadow_mode": "影子模式",
    "auto_reply_enabled": "全自动回复",
    "service_hours": "客服工作时间",
    "auto_reply_schedule": "全自动生效时段",
    "semi_auto_intents": "半自动的问题类型",
}
"""店铺配置项（core.models.ShopConfig）的中文名。"""

CONFIG_FIELD_HINTS: dict[str, str] = {
    "display_name": "用于提示词「你是{店铺名}的客服」",
    "tone": "推荐回答的语气",
    "salutation": "推荐回答怎么称呼买家",
    "max_answer_chars": "推荐回答超过这个字数会被截短",
    "banned_words": "每行一个；回答里出现就拦截（附录 C 第 14 题）",
    "disabled_intents": "勾选的问题类型不再给推荐，由客服自己回复",
    "suggestion_count": "每条买家消息最多给几条推荐（§5.4）",
    "aftersale_amount_limit": "超过这个金额的售后操作必须店长确认（R6，§7.3.2）",
    "manager_ids": "每行一个子账号；超额售后只认这里列出的店长",
    "service_hours": "每行一个时段，如 09:00-23:00（北京时间，附录 C 第 13 题）",
    "auto_reply_schedule": "off_hours：只在客服工作时间以外自动回复；always：全天",
    "semi_auto_intents": "这些类型的高置信度推荐直接填进客服输入框，客服按发送即可",
}

LOCKED_CONFIG_REASONS: dict[str, str] = {
    "shadow_mode": "由上线节奏决定（§13.4）：在「上线」页切换，恢复智能辅助要复核人员确认",
    "auto_reply_enabled": "全自动要平台准入，由我们评估后开启（§13.4），这里不能改",
    "auto_reply_schedule": "全自动先在非工作时间试行（§15 Phase 3），由我们评估后调整",
    "semi_auto_intents": "该类型原样采纳率达到门槛后由复核人员开启（§13.4），这里不能改",
}

CONFIG_LINE_LIMITS: dict[str, dict[str, int]] = {
    "banned_words": {"max_chars": MAX_BANNED_WORD_CHARS, "max_items": MAX_BANNED_WORDS},
    "manager_ids": {"max_chars": MAX_ACCOUNT_CHARS, "max_items": MAX_MANAGER_IDS},
    "service_hours": {"max_chars": SERVICE_HOURS_LINE_CHARS, "max_items": MAX_SERVICE_WINDOWS},
}

GATE_STATUS_LABELS: dict[GateStatus, str] = {
    GateStatus.PASSED: "已达标",
    GateStatus.FAILED: "未达标",
    GateStatus.INSUFFICIENT: "数据不足",
}

GATE_CHECK_LABELS: dict[str, str] = {
    "periods": "统计周期数",
    "consecutive": "统计周期首尾相接",
    "group_adoption_rate": "采纳率（按推荐组）",
    "group_as_is_rate": "原样采纳率（按推荐组）",
    "platform_violations": "平台违规（复核人员登记）",
    "compliance_scan": "发出去的文字疑似违规（合规扫描）",
    "complaints": "投诉（复核人员登记）",
    "test_set": "测试集（评测结论）",
    "shadow_review.scored": "已打分的影子推荐样本",
    "shadow_review.factual_full_rate": "影子样本事实正确满分率",
    "shadow_review.compliance_full_rate": "影子样本合规满分率",
}
"""门槛条件名（feedback.metrics.GateCheck.name、feedback.rollout）的中文名；
"period_N.<指标>" 显示为"第 N 周" + 指标名。"""

SCORE_GUIDE: dict[str, list[str]] = {
    "factual": ["有错误事实", "基本正确但不完整", "完全正确"],
    "relevance": ["答非所问", "部分回答", "完整回答"],
    "compliance": ["有违规内容", "有风险表达", "无问题"],
    "expression": ["生硬或啰嗦", "可以接受", "自然得体"],
}
"""§13.2 回答质量评分表：各维度 0 / 1 / 2 分的含义。"""

REVIEW_STATUS_LABELS: dict[QualityReviewStatus, str] = {
    QualityReviewStatus.PENDING: "待复核",
    QualityReviewStatus.SCORED: "已打分",
    QualityReviewStatus.SKIPPED: "已跳过",
}

SUGGESTION_STATUS_LABELS: dict[SuggestionStatus, str] = {
    SuggestionStatus.SHOWN: "已展示、未操作",
    SuggestionStatus.SENT: "原样发送",
    SuggestionStatus.EDITED_SENT: "编辑后发送",
    SuggestionStatus.REJECTED: "不合适",
    SuggestionStatus.EXPIRED: "已过期",
    SuggestionStatus.SHADOW: "影子模式（未展示）",
}

HARD_RULE_LABELS: dict[str, str] = {
    "off_platform": "站外联系 / 交易",
    "absolute_term": "绝对化用语",
    "banned_word": "商家禁用词",
    "forbidden_claim": "不能说的卖点",
    "number_mismatch": "依据里没有的数字",
    "time_promise": "没有依据的时效承诺",
    "length": "超长（已截短）",
    "format": "格式（已修正）",
    "unknown_evidence": "引用了不存在的依据",
    "policy_conflict": "和店铺政策冲突",
    "duplicate_question": "和已上线 FAQ 的问题重复",
}
"""质检硬规则（core.models.HardRuleHit.rule）和保存 FAQ 时的提醒
（knowledge.admin.FaqWarning.rule）的中文名。"""


def page_config(platform: Platform) -> dict[str, Any]:
    """写进话术助手页面的配置：中文名、问卷题目、长度上限、角色分组。都是服务端常量，没有店铺数据。"""
    return {
        "platform": platform.value,
        "platform_name": PLATFORM_NAMES.get(platform, platform.value),
        "max_text_chars": MAX_TEXT_CHARS,
        "adoption_target": PHASE0_ADOPTION_TARGET,
        "human_roles": sorted(HUMAN_ROLES),
        "console_roles": sorted(CONSOLE_ROLES),
        "merchant_roles": sorted(MERCHANT_ROLES),
        "reviewer_roles": sorted(REVIEWER_ROLES),
        "roles": ROLE_LABELS,
        "intents": {i.value: INTENT_LABELS[i] for i in Intent},
        "reject_reasons": {r.value: REJECT_REASON_LABELS[r] for r in RejectReason},
        "confidence": {c.value: label for c, label in CONFIDENCE_LABELS.items()},
        "kinds": {k.value: label for k, label in DRAFT_KIND_LABELS.items()},
        "degraded": DEGRADED_LABELS,
        "questionnaire": [{"id": q.id, "text": q.text} for q in QUESTIONNAIRE],
    }


def _config_field(key: str, prop: Mapping[str, Any]) -> dict[str, Any]:
    """一个店铺配置项的输入方式：按 ShopConfig 的 JSON Schema 推出控件类型、选项和取值范围。"""
    field: dict[str, Any] = {
        "key": key,
        "label": CONFIG_FIELD_LABELS[key],
        "hint": CONFIG_FIELD_HINTS.get(key, ""),
        "editable": key in MERCHANT_CONFIG_FIELDS,
        "locked_reason": LOCKED_CONFIG_REASONS.get(key),
    }
    items = prop.get("items") or {}
    if key == "aftersale_amount_limit":
        field.update(kind="money", min=0)
    elif "enum" in prop:
        field.update(kind="select", options=list(prop["enum"]))
    elif prop.get("type") == "integer":
        field.update(kind="int", min=prop.get("minimum"), max=prop.get("maximum"))
    elif prop.get("type") == "boolean":
        field.update(kind="bool")
    elif prop.get("type") == "array" and items.get("$ref", "").endswith("/Intent"):
        field.update(kind="intents")
    elif prop.get("type") == "array":
        field.update(kind="lines", **CONFIG_LINE_LIMITS[key])
    else:
        field.update(kind="text", max_chars=MAX_DISPLAY_NAME_CHARS)
    return field


def config_fields() -> list[dict[str, Any]]:
    """店铺配置项（按 ShopConfig 的字段顺序）：中文名、说明、控件类型、能不能改、不能改的原因。"""
    props = ShopConfig.model_json_schema()["properties"]
    return [_config_field(key, props[key]) for key in ShopConfig.model_fields]


def console_page_config(platform: Platform) -> dict[str, Any]:
    """写进商家控制台页面的配置：角色分组、页签、中文名、长度上限、门槛。都是服务端常量，没有店铺数据。"""
    return {
        "platform": platform.value,
        "platform_name": PLATFORM_NAMES.get(platform, platform.value),
        "roles": ROLE_LABELS,
        "role_groups": {
            "person": sorted(PERSON_ROLES),
            "human": sorted(HUMAN_ROLES),
            "console": sorted(CONSOLE_ROLES),
            "merchant": sorted(MERCHANT_ROLES),
            "reviewer": sorted(REVIEWER_ROLES),
            "stage_shadow": sorted(MERCHANT_ROLES | REVIEWER_ROLES),
            "stage_assist": sorted(REVIEWER_ROLES),
        },
        "tabs": {tab: sorted(roles) for tab, roles in CONSOLE_TABS.items()},
        "default_tab": {"reviewer": "review"},
        "intents": {i.value: INTENT_LABELS[i] for i in Intent},
        "knowledge_status": {s.value: KNOWLEDGE_STATUS_LABELS[s] for s in KnowledgeStatus},
        "knowledge_source": {s.value: KNOWLEDGE_SOURCE_LABELS[s] for s in KnowledgeSource},
        "policies": {k.value: POLICY_LABELS[k] for k in PolicyKey},
        "item_fields": {f: ITEM_FIELD_LABELS[f] for f in MERCHANT_ITEM_FIELDS},
        "item_limits": {f: ITEM_FIELD_LIMITS[f] for f in MERCHANT_ITEM_FIELDS},
        "sync_fields": SYNC_FIELD_LABELS,
        "config_fields": config_fields(),
        "locked_config": list(LOCKED_CONFIG_FIELDS),
        "stages": {s.value: STAGE_LABELS[s] for s in RolloutStage},
        "gate_status": {s.value: GATE_STATUS_LABELS[s] for s in GateStatus},
        "gate_checks": GATE_CHECK_LABELS,
        "dimensions": dict(DIMENSIONS),
        "score_guide": SCORE_GUIDE,
        "full_score": FULL_SCORE,
        "review_status": {s.value: REVIEW_STATUS_LABELS[s] for s in QualityReviewStatus},
        "suggestion_status": {s.value: SUGGESTION_STATUS_LABELS[s] for s in SuggestionStatus},
        "kinds": {k.value: label for k, label in DRAFT_KIND_LABELS.items()},
        "reject_reasons": {r.value: REJECT_REASON_LABELS[r] for r in RejectReason},
        "hard_rules": HARD_RULE_LABELS,
        "degraded": DEGRADED_LABELS,
        "adoption_target": PHASE0_ADOPTION_TARGET,
        "period_pattern": QUALITY_PERIOD_PATTERN,
        "platform_metrics": metric_labels(),  # §12.4 平台客服指标（报表页的接入前后对比）
        "limits": {
            "question": MAX_QUESTION_CHARS,
            "answer": MAX_ANSWER_CHARS,
            "policy": MAX_POLICY_CHARS,
            "note": MAX_REVIEW_NOTE_CHARS,
            "sample_size": DEFAULT_SAMPLE_SIZE,
            "sample_size_max": MAX_SAMPLE_SIZE,
            "launch_min_full_rate": LAUNCH_MIN_FULL_RATE,
            "min_scored": DEFAULT_MIN_SCORED,
        },
    }


# ================================================================ 售后审核工作台（§7.3.2）

AFTERSALE_PAGE_PATHS: tuple[str, ...] = ("/aftersale",)
"""售后审核工作台页面地址（相对地址 aftersale/aftersale.js、assist/assist.css 都从根目录算起）。"""

AFTERSALE_ASSET_PREFIX = "/aftersale/"
AFTERSALE_ASSETS: dict[str, str] = {
    "aftersale.js": "text/javascript",
    "aftersale.css": "text/css",
}

AFTERSALE_PAGE_FILE = "aftersale.html"

AFTERSALE_CONFIRM_RULE = (
    "售后写操作必须由人确认（§4.1 原则 3、§7.3.2）：执行前会重读平台售后单，"
    "状态或状态版本变了就不操作；同一操作带幂等键，重复点击不会重复执行；"
    "谁确认的、依据哪条规则、平台返回什么都会留痕。"
)
"""确认对话框里说明的人工确认规则。"""


def aftersale_page_config(platform: Platform) -> dict[str, Any]:
    """写进售后审核工作台页面的配置：审核状态、建议、售后单类型 / 原因 / 平台状态、R1–R8 说明、
    本平台开放的写操作和平台差异提示、角色分组、长度上限。都是服务端常量，没有店铺数据。
    每张售后单实际能做哪些操作以详情接口为准（单元有没有接写接口、店铺有没有写售后权限）。"""
    return {
        "platform": platform.value,
        "platform_name": PLATFORM_NAMES.get(platform, platform.value),
        "roles": ROLE_LABELS,
        "role_groups": {
            "person": sorted(PERSON_ROLES),
            "desk": sorted(HUMAN_ROLES),  # 和售后审核接口的角色一致（AgentDep）
            "console": sorted(CONSOLE_ROLES),
        },
        "review_states": {s.value: REVIEW_STATE_LABELS[s] for s in ReviewState},
        "state_flow": [s.value for s in STATE_FLOW],
        "terminal_states": sorted(s.value for s in TERMINAL_STATES),
        "advice": {a.value: ADVICE_LABELS[a] for a in ReviewAdvice},
        "types": {t.value: AFTERSALE_TYPE_LABELS[t] for t in AfterSaleType},
        "reasons": {r.value: AFTERSALE_REASON_LABELS[r] for r in AfterSaleReason},
        "statuses": {s.value: AFTERSALE_STATUS_LABELS[s] for s in AfterSaleStatus},
        "rules": {rule: info.model_dump() for rule, info in RULES.items()},
        "actions": {a.value: AFTERSALE_ACTION_LABELS[a] for a in AfterSaleAction},
        "platform_actions": [a.value for a in AfterSaleAction if a in PLATFORM_ACTIONS[platform]],
        "platform_note": PLATFORM_NOTES[platform],
        "refuse_hint": REFUSE_HINTS[platform],
        "reason_code_required": platform == Platform.DOUYIN,
        "next_steps": {s.value: NEXT_STEP_LABELS[s] for s in NextStep},
        "needs_action": sorted(s.value for s in NEEDS_ACTION),
        "unreviewed": UNREVIEWED,
        "confirm_rule": AFTERSALE_CONFIRM_RULE,
        "limits": {
            "note": MAX_NOTE_CHARS,
            "reason_code": MAX_ID_CHARS,
            "batch": DEFAULT_BATCH,
            "batch_max": MAX_BATCH,
            "queue": DEFAULT_QUEUE_LIMIT,
        },
    }


def render_aftersale_page(platform: Platform) -> bytes:
    """售后审核工作台页面（配置已写进数据块）。"""
    return _render(AFTERSALE_PAGE_FILE, aftersale_page_config(platform))


def json_for_html(data: Any) -> str:
    """JSON → 可以放进 <script type="application/json"> 的文字：转义 < > &，
    内容里出现 "</script>" 也不会提前结束数据块。"""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _static(name: str) -> bytes:
    return resources.files("rodit.api").joinpath("static", name).read_bytes()


def _render(page_file: str, config: dict[str, Any]) -> bytes:
    html = _static(page_file).decode("utf-8")
    if html.count(CONFIG_PLACEHOLDER) != 1:
        raise RuntimeError("页面模板里的配置占位符必须恰好出现一次")
    return html.replace(CONFIG_PLACEHOLDER, json_for_html(config)).encode("utf-8")


def render_page(platform: Platform) -> bytes:
    """话术助手页面（配置已写进数据块）。"""
    return _render(PAGE_FILE, page_config(platform))


def render_console_page(platform: Platform) -> bytes:
    """商家控制台页面（配置已写进数据块）。"""
    return _render(CONSOLE_PAGE_FILE, console_page_config(platform))


# 页面：（地址，渲染函数）；静态文件：（地址前缀，文件 → 内容类型）
_PAGES: tuple[tuple[tuple[str, ...], Callable[[Platform], bytes]], ...] = (
    (PAGE_PATHS, render_page),
    (CONSOLE_PAGE_PATHS, render_console_page),
    (AFTERSALE_PAGE_PATHS, render_aftersale_page),
)
_ASSET_GROUPS: tuple[tuple[str, dict[str, str]], ...] = (
    (ASSET_PREFIX, ASSETS),
    (CONSOLE_ASSET_PREFIX, CONSOLE_ASSETS),
    (AFTERSALE_ASSET_PREFIX, AFTERSALE_ASSETS),
)


def _responder(
    body: bytes, media_type: str, headers: dict[str, str] = SECURITY_HEADERS
) -> Callable[[], Response]:
    def serve() -> Response:
        return Response(content=body, media_type=media_type, headers=headers)

    return serve


def build_web_router(platform: Platform) -> APIRouter:
    """网页工具的路由（不进 OpenAPI 文档）。文件在这里读一次，之后只从内存返回。"""
    router = APIRouter(include_in_schema=False)
    for paths, render in _PAGES:
        page = _responder(render(platform), "text/html", PAGE_HEADERS)
        for path in paths:
            router.add_api_route(path, page, methods=["GET"], name=f"web_page{path}")
    for prefix, assets in _ASSET_GROUPS:
        for name, media_type in assets.items():
            router.add_api_route(
                prefix + name,
                _responder(_static(name), media_type),
                methods=["GET"],
                name=f"web_asset_{name}",
            )
    return router


__all__ = [
    "AFTERSALE_ASSETS",
    "AFTERSALE_ASSET_PREFIX",
    "AFTERSALE_CONFIRM_RULE",
    "AFTERSALE_PAGE_PATHS",
    "ASSETS",
    "ASSET_PREFIX",
    "CONSOLE_ASSETS",
    "CONSOLE_ASSET_PREFIX",
    "CONSOLE_PAGE_PATHS",
    "CONSOLE_TABS",
    "CONTENT_SECURITY_POLICY",
    "HARD_RULE_LABELS",
    "PAGE_HEADERS",
    "PAGE_PATHS",
    "PHASE0_ADOPTION_TARGET",
    "SECURITY_HEADERS",
    "aftersale_page_config",
    "build_web_router",
    "config_fields",
    "console_page_config",
    "json_for_html",
    "page_config",
    "render_aftersale_page",
    "render_console_page",
    "render_page",
]
