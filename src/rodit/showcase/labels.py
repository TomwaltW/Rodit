"""演示站页面上的中文名和白话说明（方案原文的意思，给商家、潜在客户看）。

能复用的中文名都从各模块取（问题类型、草稿类型、硬规则、售后状态……），这里只补演示站自己
要讲解的：§6.6 八条路由规则的白话、§7.3.2 售后规则和审核状态、证据来源的优先级（§5.2.1）、
各环节名称（§12.1）。
"""

from __future__ import annotations

from rodit.core.config import QAGateConfig, RoutingConfig
from rodit.core.enums import EvidenceSource, ReviewAdvice, ReviewState, RouteAction

ROUTE_ACTION_LABELS: dict[RouteAction, str] = {
    RouteAction.OFF_PLATFORM_SCRIPT: "推荐平台规则标准话术",
    RouteAction.HUMAN_PRIORITY: "不推荐，提醒客服优先处理",
    RouteAction.SOOTHE: "推荐安抚话术并标红",
    RouteAction.HIGH_RISK_ANSWER: "推荐 + 风险提示",
    RouteAction.SPLIT_MULTI: "按问题分条推荐",
    RouteAction.ANSWER: "检索、生成、质检，出推荐",
    RouteAction.CLARIFY: "推荐一句澄清问题",
    RouteAction.NO_SUGGESTION: "不推荐，客服自行处理",
    RouteAction.TRANSFER_HUMAN: "转人工",
}


def rule_texts(routing: RoutingConfig) -> dict[int, str]:
    """§6.6 路由规则（按顺序判断，命中就停）的白话，阈值取生效配置。"""
    return {
        1: f"买家要站外联系或站外交易（概率 ≥ {routing.off_platform:g}）：推荐平台规则标准话术，"
        "优先级最高，防止违规",
        2: f"买家要求人工或要投诉（概率 ≥ {routing.need_human:g}）：提醒客服优先处理",
        3: f"买家情绪激烈（0–4 分 ≥ {routing.emotion_escalate:g}）：推荐安抚话术并标红",
        4: "问题属于高风险类型（价格、真伪、改单、质量、签收未收到、投诉）：推荐 + 风险提示",
        5: f"一条消息里有两个及以上问题（概率 ≥ {routing.multi_intent:g}）：按问题分条推荐",
        6: f"问题类型的置信度达到智能辅助阈值（≥ {routing.assist_threshold:g}）："
        "检索、生成、质检，出推荐",
        7: f"置信度低、前两名相差不到 {routing.clarify_max_gap:g}：推荐一句澄清问题",
        8: "其他情况：不推荐，客服自行处理",
    }


EMOTION_LEVELS: tuple[str, ...] = ("平静", "略有不满", "不满", "愤怒", "极度愤怒")
"""§6.5 情绪打分的五档（0–4 分）。"""


def emotion_label(score: float) -> str:
    return EMOTION_LEVELS[max(0, min(len(EMOTION_LEVELS) - 1, round(score)))]


SOURCE_LABELS: dict[EvidenceSource, str] = {
    EvidenceSource.REALTIME: "实时接口",
    EvidenceSource.PLATFORM: "平台规则",
    EvidenceSource.POLICY: "店铺政策",
    EvidenceSource.ITEM: "商品知识",
    EvidenceSource.FAQ: "FAQ 与优质回答",
    EvidenceSource.CATEGORY: "类目通用知识",
}
"""证据来源（§5.2.1 优先级：数字越小越优先，冲突时高优先级覆盖低优先级）。"""

PROVIDER_LABELS: dict[str, str] = {
    "rules": "关键词规则（离线替身）",
    "offline_heuristic": "启发式质检（离线替身）",
    "incloud_llm": "平台云内大模型（投票）",
    "classifier": "路由分类器",
    "jev": "Jev",
}

QA_QUESTION_LABELS: dict[str, str] = {
    "grounded": "有依据",
    "answers_question": "切题",
    "promise_risk": "承诺风险",
    "platform_violation": "平台违规",
    "tone": "语气（0–4）",
}
"""§5.5 第二道质检的题目。"""


def qa_thresholds(gate: QAGateConfig) -> dict[str, dict[str, float | str]]:
    """质检放行阈值（§5.5、§6.6 qa_gate）：min 表示不能低于，max 表示不能高于。"""
    return {
        "grounded": {"min": gate.grounded_min},
        "answers_question": {"min": gate.answers_question_min},
        "promise_risk": {"max": gate.promise_risk_max},
        "platform_violation": {"max": gate.platform_violation_max},
    }


LATENCY_STAGES: dict[str, str] = {
    "understand": "会话理解",
    "route": "懂 · 路由",
    "retrieve": "找 · 检索与工具",
    "generate": "写 · 生成",
    "qa": "判 · 质检",
}
"""引擎各环节（EngineResult.latency_ms 的键，§12.1）。"""

E2E_TARGET_MS = 5000
"""推荐出现（端到端）P95 ≤ 5 秒（§1.3、§12.1）。"""

REVIEW_STATE_LABELS: dict[ReviewState, str] = {
    ReviewState.NEW: "待审核",
    ReviewState.SUGGEST: "规则建议",
    ReviewState.VERIFY: "签收核查中",
    ReviewState.EVIDENCE: "等待凭证",
    ReviewState.CONFIRM: "待客服确认",
    ReviewState.NEGOTIATE: "协商中",
    ReviewState.WAIT_RETURN: "退货中",
    ReviewState.AGREED: "已同意",
    ReviewState.REFUSED: "已拒绝",
    ReviewState.PLATFORM: "平台已处理",
}
"""§7.3.2 审核状态机（另加"退货中"：同意退货后等买家寄回）。"""

STATE_STRIP: tuple[ReviewState, ...] = (
    ReviewState.NEW,
    ReviewState.SUGGEST,
    ReviewState.VERIFY,
    ReviewState.EVIDENCE,
    ReviewState.CONFIRM,
    ReviewState.NEGOTIATE,
    ReviewState.WAIT_RETURN,
    ReviewState.AGREED,
    ReviewState.REFUSED,
    ReviewState.PLATFORM,
)
"""页面状态条的顺序。"""

REVIEW_ADVICE_LABELS: dict[ReviewAdvice, str] = {
    ReviewAdvice.AGREE: "建议同意",
    ReviewAdvice.AGREE_RETURN: "建议同意退货（仓库收货后再同意退款）",
    ReviewAdvice.REFUSE: "建议拒绝",
    ReviewAdvice.NEGOTIATE: "建议先协商",
    ReviewAdvice.MANUAL: "人工决定",
    ReviewAdvice.NONE: "不操作，只同步状态",
}

AFTERSALE_RULE_TEXTS: dict[str, str] = {
    "R1": "售后单已不是待审核状态：不操作，同步状态",
    "R2": "订单未发货、申请仅退款：建议同意（平台也可能自动处理）",
    "R3": "签收 7 天内、支持七天无理由、申请退货退款：建议同意退货，仓库收到后同意退款",
    "R4": "物流显示签收、买家说没收到：先协商、发起快递核查，24 小时后仍未找到建议同意",
    "R5": "质量问题、破损、错漏发：需要凭证，客服查看后决定",
    "R6": "金额超过店铺设定的上限：必须店长确认",
    "R7": "同一买家 30 天内售后 ≥ 3 次：标记风险，人工决定",
    "R8": "以上都不命中：人工决定",
}
"""§7.3.2 审核规则的白话。"""

REFUSE_REASON_CODES: dict[str, str] = {
    "goods_used": "商品已使用、影响二次销售",
    "no_evidence": "买家未提供有效凭证",
    "out_of_period": "超过售后期限",
}
"""演示用的拒绝原因码（合成；抖店拒绝售后必须传原因码，§7.1，实际码表接入时按平台文档填）。"""

__all__ = [
    "AFTERSALE_RULE_TEXTS",
    "E2E_TARGET_MS",
    "EMOTION_LEVELS",
    "LATENCY_STAGES",
    "PROVIDER_LABELS",
    "QA_QUESTION_LABELS",
    "REFUSE_REASON_CODES",
    "REVIEW_ADVICE_LABELS",
    "REVIEW_STATE_LABELS",
    "ROUTE_ACTION_LABELS",
    "SOURCE_LABELS",
    "STATE_STRIP",
    "emotion_label",
    "qa_thresholds",
    "rule_texts",
]
