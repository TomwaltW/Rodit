"""实际发出去的文字的合规扫描（方案 §13.4 "平台违规为 0" 的代理指标、§5.5 硬规则、§2.1 话术禁区、
§12.3 广告法）。

平台下发的违规通知由复核人员按周登记（shop.rollout_incident，见 feedback.rollout）；这里扫的是我们
自己看得到的、实际发出去的文字，作为同一条门槛的代理指标（两者分开报告）：

- 客服编辑后发送的推荐：SuggestionRecord（EDITED_SENT）的 edited_text；
- 会话里客服、机器人发出的消息：ConversationRepository.list_outgoing（原样发送的推荐、
  客服自己打的字、全自动回复都在这里）。

规则和质检硬规则（rodit.qa.hard_rules）同一份词表、同样的规则名：站外引流（core.lexicon
find_off_platform_in_answer；另外存下来的文字做过个人信息最小化时，遮住的手机号、联系方式、邮箱占位符
也算）、绝对化用语（EngineConfig.qa_gate 的词表和例外）、店铺禁用词（生效配置的 banned_words）。
没有依据原文，成分标注里的 100%（"桑蚕丝 100%"）这里也会命中——只是代理指标，由复核人员按 ID 核对。

同一次发送的两份记录只算一次（记在推荐上）：写回会话的那条（msg_id "agent:{推荐 id}"），或同一会话
里文字相同的一条消息（平台回声；比较前 NFKC 归一、遮个人信息、去掉空白），每条推荐最多配一条。

输出按店铺、按 ISO 周（东八区）：扫描条数、疑似违规条数、各规则命中数、推荐 ID / 消息 ID 和命中的
规则名。不含任何文字（买家的、客服的都没有），可以写进审计（compliance.flag）和报表。纯函数，不读写仓储。
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.core.config import QAGateConfig
from rodit.core.enums import Role, SuggestionStatus
from rodit.core.lexicon import find_absolute_terms, find_off_platform_in_answer
from rodit.core.models import ShopConfig, StoredMessage, SuggestionRecord, strip_control_chars
from rodit.core.pii import CONTACT_MASK, EMAIL_MASK, PHONE_MASK, mask_pii
from rodit.feedback.review import iso_period, week_start

RULE_OFF_PLATFORM = "off_platform"
RULE_ABSOLUTE = "absolute_term"
RULE_BANNED = "banned_word"
COMPLIANCE_RULES: tuple[str, ...] = (RULE_OFF_PLATFORM, RULE_ABSOLUTE, RULE_BANNED)
"""扫描用的规则（和 core.models.HardRuleHit.rule 同名），输出按这个顺序。"""

OUTGOING_ROLES: tuple[Role, ...] = (Role.AGENT, Role.BOT)
MAX_LISTED_IDS = 100
"""汇总里最多列出多少个推荐 ID / 消息 ID（计数不受影响）。"""

_CONTACT_PLACEHOLDERS = (PHONE_MASK, CONTACT_MASK, EMAIL_MASK)
_WRITE_BACK_PREFIX = "agent:"  # 反馈写回会话的消息 ID：agent:{推荐 id}（rodit.app.service）


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComplianceRules(_Model):
    """扫描用的词表：绝对化用语和例外来自 EngineConfig.qa_gate，禁用词来自店铺的生效配置。"""

    absolute_terms: list[str]
    absolute_term_exceptions: list[str]
    banned_words: list[str] = Field(default_factory=list)

    @classmethod
    def from_config(cls, qa: QAGateConfig, shop: ShopConfig) -> ComplianceRules:
        return cls(
            absolute_terms=list(qa.absolute_terms),
            absolute_term_exceptions=list(qa.absolute_term_exceptions),
            banned_words=[w.strip() for w in shop.banned_words if w and w.strip()],
        )


def scan_text(text: str, rules: ComplianceRules) -> list[str]:
    """一段发出去的文字命中的规则名（按 COMPLIANCE_RULES 的顺序，不重复）。"""
    text = text or ""
    hits: set[str] = set()
    if find_off_platform_in_answer(text) or any(p in text for p in _CONTACT_PLACEHOLDERS):
        hits.add(RULE_OFF_PLATFORM)
    if find_absolute_terms(
        text, terms=rules.absolute_terms, exceptions=rules.absolute_term_exceptions
    ):
        hits.add(RULE_ABSOLUTE)
    norm = unicodedata.normalize("NFKC", text).lower()
    if any(unicodedata.normalize("NFKC", w).lower() in norm for w in rules.banned_words):
        hits.add(RULE_BANNED)
    return [r for r in COMPLIANCE_RULES if r in hits]


# ================================================================ 按周扫描


class ComplianceFlag(_Model):
    """一条疑似违规的发送：推荐 ID 或消息 ID + 命中的规则。没有文字。"""

    source: Literal["edited_suggestion", "message"]
    suggestion_id: int | None = None
    msg_id: str | None = None
    role: Role | None = None  # 消息：客服还是机器人
    rules: list[str]
    at: AwareDatetime


class ComplianceWeek(_Model):
    """一个店铺一个 ISO 周（和窗口的交集）的扫描结果。flagged 是疑似违规的发送条数。"""

    shop_id: str
    period: str
    since: AwareDatetime
    until: AwareDatetime
    scanned_suggestions: int = 0
    scanned_messages: int = 0
    duplicates: int = 0  # 和编辑后发送的推荐是同一次发送、没有重复计数的消息
    flagged: int = 0
    rule_counts: dict[str, int] = Field(default_factory=dict)
    flags: list[ComplianceFlag] = Field(default_factory=list)


def _send_key(text: str | None) -> str:
    """同一次发送的比较形式：去控制字符、NFKC 归一、遮个人信息、去掉所有空白。"""
    cleaned = unicodedata.normalize("NFKC", strip_control_chars(text or ""))
    return "".join(mask_pii(cleaned).split())


def _in(at: datetime, since: datetime, until: datetime) -> bool:
    return since <= at < until


def _weeks(since: datetime, until: datetime) -> list[tuple[str, datetime, datetime]]:
    out: list[tuple[str, datetime, datetime]] = []
    cursor = week_start(since)
    while cursor < until:
        end = cursor + timedelta(days=7)
        out.append((iso_period(cursor), max(cursor, since), min(end, until)))
        cursor = end
    return out


def scan_sent(
    records: Iterable[SuggestionRecord],
    messages: Iterable[StoredMessage],
    *,
    shop_id: str,
    rules: ComplianceRules,
    since: datetime,
    until: datetime,
) -> list[ComplianceWeek]:
    """扫描 [since, until) 里这家店实际发出去的文字，按 ISO 周返回（时间升序，没有数据的周也列出）。

    records：这家店的推荐记录（只取 EDITED_SENT 且有 edited_text 的，按 created_at 落到周）；
    messages：ConversationRepository.list_outgoing 的结果（只取客服、机器人的，按 created_at）。
    别的店铺、窗口外的记录忽略。"""
    if since >= until:
        raise ValueError("since 要早于 until")
    edited = [
        r
        for r in records
        if r.shop_id == shop_id
        and r.status == SuggestionStatus.EDITED_SENT
        and r.edited_text
        and _in(r.created_at, since, until)
    ]
    outgoing = [
        m
        for m in messages
        if m.shop_id == shop_id
        and m.role in OUTGOING_ROLES
        and m.text
        and _in(m.created_at, since, until)
    ]
    # 同一次发送：先按写回的消息 ID 配，再按同一会话里文字相同的配（每条推荐最多配一条）
    open_by_id = {r.id: r for r in edited if r.id is not None}
    open_by_text: dict[tuple[str, str], list[int]] = {}
    for r in edited:
        if r.id is not None:
            open_by_text.setdefault((r.conversation_id, _send_key(r.edited_text)), []).append(r.id)
    duplicate_ids: set[str] = set()
    for m in outgoing:
        if m.msg_id.startswith(_WRITE_BACK_PREFIX):
            sid = m.msg_id[len(_WRITE_BACK_PREFIX) :]
            if sid.isdigit() and int(sid) in open_by_id:
                open_by_id.pop(int(sid))
                duplicate_ids.add(m.msg_id)
    for m in outgoing:
        if m.msg_id in duplicate_ids:
            continue
        candidates = open_by_text.get((m.conversation_id, _send_key(m.text)), [])
        while candidates:
            sid = candidates.pop(0)
            if open_by_id.pop(sid, None) is not None:
                duplicate_ids.add(m.msg_id)
                break

    weeks: list[ComplianceWeek] = []
    for period, start, end in _weeks(since, until):
        flags: list[ComplianceFlag] = []
        week_edited = [r for r in edited if _in(r.created_at, start, end)]
        week_messages = [m for m in outgoing if _in(m.created_at, start, end)]
        for r in week_edited:
            hit = scan_text(r.edited_text or "", rules)
            if hit:
                flags.append(
                    ComplianceFlag(
                        source="edited_suggestion", suggestion_id=r.id, rules=hit, at=r.created_at
                    )
                )
        duplicates = 0
        for m in week_messages:
            if m.msg_id in duplicate_ids:
                duplicates += 1
                continue
            hit = scan_text(m.text or "", rules)
            if hit:
                flags.append(
                    ComplianceFlag(
                        source="message", msg_id=m.msg_id, role=m.role, rules=hit, at=m.created_at
                    )
                )
        flags.sort(key=lambda f: f.at)
        counts = Counter(rule for f in flags for rule in f.rules)
        weeks.append(
            ComplianceWeek(
                shop_id=shop_id,
                period=period,
                since=start,
                until=end,
                scanned_suggestions=len(week_edited),
                scanned_messages=len(week_messages),
                duplicates=duplicates,
                flagged=len(flags),
                rule_counts={r: counts[r] for r in COMPLIANCE_RULES if counts[r]},
                flags=flags,
            )
        )
    return weeks


# ================================================================ 汇总


class ComplianceSummary(_Model):
    """一段窗口的扫描汇总（上线状态、验收报告用）：计数和 ID，没有文字。"""

    since: AwareDatetime
    until: AwareDatetime
    scanned_suggestions: int = 0
    scanned_messages: int = 0
    flagged: int = 0  # 疑似违规的发送条数（代理指标，门槛要求为 0）
    rule_counts: dict[str, int] = Field(default_factory=dict)
    by_period: dict[str, int] = Field(default_factory=dict)  # 各周疑似违规条数
    suggestion_ids: list[int] = Field(default_factory=list)
    message_ids: list[str] = Field(default_factory=list)
    ids_truncated: bool = False  # ID 超过 MAX_LISTED_IDS 条时只列前面的


def summarize_scan(
    weeks: Sequence[ComplianceWeek], *, max_ids: int = MAX_LISTED_IDS
) -> ComplianceSummary:
    """按周的扫描结果 → 窗口汇总（weeks 时间升序，至少一周）。"""
    if not weeks:
        raise ValueError("没有扫描结果")
    flags = [f for w in weeks for f in w.flags]
    counts = Counter(rule for f in flags for rule in f.rules)
    suggestion_ids = [f.suggestion_id for f in flags if f.suggestion_id is not None]
    message_ids = [f.msg_id for f in flags if f.msg_id is not None]
    return ComplianceSummary(
        since=weeks[0].since,
        until=weeks[-1].until,
        scanned_suggestions=sum(w.scanned_suggestions for w in weeks),
        scanned_messages=sum(w.scanned_messages for w in weeks),
        flagged=sum(w.flagged for w in weeks),
        rule_counts={r: counts[r] for r in COMPLIANCE_RULES if counts[r]},
        by_period={w.period: w.flagged for w in weeks},
        suggestion_ids=suggestion_ids[:max_ids],
        message_ids=message_ids[:max_ids],
        ids_truncated=len(suggestion_ids) > max_ids or len(message_ids) > max_ids,
    )


def flag_detail(week: ComplianceWeek) -> dict[str, object]:
    """写进 compliance.flag 审计事件的 detail：周期、计数、各规则命中数、
    推荐 / 消息 ID 和规则名（没有文字）。"""
    return {
        "period": week.period,
        "since": week.since.isoformat(),
        "until": week.until.isoformat(),
        "scanned_suggestions": week.scanned_suggestions,
        "scanned_messages": week.scanned_messages,
        "duplicates": week.duplicates,
        "flagged": week.flagged,
        "rule_counts": dict(week.rule_counts),
        "flags": [
            {
                "suggestion_id": f.suggestion_id,
                "msg_id": f.msg_id,
                "role": f.role.value if f.role is not None else None,
                "rules": list(f.rules),
            }
            for f in week.flags[:MAX_LISTED_IDS]
        ],
        "flags_truncated": len(week.flags) > MAX_LISTED_IDS,
    }


__all__ = [
    "COMPLIANCE_RULES",
    "MAX_LISTED_IDS",
    "OUTGOING_ROLES",
    "RULE_ABSOLUTE",
    "RULE_BANNED",
    "RULE_OFF_PLATFORM",
    "ComplianceFlag",
    "ComplianceRules",
    "ComplianceSummary",
    "ComplianceWeek",
    "flag_detail",
    "scan_sent",
    "scan_text",
    "summarize_scan",
]
