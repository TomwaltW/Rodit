"""商家控制台的知识编辑与店铺配置（方案 §5.2 知识中心、§5.2.1 政策改动即时生效、§5.2.2 商品卡片、
§9.1 知识编辑跟着单元走、§9.3 配置分层）。只有领域逻辑，不含 HTTP；只依赖 rodit.core。

FAQ（§5.2.1 优先级 4）：
- 商家录入：source=MERCHANT、status=APPROVED、approved_by=商家本人，立刻可检索；
- 修改：旧条目下线（比较后更新，只从 APPROVED 下线）、新条目上线，**不在原 id 上改写**——
  推荐记录里的证据 ID（如 faq:12）必须一直指向买家当时看到的那段话（§4.1 原则 6 留痕）；
  来源（source）表示来源渠道，修改不改变它；
- 审核候选时顺手修改（approve-edited）：没有改动就是普通的审核通过（原地 CANDIDATE → APPROVED）；
  有改动时候选条目下线、改过的内容作为新条目上线（审计里记候选 id 和新 id）。
  这样只用已有的三种状态，候选条目的内容也保持客服当时写的原样；
- 保存前检查（不拦截，只提醒商家）：回答时会不会被质检拦截（站外联系、绝对化用语、禁用词……，
  检查函数由组装层注入，见 AnswerChecker）、和店铺政策冲突（回答时以政策为准，这条 FAQ 用不上）、
  和已上线 FAQ 的问题重复。

店铺政策（§5.2.1 优先级 2）：只能写已知的政策键（PolicyKey），改动即时生效——检索每次都从仓储读
政策（knowledge.retrieval），会话理解每条消息都重新读店铺配置，中间没有缓存。

商品卡片：商家只能改自己填的字段（size_advice、care、usage、forbidden_claims）；标题、属性、SKU、
尺码表、详情文字属于平台，只由商品同步（knowledge.sync）更新。

店铺配置（§9.3）：商家能改的是语气、称呼、回答长度、禁用词、关闭的问题类型、推荐条数、售后金额上限、
店长账号、客服工作时间（北京时间，附录 C 第 13 题；全自动默认只在工作时间以外生效，§15 Phase 3）；
影子模式、全自动开关和生效时段、半自动的类型属于上线节奏和平台准入（§13.4），商家不能改。值为 None
表示删掉店铺覆盖项、回到类目 / 平台默认。

所有写操作写审计（core.audit.AuditAction），detail 只放 ID、字段名、条数，不放问答原文和取值。
调用方负责和取消订购互斥（app.console 在 processing(shop_id) 里调用）；这里每次写之前都重新确认店铺
还在（已删除或停用抛 ShopInactive），删掉的店铺不会被写回来。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict

from rodit.core import evidence as ev
from rodit.core.audit import AuditAction
from rodit.core.clock import Clock
from rodit.core.config import EngineConfig, effective_shop_config, resolve_shop_config
from rodit.core.enums import (
    POLICY_LABELS,
    EvidenceSource,
    Intent,
    KnowledgeSource,
    KnowledgeStatus,
    PolicyKey,
    ShopStatus,
)
from rodit.core.errors import (
    ActionNotAllowed,
    ConcurrencyConflict,
    ShopInactive,
    TenantIsolationError,
)
from rodit.core.models import (
    SHOP_CONFIG_FIELDS,
    AuditEvent,
    Evidence,
    ItemCard,
    KnowledgeItem,
    PolicyEntry,
    Shop,
    ShopConfig,
    strip_control_chars,
)
from rodit.core.ports import AuditSink, KnowledgeRepository
from rodit.core.schedule import MAX_SERVICE_WINDOWS, normalize_service_hours
from rodit.knowledge.conflict import Candidate, ordered_topics, resolve_conflicts
from rodit.knowledge.questionnaire import (
    MAX_BANNED_WORD_CHARS,
    MAX_SERVICE_HOURS_TEXT,
    parse_banned_words,
    parse_service_hours_text,
    question_for_policy,
)

# 检索器里的证据构造：保存前的冲突检查必须和回答时用同一份（同一个包内的私有函数）
from rodit.knowledge.retrieval import _faq_candidate, _policy_evidence
from rodit.knowledge.search import normalize_question

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 200
MAX_ANSWER_CHARS = 1000
MAX_POLICY_CHARS = 1000
MAX_SIZE_ADVICE_CHARS = 500
MAX_ITEM_LINE_CHARS = 200  # care / usage 每一条
MAX_CLAIM_CHARS = 50  # forbidden_claims 每一条
MAX_ITEM_LIST = 30
MAX_DISPLAY_NAME_CHARS = 64
MAX_BANNED_WORDS = 100
MAX_MANAGER_IDS = 50
MAX_ACCOUNT_CHARS = 64
MAX_SERVICE_HOURS_CHARS = 20  # 客服工作时间每一行（"早上9点到晚上11点半"），同设置页的输入上限

_NOT_HUMAN = frozenset({"system", "bot", "auto"})

FAQ_FIELDS: tuple[str, ...] = ("question", "answer", "item_id", "intent")
"""FAQ 上商家能填、能改的字段。"""

MERCHANT_ITEM_FIELDS: tuple[str, ...] = ("size_advice", "care", "usage", "forbidden_claims")
"""商品卡片上商家填写的字段（§5.2.2）；其余字段属于平台，只由商品同步更新。"""

MERCHANT_CONFIG_FIELDS: tuple[str, ...] = (
    "display_name",
    "tone",
    "salutation",
    "max_answer_chars",
    "banned_words",
    "disabled_intents",
    "suggestion_count",
    "aftersale_amount_limit",
    "manager_ids",
    "service_hours",
)
"""商家能改的店铺配置项（§9.3）。白名单：core 以后新增的配置项默认不开放给商家。"""

LOCKED_CONFIG_FIELDS: tuple[str, ...] = (
    "shadow_mode",
    "auto_reply_enabled",
    "auto_reply_schedule",
    "semi_auto_intents",
)
"""商家不能改的配置项：上线阶段（影子 / 智能辅助，§13.4）、全自动总开关和生效时段（平台准入，
§15 Phase 3 先在非工作时间试行）、半自动开放的类型（按门槛由复核人员开启，§13.4）。"""

assert set(MERCHANT_CONFIG_FIELDS) <= SHOP_CONFIG_FIELDS
assert set(LOCKED_CONFIG_FIELDS) <= SHOP_CONFIG_FIELDS

MSG_LOCKED_CONFIG = (
    "影子模式和全自动开关由上线节奏和平台准入决定（§13.4），商家不能直接修改：{keys}"
)

_POLICY_TOPICS = frozenset(ev.topic_for_policy(k.value) for k in PolicyKey)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ================================================================ 公共


def require_human(value: object, what: str = "操作人") -> str:
    """写操作必须有具体的人（§4.1 原则 3）：不能为空，也不能是 system / bot / auto。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what}不能为空")
    if value.strip().lower() in _NOT_HUMAN:
        raise ValueError(f"{what}必须是具体的人")
    return value.strip()


def require_shop(repo: KnowledgeRepository, shop_id: str) -> Shop:
    """店铺必须还在、没有停用（§8.4 取消订购后删除的店铺不能被写回来）。"""
    shop = repo.get_shop(shop_id)
    if shop is None:
        raise ShopInactive(shop_id)
    if shop.id != shop_id:
        raise TenantIsolationError("知识仓储返回了别的店铺")
    if shop.status == ShopStatus.SUSPENDED:
        raise ShopInactive(shop_id, shop.status)
    return shop


def clean_text(value: object, what: str, max_chars: int) -> str:
    """商家填写的文字：去掉 NUL 等控制字符和首尾空白；不能为空、不能超长。"""
    if value is None:
        raise ValueError(f"{what}不能为空")
    if not isinstance(value, str):
        raise ValueError(f"{what}必须是文字")
    text = strip_control_chars(value).strip()
    if not text:
        raise ValueError(f"{what}不能为空")
    if len(text) > max_chars:
        raise ValueError(f"{what}最多 {max_chars} 个字")
    return text


def clean_lines(
    value: object, what: str, *, max_chars: int, max_items: int = MAX_ITEM_LIST
) -> list[str]:
    """文字列表（洗涤说明、禁用词……）：逐条清洗，去掉空的，去重保序。None 表示清空。"""
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise ValueError(f"{what}必须是文字列表")
    out: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            raise ValueError(f"{what}必须是文字列表")
        text = strip_control_chars(raw).strip()
        if not text or text in out:
            continue
        if len(text) > max_chars:
            raise ValueError(f"{what}每一条最多 {max_chars} 个字")
        out.append(text)
    if len(out) > max_items:
        raise ValueError(f"{what}最多 {max_items} 条")
    return out


# ================================================================ FAQ


class FaqWarning(_Model):
    """保存 FAQ 时给商家的提醒（不拦截保存）。

    rule：质检硬规则名（off_platform、absolute_term、banned_word、forbidden_claim、time_promise、
    length、format……，见 core.models.HardRuleHit），或 policy_conflict（和店铺政策冲突）、
    duplicate_question（和已上线 FAQ 的问题重复）。
    blocks_answer：回答时这条 FAQ 会被质检拦截、或被店铺政策覆盖——实际用不上，建议修改。
    """

    rule: str
    detail: str
    blocks_answer: bool


@dataclass(frozen=True)
class FaqCheck:
    """保存前质检用的材料：和回答时一样的店铺生效配置、当前商品、证据。

    - evidence：这条 FAQ 在回答时的证据（id 用 faq:0 占位，内容和检索器构造的一致）；
    - reference：店铺政策和商品卡片的证据（不含这条 FAQ），判断回答里的承诺有没有别的依据。
    """

    shop: Shop
    config: ShopConfig
    question: str
    answer: str
    item: ItemCard | None
    evidence: Evidence
    reference: list[Evidence]
    now: datetime


AnswerChecker = Callable[[FaqCheck], list[FaqWarning]]
"""保存前的回答检查（由组装层注入，如质检硬规则 rodit.qa.check_hard_rules）。"""


@dataclass(frozen=True)
class FaqChanges:
    """修改 FAQ：只改 provided 里列出的字段。

    没列出的保持原样；item_id / intent 列出且为 None 表示清空（改成通用 FAQ / 不限问题类型）。
    """

    question: str | None = None
    answer: str | None = None
    item_id: str | None = None
    intent: Intent | None = None
    provided: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def of(cls, **values: Any) -> FaqChanges:
        unknown = sorted(set(values) - set(FAQ_FIELDS))
        if unknown:
            raise ValueError(f"FAQ 没有这些字段：{unknown}")
        return cls(**values, provided=frozenset(values))


class FaqChange(_Model):
    """一次 FAQ 写操作的结果：上线的条目、下线的旧条目（修改时）、给商家的提醒。"""

    item: KnowledgeItem
    retired: KnowledgeItem | None = None
    warnings: list[FaqWarning] = []


@dataclass(frozen=True)
class _FaqContent:
    question: str
    answer: str
    item_id: str | None
    intent: Intent | None

    def changed_fields(self, item: KnowledgeItem) -> list[str]:
        return [f for f in FAQ_FIELDS if getattr(self, f) != getattr(item, f)]


# ================================================================ 政策、配置的读视图


class PolicyRow(_Model):
    """一项店铺政策（控制台列表）。value 为 None 表示还没填（附录 C 问卷第几题填）。"""

    key: str
    label: str
    value: str | None = None
    updated_at: AwareDatetime | None = None
    questionnaire_id: int | None = None
    known: bool = True  # 不是 PolicyKey 的旧数据为 False（只读，不能再写）


class ShopConfigSnapshot(_Model):
    """店铺配置（§9.3）：生效值、不含店铺覆盖项时的默认值、店铺覆盖项、哪些能改。"""

    effective: ShopConfig
    defaults: ShopConfig  # 平台默认 → 类目默认（删掉覆盖项后回到这里）
    overrides: dict[str, Any]
    editable: list[str]
    locked: list[str]
    changed: list[str] = []  # 本次改了（新增或改值）的覆盖项
    removed: list[str] = []  # 本次删掉的覆盖项


# ================================================================ 入口


class KnowledgeAdmin:
    """商家控制台的知识编辑与店铺配置。

    - on_faq_changed(shop_id)：已上线 FAQ 变了之后调用（组装层接 FaqSearcher.index，新 FAQ 立刻
      有向量）；失败只记日志——检索前还会懒索引补上；
    - answer_checker：保存 FAQ 前的回答检查（见 AnswerChecker），为空时只做知识中心自己的检查；
    - lock：商品卡片、店铺配置"读—改—写"用的锁，和商品同步共用一把（同一进程内不互相覆盖）。
    """

    def __init__(
        self,
        repo: KnowledgeRepository,
        audit: AuditSink,
        clock: Clock,
        *,
        config: EngineConfig,
        on_faq_changed: Callable[[str], object] | None = None,
        answer_checker: AnswerChecker | None = None,
        lock: threading.Lock | None = None,
    ) -> None:
        self._repo = repo
        self._audit = audit
        self._clock = clock
        self._config = config
        self._on_faq_changed = on_faq_changed
        self._checker = answer_checker
        self._lock = lock or threading.Lock()

    # ================================================================ FAQ

    def list_faqs(
        self,
        shop_id: str,
        *,
        status: KnowledgeStatus | None = None,
        item_id: str | None = None,
    ) -> list[KnowledgeItem]:
        """本店的 FAQ（按 id 升序）。status 为空不过滤状态；item_id 只列这个商品专属的 FAQ。"""
        items = self._repo.list_knowledge(shop_id, status=status, item_id=item_id)
        for k in items:
            if k.shop_id != shop_id:
                raise TenantIsolationError("知识仓储返回了别的店铺的 FAQ")
        if item_id is not None:
            items = [k for k in items if k.item_id == item_id]
        return items

    def check_faq(
        self,
        shop_id: str,
        *,
        question: str,
        answer: str,
        item_id: str | None = None,
        exclude_id: int | None = None,
    ) -> list[FaqWarning]:
        """保存前检查（只读）：回答时会不会被拦截、和政策冲突、问题重复。"""
        shop = require_shop(self._repo, shop_id)
        content = self._content(
            shop_id,
            question=question,
            answer=answer,
            item_id=item_id,
            intent=None,
        )
        return self._warnings(shop, content, exclude_id=exclude_id)

    def create_faq(
        self,
        shop_id: str,
        *,
        question: str,
        answer: str,
        item_id: str | None = None,
        intent: Intent | None = None,
        merchant: str,
    ) -> FaqChange:
        """商家录入 FAQ：直接上线（商家本人审核），立刻可检索。"""
        actor = require_human(merchant, "商家")
        shop = require_shop(self._repo, shop_id)
        content = self._content(
            shop_id, question=question, answer=answer, item_id=item_id, intent=intent
        )
        warnings = self._warnings(shop, content)
        saved = self._repo.add_knowledge(
            self._new_item(shop_id, content, KnowledgeSource.MERCHANT, actor)
        )
        self._record(
            shop_id,
            actor,
            AuditAction.KNOWLEDGE_CREATED,
            {**self._faq_detail(saved), "warnings": [w.rule for w in warnings]},
        )
        self._faq_changed(shop_id)
        return FaqChange(item=saved, warnings=warnings)

    def edit_faq(
        self, shop_id: str, knowledge_id: int, changes: FaqChanges, *, merchant: str
    ) -> FaqChange:
        """修改已上线的 FAQ：旧条目下线、新条目上线（不在原 id 上改写，见模块文档）。

        不存在（或不属于该店）抛 KeyError；不是已上线状态、或并发时已被别人改掉抛
        ActionNotAllowed；没有任何改动抛 ValueError。
        """
        actor = require_human(merchant, "商家")
        shop = require_shop(self._repo, shop_id)
        old = self._load(shop_id, knowledge_id)
        if old.status != KnowledgeStatus.APPROVED:
            raise ActionNotAllowed(
                f"知识 {knowledge_id} 是 {old.status.value}，只有已上线的 FAQ 能修改"
                "（候选知识请用审核通过）"
            )
        content = self._apply(shop_id, old, changes)
        changed = content.changed_fields(old)
        if not changed:
            raise ValueError("内容没有变化")
        warnings = self._warnings(shop, content, exclude_id=old.id)
        retired = self._set_status(old, KnowledgeStatus.RETIRED, expected=KnowledgeStatus.APPROVED)
        new = self._add_or_restore(
            shop_id, content, old, restore_to=KnowledgeStatus.APPROVED, actor=actor
        )
        self._record(
            shop_id,
            actor,
            AuditAction.KNOWLEDGE_EDITED,
            {
                **self._faq_detail(new),
                "old_knowledge_id": old.id,
                "changed": changed,
                "warnings": [w.rule for w in warnings],
            },
        )
        self._faq_changed(shop_id)
        return FaqChange(item=new, retired=retired, warnings=warnings)

    def approve_edited(
        self, shop_id: str, knowledge_id: int, changes: FaqChanges, *, merchant: str
    ) -> FaqChange:
        """审核候选知识时顺手修改（§5.8）。

        没有改动：原地审核通过（CANDIDATE → APPROVED，和普通审核一样）；
        有改动：候选条目下线，改过的内容作为新条目上线（来源不变，审核人是商家）。
        审计都是 knowledge.approved（有改动时带 new_knowledge_id 和改了哪些字段）。
        """
        actor = require_human(merchant, "商家")
        shop = require_shop(self._repo, shop_id)
        candidate = self._load(shop_id, knowledge_id)
        if candidate.status != KnowledgeStatus.CANDIDATE:
            raise ActionNotAllowed(
                f"知识 {knowledge_id} 是 {candidate.status.value}，只有候选知识能审核通过"
            )
        content = self._apply(shop_id, candidate, changes)
        changed = content.changed_fields(candidate)
        warnings = self._warnings(shop, content, exclude_id=candidate.id)
        base = {
            "knowledge_id": candidate.id,
            "source": candidate.source.value,
            "status_before": candidate.status.value,
            "edited": bool(changed),
        }
        if not changed:
            approved = self._set_status(
                candidate,
                KnowledgeStatus.APPROVED,
                expected=KnowledgeStatus.CANDIDATE,
                approved_by=actor,
            )
            self._record(
                shop_id,
                actor,
                AuditAction.KNOWLEDGE_APPROVED,
                {
                    **base,
                    "intent": approved.intent.value if approved.intent else None,
                    "item_id": approved.item_id,
                    "warnings": [w.rule for w in warnings],
                },
            )
            self._faq_changed(shop_id)
            return FaqChange(item=approved, warnings=warnings)
        retired = self._set_status(
            candidate, KnowledgeStatus.RETIRED, expected=KnowledgeStatus.CANDIDATE
        )
        new = self._add_or_restore(
            shop_id, content, candidate, restore_to=KnowledgeStatus.CANDIDATE, actor=actor
        )
        self._record(
            shop_id,
            actor,
            AuditAction.KNOWLEDGE_APPROVED,
            {
                **base,
                "new_knowledge_id": new.id,
                "intent": new.intent.value if new.intent else None,
                "item_id": new.item_id,
                "changed": changed,
                "warnings": [w.rule for w in warnings],
            },
        )
        self._faq_changed(shop_id)
        return FaqChange(item=new, retired=retired, warnings=warnings)

    # ---------------------------------------------------------------- FAQ 内部

    def _load(self, shop_id: str, knowledge_id: int) -> KnowledgeItem:
        item = self._repo.get_knowledge(shop_id, knowledge_id)
        if item is None:
            raise KeyError(f"店铺 {shop_id} 没有知识 {knowledge_id}")
        if item.shop_id != shop_id:
            raise TenantIsolationError("知识仓储返回了别的店铺的 FAQ")
        return item

    def _content(
        self,
        shop_id: str,
        *,
        question: object,
        answer: object,
        item_id: object,
        intent: object,
    ) -> _FaqContent:
        q = clean_text(question, "问题", MAX_QUESTION_CHARS)
        a = clean_text(answer, "回答", MAX_ANSWER_CHARS)
        if item_id is not None:
            item_id = self._require_item(shop_id, item_id).item_id
        if intent is not None and not isinstance(intent, Intent):
            try:
                intent = Intent(intent)
            except ValueError as exc:
                raise ValueError("未知的问题类型") from exc
        return _FaqContent(question=q, answer=a, item_id=item_id, intent=intent)

    def _apply(self, shop_id: str, item: KnowledgeItem, changes: FaqChanges) -> _FaqContent:
        values = {f: getattr(item, f) for f in FAQ_FIELDS}
        for f in changes.provided:
            values[f] = getattr(changes, f)
        if "item_id" not in changes.provided:
            # 没改商品范围：原来的商品卡片后来被删了也照样保留（不因为改一个字而改变范围）
            content = self._content(shop_id, **{**values, "item_id": None})
            return _FaqContent(content.question, content.answer, item.item_id, content.intent)
        return self._content(shop_id, **values)

    def _require_item(self, shop_id: str, item_id: object) -> ItemCard:
        """FAQ 限定的商品必须是本店的商品卡片（别的店铺的商品和不存在一样，不透露是否存在）。"""
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("商品 ID 不能为空")
        card = self._repo.get_item_card(shop_id, item_id.strip())
        if card is None:
            raise ValueError("本店没有这个商品")
        if card.shop_id != shop_id:
            raise TenantIsolationError("知识仓储返回了别的店铺的商品")
        return card

    def _new_item(
        self, shop_id: str, content: _FaqContent, source: KnowledgeSource, actor: str
    ) -> KnowledgeItem:
        return KnowledgeItem(
            shop_id=shop_id,
            question=content.question,
            answer=content.answer,
            source=source,
            status=KnowledgeStatus.APPROVED,
            item_id=content.item_id,
            intent=content.intent,
            approved_by=actor,
            created_at=self._clock.now(),
        )

    def _set_status(
        self,
        item: KnowledgeItem,
        status: KnowledgeStatus,
        *,
        expected: KnowledgeStatus,
        approved_by: str | None = None,
    ) -> KnowledgeItem:
        """比较后更新：状态已经被别人（别的实例）改了就不写，提示刷新。"""
        assert item.id is not None
        try:
            return self._repo.set_knowledge_status(
                item.shop_id, item.id, status, approved_by=approved_by, expected_status=expected
            )
        except ConcurrencyConflict as exc:
            raise ActionNotAllowed(f"知识 {item.id} 的状态已经变化，请刷新后再操作") from exc

    def _add_or_restore(
        self,
        shop_id: str,
        content: _FaqContent,
        old: KnowledgeItem,
        *,
        restore_to: KnowledgeStatus,
        actor: str,
    ) -> KnowledgeItem:
        """旧条目已下线，写入新条目（审核人是这次操作的商家）；写不进去时把旧条目恢复原状态
        （尽力而为），再抛出原异常。"""
        try:
            return self._repo.add_knowledge(self._new_item(shop_id, content, old.source, actor))
        except Exception:
            assert old.id is not None
            try:
                self._repo.set_knowledge_status(
                    shop_id, old.id, restore_to, expected_status=KnowledgeStatus.RETIRED
                )
            except Exception as exc:  # noqa: BLE001 — 恢复失败只记日志，抛出原来的异常
                logger.error("FAQ 修改失败且旧条目 %s 恢复失败（%s）", old.id, type(exc).__name__)
            raise

    def _warnings(
        self, shop: Shop, content: _FaqContent, *, exclude_id: int | None = None
    ) -> list[FaqWarning]:
        item = self._repo.get_item_card(shop.id, content.item_id) if content.item_id else None
        policies = self._repo.get_policies(shop.id)
        probe = KnowledgeItem(
            id=0,
            shop_id=shop.id,
            question=content.question,
            answer=content.answer,
            source=KnowledgeSource.MERCHANT,
            status=KnowledgeStatus.APPROVED,
            item_id=content.item_id,
            intent=content.intent,
        )
        faq_candidate = _faq_candidate(probe, 1.0)
        warnings: list[FaqWarning] = []
        if self._checker is not None:
            check = FaqCheck(
                shop=shop,
                config=effective_shop_config(self._config, shop),
                question=content.question,
                answer=content.answer,
                item=item,
                evidence=faq_candidate.evidence,
                reference=_reference_evidence(policies, item),
                now=self._clock.now(),
            )
            warnings.extend(self._checker(check))
        warnings.extend(_policy_conflicts(faq_candidate, policies))
        warnings.extend(self._duplicates(shop.id, content, exclude_id))
        return warnings

    def _duplicates(
        self, shop_id: str, content: _FaqContent, exclude_id: int | None
    ) -> list[FaqWarning]:
        norm = normalize_question(content.question)
        if not norm:
            return []
        same = [
            k.id
            for k in self._repo.list_knowledge(shop_id, status=KnowledgeStatus.APPROVED)
            if k.id != exclude_id
            and k.item_id == content.item_id
            and normalize_question(k.question) == norm
        ]
        if not same:
            return []
        ids = "、".join(str(i) for i in same)
        return [
            FaqWarning(
                rule="duplicate_question",
                detail=f"已有同样问题的 FAQ（{ids}），回答时会同时检索到，建议下线其中一条",
                blocks_answer=False,
            )
        ]

    def _faq_changed(self, shop_id: str) -> None:
        if self._on_faq_changed is None:
            return
        try:
            self._on_faq_changed(shop_id)
        except Exception as exc:  # noqa: BLE001 — 已经写入；检索前的懒索引会补上
            logger.warning("FAQ 建索引失败（%s），下次检索时补上", type(exc).__name__)

    @staticmethod
    def _faq_detail(item: KnowledgeItem) -> dict[str, Any]:
        return {
            "knowledge_id": item.id,
            "source": item.source.value,
            "intent": item.intent.value if item.intent else None,
            "item_id": item.item_id,
        }

    # ================================================================ 店铺政策

    def list_policies(self, shop_id: str) -> list[PolicyRow]:
        """全部政策项：已知的按 PolicyKey 顺序（没填的 value 为 None），库里其余的键排在后面。"""
        entries = self._repo.get_policies(shop_id)
        for entry in entries.values():
            if entry.shop_id != shop_id:
                raise TenantIsolationError("知识仓储返回了别的店铺的政策")
        rows = [_policy_row(key.value, entries.get(key.value)) for key in PolicyKey]
        known = {k.value for k in PolicyKey}
        rows.extend(_policy_row(k, entries[k]) for k in sorted(entries) if k not in known)
        return rows

    def set_policy(self, shop_id: str, key: str, value: str, *, merchant: str) -> PolicyRow:
        """改一项店铺政策（只能是已知的政策键）。改动即时生效（§5.2.1）；内容没变不写。"""
        actor = require_human(merchant, "商家")
        try:
            policy_key = PolicyKey(key)
        except ValueError as exc:
            raise ValueError("未知的店铺政策项") from exc
        text = clean_text(value, "政策内容", MAX_POLICY_CHARS)
        require_shop(self._repo, shop_id)
        existing = self._repo.get_policies(shop_id).get(policy_key.value)
        if existing is not None and existing.shop_id == shop_id and existing.value == text:
            return _policy_row(policy_key.value, existing)
        entry = PolicyEntry(
            shop_id=shop_id, key=policy_key.value, value=text, updated_at=self._clock.now()
        )
        self._repo.upsert_policy(entry)
        self._record(
            shop_id,
            actor,
            AuditAction.POLICY_UPDATED,
            {"key": policy_key.value, "created": existing is None},
        )
        return _policy_row(policy_key.value, entry)

    def clear_policy(self, shop_id: str, key: str, *, merchant: str) -> PolicyRow:
        """清空一项店铺政策（不再适用，如赠品活动结束）：之后回答不再引用它，缺了会提醒补（§5.2.3）。
        本来就没填不写、不记审计。"""
        actor = require_human(merchant, "商家")
        try:
            policy_key = PolicyKey(key)
        except ValueError as exc:
            raise ValueError("未知的店铺政策项") from exc
        require_shop(self._repo, shop_id)
        if self._repo.delete_policy(shop_id, policy_key.value):
            self._record(
                shop_id,
                actor,
                AuditAction.POLICY_UPDATED,
                {"key": policy_key.value, "cleared": True},
            )
        return _policy_row(policy_key.value, None)

    # ================================================================ 商品卡片

    def list_item_cards(self, shop_id: str) -> list[ItemCard]:
        cards = self._repo.list_item_cards(shop_id)
        for card in cards:
            if card.shop_id != shop_id:
                raise TenantIsolationError("知识仓储返回了别的店铺的商品")
        return cards

    def get_item_card(self, shop_id: str, item_id: str) -> ItemCard:
        card = self._repo.get_item_card(shop_id, item_id)
        if card is None:
            raise KeyError(f"店铺 {shop_id} 没有商品 {item_id}")
        if card.shop_id != shop_id:
            raise TenantIsolationError("知识仓储返回了别的店铺的商品")
        return card

    def update_item_card(
        self, shop_id: str, item_id: str, changes: Mapping[str, Any], *, merchant: str
    ) -> tuple[ItemCard, list[str]]:
        """改商品卡片上商家填写的字段（§5.2.2）。返回 (卡片, 改了的字段)；没有改动不写。

        size_advice 为 None 或空白表示清空；列表字段为 None 表示清空。平台字段不能改（ValueError）。
        """
        actor = require_human(merchant, "商家")
        if not changes:
            raise ValueError("没有要修改的字段")
        platform_owned = sorted(set(changes) - set(MERCHANT_ITEM_FIELDS))
        if platform_owned:
            raise ValueError(
                f"商品卡片上只能改商家填写的字段（{'、'.join(MERCHANT_ITEM_FIELDS)}），"
                "标题、属性、SKU、尺码表、详情文字由商品同步更新"
            )
        normalized = _normalize_item_fields(changes)
        with self._lock:
            require_shop(self._repo, shop_id)
            card = self.get_item_card(shop_id, item_id)
            changed = [f for f in MERCHANT_ITEM_FIELDS if f in normalized]
            changed = [f for f in changed if getattr(card, f) != normalized[f]]
            if not changed:
                return card, []
            # 只写改了的字段（仓储里原子地读改写）：别的实例上同时跑的商品同步改的是平台字段，
            # 两边的修改都留下
            saved = self._repo.update_item_card(
                shop_id,
                item_id,
                {**{f: normalized[f] for f in changed}, "updated_at": self._clock.now()},
            )
            if saved is None:
                raise KeyError(f"店铺 {shop_id} 没有商品 {item_id}")
            updated = saved
        self._record(
            shop_id,
            actor,
            AuditAction.ITEM_CARD_UPDATED,
            {"item_id": item_id, "fields": changed},
        )
        return updated, changed

    # ================================================================ 店铺配置（§9.3）

    def get_config(self, shop_id: str) -> ShopConfigSnapshot:
        return self._snapshot(require_shop(self._repo, shop_id))

    def update_config(
        self, shop_id: str, changes: Mapping[str, Any], *, merchant: str
    ) -> ShopConfigSnapshot:
        """改店铺覆盖项（§9.3）。值为 None 删掉覆盖项（回到类目 / 平台默认）。

        未知的配置项、取值不合法抛 ValueError；影子模式、全自动开关等不开放的项抛 ActionNotAllowed。
        没有实际改动时不写库、不记审计。
        """
        actor = require_human(merchant, "商家")
        if not changes:
            raise ValueError("没有要修改的配置项")
        unknown = sorted(set(changes) - SHOP_CONFIG_FIELDS)
        if unknown:
            raise ValueError(f"未知的店铺配置项：{unknown}")
        locked = sorted(set(changes) - set(MERCHANT_CONFIG_FIELDS))
        if locked:
            raise ActionNotAllowed(MSG_LOCKED_CONFIG.format(keys="、".join(locked)))
        normalized = {k: _normalize_config_value(k, v) for k, v in changes.items()}
        with self._lock:
            shop = require_shop(self._repo, shop_id)
            overrides = dict(shop.config_overrides)
            changed: list[str] = []
            removed: list[str] = []
            for key in MERCHANT_CONFIG_FIELDS:
                if key not in normalized:
                    continue
                value = normalized[key]
                if value is None:
                    if key in overrides:
                        del overrides[key]
                        removed.append(key)
                elif key not in overrides or _canonical(key, overrides[key]) != value:
                    overrides[key] = value
                    changed.append(key)
            if not changed and not removed:
                return self._snapshot(shop)
            preview = Shop.model_validate(
                {**shop.model_dump(mode="json"), "config_overrides": overrides}
            )
            effective_shop_config(self._config, preview)  # 合并平台 / 类目默认后也必须合法
            # 只写改了的键（仓储里原子地读改写）：复核人员同时切换影子模式、填问卷改别的键，
            # 两边的修改都留下
            saved = self._repo.update_shop_config(
                shop_id,
                {**{k: overrides[k] for k in changed}, **dict.fromkeys(removed)},
            )
            if saved is None:
                raise ShopInactive(shop_id)
            updated = saved
        self._record(
            shop_id,
            actor,
            AuditAction.CONFIG_UPDATED,
            {"changed": sorted(changed), "removed": sorted(removed)},
        )
        return self._snapshot(updated, changed=changed, removed=removed)

    def _snapshot(
        self,
        shop: Shop,
        *,
        changed: Sequence[str] = (),
        removed: Sequence[str] = (),
    ) -> ShopConfigSnapshot:
        return ShopConfigSnapshot(
            effective=effective_shop_config(self._config, shop),
            defaults=resolve_shop_config(
                self._config, platform=shop.platform, category=shop.category
            ),
            overrides=dict(shop.config_overrides),
            editable=list(MERCHANT_CONFIG_FIELDS),
            locked=list(LOCKED_CONFIG_FIELDS),
            changed=sorted(changed),
            removed=sorted(removed),
        )

    # ================================================================ 审计

    def _record(self, shop_id: str, actor: str, action: AuditAction, detail: dict) -> None:
        self._audit.record(
            AuditEvent(
                shop_id=shop_id,
                actor=actor,
                action=action.value,
                detail=detail,
                created_at=self._clock.now(),
            )
        )


# ================================================================ 模块内工具


def _policy_row(key: str, entry: PolicyEntry | None) -> PolicyRow:
    try:
        policy_key: PolicyKey | None = PolicyKey(key)
    except ValueError:
        policy_key = None
    question = question_for_policy(policy_key) if policy_key is not None else None
    return PolicyRow(
        key=key,
        label=POLICY_LABELS[policy_key] if policy_key is not None else key,
        value=entry.value if entry is not None else None,
        updated_at=entry.updated_at if entry is not None else None,
        questionnaire_id=question.id if question is not None else None,
        known=policy_key is not None,
    )


def _reference_evidence(
    policies: Mapping[str, PolicyEntry], item: ItemCard | None
) -> list[Evidence]:
    """这条 FAQ 之外的依据：店铺政策 + 当前商品卡片上的文字。"""
    out = [_policy_evidence(e) for _, e in sorted(policies.items()) if e.value.strip()]
    if item is not None:
        texts = [
            item.title,
            *(f"{k}：{v}" for k, v in item.attributes.items()),
            item.size_advice or "",
            *item.care,
            *item.usage,
            item.detail_text or "",
        ]
        out.append(
            Evidence(
                id=ev.item_detail(item.item_id),
                source=EvidenceSource.ITEM,
                label=ev.item_label("商品卡片"),
                content="\n".join(t for t in texts if t),
            )
        )
    return out


def _policy_conflicts(faq: Candidate, policies: Mapping[str, PolicyEntry]) -> list[FaqWarning]:
    """回答涉及的店铺政策主题上，和政策内容冲突：回答时这条 FAQ 被政策覆盖（§5.2.1），用不上。"""
    topics = ordered_topics(set(faq.topics) & _POLICY_TOPICS)
    referees = [
        Candidate(_policy_evidence(policies[t]))
        for t in topics
        if t in policies and policies[t].value.strip()
    ]
    if not referees:
        return []
    _, lost = resolve_conflicts([faq, *referees])
    if faq.evidence.id not in lost:
        return []
    labels = "、".join(POLICY_LABELS[PolicyKey(r.evidence.topic)] for r in referees)
    return [
        FaqWarning(
            rule="policy_conflict",
            detail=f"和店铺政策（{labels}）不一致：回答时以店铺政策为准，这条 FAQ 不会被使用",
            blocks_answer=True,
        )
    ]


def _normalize_item_fields(changes: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "size_advice" in changes:
        value = changes["size_advice"]
        if value is None or (isinstance(value, str) and not strip_control_chars(value).strip()):
            out["size_advice"] = None
        else:
            out["size_advice"] = clean_text(value, "尺码建议", MAX_SIZE_ADVICE_CHARS)
    for name, what, max_chars in (
        ("care", "洗涤保养", MAX_ITEM_LINE_CHARS),
        ("usage", "使用说明", MAX_ITEM_LINE_CHARS),
        ("forbidden_claims", "不能说的卖点", MAX_CLAIM_CHARS),
    ):
        if name in changes:
            out[name] = clean_lines(changes[name], what, max_chars=max_chars)
    return out


def _canonical(key: str, value: Any) -> Any:
    """配置项取值的规范形式（JSON 原生值，和存进 config_overrides 的一致）。"""
    return ShopConfig.model_validate({key: value}).model_dump(mode="json")[key]


def _normalize_config_value(key: str, value: Any) -> Any:
    """商家提交的一项配置 → 存进覆盖项的值；None 表示删掉覆盖项。取值不合法抛 ValueError。"""
    if value is None:
        return None
    if isinstance(value, bool):  # 商家能改的项没有开关类的；不让 true 被当成 1
        raise ValueError(f"配置项 {key} 的取值不合法")
    if key == "display_name":
        if not isinstance(value, str):
            raise ValueError("店铺名必须是文字")
        value = strip_control_chars(value).strip()
        if len(value) > MAX_DISPLAY_NAME_CHARS:
            raise ValueError(f"店铺名最多 {MAX_DISPLAY_NAME_CHARS} 个字")
    elif key == "banned_words":
        if isinstance(value, str):  # 一段文字：按问卷第 14 题的规则切开
            value = parse_banned_words(strip_control_chars(value))
        value = clean_lines(
            value, "禁用词", max_chars=MAX_BANNED_WORD_CHARS, max_items=MAX_BANNED_WORDS
        )
    elif key == "manager_ids":
        value = clean_lines(
            value, "店长账号", max_chars=MAX_ACCOUNT_CHARS, max_items=MAX_MANAGER_IDS
        )
        for account in value:
            require_human(account, "店长账号")
    elif key == "service_hours":
        value = _service_hours(value)
    elif key == "disabled_intents":
        if not isinstance(value, list | tuple):
            raise ValueError("关闭的问题类型必须是列表")
        return list(dict.fromkeys(_canonical(key, list(value))))  # 先校验取值，再去重保序
    elif key == "aftersale_amount_limit":
        try:
            amount = Decimal(str(value).strip())
        except InvalidOperation as exc:
            raise ValueError("售后金额上限必须是金额") from exc
        if not amount.is_finite() or amount < 0:
            raise ValueError("售后金额上限必须是不小于 0 的金额")
        value = amount
    elif isinstance(value, str):
        value = strip_control_chars(value).strip()
    return _canonical(key, value)


def _service_hours(value: Any) -> list[str]:
    """客服工作时间（北京时间，附录 C 第 13 题）→ 统一写法的时段列表。

    设置页按行提交（每行一个时段，"9:00-23:00""早9点到晚11点"都行，见
    questionnaire.parse_service_hours_text）；也接受一段文字（同问卷第 13 题的写法）。
    空列表 / 空文字表示没填（全自动默认只在工作时间以外生效，没填就不自动）。
    看不懂、时段太多、时间不对抛 ValueError，写明哪一行、怎么改。"""
    if isinstance(value, str):
        text = strip_control_chars(value).strip()
        if len(text) > MAX_SERVICE_HOURS_TEXT:
            raise ValueError(f"客服工作时间最多 {MAX_SERVICE_HOURS_TEXT} 个字")
        lines = [text] if text else []
    else:
        lines = clean_lines(
            value,
            "客服工作时间",
            max_chars=MAX_SERVICE_HOURS_CHARS,
            max_items=MAX_SERVICE_WINDOWS,
        )
    windows: list[str] = []
    for line in lines:
        try:
            windows.extend(parse_service_hours_text(line))
        except ValueError as exc:
            raise ValueError(f"客服工作时间「{line}」：{exc}") from None
    try:
        return normalize_service_hours(windows)
    except ValueError as exc:
        raise ValueError(f"客服工作时间：{exc}") from None


__all__ = [
    "FAQ_FIELDS",
    "LOCKED_CONFIG_FIELDS",
    "MAX_ANSWER_CHARS",
    "MAX_CLAIM_CHARS",
    "MAX_ITEM_LINE_CHARS",
    "MAX_POLICY_CHARS",
    "MAX_QUESTION_CHARS",
    "MAX_SERVICE_HOURS_CHARS",
    "MAX_SIZE_ADVICE_CHARS",
    "MERCHANT_CONFIG_FIELDS",
    "MERCHANT_ITEM_FIELDS",
    "AnswerChecker",
    "FaqChange",
    "FaqChanges",
    "FaqCheck",
    "FaqWarning",
    "KnowledgeAdmin",
    "PolicyRow",
    "ShopConfigSnapshot",
    "clean_lines",
    "clean_text",
    "require_human",
    "require_shop",
]
