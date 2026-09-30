"""商家控制台（方案 §9.1 "知识编辑、配置、报表都跟着单元走"、§5.2 知识中心、§5.2.3 商品同步、
§9.3 店铺配置）。

所有写操作都在 processing(shop_id) 里做（和取消订购互斥，§8.4：删完的店铺不会被写回来——
进了闸门还要确认店铺还在，见 knowledge.admin.require_shop），
操作人取自调用方身份，写审计（core.audit.AuditAction 的 knowledge.* / shop.config_updated）。

组装：
- 领域逻辑在 rodit.knowledge.admin（FAQ、政策、商品卡片、店铺配置）和
  rodit.knowledge.sync（商品同步）；
- 保存 FAQ 前的回答检查接质检硬规则（rodit.qa.check_hard_rules，§5.5 第一道），和回答时同样的
  店铺生效配置、当前商品、证据（这条 FAQ 自己）：命中拦截类规则说明这条 FAQ 回答时会被拦截；
  另外单独看时效承诺在店铺政策、商品卡片里有没有依据（FAQ 直接发给买家时，自己就是依据，
  质检拦不住没法兑现的承诺，§7.2）。这些都只是提醒，不拦截保存；
- 已上线 FAQ 变了之后立刻给新条目算向量（FaqSearcher.index，和反馈闭环审核通过一样）；
- 商品卡片、店铺配置的"读—改—写"和商品同步共用一把锁（同一进程内不互相覆盖）。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any

from rodit.app.unit import Unit
from rodit.core.enums import DraftKind, Intent, KnowledgeStatus, Role
from rodit.core.models import (
    ConversationContext,
    Draft,
    EvidenceBundle,
    ItemCard,
    KnowledgeItem,
    Turn,
)
from rodit.core.scopes import MVP_SCOPES
from rodit.knowledge.admin import (
    FaqChange,
    FaqChanges,
    FaqCheck,
    FaqWarning,
    KnowledgeAdmin,
    PolicyRow,
    ShopConfigSnapshot,
    require_human,
    require_shop,
)
from rodit.knowledge.sync import ItemSync, SyncReport
from rodit.qa import check_hard_rules

ShopGate = Callable[[str], AbstractContextManager[None]]
"""AssistService.processing：按店铺的读方闸门（和取消订购互斥）。"""

_CHECK_CONVERSATION = "console:faq_check"


class ConsoleService:
    """商家控制台的应用服务。读操作直接读；写操作一律在 processing(shop_id) 里做。

    出错时抛业务异常，由接口层映射：KeyError（不存在）、ValueError（取值不合法）、
    ActionNotAllowed（状态不对 / 不开放的配置项）、ShopInactive（店铺已删除或停用）、
    ItemSyncUnavailable（没接商品接口）、ToolUnavailable（商品接口故障）、
    AuthExpired（商品同步时店铺授权无效）。
    """

    def __init__(self, unit: Unit, processing: ShopGate) -> None:
        self.unit = unit
        self._processing = processing
        self._lock = threading.Lock()
        self.admin = KnowledgeAdmin(
            unit.repos.knowledge,
            unit.repos.audit,
            unit.clock,
            config=unit.config,
            on_faq_changed=unit.searcher.index,
            answer_checker=self._answer_warnings,
            lock=self._lock,
        )

    # ================================================================ FAQ

    def list_faqs(
        self,
        shop_id: str,
        *,
        status: KnowledgeStatus | None = None,
        item_id: str | None = None,
    ) -> list[KnowledgeItem]:
        return self.admin.list_faqs(shop_id, status=status, item_id=item_id)

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
        with self._processing(shop_id):
            return self.admin.create_faq(
                shop_id,
                question=question,
                answer=answer,
                item_id=item_id,
                intent=intent,
                merchant=merchant,
            )

    def edit_faq(
        self, shop_id: str, knowledge_id: int, changes: FaqChanges, *, merchant: str
    ) -> FaqChange:
        with self._processing(shop_id):
            return self.admin.edit_faq(shop_id, knowledge_id, changes, merchant=merchant)

    def approve_edited(
        self, shop_id: str, knowledge_id: int, changes: FaqChanges, *, merchant: str
    ) -> FaqChange:
        with self._processing(shop_id):
            return self.admin.approve_edited(shop_id, knowledge_id, changes, merchant=merchant)

    # ================================================================ 店铺政策

    def list_policies(self, shop_id: str) -> list[PolicyRow]:
        return self.admin.list_policies(shop_id)

    def set_policy(self, shop_id: str, key: str, value: str, *, merchant: str) -> PolicyRow:
        with self._processing(shop_id):
            return self.admin.set_policy(shop_id, key, value, merchant=merchant)

    def clear_policy(self, shop_id: str, key: str, *, merchant: str) -> PolicyRow:
        with self._processing(shop_id):
            return self.admin.clear_policy(shop_id, key, merchant=merchant)

    # ================================================================ 商品卡片

    def list_items(self, shop_id: str) -> list[ItemCard]:
        return self.admin.list_item_cards(shop_id)

    def get_item(self, shop_id: str, item_id: str) -> ItemCard:
        return self.admin.get_item_card(shop_id, item_id)

    def update_item(
        self, shop_id: str, item_id: str, changes: Mapping[str, Any], *, merchant: str
    ) -> tuple[ItemCard, list[str]]:
        with self._processing(shop_id):
            return self.admin.update_item_card(shop_id, item_id, changes, merchant=merchant)

    def sync_items(self, shop_id: str, *, item_id: str | None = None, merchant: str) -> SyncReport:
        """平台商品同步（§5.2.3）：item_id 为空时同步整个店铺。没有接入商品接口或 OCR 时抛
        ItemSyncUnavailable；商品接口故障抛 ToolUnavailable。

        调平台商品接口前先查店铺授权（§8.4 授权过期时停止处理）：没有授权、过期、缺读商品权限、
        需要重新授权都抛 AuthExpired，商品接口一次都不调。"""
        actor = require_human(merchant, "商家")
        sync = ItemSync(
            self.unit.repos.knowledge,
            self.unit.catalog,
            self.unit.ocr,
            self.unit.repos.audit,
            self.unit.clock,
            platform=self.unit.platform,
            lock=self._lock,
        )
        with self._processing(shop_id):
            require_shop(self.unit.repos.knowledge, shop_id)
            self.unit.auth.require(shop_id, MVP_SCOPES)
            if item_id is None:
                return sync.sync_shop(shop_id, operator=actor)
            return sync.sync_item(shop_id, item_id, operator=actor)

    # ================================================================ 店铺配置（§9.3）

    def get_config(self, shop_id: str) -> ShopConfigSnapshot:
        return self.admin.get_config(shop_id)

    def update_config(
        self, shop_id: str, changes: Mapping[str, Any], *, merchant: str
    ) -> ShopConfigSnapshot:
        with self._processing(shop_id):
            return self.admin.update_config(shop_id, changes, merchant=merchant)

    # ================================================================ 保存 FAQ 前的回答检查

    def _answer_warnings(self, check: FaqCheck) -> list[FaqWarning]:
        """质检硬规则（§5.5 第一道）：模拟回答时的情形（这条 FAQ 是唯一的依据、FAQ 直答）。"""
        ctx = ConversationContext(
            platform=check.shop.platform,
            shop=check.shop,
            config=check.config,
            conversation_id=_CHECK_CONVERSATION,
            buyer_ref=_CHECK_CONVERSATION,
            current=Turn(role=Role.BUYER, text=check.question, at=check.now),
            item=check.item,
            now=check.now,
        )
        qa_config = self.unit.config.qa_gate
        draft = Draft(
            text=check.answer,
            evidence_ids=[check.evidence.id],
            kind=DraftKind.FAQ,
            faq_id=0,
        )
        _, hits = check_hard_rules(
            check.answer,
            ctx=ctx,
            draft=draft,
            evidence=EvidenceBundle(evidences=[check.evidence]),
            config=qa_config,
        )
        warnings: list[FaqWarning] = []
        for hit in hits:
            if hit.rule == "unknown_evidence":
                continue
            if hit.rule == "length":
                limit = check.config.max_answer_chars
                detail = (
                    f"回答 {len(check.answer)} 字，超过店铺设置的 {limit} 字上限，"
                    "发给买家时会被截断"
                )
                warnings.append(FaqWarning(rule="length", detail=detail, blocks_answer=False))
            elif hit.rule == "format":
                warnings.append(
                    FaqWarning(
                        rule="format",
                        detail="包含 Markdown 格式，发给买家时会被去掉",
                        blocks_answer=False,
                    )
                )
            else:
                warnings.append(
                    FaqWarning(
                        rule=hit.rule,
                        detail=f"{hit.detail}（回答时会被质检拦截）",
                        blocks_answer=True,
                    )
                )
        # 时效承诺：FAQ 自己不算依据，店铺政策、商品卡片里都没有就提醒（能不能兑现要商家确认）
        _, promise_hits = check_hard_rules(
            check.answer,
            ctx=ctx,
            draft=Draft(text=check.answer, kind=DraftKind.FAQ),
            evidence=EvidenceBundle(evidences=list(check.reference)),
            config=qa_config,
        )
        if any(w.rule == "time_promise" for w in warnings):
            return warnings  # 回答时就会被拦截，上面已经提醒过
        for hit in promise_hits:
            if hit.rule != "time_promise":
                continue
            detail = (
                f"包含时效承诺「{hit.span}」，店铺政策和商品卡片里都没有依据；"
                "FAQ 会原样发给买家，请确认能兑现（§7.2 不承诺没有依据的时效）"
            )
            warnings.append(FaqWarning(rule="time_promise", detail=detail, blocks_answer=False))
        return warnings


__all__ = ["ConsoleService", "ShopGate"]
