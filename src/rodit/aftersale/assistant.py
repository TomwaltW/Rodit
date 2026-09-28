"""售后审核助手（方案 §7.3.2、§11.3；§4.1 原则 3 人在回路、原则 6 一切留痕）。

对每张待审核的售后单：拉取售后单、订单、物流、买家售后记录 → 规则 R1–R8 给出建议和理由 →
状态机 → 客服确认 → 重读平台状态 → 调平台写接口（幂等键）→ 审计。问答引擎永远不碰这里。

- review：NEW → SUGGEST →（CONFIRM | VERIFY | EVIDENCE | PLATFORM），
  新建记录（expected_version=None）；已有记录时等同 refresh。
- 商家在平台后台同意了退货（平台状态 WAIT_BUYER_RETURN）：任何环节读到都进 WAIT_RETURN，
  不进 PLATFORM 终态——买家寄回后还要建议同意退款（R3 第二步）。
- refresh：重读平台状态推进状态机——平台已处理 → PLATFORM；签收核查满 verify_hours → CONFIRM
  （建议同意）；凭证已补充 → CONFIRM（重新跑规则）；退货已寄回（WAIT_SELLER_CONFIRM）→ CONFIRM
  （建议同意退款）；待确认时平台状态或状态版本变了 → 重新跑规则。
- negotiate / negotiation_done：CONFIRM ⇄ NEGOTIATE，记住协商过（淘宝拒绝前置条件）。
- confirm：只接受 HumanConfirmation；R6 要店长确认；平台规则（拼多多无拒绝接口、淘宝拒绝要先协商、
  抖店拒绝要原因码、未注入写接口）；调用前重读售后单：已不可操作 → PLATFORM、不调平台；
  状态版本和客服确认时看到的不一致，或者建议是按旧版本给的（platform_status_version）
  → ActionNotAllowed（请刷新）；幂等键 = 售后单 + 操作 + 平台状态版本。

留痕：aftersale.suggest（规则建议）、aftersale.transition（每次迁移）、aftersale.confirm（谁确认了
什么）、aftersale.action（平台返回结果）。审计 detail 不放买家原文（售后说明不进审计）。
迁移审计在记录保存成功后写；确认和平台结果在发生时就写（平台已经操作过，保存失败也要留痕）。

网关必须读平台最新数据（§7.3.2 调用前重新读取）：传入带缓存的网关（有 uncached() 方法，
如 tools.gateway.CachingToolGateway）时自动改用其内层网关。
ConcurrencyConflict（乐观锁）原样抛出，由调用方刷新后重试。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from rodit.aftersale.advisor import AFTERSALE_STATUS_LABELS
from rodit.aftersale.rules import REASON_WAIT_SELLER_CONFIRM, ReviewSuggestion, evaluate
from rodit.aftersale.state_machine import (
    TERMINAL_STATES,
    transition,
    was_negotiated,
    with_result,
)
from rodit.core.audit import AuditAction
from rodit.core.clock import Clock
from rodit.core.config import AfterSaleConfig
from rodit.core.enums import (
    ACTIONABLE_STATUSES,
    AfterSaleAction,
    AfterSaleStatus,
    AfterSaleType,
    Platform,
    ReviewAdvice,
    ReviewState,
    is_actionable,
)
from rodit.core.errors import ActionNotAllowed, TenantIsolationError, ToolUnavailable
from rodit.core.evidence import PLATFORM_NAMES
from rodit.core.models import (
    ActionResult,
    AfterSaleInfo,
    AfterSaleReview,
    AuditEvent,
    HumanConfirmation,
    LogisticsTrace,
    OrderInfo,
    ShopConfig,
    aftersale_idempotency_key,
)
from rodit.core.pii import mask_pii
from rodit.core.ports import AfterSaleActionPort, AfterSaleReviewRepository, AuditSink, ToolGateway

SYSTEM_ACTOR = "system"

ACTION_LABELS: dict[AfterSaleAction, str] = {
    AfterSaleAction.AGREE_REFUND: "同意退款",
    AfterSaleAction.AGREE_RETURN: "同意退货",
    AfterSaleAction.REFUSE: "拒绝售后",
}

_NEXT_STATE: dict[AfterSaleAction, ReviewState] = {
    AfterSaleAction.AGREE_RETURN: ReviewState.WAIT_RETURN,
    AfterSaleAction.AGREE_REFUND: ReviewState.AGREED,
    AfterSaleAction.REFUSE: ReviewState.REFUSED,
}

MSG_NO_PORT = "MVP 未申请写售后权限，请在平台后台操作"
MSG_STALE = "售后单状态已变化，请刷新后重新确认"
MSG_MANAGER = "金额超过店铺上限，必须店长确认（R6）"
MSG_PDD_REFUSE = "拼多多未开放拒绝售后接口，请在商家后台手动操作"
MSG_TAOBAO_REFUSE = "淘宝拒绝售后前需要先与买家协商（2025-10-15 起部分场景协商前置），请先选择协商"
MSG_DOUYIN_REFUSE = "抖店拒绝售后必须选择拒绝原因码"

_NON_HUMAN = {"system", "bot", "auto"}


def _synced(rec: AfterSaleReview, info: AfterSaleInfo) -> AfterSaleReview:
    """记下刚读到的平台状态和状态版本。"""
    return rec.model_copy(
        update={"platform_status": info.status, "platform_status_version": info.status_version}
    )


def _platform_changed(rec: AfterSaleReview, info: AfterSaleInfo) -> bool:
    """平台上的售后单和审核记录上次同步时相比变了（状态或状态版本）。"""
    return info.status != rec.platform_status or info.status_version != (
        rec.platform_status_version
    )


def _require_human(operator: str) -> None:
    if not operator or operator.strip().lower() in _NON_HUMAN:
        raise ActionNotAllowed("售后审核操作必须由人执行（§4.1 原则 3）")


class ReviewAssistant:
    """售后审核助手。只有它能用 AfterSaleActionPort，且每次写操作都要客服确认。"""

    def __init__(
        self,
        gateway: ToolGateway,
        repo: AfterSaleReviewRepository,
        audit: AuditSink,
        clock: Clock,
        config: AfterSaleConfig,
        *,
        shop_config: Callable[[str], ShopConfig],
        action_port: AfterSaleActionPort | None,
    ) -> None:
        uncached = getattr(gateway, "uncached", None)
        self._gateway: ToolGateway = uncached() if callable(uncached) else gateway
        self._repo = repo
        self._audit = audit
        self._clock = clock
        self._config = config
        self._shop_config = shop_config
        self._port = action_port
        self.platform: Platform = (
            action_port.platform if action_port is not None else self._gateway.platform
        )

    # ================================================================ 公开接口

    def review(self, shop_id: str, aftersale_id: str) -> AfterSaleReview:
        """拉取售后单并跑规则，新建审核记录。已有记录时等同 refresh。

        售后单不存在或不属于该店铺抛 KeyError（和知识候选等资源一致，接口层映射为 404）。
        """
        if self._repo.get(shop_id, aftersale_id) is not None:
            return self.refresh(shop_id, aftersale_id)
        now = self._clock.now()
        info = self._read(shop_id, aftersale_id)
        if info is None:
            raise KeyError(f"售后单 {aftersale_id} 不存在或不属于店铺 {shop_id}")
        pending: list[AuditEvent] = []
        rec = AfterSaleReview(
            shop_id=shop_id,
            aftersale_id=aftersale_id,
            platform_status=info.status,
            platform_status_version=info.status_version,
            state=ReviewState.NEW,
        )
        rec = self._move(rec, ReviewState.SUGGEST, now, pending)
        if info.status == AfterSaleStatus.WAIT_BUYER_RETURN:
            # 商家已在平台后台同意退货：等买家寄回，之后还要建议同意退款（R3 第二步）
            rec = self._to_wait_return(rec, info, now, pending)
        else:
            suggestion = self._evaluate(shop_id, info, now)
            rec = self._apply(rec, suggestion, now, pending)
            rec = self._move(rec, suggestion.next_state, now, pending)
        saved = self._repo.save(rec, expected_version=None)
        self._flush(pending)
        return saved

    def refresh(self, shop_id: str, aftersale_id: str) -> AfterSaleReview:
        """重读平台状态，推进状态机（没有记录时等同 review）。终态不再读平台。"""
        current = self._repo.get(shop_id, aftersale_id)
        if current is None:
            return self.review(shop_id, aftersale_id)
        if current.state in TERMINAL_STATES:
            return current
        now = self._clock.now()
        info = self._read(shop_id, aftersale_id)
        pending: list[AuditEvent] = []
        rec = current
        if info is None:
            rec = self._to_platform(rec, None, now, pending)
        else:
            rec = _synced(rec, info)
            waiting_return = (
                rec.state == ReviewState.WAIT_RETURN
                and info.status == AfterSaleStatus.WAIT_BUYER_RETURN
            )
            if not is_actionable(info.status) and not waiting_return:
                rec = self._to_platform(rec, info, now, pending)
            elif rec.state == ReviewState.VERIFY:
                rec = self._verify_timeout(rec, now, pending)
            elif rec.state == ReviewState.EVIDENCE and info.evidence_count > 0:
                rec = self._apply(rec, self._evaluate(shop_id, info, now), now, pending)
                rec = self._move(rec, ReviewState.CONFIRM, now, pending, note="凭证已补充")
            elif rec.state == ReviewState.WAIT_RETURN and not waiting_return:
                rec = self._apply(rec, self._evaluate(shop_id, info, now), now, pending)
                rec = self._move(rec, ReviewState.CONFIRM, now, pending, note="买家已寄回")
            elif rec.state == ReviewState.CONFIRM and _platform_changed(current, info):
                # 待确认期间平台上的售后单变了（如商家在后台同意退货、买家改了金额、补了凭证）：
                # 重新给建议
                rec = self._reevaluate(rec, info, now, pending)
        if rec == current:
            return current
        saved = self._repo.save(rec, expected_version=current.version)
        self._flush(pending)
        return saved

    def negotiate(
        self, shop_id: str, aftersale_id: str, *, operator: str, note: str | None = None
    ) -> AfterSaleReview:
        """客服选择协商：CONFIRM → NEGOTIATE（记住协商过）。"""
        _require_human(operator)
        current = self._get(shop_id, aftersale_id)
        if current.state != ReviewState.CONFIRM:
            raise ActionNotAllowed(f"售后审核当前是 {current.state}，只有待客服确认时才能发起协商")
        pending: list[AuditEvent] = []
        rec = self._move(
            current, ReviewState.NEGOTIATE, self._clock.now(), pending, by=operator, note=note
        )
        saved = self._repo.save(rec, expected_version=current.version)
        self._flush(pending)
        return saved

    def negotiation_done(
        self, shop_id: str, aftersale_id: str, *, operator: str, outcome: str | None = None
    ) -> AfterSaleReview:
        """协商有结果：NEGOTIATE → CONFIRM；协商期间平台已处理（如买家撤销）则 → PLATFORM。"""
        _require_human(operator)
        current = self._get(shop_id, aftersale_id)
        if current.state != ReviewState.NEGOTIATE:
            raise ActionNotAllowed(f"售后审核当前是 {current.state}，不在协商中")
        now = self._clock.now()
        info = self._read(current.shop_id, aftersale_id)
        pending: list[AuditEvent] = []
        if info is None or not is_actionable(info.status):
            rec = self._to_platform(current, info, now, pending)
        else:
            rec = current.model_copy(update={"platform_status": info.status})
            rec = self._move(rec, ReviewState.CONFIRM, now, pending, by=operator, note=outcome)
        saved = self._repo.save(rec, expected_version=current.version)
        self._flush(pending)
        return saved

    def confirm(
        self,
        shop_id: str,
        aftersale_id: str,
        action: AfterSaleAction,
        *,
        confirmation: HumanConfirmation,
        reason_code: str | None = None,
        reason_text: str | None = None,
    ) -> AfterSaleReview:
        """客服确认后执行平台写操作（§7.3.2 防止误操作）。"""
        current = self._get(shop_id, aftersale_id)
        if current.state != ReviewState.CONFIRM:
            raise ActionNotAllowed(f"售后审核当前是 {current.state}，不在待客服确认状态")
        shop_config = self._shop_config(shop_id)
        if current.requires_manager:
            self._check_manager(shop_config, confirmation)
        port = self._check_platform_rules(current, action, reason_code)

        # 调用前重读售后单（不读缓存）
        now = self._clock.now()
        info = self._read(shop_id, aftersale_id)
        pending: list[AuditEvent] = []
        if info is None or not is_actionable(info.status):
            rec = self._to_platform(current, info, now, pending)
            saved = self._repo.save(rec, expected_version=current.version)
            self._flush(pending)
            return saved
        if info.status not in ACTIONABLE_STATUSES[action]:
            raise ActionNotAllowed(
                f"售后单当前是「{AFTERSALE_STATUS_LABELS[info.status]}」，"
                f"不能{ACTION_LABELS[action]}，请刷新后重新确认"
            )
        if action == AfterSaleAction.AGREE_RETURN and info.type == AfterSaleType.REFUND_ONLY:
            raise ActionNotAllowed("仅退款的售后单不需要同意退货，请选择同意退款")
        if confirmation.expected_status_version != info.status_version:
            raise ActionNotAllowed(MSG_STALE)
        if _platform_changed(current, info):
            # 客服看到的版本是新的，但建议是按旧版本的售后单给的：先刷新建议再确认
            raise ActionNotAllowed(MSG_STALE)
        # R6 按重读到的金额再查一次：买家改了申请金额（状态不变、版本变了）也要店长确认
        if info.amount > shop_config.aftersale_amount_limit:
            self._check_manager(shop_config, confirmation)

        key = aftersale_idempotency_key(aftersale_id, action, info.status_version)
        base = {
            "aftersale_id": aftersale_id,
            "action": str(action),
            "rules": list(current.matched_rules),
            "suggestion": str(current.suggestion) if current.suggestion else None,
            "idempotency_key": key,
        }
        self._record(
            shop_id,
            AuditAction.REVIEW_CONFIRM,
            confirmation.operator,
            now,
            {
                **base,
                "by_manager": confirmation.by_manager,
                "expected_status_version": confirmation.expected_status_version,
                "reason_code": reason_code,
                "reason_text": mask_pii(reason_text) if reason_text else None,
            },
        )
        try:
            result = self._call_port(
                port, action, shop_id, aftersale_id, confirmation, key, reason_code, reason_text
            )
        except (ActionNotAllowed, ToolUnavailable) as e:
            self._record(
                shop_id,
                AuditAction.REVIEW_ACTION,
                confirmation.operator,
                now,
                {**base, "ok": False, "error": f"{type(e).__name__}: {e}"},
            )
            raise
        self._record(
            shop_id,
            AuditAction.REVIEW_ACTION,
            confirmation.operator,
            now,
            {
                **base,
                "ok": result.ok,
                "platform_code": result.platform_code,
                "duplicate": result.duplicate,
                "message": result.message,
            },
        )

        rec = _synced(with_result(current, result), info)
        if result.ok:
            rec = rec.model_copy(update={"confirmed_by": confirmation.operator})
            rec = self._move(
                rec, _NEXT_STATE[action], now, pending, by=confirmation.operator, note=str(action)
            )
        elif result.platform_code == "STATUS_CHANGED":
            rec = self._after_status_changed(rec, now, pending)
        saved = self._repo.save(rec, expected_version=current.version)
        self._flush(pending)
        return saved

    # ================================================================ 内部

    def _get(self, shop_id: str, aftersale_id: str) -> AfterSaleReview:
        rec = self._repo.get(shop_id, aftersale_id)
        if rec is None:
            raise ActionNotAllowed(f"售后单 {aftersale_id} 还没有审核记录，请先生成审核建议")
        return rec

    def _read(self, shop_id: str, aftersale_id: str) -> AfterSaleInfo | None:
        info = self._gateway.get_aftersale(shop_id, aftersale_id)
        if info is not None and info.shop_id != shop_id:
            raise TenantIsolationError(f"售后单 {aftersale_id} 不属于店铺 {shop_id}（§9.2）")
        return info

    def _evaluate(self, shop_id: str, info: AfterSaleInfo, now: datetime) -> ReviewSuggestion:
        """取订单、物流、买家售后记录后跑规则；查不到的数据按缺失处理并记风险提示（§12.2）。"""
        order: OrderInfo | None = None
        trace: LogisticsTrace | None = None
        history: list[AfterSaleInfo] = []
        flags: list[str] = []
        if is_actionable(info.status):
            try:
                order = self._gateway.get_order(shop_id, info.order_id)
            except ToolUnavailable:
                flags.append("订单暂时查不到，建议按缺少订单信息人工核实")
            try:
                trace = self._gateway.get_logistics(shop_id, info.order_id)
            except ToolUnavailable:
                flags.append("物流暂时查不到，建议人工核实签收情况")
            try:
                since = now - timedelta(days=self._config.repeat_window_days)
                history = self._gateway.list_aftersales(
                    shop_id, buyer_ref=info.buyer_ref, since=since
                )
            except ToolUnavailable:
                flags.append("买家售后记录暂时查不到，未检查 R7")
        suggestion = evaluate(
            info,
            order=order,
            trace=trace,
            buyer_history=history,
            config=self._config,
            shop_config=self._shop_config(shop_id),
            now=now,
        )
        if flags:
            suggestion = suggestion.model_copy(
                update={"risk_flags": [*suggestion.risk_flags, *flags]}
            )
        return suggestion

    def _apply(
        self,
        rec: AfterSaleReview,
        s: ReviewSuggestion,
        now: datetime,
        pending: list[AuditEvent],
    ) -> AfterSaleReview:
        pending.append(
            self._event(
                rec.shop_id,
                AuditAction.REVIEW_SUGGEST,
                SYSTEM_ACTOR,
                now,
                {
                    "aftersale_id": rec.aftersale_id,
                    "rules": list(s.matched_rules),
                    "suggestion": str(s.advice),
                    "next_state": str(s.next_state),
                    "requires_manager": s.requires_manager,
                    "risk_flags": list(s.risk_flags),
                    "platform_status": str(rec.platform_status),
                },
            )
        )
        return rec.model_copy(
            update={
                "matched_rules": list(s.matched_rules),
                "suggestion": s.advice,
                "reason": s.reason,
                "requires_manager": s.requires_manager,
                "risk_flags": list(s.risk_flags),
            }
        )

    def _move(
        self,
        rec: AfterSaleReview,
        to: ReviewState,
        now: datetime,
        pending: list[AuditEvent],
        *,
        by: str | None = None,
        note: str | None = None,
    ) -> AfterSaleReview:
        # 客服填写的备注可能带买家手机号等，记录和审计里都只存脱敏后的
        note = mask_pii(note) if note else note
        moved = transition(rec, to, at=now, by=by, note=note)
        detail: dict[str, Any] = {
            "aftersale_id": rec.aftersale_id,
            "from": str(rec.state),
            "to": str(to),
        }
        if note:
            detail["note"] = note
        pending.append(
            self._event(rec.shop_id, AuditAction.REVIEW_TRANSITION, by or SYSTEM_ACTOR, now, detail)
        )
        return moved

    def _to_platform(
        self,
        rec: AfterSaleReview,
        info: AfterSaleInfo | None,
        now: datetime,
        pending: list[AuditEvent],
    ) -> AfterSaleReview:
        """R1：平台已处理（或售后单已查不到）→ 不操作，同步状态。

        平台状态是"等待买家退货"（商家在平台后台同意了退货）时不进终态，改进 WAIT_RETURN：
        买家寄回后还要给"同意退款"的建议（§7.3.2 R3 仓库确认收到后同意退款）。"""
        if info is not None and info.status == AfterSaleStatus.WAIT_BUYER_RETURN:
            return self._to_wait_return(rec, info, now, pending)
        if info is not None:
            rec = _synced(rec, info)
            reason = f"售后单已是「{AFTERSALE_STATUS_LABELS[info.status]}」，不操作，只同步状态"
        else:
            reason = "售后单在平台已查不到，不操作"
        s = ReviewSuggestion(
            matched_rules=["R1"],
            advice=ReviewAdvice.NONE,
            next_state=ReviewState.PLATFORM,
            reason=reason,
            requires_manager=False,
            risk_flags=list(rec.risk_flags),
        )
        rec = self._apply(rec, s, now, pending)
        return self._move(rec, ReviewState.PLATFORM, now, pending)

    def _to_wait_return(
        self,
        rec: AfterSaleReview,
        info: AfterSaleInfo,
        now: datetime,
        pending: list[AuditEvent],
    ) -> AfterSaleReview:
        """商家已在平台后台同意退货（平台状态 WAIT_BUYER_RETURN）→ WAIT_RETURN。

        状态机里只有 CONFIRM 能进 WAIT_RETURN（REVIEW_TRANSITIONS）：其他状态先回到 CONFIRM。
        买家寄回（WAIT_SELLER_CONFIRM）后 refresh 重新跑规则，给出 R3 第二步"同意退款"的建议。"""
        rec = _synced(rec, info)
        if rec.state == ReviewState.WAIT_RETURN:
            return rec
        reason = (
            f"商家已在平台后台同意退货，等待买家寄回；{REASON_WAIT_SELLER_CONFIRM}（R3 第二步）"
        )
        s = ReviewSuggestion(
            matched_rules=["R1"],
            advice=ReviewAdvice.NONE,
            next_state=ReviewState.WAIT_RETURN,
            reason=reason,
            requires_manager=rec.requires_manager,
            risk_flags=list(rec.risk_flags),
        )
        rec = self._apply(rec, s, now, pending)
        note = "平台上已同意退货"
        if rec.state != ReviewState.CONFIRM:
            rec = self._move(rec, ReviewState.CONFIRM, now, pending, note=note)
        return self._move(rec, ReviewState.WAIT_RETURN, now, pending, note=note)

    def _verify_elapsed(self, rec: AfterSaleReview, now: datetime) -> bool:
        """R4 签收核查已经开始并满 verify_hours。"""
        started = rec.verify_started_at
        return started is not None and now - started >= timedelta(hours=self._config.verify_hours)

    def _verify_done(
        self, matched_rules: list[str], requires_manager: bool, risk_flags: list[str]
    ) -> ReviewSuggestion:
        """R4 签收核查满 verify_hours 仍未找到：建议同意；有 R7 标记时人工决定；R6 要店长确认。"""
        hours = self._config.verify_hours
        if "R7" in matched_rules:
            advice = ReviewAdvice.MANUAL
            reason = f"{hours} 小时后仍未找到；该买家有频繁售后风险标记，人工决定（R4、R7）"
        else:
            advice = ReviewAdvice.AGREE
            reason = f"{hours} 小时后仍未找到，建议同意（R4）"
        if requires_manager:
            reason += "；金额超过店铺上限，必须店长确认（R6）"
        return ReviewSuggestion(
            matched_rules=list(matched_rules),
            advice=advice,
            next_state=ReviewState.CONFIRM,
            reason=reason,
            requires_manager=requires_manager,
            risk_flags=list(risk_flags),
        )

    def _verify_timeout(
        self, rec: AfterSaleReview, now: datetime, pending: list[AuditEvent]
    ) -> AfterSaleReview:
        """R4：签收核查满 verify_hours 仍未找到（售后单仍待处理）→ CONFIRM，建议同意。"""
        if not self._verify_elapsed(rec, now):
            return rec
        s = self._verify_done(rec.matched_rules, rec.requires_manager, rec.risk_flags)
        rec = self._apply(rec, s, now, pending)
        hours = self._config.verify_hours
        return self._move(rec, ReviewState.CONFIRM, now, pending, note=f"签收核查满 {hours} 小时")

    def _reevaluate(
        self,
        rec: AfterSaleReview,
        info: AfterSaleInfo,
        now: datetime,
        pending: list[AuditEvent],
        *,
        reason_prefix: str | None = None,
    ) -> AfterSaleReview:
        """待确认期间平台上的售后单变了：按新数据重新跑规则，留在 CONFIRM。

        签收核查已经做完（R4 满 verify_hours）的不回到"先协商"，沿用核查结论，R6 / R7 按新数据。
        """
        s = self._evaluate(rec.shop_id, info, now)
        if s.next_state == ReviewState.VERIFY and self._verify_elapsed(rec, now):
            s = self._verify_done(s.matched_rules, s.requires_manager, s.risk_flags)
        if reason_prefix:
            s = s.model_copy(update={"reason": f"{reason_prefix}。{s.reason}"})
        return self._apply(rec, s, now, pending)

    def _after_status_changed(
        self, rec: AfterSaleReview, now: datetime, pending: list[AuditEvent]
    ) -> AfterSaleReview:
        """平台返回 STATUS_CHANGED：再读一次。

        已不可操作 → PLATFORM；否则留在 CONFIRM 重新给建议。"""
        try:
            latest = self._read(rec.shop_id, rec.aftersale_id)
        except ToolUnavailable:
            return rec.model_copy(update={"reason": f"{MSG_STALE}（{rec.reason or ''}）"})
        if latest is None or not is_actionable(latest.status):
            return self._to_platform(rec, latest, now, pending)
        rec = _synced(rec, latest)
        return self._reevaluate(rec, latest, now, pending, reason_prefix=MSG_STALE)

    @staticmethod
    def _check_manager(shop_config: ShopConfig, confirmation: HumanConfirmation) -> None:
        """R6：必须由店长（ShopConfig.manager_ids）本人、以店长身份确认。"""
        if not (confirmation.by_manager and confirmation.operator in shop_config.manager_ids):
            raise ActionNotAllowed(MSG_MANAGER)

    def _check_platform_rules(
        self, rec: AfterSaleReview, action: AfterSaleAction, reason_code: str | None
    ) -> AfterSaleActionPort:
        """平台差异（§7.3.2）：不依赖平台实时状态的检查，在重读之前做。"""
        port = self._port
        if port is None:
            raise ActionNotAllowed(MSG_NO_PORT)
        if action not in port.supported:
            if action == AfterSaleAction.REFUSE and self.platform == Platform.PDD:
                raise ActionNotAllowed(MSG_PDD_REFUSE)
            name = PLATFORM_NAMES.get(self.platform, str(self.platform))
            raise ActionNotAllowed(f"{name}不支持{ACTION_LABELS[action]}，请在商家后台手动操作")
        if action == AfterSaleAction.REFUSE:
            if self.platform == Platform.TAOBAO and not was_negotiated(rec):
                raise ActionNotAllowed(MSG_TAOBAO_REFUSE)
            if self.platform == Platform.DOUYIN and not reason_code:
                raise ActionNotAllowed(MSG_DOUYIN_REFUSE)
        return port

    @staticmethod
    def _call_port(
        port: AfterSaleActionPort,
        action: AfterSaleAction,
        shop_id: str,
        aftersale_id: str,
        confirmation: HumanConfirmation,
        key: str,
        reason_code: str | None,
        reason_text: str | None,
    ) -> ActionResult:
        if action == AfterSaleAction.AGREE_REFUND:
            return port.agree_refund(
                shop_id, aftersale_id, confirmation=confirmation, idempotency_key=key
            )
        if action == AfterSaleAction.AGREE_RETURN:
            return port.agree_return(
                shop_id, aftersale_id, confirmation=confirmation, idempotency_key=key
            )
        return port.refuse(
            shop_id,
            aftersale_id,
            confirmation=confirmation,
            idempotency_key=key,
            reason_code=reason_code,
            reason_text=reason_text,
        )

    @staticmethod
    def _event(
        shop_id: str, action: AuditAction, actor: str, now: datetime, detail: dict[str, Any]
    ) -> AuditEvent:
        return AuditEvent(
            shop_id=shop_id, actor=actor, action=str(action), detail=detail, created_at=now
        )

    def _record(
        self, shop_id: str, action: AuditAction, actor: str, now: datetime, detail: dict[str, Any]
    ) -> None:
        self._audit.record(self._event(shop_id, action, actor, now, detail))

    def _flush(self, pending: list[AuditEvent]) -> None:
        for event in pending:
            self._audit.record(event)
