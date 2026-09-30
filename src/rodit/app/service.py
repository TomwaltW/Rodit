"""应用服务：把平台事件、客服操作、商家操作接到一个平台单元上（方案 §4.2、§5.6、§5.8、§7.3、§8）。

一条平台事件（§4.2、§8.2）：
    店铺授权检查（§8.4）→ 适配器 on_event：
    买家消息 → 按会话串行（§8.3 按 conversation_id 保证顺序；晚到的旧消息由引擎按顺序处理）
    → 问答引擎（去重、懂→找→写→判）→ 结果：
       影子模式：什么都不展示（§13.4）；
       否则：侧边栏展示推荐（模式 B，§5.6）；展示失败只记日志，全自动照常往下走；
       auto_reply：由适配器发送（只有模式 A 的全自动能发），发送成功记为已发送；
       发送失败（ok=False 或抛错）→ 转人工（§12.2 全自动模式下任何故障由客服接管）；
       transfer_to_human：请适配器转人工（§5.7）；转人工失败记错误日志（需要人工跟进）。
    客服 / 机器人回声 → 写进会话（§5.3 最近几轮对话；和反馈、自动回复写回的同一次发送不重复存，
    见 _store_outgoing）；
    撤回 → 从会话里去掉那条消息（不再进上下文，原文不留），按它给的推荐过期、侧边栏刷新成剩下的
    （§5.3）；撤回先于那条消息到达时由会话仓储记墓碑，消息到了丢弃（§8.3）；写不进去时抛给平台重推。
客服操作（§5.8）：发送 / 编辑后发送 / 不合适 → 反馈闭环；发出去的文字写回会话，作为下一轮上下文
（反馈已经记下后写回失败只记日志，不报失败——报了客服重试也会被拒）。
发出的文字写回会话（回声、反馈写回、全自动写回）和撤回都持有这个会话的锁，查重和写入之间不会插进
另一份记录。锁的顺序固定：先店铺处理闸，再会话锁。
取消订购（§8.4）与处理互斥：按店铺的读写闸，处理消息、客服 / 商家操作是读方，取消订购是写方——
等在途的处理做完、期间不开始新的，再等引擎留在后台的任务（超过截止时间被丢下的分条、质检）做完，
删完数据再放行（之后的事件授权检查不过）。
商家操作：审核候选知识、看缺失知识、填政策问卷（附录 C）、看报表（§12.4）。
售后审核（§7.3.2）：拉单、刷新、协商、客服确认后执行（写操作一律要人确认）。每一步先查店铺授权
（§8.4 授权过期时停止处理）：授权无效时抛 AuthExpired，一次平台接口都不调；客服确认执行写操作时
（单元接了售后写接口）还要有写售后权限（refund.write，二期才申请，§8.4 最小权限），
没有就不碰写接口。
店铺生命周期（§8.4）：开通、状态、订购事件、续期、过期清理见 rodit.app.tenancy
（AssistService.tenancy）。
"""

from __future__ import annotations

import logging
import threading
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from rodit.adapters import StandaloneAssistAdapter
from rodit.app.console import ConsoleService
from rodit.app.rollout import RolloutService
from rodit.app.tenancy import TenancyService
from rodit.app.unit import Unit
from rodit.core.enums import (
    AfterSaleAction,
    Intent,
    Mode,
    RejectReason,
    Role,
    ShopStatus,
    SuggestionStatus,
)
from rodit.core.errors import (
    ActionNotAllowed,
    ConcurrencyConflict,
    ShopInactive,
    TenantIsolationError,
)
from rodit.core.models import (
    AfterSaleReview,
    EngineResult,
    HumanConfirmation,
    KnowledgeItem,
    StandardMessage,
    StoredMessage,
    SuggestionCard,
    strip_control_chars,
)
from rodit.core.pii import mask_pii
from rodit.core.ports import MessageRecall, PlatformAdapter
from rodit.core.scopes import MVP_SCOPES, SCOPE_REFUND_WRITE
from rodit.feedback import (
    AdoptionReport,
    FeedbackOutcome,
    QAReport,
    adoption_report,
    qa_report,
)
from rodit.knowledge import apply_questionnaire, describe_gap, find_gaps

logger = logging.getLogger(__name__)

REASON_SEND_FAILED = "全自动回复发送失败，转人工（§12.2）"
BACKGROUND_DRAIN_S = 30.0
"""取消订购删数据之前，最多等引擎为这家店留在后台的任务多久（秒，见 AssistService.revoke_shop）。
单次平台、大模型调用都有自己的超时，正常到不了这个上限。"""
# 同一次发送可能有两份记录：写回的（反馈 agent:{推荐 id}、全自动没有平台消息 ID 时 bot:{推荐 id}）
# 和平台推回来的回声（平台消息 ID）。两份相隔不超过这个时间、文字相同、中间没有买家消息才算同一次
_OUTGOING_DEDUPE_WINDOW = timedelta(minutes=2)
_OUTGOING_DEDUPE_SCAN = 20
_MIRRORED_PREFIXES = ("agent:", "bot:")

AFTERSALE_WRITE_SCOPES: dict[AfterSaleAction, frozenset[str]] = {
    AfterSaleAction.AGREE_REFUND: frozenset({SCOPE_REFUND_WRITE}),
    AfterSaleAction.AGREE_RETURN: frozenset({SCOPE_REFUND_WRITE}),
    AfterSaleAction.REFUSE: frozenset({SCOPE_REFUND_WRITE}),
}
"""客服确认后执行的售后写操作要的权限（§8.4 写售后到二期再申请）。没列出的操作也按写售后处理。"""


def aftersale_scopes(action: AfterSaleAction) -> frozenset[str]:
    """执行这个售后写操作时店铺授权必须具备的权限：只读权限 + 写售后。"""
    return MVP_SCOPES | AFTERSALE_WRITE_SCOPES.get(action, frozenset({SCOPE_REFUND_WRITE}))


def _is_mirrored(msg_id: str) -> bool:
    """写回的记录（不是平台消息 ID）。"""
    return msg_id.startswith(_MIRRORED_PREFIXES)


def _reply_key(text: str | None) -> str:
    """比较两份记录是不是同一句话的形式：去掉控制字符、NFKC 归一、遮掉个人信息、合并空白。
    回声是原文，写回的记录可能做过个人信息最小化（[手机号]、[地址]）、全半角不同，按这个形式两边一致。"""
    cleaned = unicodedata.normalize("NFKC", strip_control_chars(text or ""))
    return " ".join(mask_pii(cleaned).split())


def _same_send(
    recent: Sequence[StoredMessage], *, msg_id: str, text: str, at: datetime
) -> StoredMessage | None:
    """recent 里和这条（msg_id、text、at）属于同一次发送的另一份记录；没有返回 None。

    只配"写回的 ↔ 平台回声"：两条平台消息、两条写回的记录永远是两条；回声晚于写回的记录、
    中间又隔着买家消息的，不是同一次发送。写回的记录用的是服务端收到反馈的时间，只是实际发送时间
    的上限：回声（平台发送时间）早于它时，中间的买家消息是发送之后才到的，不影响配对。
    有几条可配时取最早的（按发送顺序一一对上）。
    """
    mirrored = _is_mirrored(msg_id)
    key = _reply_key(text)
    for m in recent:
        if m.role == Role.BUYER or _is_mirrored(m.msg_id) == mirrored:
            continue
        if abs(m.created_at - at) > _OUTGOING_DEDUPE_WINDOW or _reply_key(m.text) != key:
            continue
        mirrored_at, echo_at = (at, m.created_at) if mirrored else (m.created_at, at)
        if echo_at > mirrored_at and any(
            b.role == Role.BUYER and mirrored_at < b.created_at < echo_at for b in recent
        ):
            continue
        return m
    return None


@dataclass
class _LockEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0  # 持有和等待这把锁的线程数


class _KeyedLocks:
    """按键的互斥锁（同一会话串行，§8.3）。没人持有、也没人在等的键立即删掉：
    占用的内存只和正在处理的会话数有关，不随见过的会话数增长。"""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[tuple[str, str], _LockEntry] = {}

    def __len__(self) -> int:
        with self._guard:
            return len(self._entries)

    @contextmanager
    def hold(self, key: tuple[str, str]) -> Iterator[None]:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = self._entries[key] = _LockEntry()
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    del self._entries[key]


class _ShopGate:
    """按店铺的读写闸（§8.4 取消订购后停止处理并删除数据）。

    processing（读方，可以并发）：处理平台事件、客服 / 商家操作；
    revoking（写方，独占）：等这家店在途的处理全部做完，期间新来的处理排队，写方做完再放行。
    有写方在等时新的读方也排队（写优先），繁忙的店铺也能取消订购。不可重入：持有读方时不能
    再取写方。没有进行中的处理时不占内存。
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._active: dict[str, int] = {}
        self._revoking: dict[str, int] = {}  # 正在或等着取消订购的次数
        self._exclusive: set[str] = set()

    @contextmanager
    def processing(self, shop_id: str) -> Iterator[None]:
        with self._cond:
            while self._revoking.get(shop_id):
                self._cond.wait()
            self._active[shop_id] = self._active.get(shop_id, 0) + 1
        try:
            yield
        finally:
            with self._cond:
                _decrement(self._active, shop_id)
                self._cond.notify_all()

    @contextmanager
    def revoking(self, shop_id: str) -> Iterator[None]:
        with self._cond:
            self._revoking[shop_id] = self._revoking.get(shop_id, 0) + 1
            while self._active.get(shop_id) or shop_id in self._exclusive:
                self._cond.wait()
            self._exclusive.add(shop_id)
        try:
            yield
        finally:
            with self._cond:
                self._exclusive.discard(shop_id)
                _decrement(self._revoking, shop_id)
                self._cond.notify_all()


def _decrement(counts: dict[str, int], key: str) -> None:
    n = counts.get(key, 0) - 1
    if n > 0:
        counts[key] = n
    else:
        counts.pop(key, None)


class AssistService:
    """一个平台单元的应用服务。线程安全：同一会话的消息串行处理，不同会话并行；
    取消订购等在途处理做完再删数据（见模块文档）。"""

    def __init__(self, unit: Unit) -> None:
        self.unit = unit
        self._locks = _KeyedLocks()
        self._gate = _ShopGate()
        self.background_drain_s = BACKGROUND_DRAIN_S
        self.console = ConsoleService(unit, self.processing)  # 商家控制台（§9.1、§5.2）
        self.rollout = RolloutService(unit, self.processing)  # 上线节奏与每周复核（§13.4、§13.5）
        # 店铺生命周期：开通、状态、订购事件、续期、过期清理（§8.4）
        self.tenancy = TenancyService(
            unit, self.processing, self.revoke_shop, exclusive=self.exclusive
        )

    @contextmanager
    def processing(self, shop_id: str) -> Iterator[None]:
        """按店铺的读方闸门：客服 / 商家 / 复核操作都在里面做，取消订购等它们做完再删数据（§8.4）。
        不可重入：持有时不能再调取消订购。"""
        with self._gate.processing(shop_id):
            yield

    @contextmanager
    def exclusive(self, shop_id: str) -> Iterator[None]:
        """按店铺的写方闸门（和取消订购同一把）：等这家店在途的处理做完、期间新来的排队。
        店铺生命周期改店铺记录（开通、切换状态）时用，读改写之间不会插进同一进程里的其他修改。
        不可重入：持有时不能再调 processing 或取消订购。"""
        with self._gate.revoking(shop_id):
            yield

    # ================================================================ 买家消息

    def handle_event(
        self, access: str, raw: Mapping[str, Any], *, expected_shop_id: str | None = None
    ) -> EngineResult | None:
        """平台推送（模式 A）/ 插件上报（模式 B）的一条原始事件。不是买家消息、重复消息返回 None；
        客服 / 机器人回声写进会话、撤回从会话里去掉（都返回 None，§5.3）。

        先不改状态地看事件属于哪家店（peek_shop_id），核对 expected_shop_id（调用方身份）和授权，
        通过后才交给适配器处理——一个店铺的事件不能改动另一个店铺的状态（§9.2），
        授权过期的店铺连去重记录都不留（§8.4）。
        授权无效抛 AuthExpired；店铺不符抛 TenantIsolationError；
        引擎本身不会因 AI 故障抛错（§12.2）。
        """
        adapter = self.unit.adapters.get(self.unit.platform, access)
        event = dict(raw)
        shop_id = adapter.peek_shop_id(event)
        if shop_id is None:
            return None
        if expected_shop_id is not None and shop_id != expected_shop_id:
            raise TenantIsolationError("事件不属于调用方的店铺")
        with self._gate.processing(shop_id):
            self.unit.auth.require(shop_id)
            on_event = getattr(adapter, "on_event", None)  # 只实现了 on_message 的适配器照常可用
            parsed = on_event(event) if callable(on_event) else adapter.on_message(event)
            if parsed is None:
                return None
            if parsed.shop_id != shop_id:  # 适配器实现不一致时兜底
                raise TenantIsolationError("事件解析前后店铺不一致")
            if isinstance(parsed, StandardMessage):
                return self._process(adapter, parsed)
            if isinstance(parsed, StoredMessage):
                self._record_echo(parsed)
            elif isinstance(parsed, MessageRecall):
                self._record_recall(adapter, parsed)
            return None

    def _record_echo(self, echo: StoredMessage) -> None:
        """客服手打的回复（没用推荐）只能从回声里拿到：写进会话，作为最近几轮对话（§5.3）。
        调用方已持有店铺处理闸；这里再取会话锁（和反馈写回、全自动写回串行）。

        写库失败时异常向外抛，让平台重推：回声要接替反馈写回的记录（改成平台 msg_id），
        之后按平台 msg_id 撤回才找得到；重推是幂等的（按 msg_id 去重）。"""
        if echo.role == Role.BUYER or not (echo.text or "").strip():
            return
        with self._locks.hold((echo.shop_id, echo.conversation_id)):
            self._store_outgoing(
                shop_id=echo.shop_id,
                conversation_id=echo.conversation_id,
                msg_id=echo.msg_id,
                role=echo.role,
                text=echo.text or "",
                at=echo.created_at,
            )

    def _record_recall(self, adapter: PlatformAdapter, recall: MessageRecall) -> None:
        """撤回：那条消息不再进上下文，原文不留（§5.3、§12.3 最小化）；按它给的推荐不能再发——
        引擎把这些推荐过期（on_recall），侧边栏刷新成会话里剩下的推荐（通常是清空）。
        撤回先于那条消息到达（§8.3 乱序）：会话仓储记墓碑，消息到了丢弃（不保存、不回答）。

        写不进去时异常向外抛（平台收到失败会重推；适配器不按 msg_id 挡掉重推的撤回，
        recall_message 和推荐过期都是幂等的，重推时补上没做完的那一步）。侧边栏刷新失败只记日志
        （和展示推荐一样）。持有会话锁：回声正在接替写回的记录、引擎正在处理这个会话时，等它做完。
        """
        with self._locks.hold((recall.shop_id, recall.conversation_id)):
            self.unit.repos.conversations.recall_message(
                recall.shop_id, recall.conversation_id, recall.recalled_msg_id
            )
            remaining = self.unit.engine.on_recall(
                recall.shop_id, recall.conversation_id, recall.recalled_msg_id
            )
            if remaining is not None:
                self._show(adapter, recall.shop_id, recall.conversation_id, remaining)

    def ask_standalone(
        self,
        *,
        shop_id: str,
        session_id: str,
        question: str,
        item_id: str | None = None,
        sku_id: str | None = None,
        agent_id: str | None = None,
        request_id: str | None = None,
    ) -> EngineResult | None:
        """模式 C 独立话术助手（§8.1）：客服粘贴买家问题，拿推荐回答。不需要平台授权。"""
        adapter = self.unit.adapters.get(self.unit.platform, "C")
        if not isinstance(adapter, StandaloneAssistAdapter):
            raise ActionNotAllowed("本单元没有配置模式 C 独立话术助手")
        with self._gate.processing(shop_id):
            shop = self.unit.repos.knowledge.get_shop(shop_id)
            if shop is None or shop.status == ShopStatus.SUSPENDED:
                raise ShopInactive(shop_id, shop.status if shop else None)
            raw: dict[str, Any] = {
                "shop_id": shop_id,
                "session_id": session_id,
                "question": question,
            }
            for key, value in (
                ("item_id", item_id),
                ("sku_id", sku_id),
                ("agent_id", agent_id),
                ("request_id", request_id),
            ):
                if value is not None:
                    raw[key] = value
            msg = adapter.on_message(raw)
            if msg is None:
                return None
            return self._process(adapter, msg)

    def _process(self, adapter: PlatformAdapter, msg: StandardMessage) -> EngineResult | None:
        """调用方已持有该店铺的处理闸（_gate.processing）。"""
        with self._locks.hold((msg.shop_id, msg.conversation_id)):
            result = self.unit.engine.handle(msg)
            if result is None:
                return None
            if result.mode != Mode.SHADOW:
                self._show(
                    adapter,
                    msg.shop_id,
                    msg.conversation_id,
                    result.suggestions,
                    alerts=result.alerts,
                    notice=result.no_suggestion_reason,
                )
            reason = result.no_suggestion_reason or "转人工"
            if result.auto_reply is not None and result.mode == Mode.AUTO:
                if not self._send_auto_reply(adapter, msg, result):
                    # 买家由机器人接待，回复没发出去：不能既不回也不转（§12.2、§5.7）
                    reason = REASON_SEND_FAILED
                    result = result.model_copy(update={"transfer_to_human": True})
            if result.transfer_to_human:
                self._transfer(adapter, msg, reason)
            return result

    def _show(
        self,
        adapter: PlatformAdapter,
        shop_id: str,
        conversation_id: str,
        cards: list[SuggestionCard],
        *,
        alerts: Sequence[str] = (),
        notice: str | None = None,
    ) -> None:
        """把推荐推到侧边栏 / 工作台（空列表表示清空），连同标红提醒（EngineResult.alerts）和
        没有推荐时的原因（no_suggestion_reason）。失败不抛给平台：消息已经存下，重推会被
        去重；全自动照常发送（发送失败再转人工，§12.2），智能辅助下客服本来就在接待。
        撤回后刷新侧边栏时不带提醒（那条消息的提醒随它的推荐一起过期）。"""
        try:
            adapter.show_suggestions(
                shop_id, conversation_id, cards, alerts=list(alerts), notice=notice
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "推荐展示失败（%s）：店铺 %s 会话 %s",
                type(exc).__name__,
                shop_id,
                conversation_id,
            )

    def _transfer(self, adapter: PlatformAdapter, msg: StandardMessage, reason: str) -> None:
        try:
            outcome = adapter.transfer_to_human(msg.shop_id, msg.conversation_id, reason)
        except ActionNotAllowed:
            return  # 该接入模式不支持转人工：客服本来就在接待（智能辅助）
        except Exception as exc:  # noqa: BLE001 — 不抛给平台：重推会被去重，更没人接
            logger.error(
                "转人工失败（%s）：店铺 %s 会话 %s 需要人工跟进",
                type(exc).__name__,
                msg.shop_id,
                msg.conversation_id,
            )
            return
        if not outcome.ok:
            logger.error(
                "转人工失败（平台返回失败）：店铺 %s 会话 %s 需要人工跟进",
                msg.shop_id,
                msg.conversation_id,
            )

    def _send_auto_reply(
        self, adapter: PlatformAdapter, msg: StandardMessage, result: EngineResult
    ) -> bool:
        """发送全自动回复；发出去了返回 True。发送失败（ok=False、接入模式不能发、抛错）返回
        False，推荐记录保持 SHOWN，留给接手的客服。调用方（_process）已持有会话锁。"""
        reply = result.auto_reply
        assert reply is not None
        try:
            sent = adapter.send(msg.shop_id, msg.conversation_id, reply.text)
        except Exception as exc:  # noqa: BLE001 — 含 ActionNotAllowed（接入模式不能发送）
            logger.warning("全自动回复发送失败（%s），转人工", type(exc).__name__)
            return False
        if not sent.ok:
            logger.warning("全自动回复发送失败（平台返回失败），转人工")
            return False
        try:
            self._store_outgoing(
                shop_id=msg.shop_id,
                conversation_id=msg.conversation_id,
                msg_id=sent.platform_msg_id or f"bot:{reply.suggestion_id}",
                role=Role.BOT,
                text=reply.text,
                at=self.unit.clock.now(),
            )
            if reply.suggestion_id is not None:
                self._mark_sent_by_bot(msg.shop_id, reply.suggestion_id)
        except Exception as exc:  # noqa: BLE001 — 已经发出去了，只是记录没写上
            logger.warning("全自动回复已发送，记录写入失败（%s）", type(exc).__name__)
        return True

    def _mark_sent_by_bot(self, shop_id: str, suggestion_id: int) -> None:
        record = self.unit.repos.suggestions.get(shop_id, suggestion_id)
        if record is None:
            return
        try:
            self.unit.repos.suggestions.update(
                record.model_copy(update={"status": SuggestionStatus.SENT, "agent_id": "bot"}),
                expected_status=SuggestionStatus.SHOWN,
            )
        except ConcurrencyConflict:
            pass

    def _store_outgoing(
        self,
        *,
        shop_id: str,
        conversation_id: str,
        msg_id: str,
        role: Role,
        text: str,
        at: datetime,
    ) -> bool:
        """客服 / 机器人发出的文字写回会话（§5.3 多轮上下文）。存了返回 True。

        调用方必须持有这个会话的锁（self._locks，不可重入，所以不在这里取）：查重和写入之间
        不能插进同一次发送的另一份记录。
        同一次发送最多两份记录：写回的（反馈 agent:{推荐 id}、全自动没有平台消息 ID 时
        bot:{推荐 id}）和平台回声（平台消息 ID），只有这两份之间去重（见 _same_send）：
        - 回声先到：写回的不再存；
        - 写回的先到：回声接替它——先撤回写回的那条，再按平台消息 ID 存回声的原文和时间
          （角色沿用写回的，机器人仍记为机器人），之后撤回这个平台消息 ID 能找到它。
        按原文比较会受脱敏影响，比较用 _reply_key。
        """
        repo = self.unit.repos.conversations
        recent = repo.recent_messages(shop_id, conversation_id, limit=_OUTGOING_DEDUPE_SCAN)
        if any(m.msg_id == msg_id for m in recent):
            return False
        twin = _same_send(recent, msg_id=msg_id, text=text, at=at)
        if twin is not None:
            if _is_mirrored(msg_id):
                return False
            # 先撤回写回的那条（原文不留两份；这一步之后失败最多少一轮上下文），再存回声
            repo.recall_message(shop_id, conversation_id, twin.msg_id)
            role = twin.role
        return repo.add_outgoing(
            shop_id=shop_id,
            conversation_id=conversation_id,
            msg_id=msg_id,
            role=role,
            text=text,
            at=at,
        )

    # ================================================================ 客服反馈（§5.8）

    def _mirror_agent_reply(
        self, outcome: FeedbackOutcome, text: str, *, at: datetime | None = None
    ) -> None:
        """客服发出的文字写回会话，作为下一轮的多轮上下文（§5.3）。回声已经先到时不重复存。

        反馈（状态、标注、审计）已经记下了：写回失败只记日志、不抛——抛出去客服会按提示重试，
        而重试会被拒（推荐已不在展示中）；少这一轮上下文和回声写不进去一样，不影响下一轮出推荐。
        """
        record = outcome.record
        # 时间在等会话锁之前取：引擎处理这个会话可能要几秒，锁后再取会把写回的记录排到
        # 这期间到的买家消息后面
        at = at or self.unit.clock.now()
        with self._locks.hold((record.shop_id, record.conversation_id)):
            try:
                self._store_outgoing(
                    shop_id=record.shop_id,
                    conversation_id=record.conversation_id,
                    msg_id=f"agent:{record.id}",
                    role=Role.AGENT,
                    text=text,
                    at=at,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("客服回复写回会话失败（%s）", type(exc).__name__)

    def feedback_sent(self, shop_id: str, suggestion_id: int, *, agent_id: str) -> FeedbackOutcome:
        with self._gate.processing(shop_id):
            outcome = self.unit.feedback.record_sent(shop_id, suggestion_id, agent_id=agent_id)
            self._mirror_agent_reply(outcome, outcome.record.text, at=self.unit.clock.now())
            return outcome

    def feedback_edited(
        self, shop_id: str, suggestion_id: int, *, agent_id: str, edited_text: str
    ) -> FeedbackOutcome:
        with self._gate.processing(shop_id):
            outcome = self.unit.feedback.record_edited(
                shop_id, suggestion_id, agent_id=agent_id, edited_text=edited_text
            )
            # 写回实际发出去的文字（回声也是原文）；反馈记录里的 edited_text 做过个人信息最小化，
            # 是给候选知识和标注用的。会话里的消息都存原文，进提示词时再脱敏（§5.3）
            self._mirror_agent_reply(
                outcome, strip_control_chars(edited_text).strip(), at=self.unit.clock.now()
            )
            return outcome

    def feedback_rejected(
        self,
        shop_id: str,
        suggestion_id: int,
        *,
        agent_id: str,
        reason: RejectReason,
        true_intent: Intent | None = None,
    ) -> FeedbackOutcome:
        with self._gate.processing(shop_id):
            return self.unit.feedback.record_rejected(
                shop_id, suggestion_id, agent_id=agent_id, reason=reason, true_intent=true_intent
            )

    # ================================================================ 商家（知识中心）

    def pending_candidates(self, shop_id: str) -> list[KnowledgeItem]:
        return self.unit.feedback.pending_candidates(shop_id)

    def approve_candidate(self, shop_id: str, knowledge_id: int, *, merchant: str) -> KnowledgeItem:
        with self._gate.processing(shop_id):
            return self.unit.feedback.approve_candidate(shop_id, knowledge_id, merchant=merchant)

    def retire_knowledge(self, shop_id: str, knowledge_id: int, *, merchant: str) -> KnowledgeItem:
        with self._gate.processing(shop_id):
            return self.unit.feedback.retire(shop_id, knowledge_id, merchant=merchant)

    def knowledge_gaps(self, shop_id: str) -> list[dict[str, str]]:
        """要提醒商家补的知识：冷启动检查（§5.2.3）+ 引擎和反馈记下的缺失（§5.7、§5.8），去重。"""
        markers = list(find_gaps(self.unit.repos.knowledge, shop_id))
        for gap in self.unit.repos.feedback.list_gaps(shop_id):
            if gap.missing not in markers:
                markers.append(gap.missing)
        return [{"marker": m, "description": describe_gap(m)} for m in markers]

    def apply_questionnaire(self, shop_id: str, answers: Mapping[int, str]) -> dict[str, Any]:
        """店铺政策问卷（附录 C）：写入政策，合并店铺配置覆盖项。"""
        repo = self.unit.repos.knowledge
        # 回答先去掉 NUL 等控制字符（PostgreSQL 的 TEXT / JSONB 写不进去）
        cleaned = {
            k: strip_control_chars(v) if isinstance(v, str) else v for k, v in answers.items()
        }
        with self._gate.processing(shop_id):  # 读改写不能跨过取消订购（否则会把删掉的店铺写回来）
            shop = repo.get_shop(shop_id)
            if shop is None:
                raise KeyError(shop_id)
            entries, patch = apply_questionnaire(shop_id, cleaned, self.unit.clock.now())
            for entry in entries:
                repo.upsert_policy(entry)
            # 只写问卷改到的配置键（仓储里原子地读改写，并发的配置修改不会互相覆盖）
            if patch and repo.update_shop_config(shop_id, patch) is None:
                raise KeyError(shop_id)
            return {"policies": [e.key for e in entries], "config": sorted(patch)}

    # ================================================================ 报表（§12.4）

    def adoption(self, shop_id: str, *, since: datetime | None = None) -> AdoptionReport:
        records = self.unit.repos.suggestions.list(shop_id, since=since)
        return adoption_report(records, since=since)

    def qa(self, shop_id: str, *, since: datetime | None = None) -> QAReport:
        return qa_report(self.unit.repos.audit, shop_id, since)

    # ================================================================ 售后审核（§7.3.2）

    def review_aftersale(self, shop_id: str, aftersale_id: str) -> AfterSaleReview:
        """拉单给建议（已有记录时重读平台状态）。授权无效抛 AuthExpired，不调平台接口（§8.4）。"""
        with self._gate.processing(shop_id):
            self.unit.auth.require(shop_id)
            existing = self.unit.repos.reviews.get(shop_id, aftersale_id)
            if existing is None:
                return self.unit.reviews.review(shop_id, aftersale_id)
            return self.unit.reviews.refresh(shop_id, aftersale_id)

    def negotiate_aftersale(
        self, shop_id: str, aftersale_id: str, *, operator: str, note: str | None = None
    ) -> AfterSaleReview:
        with self._gate.processing(shop_id):
            self.unit.auth.require(shop_id)
            return self.unit.reviews.negotiate(shop_id, aftersale_id, operator=operator, note=note)

    def negotiation_done(
        self, shop_id: str, aftersale_id: str, *, operator: str, outcome: str | None = None
    ) -> AfterSaleReview:
        with self._gate.processing(shop_id):
            self.unit.auth.require(shop_id)
            return self.unit.reviews.negotiation_done(
                shop_id, aftersale_id, operator=operator, outcome=outcome
            )

    def confirm_aftersale(
        self,
        shop_id: str,
        aftersale_id: str,
        action: AfterSaleAction,
        *,
        operator: str,
        expected_status_version: str,
        by_manager: bool = False,
        reason_code: str | None = None,
        reason_text: str | None = None,
    ) -> AfterSaleReview:
        """客服确认后执行（§7.3.2）：确认人、看到的平台状态版本都要带上，助手会重读再操作。

        先查授权（§8.4）：授权无效抛 AuthExpired；单元接了售后写接口时还要有写售后权限
        （aftersale_scopes，二期才申请），没有就抛 AuthExpired，写接口一次都不调。
        没接写接口（MVP）时审核助手本来就拒绝执行（ActionNotAllowed），不另查写权限。"""
        confirmation = HumanConfirmation(
            operator=operator,
            confirmed_at=self.unit.clock.now(),
            expected_status_version=expected_status_version,
            by_manager=by_manager,
        )
        with self._gate.processing(shop_id):
            scopes = aftersale_scopes(action) if self.unit.action_port is not None else MVP_SCOPES
            self.unit.auth.require(shop_id, scopes)
            return self.unit.reviews.confirm(
                shop_id,
                aftersale_id,
                action,
                confirmation=confirmation,
                reason_code=reason_code,
                reason_text=reason_text,
            )

    # ================================================================ 店铺生命周期（§8.4）

    def revoke_shop(
        self,
        shop_id: str,
        *,
        guard: Callable[[], bool] | None = None,
        on_purged: Callable[[dict[str, int]], None] | None = None,
    ) -> dict[str, int] | None:
        """取消订购 / 授权过期：删令牌、停止处理，并按约定删除该店铺在本单元的全部数据。
        返回各项删除条数。

        先等这家店在途的处理做完（期间新来的排队），再等引擎为这家店留在后台的任务做完
        （到截止时间被丢下、还在跑的分条和质检：它们迟到的工具缓存、大模型用量写入要一并删掉，
        最多等 background_drain_s 秒），再删令牌和数据，最后放行：在途处理写下的记录一并删掉，
        排队的事件授权检查不过（§8.4）。
        guard：在写方闸门里、删之前判断（订购事件是不是晚到的旧事件、是否还过期）；返回 False 时
        什么都不删，返回 None。on_purged(各项删除条数)：删完之后、放开闸门之前调（写清理记录）。
        """
        purgers: list[object] = [*self.unit.repos.purgers(), self.unit.cache]
        for platform, access in self.unit.adapters.keys():
            adapter = self.unit.adapters.get(platform, access)
            if hasattr(adapter, "purge_shop"):
                purgers.append(adapter)
        purgers.extend(self.unit.extra_purgers)
        with self._gate.revoking(shop_id):
            if guard is not None and not guard():
                return None
            self._drain_background(shop_id)
            counts = self.unit.auth.revoke(shop_id, purgers)
            if on_purged is not None:
                on_purged(counts)
            return counts

    def _drain_background(self, shop_id: str) -> None:
        """等引擎为这家店留在后台的任务做完（调用方持有写方闸门：期间不会有新的）。
        单次平台 / 大模型调用都有自己的超时，这里再兜一个上限；超时照常删除并记警告。"""
        wait = getattr(self.unit.engine, "wait_background", None)
        if wait is None:
            return
        if not wait(shop_id, timeout=self.background_drain_s):
            logger.warning(
                "取消订购：店铺 %s 还有后台任务超过 %.0f 秒没做完，照常删除"
                "（之后迟到的写入不在这次清理里）",
                shop_id,
                self.background_drain_s,
            )


__all__ = ["AFTERSALE_WRITE_SCOPES", "AssistService", "aftersale_scopes"]
