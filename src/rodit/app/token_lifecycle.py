"""店铺令牌续期（方案 §8.4 "按店铺保存访问令牌并按期刷新；令牌只存放在对应的平台云内"）。

续期任务（python -m rodit.ops refresh-tokens，定时在平台单元内跑）：

    ShopAuthRepository.expiring_shop_ids(现在 + TenancyConfig.refresh_ahead_hours)
    （只有店铺 ID 和到期时间，不跨店读令牌）→ 逐店，在 processing(shop_id) 里（和取消订购互斥）：
    重读授权 → 有续期令牌、还没被标记需要重新授权 → TokenRefreshPort.refresh：
      成功：保存新令牌（平台没换续期令牌时沿用旧的），refreshed_at = 现在，审计 auth.refreshed；
      AuthExpired（续期令牌无效、过期、被商家撤销）：标记 needs_reauth（之后 AuthGuard.require
        不通过，问答链路 403 shop_not_authorized），审计 auth.refresh_failed；
      ToolUnavailable（限流、故障、超时）：什么都不改，下次再试；
      其他异常（平台云代码的问题）：只记异常类型，下次再试，任务退出码非 0。
    续期期间授权被换掉了（重新订购、取消订购）：不覆盖，按新的授权来。

令牌（access_token、refresh_token）只在内存里以 SecretStr 传递：不进日志、审计、异常消息和命令行
参数；日志只有店铺 ID、异常类型和失败原因（ToolUnavailable.reason）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from rodit.app.unit import Unit
from rodit.core.audit import AuditAction
from rodit.core.errors import AuthExpired, RoditError, TokenDecryptionError, ToolUnavailable
from rodit.core.models import AuditEvent, ShopAuth
from rodit.core.ports import TokenRefreshPort

logger = logging.getLogger(__name__)

ShopGate = Callable[[str], AbstractContextManager[None]]
"""AssistService.processing：按店铺的读方闸门（和取消订购互斥）。"""

REFRESH_ACTOR = "system"


class RefreshUnavailable(RoditError):
    """本单元没有接平台的令牌续期接口（PlatformBindings.token_refresher）：续期任务不能跑。"""


class RefreshOutcome(StrEnum):
    """一家店这次续期的结果（RefreshReport 按它分组，只有店铺 ID）。"""

    REFRESHED = "refreshed"  # 换到了新令牌
    NEEDS_REAUTH = "needs_reauth"  # 平台拒绝续期 / 令牌读不出来：已标记需要重新授权
    RETRY = "retry"  # 平台接口限流、故障、超时：下次再试
    ERROR = "error"  # 平台云代码抛了意料之外的异常：下次再试，要人看
    NO_REFRESH_TOKEN = "no_refresh_token"  # 没有续期令牌：只能等商家重新授权
    REFRESH_EXPIRED = "refresh_expired"  # 续期令牌自己也过期了：只能等商家重新授权
    ALREADY_REAUTH = "already_needs_reauth"  # 之前已标记需要重新授权：不再调平台
    SKIPPED = "skipped"  # 授权已删除、已经续过、或者续期期间被换掉了


@dataclass
class RefreshReport:
    """续期任务的结果：每种结果对应的店铺 ID（不含令牌）。"""

    shops: dict[RefreshOutcome, list[str]] = field(default_factory=dict)

    def add(self, shop_id: str, outcome: RefreshOutcome) -> None:
        self.shops.setdefault(outcome, []).append(shop_id)

    def of(self, outcome: RefreshOutcome) -> list[str]:
        return list(self.shops.get(outcome, []))

    @property
    def checked(self) -> int:
        return sum(len(v) for v in self.shops.values())

    def counts(self) -> dict[str, int]:
        """各结果的店铺数（全部结果都列出，没有的为 0）。"""
        return {o.value: len(self.shops.get(o, [])) for o in RefreshOutcome}


def refresh_expiring_tokens(
    unit: Unit, *, processing: ShopGate, now: datetime | None = None
) -> RefreshReport:
    """给 access_token 在 refresh_ahead_hours 之内到期（含已过期）的店铺续期，见模块说明。

    没接续期接口时抛 RefreshUnavailable。一家店失败不影响其他店；存储故障（StorageUnavailable 等）
    向外抛（这次任务整体失败，下次再跑）。
    """
    refresher = unit.token_refresher
    if refresher is None:
        raise RefreshUnavailable(
            "本单元没有接令牌续期接口（PlatformBindings.token_refresher），续期任务不能跑"
        )
    now = now or unit.clock.now()
    before = now + timedelta(hours=unit.config.tenancy.refresh_ahead_hours)
    report = RefreshReport()
    for shop_id, _ in unit.repos.auth.expiring_shop_ids(before):
        with processing(shop_id):
            outcome = _refresh_one(unit, refresher, shop_id, before=before, now=now)
        report.add(shop_id, outcome)
    return report


def _refresh_one(
    unit: Unit, refresher: TokenRefreshPort, shop_id: str, *, before: datetime, now: datetime
) -> RefreshOutcome:
    """调用方持有这家店的处理闸（和取消订购互斥）。"""
    repo = unit.repos.auth
    try:
        auth = repo.get(shop_id)
    except TokenDecryptionError:
        # 令牌读不出来（换过密钥、旧版加密）：续不了，按需要重新授权处理（AuthGuard 本来就拒绝）
        repo.mark_needs_reauth(shop_id)
        _record(unit, shop_id, AuditAction.AUTH_REFRESH_FAILED, now, {"reason": "token_unreadable"})
        logger.warning("令牌续期：店铺 %s 的令牌读不出来，已标记需要重新授权", shop_id)
        return RefreshOutcome.NEEDS_REAUTH
    if auth is None or auth.expires_at > before:
        return RefreshOutcome.SKIPPED  # 已经取消订购，或者列出之后有人续过 / 重新授权了
    if auth.needs_reauth:
        return RefreshOutcome.ALREADY_REAUTH
    if auth.refresh_token is None:
        return RefreshOutcome.NO_REFRESH_TOKEN
    if auth.refresh_expires_at is not None and auth.refresh_expires_at <= now:
        # 不调平台（必然被拒）；access_token 到期后授权检查自然不过，到期告警里看得到
        return RefreshOutcome.REFRESH_EXPIRED
    try:
        fresh = refresher.refresh(shop_id, auth.refresh_token)
    except AuthExpired:
        repo.mark_needs_reauth(shop_id)
        _record(
            unit,
            shop_id,
            AuditAction.AUTH_REFRESH_FAILED,
            now,
            {"reason": "rejected", "expires_at": auth.expires_at.isoformat()},
        )
        logger.warning("令牌续期被平台拒绝：店铺 %s 需要商家重新授权", shop_id)
        return RefreshOutcome.NEEDS_REAUTH
    except ToolUnavailable as exc:
        logger.info("令牌续期暂时失败（%s）：店铺 %s 下次再试", exc.reason, shop_id)
        return RefreshOutcome.RETRY
    except Exception as exc:  # noqa: BLE001 — 平台云代码的问题：只记类型（说明里可能有令牌）
        logger.error("令牌续期出错（%s）：店铺 %s 下次再试", type(exc).__name__, shop_id)
        return RefreshOutcome.ERROR
    current = repo.get(shop_id)
    if current is None or not _same_grant(current, auth):
        # 续期期间重新订购 / 取消订购了：按新的授权来，不拿续期结果覆盖
        logger.info("令牌续期：店铺 %s 的授权在续期期间变了，不覆盖", shop_id)
        return RefreshOutcome.SKIPPED
    rotated = fresh.refresh_token is not None
    if rotated:  # 平台换了续期令牌：连同它的有效期一起换
        refresh_token, refresh_expires_at = fresh.refresh_token, fresh.refresh_expires_at
    else:  # 没换：沿用旧的（平台单独给了新的有效期时用新的）
        refresh_token = auth.refresh_token
        refresh_expires_at = fresh.refresh_expires_at or auth.refresh_expires_at
    repo.save(
        ShopAuth(
            shop_id=shop_id,
            access_token=fresh.access_token,
            scopes=list(auth.scopes),
            expires_at=fresh.expires_at,
            refresh_token=refresh_token,
            refresh_expires_at=refresh_expires_at,
            refreshed_at=now,
            needs_reauth=False,
        )
    )
    _record(
        unit,
        shop_id,
        AuditAction.AUTH_REFRESHED,
        now,
        {
            "previous_expires_at": auth.expires_at.isoformat(),
            "expires_at": fresh.expires_at.isoformat(),
            "refresh_rotated": rotated,
        },
    )
    return RefreshOutcome.REFRESHED


def _same_grant(a: ShopAuth, b: ShopAuth) -> bool:
    """两次读到的是不是同一份授权（令牌、有效期、标记都没变）。只在内存里比较，不输出。"""
    return (
        a.access_token == b.access_token
        and a.refresh_token == b.refresh_token
        and a.expires_at == b.expires_at
        and a.needs_reauth == b.needs_reauth
    )


def _record(
    unit: Unit, shop_id: str, action: AuditAction, now: datetime, detail: dict[str, Any]
) -> None:
    unit.repos.audit.record(
        AuditEvent(
            shop_id=shop_id, actor=REFRESH_ACTOR, action=action, detail=detail, created_at=now
        )
    )


__all__ = [
    "RefreshOutcome",
    "RefreshReport",
    "RefreshUnavailable",
    "refresh_expiring_tokens",
]
