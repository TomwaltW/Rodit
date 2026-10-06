"""平台客服指标的应用服务（方案 §1.3 "平台客服指标较接入前改善"、§12.4 "平台客服指标接入前后对比"、
§15 Phase 2 "商家报表（平台指标接入前后对比）"）。口径和对比算法在 rodit.feedback.platform_metrics。

- 同步（sync）：从平台云内的只读接口（Unit.metrics_source，core.ports.PlatformMetricsSource）拉
  最近 days 天（到昨天为止：平台按天出数，当天的还不全）的数据，按 source=platform 写进仓储。
  调接口前先查店铺授权（§8.4，没有授权 / 过期 / 缺权限抛 AuthExpired，接口一次都不调）；
  整段拉完、逐条核对之后才写：平台接口故障（ToolUnavailable）时什么都不写；返回了别的店铺的
  数据按串店处理（TenantIsolationError，什么都不写，审计 tenant.violation，§9.2）；不是本平台
  考核的指标、日期不在范围内、来源不是平台、数值或样本量不合法、同一指标同一天重复的逐条跳过并
  计数（SKIP_LABELS）。写入是按（店铺、指标、日期、来源）覆盖，重复同步结果一样；仓储没有批量
  写入，写到一半数据库出错时已写的留着、异常照抛，重新同步一次即可补齐（同步审计只在全部写完后记）。
- 商家登记（record）：商家从平台后台抄录的数（source=merchant，登记人取调用方身份）：指标必须是
  本平台考核的，比例在 0–1、时长有合理上限，日期不晚于今天、最多回溯 400 天；和已登记的一样时
  不写、不记审计。
- 列表（list_samples）和报表（report）只读，不写审计。报表的接入日期：填写的 > 第一次切到智能辅助
  （上线记录 shop.rollout_stage）> 这次开通授权的时间（最早的开通记录 auth.granted，优先用它记下的
  平台事件时间；续订也写 auth.granted 但不会更早，取消订购时和店铺的审计一起删掉）> 没有开通记录时
  用 ShopAuth.granted_at（按 stamp 读，不解密令牌；每次续订都会改写它，只作兜底）> 未知（报表请
  商家填写）。没开影子模式的店铺没有上线记录，多半落到开通授权这一档。
- 写操作都在 processing(shop_id) 里做（和取消订购互斥，§8.4），进了闸门先确认店铺还在、没停用
  （删完的店铺不会被写回来）；写完记审计（AuditAction.PLATFORM_METRICS_SYNCED /
  PLATFORM_METRICS_RECORDED：指标、日期、条数、数值，没有买家数据）。
- 系统估算（MetricSource.COMPUTED）不在这里产生：现有端口算不出和平台口径一致、接入前后可比的数。
  售后平均审核时长要"买家申请 → 商家第一次处理"的时间，平台只读接口只有售后单的创建时间和最后
  更新时间（之后的退款、退货都会刷新它），商家处理的时间只在我们的审核助手执行时留审计（接入后
  才有）——接入前用最后更新时间、接入后用审计，口径不同，对比会偏向接入后；响应率、响应时长要
  接入前的会话记录，接入前我们没有。报表照样认 computed 的数据（优先级最低，并提示口径近似）。

出错时抛业务异常，由接口层映射：PlatformMetricsUnavailable（单元没有指标仓储）、MetricsSourceMissing
（没接平台指标接口）、ToolUnavailable（平台接口故障）、ValueError（取值不合法）、ShopInactive、
AuthExpired、TenantIsolationError。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import date, timedelta
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from rodit.app.unit import Unit
from rodit.core.audit import AuditAction
from rodit.core.clock import CST
from rodit.core.enums import MetricSource, PlatformMetric
from rodit.core.errors import RoditError, TenantIsolationError
from rodit.core.models import AuditEvent, PlatformMetricSample, strip_control_chars
from rodit.core.platform_metrics import metrics_for
from rodit.core.ports import PlatformMetricsRepository, PlatformMetricsSource
from rodit.core.scopes import MVP_SCOPES
from rodit.feedback.platform_metrics import (
    AUTH_GRANTED_ACTION,
    DEFAULT_WINDOW_DAYS,
    MAX_HISTORY_DAYS,
    ROLLOUT_STAGE_ACTION,
    GoLive,
    PlatformMetricsReport,
    build_report,
    check_day,
    check_metric,
    check_sample_size,
    check_value,
    check_window_days,
    resolve_go_live,
)
from rodit.knowledge.admin import require_human, require_shop

logger = logging.getLogger(__name__)

ShopGate = Callable[[str], AbstractContextManager[None]]
"""AssistService.processing：按店铺的读方闸门（和取消订购互斥）。"""

PLATFORM_METRICS_SYNCED = AuditAction.PLATFORM_METRICS_SYNCED.value
"""同 AuditAction.PLATFORM_METRICS_SYNCED（保留旧名字，新代码直接用 AuditAction）：从平台接口同步了
一段日期的指标（范围、拉到 / 写入 / 跳过的条数、指标）。"""
PLATFORM_METRICS_RECORDED = AuditAction.PLATFORM_METRICS_RECORDED.value
"""同 AuditAction.PLATFORM_METRICS_RECORDED（保留旧名字，新代码直接用 AuditAction）：商家登记了一天
的一项指标（指标、日期、数值、样本量、是否覆盖了之前登记的）。"""
TENANT_STAGE_SYNC = "platform_metrics_sync"
"""tenant.violation 的 stage：平台指标接口返回了别的店铺的数据。"""

DEFAULT_SYNC_DAYS = 2 * DEFAULT_WINDOW_DAYS
"""默认同步最近多少天：够默认报表的接入前、接入后各 28 天。"""
SYNC_ACTORS = frozenset({"system", "ops"})
"""同步的操作人可以是系统 / 运维任务（读平台数据，不是商家的决定）；别的必须是具体的人。"""

SKIP_OTHER_PLATFORM = "other_platform"
SKIP_OUT_OF_RANGE = "out_of_range"
SKIP_NOT_PLATFORM = "not_platform_source"
SKIP_INVALID = "invalid_value"
SKIP_DUPLICATE = "duplicate"
SKIP_LABELS: dict[str, str] = {
    SKIP_OTHER_PLATFORM: "不是本平台考核的指标",
    SKIP_OUT_OF_RANGE: "日期不在同步范围内",
    SKIP_NOT_PLATFORM: "来源不是平台",
    SKIP_INVALID: "数值或样本量不合法",
    SKIP_DUPLICATE: "同一指标同一天重复",
}

MSG_NO_REPO = "本单元没有平台指标存储，商家报表没有接入前后对比（§12.4）"
MSG_NO_SOURCE = "本单元没有接平台指标接口：可以在报表页手工登记平台后台的数据"
MSG_TENANT = "平台指标接口返回了别的店铺的数据，已拒绝写入（§9.2）"


class PlatformMetricsUnavailable(RoditError):
    """单元没有平台指标仓储（Repos.platform_metrics 为空）。"""

    def __init__(self) -> None:
        super().__init__(MSG_NO_REPO)


class MetricsSourceMissing(RoditError):
    """单元没有接平台指标接口（Unit.metrics_source 为空）：只能商家登记。"""

    def __init__(self) -> None:
        super().__init__(MSG_NO_SOURCE)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SyncResult(_Model):
    """一次同步的结果（只有条数和指标）。"""

    shop_id: str
    since: date
    until: date
    fetched: int  # 平台接口返回的条数
    written: int  # 写进仓储的条数（覆盖也算）
    skipped: dict[str, int] = Field(default_factory=dict)  # 跳过的原因 → 条数（SKIP_LABELS）
    metrics: list[PlatformMetric] = Field(default_factory=list)  # 写入了哪些指标
    days: int = 0  # 写入涉及的天数
    synced_at: AwareDatetime


class RecordResult(_Model):
    """商家登记的结果。changed=False：和已登记的一样，没有写。"""

    sample: PlatformMetricSample
    replaced: bool  # 覆盖了这一天之前登记的数
    changed: bool


class SampleList(_Model):
    """原始数据（每个来源各一条，不做取舍）。"""

    shop_id: str
    since: date
    until: date
    samples: list[PlatformMetricSample]


class PlatformMetricsService:
    """平台客服指标的应用服务。读操作直接读；写操作一律在 processing(shop_id) 里做。"""

    def __init__(self, unit: Unit, processing: ShopGate) -> None:
        self.unit = unit
        self._processing = processing

    # ================================================================ 通用

    def _repo(self) -> PlatformMetricsRepository:
        repo = self.unit.repos.platform_metrics
        if repo is None:
            raise PlatformMetricsUnavailable()
        return repo

    def today(self) -> date:
        """东八区的今天（平台按自然日出数）。"""
        return self.unit.clock.now().astimezone(CST).date()

    def _audit(self, shop_id: str, actor: str, action: str, detail: dict[str, Any]) -> None:
        self.unit.repos.audit.record(
            AuditEvent(
                shop_id=shop_id,
                actor=actor,
                action=action,
                detail=detail,
                created_at=self.unit.clock.now(),
            )
        )

    # ================================================================ 同步（平台接口）

    def sync(self, shop_id: str, *, actor: str, days: int = DEFAULT_SYNC_DAYS) -> SyncResult:
        """拉最近 days 天（1–400，到昨天为止）的平台数据写进仓储，见模块说明。"""
        who = _sync_actor(actor)
        if isinstance(days, bool) or not isinstance(days, int):
            raise ValueError("同步天数必须是整数")
        if not 1 <= days <= MAX_HISTORY_DAYS:
            raise ValueError(f"同步天数要在 1–{MAX_HISTORY_DAYS} 天之间")
        repo = self._repo()
        source = self.unit.metrics_source
        if source is None:
            raise MetricsSourceMissing()
        today = self.today()
        since, until = today - timedelta(days=days), today - timedelta(days=1)
        with self._processing(shop_id):
            require_shop(self.unit.repos.knowledge, shop_id)
            self.unit.auth.require(shop_id, MVP_SCOPES)  # §8.4 授权无效时一次都不调平台接口
            fetched = source.fetch(shop_id, since, until)  # 故障抛 ToolUnavailable：什么都不写
            if any(s.shop_id != shop_id for s in fetched):
                self._audit(
                    shop_id,
                    "system",
                    AuditAction.TENANT_VIOLATION.value,
                    {"stage": TENANT_STAGE_SYNC},  # 只记在哪一步拦下的，不记任何数据
                )
                raise TenantIsolationError(MSG_TENANT)
            keep, skipped = self._check_fetched(source, fetched, since=since, until=until)
            now = self.unit.clock.now()
            for s in keep:
                repo.upsert(
                    PlatformMetricSample(
                        shop_id=shop_id,
                        metric=s.metric,
                        day=s.day,
                        value=s.value,
                        source=MetricSource.PLATFORM,
                        sample_size=s.sample_size,
                        recorded_by=None,
                        recorded_at=now,
                    )
                )
            written = {s.metric for s in keep}
            metrics = [m for m in metrics_for(self.unit.platform) if m in written]
            result = SyncResult(
                shop_id=shop_id,
                since=since,
                until=until,
                fetched=len(fetched),
                written=len(keep),
                skipped=skipped,
                metrics=metrics,
                days=len({s.day for s in keep}),
                synced_at=now,
            )
            self._audit(
                shop_id,
                who,
                AuditAction.PLATFORM_METRICS_SYNCED.value,
                {
                    "since": since.isoformat(),
                    "until": until.isoformat(),
                    "fetched": result.fetched,
                    "written": result.written,
                    "skipped": dict(skipped),
                    "metrics": [m.value for m in metrics],
                },
            )
            return result

    def _check_fetched(
        self,
        source: PlatformMetricsSource,
        fetched: list[PlatformMetricSample],
        *,
        since: date,
        until: date,
    ) -> tuple[list[PlatformMetricSample], dict[str, int]]:
        """逐条核对平台接口返回的数据：(要写的, 跳过的原因 → 条数)。"""
        allowed = set(metrics_for(self.unit.platform))
        keep: list[PlatformMetricSample] = []
        seen: set[tuple[PlatformMetric, date]] = set()
        skipped: dict[str, int] = {}

        def skip(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        for s in fetched:
            if s.metric not in allowed:
                skip(SKIP_OTHER_PLATFORM)
            elif not since <= s.day <= until:
                skip(SKIP_OUT_OF_RANGE)
            elif s.source != MetricSource.PLATFORM:
                skip(SKIP_NOT_PLATFORM)
            elif not _valid(s):
                skip(SKIP_INVALID)
            elif (s.metric, s.day) in seen:
                skip(SKIP_DUPLICATE)
            else:
                seen.add((s.metric, s.day))
                keep.append(s)
        if skipped:
            logger.warning(
                "平台指标接口（%s）返回的数据有 %d 条没有写入：%s",
                type(source).__name__,
                sum(skipped.values()),
                skipped,
            )
        return keep, {k: skipped[k] for k in SKIP_LABELS if k in skipped}

    # ================================================================ 商家登记

    def record(
        self,
        shop_id: str,
        *,
        metric: PlatformMetric | str,
        day: date,
        value: float,
        sample_size: int | None = None,
        merchant: str,
    ) -> RecordResult:
        """商家登记一天的一项指标（source=merchant），见模块说明。"""
        actor = require_human(_clean(merchant), "商家")
        m = check_metric(self.unit.platform, metric)
        number = check_value(m, value)
        size = check_sample_size(sample_size)
        check_day(day, self.today(), "登记的日期")
        repo = self._repo()
        with self._processing(shop_id):
            require_shop(self.unit.repos.knowledge, shop_id)
            existing = [
                s
                for s in repo.list(shop_id, metric=m, since=day, until=day)
                if s.source == MetricSource.MERCHANT and s.shop_id == shop_id
            ]
            previous = existing[0] if existing else None
            if previous is not None and previous.value == number and previous.sample_size == size:
                return RecordResult(sample=previous, replaced=False, changed=False)
            saved = repo.upsert(
                PlatformMetricSample(
                    shop_id=shop_id,
                    metric=m,
                    day=day,
                    value=number,
                    source=MetricSource.MERCHANT,
                    sample_size=size,
                    recorded_by=actor,
                    recorded_at=self.unit.clock.now(),
                )
            )
            self._audit(
                shop_id,
                actor,
                AuditAction.PLATFORM_METRICS_RECORDED.value,
                {
                    "metric": m.value,
                    "day": day.isoformat(),
                    "value": number,
                    "sample_size": size,
                    "replaced": previous is not None,
                    "previous_value": previous.value if previous is not None else None,
                },
            )
            return RecordResult(sample=saved, replaced=previous is not None, changed=True)

    # ================================================================ 查看

    def list_samples(
        self,
        shop_id: str,
        *,
        since: date | None = None,
        until: date | None = None,
        metric: PlatformMetric | str | None = None,
    ) -> SampleList:
        """原始数据（每个来源各一条）。默认最近 56 天（到今天）；范围最多 400 天。"""
        today = self.today()
        until = until or today
        since = since or until - timedelta(days=DEFAULT_SYNC_DAYS - 1)
        if since > until:
            raise ValueError("起始日期不能晚于结束日期")
        if (until - since).days + 1 > MAX_HISTORY_DAYS:
            raise ValueError(f"一次最多查 {MAX_HISTORY_DAYS} 天")
        m = check_metric(self.unit.platform, metric) if metric is not None else None
        rows = self._repo().list(shop_id, metric=m, since=since, until=until)
        return SampleList(
            shop_id=shop_id,
            since=since,
            until=until,
            samples=[s for s in rows if s.shop_id == shop_id],
        )

    def go_live(self, shop_id: str, explicit: date | None = None) -> GoLive:
        """报表用的接入日期（见模块说明）。填写的日期不能晚于今天、最多回溯 400 天。"""
        if explicit is not None:
            check_day(explicit, self.today(), "接入日期")
            return resolve_go_live(explicit)
        audit = self.unit.repos.audit
        events = audit.list_events(shop_id, action=ROLLOUT_STAGE_ACTION)
        grants = audit.list_events(shop_id, action=AUTH_GRANTED_ACTION)
        stamp = self.unit.repos.auth.stamp(shop_id)
        return resolve_go_live(
            None,
            stage_events=[e for e in events if e.shop_id == shop_id],
            grant_events=[e for e in grants if e.shop_id == shop_id],
            granted_at=stamp.granted_at if stamp is not None else None,
        )

    def report(
        self,
        shop_id: str,
        *,
        go_live: date | None = None,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> PlatformMetricsReport:
        """接入前后对比报表（§12.4）。只读。"""
        window_days = check_window_days(window_days)
        repo = self._repo()
        today = self.today()
        resolved = self.go_live(shop_id, go_live)
        samples: list[PlatformMetricSample] = []
        if resolved.day is not None:
            samples = repo.list(
                shop_id,
                since=resolved.day - timedelta(days=window_days),
                until=min(resolved.day + timedelta(days=window_days - 1), today),
            )
        return build_report(
            shop_id=shop_id,
            platform=self.unit.platform,
            samples=samples,
            go_live=resolved,
            window_days=window_days,
            today=today,
        )


def _sync_actor(actor: object) -> str:
    """同步的操作人：系统 / 运维任务，或具体的人（店长、商家）。"""
    if isinstance(actor, str) and actor.strip().lower() in SYNC_ACTORS:
        return actor.strip().lower()
    return require_human(_clean(actor), "操作人")


def _clean(value: object) -> object:
    """账号名去掉 NUL 等控制字符（写进审计和 recorded_by；PostgreSQL 的 TEXT 不接受 NUL）。"""
    return strip_control_chars(value) if isinstance(value, str) else value


def _valid(s: PlatformMetricSample) -> bool:
    try:
        check_value(s.metric, s.value)
        check_sample_size(s.sample_size)
    except ValueError:
        return False
    return True


__all__ = [
    "DEFAULT_SYNC_DAYS",
    "PLATFORM_METRICS_RECORDED",
    "PLATFORM_METRICS_SYNCED",
    "SKIP_LABELS",
    "TENANT_STAGE_SYNC",
    "MetricsSourceMissing",
    "PlatformMetricsService",
    "PlatformMetricsUnavailable",
    "RecordResult",
    "SampleList",
    "SyncResult",
]
