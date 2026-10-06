"""售后审核工作台接口（方案 §7.3.2 商家侧审核助手、§15 Phase 2、§2.1 售后平均审核时长）。

单张售后单的规则建议、协商、确认执行在 rodit.api.app
（/v1/shops/{shop_id}/aftersales/{aftersale_id}/…）；这里是客服的工作台：

| 接口 | 说明 |
|---|---|
| GET aftersales | 待审核列表：平台待审核 + 本地还没结束的审核记录，要处理的在前、再按申请时间 |
| POST aftersales/review-pending | 批量拉取并跑规则（逐张走 review_aftersale），单张失败记在那一行 |
| GET aftersales/stats | 各状态单数、各规则的待处理单数、平均审核时长（§2.1）、已等待时长 |
| GET aftersales/{aftersale_id} | 详情：审核记录 + 平台当前状态 + 能做哪些操作、为什么不能做 |

角色和已有的售后审核接口一样（客服、店长、商家：AgentDep）。鉴权顺序见 rodit.api.deps；之后先查店铺
授权（§8.4 授权过期时停止处理，403 shop_not_authorized），一次平台接口都不调。
GET 不写任何东西（不建审核记录、不写审计）。业务逻辑在 rodit.app.aftersale_desk。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Query

from rodit.api.deps import AfterSaleId, AgentDep, ServiceDep, Since
from rodit.api.schemas_aftersale import BatchReviewView, DeskDetailView, ReviewPendingRequest
from rodit.app.aftersale_desk import (
    DEFAULT_QUEUE_LIMIT,
    MAX_QUEUE_LIMIT,
    QUEUE_FILTERS,
    AfterSaleDesk,
    AfterSaleQueue,
    DeskStats,
)
from rodit.app.service import AssistService
from rodit.core.enums import AfterSaleStatus

TAGS = ["售后审核"]

StateFilter = Annotated[
    str | None,
    Query(
        pattern="^(" + "|".join(QUEUE_FILTERS) + ")$",
        description="unreviewed（未跑规则）或审核状态；不给时列出还没结束的",
    ),
]
RuleFilter = Annotated[str | None, Query(pattern="^R[1-8]$", description="命中的规则，如 R4")]
Limit = Annotated[int, Query(ge=1, le=MAX_QUEUE_LIMIT, description="最多返回几行")]

router = APIRouter()


def _desk(service: AssistService) -> AfterSaleDesk:
    return AfterSaleDesk(service.unit, service.review_aftersale)


@router.get("/v1/shops/{shop_id}/aftersales", response_model=AfterSaleQueue, tags=TAGS)
def aftersale_queue(
    principal: AgentDep,
    service: ServiceDep,
    state: StateFilter = None,
    status: Annotated[AfterSaleStatus | None, Query(description="平台状态")] = None,
    rule: RuleFilter = None,
    limit: Limit = DEFAULT_QUEUE_LIMIT,
) -> AfterSaleQueue:
    """待审核列表（只读）：平台待审核售后单 + 本地还没结束的审核记录（签收核查中、等待凭证、
    待客服确认、协商中、退货中）。要处理的（未跑规则、待客服确认、需要刷新）在前，再按申请时间
    从早到晚。
    平台列表读不到时只列本地审核记录（platform_unavailable=true，§12.2）。"""
    return _desk(service).queue(
        principal.shop_id, state=state, status=status, rule=rule, limit=limit
    )


@router.post(
    "/v1/shops/{shop_id}/aftersales/review-pending", response_model=BatchReviewView, tags=TAGS
)
def aftersale_review_pending(
    principal: AgentDep,
    service: ServiceDep,
    body: Annotated[ReviewPendingRequest | None, Body()] = None,
) -> BatchReviewView:
    """批量拉取并跑规则（§7.3.2）：平台上待审核、还没有建议的售后单，按申请时间从早到晚
    最多 limit 张。每张和单张接口一样（店铺闸门、授权、审计、状态机）；单张失败（平台接口故障、
    并发冲突……）记在那一行，不是整个请求失败。平台待审核列表读不到时 503。只读平台，不调写接口。"""
    limit = (body or ReviewPendingRequest()).limit
    result = _desk(service).review_pending(principal.shop_id, limit=limit)
    return BatchReviewView.from_result(result)


@router.get("/v1/shops/{shop_id}/aftersales/stats", response_model=DeskStats, tags=TAGS)
def aftersale_stats(principal: AgentDep, service: ServiceDep, since: Since = None) -> DeskStats:
    """工作台统计（只读）：各审核状态的单数（还没结束的是当前工作量；已结束的只数 since 之后的）、
    各规则的待处理单数、since 之后由人确认执行的平均审核时长（申请 → 确认执行，
    抖店"售后平均审核时长"的近似，§2.1）、还没处理完的已等待时长。"""
    return _desk(service).stats(principal.shop_id, since=since)


@router.get(
    "/v1/shops/{shop_id}/aftersales/{aftersale_id}", response_model=DeskDetailView, tags=TAGS
)
def aftersale_detail(
    aftersale_id: AfterSaleId, principal: AgentDep, service: ServiceDep
) -> DeskDetailView:
    """一张售后单的详情（只读）：审核记录（没跑规则时为空）、平台当前状态（重读，不走缓存；确认时带上
    platform.status_version）、能做哪些操作和不能做的原因（平台差异、写权限、R6 店长确认）。"""
    detail = _desk(service).detail(
        principal.shop_id,
        aftersale_id,
        operator=principal.user_id,
        is_manager=principal.is_manager,
    )
    return DeskDetailView.from_detail(detail)


__all__ = ["router"]
