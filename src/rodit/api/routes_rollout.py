"""上线节奏与持续评估接口（方案 §13.4 影子对比与阶段门槛、§13.5 每周人工复核）。

鉴权一律用 rodit.api.deps 的 *Dep 或 shop_guard（§9.2 串店检查）；业务逻辑在 rodit.app.rollout。

| 接口 | 角色 |
|---|---|
| GET rollout/status、rollout/shadow、quality/report（只有计数、分数、推荐 ID） | 店长、商家、复核 |
| POST rollout/stage → shadow（停止展示，安全方向） | 商家、复核 |
| POST rollout/stage → assist（要过 §13.4 门槛或强制） | 复核 |
| POST rollout/incidents（登记平台违规通知、投诉，201） | 复核 |
| POST rollout/compliance-reviews（确认合规扫描的疑似违规或标记误报，201） | 复核 |
| GET rollout/incidents（登记列表：周、条数、复核人员备注） | 店长、商家、复核 |
| POST quality/sample、quality/reviews/{id} | 复核 |
| GET quality/reviews（含推荐文字，每次都留访问记录） | 复核 |

（"复核"指我们的复核人员，reviewer 角色。）

复核人、操作人一律取自调用方身份，不取请求体。影子 → 智能辅助的门槛没过、又没有强制切换时返回
409 gate_not_passed，响应里带门槛结果（哪些条件没过、哪些没法检查），店铺配置不变。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, status

from rodit.api.auth import REVIEWER_ROLES
from rodit.api.deps import (
    MSG_FORBIDDEN_ROLE,
    ReviewerDep,
    RolloutReadDep,
    ServiceDep,
    Since,
    StageDep,
)
from rodit.api.errors import ApiError
from rodit.api.schemas_rollout import (
    ComplianceReviewRequest,
    IncidentRequest,
    QualityReviewView,
    SampleRequest,
    ScoreRequest,
    ShadowView,
    StageRequest,
)
from rodit.app.rollout import (
    INCIDENT_LIST_WEEKS,
    ComplianceReviewRecord,
    IncidentList,
    IncidentRecord,
    ReviewList,
    RolloutGateNotPassed,
    StageChange,
)
from rodit.core.enums import QualityReviewStatus
from rodit.core.models import QUALITY_PERIOD_PATTERN
from rodit.feedback.review import QualityReport, SampleResult
from rodit.feedback.rollout import RolloutStage, RolloutStatus

ERROR_GATE_NOT_PASSED = "gate_not_passed"


Period = Annotated[
    str | None,
    Query(pattern=QUALITY_PERIOD_PATTERN, description="ISO 周，如 2026-W40；默认上一个完整周"),
]
ReviewId = Annotated[int, Path(ge=1)]

_GATE_409 = {
    409: {"description": "影子 → 智能辅助的门槛没过（gate_not_passed，带门槛结果）或不允许的切换"}
}

router = APIRouter()


# ---------------------------------------------------------------- 上线节奏（§13.4）


@router.get("/v1/shops/{shop_id}/rollout/status", response_model=RolloutStatus, tags=["上线节奏"])
def rollout_status(principal: RolloutReadDep, service: ServiceDep) -> RolloutStatus:
    """当前阶段、建议、影子对比汇总、各阶段门槛（影子 → 智能辅助、智能辅助 → 下一阶段、
    开放全自动的各类型），以及需要人工确认的条件。只读，没有文字。"""
    return service.rollout.status(principal.shop_id)


@router.get("/v1/shops/{shop_id}/rollout/shadow", response_model=ShadowView, tags=["上线节奏"])
def rollout_shadow(
    principal: RolloutReadDep, service: ServiceDep, since: Since = None
) -> ShadowView:
    """影子模式的推荐和客服实际回复的对比（§13.4）：逐组的推荐 ID 和字符重合度。
    since 默认最近两个完整周的开始。只读，没有文字。"""
    return ShadowView.from_comparison(service.rollout.shadow(principal.shop_id, since=since))


@router.post(
    "/v1/shops/{shop_id}/rollout/stage",
    response_model=StageChange,
    responses=_GATE_409,
    tags=["上线节奏"],
)
def rollout_stage(body: StageRequest, principal: StageDep, service: ServiceDep) -> StageChange:
    """切换影子模式 / 智能辅助（§13.4）。切到影子（不再展示推荐）商家和复核人员都可以；
    切到智能辅助只有复核人员可以，要过门槛，没过时 force=true 并写明原因才切。"""
    target = RolloutStage(body.stage)
    if target != RolloutStage.SHADOW and principal.role not in REVIEWER_ROLES:
        raise ApiError(403, "forbidden_role", MSG_FORBIDDEN_ROLE)
    try:
        return service.rollout.set_stage(
            principal.shop_id,
            target,
            operator=principal.user_id,
            force=body.force,
            note=body.note,
            role=principal.role,
        )
    except RolloutGateNotPassed as exc:
        raise ApiError(
            409, ERROR_GATE_NOT_PASSED, str(exc), extra={"gate": exc.gate.model_dump(mode="json")}
        ) from None


@router.post(
    "/v1/shops/{shop_id}/rollout/incidents",
    response_model=IncidentRecord,
    status_code=status.HTTP_201_CREATED,
    tags=["上线节奏"],
)
def rollout_incident(
    body: IncidentRequest, principal: ReviewerDep, service: ServiceDep
) -> IncidentRecord:
    """复核人员登记某一周平台下发的违规通知（violation）或买家投诉（complaint）的总数（§13.4
    "平台违规为 0""没有因此产生的投诉"）。没有也登记 0，门槛才算检查过；同一类、同一周再登记
    以最后一次为准。记进审计（shop.rollout_incident），复核人取自调用方身份。"""
    return service.rollout.record_incident(
        principal.shop_id,
        kind=body.kind,
        week=body.week,
        count=body.count,
        reviewer=principal.user_id,
        note=body.note,
    )


@router.post(
    "/v1/shops/{shop_id}/rollout/compliance-reviews",
    response_model=ComplianceReviewRecord,
    status_code=status.HTTP_201_CREATED,
    tags=["上线节奏"],
)
def rollout_compliance_review(
    body: ComplianceReviewRequest, principal: ReviewerDep, service: ServiceDep
) -> ComplianceReviewRecord:
    """复核人员确认合规扫描的一条疑似违规（confirmed：确实违规，§13.4 门槛不通过）或标记误报
    （dismissed：不计入）。扫描命中只是候选：没确认的在上线状态里列为 unchecked，不算不通过。
    ID 取自上线状态里合规扫描列出的推荐 ID / 消息 ID；不是最近扫描出来的疑似违规时 404。
    同一条再标记以最后一次为准。记进审计（compliance.reviewed，不记文字），复核人取自调用方身份。"""
    return service.rollout.review_compliance(
        principal.shop_id,
        decision=body.decision,
        reviewer=principal.user_id,
        suggestion_id=body.suggestion_id,
        msg_id=body.msg_id,
        note=body.note,
    )


@router.get("/v1/shops/{shop_id}/rollout/incidents", response_model=IncidentList, tags=["上线节奏"])
def rollout_incidents(
    principal: RolloutReadDep,
    service: ServiceDep,
    weeks: Annotated[
        int, Query(ge=0, le=520, description="本周和之前多少周（默认 8）")
    ] = INCIDENT_LIST_WEEKS,
) -> IncidentList:
    """平台违规、投诉的登记（新的周在前）：周、条数、复核人员的备注；被后来的登记覆盖的
    current=false。只读。"""
    return service.rollout.incidents(principal.shop_id, weeks=weeks)


# ---------------------------------------------------------------- 每周人工复核（§13.5）


@router.post("/v1/shops/{shop_id}/quality/sample", response_model=SampleResult, tags=["每周复核"])
def quality_sample(
    body: SampleRequest, principal: ReviewerDep, service: ServiceDep
) -> SampleResult:
    """抽一个周期的复核样本（默认上一个完整周、50 条，按问题类型分层）。重复调用只补不换。"""
    return service.rollout.sample(
        principal.shop_id, reviewer=principal.user_id, period=body.period, size=body.size
    )


@router.get("/v1/shops/{shop_id}/quality/reviews", response_model=ReviewList, tags=["每周复核"])
def quality_reviews(
    principal: ReviewerDep,
    service: ServiceDep,
    period: Period = None,
    status: Annotated[QualityReviewStatus | None, Query()] = None,
) -> ReviewList:
    """复核列表：样本 + 推荐文字、买家问题、编辑后发出的文字、依据 ID（都已做个人信息最小化）。
    只给复核人员；每次返回都记访问记录（quality.viewed，§9.2）。"""
    return service.rollout.list_reviews(
        principal.shop_id, viewer=principal.user_id, period=period, status=status
    )


@router.post(
    "/v1/shops/{shop_id}/quality/reviews/{review_id}",
    response_model=QualityReviewView,
    tags=["每周复核"],
)
def quality_score(
    review_id: ReviewId, body: ScoreRequest, principal: ReviewerDep, service: ServiceDep
) -> QualityReviewView:
    """记下复核结果（§13.2 事实正确、切题、合规、表达各 0–2 分，或跳过）。可以改判。"""
    review = service.rollout.score(
        principal.shop_id,
        review_id,
        status=body.status,
        reviewer=principal.user_id,
        scores=body.scores,
        note=body.note,
    )
    return QualityReviewView.from_review(review)


@router.get("/v1/shops/{shop_id}/quality/report", response_model=QualityReport, tags=["每周复核"])
def quality_report(
    principal: RolloutReadDep, service: ServiceDep, period: Period = None
) -> QualityReport:
    """一个周期的复核报表：抽样 / 已打分 / 跳过 / 待复核条数，各维度满分率和平均分，
    §13.2 上线门槛（事实正确、合规满分率 ≥ 95%），按问题类型拆分。没有文字。"""
    return service.rollout.report(principal.shop_id, period=period)


__all__ = ["ERROR_GATE_NOT_PASSED", "router"]
