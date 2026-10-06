"""上线节奏与每周复核接口的请求和响应模型（方案 §13.4 影子对比与阶段门槛、§13.5 每周人工复核、
§13.2 四个维度打分）。

请求模型一律 extra="forbid"：复核人、操作人只取自调用方身份（Principal），请求体里带 reviewer /
operator / shop_id 等字段直接 422（§4.1 原则 3）。

响应里的上线状态、影子对比、复核报表、平台违规 / 投诉登记只有计数、分数和推荐 / 消息 ID
（登记另有复核人员的备注），没有文字；只有复核列表（复核人员专用，每次返回都留访问记录，§9.2）
带推荐文字和买家问题（已做个人信息最小化）。影子对比明细不返回平台消息 ID；上线状态里合规扫描
列出的消息 ID 是客服 / 机器人发出的消息的 ID（给复核人员核对），不是买家消息的。
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from rodit.app.rollout import MAX_NOTE_CHARS, ShadowComparison
from rodit.core.enums import Intent, QualityReviewStatus
from rodit.core.models import QUALITY_PERIOD_PATTERN, QualityReview, QualityScores
from rodit.feedback.compliance import MAX_MSG_ID_CHARS
from rodit.feedback.metrics import ShadowIntentStats
from rodit.feedback.review import DEFAULT_SAMPLE_SIZE, MAX_SAMPLE_SIZE
from rodit.feedback.rollout import MAX_INCIDENT_COUNT


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ================================================================ 请求


class StageRequest(_Request):
    """切换上线阶段（§13.4）。切到影子随时可以；影子 → 智能辅助要过门槛，没过时 force=true
    强制切换，且必须写明原因（note）。操作人取自调用方身份。"""

    stage: Literal["shadow", "assist"]
    force: bool = False
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class SemiAutoRequest(_Request):
    """开 / 关某个问题类型的半自动（§5.6、§13.4）：enabled=true 只有复核人员能用，要过门槛（该类型
    原样采纳率 ≥ 90%），没过时 force=true 强制开启，且必须写明原因（note）；高风险类型永远不能开。
    enabled=false 商家和复核人员都能用。操作人取自调用方身份。"""

    intent: Intent
    enabled: bool = Field(strict=True)
    force: bool = False
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class SampleRequest(_Request):
    """每周抽样（§13.5）。period 默认上一个完整的 ISO 周（东八区），size 默认 50。"""

    period: str | None = Field(default=None, pattern=QUALITY_PERIOD_PATTERN)
    size: int = Field(default=DEFAULT_SAMPLE_SIZE, ge=1, le=MAX_SAMPLE_SIZE)


class IncidentRequest(_Request):
    """复核人员登记某一周平台下发的违规通知（violation）或买家投诉（complaint）的总数（§13.4）。
    没有也登记 0，门槛才算检查过；同一类、同一周再登记以最后一次为准。备注会做个人信息最小化，
    不要贴通知或投诉的原文。复核人取自调用方身份。"""

    kind: Literal["violation", "complaint"]
    week: str = Field(pattern=QUALITY_PERIOD_PATTERN)
    count: int = Field(ge=0, le=MAX_INCIDENT_COUNT, strict=True)
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class ComplianceReviewRequest(_Request):
    """复核人员确认合规扫描的一条疑似违规（confirmed：确实违规，门槛不通过）或标记误报
    （dismissed：不计入，如依据里原样有的"桑蚕丝 100%"）。suggestion_id（编辑后发送的推荐）和
    msg_id（客服 / 机器人发出的消息）给且只给一个——上线状态里合规扫描列出的 ID。备注会做个人信息
    最小化，不要贴原文。复核人取自调用方身份。"""

    suggestion_id: int | None = Field(default=None, ge=1, strict=True)
    msg_id: str | None = Field(default=None, min_length=1, max_length=MAX_MSG_ID_CHARS)
    decision: Literal["confirmed", "dismissed"]
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)

    @model_validator(mode="after")
    def _one_ref(self) -> ComplianceReviewRequest:
        if (self.suggestion_id is None) == (self.msg_id is None):
            raise ValueError("suggestion_id 和 msg_id 要给且只给一个")
        return self


class ScoreRequest(_Request):
    """复核结果（§13.2）：scored 带四个维度的分数（0–2），skipped 不带分数。备注会做个人信息
    最小化后保存，审计里不记备注原文。复核人取自调用方身份。"""

    status: Literal["scored", "skipped"]
    scores: QualityScores | None = None
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)

    @model_validator(mode="after")
    def _scores_match_status(self) -> ScoreRequest:
        if self.status == "scored" and self.scores is None:
            raise ValueError("打分（scored）要给出四个维度的分数")
        if self.status == "skipped" and self.scores is not None:
            raise ValueError("跳过（skipped）不能带分数")
        return self


# ================================================================ 响应


class ShadowItemView(_View):
    """一个影子推荐组的对比（组内最像客服实际回复的一条）。不含文字、不含平台消息 ID。"""

    suggestion_id: int | None = None
    intent: Intent
    similarity: float | None = None  # 客服没有回复时为 None


class ShadowView(_View):
    """影子对比明细（§13.4）：字符重合度只是参考，帮复核人员挑差异大的样本。"""

    since: AwareDatetime
    until: AwareDatetime
    groups: int
    compared: int
    no_reply: int
    match_threshold: float
    mean_similarity: float | None = None
    median_similarity: float | None = None
    match_rate: float | None = None
    by_intent: dict[Intent, ShadowIntentStats] = Field(default_factory=dict)
    items: list[ShadowItemView] = Field(default_factory=list)

    @classmethod
    def from_comparison(cls, comparison: ShadowComparison) -> ShadowView:
        r = comparison.report
        return cls(
            since=comparison.since,
            until=comparison.until,
            groups=r.groups,
            compared=r.compared,
            no_reply=r.no_reply,
            match_threshold=r.match_threshold,
            mean_similarity=r.mean_similarity,
            median_similarity=r.median_similarity,
            match_rate=r.match_rate,
            by_intent=r.by_intent,
            items=[
                ShadowItemView(
                    suggestion_id=i.suggestion_id, intent=i.intent, similarity=i.similarity
                )
                for i in r.items
            ],
        )


class QualityReviewView(_View):
    """复核样本（打分后返回）。不含推荐文字。"""

    id: int
    period: str
    suggestion_id: int
    intent: Intent
    status: QualityReviewStatus
    scores: QualityScores | None = None
    reviewer: str | None = None
    note: str | None = None
    sampled_at: AwareDatetime
    reviewed_at: AwareDatetime | None = None

    @classmethod
    def from_review(cls, review: QualityReview) -> QualityReviewView:
        assert review.id is not None
        return cls(
            id=review.id,
            period=review.period,
            suggestion_id=review.suggestion_id,
            intent=review.intent,
            status=review.status,
            scores=review.scores,
            reviewer=review.reviewer,
            note=review.note,
            sampled_at=review.sampled_at,
            reviewed_at=review.reviewed_at,
        )


__all__ = [
    "ComplianceReviewRequest",
    "IncidentRequest",
    "QualityReviewView",
    "SampleRequest",
    "ScoreRequest",
    "SemiAutoRequest",
    "ShadowItemView",
    "ShadowView",
    "StageRequest",
]
