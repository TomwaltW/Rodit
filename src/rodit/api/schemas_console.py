"""商家控制台接口的请求和响应模型（方案 §5.2 知识中心、§5.2.2 商品卡片、§5.2.3 商品同步、
§9.3 店铺配置）。

请求模型一律 extra="forbid"：操作人（merchant / approved_by / operator）、shop_id 都取自调用方身份
和路径，请求体里带这些字段直接 422（§4.1 原则 3）。商品卡片只接受商家填写的字段，平台字段
（标题、属性、SKU……）也是 422。

修改类请求（FAQ 修改、审核时修改、商品卡片）只改请求体里出现的字段；出现且为 null 表示清空
（item_id / intent / size_advice / 列表字段），没出现的保持原样。
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from rodit.api.schemas import KnowledgeView
from rodit.core.enums import Intent
from rodit.core.models import ItemCard, ShopConfig
from rodit.knowledge.admin import (
    MAX_ANSWER_CHARS,
    MAX_CLAIM_CHARS,
    MAX_ITEM_LINE_CHARS,
    MAX_POLICY_CHARS,
    MAX_QUESTION_CHARS,
    MAX_SIZE_ADVICE_CHARS,
    MERCHANT_ITEM_FIELDS,
    FaqChange,
    FaqChanges,
    FaqWarning,
    PolicyRow,
    ShopConfigSnapshot,
)
from rodit.knowledge.sync import SyncReport

MAX_ID_CHARS = 128
MAX_LIST_ITEMS = 30
MAX_CONFIG_KEYS = 20


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid")


Id = Annotated[str, Field(min_length=1, max_length=MAX_ID_CHARS)]
Question = Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]
Answer = Annotated[str, Field(min_length=1, max_length=MAX_ANSWER_CHARS)]


# ================================================================ FAQ（§5.2.1 优先级 4）


class FaqCreateRequest(_Request):
    """商家录入 FAQ：直接上线。item_id 为空是通用 FAQ，否则只在问到这个商品时用。"""

    question: Question
    answer: Answer
    item_id: Id | None = None
    intent: Intent | None = None


class FaqEditRequest(_Request):
    """修改 FAQ / 审核候选知识时修改：只改出现的字段（至少一个）。"""

    question: Question | None = None
    answer: Answer | None = None
    item_id: Id | None = None
    intent: Intent | None = None

    @model_validator(mode="after")
    def _something(self) -> FaqEditRequest:
        if not self.model_fields_set:
            raise ValueError("至少要改一个字段（question、answer、item_id、intent）")
        return self

    def changes(self) -> FaqChanges:
        return FaqChanges.of(**{f: getattr(self, f) for f in self.model_fields_set})


class FaqApproveEditedRequest(FaqEditRequest):
    """审核候选知识时修改（§5.8）：只改出现的字段。所有字段都和候选相同时就是普通的审核通过。"""


class FaqWarningView(_View):
    """保存时的提醒（不拦截保存）：blocks_answer 为真表示回答时这条 FAQ 会被拦截或被政策覆盖。"""

    rule: str
    detail: str
    blocks_answer: bool

    @classmethod
    def of(cls, warning: FaqWarning) -> FaqWarningView:
        return cls(**warning.model_dump())


class FaqChangeView(_View):
    """FAQ 写操作的结果：上线的条目、下线的旧条目（修改时，旧 id 不再用于回答）、提醒。"""

    knowledge: KnowledgeView
    retired: KnowledgeView | None = None
    warnings: list[FaqWarningView] = []

    @classmethod
    def of(cls, change: FaqChange) -> FaqChangeView:
        return cls(
            knowledge=KnowledgeView.from_item(change.item),
            retired=KnowledgeView.from_item(change.retired) if change.retired else None,
            warnings=[FaqWarningView.of(w) for w in change.warnings],
        )


# ================================================================ 店铺政策（§5.2.1 优先级 2）


class PolicyUpdateRequest(_Request):
    """改一项店铺政策。改动即时生效。"""

    value: str = Field(min_length=1, max_length=MAX_POLICY_CHARS)


class PolicyView(_View):
    """一项店铺政策。value 为 null 表示还没填（questionnaire_id 是附录 C 问卷的题号）。"""

    key: str
    label: str
    value: str | None = None
    updated_at: AwareDatetime | None = None
    questionnaire_id: int | None = None
    known: bool = True

    @classmethod
    def of(cls, row: PolicyRow) -> PolicyView:
        return cls(**row.model_dump())


# ================================================================ 商品卡片（§5.2.2）

ItemLine = Annotated[str, Field(max_length=MAX_ITEM_LINE_CHARS)]
ClaimLine = Annotated[str, Field(max_length=MAX_CLAIM_CHARS)]


class ItemCardUpdateRequest(_Request):
    """改商品卡片上商家填写的字段（至少一个）；null 表示清空。平台字段不在这里（422）。"""

    size_advice: str | None = Field(default=None, max_length=MAX_SIZE_ADVICE_CHARS)
    care: list[ItemLine] | None = Field(default=None, max_length=MAX_LIST_ITEMS)
    usage: list[ItemLine] | None = Field(default=None, max_length=MAX_LIST_ITEMS)
    forbidden_claims: list[ClaimLine] | None = Field(default=None, max_length=MAX_LIST_ITEMS)

    @model_validator(mode="after")
    def _something(self) -> ItemCardUpdateRequest:
        if not self.model_fields_set:
            raise ValueError(f"至少要改一个字段（{'、'.join(MERCHANT_ITEM_FIELDS)}）")
        return self

    def changes(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in MERCHANT_ITEM_FIELDS if f in self.model_fields_set}


class ItemCardSummaryView(_View):
    """商品列表里的一项：商家填了哪些字段（filled），方便提醒补尺码建议等（§5.2.3 第 3 步）。"""

    item_id: str
    title: str
    category: str | None = None
    updated_at: AwareDatetime
    filled: list[str]

    @classmethod
    def of(cls, card: ItemCard) -> ItemCardSummaryView:
        return cls(
            item_id=card.item_id,
            title=card.title,
            category=card.category,
            updated_at=card.updated_at,
            filled=[f for f in MERCHANT_ITEM_FIELDS if getattr(card, f)],
        )


class ItemCardView(_View):
    """商品卡片。平台字段只由商品同步更新；merchant_fields 列出商家能改的字段。"""

    item_id: str
    platform: str
    title: str
    category: str | None = None
    attributes: dict[str, str]
    skus: list[dict[str, Any]]
    size_chart: list[dict[str, Any]]
    detail_text: str | None = None
    size_advice: str | None = None
    care: list[str]
    usage: list[str]
    forbidden_claims: list[str]
    updated_at: AwareDatetime
    merchant_fields: list[str]
    changed: list[str] = []  # 本次修改实际改了的字段

    @classmethod
    def of(cls, card: ItemCard, *, changed: list[str] | None = None) -> ItemCardView:
        return cls(
            item_id=card.item_id,
            platform=card.platform.value,
            title=card.title,
            category=card.category,
            attributes=dict(card.attributes),
            skus=[s.model_dump(mode="json") for s in card.skus],
            size_chart=[dict(row) for row in card.size_chart],
            detail_text=card.detail_text,
            size_advice=card.size_advice,
            care=list(card.care),
            usage=list(card.usage),
            forbidden_claims=list(card.forbidden_claims),
            updated_at=card.updated_at,
            merchant_fields=list(MERCHANT_ITEM_FIELDS),
            changed=list(changed or []),
        )


# ================================================================ 商品同步（§5.2.3）


class ItemSyncRequest(_Request):
    """平台商品同步：item_id 为空（或不带请求体）同步整个店铺。"""

    item_id: Id | None = None


class SyncReportView(_View):
    """同步结果：各类商品 ID 和条数。rejected 是不属于本店 / 本平台、没有写入的商品数。"""

    scope: str
    item_id: str | None = None
    created: list[str]
    updated: list[str]
    unchanged: list[str]
    off_sale: list[str]
    missing_on_platform: list[str]
    ocr_failed_items: list[str]
    ocr_failures: int
    rejected: int
    counts: dict[str, int]
    synced_at: AwareDatetime

    @classmethod
    def of(cls, report: SyncReport) -> SyncReportView:
        data = report.model_dump(exclude={"shop_id"})
        return cls(**data, counts=report.counts())


# ================================================================ 店铺配置（§9.3）


class ConfigUpdateRequest(_Request):
    """改店铺配置：配置项 → 新值；null 表示删掉店铺覆盖项（回到类目 / 平台默认）。

    只能改 editable 里的项；影子模式、全自动开关（locked）商家不能改（409）。
    """

    changes: dict[str, Any] = Field(min_length=1, max_length=MAX_CONFIG_KEYS)


class ConfigView(_View):
    """店铺配置：生效值、默认值（没有店铺覆盖项时）、店铺覆盖项、能改 / 不能改的项。"""

    effective: ShopConfig
    defaults: ShopConfig
    overrides: dict[str, Any]
    editable: list[str]
    locked: list[str]
    changed: list[str] = []
    removed: list[str] = []

    @classmethod
    def of(cls, snapshot: ShopConfigSnapshot) -> ConfigView:
        return cls(**snapshot.model_dump())


__all__ = [
    "ConfigUpdateRequest",
    "ConfigView",
    "FaqApproveEditedRequest",
    "FaqChangeView",
    "FaqCreateRequest",
    "FaqEditRequest",
    "FaqWarningView",
    "ItemCardSummaryView",
    "ItemCardUpdateRequest",
    "ItemCardView",
    "ItemSyncRequest",
    "PolicyUpdateRequest",
    "PolicyView",
    "SyncReportView",
]
