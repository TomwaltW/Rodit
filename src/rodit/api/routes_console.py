"""商家控制台接口（方案 §9.1 "知识编辑、配置、报表"、§5.2 知识中心、§5.2.3 商品同步、
§9.3 店铺配置）。

鉴权一律用 rodit.api.deps 的 *Dep（§9.2 串店检查）；业务逻辑在 rodit.app.console。
读接口给店长和商家（ConsoleDep），写接口只给商家本人（MerchantDep）；操作人取自调用方身份，
请求体里不接受（schemas_console 的请求模型 extra="forbid"）。

| 方法 | 路径（/v1/shops/{shop_id}/console 之后） | 角色 | 说明 |
|---|---|---|---|
| GET | /faqs?status=&item_id= | 店长、商家 | FAQ 列表（按状态、商品筛选） |
| POST | /faqs | 商家 | 录入 FAQ（直接上线） |
| POST | /faqs/{knowledge_id}/edit | 商家 | 修改：旧条目下线、新条目上线 |
| POST | /candidates/{knowledge_id}/approve-edited | 商家 | 审核候选知识时修改 |
| GET | /policies | 店长、商家 | 店铺政策 |
| POST | /policies/{key} | 商家 | 改一项政策（即时生效） |
| POST | /policies/{key}/clear | 商家 | 清空一项政策（不再适用时） |
| GET | /items | 店长、商家 | 商品卡片列表 |
| POST | /items/sync | 商家 | 平台商品同步 + 详情图 OCR |
| GET | /items/{item_id} | 店长、商家 | 商品卡片 |
| POST | /items/{item_id} | 商家 | 改商家填写的字段 |
| GET | /config | 店长、商家 | 店铺配置 |
| POST | /config | 商家 | 改店铺配置 |

/items/sync 注册在 /items/{item_id} 之前（同为 POST，先匹配到同步）；平台商品 ID 不会是 "sync"。
没有接入商品接口或 OCR 的单元，同步返回 503 sync_unavailable；
商品接口故障返回 503 tool_unavailable。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Path, Query

from rodit.api.deps import ConsoleDep, EmptyBody, KnowledgeId, MerchantDep, ServiceDep
from rodit.api.errors import ApiError
from rodit.api.schemas import KnowledgeView
from rodit.api.schemas_console import (
    ConfigUpdateRequest,
    ConfigView,
    FaqApproveEditedRequest,
    FaqChangeView,
    FaqCreateRequest,
    FaqEditRequest,
    ItemCardSummaryView,
    ItemCardUpdateRequest,
    ItemCardView,
    ItemSyncRequest,
    PolicyUpdateRequest,
    PolicyView,
    SyncReportView,
)
from rodit.core.enums import KnowledgeStatus, PolicyKey
from rodit.knowledge.sync import MSG_SYNC_UNAVAILABLE, ItemSyncUnavailable

router = APIRouter(tags=["商家控制台"])

_BASE = "/v1/shops/{shop_id}/console"

ItemId = Annotated[str, Path(min_length=1, max_length=128)]


# ---------------------------------------------------------------- FAQ（§5.2.1 优先级 4、§5.8）


@router.get(f"{_BASE}/faqs", response_model=list[KnowledgeView])
def list_faqs(
    principal: ConsoleDep,
    service: ServiceDep,
    status: Annotated[KnowledgeStatus | None, Query(description="不填列出全部状态")] = None,
    item_id: Annotated[
        str | None, Query(min_length=1, max_length=128, description="只列这个商品专属的 FAQ")
    ] = None,
) -> list[KnowledgeView]:
    """本店的 FAQ（按 id 升序）：已上线、候选、已下线。"""
    items = service.console.list_faqs(principal.shop_id, status=status, item_id=item_id)
    return [KnowledgeView.from_item(k) for k in items]


@router.post(f"{_BASE}/faqs", response_model=FaqChangeView)
def create_faq(
    body: FaqCreateRequest, principal: MerchantDep, service: ServiceDep
) -> FaqChangeView:
    """商家录入 FAQ：直接上线、立刻可检索。warnings 提醒回答时会不会被质检拦截（不拦截保存）。"""
    change = service.console.create_faq(
        principal.shop_id,
        question=body.question,
        answer=body.answer,
        item_id=body.item_id,
        intent=body.intent,
        merchant=principal.user_id,
    )
    return FaqChangeView.of(change)


@router.post(f"{_BASE}/faqs/{{knowledge_id}}/edit", response_model=FaqChangeView)
def edit_faq(
    knowledge_id: KnowledgeId, body: FaqEditRequest, principal: MerchantDep, service: ServiceDep
) -> FaqChangeView:
    """修改已上线的 FAQ：旧条目下线（retired），改过的内容作为新条目上线（knowledge）。

    不在原 id 上改写：已有推荐的证据 faq:{旧 id} 仍指向买家当时看到的内容。
    """
    change = service.console.edit_faq(
        principal.shop_id, knowledge_id, body.changes(), merchant=principal.user_id
    )
    return FaqChangeView.of(change)


@router.post(f"{_BASE}/candidates/{{knowledge_id}}/approve-edited", response_model=FaqChangeView)
def approve_edited(
    knowledge_id: KnowledgeId,
    body: FaqApproveEditedRequest,
    principal: MerchantDep,
    service: ServiceDep,
) -> FaqChangeView:
    """审核候选知识时顺手修改（§5.8）：有改动时候选条目下线、改过的内容上线；没有改动就是普通审核。"""
    change = service.console.approve_edited(
        principal.shop_id, knowledge_id, body.changes(), merchant=principal.user_id
    )
    return FaqChangeView.of(change)


# ---------------------------------------------------------------- 店铺政策（§5.2.1 优先级 2）


@router.get(f"{_BASE}/policies", response_model=list[PolicyView])
def list_policies(principal: ConsoleDep, service: ServiceDep) -> list[PolicyView]:
    """全部店铺政策项（没填的 value 为 null，附带问卷题号）。"""
    return [PolicyView.of(row) for row in service.console.list_policies(principal.shop_id)]


@router.post(f"{_BASE}/policies/{{key}}", response_model=PolicyView)
def set_policy(
    key: Annotated[PolicyKey, Path()],
    body: PolicyUpdateRequest,
    principal: MerchantDep,
    service: ServiceDep,
) -> PolicyView:
    """改一项店铺政策（只能是已知的政策项）。改动即时生效（§5.2.1）。"""
    row = service.console.set_policy(
        principal.shop_id, key.value, body.value, merchant=principal.user_id
    )
    return PolicyView.of(row)


@router.post(f"{_BASE}/policies/{{key}}/clear", response_model=PolicyView)
def clear_policy(
    key: Annotated[PolicyKey, Path()],
    principal: MerchantDep,
    service: ServiceDep,
    body: EmptyBody = None,
) -> PolicyView:
    """清空一项店铺政策（不再适用时）：之后回答不再引用它，"要补的知识"里会提醒。"""
    row = service.console.clear_policy(principal.shop_id, key.value, merchant=principal.user_id)
    return PolicyView.of(row)


# ---------------------------------------------------------------- 商品卡片与同步（§5.2.2、§5.2.3）


@router.get(f"{_BASE}/items", response_model=list[ItemCardSummaryView])
def list_items(principal: ConsoleDep, service: ServiceDep) -> list[ItemCardSummaryView]:
    """本店的商品卡片（按 item_id 升序）：标题、类目、商家填了哪些字段。"""
    return [ItemCardSummaryView.of(c) for c in service.console.list_items(principal.shop_id)]


@router.post(f"{_BASE}/items/sync", response_model=SyncReportView)
def sync_items(
    principal: MerchantDep,
    service: ServiceDep,
    body: Annotated[ItemSyncRequest | None, Body()] = None,
) -> SyncReportView:
    """平台商品同步（§5.2.3）：新商品建卡片，已有卡片更新平台字段（商家填的字段不动），
    详情图 OCR 文字并入详情；已下架、平台上没有了的只报告不删除。"""
    try:
        report = service.console.sync_items(
            principal.shop_id,
            item_id=body.item_id if body else None,
            merchant=principal.user_id,
        )
    except ItemSyncUnavailable as exc:
        raise ApiError(503, "sync_unavailable", MSG_SYNC_UNAVAILABLE) from exc
    return SyncReportView.of(report)


@router.get(f"{_BASE}/items/{{item_id}}", response_model=ItemCardView)
def get_item(item_id: ItemId, principal: ConsoleDep, service: ServiceDep) -> ItemCardView:
    """商品卡片（平台字段 + 商家填写的字段）。"""
    return ItemCardView.of(service.console.get_item(principal.shop_id, item_id))


@router.post(f"{_BASE}/items/{{item_id}}", response_model=ItemCardView)
def update_item(
    item_id: ItemId, body: ItemCardUpdateRequest, principal: MerchantDep, service: ServiceDep
) -> ItemCardView:
    """改商品卡片上商家填写的字段（尺码建议、洗涤保养、使用说明、不能说的卖点）。"""
    card, changed = service.console.update_item(
        principal.shop_id, item_id, body.changes(), merchant=principal.user_id
    )
    return ItemCardView.of(card, changed=changed)


# ---------------------------------------------------------------- 店铺配置（§9.3）


@router.get(f"{_BASE}/config", response_model=ConfigView)
def get_config(principal: ConsoleDep, service: ServiceDep) -> ConfigView:
    """店铺配置：生效值、默认值、店铺覆盖项、商家能改 / 不能改的项。"""
    return ConfigView.of(service.console.get_config(principal.shop_id))


@router.post(f"{_BASE}/config", response_model=ConfigView)
def update_config(
    body: ConfigUpdateRequest, principal: MerchantDep, service: ServiceDep
) -> ConfigView:
    """改店铺配置（§9.3）。null 删掉覆盖项；影子模式、全自动开关不能改（409）。"""
    snapshot = service.console.update_config(
        principal.shop_id, body.changes, merchant=principal.user_id
    )
    return ConfigView.of(snapshot)


__all__ = ["router"]
