"""HTTP 接口（方案 §8.1 模式 C、§5.8 反馈、§7.3.2 售后审核、§8.4 店铺授权、§9.2 租户隔离、
§12.4 报表）。

create_app 是应用工厂：一个 AssistService（一个平台单元，§9.1）+ 一个 Authenticator，没有全局状态。
接口层只调用 AssistService，另外只读 Unit 的字段（店铺状态、店铺授权、审核记录、平台售后单、
适配器注册表、商品卡片）做鉴权前置检查和展示。

模式 C 网页工具（rodit.api.web）：GET / 和 /assist 是自带的静态页面，页面本身不含数据、不需要鉴权；
页面脚本用 /v1/me 登录（令牌 → 身份；部署配置了大模型公示信息时一并返回，页脚显示，§12.3），
用 /v1/shops/{shop_id}/items 选商品，其余都是下面的接口。

鉴权顺序（每个店铺范围的路由都一样，§9.2 串店测试覆盖全部路由）：
    令牌 → Principal（401）→ 路径 shop_id 与 principal.shop_id 一致（403 tenant_isolation）
    → 角色（403 forbidden_role）→ 店铺存在且未停用（403 shop_inactive；取消订购接口除外，可重试）。

平台事件接口 /v1/platform/{access}/events 给平台对接用（integration 角色）：先用适配器无副作用地
解析出事件所属店铺，和调用方的店铺比对、检查店铺授权（§8.4），都通过了才交给应用服务处理——
跨店事件不会在适配器里留下任何痕迹（去重记录、会话上下文、撤回）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Body, FastAPI, Path, Response

from rodit.api import routes_console, routes_rollout, routes_tenancy
from rodit.api.auth import PERSON_ROLES, Authenticator, Principal
from rodit.api.deps import (
    MSG_FORBIDDEN_ROLE,
    AfterSaleId,
    AgentDep,
    ConsoleDep,
    EmptyBody,
    IntegrationDep,
    KnowledgeId,
    MerchantDep,
    PrincipalDep,
    RevokeDep,
    ServiceDep,
    Since,
    SuggestionId,
    require_active_shop,
)
from rodit.api.errors import ApiError, install_error_handlers, tenant_violation
from rodit.api.schemas import (
    AfterSaleReviewView,
    ConfirmRequest,
    EditedRequest,
    EngineResultView,
    FeedbackView,
    GapView,
    HealthView,
    ItemSummaryView,
    KnowledgeView,
    LLMDisclosureView,
    MeView,
    NegotiateRequest,
    NegotiationDoneRequest,
    QuestionnaireRequest,
    QuestionnaireView,
    RejectedRequest,
    RevokeView,
    StandaloneAskRequest,
)
from rodit.api.web import build_web_router
from rodit.app.service import AssistService
from rodit.core.enums import ReviewState
from rodit.core.errors import RoditError, TenantIsolationError
from rodit.core.models import AfterSaleReview, EngineResult
from rodit.core.ports import PlatformAdapter
from rodit.feedback import AdoptionReport, FeedbackOutcome, QAReport

API_TITLE = "Rodit 智能客服接口"
API_VERSION = "0.1.0"

_NO_CONTENT = {204: {"description": "不是买家消息、重复消息或无法识别，没有处理"}}

MSG_MANAGER_REQUIRED = "金额超过店铺上限，必须店长确认（R6，§7.3.2）"
MSG_UNVERIFIABLE = "无法在处理前核对事件所属店铺，已拒绝（§9.2）"


# ================================================================ 依赖（rodit.api.deps）


# ================================================================ 视图


def _result_view(principal: Principal, result: EngineResult | None) -> EngineResultView | Response:
    if result is None:
        return Response(status_code=204)
    if result.shop_id != principal.shop_id:  # 纵深防御：不可能发生，发生了也不返回别的店铺的数据
        raise TenantIsolationError("结果不属于调用方店铺")
    return EngineResultView.from_result(result)


def _feedback_view(outcome: FeedbackOutcome) -> FeedbackView:
    record = outcome.record
    assert record.id is not None
    return FeedbackView(
        suggestion_id=record.id,
        status=record.status,
        candidate_knowledge_id=outcome.candidate.id if outcome.candidate else None,
        candidate_skipped=outcome.candidate_skipped,
        knowledge_gap=outcome.gap.missing if outcome.gap else None,
        labeled=outcome.label is not None,
    )


def _review_view(service: AssistService, review: AfterSaleReview) -> AfterSaleReviewView:
    """待客服确认时附上平台售后单的当前状态（确认要带 status_version）。

    读平台原始接口、不走缓存：缓存里的状态版本可能已经过期（§7.3.2 调用前重新读取）。
    """
    info = None
    if review.state == ReviewState.CONFIRM:
        try:
            info = service.unit.tools.get_aftersale(review.shop_id, review.aftersale_id)
        except RoditError:
            info = None  # 平台接口故障：审核记录照常返回，客服刷新后再确认（§12.2）
        if info is not None and info.shop_id != review.shop_id:
            info = None
    return AfterSaleReviewView.from_review(review, platform=info)


# ================================================================ 平台事件：处理前核对店铺


def _event_shop_id(adapter: PlatformAdapter, raw: Mapping[str, Any]) -> str | None:
    """不改变适配器状态地解析出事件所属店铺；适配器不会处理的事件返回 None。

    用 PlatformAdapter.peek_shop_id（协议要求不改任何状态）；没有实现时拒绝处理
    （无法在处理前核对店铺）。
    """
    peek = getattr(adapter, "peek_shop_id", None)
    if not callable(peek):
        raise TenantIsolationError(MSG_UNVERIFIABLE)
    shop_id = peek(dict(raw))
    if not isinstance(shop_id, str) or not shop_id.strip():
        return None
    return shop_id


# ================================================================ 路由

router = APIRouter()


@router.get("/healthz", response_model=HealthView, tags=["运维"])
def healthz() -> HealthView:
    """存活检查（不需要鉴权，不返回任何店铺信息）。"""
    return HealthView()


def _disclosure(service: AssistService) -> LLMDisclosureView | None:
    """部署配置里对外公示的大模型名称、备案号（§12.3）；都没填时为 None（响应里不出现）。"""
    llm = service.unit.config.llm
    public_name = (llm.public_name or "").strip() or None
    filing_no = (llm.filing_no or "").strip() or None
    if public_name is None and filing_no is None:
        return None
    return LLMDisclosureView(public_name=public_name, filing_no=filing_no)


@router.get("/v1/me", response_model=MeView, response_model_exclude_none=True, tags=["模式 C"])
def whoami(principal: PrincipalDep, service: ServiceDep) -> MeView:
    """当前令牌对应的身份（网页工具登录用：模式 C 话术助手、商家控制台）：
    店铺、账号、角色、平台单元；部署配置了大模型公示信息时另带 llm（公示名称、备案号，§12.3），
    网页工具在页脚显示。

    具体的人（PERSON_ROLES：客服、店长、商家、复核人员）都能用；平台对接的令牌不是人，
    不能登录网页工具（403 forbidden_role）。各页面按角色决定显示什么：复核人员在话术助手页只看到
    去控制台的提示，客服在控制台只看到去话术助手的提示。
    只返回调用方自己的身份，不回显令牌、不读店铺数据；店铺停用时照常返回（之后的店铺接口是 403）。
    """
    if principal.role not in PERSON_ROLES:
        raise ApiError(403, "forbidden_role", MSG_FORBIDDEN_ROLE)
    return MeView(
        shop_id=principal.shop_id,
        user_id=principal.user_id,
        role=principal.role,
        platform=service.unit.platform,
        llm=_disclosure(service),
    )


@router.get("/v1/shops/{shop_id}/items", response_model=list[ItemSummaryView], tags=["模式 C"])
def list_items(principal: AgentDep, service: ServiceDep) -> list[ItemSummaryView]:
    """本店的商品（按 item_id 升序，只有 ID 和标题）：模式 C 网页工具选当前商品（§5.3、§8.1）。

    只读。
    """
    cards = service.unit.repos.knowledge.list_item_cards(principal.shop_id)
    return [
        ItemSummaryView(item_id=card.item_id, title=card.title)
        for card in cards
        if card.shop_id == principal.shop_id  # 纵深防御：仓储已按店铺隔离（§9.2）
    ]


@router.post(
    "/v1/shops/{shop_id}/standalone/ask",
    response_model=EngineResultView,
    responses=_NO_CONTENT,
    tags=["模式 C"],
)
def standalone_ask(
    body: StandaloneAskRequest, principal: AgentDep, service: ServiceDep
) -> EngineResultView | Response:
    """模式 C 独立话术助手（§8.1）：客服粘贴买家问题，拿推荐回答（§5.6 推荐卡片）。

    客服账号取自调用方身份；影子模式（§13.4）不返回推荐。同一 request_id 重复提交返回 204。
    """
    result = service.ask_standalone(
        shop_id=principal.shop_id,
        session_id=body.session_id,
        question=body.question,
        item_id=body.item_id,
        sku_id=body.sku_id,
        agent_id=principal.user_id,
        request_id=body.request_id,
    )
    return _result_view(principal, result)


@router.post(
    "/v1/platform/{access}/events",
    response_model=EngineResultView,
    responses=_NO_CONTENT,
    tags=["平台接入"],
)
def platform_event(
    access: Annotated[str, Path(pattern="^[AB]$", description="A 平台推送 / B 插件上报")],
    raw: Annotated[dict[str, Any], Body()],
    principal: IntegrationDep,
    service: ServiceDep,
) -> EngineResultView | Response:
    """模式 A 平台推送 / 模式 B 插件上报的一条原始事件（§8.1、§8.2）。只给平台对接用。

    处理前：适配器解析出事件所属店铺 → 必须是调用方的店铺（§9.2）→ 店铺存在、授权有效（§8.4）。
    不是买家消息（客服回声、撤回、系统提示）、重复消息、无法识别时返回 204。
    """
    unit = service.unit
    adapter = unit.adapters.get(unit.platform, access)  # 没有注册该接入模式 → KeyError → 404
    shop_id = _event_shop_id(adapter, raw)
    if shop_id is None:
        return Response(status_code=204)
    if shop_id != principal.shop_id:
        raise tenant_violation()
    require_active_shop(service, shop_id)
    unit.auth.require(shop_id)  # 授权无效 → AuthExpired → 403，此时适配器还没有碰这条事件
    return _result_view(principal, service.handle_event(access, raw))


# ---------------------------------------------------------------- 客服反馈（§5.8）


@router.post(
    "/v1/shops/{shop_id}/suggestions/{suggestion_id}/sent",
    response_model=FeedbackView,
    tags=["反馈"],
)
def suggestion_sent(
    suggestion_id: SuggestionId,
    principal: AgentDep,
    service: ServiceDep,
    body: EmptyBody = None,
) -> FeedbackView:
    """原样发送（§5.8 正样本）。客服账号取自调用方身份。"""
    outcome = service.feedback_sent(principal.shop_id, suggestion_id, agent_id=principal.user_id)
    return _feedback_view(outcome)


@router.post(
    "/v1/shops/{shop_id}/suggestions/{suggestion_id}/edited",
    response_model=FeedbackView,
    tags=["反馈"],
)
def suggestion_edited(
    suggestion_id: SuggestionId,
    body: EditedRequest,
    principal: AgentDep,
    service: ServiceDep,
) -> FeedbackView:
    """编辑后发送（§5.8）：记录修改前后，生成候选知识等商家审核。"""
    outcome = service.feedback_edited(
        principal.shop_id,
        suggestion_id,
        agent_id=principal.user_id,
        edited_text=body.edited_text,
    )
    return _feedback_view(outcome)


@router.post(
    "/v1/shops/{shop_id}/suggestions/{suggestion_id}/rejected",
    response_model=FeedbackView,
    tags=["反馈"],
)
def suggestion_rejected(
    suggestion_id: SuggestionId,
    body: RejectedRequest,
    principal: AgentDep,
    service: ServiceDep,
) -> FeedbackView:
    """不合适（§5.8）：知识缺失 → 提醒商家补知识；理解错误 → 判断层标注集；表达不好 → 调优队列。"""
    outcome = service.feedback_rejected(
        principal.shop_id,
        suggestion_id,
        agent_id=principal.user_id,
        reason=body.reason,
        true_intent=body.true_intent,
    )
    return _feedback_view(outcome)


# ---------------------------------------------------------------- 商家：知识中心（§5.2、§5.8）


@router.get(
    "/v1/shops/{shop_id}/knowledge/candidates",
    response_model=list[KnowledgeView],
    tags=["知识"],
)
def knowledge_candidates(principal: MerchantDep, service: ServiceDep) -> list[KnowledgeView]:
    """等商家审核的候选知识（§5.8 编辑后发送 → 候选知识 → 商家审核）。"""
    return [KnowledgeView.from_item(k) for k in service.pending_candidates(principal.shop_id)]


@router.post(
    "/v1/shops/{shop_id}/knowledge/candidates/{knowledge_id}/approve",
    response_model=KnowledgeView,
    tags=["知识"],
)
def knowledge_approve(
    knowledge_id: KnowledgeId,
    principal: MerchantDep,
    service: ServiceDep,
    body: EmptyBody = None,
) -> KnowledgeView:
    """商家审核通过候选知识（§5.8）：此后才能用于回答。审核人取自调用方身份。"""
    item = service.approve_candidate(principal.shop_id, knowledge_id, merchant=principal.user_id)
    return KnowledgeView.from_item(item)


@router.post(
    "/v1/shops/{shop_id}/knowledge/{knowledge_id}/retire",
    response_model=KnowledgeView,
    tags=["知识"],
)
def knowledge_retire(
    knowledge_id: KnowledgeId,
    principal: MerchantDep,
    service: ServiceDep,
    body: EmptyBody = None,
) -> KnowledgeView:
    """商家驳回候选知识或下线 FAQ（§5.8）。"""
    item = service.retire_knowledge(principal.shop_id, knowledge_id, merchant=principal.user_id)
    return KnowledgeView.from_item(item)


@router.get("/v1/shops/{shop_id}/knowledge/gaps", response_model=list[GapView], tags=["知识"])
def knowledge_gaps(principal: ConsoleDep, service: ServiceDep) -> list[GapView]:
    """要提醒商家补的知识：冷启动检查（§5.2.3）+ 引擎和反馈记下的缺失（§5.7、§5.8）。"""
    return [GapView(**gap) for gap in service.knowledge_gaps(principal.shop_id)]


@router.post("/v1/shops/{shop_id}/questionnaire", response_model=QuestionnaireView, tags=["知识"])
def questionnaire(
    body: QuestionnaireRequest, principal: MerchantDep, service: ServiceDep
) -> QuestionnaireView:
    """店铺政策问卷（§5.2.3 冷启动、附录 C）：写入政策，合并店铺配置（§9.3）。"""
    applied = service.apply_questionnaire(principal.shop_id, body.answers)
    return QuestionnaireView(policies=applied["policies"], config=applied["config"])


# ---------------------------------------------------------------- 报表（§12.4）


@router.get("/v1/shops/{shop_id}/reports/adoption", response_model=AdoptionReport, tags=["报表"])
def report_adoption(
    principal: ConsoleDep, service: ServiceDep, since: Since = None
) -> AdoptionReport:
    """推荐采纳率（原样 / 编辑后 / 弃用），按问题类型拆分，含弃用原因分布（§12.4）。"""
    return service.adoption(principal.shop_id, since=since)


@router.get("/v1/shops/{shop_id}/reports/qa", response_model=QAReport, tags=["报表"])
def report_qa(principal: ConsoleDep, service: ServiceDep, since: Since = None) -> QAReport:
    """质检拦截率及原因、硬规则命中、降级、各环节耗时（§12.4）。"""
    return service.qa(principal.shop_id, since=since)


# ---------------------------------------------------------------- 售后审核（§7.3.2）


@router.post(
    "/v1/shops/{shop_id}/aftersales/{aftersale_id}/review",
    response_model=AfterSaleReviewView,
    tags=["售后审核"],
)
def aftersale_review(
    aftersale_id: AfterSaleId,
    principal: AgentDep,
    service: ServiceDep,
    body: EmptyBody = None,
) -> AfterSaleReviewView:
    """拉取售后单并跑规则 R1–R8 给出建议（已有记录时重读平台状态、推进状态机）。只读。"""
    review = service.review_aftersale(principal.shop_id, aftersale_id)
    return _review_view(service, review)


@router.post(
    "/v1/shops/{shop_id}/aftersales/{aftersale_id}/negotiate",
    response_model=AfterSaleReviewView,
    tags=["售后审核"],
)
def aftersale_negotiate(
    aftersale_id: AfterSaleId,
    principal: AgentDep,
    service: ServiceDep,
    body: Annotated[NegotiateRequest | None, Body()] = None,
) -> AfterSaleReviewView:
    """客服选择协商（待客服确认 → 协商中）。操作人取自调用方身份。"""
    review = service.negotiate_aftersale(
        principal.shop_id,
        aftersale_id,
        operator=principal.user_id,
        note=body.note if body else None,
    )
    return _review_view(service, review)


@router.post(
    "/v1/shops/{shop_id}/aftersales/{aftersale_id}/negotiation-done",
    response_model=AfterSaleReviewView,
    tags=["售后审核"],
)
def aftersale_negotiation_done(
    aftersale_id: AfterSaleId,
    principal: AgentDep,
    service: ServiceDep,
    body: Annotated[NegotiationDoneRequest | None, Body()] = None,
) -> AfterSaleReviewView:
    """协商有结果（协商中 → 待客服确认；平台已处理则同步状态）。"""
    review = service.negotiation_done(
        principal.shop_id,
        aftersale_id,
        operator=principal.user_id,
        outcome=body.outcome if body else None,
    )
    return _review_view(service, review)


@router.post(
    "/v1/shops/{shop_id}/aftersales/{aftersale_id}/confirm",
    response_model=AfterSaleReviewView,
    tags=["售后审核"],
)
def aftersale_confirm(
    aftersale_id: AfterSaleId,
    body: ConfirmRequest,
    principal: AgentDep,
    service: ServiceDep,
) -> AfterSaleReviewView:
    """客服确认后执行售后写操作（§7.3.2，全部接口里唯一的写平台操作，§4.1 原则 3）。

    确认人 = 调用方；by_manager 只在调用方是店长角色时为真（R6），都不取请求体。
    审核助手会重读售后单、核对状态版本、用幂等键调用平台，并留痕。
    """
    existing = service.unit.repos.reviews.get(principal.shop_id, aftersale_id)
    if existing is not None and existing.requires_manager and not principal.is_manager:
        raise ApiError(403, "manager_required", MSG_MANAGER_REQUIRED)
    review = service.confirm_aftersale(
        principal.shop_id,
        aftersale_id,
        body.action,
        operator=principal.user_id,
        expected_status_version=body.expected_status_version,
        by_manager=principal.is_manager,
        reason_code=body.reason_code,
        reason_text=body.reason_text,
    )
    return _review_view(service, review)


# ---------------------------------------------------------------- 店铺生命周期（§8.4）


@router.post("/v1/shops/{shop_id}/revoke", response_model=RevokeView, tags=["店铺"])
def revoke(principal: RevokeDep, service: ServiceDep, body: EmptyBody = None) -> RevokeView:
    """取消订购 / 授权过期：删令牌、停止处理，删除该店铺在本单元的全部数据（§8.4）。

    可以重试（各存储的删除是幂等的）；之后该店铺的所有请求都是 403。
    """
    counts = service.revoke_shop(principal.shop_id)
    return RevokeView(counts=dict(counts), total=sum(counts.values()))


# ================================================================ 工厂


def create_app(
    service: AssistService, *, authenticator: Authenticator, web_page: bool = True
) -> FastAPI:
    """创建一个平台单元（§9.1）的 HTTP 应用。

    service 和 authenticator 挂在这个应用的 app.state 上，没有模块级的全局状态。
    web_page：是否挂模式 C 网页工具（GET / 和 /assist，§8.1）；只做模式 A / B 的单元可以关掉。
    """
    if not isinstance(service, AssistService):
        raise TypeError("service 必须是 AssistService")
    if not isinstance(authenticator, Authenticator):
        raise TypeError("authenticator 必须实现 Authenticator.authenticate")
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=f"平台单元：{service.unit.platform.value}（方案 §9.1）",
    )
    app.state.rodit_service = service
    app.state.rodit_authenticator = authenticator
    install_error_handlers(app)
    app.include_router(router)
    app.include_router(routes_console.router)  # 商家控制台：知识编辑、店铺配置、商品同步（§9.1）
    app.include_router(routes_rollout.router)  # 上线节奏与每周复核（§13.4、§13.5）
    app.include_router(routes_tenancy.router)  # 店铺生命周期：开通、订购事件（§8.4）
    if web_page:
        app.include_router(build_web_router(service.unit.platform))
    return app


__all__ = ["API_TITLE", "API_VERSION", "create_app"]
