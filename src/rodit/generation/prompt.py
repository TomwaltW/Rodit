"""生成提示词（方案 §5.4 提示词骨架、§5.3 最小化个人信息、§5.5 重新生成、§6.6 规则 3–5）。

骨架（system）：角色 + 五条规则；用户消息：【依据】【对话】【要求】+ JSON 输出格式。

- 店铺名、语气、称呼、字数上限都取 ctx.config（店铺生效配置），
  平台名取 core.evidence.PLATFORM_NAMES；
- 依据按来源优先级排列，每条 "[{id}] {label}：{content}"，
  总长不超过 GenerationConfig.max_evidence_chars；
- 对话取 core.states.recent_turns（最近 history_rounds 轮 + 当前消息），带角色；
- 买家 / 客服文字、依据内容、质检原因一律先遮个人信息（generation.masking.mask_text），
  提示词里没有 buyer_ref、昵称、手机号、地址；
- 多问题分条（§6.6 规则 5）：引擎按 part 复制出来的路由仍带 rule=5，据此只回答该部分，也可以显式传
  focus_intent；安抚（规则 3）要求先道歉安抚；高风险（规则 4）加谨慎要求和 route.risk_hints；
- 签收未收到（§7.3.1、§11.3）加一条要求：说明签收细分状态、驿站 / 快递柜给取件指引、
  还是没找到就在订单页申请售后；
- 买家发了图片（ctx.image_findings，§15 Phase 3）加一条要求：对话里【买家发来图片：…】是系统对图片的
  识别，只供参考，不据此下结论、不承诺处理结果，质量问题请买家申请售后时上传照片；
- 重新生成时附上"上次回答未通过质检：…，请修改"。
"""

from __future__ import annotations

import re
from typing import Literal

from rodit.core.budget import clamp_timeout
from rodit.core.config import GenerationConfig
from rodit.core.enums import INTENT_LABELS, Intent, Mode, Role, RouteAction
from rodit.core.evidence import PLATFORM_NAMES
from rodit.core.evidence_line import evidence_line_parts
from rodit.core.models import ConversationContext, Evidence, EvidenceBundle, RouteDecision
from rodit.core.ports import LLMMessage, LLMRequest
from rodit.core.states import recent_turns
from rodit.generation.masking import mask_text

PromptVariant = Literal["direct", "with_followup"]

SPLIT_MULTI_RULE = 5
"""§6.6 规则 5（多问题分条推荐）的序号。
引擎按 part 复制路由时 rule 不变，据此识别"只答这一部分"。"""

CONFIRM_PHRASE = "这个我帮您确认一下"

NOT_RECEIVED_REQUIREMENT = (
    "买家说显示签收但没收到：先按【依据】说明物流的签收细分状态（本人签收、驿站或快递柜代收），"
    "驿站或快递柜代收时给出地点和取件指引；再告诉买家，如果还是没找到，可以在订单页申请售后。"
    "不要替买家申请售后，也不要承诺赔偿或补发。"
)
"""签收未收到（§7.3.1、§11.3）的回答要求：物流细分状态、驿站 / 快递柜信息、没找到引导申请售后。"""

IMAGE_REQUIREMENT = (
    "买家发了图片：对话里【买家发来图片：…】是系统对图片内容的识别，只供参考。"
    "不要根据图片下结论，也不要承诺退款、补发、赔偿等处理结果；"
    "商品破损、有瑕疵或发错的，请买家在申请售后时上传照片。"
)
"""买家发了图片时的回答要求（§15 Phase 3 图片理解只是参考、§7.3.2 R5 由客服看凭证决定）。"""

_ROLE_NAMES = {Role.BUYER: "买家", Role.AGENT: "客服", Role.BOT: "客服（自动回复）"}
_TONE_HINTS = {
    "亲切": "亲切，可以用“哦”“呢”等语气词",
    "专业": "专业、礼貌",
    "简洁": "简洁、直接",
}
_WHITESPACE = re.compile(r"\s+")
_MIN_PARTIAL = 10  # 依据超长时，剩余预算至少这么多字才放半条


def split_focus_intent(route: RouteDecision) -> Intent | None:
    """多问题消息里本条要回答的那个问题类型；不是多问题分条时返回 None。"""
    if route.action == RouteAction.SPLIT_MULTI or route.rule == SPLIT_MULTI_RULE:
        return route.intent
    return None


def answer_char_limit(ctx: ConversationContext, config: GenerationConfig) -> int:
    """提示词里的字数上限。全自动时引擎会在开头加 AI 标识前缀（§12.3），提前留出位置。"""
    limit = ctx.config.max_answer_chars
    if ctx.mode == Mode.AUTO:
        limit -= len(config.auto_reply_prefix)
    return max(limit, 10)


def _system_prompt(ctx: ConversationContext, config: GenerationConfig) -> str:
    shop = ctx.config.display_name.strip() or "本店"
    platform = PLATFORM_NAMES.get(ctx.platform, str(ctx.platform))
    tone = _TONE_HINTS.get(ctx.config.tone, ctx.config.tone)
    limit = answer_char_limit(ctx, config)
    return "\n".join(
        [
            f"你是{shop}的客服，正在{platform}上回答买家的咨询。",
            "规则：",
            f"1. 只使用【依据】中的信息回答；依据里没有的内容，回复“{CONFIRM_PHRASE}”，不要猜。",
            "2. 不承诺依据之外的任何事情，包括价格、赠品、到货时间、赔偿。",
            "3. 不提供、也不索要任何站外联系方式、链接或收款方式。",
            "4. 不使用“最”“第一”“绝对”等绝对化用语，不做功效或医疗方面的承诺。",
            f"5. 语气：{tone}，称呼买家为“{ctx.config.salutation}”；长度不超过 {limit} 字；"
            "一次最多问买家一个问题。",
            "回答是直接发给买家的纯文本：不要用 Markdown、列表符号或链接。",
        ]
    )


def _one_line(text: str) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def _evidence_block(evidence: EvidenceBundle, budget: int) -> list[str]:
    """按来源优先级排列依据，总长（含换行）不超过 budget。

    每行的写法和计长用 core.evidence_line，和检索的长度预算（knowledge.conflict.trim_to_budget）
    一致：检索留下的依据这里都放得下。万一放不下（证据包不是检索给的），跳过那一条、继续放后面的，
    不在第一条放不下处整体截断；一条都放不下（单条超长）时截断放第一条，不交空依据。
    """
    ordered: list[Evidence] = sorted(evidence.evidences, key=lambda e: e.priority)
    lines: list[str] = []
    used = 0
    for ev in ordered:
        head, content = evidence_line_parts(ev)
        line = head + content
        sep = 1 if lines else 0  # 行间换行也算进预算
        remaining = budget - used - sep
        if len(line) <= remaining:
            lines.append(line)
            used += sep + len(line)
            continue
        room = remaining - len(head) - 1
        if not lines and room >= _MIN_PARTIAL:
            lines.append(head + content[:room] + "…")
            used += len(lines[-1])
    return lines


def _dialogue_block(ctx: ConversationContext, config: GenerationConfig) -> list[str]:
    turns = recent_turns(ctx, history_rounds=config.history_rounds)
    lines: list[str] = []
    for i, turn in enumerate(turns):
        name = _ROLE_NAMES.get(turn.role, str(turn.role))
        if i == len(turns) - 1:
            name += "（当前消息）"
        lines.append(f"{name}：{_one_line(mask_text(turn.text))}")
    return lines


def _requirements(
    route: RouteDecision,
    *,
    variant: PromptVariant,
    focus_intent: Intent | None,
    retry_feedback: list[str] | None,
) -> list[str]:
    intent = focus_intent or route.intent
    reqs = [f"买家的问题类型：{INTENT_LABELS[intent]}。"]
    if focus_intent is not None:
        reqs.append(
            f"买家这条消息包含多个问题，本条只回答关于{INTENT_LABELS[focus_intent]}的部分。"
        )
    if route.action == RouteAction.SOOTHE:
        reqs.append("买家情绪比较激动：先用一句道歉或安抚的话开头，再回答问题。")
    elif intent == Intent.QUALITY_ISSUE:  # §7.3.1 收到的东西坏了：情绪平稳也要先安抚
        reqs.append("买家反映收到的商品有问题：先用一句道歉安抚的话开头，再说明怎么处理。")
    if intent == Intent.NOT_RECEIVED:  # §7.3.1、§11.3 显示签收但没收到
        reqs.append(NOT_RECEIVED_REQUIREMENT)
    if route.high_risk or route.action == RouteAction.HIGH_RISK_ANSWER:
        reqs.append(
            "这是高风险问题：只说依据里有的事实，不做任何依据之外的承诺；"
            f"拿不准的内容回复“{CONFIRM_PHRASE}”。"
        )
    for hint in route.risk_hints:
        reqs.append(f"风险提示：{_one_line(mask_text(hint))}。")
    if variant == "with_followup":
        reqs.append(
            "先直接回答买家的问题，再在结尾追问一个问题（只问一个），帮助确认买家的具体需求。"
        )
    else:
        reqs.append("直接回答买家的问题。")
    reasons = [_one_line(mask_text(r)) for r in (retry_feedback or []) if r and r.strip()]
    if reasons:
        reqs.append(f"上次回答未通过质检：{'；'.join(reasons)}，请修改。")
    return [f"- {r}" for r in reqs]


_OUTPUT_FORMAT = (
    '请只输出 JSON：{"answer": "...", "evidence_ids": ["..."], "need_clarify": false}\n'
    "evidence_ids 填回答用到的依据 ID（【依据】里方括号中的内容）；"
    "依据不够、需要先问买家时 need_clarify 填 true，answer 写那一个问题。"
)


def build_generation_messages(
    ctx: ConversationContext,
    route: RouteDecision,
    evidence: EvidenceBundle,
    *,
    variant: PromptVariant = "direct",
    config: GenerationConfig,
    retry_feedback: list[str] | None = None,
    focus_intent: Intent | None = None,
) -> list[LLMMessage]:
    """按 §5.4 骨架构造提示词。focus_intent 为空时从路由推断（见 split_focus_intent）。"""
    focus = focus_intent if focus_intent is not None else split_focus_intent(route)
    evidence_lines = _evidence_block(evidence, config.max_evidence_chars)
    user = "\n".join(
        [
            "【依据】",
            *(evidence_lines or ["（没有找到依据）"]),
            "【对话】",
            *_dialogue_block(ctx, config),
            "【要求】",
            *_requirements(
                route, variant=variant, focus_intent=focus, retry_feedback=retry_feedback
            ),
            *([f"- {IMAGE_REQUIREMENT}"] if ctx.image_findings else []),
            _OUTPUT_FORMAT,
        ]
    )
    return [
        LLMMessage(role="system", content=_system_prompt(ctx, config)),
        LLMMessage(role="user", content=user),
    ]


def build_generation_request(
    ctx: ConversationContext,
    route: RouteDecision,
    evidence: EvidenceBundle,
    *,
    variant: PromptVariant = "direct",
    config: GenerationConfig,
    retry_feedback: list[str] | None = None,
    focus_intent: Intent | None = None,
) -> LLMRequest:
    """生成请求：purpose="generate"、JSON 输出，成本归属当前店铺（§12.4）。

    单次调用的超时不超过当前截止时间的剩余时间（core.budget.clamp_timeout，§12.1）；
    没有截止时间时就是 GenerationConfig.timeout_s。"""
    return LLMRequest(
        purpose="generate",
        messages=build_generation_messages(
            ctx,
            route,
            evidence,
            variant=variant,
            config=config,
            retry_feedback=retry_feedback,
            focus_intent=focus_intent,
        ),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        json_mode=True,
        timeout_s=clamp_timeout(config.timeout_s),
        shop_id=ctx.shop_id,
    )
