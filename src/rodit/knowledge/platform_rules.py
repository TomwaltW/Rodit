"""平台规则（方案 §1.4、§2.1、§5.1、§7.3.1、§11.4）：按平台维护的静态文字，作为 PLATFORM 证据。

平台规则是硬约束，店铺政策不能覆盖（EvidenceSource.PLATFORM 优先级 2，排在实时数据之后）。
内容是合成的、按方案调研结论写的通用说法：

- 申请售后：售后只能由买家在订单页 → 申请售后发起，客服不能替买家发起（§7.3.1、§1.4 非目标）；
- 退款到账：以平台规则为准，只说"按原支付方式退回、到账时间因支付方式而异、看退款详情"，
  不写具体天数，不做任何到账承诺（§5.4 规则 2）；
- 改单：发货前可以在订单页修改地址或请客服协助，发货后不能再改；
- 站外交易：平台禁止站外联系和交易（§2.1 话术禁区、§11.4 标准话术）。

这些文字会进大模型提示词、也可能被原样引用进回答，所以：不含绝对化用语（§5.5）、
不出现任何站外联系方式或支付渠道名称（core.lexicon 的回答检查一律拦截）。
平台细节没有核实的（订单页入口的按钮名称、各支付方式到账时长等），把【待确认】事项记在
pending 里（对应附录 A 需要向平台确认的问题），不写进 text，避免被引用给买家。
"""

from __future__ import annotations

from dataclasses import dataclass

from rodit.core import evidence as ev
from rodit.core.enums import EvidenceSource, Platform
from rodit.core.evidence import PlatformRuleKey
from rodit.core.models import Evidence


@dataclass(frozen=True)
class PlatformRuleText:
    platform: Platform
    key: PlatformRuleKey
    text: str  # 给大模型看、可以引用给买家的文字
    pending: str | None = None  # 【待确认】事项，只给维护者看


_K = PlatformRuleKey
_P = Platform

_APPLY_TAIL = (
    "选择售后类型、按页面提示填写原因（质量问题请上传照片）后提交。客服不能替买家发起售后。"
)
_REFUND_TIMING = (
    "退款到账时间以平台规则为准：商家同意退款后，款项按原支付方式退回，"
    "不同支付方式的到账时间不同，具体进度可以在订单的退款详情里查看。"
)
_MODIFY_ORDER = (
    "订单发货前，买家可以在订单页修改收货地址，或请客服协助修改；"
    "订单发货后，收货地址和规格都不能再修改。发货前需要取消的，可以在订单页申请退款。"
)

_RULES: tuple[PlatformRuleText, ...] = (
    # ---- 申请售后（§7.3.1 "给出该平台订单页 → 申请售后的操作指引（平台只允许买家发起）"）
    PlatformRuleText(
        _P.TAOBAO,
        _K.APPLY_AFTERSALE,
        "售后只能由买家本人在订单页发起：在淘宝「我的淘宝 → 我的订单」里找到这笔订单，"
        "点「申请售后」（或「退款 / 售后」），" + _APPLY_TAIL,
        pending="【待确认】淘宝订单页售后入口的名称以 App 当前版本为准",
    ),
    PlatformRuleText(
        _P.DOUYIN,
        _K.APPLY_AFTERSALE,
        "售后只能由买家本人在订单页发起：在抖音「我 → 订单」里找到这笔订单，"
        "点「申请售后」，" + _APPLY_TAIL,
        pending="【待确认】抖音 App 订单入口路径和售后按钮名称",
    ),
    PlatformRuleText(
        _P.PDD,
        _K.APPLY_AFTERSALE,
        "售后只能由买家本人在订单页发起：在拼多多「个人中心 → 我的订单」里找到这笔订单，"
        "点「申请售后」（或「退款 / 售后」），" + _APPLY_TAIL,
        pending="【待确认】拼多多订单页售后入口的名称以 App 当前版本为准",
    ),
    # ---- 退款到账（§7.3.1 "说明当前进度和平台的一般到账规则"；不做到账承诺）
    PlatformRuleText(
        _P.TAOBAO,
        _K.REFUND_TIMING,
        _REFUND_TIMING,
        pending="【待确认】淘宝各支付方式的一般到账时长（确认后才能写进规则文字）",
    ),
    PlatformRuleText(
        _P.DOUYIN,
        _K.REFUND_TIMING,
        _REFUND_TIMING,
        pending="【待确认】抖店各支付方式的一般到账时长（确认后才能写进规则文字）",
    ),
    PlatformRuleText(
        _P.PDD,
        _K.REFUND_TIMING,
        _REFUND_TIMING,
        pending="【待确认】拼多多各支付方式的一般到账时长（确认后才能写进规则文字）",
    ),
    # ---- 改地址 / 改规格 / 取消（§5.1 modify_order：订单工具 + 平台规则）
    PlatformRuleText(
        _P.TAOBAO,
        _K.MODIFY_ORDER,
        _MODIFY_ORDER,
        pending="【待确认】淘宝买家自助改地址是否需要商家同意、改规格是否只能由商家操作",
    ),
    PlatformRuleText(
        _P.DOUYIN,
        _K.MODIFY_ORDER,
        _MODIFY_ORDER,
        pending="【待确认】抖店买家自助改地址的时限和条件",
    ),
    PlatformRuleText(
        _P.PDD,
        _K.MODIFY_ORDER,
        _MODIFY_ORDER,
        pending="【待确认】拼多多买家自助修改收货信息的时限和条件",
    ),
    # ---- 站外交易（§2.1 话术禁区、§11.4 标准话术）
    PlatformRuleText(
        _P.TAOBAO,
        _K.OFF_PLATFORM,
        "平台规定不能通过站外联系和交易：沟通、下单和付款都请在淘宝内完成，"
        "订单和售后才有平台保障。",
        pending="【待确认】淘宝站外引流条款原文（§2.1 未查到）",
    ),
    PlatformRuleText(
        _P.DOUYIN,
        _K.OFF_PLATFORM,
        "平台规定不能通过站外联系和交易：沟通请在抖店客服里进行，下单和付款都在抖音内完成，"
        "订单和售后才有平台保障。",
    ),
    PlatformRuleText(
        _P.PDD,
        _K.OFF_PLATFORM,
        "平台规定不能通过站外联系和交易，也不能站外收付款：沟通、下单和付款都请在拼多多内完成，"
        "订单和售后才有平台保障。",
    ),
)

PLATFORM_RULES: dict[tuple[Platform, PlatformRuleKey], PlatformRuleText] = {
    (r.platform, r.key): r for r in _RULES
}
"""(平台, 规则键) → 规则文字。每个平台 × 每个 PlatformRuleKey 都有一条（测试保证）。"""


def find_platform_rule(
    platform: Platform | str, key: PlatformRuleKey | str
) -> PlatformRuleText | None:
    try:
        return PLATFORM_RULES.get((Platform(platform), PlatformRuleKey(key)))
    except ValueError:
        return None


def platform_rule_text(platform: Platform | str, key: PlatformRuleKey | str) -> str:
    """规则文字；没有这条规则时抛 KeyError。"""
    rule = find_platform_rule(platform, key)
    if rule is None:
        raise KeyError(f"没有平台规则：{platform}:{key}")
    return rule.text


def platform_rule_evidence(platform: Platform | str, key: PlatformRuleKey | str) -> Evidence:
    """平台规则证据（source=PLATFORM，data={"platform","key","text"}）。没有这条规则时抛 KeyError。

    topic 为空：平台规则和同名的店铺政策（如 modify_order）是互补关系（平台底线 + 店铺做法），
    不能互相覆盖。
    """
    rule = find_platform_rule(platform, key)
    if rule is None:
        raise KeyError(f"没有平台规则：{platform}:{key}")
    return Evidence(
        id=ev.platform_rule(rule.platform, rule.key),
        source=EvidenceSource.PLATFORM,
        label=ev.platform_rule_label(rule.platform, rule.key),
        content=rule.text,
        data={"platform": rule.platform.value, "key": rule.key.value, "text": rule.text},
    )


def pending_platform_rules() -> list[PlatformRuleText]:
    """带【待确认】事项的规则（Phase 0 向平台逐条确认用，附录 A）。"""
    return [r for r in _RULES if r.pending]
