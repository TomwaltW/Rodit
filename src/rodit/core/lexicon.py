"""共享词表（方案 §2.1 话术禁区、§5.5 硬规则、§6.6 规则 1）。

规则路由（识别买家的站外交易请求）和答案质检（拦截回答里的站外引流、绝对化用语）用同一份词表，
避免两边各维护一份、互相不一致。商家补充的禁用词在 ShopConfig.banned_words，不在这里。
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------- 站外联系 / 站外交易（§2.1）

# 强信号：站外联系渠道本身。买家消息、回答里出现都算
_CONTACT_STRONG = (
    r"微信|weixin|wechat|威信|薇信|徽信|v\s*信|vx|(?<![a-z])wx(?![a-z])",
    r"加\s*我|加\s*(?:个|一下)?\s*(?:好友|微|v(?![a-z])|q(?![a-z])|扣)",
    r"(?<![a-z])qq(?![a-z])|扣扣|企鹅号",
    r"二维码|扫码|收款码",
    r"https?://|www\.|\.(?:com|cn|net)\b",
    r"(?<!\d)1[3-9]\d(?:[\s.-]?\d{4}){2}(?!\d)",  # 手机号码本身
)
# 弱信号：手机号、电话、联系方式这类词本身有正常用法（"能改个手机号吗""快递员没打电话"），
# 回答里出现算引导（客服不该索要或提供），买家消息里只在"留 / 给 / 加 / 要 + 联系方式"时算请求
_CONTACT_WORDS = r"手机号|电话号码?|打电话|联系方式"
_CONTACT_SOLICIT = (
    r"(?:留|给|发|加|要|换|报)\s*(?:个|一下|下)?\s*(?:你的|您的|你们的|我的|我)?\s*"
    r"(?:手机号?|电话(?:号码?)?|联系方式)"
    r"|(?:你的|您的|你们的)\s*(?:手机号?|电话(?:号码?)?|联系方式)"
)

# 回答侧：强信号 + 联系方式词语（§5.5 站外引流）
OFF_PLATFORM_CONTACT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (*_CONTACT_STRONG, _CONTACT_WORDS)
)
# 买家侧：强信号 + 索要 / 交换联系方式（§5.1 off_platform、§6.6 规则 1）
OFF_PLATFORM_REQUEST_CONTACT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (*_CONTACT_STRONG, _CONTACT_SOLICIT)
)

# 站外交易：买家消息里出现说明在要站外交易（§5.1 off_platform）；回答里只在未被否定时拦截
OFF_PLATFORM_TRADE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (r"支付宝|转账|私下|线下交易|站外|私聊|红包返现|返现|直接打钱",)
)

_NEGATION = re.compile(r"(?:不|别|勿|禁止|不能|不要|无法|不可以|不支持|不会)[^，。！？,.!?]{0,8}$")

# core.pii.mask_pii 插入的占位符（"[手机号]""[联系方式]"……）不是买家或客服写的内容，
# 匹配前先去掉，否则"改地址，电话 [手机号]"会被当成站外联系。
_MASK_PLACEHOLDER = re.compile(r"\[(?:手机号|证件号|邮箱|卡号|地址|联系方式)\]")


def _norm(text: str) -> str:
    return _MASK_PLACEHOLDER.sub(" ", unicodedata.normalize("NFKC", text or ""))


def find_off_platform_request(text: str) -> list[str]:
    """买家消息里的站外联系 / 交易请求片段（规则路由 §6.6 规则 1 用）。

    文本可以是原文或 mask_pii 之后的文本：遮盖占位符不算命中；原文里的手机号码本身算
    （买家主动报手机号、要求站外联系）。
    """
    norm = _norm(text)
    hits: list[str] = []
    for pattern in (*OFF_PLATFORM_REQUEST_CONTACT_PATTERNS, *OFF_PLATFORM_TRADE_PATTERNS):
        hits.extend(m.group(0) for m in pattern.finditer(norm))
    return hits


def find_off_platform_in_answer(text: str) -> list[str]:
    """回答里的站外引流片段（质检硬规则 §5.5 用）。

    联系方式一律拦截（即使是否定句，回答里也不该出现微信号、链接、手机号）；
    "联系方式"一词和站外交易类词语在被否定时放行，如标准话术"平台规定不能通过站外联系和交易哦"（§11.4）。
    """
    norm = _norm(text)
    hits: list[str] = []
    for pattern in OFF_PLATFORM_CONTACT_PATTERNS:
        for m in pattern.finditer(norm):
            if m.group(0) in {"联系方式"} and _NEGATION.search(norm[: m.start()]):
                continue
            hits.append(m.group(0))
    for pattern in OFF_PLATFORM_TRADE_PATTERNS:
        for m in pattern.finditer(norm):
            if _NEGATION.search(norm[: m.start()]):
                continue
            hits.append(m.group(0))
    return hits


# ---------------------------------------------------------------- 绝对化用语（§5.5、§12.3 广告法）

ABSOLUTE_TERMS: tuple[str, ...] = (
    "最",
    "第一",
    "绝对",
    "100%",
    "百分百",
    "顶级",
    "极致",
    "唯一",
    "万能",
    "永久",
    "保证",
    "一定",
    "肯定",
)

ABSOLUTE_TERM_EXCEPTIONS: tuple[str, ...] = (
    # 时间、顺序、数量意义上的"最"，不是宣传用语
    "最新",
    "最近",
    "最后",
    "最晚",
    "最早",
    "最多",
    "最少",
    "最长",
    "最短",
    "最迟",
    "最低消费",
    # "第一"的非比较用法
    "第一时间",
    "第一次",
    "第一件",
    "第一步",
    # 否定或程度用法
    "不一定",
    "一定程度",
    "不保证",
)


def find_absolute_terms(
    text: str,
    *,
    terms: tuple[str, ...] | list[str] = ABSOLUTE_TERMS,
    exceptions: tuple[str, ...] | list[str] = ABSOLUTE_TERM_EXCEPTIONS,
) -> list[str]:
    """找出绝对化用语：先把例外词组整体遮掉，再做子串匹配。返回命中的词（按出现顺序，去重）。"""
    norm = _norm(text)
    for ex in sorted(exceptions, key=len, reverse=True):
        norm = norm.replace(unicodedata.normalize("NFKC", ex), "\u0000" * len(ex))
    found: list[tuple[int, str]] = []
    for term in terms:
        t = unicodedata.normalize("NFKC", term)
        start = norm.find(t)
        if start >= 0:
            found.append((start, term))
    seen: set[str] = set()
    out: list[str] = []
    for _, term in sorted(found):
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out
