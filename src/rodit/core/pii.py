"""个人信息最小化（方案 §5.3、§12.3）。

- mask_pii：任何要进大模型提示词、判断层请求、审计日志的买家文本都先过它。
  买家昵称、手机号、收件地址不放进提示词；这里遮掉买家自己打出来的手机号、证件号、邮箱、
  银行卡号、微信号 / QQ 号和收件地址（街道门牌、小区楼栋、门牌后的楼宇名和房号）。
  手机号不论怎么分组（空格、短横、点）、写成中文数字（"一三八……"）都算。
  平台订单号、运单号不是个人信息，而且回答要用，保留原样。
  省、市这一级的地名保留（"新疆能到吗"要靠它回答），只遮能定位到人的部分。
- BuyerRefHasher / hash_buyer_id：buyer_ref = 平台买家 ID 的哈希（§8.3），不存昵称。
  盐来自部署环境变量（PrivacyConfig.buyer_salt_env），仓库里没有真实值。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import unicodedata
from datetime import date

PHONE_MASK = "[手机号]"
ID_MASK = "[证件号]"
EMAIL_MASK = "[邮箱]"
CARD_MASK = "[卡号]"
ADDRESS_MASK = "[地址]"
CONTACT_MASK = "[联系方式]"

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_LONG_DIGITS = re.compile(r"(?<![0-9A-Za-z])\d{15,19}[0-9Xx]?(?![0-9A-Za-z])")
# 订单号 / 运单号：前面 8 个字符内有这些词的长数字串不当个人信息处理
_ORDER_HINT = re.compile(r"(?:订单|单号|运单|快递单|物流单|编号|交易号|售后单)[号码]?[\s:：#]*$")
# 手机号：11 位，数字之间可以有一个空格、点或短横（3-4-4、4-3-4、两两分组……都算）
_MOBILE = re.compile(r"(?<!\d)(?:\+?86[\s-]?)?1[3-9](?:[\s.-]?\d){9}(?!\d)")
_LANDLINE = re.compile(r"(?<!\d)(?:0\d{2,3}[-\s]\d{7,8}|\(0\d{2,3}\)\s?\d{7,8})(?!\d)")
# 微信号 / QQ 号：保留关键词（路由要靠它识别站外联系），只遮号码本身
# "微 信"（中间有空格）、"薇"、"v：" 也是常见的规避写法
_IM_ID = re.compile(
    r"((?:微\s*信|weixin|wechat|威信|薇信|薇|v\s*信|vx|wx|(?<![A-Za-z])v(?=\s*[:：])|QQ|qq|扣扣)"
    r"号?\s*(?:是|为)?\s*[:：]?\s*)"
    r"([A-Za-z][-_A-Za-z0-9]{5,19}|\d{5,11})(?![-_A-Za-z0-9])",
    re.IGNORECASE,
)

# 地址：必须有"街道 + 门牌号"或"小区 / 大厦 + 楼栋"，然后向前吸收省市区，向后吸收楼宇名和楼栋单元室
_NUM = r"\d+(?:-\d+)?"
_CN_ORDINAL = r"[一二三四五六七八九十\d]+"
_STREET_CORE = re.compile(
    rf"[一-龥A-Za-z0-9]{{1,8}}?(?:路|街|巷|弄|大道|胡同)(?:[东西南北中]段)?\s*{_NUM}\s*号(?:院|楼)?"
)
# 小区 / 楼盘：关键词明确的（小区、花园……村）后面跟"数字 + 栋 / 号 / 组"；
# 关键词容易出现在日常说法里的（城、里、一区、3 期）只认"栋 / 幢 / 座 / 组 / 单元"，
# 不认"号"——"家里 3 号有人""第一期 5 号开始"是日期，不是地址。
_COMMUNITY_CORE = re.compile(
    rf"[一-龥A-Za-z0-9]{{1,8}}?(?:"
    rf"(?:小区|花园|公寓|大厦|家园|新村|苑|广场|号院|(?<![农乡])村)\s*{_NUM}\s*"
    r"(?:栋|幢|号楼|座|号|组)"
    rf"|(?:城|里|{_CN_ORDINAL}(?:区|期))\s*{_NUM}\s*(?:栋|幢|号楼|座|组|单元))"
)
# 门牌号 / 楼栋后面的楼宇名（"…广场""…大厦"，最多 8 个字，遇到语气词、人称就不算）
_BUILDING = (
    r"(?:(?![吗呢吧啊呀哦了我你想要请])[一-龥A-Za-z0-9]){0,8}?"
    r"(?:大厦|广场|中心|大楼|写字楼|公寓|花园|小区|苑|园|城)"
)
_UNIT = rf"(?:{_NUM}\s*(?:栋|幢|号楼|座|单元|层|楼|室|号|组|期|区)|[A-Za-z]\d*\s*(?:座|区|栋))"
_ADDRESS_TAIL = re.compile(rf"^(?:\s*{_BUILDING})?(?:\s*{_UNIT})*(?:\s*\d{{3,4}}(?!\d))?(?:\s*室)?")
# 地址前面常见的引导词：不属于地址。_CUT_STOP 用来剪掉被正则吃进来的引导词（不含空白：
# 空白只会出现在门牌号附近，属于地址本身）；_EXPAND_STOP 用来决定向前吸收省市区时在哪里停。
_CUT_STOP = set("改址到在住是成寄送发收为我的给往换填写：:，,。")
_EXPAND_STOP = _CUT_STOP | set(" \t\n\u3000")
_MAX_PREFIX = 12


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK = "10X98765432"


def _id_card_ok(s: str) -> bool:
    if len(s) != 18 or not s[:17].isdigit():
        return False
    try:
        date(int(s[6:10]), int(s[10:12]), int(s[12:14]))
    except ValueError:
        return False
    if not 1900 <= int(s[6:10]) <= 2100:
        return False
    check = _ID_CHECK[sum(int(c) * w for c, w in zip(s[:17], _ID_WEIGHTS, strict=True)) % 11]
    return s[17].upper() == check


_Span = tuple[int, int, str]  # 归一化文本上的 [start, end) 和替换文字

_CN_DIGITS = dict(zip("零〇幺一二三四五六七八九", "001123456789", strict=True))
_CN_DIGIT_RUN = re.compile(r"[零〇幺一二三四五六七八九](?:[\s.-]?[零〇幺一二三四五六七八九]){6,}")


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """逐字做 NFKC 归一，记录归一化文本每个字符对应的原文位置（用于把遮盖映射回原文）。"""
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch):
            chars.append(c)
            index.append(i)
    # 连续 7 个以上的中文数字（如"一三八一二三四五六七八"）逐字换成阿拉伯数字：
    # 一字对一字，位置映射不变
    for m in _CN_DIGIT_RUN.finditer("".join(chars)):
        for j in range(m.start(), m.end()):
            chars[j] = _CN_DIGITS.get(chars[j], chars[j])
    return "".join(chars), index


def _long_digit_spans(norm: str) -> list[_Span]:
    spans: list[_Span] = []
    for m in _LONG_DIGITS.finditer(norm):
        s = m.group(0)
        if _ORDER_HINT.search(norm[max(0, m.start() - 8) : m.start()]):
            continue  # 订单号、运单号保留
        if len(s) == 18 and _id_card_ok(s):
            spans.append((m.start(), m.end(), ID_MASK))
        elif s.isdigit() and 16 <= len(s) <= 19 and _luhn_ok(s):
            spans.append((m.start(), m.end(), CARD_MASK))
    return spans


def _address_spans(norm: str) -> list[_Span]:
    raw: list[tuple[int, int]] = []
    for pattern in (_STREET_CORE, _COMMUNITY_CORE):
        for m in pattern.finditer(norm):
            start, end = m.start(), m.end()
            # 剪掉被正则吃进来的引导词（如"住幸福里小区""地址到西湖区"）：只看门牌数字之前的部分
            core = norm[start:end]
            digit = re.search(r"\d", core)
            limit = digit.start() if digit else len(core)
            cut = max((i for i, ch in enumerate(core[:limit]) if ch in _CUT_STOP), default=-1)
            start += cut + 1
            # 向前吸收省、市、区等（最多 _MAX_PREFIX 个字，遇到引导词或空白停）
            steps = 0
            while start > 0 and steps < _MAX_PREFIX:
                ch = norm[start - 1]
                if ch in _EXPAND_STOP or not ("\u4e00" <= ch <= "\u9fa5"):
                    break
                start -= 1
                steps += 1
            tail = _ADDRESS_TAIL.match(norm[end:])
            if tail:
                end += tail.end()
            raw.append((start, end))
    raw.sort()
    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(s, e, ADDRESS_MASK) for s, e in merged]


def _find_spans(norm: str) -> list[_Span]:
    """按优先级收集要遮的片段，重叠的只保留优先级高的：
    邮箱 → 微信 / QQ 号 → 长数字（证件号、银行卡；订单号保留）→ 手机 → 座机 → 地址。"""
    groups: list[list[_Span]] = [
        [(m.start(), m.end(), EMAIL_MASK) for m in _EMAIL.finditer(norm)],
        [(m.start(2), m.end(2), CONTACT_MASK) for m in _IM_ID.finditer(norm)],
        _long_digit_spans(norm),
        [(m.start(), m.end(), PHONE_MASK) for m in _MOBILE.finditer(norm)],
        [(m.start(), m.end(), PHONE_MASK) for m in _LANDLINE.finditer(norm)],
        _address_spans(norm),
    ]
    accepted: list[_Span] = []
    for group in groups:
        for span in group:
            if all(span[1] <= a[0] or span[0] >= a[1] for a in accepted):
                accepted.append(span)
    return sorted(accepted)


def mask_pii(text: str) -> str:
    """遮掉文本里的个人信息。

    匹配在 NFKC 归一化后的文本上做（全角数字、字母也能识别），替换落回原文：
    没有被遮的部分保持原样（中文全角标点不变），同一段文本重复调用结果不变。
    """
    if not text:
        return text
    norm, index = _normalize_with_map(text)
    spans = _find_spans(norm)
    if not spans:
        return text
    out: list[str] = []
    last = 0
    for s, e, token in spans:
        orig_start, orig_end = index[s], index[e - 1] + 1
        out.append(text[last:orig_start])
        out.append(token)
        last = orig_end
    out.append(text[last:])
    return "".join(out)


def contains_pii(text: str) -> bool:
    """文本里是否有会被 mask_pii 遮掉的个人信息。"""
    if not text:
        return False
    norm, _ = _normalize_with_map(text)
    return bool(_find_spans(norm))


# ---------------------------------------------------------------- buyer_ref


def hash_buyer_id(platform: str, raw_buyer_id: str, *, salt: str) -> str:
    """buyer_ref：同一平台内稳定、不可逆。形如 "b_" + 24 位十六进制。"""
    if not salt:
        raise ValueError("buyer_ref 哈希必须提供盐")
    digest = hmac.new(salt.encode(), f"{platform}:{raw_buyer_id}".encode(), hashlib.sha256)
    return "b_" + digest.hexdigest()[:24]


class BuyerRefHasher:
    """接入层和平台工具共用的 buyer_ref 生成器（按依赖注入传递，保证两边用同一个盐）。

    平台工具按买家查询时（最近订单、R7 售后次数），实现方在平台云内维护 buyer_ref → 平台买家 ID
    的映射，或对拉到的记录用同一个 hasher 计算后再过滤；映射不出单元。
    """

    def __init__(self, salt: str) -> None:
        if not salt:
            raise ValueError("buyer_ref 哈希必须提供盐")
        self._salt = salt

    def __repr__(self) -> str:  # 不在日志里暴露盐
        return "BuyerRefHasher(salt=***)"

    def __call__(self, platform: str, raw_buyer_id: str) -> str:
        return hash_buyer_id(platform, raw_buyer_id, salt=self._salt)

    @classmethod
    def from_env(cls, env_name: str = "RODIT_BUYER_SALT") -> BuyerRefHasher:
        salt = os.environ.get(env_name, "")
        if not salt:
            raise RuntimeError(f"环境变量 {env_name} 未设置：生产部署必须配置 buyer_ref 的盐")
        return cls(salt)
