"""个人信息最小化（方案 §5.3、§12.3）。

- mask_pii：任何要进大模型提示词、判断层请求、审计日志的买家文本都先过它。
  买家昵称、手机号、收件地址不放进提示词；这里遮掉买家自己打出来的手机号、证件号、邮箱、
  银行卡号、微信号 / QQ 号和收件地址（街道门牌、小区楼栋）。
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
_MOBILE = re.compile(r"(?<!\d)(?:\+?86[\s-]?)?1[3-9]\d(?:[\s.-]?\d{4}){2}(?!\d)")
_LANDLINE = re.compile(r"(?<!\d)(?:0\d{2,3}[-\s]\d{7,8}|\(0\d{2,3}\)\s?\d{7,8})(?!\d)")
# 微信号 / QQ 号：保留关键词（路由要靠它识别站外联系），只遮号码本身
_IM_ID = re.compile(
    r"((?:微信|weixin|wechat|威信|薇信|v\s*信|vx|wx|QQ|qq|扣扣)号?\s*(?:是|为)?\s*[:：]?\s*)"
    r"([A-Za-z][-_A-Za-z0-9]{5,19}|\d{5,11})(?![-_A-Za-z0-9])",
    re.IGNORECASE,
)

# 地址：必须有"街道 + 门牌号"或"小区 / 大厦 + 楼栋"，然后向前吸收省市区，向后吸收楼栋单元室
_NUM = r"\d+(?:-\d+)?"
_STREET_CORE = re.compile(
    rf"[一-龥A-Za-z0-9]{{1,8}}?(?:路|街|巷|弄|大道|胡同)\s*{_NUM}\s*号(?:院|楼)?"
)
_COMMUNITY_CORE = re.compile(
    rf"[一-龥A-Za-z0-9]{{1,8}}?(?:小区|花园|公寓|大厦|家园|新村|苑|广场)\s*{_NUM}\s*"
    r"(?:栋|幢|号楼|座|号)"
)
_ADDRESS_TAIL = re.compile(
    rf"^(?:\s*(?:{_NUM}\s*(?:栋|幢|号楼|座|单元|层|楼|室|号)|[A-Za-z]\s*座))+(?:\s*\d{{3,4}}(?!\d))?"
)
# 地址前面常见的引导词：不属于地址，向前吸收时遇到就停
_ADDRESS_STOP = set("改址到在住是成寄送发收为我的给往换填写：:，,。 \t\n")
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


def _mask_long_digits(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        s = m.group(0)
        if _ORDER_HINT.search(text[max(0, m.start() - 8) : m.start()]):
            return s  # 订单号、运单号保留
        if len(s) == 18 and _id_card_ok(s):
            return ID_MASK
        if s.isdigit() and 16 <= len(s) <= 19 and _luhn_ok(s):
            return CARD_MASK
        return s

    return _LONG_DIGITS.sub(repl, text)


def _mask_addresses(text: str) -> str:
    spans: list[tuple[int, int]] = []
    for pattern in (_STREET_CORE, _COMMUNITY_CORE):
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            # 去掉被正则吃进来的引导词（如"住幸福里小区""地址到西湖区"）：从最后一个引导字之后开始
            core = text[start:end]
            cut = max((i for i, ch in enumerate(core[:9]) if ch in _ADDRESS_STOP), default=-1)
            start += cut + 1
            # 向前吸收省、市、区等（最多 _MAX_PREFIX 个字，遇到引导词停）
            steps = 0
            while start > 0 and steps < _MAX_PREFIX:
                ch = text[start - 1]
                if ch in _ADDRESS_STOP or not ("一" <= ch <= "龥"):
                    break
                start -= 1
                steps += 1
            tail = _ADDRESS_TAIL.match(text[end:])
            if tail:
                end += tail.end()
            spans.append((start, end))
    if not spans:
        return text
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    out, last = [], 0
    for s, e in merged:
        out.append(text[last:s])
        out.append(ADDRESS_MASK)
        last = e
    out.append(text[last:])
    return "".join(out)


def mask_pii(text: str) -> str:
    """遮掉文本里的个人信息，返回 NFKC 归一后的文本（全角数字、字母会变成半角）。

    顺序：邮箱 → 微信 / QQ 号 → 长数字（证件号、银行卡；订单号保留）→ 手机 → 座机 → 地址。
    """
    if not text:
        return text
    out = unicodedata.normalize("NFKC", text)
    out = _EMAIL.sub(EMAIL_MASK, out)
    out = _IM_ID.sub(lambda m: m.group(1) + CONTACT_MASK, out)
    out = _mask_long_digits(out)
    out = _MOBILE.sub(PHONE_MASK, out)
    out = _LANDLINE.sub(PHONE_MASK, out)
    out = _mask_addresses(out)
    return out


def contains_pii(text: str) -> bool:
    """文本里是否有会被 mask_pii 遮掉的个人信息（忽略 NFKC 归一带来的变化）。"""
    norm = unicodedata.normalize("NFKC", text or "")
    return mask_pii(norm) != norm


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
