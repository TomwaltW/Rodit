"""个人信息最小化（方案 §5.3、§12.3）。

- mask_pii：任何要进大模型提示词、判断层请求、审计日志的买家文本都先过它。
  买家昵称、手机号、收件地址不放进提示词；这里遮掉买家自己打出来的手机号、证件号、邮箱、
  银行卡号和明显的收件地址。
- hash_buyer_id：buyer_ref = 平台买家 ID 的哈希（§8.3），不存昵称。盐来自部署配置，
  仓库里只有开发用的默认值。
"""

from __future__ import annotations

import hashlib
import hmac
import re

PHONE_MASK = "[手机号]"
ID_MASK = "[证件号]"
EMAIL_MASK = "[邮箱]"
CARD_MASK = "[卡号]"
ADDRESS_MASK = "[地址]"

_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_BANK_CARD = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
# 大陆手机号，允许 +86 / 86 前缀和空格、短横分隔
_MOBILE = re.compile(r"(?<!\d)(?:\+?86[\s-]?)?1[3-9]\d(?:[\s-]?\d{4}){2}(?!\d)")
_LANDLINE = re.compile(r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# 收件地址：省 / 市 / 区县后面跟着路、街、号、小区、栋、室等
_ADDRESS = re.compile(
    r"[一-龥]{2,}(?:省|自治区|市)[一-龥\w\s-]{0,30}?"
    r"(?:路|街|道|巷|弄|号|小区|大厦|栋|幢|单元|室)[一-龥\w-]{0,20}"
)

_DEV_SALT = "rodit-dev-salt-not-for-production"


def mask_pii(text: str) -> str:
    """遮掉文本里的个人信息。顺序很重要：先长后短，避免证件号被当成手机号。"""
    if not text:
        return text
    out = _EMAIL.sub(EMAIL_MASK, text)
    out = _ID_CARD.sub(ID_MASK, out)
    out = _BANK_CARD.sub(CARD_MASK, out)
    out = _MOBILE.sub(PHONE_MASK, out)
    out = _LANDLINE.sub(PHONE_MASK, out)
    out = _ADDRESS.sub(ADDRESS_MASK, out)
    return out


def contains_pii(text: str) -> bool:
    return mask_pii(text) != text


def hash_buyer_id(platform: str, raw_buyer_id: str, *, salt: str = _DEV_SALT) -> str:
    """buyer_ref：同一平台内稳定、不可逆。形如 "b_7f3a91c2d4e5"。"""
    digest = hmac.new(salt.encode(), f"{platform}:{raw_buyer_id}".encode(), hashlib.sha256)
    return "b_" + digest.hexdigest()[:12]
