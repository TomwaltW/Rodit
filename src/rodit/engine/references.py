"""会话理解里的指代识别（方案 §5.3、§5.7）。

"这个 / 那件黑色的 / 刚买的那个 / 订单号 xxx / 没听懂"——都在买家原文（Turn.text）上识别：
订单号、规格值不能先脱敏再识别（core.pii.mask_pii 会改写长数字和全角字符）。
这里只做纯文本判断，不调用任何仓储或工具；查商品卡片、查订单由 ConversationBuilder 负责。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from rodit.core.models import ItemCard, Sku


def _norm(text: str | None) -> str:
    return unicodedata.normalize("NFKC", text or "")


# ---------------------------------------------------------------- 商品指代

# 指示代词："这个 / 这款 / 那件 / 这条……"；"它"排除"其它"
_DEMONSTRATIVE = re.compile(r"[这那](?:一)?(?:个|款|件|条|双|套|只|身|顶|副|瓶|盒)|(?<!其)它")


def has_demonstrative(text: str) -> bool:
    """买家是否用指示代词指商品（"这个有 M 码吗"），商品不止一个时要澄清（§5.2.4）。"""
    return bool(_DEMONSTRATIVE.search(_norm(text)))


def item_names(card: ItemCard) -> list[str]:
    """买家可能用来称呼这件商品的词：类目末级（"连衣裙""牛仔裤"），至少两个字。"""
    names: list[str] = []
    if card.category:
        leaf = card.category.split("/")[-1].strip()
        if len(leaf) >= 2:
            names.append(leaf)
    return names


def names_item(card: ItemCard, text: str) -> bool:
    """文本里是否点名了这件商品（按类目末级）。"""
    norm = _norm(text)
    return any(name in norm for name in item_names(card))


# ---------------------------------------------------------------- SKU 指代（"那件黑色的""L 码的"）

_ASCII_ALPHA = re.compile(r"^[A-Za-z]+$")
_DIGITS = re.compile(r"^\d+$")


def _value_pattern(value: str) -> re.Pattern[str]:
    """规格值在买家文本里的匹配方式。

    - 纯字母尺码（S / M / L / XL）：前后不能紧挨字母，忽略大小写（避免 "XL" 里匹配到 "L"、
      "cm" 里匹配到 "m"）；
    - 纯数字尺码（26 / 27）：必须带尺码语境——后面跟"码 / 的 / 寸 / 腰"，或前面是
      "尺码 / 码数 / 腰围 / 穿"，避免把"126 斤""27 号发货"（日期）当成尺码；
    - 其他（黑色、浅蓝、均码）：子串匹配；以"色"结尾的颜色也接受口语"黑的 / 黑款"。
    """
    v = _norm(value).strip()
    esc = re.escape(v)
    if _ASCII_ALPHA.match(v):
        return re.compile(rf"(?<![A-Za-z]){esc}(?![A-Za-z])", re.IGNORECASE)
    if _DIGITS.match(v):
        return re.compile(
            rf"(?<![0-9.]){esc}\s*(?:码|的|寸|腰)|(?:尺码|码数|腰围|穿)\s*{esc}(?![0-9])"
        )
    if len(v) >= 2 and v.endswith("色"):
        stem = re.escape(v[:-1])
        return re.compile(rf"{esc}|{stem}(?:的|款)")
    return re.compile(esc)


def value_mentioned(value: str, text: str) -> bool:
    if not value or not value.strip():
        return False
    return bool(_value_pattern(value).search(_norm(text)))


def match_skus(card: ItemCard, text: str) -> list[Sku] | None:
    """按文本里提到的规格值筛 SKU（§5.3 "那件黑色的""L 码的"）。

    返回和所有提到的规格值都吻合的 SKU（同一属性提到多个值时任一即可，
    例如"黑色和白色"会匹配多个 SKU，调用方据此判断不唯一）；
    文本里没有提到这件商品的任何规格值时返回 None。
    """
    mentioned: dict[str, set[str]] = {}
    for sku in card.skus:
        for name, value in sku.props().items():
            if value_mentioned(value, text):
                mentioned.setdefault(name, set()).add(value)
    if not mentioned:
        return None
    return [
        s
        for s in card.skus
        if all(s.props().get(name) in values for name, values in mentioned.items())
    ]


# 规格值当"指代"用：后面跟"的 / 那 / 这 / 款"（"黑色的""L 码的那件"），
# 或前面是"那件 / 这条……"（"那件黑色"）。"这个有 M 码吗"是在问有没有 M 码，不是指"M 码那件"。
_REF_AFTER = re.compile(r"^\s*(?:码|号|色)?\s*(?:的|那|这|款)")
_REF_BEFORE = re.compile(r"[这那]一?(?:个|款|件|条|双|套|只)\s*$")


def value_as_reference(value: str, text: str) -> bool:
    """文本里是否用这个规格值来指代商品（"那件黑色的""黑的那件""L 码的"）。"""
    if not value or not value.strip():
        return False
    norm = _norm(text)
    for m in _value_pattern(value).finditer(norm):
        if m.group(0).endswith(("的", "款")):
            return True
        if _REF_AFTER.match(norm[m.end() :]) or _REF_BEFORE.search(norm[: m.start()]):
            return True
    return False


def refers_by_sku(card: ItemCard, text: str) -> bool:
    """买家是否用这件商品的规格值指代它（多个商品时用来消歧，§5.3 "那件黑色的"）。"""
    matched = match_skus(card, text)
    if not matched:
        return False
    return any(value_as_reference(v, text) for s in matched for v in s.props().values())


# ---------------------------------------------------------------- 订单指代

# "刚买的那个 / 上次买的 / 最近一单"：指该买家在本店最近的订单
_RECENT_ORDER = re.compile(
    r"(?:刚刚?|刚才|上次|上回|最近|前两天|前几天|昨天|今天)\s*(?:买|拍|下单|下的单|付款|付的款)"
    r"|上一单|最近一单|最新的?(?:一单|订单)"
)

# 订单号：前面有"订单号 / 单号"的字母数字串，或和最近订单 ID 完全一致的字母数字串
_HINTED_ORDER = re.compile(
    r"(?:订单编号|订单号|订单|单号)\s*(?:是|为)?\s*[:：#]?\s*([A-Za-z0-9][A-Za-z0-9_-]{5,39})"
)
_TOKEN = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9][A-Za-z0-9_-]{3,39}(?![A-Za-z0-9_-])")


def refers_to_recent_order(text: str) -> bool:
    """买家是否在说"刚买的那个 / 上次买的"（§5.3）。"""
    return bool(_RECENT_ORDER.search(_norm(text)))


def find_known_order_id(text: str, order_ids: Iterable[str]) -> str | None:
    """文本里是否直接写了某个已知订单的 ID（多个时取最先出现的）。"""
    known = set(order_ids)
    if not known:
        return None
    for m in _TOKEN.finditer(_norm(text)):
        if m.group(0) in known:
            return m.group(0)
    return None


def hinted_order_ids(text: str) -> list[str]:
    """ "订单号 xxx"里的订单号（按出现顺序去重）。只从原文取，不能先脱敏。"""
    out: list[str] = []
    for m in _HINTED_ORDER.finditer(_norm(text)):
        oid = m.group(1)
        if oid not in out:
            out.append(oid)
    return out


# ---------------------------------------------------------------- 没听懂 / 答非所问（§5.7）

_CONFUSED = re.compile(
    r"没听懂|听不懂|没看懂|看不懂|答非所问|不是问这个|我不是问|不是这个意思|问的不是"
    r"|你在说什么|你说的什么|你在说啥|说的啥|牛头不对马嘴|驴唇不对马嘴|文不对题"
)


def expresses_confusion(text: str) -> bool:
    """买家是否在表示没听懂 / 答非所问（§5.7 连续两轮 → 提示客服接管）。"""
    return bool(_CONFUSED.search(_norm(text)))
