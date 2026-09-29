"""类目问题清单与知识缺失检查（方案 §5.2.3 冷启动第 3 步 "按类目给出常见问题清单，
提示商家补齐缺的知识"、§5.8 "知识缺失 → 提醒商家补知识"）。

- 清单按类目分组（core 之外的分组规则见 knowledge.category）：服饰一份，其余用通用清单；
  每一项是买家常问的一个问题，以及回答它需要的店铺政策键 / 商品字段。
- find_gaps 按清单检查一家店：店铺政策按店铺主营类目的清单，商品字段按商品自己的类目
  （商品没有类目时用店铺类目；商品类目没有专门清单时用通用清单）。返回的缺失标记和检索时
  EvidenceBundle.missing 的写法一致（policy:{key}、item:{item_id}:{field}），
  可以直接写进 KnowledgeGap。
- 商品字段：attributes（有任一属性）、attr:{规范名}（按 knowledge.items.ATTRIBUTE_TOPICS 的
  属性名别名判断，"材质成分"也算面料）、size_chart、size_advice（均码商品不要求）、care、usage。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from rodit.core import evidence as ev
from rodit.core.enums import PolicyKey
from rodit.core.errors import KnowledgeUnavailable
from rodit.core.models import ItemCard
from rodit.core.ports import KnowledgeRepository
from rodit.knowledge.category import APPAREL, category_group
from rodit.knowledge.items import attribute_topic_of
from rodit.knowledge.questionnaire import question_for_policy
from rodit.knowledge.text import for_matching

_T = TypeVar("_T")


@dataclass(frozen=True)
class ChecklistEntry:
    """类目问题清单的一项：买家常问的问题 + 回答它需要的知识。"""

    question: str
    policy_keys: tuple[PolicyKey, ...] = ()
    item_fields: tuple[str, ...] = ()


_P = PolicyKey

_COMMON_POLICY_ENTRIES: tuple[ChecklistEntry, ...] = (
    ChecklistEntry("什么时候发货？预售怎么发？", (_P.SHIP_TIME, _P.PRESALE)),
    ChecklistEntry("发什么快递？能指定快递吗？", (_P.COURIER, _P.COURIER_CHOICE)),
    ChecklistEntry("偏远地区能发吗？要补运费吗？", (_P.REMOTE_AREAS,)),
    ChecklistEntry("支持七天无理由吗？哪些不能退？", (_P.SEVEN_DAY_RETURN, _P.NON_RETURNABLE)),
    ChecklistEntry("退货运费谁出？有运费险吗？", (_P.RETURN_SHIPPING, _P.SHIPPING_INSURANCE)),
    ChecklistEntry("收到有质量问题怎么办？", (_P.QUALITY_ISSUE,)),
    ChecklistEntry("能开发票吗？", (_P.INVOICE,)),
    ChecklistEntry("有赠品吗？", (_P.GIFT,)),
    ChecklistEntry("下单后能改地址、改规格吗？", (_P.MODIFY_ORDER,)),
    ChecklistEntry("有什么优惠？能便宜点吗？", (_P.PROMOTIONS, _P.PROMISE_BOUNDARY)),
    ChecklistEntry("客服几点在线？", (_P.SERVICE_HOURS,)),
)

APPAREL_CHECKLIST: tuple[ChecklistEntry, ...] = (
    *_COMMON_POLICY_ENTRIES,
    ChecklistEntry("尺码偏大还是偏小？", (_P.SIZE_TENDENCY,)),
    ChecklistEntry("这是什么面料？是纯棉的吗？", item_fields=("attr:面料",)),
    ChecklistEntry("我穿什么码？", item_fields=("size_chart", "size_advice")),
    ChecklistEntry("能机洗吗？怎么洗？", item_fields=("care",)),
)
"""服饰（女装、男装……）的问题清单。服饰一般没有保修，不要求 warranty。"""

GENERIC_CHECKLIST: tuple[ChecklistEntry, ...] = (
    *_COMMON_POLICY_ENTRIES,
    ChecklistEntry("保修多久？怎么报修？", (_P.WARRANTY,)),
    ChecklistEntry("这个是什么规格、参数？", item_fields=("attributes",)),
    ChecklistEntry("怎么使用、安装？", item_fields=("usage",)),
)
"""没有专门清单的类目用这一份。"""

CHECKLISTS: dict[str, tuple[ChecklistEntry, ...]] = {APPAREL: APPAREL_CHECKLIST}


def checklist_for(category: str | None) -> tuple[ChecklistEntry, ...]:
    """类目（如 "女装"、"女装/连衣裙"）对应的问题清单；没有专门清单时返回通用清单。"""
    return CHECKLISTS.get(category_group(category) or "", GENERIC_CHECKLIST)


def _required_policy_keys(checklist: tuple[ChecklistEntry, ...]) -> list[PolicyKey]:
    keys = {k for entry in checklist for k in entry.policy_keys}
    return [k for k in PolicyKey if k in keys]  # 按 PolicyKey 声明顺序


def _required_item_fields(checklist: tuple[ChecklistEntry, ...]) -> list[str]:
    out: list[str] = []
    for entry in checklist:
        for f in entry.item_fields:
            if f not in out:
                out.append(f)
    return out


ONE_SIZE_VALUES = frozenset({"均码", "f", "free", "freesize", "one size", "onesize"})


def _is_one_size(card: ItemCard) -> bool:
    """所有 SKU 的尺码都是均码：不需要尺码表和尺码建议。"""
    sizes = [s.props().get("尺码") for s in card.skus]
    return bool(sizes) and all(
        size is not None and for_matching(size).strip() in ONE_SIZE_VALUES for size in sizes
    )


def _field_missing(card: ItemCard, field: str) -> bool:
    if field == "attributes":
        return not any(v.strip() for v in card.attributes.values())
    if field.startswith("attr:"):
        wanted = field.removeprefix("attr:")
        return not any(
            value.strip()
            and (name == wanted or getattr(attribute_topic_of(name), "name", None) == wanted)
            for name, value in card.attributes.items()
        )
    if field in {"size_chart", "size_advice"} and _is_one_size(card):
        return False
    if field == "size_chart":
        return not card.size_chart
    if field == "size_advice":
        return not (card.size_advice or "").strip()
    if field == "care":
        return not card.care
    if field == "usage":
        return not card.usage
    raise ValueError(f"未知的商品字段：{field}")


def _repo_call(fn: Callable[..., _T], *args: object) -> _T:
    try:
        return fn(*args)
    except Exception as e:
        raise KnowledgeUnavailable(f"知识库不可用：{type(e).__name__}") from e


def find_gaps(repo: KnowledgeRepository, shop_id: str) -> list[str]:
    """一家店按类目问题清单缺哪些知识，返回缺失标记（先政策、后商品；商品按 item_id 升序）。

    店铺不存在抛 KeyError；仓储报错抛 KnowledgeUnavailable。
    """
    shop = _repo_call(repo.get_shop, shop_id)
    if shop is None:
        raise KeyError(shop_id)
    policies = _repo_call(repo.get_policies, shop_id)
    cards = _repo_call(repo.list_item_cards, shop_id)

    gaps: list[str] = []
    for key in _required_policy_keys(checklist_for(shop.category)):
        entry = policies.get(key.value)
        if entry is None or not entry.value.strip():
            gaps.append(ev.missing_policy(key.value))
    for card in sorted(cards, key=lambda c: c.item_id):
        checklist = checklist_for(card.category or shop.category)  # 商品没有类目时用店铺类目
        for field in _required_item_fields(checklist):
            if _field_missing(card, field):
                gaps.append(ev.missing_item_field(card.item_id, field))
    return gaps


_ITEM_FIELD_LABELS = {
    "attributes": "商品属性",
    "size_chart": "尺码表",
    "size_advice": "尺码建议",
    "care": "洗涤保养",
    "usage": "使用说明",
    "title": "标题",
}


def describe_gap(marker: str) -> str:
    """缺失标记 → 给商家看的说明，如 "店铺政策 · 发票（问卷第 7 题）"、"商品 i_777 · 尺码建议"。"""
    if marker == ev.MISSING_ITEM:
        return "会话里没有识别到当前商品"
    if marker.startswith("policy:"):
        key = marker.removeprefix("policy:")
        q = question_for_policy(key)
        suffix = f"（问卷第 {q.id} 题）" if q else ""
        return f"{ev.policy_label(key)}{suffix}"
    if marker.startswith("item:"):
        item_id, _, field = marker.removeprefix("item:").partition(":")
        if field.startswith("attr:"):
            label = f"属性 · {field.removeprefix('attr:')}"
        else:
            label = _ITEM_FIELD_LABELS.get(field, field)
        return f"当前商品 · {label}" if item_id == "?" else f"商品 {item_id} · {label}"
    return marker
