"""类目通用知识（方案 §5.2.1 优先级最低的来源 "洗涤保养、电器通用常识等，我们维护"）。

- 类目分组：商品类目（如 "女装/连衣裙"）或店铺主营类目（如 "女装"）取第一段映射到分组
  （服饰 / 家电）；映射不到的类目没有通用知识。
- 每条知识带问题关键词：只有买家问到相关内容（洗、晒、熨……）才作为依据，避免无关的通用
  知识占用证据长度。
- 每条知识带主题（care / usage）：商品卡片自己有洗涤保养、使用说明时，同主题的类目通用知识
  被覆盖（商品知识优先级更高，§5.2.1），只在商品没填时兜底。
"""

from __future__ import annotations

from dataclasses import dataclass

from rodit.core import evidence as ev
from rodit.core.enums import EvidenceSource
from rodit.core.models import Evidence

TOPIC_CARE = "care"
"""洗涤保养主题：商品卡片的 care 和类目通用的洗涤建议互相冲突时，以商品为准。"""

TOPIC_USAGE = "usage"
"""使用说明主题：商品卡片的 usage 和类目通用的使用建议互相冲突时，以商品为准。"""

APPAREL = "服饰"
APPLIANCE = "家电"

CATEGORY_GROUPS: dict[str, str] = {
    "女装": APPAREL,
    "男装": APPAREL,
    "童装": APPAREL,
    "服饰": APPAREL,
    "服装": APPAREL,
    "内衣": APPAREL,
    "家电": APPLIANCE,
    "小家电": APPLIANCE,
    "电器": APPLIANCE,
    "数码": APPLIANCE,
}


def category_group(category: str | None) -> str | None:
    """类目 → 分组；"女装/连衣裙" 取第一段 "女装"。映射不到返回 None。"""
    if not category:
        return None
    head = category.split("/", 1)[0].strip()
    return CATEGORY_GROUPS.get(head)


@dataclass(frozen=True)
class CategoryTip:
    key: str  # 证据 ID 用 category:{key}
    title: str  # 展示标签 "类目通用知识 · {title}"
    text: str
    keywords: tuple[str, ...]  # 买家问题里出现任一个才使用
    topic: str


_WASH_KEYWORDS = ("洗", "掉色", "褪色", "缩水", "保养", "打理")

CATEGORY_TIPS: dict[str, tuple[CategoryTip, ...]] = {
    APPAREL: (
        CategoryTip(
            key="apparel_wash",
            title="服饰洗涤",
            text=(
                "服饰洗涤通用建议：按洗水标说明洗涤；深色衣物首次洗涤可能轻微掉色，"
                "建议单独用冷水洗；带刺绣、亮片等装饰的衣物建议手洗或反面洗涤。"
            ),
            keywords=_WASH_KEYWORDS,
            topic=TOPIC_CARE,
        ),
        CategoryTip(
            key="apparel_dry",
            title="服饰晾晒",
            text=(
                "服饰晾晒通用建议：反面晾晒、避免长时间暴晒可以减少褪色；"
                "针织类衣物建议平铺晾干，防止变形。"
            ),
            keywords=("晒", "晾", "烘干", "褪色", "变形", "保养", "打理"),
            topic=TOPIC_CARE,
        ),
        CategoryTip(
            key="apparel_iron",
            title="服饰熨烫",
            text="服饰熨烫通用建议：按洗水标的温度熨烫；真丝、化纤类面料用低温，可以垫一层布再熨。",
            keywords=("熨", "皱", "褶"),
            topic=TOPIC_CARE,
        ),
    ),
    APPLIANCE: (
        CategoryTip(
            key="appliance_first_use",
            title="家电首次使用",
            text=(
                "家电使用通用建议：首次使用前请先阅读说明书，按说明安装和通电；"
                "使用中出现异味、异响时请先断电，再联系客服处理。"
            ),
            keywords=("怎么用", "使用", "用法", "安装", "第一次", "首次", "说明书", "异响", "异味"),
            topic=TOPIC_USAGE,
        ),
        CategoryTip(
            key="appliance_clean",
            title="家电清洁",
            text=(
                "家电清洁通用建议：清洁前先断电，外壳用干布或拧干的湿布擦拭，"
                "不要直接用水冲洗电器主体。"
            ),
            keywords=("清洁", "清洗", "洗", "擦", "保养"),
            topic=TOPIC_CARE,
        ),
    ),
}


def category_tips(group: str | None, question: str) -> list[CategoryTip]:
    """该分组里和问题相关的通用知识（按表内顺序）。"""
    if group is None:
        return []
    return [t for t in CATEGORY_TIPS.get(group, ()) if any(k in question for k in t.keywords)]


def category_tip_evidence(tip: CategoryTip) -> Evidence:
    """类目通用知识证据（source=CATEGORY，data={"text"}）。"""
    return Evidence(
        id=ev.category(tip.key),
        source=EvidenceSource.CATEGORY,
        label=f"类目通用知识 · {tip.title}",
        content=tip.text,
        data={"text": tip.text},
        topic=tip.topic,
    )
