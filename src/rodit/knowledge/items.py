"""商品知识证据（方案 §5.2.2 商品知识卡片、§5.2.4 "结构化优先：问面料就取 attributes.面料"）。

每个构造函数返回 (候选证据, 缺失标记)。证据 ID / data 结构严格按 core.evidence 模块文档：
- item:{id}:title        {"title"}
- item:{id}:attr:{name}  {"name", "value"}
- item:{id}:sku          {"skus": [Sku...]}
- item:{id}:size_chart   {"rows": [...], "conservative": bool}
- item:{id}:size_advice  {"text"}
- item:{id}:care         {"care": [...]}
- item:{id}:usage        {"usage": [...]}
- item:{id}:detail:{seg} {"text", "seg"}
价格和库存不写死在卡片里（§5.2.2），SKU 证据只列规格组合，并提示以实时查询为准。
"""

from __future__ import annotations

from dataclasses import dataclass

from rodit.core import evidence as ev
from rodit.core.enums import EvidenceSource
from rodit.core.models import Evidence, ItemCard
from rodit.knowledge.category import TOPIC_CARE, TOPIC_USAGE
from rodit.knowledge.conflict import Candidate, ordered_topics
from rodit.knowledge.text import for_matching, minimized, relevance

# ---------------------------------------------------------------- 属性：问题关键词 → 属性


@dataclass(frozen=True)
class AttributeTopic:
    """一类商品属性。

    - name：规范名，缺失标记写 item:{id}:attr:{name}；
    - aliases：卡片上的属性名包含其中任一个，就算这一类（平台同步来的属性名不统一，
      如"材质成分"也算面料）；
    - keywords：买家问题里出现其中任一个，就算问到这一类。
    """

    name: str
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]


ATTRIBUTE_TOPICS: tuple[AttributeTopic, ...] = (
    AttributeTopic(
        "面料",
        ("面料", "材质", "成分", "材料", "布料", "里料"),
        (
            "面料",
            "材质",
            "成分",
            "材料",
            "布料",
            "料子",
            "纯棉",
            "全棉",
            "棉",
            "真丝",
            "桑蚕丝",
            "羊毛",
            "羊绒",
            "涤纶",
            "聚酯",
            "化纤",
            "氨纶",
            "亚麻",
            "棉麻",
            "莫代尔",
            "起球",
        ),
    ),
    AttributeTopic(
        "厚薄",
        ("厚薄", "厚度"),
        ("厚", "薄", "透", "保暖", "季节", "冬天", "夏天", "春秋"),
    ),
    AttributeTopic("袖长", ("袖长", "袖型"), ("袖",)),
    AttributeTopic("弹力", ("弹力", "弹性"), ("弹",)),
    AttributeTopic(
        "版型",
        ("版型", "廓形"),
        ("版型", "显瘦", "宽松", "修身", "紧身", "显胖", "收腰", "廓形", "oversize"),
    ),
    AttributeTopic("裤长", ("裤长",), ("裤长", "九分", "七分", "长裤", "短裤")),
    AttributeTopic("腰型", ("腰型", "腰高"), ("腰型", "高腰", "中腰", "低腰", "腰高")),
    AttributeTopic(
        "领型",
        ("领型", "领口"),
        ("领型", "领口", "圆领", "v领", "翻领", "高领", "立领"),
    ),
    AttributeTopic("长度", ("衣长", "裙长", "长度"), ("衣长", "裙长", "多长", "长度")),
    AttributeTopic("定制", ("定制", "刻字", "刺绣"), ("定制", "刻字", "绣", "印字")),
    AttributeTopic("产地", ("产地",), ("产地", "哪里产", "哪产的", "国产", "进口")),
)


def attribute_topic_of(attr_name: str) -> AttributeTopic | None:
    """卡片属性名属于哪一类（按 aliases 包含关系）。"""
    name = for_matching(attr_name)
    return next((t for t in ATTRIBUTE_TOPICS if any(a in name for a in t.aliases)), None)


def asked_attribute_topics(question: str) -> list[AttributeTopic]:
    q = for_matching(question)
    return [t for t in ATTRIBUTE_TOPICS if any(k in q for k in t.keywords)]


def _named_in_question(attr_name: str, question: str) -> bool:
    """属性名（或它的前两个字，如"电池容量"的"电池"）直接出现在问题里。"""
    name = for_matching(attr_name)
    q = for_matching(question)
    return bool(name) and (name in q or (len(name) >= 2 and name[:2] in q))


# ---------------------------------------------------------------- 证据构造


def _item_evidence(
    evidence_id: str,
    field_label: str,
    content: str,
    data: dict,
    *,
    topic: str | None = None,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        source=EvidenceSource.ITEM,
        label=ev.item_label(field_label),
        content=minimized(content),
        data=data,
        topic=topic,
    )


def title_candidate(card: ItemCard) -> Candidate:
    return Candidate(
        _item_evidence(
            ev.item_title(card.item_id), "标题", f"商品标题：{card.title}", {"title": card.title}
        )
    )


def attribute_candidates(card: ItemCard, question: str) -> tuple[list[Candidate], list[str]]:
    """问题相关的属性；问题没问到具体属性（或问到的卡片上都没有）时给全部属性。

    问到但卡片上没有的属性类记缺失（item:{id}:attr:{规范名}）；卡片完全没有属性记
    item:{id}:attributes。
    """
    asked = asked_attribute_topics(question)
    by_topic: dict[str, list[str]] = {}
    for name in card.attributes:
        topic = attribute_topic_of(name)
        if topic is not None:
            by_topic.setdefault(topic.name, []).append(name)

    missing = [
        ev.missing_item_field(card.item_id, f"attr:{t.name}")
        for t in asked
        if t.name not in by_topic
    ]
    if not card.attributes:
        if not asked:
            missing.append(ev.missing_item_field(card.item_id, "attributes"))
        return [], missing

    asked_names = {t.name for t in asked}
    relevant = [
        name
        for name in card.attributes
        if ((topic := attribute_topic_of(name)) is not None and topic.name in asked_names)
        or _named_in_question(name, question)
    ]
    if not relevant:
        relevant = list(card.attributes)
    cands = [
        Candidate(
            _item_evidence(
                ev.item_attr(card.item_id, name),
                name,
                f"{name}：{card.attributes[name]}",
                {"name": name, "value": card.attributes[name]},
            )
        )
        for name in relevant
    ]
    return cands, missing


def _sku_desc(props: dict[str, str]) -> str:
    """规格文字，如 "黑色 / M"。不写 sku_id（内部编号，不该出现在给买家的回答里）。"""
    return " / ".join(v for v in props.values() if v) or "默认规格"


def sku_candidate(card: ItemCard, sku_id: str | None) -> Candidate | None:
    """SKU 规格组合；买家当前选中的 SKU（会话里的商品卡片）在文字里点明。没有 SKU 返回 None。"""
    if not card.skus:
        return None
    combos = "；".join(_sku_desc(s.props()) for s in card.skus)
    content = f"可选规格：{combos}（有没有货、价格多少以实时查询为准）"
    selected = card.find_sku(sku_id) if sku_id else None
    if selected is not None:
        content += f"。买家当前选择的规格：{_sku_desc(selected.props())}"
    return Candidate(
        _item_evidence(
            ev.item_sku(card.item_id),
            "规格",
            content,
            {"skus": [s.model_dump(mode="json") for s in card.skus]},
        )
    )


CONSERVATIVE_NOTE = "（商家没有填写尺码建议：只能按尺码表保守推荐，请买家对照自己的实际尺寸确认）"


def size_candidates(card: ItemCard) -> tuple[list[Candidate], list[str]]:
    """尺码表 + 尺码建议（§5.2.2 "size_advice 是尺码推荐最重要的依据；
    只有尺码表时，推荐要更保守"）。

    没有尺码建议时尺码表 data.conservative=True，文字里也注明要保守；缺的记
    item:{id}:size_advice / item:{id}:size_chart。
    """
    cands: list[Candidate] = []
    missing: list[str] = []
    advice = (card.size_advice or "").strip()
    if card.size_chart:
        rows = "；".join("、".join(f"{k} {v}" for k, v in row.items()) for row in card.size_chart)
        content = f"尺码表：{rows}"
        if not advice:
            content += CONSERVATIVE_NOTE
        cands.append(
            Candidate(
                _item_evidence(
                    ev.item_size_chart(card.item_id),
                    "尺码表",
                    content,
                    {"rows": [dict(r) for r in card.size_chart], "conservative": not advice},
                )
            )
        )
    else:
        missing.append(ev.missing_item_field(card.item_id, "size_chart"))
    if advice:
        cands.append(
            Candidate(
                _item_evidence(
                    ev.item_size_advice(card.item_id),
                    "尺码建议",
                    f"尺码建议：{advice}",
                    {"text": advice},
                )
            )
        )
    else:
        missing.append(ev.missing_item_field(card.item_id, "size_advice"))
    return cands, missing


_CARE_KEYWORDS = (
    "洗",
    "晒",
    "晾",
    "熨",
    "烘",
    "保养",
    "打理",
    "清洁",
    "掉色",
    "褪色",
    "缩水",
)
_USAGE_KEYWORDS = (
    "怎么用",
    "使用",
    "用法",
    "安装",
    "组装",
    "操作",
    "说明书",
    "充电",
    "开机",
)


def care_usage_candidates(card: ItemCard, question: str) -> tuple[list[Candidate], list[str]]:
    """洗涤保养、使用说明（topic 分别是 care / usage，覆盖同主题的类目通用知识）。

    问到洗涤而卡片没有 care、问到使用而卡片没有 usage 时记缺失；问题两类都没明确问到、
    卡片两样都没有时两样都记缺失。
    """
    q = for_matching(question)
    asks_care = any(k in q for k in _CARE_KEYWORDS)
    asks_usage = any(k in q for k in _USAGE_KEYWORDS)
    cands: list[Candidate] = []
    missing: list[str] = []
    if card.care:
        cands.append(
            Candidate(
                _item_evidence(
                    ev.item_care(card.item_id),
                    "洗涤保养",
                    "洗涤保养：" + "；".join(card.care),
                    {"care": list(card.care)},
                    topic=TOPIC_CARE,
                ),
                topics=frozenset({TOPIC_CARE}),
            )
        )
    if card.usage:
        cands.append(
            Candidate(
                _item_evidence(
                    ev.item_usage(card.item_id),
                    "使用说明",
                    "使用说明：" + "；".join(card.usage),
                    {"usage": list(card.usage)},
                    topic=TOPIC_USAGE,
                ),
                topics=frozenset({TOPIC_USAGE}),
            )
        )
    nothing = not card.care and not card.usage
    if not card.care and (asks_care or (nothing and not asks_usage)):
        missing.append(ev.missing_item_field(card.item_id, "care"))
    if not card.usage and (asks_usage or (nothing and not asks_care)):
        missing.append(ev.missing_item_field(card.item_id, "usage"))
    return cands, missing


def detail_candidates(card: ItemCard, question: str) -> list[Candidate]:
    """详情页文字按句切段，每句一条证据；主题用 detect_topics 打（冲突时只删那一句）。"""
    out: list[Candidate] = []
    for seg, sentence in enumerate(ev.split_sentences(card.detail_text or "")):
        topics = ordered_topics(ev.detect_topics(sentence))
        out.append(
            Candidate(
                _item_evidence(
                    ev.item_detail(card.item_id, seg),
                    "详情页",
                    f"详情页：{sentence}",
                    {"text": sentence, "seg": seg},
                    topic=topics[0] if topics else None,
                ),
                topics=frozenset(topics),
                rank=relevance(question, sentence),
            )
        )
    return out
