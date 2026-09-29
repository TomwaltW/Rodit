"""店铺政策问卷（方案 §5.2.3 冷启动第 1 步、附录 C）。

商家开通时回答 15 个问题：1–13 写进店铺政策（PolicyKey），14 禁用词、15 语气和称呼写进
店铺配置覆盖项（Shop.config_overrides，§9.3）。

一题可能对应两个政策键（如第 1 题"付款后多久发货？预售商品怎么说明？"对应 ship_time 和
presale），商家只写一段回答。拆分规则：
- 回答按句号、分号、逗号等切成分句；
- 每个分句按关键词归到一个键：先看"次要"键（第二个及以后，说法更具体，如"预售""运费险"），
  再看主键；
- 都没命中的分句跟随上一个分句（"可以指定顺丰，需补运费差价 10 元"的后半句跟随前半句），
  第一个分句没命中就归主键；
- 归到同一个键的分句按原顺序拼回去。没有分句归到的键不写（留给缺失检查提醒商家补）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rodit.core.enums import PolicyKey
from rodit.core.models import SHOP_CONFIG_FIELDS, PolicyEntry, ShopConfig


@dataclass(frozen=True)
class PolicyTarget:
    """问卷回答要写进的政策键；keywords 用于把回答里的分句归到这个键。"""

    key: PolicyKey
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionnaireItem:
    """一道问卷题。policy_targets 第一个是主键；config_fields 是写进店铺配置的字段。"""

    id: int
    text: str
    policy_targets: tuple[PolicyTarget, ...] = ()
    config_fields: tuple[str, ...] = ()

    @property
    def policy_keys(self) -> tuple[PolicyKey, ...]:
        return tuple(t.key for t in self.policy_targets)


_P = PolicyKey


def _q(qid: int, text: str, *targets: PolicyTarget) -> QuestionnaireItem:
    return QuestionnaireItem(id=qid, text=text, policy_targets=targets)


QUESTIONNAIRE: tuple[QuestionnaireItem, ...] = (
    _q(
        1,
        "付款后多久发货？预售商品怎么说明？",
        PolicyTarget(
            _P.SHIP_TIME, ("发货", "发出", "小时", "工作日", "当天", "次日", "现货", "天内")
        ),
        PolicyTarget(_P.PRESALE, ("预售",)),
    ),
    _q(
        2,
        "默认发哪家快递？能否指定快递，要不要加钱？",
        PolicyTarget(
            _P.COURIER,
            ("默认", "快递", "中通", "圆通", "申通", "韵达", "顺丰", "邮政", "极兔", "京东"),
        ),
        PolicyTarget(
            _P.COURIER_CHOICE,
            (
                "指定",
                "可选",
                "可以选",
                "换快递",
                "加钱",
                "补差价",
                "差价",
                "另付",
                "加收",
                "补运费",
            ),
        ),
    ),
    _q(3, "哪些地区不发货或需要补运费？", PolicyTarget(_P.REMOTE_AREAS)),
    _q(
        4,
        "是否支持七天无理由退货？哪些商品不支持？",
        PolicyTarget(
            _P.SEVEN_DAY_RETURN, ("七天无理由", "7天无理由", "7 天无理由", "退换", "退货")
        ),
        PolicyTarget(
            _P.NON_RETURNABLE,
            (
                "除外",
                "定制",
                "贴身",
                "内衣",
                "内裤",
                "袜",
                "清仓",
                "特价",
                "拆封",
                "生鲜",
                "商品不支持",
                "款不支持",
                "类不支持",
            ),
        ),
    ),
    _q(
        5,
        "退货运费由谁承担？是否送运费险？",
        PolicyTarget(_P.RETURN_SHIPPING, ("运费", "承担", "买家出", "卖家出", "店铺出", "包邮")),
        PolicyTarget(_P.SHIPPING_INSURANCE, ("运费险",)),
    ),
    _q(6, "质量问题怎么处理？需要买家提供哪些凭证？", PolicyTarget(_P.QUALITY_ISSUE)),
    _q(7, "能开发票吗？什么类型、多久开出？", PolicyTarget(_P.INVOICE)),
    _q(8, "有没有赠品？什么条件下送？", PolicyTarget(_P.GIFT)),
    _q(9, "下单后能否改地址、改规格？在什么时间之前可以改？", PolicyTarget(_P.MODIFY_ORDER)),
    _q(
        10,
        "目前有哪些优惠？客服可以承诺什么、不能承诺什么？",
        PolicyTarget(_P.PROMOTIONS, ("满", "减", "券", "折", "优惠", "活动", "包邮")),
        PolicyTarget(
            _P.PROMISE_BOUNDARY,
            ("承诺", "答应", "不能保证", "不能给", "不能说", "不可以给", "不要给"),
        ),
    ),
    _q(11, "保修期多久？怎么报修？", PolicyTarget(_P.WARRANTY)),
    _q(12, "尺码偏大还是偏小？（服饰类）", PolicyTarget(_P.SIZE_TENDENCY)),
    _q(13, "客服的工作时间？", PolicyTarget(_P.SERVICE_HOURS)),
    QuestionnaireItem(
        id=14,
        text="哪些话绝对不能说？（商家禁用词、不能宣传的功效）",
        config_fields=("banned_words",),
    ),
    QuestionnaireItem(
        id=15,
        text='回复语气偏好（亲切 / 专业 / 简洁），称呼用"亲"还是"您"？',
        config_fields=("tone", "salutation"),
    ),
)
"""附录 C 的 15 道题（题目文字与方案一致）。"""

QUESTIONNAIRE_BY_ID: dict[int, QuestionnaireItem] = {q.id: q for q in QUESTIONNAIRE}


def question_for_policy(key: PolicyKey | str) -> QuestionnaireItem | None:
    """某个政策键由第几题填写（缺失提醒里告诉商家去补哪一题）。"""
    return next((q for q in QUESTIONNAIRE if key in {k.value for k in q.policy_keys}), None)


# ================================================================ 1–13：政策

_CLAUSE_SPLIT = re.compile(r"(?<=[，,；;。！!？?\n])")
_CLAUSE_TRAIL = "，,；;、 \t\n"


def split_policy_answer(item: QuestionnaireItem, answer: str) -> dict[PolicyKey, str]:
    """把一题的回答拆到它对应的政策键（规则见模块文档）。"""
    text = answer.strip()
    if not text or not item.policy_targets:
        return {}
    primary, *secondary = item.policy_targets
    if not secondary:
        return {primary.key: text}
    check_order = (*secondary, primary)
    assigned: dict[PolicyKey, list[str]] = {}
    current = primary.key
    for clause in (c for c in _CLAUSE_SPLIT.split(text) if c.strip()):
        hit = next((t.key for t in check_order if any(k in clause for k in t.keywords)), None)
        current = hit or current
        assigned.setdefault(current, []).append(clause.strip())
    out: dict[PolicyKey, str] = {}
    for target in item.policy_targets:
        clauses = assigned.get(target.key)
        if clauses:
            out[target.key] = "".join(clauses).rstrip(_CLAUSE_TRAIL)
    return out


# ================================================================ 14：禁用词

_NONE_ANSWERS = {"无", "没有", "暂无", "不限", "没", "无禁用词", "没有禁用词", "都可以"}
_BANNED_SPLIT = re.compile(r"[，,、；;。/|\n\s]+")
_QUOTES = re.compile(r"[\"'“”‘’「」『』《》【】()（）]")
_BANNED_PREFIX = re.compile(
    r"^(?:禁用词|禁用|不能宣传|不要宣传|不能说|不要说|禁止说|别说|不能提|不要提|不能用|不要用)[:：]?"
)
MAX_BANNED_WORD_CHARS = 30


def parse_banned_words(answer: str) -> list[str]:
    """第 14 题：禁用词列表。按顿号、逗号、引号等切开，去掉"不能说"之类的引导语，去重保序。"""
    text = answer.strip()
    if text in _NONE_ANSWERS:
        return []
    words: list[str] = []
    for raw in _BANNED_SPLIT.split(_QUOTES.sub("、", text)):
        word = raw.strip()
        while True:
            stripped = _BANNED_PREFIX.sub("", word).strip()
            if stripped == word:
                break
            word = stripped
        if word and word not in _NONE_ANSWERS and len(word) <= MAX_BANNED_WORD_CHARS:
            if word not in words:
                words.append(word)
    return words


# ================================================================ 15：语气与称呼

_TONE_WORDS: tuple[tuple[str, str], ...] = (
    ("亲切", "亲切"),
    ("热情", "亲切"),
    ("活泼", "亲切"),
    ("专业", "专业"),
    ("正式", "专业"),
    ("严谨", "专业"),
    ("简洁", "简洁"),
    ("简短", "简洁"),
    ("简单", "简洁"),
    ("干脆", "简洁"),
)
_NEGATION = re.compile(r"(?:不|别|勿|禁|莫)[^，,。；;！!？?]{0,3}$")


def _first_affirmed(text: str, words: tuple[tuple[str, str], ...]) -> str | None:
    """最早出现、且前面没有被"不 / 别 / 不要用"否定的词对应的取值。"""
    best: tuple[int, str] | None = None
    for word, value in words:
        for m in re.finditer(re.escape(word), text):
            if _NEGATION.search(text[max(0, m.start() - 4) : m.start()]):
                continue
            if best is None or m.start() < best[0]:
                best = (m.start(), value)
            break
    return best[1] if best else None


def parse_tone(answer: str) -> dict[str, str]:
    """第 15 题：返回 {"tone": ..., "salutation": ...} 中能识别出来的项。

    "亲切"里的"亲"不算称呼；被否定的词（"不要用亲"）跳过。
    """
    text = answer.strip()
    out: dict[str, str] = {}
    tone = _first_affirmed(text, _TONE_WORDS)
    if tone:
        out["tone"] = tone
    # 先把语气词遮掉（等长占位，保持位置），再找称呼
    masked = text
    for word, _ in _TONE_WORDS:
        masked = masked.replace(word, "\u0000" * len(word))
    salutation = _first_affirmed(masked, (("您", "您"), ("亲", "亲")))
    if salutation:
        out["salutation"] = salutation
    return out


# ================================================================ 入口


def apply_questionnaire(
    shop_id: str, answers: dict[int, str], now: datetime
) -> tuple[list[PolicyEntry], dict[str, Any]]:
    """把问卷回答变成店铺政策条目和店铺配置覆盖项补丁。

    - answers：题号 → 回答文字；空白回答跳过；题号不存在抛 ValueError；
    - now：写进 PolicyEntry.updated_at，必须带时区；
    - 返回 (政策条目，按题号、键顺序；config_overrides 补丁)。补丁里的 banned_words 是完整列表
      （替换，不追加），合并进 Shop.config_overrides 由调用方完成。
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now 必须带时区")
    unknown = sorted(set(answers) - set(QUESTIONNAIRE_BY_ID))
    if unknown:
        raise ValueError(f"问卷没有这些题号：{unknown}")
    entries: list[PolicyEntry] = []
    patch: dict[str, Any] = {}
    for item in QUESTIONNAIRE:
        answer = answers.get(item.id)
        if answer is None:
            continue
        if not isinstance(answer, str):
            raise TypeError(f"第 {item.id} 题的回答必须是文字")
        if not answer.strip():
            continue
        for key, value in split_policy_answer(item, answer).items():
            entries.append(PolicyEntry(shop_id=shop_id, key=key.value, value=value, updated_at=now))
        if "banned_words" in item.config_fields:
            patch["banned_words"] = parse_banned_words(answer)
        if "tone" in item.config_fields:
            patch.update(parse_tone(answer))
    assert set(patch) <= SHOP_CONFIG_FIELDS
    ShopConfig.model_validate(patch)  # 补丁取值必须合法
    return entries, patch
