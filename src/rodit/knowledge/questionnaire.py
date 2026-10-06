"""店铺政策问卷（方案 §5.2.3 冷启动第 1 步、附录 C）。

商家开通时回答 15 个问题：1–13 写进店铺政策（PolicyKey），14 禁用词、15 语气和称呼写进
店铺配置覆盖项（Shop.config_overrides，§9.3）；13 客服工作时间看得懂时（"9:00-23:00"、
"早9点到晚11点"……，见 parse_service_hours_text）同时写进 ShopConfig.service_hours（北京时间，
全自动先在非工作时间试行用，§15 Phase 3）；看不懂时政策原文照写，ShopConfig.service_hours 清空
（替换掉原来的时段：不知道客服什么时候在岗就不全自动），并提醒商家到设置页填写
（questionnaire_warnings）。

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
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rodit.core.enums import PolicyKey
from rodit.core.models import SHOP_CONFIG_FIELDS, PolicyEntry, ShopConfig
from rodit.core.schedule import normalize_service_hours, parse_window


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
    QuestionnaireItem(
        id=13,
        text="客服的工作时间？",
        policy_targets=(PolicyTarget(_P.SERVICE_HOURS),),
        config_fields=("service_hours",),  # 看得懂时（parse_service_hours_text）
    ),
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


# ================================================================ 13：客服工作时间

_NUM = r"(?:\d{1,2}|[零〇一二两三四五六七八九十]{1,3})"
_PERIOD = r"(?:凌晨|清晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里|夜间|次日|第二天|早|晚)"
_CLOCK = rf"({_PERIOD})?\s*({_NUM})\s*(?:(:)\s*(\d{{2}})|(点|时)\s*(半|整|{_NUM}\s*分?)?)?"
_HOURS_RANGE = re.compile(_CLOCK + r"\s*(?:-|–|—|~|至|到)\s*" + _CLOCK)
_HOURS_SHORT = re.compile(
    rf"早上?\s*({_NUM})\s*点?\s*(半)?\s*[,、]?\s*晚上?\s*({_NUM})\s*点?\s*(半)?"
)
_HOURS_DAYS = re.compile(r"周[一二三四五六日天末]|星期|礼拜|工作日|节假日|节日|假期|法定|双休|单休")
_HOURS_ALL_DAY = re.compile(r"全天|24\s*(?:小时|h)|二十四小时|不打烊")
_HOURS_OFF_DUTY = re.compile(
    r"休息|午休|午饭|吃饭|不在|没有客服|没客服|无客服|没人|无人|离线|下线|下班|机器人|自动|托管|值班"
    r"|留言|除外|以外|之外|不含|不包括|除了|暂停"
)
"""说客服不在岗（或"除了……"）的字眼：写了这些，文字里的时段可能是下班、午休、机器人值班的时间，
按时段读出来就把它当成了客服在岗的时间（全自动反而在客服在岗时回复），所以整段不认。"""
_HOURS_HINT = "请只写客服在岗的时段，如 09:00-23:00"
_PM = frozenset({"下午", "傍晚", "晚上", "晚", "夜里", "夜间"})
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4}
_CN_DIGITS |= {"五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
MAX_SERVICE_HOURS_TEXT = 200


def _cn_number(text: str) -> int:
    """ "9" / "九" / "十一" / "二十三" → 整数。"""
    text = text.strip()
    if text.isdigit():
        return int(text)
    if "十" not in text:
        if len(text) != 1:
            raise ValueError(f"看不懂的数字：{text}")
        return _CN_DIGITS[text]
    tens, _, ones = text.partition("十")
    return (_CN_DIGITS[tens] if tens else 1) * 10 + (_CN_DIGITS[ones] if ones else 0)


def _minute(colon: str | None, dian_minute: str | None) -> int:
    if colon is not None:
        return int(colon)
    if not dian_minute or dian_minute == "整":
        return 0
    if dian_minute == "半":
        return 30
    return _cn_number(dian_minute.rstrip("分").strip())


def _hour_24(period: str | None, hour: int, *, end: bool) -> int:
    """按"早 / 下午 / 晚上"换成 24 小时制。晚上 12 点、晚上 0 点都是半夜（不是中午）：
    作结束是 24，作开始是 0。"""
    if period in _PM:
        if hour in (0, 12):
            return 24 if end else 0
        return hour + 12 if hour < 12 else hour
    if period == "中午":
        return hour + 12 if hour <= 2 else hour
    if period == "凌晨" and hour == 12:
        return 0
    return hour


def _window(start: tuple[int, int], finish: tuple[int, int]) -> str:
    (sh, sm), (eh, em) = start, finish
    if (eh, em) == (0, 0) and (sh, sm) != (0, 0):
        eh = 24  # 到 0 点 = 到半夜
    text = f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}"
    return parse_window(text).text()  # 时间不对、开始等于结束时抛 ValueError


def parse_service_hours_text(answer: str) -> list[str]:
    """第 13 题 / 设置页填的客服工作时间 → 统一写法的时段（"HH:MM-HH:MM"，北京时间，可以跨午夜）。

    认得："9:00-23:00""09：00～23：00""9点到22点""早9点到晚11点""早上九点到晚上十一点半"
    "下午1点到5点""8:00-12:00，13:30-22:00"（多个时段）"早9晚11""全天 / 24 小时"。
    晚上 / 夜里的 0 点、12 点是半夜（"早9点到晚0点" → 09:00-24:00）。
    看不懂时抛 ValueError（原因给商家看），调用方决定怎么办（问卷清空时段并提醒，设置页提示改写）。
    宁可不认也不猜——猜错了全自动会在客服在岗时回复：
    - 按星期、节假日区分的（"周一至周五……"）：一个店铺只有一套每天的时段，不猜；
    - 写了不在岗的说明（"晚上11点到早上9点不在线""12:00-13:30午休""机器人值班""……除外"，
      见 _HOURS_OFF_DUTY）：里面的时段可能是下班时间，不猜；
    - 既写全天 / 24 小时又写了具体时段（"全天在线，12:00-13:30"）：说不准哪个是客服在岗的时间；
    - 结束时间是 1–12 点、又没写上午 / 晚上、也不是 HH:MM 写法（"9点到11点"）：说不准是上午 11 点
      还是晚上 11 点，不猜。开始写了下午 / 晚上时，结束跟着算（"下午1点到5点" → 13:00-17:00）。
    """
    text = unicodedata.normalize("NFKC", answer or "").strip()
    if not text:
        raise ValueError("没有填写")
    if len(text) > MAX_SERVICE_HOURS_TEXT:
        raise ValueError(f"最多 {MAX_SERVICE_HOURS_TEXT} 个字")
    if _HOURS_DAYS.search(text):
        raise ValueError("按星期或节假日区分的时间暂不支持，请写每天的时段，如 09:00-23:00")
    off_duty = _HOURS_OFF_DUTY.search(text)
    if off_duty is not None:
        raise ValueError(
            f"写了「{off_duty.group(0)}」这类不在岗的说明，分不清哪段是客服在岗的时间，{_HOURS_HINT}"
        )
    all_day = _HOURS_ALL_DAY.search(text) is not None
    windows: list[str] = []
    for m in _HOURS_RANGE.finditer(text):
        p1, h1, _, cm1, _, dm1, p2, h2, c2, cm2, _, dm2 = m.groups()
        start_hour, end_hour = _cn_number(h1), _cn_number(h2)
        if p2 is None and c2 is None and 1 <= end_hour <= 12:
            if p1 not in _PM:
                said = m.group(0).strip()
                raise ValueError(f"「{said}」分不清上午还是晚上，请用 24 小时制，如 09:00-23:00")
            p2 = p1  # "下午1点到5点"：结束跟着开始算
        start = (_hour_24(p1, start_hour, end=False), _minute(cm1, dm1))
        finish = (_hour_24(p2, end_hour, end=True), _minute(cm2, dm2))
        windows.append(_window(start, finish))
    if not windows:
        short = _HOURS_SHORT.search(text)
        if short is not None:
            h1, half1, h2, half2 = short.groups()
            start = (_cn_number(h1), 30 if half1 else 0)
            finish = (_hour_24("晚", _cn_number(h2), end=True), 30 if half2 else 0)
            windows.append(_window(start, finish))
    if all_day:
        if windows:  # 原来只认具体时段、把"全天"丢掉：具体时段可能是午休等不在岗的时间
            raise ValueError(
                "既写了全天又写了具体时段，分不清客服什么时候在岗；全天在岗只写「全天」，"
                f"否则{_HOURS_HINT}"
            )
        windows.append("00:00-24:00")
    if not windows:
        raise ValueError("没有认出时段，请写成 09:00-23:00 这样（北京时间）")
    return normalize_service_hours(windows)


def _service_hours_answer(answer: str) -> tuple[list[str], str | None]:
    """第 13 题的回答 → (时段, 看不懂的原因)。看不懂时时段是空列表（清空：不知道就不全自动）。"""
    try:
        return parse_service_hours_text(answer), None
    except ValueError as exc:
        return [], str(exc)


SERVICE_HOURS_WARNING = (
    "第 {qid} 题的客服工作时间没有看懂：{why}。店铺的客服工作时间已清空（不知道客服什么时候在岗，"
    "就不开全自动），请到控制台「设置」里按 09:00-23:00 这样填写客服工作时间（北京时间）。"
)
"""问卷第 13 题看不懂时给商家的提醒（问卷接口响应的 warnings）。"""


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
      （替换，不追加），合并进 Shop.config_overrides 由调用方完成；答了第 13 题时补丁里有
      service_hours（完整列表，替换）：看不懂时是空列表——原来的时段不能留着（全自动会按旧时段跑），
      不知道客服什么时候在岗就不全自动（engine.modes 的 no_service_hours）；给商家的提醒见
      questionnaire_warnings。
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
        if "service_hours" in item.config_fields:
            patch["service_hours"], _ = _service_hours_answer(answer)  # 看不懂时清空
    assert set(patch) <= SHOP_CONFIG_FIELDS
    ShopConfig.model_validate(patch)  # 补丁取值必须合法
    return entries, patch


def questionnaire_warnings(answers: Mapping[int, str]) -> list[str]:
    """问卷回答里要提醒商家的事（问卷接口响应里给商家看）：目前只有客服工作时间（第 13 题）
    没看懂——apply_questionnaire 已把时段清空，请商家到设置页按 09:00-23:00 这样填写。"""
    warnings: list[str] = []
    for item in QUESTIONNAIRE:
        answer = answers.get(item.id)
        if "service_hours" not in item.config_fields or not isinstance(answer, str):
            continue
        if not answer.strip():
            continue
        _, why = _service_hours_answer(answer)
        if why is not None:
            warnings.append(SERVICE_HOURS_WARNING.format(qid=item.id, why=why.rstrip("。")))
    return warnings
