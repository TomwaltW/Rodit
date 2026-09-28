"""质检第一道：硬规则（方案 §5.5 第一道、§2.1 话术禁区、§7.2 回答规则、§11.5、§12.3）。

| 规则 | 做法 | 是否拦截 |
|---|---|---|
| format | 去掉 Markdown（**、__、#、`、[x](url)→x、列表符号） | 否，改写文本 |
| length | 超过 ctx.config.max_answer_chars 在句子边界截断 | 否，改写文本 |
| off_platform | core.lexicon.find_off_platform_in_answer（原文和清洗后都查） | 是 |
| absolute_term | core.lexicon.find_absolute_terms（词表和例外；依据里原样有的成分标注放行） | 是 |
| banned_word | ctx.config.banned_words（商家禁用词，附录 C 第 14 题） | 是 |
| forbidden_claim | 当前商品和证据里商品卡片的 forbidden_claims（被否定时放行） | 是 |
| number_mismatch | 回答里的数字必须出现在证据里；买家说过的只放行身体尺寸（qa.numbers） | 是 |
| time_promise | 明天 / 马上到 / 3 天左右到 / 30 号送达……证据里没有就拦截（§7.2、§11.5） | 是 |
| unknown_evidence | draft.evidence_ids 引用了证据包里没有的来源 | 是 |

内容检查都在"去掉 Markdown、截断之前"的全文上做：模型写了违规内容，哪怕被截断掉也要重新生成。
时效承诺只看肯定的说法："没法确定明天能不能送到""不一定明天能到"是如实说明，不拦截；
不确定只管同一小句，"不一定今天到但明天就能到"里的"明天就能到"照样拦截。
"周一到周日都在线""1 号到 7 号放假"是范围，不是到货承诺。
命中说明（detail）是具体的中文原因，展示给客服、带回重新生成；
站外联系、数字片段先遮个人信息（qa.masking.mask_text），原因里不出现手机号原文。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable

from rodit.core.config import QAGateConfig
from rodit.core.enums import Role
from rodit.core.lexicon import find_absolute_terms, find_off_platform_in_answer
from rodit.core.models import (
    NON_BLOCKING_HARD_RULES,
    ConversationContext,
    Draft,
    EvidenceBundle,
    HardRuleHit,
)
from rodit.qa.masking import mask_text
from rodit.qa.numbers import NumberSet, unsupported_numbers
from rodit.qa.text import strip_markdown, truncate

_WHITESPACE = re.compile(r"\s+")


def _norm(text: str) -> str:
    """NFKC、去空白，用于子串比对。"""
    return _WHITESPACE.sub("", unicodedata.normalize("NFKC", text or ""))


# ---------------------------------------------------------------- 证据语料


class EvidenceCorpus:
    """一次质检用到的依据文本：证据的 content、data、获取时间，以及买家自己的消息（只用于数字）。"""

    def __init__(self, ctx: ConversationContext, evidence: EvidenceBundle) -> None:
        texts: list[str] = []
        for e in evidence.evidences:
            texts.append(e.content)
            if e.data:
                texts.append(json.dumps(e.data, ensure_ascii=False, default=str))
            if e.fetched_at is not None:
                texts.append(e.fetched_at.isoformat())
        self.evidence_text = _norm("\n".join(texts))
        self.numbers = NumberSet()
        for t in texts:
            self.numbers.add_text(t)
        # 买家自己说的身高体重、尺码，回答里复述不算编造；买家报的价格、时长不算依据（§5.5）
        self.numbers.add_buyer_text(ctx.current.text)
        for turn in ctx.history:
            if turn.role == Role.BUYER:
                self.numbers.add_buyer_text(turn.text)

    def contains(self, phrase: str) -> bool:
        return _norm(phrase) in self.evidence_text

    def search(self, pattern: re.Pattern[str]) -> bool:
        return pattern.search(self.evidence_text) is not None


# ---------------------------------------------------------------- 时效承诺（§7.2、§11.5）

_CN_NUM = "[零一二两三四五六七八九十]{1,3}"
_ARRIVE_WORDS = "到货|送到|送达|签收|到"
_ARRIVE = rf"(?:{_ARRIVE_WORDS})"
_MODAL = r"(?:就)?(?:能|会|可以)?"

# 时长和到货之间可以有"内 / 左右 / 后"（"3 天内到""3 天左右可以到""3 天后就能到货"）
_DURATION_QUALIFIER = r"(?:内|之内|以内|左右|前后|以后|后)?"

# 从具体到一般；片段重叠时只报最具体的那条
# 日期（"10 月 8 日到货"）不是时长；写了月份的日期由数字一致性检查
_PROMISE_N_DAYS = re.compile(
    rf"(?<!月)(?<!月\s)(?<!\d)(?P<core>(?:\d+|{_CN_NUM})\s*(?:个)?\s*(?:小时|天|日|工作日))"
    rf"\s*{_DURATION_QUALIFIER}\s*{_MODAL}{_ARRIVE}"
)
_PROMISE_DAY_DELIVERY = re.compile(r"(?:\d+|[一二两三次当隔])日达")
# 某月几号、星期几前到 / 发货（"30 号送达""周五前能到"）。
# "周一到周日都在线""1 号到 7 号放假"里的"到"是范围，后面紧跟星期几 / 几号时不算
_PROMISE_DATE = re.compile(
    rf"(?:(?<![\d.])(?<!月)(?<!月\s)(?P<day>\d{{1,2}})\s*号|(?:周|星期|礼拜)[一二三四五六日天末])"
    rf"\s*(?:之前|以前|前)?\s*{_MODAL}(?:{_ARRIVE_WORDS}|发货|发出)"
    r"(?!\s*(?:周|星期|礼拜|\d{1,2}\s*号))"
)
_PROMISE_IMMEDIATE = re.compile(rf"(?:马上|立刻|立即|很快){_MODAL}(?:{_ARRIVE_WORDS}|发货|发出)")
_PROMISE_DAY_WORD = re.compile(
    r"今天|今日|明天|明日|(?<!售)后天|今晚|今夜|明早|明晚|当天|当日|次日|隔天|隔日"
)
# 证据里的预计送达 / 发货时间（"平台预计送达时间 09-30 18:00""预计 10 月 8 日发货"）
_ETA_IN_EVIDENCE = re.compile(r"预计.{0,8}(?:送达|到达|到货|发货)")

# 否定、不确定的说法不是承诺："没法确定明天能不能送到""不一定明天能到""明天能不能到要看派送"
_UNCERTAIN = (
    r"没法确定|无法确定|不能确定|不确定|没办法确定|不一定|不能保证|没法保证|无法保证|"
    r"没办法保证|不敢保证|不好说|说不准|说不好"
)
# 往回看只看同一小句："不一定今天到但明天就能到""说不准 明天就能到""没法保证今天到~明天就能送到"
# 里的"明天"在转折词、空格、"~"后面，是另一句肯定的承诺，照样拦截
_UNCERTAIN_BEFORE = re.compile(
    rf"(?:{_UNCERTAIN})(?:(?!但|不过|可是|然而)[^，。！？,.!?；;\n\s~…]){{0,8}}$"
)
# 紧跟在承诺词后面（用 match(text, pos)，不能写 ^）
_UNCERTAIN_AFTER = re.compile(rf"\s*(?:能不能|会不会|可不可以|是否|能否|{_UNCERTAIN})")


def _uncertain(norm: str, m: re.Match[str]) -> bool:
    before = norm[max(0, m.start() - 16) : m.start()]
    return bool(_UNCERTAIN_BEFORE.search(before) or _UNCERTAIN_AFTER.match(norm, m.end()))


def _date_promise_supported(m: re.Match[str], corpus: EvidenceCorpus) -> bool:
    """ "30 号送达"：证据里有预计送达 / 发货时间，且有这一天的日期；"周五前到"：证据里原样有。"""
    day = m.group("day")
    if day is None:
        return corpus.contains(m.group(0))
    days = {d for _, d in corpus.numbers.dates}
    return int(day) in days and corpus.search(_ETA_IN_EVIDENCE)


def _time_promise_hits(text: str, corpus: EvidenceCorpus) -> list[HardRuleHit]:
    norm = unicodedata.normalize("NFKC", text)
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str]] = []

    def claim(m: re.Match[str]) -> bool:
        if any(m.start() < e and s < m.end() for s, e in taken):
            return False
        taken.append(m.span())
        return True

    for m in _PROMISE_N_DAYS.finditer(norm):
        if not claim(m) or _uncertain(norm, m):
            continue
        # "48 小时内发货"不能放行"48 小时内送达"：证据里要有同样的时长 + 送达
        core = re.escape(_norm(m.group("core")))
        allowed = rf"{core}{_DURATION_QUALIFIER}{_MODAL}{_ARRIVE}"
        if not corpus.search(re.compile(allowed)):
            found.append((m.start(), m.group(0)))
    for m in _PROMISE_DATE.finditer(norm):
        if claim(m) and not _uncertain(norm, m) and not _date_promise_supported(m, corpus):
            found.append((m.start(), m.group(0)))
    for pattern in (_PROMISE_DAY_DELIVERY, _PROMISE_IMMEDIATE, _PROMISE_DAY_WORD):
        for m in pattern.finditer(norm):
            if claim(m) and not _uncertain(norm, m) and not corpus.contains(m.group(0)):
                found.append((m.start(), m.group(0)))
    found.sort()
    return [
        HardRuleHit(rule="time_promise", detail=f"包含依据中没有的时效承诺：{span}", span=span)
        for span in _unique(s for _, s in found)
    ]


# ---------------------------------------------------------------- 商家禁止的卖点

# 紧挨在卖点前面的否定："不防晒""没有抗菌功能""不具备防晒""无法防晒"；"不仅防晒"不算否定
_CLAIM_NEGATION = re.compile(
    r"(?:没有|没|无|不|非|并非|不是)(?:法|能|会|具备|具有|带|含|有|是|算)?(?:任何)?$"
)


def _forbidden_claims(ctx: ConversationContext, evidence: EvidenceBundle) -> list[str]:
    claims: list[str] = list(ctx.item.forbidden_claims) if ctx.item is not None else []
    for e in evidence.evidences:
        extra = e.data.get("forbidden_claims")
        if isinstance(extra, list):
            claims.extend(str(c) for c in extra)
    return _unique(c.strip() for c in claims if c and c.strip())


def _forbidden_claim_hits(
    text: str, ctx: ConversationContext, evidence: EvidenceBundle
) -> list[HardRuleHit]:
    norm = unicodedata.normalize("NFKC", text)
    hits: list[HardRuleHit] = []
    for claim in _forbidden_claims(ctx, evidence):
        target = unicodedata.normalize("NFKC", claim)
        for m in re.finditer(re.escape(target), norm):
            if _CLAIM_NEGATION.search(norm[max(0, m.start() - 6) : m.start()]):
                continue
            hits.append(
                HardRuleHit(
                    rule="forbidden_claim", detail=f"包含商家禁止的卖点：{claim}", span=claim
                )
            )
            break
    return hits


# ---------------------------------------------------------------- 其他规则


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _off_platform_hits(raw: str, cleaned: str) -> list[HardRuleHit]:
    spans = _unique(
        mask_text(s)
        for s in [*find_off_platform_in_answer(raw), *find_off_platform_in_answer(cleaned)]
    )
    return [
        HardRuleHit(rule="off_platform", detail=f"包含站外联系或站外交易内容：{s}", span=s)
        for s in spans
    ]


def _absolute_hits(text: str, config: QAGateConfig, corpus: EvidenceCorpus) -> list[HardRuleHit]:
    terms = find_absolute_terms(
        text,
        terms=config.absolute_terms,
        exceptions=config.absolute_term_exceptions,
        evidence_text=corpus.evidence_text,
    )
    return [HardRuleHit(rule="absolute_term", detail=f"包含绝对化用语：{t}", span=t) for t in terms]


def _banned_word_hits(text: str, banned: list[str]) -> list[HardRuleHit]:
    norm = unicodedata.normalize("NFKC", text).lower()
    hits: list[HardRuleHit] = []
    for word in _unique(w.strip() for w in banned if w and w.strip()):
        if unicodedata.normalize("NFKC", word).lower() in norm:
            hits.append(
                HardRuleHit(rule="banned_word", detail=f"包含商家禁用词：{word}", span=word)
            )
    return hits


def _number_hits(text: str, corpus: EvidenceCorpus) -> list[HardRuleHit]:
    spans = _unique(mask_text(t.text) for t in unsupported_numbers(text, corpus.numbers))
    return [
        HardRuleHit(rule="number_mismatch", detail=f"包含依据中找不到的数字：{span}", span=span)
        for span in spans
    ]


def _unknown_evidence_hits(draft: Draft, evidence: EvidenceBundle) -> list[HardRuleHit]:
    known = set(evidence.ids())
    overridden = set(evidence.overridden_ids)
    hits: list[HardRuleHit] = []
    for eid in _unique(draft.evidence_ids):
        if eid in known:
            continue
        if eid in overridden:
            detail = f"引用了已被更高优先级依据覆盖的来源：{eid}"
        else:
            detail = f"引用了依据中不存在的来源：{eid}"
        hits.append(HardRuleHit(rule="unknown_evidence", detail=detail, span=eid))
    return hits


# ---------------------------------------------------------------- 入口


def check_hard_rules(
    text: str,
    *,
    ctx: ConversationContext,
    draft: Draft,
    evidence: EvidenceBundle,
    config: QAGateConfig,
) -> tuple[str, list[HardRuleHit]]:
    """跑全部硬规则。返回 (清洗后的文本, 命中列表)。

    清洗后的文本去掉了 Markdown、超长时已截断，是最终展示 / 发送的文本；
    命中里 format / length 是自动修正（NON_BLOCKING_HARD_RULES），其余都拦截。
    """
    hits: list[HardRuleHit] = []
    cleaned, changed = strip_markdown(text)
    if changed:
        hits.append(HardRuleHit(rule="format", detail="去掉了平台不支持的 Markdown 格式"))
    final = cleaned
    limit = ctx.config.max_answer_chars
    if len(cleaned) > limit:
        final = truncate(cleaned, limit)
        hits.append(
            HardRuleHit(
                rule="length",
                detail=f"回答 {len(cleaned)} 字，超过 {limit} 字上限，已截断到 {len(final)} 字",
            )
        )
    corpus = EvidenceCorpus(ctx, evidence)
    hits.extend(_off_platform_hits(text, cleaned))
    hits.extend(_absolute_hits(cleaned, config, corpus))
    hits.extend(_banned_word_hits(cleaned, ctx.config.banned_words))
    hits.extend(_forbidden_claim_hits(cleaned, ctx, evidence))
    hits.extend(_number_hits(cleaned, corpus))
    hits.extend(_time_promise_hits(cleaned, corpus))
    hits.extend(_unknown_evidence_hits(draft, evidence))
    return final, hits


def blocking_hits(hits: Iterable[HardRuleHit]) -> list[HardRuleHit]:
    """拦截类命中（format / length 之外的全部规则）。"""
    return [h for h in hits if h.rule not in NON_BLOCKING_HARD_RULES]
