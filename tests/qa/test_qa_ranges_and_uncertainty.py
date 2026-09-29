"""复审第二轮（§5.5 第一道数字一致性、§7.2、§11.1、§11.5）：

- 不确定说法的回看窗口遇到"但 / 不过"、空格、"~"就停：后面肯定的"明天就能到"照样拦截；
- 问尺码的消息里"宽松一点""日常穿""夏天穿""161到163""给我发"不算价格 / 时间用语，
  买家没写单位的身高照样算身体尺寸；身体尺寸的范围（"身高161-163"）两头都算；
- "周一到周日""1 号到 7 号"里的"到"是范围，不是到货承诺；
- 证据里的范围和比例（"10-15 分钟""10-20㎡""1:20 兑水""10-12 岁"）照样放行里面的数字。
"""

from __future__ import annotations

import pytest

from rodit.core.config import QAGateConfig
from rodit.core.enums import DraftKind, EvidenceSource
from rodit.core.models import Draft, Evidence, EvidenceBundle
from rodit.qa import NumberSet, blocking_hits, check_hard_rules, unsupported_numbers
from rodit.testing.fakes import make_context

CFG = QAGateConfig()


def blocking(text, ctx, bundle) -> list[tuple[str, str | None]]:
    draft = Draft(text=text, kind=DraftKind.LLM, evidence_ids=bundle.ids())
    _, hits = check_hard_rules(text, ctx=ctx, draft=draft, evidence=bundle, config=CFG)
    return [(h.rule, h.span) for h in blocking_hits(hits)]


def text_evidence(content: str) -> EvidenceBundle:
    return EvidenceBundle(
        evidences=[
            Evidence(
                id="item:i_555:attr:说明",
                source=EvidenceSource.ITEM,
                label="商品属性 · 说明",
                content=content,
                data={},
            )
        ]
    )


# ---------------------------------------------------------------- 不确定说法不跨到下一句


@pytest.mark.parametrize(
    "text",
    [
        "亲，不一定今天到但明天就能到",
        "亲，今天不一定能发出去但明天就能到",
        "亲，不确定今天能不能发 明天就能到",
        "亲，具体几点说不准 明天就能到哦",
        "亲，没法保证今天到~明天就能送到",
        "亲，不一定今天到不过明天就能到",
    ],
)
def test_uncertainty_does_not_cover_the_next_affirmative_promise(world, ev, text):
    """o_1002 运输中、没有预计送达时间：跟在"不一定…但"后面的"明天就能到"是承诺，要拦截。"""
    ctx = make_context("明天能到吗？", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    assert ("time_promise", "明天") in blocking(text, ctx, bundle), text


@pytest.mark.parametrize(
    "text",
    ["亲，没法确定明天能不能送到", "亲，不一定明天能到", "亲，说不准明天能到哦"],
)
def test_uncertain_statement_itself_still_passes(world, ev, text):
    ctx = make_context("明天能到吗？", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    assert blocking(text, ctx, bundle) == [], text


# ---------------------------------------------------------------- 问尺码的常见说法


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("163 108斤穿什么码", "亲，您 163、108 斤建议 M 码哦"),
        ("163 108斤想穿宽松一点选什么码", "亲，您 163、108 斤建议 M 码哦"),
        ("163 108斤日常穿什么码", "亲，您 163、108 斤建议 M 码哦"),
        ("163 108斤夏天穿什么码", "亲，您 163、108 斤建议 M 码哦"),
        ("身高161到163之间 体重108斤穿什么码", "亲，身高 161 到 163、体重 108 斤建议 M 码哦"),
        ("163 108斤想穿宽松一点选什么码", "亲，您身高 163 cm、体重 108 斤建议 M 码哦"),
        ("158 95斤 给我发什么码", "亲，您 158、95 斤建议 S 码哦"),
    ],
)
def test_ordinary_sizing_words_keep_buyer_height(world, ev, question, answer):
    """§11.1：回答复述买家报的身高体重不算编造；"一点""日常""夏天""到""发"不是价格 / 时间。"""
    ctx = make_context(question, world, item_id="i_555")
    bundle = EvidenceBundle(evidences=[ev.size_advice()])
    assert blocking(answer, ctx, bundle) == [], question


def test_body_range_counts_both_ends_even_in_price_clause():
    """ "身高161-163"：前面是"身高"，范围的另一头也是身高（分句里有"包邮"也一样）。"""
    ns = NumberSet()
    ns.add_text("尺码建议：M 码适合 100–115 斤")
    ns.add_buyer_text("身高161-163 包邮吗，穿什么码")
    assert unsupported_numbers("亲，身高 161-163 建议 M 码", ns) == []
    # 同一分句里的价格照样不算依据
    assert [t.text for t in unsupported_numbers("亲，可以 150 包邮", ns)] == ["150"]


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("163 108斤穿什么码 200 块行吗", "亲，200 可以"),
        ("163 108斤穿什么码，打 8 折行吗", "亲，可以打 8 折"),
        ("163 108斤穿什么码，3 天能到吗", "亲，3 天能到哦"),
        ("163 108斤穿什么码，150 包邮卖吗", "亲，150 包邮可以"),
        ("163 108斤穿什么码，满 150 减 20 吗", "亲，满 150 减 20 哦"),
    ],
)
def test_price_and_time_clauses_still_not_evidence(question, answer):
    ns = NumberSet()
    ns.add_text("尺码建议：M 码适合 100–115 斤")
    ns.add_buyer_text(question)
    assert unsupported_numbers(answer, ns) != [], question


# ---------------------------------------------------------------- "周一到周日"是范围


@pytest.mark.parametrize(
    ("evidence", "answer"),
    [
        ("客服在线时间周一至周日 9:00–23:00", "亲，客服周一到周日都在线哦"),
        ("现货付款后 48 小时内发货，周一至周五正常发货", "亲，我们周一到周五都正常发货哦"),
        (
            "国庆 10 月 1 日至 10 月 7 日放假，10 月 8 日起恢复发货",
            "亲，1 号到 7 号放假，8 号起恢复发货哦",
        ),
        ("现货付款后 48 小时内发货，周末不发货", "亲，周六周日不发货，周一到周五正常发货哦"),
    ],
)
def test_weekday_and_day_ranges_are_not_promises(world, evidence, answer):
    ctx = make_context("你们什么时候发货？", world, item_id="i_555")
    assert blocking(answer, ctx, text_evidence(evidence)) == [], answer


@pytest.mark.parametrize(
    "answer",
    ["亲，周五前就能到哦", "亲，30 号送达哦", "亲，周一到周三能到哦", "亲，1 号到 7 号能送到"],
)
def test_real_weekday_and_day_promises_still_blocked(world, answer):
    ctx = make_context("什么时候到？", world, item_id="i_555")
    rules = [r for r, _ in blocking(answer, ctx, text_evidence("现货付款后 48 小时内发货"))]
    assert "time_promise" in rules, answer


# ---------------------------------------------------------------- 证据里的范围、比例


@pytest.mark.parametrize(
    ("evidence", "answer"),
    [
        ("洗涤说明：先用冷水浸泡 10-15 分钟再手洗", "亲，先用冷水浸泡 15 分钟左右再手洗哦"),
        ("适用面积 10-20㎡", "亲，20 平米以内的房间都可以用"),
        ("清洁剂用法：按 1:20 兑水稀释后擦拭", "亲，1 份清洁剂兑 20 份水稀释后擦拭就行"),
        ("童装 适合 10-12 岁宝宝穿", "亲，这款适合 10 到 12 岁的孩子哦"),
        ("充电说明：充电 10-12 分钟可用一天", "亲，充电 12 分钟左右就可以用一天哦"),
    ],
)
def test_ranges_and_ratios_in_evidence_support_their_numbers(world, evidence, answer):
    ctx = make_context("这个怎么用？", world, item_id="i_555")
    assert blocking(answer, ctx, text_evidence(evidence)) == [], answer


def test_standalone_month_day_keeps_values_but_fmt_time_does_not():
    """没跟时分的"10-12"可能是范围：日期记下，数字也照样算；fmt_time 写法（日期 + 时分）
    和钟点时间整体比较，分量不当普通数字。"""
    ns = NumberSet()
    ns.add_text("童装尺码对照：S 对应 10-12，M 对应 12-14")
    assert unsupported_numbers("S 码对应 10 到 12", ns) == []
    assert (10, 12) in ns.dates
    ns = NumberSet()
    ns.add_text("截至 09-28 10:30；客服 9:00–23:30")
    assert [t.text for t in unsupported_numbers("满 28 减 10，30 元", ns)] == ["28", "10", "30 元"]
    assert [t.text for t in unsupported_numbers("满 23 减 9", ns)] == ["23", "9"]
    assert unsupported_numbers("截至 09-28 10:30", ns) == []


# ---------------------------------------------------------------- 复审第三轮（7f9a2c0 之后的回归）


PRICE_AND_SIZE = (
    "法式碎花连衣裙：售价 259.00 元，活动：店铺满 199 减 20；尺码建议：M 码适合 100–115 斤"
)


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("M码今天拍是239吗", "亲，M 码是 239 哦"),
        ("M码到手239吗", "亲，M 码到手 239 哦"),
        ("穿M码 周末239能拍吗", "亲，239 可以拍哦"),
        ("M码能少点吗 239行不行", "亲，239 可以的"),
    ],
)
def test_buyer_price_in_a_sizing_message_is_not_evidence(world, question, answer):
    """问尺码的消息里买家报的价格（"今天 / 到手 / 周末 / 少点 239"）不能给回答里的价格当依据。"""
    ctx = make_context(question, world, item_id="i_555")
    assert ("number_mismatch", "239") in blocking(answer, ctx, text_evidence(PRICE_AND_SIZE))


@pytest.mark.parametrize(
    "text",
    [
        "亲，现货是付款后 48 小时内发货，没法保证 48 小时内送到哦，要看快递派送",
        "亲，不能保证 48 小时内送达哦，发货后以物流为准",
        "亲，不一定 48 小时内能到哦",
    ],
)
def test_hedge_before_a_spaced_number_is_not_a_promise(world, ev, text):
    """ "没法保证 48 小时内送到"里汉字和数字之间的空格不打断不确定说法（§7.2 如实说明）。"""
    ctx = make_context("48小时能送到吗？", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002(), ev.policy("ship_time")])
    assert [r for r in blocking(text, ctx, bundle) if r[0] == "time_promise"] == []


@pytest.mark.parametrize("text", ["亲，我们周一到周五发货，周末不发哦", "亲，周一至周五发货哦"])
def test_weekday_range_paraphrasing_evidence_is_not_a_promise(world, text):
    ctx = make_context("周末发货吗", world)
    bundle = text_evidence("现货付款后 48 小时内发货，周一至周五正常发货")
    assert [r for r in blocking(text, ctx, bundle) if r[0] == "time_promise"] == []


def test_weekday_promise_without_evidence_still_blocked(world, ev):
    ctx = make_context("什么时候能到", world, order_id="o_1002")
    bundle = EvidenceBundle(evidences=[ev.o_1002()])
    assert any(r[0] == "time_promise" for r in blocking("亲，周五到周六能到哦", ctx, bundle))


@pytest.mark.parametrize(
    ("evidence", "answer"),
    [
        ("预售款 10-08 周三开始发货", "亲，预售款 10 月 8 日开始发货哦"),
        ("国庆 10-01 至 10-07 期间暂停发货", "亲，10 月 7 日之后恢复发货哦"),
        ("预售款 10-08 分批发货", "亲，预售款 10 月 8 日起分批发货哦"),
    ],
)
def test_real_dates_followed_by_weekday_or_period_are_dates(world, evidence, answer):
    ctx = make_context("什么时候发货", world)
    assert blocking(answer, ctx, text_evidence(evidence)) == []


def test_clock_time_before_exchange_is_not_a_ratio(world):
    """ "20:00 兑换"是钟点，不是比例：20 不能给"立减 20 元"当依据。"""
    ctx = make_context("有优惠吗", world)
    hits = blocking("亲，现在下单立减 20 元哦", ctx, text_evidence("积分每天 20:00 兑换优惠券"))
    assert ("number_mismatch", "20 元") in hits
