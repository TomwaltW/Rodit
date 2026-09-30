"""实际发出去的文字的合规扫描（方案 §13.4 "平台违规为 0" 的代理指标、§5.5 硬规则、§2.1 话术禁区）：
客服编辑后发送的推荐 + 会话里客服 / 机器人发出的消息，按店铺、按 ISO 周给计数、推荐 / 消息 ID 和
规则名，不含任何文字。数据全部合成。"""

from __future__ import annotations

from datetime import datetime, timedelta

from rodit.core.clock import CST
from rodit.core.config import EngineConfig
from rodit.core.enums import DraftKind, Intent, Role, SuggestionStatus
from rodit.core.models import ShopConfig, StoredMessage, SuggestionRecord
from rodit.feedback.compliance import (
    RULE_ABSOLUTE,
    RULE_BANNED,
    RULE_OFF_PLATFORM,
    ComplianceRules,
    scan_sent,
    scan_text,
    summarize_scan,
)
from rodit.testing.fixtures import SHOP_A, SHOP_B

W39 = datetime(2026, 9, 21, tzinfo=CST)  # 周一
W40 = W39 + timedelta(days=7)
RULES = ComplianceRules.from_config(EngineConfig().qa_gate, ShopConfig())
_ids = iter(range(100, 10_000))

EDITED = "加我微信 abc123 发您"
AGENT_TEXT = "这款是全网最低价"
CLEAN = "亲，已经发货了哦，请耐心等待~"
SECRET_WORDS = ("微信", "abc123", "加我", "最低价", "全网", "发您", "耐心等待")


def edited(text: str, *, at: datetime = W39, shop_id: str = SHOP_A, conv: str = "c1"):
    return SuggestionRecord(
        id=next(_ids),
        shop_id=shop_id,
        conversation_id=conv,
        msg_id=f"b{next(_ids)}",
        intent=Intent.SHIPPING_POLICY,
        text="亲，现货付款后 48 小时内发货哦~",
        kind=DraftKind.LLM,
        status=SuggestionStatus.EDITED_SENT,
        edited_text=text,
        agent_id="kf_a01",
        created_at=at + timedelta(hours=10),
    )


def msg(
    text: str,
    *,
    at: datetime = W39,
    role: Role = Role.AGENT,
    msg_id: str | None = None,
    shop_id: str = SHOP_A,
    conv: str = "c2",
):
    return StoredMessage(
        msg_id=msg_id or f"out_{next(_ids)}",
        shop_id=shop_id,
        conversation_id=conv,
        role=role,
        text=text,
        created_at=at + timedelta(hours=11),
    )


def scan(records, messages, *, rules: ComplianceRules = RULES, since=W39, until=W40):
    return scan_sent(records, messages, shop_id=SHOP_A, rules=rules, since=since, until=until)


def test_two_potential_violations_in_the_week_and_no_text_in_the_output():
    record = edited(EDITED)
    agent = msg(AGENT_TEXT)
    [week] = scan([record], [agent, msg(CLEAN)])
    assert week.period == "2026-W39" and week.shop_id == SHOP_A
    assert week.since == W39 and week.until == W40
    assert week.flagged == 2
    assert week.scanned_suggestions == 1 and week.scanned_messages == 2
    assert week.rule_counts == {RULE_OFF_PLATFORM: 1, RULE_ABSOLUTE: 1}
    by_source = {(f.source, f.suggestion_id, f.msg_id): f.rules for f in week.flags}
    assert by_source == {
        ("edited_suggestion", record.id, None): [RULE_OFF_PLATFORM],
        ("message", None, agent.msg_id): [RULE_ABSOLUTE],
    }
    dumped = week.model_dump_json()
    for word in SECRET_WORDS:
        assert word not in dumped, word
    summary = summarize_scan([week])
    assert summary.flagged == 2 and summary.suggestion_ids == [record.id]
    assert summary.message_ids == [agent.msg_id]
    assert summary.by_period == {"2026-W39": 2}
    for word in SECRET_WORDS:
        assert word not in summary.model_dump_json(), word


def test_clean_text_gives_zero():
    [week] = scan([edited(CLEAN)], [msg(CLEAN), msg("好的亲，最近发货都很快", role=Role.BOT)])
    assert week.flagged == 0 and week.flags == [] and week.rule_counts == {}
    assert week.scanned_suggestions == 1 and week.scanned_messages == 2


def test_the_same_send_is_counted_once():
    """推荐的 edited_text 和写回会话的那条（agent:{推荐 id}）、平台回声（同一会话、同样的文字）
    是同一次发送：只算一条（记在推荐上）。"""
    record = edited(EDITED, conv="c9")
    write_back = msg(EDITED, msg_id=f"agent:{record.id}", conv="c9")
    [week] = scan([record], [write_back])
    assert week.flagged == 1 and week.duplicates == 1
    assert [f.source for f in week.flags] == ["edited_suggestion"]
    echo = msg(EDITED.replace(" ", "  "), msg_id="platform_77", conv="c9")
    [week] = scan([record], [echo])
    assert week.flagged == 1 and week.duplicates == 1
    # 别的会话里同样的文字是另一次发送
    [week] = scan([record], [msg(EDITED, conv="c10")])
    assert week.flagged == 2 and week.duplicates == 0
    # 同一会话里客服又发了一遍：只有一条配得上
    [week] = scan([record], [write_back, echo])
    assert week.flagged == 2 and week.duplicates == 1


def test_rules_cover_shop_banned_words_and_masked_contacts():
    rules = ComplianceRules.from_config(EngineConfig().qa_gate, ShopConfig(banned_words=["秒杀"]))
    assert scan_text("今天秒杀价哦", rules) == [RULE_BANNED]
    assert scan_text("今天秒杀价哦", RULES) == []
    # 存下来的文字做过个人信息最小化：遮住的手机号、联系方式也是站外联系
    assert scan_text("您加这个号 [手机号] 就行", RULES) == [RULE_OFF_PLATFORM]
    assert scan_text("已按 [地址] 发出", RULES) == []
    assert scan_text("最新款、第一时间发货", RULES) == []  # 例外词不算绝对化用语
    assert scan_text("平台规定不能通过站外联系和交易哦", RULES) == []  # 标准话术（§11.4）
    assert scan_text("保证全网最低，加 vx", RULES) == [RULE_OFF_PLATFORM, RULE_ABSOLUTE]


def test_only_outgoing_text_of_this_shop_in_the_window_counts():
    records = [
        edited(EDITED, shop_id=SHOP_B),  # 别的店铺
        edited(EDITED, at=W40),  # 窗口外
        edited(EDITED).model_copy(update={"status": SuggestionStatus.SENT}),  # 原样发送：看消息
        edited(EDITED).model_copy(update={"edited_text": None}),
    ]
    messages = [
        msg(AGENT_TEXT, role=Role.BUYER),  # 买家的话不算
        msg(AGENT_TEXT, shop_id=SHOP_B),
        msg(AGENT_TEXT, at=W39 - timedelta(days=7)),
    ]
    [week] = scan(records, messages)
    assert week.flagged == 0 and week.scanned_suggestions == 0 and week.scanned_messages == 0


def test_the_window_is_split_into_iso_weeks():
    since = W39 + timedelta(days=3)  # 从周四开始
    until = W40 + timedelta(days=2)
    weeks = scan(
        [edited(EDITED, at=W39 + timedelta(days=4))],
        [msg(AGENT_TEXT, at=W40 + timedelta(days=1)), msg(AGENT_TEXT, at=W39)],
        since=since,
        until=until,
    )
    assert [(w.period, w.since, w.until, w.flagged) for w in weeks] == [
        ("2026-W39", since, W40, 1),
        ("2026-W40", W40, until, 1),
    ]
    summary = summarize_scan(weeks)
    assert summary.since == since and summary.until == until
    assert summary.flagged == 2 and summary.by_period == {"2026-W39": 1, "2026-W40": 1}
    assert summary.rule_counts == {RULE_OFF_PLATFORM: 1, RULE_ABSOLUTE: 1}


def test_listed_ids_are_capped_but_counts_are_not():
    messages = [msg(AGENT_TEXT) for _ in range(7)]
    summary = summarize_scan(scan([], messages), max_ids=3)
    assert summary.flagged == 7 and len(summary.message_ids) == 3 and summary.ids_truncated
