"""平台规则（§7.3.1、§1.4、§2.1、§11.4）：覆盖所有平台 × 规则键，文字本身不违反回答硬规则。"""

from __future__ import annotations

import re

import pytest

from rodit.core import evidence as ev
from rodit.core.enums import EvidenceSource, Platform
from rodit.core.evidence import PlatformRuleKey
from rodit.core.lexicon import find_absolute_terms, find_off_platform_in_answer
from rodit.core.pii import contains_pii
from rodit.knowledge import (
    PLATFORM_RULES,
    find_platform_rule,
    pending_platform_rules,
    platform_rule_evidence,
    platform_rule_text,
)

ALL = [(p, k) for p in Platform for k in PlatformRuleKey]


def test_every_platform_has_every_rule():
    assert set(PLATFORM_RULES) == set(ALL)


@pytest.mark.parametrize(("platform", "key"), ALL)
def test_rule_evidence_shape(platform, key):
    e = platform_rule_evidence(platform, key)
    assert e.id == ev.platform_rule(platform, key) == f"platform:{platform}:{key}"
    assert e.source == EvidenceSource.PLATFORM and e.priority == 2
    assert e.label == ev.platform_rule_label(platform, key)
    assert set(e.data) == {"platform", "key", "text"}
    assert e.data == {"platform": platform.value, "key": key.value, "text": e.content}
    assert e.topic is None  # 与同名店铺政策互补，不互相覆盖
    assert e.score is None and e.content.strip()


@pytest.mark.parametrize(("platform", "key"), ALL)
def test_rule_text_is_safe_to_quote_to_buyers(platform, key):
    text = platform_rule_text(platform, key)
    assert find_off_platform_in_answer(text) == []
    assert find_absolute_terms(text) == []
    assert not contains_pii(text)
    assert "【待确认】" not in text  # 待确认事项只在 pending 里，不会被引用给买家


@pytest.mark.parametrize("platform", list(Platform))
def test_apply_aftersale_only_buyer_can_start_it(platform):
    text = platform_rule_text(platform, PlatformRuleKey.APPLY_AFTERSALE)
    assert "买家" in text and "订单页" in text and "申请售后" in text
    assert "客服不能替买家发起售后" in text  # §1.4 非目标


@pytest.mark.parametrize("platform", list(Platform))
def test_refund_timing_makes_no_hard_promise(platform):
    text = platform_rule_text(platform, PlatformRuleKey.REFUND_TIMING)
    assert text.startswith("退款到账时间以平台规则为准")
    assert re.search(r"\d", text) is None  # 不写具体天数


@pytest.mark.parametrize("platform", list(Platform))
def test_modify_order_before_and_after_shipping(platform):
    text = platform_rule_text(platform, PlatformRuleKey.MODIFY_ORDER)
    assert "发货前" in text and "发货后" in text and "不能再修改" in text


@pytest.mark.parametrize("platform", list(Platform))
def test_off_platform_rule(platform):
    assert platform_rule_text(platform, PlatformRuleKey.OFF_PLATFORM).startswith(
        "平台规定不能通过站外联系和交易"
    )


def test_platform_specific_names():
    assert "淘宝" in platform_rule_text(Platform.TAOBAO, PlatformRuleKey.APPLY_AFTERSALE)
    assert "抖音" in platform_rule_text(Platform.DOUYIN, PlatformRuleKey.APPLY_AFTERSALE)
    assert "拼多多" in platform_rule_text(Platform.PDD, PlatformRuleKey.APPLY_AFTERSALE)
    assert platform_rule_evidence("pdd", "apply_aftersale").label == "平台规则 · 拼多多申请售后"


def test_pending_items_are_marked():
    pending = pending_platform_rules()
    assert pending and all(r.pending.startswith("【待确认】") for r in pending)
    # 各平台的售后入口、到账时长都待确认（附录 A）
    keys = {(r.platform, r.key) for r in pending}
    for p in Platform:
        assert (p, PlatformRuleKey.APPLY_AFTERSALE) in keys
        assert (p, PlatformRuleKey.REFUND_TIMING) in keys


def test_unknown_rule():
    assert find_platform_rule("jd", "apply_aftersale") is None
    assert find_platform_rule("taobao", "nope") is None
    with pytest.raises(KeyError):
        platform_rule_evidence("taobao", "nope")
    with pytest.raises(KeyError):
        platform_rule_text("jd", "apply_aftersale")
