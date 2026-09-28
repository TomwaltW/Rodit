"""取件码脱敏不能把后面的取件截止时间一起遮掉（§7.2 取件指引、§5.5 数字一致性）。

复审发现：空格分组的取件码规则把"取件码 60125477 18:00 前取出"里的"18"也当成码的一组，
证据里只剩"[取件码]:00 前取出"，如实说"请在 18:00 前取出"的回答被判数字不一致。
"""

from __future__ import annotations

import pytest

from rodit.aftersale.advisor import DefaultAfterSaleAdvisor
from rodit.core import evidence as ev
from rodit.core.config import QAGateConfig
from rodit.core.enums import DraftKind, Intent
from rodit.core.models import Draft
from rodit.qa import blocking_hits, check_hard_rules
from rodit.testing.fakes import FakeClock, InMemoryToolGateway, make_context
from rodit.testing.fixtures import build_demo_world
from rodit.tools.evidence import DefaultToolEvidenceProvider
from rodit.tools.logistics import DefaultLogisticsAdvisor
from rodit.tools.normalize import PICKUP_CODE_MASK, scrub_pickup_code


@pytest.mark.parametrize(
    ("text", "code", "kept"),
    [
        ("快件已存入丰巢柜，取件码 60125477 18:00 前取出，超时收费", "60125477", "18:00 前取出"),
        ("您的快件已到幸福里驿站，取件码 5-2-3021 21点前有效", "5-2-3021", "21点前有效"),
        ("取件码 60125477 18：00 前取出", "60125477", "18：00 前取出"),
        ("取件码 A12 3456 20:30 前取件", "3456", "20:30 前取件"),
        ("取件码 60125477 72小时内取件", "60125477", "72小时内取件"),
        ("取件码 60125477 18时前取出", "60125477", "18时前取出"),
    ],
)
def test_pickup_deadline_survives_code_mask(text, code, kept):
    out = scrub_pickup_code(text)
    assert code not in out and PICKUP_CODE_MASK in out, out
    assert kept in out, out


def test_pickup_deadline_reaches_evidence_and_supports_answer():
    """o_1003 最后一条轨迹写着取件截止时间：证据里保留 18:00，如实转述的回答不被拦截。"""
    w = build_demo_world()
    desc = "快件已存入幸福里丰巢柜，取件码 60125477 18:00 前取出，超时收费"
    for i, t in enumerate(w.traces):
        if t.order_id == "o_1003":
            last = t.events[-1].model_copy(update={"description": desc})
            w.traces[i] = t.model_copy(update={"events": [*t.events[:-1], last]})
    clock = FakeClock(w.now)
    provider = DefaultToolEvidenceProvider(
        InMemoryToolGateway(w),
        w.policy_map,
        DefaultLogisticsAdvisor(),
        DefaultAfterSaleAdvisor(),
        clock,
    )
    ctx = make_context("快递到哪了", w, order_id="o_1003")
    bundle = provider.collect(ctx, Intent.LOGISTICS_TRACK, ev.needs_for(Intent.LOGISTICS_TRACK))
    log = bundle.get(ev.tool_logistics("o_1003"))
    assert log is not None
    trace_text = log.data["trace"]["events"][-1]["description"]
    assert "60125477" not in trace_text and "18:00 前取出" in trace_text, trace_text

    text = "亲，包裹已存入幸福里丰巢柜，请在 18:00 前凭取件码取出哦"
    draft = Draft(text=text, kind=DraftKind.LLM, evidence_ids=bundle.ids())
    _, hits = check_hard_rules(text, ctx=ctx, draft=draft, evidence=bundle, config=QAGateConfig())
    assert [(h.rule, h.span) for h in blocking_hits(hits)] == []
