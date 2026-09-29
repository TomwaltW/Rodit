"""质检：decision_log 写失败不能把已经判完的结果丢掉（集成审查修复，§12.2）。"""

from __future__ import annotations

import logging

from rodit.core.audit import InMemoryAuditSink
from rodit.core.config import QAGateConfig
from rodit.core.enums import DraftKind, Intent
from rodit.core.errors import StorageUnavailable
from rodit.core.models import Draft, EvidenceBundle
from rodit.qa import DefaultQualityGate
from rodit.testing.fakes import FakeClock, FakeDecisionProvider, make_context, qa_answers


class _BrokenAudit(InMemoryAuditSink):
    def log_decision(self, entry) -> None:  # noqa: ANN001
        raise StorageUnavailable("decision_log 写入失败：连接数已满 13812345678")


def test_decision_log_failure_keeps_the_qa_result(world, ev, route, caplog):
    gate = DefaultQualityGate(
        FakeDecisionProvider(qa_answers()), QAGateConfig(), _BrokenAudit(), FakeClock()
    )
    ctx = make_context("这个是纯棉的吗？", world, item_id="i_555")
    bundle = EvidenceBundle(evidences=[ev.item_attr("面料")])
    draft = Draft(
        text="亲，这款面料是棉 95%、氨纶 5% 哦",
        kind=DraftKind.LLM,
        evidence_ids=["item:i_555:attr:面料"],
    )
    with caplog.at_level(logging.WARNING):
        result = gate.check(ctx, route(Intent.PRODUCT_SPEC), draft, bundle)
    assert result.passed and result.model_checked and not result.degraded
    assert "StorageUnavailable" in caplog.text
    assert "13812345678" not in caplog.text  # 只记异常类型
