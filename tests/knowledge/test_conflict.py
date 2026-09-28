"""冲突处理（§5.2.1）与长度预算（§14）的单元测试。"""

from __future__ import annotations

from rodit.core.enums import EvidenceSource
from rodit.core.models import Evidence
from rodit.knowledge.conflict import (
    STRUCTURED_RANK,
    Candidate,
    content_chars,
    dedupe,
    ordered_topics,
    resolve_conflicts,
    trim_to_budget,
)


def _e(eid: str, source: EvidenceSource, *, topic: str | None = None, chars: int = 10, **kw):
    return Evidence(id=eid, source=source, label=eid, content="字" * chars, topic=topic, **kw)


def test_candidate_always_includes_evidence_topic():
    c = Candidate(_e("policy:ship_time", EvidenceSource.POLICY, topic="ship_time"))
    assert c.topics == {"ship_time"} and c.rank == STRUCTURED_RANK
    c2 = Candidate(
        _e("item:i:detail:0", EvidenceSource.ITEM, topic="ship_time"), topics={"courier"}
    )
    assert c2.topics == {"ship_time", "courier"}
    tool = Candidate.of(_e("faq:1", EvidenceSource.FAQ, score=0.7))
    assert tool.rank == 0.7 and tool.topics == frozenset()


def test_higher_priority_wins_and_losers_are_recorded_in_order():
    cands = [
        Candidate(_e("item:i:detail:1", EvidenceSource.ITEM, topic="ship_time")),
        Candidate(_e("faq:2", EvidenceSource.FAQ, topic="ship_time")),
        Candidate(_e("policy:ship_time", EvidenceSource.POLICY, topic="ship_time")),
        Candidate(_e("item:i:detail:2", EvidenceSource.ITEM)),
    ]
    kept, lost = resolve_conflicts(cands)
    assert [c.evidence.id for c in kept] == ["policy:ship_time", "item:i:detail:2"]
    assert lost == ["item:i:detail:1", "faq:2"]


def test_same_priority_does_not_override():
    cands = [
        Candidate(_e("policy:a", EvidenceSource.POLICY, topic="x")),
        Candidate(_e("policy:b", EvidenceSource.POLICY, topic="x")),
    ]
    kept, lost = resolve_conflicts(cands)
    assert len(kept) == 2 and lost == []


def test_multi_topic_sentence_removed_when_any_topic_is_taken():
    seg = Candidate(
        _e("item:i:detail:0", EvidenceSource.ITEM, topic="ship_time"),
        topics=frozenset({"ship_time", "courier"}),
    )
    courier = Candidate(_e("policy:courier", EvidenceSource.POLICY, topic="courier"))
    kept, lost = resolve_conflicts([seg, courier])
    assert lost == ["item:i:detail:0"] and [c.evidence.id for c in kept] == ["policy:courier"]


def test_realtime_overrides_policy_on_shared_topic():
    kept, lost = resolve_conflicts(
        [
            Candidate(_e("policy:promotions", EvidenceSource.POLICY, topic="promotions")),
            Candidate.of(_e("tool:price:i_1", EvidenceSource.REALTIME, topic="promotions")),
        ]
    )
    assert lost == ["policy:promotions"] and kept[0].evidence.id == "tool:price:i_1"


def test_dedupe_keeps_first():
    a = Candidate(_e("x", EvidenceSource.ITEM, chars=1))
    b = Candidate(_e("x", EvidenceSource.ITEM, chars=2))
    assert dedupe([a, b]) == [a]


def test_trim_drops_lowest_priority_then_lowest_rank_then_latest():
    cands = [
        Candidate(_e("policy:a", EvidenceSource.POLICY, chars=40)),
        Candidate(_e("item:i:attr:a", EvidenceSource.ITEM, chars=40)),
        Candidate(_e("item:i:detail:0", EvidenceSource.ITEM, chars=40), rank=0.9),
        Candidate(_e("item:i:detail:1", EvidenceSource.ITEM, chars=40), rank=0.1),
        Candidate(_e("faq:1", EvidenceSource.FAQ, chars=40), rank=0.9),
        Candidate(_e("faq:2", EvidenceSource.FAQ, chars=40), rank=0.4),
        Candidate(_e("category:c", EvidenceSource.CATEGORY, chars=40), rank=0.0),
    ]
    kept = trim_to_budget(cands, 150)
    assert [c.evidence.id for c in kept] == ["policy:a", "item:i:attr:a", "item:i:detail:0"]
    assert content_chars(kept) <= 150
    # 预算足够时原样返回
    assert trim_to_budget(cands, 10_000) == cands
    # 同优先级同 rank：靠后的先删
    same = [Candidate(_e(f"item:i:attr:{n}", EvidenceSource.ITEM, chars=40)) for n in "abc"]
    assert [c.evidence.id for c in trim_to_budget(same, 80)] == ["item:i:attr:a", "item:i:attr:b"]


def test_trim_keeps_at_least_one_evidence():
    cands = [
        Candidate(_e("tool:order:o_1", EvidenceSource.REALTIME, chars=300)),
        Candidate(_e("policy:a", EvidenceSource.POLICY, chars=300)),
    ]
    kept = trim_to_budget(cands, 10)
    assert [c.evidence.id for c in kept] == ["tool:order:o_1"]
    assert trim_to_budget([], 10) == []


def test_ordered_topics_is_deterministic():
    assert ordered_topics({"courier", "ship_time"}) == ["ship_time", "courier"]
    assert ordered_topics({"zzz", "care", "presale"}) == ["presale", "care", "zzz"]
