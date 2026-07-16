import pandas as pd
import pytest

from copa import CandidateRecord, CandidateStateBus, CandidateTracker


def candidates():
    return [
        CandidateRecord("a", 0.9, {"price": 10}),
        CandidateRecord("b", 0.8, {"price": 20}),
    ]


def test_bus_versions_defensive_queries_and_non_destructive_rollback(tmp_path):
    tracker = CandidateTracker("run", "user", tmp_path / "trace.jsonl")
    bus = CandidateStateBus(tracker)
    bus.initialize(candidates())
    queried = bus.query()
    queried.at[0, "metadata"]["price"] = 999
    assert bus.query().at[0, "metadata"]["price"] == 10

    bus.update(
        lambda frame: frame.assign(active=[True, False]),
        module="test",
        operation="deactivate",
    )
    assert bus.version == 1
    assert len(bus.query(active_only=True)) == 1
    bus.rollback(0, "test")
    assert bus.version == 2
    assert bus.history_versions == [0, 1, 2]
    assert len(bus.query(active_only=True)) == 2
    assert len(tracker.events) == 3
    assert (tmp_path / "trace.jsonl").read_text(encoding="utf-8").count("\n") == 3


def test_bus_rejects_duplicates_and_failed_update_is_atomic():
    bus = CandidateStateBus()
    with pytest.raises(ValueError, match="Duplicate"):
        bus.initialize([CandidateRecord("a", 1.0), CandidateRecord("a", 0.5)])

    bus.initialize(candidates())
    with pytest.raises(RuntimeError, match="boom"):
        bus.update(
            lambda frame: (_ for _ in ()).throw(RuntimeError("boom")),
            module="test",
            operation="failure",
        )
    assert bus.version == 0
    assert bus.query()["item_id"].tolist() == ["a", "b"]
