import pytest
from datetime import datetime, timedelta
from margin.transitions import track, track_all, Span, Transition, ComponentHistory
from margin.observation import Observation, Op
from margin.health import Health
from margin.confidence import Confidence
from margin.ledger import Record, Ledger


def _obs(name, health, value, baseline=100.0, hib=True):
    return Observation(name, health, value, baseline, Confidence.HIGH, higher_is_better=hib)


def _rec(step, name, health_before, health_after=None, fired=False, ts=None):
    before = _obs(name, health_before, 50.0)
    after = _obs(name, health_after, 75.0) if health_after else None
    return Record(
        step=step, tag=f"s{step}",
        before=before, after=after,
        fired=fired, op=Op.RESTORE if fired else Op.NOOP,
        timestamp=ts or datetime(2026, 1, 1, 0, 0, step),
    )


class TestTrack:
    def test_single_state(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT),
            _rec(1, "x", Health.INTACT),
            _rec(2, "x", Health.INTACT),
        ])
        h = track(ledger, "x")
        assert h.name == "x"
        assert len(h.spans) == 1
        assert h.spans[0].health == Health.INTACT
        assert h.spans[0].n_steps == 3
        assert h.n_transitions == 0

    def test_one_transition(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT),
            _rec(1, "x", Health.INTACT),
            _rec(2, "x", Health.DEGRADED),
            _rec(3, "x", Health.DEGRADED),
        ])
        h = track(ledger, "x")
        assert len(h.spans) == 2
        assert h.spans[0].health == Health.INTACT
        assert h.spans[0].n_steps == 2
        assert h.spans[1].health == Health.DEGRADED
        assert h.spans[1].n_steps == 2
        assert h.n_transitions == 1
        assert h.transitions[0].from_health == Health.INTACT
        assert h.transitions[0].to_health == Health.DEGRADED

    def test_multiple_transitions(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT),
            _rec(1, "x", Health.DEGRADED),
            _rec(2, "x", Health.ABLATED),
            _rec(3, "x", Health.RECOVERING, Health.RECOVERING, fired=True),
            _rec(4, "x", Health.INTACT),
        ])
        h = track(ledger, "x")
        assert h.n_transitions == 4
        assert len(h.spans) == 5

    def test_transition_counts(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT),
            _rec(1, "x", Health.DEGRADED),
            _rec(2, "x", Health.INTACT),
            _rec(3, "x", Health.DEGRADED),
        ])
        h = track(ledger, "x")
        counts = h.transition_counts()
        assert counts[("INTACT", "DEGRADED")] == 2
        assert counts[("DEGRADED", "INTACT")] == 1

    def test_time_in_state(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT),
            _rec(1, "x", Health.INTACT),
            _rec(2, "x", Health.INTACT),
            _rec(3, "x", Health.DEGRADED),
            _rec(4, "x", Health.DEGRADED),
        ])
        h = track(ledger, "x")
        t = h.time_in_state()
        assert t["INTACT"] == 3
        assert t["DEGRADED"] == 2

    def test_last_health(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT),
            _rec(1, "x", Health.DEGRADED),
        ])
        h = track(ledger, "x")
        assert h.last_health() == Health.DEGRADED

    def test_empty_ledger(self):
        h = track(Ledger(), "x")
        assert len(h.spans) == 0
        assert h.n_transitions == 0
        assert h.last_health() is None

    def test_ignores_other_components(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT),
            _rec(1, "y", Health.DEGRADED),
            _rec(2, "x", Health.DEGRADED),
        ])
        h = track(ledger, "x")
        assert len(h.spans) == 2
        assert h.n_transitions == 1

    def test_to_dict(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT),
            _rec(1, "x", Health.DEGRADED),
        ])
        d = track(ledger, "x").to_dict()
        assert d["name"] == "x"
        assert d["n_transitions"] == 1
        assert "INTACT" in d["time_in_state"]
        assert len(d["spans"]) == 2

    def test_uses_after_when_available(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, Health.RECOVERING, fired=True),
        ])
        h = track(ledger, "x")
        assert h.spans[0].health == Health.RECOVERING


class TestTrackAll:
    def test_finds_all_components(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT),
            _rec(1, "y", Health.DEGRADED),
            _rec(2, "x", Health.DEGRADED),
        ])
        histories = track_all(ledger)
        assert set(histories.keys()) == {"x", "y"}
        assert histories["x"].n_transitions == 1
        assert histories["y"].n_transitions == 0


class TestSpan:
    def test_duration(self):
        s = Span(
            Health.INTACT, 0, 5,
            start_time=datetime(2026, 1, 1, 0, 0, 0),
            end_time=datetime(2026, 1, 1, 0, 0, 5),
        )
        assert s.duration == timedelta(seconds=5)
        assert s.n_steps == 6

    def test_no_duration_without_times(self):
        s = Span(Health.INTACT, 0, 5)
        assert s.duration is None
