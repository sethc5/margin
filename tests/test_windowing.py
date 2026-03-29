import pytest
from datetime import datetime, timedelta
from margin.observation import Observation, Op
from margin.health import Health
from margin.confidence import Confidence
from margin.ledger import Record, Ledger


def _obs(name="x", health=Health.INTACT):
    return Observation(name, health, 90.0, 100.0, Confidence.HIGH)


def _rec(step, name="x", health=Health.INTACT, ts=None):
    return Record(
        step=step, tag=f"s{step}",
        before=_obs(name, health),
        timestamp=ts or datetime(2026, 1, 1, 0, 0, step),
    )


class TestWindow:
    def test_window_filters_by_time(self):
        now = datetime(2026, 1, 1, 0, 1, 0)  # 60 seconds in
        ledger = Ledger(records=[
            _rec(0, ts=datetime(2026, 1, 1, 0, 0, 0)),   # 60s ago
            _rec(1, ts=datetime(2026, 1, 1, 0, 0, 30)),  # 30s ago
            _rec(2, ts=datetime(2026, 1, 1, 0, 0, 50)),  # 10s ago
        ])
        w = ledger.window(timedelta(seconds=35), now=now)
        assert len(w) == 2
        assert w.records[0].step == 1

    def test_window_preserves_label(self):
        ledger = Ledger(label="test", records=[_rec(0)])
        w = ledger.window(timedelta(hours=1))
        assert w.label == "test"

    def test_window_empty_when_all_old(self):
        old = datetime(2020, 1, 1)
        ledger = Ledger(records=[_rec(0, ts=old)])
        w = ledger.window(timedelta(seconds=10))
        assert len(w) == 0

    def test_windowed_stats_work(self):
        now = datetime(2026, 1, 1, 0, 1, 0)
        ledger = Ledger(records=[
            _rec(0, ts=datetime(2026, 1, 1, 0, 0, 0)),
            _rec(1, ts=datetime(2026, 1, 1, 0, 0, 50)),
        ])
        w = ledger.window(timedelta(seconds=15), now=now)
        assert len(w) == 1
        assert w.fire_rate == 0.0


class TestLastN:
    def test_last_n(self):
        ledger = Ledger(records=[_rec(i) for i in range(10)])
        w = ledger.last_n(3)
        assert len(w) == 3
        assert w.records[0].step == 7

    def test_last_n_more_than_available(self):
        ledger = Ledger(records=[_rec(0), _rec(1)])
        w = ledger.last_n(10)
        assert len(w) == 2

    def test_last_n_zero(self):
        ledger = Ledger(records=[_rec(0)])
        w = ledger.last_n(0)
        assert len(w) == 0


class TestForComponent:
    def test_filters_to_component(self):
        ledger = Ledger(records=[
            _rec(0, "x"),
            _rec(1, "y"),
            _rec(2, "x"),
            _rec(3, "y"),
            _rec(4, "x"),
        ])
        w = ledger.for_component("x")
        assert len(w) == 3
        assert all(r.before.name == "x" for r in w.records)

    def test_empty_for_missing_component(self):
        ledger = Ledger(records=[_rec(0, "x")])
        w = ledger.for_component("nonexistent")
        assert len(w) == 0

    def test_for_component_stats(self):
        ledger = Ledger(records=[
            _rec(0, "x"),
            _rec(1, "y"),
            _rec(2, "x"),
        ])
        w = ledger.for_component("x")
        assert w.fire_rate == 0.0
        assert len(w) == 2
