import pytest
from datetime import datetime
from margin.policy.temporal import (
    health_sustained, health_for_at_least,
    sigma_trending_below, fire_rate_above, no_improvement,
)
from margin.observation import Observation, Expression, Op
from margin.health import Health
from margin.confidence import Confidence
from margin.ledger import Record, Ledger


def _obs(name, health, value=50.0, baseline=100.0):
    return Observation(name, health, value, baseline, Confidence.HIGH)


def _rec(step, name, health, value=50.0, fired=False, op=Op.NOOP):
    before = _obs(name, health, value)
    after = _obs(name, health, value) if fired else None
    return Record(step=step, tag=f"s{step}", before=before, after=after,
                  fired=fired, op=op, timestamp=datetime(2026, 1, 1, 0, 0, step))


def _expr(name, health, value=50.0):
    return Expression(observations=[_obs(name, health, value)], confidence=Confidence.HIGH)


class TestHealthSustained:
    def test_true_when_sustained(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED),
            _rec(1, "x", Health.DEGRADED),
            _rec(2, "x", Health.DEGRADED),
        ])
        pred = health_sustained("x", Health.DEGRADED, 3, ledger)
        assert pred(_expr("x", Health.DEGRADED)) is True

    def test_false_when_not_enough_steps(self):
        ledger = Ledger(records=[_rec(0, "x", Health.DEGRADED)])
        pred = health_sustained("x", Health.DEGRADED, 3, ledger)
        assert pred(_expr("x", Health.DEGRADED)) is False

    def test_false_when_interrupted(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED),
            _rec(1, "x", Health.INTACT, 90.0),
            _rec(2, "x", Health.DEGRADED),
        ])
        pred = health_sustained("x", Health.DEGRADED, 3, ledger)
        assert pred(_expr("x", Health.DEGRADED)) is False

    def test_false_when_current_different(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED),
            _rec(1, "x", Health.DEGRADED),
            _rec(2, "x", Health.DEGRADED),
        ])
        pred = health_sustained("x", Health.DEGRADED, 3, ledger)
        assert pred(_expr("x", Health.INTACT, 90.0)) is False


class TestHealthForAtLeast:
    def test_any_of_set(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED),
            _rec(1, "x", Health.ABLATED, 10.0),
            _rec(2, "x", Health.DEGRADED),
        ])
        pred = health_for_at_least("x", {Health.DEGRADED, Health.ABLATED}, 3, ledger)
        assert pred(_expr("x", Health.DEGRADED)) is True

    def test_false_when_intact_interrupts(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED),
            _rec(1, "x", Health.INTACT, 90.0),
            _rec(2, "x", Health.DEGRADED),
        ])
        pred = health_for_at_least("x", {Health.DEGRADED, Health.ABLATED}, 3, ledger)
        assert pred(_expr("x", Health.DEGRADED)) is False


class TestSigmaTrendingBelow:
    def test_true_when_all_below(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, 40.0),  # sigma = -0.6
            _rec(1, "x", Health.DEGRADED, 45.0),  # sigma = -0.55
            _rec(2, "x", Health.DEGRADED, 42.0),  # sigma = -0.58
        ])
        pred = sigma_trending_below("x", -0.3, 3, ledger)
        assert pred(_expr("x", Health.DEGRADED, 43.0)) is True  # sigma = -0.57

    def test_false_when_one_above(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, 40.0),
            _rec(1, "x", Health.INTACT, 90.0),  # sigma = -0.1
            _rec(2, "x", Health.DEGRADED, 42.0),
        ])
        pred = sigma_trending_below("x", -0.3, 3, ledger)
        assert pred(_expr("x", Health.DEGRADED, 43.0)) is False


class TestFireRateAbove:
    def test_high_fire_rate(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
            _rec(1, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
            _rec(2, "x", Health.DEGRADED),
        ])
        pred = fire_rate_above("x", 0.5, 3, ledger)
        assert pred(_expr("x", Health.DEGRADED)) is True  # 2/3 > 0.5

    def test_low_fire_rate(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED),
            _rec(1, "x", Health.DEGRADED),
            _rec(2, "x", Health.DEGRADED, fired=True, op=Op.RESTORE),
        ])
        pred = fire_rate_above("x", 0.5, 3, ledger)
        assert pred(_expr("x", Health.DEGRADED)) is False  # 1/3 < 0.5


class TestNoImprovement:
    def test_true_when_corrections_not_helping(self):
        ledger = Ledger(records=[
            Record(step=0, tag="s0",
                   before=_obs("x", Health.DEGRADED, 50.0),
                   after=_obs("x", Health.DEGRADED, 48.0),
                   fired=True, op=Op.RESTORE),
            Record(step=1, tag="s1",
                   before=_obs("x", Health.DEGRADED, 48.0),
                   after=_obs("x", Health.DEGRADED, 45.0),
                   fired=True, op=Op.RESTORE),
        ])
        pred = no_improvement("x", 5, ledger)
        assert pred(_expr("x", Health.DEGRADED)) is True

    def test_false_when_no_corrections(self):
        ledger = Ledger(records=[_rec(0, "x", Health.DEGRADED)])
        pred = no_improvement("x", 5, ledger)
        assert pred(_expr("x", Health.DEGRADED)) is False
