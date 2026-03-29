import pytest
from margin.contract import (
    TermStatus, Contract,
    HealthTarget, ReachHealth, SustainHealth,
    RecoveryThreshold, NoHarmful,
)
from margin.observation import Observation, Expression, Op
from margin.health import Health
from margin.confidence import Confidence
from margin.ledger import Record, Ledger


def _obs(name, health, value=50.0, baseline=100.0):
    return Observation(name, health, value, baseline, Confidence.HIGH)


def _expr(*observations):
    return Expression(observations=list(observations), confidence=Confidence.HIGH)


def _rec(step, name, health, value=50.0, fired=False, op=Op.NOOP,
         after_health=None, after_value=None):
    before = _obs(name, health, value)
    after = None
    if after_health:
        after = _obs(name, after_health, after_value or value)
    return Record(step=step, tag=f"s{step}", before=before, after=after,
                  fired=fired, op=op)


class TestHealthTarget:
    def test_met_when_intact(self):
        t = HealthTarget("api-intact", "api", Health.INTACT)
        r = t.evaluate(Ledger(), _expr(_obs("api", Health.INTACT, 90.0)))
        assert r.status == TermStatus.MET

    def test_met_or_better(self):
        t = HealthTarget("api-degraded-ok", "api", Health.DEGRADED, or_better=True)
        r = t.evaluate(Ledger(), _expr(_obs("api", Health.INTACT, 90.0)))
        assert r.status == TermStatus.MET  # INTACT is better than DEGRADED

    def test_violated(self):
        t = HealthTarget("api-intact", "api", Health.INTACT)
        r = t.evaluate(Ledger(), _expr(_obs("api", Health.ABLATED, 10.0)))
        assert r.status == TermStatus.VIOLATED

    def test_pending_no_expr(self):
        t = HealthTarget("api-intact", "api", Health.INTACT)
        r = t.evaluate(Ledger())
        assert r.status == TermStatus.PENDING

    def test_pending_component_missing(self):
        t = HealthTarget("api-intact", "api", Health.INTACT)
        r = t.evaluate(Ledger(), _expr(_obs("other", Health.INTACT)))
        assert r.status == TermStatus.PENDING

    def test_exact_match_no_or_better(self):
        t = HealthTarget("api-recovering", "api", Health.RECOVERING, or_better=False)
        r = t.evaluate(Ledger(), _expr(_obs("api", Health.INTACT, 90.0)))
        assert r.status == TermStatus.VIOLATED  # INTACT != RECOVERING exactly


class TestReachHealth:
    def test_met_when_reached(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.ABLATED, 10.0),
            _rec(1, "x", Health.DEGRADED, 50.0),
            _rec(2, "x", Health.INTACT, 90.0),
        ])
        t = ReachHealth("x-intact-in-5", "x", Health.INTACT, within_steps=5)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.MET
        assert "step 2" in r.detail

    def test_violated_when_window_passed(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED),
            _rec(1, "x", Health.DEGRADED),
            _rec(2, "x", Health.DEGRADED),
        ])
        t = ReachHealth("x-intact-in-3", "x", Health.INTACT, within_steps=3)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.VIOLATED

    def test_pending_not_enough_steps(self):
        ledger = Ledger(records=[_rec(0, "x", Health.DEGRADED)])
        t = ReachHealth("x-intact-in-5", "x", Health.INTACT, within_steps=5)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.PENDING

    def test_empty_ledger(self):
        t = ReachHealth("x-intact", "x", Health.INTACT, within_steps=3)
        r = t.evaluate(Ledger())
        assert r.status == TermStatus.PENDING


class TestSustainHealth:
    def test_met_when_sustained(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT, 90.0),
            _rec(1, "x", Health.INTACT, 92.0),
            _rec(2, "x", Health.INTACT, 88.0),
        ])
        t = SustainHealth("x-stable", "x", Health.INTACT, for_steps=3)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.MET

    def test_violated_when_interrupted(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT, 90.0),
            _rec(1, "x", Health.DEGRADED, 50.0),
            _rec(2, "x", Health.INTACT, 90.0),
        ])
        t = SustainHealth("x-stable", "x", Health.INTACT, for_steps=3)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.VIOLATED

    def test_or_better(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT, 90.0),
            _rec(1, "x", Health.INTACT, 95.0),
            _rec(2, "x", Health.RECOVERING, 70.0),  # RECOVERING is worse than INTACT
        ])
        t = SustainHealth("x-stable", "x", Health.INTACT, for_steps=3, or_better=True)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.VIOLATED

    def test_pending(self):
        ledger = Ledger(records=[_rec(0, "x", Health.INTACT, 90.0)])
        t = SustainHealth("x-stable", "x", Health.INTACT, for_steps=3)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.PENDING


class TestRecoveryThreshold:
    def test_met(self):
        ledger = Ledger(records=[
            Record(step=0, tag="s0", before=_obs("x", Health.DEGRADED),
                   after=_obs("x", Health.INTACT, 90.0), fired=True, op=Op.RESTORE),
        ])
        t = RecoveryThreshold("good-recovery", min_recovery=0.5, over_steps=5)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.MET

    def test_violated(self):
        ledger = Ledger(records=[
            Record(step=0, tag="s0", before=_obs("x", Health.DEGRADED),
                   after=_obs("x", Health.DEGRADED, 20.0), fired=True, op=Op.RESTORE),
            Record(step=1, tag="s1", before=_obs("x", Health.DEGRADED),
                   after=_obs("x", Health.DEGRADED, 25.0), fired=True, op=Op.RESTORE),
        ] * 3)  # 6 records, all low recovery
        t = RecoveryThreshold("good-recovery", min_recovery=0.8, over_steps=5)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.VIOLATED

    def test_pending_no_fires(self):
        ledger = Ledger(records=[_rec(0, "x", Health.INTACT, 90.0)])
        t = RecoveryThreshold("good-recovery", min_recovery=0.5, over_steps=5)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.PENDING


class TestNoHarmful:
    def test_met(self):
        ledger = Ledger(records=[
            Record(step=0, tag="s0", before=_obs("x", Health.DEGRADED),
                   after=_obs("x", Health.INTACT, 90.0), fired=True, op=Op.RESTORE),
        ])
        t = NoHarmful("no-harm", over_steps=5)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.MET

    def test_violated(self):
        ledger = Ledger(records=[
            Record(step=0, tag="s0", before=_obs("x", Health.DEGRADED, 50.0),
                   after=_obs("x", Health.ABLATED, 10.0), fired=True, op=Op.RESTORE),
        ])
        t = NoHarmful("no-harm", over_steps=5)
        r = t.evaluate(ledger)
        assert r.status == TermStatus.VIOLATED


class TestContract:
    def test_all_met(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.INTACT, 90.0),
            _rec(1, "x", Health.INTACT, 92.0),
            _rec(2, "x", Health.INTACT, 88.0),
        ])
        c = Contract("sla", terms=[
            HealthTarget("x-intact", "x", Health.INTACT),
            SustainHealth("x-stable", "x", Health.INTACT, for_steps=3),
        ])
        r = c.evaluate(ledger, _expr(_obs("x", Health.INTACT, 90.0)))
        assert r.all_met
        assert not r.any_violated

    def test_mixed(self):
        ledger = Ledger(records=[
            _rec(0, "x", Health.DEGRADED),
            _rec(1, "x", Health.DEGRADED),
        ])
        c = Contract("sla", terms=[
            HealthTarget("x-intact", "x", Health.INTACT),
            ReachHealth("x-reach", "x", Health.INTACT, within_steps=5),
        ])
        r = c.evaluate(ledger, _expr(_obs("x", Health.DEGRADED)))
        assert r.any_violated  # HealthTarget violated
        assert r.any_pending   # ReachHealth still pending

    def test_to_string(self):
        ledger = Ledger(records=[_rec(0, "x", Health.INTACT, 90.0)])
        c = Contract("test", terms=[HealthTarget("x-ok", "x", Health.INTACT)])
        r = c.evaluate(ledger, _expr(_obs("x", Health.INTACT, 90.0)))
        s = r.to_string()
        assert "[+]" in s  # MET indicator

    def test_to_dict(self):
        ledger = Ledger()
        c = Contract("test", terms=[HealthTarget("x-ok", "x", Health.INTACT)])
        r = c.evaluate(ledger, _expr(_obs("x", Health.INTACT, 90.0)))
        d = r.to_dict()
        assert d["all_met"] is True
        assert d["n_met"] == 1
