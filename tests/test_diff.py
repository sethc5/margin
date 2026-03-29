import pytest
from margin.diff import diff, ComponentChange, Diff
from margin.observation import Observation, Expression
from margin.health import Health
from margin.confidence import Confidence


def _obs(name, health, value, baseline=100.0):
    return Observation(name, health, value, baseline, Confidence.HIGH)


class TestDiff:
    def test_no_changes(self):
        e = Expression(observations=[_obs("a", Health.INTACT, 90.0)])
        d = diff(e, e)
        assert not d.any_health_changed
        assert len(d.changes) == 1
        assert d.changes[0].name == "a"

    def test_health_change_detected(self):
        before = Expression(observations=[_obs("a", Health.INTACT, 90.0)])
        after = Expression(observations=[_obs("a", Health.DEGRADED, 50.0)])
        d = diff(before, after)
        assert d.any_health_changed
        assert d.changes[0].health_before == Health.INTACT
        assert d.changes[0].health_after == Health.DEGRADED

    def test_worsened(self):
        before = Expression(observations=[_obs("a", Health.INTACT, 90.0)])
        after = Expression(observations=[_obs("a", Health.ABLATED, 10.0)])
        d = diff(before, after)
        assert d.any_worsened
        assert len(d.worsened()) == 1

    def test_improved(self):
        before = Expression(observations=[_obs("a", Health.ABLATED, 10.0)])
        after = Expression(observations=[_obs("a", Health.INTACT, 90.0)])
        d = diff(before, after)
        assert d.any_improved
        assert len(d.improved()) == 1

    def test_component_appeared(self):
        before = Expression(observations=[_obs("a", Health.INTACT, 90.0)])
        after = Expression(observations=[
            _obs("a", Health.INTACT, 90.0),
            _obs("b", Health.DEGRADED, 50.0),
        ])
        d = diff(before, after)
        assert len(d.appeared()) == 1
        assert d.appeared()[0].name == "b"

    def test_component_disappeared(self):
        before = Expression(observations=[
            _obs("a", Health.INTACT, 90.0),
            _obs("b", Health.DEGRADED, 50.0),
        ])
        after = Expression(observations=[_obs("a", Health.INTACT, 90.0)])
        d = diff(before, after)
        assert len(d.disappeared()) == 1
        assert d.disappeared()[0].name == "b"

    def test_sigma_delta(self):
        before = Expression(observations=[_obs("a", Health.INTACT, 90.0)])
        after = Expression(observations=[_obs("a", Health.INTACT, 95.0)])
        d = diff(before, after)
        c = d.changes[0]
        assert not c.health_changed
        assert c.sigma_delta == pytest.approx(0.05)

    def test_empty_expressions(self):
        d = diff(Expression(), Expression())
        assert len(d.changes) == 0

    def test_to_string(self):
        before = Expression(observations=[_obs("a", Health.INTACT, 90.0)])
        after = Expression(observations=[_obs("a", Health.DEGRADED, 50.0)])
        d = diff(before, after)
        assert "INTACT" in d.to_string()
        assert "DEGRADED" in d.to_string()
        assert "→" in d.to_string()

    def test_to_dict(self):
        before = Expression(observations=[_obs("a", Health.INTACT, 90.0)])
        after = Expression(observations=[_obs("a", Health.DEGRADED, 50.0)])
        d = diff(before, after).to_dict()
        assert "changes" in d
        assert d["changes"][0]["health_changed"] is True
        assert d["changes"][0]["worsened"] is True

    def test_confidence_tracked(self):
        before = Expression(confidence=Confidence.HIGH)
        after = Expression(confidence=Confidence.LOW)
        d = diff(before, after)
        assert d.confidence_before == Confidence.HIGH
        assert d.confidence_after == Confidence.LOW


class TestComponentChange:
    def test_unchanged_sigma_string(self):
        c = ComponentChange("x", Health.INTACT, Health.INTACT, 0.5, 0.5)
        assert "unchanged" in c.to_string()

    def test_sigma_drift_string(self):
        c = ComponentChange("x", Health.INTACT, Health.INTACT, 0.5, 0.3)
        assert "σ" in c.to_string()

    def test_absent_before(self):
        c = ComponentChange("x", None, Health.INTACT, None, 0.5)
        assert c.appeared
        assert not c.disappeared
        assert c.sigma_delta is None
