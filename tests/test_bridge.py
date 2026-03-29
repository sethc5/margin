import pytest
from margin.bridge import observe, observe_many, delta
from margin.uncertain import UncertainValue
from margin.health import Health, Thresholds
from margin.confidence import Confidence
from margin.observation import Op


class TestObserve:
    def test_clearly_intact(self):
        v = UncertainValue(point=95.0, uncertainty=0.5)
        bl = UncertainValue(point=100.0, uncertainty=1.0)
        t = Thresholds(intact=80.0, ablated=30.0)
        obs = observe("x", v, bl, t)
        assert obs.health == Health.INTACT
        assert obs.confidence >= Confidence.HIGH

    def test_clearly_ablated(self):
        v = UncertainValue(point=10.0, uncertainty=0.5)
        bl = UncertainValue(point=100.0, uncertainty=1.0)
        t = Thresholds(intact=80.0, ablated=30.0)
        obs = observe("x", v, bl, t)
        assert obs.health == Health.ABLATED

    def test_uncertainty_straddles_threshold(self):
        # 80 ± 5 straddles the intact=80 boundary
        v = UncertainValue(point=80.0, uncertainty=5.0)
        bl = UncertainValue(point=100.0, uncertainty=1.0)
        t = Thresholds(intact=80.0, ablated=30.0)
        obs = observe("x", v, bl, t)
        # Confidence should be INDETERMINATE since interval straddles intact
        assert obs.confidence == Confidence.INDETERMINATE
        # INDETERMINATE → OOD
        assert obs.health == Health.OOD

    def test_recovering_with_correction(self):
        v = UncertainValue(point=50.0, uncertainty=2.0)
        bl = UncertainValue(point=100.0, uncertainty=1.0)
        t = Thresholds(intact=80.0, ablated=30.0)
        obs = observe("x", v, bl, t, correction_magnitude=1.0)
        assert obs.health == Health.RECOVERING

    def test_polarity_lower_is_better(self):
        v = UncertainValue(point=0.005, uncertainty=0.001)
        bl = UncertainValue(point=0.01, uncertainty=0.001)
        t = Thresholds(intact=0.02, ablated=0.10, higher_is_better=False)
        obs = observe("err", v, bl, t)
        assert obs.health == Health.INTACT
        assert obs.higher_is_better is False
        assert obs.sigma > 0  # below baseline in lower-is-better = positive sigma

    def test_baseline_used_for_sigma(self):
        v = UncertainValue(point=50.0, uncertainty=1.0)
        bl = UncertainValue(point=100.0, uncertainty=1.0)
        t = Thresholds(intact=80.0, ablated=30.0)
        obs = observe("x", v, bl, t)
        assert obs.baseline == 100.0
        assert obs.sigma == pytest.approx(-0.5)

    def test_provenance_from_value(self):
        v = UncertainValue(point=50.0, uncertainty=1.0, provenance=["test-prov"])
        bl = UncertainValue(point=100.0, uncertainty=1.0)
        t = Thresholds(intact=80.0, ablated=30.0)
        obs = observe("x", v, bl, t)
        assert "test-prov" in obs.provenance


class TestObserveMany:
    def test_single_component(self):
        expr = observe_many(
            values={"x": UncertainValue(point=90.0, uncertainty=1.0)},
            baselines={"x": UncertainValue(point=100.0, uncertainty=1.0)},
            thresholds=Thresholds(intact=80.0, ablated=30.0),
        )
        assert expr.health_of("x") == Health.INTACT
        assert len(expr.corrections) == 0  # no correction magnitude

    def test_correction_targets_worst(self):
        expr = observe_many(
            values={
                "a": UncertainValue(point=90.0, uncertainty=1.0),
                "b": UncertainValue(point=20.0, uncertainty=1.0),
            },
            baselines={
                "a": UncertainValue(point=100.0, uncertainty=1.0),
                "b": UncertainValue(point=100.0, uncertainty=1.0),
            },
            thresholds=Thresholds(intact=80.0, ablated=30.0),
            correction_magnitude=1.0,
            alpha=0.5,
        )
        assert len(expr.corrections) == 1
        assert expr.corrections[0].target == "b"
        assert expr.corrections[0].op == Op.RESTORE

    def test_mixed_polarity(self):
        expr = observe_many(
            values={
                "throughput": UncertainValue(point=95.0, uncertainty=1.0),
                "err": UncertainValue(point=0.08, uncertainty=0.005),
            },
            baselines={
                "throughput": UncertainValue(point=100.0, uncertainty=1.0),
                "err": UncertainValue(point=0.01, uncertainty=0.001),
            },
            thresholds=Thresholds(intact=80.0, ablated=30.0),
            component_thresholds={
                "err": Thresholds(intact=0.02, ablated=0.10, higher_is_better=False),
            },
            correction_magnitude=1.0,
            alpha=0.6,
        )
        assert expr.health_of("throughput") == Health.INTACT
        assert expr.health_of("err") in (Health.DEGRADED, Health.RECOVERING)
        assert expr.corrections[0].target == "err"

    def test_net_confidence_is_weakest(self):
        expr = observe_many(
            values={
                "tight": UncertainValue(point=90.0, uncertainty=0.1),
                "loose": UncertainValue(point=50.0, uncertainty=30.0),
            },
            baselines={
                "tight": UncertainValue(point=100.0, uncertainty=0.1),
                "loose": UncertainValue(point=100.0, uncertainty=0.1),
            },
            thresholds=Thresholds(intact=80.0, ablated=30.0),
        )
        assert expr.confidence <= Confidence.MODERATE


class TestDelta:
    def test_returns_three_values(self):
        before = UncertainValue(point=50.0, uncertainty=2.0)
        after = UncertainValue(point=80.0, uncertainty=1.5)
        bl = UncertainValue(point=100.0, uncertainty=1.0)
        t = Thresholds(intact=80.0, ablated=30.0)
        obs_b, obs_a, diff = delta("x", before, after, bl, t)
        assert obs_b.value == 50.0
        assert obs_a.value == 80.0
        assert abs(diff.point - 30.0) < 0.001
        assert diff.uncertainty > 0

    def test_diff_uncertainty_propagated(self):
        before = UncertainValue(point=50.0, uncertainty=2.0)
        after = UncertainValue(point=80.0, uncertainty=3.0)
        bl = UncertainValue(point=100.0, uncertainty=1.0)
        t = Thresholds(intact=80.0, ablated=30.0)
        _, _, diff = delta("x", before, after, bl, t)
        # Independent: sqrt(4 + 9) = 3.606
        import math
        assert abs(diff.uncertainty - math.sqrt(13)) < 0.01

    def test_lower_is_better_delta(self):
        before = UncertainValue(point=0.08, uncertainty=0.005)
        after = UncertainValue(point=0.03, uncertainty=0.003)
        bl = UncertainValue(point=0.01, uncertainty=0.001)
        t = Thresholds(intact=0.02, ablated=0.10, higher_is_better=False)
        obs_b, obs_a, diff = delta("err", before, after, bl, t)
        assert diff.point == pytest.approx(-0.05)
        assert obs_b.higher_is_better is False
        assert obs_a.higher_is_better is False
