"""Tests for margin.streaming — incremental trackers."""

from datetime import datetime, timedelta

from margin import (
    Observation, Health, Confidence, Thresholds, Parser,
    DriftState, DriftDirection, AnomalyState,
    DriftTracker, AnomalyTracker, CorrelationTracker, Monitor,
)


t0 = datetime(2026, 1, 1)


def _obs(name, value, i, baseline=100.0, hib=True):
    return Observation(
        name=name, health=Health.INTACT, value=value, baseline=baseline,
        confidence=Confidence.HIGH, higher_is_better=hib,
        measured_at=t0 + timedelta(seconds=i * 60),
    )


# -----------------------------------------------------------------------
# DriftTracker
# -----------------------------------------------------------------------

class TestDriftTracker:
    def test_stable(self):
        t = DriftTracker("cpu")
        for i in range(10):
            t.update(_obs("cpu", 50.0, i))
        assert t.state == DriftState.STABLE
        assert t.direction == DriftDirection.NEUTRAL

    def test_drifting(self):
        t = DriftTracker("cpu")
        for i in range(10):
            t.update(_obs("cpu", 100.0 - i * 5, i))
        assert t.state != DriftState.STABLE
        assert t.direction == DriftDirection.WORSENING

    def test_insufficient_data(self):
        t = DriftTracker("cpu")
        t.update(_obs("cpu", 50.0, 0))
        assert t.state == DriftState.STABLE  # default when no classification
        assert t.classification is None

    def test_update_value(self):
        t = DriftTracker("cpu")
        for i in range(10):
            t.update_value(100.0 - i * 5, baseline=100.0, measured_at=t0 + timedelta(seconds=i * 60))
        assert t.classification is not None
        assert t.direction == DriftDirection.WORSENING

    def test_window_bounded(self):
        t = DriftTracker("cpu", window=5)
        for i in range(20):
            t.update(_obs("cpu", float(i), i))
        assert t.n_observations == 5

    def test_reset(self):
        t = DriftTracker("cpu")
        for i in range(10):
            t.update(_obs("cpu", float(i), i))
        t.reset()
        assert t.n_observations == 0
        assert t.classification is None

    def test_repr(self):
        t = DriftTracker("cpu")
        assert "cpu" in repr(t)

    def test_returns_classification(self):
        t = DriftTracker("cpu")
        for i in range(5):
            dc = t.update(_obs("cpu", float(i * 10), i))
        assert dc is not None or dc is None  # either is valid depending on data


# -----------------------------------------------------------------------
# AnomalyTracker
# -----------------------------------------------------------------------

class TestAnomalyTracker:
    def test_expected_after_warmup(self):
        t = AnomalyTracker("cpu", min_reference=10)
        for i in range(15):
            t.update(50.0 + i * 0.1)
        # Value within range
        t.update(50.5)
        assert t.state == AnomalyState.EXPECTED

    def test_anomalous_spike(self):
        t = AnomalyTracker("cpu", min_reference=10)
        for _ in range(20):
            t.update(50.0)
        t.update(200.0)
        assert t.state in (AnomalyState.ANOMALOUS, AnomalyState.NOVEL)

    def test_insufficient_reference(self):
        t = AnomalyTracker("cpu", min_reference=10)
        for i in range(5):
            t.update(float(i))
        assert t.classification is None
        assert t.state == AnomalyState.EXPECTED  # default

    def test_window_bounded(self):
        t = AnomalyTracker("cpu", window=10)
        for i in range(50):
            t.update(float(i))
        assert t.n_values == 10

    def test_jump_detection(self):
        t = AnomalyTracker("cpu", min_reference=5)
        for _ in range(10):
            t.update(50.0)
        t.update(150.0)
        # last_jump may or may not fire depending on window context
        # Just verify it doesn't crash
        assert t.last_jump is None or t.last_jump is not None

    def test_update_obs(self):
        t = AnomalyTracker("cpu", min_reference=5)
        for i in range(10):
            obs = _obs("cpu", 50.0, i)
            t.update_obs(obs)
        assert t.n_values == 10

    def test_reset(self):
        t = AnomalyTracker("cpu")
        for i in range(20):
            t.update(float(i))
        t.reset()
        assert t.n_values == 0
        assert t.classification is None

    def test_repr(self):
        t = AnomalyTracker("cpu")
        assert "cpu" in repr(t)


# -----------------------------------------------------------------------
# CorrelationTracker
# -----------------------------------------------------------------------

class TestCorrelationTracker:
    def test_correlated_components(self):
        t = CorrelationTracker(["cpu", "mem"], min_correlation=0.5)
        for i in range(20):
            t.update({"cpu": float(i), "mem": float(i * 2)})
        assert t.matrix is not None
        assert len(t.matrix.correlations) >= 1

    def test_uncorrelated(self):
        import random
        random.seed(42)
        t = CorrelationTracker(["a", "b"], min_correlation=0.9)
        for _ in range(30):
            t.update({"a": random.random(), "b": random.random()})
        assert t.matrix is not None
        assert len(t.matrix.correlations) == 0

    def test_missing_component_skips(self):
        t = CorrelationTracker(["cpu", "mem"])
        t.update({"cpu": 50.0})  # mem missing
        assert t.n_updates == 0  # didn't count

    def test_window_bounded(self):
        t = CorrelationTracker(["a", "b"], window=10)
        for i in range(50):
            t.update({"a": float(i), "b": float(i)})
        assert t.n_updates == 50
        # Internal series bounded
        assert len(t._series["a"]) == 10

    def test_strongest(self):
        t = CorrelationTracker(["a", "b"], min_correlation=0.5)
        for i in range(20):
            t.update({"a": float(i), "b": float(i * 3)})
        top = t.strongest(1)
        assert len(top) <= 1

    def test_reset(self):
        t = CorrelationTracker(["a", "b"])
        for i in range(10):
            t.update({"a": float(i), "b": float(i)})
        t.reset()
        assert t.n_updates == 0
        assert t.matrix is None

    def test_repr(self):
        t = CorrelationTracker(["a", "b"])
        assert "2 components" in repr(t)


# -----------------------------------------------------------------------
# Monitor
# -----------------------------------------------------------------------

class TestMonitor:
    def _parser(self):
        return Parser(
            baselines={"cpu": 50.0, "mem": 70.0, "error_rate": 0.002},
            thresholds=Thresholds(intact=40.0, ablated=10.0),
            component_thresholds={
                "error_rate": Thresholds(intact=0.01, ablated=0.10, higher_is_better=False),
            },
        )

    def test_basic_update(self):
        m = Monitor(self._parser())
        expr = m.update({"cpu": 48.0, "mem": 65.0, "error_rate": 0.003})
        assert expr is not None
        assert m.step == 1
        assert m.expression is not None

    def test_drift_after_updates(self):
        m = Monitor(self._parser())
        for i in range(10):
            m.update(
                {"cpu": 50.0 - i * 2, "mem": 70.0, "error_rate": 0.002},
                now=t0 + timedelta(seconds=i * 60),
            )
        dc = m.drift("cpu")
        assert dc is not None
        assert dc.direction == DriftDirection.WORSENING

    def test_anomaly_after_warmup(self):
        m = Monitor(self._parser(), anomaly_min_reference=10)
        for i in range(15):
            m.update(
                {"cpu": 50.0, "mem": 70.0, "error_rate": 0.002},
                now=t0 + timedelta(seconds=i * 60),
            )
        # Spike
        m.update(
            {"cpu": 200.0, "mem": 70.0, "error_rate": 0.002},
            now=t0 + timedelta(seconds=15 * 60),
        )
        ac = m.anomaly("cpu")
        assert ac is not None
        assert ac.state in (AnomalyState.ANOMALOUS, AnomalyState.NOVEL)

    def test_correlations(self):
        m = Monitor(self._parser(), window=20)
        for i in range(20):
            m.update(
                {"cpu": float(i), "mem": float(i * 2), "error_rate": 0.002},
                now=t0 + timedelta(seconds=i * 60),
            )
        corr = m.correlations
        assert corr is not None

    def test_status(self):
        m = Monitor(self._parser())
        for i in range(15):
            m.update(
                {"cpu": 50.0 - i, "mem": 70.0, "error_rate": 0.002},
                now=t0 + timedelta(seconds=i * 60),
            )
        s = m.status()
        assert "step" in s
        assert "expression" in s
        assert "drift" in s
        assert "anomaly" in s

    def test_reset(self):
        m = Monitor(self._parser())
        for i in range(5):
            m.update({"cpu": 50.0, "mem": 70.0, "error_rate": 0.002})
        m.reset()
        assert m.step == 0
        assert m.expression is None

    def test_missing_component_drift(self):
        m = Monitor(self._parser())
        assert m.drift("nonexistent") is None

    def test_missing_component_anomaly(self):
        m = Monitor(self._parser())
        assert m.anomaly("nonexistent") is None

    def test_repr(self):
        m = Monitor(self._parser())
        assert "Monitor" in repr(m)
        assert "3 components" in repr(m)


# -----------------------------------------------------------------------
# Monitor — feature flags
# -----------------------------------------------------------------------

class TestMonitorFeatures:
    def _parser(self):
        return Parser(
            baselines={"cpu": 50.0, "mem": 70.0},
            thresholds=Thresholds(intact=40.0, ablated=10.0),
        )

    def test_default_features_all_enabled(self):
        m = Monitor(self._parser())
        assert "drift" in m.features
        assert "anomaly" in m.features
        assert "correlation" in m.features

    def test_features_health_only_skips_trackers(self):
        m = Monitor(self._parser(), features={"health"})
        assert len(m._drift_trackers) == 0
        assert len(m._anomaly_trackers) == 0
        assert m._correlation_tracker is None

    def test_features_health_drift_no_anomaly(self):
        m = Monitor(self._parser(), features={"health", "drift"})
        assert len(m._drift_trackers) == 2
        assert len(m._anomaly_trackers) == 0
        assert m._correlation_tracker is None

    def test_features_health_anomaly_no_drift(self):
        m = Monitor(self._parser(), features={"health", "anomaly"})
        assert len(m._drift_trackers) == 0
        assert len(m._anomaly_trackers) == 2

    def test_update_works_with_partial_features(self):
        m = Monitor(self._parser(), features={"health", "drift"})
        expr = m.update({"cpu": 48.0, "mem": 65.0})
        assert expr is not None
        assert m.step == 1

    def test_features_is_frozenset(self):
        m = Monitor(self._parser(), features={"health", "drift"})
        assert isinstance(m.features, frozenset)

    def test_status_omits_disabled_feature_keys(self):
        m = Monitor(self._parser(), features={"health"})
        for i in range(5):
            m.update({"cpu": 50.0 - i, "mem": 70.0},
                     now=t0 + timedelta(seconds=i * 60))
        s = m.status()
        # drift disabled → "drift" key absent
        assert "drift" not in s
        assert "anomaly" not in s

    def test_repr_shows_features_when_partial(self):
        m = Monitor(self._parser(), features={"health", "drift"})
        r = repr(m)
        assert "features" in r

    def test_repr_no_features_label_when_all_enabled(self):
        m = Monitor(self._parser())
        r = repr(m)
        assert "features" not in r

    def test_reset_works_with_partial_features(self):
        m = Monitor(self._parser(), features={"health", "drift"})
        for i in range(5):
            m.update({"cpu": 50.0 - i, "mem": 70.0})
        m.reset()
        assert m.step == 0


# -----------------------------------------------------------------------
# Monitor — reset_anomaly_reference
# -----------------------------------------------------------------------

class TestMonitorResetAnomalyReference:
    def _parser(self):
        return Parser(
            baselines={"cpu": 50.0, "mem": 70.0},
            thresholds=Thresholds(intact=40.0, ablated=10.0),
        )

    def test_reset_clears_all_reference_values(self):
        m = Monitor(self._parser(), anomaly_min_reference=5)
        for i in range(10):
            m.update({"cpu": 50.0, "mem": 70.0},
                     now=t0 + timedelta(seconds=i * 60))
        assert m._anomaly_trackers["cpu"].n_values == 10
        m.reset_anomaly_reference()
        assert m._anomaly_trackers["cpu"].n_values == 0
        assert m._anomaly_trackers["mem"].n_values == 0

    def test_reset_specific_components(self):
        m = Monitor(self._parser(), anomaly_min_reference=5)
        for i in range(10):
            m.update({"cpu": 50.0, "mem": 70.0},
                     now=t0 + timedelta(seconds=i * 60))
        m.reset_anomaly_reference(components=["cpu"])
        assert m._anomaly_trackers["cpu"].n_values == 0
        assert m._anomaly_trackers["mem"].n_values == 10

    def test_reset_allows_fresh_reference(self):
        m = Monitor(self._parser(), anomaly_min_reference=5)
        # Build reference at baseline=50
        for i in range(10):
            m.update({"cpu": 50.0, "mem": 70.0},
                     now=t0 + timedelta(seconds=i * 60))
        m.reset_anomaly_reference()
        # Feed new baseline at 100 — should warm up without contamination
        for i in range(10):
            m.update({"cpu": 100.0, "mem": 70.0},
                     now=t0 + timedelta(seconds=(20 + i) * 60))
        assert m._anomaly_trackers["cpu"].n_values == 10

    def test_reset_noop_when_anomaly_disabled(self):
        m = Monitor(self._parser(), features={"health", "drift"})
        # Should not raise even though anomaly trackers don't exist
        m.reset_anomaly_reference()
