"""Tests for margin.anomaly — statistical outlier detection."""

from datetime import datetime, timedelta

from margin import (
    Observation, Health, Confidence, Op,
    AnomalyState, AnomalyClassification, ANOMALY_SEVERITY,
    DistributionShift, Jump,
    classify_anomaly, classify_anomaly_obs,
    check_distribution, detect_jumps,
    anomaly_from_ledger, anomaly_all_from_ledger,
    distribution_shift_from_ledger,
    anomaly_is, any_anomalous, any_novel, is_novel,
    Ledger, Record, Expression,
)


def _obs(name, value, baseline=100.0, t=None):
    return Observation(
        name=name, health=Health.INTACT, value=value, baseline=baseline,
        confidence=Confidence.HIGH, measured_at=t,
    )


def _ledger_from_values(component, values, baseline=100.0):
    t0 = datetime(2026, 1, 1)
    ledger = Ledger(label="test")
    for i, v in enumerate(values):
        t = t0 + timedelta(seconds=i * 60)
        obs = _obs(component, v, baseline, t)
        ledger.append(Record(step=i, tag=f"s{i}", before=obs, after=obs, timestamp=t))
    return ledger


def _multi_ledger(components):
    t0 = datetime(2026, 1, 1)
    ledger = Ledger(label="multi")
    step = 0
    max_len = max(len(vals) for vals, _ in components.values())
    for i in range(max_len):
        t = t0 + timedelta(seconds=i * 60)
        for name, (values, baseline) in components.items():
            if i < len(values):
                obs = _obs(name, values[i], baseline, t)
                ledger.append(Record(step=step, tag=f"{name}-{i}", before=obs, after=obs, timestamp=t))
                step += 1
    return ledger


# -----------------------------------------------------------------------
# Point anomaly classification
# -----------------------------------------------------------------------

class TestClassifyAnomaly:
    def test_expected_value(self):
        ref = [100, 101, 99, 100, 102, 98, 101, 99, 100, 100]
        ac = classify_anomaly(100.5, ref, component="cpu")
        assert ac.state == AnomalyState.EXPECTED
        assert ac.expected is True
        assert ac.anomalous is False

    def test_unusual_value(self):
        ref = [100.0] * 30 + [101.0] * 30  # tight distribution, mean ~100.5, std ~0.5
        ac = classify_anomaly(103.0, ref, component="cpu")
        assert ac.state in (AnomalyState.UNUSUAL, AnomalyState.ANOMALOUS, AnomalyState.NOVEL)

    def test_anomalous_value(self):
        # Very tight reference, value far away
        ref = [50.0, 50.1, 49.9, 50.0, 50.2, 49.8, 50.0, 50.1, 49.9, 50.0]
        ac = classify_anomaly(55.0, ref, component="temp")
        assert ac.state in (AnomalyState.ANOMALOUS, AnomalyState.NOVEL)
        assert ac.anomalous is True

    def test_novel_value(self):
        ref = [10.0, 11.0, 12.0, 10.5, 11.5, 10.0, 12.0, 11.0, 10.5, 11.5]
        ac = classify_anomaly(50.0, ref, component="x")
        assert ac.state == AnomalyState.NOVEL
        assert ac.is_novel is True

    def test_novel_below_range(self):
        ref = [100.0, 101.0, 102.0, 100.5, 101.5, 100.0]
        ac = classify_anomaly(50.0, ref, component="x")
        assert ac.state == AnomalyState.NOVEL
        assert ac.is_novel is True

    def test_insufficient_reference(self):
        ac = classify_anomaly(100.0, [50.0, 60.0], component="x")
        assert ac is None

    def test_zero_std_same_value(self):
        ref = [42.0, 42.0, 42.0, 42.0, 42.0]
        ac = classify_anomaly(42.0, ref, component="x")
        assert ac.state == AnomalyState.EXPECTED

    def test_zero_std_different_value(self):
        ref = [42.0, 42.0, 42.0, 42.0, 42.0]
        ac = classify_anomaly(43.0, ref, component="x")
        assert ac.state == AnomalyState.NOVEL

    def test_z_score_sign(self):
        ref = [100.0, 100.0, 100.0, 101.0, 99.0, 100.0, 100.0, 100.0, 101.0, 99.0]
        ac_above = classify_anomaly(110.0, ref, component="x")
        ac_below = classify_anomaly(90.0, ref, component="x")
        assert ac_above.z_score > 0
        assert ac_below.z_score < 0

    def test_confidence_from_sample_size(self):
        small = classify_anomaly(100.0, [99, 100, 101], component="x")
        large = classify_anomaly(100.0, list(range(50, 150)), component="x")
        assert small.confidence <= large.confidence

    def test_serialization_roundtrip(self):
        ref = [100.0, 101.0, 99.0, 100.5, 99.5, 100.0, 101.0, 99.0, 100.5, 99.5]
        ac = classify_anomaly(100.0, ref, component="cpu")
        d = ac.to_dict()
        restored = AnomalyClassification.from_dict(d)
        assert restored.state == ac.state
        assert restored.z_score == ac.z_score
        assert restored.component == ac.component

    def test_to_atom(self):
        ref = list(range(90, 110))
        ac = classify_anomaly(105.0, ref, component="cpu")
        atom = ac.to_atom()
        assert "cpu" in atom
        assert "σ" in atom

    def test_inf_z_score_atom(self):
        """Zero-std reference produces inf z-score — atom should be clean."""
        ac = classify_anomaly(43.0, [42.0] * 10, component="x")
        assert "inf" not in ac.to_atom()
        assert ac.to_atom() == "x:NOVEL"

    def test_custom_thresholds(self):
        ref = [100.0, 100.0, 100.0, 101.0, 99.0, 100.0, 100.0, 100.0, 101.0, 99.0]
        # With very tight thresholds
        ac = classify_anomaly(101.5, ref, component="x",
                              unusual_threshold=0.5, anomalous_threshold=1.0)
        assert ac.state != AnomalyState.EXPECTED


class TestClassifyAnomalyObs:
    def test_from_observations(self):
        t0 = datetime(2026, 1, 1)
        history = [_obs("cpu", v, t=t0 + timedelta(seconds=i))
                   for i, v in enumerate([100, 101, 99, 100, 102, 98, 101, 99, 100, 100])]
        current = _obs("cpu", 100.5, t=t0 + timedelta(seconds=100))
        ac = classify_anomaly_obs(current, history)
        assert ac is not None
        assert ac.component == "cpu"


# -----------------------------------------------------------------------
# Distribution shift
# -----------------------------------------------------------------------

class TestDistributionShift:
    def test_no_shift(self):
        ref = [100 + i * 0.1 for i in range(20)]
        recent = [100 + i * 0.1 for i in range(5, 25)]  # similar range
        ds = check_distribution(recent, ref, component="cpu")
        assert ds is not None
        assert ds.state == AnomalyState.EXPECTED
        assert ds.shifted is False

    def test_mean_shift(self):
        ref = [100.0 + i * 0.1 for i in range(20)]
        recent = [200.0 + i * 0.1 for i in range(20)]  # mean doubled
        ds = check_distribution(recent, ref, component="cpu")
        assert ds is not None
        assert ds.state != AnomalyState.EXPECTED
        assert ds.mean_shift > 0.2

    def test_std_explosion(self):
        ref = [100.0, 100.1, 99.9, 100.0, 100.1, 99.9, 100.0, 100.1, 99.9, 100.0]
        recent = [80.0, 120.0, 70.0, 130.0, 60.0, 140.0, 50.0, 150.0, 40.0, 160.0]
        ds = check_distribution(recent, ref, component="x")
        assert ds is not None
        assert ds.std_ratio > 2.0
        assert ds.state != AnomalyState.EXPECTED

    def test_insufficient_data(self):
        ds = check_distribution([1, 2], [3, 4], component="x")
        assert ds is None

    def test_serialization_roundtrip(self):
        ref = list(range(20))
        recent = list(range(10, 30))
        ds = check_distribution(recent, ref, component="x")
        d = ds.to_dict()
        restored = DistributionShift.from_dict(d)
        assert restored.state == ds.state
        assert round(restored.mean_shift, 4) == round(ds.mean_shift, 4)

    def test_shape_change(self):
        import math
        # Reference: roughly normal
        ref = [50 + 5 * math.sin(i * 0.3) for i in range(30)]
        # Recent: heavy-tailed (some extreme values)
        recent = [50.0] * 20 + [10.0, 90.0, 5.0, 95.0, 0.0, 100.0, 50.0, 50.0, 50.0, 50.0]
        ds = check_distribution(recent, ref, component="x")
        assert ds is not None
        # Should detect the shape change (kurtosis or skew)


# -----------------------------------------------------------------------
# Jump detection
# -----------------------------------------------------------------------

class TestJumpDetection:
    def test_no_jumps(self):
        obs = [_obs("x", 100 + i * 0.5) for i in range(10)]
        jumps = detect_jumps(obs)
        assert jumps == []

    def test_single_jump(self):
        values = [100, 101, 100, 99, 100, 200, 201, 200, 199, 200]
        obs = [_obs("x", v) for v in values]
        jumps = detect_jumps(obs, jump_threshold=2.5)
        assert len(jumps) >= 1
        # The jump should be around index 5
        assert any(j.at_index == 5 for j in jumps)

    def test_spike_in_constant_default_threshold(self):
        """A spike in a constant series should be detected at default threshold."""
        values = [50, 50, 50, 50, 50, 150, 50, 50, 50, 50]
        obs = [_obs("x", v) for v in values]
        jumps = detect_jumps(obs)  # default threshold=3.0
        assert len(jumps) >= 1
        assert any(j.at_index == 5 for j in jumps)

    def test_jump_magnitude(self):
        values = [50, 50, 50, 50, 50, 150, 50, 50, 50, 50]
        obs = [_obs("x", v) for v in values]
        jumps = detect_jumps(obs, jump_threshold=2.0)
        assert len(jumps) >= 1
        # Jump up should have positive magnitude, jump down negative
        up_jump = [j for j in jumps if j.value_after > j.value_before]
        assert len(up_jump) >= 1
        assert up_jump[0].magnitude_sigma > 0

    def test_insufficient_data(self):
        obs = [_obs("x", 100), _obs("x", 200)]
        jumps = detect_jumps(obs)
        assert jumps == []

    def test_constant_series_no_jumps(self):
        obs = [_obs("x", 42.0) for _ in range(10)]
        jumps = detect_jumps(obs)
        assert jumps == []

    def test_serialization_roundtrip(self):
        values = [50, 50, 50, 50, 50, 150, 50, 50, 50, 50]
        obs = [_obs("x", v) for v in values]
        jumps = detect_jumps(obs, jump_threshold=2.0)
        if jumps:
            d = jumps[0].to_dict()
            restored = Jump.from_dict(d)
            assert restored.magnitude_sigma == jumps[0].magnitude_sigma
            assert restored.at_index == jumps[0].at_index


# -----------------------------------------------------------------------
# Ledger integration
# -----------------------------------------------------------------------

class TestLedgerIntegration:
    def test_anomaly_from_ledger_expected(self):
        # Normal values, last value in range
        values = [100, 101, 99, 100, 102, 98, 101, 99, 100, 100]
        ledger = _ledger_from_values("cpu", values)
        ac = anomaly_from_ledger(ledger, "cpu")
        assert ac is not None
        assert ac.state == AnomalyState.EXPECTED

    def test_anomaly_from_ledger_outlier(self):
        values = [50.0] * 20 + [200.0]  # spike at end
        ledger = _ledger_from_values("cpu", values)
        ac = anomaly_from_ledger(ledger, "cpu")
        assert ac is not None
        assert ac.state in (AnomalyState.ANOMALOUS, AnomalyState.NOVEL)

    def test_anomaly_from_ledger_insufficient(self):
        ledger = _ledger_from_values("cpu", [100.0, 101.0])
        ac = anomaly_from_ledger(ledger, "cpu")
        assert ac is None

    def test_anomaly_all_from_ledger(self):
        ledger = _multi_ledger({
            "cpu": ([50] * 10 + [50], 50.0),
            "mem": ([80] * 10 + [200], 80.0),
        })
        results = anomaly_all_from_ledger(ledger)
        assert "cpu" in results or "mem" in results
        if "mem" in results:
            assert results["mem"].state != AnomalyState.EXPECTED

    def test_distribution_shift_from_ledger(self):
        # First 20 values normal, last 10 shifted
        values = [100.0] * 20 + [200.0] * 10
        ledger = _ledger_from_values("cpu", values)
        ds = distribution_shift_from_ledger(ledger, "cpu")
        assert ds is not None
        assert ds.shifted is True

    def test_distribution_shift_insufficient(self):
        ledger = _ledger_from_values("cpu", [100.0] * 5)
        ds = distribution_shift_from_ledger(ledger, "cpu")
        assert ds is None


# -----------------------------------------------------------------------
# Predicates
# -----------------------------------------------------------------------

class TestPredicates:
    def _expr(self):
        return Expression(observations=[], confidence=Confidence.HIGH)

    def test_anomaly_is(self):
        values = [50.0] * 20 + [50.0]
        ledger = _ledger_from_values("cpu", values)
        pred = anomaly_is("cpu", AnomalyState.EXPECTED, ledger)
        assert pred(self._expr()) is True

    def test_anomaly_is_negative(self):
        values = [50.0] * 20 + [50.0]
        ledger = _ledger_from_values("cpu", values)
        pred = anomaly_is("cpu", AnomalyState.NOVEL, ledger)
        assert pred(self._expr()) is False

    def test_any_anomalous(self):
        ledger = _multi_ledger({
            "cpu": ([50] * 20 + [50], 50.0),
            "mem": ([80] * 20 + [300], 80.0),
        })
        pred = any_anomalous(ledger)
        assert pred(self._expr()) is True

    def test_any_anomalous_all_normal(self):
        ledger = _multi_ledger({
            "cpu": ([50] * 20 + [50], 50.0),
            "mem": ([80] * 20 + [80], 80.0),
        })
        pred = any_anomalous(ledger)
        assert pred(self._expr()) is False

    def test_any_novel(self):
        ledger = _multi_ledger({
            "cpu": ([50] * 20 + [500], 50.0),  # way outside range
        })
        pred = any_novel(ledger)
        assert pred(self._expr()) is True

    def test_is_novel(self):
        values = [50.0] * 20 + [500.0]
        ledger = _ledger_from_values("cpu", values)
        pred = is_novel("cpu", ledger)
        assert pred(self._expr()) is True

    def test_predicate_missing_component(self):
        ledger = _ledger_from_values("cpu", [50] * 10)
        pred = anomaly_is("nonexistent", AnomalyState.EXPECTED, ledger)
        assert pred(self._expr()) is False


# -----------------------------------------------------------------------
# Severity ordering
# -----------------------------------------------------------------------

class TestSeverity:
    def test_ordering(self):
        assert ANOMALY_SEVERITY[AnomalyState.EXPECTED] < ANOMALY_SEVERITY[AnomalyState.UNUSUAL]
        assert ANOMALY_SEVERITY[AnomalyState.UNUSUAL] < ANOMALY_SEVERITY[AnomalyState.ANOMALOUS]
        assert ANOMALY_SEVERITY[AnomalyState.ANOMALOUS] < ANOMALY_SEVERITY[AnomalyState.NOVEL]
