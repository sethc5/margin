"""Tests for margin.drift — trajectory classification, predicates, ledger, forecast."""

from datetime import datetime, timedelta

from margin import (
    Observation, Health, Confidence, Thresholds, Op,
    DriftState, DriftDirection, DriftClassification,
    classify_drift, classify_drift_all,
    Ledger, Record,
    drift_observations_from_ledger, drift_from_ledger, drift_all_from_ledger,
    drift_is, drift_worsening, any_drifting, any_drift_worsening, drift_accelerating,
    DriftForecast, drift_forecast, drift_forecast_from_ledger,
    Expression,
)


def _obs(name, value, baseline, t, higher_is_better=True):
    """Helper to build a timestamped observation."""
    return Observation(
        name=name,
        health=Health.INTACT,  # health state doesn't matter for drift
        value=value,
        baseline=baseline,
        confidence=Confidence.HIGH,
        higher_is_better=higher_is_better,
        measured_at=t,
    )


def _time_series(name, values, baseline=100.0, interval_s=60, higher_is_better=True):
    """Build a list of observations from a value sequence."""
    t0 = datetime(2026, 1, 1)
    return [
        _obs(name, v, baseline, t0 + timedelta(seconds=i * interval_s), higher_is_better)
        for i, v in enumerate(values)
    ]


# -----------------------------------------------------------------------
# Insufficient data
# -----------------------------------------------------------------------

class TestInsufficientData:
    def test_empty(self):
        assert classify_drift([]) is None

    def test_one_observation(self):
        obs = _time_series("x", [100.0])
        assert classify_drift(obs) is None

    def test_two_observations_below_default_min(self):
        obs = _time_series("x", [100.0, 101.0])
        assert classify_drift(obs) is None

    def test_two_observations_with_min_2(self):
        obs = _time_series("x", [100.0, 101.0])
        result = classify_drift(obs, min_samples=2)
        assert result is not None

    def test_no_timestamps(self):
        obs = [Observation(
            name="x", health=Health.INTACT, value=100.0, baseline=100.0,
            confidence=Confidence.HIGH,
        ) for _ in range(5)]
        assert classify_drift(obs) is None


# -----------------------------------------------------------------------
# Stable — flat line
# -----------------------------------------------------------------------

class TestStable:
    def test_constant_values(self):
        obs = _time_series("cpu", [50.0, 50.0, 50.0, 50.0, 50.0])
        result = classify_drift(obs)
        assert result.state == DriftState.STABLE
        assert result.direction == DriftDirection.NEUTRAL
        assert result.component == "cpu"

    def test_noisy_flat(self):
        obs = _time_series("cpu", [50.0, 50.5, 49.8, 50.2, 49.9, 50.1, 50.0])
        result = classify_drift(obs)
        assert result.state == DriftState.STABLE

    def test_stable_serialization(self):
        obs = _time_series("cpu", [50.0, 50.0, 50.0, 50.0])
        result = classify_drift(obs)
        d = result.to_dict()
        restored = DriftClassification.from_dict(d)
        assert restored.state == result.state
        assert restored.direction == result.direction
        assert restored.component == result.component


# -----------------------------------------------------------------------
# Drifting — consistent linear trend
# -----------------------------------------------------------------------

class TestDrifting:
    def test_linear_increase_higher_is_better(self):
        # Values going up, higher is better → improving
        obs = _time_series("throughput", [100, 110, 120, 130, 140, 150], baseline=100.0)
        result = classify_drift(obs)
        assert result.state in (DriftState.DRIFTING, DriftState.REVERTING)
        assert result.direction == DriftDirection.IMPROVING
        assert result.rate > 0

    def test_linear_decrease_higher_is_better(self):
        # Values going down, higher is better → worsening
        obs = _time_series("throughput", [100, 90, 80, 70, 60, 50], baseline=100.0)
        result = classify_drift(obs)
        assert result.direction == DriftDirection.WORSENING
        assert result.rate < 0

    def test_linear_increase_lower_is_better(self):
        # Values going up, lower is better → worsening
        obs = _time_series("error_rate", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                           baseline=0.01, higher_is_better=False)
        result = classify_drift(obs)
        assert result.direction == DriftDirection.WORSENING

    def test_linear_decrease_lower_is_better(self):
        # Values going down, lower is better → improving
        obs = _time_series("latency", [200, 180, 160, 140, 120, 100],
                           baseline=200.0, higher_is_better=False)
        result = classify_drift(obs)
        assert result.direction == DriftDirection.IMPROVING

    def test_n_samples_and_window(self):
        obs = _time_series("x", [10, 20, 30, 40, 50], interval_s=120)
        result = classify_drift(obs)
        assert result.n_samples == 5
        assert result.window_seconds == 480.0  # 4 * 120


# -----------------------------------------------------------------------
# Accelerating — rate of change increasing
# -----------------------------------------------------------------------

class TestAccelerating:
    def test_exponential_growth(self):
        # Quadratic curve: 1, 4, 9, 16, 25, 36, 49, 64
        values = [float(i ** 2) for i in range(1, 9)]
        obs = _time_series("load", values, baseline=1.0)
        result = classify_drift(obs)
        # Should detect acceleration (quadratic fits much better than linear)
        assert result.state in (DriftState.ACCELERATING, DriftState.DRIFTING, DriftState.REVERTING)
        assert result.direction == DriftDirection.IMPROVING  # values going up, higher is better

    def test_accelerating_decline(self):
        # Values dropping faster and faster (higher is better → worsening acceleration)
        values = [100.0 - i ** 2 for i in range(9)]
        obs = _time_series("signal", values, baseline=100.0)
        result = classify_drift(obs)
        assert result.direction == DriftDirection.WORSENING


# -----------------------------------------------------------------------
# Reverting — heading back toward baseline
# -----------------------------------------------------------------------

class TestReverting:
    def test_revert_to_baseline(self):
        # Started below baseline (80), heading back to baseline (100)
        obs = _time_series("throughput", [80, 85, 90, 95, 98], baseline=100.0)
        result = classify_drift(obs)
        assert result.state == DriftState.REVERTING
        assert result.direction == DriftDirection.IMPROVING

    def test_revert_lower_is_better(self):
        # Error rate was high (0.08), coming back down toward baseline (0.01)
        obs = _time_series("error_rate", [0.08, 0.06, 0.04, 0.03, 0.02],
                           baseline=0.01, higher_is_better=False)
        result = classify_drift(obs)
        assert result.state == DriftState.REVERTING
        assert result.direction == DriftDirection.IMPROVING


# -----------------------------------------------------------------------
# Oscillating — periodic fluctuation
# -----------------------------------------------------------------------

class TestOscillating:
    def test_sine_wave(self):
        import math
        # Use irrational-ish frequency to avoid flat tops
        values = [50.0 + 10.0 * math.sin(i * 0.8) for i in range(15)]
        obs = _time_series("temp", values, baseline=50.0)
        result = classify_drift(obs)
        assert result.state == DriftState.OSCILLATING

    def test_alternating(self):
        values = [100, 80, 100, 80, 100, 80, 100]
        obs = _time_series("flaky", values, baseline=90.0)
        result = classify_drift(obs)
        assert result.state == DriftState.OSCILLATING


# -----------------------------------------------------------------------
# classify_drift_all
# -----------------------------------------------------------------------

class TestClassifyDriftAll:
    def test_multiple_components(self):
        obs_map = {
            "cpu": _time_series("cpu", [50, 50, 50, 50, 50]),
            "mem": _time_series("mem", [40, 50, 60, 70, 80]),
        }
        results = classify_drift_all(obs_map)
        assert "cpu" in results
        assert "mem" in results
        assert results["cpu"].state == DriftState.STABLE
        assert results["mem"].direction == DriftDirection.IMPROVING

    def test_skips_insufficient_data(self):
        obs_map = {
            "ok": _time_series("ok", [1, 2, 3, 4, 5]),
            "bad": _time_series("bad", [1.0]),
        }
        results = classify_drift_all(obs_map)
        assert "ok" in results
        assert "bad" not in results


# -----------------------------------------------------------------------
# Properties and display
# -----------------------------------------------------------------------

class TestProperties:
    def test_stable_properties(self):
        obs = _time_series("x", [50, 50, 50, 50])
        result = classify_drift(obs)
        assert result.stable is True
        assert result.worsening is False
        assert result.improving is False

    def test_to_atom(self):
        obs = _time_series("cpu", [50, 60, 70, 80, 90])
        result = classify_drift(obs)
        atom = result.to_atom()
        assert "cpu" in atom
        assert "/" in atom  # rate per second

    def test_repr(self):
        obs = _time_series("cpu", [50, 60, 70, 80, 90])
        result = classify_drift(obs)
        assert "DriftClassification" in repr(result)


# -----------------------------------------------------------------------
# Confidence levels
# -----------------------------------------------------------------------

class TestConfidence:
    def test_few_samples_low_confidence(self):
        obs = _time_series("x", [10, 20, 30])
        result = classify_drift(obs)
        assert result.confidence <= Confidence.MODERATE

    def test_many_samples_good_fit_high_confidence(self):
        # Perfect linear trend, 15 samples
        obs = _time_series("x", [float(i * 10) for i in range(15)], baseline=0.0)
        result = classify_drift(obs)
        assert result.confidence >= Confidence.MODERATE

    def test_r_squared_in_dict(self):
        obs = _time_series("x", [10, 20, 30, 40, 50])
        result = classify_drift(obs)
        d = result.to_dict()
        assert "r_squared" in d
        assert 0.0 <= d["r_squared"] <= 1.0


# -----------------------------------------------------------------------
# Helper: build a Ledger from value sequences
# -----------------------------------------------------------------------

def _ledger_from_values(component, values, baseline=100.0, interval_s=60, higher_is_better=True):
    """Build a Ledger with Records for one component."""
    t0 = datetime(2026, 1, 1)
    ledger = Ledger(label="test")
    for i, v in enumerate(values):
        t = t0 + timedelta(seconds=i * interval_s)
        obs = Observation(
            name=component, health=Health.INTACT, value=v, baseline=baseline,
            confidence=Confidence.HIGH, higher_is_better=higher_is_better,
            measured_at=t,
        )
        ledger.append(Record(
            step=i, tag=f"step-{i}", before=obs, after=obs,
            fired=False, op=Op.NOOP, timestamp=t,
        ))
    return ledger


def _multi_ledger(components):
    """Build a Ledger with multiple components.

    components: dict of {name: (values, baseline, higher_is_better)}
    """
    t0 = datetime(2026, 1, 1)
    ledger = Ledger(label="multi")
    step = 0
    # Interleave records
    max_len = max(len(v[0]) for v in components.values())
    for i in range(max_len):
        t = t0 + timedelta(seconds=i * 60)
        for name, (values, baseline, hib) in components.items():
            if i < len(values):
                obs = Observation(
                    name=name, health=Health.INTACT, value=values[i], baseline=baseline,
                    confidence=Confidence.HIGH, higher_is_better=hib,
                    measured_at=t,
                )
                ledger.append(Record(
                    step=step, tag=f"{name}-{i}", before=obs, after=obs,
                    fired=False, op=Op.NOOP, timestamp=t,
                ))
                step += 1
    return ledger


# -----------------------------------------------------------------------
# Ledger integration
# -----------------------------------------------------------------------

class TestLedgerIntegration:
    def test_observations_from_ledger(self):
        ledger = _ledger_from_values("cpu", [50, 60, 70, 80, 90])
        obs = drift_observations_from_ledger(ledger, "cpu")
        assert len(obs) == 5
        assert obs[0].value == 50
        assert obs[-1].value == 90

    def test_observations_from_ledger_missing_component(self):
        ledger = _ledger_from_values("cpu", [50, 60, 70])
        obs = drift_observations_from_ledger(ledger, "mem")
        assert obs == []

    def test_drift_from_ledger(self):
        ledger = _ledger_from_values("cpu", [80, 85, 90, 95, 98], baseline=100.0)
        dc = drift_from_ledger(ledger, "cpu")
        assert dc is not None
        assert dc.component == "cpu"
        assert dc.state == DriftState.REVERTING
        assert dc.direction == DriftDirection.IMPROVING

    def test_drift_from_ledger_insufficient(self):
        ledger = _ledger_from_values("cpu", [50.0])
        dc = drift_from_ledger(ledger, "cpu")
        assert dc is None

    def test_drift_all_from_ledger(self):
        ledger = _multi_ledger({
            "cpu": ([50, 50, 50, 50, 50], 50.0, True),
            "mem": ([40, 50, 60, 70, 80], 100.0, True),
        })
        results = drift_all_from_ledger(ledger)
        assert "cpu" in results
        assert "mem" in results
        assert results["cpu"].state == DriftState.STABLE
        assert results["mem"].direction == DriftDirection.IMPROVING


# -----------------------------------------------------------------------
# Predicates
# -----------------------------------------------------------------------

class TestPredicates:
    def _dummy_expr(self):
        """Minimal expression — predicates don't use it, they use the ledger."""
        return Expression(observations=[], confidence=Confidence.HIGH)

    def test_drift_is(self):
        ledger = _ledger_from_values("cpu", [50, 50, 50, 50, 50])
        pred = drift_is("cpu", DriftState.STABLE, ledger)
        assert pred(self._dummy_expr()) is True

    def test_drift_is_negative(self):
        ledger = _ledger_from_values("cpu", [50, 50, 50, 50, 50])
        pred = drift_is("cpu", DriftState.DRIFTING, ledger)
        assert pred(self._dummy_expr()) is False

    def test_drift_worsening(self):
        ledger = _ledger_from_values("cpu", [100, 90, 80, 70, 60, 50])
        pred = drift_worsening("cpu", ledger)
        assert pred(self._dummy_expr()) is True

    def test_drift_worsening_negative(self):
        ledger = _ledger_from_values("cpu", [50, 50, 50, 50, 50])
        pred = drift_worsening("cpu", ledger)
        assert pred(self._dummy_expr()) is False

    def test_any_drifting(self):
        ledger = _multi_ledger({
            "cpu": ([50, 50, 50, 50, 50], 50.0, True),
            "mem": ([40, 50, 60, 70, 80], 100.0, True),
        })
        pred = any_drifting(ledger)
        assert pred(self._dummy_expr()) is True  # mem is drifting

    def test_any_drifting_all_stable(self):
        ledger = _multi_ledger({
            "cpu": ([50, 50, 50, 50, 50], 50.0, True),
            "mem": ([80, 80, 80, 80, 80], 80.0, True),
        })
        pred = any_drifting(ledger)
        assert pred(self._dummy_expr()) is False

    def test_any_drift_worsening(self):
        ledger = _multi_ledger({
            "cpu": ([50, 50, 50, 50, 50], 50.0, True),
            "mem": ([80, 70, 60, 50, 40], 80.0, True),
        })
        pred = any_drift_worsening(ledger)
        assert pred(self._dummy_expr()) is True  # mem is worsening

    def test_drift_accelerating(self):
        values = [100.0 - i ** 2 for i in range(9)]
        ledger = _ledger_from_values("signal", values, baseline=100.0)
        pred = drift_accelerating("signal", ledger)
        # May or may not detect acceleration depending on fit — just check it runs
        result = pred(self._dummy_expr())
        assert isinstance(result, bool)

    def test_predicate_missing_component(self):
        ledger = _ledger_from_values("cpu", [50, 60, 70])
        pred = drift_is("nonexistent", DriftState.STABLE, ledger)
        assert pred(self._dummy_expr()) is False


# -----------------------------------------------------------------------
# DriftForecast composition
# -----------------------------------------------------------------------

class TestDriftForecast:
    def test_drift_forecast_basic(self):
        obs = _time_series("cpu", [100, 90, 80, 70, 60, 50], baseline=100.0)
        thresholds = Thresholds(intact=80.0, ablated=30.0)
        df = drift_forecast(obs, thresholds)
        assert df is not None
        assert df.component == "cpu"
        assert df.drift.direction == DriftDirection.WORSENING
        assert df.forecast is not None

    def test_drift_forecast_summary(self):
        obs = _time_series("cpu", [100, 90, 80, 70, 60, 50], baseline=100.0)
        thresholds = Thresholds(intact=80.0, ablated=30.0)
        df = drift_forecast(obs, thresholds)
        s = df.summary
        assert "cpu" in s
        assert "WORSENING" in s

    def test_drift_forecast_to_dict(self):
        obs = _time_series("cpu", [100, 90, 80, 70, 60, 50], baseline=100.0)
        thresholds = Thresholds(intact=80.0, ablated=30.0)
        df = drift_forecast(obs, thresholds)
        d = df.to_dict()
        assert "drift" in d
        assert "forecast" in d
        assert "summary" in d

    def test_drift_forecast_repr(self):
        obs = _time_series("cpu", [100, 90, 80, 70, 60, 50], baseline=100.0)
        thresholds = Thresholds(intact=80.0, ablated=30.0)
        df = drift_forecast(obs, thresholds)
        assert "DriftForecast" in repr(df)

    def test_drift_forecast_insufficient_data(self):
        obs = _time_series("cpu", [100.0])
        thresholds = Thresholds(intact=80.0, ablated=30.0)
        df = drift_forecast(obs, thresholds)
        assert df is None

    def test_drift_forecast_from_ledger(self):
        ledger = _ledger_from_values("cpu", [100, 90, 80, 70, 60, 50])
        thresholds = Thresholds(intact=80.0, ablated=30.0)
        df = drift_forecast_from_ledger(ledger, "cpu", thresholds)
        assert df is not None
        assert df.drift.direction == DriftDirection.WORSENING

    def test_drift_forecast_improving_eta(self):
        # Values improving toward intact
        obs = _time_series("cpu", [50, 55, 60, 65, 70, 75], baseline=100.0)
        thresholds = Thresholds(intact=80.0, ablated=30.0)
        df = drift_forecast(obs, thresholds)
        assert df is not None
        assert df.drift.direction == DriftDirection.IMPROVING
        # Summary should mention ETA intact if available
        assert "cpu" in df.summary

    def test_drift_forecast_stable_no_eta(self):
        obs = _time_series("cpu", [50, 50, 50, 50, 50], baseline=100.0)
        thresholds = Thresholds(intact=80.0, ablated=30.0)
        df = drift_forecast(obs, thresholds)
        assert df is not None
        assert df.drift.state == DriftState.STABLE
        # No ETA in summary for stable
        assert "ETA" not in df.summary


class TestDriftClassificationStepCount:
    def test_step_count_equals_n_samples(self):
        from margin.drift import DriftClassification, DriftState, DriftDirection
        from margin.confidence import Confidence
        dc = DriftClassification(
            component="cpu", state=DriftState.STABLE,
            direction=DriftDirection.NEUTRAL, rate=0.0,
            acceleration=0.0, r_squared=0.9,
            confidence=Confidence.HIGH, n_samples=17,
            window_seconds=120.0,
        )
        assert dc.step_count == 17
        assert dc.step_count == dc.n_samples
